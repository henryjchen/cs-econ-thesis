"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, scikit-learn, matplotlib, seaborn, pandas
Data Dependencies: Clay embeddings (2010/2020), INEGI census data (2010/2020)
Description: Analysis runner for population density prediction using Ridge and XGBoost on aggregated CLAY features.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else iter(())

# Handle tqdm in Slurm logs
_IS_TTY = sys.stdout.isatty()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg

_BEST_PARAMS_PATH = Path(__file__).with_name("best_params.json")


def _cv_key(*, leave_one_state_out: bool, folds: int, seed: int) -> str:
    """Generates a unique key for the CV configuration.

    Args:
        leave_one_state_out: Whether LOSO CV is used.
        folds: Number of folds.
        seed: Random seed.

    Returns:
        String key for the configuration.
    """
    if leave_one_state_out:
        return "leave_one_state_out"
    return f"kfold{int(folds)}_seed{int(seed)}"


def _load_best_params(path: Path) -> dict:
    """Loads tuned hyperparameters from a JSON file.

    Args:
        path: Path to the best_params.json file.

    Returns:
        Dictionary of parameters or empty dict if not found.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not read best params at {path} ({e}); ignoring.")
        return {}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def _emb_columns(df: pd.DataFrame) -> list[str]:
    """Identifies and sorts embedding columns.

    Args:
        df: Input DataFrame.

    Returns:
        Sorted list of embedding column names.
    """
    cols = [c for c in df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def _state_code(ageb_id: pd.Series) -> pd.Series:
    """Extracts 2-digit INEGI state code from CVEGEO.

    Args:
        ageb_id: Series of CVEGEO identifiers.

    Returns:
        Series of 2-digit state codes.
    """
    s = ageb_id.astype(str)
    return s.str.slice(0, 2).str.zfill(2)


def _load(
    parquet: Path,
    target: str,
    *,
    min_target: float | None = None,
    max_target: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Loads features and targets from parquet, applying filters.

    Args:
        parquet: Path to the source parquet file.
        target: Target column name.
        min_target: Minimum threshold for target values.
        max_target: Maximum threshold for target values.

    Returns:
        Tuple of (full DataFrame, X matrix, y vector, state labels).
    """
    df = pd.read_parquet(parquet)
    if target not in df.columns:
        raise KeyError(
            f"target '{target}' not in parquet columns; "
            f"available non-embedding cols: {[c for c in df.columns if not c.startswith('emb_')]}"
        )
    emb_cols = _emb_columns(df)
    if len(emb_cols) != fcfg.EMBEDDING_DIM:
        raise RuntimeError(
            f"expected {fcfg.EMBEDDING_DIM} embedding columns, found {len(emb_cols)}"
        )

    mask = (
        df[emb_cols].notna().all(axis=1)
        & np.isfinite(df[target])
    )
    dropped = int((~mask).sum())
    if dropped:
        print(f"Dropping {dropped:,} rows with non-finite {target} or embeddings")
    df = df.loc[mask].reset_index(drop=True)

    if min_target is not None or max_target is not None:
        tvals = df[target].to_numpy()
        keep = np.ones(len(df), dtype=bool)
        if min_target is not None:
            keep &= tvals >= float(min_target)
        if max_target is not None:
            keep &= tvals <= float(max_target)
        n_cut = int((~keep).sum())
        if n_cut:
            rng_msg = []
            if min_target is not None:
                rng_msg.append(f"{target} >= {min_target}")
            if max_target is not None:
                rng_msg.append(f"{target} <= {max_target}")
            print(
                f"Dropping {n_cut:,} rows outside target range "
                f"({', '.join(rng_msg)})"
            )
        df = df.loc[keep].reset_index(drop=True)

    X = df[emb_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[target].to_numpy(dtype=np.float32, copy=False)
    state = _state_code(df["ageb_id"])
    return df, X, y, state


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _r2(y: np.ndarray, p: np.ndarray) -> float:
    """Calculates R-squared score."""
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    """Calculates Mean Absolute Error."""
    return float(np.mean(np.abs(y - p)))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calculates standard regression metrics.

    Args:
        y: True values.
        p: Predicted values.

    Returns:
        Dictionary with r2, mae, rmse, and n.
    """
    return {"r2": _r2(y, p), "mae": _mae(y, p), "rmse": _rmse(y, p), "n": int(len(y))}


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------
def _build_features(
    X_emb: np.ndarray,
    state: pd.Series,
    *,
    add_state_dummies: bool,
    state_only: bool,
) -> tuple[np.ndarray, int]:
    """Constructs the feature matrix with optional state dummies.

    Args:
        X_emb: Raw embedding matrix.
        state: Series of state codes.
        add_state_dummies: Whether to append one-hot state codes.
        state_only: Whether to use only state codes as features.

    Returns:
        Tuple of (final X matrix, embedding dimension count).
    """
    dummies = None
    if add_state_dummies or state_only:
        dummies = pd.get_dummies(state, prefix="st", dtype=np.float32).to_numpy()
        print(f"  state dummies: {dummies.shape[1]} columns")

    if state_only:
        return dummies, 0
    if add_state_dummies:
        X = np.concatenate([X_emb, dummies], axis=1).astype(np.float32, copy=False)
        return X, X_emb.shape[1]
    return X_emb, X_emb.shape[1]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _fit_ridge_oof(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, alpha: float, emb_dim: int
) -> np.ndarray:
    """Fits Ridge regression with out-of-fold predictions.

    Args:
        X: Feature matrix.
        y: Target vector.
        folds: Array of fold indices.
        alpha: Regularization strength.
        emb_dim: Number of embedding columns to scale.

    Returns:
        Array of OOF predictions.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    fold_ids = np.unique(folds)
    preds = np.full_like(y, np.nan, dtype=np.float32)
    bar = tqdm(
        total=len(fold_ids),
        desc="Ridge folds",
        disable=not _IS_TTY,
        leave=False,
    )
    for i, f in enumerate(fold_ids):
        t0 = time.time()
        tr = folds != f
        te = ~tr
        Xtr = X[tr].astype(np.float32, copy=True)
        Xte = X[te].astype(np.float32, copy=True)
        if emb_dim > 0:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr[:, :emb_dim])
            Xtr[:, :emb_dim] = scaler.transform(Xtr[:, :emb_dim])
            Xte[:, :emb_dim] = scaler.transform(Xte[:, :emb_dim])
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(Xtr, y[tr])
        preds[te] = model.predict(Xte).astype(np.float32)
        bar.update(1)
        if not _IS_TTY:
            print(
                f"  Ridge fold {i + 1}/{len(fold_ids)} done in {time.time() - t0:.1f}s",
                flush=True,
            )
    bar.close()
    return preds


def _fit_xgb_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    n_estimators: int,
    early_stop: int,
    *,
    xgb_params: dict | None = None,
) -> np.ndarray:
    """Fits XGBoost regressor with out-of-fold predictions.

    Args:
        X: Feature matrix.
        y: Target vector.
        folds: Array of fold indices.
        n_estimators: Maximum number of trees.
        early_stop: Early stopping rounds.
        xgb_params: Dictionary of XGBoost hyperparameters.

    Returns:
        Array of OOF predictions.
    """
    import xgboost as xgb
    from xgboost import XGBRegressor

    class _TQDMProgress(xgb.callback.TrainingCallback):
        """XGBoost callback to update tqdm progress bar."""

        def __init__(self, total: int, desc: str, disable: bool):
            self.bar = tqdm(total=total, desc=desc, disable=disable, leave=False)

        def after_iteration(self, model, epoch, evals_log):  # type: ignore[override]
            self.bar.update(1)
            try:
                name, metrics = next(iter(evals_log.items()))
                metric_name, values = next(iter(metrics.items()))
                self.bar.set_postfix_str(f"{metric_name}={values[-1]:.4f}")
            except StopIteration:
                pass
            return False

        def after_training(self, model):  # type: ignore[override]
            self.bar.close()
            return model

    fold_ids = np.unique(folds)
    preds = np.full_like(y, np.nan, dtype=np.float32)
    outer = tqdm(
        total=len(fold_ids),
        desc="XGB folds",
        disable=not _IS_TTY,
        position=0,
    )
    for i, f in enumerate(fold_ids):
        t0 = time.time()
        tr_all = folds != f
        te = ~tr_all
        idx_tr_all = np.where(tr_all)[0]
        rng = np.random.default_rng(0)
        rng.shuffle(idx_tr_all)
        n_val = max(1, int(0.1 * len(idx_tr_all)))
        idx_val = idx_tr_all[:n_val]
        idx_tr = idx_tr_all[n_val:]

        inner_cb = _TQDMProgress(
            total=n_estimators,
            desc=f"  fold {i + 1}/{len(fold_ids)} boost",
            disable=not _IS_TTY,
        )
        verbose_every = 0 if _IS_TTY else 100

        p = dict(xgb_params or {})
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=float(p.pop("learning_rate", 0.05)),
            max_depth=int(p.pop("max_depth", 6)),
            subsample=float(p.pop("subsample", 0.9)),
            colsample_bytree=float(p.pop("colsample_bytree", 0.9)),
            reg_lambda=float(p.pop("reg_lambda", 1.0)),
            min_child_weight=float(p.pop("min_child_weight", 1.0)),
            gamma=float(p.pop("gamma", 0.0)),
            objective="reg:squarederror",
            tree_method=str(p.pop("tree_method", "hist")),
            random_state=int(p.pop("random_state", 0)),
            early_stopping_rounds=early_stop,
            n_jobs=-1,
            callbacks=[inner_cb] if _IS_TTY else None,
            **p,
        )
        if _IS_TTY:
            model.fit(
                X[idx_tr], y[idx_tr],
                eval_set=[(X[idx_val], y[idx_val])],
                verbose=False,
            )
        else:
            print(f"  XGB fold {i + 1}/{len(fold_ids)}: fitting...", flush=True)
            model.fit(
                X[idx_tr], y[idx_tr],
                eval_set=[(X[idx_val], y[idx_val])],
                verbose=verbose_every,
            )
        preds[te] = model.predict(X[te]).astype(np.float32)
        outer.update(1)
        if not _IS_TTY:
            best_it = getattr(model, "best_iteration", None)
            print(
                f"  XGB fold {i + 1}/{len(fold_ids)} done in {time.time() - t0:.1f}s "
                f"(best_iteration={best_it})",
                flush=True,
            )
    outer.close()
    return preds


# -----------------------------------------------------------------------------
# CV schemes
# -----------------------------------------------------------------------------
def _kfold_assign(n: int, k: int, seed: int) -> np.ndarray:
    """Assigns observation indices to folds."""
    from sklearn.model_selection import KFold
    folds = np.empty(n, dtype=np.int32)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(kf.split(np.arange(n))):
        folds[te] = i
    return folds


def _leave_one_group_out(state: pd.Series) -> np.ndarray:
    """Assigns folds based on unique state codes."""
    uniq = pd.Index(sorted(state.unique()))
    lut = {s: i for i, s in enumerate(uniq)}
    return state.map(lut).to_numpy(dtype=np.int32)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def _per_state(y: np.ndarray, p: np.ndarray, state: pd.Series) -> pd.DataFrame:
    """Calculates metrics for each state individually."""
    df = pd.DataFrame({"state": state.values, "y": y, "p": p})
    rows = []
    for s, g in df.groupby("state"):
        if len(g) < 2 or g["y"].std() == 0:
            rows.append({"state": s, "n": len(g), "r2": np.nan, "mae": _mae(g["y"].values, g["p"].values), "rmse": _rmse(g["y"].values, g["p"].values)})
        else:
            rows.append({"state": s, **_metrics(g["y"].values, g["p"].values)})
    out = pd.DataFrame(rows).sort_values("state").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    """CLI entry point for model analysis."""
    parser = argparse.ArgumentParser(description="Analysis runner for CLAY features.")
    fcfg.add_year_arg(parser)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Source features parquet (default: the selected --year's "
             "ageb_features.parquet under its per-year out dir).",
    )
    parser.add_argument("--target", type=str, default="log_popden")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=None,
        help="Ridge regularization strength. If omitted, uses tuned per-year best "
             "params when available; otherwise defaults to 1.0.",
    )
    parser.add_argument("--xgb-rounds", type=int, default=2000)
    parser.add_argument("--xgb-early-stop", type=int, default=50)
    parser.add_argument("--xgb-learning-rate", type=float, default=None)
    parser.add_argument("--xgb-max-depth", type=int, default=None)
    parser.add_argument("--xgb-subsample", type=float, default=None)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=None)
    parser.add_argument("--xgb-reg-lambda", type=float, default=None)
    parser.add_argument("--xgb-min-child-weight", type=float, default=None)
    parser.add_argument("--xgb-gamma", type=float, default=None)
    parser.add_argument(
        "--leave-one-state-out",
        action="store_true",
        help="Use state-level leave-one-out CV instead of random K-fold.",
    )
    parser.add_argument(
        "--skip-xgb",
        action="store_true",
        help="Skip XGBoost (useful if it isn't installed).",
    )
    parser.add_argument(
        "--state-dummies",
        action="store_true",
        help="Append one-hot state indicators (from CVEGEO first 2 chars) to the "
             "feature matrix. With --leave-one-state-out the held-out state's dummy "
             "column will be all-zero in training, so it contributes nothing in that "
             "mode and you should typically leave this flag off there.",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Use ONLY state-indicator features (no CLAY embeddings) as a baseline.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination directory for metrics/OOF artifacts (default: "
             "``<year_out>/analysis`` where <year_out> is selected by "
             "--year).",
    )
    parser.add_argument(
        "--min-target",
        type=float,
        default=None,
        help="Drop rows whose target is strictly below this value (after "
             "dropping NaN/inf targets). The shapefile defines log_popden = "
             "log1p(pop_dens), so log_popden == 0 means POBTOT == 0 "
             "(unpopulated). Use --min-target 1e-9 to drop empty rows; use "
             "--min-target 1.0 to also drop the sparse rural tail "
             "(pop_dens < ~1.7 people/km^2).",
    )
    parser.add_argument(
        "--drop-zero-target",
        action="store_true",
        help="Shorthand for '--min-target 1e-9'. Filters out rows where the "
             "target is exactly 0 (for log1p-encoded targets, this is the "
             "'POBTOT == 0' set).",
    )
    parser.add_argument(
        "--max-target",
        type=float,
        default=None,
        help="Drop rows whose target is above this value (rarely useful; "
             "handy for ignoring outlier dense AGEBs).",
    )
    best_group = parser.add_mutually_exclusive_group()
    best_group.add_argument(
        "--use-best-params",
        action="store_true",
        default=True,
        help=f"Load tuned params from {_BEST_PARAMS_PATH.name} when available "
             "(default: enabled).",
    )
    best_group.add_argument(
        "--no-best-params",
        action="store_true",
        help="Disable loading tuned params, even if best_params.json exists.",
    )
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    if args.parquet is None:
        args.parquet = ds.polygon_features_path
    if args.out_dir is None:
        args.out_dir = ds.out_dir / "analysis"

    if args.state_only and args.state_dummies:
        print("--state-only implies --state-dummies; ignoring redundant flag.")
    if args.state_only and args.leave_one_state_out:
        parser.error("--state-only is meaningless with --leave-one-state-out "
                     "(the held-out state's column is always zero in training).")

    if args.drop_zero_target:
        if args.min_target is not None and args.min_target > 1e-9:
            print(f"--drop-zero-target ignored (--min-target={args.min_target} "
                  f"is already stricter).")
        else:
            args.min_target = 1e-9

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Year   : {ds.year}")
    print(f"Parquet: {args.parquet}")
    print(f"Target : {args.target}")
    if args.min_target is not None or args.max_target is not None:
        print(
            f"Filter : target in "
            f"[{args.min_target if args.min_target is not None else '-inf'}, "
            f"{args.max_target if args.max_target is not None else '+inf'}]"
        )
    df, X_emb, y, state = _load(
        args.parquet, args.target,
        min_target=args.min_target, max_target=args.max_target,
    )
    print(f"Shape  : emb={X_emb.shape}, y={y.shape}, states={state.nunique()}")

    if args.state_only:
        feature_set = "state"
    elif args.state_dummies:
        feature_set = "emb+state"
    else:
        feature_set = "emb"
    print(f"Features: {feature_set}")
    X, emb_dim = _build_features(
        X_emb, state,
        add_state_dummies=args.state_dummies,
        state_only=args.state_only,
    )
    print(f"  final X shape: {X.shape} (emb_dim={emb_dim}, dummy_dim={X.shape[1] - emb_dim})")

    if args.leave_one_state_out:
        folds = _leave_one_group_out(state)
        cv_name = f"leave-one-state-out (k={len(np.unique(folds))})"
    else:
        folds = _kfold_assign(len(y), args.folds, args.seed)
        cv_name = f"random {args.folds}-fold (seed={args.seed})"
    print(f"CV     : {cv_name}")

    # Resolve model hyperparameters, optionally loading tuned best params.
    cv_key = _cv_key(
        leave_one_state_out=bool(args.leave_one_state_out),
        folds=int(args.folds),
        seed=int(args.seed),
    )
    use_best = bool(args.use_best_params) and (not bool(args.no_best_params))
    best_entry = None
    if use_best:
        best = _load_best_params(_BEST_PARAMS_PATH)
        best_entry = (
            best.get(str(ds.year), {})
            .get(args.target, {})
            .get(cv_key)
        )
        if isinstance(best_entry, dict):
            if best_entry.get("feature_set") not in (None, feature_set):
                print(
                    f"Best params found for {ds.year}/{args.target}/{cv_key} but "
                    f"feature_set={best_entry.get('feature_set')} != current {feature_set}; "
                    f"ignoring."
                )
                best_entry = None
        else:
            best_entry = None

    if args.ridge_alpha is None:
        args.ridge_alpha = float(best_entry.get("ridge", {}).get("alpha", 1.0)) if best_entry else 1.0

    xgb_params: dict = {}
    if best_entry and isinstance(best_entry.get("xgboost"), dict):
        xgb_params.update(best_entry["xgboost"])
    # CLI overrides always win.
    if args.xgb_learning_rate is not None:
        xgb_params["learning_rate"] = float(args.xgb_learning_rate)
    if args.xgb_max_depth is not None:
        xgb_params["max_depth"] = int(args.xgb_max_depth)
    if args.xgb_subsample is not None:
        xgb_params["subsample"] = float(args.xgb_subsample)
    if args.xgb_colsample_bytree is not None:
        xgb_params["colsample_bytree"] = float(args.xgb_colsample_bytree)
    if args.xgb_reg_lambda is not None:
        xgb_params["reg_lambda"] = float(args.xgb_reg_lambda)
    if args.xgb_min_child_weight is not None:
        xgb_params["min_child_weight"] = float(args.xgb_min_child_weight)
    if args.xgb_gamma is not None:
        xgb_params["gamma"] = float(args.xgb_gamma)

    if best_entry:
        print(f"Params  : ridge_alpha={args.ridge_alpha} (best_params.json)")
        if xgb_params:
            print(f"        xgb_params={xgb_params}")
    else:
        print(f"Params  : ridge_alpha={args.ridge_alpha}")

    # Carry ``scenario`` through to the OOF parquet (if the source has it) so
    # ``plots.py`` can redraw the per-scenario residual figure without needing
    # the source features parquet.
    oof_cols: dict[str, np.ndarray] = {
        "ageb_id": df["ageb_id"].values,
        "state":   state.values,
        "y":       y,
    }
    if "scenario" in df.columns:
        oof_cols["scenario"] = df["scenario"].values
    oof = pd.DataFrame(oof_cols)
    overall: dict[str, dict[str, float]] = {}
    per_state: dict[str, pd.DataFrame] = {}

    # Ridge
    t0 = time.time()
    print("\n[Ridge]")
    ridge_p = _fit_ridge_oof(X, y, folds, alpha=args.ridge_alpha, emb_dim=emb_dim)
    m = _metrics(y, ridge_p)
    print(f"  overall: R^2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
          f"(fit+predict: {time.time() - t0:.1f}s)")
    overall["ridge"] = m
    oof["pred_ridge"] = ridge_p
    per_state["ridge"] = _per_state(y, ridge_p, state)

    # XGBoost
    if not args.skip_xgb:
        try:
            from xgboost import XGBRegressor  # noqa: F401
        except ImportError:
            print("\n[XGBoost] package not available; skipping. "
                  "Install with: pip install xgboost")
        else:
            t0 = time.time()
            print("\n[XGBoost]")
            xgb_p = _fit_xgb_oof(
                X, y, folds,
                n_estimators=args.xgb_rounds,
                early_stop=args.xgb_early_stop,
                xgb_params=xgb_params or None,
            )
            m = _metrics(y, xgb_p)
            print(f"  overall: R^2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                  f"(fit+predict: {time.time() - t0:.1f}s)")
            overall["xgboost"] = m
            oof["pred_xgb"] = xgb_p
            per_state["xgboost"] = _per_state(y, xgb_p, state)

    # --- Save ---
    # Tag filenames with the feature set so runs with different feature configs
    # don't overwrite each other's outputs.
    tag = feature_set.replace("+", "-")  # "emb", "emb-state", "state"

    # Auto-detect unit (AGEBs vs municipios) from the parquet filename. We
    # persist it into the meta JSON so ``plots.py`` can re-use it without
    # having to open the source features parquet.
    parquet_name = args.parquet.name.lower()
    unit_name = "municipios" if "municipio" in parquet_name else "AGEBs"

    overall_meta = {
        "feature_set": feature_set,
        "cv": cv_name,
        "target": args.target,
        "min_target": args.min_target,
        "max_target": args.max_target,
        "n_rows": int(len(y)),
        "n_states": int(state.nunique()),
        "unit_name": unit_name,
        "source_parquet": str(args.parquet),
        "tag": tag,
        "metrics": overall,
    }
    with open(args.out_dir / f"metrics_overall_{tag}.json", "w") as f:
        json.dump(overall_meta, f, indent=2)
    print(f"\nWrote {args.out_dir / f'metrics_overall_{tag}.json'}")

    state_rows = []
    for model_name, d in per_state.items():
        d = d.copy()
        d.insert(0, "model", model_name)
        state_rows.append(d)
    by_state_df = pd.concat(state_rows, ignore_index=True)
    by_state_df.to_csv(args.out_dir / f"metrics_by_state_{tag}.csv", index=False)
    print(f"Wrote {args.out_dir / f'metrics_by_state_{tag}.csv'}")

    oof.to_parquet(args.out_dir / f"predictions_oof_{tag}.parquet", index=False)
    print(f"Wrote {args.out_dir / f'predictions_oof_{tag}.parquet'}")

    print("\n--- Per-state R^2 (first 10 states) ---")
    for name, d in per_state.items():
        print(f"\n{name}:")
        print(d.head(10).to_string(index=False))

    # Figures are rendered by a separate script that re-reads these artifacts,
    # so you can iterate on plots without re-running Ridge/XGBoost.
    print(
        "\nModel artifacts written. To render figures, run:\n"
        f"  python -m embedding_extract.evaluation.plots "
        f"--out-dir {args.out_dir} --tag {tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
