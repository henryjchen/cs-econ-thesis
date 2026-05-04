"""
Project Context: 2020 Mexico Clay Pipeline
Temporal Change Analysis (2010 vs 2020)
Environment Specs: Local machine / Yale Bouchet HPC, scikit-learn, matplotlib, seaborn, pandas
Data Dependencies: Clay embeddings (2010/2020), INEGI census data (2010/2020)
Description: Evaluates model transferability and baseline comparisons across census years.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.evaluation import analysis as fanalysis


def _load_ageb_features(
    path: Path,
    target: str,
    *,
    min_target: float | None,
    max_target: float | None,
    require_target: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, pd.Series]:
    """Loads AGEB features and target from parquet with memory optimizations.

    Args:
        path: Path to the parquet file.
        target: Target column name.
        min_target: Minimum target threshold.
        max_target: Maximum target threshold.
        require_target: Whether to fail if target is missing.

    Returns:
        Tuple of (DataFrame, X matrix, y vector, state labels).
    """
    schema_names = pq.read_schema(path).names
    emb_cols = sorted(
        [c for c in schema_names if c.startswith("emb_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if len(emb_cols) != fcfg.EMBEDDING_DIM:
        raise RuntimeError(f"{path}: expected {fcfg.EMBEDDING_DIM} emb columns, found {len(emb_cols)}")
    if "ageb_id" not in schema_names:
        raise KeyError(f"{path}: missing ageb_id")
    if require_target and target not in schema_names:
        raise KeyError(f"{path}: missing target {target}")

    read_cols = ["ageb_id", *emb_cols]
    if require_target:
        read_cols.insert(1, target)
    df = pd.read_parquet(path, columns=read_cols)

    df[emb_cols] = df[emb_cols].astype(np.float32, copy=False)
    if require_target:
        df[target] = df[target].astype(np.float32, copy=False)

    mask = df[emb_cols].notna().all(axis=1)
    if require_target:
        mask &= np.isfinite(df[target])
    dropped = int((~mask).sum())
    if dropped:
        print(f"{path.name}: dropping {dropped:,} rows with non-finite embeddings/target")
    df = df.loc[mask].copy()

    if require_target and (min_target is not None or max_target is not None):
        tvals = df[target].to_numpy()
        keep = np.ones(len(df), dtype=bool)
        if min_target is not None:
            keep &= tvals >= float(min_target)
        if max_target is not None:
            keep &= tvals <= float(max_target)
        n_cut = int((~keep).sum())
        if n_cut:
            print(f"{path.name}: dropping {n_cut:,} rows outside target range")
        df = df.loc[keep].copy()

    before = len(df)
    df = df.drop_duplicates("ageb_id", keep="first").reset_index(drop=True)
    if len(df) != before:
        print(f"{path.name}: dropped {before - len(df):,} duplicate ageb_id rows")

    X = df[emb_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[target].to_numpy(dtype=np.float32, copy=False) if require_target else None
    state = fanalysis._state_code(df["ageb_id"])
    return df, X, y, state


def _parquet_nrows(path: Path) -> int:
    """Returns the number of rows in a parquet file."""
    meta = pq.ParquetFile(path).metadata
    if meta is None:
        return 0
    return int(meta.num_rows)


def _print_memory_estimate(path_2010: Path, path_2020: Path) -> None:
    """Prints a RAM estimate for the numeric arrays.

    Args:
        path_2010: Path to 2010 feature parquet.
        path_2020: Path to 2020 feature parquet.
    """
    n10 = _parquet_nrows(path_2010)
    n20 = _parquet_nrows(path_2020)
    d = fcfg.EMBEDDING_DIM
    f32 = np.dtype(np.float32).itemsize

    x10_gb = n10 * d * f32 / 1e9
    x20_gb = n20 * d * f32 / 1e9
    matched_lower = min(n10, n20)
    baseline_delta_gb = matched_lower * (2 * d) * f32 / 1e9
    lower_bound = x10_gb + x20_gb + baseline_delta_gb

    print("\nApprox dense-array RAM lower bound (float32 only):")
    print(f"  X2010 embeddings       : {x10_gb:6.2f} GB  ({n10:,} x {d})")
    print(f"  X2020 embeddings       : {x20_gb:6.2f} GB  ({n20:,} x {d})")
    print(f"  [X2010, X2020-X2010]   : {baseline_delta_gb:6.2f} GB  (<= {matched_lower:,} x {2*d})")
    print(f"  subtotal lower bound   : {lower_bound:6.2f} GB")
    print("  note: pandas merge + model buffers can make true peak materially higher.\n")


def _best_params(target: str, *, folds: int, seed: int) -> tuple[float, dict]:
    """Retrieves best params, favoring 2020 tuned values.

    Args:
        target: Target column name.
        folds: CV fold count.
        seed: Random seed.

    Returns:
        Tuple of (ridge_alpha, xgb_params).
    """
    best = fanalysis._load_best_params(Path(fanalysis.__file__).with_name("best_params.json"))
    key = fanalysis._cv_key(leave_one_state_out=False, folds=folds, seed=seed)
    entry = (
        best.get("2020", {}).get(target, {}).get(key)
        or best.get("2010", {}).get(target, {}).get(key)
        or {}
    )
    ridge_alpha = float(entry.get("ridge", {}).get("alpha", 1.0))
    xgb_params = dict(entry.get("xgboost", {})) if isinstance(entry.get("xgboost"), dict) else {}
    return ridge_alpha, xgb_params


def _fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Trains Ridge on train set and predicts on test set."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(Xtr, y_train)
    return model.predict(Xte).astype(np.float32)


def _fit_predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    params: dict,
    n_estimators: int,
) -> np.ndarray:
    """Trains XGBoost on train set and predicts on test set."""
    from xgboost import XGBRegressor

    p = dict(params)
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
        n_jobs=-1,
        **p,
    )
    model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test).astype(np.float32)


def _write_metrics(
    out_dir: Path,
    *,
    name: str,
    target: str,
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    state: pd.Series,
    meta: dict,
) -> None:
    """Saves overall and per-state metrics to disk."""
    metrics = {model: fanalysis._metrics(y, p) for model, p in preds.items()}
    with open(out_dir / f"metrics_overall_{name}.json", "w") as f:
        json.dump({"target": target, "experiment": name, "metrics": metrics, **meta}, f, indent=2)

    state_rows = []
    for model, p in preds.items():
        d = fanalysis._per_state(y, p, state)
        d.insert(0, "model", model)
        state_rows.append(d)
    pd.concat(state_rows, ignore_index=True).to_csv(
        out_dir / f"metrics_by_state_{name}.csv", index=False
    )
    print(f"Wrote metrics for {name}: {metrics}")


def main() -> int:
    """CLI entry point for cross-temporal evaluation."""
    parser = argparse.ArgumentParser(description="Cross-temporal evaluation (2010 -> 2020).")
    parser.add_argument("--target", type=str, default="log_popden")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-zero-target", action="store_true")
    parser.add_argument("--min-target", type=float, default=None)
    parser.add_argument("--max-target", type=float, default=None)
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--xgb-rounds", type=int, default=2000)
    parser.add_argument("--xgb-early-stop", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    ds2010 = fcfg.for_year(2010)
    ds2020 = fcfg.for_year(2020)
    out_dir = args.out_dir or (ds2020.out_dir / "cross_temporal")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.drop_zero_target and (args.min_target is None or args.min_target <= 1e-9):
        args.min_target = 1e-9

    ridge_alpha, xgb_params = _best_params(args.target, folds=args.folds, seed=args.seed)
    print(f"2010 parquet: {ds2010.polygon_features_path}")
    print(f"2020 parquet: {ds2020.polygon_features_path}")
    print(f"Output      : {out_dir}")
    print(f"Target      : {args.target}")
    print(f"Params      : ridge_alpha={ridge_alpha}; xgb_params={xgb_params}")
    _print_memory_estimate(ds2010.polygon_features_path, ds2020.polygon_features_path)

    df10, X10, y10, _state10 = _load_ageb_features(
        ds2010.polygon_features_path,
        args.target,
        min_target=args.min_target,
        max_target=args.max_target,
        require_target=True,
    )
    df20, X20, y20, state20 = _load_ageb_features(
        ds2020.polygon_features_path,
        args.target,
        min_target=args.min_target,
        max_target=args.max_target,
        require_target=True,
    )
    assert y10 is not None and y20 is not None

    # Experiment 1: train 2010 -> predict 2020.
    print("\n[transfer_2010_to_2020]")
    t0 = time.time()
    transfer_preds = {
        "ridge": _fit_predict_ridge(X10, y10, X20, alpha=ridge_alpha),
    }
    if not args.skip_xgb:
        transfer_preds["xgboost"] = _fit_predict_xgb(
            X10, y10, X20, params=xgb_params, n_estimators=args.xgb_rounds
        )
    pred_transfer = pd.DataFrame({
        "ageb_id": df20["ageb_id"].values,
        "state": state20.values,
        "y": y20,
        **{f"pred_{k}": v for k, v in transfer_preds.items()},
    })
    pred_transfer.to_parquet(out_dir / "predictions_transfer_2010_to_2020.parquet", index=False)
    _write_metrics(
        out_dir,
        name="transfer_2010_to_2020",
        target=args.target,
        y=y20,
        preds=transfer_preds,
        state=state20,
        meta={
            "train_year": 2010,
            "test_year": 2020,
            "n_train": int(len(y10)),
            "n_test": int(len(y20)),
            "seconds": float(time.time() - t0),
        },
    )

    # Shared matched-ID panel for experiments 2 and 3.
    print("\n[matched_2010_2020_panel]")
    emb_cols = fanalysis._emb_columns(df10)
    keep_cols = ["ageb_id", args.target, *emb_cols]
    left = df10[keep_cols].add_suffix("_2010").rename(columns={"ageb_id_2010": "ageb_id"})
    right = df20[keep_cols].add_suffix("_2020").rename(columns={"ageb_id_2020": "ageb_id"})
    matched = left.merge(right, on="ageb_id", how="inner")
    del left
    del right
    gc.collect()
    print(f"Matched AGEBs: {len(matched):,}")

    X10m = matched[[f"{c}_2010" for c in emb_cols]].to_numpy(dtype=np.float32, copy=False)
    X20m = matched[[f"{c}_2020" for c in emb_cols]].to_numpy(dtype=np.float32, copy=False)
    # Avoid building a full temporary `(X20m - X10m)` array before concatenate.
    # Allocate the final matrix once, then subtract in-place in its second half.
    emb_dim = X10m.shape[1]
    X_delta = np.empty((X10m.shape[0], emb_dim * 2), dtype=np.float32)
    X_delta[:, :emb_dim] = X10m
    X_delta[:, emb_dim:] = X20m
    X_delta[:, emb_dim:] -= X10m
    y_delta = matched[f"{args.target}_2020"].to_numpy(dtype=np.float32, copy=False)
    state_delta = fanalysis._state_code(matched["ageb_id"])
    folds = fanalysis._kfold_assign(len(y_delta), args.folds, args.seed)
    del X20m
    gc.collect()

    # Experiment 2: matched-ID 2010 baseline embeddings -> 2020 target.
    print("\n[baseline_2010_2020_cv]")
    t0 = time.time()
    baseline_preds = {
        "ridge": fanalysis._fit_ridge_oof(
            X10m, y_delta, folds, alpha=ridge_alpha, emb_dim=X10m.shape[1]
        ),
    }
    if not args.skip_xgb:
        baseline_preds["xgboost"] = fanalysis._fit_xgb_oof(
            X10m,
            y_delta,
            folds,
            n_estimators=args.xgb_rounds,
            early_stop=args.xgb_early_stop,
            xgb_params=xgb_params or None,
        )
    pred_baseline = pd.DataFrame({
        "ageb_id": matched["ageb_id"].values,
        "state": state_delta.values,
        "y": y_delta,
        **{f"pred_{k}": v for k, v in baseline_preds.items()},
    })
    pred_baseline.to_parquet(out_dir / "predictions_baseline_2010_2020_cv.parquet", index=False)
    _write_metrics(
        out_dir,
        name="baseline_2010_2020_cv",
        target=args.target,
        y=y_delta,
        preds=baseline_preds,
        state=state_delta,
        meta={
            "cv": f"random {args.folds}-fold (seed={args.seed})",
            "feature_set": "emb2010",
            "n_rows": int(len(y_delta)),
            "seconds": float(time.time() - t0),
        },
    )

    # Experiment 3: matched-ID baseline + embedding delta -> 2020 target.
    print("\n[baseline_delta_2020_cv]")
    t0 = time.time()
    delta_preds = {
        "ridge": fanalysis._fit_ridge_oof(
            X_delta, y_delta, folds, alpha=ridge_alpha, emb_dim=X_delta.shape[1]
        ),
    }
    if not args.skip_xgb:
        delta_preds["xgboost"] = fanalysis._fit_xgb_oof(
            X_delta,
            y_delta,
            folds,
            n_estimators=args.xgb_rounds,
            early_stop=args.xgb_early_stop,
            xgb_params=xgb_params or None,
        )
    pred_delta = pd.DataFrame({
        "ageb_id": matched["ageb_id"].values,
        "state": state_delta.values,
        "y": y_delta,
        **{f"pred_{k}": v for k, v in delta_preds.items()},
    })
    pred_delta.to_parquet(out_dir / "predictions_baseline_delta_2020_cv.parquet", index=False)
    _write_metrics(
        out_dir,
        name="baseline_delta_2020_cv",
        target=args.target,
        y=y_delta,
        preds=delta_preds,
        state=state_delta,
        meta={
            "cv": f"random {args.folds}-fold (seed={args.seed})",
            "feature_set": "emb2010_plus_delta",
            "n_rows": int(len(y_delta)),
            "seconds": float(time.time() - t0),
        },
    )

    print("\nDone. Outputs:")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

