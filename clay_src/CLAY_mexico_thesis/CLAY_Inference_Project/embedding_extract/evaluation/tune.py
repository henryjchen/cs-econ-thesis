"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, scikit-learn, matplotlib, seaborn, pandas
Data Dependencies: Clay embeddings (2010/2020), INEGI census data (2010/2020)
Description: Hyperparameter optimization for Ridge and XGBoost models. Updates shared parameter configuration for both 2010 and 2020 pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.evaluation import analysis as fanalysis


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


def _load_best_params(path: Path) -> dict[str, Any]:
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
    except Exception:
        return {}


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Writes a dictionary to a JSON file atomically.

    Args:
        path: Destination path.
        obj: Dictionary to save.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    """Calculates R-squared score."""
    y = y.astype(np.float64, copy=False)
    p = p.astype(np.float64, copy=False)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Samples from a log-uniform distribution on [lo, hi]."""
    if lo <= 0 or hi <= 0:
        raise ValueError("log-uniform bounds must be positive")
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


@dataclass(frozen=True)
class _TuneResult:
    """Container for hyperparameter tuning results."""
    best_score: float
    best_params: dict[str, Any]
    scores: list[dict[str, Any]]


def _tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    alphas: list[float],
    emb_dim: int,
) -> _TuneResult:
    """Performs grid search for Ridge regularization strength.

    Args:
        X: Feature matrix.
        y: Target vector.
        folds: Fold indices.
        alphas: Grid of alpha values to test.
        emb_dim: Number of embedding columns.

    Returns:
        _TuneResult object with best params and all scores.
    """
    rows: list[dict[str, Any]] = []
    best = (-np.inf, None)
    for a in alphas:
        p = fanalysis._fit_ridge_oof(X, y, folds, alpha=float(a), emb_dim=emb_dim)
        score = _r2(y, p)
        rows.append({"alpha": float(a), "r2": float(score)})
        if np.isfinite(score) and score > best[0]:
            best = (score, float(a))
    if best[1] is None:
        raise RuntimeError("Ridge tuning failed: no finite scores")
    return _TuneResult(
        best_score=float(best[0]),
        best_params={"alpha": float(best[1])},
        scores=rows,
    )


def _tune_xgb_random(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    trials: int,
    seed: int,
    n_estimators: int,
    early_stop: int,
) -> _TuneResult:
    """Performs random search for XGBoost hyperparameters.

    Args:
        X: Feature matrix.
        y: Target vector.
        folds: Fold indices.
        trials: Number of random search trials.
        seed: Random seed.
        n_estimators: Max number of trees.
        early_stop: Early stopping rounds.

    Returns:
        _TuneResult object with best params and all scores.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params: dict[str, Any] | None = None

    for t in range(int(trials)):
        params = {
            "learning_rate": _log_uniform(rng, 0.01, 0.2),
            "max_depth": int(rng.integers(3, 11)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda": _log_uniform(rng, 1e-3, 100.0),
            "reg_alpha": _log_uniform(rng, 1e-3, 10.0),
            "min_child_weight": _log_uniform(rng, 0.1, 50.0),
            "gamma": _log_uniform(rng, 1e-4, 10.0),
        }

        t0 = time.time()
        p = fanalysis._fit_xgb_oof(
            X,
            y,
            folds,
            n_estimators=n_estimators,
            early_stop=early_stop,
            xgb_params=params,
        )
        score = _r2(y, p)
        rows.append({"trial": t, "r2": float(score), "seconds": float(time.time() - t0), **params})
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_params = params
        print(f"[xgb] trial {t + 1}/{trials}: r2={score:.4f}", flush=True)

    if best_params is None:
        raise RuntimeError("XGBoost tuning failed: no finite scores")
    return _TuneResult(best_score=float(best_score), best_params=best_params, scores=rows)


def main() -> int:
    """CLI entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for CLAY features.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--target", type=str, default="log_popden")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-zero-target", action="store_true")
    parser.add_argument("--min-target", type=float, default=None)
    parser.add_argument("--max-target", type=float, default=None)
    parser.add_argument(
        "--ridge-alphas",
        type=str,
        default="1e-3,1e-2,1e-1,1,3,10,30,100,300,1000,2500,5000,7500,10000",
        help="Comma-separated ridge alpha grid.",
    )
    parser.add_argument("--xgb-trials", type=int, default=100)
    parser.add_argument("--xgb-rounds", type=int, default=2000)
    parser.add_argument("--xgb-early-stop", type=int, default=50)
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    parquet = args.parquet if args.parquet is not None else ds.polygon_features_path

    if args.drop_zero_target:
        if args.min_target is None or args.min_target <= 1e-9:
            args.min_target = 1e-9

    print(f"Year   : {ds.year}")
    print(f"Parquet: {parquet}")
    print(f"Target : {args.target}")

    df, X_emb, y, state = fanalysis._load(
        parquet,
        args.target,
        min_target=args.min_target,
        max_target=args.max_target,
    )
    feature_set = "emb"
    X, emb_dim = fanalysis._build_features(
        X_emb,
        state,
        add_state_dummies=False,
        state_only=False,
    )

    folds = fanalysis._kfold_assign(len(y), args.folds, args.seed)
    cv_key = _cv_key(leave_one_state_out=False, folds=args.folds, seed=args.seed)

    alphas = [float(x) for x in args.ridge_alphas.split(",") if x.strip()]
    print(f"CV     : {cv_key}")
    print(f"Rows   : {len(y):,}  emb_dim={emb_dim}")
    print(f"Ridge  : {len(alphas)} alphas")
    print(f"XGB    : {args.xgb_trials} trials (rounds={args.xgb_rounds}, early_stop={args.xgb_early_stop})")

    ridge_res = _tune_ridge(X, y, folds, alphas=alphas, emb_dim=emb_dim)
    print(f"[ridge] best r2={ridge_res.best_score:.4f} alpha={ridge_res.best_params['alpha']}", flush=True)

    xgb_res = _tune_xgb_random(
        X,
        y,
        folds,
        trials=args.xgb_trials,
        seed=args.seed,
        n_estimators=args.xgb_rounds,
        early_stop=args.xgb_early_stop,
    )
    print(f"[xgb] best r2={xgb_res.best_score:.4f} params={xgb_res.best_params}", flush=True)

    best = _load_best_params(_BEST_PARAMS_PATH)
    best.setdefault(str(ds.year), {})
    best[str(ds.year)].setdefault(args.target, {})
    best[str(ds.year)][args.target][cv_key] = {
        "feature_set": feature_set,
        "ridge": ridge_res.best_params,
        "xgboost": xgb_res.best_params,
    }
    _atomic_write_json(_BEST_PARAMS_PATH, best)
    print(f"Wrote {_BEST_PARAMS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    fcfg.add_year_arg(parser)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--target", type=str, default="log_popden")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-zero-target", action="store_true")
    parser.add_argument("--min-target", type=float, default=None)
    parser.add_argument("--max-target", type=float, default=None)
    parser.add_argument(
        "--ridge-alphas",
        type=str,
        default="1e-3,1e-2,1e-1,1,3,10,30,100,300,1000,2500,5000,7500,10000",
        help="Comma-separated ridge alpha grid.",
    )
    parser.add_argument("--xgb-trials", type=int, default=100)
    parser.add_argument("--xgb-rounds", type=int, default=2000)
    parser.add_argument("--xgb-early-stop", type=int, default=50)
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    parquet = args.parquet if args.parquet is not None else ds.polygon_features_path

    if args.drop_zero_target:
        if args.min_target is None or args.min_target <= 1e-9:
            args.min_target = 1e-9

    print(f"Year   : {ds.year}")
    print(f"Parquet: {parquet}")
    print(f"Target : {args.target}")

    df, X_emb, y, state = fanalysis._load(
        parquet,
        args.target,
        min_target=args.min_target,
        max_target=args.max_target,
    )
    feature_set = "emb"
    X, emb_dim = fanalysis._build_features(
        X_emb,
        state,
        add_state_dummies=False,
        state_only=False,
    )

    folds = fanalysis._kfold_assign(len(y), args.folds, args.seed)
    cv_key = _cv_key(leave_one_state_out=False, folds=args.folds, seed=args.seed)

    alphas = [float(x) for x in args.ridge_alphas.split(",") if x.strip()]
    print(f"CV     : {cv_key}")
    print(f"Rows   : {len(y):,}  emb_dim={emb_dim}")
    print(f"Ridge  : {len(alphas)} alphas")
    print(f"XGB    : {args.xgb_trials} trials (rounds={args.xgb_rounds}, early_stop={args.xgb_early_stop})")

    ridge_res = _tune_ridge(X, y, folds, alphas=alphas, emb_dim=emb_dim)
    print(f"[ridge] best r2={ridge_res.best_score:.4f} alpha={ridge_res.best_params['alpha']}", flush=True)

    xgb_res = _tune_xgb_random(
        X,
        y,
        folds,
        trials=args.xgb_trials,
        seed=args.seed,
        n_estimators=args.xgb_rounds,
        early_stop=args.xgb_early_stop,
    )
    print(f"[xgb] best r2={xgb_res.best_score:.4f} params={xgb_res.best_params}", flush=True)

    best = _load_best_params(_BEST_PARAMS_PATH)
    best.setdefault(str(ds.year), {})
    best[str(ds.year)].setdefault(args.target, {})
    best[str(ds.year)][args.target][cv_key] = {
        "feature_set": feature_set,
        "ridge": ridge_res.best_params,
        "xgboost": xgb_res.best_params,
    }
    _atomic_write_json(_BEST_PARAMS_PATH, best)
    print(f"Wrote {_BEST_PARAMS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

