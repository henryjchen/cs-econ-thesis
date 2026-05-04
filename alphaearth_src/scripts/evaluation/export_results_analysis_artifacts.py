"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, XGBoost, scikit-learn
Data Dependencies: Cleaned AGEB/Locality CSVs, AlphaEarth embeddings (00_all_mexico_embeddings_combined.csv)
Description: Generates CLAY-compatible evaluation artifacts (metrics and predictions) 
             for AGEB and municipio levels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load sibling module without package imports
_spec = importlib.util.spec_from_file_location(
    "municipio_evaluation",
    Path(__file__).resolve().parent / "municipio_evaluation.py",
)
assert _spec and _spec.loader
_mun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mun)


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y, p)))


def _metrics_vec(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """
    Computes standard regression metrics.

    Args:
        y: Ground truth values.
        p: Predicted values.

    Returns:
        dict: r2, mae, rmse, and n.
    """
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": _rmse(y, p),
        "n": int(len(y)),
    }


def _per_state(y: np.ndarray, p: np.ndarray, state: pd.Series) -> pd.DataFrame:
    """
    Computes metrics grouped by state.

    Args:
        y: Ground truth values.
        p: Predicted values.
        state: Series of state identifiers.

    Returns:
        pd.DataFrame: Metrics per state.
    """
    df = pd.DataFrame({"state": state.astype(str).str.zfill(2).values, "y": y, "p": p})
    rows = []
    for s, g in df.groupby("state"):
        yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
        if len(g) < 2 or np.nanstd(yy) == 0:
            rows.append(
                {
                    "state": s,
                    "n": len(g),
                    "r2": np.nan,
                    "mae": float(mean_absolute_error(yy, pp)),
                    "rmse": _rmse(yy, pp),
                }
            )
        else:
            rows.append({"state": s, **_metrics_vec(yy, pp)})
    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def _write_bundle(
    out_dir: Path,
    *,
    oof: pd.DataFrame,
    y_col: str,
    source_label: str,
    unit_name: str,
    cv_name: str,
    target: str,
) -> None:
    """
    Writes a bundle of metrics (JSON), state metrics (CSV), and OOF predictions (Parquet).

    Args:
        out_dir: Output directory.
        oof: DataFrame containing OOF predictions.
        y_col: Column name for ground truth.
        source_label: Data source description.
        unit_name: Geographic unit name (e.g., AGEBs).
        cv_name: CV strategy description.
        target: Target variable name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y = oof[y_col].to_numpy(dtype=float)
    st = oof["state"]
    tag = "emb"

    overall: dict[str, dict[str, float]] = {
        "ridge": _metrics_vec(y, oof["pred_ridge"].to_numpy(dtype=float)),
        "xgboost": _metrics_vec(y, oof["pred_xgb"].to_numpy(dtype=float)),
    }
    meta = {
        "feature_set": "emb",
        "cv": cv_name,
        "target": target,
        "min_target": None,
        "max_target": None,
        "n_rows": int(len(oof)),
        "n_states": int(st.astype(str).str.zfill(2).nunique()),
        "unit_name": unit_name,
        "source_parquet": source_label,
        "tag": tag,
        "metrics": overall,
    }
    with open(out_dir / f"metrics_overall_{tag}.json", "w") as f:
        json.dump(meta, f, indent=2)

    state_parts = []
    for model_name, col in [("ridge", "pred_ridge"), ("xgboost", "pred_xgb")]:
        d = _per_state(y, oof[col].to_numpy(dtype=float), st)
        d.insert(0, "model", model_name)
        state_parts.append(d)
    pd.concat(state_parts, ignore_index=True).to_csv(
        out_dir / f"metrics_by_state_{tag}.csv", index=False
    )

    oof_out = oof.copy()
    oof_out.to_parquet(out_dir / f"predictions_oof_{tag}.parquet", index=False)
    print(f"Wrote {out_dir} (metrics + parquet, n={len(oof_out):,})")


def main() -> int:
    """Main execution block: exports artifacts for AGEB and municipio levels."""
    p = argparse.ArgumentParser(description="Export evaluation artifacts.")
    default_results = PROJECT_ROOT.parent / "results_analysis" / "alphaearth2020"
    p.add_argument(
        "--out-root",
        type=Path,
        default=default_results,
        help="Root folder for analysis_* subdirectories.",
    )
    args = p.parse_args()
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    cv_name = "random 5-fold (seed=42)"
    target = "log_popden"
    csv_path = _mun.EMBEDDINGS_FILE
    source_label = str(csv_path.relative_to(PROJECT_ROOT))

    df = pd.read_csv(csv_path)
    X, y_den, mask, _state_dummy_cols = _mun.load_xy(df)
    oof_ridge, oof_xgb = _mun.collect_oof_predictions(X, y_den)
    work = df.loc[mask].copy()
    work["state"] = work["CVE_ENT"].astype(str).str.zfill(2)
    
    if "CVEGEO" not in work.columns:
        raise KeyError("Expected CVEGEO in embeddings CSV for ageb_id column.")
    
    # AGEB-level artifacts
    ageb_oof = pd.DataFrame(
        {
            "ageb_id": work["CVEGEO"].astype(str),
            "state": work["state"],
            "y": y_den.to_numpy(dtype=float),
            "pred_ridge": oof_ridge,
            "pred_xgb": oof_xgb,
        }
    )
    _write_bundle(
        out_root / "analysis_ageb",
        oof=ageb_oof,
        y_col="y",
        source_label=source_label,
        unit_name="AGEBs",
        cv_name=cv_name,
        target=target,
    )

    mun_out = _mun.evaluate_municipio(min_polygons=1)
    rollup = mun_out["rollup_table"]
    r_series: pd.Series = mun_out["direct_pred_ridge"]
    x_series: pd.Series = mun_out["direct_pred_xgb"]

    # Municipio rollup artifacts
    rollup_oof = pd.DataFrame(
        {
            "ageb_id": rollup.index.astype(str),
            "state": rollup.index.astype(str).str.slice(0, 2),
            "y": rollup["ln_density"].to_numpy(dtype=float),
            "pred_ridge": rollup["ln_density_roll_r"].to_numpy(dtype=float),
            "pred_xgb": rollup["ln_density_roll_x"].to_numpy(dtype=float),
        }
    )
    _write_bundle(
        out_root / "analysis_muni_rollup",
        oof=rollup_oof,
        y_col="y",
        source_label=source_label + " (municipio rollup from polygon OOF)",
        unit_name="municipios",
        cv_name=cv_name,
        target=target,
    )

    # Municipio direct artifacts
    idx = pd.Index(rollup.index)
    y_m = rollup["ln_density"].loc[idx].to_numpy(dtype=float)
    rr = r_series.reindex(idx).to_numpy(dtype=float)
    xx = x_series.reindex(idx).to_numpy(dtype=float)
    direct_oof = pd.DataFrame(
        {
            "ageb_id": rollup.index.astype(str),
            "state": rollup.index.astype(str).str.slice(0, 2),
            "y": y_m,
            "pred_ridge": rr,
            "pred_xgb": xx,
        }
    )
    _write_bundle(
        out_root / "analysis_muni_direct",
        oof=direct_oof,
        y_col="y",
        source_label=source_label + " (municipio direct, fold-mean checkpoint preds)",
        unit_name="municipios",
        cv_name=cv_name,
        target=target,
    )

    print(f"\nAll bundles written under: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
