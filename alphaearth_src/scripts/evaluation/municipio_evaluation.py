"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, XGBoost, scikit-learn
Data Dependencies: Cleaned AGEB/Locality CSVs, AlphaEarth embeddings (00_all_mexico_embeddings_combined.csv)
Description: Evaluates population density models at the municipality level using 
             both roll-up and direct aggregation approaches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings" / "00_all_mexico_embeddings_combined.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "models" / "checkpoints"
DEFAULT_METRICS_JSON = PROJECT_ROOT / "outputs" / "evaluation" / "municipio_evaluation.json"

_NON_SERIALIZED_KEYS = frozenset({"rollup_table", "direct_pred_ridge", "direct_pred_xgb"})


def metrics_summary_dict(result: dict) -> dict:
    """
    Filters a result dictionary for JSON-serializable metric keys.

    Args:
        result: Dictionary of evaluation results.

    Returns:
        dict: Filtered dictionary without DataFrames or Series.
    """
    return {k: v for k, v in result.items() if k not in _NON_SERIALIZED_KEYS}


def save_municipio_metrics_json(
    result: dict,
    path: Path | None = None,
) -> Path:
    """
    Writes filtered municipality metrics to a JSON file.

    Args:
        result: Dictionary of evaluation results.
        path: Optional output path.

    Returns:
        Path: The path to the written JSON file.
    """
    out_path = path or DEFAULT_METRICS_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_summary_dict(result), f, indent=2)
    return out_path


def _emb_cols() -> list[str]:
    """Returns the list of 64 embedding column names."""
    return [f"A{i:02d}" for i in range(64)]


def load_xy(df: pd.DataFrame):
    """
    Prepares features and target for evaluation, matching training logic.

    Args:
        df: Input DataFrame with embeddings and census data.

    Returns:
        tuple: (X, y_den, mask, state_dummy_cols)
    """
    state_dummies = pd.get_dummies(df["CVE_ENT"], prefix="ENT", drop_first=True)
    embedding_cols = _emb_cols()
    X_base = df[embedding_cols]
    y_log_pobtot = df["log_POBTOT"]
    y_log_popden = df["log_popden"]
    X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_log_pobtot = y_log_pobtot.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_log_popden = y_log_popden.replace([np.inf, -np.inf], np.nan).fillna(0)
    mask = (y_log_popden != 0) & (y_log_pobtot != 0)
    X_filt = X_base[mask].copy()
    state_dummies_filt = state_dummies[mask].copy()
    X = pd.concat([X_filt, state_dummies_filt], axis=1)
    y_den = y_log_popden[mask].copy()
    return X, y_den, mask, state_dummies.columns


def mun_key_series(df: pd.DataFrame) -> pd.Series:
    """Constructs 5-digit ENT+MUN keys as strings."""
    ent = pd.to_numeric(df["CVE_ENT"], errors="coerce")
    mun = pd.to_numeric(df["CVE_MUN"], errors="coerce")
    return (
        ent.astype("Int64").astype(str).str.zfill(2)
        + mun.astype("Int64").astype(str).str.zfill(3)
    )


def collect_oof_predictions(X: pd.DataFrame, y_den: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Collects out-of-fold predictions using saved checkpoints.

    Args:
        X: Feature matrix.
        y_den: Target series.

    Returns:
        tuple: (oof_ridge, oof_xgb) arrays of predictions.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_ridge = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    fold = 1
    for train_index, test_index in kf.split(X):
        scaler = joblib.load(CHECKPOINT_DIR / f"scaler_fold{fold}.pkl")
        ridge = joblib.load(CHECKPOINT_DIR / f"ridge_fold{fold}.pkl")
        model = xgb.XGBRegressor()
        model.load_model(str(CHECKPOINT_DIR / f"xgb_fold{fold}.json"))
        X_test = X.iloc[test_index]
        oof_ridge[test_index] = ridge.predict(scaler.transform(X_test))
        oof_xgb[test_index] = model.predict(X_test)
        fold += 1
    return oof_ridge, oof_xgb


def rollup_municipio_table(
    work: pd.DataFrame,
    oof_ridge: np.ndarray,
    oof_xgb: np.ndarray,
    y_ln_polygon: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregates polygon predictions to municipio level using area-weighted sums.

    Args:
        work: Filtered DataFrame of polygons.
        oof_ridge: OOF Ridge predictions.
        oof_xgb: OOF XGBoost predictions.
        y_ln_polygon: Ground truth log-density.

    Returns:
        pd.DataFrame: Table of aggregated municipio metrics.
    """
    area = (
        pd.to_numeric(work["area_sqkm"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    pred_pop_r = np.exp(oof_ridge) * area
    pred_pop_x = np.exp(oof_xgb) * area
    true_pop = np.exp(y_ln_polygon) * area

    g = (
        pd.DataFrame(
            {
                "mun_key": work["mun_key"].values,
                "area": area.values,
                "true_pop": true_pop,
                "pred_pop_r": pred_pop_r,
                "pred_pop_x": pred_pop_x,
            }
        )
        .groupby("mun_key", sort=True)
        .agg(
            area_sum=("area", "sum"),
            true_pop_sum=("true_pop", "sum"),
            pred_pop_sum_r=("pred_pop_r", "sum"),
            pred_pop_sum_x=("pred_pop_x", "sum"),
        )
    )
    g = g[g["area_sum"] > 0]
    poly_counts = work.groupby("mun_key").size()
    g["n_polygons"] = poly_counts.reindex(g.index).fillna(0).astype(int)
    eps = 1e-12
    g["ln_density"] = np.log(g["true_pop_sum"].clip(lower=eps) / g["area_sum"])
    g["ln_density_roll_r"] = np.log(g["pred_pop_sum_r"].clip(lower=eps) / g["area_sum"])
    g["ln_density_roll_x"] = np.log(g["pred_pop_sum_x"].clip(lower=eps) / g["area_sum"])
    return g


def direct_municipio_predictions(
    work: pd.DataFrame,
    X_columns: pd.Index,
    state_dummy_cols: pd.Index,
    emb_cols: list[str],
    ent_categories: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts municipio density using area-weighted mean embeddings.

    Args:
        work: DataFrame of polygons.
        X_columns: Expected feature names.
        state_dummy_cols: Expected state dummy names.
        emb_cols: Embedding column names.
        ent_categories: List of all possible ENT codes.

    Returns:
        tuple: (ridge_mean, xgb_mean) predictions averaged across folds.
    """
    mun_keys = []
    rows = []
    for mk, grp in work.groupby("mun_key", sort=False):
        w = (
            pd.to_numeric(grp["area_sqkm"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=1e-12)
            .to_numpy()
        )
        emb = grp[emb_cols].to_numpy(dtype=float)
        wm = (emb * w[:, None]).sum(axis=0) / w.sum()
        st = grp["CVE_ENT"].iloc[0]
        mun_keys.append(mk)
        rows.append((wm, st))

    emb_mat = np.stack([r[0] for r in rows], axis=0)
    X_emb = pd.DataFrame(emb_mat, columns=emb_cols, index=mun_keys)

    state_cat = pd.Categorical([r[1] for r in rows], categories=ent_categories)
    st_dum = pd.get_dummies(state_cat, prefix="ENT", drop_first=True)
    st_dum.index = mun_keys
    st_dum = st_dum.reindex(columns=list(state_dummy_cols), fill_value=0)

    X_mun = pd.concat([X_emb, st_dum], axis=1)
    X_mun.index = mun_keys
    X_mun = X_mun.reindex(columns=list(X_columns), fill_value=0.0)

    ridge_preds = []
    xgb_preds = []
    for fold in range(1, 6):
        scaler = joblib.load(CHECKPOINT_DIR / f"scaler_fold{fold}.pkl")
        ridge = joblib.load(CHECKPOINT_DIR / f"ridge_fold{fold}.pkl")
        model = xgb.XGBRegressor()
        model.load_model(str(CHECKPOINT_DIR / f"xgb_fold{fold}.json"))
        ridge_preds.append(ridge.predict(scaler.transform(X_mun)))
        xgb_preds.append(model.predict(X_mun))

    ridge_mean = np.mean(np.stack(ridge_preds, axis=0), axis=0)
    xgb_mean = np.mean(np.stack(xgb_preds, axis=0), axis=0)
    return ridge_mean, xgb_mean


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Helper to compute r2, mae, and median_ae for log-density."""
    return {
        "r2_ln": float(r2_score(y_true, y_pred)),
        "mae_ln": float(mean_absolute_error(y_true, y_pred)),
        "med_ae_ln": float(median_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def mun_keys_series_unique(work: pd.DataFrame) -> pd.Index:
    """Returns unique municipio keys in original groupby order."""
    keys = []
    for mk, _ in work.groupby("mun_key", sort=False):
        keys.append(mk)
    return pd.Index(keys)


def evaluate_municipio(
    min_polygons: int = 1,
) -> dict:
    """
    Main evaluation routine for municipality-level performance.

    Args:
        min_polygons: Minimum number of polygons required to include a municipio.

    Returns:
        dict: Bundle of metrics and prediction tables.
    """
    df = pd.read_csv(EMBEDDINGS_FILE)
    X, y_den, mask, state_dummy_cols = load_xy(df)
    oof_ridge, oof_xgb = collect_oof_predictions(X, y_den)

    ent_categories = sorted(pd.unique(pd.to_numeric(df["CVE_ENT"], errors="coerce").dropna()))

    work = df.loc[mask].copy()
    work["mun_key"] = mun_key_series(work)
    work = work[work["mun_key"].notna() & (work["mun_key"].str.len() == 5)]

    y_ln_vec = y_den.to_numpy(dtype=float)

    rollup = rollup_municipio_table(work, oof_ridge, oof_xgb, y_ln_vec)
    if min_polygons > 1:
        rollup = rollup[rollup["n_polygons"] >= min_polygons]

    y_act = rollup["ln_density"].to_numpy()

    y_rr = rollup["ln_density_roll_r"].to_numpy()
    y_rx = rollup["ln_density_roll_x"].to_numpy()

    roll_r = _metrics(y_act, y_rr)
    roll_x = _metrics(y_act, y_rx)

    # Direct approach: predict on aggregated embeddings
    truth_idx = list(rollup.index)
    work_mun = work[work["mun_key"].isin(truth_idx)]

    r_dir, x_dir = direct_municipio_predictions(
        work_mun,
        X.columns,
        state_dummy_cols,
        _emb_cols(),
        ent_categories,
    )
    r_series = pd.Series(r_dir, index=mun_keys_series_unique(work_mun))
    x_series = pd.Series(x_dir, index=r_series.index)
    r_series = r_series.reindex(truth_idx)
    x_series = x_series.reindex(truth_idx)

    dir_r = _metrics(y_act, r_series.to_numpy(dtype=float))
    dir_x = _metrics(y_act, x_series.to_numpy(dtype=float))

    return {
        "n_municipios": int(len(rollup)),
        "min_polygons_filter": min_polygons,
        "rollup_ridge_ln_density": roll_r,
        "rollup_xgb_ln_density": roll_x,
        "direct_ridge_ln_density": dir_r,
        "direct_xgb_ln_density": dir_x,
        "rollup_table": rollup,
        "direct_pred_ridge": r_series,
        "direct_pred_xgb": x_series,
    }


def main():
    """Main execution block: computes and prints/saves municipality metrics."""
    p = argparse.ArgumentParser(description="Municipality-level AlphaEarth evaluation.")
    p.add_argument("--min-polygons", type=int, default=1, help="Drop municipios with fewer polygons.")
    p.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_METRICS_JSON,
        help="Path for metrics JSON.",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Print metrics only; do not write a JSON file.",
    )
    args = p.parse_args()

    out = evaluate_municipio(min_polygons=args.min_polygons)
    summary = metrics_summary_dict(out)
    print(json.dumps(summary, indent=2))
    if not args.no_json:
        written = save_municipio_metrics_json(out, path=args.json_out)
        print(f"\nWrote metrics JSON to: {written}")


if __name__ == "__main__":
    main()
