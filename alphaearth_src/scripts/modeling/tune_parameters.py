"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, XGBoost, scikit-learn
Data Dependencies: Cleaned AGEB/Locality CSVs, AlphaEarth embeddings (00_all_mexico_embeddings_combined.csv)
Description: Performs hyperparameter tuning for Ridge and XGBoost models using RandomizedSearchCV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings" / "00_all_mexico_embeddings_combined.csv"
TUNE_OUTPUT_FILE = PROJECT_ROOT / "outputs" / "models" / "best_tune_params.json"


def _json_safe(obj):
    """
    Recursively converts numpy types to standard Python types for JSON serialization.

    Args:
        obj: The object to convert.

    Returns:
        The JSON-safe version of the object.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_xy_popden():
    """
    Loads embeddings and census data, creates state dummies, and filters the dataset.

    Returns:
        tuple: (X, y_den) containing the feature matrix and target series.
    """
    df = pd.read_csv(EMBEDDINGS_FILE)
    state_dummies = pd.get_dummies(df["CVE_ENT"], prefix="ENT", drop_first=True)
    embedding_cols = [f"A{i:02d}" for i in range(64)]
    X_base = df[embedding_cols]
    y_log_pobtot = df["log_POBTOT"]
    y_log_popden = df["log_popden"]
    
    # Clean data
    X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_log_pobtot = y_log_pobtot.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_log_popden = y_log_popden.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Filter rows with zero pop/density
    mask = (y_log_popden != 0) & (y_log_pobtot != 0)
    X_filt = X_base[mask].copy()
    state_dummies_filt = state_dummies[mask].copy()
    X = pd.concat([X_filt, state_dummies_filt], axis=1)
    y_den = y_log_popden[mask].copy()
    return X, y_den


def main():
    """Main execution block: tunes XGBoost and Ridge, then saves optimal parameters."""
    print("Loading data...")
    X, y_den = load_xy_popden()
    print(f"Dataset size after filtering: {len(X)} rows")
    print(f"Number of features: {X.shape[1]}")

    X_train, _, y_train, _ = train_test_split(
        X, y_den, test_size=0.2, random_state=42
    )

    # --- XGBoost Tuning ---
    xgb_param_distributions = {
        "n_estimators": [500, 800, 1000, 1200, 1500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    print("\nStarting RandomizedSearchCV for XGBoost...")
    xgb_base = xgb.XGBRegressor(n_jobs=-1, random_state=42, tree_method="hist")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_distributions,
        n_iter=30,
        scoring="r2",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = _json_safe(xgb_search.best_params_)
    print(f"\nBest XGBoost Params: {xgb_best}")
    print(f"Best CV R2 (XGB): {xgb_search.best_score_:.4f}")

    # --- Ridge Tuning ---
    ridge_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    ridge_param_distributions = {
        "ridge__alpha": loguniform(1e-3, 1e4),
    }
    print("\nStarting RandomizedSearchCV for Ridge...")
    ridge_search = RandomizedSearchCV(
        estimator=ridge_pipe,
        param_distributions=ridge_param_distributions,
        n_iter=40,
        scoring="r2",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    ridge_search.fit(X_train, y_train)
    ridge_alpha = float(ridge_search.best_params_["ridge__alpha"])
    print(f"\nBest Ridge alpha: {ridge_alpha}")
    print(f"Best CV R2 (Ridge): {ridge_search.best_score_:.4f}")

    # Prepare output payload
    payload = {
        "target": "log_popden",
        "xgb": {
            "best_params": xgb_best,
            "best_cv_r2": float(xgb_search.best_score_),
        },
        "ridge": {
            "best_params": {"alpha": ridge_alpha},
            "best_cv_r2": float(ridge_search.best_score_),
        },
    }

    TUNE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving tuning results to {TUNE_OUTPUT_FILE}...")
    with open(TUNE_OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("Tuning completed successfully.")


if __name__ == "__main__":
    main()
