"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, XGBoost, scikit-learn
Data Dependencies: Cleaned AGEB/Locality CSVs, AlphaEarth embeddings (00_all_mexico_embeddings_combined.csv)
Description: Trains final 5-fold cross-validation models (Ridge and XGBoost) using optimal 
             hyperparameters and saves checkpoints.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings" / "00_all_mexico_embeddings_combined.csv"
TUNE_PARAMS_FILE = PROJECT_ROOT / "outputs" / "models" / "best_tune_params.json"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Loads data, prepares features, and trains fold-specific Ridge and XGBoost models."""
    print("Loading data...")
    df = pd.read_csv(EMBEDDINGS_FILE)
    
    print("Creating state indicator variables...")
    # One-hot encode state codes
    state_dummies = pd.get_dummies(df['CVE_ENT'], prefix='ENT', drop_first=True)
    
    embedding_cols = [f'A{i:02d}' for i in range(64)]
    
    X_base = df[embedding_cols]
    y_log_pobtot = df["log_POBTOT"]
    y_log_popden = df["log_popden"]
    
    # Handle infinities and NaNs
    X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_log_pobtot = y_log_pobtot.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_log_popden = y_log_popden.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Filter out rows with zero population/density
    mask = (y_log_popden != 0) & (y_log_pobtot != 0)
    
    X_filt = X_base[mask].copy()
    state_dummies_filt = state_dummies[mask].copy()
    
    # Merge embeddings and state dummies
    X = pd.concat([X_filt, state_dummies_filt], axis=1)
    y_den = y_log_popden[mask].copy()
    
    print(f"Dataset size after filtering: {len(X)} rows")
    print(f"Number of features: {X.shape[1]}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Load optimal hyperparameters from tuning output
    with open(TUNE_PARAMS_FILE, "r") as f:
        tune_cfg = json.load(f)
    best_xgb_params = tune_cfg["xgb"]["best_params"]
    ridge_alpha = tune_cfg["ridge"]["best_params"]["alpha"]
    print(f"Loaded optimal XGBoost parameters: {best_xgb_params}")
    print(f"Loaded optimal Ridge alpha: {ridge_alpha}")
    
    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_den.iloc[train_index], y_den.iloc[test_index]
        
        # 1. Ridge Regression: Scale and train
        print("Training Ridge Regression...")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train_sc, y_train)
        
        y_pred_ridge = ridge.predict(X_test_sc)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        print(f"Ridge R2: {r2_ridge:.4f}")
        
        # Save Ridge artifacts
        joblib.dump(scaler, CHECKPOINT_DIR / f"scaler_fold{fold}.pkl")
        joblib.dump(ridge, CHECKPOINT_DIR / f"ridge_fold{fold}.pkl")
        
        # 2. XGBoost: Train with pre-tuned parameters
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            **best_xgb_params,
            n_jobs=-1,
            random_state=42,
            tree_method="hist"
        )
        
        xgb_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict(X_test)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        print(f"XGBoost R2: {r2_xgb:.4f}")
        
        # Save XGBoost model
        xgb_model.save_model(str(CHECKPOINT_DIR / f"xgb_fold{fold}.json"))
        
        fold += 1
        
    print("\nTraining completed and checkpoints saved.")

if __name__ == "__main__":
    main()
