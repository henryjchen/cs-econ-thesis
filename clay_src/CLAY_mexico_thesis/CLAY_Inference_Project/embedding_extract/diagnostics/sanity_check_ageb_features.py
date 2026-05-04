"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Sanity checks for ageb_features.parquet, validating schema, row counts, scenario labels, and embedding finiteness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg


def _emb_columns(df: pd.DataFrame) -> list[str]:
    """Retrieves and sorts embedding columns numerically.

    Args:
        df: Input DataFrame.

    Returns:
        Sorted list of embedding column names.
    """
    cols = [c for c in df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def _fail(msgs: list[str], msg: str) -> None:
    """Logs and prints a failure message."""
    msgs.append(f"FAIL: {msg}")
    print(f"  FAIL: {msg}")


def _ok(msgs: list[str], msg: str) -> None:
    """Logs and prints a success message."""
    msgs.append(f"OK: {msg}")
    print(f"  OK: {msg}")


def main() -> int:
    """Main execution entry point for AGEB features sanity checks."""
    parser = argparse.ArgumentParser(description="Sanity check AGEB features.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--parquet", type=Path, default=None, help="Input parquet path.")
    parser.add_argument("--shapefile", type=Path, default=None, help="Reference shapefile path.")
    parser.add_argument("--no-shapefile", action="store_true", help="Skip shapefile cross-check.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    parquet_path = args.parquet or ds.polygon_features_path
    shapefile_path = args.shapefile or ds.shapefile_path

    log: list[str] = []
    failed = False

    if not parquet_path.exists():
        _fail(log, f"parquet not found: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path)
    n = len(df)

    # Validate required columns and embedding dimension.
    need_meta = ["ageb_id", "scenario", "polygon_area_m2"]
    for c in need_meta:
        if c not in df.columns:
            _fail(log, f"missing required column: {c}")
            failed = True
            
    emb_cols = _emb_columns(df)
    if len(emb_cols) != fcfg.EMBEDDING_DIM:
        _fail(log, f"expected {fcfg.EMBEDDING_DIM} emb_* columns, found {len(emb_cols)}")
        failed = True
    else:
        _ok(log, "768 embedding columns present")

    if failed:
        return 1

    # Check for duplicate IDs and valid scenario labels.
    if df["ageb_id"].duplicated().any():
        _fail(log, f"duplicate ageb_id count: {int(df['ageb_id'].duplicated().sum())}")
        failed = True
    else:
        _ok(log, "ageb_id is unique")

    allowed = {"A", "A_missing", "B"}
    bad_scen = set(df["scenario"].unique()) - allowed
    if bad_scen:
        _fail(log, f"unexpected scenario value(s): {bad_scen}")
        failed = True
    else:
        _ok(log, "scenario values are valid")

    # Validate area metrics and embedding finiteness.
    if not np.isfinite(df["polygon_area_m2"]).all() or (df["polygon_area_m2"] <= 0).any():
        _fail(log, "polygon_area_m2 is non-finite or non-positive")
        failed = True
    else:
        _ok(log, "polygon_area_m2 is valid")

    X = df[emb_cols].to_numpy(dtype=np.float64, copy=False)
    row_finite = np.isfinite(X).all(axis=1)
    row_all_nan = np.isnan(X).all(axis=1)
    missing_scen = df["scenario"] == "A_missing"

    if (missing_scen & row_finite).any():
        _fail(log, "A_missing rows have finite embeddings")
        failed = True
    
    if missing_scen.any() and (missing_scen & ~row_all_nan).any():
        _fail(log, "A_missing rows are not all-NaN")
        failed = True

    filled = row_finite & ~missing_scen
    if filled.any():
        row_l2 = np.linalg.norm(X[filled], axis=1)
        if (row_l2 == 0).any():
            _fail(log, "finite embeddings have zero L2 norm")
            failed = True
        else:
            _ok(log, "finite embeddings are non-zero")

    # Ensure critical passthrough columns are numeric.
    numeric_passthrough = {"area_sqkm", "pop_dens", "log_popden", "POBTOT", "log_POBTOT"}
    for col in fcfg.POLYGON_PASSTHROUGH_COLUMNS:
        if col in df.columns and col in numeric_passthrough:
            if not pd.api.types.is_numeric_dtype(df[col]):
                _fail(log, f"'{col}' is not numeric; rollup will fail.")
                failed = True

    # Optional cross-check with source shapefile row count.
    if not args.no_shapefile:
        try:
            import geopandas as gpd
            if shapefile_path.exists():
                shp_n = len(gpd.read_file(shapefile_path))
                if shp_n != n:
                    _fail(log, f"row count mismatch: parquet={n}, shapefile={shp_n}")
                    failed = True
                else:
                    _ok(log, "row count matches shapefile")
        except ImportError:
            pass

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
