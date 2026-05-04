"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Aggregates 240m CLAY cell embeddings directly to dissolved municipio polygons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as e:
    raise ImportError(
        "aggregate_to_municipios_direct requires geopandas."
    ) from e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.pipeline.aggregate_to_polygons import (
    _load_cells_and_embeddings,
    aggregate,
)
from embedding_extract.core.grid_utils import read_polygons


ID_WIDTH = 5  # state (2) + municipio (3)


def _safe_log(x: pd.Series | np.ndarray) -> np.ndarray:
    """Computes log1p(x) with defensive checks.

    Args:
        x: Input data to log-transform.

    Returns:
        Log-transformed array with NaNs for non-finite or negative inputs.
    """
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr >= 0)
    out[mask] = np.log1p(arr[mask])
    return out


def dissolve_to_municipios(
    agebs: "gpd.GeoDataFrame",
    *,
    id_width: int = ID_WIDTH,
) -> "gpd.GeoDataFrame":
    """Dissolves AGEB layer into municipio polygons.

    Args:
        agebs: GeoDataFrame containing AGEB polygons and population data.
        id_width: Number of characters from CVEGEO to use for grouping.

    Returns:
        GeoDataFrame with dissolved geometries and recomputed population metrics.
    """
    if "ageb_id" not in agebs.columns:
        raise KeyError("expected 'ageb_id' column (from read_polygons)")

    muni_id = agebs["ageb_id"].astype(str).str.slice(0, id_width)
    if (muni_id.str.len() != id_width).any():
        bad = int((muni_id.str.len() != id_width).sum())
        raise RuntimeError(
            f"{bad} ageb_ids shorter than {id_width} chars; refusing to dissolve."
        )

    work = agebs.copy()
    work["_muni"] = muni_id

    # Ensure numeric types for population columns to avoid concatenation during dissolve.
    for col in ("POBTOT",):
        if col in work.columns and not pd.api.types.is_numeric_dtype(work[col]):
            coerced = pd.to_numeric(work[col], errors="coerce")
            work[col] = coerced.astype("float64")

    # Dissolve geometry and aggregate population counts.
    agg_cols: dict[str, str] = {}
    if "POBTOT" in work.columns:
        agg_cols["POBTOT"] = "sum"
    work["_one"] = 1
    agg_cols["_one"] = "sum"

    print(f"Dissolving {len(work):,} AGEBs into municipios...")
    muni = work.dissolve(by="_muni", aggfunc=agg_cols, as_index=False)
    muni = muni.rename(columns={"_muni": "ageb_id", "_one": "n_agebs"})

    # Recompute areal quantities from authoritative dissolved geometry.
    muni["polygon_area_m2"] = muni.geometry.area.to_numpy()
    muni["area_sqkm"] = muni["polygon_area_m2"] / 1e6

    if "POBTOT" in muni.columns:
        pop = muni["POBTOT"].astype(float).to_numpy()
        area_km = muni["area_sqkm"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            muni["pop_dens"] = np.where(area_km > 0, pop / area_km, np.nan)
        muni["log_popden"] = _safe_log(muni["pop_dens"])
        muni["log_POBTOT"] = _safe_log(pop)

    muni["state"] = muni["ageb_id"].str.slice(0, 2)
    muni = muni.reset_index(drop=True)
    muni["_row"] = np.arange(len(muni), dtype=np.int64)

    return muni


def main() -> int:
    """Main execution entry point for direct municipio aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate embeddings to municipios.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path.")
    parser.add_argument("--id-width", type=int, default=ID_WIDTH, help="CVEGEO prefix width.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    output_path = args.output or ds.municipio_features_direct_path

    ds.out_dir.mkdir(parents=True, exist_ok=True)
    agebs = read_polygons(ds.shapefile_path)
    munis = dissolve_to_municipios(agebs, id_width=args.id_width)

    cells, embeddings = _load_cells_and_embeddings(ds.grid_path, ds.embeddings_npy_path)
    if cells.crs != munis.crs:
        cells = cells.to_crs(munis.crs)

    # Perform spatial aggregation of cell embeddings to municipio polygons.
    out = aggregate(
        cells=cells,
        embeddings=embeddings,
        polygons=munis,
        cell_size_m=fcfg.CELL_SIZE_M,
        passthrough_columns=(),
    )

    # Re-insert labels and authoritative area metrics.
    label_cols = ["state", "n_agebs", "area_sqkm", "POBTOT", "pop_dens", "log_popden", "log_POBTOT"]
    label_cols = [c for c in label_cols if c in munis.columns]

    for i, col in enumerate(label_cols):
        out.insert(3 + i, col, munis[col].to_numpy())

    out["polygon_area_m2"] = munis["polygon_area_m2"].to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Wrote {output_path} ({len(out):,} municipios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
