"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Aggregates 240m CLAY embeddings to AGEB polygons using centroid lookup (Scenario A) or spatial averaging (Scenario B).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError as e:
    raise ImportError("aggregate_to_polygons requires geopandas and shapely.") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.core.grid_utils import read_polygons


def _load_cells_and_embeddings(
    cells_path: Path, emb_path: Path
) -> tuple["gpd.GeoDataFrame", np.ndarray]:
    """Loads grid cells and memory-mapped embeddings.

    Args:
        cells_path: Path to cells parquet or shapefile.
        emb_path: Path to npy embeddings.

    Returns:
        Tuple of (cells GeoDataFrame, embeddings ndarray).
    """
    cells = gpd.read_parquet(cells_path) if cells_path.suffix == ".parquet" else gpd.read_file(cells_path)
    embeddings = np.load(emb_path, mmap_mode="r")
    if len(cells) != embeddings.shape[0]:
        raise ValueError(f"Row mismatch: cells ({len(cells)}) vs embeddings ({embeddings.shape[0]}).")
    return cells, embeddings


def _cell_size_area(cell_size_m: float) -> float:
    """Computes square area of a cell."""
    return float(cell_size_m) * float(cell_size_m)


def _centroid_lookup_vector(
    centroid: Point,
    cells_tree: "gpd.GeoDataFrame",
    embeddings: np.ndarray,
    cell_size_m: float,
) -> np.ndarray | None:
    """Finds the embedding for the cell containing the polygon centroid.

    Args:
        centroid: Representative point of the polygon.
        cells_tree: GeoDataFrame of grid cells.
        embeddings: Array of cell embeddings.
        cell_size_m: Edge length of a cell in meters.

    Returns:
        Embedding vector or None if no cell matches.
    """
    idx = cells_tree.sindex.query(centroid, predicate="within")
    if len(idx) == 0:
        idx = cells_tree.sindex.query(centroid, predicate="intersects")
    if len(idx) == 0:
        nearest = cells_tree.sindex.nearest(centroid, return_all=False)
        if nearest is None or len(nearest[0]) == 0:
            return None
        return embeddings[int(nearest[1][0])]
    return embeddings[int(idx[0])]


def aggregate(
    cells: "gpd.GeoDataFrame",
    embeddings: np.ndarray,
    polygons: "gpd.GeoDataFrame",
    *,
    cell_size_m: float,
    passthrough_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Performs spatial aggregation of cell embeddings to polygons.

    Args:
        cells: GeoDataFrame of 240m grid cells.
        embeddings: Memory-mapped array of embeddings.
        polygons: GeoDataFrame of target polygons (AGEBs).
        cell_size_m: Cell size in meters.
        passthrough_columns: Attributes to copy from polygons to output.

    Returns:
        DataFrame containing polygon IDs, scenario labels, and aggregated embeddings.
    """
    assert cells.crs == polygons.crs, "CRS mismatch between cells and polygons."
    cells = cells.reset_index(drop=True)
    if "cell_id" not in cells.columns:
        cells["cell_id"] = np.arange(len(cells), dtype=np.int64)

    cell_centers = gpd.GeoDataFrame(
        {"cell_id": cells["cell_id"].values},
        geometry=gpd.points_from_xy(cells["cx_m"].values, cells["cy_m"].values),
        crs=cells.crs,
    )
    sjoin = gpd.sjoin(
        cell_centers,
        polygons[["ageb_id", "_row", "geometry"]],
        how="inner",
        predicate="within",
    )
    center_cells_by_poly: dict[object, np.ndarray] = {
        row_idx: grp["cell_id"].to_numpy() for row_idx, grp in sjoin.groupby("_row")
    }

    cell_area = _cell_size_area(cell_size_m)
    emb_dim = embeddings.shape[1]
    row_indices = polygons["_row"].to_numpy()
    geoms = polygons.geometry.values

    out_rows = []
    scenarios: list[str] = []
    for k in range(len(polygons)):
        row_idx = row_indices[k]
        geom = geoms[k]
        area = float(geom.area)
        cell_ids = center_cells_by_poly.get(row_idx, np.empty(0, dtype=np.int64))

        # Scenario A: Polygon is small or contains no cell centers; use centroid lookup.
        if area < cell_area or len(cell_ids) == 0:
            rep = geom.representative_point()
            vec = _centroid_lookup_vector(rep, cells, embeddings, cell_size_m)
            if vec is None:
                out_rows.append(np.full(emb_dim, np.nan, dtype=np.float32))
                scenarios.append("A_missing")
                continue
            out_rows.append(np.asarray(vec, dtype=np.float32))
            scenarios.append("A")
            continue

        # Scenario B: Polygon contains multiple cell centers; average their embeddings.
        vec = np.nanmean(embeddings[cell_ids].astype(np.float32), axis=0)
        out_rows.append(vec)
        scenarios.append("B")

    mat = np.vstack(out_rows) if out_rows else np.empty((0, emb_dim), dtype=np.float32)
    out = pd.DataFrame(mat, columns=[f"emb_{i}" for i in range(emb_dim)])

    out.insert(0, "ageb_id", polygons["ageb_id"].to_numpy())
    out.insert(1, "scenario", scenarios)
    out.insert(2, "polygon_area_m2", polygons.geometry.area.to_numpy())

    # Passthrough attributes with numeric coercion for DBF safety.
    insert_at = 3
    for col in passthrough_columns:
        if col not in polygons.columns or col in out.columns:
            continue
        values = polygons[col]
        if not pd.api.types.is_numeric_dtype(values):
            coerced = pd.to_numeric(values, errors="coerce")
            if coerced.notna().any():
                values = coerced
        
        if pd.api.types.is_numeric_dtype(values):
            out.insert(insert_at, col, values.to_numpy(dtype="float64", na_value=np.nan))
        else:
            out.insert(insert_at, col, values.to_numpy())
        insert_at += 1
    return out


def main() -> None:
    """Main execution entry point for AGEB aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate embeddings to polygons.")
    fcfg.add_year_arg(parser)
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    ds.out_dir.mkdir(parents=True, exist_ok=True)
    
    polygons = read_polygons(ds.shapefile_path)
    cells, embeddings = _load_cells_and_embeddings(ds.grid_path, ds.embeddings_npy_path)

    if cells.crs != polygons.crs:
        cells = cells.to_crs(polygons.crs)

    out = aggregate(
        cells=cells,
        embeddings=embeddings,
        polygons=polygons,
        cell_size_m=fcfg.CELL_SIZE_M,
        passthrough_columns=tuple(fcfg.POLYGON_PASSTHROUGH_COLUMNS),
    )
    out.to_parquet(ds.polygon_features_path, index=False)
    print(f"Wrote {ds.polygon_features_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
