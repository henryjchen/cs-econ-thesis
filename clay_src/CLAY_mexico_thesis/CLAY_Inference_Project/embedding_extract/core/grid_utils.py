"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, pandas
Data Dependencies: Clay model checkpoints, INEGI census CSVs, GeoTIFF imagery
Description: Grid management utilities for tiling Mexico with CLAY input patches.
             Handles coordinate transformations, chunking, and cell-level grid generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, box
except ImportError as e:
    raise ImportError(
        "grid_utils requires geopandas and shapely. "
        "Install with: pip install geopandas shapely pyproj"
    ) from e

from embedding_extract import config as fcfg


@dataclass
class ChunkSpec:
    """Specifies a CLAY input patch (16x16 grid of embedding cells).

    Attributes:
        chunk_id: Unique identifier for the chunk.
        ix: Column index in the global meter grid.
        iy: Row index in the global meter grid.
        x_min, y_min, x_max, y_max: Bounding box coordinates in WORKING_CRS.
    """
    chunk_id: int
    ix: int
    iy: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center_m(self) -> Tuple[float, float]:
        """Returns (x, y) center of the chunk in meters."""
        return 0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max)


def read_polygons(shapefile_path=None) -> "gpd.GeoDataFrame":
    """Loads AGEB polygons and prepares them for processing.

    Args:
        shapefile_path: Path to .shp file. Defaults to config value.

    Returns:
        gpd.GeoDataFrame: Polygons in WORKING_CRS with unique ageb_id.
    """
    shp = shapefile_path or fcfg.SHAPEFILE_PATH
    gdf = gpd.read_file(shp)
    id_col = _pick_id_column(gdf)
    gdf = gdf.rename(columns={id_col: "ageb_id"}) if id_col != "ageb_id" else gdf
    if "ageb_id" not in gdf.columns:
        gdf["ageb_id"] = gdf.index.astype(str)
    gdf["_row"] = np.arange(len(gdf), dtype=np.int64)
    gdf = gdf.to_crs(fcfg.WORKING_CRS)
    gdf["geometry"] = gdf["geometry"].buffer(0)  # Repair invalid rings
    return gdf


def _pick_id_column(gdf: "gpd.GeoDataFrame") -> str:
    """Identifies the best candidate for an ID column."""
    for c in fcfg.AGEB_ID_CANDIDATES:
        if c in gdf.columns:
            return c
    return gdf.columns[0]


def aligned_grid_origin(x_min: float, y_min: float, step: float) -> Tuple[float, float]:
    """Snaps origin to integer multiples of step for reproducibility."""
    return np.floor(x_min / step) * step, np.floor(y_min / step) * step


def iter_chunks(bounds_m: Tuple[float, float, float, float]) -> Iterator[ChunkSpec]:
    """Yields chunks that tile a given bounding box.

    Args:
        bounds_m: (minx, miny, maxx, maxy) in meters.

    Yields:
        ChunkSpec: Specification for each tiling chunk.
    """
    x_min, y_min, x_max, y_max = bounds_m
    chunk_size = fcfg.CELL_SIZE_M * fcfg.EMBED_GRID
    x0, y0 = aligned_grid_origin(x_min, y_min, chunk_size)
    nx = int(np.ceil((x_max - x0) / chunk_size))
    ny = int(np.ceil((y_max - y0) / chunk_size))
    cid = 0
    for iy in range(ny):
        for ix in range(nx):
            cx_min = x0 + ix * chunk_size
            cy_min = y0 + iy * chunk_size
            yield ChunkSpec(
                chunk_id=cid,
                ix=ix,
                iy=iy,
                x_min=cx_min,
                y_min=cy_min,
                x_max=cx_min + chunk_size,
                y_max=cy_min + chunk_size,
            )
            cid += 1


def chunks_geodataframe(
    polygons_m: "gpd.GeoDataFrame",
    *,
    filter_by_polygons: bool = True,
) -> "gpd.GeoDataFrame":
    """Creates a GeoDataFrame of chunks covering the polygons.

    Args:
        polygons_m: Input polygons in meters.
        filter_by_polygons: If True, keep only chunks intersecting polygons.

    Returns:
        gpd.GeoDataFrame: Grid of chunks.
    """
    bounds = polygons_m.total_bounds
    rows = list(iter_chunks(tuple(bounds)))
    chunk_gdf = gpd.GeoDataFrame(
        {
            "chunk_id": [c.chunk_id for c in rows],
            "ix": [c.ix for c in rows],
            "iy": [c.iy for c in rows],
        },
        geometry=[box(c.x_min, c.y_min, c.x_max, c.y_max) for c in rows],
        crs=polygons_m.crs,
    )
    if filter_by_polygons:
        mask = chunk_gdf.sindex.query(polygons_m.geometry.union_all(), predicate="intersects")
        chunk_gdf = chunk_gdf.iloc[mask].sort_values("chunk_id").reset_index(drop=True)
    return chunk_gdf


def cells_for_chunk(chunk: ChunkSpec, *, crs: str) -> "gpd.GeoDataFrame":
    """Expands a single chunk into a grid of embedding cells.

    Args:
        chunk: Chunk specification.
        crs: Coordinate reference system.

    Returns:
        gpd.GeoDataFrame: Individual cell squares.
    """
    step = fcfg.CELL_SIZE_M
    n = fcfg.EMBED_GRID
    xs = chunk.x_min + np.arange(n) * step
    ys = chunk.y_min + np.arange(n) * step
    geoms, ixs, iys = [], [], []
    for iy in range(n):
        for ix in range(n):
            x_min, y_min = xs[ix], ys[iy]
            geoms.append(box(x_min, y_min, x_min + step, y_min + step))
            ixs.append(ix)
            iys.append(iy)
    return gpd.GeoDataFrame(
        {
            "chunk_id": chunk.chunk_id,
            "cell_ix_in_chunk": ixs,
            "cell_iy_in_chunk": iys,
        },
        geometry=geoms,
        crs=crs,
    )


def build_cell_grid(polygons_m: "gpd.GeoDataFrame", chunks_m: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Builds a complete cell-level grid from chunk GeoDataFrame.

    Args:
        polygons_m: Input polygons for CRS reference.
        chunks_m: Chunk-level GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Full cell grid with stable IDs and centroids.
    """
    frames = []
    for row in chunks_m.itertuples(index=False):
        c = ChunkSpec(
            chunk_id=int(row.chunk_id),
            ix=int(row.ix),
            iy=int(row.iy),
            x_min=float(row.geometry.bounds[0]),
            y_min=float(row.geometry.bounds[1]),
            x_max=float(row.geometry.bounds[2]),
            y_max=float(row.geometry.bounds[3]),
        )
        frames.append(cells_for_chunk(c, crs=polygons_m.crs))
    
    cells = pd.concat(frames, ignore_index=True)
    cells = gpd.GeoDataFrame(cells, geometry="geometry", crs=polygons_m.crs)
    cells["cell_id"] = np.arange(len(cells), dtype=np.int64)
    cells["cx_m"] = cells.geometry.centroid.x
    cells["cy_m"] = cells.geometry.centroid.y
    return cells[["cell_id", "chunk_id", "cell_ix_in_chunk", "cell_iy_in_chunk", "cx_m", "cy_m", "geometry"]]
