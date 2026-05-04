"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Extract 240m CLAY embeddings for the AGEB shapefile by downloading patches from Earth Engine and running the CLAY encoder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import io
import random
import sys
import time
from pathlib import Path
import pyarrow.parquet as pq

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.core.clay_utils import (
    build_datacube,
    get_waves,
    initialize_earth_engine,
    load_clay_encoder,
    load_mean_std,
)
from embedding_extract.core.grid_utils import (
    build_cell_grid,
    chunks_geodataframe,
    read_polygons,
)


def _get_inputs_image(asset_id: str, fallback_year: int):
    """Retrieves Earth Engine image and its start date.

    Args:
        asset_id: GEE asset identifier.
        fallback_year: Year to use if image lacks a timestamp.

    Returns:
        Tuple of (ee.Image, datetime.datetime).
    """
    import ee

    image = ee.Image(asset_id).unmask(0)
    ts = image.get("system:time_start").getInfo()
    date = (
        dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
        if ts
        else dt.datetime(fallback_year, 1, 1, tzinfo=dt.timezone.utc)
    )
    return image, date


def _download_patch(
    image,
    bounds_m: tuple[float, float, float, float],
    bands: list[str],
    patch_pixels: int,
    working_crs: str,
    max_retries: int = 5,
) -> np.ndarray:
    """Downloads a resampled image patch from GEE.

    Args:
        image: ee.Image source.
        bounds_m: (x_min, y_min, x_max, y_max) in working_crs.
        bands: List of band names to download.
        patch_pixels: Dimension of the square patch in pixels.
        working_crs: Projection to use for sampling.
        max_retries: Maximum network retry attempts.

    Returns:
        H x W x C float32 numpy array.
    """
    import ee
    import requests
    from google.api_core import exceptions

    x_min, y_min, x_max, y_max = bounds_m
    region = ee.Geometry.Rectangle([x_min, y_min, x_max, y_max], proj=working_crs, geodesic=False)
    url = image.select(bands).getDownloadURL({
        "region": region,
        "dimensions": [patch_pixels, patch_pixels],
        "crs": working_crs,
        "format": "NPY",
    })
    
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=90)
            if response.status_code == 200:
                break
            if response.status_code in (429, 503):
                time.sleep((2 ** (attempt + 1)) + random.random())
                continue
            response.raise_for_status()
        except (requests.exceptions.RequestException, exceptions.GoogleAPICallError):
            if attempt == max_retries - 1:
                raise
            time.sleep((2 ** (attempt + 1)) + random.random())
            
    if response is None or response.status_code != 200:
        raise RuntimeError("Failed to download patch.")
    arr = np.load(io.BytesIO(response.content), allow_pickle=True)
    return np.stack([np.asarray(arr[b]) for b in bands], axis=-1).astype(np.float32)


def _maybe_normalize(pixels: np.ndarray, mean_std: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Applies z-score normalization if statistics are provided."""
    if mean_std is None:
        return pixels
    mean, std = mean_std
    return (pixels - mean) / std


def _encode_chunk(encoder, datacube: dict) -> np.ndarray:
    """Encodes a datacube into embedding tokens.

    Args:
        encoder: Loaded CLAY encoder model.
        datacube: Torch-formatted input dictionary.

    Returns:
        Array of (EMBED_GRID^2, D) embeddings.
    """
    with torch.no_grad():
        enc_unmasked, _, _, _ = encoder(datacube)
    return enc_unmasked[:, 1:, :].detach().cpu().numpy()[0]


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Saves a numpy array to disk atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    tmp.replace(path)


def _load_existing_embeddings(path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Reloads existing embeddings if they match the expected shape.

    Args:
        path: Path to npy file.
        expected_shape: (N_cells, D) tuple.

    Returns:
        Numpy array (or memmap) or None if mismatch/missing.
    """
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r+")
    except Exception:
        return None
    if arr.shape != expected_shape or arr.dtype != np.float32:
        return None
    return arr


def main() -> None:
    """Main execution loop for CLAY embedding extraction."""
    parser = argparse.ArgumentParser(description="Extract CLAY embeddings.")
    fcfg.add_year_arg(parser)
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    ds.out_dir.mkdir(parents=True, exist_ok=True)
    
    polygons_m = read_polygons(ds.shapefile_path)
    chunks_m = chunks_geodataframe(polygons_m, filter_by_polygons=fcfg.FILTER_BY_AGEBS)

    if Path(ds.grid_path).exists():
        cells_m = pq.read_table(ds.grid_path).to_pandas()
    else:
        cells_m = build_cell_grid(polygons_m, chunks_m)
        cells_m.to_parquet(ds.grid_path)

    initialize_earth_engine()
    image, date = _get_inputs_image(ds.gee_asset_id, ds.study_year)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, metadata = load_clay_encoder(device)
    waves = np.asarray(get_waves(fcfg.BANDS, fcfg.PLATFORM, metadata), dtype=np.float32)
    gsd = float(metadata[fcfg.PLATFORM].gsd)
    mean_std = load_mean_std(ds.norm_stats_path)
    
    centroids_ll = chunks_m.geometry.centroid.to_crs("EPSG:4326")
    cells_m_by_chunk: dict[int, list[int]] = {}
    chunk_ids_cells = np.asarray(cells_m["chunk_id"], dtype=np.int64)
    for i, cid in enumerate(chunk_ids_cells):
        cells_m_by_chunk.setdefault(int(cid), []).append(i)

    n_chunks = len(chunks_m)
    chunk_ids = np.asarray(chunks_m["chunk_id"], dtype=np.int64)
    b = chunks_m.geometry.bounds
    x_min, y_min, x_max, y_max = b["minx"].values, b["miny"].values, b["maxx"].values, b["maxy"].values
    lon, lat = centroids_ll.x.values, centroids_ll.y.values

    del polygons_m, chunks_m, centroids_ll
    gc.collect()

    expected_shape = (len(cells_m), fcfg.EMBEDDING_DIM)
    embeddings = _load_existing_embeddings(ds.embeddings_npy_path, expected_shape)
    if embeddings is None:
        embeddings = np.lib.format.open_memmap(ds.embeddings_npy_path, mode="w+",
                                               dtype=np.float32, shape=expected_shape)
        embeddings[:] = np.nan
        if hasattr(embeddings, "flush"):
            embeddings.flush()

    CHECKPOINT_EVERY = 50
    for j in range(n_chunks):
        cid = int(chunk_ids[j])
        cell_idxs = cells_m_by_chunk.get(cid, [])

        if cell_idxs and np.isfinite(embeddings[cell_idxs]).all():
            continue

        try:
            pixels = _download_patch(image, (x_min[j], y_min[j], x_max[j], y_max[j]),
                                     list(fcfg.BANDS.keys()), fcfg.PATCH_PIXELS, fcfg.WORKING_CRS)
        except Exception as e:
            print(f"[chunk {cid}] download failed: {e}")
            continue

        if pixels.shape != (fcfg.PATCH_PIXELS, fcfg.PATCH_PIXELS, len(fcfg.BANDS)):
            continue

        pixels = _maybe_normalize(pixels, mean_std)
        datacube = build_datacube(pixels, (float(lon[j]), float(lat[j])), date, waves, gsd, fcfg.PLATFORM, device)
        tokens = _encode_chunk(encoder, datacube)
        
        tokens = tokens.reshape(fcfg.EMBED_GRID, fcfg.EMBED_GRID, -1)
        ix_arr, iy_arr = cells_m["cell_ix_in_chunk"].values, cells_m["cell_iy_in_chunk"].values
        cell_id_arr = cells_m["cell_id"].values
        for idx in cell_idxs:
            embeddings[int(cell_id_arr[idx])] = tokens[fcfg.EMBED_GRID - 1 - int(iy_arr[idx]), int(ix_arr[idx])]

        if (j + 1) % CHECKPOINT_EVERY == 0 and hasattr(embeddings, "flush"):
            embeddings.flush()

    if hasattr(embeddings, "flush"):
        embeddings.flush()
    print(f"Extraction complete: {ds.embeddings_npy_path}")


if __name__ == "__main__":
    main()
