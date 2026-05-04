"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Computes per-band Landsat normalization statistics from GEE mosaics for CLAY feature extraction.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.core.clay_utils import initialize_earth_engine
from embedding_extract.core.grid_utils import chunks_geodataframe, read_polygons
from embedding_extract.pipeline.extract_embeddings import _download_patch, _get_inputs_image


def _update_stats(
    pixels_hwc: np.ndarray,
    *,
    sum_bands: np.ndarray,
    sumsq_bands: np.ndarray,
    count_bands: np.ndarray,
) -> None:
    """Updates running sums and squares for mean/std computation.

    Args:
        pixels_hwc: H x W x C pixel array.
        sum_bands: Running sum per band.
        sumsq_bands: Running sum of squares per band.
        count_bands: Running count of finite pixels per band.
    """
    arr = np.asarray(pixels_hwc, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    finite = np.isfinite(flat)

    vals = np.where(finite, flat, 0.0)
    sum_bands += vals.sum(axis=0)
    sumsq_bands += (vals * vals).sum(axis=0)
    count_bands += finite.sum(axis=0)


def _sample_indices(n_total: int, *, n_chunks: int, seed: int, all_chunks: bool) -> np.ndarray:
    """Generates random or full-pass indices for sampling chunks.

    Args:
        n_total: Total number of available chunks.
        n_chunks: Number of chunks to sample.
        seed: Random seed for reproducibility.
        all_chunks: Whether to return all indices.

    Returns:
        Array of sampled indices.
    """
    if all_chunks or n_chunks >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_chunks, replace=False)).astype(np.int64)


def main() -> int:
    """Main execution entry point for normalization statistics calculation."""
    parser = argparse.ArgumentParser(description="Compute normalization stats.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--n-chunks", type=int, default=2000, help="Number of chunks to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-chunks", action="store_true", help="Full pass over all chunks.")
    parser.add_argument("--out", type=Path, default=None, help="Output .pth path.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    out_path = args.out or ds.norm_stats_path or (ds.out_dir / "norm_stats.pth")

    polygons_m = read_polygons(ds.shapefile_path)
    chunks_m = chunks_geodataframe(polygons_m, filter_by_polygons=fcfg.FILTER_BY_AGEBS)
    sample_idx = _sample_indices(len(chunks_m), n_chunks=max(1, int(args.n_chunks)),
                                 seed=int(args.seed), all_chunks=bool(args.all_chunks))

    bounds = chunks_m.geometry.bounds
    x_min, y_min, x_max, y_max = bounds["minx"].values, bounds["miny"].values, bounds["maxx"].values, bounds["maxy"].values

    initialize_earth_engine()
    image, _ = _get_inputs_image(ds.gee_asset_id, ds.study_year)
    band_keys = list(fcfg.BANDS.keys())
    n_bands = len(band_keys)

    sum_bands, sumsq_bands, count_bands = [np.zeros(n_bands, dtype=np.float64) for _ in range(3)]

    ok = failed = 0
    for pos, j in enumerate(sample_idx, start=1):
        try:
            pixels = _download_patch(image, (x_min[j], y_min[j], x_max[j], y_max[j]),
                                     band_keys, fcfg.PATCH_PIXELS, fcfg.WORKING_CRS)
        except Exception as e:
            failed += 1
            print(f"[index {int(j)}] download failed: {e}")
            continue

        if pixels.shape != (fcfg.PATCH_PIXELS, fcfg.PATCH_PIXELS, n_bands):
            failed += 1
            continue

        _update_stats(pixels, sum_bands=sum_bands, sumsq_bands=sumsq_bands, count_bands=count_bands)
        ok += 1

    if ok == 0 or np.any(count_bands == 0):
        raise RuntimeError("No valid pixels downloaded.")

    mean = (sum_bands / count_bands).astype(np.float32)
    var = np.maximum((sumsq_bands / count_bands) - mean.astype(np.float64) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std, "bands": band_keys, "year": int(ds.year)}, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
