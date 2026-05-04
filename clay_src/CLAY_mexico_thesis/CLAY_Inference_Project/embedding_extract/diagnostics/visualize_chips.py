"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Samples random CLAY input chunks from GEE, runs the encoder, and saves PNG diagnostic visualizations of inputs and embeddings.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from embedding_extract import config as fcfg
from embedding_extract.core.clay_utils import (
    build_datacube,
    get_waves,
    initialize_earth_engine,
    load_clay_encoder,
    load_mean_std,
)
from embedding_extract.core.grid_utils import chunks_geodataframe, read_polygons
from embedding_extract.pipeline.extract_embeddings import (
    _download_patch,
    _encode_chunk,
    _get_inputs_image,
    _maybe_normalize,
)


def _percentile_stretch(
    gray: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """Applies a percentile-based contrast stretch to a grayscale image.

    Args:
        gray: Input grayscale array.
        p_lo: Lower percentile for clipping.
        p_hi: Upper percentile for clipping.

    Returns:
        Stretched array normalized to [0, 1].
    """
    x = np.asarray(gray, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(gray, dtype=np.float32)

    lo, hi = np.percentile(x[finite], (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _rgb_from_patch(pixels_hwc: np.ndarray) -> np.ndarray:
    """Extracts and stretches RGB bands for display.

    Args:
        pixels_hwc: H x W x C pixel array.

    Returns:
        Stretched RGB array.
    """
    red = _percentile_stretch(pixels_hwc[:, :, 2])
    green = _percentile_stretch(pixels_hwc[:, :, 1])
    blue = _percentile_stretch(pixels_hwc[:, :, 0])
    return np.stack([red, green, blue], axis=-1)


def _save_chip_diagnostic_pngs(
    *,
    chip_dir: Path,
    pixels_hwc: np.ndarray,
    tokens_hw_d: np.ndarray,
    band_keys: list[str],
    band_labels: list[str],
    suptitle: str,
    dpi: int = 120,
) -> list[Path]:
    """Generates and saves diagnostic PNGs for a single chip.

    Args:
        chip_dir: Destination directory for PNGs.
        pixels_hwc: Input pixel array.
        tokens_hw_d: Encoded token embeddings.
        band_keys: List of band identifiers.
        band_labels: Display labels for bands.
        suptitle: Title for the plots.
        dpi: Resolution of saved images.

    Returns:
        List of paths to written PNG files.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    expected_shape = (fcfg.EMBED_GRID, fcfg.EMBED_GRID, fcfg.EMBEDDING_DIM)
    if tokens_hw_d.shape != expected_shape:
        raise ValueError(f"tokens_hw_d shape mismatch: {tokens_hw_d.shape}")

    inputs_dir = chip_dir / "inputs"
    embed_dir = chip_dir / "embeddings"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    token_norms = np.linalg.norm(tokens_hw_d, axis=-1)
    flat_tokens = tokens_hw_d.reshape(-1, fcfg.EMBEDDING_DIM)
    pca_xy = PCA(n_components=2, random_state=0).fit_transform(flat_tokens)
    token_index = np.arange(flat_tokens.shape[0], dtype=np.float32)

    written: list[Path] = []

    def stamp(fig) -> None:
        fig.suptitle(suptitle, fontsize=8, y=1.02)

    # Save RGB composite.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_rgb_from_patch(pixels_hwc), interpolation="nearest")
    ax.set_title("RGB composite")
    ax.axis("off")
    stamp(fig)
    path = inputs_dir / "rgb_composite.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    written.append(path)

    # Save per-band mosaic.
    thumbs = [_percentile_stretch(pixels_hwc[:, :, i]) for i in range(pixels_hwc.shape[-1])]
    band_mosaic = np.concatenate(thumbs, axis=1)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(band_mosaic, cmap="gray", interpolation="nearest", aspect="auto")
    ax.set_title("Per-band mosaic")
    ax.set_yticks([])
    ax.set_xticks([pixels_hwc.shape[1] * (i + 0.5) for i in range(pixels_hwc.shape[-1])], band_labels, fontsize=8)
    stamp(fig)
    path = inputs_dir / "bands_mosaic.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    written.append(path)

    # Save individual band images.
    for i, key in enumerate(band_keys):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(_percentile_stretch(pixels_hwc[:, :, i]), cmap="gray", interpolation="nearest")
        ax.set_title(f"{key} ({fcfg.BANDS[key]})")
        ax.axis("off")
        stamp(fig)
        path = inputs_dir / f"band_{key.replace('/', '_')}.png"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written.append(path)

    # Save token norm heatmap.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(token_norms, cmap="viridis", interpolation="nearest")
    ax.set_title("Token L2 norm (16x16)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    stamp(fig)
    path = embed_dir / "token_l2_norm.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    written.append(path)

    # Save PCA scatter plot.
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(pca_xy[:, 0], pca_xy[:, 1], c=token_index, cmap="turbo", s=10, alpha=0.85)
    ax.set_title("PCA of token vectors")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="token index")
    stamp(fig)
    path = embed_dir / "pca_tokens.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    written.append(path)

    # Save token norm distribution histogram.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_norms.ravel(), bins=40, color="steelblue", edgecolor="white")
    ax.set_title("Token L2 norms distribution")
    stamp(fig)
    path = embed_dir / "token_l2_norm_histogram.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    written.append(path)

    return written


def _sample_chunk_indices(n_chunks: int, n_chips: int, seed: int) -> list[int]:
    """Randomly samples chunk indices for visualization."""
    n = min(n_chips, n_chunks)
    return sorted(random.Random(seed).sample(range(n_chunks), n))


def main() -> None:
    """Main execution entry point for chip visualization."""
    parser = argparse.ArgumentParser(description="Visualize CLAY chips.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--n-chips", type=int, default=4, help="Number of chips to visualize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output root directory.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    out_root = (args.output_dir or (ds.out_dir / "viz_chips")).resolve()

    polygons_m = read_polygons(ds.shapefile_path)
    chunks_m = chunks_geodataframe(polygons_m, filter_by_polygons=fcfg.FILTER_BY_AGEBS)
    picked_indices = _sample_chunk_indices(len(chunks_m), args.n_chips, args.seed)

    initialize_earth_engine()
    image, date = _get_inputs_image(ds.gee_asset_id, ds.study_year)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, metadata = load_clay_encoder(device)
    waves = np.asarray(get_waves(fcfg.BANDS, fcfg.PLATFORM, metadata), dtype=np.float32)
    gsd = float(metadata[fcfg.PLATFORM].gsd)
    mean_std = load_mean_std(ds.norm_stats_path)
    
    centroids_ll = chunks_m.geometry.centroid.to_crs("EPSG:4326")
    bounds_m = chunks_m.bounds

    for k, pos in enumerate(picked_indices, start=1):
        chunk_id = int(chunks_m.iloc[pos]["chunk_id"])
        lonlat = (float(centroids_ll.iloc[pos].x), float(centroids_ll.iloc[pos].y))
        b = bounds_m.iloc[pos]
        
        try:
            pixels = _download_patch(image, (float(b["minx"]), float(b["miny"]), float(b["maxx"]), float(b["maxy"])),
                                     list(fcfg.BANDS.keys()), fcfg.PATCH_PIXELS, fcfg.WORKING_CRS)
        except Exception as e:
            print(f"Download failed for chunk {chunk_id}: {e}")
            continue

        pixels_norm = _maybe_normalize(pixels, mean_std)
        datacube = build_datacube(pixels_norm, lonlat, date, waves, gsd, fcfg.PLATFORM, device)
        tokens = _encode_chunk(encoder, datacube)
        tokens_hw = tokens.reshape(fcfg.EMBED_GRID, fcfg.EMBED_GRID, -1).astype(np.float32)

        suptitle = f"chunk_id={chunk_id} | lon={lonlat[0]:.5f}, lat={lonlat[1]:.5f}"
        _save_chip_diagnostic_pngs(chip_dir=out_root / f"chip_{chunk_id:06d}", pixels_hwc=pixels,
                                   tokens_hw_d=tokens_hw, band_keys=list(fcfg.BANDS.keys()),
                                   band_labels=[f"{k}\n{fcfg.BANDS[k]}" for k in fcfg.BANDS], suptitle=suptitle)

    print(f"Visualization complete: {out_root}")


if __name__ == "__main__":
    main()
