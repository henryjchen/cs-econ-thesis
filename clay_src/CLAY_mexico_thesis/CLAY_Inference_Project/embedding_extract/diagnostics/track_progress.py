"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Tracks the progress of feature extraction by reporting cell and chunk fill rates and generating a spatial heatmap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg


def _resolve_npy(explicit: Path | None, default_path: Path) -> Path:
    """Resolves the path to the embeddings npy file, including fallback to .tmp files.

    Args:
        explicit: User-provided path, if any.
        default_path: Default configured path.

    Returns:
        Path to the best available embeddings file.
    """
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    if default_path.exists():
        return default_path
    
    stray = default_path.with_name(default_path.name + ".tmp.npy")
    if stray.exists():
        return stray
    raise FileNotFoundError(f"No embeddings file found at {default_path} or {stray}.")


def _fmt_pct(num: int, denom: int) -> str:
    """Formats a fraction as a count and percentage string."""
    return f"{num:,} / {denom:,} ({100.0 * num / max(denom, 1):.2f}%)"


def _overall_metrics(filled: np.ndarray) -> None:
    """Prints overall cell-level fill metrics."""
    n = filled.size
    n_filled = int(filled.sum())
    print("\n=== Cell-level fill ===")
    print(f"  filled cells: {_fmt_pct(n_filled, n)}")
    print(f"  remaining   : {_fmt_pct(n - n_filled, n)}")


def _per_chunk_metrics(cells: pd.DataFrame, filled: np.ndarray) -> pd.DataFrame:
    """Computes and prints metrics aggregated by grid chunk.

    Args:
        cells: Grid cell metadata DataFrame.
        filled: Boolean array indicating finite embeddings.

    Returns:
        DataFrame containing per-chunk statistics.
    """
    df = pd.DataFrame({"chunk_id": cells["chunk_id"].to_numpy(), "filled": filled})
    per_chunk = df.groupby("chunk_id")["filled"].agg(["sum", "size"])
    per_chunk.columns = ["filled", "total"]
    per_chunk["frac"] = per_chunk["filled"] / per_chunk["total"]

    n_chunks = len(per_chunk)
    n_done = int((per_chunk["frac"] >= 1.0).sum())
    n_none = int((per_chunk["frac"] <= 0.0).sum())
    n_partial = n_chunks - n_done - n_none

    print("\n=== Chunk-level fill ===")
    print(f"  total chunks    : {n_chunks:,}")
    print(f"  fully done      : {_fmt_pct(n_done, n_chunks)}")
    print(f"  partially done  : {_fmt_pct(n_partial, n_chunks)}")
    print(f"  untouched       : {_fmt_pct(n_none, n_chunks)}")
    print(f"  fill-frac mean  : {per_chunk['frac'].mean():.4f}")
    return per_chunk


def _save_heatmap(
    cells: pd.DataFrame,
    filled: np.ndarray,
    out_path: Path,
    bins: int = 300,
) -> None:
    """Generates and saves a spatial fill-fraction heatmap.

    Args:
        cells: Grid cell metadata DataFrame.
        filled: Boolean array indicating finite embeddings.
        out_path: Destination PNG path.
        bins: Number of bins per axis for the histogram.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cx, cy = cells["cx_m"].values, cells["cy_m"].values
    filled_mask = filled.astype(np.float32)
    
    total_h, xedges, yedges = np.histogram2d(cx, cy, bins=bins)
    filled_h, _, _ = np.histogram2d(cx, cy, bins=[xedges, yedges], weights=filled_mask)
    
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total_h > 0, filled_h / total_h, np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(frac.T, origin="lower", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                   vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    
    ax.set_title(f"CLAY embedding fill-fraction over Mexico ({filled_mask.sum():,.0f} / {len(cells):,} cells finite)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fraction of cells with finite embedding")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Main execution entry point for tracking extraction progress."""
    parser = argparse.ArgumentParser(description="Track extraction progress.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--embeddings", type=Path, default=None, help="Path to embeddings.npy.")
    parser.add_argument("--grid", type=Path, default=None, help="Path to grid parquet.")
    parser.add_argument("--heatmap", type=Path, default=None, help="Output heatmap path.")
    parser.add_argument("--bins", type=int, default=300, help="Heatmap resolution.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    grid_path = args.grid or ds.grid_path
    heatmap_path = args.heatmap or ds.progress_heatmap_path
    npy_path = _resolve_npy(args.embeddings, ds.embeddings_npy_path)

    cells = pd.read_parquet(grid_path)
    embeddings = np.load(npy_path, mmap_mode="r")
    
    # Handle possible row count mismatch between grid and embeddings.
    if embeddings.shape[0] != len(cells):
        n = min(embeddings.shape[0], len(cells))
        cells, embeddings = cells.iloc[:n], embeddings[:n]

    filled = np.isfinite(embeddings).all(axis=1)

    _overall_metrics(filled)
    _per_chunk_metrics(cells, filled)

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    _save_heatmap(cells, filled, heatmap_path, bins=args.bins)


if __name__ == "__main__":
    main()
