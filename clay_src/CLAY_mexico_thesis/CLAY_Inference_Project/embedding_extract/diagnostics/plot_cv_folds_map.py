"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Visualizes evaluation CV folds on a Mexico map using AGEB polygons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_extract import config as fcfg
from embedding_extract.core.grid_utils import read_polygons


def _kfold_assign(n: int, k: int, seed: int) -> np.ndarray:
    """Matches analysis.py's fold assignment logic exactly.

    Args:
        n: Total number of samples.
        k: Number of folds.
        seed: Random state for reproducibility.

    Returns:
        Array of fold assignments for each sample.
    """
    from sklearn.model_selection import KFold

    folds = np.empty(n, dtype=np.int32)
    kf = KFold(n_splits=int(k), shuffle=True, random_state=int(seed))
    for i, (_, te) in enumerate(kf.split(np.arange(n))):
        folds[te] = i
    return folds


def _save_fold_map(
    *,
    gdf_ll,
    out_path: Path,
    k: int,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generates and saves a choropleth map of all folds.

    Args:
        gdf_ll: GeoDataFrame in EPSG:4326 with 'fold' column.
        out_path: Destination PNG path.
        k: Number of folds.
        title: Plot title.
        xlim: Optional longitude limits.
        ylim: Optional latitude limits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    base = list(plt.get_cmap("tab20").colors)
    if k > len(base):
        raise ValueError(f"folds={k} too large for default palette.")
    cmap = ListedColormap(base[:k])

    fig, ax = plt.subplots(figsize=(8.5, 9.5), dpi=150)
    ax.set_title(title, fontsize=11)
    gdf_ll.plot(ax=ax, column="fold", categorical=True, cmap=cmap, legend=True,
                linewidth=0.0, edgecolor="none", missing_kwds={"color": "lightgray"})
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_axis_off()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_fold_masks(
    *,
    gdf_ll,
    out_dir: Path,
    k: int,
    title_prefix: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Saves individual PNG masks for each fold.

    Args:
        gdf_ll: GeoDataFrame in EPSG:4326 with 'fold' column.
        out_dir: Destination directory.
        k: Number of folds.
        title_prefix: Prefix for plot titles.
        xlim: Optional longitude limits.
        ylim: Optional latitude limits.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in range(k):
        fig, ax = plt.subplots(figsize=(8.5, 9.5), dpi=150)
        ax.set_title(f"{title_prefix} — fold {f}", fontsize=11)
        
        # Plot background and highlighted fold.
        gdf_ll.plot(ax=ax, color="#d9d9d9", linewidth=0.0, edgecolor="none")
        gdf_ll.loc[gdf_ll["fold"] == f].plot(ax=ax, color="#1f77b4", linewidth=0.0, edgecolor="none")
        
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.set_axis_off()
        fig.savefig(out_dir / f"fold_{f}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    """Main execution entry point for CV fold visualization."""
    parser = argparse.ArgumentParser(description="Visualize CV folds.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--parquet", type=Path, default=None, help="Input features parquet.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--per-fold", action="store_true", help="Write individual fold masks.")
    parser.add_argument("--cdmx-only", action="store_true", help="Zoom to CDMX.")
    parser.add_argument("--pad-deg", type=float, default=0.05, help="Zoom padding.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    parquet = args.parquet or ds.polygon_features_path
    out_dir = Path(args.output_dir or (ds.out_dir / "cv_folds"))

    # Assign folds based on parquet row order.
    df = pd.read_parquet(parquet, columns=["ageb_id"])
    ageb_series = df["ageb_id"].astype(str)
    if args.cdmx_only:
        ageb_series = ageb_series.loc[ageb_series.str.startswith("09")]
    
    ageb_ids = ageb_series.to_numpy()
    folds = _kfold_assign(len(ageb_ids), args.folds, args.seed)
    assign = pd.DataFrame({"ageb_id": ageb_ids, "fold": folds.astype(int)})

    out_dir.mkdir(parents=True, exist_ok=True)
    assign.to_parquet(out_dir / "fold_assignments.parquet", index=False)

    # Merge fold assignments with polygons and reproject for plotting.
    polygons_m = read_polygons(ds.shapefile_path)
    polygons_m["ageb_id"] = polygons_m["ageb_id"].astype(str)
    if args.cdmx_only:
        polygons_m = polygons_m.loc[polygons_m["ageb_id"].str.startswith("09")].copy()
    
    gdf_ll = polygons_m.merge(assign, on="ageb_id", how="left").to_crs("EPSG:4326")

    xlim = ylim = None
    if args.cdmx_only:
        b = gdf_ll.total_bounds
        xlim = (float(b[0]) - args.pad_deg, float(b[2]) + args.pad_deg)
        ylim = (float(b[1]) - args.pad_deg, float(b[3]) + args.pad_deg)

    title = f"Random {args.folds}-fold CV assignment (seed={args.seed}) — {ds.year}"
    out_png = out_dir / ("fold_map_cdmx.png" if args.cdmx_only else "fold_map.png")
    _save_fold_map(gdf_ll=gdf_ll, out_path=out_png, k=int(args.folds), title=title, xlim=xlim, ylim=ylim)

    if args.per_fold:
        _save_fold_masks(gdf_ll=gdf_ll, out_dir=out_dir, k=int(args.folds), title_prefix=title, xlim=xlim, ylim=ylim)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
