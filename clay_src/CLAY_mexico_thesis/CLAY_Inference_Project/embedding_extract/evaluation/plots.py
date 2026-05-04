"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, scikit-learn, matplotlib, seaborn, pandas
Data Dependencies: Clay embeddings (2010/2020), INEGI census data (2010/2020)
Description: Visualization suite for model evaluation. Generates report-ready figures, histograms, and choropleths from analysis artifacts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_extract import config as fcfg


# -----------------------------------------------------------------------------
# Public constants shared with ``analysis.py``
# -----------------------------------------------------------------------------
PRED_COL: dict[str, str] = {"ridge": "pred_ridge", "xgboost": "pred_xgb"}
_MODEL_LABEL: dict[str, str] = {"ridge": "Ridge", "xgboost": "XGBoost"}

_TARGET_META: dict[str, dict[str, str]] = {
    "log_popden": {
        "short": r"ln(1 + population density)",
        "axis":  r"ln(1 + population density)",
        "pretty": "natural-log population density (people/km^2)",
    },
    "log_POBTOT": {
        "short": r"ln(1 + total population)",
        "axis":  r"ln(1 + total population)",
        "pretty": "natural-log total population (people)",
    },
    "pop_dens": {
        "short": "population density",
        "axis":  r"population density [people/km$^2$]",
        "pretty": "population density (people/km^2)",
    },
    "POBTOT": {
        "short": "total population",
        "axis":  r"total population [people]",
        "pretty": "total population (people)",
    },
}


def target_label(target: str, kind: str) -> str:
    """Look up a report-ready label for a target variable.

    Args:
        target: Target column name.
        kind: Label style ('short', 'axis', or 'pretty').

    Returns:
        Formatted label string.
    """
    return _TARGET_META.get(target, {}).get(kind, target)


_STATE_NAMES: dict[str, str] = {
    "01": "Aguascalientes",
    "02": "Baja California",
    "03": "Baja California Sur",
    "04": "Campeche",
    "05": "Coahuila",
    "06": "Colima",
    "07": "Chiapas",
    "08": "Chihuahua",
    "09": "CDMX",
    "10": "Durango",
    "11": "Guanajuato",
    "12": "Guerrero",
    "13": "Hidalgo",
    "14": "Jalisco",
    "15": "Mexico (state)",
    "16": "Michoacan",
    "17": "Morelos",
    "18": "Nayarit",
    "19": "Nuevo Leon",
    "20": "Oaxaca",
    "21": "Puebla",
    "22": "Queretaro",
    "23": "Quintana Roo",
    "24": "San Luis Potosi",
    "25": "Sinaloa",
    "26": "Sonora",
    "27": "Tabasco",
    "28": "Tamaulipas",
    "29": "Tlaxcala",
    "30": "Veracruz",
    "31": "Yucatan",
    "32": "Zacatecas",
}


def _import_matplotlib():
    """Import matplotlib with headless backend.

    Returns:
        Tuple of (plt module, LogNorm class) or (None, None) if unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        return plt, LogNorm
    except ImportError:
        return None, None


# -----------------------------------------------------------------------------
# Per-state R^2 bar chart
# -----------------------------------------------------------------------------
def save_bar(
    by_state: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    target: str,
    cv_name: str,
) -> None:
    """Generates a grouped bar chart of per-state out-of-fold R^2.

    Args:
        by_state: Dictionary of per-model state metric DataFrames.
        out_path: Destination PNG path.
        target: Target column name.
        cv_name: Name of the CV scheme used.
    """
    plt, _ = _import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping state R^2 bar chart")
        return

    states = sorted(next(iter(by_state.values()))["state"].tolist())
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(states)), 5))
    width = 0.4
    positions = np.arange(len(states))
    for i, (name, df) in enumerate(by_state.items()):
        d = df.set_index("state").reindex(states)
        ax.bar(
            positions + (i - 0.5) * width,
            d["r2"].values,
            width=width,
            label=_MODEL_LABEL.get(name, name),
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(states, rotation=90, fontsize=8)
    ax.set_xlabel("Mexican state (INEGI entidad code, CVEGEO[:2])")
    ax.set_ylabel(r"Out-of-fold $R^2$")
    ax.set_title(
        f"Out-of-fold $R^2$ by state  "
        f"-- target: {target_label(target, 'short')},  CV: {cv_name}"
    )
    ax.legend(loc="best", title="Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Diagnostic histogram bundle
# -----------------------------------------------------------------------------
def save_histograms(
    oof: pd.DataFrame,
    per_state: dict[str, pd.DataFrame],
    out_dir: Path,
    *,
    tag: str,
    target: str,
    cv_name: str,
    unit_name: str = "AGEBs",
    scenario: pd.Series | None = None,
) -> None:
    """Writes a suite of diagnostic plots for model performance.

    Args:
        oof: DataFrame containing OOF predictions and true values.
        per_state: Dictionary of per-model state metric DataFrames.
        out_dir: Directory to save plots.
        tag: Feature set identifier tag.
        target: Target column name.
        cv_name: Name of the CV scheme used.
        unit_name: Name of the observation unit (e.g., 'AGEBs').
        scenario: Optional Series indicating data aggregation scenarios.
    """
    plt, LogNorm = _import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping histograms.")
        return

    models = [m for m in ("ridge", "xgboost") if PRED_COL[m] in oof.columns]
    if not models:
        return

    target_short = target_label(target, "short")
    target_axis = target_label(target, "axis")
    resid_axis = f"Residual (observed $-$ predicted), {target_short}"
    y = oof["y"].to_numpy()

    # --- Residual distributions ---
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                              sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        p = oof[PRED_COL[m]].to_numpy()
        resid = y - p
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ax.hist(resid, bins=80, color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.axvline(float(resid.mean()), color="crimson", linewidth=1.2,
                   label=f"mean = {resid.mean():.3f}")
        ax.axvline(rmse, color="gray", linestyle="--", linewidth=0.8,
                   label=f"$\\pm$ RMSE = {rmse:.3f}")
        ax.axvline(-rmse, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"{_MODEL_LABEL.get(m, m)} residuals  (N = {len(y):,} {unit_name})"
        )
        ax.set_xlabel(resid_axis)
        ax.legend(loc="upper right", fontsize=9)
    axes[0][0].set_ylabel(f"Count ({unit_name})")
    fig.suptitle(
        f"Out-of-fold residual distribution  "
        f"-- target: {target_short},  CV: {cv_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p1 = out_dir / f"residuals_{tag}.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Residuals by scenario ---
    p2 = None
    if scenario is not None and scenario.nunique() > 1:
        scen = scenario.to_numpy()
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                                  sharey=False, squeeze=False)
        for ax, m in zip(axes[0], models):
            p = oof[PRED_COL[m]].to_numpy()
            resid = y - p
            for s in np.unique(scen):
                mask = scen == s
                if not mask.any():
                    continue
                vals = resid[mask]
                rmse_s = float(np.sqrt(np.mean(vals ** 2)))
                ax.hist(
                    vals, bins=60, histtype="step", linewidth=1.8,
                    label=f"scenario {s}  (N = {len(vals):,}, RMSE = {rmse_s:.3f})",
                )
            ax.axvline(0.0, color="black", linewidth=0.5)
            ax.set_title(f"{_MODEL_LABEL.get(m, m)} residuals by aggregation scenario")
            ax.set_xlabel(resid_axis)
            ax.legend(loc="upper right", fontsize=8)
        axes[0][0].set_ylabel(f"Count ({unit_name})")
        fig.suptitle(
            "Residuals split by centroid-lookup (A) vs spatial-average (B)  "
            f"-- target: {target_short},  CV: {cv_name}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        p2 = out_dir / f"residuals_by_scenario_{tag}.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Predicted vs observed (hexbin) ---
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5),
                              sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        p = oof[PRED_COL[m]].to_numpy()
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        hb = ax.hexbin(
            y, p, gridsize=60, cmap="viridis", mincnt=1,
            extent=(lo, hi, lo, hi), norm=LogNorm(),
        )
        ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.0, linestyle="--",
                label=r"$y = x$ (perfect)")
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y, p))
        ax.set_title(
            f"{_MODEL_LABEL.get(m, m)}: predicted vs observed  "
            f"($R^2 = {r2:.3f}$,  N = {len(y):,})"
        )
        ax.set_xlabel(f"Observed {target_axis}")
        ax.set_ylabel(f"Predicted {target_axis}")
        ax.legend(loc="upper left", fontsize=9)
        fig.colorbar(hb, ax=ax, shrink=0.85,
                     label="Count per hex-bin (log scale)")
    fig.suptitle(
        f"Predicted vs observed {target_short}  -- CV: {cv_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p3 = out_dir / f"pred_vs_actual_{tag}.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Marginal-distribution overlay ---
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                              sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        p = oof[PRED_COL[m]].to_numpy()
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        bins = np.linspace(lo, hi, 80)
        ax.hist(y, bins=bins, alpha=0.55, color="black",     label="Observed")
        ax.hist(p, bins=bins, alpha=0.55, color="steelblue",
                label=f"Predicted ({_MODEL_LABEL.get(m, m)})")
        ax.set_title(
            f"{_MODEL_LABEL.get(m, m)}: marginal distribution of observed vs predicted"
        )
        ax.set_xlabel(target_axis)
        ax.legend(loc="upper right", fontsize=9)
    axes[0][0].set_ylabel(f"Count ({unit_name})")
    fig.suptitle(
        f"Marginal distribution of observed vs predicted {target_short}  "
        f"-- CV: {cv_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p4 = out_dir / f"distribution_overlay_{tag}.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-state R^2 distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    r2_all = np.concatenate([
        d["r2"].dropna().to_numpy() for d in per_state.values()
    ]) if per_state else np.array([0.0])
    if len(r2_all) == 0:
        r2_all = np.array([0.0])
    lo = float(min(r2_all.min(), 0.0))
    hi = float(max(r2_all.max(), 1.0))
    bins = np.linspace(lo, hi, 24)
    for name, d in per_state.items():
        vals = d["r2"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(
            vals, bins=bins, alpha=0.55,
            label=(
                f"{_MODEL_LABEL.get(name, name)}  "
                f"(median $R^2$ = {np.median(vals):.3f}, "
                f"$n$ = {len(vals)} states)"
            ),
        )
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel(r"Per-state out-of-fold $R^2$")
    ax.set_ylabel("Number of Mexican states")
    ax.set_title(
        f"Distribution of per-state $R^2$  "
        f"-- target: {target_short},  CV: {cv_name}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    p5 = out_dir / f"state_r2_hist_{tag}.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for p in (p1, p2, p3, p4, p5):
        if p is not None:
            print(f"Saved {p}")


# -----------------------------------------------------------------------------
# Per-state R^2 choropleth of Mexico
# -----------------------------------------------------------------------------
_STATES_CACHE_PATH: Path = fcfg.for_year(fcfg.DEFAULT_YEAR).states_cache_path


def _load_state_polygons(
    *,
    rebuild: bool = False,
    shapefile_path: Path | None = None,
    cache_path: Path | None = None,
) -> "pd.DataFrame":
    """Loads and dissolves Mexican state polygons from AGEB shapefile.

    Args:
        rebuild: Whether to force a refresh of the cache.
        shapefile_path: Path to the source AGEB shapefile.
        cache_path: Path to the geopackage cache.

    Returns:
        GeoDataFrame of state polygons.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for the state-R^2 choropleth; "
            "install it or pass --skip-map."
        ) from exc

    default_ds = fcfg.for_year(fcfg.DEFAULT_YEAR)
    shp = shapefile_path if shapefile_path is not None else default_ds.shapefile_path
    cache = cache_path if cache_path is not None else default_ds.states_cache_path

    if cache.exists() and not rebuild:
        gdf = gpd.read_file(cache)
        ok = (
            "state" in gdf.columns
            and len(gdf) > 0
            and gdf.geometry.notna().all()
            and not gdf.geometry.is_empty.any()
        )
        if ok:
            gdf["state"] = gdf["state"].astype(str).str.zfill(2)
            print(f"  loaded {len(gdf)} state polygons from cache ({cache})")
            return gdf

    if not shp.exists():
        raise FileNotFoundError(f"AGEB shapefile not found at {shp}.")

    print(f"  building state polygons (dissolving {shp.name})...")
    ageb = gpd.read_file(shp)
    id_col = next((c for c in fcfg.AGEB_ID_CANDIDATES if c in ageb.columns), None)
    if id_col is None:
        raise KeyError(f"AGEB shapefile has no ID column.")
    ageb["state"] = ageb[id_col].astype(str).str.slice(0, 2).str.zfill(2)
    states = ageb[["state", "geometry"]].dissolve(by="state", as_index=False)
    states["state_name"] = states["state"].map(_STATE_NAMES).fillna(states["state"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    states.to_file(cache, driver="GPKG")
    print(f"  dissolved {len(ageb):,} AGEBs into {len(states)} states; cached to {cache}")
    return states


def save_state_map(
    per_state: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    target: str,
    cv_name: str,
    metric: str = "r2",
    label_states: bool = True,
    rebuild_states: bool = False,
    shapefile_path: Path | None = None,
    cache_path: Path | None = None,
) -> None:
    """Renders a choropleth map of per-state metrics.

    Args:
        per_state: Dictionary of per-model state metric DataFrames.
        out_path: Destination PNG path.
        target: Target column name.
        cv_name: Name of the CV scheme used.
        metric: Metric to plot (e.g., 'r2').
        label_states: Whether to annotate state codes on the map.
        rebuild_states: Whether to rebuild the state polygon cache.
        shapefile_path: Path to the source shapefile.
        cache_path: Path to the states cache.
    """
    plt, _ = _import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping state choropleth.")
        return

    try:
        states_gdf = _load_state_polygons(
            rebuild=rebuild_states,
            shapefile_path=shapefile_path,
            cache_path=cache_path,
        )
    except (ImportError, FileNotFoundError, KeyError) as exc:
        print(f"Skipping state choropleth: {exc}")
        return

    states_gdf = states_gdf.to_crs(fcfg.WORKING_CRS).copy()
    states_gdf = states_gdf[~states_gdf.geometry.is_empty & states_gdf.geometry.notna()]
    if states_gdf.empty:
        return

    metric_series = [
        pd.to_numeric(df[metric], errors="coerce").to_numpy()
        for df in per_state.values() if metric in df.columns
    ]
    all_vals = np.concatenate(metric_series) if metric_series else np.array([])
    finite = all_vals[np.isfinite(all_vals)]
    if metric == "r2":
        vmin = max(float(np.nanmin(finite)) if finite.size else 0.0, -1.0)
        vmax = min(float(np.nanmax(finite)) if finite.size else 1.0,  1.0)
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.1)
        cmap = "RdYlGn"
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        cmap = "viridis"

    models = list(per_state)
    ncols = len(models) or 1
    panel_w, panel_h = 7.0, 5.0
    fig, axes = plt.subplots(1, ncols, figsize=(panel_w * ncols, panel_h), squeeze=False)

    for ax, name in zip(axes[0], models):
        df = per_state[name][["state", metric]].copy()
        df["state"] = df["state"].astype(str).str.zfill(2)
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        merged = states_gdf.merge(df, on="state", how="left")

        merged.plot(
            ax=ax, column=metric, cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolor="#333333", linewidth=0.4,
            missing_kwds={"color": "lightgray", "hatch": "///"},
        )
        ax.set_aspect("equal")
        if label_states:
            reps = merged.geometry.representative_point()
            for (x, y), code in zip(zip(reps.x, reps.y), merged["state"]):
                ax.text(x, y, code, ha="center", va="center", fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        ax.set_title(_MODEL_LABEL.get(name, name), fontsize=12)
        ax.set_axis_off()

    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    sm = mcm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    metric_label = {"r2": r"Out-of-fold $R^2$", "mae": "MAE", "rmse": "RMSE"}.get(metric, metric)
    fig.colorbar(sm, ax=axes[0].tolist(), orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.6, label=metric_label)

    target_short = target_label(target, "short")
    fig.suptitle(f"Per-state {metric_label} over Mexico -- target: {target_short}, CV: {cv_name}", fontsize=11, y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Artifact Loading
# -----------------------------------------------------------------------------
@dataclass
class RunArtifacts:
    """Container for analysis run artifacts."""
    tag: str
    out_dir: Path
    target: str
    cv_name: str
    unit_name: str
    oof: pd.DataFrame
    per_state: dict[str, pd.DataFrame]
    scenario: pd.Series | None
    meta: dict


def _discover_tags(out_dir: Path) -> list[str]:
    """Infers feature set tags from artifact filenames."""
    tags: list[str] = []
    for p in sorted(out_dir.glob("predictions_oof_*.parquet")):
        name = p.stem
        if name.startswith("predictions_oof_"):
            tags.append(name[len("predictions_oof_"):])
    return tags


def _discover_analysis_dirs(parent: Path) -> list[Path]:
    """Finds directories containing analysis artifacts."""
    if not parent.is_dir():
        return []
    hits: list[Path] = []
    candidates = sorted({d for d in parent.iterdir() if d.is_dir()}, key=lambda p: p.name)
    for d in candidates:
        if any(d.glob("predictions_oof_*.parquet")):
            hits.append(d)
    return hits


def load_run(out_dir: Path, tag: str) -> RunArtifacts:
    """Loads artifacts for a single analysis run.

    Args:
        out_dir: Directory containing artifacts.
        tag: Feature set identifier tag.

    Returns:
        RunArtifacts object.
    """
    meta_path  = out_dir / f"metrics_overall_{tag}.json"
    state_path = out_dir / f"metrics_by_state_{tag}.csv"
    oof_path   = out_dir / f"predictions_oof_{tag}.parquet"

    meta = json.loads(meta_path.read_text())
    oof = pd.read_parquet(oof_path)
    by_state_df = pd.read_csv(state_path, dtype={"state": str})

    per_state = {m: g.drop(columns=["model"]).reset_index(drop=True) for m, g in by_state_df.groupby("model", sort=False)}
    scenario = oof["scenario"] if "scenario" in oof.columns else None

    return RunArtifacts(
        tag=tag, out_dir=out_dir, target=meta.get("target", "log_popden"),
        cv_name=meta.get("cv", "unknown CV"), unit_name=meta.get("unit_name", "AGEBs"),
        oof=oof, per_state=per_state, scenario=scenario, meta=meta,
    )


def render_run(
    run: RunArtifacts,
    *,
    draw_map: bool = True,
    rebuild_states: bool = False,
    shapefile_path: Path | None = None,
    states_cache_path: Path | None = None,
) -> None:
    """Renders all figures for a single run.

    Args:
        run: RunArtifacts to render.
        draw_map: Whether to render the choropleth map.
        rebuild_states: Whether to rebuild state polygons.
        shapefile_path: Path to the source shapefile.
        states_cache_path: Path to the states cache.
    """
    save_bar(run.per_state, run.out_dir / f"state_r2_bar_{run.tag}.png", target=run.target, cv_name=run.cv_name)
    save_histograms(run.oof, run.per_state, run.out_dir, tag=run.tag, target=run.target, cv_name=run.cv_name, unit_name=run.unit_name, scenario=run.scenario)
    if draw_map:
        save_state_map(run.per_state, run.out_dir / f"state_r2_map_{run.tag}.png", target=run.target, cv_name=run.cv_name, rebuild_states=rebuild_states, shapefile_path=shapefile_path, cache_path=states_cache_path)


def main() -> int:
    """CLI entry point for rendering analysis plots."""
    parser = argparse.ArgumentParser(description="Visualization suite for model evaluation.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--out-dir", type=Path, action="append", help="Analysis directories.")
    parser.add_argument("--tag", action="append", help="Feature tags to render.")
    parser.add_argument("--skip-map", action="store_true", help="Skip the choropleth map.")
    parser.add_argument("--rebuild-states", action="store_true", help="Rebuild state polygons.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    out_dirs = [Path(d) for d in args.out_dir] if args.out_dir else _discover_analysis_dirs(ds.out_dir)
    if not out_dirs:
        parser.error(f"No analysis directories found under {ds.out_dir}.")

    for out_dir in out_dirs:
        tags = args.tag if args.tag else _discover_tags(out_dir)
        for tag in tags:
            run = load_run(out_dir, tag)
            render_run(run, draw_map=not args.skip_map, rebuild_states=args.rebuild_states, shapefile_path=ds.shapefile_path, states_cache_path=ds.states_cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    plt, LogNorm = _import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping histograms.")
        return

    models = [m for m in ("ridge", "xgboost") if PRED_COL[m] in oof.columns]
    if not models:
        return

    target_short = target_label(target, "short")
    target_axis = target_label(target, "axis")
    resid_axis = f"Residual (observed $-$ predicted), {target_short}"
    y = oof["y"].to_numpy()

    # --- Residual distributions ------------------------------------------------
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                              sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        p = oof[PRED_COL[m]].to_numpy()
        resid = y - p
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ax.hist(resid, bins=80, color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.axvline(float(resid.mean()), color="crimson", linewidth=1.2,
                   label=f"mean = {resid.mean():.3f}")
        ax.axvline(rmse, color="gray", linestyle="--", linewidth=0.8,
                   label=f"$\\pm$ RMSE = {rmse:.3f}")
        ax.axvline(-rmse, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"{_MODEL_LABEL.get(m, m)} residuals  (N = {len(y):,} {unit_name})"
        )
        ax.set_xlabel(resid_axis)
        ax.legend(loc="upper right", fontsize=9)
    axes[0][0].set_ylabel(f"Count ({unit_name})")
    fig.suptitle(
        f"Out-of-fold residual distribution  "
        f"-- target: {target_short},  CV: {cv_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p1 = out_dir / f"residuals_{tag}.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Residuals split by aggregation scenario (A / B / A_missing) ----------
    p2 = None
    if scenario is not None and scenario.nunique() > 1:
        scen = scenario.to_numpy()
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                                  sharey=False, squeeze=False)
        for ax, m in zip(axes[0], models):
            p = oof[PRED_COL[m]].to_numpy()
            resid = y - p
            for s in np.unique(scen):
                mask = scen == s
                if not mask.any():
                    continue
                vals = resid[mask]
                rmse_s = float(np.sqrt(np.mean(vals ** 2)))
                ax.hist(
                    vals, bins=60, histtype="step", linewidth=1.8,
                    label=f"scenario {s}  (N = {len(vals):,}, RMSE = {rmse_s:.3f})",
                )
            ax.axvline(0.0, color="black", linewidth=0.5)
            ax.set_title(f"{_MODEL_LABEL.get(m, m)} residuals by aggregation scenario")
            ax.set_xlabel(resid_axis)
            ax.legend(loc="upper right", fontsize=8)
        axes[0][0].set_ylabel(f"Count ({unit_name})")
        fig.suptitle(
            "Residuals split by centroid-lookup (A) vs spatial-average (B)  "
            f"-- target: {target_short},  CV: {cv_name}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        p2 = out_dir / f"residuals_by_scenario_{tag}.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Predicted vs observed (hexbin) ---------------------------------------
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5),
                              sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        p = oof[PRED_COL[m]].to_numpy()
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        # LogNorm gives a log-scaled color mapping for point density. The exact
        # base doesn't matter here (counts are base-agnostic), so we label the
        # colorbar accordingly.
        hb = ax.hexbin(
            y, p, gridsize=60, cmap="viridis", mincnt=1,
            extent=(lo, hi, lo, hi), norm=LogNorm(),
        )
        ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.0, linestyle="--",
                label=r"$y = x$ (perfect)")
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y, p))
        ax.set_title(
            f"{_MODEL_LABEL.get(m, m)}: predicted vs observed  "
            f"($R^2 = {r2:.3f}$,  N = {len(y):,})"
        )
        ax.set_xlabel(f"Observed {target_axis}")
        ax.set_ylabel(f"Predicted {target_axis}")
        ax.legend(loc="upper left", fontsize=9)
        fig.colorbar(hb, ax=ax, shrink=0.85,
                     label="Count per hex-bin (log scale)")
    fig.suptitle(
        f"Predicted vs observed {target_short}  -- CV: {cv_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p3 = out_dir / f"pred_vs_actual_{tag}.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Marginal-distribution overlay (observed vs predicted) ---------------
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4),
                              sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        p = oof[PRED_COL[m]].to_numpy()
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        bins = np.linspace(lo, hi, 80)
        ax.hist(y, bins=bins, alpha=0.55, color="black",     label="Observed")
        ax.hist(p, bins=bins, alpha=0.55, color="steelblue",
                label=f"Predicted ({_MODEL_LABEL.get(m, m)})")
        ax.set_title(
            f"{_MODEL_LABEL.get(m, m)}: marginal distribution of observed vs predicted"
        )
        ax.set_xlabel(target_axis)
        ax.legend(loc="upper right", fontsize=9)
    axes[0][0].set_ylabel(f"Count ({unit_name})")
    fig.suptitle(
        f"Marginal distribution of observed vs predicted {target_short}  "
        f"-- CV: {cv_name}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p4 = out_dir / f"distribution_overlay_{tag}.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-state R^2 distribution ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    r2_all = np.concatenate([
        d["r2"].dropna().to_numpy() for d in per_state.values()
    ]) if per_state else np.array([0.0])
    if len(r2_all) == 0:
        r2_all = np.array([0.0])
    lo = float(min(r2_all.min(), 0.0))
    hi = float(max(r2_all.max(), 1.0))
    bins = np.linspace(lo, hi, 24)
    for name, d in per_state.items():
        vals = d["r2"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(
            vals, bins=bins, alpha=0.55,
            label=(
                f"{_MODEL_LABEL.get(name, name)}  "
                f"(median $R^2$ = {np.median(vals):.3f}, "
                f"$n$ = {len(vals)} states)"
            ),
        )
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel(r"Per-state out-of-fold $R^2$")
    ax.set_ylabel("Number of Mexican states")
    ax.set_title(
        f"Distribution of per-state $R^2$  "
        f"-- target: {target_short},  CV: {cv_name}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    p5 = out_dir / f"state_r2_hist_{tag}.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for p in (p1, p2, p3, p4, p5):
        if p is not None:
            print(f"Saved {p}")


# -----------------------------------------------------------------------------
# Per-state R^2 choropleth of Mexico
# -----------------------------------------------------------------------------
# Default state-polygon cache path (used if the caller doesn't override).
# Dissolving 56k AGEBs into 32 state polygons takes ~10s and is run-independent,
# so we cache it the first time we need it and reuse it thereafter. Lives in
# the per-year out/ so 2010 and 2020 dissolves don't collide.
_STATES_CACHE_PATH: Path = fcfg.for_year(fcfg.DEFAULT_YEAR).states_cache_path


def _load_state_polygons(
    *,
    rebuild: bool = False,
    shapefile_path: Path | None = None,
    cache_path: Path | None = None,
) -> "pd.DataFrame":
    """Return a GeoDataFrame of Mexican state polygons keyed by ``state``.

    Columns: ``state`` (2-char INEGI code), ``state_name``, ``geometry``.
    CRS is whatever the source shapefile uses (typically EPSG:6372 or 4326);
    the caller should reproject as needed.

    ``shapefile_path`` / ``cache_path`` default to the :data:`fcfg.DEFAULT_YEAR`
    dataset, but should be passed explicitly when rendering a specific year
    (plots.py's CLI does this via ``--year``).

    Lazy-imports geopandas so plots.py can still be imported on boxes that
    don't have it. Caches the dissolved result; pass ``rebuild=True`` to
    force a refresh.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for the state-R^2 choropleth; "
            "install it or pass --skip-map."
        ) from exc

    default_ds = fcfg.for_year(fcfg.DEFAULT_YEAR)
    shp = shapefile_path if shapefile_path is not None else default_ds.shapefile_path
    cache = cache_path if cache_path is not None else default_ds.states_cache_path

    if cache.exists() and not rebuild:
        gdf = gpd.read_file(cache)
        ok = (
            "state" in gdf.columns
            and len(gdf) > 0
            and gdf.geometry.notna().all()
            and not gdf.geometry.is_empty.any()
        )
        if ok:
            gdf["state"] = gdf["state"].astype(str).str.zfill(2)
            print(
                f"  loaded {len(gdf)} state polygons from cache "
                f"({cache})"
            )
            return gdf
        print(
            f"  state-polygon cache at {cache} looks "
            f"corrupted/empty; rebuilding from shapefile."
        )

    if not shp.exists():
        raise FileNotFoundError(
            f"AGEB shapefile not found at {shp}. "
            f"Can't build state polygons for the choropleth."
        )

    print(f"  building state polygons (dissolving {shp.name} "
          f"by first 2 chars of CVEGEO)...")
    ageb = gpd.read_file(shp)
    id_col = next((c for c in fcfg.AGEB_ID_CANDIDATES if c in ageb.columns), None)
    if id_col is None:
        raise KeyError(
            f"AGEB shapefile has no ID column in any of {fcfg.AGEB_ID_CANDIDATES}"
        )
    ageb["state"] = ageb[id_col].astype(str).str.slice(0, 2).str.zfill(2)
    states = (
        ageb[["state", "geometry"]]
        .dissolve(by="state", as_index=False)
    )
    states["state_name"] = states["state"].map(_STATE_NAMES).fillna(states["state"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    states.to_file(cache, driver="GPKG")
    print(
        f"  dissolved {len(ageb):,} AGEBs into {len(states)} state polygons; "
        f"cached to {cache}"
    )
    return states


def save_state_map(
    per_state: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    target: str,
    cv_name: str,
    metric: str = "r2",
    label_states: bool = True,
    rebuild_states: bool = False,
    shapefile_path: Path | None = None,
    cache_path: Path | None = None,
) -> None:
    """Render a choropleth of per-state ``metric`` (default R^2) per model.

    One panel per model, side-by-side, with a shared color scale so values
    are visually comparable. States missing from ``per_state`` (or with NaN
    metric) render in light gray. State codes are annotated on top of each
    polygon for quick orientation; pass ``label_states=False`` to suppress.

    Parameters
    ----------
    per_state
        ``{model_name -> DataFrame}`` where each DataFrame has at least
        ``state`` (2-char INEGI code) and the ``metric`` column.
    out_path
        Destination PNG.
    target
        Target column name -- only used for the figure title.
    cv_name
        CV description (e.g. ``"random 5-fold (seed=0)"``) -- also for title.
    metric
        Which column of ``per_state[m]`` to color by. R^2 is clamped into
        [-1, 1] for a sane color range.
    """
    plt, _ = _import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping state choropleth.")
        return

    try:
        states_gdf = _load_state_polygons(
            rebuild=rebuild_states,
            shapefile_path=shapefile_path,
            cache_path=cache_path,
        )
    except (ImportError, FileNotFoundError, KeyError) as exc:
        print(f"Skipping state choropleth: {exc}")
        return

    # Reproject to the INEGI meter grid so the map isn't distorted; EPSG:6372
    # is the project-wide "WORKING_CRS".
    states_gdf = states_gdf.to_crs(fcfg.WORKING_CRS).copy()
    # Guard: drop any empty / null geometries (shouldn't happen, but makes the
    # downstream plot call truly defensive).
    states_gdf = states_gdf[~states_gdf.geometry.is_empty]
    states_gdf = states_gdf[states_gdf.geometry.notna()]
    if states_gdf.empty:
        print("Skipping state choropleth: state geometries are empty.")
        return

    # Shared color scale across models. For R^2 we clip into [-1, 1] to keep
    # bad-state extremes from dominating the ramp; for other metrics use the
    # observed finite range.
    metric_series = [
        pd.to_numeric(df[metric], errors="coerce").to_numpy()
        for df in per_state.values() if metric in df.columns
    ]
    all_vals = np.concatenate(metric_series) if metric_series else np.array([])
    finite = all_vals[np.isfinite(all_vals)]
    if metric == "r2":
        vmin = max(float(np.nanmin(finite)) if finite.size else 0.0, -1.0)
        vmax = min(float(np.nanmax(finite)) if finite.size else 1.0,  1.0)
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.1)
        cmap = "RdYlGn"
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        cmap = "viridis"

    models = list(per_state)
    ncols = len(models) or 1
    # Figure sized to Mexico's extent so the fixed-aspect polygons fill most
    # of each panel. Mexico roughly fits in a 1.5:1 (lon:lat) box on the
    # meter grid.
    panel_w, panel_h = 7.0, 5.0
    fig, axes = plt.subplots(
        1, ncols, figsize=(panel_w * ncols, panel_h), squeeze=False,
    )

    for ax, name in zip(axes[0], models):
        df = per_state[name][["state", metric]].copy()
        df["state"] = df["state"].astype(str).str.zfill(2)
        # Force numeric dtype -- CSVs can round-trip these as strings when a
        # sibling column has mixed types, which would make geopandas' ``plot
        # column=...`` silently produce no color mapping.
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        merged = states_gdf.merge(df, on="state", how="left")

        n_have = int(merged[metric].notna().sum())
        n_miss = int(merged[metric].isna().sum())
        print(
            f"  [{name}] {n_have}/{len(merged)} states have a finite {metric} "
            f"(missing: {n_miss})"
        )

        # One plot() call with ``missing_kwds`` -- geopandas handles NaN
        # colouring for us, and we keep a visible dark edgecolor so polygons
        # are never invisible against a white figure background even if the
        # fill color is extremely light.
        merged.plot(
            ax=ax,
            column=metric,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolor="#333333",
            linewidth=0.4,
            missing_kwds={
                "color": "lightgray",
                "edgecolor": "#333333",
                "linewidth": 0.4,
                "hatch": "///",
                "label": "no data",
            },
        )

        # Lock aspect so the meter-grid looks like a map, not a stretched box.
        ax.set_aspect("equal")

        if label_states:
            # ``representative_point()`` always lies inside the polygon (unlike
            # centroid for non-convex shapes like Baja or Veracruz).
            reps = merged.geometry.representative_point()
            for (x, y), code in zip(zip(reps.x, reps.y), merged["state"]):
                ax.text(
                    x, y, code, ha="center", va="center",
                    fontsize=7, color="black",
                    bbox=dict(
                        facecolor="white", alpha=0.6, edgecolor="none",
                        boxstyle="round,pad=0.15",
                    ),
                )

        ax.set_title(_MODEL_LABEL.get(name, name), fontsize=12)
        ax.set_axis_off()

    # One shared horizontal colorbar underneath both panels.
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    sm = mcm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    metric_label = {
        "r2":   r"Out-of-fold $R^2$",
        "mae":  "MAE",
        "rmse": "RMSE",
    }.get(metric, metric)
    cbar = fig.colorbar(
        sm, ax=axes[0].tolist(), orientation="horizontal",
        fraction=0.035, pad=0.02, shrink=0.6,
    )
    cbar.set_label(metric_label)

    target_short = target_label(target, "short")
    shp_name = (
        shapefile_path.name
        if shapefile_path is not None
        else fcfg.for_year(fcfg.DEFAULT_YEAR).shapefile_path.name
    )
    fig.suptitle(
        f"Per-state {metric_label} over Mexico  "
        f"-- target: {target_short},  CV: {cv_name}  "
        f"(dissolved {shp_name}, CRS {fcfg.WORKING_CRS}; "
        f"hatched gray = no data)",
        fontsize=11, y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Load the artifacts written by analysis.py
# -----------------------------------------------------------------------------
@dataclass
class RunArtifacts:
    """Everything ``render_run()`` needs for one completed analysis run."""

    tag: str
    out_dir: Path
    target: str
    cv_name: str
    unit_name: str
    oof: pd.DataFrame
    per_state: dict[str, pd.DataFrame]
    scenario: pd.Series | None
    meta: dict


def _discover_tags(out_dir: Path) -> list[str]:
    """Tags inferred from ``predictions_oof_*.parquet`` files in ``out_dir``."""
    tags: list[str] = []
    for p in sorted(out_dir.glob("predictions_oof_*.parquet")):
        name = p.stem  # predictions_oof_{tag}
        if name.startswith("predictions_oof_"):
            tags.append(name[len("predictions_oof_"):])
    return tags


def _discover_analysis_dirs(parent: Path) -> list[Path]:
    """Return every subdirectory of ``parent`` that holds analysis artifacts.

    We recognize a directory as "an analysis run" if it contains at least one
    ``predictions_oof_*.parquet``. That covers both the single-scale default
    (``out/analysis/``) and the multi-scale wrapper script's layout
    (``out/analysis_ageb/``, ``out/analysis_muni_rollup/``,
    ``out/analysis_muni_direct/``).
    """
    if not parent.is_dir():
        return []
    hits: list[Path] = []
    # Prefer obvious "analysis*" subdirs for a tidy listing, but also accept
    # any other dir that happens to contain predictions_oof_*.parquet.
    candidates = sorted(
        {d for d in parent.iterdir() if d.is_dir()},
        key=lambda p: p.name,
    )
    for d in candidates:
        if any(d.glob("predictions_oof_*.parquet")):
            hits.append(d)
    return hits


def load_run(out_dir: Path, tag: str) -> RunArtifacts:
    """Read the three artifacts ``analysis.py`` wrote for one run.

    Parameters
    ----------
    out_dir
        Directory containing ``metrics_overall_{tag}.json``,
        ``metrics_by_state_{tag}.csv``, ``predictions_oof_{tag}.parquet``.
    tag
        The feature-set tag used at analysis time (e.g. ``"emb"``,
        ``"emb-state"``, ``"state"``).
    """
    meta_path  = out_dir / f"metrics_overall_{tag}.json"
    state_path = out_dir / f"metrics_by_state_{tag}.csv"
    oof_path   = out_dir / f"predictions_oof_{tag}.parquet"
    for p in (meta_path, state_path, oof_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing analysis artifact: {p}\n"
                f"did you run:\n"
                f"  python -m embedding_extract.evaluation.analysis "
                f"--out-dir {out_dir}"
            )

    meta = json.loads(meta_path.read_text())
    oof = pd.read_parquet(oof_path)
    by_state_df = pd.read_csv(state_path, dtype={"state": str})

    per_state: dict[str, pd.DataFrame] = {
        model: g.drop(columns=["model"]).reset_index(drop=True)
        for model, g in by_state_df.groupby("model", sort=False)
    }

    scenario: pd.Series | None = (
        oof["scenario"] if "scenario" in oof.columns else None
    )

    # Fall back gracefully for meta JSON written before these fields existed.
    target    = meta.get("target", "log_popden")
    cv_name   = meta.get("cv", "unknown CV")
    unit_name = meta.get("unit_name", "AGEBs")

    return RunArtifacts(
        tag=tag,
        out_dir=out_dir,
        target=target,
        cv_name=cv_name,
        unit_name=unit_name,
        oof=oof,
        per_state=per_state,
        scenario=scenario,
        meta=meta,
    )


def render_run(
    run: RunArtifacts,
    *,
    draw_map: bool = True,
    rebuild_states: bool = False,
    shapefile_path: Path | None = None,
    states_cache_path: Path | None = None,
) -> None:
    """Render the full figure set for one analysis run.

    Pass ``draw_map=False`` to skip the Mexico-state choropleth (useful if
    ``geopandas`` isn't installed, or for very fast dev loops -- dissolving
    the AGEB shapefile on first run takes ~10s). Pass ``rebuild_states=True``
    to force a refresh of the dissolved state polygon cache.

    ``shapefile_path`` / ``states_cache_path`` are forwarded into the
    choropleth so the CLI can render year-specific maps without mutating
    global state.
    """
    save_bar(
        run.per_state,
        run.out_dir / f"state_r2_bar_{run.tag}.png",
        target=run.target,
        cv_name=run.cv_name,
    )
    save_histograms(
        run.oof,
        run.per_state,
        run.out_dir,
        tag=run.tag,
        target=run.target,
        cv_name=run.cv_name,
        unit_name=run.unit_name,
        scenario=run.scenario,
    )
    if draw_map:
        save_state_map(
            run.per_state,
            run.out_dir / f"state_r2_map_{run.tag}.png",
            target=run.target,
            cv_name=run.cv_name,
            rebuild_states=rebuild_states,
            shapefile_path=shapefile_path,
            cache_path=states_cache_path,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    fcfg.add_year_arg(parser)
    parser.add_argument(
        "--out-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing analysis artifacts; repeatable to render "
             "several scales in one call. If omitted, every subdirectory of "
             "the selected --year's out root that contains "
             "predictions_oof_*.parquet is rendered (e.g. analysis_ageb, "
             "analysis_muni_rollup, analysis_muni_direct).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Render only this feature-set tag (repeatable). If omitted, "
             "every ``predictions_oof_*.parquet`` in --out-dir is rendered.",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Skip the Mexico-state R^2 choropleth. Useful if geopandas "
             "isn't installed or to avoid the ~10s one-time dissolve of the "
             "AGEB shapefile on first run.",
    )
    parser.add_argument(
        "--rebuild-states",
        action="store_true",
        help="Delete and rebuild the cached state-polygon file "
             "(mexico_states.gpkg under the selected year's out dir) by "
             "re-dissolving the AGEB shapefile. Use this if the cached "
             "file looks wrong.",
    )
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)

    if args.out_dir:
        # User asked for specific dirs; honor them but fail loudly if any are
        # missing (typo-catcher).
        out_dirs = [Path(d) for d in args.out_dir]
        missing = [d for d in out_dirs if not d.is_dir()]
        if missing:
            parser.error(
                "--out-dir not found: " + ", ".join(str(m) for m in missing)
            )
    else:
        # Auto-discover. Prefer the multi-scale layout run_all_analyses.sh
        # produces; fall back to the legacy single dir ``<out>/analysis/``.
        out_dirs = _discover_analysis_dirs(ds.out_dir)
        legacy = ds.out_dir / "analysis"
        if legacy.is_dir() and legacy not in out_dirs:
            if any(legacy.glob("predictions_oof_*.parquet")):
                out_dirs.insert(0, legacy)
        if not out_dirs:
            parser.error(
                f"no analysis directories with predictions_oof_*.parquet found "
                f"under {ds.out_dir}. Run analysis.py first "
                f"(or ./embedding_extract/run_all_analyses.sh "
                f"--year {ds.year})."
            )

    print(f"Year      : {ds.year}")
    print(f"Rendering {len(out_dirs)} directory/ies:")
    for d in out_dirs:
        print(f"  - {d}")

    for out_dir in out_dirs:
        tags = args.tag if args.tag else _discover_tags(out_dir)
        if not tags:
            print(f"\n[skip] {out_dir}: no predictions_oof_*.parquet found.")
            continue

        print(f"\n##### {out_dir}  tags={tags}")
        for tag in tags:
            print(f"\n=== tag: {tag} ===")
            run = load_run(out_dir, tag)
            print(
                f"  target={run.target}  cv={run.cv_name}  "
                f"unit={run.unit_name}  models={list(run.per_state)}  "
                f"rows={len(run.oof):,}"
            )
            render_run(
                run,
                draw_map=not args.skip_map,
                rebuild_states=args.rebuild_states,
                shapefile_path=ds.shapefile_path,
                states_cache_path=ds.states_cache_path,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
