"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, geopandas, matplotlib
Data Dependencies: Clay model embeddings, INEGI shapefiles, satellite chips
Description: Aggregates AGEB-level CLAY features up to municipio-level (first 5 characters of CVEGEO).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding_extract import config as fcfg


ID_WIDTH = 5  # state (2) + municipio (3)


def _emb_columns(df: pd.DataFrame) -> list[str]:
    """Retrieves and sorts embedding columns numerically.

    Args:
        df: Input DataFrame containing embedding columns.

    Returns:
        Sorted list of embedding column names (e.g., emb_0, emb_1, ...).
    """
    cols = [c for c in df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def _safe_log(x: pd.Series) -> pd.Series:
    """Computes log1p(x) with defensive checks.

    Args:
        x: Series to log-transform.

    Returns:
        Log-transformed series with NaNs for non-finite or negative inputs.
    """
    arr = x.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr >= 0)
    out[mask] = np.log1p(arr[mask])
    return pd.Series(out, index=x.index)


def _weighted_mean_embeddings(
    df: pd.DataFrame,
    emb_cols: list[str],
    group_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    """Computes group-level mean of embedding columns.

    Args:
        df: DataFrame containing embeddings and weights.
        emb_cols: List of embedding columns to aggregate.
        group_col: Column name to group by.
        weight_col: Column name for weights, or None for unweighted mean.

    Returns:
        DataFrame with aggregated embeddings.
    """
    if weight_col is None:
        return df.groupby(group_col, sort=True)[emb_cols].mean().astype(np.float32)

    w = df[weight_col].to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    emb_mat = df[emb_cols].to_numpy(dtype=np.float64)
    weighted = emb_mat * w[:, None]

    tmp = pd.DataFrame(weighted, columns=emb_cols, index=df.index)
    tmp[group_col] = df[group_col].values
    tmp["_w"] = w

    sums = tmp.groupby(group_col, sort=True)[emb_cols].sum()
    wsum = tmp.groupby(group_col, sort=True)["_w"].sum()

    wsum_safe = wsum.where(wsum > 0, np.nan)
    out = sums.div(wsum_safe, axis=0)
    return out.astype(np.float32)


def aggregate(
    df: pd.DataFrame,
    *,
    id_width: int = ID_WIDTH,
    unweighted: bool = False,
    drop_missing_embeddings: bool = True,
) -> pd.DataFrame:
    """Aggregates AGEB-level data to municipio level.

    Args:
        df: Input AGEB-level DataFrame.
        id_width: Width of the ID prefix to use for grouping.
        unweighted: Whether to use arithmetic instead of area-weighted mean.
        drop_missing_embeddings: Whether to drop rows with missing embeddings.

    Returns:
        Municipio-level aggregated DataFrame.
    """
    if "ageb_id" not in df.columns:
        raise KeyError("expected 'ageb_id' column in input parquet")

    emb_cols = _emb_columns(df)
    if len(emb_cols) != fcfg.EMBEDDING_DIM:
        raise RuntimeError(f"expected {fcfg.EMBEDDING_DIM} embedding columns, found {len(emb_cols)}")

    ageb_id = df["ageb_id"].astype(str)

    if drop_missing_embeddings:
        all_nan = df[emb_cols].isna().all(axis=1)
        if "scenario" in df.columns:
            all_nan = all_nan | (df["scenario"] == "A_missing")
        df = df.loc[~all_nan].reset_index(drop=True)
        ageb_id = ageb_id.loc[~all_nan].reset_index(drop=True)

    # Coerce numeric columns to float64 to avoid string concatenation issues.
    df = df.copy()
    for col in ("POBTOT", "polygon_area_m2", "area_sqkm"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    muni_id = ageb_id.str.slice(0, id_width)
    if (muni_id.str.len() != id_width).any():
        raise RuntimeError(f"ageb_ids shorter than {id_width} chars; ambiguous grouping.")
    df = df.assign(_muni=muni_id)

    # Aggregate embeddings and scenarios.
    weight_col = None if unweighted else "polygon_area_m2"
    if weight_col is not None and weight_col not in df.columns:
        weight_col = None
    emb_out = _weighted_mean_embeddings(df, emb_cols, "_muni", weight_col)

    if "scenario" in df.columns:
        scen = (df.assign(_one=1)
                .pivot_table(index="_muni", columns="scenario", values="_one",
                             aggfunc="sum", fill_value=0)
                .add_prefix("n_scen_"))
    else:
        scen = pd.DataFrame(index=emb_out.index)

    # Aggregate labels and recompute density metrics.
    agg_spec = {"n_agebs": ("ageb_id", "count")}
    if "POBTOT" in df.columns:
        agg_spec["POBTOT"] = ("POBTOT", "sum")
    if "polygon_area_m2" in df.columns:
        agg_spec["polygon_area_m2"] = ("polygon_area_m2", "sum")
    elif "area_sqkm" in df.columns:
        agg_spec["area_sqkm_passthrough"] = ("area_sqkm", "sum")

    label = df.groupby("_muni", sort=True).agg(**agg_spec)

    if "polygon_area_m2" in label.columns:
        label["area_sqkm"] = label["polygon_area_m2"].astype(float) / 1e6
    elif "area_sqkm_passthrough" in label.columns:
        label["area_sqkm"] = label.pop("area_sqkm_passthrough").astype(float)
        label["polygon_area_m2"] = label["area_sqkm"] * 1e6

    if "POBTOT" in label.columns and "area_sqkm" in label.columns:
        area = label["area_sqkm"].replace({0: np.nan})
        label["pop_dens"] = label["POBTOT"].astype(float) / area
        label["log_popden"] = _safe_log(label["pop_dens"])
        label["log_POBTOT"] = _safe_log(label["POBTOT"].astype(float))

    # Construct final output DataFrame.
    out = label.join(scen, how="left").join(emb_out, how="left").reset_index()
    out = out.rename(columns={"_muni": "ageb_id"})
    out.insert(1, "state", out["ageb_id"].str.slice(0, 2))

    emb_set = set(emb_cols)
    leading = ["ageb_id", "state"]
    middle = [c for c in out.columns if c not in emb_set and c not in leading]
    out = out[leading + middle + emb_cols]

    for c in out.columns:
        if c.startswith("n_scen_") or c == "n_agebs":
            out[c] = out[c].fillna(0).astype(int)

    return out


def main() -> int:
    """Main execution entry point for municipio aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate AGEB features to municipios.")
    fcfg.add_year_arg(parser)
    parser.add_argument("--input", type=Path, default=None, help="Input AGEB-level parquet.")
    parser.add_argument("--output", type=Path, default=None, help="Output municipio-level parquet.")
    parser.add_argument("--unweighted", action="store_true", help="Use arithmetic mean for embeddings.")
    parser.add_argument("--keep-missing", action="store_true", help="Keep AGEBs with missing embeddings.")
    args = parser.parse_args()

    ds = fcfg.for_year(args.year)
    input_path = args.input or ds.polygon_features_path
    output_path = args.output or ds.municipio_features_path

    df = pd.read_parquet(input_path)
    out = aggregate(df, id_width=ID_WIDTH, unweighted=args.unweighted,
                    drop_missing_embeddings=not args.keep_missing)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out):,} municipios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
