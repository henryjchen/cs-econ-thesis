"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, pandas
Data Dependencies: Clay model checkpoints, INEGI census CSVs, GeoTIFF imagery
Description: Cleans and standardizes locality-level census data from INEGI ITER files.
             Filters out aggregate rows and standardizes geographic identifiers.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data" / "inegi_csvs"
PROCESSED_DIR = BASE_DIR / "processed_data"


def clean_localities(input_file: Path, output_path: Path) -> None:
    """Cleans INEGI locality (ITER) CSV and standardizes geographic codes.

    Args:
        input_file: Path to the raw ITER CSV.
        output_path: Path to save the cleaned CSV.
    """
    print(f"Processing {input_file.name}...")

    dtype_map = {
        "entidad": str,
        "mun": str,
        "loc": str,
        "nom_loc": str,
        "pobtot": str,
    }
    
    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig", dtype=dtype_map, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding="latin-1", dtype=dtype_map, low_memory=False)
        df.columns = [c.lstrip("ï»¿") for c in df.columns]

    df.columns = [c.strip().lower() for c in df.columns]

    # Filter out national, state, and municipal totals
    exclude_exact = {
        "Total nacional",
        "Localidades de una vivienda",
        "Localidades de dos viviendas",
    }
    nom = df["nom_loc"].astype(str).str.strip()
    mask_exclude = (
        nom.isin(exclude_exact) | 
        nom.str.contains(r"Total de la entidad", case=False, na=False) | 
        nom.str.contains(r"Total del municipio", case=False, na=False)
    )
    df_clean = df[~mask_exclude].copy()

    # Filter out rows where mun or loc is 0 (aggregate rows)
    mun_int = pd.to_numeric(df_clean["mun"], errors="coerce").fillna(-1).astype(int)
    loc_int = pd.to_numeric(df_clean["loc"], errors="coerce").fillna(-1).astype(int)
    df_clean = df_clean[(mun_int != 0) & (loc_int != 0)].copy()

    # Standardize geographic codes
    ent = df_clean["entidad"].str.strip().str.zfill(2)
    mun = df_clean["mun"].str.strip().str.zfill(3)
    loc = df_clean["loc"].str.strip().str.zfill(4)
    df_clean["CVEGEO"] = ent + mun + loc

    # Clean population counts
    pob = pd.to_numeric(
        df_clean["pobtot"].astype(str).str.replace("*", "", regex=False), 
        errors="coerce"
    ).fillna(0)
    df_clean["POBTOT"] = pob
    df_clean["log_POBTOT"] = np.log(df_clean["POBTOT"] + 1)

    df_final = pd.DataFrame(
        {
            "ENTIDAD": ent,
            "MUN": mun,
            "LOC": loc,
            "CVEGEO": df_clean["CVEGEO"],
            "POBTOT": df_clean["POBTOT"],
            "log_POBTOT": df_clean["log_POBTOT"],
        }
    )

    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} localities to: {output_path}")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    input_csv = RAW_DIR / "iter_00_cpv2010.csv"
    output_csv = PROCESSED_DIR / "all_mexico_cleaned_loc_cpv2010.csv"
    clean_localities(input_csv, output_csv)
