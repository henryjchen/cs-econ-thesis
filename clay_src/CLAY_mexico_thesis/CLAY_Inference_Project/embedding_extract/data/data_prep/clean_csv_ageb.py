"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, pandas
Data Dependencies: Clay model checkpoints, INEGI census CSVs, GeoTIFF imagery
Description: Cleans and aggregates urban AGEB census data from INEGI CSV files. 
             Standardizes geographic codes and calculates population statistics.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data" / "inegi_csvs"
PROCESSED_DIR = BASE_DIR / "processed_data"

_FILENAME_STATE_RE = re.compile(r"resultados_ageb_urbana_(\d+)_cpv2010\.csv$", re.IGNORECASE)


def clean_csv(input_file: Path, output_path: Path) -> None:
    """Cleans a single INEGI census CSV and aggregates population to AGEB level.

    Args:
        input_file: Path to the raw INEGI CSV.
        output_path: Path to save the cleaned CSV.
    """
    print(f"Processing {input_file.name}...")

    dtype_map = {
        "entidad": str,
        "mun": str,
        "loc": str,
        "ageb": str,
        "mza": str,
        "pobtot": str,
    }
    
    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig", dtype=dtype_map, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding="latin-1", dtype=dtype_map, low_memory=False)
        df.columns = [c.lstrip("ï»¿") for c in df.columns]

    df.columns = [c.strip().lower() for c in df.columns]

    # Filter out aggregate rows
    mza = df["mza"].astype(str).str.strip()
    ageb = df["ageb"].astype(str).str.strip()
    if (mza != "000").any():
        df_clean = df[(mza != "000") & (ageb != "0000")].copy()
    else:
        df_clean = df[ageb != "0000"].copy()

    # Clean population counts
    pob = pd.to_numeric(
        df_clean["pobtot"].astype(str).str.replace("*", "", regex=False),
        errors="coerce",
    ).fillna(0)
    df_clean["POBTOT"] = pob

    # Aggregate to AGEB
    group_cols = ["entidad", "mun", "loc", "ageb"]
    df_agg = df_clean.groupby(group_cols, as_index=False)["POBTOT"].sum()

    # Standardize codes
    ent = df_agg["entidad"].str.strip().str.zfill(2)
    mun = df_agg["mun"].str.strip().str.zfill(3)
    loc = df_agg["loc"].str.strip().str.zfill(4)
    ag = df_agg["ageb"].str.strip().str.zfill(4)

    df_final = pd.DataFrame(
        {
            "ENTIDAD": ent,
            "MUN": mun,
            "LOC": loc,
            "AGEB": ag,
            "CVEGEO": ent + mun + loc + ag,
            "POBTOT": df_agg["POBTOT"],
            "log_POBTOT": np.log(df_agg["POBTOT"] + 1),
        }
    )

    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} AGEBs to: {output_path}")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(RAW_DIR)):
        m = _FILENAME_STATE_RE.match(filename)
        if not m:
            continue
        
        state_num = m.group(1).zfill(2)
        input_path = RAW_DIR / filename
        output_path = PROCESSED_DIR / f"{state_num}_cleaned_ageb_cpv2010.csv"
        clean_csv(input_path, output_path)
