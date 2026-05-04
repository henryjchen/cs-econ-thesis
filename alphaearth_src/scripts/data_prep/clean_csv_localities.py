"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine, pandas, numpy
Data Dependencies: INEGI 2020 census files (raw/all_mexico_loc_2020.csv)
Description: Cleans locality-level census data, removes aggregate labels, 
             and calculates population metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def clean_localities(input_file, output_path):
    """Cleans raw locality census CSV and filters for actual localities.

    Args:
        input_file (Path): Path to the raw census CSV for localities.
        output_path (Path): Path where the cleaned CSV will be saved.
    """
    print(f"Processing {input_file}...")
    
    # Load data with appropriate encoding and string types for IDs
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig', dtype={
            'ENTIDAD': str, 'MUN': str, 'LOC': str, 'NOM_LOC': str, 'POBTOT': str
        })
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='latin-1', dtype={
            'ENTIDAD': str, 'MUN': str, 'LOC': str, 'NOM_LOC': str, 'POBTOT': str
        })
        df.columns = [c.lstrip('ï»¿') for c in df.columns]

    # Exclude aggregate labels and placeholder rows
    exclude_locs = [
        'Total nacional',
        'Localidades de una vivienda',
        'Localidades de dos viviendas',
        'Total de la Entidad',
        'Total del Municipio'
    ]
    df_clean = df[~df['NOM_LOC'].isin(exclude_locs)].copy()

    # Filter out state and municipality totals
    df_clean = df_clean[(df_clean['MUN'].astype(int) != 0) & (df_clean['LOC'].astype(int) != 0)].copy()

    # Construct 9-digit CVEGEO
    df_clean['CVEGEO'] = (
        df_clean['ENTIDAD'].str.zfill(2) + 
        df_clean['MUN'].str.zfill(3) + 
        df_clean['LOC'].str.zfill(4)
    )

    # Convert population to numeric and log-transform
    df_clean['POBTOT'] = pd.to_numeric(df_clean['POBTOT'], errors='coerce').fillna(0)
    df_clean['log_POBTOT'] = np.log10(df_clean['POBTOT'] + 1)

    # Column selection and export
    final_columns = ['ENTIDAD', 'MUN', 'LOC', 'CVEGEO', 'POBTOT', 'log_POBTOT']
    df_final = df_clean[final_columns]
    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} localities to: {output_path}")

if __name__ == "__main__":
    input_csv = PROJECT_ROOT / 'data' / 'raw' / 'all_mexico_loc_2020.csv'
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / 'all_mexico_cleaned_loc_2020.csv'
    clean_localities(input_csv, output_csv)
