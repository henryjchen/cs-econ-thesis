"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine, pandas, numpy
Data Dependencies: INEGI 2020 census files (raw/*_ageb_2020.csv)
Description: Cleans and aggregates block-level census data to the AGEB level, 
             calculating population totals and log-transformed values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def clean_csv(input_file, output_path):
    """Cleans raw AGEB census CSV and aggregates data to AGEB level.

    Args:
        input_file (Path): Path to the raw census CSV.
        output_path (Path): Path where the cleaned CSV will be saved.
    """
    print(f"Processing {input_file}...")
    
    # Load data with appropriate encoding and string types for IDs
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig', dtype={
            'ENTIDAD': str, 'MUN': str, 'LOC': str, 'AGEB': str, 'MZA': str, 'POBTOT': str
        })
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='latin-1', dtype={
            'ENTIDAD': str, 'MUN': str, 'LOC': str, 'AGEB': str, 'MZA': str, 'POBTOT': str
        })
        df.columns = [c.lstrip('ï»¿') for c in df.columns]
    
    # Filter for block-level data and skip totals
    if (df['MZA'] != '000').any():
        df_clean = df[(df['MZA'] != '000') & (df['AGEB'] != '0000')].copy()
    else:
        df_clean = df[df['AGEB'] != '0000'].copy()

    # Convert population to numeric
    df_clean['POBTOT'] = pd.to_numeric(df_clean['POBTOT'], errors='coerce').fillna(0)
    
    # Aggregate to AGEB level
    group_cols = ['ENTIDAD', 'MUN', 'LOC', 'AGEB']
    df_agg = df_clean.groupby(group_cols)['POBTOT'].sum().reset_index()
    
    # Construct 13-digit CVEGEO
    df_agg['CVEGEO'] = (
        df_agg['ENTIDAD'].astype(str).str.zfill(2) + 
        df_agg['MUN'].astype(str).str.zfill(3) + 
        df_agg['LOC'].astype(str).str.zfill(4) + 
        df_agg['AGEB'].astype(str).str.zfill(4)
    )
    
    # Log-transform population
    df_agg['log_POBTOT'] = np.log10(df_agg['POBTOT'] + 1)
    
    # Column selection and export
    final_cols = ["ENTIDAD", "MUN", "LOC", "AGEB", "CVEGEO", "POBTOT", "log_POBTOT"]
    df_final = df_agg[final_cols]
    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} AGEBs to: {output_path}")

if __name__ == "__main__":
    input_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all state-level AGEB files
    for input_path in sorted(input_dir.glob("*_ageb_2020.csv")):
        if input_path.name[0:2].isdigit():
            state_num = input_path.name.split('_')[0]
            output_filename = f"{state_num}_cleaned_ageb_2020.csv"
            output_path = output_dir / output_filename
            clean_csv(input_path, output_path)
