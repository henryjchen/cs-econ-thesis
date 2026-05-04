"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine, pandas, numpy
Data Dependencies: Cleaned AGEB census files (processed/*_cleaned_ageb_2020.csv)
Description: Merges individual cleaned state-level AGEB CSV files into 
             a single national dataset.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def merge_agebs(input_dir, output_path):
    """Merges state-level cleaned AGEB CSVs into a national CSV.

    Args:
        input_dir (Path): Directory containing state-level cleaned CSVs.
        output_path (Path): Path where the merged CSV will be saved.
    """
    print(f"Reading files from {input_dir}...")
    
    # Identify and sort state-level cleaned files
    files = sorted(
        f for f in Path(input_dir).glob("*_cleaned_ageb_2020.csv")
        if f.name[0:2].isdigit()
    )
    
    all_dfs = []
    for file_path in files:
        # Load each file, ensuring geographic IDs are read as strings
        df = pd.read_csv(file_path, dtype={
            'ENTIDAD': str, 'MUN': str, 'LOC': str, 'AGEB': str, 'CVEGEO': str
        })
        all_dfs.append(df)
        print(f"Added {file_path.name} ({len(df)} rows)")

    # Concatenate all state dataframes and save
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully merged {len(files)} files.")
    print(f"Total rows: {len(master_df)}")
    print(f"Master file saved to: {output_path}")

if __name__ == "__main__":
    input_directory = PROJECT_ROOT / "data" / "processed"
    output_file = PROJECT_ROOT / "data" / "processed" / "all_mexico_cleaned_ageb_2020.csv"
    
    merge_agebs(input_directory, output_file)
