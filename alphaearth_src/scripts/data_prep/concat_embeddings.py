"""
Project Context: 2020 Mexico AlphaEarth Pipeline
Environment Specs: Local machine, pandas, numpy
Data Dependencies: AlphaEarth embedding shards (data/embeddings/*.csv)
Description: Concatenates multiple AlphaEarth embedding CSV shards into 
             a single master embedding table.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FOLDER = PROJECT_ROOT / "data" / "embeddings"
OUTPUT_FILE = INPUT_FOLDER / "00_all_mexico_embeddings_combined.csv"

def main():
    """Main execution function to concatenate embedding shards."""
    # Identify all shard CSVs, excluding the master output file if it exists
    csv_files = sorted(
        path for path in INPUT_FOLDER.glob("*.csv")
        if path.resolve() != OUTPUT_FILE.resolve()
    )

    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting concatenation...")
    df_list = []

    # Read each shard and store in a list for concatenation
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Read: {file.name} (Rows: {len(df)})")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all shards and save to disk
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSuccess! All files concatenated.")
    print(f"Total rows in final dataset: {len(combined_df)}")
    print(f"Saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
