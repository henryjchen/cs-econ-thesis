"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, pandas
Data Dependencies: Clay model checkpoints, INEGI census CSVs, GeoTIFF imagery
Description: Merges state-level cleaned AGEB census CSVs into a single national file.
             Ensures geographic identifiers are preserved as strings.
"""

import os
import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"

_MERGED_FILE_RE = re.compile(r"^\d{2}_cleaned_ageb_cpv2010\.csv$")


def merge_agebs(input_dir: Path, output_path: Path) -> None:
    """Merges all state-level AGEB CSVs in a directory into one master CSV.

    Args:
        input_dir: Directory containing cleaned state-level CSVs.
        output_path: Path to save the merged national CSV.
    """
    print(f"Reading files from {input_dir}...")

    files = sorted(f for f in os.listdir(input_dir) if _MERGED_FILE_RE.match(f))

    all_dfs = []
    for filename in files:
        file_path = input_dir / filename
        df = pd.read_csv(
            file_path,
            dtype={"ENTIDAD": str, "MUN": str, "LOC": str, "AGEB": str, "CVEGEO": str},
        )
        all_dfs.append(df)
        print(f"Added {filename} ({len(df)} rows)")

    if not all_dfs:
        print("No files found to merge.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully merged {len(files)} files.")
    print(f"Total rows: {len(master_df)}")
    print(f"Master file saved to: {output_path}")


if __name__ == "__main__":
    output_file = PROCESSED_DIR / "all_mexico_cleaned_ageb_cpv2010.csv"
    merge_agebs(PROCESSED_DIR, output_file)
