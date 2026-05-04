"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, pandas
Data Dependencies: Clay model checkpoints, INEGI census CSVs, GeoTIFF imagery
Description: Central configuration for the CLAY embedding extraction pipeline.
             Defines year-specific paths, model parameters, and output schemas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = _PROJECT_ROOT / "embedding_extract"

# --- Shared Settings ---
AGEB_ID_CANDIDATES = ("CVEGEO", "CVE_AGEB", "AGEB", "AGEB_ID", "ID", "GEO_ID")
POLYGON_PASSTHROUGH_COLUMNS = (
    "area_sqkm",
    "pop_dens",
    "log_popden",
    "POBTOT",
    "log_POBTOT",
)

# --- Grid Geometry ---
SCALE_M = 30
EMBED_GRID = 16  # 16x16 grid of 8x8 cells
PATCH_PIXELS = 128
CELL_SIZE_M = PATCH_PIXELS // EMBED_GRID * SCALE_M
EMBEDDING_DIM = 768
WORKING_CRS = "EPSG:6372"

# --- CLAY Checkpoint & Metadata ---
WEIGHTS_PATH = "/nfs/roberts/project/pi_ls2375/shared/Kenya_Inference/Kenya_Inference_Model/Clay/weights/clay-v1-base.ckpt"
METADATA_PATH = str(_PROJECT_ROOT / "Clay" / "configs" / "metadata.yaml")
PLATFORM = "landsat-c2l2-sr"
BANDS = {
    "SR_B1": "blue",
    "SR_B2": "green",
    "SR_B3": "red",
    "SR_B4": "nir08",
    "SR_B5": "swir16",
    "SR_B7": "swir22",
}

# --- Earth Engine Settings ---
GOOGLE_APPLICATION_CREDENTIALS_PATH = _PROJECT_ROOT / "mexico-key.json"
EARTH_ENGINE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

# --- Normalization Stats ---
NORM_STATS_PATH: Path | None = FEATURE_DIR / "out" / "norm_stats.pth"
FILTER_BY_AGEBS = True


@dataclass(frozen=True)
class DatasetConfig:
    """Year-dependent paths and IDs for the pipeline.

    Attributes:
        year: Census year (e.g., 2010, 2020).
        shapefile_path: Path to the AGEB polygon shapefile.
        gee_asset_id: Earth Engine asset ID for the Landsat composite.
        study_year: Reference year for GEE metadata.
        out_dir: Root directory for output artifacts.
        norm_stats_path: Path to normalization statistics.
    """

    year: int
    shapefile_path: Path
    gee_asset_id: str
    study_year: int
    out_dir: Path
    norm_stats_path: Path | None = None

    @property
    def grid_path(self) -> Path:
        """Path to embedding cells parquet."""
        return self.out_dir / "embedding_cells.parquet"

    @property
    def embeddings_npy_path(self) -> Path:
        """Path to raw embeddings numpy array."""
        return self.out_dir / "embeddings.npy"

    @property
    def polygon_features_path(self) -> Path:
        """Path to combined AGEB features parquet."""
        return self.out_dir / "ageb_features.parquet"

    @property
    def municipio_features_path(self) -> Path:
        """Path to aggregated municipio features."""
        return self.out_dir / "municipio_features.parquet"

    @property
    def municipio_features_direct_path(self) -> Path:
        """Path to direct municipio features."""
        return self.out_dir / "municipio_features_direct.parquet"

    @property
    def states_cache_path(self) -> Path:
        """Path to cached state boundaries."""
        return self.out_dir / "mexico_states.gpkg"

    @property
    def progress_heatmap_path(self) -> Path:
        """Path to extraction progress visualization."""
        return self.out_dir / "progress_heatmap.png"


DEFAULT_YEAR = 2010

YEARS: dict[int, DatasetConfig] = {
    2010: DatasetConfig(
        year=2010,
        shapefile_path=FEATURE_DIR / "data" / "all_mexico_merged_2010.shp",
        gee_asset_id="projects/mexico-census/assets/cloud_masked_mexico_enhanced_2010",
        study_year=2010,
        out_dir=FEATURE_DIR / "out",
        norm_stats_path=NORM_STATS_PATH,
    ),
    2020: DatasetConfig(
        year=2020,
        shapefile_path=FEATURE_DIR / "data" / "all_mexico_merged_2020.shp",
        gee_asset_id="projects/mexico-census/assets/cloud_masked_mexico_enhanced_2020",
        study_year=2020,
        out_dir=FEATURE_DIR / "out_2020",
        norm_stats_path=FEATURE_DIR / "out_2020" / "norm_stats.pth",
    ),
}


def for_year(year: int) -> DatasetConfig:
    """Returns DatasetConfig for a specific year.

    Args:
        year: Census year.

    Returns:
        DatasetConfig: Configuration for the requested year.

    Raises:
        ValueError: If year is not supported.
    """
    try:
        return YEARS[year]
    except KeyError as exc:
        raise ValueError(
            f"Unknown --year={year!r}. Supported: {sorted(YEARS)}."
        ) from exc


def add_year_arg(parser: argparse.ArgumentParser) -> None:
    """Adds --year CLI argument to a parser.

    Args:
        parser: argparse ArgumentParser instance.
    """
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        choices=sorted(YEARS),
        help=f"Census year to process. Default: {DEFAULT_YEAR}.",
    )


# --- Backward Compatibility ---
_DEFAULT_DS = YEARS[DEFAULT_YEAR]
SHAPEFILE_PATH = _DEFAULT_DS.shapefile_path
STUDY_INPUT_IMAGE_ASSET_ID = _DEFAULT_DS.gee_asset_id
STUDY_YEAR = _DEFAULT_DS.study_year
OUT_DIR = _DEFAULT_DS.out_dir
GRID_PATH = _DEFAULT_DS.grid_path
EMBEDDINGS_NPY_PATH = _DEFAULT_DS.embeddings_npy_path
POLYGON_FEATURES_PATH = _DEFAULT_DS.polygon_features_path
