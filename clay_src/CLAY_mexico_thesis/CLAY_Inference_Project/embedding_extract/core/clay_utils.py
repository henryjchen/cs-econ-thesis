"""
Project Context: 2020 Mexico Clay Pipeline
Environment Specs: Local machine / Yale Bouchet HPC, PyTorch, rasterio, pandas
Data Dependencies: Clay model checkpoints, INEGI census CSVs, GeoTIFF imagery
Description: Utility functions for CLAY model initialization, data normalization, 
             Earth Engine setup, and datacube construction.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from embedding_extract import config as fcfg


def normalize_timestamp(date):
    """Normalizes date into sine/cosine components for week and hour.

    Args:
        date: datetime object.

    Returns:
        tuple: ((sin_week, cos_week), (sin_hour, cos_hour))
    """
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat: float, lon: float):
    """Normalizes latitude and longitude into sine/cosine components.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        tuple: ((sin_lat, cos_lat), (sin_lon, cos_lon))
    """
    lat = lat * math.pi / 180
    lon = lon * math.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def get_waves(bands: dict, platform: str, metadata) -> list[float]:
    """Retrieves wavelengths for specified bands and platform.

    Args:
        bands: Dictionary mapping band names.
        platform: Satellite platform name.
        metadata: Metadata object containing wavelength info.

    Returns:
        list[float]: List of wavelengths.
    """
    return [metadata[platform].bands.wavelength[str(v)] for v in bands.values()]


def initialize_earth_engine() -> None:
    """Initializes Earth Engine with high-volume endpoint and service account."""
    import ee
    import google.auth

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(fcfg.GOOGLE_APPLICATION_CREDENTIALS_PATH)
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url=fcfg.EARTH_ENGINE_HIGH_VOLUME_URL,
    )


def load_mean_std(norm_stats_path: Path | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads normalization statistics (mean and std).

    Args:
        norm_stats_path: Path to .pth file. Defaults to config value.

    Returns:
        tuple: (mean, std) arrays, or None if file not found.
    """
    path = Path(norm_stats_path) if norm_stats_path is not None else fcfg.NORM_STATS_PATH
    if path is None or not Path(path).is_file():
        return None
    d = torch.load(path, weights_only=False)
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    return mean, std


def load_clay_encoder(device: torch.device):
    """Loads the CLAY encoder from a checkpoint.

    Args:
        device: Torch device to load the model onto.

    Returns:
        tuple: (encoder model, metadata Box)
    """
    import yaml
    from box import Box
    from Clay.src.model import ClayMAEModule

    weights = fcfg.WEIGHTS_PATH
    if not weights or not os.path.isfile(weights) or os.path.getsize(weights) == 0:
        raise FileNotFoundError(f"Clay weights not found/empty: {weights}")
    
    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=weights,
        metadata_path=fcfg.METADATA_PATH,
        shuffle=False,
        mask_ratio=0,
    )
    module.eval()
    encoder = module.model.encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    
    with open(fcfg.METADATA_PATH) as f:
        metadata = Box(yaml.safe_load(f))
    return encoder, metadata


def build_datacube(
    pixels_hwc: np.ndarray,
    center_lonlat: tuple[float, float],
    date,
    waves: np.ndarray,
    gsd: float,
    platform: str,
    device: torch.device,
) -> dict:
    """Formats raw pixel data and metadata into a CLAY-compatible datacube.

    Args:
        pixels_hwc: Pixel data in HWC format.
        center_lonlat: (longitude, latitude) of patch center.
        date: Acquisition date.
        waves: Band wavelengths.
        gsd: Ground sampling distance.
        platform: Satellite platform.
        device: Target torch device.

    Returns:
        dict: Tensors ready for model input.
    """
    lon, lat = center_lonlat
    lat_n, lon_n = normalize_latlon(lat, lon)
    week_n, hour_n = normalize_timestamp(date)
    
    t = torch.tensor(np.hstack((week_n, hour_n)), dtype=torch.float32, device=device).unsqueeze(0)
    ll = torch.tensor(np.hstack((lat_n, lon_n)), dtype=torch.float32, device=device).unsqueeze(0)
    pix = torch.tensor(pixels_hwc, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
    
    return {
        "platform": platform,
        "time": t,
        "latlon": ll,
        "pixels": pix,
        "gsd": torch.tensor(gsd, dtype=torch.float32, device=device),
        "waves": torch.tensor(waves, dtype=torch.float32, device=device),
    }
