"""
CounterSAR — Shared Interfaces and Class Definitions
=====================================================
Contains dataclasses, enums, type definitions, and configuration
constants shared across the CounterSAR pipeline.

Study Area : Kuttanad, Kerala, India
CRS        : EPSG:4326
Units      : Backscatter in dB, rainfall in mm, soil moisture in m³/m³,
             temperature in °C
Spatial    : 100 m standard reference grid (~0.0009° in EPSG:4326)

All data loaders and pipeline components adhere to the contracts defined herein.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ════════════════════════════════════════════════════════════════════════
#  STUDY AREA & SPATIAL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

# Geographic bounds for Kuttanad, Kerala, India (EPSG:4326)
KUTTANAD_BOUNDS: Dict[str, float] = {
    "lat_min": 9.25,
    "lat_max": 9.58,
    "lon_min": 76.30,
    "lon_max": 76.55,
}

CRS: str = "EPSG:4326"

# 100 m spatial grid resolution in geographic degrees at ~9.4°N latitude
# 1° lat ≈ 110.57 km -> 100 m ≈ 0.000904°
# 1° lon ≈ 109.82 km (cos 9.4°) -> 100 m ≈ 0.000910°
GRID_RES_METERS: float = 100.0
GRID_RES_DEG: float = 0.0009

# Revisit intervals and temporal defaults
S1_REVISIT_DAYS: int = 12
DEFAULT_START_DATE: str = "2023-01-01"
DEFAULT_END_DATE: str = "2024-06-30"


# ════════════════════════════════════════════════════════════════════════
#  CUSTOM EXCEPTIONS
# ════════════════════════════════════════════════════════════════════════

class CounterSARError(Exception):
    """Base exception for all CounterSAR pipeline errors."""
    pass


class SpatialBoundaryError(CounterSARError):
    """Raised when spatial coordinates or bounding boxes are invalid or disjoint."""
    pass


class TemporalAlignmentError(CounterSARError):
    """Raised when temporal alignment between observations and covariates fails."""
    pass


class DataFormatError(CounterSARError):
    """Raised when a dataset file format or internal structure is invalid or corrupt."""
    pass


class IncompatibleDatasetError(CounterSARError):
    """Raised when required variables, dimensions, or coordinates are missing."""
    pass


# ════════════════════════════════════════════════════════════════════════
#  ENUMS
# ════════════════════════════════════════════════════════════════════════

class DataSource(str, Enum):
    """Supported remote-sensing and environmental data sources."""
    SENTINEL1 = "sentinel1"
    NISAR = "nisar"
    CHIRPS = "chirps"
    ERA5 = "era5"
    SYNTHETIC = "synthetic"


class Polarization(str, Enum):
    """SAR polarization channels."""
    VV = "VV"
    VH = "VH"
    HH = "HH"
    HV = "HV"


class OrbitDirection(str, Enum):
    """SAR orbit pass direction."""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    UNKNOWN = "unknown"


class SARBand(str, Enum):
    """Radar frequency bands."""
    C_BAND = "C"   # Sentinel-1 (~5.4 GHz)
    L_BAND = "L"   # NISAR L-SAR (~1.25 GHz)
    S_BAND = "S"   # NISAR S-SAR (~3.2 GHz)


class VariableType(str, Enum):
    """Pipeline physical variable types."""
    BACKSCATTER_DB = "backscatter_db"
    RAINFALL_MM = "rainfall_mm"
    SOIL_MOISTURE = "soil_moisture"
    TEMPERATURE_C = "temperature_c"


# ════════════════════════════════════════════════════════════════════════
#  SPATIAL GRID SPECIFICATION
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SpatialGridSpec:
    """
    Specification for the standard 100 m reference grid over Kuttanad.
    """
    lat_min: float = KUTTANAD_BOUNDS["lat_min"]
    lat_max: float = KUTTANAD_BOUNDS["lat_max"]
    lon_min: float = KUTTANAD_BOUNDS["lon_min"]
    lon_max: float = KUTTANAD_BOUNDS["lon_max"]
    res_deg: float = GRID_RES_DEG
    crs: str = CRS

    @property
    def lats(self) -> np.ndarray:
        """1D array of latitude coordinates from lat_min to lat_max."""
        # Include endpoint with appropriate step
        return np.arange(self.lat_min, self.lat_max + self.res_deg / 2.0, self.res_deg)

    @property
    def lons(self) -> np.ndarray:
        """1D array of longitude coordinates from lon_min to lon_max."""
        return np.arange(self.lon_min, self.lon_max + self.res_deg / 2.0, self.res_deg)

    @property
    def shape(self) -> Tuple[int, int]:
        """(num_lat, num_lon) grid dimensions."""
        return (len(self.lats), len(self.lons))


# ════════════════════════════════════════════════════════════════════════
#  DATA CLASSES & RECORD CONTRACTS
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetMetadata:
    """Provenance and specification metadata for loaded datasets."""
    source: DataSource
    is_synthetic: bool
    crs: str = CRS
    time_range: Optional[Tuple[str, str]] = None
    spatial_bounds: Dict[str, float] = field(default_factory=lambda: dict(KUTTANAD_BOUNDS))
    variables: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SARObservation:
    """Single SAR backscatter observation record."""
    date: str
    polarization: Polarization
    backscatter_db: float
    orbit: OrbitDirection = OrbitDirection.UNKNOWN
    source: DataSource = DataSource.SENTINEL1
    band: SARBand = SARBand.C_BAND
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentalRecord:
    """Environmental covariates for a single timestep."""
    date: str
    rainfall_mm: float
    soil_moisture: float
    temperature_c: float
    source: DataSource = DataSource.ERA5
    antecedent_rain_3d_mm: Optional[float] = None
    antecedent_rain_7d_mm: Optional[float] = None
    is_synthetic: bool = False


@dataclass
class PredictionResult:
    """Output from a backscatter prediction model."""
    date: str
    predicted_backscatter_db: float
    prediction_std: float
    lower_bound_db: float
    upper_bound_db: float
    model_name: str
    is_synthetic: bool = False


@dataclass
class AnomalyResult:
    """A scored anomaly event."""
    date: str
    latitude: float
    longitude: float
    observed_db: float
    predicted_db: float
    z_score: float
    is_persistent: bool
    anomaly_type: str
    is_synthetic: bool = False
