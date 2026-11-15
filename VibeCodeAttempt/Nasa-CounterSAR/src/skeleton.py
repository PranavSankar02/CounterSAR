from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


KUTTANAD_BOUNDS: Dict[str, float] = {
    "lat_min": 9.25,
    "lat_max": 9.58,
    "lon_min": 76.30,
    "lon_max": 76.55,
}

GEOGRAPHIC_CRS: str = "EPSG:4326"
CRS: str = "EPSG:32643"

GRID_RES_METERS: float = 100.0

S1_REVISIT_DAYS: int = 12
DEFAULT_START_DATE: str = "2023-01-01"
DEFAULT_END_DATE: str = "2024-06-30"


class CounterSARError(Exception):
    pass


class SpatialBoundaryError(CounterSARError):
    pass


class TemporalAlignmentError(CounterSARError):
    pass


class DataFormatError(CounterSARError):
    pass


class IncompatibleDatasetError(CounterSARError):
    pass


class DataSource(str, Enum):
    SENTINEL1 = "sentinel1"
    NISAR = "nisar"
    CHIRPS = "chirps"
    ERA5 = "era5"
    SYNTHETIC = "synthetic"


class Polarization(str, Enum):
    VV = "VV"
    VH = "VH"
    HH = "HH"
    HV = "HV"


class OrbitDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    UNKNOWN = "unknown"


class SARBand(str, Enum):
    C_BAND = "C"
    L_BAND = "L"
    S_BAND = "S"


class VariableType(str, Enum):
    BACKSCATTER_DB = "backscatter_db"
    RAINFALL_MM = "rainfall_mm"
    SOIL_MOISTURE = "soil_moisture"
    TEMPERATURE_C = "temperature_c"


@dataclass(frozen=True)
class SpatialGridSpec:
    lat_min: float = KUTTANAD_BOUNDS["lat_min"]
    lat_max: float = KUTTANAD_BOUNDS["lat_max"]
    lon_min: float = KUTTANAD_BOUNDS["lon_min"]
    lon_max: float = KUTTANAD_BOUNDS["lon_max"]
    resolution_m: float = GRID_RES_METERS
    crs: str = CRS

    def __post_init__(self):
        if self.lat_min >= self.lat_max:
            raise SpatialBoundaryError("lat_min must be smaller than lat_max")

        if self.lon_min >= self.lon_max:
            raise SpatialBoundaryError("lon_min must be smaller than lon_max")

        if self.resolution_m <= 0:
            raise SpatialBoundaryError("Grid resolution must be positive")

    @property
    def geographic_bounds(self) -> Dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }

    @property
    def shape(self) -> Tuple[int, int]:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            GEOGRAPHIC_CRS,
            self.crs,
            always_xy=True
        )

        min_x, min_y = transformer.transform(
            self.lon_min,
            self.lat_min
        )

        max_x, max_y = transformer.transform(
            self.lon_max,
            self.lat_max
        )

        width = int(np.ceil((max_x - min_x) / self.resolution_m))
        height = int(np.ceil((max_y - min_y) / self.resolution_m))

        return height, width

    @property
    def x(self) -> np.ndarray:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            GEOGRAPHIC_CRS,
            self.crs,
            always_xy=True
        )

        min_x, _ = transformer.transform(
            self.lon_min,
            self.lat_min
        )

        _, max_y = transformer.transform(
            self.lon_min,
            self.lat_max
        )

        height, width = self.shape

        return min_x + (
            np.arange(width) + 0.5
        ) * self.resolution_m

    @property
    def y(self) -> np.ndarray:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            GEOGRAPHIC_CRS,
            self.crs,
            always_xy=True
        )

        _, min_y = transformer.transform(
            self.lon_min,
            self.lat_min
        )

        height, width = self.shape

        return min_y + (
            np.arange(height) + 0.5
        ) * self.resolution_m


@dataclass
class DatasetMetadata:
    source: DataSource
    is_synthetic: bool
    crs: str = CRS
    time_range: Optional[Tuple[str, str]] = None
    spatial_bounds: Dict[str, float] = field(
        default_factory=lambda: dict(KUTTANAD_BOUNDS)
    )
    variables: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SARObservation:
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
    date: str
    predicted_backscatter_db: float
    prediction_std: float
    lower_bound_db: float
    upper_bound_db: float
    model_name: str
    is_synthetic: bool = False


@dataclass
class AnomalyResult:
    date: str
    latitude: float
    longitude: float
    observed_db: float
    predicted_db: float
    z_score: float
    is_persistent: bool
    anomaly_type: str
    is_synthetic: bool = False