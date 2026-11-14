"""
CounterSAR — Data Loading, Validation, and Temporal Alignment
=============================================================
Stage 2: Core data ingestion and preprocessing pipeline.

Supported Data Sources:
  - Sentinel-1 SAR (C-band VV/VH)
  - NISAR SAR (L-band / S-band polarizations)
  - CHIRPS rainfall
  - ERA5 reanalysis (soil moisture, temperature)
  - Synthetic test data (clearly tagged with is_synthetic=True)

Capabilities:
  - Reads NetCDF (.nc) and Zarr (.zarr) inputs safely (read-only)
  - Geographic boundary validation and subsetting to Kuttanad study area
  - 100 m standard spatial reference grid (~0.0009° in EPSG:4326)
  - Standardisation of linear backscatter power to decibels (dB)
  - Temporal alignment matching daily environmental covariates to SAR passes
  - Multi-day antecedent rainfall aggregation (3-day, 7-day windows)
  - Robust error handling and missing observation diagnostics
  - Strict preservation of raw data (never mutates or overwrites inputs)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.skeleton import (
    CRS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GRID_RES_DEG,
    GRID_RES_METERS,
    KUTTANAD_BOUNDS,
    CounterSARError,
    DataFormatError,
    DataSource,
    IncompatibleDatasetError,
    OrbitDirection,
    Polarization,
    SARBand,
    SpatialBoundaryError,
    SpatialGridSpec,
    TemporalAlignmentError,
)


# ════════════════════════════════════════════════════════════════════════
#  1. COORDINATE NORMALISATION & GEOGRAPHIC BOUNDARY VALIDATION
# ════════════════════════════════════════════════════════════════════════

COORD_ALIASES = {
    "latitude": ["latitude", "lat", "y", "LATITUDE", "Lat"],
    "longitude": ["longitude", "lon", "long", "x", "LONGITUDE", "Lon"],
    "time": ["time", "date", "datetime", "TIME", "Date"],
}


def normalize_coords(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Standardise dimension and coordinate names to 'latitude', 'longitude', and 'time'.
    Also sorts coordinates in ascending order for consistent spatial slicing.
    """
    rename_dict = {}
    for standard_name, aliases in COORD_ALIASES.items():
        if standard_name in ds.coords or standard_name in ds.dims:
            continue
        for alias in aliases:
            if alias in ds.coords or alias in ds.dims:
                rename_dict[alias] = standard_name
                break

    if rename_dict:
        ds = ds.rename(rename_dict)

    # Ensure latitude and longitude are monotonically increasing if present
    if "latitude" in ds.coords and len(ds.coords["latitude"]) > 1:
        if ds.coords["latitude"].values[0] > ds.coords["latitude"].values[-1]:
            ds = ds.sortby("latitude")

    if "longitude" in ds.coords and len(ds.coords["longitude"]) > 1:
        if ds.coords["longitude"].values[0] > ds.coords["longitude"].values[-1]:
            ds = ds.sortby("longitude")

    if "time" in ds.coords and len(ds.coords["time"]) > 1:
        time_vals = ds.coords["time"].values
        # Only sort if not already monotonically increasing
        if np.any(time_vals[:-1] > time_vals[1:]):
            ds = ds.sortby("time")

    return ds


def validate_geographic_bounds(
    ds: Union[xr.Dataset, xr.DataArray],
    bounds: Dict[str, float] = KUTTANAD_BOUNDS,
) -> bool:
    """
    Validate that the dataset overlaps the specified study area bounds.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset with 'latitude' and 'longitude' coordinates.
    bounds : dict
        Dict with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'.

    Returns
    -------
    bool
        True if valid overlap exists.

    Raises
    ------
    SpatialBoundaryError
        If coordinates are missing or if the dataset has no spatial overlap.
    """
    ds_norm = normalize_coords(ds)

    if "latitude" not in ds_norm.coords or "longitude" not in ds_norm.coords:
        raise SpatialBoundaryError(
            "Dataset missing required spatial coordinates ('latitude', 'longitude'). "
            f"Available coords: {list(ds.coords.keys())}"
        )

    ds_lat_min = float(ds_norm.latitude.min())
    ds_lat_max = float(ds_norm.latitude.max())
    ds_lon_min = float(ds_norm.longitude.min())
    ds_lon_max = float(ds_norm.longitude.max())

    # Check for disjoint coverage
    has_lat_overlap = (ds_lat_max >= bounds["lat_min"]) and (ds_lat_min <= bounds["lat_max"])
    has_lon_overlap = (ds_lon_max >= bounds["lon_min"]) and (ds_lon_min <= bounds["lon_max"])

    if not (has_lat_overlap and has_lon_overlap):
        raise SpatialBoundaryError(
            f"Dataset spatial extent [lat: {ds_lat_min:.3f}..{ds_lat_max:.3f}, "
            f"lon: {ds_lon_min:.3f}..{ds_lon_max:.3f}] does not overlap Kuttanad study area "
            f"[lat: {bounds['lat_min']}..{bounds['lat_max']}, lon: {bounds['lon_min']}..{bounds['lon_max']}]."
        )

    return True


def subset_to_study_area(
    ds: Union[xr.Dataset, xr.DataArray],
    bounds: Dict[str, float] = KUTTANAD_BOUNDS,
    margin_deg: float = 0.005,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Crop dataset to the Kuttanad study area with an optional small border margin.

    Never modifies original dataset (returns a slice copy).
    """
    ds_norm = normalize_coords(ds)
    validate_geographic_bounds(ds_norm, bounds=bounds)

    lat_slice = slice(bounds["lat_min"] - margin_deg, bounds["lat_max"] + margin_deg)
    lon_slice = slice(bounds["lon_min"] - margin_deg, bounds["lon_max"] + margin_deg)

    subset = ds_norm.sel(latitude=lat_slice, longitude=lon_slice).copy()

    if subset.latitude.size == 0 or subset.longitude.size == 0:
        raise SpatialBoundaryError(
            f"Subsetting to bounds {bounds} resulted in an empty spatial grid."
        )

    return subset


# ════════════════════════════════════════════════════════════════════════
#  2. 100 M STANDARD SPATIAL REFERENCE GRID
# ════════════════════════════════════════════════════════════════════════

def get_100m_grid_spec(
    bounds: Dict[str, float] = KUTTANAD_BOUNDS,
    res_deg: float = GRID_RES_DEG,
) -> SpatialGridSpec:
    """Return the reference 100 m SpatialGridSpec for Kuttanad."""
    return SpatialGridSpec(
        lat_min=bounds["lat_min"],
        lat_max=bounds["lat_max"],
        lon_min=bounds["lon_min"],
        lon_max=bounds["lon_max"],
        res_deg=res_deg,
        crs=CRS,
    )


def regrid_to_100m(
    ds: xr.Dataset,
    grid_spec: Optional[SpatialGridSpec] = None,
    method: str = "nearest",
) -> xr.Dataset:
    """
    Resample or interpolate a spatial dataset onto the standard 100 m grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'latitude' and 'longitude' coordinates.
    grid_spec : SpatialGridSpec, optional
        Standard reference grid specification (defaults to 100 m Kuttanad grid).
    method : str
        Interpolation method ('nearest' or 'linear').

    Returns
    -------
    xr.Dataset
        Aligned dataset on the standard 100 m grid.
    """
    if grid_spec is None:
        grid_spec = get_100m_grid_spec()

    ds_norm = normalize_coords(ds)

    # Use xarray interpolation to regrid onto standard coordinates
    regridded = ds_norm.interp(
        latitude=grid_spec.lats,
        longitude=grid_spec.lons,
        method=method,
    )

    regridded.attrs["spatial_resolution_meters"] = GRID_RES_METERS
    regridded.attrs["spatial_resolution_deg"] = grid_spec.res_deg
    regridded.attrs["crs"] = grid_spec.crs
    return regridded


# ════════════════════════════════════════════════════════════════════════
#  3. BACKSCATTER STANDARDISATION (Linear Power to Decibels dB)
# ════════════════════════════════════════════════════════════════════════

def standardise_backscatter(
    values: Union[xr.DataArray, np.ndarray, float],
    is_linear: Optional[bool] = None,
    min_linear: float = 1e-5,
    min_db: float = -45.0,
    max_db: float = 15.0,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Standardise radar backscatter values to decibels (dB).

    Physical logic:
      - If values are linear power reflectance σ⁰ (positive values typically < 2.0),
        they are converted using: σ⁰_dB = 10 * log10(max(σ⁰_linear, min_linear))
      - If values are already in dB (typical range -35 dB to +5 dB), they are preserved.
      - Result is clamped to realistic radar bounds [min_db, max_db] to eliminate -inf.
      - Preserves NaNs / nodata.

    Parameters
    ----------
    values : xr.DataArray, np.ndarray, or float
        Raw backscatter values.
    is_linear : bool, optional
        Explicitly specify whether input is linear. If None, auto-detected.
    min_linear : float
        Floor to prevent log10 of zero or negative numbers.
    min_db : float
        Minimum realistic radar backscatter floor in dB.
    max_db : float
        Maximum realistic radar backscatter ceiling in dB.

    Returns
    -------
    Same type as input with backscatter in dB.
    """
    is_xarray = isinstance(values, xr.DataArray)
    arr = values.values if is_xarray else np.asarray(values)

    # Valid non-nan mask
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        return values

    valid_vals = arr[valid_mask]

    # Auto-detection heuristic if not specified
    if is_linear is None:
        # If all valid values are positive and median is < 2.0, it is linear power
        if np.all(valid_vals >= 0) and np.median(valid_vals) < 2.0:
            is_linear = True
        else:
            # Negative numbers indicate already in decibels
            is_linear = False

    result_arr = np.copy(arr).astype(float)

    if is_linear:
        # Safely convert to dB with positive floor
        clipped_linear = np.where(valid_mask, np.maximum(arr, min_linear), np.nan)
        result_arr = np.where(valid_mask, 10.0 * np.log10(clipped_linear), np.nan)
    else:
        result_arr = np.where(valid_mask, arr, np.nan)

    # Clamp to realistic physical range
    result_arr = np.where(valid_mask, np.clip(result_arr, min_db, max_db), np.nan)

    if is_xarray:
        da_out = values.copy(data=result_arr)
        da_out.attrs["units"] = "dB"
        da_out.attrs["long_name"] = "Normalized Radar Backscatter σ⁰"
        return da_out
    elif np.isscalar(values):
        return float(result_arr)
    return result_arr


# ════════════════════════════════════════════════════════════════════════
#  4. NETCDF & ZARR GENERIC FILE LOADER (Read-Only)
# ════════════════════════════════════════════════════════════════════════

def open_spatial_dataset(
    filepath_or_store: Union[str, Path, xr.Dataset],
    engine: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    Open NetCDF or Zarr datasets safely without modifying or overwriting source files.

    Parameters
    ----------
    filepath_or_store : str, Path, or xr.Dataset
        Target file path, directory (for Zarr), or already-open Dataset.
    engine : str, optional
        'netcdf4' or 'zarr'. If None, inferred from extension / directory.
    chunks : dict, optional
        Chunking dictionary for Dask / out-of-core evaluation if desired.

    Returns
    -------
    xr.Dataset (opened read-only)
    """
    if isinstance(filepath_or_store, xr.Dataset):
        return filepath_or_store.copy(deep=False)

    p = Path(filepath_or_store)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found at: {p.resolve()}")

    # Determine format engine
    if engine is None:
        if p.is_dir() or p.suffix.lower() in [".zarr", ".zr"]:
            engine = "zarr"
        elif p.suffix.lower() in [".nc", ".netcdf", ".nc4"]:
            engine = "netcdf4"
        else:
            # Fallback attempt
            engine = "netcdf4"

    try:
        if engine == "zarr":
            ds = xr.open_zarr(str(p), chunks=chunks)
        else:
            ds = xr.open_dataset(str(p), engine=engine, chunks=chunks)
    except Exception as exc:
        raise DataFormatError(
            f"Failed to open '{p.name}' using engine '{engine}'. Error: {exc}"
        ) from exc

    return normalize_coords(ds)


# ════════════════════════════════════════════════════════════════════════
#  5. SPECIFIC DATA SOURCE LOADERS & PREPROCESSORS
# ════════════════════════════════════════════════════════════════════════

def load_sentinel1(
    data: Union[str, Path, xr.Dataset],
    is_synthetic: bool = False,
    convert_to_db: bool = True,
    subset_kuttanad: bool = True,
) -> xr.Dataset:
    """
    Load and preprocess Sentinel-1 C-band SAR observations.

    Expected input variables:
      - VV backscatter ('vv', 'VV', 'sigma0_vv', 'backscatter_vv')
      - VH backscatter ('vh', 'VH', 'sigma0_vh', 'backscatter_vh')
      - Orbit pass direction ('orbit', 'pass_direction') [optional]

    Standardised output variables:
      - 'vv_db' : C-band VV backscatter in dB
      - 'vh_db' : C-band VH backscatter in dB
      - 'orbit' : Orbit pass string ('ascending' / 'descending') [if present]
    """
    ds = open_spatial_dataset(data)

    # Map variable names
    vv_candidates = ["vv_db", "vv", "VV", "sigma0_vv", "backscatter_vv"]
    vh_candidates = ["vh_db", "vh", "VH", "sigma0_vh", "backscatter_vh"]

    vv_var = next((v for v in vv_candidates if v in ds.data_vars), None)
    vh_var = next((v for v in vh_candidates if v in ds.data_vars), None)

    if vv_var is None and vh_var is None:
        raise IncompatibleDatasetError(
            "Sentinel-1 dataset must contain at least VV or VH polarizations. "
            f"Found data variables: {list(ds.data_vars.keys())}"
        )

    out_vars: Dict[str, xr.DataArray] = {}

    if vv_var:
        vv_da = ds[vv_var]
        if convert_to_db:
            vv_da = standardise_backscatter(vv_da)
        out_vars["vv_db"] = vv_da

    if vh_var:
        vh_da = ds[vh_var]
        if convert_to_db:
            vh_da = standardise_backscatter(vh_da)
        out_vars["vh_db"] = vh_da

    if "orbit" in ds.data_vars:
        out_vars["orbit"] = ds["orbit"]

    result = xr.Dataset(out_vars, coords=ds.coords)

    if subset_kuttanad and "latitude" in result.coords and "longitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs["data_source"] = DataSource.SENTINEL1.value
    result.attrs["sar_band"] = SARBand.C_BAND.value
    result.attrs["is_synthetic"] = int(bool(is_synthetic))
    result.attrs["crs"] = CRS
    result.attrs["description"] = (
        "Sentinel-1 Synthetic Aperture Radar (Simulated)"
        if is_synthetic else "Sentinel-1 Synthetic Aperture Radar"
    )

    return result


def load_nisar(
    data: Union[str, Path, xr.Dataset],
    is_synthetic: bool = False,
    convert_to_db: bool = True,
    band: SARBand = SARBand.L_BAND,
    subset_kuttanad: bool = True,
) -> xr.Dataset:
    """
    Load and preprocess NISAR L-band / S-band SAR observations where available.

    Supports dual-pol or quad-pol channels (HH, HV, VV, VH).

    Standardised output variables:
      - 'hh_db', 'hv_db', 'vv_db', 'vh_db' (as available)
    """
    ds = open_spatial_dataset(data)

    pol_map = {
        "hh_db": ["hh_db", "hh", "HH", "sigma0_hh", "HH_sigma0"],
        "hv_db": ["hv_db", "hv", "HV", "sigma0_hv", "HV_sigma0"],
        "vv_db": ["vv_db", "vv", "VV", "sigma0_vv", "VV_sigma0"],
        "vh_db": ["vh_db", "vh", "VH", "sigma0_vh", "VH_sigma0"],
    }

    out_vars: Dict[str, xr.DataArray] = {}

    for target_name, candidates in pol_map.items():
        matched = next((c for c in candidates if c in ds.data_vars), None)
        if matched:
            da = ds[matched]
            if convert_to_db:
                da = standardise_backscatter(da)
            out_vars[target_name] = da

    if not out_vars:
        raise IncompatibleDatasetError(
            "NISAR dataset must contain at least one valid polarization channel (HH, HV, VV, VH). "
            f"Found data variables: {list(ds.data_vars.keys())}"
        )

    result = xr.Dataset(out_vars, coords=ds.coords)

    if subset_kuttanad and "latitude" in result.coords and "longitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs["data_source"] = DataSource.NISAR.value
    result.attrs["sar_band"] = band.value
    result.attrs["is_synthetic"] = int(bool(is_synthetic))
    result.attrs["crs"] = CRS
    result.attrs["description"] = (
        f"NISAR {band.value}-band SAR observations (Simulated)"
        if is_synthetic else f"NISAR {band.value}-band SAR observations"
    )

    return result


def load_chirps_rainfall(
    data: Union[str, Path, xr.Dataset],
    is_synthetic: bool = False,
    subset_kuttanad: bool = True,
) -> xr.Dataset:
    """
    Load and preprocess CHIRPS precipitation data.

    Expected input variables:
      - 'rainfall_mm', 'precip', 'precipitation', 'rainfall', 'prcp'

    Standardised output variable:
      - 'rainfall_mm' : Daily precipitation depth in millimeters (clipped ≥ 0)
    """
    ds = open_spatial_dataset(data)

    rain_candidates = ["rainfall_mm", "precip", "precipitation", "rainfall", "prcp"]
    matched = next((c for c in rain_candidates if c in ds.data_vars), None)

    if matched is None:
        raise IncompatibleDatasetError(
            "CHIRPS dataset must contain a precipitation variable. "
            f"Found: {list(ds.data_vars.keys())}"
        )

    rain_da = ds[matched].copy()
    # Ensure physically valid precipitation (no negative values)
    rain_da = np.maximum(rain_da, 0.0)
    rain_da.attrs["units"] = "mm/day"
    rain_da.attrs["long_name"] = "Daily Rainfall Depth"

    result = xr.Dataset({"rainfall_mm": rain_da}, coords=ds.coords)

    if subset_kuttanad and "latitude" in result.coords and "longitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs["data_source"] = DataSource.CHIRPS.value
    result.attrs["is_synthetic"] = int(bool(is_synthetic))
    result.attrs["crs"] = CRS
    return result


def load_era5_environmental(
    data: Union[str, Path, xr.Dataset],
    is_synthetic: bool = False,
    subset_kuttanad: bool = True,
) -> xr.Dataset:
    """
    Load and preprocess ERA5 reanalysis environmental covariates (soil moisture, temperature).

    Expected input variables:
      - Soil moisture: 'soil_moisture', 'swvl1', 'volumetric_soil_water_layer_1'
      - Temperature: 'temperature_c', 't2m', 'temperature', 'surface_temperature'

    Standardised output variables:
      - 'soil_moisture' : Volumetric water content in m³/m³ (clipped 0.0 to 1.0)
      - 'temperature_c' : 2-metre air / surface temperature in degrees Celsius (°C)
    """
    ds = open_spatial_dataset(data)

    sm_candidates = ["soil_moisture", "swvl1", "volumetric_soil_water_layer_1"]
    temp_candidates = ["temperature_c", "t2m", "temperature", "surface_temperature"]

    sm_var = next((c for c in sm_candidates if c in ds.data_vars), None)
    temp_var = next((c for c in temp_candidates if c in ds.data_vars), None)

    if sm_var is None and temp_var is None:
        raise IncompatibleDatasetError(
            "ERA5 dataset must contain at least soil moisture or temperature. "
            f"Found: {list(ds.data_vars.keys())}"
        )

    out_vars: Dict[str, xr.DataArray] = {}

    if sm_var:
        sm_da = ds[sm_var].copy()
        # Physical bounds for volumetric soil moisture
        sm_da = np.clip(sm_da, 0.0, 1.0)
        sm_da.attrs["units"] = "m³/m³"
        sm_da.attrs["long_name"] = "Volumetric Soil Moisture"
        out_vars["soil_moisture"] = sm_da

    if temp_var:
        temp_da = ds[temp_var].copy()
        # Check if temperature is in Kelvin (> 100) and convert to Celsius
        finite_vals = temp_da.values[np.isfinite(temp_da.values)]
        if len(finite_vals) > 0 and np.mean(finite_vals) > 100.0:
            temp_da = temp_da - 273.15
        temp_da.attrs["units"] = "°C"
        temp_da.attrs["long_name"] = "Surface Temperature"
        out_vars["temperature_c"] = temp_da

    result = xr.Dataset(out_vars, coords=ds.coords)

    if subset_kuttanad and "latitude" in result.coords and "longitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs["data_source"] = DataSource.ERA5.value
    result.attrs["is_synthetic"] = int(bool(is_synthetic))
    result.attrs["crs"] = CRS
    return result


# ════════════════════════════════════════════════════════════════════════
#  6. TEMPORAL ALIGNMENT & ANTECEDENT RAINFALL AGGREGATION
# ════════════════════════════════════════════════════════════════════════

def align_temporal_datasets(
    sar_ds: xr.Dataset,
    env_ds: xr.Dataset,
    max_tolerance_days: int = 5,
    compute_antecedent_rainfall: bool = True,
) -> xr.Dataset:
    """
    Temporally align environmental covariates (rainfall, soil moisture, temperature)
    with SAR satellite observation timestamps.

    Also calculates 3-day and 7-day cumulative antecedent rainfall prior to each
    SAR acquisition date to account for hydrological delay.

    Parameters
    ----------
    sar_ds : xr.Dataset
        Target SAR observations containing a 'time' dimension.
    env_ds : xr.Dataset
        Daily environmental data containing a 'time' dimension.
    max_tolerance_days : int
        Maximum allowable day difference when snapping environmental data.
    compute_antecedent_rainfall : bool
        Whether to compute 3-day and 7-day antecedent rainfall sums.

    Returns
    -------
    xr.Dataset
        Unified dataset with SAR channels and temporally aligned environmental covariates.
    """
    sar_norm = normalize_coords(sar_ds)
    env_norm = normalize_coords(env_ds)

    if "time" not in sar_norm.coords or "time" not in env_norm.coords:
        raise TemporalAlignmentError("Both SAR and environmental datasets must have a 'time' coordinate.")

    sar_times = pd.to_datetime(sar_norm.time.values)
    env_times = pd.to_datetime(env_norm.time.values)

    # Compute antecedent rainfall metrics on daily environmental series if available
    extra_vars: Dict[str, xr.DataArray] = {}
    if compute_antecedent_rainfall and "rainfall_mm" in env_norm.data_vars:
        # 3-day rolling sum
        rain_3d = env_norm["rainfall_mm"].rolling(time=3, min_periods=1).sum()
        rain_3d.attrs["units"] = "mm"
        rain_3d.attrs["long_name"] = "3-Day Antecedent Cumulative Rainfall"
        env_norm["antecedent_rain_3d_mm"] = rain_3d

        # 7-day rolling sum
        rain_7d = env_norm["rainfall_mm"].rolling(time=7, min_periods=1).sum()
        rain_7d.attrs["units"] = "mm"
        rain_7d.attrs["long_name"] = "7-Day Antecedent Cumulative Rainfall"
        env_norm["antecedent_rain_7d_mm"] = rain_7d

    # Reindex or interpolate environmental covariates to SAR observation dates
    aligned_env = env_norm.reindex(
        time=sar_norm.time,
        method="nearest",
        tolerance=np.timedelta64(max_tolerance_days, "D"),
    )

    # Merge into a single harmonized dataset
    merged = xr.merge([sar_norm, aligned_env], compat="override")

    merged.attrs["temporally_aligned"] = 1
    merged.attrs["alignment_tolerance_days"] = max_tolerance_days
    merged.attrs["is_synthetic"] = int(bool(
        sar_ds.attrs.get("is_synthetic", False) or env_ds.attrs.get("is_synthetic", False)
    ))

    return merged


# ════════════════════════════════════════════════════════════════════════
#  7. MISSING DATA & SAFETY HANDLING
# ════════════════════════════════════════════════════════════════════════

def handle_missing_observations(
    ds: xr.Dataset,
    fill_method: str = "interpolate",
    max_gap_days: int = 36,
) -> xr.Dataset:
    """
    Detect, report, and optionally interpolate missing observations along time.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with time dimension.
    fill_method : str
        'interpolate' (linear time interpolation), 'ffill' (forward fill), or 'none'.
    max_gap_days : int
        Maximum gap length to interpolate over.

    Returns
    -------
    xr.Dataset with handled missing values.
    """
    ds_out = ds.copy(deep=True)

    if "time" not in ds_out.coords or ds_out.time.size < 2:
        return ds_out

    if fill_method == "interpolate":
        for var in ds_out.data_vars:
            if np.issubdtype(ds_out[var].dtype, np.number):
                ds_out[var] = ds_out[var].interpolate_na(
                    dim="time",
                    method="linear",
                    max_gap=np.timedelta64(max_gap_days, "D"),
                )
    elif fill_method == "ffill":
        ds_out = ds_out.ffill(dim="time")

    return ds_out


# ════════════════════════════════════════════════════════════════════════
#  8. SYNTHETIC INTEGRATION BRIDGE (app/mock_data.py Integration)
# ════════════════════════════════════════════════════════════════════════

def create_synthetic_datacube(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    regrid_100m: bool = False,
) -> xr.Dataset:
    """
    Build a unified xarray Dataset cube from app/mock_data.py.

    Creates an end-to-end multi-variable data cube combining:
      - Sentinel-1 synthetic SAR backscatter (VV and VH in dB)
      - CHIRPS synthetic daily rainfall (mm/day)
      - ERA5 synthetic soil moisture (m³/m³) and temperature (°C)
      - Antecedent rainfall metrics (3-day and 7-day sums)

    Clearly tagged with is_synthetic=True at the dataset and variable levels.
    """
    from app.mock_data import (
        generate_environmental_timeseries,
        generate_sar_timeseries,
        generate_spatial_backscatter,
    )

    sar_df = generate_sar_timeseries(start_date=start_date, end_date=end_date)
    env_df = generate_environmental_timeseries(start_date=start_date, end_date=end_date)

    # Convert SAR DataFrame to xarray
    sar_time_np = pd.to_datetime(sar_df["date"]).to_numpy(dtype="datetime64[ns]")
    sar_ds = xr.Dataset(
        data_vars={
            "vv_db": ("time", np.asarray(sar_df["vv_db"].values, dtype=float), {"units": "dB"}),
            "vh_db": ("time", np.asarray(sar_df["vh_db"].values, dtype=float), {"units": "dB"}),
            "orbit": ("time", np.asarray(sar_df["orbit"].values, dtype=str)),
        },
        coords={"time": sar_time_np},
        attrs={
            "data_source": DataSource.SENTINEL1.value,
            "is_synthetic": 1,
            "sar_band": SARBand.C_BAND.value,
            "crs": CRS,
        },
    )

    # Convert Environmental DataFrame to xarray
    env_time_np = pd.to_datetime(env_df["date"]).to_numpy(dtype="datetime64[ns]")
    env_ds = xr.Dataset(
        data_vars={
            "rainfall_mm": ("time", np.asarray(env_df["rainfall_mm"].values, dtype=float), {"units": "mm/day"}),
            "soil_moisture": ("time", np.asarray(env_df["soil_moisture"].values, dtype=float), {"units": "m³/m³"}),
            "temperature_c": ("time", np.asarray(env_df["temperature_c"].values, dtype=float), {"units": "°C"}),
        },
        coords={"time": env_time_np},
        attrs={
            "data_source": "environmental_reanalysis",
            "is_synthetic": 1,
            "crs": CRS,
        },
    )

    # Perform temporal alignment and antecedent rainfall computation
    aligned = align_temporal_datasets(
        sar_ds=sar_ds,
        env_ds=env_ds,
        max_tolerance_days=5,
        compute_antecedent_rainfall=True,
    )

    # Include spatial grid representation from mock_data spatial backscatter
    spatial_sample = generate_spatial_backscatter()
    aligned.coords["latitude"] = spatial_sample.latitude.values
    aligned.coords["longitude"] = spatial_sample.longitude.values

    if regrid_100m:
        aligned = regrid_to_100m(aligned)

    aligned.attrs["title"] = "CounterSAR Unified Preprocessed Data Cube"
    aligned.attrs["study_area"] = "Kuttanad, Kerala, India"
    aligned.attrs["is_synthetic"] = 1
    aligned.attrs["notice"] = "⚠️ SYNTHETIC DATA — Not real satellite observations"

    return aligned


# ════════════════════════════════════════════════════════════════════════
#  9. EXPORT & STORAGE (data/processed/)
# ════════════════════════════════════════════════════════════════════════

def save_processed_datacube(
    ds: xr.Dataset,
    filepath: Union[str, Path],
    format: Optional[str] = None,
) -> Path:
    """
    Save preprocessed datacube to data/processed/ in NetCDF or Zarr format.

    Ensures all attributes are NetCDF/Zarr compliant (converting boolean
    flags to 1/0 integers) and preserves raw data by strictly writing
    to the designated processed destination.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset to persist.
    filepath : str or Path
        Target filepath (e.g. 'data/processed/kuttanad_cube.nc').
    format : str, optional
        'netcdf' or 'zarr'. If None, inferred from filepath suffix.

    Returns
    -------
    Path to the saved dataset.
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize attributes for NetCDF compatibility
    clean_ds = ds.copy(deep=False)
    for k, v in list(clean_ds.attrs.items()):
        if isinstance(v, bool):
            clean_ds.attrs[k] = int(v)

    for var in clean_ds.data_vars:
        for k, v in list(clean_ds[var].attrs.items()):
            if isinstance(v, bool):
                clean_ds[var].attrs[k] = int(v)

    if format is None:
        format = "zarr" if p.suffix.lower() == ".zarr" else "netcdf"

    if format == "zarr":
        clean_ds.to_zarr(str(p), mode="w")
    else:
        clean_ds.to_netcdf(str(p), engine="netcdf4")

    return p

