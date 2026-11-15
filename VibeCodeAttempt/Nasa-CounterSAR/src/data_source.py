from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from src.skeleton import (
    CRS,
    GEOGRAPHIC_CRS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GRID_RES_METERS,
    KUTTANAD_BOUNDS,
    DataFormatError,
    DataSource,
    IncompatibleDatasetError,
    SARBand,
    SpatialBoundaryError,
    SpatialGridSpec,
    TemporalAlignmentError,
)


COORD_ALIASES = {
    "latitude": ["latitude", "lat", "LATITUDE", "Lat"],
    "longitude": ["longitude", "lon", "long", "LONGITUDE", "Lon"],
    "time": ["time", "date", "datetime", "TIME", "Date"],
}


def normalize_coords(ds):
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

    for name in ("latitude", "longitude", "x", "y", "time"):
        if name in ds.coords and ds.coords[name].ndim == 1:
            values = ds.coords[name].values
            if len(values) > 1 and np.any(values[:-1] > values[1:]):
                ds = ds.sortby(name)

    return ds


def _get_crs(ds):
    value = ds.attrs.get("crs")

    if value is None:
        value = ds.attrs.get("spatial_ref")

    if value is None:
        for var in ds.variables.values():
            value = var.attrs.get("crs")
            if value:
                break

    if value is None:
        if "latitude" in ds.coords and "longitude" in ds.coords:
            return GEOGRAPHIC_CRS

        raise SpatialBoundaryError("Dataset CRS is missing.")

    return str(value)


def validate_geographic_bounds(ds, bounds=KUTTANAD_BOUNDS):
    ds = normalize_coords(ds)

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise SpatialBoundaryError(
            "Geographic latitude and longitude coordinates are required."
        )

    lat = ds.latitude.values
    lon = ds.longitude.values

    if lat.size == 0 or lon.size == 0:
        raise SpatialBoundaryError("Dataset has empty spatial coordinates.")

    overlaps_lat = (
        np.nanmax(lat) >= bounds["lat_min"]
        and np.nanmin(lat) <= bounds["lat_max"]
    )

    overlaps_lon = (
        np.nanmax(lon) >= bounds["lon_min"]
        and np.nanmin(lon) <= bounds["lon_max"]
    )

    if not overlaps_lat or not overlaps_lon:
        raise SpatialBoundaryError("Dataset does not overlap the Kuttanad study area.")

    return True


def subset_to_study_area(ds, bounds=KUTTANAD_BOUNDS, margin_deg=0.005):
    ds = normalize_coords(ds)

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise SpatialBoundaryError(
            "Cannot geographically subset a dataset without latitude and longitude."
        )

    validate_geographic_bounds(ds, bounds)

    subset = ds.sel(
        latitude=slice(
            bounds["lat_min"] - margin_deg,
            bounds["lat_max"] + margin_deg
        ),
        longitude=slice(
            bounds["lon_min"] - margin_deg,
            bounds["lon_max"] + margin_deg
        )
    )

    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        raise SpatialBoundaryError("Subsetting produced an empty dataset.")

    return subset.copy()


def get_100m_grid_spec(bounds=KUTTANAD_BOUNDS):
    return SpatialGridSpec(
        lat_min=bounds["lat_min"],
        lat_max=bounds["lat_max"],
        lon_min=bounds["lon_min"],
        lon_max=bounds["lon_max"],
        resolution_m=GRID_RES_METERS,
        crs=CRS,
    )


def _source_transform(ds):
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat = ds.latitude.values
        lon = ds.longitude.values

        if len(lat) < 2 or len(lon) < 2:
            raise SpatialBoundaryError("At least two coordinates are required per spatial axis.")

        dx = float(np.median(np.diff(lon)))
        dy = float(np.median(np.diff(lat)))

        return from_origin(
            float(lon.min()) - abs(dx) / 2,
            float(lat.max()) + abs(dy) / 2,
            abs(dx),
            abs(dy)
        )

    if "x" in ds.coords and "y" in ds.coords:
        x = ds.x.values
        y = ds.y.values

        if len(x) < 2 or len(y) < 2:
            raise SpatialBoundaryError("At least two coordinates are required per spatial axis.")

        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))

        return from_origin(
            float(x.min()) - abs(dx) / 2,
            float(y.max()) + abs(dy) / 2,
            abs(dx),
            abs(dy)
        )

    raise SpatialBoundaryError("Dataset has no supported spatial coordinate pair.")


def regrid_to_100m(ds, grid_spec=None, method="nearest", categorical_vars=None):
    if grid_spec is None:
        grid_spec = get_100m_grid_spec()

    if method not in ("nearest", "linear"):
        raise ValueError("method must be 'nearest' or 'linear'.")

    ds = normalize_coords(ds)
    source_crs = _get_crs(ds)

    if "latitude" in ds.coords and "longitude" in ds.coords:
        y_dim, x_dim = "latitude", "longitude"
    elif "y" in ds.coords and "x" in ds.coords:
        y_dim, x_dim = "y", "x"
    else:
        raise SpatialBoundaryError("Dataset requires latitude/longitude or x/y coordinates.")

    if ds[y_dim].ndim != 1 or ds[x_dim].ndim != 1:
        raise SpatialBoundaryError("Only one-dimensional rectilinear coordinates are supported.")

    source_transform = _source_transform(ds)

    target_transform = from_origin(
        grid_spec.x[0] - grid_spec.resolution_m / 2,
        grid_spec.y[-1] + grid_spec.resolution_m / 2,
        grid_spec.resolution_m,
        grid_spec.resolution_m
    )

    height, width = grid_spec.shape
    categorical_vars = set(categorical_vars or [])
    output = {}

    for name, da in ds.data_vars.items():
        if y_dim not in da.dims or x_dim not in da.dims:
            output[name] = da
            continue

        other_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
        ordered = da.transpose(*other_dims, y_dim, x_dim)
        values = np.asarray(ordered.values)

        result_shape = values.shape[:-2] + (height, width)
        result = np.full(result_shape, np.nan, dtype=np.float32)

        resampling = (
            Resampling.nearest
            if method == "nearest" or name in categorical_vars
            else Resampling.bilinear
        )

        for index in np.ndindex(values.shape[:-2]):
            source = np.asarray(values[index], dtype=np.float32)
            source = np.flip(source, axis=0)

            reproject(
                source=source,
                destination=result[index],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=target_transform,
                dst_crs=grid_spec.crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=resampling,
            )

        result = np.flip(result, axis=-2)

        coords = {
            dim: ordered.coords[dim]
            for dim in other_dims
            if dim in ordered.coords
        }

        coords["y"] = grid_spec.y
        coords["x"] = grid_spec.x

        output[name] = xr.DataArray(
            result,
            dims=other_dims + ["y", "x"],
            coords=coords,
            attrs=da.attrs
        )

    result_ds = xr.Dataset(output)

    for name, coord in ds.coords.items():
        if name not in result_ds.coords and name not in (y_dim, x_dim):
            if coord.dims and all(d in result_ds.dims for d in coord.dims):
                result_ds = result_ds.assign_coords({name: coord})

    result_ds.attrs = dict(ds.attrs)
    result_ds.attrs["crs"] = grid_spec.crs
    result_ds.attrs["spatial_resolution_meters"] = grid_spec.resolution_m
    result_ds.attrs["grid_type"] = "projected_metric"

    return result_ds


def standardise_backscatter(
    values,
    is_linear=None,
    min_linear=1e-5,
    min_db=-45.0,
    max_db=15.0,
):
    is_xarray = isinstance(values, xr.DataArray)
    arr = values.values if is_xarray else np.asarray(values, dtype=float)

    if is_linear is None:
        units = values.attrs.get("units", "").lower() if is_xarray else ""

        if units in ("db", "decibel", "decibels"):
            is_linear = False
        elif units in ("linear", "power", "sigma0", "linear_power"):
            is_linear = True
        else:
            raise ValueError(
                "Backscatter units are ambiguous. Specify is_linear=True or False."
            )

    valid = np.isfinite(arr)

    if is_linear and np.any(arr[valid] < 0):
        raise ValueError("Linear backscatter power cannot contain negative values.")

    result = np.full(arr.shape, np.nan, dtype=float)

    if is_linear:
        result[valid] = 10.0 * np.log10(np.maximum(arr[valid], min_linear))
    else:
        result[valid] = arr[valid]

    result[valid] = np.clip(result[valid], min_db, max_db)

    if is_xarray:
        output = values.copy(data=result)
        output.attrs = dict(values.attrs)
        output.attrs["units"] = "dB"
        output.attrs["long_name"] = "Normalized Radar Backscatter"
        return output

    if np.isscalar(values):
        return float(result)

    return result


def open_spatial_dataset(filepath_or_store, engine=None, chunks=None):
    if isinstance(filepath_or_store, xr.Dataset):
        return filepath_or_store.copy(deep=False)

    path = Path(filepath_or_store)

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path.resolve()}")

    if engine is None:
        engine = "zarr" if path.is_dir() or path.suffix.lower() == ".zarr" else "netcdf4"

    try:
        if engine == "zarr":
            ds = xr.open_zarr(str(path), chunks=chunks)
        else:
            ds = xr.open_dataset(str(path), engine=engine, chunks=chunks)
    except Exception as exc:
        raise DataFormatError(
            f"Unable to open {path.name}: {exc}"
        ) from exc

    return normalize_coords(ds)


def _prepare_sar(data, candidates, output_names, source, band, is_synthetic, convert_to_db, subset_kuttanad):
    ds = open_spatial_dataset(data)
    out = {}

    for target, names in zip(output_names, candidates):
        matched = next((name for name in names if name in ds.data_vars), None)

        if matched is None:
            continue

        da = ds[matched]

        if convert_to_db:
            da = standardise_backscatter(da)

        out[target] = da

    if not out:
        raise IncompatibleDatasetError(
            f"No supported polarization variables found. Available: {list(ds.data_vars)}"
        )

    result = xr.Dataset(out)

    for name, coord in ds.coords.items():
        if name not in result.coords and coord.dims and all(d in result.dims for d in coord.dims):
            result = result.assign_coords({name: coord})

    if subset_kuttanad and "latitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs = dict(ds.attrs)
    result.attrs.update({
        "data_source": source.value,
        "sar_band": band.value,
        "is_synthetic": int(bool(is_synthetic)),
        "description": "Synthetic SAR observations" if is_synthetic else "SAR observations",
    })

    return result


def load_sentinel1(data, is_synthetic=False, convert_to_db=True, subset_kuttanad=True):
    candidates = [
        ["vv_db", "vv", "VV", "sigma0_vv", "backscatter_vv"],
        ["vh_db", "vh", "VH", "sigma0_vh", "backscatter_vh"],
    ]

    return _prepare_sar(
        data,
        candidates,
        ["vv_db", "vh_db"],
        DataSource.SENTINEL1,
        SARBand.C_BAND,
        is_synthetic,
        convert_to_db,
        subset_kuttanad,
    )


def load_nisar(data, is_synthetic=False, convert_to_db=True, band=SARBand.L_BAND, subset_kuttanad=True):
    candidates = [
        ["hh_db", "hh", "HH", "sigma0_hh", "HH_sigma0"],
        ["hv_db", "hv", "HV", "sigma0_hv", "HV_sigma0"],
        ["vv_db", "vv", "VV", "sigma0_vv", "VV_sigma0"],
        ["vh_db", "vh", "VH", "sigma0_vh", "VH_sigma0"],
    ]

    return _prepare_sar(
        data,
        candidates,
        ["hh_db", "hv_db", "vv_db", "vh_db"],
        DataSource.NISAR,
        band,
        is_synthetic,
        convert_to_db,
        subset_kuttanad,
    )


def load_chirps_rainfall(data, is_synthetic=False, subset_kuttanad=True):
    ds = open_spatial_dataset(data)

    candidates = ["rainfall_mm", "precip", "precipitation", "rainfall", "prcp"]
    matched = next((name for name in candidates if name in ds.data_vars), None)

    if matched is None:
        raise IncompatibleDatasetError("No rainfall variable found.")

    rain = ds[matched].copy()

    if bool((rain < 0).any()):
        raise ValueError("Rainfall contains negative values; inspect source units and nodata values.")

    rain.attrs = dict(rain.attrs)
    rain.attrs["units"] = "mm/day"

    result = rain.to_dataset(name="rainfall_mm")

    if subset_kuttanad and "latitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs.update({
        "data_source": DataSource.CHIRPS.value,
        "is_synthetic": int(bool(is_synthetic)),
    })

    return result


def load_era5_environmental(data, is_synthetic=False, subset_kuttanad=True):
    ds = open_spatial_dataset(data)

    output = {}

    sm_candidates = ["soil_moisture", "swvl1", "volumetric_soil_water_layer_1"]
    temp_candidates = ["temperature_c", "t2m", "temperature", "surface_temperature"]

    sm_name = next((name for name in sm_candidates if name in ds.data_vars), None)
    temp_name = next((name for name in temp_candidates if name in ds.data_vars), None)

    if sm_name:
        sm = ds[sm_name].copy()

        if bool(((sm < 0) | (sm > 1)).any()):
            raise ValueError("Soil moisture values outside the expected 0–1 range.")

        sm.attrs = dict(sm.attrs)
        sm.attrs["units"] = "m³/m³"
        output["soil_moisture"] = sm

    if temp_name:
        temp = ds[temp_name].copy()
        units = temp.attrs.get("units", "").lower()

        if units in ("k", "kelvin"):
            temp = temp - 273.15
        elif units not in ("c", "°c", "degc", "degrees_celsius", "degree_celsius"):
            raise ValueError("Temperature units must be specified as Kelvin or Celsius.")

        temp.attrs = dict(temp.attrs)
        temp.attrs["units"] = "°C"
        output["temperature_c"] = temp

    if not output:
        raise IncompatibleDatasetError("No supported ERA5 environmental variables found.")

    result = xr.Dataset(output)

    for name, coord in ds.coords.items():
        if name not in result.coords and coord.dims and all(d in result.dims for d in coord.dims):
            result = result.assign_coords({name: coord})

    if subset_kuttanad and "latitude" in result.coords:
        result = subset_to_study_area(result)

    result.attrs.update({
        "data_source": DataSource.ERA5.value,
        "is_synthetic": int(bool(is_synthetic)),
    })

    return result


def align_temporal_datasets(
    sar_ds,
    env_ds,
    max_tolerance_days=5,
    compute_antecedent_rainfall=True,
):
    sar_ds = normalize_coords(sar_ds)
    env_ds = normalize_coords(env_ds)

    if "time" not in sar_ds.coords or "time" not in env_ds.coords:
        raise TemporalAlignmentError("Both datasets require a time coordinate.")

    sar_ds = sar_ds.sortby("time")
    env_ds = env_ds.sortby("time")

    if not np.all(np.diff(env_ds.time.values) > np.timedelta64(0, "ns")):
        raise TemporalAlignmentError("Environmental timestamps must be unique and increasing.")

    if compute_antecedent_rainfall and "rainfall_mm" in env_ds:
        rain = env_ds["rainfall_mm"]

        rain_3d = rain.rolling(
            time=3,
            min_periods=3
        ).sum().shift(time=1)

        rain_7d = rain.rolling(
            time=7,
            min_periods=7
        ).sum().shift(time=1)

        rain_3d.attrs = {
            "units": "mm",
            "long_name": "Rainfall during the three days preceding acquisition",
        }

        rain_7d.attrs = {
            "units": "mm",
            "long_name": "Rainfall during the seven days preceding acquisition",
        }

        env_ds = env_ds.assign({
            "antecedent_rain_3d_mm": rain_3d,
            "antecedent_rain_7d_mm": rain_7d,
        })

    tolerance = np.timedelta64(max_tolerance_days, "D")

    aligned_env = env_ds.reindex(
        time=sar_ds.time,
        method="nearest",
        tolerance=tolerance,
    )

    try:
        merged = xr.merge(
            [sar_ds, aligned_env],
            compat="no_conflicts",
            join="exact",
        )
    except Exception as exc:
        raise TemporalAlignmentError(
            f"SAR and environmental grids are incompatible: {exc}"
        ) from exc

    merged.attrs = dict(sar_ds.attrs)
    merged.attrs["temporally_aligned"] = 1
    merged.attrs["alignment_tolerance_days"] = max_tolerance_days
    merged.attrs["is_synthetic"] = int(
        bool(sar_ds.attrs.get("is_synthetic", False))
        or bool(env_ds.attrs.get("is_synthetic", False))
    )

    return merged


def handle_missing_observations(ds, fill_method="none", max_gap_days=36):
    if fill_method not in ("none", "interpolate", "ffill"):
        raise ValueError("fill_method must be none, interpolate, or ffill.")

    output = ds.copy(deep=False)

    if "time" not in output.coords or output.time.size < 2:
        return output

    if fill_method == "interpolate":
        for name in output.data_vars:
            da = output[name]

            if "time" not in da.dims or not np.issubdtype(da.dtype, np.number):
                continue

            output[name] = da.interpolate_na(
                dim="time",
                method="linear",
                max_gap=np.timedelta64(max_gap_days, "D"),
            )

    elif fill_method == "ffill":
        output = output.ffill(dim="time")

    return output


def create_synthetic_datacube(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    regrid_100m=False,
):
    from app.mock_data import (
        generate_environmental_timeseries,
        generate_sar_timeseries,
    )

    sar_df = generate_sar_timeseries(
        start_date=start_date,
        end_date=end_date,
    )

    env_df = generate_environmental_timeseries(
        start_date=start_date,
        end_date=end_date,
    )

    sar_times = pd.to_datetime(sar_df["date"]).to_numpy()
    env_times = pd.to_datetime(env_df["date"]).to_numpy()

    sar_ds = xr.Dataset(
        {
            "vv_db": ("time", sar_df["vv_db"].to_numpy(dtype=float)),
            "vh_db": ("time", sar_df["vh_db"].to_numpy(dtype=float)),
            "orbit": ("time", sar_df["orbit"].astype(str).to_numpy()),
        },
        coords={"time": sar_times},
        attrs={
            "data_source": DataSource.SYNTHETIC.value,
            "is_synthetic": 1,
            "crs": CRS,
        },
    )

    env_ds = xr.Dataset(
        {
            "rainfall_mm": ("time", env_df["rainfall_mm"].to_numpy(dtype=float)),
            "soil_moisture": ("time", env_df["soil_moisture"].to_numpy(dtype=float)),
            "temperature_c": ("time", env_df["temperature_c"].to_numpy(dtype=float)),
        },
        coords={"time": env_times},
        attrs={
            "data_source": DataSource.SYNTHETIC.value,
            "is_synthetic": 1,
            "crs": CRS,
        },
    )

    result = align_temporal_datasets(sar_ds, env_ds)

    result.attrs.update({
        "title": "CounterSAR Synthetic Demonstration Dataset",
        "study_area": "Kuttanad, Kerala, India",
        "is_synthetic": 1,
        "notice": "Synthetic data. Not real satellite observations.",
    })

    if regrid_100m:
        raise ValueError(
            "Spatial synthetic generation will be enabled after mock_data.py is updated."
        )

    return result


def save_processed_datacube(ds, filepath, format=None):
    path = Path(filepath)

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing processed data: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = "zarr" if path.suffix.lower() == ".zarr" else "netcdf"

    if format not in ("netcdf", "zarr"):
        raise ValueError("format must be netcdf or zarr.")

    clean_ds = ds.copy(deep=False)

    for key, value in list(clean_ds.attrs.items()):
        if isinstance(value, bool):
            clean_ds.attrs[key] = int(value)

    for variable in clean_ds.data_vars:
        for key, value in list(clean_ds[variable].attrs.items()):
            if isinstance(value, bool):
                clean_ds[variable].attrs[key] = int(value)

    if format == "zarr":
        clean_ds.to_zarr(str(path), mode="w")
    else:
        clean_ds.to_netcdf(str(path), engine="netcdf4")

    return path