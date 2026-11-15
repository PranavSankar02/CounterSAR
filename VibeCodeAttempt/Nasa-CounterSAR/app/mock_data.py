import numpy as np
import pandas as pd
import xarray as xr


SYNTHETIC_LABEL = "SYNTHETIC DATA — Not real satellite observations"

KUTTANAD_BBOX = {
    "lat_min": 9.25,
    "lat_max": 9.58,
    "lon_min": 76.30,
    "lon_max": 76.55,
}

SPATIAL_RES = 0.0009
S1_REVISIT_DAYS = 12
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-06-30"
RNG_SEED = 42


def _seasonal_backscatter(doy):
    base = -8.0
    sw_dip = -7.0 * np.exp(-0.5 * ((doy - 200) / 40) ** 2)
    ne_dip = -2.5 * np.exp(-0.5 * ((doy - 290) / 20) ** 2)
    return base + sw_dip + ne_dip


def _seasonal_rainfall(doy):
    sw = 20.0 * np.exp(-0.5 * ((doy - 180) / 35) ** 2)
    ne = 8.0 * np.exp(-0.5 * ((doy - 300) / 25) ** 2)
    return 1.5 + sw + ne


def _seasonal_soil_moisture(doy):
    base = 0.25
    wet = 0.35 * np.exp(-0.5 * ((doy - 210) / 50) ** 2)
    return np.clip(base + wet, 0.15, 0.60)


def generate_sar_timeseries(
    start_date=DEFAULT_START,
    end_date=DEFAULT_END,
    seed=RNG_SEED,
):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq=f"{S1_REVISIT_DAYS}D")
    doy = dates.dayofyear.values.astype(float)

    vv_db = _seasonal_backscatter(doy) + rng.normal(0, 0.8, len(dates))
    vh_db = vv_db + rng.normal(-7.0, 0.5, len(dates))
    orbits = np.where(np.arange(len(dates)) % 2 == 0, "ascending", "descending")

    return pd.DataFrame({
        "date": dates,
        "vv_db": np.round(vv_db, 2),
        "vh_db": np.round(vh_db, 2),
        "orbit": orbits,
        "is_synthetic": True,
        "data_label": SYNTHETIC_LABEL,
    })


def generate_environmental_timeseries(
    start_date=DEFAULT_START,
    end_date=DEFAULT_END,
    seed=RNG_SEED,
):
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range(start_date, end_date, freq="D")
    doy = dates.dayofyear.values.astype(float)

    rain_scale = np.maximum(_seasonal_rainfall(doy) / 2.0, 0.1)
    rainfall = np.round(np.clip(rng.gamma(2.0, rain_scale), 0, 120), 1)

    sm = _seasonal_soil_moisture(doy) + rng.normal(0, 0.03, len(dates))
    soil_moisture = np.round(np.clip(sm, 0.10, 0.65), 3)

    temp_base = 28.0
    temp_seasonal = -2.0 * np.cos(2 * np.pi * doy / 365)
    temp_monsoon = -1.5 * np.exp(-0.5 * ((doy - 200) / 50) ** 2)

    temperature = np.round(
        temp_base + temp_seasonal + temp_monsoon + rng.normal(0, 1.0, len(dates)),
        1,
    )

    return pd.DataFrame({
        "date": dates,
        "rainfall_mm": rainfall,
        "soil_moisture": soil_moisture,
        "temperature_c": temperature,
        "is_synthetic": True,
        "data_label": SYNTHETIC_LABEL,
    })


def generate_spatial_backscatter(
    target_date="2023-07-15",
    seed=RNG_SEED,
):
    rng = np.random.default_rng(seed + 2)

    lats = np.arange(
        KUTTANAD_BBOX["lat_min"],
        KUTTANAD_BBOX["lat_max"],
        SPATIAL_RES,
    )

    lons = np.arange(
        KUTTANAD_BBOX["lon_min"],
        KUTTANAD_BBOX["lon_max"],
        SPATIAL_RES,
    )

    doy = float(pd.Timestamp(target_date).dayofyear)
    base_vv = _seasonal_backscatter(np.array([doy]))[0]

    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")

    water = np.sin(5 * (lat_g - 9.3)) * np.cos(8 * (lon_g - 76.35))
    water_effect = np.where(water > 0.6, -6.0, 0.0)

    urban = 3.0 * np.exp(
        -0.5 * (((lat_g - 9.49) / 0.03) ** 2 + ((lon_g - 76.34) / 0.03) ** 2)
    )

    noise = rng.normal(0, 1.5, (len(lats), len(lons)))
    vv = base_vv + water_effect + urban + noise
    vh = vv + rng.normal(-7.0, 0.5, vv.shape)

    return xr.Dataset(
        {
            "vv_db": (["latitude", "longitude"], np.round(vv, 2)),
            "vh_db": (["latitude", "longitude"], np.round(vh, 2)),
        },
        coords={"latitude": lats, "longitude": lons},
        attrs={
            "title": "CounterSAR Synthetic SAR Backscatter",
            "data_type": "SYNTHETIC",
            "is_synthetic": True,
            "warning": SYNTHETIC_LABEL,
            "study_area": "Kuttanad, Kerala, India",
            "date": target_date,
            "crs": "EPSG:4326",
            "units": "dB",
            "simulated_source": "Sentinel-1 C-band IW mode",
            "spatial_resolution_deg": SPATIAL_RES,
        },
    )


def generate_anomaly_candidates(
    sar_df,
    n_anomalies=5,
    seed=RNG_SEED,
):
    rng = np.random.default_rng(seed + 3)
    n = min(n_anomalies, len(sar_df))

    if n == 0:
        return pd.DataFrame(columns=[
            "date", "latitude", "longitude", "z_score",
            "anomaly_type", "persistent", "is_synthetic", "data_label"
        ])

    idx = rng.choice(len(sar_df), size=n, replace=False)
    dates = sar_df.iloc[idx]["date"].values

    types = [
        "unexplained_drop",
        "unexplained_rise",
        "persistent_change",
        "transient_spike",
        "gradual_shift",
    ]

    rows = []

    for i, dt in enumerate(dates):
        rows.append({
            "date": pd.Timestamp(dt),
            "latitude": round(
                rng.uniform(KUTTANAD_BBOX["lat_min"], KUTTANAD_BBOX["lat_max"]), 4
            ),
            "longitude": round(
                rng.uniform(KUTTANAD_BBOX["lon_min"], KUTTANAD_BBOX["lon_max"]), 4
            ),
            "z_score": round(rng.uniform(2.5, 5.0), 2),
            "anomaly_type": types[i % len(types)],
            "persistent": bool(rng.random() > 0.4),
            "is_synthetic": True,
            "data_label": SYNTHETIC_LABEL,
        })

    return pd.DataFrame(rows)


def generate_spatial_datacube(
    start_date=DEFAULT_START,
    end_date=DEFAULT_END,
    seed=RNG_SEED,
    grid_spec=None,
):
    from src.data_source import get_100m_grid_spec, regrid_to_100m

    rng = np.random.default_rng(seed + 10)

    if grid_spec is None:
        grid_spec = get_100m_grid_spec()

    dates = pd.date_range(
        start_date,
        end_date,
        freq=f"{S1_REVISIT_DAYS}D",
    )

    if len(dates) < 2:
        raise ValueError("At least two acquisition dates are required.")

    lats = np.arange(
        KUTTANAD_BBOX["lat_min"],
        KUTTANAD_BBOX["lat_max"],
        SPATIAL_RES,
    )

    lons = np.arange(
        KUTTANAD_BBOX["lon_min"],
        KUTTANAD_BBOX["lon_max"],
        SPATIAL_RES,
    )

    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")

    water_pattern = (
        np.sin(5 * (lat_g - 9.3))
        * np.cos(8 * (lon_g - 76.35))
    )

    water_mask = water_pattern > 0.6

    paddy_pattern = (
        np.sin(35 * (lat_g - 9.25))
        + np.cos(30 * (lon_g - 76.3))
    )

    paddy_mask = (paddy_pattern > 0.15) & ~water_mask

    urban_effect = 3.0 * np.exp(
        -0.5 * (
            ((lat_g - 9.49) / 0.03) ** 2
            + ((lon_g - 76.34) / 0.03) ** 2
        )
    )

    environmental = generate_environmental_timeseries(
        start_date,
        end_date,
        seed,
    )

    environmental = environmental.set_index("date")

    rainfall = environmental["rainfall_mm"]
    soil_moisture = environmental["soil_moisture"]
    temperature = environmental["temperature_c"]

    antecedent_rainfall = rainfall.rolling(7, min_periods=1).sum().shift(1)
    antecedent_rainfall = antecedent_rainfall.fillna(0)

    acquisition_rain = antecedent_rainfall.reindex(dates).to_numpy()
    acquisition_sm = soil_moisture.reindex(dates).to_numpy()
    acquisition_temp = temperature.reindex(dates).to_numpy()

    shape = (len(dates), len(lats), len(lons))

    vv = np.empty(shape, dtype=np.float32)
    vh = np.empty(shape, dtype=np.float32)

    rain_cube = np.empty(shape, dtype=np.float32)
    sm_cube = np.empty(shape, dtype=np.float32)
    temp_cube = np.empty(shape, dtype=np.float32)

    anomaly_mask = np.zeros(shape, dtype=np.uint8)

    anomaly_indices = [
        max(1, len(dates) // 3),
        min(int(len(dates) * 0.87), len(dates) - 1),
    ]

    anomaly_centres = [
        (9.38, 76.40),
        (9.48, 76.47),
    ]

    anomaly_strengths = [-6.0, 5.0]

    for i, date in enumerate(dates):
        doy = float(date.dayofyear)

        seasonal = _seasonal_backscatter(np.array([doy]))[0]

        seasonal_water_effect = np.where(water_mask, -6.0, 0.0)

        paddy_wetness = np.exp(
            -0.5 * ((doy - 200) / 45) ** 2
        )

        paddy_effect = np.where(
            paddy_mask,
            -3.0 * paddy_wetness,
            0.0,
        )

        rain_effect = -0.035 * acquisition_rain[i]
        moisture_effect = -2.0 * (acquisition_sm[i] - 0.25)

        spatial_variation = (
            0.8 * np.sin(12 * (lat_g - 9.25))
            + 0.5 * np.cos(15 * (lon_g - 76.3))
        )

        noise = rng.normal(0, 0.8, lat_g.shape)

        vv_frame = (
            seasonal
            + seasonal_water_effect
            + paddy_effect
            + urban_effect
            + rain_effect
            + moisture_effect
            + spatial_variation
            + noise
        )

        vh_frame = (
            vv_frame
            - 7.0
            + rng.normal(0, 0.5, lat_g.shape)
        )

        if i in anomaly_indices:
            centre_lat, centre_lon = anomaly_centres[anomaly_indices.index(i)]

            anomaly_region = (
                ((lat_g - centre_lat) / 0.025) ** 2
                + ((lon_g - centre_lon) / 0.025) ** 2
            ) <= 1.0

            vv_frame[anomaly_region] += anomaly_strengths[
                anomaly_indices.index(i)
            ]

            vh_frame[anomaly_region] += anomaly_strengths[
                anomaly_indices.index(i)
            ]

            anomaly_mask[i, anomaly_region] = 1

        vv[i] = np.clip(vv_frame, -35, 5)
        vh[i] = np.clip(vh_frame, -45, 0)

        rain_cube[i] = acquisition_rain[i]
        sm_cube[i] = acquisition_sm[i]
        temp_cube[i] = acquisition_temp[i]

    source_ds = xr.Dataset(
        {
            "vv_db": (("time", "latitude", "longitude"), vv),
            "vh_db": (("time", "latitude", "longitude"), vh),
            "rainfall_mm": (("time", "latitude", "longitude"), rain_cube),
            "soil_moisture": (("time", "latitude", "longitude"), sm_cube),
            "temperature_c": (("time", "latitude", "longitude"), temp_cube),
            "synthetic_anomaly": (
                ("time", "latitude", "longitude"),
                anomaly_mask,
            ),
            "water_mask": (
                ("latitude", "longitude"),
                water_mask.astype(np.uint8),
            ),
            "paddy_mask": (
                ("latitude", "longitude"),
                paddy_mask.astype(np.uint8),
            ),
        },
        coords={
            "time": dates,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "CounterSAR Synthetic Spatial SAR Datacube",
            "data_type": "SYNTHETIC",
            "is_synthetic": True,
            "warning": SYNTHETIC_LABEL,
            "study_area": "Kuttanad, Kerala, India",
            "source_model": "Simulated Sentinel-1 C-band IW",
            "crs": "EPSG:4326",
            "spatial_resolution_deg": SPATIAL_RES,
            "anomaly_reference": "Synthetic injected anomaly locations",
        },
    )

    result = regrid_to_100m(
        source_ds,
        grid_spec=grid_spec,
        method="linear",
        categorical_vars=[
            "synthetic_anomaly",
            "water_mask",
            "paddy_mask",
        ],
    )
    result["valid_data_mask"] = np.isfinite(
    result["vv_db"].isel(time=0)
    )
    result["synthetic_anomaly"] = (
        result["synthetic_anomaly"] > 0.5
    )

    result["water_mask"] = result["water_mask"] > 0.5
    result["paddy_mask"] = result["paddy_mask"] > 0.5

    result.attrs.update({
        "title": "CounterSAR Synthetic Spatial SAR Datacube",
        "data_type": "SYNTHETIC",
        "is_synthetic": True,
        "warning": SYNTHETIC_LABEL,
        "coverage_mask": "valid_data_mask",
        "study_area": "Kuttanad, Kerala, India",
        "source_model": "Simulated Sentinel-1 C-band IW",
        "spatial_resolution_meters": 100.0,
        "anomaly_reference": "Synthetic injected anomaly locations",
    })

    return result


def get_study_area_info():
    return {
        "name": "Kuttanad",
        "state": "Kerala",
        "country": "India",
        "description": (
            "Kuttanad is a low-lying region in the Alappuzha and Kottayam "
            "districts of Kerala, India. It is one of the few places in the "
            "world where farming is carried out below sea level. The region "
            "is characterised by extensive backwaters, paddy fields, and is "
            "highly susceptible to seasonal flooding during the monsoon."
        ),
        "bbox": KUTTANAD_BBOX,
        "centre": {
            "lat": (KUTTANAD_BBOX["lat_min"] + KUTTANAD_BBOX["lat_max"]) / 2,
            "lon": (KUTTANAD_BBOX["lon_min"] + KUTTANAD_BBOX["lon_max"]) / 2,
        },
        "elevation_range": "−2.2 m to +2.0 m MSL",
        "area_km2": "~1 100",
        "crs": "EPSG:4326",
    }