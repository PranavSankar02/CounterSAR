from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.skeleton import CRS, KUTTANAD_BOUNDS, AnomalyResult
from src.models import SeasonalClimatologyModel, WeatherAwareLGBMModel


class AnomalyVerdict(str, Enum):
    EXPECTED = "Expected"
    UNUSUAL_EXPLAINED = "Unusual but Explained"
    UNEXPLAINED_CANDIDATE = "Unexplained Candidate Event"
    INSUFFICIENT_DATA = "Insufficient Data"


VERDICT_CODE_MAP = {
    AnomalyVerdict.EXPECTED.value: 0,
    AnomalyVerdict.UNUSUAL_EXPLAINED.value: 1,
    AnomalyVerdict.UNEXPLAINED_CANDIDATE.value: 2,
    AnomalyVerdict.INSUFFICIENT_DATA.value: -1,
}

CODE_TO_VERDICT_MAP = {v: k for k, v in VERDICT_CODE_MAP.items()}


def calculate_z_scores(
    observed: np.ndarray,
    predicted: np.ndarray,
    uncertainty_std: np.ndarray,
    min_noise_floor_db: float = 0.5,
) -> np.ndarray:
    if min_noise_floor_db <= 0:
        raise ValueError("min_noise_floor_db must be positive")

    obs, pred, std = np.broadcast_arrays(
        np.asarray(observed, dtype=float),
        np.asarray(predicted, dtype=float),
        np.asarray(uncertainty_std, dtype=float),
    )

    valid = (
        np.isfinite(obs)
        & np.isfinite(pred)
        & np.isfinite(std)
        & (std >= 0)
    )

    result = np.full(obs.shape, np.nan, dtype=float)

    safe_std = np.maximum(std, min_noise_floor_db)
    result[valid] = (obs[valid] - pred[valid]) / safe_std[valid]

    return result


def calculate_adaptive_threshold(
    times: pd.DatetimeIndex,
    base_threshold: float = 2.0,
    monsoon_scale: float = 1.25,
) -> np.ndarray:
    if base_threshold <= 0:
        raise ValueError("base_threshold must be positive")

    if monsoon_scale < 1:
        raise ValueError("monsoon_scale must be at least 1")

    times = pd.DatetimeIndex(times)
    doy = times.dayofyear.values

    thresholds = np.full(len(times), base_threshold, dtype=float)

    sw_mask = (doy >= 152) & (doy <= 273)
    ne_mask = (doy >= 274) & (doy <= 334)

    thresholds[sw_mask] = base_threshold * monsoon_scale
    thresholds[ne_mask] = base_threshold * (
        1.0 + 0.5 * (monsoon_scale - 1.0)
    )

    return thresholds


def apply_persistence_filter(
    candidate_mask: np.ndarray,
    min_consecutive_passes: int = 2,
    sign_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    candidates = np.asarray(candidate_mask, dtype=bool)

    if candidates.ndim < 1:
        raise ValueError("candidate_mask must have at least one dimension")

    if min_consecutive_passes < 1:
        raise ValueError("min_consecutive_passes must be at least 1")

    if candidates.shape[0] == 0:
        return np.zeros_like(candidates, dtype=bool)

    if min_consecutive_passes == 1:
        return candidates.copy()

    if sign_array is not None:
        signs = np.broadcast_to(
            np.asarray(sign_array, dtype=float),
            candidates.shape,
        )
        signs = np.sign(signs)
    else:
        signs = None

    persistent = np.zeros_like(candidates, dtype=bool)
    n_times = candidates.shape[0]

    if min_consecutive_passes > n_times:
        return persistent

    for start in range(n_times - min_consecutive_passes + 1):
        end = start + min_consecutive_passes
        window = candidates[start:end]

        valid_run = np.all(window, axis=0)

        if signs is not None:
            sign_window = signs[start:end]
            first_sign = sign_window[0]

            same_sign = np.all(
                sign_window == first_sign[None, ...],
                axis=0,
            )

            nonzero_sign = first_sign != 0
            valid_run &= same_sign & nonzero_sign

        for index in range(start, end):
            persistent[index] |= valid_run

    return persistent


def generate_three_category_verdicts(
    z_clim: np.ndarray,
    z_weather: np.ndarray,
    thresholds: np.ndarray,
    observed: np.ndarray,
    weather_pred: np.ndarray,
    min_valid_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zc, zw, obs, weather = np.broadcast_arrays(
        np.asarray(z_clim, dtype=float),
        np.asarray(z_weather, dtype=float),
        np.asarray(observed, dtype=float),
        np.asarray(weather_pred, dtype=float),
    )

    if zc.ndim < 1:
        raise ValueError("Inputs must have a time dimension")

    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim == 1:
        if len(thresholds) != zc.shape[0]:
            raise ValueError("Threshold count must match the time dimension")

        thresholds = thresholds.reshape(
            (len(thresholds),) + (1,) * (zc.ndim - 1)
        )

    thresholds = np.broadcast_to(thresholds, zc.shape)

    verdict_codes = np.full(zc.shape, -1, dtype=np.int8)
    candidate_mask = np.zeros(zc.shape, dtype=bool)

    valid = (
        np.isfinite(zc)
        & np.isfinite(zw)
        & np.isfinite(obs)
        & np.isfinite(weather)
        & np.isfinite(thresholds)
    )

    valid_count = np.sum(np.isfinite(obs), axis=0)
    sufficient_data = valid_count >= min_valid_obs

    sufficient_mask = np.broadcast_to(
        sufficient_data,
        zc.shape,
    )

    valid &= sufficient_mask

    expected = valid & (np.abs(zc) <= thresholds) & (
        np.abs(zw) <= thresholds
    )

    explained = valid & (np.abs(zc) > thresholds) & (
        np.abs(zw) <= thresholds
    )

    unexplained = valid & (np.abs(zw) > thresholds)

    verdict_codes[expected] = VERDICT_CODE_MAP[
        AnomalyVerdict.EXPECTED.value
    ]

    verdict_codes[explained] = VERDICT_CODE_MAP[
        AnomalyVerdict.UNUSUAL_EXPLAINED.value
    ]

    verdict_codes[unexplained] = VERDICT_CODE_MAP[
        AnomalyVerdict.UNEXPLAINED_CANDIDATE.value
    ]

    candidate_mask[unexplained] = True

    verdict_strings = np.full(
        zc.shape,
        AnomalyVerdict.INSUFFICIENT_DATA.value,
        dtype="<U32",
    )

    for code, label in CODE_TO_VERDICT_MAP.items():
        verdict_strings[verdict_codes == code] = label

    return verdict_strings, verdict_codes, candidate_mask


def classify_deviation_nature(
    residual_db: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    residual, candidates = np.broadcast_arrays(
        np.asarray(residual_db, dtype=float),
        np.asarray(candidate_mask, dtype=bool),
    )

    nature = np.full(
        residual.shape,
        "expected_variation",
        dtype="<U40",
    )

    drop = candidates & np.isfinite(residual) & (residual < 0)
    jump = candidates & np.isfinite(residual) & (residual > 0)

    nature[drop] = "unexplained_backscatter_drop"
    nature[jump] = "unexplained_backscatter_jump"

    return nature


def _get_model_array(
    prediction: xr.Dataset,
    variable: str,
    reference: xr.DataArray,
) -> np.ndarray:
    if variable not in prediction:
        raise KeyError(f"Model output is missing '{variable}'")

    array = prediction[variable]

    if not isinstance(array, xr.DataArray):
        raise TypeError(f"'{variable}' must be an xarray DataArray")

    array = array.broadcast_like(reference).transpose(*reference.dims)

    return np.asarray(array.values, dtype=float)


def _get_spatial_coordinates(
    data: xr.Dataset,
) -> Dict[str, Any]:
    coordinates = {}

    for name in ("time", "y", "x"):
        if name in data.coords:
            coordinates[name] = data.coords[name]

    return coordinates


def run_anomaly_pipeline(
    data: Union[xr.Dataset, pd.DataFrame],
    target_var: str = "vv_db",
    climatology_model: Optional[SeasonalClimatologyModel] = None,
    weather_model: Optional[WeatherAwareLGBMModel] = None,
    base_threshold: float = 2.0,
    min_consecutive_passes: int = 2,
    output_netcdf_path: Optional[Union[str, Path]] = "results/pipeline_output.nc",
) -> xr.Dataset:
    if not isinstance(data, xr.Dataset):
        raise TypeError(
            "The spatial anomaly pipeline currently requires an xarray Dataset"
        )

    if target_var not in data:
        raise KeyError(f"Target variable '{target_var}' not found")

    if "time" not in data[target_var].dims:
        raise ValueError("Target variable must contain a time dimension")

    reference = data[target_var]
    times = pd.DatetimeIndex(pd.to_datetime(data.time.values))

    if climatology_model is None or not climatology_model.is_fitted:
        climatology_model = SeasonalClimatologyModel(num_harmonics=2)
        climatology_model.fit(data, target_var=target_var)

    if weather_model is None or not weather_model.is_fitted:
        weather_model = WeatherAwareLGBMModel(
            alpha_lower=0.10,
            alpha_upper=0.90,
        )
        weather_model.fit(data, target_var=target_var)

    clim_pred = climatology_model.predict(data)
    weather_pred = weather_model.predict(data)

    obs = np.asarray(reference.values, dtype=float)

    y_clim = _get_model_array(
        clim_pred,
        "predicted_backscatter_db",
        reference,
    )

    std_clim = _get_model_array(
        clim_pred,
        "prediction_std",
        reference,
    )

    y_weather = _get_model_array(
        weather_pred,
        "predicted_backscatter_db",
        reference,
    )

    std_weather = _get_model_array(
        weather_pred,
        "prediction_std",
        reference,
    )

    z_clim = calculate_z_scores(obs, y_clim, std_clim)
    z_weather = calculate_z_scores(obs, y_weather, std_weather)

    thresholds = calculate_adaptive_threshold(
        times,
        base_threshold=base_threshold,
    )

    verdict_strings, verdict_codes, candidate_mask = (
        generate_three_category_verdicts(
            z_clim=z_clim,
            z_weather=z_weather,
            thresholds=thresholds,
            observed=obs,
            weather_pred=y_weather,
        )
    )

    residual_weather = obs - y_weather

    deviation_nature = classify_deviation_nature(
        residual_weather,
        candidate_mask,
    )

    persistent_mask = apply_persistence_filter(
        candidate_mask,
        min_consecutive_passes=min_consecutive_passes,
        sign_array=residual_weather,
    )

    output_dims = reference.dims

    out_vars = {
        "observed_backscatter_db": (
            output_dims,
            obs,
            {"units": "dB"},
        ),
        "climatology_predicted_db": (
            output_dims,
            y_clim,
            {"units": "dB"},
        ),
        "weather_predicted_db": (
            output_dims,
            y_weather,
            {"units": "dB"},
        ),
        "climatology_prediction_std": (
            output_dims,
            std_clim,
            {"units": "dB"},
        ),
        "weather_prediction_std": (
            output_dims,
            std_weather,
            {"units": "dB"},
        ),
        "residual_climatology_db": (
            output_dims,
            obs - y_clim,
            {"units": "dB"},
        ),
        "residual_weather_db": (
            output_dims,
            residual_weather,
            {"units": "dB"},
        ),
        "z_score_climatology": (
            output_dims,
            z_clim,
            {"units": "standard_deviations"},
        ),
        "z_score_weather": (
            output_dims,
            z_weather,
            {"units": "standard_deviations"},
        ),
        "adaptive_threshold": (
            ("time",),
            thresholds,
            {"units": "standard_deviations"},
        ),
        "verdict_code": (
            output_dims,
            verdict_codes,
        ),
        "verdict_category": (
            output_dims,
            verdict_strings,
        ),
        "is_anomaly_candidate": (
            output_dims,
            candidate_mask,
        ),
        "is_persistent_anomaly": (
            output_dims,
            persistent_mask,
        ),
        "deviation_nature": (
            output_dims,
            deviation_nature,
        ),
    }

    for variable in (
        "weather_lower_bound_db",
        "weather_upper_bound_db",
    ):
        if variable in weather_pred:
            values = _get_model_array(
                weather_pred,
                variable,
                reference,
            )

            out_vars[variable] = (
                output_dims,
                values,
                {"units": "dB"},
            )

    result_ds = xr.Dataset(
        data_vars=out_vars,
        coords=_get_spatial_coordinates(data),
    )

    for variable in (
        "rainfall_mm",
        "soil_moisture",
        "temperature_c",
        "antecedent_rain_3d_mm",
        "antecedent_rain_7d_mm",
        "vh_db",
        "valid_data_mask",
        "water_mask",
        "paddy_mask",
    ):
        if variable in data:
            result_ds[variable] = data[variable]

    is_synthetic = bool(data.attrs.get("is_synthetic", False))

    result_ds.attrs.update({
        "title": "CounterSAR Spatial Anomaly Detection Output",
        "study_area": "Kuttanad, Kerala, India",
        "target_variable": target_var,
        "crs": data.attrs.get("crs", CRS),
        "spatial_resolution_meters": data.attrs.get(
            "spatial_resolution_meters",
            100.0,
        ),
        "base_threshold_z": float(base_threshold),
        "min_consecutive_passes": int(min_consecutive_passes),
        "total_acquisitions": int(len(times)),
        "candidate_pixel_observations": int(candidate_mask.sum()),
        "persistent_pixel_observations": int(persistent_mask.sum()),
        "is_synthetic": int(is_synthetic),
        "notice": (
            "SYNTHETIC EVALUATION. Results are not evidence of real-world "
            "flood detection performance."
            if is_synthetic
            else "Anomaly candidates require independent validation."
        ),
    })

    if output_netcdf_path is not None:
        export_pipeline_results(result_ds, output_netcdf_path)

    return result_ds


def export_pipeline_results(
    result_ds: xr.Dataset,
    output_path: Union[str, Path] = "results/pipeline_output.nc",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. Choose another path or "
            "move the existing file before exporting."
        )

    clean_ds = result_ds.copy(deep=False)

    for key, value in list(clean_ds.attrs.items()):
        if isinstance(value, bool):
            clean_ds.attrs[key] = int(value)

    clean_ds.to_netcdf(str(path), engine="netcdf4")

    return path


def to_anomaly_results(
    result_ds: xr.Dataset,
    max_results: Optional[int] = None,
) -> List[AnomalyResult]:
    results = []

    candidates = result_ds["is_anomaly_candidate"].values.astype(bool)
    persistent = result_ds["is_persistent_anomaly"].values.astype(bool)

    observed = result_ds["observed_backscatter_db"].values
    predicted = result_ds["weather_predicted_db"].values
    z_scores = result_ds["z_score_weather"].values
    nature = result_ds["deviation_nature"].values

    times = pd.to_datetime(result_ds.time.values)
    is_synthetic = bool(result_ds.attrs.get("is_synthetic", False))

    if candidates.ndim == 1:
        candidate_indices = np.flatnonzero(candidates)
    else:
        candidate_indices = np.argwhere(candidates)

    if max_results is not None:
        candidate_indices = candidate_indices[:max_results]

    y_coords = result_ds.coords.get("y")
    x_coords = result_ds.coords.get("x")

    if y_coords is not None and x_coords is not None:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            result_ds.attrs.get("crs", CRS),
            "EPSG:4326",
            always_xy=True,
        )

    for index in candidate_indices:
        if candidates.ndim == 1:
            t = int(index)
            spatial_index = ()
        else:
            t = int(index[0])
            spatial_index = tuple(index[1:])

        if len(spatial_index) == 2 and x_coords is not None and y_coords is not None:
            row, col = spatial_index

            x = float(x_coords.values[col])
            y = float(y_coords.values[row])

            lon, lat = transformer.transform(x, y)

        else:
            lat = KUTTANAD_BOUNDS["lat_min"]
            lon = KUTTANAD_BOUNDS["lon_min"]

        obs_value = float(observed[(t,) + spatial_index])
        pred_value = float(predicted[(t,) + spatial_index])
        z_value = float(z_scores[(t,) + spatial_index])

        results.append(
            AnomalyResult(
                date=times[t].strftime("%Y-%m-%d"),
                latitude=float(lat),
                longitude=float(lon),
                observed_db=obs_value,
                predicted_db=pred_value,
                z_score=z_value,
                is_persistent=bool(persistent[(t,) + spatial_index]),
                anomaly_type=str(nature[(t,) + spatial_index]),
                is_synthetic=is_synthetic,
            )
        )

    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_t, y_p = np.broadcast_arrays(
        np.asarray(y_true, dtype=bool),
        np.asarray(y_pred, dtype=bool),
    )

    tp = int(np.sum(y_t & y_p))
    fp = int(np.sum(~y_t & y_p))
    fn = int(np.sum(y_t & ~y_p))
    tn = int(np.sum(~y_t & ~y_p))

    total = len(y_t.ravel())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    far = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    pe = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn))
        / (total ** 2)
        if total
        else 0.0
    )

    kappa = (
        (accuracy - pe) / (1 - pe)
        if total and (1 - pe) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "false_alarm_rate": round(far, 3),
        "overall_accuracy": round(accuracy, 3),
        "cohens_kappa": round(kappa, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_stage6_benchmarks(
    target_var: str = "vv_db",
    random_seed: int = 42,
) -> Dict[str, Any]:
    from src.data_source import create_synthetic_datacube

    cube = create_synthetic_datacube(
        "2023-01-01",
        "2024-06-30",
    )

    times = pd.to_datetime(cube.time.values)
    n = len(times)

    ground_truth_events = np.zeros(n, dtype=bool)
    ground_truth_events[18:20] = True
    ground_truth_events[35:37] = True

    obs = np.copy(cube[target_var].values)

    obs[18:20] -= 5.5
    obs[35:37] -= 5.0

    cube[target_var] = ("time", obs)

    clim_model = SeasonalClimatologyModel(num_harmonics=2)
    clim_model.fit(cube, target_var=target_var)

    clim_result = clim_model.predict(cube)

    z_clim = calculate_z_scores(
        obs,
        clim_result["predicted_backscatter_db"].values,
        clim_result["prediction_std"].values,
    )

    pred_m1 = z_clim < -2.0

    mae_m1 = float(
        np.nanmean(
            np.abs(
                obs - clim_result["predicted_backscatter_db"].values
            )
        )
    )

    rmse_m1 = float(
        np.sqrt(
            np.nanmean(
                (
                    obs
                    - clim_result["predicted_backscatter_db"].values
                ) ** 2
            )
        )
    )

    weather_model = WeatherAwareLGBMModel(
        alpha_lower=0.10,
        alpha_upper=0.90,
    )

    weather_model.fit(cube, target_var=target_var)
    weather_result = weather_model.predict(cube)

    z_weather = calculate_z_scores(
        obs,
        weather_result["predicted_backscatter_db"].values,
        weather_result["prediction_std"].values,
    )

    pred_m2 = z_weather < -2.0

    mae_m2 = float(
        np.nanmean(
            np.abs(
                obs
                - weather_result["predicted_backscatter_db"].values
            )
        )
    )

    rmse_m2 = float(
        np.sqrt(
            np.nanmean(
                (
                    obs
                    - weather_result["predicted_backscatter_db"].values
                ) ** 2
            )
        )
    )

    delta = np.zeros(n)
    delta[1:] = obs[1:] - obs[:-1]
    pred_m3 = delta < -3.0

    dry_mask = times.month.isin([1, 2, 3])
    dry_reference = float(np.nanmean(obs[dry_mask]))

    pred_m4 = (obs - dry_reference) < -3.0

    thresholds = calculate_adaptive_threshold(times)

    _, _, candidates = generate_three_category_verdicts(
        z_clim=z_clim,
        z_weather=z_weather,
        thresholds=thresholds,
        observed=obs,
        weather_pred=weather_result["predicted_backscatter_db"].values,
    )

    pred_m5 = apply_persistence_filter(
        candidates,
        min_consecutive_passes=2,
        sign_array=obs - weather_result["predicted_backscatter_db"].values,
    )

    results = {
        "Image Differencing Baseline": {
            **compute_classification_metrics(
                ground_truth_events,
                pred_m3,
            ),
            "mae_db": np.nan,
            "rmse_db": np.nan,
        },
        "Log-Ratio Threshold Baseline": {
            **compute_classification_metrics(
                ground_truth_events,
                pred_m4,
            ),
            "mae_db": np.nan,
            "rmse_db": np.nan,
        },
        "Seasonal Climatology Baseline": {
            **compute_classification_metrics(
                ground_truth_events,
                pred_m1,
            ),
            "mae_db": round(mae_m1, 3),
            "rmse_db": round(rmse_m1, 3),
        },
        "Weather-Aware LightGBM": {
            **compute_classification_metrics(
                ground_truth_events,
                pred_m2,
            ),
            "mae_db": round(mae_m2, 3),
            "rmse_db": round(rmse_m2, 3),
        },
        "Complete CounterSAR Pipeline": {
            **compute_classification_metrics(
                ground_truth_events,
                pred_m5,
            ),
            "mae_db": round(mae_m2, 3),
            "rmse_db": round(rmse_m2, 3),
        },
    }

    return {
        "benchmark_table": results,
        "n_observations": n,
        "ground_truth_flood_count": int(np.sum(ground_truth_events)),
        "is_synthetic": True,
        "validation_note": (
            "Synthetic benchmark evaluation. Results are not evidence "
            "of operational real-world flood detection performance."
        ),
    }