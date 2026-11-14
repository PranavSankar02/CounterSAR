"""
CounterSAR — Anomaly Scoring and Persistence Filtering (Stage 4)
=================================================================
Distinguishes expected seasonal patterns and weather-explained events from
genuinely unexplained satellite radar anomalies.

Pipeline Logic:
  1. Z-Score Computation with Uncertainty:
     - Climatological Z-Score: (Observed − Seasonal Expected) / σ_clim
     - Weather-Aware Z-Score:  (Observed − Weather Expected) / σ_weather
     - Uses local/temporal model uncertainty rather than a static global scalar.

  2. Adaptive Anomaly Thresholds:
     - Automatically expands during the South-West and North-East monsoon
       windows to account for natural precipitation variance and elevated speckle.

  3. Three-Category Verdict Generation:
     - 'Expected':
       Observed backscatter matches both seasonal climatology and weather expectations.
     - 'Unusual but Explained':
       Observed backscatter deviates from historical seasonal norms, but is fully
       consistent with known weather drivers (e.g. heavy rainfall or soil saturation).
     - 'Unexplained Candidate Event':
       Observed backscatter diverges significantly from BOTH seasonal expectation
       and weather-aware predictions, indicating an unexplained physical process.

  4. Physical Deviation Classification:
     - Does NOT classify every negative drop as a "flood" (can be planned irrigation,
       soil wetting, or specular water pooling).
     - Does NOT ignore positive jumps (can be emergent vegetation double-bounce
       or surface structural changes).

  5. Temporal Persistence Filtering:
     - Confirms whether an anomaly persists across consecutive satellite acquisitions
       (≥ 2 passes, e.g. ≥ 24 days in Sentinel-1 12-day revisit cycles).
     - Rejects single-pass transient speckle, antenna artifacts, or temporary puddles.

  6. NetCDF Pipeline Output:
     - Exports a CF-compliant NetCDF cube to results/pipeline_output.nc.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.skeleton import (
    CRS,
    KUTTANAD_BOUNDS,
    AnomalyResult,
)
from src.models import (
    SeasonalClimatologyModel,
    WeatherAwareLGBMModel,
)


# ════════════════════════════════════════════════════════════════════════
#  VERDICT ENUMS & CONSTANTS
# ════════════════════════════════════════════════════════════════════════

class AnomalyVerdict(str, Enum):
    """Three-category pipeline verdict."""
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


# ════════════════════════════════════════════════════════════════════════
#  1. Z-SCORE CALCULATION WITH PREDICTION UNCERTAINTY
# ════════════════════════════════════════════════════════════════════════

def calculate_z_scores(
    observed: np.ndarray,
    predicted: np.ndarray,
    uncertainty_std: np.ndarray,
    min_noise_floor_db: float = 0.5,
) -> np.ndarray:
    """
    Calculate standardized Z-scores normalized by prediction uncertainty:
        Z = (Observed − Predicted) / max(σ_pred, σ_min)

    Parameters
    ----------
    observed : np.ndarray
        Observed radar backscatter values in dB.
    predicted : np.ndarray
        Model-predicted expected backscatter in dB.
    uncertainty_std : np.ndarray
        Model-estimated prediction standard deviation in dB.
    min_noise_floor_db : float
        Noise floor to prevent division by near-zero variance.

    Returns
    -------
    np.ndarray
        Z-scores with NaNs preserved where inputs are invalid.
    """
    obs_arr = np.asarray(observed, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    std_arr = np.asarray(uncertainty_std, dtype=float)

    safe_std = np.maximum(std_arr, min_noise_floor_db)
    valid_mask = np.isfinite(obs_arr) & np.isfinite(pred_arr) & np.isfinite(safe_std)

    z_scores = np.full_like(obs_arr, np.nan, dtype=float)
    z_scores[valid_mask] = (obs_arr[valid_mask] - pred_arr[valid_mask]) / safe_std[valid_mask]
    return z_scores


# ════════════════════════════════════════════════════════════════════════
#  2. ADAPTIVE ANOMALY THRESHOLDS
# ════════════════════════════════════════════════════════════════════════

def calculate_adaptive_threshold(
    times: pd.DatetimeIndex,
    base_threshold: float = 2.0,
    monsoon_scale: float = 1.25,
) -> np.ndarray:
    """
    Compute time-varying adaptive anomaly thresholds.

    In Kuttanad, the South-West Monsoon (June to September, DOY ~152..273)
    and North-East Monsoon (October to November, DOY ~274..334) produce
    dramatically higher natural environmental and speckle variance.
    The threshold adaptively expands during monsoon periods to prevent
    spurious false alarms, while tightening during stable dry months.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps of satellite acquisitions.
    base_threshold : float
        Baseline standard deviation threshold (default: 2.0).
    monsoon_scale : float
        Multiplier during high-variance monsoon months (default: 1.25 -> Z=2.50).

    Returns
    -------
    np.ndarray
        Threshold vector matching `times`.
    """
    doy = times.dayofyear.values.astype(float)
    thresholds = np.full(len(times), base_threshold, dtype=float)

    # South-West Monsoon (Jun–Sep: DOY 152 to 273)
    sw_mask = (doy >= 152) & (doy <= 273)
    thresholds[sw_mask] = base_threshold * monsoon_scale

    # North-East Monsoon (Oct–Nov: DOY 274 to 334)
    ne_mask = (doy >= 274) & (doy <= 334)
    thresholds[ne_mask] = base_threshold * (1.0 + 0.5 * (monsoon_scale - 1.0))

    return thresholds


# ════════════════════════════════════════════════════════════════════════
#  3. PERSISTENCE FILTERING
# ════════════════════════════════════════════════════════════════════════

def apply_persistence_filter(
    candidate_mask: np.ndarray,
    min_consecutive_passes: int = 2,
    sign_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Multi-pass persistence filtering across consecutive acquisitions.

    An anomaly candidate at timestep t is confirmed as persistent if:
      - It persists across ≥ `min_consecutive_passes` consecutive acquisitions.
      - If `sign_array` is provided, consecutive anomalies must share the same
        physical sign (e.g. continuous negative drop or continuous positive jump).

    Filters out transient radar speckle, radio frequency interference (RFI),
    and ephemeral surface puddles that vanish within hours.

    Parameters
    ----------
    candidate_mask : np.ndarray of bool
        Boolean array indicating candidate anomaly detections.
    min_consecutive_passes : int
        Minimum consecutive passes required (default: 2).
    sign_array : np.ndarray, optional
        Sign indicator (+1 for positive jump, -1 for negative drop).

    Returns
    -------
    np.ndarray of bool
        Boolean mask of confirmed persistent anomalies.
    """
    n = len(candidate_mask)
    if n == 0:
        return np.array([], dtype=bool)

    persistent_mask = np.zeros(n, dtype=bool)

    if min_consecutive_passes <= 1:
        return np.copy(candidate_mask)

    for i in range(n):
        if not candidate_mask[i]:
            continue

        # Look backwards and forwards to count consecutive active runs
        run_length = 1
        current_sign = sign_array[i] if sign_array is not None else 0

        # Check backwards
        j = i - 1
        while j >= 0 and candidate_mask[j]:
            if sign_array is not None and np.sign(sign_array[j]) != np.sign(current_sign):
                break
            run_length += 1
            j -= 1

        # Check forwards
        j = i + 1
        while j < n and candidate_mask[j]:
            if sign_array is not None and np.sign(sign_array[j]) != np.sign(current_sign):
                break
            run_length += 1
            j += 1

        if run_length >= min_consecutive_passes:
            persistent_mask[i] = True

    return persistent_mask


# ════════════════════════════════════════════════════════════════════════
#  4. THREE-CATEGORY VERDICT GENERATION
# ════════════════════════════════════════════════════════════════════════

def generate_three_category_verdicts(
    z_clim: np.ndarray,
    z_weather: np.ndarray,
    thresholds: np.ndarray,
    observed: np.ndarray,
    weather_pred: np.ndarray,
    min_valid_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate three-category verdicts by contrasting seasonal climatology
    with weather-aware predictions:

      1. 'Expected' (Code 0):
         |Z_clim| ≤ T and |Z_weather| ≤ T
         Observed radar values align with normal seasonal expectation and weather.

      2. 'Unusual but Explained' (Code 1):
         |Z_clim| > T, but |Z_weather| ≤ T
         Observed values deviate from normal seasonal climatology, but are fully
         explained by known meteorological drivers (rainfall, soil moisture, etc.).

      3. 'Unexplained Candidate Event' (Code 2):
         |Z_weather| > T (and typically |Z_clim| > T)
         Observed values diverge from BOTH seasonal climatology and the weather-aware
         prediction. Weather cannot account for the deviation.

    Returns
    -------
    verdict_strings : np.ndarray of str
    verdict_codes : np.ndarray of int (0, 1, 2, or -1 for insufficient data)
    candidate_mask : np.ndarray of bool (True where verdict is Unexplained Candidate)
    """
    n = len(z_clim)
    verdict_strings = np.full(n, AnomalyVerdict.EXPECTED.value, dtype=object)
    verdict_codes = np.zeros(n, dtype=int)
    candidate_mask = np.zeros(n, dtype=bool)

    # Check for insufficient observations
    valid_count = np.sum(np.isfinite(observed))
    if valid_count < min_valid_obs:
        verdict_strings[:] = AnomalyVerdict.INSUFFICIENT_DATA.value
        verdict_codes[:] = VERDICT_CODE_MAP[AnomalyVerdict.INSUFFICIENT_DATA.value]
        return verdict_strings, verdict_codes, candidate_mask

    for i in range(n):
        zc = z_clim[i]
        zw = z_weather[i]
        t = thresholds[i]

        if not (np.isfinite(zc) and np.isfinite(zw)):
            verdict_strings[i] = AnomalyVerdict.INSUFFICIENT_DATA.value
            verdict_codes[i] = VERDICT_CODE_MAP[AnomalyVerdict.INSUFFICIENT_DATA.value]
            continue

        abs_zc = abs(zc)
        abs_zw = abs(zw)

        if abs_zw > t:
            # Weather model cannot explain the departure
            verdict_strings[i] = AnomalyVerdict.UNEXPLAINED_CANDIDATE.value
            verdict_codes[i] = VERDICT_CODE_MAP[AnomalyVerdict.UNEXPLAINED_CANDIDATE.value]
            candidate_mask[i] = True
        elif abs_zc > t and abs_zw <= t:
            # Climatology was surprised, but weather model explained it!
            verdict_strings[i] = AnomalyVerdict.UNUSUAL_EXPLAINED.value
            verdict_codes[i] = VERDICT_CODE_MAP[AnomalyVerdict.UNUSUAL_EXPLAINED.value]
        else:
            # Within normal bounds for both
            verdict_strings[i] = AnomalyVerdict.EXPECTED.value
            verdict_codes[i] = VERDICT_CODE_MAP[AnomalyVerdict.EXPECTED.value]

    return verdict_strings, verdict_codes, candidate_mask


# ════════════════════════════════════════════════════════════════════════
#  5. PHYSICAL DEVIATION CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════

def classify_deviation_nature(
    residual_db: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """
    Classify the physical direction/character of the radar departure.

    Scientific Principle:
      - Do NOT classify every negative drop as a "flood" (can be planned irrigation
        or soil wetting).
      - Do NOT ignore positive jumps (can be emergent vegetation double-bounce
        or structural changes).

    Categories:
      - 'unexplained_backscatter_drop' (Negative departure: specular water, inundation, saturation)
      - 'unexplained_backscatter_jump' (Positive departure: double-bounce, surface roughness, emergent canopy)
      - 'expected_variation'          (Normal behavior within uncertainty bounds)
    """
    n = len(residual_db)
    nature = np.full(n, "expected_variation", dtype=object)

    for i in range(n):
        if not candidate_mask[i] or np.isnan(residual_db[i]):
            continue
        if residual_db[i] < 0:
            nature[i] = "unexplained_backscatter_drop"
        else:
            nature[i] = "unexplained_backscatter_jump"

    return nature


# ════════════════════════════════════════════════════════════════════════
#  6. END-TO-END ANOMALY DETECTION PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_anomaly_pipeline(
    data: Union[xr.Dataset, pd.DataFrame],
    target_var: str = "vv_db",
    climatology_model: Optional[SeasonalClimatologyModel] = None,
    weather_model: Optional[WeatherAwareLGBMModel] = None,
    base_threshold: float = 2.0,
    min_consecutive_passes: int = 2,
    output_netcdf_path: Optional[Union[str, Path]] = "results/pipeline_output.nc",
) -> xr.Dataset:
    """
    Execute the full CounterSAR anomaly detection and persistence filtering pipeline.

    Workflow:
      1. Prepares seasonal climatology and weather-aware predictions.
      2. Computes uncertainty-normalized Z-scores for both models.
      3. Calculates adaptive seasonal thresholds.
      4. Assigns the 3-category verdict ('Expected', 'Unusual but Explained', 'Unexplained Candidate').
      5. Runs temporal persistence filtering across consecutive acquisitions.
      6. Formats output as a rich, CF-compliant NetCDF Dataset and exports to disk.

    Parameters
    ----------
    data : xr.Dataset or pd.DataFrame
        Aligned satellite & environmental data (from src/data_source.py).
    target_var : str
        Target backscatter variable (default: 'vv_db').
    climatology_model : SeasonalClimatologyModel, optional
        Pre-fitted or new climatology model.
    weather_model : WeatherAwareLGBMModel, optional
        Pre-fitted or new weather-aware model.
    base_threshold : float
        Adaptive baseline threshold in Z-score units (default: 2.0).
    min_consecutive_passes : int
        Minimum consecutive passes for persistence confirmation (default: 2).
    output_netcdf_path : str or Path, optional
        Destination path for pipeline output (default: 'results/pipeline_output.nc').

    Returns
    -------
    xr.Dataset
        Complete pipeline output dataset containing verdicts, scores, and bounds.
    """
    # 1. Fit models if not provided
    if climatology_model is None or not climatology_model.is_fitted:
        climatology_model = SeasonalClimatologyModel(num_harmonics=2)
        climatology_model.fit(data, target_var=target_var)

    if weather_model is None or not weather_model.is_fitted:
        weather_model = WeatherAwareLGBMModel(alpha_lower=0.10, alpha_upper=0.90)
        weather_model.fit(data, target_var=target_var)

    # 2. Generate predictions from both models
    clim_pred = climatology_model.predict(data)
    weather_pred = weather_model.predict(data)

    # Extract arrays
    if isinstance(data, xr.Dataset):
        times = pd.to_datetime(data["time"].values)
        obs_db = data[target_var].values
        is_synth = int(data.attrs.get("is_synthetic", 0))
    else:
        time_col = "date" if "date" in data.columns else "time"
        times = pd.to_datetime(data[time_col].values)
        obs_db = data[target_var].values
        is_synth = int(data.get("is_synthetic", False).any() if "is_synthetic" in data else 0)

    y_clim = clim_pred["predicted_backscatter_db"].values
    std_clim = clim_pred["prediction_std"].values

    y_weather = weather_pred["predicted_backscatter_db"].values
    std_weather = weather_pred["prediction_std"].values

    # 3. Compute uncertainty-aware Z-scores
    z_clim = calculate_z_scores(obs_db, y_clim, std_clim)
    z_weather = calculate_z_scores(obs_db, y_weather, std_weather)

    # 4. Adaptive thresholds
    thresholds = calculate_adaptive_threshold(times, base_threshold=base_threshold)

    # 5. Generate 3-category verdicts
    verdict_strings, verdict_codes, candidate_mask = generate_three_category_verdicts(
        z_clim=z_clim,
        z_weather=z_weather,
        thresholds=thresholds,
        observed=obs_db,
        weather_pred=y_weather,
    )

    # 6. Physical deviation nature & Persistence filtering
    residual_weather = obs_db - y_weather
    deviation_nature = classify_deviation_nature(residual_weather, candidate_mask)

    sign_array = np.sign(residual_weather)
    persistent_mask = apply_persistence_filter(
        candidate_mask=candidate_mask,
        min_consecutive_passes=min_consecutive_passes,
        sign_array=sign_array,
    )

    # 7. Build rich NetCDF output Dataset
    out_vars = {
        "observed_backscatter_db": (
            "time", obs_db,
            {"units": "dB", "long_name": f"Observed SAR {target_var.upper()} Backscatter"}
        ),
        "climatology_predicted_db": (
            "time", y_clim,
            {"units": "dB", "long_name": "Climatology Expected Backscatter"}
        ),
        "weather_predicted_db": (
            "time", y_weather,
            {"units": "dB", "long_name": "Weather-Aware Expected Backscatter"}
        ),
        "weather_lower_bound_db": (
            "time", weather_pred["lower_bound_db"].values,
            {"units": "dB", "long_name": "Weather-Aware 10th Percentile Bound"}
        ),
        "weather_upper_bound_db": (
            "time", weather_pred["upper_bound_db"].values,
            {"units": "dB", "long_name": "Weather-Aware 90th Percentile Bound"}
        ),
        "z_score_climatology": (
            "time", z_clim,
            {"units": "std_dev", "long_name": "Z-Score vs Seasonal Climatology"}
        ),
        "z_score_weather": (
            "time", z_weather,
            {"units": "std_dev", "long_name": "Z-Score vs Weather-Aware Model"}
        ),
        "adaptive_threshold": (
            "time", thresholds,
            {"units": "std_dev", "long_name": "Adaptive Anomaly Decision Threshold"}
        ),
        "verdict_code": (
            "time", verdict_codes,
            {"long_name": "Verdict Code (0=Expected, 1=Unusual Explained, 2=Unexplained Candidate, -1=Insufficient)"}
        ),
        "verdict_category": (
            "time", np.asarray(verdict_strings, dtype=str),
            {"long_name": "Three-Category Anomaly Verdict"}
        ),
        "is_anomaly_candidate": (
            "time", candidate_mask.astype(int),
            {"long_name": "Flag indicating unexplained candidate event before persistence filtering"}
        ),
        "is_persistent_anomaly": (
            "time", persistent_mask.astype(int),
            {"long_name": f"Confirmed anomaly persisting over ≥ {min_consecutive_passes} consecutive passes"}
        ),
        "deviation_nature": (
            "time", np.asarray(deviation_nature, dtype=str),
            {"long_name": "Physical nature of deviation (drop, jump, or expected)"}
        ),
    }

    # Pass through relevant environmental covariates if available
    covariates = [
        ("rainfall_mm", "mm/day", "Daily Rainfall Depth"),
        ("soil_moisture", "m³/m³", "Volumetric Soil Moisture"),
        ("temperature_c", "°C", "Surface Temperature"),
        ("antecedent_rain_3d_mm", "mm", "3-Day Antecedent Rainfall"),
        ("antecedent_rain_7d_mm", "mm", "7-Day Antecedent Rainfall"),
        ("vh_db", "dB", "Observed SAR VH Cross-Polarisation"),
    ]
    for cov_name, unit, long_name in covariates:
        if isinstance(data, xr.Dataset) and cov_name in data:
            out_vars[cov_name] = ("time", np.asarray(data[cov_name].values, dtype=float), {"units": unit, "long_name": long_name})
        elif isinstance(data, pd.DataFrame) and cov_name in data.columns:
            out_vars[cov_name] = ("time", np.asarray(data[cov_name].values, dtype=float), {"units": unit, "long_name": long_name})

    result_ds = xr.Dataset(data_vars=out_vars, coords={"time": times})

    # Add spatial coordinates if available in input
    if isinstance(data, xr.Dataset):
        if "latitude" in data.coords:
            result_ds.coords["latitude"] = data["latitude"].values
        if "longitude" in data.coords:
            result_ds.coords["longitude"] = data["longitude"].values

    # Provenance and pipeline metadata
    result_ds.attrs["title"] = "CounterSAR Anomaly Detection Pipeline Output"
    result_ds.attrs["study_area"] = "Kuttanad, Kerala, India"
    result_ds.attrs["target_variable"] = target_var
    result_ds.attrs["base_threshold_z"] = base_threshold
    result_ds.attrs["min_consecutive_passes"] = min_consecutive_passes
    result_ds.attrs["total_observations"] = len(times)
    result_ds.attrs["unexplained_candidates_count"] = int(np.sum(candidate_mask))
    result_ds.attrs["persistent_anomalies_count"] = int(np.sum(persistent_mask))
    result_ds.attrs["is_synthetic"] = is_synth
    result_ds.attrs["crs"] = CRS
    result_ds.attrs["notice"] = (
        "⚠️ SYNTHETIC EVALUATION — Synthetic observations must not be used as evidence of real-world performance"
        if is_synth else "CounterSAR Operational Output"
    )

    # 8. Export to NetCDF if path requested
    if output_netcdf_path:
        export_pipeline_results(result_ds, output_netcdf_path)

    return result_ds


def export_pipeline_results(
    result_ds: xr.Dataset,
    output_path: Union[str, Path] = "results/pipeline_output.nc",
) -> Path:
    """
    Save pipeline output dataset to NetCDF (results/pipeline_output.nc)
    with clean, sanitized attributes.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize boolean attributes for NetCDF-4 compatibility
    clean_ds = result_ds.copy(deep=False)
    for k, v in list(clean_ds.attrs.items()):
        if isinstance(v, bool):
            clean_ds.attrs[k] = int(v)

    for var in clean_ds.data_vars:
        for k, v in list(clean_ds[var].attrs.items()):
            if isinstance(v, bool):
                clean_ds[var].attrs[k] = int(v)

    clean_ds.to_netcdf(str(p), engine="netcdf4")
    return p


def to_anomaly_results(result_ds: xr.Dataset) -> List[AnomalyResult]:
    """
    Convert persistent and candidate anomalies into a list of shared AnomalyResult dataclasses.
    """
    results = []
    times = pd.to_datetime(result_ds["time"].values)
    is_cand = result_ds["is_anomaly_candidate"].values
    is_pers = result_ds["is_persistent_anomaly"].values
    obs = result_ds["observed_backscatter_db"].values
    pred = result_ds["weather_predicted_db"].values
    z = result_ds["z_score_weather"].values
    nature = result_ds["deviation_nature"].values
    is_synth = bool(result_ds.attrs.get("is_synthetic", False))

    lat = KUTTANAD_BOUNDS["lat_min"] + 0.15
    lon = KUTTANAD_BOUNDS["lon_min"] + 0.12

    for i in range(len(times)):
        if is_cand[i] or is_pers[i]:
            results.append(
                AnomalyResult(
                    date=times[i].strftime("%Y-%m-%d"),
                    latitude=lat,
                    longitude=lon,
                    observed_db=float(obs[i]),
                    predicted_db=float(pred[i]),
                    z_score=float(z[i]) if np.isfinite(z[i]) else 0.0,
                    is_persistent=bool(is_pers[i]),
                    anomaly_type=str(nature[i]),
                    is_synthetic=is_synth,
                )
            )
    return results


# ════════════════════════════════════════════════════════════════════════
#  7. STAGE 6: BENCHMARK EVALUATION & BASELINE COMPARISONS
# ════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate Precision, Recall, F1-Score, False Alarm Rate (FAR),
    Overall Accuracy, and Cohen's Kappa agreement.
    """
    y_t = np.asarray(y_true, dtype=bool)
    y_p = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_t & y_p))
    fp = int(np.sum(~y_t & y_p))
    fn = int(np.sum(y_t & ~y_p))
    tn = int(np.sum(~y_t & ~y_p))
    total = len(y_t)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Cohen's Kappa
    po = accuracy
    pe = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (total ** 2) if total > 0 else 0.0
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

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
    """
    Rigorous benchmark comparison evaluating 5 distinct detection baselines
    against an independent synthetic flood reference dataset over Kuttanad.

    Evaluated Methods:
      1. Seasonal Climatology Baseline (Dual-harmonic DOY regression)
      2. Weather-Aware Prediction Model (LightGBM without persistence)
      3. Image Differencing Baseline (Consecutive pass Δσ⁰ < −3 dB)
      4. Log-Ratio Threshold Baseline (Ratio vs dry-season reference < −3 dB)
      5. Complete CounterSAR Pipeline (Weather-Aware + Adaptive Threshold + Persistence Filter)

    Returns
    -------
    dict with metrics, predictions, and comparative summary.
    """
    from src.data_source import create_synthetic_datacube

    # 1. Generate aligned synthetic data cube
    cube = create_synthetic_datacube("2023-01-01", "2024-06-30")
    times = pd.to_datetime(cube.time.values)
    n = len(times)

    # 2. Construct independent synthetic flood reference dataset
    # In Kuttanad, true flood / dike breach events are persistent multi-pass events:
    # Event 1: Indices 18, 19 (July 2023 SW monsoon true breach, persistent across 2 passes)
    # Event 2: Indices 35, 36 (November 2023 NE monsoon flood, persistent across 2 passes)
    ground_truth_events = np.zeros(n, dtype=bool)
    ground_truth_events[18:20] = True
    ground_truth_events[35:37] = True

    # Inject the true flood signature into the observation series at event timesteps
    # (sharp -5 to -6 dB drop simulating standing open floodwaters)
    obs_vv = np.copy(cube[target_var].values)
    obs_vv[18:20] -= 5.5
    obs_vv[35:37] -= 5.0
    cube[target_var] = ("time", obs_vv)

    # ── Method 1: Seasonal Climatology Baseline ──
    clim_model = SeasonalClimatologyModel(num_harmonics=2)
    clim_model.fit(cube, target_var=target_var)
    clim_res = clim_model.predict(cube)
    z_clim = calculate_z_scores(
        obs_vv,
        clim_res["predicted_backscatter_db"].values,
        clim_res["prediction_std"].values,
    )
    pred_m1 = z_clim < -2.0  # Significant drop vs seasonal norm
    mae_m1 = float(np.nanmean(np.abs(obs_vv - clim_res["predicted_backscatter_db"].values)))
    rmse_m1 = float(np.sqrt(np.nanmean((obs_vv - clim_res["predicted_backscatter_db"].values) ** 2)))

    # ── Method 2: Weather-Aware Prediction Model (without persistence) ──
    weather_model = WeatherAwareLGBMModel(alpha_lower=0.10, alpha_upper=0.90)
    weather_model.fit(cube, target_var=target_var)
    weather_res = weather_model.predict(cube)
    z_weather = calculate_z_scores(
        obs_vv,
        weather_res["predicted_backscatter_db"].values,
        weather_res["prediction_std"].values,
    )
    pred_m2 = z_weather < -2.0  # Significant drop vs weather prediction
    mae_m2 = float(np.nanmean(np.abs(obs_vv - weather_res["predicted_backscatter_db"].values)))
    rmse_m2 = float(np.sqrt(np.nanmean((obs_vv - weather_res["predicted_backscatter_db"].values) ** 2)))

    # ── Method 3: Image Differencing Baseline (Δσ⁰ < −3 dB) ──
    delta_consecutive = np.zeros(n)
    delta_consecutive[1:] = obs_vv[1:] - obs_vv[:-1]
    pred_m3 = delta_consecutive < -3.0

    # ── Method 4: Log-Ratio Threshold Baseline (σ⁰ − σ⁰_ref < −3 dB) ──
    # Reference is median dry-season backscatter (typically Jan–March)
    dry_mask = times.month.isin([1, 2, 3])
    dry_ref = float(np.nanmean(obs_vv[dry_mask]))
    log_ratio_db = obs_vv - dry_ref
    pred_m4 = log_ratio_db < -3.0

    # ── Method 5: Complete CounterSAR Pipeline ──
    # (Weather-Aware + Adaptive Threshold + Persistence Filtering)
    thresholds = calculate_adaptive_threshold(times, base_threshold=2.0)
    _, _, candidate_mask = generate_three_category_verdicts(
        z_clim=z_clim,
        z_weather=z_weather,
        thresholds=thresholds,
        observed=obs_vv,
        weather_pred=weather_res["predicted_backscatter_db"].values,
    )
    sign_array = np.sign(obs_vv - weather_res["predicted_backscatter_db"].values)
    pred_m5 = apply_persistence_filter(candidate_mask, min_consecutive_passes=2, sign_array=sign_array)

    # Calculate metrics for all 5 methods
    results = {
        "Image Differencing Baseline (Δσ⁰ < −3 dB)": {
            **compute_classification_metrics(ground_truth_events, pred_m3),
            "mae_db": np.nan,
            "rmse_db": np.nan,
        },
        "Log-Ratio Threshold Baseline (Ratio vs Ref < −3 dB)": {
            **compute_classification_metrics(ground_truth_events, pred_m4),
            "mae_db": np.nan,
            "rmse_db": np.nan,
        },
        "Seasonal Climatology Baseline (|Z_clim| > 2.0)": {
            **compute_classification_metrics(ground_truth_events, pred_m1),
            "mae_db": round(mae_m1, 3),
            "rmse_db": round(rmse_m1, 3),
        },
        "Weather-Aware LightGBM (No Persistence)": {
            **compute_classification_metrics(ground_truth_events, pred_m2),
            "mae_db": round(mae_m2, 3),
            "rmse_db": round(rmse_m2, 3),
        },
        "Complete CounterSAR Pipeline (Weather + Adaptive + Persistence)": {
            **compute_classification_metrics(ground_truth_events, pred_m5),
            "mae_db": round(mae_m2, 3),
            "rmse_db": round(rmse_m2, 3),
        },
    }

    return {
        "benchmark_table": results,
        "n_observations": n,
        "ground_truth_flood_count": int(np.sum(ground_truth_events)),
        "is_synthetic": True,
        "validation_note": "Synthetic benchmark evaluation: results generated via simulated Sentinel-1, CHIRPS, and ERA5 time series. Do not present as operational real-world performance.",
    }

