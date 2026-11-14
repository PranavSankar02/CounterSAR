"""
CounterSAR — Prediction Models (Stage 3)
=========================================
Physics-guided and weather-aware predictive models for SAR backscatter
over the Kuttanad study area (Kerala, India).

Models:
  1. SeasonalClimatologyModel:
     - Baseline statistical climatology using day-of-year harmonic regression
       and monthly mean / standard deviation statistics.
     - Captures annual wet / dry monsoon cycles (SW monsoon flood dip in June–Sep,
       NE monsoon in Oct–Nov, dry recovery in Dec–May).

  2. WeatherAwareLGBMModel:
     - Quantile Gradient Boosting (LightGBM) regressor.
     - Predicts expected SAR backscatter conditioned on location, season,
       rainfall (CHIRPS), soil moisture (ERA5), temperature, and antecedent rainfall.
     - Quantile regression (10th, 50th, 90th percentiles) estimates physical
       asymmetrical prediction uncertainty bounds without Gaussian assumptions.

Scientific Principles:
  - Natural variations in speckle and micro-topography produce normal fluctuations
    (typically ±1 to 2 dB). Small differences between observed and predicted radar
    values are expected and do NOT constitute anomalies or disasters.
  - Anomaly candidates only arise when observed values diverge significantly beyond
    the prediction uncertainty interval across multiple passes (persistence).
  - Synthetic data is strictly tagged throughout; synthetic observations must never
    be cited as evidence of real-world physical model performance.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.skeleton import (
    CRS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    PredictionResult,
)

# Optional LightGBM import with graceful fallback to scikit-learn
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ════════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE SAR MODEL
# ════════════════════════════════════════════════════════════════════════

class BaseSARModel(ABC):
    """
    Abstract interface for SAR backscatter prediction models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
        self.target_var = "vv_db"

    @abstractmethod
    def fit(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        target_var: str = "vv_db",
    ) -> "BaseSARModel":
        """
        Fit model on historical observations.

        Parameters
        ----------
        data : xr.Dataset or pd.DataFrame
            Dataset produced by src/data_source.py containing target and covariates.
        target_var : str
            Name of target backscatter variable (default: 'vv_db').
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        confidence_level: float = 0.90,
    ) -> xr.Dataset:
        """
        Predict expected backscatter and uncertainty bounds.

        Parameters
        ----------
        data : xr.Dataset or pd.DataFrame
            Input covariates.
        confidence_level : float
            Coverage width for uncertainty interval (default: 0.90 for 10th..90th percentile).

        Returns
        -------
        xr.Dataset
            Contains 'predicted_backscatter_db', 'prediction_std',
            'lower_bound_db', 'upper_bound_db', and 'residual_db' (if target present).
        """
        pass

    def to_prediction_results(self, pred_ds: xr.Dataset) -> List[PredictionResult]:
        """
        Convert prediction dataset into a list of shared PredictionResult dataclasses.
        """
        results = []
        times = pd.to_datetime(pred_ds["time"].values)
        y_hat = pred_ds["predicted_backscatter_db"].values
        std = pred_ds["prediction_std"].values
        lower = pred_ds["lower_bound_db"].values
        upper = pred_ds["upper_bound_db"].values
        is_synth = bool(pred_ds.attrs.get("is_synthetic", False))

        for i, t in enumerate(times):
            results.append(
                PredictionResult(
                    date=t.strftime("%Y-%m-%d"),
                    predicted_backscatter_db=float(y_hat[i]),
                    prediction_std=float(std[i]),
                    lower_bound_db=float(lower[i]),
                    upper_bound_db=float(upper[i]),
                    model_name=self.model_name,
                    is_synthetic=is_synth,
                )
            )
        return results


# ════════════════════════════════════════════════════════════════════════
#  1. SEASONAL CLIMATOLOGY BASELINE MODEL
# ════════════════════════════════════════════════════════════════════════

class SeasonalClimatologyModel(BaseSARModel):
    """
    Seasonal Climatology Baseline Model.

    Fits monthly climatological statistics (mean and standard deviation)
    and a smooth dual-frequency Fourier harmonic regression on Day of Year (DOY):
        σ⁰(doy) = a₀ + ∑_{k=1}² [aₖ sin(2π k doy / 365.25) + bₖ cos(2π k doy / 365.25)]

    Captures the seasonal cycle of Kuttanad:
      - Wet season monsoon drop (open water specular reflection)
      - Dry season vegetative & soil roughness recovery
    """

    def __init__(self, num_harmonics: int = 2):
        super().__init__(model_name="seasonal_climatology")
        self.num_harmonics = num_harmonics
        self.harmonic_weights: Optional[np.ndarray] = None
        self.monthly_stats: Dict[int, Dict[str, float]] = {}
        self.residual_std: float = 1.0

    def _harmonic_features(self, doy: np.ndarray) -> np.ndarray:
        """Construct Fourier harmonic design matrix for day-of-year."""
        features = [np.ones_like(doy)]
        for k in range(1, self.num_harmonics + 1):
            omega = 2.0 * np.pi * k / 365.25
            features.append(np.sin(omega * doy))
            features.append(np.cos(omega * doy))
        return np.column_stack(features)

    def fit(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        target_var: str = "vv_db",
    ) -> "SeasonalClimatologyModel":
        """
        Fit harmonic seasonal curve and monthly statistics.
        """
        self.target_var = target_var

        if isinstance(data, xr.Dataset):
            times = pd.to_datetime(data["time"].values)
            y = data[target_var].values
        else:
            time_col = "date" if "date" in data.columns else "time"
            times = pd.to_datetime(data[time_col].values)
            y = data[target_var].values

        # Valid finite mask
        valid = np.isfinite(y)
        if np.sum(valid) < 5:
            raise ValueError(f"Insufficient valid data points to fit climatology: {np.sum(valid)} found.")

        y_valid = y[valid]
        times_valid = times[valid]
        doy = times_valid.dayofyear.values.astype(float)
        months = times_valid.month.values

        # 1. Compute monthly statistics (mean and std per month)
        self.monthly_stats = {}
        for m in range(1, 13):
            mask_m = months == m
            if np.any(mask_m):
                self.monthly_stats[m] = {
                    "mean": float(np.mean(y_valid[mask_m])),
                    "std": float(max(np.std(y_valid[mask_m]), 0.4)),  # Minimum noise floor
                }
            else:
                self.monthly_stats[m] = {"mean": float(np.mean(y_valid)), "std": 1.0}

        # 2. Fit least-squares harmonic regression
        X = self._harmonic_features(doy)
        self.harmonic_weights, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)

        # 3. Calculate baseline residual standard deviation
        preds = X @ self.harmonic_weights
        residuals = y_valid - preds
        self.residual_std = float(max(np.std(residuals), 0.5))

        self.is_fitted = True
        return self

    def predict(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        confidence_level: float = 0.90,
    ) -> xr.Dataset:
        """
        Predict expected seasonal backscatter and uncertainty intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("SeasonalClimatologyModel must be fitted before predict() is called.")

        if isinstance(data, xr.Dataset):
            times = pd.to_datetime(data["time"].values)
            obs_y = data[self.target_var].values if self.target_var in data else None
            is_synth = int(data.attrs.get("is_synthetic", 0))
            coords = {"time": data["time"].values}
        else:
            time_col = "date" if "date" in data.columns else "time"
            times = pd.to_datetime(data[time_col].values)
            obs_y = data[self.target_var].values if self.target_var in data.columns else None
            is_synth = int(data.get("is_synthetic", False).any() if "is_synthetic" in data else 0)
            coords = {"time": times}

        doy = times.dayofyear.values.astype(float)
        X = self._harmonic_features(doy)
        y_hat = X @ self.harmonic_weights

        # Standard deviation for each observation point based on month and residual noise
        months = times.month.values
        stds = np.array([self.monthly_stats.get(m, {}).get("std", self.residual_std) for m in months])
        # Blend monthly standard deviation with residual standard deviation
        combined_std = np.sqrt(0.5 * (stds ** 2 + self.residual_std ** 2))

        # Two-tailed normal multiplier for the requested confidence level
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence_level / 2.0)

        lower_bound = y_hat - z * combined_std
        upper_bound = y_hat + z * combined_std

        data_vars = {
            "predicted_backscatter_db": ("time", y_hat, {"units": "dB", "long_name": "Climatological Expected σ⁰"}),
            "prediction_std": ("time", combined_std, {"units": "dB", "long_name": "Prediction Standard Deviation"}),
            "lower_bound_db": ("time", lower_bound, {"units": "dB", "long_name": f"{int(confidence_level*100)}% Lower Uncertainty Bound"}),
            "upper_bound_db": ("time", upper_bound, {"units": "dB", "long_name": f"{int(confidence_level*100)}% Upper Uncertainty Bound"}),
        }

        if obs_y is not None:
            residuals = obs_y - y_hat
            data_vars["observed_backscatter_db"] = ("time", obs_y, {"units": "dB"})
            data_vars["residual_db"] = ("time", residuals, {"units": "dB", "long_name": "Residual (Observed − Expected)"})

        out_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        out_ds.attrs["model_name"] = self.model_name
        out_ds.attrs["confidence_level"] = confidence_level
        out_ds.attrs["is_synthetic"] = is_synth
        out_ds.attrs["notice"] = (
            "⚠️ SYNTHETIC EVALUATION — Not real satellite observations"
            if is_synth else "Climatology Model Prediction"
        )
        return out_ds


# ════════════════════════════════════════════════════════════════════════
#  2. WEATHER-AWARE PREDICTION MODEL (LightGBM Quantile Regression)
# ════════════════════════════════════════════════════════════════════════

class WeatherAwareLGBMModel(BaseSARModel):
    """
    Weather-Aware Prediction Model using Quantile Gradient Boosting.

    Features:
      - Cyclical calendar: sin/cos Day-of-Year harmonics and month
      - Environmental covariates: daily rainfall, soil moisture, surface temperature
      - Hydrological delay: 3-day and 7-day antecedent cumulative rainfall
      - Radar geometry: orbit pass direction (ascending vs descending)
      - Spatial coordinates: latitude, longitude (when spatial grid is active)

    Quantile Loss Estimation:
      Trains three models for the 10th percentile (lower bound), 50th percentile (expected
      median backscatter), and 90th percentile (upper bound).
      This models asymmetric hydrological responses (e.g. abrupt flood drop vs gradual drying)
      without forcing an unrealistic symmetric Gaussian distribution.
    """

    def __init__(
        self,
        alpha_lower: float = 0.10,
        alpha_upper: float = 0.90,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        super().__init__(model_name="weather_aware_lightgbm")
        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.lower_model: Optional[Any] = None
        self.median_model: Optional[Any] = None
        self.upper_model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.use_lightgbm = HAS_LIGHTGBM

    def _extract_feature_matrix(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DatetimeIndex]:
        """
        Extract numerical feature matrix X and optional target vector y.
        """
        if isinstance(data, xr.Dataset):
            times = pd.to_datetime(data["time"].values)
            n = len(times)

            # Extract or compute antecedent rainfall if missing
            rain = data["rainfall_mm"].values if "rainfall_mm" in data else np.zeros(n)
            sm = data["soil_moisture"].values if "soil_moisture" in data else np.full(n, 0.3)
            temp = data["temperature_c"].values if "temperature_c" in data else np.full(n, 28.0)

            if "antecedent_rain_3d_mm" in data:
                rain_3d = data["antecedent_rain_3d_mm"].values
            else:
                rain_3d = pd.Series(rain).rolling(3, min_periods=1).sum().values

            if "antecedent_rain_7d_mm" in data:
                rain_7d = data["antecedent_rain_7d_mm"].values
            else:
                rain_7d = pd.Series(rain).rolling(7, min_periods=1).sum().values

            # Orbit encoding
            if "orbit" in data:
                orbit_raw = [str(o).lower() for o in data["orbit"].values]
                orbit_enc = np.array([1 if "desc" in o else 0 for o in orbit_raw])
            else:
                orbit_enc = np.zeros(n)

            # Target
            y = data[self.target_var].values if self.target_var in data else None

        else:
            # Pandas DataFrame
            time_col = "date" if "date" in data.columns else "time"
            times = pd.to_datetime(data[time_col].values)
            n = len(times)

            rain = data["rainfall_mm"].values if "rainfall_mm" in data.columns else np.zeros(n)
            sm = data["soil_moisture"].values if "soil_moisture" in data.columns else np.full(n, 0.3)
            temp = data["temperature_c"].values if "temperature_c" in data.columns else np.full(n, 28.0)

            if "antecedent_rain_3d_mm" in data.columns:
                rain_3d = data["antecedent_rain_3d_mm"].values
            else:
                rain_3d = pd.Series(rain).rolling(3, min_periods=1).sum().values

            if "antecedent_rain_7d_mm" in data.columns:
                rain_7d = data["antecedent_rain_7d_mm"].values
            else:
                rain_7d = pd.Series(rain).rolling(7, min_periods=1).sum().values

            if "orbit" in data.columns:
                orbit_raw = [str(o).lower() for o in data["orbit"].values]
                orbit_enc = np.array([1 if "desc" in o else 0 for o in orbit_raw])
            else:
                orbit_enc = np.zeros(n)

            y = data[self.target_var].values if self.target_var in data.columns else None

        # Build cyclical and environmental feature matrix
        doy = times.dayofyear.values.astype(float)
        sin_doy = np.sin(2.0 * np.pi * doy / 365.25)
        cos_doy = np.cos(2.0 * np.pi * doy / 365.25)
        sin_2doy = np.sin(4.0 * np.pi * doy / 365.25)
        cos_2doy = np.cos(4.0 * np.pi * doy / 365.25)
        months = times.month.values.astype(float)

        feature_cols = [
            sin_doy, cos_doy, sin_2doy, cos_2doy, months,
            rain, sm, temp, rain_3d, rain_7d, orbit_enc
        ]
        self.feature_names = [
            "sin_doy", "cos_doy", "sin_2doy", "cos_2doy", "month",
            "rainfall_mm", "soil_moisture", "temperature_c",
            "antecedent_rain_3d_mm", "antecedent_rain_7d_mm", "orbit_direction"
        ]

        X = np.column_stack(feature_cols)

        # Impute any NaNs in features with column median
        for c in range(X.shape[1]):
            nan_mask = np.isnan(X[:, c])
            if np.any(nan_mask):
                median_val = np.nanmedian(X[:, c])
                X[nan_mask, c] = 0.0 if np.isnan(median_val) else median_val

        return X, y, times

    def _init_regressor(self, alpha: float) -> Any:
        """Initialize a single quantile gradient boosting regressor."""
        if self.use_lightgbm:
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=alpha,
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

    def fit(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        target_var: str = "vv_db",
    ) -> "WeatherAwareLGBMModel":
        """
        Fit quantile gradient boosting models on environmental covariates.
        """
        self.target_var = target_var
        X, y, _ = self._extract_feature_matrix(data)

        if y is None:
            raise ValueError(f"Target variable '{target_var}' not found in training data.")

        valid = np.isfinite(y)
        if np.sum(valid) < 10:
            raise ValueError(f"Insufficient valid observations to train LightGBM: {np.sum(valid)} found.")

        X_valid = X[valid]
        y_valid = y[valid]

        # Fit lower quantile (e.g. 10th percentile)
        self.lower_model = self._init_regressor(self.alpha_lower)
        self.lower_model.fit(X_valid, y_valid)

        # Fit median (50th percentile)
        self.median_model = self._init_regressor(0.50)
        self.median_model.fit(X_valid, y_valid)

        # Fit upper quantile (e.g. 90th percentile)
        self.upper_model = self._init_regressor(self.alpha_upper)
        self.upper_model.fit(X_valid, y_valid)

        self.is_fitted = True
        return self

    def predict(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        confidence_level: float = 0.90,
    ) -> xr.Dataset:
        """
        Predict expected radar backscatter conditioned on weather, season, and geography.

        Returns quantile intervals [lower_bound_db, upper_bound_db] and estimated std.
        """
        if not self.is_fitted:
            raise RuntimeError("WeatherAwareLGBMModel must be fitted before predict() is called.")

        X, y_obs, times = self._extract_feature_matrix(data)

        # Generate quantile predictions
        lower_pred = self.lower_model.predict(X)
        median_pred = self.median_model.predict(X)
        upper_pred = self.upper_model.predict(X)

        # Enforce physical monotonicity: lower <= median <= upper
        lower_bound = np.minimum(lower_pred, median_pred)
        upper_bound = np.maximum(upper_pred, median_pred)

        # Estimate equivalent standard deviation from inter-quantile spread:
        # For standard normal, IQR(0.10 to 0.90) = 2 * 1.28155 * std
        z_equiv = 1.28155
        est_std = np.maximum((upper_bound - lower_bound) / (2.0 * z_equiv), 0.3)

        is_synth = 0
        if isinstance(data, xr.Dataset):
            coords = {"time": data["time"].values}
            is_synth = int(data.attrs.get("is_synthetic", 0))
        else:
            time_col = "date" if "date" in data.columns else "time"
            coords = {"time": pd.to_datetime(data[time_col].values)}
            is_synth = int(data.get("is_synthetic", False).any() if "is_synthetic" in data else 0)

        data_vars = {
            "predicted_backscatter_db": (
                "time", median_pred,
                {"units": "dB", "long_name": "Weather-Aware Expected Backscatter (Median)"}
            ),
            "prediction_std": (
                "time", est_std,
                {"units": "dB", "long_name": "Quantile-Estimated Standard Deviation"}
            ),
            "lower_bound_db": (
                "time", lower_bound,
                {"units": "dB", "long_name": f"{int(self.alpha_lower*100)}th Percentile Lower Bound"}
            ),
            "upper_bound_db": (
                "time", upper_bound,
                {"units": "dB", "long_name": f"{int(self.alpha_upper*100)}th Percentile Upper Bound"}
            ),
        }

        if y_obs is not None:
            residuals = y_obs - median_pred
            data_vars["observed_backscatter_db"] = ("time", y_obs, {"units": "dB"})
            data_vars["residual_db"] = (
                "time", residuals,
                {"units": "dB", "long_name": "Residual (Observed − Weather Expected)"}
            )

        out_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        out_ds.attrs["model_name"] = self.model_name
        out_ds.attrs["engine"] = "LightGBM" if self.use_lightgbm else "HistGradientBoosting"
        out_ds.attrs["quantile_lower"] = self.alpha_lower
        out_ds.attrs["quantile_upper"] = self.alpha_upper
        out_ds.attrs["is_synthetic"] = is_synth
        out_ds.attrs["notice"] = (
            "⚠️ SYNTHETIC EVALUATION — Synthetic observations must not be used as evidence of real-world performance"
            if is_synth else "Weather-Aware Quantile Regression Prediction"
        )
        return out_ds


# ════════════════════════════════════════════════════════════════════════
#  3. MODEL EVALUATION & SCIENTIFIC RESIDUAL DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════

def evaluate_model_performance(
    observed: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute regression error metrics and uncertainty interval coverage.

    Scientific Interpretation:
      - Normal radar speckle & seasonal moisture fluctuations generate typical residuals
        of ±1.0 to 1.8 dB. Small residuals do NOT constitute anomalies.
      - A good model exhibits ~80-90% coverage within the [10th, 90th] quantile bounds.
    """
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if np.sum(valid) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "coverage_pct": np.nan}

    y_t = observed[valid]
    y_p = predicted[valid]

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    coverage_pct = np.nan
    if lower_bound is not None and upper_bound is not None:
        lb = lower_bound[valid]
        ub = upper_bound[valid]
        inside = (y_t >= lb) & (y_t <= ub)
        coverage_pct = float(np.mean(inside) * 100.0)

    return {
        "mae_db": round(mae, 3),
        "rmse_db": round(rmse, 3),
        "r2_score": round(r2, 3),
        "interval_coverage_pct": round(coverage_pct, 1),
    }


# ════════════════════════════════════════════════════════════════════════
#  4. REPRODUCIBLE DEMONSTRATION (Synthetic Pipeline Test)
# ════════════════════════════════════════════════════════════════════════

def run_stage3_demo() -> Dict[str, Any]:
    """
    Demonstrate both Seasonal Climatology and Weather-Aware LightGBM models
    on synthetic test data.

    Returns performance metrics comparison dict.
    """
    from src.data_source import create_synthetic_datacube

    # 1. Ingest synthetic data cube from Stage 2 data loader
    cube = create_synthetic_datacube(
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )

    # 2. Train/Test split: 2023 for training, early 2024 for testing
    train_slice = cube.sel(time=slice("2023-01-01", "2023-12-31"))
    test_slice = cube.sel(time=slice("2024-01-01", "2024-06-30"))

    # 3. Model 1: Seasonal Climatology Baseline
    climatology = SeasonalClimatologyModel(num_harmonics=2)
    climatology.fit(train_slice, target_var="vv_db")
    clim_pred = climatology.predict(test_slice)

    clim_metrics = evaluate_model_performance(
        observed=test_slice["vv_db"].values,
        predicted=clim_pred["predicted_backscatter_db"].values,
        lower_bound=clim_pred["lower_bound_db"].values,
        upper_bound=clim_pred["upper_bound_db"].values,
    )

    # 4. Model 2: Weather-Aware LightGBM
    weather_model = WeatherAwareLGBMModel(
        alpha_lower=0.10,
        alpha_upper=0.90,
        n_estimators=60,
        learning_rate=0.08,
    )
    weather_model.fit(train_slice, target_var="vv_db")
    weather_pred = weather_model.predict(test_slice)

    weather_metrics = evaluate_model_performance(
        observed=test_slice["vv_db"].values,
        predicted=weather_pred["predicted_backscatter_db"].values,
        lower_bound=weather_pred["lower_bound_db"].values,
        upper_bound=weather_pred["upper_bound_db"].values,
    )

    return {
        "climatology_metrics": clim_metrics,
        "weather_aware_metrics": weather_metrics,
        "climatology_predictions": clim_pred,
        "weather_predictions": weather_pred,
        "is_synthetic": True,
        "notice": "⚠️ NOTE: Evaluated on synthetic demonstration data. Synthetic observations must not be used as evidence of real-world model performance.",
    }
