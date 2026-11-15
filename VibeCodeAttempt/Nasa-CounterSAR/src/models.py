from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.skeleton import PredictionResult

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseSARModel(ABC):

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
        pass

    @abstractmethod
    def predict(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        confidence_level: float = 0.90,
    ) -> xr.Dataset:
        pass

    def to_prediction_results(
        self,
        pred_ds: xr.Dataset,
        y_index: int = 0,
        x_index: int = 0,
    ) -> List[PredictionResult]:

        if "y" not in pred_ds.dims or "x" not in pred_ds.dims:
            raise ValueError("Prediction dataset must contain spatial y and x dimensions.")

        if not (0 <= y_index < pred_ds.sizes["y"]):
            raise IndexError("y_index is outside the prediction grid.")

        if not (0 <= x_index < pred_ds.sizes["x"]):
            raise IndexError("x_index is outside the prediction grid.")

        times = pd.to_datetime(pred_ds["time"].values)

        variables = [
            "predicted_backscatter_db",
            "prediction_std",
            "lower_bound_db",
            "upper_bound_db",
        ]

        values = {
            name: pred_ds[name].isel(y=y_index, x=x_index).values
            for name in variables
        }

        is_synth = bool(pred_ds.attrs.get("is_synthetic", False))
        results = []

        for i, timestamp in enumerate(times):
            row = [values[name][i] for name in variables]

            if not np.all(np.isfinite(row)):
                continue

            results.append(
                PredictionResult(
                    date=timestamp.strftime("%Y-%m-%d"),
                    predicted_backscatter_db=float(row[0]),
                    prediction_std=float(row[1]),
                    lower_bound_db=float(row[2]),
                    upper_bound_db=float(row[3]),
                    model_name=self.model_name,
                    is_synthetic=is_synth,
                )
            )

        return results


class SeasonalClimatologyModel(BaseSARModel):

    def __init__(self, num_harmonics: int = 2):
        super().__init__("seasonal_climatology")
        self.num_harmonics = num_harmonics
        self.harmonic_weights = None
        self.residual_std = None
        self.spatial_shape = None
        self.grid_coords = None
        self.valid_pixel_mask = None

    def _harmonic_features(self, doy: np.ndarray) -> np.ndarray:
        features = [np.ones_like(doy, dtype=np.float64)]

        for k in range(1, self.num_harmonics + 1):
            omega = 2.0 * np.pi * k / 365.25
            features.append(np.sin(omega * doy))
            features.append(np.cos(omega * doy))

        return np.column_stack(features)

    def _get_target(self, data: xr.Dataset, target_var: str) -> np.ndarray:
        if target_var not in data:
            raise ValueError(f"Target variable '{target_var}' not found.")

        target = data[target_var]

        if "time" not in target.dims:
            raise ValueError("Target variable must contain a time dimension.")

        if "y" not in target.dims or "x" not in target.dims:
            raise ValueError("Target variable must contain spatial y and x dimensions.")

        return np.asarray(
            target.transpose("time", "y", "x").values,
            dtype=np.float32,
        )

    def fit(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        target_var: str = "vv_db",
    ) -> "SeasonalClimatologyModel":

        if not isinstance(data, xr.Dataset):
            raise TypeError("SeasonalClimatologyModel requires an xarray Dataset.")

        self.target_var = target_var

        times = pd.DatetimeIndex(pd.to_datetime(data["time"].values))
        values = self._get_target(data, target_var)

        if len(times) < 5:
            raise ValueError("At least five acquisition dates are required.")

        n_time, height, width = values.shape
        n_pixels = height * width

        y = values.reshape(n_time, n_pixels).astype(np.float64)
        valid = np.isfinite(y)

        valid_counts = valid.sum(axis=0)
        valid_pixels = valid_counts >= max(5, self.num_harmonics * 2 + 1)

        if not np.any(valid_pixels):
            raise ValueError("No spatial pixels have sufficient valid observations.")

        doy = times.dayofyear.values.astype(float)
        X = self._harmonic_features(doy)

        y_fit = np.where(valid, y, 0.0)
        weights = valid.astype(np.float64)

        xtwx = np.einsum(
            "tf,tp,tg->pfg",
            X,
            weights,
            X,
            optimize=True,
        )

        xtwy = np.einsum(
            "tf,tp,tp->pf",
            X,
            weights,
            y_fit,
            optimize=True,
        )

        coefficients = np.zeros(
            (n_pixels, X.shape[1]),
            dtype=np.float64,
        )

        coefficients[valid_pixels] = (
            np.linalg.pinv(xtwx[valid_pixels])
            @ xtwy[valid_pixels, :, None]
        ).squeeze(-1)

        predictions = X @ coefficients.T
        residuals = np.where(valid, y - predictions, np.nan)

        counts = valid.sum(axis=0)

        residual_mean = (
            np.nansum(residuals, axis=0)
            / np.maximum(counts, 1)
        )

        residual_variance = (
            np.nansum(
                (residuals - residual_mean[None, :]) ** 2,
                axis=0,
            )
            / np.maximum(counts, 1)
        )

        residual_std = np.maximum(
            np.sqrt(residual_variance),
            0.5,
        )

        residual_std[~valid_pixels] = np.nan

        predictions[~valid] = np.nan

        self.harmonic_weights = coefficients.reshape(
            height,
            width,
            X.shape[1],
        )

        self.residual_std = residual_std.reshape(height, width)
        self.spatial_shape = (height, width)
        self.valid_pixel_mask = valid_pixels.reshape(height, width)

        self.grid_coords = {
            "y": data["y"].values,
            "x": data["x"].values,
            "time": data["time"].values,
        }

        self.is_fitted = True
        return self

    def predict(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        confidence_level: float = 0.90,
    ) -> xr.Dataset:

        if not self.is_fitted:
            raise RuntimeError("SeasonalClimatologyModel must be fitted first.")

        if not isinstance(data, xr.Dataset):
            raise TypeError("Prediction requires an xarray Dataset.")

        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")

        times = pd.DatetimeIndex(pd.to_datetime(data["time"].values))
        height, width = self.spatial_shape

        if data.sizes["y"] != height or data.sizes["x"] != width:
            raise ValueError("Prediction grid does not match the training grid.")

        doy = times.dayofyear.values.astype(float)
        X = self._harmonic_features(doy)

        coefficients = self.harmonic_weights.reshape(
            height * width,
            -1,
        )

        predictions = X @ coefficients.T
        predictions = predictions.reshape(len(times), height, width)

        std = np.broadcast_to(
            self.residual_std,
            (len(times), height, width),
        ).copy()

        from scipy.stats import norm

        z = norm.ppf(0.5 + confidence_level / 2.0)

        lower = predictions - z * std
        upper = predictions + z * std

        valid_mask = np.broadcast_to(
            self.valid_pixel_mask,
            predictions.shape,
        )

        predictions[~valid_mask] = np.nan
        std[~valid_mask] = np.nan
        lower[~valid_mask] = np.nan
        upper[~valid_mask] = np.nan

        data_vars = {
            "predicted_backscatter_db": (
                ("time", "y", "x"),
                predictions.astype(np.float32),
                {"units": "dB"},
            ),
            "prediction_std": (
                ("time", "y", "x"),
                std.astype(np.float32),
                {"units": "dB"},
            ),
            "lower_bound_db": (
                ("time", "y", "x"),
                lower.astype(np.float32),
                {"units": "dB"},
            ),
            "upper_bound_db": (
                ("time", "y", "x"),
                upper.astype(np.float32),
                {"units": "dB"},
            ),
        }

        if self.target_var in data:
            observed = self._get_target(data, self.target_var)
            residual = observed - predictions

            data_vars["observed_backscatter_db"] = (
                ("time", "y", "x"),
                observed,
                {"units": "dB"},
            )

            data_vars["residual_db"] = (
                ("time", "y", "x"),
                residual.astype(np.float32),
                {"units": "dB"},
            )

        out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": data["time"].values,
                "y": data["y"].values,
                "x": data["x"].values,
            },
        )

        out.attrs.update(data.attrs)
        out.attrs["model_name"] = self.model_name
        out.attrs["confidence_level"] = confidence_level
        out.attrs["is_synthetic"] = bool(data.attrs.get("is_synthetic", False))
        out.attrs["spatial_prediction"] = True

        return out


class WeatherAwareLGBMModel(BaseSARModel):

    def __init__(
        self,
        alpha_lower: float = 0.10,
        alpha_upper: float = 0.90,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
        max_training_samples: int = 100000,
        prediction_batch_size: int = 250000,
    ):
        super().__init__("weather_aware_lightgbm")

        if not 0 < alpha_lower < 0.5:
            raise ValueError("alpha_lower must be between 0 and 0.5.")

        if not 0.5 < alpha_upper < 1:
            raise ValueError("alpha_upper must be between 0.5 and 1.")

        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_training_samples = max_training_samples
        self.prediction_batch_size = prediction_batch_size

        self.lower_model = None
        self.median_model = None
        self.upper_model = None
        self.feature_names = []
        self.use_lightgbm = HAS_LIGHTGBM
        self.spatial_shape = None
        self.grid_coords = None
        self.valid_pixel_mask = None

    def _spatial_arrays(
        self,
        data: xr.Dataset,
        target_var: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DatetimeIndex]:

        if not isinstance(data, xr.Dataset):
            raise TypeError("WeatherAwareLGBMModel requires an xarray Dataset.")

        times = pd.DatetimeIndex(pd.to_datetime(data["time"].values))

        if "y" not in data.dims or "x" not in data.dims:
            raise ValueError("Dataset must contain spatial y and x dimensions.")

        height = data.sizes["y"]
        width = data.sizes["x"]
        n_time = len(times)

        def get_array(name, default):
            if name not in data:
                return np.full(
                    (n_time, height, width),
                    default,
                    dtype=np.float32,
                )

            da = data[name]

            if "time" not in da.dims:
                values = np.asarray(da.transpose("y", "x").values)
                return np.broadcast_to(
                    values,
                    (n_time, height, width),
                ).astype(np.float32)

            return np.asarray(
                da.transpose("time", "y", "x").values,
                dtype=np.float32,
            )

        rain = get_array("rainfall_mm", 0.0)
        soil = get_array("soil_moisture", 0.3)
        temperature = get_array("temperature_c", 28.0)

        if "antecedent_rain_3d_mm" in data:
            rain_3d = get_array("antecedent_rain_3d_mm", 0.0)
        else:
            rain_3d = (
                pd.Series(rain[:, 0, 0])
                .rolling(3, min_periods=1)
                .sum()
                .values[:, None, None]
            )
            rain_3d = np.broadcast_to(rain_3d, rain.shape).copy()

        if "antecedent_rain_7d_mm" in data:
            rain_7d = get_array("antecedent_rain_7d_mm", 0.0)
        else:
            rain_7d = (
                pd.Series(rain[:, 0, 0])
                .rolling(7, min_periods=1)
                .sum()
                .values[:, None, None]
            )
            rain_7d = np.broadcast_to(rain_7d, rain.shape).copy()

        if "orbit" in data:
            orbit_values = np.asarray(data["orbit"].values).astype(str)
            orbit = np.array(
                [
                    1.0 if "desc" in value.lower() else 0.0
                    for value in orbit_values
                ],
                dtype=np.float32,
            )
        else:
            orbit = np.zeros(n_time, dtype=np.float32)

        doy = times.dayofyear.values.astype(np.float32)
        months = times.month.values.astype(np.float32)

        sin_doy = np.sin(2.0 * np.pi * doy / 365.25)
        cos_doy = np.cos(2.0 * np.pi * doy / 365.25)
        sin_2doy = np.sin(4.0 * np.pi * doy / 365.25)
        cos_2doy = np.cos(4.0 * np.pi * doy / 365.25)

        y_coords = np.asarray(data["y"].values, dtype=np.float32)
        x_coords = np.asarray(data["x"].values, dtype=np.float32)

        y_normalized = (
            (y_coords - np.mean(y_coords))
            / max(np.std(y_coords), 1.0)
        )

        x_normalized = (
            (x_coords - np.mean(x_coords))
            / max(np.std(x_coords), 1.0)
        )

        y_grid = np.broadcast_to(
            y_normalized[None, :, None],
            (n_time, height, width),
        )

        x_grid = np.broadcast_to(
            x_normalized[None, None, :],
            (n_time, height, width),
        )

        water = get_array("water_mask", 0.0)
        paddy = get_array("paddy_mask", 0.0)

        orbit_grid = np.broadcast_to(
            orbit[:, None, None],
            rain.shape,
        )

        time_features = [
            np.broadcast_to(feature[:, None, None], rain.shape)
            for feature in (
                sin_doy,
                cos_doy,
                sin_2doy,
                cos_2doy,
                months / 12.0,
            )
        ]

        feature_arrays = time_features + [
            rain,
            soil,
            temperature,
            rain_3d,
            rain_7d,
            orbit_grid,
            x_grid,
            y_grid,
            water,
            paddy,
        ]

        feature_names = [
            "sin_doy",
            "cos_doy",
            "sin_2doy",
            "cos_2doy",
            "month",
            "rainfall_mm",
            "soil_moisture",
            "temperature_c",
            "antecedent_rain_3d_mm",
            "antecedent_rain_7d_mm",
            "orbit_direction",
            "x_normalized",
            "y_normalized",
            "water_mask",
            "paddy_mask",
        ]

        X = np.stack(feature_arrays, axis=-1)
        X = X.reshape(-1, len(feature_names))

        for column in range(X.shape[1]):
            values = X[:, column]
            invalid = ~np.isfinite(values)

            if np.any(invalid):
                finite = values[~invalid]
                replacement = float(np.median(finite)) if finite.size else 0.0
                values[invalid] = replacement

        y = None

        if target_var is not None and target_var in data:
            y = np.asarray(
                data[target_var].transpose("time", "y", "x").values,
                dtype=np.float32,
            ).reshape(-1)

        self.feature_names = feature_names

        return X.astype(np.float32), y, times

    def _init_regressor(self, alpha: float) -> Any:

        if self.use_lightgbm:
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
            )

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

        if not isinstance(data, xr.Dataset):
            raise TypeError("WeatherAwareLGBMModel requires an xarray Dataset.")

        self.target_var = target_var

        X, y, _ = self._spatial_arrays(data, target_var)

        if y is None:
            raise ValueError(f"Target variable '{target_var}' not found.")

        valid_indices = np.flatnonzero(np.isfinite(y))

        if len(valid_indices) < 100:
            raise ValueError("Insufficient valid spatial observations for training.")

        rng = np.random.default_rng(self.random_state)

        if len(valid_indices) > self.max_training_samples:
            selected = rng.choice(
                valid_indices,
                size=self.max_training_samples,
                replace=False,
            )
        else:
            selected = valid_indices

        X_train = X[selected]
        y_train = y[selected]

        self.lower_model = self._init_regressor(self.alpha_lower)
        self.median_model = self._init_regressor(0.50)
        self.upper_model = self._init_regressor(self.alpha_upper)

        self.lower_model.fit(X_train, y_train)
        self.median_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)

        self.spatial_shape = (
            data.sizes["y"],
            data.sizes["x"],
        )

        self.grid_coords = {
            "y": data["y"].values,
            "x": data["x"].values,
            "time": data["time"].values,
        }

        self.valid_pixel_mask = np.isfinite(
            data[target_var].isel(time=0).values
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        data: Union[xr.Dataset, pd.DataFrame],
        confidence_level: float = 0.90,
    ) -> xr.Dataset:

        if not self.is_fitted:
            raise RuntimeError("WeatherAwareLGBMModel must be fitted first.")

        if not isinstance(data, xr.Dataset):
            raise TypeError("Prediction requires an xarray Dataset.")

        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")

        if (
            data.sizes["y"],
            data.sizes["x"],
        ) != self.spatial_shape:
            raise ValueError("Prediction grid does not match training grid.")

        X, y_obs, times = self._spatial_arrays(data, self.target_var)

        total = len(X)

        lower = np.empty(total, dtype=np.float32)
        median = np.empty(total, dtype=np.float32)
        upper = np.empty(total, dtype=np.float32)

        for start in range(0, total, self.prediction_batch_size):
            end = min(start + self.prediction_batch_size, total)
            batch = X[start:end]

            lower[start:end] = self.lower_model.predict(batch)
            median[start:end] = self.median_model.predict(batch)
            upper[start:end] = self.upper_model.predict(batch)

        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)

        std = np.maximum(
            (upper - lower) / (2.0 * 1.28155),
            0.3,
        )

        n_time = len(times)
        height, width = self.spatial_shape
        shape = (n_time, height, width)

        median = median.reshape(shape)
        lower = lower.reshape(shape)
        upper = upper.reshape(shape)
        std = std.reshape(shape)

        valid_mask = np.broadcast_to(
            self.valid_pixel_mask,
            shape,
        )

        median[~valid_mask] = np.nan
        lower[~valid_mask] = np.nan
        upper[~valid_mask] = np.nan
        std[~valid_mask] = np.nan

        data_vars = {
            "predicted_backscatter_db": (
                ("time", "y", "x"),
                median,
                {"units": "dB"},
            ),
            "prediction_std": (
                ("time", "y", "x"),
                std,
                {"units": "dB"},
            ),
            "lower_bound_db": (
                ("time", "y", "x"),
                lower,
                {"units": "dB"},
            ),
            "upper_bound_db": (
                ("time", "y", "x"),
                upper,
                {"units": "dB"},
            ),
        }

        if y_obs is not None:
            observed = y_obs.reshape(shape)
            residual = observed - median

            data_vars["observed_backscatter_db"] = (
                ("time", "y", "x"),
                observed,
                {"units": "dB"},
            )

            data_vars["residual_db"] = (
                ("time", "y", "x"),
                residual,
                {"units": "dB"},
            )

        out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": data["time"].values,
                "y": data["y"].values,
                "x": data["x"].values,
            },
        )

        out.attrs.update(data.attrs)
        out.attrs["model_name"] = self.model_name
        out.attrs["engine"] = (
            "LightGBM" if self.use_lightgbm else "HistGradientBoosting"
        )
        out.attrs["quantile_lower"] = self.alpha_lower
        out.attrs["quantile_upper"] = self.alpha_upper
        out.attrs["is_synthetic"] = bool(data.attrs.get("is_synthetic", False))
        out.attrs["spatial_prediction"] = True
        out.attrs["training_sample_limit"] = self.max_training_samples

        return out


def evaluate_model_performance(
    observed: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
) -> Dict[str, float]:

    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted arrays must have matching shapes.")

    valid = np.isfinite(observed) & np.isfinite(predicted)

    if not np.any(valid):
        return {
            "mae_db": np.nan,
            "rmse_db": np.nan,
            "r2_score": np.nan,
            "interval_coverage_pct": np.nan,
            "valid_samples": 0,
        }

    y_true = observed[valid]
    y_pred = predicted[valid]

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    r2 = (
        float(r2_score(y_true, y_pred))
        if len(y_true) > 1
        else np.nan
    )

    coverage = np.nan

    if lower_bound is not None and upper_bound is not None:
        lower_bound = np.asarray(lower_bound)
        upper_bound = np.asarray(upper_bound)

        if lower_bound.shape != observed.shape or upper_bound.shape != observed.shape:
            raise ValueError("Uncertainty bounds must match the observed array shape.")

        valid_interval = (
            valid
            & np.isfinite(lower_bound)
            & np.isfinite(upper_bound)
        )

        if np.any(valid_interval):
            inside = (
                (observed[valid_interval] >= lower_bound[valid_interval])
                & (observed[valid_interval] <= upper_bound[valid_interval])
            )

            coverage = float(np.mean(inside) * 100.0)

    return {
        "mae_db": round(mae, 3),
        "rmse_db": round(rmse, 3),
        "r2_score": round(r2, 3) if np.isfinite(r2) else np.nan,
        "interval_coverage_pct": round(coverage, 1) if np.isfinite(coverage) else np.nan,
        "valid_samples": int(len(y_true)),
    }


def run_stage3_demo() -> Dict[str, Any]:

    from app.mock_data import generate_spatial_datacube

    cube = generate_spatial_datacube()

    train_slice = cube.sel(
        time=slice("2023-01-01", "2023-12-31")
    )

    test_slice = cube.sel(
        time=slice("2024-01-01", "2024-06-30")
    )

    if train_slice.sizes["time"] < 5 or test_slice.sizes["time"] < 1:
        raise ValueError("Insufficient training or testing dates.")

    climatology = SeasonalClimatologyModel(num_harmonics=2)
    climatology.fit(train_slice, target_var="vv_db")
    clim_pred = climatology.predict(test_slice)

    clim_metrics = evaluate_model_performance(
        observed=test_slice["vv_db"].values,
        predicted=clim_pred["predicted_backscatter_db"].values,
        lower_bound=clim_pred["lower_bound_db"].values,
        upper_bound=clim_pred["upper_bound_db"].values,
    )

    weather_model = WeatherAwareLGBMModel(
        alpha_lower=0.10,
        alpha_upper=0.90,
        n_estimators=60,
        learning_rate=0.08,
        max_training_samples=100000,
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
        "notice": (
            "Synthetic demonstration only. "
            "These metrics do not establish real-world model performance."
        ),
    }