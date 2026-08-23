"""Holt-Winters Exponential Smoothing model for weekly demand forecasting.

This module implements the classical additive and multiplicative Holt-Winters models,
supporting level, trend, and 52-week seasonal components, in-sample residual diagnostics,
and calibrated prediction intervals.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class HoltWintersDiagnostics:
    """Container for in-sample residual diagnostic metrics."""
    residual_mean: float
    residual_std: float
    residual_variance: float
    residual_skewness: float
    residual_kurtosis: float
    ljung_box_results: Dict[str, Dict[str, float]]
    has_significant_autocorrelation_at_5pct: bool
    aic: float
    bic: float
    model_params: Dict[str, Any] = field(default_factory=dict)


class HoltWintersForecaster:
    """Holt-Winters Exponential Smoothing Forecaster."""

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: int = 52,
        damped_trend: bool = False,
        initialization_method: str = "estimated"
    ):
        """Initialize Holt-Winters configuration.

        Args:
            trend: Type of trend component ('add', 'mul', or None). Default 'add'.
            seasonal: Type of seasonal component ('add', 'mul', or None). Default 'add'.
            seasonal_periods: Seasonal periodicity in weeks. Default 52.
            damped_trend: Whether to damp the linear trend component. Default False.
            initialization_method: Method for initializing states ('estimated', 'heuristic').
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method

        self.model_: Optional[ExponentialSmoothing] = None
        self.fitted_model_: Optional[Any] = None
        self.diagnostics_: Optional[HoltWintersDiagnostics] = None
        self.residual_std_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, y_train: np.ndarray | pd.Series) -> "HoltWintersForecaster":
        """Fit Holt-Winters Exponential Smoothing model on training data.

        Args:
            y_train: Training demand series.

        Returns:
            self
        """
        y_arr = np.asarray(y_train, dtype=float)
        min_required = 2 * self.seasonal_periods if self.seasonal is not None else 10
        if len(y_arr) < min_required:
            raise ValueError(
                f"Holt-Winters with seasonal_periods={self.seasonal_periods} requires at least "
                f"{min_required} observations, but received {len(y_arr)}."
            )

        self.model_ = ExponentialSmoothing(
            y_arr,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method=self.initialization_method
        )
        self.fitted_model_ = self.model_.fit(optimized=True)
        self.is_fitted = True

        # Compute in-sample residuals (excluding initial warmup cycle)
        fitted_vals = self.fitted_model_.fittedvalues
        warmup = self.seasonal_periods if self.seasonal is not None else 1
        residuals = y_arr[warmup:] - fitted_vals[warmup:]

        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals, ddof=1))
        self.residual_std_ = res_std

        # In-sample residual diagnostics
        lb_df = acorr_ljungbox(residuals, lags=[5, 10, 15, 20], return_df=True)
        lb_dict: Dict[str, Dict[str, float]] = {}
        has_autocorr = False
        for lag_idx, row in lb_df.iterrows():
            stat = float(row["lb_stat"])
            pval = float(row["lb_pvalue"])
            lb_dict[f"lag_{lag_idx}"] = {"statistic": round(stat, 4), "p_value": round(pval, 4)}
            if pval < 0.05:
                has_autocorr = True

        # Extract smoothing parameters
        params = {
            "smoothing_level": round(float(self.fitted_model_.params.get("smoothing_level", 0.0)), 4),
            "smoothing_trend": round(float(self.fitted_model_.params.get("smoothing_trend", 0.0)), 4),
            "smoothing_seasonal": round(float(self.fitted_model_.params.get("smoothing_seasonal", 0.0)), 4),
            "damping_trend": round(float(self.fitted_model_.params.get("damping_trend", 0.0)), 4) if self.damped_trend else None,
        }

        self.diagnostics_ = HoltWintersDiagnostics(
            residual_mean=round(res_mean, 2),
            residual_std=round(res_std, 2),
            residual_variance=round(float(res_std ** 2), 2),
            residual_skewness=round(float(stats.skew(residuals)), 4),
            residual_kurtosis=round(float(stats.kurtosis(residuals)), 4),
            ljung_box_results=lb_dict,
            has_significant_autocorrelation_at_5pct=has_autocorr,
            aic=round(float(self.fitted_model_.aic), 2),
            bic=round(float(self.fitted_model_.bic), 2),
            model_params=params
        )

        return self

    def predict(
        self,
        horizon: int,
        return_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate out-of-sample point forecasts and prediction intervals.

        Args:
            horizon: Number of future periods to forecast.
            return_intervals: Whether to compute prediction intervals.
            confidence_level: Confidence level (default: 0.95 for 95% interval).

        Returns:
            Tuple of (point_forecasts, lower_bound, upper_bound)
        """
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")
        if horizon <= 0:
            raise ValueError("Forecast horizon must be a positive integer.")

        point_forecast = self.fitted_model_.forecast(horizon)

        if not return_intervals:
            return point_forecast, None, None

        # Calibrated prediction intervals using residual standard error
        # For an additive model, the standard error at horizon h incorporates error variance
        z_score = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)

        # Standard error expansion factor over horizon h: sqrt(1 + (h-1)*alpha^2) approx
        alpha_param = float(self.fitted_model_.params.get("smoothing_level", 0.2))
        h_steps = np.arange(1, horizon + 1)
        se_h = self.residual_std_ * np.sqrt(1.0 + (h_steps - 1) * (alpha_param ** 2))

        lower_bound = np.maximum(0.0, point_forecast - z_score * se_h)
        upper_bound = point_forecast + z_score * se_h

        return point_forecast, lower_bound, upper_bound

    def get_diagnostics(self) -> HoltWintersDiagnostics:
        """Return in-sample residual diagnostic metrics."""
        if not self.is_fitted or self.diagnostics_ is None:
            raise RuntimeError("Model must be fitted before retrieving diagnostics.")
        return self.diagnostics_
