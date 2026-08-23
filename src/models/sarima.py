"""SARIMA forecasting module for weekly customer demand.

This module implements:
1. Reusable SARIMAForecaster with native prediction intervals and residual diagnostics.
2. Training-only candidate configuration search based on AIC/BIC.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class SARIMADiagnostics:
    """Container for in-sample SARIMA residual diagnostics."""
    residual_mean: float
    residual_std: float
    residual_variance: float
    residual_skewness: float
    residual_kurtosis: float
    ljung_box_results: Dict[str, Dict[str, float]]
    has_significant_autocorrelation_at_5pct: bool
    aic: float
    bic: float
    converged: bool
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    trend: Optional[str]


@dataclass
class SARIMACandidateResult:
    """Container for candidate model search on training data."""
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    trend: Optional[str]
    description: str
    converged: bool
    aic: Optional[float]
    bic: Optional[float]
    error_message: Optional[str] = None


class SARIMAForecaster:
    """SARIMA (Seasonal Autoregressive Integrated Moving Average) Forecaster."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 52),
        trend: Optional[str] = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False
    ):
        """Initialize SARIMA model configuration.

        Args:
            order: (p, d, q) non-seasonal order. Default (0, 1, 1).
            seasonal_order: (P, D, Q, s) seasonal order. Default (0, 1, 1, 52).
            trend: Polynomial trend string ('n', 'c', 't', 'ct' or None).
            enforce_stationarity: Whether to transform AR parameters to be stationary.
            enforce_invertibility: Whether to transform MA parameters to be invertible.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model_: Optional[SARIMAX] = None
        self.fitted_model_: Optional[SARIMAXResults] = None
        self.diagnostics_: Optional[SARIMADiagnostics] = None
        self.is_fitted: bool = False

    def fit(self, y_train: np.ndarray | pd.Series, maxiter: int = 200) -> "SARIMAForecaster":
        """Fit SARIMA model on training history.

        Args:
            y_train: Training demand series.
            maxiter: Maximum optimization iterations.

        Returns:
            self
        """
        y_arr = np.asarray(y_train, dtype=float)
        s_period = self.seasonal_order[3] if len(self.seasonal_order) > 3 else 1
        min_required = 2 * s_period if s_period > 1 else 10

        if len(y_arr) < min_required:
            raise ValueError(
                f"SARIMA with seasonal period {s_period} requires at least {min_required} observations, "
                f"but received {len(y_arr)}."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = SARIMAX(
                y_arr,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            self.fitted_model_ = self.model_.fit(disp=False, maxiter=maxiter)

        self.is_fitted = True

        # Check convergence
        converged = bool(self.fitted_model_.mle_retvals.get("converged", True))

        # In-sample residuals (exclude initial seasonal differencing warmup period)
        d_total = self.order[1] + (self.seasonal_order[1] * s_period)
        warmup = max(1, d_total)
        raw_resid = self.fitted_model_.resid
        residuals = raw_resid[warmup:] if len(raw_resid) > warmup else raw_resid

        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        # Ljung-Box test for residual autocorrelation
        lb_df = acorr_ljungbox(residuals, lags=[5, 10, 15, 20], return_df=True)
        lb_dict: Dict[str, Dict[str, float]] = {}
        has_autocorr = False
        for lag_idx, row in lb_df.iterrows():
            stat = float(row["lb_stat"])
            pval = float(row["lb_pvalue"])
            lb_dict[f"lag_{lag_idx}"] = {"statistic": round(stat, 4), "p_value": round(pval, 4)}
            if pval < 0.05:
                has_autocorr = True

        self.diagnostics_ = SARIMADiagnostics(
            residual_mean=round(res_mean, 2),
            residual_std=round(res_std, 2),
            residual_variance=round(float(res_std ** 2), 2),
            residual_skewness=round(float(stats.skew(residuals)), 4) if len(residuals) > 2 else 0.0,
            residual_kurtosis=round(float(stats.kurtosis(residuals)), 4) if len(residuals) > 2 else 0.0,
            ljung_box_results=lb_dict,
            has_significant_autocorrelation_at_5pct=has_autocorr,
            aic=round(float(self.fitted_model_.aic), 2),
            bic=round(float(self.fitted_model_.bic), 2),
            converged=converged,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend
        )

        return self

    def predict(
        self,
        horizon: int,
        return_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate out-of-sample forecasts and native state-space prediction intervals.

        Args:
            horizon: Forecast horizon steps.
            return_intervals: Whether to compute prediction intervals.
            confidence_level: Confidence level (default 0.95).

        Returns:
            Tuple of (point_forecasts, lower_bounds, upper_bounds)
        """
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")
        if horizon <= 0:
            raise ValueError("Forecast horizon must be a positive integer.")

        alpha = 1.0 - confidence_level
        forecast_res = self.fitted_model_.get_forecast(steps=horizon)
        point_forecast = np.asarray(forecast_res.predicted_mean, dtype=float)

        if not return_intervals:
            return point_forecast, None, None

        # Native SARIMAX confidence/prediction interval from state-space Kalman filter
        conf_int = forecast_res.conf_int(alpha=alpha)
        lower_bound = np.maximum(0.0, conf_int[:, 0])  # Non-negative demand constraint
        upper_bound = conf_int[:, 1]

        return point_forecast, lower_bound, upper_bound

    def get_diagnostics(self) -> SARIMADiagnostics:
        """Return in-sample diagnostic metrics."""
        if not self.is_fitted or self.diagnostics_ is None:
            raise RuntimeError("Model must be fitted before retrieving diagnostics.")
        return self.diagnostics_


def search_sarima_candidates(
    y_train: np.ndarray | pd.Series,
    candidate_configs: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int], Optional[str], str]]] = None,
    maxiter: int = 200
) -> List[SARIMACandidateResult]:
    """Evaluate candidate SARIMA configurations exclusively on the training set using AIC/BIC.

    Args:
        y_train: Training history.
        candidate_configs: List of (order, seasonal_order, trend, description).
        maxiter: Max iterations per fit.

    Returns:
        List of SARIMACandidateResult sorted by training AIC.
    """
    if candidate_configs is None:
        candidate_configs = [
            ((0, 1, 1), (0, 1, 1, 52), None, "Airline Seasonal MA (0,1,1)(0,1,1,52)"),
            ((1, 1, 0), (0, 1, 1, 52), None, "AR(1) with Seasonal MA (1,1,0)(0,1,1,52)"),
            ((1, 1, 1), (0, 1, 1, 52), None, "ARMA(1,1) with Seasonal MA (1,1,1)(0,1,1,52)"),
            ((0, 1, 1), (1, 1, 0, 52), None, "MA(1) with Seasonal AR (0,1,1)(1,1,0,52)"),
            ((1, 1, 1), (1, 1, 0, 52), None, "ARMA(1,1) with Seasonal AR (1,1,1)(1,1,0,52)"),
            ((0, 1, 1), (0, 1, 0, 52), None, "MA(1) with Pure Seasonal Diff (0,1,1)(0,1,0,52)"),
            ((1, 0, 0), (0, 1, 1, 52), "c", "AR(1) Level + Seasonal MA + Constant"),
            ((1, 0, 1), (0, 1, 1, 52), "c", "ARMA(1,1) Level + Seasonal MA + Constant"),
            ((0, 1, 2), (0, 1, 1, 52), None, "MA(2) with Seasonal MA (0,1,2)(0,1,1,52)"),
            ((2, 1, 0), (0, 1, 1, 52), None, "AR(2) with Seasonal MA (2,1,0)(0,1,1,52)"),
            ((0, 1, 1), (1, 1, 1, 52), None, "MA(1) with Seasonal ARMA(1,1) (0,1,1)(1,1,1,52)")
        ]

    results: List[SARIMACandidateResult] = []
    y_arr = np.asarray(y_train, dtype=float)

    for order, s_order, trend, desc in candidate_configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_arr,
                    order=order,
                    seasonal_order=s_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit_res = model.fit(disp=False, maxiter=maxiter)
                converged = bool(fit_res.mle_retvals.get("converged", True))
                aic = float(fit_res.aic)
                bic = float(fit_res.bic)

                results.append(SARIMACandidateResult(
                    order=order,
                    seasonal_order=s_order,
                    trend=trend,
                    description=desc,
                    converged=converged,
                    aic=round(aic, 2),
                    bic=round(bic, 2),
                    error_message=None
                ))
        except Exception as e:
            results.append(SARIMACandidateResult(
                order=order,
                seasonal_order=s_order,
                trend=trend,
                description=desc,
                converged=False,
                aic=None,
                bic=None,
                error_message=str(e)
            ))

    # Sort convergent models by AIC (ascending)
    results.sort(key=lambda x: (x.aic is None, x.aic if x.aic is not None else float("inf")))
    return results
