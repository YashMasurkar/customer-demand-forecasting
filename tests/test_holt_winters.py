"""Unit tests for Holt-Winters Exponential Smoothing forecasting model."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from src.baselines import create_chronological_split
from src.evaluation import calculate_forecast_metrics
from src.models.holt_winters import HoltWintersForecaster, HoltWintersDiagnostics


@pytest.fixture
def synthetic_seasonal_series() -> Tuple[np.ndarray, np.ndarray]:
    """Generate 156 training weeks (3 years) and 52 test weeks (1 year) with trend + seasonality."""
    np.random.seed(42)
    n_train = 156
    n_test = 52
    t = np.arange(n_train + n_test)

    # Linear trend + 52-week sine seasonality + noise
    trend = 100.0 + 0.8 * t
    seasonality = 40.0 * np.sin(2 * np.pi * t / 52.0)
    noise = np.random.normal(0, 10, size=len(t))
    full_y = trend + seasonality + noise

    return full_y[:n_train], full_y[n_train:]


def test_holt_winters_initialization_defaults():
    """Verify default parameters and un-fitted state."""
    hw = HoltWintersForecaster()
    assert hw.trend == "add"
    assert hw.seasonal == "add"
    assert hw.seasonal_periods == 52
    assert hw.damped_trend is False
    assert hw.is_fitted is False

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        hw.predict(10)

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        hw.get_diagnostics()


def test_holt_winters_insufficient_data_error():
    """Verify error when series length is less than 2 seasonal cycles (< 104 weeks)."""
    short_data = np.arange(80, dtype=float)
    hw = HoltWintersForecaster(seasonal_periods=52)

    with pytest.raises(ValueError, match="requires at least 104 observations"):
        hw.fit(short_data)


def test_holt_winters_fit_and_forecast_length(synthetic_seasonal_series):
    """Verify fit execution, forecast length, and output dimensions."""
    y_train, y_test = synthetic_seasonal_series
    hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=52)
    hw.fit(y_train)

    assert hw.is_fitted is True

    # Multi-step forecast
    horizon = len(y_test)
    preds, lower, upper = hw.predict(horizon=horizon, return_intervals=True, confidence_level=0.95)

    assert len(preds) == horizon
    assert len(lower) == horizon
    assert len(upper) == horizon

    # Prediction interval validity
    assert np.all(lower <= preds)
    assert np.all(preds <= upper)
    assert np.all(lower >= 0.0)  # Non-negative lower bound


def test_holt_winters_prediction_interval_methods(synthetic_seasonal_series):
    """Verify both analytical_approx and constant_residual interval methods."""
    y_train, y_test = synthetic_seasonal_series
    hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=52).fit(y_train)

    # Method A: Constant residual
    preds_a, low_a, up_a = hw.predict(horizon=10, interval_method="constant_residual")
    # Method B: Analytical approx
    preds_b, low_b, up_b = hw.predict(horizon=10, interval_method="analytical_approx")

    assert np.all(preds_a == preds_b)
    # Horizon expansion makes analytical approx interval at h=10 wider than constant residual
    assert (up_b[9] - low_b[9]) >= (up_a[9] - low_a[9])

    # Invalid method error
    with pytest.raises(ValueError, match="Unknown interval_method"):
        hw.predict(horizon=10, interval_method="invalid_method")


def test_holt_winters_no_test_leakage(synthetic_seasonal_series):
    """Verify that predictions at step t do not depend on test set data."""
    y_train, y_test = synthetic_seasonal_series
    hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=52)
    hw.fit(y_train)

    # Forecast should be deterministic given training series
    preds1, _, _ = hw.predict(horizon=52)
    preds2, _, _ = hw.predict(horizon=52)

    np.testing.assert_array_almost_equal(preds1, preds2)


def test_holt_winters_residual_diagnostics(synthetic_seasonal_series):
    """Verify in-sample residual diagnostic calculations and gamma documentation."""
    y_train, _ = synthetic_seasonal_series
    hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=52).fit(y_train)

    diag = hw.get_diagnostics()
    assert isinstance(diag, HoltWintersDiagnostics)
    assert abs(diag.residual_mean) < 20.0  # Near-zero mean
    assert diag.residual_std > 0.0
    assert diag.residual_variance == pytest.approx(diag.residual_std ** 2, rel=1e-2)
    assert "lag_5" in diag.ljung_box_results
    assert "lag_10" in diag.ljung_box_results
    assert "smoothing_level" in diag.model_params
    assert "smoothing_trend" in diag.model_params
    assert len(diag.seasonal_parameter_interpretation) > 0
    assert "gamma" in diag.seasonal_parameter_interpretation


def test_holt_winters_on_real_data_accuracy():
    """Integration test: Verify Holt-Winters out-of-sample accuracy on real Superstore test set."""
    processed_path = Path("data/processed/weekly_demand.csv")
    if not processed_path.exists():
        pytest.skip("Processed dataset not found.")

    weekly_df = pd.read_csv(processed_path)
    train_df, test_df, _ = create_chronological_split(weekly_df, test_size_weeks=52)

    y_train = train_df["quantity"].values.astype(float)
    y_test = test_df["quantity"].values.astype(float)

    hw = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=52).fit(y_train)
    preds, lower, upper = hw.predict(horizon=len(y_test))

    metrics = calculate_forecast_metrics(y_test, preds, model_name="Holt-Winters")

    # MAE should be substantially lower than baseline (~39.02 vs 63.75/65.63)
    assert metrics.mae < 45.0
    assert metrics.rmse < 60.0
    assert metrics.mape < 22.0
    assert abs(metrics.mean_error) < 10.0  # Low bias
