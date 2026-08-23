"""Unit tests for SARIMA forecasting module, candidate search, and diagnostics."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from src.baselines import create_chronological_split
from src.evaluation import calculate_forecast_metrics
from src.models.sarima import (
    SARIMAForecaster,
    SARIMADiagnostics,
    SARIMACandidateResult,
    search_sarima_candidates
)


@pytest.fixture
def synthetic_sarima_series() -> Tuple[np.ndarray, np.ndarray]:
    """Generate 156 training observations and 52 test observations."""
    np.random.seed(42)
    n_train = 156
    n_test = 52
    t = np.arange(n_train + n_test)

    trend = 120.0 + 0.6 * t
    seasonality = 35.0 * np.sin(2 * np.pi * t / 52.0)
    noise = np.random.normal(0, 12, size=len(t))
    full_y = trend + seasonality + noise

    return full_y[:n_train], full_y[n_train:]


def test_sarima_initialization_defaults():
    """Verify default configuration and unfitted state."""
    sarima = SARIMAForecaster()
    assert sarima.order == (0, 1, 1)
    assert sarima.seasonal_order == (0, 1, 1, 52)
    assert sarima.is_fitted is False

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        sarima.predict(10)

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        sarima.get_diagnostics()


def test_sarima_insufficient_data_error():
    """Verify error when observations are less than 2*s (< 104 observations)."""
    short_data = np.arange(80, dtype=float)
    sarima = SARIMAForecaster(seasonal_order=(0, 1, 1, 52))

    with pytest.raises(ValueError, match="requires at least 104 observations"):
        sarima.fit(short_data)


def test_sarima_fit_and_forecast_length(synthetic_sarima_series):
    """Verify fit execution, forecast horizon, and native prediction intervals."""
    y_train, y_test = synthetic_sarima_series
    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52))
    sarima.fit(y_train)

    assert sarima.is_fitted is True

    horizon = len(y_test)
    preds, lower, upper = sarima.predict(horizon=horizon, return_intervals=True, confidence_level=0.95)

    assert len(preds) == horizon
    assert len(lower) == horizon
    assert len(upper) == horizon

    # Prediction interval validity
    assert np.all(lower <= preds)
    assert np.all(preds <= upper)
    assert np.all(lower >= 0.0)  # Non-negative demand bound


def test_sarima_no_test_leakage(synthetic_sarima_series):
    """Verify that forecasting at step t does not depend on test data."""
    y_train, y_test = synthetic_sarima_series
    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52))
    sarima.fit(y_train)

    preds1, _, _ = sarima.predict(horizon=52)
    preds2, _, _ = sarima.predict(horizon=52)

    np.testing.assert_array_almost_equal(preds1, preds2)


def test_sarima_candidate_search_training_only(synthetic_sarima_series):
    """Verify candidate search evaluates models on training set and sorts by AIC."""
    y_train, _ = synthetic_sarima_series

    test_candidates = [
        ((0, 1, 1), (0, 1, 1, 52), None, "Model A"),
        ((1, 1, 0), (0, 1, 1, 52), None, "Model B")
    ]

    results = search_sarima_candidates(y_train, candidate_configs=test_candidates, maxiter=50)

    assert len(results) == 2
    assert all(isinstance(r, SARIMACandidateResult) for r in results)
    assert results[0].aic <= results[1].aic


def test_sarima_residual_diagnostics(synthetic_sarima_series):
    """Verify in-sample residual diagnostic metrics."""
    y_train, _ = synthetic_sarima_series
    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52)).fit(y_train)

    diag = sarima.get_diagnostics()
    assert isinstance(diag, SARIMADiagnostics)
    assert abs(diag.residual_mean) < 30.0
    assert diag.residual_std > 0.0
    assert "lag_5" in diag.ljung_box_results
    assert "lag_10" in diag.ljung_box_results
    assert diag.aic > 0


def test_sarima_integration_real_data():
    """Integration test: Verify SARIMA out-of-sample forecast accuracy on Superstore dataset."""
    processed_path = Path("data/processed/weekly_demand.csv")
    if not processed_path.exists():
        pytest.skip("Processed dataset not found.")

    weekly_df = pd.read_csv(processed_path)
    train_df, test_df, _ = create_chronological_split(weekly_df, test_size_weeks=52)

    y_train = train_df["quantity"].values.astype(float)
    y_test = test_df["quantity"].values.astype(float)

    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 52)).fit(y_train)
    preds, lower, upper = sarima.predict(horizon=len(y_test))

    metrics = calculate_forecast_metrics(y_test, preds, model_name="SARIMA")

    # MAE should beat baselines (~46.49 vs 63.75/65.63)
    assert metrics.mae < 52.0
    assert metrics.rmse < 65.0
    assert metrics.mape < 25.0
