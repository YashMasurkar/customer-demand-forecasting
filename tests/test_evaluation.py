"""Unit tests for forecast evaluation metrics module."""

import pytest
import numpy as np
from src.evaluation import calculate_forecast_metrics, ForecastMetrics


def test_calculate_forecast_metrics_exact_values():
    """Verify exact hand-calculated metric formulas."""
    y_true = np.array([100.0, 200.0, 150.0, 50.0])
    y_pred = np.array([110.0, 180.0, 150.0, 60.0])

    # errors = [-10, +20, 0, -10]
    # abs_errors = [10, 20, 0, 10] -> sum = 40, mean MAE = 10.0
    # squared_errors = [100, 400, 0, 100] -> mean MSE = 150.0, RMSE = sqrt(150) = 12.2474...
    # MAPE = mean([10/100, 20/200, 0/150, 10/50]) * 100 = mean([0.1, 0.1, 0.0, 0.2]) * 100 = 10.0%
    # Mean Error (bias) = mean([-10, 20, 0, -10]) = 0.0
    # Max Error = 20.0

    metrics = calculate_forecast_metrics(y_true, y_pred, model_name="TestModel")

    assert isinstance(metrics, ForecastMetrics)
    assert metrics.mae == 10.0
    assert metrics.rmse == round(np.sqrt(150.0), 2)
    assert metrics.mape == 10.0
    assert metrics.mape_applicable is True
    assert metrics.mean_error == 0.0
    assert metrics.max_error == 20.0
    assert metrics.num_eval_points == 4


def test_calculate_forecast_metrics_length_mismatch():
    """Verify error on length mismatch."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_forecast_metrics([1, 2], [1, 2, 3])


def test_calculate_forecast_metrics_empty():
    """Verify error on empty inputs."""
    with pytest.raises(ValueError, match="Cannot calculate metrics on empty arrays"):
        calculate_forecast_metrics([], [])


def test_calculate_forecast_metrics_with_zeros():
    """Verify that MAPE is excluded when actual values contain zero."""
    y_true = np.array([0.0, 50.0, 100.0])
    y_pred = np.array([10.0, 40.0, 110.0])

    metrics = calculate_forecast_metrics(y_true, y_pred, model_name="ZeroActualsModel")

    assert metrics.mape is None
    assert metrics.mape_applicable is False
    assert "contains zero" in metrics.mape_notes or "division by zero" in metrics.mape_notes
    assert metrics.mae == 10.0
    assert metrics.smape > 0  # sMAPE handles zero gracefully
