"""Forecast evaluation metrics module for time-series demand forecasting."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Container for forecast accuracy evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    mape: Optional[float]
    smape: float
    mean_error: float
    max_error: float
    num_eval_points: int
    mape_applicable: bool
    mape_notes: str


def calculate_forecast_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    model_name: str = "Model"
) -> ForecastMetrics:
    """Calculate MAE, RMSE, MAPE, sMAPE, Mean Error (bias), and Max Error.

    Args:
        y_true: Actual ground-truth values.
        y_pred: Predicted forecast values.
        model_name: Descriptive label for the evaluated model.

    Returns:
        ForecastMetrics dataclass.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) != y_pred ({len(y_p)})")
    if len(y_t) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    errors = y_t - y_p
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_error = float(np.mean(errors))  # Positive = underprediction, Negative = overprediction
    max_error = float(np.max(abs_errors))

    # Evaluate MAPE applicability
    has_zero_actuals = np.any(y_t == 0)
    has_near_zero_actuals = np.any(np.abs(y_t) < 1e-5)

    if has_zero_actuals or has_near_zero_actuals:
        mape = None
        mape_applicable = False
        mape_notes = "MAPE excluded: Actual values contain zero or near-zero observations, causing division by zero."
    else:
        mape = float(np.mean(abs_errors / y_t) * 100)
        mape_applicable = True
        min_actual = float(np.min(y_t))
        mape_notes = (
            f"MAPE is mathematically well-defined (all actual test values >= {min_actual:.1f} > 0). "
            "Note that percentage errors can still be asymmetric between low-demand and high-demand periods."
        )

    # Symmetric MAPE (sMAPE)
    denominator = (np.abs(y_t) + np.abs(y_p))
    # Handle edge case where both true and pred are 0
    zero_mask = denominator == 0
    smape_elements = np.zeros_like(denominator)
    smape_elements[~zero_mask] = (200.0 * abs_errors[~zero_mask]) / denominator[~zero_mask]
    smape = float(np.mean(smape_elements))

    return ForecastMetrics(
        model_name=model_name,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        mape=round(mape, 2) if mape is not None else None,
        smape=round(smape, 2),
        mean_error=round(mean_error, 2),
        max_error=round(max_error, 2),
        num_eval_points=len(y_t),
        mape_applicable=mape_applicable,
        mape_notes=mape_notes
    )
