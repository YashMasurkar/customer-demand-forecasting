"""Unit tests for baseline forecasting models, chronological splitting, and leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.baselines import (
    create_chronological_split,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    run_baseline_evaluation,
    BaselineEvaluationResult
)


@pytest.fixture
def sample_weekly_df() -> pd.DataFrame:
    # 209 rows matching real dataset structure (1 partial week + 208 complete weeks)
    dates = pd.date_range(start="2013-12-30", periods=209, freq="W-MON")
    is_partial = [True] + [False] * 208
    np.random.seed(42)
    # Generate positive synthetic demand
    quantity = np.random.randint(50, 400, size=209)
    quantity[0] = 13  # Partial week demand

    return pd.DataFrame({
        "week_start": dates,
        "quantity": quantity,
        "is_partial_week": is_partial
    })


def test_create_chronological_split_strict_ordering(sample_weekly_df: pd.DataFrame):
    """Verify strictly chronological split with partial week excluded and no temporal leakage."""
    train_df, test_df, split_info = create_chronological_split(
        sample_weekly_df,
        test_size_weeks=52
    )

    assert split_info.total_complete_observations == 208
    assert split_info.num_train_observations == 156
    assert split_info.num_test_observations == 52
    assert split_info.partial_weeks_excluded_count == 1
    assert split_info.partial_weeks_excluded == ["2013-12-30"]

    # Strict temporal boundary check (train strictly before test)
    assert train_df["week_start"].max() < test_df["week_start"].min()
    assert train_df["week_start"].min() == pd.Timestamp("2014-01-06")
    assert test_df["week_start"].min() == pd.Timestamp("2017-01-02")
    assert test_df["week_start"].max() == pd.Timestamp("2017-12-25")

    # No overlapping timestamps
    overlap = set(train_df["week_start"]).intersection(set(test_df["week_start"]))
    assert len(overlap) == 0


def test_create_chronological_split_insufficient_data():
    """Verify error when series is too short for requested test size."""
    short_df = pd.DataFrame({
        "week_start": pd.date_range("2020-01-06", periods=30, freq="W-MON"),
        "quantity": np.random.randint(10, 100, size=30),
        "is_partial_week": [False] * 30
    })
    with pytest.raises(ValueError, match="Insufficient complete observations"):
        create_chronological_split(short_df, test_size_weeks=52)


def test_naive_forecaster_logic():
    """Verify 1-step rolling and multi-step naive predictions."""
    y_train = np.array([100.0, 150.0, 200.0])
    y_test = np.array([220.0, 250.0, 280.0])

    forecaster = NaiveForecaster()
    # Cannot predict before fitting
    with pytest.raises(RuntimeError):
        forecaster.predict_multi_step_fixed(3)

    forecaster.fit(y_train)

    # 1-step rolling: test[0] uses train[-1] (200), test[1] uses test[0] (220), test[2] uses test[1] (250)
    preds_1step = forecaster.predict_one_step_rolling(y_test)
    expected_1step = np.array([200.0, 220.0, 250.0])
    np.testing.assert_array_equal(preds_1step, expected_1step)

    # Multi-step fixed: all 3 steps hold train[-1] (200)
    preds_fixed = forecaster.predict_multi_step_fixed(horizon=3)
    expected_fixed = np.array([200.0, 200.0, 200.0])
    np.testing.assert_array_equal(preds_fixed, expected_fixed)


def test_seasonal_naive_forecaster_lag_52():
    """Verify lag-52 seasonal naive mapping."""
    # Synthetic series of 104 weeks (2 full years)
    year1 = np.arange(1, 53, dtype=float)
    year2 = np.arange(53, 105, dtype=float)
    full_series = np.concatenate([year1, year2])

    train = full_series[:52]
    test = full_series[52:]

    snaive = SeasonalNaiveForecaster(seasonal_period=52).fit(train)

    preds = snaive.predict(full_series=full_series, test_start_idx=52, test_len=52)
    # Prediction for test (year 2) should be exact year 1 values
    np.testing.assert_array_equal(preds, year1)


def test_seasonal_naive_insufficient_history():
    """Verify that fitting with fewer than 52 observations raises ValueError."""
    short_train = np.arange(30, dtype=float)
    snaive = SeasonalNaiveForecaster(seasonal_period=52)

    with pytest.raises(ValueError, match="requires at least 52 training observations"):
        snaive.fit(short_train)


def test_no_future_leakage_in_baseline_evaluation(sample_weekly_df: pd.DataFrame):
    """Verify that predictions at step t do not use information from step > t."""
    result = run_baseline_evaluation(sample_weekly_df, target_col="quantity", test_size_weeks=52)

    assert isinstance(result, BaselineEvaluationResult)
    forecast_df = result.forecast_df

    # Verify Naive 1-step at step 0 is the last training value
    train_df, test_df, _ = create_chronological_split(sample_weekly_df, test_size_weeks=52)
    assert forecast_df.loc[0, "naive_1step_pred"] == train_df["quantity"].iloc[-1]

    # Verify for subsequent steps, naive prediction at row i matches actual from row i-1
    for i in range(1, len(forecast_df)):
        assert forecast_df.loc[i, "naive_1step_pred"] == forecast_df.loc[i - 1, "actual_demand"]


def test_run_baseline_evaluation_on_real_data():
    """Integration test: Verify execution and baseline metrics on real processed dataset."""
    processed_path = Path("data/processed/weekly_demand.csv")
    if not processed_path.exists():
        pytest.skip("Processed dataset not found.")

    df = pd.read_csv(processed_path)
    result = run_baseline_evaluation(df, target_col="quantity", test_size_weeks=52)

    assert result.split_info.num_train_observations == 156
    assert result.split_info.num_test_observations == 52

    # Verify Naive metrics are close to expected values
    assert result.naive_1step_metrics.mae == pytest.approx(63.75, abs=0.5)
    assert result.naive_1step_metrics.rmse == pytest.approx(84.39, abs=0.5)

    # Verify Seasonal Naive metrics are close to expected values
    assert result.seasonal_naive_metrics.mae == pytest.approx(65.63, abs=0.5)
    assert result.seasonal_naive_metrics.rmse == pytest.approx(84.66, abs=0.5)

    assert result.naive_1step_metrics.mape_applicable is True
