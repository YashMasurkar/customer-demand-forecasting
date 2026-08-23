"""Unit tests for weekly time-series statistical exploration module."""

import pytest
import pandas as pd
import numpy as np
from src.data_inspection import load_raw_dataset
from src.data_processing import aggregate_weekly_demand
from src.time_series_exploration import explore_weekly_time_series, TimeSeriesExplorationReport


@pytest.fixture
def weekly_data() -> pd.DataFrame:
    raw_df = load_raw_dataset("data/raw/Sample_Superstore.csv")
    weekly_df, _ = aggregate_weekly_demand(raw_df)
    return weekly_df


def test_explore_weekly_time_series_structure(weekly_data: pd.DataFrame):
    """Verify that exploration produces a well-structured diagnostic report."""
    report = explore_weekly_time_series(weekly_data)

    assert isinstance(report, TimeSeriesExplorationReport)
    assert report.num_observations == 209
    assert report.date_range["start"] == "2013-12-30"
    assert report.date_range["end"] == "2017-12-25"

    # Summary stats
    assert report.summary_stats["mean"] == pytest.approx(181.21, rel=1e-2)
    assert report.summary_stats["min"] == 13.0
    assert report.summary_stats["max"] == 564.0

    # Trend
    assert report.trend_analysis["linear_slope_per_week"] > 0
    assert report.trend_analysis["direction"] == "Upward"

    # Seasonality / Autocorrelation
    assert report.seasonality_evidence["lag_1_autocorrelation"] > 0.5
    assert report.seasonality_evidence["lag_52_autocorrelation"] > 0.4
    assert report.seasonality_evidence["has_annual_seasonality_signal"] is True

    # Annual YoY
    assert "2014" in report.annual_patterns
    assert "2017" in report.annual_patterns
    assert report.annual_patterns["2017"]["total_demand"] > report.annual_patterns["2014"]["total_demand"]

    # Stationarity diagnostics
    assert "adf_test" in report.stationarity_tests
    assert "kpss_test" in report.stationarity_tests
    assert report.stationarity_tests["adf_test"]["is_stationary_at_5pct"] is True


def test_explore_weekly_time_series_insufficient_observations():
    """Verify error on series with fewer than 10 observations."""
    short_df = pd.DataFrame({
        "week_start": pd.date_range("2020-01-06", periods=5, freq="W-MON"),
        "quantity": [10, 20, 15, 25, 30]
    })
    with pytest.raises(ValueError, match="Insufficient observations"):
        explore_weekly_time_series(short_df)


def test_explore_weekly_extreme_weeks(weekly_data: pd.DataFrame):
    """Verify extreme high and low week detection."""
    report = explore_weekly_time_series(weekly_data)

    assert report.extreme_weeks["high_weeks_count"] == 12
    # Verify that the peak week 2017-11-27 (564 units) is captured
    high_dates = [w["week"] for w in report.extreme_weeks["high_weeks"]]
    assert "2017-11-27" in high_dates
