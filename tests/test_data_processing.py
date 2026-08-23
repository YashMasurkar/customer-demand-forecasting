"""Unit tests for weekly demand transformation and validation pipeline."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_inspection import load_raw_dataset
from src.data_processing import (
    parse_and_validate_dates,
    compute_week_start,
    aggregate_weekly_demand,
    save_processed_dataset,
    WeeklyValidationProfile
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return load_raw_dataset("data/raw/Sample_Superstore.csv")


def test_parse_and_validate_dates_success(raw_df: pd.DataFrame):
    """Test date parsing on the real dataset."""
    dates = parse_and_validate_dates(raw_df, date_col="Order Date")
    assert isinstance(dates, pd.Series)
    assert len(dates) == 9994
    assert dates.isnull().sum() == 0
    assert dates.min() == pd.Timestamp("2014-01-03")
    assert dates.max() == pd.Timestamp("2017-12-30")


def test_parse_and_validate_dates_invalid_column(raw_df: pd.DataFrame):
    """Test error when date column is absent."""
    with pytest.raises(ValueError, match="Date column 'NonExistent' not found"):
        parse_and_validate_dates(raw_df, date_col="NonExistent")


def test_parse_and_validate_dates_unparseable():
    """Test error when date strings are corrupt/unparseable."""
    bad_df = pd.DataFrame({"Order Date": ["not-a-date", "2020-01-01"]})
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_and_validate_dates(bad_df, date_col="Order Date")


def test_compute_week_start_monday_alignment():
    """Verify that dates correctly align to Monday 00:00:00."""
    test_dates = pd.Series(pd.to_datetime([
        "2014-01-03",  # Friday -> Monday 2013-12-30
        "2014-01-05",  # Sunday -> Monday 2013-12-30
        "2014-01-06",  # Monday -> Monday 2014-01-06
        "2017-12-30",  # Saturday -> Monday 2017-12-25
    ]))
    week_starts = compute_week_start(test_dates)

    expected = pd.Series(pd.to_datetime([
        "2013-12-30",
        "2013-12-30",
        "2014-01-06",
        "2017-12-25"
    ]))
    pd.testing.assert_series_equal(week_starts, expected)


def test_aggregate_weekly_demand_reconciliation(raw_df: pd.DataFrame):
    """Test exact Quantity reconciliation sum(weekly Quantity) == sum(raw Quantity)."""
    weekly_df, profile = aggregate_weekly_demand(raw_df)

    assert profile.reconciliation_passed is True
    assert profile.total_raw_quantity == 37873
    assert profile.total_weekly_quantity == 37873
    assert weekly_df["quantity"].sum() == 37873


def test_aggregate_weekly_demand_dimensions(raw_df: pd.DataFrame):
    """Verify weekly series dimensions and date boundaries."""
    weekly_df, profile = aggregate_weekly_demand(raw_df)

    assert profile.num_weeks == 209
    assert profile.first_week == "2013-12-30"
    assert profile.last_week == "2017-12-25"
    assert profile.missing_weeks_count == 0
    assert profile.zero_demand_weeks_count == 0
    assert profile.duplicate_weeks_count == 0

    # Partial week detection check
    assert "is_partial_week" in weekly_df.columns
    assert weekly_df.loc[0, "is_partial_week"] == True
    assert weekly_df.loc[1:, "is_partial_week"].sum() == 0  # Only week 0 is partial
    assert len(profile.partial_weeks) == 1
    assert profile.partial_weeks[0]["week_start"] == "2013-12-30"
    assert "Exclude initial partial week" in profile.partial_week_recommendation

    assert "week_start" in weekly_df.columns
    assert "week_end" in weekly_df.columns
    assert "quantity" in weekly_df.columns
    assert "sales" in weekly_df.columns
    assert "order_count" in weekly_df.columns


def test_aggregate_weekly_demand_missing_week_detection():
    """Verify that gaps in transactional dates are detected and filled with 0 demand."""
    synthetic_gap_df = pd.DataFrame({
        "Row ID": [1, 2],
        "Order ID": ["CA-1", "CA-2"],
        "Order Date": ["01/06/2020", "01/20/2020"],  # Week 1 (2020-01-06) and Week 3 (2020-01-20); Week 2 missing
        "Quantity": [10, 20],
        "Sales": [100.0, 200.0],
        "Profit": [10.0, 20.0],
        "Discount": [0.0, 0.0]
    })

    weekly_df, profile = aggregate_weekly_demand(synthetic_gap_df)

    assert profile.num_weeks == 3
    assert profile.missing_weeks_count == 1
    assert profile.missing_weeks == ["2020-01-13"]
    assert profile.zero_demand_weeks_count == 1
    assert profile.zero_demand_weeks == ["2020-01-13"]
    assert profile.total_weekly_quantity == 30
    assert profile.reconciliation_passed is True

    # Verify zero-filled row
    missing_row = weekly_df[weekly_df["week_start"] == pd.Timestamp("2020-01-13")]
    assert len(missing_row) == 1
    assert missing_row["quantity"].iloc[0] == 0
    assert missing_row["sales"].iloc[0] == 0.0


def test_save_processed_dataset(raw_df: pd.DataFrame, tmp_path: Path):
    """Test saving processed dataset to CSV."""
    weekly_df, _ = aggregate_weekly_demand(raw_df)
    target_file = tmp_path / "test_weekly_demand.csv"
    saved_path = save_processed_dataset(weekly_df, target_file)

    assert saved_path.exists()
    reloaded = pd.read_csv(saved_path)
    assert reloaded.shape == (209, weekly_df.shape[1])
    assert reloaded["quantity"].sum() == 37873
