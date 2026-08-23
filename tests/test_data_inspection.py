"""Unit tests for dataset inspection and profile logic."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_inspection import (
    load_raw_dataset,
    inspect_dataset,
    _summarize_temporal_grain,
    DatasetProfile
)


@pytest.fixture
def raw_data_path() -> Path:
    return Path("data/raw/Sample_Superstore.csv")


def test_load_raw_dataset_success(raw_data_path: Path):
    """Test loading the real raw CSV file."""
    df = load_raw_dataset(raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (9994, 21)


def test_load_raw_dataset_file_not_found():
    """Test FileNotFoundError on non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_raw_dataset("data/raw/non_existent_file.csv")


def test_inspect_dataset_schema_and_dimensions(raw_data_path: Path):
    """Verify raw dataset schema, column count, and row count."""
    df = load_raw_dataset(raw_data_path)
    profile = inspect_dataset(df, file_path=str(raw_data_path))

    assert profile.row_count == 9994
    assert profile.column_count == 21
    assert profile.duplicate_rows_count == 0
    assert len(profile.columns) == 21
    assert "Order Date" in profile.columns
    assert "Quantity" in profile.columns
    assert "Sales" in profile.columns
    assert "Category" in profile.columns


def test_inspect_dataset_missing_values(raw_data_path: Path):
    """Verify zero missing values in standard raw Superstore CSV."""
    df = load_raw_dataset(raw_data_path)
    profile = inspect_dataset(df, file_path=str(raw_data_path))

    for col, missing_cnt in profile.missing_values.items():
        assert missing_cnt == 0, f"Column {col} has unexpected {missing_cnt} missing values"


def test_inspect_dataset_entity_counts(raw_data_path: Path):
    """Verify categorical and entity hierarchy counts."""
    df = load_raw_dataset(raw_data_path)
    profile = inspect_dataset(df, file_path=str(raw_data_path))

    assert profile.unique_counts["unique_categories"] == 3
    assert profile.unique_counts["unique_sub_categories"] == 17
    assert profile.unique_counts["unique_regions"] == 4
    assert profile.unique_counts["unique_customers"] == 793
    assert profile.unique_counts["unique_orders"] == 5009

    # Category breakdown verification
    assert set(profile.category_breakdown.keys()) == {"Office Supplies", "Furniture", "Technology"}
    assert sum(profile.category_breakdown.values()) == 9994


def test_inspect_dataset_date_range(raw_data_path: Path):
    """Verify min/max date coverage and span."""
    df = load_raw_dataset(raw_data_path)
    profile = inspect_dataset(df, file_path=str(raw_data_path))

    assert profile.date_range["order_date_min"] == "2014-01-03"
    assert profile.date_range["order_date_max"] == "2017-12-30"
    assert profile.date_range["total_calendar_days"] == 1458
    assert profile.date_range["active_order_days"] == 1237
    assert profile.date_range["days_without_orders"] == 221
    assert profile.date_range["unparsed_date_count"] == 0


def test_inspect_dataset_temporal_aggregations(raw_data_path: Path):
    """Verify total quantity consistency and period counts across daily, weekly, monthly grains."""
    df = load_raw_dataset(raw_data_path)
    profile = inspect_dataset(df, file_path=str(raw_data_path))

    total_qty = 37873.0

    # Daily
    assert profile.daily_summary["total_periods"] == 1458
    assert profile.daily_summary["zero_demand_periods"] == 221
    assert profile.daily_summary["demand_quantity"]["total"] == total_qty

    # Weekly
    assert profile.weekly_summary["total_periods"] == 209
    assert profile.weekly_summary["zero_demand_periods"] == 0
    assert profile.weekly_summary["demand_quantity"]["total"] == total_qty

    # Monthly
    assert profile.monthly_summary["total_periods"] == 48
    assert profile.monthly_summary["zero_demand_periods"] == 0
    assert profile.monthly_summary["demand_quantity"]["total"] == total_qty


def test_inspect_dataset_synthetic_edge_cases():
    """Verify inspect_dataset behavior on synthetic data with nulls and duplicates."""
    synthetic_data = pd.DataFrame({
        "Row ID": [1, 2, 3, 3],
        "Order ID": ["CA-1", "CA-2", "CA-3", "CA-3"],
        "Order Date": ["01/01/2020", "01/02/2020", "01/03/2020", "01/03/2020"],
        "Customer ID": ["C-1", "C-2", "C-3", "C-3"],
        "Category": ["Tech", None, "Office", "Office"],
        "Sub-Category": ["Phones", "Chairs", "Paper", "Paper"],
        "Region": ["East", "West", "Central", "Central"],
        "Sales": [100.0, 200.0, 50.0, 50.0],
        "Quantity": [2, 4, 1, 1],
        "Profit": [10.0, -5.0, 5.0, 5.0],
        "Discount": [0.0, 0.1, 0.0, 0.0]
    })

    profile = inspect_dataset(synthetic_data, file_path="synthetic_test.csv")
    assert profile.row_count == 4
    assert profile.duplicate_rows_count == 1
    assert profile.missing_values["Category"] == 1
    assert profile.missing_percentage["Category"] == 25.0
