"""Unit tests for feature engineering, lag correctness, rolling window integrity, and leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from src.features.feature_engineering import (
    build_weekly_feature_dataset,
    generate_feature_leakage_audit,
    extract_weekly_business_aggregations
)


@pytest.fixture
def synthetic_weekly_demand_df() -> pd.DataFrame:
    """Create a synthetic 100-week demand dataset with integer index."""
    dates = pd.date_range("2014-01-06", periods=100, freq="W-MON")
    np.random.seed(42)
    qty = np.random.randint(50, 300, size=100)
    sales = qty * 15.5
    profit = qty * 3.2
    orders = np.random.randint(5, 30, size=100)

    df = pd.DataFrame({
        "week_start": dates.strftime("%Y-%m-%d"),
        "week_end": (dates + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
        "year": dates.year,
        "week_of_year": dates.isocalendar().week,
        "is_partial_week": [False] * 100,
        "quantity": qty,
        "sales": sales,
        "profit": profit,
        "order_count": orders,
        "transaction_count": orders * 2,
        "avg_discount": np.random.uniform(0.05, 0.25, size=100)
    })
    return df.reset_index(drop=True)


def test_feature_generation_columns_and_shape(synthetic_weekly_demand_df: pd.DataFrame):
    """Verify that build_weekly_feature_dataset creates all required features."""
    feat_df, feature_cols = build_weekly_feature_dataset(synthetic_weekly_demand_df)

    assert len(feat_df) == len(synthetic_weekly_demand_df)
    assert "trend_index" in feature_cols
    assert "sin_woy" in feature_cols
    assert "cos_woy" in feature_cols
    assert "lag_1" in feature_cols
    assert "lag_52" in feature_cols
    assert "rolling_mean_4" in feature_cols
    assert "rolling_std_13" in feature_cols
    assert "lag_1_sales" in feature_cols
    assert "lag_52_sales" in feature_cols


def test_target_lags_shift_direction_exact_match(synthetic_weekly_demand_df: pd.DataFrame):
    """Verify that lag_k at row t strictly equals target[t-k] and contains no forward leakage."""
    feat_df, _ = build_weekly_feature_dataset(synthetic_weekly_demand_df)

    for lag_k in [1, 2, 4, 8, 13, 26, 52]:
        lag_col = f"lag_{lag_k}"
        # For rows before lag_k, lag_col must be NaN
        assert feat_df[lag_col].iloc[:lag_k].isna().all()
        # For all subsequent rows t >= lag_k, lag_col[t] == quantity[t - lag_k]
        for t in range(lag_k, len(feat_df)):
            assert feat_df[lag_col].iloc[t] == synthetic_weekly_demand_df["quantity"].iloc[t - lag_k]


def test_rolling_features_no_current_period_leakage(synthetic_weekly_demand_df: pd.DataFrame):
    """Verify that rolling statistics at row t strictly exclude actual target at row t."""
    feat_df, _ = build_weekly_feature_dataset(synthetic_weekly_demand_df)

    # For row t, rolling_mean_4 must equal the mean of quantity[t-4, t-3, t-2, t-1]
    for t in range(4, len(feat_df)):
        expected_window = synthetic_weekly_demand_df["quantity"].iloc[t - 4:t].values
        expected_mean = np.mean(expected_window)
        expected_std = np.std(expected_window, ddof=1)

        actual_mean = feat_df["rolling_mean_4"].iloc[t]
        actual_std = feat_df["rolling_std_4"].iloc[t]

        assert actual_mean == pytest.approx(expected_mean, rel=1e-5)
        assert actual_std == pytest.approx(expected_std, rel=1e-5)


def test_lagged_business_features_shift_direction(synthetic_weekly_demand_df: pd.DataFrame):
    """Verify that business features (sales, profit, orders) are strictly shifted backward."""
    feat_df, _ = build_weekly_feature_dataset(synthetic_weekly_demand_df)

    for t in range(1, len(feat_df)):
        assert feat_df["lag_1_sales"].iloc[t] == synthetic_weekly_demand_df["sales"].iloc[t - 1]
        assert feat_df["lag_1_profit"].iloc[t] == synthetic_weekly_demand_df["profit"].iloc[t - 1]
        assert feat_df["lag_1_order_count"].iloc[t] == synthetic_weekly_demand_df["order_count"].iloc[t - 1]


def test_prohibited_contemporaneous_features_excluded(synthetic_weekly_demand_df: pd.DataFrame):
    """Verify that raw contemporaneous business metrics and target are never in feature_cols."""
    _, feature_cols = build_weekly_feature_dataset(synthetic_weekly_demand_df)

    prohibited_names = [
        "quantity", "target_quantity", "sales", "profit", "avg_discount",
        "order_count", "transaction_count", "unique_customers", "week_start", "week_end"
    ]

    for p in prohibited_names:
        assert p not in feature_cols, f"Leakage violation: prohibited contemporaneous feature '{p}' found in feature_cols"


def test_feature_leakage_audit_contents():
    """Verify that the Feature Leakage Audit table is complete and valid."""
    audit_df = generate_feature_leakage_audit()

    assert isinstance(audit_df, pd.DataFrame)
    assert len(audit_df) >= 30
    assert "Feature" in audit_df.columns
    assert "Temporal Class" in audit_df.columns
    assert "Status" in audit_df.columns
    assert "Rationale" in audit_df.columns

    # Verify both ALLOWED and PROHIBITED entries exist
    statuses = set(audit_df["Status"].unique())
    assert "ALLOWED" in statuses
    assert any("PROHIBITED" in s for s in statuses)
