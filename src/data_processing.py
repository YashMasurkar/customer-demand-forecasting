"""Data transformation and processing pipeline for weekly demand forecasting.

This module provides functions to parse dates, perform weekly aggregation,
build continuous time-series indices, detect missing or zero-demand weeks,
and validate demand quantity reconciliation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class WeeklyValidationProfile:
    """Container for weekly dataset validation profile and reconciliation."""
    first_week: str
    last_week: str
    num_weeks: int
    total_raw_quantity: int
    total_weekly_quantity: int
    reconciliation_passed: bool
    mean_weekly_quantity: float
    median_weekly_quantity: float
    std_weekly_quantity: float
    min_weekly_quantity: float
    max_weekly_quantity: float
    missing_weeks_count: int
    missing_weeks: List[str]
    zero_demand_weeks_count: int
    zero_demand_weeks: List[str]
    duplicate_weeks_count: int
    partial_weeks: List[Dict[str, Any]] = field(default_factory=list)
    partial_week_recommendation: str = ""
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


def parse_and_validate_dates(
    df: pd.DataFrame,
    date_col: str = "Order Date"
) -> pd.Series:
    """Parse and validate date column in the raw dataset."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    # Try explicit format first, then fallback
    parsed = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")
    if parsed.isnull().any():
        parsed = pd.to_datetime(df[date_col], format="mixed", errors="coerce")

    if parsed.isnull().any():
        null_count = int(parsed.isnull().sum())
        raise ValueError(f"Failed to parse {null_count} dates in column '{date_col}'.")

    return parsed


def compute_week_start(dates: pd.Series) -> pd.Series:
    """Assign each timestamp to the Monday of its calendar week (W-MON).

    Definition:
    Week Start is defined as Monday 00:00:00.
    A transaction on Friday 2014-01-03 belongs to the week starting Monday 2013-12-30.
    """
    return dates.dt.to_period("W-SUN").dt.start_time


def aggregate_weekly_demand(
    df: pd.DataFrame,
    date_col: str = "Order Date",
    quantity_col: str = "Quantity",
    sales_col: str = "Sales",
    profit_col: str = "Profit",
    order_id_col: str = "Order ID",
    row_id_col: str = "Row ID",
    discount_col: str = "Discount"
) -> Tuple[pd.DataFrame, WeeklyValidationProfile]:
    """Aggregate transactional records into a continuous weekly demand time series.

    Returns:
        Tuple of (processed_weekly_df, validation_profile)
    """
    if df.empty:
        raise ValueError("Cannot process an empty DataFrame.")

    df_work = df.copy()
    parsed_dates = parse_and_validate_dates(df_work, date_col=date_col)
    df_work["_parsed_date"] = parsed_dates
    df_work["week_start"] = compute_week_start(parsed_dates)

    # Transaction-level sums and metrics
    agg_dict: Dict[str, Any] = {
        "quantity": (quantity_col, "sum"),
    }
    if sales_col in df_work.columns:
        agg_dict["sales"] = (sales_col, "sum")
    if profit_col in df_work.columns:
        agg_dict["profit"] = (profit_col, "sum")
    if order_id_col in df_work.columns:
        agg_dict["order_count"] = (order_id_col, "nunique")
    if row_id_col in df_work.columns:
        agg_dict["transaction_count"] = (row_id_col, "count")
    if discount_col in df_work.columns:
        agg_dict["avg_discount"] = (discount_col, "mean")

    weekly_raw = df_work.groupby("week_start").agg(**agg_dict).reset_index()

    # Determine full continuous range between min week and max week
    min_week = weekly_raw["week_start"].min()
    max_week = weekly_raw["week_start"].max()
    full_date_range = pd.date_range(start=min_week, end=max_week, freq="W-MON", name="week_start")

    # Check for missing weeks before filling
    active_weeks_set = set(weekly_raw["week_start"])
    expected_weeks_set = set(full_date_range)
    missing_weeks_set = expected_weeks_set - active_weeks_set
    missing_weeks_list = sorted([d.strftime("%Y-%m-%d") for d in missing_weeks_set])

    # Reindex to ensure strict temporal continuity
    weekly_continuous = (
        weekly_raw.set_index("week_start")
        .reindex(full_date_range)
        .rename_axis("week_start")
        .reset_index()
    )

    # Distinguish missing observation from filled metrics
    # If a week was missing in raw data, quantity is 0 demand
    weekly_continuous["quantity"] = weekly_continuous["quantity"].fillna(0).astype(int)
    if "sales" in weekly_continuous.columns:
        weekly_continuous["sales"] = weekly_continuous["sales"].fillna(0.0).round(2)
    if "profit" in weekly_continuous.columns:
        weekly_continuous["profit"] = weekly_continuous["profit"].fillna(0.0).round(2)
    if "order_count" in weekly_continuous.columns:
        weekly_continuous["order_count"] = weekly_continuous["order_count"].fillna(0).astype(int)
    if "transaction_count" in weekly_continuous.columns:
        weekly_continuous["transaction_count"] = weekly_continuous["transaction_count"].fillna(0).astype(int)
    if "avg_discount" in weekly_continuous.columns:
        weekly_continuous["avg_discount"] = weekly_continuous["avg_discount"].fillna(0.0).round(4)

    # Add temporal helper columns
    weekly_continuous["week_end"] = weekly_continuous["week_start"] + pd.Timedelta(days=6)
    weekly_continuous["year"] = weekly_continuous["week_start"].dt.year
    weekly_continuous["week_of_year"] = weekly_continuous["week_start"].dt.isocalendar().week.astype(int)

    # Partial boundary week detection:
    # Identify whether boundary weeks have full calendar day coverage from the raw dataset
    min_raw_date = parsed_dates.min()
    max_raw_date = parsed_dates.max()

    # A week is partial at the start if min raw date is after the week start date
    # A week is partial at the end if max raw date is before the week end date and is truncated
    partial_weeks_info: List[Dict[str, Any]] = []
    is_partial_list: List[bool] = []

    for idx, row in weekly_continuous.iterrows():
        w_start = row["week_start"]
        w_end = row["week_end"]
        is_partial = False
        reason = "Complete 7-day calendar week"

        if idx == 0 and min_raw_date > w_start:
            days_covered = (w_end - min_raw_date).days + 1
            is_partial = True
            reason = f"Initial boundary week: Raw data starts on {min_raw_date.strftime('%Y-%m-%d')} ({min_raw_date.day_name()}), covering only {days_covered}/7 days."
            partial_weeks_info.append({
                "week_start": w_start.strftime("%Y-%m-%d"),
                "week_end": w_end.strftime("%Y-%m-%d"),
                "boundary": "start",
                "days_covered": days_covered,
                "reason": reason
            })
        
        is_partial_list.append(is_partial)

    weekly_continuous["is_partial_week"] = is_partial_list

    # Reorder columns with week identifiers and primary target upfront
    col_order = ["week_start", "week_end", "year", "week_of_year", "is_partial_week", "quantity"]
    remaining_cols = [c for c in weekly_continuous.columns if c not in col_order]
    weekly_continuous = weekly_continuous[col_order + remaining_cols]

    # Reconciliation and Profile
    total_raw_quantity = int(df[quantity_col].sum())
    total_weekly_quantity = int(weekly_continuous["quantity"].sum())
    reconciliation_passed = (total_raw_quantity == total_weekly_quantity)

    # Zero demand weeks
    zero_demand_mask = weekly_continuous["quantity"] == 0
    zero_demand_weeks_list = [d.strftime("%Y-%m-%d") for d in weekly_continuous.loc[zero_demand_mask, "week_start"]]

    # Duplicate weeks check
    duplicate_weeks_count = int(weekly_continuous["week_start"].duplicated().sum())

    recommendation = (
        "Exclude initial partial week 2013-12-30 from model training/validation splits. "
        "Rationale: It represents only 3 calendar days (Friday Jan 3 to Sunday Jan 5, 2014) "
        "rather than a full 7-day business week, artificially distorting lag features (e.g. lag-1 demand) "
        "and rolling baseline calculations."
    )

    q_series = weekly_continuous["quantity"]
    profile = WeeklyValidationProfile(
        first_week=weekly_continuous["week_start"].min().strftime("%Y-%m-%d"),
        last_week=weekly_continuous["week_start"].max().strftime("%Y-%m-%d"),
        num_weeks=len(weekly_continuous),
        total_raw_quantity=total_raw_quantity,
        total_weekly_quantity=total_weekly_quantity,
        reconciliation_passed=reconciliation_passed,
        mean_weekly_quantity=round(float(q_series.mean()), 2),
        median_weekly_quantity=round(float(q_series.median()), 2),
        std_weekly_quantity=round(float(q_series.std()), 2),
        min_weekly_quantity=float(q_series.min()),
        max_weekly_quantity=float(q_series.max()),
        missing_weeks_count=len(missing_weeks_list),
        missing_weeks=missing_weeks_list,
        zero_demand_weeks_count=len(zero_demand_weeks_list),
        zero_demand_weeks=zero_demand_weeks_list,
        duplicate_weeks_count=duplicate_weeks_count,
        partial_weeks=partial_weeks_info,
        partial_week_recommendation=recommendation,
        additional_metrics={
            "total_sales": round(float(weekly_continuous["sales"].sum()), 2) if "sales" in weekly_continuous else None,
            "mean_weekly_sales": round(float(weekly_continuous["sales"].mean()), 2) if "sales" in weekly_continuous else None,
            "total_profit": round(float(weekly_continuous["profit"].sum()), 2) if "profit" in weekly_continuous else None,
        }
    )

    return weekly_continuous, profile


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: Path | str = "data/processed/weekly_demand.csv"
) -> Path:
    """Save processed weekly demand dataset to data/processed directory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
