"""Dataset inspection and exploratory profile module for Demand Forecasting.

This module provides functions to inspect raw tabular data, verify schema,
check nulls and duplicates, compute summary statistics, analyze date
granularity, and evaluate preliminary temporal aggregations (daily, weekly, monthly).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class DatasetProfile:
    """Structured container for dataset profile and quality inspection results."""
    file_path: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows_count: int
    unique_counts: Dict[str, int]
    category_breakdown: Dict[str, int]
    sub_category_breakdown: Dict[str, int]
    region_breakdown: Dict[str, int]
    date_range: Dict[str, Any]
    numerical_summary: Dict[str, Dict[str, float]]
    daily_summary: Dict[str, Any]
    weekly_summary: Dict[str, Any]
    monthly_summary: Dict[str, Any]
    data_quality_notes: List[str] = field(default_factory=list)


def load_raw_dataset(
    file_path: Path | str,
    encoding: str = "windows-1252"
) -> pd.DataFrame:
    """Load raw dataset with designated encoding without mutating raw files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    df = pd.read_csv(path, encoding=encoding)
    return df


def inspect_dataset(
    df: pd.DataFrame,
    file_path: str = "data/raw/Sample_Superstore.csv",
    date_col: str = "Order Date",
    quantity_col: str = "Quantity",
    sales_col: str = "Sales"
) -> DatasetProfile:
    """Perform a comprehensive inspection of the dataset and temporal aggregation profiles."""
    row_count, col_count = df.shape
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = ((df.isnull().sum() / row_count) * 100).round(4).to_dict()

    # Duplicate rows (entire row duplicate check)
    duplicate_rows_count = int(df.duplicated().sum())

    # Unique entity counts
    unique_counts = {
        "unique_orders": int(df["Order ID"].nunique()) if "Order ID" in df else 0,
        "unique_customers": int(df["Customer ID"].nunique()) if "Customer ID" in df else 0,
        "unique_products": int(df["Product ID"].nunique()) if "Product ID" in df else 0,
        "unique_product_names": int(df["Product Name"].nunique()) if "Product Name" in df else 0,
        "unique_categories": int(df["Category"].nunique()) if "Category" in df else 0,
        "unique_sub_categories": int(df["Sub-Category"].nunique()) if "Sub-Category" in df else 0,
        "unique_regions": int(df["Region"].nunique()) if "Region" in df else 0,
        "unique_states": int(df["State"].nunique()) if "State" in df else 0,
        "unique_cities": int(df["City"].nunique()) if "City" in df else 0,
        "unique_segments": int(df["Segment"].nunique()) if "Segment" in df else 0,
        "unique_ship_modes": int(df["Ship Mode"].nunique()) if "Ship Mode" in df else 0,
    }

    # Value distributions for categories/regions
    category_breakdown = df["Category"].value_counts().to_dict() if "Category" in df else {}
    sub_category_breakdown = df["Sub-Category"].value_counts().to_dict() if "Sub-Category" in df else {}
    region_breakdown = df["Region"].value_counts().to_dict() if "Region" in df else {}

    # Numerical statistics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_summary = {}
    for col in num_cols:
        desc = df[col].describe()
        numerical_summary[col] = {
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": float(desc["min"]),
            "25%": float(desc["25%"]),
            "50%": float(desc["50%"]),
            "75%": float(desc["75%"]),
            "max": float(desc["max"]),
            "skew": float(df[col].skew()) if len(df[col]) > 2 else 0.0,
        }

    # Date range and parsing analysis
    df_temp = df.copy()
    parsed_dates = pd.to_datetime(df_temp[date_col], format="%m/%d/%Y", errors="coerce")
    if parsed_dates.isnull().any():
        # Fallback to general parsing if mixed format
        parsed_dates = pd.to_datetime(df_temp[date_col], errors="coerce")

    min_date = parsed_dates.min()
    max_date = parsed_dates.max()
    total_days_span = (max_date - min_date).days + 1
    unique_active_dates = parsed_dates.nunique()

    ship_dates = pd.to_datetime(df_temp["Ship Date"], errors="coerce") if "Ship Date" in df_temp else None

    date_range = {
        "order_date_min": min_date.strftime("%Y-%m-%d"),
        "order_date_max": max_date.strftime("%Y-%m-%d"),
        "ship_date_min": ship_dates.min().strftime("%Y-%m-%d") if ship_dates is not None else None,
        "ship_date_max": ship_dates.max().strftime("%Y-%m-%d") if ship_dates is not None else None,
        "total_calendar_days": total_days_span,
        "active_order_days": unique_active_dates,
        "days_without_orders": total_days_span - unique_active_dates,
        "unparsed_date_count": int(parsed_dates.isnull().sum())
    }

    # Attach parsed date for aggregation analysis
    df_temp["_parsed_date"] = parsed_dates

    # Daily aggregation summary
    daily_agg = df_temp.groupby(pd.Grouper(key="_parsed_date", freq="D")).agg(
        total_quantity=(quantity_col, "sum"),
        total_sales=(sales_col, "sum"),
        order_count=("Order ID", "nunique"),
        transaction_count=("Row ID", "count")
    )
    daily_summary = _summarize_temporal_grain(daily_agg, "Daily (D)")

    # Weekly aggregation summary (Monday-starting weeks)
    weekly_agg = df_temp.groupby(pd.Grouper(key="_parsed_date", freq="W-MON")).agg(
        total_quantity=(quantity_col, "sum"),
        total_sales=(sales_col, "sum"),
        order_count=("Order ID", "nunique"),
        transaction_count=("Row ID", "count")
    )
    weekly_summary = _summarize_temporal_grain(weekly_agg, "Weekly (W-MON)")

    # Monthly aggregation summary (Month-end)
    monthly_agg = df_temp.groupby(pd.Grouper(key="_parsed_date", freq="ME")).agg(
        total_quantity=(quantity_col, "sum"),
        total_sales=(sales_col, "sum"),
        order_count=("Order ID", "nunique"),
        transaction_count=("Row ID", "count")
    )
    monthly_summary = _summarize_temporal_grain(monthly_agg, "Monthly (ME)")

    # Data quality findings
    notes: List[str] = []
    if date_range["days_without_orders"] > 0:
        notes.append(
            f"Transactional granularity contains {date_range['days_without_orders']} days without orders out of {total_days_span} calendar days (~{(date_range['days_without_orders']/total_days_span)*100:.1f}% zero-order days at pure daily grain)."
        )
    if duplicate_rows_count == 0:
        notes.append("No complete duplicate rows found across all 21 columns.")
    else:
        notes.append(f"Found {duplicate_rows_count} duplicate rows.")

    # Check for negative/zero values in Quantity / Sales
    zero_or_neg_qty = int((df[quantity_col] <= 0).sum())
    if zero_or_neg_qty > 0:
        notes.append(f"Found {zero_or_neg_qty} rows with Quantity <= 0.")
    else:
        notes.append("All Quantity values are strictly positive (> 0).")

    # Check for negative profits (normal in retail due to high discounts)
    neg_profits = int((df["Profit"] < 0).sum())
    notes.append(f"Found {neg_profits} transactions with negative profit (due to high discounts / margins).")

    # Postal code check
    if "Postal Code" in df:
        null_postal = int(df["Postal Code"].isnull().sum())
        if null_postal > 0:
            notes.append(f"Postal Code has {null_postal} missing values.")
        else:
            notes.append("Postal Code has 0 missing values (stored as integer / float in raw CSV).")

    return DatasetProfile(
        file_path=file_path,
        row_count=row_count,
        column_count=col_count,
        columns=columns,
        dtypes=dtypes,
        missing_values=missing_values,
        missing_percentage=missing_percentage,
        duplicate_rows_count=duplicate_rows_count,
        unique_counts=unique_counts,
        category_breakdown=category_breakdown,
        sub_category_breakdown=sub_category_breakdown,
        region_breakdown=region_breakdown,
        date_range=date_range,
        numerical_summary=numerical_summary,
        daily_summary=daily_summary,
        weekly_summary=weekly_summary,
        monthly_summary=monthly_summary,
        data_quality_notes=notes
    )


def _summarize_temporal_grain(agg_df: pd.DataFrame, grain_name: str) -> Dict[str, Any]:
    """Calculate summary statistics for a given temporal aggregation grain."""
    qty = agg_df["total_quantity"].fillna(0)
    sales = agg_df["total_sales"].fillna(0)
    total_periods = len(agg_df)
    zero_demand_periods = int((qty == 0).sum())
    mean_qty = float(qty.mean())
    std_qty = float(qty.std()) if total_periods > 1 else 0.0
    cv_qty = (std_qty / mean_qty) if mean_qty > 0 else 0.0

    return {
        "grain": grain_name,
        "total_periods": total_periods,
        "zero_demand_periods": zero_demand_periods,
        "zero_demand_ratio": round(zero_demand_periods / total_periods, 4) if total_periods > 0 else 0.0,
        "demand_quantity": {
            "total": float(qty.sum()),
            "mean": round(mean_qty, 2),
            "std": round(std_qty, 2),
            "min": float(qty.min()),
            "median": float(qty.median()),
            "max": float(qty.max()),
            "coefficient_of_variation": round(cv_qty, 4)
        },
        "sales_value": {
            "total": round(float(sales.sum()), 2),
            "mean": round(float(sales.mean()), 2),
            "std": round(float(sales.std()), 2),
            "min": round(float(sales.min()), 2),
            "median": round(float(sales.median()), 2),
            "max": round(float(sales.max()), 2),
        }
    }
