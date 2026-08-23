"""Feature engineering module for weekly demand forecasting.

Forecasting Scenario & Origin Assumption:
------------------------------------------
"At the end of completed week t, the system forecasts demand for week t+1."

Under this one-step-ahead / rolling forecasting scenario, lagged business features
(e.g., lag_1_sales, lag_1_profit, lag_1_order_count, lag_1_transaction_count,
lag_1_unique_customers) are fully observed and available because week t has already completed.

Important Horizon Limitation:
------------------------------
These lagged business features would NOT be available for a long-horizon 52-week-ahead
direct/static forecast unless separately forecasted or supplied as known exogenous variables.

Training Sample Size Consideration:
------------------------------------
Holt-Winters and SARIMA utilize the complete 2014-2016 training history (156 complete weeks).
In contrast, the ML feature matrix requires 52 historical observations to populate the lag_52
features, leaving 105 effective training observations (2014-12-29 to 2016-12-26).
This difference in effective training sample size is an inherent structural distinction
between univariate state-space smoothing and autoregressive tabular feature representations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_weekly_business_aggregations(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transactional data into weekly business breakdowns.

    Args:
        raw_df: Untouched raw Superstore DataFrame.

    Returns:
        DataFrame with Monday-aligned week_start and weekly category/regional/customer metrics.
    """
    df = raw_df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["week_start"] = (df["Order Date"] - pd.to_timedelta(df["Order Date"].dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")

    rows = []
    for week_str, grp in df.groupby("week_start"):
        rows.append({
            "week_start": week_str,
            "unique_customers": grp["Customer ID"].nunique(),
            "qty_furniture": float(grp.loc[grp["Category"] == "Furniture", "Quantity"].sum()),
            "qty_office_supplies": float(grp.loc[grp["Category"] == "Office Supplies", "Quantity"].sum()),
            "qty_technology": float(grp.loc[grp["Category"] == "Technology", "Quantity"].sum()),
            "qty_central": float(grp.loc[grp["Region"] == "Central", "Quantity"].sum()),
            "qty_east": float(grp.loc[grp["Region"] == "East", "Quantity"].sum()),
            "qty_south": float(grp.loc[grp["Region"] == "South", "Quantity"].sum()),
            "qty_west": float(grp.loc[grp["Region"] == "West", "Quantity"].sum()),
        })

    return pd.DataFrame(rows)


def build_weekly_feature_dataset(
    weekly_demand_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct complete weekly feature matrix with temporal, lag, rolling, and business features.

    Forecasting Origin Assumption:
    At the end of week t, forecast target_quantity for week t+1.
    All rolling and business features strictly use lagged historical information (t-1, t-52)
    to guarantee zero lookahead leakage.

    Args:
        weekly_demand_df: Processed continuous weekly demand DataFrame.
        raw_df: Optional raw DataFrame to extract category and customer counts.

    Returns:
        Tuple of (engineered_features_df, feature_column_names)
    """
    df = weekly_demand_df.copy()

    # Merge additional business breakdowns if raw_df is provided
    if raw_df is not None:
        biz_breakdown = extract_weekly_business_aggregations(raw_df)
        df = pd.merge(df, biz_breakdown, on="week_start", how="left").fillna(0.0)

    df["week_dt"] = pd.to_datetime(df["week_start"])

    # 1. Target Column
    df["target_quantity"] = df["quantity"].astype(float)

    # 2. Temporal Features (Known calendar variables at forecast time)
    df["trend_index"] = np.arange(len(df), dtype=float)
    df["month"] = df["week_dt"].dt.month.astype(float)
    df["quarter"] = df["week_dt"].dt.quarter.astype(float)
    df["week_of_year_feat"] = df["week_dt"].dt.isocalendar().week.astype(float)
    # Continuous cyclical sine/cosine transforms
    df["sin_woy"] = np.sin(2.0 * np.pi * df["week_of_year_feat"] / 52.1775)
    df["cos_woy"] = np.cos(2.0 * np.pi * df["week_of_year_feat"] / 52.1775)

    # 3. Target Autoregressive Lag Features (strictly shifted backward)
    target_s = df["target_quantity"]
    for lag_k in [1, 2, 4, 8, 13, 26, 52]:
        df[f"lag_{lag_k}"] = target_s.shift(lag_k)

    # 4. Rolling Statistical Features (Strictly Historical: computed on target.shift(1))
    # E.g., rolling_mean_4 for week t uses mean(y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4})
    shifted_target = target_s.shift(1)
    df["rolling_mean_4"] = shifted_target.rolling(window=4, min_periods=4).mean()
    df["rolling_mean_8"] = shifted_target.rolling(window=8, min_periods=8).mean()
    df["rolling_mean_13"] = shifted_target.rolling(window=13, min_periods=13).mean()
    df["rolling_std_4"] = shifted_target.rolling(window=4, min_periods=4).std()
    df["rolling_std_13"] = shifted_target.rolling(window=13, min_periods=13).std()

    # 5. Lagged Business Features (Strictly Historical: lag-1 and lag-52)
    # Available base business columns:
    biz_cols = ["sales", "profit", "avg_discount", "order_count", "transaction_count"]
    if "unique_customers" in df.columns:
        biz_cols.extend(["unique_customers", "qty_furniture", "qty_office_supplies", "qty_technology",
                         "qty_central", "qty_east", "qty_south", "qty_west"])

    for b_col in biz_cols:
        df[f"lag_1_{b_col}"] = df[b_col].astype(float).shift(1)
        # 52-week seasonal lag for high-signal business variables
        if b_col in ["sales", "order_count", "qty_furniture", "qty_office_supplies", "qty_technology"]:
            df[f"lag_52_{b_col}"] = df[b_col].astype(float).shift(52)

    # Exclude all unlagged contemporaneous / raw target columns from the feature list
    unlagged_raw_cols = set([
        "week_start", "week_end", "week_dt", "quantity", "year", "week_of_year", "is_partial_week", "target_quantity",
        "sales", "profit", "avg_discount", "order_count", "transaction_count", "unique_customers",
        "qty_furniture", "qty_office_supplies", "qty_technology",
        "qty_central", "qty_east", "qty_south", "qty_west"
    ])

    feature_cols = [c for c in df.columns if c not in unlagged_raw_cols]

    return df, feature_cols


def generate_feature_leakage_audit() -> pd.DataFrame:
    """Generate exhaustive documented Feature Leakage Audit table with forecast-origin context.

    Returns:
        DataFrame cataloging every feature, its source, temporal classification, availability status, and rationale.
    """
    audit_records = [
        # Temporal features
        {"Feature": "trend_index", "Category": "Temporal", "Shift/Window": "Calendar", "Temporal Class": "Known at forecast time", "Status": "ALLOWED", "Rationale": "Linear time index (0, 1, ...) deterministic and known in advance."},
        {"Feature": "month", "Category": "Temporal", "Shift/Window": "Calendar", "Temporal Class": "Known at forecast time", "Status": "ALLOWED", "Rationale": "Calendar month (1-12) known in advance."},
        {"Feature": "quarter", "Category": "Temporal", "Shift/Window": "Calendar", "Temporal Class": "Known at forecast time", "Status": "ALLOWED", "Rationale": "Calendar quarter (1-4) known in advance."},
        {"Feature": "week_of_year_feat", "Category": "Temporal", "Shift/Window": "Calendar", "Temporal Class": "Known at forecast time", "Status": "ALLOWED", "Rationale": "ISO week number (1-53) known in advance."},
        {"Feature": "sin_woy", "Category": "Temporal", "Shift/Window": "Calendar", "Temporal Class": "Known at forecast time", "Status": "ALLOWED", "Rationale": "Sinusoidal annual cyclical encoding known in advance."},
        {"Feature": "cos_woy", "Category": "Temporal", "Shift/Window": "Calendar", "Temporal Class": "Known at forecast time", "Status": "ALLOWED", "Rationale": "Cosinesoidal annual cyclical encoding known in advance."},

        # Target Lags
        {"Feature": "lag_1", "Category": "Target Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week actual demand, fully observed at prediction time (end of week t)."},
        {"Feature": "lag_2", "Category": "Target Lag", "Shift/Window": "Shift -2", "Temporal Class": "Historical (t-2)", "Status": "ALLOWED", "Rationale": "Observed 2 weeks prior."},
        {"Feature": "lag_4", "Category": "Target Lag", "Shift/Window": "Shift -4", "Temporal Class": "Historical (t-4)", "Status": "ALLOWED", "Rationale": "Observed 4 weeks prior."},
        {"Feature": "lag_8", "Category": "Target Lag", "Shift/Window": "Shift -8", "Temporal Class": "Historical (t-8)", "Status": "ALLOWED", "Rationale": "Observed 8 weeks prior."},
        {"Feature": "lag_13", "Category": "Target Lag", "Shift/Window": "Shift -13", "Temporal Class": "Historical (t-13)", "Status": "ALLOWED", "Rationale": "Observed 13 weeks (1 quarter) prior."},
        {"Feature": "lag_26", "Category": "Target Lag", "Shift/Window": "Shift -26", "Temporal Class": "Historical (t-26)", "Status": "ALLOWED", "Rationale": "Observed 26 weeks (half year) prior."},
        {"Feature": "lag_52", "Category": "Target Lag", "Shift/Window": "Shift -52", "Temporal Class": "Historical (t-52)", "Status": "ALLOWED", "Rationale": "Observed 52 weeks (1 full year) prior for annual seasonal matching."},

        # Rolling Summary Features
        {"Feature": "rolling_mean_4", "Category": "Rolling Target", "Shift/Window": "Shift -1, Win 4", "Temporal Class": "Historical (t-4 to t-1)", "Status": "ALLOWED", "Rationale": "4-week moving average computed strictly on lagged values y_{t-1}..y_{t-4}; excludes y_t."},
        {"Feature": "rolling_mean_8", "Category": "Rolling Target", "Shift/Window": "Shift -1, Win 8", "Temporal Class": "Historical (t-8 to t-1)", "Status": "ALLOWED", "Rationale": "8-week moving average computed strictly on lagged values; excludes y_t."},
        {"Feature": "rolling_mean_13", "Category": "Rolling Target", "Shift/Window": "Shift -1, Win 13", "Temporal Class": "Historical (t-13 to t-1)", "Status": "ALLOWED", "Rationale": "13-week moving average computed strictly on lagged values; excludes y_t."},
        {"Feature": "rolling_std_4", "Category": "Rolling Target", "Shift/Window": "Shift -1, Win 4", "Temporal Class": "Historical (t-4 to t-1)", "Status": "ALLOWED", "Rationale": "4-week demand standard deviation on lagged values; excludes y_t."},
        {"Feature": "rolling_std_13", "Category": "Rolling Target", "Shift/Window": "Shift -1, Win 13", "Temporal Class": "Historical (t-13 to t-1)", "Status": "ALLOWED", "Rationale": "13-week demand volatility on lagged values; excludes y_t."},

        # Lagged Business Features
        {"Feature": "lag_1_sales", "Category": "Business Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week total sales revenue observed at end of completed week t. (Note: unavailable for 52-week-ahead forecast without exogenous forecasts)."},
        {"Feature": "lag_52_sales", "Category": "Business Lag", "Shift/Window": "Shift -52", "Temporal Class": "Historical (t-52)", "Status": "ALLOWED", "Rationale": "Prior year same-week sales revenue observed at prediction time."},
        {"Feature": "lag_1_profit", "Category": "Business Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week total profit observed at end of completed week t."},
        {"Feature": "lag_1_avg_discount", "Category": "Business Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week average discount observed at end of completed week t."},
        {"Feature": "lag_1_order_count", "Category": "Business Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week unique order count observed at end of completed week t."},
        {"Feature": "lag_52_order_count", "Category": "Business Lag", "Shift/Window": "Shift -52", "Temporal Class": "Historical (t-52)", "Status": "ALLOWED", "Rationale": "Prior year same-week order count observed at prediction time."},
        {"Feature": "lag_1_transaction_count", "Category": "Business Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week transaction line item volume observed at end of completed week t."},
        {"Feature": "lag_1_unique_customers", "Category": "Business Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week unique customer count observed at end of completed week t."},
        {"Feature": "lag_1_qty_furniture", "Category": "Category Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week Furniture demand volume."},
        {"Feature": "lag_52_qty_furniture", "Category": "Category Lag", "Shift/Window": "Shift -52", "Temporal Class": "Historical (t-52)", "Status": "ALLOWED", "Rationale": "Prior year same-week Furniture demand volume."},
        {"Feature": "lag_1_qty_office_supplies", "Category": "Category Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week Office Supplies volume."},
        {"Feature": "lag_52_qty_office_supplies", "Category": "Category Lag", "Shift/Window": "Shift -52", "Temporal Class": "Historical (t-52)", "Status": "ALLOWED", "Rationale": "Prior year same-week Office Supplies volume."},
        {"Feature": "lag_1_qty_technology", "Category": "Category Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week Technology volume."},
        {"Feature": "lag_52_qty_technology", "Category": "Category Lag", "Shift/Window": "Shift -52", "Temporal Class": "Historical (t-52)", "Status": "ALLOWED", "Rationale": "Prior year same-week Technology volume."},
        {"Feature": "lag_1_qty_central", "Category": "Regional Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week Central region volume."},
        {"Feature": "lag_1_qty_east", "Category": "Regional Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week East region volume."},
        {"Feature": "lag_1_qty_south", "Category": "Regional Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week South region volume."},
        {"Feature": "lag_1_qty_west", "Category": "Regional Lag", "Shift/Window": "Shift -1", "Temporal Class": "Historical (t-1)", "Status": "ALLOWED", "Rationale": "Prior week West region volume."},

        # PROHIBITED Contemporaneous / Future Features
        {"Feature": "current_week_sales (sales)", "Category": "Business Contemporaneous", "Shift/Window": "Shift 0", "Temporal Class": "Contemporaneous (week t)", "Status": "PROHIBITED (EXCLUDED)", "Rationale": "Total sales in week t is not known until week t ends. Using it to forecast week t demand is future leakage."},
        {"Feature": "current_week_profit (profit)", "Category": "Business Contemporaneous", "Shift/Window": "Shift 0", "Temporal Class": "Contemporaneous (week t)", "Status": "PROHIBITED (EXCLUDED)", "Rationale": "Profit in week t is unknown at forecast creation."},
        {"Feature": "current_week_discount (avg_discount)", "Category": "Business Contemporaneous", "Shift/Window": "Shift 0", "Temporal Class": "Contemporaneous (week t)", "Status": "PROHIBITED (EXCLUDED)", "Rationale": "Average realized discount in week t is unknown at forecast time."},
        {"Feature": "current_week_orders (order_count)", "Category": "Business Contemporaneous", "Shift/Window": "Shift 0", "Temporal Class": "Contemporaneous (week t)", "Status": "PROHIBITED (EXCLUDED)", "Rationale": "Total order count in week t is unknown at forecast time."},
        {"Feature": "current_week_customers (unique_customers)", "Category": "Business Contemporaneous", "Shift/Window": "Shift 0", "Temporal Class": "Contemporaneous (week t)", "Status": "PROHIBITED (EXCLUDED)", "Rationale": "Customer count in week t is unknown at forecast time."},
        {"Feature": "current_week_category_qtys", "Category": "Category Contemporaneous", "Shift/Window": "Shift 0", "Temporal Class": "Contemporaneous (week t)", "Status": "PROHIBITED (EXCLUDED)", "Rationale": "Category quantities sum directly to target quantity. Using contemporaneous category quantities is trivial label leakage."},
    ]

    return pd.DataFrame(audit_records)
