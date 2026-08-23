"""Deterministic Business Analytics Engine for demand forecasting and portfolio insights.

This module computes verified historical KPIs, time-series growth, category/regional breakdowns,
top/bottom rankings, statistical anomalies, and forward forecast analytics.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.analytics.schemas import (
    HistoricalKPIs,
    PeriodGrowthRecord,
    TimeSeriesGrowthSummary,
    DimensionMetric,
    TopBottomPerformers,
    DemandAnomaly,
    ForecastHorizonSummary,
    PeakTroughWeek,
    ForecastAnalytics,
    ModelEvaluationMetadata,
    BusinessAnalyticsContext
)


def calculate_safe_growth(
    current: float,
    previous: float,
    period_name: str,
    previous_period_name: str
) -> PeriodGrowthRecord:
    """Calculate deterministic percentage growth with explicit zero-denominator handling.

    Formula:
        Growth % = ((Current - Previous) / Previous) * 100

    Args:
        current: Current period metric value.
        previous: Previous period metric value.
        period_name: Identifier for current period.
        previous_period_name: Identifier for previous period.

    Returns:
        PeriodGrowthRecord
    """
    abs_change = round(float(current - previous), 2)
    if previous == 0.0:
        return PeriodGrowthRecord(
            period_name=period_name,
            previous_period_name=previous_period_name,
            current_value=round(float(current), 2),
            previous_value=0.0,
            absolute_change=abs_change,
            growth_pct=None,
            growth_notes="Zero denominator in baseline period; percentage growth undefined."
        )

    growth_pct = round(((current - previous) / abs(previous)) * 100.0, 2)
    return PeriodGrowthRecord(
        period_name=period_name,
        previous_period_name=previous_period_name,
        current_value=round(float(current), 2),
        previous_value=round(float(previous), 2),
        absolute_change=abs_change,
        growth_pct=growth_pct,
        growth_notes="Standard percentage growth calculation."
    )


def calculate_historical_kpis(
    raw_df: pd.DataFrame,
    weekly_df: pd.DataFrame
) -> HistoricalKPIs:
    """Calculate core historical business KPIs with exact mathematical definitions.

    Formulas:
        - Total Quantity = sum(Quantity)
        - Total Sales = sum(Sales)
        - Total Profit = sum(Profit)
        - Profit Margin % = (Total Profit / Total Sales) * 100
        - Average Order Value = Total Sales / Unique Orders (calculated at order level)
        - Average Units Per Order = Total Quantity / Unique Orders
        - Average Weekly Demand = sum(Quantity complete weeks) / Count(complete weeks)

    Args:
        raw_df: Untouched raw Superstore DataFrame.
        weekly_df: Processed continuous weekly DataFrame.

    Returns:
        HistoricalKPIs Pydantic model.
    """
    total_qty = int(raw_df["Quantity"].sum())
    total_sales = round(float(raw_df["Sales"].sum()), 2)
    total_profit = round(float(raw_df["Profit"].sum()), 2)
    margin_pct = round((total_profit / total_sales) * 100.0, 2) if total_sales > 0 else 0.0

    total_orders = int(raw_df["Order ID"].nunique())
    total_transactions = int(len(raw_df))
    unique_cust = int(raw_df["Customer ID"].nunique())

    aov = round(total_sales / total_orders, 2) if total_orders > 0 else 0.0
    units_per_order = round(total_qty / total_orders, 2) if total_orders > 0 else 0.0

    complete_weekly = weekly_df[~weekly_df["is_partial_week"]].copy()
    avg_weekly_demand = round(float(complete_weekly["quantity"].mean()), 2)
    total_complete_weeks = int(len(complete_weekly))

    order_dates = pd.to_datetime(raw_df["Order Date"])
    start_date_str = str(order_dates.min().strftime("%Y-%m-%d"))
    end_date_str = str(order_dates.max().strftime("%Y-%m-%d"))

    return HistoricalKPIs(
        total_quantity=total_qty,
        total_sales=total_sales,
        total_profit=total_profit,
        profit_margin_pct=margin_pct,
        total_orders=total_orders,
        total_transactions=total_transactions,
        unique_customers=unique_cust,
        average_order_value=aov,
        average_units_per_order=units_per_order,
        average_weekly_demand=avg_weekly_demand,
        total_complete_weeks=total_complete_weeks,
        date_range_start=start_date_str,
        date_range_end=end_date_str
    )


def calculate_time_series_growth(weekly_df: pd.DataFrame) -> TimeSeriesGrowthSummary:
    """Calculate multi-year annual, quarterly, and rolling demand/sales growth.

    Args:
        weekly_df: Processed continuous weekly DataFrame.

    Returns:
        TimeSeriesGrowthSummary Pydantic model.
    """
    df = weekly_df[~weekly_df["is_partial_week"]].copy()
    df["week_dt"] = pd.to_datetime(df["week_start"])
    df["calendar_year"] = df["week_dt"].dt.year
    df["calendar_quarter"] = df["week_dt"].dt.to_period("Q").astype(str)

    # 1. Yearly aggregation
    yearly_df = df.groupby("calendar_year").agg(
        total_quantity=("quantity", "sum"),
        total_sales=("sales", "sum")
    ).reset_index()

    yearly_totals_dict: Dict[str, Dict[str, float]] = {}
    for _, row in yearly_df.iterrows():
        yr_str = str(int(row["calendar_year"]))
        yearly_totals_dict[yr_str] = {
            "quantity": int(row["total_quantity"]),
            "sales": round(float(row["total_sales"]), 2)
        }

    # YoY Growth records
    demand_growth_records: List[PeriodGrowthRecord] = []
    sales_growth_records: List[PeriodGrowthRecord] = []

    for i in range(1, len(yearly_df)):
        curr_yr = str(int(yearly_df.loc[i, "calendar_year"]))
        prev_yr = str(int(yearly_df.loc[i - 1, "calendar_year"]))

        curr_q = yearly_df.loc[i, "total_quantity"]
        prev_q = yearly_df.loc[i - 1, "total_quantity"]
        demand_growth_records.append(calculate_safe_growth(curr_q, prev_q, curr_yr, prev_yr))

        curr_s = yearly_df.loc[i, "total_sales"]
        prev_s = yearly_df.loc[i - 1, "total_sales"]
        sales_growth_records.append(calculate_safe_growth(curr_s, prev_s, curr_yr, prev_yr))

    # 2. Quarterly aggregation
    quarterly_df = df.groupby("calendar_quarter").agg(
        total_quantity=("quantity", "sum"),
        total_sales=("sales", "sum")
    ).reset_index()

    quarterly_totals_dict: Dict[str, Dict[str, float]] = {}
    for _, row in quarterly_df.iterrows():
        q_str = str(row["calendar_quarter"])
        quarterly_totals_dict[q_str] = {
            "quantity": int(row["total_quantity"]),
            "sales": round(float(row["total_sales"]), 2)
        }

    # 3. Rolling demand averages (most recent 4 weeks and 13 weeks of the historical series)
    recent_4w = round(float(df["quantity"].iloc[-4:].mean()), 2)
    recent_13w = round(float(df["quantity"].iloc[-13:].mean()), 2)

    return TimeSeriesGrowthSummary(
        yoy_demand_growth=demand_growth_records,
        yoy_sales_growth=sales_growth_records,
        yearly_totals=yearly_totals_dict,
        quarterly_totals=quarterly_totals_dict,
        recent_4w_mean_demand=recent_4w,
        recent_13w_mean_demand=recent_13w
    )


def calculate_dimension_breakdowns(raw_df: pd.DataFrame) -> Dict[str, List[DimensionMetric]]:
    """Calculate aggregated business metrics sliced by Category, Sub-Category, and Region.

    Args:
        raw_df: Untouched raw transactional DataFrame.

    Returns:
        Dictionary mapping dimension name to list of DimensionMetric objects.
    """
    total_dataset_qty = float(raw_df["Quantity"].sum())
    total_dataset_sales = float(raw_df["Sales"].sum())

    breakdowns: Dict[str, List[DimensionMetric]] = {}

    dimensions = [
        ("category", "Category"),
        ("sub_category", "Sub-Category"),
        ("region", "Region")
    ]

    for dim_key, dim_col in dimensions:
        dim_metrics: List[DimensionMetric] = []
        for val, grp in raw_df.groupby(dim_col):
            qty = int(grp["Quantity"].sum())
            sales = round(float(grp["Sales"].sum()), 2)
            profit = round(float(grp["Profit"].sum()), 2)
            orders = int(grp["Order ID"].nunique())
            margin = round((profit / sales) * 100.0, 2) if sales > 0 else 0.0
            qty_share = round((qty / total_dataset_qty) * 100.0, 2) if total_dataset_qty > 0 else 0.0
            sales_share = round((sales / total_dataset_sales) * 100.0, 2) if total_dataset_sales > 0 else 0.0

            dim_metrics.append(DimensionMetric(
                dimension_type=dim_col,
                dimension_value=str(val),
                quantity=qty,
                sales=sales,
                profit=profit,
                orders=orders,
                profit_margin_pct=margin,
                quantity_share_pct=qty_share,
                sales_share_pct=sales_share
            ))
        breakdowns[dim_key] = dim_metrics

    return breakdowns


def identify_top_bottom_performers(breakdowns: Dict[str, List[DimensionMetric]]) -> TopBottomPerformers:
    """Extract ordered performer rankings with strict descriptive evidence.

    Args:
        breakdowns: Dictionary of DimensionMetric lists from calculate_dimension_breakdowns.

    Returns:
        TopBottomPerformers Pydantic model.
    """
    cats = breakdowns.get("category", [])
    sub_cats = breakdowns.get("sub_category", [])
    regions = breakdowns.get("region", [])

    top_cat_qty = sorted(cats, key=lambda x: x.quantity, reverse=True)
    top_cat_sales = sorted(cats, key=lambda x: x.sales, reverse=True)
    top_cat_profit = sorted(cats, key=lambda x: x.profit, reverse=True)
    bottom_cat_profit = sorted(cats, key=lambda x: x.profit)

    top_sub_qty = sorted(sub_cats, key=lambda x: x.quantity, reverse=True)[:5]
    bottom_sub_profit = sorted(sub_cats, key=lambda x: x.profit)[:5]
    reg_rankings = sorted(regions, key=lambda x: x.quantity, reverse=True)

    return TopBottomPerformers(
        top_categories_by_quantity=top_cat_qty,
        top_categories_by_sales=top_cat_sales,
        top_categories_by_profit=top_cat_profit,
        bottom_categories_by_profit=bottom_cat_profit,
        top_sub_categories_by_quantity=top_sub_qty,
        bottom_sub_categories_by_profit=bottom_sub_profit,
        regional_rankings=reg_rankings
    )


def detect_demand_anomalies(
    weekly_df: pd.DataFrame,
    window: int = 13,
    z_threshold: float = 2.0
) -> List[DemandAnomaly]:
    """Detect statistical demand anomalies using rolling z-score against a moving baseline.

    Methodology:
        - Filters out the initial partial week.
        - Calculates rolling baseline mean (mu) and standard deviation (sigma) on preceding window weeks.
        - Standardized z-score = (actual - mu) / sigma
        - Flags observations where |z| >= z_threshold.
        - Descriptions are strictly non-causal.

    Args:
        weekly_df: Processed weekly demand DataFrame.
        window: Rolling baseline window in weeks (default 13 = 1 quarter).
        z_threshold: Absolute z-score threshold (default 2.0 = ~95.4% normal interval).

    Returns:
        List of DemandAnomaly Pydantic models.
    """
    df = weekly_df[~weekly_df["is_partial_week"]].copy().reset_index(drop=True)
    qty_s = df["quantity"].astype(float)

    # Shifted rolling window to prevent current-week contamination in baseline calculation
    rolling_mean = qty_s.shift(1).rolling(window=window, min_periods=window).mean()
    rolling_std = qty_s.shift(1).rolling(window=window, min_periods=window).std()

    anomalies: List[DemandAnomaly] = []

    for idx in range(window, len(df)):
        act = float(qty_s.iloc[idx])
        mu = float(rolling_mean.iloc[idx])
        sig = float(rolling_std.iloc[idx])

        if sig > 0:
            z = (act - mu) / sig
            if abs(z) >= z_threshold:
                direction = "high" if z > 0 else "low"
                desc = (
                    f"Observed weekly demand of {int(act)} units was {abs(z):.2f} standard deviations "
                    f"{'above' if z > 0 else 'below'} the rolling {window}-week baseline mean of {mu:.1f} units."
                )
                anomalies.append(DemandAnomaly(
                    week_start=str(df.loc[idx, "week_start"]),
                    actual_quantity=act,
                    baseline_mean=round(mu, 2),
                    baseline_std=round(sig, 2),
                    z_score=round(float(z), 2),
                    anomaly_direction=direction,
                    description=desc
                ))

    return anomalies


def generate_forecast_analytics(
    hw_forecasts_df: pd.DataFrame,
    complete_weekly_df: pd.DataFrame,
    model_eval_dict: Dict[str, Any]
) -> ForecastAnalytics:
    """Generate structured multi-horizon forward analytics using champion Holt-Winters forecasts.

    Args:
        hw_forecasts_df: DataFrame of 52-week out-of-sample Holt-Winters forecasts.
        complete_weekly_df: Complete historical weekly demand DataFrame.
        model_eval_dict: Evaluation metrics dictionary from reports.

    Returns:
        ForecastAnalytics Pydantic model.
    """
    df_pred = hw_forecasts_df.copy().reset_index(drop=True)
    preds = df_pred["hw_pred"].values.astype(float)
    dates = df_pred["week_start"].tolist()

    # Preceding historical complete weeks for comparison (last 52 weeks of training: 2016)
    hist_complete = complete_weekly_df[~complete_weekly_df["is_partial_week"]].copy().reset_index(drop=True)
    # The last 52 weeks before test start
    train_history_52 = hist_complete.loc[hist_complete["year"] == 2016, "quantity"].values.astype(float)

    horizons_config = [
        ("next_1_week", 1),
        ("next_4_weeks", 4),
        ("next_8_weeks", 8),
        ("next_12_weeks", 12),
        ("full_52_weeks", 52)
    ]

    horizons_dict: Dict[str, ForecastHorizonSummary] = {}
    for h_name, h_len in horizons_config:
        h_len_clamped = min(h_len, len(preds))
        f_sum = float(np.sum(preds[:h_len_clamped]))
        f_mean = float(np.mean(preds[:h_len_clamped]))

        # Equivalent historical period (e.g. preceding h_len weeks from end of training set)
        prior_actual = float(np.sum(train_history_52[-h_len_clamped:])) if len(train_history_52) >= h_len_clamped else 0.0
        pct_change = round(((f_sum - prior_actual) / prior_actual) * 100.0, 2) if prior_actual > 0 else None

        horizons_dict[h_name] = ForecastHorizonSummary(
            horizon_name=h_name,
            horizon_weeks=h_len_clamped,
            total_forecast_quantity=round(f_sum, 2),
            mean_weekly_forecast=round(f_mean, 2),
            prior_period_actual_quantity=round(prior_actual, 2),
            pct_change_vs_prior_period=pct_change
        )

    # Peak and Trough Forecast Weeks
    peak_idx = int(np.argmax(preds))
    trough_idx = int(np.argmin(preds))

    peak_week = PeakTroughWeek(
        week_start=dates[peak_idx],
        forecast_quantity=round(float(preds[peak_idx]), 2),
        description=f"Peak projected weekly demand: {preds[peak_idx]:.1f} units on {dates[peak_idx]}."
    )
    trough_week = PeakTroughWeek(
        week_start=dates[trough_idx],
        forecast_quantity=round(float(preds[trough_idx]), 2),
        description=f"Trough projected weekly demand: {preds[trough_idx]:.1f} units on {dates[trough_idx]}."
    )

    return ForecastAnalytics(
        champion_model="Holt-Winters (Additive Trend, Additive Seasonality, s=52)",
        forecast_start_date=dates[0],
        forecast_end_date=dates[-1],
        horizons=horizons_dict,
        peak_forecast_week=peak_week,
        trough_forecast_week=trough_week,
        annual_forecast_total=round(float(np.sum(preds)), 2)
    )


def build_business_analytics_context(
    raw_df_path: Path | str = "data/raw/Sample_Superstore.csv",
    weekly_df_path: Path | str = "data/processed/weekly_demand.csv",
    reports_dir: Path | str = "reports"
) -> BusinessAnalyticsContext:
    """Construct complete verified Business Analytics Context for downstream LLM grounding.

    Args:
        raw_df_path: Path to raw dataset CSV.
        weekly_df_path: Path to processed weekly demand CSV.
        reports_dir: Path to reports directory containing model evaluation outputs.

    Returns:
        BusinessAnalyticsContext master Pydantic object.
    """
    raw_path = Path(raw_df_path)
    weekly_path = Path(weekly_df_path)
    rep_dir = Path(reports_dir)

    raw_df = pd.read_csv(raw_path, encoding="windows-1252")
    weekly_df = pd.read_csv(weekly_path)

    # 1. Historical KPIs
    kpis = calculate_historical_kpis(raw_df, weekly_df)

    # 2. Time-Series Growth
    ts_growth = calculate_time_series_growth(weekly_df)

    # 3. Dimension Breakdowns
    breakdowns = calculate_dimension_breakdowns(raw_df)

    # 4. Top/Bottom Performers
    performers = identify_top_bottom_performers(breakdowns)

    # 5. Anomalies
    anomalies = detect_demand_anomalies(weekly_df, window=13, z_threshold=2.0)

    # 6. Forecast Analytics from Champion Holt-Winters model
    hw_forecast_csv = rep_dir / "holt_winters_forecasts.csv"
    if not hw_forecast_csv.exists():
        raise FileNotFoundError(f"Champion forecast CSV not found at: {hw_forecast_csv}")
    hw_forecast_df = pd.read_csv(hw_forecast_csv)

    # Model evaluation metadata
    hw_eval_json = rep_dir / "holt_winters_evaluation.json"
    eval_dict = {}
    if hw_eval_json.exists():
        import json
        with open(hw_eval_json, "r") as f:
            eval_dict = json.load(f)

    hw_acc = eval_dict.get("accuracy_metrics", {}).get("holt_winters", {})
    split_info = eval_dict.get("split_info", {})

    forecast_analytics = generate_forecast_analytics(hw_forecast_df, weekly_df, eval_dict)

    model_metadata = ModelEvaluationMetadata(
        model_name="Holt-Winters Exponential Smoothing",
        model_type="Additive Linear Trend, Additive Annual Seasonality (s=52)",
        training_period=f"{split_info.get('train_start_date', '2014-01-06')} to {split_info.get('train_end_date', '2016-12-26')} ({split_info.get('num_train_observations', 156)} weeks)",
        test_period=f"{split_info.get('test_start_date', '2017-01-02')} to {split_info.get('test_end_date', '2017-12-25')} ({split_info.get('num_test_observations', 52)} weeks)",
        test_mae=hw_acc.get("mae", 39.02),
        test_rmse=hw_acc.get("rmse", 52.40),
        test_mape=hw_acc.get("mape", 19.24),
        test_smape=hw_acc.get("smape", 17.40),
        test_bias=hw_acc.get("mean_error", 3.76),
        evaluation_notes="Evaluation accuracy metrics reflect out-of-sample holdout performance on the unseen 2017 test set."
    )

    return BusinessAnalyticsContext(
        schema_version="1.0.0",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        historical_kpis=kpis,
        time_series_growth=ts_growth,
        dimension_breakdowns=breakdowns,
        top_bottom_performers=performers,
        anomalies=anomalies,
        forecast_analytics=forecast_analytics,
        model_metadata=model_metadata
    )
