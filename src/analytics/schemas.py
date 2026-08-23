"""Pydantic schemas for the deterministic Business Analytics Engine."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HistoricalKPIs(BaseModel):
    """Core historical business key performance indicators."""
    total_quantity: int = Field(description="Total units sold across all transactions.")
    total_sales: float = Field(description="Total monetary sales revenue ($).")
    total_profit: float = Field(description="Total net profit ($).")
    profit_margin_pct: float = Field(description="Profit margin percentage: (total_profit / total_sales) * 100.")
    total_orders: int = Field(description="Total unique order count (Order ID).")
    total_transactions: int = Field(description="Total line item transactions (Row ID count).")
    unique_customers: int = Field(description="Count of distinct customers.")
    average_order_value: float = Field(description="Average sales per order: total_sales / total_orders ($).")
    average_units_per_order: float = Field(description="Average units sold per order: total_quantity / total_orders.")
    average_weekly_demand: float = Field(description="Average weekly demand units across complete weeks.")
    total_complete_weeks: int = Field(description="Count of complete weekly observations.")
    date_range_start: str = Field(description="Start date of historical dataset.")
    date_range_end: str = Field(description="End date of historical dataset.")


class PeriodGrowthRecord(BaseModel):
    """Deterministic period-over-period growth record."""
    period_name: str = Field(description="Name of the current period.")
    previous_period_name: str = Field(description="Name of the comparison period.")
    current_value: float = Field(description="Current period metric value.")
    previous_value: float = Field(description="Previous period metric value.")
    absolute_change: float = Field(description="current_value - previous_value.")
    growth_pct: Optional[float] = Field(None, description="Percentage growth: ((current - previous) / previous) * 100.")
    growth_notes: str = Field(default="", description="Calculation notes or zero-denominator status.")


class TimeSeriesGrowthSummary(BaseModel):
    """Summary of demand and sales growth across multi-year and intra-year horizons."""
    yoy_demand_growth: List[PeriodGrowthRecord] = Field(description="Year-over-year quantity growth records.")
    yoy_sales_growth: List[PeriodGrowthRecord] = Field(description="Year-over-year monetary sales growth records.")
    yearly_totals: Dict[str, Dict[str, float]] = Field(description="Annual quantity and sales totals.")
    quarterly_totals: Dict[str, Dict[str, float]] = Field(description="Quarterly quantity and sales totals.")
    recent_4w_mean_demand: float = Field(description="Mean weekly demand across the most recent 4 historical weeks.")
    recent_13w_mean_demand: float = Field(description="Mean weekly demand across the most recent 13 historical weeks (1 quarter).")


class DimensionMetric(BaseModel):
    """Aggregated business metrics for a specific dimension slice."""
    dimension_type: str = Field(description="Category, Sub-Category, or Region.")
    dimension_value: str = Field(description="Name of the slice (e.g. Technology, West, Chairs).")
    quantity: int = Field(description="Total units sold in this slice.")
    sales: float = Field(description="Total monetary sales in this slice ($).")
    profit: float = Field(description="Total profit in this slice ($).")
    orders: int = Field(description="Count of unique orders involving this slice.")
    profit_margin_pct: float = Field(description="Profit margin percentage in this slice.")
    quantity_share_pct: float = Field(description="Percentage contribution to overall dataset quantity.")
    sales_share_pct: float = Field(description="Percentage contribution to overall dataset sales.")


class TopBottomPerformers(BaseModel):
    """Ranked ranking lists for business dimensions based on empirical evidence."""
    top_categories_by_quantity: List[DimensionMetric] = Field(description="Top categories ranked by volume.")
    top_categories_by_sales: List[DimensionMetric] = Field(description="Top categories ranked by sales revenue.")
    top_categories_by_profit: List[DimensionMetric] = Field(description="Top categories ranked by net profit.")
    bottom_categories_by_profit: List[DimensionMetric] = Field(description="Categories ranked by lowest net profit.")
    top_sub_categories_by_quantity: List[DimensionMetric] = Field(description="Top 5 sub-categories by volume.")
    bottom_sub_categories_by_profit: List[DimensionMetric] = Field(description="Lowest 5 sub-categories by profit.")
    regional_rankings: List[DimensionMetric] = Field(description="Regions ranked by total quantity.")


class DemandAnomaly(BaseModel):
    """Statistically identified unusual demand observation."""
    week_start: str = Field(description="Monday week start date of the anomaly.")
    actual_quantity: float = Field(description="Observed weekly demand units.")
    baseline_mean: float = Field(description="Rolling baseline mean demand.")
    baseline_std: float = Field(description="Rolling baseline demand standard deviation.")
    z_score: float = Field(description="Standardized deviation score: (actual - mean) / std.")
    anomaly_direction: str = Field(description="'high' or 'low' demand anomaly.")
    description: str = Field(description="Objective, non-causal statistical description.")


class PeakTroughWeek(BaseModel):
    """Date and value of maximum or minimum forecast week."""
    week_start: str = Field(description="Monday week start date.")
    forecast_quantity: float = Field(description="Projected demand units.")
    description: str = Field(description="Descriptive summary.")


class ForecastHorizonSummary(BaseModel):
    """Aggregated forward forecast metrics for a specific planning horizon."""
    horizon_name: str = Field(description="e.g. next_1_week, next_4_weeks, next_8_weeks, next_12_weeks, full_52_weeks.")
    horizon_weeks: int = Field(description="Number of weeks covered in this horizon.")
    forecast_start_date: str = Field(description="Start date of this horizon.")
    forecast_end_date: str = Field(description="End date of this horizon.")
    total_forecast_quantity: float = Field(description="Sum of predicted demand units over the horizon.")
    mean_weekly_forecast: float = Field(description="Average weekly predicted demand over the horizon.")
    comparison_prior_period_dates: str = Field(description="Dates of the historical comparison period.")
    comparison_prior_period_quantity: float = Field(description="Actual demand in the equivalent comparison period.")
    pct_change_vs_prior_period: Optional[float] = Field(None, description="Percentage change vs. comparison historical period.")
    comparison_note: str = Field(description="Explicit explanation of the historical comparison period.")


class ForwardProductionForecast(BaseModel):
    """Structured forward-looking production forecast strictly for future unobserved periods."""
    model_name: str = Field(description="Champion forecasting model used (Holt-Winters).")
    forecast_origin: str = Field(description="Final observed historical week start date used to generate the forecast.")
    forecast_start_date: str = Field(description="First future unobserved forecast week start date (strictly after forecast_origin).")
    forecast_end_date: str = Field(description="Final future forecast week start date.")
    annual_forecast_total: float = Field(description="Total 52-week projected future demand.")
    comparison_historical_year: str = Field(description="Historical comparison baseline year (e.g. 2017).")
    comparison_historical_total_quantity: float = Field(description="Actual total demand in the comparison baseline year.")
    annual_projected_growth_pct: float = Field(description="Projected YoY demand growth for the full 52-week forecast vs comparison year.")
    horizons: Dict[str, ForecastHorizonSummary] = Field(description="Summaries for standard operational horizons.")
    peak_forecast_week: PeakTroughWeek = Field(description="Future week with highest projected demand.")
    trough_forecast_week: PeakTroughWeek = Field(description="Future week with lowest projected demand.")


class ModelEvaluationMetadata(BaseModel):
    """Documented holdout evaluation performance metrics for the champion model (Historical Holdout)."""
    model_name: str = Field(description="Champion model evaluated.")
    model_type: str = Field(description="Model specifications.")
    evaluation_period: str = Field(description="Historical evaluation test period (e.g. 2017 Holdout).")
    training_period: str = Field(description="Historical training period.")
    test_period: str = Field(description="Historical holdout test period.")
    test_mae: float = Field(description="Holdout Mean Absolute Error.")
    test_rmse: float = Field(description="Holdout Root Mean Squared Error.")
    test_mape: float = Field(description="Holdout Mean Absolute Percentage Error.")
    test_smape: float = Field(description="Holdout Symmetric MAPE.")
    test_bias: float = Field(description="Holdout Mean Error / Bias.")
    evaluation_notes: str = Field(description="Explicit documentation that these metrics represent historical holdout validation.")


class BusinessAnalyticsContext(BaseModel):
    """Master structured, machine-readable analytics context object for downstream consumption / LLM grounding."""
    schema_version: str = Field(default="1.0.0")
    generated_at_utc: str = Field(description="Timestamp of context generation.")
    historical_kpis: HistoricalKPIs
    time_series_growth: TimeSeriesGrowthSummary
    dimension_breakdowns: Dict[str, List[DimensionMetric]] = Field(description="Breakdowns by Category, Sub-Category, Region.")
    top_bottom_performers: TopBottomPerformers
    anomalies: List[DemandAnomaly]
    model_evaluation: ModelEvaluationMetadata = Field(description="Historical holdout evaluation benchmark on 2017.")
    forward_forecast: ForwardProductionForecast = Field(description="Genuine forward production forecast for 2018+.")
