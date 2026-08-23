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


class ForecastHorizonSummary(BaseModel):
    """Aggregated forward forecast metrics for a specific planning horizon."""
    horizon_name: str = Field(description="e.g. next_1_week, next_4_weeks, next_8_weeks, next_12_weeks, full_52_weeks.")
    horizon_weeks: int = Field(description="Number of weeks covered in this horizon.")
    total_forecast_quantity: float = Field(description="Sum of predicted demand units over the horizon.")
    mean_weekly_forecast: float = Field(description="Average weekly predicted demand over the horizon.")
    prior_period_actual_quantity: float = Field(description="Actual demand in the equivalent preceding period.")
    pct_change_vs_prior_period: Optional[float] = Field(None, description="Percentage change vs. preceding equivalent historical period.")


class PeakTroughWeek(BaseModel):
    """Date and value of maximum or minimum forecast week."""
    week_start: str
    forecast_quantity: float
    description: str


class ForecastAnalytics(BaseModel):
    """Structured forward-looking forecast outputs from the champion model."""
    champion_model: str = Field(description="Name of the champion forecasting model (Holt-Winters).")
    forecast_start_date: str = Field(description="First forecast week start date.")
    forecast_end_date: str = Field(description="Final forecast week start date.")
    horizons: Dict[str, ForecastHorizonSummary] = Field(description="Summaries for standard operational horizons.")
    peak_forecast_week: PeakTroughWeek = Field(description="Week with highest projected demand.")
    trough_forecast_week: PeakTroughWeek = Field(description="Week with lowest projected demand.")
    annual_forecast_total: float = Field(description="Total 52-week projected annual demand.")


class ModelEvaluationMetadata(BaseModel):
    """Documented holdout evaluation performance metrics for the champion model."""
    model_name: str
    model_type: str
    training_period: str
    test_period: str
    test_mae: float
    test_rmse: float
    test_mape: float
    test_smape: float
    test_bias: float
    evaluation_notes: str = Field(description="Explicitly documents that evaluation metrics reflect unseen holdout test performance.")


class BusinessAnalyticsContext(BaseModel):
    """Master structured, machine-readable analytics context object for downstream consumption / LLM grounding."""
    schema_version: str = Field(default="1.0.0")
    generated_at_utc: str = Field(description="Timestamp of context generation.")
    historical_kpis: HistoricalKPIs
    time_series_growth: TimeSeriesGrowthSummary
    dimension_breakdowns: Dict[str, List[DimensionMetric]] = Field(description="Breakdowns by Category, Sub-Category, Region.")
    top_bottom_performers: TopBottomPerformers
    anomalies: List[DemandAnomaly]
    forecast_analytics: ForecastAnalytics
    model_metadata: ModelEvaluationMetadata
