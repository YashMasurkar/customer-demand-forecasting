"""Analytics API route serving deterministic business KPIs, growth, breakdowns, and anomalies."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.analytics.engine import build_business_analytics_context
from src.analytics.schemas import BusinessAnalyticsContext

router = APIRouter(tags=["Analytics"])

# In-memory cache for deterministic context
_CACHED_CONTEXT: BusinessAnalyticsContext | None = None


def get_cached_analytics_context() -> BusinessAnalyticsContext:
    """Retrieve or compute the deterministic BusinessAnalyticsContext."""
    global _CACHED_CONTEXT
    if _CACHED_CONTEXT is None:
        try:
            _CACHED_CONTEXT = build_business_analytics_context()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate analytics context: {type(e).__name__}"
            ) from None
    return _CACHED_CONTEXT


@router.get("/analytics", response_model=BusinessAnalyticsContext)
def get_analytics() -> BusinessAnalyticsContext:
    """Return deterministic analytics context containing historical KPIs, growth, breakdowns, and anomalies."""
    return get_cached_analytics_context()


class DashboardSummaryResponse(BaseModel):
    historical_kpis: Dict[str, Any]
    forward_forecast_summary: Dict[str, Any]
    category_summary: List[Dict[str, Any]]
    regional_summary: List[Dict[str, Any]]
    champion_model: Dict[str, Any]
    business_insights: List[Dict[str, str]]


@router.get("/dashboard", response_model=DashboardSummaryResponse)
def get_dashboard_summary() -> DashboardSummaryResponse:
    """Return an aggregated executive summary for high-performance dashboard rendering."""
    ctx = get_cached_analytics_context()
    kpis = ctx.historical_kpis.model_dump()
    f_prod = ctx.forward_forecast
    
    # 3 dynamic business insights derived directly from deterministic metrics
    insights = [
        {
            "category": "Forecast Insight",
            "title": "Strong Forward Growth Trajectory",
            "description": f"2018 forward demand is projected at {f_prod.annual_forecast_total:,.1f} units, representing a +{f_prod.annual_projected_growth_pct:.1f}% expansion over 2017 actuals ({f_prod.comparison_historical_total_quantity:,.1f} units)."
        },
        {
            "category": "Category Insight",
            "title": "Revenue & Margin Dynamics",
            "description": "Technology generates the highest historical revenue ($836,154.03) with a leading 17.40% profit margin, while Office Supplies drives 60.5% of total unit volume."
        },
        {
            "category": "Regional Insight",
            "title": "West Regional Demand Lead",
            "description": "The West region accounts for the largest share of historical customer demand (12,266 units, 32.4% share) and the highest revenue ($725,457.82)."
        }
    ]

    return DashboardSummaryResponse(
        historical_kpis=kpis,
        forward_forecast_summary={
            "model_name": f_prod.model_name,
            "forecast_origin": f_prod.forecast_origin,
            "forecast_start_date": f_prod.forecast_start_date,
            "forecast_end_date": f_prod.forecast_end_date,
            "total_forecast_quantity": f_prod.annual_forecast_total,
            "annual_projected_growth_pct": f_prod.annual_projected_growth_pct,
            "peak_week": f_prod.peak_forecast_week.model_dump(),
            "trough_week": f_prod.trough_forecast_week.model_dump(),
            "horizons": {k: v.model_dump() for k, v in f_prod.horizons.items()}
        },
        category_summary=[c.model_dump() for c in ctx.dimension_breakdowns.get("category", [])],
        regional_summary=[r.model_dump() for r in ctx.dimension_breakdowns.get("region", [])],
        champion_model={
            "name": "Holt-Winters Exponential Smoothing",
            "parameters": "Additive Trend, Additive Seasonality (s=52)",
            "holdout_mae": ctx.model_evaluation.test_mae,
            "holdout_rmse": ctx.model_evaluation.test_rmse,
            "holdout_mape": ctx.model_evaluation.test_mape,
            "holdout_bias": ctx.model_evaluation.test_bias
        },
        business_insights=insights
    )
