"""Forecast API route serving forward production forecasts with horizon filtering."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path

from src.api.routes.analytics import get_cached_analytics_context

router = APIRouter(tags=["Forecast"])

ALLOWED_HORIZONS = {1, 4, 8, 12, 52}


class ForecastRecord(BaseModel):
    week_start: str
    forecast_quantity: float
    pi_lower_95: Optional[float] = None
    pi_upper_95: Optional[float] = None


class ForecastResponse(BaseModel):
    model_name: str
    forecast_origin: str
    forecast_start_date: str
    forecast_end_date: str
    horizon_weeks: int
    total_forecast_quantity: float
    mean_weekly_forecast: float
    peak_forecast_week: Dict[str, Any]
    trough_forecast_week: Dict[str, Any]
    comparison_historical_year: str
    comparison_historical_total_quantity: float
    projected_growth_pct: Optional[float]
    comparison_note: str
    forecast_records: List[ForecastRecord]


@router.get("/forecast", response_model=ForecastResponse)
def get_forward_forecast(
    horizon: int = Query(52, description="Forecast horizon in weeks. Allowed values: [1, 4, 8, 12, 52]")
) -> ForecastResponse:
    """Return forward production forecast for a specific planning horizon (1, 4, 8, 12, or 52 weeks)."""
    if horizon not in ALLOWED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon: {horizon}. Allowed horizons are [1, 4, 8, 12, 52]."
        )

    context = get_cached_analytics_context()
    f_prod = context.forward_forecast

    # Load forward production CSV
    forward_csv_path = Path("reports/forward_production_forecasts.csv")
    if forward_csv_path.exists():
        df_forward = pd.read_csv(forward_csv_path)
    else:
        # Fallback to generating on the fly if needed
        from src.analytics.engine import generate_forward_production_forecast
        weekly_df = pd.read_csv("data/processed/weekly_demand.csv")
        _, df_forward = generate_forward_production_forecast(weekly_df, horizon=52)

    # Slice by requested horizon
    df_sliced = df_forward.iloc[:horizon].copy()
    records = [
        ForecastRecord(
            week_start=str(row["week_start"]),
            forecast_quantity=float(row["forecast_quantity"]),
            pi_lower_95=float(row["pi_lower_95"]) if "pi_lower_95" in row and pd.notna(row["pi_lower_95"]) else None,
            pi_upper_95=float(row["pi_upper_95"]) if "pi_upper_95" in row and pd.notna(row["pi_upper_95"]) else None,
        )
        for _, row in df_sliced.iterrows()
    ]

    h_key_map = {
        1: "next_1_week",
        4: "next_4_weeks",
        8: "next_8_weeks",
        12: "next_12_weeks",
        52: "full_52_weeks"
    }
    h_summary = f_prod.horizons.get(h_key_map[horizon])

    total_qty = h_summary.total_forecast_quantity if h_summary else round(float(df_sliced["forecast_quantity"].sum()), 2)
    mean_qty = h_summary.mean_weekly_forecast if h_summary else round(float(df_sliced["forecast_quantity"].mean()), 2)
    start_date = str(df_sliced["week_start"].iloc[0])
    end_date = str(df_sliced["week_start"].iloc[-1])
    growth_pct = h_summary.pct_change_vs_prior_period if h_summary else None
    comp_note = h_summary.comparison_note if h_summary else ""
    prior_qty = h_summary.comparison_prior_period_quantity if h_summary else 0.0

    return ForecastResponse(
        model_name=f_prod.model_name,
        forecast_origin=f_prod.forecast_origin,
        forecast_start_date=start_date,
        forecast_end_date=end_date,
        horizon_weeks=horizon,
        total_forecast_quantity=total_qty,
        mean_weekly_forecast=mean_qty,
        peak_forecast_week=f_prod.peak_forecast_week.model_dump(),
        trough_forecast_week=f_prod.trough_forecast_week.model_dump(),
        comparison_historical_year=f_prod.comparison_historical_year,
        comparison_historical_total_quantity=prior_qty,
        projected_growth_pct=growth_pct,
        comparison_note=comp_note,
        forecast_records=records
    )
