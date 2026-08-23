"""Unit tests for Business Analytics Engine, deterministic KPIs, dimension aggregations, and forward production forecasts."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.analytics.schemas import (
    HistoricalKPIs,
    PeriodGrowthRecord,
    TimeSeriesGrowthSummary,
    DimensionMetric,
    TopBottomPerformers,
    DemandAnomaly,
    ForwardProductionForecast,
    ModelEvaluationMetadata,
    BusinessAnalyticsContext,
    FilteredPerformanceResponse
)
from src.analytics.engine import (
    calculate_safe_growth,
    calculate_historical_kpis,
    calculate_time_series_growth,
    calculate_dimension_breakdowns,
    identify_top_bottom_performers,
    detect_demand_anomalies,
    generate_forward_production_forecast,
    build_business_analytics_context,
    calculate_filtered_performance_analytics
)


@pytest.fixture
def sample_raw_and_weekly_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load real Superstore raw and weekly processed datasets if available."""
    raw_path = Path("data/raw/Sample_Superstore.csv")
    weekly_path = Path("data/processed/weekly_demand.csv")

    if not (raw_path.exists() and weekly_path.exists()):
        pytest.skip("Dataset files not found.")

    raw_df = pd.read_csv(raw_path, encoding="windows-1252")
    weekly_df = pd.read_csv(weekly_path)
    return raw_df, weekly_df


def test_calculate_safe_growth_normal_and_zero_denominator():
    """Verify safe percentage growth calculation and zero denominator handling."""
    # Positive growth: 100 -> 150 (+50%)
    rec_pos = calculate_safe_growth(150.0, 100.0, "2015", "2014")
    assert rec_pos.growth_pct == 50.0
    assert rec_pos.absolute_change == 50.0

    # Negative growth: 200 -> 150 (-25%)
    rec_neg = calculate_safe_growth(150.0, 200.0, "2015", "2014")
    assert rec_neg.growth_pct == -25.0
    assert rec_neg.absolute_change == -50.0

    # Zero previous value: 0 -> 100 (Undefined percentage, no division by zero error)
    rec_zero = calculate_safe_growth(100.0, 0.0, "2015", "2014")
    assert rec_zero.growth_pct is None
    assert "Zero denominator" in rec_zero.growth_notes
    assert rec_zero.absolute_change == 100.0


def test_calculate_historical_kpis_exact_values(sample_raw_and_weekly_dfs):
    """Verify exact calculation of core historical KPIs on full dataset."""
    raw_df, weekly_df = sample_raw_and_weekly_dfs
    kpis = calculate_historical_kpis(raw_df, weekly_df)

    assert isinstance(kpis, HistoricalKPIs)
    assert kpis.total_quantity == 37873
    assert kpis.total_sales == pytest.approx(2297200.86, abs=0.5)
    assert kpis.total_profit == pytest.approx(286397.02, abs=0.5)
    assert kpis.profit_margin_pct == pytest.approx(12.47, abs=0.05)
    assert kpis.total_orders == 5009
    assert kpis.total_transactions == 9994
    assert kpis.unique_customers == 793
    assert kpis.average_order_value == pytest.approx(458.61, abs=0.05)
    assert kpis.average_units_per_order == pytest.approx(7.56, abs=0.05)
    assert kpis.total_complete_weeks == 208


def test_calculate_time_series_growth_structure(sample_raw_and_weekly_dfs):
    """Verify time series growth summaries and chronological ordering."""
    _, weekly_df = sample_raw_and_weekly_dfs
    growth = calculate_time_series_growth(weekly_df)

    assert isinstance(growth, TimeSeriesGrowthSummary)
    assert len(growth.yoy_demand_growth) == 3  # 2015 vs 2014, 2016 vs 2015, 2017 vs 2016
    assert len(growth.yearly_totals) == 4

    # 2016 vs 2015 demand growth (+24.35%)
    yoy_2016 = [g for g in growth.yoy_demand_growth if g.period_name == "2016"][0]
    assert yoy_2016.growth_pct == pytest.approx(24.35, abs=0.1)

    # 2017 vs 2016 demand growth (+25.87%)
    yoy_2017 = [g for g in growth.yoy_demand_growth if g.period_name == "2017"][0]
    assert yoy_2017.growth_pct == pytest.approx(25.87, abs=0.1)

    assert growth.recent_4w_mean_demand > 0
    assert growth.recent_13w_mean_demand > 0


def test_calculate_dimension_breakdowns_shares_sum_to_100(sample_raw_and_weekly_dfs):
    """Verify that category and regional shares sum to exactly 100%."""
    raw_df, _ = sample_raw_and_weekly_dfs
    breakdowns = calculate_dimension_breakdowns(raw_df)

    assert "category" in breakdowns
    assert "sub_category" in breakdowns
    assert "region" in breakdowns

    # Category shares sum to 100%
    cat_qty_share_sum = sum(c.quantity_share_pct for c in breakdowns["category"])
    cat_sales_share_sum = sum(c.sales_share_pct for c in breakdowns["category"])
    assert cat_qty_share_sum == pytest.approx(100.0, abs=0.2)
    assert cat_sales_share_sum == pytest.approx(100.0, abs=0.2)

    # Regional shares sum to 100%
    reg_qty_share_sum = sum(r.quantity_share_pct for r in breakdowns["region"])
    reg_sales_share_sum = sum(r.sales_share_pct for r in breakdowns["region"])
    assert reg_qty_share_sum == pytest.approx(100.0, abs=0.2)
    assert reg_sales_share_sum == pytest.approx(100.0, abs=0.2)


def test_identify_top_bottom_performers(sample_raw_and_weekly_dfs):
    """Verify top and bottom performer ranking generation."""
    raw_df, _ = sample_raw_and_weekly_dfs
    breakdowns = calculate_dimension_breakdowns(raw_df)
    performers = identify_top_bottom_performers(breakdowns)

    assert isinstance(performers, TopBottomPerformers)
    assert performers.top_categories_by_quantity[0].dimension_value == "Office Supplies"
    assert performers.top_categories_by_sales[0].dimension_value == "Technology"
    assert performers.top_categories_by_profit[0].dimension_value == "Technology"
    assert len(performers.top_sub_categories_by_quantity) == 5
    assert len(performers.regional_rankings) == 4


def test_detect_demand_anomalies(sample_raw_and_weekly_dfs):
    """Verify rolling z-score anomaly detection and objective descriptions."""
    _, weekly_df = sample_raw_and_weekly_dfs
    anomalies = detect_demand_anomalies(weekly_df, window=13, z_threshold=2.0)

    assert isinstance(anomalies, list)
    assert len(anomalies) > 0
    assert all(isinstance(a, DemandAnomaly) for a in anomalies)
    assert all(abs(a.z_score) >= 2.0 for a in anomalies)
    assert all("standard deviations" in a.description for a in anomalies)


def test_forward_forecast_dates_strictly_after_origin(sample_raw_and_weekly_dfs):
    """Verify forward forecast dates are strictly in 2018 (after origin 2017-12-25)."""
    _, weekly_df = sample_raw_and_weekly_dfs
    f_prod, df_pred = generate_forward_production_forecast(weekly_df, horizon=52)

    assert isinstance(f_prod, ForwardProductionForecast)
    assert f_prod.forecast_origin == "2017-12-25"
    assert f_prod.forecast_start_date == "2018-01-01"
    assert f_prod.forecast_end_date == "2018-12-24"
    assert len(df_pred) == 52

    # All predicted dates must be >= 2018-01-01
    pred_dates = pd.to_datetime(df_pred["week_start"])
    assert (pred_dates >= pd.to_datetime("2018-01-01")).all()


def test_evaluation_forecast_dates_not_exposed_as_future():
    """Verify that evaluation metadata and future forecasts are strictly separated."""
    context = build_business_analytics_context()

    # Evaluation metadata must refer to 2017 holdout
    assert "2017" in context.model_evaluation.evaluation_period
    assert context.model_evaluation.test_mae == pytest.approx(39.02, abs=0.5)

    # Future forecast must start in 2018
    assert context.forward_forecast.forecast_start_date == "2018-01-01"
    assert context.forward_forecast.peak_forecast_week.week_start.startswith("2018")
    assert context.forward_forecast.trough_forecast_week.week_start.startswith("2018")


def test_forward_forecast_horizons_and_comparisons(sample_raw_and_weekly_dfs):
    """Verify horizon lengths, positive totals, and documented historical comparisons."""
    _, weekly_df = sample_raw_and_weekly_dfs
    f_prod, _ = generate_forward_production_forecast(weekly_df, horizon=52)

    assert "next_1_week" in f_prod.horizons
    assert "next_4_weeks" in f_prod.horizons
    assert "next_8_weeks" in f_prod.horizons
    assert "next_12_weeks" in f_prod.horizons
    assert "full_52_weeks" in f_prod.horizons

    h52 = f_prod.horizons["full_52_weeks"]
    assert h52.total_forecast_quantity == pytest.approx(16266.75, abs=5.0)
    assert h52.comparison_prior_period_quantity == 12420.0  # 2017 actual total
    assert h52.pct_change_vs_prior_period == pytest.approx(30.97, abs=0.2)


def test_build_business_analytics_context_json_serialization():
    """Verify end-to-end context builder execution, Pydantic validation, and JSON serialization."""
    reports_dir = Path("reports")
    if not (reports_dir / "holt_winters_evaluation.json").exists():
        pytest.skip("Holt-Winters evaluation report missing.")

    context = build_business_analytics_context()
    assert isinstance(context, BusinessAnalyticsContext)

    # Test Pydantic JSON dump and reload
    json_str = context.model_dump_json()
    reloaded_dict = json.loads(json_str)

    assert reloaded_dict["schema_version"] == "1.0.0"
    assert "historical_kpis" in reloaded_dict
    assert "time_series_growth" in reloaded_dict
    assert "dimension_breakdowns" in reloaded_dict
    assert "top_bottom_performers" in reloaded_dict
    assert "anomalies" in reloaded_dict
    assert "model_evaluation" in reloaded_dict
    assert "forward_forecast" in reloaded_dict
    assert reloaded_dict["model_evaluation"]["test_mae"] == pytest.approx(39.02, abs=0.5)
    assert reloaded_dict["forward_forecast"]["forecast_start_date"] == "2018-01-01"


def test_calculate_filtered_performance_analytics_various_combinations(sample_raw_and_weekly_dfs):
    """Verify calculate_filtered_performance_analytics produces exact deterministic results across filter combinations."""
    raw_df, _ = sample_raw_and_weekly_dfs

    # 1. Unfiltered / All
    res_all = calculate_filtered_performance_analytics(raw_df, year="All", category="All", region="All")
    assert isinstance(res_all, FilteredPerformanceResponse)
    assert res_all.record_count == 9994
    assert res_all.filtered_kpis.total_quantity == 37873
    assert res_all.filtered_kpis.total_sales == pytest.approx(2297200.86, abs=1.0)
    assert res_all.filtered_kpis.total_profit == pytest.approx(286397.02, abs=1.0)
    assert len(res_all.category_summary) == 3
    assert len(res_all.regional_summary) == 4

    # 2. Year = 2017
    res_2017 = calculate_filtered_performance_analytics(raw_df, year="2017")
    assert res_2017.record_count == 3312
    assert res_2017.filtered_kpis.total_quantity == 12476

    # 3. Category = Technology
    res_tech = calculate_filtered_performance_analytics(raw_df, category="Technology")
    assert res_tech.filtered_kpis.total_quantity == 6939
    assert res_tech.filtered_kpis.total_sales == pytest.approx(836154.03, abs=1.0)

    # 4. Region = West
    res_west = calculate_filtered_performance_analytics(raw_df, region="West")
    assert res_west.filtered_kpis.total_quantity == 12266
    assert res_west.filtered_kpis.total_sales == pytest.approx(725457.82, abs=1.0)

    # 5. Combined: Year=2017 + Category=Technology + Region=West
    res_comb = calculate_filtered_performance_analytics(raw_df, year="2017", category="Technology", region="West")
    assert res_comb.record_count == 213
    assert res_comb.filtered_kpis.total_quantity == 851
    assert res_comb.filtered_kpis.total_sales == pytest.approx(95959.15, abs=1.0)
    assert res_comb.filtered_kpis.total_profit == pytest.approx(18983.80, abs=1.0)
    assert res_comb.filtered_kpis.total_orders == 177
    assert res_comb.filtered_kpis.profit_margin_pct == pytest.approx(19.78, abs=0.1)

