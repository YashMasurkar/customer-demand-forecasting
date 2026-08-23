"""Executable script to generate deterministic Business Analytics Context and export JSON report."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.engine import build_business_analytics_context


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "Sample_Superstore.csv"
    weekly_path = PROJECT_ROOT / "data" / "processed" / "weekly_demand.csv"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Executing Deterministic Business Analytics Engine...")
    context = build_business_analytics_context(
        raw_df_path=raw_path,
        weekly_df_path=weekly_path,
        reports_dir=reports_dir
    )

    # Export structured JSON context
    output_json_path = reports_dir / "business_analytics_context.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(context.model_dump_json(indent=2))
    print(f"\nSuccessfully generated and exported: {output_json_path}")

    # Display structured summary
    kpis = context.historical_kpis
    growth = context.time_series_growth
    top_cat = context.top_bottom_performers
    f_prod = context.forward_forecast
    m_eval = context.model_evaluation

    print("\n" + "=" * 85)
    print("PHASE 7: BUSINESS ANALYTICS ENGINE - STRUCTURED FACTUAL OUTPUT")
    print("=" * 85)

    print("\n1. HISTORICAL BUSINESS KPIS (Full Dataset: 2014-2017)")
    print("-" * 75)
    print(f"  - Total Demand Sold:            {kpis.total_quantity:,} units")
    print(f"  - Total Revenue (Sales):        ${kpis.total_sales:,.2f}")
    print(f"  - Total Net Profit:             ${kpis.total_profit:,.2f}")
    print(f"  - Overall Profit Margin:        {kpis.profit_margin_pct:.2f}%")
    print(f"  - Unique Orders / Transactions: {kpis.total_orders:,} orders ({kpis.total_transactions:,} lines)")
    print(f"  - Distinct Customers:           {kpis.unique_customers:,} accounts")
    print(f"  - Average Order Value (AOV):    ${kpis.average_order_value:,.2f} / order")
    print(f"  - Average Units Per Order:      {kpis.average_units_per_order:.2f} units / order")
    print(f"  - Average Weekly Demand:        {kpis.average_weekly_demand:.2f} units / week (over {kpis.total_complete_weeks} weeks)")

    print("\n2. ANNUAL GROWTH DYNAMICS (YoY)")
    print("-" * 75)
    for d_rec, s_rec in zip(growth.yoy_demand_growth, growth.yoy_sales_growth):
        print(f"  - {d_rec.period_name} vs {d_rec.previous_period_name}:")
        print(f"    * Quantity Growth: {d_rec.previous_value:,.0f} -> {d_rec.current_value:,.0f} units ({d_rec.growth_pct:+.2f}%)")
        print(f"    * Sales Growth:    ${s_rec.previous_value:,.2f} -> ${s_rec.current_value:,.2f} ({s_rec.growth_pct:+.2f}%)")

    print("\n3. CATEGORY VOLUME & CONTRIBUTION RANKINGS")
    print("-" * 75)
    for idx, cat in enumerate(top_cat.top_categories_by_quantity, 1):
        print(f"  {idx}. {cat.dimension_value:<18} | Volume: {cat.quantity:>6,} units ({cat.quantity_share_pct:>5.1f}% share) | Sales: ${cat.sales:>10,.2f} | Margin: {cat.profit_margin_pct:>5.1f}%")

    print("\n4. DEMAND ANOMALIES DETECTED (|Z-score| >= 2.0 on 13-week baseline)")
    print("-" * 75)
    for a in context.anomalies:
        print(f"  * Week {a.week_start}: {int(a.actual_quantity):>3} units | Baseline: {a.baseline_mean:>5.1f} | Z-Score: {a.z_score:>+4.2f} ({a.anomaly_direction.upper()})")

    print("\n5. FORWARD BUSINESS FORECAST (Genuinely Future 2018 Projections)")
    print("-" * 75)
    print(f"  - Champion Model:        {f_prod.model_name}")
    print(f"  - Forecast Origin:       {f_prod.forecast_origin} (Last observed complete historical week)")
    print(f"  - First Future Forecast: {f_prod.forecast_start_date}")
    print(f"  - Final Future Forecast: {f_prod.forecast_end_date}")
    print(f"  - 52-Week Total Demand:  {f_prod.annual_forecast_total:,.1f} units (vs {f_prod.comparison_historical_year} actuals: {f_prod.annual_projected_growth_pct:+.2f}%)")
    print("\n  Operational Forecast Horizons:")
    for h_key, h_summary in f_prod.horizons.items():
        pct_str = f"{h_summary.pct_change_vs_prior_period:+.2f}%" if h_summary.pct_change_vs_prior_period is not None else "N/A"
        print(f"    * {h_summary.horizon_name:<15} ({h_summary.forecast_start_date} to {h_summary.forecast_end_date}): {h_summary.total_forecast_quantity:>8,.1f} units total ({h_summary.mean_weekly_forecast:>5.1f} units/wk) | vs Baseline ({h_summary.comparison_prior_period_quantity:>6.1f} units): {pct_str:>8}")
        print(f"      Note: {h_summary.comparison_note}")
    print(f"\n  - Future Peak Forecast Week:   {f_prod.peak_forecast_week.week_start} ({f_prod.peak_forecast_week.forecast_quantity:.1f} units)")
    print(f"  - Future Trough Forecast Week: {f_prod.trough_forecast_week.week_start} ({f_prod.trough_forecast_week.forecast_quantity:.1f} units)")

    print("\n6. HISTORICAL MODEL EVALUATION METADATA (Holdout Benchmark on 2017)")
    print("-" * 75)
    print(f"  - Model:     {m_eval.model_name} ({m_eval.model_type})")
    print(f"  - Training:  {m_eval.training_period}")
    print(f"  - Test Set:  {m_eval.test_period}")
    print(f"  - Accuracy:  MAE = {m_eval.test_mae:.2f}, RMSE = {m_eval.test_rmse:.2f}, MAPE = {m_eval.test_mape:.2f}%, Bias = {m_eval.test_bias:+.2f} units")
    print(f"  - Note:      {m_eval.evaluation_notes}")
    print("=" * 85)


if __name__ == "__main__":
    main()
