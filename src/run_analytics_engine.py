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
    f_anal = context.forecast_analytics
    meta = context.model_metadata

    print("\n" + "=" * 80)
    print("PHASE 7: BUSINESS ANALYTICS ENGINE — STRUCTURED FACTUAL OUTPUT")
    print("=" * 80)

    print("\n1. HISTORICAL BUSINESS KPIS (Full Dataset: 2014-2017)")
    print("-" * 65)
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
    print("-" * 65)
    for d_rec, s_rec in zip(growth.yoy_demand_growth, growth.yoy_sales_growth):
        print(f"  - {d_rec.period_name} vs {d_rec.previous_period_name}:")
        print(f"    * Quantity Growth: {d_rec.previous_value:,.0f} -> {d_rec.current_value:,.0f} units ({d_rec.growth_pct:+.2f}%)")
        print(f"    * Sales Growth:    ${s_rec.previous_value:,.2f} -> ${s_rec.current_value:,.2f} ({s_rec.growth_pct:+.2f}%)")

    print("\n3. CATEGORY VOLUME & CONTRIBUTION RANKINGS")
    print("-" * 65)
    for idx, cat in enumerate(top_cat.top_categories_by_quantity, 1):
        print(f"  {idx}. {cat.dimension_value:<18} | Volume: {cat.quantity:>6,} units ({cat.quantity_share_pct:>5.1f}% share) | Sales: ${cat.sales:>10,.2f} | Margin: {cat.profit_margin_pct:>5.1f}%")

    print("\n4. DEMAND ANOMALIES DETECTED (|Z-score| >= 2.0 on 13-week baseline)")
    print("-" * 65)
    for a in context.anomalies:
        print(f"  * Week {a.week_start}: {int(a.actual_quantity):>3} units | Baseline: {a.baseline_mean:>5.1f} | Z-Score: {a.z_score:>+4.2f} ({a.anomaly_direction.upper()})")

    print("\n5. CHAMPION FORECAST HORIZON ANALYTICS (Holt-Winters Projections)")
    print("-" * 65)
    for h_key, h_summary in f_anal.horizons.items():
        pct_str = f"{h_summary.pct_change_vs_prior_period:+.2f}%" if h_summary.pct_change_vs_prior_period is not None else "N/A"
        print(f"  - {h_summary.horizon_name:<15}: {h_summary.total_forecast_quantity:>8,.1f} units total ({h_summary.mean_weekly_forecast:>5.1f} units/wk) | vs Prior Period: {pct_str:>8}")
    print(f"  - Peak Forecast Week:   {f_anal.peak_forecast_week.week_start} ({f_anal.peak_forecast_week.forecast_quantity:.1f} units)")
    print(f"  - Trough Forecast Week: {f_anal.trough_forecast_week.week_start} ({f_anal.trough_forecast_week.forecast_quantity:.1f} units)")
    print(f"  - Full Year Forecast:   {f_anal.annual_forecast_total:,.1f} units")

    print("\n6. CHAMPION MODEL EVALUATION METADATA (Holdout Benchmark)")
    print("-" * 65)
    print(f"  - Model:     {meta.model_name} ({meta.model_type})")
    print(f"  - Training:  {meta.training_period}")
    print(f"  - Test Set:  {meta.test_period}")
    print(f"  - Accuracy:  MAE = {meta.test_mae:.2f}, RMSE = {meta.test_rmse:.2f}, MAPE = {meta.test_mape:.2f}%, Bias = {meta.test_bias:+.2f} units")
    print("=" * 80)


if __name__ == "__main__":
    main()
