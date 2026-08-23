"""Executable script to build, validate, and explore the weekly demand dataset."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_inspection import load_raw_dataset
from src.data_processing import aggregate_weekly_demand, save_processed_dataset
from src.time_series_exploration import explore_weekly_time_series


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "Sample_Superstore.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "weekly_demand.csv"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from: {raw_path}")
    raw_df = load_raw_dataset(raw_path)

    print("Aggregating transactional records into weekly demand series (W-MON)...")
    weekly_df, val_profile = aggregate_weekly_demand(raw_df)

    print(f"Saving processed weekly dataset to: {output_path}")
    save_processed_dataset(weekly_df, output_path)

    print("\n" + "=" * 65)
    print("PHASE 2A: WEEKLY DATASET VALIDATION PROFILE")
    print("=" * 65)
    print(f"First Week (Week Start):    {val_profile.first_week}")
    print(f"Last Week (Week Start):     {val_profile.last_week}")
    print(f"Total Weekly Observations:  {val_profile.num_weeks}")
    print(f"Total Raw Quantity Sum:     {val_profile.total_raw_quantity:,}")
    print(f"Total Weekly Quantity Sum:  {val_profile.total_weekly_quantity:,}")
    print(f"Reconciliation Passed:      {val_profile.reconciliation_passed} (sum difference: {val_profile.total_weekly_quantity - val_profile.total_raw_quantity})")
    print(f"Mean Weekly Demand:         {val_profile.mean_weekly_quantity:.2f}")
    print(f"Median Weekly Demand:       {val_profile.median_weekly_quantity:.2f}")
    print(f"Std Weekly Demand:          {val_profile.std_weekly_quantity:.2f}")
    print(f"Min / Max Weekly Demand:    {val_profile.min_weekly_quantity} / {val_profile.max_weekly_quantity}")
    print(f"Missing Weeks Count:        {val_profile.missing_weeks_count}")
    print(f"Zero-Demand Weeks Count:    {val_profile.zero_demand_weeks_count}")
    print(f"Duplicate Weeks Count:      {val_profile.duplicate_weeks_count}")

    with open(reports_dir / "weekly_validation_profile.json", "w") as f:
        json.dump(val_profile.__dict__, f, indent=2)

    print("\n" + "=" * 65)
    print("PHASE 2B: TIME-SERIES STATISTICAL EXPLORATION")
    print("=" * 65)
    ts_report = explore_weekly_time_series(weekly_df)

    print("1. Trend Analysis:")
    print(f"   - Linear Trend Slope:     +{ts_report.trend_analysis['linear_slope_per_week']:.4f} units / week")
    print(f"   - Direction:              {ts_report.trend_analysis['direction']}")
    print(f"   - Total Trend Expansion:  +{ts_report.trend_analysis['total_estimated_trend_change']:.2f} units over horizon")

    print("\n2. Annual Demand Summary:")
    for yr, data in ts_report.annual_patterns.items():
        growth_str = f" (YoY: {data['yoy_demand_growth_pct']:+.2f}%)" if data['yoy_demand_growth_pct'] is not None else ""
        print(f"   - Year {yr}: Total = {data['total_demand']:,} units, Mean = {data['mean_weekly_demand']:.1f}/wk, Weeks = {data['num_weeks']}{growth_str}")

    print("\n3. Autocorrelation & Seasonality Diagnostics:")
    print(f"   - Lag 1 Autocorrelation (r_1):   {ts_report.seasonality_evidence['lag_1_autocorrelation']:.4f}")
    print(f"   - Lag 52 Autocorrelation (r_52): {ts_report.seasonality_evidence['lag_52_autocorrelation']:.4f}")
    print("   - Top Positive ACF Lags:         " + ", ".join([f"Lag {x['lag']} (r={x['autocorrelation']:.3f})" for x in ts_report.seasonality_evidence['top_autocorrelation_lags']]))

    print("\n4. Volatility & Rolling Range:")
    print(f"   - Coefficient of Variation (CV): {ts_report.volatility_metrics['coefficient_of_variation']:.4f}")
    print(f"   - 4-Week Rolling Mean Range:     {ts_report.rolling_statistics['rolling_4w_mean']['min']} to {ts_report.rolling_statistics['rolling_4w_mean']['max']} units")
    print(f"   - 12-Week Rolling Mean Range:    {ts_report.rolling_statistics['rolling_12w_mean']['min']} to {ts_report.rolling_statistics['rolling_12w_mean']['max']} units")

    print("\n5. Outlier & Extreme Demand Weeks:")
    print(f"   - Unusually High Weeks (> {ts_report.extreme_weeks['high_threshold']:.1f} units): {ts_report.extreme_weeks['high_weeks_count']} occurrences")
    for w in ts_report.extreme_weeks['high_weeks']:
        print(f"     * {w['week']}: {w['quantity']} units")
    print(f"   - Unusually Low Weeks (< {ts_report.extreme_weeks['low_threshold']:.1f} units): {ts_report.extreme_weeks['low_weeks_count']} occurrences")
    for w in ts_report.extreme_weeks['low_weeks']:
        print(f"     * {w['week']}: {w['quantity']} units")

    print("\n6. Stationarity Diagnostics:")
    print(f"   - ADF Test Statistic:  {ts_report.stationarity_tests['adf_test']['test_statistic']:.4f} (p-value: {ts_report.stationarity_tests['adf_test']['p_value']})")
    print(f"   - KPSS Test Statistic: {ts_report.stationarity_tests['kpss_test']['test_statistic']:.4f} (p-value: {ts_report.stationarity_tests['kpss_test']['p_value']})")
    print(f"   - Interpretation:      {ts_report.stationarity_tests['joint_interpretation']}")

    print("\nKey Analytical Findings:")
    for finding in ts_report.key_findings:
        print(f"  * {finding}")

    with open(reports_dir / "time_series_exploration.json", "w") as f:
        json.dump(ts_report.__dict__, f, indent=2)

    print(f"\nAll reports generated successfully in {reports_dir}")


if __name__ == "__main__":
    main()
