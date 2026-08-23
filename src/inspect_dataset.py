import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_inspection import load_raw_dataset, inspect_dataset


def main():
    raw_path = Path("data/raw/Sample_Superstore.csv")
    print(f"Loading raw dataset from {raw_path}...")
    df = load_raw_dataset(raw_path)

    print("Running comprehensive dataset inspection...")
    profile = inspect_dataset(df, file_path=str(raw_path))

    print("\n" + "=" * 60)
    print("DATASET INSPECTION SUMMARY REPORT")
    print("=" * 60)
    print(f"File Path:            {profile.file_path}")
    print(f"Total Rows:           {profile.row_count}")
    print(f"Total Columns:        {profile.column_count}")
    print(f"Duplicate Rows:       {profile.duplicate_rows_count}")
    print(f"Date Range (Orders):  {profile.date_range['order_date_min']} to {profile.date_range['order_date_max']}")
    print(f"Calendar Days Span:   {profile.date_range['total_calendar_days']} days")
    print(f"Active Order Days:    {profile.date_range['active_order_days']} days")
    print(f"Zero-Order Days:      {profile.date_range['days_without_orders']} days")

    print("\nUnique Entities:")
    for entity, count in profile.unique_counts.items():
        print(f"  - {entity:<24}: {count}")

    print("\nCategories Breakdown:")
    for cat, count in profile.category_breakdown.items():
        print(f"  - {cat:<20}: {count} rows ({count/profile.row_count*100:.1f}%)")

    print("\nSub-Categories Breakdown:")
    for subcat, count in profile.sub_category_breakdown.items():
        print(f"  - {subcat:<20}: {count} rows")

    print("\nRegions Breakdown:")
    for region, count in profile.region_breakdown.items():
        print(f"  - {region:<20}: {count} rows")

    print("\nNumerical Features Summary:")
    for col, stats in profile.numerical_summary.items():
        print(f"  [{col}] Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Median: {stats['50%']:.2f}, Max: {stats['max']:.2f}, Skew: {stats['skew']:.2f}")

    print("\nMissing Values Count per Column:")
    has_missing = False
    for col, missing_cnt in profile.missing_values.items():
        if missing_cnt > 0:
            print(f"  - {col}: {missing_cnt} ({profile.missing_percentage[col]}%)")
            has_missing = True
    if not has_missing:
        print("  None (0 missing values across all 21 columns)")

    print("\n" + "-" * 60)
    print("PRELIMINARY TEMPORAL AGGREGATION COMPARISONS")
    print("-" * 60)
    for summary in [profile.daily_summary, profile.weekly_summary, profile.monthly_summary]:
        print(f"Grain: {summary['grain']}")
        print(f"  - Total Periods:         {summary['total_periods']}")
        print(f"  - Zero-Demand Periods:   {summary['zero_demand_periods']} ({summary['zero_demand_ratio']*100:.2f}%)")
        print(f"  - Total Demand Quantity: {summary['demand_quantity']['total']:,.0f}")
        print(f"  - Mean Demand Quantity:  {summary['demand_quantity']['mean']:,.2f} (Std: {summary['demand_quantity']['std']:,.2f}, CV: {summary['demand_quantity']['coefficient_of_variation']:.4f})")
        print(f"  - Min / Median / Max Qty: {summary['demand_quantity']['min']} / {summary['demand_quantity']['median']} / {summary['demand_quantity']['max']}")
        print(f"  - Mean Sales Value:      ${summary['sales_value']['mean']:,.2f} (Total: ${summary['sales_value']['total']:,.2f})")
        print()

    print("Data Quality Notes:")
    for note in profile.data_quality_notes:
        print(f"  * {note}")

    # Optionally persist profile as JSON in reports directory for reference
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "dataset_profile.json", "w") as f:
        # Convert profile to serializable dict
        json.dump(profile.__dict__, f, indent=2)
    print(f"\nInspection JSON report saved to: {reports_dir / 'dataset_profile.json'}")


if __name__ == "__main__":
    main()
