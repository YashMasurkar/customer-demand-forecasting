"""Executable script to run baseline forecasting models, evaluate accuracy, and generate reports."""

import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import run_baseline_evaluation
from src.visualization import plot_baseline_forecast_comparison


def main():
    processed_path = PROJECT_ROOT / "data" / "processed" / "weekly_demand.csv"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed weekly demand data from: {processed_path}")
    weekly_df = pd.read_csv(processed_path)

    print("Running chronological baseline evaluation (52-week test holdout)...")
    result = run_baseline_evaluation(weekly_df, target_col="quantity", test_size_weeks=52)

    print("\n" + "=" * 70)
    print("PHASE 3: BASELINE FORECAST EVALUATION REPORT")
    print("=" * 70)
    split = result.split_info
    print(f"Total Complete Weeks:       {split.total_complete_observations}")
    print(f"Excluded Partial Weeks:     {split.partial_weeks_excluded_count} ({split.partial_weeks_excluded})")
    print(f"Training Period:            {split.train_start_date} to {split.train_end_date} ({split.num_train_observations} weeks, {100*(1-split.test_ratio):.1f}%)")
    print(f"Test Period (Unseen):       {split.test_start_date} to {split.test_end_date} ({split.num_test_observations} weeks, {100*split.test_ratio:.1f}%)")

    print("\n" + "-" * 70)
    print("BASELINE ACCURACY COMPARISON TABLE")
    print("-" * 70)
    header = f"{'Model':<26} | {'MAE':>8} | {'RMSE':>8} | {'MAPE':>8} | {'sMAPE':>8} | {'Bias (ME)':>10}"
    print(header)
    print("-" * len(header))

    models = [
        result.naive_1step_metrics,
        result.seasonal_naive_metrics,
        result.naive_fixed_metrics
    ]

    for m in models:
        mape_str = f"{m.mape:.2f}%" if m.mape is not None else "N/A"
        print(f"{m.model_name:<26} | {m.mae:>8.2f} | {m.rmse:>8.2f} | {mape_str:>8} | {m.smape:>7.2f}% | {m.mean_error:>+10.2f}")

    print("\nMAPE Applicability Assessment:")
    print(f"  * {result.naive_1step_metrics.mape_notes}")

    print(f"\nComparative Assessment:\n  * {result.better_model_summary}")

    # Save JSON evaluation report
    report_dict = {
        "split_info": result.split_info.__dict__,
        "naive_1step_metrics": result.naive_1step_metrics.__dict__,
        "seasonal_naive_metrics": result.seasonal_naive_metrics.__dict__,
        "naive_fixed_metrics": result.naive_fixed_metrics.__dict__,
        "better_model_1step": result.better_model_1step,
        "better_model_summary": result.better_model_summary
    }

    report_json_path = reports_dir / "baseline_evaluation.json"
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nSaved evaluation metrics JSON: {report_json_path}")

    # Save forecast CSV
    forecasts_csv_path = reports_dir / "baseline_forecasts.csv"
    result.forecast_df.to_csv(forecasts_csv_path, index=False)
    print(f"Saved test forecasts CSV:      {forecasts_csv_path}")

    # Generate and save chart
    chart_path = reports_dir / "baseline_forecast_comparison.png"
    complete_df = weekly_df[~weekly_df["is_partial_week"]].copy().reset_index(drop=True)
    plot_baseline_forecast_comparison(
        complete_df=complete_df,
        forecast_df=result.forecast_df,
        split_date_str=split.test_start_date,
        output_path=chart_path
    )


if __name__ == "__main__":
    main()
