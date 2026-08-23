"""Visualization module for time-series demand forecasting and baseline model comparisons."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_baseline_forecast_comparison(
    complete_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    split_date_str: str,
    output_path: Path | str = "reports/baseline_forecast_comparison.png"
) -> Path:
    """Generate and save publication-quality plot comparing actuals vs baseline forecasts.

    Args:
        complete_df: Full complete weekly dataframe (columns: week_start, quantity).
        forecast_df: Test forecast dataframe (columns: week_start, actual_demand, naive_1step_pred, seasonal_naive_pred).
        split_date_str: String date marking the start of the test split (e.g. '2017-01-02').
        output_path: Target PNG file path.

    Returns:
        Path to the saved PNG image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_full = complete_df.copy()
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_pred = forecast_df.copy()
    df_pred["week_start"] = pd.to_datetime(df_pred["week_start"])

    split_dt = pd.to_datetime(split_date_str)

    # Styling setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True)

    # 1. Main Time Series Plot
    ax1.plot(
        df_full["week_start"],
        df_full["quantity"],
        label="Actual Historical Demand",
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.85
    )

    # Plot test forecasts
    ax1.plot(
        df_pred["week_start"],
        df_pred["naive_1step_pred"],
        label="Naive (Lag-1) Forecast",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2.0
    )
    ax1.plot(
        df_pred["week_start"],
        df_pred["seasonal_naive_pred"],
        label="Seasonal Naive (Lag-52) Forecast",
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.0
    )

    # Train / Test Demarcation
    ax1.axvline(x=split_dt, color="#d62728", linestyle=":", linewidth=2.2, label=f"Train/Test Cutoff ({split_date_str})")
    ax1.axvspan(df_full["week_start"].min(), split_dt, color="#f0f2f6", alpha=0.5, label="Training Period (2014-2016)")
    ax1.axvspan(split_dt, df_full["week_start"].max(), color="#e8f4f8", alpha=0.5, label="Unseen Test Period (2017)")

    ax1.set_title("Customer Demand Forecasting — Baseline Models vs Actuals (Weekly Quantity)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Weekly Demand (Quantity)", fontsize=12, fontweight="normal")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Test Period Error / Residuals Plot
    errors_naive = df_pred["actual_demand"] - df_pred["naive_1step_pred"]
    errors_snaive = df_pred["actual_demand"] - df_pred["seasonal_naive_pred"]

    ax2.plot(df_pred["week_start"], errors_naive, label="Naive Error (Actual - Pred)", color="#ff7f0e", linestyle="--", alpha=0.85)
    ax2.plot(df_pred["week_start"], errors_snaive, label="Seasonal Naive Error (Actual - Pred)", color="#2ca02c", linestyle="-.", alpha=0.85)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.axvline(x=split_dt, color="#d62728", linestyle=":", linewidth=2.2)
    ax2.set_ylabel("Residual Error", fontsize=11, fontweight="normal")
    ax2.set_xlabel("Week Start Date", fontsize=12, fontweight="normal")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Forecast comparison plot successfully saved to: {path}")
    return path
