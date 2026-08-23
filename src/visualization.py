"""Visualization module for time-series demand forecasting, baselines, Holt-Winters, SARIMA, and ML."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_baseline_forecast_comparison(
    complete_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    split_date_str: str,
    output_path: Path | str = "reports/baseline_forecast_comparison.png"
) -> Path:
    """Generate and save plot comparing actuals vs baseline forecasts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_full = complete_df.copy()
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_pred = forecast_df.copy()
    df_pred["week_start"] = pd.to_datetime(df_pred["week_start"])

    split_dt = pd.to_datetime(split_date_str)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True)

    # 1. Main Time Series Plot
    ax1.plot(df_full["week_start"], df_full["quantity"], label="Actual Historical Demand", color="#1f77b4", linewidth=1.8, alpha=0.85)
    ax1.plot(df_pred["week_start"], df_pred["naive_1step_pred"], label="Naive (Lag-1) Forecast", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax1.plot(df_pred["week_start"], df_pred["seasonal_naive_pred"], label="Seasonal Naive (Lag-52) Forecast", color="#2ca02c", linestyle="-.", linewidth=2.0)

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
    return path


def plot_holt_winters_comparison(
    complete_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    split_date_str: str,
    output_path: Path | str = "reports/holt_winters_forecast_comparison.png"
) -> Path:
    """Generate and save plot comparing Holt-Winters, baselines, and actuals."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_full = complete_df.copy()
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_pred = forecast_df.copy()
    df_pred["week_start"] = pd.to_datetime(df_pred["week_start"])

    split_dt = pd.to_datetime(split_date_str)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True)

    # 1. Main Time Series Plot
    ax1.plot(df_full["week_start"], df_full["quantity"], label="Actual Historical Demand", color="#1f77b4", linewidth=1.8, alpha=0.85)
    ax1.plot(df_pred["week_start"], df_pred["hw_pred"], label="Holt-Winters Forecast (Add Trend, Add Season)", color="#9467bd", linewidth=2.4)
    if "hw_pi_lower" in df_pred.columns and "hw_pi_upper" in df_pred.columns:
        ax1.fill_between(
            df_pred["week_start"],
            df_pred["hw_pi_lower"],
            df_pred["hw_pi_upper"],
            color="#9467bd",
            alpha=0.20,
            label="Holt-Winters 95% Prediction Interval"
        )
    if "seasonal_naive_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["seasonal_naive_pred"], label="Seasonal Naive (Lag-52)", color="#2ca02c", linestyle="-.", linewidth=1.8, alpha=0.8)
    if "naive_1step_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["naive_1step_pred"], label="Naive (Lag-1)", color="#ff7f0e", linestyle="--", linewidth=1.6, alpha=0.75)

    ax1.axvline(x=split_dt, color="#d62728", linestyle=":", linewidth=2.2, label=f"Train/Test Cutoff ({split_date_str})")
    ax1.axvspan(df_full["week_start"].min(), split_dt, color="#f0f2f6", alpha=0.5, label="Training Set (2014-2016)")
    ax1.axvspan(split_dt, df_full["week_start"].max(), color="#e8f4f8", alpha=0.5, label="Unseen Test Set (2017)")

    ax1.set_title("Customer Demand Forecasting — Holt-Winters vs Baselines (Weekly Quantity)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Weekly Demand (Quantity)", fontsize=12, fontweight="normal")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Residual Error Plot
    hw_error = df_pred["actual_demand"] - df_pred["hw_pred"]
    ax2.plot(df_pred["week_start"], hw_error, label="Holt-Winters Error (Actual - Pred)", color="#9467bd", linewidth=1.8)
    if "seasonal_naive_pred" in df_pred.columns:
        snaive_error = df_pred["actual_demand"] - df_pred["seasonal_naive_pred"]
        ax2.plot(df_pred["week_start"], snaive_error, label="Seasonal Naive Error", color="#2ca02c", linestyle="-.", alpha=0.7)

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.axvline(x=split_dt, color="#d62728", linestyle=":", linewidth=2.2)
    ax2.set_ylabel("Residual Error", fontsize=11, fontweight="normal")
    ax2.set_xlabel("Week Start Date", fontsize=12, fontweight="normal")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sarima_comparison(
    complete_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    split_date_str: str,
    sarima_label: str = "SARIMA(0,1,1)(0,1,1,52)",
    output_path: Path | str = "reports/sarima_forecast_comparison.png"
) -> Path:
    """Generate and save plot comparing SARIMA forecast, native prediction intervals, and actuals."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_full = complete_df.copy()
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_pred = forecast_df.copy()
    df_pred["week_start"] = pd.to_datetime(df_pred["week_start"])

    split_dt = pd.to_datetime(split_date_str)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True)

    # 1. Main Time Series Plot
    ax1.plot(df_full["week_start"], df_full["quantity"], label="Actual Historical Demand", color="#1f77b4", linewidth=1.8, alpha=0.85)
    ax1.plot(df_pred["week_start"], df_pred["sarima_pred"], label=f"SARIMA Forecast: {sarima_label}", color="#d62728", linewidth=2.4)
    if "sarima_pi_lower" in df_pred.columns and "sarima_pi_upper" in df_pred.columns:
        ax1.fill_between(
            df_pred["week_start"],
            df_pred["sarima_pi_lower"],
            df_pred["sarima_pi_upper"],
            color="#d62728",
            alpha=0.18,
            label="SARIMA 95% Native State-Space Prediction Interval"
        )

    ax1.axvline(x=split_dt, color="#333333", linestyle=":", linewidth=2.2, label=f"Train/Test Cutoff ({split_date_str})")
    ax1.axvspan(df_full["week_start"].min(), split_dt, color="#f0f2f6", alpha=0.5, label="Training Set (2014-2016)")
    ax1.axvspan(split_dt, df_full["week_start"].max(), color="#fce8e6", alpha=0.4, label="Unseen Test Set (2017)")

    ax1.set_title("Customer Demand Forecasting — SARIMA Model vs Actuals (Weekly Quantity)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Weekly Demand (Quantity)", fontsize=12, fontweight="normal")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Residual Error Plot
    sarima_error = df_pred["actual_demand"] - df_pred["sarima_pred"]
    ax2.plot(df_pred["week_start"], sarima_error, label="SARIMA Error (Actual - Pred)", color="#d62728", linewidth=1.8)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.axvline(x=split_dt, color="#333333", linestyle=":", linewidth=2.2)
    ax2.set_ylabel("Residual Error", fontsize=11, fontweight="normal")
    ax2.set_xlabel("Week Start Date", fontsize=12, fontweight="normal")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ml_forecast_comparison(
    complete_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    split_date_str: str,
    ml_model_label: str = "Ridge Regression (alpha=100.0)",
    output_path: Path | str = "reports/ml_forecast_comparison.png"
) -> Path:
    """Generate and save plot comparing ML forecast, actuals, and residuals."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_full = complete_df.copy()
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_pred = forecast_df.copy()
    df_pred["week_start"] = pd.to_datetime(df_pred["week_start"])

    split_dt = pd.to_datetime(split_date_str)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True)

    # 1. Main Time Series Plot
    ax1.plot(df_full["week_start"], df_full["quantity"], label="Actual Historical Demand", color="#1f77b4", linewidth=1.8, alpha=0.85)
    ax1.plot(df_pred["week_start"], df_pred["ml_pred"], label=f"ML Forecast: {ml_model_label}", color="#e377c2", linewidth=2.4)
    if "hw_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["hw_pred"], label="Holt-Winters Benchmark", color="#9467bd", linestyle="--", linewidth=1.8, alpha=0.8)

    ax1.axvline(x=split_dt, color="#333333", linestyle=":", linewidth=2.2, label=f"Train/Test Cutoff ({split_date_str})")
    ax1.axvspan(df_full["week_start"].min(), split_dt, color="#f0f2f6", alpha=0.5, label="Training Set (2014-2016)")
    ax1.axvspan(split_dt, df_full["week_start"].max(), color="#fdf0f7", alpha=0.4, label="Unseen Test Set (2017)")

    ax1.set_title(f"Customer Demand Forecasting — ML Model ({ml_model_label}) vs Actuals", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Weekly Demand (Quantity)", fontsize=12, fontweight="normal")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Residual Error Plot
    ml_error = df_pred["actual_demand"] - df_pred["ml_pred"]
    ax2.plot(df_pred["week_start"], ml_error, label="ML Error (Actual - Pred)", color="#e377c2", linewidth=1.8)
    if "hw_pred" in df_pred.columns:
        hw_error = df_pred["actual_demand"] - df_pred["hw_pred"]
        ax2.plot(df_pred["week_start"], hw_error, label="Holt-Winters Error", color="#9467bd", linestyle="--", alpha=0.7)

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.axvline(x=split_dt, color="#333333", linestyle=":", linewidth=2.2)
    ax2.set_ylabel("Residual Error", fontsize=11, fontweight="normal")
    ax2.set_xlabel("Week Start Date", fontsize=12, fontweight="normal")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importances(
    importance_records: List[Any],
    top_n: int = 15,
    model_name: str = "Ridge Regression",
    output_path: Path | str = "reports/feature_importance.png"
) -> Path:
    """Generate and save horizontal bar chart of top feature importances with non-causal labeling."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top_records = importance_records[:top_n]
    names = [r.feature_name if hasattr(r, "feature_name") else r["feature_name"] for r in top_records]
    vals = [r.importance_value if hasattr(r, "importance_value") else r["importance_value"] for r in top_records]

    # Reverse for top-down display
    names = names[::-1]
    vals = vals[::-1]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(names, vals, color="#1f77b4", edgecolor="#0e4975", alpha=0.85, height=0.65)
    ax.set_title(f"Top {top_n} Engineered Features by Predictive Importance — {model_name}\n(Descriptive statistical association; non-causal)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Absolute Importance / Standardized Coefficient Magnitude", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max(vals) * 0.01), bar.get_y() + bar.get_height() / 2.0, f"{width:.3f}",
                va="center", ha="left", fontsize=9, color="#333333")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all_models_comparison(
    forecast_df: pd.DataFrame,
    output_path: Path | str = "reports/all_models_comparison.png"
) -> Path:
    """Generate side-by-side comparison of test period actuals vs all models."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_pred = forecast_df.copy()
    df_pred["week_start"] = pd.to_datetime(df_pred["week_start"])

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]}, sharex=True)

    # 1. Forecast Overlay
    ax1.plot(df_pred["week_start"], df_pred["actual_demand"], label="Actual 2017 Demand", color="#1f77b4", linewidth=2.5, marker="o", markersize=3.5)
    if "hw_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["hw_pred"], label="Holt-Winters (MAE: 39.02)", color="#9467bd", linewidth=2.2)
    if "sarima_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["sarima_pred"], label="SARIMA(0,1,1)(0,1,1,52) (MAE: 46.49)", color="#d62728", linewidth=2.0, linestyle="--")
    if "ml_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["ml_pred"], label="ML: Ridge Reg (MAE: 50.02)", color="#e377c2", linewidth=2.0, linestyle=":")
    if "seasonal_naive_pred" in df_pred.columns:
        ax1.plot(df_pred["week_start"], df_pred["seasonal_naive_pred"], label="Seasonal Naive (MAE: 65.63)", color="#2ca02c", linestyle="-.", linewidth=1.6, alpha=0.7)

    ax1.set_title("Test Period Forecast Comparison: Actuals vs Holt-Winters vs SARIMA vs ML vs Baselines (2017)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Weekly Demand (Quantity)", fontsize=12, fontweight="normal")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Residual Errors
    if "hw_pred" in df_pred.columns:
        ax2.plot(df_pred["week_start"], df_pred["actual_demand"] - df_pred["hw_pred"], label="Holt-Winters Error", color="#9467bd", linewidth=1.8)
    if "sarima_pred" in df_pred.columns:
        ax2.plot(df_pred["week_start"], df_pred["actual_demand"] - df_pred["sarima_pred"], label="SARIMA Error", color="#d62728", linewidth=1.8, linestyle="--")
    if "ml_pred" in df_pred.columns:
        ax2.plot(df_pred["week_start"], df_pred["actual_demand"] - df_pred["ml_pred"], label="ML Error", color="#e377c2", linewidth=1.8, linestyle=":")
    if "seasonal_naive_pred" in df_pred.columns:
        ax2.plot(df_pred["week_start"], df_pred["actual_demand"] - df_pred["seasonal_naive_pred"], label="Seasonal Naive Error", color="#2ca02c", linestyle="-.", alpha=0.7)

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Error (Actual - Pred)", fontsize=11, fontweight="normal")
    ax2.set_xlabel("Week Start Date (2017)", fontsize=12, fontweight="normal")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.0)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
