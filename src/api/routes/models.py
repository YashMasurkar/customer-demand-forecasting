"""Model evaluation API route serving benchmark results across all evaluated forecasting models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["Models"])


class ModelBenchmarkRecord(BaseModel):
    model_name: str
    model_family: str
    mae: float
    rmse: float
    mape: float
    smape: float
    bias: float
    is_champion: bool = False
    rank: int


class ModelsResponse(BaseModel):
    champion_model: str
    evaluation_period: str
    dataset_context: str
    models: List[ModelBenchmarkRecord]


@router.get("/models", response_model=ModelsResponse)
def get_models_benchmark() -> ModelsResponse:
    """Return comprehensive out-of-sample holdout benchmarks across all evaluated forecasting models."""
    # Pre-compiled verified benchmark records from Phases 3, 4, 5, and 6
    benchmark_data = [
        ModelBenchmarkRecord(
            model_name="Holt-Winters (Add Trend, Add Season)",
            model_family="Classical Exponential Smoothing",
            mae=39.02,
            rmse=52.40,
            mape=19.24,
            smape=17.40,
            bias=3.76,
            is_champion=True,
            rank=1
        ),
        ModelBenchmarkRecord(
            model_name="SARIMA(0, 1, 1)(0, 1, 1, 52)",
            model_family="Classical State-Space ARIMA",
            mae=46.49,
            rmse=59.68,
            mape=20.86,
            smape=20.76,
            bias=20.97,
            is_champion=False,
            rank=2
        ),
        ModelBenchmarkRecord(
            model_name="Ridge Regression (alpha=100.0)",
            model_family="Machine Learning (Linear Regularized)",
            mae=50.02,
            rmse=66.79,
            mape=22.50,
            smape=21.28,
            bias=14.65,
            is_champion=False,
            rank=3
        ),
        ModelBenchmarkRecord(
            model_name="Ridge Regression (alpha=10.0)",
            model_family="Machine Learning (Linear Regularized)",
            mae=52.01,
            rmse=65.66,
            mape=24.22,
            smape=22.25,
            bias=4.64,
            is_champion=False,
            rank=4
        ),
        ModelBenchmarkRecord(
            model_name="Ridge Regression (alpha=1.0)",
            model_family="Machine Learning (Linear Regularized)",
            mae=57.28,
            rmse=69.92,
            mape=27.04,
            smape=24.32,
            bias=-1.06,
            is_champion=False,
            rank=5
        ),
        ModelBenchmarkRecord(
            model_name="Linear Regression (OLS)",
            model_family="Machine Learning (Ordinary Least Squares)",
            mae=60.35,
            rmse=79.84,
            mape=27.52,
            smape=25.07,
            bias=2.92,
            is_champion=False,
            rank=6
        ),
        ModelBenchmarkRecord(
            model_name="Gradient Boosting (depth=3)",
            model_family="Machine Learning (Tree Ensemble)",
            mae=63.25,
            rmse=81.61,
            mape=26.38,
            smape=26.26,
            bias=25.03,
            is_champion=False,
            rank=7
        ),
        ModelBenchmarkRecord(
            model_name="Naive (Lag-1)",
            model_family="Simple Baseline",
            mae=63.75,
            rmse=84.39,
            mape=30.93,
            smape=27.86,
            bias=0.25,
            is_champion=False,
            rank=8
        ),
        ModelBenchmarkRecord(
            model_name="Seasonal Naive (Lag-52)",
            model_family="Seasonal Baseline",
            mae=65.63,
            rmse=84.66,
            mape=28.03,
            smape=33.41,
            bias=49.10,
            is_champion=False,
            rank=9
        ),
        ModelBenchmarkRecord(
            model_name="Random Forest (max_depth=6)",
            model_family="Machine Learning (Tree Ensemble)",
            mae=67.76,
            rmse=86.90,
            mape=27.80,
            smape=28.45,
            bias=32.93,
            is_champion=False,
            rank=10
        ),
        ModelBenchmarkRecord(
            model_name="Random Forest (max_depth=4)",
            model_family="Machine Learning (Tree Ensemble)",
            mae=68.10,
            rmse=86.49,
            mape=27.85,
            smape=28.68,
            bias=34.09,
            is_champion=False,
            rank=11
        ),
        ModelBenchmarkRecord(
            model_name="HistGradientBoosting",
            model_family="Machine Learning (Tree Ensemble)",
            mae=71.17,
            rmse=87.88,
            mape=30.29,
            smape=31.86,
            bias=32.43,
            is_champion=False,
            rank=12
        ),
    ]

    return ModelsResponse(
        champion_model="Holt-Winters (Additive Trend, Additive Seasonality, s=52)",
        evaluation_period="2017 Chronological Holdout (52 Unseen Weeks: 2017-01-02 to 2017-12-25)",
        dataset_context="Empirical benchmark on Tableau Superstore weekly demand series (3-year training history 2014-2016). Holt-Winters minimized MAE (39.02) and RMSE (52.40) while capturing multi-year trend and 52-week seasonality without tree extrapolation decay.",
        models=benchmark_data
    )
