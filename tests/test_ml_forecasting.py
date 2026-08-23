"""Unit tests for ML forecasting pipeline, Cross-Validation, and Ridge coefficient analysis."""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from src.features.feature_engineering import build_weekly_feature_dataset
from src.models.ml_forecasting import (
    MLDemandForecaster,
    run_time_series_cv,
    MLCVResult,
    FeatureImportanceRecord
)


@pytest.fixture
def sample_ml_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Generate synthetic train/test feature matrices."""
    np.random.seed(42)
    n_train = 80
    n_test = 20

    dates_train = pd.date_range("2014-01-06", periods=n_train, freq="W-MON")
    dates_test = pd.date_range("2015-08-01", periods=n_test, freq="W-MON")

    cols = ["trend_index", "month", "lag_1", "lag_2", "rolling_mean_4", "lag_1_sales"]
    X_train = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
    y_train = np.random.uniform(50, 300, size=n_train)

    X_test = pd.DataFrame(np.random.randn(n_test, len(cols)), columns=cols)
    y_test = np.random.uniform(60, 320, size=n_test)

    return X_train, y_train, X_test, y_test


def test_ml_forecaster_initialization_defaults():
    """Verify default initialization and unfitted error handling."""
    forecaster = MLDemandForecaster(model_type="ridge", model_params={"alpha": 10.0})
    assert forecaster.model_type == "ridge"
    assert forecaster.is_fitted is False

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        forecaster.predict(pd.DataFrame([[1, 2, 3]]))

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        forecaster.get_feature_importances()


def test_ml_forecaster_fit_and_predict_length(sample_ml_data):
    """Verify fit and prediction outputs for Ridge, Random Forest, and Gradient Boosting."""
    X_train, y_train, X_test, y_test = sample_ml_data

    for m_type in ["ridge", "random_forest", "gradient_boosting", "hist_gradient_boosting"]:
        forecaster = MLDemandForecaster(model_type=m_type, random_state=42)
        forecaster.fit(X_train, y_train)

        assert forecaster.is_fitted is True
        preds = forecaster.predict(X_test)

        assert len(preds) == len(X_test)
        assert np.all(preds >= 0.0)  # Non-negative clipping


def test_ridge_feature_coefficient_analysis(sample_ml_data):
    """Verify Ridge standardized coefficient extraction and ranking."""
    X_train, y_train, _, _ = sample_ml_data

    ridge = MLDemandForecaster(model_type="ridge", model_params={"alpha": 10.0}).fit(X_train, y_train)
    coef_records = ridge.get_ridge_coefficients()

    assert len(coef_records) == X_train.shape[1]
    assert all(isinstance(r, FeatureImportanceRecord) for r in coef_records)
    # Ranked descending
    assert coef_records[0].importance_value >= coef_records[-1].importance_value
    assert coef_records[0].rank == 1


def test_time_series_cv_chronological_ordering(sample_ml_data):
    """Verify TimeSeriesSplit cross validation runs and sorts models by mean CV MAE."""
    X_train, y_train, _, _ = sample_ml_data

    candidate_models = {
        "Ridge (alpha=100.0)": ("ridge", {"alpha": 100.0}),
        "Random Forest": ("random_forest", {"n_estimators": 20, "max_depth": 3}),
    }

    cv_results = run_time_series_cv(X_train, y_train, candidate_models, n_splits=3, random_state=42)

    assert len(cv_results) == 2
    assert all(isinstance(r, MLCVResult) for r in cv_results)
    assert cv_results[0].mean_cv_mae <= cv_results[1].mean_cv_mae
    assert len(cv_results[0].cv_maes) == 3


def test_real_data_ml_pipeline_integration():
    """Integration test: Verify complete feature extraction and ML pipeline on real data."""
    processed_path = Path("data/processed/weekly_demand.csv")
    if not processed_path.exists():
        pytest.skip("Processed dataset not found.")

    weekly_df = pd.read_csv(processed_path)
    feat_df, feature_cols = build_weekly_feature_dataset(weekly_df)

    train_mask = (feat_df["week_dt"].dt.year < 2017) & (~feat_df["is_partial_week"]) & (~feat_df["lag_52"].isna())
    test_mask = (feat_df["week_dt"].dt.year == 2017) & (~feat_df["is_partial_week"])

    X_train = feat_df.loc[train_mask, feature_cols]
    y_train = feat_df.loc[train_mask, "target_quantity"].values
    X_test = feat_df.loc[test_mask, feature_cols]
    y_test = feat_df.loc[test_mask, "target_quantity"].values

    forecaster = MLDemandForecaster(model_type="ridge", model_params={"alpha": 100.0})
    forecaster.fit(X_train, y_train)
    preds = forecaster.predict(X_test)

    assert len(preds) == 52
    mae = np.mean(np.abs(y_test - preds))
    assert mae < 60.0  # Ridge should achieve MAE ~50.02
