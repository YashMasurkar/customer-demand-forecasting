"""Machine Learning forecasting module for weekly customer demand.

This module implements:
1. Chronological expanding-window Cross-Validation (TimeSeriesSplit) on the training set.
2. Training and evaluation of Ridge, Random Forest, and Gradient Boosting models.
3. Feature importance analysis and model selection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation import calculate_forecast_metrics, ForecastMetrics


@dataclass
class MLCVResult:
    """Container for Cross-Validation performance on training data."""
    model_name: str
    cv_maes: List[float]
    cv_rmses: List[float]
    mean_cv_mae: float
    mean_cv_rmse: float
    std_cv_mae: float
    std_cv_rmse: float


@dataclass
class FeatureImportanceRecord:
    """Container for individual feature importance metrics."""
    feature_name: str
    importance_value: float
    rank: int


class MLDemandForecaster:
    """Machine Learning Demand Forecaster managing training, CV, prediction, and feature importance."""

    def __init__(
        self,
        model_type: str = "ridge",
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """Initialize ML forecaster.

        Args:
            model_type: 'ridge', 'random_forest', 'gradient_boosting', or 'hist_gradient_boosting'.
            model_params: Dictionary of model hyperparameters.
            random_state: Seed for reproducibility.
        """
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.random_state = random_state

        self.estimator_ = self._create_estimator()
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _create_estimator(self) -> Any:
        """Create sklearn model pipeline or estimator based on model_type."""
        if self.model_type == "ridge":
            alpha = self.model_params.get("alpha", 100.0)
            return Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha, random_state=self.random_state))
            ])
        elif self.model_type == "linear":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ])
        elif self.model_type == "random_forest":
            n_estimators = self.model_params.get("n_estimators", 100)
            max_depth = self.model_params.get("max_depth", 5)
            min_samples_leaf = self.model_params.get("min_samples_leaf", 2)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state
            )
        elif self.model_type == "gradient_boosting":
            n_estimators = self.model_params.get("n_estimators", 100)
            learning_rate = self.model_params.get("learning_rate", 0.05)
            max_depth = self.model_params.get("max_depth", 3)
            min_samples_leaf = self.model_params.get("min_samples_leaf", 3)
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state
            )
        elif self.model_type == "hist_gradient_boosting":
            learning_rate = self.model_params.get("learning_rate", 0.05)
            max_iter = self.model_params.get("max_iter", 100)
            min_samples_leaf = self.model_params.get("min_samples_leaf", 3)
            return HistGradientBoostingRegressor(
                learning_rate=learning_rate,
                max_iter=max_iter,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray | pd.Series) -> "MLDemandForecaster":
        """Fit ML estimator on training features and target.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training target demand array.

        Returns:
            self
        """
        self.feature_names_ = list(X_train.columns)
        self.estimator_.fit(X_train, np.asarray(y_train, dtype=float))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for given feature DataFrame.

        Args:
            X: Feature DataFrame matching training features.

        Returns:
            1D array of predicted demand values (clipped to non-negative).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        raw_preds = self.estimator_.predict(X)
        return np.maximum(0.0, np.asarray(raw_preds, dtype=float))

    def get_feature_importances(self) -> List[FeatureImportanceRecord]:
        """Extract and rank feature importances or coefficients.

        Returns:
            List of FeatureImportanceRecord sorted by absolute importance descending.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extracting feature importances.")

        if hasattr(self.estimator_, "named_steps"):
            # Pipeline (e.g. Ridge / Linear)
            reg = self.estimator_.named_steps["regressor"]
            if hasattr(reg, "coef_"):
                raw_imp = np.abs(reg.coef_)
            else:
                raw_imp = np.zeros(len(self.feature_names_))
        elif hasattr(self.estimator_, "feature_importances_"):
            raw_imp = self.estimator_.feature_importances_
        else:
            raw_imp = np.zeros(len(self.feature_names_))

        records = [
            FeatureImportanceRecord(feature_name=f_name, importance_value=round(float(imp), 5), rank=0)
            for f_name, imp in zip(self.feature_names_, raw_imp)
        ]
        records.sort(key=lambda r: r.importance_value, reverse=True)
        for idx, r in enumerate(records, start=1):
            r.rank = idx
        return records


def run_time_series_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    candidate_models: Dict[str, Tuple[str, Dict[str, Any]]],
    n_splits: int = 4,
    random_state: int = 42
) -> List[MLCVResult]:
    """Perform chronological TimeSeriesSplit cross-validation strictly on training data.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target array.
        candidate_models: Dict of model_name -> (model_type, model_params).
        n_splits: Number of TimeSeriesSplit folds.
        random_state: Seed for reproducibility.

    Returns:
        List of MLCVResult sorted by mean_cv_mae ascending.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_arr = np.asarray(y_train, dtype=float)
    cv_results: List[MLCVResult] = []

    for name, (m_type, m_params) in candidate_models.items():
        fold_maes = []
        fold_rmses = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_arr[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_arr[val_idx]

            forecaster = MLDemandForecaster(model_type=m_type, model_params=m_params, random_state=random_state)
            forecaster.fit(X_tr, y_tr)
            preds = forecaster.predict(X_val)

            fold_maes.append(float(np.mean(np.abs(y_val - preds))))
            fold_rmses.append(float(np.sqrt(np.mean((y_val - preds) ** 2))))

        res = MLCVResult(
            model_name=name,
            cv_maes=[round(m, 2) for m in fold_maes],
            cv_rmses=[round(r, 2) for r in fold_rmses],
            mean_cv_mae=round(float(np.mean(fold_maes)), 2),
            mean_cv_rmse=round(float(np.mean(fold_rmses)), 2),
            std_cv_mae=round(float(np.std(fold_maes, ddof=1)), 2),
            std_cv_rmse=round(float(np.std(fold_rmses, ddof=1)), 2)
        )
        cv_results.append(res)

    cv_results.sort(key=lambda r: r.mean_cv_mae)
    return cv_results
