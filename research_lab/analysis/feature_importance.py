"""
Feature Importance Analyzer
===========================
Evaluates which features are most predictive of forward returns using
ensemble tree models and permutation importance.

Methods
-------
- Random Forest feature importance (Gini / MDI)
- Permutation importance (model-agnostic)
- SHAP-compatible feature ranking

This is a core tool in the Research Lab for understanding
which market features drive alpha before committing to a strategy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureImportanceAnalyzer:
    """
    Compute feature importance for predicting forward returns.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the ensemble.
    method : str
        ``"random_forest"`` or ``"gradient_boosting"``.

    Example
    -------
    >>> analyzer = FeatureImportanceAnalyzer()
    >>> report = analyzer.analyze(features_df, forward_returns)
    >>> print(report[["feature", "importance"]].head(10))
    """

    def __init__(
        self,
        n_estimators: int = 200,
        method: str = "random_forest",
    ) -> None:
        self.n_estimators = n_estimators
        self.method = method

    def _build_model(self) -> Any:
        """Create the ensemble model."""
        if self.method == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                random_state=42,
            )
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    def analyze(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Compute feature importance ranking.

        Parameters
        ----------
        features : DataFrame
            Feature matrix (columns = feature names).
        forward_returns : Series
            Forward returns aligned with features.
        top_n : int
            Number of top features to return.

        Returns
        -------
        DataFrame with ``feature``, ``importance``, ``importance_pct``,
        ``cumulative_pct``, and ``rank``.
        """
        # Prepare data
        df = features.copy()
        df["target"] = (forward_returns > 0).astype(int)  # Binary: up/down
        df = df.dropna()

        X = df.drop(columns=["target"])
        y = df["target"]

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Train model
        model = self._build_model()
        model.fit(X_scaled, y)

        # MDI importance
        mdi_importance = pd.Series(model.feature_importances_, index=X.columns)

        # Permutation importance (on last 20% of data — time-aware)
        split_idx = int(len(X_scaled) * 0.8)
        X_test = X_scaled.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_importance = pd.Series(perm.importances_mean, index=X.columns)

        # Combined score (average of normalised MDI + permutation)
        mdi_norm = mdi_importance / mdi_importance.sum()
        perm_norm = perm_importance / (perm_importance.sum() + 1e-8)
        combined = (mdi_norm + perm_norm) / 2

        # Build report
        report = pd.DataFrame({
            "feature": combined.index,
            "importance": combined.values,
            "mdi_importance": mdi_importance.values,
            "perm_importance": perm_importance.values,
        })
        report = report.sort_values("importance", ascending=False).head(top_n)
        report["importance_pct"] = (report["importance"] / report["importance"].sum() * 100).round(2)
        report["cumulative_pct"] = report["importance_pct"].cumsum().round(2)
        report["rank"] = range(1, len(report) + 1)
        report = report.reset_index(drop=True)

        logger.info("Feature importance computed: top feature = %s (%.2f%%)",
                     report.iloc[0]["feature"], report.iloc[0]["importance_pct"])
        return report

    def cross_validate_importance(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Compute feature importance stability across time-series CV folds.

        Returns DataFrame with ``feature``, ``mean_importance``, ``std_importance``,
        ``stability_score`` (mean/std — higher = more stable).
        """
        df = features.copy()
        df["target"] = (forward_returns > 0).astype(int)
        df = df.dropna()

        X = df.drop(columns=["target"])
        y = df["target"]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_importances: list[pd.Series] = []

        for train_idx, test_idx in tscv.split(X):
            model = self._build_model()
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train_idx])
            model.fit(X_train, y.iloc[train_idx])
            imp = pd.Series(model.feature_importances_, index=X.columns)
            all_importances.append(imp)

        imp_df = pd.DataFrame(all_importances)
        result = pd.DataFrame({
            "feature": X.columns,
            "mean_importance": imp_df.mean().values,
            "std_importance": imp_df.std().values,
        })
        result["stability_score"] = (
            result["mean_importance"] / (result["std_importance"] + 1e-8)
        ).round(4)
        result = result.sort_values("mean_importance", ascending=False).reset_index(drop=True)

        logger.info("CV importance: %d folds, most stable = %s (score=%.2f)",
                     n_splits, result.iloc[0]["feature"], result.iloc[0]["stability_score"])
        return result
