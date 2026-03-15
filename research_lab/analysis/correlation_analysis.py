"""
Correlation Analysis
====================
Analyses feature correlations, identifies redundant features, and studies
the relationship between features and forward returns.

Tools for:
- Feature-to-return correlation matrix
- Inter-feature correlation clustering
- Signal decay analysis (how quickly a signal loses predictive power)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """
    Analyse correlations between features and returns.

    Example
    -------
    >>> ca = CorrelationAnalyzer()
    >>> decay = ca.signal_decay(features["rsi"], returns_df, max_lag=24)
    """

    @staticmethod
    def feature_return_correlation(
        features: pd.DataFrame,
        forward_returns: pd.Series,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute correlation between each feature and forward returns.

        Parameters
        ----------
        method : str
            ``"spearman"`` (rank), ``"pearson"`` (linear), or ``"kendall"``.

        Returns
        -------
        DataFrame with ``feature``, ``correlation``, ``p_value``,
        ``abs_correlation``, sorted by absolute correlation.
        """
        results: list[dict] = []
        for col in features.columns:
            aligned = pd.DataFrame({
                "feature": features[col],
                "returns": forward_returns,
            }).dropna()

            if len(aligned) < 20:
                continue

            if method == "spearman":
                corr, pval = stats.spearmanr(aligned["feature"], aligned["returns"])
            elif method == "kendall":
                corr, pval = stats.kendalltau(aligned["feature"], aligned["returns"])
            else:
                corr, pval = stats.pearsonr(aligned["feature"], aligned["returns"])

            results.append({
                "feature": col,
                "correlation": round(float(corr), 6) if not np.isnan(corr) else 0.0,
                "p_value": round(float(pval), 6) if not np.isnan(pval) else 1.0,
            })

        df = pd.DataFrame(results)
        df["abs_correlation"] = df["correlation"].abs()
        df["significant"] = df["p_value"] < 0.05
        df = df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

        logger.info("Correlation analysis: %d features, %d significant (p<0.05)",
                     len(df), df["significant"].sum())
        return df

    @staticmethod
    def inter_feature_correlation(
        features: pd.DataFrame,
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """
        Identify highly correlated feature pairs (potential redundancy).

        Parameters
        ----------
        threshold : float
            Absolute correlation threshold for flagging.

        Returns
        -------
        DataFrame of correlated pairs above threshold.
        """
        corr_matrix = features.corr(method="spearman")
        pairs: list[dict] = []

        for i, col_a in enumerate(corr_matrix.columns):
            for j, col_b in enumerate(corr_matrix.columns):
                if i >= j:
                    continue
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    pairs.append({
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(corr_val, 4),
                        "recommendation": "DROP_ONE",
                    })

        result = pd.DataFrame(pairs)
        if not result.empty:
            result = result.sort_values("correlation", ascending=False, key=abs)
        logger.info("Found %d highly correlated pairs (|r| >= %.2f)", len(result), threshold)
        return result.reset_index(drop=True)

    @staticmethod
    def signal_decay(
        signal: pd.Series,
        returns: pd.Series,
        max_lag: int = 24,
    ) -> pd.DataFrame:
        """
        Measure how quickly a signal's predictive power decays over time.

        Computes Spearman IC at increasing lags (1, 2, ..., max_lag).

        Parameters
        ----------
        signal : Series
            Alpha signal values.
        returns : Series
            Raw returns (not shifted — shifting is done here).
        max_lag : int
            Maximum forward periods to test.

        Returns
        -------
        DataFrame with ``lag``, ``ic``, ``p_value``, ``significant``.
        Higher IC at longer lags = slower decay = more persistent alpha.
        """
        decay_data: list[dict] = []

        for lag in range(1, max_lag + 1):
            fwd = returns.shift(-lag)
            aligned = pd.DataFrame({"signal": signal, "returns": fwd}).dropna()

            if len(aligned) < 30:
                continue

            ic, pval = stats.spearmanr(aligned["signal"], aligned["returns"])
            decay_data.append({
                "lag": lag,
                "ic": round(float(ic), 6) if not np.isnan(ic) else 0.0,
                "p_value": round(float(pval), 6) if not np.isnan(pval) else 1.0,
                "significant": pval < 0.05 if not np.isnan(pval) else False,
            })

        df = pd.DataFrame(decay_data)
        if not df.empty:
            half_life_rows = df[df["ic"].abs() < df["ic"].abs().iloc[0] / 2]
            half_life = int(half_life_rows.iloc[0]["lag"]) if not half_life_rows.empty else max_lag
            logger.info("Signal decay: IC at lag=1: %.4f, half-life: %d periods",
                         df.iloc[0]["ic"], half_life)
        return df
