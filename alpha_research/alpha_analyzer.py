"""
Alpha Analyzer
==============
Measures predictive power and stability of alpha signals.

Computed metrics:
- Information Coefficient (IC) — rank correlation with forward returns
- Sharpe ratio of signal
- Win rate
- Stability — consistency of IC over rolling windows
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_research.alpha_signal import AlphaSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class AlphaAnalyzer:
    """
    Analyses the predictive quality of alpha signals.

    Parameters
    ----------
    forward_returns : Series
        Forward returns to evaluate signals against the same index.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.

    Example
    -------
    >>> analyzer = AlphaAnalyzer(forward_returns)
    >>> report = analyzer.analyze(signal)
    """

    def __init__(
        self,
        forward_returns: pd.Series,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.forward_returns = forward_returns
        self.risk_free_rate = risk_free_rate

    def information_coefficient(self, signal: AlphaSignal) -> float:
        """
        Compute Spearman rank IC between signal and forward returns.

        Returns correlation in [-1, 1].
        """
        aligned = pd.DataFrame({
            "signal": signal.values,
            "returns": self.forward_returns,
        }).dropna()

        if len(aligned) < 10:
            return 0.0

        ic, _ = stats.spearmanr(aligned["signal"], aligned["returns"])
        return float(ic) if not np.isnan(ic) else 0.0

    def sharpe_ratio(self, signal: AlphaSignal, periods_per_year: int = 252 * 24) -> float:
        """Annualised Sharpe ratio of the signal-weighted returns."""
        aligned = pd.DataFrame({
            "signal": signal.values,
            "returns": self.forward_returns,
        }).dropna()

        signal_returns = aligned["signal"] * aligned["returns"]
        if signal_returns.std() == 0:
            return 0.0

        excess = signal_returns.mean() - self.risk_free_rate / periods_per_year
        return float(excess / signal_returns.std() * np.sqrt(periods_per_year))

    def win_rate(self, signal: AlphaSignal) -> float:
        """Percentage of periods where signal direction matches return direction."""
        aligned = pd.DataFrame({
            "signal": signal.values,
            "returns": self.forward_returns,
        }).dropna()

        correct = (aligned["signal"] * aligned["returns"]) > 0
        return float(correct.mean() * 100) if len(aligned) > 0 else 0.0

    def stability(self, signal: AlphaSignal, window: int = 100) -> float:
        """
        IC stability — fraction of rolling windows where IC > 0.

        Higher = more consistent alphas.
        """
        aligned = pd.DataFrame({
            "signal": signal.values,
            "returns": self.forward_returns,
        }).dropna()

        if len(aligned) < window * 2:
            return 0.0

        rolling_ics: list[float] = []
        for i in range(0, len(aligned) - window, window // 2):
            chunk = aligned.iloc[i: i + window]
            ic, _ = stats.spearmanr(chunk["signal"], chunk["returns"])
            if not np.isnan(ic):
                rolling_ics.append(ic)

        if not rolling_ics:
            return 0.0

        positive_ratio = sum(1 for ic in rolling_ics if ic > 0) / len(rolling_ics)
        return float(positive_ratio)

    def analyze(self, signal: AlphaSignal) -> dict[str, Any]:
        """
        Full analysis report for an alpha signal.

        Returns dict with ``name, ic, sharpe, win_rate, stability, rating``.
        """
        ic = self.information_coefficient(signal)
        sharpe = self.sharpe_ratio(signal)
        wr = self.win_rate(signal)
        stab = self.stability(signal)

        # Quality rating
        score = abs(ic) * 0.3 + min(abs(sharpe), 3) / 3 * 0.3 + wr / 100 * 0.2 + stab * 0.2
        if score > 0.6:
            rating = "STRONG"
        elif score > 0.4:
            rating = "MODERATE"
        else:
            rating = "WEAK"

        report = {
            "name": signal.name,
            "ic": round(ic, 4),
            "sharpe": round(sharpe, 4),
            "win_rate": round(wr, 2),
            "stability": round(stab, 4),
            "rating": rating,
        }
        logger.info("Alpha [%s] IC=%.4f Sharpe=%.2f WR=%.1f%% → %s", signal.name, ic, sharpe, wr, rating)
        return report

    def rank_signals(self, signals: list[AlphaSignal]) -> pd.DataFrame:
        """
        Analyze and rank multiple signals by quality.

        Returns DataFrame sorted by IC descending.
        """
        reports = [self.analyze(s) for s in signals]
        df = pd.DataFrame(reports)
        df = df.sort_values("ic", ascending=False, key=abs).reset_index(drop=True)
        return df
