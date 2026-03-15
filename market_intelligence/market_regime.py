"""
Market Regime Detector
======================
Classifies market conditions into regimes using rolling statistics.

Regimes
-------
- ``TRENDING_UP`` — Sustained upward price movement
- ``TRENDING_DOWN`` — Sustained downward price movement
- ``RANGING`` — Low-volatility sideways market
- ``HIGH_VOLATILITY`` — Elevated volatility without clear direction
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class Regime(str, Enum):
    """Market regime labels."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class MarketRegimeDetector:
    """
    Classifies market regime from OHLCV data using rolling statistics.

    Parameters
    ----------
    trend_window : int
        Window for trend detection (SMA slope).
    vol_window : int
        Window for volatility measurement.
    vol_threshold : float
        Annualised volatility threshold for HIGH_VOLATILITY.
    trend_threshold : float
        Minimum SMA slope (per-bar %) for a trend classification.

    Example
    -------
    >>> detector = MarketRegimeDetector()
    >>> regime = detector.detect(ohlcv_df)
    """

    def __init__(
        self,
        trend_window: int = 50,
        vol_window: int = 20,
        vol_threshold: float = 0.6,
        trend_threshold: float = 0.001,
    ) -> None:
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.trend_threshold = trend_threshold

    def detect(self, df: pd.DataFrame) -> Regime:
        """
        Classify the current market regime.

        Parameters
        ----------
        df : DataFrame
            Must contain a ``close`` column with at least ``trend_window`` rows.
        """
        if len(df) < self.trend_window:
            return Regime.UNKNOWN

        close = df["close"].astype(float)

        # Rolling volatility (annualised assuming hourly data)
        returns = close.pct_change().dropna()
        rolling_vol = returns.rolling(self.vol_window).std().iloc[-1]
        annualised_vol = rolling_vol * np.sqrt(365 * 24) if not np.isnan(rolling_vol) else 0.0

        # SMA slope as trend indicator
        sma = close.rolling(self.trend_window).mean()
        if len(sma.dropna()) < 2:
            return Regime.UNKNOWN
        sma_slope = (sma.iloc[-1] - sma.iloc[-2]) / sma.iloc[-2] if sma.iloc[-2] != 0 else 0.0

        # Classification logic
        if annualised_vol > self.vol_threshold:
            regime = Regime.HIGH_VOLATILITY
        elif sma_slope > self.trend_threshold:
            regime = Regime.TRENDING_UP
        elif sma_slope < -self.trend_threshold:
            regime = Regime.TRENDING_DOWN
        else:
            regime = Regime.RANGING

        logger.info(
            "Regime: %s (vol=%.3f, sma_slope=%.5f)",
            regime.value, annualised_vol, sma_slope,
        )
        return regime

    def detect_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute regime labels for each row using expanding windows.

        Returns a Series of ``Regime`` values aligned with the DataFrame index.
        """
        regimes = []
        for i in range(len(df)):
            if i < self.trend_window:
                regimes.append(Regime.UNKNOWN.value)
            else:
                window_df = df.iloc[max(0, i - self.trend_window * 2):i + 1]
                regimes.append(self.detect(window_df).value)
        return pd.Series(regimes, index=df.index, name="regime")
