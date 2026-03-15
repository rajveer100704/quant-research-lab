"""
Technical Indicators
====================
Computes standard technical analysis indicators from OHLCV data.
All functions accept and return pandas DataFrames to maintain pipeline composability.
"""

from __future__ import annotations

import pandas as pd
import ta

from utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Compute technical indicators from OHLCV DataFrames.

    The input DataFrame must contain columns:
    ``open, high, low, close, volume``.

    Example
    -------
    >>> ti = TechnicalIndicators()
    >>> df = ti.compute_all(ohlcv_df)
    """

    @staticmethod
    def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        df = df.copy()
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=window).rsi()
        return df

    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
    ) -> pd.DataFrame:
        """Add MACD, MACD signal, and MACD histogram."""
        df = df.copy()
        macd = ta.trend.MACD(
            close=df["close"],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        return df

    @staticmethod
    def compute_moving_averages(
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add simple and exponential moving averages."""
        df = df.copy()
        windows = windows or [7, 20, 50, 200]
        for w in windows:
            df[f"sma_{w}"] = df["close"].rolling(window=w).mean()
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        return df

    @staticmethod
    def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add rolling volatility (standard deviation of returns)."""
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=window).std()
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).average_true_range()
        df["bollinger_high"] = ta.volatility.BollingerBands(
            close=df["close"], window=window
        ).bollinger_hband()
        df["bollinger_low"] = ta.volatility.BollingerBands(
            close=df["close"], window=window
        ).bollinger_lband()
        return df

    @staticmethod
    def compute_momentum(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add momentum and rate-of-change indicators."""
        df = df.copy()
        df["momentum"] = df["close"].diff(window)
        df["roc"] = df["close"].pct_change(periods=window) * 100
        df["stochastic_k"] = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).stoch()
        return df

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all technical indicators to the DataFrame."""
        logger.info("Computing all technical indicators (%d rows)", len(df))
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_moving_averages(df)
        df = self.compute_volatility(df)
        df = self.compute_momentum(df)
        logger.info("Technical indicators computed: %d features", len(df.columns))
        return df
