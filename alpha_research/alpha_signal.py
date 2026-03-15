"""
Alpha Signal Framework
======================
Base classes for defining, generating, and storing alpha signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlphaSignal:
    """
    Container for a single alpha signal.

    Attributes
    ----------
    name : str
        Descriptive signal name (e.g. ``"momentum_rsi_cross"``).
    values : Series
        Signal values aligned with timestamps.
    metadata : dict
        Additional metadata (parameters, source, etc.).
    """

    name: str
    values: pd.Series
    metadata: dict = field(default_factory=dict)

    @property
    def sharpe(self) -> float:
        """Quick Sharpe ratio of the signal values."""
        if self.values.std() == 0:
            return 0.0
        return float(self.values.mean() / self.values.std() * np.sqrt(252 * 24))


class AlphaGenerator(ABC):
    """
    Abstract base class for alpha signal generators.

    Subclasses implement ``generate()`` to produce an ``AlphaSignal``
    from a feature DataFrame.

    Example
    -------
    >>> class MomentumAlpha(AlphaGenerator):
    ...     def generate(self, df):
    ...         signal = df["close"].pct_change(12)
    ...         return AlphaSignal("momentum_12h", signal)
    """

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> AlphaSignal:
        """Generate an alpha signal from features."""
        ...


# --- Built-in Alpha Generators ---


class MomentumAlpha(AlphaGenerator):
    """Price momentum over a lookback period."""

    def __init__(self, lookback: int = 12) -> None:
        self.lookback = lookback

    def generate(self, df: pd.DataFrame) -> AlphaSignal:
        signal = df["close"].pct_change(self.lookback)
        return AlphaSignal(
            name=f"momentum_{self.lookback}",
            values=signal,
            metadata={"lookback": self.lookback},
        )


class MeanReversionAlpha(AlphaGenerator):
    """Z-score of price relative to rolling mean (mean-reversion signal)."""

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate(self, df: pd.DataFrame) -> AlphaSignal:
        rolling_mean = df["close"].rolling(self.window).mean()
        rolling_std = df["close"].rolling(self.window).std()
        z_score = -(df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
        return AlphaSignal(
            name=f"mean_reversion_{self.window}",
            values=z_score,
            metadata={"window": self.window},
        )


class RSICrossAlpha(AlphaGenerator):
    """Signal based on RSI crossing overbought/oversold thresholds."""

    def __init__(self, overbought: float = 70, oversold: float = 30) -> None:
        self.overbought = overbought
        self.oversold = oversold

    def generate(self, df: pd.DataFrame) -> AlphaSignal:
        if "rsi" not in df.columns:
            raise ValueError("DataFrame must contain 'rsi' column")
        signal = pd.Series(0.0, index=df.index)
        signal[df["rsi"] < self.oversold] = 1.0   # Bullish
        signal[df["rsi"] > self.overbought] = -1.0  # Bearish
        return AlphaSignal(
            name="rsi_cross",
            values=signal,
            metadata={"overbought": self.overbought, "oversold": self.oversold},
        )


class OrderFlowAlpha(AlphaGenerator):
    """Alpha signal derived from order book imbalance."""

    def generate(self, df: pd.DataFrame) -> AlphaSignal:
        if "order_imbalance" not in df.columns:
            raise ValueError("DataFrame must contain 'order_imbalance' column")
        return AlphaSignal(
            name="order_flow",
            values=df["order_imbalance"],
            metadata={"source": "orderbook"},
        )
