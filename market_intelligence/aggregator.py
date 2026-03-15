"""
Market Intelligence Aggregator
==============================
Combines technical, order book, and sentiment signals into a unified market state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketState:
    """Unified market state snapshot."""

    timestamp: str
    symbol: str

    # Technical
    rsi: float = 0.0
    macd_histogram: float = 0.0
    volatility: float = 0.0
    trend_score: float = 0.0  # [-1, 1] based on MA alignment

    # Order book
    order_imbalance: float = 0.0
    spread_bps: float = 0.0
    depth_ratio: float = 0.0  # bid_depth / ask_depth

    # Sentiment
    sentiment_score: float = 0.0

    # Composite
    composite_score: float = 0.0

    regime: str = "UNKNOWN"


class MarketIntelligence:
    """
    Aggregates signals from multiple data sources into a ``MarketState``.

    Parameters
    ----------
    technical_df : DataFrame
        Output from ``TechnicalIndicators.compute_all()``.
    orderbook_df : DataFrame
        Output from ``OrderBookFeatures.compute_all()``.
    sentiment_df : DataFrame
        Output from ``SentimentAnalyzer.score_articles()``.

    Example
    -------
    >>> mi = MarketIntelligence(tech_df, ob_df, sent_df, symbol="BTCUSDT")
    >>> state = mi.get_latest_state()
    """

    def __init__(
        self,
        technical_df: Optional[pd.DataFrame] = None,
        orderbook_df: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSDT",
    ) -> None:
        self.technical_df = technical_df if technical_df is not None else pd.DataFrame()
        self.orderbook_df = orderbook_df if orderbook_df is not None else pd.DataFrame()
        self.sentiment_df = sentiment_df if sentiment_df is not None else pd.DataFrame()
        self.symbol = symbol

    def _compute_trend_score(self, row: pd.Series) -> float:
        """
        Score trend direction based on MA alignment.

        Returns value in [-1, 1]: positive = bullish.
        """
        score = 0.0
        close = row.get("close", 0)
        if close > 0:
            for ma in ["sma_20", "sma_50", "ema_20"]:
                if ma in row and row[ma] > 0:
                    score += 1.0 if close > row[ma] else -1.0
        return score / 3.0  # Normalise

    def get_latest_state(self) -> MarketState:
        """Build the latest market state from available data."""
        state = MarketState(
            timestamp=datetime.utcnow().isoformat(),
            symbol=self.symbol,
        )

        # Technical signals
        if not self.technical_df.empty:
            latest = self.technical_df.iloc[-1]
            state.rsi = float(latest.get("rsi", 0))
            state.macd_histogram = float(latest.get("macd_histogram", 0))
            state.volatility = float(latest.get("volatility", 0))
            state.trend_score = self._compute_trend_score(latest)

        # Order book signals
        if not self.orderbook_df.empty:
            latest_ob = self.orderbook_df.iloc[-1]
            state.order_imbalance = float(latest_ob.get("order_imbalance", 0))
            state.spread_bps = float(latest_ob.get("spread_bps", 0))
            bid = float(latest_ob.get("bid_depth", 1))
            ask = float(latest_ob.get("ask_depth", 1))
            state.depth_ratio = bid / ask if ask > 0 else 1.0

        # Sentiment
        if not self.sentiment_df.empty:
            state.sentiment_score = float(
                self.sentiment_df["sentiment_compound"].mean()
            )

        # Composite score: weighted combination
        state.composite_score = (
            0.30 * state.trend_score
            + 0.25 * state.order_imbalance
            + 0.20 * state.sentiment_score
            + 0.15 * (1.0 if state.rsi < 30 else (-1.0 if state.rsi > 70 else 0.0))
            + 0.10 * state.macd_histogram / max(abs(state.macd_histogram), 1e-8)
        )

        logger.info(
            "MarketState [%s] composite=%.3f regime=%s",
            self.symbol, state.composite_score, state.regime,
        )
        return state
