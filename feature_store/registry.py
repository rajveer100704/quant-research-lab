"""
Feature Registry
================
Catalog of known features with metadata for documentation and discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMeta:
    """Metadata for a single feature."""

    name: str
    category: str  # "technical", "orderbook", "sentiment"
    description: str
    dtype: str = "float64"
    source: str = ""


# --- Default Feature Catalog ---

DEFAULT_FEATURES: list[FeatureMeta] = [
    # Technical
    FeatureMeta("rsi", "technical", "Relative Strength Index (14-period)"),
    FeatureMeta("macd", "technical", "MACD line"),
    FeatureMeta("macd_signal", "technical", "MACD signal line"),
    FeatureMeta("macd_histogram", "technical", "MACD histogram"),
    FeatureMeta("sma_20", "technical", "20-period simple moving average"),
    FeatureMeta("ema_20", "technical", "20-period exponential moving average"),
    FeatureMeta("sma_50", "technical", "50-period simple moving average"),
    FeatureMeta("volatility", "technical", "Rolling return standard deviation"),
    FeatureMeta("atr", "technical", "Average True Range"),
    FeatureMeta("momentum", "technical", "Price momentum (14-period diff)"),
    FeatureMeta("roc", "technical", "Rate of change (%)"),
    FeatureMeta("stochastic_k", "technical", "Stochastic %K"),
    FeatureMeta("bollinger_high", "technical", "Bollinger Band upper"),
    FeatureMeta("bollinger_low", "technical", "Bollinger Band lower"),
    # Order book
    FeatureMeta("spread_bps", "orderbook", "Bid-ask spread in basis points"),
    FeatureMeta("order_imbalance", "orderbook", "Bid/ask volume imbalance [-1, 1]"),
    FeatureMeta("bid_depth", "orderbook", "Total bid-side liquidity depth"),
    FeatureMeta("ask_depth", "orderbook", "Total ask-side liquidity depth"),
    FeatureMeta("volume_delta", "orderbook", "Change in net order flow"),
    # Sentiment
    FeatureMeta("sentiment_compound", "sentiment", "VADER compound score [-1, 1]"),
    FeatureMeta("sentiment_pos", "sentiment", "VADER positive proportion"),
    FeatureMeta("sentiment_neg", "sentiment", "VADER negative proportion"),
]


class FeatureRegistry:
    """
    In-memory registry of feature metadata.

    Example
    -------
    >>> registry = FeatureRegistry()
    >>> registry.list_by_category("technical")
    """

    def __init__(self, features: Optional[list[FeatureMeta]] = None) -> None:
        self._features: dict[str, FeatureMeta] = {}
        for f in (features or DEFAULT_FEATURES):
            self.register(f)

    def register(self, meta: FeatureMeta) -> None:
        """Add a feature to the registry."""
        self._features[meta.name] = meta

    def get(self, name: str) -> Optional[FeatureMeta]:
        """Look up feature metadata by name."""
        return self._features.get(name)

    def list_all(self) -> list[FeatureMeta]:
        """Return all registered features."""
        return list(self._features.values())

    def list_by_category(self, category: str) -> list[FeatureMeta]:
        """Filter features by category."""
        return [f for f in self._features.values() if f.category == category]

    def summary(self) -> str:
        """Return a readable summary table."""
        lines = [f"{'Name':<25} {'Category':<12} Description"]
        lines.append("-" * 70)
        for f in self._features.values():
            lines.append(f"{f.name:<25} {f.category:<12} {f.description}")
        return "\n".join(lines)
