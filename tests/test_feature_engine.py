"""
Tests — Feature Engine
======================
Validates technical indicators, order book features, and sentiment scoring.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 10,
        "high": close + abs(np.random.randn(n) * 50),
        "low": close - abs(np.random.randn(n) * 50),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    })


@pytest.fixture
def sample_orderbook_snapshots() -> list[dict]:
    """Generate synthetic order book snapshots."""
    snapshots = []
    for i in range(10):
        price = 50000 + i * 10
        snapshots.append({
            "timestamp": f"2025-01-01T{i:02d}:00:00",
            "bids": [[str(price - j * 5), str(np.random.uniform(0.1, 5))] for j in range(10)],
            "asks": [[str(price + j * 5), str(np.random.uniform(0.1, 5))] for j in range(10)],
        })
    return snapshots


class TestTechnicalIndicators:
    """Test suite for technical indicators computation."""

    def test_compute_rsi(self, sample_ohlcv: pd.DataFrame) -> None:
        from feature_engine.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.compute_rsi(sample_ohlcv)
        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        valid = result["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_compute_macd(self, sample_ohlcv: pd.DataFrame) -> None:
        from feature_engine.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.compute_macd(sample_ohlcv)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_compute_moving_averages(self, sample_ohlcv: pd.DataFrame) -> None:
        from feature_engine.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.compute_moving_averages(sample_ohlcv, windows=[7, 20])
        assert "sma_7" in result.columns
        assert "ema_20" in result.columns

    def test_compute_volatility(self, sample_ohlcv: pd.DataFrame) -> None:
        from feature_engine.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.compute_volatility(sample_ohlcv)
        assert "volatility" in result.columns
        assert "atr" in result.columns
        assert "bollinger_high" in result.columns

    def test_compute_all(self, sample_ohlcv: pd.DataFrame) -> None:
        from feature_engine.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        expected_cols = ["rsi", "macd", "volatility", "momentum"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestOrderBookFeatures:
    """Test suite for order book microstructure features."""

    def test_bid_ask_spread(self, sample_orderbook_snapshots: list[dict]) -> None:
        from feature_engine.orderbook_features import OrderBookFeatures
        obf = OrderBookFeatures()
        result = obf.compute_bid_ask_spread(sample_orderbook_snapshots[0])
        assert "spread" in result
        assert result["spread"] >= 0

    def test_order_imbalance(self, sample_orderbook_snapshots: list[dict]) -> None:
        from feature_engine.orderbook_features import OrderBookFeatures
        obf = OrderBookFeatures()
        imbalance = obf.compute_order_imbalance(sample_orderbook_snapshots[0])
        assert -1.0 <= imbalance <= 1.0

    def test_compute_depth(self, sample_orderbook_snapshots: list[dict]) -> None:
        from feature_engine.orderbook_features import OrderBookFeatures
        obf = OrderBookFeatures()
        depth = obf.compute_depth(sample_orderbook_snapshots[0])
        assert depth["total_depth"] >= 0

    def test_compute_all(self, sample_orderbook_snapshots: list[dict]) -> None:
        from feature_engine.orderbook_features import OrderBookFeatures
        obf = OrderBookFeatures()
        df = obf.compute_all(sample_orderbook_snapshots)
        assert len(df) == len(sample_orderbook_snapshots)
        assert "order_imbalance" in df.columns


class TestSentimentAnalyzer:
    """Test suite for sentiment analysis."""

    def test_score_text(self) -> None:
        from feature_engine.sentiment_features import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        # Use strongly positive words that VADER definitively scores
        result = analyzer.score_text("This is great, amazing, and excellent news!")
        assert "compound" in result
        assert result["compound"] > 0  # Positive sentiment

        # Negative text should score negative
        result_neg = analyzer.score_text("Terrible crash, awful losses, devastating collapse.")
        assert result_neg["compound"] < 0

    def test_score_articles(self) -> None:
        from feature_engine.sentiment_features import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        articles = [
            {"title": "Bitcoin crashes 20%", "published_on": "2025-01-01T00:00:00"},
            {"title": "ETH reaches record volume", "published_on": "2025-01-01T01:00:00"},
        ]
        df = analyzer.score_articles(articles)
        assert len(df) == 2
        assert "sentiment_compound" in df.columns
