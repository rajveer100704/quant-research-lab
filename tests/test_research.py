"""
Tests — Alpha Research, Feature Store, and Experiments
======================================================
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile


class TestAlphaResearch:
    """Test suite for alpha signal generation and analysis."""

    @pytest.fixture
    def features_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 500
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        return pd.DataFrame({
            "close": close,
            "rsi": np.random.uniform(20, 80, n),
            "order_imbalance": np.random.uniform(-1, 1, n),
        })

    def test_momentum_alpha(self, features_df: pd.DataFrame) -> None:
        from alpha_research.alpha_signal import MomentumAlpha
        alpha = MomentumAlpha(lookback=12)
        signal = alpha.generate(features_df)
        assert signal.name == "momentum_12"
        assert len(signal.values) == len(features_df)

    def test_rsi_cross_alpha(self, features_df: pd.DataFrame) -> None:
        from alpha_research.alpha_signal import RSICrossAlpha
        alpha = RSICrossAlpha()
        signal = alpha.generate(features_df)
        unique_vals = set(signal.values.dropna().unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_alpha_analyzer(self, features_df: pd.DataFrame) -> None:
        from alpha_research.alpha_signal import MomentumAlpha
        from alpha_research.alpha_analyzer import AlphaAnalyzer

        forward_returns = features_df["close"].pct_change().shift(-1)
        analyzer = AlphaAnalyzer(forward_returns)

        signal = MomentumAlpha(12).generate(features_df)
        report = analyzer.analyze(signal)

        assert "ic" in report
        assert "sharpe" in report
        assert "win_rate" in report
        assert report["rating"] in ["STRONG", "MODERATE", "WEAK"]


class TestFeatureStore:
    """Test suite for feature store persistence."""

    def test_save_and_load(self) -> None:
        from feature_store.store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(base_dir=Path(tmpdir))
            df = pd.DataFrame({"rsi": [45.0, 55.0, 65.0], "macd": [0.1, -0.2, 0.3]})
            store.save_features(df, symbol="BTC", feature_set="technical", date="2025-01-01")
            loaded = store.load_features(symbol="BTC", feature_set="technical", date="2025-01-01")
            assert len(loaded) == 3
            assert list(loaded.columns) == ["rsi", "macd"]

    def test_list_features(self) -> None:
        from feature_store.store import FeatureStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(base_dir=Path(tmpdir))
            df = pd.DataFrame({"x": [1, 2, 3]})
            store.save_features(df, "BTC", "technical", "2025-03-01")
            store.save_features(df, "BTC", "orderbook", "2025-03-01")
            features = store.list_features("BTC")
            assert len(features) == 2


class TestFeatureRegistry:
    """Test suite for feature registry."""

    def test_default_features(self) -> None:
        from feature_store.registry import FeatureRegistry
        registry = FeatureRegistry()
        all_features = registry.list_all()
        assert len(all_features) > 10

    def test_filter_by_category(self) -> None:
        from feature_store.registry import FeatureRegistry
        registry = FeatureRegistry()
        tech = registry.list_by_category("technical")
        assert all(f.category == "technical" for f in tech)


class TestExperimentManager:
    """Test suite for experiment tracking."""

    def test_experiment_lifecycle(self) -> None:
        from experiments.experiment_manager import ExperimentManager
        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=Path(tmpdir))
            exp_id = em.start_experiment("test_exp", params={"lr": 0.001})
            em.log_metric(exp_id, "loss", 0.5)
            em.log_metric(exp_id, "loss", 0.3)
            em.end_experiment(exp_id)

            exp = em.get_experiment(exp_id)
            assert exp["name"] == "test_exp"
            assert exp["status"] == "COMPLETED"
            assert len(exp["metrics"]["loss"]) == 2
