"""
Tests — Research Lab
====================
Validates feature importance, correlation analysis, and experiment templates.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from research_lab.analysis.feature_importance import FeatureImportanceAnalyzer
from research_lab.analysis.correlation_analysis import CorrelationAnalyzer
from research_lab.experiments.experiment_template import MomentumExperiment
from research_lab.reports.alpha_report_generator import AlphaReportGenerator


@pytest.fixture
def research_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic features and forward returns for research testing."""
    np.random.seed(42)
    n = 200
    features = pd.DataFrame({
        "rsi": np.random.uniform(20, 80, n),
        "macd": np.random.randn(n),
        "vol": np.random.uniform(0.01, 0.05, n),
        "close": 100 + np.cumsum(np.random.randn(n)),
    })
    # Make RSI slightly predictive
    forward_returns = 0.1 * (features["rsi"] - 50) / 50 + np.random.randn(n) * 0.01
    return features, forward_returns


class TestFeatureImportance:
    def test_analyze(self, research_data):
        features, returns = research_data
        analyzer = FeatureImportanceAnalyzer(n_estimators=50)
        report = analyzer.analyze(features, returns, top_n=3)
        assert len(report) <= 3
        assert "feature" in report.columns
        assert "importance" in report.columns

    def test_cross_validate(self, research_data):
        features, returns = research_data
        analyzer = FeatureImportanceAnalyzer(n_estimators=10)
        result = analyzer.cross_validate_importance(features, returns, n_splits=2)
        assert "stability_score" in result.columns
        assert not result.empty


class TestCorrelationAnalysis:
    def test_feature_return_correlation(self, research_data):
        features, returns = research_data
        ca = CorrelationAnalyzer()
        df = ca.feature_return_correlation(features, returns)
        assert "correlation" in df.columns
        assert "p_value" in df.columns

    def test_inter_feature_correlation(self, research_data):
        features, _ = research_data
        # Create a highly correlated feature
        features["rsi_2"] = features["rsi"] + np.random.randn(len(features)) * 0.001
        ca = CorrelationAnalyzer()
        pairs = ca.inter_feature_correlation(features, threshold=0.9)
        assert not pairs.empty
        assert "rsi" in pairs.values or "rsi_2" in pairs.values

    def test_signal_decay(self, research_data):
        features, returns = research_data
        ca = CorrelationAnalyzer()
        decay = ca.signal_decay(features["rsi"], returns, max_lag=5)
        assert len(decay) == 5
        assert "ic" in decay.columns


class TestResearchExperiment:
    def test_momentum_experiment(self, research_data):
        features, returns = research_data
        exp = MomentumExperiment(lookback=10)
        result = exp.execute(features, returns)
        assert result.hypothesis is not None
        assert isinstance(result.accepted, bool)
        assert "ic" in result.metrics


class TestAlphaReportGenerator:
    def test_generate_and_save(self, research_data):
        features, returns = research_data
        from alpha_research.alpha_signal import AlphaSignal
        signal = AlphaSignal(name="test_signal", values=features["rsi"], metadata={})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = AlphaReportGenerator(output_dir=Path(tmpdir))
            report_md = gen.generate(signal, returns, features)
            assert "Alpha Signal Report" in report_md
            
            path = gen.save(report_md, "test_report.md")
            assert path.exists()
            assert path.read_text().startswith("# Alpha Signal Report")
