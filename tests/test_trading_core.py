"""
Tests — Risk Management, Simulator, Signal Fusion, and RL Environment
=====================================================================
"""

import numpy as np
import pandas as pd
import pytest


class TestRiskManager:
    """Test suite for risk management controls."""

    def test_kelly_criterion(self) -> None:
        from risk_management.risk_manager import RiskManager
        rm = RiskManager(portfolio_value=10_000)
        size = rm.kelly_criterion(win_rate=0.6, avg_win=0.03, avg_loss=0.02)
        assert 0 <= size <= rm.max_position_pct

    def test_stop_loss(self) -> None:
        from risk_management.risk_manager import RiskManager
        rm = RiskManager(portfolio_value=10_000)
        sl = rm.compute_stop_loss(entry_price=50_000, side="BUY")
        assert sl < 50_000

    def test_circuit_breaker(self) -> None:
        from risk_management.risk_manager import RiskManager
        rm = RiskManager(portfolio_value=10_000, max_drawdown_pct=0.10)
        rm.peak_value = 12_000
        rm.portfolio_value = 10_000  # 16.7% drawdown > 10% limit
        decision = rm.check_risk(entry_price=50_000)
        assert not decision.approved
        assert "drawdown" in decision.reason.lower()

    def test_risk_approved(self) -> None:
        from risk_management.risk_manager import RiskManager
        rm = RiskManager(portfolio_value=10_000)
        decision = rm.check_risk(entry_price=50_000, signal_strength=0.7)
        assert decision.approved
        assert decision.position_size > 0


class TestMarketSimulator:
    """Test suite for the market simulator."""

    @pytest.fixture
    def sim_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 100
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        return pd.DataFrame({
            "open": close - 5,
            "high": close + 50,
            "low": close - 50,
            "close": close,
            "volume": np.random.uniform(100, 1000, n),
        })

    def test_simulation_runs(self, sim_data: pd.DataFrame) -> None:
        from market_simulator.simulator import MarketSimulator
        sim = MarketSimulator(sim_data, initial_balance=10_000)
        signals = pd.Series(["HOLD"] * len(sim_data))
        signals.iloc[10] = "BUY"
        signals.iloc[50] = "SELL"
        sim.run(signals)
        results = sim.get_results()
        assert "equity_curve" in results
        assert len(results["equity_curve"]) == len(sim_data)

    def test_slippage_applied(self, sim_data: pd.DataFrame) -> None:
        from market_simulator.simulator import MarketSimulator
        sim = MarketSimulator(sim_data, slippage_bps=10.0)
        signals = pd.Series(["HOLD"] * len(sim_data))
        signals.iloc[5] = "BUY"
        sim.run(signals)
        if sim.trade_log:
            assert sim.trade_log[0]["slippage"] > 0


class TestSignalFusion:
    """Test suite for signal fusion engine."""

    def test_weighted_average(self) -> None:
        from signal_fusion.fusion_engine import SignalFusion, TradeDecision
        fusion = SignalFusion(weights={"a": 0.5, "b": 0.5})
        decision, score = fusion.weighted_average({"a": 0.8, "b": 0.6})
        assert decision == TradeDecision.BUY
        assert score > 0

    def test_majority_vote(self) -> None:
        from signal_fusion.fusion_engine import SignalFusion, TradeDecision
        decision, confidence = SignalFusion.majority_vote({"a": 1.0, "b": -1.0, "c": 1.0})
        assert decision == TradeDecision.BUY
        assert confidence > 0.5


class TestTradingEnv:
    """Test suite for the RL trading environment."""

    @pytest.fixture
    def env_data(self) -> np.ndarray:
        np.random.seed(42)
        n = 200
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        volume = np.random.uniform(100, 1000, n)
        rsi = np.random.uniform(20, 80, n)
        return np.column_stack([close, volume, rsi])

    def test_env_reset(self, env_data: np.ndarray) -> None:
        from reinforcement_learning.environment import TradingEnv
        env = TradingEnv(env_data)
        obs, info = env.reset()
        assert obs.shape[0] == env_data.shape[1] + 3  # features + position state

    def test_env_step(self, env_data: np.ndarray) -> None:
        from reinforcement_learning.environment import TradingEnv
        env = TradingEnv(env_data)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # BUY
        assert "portfolio_value" in info
        assert not terminated  # Should not be done after 1 step

    def test_buy_sell_cycle(self, env_data: np.ndarray) -> None:
        from reinforcement_learning.environment import TradingEnv
        env = TradingEnv(env_data)
        obs, _ = env.reset()
        obs, _, _, _, info1 = env.step(1)  # BUY
        assert info1["position"] > 0
        obs, _, _, _, info2 = env.step(2)  # SELL
        assert info2["position"] == 0
