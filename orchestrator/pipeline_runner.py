"""
Trading Pipeline Orchestrator
=============================
Central coordinator that runs the full research pipeline end-to-end:

    collect data → compute features → store features
    → generate signals → run models → fuse signals
    → apply risk management → simulate trades → generate report

This is the single entry point for running experiments.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import get_settings
from data_pipeline.market_data_collector import MarketDataCollector
from feature_engine.technical_indicators import TechnicalIndicators
from feature_store.store import FeatureStore
from market_intelligence.aggregator import MarketIntelligence
from market_intelligence.market_regime import MarketRegimeDetector
from alpha_research.alpha_signal import MomentumAlpha, MeanReversionAlpha, RSICrossAlpha
from alpha_research.alpha_analyzer import AlphaAnalyzer
from signal_fusion.fusion_engine import SignalFusion
from risk_management.risk_manager import RiskManager
from market_simulator.simulator import MarketSimulator
from utils.logger import get_logger

logger = get_logger(__name__)


class TradingPipeline:
    """
    End-to-end trading research pipeline.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. ``"BTCUSDT"``).
    interval : str
        Kline interval (e.g. ``"1h"``).
    initial_balance : float
        Starting portfolio balance for simulation.

    Example
    -------
    >>> pipeline = TradingPipeline(symbol="BTCUSDT", interval="1h")
    >>> results = pipeline.run()
    >>> print(results["simulation"]["total_return_pct"])
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        initial_balance: float = 10_000.0,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.results: dict[str, Any] = {}

    def run(self, data_limit: int = 2000) -> dict[str, Any]:
        """
        Execute the full pipeline.

        Steps
        -----
        1. Collect market data
        2. Compute technical features
        3. Store features
        4. Detect market regime
        5. Generate alpha signals
        6. Analyse alpha quality
        7. Fuse signals
        8. Apply risk management
        9. Run market simulation
        10. Return results

        Parameters
        ----------
        data_limit : int
            Number of candles to fetch.

        Returns
        -------
        dict with keys: ``ohlcv``, ``features``, ``regime``, ``alpha_report``,
        ``fusion_decision``, ``risk_decision``, ``simulation``.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE START: %s @ %s", self.symbol, self.interval)
        logger.info("=" * 60)

        # --- Step 1: Collect Data ---
        logger.info("[1/9] Collecting market data...")
        collector = MarketDataCollector(symbols=[self.symbol], interval=self.interval)
        ohlcv = collector.fetch_ohlcv(self.symbol, limit=data_limit)

        if ohlcv.empty:
            logger.error("No data fetched. Pipeline aborted.")
            return {"error": "No data available"}

        # --- Step 2: Compute Features ---
        logger.info("[2/9] Computing technical features...")
        ti = TechnicalIndicators()
        features = ti.compute_all(ohlcv)
        features.dropna(inplace=True)

        # --- Step 3: Store Features ---
        logger.info("[3/9] Storing features...")
        store = FeatureStore()
        store.save_features(features, symbol=self.symbol.replace("USDT", ""), feature_set="technical")

        # --- Step 4: Market Regime ---
        logger.info("[4/9] Detecting market regime...")
        detector = MarketRegimeDetector()
        regime = detector.detect(features)
        logger.info("Current regime: %s", regime.value)

        # --- Step 5: Generate Alpha Signals ---
        logger.info("[5/9] Generating alpha signals...")
        alphas = [
            MomentumAlpha(lookback=12).generate(features),
            MeanReversionAlpha(window=20).generate(features),
            RSICrossAlpha().generate(features),
        ]

        # --- Step 6: Analyse Alphas ---
        logger.info("[6/9] Analysing alpha quality...")
        forward_returns = features["close"].pct_change().shift(-1)
        analyzer = AlphaAnalyzer(forward_returns)
        alpha_report = analyzer.rank_signals(alphas)

        # --- Step 7: Fuse Signals ---
        logger.info("[7/9] Fusing signals...")
        fusion = SignalFusion(weights={
            "momentum_12": 0.3,
            "mean_reversion_20": 0.3,
            "rsi_cross": 0.4,
        })
        latest_signals = {}
        for alpha in alphas:
            last_val = alpha.values.dropna().iloc[-1] if not alpha.values.dropna().empty else 0.0
            latest_signals[alpha.name] = float(last_val)

        decision, score = fusion.weighted_average(latest_signals)

        # --- Step 8: Risk Management ---
        logger.info("[8/9] Applying risk controls...")
        rm = RiskManager(portfolio_value=self.initial_balance)
        price = float(features["close"].iloc[-1])
        risk_decision = rm.check_risk(
            entry_price=price,
            signal_strength=abs(score),
        )

        # --- Step 9: Simulate ---
        logger.info("[9/9] Running market simulation...")
        # Generate trade signals from fused alpha
        fused_signal = pd.Series("HOLD", index=features.index)
        for alpha in alphas:
            buy_mask = alpha.values > 0
            sell_mask = alpha.values < 0
            fused_signal[buy_mask] = "BUY"
            fused_signal[sell_mask] = "SELL"

        sim = MarketSimulator(features, initial_balance=self.initial_balance)
        sim.run(fused_signal)
        sim_results = sim.get_results()

        # --- Compile Results ---
        self.results = {
            "symbol": self.symbol,
            "interval": self.interval,
            "regime": regime.value,
            "n_bars": len(features),
            "alpha_report": alpha_report.to_dict("records"),
            "fusion_decision": decision.value,
            "fusion_score": round(score, 4),
            "risk_approved": risk_decision.approved,
            "risk_position_size": round(risk_decision.position_size, 4),
            "simulation": sim_results,
        }

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE: return=%.2f%%, Sharpe=%.2f",
                     sim_results["total_return_pct"], sim_results["sharpe_ratio"])
        logger.info("=" * 60)

        return self.results
