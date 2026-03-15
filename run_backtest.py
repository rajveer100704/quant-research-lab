#!/usr/bin/env python
"""
run_backtest.py — Run a backtest simulation
=============================================

Usage::

    python run_backtest.py
    python run_backtest.py --symbol BTCUSDT --limit 3000 --strategy momentum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline.market_data_collector import MarketDataCollector
from feature_engine.technical_indicators import TechnicalIndicators
from alpha_research.alpha_signal import MomentumAlpha, MeanReversionAlpha, RSICrossAlpha
from market_simulator.simulator import MarketSimulator
from utils.logger import get_logger

logger = get_logger(__name__)


STRATEGIES = {
    "momentum": lambda df: MomentumAlpha(12).generate(df),
    "mean_reversion": lambda df: MeanReversionAlpha(20).generate(df),
    "rsi_cross": lambda df: RSICrossAlpha().generate(df),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Backtest Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  momentum        — 12-period price momentum
  mean_reversion  — 20-period mean reversion z-score
  rsi_cross       — RSI overbought/oversold crossover

Examples:
  python run_backtest.py
  python run_backtest.py --strategy rsi_cross --limit 5000
  python run_backtest.py --symbol ETHUSDT --balance 50000
        """,
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--strategy", type=str, default="momentum", choices=STRATEGIES.keys())
    parser.add_argument("--limit", type=int, default=2000, help="Candles to fetch")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Initial balance")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (default: 0.1%%)")
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in basis points")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON to this path")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Quant Trading Platform — Backtest Engine")
    print("=" * 60)
    print(f"  Symbol     : {args.symbol}")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Candles    : {args.limit}")
    print(f"  Balance    : ${args.balance:,.2f}")
    print(f"  Commission : {args.commission * 100:.2f}%")
    print(f"  Slippage   : {args.slippage} bps")
    print("=" * 60)

    # Fetch data
    print("\n📊 Fetching market data...")
    collector = MarketDataCollector(symbols=[args.symbol])
    ohlcv = collector.fetch_ohlcv(args.symbol, limit=args.limit)

    if ohlcv.empty:
        print("❌ No data available. Check your API connection.")
        sys.exit(1)
    print(f"   Loaded {len(ohlcv)} candles")

    # Compute features
    print("🔧 Computing technical indicators...")
    ti = TechnicalIndicators()
    features = ti.compute_all(ohlcv)
    features.dropna(inplace=True)
    print(f"   Features computed: {len(features.columns)} columns")

    # Generate signals
    print(f"📡 Generating {args.strategy} signals...")
    alpha = STRATEGIES[args.strategy](features)
    signals = pd.Series("HOLD", index=features.index)
    signals[alpha.values > 0] = "BUY"
    signals[alpha.values < 0] = "SELL"

    buy_count = (signals == "BUY").sum()
    sell_count = (signals == "SELL").sum()
    print(f"   Signals: {buy_count} BUY | {sell_count} SELL | {len(signals) - buy_count - sell_count} HOLD")

    # Run simulation
    print("🎯 Running simulation...")
    sim = MarketSimulator(
        features,
        initial_balance=args.balance,
        commission_rate=args.commission,
        slippage_bps=args.slippage,
    )
    sim.run(signals)
    results = sim.get_results()

    # Print results
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Strategy        : {args.strategy}")
    print(f"  Total Return    : {results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio    : {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown    : {results['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades    : {results['n_trades']}")
    print(f"  Final Portfolio  : ${results['final_portfolio']:,.2f}")
    print("=" * 60)

    # Trade log summary
    if results["trade_log"]:
        pnl_trades = [t for t in results["trade_log"] if "pnl" in t]
        if pnl_trades:
            wins = sum(1 for t in pnl_trades if t["pnl"] > 0)
            losses = len(pnl_trades) - wins
            total_pnl = sum(t["pnl"] for t in pnl_trades)
            print(f"\n  Wins: {wins} | Losses: {losses} | Net PnL: ${total_pnl:,.2f}")

    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable trade log for clean JSON
        save_results = {k: v for k, v in results.items() if k != "trade_log"}
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"\n📁 Results saved → {output_path}")


if __name__ == "__main__":
    main()
