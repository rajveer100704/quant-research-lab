#!/usr/bin/env python
"""
run_pipeline.py — Run the full trading research pipeline
=========================================================

Usage::

    python run_pipeline.py
    python run_pipeline.py --symbol ETHUSDT --interval 4h --limit 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator.pipeline_runner import TradingPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AI Quant Trading Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --symbol ETHUSDT --interval 4h
  python run_pipeline.py --symbol BTCUSDT --limit 5000 --balance 50000
        """,
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1h", help="Kline interval (default: 1h)")
    parser.add_argument("--limit", type=int, default=2000, help="Number of candles (default: 2000)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Initial balance (default: 10000)")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON to this path")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Quant Trading Research Platform — Pipeline Runner")
    print("=" * 60)
    print(f"  Symbol    : {args.symbol}")
    print(f"  Interval  : {args.interval}")
    print(f"  Candles   : {args.limit}")
    print(f"  Balance   : ${args.balance:,.2f}")
    print("=" * 60)

    pipeline = TradingPipeline(
        symbol=args.symbol,
        interval=args.interval,
        initial_balance=args.balance,
    )
    results = pipeline.run(data_limit=args.limit)

    if "error" in results:
        print(f"\n❌ Pipeline failed: {results['error']}")
        sys.exit(1)

    # Print summary
    sim = results.get("simulation", {})
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Regime           : {results.get('regime', 'N/A')}")
    print(f"  Fusion Decision  : {results.get('fusion_decision', 'N/A')}")
    print(f"  Risk Approved    : {results.get('risk_approved', 'N/A')}")
    print(f"  Total Return     : {sim.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe Ratio     : {sim.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown     : {sim.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total Trades     : {sim.get('n_trades', 0)}")
    print(f"  Final Portfolio  : ${sim.get('final_portfolio', 0):,.2f}")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        # Remove non-serializable items
        save_results = {k: v for k, v in results.items() if k != "simulation"}
        save_results["simulation"] = {k: v for k, v in sim.items() if k != "equity_curve"}
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"\n📁 Results saved → {output_path}")


if __name__ == "__main__":
    main()
