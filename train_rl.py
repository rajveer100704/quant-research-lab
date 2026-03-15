#!/usr/bin/env python
"""
train_rl.py — Train a reinforcement learning trading agent
============================================================

Usage::

    python train_rl.py
    python train_rl.py --algorithm A2C --timesteps 200000 --symbol ETHUSDT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reinforcement_learning.train import prepare_training_data
from reinforcement_learning.agent import RLTrader
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RL Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_rl.py
  python train_rl.py --algorithm A2C --timesteps 200000
  python train_rl.py --symbol ETHUSDT --balance 50000
        """,
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Initial balance")
    parser.add_argument("--limit", type=int, default=5000, help="Candles to fetch")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Quant Trading Platform — RL Agent Training")
    print("=" * 60)
    print(f"  Symbol     : {args.symbol}")
    print(f"  Algorithm  : {args.algorithm}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  Balance    : ${args.balance:,.2f}")
    print("=" * 60)

    print("\n📊 Preparing training data...")
    train_data, val_data = prepare_training_data(args.symbol, args.limit)
    print(f"   Train: {len(train_data)} steps | Val: {len(val_data)} steps")

    print(f"\n🤖 Training {args.algorithm} agent...")
    trader = RLTrader(
        train_data=train_data,
        val_data=val_data,
        algorithm=args.algorithm,
        initial_balance=args.balance,
    )
    trader.train(total_timesteps=args.timesteps)

    print("\n" + "=" * 60)
    print(f"  ✅ Training complete!")
    print(f"  Model saved → models/rl/{args.algorithm.lower()}_trader")
    print("=" * 60)


if __name__ == "__main__":
    main()
