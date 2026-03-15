"""
RL Agent Training Script
========================
CLI entry point for training reinforcement learning trading agents.

Usage::

    python -m reinforcement_learning.train --symbol BTCUSDT --algorithm PPO --timesteps 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings
from data_pipeline.market_data_collector import MarketDataCollector
from feature_engine.technical_indicators import TechnicalIndicators
from reinforcement_learning.agent import RLTrader
from utils.logger import get_logger

logger = get_logger(__name__)


def prepare_training_data(symbol: str, limit: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch OHLCV data, compute features, and prepare train/val arrays.

    The RL environment expects a feature matrix where the first column
    is the close price (used for PnL calculations). Remaining columns
    are normalised technical indicators.

    Returns
    -------
    train_data, val_data : ndarrays of shape ``(n_steps, n_features)``
    """
    collector = MarketDataCollector(symbols=[symbol])
    df = collector.fetch_extended_history(symbol, total_candles=limit)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # Rename for ta library
    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})

    # Compute features
    ti = TechnicalIndicators()
    df = ti.compute_all(df)
    df.dropna(inplace=True)

    # Select features for RL
    feature_cols = [
        "close", "volume", "rsi", "macd", "macd_histogram",
        "volatility", "momentum", "roc",
    ]
    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values.astype(np.float32)

    # Normalise (z-score per column)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    # Keep close price un-normalised (column 0) for PnL
    data[:, 1:] = (data[:, 1:] - mean[1:]) / std[1:]

    # Chronological split (80/20)
    split = int(len(data) * 0.8)
    return data[:split], data[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--limit", type=int, default=5000, help="Candles to fetch")
    args = parser.parse_args()

    logger.info("Preparing training data for %s...", args.symbol)
    train_data, val_data = prepare_training_data(args.symbol, args.limit)
    logger.info("Train: %d steps, Val: %d steps", len(train_data), len(val_data))

    trader = RLTrader(
        train_data=train_data,
        val_data=val_data,
        algorithm=args.algorithm,
        initial_balance=args.balance,
    )
    trader.train(total_timesteps=args.timesteps)
    logger.info("Done! Model saved to models/rl/")


if __name__ == "__main__":
    main()
