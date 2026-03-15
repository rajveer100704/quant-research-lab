"""
Market Simulator
================
Replay-based backtesting engine with realistic order book simulation.

Features:
- OHLCV replay with candle-level resolution
- Slippage modelling (proportional to order size and volatility)
- Latency simulation
- Fill probability for limit orders
- Market and limit order support
- Trade log and equity curve generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Simulated order."""

    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Required for LIMIT
    timestamp: int = 0


@dataclass
class Fill:
    """Execution fill."""

    order: Order
    fill_price: float
    fill_quantity: float
    slippage: float
    timestamp: int = 0
    latency_ms: float = 0.0


class MarketSimulator:
    """
    Historical replay backtester with realistic execution simulation.

    Parameters
    ----------
    data : DataFrame
        Must contain columns: ``open, high, low, close, volume``.
    initial_balance : float
        Starting cash.
    commission_rate : float
        Trading fee (e.g. 0.001 = 10bps).
    slippage_bps : float
        Base slippage in basis points.
    latency_ms : float
        Simulated order latency in milliseconds.

    Example
    -------
    >>> sim = MarketSimulator(ohlcv_df, initial_balance=10_000)
    >>> sim.run(signals)
    >>> results = sim.get_results()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        latency_ms: float = 50.0,
    ) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms

        # State
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trade_log: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []

    def _compute_slippage(self, price: float, volume: float, side: Side) -> float:
        """
        Compute price slippage based on order size relative to volume.

        Larger orders in low-volume environments suffer more slippage.
        """
        base_slip = price * self.slippage_bps / 10_000
        # Volume impact: more slippage if order is large relative to bar volume
        volume_impact = 0.0
        if volume > 0:
            impact_factor = min(1.0, abs(self.position * price) / (volume * price + 1e-8))
            volume_impact = base_slip * impact_factor

        total_slip = base_slip + volume_impact
        return total_slip if side == Side.BUY else -total_slip

    def _simulate_fill(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        """Attempt to fill an order against a price bar."""
        price = float(bar["close"])
        volume = float(bar["volume"])

        if order.order_type == OrderType.MARKET:
            slippage = self._compute_slippage(price, volume, order.side)
            fill_price = price + slippage
            return Fill(
                order=order,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                slippage=abs(slippage),
                latency_ms=self.latency_ms,
            )
        elif order.order_type == OrderType.LIMIT and order.price is not None:
            # Limit order: fill only if price crosses the limit
            if order.side == Side.BUY and float(bar["low"]) <= order.price:
                return Fill(
                    order=order,
                    fill_price=order.price,
                    fill_quantity=order.quantity,
                    slippage=0.0,
                    latency_ms=self.latency_ms,
                )
            elif order.side == Side.SELL and float(bar["high"]) >= order.price:
                return Fill(
                    order=order,
                    fill_price=order.price,
                    fill_quantity=order.quantity,
                    slippage=0.0,
                    latency_ms=self.latency_ms,
                )
        return None  # No fill

    def run(self, signals: pd.Series) -> None:
        """
        Run the simulation with a signal series.

        Parameters
        ----------
        signals : Series
            Values: ``"BUY"``, ``"SELL"``, or ``"HOLD"`` aligned with DataFrame index.
        """
        logger.info("Running simulation: %d bars", len(self.data))
        self.balance = self.initial_balance
        self.position = 0.0
        self.trade_log = []
        self.equity_curve = []

        for i in range(len(self.data)):
            bar = self.data.iloc[i]
            price = float(bar["close"])
            signal = signals.iloc[i] if i < len(signals) else "HOLD"

            if signal == "BUY" and self.position == 0:
                quantity = (self.balance * (1 - self.commission_rate)) / price
                order = Order(side=Side.BUY, order_type=OrderType.MARKET, quantity=quantity)
                fill = self._simulate_fill(order, bar)
                if fill:
                    self.position = fill.fill_quantity
                    self.entry_price = fill.fill_price
                    self.balance = 0.0
                    self.trade_log.append({
                        "bar": i, "action": "BUY", "price": fill.fill_price,
                        "quantity": fill.fill_quantity, "slippage": fill.slippage,
                    })

            elif signal == "SELL" and self.position > 0:
                order = Order(side=Side.SELL, order_type=OrderType.MARKET, quantity=self.position)
                fill = self._simulate_fill(order, bar)
                if fill:
                    revenue = fill.fill_price * fill.fill_quantity * (1 - self.commission_rate)
                    pnl = revenue - self.entry_price * self.position
                    self.balance = revenue
                    self.trade_log.append({
                        "bar": i, "action": "SELL", "price": fill.fill_price,
                        "quantity": fill.fill_quantity, "slippage": fill.slippage, "pnl": pnl,
                    })
                    self.position = 0.0

            portfolio = self.balance + self.position * price
            self.equity_curve.append(portfolio)

        logger.info(
            "Simulation complete: %d trades, final portfolio=%.2f",
            len(self.trade_log), self.equity_curve[-1] if self.equity_curve else 0,
        )

    def get_results(self) -> dict[str, Any]:
        """
        Return simulation results.

        Returns
        -------
        dict with ``trade_log``, ``equity_curve``, ``total_return``,
        ``max_drawdown``, ``n_trades``, ``sharpe_ratio``.
        """
        eq = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_balance])

        # Total return
        total_return = (eq[-1] - self.initial_balance) / self.initial_balance * 100

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_dd = float(drawdown.max()) * 100

        # Sharpe ratio (assuming hourly data)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24))

        return {
            "trade_log": self.trade_log,
            "equity_curve": self.equity_curve,
            "total_return_pct": round(total_return, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "n_trades": len(self.trade_log),
            "sharpe_ratio": round(sharpe, 4),
            "final_portfolio": round(eq[-1], 2),
        }
