"""
Execution Engine
================
Exchange interaction layer for submitting, cancelling, and tracking orders.

Supports:
- Paper trading mode (dry-run, no real orders)
- Live mode via Binance API
- Order status tracking

Adapted from the existing quant_research_lab BinanceClient implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from binance.client import Client
from binance.enums import (
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)
from binance.exceptions import BinanceAPIException

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrderResult:
    """Result of an order submission."""

    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""
    message: str = ""
    timestamp: str = ""


class ExecutionEngine:
    """
    Manages order lifecycle on Binance.

    Parameters
    ----------
    paper_trading : bool
        If ``True``, simulate orders without hitting the exchange.

    Example
    -------
    >>> engine = ExecutionEngine(paper_trading=True)
    >>> result = engine.submit_order("BTCUSDT", "BUY", "MARKET", quantity=0.001)
    """

    def __init__(self, paper_trading: bool = True) -> None:
        settings = get_settings()
        self.paper_trading = paper_trading
        self._client: Optional[Client] = None
        self._paper_orders: list[OrderResult] = []
        self._order_counter = 0

        if not paper_trading:
            try:
                self._client = Client(
                    settings.binance.api_key,
                    settings.binance.api_secret,
                    testnet=settings.binance.testnet,
                )
                logger.info("ExecutionEngine: live mode (testnet=%s)", settings.binance.testnet)
            except Exception as exc:
                logger.error("Failed to init Binance client: %s. Falling back to paper.", exc)
                self.paper_trading = True
        else:
            logger.info("ExecutionEngine: paper trading mode")

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: float = 0.0,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> OrderResult:
        """
        Submit an order to the exchange.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. ``"BTCUSDT"``).
        side : str
            ``"BUY"`` or ``"SELL"``.
        order_type : str
            ``"MARKET"`` or ``"LIMIT"``.
        quantity : float
            Order quantity.
        price : float, optional
            Required for LIMIT orders.
        """
        if self.paper_trading:
            return self._paper_order(symbol, side, order_type, quantity, price)
        return self._live_order(symbol, side, order_type, quantity, price, time_in_force)

    def _paper_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
    ) -> OrderResult:
        """Simulate an order without exchange interaction."""
        self._order_counter += 1
        result = OrderResult(
            success=True,
            order_id=f"PAPER-{self._order_counter:06d}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price or 0.0,
            status="FILLED",
            message="Paper trade executed",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._paper_orders.append(result)
        logger.info("Paper order: %s %s %.6f %s @ %s", side, symbol, quantity, order_type, price or "MARKET")
        return result

    def _live_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        time_in_force: str,
    ) -> OrderResult:
        """Submit a real order to Binance."""
        if self._client is None:
            return OrderResult(success=False, message="Binance client not initialised")

        try:
            binance_side = SIDE_BUY if side.upper() == "BUY" else SIDE_SELL

            if order_type.upper() == "MARKET":
                resp = self._client.create_order(
                    symbol=symbol.upper(),
                    side=binance_side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity,
                )
            elif order_type.upper() == "LIMIT":
                if price is None:
                    return OrderResult(success=False, message="Limit order requires price")
                resp = self._client.create_order(
                    symbol=symbol.upper(),
                    side=binance_side,
                    type=ORDER_TYPE_LIMIT,
                    quantity=quantity,
                    price=str(price),
                    timeInForce=time_in_force,
                )
            else:
                return OrderResult(success=False, message=f"Unsupported order type: {order_type}")

            result = OrderResult(
                success=True,
                order_id=str(resp.get("orderId", "")),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=float(resp.get("executedQty", quantity)),
                price=float(resp.get("price", 0)),
                status=resp.get("status", "UNKNOWN"),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("Live order filled: %s", result.order_id)
            return result

        except BinanceAPIException as exc:
            logger.error("Binance order failed: %s", exc)
            return OrderResult(success=False, message=str(exc))

    def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        if self.paper_trading:
            logger.info("Paper cancel: %s", order_id)
            return {"status": "cancelled", "order_id": order_id}

        if self._client is None:
            return {"status": "error", "message": "Client not initialised"}

        try:
            result = self._client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info("Order cancelled: %s", order_id)
            return {"status": "cancelled", "data": result}
        except BinanceAPIException as exc:
            return {"status": "error", "message": str(exc)}

    def get_order_status(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Check the status of an order."""
        if self.paper_trading:
            for order in self._paper_orders:
                if order.order_id == order_id:
                    return {"status": order.status, "order": order}
            return {"status": "NOT_FOUND"}

        if self._client is None:
            return {"status": "error", "message": "Client not initialised"}

        try:
            result = self._client.get_order(symbol=symbol, orderId=order_id)
            return {"status": result.get("status", "UNKNOWN"), "data": result}
        except BinanceAPIException as exc:
            return {"status": "error", "message": str(exc)}

    def get_trade_history(self) -> list[OrderResult]:
        """Return paper trading history."""
        return self._paper_orders.copy()
