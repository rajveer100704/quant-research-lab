"""
Order Book Stream
=================
Real-time order book ingestion via Binance WebSocket.
Captures depth snapshots and stores them for microstructure analysis.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderBookStream:
    """
    Streams real-time order book updates from Binance WebSocket.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    depth : int
        Number of price levels (5, 10, or 20).
    data_dir : Path, optional
        Storage directory override.

    Example
    -------
    >>> stream = OrderBookStream("BTCUSDT", depth=10)
    >>> asyncio.run(stream.run(duration_seconds=60))
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        depth: int = 10,
        data_dir: Optional[Path] = None,
    ) -> None:
        settings = get_settings()
        self.symbol = symbol.lower()
        self.depth = depth
        self.data_dir = data_dir or settings.data.raw_data_dir / "orderbook"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.ws_url = f"{settings.binance.ws_url}/{self.symbol}@depth{self.depth}@100ms"
        self._snapshots: list[dict[str, Any]] = []

    async def _connect(self, duration_seconds: int = 60) -> None:
        """Connect to WebSocket and accumulate snapshots."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required: pip install websockets")
            return

        logger.info("Connecting to %s", self.ws_url)
        end_time = asyncio.get_event_loop().time() + duration_seconds

        try:
            async with websockets.connect(self.ws_url) as ws:
                logger.info("WebSocket connected for %s", self.symbol.upper())
                while asyncio.get_event_loop().time() < end_time:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(msg)
                        snapshot = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "symbol": self.symbol.upper(),
                            "bids": data.get("bids", data.get("b", [])),
                            "asks": data.get("asks", data.get("a", [])),
                        }
                        self._snapshots.append(snapshot)
                    except asyncio.TimeoutError:
                        continue
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)

        logger.info("Collected %d order book snapshots", len(self._snapshots))

    def run(self, duration_seconds: int = 60) -> list[dict[str, Any]]:
        """
        Run the streaming loop for *duration_seconds* and return snapshots.
        """
        asyncio.run(self._connect(duration_seconds))
        return self._snapshots

    def save(self) -> Path:
        """Persist collected snapshots to a JSON file."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.data_dir / f"{self.symbol.upper()}_orderbook_{ts}.json"
        with open(path, "w") as f:
            json.dump(self._snapshots, f, indent=2)
        logger.info("Order book data saved → %s", path)
        return path

    def get_latest_snapshot(self) -> Optional[dict[str, Any]]:
        """Return the most recent snapshot, if available."""
        return self._snapshots[-1] if self._snapshots else None
