"""
Order Book Features
===================
Computes microstructure features from order book snapshots:
bid-ask spread, order imbalance, depth, and volume delta.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class OrderBookFeatures:
    """
    Extracts microstructure features from order book snapshots.

    Each snapshot is expected to be a dict with keys:
    ``bids`` and ``asks``, where each is a list of ``[price, quantity]`` pairs.

    Example
    -------
    >>> obf = OrderBookFeatures()
    >>> features = obf.compute_all(snapshots)
    """

    @staticmethod
    def compute_bid_ask_spread(snapshot: dict[str, Any]) -> dict[str, float]:
        """
        Compute the bid-ask spread from a single order book snapshot.

        Returns
        -------
        dict with ``best_bid``, ``best_ask``, ``spread``, ``spread_bps``.
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks:
            return {"best_bid": 0.0, "best_ask": 0.0, "spread": 0.0, "spread_bps": 0.0}

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10_000 if mid_price > 0 else 0.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_bps": spread_bps,
        }

    @staticmethod
    def compute_order_imbalance(snapshot: dict[str, Any], levels: int = 5) -> float:
        """
        Compute order-flow imbalance across the top *levels*.

        .. math::
            OI = \\frac{\\sum bid\\_qty - \\sum ask\\_qty}{\\sum bid\\_qty + \\sum ask\\_qty}

        Returns a value in ``[-1, 1]`` where positive = more buying pressure.
        """
        bids = snapshot.get("bids", [])[:levels]
        asks = snapshot.get("asks", [])[:levels]

        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    @staticmethod
    def compute_depth(snapshot: dict[str, Any], levels: int = 10) -> dict[str, float]:
        """
        Compute total liquidity depth on each side.

        Returns
        -------
        dict with ``bid_depth``, ``ask_depth``, ``total_depth``.
        """
        bids = snapshot.get("bids", [])[:levels]
        asks = snapshot.get("asks", [])[:levels]

        bid_depth = sum(float(b[0]) * float(b[1]) for b in bids)
        ask_depth = sum(float(a[0]) * float(a[1]) for a in asks)

        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
        }

    @staticmethod
    def compute_volume_delta(
        snapshot_prev: dict[str, Any],
        snapshot_curr: dict[str, Any],
        levels: int = 5,
    ) -> float:
        """
        Compute the change in net order flow between two consecutive snapshots.

        Returns positive value when buying pressure is increasing.
        """
        def _net(snap: dict[str, Any]) -> float:
            bids = snap.get("bids", [])[:levels]
            asks = snap.get("asks", [])[:levels]
            return sum(float(b[1]) for b in bids) - sum(float(a[1]) for a in asks)

        return _net(snapshot_curr) - _net(snapshot_prev)

    def compute_all(self, snapshots: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Process a list of order book snapshots into a features DataFrame.

        Columns: ``timestamp, best_bid, best_ask, spread, spread_bps,
        order_imbalance, bid_depth, ask_depth, total_depth, volume_delta``.
        """
        logger.info("Computing order book features for %d snapshots", len(snapshots))
        records: list[dict[str, Any]] = []

        for i, snap in enumerate(snapshots):
            spread_data = self.compute_bid_ask_spread(snap)
            imbalance = self.compute_order_imbalance(snap)
            depth = self.compute_depth(snap)
            vol_delta = (
                self.compute_volume_delta(snapshots[i - 1], snap)
                if i > 0
                else 0.0
            )

            record = {
                "timestamp": snap.get("timestamp", ""),
                **spread_data,
                "order_imbalance": imbalance,
                **depth,
                "volume_delta": vol_delta,
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info("Order book features computed: %d rows, %d columns", len(df), len(df.columns))
        return df
