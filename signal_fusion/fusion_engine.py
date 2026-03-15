"""
Signal Fusion Engine
====================
Combines multiple trading signals into a unified trade decision.

Methods:
- **Weighted average** — linear combination with configurable weights
- **Majority voting** — consensus across signals
- **Meta-model** — logistic regression trained on signal outputs
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.logger import get_logger

logger = get_logger(__name__)


class TradeDecision(str, Enum):
    """Final trade decision."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalFusion:
    """
    Fuses multiple signals into a single trade decision.

    Parameters
    ----------
    weights : dict[str, float], optional
        Signal name → weight mapping for weighted average.
        Defaults to equal weights.

    Example
    -------
    >>> fusion = SignalFusion(weights={"lstm": 0.3, "rl": 0.3, "sentiment": 0.2, "orderflow": 0.2})
    >>> decision = fusion.weighted_average(signals)
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or {}
        self._meta_model: Optional[LogisticRegression] = None

    def weighted_average(
        self,
        signals: dict[str, float],
        buy_threshold: float = 0.2,
        sell_threshold: float = -0.2,
    ) -> tuple[TradeDecision, float]:
        """
        Compute weighted average of signals and threshold to a decision.

        Parameters
        ----------
        signals : dict[str, float]
            Signal name → value (expected in [-1, 1]).
        buy_threshold / sell_threshold : float
            Thresholds for buy/sell decision.

        Returns
        -------
        (TradeDecision, composite_score)
        """
        if not signals:
            return TradeDecision.HOLD, 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for name, value in signals.items():
            w = self.weights.get(name, 1.0 / len(signals))
            weighted_sum += w * value
            total_weight += w

        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        if score >= buy_threshold:
            decision = TradeDecision.BUY
        elif score <= sell_threshold:
            decision = TradeDecision.SELL
        else:
            decision = TradeDecision.HOLD

        logger.info("Fusion: score=%.4f → %s", score, decision.value)
        return decision, score

    @staticmethod
    def majority_vote(signals: dict[str, float]) -> tuple[TradeDecision, float]:
        """
        Simple majority voting: each signal votes BUY (>0), SELL (<0), or HOLD (0).

        Returns
        -------
        (TradeDecision, confidence) where confidence = vote_fraction.
        """
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for value in signals.values():
            if value > 0:
                votes["BUY"] += 1
            elif value < 0:
                votes["SELL"] += 1
            else:
                votes["HOLD"] += 1

        total = sum(votes.values())
        winner = max(votes, key=votes.get)  # type: ignore[arg-type]
        confidence = votes[winner] / total if total > 0 else 0.0

        return TradeDecision(winner), confidence

    def train_meta_model(
        self,
        signal_history: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """
        Train a logistic regression meta-model on historical signal outputs.

        Parameters
        ----------
        signal_history : DataFrame
            Columns = signal names, rows = timesteps, values = signal outputs.
        labels : Series
            Target labels (0=HOLD, 1=BUY, 2=SELL).
        """
        X = signal_history.dropna()
        y = labels.loc[X.index]

        self._meta_model = LogisticRegression(
            multi_class="multinomial", max_iter=500, random_state=42
        )
        self._meta_model.fit(X, y)
        accuracy = self._meta_model.score(X, y)
        logger.info("Meta-model trained: accuracy=%.3f", accuracy)

    def meta_predict(self, signals: dict[str, float]) -> tuple[TradeDecision, float]:
        """
        Predict trade decision using the trained meta-model.

        Returns (decision, confidence).
        """
        if self._meta_model is None:
            raise RuntimeError("Meta-model not trained. Call train_meta_model() first.")

        X = np.array([list(signals.values())]).reshape(1, -1)
        pred = self._meta_model.predict(X)[0]
        proba = self._meta_model.predict_proba(X).max()

        label_map = {0: TradeDecision.HOLD, 1: TradeDecision.BUY, 2: TradeDecision.SELL}
        return label_map.get(pred, TradeDecision.HOLD), float(proba)
