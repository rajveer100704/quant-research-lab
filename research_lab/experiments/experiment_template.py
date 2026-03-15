"""
Experiment Template
===================
Reusable template for running structured alpha research experiments.

Usage
-----
Subclass ``ResearchExperiment`` and implement ``hypothesis()``,
``run_experiment()``, and ``evaluate()`` to create a new experiment.

This mirrors the scientific method used in quant research:
1. State a hypothesis
2. Collect data and features
3. Run the test
4. Evaluate results
5. Generate a report

Example
-------
>>> class MomentumExperiment(ResearchExperiment):
...     def hypothesis(self):
...         return "12-period momentum predicts 1h forward returns with IC > 0.05"
...
...     def run_experiment(self, features):
...         from alpha_research.alpha_signal import MomentumAlpha
...         return MomentumAlpha(12).generate(features)
...
...     def evaluate(self, signal, returns):
...         from alpha_research.alpha_analyzer import AlphaAnalyzer
...         return AlphaAnalyzer(returns).analyze(signal)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from experiments.experiment_manager import ExperimentManager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment outputs."""

    hypothesis: str
    conclusion: str
    metrics: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    timestamp: str = ""


class ResearchExperiment(ABC):
    """
    Abstract base class for structured research experiments.

    Enforces the scientific method:
    hypothesis → experiment → evaluation → conclusion.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._em = ExperimentManager()

    @abstractmethod
    def hypothesis(self) -> str:
        """State the research hypothesis as a string."""
        ...

    @abstractmethod
    def run_experiment(self, features: pd.DataFrame) -> Any:
        """Execute the experiment logic. Returns raw results."""
        ...

    @abstractmethod
    def evaluate(self, result: Any, forward_returns: pd.Series) -> dict[str, Any]:
        """Evaluate the experiment results quantitatively."""
        ...

    def execute(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> ExperimentResult:
        """
        Run the full experiment lifecycle.

        1. Log hypothesis
        2. Run experiment
        3. Evaluate results
        4. Determine conclusion
        5. Track in experiment manager
        """
        hyp = self.hypothesis()
        logger.info("=" * 60)
        logger.info("EXPERIMENT: %s", self.name)
        logger.info("HYPOTHESIS: %s", hyp)
        logger.info("=" * 60)

        # Track experiment
        exp_id = self._em.start_experiment(
            name=self.name,
            params={"hypothesis": hyp, "description": self.description},
            tags=["research_lab"],
        )

        # Run
        raw_result = self.run_experiment(features)

        # Evaluate
        metrics = self.evaluate(raw_result, forward_returns)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._em.log_metric(exp_id, key, float(value))

        # Conclusion
        accepted = self._determine_acceptance(metrics)
        conclusion = (
            f"ACCEPTED: Hypothesis supported by data. Key metrics: {metrics}"
            if accepted
            else f"REJECTED: Insufficient evidence. Key metrics: {metrics}"
        )

        self._em.end_experiment(exp_id, status="COMPLETED")

        result = ExperimentResult(
            hypothesis=hyp,
            conclusion=conclusion,
            metrics=metrics,
            accepted=accepted,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info("CONCLUSION: %s", conclusion)
        return result

    def _determine_acceptance(self, metrics: dict[str, Any]) -> bool:
        """Determine if the hypothesis is accepted based on metrics."""
        ic = abs(metrics.get("ic", 0))
        sharpe = metrics.get("sharpe", 0)
        win_rate = metrics.get("win_rate", 50)

        # Minimum quality thresholds
        return ic > 0.03 and sharpe > 0.5 and win_rate > 52


# --- Built-in Experiment Templates ---


class MomentumExperiment(ResearchExperiment):
    """Test if price momentum predicts forward returns."""

    def __init__(self, lookback: int = 12) -> None:
        super().__init__(
            name=f"momentum_{lookback}_alpha",
            description=f"Test {lookback}-period momentum as alpha signal",
        )
        self.lookback = lookback

    def hypothesis(self) -> str:
        return f"{self.lookback}-period price momentum generates positive IC (> 0.03) and Sharpe (> 0.5)"

    def run_experiment(self, features: pd.DataFrame) -> Any:
        from alpha_research.alpha_signal import MomentumAlpha
        return MomentumAlpha(self.lookback).generate(features)

    def evaluate(self, result: Any, forward_returns: pd.Series) -> dict[str, Any]:
        from alpha_research.alpha_analyzer import AlphaAnalyzer
        analyzer = AlphaAnalyzer(forward_returns)
        return analyzer.analyze(result)


class MeanReversionExperiment(ResearchExperiment):
    """Test if mean-reversion Z-score predicts forward returns."""

    def __init__(self, window: int = 20) -> None:
        super().__init__(
            name=f"mean_reversion_{window}_alpha",
            description=f"Test {window}-period mean reversion as alpha signal",
        )
        self.window = window

    def hypothesis(self) -> str:
        return f"Price z-score relative to {self.window}-period mean has predictive power for reversals"

    def run_experiment(self, features: pd.DataFrame) -> Any:
        from alpha_research.alpha_signal import MeanReversionAlpha
        return MeanReversionAlpha(self.window).generate(features)

    def evaluate(self, result: Any, forward_returns: pd.Series) -> dict[str, Any]:
        from alpha_research.alpha_analyzer import AlphaAnalyzer
        analyzer = AlphaAnalyzer(forward_returns)
        return analyzer.analyze(result)


class OrderFlowExperiment(ResearchExperiment):
    """Test if order book imbalance predicts short-term price movement."""

    def __init__(self) -> None:
        super().__init__(
            name="order_flow_alpha",
            description="Test order book imbalance as alpha signal",
        )

    def hypothesis(self) -> str:
        return "Order book imbalance (bid/ask volume ratio) predicts next-period returns with IC > 0.03"

    def run_experiment(self, features: pd.DataFrame) -> Any:
        from alpha_research.alpha_signal import OrderFlowAlpha
        return OrderFlowAlpha().generate(features)

    def evaluate(self, result: Any, forward_returns: pd.Series) -> dict[str, Any]:
        from alpha_research.alpha_analyzer import AlphaAnalyzer
        analyzer = AlphaAnalyzer(forward_returns)
        return analyzer.analyze(result)
