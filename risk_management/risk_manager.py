"""
Risk Manager
============
Professional risk controls for the trading platform.

Features:
- Kelly criterion position sizing
- Stop-loss enforcement
- Maximum drawdown circuit breaker
- Exposure limits
- Risk-adjusted decision gating
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskDecision:
    """Output from the risk manager."""

    approved: bool
    position_size: float  # Fraction of portfolio to risk
    stop_loss_price: float
    reason: str = ""


class RiskManager:
    """
    Enforces risk controls before trade execution.

    Parameters
    ----------
    portfolio_value : float
        Current total portfolio value.
    max_position_pct : float
        Maximum single-position size as fraction of portfolio.
    max_drawdown_pct : float
        Drawdown threshold to trigger circuit breaker.
    stop_loss_pct : float
        Default stop-loss distance (%).
    max_open_positions : int
        Maximum concurrent positions.

    Example
    -------
    >>> rm = RiskManager(portfolio_value=10_000)
    >>> decision = rm.check_risk(entry_price=50000, signal_strength=0.7, current_drawdown=0.05)
    """

    def __init__(
        self,
        portfolio_value: float = 10_000.0,
        max_position_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_open_positions: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct or settings.risk.max_position_pct
        self.max_drawdown_pct = max_drawdown_pct or settings.risk.max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct or settings.risk.stop_loss_pct
        self.max_open_positions = max_open_positions or settings.risk.max_open_positions
        self.current_open_positions = 0
        self.peak_value = portfolio_value

    def update_portfolio(self, current_value: float) -> None:
        """Update the portfolio value and peak tracking."""
        self.portfolio_value = current_value
        self.peak_value = max(self.peak_value, current_value)

    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.portfolio_value) / self.peak_value

    def kelly_criterion(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """
        Compute Kelly fraction for optimal position sizing.

        .. math::
            f^* = \\frac{p}{a} - \\frac{q}{b}

        where p=win_rate, q=1-p, a=avg_loss, b=avg_win.
        Result is clamped to ``[0, max_position_pct]``.
        """
        if avg_loss == 0 or avg_win == 0:
            return self.max_position_pct * 0.5  # Conservative default

        q = 1 - win_rate
        kelly = win_rate / abs(avg_loss) - q / avg_win

        # Half-Kelly for safety (industry practice)
        kelly = kelly * 0.5
        return max(0.0, min(kelly, self.max_position_pct))

    def compute_stop_loss(self, entry_price: float, side: str = "BUY") -> float:
        """Compute stop-loss price based on configured percentage."""
        if side == "BUY":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def check_risk(
        self,
        entry_price: float,
        signal_strength: float = 0.5,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
    ) -> RiskDecision:
        """
        Evaluate whether a trade should be approved.

        Parameters
        ----------
        entry_price : float
            Proposed entry price.
        signal_strength : float
            Signal confidence in [0, 1].
        win_rate / avg_win / avg_loss : float
            Historical strategy stats for Kelly sizing.

        Returns
        -------
        RiskDecision with approval status, position size, and stop-loss.
        """
        # Circuit breaker: max drawdown
        if self.current_drawdown >= self.max_drawdown_pct:
            logger.warning(
                "CIRCUIT BREAKER: Drawdown %.2f%% exceeds limit %.2f%%",
                self.current_drawdown * 100, self.max_drawdown_pct * 100,
            )
            return RiskDecision(
                approved=False, position_size=0.0,
                stop_loss_price=0.0, reason="Max drawdown reached",
            )

        # Position limit check
        if self.current_open_positions >= self.max_open_positions:
            return RiskDecision(
                approved=False, position_size=0.0,
                stop_loss_price=0.0, reason="Max open positions reached",
            )

        # Signal strength gate
        if signal_strength < 0.3:
            return RiskDecision(
                approved=False, position_size=0.0,
                stop_loss_price=0.0, reason="Signal too weak",
            )

        # Position sizing via Kelly
        kelly_size = self.kelly_criterion(win_rate, avg_win, avg_loss)

        # Scale by signal strength
        position_size = kelly_size * signal_strength

        # Compute stop-loss
        stop_loss = self.compute_stop_loss(entry_price)

        logger.info(
            "Risk check PASSED: size=%.4f, stop=%.2f, drawdown=%.2f%%",
            position_size, stop_loss, self.current_drawdown * 100,
        )
        return RiskDecision(
            approved=True,
            position_size=position_size,
            stop_loss_price=stop_loss,
        )
