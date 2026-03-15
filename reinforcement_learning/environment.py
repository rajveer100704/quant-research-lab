"""
Trading Environment
===================
Gymnasium-compatible trading environment for reinforcement learning.

Observation Space
-----------------
A vector of normalised market indicators + position state:
``[close, volume, rsi, macd, volatility, sma_20, ema_20,
  position_flag, unrealised_pnl, portfolio_value]``

Action Space
------------
- ``0`` = HOLD
- ``1`` = BUY
- ``2`` = SELL

Reward Function
---------------
Step reward based on portfolio change with drawdown penalty:
``reward = Δportfolio_value - drawdown_penalty``
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.logger import get_logger

logger = get_logger(__name__)


class TradingEnv(gym.Env):
    """
    Gymnasium trading environment for RL agent training.

    Parameters
    ----------
    df : ndarray of shape ``(n_steps, n_features)``
        Feature matrix (must include ``close`` as first column).
    initial_balance : float
        Starting cash balance.
    commission : float
        Trading commission rate (e.g. 0.001 = 0.1%).
    max_steps : int, optional
        Maximum episode length. Defaults to data length.

    Training Data
    -------------
    The environment expects pre-computed features from the feature engine:
    - Historical OHLCV data (close prices for PnL)
    - Technical indicators (RSI, MACD, volatility)
    - Order book features (imbalance, spread)
    These are concatenated into a feature matrix before training.
    """

    metadata = {"render_modes": ["human"]}

    # Action constants
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df: np.ndarray,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data = df.astype(np.float32)
        self.n_steps = len(df)
        self.n_features = df.shape[1]
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps or self.n_steps - 1

        # Observation: market features + [position, unrealised_pnl, portfolio_value]
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # State variables (initialised in reset)
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Units held
        self.entry_price = 0.0
        self.portfolio_value = initial_balance
        self.peak_value = initial_balance
        self.trade_log: list[dict[str, Any]] = []

    def _get_price(self) -> float:
        """Get current close price (first column)."""
        return float(self.data[self.current_step, 0])

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        market = self.data[self.current_step]
        price = self._get_price()
        unrealised_pnl = (price - self.entry_price) * self.position if self.position > 0 else 0.0
        self.portfolio_value = self.balance + self.position * price

        position_flag = 1.0 if self.position > 0 else 0.0
        extra = np.array(
            [position_flag, unrealised_pnl / self.initial_balance, self.portfolio_value / self.initial_balance],
            dtype=np.float32,
        )
        return np.concatenate([market, extra])

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.trade_log = []
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.

        Returns ``(obs, reward, terminated, truncated, info)``.
        """
        price = self._get_price()
        prev_value = self.portfolio_value

        # Execute action
        if action == self.BUY and self.position == 0:
            # Buy with all available balance
            cost = self.balance * (1 - self.commission)
            self.position = cost / price
            self.entry_price = price
            self.balance = 0.0
            self.trade_log.append({"step": self.current_step, "action": "BUY", "price": price})

        elif action == self.SELL and self.position > 0:
            # Sell entire position
            revenue = self.position * price * (1 - self.commission)
            self.balance = revenue
            self.trade_log.append({
                "step": self.current_step, "action": "SELL", "price": price,
                "pnl": revenue - self.entry_price * self.position,
            })
            self.position = 0.0
            self.entry_price = 0.0

        # Update portfolio value
        self.portfolio_value = self.balance + self.position * price
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Reward: portfolio change + drawdown penalty
        portfolio_change = (self.portfolio_value - prev_value) / self.initial_balance
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        reward = portfolio_change - 0.5 * drawdown

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "drawdown": drawdown,
            "n_trades": len(self.trade_log),
        }

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Print current state."""
        price = self._get_price() if self.current_step < self.n_steps else 0.0
        print(
            f"Step {self.current_step}: price={price:.2f} "
            f"balance={self.balance:.2f} position={self.position:.4f} "
            f"portfolio={self.portfolio_value:.2f}"
        )
