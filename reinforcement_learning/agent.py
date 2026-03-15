"""
RL Trading Agent
================
Wrapper around Stable-Baselines3 for training and evaluating RL trading agents.
Supports PPO and A2C algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config.settings import get_settings
from reinforcement_learning.environment import TradingEnv
from utils.logger import get_logger

logger = get_logger(__name__)


class RLTrader:
    """
    Reinforcement learning trading agent using Stable-Baselines3.

    Parameters
    ----------
    train_data : ndarray
        Training feature matrix (n_steps × n_features).
    val_data : ndarray, optional
        Validation feature matrix for evaluation callback.
    algorithm : str
        ``"PPO"`` or ``"A2C"``.
    initial_balance : float
        Starting portfolio balance.

    Example
    -------
    >>> trader = RLTrader(train_features, algorithm="PPO")
    >>> trader.train(total_timesteps=100_000)
    >>> actions = trader.predict(test_features)
    """

    ALGORITHMS = {"PPO": PPO, "A2C": A2C}

    def __init__(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        algorithm: str = "PPO",
        initial_balance: float = 10_000.0,
    ) -> None:
        settings = get_settings()
        self.algorithm_name = algorithm.upper()
        self.save_dir = settings.model.models_dir / "rl"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training environment
        self.train_env = DummyVecEnv(
            [lambda: TradingEnv(train_data, initial_balance=initial_balance)]
        )

        # Validation environment
        self.eval_env = None
        if val_data is not None:
            self.eval_env = DummyVecEnv(
                [lambda: TradingEnv(val_data, initial_balance=initial_balance)]
            )

        # Initialise model
        AlgoClass = self.ALGORITHMS.get(self.algorithm_name)
        if AlgoClass is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use PPO or A2C.")

        self.model = AlgoClass(
            "MlpPolicy",
            self.train_env,
            learning_rate=settings.model.rl_learning_rate,
            verbose=0,
        )
        logger.info("RLTrader initialised with %s", self.algorithm_name)

    def train(
        self,
        total_timesteps: Optional[int] = None,
    ) -> None:
        """
        Train the RL agent.

        Parameters
        ----------
        total_timesteps : int, optional
            Override default from config.
        """
        settings = get_settings()
        timesteps = total_timesteps or settings.model.rl_total_timesteps

        callbacks = []
        if self.eval_env is not None:
            eval_cb = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.save_dir),
                log_path=str(self.save_dir / "logs"),
                eval_freq=timesteps // 10,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_cb)

        logger.info("Training %s for %d timesteps...", self.algorithm_name, timesteps)
        self.model.learn(total_timesteps=timesteps, callback=callbacks)
        self.save()
        logger.info("Training complete.")

    def predict(self, data: np.ndarray) -> list[int]:
        """
        Run the trained agent on new data and return action sequence.

        Returns list of actions (0=HOLD, 1=BUY, 2=SELL).
        """
        env = TradingEnv(data)
        obs, _ = env.reset()
        actions: list[int] = []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            actions.append(int(action))
            done = terminated or truncated

        return actions

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model weights."""
        path = path or self.save_dir / f"{self.algorithm_name.lower()}_trader"
        self.model.save(str(path))
        logger.info("Model saved → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> None:
        """Load model weights."""
        path = path or self.save_dir / f"{self.algorithm_name.lower()}_trader"
        AlgoClass = self.ALGORITHMS[self.algorithm_name]
        self.model = AlgoClass.load(str(path), env=self.train_env)
        logger.info("Model loaded ← %s", path)
