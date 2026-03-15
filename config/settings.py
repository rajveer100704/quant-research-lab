"""
Platform Configuration
======================
Centralized settings using Pydantic BaseSettings.
Configuration is loaded from environment variables and .env files.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class BinanceConfig(BaseSettings):
    """Binance exchange API configuration."""

    api_key: str = Field(default="", description="Binance API key")
    api_secret: str = Field(default="", description="Binance API secret")
    testnet: bool = Field(default=True, description="Use Binance testnet")
    base_url: str = Field(
        default="https://testnet.binance.vision",
        description="Binance REST API base URL",
    )
    ws_url: str = Field(
        default="wss://testnet.binance.vision/ws",
        description="Binance WebSocket URL",
    )

    model_config = {"env_prefix": "BINANCE_"}


class DataConfig(BaseSettings):
    """Data storage and ingestion configuration."""

    raw_data_dir: Path = Field(
        default=Path("data/raw"), description="Directory for raw market data"
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"), description="Directory for processed data"
    )
    features_dir: Path = Field(
        default=Path("data/features"), description="Feature store directory"
    )
    default_symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT"], description="Default trading pairs"
    )
    default_interval: str = Field(
        default="1h", description="Default kline interval"
    )
    ohlcv_limit: int = Field(
        default=1000, description="Default OHLCV fetch limit"
    )

    model_config = {"env_prefix": "DATA_"}


class ModelConfig(BaseSettings):
    """Machine learning model configuration."""

    lstm_hidden_size: int = Field(default=128, description="LSTM hidden layer size")
    lstm_num_layers: int = Field(default=2, description="Number of LSTM layers")
    lstm_sequence_length: int = Field(default=60, description="Input sequence length")
    lstm_learning_rate: float = Field(default=1e-3, description="Learning rate")
    lstm_epochs: int = Field(default=50, description="Training epochs")
    lstm_batch_size: int = Field(default=32, description="Batch size")

    rl_algorithm: str = Field(default="PPO", description="RL algorithm (PPO, A2C)")
    rl_total_timesteps: int = Field(default=100_000, description="Total training steps")
    rl_learning_rate: float = Field(default=3e-4, description="RL learning rate")

    models_dir: Path = Field(
        default=Path("models/"), description="Saved models directory"
    )

    model_config = {"env_prefix": "MODEL_"}


class RiskConfig(BaseSettings):
    """Risk management configuration."""

    max_position_pct: float = Field(
        default=0.1, description="Max position size as fraction of portfolio"
    )
    max_drawdown_pct: float = Field(
        default=0.15, description="Max drawdown before circuit breaker"
    )
    stop_loss_pct: float = Field(
        default=0.02, description="Default stop-loss percentage"
    )
    max_open_positions: int = Field(
        default=3, description="Max concurrent open positions"
    )
    risk_free_rate: float = Field(
        default=0.04, description="Annual risk-free rate for Sharpe calculation"
    )

    model_config = {"env_prefix": "RISK_"}


class Settings(BaseSettings):
    """Root platform settings aggregating all sub-configs."""

    project_name: str = "AI Quant Trading Research Platform"
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    experiments_dir: Path = Field(
        default=Path("experiments/"), description="Experiment tracking directory"
    )

    model_config = {"env_prefix": "PLATFORM_", "env_file": ".env", "extra": "ignore"}


# --- Singleton ---
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_trading_config(path: Optional[str] = None) -> dict:
    """
    Load trading configuration from YAML file.

    Merges YAML values into the Settings singleton for experiment
    reproducibility. Returns the raw YAML dict.

    Parameters
    ----------
    path : str, optional
        Path to the YAML file. Defaults to ``config/trading.yaml``.
    """
    import yaml

    config_path = Path(path) if path else Path("config/trading.yaml")
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Merge into Settings
    settings = get_settings()

    if "risk" in config:
        for key, value in config["risk"].items():
            if hasattr(settings.risk, key):
                setattr(settings.risk, key, value)

    if "models" in config:
        ml = config["models"]
        if "lstm" in ml:
            for key, value in ml["lstm"].items():
                attr = f"lstm_{key}"
                if hasattr(settings.model, attr):
                    setattr(settings.model, attr, value)
        if "rl" in ml:
            for key, value in ml["rl"].items():
                attr = f"rl_{key}"
                if hasattr(settings.model, attr):
                    setattr(settings.model, attr, value)

    return config
