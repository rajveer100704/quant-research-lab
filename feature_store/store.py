"""
Feature Store
=============
Date-partitioned, Parquet-backed feature storage for reproducible research.

Directory structure::

    data/features/
        BTC/
            2025-03-20/
                technical.parquet
                orderbook.parquet
                sentiment.parquet
            2025-03-21/
                ...
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Parquet-backed feature store with date-partitioned versioning.

    Parameters
    ----------
    base_dir : Path, optional
        Root directory for feature storage.

    Example
    -------
    >>> store = FeatureStore()
    >>> store.save_features(df, symbol="BTC", feature_set="technical")
    >>> df = store.load_features(symbol="BTC", feature_set="technical")
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.base_dir = base_dir or settings.data.features_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _partition_path(
        self, symbol: str, feature_set: str, date: Optional[str] = None
    ) -> Path:
        """Build the path: ``base_dir/<symbol>/<date>/<feature_set>.parquet``."""
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        return self.base_dir / symbol.upper() / date / f"{feature_set}.parquet"

    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_set: str,
        date: Optional[str] = None,
    ) -> Path:
        """
        Save a feature DataFrame to the store.

        Parameters
        ----------
        df : DataFrame
            Features to persist.
        symbol : str
            Asset symbol (e.g. ``"BTC"``).
        feature_set : str
            Feature group name (e.g. ``"technical"``, ``"orderbook"``).
        date : str, optional
            Date partition (``"YYYY-MM-DD"``). Defaults to today.

        Returns
        -------
        Path to the saved Parquet file.
        """
        path = self._partition_path(symbol, feature_set, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved features → %s (%d rows)", path, len(df))
        return path

    def load_features(
        self,
        symbol: str,
        feature_set: str,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load features from the store.

        Parameters
        ----------
        date : str, optional
            If ``None``, loads the most recent partition.
        """
        if date:
            path = self._partition_path(symbol, feature_set, date)
        else:
            # Find latest date partition
            symbol_dir = self.base_dir / symbol.upper()
            if not symbol_dir.exists():
                logger.warning("No data for %s", symbol)
                return pd.DataFrame()
            date_dirs = sorted(d.name for d in symbol_dir.iterdir() if d.is_dir())
            if not date_dirs:
                return pd.DataFrame()
            path = symbol_dir / date_dirs[-1] / f"{feature_set}.parquet"

        if not path.exists():
            logger.warning("Feature file not found: %s", path)
            return pd.DataFrame()

        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded features ← %s (%d rows)", path, len(df))
        return df

    def list_features(self, symbol: Optional[str] = None) -> list[dict[str, str]]:
        """
        List available feature sets.

        Returns list of dicts with ``symbol``, ``date``, ``feature_set``.
        """
        results: list[dict[str, str]] = []
        search_dirs = (
            [self.base_dir / symbol.upper()] if symbol
            else [d for d in self.base_dir.iterdir() if d.is_dir()]
        )

        for sym_dir in search_dirs:
            if not sym_dir.exists():
                continue
            for date_dir in sorted(sym_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                for pq in date_dir.glob("*.parquet"):
                    results.append({
                        "symbol": sym_dir.name,
                        "date": date_dir.name,
                        "feature_set": pq.stem,
                    })
        return results
