"""
Market Data Collector
=====================
Fetches historical and live OHLCV data from Binance REST API.
Stores results in Parquet format for downstream processing.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataCollector:
    """
    Collects OHLCV candlestick data from Binance and persists to Parquet.

    Parameters
    ----------
    symbols : list[str]
        Trading pair symbols (e.g. ``["BTCUSDT", "ETHUSDT"]``).
    interval : str
        Kline interval (``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``).
    data_dir : Path, optional
        Override the default raw-data directory.

    Example
    -------
    >>> collector = MarketDataCollector(symbols=["BTCUSDT"], interval="1h")
    >>> df = collector.fetch_ohlcv("BTCUSDT", limit=500)
    >>> collector.save(df, "BTCUSDT")
    """

    KLINE_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        interval: str = "1h",
        data_dir: Optional[Path] = None,
    ) -> None:
        settings = get_settings()
        self.symbols = symbols or settings.data.default_symbols
        self.interval = interval or settings.data.default_interval
        self.data_dir = data_dir or settings.data.raw_data_dir

        self._client: Optional[Client] = None
        try:
            if settings.binance.api_key:
                self._client = Client(
                    settings.binance.api_key,
                    settings.binance.api_secret,
                    testnet=settings.binance.testnet,
                )
            else:
                # Public endpoints only (no auth needed for klines)
                self._client = Client("", "")
            logger.info("Binance client initialised (testnet=%s)", settings.binance.testnet)
        except Exception as exc:
            logger.warning("Binance client init failed: %s — using public endpoints", exc)
            self._client = Client("", "")

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines for *symbol*.

        Returns a DataFrame with columns:
        ``open_time, open, high, low, close, volume, close_time,
        quote_volume, num_trades, taker_buy_base_volume,
        taker_buy_quote_volume``.
        """
        interval = interval or self.interval
        logger.info("Fetching %d klines for %s @ %s", limit, symbol, interval)

        kwargs: dict = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if start_time:
            kwargs["startTime"] = start_time
        if end_time:
            kwargs["endTime"] = end_time

        try:
            raw = self._client.get_klines(**kwargs)
        except BinanceAPIException as exc:
            logger.error("Binance API error fetching %s: %s", symbol, exc)
            return pd.DataFrame(columns=self.KLINE_COLUMNS[:-1])

        df = pd.DataFrame(raw, columns=self.KLINE_COLUMNS)
        df.drop(columns=["ignore"], inplace=True)

        # Cast numeric columns
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["num_trades"] = df["num_trades"].astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        logger.info("Fetched %d rows for %s", len(df), symbol)
        return df

    def fetch_extended_history(
        self,
        symbol: str,
        total_candles: int = 5000,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch more than 1000 candles by paginating backward through time.
        """
        interval = interval or self.interval
        all_data: list[pd.DataFrame] = []
        fetched = 0
        end_time = None

        while fetched < total_candles:
            batch_size = min(1000, total_candles - fetched)
            df = self.fetch_ohlcv(symbol, interval=interval, limit=batch_size, end_time=end_time)
            if df.empty:
                break
            all_data.append(df)
            fetched += len(df)
            # Move end_time to just before the earliest candle in this batch
            end_time = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
            time.sleep(0.2)  # Rate-limit courtesy

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined.sort_values("open_time", inplace=True)
        combined.drop_duplicates(subset=["open_time"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        logger.info("Extended fetch: %d total rows for %s", len(combined), symbol)
        return combined

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, symbol: str, fmt: str = "parquet") -> Path:
        """
        Save DataFrame to ``data/raw/<symbol>/``.

        Parameters
        ----------
        fmt : str
            ``"parquet"``, ``"csv"``, or ``"json"``.
        """
        symbol_dir = self.data_dir / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.upper()}_{self.interval}_{timestamp}"

        if fmt == "parquet":
            path = symbol_dir / f"{filename}.parquet"
            df.to_parquet(path, index=False, engine="pyarrow")
        elif fmt == "csv":
            path = symbol_dir / f"{filename}.csv"
            df.to_csv(path, index=False)
        elif fmt == "json":
            path = symbol_dir / f"{filename}.json"
            df.to_json(path, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info("Saved %s → %s", symbol, path)
        return path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def collect_all(self, limit: int = 1000) -> dict[str, pd.DataFrame]:
        """Fetch and save OHLCV for all configured symbols."""
        results: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, limit=limit)
            if not df.empty:
                self.save(df, symbol)
            results[symbol] = df
        return results
