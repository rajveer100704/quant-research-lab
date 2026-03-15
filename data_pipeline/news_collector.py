"""
News Collector
==============
Collects crypto-related news headlines for sentiment analysis.
Uses free public APIs (CryptoCompare News API).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# CryptoCompare public news endpoint (no API key required for basic usage)
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"


class NewsCollector:
    """
    Fetches crypto news headlines from CryptoCompare.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to store collected news data.
    categories : list[str], optional
        News categories to filter (e.g. ``["BTC", "ETH", "Trading"]``).

    Example
    -------
    >>> collector = NewsCollector(categories=["BTC"])
    >>> articles = collector.fetch(limit=20)
    >>> collector.save(articles)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        categories: Optional[list[str]] = None,
    ) -> None:
        settings = get_settings()
        self.data_dir = data_dir or settings.data.raw_data_dir / "news"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.categories = categories or ["BTC", "ETH", "Trading"]

    def fetch(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Fetch latest news articles.

        Returns a list of dicts with keys:
        ``title, body, source, url, published_on, categories, sentiment``.
        """
        params: dict[str, Any] = {"lang": "EN"}
        if self.categories:
            params["categories"] = ",".join(self.categories)

        logger.info("Fetching news from CryptoCompare (categories=%s)", self.categories)

        try:
            response = requests.get(CRYPTOCOMPARE_NEWS_URL, params=params, timeout=15)
            response.raise_for_status()
            raw_data = response.json()
        except requests.RequestException as exc:
            logger.error("News fetch failed: %s", exc)
            return []

        articles: list[dict[str, Any]] = []
        for item in raw_data.get("Data", [])[:limit]:
            articles.append({
                "title": item.get("title", ""),
                "body": item.get("body", "")[:500],  # Truncate for storage
                "source": item.get("source_info", {}).get("name", "unknown"),
                "url": item.get("url", ""),
                "published_on": datetime.fromtimestamp(
                    item.get("published_on", 0), tz=timezone.utc
                ).isoformat(),
                "categories": item.get("categories", ""),
            })

        logger.info("Fetched %d articles", len(articles))
        return articles

    def save(self, articles: list[dict[str, Any]]) -> Path:
        """Save fetched articles to JSON."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.data_dir / f"news_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        logger.info("News saved → %s (%d articles)", path, len(articles))
        return path
