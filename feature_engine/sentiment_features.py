"""
Sentiment Features
==================
Computes sentiment scores from news headlines using VADER sentiment analysis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Computes sentiment scores for news articles using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) provides
    compound, positive, negative, and neutral scores.

    Example
    -------
    >>> analyzer = SentimentAnalyzer()
    >>> scores = analyzer.score_articles(articles)
    """

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> dict[str, float]:
        """
        Score a single text string.

        Returns
        -------
        dict with ``compound``, ``pos``, ``neg``, ``neu`` scores.
        ``compound`` is in ``[-1, 1]`` (most negative to most positive).
        """
        return self._analyzer.polarity_scores(text)

    def score_articles(self, articles: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Score a batch of news articles.

        Parameters
        ----------
        articles : list[dict]
            Each dict must have at least ``"title"`` and optionally ``"body"``.

        Returns
        -------
        DataFrame with columns:
        ``timestamp, title, sentiment_compound, sentiment_pos,
        sentiment_neg, sentiment_neu``.
        """
        logger.info("Scoring sentiment for %d articles", len(articles))
        records: list[dict[str, Any]] = []

        for article in articles:
            title = article.get("title", "")
            body = article.get("body", "")
            # Score on title (higher signal) combined with body snippet
            combined_text = f"{title}. {body[:200]}" if body else title
            scores = self.score_text(combined_text)

            records.append({
                "timestamp": article.get("published_on", ""),
                "title": title,
                "source": article.get("source", ""),
                "sentiment_compound": scores["compound"],
                "sentiment_pos": scores["pos"],
                "sentiment_neg": scores["neg"],
                "sentiment_neu": scores["neu"],
            })

        df = pd.DataFrame(records)
        logger.info(
            "Sentiment scored: mean compound=%.3f",
            df["sentiment_compound"].mean() if not df.empty else 0.0,
        )
        return df

    @staticmethod
    def aggregate_sentiment(
        scores_df: pd.DataFrame, window: str = "1h"
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores into time-windowed averages.

        Parameters
        ----------
        window : str
            Pandas offset alias (``"1h"``, ``"4h"``, ``"1D"``).
        """
        if scores_df.empty:
            return scores_df

        df = scores_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        agg = df[["sentiment_compound", "sentiment_pos", "sentiment_neg"]].resample(
            window
        ).agg(["mean", "count"])

        # Flatten multi-level columns
        agg.columns = ["_".join(col).strip() for col in agg.columns]
        return agg.reset_index()
