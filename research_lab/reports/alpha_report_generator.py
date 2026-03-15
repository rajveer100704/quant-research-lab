"""
Alpha Report Generator
======================
Generates structured research reports for alpha signal evaluation.

Output: Markdown reports saved to ``research_lab/reports/generated/``.

Example report::

    ================================================================
    ALPHA SIGNAL REPORT — momentum_12
    Date: 2025-03-14T20:30:00
    ================================================================

    Summary
    -------
    Signal Name      : momentum_12
    Rating           : STRONG
    Recommendation   : PROMOTE TO STRATEGY ENGINE

    Performance Metrics
    -------------------
    Information Coefficient : 0.0850
    Sharpe Ratio           : 1.4500
    Win Rate               : 58.20%
    Stability              : 0.7200

    Signal Decay
    ------------
    Half-Life              : 8 periods
    Persistence            : MODERATE

    Conclusion
    ----------
    The momentum_12 signal shows strong predictive power with
    consistent IC across rolling windows. Recommended for
    integration into the signal fusion engine with weight 0.30.
    ================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from alpha_research.alpha_signal import AlphaSignal, AlphaGenerator
from alpha_research.alpha_analyzer import AlphaAnalyzer
from research_lab.analysis.correlation_analysis import CorrelationAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)


class AlphaReportGenerator:
    """
    Generate comprehensive alpha research reports.

    Parameters
    ----------
    output_dir : Path, optional
        Directory for saved reports.

    Example
    -------
    >>> gen = AlphaReportGenerator()
    >>> report = gen.generate(signal, forward_returns, features)
    >>> gen.save(report, "momentum_12_report.md")
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("research_lab/reports/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        signal: AlphaSignal,
        forward_returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a full research report for an alpha signal.

        Returns the report as a Markdown string.
        """
        analyzer = AlphaAnalyzer(forward_returns)
        metrics = analyzer.analyze(signal)

        # Signal decay analysis
        decay_text = "Not computed (requires raw returns series)."
        half_life = "N/A"
        if features is not None and "close" in features.columns:
            ca = CorrelationAnalyzer()
            raw_returns = features["close"].pct_change()
            decay_df = ca.signal_decay(signal.values, raw_returns, max_lag=24)
            if not decay_df.empty:
                # Find half-life
                initial_ic = abs(decay_df.iloc[0]["ic"])
                hl_rows = decay_df[decay_df["ic"].abs() < initial_ic / 2]
                half_life = str(hl_rows.iloc[0]["lag"]) if not hl_rows.empty else ">24"
                persistence = "HIGH" if half_life == ">24" else (
                    "MODERATE" if int(half_life) > 6 else "LOW"
                )
                decay_text = f"Half-Life: {half_life} periods | Persistence: {persistence}"

        # Rating and recommendation
        rating = metrics["rating"]
        if rating == "STRONG":
            recommendation = "PROMOTE TO STRATEGY ENGINE"
            weight_suggestion = 0.30
        elif rating == "MODERATE":
            recommendation = "CONTINUE MONITORING — potential if combined"
            weight_suggestion = 0.15
        else:
            recommendation = "DO NOT USE — insufficient predictive power"
            weight_suggestion = 0.0

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")

        report = f"""# Alpha Signal Report — {signal.name}

**Generated**: {now}

---

## Summary

| Field | Value |
|---|---|
| Signal Name | `{signal.name}` |
| Rating | **{rating}** |
| Recommendation | {recommendation} |
| Suggested Weight | {weight_suggestion} |

---

## Performance Metrics

| Metric | Value |
|---|---|
| Information Coefficient (IC) | {metrics['ic']:.4f} |
| Sharpe Ratio | {metrics['sharpe']:.4f} |
| Win Rate | {metrics['win_rate']:.2f}% |
| Stability | {metrics['stability']:.4f} |

---

## Signal Decay

{decay_text}

---

## Signal Metadata

```
{signal.metadata}
```

---

## Conclusion

"""
        if rating == "STRONG":
            report += (
                f"The **{signal.name}** signal shows **strong** predictive power with "
                f"IC={metrics['ic']:.4f} and Sharpe={metrics['sharpe']:.2f}. "
                f"Signal stability across rolling windows is {metrics['stability']:.2f}, "
                f"indicating consistent alpha. "
                f"**Recommended for integration** into the signal fusion engine "
                f"with suggested weight {weight_suggestion}."
            )
        elif rating == "MODERATE":
            report += (
                f"The **{signal.name}** signal shows **moderate** predictive power. "
                f"IC={metrics['ic']:.4f} suggests some information content, "
                f"but Sharpe={metrics['sharpe']:.2f} is below the 1.0 threshold. "
                f"Consider combining with other signals for improved performance."
            )
        else:
            report += (
                f"The **{signal.name}** signal shows **weak** predictive power. "
                f"IC={metrics['ic']:.4f} and Sharpe={metrics['sharpe']:.2f} are below "
                f"minimum thresholds. **Not recommended** for production use."
            )

        report += f"\n\n---\n\n*Report generated by AI Quant Trading Research Platform*\n"

        logger.info("Report generated for %s: %s → %s", signal.name, rating, recommendation)
        return report

    def save(self, report: str, filename: Optional[str] = None) -> Path:
        """Save report to disk as Markdown."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"alpha_report_{timestamp}.md"
        path = self.output_dir / filename
        path.write_text(report, encoding="utf-8")
        logger.info("Report saved → %s", path)
        return path

    def batch_report(
        self,
        signals: list[AlphaSignal],
        forward_returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a combined report comparing multiple signals.
        """
        analyzer = AlphaAnalyzer(forward_returns)
        ranking = analyzer.rank_signals(signals)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
        report = f"# Alpha Signal Comparison Report\n\n**Generated**: {now}\n\n---\n\n"
        report += "## Signal Ranking\n\n"
        report += "| Rank | Signal | IC | Sharpe | Win Rate | Stability | Rating |\n"
        report += "|------|--------|----|--------|----------|-----------|--------|\n"

        for i, row in ranking.iterrows():
            report += (
                f"| {i + 1} | `{row['name']}` | {row['ic']:.4f} | "
                f"{row['sharpe']:.2f} | {row['win_rate']:.1f}% | "
                f"{row['stability']:.2f} | {row['rating']} |\n"
            )

        report += "\n---\n\n"

        # Individual reports
        for signal in signals:
            report += self.generate(signal, forward_returns, features)
            report += "\n\n---\n\n"

        return report
