#!/usr/bin/env python
"""
validate_alpha.py — Statistical Validation of Alpha Signals
===========================================================
This script performs feature importance and correlation analysis
on the generated feature store results to validate signal predictiveness.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_lab.analysis.feature_importance import FeatureImportanceAnalyzer
from research_lab.analysis.correlation_analysis import CorrelationAnalyzer
from research_lab.reports.alpha_report_generator import AlphaReportGenerator
from alpha_research.alpha_signal import MomentumAlpha, RSICrossAlpha
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    print("=" * 60)
    print("  PHASE 2: ALPHA DISCOVERY VALIDATION")
    print("=" * 60)

    # 1. Load latest feature partition
    feature_dir = Path("data/features/BTC")
    if not feature_dir.exists():
        print("❌ Feature directory not found. Run Phase 1 first.")
        return

    latest_partition = sorted(feature_dir.iterdir())[-1]
    df = pd.read_parquet(latest_partition / "technical.parquet")
    print(f"Loaded {len(df)} rows from {latest_partition.name}")

    # Prepare targets (forward returns)
    df['target'] = df['close'].pct_change(4).shift(-4) # 4-period forward return
    df.dropna(inplace=True)

    # 2. Feature Importance Analysis
    print("\nRunning Feature Importance Analysis...")
    fia = FeatureImportanceAnalyzer()
    features_to_test = ['rsi', 'macd', 'volatility', 'momentum', 'roc']
    importance_df = fia.analyze(df[features_to_test], df['target'], top_n=5)
    print(importance_df.head())

    # 3. Correlation & IC Analysis
    print("\nRunning Correlation & Signal Decay Analysis...")
    ca = CorrelationAnalyzer()
    corr_results = ca.feature_return_correlation(df[features_to_test], df['target'])
    print(corr_results.head())

    # 4. Generate Alpha Research Report
    print("\nGenerating Alpha Research Report...")
    from alpha_research.alpha_signal import MomentumAlpha, RSICrossAlpha
    from alpha_research.alpha_analyzer import AlphaAnalyzer

    # Create signal instances
    mom_sig_obj = MomentumAlpha(12).generate(df)
    rsi_sig_obj = RSICrossAlpha(14).generate(df)

    # Initialize Generator
    report_gen = AlphaReportGenerator(output_dir=Path("research_lab/reports"))
    
    # Generate a combined comparison report
    signals = [mom_sig_obj, rsi_sig_obj]
    # We need to ensure signals are generated and assigned values for the analyzer
    report_text = report_gen.batch_report(signals, df['target'], features=df)
    report_path = report_gen.save(report_text, "alpha_validation_report.md")
    print(f"Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
