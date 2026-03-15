# Research Lab — Jupyter Notebooks

This directory contains interactive research notebooks for alpha discovery
and strategy experimentation.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `alpha_discovery.ipynb` | Test new alpha signals and measure IC/Sharpe |
| `rl_strategy_experiments.ipynb` | Compare RL policies (PPO, A2C, DQN) |
| `microstructure_analysis.ipynb` | Analyse order book features and market impact |
| `feature_importance.ipynb` | Evaluate which features drive predictive power |

## Quick Start

```bash
jupyter notebook research_lab/notebooks/
```

## Usage Pattern

```python
# 1. Load features from the store
from feature_store import FeatureStore
store = FeatureStore()
features = store.load_features(symbol="BTC", feature_set="technical")

# 2. Generate an alpha signal
from alpha_research.alpha_signal import MomentumAlpha
signal = MomentumAlpha(lookback=12).generate(features)

# 3. Evaluate the signal
from alpha_research.alpha_analyzer import AlphaAnalyzer
fwd_returns = features["close"].pct_change().shift(-1)
report = AlphaAnalyzer(fwd_returns).analyze(signal)

# 4. Generate a research report
from research_lab.reports.alpha_report_generator import AlphaReportGenerator
gen = AlphaReportGenerator()
md_report = gen.generate(signal, fwd_returns, features)
gen.save(md_report, "momentum_12_report.md")
```
