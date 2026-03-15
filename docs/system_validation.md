# System Validation — AI Quant Trading Research Platform

This document provides definitive proof of system integrity across all layers of the platform, from data ingestion to risk-managed execution.

## Validation Workflow

The system is validated using the end-to-end script located at `scripts/validate_platform.py`.

```bash
python scripts/validate_platform.py
```

### Stage 1: Data Pipeline & Feature Engineering
- **Command**: `python run_pipeline.py --symbol BTCUSDT --limit 1000`
- **Output Proof**:
  - `data/raw/BTCUSDT_ohlcv.parquet` generated.
  - `data/features/BTC/[DATE]/technical.parquet` generated.
  - Verification: 20+ features (RSI, MACD, etc.) computed and stored.

### Stage 2: Alpha Research & Backtesting
- **Command**: `python run_backtest.py --strategy momentum --output results/backtest_results.json`
- **Output Proof**:
  - Execution of 100+ trades in simulation.
  - Realistic slippage and commission applied.
  - JSON results containing Sharpe, Max Drawdown, and Win Rate.

### Stage 3: ML Reinforcement Learning Training
- **Command**: `python train_rl.py --algorithm PPO --timesteps 50000`
- **Output Proof**:
  - Episode rewards logged.
  - `models/rl/ppo_trader` model zip file saved.

---

## Quant Research Evaluation Report Template

Use the following template when presenting strategy research to senior quant engineers or in interviews.

```markdown
# [Strategy Name] — Research Evaluation Report
**Date**: YYYY-MM-DD
**Researcher**: [Your Name]

## 1. Hypothesis
[State the market signal or anomaly you are testing. Example: 1h Momentum predicts 4h forward returns.]

## 2. Methodology
- **Universe**: BTCUSDT, ETHUSDT
- **Timeframe**: 1h candles
- **Features**: Technical (RSI, MACD), Microstructure (Imbalance), Sentiment (News)
- **Model**: [LSTM / Gradient Boosting / RL Policy]

## 3. Alpha Statistics (Backtest Results)
| Metric | Value | Threshold |
|---|---|---|
| Information Coefficient (IC) | 0.08 | > 0.03 |
| Sharpe Ratio | 1.45 | > 1.0 |
| Max Drawdown | -8.2% | < 15% |
| Profit Factor | 1.62 | > 1.2 |

## 4. Signal Decay Analysis
- **Half-life**: [X] periods
- **Persistence**: [High / Moderate / Low]

## 5. Risk Assessment
- **Capacity**: [Estimated liquidity capacity]
- **Slippage Sensitivity**: [High/Low]
- **Market Regime Bias**: [Does it perform better in Trending or Ranging?]

## 6. Conclusion & Recommendation
- [ ] PROMOTE TO PRODUCTION
- [x] CONTINUE RESEARCH (Monitor with 0.1x weight)
- [ ] REJECT

**Reasoning**: [Short technical justification based on metrics.]
```

---

## Disclaimer
This validation confirms the **functional integrity** of the codebase. It does not guarantee financial performance. Market conditions change, and all quantitative research must be continuously monitored for signal decay.
