#!/bin/bash
# ============================================================
# AI Quant Trading Research Platform — Demo Runner
# Automatically executes the full research-to-visualization pipeline.
# ============================================================

# Exit on error
set -e

echo "============================================================"
echo "  🚀 Starting AI Quant Trading Research Platform Demo"
echo "============================================================"

# 1. Run the data pipeline (fetch OHLCV data)
echo -e "\n[1/4] 📊 Ingesting Market Data..."
python run_pipeline.py --symbol BTCUSDT --limit 2000 --interval 1h

# 2. Train Reinforcement Learning Model
echo -e "\n[2/4] 🤖 Training RL Agent (PPO)..."
python train_rl.py --algorithm PPO --timesteps 50000 --symbol BTCUSDT

# 3. Run Backtest Simulation
echo -e "\n[3/4] 🎯 Running Backtest Simulation..."
python run_backtest.py --strategy momentum --limit 2000 --symbol BTCUSDT

# 4. Launch Dashboard
echo -e "\n[4/4] 🖥️ Launching Research Dashboard..."
echo "Access the dashboard at http://localhost:8501"
streamlit run dashboard/app.py
