#!/usr/bin/env python
"""
validate_platform.py — End-to-End System Validation
===================================================
This script executes the core pipelines of the AI Quant Trading Research Platform
to verify system integrity and generate proof-of-work artifacts.

Workflow:
1. Data Pipeline & Features
2. Strategy Backtest & Simulation
3. ML Reinforcement Learning Training
4. Reporting & Results
"""

import subprocess
import json
import os
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("=" * 60)
print("  AI QUANT PLATFORM — SYSTEM VALIDATION")
print("=" * 60)

def run_command(cmd, log_name):
    print(f"\n🚀 Running: {cmd}")
    # Ensure current directory is in PYTHONPATH for subprocesses
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True, env=env)
        print(f"   ✅ {log_name} completed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {log_name} failed.")
        print(f"   Error: {e.stderr}")
        return None

# Step 1 — Run Data Pipeline
print("\n[STEP 1] Data Ingestion + Feature Engineering")
run_command(
    "python run_pipeline.py --symbol BTCUSDT --interval 1h --limit 1000",
    "Data Pipeline"
)

# Step 2 — Run Strategy Backtest (with JSON output)
print("\n[STEP 2] Strategy Simulation & Backtesting")
backtest_json = "results/backtest_results.json"
run_command(
    f"python run_backtest.py --strategy momentum --limit 2000 --output {backtest_json}",
    "Strategy Backtest"
)

# Step 3 — Train RL Agent (Reduced timesteps for quick validation)
print("\n[STEP 3] Reinforcement Learning Training")
run_command(
    "python train_rl.py --algorithm PPO --timesteps 10000",
    "RL Training"
)

# Step 4 — Evaluate Results
print("\n[STEP 4] Results Evaluation")
results_path = Path(backtest_json)

if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    print("-" * 40)
    print("  VALIDATION METRICS")
    print("-" * 40)
    print(f"  Strategy      : {results.get('strategy', 'N/A')}")
    print(f"  Total Return  : {results.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe Ratio  : {results.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown  : {results.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Trades        : {results.get('n_trades', 0)}")
    print("-" * 40)
    
    # Validation logic
    if results.get('n_trades', 0) > 0:
        print("  ✅ PASS: Simulator generated trades.")
    else:
        print("  ⚠️ WARNING: No trades generated during backtest.")
        
    if abs(results.get('total_return_pct', 0)) < 1000: # Sanity check for realistic values
        print("  ✅ PASS: Return metrics are within realistic bounds.")
else:
    print("  ❌ ERROR: Backtest results file not found.")

print("\n" + "=" * 60)
print("  SYSTEM VALIDATION COMPLETE")
print("=" * 60)
print(f"  Artifacts saved to: results/")
print(f"  Feature store partition: data/features/")
print("=" * 60)
