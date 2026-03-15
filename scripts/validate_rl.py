#!/usr/bin/env python
"""
validate_rl.py — Reinforcement Learning Training Validation
===========================================================
Trains the RL agent and generates a learning reward curve.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reinforcement_learning.train import prepare_training_data
from reinforcement_learning.agent import RLTrader
from config.settings import get_settings

def main():
    print("=" * 60)
    print("  PHASE 3: ML EXPERIMENTATION VALIDATION")
    print("=" * 60)

    symbol = "BTCUSDT"
    timesteps = 50000
    
    print(f"Preparing data for {symbol}...")
    train_data, val_data = prepare_training_data(symbol, limit=2000)
    
    print(f"Initializing RL Trader (PPO)...")
    trader = RLTrader(
        train_data=train_data,
        val_data=val_data,
        algorithm="PPO",
        initial_balance=10000.0,
    )
    
    print(f"Training for {timesteps} steps...")
    trader.train(total_timesteps=timesteps)
    
    # Generate Training Curve from EvalCallback logs
    print("\nGenerating Training Reward Curve...")
    log_path = Path("models/rl/logs/evaluations.npz")
    
    if log_path.exists():
        data = np.load(log_path)
        # evaluations.npz contains: timesteps, results, ep_lengths, [is_success]
        # results is (n_evaluations, n_eval_episodes)
        eval_timesteps = data['timesteps']
        eval_results = data['results']
        mean_rewards = np.mean(eval_results, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(eval_timesteps, mean_rewards, marker='o', linestyle='-', color='#00aaff', label='Mean Eval Reward')
        plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
        plt.title('RL Training Reward Curve')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Portfolio Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_plot = Path("validation_plots/rl_training_curve.png")
        plt.savefig(output_plot)
        print(f"Reward curve saved -> {output_plot}")
    else:
        print("⚠️ No evaluation logs found to plot.")

    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
