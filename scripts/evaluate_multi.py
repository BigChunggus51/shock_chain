#!/usr/bin/env python3
"""
Evaluate Multi-Agent vs Single-Agent policies.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_all_configs
from src.environment.supply_chain_env import SupplyChainEnv
from src.environment.multi_agent_env import MultiAgentSupplyChainEnv
from src.agents.multi_agent import MultiAgentTrainer
from stable_baselines3 import PPO

def evaluate_single_agent(model, env_config, reward_config, topology_config, shock_config, n_episodes=5):
    env = SupplyChainEnv(
        env_config=env_config,
        reward_config=reward_config,
        topology_config=topology_config,
        shock_config=shock_config,
        shock_mode="stochastic",
    )
    rewards = []
    bullwhip_ratios = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        rewards.append(ep_reward)
        metrics = env.episode_metrics
        # Bullwhip = var(orders) / var(demand)
        orders = [m.get("factory/order_qty", 0) for m in metrics]
        demand = [m.get("retailer/order_qty", 0) for m in metrics] # Approx demand
        v_orders = np.var(orders)
        v_demand = np.var(demand) if np.var(demand) > 0 else 1
        bullwhip_ratios.append(v_orders / v_demand)
        
    return np.mean(rewards), np.std(rewards), np.mean(bullwhip_ratios)

def evaluate_multi_agent(trainer, n_episodes=5):
    mean_reward = trainer._evaluate_round(round_idx=999, n_episodes=n_episodes)
    return mean_reward

def main():
    configs = load_all_configs("configs/default.yaml", "configs/shocks.yaml")
    
    print("Evaluating Single-Agent (PPO)...")
    try:
        sa_model = PPO.load("outputs/models/final_model")
        sa_mean, sa_std, sa_bw = evaluate_single_agent(
            sa_model, configs["environment"], configs["reward"], 
            configs["topology"], configs["shocks"]
        )
        print(f"Single-Agent -> Reward: {sa_mean:.1f} ± {sa_std:.1f} | Factory Bullwhip Ratio: {sa_bw:.2f}")
    except Exception as e:
        print(f"Single-agent model not found: {e}")
        sa_mean = -9999

    print("\nEvaluating Multi-Agent (IPPO)...")
    try:
        trainer = MultiAgentTrainer(
            env_config=configs["environment"],
            reward_config=configs["reward"],
            topology_config=configs["topology"],
            shock_config=configs["shocks"],
            training_config=configs["training"],
            output_dir="outputs_multi",
        )
        trainer.load_models()
        # Custom evaluation for multi-agent to extract bullwhip
        ma_rewards = []
        ma_bullwhip = []
        for ep in range(5):
            seed = 1000 + ep
            env = trainer._make_base_env(seed)
            obs, _ = env.reset(seed=seed)
            ep_reward = 0
            while env.agents:
                actions = {}
                for agent in env.agents:
                    action, _ = trainer.models[agent].predict(obs[agent], deterministic=True)
                    actions[agent] = action
                obs, rewards, term, trunc, _ = env.step(actions)
                ep_reward += sum(rewards.values())
            ma_rewards.append(ep_reward)
            metrics = env.episode_metrics
            orders = [m.get("factory/order_qty", 0) for m in metrics]
            demand = [m.get("retailer/order_qty", 0) for m in metrics]
            v_orders = np.var(orders)
            v_demand = np.var(demand) if np.var(demand) > 0 else 1
            ma_bullwhip.append(v_orders / v_demand)
        
        ma_mean, ma_std = np.mean(ma_rewards), np.std(ma_rewards)
        ma_bw = np.mean(ma_bullwhip)
        print(f"Multi-Agent  -> Reward: {ma_mean:.1f} ± {ma_std:.1f} | Factory Bullwhip Ratio: {ma_bw:.2f}")
    except Exception as e:
        print(f"Multi-agent evaluation failed: {e}")

    print("\n--- Summary ---")
    print("Multi-Agent architectures with CTDE typically show smoother order patterns")
    print("(lower Bullwhip ratio) because agents aren't trying to centrally plan the")
    print("entire pipeline, leading to less over-reaction to local noise.")

if __name__ == "__main__":
    main()
