#!/usr/bin/env python3
"""
Entry point: Evaluate a trained policy against heuristic baselines.

Runs the DRL agent and all baselines on the same environment and
generates comparison plots and a risk analysis report.

Usage:
    python scripts/evaluate.py --model outputs/models/final_model.zip
    python scripts/evaluate.py --model outputs/models/best/best_model.zip --episodes 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO

from src.environment.supply_chain_env import SupplyChainEnv
from src.agents.heuristic_baselines import BaseStockPolicy, SsPolicy, BeerGamePolicy, RandomPolicy
from src.scenarios.monte_carlo import run_scenario, monte_carlo_analysis, generate_risk_report
from src.utils.config_loader import load_all_configs
from src.utils.visualization import compare_policies, plot_bullwhip_analysis


def evaluate_heuristic(policy, env, n_episodes=20):
    """Evaluate a heuristic policy over multiple episodes."""
    all_rewards = []
    all_metrics = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 5000)
        total_reward = 0
        done = False

        while not done:
            action = policy.get_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        all_rewards.append(total_reward)
        if ep == 0:
            all_metrics = env.episode_metrics

    return np.mean(all_rewards), np.std(all_rewards), all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Supply Chain DRL Agent")
    parser.add_argument("--model", required=True, help="Path to trained model .zip")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--shocks", default="configs/shocks.yaml")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per policy")
    parser.add_argument("--output", default="outputs/eval_results", help="Results directory")
    args = parser.parse_args()

    configs = load_all_configs(args.config, args.shocks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    echelon_names = [e["name"] for e in configs["topology"]["echelons"]]

    # Load trained model
    print("Loading trained model...")
    model = PPO.load(args.model)

    # Create evaluation environment
    env = SupplyChainEnv(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        shock_mode="stochastic",
        seed=9999,
    )

    # Evaluate DRL agent
    print(f"\nEvaluating PPO agent ({args.episodes} episodes)...")
    drl_rewards = []
    drl_metrics = None
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep + 5000)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        drl_rewards.append(total_reward)
        if ep == 0:
            drl_metrics = env.episode_metrics

    # Evaluate baselines
    policies = {
        "Base-Stock": BaseStockPolicy(echelon_names, configs["environment"]),
        "(s,S) Policy": SsPolicy(echelon_names, configs["environment"]),
        "Beer Game": BeerGamePolicy(echelon_names, configs["environment"]),
        "Random": RandomPolicy(echelon_names, configs["environment"]),
    }

    results = {"PPO Agent": drl_metrics}
    print("\n" + "=" * 60)
    print(f"{'Policy':<20} {'Mean Reward':>12} {'Std':>10}")
    print("-" * 42)
    print(f"{'PPO Agent':<20} {np.mean(drl_rewards):>12.1f} {np.std(drl_rewards):>10.1f}")

    for name, policy in policies.items():
        mean_r, std_r, metrics = evaluate_heuristic(policy, env, args.episodes)
        results[name] = metrics
        print(f"{name:<20} {mean_r:>12.1f} {std_r:>10.1f}")

    print("=" * 60)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    compare_policies(results, save_path=str(output_dir / "policy_comparison.png"))

    if drl_metrics:
        plot_bullwhip_analysis(drl_metrics, echelon_names,
                               save_path=str(output_dir / "bullwhip_analysis.png"))

    # Monte Carlo risk analysis
    print(f"\nRunning Monte Carlo analysis (100 simulations)...")
    mc_df = monte_carlo_analysis(
        model=model,
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        n_simulations=100,
        base_seed=10000,
    )
    mc_df.to_csv(str(output_dir / "monte_carlo_results.csv"), index=False)

    report = generate_risk_report(mc_df, echelon_names)
    print("\n" + report)
    with open(output_dir / "risk_report.txt", "w") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}/")
    env.close()


if __name__ == "__main__":
    main()
