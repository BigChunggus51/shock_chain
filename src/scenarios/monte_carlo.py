"""
Monte Carlo Scenario Runner for What-If Analysis.

After training, freezes the agent's policy and runs N parallel simulations
with different shock sequences to produce risk distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from stable_baselines3 import PPO

from src.environment.supply_chain_env import SupplyChainEnv


def run_scenario(
    model: PPO,
    env_config: dict,
    reward_config: dict,
    topology_config: dict,
    shock_config: dict,
    shock_mode: str = "stochastic",
    scenario_preset: str | None = None,
    seed: int = 42,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run a single episode with the trained policy and collect metrics."""
    env = SupplyChainEnv(
        env_config=env_config, reward_config=reward_config,
        topology_config=topology_config, shock_config=shock_config,
        shock_mode=shock_mode, scenario_preset=scenario_preset, seed=seed,
    )

    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    echelon_names = [e["name"] for e in topology_config["echelons"]]
    metrics = env.episode_metrics
    shock_history = env.shock_injector.shock_history

    # Compute summary statistics
    summary = {
        "seed": seed,
        "total_reward": total_reward,
        "num_shocks": len([s for s in shock_history if s["event"] == "triggered"]),
        "shock_types": [s["type"] for s in shock_history if s["event"] == "triggered"],
    }

    for name in echelon_names:
        inventories = [m.get(f"{name}/inventory", 0) for m in metrics]
        backlogs = [m.get(f"{name}/backlog", 0) for m in metrics]
        fill_rates = [m.get(f"{name}/fill_rate", 1) for m in metrics]

        summary[f"{name}_mean_inventory"] = np.mean(inventories)
        summary[f"{name}_min_inventory"] = np.min(inventories)
        summary[f"{name}_max_backlog"] = np.max(backlogs)
        summary[f"{name}_mean_fill_rate"] = np.mean(fill_rates)
        summary[f"{name}_stockout_days"] = sum(1 for b in backlogs if b > 0)

    env.close()
    return summary


def monte_carlo_analysis(
    model: PPO,
    env_config: dict,
    reward_config: dict,
    topology_config: dict,
    shock_config: dict,
    n_simulations: int = 1000,
    shock_mode: str = "stochastic",
    scenario_preset: str | None = None,
    base_seed: int = 0,
) -> pd.DataFrame:
    """
    Run N simulations with different random seeds and aggregate results.

    Returns a DataFrame where each row is one simulation run's summary.
    """
    results = []

    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"  Scenario {i + 1}/{n_simulations}...")

        summary = run_scenario(
            model=model,
            env_config=env_config, reward_config=reward_config,
            topology_config=topology_config, shock_config=shock_config,
            shock_mode=shock_mode, scenario_preset=scenario_preset,
            seed=base_seed + i,
        )
        results.append(summary)

    return pd.DataFrame(results)


def generate_risk_report(df: pd.DataFrame, echelon_names: list[str]) -> str:
    """Generate a text-based risk distribution report from Monte Carlo results."""
    lines = ["=" * 60, "SUPPLY CHAIN RISK ANALYSIS REPORT", "=" * 60, ""]
    lines.append(f"Simulations run: {len(df)}")
    lines.append(f"Mean total reward: {df['total_reward'].mean():.1f} ± {df['total_reward'].std():.1f}")
    lines.append(f"Mean shocks per episode: {df['num_shocks'].mean():.1f}")
    lines.append("")

    for name in echelon_names:
        lines.append(f"--- {name.upper()} ---")
        lines.append(f"  Mean fill rate:    {df[f'{name}_mean_fill_rate'].mean():.1%}")
        lines.append(f"  Stockout probability: {(df[f'{name}_stockout_days'] > 0).mean():.1%}")
        lines.append(f"  Mean stockout days:   {df[f'{name}_stockout_days'].mean():.1f}")
        lines.append(f"  Worst-case backlog:   {df[f'{name}_max_backlog'].max():.0f} units")
        lines.append("")

    return "\n".join(lines)
