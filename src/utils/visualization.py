"""
Visualization utilities for supply chain simulation analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


# Use a clean, modern style
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)


def plot_inventory_trajectories(
    metrics: list[dict],
    echelon_names: list[str],
    title: str = "Inventory Levels Over Time",
    save_path: str | None = None,
):
    """Plot inventory levels for each echelon over the episode."""
    fig, axes = plt.subplots(len(echelon_names), 1, figsize=(14, 4 * len(echelon_names)),
                             sharex=True)
    if len(echelon_names) == 1:
        axes = [axes]

    steps = [m["step"] for m in metrics]

    for ax, name in zip(axes, echelon_names):
        inv = [m.get(f"{name}/inventory", 0) for m in metrics]
        backlog = [m.get(f"{name}/backlog", 0) for m in metrics]

        ax.fill_between(steps, inv, alpha=0.3, label="Inventory", color="#2196F3")
        ax.plot(steps, inv, linewidth=1.5, color="#2196F3")
        ax.fill_between(steps, [-b for b in backlog], alpha=0.3, label="Backlog", color="#F44336")
        ax.plot(steps, [-b for b in backlog], linewidth=1.5, color="#F44336")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Units")
        ax.set_title(f"{name.capitalize()}", fontweight="bold")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Decision Epoch (Day)")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_bullwhip_analysis(
    metrics: list[dict],
    echelon_names: list[str],
    save_path: str | None = None,
):
    """
    Analyze and plot the Bullwhip Effect by comparing order variance
    amplification across echelons.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Order quantities over time
    for name in echelon_names:
        orders = [m.get(f"{name}/order_qty", 0) for m in metrics]
        steps = [m["step"] for m in metrics]
        ax1.plot(steps, orders, label=name.capitalize(), linewidth=1.5)

    ax1.set_xlabel("Decision Epoch")
    ax1.set_ylabel("Order Quantity")
    ax1.set_title("Order Patterns by Echelon", fontweight="bold")
    ax1.legend()

    # Right: Variance amplification ratio (bar chart)
    variances = {}
    for name in echelon_names:
        orders = [m.get(f"{name}/order_qty", 0) for m in metrics]
        variances[name] = np.var(orders) if orders else 0

    # Bullwhip ratio = variance(orders) / variance(demand)
    # Use retailer demand as the baseline
    demand_var = variances.get(echelon_names[0], 1)
    if demand_var == 0:
        demand_var = 1

    ratios = {name: var / demand_var for name, var in variances.items()}

    colors = ["#4CAF50", "#FF9800", "#F44336"][:len(echelon_names)]
    bars = ax2.bar(
        [n.capitalize() for n in echelon_names],
        [ratios[n] for n in echelon_names],
        color=colors, edgecolor="white", linewidth=1.5,
    )
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="No amplification")
    ax2.set_ylabel("Variance Amplification Ratio")
    ax2.set_title("Bullwhip Effect Measurement", fontweight="bold")
    ax2.legend()

    for bar, ratio in zip(bars, [ratios[n] for n in echelon_names]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{ratio:.2f}x", ha="center", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_reward_breakdown(
    metrics: list[dict],
    save_path: str | None = None,
):
    """Plot cumulative reward over the episode."""
    steps = [m["step"] for m in metrics]
    rewards = [m["reward"] for m in metrics]
    cumulative = np.cumsum(rewards)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(steps, rewards, linewidth=1, alpha=0.6, color="#9C27B0")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    # Rolling average
    window = min(14, len(rewards))
    if window > 1:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(steps[window - 1:], rolling_avg, linewidth=2, color="#9C27B0",
                 label=f"{window}-day rolling avg")
    ax1.set_ylabel("Reward")
    ax1.set_title("Per-Step Reward", fontweight="bold")
    ax1.legend()

    ax2.plot(steps, cumulative, linewidth=2, color="#00BCD4")
    ax2.fill_between(steps, cumulative, alpha=0.2, color="#00BCD4")
    ax2.set_xlabel("Decision Epoch")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Cumulative Reward", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def compare_policies(
    results: dict[str, list[dict]],
    metric_key: str = "reward",
    title: str = "Policy Comparison",
    save_path: str | None = None,
):
    """
    Compare multiple policies on a given metric.

    Parameters
    ----------
    results : dict[str, list[dict]]
        Mapping of policy_name -> list of episode metrics.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for policy_name, metrics in results.items():
        values = [m.get(metric_key, 0) for m in metrics]
        cumulative = np.cumsum(values)
        steps = [m["step"] for m in metrics]
        ax.plot(steps, cumulative, linewidth=2, label=policy_name)

    ax.set_xlabel("Decision Epoch")
    ax.set_ylabel(f"Cumulative {metric_key.capitalize()}")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
