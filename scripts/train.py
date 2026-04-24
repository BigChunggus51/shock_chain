#!/usr/bin/env python3
"""
Entry point: Train a single-agent PPO policy on the supply chain environment.

Usage:
    python scripts/train.py
    python scripts/train.py --timesteps 1000000
    python scripts/train.py --config configs/default.yaml --shocks configs/shocks.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_all_configs
from src.agents.single_agent import SingleAgentTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Supply Chain DRL Agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to env config")
    parser.add_argument("--shocks", default="configs/shocks.yaml", help="Path to shock config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel training environments")
    parser.add_argument("--shock-mode", default="stochastic", choices=["stochastic", "deterministic"])
    parser.add_argument("--scenario", default=None, help="Named scenario preset")
    args = parser.parse_args()

    # Load configs
    configs = load_all_configs(args.config, args.shocks)

    print("=" * 60)
    print("SUPPLY CHAIN SHOCK SIMULATION — TRAINING")
    print("=" * 60)
    print(f"Algorithm: {configs['training'].get('algorithm', 'PPO')}")
    print(f"Shock mode: {args.shock_mode}")
    print(f"Scenario: {args.scenario or 'None (random shocks)'}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Initialize trainer
    trainer = SingleAgentTrainer(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        training_config=configs["training"],
        output_dir=args.output,
        shock_mode=args.shock_mode,
        scenario_preset=args.scenario,
        n_envs=args.n_envs,
    )

    # Train
    try:
        model = trainer.train(total_timesteps=args.timesteps)
        print("\nTraining complete!")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
