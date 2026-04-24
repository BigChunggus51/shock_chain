#!/usr/bin/env python3
"""
Entry point: Train multi-agent IPPO policies on the supply chain environment.

Each echelon (Retailer, Warehouse, Factory) gets its own PPO agent.
Training uses round-robin: train one agent while others use their latest policy.

Usage:
    python scripts/train_multi.py
    python scripts/train_multi.py --timesteps 200000 --rounds 10
    python scripts/train_multi.py --no-demand-sharing  # disable anti-Bullwhip
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_all_configs
from src.agents.multi_agent import MultiAgentTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent Supply Chain DRL")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to env config")
    parser.add_argument("--shocks", default="configs/shocks.yaml", help="Path to shock config")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps per agent (default: from config)")
    parser.add_argument("--rounds", type=int, default=5, help="Round-robin training rounds")
    parser.add_argument("--output", default="outputs_multi", help="Output directory")
    parser.add_argument("--shock-mode", default="stochastic",
                        choices=["stochastic", "deterministic"])
    parser.add_argument("--scenario", default=None, help="Named scenario preset")
    parser.add_argument("--no-demand-sharing", action="store_true",
                        help="Disable demand signal sharing (increases Bullwhip)")
    parser.add_argument("--no-capacity-sharing", action="store_true",
                        help="Disable capacity signal sharing")
    args = parser.parse_args()

    configs = load_all_configs(args.config, args.shocks)

    print("=" * 60)
    print("SUPPLY CHAIN SHOCK SIMULATION — MULTI-AGENT TRAINING")
    print("=" * 60)
    print(f"Shock mode: {args.shock_mode}")
    print(f"Scenario: {args.scenario or 'None (random shocks)'}")
    print(f"Demand sharing: {not args.no_demand_sharing}")
    print(f"Capacity sharing: {not args.no_capacity_sharing}")
    print(f"Output: {args.output}")
    print("=" * 60)

    trainer = MultiAgentTrainer(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        training_config=configs["training"],
        output_dir=args.output,
        shock_mode=args.shock_mode,
        scenario_preset=args.scenario,
        share_demand_signals=not args.no_demand_sharing,
        share_capacity_signals=not args.no_capacity_sharing,
    )

    try:
        models = trainer.train(
            total_timesteps_per_agent=args.timesteps,
            n_rounds=args.rounds,
        )
        print("\nMulti-agent training complete!")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
