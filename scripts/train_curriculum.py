#!/usr/bin/env python3
"""
Curriculum Learning Training Script for Supply Chain DRL.
Gradually introduces shocks to improve stability and fill rates.

Phase 1: Calm Seas (No Shocks) - 100K steps
Phase 2: Mild Noise (Stochastic Shocks) - 200K steps
Phase 3: Black Swan Events (Cascading Crisis) - 200K steps
"""
import sys
from pathlib import Path
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_all_configs
from src.environment.supply_chain_env import SupplyChainEnv

def make_env(configs, shock_mode, scenario):
    return SupplyChainEnv(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        shock_mode=shock_mode,
        scenario_preset=scenario,
        seed=42
    )

def main():
    configs = load_all_configs("configs/default.yaml", "configs/shocks.yaml")
    output_dir = Path("outputs_curriculum")
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CURRICULUM LEARNING - PHASE 1: Calm Seas (No shocks)")
    print("="*60)
    # Phase 1: No shocks
    env1 = SupplyChainEnv(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        shock_mode="deterministic",
        scenario_preset="calm",
        seed=42
    )
    
    model = PPO("MlpPolicy", env1, verbose=0, tensorboard_log=str(output_dir / "logs"))
    model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name="P1_Calm")
    model.save(str(output_dir / "models" / "phase1_model"))
    
    print("="*60)
    print("CURRICULUM LEARNING - PHASE 2: Mild Noise (Stochastic)")
    print("="*60)
    env2 = make_env(configs, "stochastic", None)
    model.set_env(env2)
    model.learn(total_timesteps=200_000, reset_num_timesteps=False, tb_log_name="P2_Stochastic")
    model.save(str(output_dir / "models" / "phase2_model"))
    
    print("="*60)
    print("CURRICULUM LEARNING - PHASE 3: Black Swan (Cascading Crisis)")
    print("="*60)
    env3 = make_env(configs, "deterministic", "cascading_crisis")
    model.set_env(env3)
    model.learn(total_timesteps=200_000, reset_num_timesteps=False, tb_log_name="P3_Crisis")
    model.save(str(output_dir / "models" / "final_curriculum_model"))

    print("\nCurriculum Training Complete! Models saved to outputs_curriculum/models/")

if __name__ == "__main__":
    main()
