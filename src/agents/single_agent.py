"""
Single-Agent DRL Trainer using Stable Baselines3.
"""

from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.supply_chain_env import SupplyChainEnv


def make_env(env_config, reward_config, topology_config, shock_config,
             shock_mode="stochastic", scenario_preset=None, seed=42, rank=0):
    def _init():
        env = SupplyChainEnv(
            env_config=env_config, reward_config=reward_config,
            topology_config=topology_config, shock_config=shock_config,
            shock_mode=shock_mode, scenario_preset=scenario_preset, seed=seed + rank,
        )
        return Monitor(env)
    return _init


class SingleAgentTrainer:
    def __init__(self, env_config, reward_config, topology_config, shock_config,
                 training_config, output_dir="outputs", shock_mode="stochastic",
                 scenario_preset=None, n_envs=4):
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.n_envs = n_envs

        self.log_dir = self.output_dir / "logs"
        self.model_dir = self.output_dir / "models"
        self.eval_dir = self.output_dir / "eval"
        for d in [self.log_dir, self.model_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        seed = training_config.get("seed", 42)

        self.train_env = DummyVecEnv([
            make_env(env_config, reward_config, topology_config, shock_config,
                     shock_mode, scenario_preset, seed, rank=i)
            for i in range(n_envs)
        ])

        self.eval_env = DummyVecEnv([
            make_env(env_config, reward_config, topology_config, shock_config,
                     "stochastic", None, seed + 1000)
        ])

        tc = training_config
        self.model = PPO(
            "MlpPolicy", self.train_env,
            learning_rate=tc.get("learning_rate", 3e-4),
            n_steps=tc.get("n_steps", 2048),
            batch_size=tc.get("batch_size", 64),
            n_epochs=tc.get("n_epochs", 10),
            gamma=tc.get("gamma", 0.99),
            gae_lambda=tc.get("gae_lambda", 0.95),
            clip_range=tc.get("clip_range", 0.2),
            ent_coef=tc.get("ent_coef", 0.01),
            verbose=1, seed=seed,
            tensorboard_log=str(self.log_dir),
        )

    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = self.training_config.get("total_timesteps", 500_000)

        eval_freq = self.training_config.get("eval_freq", 10_000)
        eval_episodes = self.training_config.get("eval_episodes", 20)

        eval_cb = EvalCallback(
            self.eval_env, best_model_save_path=str(self.model_dir / "best"),
            log_path=str(self.eval_dir),
            eval_freq=max(eval_freq // self.n_envs, 1),
            n_eval_episodes=eval_episodes, deterministic=True,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=max(50_000 // self.n_envs, 1),
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix="ppo_supply_chain",
        )

        print(f"Training for {total_timesteps:,} steps | {self.n_envs} envs | Logs: {self.log_dir}")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([eval_cb, ckpt_cb]),
            log_interval=self.training_config.get("log_interval", 10),
            tb_log_name="ppo_supply_chain",
        )

        final_path = str(self.model_dir / "final_model")
        self.model.save(final_path)
        print(f"Final model saved to {final_path}")
        return self.model

    def load_model(self, path):
        self.model = PPO.load(path, env=self.train_env)
        return self.model

    def cleanup(self):
        self.train_env.close()
        self.eval_env.close()
