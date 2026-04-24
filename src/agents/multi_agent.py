"""
Multi-Agent Trainer using Independent PPO (IPPO) with PettingZoo.

Implements Centralized Training with Decentralized Execution (CTDE):
- Each agent has its own PPO policy network (actor)
- During training, agents share a vectorized environment
- During execution, each agent acts independently on local observations

This uses SuperSuit wrappers to convert PettingZoo parallel envs into
SB3-compatible vectorized environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.environment.multi_agent_env import MultiAgentSupplyChainEnv


import gymnasium as gym

class _PettingZooToGymWrapper(gym.Env):
    """
    Lightweight wrapper that converts a PettingZoo ParallelEnv into
    a single-agent Gym-like interface for a specific agent.

    Other agents use frozen baseline policies during training.
    This is the standard IPPO approach.
    """

    def __init__(
        self,
        env: MultiAgentSupplyChainEnv,
        agent_id: str,
        other_policies: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.env = env
        self.agent_id = agent_id
        self.other_policies = other_policies or {}
        self._last_observations = {}

        # Expose Gym interface
        self.observation_space = env.observation_space(agent_id)
        self.action_space = env.action_space(agent_id)
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.spec = None

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        return observations[self.agent_id], infos.get(self.agent_id, {})

    def step(self, action):
        # Build actions for all agents
        actions = {}
        for agent in self.env.possible_agents:
            if agent == self.agent_id:
                actions[agent] = action
            elif agent in self.other_policies:
                # Use the other agent's policy
                obs = self._last_observations.get(agent)
                if obs is not None:
                    other_action, _ = self.other_policies[agent].predict(
                        obs, deterministic=False
                    )
                    actions[agent] = other_action
                else:
                    actions[agent] = self.env.action_space(agent).sample()
            else:
                # Default: sample random action
                actions[agent] = self.env.action_space(agent).sample()

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        self._last_observations = observations

        # Return only this agent's data
        obs = observations.get(self.agent_id, np.zeros(self.observation_space.shape))
        reward = rewards.get(self.agent_id, 0.0)
        terminated = terminations.get(self.agent_id, False)
        truncated = truncations.get(self.agent_id, False)
        info = infos.get(self.agent_id, {})

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        pass


class MultiAgentTrainer:
    """
    IPPO (Independent PPO) trainer for multi-agent supply chain control.

    Trains each agent's PPO policy independently while they share the
    same environment. Uses round-robin training: train agent A for N steps
    while others use their latest policy, then train agent B, etc.

    Parameters
    ----------
    env_config : dict
        Environment parameters.
    reward_config : dict
        Reward coefficients.
    topology_config : dict
        Supply chain topology.
    shock_config : dict
        Shock catalog and presets.
    training_config : dict
        PPO hyperparameters.
    output_dir : str
        Directory for logs, checkpoints, and saved models.
    shock_mode : str
        "stochastic" or "deterministic".
    scenario_preset : str | None
        Named scenario preset.
    share_demand_signals : bool
        Enable demand signal sharing (anti-Bullwhip).
    share_capacity_signals : bool
        Enable capacity signal sharing (anti-Bullwhip).
    """

    def __init__(
        self,
        env_config: dict,
        reward_config: dict,
        topology_config: dict,
        shock_config: dict,
        training_config: dict,
        output_dir: str = "outputs_multi",
        shock_mode: str = "stochastic",
        scenario_preset: str | None = None,
        share_demand_signals: bool = True,
        share_capacity_signals: bool = True,
    ):
        self.env_config = env_config
        self.reward_config = reward_config
        self.topology_config = topology_config
        self.shock_config = shock_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.shock_mode = shock_mode
        self.scenario_preset = scenario_preset
        self.share_demand_signals = share_demand_signals
        self.share_capacity_signals = share_capacity_signals

        # Parse agent names from topology
        self.agent_names = [e["name"] for e in topology_config["echelons"]]

        # Create output directories
        for agent_name in self.agent_names:
            (self.output_dir / "models" / agent_name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "eval").mkdir(parents=True, exist_ok=True)

        # Initialize per-agent models (will be created during training)
        self.models: dict[str, PPO] = {}
        self._envs: list = []

        seed = training_config.get("seed", 42)

        # Create initial PPO model for each agent
        for i, agent_name in enumerate(self.agent_names):
            base_env = self._make_base_env(seed + i * 100)
            wrapped = Monitor(_PettingZooToGymWrapper(base_env, agent_name))
            self._envs.append(wrapped)

            tc = training_config
            self.models[agent_name] = PPO(
                "MlpPolicy",
                wrapped,
                learning_rate=tc.get("learning_rate", 3e-4),
                n_steps=tc.get("n_steps", 2048),
                batch_size=tc.get("batch_size", 64),
                n_epochs=tc.get("n_epochs", 10),
                gamma=tc.get("gamma", 0.99),
                gae_lambda=tc.get("gae_lambda", 0.95),
                clip_range=tc.get("clip_range", 0.2),
                ent_coef=tc.get("ent_coef", 0.01),
                verbose=0,
                seed=seed + i,
                tensorboard_log=str(self.output_dir / "logs"),
            )

    def _make_base_env(self, seed: int) -> MultiAgentSupplyChainEnv:
        """Create a fresh multi-agent environment instance."""
        return MultiAgentSupplyChainEnv(
            env_config=self.env_config,
            reward_config=self.reward_config,
            topology_config=self.topology_config,
            shock_config=self.shock_config,
            shock_mode=self.shock_mode,
            scenario_preset=self.scenario_preset,
            seed=seed,
            share_demand_signals=self.share_demand_signals,
            share_capacity_signals=self.share_capacity_signals,
        )

    def train(
        self,
        total_timesteps_per_agent: int | None = None,
        n_rounds: int = 5,
    ) -> dict[str, PPO]:
        """
        Train all agents using round-robin IPPO.

        Each round:
        1. For each agent, train for (total_timesteps / n_rounds) steps
        2. Update the other agents' policies with their latest weights
        3. Repeat

        Parameters
        ----------
        total_timesteps_per_agent : int | None
            Total training steps per agent. If None, uses config value.
        n_rounds : int
            Number of round-robin training rounds.

        Returns
        -------
        dict[str, PPO]
            Trained models for each agent.
        """
        if total_timesteps_per_agent is None:
            total_timesteps_per_agent = self.training_config.get("total_timesteps", 500_000)

        steps_per_round = total_timesteps_per_agent // n_rounds

        print(f"{'='*60}")
        print(f"MULTI-AGENT IPPO TRAINING")
        print(f"{'='*60}")
        print(f"Agents: {self.agent_names}")
        print(f"Total steps/agent: {total_timesteps_per_agent:,}")
        print(f"Rounds: {n_rounds}")
        print(f"Steps/round/agent: {steps_per_round:,}")
        print(f"Demand sharing: {self.share_demand_signals}")
        print(f"Capacity sharing: {self.share_capacity_signals}")
        print(f"{'='*60}")

        for round_idx in range(n_rounds):
            print(f"\n--- Round {round_idx + 1}/{n_rounds} ---")

            for agent_name in self.agent_names:
                print(f"  Training {agent_name}... ", end="", flush=True)

                # Rebuild environment with latest policies for other agents
                other_policies = {
                    name: model
                    for name, model in self.models.items()
                    if name != agent_name
                }

                seed = self.training_config.get("seed", 42) + round_idx * 1000
                base_env = self._make_base_env(seed)
                wrapped = Monitor(
                    _PettingZooToGymWrapper(base_env, agent_name, other_policies)
                )

                self.models[agent_name].set_env(wrapped)

                self.models[agent_name].learn(
                    total_timesteps=steps_per_round,
                    reset_num_timesteps=False,
                    tb_log_name=f"ippo_{agent_name}",
                    log_interval=None,
                )

                print(f"done")

            # Save checkpoint after each round
            for agent_name in self.agent_names:
                ckpt_path = self.output_dir / "models" / agent_name / f"round_{round_idx + 1}"
                self.models[agent_name].save(str(ckpt_path))

            # Evaluate after each round
            mean_reward = self._evaluate_round(round_idx + 1)
            print(f"  Round {round_idx + 1} eval: mean_reward = {mean_reward:.1f}")

        # Save final models
        for agent_name in self.agent_names:
            final_path = self.output_dir / "models" / agent_name / "final_model"
            self.models[agent_name].save(str(final_path))
            print(f"Saved {agent_name} -> {final_path}")

        return self.models

    def _evaluate_round(
        self,
        round_idx: int,
        n_episodes: int = 5,
    ) -> float:
        """Run evaluation episodes with all agents using their latest policies."""
        total_rewards = []

        for ep in range(n_episodes):
            seed = 9000 + round_idx * 100 + ep
            env = self._make_base_env(seed)
            observations, infos = env.reset(seed=seed)
            episode_reward = 0.0

            while env.agents:  # PettingZoo convention: empty when done
                actions = {}
                for agent_name in env.agents:
                    obs = observations[agent_name]
                    action, _ = self.models[agent_name].predict(obs, deterministic=True)
                    actions[agent_name] = action

                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())

            total_rewards.append(episode_reward)

        mean_reward = float(np.mean(total_rewards))

        # Log to file
        eval_file = self.output_dir / "eval" / "round_evals.txt"
        with open(eval_file, "a") as f:
            f.write(f"Round {round_idx}: mean_reward={mean_reward:.2f}, "
                    f"std={np.std(total_rewards):.2f}\n")

        return mean_reward

    def load_models(self, round_name: str = "final_model") -> dict[str, PPO]:
        """Load pre-trained models for all agents."""
        for agent_name in self.agent_names:
            path = self.output_dir / "models" / agent_name / round_name
            self.models[agent_name] = PPO.load(str(path))
        return self.models

    def cleanup(self):
        """Close all environments."""
        for env in self._envs:
            env.close()
