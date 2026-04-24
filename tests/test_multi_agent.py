"""Tests for the multi-agent PettingZoo environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.environment.multi_agent_env import MultiAgentSupplyChainEnv
from src.utils.config_loader import load_all_configs


@pytest.fixture
def configs():
    return load_all_configs()


@pytest.fixture
def env(configs):
    e = MultiAgentSupplyChainEnv(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        shock_mode="stochastic",
        seed=42,
    )
    yield e


class TestMultiAgentEnv:
    def test_initialization(self, env):
        assert len(env.possible_agents) == 3
        assert "retailer" in env.possible_agents
        assert "warehouse" in env.possible_agents
        assert "factory" in env.possible_agents

    def test_reset(self, env):
        observations, infos = env.reset()
        assert len(observations) == 3
        for agent in env.possible_agents:
            obs = observations[agent]
            assert obs.shape == env.observation_space(agent).shape
            assert obs.dtype == np.float32

    def test_step(self, env):
        observations, _ = env.reset()

        # All agents act simultaneously
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()

        obs, rewards, terms, truncs, infos = env.step(actions)

        for agent in env.possible_agents:
            assert isinstance(rewards[agent], float)
            assert isinstance(terms[agent], bool)
            assert isinstance(truncs[agent], bool)
            assert "inventory" in infos[agent]
            assert "fill_rate" in infos[agent]

    def test_episode_runs_to_completion(self, env):
        observations, _ = env.reset()
        steps = 0

        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terms, truncs, infos = env.step(actions)
            steps += 1

        assert steps == env._max_steps
        assert len(env.agents) == 0  # PettingZoo convention

    def test_observation_spaces_per_agent(self, env):
        for agent in env.possible_agents:
            space = env.observation_space(agent)
            assert space.shape[0] == env._obs_size

    def test_action_spaces_per_agent(self, env):
        for agent in env.possible_agents:
            space = env.action_space(agent)
            # order_bin (9 choices) + expedite (2 choices)
            assert space.shape == (2,)
            assert space.nvec[0] == 9
            assert space.nvec[1] == 2

    def test_shared_demand_signals(self, configs):
        """Agents should receive demand history when sharing is enabled."""
        env = MultiAgentSupplyChainEnv(
            env_config=configs["environment"],
            reward_config=configs["reward"],
            topology_config=configs["topology"],
            shock_config=configs["shocks"],
            share_demand_signals=True,
            share_capacity_signals=True,
            seed=42,
        )
        obs, _ = env.reset()

        # Run a few steps to build demand history
        for _ in range(5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)

        # All agents should have the same shared demand history in their obs
        # (local features differ but shared context is identical)
        retailer_obs = obs["retailer"]
        warehouse_obs = obs["warehouse"]

        # Demand history starts at index 5 (after local features), spans 14 values
        demand_start = 5
        demand_end = demand_start + 14
        np.testing.assert_array_equal(
            retailer_obs[demand_start:demand_end],
            warehouse_obs[demand_start:demand_end],
        )

    def test_no_sharing_reduces_obs_size(self, configs):
        """Disabling sharing should reduce observation size."""
        env_shared = MultiAgentSupplyChainEnv(
            env_config=configs["environment"],
            reward_config=configs["reward"],
            topology_config=configs["topology"],
            shock_config=configs["shocks"],
            share_demand_signals=True,
            share_capacity_signals=True,
            seed=42,
        )
        env_no_share = MultiAgentSupplyChainEnv(
            env_config=configs["environment"],
            reward_config=configs["reward"],
            topology_config=configs["topology"],
            shock_config=configs["shocks"],
            share_demand_signals=False,
            share_capacity_signals=False,
            seed=42,
        )

        shared_size = env_shared.observation_space("retailer").shape[0]
        no_share_size = env_no_share.observation_space("retailer").shape[0]

        # Without sharing, obs should be smaller (no demand history or capacity signals)
        assert no_share_size < shared_size
        # Difference should be 14 (demand) + 3 (capacity) = 17
        assert shared_size - no_share_size == 17

    def test_per_agent_rewards(self, env):
        """Each agent should receive its own reward."""
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # Rewards should differ between echelons (different inventory dynamics)
        assert len(rewards) == 3
        for agent, reward in rewards.items():
            assert isinstance(reward, float)

    def test_render(self, configs):
        env = MultiAgentSupplyChainEnv(
            env_config=configs["environment"],
            reward_config=configs["reward"],
            topology_config=configs["topology"],
            shock_config=configs["shocks"],
            render_mode="ansi",
            seed=42,
        )
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        output = env.render()
        assert output is not None
        assert "Step" in output
        assert "retailer" in output
