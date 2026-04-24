"""Tests for the DES engine and Gymnasium environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.environment.des_engine import SupplyChainDES
from src.environment.supply_chain_env import SupplyChainEnv
from src.utils.config_loader import load_all_configs


@pytest.fixture
def configs():
    return load_all_configs()


@pytest.fixture
def env(configs):
    e = SupplyChainEnv(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        shock_mode="stochastic",
        seed=42,
    )
    yield e
    e.close()


class TestDESEngine:
    def test_initialization(self, configs):
        rng = np.random.default_rng(42)
        des = SupplyChainDES(configs["environment"], configs["topology"]["echelons"], rng)
        assert len(des.echelons) == 3
        for name, echelon in des.echelons.items():
            assert echelon.inventory == configs["environment"]["initial_inventory"]
            assert echelon.backlog == 0.0

    def test_advance_epoch(self, configs):
        rng = np.random.default_rng(42)
        des = SupplyChainDES(configs["environment"], configs["topology"]["echelons"], rng)
        actions = {
            "retailer": {"order_qty": 100, "expedite": False},
            "warehouse": {"order_qty": 100, "expedite": False},
            "factory": {"order_qty": 100, "expedite": False},
        }
        states = des.advance_epoch(actions)
        assert len(states) == 3
        # Inventory should have decreased due to demand
        assert states["retailer"].inventory < configs["environment"]["initial_inventory"]

    def test_reset(self, configs):
        rng = np.random.default_rng(42)
        des = SupplyChainDES(configs["environment"], configs["topology"]["echelons"], rng)
        actions = {"retailer": {"order_qty": 50}, "warehouse": {"order_qty": 50}, "factory": {"order_qty": 50}}
        des.advance_epoch(actions)
        des.reset()
        for echelon in des.echelons.values():
            assert echelon.inventory == configs["environment"]["initial_inventory"]


class TestSupplyChainEnv:
    def test_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert "step" in info

    def test_step(self, env):
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_runs_to_completion(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == env._max_steps

    def test_render(self, env):
        env.render_mode = "ansi"
        env.reset()
        action = env.action_space.sample()
        env.step(action)
        output = env.render()
        assert output is not None
        assert "Step" in output
