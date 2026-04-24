"""Tests for the Shock Injector module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.environment.des_engine import SupplyChainDES
from src.environment.shock_injector import ShockInjector
from src.utils.config_loader import load_all_configs


@pytest.fixture
def configs():
    return load_all_configs()


@pytest.fixture
def des(configs):
    rng = np.random.default_rng(42)
    return SupplyChainDES(configs["environment"], configs["topology"]["echelons"], rng)


class TestShockInjector:
    def test_initialization(self, configs):
        rng = np.random.default_rng(42)
        injector = ShockInjector(configs["shocks"], rng)
        assert len(injector.active_shocks) == 0
        assert not injector.has_active_shocks

    def test_stochastic_triggering(self, configs, des):
        """Run many steps — shocks should eventually trigger."""
        rng = np.random.default_rng(123)
        injector = ShockInjector(configs["shocks"], rng, mode="stochastic")

        any_triggered = False
        for step in range(500):
            triggered = injector.step(step, des)
            if triggered:
                any_triggered = True
                break

        assert any_triggered, "No shocks triggered in 500 steps — check probabilities"

    def test_deterministic_triggering(self, configs, des):
        """Preset scenarios should trigger at exact steps."""
        rng = np.random.default_rng(42)
        injector = ShockInjector(
            configs["shocks"], rng,
            mode="deterministic",
            scenario_preset="single_surge",
        )

        # Step to trigger point
        for step in range(61):
            triggered = injector.step(step, des)
            if step == 60:
                assert "demand_surge" in triggered

    def test_shock_expiration(self, configs, des):
        """Shocks should deactivate after their duration."""
        rng = np.random.default_rng(42)
        injector = ShockInjector(
            configs["shocks"], rng,
            mode="deterministic",
            scenario_preset="single_surge",
        )

        # Trigger the shock
        for step in range(61):
            injector.step(step, des)

        assert injector.has_active_shocks

        # Run past the shock duration (14 steps for demand_surge)
        for step in range(61, 80):
            injector.step(step, des)

        assert not injector.has_active_shocks

    def test_shock_indicators(self, configs, des):
        rng = np.random.default_rng(42)
        injector = ShockInjector(configs["shocks"], rng)
        indicators = injector.get_shock_indicators()
        assert "demand_shock" in indicators
        assert "supply_shock" in indicators
        assert "logistics_shock" in indicators

    def test_reset(self, configs, des):
        rng = np.random.default_rng(42)
        injector = ShockInjector(
            configs["shocks"], rng,
            mode="deterministic",
            scenario_preset="single_surge",
        )
        for step in range(65):
            injector.step(step, des)
        injector.reset()
        assert len(injector.active_shocks) == 0
        assert len(injector.shock_history) == 0
