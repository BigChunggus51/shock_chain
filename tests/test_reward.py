"""Tests for the composite reward function."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.environment.des_engine import EchelonState
from src.rewards.composite_reward import (
    compute_reward,
    compute_global_reward,
    RewardCoefficients,
)


@pytest.fixture
def default_coefficients():
    return RewardCoefficients()


class TestCompositeReward:
    def test_zero_activity_reward(self, default_coefficients):
        """An echelon with no activity should get only holding + service terms."""
        echelon = EchelonState(
            name="test", echelon_type="warehouse", upstream="factory",
            inventory=100, backlog=0, last_order_qty=0,
        )
        echelon.total_demand_this_epoch = 50
        echelon.total_fulfilled_this_epoch = 50
        echelon.total_holding_cost_this_epoch = 100

        result = compute_reward(echelon, previous_order_qty=0, coefficients=default_coefficients)
        assert result.stockout_term == 0.0  # No backlog
        assert result.holding_term < 0  # Holding cost is a penalty
        assert result.service_term > 0  # Full fill rate = bonus
        assert result.stability_term == 0.0  # No order change

    def test_stockout_penalty_dominates(self, default_coefficients):
        """Large backlog should produce strongly negative reward."""
        echelon = EchelonState(
            name="test", echelon_type="retailer", upstream="warehouse",
            inventory=0, backlog=500, last_order_qty=100,
        )
        echelon.total_demand_this_epoch = 600
        echelon.total_fulfilled_this_epoch = 100
        echelon.total_holding_cost_this_epoch = 0

        result = compute_reward(echelon, previous_order_qty=100, coefficients=default_coefficients)
        assert result.stockout_term == -5000.0  # -10 * 500
        assert result.total < -4000  # Should be very negative

    def test_stability_penalty(self, default_coefficients):
        """Large order change should incur stability penalty."""
        echelon = EchelonState(
            name="test", echelon_type="warehouse", upstream="factory",
            inventory=200, backlog=0, last_order_qty=300,
        )
        echelon.total_demand_this_epoch = 50
        echelon.total_fulfilled_this_epoch = 50
        echelon.total_holding_cost_this_epoch = 200

        result = compute_reward(echelon, previous_order_qty=50, coefficients=default_coefficients)
        # |300 - 50| = 250, penalty = -2.0 * 250 = -500
        assert result.stability_term == -500.0

    def test_expedite_cost(self, default_coefficients):
        echelon = EchelonState(
            name="test", echelon_type="retailer", upstream="warehouse",
            inventory=100, backlog=0, last_order_qty=200,
        )
        echelon.expedited_this_epoch = True
        echelon.total_demand_this_epoch = 50
        echelon.total_fulfilled_this_epoch = 50
        echelon.total_holding_cost_this_epoch = 100

        result = compute_reward(echelon, previous_order_qty=200, coefficients=default_coefficients)
        assert result.expedite_term == -600.0  # -3.0 * 200

    def test_global_reward_sum(self, default_coefficients):
        states = {}
        for name in ["retailer", "warehouse"]:
            e = EchelonState(name=name, echelon_type=name, upstream="x",
                             inventory=100, backlog=0, last_order_qty=50)
            e.total_demand_this_epoch = 50
            e.total_fulfilled_this_epoch = 50
            e.total_holding_cost_this_epoch = 100
            states[name] = e

        prev_orders = {"retailer": 50, "warehouse": 50}
        global_r, breakdowns = compute_global_reward(states, prev_orders, default_coefficients, "sum")
        assert len(breakdowns) == 2
        assert global_r == breakdowns["retailer"].total + breakdowns["warehouse"].total
