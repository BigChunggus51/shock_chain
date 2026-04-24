"""
Composite Reward Function for Supply Chain DRL.

Implements the multi-term reward function that balances service level
against cost efficiency, with an explicit anti-Bullwhip stability term.
"""

from __future__ import annotations

from dataclasses import dataclass
from src.environment.des_engine import EchelonState


@dataclass
class RewardCoefficients:
    """Tunable reward function coefficients."""

    stockout_penalty: float = 10.0   # α
    holding_cost: float = 1.0        # β
    expedite_cost: float = 3.0       # γ
    order_cost: float = 0.5          # δ
    service_bonus: float = 5.0       # λ
    stability_penalty: float = 2.0   # μ

    @classmethod
    def from_config(cls, reward_config: dict) -> "RewardCoefficients":
        """Create coefficients from a YAML config dict."""
        return cls(
            stockout_penalty=reward_config.get("stockout_penalty", 10.0),
            holding_cost=reward_config.get("holding_cost", 1.0),
            expedite_cost=reward_config.get("expedite_cost", 3.0),
            order_cost=reward_config.get("order_cost", 0.5),
            service_bonus=reward_config.get("service_bonus", 5.0),
            stability_penalty=reward_config.get("stability_penalty", 2.0),
        )


@dataclass
class RewardBreakdown:
    """Detailed breakdown of a single reward computation for logging."""

    stockout_term: float
    holding_term: float
    expedite_term: float
    order_term: float
    service_term: float
    stability_term: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "reward/stockout": self.stockout_term,
            "reward/holding": self.holding_term,
            "reward/expedite": self.expedite_term,
            "reward/order": self.order_term,
            "reward/service": self.service_term,
            "reward/stability": self.stability_term,
            "reward/total": self.total,
        }


def compute_reward(
    echelon: EchelonState,
    previous_order_qty: float,
    coefficients: RewardCoefficients,
) -> RewardBreakdown:
    """
    Compute the composite reward for a single echelon after one epoch.

    R(s, a) = -α·backlog - β·inventory - γ·expedite·order_qty
              - δ·order_qty + λ·fill_rate - μ·|order(t) - order(t-1)|

    Parameters
    ----------
    echelon : EchelonState
        Current state of the echelon after the epoch.
    previous_order_qty : float
        The order quantity from the previous epoch (for stability calculation).
    coefficients : RewardCoefficients
        Tunable penalty/bonus weights.

    Returns
    -------
    RewardBreakdown
        Itemized reward components and total.
    """
    c = coefficients

    # Penalty terms (all negative contributions)
    stockout_term = -c.stockout_penalty * max(0.0, echelon.backlog)
    holding_term = -c.holding_cost * max(0.0, echelon.total_holding_cost_this_epoch)
    expedite_term = (
        -c.expedite_cost * echelon.last_order_qty if echelon.expedited_this_epoch else 0.0
    )
    order_term = -c.order_cost * echelon.last_order_qty

    # Bonus terms (positive contributions)
    service_term = c.service_bonus * echelon.fill_rate

    # Anti-Bullwhip stability term (penalizes order variance)
    order_change = abs(echelon.last_order_qty - previous_order_qty)
    stability_term = -c.stability_penalty * order_change

    total = (
        stockout_term
        + holding_term
        + expedite_term
        + order_term
        + service_term
        + stability_term
    )

    return RewardBreakdown(
        stockout_term=stockout_term,
        holding_term=holding_term,
        expedite_term=expedite_term,
        order_term=order_term,
        service_term=service_term,
        stability_term=stability_term,
        total=total,
    )


def compute_global_reward(
    echelon_states: dict[str, EchelonState],
    previous_orders: dict[str, float],
    coefficients: RewardCoefficients,
    aggregation: str = "sum",
) -> tuple[float, dict[str, RewardBreakdown]]:
    """
    Compute the global reward across all echelons.

    Parameters
    ----------
    echelon_states : dict[str, EchelonState]
        Current state of all echelons.
    previous_orders : dict[str, float]
        Previous order quantities for each echelon.
    coefficients : RewardCoefficients
        Shared reward coefficients.
    aggregation : str
        "sum" — total reward across echelons.
        "mean" — average reward across echelons.

    Returns
    -------
    tuple[float, dict[str, RewardBreakdown]]
        Global scalar reward and per-echelon breakdowns.
    """
    breakdowns: dict[str, RewardBreakdown] = {}

    for name, echelon in echelon_states.items():
        prev_order = previous_orders.get(name, 0.0)
        breakdowns[name] = compute_reward(echelon, prev_order, coefficients)

    totals = [b.total for b in breakdowns.values()]

    if aggregation == "mean" and totals:
        global_reward = sum(totals) / len(totals)
    else:
        global_reward = sum(totals)

    return global_reward, breakdowns
