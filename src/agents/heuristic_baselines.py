"""
Heuristic Baseline Policies for Supply Chain Management.

These classical inventory control policies serve as benchmarks to evaluate
whether the DRL agent provides meaningful improvement. A DRL agent that
can't beat these baselines under normal conditions is undertrained.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class BasePolicy:
    """Abstract base class for inventory policies."""

    def __init__(self, echelon_names: list[str], config: dict):
        self.echelon_names = echelon_names
        self.config = config

    def get_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        """Return an action array compatible with SupplyChainEnv's action space."""
        raise NotImplementedError

    def reset(self):
        """Reset any internal state between episodes."""
        pass


class RandomPolicy(BasePolicy):
    """Uniformly random actions — lower-bound benchmark."""

    def __init__(self, echelon_names: list[str], config: dict, seed: int = 42):
        super().__init__(echelon_names, config)
        self.rng = np.random.default_rng(seed)
        self.n_order_bins = 9  # matches ORDER_BINS in supply_chain_env.py

    def get_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(self.echelon_names)
        order_indices = self.rng.integers(0, self.n_order_bins, size=n)
        expedite_flags = self.rng.integers(0, 2, size=n)
        return np.concatenate([order_indices, expedite_flags])


class BaseStockPolicy(BasePolicy):
    """
    Base-Stock (Order-Up-To) Policy.

    At each epoch, order enough to bring inventory + pipeline back up to
    a target level S. Never expedites.

    target_stock = S (configurable, default = 2 * mean_demand * lead_time)
    order = max(0, S - inventory - pipeline_inventory + backlog)
    """

    def __init__(
        self,
        echelon_names: list[str],
        config: dict,
        target_stock: float | None = None,
    ):
        super().__init__(echelon_names, config)
        if target_stock is None:
            # Heuristic: cover 2x the expected demand during lead time
            self.target_stock = (
                2.0 * config.get("mean_demand", 50) * config.get("base_lead_time", 5)
            )
        else:
            self.target_stock = target_stock

        self.n_order_bins = 9
        self.order_bins = np.array([0, 25, 50, 75, 100, 150, 200, 300, 500], dtype=np.float32)

    def _qty_to_bin_index(self, qty: float) -> int:
        """Map a continuous order quantity to the nearest discrete bin index."""
        return int(np.argmin(np.abs(self.order_bins - qty)))

    def get_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(self.echelon_names)
        order_indices = []

        for i, name in enumerate(self.echelon_names):
            # Extract features from observation (5 features per echelon, normalized)
            base_idx = i * 5
            inv = observation[base_idx] * self.config.get("warehouse_capacity", 2000)
            pipeline = observation[base_idx + 1] * self.config.get("max_order_quantity", 500)
            backlog = observation[base_idx + 2] * self.config.get("mean_demand", 50)

            # Order-up-to calculation
            inventory_position = inv + pipeline - backlog
            order_qty = max(0.0, self.target_stock - inventory_position)
            order_qty = min(order_qty, self.config.get("max_order_quantity", 500))

            order_indices.append(self._qty_to_bin_index(order_qty))

        # Never expedite
        expedite_flags = [0] * n

        return np.array(order_indices + expedite_flags, dtype=np.int64)


class SsPolicy(BasePolicy):
    """
    (s, S) Reorder Point Policy.

    Order up to level S only when inventory position drops below s.
    Otherwise, order nothing.

    s = reorder_point (default = mean_demand * lead_time)
    S = order_up_to (default = 2 * s)
    """

    def __init__(
        self,
        echelon_names: list[str],
        config: dict,
        reorder_point: float | None = None,
        order_up_to: float | None = None,
    ):
        super().__init__(echelon_names, config)
        mean_demand = config.get("mean_demand", 50)
        lead_time = config.get("base_lead_time", 5)

        self.s = reorder_point if reorder_point is not None else mean_demand * lead_time
        self.S = order_up_to if order_up_to is not None else 2.0 * self.s

        self.n_order_bins = 9
        self.order_bins = np.array([0, 25, 50, 75, 100, 150, 200, 300, 500], dtype=np.float32)

    def _qty_to_bin_index(self, qty: float) -> int:
        return int(np.argmin(np.abs(self.order_bins - qty)))

    def get_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(self.echelon_names)
        order_indices = []

        for i, name in enumerate(self.echelon_names):
            base_idx = i * 5
            inv = observation[base_idx] * self.config.get("warehouse_capacity", 2000)
            pipeline = observation[base_idx + 1] * self.config.get("max_order_quantity", 500)
            backlog = observation[base_idx + 2] * self.config.get("mean_demand", 50)

            inventory_position = inv + pipeline - backlog

            if inventory_position < self.s:
                order_qty = max(0.0, self.S - inventory_position)
                order_qty = min(order_qty, self.config.get("max_order_quantity", 500))
            else:
                order_qty = 0.0

            order_indices.append(self._qty_to_bin_index(order_qty))

        expedite_flags = [0] * n
        return np.array(order_indices + expedite_flags, dtype=np.int64)


class BeerGamePolicy(BasePolicy):
    """
    Reactive 'Beer Game' Heuristic.

    Simply orders whatever was demanded last epoch. This is the classic
    policy that amplifies the Bullwhip Effect — a useful adversarial baseline.
    """

    def __init__(self, echelon_names: list[str], config: dict):
        super().__init__(echelon_names, config)
        self.order_bins = np.array([0, 25, 50, 75, 100, 150, 200, 300, 500], dtype=np.float32)

    def _qty_to_bin_index(self, qty: float) -> int:
        return int(np.argmin(np.abs(self.order_bins - qty)))

    def get_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(self.echelon_names)
        order_indices = []

        # Use the most recent demand from the demand history portion of obs
        # Demand history starts at index n_echelons * 5
        demand_start = n * 5
        # Last value in the 14-step demand window
        last_demand_normalized = observation[demand_start + 13]
        last_demand = last_demand_normalized * self.config.get("mean_demand", 50)

        for i in range(n):
            order_qty = max(0.0, min(last_demand, self.config.get("max_order_quantity", 500)))
            order_indices.append(self._qty_to_bin_index(order_qty))

        expedite_flags = [0] * n
        return np.array(order_indices + expedite_flags, dtype=np.int64)
