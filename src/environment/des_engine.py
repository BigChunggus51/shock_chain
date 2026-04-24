"""
Discrete Event Simulation engine for supply chain dynamics.

Uses SimPy to model stochastic demand arrivals, shipment transit,
inventory management, and resource contention at each echelon.
The DRL agent interacts at the epoch level while many fine-grained
DES events fire within each epoch.
"""

from __future__ import annotations

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EchelonState:
    """Mutable state container for a single supply chain echelon."""

    name: str
    echelon_type: str  # "retailer", "warehouse", "factory"
    upstream: str  # name of upstream echelon or "external"
    inventory: float = 0.0
    backlog: float = 0.0
    pipeline_inventory: float = 0.0  # units in transit
    last_order_qty: float = 0.0
    total_demand_this_epoch: float = 0.0
    total_fulfilled_this_epoch: float = 0.0
    total_holding_cost_this_epoch: float = 0.0
    total_stockout_cost_this_epoch: float = 0.0
    expedited_this_epoch: bool = False

    def reset_epoch_counters(self):
        """Clear per-epoch accumulators before a new decision epoch."""
        self.total_demand_this_epoch = 0.0
        self.total_fulfilled_this_epoch = 0.0
        self.total_holding_cost_this_epoch = 0.0
        self.total_stockout_cost_this_epoch = 0.0
        self.expedited_this_epoch = False

    @property
    def fill_rate(self) -> float:
        """Fraction of demand fulfilled this epoch (0-1)."""
        if self.total_demand_this_epoch <= 0:
            return 1.0
        return min(1.0, self.total_fulfilled_this_epoch / self.total_demand_this_epoch)


class SupplyChainDES:
    """
    Core SimPy-based Discrete Event Simulation for a multi-echelon supply chain.

    This class manages the simulation clock, demand generation, shipment transit,
    and inventory updates. It is designed to be wrapped by a Gymnasium environment
    that calls `advance_epoch()` once per agent decision step.

    Parameters
    ----------
    config : dict
        Environment configuration loaded from configs/default.yaml.
    topology : list[dict]
        List of echelon definitions from configs/default.yaml -> topology.echelons.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    """

    def __init__(self, config: dict, topology: list[dict], rng: np.random.Generator):
        self.config = config
        self.rng = rng

        # Initialize SimPy environment
        self.sim = simpy.Environment()

        # Build echelon state objects
        self.echelons: dict[str, EchelonState] = {}
        for echelon_def in topology:
            name = echelon_def["name"]
            self.echelons[name] = EchelonState(
                name=name,
                echelon_type=echelon_def["type"],
                upstream=echelon_def["upstream"],
                inventory=float(config["initial_inventory"]),
            )

        # Global modifiers (can be altered by ShockInjector)
        self.demand_multiplier: float = 1.0
        self.lead_time_multiplier: float = 1.0
        self.supplier_capacity: float = float(config["max_supplier_capacity"])

        # Demand history buffer for state observations
        self._demand_history: list[float] = []

        # Start the persistent demand process for each retailer
        for name, echelon in self.echelons.items():
            if echelon.echelon_type == "retailer":
                self.sim.process(self._demand_process(name))

    def reset(self) -> None:
        """Reset the entire simulation to initial conditions."""
        self.sim = simpy.Environment()
        self.demand_multiplier = 1.0
        self.lead_time_multiplier = 1.0
        self.supplier_capacity = float(self.config["max_supplier_capacity"])
        self._demand_history = []

        for echelon in self.echelons.values():
            echelon.inventory = float(self.config["initial_inventory"])
            echelon.backlog = 0.0
            echelon.pipeline_inventory = 0.0
            echelon.last_order_qty = 0.0
            echelon.reset_epoch_counters()

        # Restart demand processes
        for name, echelon in self.echelons.items():
            if echelon.echelon_type == "retailer":
                self.sim.process(self._demand_process(name))

    def _demand_process(self, echelon_name: str):
        """
        SimPy process: generates stochastic customer demand at regular intervals.

        Demand arrives every time unit within an epoch. The demand rate is
        modulated by `self.demand_multiplier` (altered during shock events).
        """
        echelon = self.echelons[echelon_name]

        while True:
            # Sample demand from Poisson distribution, modulated by shock multiplier
            base_demand = self.rng.poisson(self.config["mean_demand"])
            actual_demand = max(0, int(base_demand * self.demand_multiplier))

            # Attempt to fulfill from on-hand inventory
            fulfilled = min(actual_demand, echelon.inventory)
            echelon.inventory -= fulfilled
            unfulfilled = actual_demand - fulfilled

            # Add unfulfilled demand to backlog
            echelon.backlog += unfulfilled

            # Update epoch counters
            echelon.total_demand_this_epoch += actual_demand
            echelon.total_fulfilled_this_epoch += fulfilled
            echelon.total_stockout_cost_this_epoch += unfulfilled

            # Record demand for observation history
            self._demand_history.append(actual_demand)

            yield self.sim.timeout(1)

    def _shipment_process(
        self,
        source: str,
        destination: str,
        quantity: float,
        expedited: bool = False,
    ):
        """
        SimPy process: models a shipment traveling from source to destination.

        Lead time is sampled from a normal distribution, modulated by the
        global `lead_time_multiplier`. Expedited shipments have halved lead time.

        Parameters
        ----------
        source : str
            Name of the upstream echelon or "external".
        destination : str
            Name of the receiving echelon.
        quantity : float
            Number of units being shipped.
        expedited : bool
            If True, lead time is halved (at extra cost).
        """
        dest_echelon = self.echelons[destination]

        # Track pipeline inventory
        dest_echelon.pipeline_inventory += quantity

        # Calculate lead time
        base_lt = self.config["base_lead_time"] * self.lead_time_multiplier
        if expedited:
            base_lt *= 0.5

        lead_time = max(1, int(self.rng.normal(base_lt, self.config["lead_time_std"])))

        # Transit delay
        yield self.sim.timeout(lead_time)

        # Apply supplier capacity constraint for external shipments
        if source == "external":
            delivered = min(quantity, self.supplier_capacity)
        else:
            # For internal shipments, check source inventory
            source_echelon = self.echelons[source]
            delivered = min(quantity, source_echelon.inventory)
            source_echelon.inventory -= delivered

        # Deliver to destination
        dest_echelon.inventory += delivered
        dest_echelon.pipeline_inventory -= quantity  # Remove from pipeline

        # Automatically fulfill backlog with newly arrived stock
        backlog_fulfilled = min(dest_echelon.backlog, dest_echelon.inventory)
        dest_echelon.inventory -= backlog_fulfilled
        dest_echelon.backlog -= backlog_fulfilled
        dest_echelon.total_fulfilled_this_epoch += backlog_fulfilled

    def advance_epoch(self, actions: dict[str, dict[str, Any]]) -> dict[str, EchelonState]:
        """
        Advance the simulation by one decision epoch.

        This is the primary interface called by the Gymnasium environment's
        `step()` method. It:
        1. Resets per-epoch counters
        2. Schedules shipments based on agent actions
        3. Runs the SimPy event loop for `epoch_length` time units
        4. Computes holding costs
        5. Returns updated echelon states

        Parameters
        ----------
        actions : dict[str, dict[str, Any]]
            Mapping of echelon name -> {"order_qty": float, "expedite": bool}.

        Returns
        -------
        dict[str, EchelonState]
            Updated state of each echelon after the epoch.
        """
        epoch_length = self.config["epoch_length"]

        # Reset per-epoch accumulators
        for echelon in self.echelons.values():
            echelon.reset_epoch_counters()

        # Schedule new orders based on agent actions
        for echelon_name, action in actions.items():
            echelon = self.echelons[echelon_name]
            order_qty = max(0.0, float(action.get("order_qty", 0)))
            expedite = bool(action.get("expedite", False))

            echelon.last_order_qty = order_qty
            echelon.expedited_this_epoch = expedite

            if order_qty > 0:
                self.sim.process(
                    self._shipment_process(
                        source=echelon.upstream,
                        destination=echelon_name,
                        quantity=order_qty,
                        expedited=expedite,
                    )
                )

        # Run simulation for one epoch
        target_time = self.sim.now + epoch_length
        self.sim.run(until=target_time)

        # Compute holding costs (assessed at end of epoch on on-hand inventory)
        for echelon in self.echelons.values():
            echelon.total_holding_cost_this_epoch = max(0.0, echelon.inventory)

        return dict(self.echelons)

    def get_demand_history(self, window: int = 14) -> list[float]:
        """Return the last `window` demand observations."""
        return self._demand_history[-window:]

    @property
    def current_time(self) -> float:
        """Current simulation clock time."""
        return self.sim.now
