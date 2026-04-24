"""
Shock Injector — Disruption Event Manager.

Manages the lifecycle of supply chain shocks: probabilistic triggering,
duration tracking, state modification, and recovery. Loads shock definitions
from configs/shocks.yaml and integrates with the DES engine.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from src.environment.des_engine import SupplyChainDES


@dataclass
class ActiveShock:
    """Tracks the state of a currently active shock event."""

    shock_type: str
    params: dict[str, Any]
    remaining_steps: int
    trigger_step: int  # The epoch step when this shock was activated


class ShockInjector:
    """
    Manages probabilistic and deterministic shock injection into the DES engine.

    Shocks modify global simulation parameters (demand multipliers, lead times,
    supplier capacity) for their duration, then revert. Multiple shocks can be
    active simultaneously, and their effects compound multiplicatively.

    Parameters
    ----------
    shock_config : dict
        Shock catalog loaded from configs/shocks.yaml.
    rng : np.random.Generator
        Seeded RNG for reproducible shock triggering.
    mode : str
        "stochastic" — shocks trigger probabilistically each step.
        "deterministic" — shocks trigger at pre-scheduled steps.
    scenario_preset : str | None
        If set, use a named preset from the shock config.
    """

    def __init__(
        self,
        shock_config: dict,
        rng: np.random.Generator,
        mode: str = "stochastic",
        scenario_preset: str | None = None,
    ):
        self.shock_catalog: dict = shock_config.get("shocks", {})
        self.scenario_presets: dict = shock_config.get("scenario_presets", {})
        self.rng = rng
        self.mode = mode
        self.scenario_preset = scenario_preset

        # Active shock tracking
        self.active_shocks: list[ActiveShock] = []
        self._shock_history: list[dict] = []

        # Deterministic schedule (populated from preset)
        self._scheduled_shocks: list[dict] = []
        self._probability_multiplier: float = 1.0

        if scenario_preset and scenario_preset in self.scenario_presets:
            preset = self.scenario_presets[scenario_preset]
            self._scheduled_shocks = preset.get("scheduled_shocks", [])
            self._probability_multiplier = preset.get("probability_multiplier", 1.0)

    def step(self, current_step: int, des: SupplyChainDES) -> list[str]:
        """
        Called once per decision epoch. Handles shock lifecycle:
        1. Check for new shock triggers (probabilistic or deterministic)
        2. Decrement duration of active shocks
        3. Deactivate expired shocks and restore state
        4. Apply cumulative effects of all active shocks to the DES

        Parameters
        ----------
        current_step : int
            The current decision epoch index.
        des : SupplyChainDES
            Reference to the DES engine whose parameters we modify.

        Returns
        -------
        list[str]
            Names of newly triggered shocks this step (for logging).
        """
        newly_triggered = []

        # --- Phase 1: Trigger new shocks ---
        if self.mode == "stochastic":
            newly_triggered.extend(self._stochastic_trigger(current_step))
        elif self.mode == "deterministic":
            newly_triggered.extend(self._deterministic_trigger(current_step))

        # --- Phase 2: Decrement active shock durations ---
        still_active = []
        for shock in self.active_shocks:
            shock.remaining_steps -= 1
            if shock.remaining_steps > 0:
                still_active.append(shock)
            else:
                # Log shock expiration
                self._shock_history.append(
                    {
                        "type": shock.shock_type,
                        "trigger_step": shock.trigger_step,
                        "end_step": current_step,
                        "event": "expired",
                    }
                )
        self.active_shocks = still_active

        # --- Phase 3: Apply cumulative effects ---
        self._apply_effects(des)

        return newly_triggered

    def _stochastic_trigger(self, current_step: int) -> list[str]:
        """Probabilistically trigger shocks based on per-step probabilities."""
        triggered = []

        for shock_type, shock_def in self.shock_catalog.items():
            # Don't stack the same shock type
            if any(s.shock_type == shock_type for s in self.active_shocks):
                continue

            prob = shock_def["probability_per_step"] * self._probability_multiplier
            if self.rng.random() < prob:
                self._activate_shock(shock_type, shock_def, current_step)
                triggered.append(shock_type)

        return triggered

    def _deterministic_trigger(self, current_step: int) -> list[str]:
        """Trigger shocks at pre-scheduled steps (from scenario presets)."""
        triggered = []

        for scheduled in self._scheduled_shocks:
            if scheduled["trigger_step"] == current_step:
                shock_type = scheduled["type"]
                if shock_type in self.shock_catalog:
                    shock_def = self.shock_catalog[shock_type]
                    self._activate_shock(shock_type, shock_def, current_step)
                    triggered.append(shock_type)

        return triggered

    def _activate_shock(self, shock_type: str, shock_def: dict, current_step: int):
        """Create and register a new active shock."""
        active = ActiveShock(
            shock_type=shock_type,
            params=dict(shock_def.get("params", {})),
            remaining_steps=shock_def["duration_steps"],
            trigger_step=current_step,
        )
        self.active_shocks.append(active)
        self._shock_history.append(
            {
                "type": shock_type,
                "trigger_step": current_step,
                "duration": shock_def["duration_steps"],
                "event": "triggered",
            }
        )

    def _apply_effects(self, des: SupplyChainDES):
        """
        Apply the cumulative effect of all active shocks to the DES engine.

        Effects are compounded multiplicatively. For example, if both a
        "logistics_bottleneck" (2x lead time) and "port_congestion" (2.5x lead time)
        are active, the effective lead time multiplier is 5.0x.
        """
        # Reset to baseline
        des.demand_multiplier = 1.0
        des.lead_time_multiplier = 1.0
        des.supplier_capacity = float(des.config["max_supplier_capacity"])

        for shock in self.active_shocks:
            params = shock.params

            if "demand_multiplier" in params:
                des.demand_multiplier *= params["demand_multiplier"]

            if "lead_time_multiplier" in params:
                des.lead_time_multiplier *= params["lead_time_multiplier"]

            if "supplier_capacity_override" in params:
                des.supplier_capacity = float(params["supplier_capacity_override"])

            if "supplier_capacity_multiplier" in params:
                des.supplier_capacity *= params["supplier_capacity_multiplier"]

            # One-shot effects (e.g., quality recall scrapping inventory)
            if "inventory_scrap_fraction" in params and shock.remaining_steps == (
                self.shock_catalog[shock.shock_type]["duration_steps"] - 1
            ):
                scrap_frac = params["inventory_scrap_fraction"]
                for echelon in des.echelons.values():
                    echelon.inventory *= (1.0 - scrap_frac)

    def get_shock_indicators(self) -> dict[str, float]:
        """
        Return the current shock state as a feature vector for the agent's observation.

        Returns
        -------
        dict[str, float]
            Mapping of shock category -> intensity (0.0 if inactive).
        """
        indicators = {
            "demand_shock": 0.0,
            "supply_shock": 0.0,
            "logistics_shock": 0.0,
        }

        for shock in self.active_shocks:
            if shock.shock_type in ("demand_surge",):
                indicators["demand_shock"] = shock.params.get("demand_multiplier", 1.0)
            elif shock.shock_type in ("supplier_outage", "quality_recall"):
                indicators["supply_shock"] = 1.0
            elif shock.shock_type in ("logistics_bottleneck", "port_congestion"):
                indicators["logistics_shock"] = shock.params.get("lead_time_multiplier", 1.0)

        return indicators

    @property
    def shock_history(self) -> list[dict]:
        """Full history of all shock events for post-analysis."""
        return list(self._shock_history)

    @property
    def has_active_shocks(self) -> bool:
        """Whether any shock is currently active."""
        return len(self.active_shocks) > 0

    def reset(self):
        """Clear all active shocks and history."""
        self.active_shocks = []
        self._shock_history = []
