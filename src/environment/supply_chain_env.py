"""
Gymnasium Environment Wrapper for Single-Agent Supply Chain Control.

Wraps the SimPy DES engine into a standard Gymnasium environment where
a single agent controls all echelons simultaneously. This is the Phase 1
MVP environment — use multi_agent_env.py for the CTDE multi-agent version.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any

from src.environment.des_engine import SupplyChainDES
from src.environment.shock_injector import ShockInjector
from src.rewards.composite_reward import (
    RewardCoefficients,
    compute_global_reward,
)


class SupplyChainEnv(gym.Env):
    """
    Single-agent Gymnasium environment for supply chain management.

    The agent observes the full state of all echelons and outputs
    order quantities for each echelon at every decision epoch.

    Observation Space:
        For each echelon: [inventory, pipeline_inv, backlog, fill_rate, last_order]
        Plus: [demand_history (H values), shock_indicators (3), time_features (2)]

    Action Space:
        Discrete: order quantity index per echelon (from predefined bins)
        + expedite flag per echelon

    Parameters
    ----------
    env_config : dict
        Environment configuration from configs/default.yaml -> environment.
    reward_config : dict
        Reward coefficients from configs/default.yaml -> reward.
    topology_config : dict
        Supply chain topology from configs/default.yaml -> topology.
    shock_config : dict
        Shock definitions from configs/shocks.yaml.
    shock_mode : str
        "stochastic" or "deterministic".
    scenario_preset : str | None
        Named shock scenario preset.
    seed : int
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # Discrete order quantity bins
    ORDER_BINS = np.array([0, 25, 50, 75, 100, 150, 200, 300, 500], dtype=np.float32)
    DEMAND_HISTORY_WINDOW = 14

    def __init__(
        self,
        env_config: dict,
        reward_config: dict,
        topology_config: dict,
        shock_config: dict,
        shock_mode: str = "stochastic",
        scenario_preset: str | None = None,
        seed: int = 42,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.env_config = env_config
        self.render_mode = render_mode

        # Setup RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Parse topology
        self.topology = topology_config["echelons"]
        self.echelon_names = [e["name"] for e in self.topology]
        self.n_echelons = len(self.echelon_names)

        # Initialize DES engine
        self.des = SupplyChainDES(env_config, self.topology, self.rng)

        # Initialize shock injector
        self.shock_injector = ShockInjector(
            shock_config=shock_config,
            rng=self.rng,
            mode=shock_mode,
            scenario_preset=scenario_preset,
        )

        # Reward function
        self.reward_coefficients = RewardCoefficients.from_config(reward_config)

        # --- Define observation space ---
        # Per echelon: inventory, pipeline, backlog, fill_rate, last_order = 5
        # Global: demand_history (14) + shock_indicators (3) + time_features (2) = 19
        obs_dim = self.n_echelons * 5 + self.DEMAND_HISTORY_WINDOW + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # --- Define action space ---
        # For each echelon: order_bin_index (len(ORDER_BINS)) + expedite (2)
        self.action_space = spaces.MultiDiscrete(
            [len(self.ORDER_BINS)] * self.n_echelons + [2] * self.n_echelons
        )

        # Episode state tracking
        self._step_count = 0
        self._max_steps = env_config.get("max_episode_steps", 365)
        self._previous_orders: dict[str, float] = {name: 0.0 for name in self.echelon_names}

        # Metrics logging
        self._episode_reward = 0.0
        self._episode_metrics: list[dict] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.des.reset()
        self.shock_injector.reset()

        # Re-initialize DES with fresh RNG
        self.des = SupplyChainDES(self.env_config, self.topology, self.rng)

        self._step_count = 0
        self._previous_orders = {name: 0.0 for name in self.echelon_names}
        self._episode_reward = 0.0
        self._episode_metrics = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one decision epoch.

        Parameters
        ----------
        action : np.ndarray
            Multi-discrete action: [order_bin_idx_0, ..., order_bin_idx_n,
                                     expedite_0, ..., expedite_n]

        Returns
        -------
        tuple
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Parse actions
        order_indices = action[: self.n_echelons]
        expedite_flags = action[self.n_echelons :]

        actions_dict = {}
        for i, name in enumerate(self.echelon_names):
            order_qty = float(self.ORDER_BINS[order_indices[i]])
            expedite = bool(expedite_flags[i])
            actions_dict[name] = {"order_qty": order_qty, "expedite": expedite}

        # Inject shocks
        triggered_shocks = self.shock_injector.step(self._step_count, self.des)

        # Advance the DES by one epoch
        echelon_states = self.des.advance_epoch(actions_dict)

        # Compute reward
        global_reward, breakdowns = compute_global_reward(
            echelon_states=echelon_states,
            previous_orders=self._previous_orders,
            coefficients=self.reward_coefficients,
            aggregation="sum",
        )

        # Normalize reward to a tractable range for PPO
        # Scale factor based on worst-case single-echelon penalty per step
        reward_scale = self.n_echelons * self.env_config.get("mean_demand", 50) * 10.0
        scaled_reward = global_reward / max(reward_scale, 1.0)

        # Update previous orders for next step's stability calculation
        self._previous_orders = {
            name: echelon_states[name].last_order_qty for name in self.echelon_names
        }

        # Track episode metrics
        self._episode_reward += scaled_reward
        step_metrics = {
            "step": self._step_count,
            "reward": scaled_reward,
            "raw_reward": global_reward,
            "triggered_shocks": triggered_shocks,
        }
        for name in self.echelon_names:
            es = echelon_states[name]
            step_metrics[f"{name}/inventory"] = es.inventory
            step_metrics[f"{name}/backlog"] = es.backlog
            step_metrics[f"{name}/fill_rate"] = es.fill_rate
            step_metrics[f"{name}/order_qty"] = es.last_order_qty
        self._episode_metrics.append(step_metrics)

        # Check termination
        terminated = False
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        info = self._get_info(
            reward_breakdowns=breakdowns,
            triggered_shocks=triggered_shocks,
        )

        return obs, scaled_reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct the flat observation vector from current DES state."""
        obs_parts = []

        # Per-echelon features
        for name in self.echelon_names:
            es = self.des.echelons[name]
            obs_parts.extend([
                es.inventory / self.env_config.get("warehouse_capacity", 2000),  # normalized
                es.pipeline_inventory / self.env_config.get("max_order_quantity", 500),
                es.backlog / self.env_config.get("mean_demand", 50),
                es.fill_rate,
                es.last_order_qty / self.env_config.get("max_order_quantity", 500),
            ])

        # Demand history (zero-padded if not enough history)
        demand_hist = self.des.get_demand_history(self.DEMAND_HISTORY_WINDOW)
        padded_hist = [0.0] * (self.DEMAND_HISTORY_WINDOW - len(demand_hist)) + demand_hist
        mean_demand = self.env_config.get("mean_demand", 50)
        obs_parts.extend([d / mean_demand for d in padded_hist])  # normalized

        # Shock indicators
        indicators = self.shock_injector.get_shock_indicators()
        obs_parts.extend([
            indicators["demand_shock"],
            indicators["supply_shock"],
            indicators["logistics_shock"],
        ])

        # Time features (cyclical encoding)
        if self._step_count > 0:
            day_of_week = (self._step_count % 7) / 7.0
            week_of_year = (self._step_count % 52) / 52.0
        else:
            day_of_week = 0.0
            week_of_year = 0.0
        obs_parts.extend([day_of_week, week_of_year])

        return np.array(obs_parts, dtype=np.float32)

    def _get_info(
        self,
        reward_breakdowns: dict | None = None,
        triggered_shocks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Construct the info dict for logging and debugging."""
        info: dict[str, Any] = {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "active_shocks": [s.shock_type for s in self.shock_injector.active_shocks],
            "sim_time": self.des.current_time,
        }

        if triggered_shocks:
            info["triggered_shocks"] = triggered_shocks

        if reward_breakdowns:
            info["reward_breakdowns"] = {
                name: bd.to_dict() for name, bd in reward_breakdowns.items()
            }

        # Per-echelon summary
        for name in self.echelon_names:
            es = self.des.echelons[name]
            info[f"{name}_inventory"] = es.inventory
            info[f"{name}_backlog"] = es.backlog
            info[f"{name}_fill_rate"] = es.fill_rate

        return info

    def render(self) -> str | None:
        """Render current state as text."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            lines = [f"=== Step {self._step_count}/{self._max_steps} ==="]
            for name in self.echelon_names:
                es = self.des.echelons[name]
                lines.append(
                    f"  {name:>12s}: inv={es.inventory:6.0f}  "
                    f"backlog={es.backlog:5.0f}  "
                    f"fill={es.fill_rate:.1%}  "
                    f"order={es.last_order_qty:5.0f}"
                )
            active = [s.shock_type for s in self.shock_injector.active_shocks]
            lines.append(f"  Active shocks: {active if active else 'None'}")
            lines.append(f"  Episode reward: {self._episode_reward:.1f}")
            output = "\n".join(lines)
            if self.render_mode == "human":
                print(output)
            return output
        return None

    @property
    def episode_metrics(self) -> list[dict]:
        """Full per-step metrics for the current episode."""
        return self._episode_metrics
