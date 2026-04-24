"""
Multi-Agent PettingZoo Environment for Supply Chain Control (CTDE).

Each supply chain echelon (Retailer, Warehouse, Factory) is controlled by
an independent agent. Uses PettingZoo's Parallel API so all agents act
simultaneously each step.

Architecture: Centralized Training with Decentralized Execution (CTDE)
- Training: A shared critic sees all agents' observations and actions
- Execution: Each agent uses only its local observation to decide

This environment supports:
- Independent agent observations with optional shared context
- Per-agent reward computation with anti-Bullwhip stability terms
- Shared communication channel for demand/capacity signaling
"""

from __future__ import annotations

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.environment.des_engine import SupplyChainDES
from src.environment.shock_injector import ShockInjector
from src.rewards.composite_reward import (
    RewardCoefficients,
    compute_reward,
)


class MultiAgentSupplyChainEnv(ParallelEnv):
    """
    PettingZoo Parallel environment for multi-agent supply chain control.

    Each agent controls one echelon and receives:
    - Local observation: its own inventory, pipeline, backlog, fill rate
    - Shared context: demand signals from retailers, capacity from factory,
      shock indicators, and time features

    Parameters
    ----------
    env_config : dict
        Environment parameters from configs/default.yaml -> environment.
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
    share_demand_signals : bool
        If True, all agents see raw retailer demand (anti-Bullwhip mechanism).
    share_capacity_signals : bool
        If True, all agents see factory capacity/backlog (anti-Bullwhip mechanism).
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "supply_chain_v1"}

    # Discrete order quantity bins (shared across agents)
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
        share_demand_signals: bool = True,
        share_capacity_signals: bool = True,
    ):
        self.env_config = env_config
        self.render_mode = render_mode
        self.share_demand_signals = share_demand_signals
        self.share_capacity_signals = share_capacity_signals

        # Setup RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Parse topology
        self.topology = topology_config["echelons"]
        self.echelon_names = [e["name"] for e in self.topology]
        self.n_echelons = len(self.echelon_names)

        # PettingZoo required attributes
        self.possible_agents = list(self.echelon_names)
        self.agents = list(self.echelon_names)

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

        # Reward normalization scale
        self._reward_scale = max(
            1.0, env_config.get("mean_demand", 50) * 10.0
        )

        # Episode state tracking
        self._step_count = 0
        self._max_steps = env_config.get("max_episode_steps", 365)
        self._previous_orders: dict[str, float] = {
            name: 0.0 for name in self.echelon_names
        }

        # Shared communication buffer (populated each step)
        self._shared_buffer: dict[str, float] = {}

        # Metrics
        self._episode_metrics: list[dict] = []

        # Build observation and action spaces
        self._obs_size = self._compute_obs_size()

    def _compute_obs_size(self) -> int:
        """Calculate the observation vector size for each agent."""
        # Local features: inventory, pipeline, backlog, fill_rate, last_order = 5
        local_size = 5

        # Shared context
        shared_size = 0
        # Shock indicators: 3
        shared_size += 3
        # Time features: 2
        shared_size += 2

        # Demand signals from all retailers (if sharing enabled)
        if self.share_demand_signals:
            shared_size += self.DEMAND_HISTORY_WINDOW  # demand history

        # Capacity signals from factory (if sharing enabled)
        if self.share_capacity_signals:
            # factory inventory (norm), factory backlog (norm), supplier capacity (norm)
            shared_size += 3

        # Other agents' inventory levels (for coordination)
        shared_size += (self.n_echelons - 1)  # other agents' normalized inventory

        return local_size + shared_size

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Return the observation space for an agent."""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_size,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Return the action space for an agent (order_bin + expedite)."""
        return spaces.MultiDiscrete([len(self.ORDER_BINS), 2])

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment and return initial observations for all agents."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self.des = SupplyChainDES(self.env_config, self.topology, self.rng)
        self.shock_injector.reset()
        self._step_count = 0
        self._previous_orders = {name: 0.0 for name in self.echelon_names}
        self._shared_buffer = {}
        self._episode_metrics = []

        observations = {agent: self._get_agent_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],  # observations
        dict[str, float],       # rewards
        dict[str, bool],        # terminations
        dict[str, bool],        # truncations
        dict[str, dict],        # infos
    ]:
        """
        All agents act simultaneously. Each agent's action is
        [order_bin_index, expedite_flag].
        """
        self._step_count += 1

        # Parse actions into DES format
        actions_dict = {}
        for agent_name, action in actions.items():
            order_qty = float(self.ORDER_BINS[int(action[0])])
            expedite = bool(action[1])
            actions_dict[agent_name] = {"order_qty": order_qty, "expedite": expedite}

        # Inject shocks
        triggered_shocks = self.shock_injector.step(self._step_count, self.des)

        # Advance the DES by one epoch
        echelon_states = self.des.advance_epoch(actions_dict)

        # Update shared communication buffer
        self._update_shared_buffer()

        # Compute per-agent rewards
        rewards = {}
        reward_breakdowns = {}
        for agent_name in self.agents:
            echelon = echelon_states[agent_name]
            prev_order = self._previous_orders.get(agent_name, 0.0)
            breakdown = compute_reward(echelon, prev_order, self.reward_coefficients)
            # Normalize reward
            rewards[agent_name] = breakdown.total / self._reward_scale
            reward_breakdowns[agent_name] = breakdown

        # Update previous orders
        self._previous_orders = {
            name: echelon_states[name].last_order_qty for name in self.echelon_names
        }

        # Check termination
        truncated = self._step_count >= self._max_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # Build observations
        observations = {agent: self._get_agent_observation(agent) for agent in self.agents}

        # Build infos
        infos = {}
        for agent_name in self.agents:
            es = echelon_states[agent_name]
            infos[agent_name] = {
                "step": self._step_count,
                "inventory": es.inventory,
                "backlog": es.backlog,
                "fill_rate": es.fill_rate,
                "order_qty": es.last_order_qty,
                "active_shocks": [s.shock_type for s in self.shock_injector.active_shocks],
                "triggered_shocks": triggered_shocks,
                "reward_breakdown": reward_breakdowns[agent_name].to_dict(),
            }

        # Log metrics
        step_metrics = {
            "step": self._step_count,
            "triggered_shocks": triggered_shocks,
        }
        for name in self.echelon_names:
            es = echelon_states[name]
            step_metrics[f"{name}/inventory"] = es.inventory
            step_metrics[f"{name}/backlog"] = es.backlog
            step_metrics[f"{name}/fill_rate"] = es.fill_rate
            step_metrics[f"{name}/order_qty"] = es.last_order_qty
            step_metrics[f"{name}/reward"] = rewards[name]
        self._episode_metrics.append(step_metrics)

        # If episode is done, clear agents list (PettingZoo convention)
        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _update_shared_buffer(self):
        """
        Update the shared communication channel with signals from all echelons.

        This is the anti-Bullwhip mechanism: agents can see:
        - Raw demand at retail level (not distorted upstream orders)
        - Factory capacity and backlog (prevents panic ordering)
        """
        self._shared_buffer = {}

        # Demand history (from DES engine)
        demand_hist = self.des.get_demand_history(self.DEMAND_HISTORY_WINDOW)
        padded = [0.0] * (self.DEMAND_HISTORY_WINDOW - len(demand_hist)) + demand_hist
        mean_demand = self.env_config.get("mean_demand", 50)
        self._shared_buffer["demand_history"] = [d / mean_demand for d in padded]

        # Factory capacity signals
        factory_echelon = None
        for name, echelon in self.des.echelons.items():
            if echelon.echelon_type == "factory":
                factory_echelon = echelon
                break

        if factory_echelon:
            warehouse_cap = self.env_config.get("warehouse_capacity", 2000)
            self._shared_buffer["factory_inventory"] = factory_echelon.inventory / warehouse_cap
            self._shared_buffer["factory_backlog"] = factory_echelon.backlog / mean_demand
            self._shared_buffer["supplier_capacity"] = (
                self.des.supplier_capacity / self.env_config.get("max_supplier_capacity", 300)
            )

    def _get_agent_observation(self, agent_name: str) -> np.ndarray:
        """
        Build the observation vector for a specific agent.

        Structure:
        [LOCAL: inv, pipeline, backlog, fill_rate, last_order]
        [SHARED: demand_history (14), factory_signals (3), other_invs (n-1),
                 shock_indicators (3), time_features (2)]
        """
        obs_parts = []

        # --- Local features (agent's own echelon) ---
        es = self.des.echelons[agent_name]
        warehouse_cap = self.env_config.get("warehouse_capacity", 2000)
        max_order = self.env_config.get("max_order_quantity", 500)
        mean_demand = self.env_config.get("mean_demand", 50)

        obs_parts.extend([
            es.inventory / warehouse_cap,
            es.pipeline_inventory / max_order,
            es.backlog / mean_demand,
            es.fill_rate,
            es.last_order_qty / max_order,
        ])

        # --- Shared demand signals ---
        if self.share_demand_signals:
            demand_hist = self._shared_buffer.get("demand_history", [0.0] * self.DEMAND_HISTORY_WINDOW)
            obs_parts.extend(demand_hist)

        # --- Shared capacity signals ---
        if self.share_capacity_signals:
            obs_parts.extend([
                self._shared_buffer.get("factory_inventory", 0.0),
                self._shared_buffer.get("factory_backlog", 0.0),
                self._shared_buffer.get("supplier_capacity", 1.0),
            ])

        # --- Other agents' inventory levels ---
        for other_name in self.echelon_names:
            if other_name != agent_name:
                other_es = self.des.echelons[other_name]
                obs_parts.append(other_es.inventory / warehouse_cap)

        # --- Shock indicators ---
        indicators = self.shock_injector.get_shock_indicators()
        obs_parts.extend([
            indicators["demand_shock"],
            indicators["supply_shock"],
            indicators["logistics_shock"],
        ])

        # --- Time features ---
        if self._step_count > 0:
            day_of_week = (self._step_count % 7) / 7.0
            week_of_year = (self._step_count % 52) / 52.0
        else:
            day_of_week = 0.0
            week_of_year = 0.0
        obs_parts.extend([day_of_week, week_of_year])

        return np.array(obs_parts, dtype=np.float32)

    def render(self) -> str | None:
        """Render current state as text."""
        if self.render_mode in ("ansi", "human"):
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
            output = "\n".join(lines)
            if self.render_mode == "human":
                print(output)
            return output
        return None

    @property
    def episode_metrics(self) -> list[dict]:
        """Full per-step metrics for the current episode."""
        return self._episode_metrics
