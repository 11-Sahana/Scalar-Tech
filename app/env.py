"""
Core OpenEnv-compatible environment for the Supply Disruption Responder.

Interface:
    reset()  → Observation
    step(action) → (Observation, Reward, done: bool, info: dict)
    state()  → dict  (full internal state snapshot)
"""

from __future__ import annotations
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.models import (
    Action, CustomerOrder, Observation, OrderPriority,
    OrderStatus, Reward, RewardBreakdown, Supplier,
)
from app.utils import (
    CANCEL_PENALTY_FACTOR, DELAY_PENALTY_FACTOR, PRIORITY_WEIGHT,
    generate_scenario, order_map, supplier_map, supplier_score,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 5   # agent has at most 5 decision turns per episode


class SupplyDisruptionEnv:
    """
    OpenEnv-compatible environment simulating a supply chain disruption.

    The agent iteratively decides which orders to fulfil / delay / cancel
    and which alternative supplier to activate.  The episode ends when:
      - all orders have been assigned a final status, OR
      - the agent exceeds MAX_STEPS.
    """

    def __init__(self, seed: int = 42, scenario_override=None):
        self.seed = seed
        self.scenario_override = scenario_override
        self._scenario: Dict[str, Any] = {}
        self._step_count: int = 0
        self._orders: Dict[str, CustomerOrder] = {}
        self._suppliers: Dict[str, Supplier] = {}
        self._budget_remaining: float = 0.0
        self._cumulative_reward: float = 0.0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Initialise (or re-initialise) the environment and return the first observation."""
        #self._scenario = generate_scenario(self.seed)
        if self.scenario_override:
            self._scenario = self.scenario_override
        else:
            self._scenario = generate_scenario(self.seed)
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._history = []

        # Deep-copy so mutations don't bleed into the scenario template
        self._orders = order_map(
            [copy.deepcopy(o) for o in self._scenario["orders"]]
        )
        self._suppliers = supplier_map(
            [copy.deepcopy(s) for s in self._scenario["suppliers"]]
        )
        self._budget_remaining = float(self._scenario["budget"])
        self._inventory = copy.deepcopy(self._scenario["inventory"])

        obs = self._build_observation()
        logger.info("Environment reset. Budget=%.2f, Orders=%d",
                    self._budget_remaining, len(self._orders))
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action and advance the simulation by one step.

        Returns
        -------
        observation  : Observation  - updated environment view
        reward       : Reward       - dense reward with breakdown
        done         : bool         - True when episode ends
        info         : dict         - auxiliary diagnostics
        """
        if not action.validate_disjoint():
            raise ValueError("Action contains overlapping order IDs across lists.")

        self._step_count += 1
        reward_breakdown = RewardBreakdown()
        info: Dict[str, Any] = {"step": self._step_count, "warnings": []}

        # ---- 1. Resolve supplier procurement ----
        procured_units = 0
        if action.use_supplier:
            supplier = self._suppliers.get(action.use_supplier)
            if supplier is None:
                info["warnings"].append(f"Unknown supplier: {action.use_supplier}")
            elif supplier.available_qty == 0:
                info["warnings"].append(f"Supplier {action.use_supplier} has no stock.")
            else:
                # Figure out how many units we need across all fulfil orders
                needed = sum(
                    self._orders[oid].quantity
                    for oid in action.fulfill_orders
                    if oid in self._orders and
                       self._orders[oid].status == OrderStatus.PENDING
                )
                on_hand = self._inventory[self._scenario["sku"]].on_hand_qty
                shortfall = max(0, needed - on_hand)
                to_buy = min(shortfall, supplier.available_qty)

                cost = to_buy * supplier.unit_cost
                if cost > self._budget_remaining:
                    # Buy what we can afford
                    to_buy = int(self._budget_remaining // supplier.unit_cost)
                    cost = to_buy * supplier.unit_cost
                    info["warnings"].append(
                        f"Budget limited procurement to {to_buy} units."
                    )

                self._budget_remaining -= cost
                self._inventory[self._scenario["sku"]].on_hand_qty += to_buy
                supplier.available_qty -= to_buy
                procured_units = to_buy

                # Reliability bonus
                reward_breakdown.supplier_bonus += supplier.reliability * 50
                logger.debug("Procured %d units from %s at cost %.2f",
                             to_buy, action.use_supplier, cost)

        # ---- 2. Fulfil orders ----
        inventory_item = self._inventory[self._scenario["sku"]]
        for oid in action.fulfill_orders:
            order = self._orders.get(oid)
            if order is None:
                info["warnings"].append(f"Unknown order: {oid}")
                continue
            if order.status != OrderStatus.PENDING:
                continue
            if inventory_item.on_hand_qty >= order.quantity:
                inventory_item.on_hand_qty -= order.quantity
                order.status = OrderStatus.FULFILLED
                pw = PRIORITY_WEIGHT.get(order.priority, 1.0)
                reward_breakdown.fulfillment_score += order.revenue * pw * 0.01
                logger.debug("Fulfilled order %s (priority=%s)", oid, order.priority)
            else:
                info["warnings"].append(
                    f"Insufficient inventory for order {oid} "
                    f"(need {order.quantity}, have {inventory_item.on_hand_qty})."
                )

        # ---- 3. Delay orders ----
        for oid in action.delay_orders:
            order = self._orders.get(oid)
            if order is None or order.status != OrderStatus.PENDING:
                continue
            order.status = OrderStatus.DELAYED
            pf = DELAY_PENALTY_FACTOR.get(order.priority, 0.5)
            reward_breakdown.delay_penalty -= order.revenue * pf * 0.005
            logger.debug("Delayed order %s", oid)

        # ---- 4. Cancel orders ----
        for oid in action.cancel_orders:
            order = self._orders.get(oid)
            if order is None or order.status != OrderStatus.PENDING:
                continue
            order.status = OrderStatus.CANCELLED
            pf = CANCEL_PENALTY_FACTOR.get(order.priority, 1.0)
            reward_breakdown.cancel_penalty -= order.revenue * pf * 0.008
            logger.debug("Cancelled order %s", oid)

        # ---- 5. Budget penalty (if over) ----
        if self._budget_remaining < 0:
            reward_breakdown.budget_penalty += self._budget_remaining * 0.1  # negative
            info["warnings"].append("Budget exceeded!")

        # ---- 6. Aggregate reward ----
        reward_breakdown.total = (
            reward_breakdown.fulfillment_score
            + reward_breakdown.delay_penalty
            + reward_breakdown.cancel_penalty
            + reward_breakdown.budget_penalty
            + reward_breakdown.supplier_bonus
        )
        self._cumulative_reward += reward_breakdown.total

        reward = Reward(
            step=self._step_count,
            value=reward_breakdown.total,
            breakdown=reward_breakdown,
            info={"cumulative": self._cumulative_reward, "procured_units": procured_units},
        )

        # ---- 7. Log history ----
        self._history.append({
            "step": self._step_count,
            "action": action.dict(),
            "reward": reward.dict(),
        })

        # ---- 8. Check termination ----
        done = self._is_done()
        obs = self._build_observation(done=done)

        logger.info("Step %d | reward=%.3f | cumulative=%.3f | done=%s",
                    self._step_count, reward.value, self._cumulative_reward, done)
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return a full snapshot of internal state (for debugging / evaluation)."""
        return {
            "step": self._step_count,
            "budget_remaining": self._budget_remaining,
            "cumulative_reward": self._cumulative_reward,
            "inventory": {
                sku: item.dict() for sku, item in self._inventory.items()
            },
            "orders": [o.dict() for o in self._orders.values()],
            "suppliers": [s.dict() for s in self._suppliers.values()],
            "history": self._history,
            "done": self._is_done(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        pending = [o for o in self._orders.values() if o.status == OrderStatus.PENDING]
        return len(pending) == 0 or self._step_count >= MAX_STEPS

    def _build_observation(self, done: bool = False) -> Observation:
        pending_count   = sum(1 for o in self._orders.values() if o.status == OrderStatus.PENDING)
        fulfilled_count = sum(1 for o in self._orders.values() if o.status == OrderStatus.FULFILLED)
        delayed_count   = sum(1 for o in self._orders.values() if o.status == OrderStatus.DELAYED)
        cancelled_count = sum(1 for o in self._orders.values() if o.status == OrderStatus.CANCELLED)

        return Observation(
            step=self._step_count,
            budget_remaining=self._budget_remaining,
            inventory=self._inventory,
            orders=list(self._orders.values()),
            suppliers=list(self._suppliers.values()),
            disruption=self._scenario["disruption"],
            metrics={
                "pending_orders":   pending_count,
                "fulfilled_orders": fulfilled_count,
                "delayed_orders":   delayed_count,
                "cancelled_orders": cancelled_count,
                "cumulative_reward": self._cumulative_reward,
            },
            done=done,
        )
