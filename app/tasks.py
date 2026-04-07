"""
Task definitions for the Supply Disruption Responder.

Three tasks of increasing difficulty:
  EASY   – Identify at-risk orders
  MEDIUM – Rank alternative suppliers
  HARD   – Produce a full contingency plan
"""

from __future__ import annotations
from typing import Any, Dict, List

from app.models import Observation, OrderPriority, OrderStatus
from app.utils import compute_order_urgency_score, supplier_score


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id":          "easy",
        "name":        "At-Risk Order Identification",
        "difficulty":  "easy",
        "description": (
            "Given the disruption event and current inventory levels, "
            "identify which customer orders are at risk of not being fulfilled "
            "on time.  Output the list of at-risk order IDs, sorted by urgency."
        ),
        "output_schema": {
            "at_risk_order_ids": "List[str]",
        },
    },
    "medium": {
        "id":          "medium",
        "name":        "Supplier Ranking",
        "difficulty":  "medium",
        "description": (
            "Given the alternative suppliers and the total units required, "
            "rank the non-disrupted suppliers from best to worst choice. "
            "Consider cost, lead time, reliability, and available quantity."
        ),
        "output_schema": {
            "ranked_supplier_ids": "List[str]",
            "rationale":           "str (optional)",
        },
    },
    "hard": {
        "id":          "hard",
        "name":        "Full Contingency Plan",
        "difficulty":  "hard",
        "description": (
            "Produce a complete contingency plan.  Specify which orders to "
            "fulfil, delay, or cancel, and which supplier to engage. "
            "The plan must respect the budget constraint, maximise revenue "
            "from high-priority orders, and minimise unnecessary delays."
        ),
        "output_schema": {
            "fulfill_orders": "List[str]",
            "delay_orders":   "List[str]",
            "cancel_orders":  "List[str]",
            "use_supplier":   "str",
            "reasoning":      "str (optional)",
        },
    },
}


# ---------------------------------------------------------------------------
# Helper: compute ground-truth for grader reference
# ---------------------------------------------------------------------------

def get_at_risk_order_ids(obs: Observation) -> List[str]:
    """
    Returns the IDs of orders considered 'at risk'.
    An order is at risk when:
      - It is still pending, AND
      - Available on-hand inventory is insufficient to cover all pending orders
        ahead of and including this one when sorted by urgency.
    """
    sku = obs.disruption.affected_sku
    inventory_available = obs.inventory[sku].available_qty

    pending = [o for o in obs.orders if o.status == OrderStatus.PENDING]
    sorted_orders = sorted(pending, key=compute_order_urgency_score, reverse=True)

    at_risk = []
    running_demand = 0
    for order in sorted_orders:
        running_demand += order.quantity
        if running_demand > inventory_available:
            at_risk.append(order.order_id)

    return at_risk


def get_optimal_supplier_ranking(obs: Observation) -> List[str]:
    """
    Rank non-disrupted, available suppliers by composite score (descending).
    The disrupted primary supplier is excluded.
    """
    available = [
        s for s in obs.suppliers
        if not s.is_primary and s.available_qty > 0
    ]
    ranked = sorted(available, key=supplier_score, reverse=True)
    return [s.supplier_id for s in ranked]


def get_reference_contingency_plan(obs: Observation) -> Dict[str, Any]:
    """
    Compute a near-optimal reference plan used by the hard task grader.
    Strategy:
      1. Sort orders by urgency (priority + deadline).
      2. Pick best supplier.
      3. Greedily fulfil top-urgency orders within budget + procurable stock.
      4. Delay medium/low that can't be fulfilled in time.
      5. Cancel only if no path to fulfillment and order is low-priority.
    """
    sku = obs.disruption.affected_sku
    on_hand = obs.inventory[sku].available_qty
    budget  = obs.budget_remaining

    best_supplier_ids = get_optimal_supplier_ranking(obs)
    best_supplier_id  = best_supplier_ids[0] if best_supplier_ids else None
    best_supplier     = next(
        (s for s in obs.suppliers if s.supplier_id == best_supplier_id), None
    )

    # How much stock can we afford from best supplier?
    procurable = 0
    if best_supplier:
        procurable = min(
            best_supplier.available_qty,
            int(budget // best_supplier.unit_cost),
        )

    total_available = on_hand + procurable

    pending = [o for o in obs.orders if o.status == OrderStatus.PENDING]
    sorted_orders = sorted(pending, key=compute_order_urgency_score, reverse=True)

    fulfill, delay, cancel = [], [], []
    remaining = total_available
    for order in sorted_orders:
        if remaining >= order.quantity:
            fulfill.append(order.order_id)
            remaining -= order.quantity
        else:
            # Delay high-priority, cancel truly low-priority if heavily over
            if order.priority in (OrderPriority.LOW,) and remaining == 0:
                cancel.append(order.order_id)
            else:
                delay.append(order.order_id)

    return {
        "fulfill_orders": fulfill,
        "delay_orders":   delay,
        "cancel_orders":  cancel,
        "use_supplier":   best_supplier_id,
    }
