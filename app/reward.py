"""
Reward function for the Supply Disruption Responder.

Provides a dense, incremental reward that signals quality of each decision.
Importing this module separately lets researchers swap reward schemes without
touching the core environment logic.
"""

from __future__ import annotations
from typing import Dict, List

from app.models import (
    Action, CustomerOrder, OrderPriority, OrderStatus,
    Reward, RewardBreakdown, Supplier,
)
from app.utils import CANCEL_PENALTY_FACTOR, DELAY_PENALTY_FACTOR, PRIORITY_WEIGHT


# ---------------------------------------------------------------------------
# Configurable reward hyper-parameters
# ---------------------------------------------------------------------------

class RewardConfig:
    # Revenue multiplier per priority weight unit
    FULFILLMENT_SCALE: float = 0.01
    # Penalty multiplier on order revenue for delays
    DELAY_SCALE:       float = 0.005
    # Penalty multiplier on order revenue for cancellations
    CANCEL_SCALE:      float = 0.008
    # Flat bonus for reliability (reliability * SUPPLIER_BONUS_SCALE)
    SUPPLIER_BONUS_SCALE: float = 50.0
    # Penalty rate for every dollar over budget
    OVER_BUDGET_RATE:  float = 0.1


def compute_step_reward(
    step: int,
    fulfilled_orders:  List[CustomerOrder],
    delayed_orders:    List[CustomerOrder],
    cancelled_orders:  List[CustomerOrder],
    chosen_supplier:   Supplier | None,
    budget_overage:    float,
    cfg: RewardConfig = RewardConfig(),
) -> Reward:
    """
    Compute a dense step-level reward given the outcomes of a single action.

    Parameters
    ----------
    step             : current step index
    fulfilled_orders : orders that were successfully fulfilled this step
    delayed_orders   : orders marked as delayed this step
    cancelled_orders : orders marked as cancelled this step
    chosen_supplier  : the supplier activated this step (or None)
    budget_overage   : how much the budget was exceeded (0 if within budget)
    cfg              : reward hyper-parameters
    """
    rb = RewardBreakdown()

    # ── Positive: fulfillment revenue weighted by priority ──
    for order in fulfilled_orders:
        pw = PRIORITY_WEIGHT.get(order.priority, 1.0)
        rb.fulfillment_score += order.revenue * pw * cfg.FULFILLMENT_SCALE

    # ── Negative: delay penalty weighted by priority and revenue ──
    for order in delayed_orders:
        pf = DELAY_PENALTY_FACTOR.get(order.priority, 0.5)
        rb.delay_penalty -= order.revenue * pf * cfg.DELAY_SCALE

    # ── Negative: cancellation penalty (steeper than delay) ──
    for order in cancelled_orders:
        pf = CANCEL_PENALTY_FACTOR.get(order.priority, 1.0)
        rb.cancel_penalty -= order.revenue * pf * cfg.CANCEL_SCALE

    # ── Negative: budget over-run ──
    if budget_overage > 0:
        rb.budget_penalty = -(budget_overage * cfg.OVER_BUDGET_RATE)

    # ── Positive: reliable supplier bonus ──
    if chosen_supplier and not chosen_supplier.is_primary:
        rb.supplier_bonus += chosen_supplier.reliability * cfg.SUPPLIER_BONUS_SCALE

    rb.total = (
        rb.fulfillment_score
        + rb.delay_penalty
        + rb.cancel_penalty
        + rb.budget_penalty
        + rb.supplier_bonus
    )

    return Reward(
        step=step,
        value=rb.total,
        breakdown=rb,
    )


def normalise_reward(raw: float, min_r: float = -500.0, max_r: float = 500.0) -> float:
    """Map raw reward to [-1, 1] for RL training use."""
    return max(-1.0, min(1.0, (raw - min_r) / (max_r - min_r) * 2 - 1))
