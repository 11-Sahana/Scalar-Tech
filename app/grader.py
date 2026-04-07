"""
Grading logic for all three task difficulty levels.
Each grader returns a normalised score in [0.0, 1.0].
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

from app.models import Observation, OrderPriority
from app.tasks import (
    get_at_risk_order_ids,
    get_optimal_supplier_ranking,
    get_reference_contingency_plan,
)
from app.utils import PRIORITY_WEIGHT, supplier_score


# ---------------------------------------------------------------------------
# Easy grader
# ---------------------------------------------------------------------------

def grade_easy(
    submission: Dict[str, Any],
    obs: Observation,
) -> Dict[str, float]:
    """
    Task: identify at-risk orders.
    Metric: F1 score between submitted IDs and ground-truth IDs.
    """
    predicted: List[str] = submission.get("at_risk_order_ids", [])
    ground_truth: List[str] = get_at_risk_order_ids(obs)

    pred_set = set(predicted)
    gt_set   = set(ground_truth)

    if not gt_set and not pred_set:
        return {"score": 1.0, "detail": "No at-risk orders; correctly predicted none."}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "score":     round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "predicted": sorted(predicted),
        "expected":  sorted(ground_truth),
    }


# ---------------------------------------------------------------------------
# Medium grader
# ---------------------------------------------------------------------------

def grade_medium(
    submission: Dict[str, Any],
    obs: Observation,
) -> Dict[str, float]:
    """
    Task: rank suppliers.
    Metric: Spearman rank-correlation between submitted and optimal rankings.
    Only non-disrupted suppliers with available stock are considered.
    """
    submitted_ranking: List[str] = submission.get("ranked_supplier_ids", [])
    optimal_ranking:   List[str] = get_optimal_supplier_ranking(obs)

    # Filter submission to only valid, available supplier IDs
    valid_ids = set(optimal_ranking)
    submitted_filtered = [s for s in submitted_ranking if s in valid_ids]

    # If submission is missing or completely wrong
    if not submitted_filtered or not optimal_ranking:
        return {"score": 0.0, "detail": "Empty or invalid supplier ranking."}

    # Build rank maps (rank starts at 1 for best)
    def rank_map(lst: List[str]) -> Dict[str, int]:
        return {sid: i + 1 for i, sid in enumerate(lst)}

    opt_ranks  = rank_map(optimal_ranking)
    sub_ranks  = {sid: submitted_filtered.index(sid) + 1
                  for sid in submitted_filtered if sid in submitted_filtered}

    # Compute Spearman correlation over the intersection
    common = [sid for sid in optimal_ranking if sid in sub_ranks]
    if len(common) < 2:
        # Score based purely on whether the top pick matches
        top_match = 1.0 if (submitted_filtered and submitted_filtered[0] == optimal_ranking[0]) else 0.0
        return {"score": top_match, "detail": "Insufficient overlap for Spearman; graded on top pick."}

    n  = len(common)
    d2 = sum((opt_ranks[sid] - sub_ranks[sid]) ** 2 for sid in common)
    rho = 1 - (6 * d2) / (n * (n ** 2 - 1))

    # Normalise rho from [-1,1] to [0,1]
    score = (rho + 1) / 2

    return {
        "score":            round(score, 4),
        "spearman_rho":     round(rho,   4),
        "submitted_ranking": submitted_filtered,
        "optimal_ranking":   optimal_ranking,
    }


# ---------------------------------------------------------------------------
# Hard grader
# ---------------------------------------------------------------------------

def grade_hard(
    submission: Dict[str, Any],
    obs: Observation,
) -> Dict[str, float]:
    """
    Task: full contingency plan.
    Sub-scores (each 0–1), then weighted average.

      Component                   Weight
      ─────────────────────────────────
      High-priority fulfillment    0.40
      Budget adherence             0.20
      Minimal delay score          0.20
      Supplier optimality          0.20
    """
    ref = get_reference_contingency_plan(obs)

    fulfill_ids = set(submission.get("fulfill_orders", []))
    delay_ids   = set(submission.get("delay_orders",   []))
    cancel_ids  = set(submission.get("cancel_orders",  []))
    supplier_id = submission.get("use_supplier")

    order_map_  = {o.order_id: o for o in obs.orders}

    # ── 1. High-priority fulfillment score ──
    ref_fulfill_set = set(ref["fulfill_orders"])
    high_pri_ids = {
        oid for oid in ref_fulfill_set
        if order_map_.get(oid) and order_map_[oid].priority
           in (OrderPriority.CRITICAL, OrderPriority.HIGH)
    }
    if high_pri_ids:
        hp_fulfilled = fulfill_ids & high_pri_ids
        hp_score = len(hp_fulfilled) / len(high_pri_ids)
    else:
        hp_score = 1.0

    # ── 2. Budget adherence ──
    chosen_supplier = next(
        (s for s in obs.suppliers if s.supplier_id == supplier_id), None
    )
    total_qty_to_buy = max(
        0,
        sum(order_map_[oid].quantity for oid in fulfill_ids if oid in order_map_)
        - obs.inventory[obs.disruption.affected_sku].available_qty,
    )
    procurement_cost = (
        total_qty_to_buy * chosen_supplier.unit_cost if chosen_supplier else 0
    )
    over_budget = max(0, procurement_cost - obs.budget_remaining)
    if obs.budget_remaining > 0:
        budget_score = max(0.0, 1.0 - over_budget / obs.budget_remaining)
    else:
        budget_score = 1.0 if over_budget == 0 else 0.0

    # ── 3. Minimal delay score ──
    # Penalise unnecessary delays of high-priority orders that *could* have been fulfilled
    unnecessary_delays = (delay_ids | cancel_ids) & set(ref["fulfill_orders"])
    # Weight by priority
    penalty = sum(
        PRIORITY_WEIGHT.get(order_map_[oid].priority, 1.0)
        for oid in unnecessary_delays if oid in order_map_
    )
    max_penalty = sum(
        PRIORITY_WEIGHT.get(order_map_[oid].priority, 1.0)
        for oid in ref["fulfill_orders"] if oid in order_map_
    )
    delay_score = max(0.0, 1.0 - (penalty / max_penalty)) if max_penalty > 0 else 1.0

    # ── 4. Supplier optimality ──
    optimal_ranking = get_optimal_supplier_ranking(obs)
    if supplier_id and optimal_ranking:
        if supplier_id == optimal_ranking[0]:
            supplier_opt_score = 1.0
        elif supplier_id in optimal_ranking:
            idx = optimal_ranking.index(supplier_id)
            supplier_opt_score = max(0.0, 1.0 - idx / len(optimal_ranking))
        else:
            # Chose disrupted or unknown supplier
            supplier_opt_score = 0.0
    else:
        supplier_opt_score = 0.0

    # ── Weighted total ──
    weights = {"hp": 0.40, "budget": 0.20, "delay": 0.20, "supplier": 0.20}
    total = (
        weights["hp"]       * hp_score
        + weights["budget"]   * budget_score
        + weights["delay"]    * delay_score
        + weights["supplier"] * supplier_opt_score
    )

    return {
        "score":                    round(total, 4),
        "high_priority_score":      round(hp_score,           4),
        "budget_adherence_score":   round(budget_score,       4),
        "minimal_delay_score":      round(delay_score,        4),
        "supplier_optimality_score":round(supplier_opt_score, 4),
        "procurement_cost":         round(procurement_cost,   2),
        "budget_remaining":         round(obs.budget_remaining, 2),
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def grade(
    task_id: str,
    submission: Dict[str, Any],
    obs: Observation,
) -> Dict[str, Any]:
    """Dispatch to the appropriate grader and return result with task metadata."""
    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task: {task_id!r}. Valid: {list(graders)}")

    result = graders[task_id](submission, obs)
    result["task_id"] = task_id
    return result
