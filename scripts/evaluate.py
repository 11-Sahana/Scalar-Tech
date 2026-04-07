"""
scripts/evaluate.py
────────────────────
Evaluate the Supply Disruption environment across all three tasks using
a deterministic rule-based submission.  Prints per-task scores and a
weighted overall score.

Usage:
    python scripts/evaluate.py [--seed 42] [--verbose]
"""

from __future__ import annotations
import argparse
import json
import sys
import os
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.env import SupplyDisruptionEnv
from app.grader import grade
from app.tasks import (
    TASKS,
    get_at_risk_order_ids,
    get_optimal_supplier_ranking,
    get_reference_contingency_plan,
)
from app.utils import setup_logging


# ---------------------------------------------------------------------------
# Deterministic reference submissions
# ---------------------------------------------------------------------------

def make_easy_submission(obs) -> Dict[str, Any]:
    """Perfect submission: uses the ground-truth helper directly."""
    return {"at_risk_order_ids": get_at_risk_order_ids(obs)}


def make_medium_submission(obs) -> Dict[str, Any]:
    """Perfect submission: uses the ground-truth ranking helper."""
    return {
        "ranked_supplier_ids": get_optimal_supplier_ranking(obs),
        "rationale": "Ranked by composite score: reliability × 0.5 + cost_inv × 0.3 + time_inv × 0.2",
    }


def make_hard_submission(obs) -> Dict[str, Any]:
    """Near-optimal submission from the reference planner."""
    plan = get_reference_contingency_plan(obs)
    return {
        "fulfill_orders": plan["fulfill_orders"],
        "delay_orders":   plan["delay_orders"],
        "cancel_orders":  plan["cancel_orders"],
        "use_supplier":   plan["use_supplier"],
        "reasoning":      "Greedy: sorted by urgency, procure from best supplier within budget.",
    }


# ---------------------------------------------------------------------------
# Imperfect submissions (to demonstrate non-trivial scoring)
# ---------------------------------------------------------------------------

def make_imperfect_easy_submission(obs) -> Dict[str, Any]:
    """Returns only half of at-risk orders (simulates partial recall)."""
    at_risk = get_at_risk_order_ids(obs)
    return {"at_risk_order_ids": at_risk[: len(at_risk) // 2 + 1]}


def make_imperfect_medium_submission(obs) -> Dict[str, Any]:
    """Returns suppliers in reverse-optimal order."""
    ranking = get_optimal_supplier_ranking(obs)
    return {"ranked_supplier_ids": list(reversed(ranking))}


def make_imperfect_hard_submission(obs) -> Dict[str, Any]:
    """Cancels instead of delays; picks a suboptimal supplier."""
    from app.models import OrderPriority, OrderStatus
    from app.tasks import get_optimal_supplier_ranking
    pending = [o for o in obs.orders if o.status == OrderStatus.PENDING]
    fulfill = [o.order_id for o in pending if o.priority == OrderPriority.CRITICAL]
    cancel  = [o.order_id for o in pending if o.priority != OrderPriority.CRITICAL]
    ranking = get_optimal_supplier_ranking(obs)
    worst   = ranking[-1] if ranking else None
    return {
        "fulfill_orders": fulfill,
        "delay_orders":   [],
        "cancel_orders":  cancel,
        "use_supplier":   worst,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

WEIGHTS = {"easy": 0.20, "medium": 0.30, "hard": 0.50}


def run_evaluation(seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    setup_logging()
    env = SupplyDisruptionEnv(seed=seed)
    obs = env.reset()

    submissions = {
        "easy":   make_easy_submission(obs),
        "medium": make_medium_submission(obs),
        "hard":   make_hard_submission(obs),
    }

    imperfect_submissions = {
        "easy":   make_imperfect_easy_submission(obs),
        "medium": make_imperfect_medium_submission(obs),
        "hard":   make_imperfect_hard_submission(obs),
    }

    print("\n" + "=" * 70)
    print("  SUPPLY DISRUPTION RESPONDER  –  Evaluation Report")
    print("=" * 70)

    all_scores = {}

    for task_id, meta in TASKS.items():
        print(f"\n── Task: {meta['name']} ({task_id.upper()}) ─────────────────────────────")
        print(f"   {meta['description'][:120]}...")

        # Reference (near-optimal) score
        ref_result = grade(task_id, submissions[task_id], obs)
        ref_score = ref_result["score"]

        # Imperfect score
        imp_result = grade(task_id, imperfect_submissions[task_id], obs)
        imp_score = imp_result["score"]

        all_scores[task_id] = ref_score

        print(f"\n   Reference (near-optimal) score : {ref_score:.4f}")
        print(f"   Imperfect agent score          : {imp_score:.4f}")

        if verbose:
            print("\n   Reference details:")
            for k, v in ref_result.items():
                if k not in ("task_id",):
                    print(f"     {k}: {v}")
            print("\n   Imperfect details:")
            for k, v in imp_result.items():
                if k not in ("task_id",):
                    print(f"     {k}: {v}")

    # Weighted overall
    weighted_total = sum(WEIGHTS[tid] * all_scores[tid] for tid in TASKS)
    print("\n" + "=" * 70)
    print("  WEIGHTED OVERALL SCORE (reference submissions)")
    print("=" * 70)
    for tid, sc in all_scores.items():
        print(f"  {tid:6s}  (weight={WEIGHTS[tid]:.2f})  score={sc:.4f}  contribution={WEIGHTS[tid]*sc:.4f}")
    print(f"\n  FINAL SCORE: {weighted_total:.4f}")
    print("=" * 70)

    # Scenario summary
    sku = obs.disruption.affected_sku
    print(f"\nScenario seed={seed}")
    print(f"Budget : ${obs.budget_remaining:,.2f}")
    print(f"Inventory {sku}: {obs.inventory[sku].on_hand_qty} units")
    print(f"Orders : {len(obs.orders)}")
    print(f"Suppliers (non-disrupted): {sum(1 for s in obs.suppliers if not s.is_primary)}")

    return {
        "seed": seed,
        "task_scores": all_scores,
        "weighted_score": weighted_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all Supply Disruption tasks.")
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_evaluation(seed=args.seed, verbose=args.verbose)
    sys.exit(0)
