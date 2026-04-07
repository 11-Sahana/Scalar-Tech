"""
scripts/run_env.py
------------------
Manually step through the Supply Disruption environment with a hard-coded
rule-based policy.  Useful for sanity-checking the environment without
needing an OpenAI API key.

Usage:
    python scripts/run_env.py [--seed 42]
"""

from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.env import SupplyDisruptionEnv
from app.models import Action, OrderStatus
from app.tasks import get_reference_contingency_plan
from app.utils import setup_logging, compute_order_urgency_score


def rule_based_action(obs) -> Action:
    plan = get_reference_contingency_plan(obs)
    return Action(
        fulfill_orders=plan["fulfill_orders"],
        delay_orders=plan["delay_orders"],
        cancel_orders=plan["cancel_orders"],
        use_supplier=plan["use_supplier"],
        reasoning="Rule-based: fulfil highest-urgency orders within budget.",
    )


def main():
    parser = argparse.ArgumentParser(description="Run Supply Disruption Env manually.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    env = SupplyDisruptionEnv(seed=args.seed)
    obs = env.reset()

    print("=" * 70)
    print("  SUPPLY DISRUPTION RESPONDER  -  Manual Run")
    print("=" * 70)
    print(f"\nDisruption: {obs.disruption.description}")
    print(f"Budget    : ${obs.budget_remaining:,.2f}")
    sku = obs.disruption.affected_sku
    print(f"Inventory : {obs.inventory[sku].on_hand_qty} units of {sku}\n")

    print("Customer Orders:")
    for o in sorted(obs.orders, key=compute_order_urgency_score, reverse=True):
        print(f"  {o.order_id}  priority={o.priority:8s}  qty={o.quantity:3d}"
              f"  deadline={o.deadline_days}d  revenue=${o.revenue:,.0f}")

    print("\nAlternative Suppliers:")
    for s in obs.suppliers:
        disrupted = " <- DISRUPTED" if s.is_primary else ""
        print(f"  {s.supplier_id}  reliability={s.reliability:.0%}  "
              f"cost=${s.unit_cost:.2f}/u  lead={s.lead_time_days}d"
              f"  avail={s.available_qty}{disrupted}")

    total_reward = 0.0
    step = 0
    while not obs.done:
        action = rule_based_action(obs)
        print(f"\n-- Step {step + 1} Action ----------------------")
        print(f"  Fulfil  : {action.fulfill_orders}")
        print(f"  Delay   : {action.delay_orders}")
        print(f"  Cancel  : {action.cancel_orders}")
        print(f"  Supplier: {action.use_supplier}")

        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        step += 1

        print(f"\n  Step reward   : {reward.value:+.3f}")
        print(f"  Fulfillment   : {reward.breakdown.fulfillment_score:+.3f}")
        print(f"  Delay penalty : {reward.breakdown.delay_penalty:+.3f}")
        print(f"  Cancel penalty: {reward.breakdown.cancel_penalty:+.3f}")
        print(f"  Budget penalty: {reward.breakdown.budget_penalty:+.3f}")
        print(f"  Supplier bonus: {reward.breakdown.supplier_bonus:+.3f}")

        if info.get("warnings"):
            print(f"  Warnings: {info['warnings']}")

        if done:
            break

    final = env.state()
    print("\n" + "=" * 70)
    print("EPISODE COMPLETE")
    print("=" * 70)
    print(f"Steps taken      : {step}")
    print(f"Total reward     : {total_reward:+.3f}")
    print(f"Budget remaining : ${final['budget_remaining']:,.2f}")
    print("\nFinal Order Statuses:")
    for o in final["orders"]:
        status = o["status"].replace("OrderStatus.", "") if "OrderStatus." in str(o["status"]) else o["status"]
        print(f"  {o['order_id']:8s}  {o['priority']:8s}  -> {status}")

    print("\n(Run scripts/evaluate.py to get formal task scores.)")


if __name__ == "__main__":
    main()