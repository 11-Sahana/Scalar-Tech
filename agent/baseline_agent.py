"""
Baseline AI agent for the Supply Disruption Responder.

Uses the OpenAI Chat Completions API to convert environment observations
into structured actions.  Runs a full episode and reports the final score.

Usage:
    OPENAI_API_KEY=sk-... python agent/baseline_agent.py
"""

from __future__ import annotations
import json
import logging
import os
import sys
import textwrap
from typing import Any, Dict

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.env import SupplyDisruptionEnv
from app.grader import grade
from app.models import Action, Observation
from app.tasks import get_reference_contingency_plan
from app.utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return textwrap.dedent("""
    You are an expert supply chain risk manager.
    You are given a structured observation of a supply chain disruption scenario.
    Your task is to produce a JSON action object that best handles the situation.

    The action MUST follow this exact schema (all fields required):
    {
      "fulfill_orders": ["ORD-xxx", ...],   // order IDs to fulfil now
      "delay_orders":   ["ORD-xxx", ...],   // order IDs to delay
      "cancel_orders":  ["ORD-xxx", ...],   // order IDs to cancel
      "use_supplier":   "SUP-xxx",          // supplier ID to source from
      "reasoning":      "brief explanation"
    }

    Rules:
    - Each order must appear in exactly ONE list (fulfill, delay, or cancel).
    - Choose the supplier with the best balance of reliability, cost, and lead time.
    - Do NOT exceed the available budget.
    - Prioritise CRITICAL and HIGH priority orders for fulfillment.
    - Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.
    """).strip()


def build_user_prompt(obs: Observation) -> str:
    inv_sku = obs.disruption.affected_sku
    inv_qty = obs.inventory.get(inv_sku)
    on_hand = inv_qty.on_hand_qty if inv_qty else 0

    orders_txt = "\n".join(
        f"  {o.order_id}: customer={o.customer}, priority={o.priority}, "
        f"qty={o.quantity}, deadline_days={o.deadline_days}, "
        f"revenue=${o.revenue:.0f}, status={o.status}"
        for o in obs.orders
    )

    suppliers_txt = "\n".join(
        f"  {s.supplier_id}: name={s.name}, unit_cost=${s.unit_cost:.2f}, "
        f"lead_time={s.lead_time_days}d, reliability={s.reliability:.0%}, "
        f"available_qty={s.available_qty}"
        for s in obs.suppliers
    )

    return textwrap.dedent(f"""
    === SUPPLY CHAIN DISRUPTION SCENARIO ===

    Step: {obs.step}
    Budget remaining: ${obs.budget_remaining:,.2f}
    On-hand inventory ({inv_sku}): {on_hand} units

    DISRUPTION EVENT:
      {obs.disruption.description}
      Affected SKU: {obs.disruption.affected_sku}
      Severity: {obs.disruption.severity}
      Est. recovery: {obs.disruption.estimated_recovery_days} days

    CUSTOMER ORDERS:
    {orders_txt}

    ALTERNATIVE SUPPLIERS:
    {suppliers_txt}

    Produce the action JSON now.
    """).strip()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class BaselineAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.system_prompt = build_system_prompt()

    def act(self, obs: Observation) -> Action:
        user_prompt = build_user_prompt(obs)
        logger.debug("Sending prompt to %s:\n%s", self.model, user_prompt[:500])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        logger.debug("Raw response: %s", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s\nRaw: %s", e, raw)
            # Fallback: do nothing
            data = {"fulfill_orders": [], "delay_orders": [], "cancel_orders": []}

        # Ensure all orders not mentioned are explicitly delayed
        mentioned = set(
            data.get("fulfill_orders", []) +
            data.get("delay_orders",   []) +
            data.get("cancel_orders",  [])
        )
        pending_ids = [o.order_id for o in obs.orders if o.order_id not in mentioned]
        data.setdefault("delay_orders", [])
        data["delay_orders"] += pending_ids

        return Action(**data)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(seed: int = 42, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    setup_logging()
    env = SupplyDisruptionEnv(seed=seed)
    obs = env.reset()

    logger.info("Starting episode | seed=%d | model=%s", seed, model)

    try:
        agent = BaselineAgent(model=model)
    except KeyError:
        logger.error("OPENAI_API_KEY not set.  Export it before running.")
        sys.exit(1)

    total_reward = 0.0
    step = 0

    while not obs.done:
        action = agent.act(obs)
        logger.info(
            "Step %d action: fulfill=%s delay=%s cancel=%s supplier=%s",
            step, action.fulfill_orders, action.delay_orders,
            action.cancel_orders, action.use_supplier,
        )
        if action.reasoning:
            logger.info("Agent reasoning: %s", action.reasoning)

        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        step += 1

        if info.get("warnings"):
            logger.warning("Step warnings: %s", info["warnings"])

        if done:
            break

    # ---------- Evaluate against formal tasks ----------
    print("\n" + "=" * 60)
    print("EPISODE COMPLETE")
    print("=" * 60)
    print(f"Total reward  : {total_reward:.3f}")
    print(f"Steps taken   : {step}")

    final_state = env.state()
    print("\nOrder outcomes:")
    for o in final_state["orders"]:
        print(f"  {o['order_id']} ({o['priority']:8s}) → {o['status']}")

    # Grade the hard task based on the final state actions
    hard_submission = {
        "fulfill_orders": [
            o["order_id"] for o in final_state["orders"] if o["status"] == "fulfilled"
        ],
        "delay_orders": [
            o["order_id"] for o in final_state["orders"] if o["status"] == "delayed"
        ],
        "cancel_orders": [
            o["order_id"] for o in final_state["orders"] if o["status"] == "cancelled"
        ],
        "use_supplier": action.use_supplier,
    }

    score = grade("hard", hard_submission, obs)
    print(f"\nHard task score : {score['score']:.4f}")
    print(f"  HP fulfillment : {score['high_priority_score']:.4f}")
    print(f"  Budget adherence: {score['budget_adherence_score']:.4f}")
    print(f"  Minimal delay  : {score['minimal_delay_score']:.4f}")
    print(f"  Supplier opt.  : {score['supplier_optimality_score']:.4f}")
    print("=" * 60)

    return {"total_reward": total_reward, "steps": step, "score": score}


if __name__ == "__main__":
    run_episode()
