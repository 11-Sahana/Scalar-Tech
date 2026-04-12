
"""
inference.py  —  Supply Disruption Responder
=============================================
Emits the mandatory OpenEnv stdout protocol for all 3 tasks:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.0000> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.0000> rewards=<r1,r2,...>

All reward and score values are strictly in [0.0, 1.0].
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.env import SupplyDisruptionEnv
from app.models import Action
from app.tasks import (
    TASKS,
    get_at_risk_order_ids,
    get_optimal_supplier_ranking,
    get_reference_contingency_plan,
)
from app.grader import grade

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SEED         = int(os.getenv("SUPPLY_SEED", "42"))
BENCHMARK    = "supply-disruption-env"

MAX_STEPS    = 5
MAX_TOKENS   = 512
TEMPERATURE  = 0.2

SUCCESS_SCORE_THRESHOLD = 0.5


def clamp(v: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return round(min(max(float(v), 0.0), 1.0), 4)


# ── Protocol helpers ───────────────────────────────────────────────────────────

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action.replace(chr(10), ' ')} "
        f"reward={clamp(reward):.4f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{clamp(r):.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={clamp(score):.4f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "easy": textwrap.dedent("""
        You are a supply chain analyst. A primary supplier has failed.
        Identify which pending orders are AT RISK given current inventory.
        Respond ONLY with valid JSON (no markdown):
        {"at_risk_order_ids": ["ORD-XXX", ...]}
    """).strip(),

    "medium": textwrap.dedent("""
        You are a procurement expert. A primary supplier has failed.
        Rank available alternative suppliers from BEST to WORST using:
        reliability (50%), inverse cost (30%), inverse lead time (20%).
        Respond ONLY with valid JSON (no markdown):
        {"ranked_supplier_ids": ["SUP-XXX", ...], "rationale": "..."}
    """).strip(),

    "hard": textwrap.dedent("""
        You are a supply chain crisis manager. A primary supplier has failed.
        Produce a full contingency plan. Every pending order must appear in
        exactly ONE of: fulfill_orders, delay_orders, cancel_orders.
        Respond ONLY with valid JSON (no markdown):
        {
          "fulfill_orders": ["ORD-XXX", ...],
          "delay_orders":   ["ORD-YYY", ...],
          "cancel_orders":  ["ORD-ZZZ", ...],
          "use_supplier":   "SUP-XXX",
          "reasoning":      "..."
        }
    """).strip(),
}


def build_user_prompt(obs, task_name: str) -> str:
    sku = obs.disruption.affected_sku
    inv = obs.inventory[sku]
    orders_lines = [
        f"  {o.order_id}: priority={getattr(o.priority, 'value', o.priority)}"
        f"  qty={o.quantity}  deadline={o.deadline_days}d  revenue=${o.revenue:.0f}"
        for o in obs.orders
        if getattr(o.status, 'value', o.status) == "pending"
    ]
    suppliers_lines = [
        f"  {s.supplier_id}: reliability={s.reliability} "
        f"cost=${s.unit_cost:.2f}/unit lead={s.lead_time_days}d available={s.available_qty}"
        for s in obs.suppliers
        if not s.is_primary and s.available_qty > 0
    ]
    return textwrap.dedent(f"""
        DISRUPTION: {obs.disruption.description}
        INVENTORY {sku}: on_hand={inv.on_hand_qty} available={inv.available_qty}
        BUDGET: ${obs.budget_remaining:,.2f}
        PENDING ORDERS:
        {chr(10).join(orders_lines) or '  (none)'}
        ALTERNATIVE SUPPLIERS:
        {chr(10).join(suppliers_lines) or '  (none)'}
        Task: {TASKS[task_name]['name']}
    """).strip()


def call_llm(client: OpenAI, task_name: str, user_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_name]},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def parse_json(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            l for l in cleaned.split("\n") if not l.strip().startswith("```")
        ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def fallback(task_name: str, obs) -> Dict[str, Any]:
    if task_name == "easy":
        return {"at_risk_order_ids": get_at_risk_order_ids(obs)}
    elif task_name == "medium":
        return {
            "ranked_supplier_ids": get_optimal_supplier_ranking(obs),
            "rationale": "Fallback: optimal ranking",
        }
    else:
        plan = get_reference_contingency_plan(obs)
        return {**plan, "reasoning": "Fallback: reference planner"}


def to_action(submission: Dict[str, Any], obs) -> Action:
    pending = {
        o.order_id for o in obs.orders
        if getattr(o.status, 'value', o.status) == "pending"
    }
    fulfill = [x for x in submission.get("fulfill_orders", []) if x in pending]
    delay   = [x for x in submission.get("delay_orders",   []) if x in pending]
    cancel  = [x for x in submission.get("cancel_orders",  []) if x in pending]
    delay.extend(sorted(pending - set(fulfill) - set(delay) - set(cancel)))
    return Action(
        fulfill_orders=fulfill, delay_orders=delay, cancel_orders=cancel,
        use_supplier=submission.get("use_supplier"),
        reasoning=submission.get("reasoning", ""),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in ["easy", "medium", "hard"]:
        env        = SupplyDisruptionEnv(seed=SEED)
        obs        = env.reset()
        rewards:   List[float]    = []
        steps_taken = 0
        score      = 0.0
        success    = False
        submission: Dict[str, Any] = {}

        log_start(task_name)

        try:
            done = obs.done

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                raw        = call_llm(client, task_name, build_user_prompt(obs, task_name))
                submission = parse_json(raw)
                error_msg: Optional[str] = None

                if not submission:
                    error_msg  = "empty_or_invalid_json"
                    submission = fallback(task_name, obs)

                # Grade — score is already in [0, 1]
                result     = grade(task_name, submission, obs)
                step_score = clamp(result.get("score", 0.0))

                if task_name == "hard":
                    obs, reward_obj, done, info = env.step(to_action(submission, obs))
                    # Normalise raw env reward to [0, 1] by dividing by max possible
                    raw_reward  = float(reward_obj.value)
                    # reward_obj.value can be large; clamp it to [0,1]
                    step_reward = clamp(raw_reward / max(abs(raw_reward), 1.0)) if raw_reward > 1.0 else clamp(raw_reward)
                    if info.get("warnings"):
                        error_msg = error_msg or "; ".join(info["warnings"])
                else:
                    # easy / medium: reward = grader score (already 0-1)
                    step_reward = step_score
                    done = True

                rewards.append(step_reward)
                steps_taken = step
                log_step(step, json.dumps(submission, separators=(",", ":")),
                         step_reward, done, error_msg)

                if done:
                    break

            # Final score — re-grade on clean obs
            fresh_obs   = SupplyDisruptionEnv(seed=SEED).reset()
            final       = grade(task_name, submission, fresh_obs)
            score       = clamp(final.get("score", 0.0))
            success     = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Exception in {task_name}: {exc}", flush=True)
            import traceback; traceback.print_exc(file=sys.stdout)

        finally:
            log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()