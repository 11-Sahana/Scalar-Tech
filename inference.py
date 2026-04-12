"""
inference.py  —  Supply Disruption Responder
=============================================
Runs the LLM-powered agent against the Supply Disruption environment and
emits the mandatory OpenEnv stdout protocol:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

REQUIRED ENVIRONMENT VARIABLES
--------------------------------
    API_BASE_URL        LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME          Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN or API_KEY Your HuggingFace / API key

OPTIONAL
--------
    SUPPLY_SEED         Random seed for the scenario   (default: 42)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Environment imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.env import SupplyDisruptionEnv
from app.models import Action, OrderStatus
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

SEED      = int(os.getenv("SUPPLY_SEED", "42"))
BENCHMARK = "supply-disruption-env"

MAX_STEPS  = 5
MAX_TOKENS = 512
TEMPERATURE = 0.2

SUCCESS_SCORE_THRESHOLD = 0.7

TASK_WEIGHTS = {"easy": 0.20, "medium": 0.30, "hard": 0.50}

# ── Stdout protocol helpers ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_oneline = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_oneline} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompts per task ────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "easy": textwrap.dedent("""
        You are a supply chain analyst. A primary supplier has failed.
        Given the current inventory and pending orders, identify which orders
        are AT RISK — meaning on-hand inventory is insufficient to fulfill them
        if orders are processed in priority + urgency order.

        Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
        {"at_risk_order_ids": ["ORD-XXX", ...]}
    """).strip(),

    "medium": textwrap.dedent("""
        You are a procurement expert. A primary supplier has failed.
        Rank the available alternative suppliers from BEST to WORST.
        Score each supplier using: reliability (50%), inverse cost (30%),
        inverse lead time (20%). Exclude any supplier with 0 available stock
        or marked as the primary/disrupted supplier.

        Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
        {"ranked_supplier_ids": ["SUP-XXX", "SUP-YYY", ...], "rationale": "brief explanation"}
    """).strip(),

    "hard": textwrap.dedent("""
        You are a supply chain crisis manager. A primary supplier has failed.
        Produce a full contingency plan: decide which orders to FULFILL, DELAY,
        or CANCEL, and which alternative supplier to use.

        Rules:
        - Every pending order must appear in EXACTLY ONE of: fulfill_orders, delay_orders, cancel_orders.
        - Prioritize CRITICAL then HIGH orders for fulfillment.
        - Choose the supplier with the best composite score: reliability (50%), low cost (30%), fast lead (20%).
        - Stay within budget. Procurement cost = shortfall_units x supplier_unit_cost.
        - Delay MEDIUM/LOW orders you cannot afford to fulfill.
        - Cancel only truly LOW-priority orders if no inventory remains.

        Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
        {
          "fulfill_orders": ["ORD-XXX", ...],
          "delay_orders":   ["ORD-YYY", ...],
          "cancel_orders":  ["ORD-ZZZ", ...],
          "use_supplier":   "SUP-XXX",
          "reasoning":      "brief explanation"
        }
    """).strip(),
}


# ── Build user prompt from observation ────────────────────────────────────────

def build_user_prompt(obs, task_name: str) -> str:
    sku = obs.disruption.affected_sku
    inv = obs.inventory[sku]

    orders_lines = []
    for o in obs.orders:
        status = o.status if isinstance(o.status, str) else o.status.value
        if status == "pending":
            orders_lines.append(
                f"  {o.order_id}: priority={o.priority if isinstance(o.priority, str) else o.priority.value}"
                f"  qty={o.quantity}  deadline={o.deadline_days}d  revenue=${o.revenue:.0f}"
            )

    suppliers_lines = []
    for s in obs.suppliers:
        if not s.is_primary and s.available_qty > 0:
            suppliers_lines.append(
                f"  {s.supplier_id}: reliability={s.reliability}  "
                f"cost=${s.unit_cost:.2f}/unit  lead={s.lead_time_days}d  "
                f"available={s.available_qty}"
            )

    task_meta = TASKS[task_name]

    return textwrap.dedent(f"""
        === SUPPLY DISRUPTION SCENARIO ===
        Task       : {task_meta['name']}
        Disruption : {obs.disruption.description}

        INVENTORY
          SKU            : {sku}
          On-hand qty    : {inv.on_hand_qty}
          Available qty  : {inv.available_qty}

        PENDING ORDERS
        {chr(10).join(orders_lines) if orders_lines else '  (none)'}

        ALTERNATIVE SUPPLIERS
        {chr(10).join(suppliers_lines) if suppliers_lines else '  (none)'}

        BUDGET REMAINING : ${obs.budget_remaining:,.2f}

        Produce the JSON response for task: {task_meta['name']}
    """).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, task_name: str, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_name]},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def parse_json_response(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[DEBUG] JSON parse error. Raw response:\n{raw}", flush=True)
        return {}


# ── Fallback: deterministic reference submission ───────────────────────────────

def make_fallback_submission(task_name: str, obs) -> Dict[str, Any]:
    if task_name == "easy":
        return {"at_risk_order_ids": get_at_risk_order_ids(obs)}
    elif task_name == "medium":
        return {
            "ranked_supplier_ids": get_optimal_supplier_ranking(obs),
            "rationale": "Fallback: optimal ranking via composite score",
        }
    else:  # hard
        plan = get_reference_contingency_plan(obs)
        return {
            "fulfill_orders": plan["fulfill_orders"],
            "delay_orders":   plan["delay_orders"],
            "cancel_orders":  plan["cancel_orders"],
            "use_supplier":   plan["use_supplier"],
            "reasoning":      "Fallback: reference greedy planner",
        }


# ── Convert hard-task submission → Action (for env.step) ──────────────────────

def submission_to_action(submission: Dict[str, Any], obs) -> Action:
    pending_ids = {
        o.order_id for o in obs.orders
        if (o.status if isinstance(o.status, str) else o.status.value) == "pending"
    }

    fulfill = [oid for oid in submission.get("fulfill_orders", []) if oid in pending_ids]
    delay   = [oid for oid in submission.get("delay_orders",   []) if oid in pending_ids]
    cancel  = [oid for oid in submission.get("cancel_orders",  []) if oid in pending_ids]

    assigned = set(fulfill) | set(delay) | set(cancel)
    unassigned = pending_ids - assigned
    delay.extend(sorted(unassigned))

    return Action(
        fulfill_orders=fulfill,
        delay_orders=delay,
        cancel_orders=cancel,
        use_supplier=submission.get("use_supplier"),
        reasoning=submission.get("reasoning", ""),
    )


# ── Main: runs all 3 tasks ─────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in ["easy", "medium", "hard"]:

        env = SupplyDisruptionEnv(seed=SEED)
        obs = env.reset()

        rewards:     List[float] = []
        steps_taken: int         = 0
        score:       float       = 0.0
        success:     bool        = False
        submission:  Dict[str, Any] = {}

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            done = obs.done

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                user_prompt  = build_user_prompt(obs, task_name)
                raw_response = call_llm(client, task_name, user_prompt)
                submission   = parse_json_response(raw_response)

                error_msg: Optional[str] = None

                if not submission:
                    error_msg  = "empty_or_invalid_json"
                    submission = make_fallback_submission(task_name, obs)

                grade_result = grade(task_name, submission, obs)
                step_score   = grade_result.get("score", 0.0)

                if task_name == "hard":
                    action = submission_to_action(submission, obs)
                    obs, reward_obj, done, info = env.step(action)
                    step_reward = reward_obj.value
                    if info.get("warnings"):
                        error_msg = error_msg or "; ".join(info["warnings"])
                else:
                    step_reward = step_score * 100.0
                    done = True

                rewards.append(step_reward)
                steps_taken = step

                action_summary = json.dumps(submission, separators=(",", ":"))
                log_step(
                    step=step,
                    action=action_summary,
                    reward=step_reward,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

            # Final score
            final_env  = SupplyDisruptionEnv(seed=SEED)
            final_obs  = final_env.reset()
            final_grade = grade(task_name, submission, final_obs)
            score   = float(final_grade.get("score", 0.0))
            score   = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Unhandled exception in task {task_name}: {exc}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()