"""
server/app.py
-------------
FastAPI application entry point for the Supply Disruption Responder.
Exposes the standard OpenEnv HTTP interface PLUS /tasks and /grader
endpoints that the hackathon validator requires.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.env import SupplyDisruptionEnv
from app.models import Action, Observation
from app.grader import grade_easy, grade_medium, grade_hard
from app.tasks import TASKS

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Supply Disruption Responder", version="1.0.0")

# Shared environment instance
_env = SupplyDisruptionEnv(seed=42)
_obs: Optional[Observation] = None
_done: bool = False


# ── Standard OpenEnv endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(seed: int = 42):
    global _env, _obs, _done
    _env  = SupplyDisruptionEnv(seed=seed)
    _obs  = _env.reset()
    _done = False
    return _obs.dict()


@app.post("/step")
def step(action: Action):
    global _obs, _done
    if _obs is None:
        _obs = _env.reset()
    if _done:
        return {
            "observation": _obs.dict(),
            "reward": 0.0,
            "done": True,
            "info": {},
        }
    _obs, reward_obj, _done, info = _env.step(action)
    return {
        "observation": _obs.dict(),
        "reward": reward_obj.value,
        "done": _done,
        "info": info,
    }


@app.get("/state")
def state():
    return _env.state()


# ── /tasks — required by the hackathon validator ──────────────────────────────

@app.get("/tasks")
def list_tasks():
    """Return all 3 tasks with their grader info — required by OpenEnv validator."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "At-Risk Order Identification",
                "description": TASKS["easy"]["description"],
                "difficulty": "easy",
                "max_attempts": 5,
                "scoring": "F1 score (0.0–1.0)",
                "grader": "app.grader.grade_easy",
                "action_schema": {"at_risk_order_ids": "List[str]"},
            },
            {
                "id": "medium",
                "name": "Supplier Ranking",
                "description": TASKS["medium"]["description"],
                "difficulty": "medium",
                "max_attempts": 5,
                "scoring": "Spearman rank correlation (0.0–1.0)",
                "grader": "app.grader.grade_medium",
                "action_schema": {
                    "ranked_supplier_ids": "List[str]",
                    "rationale": "str (optional)",
                },
            },
            {
                "id": "hard",
                "name": "Full Contingency Plan",
                "description": TASKS["hard"]["description"],
                "difficulty": "hard",
                "max_attempts": 5,
                "scoring": "Weighted composite (0.0–1.0)",
                "grader": "app.grader.grade_hard",
                "action_schema": {
                    "fulfill_orders": "List[str]",
                    "delay_orders": "List[str]",
                    "cancel_orders": "List[str]",
                    "use_supplier": "str",
                    "reasoning": "str (optional)",
                },
            },
        ]
    }


# ── /grader — required by the hackathon validator ─────────────────────────────

class GraderRequest(BaseModel):
    task_id: str
    submission: Dict[str, Any]
    seed: int = 42


@app.post("/grader")
def run_grader(req: GraderRequest):
    """Grade a submission for any task without running a full episode."""
    env = SupplyDisruptionEnv(seed=req.seed)
    obs = env.reset()

    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }

    if req.task_id not in graders:
        return {
            "error": f"Unknown task_id '{req.task_id}'. Valid: {list(graders.keys())}"
        }

    result = graders[req.task_id](req.submission, obs)
    result["task_id"] = req.task_id
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()