# Supply Disruption Responder 🚚⚡

An **OpenEnv-compatible** environment that places an AI agent at the centre of
a real-world supply chain crisis.  A primary manufacturer has unexpectedly
failed, threatening millions in customer revenue.  The agent must respond in
real time — deciding which orders to fulfil, delay, or cancel, and which
alternative supplier to activate — all under hard inventory, deadline, and
budget constraints.

---

## 📌 Problem Motivation

Supply chain disruptions cost the global economy hundreds of billions of
dollars each year.  Fires, floods, geopolitical events, and pandemic shocks
routinely knock out primary suppliers with little warning.  Procurement teams
must make time-critical, multi-objective decisions:

- Which **customers** absorb the impact (priority)?
- Which **supplier** can step in fastest at acceptable cost and reliability?
- How much of the **budget** should be spent on emergency procurement?

This environment makes those trade-offs explicit and measurable.

---

## 🗂️ Project Structure

```
supply-disruption-env/
├── app/
│   ├── env.py          # Core OpenEnv environment (step / reset / state)
│   ├── models.py       # Pydantic models: Observation, Action, Reward
│   ├── tasks.py        # Task definitions + ground-truth helpers
│   ├── grader.py       # Per-task graders returning 0.0–1.0
│   ├── reward.py       # Dense reward function
│   └── utils.py        # Scenario generator + helpers
├── agent/
│   └── baseline_agent.py   # GPT-powered agent
├── configs/
│   └── openenv.yaml        # OpenEnv metadata
├── scripts/
│   ├── run_env.py          # Manual / rule-based run
│   └── evaluate.py         # Formal task evaluation
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Environment Design

### Scenario

Each episode is seeded and generates:

| Component         | Details |
|-------------------|---------|
| **Disruption**    | Warehouse fire at primary supplier; full stoppage of `SKU-WIDGET-PRO` |
| **Inventory**     | 80–120 units on hand |
| **Orders**        | 8 customer orders spanning CRITICAL → LOW priority, varying quantities and deadlines |
| **Suppliers**     | 4 alternative suppliers with differing cost / lead time / reliability trade-offs |
| **Budget**        | $14 000–$18 000 for emergency procurement |

### Key Trade-offs

- **Beta** (SUP-BETA): fast (3 days), reliable (92 %), but expensive ($24–28/unit)
- **Gamma** (SUP-GAMMA): medium cost ($20–23), medium reliability (78 %), 7-day lead
- **Delta** (SUP-DELTA): fastest (1 day), most reliable (97 %), most expensive ($30–36)
- **Epsilon** (SUP-EPSILON): cheapest ($14–17), but slow (18 days) and unreliable (65 %)

---

## 🔭 Observation Space

```json
{
  "step": 0,
  "budget_remaining": 15800.0,
  "inventory": { "SKU-WIDGET-PRO": { "on_hand_qty": 95, "reserved_qty": 0 } },
  "orders": [ { "order_id": "ORD-001", "priority": "critical", "quantity": 48, ... } ],
  "suppliers": [ { "supplier_id": "SUP-BETA", "reliability": 0.92, ... } ],
  "disruption": { "affected_sku": "SKU-WIDGET-PRO", "severity": "full", ... },
  "metrics": { "pending_orders": 8, "fulfilled_orders": 0, ... },
  "done": false
}
```

---

## 🎮 Action Space

```json
{
  "fulfill_orders": ["ORD-001", "ORD-002"],
  "delay_orders":   ["ORD-005", "ORD-006"],
  "cancel_orders":  ["ORD-008"],
  "use_supplier":   "SUP-BETA",
  "reasoning":      "Optional free-text for logging"
}
```

Constraint: each order ID must appear in **exactly one** list.

---

## 🧪 Tasks

### EASY – At-Risk Order Identification
Identify which pending orders cannot be fulfilled with current on-hand inventory.

- **Input**: observation (inventory + orders)
- **Output**: `{ "at_risk_order_ids": ["ORD-003", ...] }`
- **Metric**: F1 score vs. ground truth

### MEDIUM – Supplier Ranking
Rank all non-disrupted, available suppliers best → worst.

- **Input**: observation (suppliers)
- **Output**: `{ "ranked_supplier_ids": ["SUP-DELTA", "SUP-BETA", ...] }`
- **Metric**: Spearman rank correlation vs. optimal order

### HARD – Full Contingency Plan
Produce a complete action plan that maximises revenue and minimises damage.

- **Input**: full observation
- **Output**: `{ "fulfill_orders": [...], "delay_orders": [...], "cancel_orders": [...], "use_supplier": "..." }`
- **Metric**: Weighted composite (see table below)

| Sub-metric                  | Weight |
|-----------------------------|--------|
| High-priority fulfillment   | 40 %   |
| Budget adherence            | 20 %   |
| Minimal unnecessary delays  | 20 %   |
| Supplier optimality         | 20 %   |

---

## 🏆 Reward Function

The reward is **dense and incremental** — the agent receives signal after every
step, not just at episode end.

| Component          | Direction | Formula |
|--------------------|-----------|---------|
| Fulfillment        | ＋        | `revenue × priority_weight × 0.01` |
| Delay penalty      | −         | `revenue × delay_factor × 0.005` |
| Cancel penalty     | −         | `revenue × cancel_factor × 0.008` |
| Budget overage     | −         | `overage × 0.10` |
| Supplier bonus     | ＋        | `reliability × 50` |

Priority weights: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1.

---

## 📊 Baseline Scores

Evaluated with seed=42 using the reference (near-optimal) rule-based agent:

| Task    | Weight | Reference Score |
|---------|--------|-----------------|
| Easy    | 20 %   | 1.0000          |
| Medium  | 30 %   | 1.0000          |
| Hard    | 50 %   | ~0.95+          |
| **Overall** | — | **~0.975**      |

An imperfect agent (partial recall / reversed ranking / poor supplier) typically
scores 0.40–0.60 on the hard task.

---

## 🛠️ Setup & Usage

### Prerequisites

```bash
python >= 3.11
pip install -r requirements.txt
```

### Run the environment manually (no API key needed)

```bash
python scripts/run_env.py
# or with a different seed:
python scripts/run_env.py --seed 123
```

### Evaluate all tasks

```bash
python scripts/evaluate.py
python scripts/evaluate.py --verbose
```

### Run the GPT-powered baseline agent

```bash
export OPENAI_API_KEY=sk-...
python agent/baseline_agent.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t supply-env .

# Run evaluation (no API key required)
docker run supply-env

# Run GPT baseline agent
docker run -e OPENAI_API_KEY=sk-... supply-env python agent/baseline_agent.py

# Different seed
docker run supply-env python scripts/evaluate.py --seed 99
```

---

## 🔌 OpenEnv API

```python
from app.env import SupplyDisruptionEnv
from app.models import Action

env = SupplyDisruptionEnv(seed=42)
obs = env.reset()

action = Action(
    fulfill_orders=["ORD-001", "ORD-002"],
    delay_orders=["ORD-005"],
    cancel_orders=[],
    use_supplier="SUP-BETA",
)
obs, reward, done, info = env.step(action)

print(env.state())   # full internal state snapshot
```

---

## 📄 License

MIT
