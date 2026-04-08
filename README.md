# Supply Disruption Responder

An **OpenEnv-compatible** environment that places an AI agent at the centre of a real-world supply chain crisis. A primary manufacturer has unexpectedly failed, threatening millions in customer revenue. The agent must respond in real time — deciding which orders to fulfil, delay, or cancel, and which alternative supplier to activate — all under hard inventory, deadline, and budget constraints.

---

## 📌 Problem Motivation

Supply chain disruptions cost the global economy hundreds of billions of dollars each year. Fires, floods, geopolitical events, and pandemic shocks routinely knock out primary suppliers with little warning. Procurement teams must make time-critical, multi-objective decisions:

- Which **customers** absorb the impact (priority)?
- Which **supplier** can step in fastest at acceptable cost and reliability?
- How much of the **budget** should be spent on emergency procurement?

This environment makes those trade-offs explicit and measurable.

---

## 🗂️ Project Structure

```
Scalar-Tech-main/
├── app/
│   ├── env.py              # Core OpenEnv environment (step / reset / state)
│   ├── models.py           # Pydantic models: Observation, Action, Reward
│   ├── tasks.py            # Task definitions + ground-truth helpers
│   ├── grader.py           # Per-task graders returning 0.0–1.0
│   ├── reward.py           # Dense reward function
│   ├── data_loader.py      # Data loading utilities
│   └── utils.py            # Scenario generator + helpers
├── agent/
│   └── baseline_agent.py   # GPT-powered baseline agent
├── server/
│   ├── app.py              # FastAPI server exposing OpenEnv HTTP interface
│   └── Dockerfile          # Server-specific Docker config
├── scripts/
│   ├── run_env.py          # Manual / rule-based run
│   ├── evaluate.py         # Formal task evaluation (all 3 tasks)
│   └── run_real_data.py    # Run with real-world data
├── inference.py            # ⭐ LLM inference script (OpenEnv submission entry point)
├── client.py               # OpenEnv client wrapper
├── models.py               # Top-level model re-exports
├── openenv.yaml            # OpenEnv metadata
├── requirements.txt        # Python dependencies
├── Dockerfile              # Main Docker image
└── README.md
```

---

## 🧠 Environment Design

### Scenario

Each episode is seeded and generates:

| Component      | Details                                                                    |
|----------------|----------------------------------------------------------------------------|
| **Disruption** | Warehouse fire at primary supplier; full stoppage of `SKU-WIDGET-PRO`     |
| **Inventory**  | 80–120 units on hand                                                       |
| **Orders**     | 8 customer orders spanning CRITICAL → LOW priority, varying quantities and deadlines |
| **Suppliers**  | 4 alternative suppliers with differing cost / lead time / reliability      |
| **Budget**     | $14,000–$18,000 for emergency procurement                                  |

### Supplier Trade-offs

| Supplier            | Cost/unit   | Lead Time | Reliability | Notes              |
|---------------------|-------------|-----------|-------------|--------------------|
| SUP-BETA            | $24–28      | 3 days    | 92%         | Fast, reliable     |
| SUP-GAMMA           | $20–23      | 7 days    | 78%         | Balanced           |
| SUP-DELTA           | $30–36      | 1 day     | 97%         | Best, most expensive |
| SUP-EPSILON         | $14–17      | 18 days   | 65%         | Cheapest, slowest  |

---

## 🔭 Observation Space

```json
{
  "step": 0,
  "budget_remaining": 15800.0,
  "inventory": { "SKU-WIDGET-PRO": { "on_hand_qty": 95, "reserved_qty": 0 } },
  "orders": [ { "order_id": "ORD-001", "priority": "critical", "quantity": 48 } ],
  "suppliers": [ { "supplier_id": "SUP-BETA", "reliability": 0.92 } ],
  "disruption": { "affected_sku": "SKU-WIDGET-PRO", "severity": "full" },
  "metrics": { "pending_orders": 8, "fulfilled_orders": 0 },
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

> **Constraint:** each order ID must appear in **exactly one** list.

---

## 🧪 Tasks

### EASY – At-Risk Order Identification
Identify which pending orders cannot be fulfilled with current on-hand inventory.
- **Output:** `{ "at_risk_order_ids": ["ORD-003", ...] }`
- **Metric:** F1 score vs. ground truth

### MEDIUM – Supplier Ranking
Rank all available (non-disrupted) suppliers best → worst.
- **Output:** `{ "ranked_supplier_ids": ["SUP-DELTA", "SUP-BETA", ...] }`
- **Metric:** Spearman rank correlation vs. optimal order

### HARD – Full Contingency Plan
Produce a complete action plan that maximises revenue and minimises damage.
- **Output:** `{ "fulfill_orders": [...], "delay_orders": [...], "cancel_orders": [...], "use_supplier": "..." }`
- **Metric:** Weighted composite

| Sub-metric                  | Weight |
|-----------------------------|--------|
| High-priority fulfillment   | 40%    |
| Budget adherence            | 20%    |
| Minimal unnecessary delays  | 20%    |
| Supplier optimality         | 20%    |

---

## 🏆 Reward Function

Dense and incremental — the agent receives signal after every step:

| Component      | Direction | Formula                                      |
|----------------|-----------|----------------------------------------------|
| Fulfillment    | ＋        | `revenue × priority_weight × 0.01`          |
| Delay penalty  | −         | `revenue × delay_factor × 0.005`            |
| Cancel penalty | −         | `revenue × cancel_factor × 0.008`           |
| Budget overage | −         | `overage × 0.10`                            |
| Supplier bonus | ＋        | `reliability × 50`                          |

Priority weights: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1.

---

## 📊 Baseline Scores (seed=42)

| Task        | Weight | Score  | Contribution |
|-------------|--------|--------|--------------|
| Easy        | 20%    | 1.0000 | 0.2000       |
| Medium      | 30%    | 1.0000 | 0.3000       |
| Hard        | 50%    | 1.0000 | 0.5000       |
| **Overall** |        |        | **1.0000**   |

**Scenario (seed=42):** Budget $16,792 · 120 units on hand · 8 orders · 4 active suppliers

**Episode summary (reference agent, 1 step):**
```
Fulfil  : ORD-001 (critical), ORD-002 (critical), ORD-003 (high), ORD-004 (high), ORD-006 (medium)
Delay   : ORD-005 (medium), ORD-007 (low), ORD-008 (low)
Supplier: SUP-DELTA (97% reliability, 1-day lead time)

Step reward     : +1837.39
  Fulfillment   : +1808.13
  Delay penalty :   -19.24
  Supplier bonus:   +48.50
Budget remaining: $15,433.15
```

---

## 🛠️ Setup & Usage

### Prerequisites

```
Python >= 3.11
pip install -r requirements.txt
```

### 1. Run the environment manually (no API key needed)

```bash
python scripts/run_env.py
# With a different seed:
python scripts/run_env.py --seed 123
```

### 2. Evaluate all tasks (rule-based reference agent)

```bash
python scripts/evaluate.py
python scripts/evaluate.py --verbose
python scripts/evaluate.py --seed 99
```

### 3. Run the LLM inference agent (requires API key)

Set your environment variables first:

```bash
# Required
export HF_TOKEN=hf_...              # or API_KEY=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Optional — controls which task the agent attempts
export SUPPLY_TASK=hard             # easy | medium | hard  (default: hard)
export SUPPLY_SEED=42               # random seed           (default: 42)
```

Then run:

```bash
python inference.py
```

**Example output:**
```
[START] task=hard env=supply-disruption-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"fulfill_orders":["ORD-001","ORD-002"],...} reward=1837.39 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1837.39
```

### 4. Run the GPT-powered baseline agent

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

# Run LLM inference agent
docker run \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e SUPPLY_TASK=hard \
  supply-env python inference.py

# Run GPT baseline agent
docker run -e OPENAI_API_KEY=sk-... supply-env python agent/baseline_agent.py

# Different seed
docker run supply-env python scripts/evaluate.py --seed 99
```

---

## 🔌 OpenEnv API (programmatic)

```python
from app.env import SupplyDisruptionEnv
from app.models import Action

env = SupplyDisruptionEnv(seed=42)
obs = env.reset()

action = Action(
    fulfill_orders=["ORD-001", "ORD-002"],
    delay_orders=["ORD-005"],
    cancel_orders=[],
    use_supplier="SUP-DELTA",
)
obs, reward, done, info = env.step(action)

print(env.state())   # full internal state snapshot
```

---

## 📁 Key File Reference

| File | Purpose |
|------|---------|
| `inference.py` | **Main submission entry point.** LLM agent; emits `[START]` / `[STEP]` / `[END]` protocol |
| `app/env.py` | Core environment — `reset()`, `step()`, `state()` |
| `app/grader.py` | Per-task scoring logic (F1, Spearman, weighted composite) |
| `app/tasks.py` | Task definitions + ground-truth helpers |
| `app/utils.py` | Scenario generator, priority weights, supplier scoring |
| `scripts/evaluate.py` | Run full evaluation with reference submissions |
| `openenv.yaml` | OpenEnv metadata (name, version, tags) |

---

## 📄 License

MIT
