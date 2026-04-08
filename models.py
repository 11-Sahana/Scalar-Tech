"""
Pydantic models for the Supply Disruption Responder environment.
Defines the structured data types for observations, actions, and rewards.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING   = "pending"
    FULFILLED = "fulfilled"
    DELAYED   = "delayed"
    CANCELLED = "cancelled"


class OrderPriority(str, Enum):
    CRITICAL = "critical"   # SLA breach → severe penalty
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

class CustomerOrder(BaseModel):
    order_id:    str
    customer:    str
    product_sku: str
    quantity:    int
    priority:    OrderPriority
    deadline_days: int          # days until deadline from scenario start
    revenue:     float          # USD value of fulfilling this order
    status:      OrderStatus = OrderStatus.PENDING

    class Config:
        use_enum_values = True


class Supplier(BaseModel):
    supplier_id:  str
    name:         str
    product_sku:  str           # SKU this supplier can provide
    unit_cost:    float         # cost per unit
    lead_time_days: int         # days until delivery
    reliability:  float         # 0.0–1.0 probability of on-time delivery
    available_qty: int          # max units they can supply right now
    is_primary:   bool = False  # True if this was the disrupted primary supplier


class DisruptionEvent(BaseModel):
    event_id:       str
    description:    str
    affected_sku:   str
    disrupted_supplier_id: str
    severity:       str         # "partial" | "full"
    estimated_recovery_days: int


class InventoryItem(BaseModel):
    product_sku: str
    on_hand_qty: int
    reserved_qty: int = 0

    @property
    def available_qty(self) -> int:
        return self.on_hand_qty - self.reserved_qty


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full observation returned to the agent at each step.
    Contains all information needed to make a decision.
    """
    step:             int
    budget_remaining: float
    inventory:        Dict[str, InventoryItem]          # sku → item
    orders:           List[CustomerOrder]
    suppliers:        List[Supplier]
    disruption:       DisruptionEvent
    metrics: Dict[str, Any] = Field(default_factory=dict)
    done:             bool = False

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Structured action produced by the agent.

    fulfill_orders  - order IDs to fulfill immediately from inventory / new supplier
    delay_orders    - order IDs to push back (partial penalty)
    cancel_orders   - order IDs to cancel (larger penalty, avoid budget blowout)
    use_supplier    - supplier_id to source additional stock from
    reasoning       - optional free-text explanation (logged, not evaluated)
    """
    fulfill_orders: List[str] = Field(default_factory=list)
    delay_orders:   List[str] = Field(default_factory=list)
    cancel_orders:  List[str] = Field(default_factory=list)
    use_supplier:   Optional[str] = None
    reasoning:      Optional[str] = None

    def validate_disjoint(self) -> bool:
        """Ensure each order appears in at most one list."""
        all_ids = self.fulfill_orders + self.delay_orders + self.cancel_orders
        return len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed reward signal so the agent (and evaluators) can introspect."""
    fulfillment_score:  float = 0.0   # positive: revenue earned
    delay_penalty:      float = 0.0   # negative
    cancel_penalty:     float = 0.0   # negative
    budget_penalty:     float = 0.0   # negative if over budget
    supplier_bonus:     float = 0.0   # positive for reliable supplier choice
    total:              float = 0.0


class Reward(BaseModel):
    step:      int
    value:     float
    breakdown: RewardBreakdown
    info:      Dict[str, Any] = Field(default_factory=dict)</content>
<parameter name="filePath">c:\supply-disruption-env\supply-disruption-env\models.py