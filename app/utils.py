"""
Utility helpers for the Supply Disruption Responder environment.
"""

from __future__ import annotations
import random
import logging
from typing import List, Dict, Any

from app.models import (
    CustomerOrder, Supplier, DisruptionEvent,
    InventoryItem, OrderPriority, OrderStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority weights (used across grader / reward)
# ---------------------------------------------------------------------------

PRIORITY_WEIGHT: Dict[str, float] = {
    OrderPriority.CRITICAL: 4.0,
    OrderPriority.HIGH:     3.0,
    OrderPriority.MEDIUM:   2.0,
    OrderPriority.LOW:      1.0,
}

CANCEL_PENALTY_FACTOR: Dict[str, float] = {
    OrderPriority.CRITICAL: 2.0,
    OrderPriority.HIGH:     1.5,
    OrderPriority.MEDIUM:   1.0,
    OrderPriority.LOW:      0.5,
}

DELAY_PENALTY_FACTOR: Dict[str, float] = {
    OrderPriority.CRITICAL: 1.5,
    OrderPriority.HIGH:     1.0,
    OrderPriority.MEDIUM:   0.5,
    OrderPriority.LOW:      0.2,
}

# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def generate_scenario(seed: int = 42) -> Dict[str, Any]:
    """
    Generate a reproducible (but randomised) supply chain scenario.
    Returns raw dicts that the environment and tasks will consume.
    """
    rng = random.Random(seed)

    sku = "SKU-WIDGET-PRO"

    # Disruption
    disruption = DisruptionEvent(
        event_id="EVT-001",
        description=(
            "Primary supplier Alpha Manufacturing suffered a warehouse fire, "
            "halting all shipments of SKU-WIDGET-PRO for an estimated 14 days."
        ),
        affected_sku=sku,
        disrupted_supplier_id="SUP-ALPHA",
        severity="full",
        estimated_recovery_days=14,
    )

    # Inventory (partial stock on hand)
    inventory = {
        sku: InventoryItem(
            product_sku=sku,
            on_hand_qty=rng.randint(80, 120),
        )
    }

    # Customer orders – mix of priorities
    order_templates = [
        ("ORD-001", "MegaRetail Corp",    OrderPriority.CRITICAL, rng.randint(40, 60),  2,  rng.uniform(18000, 22000)),
        ("ORD-002", "GovHealth Supply",   OrderPriority.CRITICAL, rng.randint(25, 35),  3,  rng.uniform(12000, 16000)),
        ("ORD-003", "TechGadgets Inc",    OrderPriority.HIGH,     rng.randint(30, 50),  5,  rng.uniform(8000,  12000)),
        ("ORD-004", "QuickShop Online",   OrderPriority.HIGH,     rng.randint(20, 40),  4,  rng.uniform(6000,   9000)),
        ("ORD-005", "HomePlus Retail",    OrderPriority.MEDIUM,   rng.randint(50, 70),  7,  rng.uniform(5000,   8000)),
        ("ORD-006", "Startup Gadgets",    OrderPriority.MEDIUM,   rng.randint(15, 25), 10,  rng.uniform(2000,   4000)),
        ("ORD-007", "Budget Electronics", OrderPriority.LOW,      rng.randint(60, 80), 14,  rng.uniform(3000,   5000)),
        ("ORD-008", "Wholesale Depot",    OrderPriority.LOW,      rng.randint(30, 50), 12,  rng.uniform(2000,   3500)),
    ]
    orders = [
        CustomerOrder(
            order_id=oid, customer=cust, product_sku=sku,
            quantity=qty, priority=pri, deadline_days=ddl, revenue=rev,
        )
        for oid, cust, pri, qty, ddl, rev in order_templates
    ]

    # Alternative suppliers
    suppliers = [
        Supplier(
            supplier_id="SUP-ALPHA",
            name="Alpha Manufacturing",
            product_sku=sku,
            unit_cost=rng.uniform(18, 22),
            lead_time_days=14,
            reliability=0.0,   # currently disrupted
            available_qty=0,
            is_primary=True,
        ),
        Supplier(
            supplier_id="SUP-BETA",
            name="Beta Components Ltd",
            product_sku=sku,
            unit_cost=rng.uniform(24, 28),
            lead_time_days=3,
            reliability=0.92,
            available_qty=rng.randint(150, 200),
        ),
        Supplier(
            supplier_id="SUP-GAMMA",
            name="Gamma Global Supplies",
            product_sku=sku,
            unit_cost=rng.uniform(20, 23),
            lead_time_days=7,
            reliability=0.78,
            available_qty=rng.randint(100, 150),
        ),
        Supplier(
            supplier_id="SUP-DELTA",
            name="Delta Express Parts",
            product_sku=sku,
            unit_cost=rng.uniform(30, 36),
            lead_time_days=1,
            reliability=0.97,
            available_qty=rng.randint(80, 120),
        ),
        Supplier(
            supplier_id="SUP-EPSILON",
            name="Epsilon Offshore Mfg",
            product_sku=sku,
            unit_cost=rng.uniform(14, 17),
            lead_time_days=18,
            reliability=0.65,
            available_qty=rng.randint(300, 400),
        ),
    ]

    budget = rng.uniform(14000, 18000)

    return {
        "disruption": disruption,
        "inventory": inventory,
        "orders": orders,
        "suppliers": suppliers,
        "budget": budget,
        "sku": sku,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def order_map(orders: List[CustomerOrder]) -> Dict[str, CustomerOrder]:
    return {o.order_id: o for o in orders}


def supplier_map(suppliers: List[Supplier]) -> Dict[str, Supplier]:
    return {s.supplier_id: s for s in suppliers}


def compute_order_urgency_score(order: CustomerOrder) -> float:
    """Higher score = more urgent. Used for baseline sorting."""
    pw = PRIORITY_WEIGHT.get(order.priority, 1.0)
    return pw * (1.0 / max(order.deadline_days, 1))


def supplier_score(supplier: Supplier) -> float:
    """
    Composite score: high reliability, low cost, short lead time = better.
    Normalised heuristic, not absolute.
    """
    if supplier.available_qty == 0:
        return -1.0
    cost_score = 1.0 / max(supplier.unit_cost, 1)
    time_score = 1.0 / max(supplier.lead_time_days, 1)
    return supplier.reliability * 0.5 + cost_score * 0.3 + time_score * 0.2


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
