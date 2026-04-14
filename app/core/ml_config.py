"""
Single source of truth for ML-required fields.
To add a field: add one entry to ML_FIELDS.
To remove a field: delete its entry.
The entire pipeline adapts automatically.
"""

from __future__ import annotations

# ─── ML field registry ────────────────────────────────────────────────────────
# Each entry defines:
#   table   : which DB table stores this field (None = derived/computed only)
#   field   : the DB column name (None = not stored directly)
#   derivable_from : list of (formula_label, [required_field_keys]) in priority order

ML_FIELDS: dict[str, dict] = {
    "quantity": {
        "table": "order_item",
        "field": "quantity",
        "derivable_from": [],
    },
    "price": {
        "table": "product",
        "field": "price",
        "derivable_from": [],
    },
    "cost": {
        "table": "product",
        "field": "cost",
        "derivable_from": [
            ("revenue − profit", ["revenue", "profit"]),
        ],
    },
    "stock": {
        "table": "product",
        "field": "stock",
        "derivable_from": [],
    },
    "revenue": {
        "table": None,
        "field": None,
        "derivable_from": [
            ("price × quantity", ["price", "quantity"]),
        ],
    },
    "profit": {
        "table": None,
        "field": None,
        "derivable_from": [
            ("revenue − cost", ["revenue", "cost"]),
        ],
    },
    "profit_margin": {
        "table": None,
        "field": None,
        "derivable_from": [
            ("profit ÷ revenue × 100", ["profit", "revenue"]),
        ],
    },
    "order_date": {
        "table": "order",
        "field": "createdAt",
        "derivable_from": [],
    },
    "product_id": {
        "table": "product",
        "field": "externalAccId",
        "derivable_from": [],
    },
    "customer_id": {
        "table": "customer",
        "field": "clerkId",
        "derivable_from": [],
    },
}

# Flat list for quick membership checks
ML_REQUIRED_KEYS: list[str] = list(ML_FIELDS.keys())


def get_derivation_rules() -> list[tuple[str, str, list[str]]]:
    """
    Returns ordered derivation rules as (field_key, formula_label, [requires]).
    Ordered so that derived fields can unlock further derivations.
    """
    rules = []
    # Revenue first — unlocks profit and profit_margin
    priority_order = [
        "revenue", "profit", "profit_margin", "cost",
    ]
    seen = set()
    for key in priority_order:
        if key in ML_FIELDS and ML_FIELDS[key]["derivable_from"]:
            for formula, requires in ML_FIELDS[key]["derivable_from"]:
                rules.append((key, formula, requires))
                seen.add(key)
    # Remaining fields
    for key, cfg in ML_FIELDS.items():
        if key not in seen and cfg["derivable_from"]:
            for formula, requires in cfg["derivable_from"]:
                rules.append((key, formula, requires))
    return rules