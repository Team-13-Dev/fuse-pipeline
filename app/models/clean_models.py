from __future__ import annotations

from typing import Any
from pydantic import BaseModel


# ─── Input ────────────────────────────────────────────────────────────────────

class ConfirmedMapping(BaseModel):
    """
    The user-confirmed (and possibly edited) mapping from /parse output.
    Sent as a JSON string in the multipart form alongside the file.
    """
    # original_col → "table.field" or "__attributes__.key" or "__unmapped__"
    confirmed: dict[str, str]

    # Attribute columns: original_col → attribute key (e.g. "size", "color")
    # Populated from /parse attribute_columns, user may edit
    attribute_columns: dict[str, str] = {}

    # Columns the user explicitly wants to ignore entirely
    ignored_columns: list[str] = []

    # Header row index detected by /parse (re-used here to re-read the file)
    header_row_index: int = 0

    # Optional: user-supplied cost percentage when cost column is missing
    # e.g. 0.60 means cost = 60% of price
    cost_pct: float | None = None

    # What to do with truly missing ML fields: "placeholder" or "skip_feature"
    # Keys are ML field names, values are user decisions
    missing_field_decisions: dict[str, str] = {}


# ─── Per-entity output models ─────────────────────────────────────────────────

class CleanedCustomer(BaseModel):
    # DB fields
    clerkId:     str | None = None      # from customer_id column
    fullName:    str | None = None
    email:       str | None = None
    phoneNumber: str | None = None
    segment:     str | None = None
    # Extra columns that didn't map to schema
    metadata:    dict[str, Any] = {}


class CleanedProduct(BaseModel):
    # DB fields
    name:          str
    price:         float | None = None
    cost:          float | None = None
    stock:         int | None = None
    description:   str | None = None
    externalAccId: str | None = None   # from product_id column
    # ML-derived
    _is_upsert_key: str = "name"       # Next.js uses name to upsert


class CleanedOrder(BaseModel):
    # DB fields
    externalOrderId: str | None = None  # original order_id from file
    status:          str = "pending"
    total:           float | None = None
    orderVoucher:    str | None = None
    address:         str | None = None
    createdAt:       str | None = None  # ISO date string
    # FK references (by external ID, Next.js resolves to UUID)
    customerExternalId: str | None = None
    # ML-derived fields (attached for model use, not stored in order table)
    revenue:        float | None = None
    profit:         float | None = None
    profit_margin:  float | None = None


class CleanedOrderItem(BaseModel):
    # FK references
    orderExternalId:   str | None = None
    productName:       str | None = None   # Next.js resolves to product UUID
    # DB fields
    quantity:          int = 1
    unitPrice:         float | None = None
    itemDiscount:      float = 0.0
    # Clothing attributes → orderItem.attributes jsonb
    attributes:        dict[str, Any] = {}
    # Unknown columns → orderItem.metadata jsonb
    metadata:          dict[str, Any] = {}


# ─── Summary ──────────────────────────────────────────────────────────────────

class CleanSummary(BaseModel):
    total_rows:        int
    clean_rows:        int
    failed_rows:       int
    customers_found:   int
    products_found:    int
    orders_found:      int
    order_items_found: int
    ml_coverage_pct:   float


class FailedRow(BaseModel):
    row_index: int
    raw_data:  dict[str, Any]
    reason:    str


class CleanedEntities(BaseModel):
    customers:   list[CleanedCustomer]   = []
    products:    list[CleanedProduct]    = []
    orders:      list[CleanedOrder]      = []
    order_items: list[CleanedOrderItem]  = []


# ─── Action required (user must decide) ──────────────────────────────────────

class MissingFieldAction(BaseModel):
    field:       str
    reason:      str
    options:     list[str]   # ["placeholder", "skip_feature"]
    affects:     list[str]   # which entities are affected


class ActionRequired(BaseModel):
    """
    Returned when a truly missing ML field needs a user decision
    before cleaning can complete.
    Status 202 is returned with this payload.
    """
    fields: list[MissingFieldAction]
    message: str = (
        "Some ML-required fields could not be found or derived. "
        "Please provide decisions for each field and resubmit."
    )


# ─── Full response ────────────────────────────────────────────────────────────

class CleanResponse(BaseModel):
    summary:         CleanSummary
    entities:        CleanedEntities
    failed_rows:     list[FailedRow]      = []
    derived_fields:  list[str]            = []
    warnings:        list[str]            = []
    action_required: ActionRequired | None = None