from pydantic import BaseModel
from typing import Any


# ─── SSE progress event ───────────────────────────────────────────────────────

class ProgressEvent(BaseModel):
    stage:   str    # "filtering" | "products" | "customers" | "orders" | "items" | "done" | "error"
    pct:     int    # 0–100
    detail:  str    # human-readable message shown in UI
    counts:  dict[str, int] = {}  # running entity counts


# ─── Input ────────────────────────────────────────────────────────────────────

class ConfirmedMapping(BaseModel):
    confirmed:               dict[str, str]   # col → target (table.field or flat ML key)
    attribute_columns:       dict[str, str]   # col → attribute key
    ignored_columns:         list[str]
    header_row_index:        int = 0
    cost_pct:                float | None = None
    missing_field_decisions: dict[str, str]   # ML field → "placeholder"|"skip_feature"


# ─── Entity models ────────────────────────────────────────────────────────────

class CleanedCustomer(BaseModel):
    clerkId:     str | None = None
    fullName:    str | None = None
    email:       str | None = None
    phoneNumber: str | None = None
    segment:     str | None = None
    metadata:    dict[str, Any] = {}


class CleanedProduct(BaseModel):
    name:          str
    externalAccId: str | None = None  # StockCode / SKU
    price:         float | None = None
    cost:          float | None = None
    stock:         int | None = None
    description:   str | None = None


class CleanedOrder(BaseModel):
    externalOrderId:    str | None = None
    customerExternalId: str | None = None  # clerkId used to link customer
    status:             str = "completed"
    total:              float | None = None
    revenue:            float | None = None
    profit:             float | None = None
    profit_margin:      float | None = None
    createdAt:          str | None = None
    address:            str | None = None
    orderVoucher:       str | None = None


class CleanedOrderItem(BaseModel):
    orderExternalId: str | None = None
    productAccId:    str | None = None  # externalAccId used to link product
    productName:     str | None = None  # fallback if no accId
    quantity:        int = 1
    unitPrice:       float | None = None
    itemDiscount:    float = 0.0
    revenue:         float | None = None
    profit:          float | None = None
    attributes:      dict[str, Any] = {}
    metadata:        dict[str, Any] = {}


# ─── Summary + response ───────────────────────────────────────────────────────

class CleanSummary(BaseModel):
    total_rows:          int
    clean_rows:          int
    failed_rows:         int
    cancelled_rows:      int
    customers_found:     int
    products_found:      int
    orders_found:        int
    order_items_found:   int
    ml_coverage_pct:     float
    derived_fields:      list[str]


class FailedRow(BaseModel):
    row_index: int
    reason:    str


class CleanedEntities(BaseModel):
    customers:   list[CleanedCustomer]  = []
    products:    list[CleanedProduct]   = []
    orders:      list[CleanedOrder]     = []
    order_items: list[CleanedOrderItem] = []


class MissingFieldAction(BaseModel):
    field:   str
    reason:  str
    options: list[str]
    affects: list[str]


class ActionRequired(BaseModel):
    fields:  list[MissingFieldAction]
    message: str = (
        "Some analytics fields could not be found or calculated. "
        "Please choose what to do with each one."
    )


class CleanResponse(BaseModel):
    summary:         CleanSummary
    entities:        CleanedEntities
    failed_rows:     list[FailedRow]      = []
    warnings:        list[str]            = []
    action_required: ActionRequired | None = None