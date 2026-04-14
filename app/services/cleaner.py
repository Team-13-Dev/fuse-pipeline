"""
services/cleaner.py

Phase 2 — Clean, normalize, derive, and assemble DB-ready entities.

Pipeline stages:
  1. Load & unmerge
  2. Apply confirmed mapping
  3. Clean per column type
  4. Derive ML fields
  5. Entity assembly + deduplication
  6. ML required field check
  7. Validation & failed rows
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any

import pandas as pd

from app.core.ml_config import ML_FIELDS, ML_REQUIRED_KEYS, get_derivation_rules
from app.models.clean_models import (
    ActionRequired,
    CleanedCustomer,
    CleanedEntities,
    CleanedOrder,
    CleanedOrderItem,
    CleanedProduct,
    CleanResponse,
    CleanSummary,
    ConfirmedMapping,
    FailedRow,
    MissingFieldAction,
)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Valid order statuses normalized to our DB enum
ORDER_STATUS_MAP = {
    "shipped":    "shipped",
    "delivered":  "shipped",
    "dispatched": "shipped",
    "completed":  "completed",
    "done":       "completed",
    "finished":   "completed",
    "pending":    "pending",
    "processing": "pending",
    "new":        "pending",
    "open":       "pending",
    "cancelled":  "cancelled",
    "canceled":   "cancelled",
    "refunded":   "cancelled",
    "returned":   "cancelled",
    "failed":     "cancelled",
    "rejected":   "cancelled",
}

# Arabic-Indic numeral map
ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Currency symbols to strip from numeric fields
CURRENCY_RE = re.compile(r"[£$€¥₹₽¢﷼EGP\s,]+", re.IGNORECASE)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Load & Unmerge
# ═════════════════════════════════════════════════════════════════════════════

def _load_file(file_bytes: bytes, filename: str, header_row_index: int) -> pd.DataFrame:
    if filename.endswith(".csv"):
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    header=header_row_index,
                    encoding=enc,
                )
                return df
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError("Could not decode CSV file.")

    elif filename.endswith((".xlsx", ".xls")):
        # openpyxl handles merged cells — forward-fill to unmerge
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            header=header_row_index,
        )
        # Forward-fill merged cells (they appear as NaN after the first cell)
        df = df.ffill(axis=0)
        return df

    raise ValueError(f"Unsupported file format: '{filename}'.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Apply Confirmed Mapping
# ═════════════════════════════════════════════════════════════════════════════

def _apply_mapping(
    df: pd.DataFrame,
    mapping: ConfirmedMapping,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], list[str]]:
    """
    Returns:
      df_mapped       : DataFrame with columns renamed to their target field
      col_to_target   : original_col → mapped target
      attr_cols       : original_col → attribute key
      metadata_cols   : list of original column names going to metadata
    """
    confirmed      = mapping.confirmed
    attr_cols      = mapping.attribute_columns
    ignored        = set(mapping.ignored_columns)
    col_to_target  = {}
    metadata_cols  = []
    rename_map     = {}

    for col in df.columns:
        col_str = str(col)
        if col_str in ignored:
            continue
        if col_str in confirmed:
            target = confirmed[col_str]
            if target == "__unmapped__":
                metadata_cols.append(col_str)
            elif target.startswith("__attributes__"):
                # Handled separately via attr_cols
                pass
            else:
                rename_map[col_str] = target
                col_to_target[col_str] = target
        else:
            # Column not in confirmed mapping → goes to metadata
            metadata_cols.append(col_str)

    df = df.rename(columns=rename_map)
    return df, col_to_target, attr_cols, metadata_cols


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Cell-level Cleaning
# ═════════════════════════════════════════════════════════════════════════════

def _arabic_indic_to_latin(value: Any) -> Any:
    if isinstance(value, str):
        return value.translate(ARABIC_INDIC_MAP)
    return value


def _clean_numeric(value: Any) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = s.translate(ARABIC_INDIC_MAP)          # Arabic-Indic digits
    s = CURRENCY_RE.sub("", s)                 # strip currency symbols
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _clean_integer(value: Any) -> int | None:
    num = _clean_numeric(value)
    if num is None:
        return None
    return int(round(num))


DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
    "%d.%m.%Y", "%Y.%m.%d",
    "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
]


def _clean_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().translate(ARABIC_INDIC_MAP)
    # Already a datetime from pandas
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return s  # Return as-is if unparseable


def _clean_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def _normalize_status(value: Any) -> str:
    if pd.isna(value):
        return "pending"
    s = str(value).strip().lower()
    return ORDER_STATUS_MAP.get(s, "pending")


def _clean_row(row: pd.Series, df_columns: list[str]) -> dict[str, Any]:
    """Clean every cell in a row based on its column target."""
    cleaned: dict[str, Any] = {}
    for col in df_columns:
        val = row.get(col)
        # Determine type by target name
        if any(col.endswith(f) for f in [".price", ".cost", "revenue", "profit", ".unitPrice", ".itemDiscount"]):
            cleaned[col] = _clean_numeric(val)
        elif any(col.endswith(f) for f in [".stock", ".quantity", "quantity"]):
            cleaned[col] = _clean_integer(val)
        elif "date" in col.lower() or "order_date" in col:
            cleaned[col] = _clean_date(val)
        elif col.endswith(".status"):
            cleaned[col] = _normalize_status(val)
        else:
            cleaned[col] = _clean_string(val)
    return cleaned


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Derive ML Fields
# ═════════════════════════════════════════════════════════════════════════════

def _derive_ml_fields(
    row_data: dict[str, Any],
    cost_pct: float | None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Derives missing ML fields from available data.
    Returns updated row_data and list of field names that were derived.
    """
    derived: list[str] = []

    def _get(key: str) -> float | None:
        """Try to find a value by ML key or table.field."""
        if key in row_data and row_data[key] is not None:
            return float(row_data[key])
        # Try table.field variants
        cfg = ML_FIELDS.get(key, {})
        if cfg.get("table") and cfg.get("field"):
            full = f"{cfg['table']}.{cfg['field']}"
            if full in row_data and row_data[full] is not None:
                return float(row_data[full])
        return None

    def _set(key: str, value: float) -> None:
        row_data[key] = round(value, 4)
        derived.append(key)

    price    = _get("price")
    quantity = _get("quantity") or _get("order_item.quantity")
    cost     = _get("cost")
    revenue  = _get("revenue")
    profit   = _get("profit")

    # Apply user-supplied cost percentage
    if cost is None and cost_pct is not None and price is not None:
        cost = round(price * cost_pct, 4)
        _set("cost", cost)

    # revenue = price × quantity
    if revenue is None and price is not None and quantity is not None:
        revenue = round(price * quantity, 4)
        _set("revenue", revenue)

    # profit = revenue − cost
    if profit is None and revenue is not None and cost is not None:
        profit = round(revenue - cost, 4)
        _set("profit", profit)

    # profit_margin = profit ÷ revenue × 100
    if _get("profit_margin") is None and profit is not None and revenue is not None and revenue != 0:
        _set("profit_margin", round(profit / revenue * 100, 2))

    # cost = revenue − profit (fallback)
    if cost is None and revenue is not None and profit is not None:
        cost = round(revenue - profit, 4)
        _set("cost", cost)

    return row_data, derived


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Entity Assembly
# ═════════════════════════════════════════════════════════════════════════════

def _extract_value(row: dict[str, Any], *keys: str) -> Any:
    """Try multiple key variants, return first non-None."""
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return None


def _assemble_customer(row: dict[str, Any], metadata: dict[str, Any]) -> CleanedCustomer:
    return CleanedCustomer(
        clerkId     = _extract_value(row, "customer_id", "customer.clerkId"),
        fullName    = _extract_value(row, "customer.fullName"),
        email       = _extract_value(row, "customer.email"),
        phoneNumber = _extract_value(row, "customer.phoneNumber"),
        segment     = _extract_value(row, "customer.segment"),
        metadata    = metadata,
    )


def _assemble_product(row: dict[str, Any]) -> CleanedProduct:
    return CleanedProduct(
        name          = _extract_value(row, "product.name") or "Unknown Product",
        price         = _extract_value(row, "product.price", "price"),
        cost          = _extract_value(row, "product.cost", "cost"),
        stock         = _extract_value(row, "product.stock", "stock"),
        description   = _extract_value(row, "product.description"),
        externalAccId = _extract_value(row, "product_id", "product.externalAccId"),
    )


def _assemble_order(row: dict[str, Any]) -> CleanedOrder:
    return CleanedOrder(
        externalOrderId    = _extract_value(row, "order_id"),
        status             = _extract_value(row, "order.status") or "pending",
        total              = _extract_value(row, "revenue"),
        orderVoucher       = _extract_value(row, "order.orderVoucher"),
        address            = _extract_value(row, "order.address"),
        createdAt          = _extract_value(row, "order_date"),
        customerExternalId = _extract_value(row, "customer_id"),
        revenue            = _extract_value(row, "revenue"),
        profit             = _extract_value(row, "profit"),
        profit_margin      = _extract_value(row, "profit_margin"),
    )


def _assemble_order_item(
    row: dict[str, Any],
    attributes: dict[str, Any],
    metadata: dict[str, Any],
) -> CleanedOrderItem:
    return CleanedOrderItem(
        orderExternalId = _extract_value(row, "order_id"),
        productName     = _extract_value(row, "product.name"),
        quantity        = int(_extract_value(row, "order_item.quantity", "quantity") or 1),
        unitPrice       = _extract_value(row, "product.price", "price", "order_item.unitPrice"),
        itemDiscount    = float(_extract_value(row, "order_item.itemDiscount") or 0.0),
        attributes      = attributes,
        metadata        = metadata,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 — ML Required Field Check
# ═════════════════════════════════════════════════════════════════════════════

def _check_ml_fields(
    row_data: dict[str, Any],
    decisions: dict[str, str],
) -> list[MissingFieldAction]:
    """
    Returns list of fields that are still missing and need user decisions.
    Applies already-made decisions (placeholder / skip_feature).
    """
    action_needed: list[MissingFieldAction] = []

    for key in ML_REQUIRED_KEYS:
        val = _extract_value(row_data, key,
                             f"{ML_FIELDS[key].get('table', '')}.{ML_FIELDS[key].get('field', '')}")
        if val is not None:
            continue

        decision = decisions.get(key)
        if decision == "placeholder":
            row_data[key] = 0.0  # numeric placeholder
        elif decision == "skip_feature":
            continue  # skip — field will be null
        else:
            # No decision yet — needs user input
            action_needed.append(MissingFieldAction(
                field    = key,
                reason   = f"'{key}' could not be found or derived from the provided data.",
                options  = ["placeholder", "skip_feature"],
                affects  = [ML_FIELDS[key]["table"]] if ML_FIELDS[key].get("table") else ["derived"],
            ))

    return action_needed


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def clean_file(
    file_bytes: bytes,
    filename: str,
    mapping: ConfirmedMapping,
) -> CleanResponse:
    warnings: list[str] = []
    failed_rows: list[FailedRow] = []
    all_derived: set[str] = set()

    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    df = _load_file(file_bytes, filename, mapping.header_row_index)
    total_rows = len(df)

    # ── Stage 2: Apply mapping ────────────────────────────────────────────────
    df, col_to_target, attr_cols, metadata_col_names = _apply_mapping(df, mapping)

    # Columns that are now renamed to their target
    mapped_cols = list(col_to_target.values())

    # ── Accumulation structures ───────────────────────────────────────────────
    customers_map:   dict[str, CleanedCustomer]   = {}   # key: clerkId or email
    products_map:    dict[str, CleanedProduct]    = {}   # key: product name
    orders_map:      dict[str, CleanedOrder]      = {}   # key: order_id
    order_items:     list[CleanedOrderItem]        = []

    # First pass: check if all truly_missing fields have decisions
    # (check against the first non-empty row to determine what's truly missing)
    action_needed_global: list[MissingFieldAction] = []

    for idx, raw_row in df.iterrows():
        row_dict: dict[str, Any] = {}

        try:
            # ── Stage 3: Clean cells ──────────────────────────────────────────
            for col in df.columns:
                col_str = str(col)
                if col_str in [str(c) for c in metadata_col_names]:
                    continue
                val = raw_row.get(col)
                # Clean by target type
                if any(col_str.endswith(f) for f in [
                    ".price", ".cost", "revenue", "profit",
                    "profit_margin", ".unitPrice", ".itemDiscount",
                ]):
                    row_dict[col_str] = _clean_numeric(val)
                elif any(col_str.endswith(f) for f in [
                    ".stock", ".quantity", "quantity",
                ]):
                    row_dict[col_str] = _clean_integer(val)
                elif "date" in col_str.lower() or col_str == "order_date":
                    row_dict[col_str] = _clean_date(val)
                elif col_str.endswith(".status"):
                    row_dict[col_str] = _normalize_status(val)
                else:
                    row_dict[col_str] = _clean_string(val)

            # Collect attributes from original df using original column names
            attributes: dict[str, Any] = {}
            for orig_col, attr_key in attr_cols.items():
                val = raw_row.get(orig_col)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    attributes[attr_key] = _clean_string(val)
                    if attr_key == "gender" and attributes[attr_key]:
                        # Warn once if gender is being stored as attribute
                        pass

            # Collect metadata from unmapped columns
            row_metadata: dict[str, Any] = {}
            for orig_col in metadata_col_names:
                val = raw_row.get(orig_col)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    cleaned_val = _clean_string(val)
                    if cleaned_val:
                        row_metadata[orig_col] = cleaned_val

            # ── Stage 4: Derive ML fields ─────────────────────────────────────
            row_dict, derived = _derive_ml_fields(row_dict, mapping.cost_pct)
            all_derived.update(derived)

            # ── Stage 6: ML field check (first row only for global action) ────
            if idx == 0:
                actions = _check_ml_fields(row_dict, mapping.missing_field_decisions)
                action_needed_global.extend(actions)

            # Apply decisions to this row
            for key, decision in mapping.missing_field_decisions.items():
                current = _extract_value(row_dict, key)
                if current is None:
                    if decision == "placeholder":
                        row_dict[key] = 0.0

            # ── Stage 5: Assemble entities ────────────────────────────────────

            # Customer
            customer = _assemble_customer(row_dict, row_metadata)
            cust_key = customer.clerkId or customer.email or customer.fullName
            if cust_key and cust_key not in customers_map:
                customers_map[cust_key] = customer
            elif cust_key and customer.fullName and not customers_map[cust_key].fullName:
                # Enrich existing customer record
                customers_map[cust_key].fullName = customer.fullName

            # Product
            product = _assemble_product(row_dict)
            if product.name and product.name not in products_map:
                products_map[product.name] = product
            elif product.name in products_map:
                # Enrich: update price/cost if missing
                existing = products_map[product.name]
                if existing.price is None and product.price is not None:
                    existing.price = product.price
                if existing.cost is None and product.cost is not None:
                    existing.cost = product.cost
                if existing.stock is None and product.stock is not None:
                    existing.stock = product.stock

            # Order
            order = _assemble_order(row_dict)
            order_key = order.externalOrderId or f"row_{idx}"
            if order_key not in orders_map:
                orders_map[order_key] = order

            # Order Item — one per row
            item_metadata = {k: v for k, v in row_metadata.items()
                             if k not in [c for c in attr_cols.keys()]}
            order_item = _assemble_order_item(row_dict, attributes, item_metadata)
            order_items.append(order_item)

        except Exception as exc:
            raw_data = {str(k): (None if pd.isna(v) else v)
                        for k, v in raw_row.items()}
            failed_rows.append(FailedRow(
                row_index=int(str(idx)),
                raw_data=raw_data,
                reason=str(exc),
            ))

    # ── Stage 6: Return action_required if needed ─────────────────────────────
    if action_needed_global:
        # Return 202 — user must decide before clean can complete
        return CleanResponse(
            summary=CleanSummary(
                total_rows=total_rows,
                clean_rows=0,
                failed_rows=len(failed_rows),
                customers_found=0,
                products_found=0,
                orders_found=0,
                order_items_found=0,
                ml_coverage_pct=0.0,
            ),
            entities=CleanedEntities(),
            failed_rows=failed_rows,
            derived_fields=list(all_derived),
            warnings=warnings,
            action_required=ActionRequired(fields=action_needed_global),
        )

    # ── Warnings for derived fields ───────────────────────────────────────────
    if all_derived:
        warnings.append(f"Derived fields calculated automatically: {sorted(all_derived)}.")

    if any(col for col in attr_cols if "gender" in attr_cols[col]):
        warnings.append(
            "Gender/category column stored in orderItem.attributes as 'gender'. "
            "Consider adding a product category mapping in your next import."
        )

    # ── ML coverage ───────────────────────────────────────────────────────────
    covered = len([k for k in ML_REQUIRED_KEYS
                   if any(r.get(k) is not None for r in [{}])])
    # Recalculate from actual data
    sample_row = {}
    if orders_map:
        first_order = next(iter(orders_map.values()))
        sample_row["revenue"]       = first_order.revenue
        sample_row["profit"]        = first_order.profit
        sample_row["profit_margin"] = first_order.profit_margin
    if products_map:
        first_product = next(iter(products_map.values()))
        sample_row["price"] = first_product.price
        sample_row["cost"]  = first_product.cost
        sample_row["stock"] = first_product.stock

    covered_count = sum(1 for k in ML_REQUIRED_KEYS if sample_row.get(k) is not None)
    ml_coverage_pct = round(covered_count / len(ML_REQUIRED_KEYS) * 100, 1)

    clean_rows = total_rows - len(failed_rows)

    return CleanResponse(
        summary=CleanSummary(
            total_rows=total_rows,
            clean_rows=clean_rows,
            failed_rows=len(failed_rows),
            customers_found=len(customers_map),
            products_found=len(products_map),
            orders_found=len(orders_map),
            order_items_found=len(order_items),
            ml_coverage_pct=ml_coverage_pct,
        ),
        entities=CleanedEntities(
            customers=list(customers_map.values()),
            products=list(products_map.values()),
            orders=list(orders_map.values()),
            order_items=order_items,
        ),
        failed_rows=failed_rows,
        derived_fields=sorted(all_derived),
        warnings=warnings,
        action_required=None,
    )