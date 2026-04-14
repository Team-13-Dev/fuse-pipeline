"""
services/parser.py

Phase 1 — Parse & Map only.
No data cleaning, no type coercion, no filling.
Returns: mapping, confidence scores, unmapped columns, 5-row sample.
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from app.models.parse_models import ColumnMapping, ParseResponse


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA & ALIASES
# ═════════════════════════════════════════════════════════════════════════════

ML_REQUIRED_COLUMNS = [
    "quantity",
    "price",
    "cost",
    "stock",
    "profit",
    "order_date",
    "product_id",
    "revenue",
    "profit_margin",
    "customer_id",
]

FIELD_ALIASES: dict[str, list[str]] = {
    "quantity": [
        "quantity", "qty", "units", "amount", "count", "num_items",
        "number_of_units", "number_of_items", "no_of_units", "no_of_items",
        "unit_quantity", "order_quantity", "ordered_qty", "purchase_qty",
        "sold_qty", "sold_units", "quantity_sold", "units_sold",
        "quantity_ordered", "quantity_purchased", "quantity_bought",
        "pieces", "pcs", "packs", "boxes", "cartons",
        "volume", "order_volume", "sales_volume",
        "items", "item_count", "item_quantity",
        "q_ty", "quant", "quantite",
        "units_ordered", "units_purchased",
    ],
    "price": [
        "price", "unit_price", "selling_price", "sale_price", "retail_price",
        "list_price", "base_price", "original_price", "regular_price",
        "msrp", "rrp", "recommended_retail_price",
        "price_per_unit", "price_each", "per_unit_price",
        "item_price", "product_price", "article_price",
        "net_price", "gross_price", "taxed_price", "pretax_price",
        "final_price", "checkout_price", "invoice_price",
        "discounted_price", "marked_price", "marked_down_price",
        "avg_price", "average_price", "mean_price", "prices_list",
    ],
    "cost": [
        "cost", "unit_cost", "cogs", "purchase_price", "cost_price",
        "cost_of_goods", "cost_of_goods_sold", "cost_of_sales",
        "landed_cost", "total_cost", "item_cost", "product_cost",
        "manufacturing_cost", "production_cost", "material_cost",
        "procurement_cost", "acquisition_cost", "buying_price",
        "supplier_price", "vendor_price", "wholesale_price",
        "overhead_cost", "direct_cost", "indirect_cost",
        "variable_cost", "fixed_cost", "blended_cost",
        "avg_cost", "average_cost", "mean_cost",
        "expense", "expenses", "expenditure", "calculated_cost",
    ],
    "stock": [
        "stock", "inventory", "stock_level", "on_hand", "available_qty", "stocks_list",
        "stock_quantity", "stock_count", "stock_units", "stock_amount",
        "current_stock", "current_inventory", "closing_stock",
        "opening_stock", "beginning_stock", "ending_stock",
        "units_in_stock", "units_available", "units_on_hand",
        "warehouse_qty", "warehouse_stock", "shelf_qty",
        "reorder_level", "safety_stock", "buffer_stock",
        "inventory_level", "inventory_count", "inventory_qty",
        "in_stock", "available_inventory", "remaining_stock",
        "remaining_qty", "leftover_qty", "leftover_stock",
        "qty_on_hand", "qty_available", "qty_in_stock",
    ],
    "profit": [
        "profit", "net_profit", "gross_profit", "earnings",
        "profit_amount", "profit_value", "profit_total",
        "net_income", "net_earnings", "net_gain",
        "operating_profit", "operating_income", "operating_earnings",
        "ebit", "ebitda",
        "contribution_margin", "gross_contribution",
        "profit_loss", "p_l", "pnl",
        "income", "gain", "surplus",
        "return", "returns", "roi",
    ],
    "profit_margin": [
        "profit_margin", "gross_margin", "net_margin", "operating_margin",
        "margin", "margin_pct", "margin_percent", "margin_percentage",
        "profit_pct", "profit_percent", "profit_percentage",
        "markup", "markup_pct",
    ],
    "revenue": [
        "revenue", "total", "sales", "total_sales", "turnover",
        "gross_revenue", "net_revenue", "total_revenue",
        "gross_sales", "net_sales", "total_income",
        "sales_amount", "sales_value", "sales_total",
        "sales_revenue", "revenue_total", "revenue_amount",
        "billing_amount", "billed_amount", "invoice_amount",
        "transaction_amount", "transaction_value", "transaction_total",
        "order_amount", "order_value", "order_total",
        "line_total", "subtotal", "sub_total", "grand_total",
        "receipts", "proceeds", "takings",
        "top_line", "topline",
    ],
    "order_date": [
        "order_date", "date", "transaction_date", "purchase_date", "sale_date",
        "order_time", "order_datetime", "order_placed_date",
        "purchase_time", "purchase_datetime",
        "transaction_time", "transaction_datetime",
        "sale_time", "sale_datetime",
        "invoice_date", "invoice_time",
        "created_at", "created_date", "creation_date",
        "placed_at", "placed_date",
        "booking_date", "booking_time",
        "ship_date", "shipment_date", "shipping_date",
        "delivery_date", "dispatch_date",
        "recorded_date", "recorded_at",
        "event_date", "timestamp", "time_stamp",
        "date_of_purchase", "date_of_sale", "date_of_order",
    ],
    "order_id": [
        "order_id", "transaction_id", "order_number", "sale_id", "invoice_id",
        "order_no", "order_ref", "order_reference",
        "transaction_no", "transaction_ref", "transaction_number",
        "invoice_no", "invoice_number", "invoice_ref",
        "receipt_id", "receipt_no", "receipt_number",
        "booking_id", "booking_no", "booking_number",
        "confirmation_id", "confirmation_number", "confirmation_no",
        "reference_id", "reference_no", "reference_number",
        "record_id", "record_no", "record_number",
        "sale_number", "sales_id", "sales_no",
        "purchase_id", "purchase_no", "purchase_number",
        "po_number", "po_id", "purchase_order_id", "purchase_order_no",
        "contract_id", "contract_no", "contract_number",
        "case_id", "case_no", "ticket_id", "ticket_no",
    ],
    "product_id": [
        "product_id", "item_id", "sku", "product_code", "item_code",
        "product_no", "product_number", "product_ref", "product_reference",
        "item_no", "item_number", "item_ref",
        "article_id", "article_no", "article_number", "article_code",
        "part_id", "part_no", "part_number", "part_code",
        "variant_id", "variant_code", "variant_no",
        "catalog_id", "catalogue_id", "catalog_no", "catalogue_no",
        "barcode", "bar_code", "upc", "ean", "gtin", "isbn",
        "model_no", "model_number", "model_id", "model_code",
        "style_id", "style_no", "style_code",
        "good_id", "goods_id",
        "merchandise_id", "merchandise_code",
    ],
    "customer_id": [
        "customer_id", "client_id", "buyer_id", "shopper_id", "consumer_id",
        "customer_no", "customer_number", "customer_code", "customer_ref",
        "client_no", "client_number", "client_code", "client_ref",
        "member_id", "member_no", "member_number",
        "account_id", "account_no", "account_number",
        "user_id", "user_no",
        "contact_id", "contact_no",
        "loyalty_id", "loyalty_no",
        "cx_id", "cust_id",
    ],
}

# Clothing-specific → maps to orderItem.attributes (jsonb)
ATTRIBUTE_ALIASES: dict[str, list[str]] = {
    "size": [
        "size", "sizes", "size_code", "size_label",
        "clothing_size", "garment_size", "apparel_size",
        "s_m_l", "xs_s_m_l_xl", "uk_size", "eu_size", "us_size",
        "waist", "chest", "length",
        # Arabic transliterations handled separately
        "maqas", "qias", "qiyas",
    ],
    "color": [
        "color", "colour", "colors", "colour_name", "color_name",
        "shade", "hue", "tint",
        "item_color", "product_color", "variant_color",
        "lawn", "loun", "lon",  # transliterations of لون
    ],
    "season": [
        "season", "seasons", "season_code", "seasonal",
        "collection_season", "sales_season",
        "spring_summer", "autumn_winter", "ss", "aw", "fw",
        "mawsim", "mawsem",  # transliterations of موسم
    ],
    "collection": [
        "collection", "collections", "collection_name", "collection_code",
        "line", "line_name", "product_line", "fashion_line",
        "series", "range", "range_name",
        "capsule", "capsule_collection",
        "majmuaa", "koleksyon",
    ],
    "gender": [
        "gender", "gender_code", "target_gender",
        "mens", "womens", "unisex", "kids", "boys", "girls",
        "male", "female",
        "rjali", "harimy", "harimi", "jinsayn", "jinsain",  # Arabic transliterations
        "men", "women",
    ],
}

# Arabic character transliteration map (character-level)
ARABIC_TRANSLIT: dict[str, str] = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa",
    "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "a", "غ": "gh",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l",
    "م": "m", "ن": "n", "ه": "h", "و": "w",
    "ي": "y", "ى": "a", "ة": "a",
    "ء": "", "ئ": "y", "ؤ": "w",
    # Common diacritics — strip them
    "\u064b": "", "\u064c": "", "\u064d": "",
    "\u064e": "", "\u064f": "", "\u0650": "",
    "\u0651": "", "\u0652": "",
    " ": "_",
}

# Arabic aliases added directly to FIELD_ALIASES
ARABIC_FIELD_ALIASES: dict[str, list[str]] = {
    "quantity":     ["كمية", "الكمية", "عدد", "وحدات"],
    "price":        ["سعر", "السعر", "سعر_البيع", "سعر_الوحدة"],
    "cost":         ["تكلفة", "التكلفة", "سعر_التكلفة", "تكلفة_الوحدة"],
    "stock":        ["مخزون", "المخزون", "الكميه_المتاحه", "متاح"],
    "profit":       ["ربح", "الربح", "صافي_الربح"],
    "profit_margin":["هامش_الربح", "هامش_ربح", "نسبة_الربح"],
    "revenue":      ["إيرادات", "المبيعات", "إجمالي_المبيعات", "مبيعات"],
    "order_date":   ["تاريخ_الطلب", "تاريخ", "تاريخ_البيع"],
    "order_id":     ["رقم_الطلب", "معرف_الطلب", "كود_الطلب"],
    "product_id":   ["معرف_المنتج", "كود_المنتج", "رقم_المنتج"],
    "customer_id":  ["معرف_العميل", "كود_العميل", "رقم_العميل"],
}

ARABIC_ATTRIBUTE_ALIASES: dict[str, list[str]] = {
    "size":       ["مقاس", "المقاس", "قياس"],
    "color":      ["لون", "اللون", "لونه"],
    "season":     ["موسم", "الموسم"],
    "collection": ["مجموعة", "الكولكشن", "كولكشن", "تشكيلة"],
    "gender":     ["جنس", "جنسين", "رجالي", "حريمي"],
}

FUZZY_THRESHOLD = 80


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _transliterate_arabic(text: str) -> str:
    """Character-by-character Arabic → Latin transliteration."""
    result = []
    for ch in text:
        result.append(ARABIC_TRANSLIT.get(ch, ch))
    return "".join(result)


def normalize_column_name(col: str) -> str:
    """Lowercase, strip, replace spaces/special chars with underscores."""
    col = str(col).strip()
    # Normalize unicode (NFD → drop combining marks for Latin, keep Arabic)
    col = unicodedata.normalize("NFC", col)
    col = col.lower()
    col = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "_", col)
    return col.strip("_")


def _detect_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    """
    Scan the first `max_scan` rows and return the index of the row
    most likely to be the header.

    Scoring per row:
      +2  for each cell that looks like a column name
          (non-numeric string, no spaces or only underscores/spaces, len 2–40)
      +1  for each non-empty cell
      -5  if the row has fewer than half the columns of the widest row
    """
    # Widest row in scan window (to penalise sparse title rows)
    max_cols = max(
        df_raw.iloc[i].count()
        for i in range(min(max_scan, len(df_raw)))
    )

    best_row = 0
    best_score = -999

    for i in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[i]
        filled = row.count()
        score: float = 0.0

        # Penalise rows with very few filled cells vs the widest row
        if filled < max_cols * 0.5:
            score -= 5

        for v in row:
            if pd.isna(v):
                continue
            s = str(v).strip()
            score += 0.5  # any non-empty cell

            # Looks like a column name: mostly letters/underscores, 2–40 chars
            if re.fullmatch(r"[a-zA-Z\u0600-\u06FF][a-zA-Z0-9 _\-\u0600-\u06FF]{1,39}", s):
                # Extra bonus if it doesn't look like a pure number
                if not re.fullmatch(r"[\d,.\-\s]+", s):
                    score += 2

        if score > best_score:
            best_score = score
            best_row = i

    return best_row


# ═════════════════════════════════════════════════════════════════════════════
# MAPPING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _build_alias_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      alias_to_field  — alias string → schema field name
      arabic_to_field — Arabic alias string → schema field name (exact match)
    """
    alias_to_field: dict[str, str] = {}
    arabic_to_field: dict[str, str] = {}

    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            alias_to_field[alias] = field

    # Arabic exact aliases
    for field, ar_aliases in ARABIC_FIELD_ALIASES.items():
        for alias in ar_aliases:
            arabic_to_field[alias] = field

    return alias_to_field, arabic_to_field


def _build_attribute_lookup() -> tuple[dict[str, str], dict[str, str]]:
    attr_alias: dict[str, str] = {}
    attr_arabic: dict[str, str] = {}

    for attr_key, aliases in ATTRIBUTE_ALIASES.items():
        for alias in aliases:
            attr_alias[alias] = attr_key

    for attr_key, ar_aliases in ARABIC_ATTRIBUTE_ALIASES.items():
        for alias in ar_aliases:
            attr_arabic[alias] = attr_key

    return attr_alias, attr_arabic


def map_columns(
    columns: list[str],
) -> tuple[
    list[ColumnMapping],   # all mappings (schema + attribute + unmapped)
    list[str],             # warnings
]:
    alias_to_field, arabic_to_field = _build_alias_lookup()
    attr_alias, attr_arabic = _build_attribute_lookup()
    all_schema_aliases = list(alias_to_field.keys())

    warnings: list[str] = []
    arabic_detected = False

    # ── Step 1: score every column ────────────────────────────────────────────
    # Each entry: (original_col, mapped_to, score, method)
    candidates: list[tuple[str, str, float, str]] = []

    for col in columns:
        norm = normalize_column_name(col)
        is_arabic = _has_arabic(col) or _has_arabic(norm)
        if is_arabic:
            arabic_detected = True

        matched_to: str | None = None
        score: float = 0.0
        method: str = "unmatched"

        # ── 1a. Arabic exact match (schema fields) ────────────────────────
        if is_arabic and col.strip() in arabic_to_field:
            matched_to = arabic_to_field[col.strip()]
            score = 100.0
            method = "arabic_alias"

        # ── 1b. Arabic exact match (attribute fields) ─────────────────────
        elif is_arabic and col.strip() in attr_arabic:
            matched_to = f"__attributes__.{attr_arabic[col.strip()]}"
            score = 100.0
            method = "arabic_alias"

        # ── 1c. Transliterate Arabic → Latin, then fuzzy schema match ─────
        elif is_arabic:
            translit = normalize_column_name(_transliterate_arabic(col))
            # Try schema aliases
            result = process.extractOne(
                translit,
                all_schema_aliases,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if result:
                matched_alias, sc, _ = result
                matched_to = alias_to_field[matched_alias]
                score = float(sc)
                method = "arabic_transliterated"
            else:
                # Try attribute aliases
                attr_result = process.extractOne(
                    translit,
                    list(attr_alias.keys()),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=FUZZY_THRESHOLD,
                )
                if attr_result:
                    matched_alias, sc, _ = attr_result
                    matched_to = f"__attributes__.{attr_alias[matched_alias]}"
                    score = float(sc)
                    method = "arabic_transliterated"

        else:
            # ── 1d. Latin fuzzy match — schema fields ─────────────────────
            result = process.extractOne(
                norm,
                all_schema_aliases,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if result:
                matched_alias, sc, _ = result
                matched_to = alias_to_field[matched_alias]
                score = float(sc)
                method = "fuzzy" if sc < 100 else "exact"
            else:
                # ── 1e. Latin fuzzy match — attribute fields ───────────────
                attr_result = process.extractOne(
                    norm,
                    list(attr_alias.keys()),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=FUZZY_THRESHOLD,
                )
                if attr_result:
                    matched_alias, sc, _ = attr_result
                    matched_to = f"__attributes__.{attr_alias[matched_alias]}"
                    score = float(sc)
                    method = "fuzzy" if sc < 100 else "exact"

        candidates.append((col, matched_to or "__unmapped__", score, method))

    if arabic_detected:
        warnings.append("Arabic column headers detected — transliteration applied.")

    # ── Step 2: resolve conflicts (highest score claims each schema field) ────
    # Sort descending by score
    candidates.sort(key=lambda x: x[2], reverse=True)

    used_targets: set[str] = set()
    mappings: list[ColumnMapping] = []

    for col, target, score, method in candidates:
        if target == "__unmapped__":
            mappings.append(ColumnMapping(
                original_column=col,
                mapped_to="__unmapped__",
                confidence=0.0,
                match_method="unmatched",
            ))
            continue

        # Attribute targets are unique per attribute key — allow one per attr
        if target in used_targets:
            warnings.append(
                f"Column '{col}' matched '{target}' (score {score:.0f}) "
                f"but already claimed by another column — moved to unmapped."
            )
            mappings.append(ColumnMapping(
                original_column=col,
                mapped_to="__unmapped__",
                confidence=score,
                match_method=method,
            ))
        else:
            used_targets.add(target)
            mappings.append(ColumnMapping(
                original_column=col,
                mapped_to=target,
                confidence=score,
                match_method=method,
            ))

    return mappings, warnings


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def parse_file(file_bytes: bytes, filename: str) -> ParseResponse:
    warnings: list[str] = []

    # ── Load raw (no header assumption) ──────────────────────────────────────
    if filename.endswith(".csv"):
        encodings = ["utf-8", "latin-1", "cp1252"]
        file_text: str | None = None
        used_enc = "utf-8"
        for enc in encodings:
            try:
                file_text = file_bytes.decode(enc)
                used_enc = enc
                break
            except UnicodeDecodeError:
                continue
        if file_text is None:
            raise ValueError("Could not decode CSV file. Try saving as UTF-8.")

        # Find the max number of commas in the first 20 lines
        # to determine the true column count (handles short title rows)
        lines = file_text.splitlines()
        scan_lines = lines[:20]
        max_fields = max((line.count(",") + 1) for line in scan_lines if line.strip())

        # Build column names 0..max_fields-1 for raw load
        raw_names = list(range(max_fields))
        df_raw = pd.read_csv(
            io.StringIO(file_text),
            header=None,
            names=raw_names,
            encoding=used_enc,
            on_bad_lines="skip",
            engine="python",
        )
    elif filename.endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    else:
        raise ValueError(f"Unsupported file format: '{filename}'. Use .csv or .xlsx.")

    # ── Detect header row ─────────────────────────────────────────────────────
    header_row_idx = _detect_header_row(df_raw)
    if header_row_idx > 0:
        warnings.append(
            f"Header detected at row {header_row_idx} (not row 0). "
            f"Rows 0–{header_row_idx - 1} skipped as pre-header content."
        )

    # Re-read with correct header
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                header=header_row_idx,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    header=header_row_idx,
                    encoding="latin-1",
                )
            except Exception:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    header=header_row_idx,
                    encoding="cp1252",
                )
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), header=header_row_idx)

    original_columns: list[str] = [str(c) for c in df.columns.tolist()]

    # ── Run mapping ───────────────────────────────────────────────────────────
    mappings, map_warnings = map_columns(original_columns)
    warnings.extend(map_warnings)

    # ── Build response structures ─────────────────────────────────────────────
    detected_mapping: dict[str, str] = {}
    confidence_scores: dict[str, float] = {}
    match_methods: dict[str, str] = {}
    attribute_columns: dict[str, str] = {}
    unmapped_columns: list[str] = []

    for m in mappings:
        if m.mapped_to == "__unmapped__":
            unmapped_columns.append(m.original_column)
        elif m.mapped_to.startswith("__attributes__."):
            attr_key = m.mapped_to.replace("__attributes__.", "")
            attribute_columns[m.original_column] = attr_key
            detected_mapping[m.original_column] = m.mapped_to
            confidence_scores[m.original_column] = m.confidence
            match_methods[m.original_column] = m.match_method
        else:
            detected_mapping[m.original_column] = m.mapped_to
            confidence_scores[m.original_column] = m.confidence
            match_methods[m.original_column] = m.match_method

    # ── ML coverage check ─────────────────────────────────────────────────────
    mapped_schema_fields = set(
        v for v in detected_mapping.values()
        if not v.startswith("__attributes__")
    )
    ml_missing = [f for f in ML_REQUIRED_COLUMNS if f not in mapped_schema_fields]
    ml_coverage_pct = round(
        (len(ML_REQUIRED_COLUMNS) - len(ml_missing)) / len(ML_REQUIRED_COLUMNS) * 100, 1
    )

    if ml_missing:
        warnings.append(
            f"ML required columns not found in file: {ml_missing}. "
            f"Coverage: {ml_coverage_pct}%."
        )

    # ── Sample rows (raw — no transformation) ────────────────────────────────
    sample_df = df.head(5)
    # Replace NaN with None for JSON serialization
    sample_rows: list[dict[str, Any]] = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample_df.to_dict(orient="records")
    ]

    return ParseResponse(
        header_row_index=header_row_idx,
        detected_mapping=detected_mapping,
        confidence_scores=confidence_scores,
        match_methods=match_methods,
        ml_required=ML_REQUIRED_COLUMNS,
        ml_missing=ml_missing,
        ml_coverage_pct=ml_coverage_pct,
        attribute_columns=attribute_columns,
        unmapped_columns=unmapped_columns,
        sample_rows=sample_rows,
        warnings=warnings,
    )