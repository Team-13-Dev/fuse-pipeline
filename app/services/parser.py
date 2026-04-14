"""
services/parser.py

Phase 1 — Parse & Map only.
No data cleaning, no type coercion, no filling.
Returns: mapping, confidence scores, unmapped columns, 5-row sample.

Field aliases now cover the full DB schema using "table.field" dot notation.
ML-required fields keep their flat keys since they are derived/computed.
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from app.core.ml_config import ML_FIELDS, ML_REQUIRED_KEYS, get_derivation_rules
from app.models.parse_models import ColumnMapping, DerivableField, ParseResponse


# ═════════════════════════════════════════════════════════════════════════════
# FULL DB SCHEMA FIELD ALIASES
# Format: "table.field" → [aliases...]
# ML computed fields use flat keys (no dot notation)
# ═════════════════════════════════════════════════════════════════════════════

FIELD_ALIASES: dict[str, list[str]] = {

    # ── customer ──────────────────────────────────────────────────────────────
    "customer.fullName": [
        "customername", "customer_name", "full_name", "fullname",
        "name", "client_name", "buyer_name", "shopper_name",
        "firstname", "first_name", "lastname", "last_name",
        "contact_name", "person_name", "recipient_name",
        "اسم_العميل", "الاسم", "اسم",
    ],
    "customer.email": [
        "email", "email_address", "emailaddress", "e_mail",
        "mail", "customer_email", "client_email", "buyer_email",
        "contact_email", "user_email", "electronic_mail",
        "البريد_الالكتروني", "ايميل",
    ],
    "customer.phoneNumber": [
        "phone", "phonenumber", "phone_number", "mobile",
        "mobile_number", "cell", "cell_number", "telephone",
        "tel", "contact_number", "contact_phone",
        "customer_phone", "client_phone", "buyer_phone",
        "phone_no", "mob", "mob_no", "phone_num",
        "رقم_الهاتف", "موبايل", "تليفون",
    ],
    "customer.segment": [
        "segment", "customer_segment", "client_segment",
        "customer_type", "client_type", "buyer_type",
        "tier", "customer_tier", "loyalty_tier",
        "class", "customer_class", "membership",
        "شريحة_العميل", "نوع_العميل",
    ],

    # ── product ───────────────────────────────────────────────────────────────
    "product.name": [
        "product", "product_name", "productname", "item",
        "item_name", "itemname", "article", "article_name",
        "good", "goods_name", "merchandise", "merchandise_name",
        "product_title", "item_title", "product_description_short",
        "اسم_المنتج", "المنتج", "صنف",
    ],
    "product.description": [
        "description", "product_description", "item_description",
        "details", "product_details", "item_details",
        "notes", "product_notes", "remarks",
        "وصف", "وصف_المنتج",
    ],
    "product.stock": [
        "stock", "inventory", "stock_level", "on_hand", "available_qty", "stocks_list",
        "stock_quantity", "stock_count", "stock_units",
        "current_stock", "current_inventory", "closing_stock",
        "units_in_stock", "units_available", "units_on_hand",
        "qty_on_hand", "qty_available", "qty_in_stock",
        "مخزون", "المخزون", "الكميه_المتاحه",
    ],
    "product.price": [
        "price", "unit_price", "selling_price", "sale_price", "retail_price",
        "list_price", "base_price", "original_price", "regular_price",
        "price_per_unit", "price_each", "per_unit_price",
        "item_price", "product_price", "net_price", "gross_price",
        "final_price", "checkout_price", "invoice_price",
        "avg_price", "average_price", "prices_list",
        "سعر", "السعر", "سعر_البيع", "سعر_الوحدة",
    ],
    "product.cost": [
        "cost", "unit_cost", "cogs", "purchase_price", "cost_price",
        "cost_of_goods", "cost_of_goods_sold", "landed_cost",
        "item_cost", "product_cost", "manufacturing_cost",
        "buying_price", "supplier_price", "vendor_price", "wholesale_price",
        "avg_cost", "average_cost", "calculated_cost",
        "تكلفة", "التكلفة", "سعر_التكلفة",
    ],

    # ── order ─────────────────────────────────────────────────────────────────
    "order.status": [
        "status", "order_status", "orderstatus",
        "fulfillment_status", "shipment_status", "delivery_status",
        "payment_status", "transaction_status",
        "state", "order_state",
        "حالة_الطلب", "حالة",
    ],
    "order.address": [
        "address", "shipping_address", "delivery_address",
        "customer_address", "ship_to", "deliver_to",
        "street_address", "full_address",
        "عنوان", "عنوان_الشحن",
    ],
    "order.orderVoucher": [
        "voucher", "coupon", "coupon_code", "voucher_code",
        "promo_code", "discount_code", "order_voucher",
        "كوبون", "قسيمة",
    ],

    # ── order_item ────────────────────────────────────────────────────────────
    "order_item.quantity": [
        "quantity", "qty", "units", "amount", "count",
        "num_items", "number_of_units", "number_of_items",
        "unit_quantity", "order_quantity", "ordered_qty",
        "sold_qty", "sold_units", "quantity_sold", "units_sold",
        "pieces", "pcs", "packs",
        "كمية", "الكمية", "عدد", "وحدات",
    ],
    "order_item.unitPrice": [
        "unit_price", "price_per_unit", "item_price", "per_unit_price",
        "line_price", "each_price", "single_price",
        "سعر_الوحدة",
    ],
    "order_item.itemDiscount": [
        "discount", "item_discount", "line_discount",
        "product_discount", "unit_discount",
        "discount_amount", "discount_value",
        "خصم", "خصم_المنتج",
    ],

    # ── ML computed / derived fields (flat keys, no dot notation) ─────────────
    "quantity": [
        "quantity", "qty", "units_ordered",
        "كمية", "الكمية",
    ],
    "price": [
        "price", "selling_price", "unit_price",
        "سعر", "السعر",
    ],
    "cost": [
        "cost", "unit_cost", "cost_price", "calculated_cost",
        "تكلفة", "التكلفة",
    ],
    "stock": [
        "stock", "inventory", "available_qty",
        "مخزون", "المخزون",
    ],
    "revenue": [
        "revenue", "total", "sales", "total_sales", "turnover",
        "total_amount", "totalamount", "total_price", "totalprice",
        "net_amount", "order_total", "grand_total", "subtotal",
        "sale_total", "payment_amount", "invoice_total",
        "إيرادات", "المبيعات", "مبيعات",
    ],
    "profit": [
        "profit", "net_profit", "gross_profit", "earnings",
        "profit_amount", "net_income", "net_gain",
        "ربح", "الربح", "صافي_الربح",
    ],
    "profit_margin": [
        "profit_margin", "gross_margin", "net_margin",
        "margin", "margin_pct", "margin_percent",
        "هامش_الربح", "هامش_ربح",
    ],
    "order_date": [
        "order_date", "date", "transaction_date", "purchase_date", "sale_date",
        "order_time", "order_datetime", "created_at", "created_date",
        "invoice_date", "ship_date", "delivery_date",
        "تاريخ_الطلب", "تاريخ", "تاريخ_البيع",
    ],
    "order_id": [
        "order_id", "transaction_id", "order_number", "sale_id", "invoice_id",
        "order_no", "order_ref", "receipt_id", "confirmation_id",
        "reference_id", "record_id",
        "رقم_الطلب", "معرف_الطلب",
    ],
    "product_id": [
        "product_id", "item_id", "sku", "product_code", "item_code",
        "product_no", "article_id", "part_no", "barcode", "upc", "ean",
        "model_no", "style_id",
        "معرف_المنتج", "كود_المنتج",
    ],
    "customer_id": [
        "customer_id", "client_id", "buyer_id", "shopper_id",
        "customer_no", "member_id", "account_id", "loyalty_id",
        "cx_id", "cust_id",
        "معرف_العميل", "كود_العميل",
    ],
}

# ─── Clothing-specific → orderItem.attributes jsonb ──────────────────────────
ATTRIBUTE_ALIASES: dict[str, list[str]] = {
    "size": [
        "size", "sizes", "size_code", "size_label",
        "clothing_size", "garment_size", "apparel_size",
        "s_m_l", "xs_s_m_l_xl", "uk_size", "eu_size", "us_size",
        "waist", "chest", "length",
        "maqas", "qias", "qiyas",
        "مقاس", "المقاس", "قياس",
    ],
    "color": [
        "color", "colour", "colors", "colour_name", "color_name",
        "shade", "hue", "item_color", "product_color",
        "lawn", "loun", "lon",
        "لون", "اللون", "لونه",
    ],
    "season": [
        "season", "seasons", "season_code", "seasonal",
        "collection_season", "spring_summer", "autumn_winter",
        "مawsim", "mawsem",
        "موسم", "الموسم",
    ],
    "collection": [
        "collection", "collections", "collection_name", "collection_code",
        "line", "product_line", "series", "range",
        "capsule", "capsule_collection",
        "مجموعة", "الكولكشن", "كولكشن", "تشكيلة",
    ],
    "gender": [
        "gender", "gender_code", "target_gender", "target_audience",
        "mens", "womens", "unisex", "kids", "boys", "girls",
        "male", "female", "men", "women",
        "rjali", "harimy", "harimi",
        "جنس", "رجالي", "حريمي",
    ],
}

# ─── Arabic character transliteration ────────────────────────────────────────
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
    "\u064b": "", "\u064c": "", "\u064d": "",
    "\u064e": "", "\u064f": "", "\u0650": "",
    "\u0651": "", "\u0652": "",
    " ": "_",
}

FUZZY_THRESHOLD = 85   # High enough to block false positives like Country→quantity

# Columns that should never be auto-mapped via fuzzy matching
# (too generic / likely to produce false positives)
FUZZY_DENYLIST = {
    "country", "city", "region", "state", "province",
    "continent", "area", "zone", "location", "locale",
}


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _transliterate_arabic(text: str) -> str:
    return "".join(ARABIC_TRANSLIT.get(ch, ch) for ch in text)


def normalize_column_name(col: str) -> str:
    col = str(col).strip()
    col = unicodedata.normalize("NFC", col)
    col = col.lower()
    col = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "_", col)
    return col.strip("_")


def _detect_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    max_cols = max(
        df_raw.iloc[i].count()
        for i in range(min(max_scan, len(df_raw)))
    )
    best_row, best_score = 0, -999.0
    for i in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[i]
        filled = row.count()
        score = 0.0
        if filled < max_cols * 0.5:
            score -= 5
        for v in row:
            if pd.isna(v):
                continue
            s = str(v).strip()
            score += 0.5
            if re.fullmatch(r"[a-zA-Z\u0600-\u06FF][a-zA-Z0-9 _\-\u0600-\u06FF]{1,39}", s):
                if not re.fullmatch(r"[\d,.\-\s]+", s):
                    score += 2
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


# ═════════════════════════════════════════════════════════════════════════════
# MAPPING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _build_lookups() -> tuple[dict[str, str], dict[str, str]]:
    """Build alias → target lookups for schema fields and attribute fields."""
    alias_to_field: dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            norm = normalize_column_name(alias)
            alias_to_field[norm] = field

    attr_alias: dict[str, str] = {}
    for attr_key, aliases in ATTRIBUTE_ALIASES.items():
        for alias in aliases:
            norm = normalize_column_name(alias)
            attr_alias[norm] = attr_key

    return alias_to_field, attr_alias


def map_columns(columns: list[str]) -> tuple[list[ColumnMapping], list[str]]:
    alias_to_field, attr_alias = _build_lookups()
    all_schema_aliases = list(alias_to_field.keys())
    all_attr_aliases   = list(attr_alias.keys())

    warnings: list[str] = []
    arabic_detected = False
    candidates: list[tuple[str, str, float, str]] = []

    for col in columns:
        norm = normalize_column_name(col)
        is_arabic = _has_arabic(col) or _has_arabic(norm)
        if is_arabic:
            arabic_detected = True

        matched_to: str | None = None
        score: float = 0.0
        method: str = "unmatched"

        # 1. Direct exact match (normalized)
        if norm in alias_to_field:
            matched_to = alias_to_field[norm]
            score = 100.0
            method = "exact"
        elif norm in attr_alias:
            matched_to = f"__attributes__.{attr_alias[norm]}"
            score = 100.0
            method = "exact"
        else:
            # 2. Denylist guard — skip fuzzy for columns that are clearly not schema fields
            if norm in FUZZY_DENYLIST:
                candidates.append((col, "__unmapped__", 0.0, "unmatched"))
                continue

            # 3. Arabic transliteration
            lookup_str = normalize_column_name(_transliterate_arabic(col)) if is_arabic else norm

            # 4. Fuzzy match against schema aliases
            result = process.extractOne(
                lookup_str,
                all_schema_aliases,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if result:
                matched_alias, sc, _ = result
                matched_to = alias_to_field[matched_alias]
                score = float(sc)
                method = "arabic_transliterated" if is_arabic else ("exact" if sc == 100 else "fuzzy")
            else:
                # 4. Fuzzy match against attribute aliases
                attr_result = process.extractOne(
                    lookup_str,
                    all_attr_aliases,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=FUZZY_THRESHOLD,
                )
                if attr_result:
                    matched_alias, sc, _ = attr_result
                    matched_to = f"__attributes__.{attr_alias[matched_alias]}"
                    score = float(sc)
                    method = "arabic_transliterated" if is_arabic else ("exact" if sc == 100 else "fuzzy")

        candidates.append((col, matched_to or "__unmapped__", score, method))

    if arabic_detected:
        warnings.append("Arabic column headers detected — transliteration applied.")

    # Conflict resolution: highest score claims each target
    candidates.sort(key=lambda x: x[2], reverse=True)
    used_targets: set[str] = set()
    mappings: list[ColumnMapping] = []

    for col, target, score, method in candidates:
        if target == "__unmapped__":
            mappings.append(ColumnMapping(
                original_column=col, mapped_to="__unmapped__",
                confidence=0.0, match_method="unmatched",
            ))
            continue
        if target in used_targets:
            warnings.append(
                f"Column '{col}' matched '{target}' (score {score:.0f}) "
                f"but already claimed — moved to unmapped."
            )
            mappings.append(ColumnMapping(
                original_column=col, mapped_to="__unmapped__",
                confidence=score, match_method=method,
            ))
        else:
            used_targets.add(target)
            mappings.append(ColumnMapping(
                original_column=col, mapped_to=target,
                confidence=score, match_method=method,
            ))

    return mappings, warnings


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def parse_file(file_bytes: bytes, filename: str) -> ParseResponse:
    warnings: list[str] = []

    # ── Load raw ──────────────────────────────────────────────────────────────
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
        lines = file_text.splitlines()
        max_fields = max((l.count(",") + 1) for l in lines[:20] if l.strip())
        raw_names = list(range(max_fields))
        df_raw = pd.read_csv(
            io.StringIO(file_text), header=None, names=raw_names,
            encoding=used_enc, on_bad_lines="skip", engine="python",
        )
    elif filename.endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    else:
        raise ValueError(f"Unsupported file format: '{filename}'. Use .csv or .xlsx.")

    # ── Detect header ─────────────────────────────────────────────────────────
    header_row_idx = _detect_header_row(df_raw)
    if header_row_idx > 0:
        warnings.append(
            f"Header detected at row {header_row_idx}. "
            f"Rows 0–{header_row_idx - 1} skipped."
        )

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), header=header_row_idx, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), header=header_row_idx, encoding="latin-1")
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), header=header_row_idx)

    original_columns: list[str] = [str(c) for c in df.columns.tolist()]

    # ── Map ───────────────────────────────────────────────────────────────────
    mappings, map_warnings = map_columns(original_columns)
    warnings.extend(map_warnings)

    # ── Build response structures ─────────────────────────────────────────────
    detected_mapping:  dict[str, str]   = {}
    confidence_scores: dict[str, float] = {}
    match_methods:     dict[str, str]   = {}
    attribute_columns: dict[str, str]   = {}
    unmapped_columns:  list[str]        = []

    for m in mappings:
        if m.mapped_to == "__unmapped__":
            unmapped_columns.append(m.original_column)
        elif m.mapped_to.startswith("__attributes__."):
            attr_key = m.mapped_to.replace("__attributes__.", "")
            attribute_columns[m.original_column] = attr_key
            detected_mapping[m.original_column]  = m.mapped_to
            confidence_scores[m.original_column] = m.confidence
            match_methods[m.original_column]     = m.match_method
        else:
            detected_mapping[m.original_column]  = m.mapped_to
            confidence_scores[m.original_column] = m.confidence
            match_methods[m.original_column]     = m.match_method

    # ── ML coverage ───────────────────────────────────────────────────────────
    mapped_targets = set(detected_mapping.values())
    # Extract flat ML keys from mapped targets
    def _to_ml_key(target: str) -> str | None:
        # If it's a flat ML key
        if target in ML_REQUIRED_KEYS:
            return target
        # If it's a table.field that maps to an ML key
        for ml_key, cfg in ML_FIELDS.items():
            if cfg["table"] and f"{cfg['table']}.{cfg['field']}" == target:
                return ml_key
        return None

    covered_ml_keys: set[str] = set()
    for target in mapped_targets:
        key = _to_ml_key(target)
        if key:
            covered_ml_keys.add(key)

    ml_missing = [k for k in ML_REQUIRED_KEYS if k not in covered_ml_keys]

    # ── Derivation engine ─────────────────────────────────────────────────────
    derivation_rules = get_derivation_rules()
    derivable_fields: list[DerivableField] = []
    truly_missing = list(ml_missing)

    # Build field→original_col map for source_columns display
    target_to_col: dict[str, str] = {v: k for k, v in detected_mapping.items()
                                      if not v.startswith("__attributes__")}

    for _ in range(6):  # max passes
        resolved_any = False
        for field, formula, requires in derivation_rules:
            if field not in truly_missing:
                continue
            available = covered_ml_keys | {d.field for d in derivable_fields}
            if all(r in available for r in requires):
                source_cols = [
                    target_to_col.get(r, target_to_col.get(f"{ML_FIELDS.get(r, {}).get('table', '')}.{ML_FIELDS.get(r, {}).get('field', '')}", f"[derived:{r}]"))
                    for r in requires
                ]
                derivable_fields.append(DerivableField(
                    field=field, formula=formula,
                    requires=requires, source_columns=source_cols,
                ))
                truly_missing.remove(field)
                resolved_any = True
        if not resolved_any:
            break

    effective = len(ML_REQUIRED_KEYS) - len(truly_missing)
    ml_coverage_pct = round(effective / len(ML_REQUIRED_KEYS) * 100, 1)

    if derivable_fields:
        warnings.append(
            f"Fields to be derived during cleaning: {[d.field for d in derivable_fields]}."
        )
    for field in truly_missing:
        warnings.append(f"Cannot resolve ML field '{field}' — user decision required at /clean.")

    # ── Sample rows ───────────────────────────────────────────────────────────
    sample_rows: list[dict[str, Any]] = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in df.head(5).to_dict(orient="records")
    ]

    return ParseResponse(
        header_row_index=header_row_idx,
        detected_mapping=detected_mapping,
        confidence_scores=confidence_scores,
        match_methods=match_methods,
        ml_required=ML_REQUIRED_KEYS,
        ml_missing=ml_missing,
        ml_coverage_pct=ml_coverage_pct,
        derivable_fields=derivable_fields,
        truly_missing=truly_missing,
        attribute_columns=attribute_columns,
        unmapped_columns=unmapped_columns,
        sample_rows=sample_rows,
        warnings=warnings,
    )