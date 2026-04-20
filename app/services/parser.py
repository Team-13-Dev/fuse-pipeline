"""
services/parser.py  — v3  (fast loader)

Smarter column mapping with:
- human-readable labels for every field
- grouped output (Customer / Product / Order / Notes)
- sample values per column shown to the user
- retail-aware aliases (InvoiceNo, StockCode, Description, UnitPrice)
- geographic denylist preventing Country→quantity false positives
- higher fuzzy threshold for numeric/date targets
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
# FIELD REGISTRY
# Each entry: target → { aliases, human_label, group }
# ═════════════════════════════════════════════════════════════════════════════

FIELD_REGISTRY: dict[str, dict] = {

    # ── Customer ──────────────────────────────────────────────────────────────
    "customer_id": {
        "human_label": "Customer ID",
        "group": "Customer",
        "aliases": [
            "customerid", "customer_id", "client_id", "buyer_id",
            "customer_no", "member_id", "account_id", "loyalty_id",
            "cust_id", "cx_id", "user_id",
            "معرف_العميل", "كود_العميل", "رقم_العميل",
        ],
    },
    "customer.fullName": {
        "human_label": "Customer name",
        "group": "Customer",
        "aliases": [
            "customername", "customer_name", "full_name", "fullname",
            "name", "client_name", "buyer_name", "recipient_name",
            "firstname", "first_name", "lastname", "last_name",
            "اسم_العميل", "الاسم", "اسم",
        ],
    },
    "customer.email": {
        "human_label": "Email address",
        "group": "Customer",
        "aliases": [
            "email", "email_address", "emailaddress", "e_mail",
            "mail", "customer_email", "client_email",
            "البريد_الالكتروني", "ايميل",
        ],
    },
    "customer.phoneNumber": {
        "human_label": "Phone number",
        "group": "Customer",
        "aliases": [
            "phone", "phonenumber", "phone_number", "mobile",
            "mobile_number", "cell", "telephone", "tel",
            "contact_number", "mob",
            "رقم_الهاتف", "موبايل", "تليفون",
        ],
    },
    "customer.segment": {
        "human_label": "Customer segment / type",
        "group": "Customer",
        "aliases": [
            "segment", "customer_segment", "customer_type",
            "tier", "customer_tier", "loyalty_tier", "membership",
            "شريحة_العميل", "نوع_العميل",
        ],
    },

    # ── Product ───────────────────────────────────────────────────────────────
    "product_id": {
        "human_label": "Product SKU / code",
        "group": "Product",
        "aliases": [
            "stockcode", "stock_code", "sku", "sku_code", "skucode", "product_id", "item_id",
            "product_code", "item_code", "article_id", "part_no",
            "barcode", "upc", "ean", "model_no", "style_id",
            "معرف_المنتج", "كود_المنتج", "رقم_المنتج",
        ],
    },
    "product.name": {
        "human_label": "Product name",
        "group": "Product",
        "aliases": [
            "description", "product_name", "productname", "item_name",
            "product", "item", "article", "good", "merchandise",
            "product_title", "item_title",
            "اسم_المنتج", "المنتج", "صنف", "وصف_المنتج",
        ],
    },
    "product.price": {
        "human_label": "Selling price",
        "group": "Product",
        "aliases": [
            "unitprice", "unit_price", "price", "selling_price", "sale_price",
            "retail_price", "list_price", "base_price", "price_per_unit",
            "item_price", "product_price", "prices_list",
            "سعر", "السعر", "سعر_البيع", "سعر_الوحدة",
        ],
    },
    "product.cost": {
        "human_label": "Cost price",
        "group": "Product",
        "aliases": [
            "cost", "unit_cost", "unit_cost_egp", "cost_egp", "cogs", "purchase_price", "cost_price",
            "cost_of_goods", "buying_price", "supplier_price",
            "wholesale_price", "avg_cost", "calculated_cost",
            "تكلفة", "التكلفة", "سعر_التكلفة",
        ],
    },
    "product.stock": {
        "human_label": "Stock / inventory",
        "group": "Product",
        "aliases": [
            "stock", "inventory", "stock_level", "on_hand",
            "available_qty", "stock_quantity", "current_stock",
            "units_in_stock", "units_available",
            "مخزون", "المخزون",
        ],
    },
    "product.description": {
        "human_label": "Product description",
        "group": "Product",
        "aliases": [
            "product_description", "item_description",
            "details", "product_details", "product_notes",
            "وصف", "وصف_المنتج",
        ],
    },

    # ── Order ─────────────────────────────────────────────────────────────────
    "order_id": {
        "human_label": "Order / invoice number",
        "group": "Order",
        "aliases": [
            "invoiceno", "invoiceno", "invoice_no", "invoice_number", "order_id",
            "order_no", "order_number", "transaction_id", "sale_id",
            "receipt_id", "receipt_no", "confirmation_id",
            "reference_id", "purchase_order_id", "po_number",
            "رقم_الطلب", "معرف_الطلب", "رقم_الفاتورة",
        ],
    },
    "order_date": {
        "human_label": "Order date",
        "group": "Order",
        "aliases": [
            "invoicedate", "invoice_date", "order_date", "date",
            "transaction_date", "purchase_date", "sale_date",
            "order_time", "order_datetime", "created_at",
            "تاريخ_الطلب", "تاريخ", "تاريخ_البيع",
        ],
    },
    "order_item.quantity": {
        "human_label": "Quantity sold",
        "group": "Order",
        "aliases": [
            "quantity", "qty", "units", "amount", "count",
            "num_items", "number_of_units", "ordered_qty",
            "sold_qty", "units_sold", "pieces", "pcs",
            "كمية", "الكمية", "عدد", "وحدات",
        ],
    },
    "order.status": {
        "human_label": "Order status",
        "group": "Order",
        "aliases": [
            "status", "order_status", "orderstatus",
            "fulfillment_status", "delivery_status", "payment_status",
            "state", "order_state",
            "حالة_الطلب", "حالة",
        ],
    },
    "order.address": {
        "human_label": "Delivery address",
        "group": "Order",
        "aliases": [
            "address", "shipping_address", "delivery_address",
            "ship_to", "deliver_to", "full_address",
            "عنوان", "عنوان_الشحن",
        ],
    },

    # ── ML computed ───────────────────────────────────────────────────────────
    "revenue": {
        "human_label": "Total revenue / sales amount",
        "group": "Order",
        "aliases": [
            "revenue", "total", "total_amount", "totalprice",
            "sales", "total_sales", "grand_total", "subtotal",
            "order_total", "invoice_total", "payment_amount",
            "إيرادات", "المبيعات", "مبيعات",
        ],
    },
    "profit": {
        "human_label": "Profit",
        "group": "Order",
        "aliases": [
            "profit", "net_profit", "gross_profit", "earnings",
            "net_income", "net_gain",
            "ربح", "الربح", "صافي_الربح",
        ],
    },
    "profit_margin": {
        "human_label": "Profit margin %",
        "group": "Order",
        "aliases": [
            "profit_margin", "gross_margin", "net_margin",
            "margin", "margin_pct", "margin_percent",
            "هامش_الربح", "هامش_ربح",
        ],
    },
}

# ── Attribute fields (clothing/variant data → orderItem.attributes) ───────────
ATTRIBUTE_REGISTRY: dict[str, dict] = {
    "size": {
        "human_label": "Size",
        "aliases": ["size", "sizes", "size_code", "clothing_size", "apparel_size", "fit", "size_fit",
                    "مقاس", "المقاس", "قياس"],
    },
    "color": {
        "human_label": "Colour",
        "aliases": ["color", "colour", "colors", "shade", "colour_shade", "color_shade", "item_color",
                    "لون", "اللون"],
    },
    "gender": {
        "human_label": "Gender / audience",
        "aliases": ["gender", "target_gender", "mens", "womens", "unisex",
                    "جنس", "رجالي", "حريمي"],
    },
    "season": {
        "human_label": "Season",
        "aliases": ["season", "seasons", "seasonal", "collection_season",
                    "موسم", "الموسم"],
    },
    "collection": {
        "human_label": "Collection / line",
        "aliases": ["collection", "line", "product_line", "series", "range",
                    "مجموعة", "الكولكشن"],
    },
}

# Columns that must never be fuzzy-matched (geographic / irrelevant)
FUZZY_DENYLIST = {
    "country", "city", "region", "state", "province",
    "continent", "area", "zone", "locale", "location",
    "nationality", "currency",
}

# Arabic transliteration table
ARABIC_TRANSLIT: dict[str, str] = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ب": "b", "ت": "t",
    "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w",
    "ي": "y", "ى": "a", "ة": "a", "ء": "", "ئ": "y", "ؤ": "w",
    "\u064b": "", "\u064c": "", "\u064d": "", "\u064e": "",
    "\u064f": "", "\u0650": "", "\u0651": "", "\u0652": "", " ": "_",
}

FUZZY_THRESHOLD = 82

# --- Known metadata columns: unmapped but labelled nicely ---
METADATA_LABELS: dict[str, tuple[str, str]] = {
    "invoiceno":      ("Invoice number (duplicate of order ID)", "Notes"),
    "invoice_no":     ("Invoice number (duplicate of order ID)", "Notes"),
    "payment_method": ("Payment method",        "Notes"),
    "pay_method":     ("Payment method",        "Notes"),
    "payment":        ("Payment method",        "Notes"),
    "payment_type":   ("Payment method",        "Notes"),
    "order_source":   ("Sales channel",         "Notes"),
    "channel":        ("Sales channel",         "Notes"),
    "source":         ("Sales channel",         "Notes"),
    "sales_rep":      ("Sales representative",  "Notes"),
    "rep":            ("Sales representative",  "Notes"),
    "agent":          ("Sales representative",  "Notes"),
    "return_reason":  ("Return / refund reason","Notes"),
    "refund_reason":  ("Return / refund reason","Notes"),
    "promo_disc":     ("Promotional discount",  "Notes"),
    "voucher_code":   ("Voucher / coupon code", "Notes"),
    "coupon":         ("Voucher / coupon code", "Notes"),
    "promo_code":     ("Voucher / coupon code", "Notes"),
    "category":       ("Product category",      "Notes"),
    "category_name":  ("Product category",      "Notes"),
    "sub_category":   ("Product sub-category",  "Notes"),
    "type":           ("Product type",          "Notes"),
    "comments":       ("Order notes",           "Notes"),
    "remarks":        ("Order notes",           "Notes"),
    "notes":          ("Order notes",           "Notes"),
    "city":           ("City / area",           "Notes"),
    "city_area":      ("City / area",           "Notes"),
    "region":         ("Region",                "Notes"),
}



# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _transliterate(text: str) -> str:
    return "".join(ARABIC_TRANSLIT.get(c, c) for c in text)


def normalize(col: str) -> str:
    col = str(col).strip()
    col = unicodedata.normalize("NFC", col).lower()
    col = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "_", col)
    return col.strip("_")


def _detect_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    max_cols = max(df_raw.iloc[i].count() for i in range(min(max_scan, len(df_raw))))
    best_row, best_score = 0, -999.0
    for i in range(min(max_scan, len(df_raw))):
        row   = df_raw.iloc[i]
        score = 0.0
        if row.count() < max_cols * 0.5:
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
            best_score, best_row = score, i
    return best_row


def _sample_values(series: pd.Series, n: int = 3) -> list[str]:
    """Return up to n non-null, non-empty string samples from a column."""
    seen: list[str] = []
    for v in series.dropna():
        s = str(v).strip()
        if s and s not in seen:
            seen.append(s)
        if len(seen) >= n:
            break
    return seen


# ═════════════════════════════════════════════════════════════════════════════
# MAPPING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _build_lookups() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """
    Returns:
      alias_to_target  — normalized alias → target key
      alias_to_attr    — normalized alias → attribute key
      target_to_label  — target key → human label
      target_to_group  — target key → group name
    """
    alias_to_target: dict[str, str] = {}
    target_to_label: dict[str, str] = {}
    target_to_group: dict[str, str] = {}

    for target, cfg in FIELD_REGISTRY.items():
        target_to_label[target] = cfg["human_label"]
        target_to_group[target] = cfg["group"]
        for alias in cfg["aliases"]:
            alias_to_target[normalize(alias)] = target

    alias_to_attr: dict[str, str] = {}
    for attr_key, cfg in ATTRIBUTE_REGISTRY.items():
        for alias in cfg["aliases"]:
            alias_to_attr[normalize(alias)] = attr_key

    return alias_to_target, alias_to_attr, target_to_label, target_to_group



# =============================================================================
# ADAPTIVE SAMPLING
# =============================================================================

SAMPLE_MAX = 1_000   # hard cap on rows analysed per column
SAMPLE_PCT = 0.10    # use 10% of dataset

def _get_sample(df):
    n = len(df)
    if n <= 50:          # tiny dataset: use everything
        return df
    size = min(SAMPLE_MAX, max(50, int(n * SAMPLE_PCT)))
    return df if n <= size else df.sample(n=size, random_state=42)


# =============================================================================
# SIGNAL 2 — DATA TYPE & SHAPE
# =============================================================================

_DATE_RE   = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
_EMAIL_RE  = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE  = re.compile(r"^[+\d][\d\s\-().]{6,18}$")
_ID_RE     = re.compile(r"^[A-Z]{0,4}[\-_]?\d{3,12}$|^[A-Z]{1,4}\d{2,8}$", re.IGNORECASE)
_SKU_RE    = re.compile(r"^[A-Z0-9]{3,6}[\-_]?[A-Z0-9]{0,6}$", re.IGNORECASE)
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _type_signal(series):
    scores = {}
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return scores

    total    = len(series)
    null_pct = 1 - (n / total)

    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    num_pct = len(numeric) / n

    if num_pct >= 0.85:
        mn, mx, med = float(numeric.min()), float(numeric.max()), float(numeric.median())
        unique_pct  = numeric.nunique() / n
        all_int     = bool((numeric == numeric.round()).all())

        if all_int and mn >= 0 and mx <= 50_000 and med <= 200 and unique_pct < 0.5:
            scores["order_item.quantity"] = 80
        if mx > 0 and mn >= 0 and mx <= 1_000_000:
            ps = 65 if 0.01 <= med <= 10_000 else 0
            if unique_pct > 0.3:
                ps += 10
            # Boost for columns with decimal values (clearly monetary)
            has_decimals = not all_int and bool((numeric % 1 != 0).any())
            if has_decimals and 0.01 <= med <= 10_000:
                ps = max(ps, 80)
            if ps:
                scores["product.price"] = ps
        if med > 1 and mx > 100:
            scores["revenue"] = 65
        # Strong revenue signal: values span a wide range (totals vary more than unit prices)
        if med > 1 and mx > 10 and (mx / max(med, 0.01)) > 2.0:
            scores["revenue"] = max(scores.get("revenue", 0), 75)
        if mn >= 0 and mx <= 100 and med >= 5 and med <= 80:
            scores["profit_margin"] = 60
        if mn >= 0 and mx < 100_000:
            scores["profit"] = 55
        if all_int and mn >= 0:
            scores["product.stock"] = 60
        if all_int and unique_pct > 0.7 and mn > 0:
            scores["customer_id"] = 70
            scores["order_id"]    = 68

    str_vals = non_null.astype(str).str.strip()
    unique_n = str_vals.nunique()
    uq_str   = unique_n / n

    email_m = str_vals.apply(lambda v: bool(_EMAIL_RE.match(v))).mean()
    if email_m > 0.7:
        scores["customer.email"] = min(100, 70 + email_m * 30)

    # Phone: must have non-digit chars (+, spaces, dashes) or start with country code
    phone_m = str_vals.apply(
        lambda v: bool(_PHONE_RE.match(v)) and bool(re.search(r"[+\-\s()]", v))
    ).mean()
    if phone_m > 0.6:
        scores["customer.phoneNumber"] = min(100, 60 + phone_m * 40)

    date_m = str_vals.apply(lambda v: bool(_DATE_RE.match(v))).mean()
    if date_m > 0.7:
        scores["order_date"] = min(100, 70 + date_m * 30)
    if date_m < 0.7:
        try:
            parsed = pd.to_datetime(non_null.head(30), errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                scores["order_date"] = 75
                date_m = 0.8  # mark as date-like
        except Exception:
            pass
    # If column is date-like, suppress all other signals — date is unambiguous
    if scores.get("order_date", 0) >= 75:
        date_score = scores["order_date"]
        scores.clear()
        scores["order_date"] = date_score

    id_m  = str_vals.apply(lambda v: bool(_ID_RE.match(v))).mean()
    sku_m = str_vals.apply(lambda v: bool(_SKU_RE.match(v))).mean()
    if id_m > 0.5 and uq_str > 0.3:  # lower uniqueness threshold — IDs can repeat
        scores["order_id"]    = 72
        scores["customer_id"] = 70
    if sku_m > 0.5 and uq_str > 0.2:
        scores["product_id"]  = 75

    avg_len = str_vals.str.len().mean()
    if uq_str > 0.8 and avg_len < 40:
        scores["customer.fullName"] = 60
        scores["customer_id"]       = max(scores.get("customer_id", 0), 58)
    if uq_str > 0.3 and 5 < avg_len < 80:
        scores["product.name"]        = 65
        scores["product.description"] = 55
    # Low cardinality alone is not enough for status/segment
    # Pattern signal handles those based on actual value content

    return scores


# =============================================================================
# SIGNAL 3 — VALUE CONTENT PATTERNS
# =============================================================================

_STATUS_VALUES = {
    "shipped","delivered","pending","cancelled","canceled",
    "completed","processing","returned","refunded","failed",
}


def _pattern_signal(series):
    scores   = {}
    non_null = series.dropna()
    if len(non_null) == 0:
        return scores

    str_vals = non_null.astype(str).str.strip()

    status_pct = str_vals.str.lower().apply(lambda v: v in _STATUS_VALUES).mean()
    if status_pct > 0.5:
        scores["order.status"] = min(100, 70 + status_pct * 30)
    # Do NOT fire status signal if no actual status words found
    # (prevents C001/C002 style IDs from being classified as statuses)

    curr_pct = str_vals.apply(
        lambda v: bool(re.search(r"[£$€¥₹﷼EGP]", v, re.IGNORECASE))
    ).mean()
    if curr_pct > 0.3:
        scores["product.price"] = max(scores.get("product.price", 0), 72)
        scores["revenue"]       = max(scores.get("revenue", 0), 68)

    inv_pct = str_vals.apply(
        lambda v: bool(re.match(r"^(C?\d{4,8}|INV|ORD|REC|TXN)[\-_]?\d*$", v, re.IGNORECASE))
    ).mean()
    if inv_pct > 0.4:
        scores["order_id"] = max(scores.get("order_id", 0), 80)

    name_pct = str_vals.apply(
        lambda v: bool(re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", v)) or
                  bool(_ARABIC_RE.search(v) and len(v.split()) >= 2)
    ).mean()
    if name_pct > 0.5:
        scores["customer.fullName"] = max(scores.get("customer.fullName", 0), 78)

    prod_pct = str_vals.apply(
        lambda v: bool(re.match(r"^[A-Z][A-Za-z\s\-]{3,60}$", v)) and not v.isnumeric()
    ).mean()
    if prod_pct > 0.4:
        scores["product.name"] = max(scores.get("product.name", 0), 72)

    addr_pct = str_vals.apply(
        lambda v: bool(re.search(
            r"\d+.{3,}(st|street|rd|ave|blvd|road|شارع)", v, re.IGNORECASE
        ))
    ).mean()
    if addr_pct > 0.3:
        scores["order.address"] = 80

    return scores


# =============================================================================
# SIGNAL 4 — CO-OCCURRENCE (A × B ≈ C → C is revenue)
# =============================================================================

def _cooccurrence_signal(df_sample, columns):
    bonuses = {c: {} for c in columns}

    num_cols = []
    for col in columns:
        try:
            s = pd.to_numeric(df_sample[col], errors="coerce").dropna()
            if len(s) > 10 and s.min() >= 0:
                num_cols.append((col, s))
        except Exception:
            pass

    if len(num_cols) < 3:
        return bonuses

    for i, (ca, a) in enumerate(num_cols):
        for j, (cb, b) in enumerate(num_cols):
            if j <= i:
                continue
            for k, (cc, c) in enumerate(num_cols):
                if k in (i, j):
                    continue
                common = a.index.intersection(b.index).intersection(c.index)
                if len(common) < 5:
                    continue
                product = (a.loc[common] * b.loc[common]).round(2)
                target  = c.loc[common].round(2)
                ratio   = (product / target.replace(0, float("nan"))).dropna()
                if len(ratio) < 5:
                    continue
                if 0.90 <= float(ratio.median()) <= 1.10 and float(ratio.std()) < 0.30:
                    bonuses[cc]["revenue"] = max(bonuses[cc].get("revenue", 0), 50)  # strong signal
                    # Higher median = price, lower median = quantity
                    if float(a.median()) < float(b.median()):
                        bonuses[ca]["order_item.quantity"] = max(bonuses[ca].get("order_item.quantity", 0), 40)
                        bonuses[cb]["product.price"]       = max(bonuses[cb].get("product.price", 0), 40)
                    else:
                        bonuses[cb]["order_item.quantity"] = max(bonuses[cb].get("order_item.quantity", 0), 40)
                        bonuses[ca]["product.price"]       = max(bonuses[ca].get("product.price", 0), 40)
    return bonuses


# =============================================================================
# COMBINED SCORE — weights: name 40%, type 30%, pattern 20%, co-occ 10%
# =============================================================================

W_NAME = 0.40; W_TYPE = 0.30; W_PATTERN = 0.20; W_COOC = 0.10
MIN_COMBINED = 18.0  # data signals alone can score 20-30 without name match


def _combine(name_score, type_scores, pattern_scores, cooc_bonus, target):
    cb = min(100, cooc_bonus.get(target, 0) * 3)
    return round(
        name_score                        * W_NAME +
        type_scores.get(target,    0)    * W_TYPE +
        pattern_scores.get(target, 0)    * W_PATTERN +
        cb                                * W_COOC,
        1,
    )


def map_columns(
    columns: list[str],
    df: pd.DataFrame,
) -> tuple[list[ColumnMapping], list[str]]:
    alias_to_target, alias_to_attr, target_to_label, target_to_group = _build_lookups()
    schema_aliases = list(alias_to_target.keys())
    attr_aliases   = list(alias_to_attr.keys())

    warnings:       list[str] = []
    arabic_detected           = False

    # Sample once for all data-signal analysis
    df_sample = _get_sample(df)
    n_total, n_sample = len(df), len(df_sample)
    if n_sample < n_total:
        warnings.append(
            f"Column analysis used a {n_sample:,}-row sample "
            f"({int(n_sample/n_total*100)}% of {n_total:,} rows) for speed."
        )

    # Pre-compute signals 2, 3, 4 for every column
    type_sig:    dict[str, dict] = {}
    pattern_sig: dict[str, dict] = {}
    for col in columns:
        try:
            type_sig[col]    = _type_signal(df_sample[col])
            pattern_sig[col] = _pattern_signal(df_sample[col])
        except Exception:
            type_sig[col]    = {}
            pattern_sig[col] = {}
    cooc_sig = _cooccurrence_signal(df_sample, columns)

    # Score every column
    # (col, target, combined, name_score, method, label, group, samples)
    candidates = []

    for col in columns:
        norm      = normalize(col)
        is_arabic = _has_arabic(col)
        if is_arabic:
            arabic_detected = True

        samples = _sample_values(df[col])

        # Geographic denylist
        if norm in FUZZY_DENYLIST:
            _ml = METADATA_LABELS.get(norm, (col, "Notes"))
            candidates.append((col, "__unmapped__", 0.0, 0.0, "unmatched", _ml[0], _ml[1], samples))
            continue

        # Signal 1: name matching
        name_score  = 0.0
        name_target = None
        name_method = "unmatched"
        name_label  = col
        name_group  = "Notes"

        if norm in alias_to_target:
            name_target = alias_to_target[norm]
            name_score  = 100.0
            name_method = "exact"
        elif norm in alias_to_attr:
            ak          = alias_to_attr[norm]
            name_target = f"__attributes__.{ak}"
            name_score  = 100.0
            name_method = "exact"
            name_label  = ATTRIBUTE_REGISTRY[ak]["human_label"]
            name_group  = "Product"
        else:
            lookup = normalize(_transliterate(col)) if is_arabic else norm
            res = process.extractOne(
                lookup, schema_aliases,
                scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD,
            )
            if res:
                best_alias, sc, _ = res
                name_target = alias_to_target[best_alias]
                name_score  = float(sc)
                name_method = "arabic_transliterated" if is_arabic else "fuzzy"
            else:
                attr_res = process.extractOne(
                    lookup, attr_aliases,
                    scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD,
                )
                if attr_res:
                    best_alias, sc, _ = attr_res
                    ak          = alias_to_attr[best_alias]
                    name_target = f"__attributes__.{ak}"
                    name_score  = float(sc)
                    name_method = "arabic_transliterated" if is_arabic else "fuzzy"
                    name_label  = ATTRIBUTE_REGISTRY[ak]["human_label"]
                    name_group  = "Product"

        if name_target and not name_target.startswith("__attributes__"):
            name_label = target_to_label.get(name_target, col)
            name_group = target_to_group.get(name_target, "Notes")

        # Collect all targets suggested by signals 2+3 (and name)
        data_targets: set[str] = set(type_sig[col]) | set(pattern_sig[col])
        if name_target:
            data_targets.add(name_target)  # always include, even attr targets

        # Attribute exact matches: lock score so no data signal can override
        if name_target and name_target.startswith("__attributes__") and name_score == 100.0:
            best_combined = 100.0
            best_target   = name_target
            best_method   = name_method
            best_label    = name_label
            best_group    = name_group
        else:
            # Find best target by combined score
            best_combined = 0.0
            best_target   = name_target or "__unmapped__"
            best_method   = name_method
            best_label    = name_label
            best_group    = name_group

        for t in data_targets:
            ns       = name_score if t == name_target else 0.0
            combined = _combine(ns, type_sig[col], pattern_sig[col], cooc_sig.get(col, {}), t)
            if combined > best_combined:
                best_combined = combined
                best_target   = t
                if t != name_target:
                    best_method = "data_signal" if ns == 0 else best_method

        # Update label/group after loop based on final best_target
        if not best_target.startswith("__attributes__") and best_target != "__unmapped__":
            best_label = target_to_label.get(best_target, col)
            best_group = target_to_group.get(best_target, "Notes")

        # Dynamic threshold based on which signals fired
        # Pattern signal is highly specific (email, phone, date, invoice patterns)
        # so pattern-driven matches can pass at a lower threshold
        has_pattern  = bool(pattern_sig.get(col))
        has_type     = bool(type_sig.get(col))
        if name_score == 100.0:
            threshold = 0      # exact name match: always accept
        elif has_pattern and not has_type:
            threshold = 12     # pattern-only: relatively specific, lower bar
        elif has_pattern and has_type:
            threshold = 16     # both signals: good confidence
        else:
            threshold = 18     # type-only or neither: require more

        if best_combined < threshold and best_target != "__unmapped__":
            best_target   = "__unmapped__"
            best_combined = 0.0
        # Apply friendly label for known metadata columns
        if best_target == "__unmapped__" and norm in METADATA_LABELS:
            best_label  = METADATA_LABELS[norm][0]
            best_group  = METADATA_LABELS[norm][1]
            best_method = "unmatched"
        candidates.append((
            col, best_target, best_combined, name_score,
            best_method, best_label, best_group, samples,
        ))

    if arabic_detected:
        warnings.append("Arabic column headers detected — transliteration applied.")

    # Conflict resolution: highest combined score claims each target
    candidates.sort(key=lambda x: x[2], reverse=True)
    used: set[str] = set()
    mappings: list[ColumnMapping] = []

    for col, target, combined, name_sc, method, label, group, samples in candidates:
        if target == "__unmapped__":
            mappings.append(ColumnMapping(
                original_column=col, mapped_to="__unmapped__",
                confidence=0.0, match_method="unmatched",
                human_label=label, group=group, sample_values=samples,
            ))
            continue
        if target in used:
            warnings.append(
                f"'{col}' matched '{label}' but that field was already claimed — moved to Notes."
            )
            _ndm = normalize(col)
            _dlm = METADATA_LABELS.get(_ndm, (col, 'Notes'))
            mappings.append(ColumnMapping(
                original_column=col, mapped_to='__unmapped__',
                confidence=combined, match_method=method,
                human_label=_dlm[0], group=_dlm[1], sample_values=samples,
            ))
        else:
            used.add(target)
            # Exact alias matches always display as 100% confidence
            display_conf = 100.0 if name_sc == 100.0 else combined
            mappings.append(ColumnMapping(
                original_column=col, mapped_to=target,
                confidence=display_conf, match_method=method,
                human_label=label, group=group, sample_values=samples,
            ))

    return mappings, warnings


# ═════════════════════════════════════════════════════════════════════════════
# FAST FILE LOADER — separate strategies for parse vs clean
# ═════════════════════════════════════════════════════════════════════════════

def _row_count_csv(file_bytes: bytes) -> int:
    """Count rows in a CSV by counting newlines. O(n) but very fast in Python."""
    return file_bytes.count(b"\n") - 1  # subtract header line


def _load_for_parse(
    file_bytes: bytes,
    filename:   str,
    header_row: int,
    max_rows:   int = 1_000,
) -> tuple[pd.DataFrame, int]:
    """
    Load only what /parse needs: columns + a small sample for signal analysis.
    Returns (sample_df, estimated_total_rows).

    Strategy:
    - CSV  : read nrows=max_rows (near-instant), count total rows separately
    - xlsx : use calamine engine (9x faster than openpyxl), still reads full file
             because calamine ignores nrows — but returns sample slice only
    """
    if filename.endswith(".csv"):
        # Try encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                sample = pd.read_csv(
                    io.BytesIO(file_bytes),
                    header=header_row,
                    nrows=max_rows,
                    encoding=enc,
                    low_memory=False,
                )
                total = _row_count_csv(file_bytes)
                return sample, total
            except Exception:
                continue
        raise ValueError("Could not decode CSV file.")

    if filename.endswith((".xlsx", ".xls")):
        # calamine is ~9x faster than openpyxl for large files
        try:
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                header=header_row,
                engine="calamine",
            )
        except Exception:
            # Fallback to openpyxl if calamine unavailable
            df = pd.read_excel(io.BytesIO(file_bytes), header=header_row)

        total  = len(df)
        sample = df.head(max_rows)
        return sample, total

    raise ValueError(f"Unsupported file format: \'{filename}\'.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY — parse only (fast, sample-based)
# ═════════════════════════════════════════════════════════════════════════════

async def parse_file(file_bytes: bytes, filename: str) -> ParseResponse:
    """
    Parse a file and return column mappings.

    Speed profile (350k row xlsx):
      Old: read full file with openpyxl      → 216s
      New: read 1000-row sample with calamine →  14s  (xlsx)
           read 1000-row sample from CSV      →  0.15s (csv)

    The sample is sufficient for all 4 mapping signals:
    data type, value patterns, co-occurrence, and name matching.
    """
    warnings: list[str] = []

    # ── Detect header row from first few raw rows ──────────────────────────────
    if filename.endswith(".csv"):
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df_raw = pd.read_csv(
                    io.BytesIO(file_bytes), header=None, nrows=15,
                    encoding=enc, on_bad_lines="skip", engine="python",
                )
                break
            except Exception:
                continue
        else:
            raise ValueError("Could not decode CSV file.")
    else:
        try:
            df_raw = pd.read_excel(
                io.BytesIO(file_bytes), header=None, nrows=15, engine="calamine"
            )
        except Exception:
            df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None, nrows=15)

    header_idx = _detect_header_row(df_raw)
    if header_idx > 0:
        warnings.append(
            f"Data starts at row {header_idx + 1} — "
            f"{header_idx} header row(s) skipped."
        )

    # ── Load sample + get total row count ─────────────────────────────────────
    df, total_rows = _load_for_parse(file_bytes, filename, header_idx)

    original_columns = [str(c) for c in df.columns.tolist()]
    total_columns    = len(original_columns)

    # ── Map columns (all 4 signals run on the sample) ─────────────────────────
    mappings, map_warnings = map_columns(original_columns, df)
    warnings.extend(map_warnings)

    # ── Build flat response dicts ──────────────────────────────────────────────
    detected_mapping:  dict[str, str]         = {}
    confidence_scores: dict[str, float]       = {}
    match_methods:     dict[str, str]         = {}
    human_labels:      dict[str, str]         = {}
    groups:            dict[str, str]         = {}
    sample_values_map: dict[str, list[str]]   = {}
    attribute_columns: dict[str, str]         = {}
    unmapped_columns:  list[str]              = []

    for m in mappings:
        sample_values_map[m.original_column] = m.sample_values
        if m.mapped_to == "__unmapped__":
            unmapped_columns.append(m.original_column)
            human_labels[m.original_column] = m.human_label  # use METADATA_LABELS label
            groups[m.original_column]       = m.group
        elif m.mapped_to.startswith("__attributes__."):
            attr_key = m.mapped_to.split(".", 1)[1]
            attribute_columns[m.original_column] = attr_key
            detected_mapping[m.original_column]  = m.mapped_to
            confidence_scores[m.original_column] = m.confidence
            match_methods[m.original_column]     = m.match_method
            human_labels[m.original_column]      = m.human_label
            groups[m.original_column]            = "Product"
        else:
            detected_mapping[m.original_column]  = m.mapped_to
            confidence_scores[m.original_column] = m.confidence
            match_methods[m.original_column]     = m.match_method
            human_labels[m.original_column]      = m.human_label
            groups[m.original_column]            = m.group

    # ── ML coverage ────────────────────────────────────────────────────────────
    mapped_targets = set(detected_mapping.values())

    def to_ml_key(target: str) -> str | None:
        if target in ML_REQUIRED_KEYS:
            return target
        for k, cfg in ML_FIELDS.items():
            if cfg.get("table") and f"{cfg['table']}.{cfg['field']}" == target:
                return k
        return None

    covered: set[str] = set()
    for t in mapped_targets:
        k = to_ml_key(t)
        if k:
            covered.add(k)

    ml_missing = [k for k in ML_REQUIRED_KEYS if k not in covered]

    # ── Derivation ─────────────────────────────────────────────────────────────
    derivation_rules = get_derivation_rules()
    derivable: list[DerivableField] = []
    truly_missing = list(ml_missing)
    target_to_col = {v: k for k, v in detected_mapping.items()}

    for _ in range(6):
        resolved = False
        for field, formula, requires in derivation_rules:
            if field not in truly_missing:
                continue
            available = covered | {d.field for d in derivable}
            if all(r in available for r in requires):
                src = [target_to_col.get(r, f"[derived:{r}]") for r in requires]
                derivable.append(DerivableField(
                    field=field, formula=formula,
                    requires=requires, source_columns=src,
                ))
                truly_missing.remove(field)
                resolved = True
        if not resolved:
            break

    effective    = len(ML_REQUIRED_KEYS) - len(truly_missing)
    coverage_pct = round(effective / max(len(ML_REQUIRED_KEYS), 1) * 100, 1)

    if derivable:
        warnings.append(
            f"These analytics fields will be calculated automatically: "
            f"{', '.join(d.field for d in derivable)}."
        )

    return ParseResponse(
        header_row_index  = header_idx,
        detected_mapping  = detected_mapping,
        confidence_scores = confidence_scores,
        match_methods     = match_methods,
        human_labels      = human_labels,
        groups            = groups,
        sample_values     = sample_values_map,
        ml_required       = ML_REQUIRED_KEYS,
        ml_missing        = ml_missing,
        ml_coverage_pct   = coverage_pct,
        derivable_fields  = derivable,
        truly_missing     = truly_missing,
        attribute_columns = attribute_columns,
        unmapped_columns  = unmapped_columns,
        total_rows        = total_rows,
        total_columns     = total_columns,
        warnings          = warnings,
    )