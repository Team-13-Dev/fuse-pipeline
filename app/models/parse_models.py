from pydantic import BaseModel
from typing import Any


# ─── Per-column mapping result ────────────────────────────────────────────────

class ColumnMapping(BaseModel):
    original_column: str
    mapped_to:       str          # target key, "__attributes__.key", or "__unmapped__"
    confidence:      float        # 0–100
    match_method:    str          # "exact" | "fuzzy" | "data_signal" | "unmatched"
    human_label:     str          # plain-language label shown to the user
    group:           str          # "Customer" | "Product" | "Order" | "Notes"
    sample_values:   list[str] = []  # up to 3 real values from the column


# ─── Derivable field descriptor ──────────────────────────────────────────────

class DerivableField(BaseModel):
    field:          str          # ML field name (e.g. "revenue")
    formula:        str          # human formula (e.g. "price × quantity")
    requires:       list[str]    # required ML fields
    source_columns: list[str]    # original column names that supply them


# ─── Full parse response ──────────────────────────────────────────────────────

class ParseResponse(BaseModel):
    # Header detection
    header_row_index: int

    # Core mapping results
    detected_mapping:  dict[str, str]    # original_col → target
    confidence_scores: dict[str, float]  # original_col → score 0–100
    match_methods:     dict[str, str]    # original_col → method

    # Human-readable labels and grouping (for the mapping UI)
    human_labels: dict[str, str]         # original_col → plain-language label
    groups:       dict[str, str]         # original_col → group name
    sample_values: dict[str, list[str]]  # original_col → up to 3 sample values

    # ML coverage
    ml_required:     list[str]           # all required ML fields
    ml_missing:      list[str]           # required but not found
    ml_coverage_pct: float               # 0–100

    # Derivation engine
    derivable_fields: list[DerivableField]  # missing but calculable at clean time
    truly_missing:    list[str]             # missing AND cannot be derived

    # Clothing attributes (→ orderItem.attributes jsonb)
    attribute_columns: dict[str, str]    # original_col → attribute key (e.g. "size")

    # Columns that go to metadata
    unmapped_columns: list[str]

    # File stats
    total_rows:    int
    total_columns: int

    # Warnings (shown in UI)
    warnings: list[str]