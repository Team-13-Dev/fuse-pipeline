from pydantic import BaseModel
from typing import Any


# ─── Per-column mapping result ────────────────────────────────────────────────

class ColumnMapping(BaseModel):
    original_column: str
    mapped_to: str          # schema field name, "__attributes__", or "__unmapped__"
    confidence: float       # fuzzy score 0–100
    match_method: str       # "exact", "fuzzy", "arabic_transliterated", "arabic_alias"


# ─── Full parse response ──────────────────────────────────────────────────────

class ParseResponse(BaseModel):
    # Header
    header_row_index: int                        # 0-based row index where header was detected

    # Mapping results
    detected_mapping: dict[str, str]             # original_col → schema_field
    confidence_scores: dict[str, float]          # original_col → score
    match_methods: dict[str, str]                # original_col → method

    # ML required columns coverage
    ml_required: list[str]                       # full list of required fields
    ml_missing: list[str]                        # required fields not found in file
    ml_coverage_pct: float                       # 0–100

    # Attribute columns (clothing-specific → orderItem.attributes jsonb)
    attribute_columns: dict[str, str]            # original_col → attribute key (e.g. "size", "color")

    # Unmapped
    unmapped_columns: list[str]                  # columns that couldn't be mapped

    # Preview
    sample_rows: list[dict[str, Any]]            # 5 raw rows (from detected header onward)

    # Warnings
    warnings: list[str]                          # e.g. "Arabic columns detected", "cost missing"