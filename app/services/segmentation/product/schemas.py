"""
app/services/segmentation/product/schemas.py

Pydantic models for the product segmentation route interface.
"""
from pydantic import BaseModel
from typing import Any


# ── Route inputs ─────────────────────────────────────────────────────────────

class MaybeSegmentRequest(BaseModel):
    business_id:  str
    triggered_by: str   # 'auto:clean' | 'auto:threshold' | 'manual'


class ForceSegmentRequest(BaseModel):
    business_id:  str
    triggered_by: str = "manual"


# ── Route outputs ────────────────────────────────────────────────────────────

class MaybeSegmentResponse(BaseModel):
    job_id:   str | None      # None when skipped
    will_run: bool
    reason:   str
    detail:   str


# ── Internal pipeline result types (used by runner) ──────────────────────────

class ProductLabel(BaseModel):
    product_id:   str
    cluster:      int
    cluster_name: str


class ClusterStats(BaseModel):
    cluster:           int
    cluster_name:      str
    num_products:      int
    avg_profit:        float
    total_profit:      float
    avg_revenue:       float
    total_revenue:     float
    avg_price:         float
    avg_cost:          float
    avg_margin:        float
    avg_stock:         float
    avg_quantity:      float
    revenue_share_pct: float
    profit_share_pct:  float
    top_products:      list[dict[str, Any]]
    bottom_products:   list[dict[str, Any]]


class SegmentationResult(BaseModel):
    model_used:       str
    best_k:           int
    silhouette_score: float
    total_rows:       int
    product_labels:   list[ProductLabel]
    cluster_stats:    list[ClusterStats]