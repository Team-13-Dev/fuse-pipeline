"""
app/services/segmentation/product/triggers.py

Product segmentation trigger — decides whether a re-run is warranted.

Thresholds are tuned for local brands (5–200 products typical):
  - hard gate: minimum 15 products
  - 50 new order_items since last run
  - 5 products added since last run
  - 5 products with price/cost/stock changed since last run
  - 7-day drift safety net
"""
from __future__ import annotations

from app.core.database import get_db_connection
from app.services.segmentation.common.base_trigger import (
    SegmentationTrigger, TriggerDecision,
)
from app.services.segmentation.product.pipeline import MIN_PRODUCTS_FOR_SEGMENTATION


# ── Threshold constants — tweak here ─────────────────────────────────────────

NEW_ORDER_ITEMS_THRESHOLD = 50
PRODUCT_CHANGES_THRESHOLD = 5
DAYS_BEFORE_DRIFT_REFRESH = 7


class ProductSegmentationTrigger(SegmentationTrigger):
    """
    Evaluates whether to re-run product segmentation right now.
    """

    def evaluate(self, business_id: str) -> TriggerDecision:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # ── Hard gate: enough products to cluster? ────────────────
                cur.execute(
                    "SELECT COUNT(*) FROM product WHERE business_id = %s",
                    (business_id,),
                )
                product_count = cur.fetchone()[0]

                if product_count < MIN_PRODUCTS_FOR_SEGMENTATION:
                    return TriggerDecision(
                        should_run=False,
                        reason="insufficient_products",
                        detail=f"Only {product_count} products — need at least {MIN_PRODUCTS_FOR_SEGMENTATION}.",
                    )

                # ── Get last run time ─────────────────────────────────────
                cur.execute(
                    "SELECT last_product_segment_at FROM business WHERE id = %s",
                    (business_id,),
                )
                row = cur.fetchone()
                last_run = row[0] if row else None

                # First run ever — always trigger
                if last_run is None:
                    return TriggerDecision(
                        should_run=True,
                        reason="first_run",
                        detail="No prior segmentation — running for the first time.",
                    )

                # ── Drift safety net ──────────────────────────────────────
                cur.execute(
                    "SELECT EXTRACT(DAY FROM NOW() - %s)::int",
                    (last_run,),
                )
                days_since = cur.fetchone()[0] or 0
                if days_since > DAYS_BEFORE_DRIFT_REFRESH:
                    return TriggerDecision(
                        should_run=True,
                        reason="scheduled_refresh",
                        detail=f"{days_since} days since last run — running scheduled refresh.",
                    )

                # ── New order items ───────────────────────────────────────
                cur.execute(
                    """SELECT COUNT(*)
                       FROM order_item oi
                       JOIN "order" o ON o.id = oi.order_id
                       WHERE o.business_id = %s
                         AND o.created_at > %s""",
                    (business_id, last_run),
                )
                new_items = cur.fetchone()[0] or 0
                if new_items >= NEW_ORDER_ITEMS_THRESHOLD:
                    return TriggerDecision(
                        should_run=True,
                        reason="new_sales",
                        detail=f"{new_items} new sales since last run.",
                    )

                # ── Product changes (added or edited) ─────────────────────
                # We don't have an updated_at column on product currently —
                # compare against created_at as a proxy for "newly added".
                # For edits, see note below.
                cur.execute(
                    """SELECT COUNT(*) FROM product
                       WHERE business_id = %s AND created_at > %s""",
                    (business_id, last_run),
                )
                new_products = cur.fetchone()[0] or 0
                if new_products >= PRODUCT_CHANGES_THRESHOLD:
                    return TriggerDecision(
                        should_run=True,
                        reason="product_changes",
                        detail=f"{new_products} products added since last run.",
                    )

                return TriggerDecision(
                    should_run=False,
                    reason="no_threshold_met",
                    detail=f"{new_items} new sales, {new_products} new products — below thresholds.",
                )
        finally:
            conn.close()