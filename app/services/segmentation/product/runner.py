"""
app/services/segmentation/product/runner.py

Bridges the DB (Neon) and the pure ML pipeline.

  1. Read aggregated product+order data from Neon
  2. Build a DataFrame
  3. Run the ML pipeline
  4. Write product_segment + product_cluster_summary rows
  5. Update the analysis_job row with progress / status
  6. Stamp business.last_product_segment_at on success
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from typing import Any

import pandas as pd

from app.core.database import get_db_connection
from app.services.segmentation.common.job_tracker import (
    update_job, mark_done, mark_failed, mark_skipped,
)
from app.services.segmentation.product.pipeline import (
    run_product_segmentation, InsufficientDataError,
)
from app.services.segmentation.product.schemas import SegmentationResult


def _log(msg: str) -> None:
    print(f"[product_seg_runner] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load product DataFrame from Neon
# ─────────────────────────────────────────────────────────────────────────────

def _load_product_dataframe(business_id: str) -> pd.DataFrame:
    """
    Joins product + order_item to compute revenue/profit/quantity per product.
    Returns a DataFrame with the columns the ML pipeline needs.
    """
    sql = """
    SELECT
        p.id::text       AS product_id,
        p.price::numeric AS price,
        COALESCE(NULLIF(p.cost::numeric, 0), p.price::numeric * 0.5) AS cost,
        GREATEST(COALESCE(p.stock, 0), 1)::numeric AS stock,
        COALESCE(SUM(oi.quantity), 0)::numeric                                                AS quantity,
        COALESCE(SUM(oi.quantity * oi.unit_price), 0)::numeric                                AS revenue,
        COALESCE(SUM(oi.quantity * (oi.unit_price - COALESCE(NULLIF(p.cost::numeric, 0), p.price::numeric * 0.5))), 0)::numeric AS profit
    FROM product p
    LEFT JOIN order_item oi ON oi.product_id = p.id
    WHERE p.business_id = %s
    GROUP BY p.id, p.price, p.cost, p.stock
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (business_id,))
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    # Cast numerics
    for c in ["price", "cost", "stock", "quantity", "revenue", "profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive profit_margin as the model expects it
    df["profit_margin"] = df.apply(
        lambda r: float(r["profit"] / r["revenue"]) * 100
        if r["revenue"] and r["revenue"] > 0 else 0.0,
        axis=1,
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Persist results to Neon
# ─────────────────────────────────────────────────────────────────────────────

def _persist_results(
    business_id: str,
    job_id:      str,
    result:      SegmentationResult,
) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Replace product segments (delete-then-insert is simpler than upsert
            # for small datasets; will revisit if perf becomes an issue)
            cur.execute(
                "DELETE FROM product_segment WHERE business_id = %s",
                (business_id,),
            )

            for pl in result.product_labels:
                cur.execute(
                    """INSERT INTO product_segment
                       (business_id, product_id, cluster, cluster_name, job_id, updated_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())""",
                    (business_id, pl.product_id, pl.cluster, pl.cluster_name, job_id),
                )

            # Replace cluster summary rows (delete just the prior summary for this business)
            cur.execute(
                "DELETE FROM product_cluster_summary WHERE business_id = %s",
                (business_id,),
            )

            for cs in result.cluster_stats:
                cur.execute(
                    """INSERT INTO product_cluster_summary
                       (id, business_id, job_id, cluster, cluster_name, num_products,
                        avg_profit, total_profit, avg_revenue, total_revenue,
                        avg_price, avg_cost, avg_margin, avg_stock, avg_quantity,
                        revenue_share_pct, profit_share_pct, top_products, bottom_products)
                       VALUES (%s,%s,%s,%s,%s,%s,
                               %s,%s,%s,%s,
                               %s,%s,%s,%s,%s,
                               %s,%s,%s::jsonb,%s::jsonb)""",
                    (
                        str(uuid.uuid4()), business_id, job_id,
                        cs.cluster, cs.cluster_name, cs.num_products,
                        cs.avg_profit, cs.total_profit,
                        cs.avg_revenue, cs.total_revenue,
                        cs.avg_price, cs.avg_cost, cs.avg_margin, cs.avg_stock, cs.avg_quantity,
                        cs.revenue_share_pct, cs.profit_share_pct,
                        json.dumps(cs.top_products),
                        json.dumps(cs.bottom_products),
                    ),
                )

            # Stamp business.last_product_segment_at
            cur.execute(
                "UPDATE business SET last_product_segment_at = NOW() WHERE id = %s",
                (business_id,),
            )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Top-level runner — called as background task
# ─────────────────────────────────────────────────────────────────────────────

async def run_product_segmentation_job(business_id: str, job_id: str) -> None:
    """
    Async wrapper that runs the full job. Catches all exceptions and updates
    the job row accordingly. Never raises — fire-and-forget safe.
    """
    t0 = time.time()
    try:
        update_job(job_id, status="running", started=True, progress=10,
                   detail="Loading product data from database…")

        df = _load_product_dataframe(business_id)

        if df.empty:
            mark_skipped(job_id, "No products found in database.")
            return

        update_job(job_id, progress=40, detail="Running clustering models…")

        try:
            result = run_product_segmentation(df)
        except InsufficientDataError as exc:
            mark_skipped(job_id, str(exc))
            return

        update_job(job_id, progress=80, detail="Saving results…")

        _persist_results(business_id, job_id, result)

        elapsed = round(time.time() - t0, 1)
        mark_done(
            job_id,
            result_meta={
                "model_used":       result.model_used,
                "best_k":           result.best_k,
                "silhouette_score": result.silhouette_score,
                "total_rows":       result.total_rows,
                "elapsed_seconds":  elapsed,
            },
            detail=f"Found {result.best_k} segments using {result.model_used} (silhouette {result.silhouette_score})",
        )
        _log(f"job {job_id} done in {elapsed}s")

    except Exception as exc:
        traceback.print_exc()
        mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        _log(f"job {job_id} FAILED: {exc}")