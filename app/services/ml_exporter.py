"""
app/services/ml_exporter.py

Build an ML-ready Excel sheet from cleaned entities and upload it to
Supabase Storage in the `pipeline-exports` bucket.

Output columns (denormalized, one row per order item):
    order_id, price, cost, quantity, stock, profit_margin,
    revenue, customer_id, order_date

Uses raw httpx (matching app/core/storage.py style) — no supabase Python SDK.
"""
from __future__ import annotations

import io
import os
import time
import uuid
from typing import Any

import httpx
import openpyxl

EXPORT_BUCKET = "pipeline-exports"

SUPABASE_URL              = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]


def _log(msg: str) -> None:
    print(f"[ml_export] {msg}", flush=True)


def build_ml_dataset(
    customers:   list[dict],
    products:    list[dict],
    orders:      list[dict],
    order_items: list[dict],
) -> list[dict[str, Any]]:
    """Join entities in memory and return denormalized rows."""
    # Index products
    prod_by_acc:  dict[str, dict] = {}
    prod_by_name: dict[str, dict] = {}
    for p in products:
        if p.get("externalAccId"): prod_by_acc[str(p["externalAccId"])] = p
        if p.get("name"):          prod_by_name[str(p["name"])]         = p

    # Index orders by externalOrderId
    order_by_ext: dict[str, dict] = {}
    for o in orders:
        if o.get("externalOrderId"):
            order_by_ext[str(o["externalOrderId"])] = o

    rows: list[dict[str, Any]] = []
    for item in order_items:
        ext_oid = item.get("orderExternalId")
        order   = order_by_ext.get(str(ext_oid)) if ext_oid else None

        prod = None
        if item.get("productAccId"):
            prod = prod_by_acc.get(str(item["productAccId"]))
        if not prod and item.get("productName"):
            prod = prod_by_name.get(str(item["productName"]))

        rows.append({
            "order_id":       ext_oid,
            "price":          item.get("unitPrice"),
            "cost":           prod.get("cost")  if prod else None,
            "quantity":       item.get("quantity"),
            "stock":          prod.get("stock") if prod else None,
            "profit_margin":  order.get("profit_margin") if order else None,
            "revenue":        item.get("revenue"),
            "customer_id":    order.get("customerExternalId") if order else None,
            "order_date":     order.get("createdAt") if order else None,
        })

    return rows


def write_xlsx_to_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Stream-write rows to an in-memory XLSX file."""
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("ml_dataset")

    headers = [
        "order_id", "price", "cost", "quantity", "stock",
        "profit_margin", "revenue", "customer_id", "order_date",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _upload_to_supabase(path: str, file_bytes: bytes) -> None:
    """Upload bytes to Supabase Storage via raw HTTP — same pattern as storage.py."""
    url = f"{SUPABASE_URL}/storage/v1/object/{EXPORT_BUCKET}/{path}"
    headers = {
        "Authorization":  f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey":         SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type":   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "x-upsert":       "false",
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, content=file_bytes)

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Supabase upload failed {response.status_code}: {response.text[:300]}"
        )


async def export_ml_dataset(
    business_id: str,
    customers:   list[dict],
    products:    list[dict],
    orders:      list[dict],
    order_items: list[dict],
) -> dict[str, Any]:
    """
    Build ML dataset → write XLSX → upload to pipeline-exports bucket.
    Returns: { file_id, filename, row_count, size_bytes, bucket }
    """
    t0 = time.time()
    _log(f"building ML dataset from {len(order_items)} order items…")
    rows = build_ml_dataset(customers, products, orders, order_items)
    _log(f"prepared {len(rows)} rows in {time.time() - t0:.1f}s")

    t1 = time.time()
    xlsx_bytes = write_xlsx_to_bytes(rows)
    size_bytes = len(xlsx_bytes)
    _log(f"wrote xlsx ({size_bytes / 1024 / 1024:.2f} MB) in {time.time() - t1:.1f}s")

    t2 = time.time()
    timestamp = int(time.time() * 1000)
    rand_id   = uuid.uuid4().hex[:8]
    path      = f"{business_id}/{timestamp}-{rand_id}-ml_dataset.xlsx"

    _upload_to_supabase(path, xlsx_bytes)
    _log(f"uploaded to {EXPORT_BUCKET}/{path} in {time.time() - t2:.1f}s")

    return {
        "file_id":    path,
        "filename":   "ml_dataset.xlsx",
        "row_count":  len(rows),
        "size_bytes": size_bytes,
        "bucket":     EXPORT_BUCKET,
    }