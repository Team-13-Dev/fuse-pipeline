"""
Direct Postgres writer for fuse-pipeline.
Connects to Neon using DATABASE_URL env var.
Uses psycopg2 execute_values for efficient batching.
Column names match db/schema.ts exactly.
"""
from __future__ import annotations

import json
import os
import uuid

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ["DATABASE_URL"]
BATCH = 200  # safe for Neon's 65535 parameter limit


def _connect():
    return psycopg2.connect(DATABASE_URL, connect_timeout=30)


def _s(v) -> str | None:
    """Safe string — returns None for null/nan/empty."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _sanitize_attrs(attrs: dict | None) -> dict | None:
    if not attrs:
        return None
    safe = {k: str(v) for k, v in attrs.items() if v is not None}
    return safe or None


def store_entities(
    business_id: str,
    customers:   list[dict],
    products:    list[dict],
    orders:      list[dict],
    order_items: list[dict],
) -> dict:
    stats = {
        "customers_inserted": 0, "customers_skipped": 0,
        "products_inserted":  0, "products_skipped":  0,
        "orders_inserted":    0, "order_items_inserted": 0,
    }

    conn = _connect()
    try:
        with conn:  # single transaction — auto-commit on exit, rollback on exception
            with conn.cursor() as cur:

                # ── 1. Customers ──────────────────────────────────────────────
                clerk_map: dict[str, str] = {}   # clerk_id  → db uuid
                email_map: dict[str, str] = {}   # email     → db uuid
                name_map:  dict[str, str] = {}   # full_name → db uuid

                if customers:
                    # Load all existing customers for this business once
                    cur.execute(
                        "SELECT id, clerk_id, email, full_name "
                        "FROM customer WHERE business_id = %s",
                        (business_id,),
                    )
                    for row in cur.fetchall():
                        db_id = str(row[0])
                        if row[1]: clerk_map[row[1]] = db_id
                        if row[2]: email_map[row[2]] = db_id
                        if row[3]: name_map[row[3]]  = db_id
                        stats["customers_skipped"] += 1

                    new_custs: list[tuple] = []
                    for c in customers:
                        ck = _s(c.get("clerkId"))
                        em = _s(c.get("email"))
                        nm = _s(c.get("fullName")) or (f"Customer {ck}" if ck else "Unknown")
                        if (ck and ck in clerk_map) or (em and em in email_map):
                            continue
                        new_id = str(uuid.uuid4())
                        new_custs.append((
                            new_id, business_id, ck, nm, em,
                            _s(c.get("phoneNumber")),
                            _s(c.get("segment")),
                        ))
                        if ck: clerk_map[ck] = new_id
                        if em: email_map[em] = new_id
                        name_map[nm] = new_id

                    for i in range(0, len(new_custs), BATCH):
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO customer
                               (id, business_id, clerk_id, full_name, email,
                                phone_number, segment)
                               VALUES %s ON CONFLICT DO NOTHING""",
                            new_custs[i:i + BATCH],
                        )
                        stats["customers_inserted"] += cur.rowcount

                # ── 2. Products ───────────────────────────────────────────────
                acc_map:   dict[str, str] = {}   # external_acc_id → db uuid
                pname_map: dict[str, str] = {}   # name            → db uuid

                if products:
                    cur.execute(
                        "SELECT id, external_acc_id, name "
                        "FROM product WHERE business_id = %s",
                        (business_id,),
                    )
                    for row in cur.fetchall():
                        db_id = str(row[0])
                        if row[1]: acc_map[row[1]]   = db_id
                        if row[2]: pname_map[row[2]] = db_id
                        stats["products_skipped"] += 1

                    new_prods: list[tuple] = []
                    for p in products:
                        nm  = _s(p.get("name"))
                        acc = _s(p.get("externalAccId"))
                        if not nm:
                            continue
                        if (acc and acc in acc_map) or nm in pname_map:
                            continue
                        new_id = str(uuid.uuid4())
                        new_prods.append((
                            new_id, business_id, nm,
                            str(p.get("price") or "0"),
                            str(p.get("cost")) if p.get("cost") is not None else None,
                            int(p.get("stock") or 0),
                            _s(p.get("description")),
                            acc,
                        ))
                        if acc: acc_map[acc]  = new_id
                        pname_map[nm] = new_id

                    for i in range(0, len(new_prods), BATCH):
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO product
                               (id, business_id, name, price, cost,
                                stock, description, external_acc_id)
                               VALUES %s ON CONFLICT DO NOTHING""",
                            new_prods[i:i + BATCH],
                        )
                        stats["products_inserted"] += cur.rowcount

                # ── 3. Orders + items (always together — items need new order UUIDs) ──
                if orders:
                    # Index items by their externalOrderId for O(1) lookup
                    items_by_ext: dict[str, list[dict]] = {}
                    for item in order_items:
                        key = _s(item.get("orderExternalId")) or ""
                        items_by_ext.setdefault(key, []).append(item)

                    order_rows: list[tuple] = []   # (db cols...)
                    ext_id_to_db: dict[str, str] = {}  # externalOrderId → new db uuid

                    for o in orders:
                        ext_cust = _s(o.get("customerExternalId")) or ""
                        cust_db  = (
                            clerk_map.get(ext_cust)
                            or email_map.get(ext_cust)
                            or name_map.get(ext_cust)
                        )
                        if not cust_db:
                            continue  # can't link to a customer — skip

                        order_db_id  = str(uuid.uuid4())
                        ext_order_id = _s(o.get("externalOrderId")) or order_db_id
                        ext_id_to_db[ext_order_id] = order_db_id

                        order_rows.append((
                            order_db_id,
                            business_id,
                            cust_db,
                            str(o.get("total") or o.get("revenue") or "0"),
                            o.get("status") or "completed",
                            _s(o.get("orderVoucher")),
                            "0",  # order_discount
                            _s(o.get("address")),
                            _s(o.get("createdAt")),  # None → Postgres uses defaultNow()
                        ))

                    # Batch insert orders
                    for i in range(0, len(order_rows), BATCH):
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO "order"
                               (id, business_id, customer_id, total, status,
                                order_voucher, order_discount, address,
                                created_at)
                               VALUES %s ON CONFLICT DO NOTHING""",
                            [
                                (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                                 r[8])  # None is fine — column has defaultNow()
                                for r in order_rows[i:i + BATCH]
                            ],
                        )
                        stats["orders_inserted"] += cur.rowcount

                    # Build item rows using the in-memory ext_id_to_db map
                    item_rows: list[tuple] = []
                    for ext_oid, items in items_by_ext.items():
                        order_db_id = ext_id_to_db.get(ext_oid)
                        if not order_db_id:
                            continue
                        for item in items:
                            prod_db = (
                                acc_map.get(_s(item.get("productAccId")) or "")
                                or pname_map.get(_s(item.get("productName")) or "")
                            )
                            if not prod_db:
                                continue
                            attrs = _sanitize_attrs(item.get("attributes"))
                            item_rows.append((
                                str(uuid.uuid4()),
                                order_db_id,
                                prod_db,
                                int(item.get("quantity") or 1),
                                str(item.get("unitPrice") or "0"),
                                str(item.get("itemDiscount") or "0"),
                                json.dumps(attrs) if attrs else None,
                            ))

                    for i in range(0, len(item_rows), BATCH):
                        psycopg2.extras.execute_values(
                            cur,
                            """INSERT INTO order_item
                               (id, order_id, product_id, quantity,
                                unit_price, item_discount, attributes)
                               VALUES %s ON CONFLICT DO NOTHING""",
                            item_rows[i:i + BATCH],
                        )
                        stats["order_items_inserted"] += cur.rowcount

    finally:
        conn.close()

    return stats