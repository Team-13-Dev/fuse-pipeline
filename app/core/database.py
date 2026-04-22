"""
Direct Postgres writer for fuse-pipeline.
- Separate connection per stage to survive Neon idle timeouts
- Explicit commit/rollback with full error logging
- Larger batches (500) to reduce round-trips on slow networks
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import traceback

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ["DATABASE_URL"]
BATCH = 500  # 500 rows × ~8 cols = 4000 params, well under Neon's 65535 limit


def _connect():
    """Open a new Neon connection with keepalive settings."""
    conn = psycopg2.connect(
        DATABASE_URL,
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    conn.autocommit = False
    return conn


def _s(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _sanitize_attrs(attrs: dict | None) -> dict | None:
    if not attrs:
        return None
    safe = {k: str(v) for k, v in attrs.items() if v is not None}
    return safe or None


def _log(msg: str) -> None:
    print(f"[store] {msg}", flush=True)


# ─── Stage runners — each opens its own connection ───────────────────────────

def _stage_customers(business_id: str, customers: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Returns (stats, clerk_map, email_map, name_map)."""
    stats = {"customers_inserted": 0, "customers_skipped": 0}
    clerk_map: dict[str, str] = {}
    email_map: dict[str, str] = {}
    name_map:  dict[str, str] = {}

    if not customers:
        return stats, clerk_map, email_map, name_map

    t0 = time.time()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, clerk_id, email, full_name FROM customer WHERE business_id = %s",
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
                new_custs.append((new_id, business_id, ck, nm, em,
                                  _s(c.get("phoneNumber")), _s(c.get("segment"))))
                if ck: clerk_map[ck] = new_id
                if em: email_map[em] = new_id
                name_map[nm] = new_id

            for i in range(0, len(new_custs), BATCH):
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO customer
                       (id, business_id, clerk_id, full_name, email, phone_number, segment)
                       VALUES %s ON CONFLICT DO NOTHING""",
                    new_custs[i:i + BATCH],
                )
                stats["customers_inserted"] += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _log(f"customers: +{stats['customers_inserted']} new, {stats['customers_skipped']} existing ({time.time()-t0:.1f}s)")
    return stats, clerk_map, email_map, name_map


def _stage_products(business_id: str, products: list[dict]) -> tuple[dict, dict, dict]:
    """Returns (stats, acc_map, pname_map)."""
    stats = {"products_inserted": 0, "products_skipped": 0}
    acc_map:   dict[str, str] = {}
    pname_map: dict[str, str] = {}

    if not products:
        return stats, acc_map, pname_map

    t0 = time.time()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, external_acc_id, name FROM product WHERE business_id = %s",
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
                if not nm or (acc and acc in acc_map) or nm in pname_map:
                    continue
                new_id = str(uuid.uuid4())
                new_prods.append((new_id, business_id, nm,
                                  str(p.get("price") or "0"),
                                  str(p.get("cost")) if p.get("cost") is not None else None,
                                  int(p.get("stock") or 0),
                                  _s(p.get("description")), acc))
                if acc: acc_map[acc]  = new_id
                pname_map[nm] = new_id

            _log(f"products: preparing {len(new_prods)} new rows for insert…")

            for i in range(0, len(new_prods), BATCH):
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO product
                       (id, business_id, name, price, cost, stock, description, external_acc_id)
                       VALUES %s ON CONFLICT DO NOTHING""",
                    new_prods[i:i + BATCH],
                )
                stats["products_inserted"] += cur.rowcount
                if i % (BATCH * 20) == 0 and i > 0:
                    _log(f"products: {i}/{len(new_prods)} inserted so far…")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _log(f"products: +{stats['products_inserted']} new, {stats['products_skipped']} existing ({time.time()-t0:.1f}s)")
    return stats, acc_map, pname_map


def _stage_orders_and_items(
    business_id: str,
    orders:      list[dict],
    order_items: list[dict],
    clerk_map:   dict, email_map: dict, name_map: dict,
    acc_map:     dict, pname_map: dict,
) -> dict:
    stats = {"orders_inserted": 0, "order_items_inserted": 0}
    if not orders:
        return stats

    t0 = time.time()

    # Index items by externalOrderId
    items_by_ext: dict[str, list[dict]] = {}
    for item in order_items:
        key = _s(item.get("orderExternalId")) or ""
        items_by_ext.setdefault(key, []).append(item)

    # Build order rows + ext_id_to_db map in memory (no DB access)
    order_rows:   list[tuple] = []
    ext_id_to_db: dict[str, str] = {}

    for o in orders:
        ext_cust = _s(o.get("customerExternalId")) or ""
        cust_db  = clerk_map.get(ext_cust) or email_map.get(ext_cust) or name_map.get(ext_cust)
        if not cust_db:
            continue

        order_db_id  = str(uuid.uuid4())
        ext_order_id = _s(o.get("externalOrderId")) or order_db_id
        ext_id_to_db[ext_order_id] = order_db_id

        order_rows.append((
            order_db_id, business_id, cust_db,
            str(o.get("total") or o.get("revenue") or "0"),
            o.get("status") or "completed",
            _s(o.get("orderVoucher")),
            "0",
            _s(o.get("address")),
            _s(o.get("createdAt")),
        ))

    rows_with_date    = [r for r in order_rows if r[8] is not None]
    rows_without_date = [r for r in order_rows if r[8] is None]

    # Build item rows in memory
    item_rows: list[tuple] = []
    for ext_oid, items in items_by_ext.items():
        order_db_id = ext_id_to_db.get(ext_oid)
        if not order_db_id:
            continue
        for item in items:
            prod_db = (acc_map.get(_s(item.get("productAccId")) or "")
                       or pname_map.get(_s(item.get("productName")) or ""))
            if not prod_db:
                continue
            attrs = _sanitize_attrs(item.get("attributes"))
            item_rows.append((
                str(uuid.uuid4()), order_db_id, prod_db,
                int(item.get("quantity") or 1),
                str(item.get("unitPrice") or "0"),
                str(item.get("itemDiscount") or "0"),
                json.dumps(attrs) if attrs else None,
            ))

    _log(f"orders: prepared {len(rows_with_date)} with dates, {len(rows_without_date)} without; items: {len(item_rows)}")

    # Insert orders
    conn = _connect()
    try:
        with conn.cursor() as cur:
            for i in range(0, len(rows_with_date), BATCH):
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO "order"
                       (id, business_id, customer_id, total, status,
                        order_voucher, order_discount, address, created_at)
                       VALUES %s ON CONFLICT DO NOTHING""",
                    rows_with_date[i:i + BATCH],
                )
                stats["orders_inserted"] += cur.rowcount

            for i in range(0, len(rows_without_date), BATCH):
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO "order"
                       (id, business_id, customer_id, total, status,
                        order_voucher, order_discount, address)
                       VALUES %s ON CONFLICT DO NOTHING""",
                    [(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]) for r in rows_without_date[i:i + BATCH]],
                )
                stats["orders_inserted"] += cur.rowcount
        conn.commit()
        _log(f"orders: +{stats['orders_inserted']} ({time.time()-t0:.1f}s)")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Insert items — fresh connection
    t1 = time.time()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            for i in range(0, len(item_rows), BATCH):
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO order_item
                       (id, order_id, product_id, quantity, unit_price, item_discount, attributes)
                       VALUES %s ON CONFLICT DO NOTHING""",
                    item_rows[i:i + BATCH],
                )
                stats["order_items_inserted"] += cur.rowcount
                if i % (BATCH * 20) == 0 and i > 0:
                    _log(f"order_items: {i}/{len(item_rows)} inserted so far…")
        conn.commit()
        _log(f"order_items: +{stats['order_items_inserted']} ({time.time()-t1:.1f}s)")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return stats


# ─── Public entry ────────────────────────────────────────────────────────────

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

    _log(f"starting: {len(customers)} customers, {len(products)} products, "
         f"{len(orders)} orders, {len(order_items)} items")

    try:
        s1, clerk_map, email_map, name_map = _stage_customers(business_id, customers)
        stats.update(s1)

        s2, acc_map, pname_map = _stage_products(business_id, products)
        stats.update(s2)

        s3 = _stage_orders_and_items(
            business_id, orders, order_items,
            clerk_map, email_map, name_map,
            acc_map, pname_map,
        )
        stats.update(s3)

        _log(f"DONE — {stats}")
        return stats

    except Exception as exc:
        _log(f"FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        raise