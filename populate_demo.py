import psycopg2, uuid, random
from datetime import datetime, timedelta

conn = psycopg2.connect(
    'postgresql://neondb_owner:npg_LOVbN0GWg6HK@ep-cool-darkness-ai3yrsk9-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require',
    connect_timeout=15
)
cur = conn.cursor()
BID = '1428f4e6-8dbf-4a57-8875-f07b11a5189f'

# 1. Wipe existing orders (re-run safe)
print("Clearing old orders…")
cur.execute('DELETE FROM order_item WHERE order_id IN (SELECT id FROM "order" WHERE business_id = %s)', (BID,))
conn.commit()
cur.execute('DELETE FROM "order" WHERE business_id = %s', (BID,))
conn.commit()
print("Cleared old orders")

# 2. (product_id, new_stock, target_qty_sold)
PLAN = [
    # ── Premium Stars (high absolute margin, fast-selling, low stock) ─────────
    ('5e9d66eb-9458-4076-8e00-c1300bbeb303', 5,   580),  # Puffer Jacket Black
    ('105063b5-bd28-4e8c-9eb0-3bcd05b16d35', 6,   540),  # Zip-Up Jacket Navy
    ('73e3e32c-5eca-47b7-844d-ba33cd7b2a82', 8,   490),  # Denim Jacket Blue
    ('6dbfbf3b-c6a6-4466-b30d-ba0528722fde', 7,   460),  # Black Sneakers
    ('2865333d-0fbe-4ae8-9a39-0ea48104a641', 9,   430),  # Beige Sneakers
    # ── High Margin, Slow Movers (good margins, massively overstocked) ─────────
    ('4abe0659-4ef0-4e7a-8a85-76de8db906cd', 120,  45),  # Printed Hoodie Black
    ('94d4a3ee-d4e9-4adf-bf15-14362d69eb41', 100,  52),  # Zip Hoodie Navy
    ('6363c13d-b5a4-4395-a266-5e2615247d6d',  90,  38),  # Grey Hoodie Plain
    ('1601340e-4569-4867-8d36-ec1bdf3e4742',  80,  42),  # Ripped Jeans Grey
    ('557cbf94-8ae2-4f57-b0cf-08f78b76ee27',  85,  48),  # Classic Blue Jeans
    # ── High Volume Budget (tees/accessories — thin absolute margin, huge qty) ─
    ('136c1176-1351-4327-8882-f180fb5bb55b', 250, 850),  # Basic White Tee
    ('ce9315b9-fa18-4a01-b957-de06a0bf81cf', 230, 800),  # Striped Tee Navy
    ('6bafd160-bdc4-48af-979e-04cd3da9b7a4', 200, 750),  # Logo Cap Black
    ('48cdc879-2247-439c-bae1-f7218d8dc526', 220, 820),  # Graphic Print Tee
    ('397e3f3d-c845-43e6-b0af-6d0e66726878', 180, 780),  # OverSize Black T-shirt
    ('8e37fac0-05ac-42f3-8706-9ebfd447a201', 190, 720),  # Leather Belt Black
    ('317f160f-fc4f-40c1-a9eb-0c995b4ae808', 170, 700),  # Canvas Tote Bag
    # ── Underperformers (barely selling, overstocked) ─────────────────────────
    ('ae3d17ee-9ed6-43ee-a701-a37fa8d3dda6',  80,  18),  # Cargo Shorts Khaki
    ('457aba27-21be-4099-967b-343d1c35fb60',  90,  22),  # Cargo Pants Beige
    ('f16c5f18-13fc-4114-8fe2-45c43700e8a2',  75,  15),  # Polo Striped Blue
    ('e6846dd8-a8fc-4497-b1a7-44e5ba543de5',  85,  12),  # Black Chinos
    ('6639995b-27f6-4132-97bf-59069dfa62e3',  70,  20),  # Sweatpants Grey
    # ── Balanced Mid-tier ─────────────────────────────────────────────────────
    ('4a4510ad-7275-4ca9-8620-4188d92a7cd9',  30, 200),  # Oxford Shirt Blue
    ('5297e2a0-1f09-4edc-a4fd-97a481aa99a2',  28, 190),  # Flannel Shirt Green
    ('868daa77-a155-4c29-b5d4-3bf58793a754',  35, 220),  # Linen Shirt White
    ('25526ad2-4cc0-416a-a208-6db2ac918c4b',  32, 210),  # Classic Polo White
    ('ad8b92df-f178-436d-8012-3514f1df27e7',  30, 195),  # Pique Polo Black
    ('b13eea20-02c1-4ec2-84a2-c08f2697ab85',  28, 185),  # Skinny Black Jeans
    ('a39501a4-b7a7-4868-903c-c75a8085d872',  32, 175),  # Slim Fit Chinos Beige
    ('88c3dacc-58c8-4c2e-af67-96dfbaf819b8',  30, 165),  # Sweatshirt Crew Blue
    ('b8636f24-cb6f-4cab-8673-81f0da9d7b9b',  35, 210),  # White Canvas Shoes
]

# 3. Update stocks
for pid, stock, _ in PLAN:
    cur.execute('UPDATE product SET stock = %s WHERE id = %s', (stock, pid))
conn.commit()
print("Updated stocks")

# 4. Load prices and customers
cur.execute('SELECT id, price FROM product WHERE business_id = %s', (BID,))
price_map = {r[0]: float(r[1]) for r in cur.fetchall()}

cur.execute('SELECT id FROM customer WHERE business_id = %s', (BID,))
customers = [r[0] for r in cur.fetchall()]
print(f"Using {len(customers)} customers")

# 5. Build ticket pool and batch into orders
random.seed(42)
now = datetime.utcnow()

tickets = []
for pid, _, target_qty in PLAN:
    price = price_map[pid]
    remaining = target_qty
    while remaining > 0:
        qty = min(random.randint(1, 3), remaining)
        tickets.append((pid, price, qty))
        remaining -= qty

random.shuffle(tickets)

orders_created = 0
units_created = 0
i = 0
BATCH_COMMIT = 200  # commit every N orders to avoid pooler timeout

while i < len(tickets):
    batch = tickets[i:i + random.randint(1, 4)]
    i += len(batch)

    customer_id = random.choice(customers)
    created_at = now - timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
    total = sum(p * q for _, p, q in batch)

    # Realistic order status distribution:
    # delivered 45%, shipped 20%, confirmed 10%, pending 10%, cancelled 10%, refunded 5%
    status = random.choices(
        ["delivered", "shipped", "confirmed", "pending", "cancelled", "refunded"],
        weights=[45, 20, 10, 10, 10, 5],
        k=1
    )[0]

    oid = str(uuid.uuid4())
    cur.execute(
        'INSERT INTO "order" (id, customer_id, business_id, total, status, created_at) VALUES (%s,%s,%s,%s,%s,%s)',
        (oid, customer_id, BID, total, status, created_at)
    )
    for pid, price, qty in batch:
        cur.execute(
            'INSERT INTO order_item (id, order_id, product_id, quantity, unit_price) VALUES (%s,%s,%s,%s,%s)',
            (str(uuid.uuid4()), oid, pid, qty, price)
        )
        units_created += qty
    orders_created += 1

    if orders_created % BATCH_COMMIT == 0:
        conn.commit()
        print(f"  committed {orders_created} orders so far…")

conn.commit()
print(f"Done: {orders_created} orders, {units_created} total units")
conn.close()
