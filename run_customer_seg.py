"""
One-shot customer segmentation for fuse-store.
Runs RFM pipeline on real DB data and persists results.
"""
import os, uuid, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler

os.environ["DATABASE_URL"] = (
    "postgresql://neondb_owner:npg_LOVbN0GWg6HK"
    "@ep-cool-darkness-ai3yrsk9-pooler.c-4.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require&channel_binding=require"
)
from app.core.database import get_db_connection

BUSINESS_ID = "1428f4e6-8dbf-4a57-8875-f07b11a5189f"

# ── 1. Load RFM data ──────────────────────────────────────────────────────────
print("Loading RFM data from Neon...")
conn = get_db_connection()
cur  = conn.cursor()
cur.execute("""
  SELECT
    c.id::text                                                AS customer_id,
    c.full_name,
    EXTRACT(DAY FROM NOW() - MAX(o.created_at))::float        AS recency,
    COUNT(DISTINCT o.id)::float                               AS frequency,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0)::float      AS monetary,
    COALESCE(AVG(o.total::numeric), 0)::float                 AS aov,
    EXTRACT(DAY FROM MAX(o.created_at) - MIN(o.created_at))::float AS tenure
  FROM customer c
  JOIN "order"   o  ON o.customer_id = c.id AND o.business_id = %s
  JOIN order_item oi ON oi.order_id  = o.id
  WHERE c.business_id = %s
  GROUP BY c.id, c.full_name
  HAVING COUNT(DISTINCT o.id) > 0
""", (BUSINESS_ID, BUSINESS_ID))
cols = [d[0] for d in cur.description]
rows = cur.fetchall()
conn.close()

rfm = pd.DataFrame(rows, columns=cols)
for col in ["recency", "frequency", "monetary", "aov", "tenure"]:
    rfm[col] = pd.to_numeric(rfm[col], errors="coerce").fillna(0)

print(f"Loaded {len(rfm)} customers")
print(rfm[["recency", "frequency", "monetary"]].describe().round(1))

# ── 2. Preprocess (clip + log1p + RobustScaler) ───────────────────────────────
feat_cols = ["recency", "frequency", "monetary", "aov"]
rfm_p = rfm.copy()
for col in feat_cols:
    lo = rfm_p[col].quantile(0.01)
    hi = rfm_p[col].quantile(0.99)
    rfm_p[col] = rfm_p[col].clip(lo, hi)
for col in feat_cols:
    rfm_p[col] = np.log1p(rfm_p[col])

scaler = RobustScaler()
X = scaler.fit_transform(rfm_p[feat_cols])
print("Preprocessing done")

# ── 3. Dynamic K selection ────────────────────────────────────────────────────
n = len(rfm)
gmm_pen   = 0.15 if n < 500 else (0.08 if n < 2000 else 0.05)
force_min = 4
k_range   = range(2, 8)

results    = {}
inertias   = {}
bic_scores = {}

for k in k_range:
    km = KMeans(n_clusters=k, n_init=30, max_iter=300, random_state=42)
    km.fit(X)
    km_labels = km.labels_
    km_sil    = silhouette_score(X, km_labels)
    km_db     = davies_bouldin_score(X, km_labels)
    inertias[k] = km.inertia_

    gmm = GaussianMixture(n_components=k, n_init=5, covariance_type="full", random_state=42)
    gmm.fit(X)
    gmm_labels  = gmm.predict(X)
    gmm_sil_raw = silhouette_score(X, gmm_labels)
    gmm_bic     = gmm.bic(X)
    bic_scores[k] = gmm_bic

    gmm_sil_adj = gmm_sil_raw - gmm_pen
    if k > 2:
        prev_bic    = bic_scores.get(k - 1, gmm_bic)
        improvement = (prev_bic - gmm_bic) / (abs(prev_bic) + 1e-9)
        if improvement <= 0.01:
            gmm_sil_adj -= 0.10

    winner   = "KMeans" if km_sil >= gmm_sil_adj else "GMM"
    best_sil = max(km_sil, gmm_sil_adj)
    results[k] = dict(
        kmeans=km, gmm=gmm,
        km_labels=km_labels, gmm_labels=gmm_labels,
        km_sil=km_sil, km_db=km_db,
        gmm_sil_raw=gmm_sil_raw, gmm_sil_adj=gmm_sil_adj,
        winner=winner, best_sil=best_sil,
        km_inertia=km.inertia_, gmm_bic=gmm_bic,
    )

ks     = sorted(results.keys())
sil_v  = {k: results[k]["best_sil"]          for k in ks}
db_v   = {k: 1 / (1 + results[k]["km_db"])   for k in ks}
elbow_v: dict = {}
for i, k in enumerate(ks):
    if i == 0:
        elbow_v[k] = 1.0
    else:
        pk = ks[i - 1]
        elbow_v[k] = (inertias[pk] - inertias[k]) / (inertias[pk] + 1e-9)

for d in [sil_v, db_v, elbow_v]:
    mn, mx = min(d.values()), max(d.values())
    for k in ks:
        d[k] = (d[k] - mn) / (mx - mn + 1e-9)

combined     = {k: 0.60 * sil_v[k] + 0.25 * db_v[k] + 0.15 * elbow_v[k] for k in ks}
best_k_data  = max(combined, key=lambda k: combined[k])
best_k_final = max(best_k_data, force_min)

print("\nK selection:")
for k in ks:
    r = results[k]
    marker = " <-- CHOSEN" if k == best_k_final else ""
    print(f"  K={k}  KM={r['km_sil']:.4f}  GMM_adj={r['gmm_sil_adj']:.4f}"
          f"  combined={combined[k]:.3f}  {r['winner']}{marker}")
print(f"  data_best={best_k_data}  final={best_k_final}")

# ── 4. Final model ────────────────────────────────────────────────────────────
res    = results[best_k_final]
winner = res["winner"]
model  = res["kmeans"]    if winner == "KMeans" else res["gmm"]
labels = res["km_labels"] if winner == "KMeans" else res["gmm_labels"]
best_sil_raw = res["km_sil"] if winner == "KMeans" else res["gmm_sil_raw"]
print(f"\nFinal: {winner} K={best_k_final} silhouette={best_sil_raw:.4f}")

# ── 5. Label segments by composite RFM score ──────────────────────────────────
SEGMENT_NAMES = {
    2: ["Active Customers",  "Churned Customers"],
    3: ["Champions",         "Need Attention",    "At-Risk"],
    4: ["Champions",         "Loyal Customers",   "At-Risk",        "Lost Customers"],
    5: ["Champions",         "Loyal Customers",   "Promising",      "At-Risk",        "Lost Customers"],
    6: ["Champions",         "Loyal Customers",   "Promising",      "Need Attention", "At-Risk",        "Lost Customers"],
    7: ["Champions",         "Loyal Customers",   "Promising",      "New Customers",  "Need Attention", "At-Risk",        "Lost Customers"],
    8: ["Champions",         "Loyal Customers",   "High Value",     "Promising",      "New Customers",  "Need Attention", "At-Risk",        "Lost Customers"],
}
ACTION_MAP = {
    "Champions":         {"churn_risk": "LOW",       "priority": "Critical", "channel": "Email + App Push",  "offer": "VIP Exclusive Access",    "upsell": "Yes",   "campaign_freq": "Weekly"},
    "Loyal Customers":   {"churn_risk": "LOW",       "priority": "High",     "channel": "Email",             "offer": "Loyalty Points Reward",   "upsell": "Yes",   "campaign_freq": "Bi-Weekly"},
    "High Value":        {"churn_risk": "LOW",       "priority": "High",     "channel": "Personal Outreach", "offer": "Exclusive Deal",          "upsell": "Yes",   "campaign_freq": "Weekly"},
    "Promising":         {"churn_risk": "MEDIUM",    "priority": "Medium",   "channel": "Email + Social",    "offer": "Welcome Gift / Discount", "upsell": "Maybe", "campaign_freq": "Monthly"},
    "New Customers":     {"churn_risk": "MEDIUM",    "priority": "Medium",   "channel": "Email",             "offer": "Onboarding Series",       "upsell": "Maybe", "campaign_freq": "Weekly"},
    "Need Attention":    {"churn_risk": "HIGH",      "priority": "Medium",   "channel": "Email + SMS",       "offer": "Re-engagement Offer",     "upsell": "No",    "campaign_freq": "Bi-Weekly"},
    "Active Customers":  {"churn_risk": "MEDIUM",    "priority": "Medium",   "channel": "Email",             "offer": "Engagement Campaign",     "upsell": "Maybe", "campaign_freq": "Monthly"},
    "At-Risk":           {"churn_risk": "VERY HIGH", "priority": "Critical", "channel": "Email + SMS",       "offer": "Win-Back Promo 20% off",  "upsell": "No",    "campaign_freq": "Bi-Weekly"},
    "Lost Customers":    {"churn_risk": "CRITICAL",  "priority": "Low",      "channel": "Email",             "offer": "Comeback Deal 30% off",   "upsell": "No",    "campaign_freq": "Quarterly"},
    "Churned Customers": {"churn_risk": "CRITICAL",  "priority": "Low",      "channel": "Email",             "offer": "Win-Back Campaign",       "upsell": "No",    "campaign_freq": "Quarterly"},
}
DEFAULT_ACTION = {"churn_risk": "MEDIUM", "priority": "Medium", "channel": "Email", "offer": "Special Deal", "upsell": "Maybe", "campaign_freq": "Monthly"}

rfm2 = rfm.copy()
rfm2["cluster"] = labels

stats_rank = rfm2.groupby("cluster")[["recency", "frequency", "monetary"]].median()
stats_rank["score"] = (
    stats_rank["frequency"] / (stats_rank["frequency"].max() + 1e-9) +
    stats_rank["monetary"]  / (stats_rank["monetary"].max()  + 1e-9) -
    stats_rank["recency"]   / (stats_rank["recency"].max()   + 1e-9)
)
ranked = stats_rank.sort_values("score", ascending=False)
names  = SEGMENT_NAMES.get(best_k_final, [f"Segment {i}" for i in range(best_k_final)])
cluster_to_seg = {int(c): names[i] for i, c in enumerate(ranked.index)}
rfm2["segment"] = rfm2["cluster"].map(cluster_to_seg)

print("\nSegment summary:")
total_m = rfm2["monetary"].sum() or 1
for seg_name in names:
    grp = rfm2[rfm2["segment"] == seg_name]
    if grp.empty:
        continue
    rev_pct = grp["monetary"].sum() / total_m * 100
    print(f"  {seg_name:<22} {len(grp):>4} customers  "
          f"R={grp['recency'].median():.0f}d  "
          f"F={grp['frequency'].median():.0f}  "
          f"M=EGP{grp['monetary'].median():,.0f}  "
          f"rev={rev_pct:.1f}%")

# ── 6. Build summaries ────────────────────────────────────────────────────────
total_customers = len(rfm2)
cluster_summaries = []
for cluster_id, seg_name in cluster_to_seg.items():
    grp    = rfm2[rfm2["cluster"] == cluster_id]
    action = ACTION_MAP.get(seg_name, DEFAULT_ACTION)
    top_custs = (
        grp.nlargest(5, "monetary")
           [["customer_id", "full_name", "monetary", "frequency", "recency"]]
           .to_dict(orient="records")
    )
    cluster_summaries.append({
        "cluster":          cluster_id,
        "segment_name":     seg_name,
        "num_customers":    len(grp),
        "recency_median":   round(float(grp["recency"].median()),   2),
        "frequency_median": round(float(grp["frequency"].median()), 2),
        "monetary_median":  round(float(grp["monetary"].median()),  2),
        "monetary_sum":     round(float(grp["monetary"].sum()),     2),
        "aov_median":       round(float(grp["aov"].median()),       2),
        "tenure_median":    round(float(grp["tenure"].median()),    2),
        "revenue_pct":      round(float(grp["monetary"].sum()) / total_m * 100,    2),
        "customer_pct":     round(len(grp) / total_customers * 100, 2),
        "churn_risk":       action["churn_risk"],
        "priority":         action["priority"],
        "channel":          action["channel"],
        "offer":            action["offer"],
        "upsell":           action["upsell"],
        "campaign_freq":    action["campaign_freq"],
        "top_customers":    [
            {"customer_id": r["customer_id"],
             "name":        r["full_name"],
             "monetary":    float(r["monetary"]),
             "frequency":   float(r["frequency"]),
             "recency":     float(r["recency"])}
            for r in top_custs
        ],
    })


# ── Sanitize NumPy types before DB insert ─────────────────────────────────────
for cs in cluster_summaries:
    for key in ["recency_median", "frequency_median", "monetary_median",
                "monetary_sum", "aov_median", "tenure_median",
                "revenue_pct", "customer_pct"]:
        cs[key] = float(cs[key])
    cs["num_customers"] = int(cs["num_customers"])
    cs["cluster"] = int(cs["cluster"])
    for cust in cs["top_customers"]:
        cust["monetary"]  = float(cust["monetary"])
        cust["frequency"] = float(cust["frequency"])
        cust["recency"]   = float(cust["recency"])

# ── 7. Persist ────────────────────────────────────────────────────────────────
print("\nPersisting to DB...")
conn = get_db_connection()
cur  = conn.cursor()

job_id = str(uuid.uuid4())
cur.execute("""
  INSERT INTO analysis_job (id, business_id, type, status, progress, triggered_by, started_at, finished_at)
  VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
""", (job_id, BUSINESS_ID, "customer_segmentation", "done", 100, "manual"))

cur.execute("DELETE FROM customer_cluster_summary WHERE business_id = %s", (BUSINESS_ID,))

for cs in cluster_summaries:
    cur.execute("""
      INSERT INTO customer_cluster_summary
        (id, business_id, job_id, cluster, segment_name, num_customers,
         recency_median, frequency_median, monetary_median, monetary_sum,
         aov_median, tenure_median, revenue_pct, customer_pct,
         churn_risk, priority, channel, offer, upsell, campaign_freq, top_customers)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
    """, (
        str(uuid.uuid4()), BUSINESS_ID, job_id,
        cs["cluster"], cs["segment_name"], cs["num_customers"],
        cs["recency_median"], cs["frequency_median"], cs["monetary_median"], cs["monetary_sum"],
        cs["aov_median"], cs["tenure_median"], cs["revenue_pct"], cs["customer_pct"],
        cs["churn_risk"], cs["priority"], cs["channel"], cs["offer"], cs["upsell"], cs["campaign_freq"],
        json.dumps(cs["top_customers"]),
    ))

for _, row in rfm2.iterrows():
    cur.execute("UPDATE customer SET segment = %s WHERE id = %s",
                (row["segment"], row["customer_id"]))

cur.execute("UPDATE business SET last_customer_segment_at = NOW() WHERE id = %s", (BUSINESS_ID,))

conn.commit()
conn.close()

print("Done.")
print(f"  job_id    : {job_id}")
print(f"  model     : {winner}  K={best_k_final}  silhouette={best_sil_raw:.4f}")
print(f"  segments  : {[cs['segment_name'] for cs in cluster_summaries]}")
print(f"  customers : {total_customers}")
