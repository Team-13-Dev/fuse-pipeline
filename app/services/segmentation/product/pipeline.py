"""
app/services/segmentation/product/pipeline.py

Pure ML pipeline for product segmentation.
Takes a DataFrame with the required columns, returns clustering results.
No I/O, no side effects — easy to test and reuse.

Required input columns:
    product_id, price, cost, quantity, stock, revenue, profit, profit_margin
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from app.services.segmentation.product.schemas import (
    SegmentationResult, ProductLabel, ClusterStats,
)


REQUIRED_COLUMNS = [
    "product_id", "price", "cost", "quantity",
    "stock", "revenue", "profit", "profit_margin",
]

# Minimum products required to produce meaningful clusters.
# With <15 products clustering is statistically unreliable.
MIN_PRODUCTS_FOR_SEGMENTATION = 15


class InsufficientDataError(Exception):
    """Raised when input doesn't meet minimum requirements for clustering."""


def _log(msg: str) -> None:
    print(f"[product_segmentation] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validate
# ─────────────────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InsufficientDataError(f"Missing required columns: {missing}")

    if len(df) < MIN_PRODUCTS_FOR_SEGMENTATION:
        raise InsufficientDataError(
            f"Need at least {MIN_PRODUCTS_FOR_SEGMENTATION} products for segmentation, got {len(df)}"
        )

    extra_cols = [c for c in ["product_name"] if c in df.columns]
    return df[REQUIRED_COLUMNS + extra_cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clustering features. We fill NaNs rather than drop rows because
    local brands often have sparse data — we'd rather cluster all products
    with sensible defaults than refuse to cluster anything.
    """
    df = df[df["cost"] >= 0].reset_index(drop=True)

    df["absolute_margin"] = df["price"] - df["cost"]
    df["stock_turnover"]  = (df["quantity"] / df["stock"].replace(0, np.nan)).fillna(0)

    cluster_features = ["profit_margin", "absolute_margin", "stock_turnover", "quantity"]
    df = df.dropna(subset=cluster_features).reset_index(drop=True)

    if len(df) < MIN_PRODUCTS_FOR_SEGMENTATION:
        raise InsufficientDataError(
            f"After cleaning, only {len(df)} products remain (need {MIN_PRODUCTS_FOR_SEGMENTATION})."
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transform & scale
# ─────────────────────────────────────────────────────────────────────────────

def _transform_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["absolute_margin_log"] = np.log1p(work["absolute_margin"].clip(lower=-1 + 1e-6))
    work["quantity_log"]        = np.log1p(work["quantity"])

    cols_to_transform = [
        "absolute_margin_log", "quantity_log",
        "profit_margin", "stock_turnover",
    ]
    pt = PowerTransformer(method="yeo-johnson")
    work[cols_to_transform] = pt.fit_transform(work[cols_to_transform])

    upper = work["quantity_log"].quantile(0.99)
    work["quantity_log"] = np.clip(work["quantity_log"], None, upper)

    scaler = RobustScaler()
    work[cols_to_transform] = scaler.fit_transform(work[cols_to_transform])

    cluster_cols = ["profit_margin", "absolute_margin_log", "stock_turnover", "quantity_log"]
    df_cluster   = work[cluster_cols].copy()
    return df_cluster, work


# ─────────────────────────────────────────────────────────────────────────────
# 4. Find best (model, k)
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_model(df_cluster: pd.DataFrame):
    # Cap k so clusters stay meaningful: at most 6, and never more than n/5
    # (keeps average cluster size >= 5 products).
    max_k = min(6, len(df_cluster) // 5, len(df_cluster) - 1)
    k_values = range(2, max_k + 1)

    best_score, best_model, best_labels, best_k, model_name = -1, None, None, 2, ""

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(df_cluster)
        if len(set(km_labels)) > 1:
            min_cluster_size = max(2, len(df_cluster) // 10)
            if min(np.bincount(km_labels)) >= min_cluster_size:
                km_score = silhouette_score(df_cluster, km_labels)
            if km_score > best_score:
                best_score, best_model, best_labels, best_k, model_name = (
                    km_score, km, km_labels, k, "KMeans"
                )

        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="diag")
        gmm_labels = gmm.fit_predict(df_cluster)
        if len(set(gmm_labels)) > 1:
            gmm_score = silhouette_score(df_cluster, gmm_labels)
            if gmm_score > best_score:
                best_score, best_model, best_labels, best_k, model_name = (
                    gmm_score, gmm, gmm_labels, k, "GMM"
                )

    if best_model is None:
        raise InsufficientDataError("Could not form valid clusters from this dataset.")

    return best_model, best_labels, best_k, round(float(best_score), 4), model_name


# ─────────────────────────────────────────────────────────────────────────────
# 5. Name clusters
# ─────────────────────────────────────────────────────────────────────────────

def _name_clusters(model, model_name: str, best_k: int) -> dict[int, str]:
    all_features = ["profit_margin", "absolute_margin_log", "stock_turnover", "quantity_log"]
    if model_name == "KMeans":
        centers = pd.DataFrame(model.cluster_centers_, columns=all_features)
    else:
        centers = pd.DataFrame(model.means_, columns=all_features)

    # Use all 4 features for naming — quantity is the strongest differentiator
    # in single-category stores where margins are all similar.
    ranks = centers.rank(method="first").astype(int)

    names: dict[int, str] = {}
    for i in range(best_k):
        absolute_rank = ranks.loc[i, "absolute_margin_log"]
        turnover_rank = ranks.loc[i, "stock_turnover"]
        quantity_rank = ranks.loc[i, "quantity_log"]
        margin_rank   = ranks.loc[i, "profit_margin"]

        hi = best_k * 0.6
        lo = best_k * 0.4

        high_absolute = absolute_rank >= hi
        high_turnover = turnover_rank >= hi
        high_quantity = quantity_rank >= hi
        low_absolute  = absolute_rank <= lo
        low_turnover  = turnover_rank <= lo
        low_quantity  = quantity_rank <= lo
        low_margin    = margin_rank   <= lo

        if   high_absolute and high_turnover and high_quantity:  name = "Premium Stars"
        elif high_absolute and low_turnover  and low_quantity:   name = "High Margin, Slow Movers"
        elif low_absolute  and high_quantity:                    name = "High Volume, Thin Margin"
        elif low_turnover  and low_quantity:                     name = "Underperformers"
        elif high_turnover and high_quantity and not low_margin: name = "Fast Movers"
        elif high_quantity and not high_turnover:                name = "Volume Sellers"
        elif low_quantity  and not low_turnover:                 name = "Dormant Inventory"
        elif high_absolute and not low_turnover:                 name = "Solid Performers"
        else:                                                    name = "Mid-Tier"
        names[i] = name

    # Safety-net deduplication — with 9 names and max k=6 this rarely fires.
    seen: dict[str, int] = {}
    for i, name in list(names.items()):
        if name in seen:
            seen[name] += 1
            names[i] = f"{name} {seen[name]}"
        else:
            seen[name] = 1
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 6. Build response
# ─────────────────────────────────────────────────────────────────────────────

def _build_result(
    df_raw: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str],
    model_name: str,
    best_k: int,
    silhouette: float,
) -> SegmentationResult:
    df = df_raw.copy()
    df["cluster"]      = labels
    df["cluster_name"] = df["cluster"].map(cluster_names)

    product_labels = [
        ProductLabel(
            product_id=str(r["product_id"]),
            cluster=int(r["cluster"]),
            cluster_name=str(r["cluster_name"]),
        )
        for _, r in df[["product_id", "cluster", "cluster_name"]].iterrows()
    ]

    agg = df.groupby(["cluster", "cluster_name"]).agg(
        num_products  =("profit",        "count"),
        avg_profit    =("profit",        "mean"),
        total_profit  =("profit",        "sum"),
        avg_revenue   =("revenue",       "mean"),
        total_revenue =("revenue",       "sum"),
        avg_price     =("price",         "mean"),
        avg_cost      =("cost",          "mean"),
        avg_margin    =("profit_margin", "mean"),
        avg_stock     =("stock",         "mean"),
        avg_quantity  =("quantity",      "mean"),
    ).round(2).reset_index()

    total_rev_sum = float(agg["total_revenue"].sum()) or 1
    total_pft_sum = float(agg["total_profit"].sum())  or 1
    agg["revenue_share_pct"] = (agg["total_revenue"] / total_rev_sum * 100).round(2)
    agg["profit_share_pct"]  = (agg["total_profit"]  / total_pft_sum * 100).round(2)

    cluster_stats: list[ClusterStats] = []
    for _, row in agg.iterrows():
        cl_id   = int(row["cluster"])
        cl_name = str(row["cluster_name"])
        in_cl   = df[df["cluster"] == cl_id]

        scored = in_cl.copy()
        scored["composite_score"] = scored["profit_margin"] * np.sign(scored["profit"]) * np.log1p(scored["profit"].abs())

        id_cols = ["product_id", "composite_score", "price", "profit", "profit_margin"]
        if "product_name" in scored.columns:
            id_cols.insert(1, "product_name")

        # Cap n so top and bottom lists never draw from the same products.
        # floor(size/2) guarantees the two halves are disjoint.
        n = min(5, len(in_cl) // 2)

        top_n = scored.nlargest(n, "composite_score")[id_cols] if n > 0 else scored.iloc[0:0][id_cols]
        top   = [{"product_id":    str(r["product_id"]),
                  "product_name":  str(r["product_name"]) if pd.notna(r.get("product_name")) else None,
                  "price":         float(r["price"]),
                  "profit":        float(r["profit"]),
                  "composite_score": round(float(r["composite_score"]), 4)}
                 for _, r in top_n.iterrows()]

        bot_n = scored.nsmallest(n, "composite_score")[id_cols] if n > 0 else scored.iloc[0:0][id_cols]
        bot   = [{"product_id":    str(r["product_id"]),
                  "product_name":  str(r["product_name"]) if pd.notna(r.get("product_name")) else None,
                  "price":         float(r["price"]),
                  "profit_margin": float(r["profit_margin"]),
                  "composite_score": round(float(r["composite_score"]), 4)}
                 for _, r in bot_n.iterrows()]

        cluster_stats.append(ClusterStats(
            cluster           =cl_id,
            cluster_name      =cl_name,
            num_products      =int(row["num_products"]),
            avg_profit        =float(row["avg_profit"]),
            total_profit      =float(row["total_profit"]),
            avg_revenue       =float(row["avg_revenue"]),
            total_revenue     =float(row["total_revenue"]),
            avg_price         =float(row["avg_price"]),
            avg_cost          =float(row["avg_cost"]),
            avg_margin        =float(row["avg_margin"]),
            avg_stock         =float(row["avg_stock"]),
            avg_quantity      =float(row["avg_quantity"]),
            revenue_share_pct =float(row["revenue_share_pct"]),
            profit_share_pct  =float(row["profit_share_pct"]),
            top_products      =top,
            bottom_products   =bot,
        ))

    return SegmentationResult(
        model_used       =model_name,
        best_k           =best_k,
        silhouette_score =silhouette,
        total_rows       =len(df),
        product_labels   =product_labels,
        cluster_stats    =cluster_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_product_segmentation(df_raw: pd.DataFrame) -> SegmentationResult:
    """
    Run the full product segmentation pipeline on a raw DataFrame.
    Raises InsufficientDataError if the data isn't usable.
    """
    _log(f"input rows: {len(df_raw)}")
    df = _validate(df_raw)
    df = _engineer_features(df)
    _log(f"after feature engineering: {len(df)} products")

    df_cluster, df_enriched = _transform_and_scale(df)
    model, labels, best_k, silhouette, model_name = _find_best_model(df_cluster)
    _log(f"best model: {model_name}, k={best_k}, silhouette={silhouette}")

    cluster_names = _name_clusters(model, model_name, best_k)
    return _build_result(df, labels, cluster_names, model_name, best_k, silhouette)