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

    return df[REQUIRED_COLUMNS].copy()


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
    df["stock_turnover"]  = df["quantity"] / df["stock"].replace(0, np.nan)

    cluster_features = ["profit_margin", "absolute_margin", "stock_turnover", "quantity"]

    # Fill any remaining NaNs with column medians (or 0 if a column is all NaN).
    for col in cluster_features:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())

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
    work["absolute_margin_log"] = np.log1p(work["absolute_margin"])
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

    cluster_cols = ["profit_margin", "absolute_margin_log", "stock_turnover"]
    df_cluster   = work[cluster_cols].copy()
    return df_cluster, work


# ─────────────────────────────────────────────────────────────────────────────
# 4. Find best (model, k)
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_model(df_cluster: pd.DataFrame):
    # Cap k at min(10, n-1) — can't have more clusters than samples
    max_k = min(10, len(df_cluster) - 1)
    k_values = range(2, max_k + 1)

    best_score, best_model, best_labels, best_k, model_name = -1, None, None, 2, ""

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(df_cluster)
        if len(set(km_labels)) > 1:
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
    features = ["profit_margin", "absolute_margin_log", "stock_turnover"]
    if model_name == "KMeans":
        centers = pd.DataFrame(model.cluster_centers_, columns=features)
    else:
        centers = pd.DataFrame(model.means_, columns=features)
    ranks = centers.rank().astype(int)

    names: dict[int, str] = {}
    for i in range(best_k):
        margin_rank   = ranks.loc[i, "profit_margin"]
        absolute_rank = ranks.loc[i, "absolute_margin_log"]
        turnover_rank = ranks.loc[i, "stock_turnover"]

        high_margin   = margin_rank   >= best_k * 0.6
        high_absolute = absolute_rank >= best_k * 0.6
        high_turnover = turnover_rank >= best_k * 0.6
        low_margin    = margin_rank   <= best_k * 0.4
        low_turnover  = turnover_rank <= best_k * 0.4

        if   high_margin and high_absolute and high_turnover:     name = "Premium Stars"
        elif high_margin and high_absolute and not high_turnover: name = "High Margin, Slow Movers"
        elif high_turnover and low_margin:                        name = "Low Margin, High Velocity"
        elif low_margin   and low_turnover:                       name = "Underperformers"
        elif high_turnover and not low_margin:                    name = "Fast Movers"
        else:                                                     name = "Balanced Performance"
        names[i] = name

    seen: dict[str, int] = {}
    for i, name in list(names.items()):
        if name in seen:
            seen[name] += 1
            names[i] = f"{name} ({seen[name]})"
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

        top_n = in_cl.nlargest(5, "profit")[["product_id", "profit", "revenue"]]
        top   = [{"product_id": str(r["product_id"]), "profit": float(r["profit"]),
                  "revenue":    float(r["revenue"])}
                 for _, r in top_n.iterrows()]

        bot_n = in_cl.nsmallest(5, "profit_margin")[["product_id", "profit_margin"]]
        bot   = [{"product_id":    str(r["product_id"]),
                  "profit_margin": float(r["profit_margin"])}
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
    return _build_result(df_enriched, labels, cluster_names, model_name, best_k, silhouette)