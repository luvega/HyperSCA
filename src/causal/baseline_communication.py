"""Step2 基线通讯网络与空间因果优势评估。"""
from __future__ import annotations

import numpy as np


def build_baseline_communication_network(
    cluster_expr: np.ndarray,
    threshold_quantile: float = 0.8,
) -> np.ndarray:
    """基于 cluster 表达相关性构建传统通讯网络基线。"""
    if cluster_expr.ndim != 2:
        raise ValueError("cluster_expr must be 2D array")
    # 节点是 cluster，相关性基于基因表达谱
    corr = np.corrcoef(cluster_expr)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    abs_corr = np.abs(corr)
    thr = float(np.quantile(abs_corr[abs_corr > 0], threshold_quantile)) if np.any(abs_corr > 0) else 0.0
    adj = (abs_corr >= thr).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj


def compare_spatial_causal_advantage(
    causal_adj: np.ndarray,
    baseline_adj: np.ndarray,
    spatial_adj: np.ndarray | None = None,
) -> dict:
    """比较空间约束因果图与传统通讯网络。"""
    if causal_adj.shape != baseline_adj.shape:
        raise ValueError("causal_adj and baseline_adj must have same shape")
    k = causal_adj.shape[0]
    max_edges = max(k * (k - 1), 1)
    n_causal = int(causal_adj.sum())
    n_base = int(baseline_adj.sum())
    overlap = int(np.logical_and(causal_adj > 0, baseline_adj > 0).sum())
    jaccard = overlap / max(int(np.logical_or(causal_adj > 0, baseline_adj > 0).sum()), 1)

    metrics = {
        "n_nodes": int(k),
        "n_edges_causal": n_causal,
        "n_edges_baseline": n_base,
        "edge_density_causal": n_causal / max_edges,
        "edge_density_baseline": n_base / max_edges,
        "edge_overlap_jaccard": float(jaccard),
        "novel_edge_ratio_vs_baseline": float(max(n_causal - overlap, 0) / max(n_causal, 1)),
    }
    if spatial_adj is not None and spatial_adj.shape == causal_adj.shape:
        spatial_mask = spatial_adj > 0
        causal_spatial_consistency = float(np.logical_and(causal_adj > 0, spatial_mask).sum() / max(n_causal, 1))
        baseline_spatial_consistency = float(np.logical_and(baseline_adj > 0, spatial_mask).sum() / max(n_base, 1))
        metrics["causal_spatial_consistency"] = causal_spatial_consistency
        metrics["baseline_spatial_consistency"] = baseline_spatial_consistency
        metrics["spatial_consistency_gain"] = causal_spatial_consistency - baseline_spatial_consistency
    return metrics
