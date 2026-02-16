"""阶段 3: 反事实质量评估指标。

对齐 docs/evaluation_suite.md §3:
    - R² (mean / var)
    - PCC (Pearson Correlation)
    - MSE
    - Marker Direction Accuracy
    - Marker Magnitude Ranking (Spearman)
    - DEG Overlap (Jaccard)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


def r2_mean(
    observed: np.ndarray,
    counterfactual: np.ndarray,
) -> float:
    """跨基因 mean(predicted) vs mean(ground_truth) 的 R²。

    Parameters
    ----------
    observed, counterfactual : (N, G)
    """
    obs_mean = np.mean(observed, axis=0)
    cf_mean = np.mean(counterfactual, axis=0)
    ss_res = np.sum((cf_mean - obs_mean) ** 2)
    ss_tot = np.sum((obs_mean - obs_mean.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def r2_var(
    observed: np.ndarray,
    counterfactual: np.ndarray,
) -> float:
    """跨基因 var(predicted) vs var(ground_truth) 的 R²。"""
    obs_var = np.var(observed, axis=0)
    cf_var = np.var(counterfactual, axis=0)
    ss_res = np.sum((cf_var - obs_var) ** 2)
    ss_tot = np.sum((obs_var - obs_var.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def pcc_median(
    observed: np.ndarray,
    counterfactual: np.ndarray,
) -> float:
    """逐基因 Pearson 相关系数的中位数。"""
    n_genes = observed.shape[1]
    pccs = []
    for g in range(n_genes):
        o = observed[:, g]
        c = counterfactual[:, g]
        if np.std(o) < 1e-12 or np.std(c) < 1e-12:
            continue
        r, _ = stats.pearsonr(o, c)
        if np.isfinite(r):
            pccs.append(r)
    return float(np.median(pccs)) if pccs else 0.0


def mse(
    observed: np.ndarray,
    counterfactual: np.ndarray,
) -> float:
    """全矩阵均方误差。"""
    return float(np.mean((counterfactual - observed) ** 2))


def marker_direction_accuracy(
    observed: np.ndarray,
    counterfactual: np.ndarray,
    expected_directions: dict[str, int],
    gene_names: list[str],
) -> float:
    """Marker 基因表达变化方向准确率。

    Parameters
    ----------
    observed, counterfactual : (N, G)
    expected_directions : {gene_name: -1 or +1}
        -1 表示预期下调，+1 表示预期上调。
    gene_names : 基因名列表（长度 G）

    Returns
    -------
    float in [0, 1]
    """
    if not expected_directions:
        return 0.0
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    correct = 0
    total = 0
    obs_mean = np.mean(observed, axis=0)
    cf_mean = np.mean(counterfactual, axis=0)
    for gene, expected in expected_directions.items():
        idx = gene_to_idx.get(gene.upper())
        if idx is None:
            continue
        actual_direction = 1 if cf_mean[idx] > obs_mean[idx] else -1
        if actual_direction == expected:
            correct += 1
        total += 1
    return float(correct / max(total, 1))


def marker_magnitude_ranking(
    observed: np.ndarray,
    counterfactual: np.ndarray,
    marker_genes: list[str],
    gene_names: list[str],
    expected_ranking: Optional[list[str]] = None,
) -> float:
    """Marker 基因变化幅度排序与预期的 Spearman 相关性。"""
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    obs_mean = np.mean(observed, axis=0)
    cf_mean = np.mean(counterfactual, axis=0)

    magnitudes = []
    valid_genes = []
    for g in marker_genes:
        idx = gene_to_idx.get(g.upper())
        if idx is not None:
            magnitudes.append(abs(cf_mean[idx] - obs_mean[idx]))
            valid_genes.append(g.upper())

    if len(valid_genes) < 2:
        return 0.0

    if expected_ranking is not None:
        expected_order = [g.upper() for g in expected_ranking if g.upper() in valid_genes]
        if len(expected_order) < 2:
            return 0.0
        actual_ranks = np.argsort(np.argsort([-m for m in magnitudes]))
        expected_ranks = np.array([valid_genes.index(g) for g in expected_order])
        actual_selected = actual_ranks[[valid_genes.index(g) for g in expected_order]]
        rho, _ = stats.spearmanr(actual_selected, np.arange(len(expected_order)))
        return float(rho) if np.isfinite(rho) else 0.0
    return 0.0


def deg_overlap_jaccard(
    observed: np.ndarray,
    counterfactual: np.ndarray,
    gene_names: list[str],
    reference_degs: set[str],
    top_k: int = 100,
) -> float:
    """反事实 DEGs 与参考 DEGs 的 Jaccard 指数。"""
    obs_mean = np.mean(observed, axis=0)
    cf_mean = np.mean(counterfactual, axis=0)
    fold_change = np.abs(cf_mean - obs_mean)
    top_indices = np.argsort(fold_change)[::-1][:top_k]
    predicted_degs = {gene_names[i].upper() for i in top_indices}
    ref = {g.upper() for g in reference_degs}
    intersection = predicted_degs & ref
    union = predicted_degs | ref
    return float(len(intersection) / max(len(union), 1))


def evaluate_counterfactual(
    observed: np.ndarray,
    counterfactual: np.ndarray,
    gene_names: Optional[list[str]] = None,
    expected_directions: Optional[dict[str, int]] = None,
    reference_degs: Optional[set[str]] = None,
) -> dict[str, float]:
    """一站式反事实质量评估。"""
    result: dict[str, float] = {
        "r2_mean": r2_mean(observed, counterfactual),
        "r2_var": r2_var(observed, counterfactual),
        "pcc_median": pcc_median(observed, counterfactual),
        "mse": mse(observed, counterfactual),
    }
    if gene_names is not None and expected_directions:
        result["marker_direction_accuracy"] = marker_direction_accuracy(
            observed, counterfactual, expected_directions, gene_names
        )
    if gene_names is not None and reference_degs:
        result["deg_overlap_jaccard"] = deg_overlap_jaccard(
            observed, counterfactual, gene_names, reference_degs
        )
    return result
