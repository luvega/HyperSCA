"""阶段 2 因果边可信度评估指标

指标覆盖（对齐 docs/evaluation_suite.md §2）:
- 结构层面: Bootstrap Edge Frequency, Graph Sparsity, Falsification p-value
- 解缠质量: HSIC(Z_int, Z_ext), Z_ext/Z_int Neighbor Predictivity
- 生物学: Known Axis Recall, Direction Accuracy, Signaling Flow Completeness
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# =========================================================================
# HSIC (numpy 版，用于评估)
# =========================================================================

def compute_hsic(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """HSIC (Hilbert-Schmidt Independence Criterion)

    使用 RBF 核衡量 X 与 Y 的统计独立性。

    Parameters
    ----------
    X : (N, d1) 第一组变量（如 z_int）
    Y : (N, d2) 第二组变量（如 z_ext）
    sigma : RBF 核带宽（None = median heuristic）

    Returns
    -------
    float: HSIC 值（越小越独立）
    """
    N = X.shape[0]
    if N < 3:
        return 0.0

    # RBF 核
    if sigma is None:
        dists_x = pdist(X)
        dists_y = pdist(Y)
        sigma_x = float(np.median(dists_x[dists_x > 0])) if np.any(dists_x > 0) else 1.0
        sigma_y = float(np.median(dists_y[dists_y > 0])) if np.any(dists_y > 0) else 1.0
    else:
        sigma_x = sigma_y = sigma

    K = np.exp(-squareform(pdist(X, "sqeuclidean")) / (2 * sigma_x ** 2 + 1e-10))
    L = np.exp(-squareform(pdist(Y, "sqeuclidean")) / (2 * sigma_y ** 2 + 1e-10))

    H = np.eye(N) - 1.0 / N
    hsic = np.trace(K @ H @ L @ H) / ((N - 1) ** 2)
    return float(hsic)


# =========================================================================
# 解缠质量
# =========================================================================

def compute_neighbor_predictivity(
    z: np.ndarray,
    neighbor_composition: np.ndarray,
) -> float:
    """Z 对邻居组成的预测力

    以邻居细胞类型比例为标签，Z 回归 R²。

    Parameters
    ----------
    z : (N, d) 潜变量
    neighbor_composition : (N, K) 邻居组成比例矩阵

    Returns
    -------
    float: 平均 R²
    """
    if z.shape[0] < 5:
        return 0.0

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    r2_scores = []
    for k in range(neighbor_composition.shape[1]):
        y = neighbor_composition[:, k]
        if np.std(y) < 1e-10:
            continue
        model = Ridge(alpha=1.0)
        model.fit(z_scaled, y)
        pred = model.predict(z_scaled)
        r2 = r2_score(y, pred)
        r2_scores.append(r2)

    return float(np.mean(r2_scores)) if r2_scores else 0.0


def compute_neighbor_composition(
    labels: np.ndarray,
    adj: np.ndarray,
) -> np.ndarray:
    """计算每个节点的邻居类型组成比例

    Parameters
    ----------
    labels : (N,) 整数类型标签
    adj : (N, N) 邻接矩阵

    Returns
    -------
    (N, K) 邻居组成比例矩阵
    """
    n_types = int(labels.max()) + 1
    N = len(labels)
    comp = np.zeros((N, n_types))

    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        for j in neighbors:
            comp[i, labels[j]] += 1
        comp[i] /= comp[i].sum() + 1e-10

    return comp


# =========================================================================
# 图结构指标
# =========================================================================

def compute_graph_sparsity(adjacency: np.ndarray) -> float:
    """因果图稀疏度 = |E| / (|V| * (|V|-1))"""
    K = adjacency.shape[0]
    n_edges = int(adjacency.sum())
    max_edges = K * (K - 1)
    return n_edges / max(max_edges, 1)


def compute_bootstrap_freq_stats(
    freq_matrix: np.ndarray,
    adjacency: np.ndarray,
) -> dict:
    """Bootstrap 频率统计

    Returns
    -------
    dict with: mean, median, std, min, max
    """
    vals = freq_matrix[adjacency > 0]
    if len(vals) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


# =========================================================================
# 一站式评估
# =========================================================================

def evaluate_causal(
    adjacency: np.ndarray,
    bootstrap_freq: np.ndarray,
    z_int: Optional[np.ndarray] = None,
    z_ext: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    cluster_adj: Optional[np.ndarray] = None,
    known_axis_results: Optional[dict] = None,
    falsification_results: Optional[dict] = None,
    signaling_flow_summary: Optional[dict] = None,
) -> dict:
    """一站式因果评估

    Parameters
    ----------
    adjacency : (K, K) 二值邻接
    bootstrap_freq : (K, K) 频率矩阵
    z_int, z_ext : 解缠嵌入
    labels : 节点类型标签
    cluster_adj : 节点邻接矩阵（用于邻居预测评估）
    known_axis_results : evaluate_known_axes 输出
    falsification_results : DoWhy 验证输出
    signaling_flow_summary : 信号流汇总

    Returns
    -------
    dict: 全部指标
    """
    metrics: dict = {}

    # --- 图结构 ---
    metrics["graph_sparsity"] = compute_graph_sparsity(adjacency)
    freq_stats = compute_bootstrap_freq_stats(bootstrap_freq, adjacency)
    metrics["mean_bootstrap_freq"] = freq_stats["mean"]
    metrics["median_bootstrap_freq"] = freq_stats["median"]
    metrics["n_edges"] = int(adjacency.sum())
    metrics["n_nodes"] = adjacency.shape[0]

    # --- 解缠质量 ---
    if z_int is not None and z_ext is not None:
        metrics["hsic_z_int_z_ext"] = compute_hsic(z_int, z_ext)

        if labels is not None and cluster_adj is not None:
            neighbor_comp = compute_neighbor_composition(labels, cluster_adj)
            metrics["z_ext_neighbor_r2"] = compute_neighbor_predictivity(
                z_ext, neighbor_comp
            )
            metrics["z_int_neighbor_r2"] = compute_neighbor_predictivity(
                z_int, neighbor_comp
            )

    # --- 已知轴 ---
    if known_axis_results is not None:
        metrics["known_axis_recall"] = known_axis_results.get(
            "known_axis_recall", 0.0
        )
        metrics["direction_accuracy"] = known_axis_results.get(
            "direction_accuracy", 0.0
        )
        metrics["n_axes_tested"] = known_axis_results.get("n_axes_tested", 0)

    # --- DoWhy 验证 ---
    if falsification_results is not None:
        metrics["falsification_pvalue"] = falsification_results.get(
            "mean_pvalue", float("nan")
        )
        metrics["structure_rejected"] = falsification_results.get(
            "rejected", False
        )

    # --- 信号流 ---
    if signaling_flow_summary is not None:
        metrics["signaling_flow_completeness"] = signaling_flow_summary.get(
            "flow_completeness", 0.0
        )
        metrics["n_complete_flows"] = signaling_flow_summary.get(
            "n_complete_flows", 0
        )

    return metrics
