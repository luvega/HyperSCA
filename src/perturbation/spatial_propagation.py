"""阶段 3: 空间传播模拟 (Spatial Propagation)。

沿因果图的拓扑序传播扰动效应，并叠加距离衰减核:
    κ(d_ij) = exp(-d_ij² / 2ℓ²)

输出 BFS 层级统计与每细胞传播效应。

参考思想: DynPerturb 时空嵌入传播（adapter 模式）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


def _bfs_layers(
    adj: np.ndarray, source_nodes: list[int], max_depth: int
) -> list[list[int]]:
    """沿邻接矩阵做 BFS，返回每层节点列表。"""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    for s in source_nodes:
        visited[s] = True
    layers: list[list[int]] = [list(source_nodes)]
    frontier = list(source_nodes)
    for _ in range(max_depth):
        next_frontier: list[int] = []
        for node in frontier:
            children = np.where(adj[node] > 0)[0]
            for c in children:
                if not visited[c]:
                    visited[c] = True
                    next_frontier.append(int(c))
        if not next_frontier:
            break
        layers.append(next_frontier)
        frontier = next_frontier
    return layers


def propagate_perturbation(
    causal_adj: np.ndarray,
    source_nodes: list[int],
    source_delta: np.ndarray,
    spatial_coords: Optional[np.ndarray] = None,
    *,
    decay_length: float = 150.0,
    max_depth: int = 5,
    convergence_tol: float = 1e-3,
) -> dict:
    """沿因果图传播扰动效应。

    Parameters
    ----------
    causal_adj : (K, K) 因果邻接矩阵
    source_nodes : 源节点索引列表
    source_delta : (K,) 或 (K, G) 源节点初始效应向量
    spatial_coords : (K, 2) 空间坐标（可选，用于距离衰减）
    decay_length : 距离衰减核的特征尺度 ℓ
    max_depth : 最大传播深度
    convergence_tol : 收敛容差

    Returns
    -------
    dict with:
        effect : (K,) or (K, G) 每个节点的累积效应
        bfs_layers : list[dict] with keys hop, nodes, mean_effect, n_cells
        fit_params : dict with ell, r2（若有空间坐标）
    """
    K = causal_adj.shape[0]
    is_vector = source_delta.ndim == 2
    effect = np.zeros_like(source_delta, dtype=float)

    # 初始化源节点效应
    for s in source_nodes:
        effect[s] = source_delta[s]

    # BFS 传播
    layers = _bfs_layers(causal_adj, source_nodes, max_depth)

    # 空间距离矩阵
    dist_matrix = None
    if spatial_coords is not None and spatial_coords.shape[0] == K:
        dist_matrix = cdist(spatial_coords, spatial_coords, metric="euclidean")

    bfs_stats: list[dict] = []
    # layer 0 = source
    src_eff = np.abs(effect[source_nodes]).mean() if len(source_nodes) > 0 else 0.0
    if is_vector:
        src_eff = float(np.mean(np.abs(effect[source_nodes])))
    else:
        src_eff = float(np.abs(effect[source_nodes]).mean())
    bfs_stats.append({"hop": 0, "nodes": source_nodes, "mean_effect": src_eff, "n_cells": len(source_nodes)})

    for hop_idx, layer_nodes in enumerate(layers[1:], start=1):
        for node in layer_nodes:
            parents = np.where(causal_adj[:, node] > 0)[0]
            if len(parents) == 0:
                continue
            # 沿因果边汇聚上游效应
            parent_effects = effect[parents]
            edge_weights = causal_adj[parents, node]
            if is_vector:
                weighted = parent_effects * edge_weights[:, None]
            else:
                weighted = parent_effects * edge_weights

            agg = weighted.mean(axis=0) if len(parents) > 1 else weighted[0]

            # 距离衰减
            if dist_matrix is not None:
                dists = dist_matrix[parents, node]
                decay = np.exp(-(dists ** 2) / (2.0 * decay_length ** 2))
                if is_vector:
                    agg = (parent_effects * edge_weights[:, None] * decay[:, None]).sum(axis=0) / max(decay.sum(), 1e-12)
                else:
                    agg = (parent_effects * edge_weights * decay).sum() / max(decay.sum(), 1e-12)

            effect[node] = agg

        # BFS 层统计
        if layer_nodes:
            if is_vector:
                layer_eff = float(np.mean(np.abs(effect[layer_nodes])))
            else:
                layer_eff = float(np.abs(effect[layer_nodes]).mean())
            bfs_stats.append({
                "hop": hop_idx,
                "nodes": layer_nodes,
                "mean_effect": layer_eff,
                "n_cells": len(layer_nodes),
            })

        # 收敛检查
        if bfs_stats[-1]["mean_effect"] < convergence_tol:
            break

    # 梯度衰减拟合
    fit_params = {}
    if dist_matrix is not None and len(source_nodes) > 0:
        fit_params = _fit_gradient_decay(effect, dist_matrix, source_nodes, decay_length)

    return {
        "effect": effect,
        "bfs_layers": bfs_stats,
        "fit_params": fit_params,
    }


def _fit_gradient_decay(
    effect: np.ndarray,
    dist_matrix: np.ndarray,
    source_nodes: list[int],
    default_ell: float,
) -> dict:
    """拟合效应幅度 vs 空间距离的指数衰减模型。"""
    # 效应幅度
    if effect.ndim == 2:
        magnitudes = np.mean(np.abs(effect), axis=1)
    else:
        magnitudes = np.abs(effect)

    # 到最近源节点的距离
    if len(source_nodes) == 0:
        return {"ell": default_ell, "r2": 0.0}
    source_dists = dist_matrix[:, source_nodes].min(axis=1)

    # 只取有效应的节点
    mask = magnitudes > 1e-12
    if mask.sum() < 3:
        return {"ell": default_ell, "r2": 0.0}

    d = source_dists[mask]
    m = magnitudes[mask]

    # 对数线性拟合 log(m) = -d/ℓ + c
    log_m = np.log(np.clip(m, 1e-12, None))
    try:
        coeffs = np.polyfit(d, log_m, 1)
        slope = coeffs[0]
        ell = -1.0 / slope if abs(slope) > 1e-12 else default_ell
        ell = max(ell, 1.0)  # 防止负数或极小值

        # R²
        pred = coeffs[0] * d + coeffs[1]
        ss_res = np.sum((log_m - pred) ** 2)
        ss_tot = np.sum((log_m - log_m.mean()) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    except (np.linalg.LinAlgError, ValueError):
        ell = default_ell
        r2 = 0.0

    return {"ell": float(ell), "r2": float(r2)}
