"""阶段 3: 空间传播一致性评估指标。

对齐 docs/evaluation_suite.md §4:
    - Moran's I（扰动前后）
    - Gradient Decay R²
    - Characteristic Length ℓ
    - Propagation Depth
    - Spatial-Causal Correlation
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


def morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> float:
    """计算 Moran's I 空间自相关指标。

    Parameters
    ----------
    values : (N,) 基因表达或效应值
    coords : (N, 2) 空间坐标
    k : 近邻数
    """
    n = len(values)
    if n < 3:
        return 0.0

    # 构建 k-NN 权重矩阵
    dists = cdist(coords, coords, metric="euclidean")
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors = np.argsort(dists[i])[:k + 1]
        for j in neighbors:
            if j != i:
                W[i, j] = 1.0

    # 行标准化
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    z = values - values.mean()
    S0 = W.sum()
    if S0 < 1e-12 or np.var(z) < 1e-12:
        return 0.0

    numerator = n * float(z @ W @ z)
    denominator = S0 * float(z @ z)
    return numerator / max(denominator, 1e-12)


def delta_morans_i(
    observed_expr: np.ndarray,
    counterfactual_expr: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> dict[str, float]:
    """KO 前后 Moran's I 变化。"""
    i_obs = morans_i(observed_expr, coords, k)
    i_cf = morans_i(counterfactual_expr, coords, k)
    return {
        "morans_i_obs": i_obs,
        "morans_i_cf": i_cf,
        "delta_morans_i": i_cf - i_obs,
    }


def gradient_decay_r2(
    effect_magnitudes: np.ndarray,
    distances: np.ndarray,
) -> dict[str, float]:
    """拟合指数衰减模型并返回 R² 与特征长度 ℓ。

    Parameters
    ----------
    effect_magnitudes : (M,)
    distances : (M,)
    """
    mask = effect_magnitudes > 1e-12
    if mask.sum() < 3:
        return {"gradient_decay_r2": 0.0, "characteristic_length": 0.0}

    d = distances[mask]
    m = effect_magnitudes[mask]
    log_m = np.log(np.clip(m, 1e-12, None))

    try:
        coeffs = np.polyfit(d, log_m, 1)
        slope = coeffs[0]
        ell = -1.0 / slope if abs(slope) > 1e-12 else 0.0
        ell = max(ell, 0.0)

        pred = coeffs[0] * d + coeffs[1]
        ss_res = np.sum((log_m - pred) ** 2)
        ss_tot = np.sum((log_m - log_m.mean()) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    except (np.linalg.LinAlgError, ValueError):
        ell = 0.0
        r2 = 0.0

    return {"gradient_decay_r2": float(r2), "characteristic_length": float(ell)}


def propagation_depth(
    bfs_layers: list[dict],
    threshold: float = 0.01,
) -> int:
    """计算扰动显著影响的最大跳数。"""
    depth = 0
    for layer in bfs_layers:
        if layer.get("mean_effect", 0.0) >= threshold:
            depth = layer.get("hop", depth)
    return depth


def spatial_causal_correlation(
    causal_adj: np.ndarray,
    spatial_coords: np.ndarray,
) -> float:
    """空间距离与因果效应强度的 Spearman 相关。"""
    from scipy import stats as sp_stats

    K = causal_adj.shape[0]
    dists = cdist(spatial_coords, spatial_coords, metric="euclidean")
    spatial_vals = []
    causal_vals = []
    for i in range(K):
        for j in range(K):
            if i != j and causal_adj[i, j] > 0:
                spatial_vals.append(dists[i, j])
                causal_vals.append(causal_adj[i, j])
    if len(spatial_vals) < 3:
        return 0.0
    rho, _ = sp_stats.spearmanr(spatial_vals, causal_vals)
    return float(rho) if np.isfinite(rho) else 0.0


def evaluate_spatial_propagation(
    coords: np.ndarray,
    effect_magnitudes: np.ndarray,
    source_distances: np.ndarray,
    bfs_layers: Optional[list[dict]] = None,
    causal_adj: Optional[np.ndarray] = None,
    observed_expr: Optional[np.ndarray] = None,
    counterfactual_expr: Optional[np.ndarray] = None,
    threshold: float = 0.01,
    k: int = 6,
) -> dict[str, float]:
    """一站式空间传播评估。"""
    result: dict[str, float] = {}

    # Moran's I
    if observed_expr is not None and counterfactual_expr is not None:
        mi = delta_morans_i(observed_expr, counterfactual_expr, coords, k)
        result.update(mi)

    # Gradient decay
    gd = gradient_decay_r2(effect_magnitudes, source_distances)
    result.update(gd)

    # Propagation depth
    if bfs_layers is not None:
        result["propagation_depth"] = propagation_depth(bfs_layers, threshold)

    # Spatial-causal correlation
    if causal_adj is not None:
        result["spatial_causal_correlation"] = spatial_causal_correlation(
            causal_adj, coords
        )

    return result
