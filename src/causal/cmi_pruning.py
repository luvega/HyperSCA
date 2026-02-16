"""条件互信息 (CMI) 剪枝 + Bootstrap 聚合

实现因果结构学习流程:
1. 基于偏相关的条件独立性检验（线性高斯近似）
2. PC 算法骨架学习 + V-structure 定向
3. Block Bootstrap 聚合 → 边频率矩阵
4. 阈值剪枝 → 稀疏因果图

参考实现（adapter 模式，不直接 import）:
    references/flowsig/src/flowsig/tools/_network.py
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


# =========================================================================
# 偏相关与条件独立性检验
# =========================================================================

def partial_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """计算偏相关矩阵

    通过精度矩阵（逆协方差矩阵）计算:
        rho_{ij|rest} = -P_{ij} / sqrt(P_{ii} * P_{jj})

    Parameters
    ----------
    data : (N, p) 观测矩阵

    Returns
    -------
    (p, p) 偏相关矩阵
    """
    cov = np.cov(data, rowvar=False)
    # 正则化以避免奇异
    cov += np.eye(cov.shape[0]) * 1e-6
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov)

    diag = np.sqrt(np.diag(precision))
    diag[diag < 1e-10] = 1e-10
    pcor = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcor, 1.0)
    return pcor


def partial_corr_test(
    data: np.ndarray,
    i: int,
    j: int,
    cond_set: list[int],
) -> float:
    """偏相关条件独立性检验

    H0: X_i ⊥ X_j | X_{cond_set}

    Parameters
    ----------
    data : (N, p)
    i, j : 被测变量索引
    cond_set : 条件变量索引集合

    Returns
    -------
    p-value（越大越支持独立性）
    """
    N = data.shape[0]

    if len(cond_set) == 0:
        # 简单相关
        r, pval = stats.pearsonr(data[:, i], data[:, j])
        return pval

    # 通过精度矩阵子集计算
    idx = [i, j] + list(cond_set)
    sub_data = data[:, idx]
    cov = np.cov(sub_data, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov)

    # 偏相关 rho_{0,1 | rest}
    rho = -precision[0, 1] / np.sqrt(
        abs(precision[0, 0] * precision[1, 1]) + 1e-10
    )
    rho = np.clip(rho, -0.9999, 0.9999)

    # Fisher Z 变换
    z = 0.5 * np.log((1 + rho) / (1 - rho + 1e-10))
    dof = max(N - len(cond_set) - 2, 1)
    z_stat = abs(z) * np.sqrt(dof)
    pval = 2 * (1 - stats.norm.cdf(z_stat))

    return float(pval)


# =========================================================================
# PC 算法
# =========================================================================

def pc_skeleton(
    data: np.ndarray,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> tuple[np.ndarray, dict]:
    """PC 算法骨架学习

    Parameters
    ----------
    data : (N, p)
    alpha : 条件独立性检验显著性水平
    max_cond_set : 最大条件集大小

    Returns
    -------
    adj : (p, p) 无向邻接矩阵
    sep_sets : dict, (i,j) → 使 i⊥j 的条件集
    """
    p = data.shape[1]
    adj = np.ones((p, p)) - np.eye(p)
    sep_sets: dict[tuple[int, int], list[int]] = {}

    for d in range(max_cond_set + 1):
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 0:
                    continue
                # i 的邻居（不含 j）
                neighbors = [
                    k for k in range(p)
                    if k != i and k != j and adj[i, k] == 1
                ]
                if len(neighbors) < d:
                    continue

                # 测试所有大小为 d 的条件子集
                found_independent = False
                for cond_set in combinations(neighbors, d):
                    pval = partial_corr_test(data, i, j, list(cond_set))
                    if pval > alpha:
                        adj[i, j] = adj[j, i] = 0
                        key = (min(i, j), max(i, j))
                        sep_sets[key] = list(cond_set)
                        found_independent = True
                        break
                if found_independent:
                    continue

    return adj, sep_sets


def orient_v_structures(
    adj: np.ndarray,
    sep_sets: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    """V-structure 定向

    对于三元组 i - k - j（i 与 j 不相邻），
    如果 k 不在 sep(i,j) 中，则定向为 i → k ← j

    Parameters
    ----------
    adj : (p, p) 无向邻接矩阵
    sep_sets : PC 骨架输出的分离集

    Returns
    -------
    dag : (p, p) 部分定向邻接矩阵（dag[i,j]=1 表示 i→j）
    """
    p = adj.shape[0]
    dag = adj.copy()

    for k in range(p):
        # 找 k 的所有邻居对 (i, j)，且 i, j 不直接相邻
        neighbors = [n for n in range(p) if adj[k, n] == 1]
        for idx_a in range(len(neighbors)):
            for idx_b in range(idx_a + 1, len(neighbors)):
                i, j = neighbors[idx_a], neighbors[idx_b]
                if adj[i, j] == 1:
                    continue  # i 与 j 相邻，跳过

                key = (min(i, j), max(i, j))
                sep = sep_sets.get(key, [])
                if k not in sep:
                    # V-structure: i → k ← j
                    dag[k, i] = 0  # 删除 k→i
                    dag[k, j] = 0  # 删除 k→j
                    # 保留 i→k 和 j→k

    return dag


def propagate_orientations(dag: np.ndarray) -> np.ndarray:
    """Meek 规则传播边方向

    简化版：对剩余无向边尝试定向以避免产生新 V-structure 或环。

    Parameters
    ----------
    dag : (p, p) 部分定向邻接矩阵

    Returns
    -------
    dag : (p, p) 更完整的定向
    """
    p = dag.shape[0]
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for j in range(p):
                if dag[i, j] == 1 and dag[j, i] == 1:
                    # 无向边 i--j，尝试 Meek R1:
                    # 如果存在 k→i 且 k 与 j 不相邻，则定向 i→j
                    for k in range(p):
                        if k == i or k == j:
                            continue
                        if dag[k, i] == 1 and dag[i, k] == 0:
                            # k→i 存在
                            if dag[k, j] == 0 and dag[j, k] == 0:
                                # k 与 j 不相邻
                                dag[j, i] = 0  # 定向 i→j
                                changed = True
                                break
    return dag


def pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> np.ndarray:
    """完整 PC 算法

    1. 骨架学习
    2. V-structure 定向
    3. Meek 规则传播

    Parameters
    ----------
    data : (N, p) 观测数据
    alpha : 显著性水平
    max_cond_set : 最大条件集大小

    Returns
    -------
    dag : (p, p) 因果邻接矩阵（dag[i,j]=1 表示 i→j）
    """
    adj, sep_sets = pc_skeleton(data, alpha, max_cond_set)
    dag = orient_v_structures(adj, sep_sets)
    dag = propagate_orientations(dag)

    # 对剩余无向边（dag[i,j]==1 且 dag[j,i]==1），随机定向（保留一个方向）
    p = dag.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            if dag[i, j] == 1 and dag[j, i] == 1:
                dag[j, i] = 0  # 默认 i→j

    return dag


# =========================================================================
# Bootstrap 聚合
# =========================================================================

def bootstrap_causal_discovery(
    data: np.ndarray,
    n_bootstraps: int = 100,
    alpha: float = 0.05,
    max_cond_set: int = 3,
    spatial_blocks: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Bootstrap 聚合因果发现

    对数据进行 B 次 bootstrap 采样，每次运行 PC 算法，
    对所有 DAG 的边频率进行聚合。

    Parameters
    ----------
    data : (N, p) 观测数据
    n_bootstraps : Bootstrap 次数 B
    alpha : CI 检验显著性
    max_cond_set : PC 最大条件集
    spatial_blocks : (N,) 空间块标签（用于 block bootstrap）
    seed : 随机种子
    verbose : 打印进度

    Returns
    -------
    freq_matrix : (p, p) 边频率矩阵，freq[i,j] = 边 i→j 的出现频率
    """
    rng = np.random.default_rng(seed)
    N, p = data.shape
    freq_matrix = np.zeros((p, p))

    for b in range(n_bootstraps):
        if spatial_blocks is not None:
            # Block bootstrap: 按空间块采样
            unique_blocks = np.unique(spatial_blocks)
            sampled_blocks = rng.choice(
                unique_blocks, size=len(unique_blocks), replace=True
            )
            indices = []
            for blk in sampled_blocks:
                indices.extend(np.where(spatial_blocks == blk)[0].tolist())
            indices = np.array(indices)
        else:
            # 普通 bootstrap
            indices = rng.choice(N, size=N, replace=True)

        boot_data = data[indices]
        dag = pc_algorithm(boot_data, alpha=alpha, max_cond_set=max_cond_set)
        freq_matrix += dag

        if verbose and ((b + 1) % 20 == 0 or b == 0):
            n_edges = int(dag.sum())
            print(f"  [bootstrap] {b+1}/{n_bootstraps}: edges={n_edges}")

    freq_matrix /= n_bootstraps
    return freq_matrix


def threshold_pruning(
    freq_matrix: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """阈值剪枝

    Parameters
    ----------
    freq_matrix : (p, p) 边频率矩阵
    threshold : 保留阈值（freq >= threshold 保留）

    Returns
    -------
    adjacency : (p, p) 二值邻接矩阵
    pruned_freq : (p, p) 剪枝后的频率矩阵
    """
    adjacency = (freq_matrix >= threshold).astype(float)
    pruned_freq = freq_matrix * adjacency
    return adjacency, pruned_freq
