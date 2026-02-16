"""嵌入质量评估指标

提供 Phase 1 双曲嵌入的定量评估:
- Distortion Score
- delta-Hyperbolicity (Gromov)
- ARI / NMI (聚类一致性)
- Silhouette Score
- Triplet Accuracy (距离序保持)
- 一站式 evaluate_embedding()
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.cluster import KMeans


def compute_distortion(
    embeddings: np.ndarray,
    dist_original: np.ndarray,
    dist_embedded: Optional[np.ndarray] = None,
    sample_size: int = 2000,
) -> float:
    """距离畸变评分

    D = mean(|d_emb(i,j) / d_orig(i,j) - 1|)  for sampled pairs

    Parameters
    ----------
    embeddings : (N, d) Poincare 嵌入
    dist_original : (N, N) 原始空间距离矩阵
    dist_embedded : (N, N) 嵌入空间距离矩阵（None 则自动计算 Poincare 距离）
    sample_size : int
        随机采样的节点对数

    Returns
    -------
    float
        畸变分数（越小越好）
    """
    N = len(embeddings)

    if dist_embedded is None:
        # 计算 Poincare 距离
        from src.models.hyperbolic.poincare import poincare_distance
        import torch
        emb_t = torch.tensor(embeddings, dtype=torch.float64)
        dist_embedded = np.zeros((N, N))
        for i in range(N):
            dist_embedded[i] = poincare_distance(
                emb_t[i:i+1].expand(N, -1), emb_t
            ).numpy()

    # 随机采样节点对
    rng = np.random.default_rng(42)
    n_pairs = min(sample_size, N * (N - 1) // 2)
    pairs = rng.choice(N * (N - 1) // 2, size=n_pairs, replace=False)

    # 解码对索引
    distortion_sum = 0.0
    count = 0
    for p in pairs:
        # 将线性索引转为 (i, j) 对
        i = int((-1 + np.sqrt(1 + 8 * p)) / 2) + 1
        j = p - i * (i - 1) // 2
        if i >= N or j >= N:
            continue

        d_orig = dist_original[i, j]
        d_emb = dist_embedded[i, j]
        if d_orig > 1e-10:
            distortion_sum += abs(d_emb / d_orig - 1.0)
            count += 1

    return distortion_sum / max(count, 1)


def compute_ari(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """聚类 ARI (Adjusted Rand Index)

    在嵌入空间做 KMeans 聚类，与真实标签比较。

    Returns
    -------
    float
        ARI [-1, 1]，越高越好
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels_true))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(embeddings)
    return float(adjusted_rand_score(labels_true, labels_pred))


def compute_nmi(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """聚类 NMI (Normalized Mutual Information)

    Returns
    -------
    float
        NMI [0, 1]，越高越好
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels_true))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(embeddings)
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
) -> float:
    """轮廓系数

    Returns
    -------
    float
        Silhouette [-1, 1]，越高越好
    """
    if len(embeddings) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    return float(silhouette_score(embeddings, labels))


def compute_triplet_accuracy(
    embeddings: np.ndarray,
    dist_original: np.ndarray,
    n_triplets: int = 10000,
) -> float:
    """三元组距离序保持率

    随机采样 (i, j, k)，检查 d_orig(i,j) < d_orig(i,k) 是否
    在嵌入空间中保持。

    Returns
    -------
    float
        准确率 [0, 1]
    """
    N = len(embeddings)
    rng = np.random.default_rng(42)

    # 嵌入距离矩阵（欧氏近似，用于大规模）
    from sklearn.metrics import pairwise_distances
    dist_emb = pairwise_distances(embeddings)

    correct = 0
    total = 0
    for _ in range(n_triplets):
        i, j, k = rng.choice(N, size=3, replace=False)
        d_orig_ij = dist_original[i, j]
        d_orig_ik = dist_original[i, k]
        d_emb_ij = dist_emb[i, j]
        d_emb_ik = dist_emb[i, k]

        if abs(d_orig_ij - d_orig_ik) < 1e-10:
            continue

        if (d_orig_ij < d_orig_ik) == (d_emb_ij < d_emb_ik):
            correct += 1
        total += 1

    return correct / max(total, 1)


def evaluate_embedding(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    dist_original: Optional[np.ndarray] = None,
) -> dict:
    """一站式嵌入质量评估

    Parameters
    ----------
    embeddings : (N, d) Poincare 嵌入
    labels : (N,) 细胞类型标签
    dist_original : (N, N) 原始距离矩阵（可选）

    Returns
    -------
    dict
        所有指标
    """
    metrics = {
        "n_cells": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
        "embedding_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
    }

    if labels is not None:
        n_clusters = len(np.unique(labels))
        metrics["n_clusters"] = n_clusters
        metrics["ari"] = compute_ari(embeddings, labels, n_clusters)
        metrics["nmi"] = compute_nmi(embeddings, labels, n_clusters)

        if len(np.unique(labels)) > 1:
            metrics["silhouette"] = compute_silhouette(embeddings, labels)

    if dist_original is not None:
        metrics["distortion"] = compute_distortion(
            embeddings, dist_original
        )
        metrics["triplet_accuracy"] = compute_triplet_accuracy(
            embeddings, dist_original
        )

    return metrics
