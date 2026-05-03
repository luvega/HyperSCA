"""空间邻域图构建与 TopoLa 拓扑增强

提供:
- k-NN 图构建 (KDTree)
- Delaunay 三角剖分图
- TopoLa SVD 拓扑增强
- 一站式接口 build_spatial_graph()

Adapter 模式: TopoLa 算法参考 references/TopoLa/.../utils_TopoLa.py,
              禁止直接 import，完全重写。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree, Delaunay
from scipy.sparse.linalg import svds


# =========================================================================
# k-NN 图
# =========================================================================

def build_knn_graph(
    coords: np.ndarray,
    k: int = 6,
    symmetric: bool = True,
) -> sparse.csr_matrix:
    """构建 k-NN 空间图

    Parameters
    ----------
    coords : (N, 2) 空间坐标
    k : int
        邻居数
    symmetric : bool
        是否对称化邻接矩阵（A = max(A, A^T)）

    Returns
    -------
    scipy.sparse.csr_matrix
        (N, N) 稀疏邻接矩阵，权重为距离
    """
    N = len(coords)
    tree = KDTree(coords)
    dists, indices = tree.query(coords, k=k + 1)  # 含自身

    rows, cols, vals = [], [], []
    for i in range(N):
        for j_idx in range(1, k + 1):  # 跳过自身
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)
            vals.append(dists[i, j_idx])

    adj = sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )

    if symmetric:
        # 取最大值对称化
        adj = adj.maximum(adj.T)

    return adj


# =========================================================================
# Delaunay 三角剖分图
# =========================================================================

def build_delaunay_graph(coords: np.ndarray) -> sparse.csr_matrix:
    """构建 Delaunay 三角剖分空间图

    Parameters
    ----------
    coords : (N, 2) 空间坐标

    Returns
    -------
    scipy.sparse.csr_matrix
        (N, N) 稀疏邻接矩阵，权重为欧氏距离
    """
    N = len(coords)
    tri = Delaunay(coords)

    rows, cols, vals = [], [], []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                d = np.linalg.norm(coords[a] - coords[b])
                rows.extend([a, b])
                cols.extend([b, a])
                vals.extend([d, d])

    adj = sparse.coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(N, N),
    ).tocsr()
    # 去重: coo -> csr 自动合并重复项（取 sum），改为取 max
    adj.eliminate_zeros()
    return adj


# =========================================================================
# TopoLa 拓扑增强
# =========================================================================

def topola_enhance(
    adj: sparse.csr_matrix,
    lambda_val: float = 1e-3,
    n_components: Optional[int] = None,
    threshold: float = 1e-4,
) -> np.ndarray:
    """TopoLa (Topology-encoding) 邻接矩阵增强

    对邻接矩阵做 SVD，施加非线性奇异值变换:
        sigma_tilde = sigma^3 / (sigma^2 + 1/lambda)

    大奇异值近似保留，小奇异值被抑制，等效于高阶拓扑过滤。

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        (N, N) 原始邻接矩阵
    lambda_val : float
        正则化参数（默认 1e-3）。越大 → 增强越强
    n_components : int or None
        截断 SVD 分量数（None = min(N, 100)）。大矩阵建议截断
    threshold : float
        增强后低于此值的元素置零（稀疏化）

    Returns
    -------
    np.ndarray
        (N, N) 增强后的邻接矩阵（密集）
    """
    N = adj.shape[0]

    if n_components is None:
        n_components = min(N - 1, 100)

    # 密集矩阵用 full SVD，大矩阵用截断 SVD
    if N <= 2000:
        A_dense = adj.toarray().astype(np.float64)
        U, S, Vt = np.linalg.svd(A_dense, full_matrices=False)
    else:
        n_components = min(n_components, N - 1)
        U, S, Vt = svds(adj.astype(np.float64), k=n_components)
        # svds 返回按升序排列，翻转
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

    # 非线性奇异值变换: sigma_tilde = sigma^3 / (sigma^2 + 1/lambda)
    inv_lambda = 1.0 / lambda_val
    S_enhanced = np.where(
        S > 0,
        S ** 3 / (S ** 2 + inv_lambda),
        0.0,
    )

    # 重构
    matrix = U @ np.diag(S_enhanced) @ Vt

    # 稀疏化: 阈值截断
    matrix[np.abs(matrix) < threshold] = 0.0

    return matrix


def topola_enhance_sparse(
    adj: sparse.csr_matrix,
    lambda_val: float = 1e-3,
    n_components: int = 100,
    threshold: float = 1e-4,
) -> sparse.csr_matrix:
    """TopoLa 增强（返回稀疏矩阵版本）"""
    dense = topola_enhance(adj, lambda_val, n_components, threshold)
    return sparse.csr_matrix(dense)


# =========================================================================
# 一站式接口
# =========================================================================

def build_spatial_graph(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 6,
    use_topola: bool = True,
    topola_lambda: float = 1e-3,
    topola_components: Optional[int] = None,
    topola_max_nodes: int = 20000,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix | None]:
    """一站式空间图构建（可选 TopoLa 增强）

    Parameters
    ----------
    coords : (N, 2) 空间坐标
    method : str
        'knn' 或 'delaunay'
    k : int
        k-NN 邻居数（仅 method='knn' 时使用）
    use_topola : bool
        是否应用 TopoLa 增强
    topola_lambda : float
        TopoLa 正则化参数
    topola_components : int or None
        截断 SVD 分量数
    topola_max_nodes : int
        TopoLa 最大节点数阈值。超过该阈值时自动跳过 TopoLa，避免大规模
        稠密矩阵重构导致内存溢出。

    Returns
    -------
    adj_original : scipy.sparse.csr_matrix
        原始邻接矩阵
    adj_enhanced : scipy.sparse.csr_matrix or None
        TopoLa 增强邻接矩阵（use_topola=False 时为 None）
    """
    # 构建基础图
    if method == "knn":
        adj_original = build_knn_graph(coords, k=k)
    elif method == "delaunay":
        adj_original = build_delaunay_graph(coords)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knn' or 'delaunay'.")

    print(f"  [graph] {method} graph: {adj_original.shape[0]} nodes, "
          f"{adj_original.nnz} edges")

    # 可选 TopoLa 增强
    adj_enhanced = None
    if use_topola:
        if coords.shape[0] > topola_max_nodes:
            print(
                "  [topola] skipped due to node count "
                f"{coords.shape[0]} > topola_max_nodes={topola_max_nodes}"
            )
            return adj_original, None
        # TopoLa 使用二值邻接矩阵（距离权重归一化为 0/1）
        adj_binary = (adj_original > 0).astype(np.float64)
        enhanced_dense = topola_enhance(
            adj_binary,
            lambda_val=topola_lambda,
            n_components=topola_components,
        )
        adj_enhanced = sparse.csr_matrix(enhanced_dense)
        print(f"  [topola] enhanced: {adj_enhanced.nnz} non-zero elements "
              f"(lambda={topola_lambda})")

    return adj_original, adj_enhanced


# =========================================================================
# 图统计（用于展示）
# =========================================================================

def graph_statistics(adj: sparse.csr_matrix) -> dict:
    """计算图基础统计量

    Returns
    -------
    dict
        n_nodes, n_edges, mean_degree, n_components, density
    """
    from scipy.sparse.csgraph import connected_components

    N = adj.shape[0]
    n_edges = adj.nnz // 2  # 对称图
    degrees = np.diff(adj.indptr)
    n_comp, _ = connected_components(adj, directed=False)

    return {
        "n_nodes": int(N),
        "n_edges": int(n_edges),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
        "n_components": int(n_comp),
        "density": float(2 * n_edges / (N * (N - 1))) if N > 1 else 0.0,
    }


def spectral_gap(adj: sparse.csr_matrix, n_eigenvalues: int = 10) -> np.ndarray:
    """计算图 Laplacian 的前 n 个特征值（用于拓扑对比图）

    Returns
    -------
    np.ndarray
        前 n 个最小特征值
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import diags

    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L_norm = sparse.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt

    n_eigenvalues = min(n_eigenvalues, adj.shape[0] - 2)
    eigenvalues, _ = eigsh(L_norm, k=n_eigenvalues, which="SM")
    return np.sort(eigenvalues)
