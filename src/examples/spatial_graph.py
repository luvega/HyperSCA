"""Example 02: Visium 空间邻域图构建

提供 kNN 图构建和多种空间可视化模板：
- 基础空间图（散点 + kNN 边）
- 增强版空间图（按 patient / 组织区 / 聚类标签着色）
- 边距离分布图
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.utils.plot_style import (
    PALETTE_CATEGORICAL,
    CMAP_SPATIAL,
    apply_style,
    create_figure,
    get_color_mapping,
    save_figure,
    add_watermark,
)

apply_style()


# =========================================================================
# 数据处理
# =========================================================================

def filter_in_tissue(positions: pd.DataFrame) -> pd.DataFrame:
    """保留 in_tissue == 1 的 spot"""
    return positions[positions["in_tissue"] == 1].copy().reset_index(drop=True)


def build_knn_edges(
    coords: np.ndarray, k: int = 6
) -> pd.DataFrame:
    """基于空间坐标构建 k-NN 边表

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) 空间坐标
    k : int
        邻居数

    Returns
    -------
    pd.DataFrame
        列: source, target, distance
    """
    tree = KDTree(coords)
    dists, indices = tree.query(coords, k=k + 1)  # 含自身
    rows = []
    for i in range(len(coords)):
        for j_idx in range(1, k + 1):  # 跳过自身
            rows.append({
                "source": i,
                "target": indices[i, j_idx],
                "distance": dists[i, j_idx],
            })
    return pd.DataFrame(rows)


def graph_stats(edges: pd.DataFrame, n_nodes: int) -> dict:
    """计算图的基础统计"""
    return {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "mean_distance": float(edges["distance"].mean()),
        "max_distance": float(edges["distance"].max()),
        "min_distance": float(edges["distance"].min()),
    }


# =========================================================================
# 可视化 — 基础空间图（向后兼容）
# =========================================================================

def plot_spatial_graph(
    coords: np.ndarray,
    edges: pd.DataFrame,
    save_path: str | None = None,
    max_edges: int = 5000,
):
    """绘制空间散点 + kNN 边覆盖图（基础版，向后兼容）"""
    fig, ax = create_figure(figsize=(8, 8))

    # 使用 LineCollection 加速绘制
    edge_sample = edges if len(edges) <= max_edges else edges.sample(max_edges, random_state=42)
    segments = []
    for _, row in edge_sample.iterrows():
        s, t = int(row["source"]), int(row["target"])
        segments.append([coords[s], coords[t]])
    lc = LineCollection(segments, colors="lightgray", linewidths=0.3, zorder=1)
    ax.add_collection(lc)

    ax.scatter(coords[:, 0], coords[:, 1], s=3, c="steelblue", zorder=2)
    ax.set_aspect("equal")
    ax.set_title(f"Visium Spatial Graph (k-NN, k={len(edges) // max(1, len(set(edges['source'])))}")
    ax.invert_yaxis()
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "spatial_graph_basic"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 增强版空间图（按分组着色）
# =========================================================================

def plot_spatial_graph_colored(
    coords: np.ndarray,
    edges: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    label_name: str = "Group",
    save_path: str | None = None,
    max_edges: int = 5000,
    point_size: float = 6,
    palette: list[str] | dict[str, str] | None = None,
    show_edges: bool = True,
) -> plt.Figure:
    """增强版空间图 — 按分类标签着色

    Parameters
    ----------
    coords : (N, 2) 空间坐标
    edges : kNN 边表
    labels : 长度 N 的分类标签（如 patient, region, cluster）
    label_name : 图例标题
    save_path : 保存路径
    max_edges : 最大绘制边数
    point_size : 点大小
    palette : 色板
    show_edges : 是否绘制边
    """
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels))
    cmap = get_color_mapping(unique_labels, palette)
    colors = [cmap[lab] for lab in labels]

    fig, ax = create_figure(figsize=(9, 9))

    # 边
    if show_edges:
        edge_sample = edges if len(edges) <= max_edges else edges.sample(max_edges, random_state=42)
        segments = []
        for _, row in edge_sample.iterrows():
            s, t = int(row["source"]), int(row["target"])
            segments.append([coords[s], coords[t]])
        lc = LineCollection(segments, colors="#E0E0E0", linewidths=0.2, zorder=1, alpha=0.5)
        ax.add_collection(lc)

    # 节点（按组绘制以生成图例）
    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=point_size, c=cmap[lab], label=lab, zorder=2,
            edgecolors="none", alpha=0.85,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"Visium Spatial Graph — colored by {label_name}")

    # 自适应图例
    ncol = max(1, len(unique_labels) // 12)
    ax.legend(
        title=label_name, loc="upper left", bbox_to_anchor=(1.02, 1),
        markerscale=2, fontsize=8, title_fontsize=9, ncol=ncol,
        frameon=True, fancybox=True, framealpha=0.8,
    )
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "spatial_graph_colored", "label": label_name})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 连续值空间热图
# =========================================================================

def plot_spatial_heatmap(
    coords: np.ndarray,
    values: np.ndarray,
    value_name: str = "Value",
    cmap: str = CMAP_SPATIAL,
    save_path: str | None = None,
    point_size: float = 8,
) -> plt.Figure:
    """在空间坐标上展示连续值热图

    Parameters
    ----------
    coords : (N, 2) 空间坐标
    values : 长度 N 的连续值
    value_name : colorbar 标签
    cmap : colormap 名
    """
    fig, ax = create_figure(figsize=(9, 9))

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=point_size, c=values, cmap=cmap, zorder=2,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(value_name, fontsize=10)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"Spatial Heatmap — {value_name}")
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "spatial_heatmap", "value": value_name})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 边距离分布图
# =========================================================================

def plot_edge_distance_distribution(
    edges: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """绘制 kNN 边距离的分布直方图 + KDE"""
    fig, ax = create_figure(figsize=(8, 4))

    ax.hist(
        edges["distance"], bins=80,
        color="#4477AA", edgecolor="white", alpha=0.7,
        density=True, label="Histogram",
    )

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(edges["distance"])
    x_range = np.linspace(edges["distance"].min(), edges["distance"].max(), 200)
    ax.plot(x_range, kde(x_range), color="#EE6677", linewidth=2, label="KDE")

    ax.axvline(
        edges["distance"].mean(), color="#228833",
        linestyle="--", linewidth=1.2, label=f"Mean = {edges['distance'].mean():.1f}",
    )

    ax.set_xlabel("Edge Distance (px)")
    ax.set_ylabel("Density")
    ax.set_title("k-NN Edge Distance Distribution")
    ax.legend()
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "edge_distance_dist"})
    else:
        plt.close(fig)
    return fig
