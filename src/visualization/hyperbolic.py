"""Phase 1 可视化：双曲嵌入与单细胞参考图谱

提供以下图类型：
- Poincaré 圆盘 2D 嵌入图（按细胞类型/状态着色）
- 3D 双曲投影图（Lorentz hyperboloid 可选视角）
- 径向梯度分支图（沿 Poincaré 半径分层验证 H1.3）
- 欧氏 baseline 对照面板（PCA/UMAP vs Hyperbolic 并排）
- 指标面板（Distortion、ARI/NMI、Silhouette、Branch Purity）

所有函数在数据就绪前返回 *带占位信息的 Figure*，
数据就绪后传入真实张量即可产出正式图。

对齐文档:
    - docs/technical_roadmap.md §3
    - docs/evaluation_suite.md §1
参考范式: scDHMap, TopoLa
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot_style import (
    PALETTE_CATEGORICAL,
    PALETTE_CELLTYPE,
    CMAP_PSEUDOTIME,
    CMAP_PSEUDOTIME_ALT,
    CMAP_SEQUENTIAL,
    apply_style,
    create_figure,
    get_color_mapping,
    save_figure,
    add_watermark,
)

apply_style()


# =========================================================================
# Poincaré 圆盘 2D 嵌入图
# =========================================================================

def plot_poincare_disk(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    label_name: str = "Cell Type",
    radial_values: np.ndarray | None = None,
    radial_name: str = "Radius",
    palette: list[str] | dict[str, str] | None = None,
    point_size: float = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """在 Poincaré 圆盘上绘制 2D 双曲嵌入

    Parameters
    ----------
    embeddings : (N, 2)
        Poincaré 球坐标（范数 < 1）
    labels : (N,), optional
        分类标签（细胞类型、聚类等），用于着色
    radial_values : (N,), optional
        连续值（如半径/分化程度），用于 colorbar 着色（与 labels 互斥）
    palette : 色板
    save_path : 保存路径

    Notes
    -----
    当 `embeddings` 为 None 或空时，绘制带说明文字的占位图。
    """
    fig, ax = create_figure(figsize=(9, 9))

    # 绘制单位圆边界
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.2, alpha=0.6)
    ax.fill(np.cos(theta), np.sin(theta), color="#F8F8F8", zorder=0)

    # 半径参考圈
    for r in [0.25, 0.5, 0.75]:
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="#DDDDDD", linewidth=0.5, linestyle="--", zorder=0)
        ax.text(r + 0.02, 0.02, f"r={r}", fontsize=7, color="#AAAAAA")

    if embeddings is None or len(embeddings) == 0:
        ax.text(0, 0, "Awaiting\nembedding data",
                ha="center", va="center", fontsize=16, color="#999999",
                style="italic")
    elif radial_values is not None:
        cmap_name = CMAP_SEQUENTIAL
        rn = str(radial_name).lower()
        if ("pseudo" in rn) or ("time" in rn) or ("trajectory" in rn):
            cmap_name = CMAP_PSEUDOTIME
            if len(np.unique(radial_values)) > 256:
                cmap_name = CMAP_PSEUDOTIME_ALT
        sc = ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=radial_values, cmap=cmap_name,
            s=point_size, edgecolors="none", alpha=0.8, zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(radial_name, fontsize=10)
    elif labels is not None:
        unique = sorted(set(labels))
        cmap = get_color_mapping(unique, palette or PALETTE_CELLTYPE)
        for lab in unique:
            mask = labels == lab
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                s=point_size, c=cmap[lab], label=lab,
                edgecolors="none", alpha=0.8, zorder=2,
            )
        ncol = max(1, len(unique) // 10)
        ax.legend(
            title=label_name, loc="upper left", bbox_to_anchor=(1.02, 1),
            markerscale=2.5, fontsize=8, title_fontsize=9, ncol=ncol,
        )
    else:
        ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            s=point_size, c="#4477AA", edgecolors="none", alpha=0.7, zorder=2,
        )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title("Poincaré Disk Embedding", fontsize=14)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "poincare_disk"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 3D Hyperboloid 投影图
# =========================================================================

def plot_hyperboloid_3d(
    embeddings_3d: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    label_name: str = "Cell Type",
    palette: list[str] | dict[str, str] | None = None,
    elev: float = 25,
    azim: float = 45,
    save_path: str | None = None,
) -> plt.Figure:
    """Lorentz Hyperboloid 3D 可视化

    Parameters
    ----------
    embeddings_3d : (N, 3)
        Lorentz 坐标 (t, x1, x2)，满足 -t² + x1² + x2² = -1
    labels : (N,), optional
        分类标签
    elev, azim : 3D 观察角度
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制参考曲面
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2, 30)
    u, v = np.meshgrid(u, v)
    x_surf = np.sinh(v) * np.cos(u)
    y_surf = np.sinh(v) * np.sin(u)
    z_surf = np.cosh(v)
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.05, color="#CCCCCC")

    if embeddings_3d is None or len(embeddings_3d) == 0:
        ax.text2D(0.5, 0.5, "Awaiting\n3D embedding data",
                  ha="center", va="center", fontsize=14, color="#999999",
                  style="italic", transform=ax.transAxes)
    elif labels is not None:
        unique = sorted(set(labels))
        cmap = get_color_mapping(unique, palette or PALETTE_CELLTYPE)
        for lab in unique:
            mask = labels == lab
            ax.scatter(
                embeddings_3d[mask, 1], embeddings_3d[mask, 2], embeddings_3d[mask, 0],
                s=3, c=cmap[lab], label=lab, alpha=0.7,
            )
        ax.legend(title=label_name, fontsize=7, title_fontsize=8, markerscale=2)
    else:
        ax.scatter(
            embeddings_3d[:, 1], embeddings_3d[:, 2], embeddings_3d[:, 0],
            s=3, c="#4477AA", alpha=0.6,
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$t$ (time-like)")
    ax.set_title("Lorentz Hyperboloid Embedding")
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        save_figure(fig, save_path, config={"chart": "hyperboloid_3d"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 径向梯度分支图
# =========================================================================

def plot_radial_branch(
    embeddings: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    label_name: str = "Cell State",
    n_bins: int = 10,
    palette: list[str] | dict[str, str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """沿 Poincaré 半径方向分层的分支纯度堆叠面积图

    验证假设 H1.3: 细胞分化状态沿径向有序排列。

    Parameters
    ----------
    embeddings : (N, 2)
        Poincaré 坐标
    labels : (N,)
        细胞状态标签
    n_bins : int
        径向分层数
    """
    fig, axes = create_figure(1, 2, figsize=(14, 5))

    if embeddings is None or labels is None:
        for ax in axes:
            ax.text(0.5, 0.5, "Awaiting data",
                    ha="center", va="center", fontsize=14, color="#999999",
                    transform=ax.transAxes, style="italic")
        axes[0].set_title("Radial Composition (stacked area)")
        axes[1].set_title("Branch Purity per Radial Bin")
    else:
        radii = np.linalg.norm(embeddings, axis=1)
        bins = np.linspace(0, radii.max() + 1e-6, n_bins + 1)
        bin_idx = np.digitize(radii, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        unique_labels = sorted(set(labels))
        cmap = get_color_mapping(unique_labels, palette or PALETTE_CELLTYPE)

        # 堆叠面积数据
        fractions = np.zeros((n_bins, len(unique_labels)))
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                for j, lab in enumerate(unique_labels):
                    fractions[b, j] = (labels[mask] == lab).sum() / mask.sum()

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Panel 1: 堆叠面积
        ax = axes[0]
        colors = [cmap[lab] for lab in unique_labels]
        ax.stackplot(bin_centers, fractions.T, labels=unique_labels, colors=colors, alpha=0.85)
        ax.set_xlabel("Poincaré Radius")
        ax.set_ylabel("Fraction")
        ax.set_title("Radial Composition (stacked area)")
        ax.legend(loc="upper left", fontsize=7, ncol=max(1, len(unique_labels) // 6))

        # Panel 2: 分支纯度
        ax = axes[1]
        purity = fractions.max(axis=1)
        bar_colors = [cmap[unique_labels[fractions[b].argmax()]] for b in range(n_bins)]
        ax.bar(bin_centers, purity, width=(bins[1] - bins[0]) * 0.8,
               color=bar_colors, edgecolor="white")
        ax.axhline(0.7, color="#EE6677", linestyle="--", linewidth=1, label="Purity threshold = 0.7")
        ax.set_xlabel("Poincaré Radius")
        ax.set_ylabel("Branch Purity")
        ax.set_title("Branch Purity per Radial Bin")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle("Radial Gradient & Branch Purity (H1.3 Validation)", fontsize=13, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "radial_branch"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 欧氏 baseline 对照面板
# =========================================================================

def plot_baseline_comparison(
    embeddings_dict: dict[str, np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    label_name: str = "Cell Type",
    palette: list[str] | dict[str, str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """并排对比多种嵌入方法

    Parameters
    ----------
    embeddings_dict : dict
        键: 方法名 (e.g. "PCA+UMAP", "scVI", "Poincaré"), 值: (N, 2) 坐标
    labels : (N,)
        分类标签
    """
    if embeddings_dict is None:
        embeddings_dict = {
            "PCA + UMAP": None,
            "scVI Latent": None,
            "Poincaré (Ours)": None,
        }

    n_methods = len(embeddings_dict)
    fig, axes = create_figure(1, n_methods, figsize=(6 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    unique = sorted(set(labels)) if labels is not None else []
    cmap = get_color_mapping(unique, palette or PALETTE_CELLTYPE) if unique else {}

    for ax, (method, emb) in zip(axes, embeddings_dict.items()):
        if emb is None or len(emb) == 0:
            ax.text(0.5, 0.5, "Awaiting data",
                    ha="center", va="center", fontsize=12, color="#999999",
                    transform=ax.transAxes, style="italic")
        elif labels is not None:
            for lab in unique:
                mask = labels == lab
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=3, c=cmap[lab], label=lab, alpha=0.7, edgecolors="none")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=3, c="#4477AA", alpha=0.6, edgecolors="none")

        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")

        # Poincaré 圆盘边界
        if "poincar" in method.lower():
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.4)

    # 共享图例（最后一个子图）
    if unique and labels is not None:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=cmap[lab], label=lab) for lab in unique]
        fig.legend(
            handles=handles, title=label_name,
            loc="center right", bbox_to_anchor=(1.12, 0.5),
            fontsize=7, title_fontsize=8,
        )

    fig.suptitle("Embedding Method Comparison (Euclidean vs Hyperbolic)", fontsize=14, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "baseline_comparison"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 指标仪表盘（Distortion, ARI, NMI, Silhouette, Branch Purity）
# =========================================================================

def plot_embedding_metrics_dashboard(
    metrics: dict[str, float] | None = None,
    baseline_metrics: dict[str, dict[str, float]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """嵌入质量指标 Dashboard

    Parameters
    ----------
    metrics : dict
        当前模型的指标，键如 "distortion", "ari", "nmi", "silhouette", "branch_purity"
    baseline_metrics : dict[str, dict]
        Baseline 方法名 -> 指标字典
    """
    # 默认占位
    metric_names = ["Distortion ↓", "ARI ↑", "NMI ↑", "Silhouette ↑", "Branch Purity ↑"]
    metric_keys = ["distortion", "ari", "nmi", "silhouette", "branch_purity"]
    thresholds = [0.2, 0.5, 0.5, 0.3, 0.7]  # 期望阈值
    invert = [True, False, False, False, False]  # True = 越小越好

    fig, axes = create_figure(1, len(metric_keys), figsize=(4 * len(metric_keys), 4))

    for i, (ax, name, key, thresh, inv) in enumerate(
        zip(axes, metric_names, metric_keys, thresholds, invert)
    ):
        # 收集数据
        values = []
        method_labels = []

        if metrics is not None and key in metrics:
            values.append(metrics[key])
            method_labels.append("HyperSCA")

        if baseline_metrics:
            for bname, bm in baseline_metrics.items():
                if key in bm:
                    values.append(bm[key])
                    method_labels.append(bname)

        if not values:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=14, color="#999999", transform=ax.transAxes, style="italic")
        else:
            colors = []
            for j, v in enumerate(values):
                if j == 0:  # HyperSCA
                    ok = v <= thresh if inv else v >= thresh
                    colors.append("#228833" if ok else "#EE6677")
                else:
                    colors.append("#BBBBBB")

            ax.barh(range(len(values)), values, color=colors, edgecolor="white", height=0.6)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(method_labels, fontsize=9)

            # 阈值线
            ax.axvline(thresh, color="#CCBB44", linestyle="--", linewidth=1.2,
                       label=f"Target = {thresh}")
            ax.legend(fontsize=7)

            for j, v in enumerate(values):
                ax.text(v + 0.01, j, f"{v:.3f}", va="center", fontsize=9)

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlim(0, None)

    fig.suptitle("Step 1 — Embedding Quality Metrics", fontsize=14, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "embedding_metrics_dashboard"})
    else:
        plt.close(fig)
    return fig
