#!/usr/bin/env python
"""阶段 1 展示图生成脚本

基于 run_step1.py 的产物，产出三组展示图:
1. graph_topology: 原始 kNN vs TopoLa 增强对比
2. embed_core: Poincare 散点 / Hyperboloid 3D / 径向分支
3. baseline_compare: PCA / UMAP / scVI vs Hyperbolic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import sparse
import scanpy as sc
from sklearn.manifold import trustworthiness

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_style import (
    apply_cns_style, create_figure, save_figure, add_watermark,
    PALETTE_CATEGORICAL, CMAP_SPATIAL, get_color_mapping,
)
from src.data.spatial_graph import graph_statistics, spectral_gap

apply_cns_style()

# 输出目录
STEP1_DIR = PROJECT_ROOT / "results" / "step1"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "step1"
for d in ["preview", "compare", "topology"]:
    (FIG_DIR / d).mkdir(parents=True, exist_ok=True)


# =========================================================================
# 1. Graph Topology: kNN vs TopoLa
# =========================================================================

def plot_graph_topology_comparison():
    """原始 kNN vs TopoLa 增强对比图（4 panel）"""
    print("[Figure] Graph topology comparison...")

    adj_orig = sparse.load_npz(STEP1_DIR / "adj_original.npz")
    adj_enh = sparse.load_npz(STEP1_DIR / "adj_enhanced.npz")

    stats_o = graph_statistics(adj_orig.astype(float))
    stats_e = graph_statistics(adj_enh)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Panel A: 度分布对比 ---
    ax = axes[0, 0]
    deg_orig = np.diff(adj_orig.indptr)
    deg_enh = np.diff(adj_enh.indptr)
    ax.hist(deg_orig, bins=30, alpha=0.7, label=f"kNN (mean={deg_orig.mean():.1f})",
            color="#4477AA", edgecolor="white")
    ax.hist(deg_enh, bins=30, alpha=0.7, label=f"TopoLa (mean={deg_enh.mean():.1f})",
            color="#EE6677", edgecolor="white")
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Count")
    ax.set_title("A. Degree Distribution")
    ax.legend()

    # --- Panel B: 边权分布对比 ---
    ax = axes[0, 1]
    w_orig = adj_orig.data[adj_orig.data > 0]
    w_enh = adj_enh.data[adj_enh.data > 0]
    ax.hist(w_orig, bins=50, alpha=0.7, label="kNN", color="#4477AA",
            density=True, edgecolor="white")
    ax.hist(np.clip(w_enh, 0, np.percentile(w_enh, 99)), bins=50, alpha=0.7,
            label="TopoLa", color="#EE6677", density=True, edgecolor="white")
    ax.set_xlabel("Edge Weight")
    ax.set_ylabel("Density")
    ax.set_title("B. Edge Weight Distribution")
    ax.legend()

    # --- Panel C: 谱特征对比 ---
    ax = axes[1, 0]
    try:
        eig_orig = spectral_gap(adj_orig.astype(float), n_eigenvalues=20)
        eig_enh = spectral_gap(adj_enh, n_eigenvalues=20)
        ax.plot(range(len(eig_orig)), eig_orig, "o-", color="#4477AA",
                markersize=5, label="kNN")
        ax.plot(range(len(eig_enh)), eig_enh, "s-", color="#EE6677",
                markersize=5, label="TopoLa")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("C. Laplacian Spectrum")
        ax.legend()
    except Exception as e:
        ax.text(0.5, 0.5, f"Spectral gap\nerror: {e}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C. Laplacian Spectrum (error)")

    # --- Panel D: 统计量对比表 ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Metric", "kNN", "TopoLa"],
        ["Nodes", f"{stats_o['n_nodes']}", f"{stats_e['n_nodes']}"],
        ["Edges", f"{stats_o['n_edges']:,}", f"{stats_e['n_edges']:,}"],
        ["Mean Degree", f"{stats_o['mean_degree']:.1f}", f"{stats_e['mean_degree']:.1f}"],
        ["Components", f"{stats_o['n_components']}", f"{stats_e['n_components']}"],
        ["Density", f"{stats_o['density']:.4f}", f"{stats_e['density']:.4f}"],
    ]
    table = ax.table(
        cellText=table_data[1:], colLabels=table_data[0],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("D. Graph Statistics", pad=20)

    fig.suptitle("Spatial Graph: kNN vs TopoLa Enhancement", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    add_watermark(axes[0, 0])

    save_figure(fig, str(FIG_DIR / "topology" / "knn_vs_topola_comparison.png"),
                config={"chart": "graph_topology_4panel"})
    print(f"  Saved: topology/knn_vs_topola_comparison.png")


# =========================================================================
# 2. Embedding Core: Poincare + 3D + Radial
# =========================================================================

def plot_embedding_core():
    """Poincare 散点 / 3D Hyperboloid / 径向分支图"""
    print("[Figure] Embedding core visualizations...")

    adata = sc.read_h5ad(str(STEP1_DIR / "adata_embedded.h5ad"))
    emb_poincare = adata.obsm["X_poincare"]
    emb_lorentz = adata.obsm["X_lorentz"]

    # 确定标签列
    label_col = None
    for col in ["Level1", "Level2", "leiden", "louvain"]:
        if col in adata.obs.columns:
            label_col = col
            break

    # 如果没有标签，用 leiden 聚类
    if label_col is None:
        adata.obsm["X_pca_temp"] = emb_poincare[:, :min(emb_poincare.shape[1], 50)]
        sc.pp.neighbors(adata, use_rep="X_pca_temp", n_neighbors=15)
        sc.tl.leiden(adata, resolution=0.5)
        label_col = "leiden"

    labels = adata.obs[label_col].values
    unique_labels = sorted(set(labels))
    cmap = get_color_mapping(unique_labels)

    # --- Panel 1: Poincare Disk (2D) ---
    fig1, ax1 = create_figure(figsize=(9, 9))

    # 绘制 Poincare 单位圆
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.5, alpha=0.3)

    # 取前 2 维
    x_2d = emb_poincare[:, 0]
    y_2d = emb_poincare[:, 1]

    for lab in unique_labels:
        mask = labels == lab
        ax1.scatter(x_2d[mask], y_2d[mask], s=8, c=cmap[lab],
                    label=lab, alpha=0.8, edgecolors="none")

    ax1.set_aspect("equal")
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_title(f"Poincare Disk Embedding (d1 x d2, colored by {label_col})")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7,
               markerscale=2, frameon=True)
    add_watermark(ax1)

    save_figure(fig1, str(FIG_DIR / "preview" / "poincare_disk_2d.png"),
                config={"chart": "poincare_disk", "dims": "0,1", "label": label_col})
    print(f"  Saved: preview/poincare_disk_2d.png")

    # --- Panel 2: 3D Hyperboloid ---
    if emb_lorentz.shape[1] >= 3:
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection="3d")

        # 取前 3 个空间分量（跳过时间分量 x_0）
        z1 = emb_lorentz[:, 1]
        z2 = emb_lorentz[:, 2]
        z0 = emb_lorentz[:, 0]  # 时间分量

        for lab in unique_labels:
            mask = labels == lab
            ax2.scatter(z1[mask], z2[mask], z0[mask], s=5,
                        c=cmap[lab], label=lab, alpha=0.7)

        ax2.set_xlabel("$z_1$")
        ax2.set_ylabel("$z_2$")
        ax2.set_zlabel("$z_0$ (time)")
        ax2.set_title(f"Lorentz Hyperboloid (z0, z1, z2, colored by {label_col})")
        ax2.view_init(elev=25, azim=45)

        save_figure(fig2, str(FIG_DIR / "preview" / "hyperboloid_3d.png"),
                    config={"chart": "hyperboloid_3d", "label": label_col})
        print(f"  Saved: preview/hyperboloid_3d.png")

    # --- Panel 3: Radial Distribution ---
    fig3, ax3 = create_figure(figsize=(10, 5))

    radii = np.linalg.norm(emb_poincare, axis=1)

    for lab in unique_labels:
        mask = labels == lab
        r = radii[mask]
        ax3.hist(r, bins=40, alpha=0.5, label=lab, color=cmap[lab],
                 density=True, edgecolor="none")

    ax3.set_xlabel("Poincare Radius ||z||")
    ax3.set_ylabel("Density")
    ax3.set_title(f"Radial Distribution by {label_col}")
    ax3.legend(fontsize=7, ncol=max(1, len(unique_labels) // 5))
    ax3.axvline(radii.mean(), color="red", linestyle="--", linewidth=1,
                label=f"mean={radii.mean():.3f}")
    add_watermark(ax3)

    save_figure(fig3, str(FIG_DIR / "preview" / "radial_distribution.png"),
                config={"chart": "radial_distribution", "label": label_col})
    print(f"  Saved: preview/radial_distribution.png")

    return adata, emb_poincare, labels, label_col


# =========================================================================
# 3. Baseline Comparison: PCA / UMAP vs Hyperbolic
# =========================================================================

def plot_baseline_comparison(adata, emb_poincare, labels, label_col):
    """PCA / UMAP / scVI vs Hyperbolic 对比图"""
    print("[Figure] Baseline comparison...")

    unique_labels = sorted(set(labels))
    cmap = get_color_mapping(unique_labels)

    # 计算 PCA 和 UMAP
    adata_work = adata.copy()
    sc.tl.pca(adata_work, n_comps=50)
    sc.pp.neighbors(adata_work, n_pcs=50)
    sc.tl.umap(adata_work)

    pca_2d = adata_work.obsm["X_pca"][:, :2]
    umap_2d = adata_work.obsm["X_umap"]

    # 4-panel 对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    methods = [
        ("PCA (dim 1-2)", pca_2d, axes[0, 0]),
        ("UMAP", umap_2d, axes[0, 1]),
        ("Poincare (dim 1-2)", emb_poincare[:, :2], axes[1, 0]),
    ]

    for title, coords, ax in methods:
        for lab in unique_labels:
            mask = labels == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6,
                       c=cmap[lab], label=lab, alpha=0.7, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        if "Poincare" in title:
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1, alpha=0.3)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

    # Panel 4: 指标表
    ax_table = axes[1, 1]
    ax_table.axis("off")

    # 计算 Silhouette 对比
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)

    sil_pca = silhouette_score(pca_2d, labels_int, sample_size=min(3000, len(labels_int)))
    sil_umap = silhouette_score(umap_2d, labels_int, sample_size=min(3000, len(labels_int)))
    sil_poincare = silhouette_score(emb_poincare, labels_int, sample_size=min(3000, len(labels_int)))

    table_data = [
        ["Method", "Dims", "Silhouette"],
        ["PCA", "2", f"{sil_pca:.4f}"],
        ["UMAP", "2", f"{sil_umap:.4f}"],
        ["Poincare", f"{emb_poincare.shape[1]}", f"{sil_poincare:.4f}"],
    ]
    table = ax_table.table(
        cellText=table_data[1:], colLabels=table_data[0],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax_table.set_title("Embedding Quality Comparison", pad=20, fontsize=12)

    # 共享图例
    handles, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbl, loc="lower center", ncol=min(8, len(unique_labels)),
               fontsize=8, markerscale=2, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Embedding Baseline Comparison (colored by {label_col})",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    save_figure(fig, str(FIG_DIR / "compare" / "baseline_comparison_4panel.png"),
                config={"chart": "baseline_comparison", "label": label_col})
    print(f"  Saved: compare/baseline_comparison_4panel.png")


def _knn_overlap_score(base: np.ndarray, emb: np.ndarray, k: int = 15) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = min(len(base), len(emb))
    if n < (k + 2):
        return 0.0
    nn_base = NearestNeighbors(n_neighbors=k + 1).fit(base)
    nn_emb = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx_base = nn_base.kneighbors(return_distance=False)[:, 1:]
    idx_emb = nn_emb.kneighbors(return_distance=False)[:, 1:]
    overlaps = []
    for i in range(n):
        a = set(idx_base[i].tolist())
        b = set(idx_emb[i].tolist())
        overlaps.append(len(a & b) / float(k))
    return float(np.mean(overlaps))


def plot_cns_step1_advantage(adata, emb_poincare, labels, label_col):
    """CNS 级展示: 双曲嵌入 vs UMAP 优势证据图。"""
    print("[Figure] CNS step1 advantage panel...")

    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    adata_work = adata.copy()
    sc.tl.pca(adata_work, n_comps=50)
    sc.pp.neighbors(adata_work, n_pcs=50)
    sc.tl.umap(adata_work)
    umap_2d = adata_work.obsm["X_umap"]
    raw_pca = adata_work.obsm["X_pca"][:, :30]

    # 抽样，避免大样本下度量过慢
    n = len(labels)
    sample_n = min(3000, n)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=sample_n, replace=False)) if sample_n < n else np.arange(n)

    le = LabelEncoder()
    y = le.fit_transform(np.asarray(labels)[idx])
    hyp = emb_poincare[idx, : min(emb_poincare.shape[1], 10)]
    um2 = umap_2d[idx]
    raw = raw_pca[idx]

    # 量化指标
    sil_h = float(silhouette_score(hyp, y))
    sil_u = float(silhouette_score(um2, y))
    tw_h = float(trustworthiness(raw, hyp, n_neighbors=15))
    tw_u = float(trustworthiness(raw, um2, n_neighbors=15))
    knn_h = _knn_overlap_score(raw, hyp, k=15)
    knn_u = _knn_overlap_score(raw, um2, k=15)

    # 层级保真: 半径与到群中心距离相关性
    raw_center = raw.mean(axis=0, keepdims=True)
    raw_dist = np.linalg.norm(raw - raw_center, axis=1)
    hyp_radius = np.linalg.norm(hyp[:, :2], axis=1)
    um_radius = np.linalg.norm(um2 - um2.mean(axis=0, keepdims=True), axis=1)
    hier_h = float(np.corrcoef(raw_dist, hyp_radius)[0, 1])
    hier_u = float(np.corrcoef(raw_dist, um_radius)[0, 1])

    unique_labels = sorted(set(labels))
    cmap = get_color_mapping(unique_labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    ax_u, ax_h, ax_b, ax_s = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for lab in unique_labels:
        mask_all = np.asarray(labels)[idx] == lab
        ax_u.scatter(um2[mask_all, 0], um2[mask_all, 1], s=8, c=cmap[lab], alpha=0.7, edgecolors="none")
        ax_h.scatter(hyp[mask_all, 0], hyp[mask_all, 1], s=8, c=cmap[lab], alpha=0.7, edgecolors="none")
    ax_u.set_title("A. UMAP")
    ax_h.set_title("B. Hyperbolic (Poincare)")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_h.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1, alpha=0.25)
    ax_h.set_xlim(-1.05, 1.05)
    ax_h.set_ylim(-1.05, 1.05)
    ax_h.set_aspect("equal")

    metric_names = ["Silhouette", "Trustworthiness", "kNN overlap", "Hierarchy corr"]
    vals_u = [sil_u, tw_u, knn_u, hier_u]
    vals_h = [sil_h, tw_h, knn_h, hier_h]
    x = np.arange(len(metric_names))
    w = 0.35
    ax_b.bar(x - w / 2, vals_u, width=w, color="#4477AA", label="UMAP")
    ax_b.bar(x + w / 2, vals_h, width=w, color="#EE6677", label="Hyperbolic")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(metric_names, rotation=12)
    ax_b.set_ylabel("Score")
    ax_b.set_title("C. Quantitative Advantage Metrics")
    ax_b.legend()

    ax_s.scatter(raw_dist, um_radius, s=8, alpha=0.45, color="#4477AA", label=f"UMAP r={hier_u:.3f}")
    ax_s.scatter(raw_dist, hyp_radius, s=8, alpha=0.45, color="#EE6677", label=f"Hyperbolic r={hier_h:.3f}")
    ax_s.set_xlabel("Raw-space radial distance")
    ax_s.set_ylabel("Embedding radial distance")
    ax_s.set_title("D. Hierarchy Preservation")
    ax_s.legend()

    fig.suptitle(f"CNS Figure (Step1): Hyperbolic Embedding Advantage ({label_col})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, str(FIG_DIR / "compare" / "cns_step1_hyperbolic_vs_umap.png"),
                config={"chart": "cns_step1_hyperbolic_vs_umap"})

    metrics = {
        "umap": {
            "silhouette": sil_u,
            "trustworthiness": tw_u,
            "knn_overlap": knn_u,
            "hierarchy_correlation": hier_u,
        },
        "hyperbolic": {
            "silhouette": sil_h,
            "trustworthiness": tw_h,
            "knn_overlap": knn_h,
            "hierarchy_correlation": hier_h,
        },
        "label_column": label_col,
        "n_eval_cells": int(sample_n),
    }
    (FIG_DIR / "compare" / "cns_step1_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("  Saved: compare/cns_step1_hyperbolic_vs_umap.png")
    print("  Saved: compare/cns_step1_metrics.json")


# =========================================================================
# 4. Training Loss Curves
# =========================================================================

def plot_training_losses():
    """训练损失曲线"""
    print("[Figure] Training loss curves...")

    with open(STEP1_DIR / "training_losses.json") as f:
        losses = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(losses["total"]) + 1)

    ax = axes[0]
    ax.plot(epochs, losses["total"], color="#4477AA", linewidth=1.5)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax = axes[1]
    ax.plot(epochs, losses["recon"], color="#EE6677", linewidth=1.5, label="Recon")
    ax.set_title("Reconstruction Loss (NB)")
    ax.set_xlabel("Epoch")

    ax = axes[2]
    ax.plot(epochs, losses["kl"], color="#228833", linewidth=1.5, label="KL")
    ax2 = ax.twinx()
    ax2.plot(epochs, losses["topo"], color="#CCBB44", linewidth=1.5, label="Topo")
    ax.set_title("KL + Topo Regularization")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL", color="#228833")
    ax2.set_ylabel("Topo", color="#CCBB44")

    fig.suptitle("H-VAE Training Diagnostics", fontsize=13)
    fig.tight_layout()
    add_watermark(axes[0])

    save_figure(fig, str(FIG_DIR / "preview" / "training_losses.png"),
                config={"chart": "training_losses"})
    print(f"  Saved: preview/training_losses.png")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("HyperSCA Stage 1 - Figure Generation")
    print("=" * 60)

    # Check results exist
    if not (STEP1_DIR / "adata_embedded.h5ad").exists():
        print("[ERROR] results/step1/adata_embedded.h5ad not found.")
        print("Run scripts/run_step1.py first.")
        return 1

    plot_graph_topology_comparison()
    print()

    adata, emb_poincare, labels, label_col = plot_embedding_core()
    print()

    plot_baseline_comparison(adata, emb_poincare, labels, label_col)
    print()

    plot_cns_step1_advantage(adata, emb_poincare, labels, label_col)
    print()

    plot_training_losses()
    print()

    print("=" * 60)
    print("[DONE] All figures generated.")
    print(f"Output: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
