"""Phase 3 可视化：反事实扰动与空间传播

提供以下图类型：
- 干预前后表达对比（目标基因与 marker, volcano/bar）
- 反事实空间热图（局部 vs 远端效应, 双面板）
- 传播深度与梯度衰减图（BFS 分层 + 距离衰减拟合）
- 指标面板（R²/PCC/MSE, Marker Direction, Moran's I, Propagation Depth）
- 多靶点对比热图

所有函数在数据就绪前返回带占位信息的 Figure。

对齐文档:
    - docs/technical_roadmap.md §5
    - docs/evaluation_suite.md §3-4
参考范式: CPA, scGen, DynPerturb
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot_style import (
    PALETTE_CATEGORICAL,
    CMAP_SEQUENTIAL,
    CMAP_DIVERGING,
    CMAP_SPATIAL,
    COLOR_GOOD,
    COLOR_WARN,
    COLOR_BAD,
    apply_style,
    create_figure,
    save_figure,
    add_watermark,
)

apply_style()


# =========================================================================
# 干预前后表达对比
# =========================================================================

def plot_perturbation_comparison(
    gene_names: list[str] | None = None,
    observed: np.ndarray | None = None,
    counterfactual: np.ndarray | None = None,
    target_gene: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """干预前后基因表达变化条形图

    Parameters
    ----------
    gene_names : 基因名列表
    observed : (G,) 观测均值
    counterfactual : (G,) 反事实均值
    target_gene : 被干预的靶基因名
    """
    fig, axes = create_figure(1, 2, figsize=(14, 6))

    if gene_names is None or observed is None:
        for ax in axes:
            ax.text(0.5, 0.5, "Awaiting\nperturbation data",
                    ha="center", va="center", fontsize=14, color="#999999",
                    transform=ax.transAxes, style="italic")
        axes[0].set_title("Expression Change (Top DEGs)")
        axes[1].set_title("Fold Change Distribution")
    else:
        gene_names = np.asarray(gene_names)
        fold_change = counterfactual - observed  # log-scale difference

        # 按变化量排序取 top
        order = np.argsort(np.abs(fold_change))[::-1]
        top_n = min(25, len(gene_names))
        top_idx = order[:top_n]

        # --- Panel 1: Paired bar ---
        ax = axes[0]
        y = np.arange(top_n)
        bar_w = 0.35
        ax.barh(y + bar_w / 2, observed[top_idx], bar_w,
                color="#4477AA", label="Observed", edgecolor="white")
        ax.barh(y - bar_w / 2, counterfactual[top_idx], bar_w,
                color="#EE6677", label="Counterfactual", edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(gene_names[top_idx], fontsize=8)
        ax.set_xlabel("Mean Expression")
        ax.set_title(f"Top-{top_n} DEGs (KO: {target_gene})")
        ax.legend(fontsize=9)

        # --- Panel 2: Fold-change histogram ---
        ax = axes[1]
        ax.hist(fold_change, bins=60, color="#AA3377", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("Expression Change (CF − Obs)")
        ax.set_ylabel("Gene Count")
        ax.set_title("Fold Change Distribution")

        # 标注靶基因
        if target_gene and target_gene in gene_names:
            tgt_idx = np.where(gene_names == target_gene)[0][0]
            tgt_fc = fold_change[tgt_idx]
            ax.axvline(tgt_fc, color="#EE6677", linestyle="--", linewidth=1.5,
                       label=f"{target_gene}: {tgt_fc:+.3f}")
            ax.legend(fontsize=9)

    fig.suptitle(f"Perturbation Analysis — {target_gene or 'Target TBD'} Knockout",
                 fontsize=14, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "perturbation_comparison", "target": target_gene})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 反事实空间热图
# =========================================================================

def plot_counterfactual_spatial(
    coords: np.ndarray | None = None,
    observed_expr: np.ndarray | None = None,
    counterfactual_expr: np.ndarray | None = None,
    gene_name: str = "Gene",
    save_path: str | None = None,
    point_size: float = 6,
) -> plt.Figure:
    """干预前后空间表达热图（双面板）

    Parameters
    ----------
    coords : (N, 2) 空间坐标
    observed_expr : (N,) 观测基因表达
    counterfactual_expr : (N,) 反事实基因表达
    gene_name : 基因名
    """
    fig, axes = create_figure(1, 3, figsize=(18, 6))

    titles = [f"Observed — {gene_name}", f"Counterfactual — {gene_name}", f"Δ Expression"]

    if coords is None:
        for ax, title in zip(axes, titles):
            ax.text(0.5, 0.5, "Awaiting\nspatial data",
                    ha="center", va="center", fontsize=14, color="#999999",
                    transform=ax.transAxes, style="italic")
            ax.set_title(title)
    else:
        # 共享 colorbar 范围
        vmin = min(observed_expr.min(), counterfactual_expr.min())
        vmax = max(observed_expr.max(), counterfactual_expr.max())

        # Panel 1: Observed
        sc1 = axes[0].scatter(
            coords[:, 0], coords[:, 1], c=observed_expr,
            cmap=CMAP_SPATIAL, s=point_size, edgecolors="none", vmin=vmin, vmax=vmax,
        )
        fig.colorbar(sc1, ax=axes[0], shrink=0.6, pad=0.02)

        # Panel 2: Counterfactual
        sc2 = axes[1].scatter(
            coords[:, 0], coords[:, 1], c=counterfactual_expr,
            cmap=CMAP_SPATIAL, s=point_size, edgecolors="none", vmin=vmin, vmax=vmax,
        )
        fig.colorbar(sc2, ax=axes[1], shrink=0.6, pad=0.02)

        # Panel 3: Delta
        delta = counterfactual_expr - observed_expr
        d_abs = max(abs(delta.min()), abs(delta.max()), 1e-6)
        sc3 = axes[2].scatter(
            coords[:, 0], coords[:, 1], c=delta,
            cmap=CMAP_DIVERGING, s=point_size, edgecolors="none",
            vmin=-d_abs, vmax=d_abs,
        )
        fig.colorbar(sc3, ax=axes[2], shrink=0.6, pad=0.02)

        for ax in axes:
            ax.set_aspect("equal")
            ax.invert_yaxis()

        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=12)

    fig.suptitle(f"Counterfactual Spatial Expression — {gene_name}", fontsize=14, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "counterfactual_spatial", "gene": gene_name})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 传播深度与梯度衰减
# =========================================================================

def plot_propagation_gradient(
    distances: np.ndarray | None = None,
    effect_magnitudes: np.ndarray | None = None,
    fit_params: dict | None = None,
    bfs_layers: list[dict] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """传播深度与梯度衰减双面板

    Panel 1: 散点图 — effect magnitude vs spatial distance + 拟合衰减曲线
    Panel 2: BFS 分层条形图 — 每层平均效应

    Parameters
    ----------
    distances : (M,) 空间距离
    effect_magnitudes : (M,) 效应大小 |Δx|
    fit_params : dict with keys "ell" (characteristic length), "r2" (fit R²)
    bfs_layers : list[dict] with keys "hop", "mean_effect", "n_cells"
    """
    fig, axes = create_figure(1, 2, figsize=(14, 5))

    # --- Panel 1: 梯度衰减散点 ---
    ax = axes[0]
    if distances is None or effect_magnitudes is None:
        ax.text(0.5, 0.5, "Awaiting\npropagation data",
                ha="center", va="center", fontsize=14, color="#999999",
                transform=ax.transAxes, style="italic")
    else:
        ax.scatter(distances, effect_magnitudes, s=3, alpha=0.4, c="#4477AA", edgecolors="none")

        # 拟合曲线
        if fit_params and "ell" in fit_params:
            ell = fit_params["ell"]
            r2 = fit_params.get("r2", 0)
            x_fit = np.linspace(0, distances.max(), 200)
            amp = effect_magnitudes.max()
            y_fit = amp * np.exp(-x_fit / ell)
            ax.plot(x_fit, y_fit, color="#EE6677", linewidth=2,
                    label=f"exp(-d/ℓ), ℓ={ell:.1f}, R²={r2:.3f}")
            ax.legend(fontsize=9)

        ax.set_xlabel("Spatial Distance")
        ax.set_ylabel("|Δ Expression|")

    ax.set_title("Gradient Decay with Distance", fontsize=12)

    # --- Panel 2: BFS 分层 ---
    ax = axes[1]
    if bfs_layers is None:
        ax.text(0.5, 0.5, "Awaiting\nBFS layer data",
                ha="center", va="center", fontsize=14, color="#999999",
                transform=ax.transAxes, style="italic")
    else:
        hops = [d["hop"] for d in bfs_layers]
        means = [d["mean_effect"] for d in bfs_layers]
        n_cells = [d["n_cells"] for d in bfs_layers]

        colors = [COLOR_GOOD if m > 0.05 else COLOR_WARN if m > 0.01 else COLOR_BAD
                  for m in means]
        ax.bar(hops, means, color=colors, edgecolor="white", width=0.7)
        for h, m, n in zip(hops, means, n_cells):
            ax.text(h, m + max(means) * 0.02, f"n={n}", ha="center", fontsize=8, color="#555555")
        ax.set_xlabel("BFS Hop")
        ax.set_ylabel("Mean |Δ Expression|")

    ax.set_title("Propagation Depth (BFS layers)", fontsize=12)

    fig.suptitle("Perturbation Propagation Analysis", fontsize=14, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "propagation_gradient"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 多靶点对比热图
# =========================================================================

def plot_multi_target_heatmap(
    targets: list[str] | None = None,
    marker_genes: list[str] | None = None,
    fold_changes: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """多靶点 × marker 基因的变化热图

    Parameters
    ----------
    targets : 靶基因列表
    marker_genes : marker 基因列表
    fold_changes : (n_targets, n_markers) 表达变化矩阵
    """
    fig, ax = create_figure(figsize=(max(8, len(marker_genes or []) * 0.5 + 2),
                                      max(5, len(targets or []) * 0.5 + 2)))

    if targets is None or fold_changes is None:
        ax.text(0.5, 0.5, "Awaiting\nmulti-target data",
                ha="center", va="center", fontsize=14, color="#999999",
                transform=ax.transAxes, style="italic")
        ax.set_title("Multi-target Perturbation Comparison")
    else:
        d_abs = max(abs(fold_changes.min()), abs(fold_changes.max()), 1e-6)
        im = ax.imshow(fold_changes, cmap=CMAP_DIVERGING, aspect="auto",
                       vmin=-d_abs, vmax=d_abs)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Expression Change (CF − Obs)", fontsize=10)

        ax.set_xticks(range(len(marker_genes)))
        ax.set_xticklabels(marker_genes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels([f"{t} KO" for t in targets], fontsize=9)
        ax.set_xlabel("Marker Genes")
        ax.set_ylabel("Perturbation Target")
        ax.set_title("Multi-target Perturbation Comparison", fontsize=13)

        # 数值标注
        for i in range(len(targets)):
            for j in range(len(marker_genes)):
                val = fold_changes[i, j]
                color = "white" if abs(val) > d_abs * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    add_watermark(ax)
    if save_path:
        save_figure(fig, save_path, config={"chart": "multi_target_heatmap"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 反事实指标仪表盘
# =========================================================================

def plot_perturbation_metrics_dashboard(
    metrics: dict[str, float] | None = None,
    spatial_metrics: dict[str, float] | None = None,
    target_gene: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Step 3 反事实与空间一致性指标 Dashboard

    Parameters
    ----------
    metrics : dict
        键如 "r2_mean", "r2_var", "pcc_median", "mse",
        "marker_direction_accuracy", "deg_overlap_jaccard"
    spatial_metrics : dict
        键如 "morans_i_obs", "morans_i_cf", "delta_morans_i",
        "gradient_decay_r2", "propagation_depth"
    """
    metric_defs = [
        # (display, key, threshold, is_lower_better, target_str)
        ("R² (mean) ↑", "r2_mean", 0.8, False, "> 0.8"),
        ("PCC median ↑", "pcc_median", 0.7, False, "> 0.7"),
        ("MSE ↓", "mse", 0.05, True, "→ 0"),
        ("Marker Dir Acc ↑", "marker_direction_accuracy", 0.8, False, "> 0.8"),
    ]
    spatial_defs = [
        ("Δ Moran's I", "delta_morans_i", None, None, "方向一致"),
        ("Gradient R² ↑", "gradient_decay_r2", 0.3, False, "> 0.3"),
        ("Propagation Depth", "propagation_depth", None, None, "与拓扑一致"),
    ]

    n_total = len(metric_defs) + len(spatial_defs)
    ncols = 4
    nrows = (n_total + ncols - 1) // ncols
    fig, axes = create_figure(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    all_defs = metric_defs + spatial_defs
    all_metrics = {}
    if metrics:
        all_metrics.update(metrics)
    if spatial_metrics:
        all_metrics.update(spatial_metrics)

    for i, (name, key, thresh, inv, target_str) in enumerate(all_defs):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        if key in all_metrics:
            val = all_metrics[key]
            if thresh is not None and inv is not None:
                ok = val <= thresh if inv else val >= thresh
                color = COLOR_GOOD if ok else COLOR_BAD
            else:
                color = "#4477AA"
            ax.barh([0], [val], color=color, edgecolor="white", height=0.5)
            if thresh is not None:
                ax.axvline(thresh, color=COLOR_WARN, linestyle="--", linewidth=1.2)
            ax.text(val + 0.005, 0, f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
            ax.text(0.98, 0.95, f"Target: {target_str}", ha="right", va="top",
                    fontsize=8, color="#888888", transform=ax.transAxes)
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=14, color="#999999", transform=ax.transAxes, style="italic")
        ax.set_title(name, fontsize=11, fontweight="bold")

    # 隐藏多余子图
    for j in range(len(all_defs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    title = "Step 3 — Counterfactual & Spatial Consistency Metrics"
    if target_gene:
        title += f" ({target_gene} KO)"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    add_watermark(axes_flat[min(len(all_defs) - 1, len(axes_flat) - 1)])

    if save_path:
        save_figure(fig, save_path, config={"chart": "perturbation_metrics_dashboard", "target": target_gene})
    else:
        plt.close(fig)
    return fig


def plot_interaction_target_ranking(
    ranked_targets=None,
    top_n: int = 20,
    score_col: str = "target_priority_score",
    save_path: str | None = None,
) -> plt.Figure:
    """候选互作靶点排名图（基于反事实优先级分数）。"""
    fig, ax = create_figure(figsize=(12, 7))

    if ranked_targets is None or len(ranked_targets) == 0:
        ax.text(
            0.5,
            0.5,
            "Awaiting\ntarget ranking data",
            ha="center",
            va="center",
            fontsize=14,
            color="#999999",
            transform=ax.transAxes,
            style="italic",
        )
        ax.set_title("Counterfactual Interaction Target Ranking")
    else:
        df = ranked_targets.copy()
        if score_col not in df.columns:
            score_col = df.columns[-1]
        df = df.sort_values(score_col, ascending=False).head(top_n).copy()
        labels = [f"{r['ligand']}→{r['receptor']}" for _, r in df.iterrows()]
        y = np.arange(len(df))
        colors = ["#EE6677" if bool(v) else "#4477AA" for v in df.get("prior_hit", [False] * len(df))]

        ax.barh(y, df[score_col].values, color=colors, edgecolor="white", alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Priority Score")
        ax.set_title(f"Top-{len(df)} Counterfactual Interaction Targets")
        ax.grid(axis="x", alpha=0.2, linestyle="--")

        for i, (_, row) in enumerate(df.iterrows()):
            suffix = " [prior]" if bool(row.get("prior_hit", False)) else ""
            ax.text(
                row[score_col] + max(df[score_col].max() * 0.01, 1e-6),
                i,
                f"{row[score_col]:.3f}{suffix}",
                va="center",
                fontsize=8,
                color="#333333",
            )

    add_watermark(ax)
    if save_path:
        save_figure(fig, save_path, config={"chart": "interaction_target_ranking"})
    else:
        plt.close(fig)
    return fig


def plot_step3_overview_dashboard(
    top_targets=None,
    method_name: str = "",
    pathway_counts: dict[str, int] | None = None,
    prior_hit_rate: float | None = None,
    score_values: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """阶段3总览仪表盘：多靶点候选汇总。"""
    fig, axes = create_figure(2, 2, figsize=(14, 10))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Panel 1: 全局 top 候选
    ax = axes[0]
    if top_targets is None or len(top_targets) == 0:
        ax.text(0.5, 0.5, "Awaiting\noverview targets", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="#999999", style="italic")
        ax.set_title("Global Top Targets")
    else:
        df = top_targets.copy()
        df = df.sort_values("target_priority_score", ascending=False).head(12)
        labels = [f"{r['target_gene']}:{r['ligand']}→{r['receptor']}" for _, r in df.iterrows()]
        y = np.arange(len(df))
        colors = ["#EE6677" if bool(v) else "#4477AA" for v in df.get("prior_hit", [False] * len(df))]
        ax.barh(y, df["target_priority_score"].values, color=colors, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Global Top Interaction Targets")
        ax.set_xlabel("Priority Score")

    # Panel 2: prior hit rate
    ax = axes[1]
    if prior_hit_rate is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
                fontsize=16, color="#999999", style="italic")
        ax.set_title("Prior Hit Rate")
    else:
        val = float(np.clip(prior_hit_rate, 0.0, 1.0))
        ax.bar(["Prior Hit"], [val], color="#44AA99", edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_title("Prior Hit Rate (OmniPath/LIANA/NicheNet)")
        ax.text(0, val + 0.03, f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")

    # Panel 3: score distribution
    ax = axes[2]
    if score_values is None or len(score_values) == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
                fontsize=16, color="#999999", style="italic")
        ax.set_title("Score Distribution")
    else:
        ax.hist(score_values, bins=40, color="#AA3377", alpha=0.85, edgecolor="white")
        ax.set_title("Target Priority Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    # Panel 4: pathway frequency
    ax = axes[3]
    if not pathway_counts:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
                fontsize=16, color="#999999", style="italic")
        ax.set_title("Pathway Summary")
    else:
        items = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [k if k else "unknown" for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(np.arange(len(items)), vals, color="#4477AA", edgecolor="white")
        ax.set_yticks(np.arange(len(items)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Top Pathways in Candidate Targets")

    title = "Step 3 Overview Dashboard"
    if method_name:
        title += f" ({method_name})"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    add_watermark(axes[-1])
    if save_path:
        save_figure(fig, save_path, config={"chart": "step3_overview_dashboard", "method": method_name})
    else:
        plt.close(fig)
    return fig
