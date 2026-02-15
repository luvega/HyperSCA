"""Phase 2 可视化：因果图与多层信号流

提供以下图类型：
- 因果有向图（DAG，边粗细=bootstrap 频率，颜色=arrow strength）
- 多层信号流图（Ligand→Receptor→TF→Target 分层 Sankey/Alluvial）
- 关键轴证据卡（如 CAF→TAM/Treg 子图高亮）
- 邻接热图 / 因果矩阵
- 指标面板（Falsification p, Graph Sparsity, Known Axis Recovery, Direction Accuracy）

所有函数在数据就绪前返回带占位信息的 Figure。

对齐文档:
    - docs/engineering_blueprint.md
    - docs/evaluation_suite.md §2
参考范式: FlowSig, DoWhy
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from src.utils.plot_style import (
    PALETTE_CATEGORICAL,
    CMAP_SEQUENTIAL,
    CMAP_DIVERGING,
    COLOR_GOOD,
    COLOR_WARN,
    COLOR_BAD,
    apply_style,
    create_figure,
    get_color_mapping,
    save_figure,
    add_watermark,
)

apply_style()


# =========================================================================
# 因果有向图 (DAG)
# =========================================================================

def plot_causal_dag(
    adjacency: np.ndarray | None = None,
    node_labels: list[str] | None = None,
    bootstrap_freq: np.ndarray | None = None,
    arrow_strength: np.ndarray | None = None,
    layout: str = "circular",
    freq_threshold: float = 0.5,
    save_path: str | None = None,
) -> plt.Figure:
    """因果有向图

    Parameters
    ----------
    adjacency : (K, K)
        二值邻接矩阵，adjacency[i,j]=1 表示 i→j
    node_labels : 长度 K 的节点名称
    bootstrap_freq : (K, K)
        每条边的 bootstrap 出现频率 [0,1]，映射到边粗细
    arrow_strength : (K, K)
        因果效应强度，映射到边颜色
    layout : 'circular' | 'spring' | 'hierarchical'
    freq_threshold : 只绘制 freq >= 此值的边
    """
    fig, ax = create_figure(figsize=(10, 10))

    if adjacency is None:
        ax.text(0.5, 0.5, "Awaiting\ncausal graph data",
                ha="center", va="center", fontsize=16, color="#999999",
                style="italic", transform=ax.transAxes)
        ax.set_title("Causal DAG (Step 2)")
        add_watermark(ax)
        if save_path:
            save_figure(fig, save_path, config={"chart": "causal_dag_placeholder"})
        else:
            plt.close(fig)
        return fig

    K = adjacency.shape[0]
    if node_labels is None:
        node_labels = [f"Node {i}" for i in range(K)]

    # 布局计算
    positions = _compute_layout(K, layout)

    # 节点颜色
    cmap = get_color_mapping(node_labels)
    node_colors = [cmap[lab] for lab in node_labels]

    # 绘制边
    for i in range(K):
        for j in range(K):
            if adjacency[i, j] == 0:
                continue
            freq = bootstrap_freq[i, j] if bootstrap_freq is not None else 1.0
            if freq < freq_threshold:
                continue
            strength = arrow_strength[i, j] if arrow_strength is not None else 0.5

            # 粗细 ∝ freq, 颜色 ∝ strength
            lw = 0.5 + 4.0 * freq
            color = plt.cm.get_cmap(CMAP_SEQUENTIAL)(min(strength / max(arrow_strength.max(), 1e-6), 1.0)
                                                      if arrow_strength is not None else 0.5)

            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            ax.annotate(
                "", xy=(positions[j, 0], positions[j, 1]),
                xytext=(positions[i, 0], positions[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=lw, shrinkA=12, shrinkB=12,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=1,
            )

    # 绘制节点
    for i in range(K):
        ax.scatter(
            positions[i, 0], positions[i, 1],
            s=600, c=node_colors[i], edgecolors="white", linewidths=2, zorder=3,
        )
        ax.text(
            positions[i, 0], positions[i, 1] - 0.08,
            node_labels[i], ha="center", va="top", fontsize=8, fontweight="bold",
            zorder=4,
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Causal DAG — Cell Communication Network", fontsize=14)
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "causal_dag"})
    else:
        plt.close(fig)
    return fig


def _compute_layout(n: int, method: str = "circular") -> np.ndarray:
    """简易节点布局"""
    if method == "circular":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    elif method == "spring":
        # 简化 spring layout（真实场景建议用 networkx.spring_layout）
        rng = np.random.RandomState(42)
        return rng.randn(n, 2) * 0.8
    else:  # hierarchical (两层)
        half = n // 2
        upper = np.column_stack([np.linspace(-1, 1, half), np.ones(half) * 0.5])
        lower = np.column_stack([np.linspace(-1, 1, n - half), np.ones(n - half) * -0.5])
        return np.vstack([upper, lower])


# =========================================================================
# 多层信号流图（Sankey / Alluvial 风格）
# =========================================================================

def plot_signaling_flow(
    flow_edges: list[dict] | None = None,
    layer_names: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """4 层信号流可视化 (Ligand → Receptor → TF → Target)

    Parameters
    ----------
    flow_edges : list[dict]
        每条流边: {"source_layer": int, "source": str,
                    "target_layer": int, "target": str,
                    "weight": float}
    layer_names : list[str]
        各层名称，默认 ["Ligand", "Receptor", "TF", "Target"]
    """
    if layer_names is None:
        layer_names = ["Ligand", "Receptor", "TF", "Target"]

    n_layers = len(layer_names)
    fig, ax = create_figure(figsize=(14, 8))

    if flow_edges is None:
        # 占位
        for i, name in enumerate(layer_names):
            x = i / (n_layers - 1)
            ax.text(x, 0.5, name, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#999999",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0", edgecolor="#CCCCCC"))
            if i < n_layers - 1:
                ax.annotate(
                    "", xy=((i + 1) / (n_layers - 1) - 0.06, 0.5),
                    xytext=(x + 0.06, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#CCCCCC", lw=2),
                )
        ax.text(0.5, 0.15, "Awaiting signaling flow data",
                ha="center", va="center", fontsize=12, color="#AAAAAA",
                style="italic", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title("Multi-layer Signaling Flow (Step 2)", fontsize=14)
        add_watermark(ax)
        if save_path:
            save_figure(fig, save_path, config={"chart": "signaling_flow_placeholder"})
        else:
            plt.close(fig)
        return fig

    # 收集每层节点
    layer_nodes: dict[int, list[str]] = {i: [] for i in range(n_layers)}
    for e in flow_edges:
        sl, tl = e["source_layer"], e["target_layer"]
        if e["source"] not in layer_nodes[sl]:
            layer_nodes[sl].append(e["source"])
        if e["target"] not in layer_nodes[tl]:
            layer_nodes[tl].append(e["target"])

    # 节点位置
    node_pos: dict[str, tuple[float, float]] = {}
    for layer_idx in range(n_layers):
        nodes = layer_nodes[layer_idx]
        x = layer_idx / max(n_layers - 1, 1)
        for j, node in enumerate(nodes):
            y = (j + 1) / (len(nodes) + 1)
            node_pos[f"{layer_idx}:{node}"] = (x, y)

    # 绘制流
    max_w = max((e["weight"] for e in flow_edges), default=1)
    for e in flow_edges:
        src_key = f"{e['source_layer']}:{e['source']}"
        tgt_key = f"{e['target_layer']}:{e['target']}"
        if src_key not in node_pos or tgt_key not in node_pos:
            continue
        sx, sy = node_pos[src_key]
        tx, ty = node_pos[tgt_key]
        w = e["weight"] / max_w
        ax.annotate(
            "", xy=(tx, ty), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>", color=plt.cm.get_cmap(CMAP_SEQUENTIAL)(w),
                lw=0.5 + 4 * w, alpha=0.6 + 0.4 * w,
                shrinkA=8, shrinkB=8,
                connectionstyle="arc3,rad=0.05",
            ),
        )

    # 绘制节点
    all_nodes = set()
    for layer_idx in range(n_layers):
        for node in layer_nodes[layer_idx]:
            key = f"{layer_idx}:{node}"
            x, y = node_pos[key]
            ax.scatter(x, y, s=300, c=PALETTE_CATEGORICAL[layer_idx % len(PALETTE_CATEGORICAL)],
                       edgecolors="white", linewidths=1.5, zorder=3)
            ax.text(x, y - 0.035, node, ha="center", va="top", fontsize=7, zorder=4)

    # 层标题
    for i, name in enumerate(layer_names):
        x = i / max(n_layers - 1, 1)
        ax.text(x, 1.05, name, ha="center", va="bottom", fontsize=12,
                fontweight="bold", transform=ax.transAxes)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.axis("off")
    ax.set_title("Multi-layer Signaling Flow (Ligand → Receptor → TF → Target)", fontsize=14)
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "signaling_flow"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 因果邻接热图
# =========================================================================

def plot_causal_heatmap(
    adjacency: np.ndarray | None = None,
    node_labels: list[str] | None = None,
    values: np.ndarray | None = None,
    value_name: str = "Bootstrap Frequency",
    save_path: str | None = None,
) -> plt.Figure:
    """因果邻接矩阵热图

    Parameters
    ----------
    adjacency : (K, K) 二值邻接
    values : (K, K) 连续值（如 bootstrap freq 或 arrow strength），覆盖邻接显示
    """
    fig, ax = create_figure(figsize=(9, 8))

    if adjacency is None and values is None:
        ax.text(0.5, 0.5, "Awaiting\nadjacency data",
                ha="center", va="center", fontsize=16, color="#999999",
                style="italic", transform=ax.transAxes)
        ax.set_title(f"Causal Adjacency — {value_name}")
        add_watermark(ax)
        if save_path:
            save_figure(fig, save_path, config={"chart": "causal_heatmap_placeholder"})
        else:
            plt.close(fig)
        return fig

    mat = values if values is not None else adjacency.astype(float)
    K = mat.shape[0]
    if node_labels is None:
        node_labels = [f"Node {i}" for i in range(K)]

    im = ax.imshow(mat, cmap=CMAP_SEQUENTIAL, aspect="equal", vmin=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(value_name, fontsize=10)

    ax.set_xticks(range(K))
    ax.set_xticklabels(node_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K))
    ax.set_yticklabels(node_labels, fontsize=9)
    ax.set_xlabel("Target (effect)")
    ax.set_ylabel("Source (cause)")
    ax.set_title(f"Causal Adjacency — {value_name}", fontsize=13)

    # 数值标注
    for i in range(K):
        for j in range(K):
            val = mat[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > mat.max() * 0.6 else "black")

    add_watermark(ax)
    if save_path:
        save_figure(fig, save_path, config={"chart": "causal_heatmap"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 关键轴证据卡
# =========================================================================

def plot_key_axis_evidence(
    axis_name: str = "CAF → TAM / Treg",
    sub_adjacency: np.ndarray | None = None,
    sub_labels: list[str] | None = None,
    bootstrap_freq: np.ndarray | None = None,
    arrow_strength: np.ndarray | None = None,
    supporting_text: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """单条关键信号轴的证据卡

    包含：子图可视化 + bootstrap 频率柱 + arrow strength + 文字证据摘要。

    Parameters
    ----------
    axis_name : 信号轴名称
    sub_adjacency : 子图邻接矩阵
    sub_labels : 子图节点名
    supporting_text : 文字证据描述
    """
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # --- 左: 子图 ---
    ax_graph = fig.add_subplot(gs[0])
    if sub_adjacency is None:
        ax_graph.text(0.5, 0.5, "Sub-graph\nawaiting data",
                      ha="center", va="center", fontsize=12, color="#999999",
                      transform=ax_graph.transAxes, style="italic")
    else:
        K = sub_adjacency.shape[0]
        labels = sub_labels or [f"N{i}" for i in range(K)]
        positions = _compute_layout(K, "circular")
        cmap = get_color_mapping(labels)
        for i in range(K):
            for j in range(K):
                if sub_adjacency[i, j] == 0:
                    continue
                freq = bootstrap_freq[i, j] if bootstrap_freq is not None else 0.7
                ax_graph.annotate(
                    "", xy=positions[j], xytext=positions[i],
                    arrowprops=dict(arrowstyle="-|>", color="#4477AA",
                                    lw=1 + 3 * freq, shrinkA=10, shrinkB=10),
                    zorder=1,
                )
            ax_graph.scatter(*positions[i], s=500, c=cmap[labels[i]],
                             edgecolors="white", linewidths=2, zorder=3)
            ax_graph.text(positions[i][0], positions[i][1] - 0.12, labels[i],
                          ha="center", fontsize=9, fontweight="bold", zorder=4)
        ax_graph.set_xlim(-1.5, 1.5)
        ax_graph.set_ylim(-1.5, 1.5)
        ax_graph.set_aspect("equal")
    ax_graph.axis("off")
    ax_graph.set_title("Sub-graph", fontsize=11)

    # --- 中: Bootstrap 频率柱 ---
    ax_freq = fig.add_subplot(gs[1])
    if sub_adjacency is not None and bootstrap_freq is not None:
        K = sub_adjacency.shape[0]
        labels = sub_labels or [f"N{i}" for i in range(K)]
        edges = []
        freqs = []
        for i in range(K):
            for j in range(K):
                if sub_adjacency[i, j]:
                    edges.append(f"{labels[i]}→{labels[j]}")
                    freqs.append(bootstrap_freq[i, j])
        y = range(len(edges))
        colors = [COLOR_GOOD if f >= 0.8 else COLOR_WARN if f >= 0.5 else COLOR_BAD for f in freqs]
        ax_freq.barh(y, freqs, color=colors, edgecolor="white", height=0.6)
        ax_freq.set_yticks(y)
        ax_freq.set_yticklabels(edges, fontsize=8)
        ax_freq.axvline(0.5, color=COLOR_WARN, linestyle="--", linewidth=1, alpha=0.7)
        ax_freq.axvline(0.8, color=COLOR_GOOD, linestyle="--", linewidth=1, alpha=0.7)
        ax_freq.set_xlim(0, 1.05)
    else:
        ax_freq.text(0.5, 0.5, "N/A", ha="center", va="center",
                     fontsize=14, color="#999999", transform=ax_freq.transAxes)
    ax_freq.set_title("Bootstrap Frequency", fontsize=11)
    ax_freq.set_xlabel("Frequency")

    # --- 右: 文字证据 ---
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis("off")
    text = supporting_text or "Evidence summary will be populated\nwhen causal analysis is complete."
    ax_text.text(0.05, 0.95, text, ha="left", va="top", fontsize=9,
                 wrap=True, transform=ax_text.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F8F8", edgecolor="#DDDDDD"))
    ax_text.set_title("Evidence", fontsize=11)

    fig.suptitle(f"Key Axis Evidence Card — {axis_name}", fontsize=14, fontweight="bold")
    add_watermark(ax_freq)

    if save_path:
        save_figure(fig, save_path, config={"chart": "key_axis_evidence", "axis": axis_name})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 因果指标仪表盘
# =========================================================================

def plot_causal_metrics_dashboard(
    metrics: dict[str, float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Step 2 因果指标 Dashboard

    Parameters
    ----------
    metrics : dict
        键如 "falsification_pvalue", "graph_sparsity", "known_axis_recall",
        "direction_accuracy", "mean_bootstrap_freq", "hsic"
    """
    metric_defs = [
        ("Falsification p ↑", "falsification_pvalue", 0.05, False, "p > 0.05"),
        ("Graph Sparsity ↓", "graph_sparsity", 0.1, True, "< 0.1"),
        ("Known Axis Recall ↑", "known_axis_recall", 0.6, False, "> 0.6"),
        ("Direction Accuracy ↑", "direction_accuracy", 0.8, False, "> 0.8"),
        ("Mean Bootstrap Freq ↑", "mean_bootstrap_freq", 0.5, False, "> 0.5"),
        ("HSIC(Z_int, Z_ext) ↓", "hsic", 0.01, True, "→ 0"),
    ]

    fig, axes = create_figure(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (name, key, thresh, inv, target_str) in enumerate(metric_defs):
        ax = axes[i]
        if metrics is not None and key in metrics:
            val = metrics[key]
            ok = val <= thresh if inv else val >= thresh
            color = COLOR_GOOD if ok else COLOR_BAD

            # 仪表风格
            ax.barh([0], [val], color=color, edgecolor="white", height=0.5)
            ax.axvline(thresh, color=COLOR_WARN, linestyle="--", linewidth=1.5)
            ax.text(val + 0.01, 0, f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
            ax.text(0.98, 0.95, f"Target: {target_str}", ha="right", va="top",
                    fontsize=8, color="#888888", transform=ax.transAxes)
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=14, color="#999999", transform=ax.transAxes, style="italic")
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.suptitle("Step 2 — Causal Edge Reliability Metrics", fontsize=14, fontweight="bold")
    add_watermark(axes[-1])

    if save_path:
        save_figure(fig, save_path, config={"chart": "causal_metrics_dashboard"})
    else:
        plt.close(fig)
    return fig
