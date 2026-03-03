#!/usr/bin/env python
"""阶段 2 展示图生成

生成图包:
    results/figures/step2/
    ├── causal_dag.png           — 因果有向图
    ├── causal_heatmap.png       — 因果邻接热图
    ├── signaling_flow.png       — 多层信号流
    ├── key_axis_evidence.png    — 关键轴证据卡 (per axis)
    ├── metrics_dashboard.png    — 因果指标仪表盘
    ├── disentangle_loss.png     — 解缠训练损失曲线
    └── disentangle_hsic.png     — HSIC 收敛曲线

借鉴 FlowSig / DoWhy 展示风格:
    - 分层布局（FlowSig multipartite_layout 风格）
    - 边宽 ∝ bootstrap 频率，边色 ∝ arrow strength
    - 带生物学解读标注
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="HyperSCA Stage 2: Generate Figures"
    )
    parser.add_argument("--input-dir", type=str, default="results/step2",
                        help="Step 2 output directory")
    parser.add_argument("--output-dir", type=str, default="results/figures/step2",
                        help="Figure output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("HyperSCA Stage 2: Generating Figures")
    print("=" * 60)
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # 加载产物
    adjacency = np.load(input_dir / "causal_adjacency.npy")
    freq_matrix = np.load(input_dir / "bootstrap_freq_matrix.npy")

    arrow_strength = None
    if (input_dir / "arrow_strength.npy").exists():
        arrow_strength = np.load(input_dir / "arrow_strength.npy")
    baseline_adj = None
    baseline_metrics = {}
    if (input_dir / "baseline_comm_adjacency.npy").exists():
        baseline_adj = np.load(input_dir / "baseline_comm_adjacency.npy")
    if (input_dir / "baseline_compare_metrics.json").exists():
        with open(input_dir / "baseline_compare_metrics.json", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    with open(input_dir / "node_info.json") as f:
        node_info = json.load(f)
    node_labels = node_info["node_labels"]
    type_mapping = node_info.get("type_mapping", {})

    with open(input_dir / "step2_metrics.json") as f:
        metrics = json.load(f)

    try:
        with open(input_dir / "key_axes_evidence.json", encoding="utf-8") as f:
            axis_results = json.load(f)
    except UnicodeDecodeError:
        with open(input_dir / "key_axes_evidence.json", encoding="utf-8-sig") as f:
            content = f.read()
        # 清理不可解码字符
        content = content.encode("utf-8", errors="replace").decode("utf-8")
        axis_results = json.loads(content)

    with open(input_dir / "signaling_flow_edges.json") as f:
        flow_edges = json.load(f)

    with open(input_dir / "disentangle_losses.json") as f:
        losses = json.load(f)

    # ---- 导入可视化模块 ----
    from src.visualization.causal import (
        plot_causal_dag,
        plot_signaling_flow,
        plot_causal_heatmap,
        plot_key_axis_evidence,
        plot_causal_metrics_dashboard,
    )
    from src.utils.plot_style import (
        CMAP_EXPRESSION,
        PALETTE_DISCRETE_HIGH_CONTRAST,
        apply_cns_style,
        create_figure,
        save_figure,
    )

    apply_cns_style()

    # ---- 1. 因果 DAG ----
    print("\n  [1/7] Causal DAG...")
    # 用 type_mapping 作为显示标签
    display_labels = [
        f"{label}\n({type_mapping.get(label, '')})"
        if type_mapping.get(label, label) != label
        else label
        for label in node_labels
    ]
    plot_causal_dag(
        adjacency=adjacency,
        node_labels=display_labels,
        bootstrap_freq=freq_matrix,
        arrow_strength=arrow_strength if arrow_strength is not None else freq_matrix,
        layout="circular",
        freq_threshold=0.3,
        save_path=str(output_dir / "causal_dag.png"),
    )

    # ---- 2. 因果热图 ----
    print("  [2/7] Causal heatmap...")
    plot_causal_heatmap(
        adjacency=adjacency,
        node_labels=node_labels,
        values=freq_matrix,
        value_name="Bootstrap Frequency",
        save_path=str(output_dir / "causal_heatmap_freq.png"),
    )
    if arrow_strength is not None:
        plot_causal_heatmap(
            adjacency=adjacency,
            node_labels=node_labels,
            values=arrow_strength,
            value_name="Arrow Strength",
            save_path=str(output_dir / "causal_heatmap_strength.png"),
        )

    # ---- 3. CNS 对比图（空间因果 vs 传统通讯）----
    print("  [3/10] CNS comparison figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.ravel()
    im1 = ax1.imshow(adjacency, cmap=CMAP_EXPRESSION, vmin=0, vmax=max(adjacency.max(), 1))
    ax1.set_title("A. Spatial-Constrained Causal Graph")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if baseline_adj is not None:
        im2 = ax2.imshow(baseline_adj, cmap=CMAP_EXPRESSION, vmin=0, vmax=max(baseline_adj.max(), 1))
        ax2.set_title("B. Traditional Communication Baseline")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    else:
        ax2.text(0.5, 0.5, "baseline_comm_adjacency.npy missing", ha="center", va="center")
        ax2.set_title("B. Traditional Communication Baseline")

    metric_labels = ["edge_density_causal", "edge_density_baseline", "edge_overlap_jaccard"]
    metric_vals = [float(baseline_metrics.get(k, 0.0)) for k in metric_labels]
    ax3.bar(
        range(len(metric_labels)),
        metric_vals,
        color=[
            PALETTE_DISCRETE_HIGH_CONTRAST[0],
            PALETTE_DISCRETE_HIGH_CONTRAST[1],
            PALETTE_DISCRETE_HIGH_CONTRAST[2],
        ],
    )
    ax3.set_xticks(range(len(metric_labels)))
    ax3.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax3.set_title("C. Quantitative Comparison")

    # 生态位+靶点空间分布: 使用节点类型占比近似展示
    type_vals = list(type_mapping.values())
    type_counts = {}
    for t in type_vals:
        type_counts[t] = type_counts.get(t, 0) + 1
    keys = list(type_counts.keys())[:12]
    vals = [type_counts[k] for k in keys]
    ax4.barh(keys, vals, color=PALETTE_DISCRETE_HIGH_CONTRAST[3])
    ax4.set_title("D. Niche Strata / Target Spatial Context")
    ax4.set_xlabel("Node count")

    fig.suptitle("CNS Figure (Step2): Spatial Causal Inference Advantage", fontsize=14)
    save_figure(fig, str(output_dir / "cns_step2_spatial_causal_advantage.png"),
                config={"chart": "cns_step2_spatial_causal_advantage"})

    # ---- 4. 空间约束有效性证据图 ----
    print("  [4/10] Spatial-constraint evidence...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    n_causal = int((adjacency > 0).sum())
    n_base = int((baseline_adj > 0).sum()) if baseline_adj is not None else 0
    mean_causal = float(adjacency[adjacency > 0].mean()) if np.any(adjacency > 0) else 0.0
    mean_base = float(baseline_adj[baseline_adj > 0].mean()) if baseline_adj is not None and np.any(baseline_adj > 0) else 0.0
    ax1.bar(
        ["baseline", "spatial_causal"],
        [n_base, n_causal],
        color=[PALETTE_DISCRETE_HIGH_CONTRAST[1], PALETTE_DISCRETE_HIGH_CONTRAST[0]],
        edgecolor="white",
    )
    ax1.set_title("A. Edge count after spatial constraints")
    ax1.set_ylabel("edge count")
    ax2.bar(
        ["baseline", "spatial_causal"],
        [mean_base, mean_causal],
        color=[PALETTE_DISCRETE_HIGH_CONTRAST[1], PALETTE_DISCRETE_HIGH_CONTRAST[0]],
        edgecolor="white",
    )
    ax2.set_title("B. Mean edge strength")
    ax2.set_ylabel("mean strength")
    fig.suptitle("Spatial Constraint Effectiveness")
    save_figure(fig, str(output_dir / "spatial_constraint_effectiveness.png"),
                config={"chart": "spatial_constraint_effectiveness"})

    # ---- 5. 方向性案例证据图 ----
    print("  [5/10] Directionality case figure...")
    type_of = lambda idx: type_mapping.get(node_labels[idx], node_labels[idx])

    def _dir_strength(src_type: str, tgt_type: str) -> float:
        src_idx = [i for i in range(len(node_labels)) if type_of(i) == src_type]
        tgt_idx = [i for i in range(len(node_labels)) if type_of(i) == tgt_type]
        if not src_idx or not tgt_idx:
            return 0.0
        mat = (arrow_strength if arrow_strength is not None else freq_matrix)
        sub = mat[np.ix_(src_idx, tgt_idx)]
        return float(sub.mean()) if sub.size > 0 else 0.0

    cases = [("CAF", "TAM"), ("CAF", "Treg"), ("TAM", "CD8T")]
    labels_case, forward_vals, reverse_vals = [], [], []
    for s, t in cases:
        labels_case.append(f"{s}->{t}")
        forward_vals.append(_dir_strength(s, t))
        reverse_vals.append(_dir_strength(t, s))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    x = np.arange(len(labels_case))
    w = 0.35
    ax1.bar(
        x - w / 2, forward_vals, width=w,
        color=PALETTE_DISCRETE_HIGH_CONTRAST[1], label="Forward",
    )
    ax1.bar(
        x + w / 2, reverse_vals, width=w,
        color=PALETTE_DISCRETE_HIGH_CONTRAST[0], label="Reverse",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_case, rotation=15)
    ax1.set_ylabel("mean directional strength")
    ax1.set_title("A. Forward vs reverse directional strength")
    ax1.legend()

    direction_idx = []
    for f, r in zip(forward_vals, reverse_vals):
        d = (f - r) / max(abs(f) + abs(r), 1e-8)
        direction_idx.append(d)
    colors = []
    for v in direction_idx:
        if abs(v) < 0.05:
            colors.append("#E0E0E0")
        elif v > 0:
            colors.append("#4575B4")
        else:
            colors.append("#D73027")
    ax2.bar(labels_case, direction_idx, color=colors, edgecolor="white")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylim(-1, 1)
    ax2.set_title("B. Directionality index (forward-reverse)")
    ax2.set_ylabel("index")

    fig.suptitle("Directionality Evidence (CAF/POSTN-related axes)")
    save_figure(fig, str(output_dir / "directionality_case_postn_myeloid.png"),
                config={"chart": "directionality_case_postn_myeloid"})

    # ---- 6. 信号流 ----
    print("  [6/10] Signaling flow...")
    if flow_edges:
        plot_signaling_flow(
            flow_edges=flow_edges,
            layer_names=["Ligand", "Receptor", "TF", "Target"],
            save_path=str(output_dir / "signaling_flow.png"),
        )
    else:
        plot_signaling_flow(
            save_path=str(output_dir / "signaling_flow.png"),
        )

    # ---- 7. 关键轴证据卡 ----
    print("  [7/10] Key axis evidence cards...")
    for i, ax in enumerate(axis_results.get("per_axis", [])):
        if not ax.get("found"):
            continue
        # 构建子图
        src_type = ax.get("source_type", "?")
        tgt_type = ax.get("target_type", "?")
        src_nodes = [
            j for j, l in enumerate(node_labels)
            if type_mapping.get(l, l) == src_type
        ]
        tgt_nodes = [
            j for j, l in enumerate(node_labels)
            if type_mapping.get(l, l) == tgt_type
        ]
        all_nodes = src_nodes + tgt_nodes
        if len(all_nodes) < 2:
            continue

        sub_adj = adjacency[np.ix_(all_nodes, all_nodes)]
        sub_freq = freq_matrix[np.ix_(all_nodes, all_nodes)]
        sub_labels = [node_labels[n] for n in all_nodes]

        safe_name = ax["name"].replace("→", "_to_").replace("/", "_").replace(" ", "_")
        plot_key_axis_evidence(
            axis_name=ax["name"],
            sub_adjacency=sub_adj,
            sub_labels=sub_labels,
            bootstrap_freq=sub_freq,
            arrow_strength=(
                arrow_strength[np.ix_(all_nodes, all_nodes)]
                if arrow_strength is not None else None
            ),
            supporting_text=ax.get("evidence", ""),
            save_path=str(output_dir / f"key_axis_{safe_name}.png"),
        )

    # ---- 8. 指标仪表盘 ----
    print("  [8/10] Metrics dashboard...")
    plot_causal_metrics_dashboard(
        metrics=metrics,
        save_path=str(output_dir / "metrics_dashboard.png"),
    )

    # ---- 9. 解缠损失曲线 ----
    print("  [9/10] Disentangle loss curves...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(losses["total"], color=PALETTE_DISCRETE_HIGH_CONTRAST[1], linewidth=1.5)
    axes[0].set_title("Total Loss", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(losses["recon"], color=PALETTE_DISCRETE_HIGH_CONTRAST[2], linewidth=1.5)
    axes[1].set_title("Reconstruction Loss", fontsize=11)
    axes[1].set_xlabel("Epoch")

    axes[2].plot(losses["hsic"], color=PALETTE_DISCRETE_HIGH_CONTRAST[0], linewidth=1.5)
    axes[2].set_title("HSIC(Z_int, Z_ext)", fontsize=11)
    axes[2].set_xlabel("Epoch")
    axes[2].axhline(
        y=0.01,
        color=PALETTE_DISCRETE_HIGH_CONTRAST[5],
        linestyle="--",
        alpha=0.7,
        label="target",
    )
    axes[2].legend(fontsize=9)

    fig.suptitle("Disentangle Model Training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, str(output_dir / "disentangle_loss.png"),
                config={"chart": "disentangle_loss"})

    # ---- 10. 图摘要文本 ----
    print("  [10/10] Summary text...")
    summary_lines = [
        f"Step 2 Figures Generated",
        f"========================",
        f"Nodes: {len(node_labels)}",
        f"Edges: {metrics.get('n_edges', 0)}",
        f"Sparsity: {metrics.get('graph_sparsity', 0):.4f}",
        f"Known Axis Recall: {metrics.get('known_axis_recall', 0):.2%}",
        f"Direction Accuracy: {metrics.get('direction_accuracy', 0):.2%}",
        f"SpatialConstraint_Edges: causal={n_causal}, baseline={n_base}",
        f"HSIC: {metrics.get('hsic_z_int_z_ext', 'N/A')}",
    ]
    with open(output_dir / "figure_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n[DONE] All figures saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
