"""MVP Integration 展示图生成

Supports the new mode-specific directory layout:
  mvp/
    hyperbolic/  (step2/, step3/, geometry/)
    euclidean/   (step2/, step3/, geometry/)
    run_summary.json

生成图包 (results/figures/integration/):
  1. causal_dag.png            — 因果 DAG（先验边高亮）
  2. causal_heatmap.png        — 因果邻接热图
  3. signaling_flow.png        — 4 层信号流 (Ligand→Receptor→TF→Target)
  4. poincare_embedding.png    — Poincaré 盘双曲嵌入
  5. propagation_*.png         — 动态靶点空间传播梯度
  6. target_ranking_*.png      — 动态靶点互作排序
  7. metrics_comparison.png    — Hyperbolic vs Euclidean 对比雷达图
  8. evidence_summary.png      — 靶点跨数据集证据仪表盘
  9. disentangle_loss.png      — 解缠训练曲线
 10. pipeline_overview.png     — MVP 流水线总览示意
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.plot_style import apply_style, save_figure

MVP_DIR = ROOT / "results" / "integration" / "mvp"
FIG_DIR = ROOT / "results" / "figures" / "integration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TME_PALETTE = {
    "CAF": "#E64B35",
    "TAM": "#4DBBD5",
    "CD4T": "#00A087",
    "CD8T": "#3C5488",
    "Treg": "#F39B7F",
    "DC": "#8491B4",
    "Neutrophil": "#91D1C2",
    "Endothelial": "#B09C85",
    "Monocyte": "#7E6148",
    "NK": "#E377C2",
    "Mast": "#BCBD22",
    "Stromal": "#FF7F0E",
}

ANCHOR_TARGETS = ["MFAP2", "POSTN", "INHBA"]


def _find_mode_dir(base: Path) -> Path:
    """Return the best available mode directory (prefer hyperbolic)."""
    for m in ["hyperbolic", "euclidean"]:
        d = base / m
        if (d / "step2" / "causal_adjacency.npy").exists():
            return d
    if (base / "step2" / "causal_adjacency.npy").exists():
        return base
    return base / "hyperbolic"


def _discover_targets(s3_dir: Path) -> list[str]:
    """Read target gene list from step3_metrics.json."""
    m_path = s3_dir / "step3_metrics.json"
    if m_path.exists():
        m = json.loads(m_path.read_text(encoding="utf-8"))
        return m.get("targets", ANCHOR_TARGETS)
    prop_files = sorted(s3_dir.glob("propagation_*.json"))
    return [f.stem.replace("propagation_", "") for f in prop_files] or ANCHOR_TARGETS


def load_results(base_dir: Path) -> dict:
    """Load Step2/Step3 results from mode-aware directory layout."""
    mode_dir = _find_mode_dir(base_dir)
    s2 = mode_dir / "step2"
    s3 = mode_dir / "step3"
    geo = mode_dir / "geometry"

    r: dict = {"mode_dir": mode_dir}

    for name, fname in [("adj", "causal_adjacency.npy"),
                         ("freq", "bootstrap_freq_matrix.npy"),
                         ("freq", "bootstrap_freq.npy"),
                         ("z_int", "z_int.npy"), ("z_ext", "z_ext.npy")]:
        f = s2 / fname
        if f.exists() and name not in r:
            r[name] = np.load(f)

    for name, fname in [("info", "node_info.json"), ("metrics", "step2_metrics.json"),
                         ("flow", "signaling_flow_edges.json"),
                         ("flow", "flow_edges.json"),
                         ("axes", "key_axes_evidence.json"),
                         ("axes", "axis_results.json"),
                         ("losses", "disentangle_losses.json"),
                         ("losses", "losses.json")]:
        f = s2 / fname
        if f.exists() and name not in r:
            r[name] = json.loads(f.read_text(encoding="utf-8"))

    s3_met = s3 / "step3_metrics.json"
    if s3_met.exists():
        r["s3_metrics"] = json.loads(s3_met.read_text(encoding="utf-8"))
    else:
        r["s3_metrics"] = {"per_target": {}}

    # Geometry embeddings (new layout: geometry/ under mode_dir)
    if geo.exists():
        for emb_name in ["embedding.csv", "hyperbolic_embedding.csv"]:
            ef = geo / emb_name
            if ef.exists() and "hyp_emb" not in r:
                r["hyp_emb"] = pd.read_csv(ef, index_col=0)
        for mf in ["metrics.json", "hyperbolic_metrics.json"]:
            mfp = geo / mf
            if mfp.exists() and "hyp_metrics" not in r:
                r["hyp_metrics"] = json.loads(mfp.read_text(encoding="utf-8"))

    # Try loading euclidean mode embeddings
    euc_dir = base_dir / "euclidean" / "geometry"
    if euc_dir.exists():
        for emb_name in ["embedding.csv", "euclidean_embedding.csv"]:
            ef = euc_dir / emb_name
            if ef.exists() and "euc_emb" not in r:
                r["euc_emb"] = pd.read_csv(ef, index_col=0)
        for mf in ["metrics.json", "euclidean_metrics.json"]:
            mfp = euc_dir / mf
            if mfp.exists() and "euc_metrics" not in r:
                r["euc_metrics"] = json.loads(mfp.read_text(encoding="utf-8"))

    # Dynamic target list
    targets = _discover_targets(s3)
    r["target_genes"] = targets
    for tgt in targets:
        prop_f = s3 / f"propagation_{tgt}.json"
        if prop_f.exists():
            r[f"prop_{tgt}"] = json.loads(prop_f.read_text(encoding="utf-8"))
        rank_f = s3 / f"interaction_targets_{tgt}.csv"
        if rank_f.exists():
            r[f"rank_{tgt}"] = pd.read_csv(rank_f)

    return r


# ────────────────────────────────────────────────────────────────────
# Fig 1: Causal DAG
# ────────────────────────────────────────────────────────────────────
def fig_causal_dag(r: dict):
    from src.visualization.causal import plot_causal_dag
    labels = r["info"]["node_labels"]
    tm = r["info"].get("type_mapping", {})
    display = [f"{l}\n({tm.get(l, '')})" if tm.get(l, l) != l else l for l in labels]
    plot_causal_dag(
        adjacency=r["adj"], node_labels=display,
        bootstrap_freq=r["freq"],
        arrow_strength=r["freq"],
        layout="circular", freq_threshold=0.0,
        save_path=str(FIG_DIR / "causal_dag.png"),
    )
    print("  [1] causal_dag.png")


# ────────────────────────────────────────────────────────────────────
# Fig 2: Causal Heatmap
# ────────────────────────────────────────────────────────────────────
def fig_causal_heatmap(r: dict):
    from src.visualization.causal import plot_causal_heatmap
    plot_causal_heatmap(
        adjacency=r["adj"], node_labels=r["info"]["node_labels"],
        values=r["freq"], value_name="Bootstrap Frequency",
        save_path=str(FIG_DIR / "causal_heatmap.png"),
    )
    print("  [2] causal_heatmap.png")


# ────────────────────────────────────────────────────────────────────
# Fig 3: Signaling Flow
# ────────────────────────────────────────────────────────────────────
def fig_signaling_flow(r: dict):
    from src.visualization.causal import plot_signaling_flow
    plot_signaling_flow(
        flow_edges=r["flow"],
        layer_names=["Ligand", "Receptor", "TF", "Target"],
        save_path=str(FIG_DIR / "signaling_flow.png"),
    )
    print("  [3] signaling_flow.png")


# ────────────────────────────────────────────────────────────────────
# Fig 4: Poincaré Disk Embedding
# ────────────────────────────────────────────────────────────────────
def fig_poincare_embedding(r: dict):
    if "hyp_emb" not in r:
        print("  [4] SKIP poincare_embedding.png (no data)")
        return
    from src.visualization.hyperbolic import plot_poincare_disk
    labels = list(r["hyp_emb"].index)
    tm = r["info"].get("type_mapping", {})
    type_labels = [tm.get(l, l) for l in labels]
    emb = r["hyp_emb"].values
    plot_poincare_disk(
        embeddings=emb, labels=type_labels, label_name="TME Role",
        palette=TME_PALETTE, point_size=120,
        save_path=str(FIG_DIR / "poincare_embedding.png"),
    )
    print("  [4] poincare_embedding.png")


# ────────────────────────────────────────────────────────────────────
# Fig 5: Propagation Gradient (per target)
# ────────────────────────────────────────────────────────────────────
def fig_propagation(r: dict):
    for tgt in r.get("target_genes", ANCHOR_TARGETS):
        prop = r.get(f"prop_{tgt}")
        if not prop or not prop.get("bfs_layers"):
            print(f"  [5] SKIP propagation_{tgt}.png (no data)")
            continue
        bfs = prop["bfs_layers"]
        hops = [layer["hop"] for layer in bfs]
        effects = [layer["mean_effect"] for layer in bfs]
        node_labels = r["info"]["node_labels"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        colors = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F"]
        bars = ax1.bar(hops, effects, color=[colors[min(h, 3)] for h in hops],
                       edgecolor="white", linewidth=0.8, width=0.6)
        ax1.set_xlabel("BFS Hop Distance", fontsize=11)
        ax1.set_ylabel("Mean Effect Magnitude", fontsize=11)
        ax1.set_title(f"{tgt} KO — Propagation by Hop", fontsize=12, fontweight="bold")
        ax1.set_xticks(hops)
        max_eff = max(effects) if effects else 1.0
        for i, (h, e) in enumerate(zip(hops, effects)):
            nodes = [node_labels[n] for n in bfs[i]["nodes"] if n < len(node_labels)]
            label = ", ".join(nodes[:2])
            if len(nodes) > 2:
                label += f" +{len(nodes)-2}"
            y_off = 0.04 * max_eff + (0.02 * max_eff * (i % 2))
            ax1.text(h, e + y_off, label,
                     ha="center", va="bottom", fontsize=6, rotation=25,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white",
                               ec="none", alpha=0.6))

        fit = prop.get("fit_params", {})
        if fit.get("amplitude") is not None:
            x_fit = np.linspace(0, max(hops) + 0.5, 50)
            amp = fit["amplitude"]
            length = fit.get("length_scale", 1.0)
            y_fit = amp * np.exp(-x_fit**2 / (2 * length**2))
            ax2.plot(x_fit, y_fit, "r--", linewidth=2, label=f"Fit: A={amp:.2f}, l={length:.2f}")
            ax2.scatter(hops, effects, c=[colors[min(h, 3)] for h in hops],
                        s=80, zorder=5, edgecolors="black", linewidth=0.5)
            r2 = r["s3_metrics"].get("per_target", {}).get(tgt, {}).get(
                "spatial_quality", {}).get("gradient_decay_r2", 0)
            ax2.set_title(f"Gradient Decay (R²={r2:.3f})", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Hop Distance")
            ax2.set_ylabel("Effect Magnitude")
            ax2.legend(fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No fit data", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=12, color="gray")

        fig.suptitle(f"Spatial Propagation: {tgt} Knockout", fontsize=13)
        fig.tight_layout()
        save_figure(fig, str(FIG_DIR / f"propagation_{tgt}.png"),
                    config={"target": tgt})
        print(f"  [5] propagation_{tgt}.png")


# ────────────────────────────────────────────────────────────────────
# Fig 6: Target Ranking (per target)
# ────────────────────────────────────────────────────────────────────
def fig_target_ranking(r: dict):
    for tgt in r.get("target_genes", ANCHOR_TARGETS):
        ranked = r.get(f"rank_{tgt}")
        if ranked is None or ranked.empty:
            print(f"  [6] SKIP target_ranking_{tgt}.png (no data)")
            continue
        from src.visualization.perturbation import plot_interaction_target_ranking
        plot_interaction_target_ranking(
            ranked_targets=ranked, top_n=min(10, len(ranked)),
            score_col="target_priority_score",
            save_path=str(FIG_DIR / f"target_ranking_{tgt}.png"),
        )
        print(f"  [6] target_ranking_{tgt}.png")


# ────────────────────────────────────────────────────────────────────
# Fig 7: Metrics Comparison (Hyperbolic vs Euclidean radar)
# ────────────────────────────────────────────────────────────────────
def fig_metrics_comparison(r: dict):
    h_s2 = r.get("hyp_s2_metrics")
    e_s2 = r.get("euc_s2_metrics")
    if not h_s2 or not e_s2:
        print("  [7] SKIP metrics_comparison.png (need both modes)")
        return

    categories = [
        "HSIC\n(lower=better)", "z_ext R²", "z_int R²",
        "Falsification\np-value", "Gradient R²\n(POSTN)", "Gradient R²\n(INHBA)",
    ]
    h_vals = [
        1.0 - min(h_s2.get("hsic_z_int_z_ext", 0) * 100, 1.0),
        h_s2.get("z_ext_neighbor_r2", 0),
        h_s2.get("z_int_neighbor_r2", 0),
        min(h_s2.get("falsification_pvalue", 0) * 100, 1.0),
        r.get("hyp_postn_grad_r2", 0.5),
        r.get("hyp_inhba_grad_r2", 0.5),
    ]
    e_vals = [
        1.0 - min(e_s2.get("hsic_z_int_z_ext", 0) * 100, 1.0),
        e_s2.get("z_ext_neighbor_r2", 0),
        e_s2.get("z_int_neighbor_r2", 0),
        min(e_s2.get("falsification_pvalue", 0) * 100, 1.0),
        r.get("euc_postn_grad_r2", 0.5),
        r.get("euc_inhba_grad_r2", 0.5),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    h_vals += h_vals[:1]
    e_vals += e_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, h_vals, "o-", linewidth=2, label="Hyperbolic", color="#E64B35")
    ax.fill(angles, h_vals, alpha=0.15, color="#E64B35")
    ax.plot(angles, e_vals, "s-", linewidth=2, label="Euclidean", color="#4DBBD5")
    ax.fill(angles, e_vals, alpha=0.15, color="#4DBBD5")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Hyperbolic vs Euclidean: Key Metrics", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    save_figure(fig, str(FIG_DIR / "metrics_comparison.png"),
                config={"chart": "radar_comparison"})
    print("  [7] metrics_comparison.png")


# ────────────────────────────────────────────────────────────────────
# Fig 8: Three-target Evidence Summary
# ────────────────────────────────────────────────────────────────────
def fig_evidence_summary(r: dict):
    targets = r.get("target_genes", ANCHOR_TARGETS)[:6]
    s3 = r.get("s3_metrics", {}).get("per_target", {})

    n_cols = min(len(targets), 3)
    n_rows = max(1, (len(targets) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    target_colors = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4", "#91D1C2"]

    for idx, tgt in enumerate(targets):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i][col_i]
        tm = s3.get(tgt, {})
        cf = tm.get("cf_quality", {})
        sp = tm.get("spatial_quality", {})
        n_ranked = tm.get("n_ranked_targets", 0)

        metric_names = ["R²(mean)", "Dir. Acc.", "Grad. R²", "Prop. Depth", "Ranked\nTargets"]
        raw_vals = [
            cf.get("r2_mean", 0),
            cf.get("marker_direction_accuracy", 0),
            sp.get("gradient_decay_r2", 0),
            sp.get("propagation_depth", 0),
            n_ranked,
        ]
        bar_vals = [
            raw_vals[0],
            raw_vals[1],
            raw_vals[2],
            raw_vals[3] / 4.0,
            min(raw_vals[4] / 10.0, 1.0),
        ]

        bar_colors = ["#3C5488", "#E64B35", "#00A087", "#F39B7F", "#4DBBD5"]
        bars = ax.barh(metric_names, bar_vals, color=bar_colors,
                       edgecolor="white", height=0.6)

        for bar_idx, (bar, bv, rv) in enumerate(zip(bars, bar_vals, raw_vals)):
            display_v = rv if bar_idx == 3 else (rv if bar_idx == 4 else bv)
            label = f"{display_v:.3f}" if display_v < 10 else f"{int(display_v)}"
            ax.text(min(bv + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=8)

        ax.set_xlim(0, 1.15)
        tag = " *" if tgt in ANCHOR_TARGETS else ""
        ax.set_title(f"{tgt}{tag}", fontsize=12, fontweight="bold",
                     color=target_colors[idx % len(target_colors)])
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.3)

    for idx in range(len(targets), n_rows * n_cols):
        row_i, col_i = divmod(idx, n_cols)
        axes[row_i][col_i].set_visible(False)

    fig.suptitle("Target Evidence Dashboard (* = anchor)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, str(FIG_DIR / "evidence_summary.png"),
                config={"chart": "evidence_dashboard"})
    print("  [8] evidence_summary.png")


# ────────────────────────────────────────────────────────────────────
# Fig 9: Disentangle Loss Curves
# ────────────────────────────────────────────────────────────────────
def fig_disentangle_loss(r: dict):
    losses = r.get("losses", {})
    if not losses:
        print("  [9] SKIP disentangle_loss.png (no data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses["total"], color="#4477AA", linewidth=1.5)
    axes[0].set_title("Total Loss", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    axes[1].plot(losses["recon"], color="#228833", linewidth=1.5)
    axes[1].set_title("Reconstruction Loss", fontsize=11)
    axes[1].set_xlabel("Epoch")

    axes[2].plot(losses["hsic"], color="#EE6677", linewidth=1.5)
    axes[2].set_title("HSIC(z_int, z_ext)", fontsize=11)
    axes[2].set_xlabel("Epoch")
    axes[2].axhline(y=0.01, color="#CCBB44", linestyle="--", alpha=0.7, label="target")
    axes[2].legend(fontsize=9)

    fig.suptitle("Disentangle Model Training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, str(FIG_DIR / "disentangle_loss.png"),
                config={"chart": "disentangle_loss"})
    print("  [9] disentangle_loss.png")


# ────────────────────────────────────────────────────────────────────
# Fig 10: Pipeline Overview Schematic
# ────────────────────────────────────────────────────────────────────
def fig_pipeline_overview(r: dict):
    targets = r.get("target_genes", ANCHOR_TARGETS)
    n_targets = len(targets)
    anchor_str = ", ".join(t for t in targets if t in ANCHOR_TARGETS)
    disc_str = ", ".join(t for t in targets if t not in ANCHOR_TARGETS)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        (0.3, 5.2, 2.0, 1.2, "#E8D5B7",
         "Phase A\nCluster Expression\nscCRC_Neu pseudo-bulk"),
        (0.3, 3.2, 2.0, 1.2, "#B7D5E8",
         "Phase B\nSpatial Adjacency\nST_CRC_MSS + IFNG"),
        (0.3, 1.2, 2.0, 1.2, "#D5E8B7",
         "Phase B.5\nDual Geometry\nHyperbolic + Euclidean"),
        (3.3, 3.5, 2.2, 2.5, "#FFE0E0",
         "Phase C: Step2\nDisentangle (GCN+MLP)\nPC Algorithm ×100\nDoWhy Validation\nSignaling Flow"),
        (6.5, 3.5, 2.6, 2.5, "#E0E0FF",
         f"Phase D: Step3\n{n_targets} targets KO\nAnchors: {anchor_str}\n"
         f"Discovered: {disc_str[:30]}\nSpatial Propagation"),
        (6.5, 1.2, 2.6, 1.2, "#E0FFE0",
         "Phase E\nEvidence Report\nMMR Stratification"),
        (3.3, 1.2, 2.2, 1.2, "#FFF0D0",
         "Niche Clustering\nCross-sample metrics\nMMR/MSI annotation"),
    ]

    for x, y, w, h, color, text in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#555555", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.5, fontweight="bold", linespacing=1.4)

    arrows = [
        ((2.3, 5.8), (3.3, 5.2)),
        ((2.3, 3.8), (3.3, 4.5)),
        ((2.3, 1.8), (3.3, 3.8)),
        ((5.5, 4.75), (6.5, 4.75)),
        ((8.0, 3.5), (8.0, 2.4)),
        ((5.5, 1.8), (6.5, 1.8)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    lw=2, connectionstyle="arc3,rad=0.05"))

    ax.set_title("HyperSCA MVP Integration Pipeline Overview",
                 fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    save_figure(fig, str(FIG_DIR / "pipeline_overview.png"),
                config={"chart": "pipeline_overview"})
    print("  [10] pipeline_overview.png")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main():
    print("HyperSCA MVP Integration: Generating Figures")
    print("=" * 60)

    apply_style()
    r = load_results(MVP_DIR)

    # Load both-mode metrics for comparison figure
    for mode_name, prefix in [("hyperbolic", "hyp"), ("euclidean", "euc")]:
        mode_s2 = MVP_DIR / mode_name / "step2" / "step2_metrics.json"
        if mode_s2.exists():
            r[f"{prefix}_s2_metrics"] = json.loads(mode_s2.read_text(encoding="utf-8"))
        mode_s3 = MVP_DIR / mode_name / "step3" / "step3_metrics.json"
        if mode_s3.exists():
            s3m = json.loads(mode_s3.read_text(encoding="utf-8"))
            targets = r.get("target_genes", ANCHOR_TARGETS)
            for tgt in targets[:3]:
                gr2 = s3m.get("per_target", {}).get(tgt, {}).get(
                    "spatial_quality", {}).get("gradient_decay_r2", 0.5)
                r[f"{prefix}_{tgt.lower()}_grad_r2"] = gr2

    fig_causal_dag(r)
    fig_causal_heatmap(r)
    fig_signaling_flow(r)
    fig_poincare_embedding(r)
    fig_propagation(r)
    fig_target_ranking(r)
    fig_metrics_comparison(r)
    fig_evidence_summary(r)
    fig_disentangle_loss(r)
    fig_pipeline_overview(r)

    print(f"\n[DONE] All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
