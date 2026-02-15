"""Example 04: Xenium 基因面板摘要

提供实验元信息解析、面板统计以及可视化模板：
- Descriptor 构成环形图（gene / negative control / blank codeword 等）
- Source panel 堆叠条形图
- 面板组成综合 Dashboard
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plot_style import (
    PALETTE_CATEGORICAL,
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

def parse_experiment_info(experiment: dict) -> dict:
    """提取实验元信息关键字段"""
    return {
        "run_name": experiment.get("run_name", ""),
        "region_name": experiment.get("region_name", ""),
        "preservation_method": experiment.get("preservation_method", ""),
        "num_cells": experiment.get("num_cells", 0),
        "transcripts_per_cell": experiment.get("transcripts_per_cell", 0),
        "transcripts_per_100um": experiment.get("transcripts_per_100um", 0),
        "panel_name": experiment.get("panel_name", ""),
        "panel_organism": experiment.get("panel_organism", ""),
        "panel_tissue_type": experiment.get("panel_tissue_type", ""),
        "panel_num_targets_predesigned": experiment.get("panel_num_targets_predesigned", 0),
        "panel_num_targets_custom": experiment.get("panel_num_targets_custom", 0),
        "pixel_size": experiment.get("pixel_size", 0),
        "analysis_sw_version": experiment.get("analysis_sw_version", ""),
    }


def parse_targets(targets: list[dict]) -> pd.DataFrame:
    """解析基因面板 targets 为 DataFrame

    Returns
    -------
    pd.DataFrame
        列: gene_name, ensembl_id, descriptor, source_panel
    """
    rows = []
    for t in targets:
        td = t.get("type", {})
        descriptor = td.get("descriptor", "unknown")
        data = td.get("data", {})
        src = t.get("source", {}).get("identity", {})
        rows.append({
            "gene_name": data.get("name", ""),
            "ensembl_id": data.get("id", ""),
            "descriptor": descriptor,
            "source_panel": src.get("name", ""),
        })
    return pd.DataFrame(rows)


def panel_stats(targets_df: pd.DataFrame) -> dict:
    """面板统计"""
    return {
        "total_targets": len(targets_df),
        "gene_targets": int((targets_df["descriptor"] == "gene").sum()),
        "other_targets": int((targets_df["descriptor"] != "gene").sum()),
        "unique_sources": targets_df["source_panel"].nunique(),
        "sources": targets_df["source_panel"].value_counts().to_dict(),
    }


def generate_report_md(exp_info: dict, stats: dict) -> str:
    """生成可读的 Markdown 实验报告"""
    lines = [
        "# Xenium Experiment Report",
        "",
        "## Experiment Info",
        "",
        f"- **Run**: {exp_info['run_name']}",
        f"- **Region**: {exp_info['region_name']}",
        f"- **Preservation**: {exp_info['preservation_method']}",
        f"- **Num Cells**: {exp_info['num_cells']:,}",
        f"- **Transcripts/Cell**: {exp_info['transcripts_per_cell']}",
        f"- **Transcripts/100um**: {exp_info['transcripts_per_100um']:.1f}",
        f"- **Panel**: {exp_info['panel_name']}",
        f"- **Organism**: {exp_info['panel_organism']}",
        f"- **Tissue**: {exp_info['panel_tissue_type']}",
        f"- **Predesigned Targets**: {exp_info['panel_num_targets_predesigned']}",
        f"- **Custom Targets**: {exp_info['panel_num_targets_custom']}",
        f"- **Pixel Size**: {exp_info['pixel_size']} um",
        f"- **Software**: {exp_info['analysis_sw_version']}",
        "",
        "## Panel Statistics",
        "",
        f"- **Total Targets**: {stats['total_targets']}",
        f"- **Gene Targets**: {stats['gene_targets']}",
        f"- **Other Targets**: {stats['other_targets']}",
        "",
        "### Source Panels",
        "",
    ]
    for src, cnt in stats.get("sources", {}).items():
        lines.append(f"- {src}: {cnt}")
    lines.append("")
    return "\n".join(lines)


# =========================================================================
# 可视化 — Descriptor 构成环形图
# =========================================================================

def plot_descriptor_donut(
    targets_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Descriptor 类型（gene / negative control / blank codeword 等）环形图

    Parameters
    ----------
    targets_df : pd.DataFrame
        parse_targets 的输出
    """
    desc_counts = targets_df["descriptor"].value_counts()
    labels = desc_counts.index.tolist()
    sizes = desc_counts.values
    cmap = get_color_mapping(labels)
    colors = [cmap[lab] for lab in labels]

    fig, ax = create_figure(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * sum(sizes)))})",
        pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)

    # 中心文字
    ax.text(0, 0, f"Total\n{sum(sizes)}", ha="center", va="center",
            fontsize=14, fontweight="bold")

    ax.set_title("Xenium Panel: Target Descriptor Composition", pad=15)

    # 图例
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=cmap[lab], edgecolor="white", label=f"{lab} ({cnt})")
        for lab, cnt in zip(labels, sizes)
    ]
    ax.legend(
        handles=legend_handles, title="Descriptor",
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=9, title_fontsize=10,
    )

    add_watermark(ax)
    if save_path:
        save_figure(fig, save_path, config={"chart": "descriptor_donut"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — Source Panel 条形图
# =========================================================================

def plot_source_bar(
    targets_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Source panel 来源统计水平条形图

    Parameters
    ----------
    targets_df : pd.DataFrame
        parse_targets 的输出
    """
    src_counts = targets_df["source_panel"].value_counts().sort_values(ascending=True)
    cmap = get_color_mapping(src_counts.index.tolist())
    colors = [cmap[lab] for lab in src_counts.index]

    fig, ax = create_figure(figsize=(9, max(4, len(src_counts) * 0.6)))

    ax.barh(
        range(len(src_counts)), src_counts.values,
        color=colors, edgecolor="white", height=0.65,
    )
    ax.set_yticks(range(len(src_counts)))
    ax.set_yticklabels(src_counts.index, fontsize=9)

    # 数值标注
    max_val = src_counts.max()
    for i, val in enumerate(src_counts.values):
        ax.text(val + max_val * 0.01, i, f"{val}", va="center", fontsize=9)

    ax.set_xlabel("Number of Targets")
    ax.set_title("Xenium Panel: Targets by Source Panel")
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "source_bar"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 面板组成综合 Dashboard（descriptor × source 堆叠）
# =========================================================================

def plot_panel_composition_dashboard(
    targets_df: pd.DataFrame,
    exp_info: dict | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """综合 2-panel Dashboard

    左: Descriptor 环形图
    右: Source × Descriptor 堆叠水平条形图

    Parameters
    ----------
    targets_df : pd.DataFrame
        parse_targets 的输出
    exp_info : dict, optional
        实验元信息（用于标题补充）
    """
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.35)

    # --- 左: Descriptor 环形 ---
    ax_left = fig.add_subplot(gs[0])
    desc_counts = targets_df["descriptor"].value_counts()
    desc_labels = desc_counts.index.tolist()
    desc_cmap = get_color_mapping(desc_labels)
    desc_colors = [desc_cmap[lab] for lab in desc_labels]

    ax_left.pie(
        desc_counts.values,
        labels=None,
        colors=desc_colors,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
    )
    ax_left.text(0, 0, f"Total\n{desc_counts.sum()}", ha="center", va="center",
                 fontsize=13, fontweight="bold")
    ax_left.set_title("Descriptor Composition")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=desc_cmap[lab], edgecolor="white", label=lab)
        for lab in desc_labels
    ]
    ax_left.legend(
        handles=legend_handles, loc="upper left",
        fontsize=8, frameon=True, framealpha=0.8,
    )

    # --- 右: Source × Descriptor 堆叠条形 ---
    ax_right = fig.add_subplot(gs[1])
    cross = pd.crosstab(targets_df["source_panel"], targets_df["descriptor"])
    cross = cross.loc[cross.sum(axis=1).sort_values(ascending=True).index]

    bottom = np.zeros(len(cross))
    for desc in cross.columns:
        color = desc_cmap.get(desc, "#BBBBBB")
        vals = cross[desc].values
        ax_right.barh(
            range(len(cross)), vals,
            left=bottom, color=color, edgecolor="white",
            height=0.65, label=desc,
        )
        bottom += vals

    ax_right.set_yticks(range(len(cross)))
    ax_right.set_yticklabels(cross.index, fontsize=9)
    ax_right.set_xlabel("Number of Targets")
    ax_right.set_title("Source Panel × Descriptor")
    ax_right.legend(fontsize=8, title="Descriptor", title_fontsize=9)

    # 全图标题
    title = "Xenium Gene Panel Composition"
    if exp_info:
        title += f"  —  {exp_info.get('panel_name', '')}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    add_watermark(ax_right)
    if save_path:
        save_figure(fig, save_path, config={"chart": "panel_composition_dashboard"})
    else:
        plt.close(fig)
    return fig
