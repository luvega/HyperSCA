"""Example 01: Chromium scRNA-seq 细胞元数据 QC 与统计

提供细胞过滤、类型统计以及多种可视化模板：
- 基础条形图 (Level1)
- 嵌套条形图 (Level1 × Level2)
- Sunburst 环形图 (Level1 → Level2 层级)
- 患者 QC 堆叠条形图
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plot_style import (
    PALETTE_CATEGORICAL,
    PALETTE_CELLTYPE,
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

def filter_kept_cells(df: pd.DataFrame) -> pd.DataFrame:
    """保留 QCFilter == 'Keep' 的细胞"""
    return df[df["QCFilter"] == "Keep"].copy()


def celltype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按 Level1 / Level2 统计细胞数量与比例

    Returns
    -------
    pd.DataFrame
        列: Level1, Level2, count, fraction
    """
    counts = (
        df.groupby(["Level1", "Level2"])
        .size()
        .reset_index(name="count")
    )
    counts["fraction"] = counts["count"] / counts["count"].sum()
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按患者统计细胞数

    Returns
    -------
    pd.DataFrame
        列: Patient, total, kept, removed
    """
    total = df.groupby("Patient").size().rename("total")
    kept = df[df["QCFilter"] == "Keep"].groupby("Patient").size().rename("kept")
    removed = df[df["QCFilter"] != "Keep"].groupby("Patient").size().rename("removed")
    summary = pd.concat([total, kept, removed], axis=1).fillna(0).astype(int)
    return summary.reset_index()


# =========================================================================
# 可视化 — 基础条形图（保留向后兼容）
# =========================================================================

def plot_celltype_bar(summary: pd.DataFrame, save_path: str | None = None):
    """绘制 Level1 细胞类型条形图（基础版，向后兼容）"""
    level1 = summary.groupby("Level1")["count"].sum().sort_values(ascending=True)
    cmap = get_color_mapping(level1.index.tolist(), PALETTE_CELLTYPE)
    colors = [cmap[lab] for lab in level1.index]

    fig, ax = create_figure(figsize=(8, 5))
    level1.plot.barh(ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cell Count")
    ax.set_title("Chromium scRNA-seq: Cell Type Distribution (Level1)")
    add_watermark(ax)
    if save_path:
        save_figure(fig, save_path, config={"chart": "celltype_bar_level1"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 嵌套条形图 (Level1 × Level2)
# =========================================================================

def plot_celltype_nested_bar(
    summary: pd.DataFrame,
    save_path: str | None = None,
    top_n: int | None = None,
) -> plt.Figure:
    """绘制嵌套分组水平条形图

    每个 Level1 大类作为一个组，组内列出 Level2 子类型条形。

    Parameters
    ----------
    summary : pd.DataFrame
        celltype_summary 的输出
    save_path : str, optional
        保存路径
    top_n : int, optional
        只展示每个 Level1 内 top_n 的 Level2
    """
    df = summary.copy()
    if top_n:
        df = (
            df.sort_values("count", ascending=False)
            .groupby("Level1")
            .head(top_n)
        )

    # 排序: 按 Level1 总量降序，组内按 count 降序
    level1_order = (
        df.groupby("Level1")["count"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df["Level1"] = pd.Categorical(df["Level1"], categories=level1_order, ordered=True)
    df = df.sort_values(["Level1", "count"], ascending=[True, False])

    # 颜色
    l1_colors = get_color_mapping(level1_order, PALETTE_CELLTYPE)

    # 构建嵌套行标签
    labels = []
    group_positions = {}  # Level1 -> y center
    y = 0
    yticks, yticklabels = [], []
    bar_colors = []
    bar_values = []
    bar_fractions = []

    for l1 in level1_order:
        sub = df[df["Level1"] == l1].sort_values("count", ascending=True)
        start_y = y
        base_color = l1_colors[l1]
        for _, row in sub.iterrows():
            yticks.append(y)
            yticklabels.append(f"  {row['Level2']}")
            bar_values.append(row["count"])
            bar_fractions.append(row["fraction"])
            bar_colors.append(base_color)
            y += 1
        group_positions[l1] = (start_y + y - 1) / 2
        y += 0.8  # 组间间隔

    n_bars = len(bar_values)
    height = max(6, n_bars * 0.35 + 2)
    fig, ax = create_figure(figsize=(10, height))

    ax.barh(
        range(len(bar_values)), bar_values,
        color=bar_colors, edgecolor="white", linewidth=0.4, height=0.75,
    )

    # 数值标注
    max_val = max(bar_values) if bar_values else 1
    for i, (val, frac) in enumerate(zip(bar_values, bar_fractions)):
        ax.text(
            val + max_val * 0.01, i,
            f"{val:,}  ({frac:.1%})",
            va="center", fontsize=8, color="#555555",
        )

    ax.set_yticks(range(len(bar_values)))
    ax.set_yticklabels(yticklabels, fontsize=9)

    # Level1 组标签
    for l1, yc in group_positions.items():
        ax.text(
            -max_val * 0.02, yc, l1,
            va="center", ha="right", fontsize=10, fontweight="bold",
            color=l1_colors[l1],
        )

    ax.set_xlabel("Cell Count")
    ax.set_title("Chromium scRNA-seq: Cell Type Hierarchy (Level1 → Level2)")
    ax.set_xlim(0, max_val * 1.25)
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "celltype_nested_bar"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — Sunburst 环形层级图
# =========================================================================

def plot_celltype_sunburst(
    summary: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """绘制双层 Sunburst（环形扇区图）展示 Level1 → Level2 层级

    外圈为 Level2，内圈为 Level1。

    Parameters
    ----------
    summary : pd.DataFrame
        celltype_summary 的输出
    save_path : str, optional
        保存路径
    """
    # 汇总
    l1_counts = summary.groupby("Level1")["count"].sum().sort_values(ascending=False)
    l1_order = l1_counts.index.tolist()
    l1_colors = get_color_mapping(l1_order, PALETTE_CELLTYPE)

    # 内圈数据
    inner_sizes = l1_counts.values
    inner_labels = l1_order
    inner_colors = [l1_colors[lab] for lab in inner_labels]

    # 外圈数据（按 Level1 分组排列）
    outer_sizes = []
    outer_labels = []
    outer_colors = []
    for l1 in l1_order:
        sub = (
            summary[summary["Level1"] == l1]
            .sort_values("count", ascending=False)
        )
        base_rgb = matplotlib.colors.to_rgb(l1_colors[l1])
        n_sub = len(sub)
        for j, (_, row) in enumerate(sub.iterrows()):
            outer_sizes.append(row["count"])
            outer_labels.append(row["Level2"])
            # 渐变色：从基色到浅化
            factor = 0.3 + 0.7 * (1 - j / max(n_sub, 1))
            c = tuple(min(1, ch * factor + (1 - factor) * 0.95) for ch in base_rgb)
            outer_colors.append(c)

    fig, ax = create_figure(figsize=(10, 10))

    # 内圈
    wedges_inner, _ = ax.pie(
        inner_sizes,
        radius=0.7,
        colors=inner_colors,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5),
        startangle=90,
    )

    # 外圈
    wedges_outer, _ = ax.pie(
        outer_sizes,
        radius=1.05,
        colors=outer_colors,
        wedgeprops=dict(width=0.3, edgecolor="white", linewidth=0.8),
        startangle=90,
    )

    # 内圈标签
    for i, (wedge, label) in enumerate(zip(wedges_inner, inner_labels)):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = 0.52 * np.cos(np.deg2rad(ang))
        y = 0.52 * np.sin(np.deg2rad(ang))
        ha = "center"
        ax.text(x, y, label, ha=ha, va="center", fontsize=8, fontweight="bold")

    # 标题在中心
    total = int(sum(inner_sizes))
    ax.text(0, 0, f"Total\n{total:,}", ha="center", va="center", fontsize=13, fontweight="bold")

    ax.set_title("Cell Type Hierarchy Sunburst (Level1 → Level2)", pad=20)

    # 图例（外圈 top-10）
    top_outer = sorted(
        zip(outer_sizes, outer_labels, outer_colors),
        key=lambda x: x[0], reverse=True,
    )[:10]
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c, edgecolor="white", label=f"{lab} ({cnt:,})")
        for cnt, lab, c in top_outer
    ]
    ax.legend(
        handles=legend_handles, title="Top-10 Level2",
        loc="center left", bbox_to_anchor=(1.05, 0.5),
        fontsize=8, title_fontsize=9,
    )

    if save_path:
        save_figure(fig, save_path, config={"chart": "celltype_sunburst"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 患者 QC 堆叠条形图
# =========================================================================

def plot_patient_qc_bar(
    patient_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """绘制按患者分组的 QC 保留/丢弃堆叠条形图

    Parameters
    ----------
    patient_df : pd.DataFrame
        patient_summary 的输出, 列: Patient, total, kept, removed
    """
    df = patient_df.sort_values("total", ascending=True)

    fig, ax = create_figure(figsize=(9, max(5, len(df) * 0.4)))
    y = np.arange(len(df))
    ax.barh(y, df["kept"], color="#228833", edgecolor="white", label="Kept", height=0.7)
    ax.barh(y, df["removed"], left=df["kept"], color="#EE6677", edgecolor="white", label="Removed", height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Patient"])
    ax.set_xlabel("Cell Count")
    ax.set_title("QC Filter Results by Patient")
    ax.legend(loc="lower right")

    # 百分比标注
    for i, (_, row) in enumerate(df.iterrows()):
        pct = row["kept"] / row["total"] * 100 if row["total"] > 0 else 0
        ax.text(
            row["total"] + max(df["total"]) * 0.01, i,
            f"{pct:.0f}%", va="center", fontsize=8, color="#555555",
        )

    add_watermark(ax)
    if save_path:
        save_figure(fig, save_path, config={"chart": "patient_qc_bar"})
    else:
        plt.close(fig)
    return fig
