"""Example 03: VisiumHD 细胞/核分割统计

提供分割面积计算与多种可视化模板：
- 基础面积分布直方图
- 增强 3-panel 分割质量图（Cell Area / Nucleus Area / NC Ratio）
- 联合散点图（Cell Area vs Nucleus Area，颜色映射 NC Ratio）
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plot_style import (
    CMAP_SEQUENTIAL,
    CMAP_DIVERGING,
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
# 数据处理
# =========================================================================

def polygon_area(coords: list[list[float]]) -> float:
    """Shoelace 公式计算多边形面积（单位: 像素²）"""
    pts = np.array(coords)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_areas(features: list[dict]) -> pd.DataFrame:
    """计算每个分割多边形的面积

    Returns
    -------
    pd.DataFrame
        列: cell_id, area
    """
    rows = []
    for feat in features:
        cid = feat.get("properties", {}).get("cell_id", None)
        geom = feat.get("geometry", {})
        rings = geom.get("coordinates", [[]])
        area = polygon_area(rings[0]) if rings and rings[0] else 0.0
        rows.append({"cell_id": cid, "area": area})
    return pd.DataFrame(rows)


def compute_nc_ratio(
    cell_areas: pd.DataFrame, nucleus_areas: pd.DataFrame
) -> pd.DataFrame:
    """计算核质比（nucleus / cell）并返回合并表

    Returns
    -------
    pd.DataFrame
        列: cell_id, area_cell, area_nucleus, nc_ratio
    """
    merged = cell_areas.merge(
        nucleus_areas, on="cell_id", suffixes=("_cell", "_nucleus"), how="inner"
    )
    merged["nc_ratio"] = merged["area_nucleus"] / merged["area_cell"].replace(0, np.nan)
    return merged


def segmentation_summary(
    cell_areas: pd.DataFrame, nucleus_areas: pd.DataFrame
) -> dict:
    """汇总分割统计"""
    merged = compute_nc_ratio(cell_areas, nucleus_areas)
    return {
        "n_cells": len(cell_areas),
        "n_nuclei": len(nucleus_areas),
        "n_matched": len(merged),
        "cell_area_mean": float(cell_areas["area"].mean()),
        "cell_area_median": float(cell_areas["area"].median()),
        "nucleus_area_mean": float(nucleus_areas["area"].mean()),
        "nucleus_area_median": float(nucleus_areas["area"].median()),
        "nc_ratio_mean": float(merged["nc_ratio"].mean()),
        "nc_ratio_median": float(merged["nc_ratio"].median()),
    }


# =========================================================================
# 可视化 — 基础面积直方图（向后兼容）
# =========================================================================

def plot_area_hist(
    cell_areas: pd.DataFrame,
    nucleus_areas: pd.DataFrame,
    save_path: str | None = None,
):
    """绘制细胞与核面积分布直方图（基础版，向后兼容）"""
    fig, axes = create_figure(1, 2, figsize=(12, 4))

    axes[0].hist(cell_areas["area"], bins=80, color="#4477AA", edgecolor="white", alpha=0.85)
    axes[0].set_title("Cell Area Distribution")
    axes[0].set_xlabel("Area (px²)")
    axes[0].set_ylabel("Count")

    axes[1].hist(nucleus_areas["area"], bins=80, color="#EE6677", edgecolor="white", alpha=0.85)
    axes[1].set_title("Nucleus Area Distribution")
    axes[1].set_xlabel("Area (px²)")
    axes[1].set_ylabel("Count")

    fig.suptitle("VisiumHD Segmentation Statistics", fontsize=14, fontweight="bold")
    if save_path:
        save_figure(fig, save_path, config={"chart": "area_hist_basic"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — 增强版 3-Panel 分割质量图
# =========================================================================

def plot_segmentation_quality(
    cell_areas: pd.DataFrame,
    nucleus_areas: pd.DataFrame,
    save_path: str | None = None,
    nc_thresholds: tuple[float, float] = (0.2, 0.8),
) -> plt.Figure:
    """3-panel 分割质量综合图

    Panel 1: Cell Area 分布 + 中位数线
    Panel 2: Nucleus Area 分布 + 中位数线
    Panel 3: NC Ratio 分布 + 正常范围区间

    Parameters
    ----------
    nc_thresholds : (low, high)
        核质比的正常参考区间，用于绿色底色标注
    """
    merged = compute_nc_ratio(cell_areas, nucleus_areas)

    fig, axes = create_figure(1, 3, figsize=(16, 5))

    # --- Panel 1: Cell Area ---
    ax = axes[0]
    ax.hist(cell_areas["area"], bins=80, color="#4477AA", edgecolor="white", alpha=0.85)
    med_cell = cell_areas["area"].median()
    ax.axvline(med_cell, color="#228833", linestyle="--", linewidth=1.5,
               label=f"Median = {med_cell:.0f}")
    ax.set_xlabel("Area (px²)")
    ax.set_ylabel("Count")
    ax.set_title("Cell Area")
    ax.legend(fontsize=9)
    add_watermark(ax)

    # --- Panel 2: Nucleus Area ---
    ax = axes[1]
    ax.hist(nucleus_areas["area"], bins=80, color="#EE6677", edgecolor="white", alpha=0.85)
    med_nuc = nucleus_areas["area"].median()
    ax.axvline(med_nuc, color="#228833", linestyle="--", linewidth=1.5,
               label=f"Median = {med_nuc:.0f}")
    ax.set_xlabel("Area (px²)")
    ax.set_ylabel("Count")
    ax.set_title("Nucleus Area")
    ax.legend(fontsize=9)

    # --- Panel 3: NC Ratio ---
    ax = axes[2]
    nc = merged["nc_ratio"].dropna()
    ax.hist(nc, bins=80, color="#AA3377", edgecolor="white", alpha=0.85)
    med_nc = nc.median()
    ax.axvline(med_nc, color="#228833", linestyle="--", linewidth=1.5,
               label=f"Median = {med_nc:.3f}")
    # 正常参考区间
    ax.axvspan(nc_thresholds[0], nc_thresholds[1], alpha=0.1, color=COLOR_GOOD,
               label=f"Ref range [{nc_thresholds[0]}, {nc_thresholds[1]}]")
    ax.set_xlabel("Nucleus / Cell Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Nuclear-Cytoplasmic Ratio")
    ax.legend(fontsize=9)

    fig.suptitle("VisiumHD Segmentation Quality Dashboard", fontsize=14, fontweight="bold")

    if save_path:
        save_figure(fig, save_path, config={"chart": "segmentation_quality_3panel"})
    else:
        plt.close(fig)
    return fig


# =========================================================================
# 可视化 — Cell vs Nucleus 联合散点图
# =========================================================================

def plot_area_scatter(
    cell_areas: pd.DataFrame,
    nucleus_areas: pd.DataFrame,
    save_path: str | None = None,
    max_points: int = 10000,
) -> plt.Figure:
    """Cell Area vs Nucleus Area 散点图，颜色映射 NC Ratio

    Parameters
    ----------
    max_points : int
        最多绘制的点数（随机采样）
    """
    merged = compute_nc_ratio(cell_areas, nucleus_areas)
    if len(merged) > max_points:
        merged = merged.sample(max_points, random_state=42)

    fig, ax = create_figure(figsize=(8, 7))

    sc = ax.scatter(
        merged["area_cell"], merged["area_nucleus"],
        c=merged["nc_ratio"], cmap=CMAP_SEQUENTIAL,
        s=4, alpha=0.6, edgecolors="none",
        vmin=0, vmax=1,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("NC Ratio", fontsize=10)

    # 对角参考线
    lim = max(merged["area_cell"].max(), merged["area_nucleus"].max())
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.3, label="1:1 line")

    ax.set_xlabel("Cell Area (px²)")
    ax.set_ylabel("Nucleus Area (px²)")
    ax.set_title("Cell vs Nucleus Area (color = NC Ratio)")
    ax.legend(fontsize=9)
    add_watermark(ax)

    if save_path:
        save_figure(fig, save_path, config={"chart": "area_scatter_nc"})
    else:
        plt.close(fig)
    return fig
