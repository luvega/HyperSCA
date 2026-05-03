"""HyperSCA 统一绘图风格与复用工具

提供全局色板、rcParams、辅助函数，确保全项目图表视觉一致。

使用方式
--------
>>> from src.utils.plot_style import apply_style, PALETTE, save_figure
>>> apply_style()  # 一次性设置全局 rcParams
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# =============================================================================
# 色板定义（按项目作图规范统一）
# =============================================================================

# 离散高对比色板：用于单细胞降维散点、堆叠柱状图等分类任务
PALETTE_DISCRETE_HIGH_CONTRAST: list[str] = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#FFFF33",  # yellow
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
]
PALETTE_CATEGORICAL: list[str] = PALETTE_DISCRETE_HIGH_CONTRAST

# Level1 细胞群专用色板（可手动映射）
PALETTE_CELLTYPE: dict[str, str] = {
    "T cell": "#377EB8",
    "B cell": "#984EA3",
    "Myeloid": "#A65628",
    "Stromal": "#4DAF4A",
    "Epithelial": "#E41A1C",
    "NK": "#F781BF",
    "Mast": "#999999",
    "pDC": "#377EB8",
    "ILC": "#984EA3",
    "Neutrophil": "#A65628",
    "Neutrophils": "#A65628",
}

# 连续色板（语义化）
# - 表达热图：白 -> 深红，确保 0 表达接近纯白
# - 发散色板：用于 logFC / 预后正负向等对称统计量
# - 拟时间：用于发育/轨迹梯度
CMAP_EXPRESSION = mcolors.LinearSegmentedColormap.from_list(
    "hypersca_expression_white_red",
    ["#FFFFFF", "#FEE5D9", "#FCAE91", "#FB6A4A", "#CB181D", "#67000D"],
)
CMAP_SEQUENTIAL = CMAP_EXPRESSION
CMAP_DIVERGING = "coolwarm"
CMAP_SPATIAL = CMAP_EXPRESSION
CMAP_SPATIAL_DIVERGING = "coolwarm"
CMAP_PSEUDOTIME = "Spectral_r"
CMAP_PSEUDOTIME_ALT = "turbo"

# 语义颜色
COLOR_GOOD = "#228833"
COLOR_WARN = "#CCBB44"
COLOR_BAD = "#EE6677"
COLOR_NEUTRAL = "#BBBBBB"

# 简写
PALETTE = PALETTE_CATEGORICAL


# =============================================================================
# 全局 rcParams 设置
# =============================================================================

_HYPERSCA_RC: dict[str, Any] = {
    # --- 字体 ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # --- 线 / 标记 ---
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    # --- 图框 ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    # --- 颜色循环 ---
    "axes.prop_cycle": plt.cycler("color", PALETTE_CATEGORICAL),
    # --- 保存 ---
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # --- 其他 ---
    "figure.figsize": (8, 6),
    "figure.constrained_layout.use": True,
}

_CNS_FIGURE_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    """应用 HyperSCA 全局 matplotlib 样式"""
    plt.rcParams.update(_HYPERSCA_RC)
    try:
        import seaborn as sns  # optional
        sns.set_theme(style="white", rc={"axes.grid": False})
    except Exception:
        pass


def apply_cns_style() -> None:
    """应用 CNS/Cell 风格绘图参数（细线、无网格、严格字号层级）"""
    plt.rcParams.update(_HYPERSCA_RC)
    plt.rcParams.update(_CNS_FIGURE_RC)
    try:
        import seaborn as sns  # optional
        sns.set_theme(style="white", rc={"axes.grid": False, "axes.facecolor": "white"})
    except Exception:
        pass


def reset_style() -> None:
    """恢复 matplotlib 默认样式"""
    plt.rcdefaults()


# =============================================================================
# 辅助函数
# =============================================================================

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[plt.Figure, Any]:
    """创建标准化 Figure + Axes

    Parameters
    ----------
    nrows, ncols : int
        子图行列数
    figsize : tuple, optional
        若不指定则自动计算
    **kwargs
        传递给 plt.subplots

    Returns
    -------
    fig, axes
    """
    if figsize is None:
        w = 6 * ncols + 1
        h = 5 * nrows + 0.5
        figsize = (min(w, 20), min(h, 16))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def get_dynamic_point_size(
    n_obs: int,
    *,
    min_size: float = 2.0,
    max_size: float = 14.0,
    ref_n: int = 50000,
) -> float:
    """根据样本量自适应散点大小（大样本更小点径）"""
    n = max(int(n_obs), 1)
    ratio = min(max(float(ref_n) / float(n), 0.05), 1.0)
    return float(min_size + (max_size - min_size) * ratio)


def get_dynamic_alpha(
    n_obs: int,
    *,
    min_alpha: float = 0.6,
    max_alpha: float = 0.8,
    ref_n: int = 50000,
) -> float:
    """根据样本量自适应透明度（大样本更低 alpha）"""
    n = max(int(n_obs), 1)
    ratio = min(max(float(ref_n) / float(n), 0.05), 1.0)
    return float(min_alpha + (max_alpha - min_alpha) * ratio)


def normalize_rowwise(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """行归一化，常用于组分堆叠柱图前处理"""
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        s = max(float(np.sum(x)), eps)
        return x / s
    rs = np.sum(x, axis=1, keepdims=True)
    rs = np.maximum(rs, eps)
    return x / rs


def place_legend_outside(
    ax: plt.Axes,
    *,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (1.02, 1.0),
    **kwargs,
):
    """统一外置图例位置，减少内部遮挡"""
    return ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=False, **kwargs)


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 200,
    config: dict | None = None,
    data_version: str = "",
    model_version: str = "",
    seed: int | None = None,
    close: bool = True,
) -> Path:
    """保存图片并附带元信息 JSON

    Parameters
    ----------
    fig : matplotlib Figure
    path : 输出路径（.png / .pdf / .svg）
    dpi : 分辨率
    config : 可选配置字典
    data_version, model_version : 版本标签
    seed : 随机种子
    close : 是否保存后关闭 figure

    Returns
    -------
    Path : 保存的图片路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi)

    # 元信息 sidecar
    meta = {
        "figure": path.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dpi": dpi,
        "data_version": data_version,
        "model_version": model_version,
        "seed": seed,
    }
    if config:
        meta["config"] = config
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if close:
        plt.close(fig)
    return path


def get_color_mapping(
    labels: list[str] | np.ndarray,
    palette: list[str] | dict[str, str] | None = None,
) -> dict[str, str]:
    """为标签列表生成颜色映射

    Parameters
    ----------
    labels : 标签序列
    palette : 色板（list 或 dict）；None 使用默认

    Returns
    -------
    dict[str, str] : label -> hex color
    """
    unique = sorted(set(labels))

    if isinstance(palette, dict):
        # 用给定字典补齐缺失标签
        remaining = [c for c in PALETTE_CATEGORICAL if c not in palette.values()]
        mapping = {}
        idx = 0
        for lab in unique:
            if lab in palette:
                mapping[lab] = palette[lab]
            else:
                mapping[lab] = remaining[idx % len(remaining)]
                idx += 1
        return mapping
    pal = palette or PALETTE_CATEGORICAL
    return {lab: pal[i % len(pal)] for i, lab in enumerate(unique)}


def add_watermark(ax: plt.Axes, text: str = "HyperSCA", alpha: float = 0.08) -> None:
    """在 Axes 右下角添加水印"""
    ax.text(
        0.98, 0.02, text,
        transform=ax.transAxes,
        fontsize=14, color="gray", alpha=alpha,
        ha="right", va="bottom", style="italic",
    )


def truncate_colormap(cmap_name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """截取 colormap 的子区间"""
    cmap = plt.get_cmap(cmap_name)
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
