"""数据预处理模块

封装 scanpy 标准预处理流程，输出可直接用于 H-VAE 训练的 AnnData。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scanpy as sc
import anndata as ad


def filter_genes_cells(
    adata: ad.AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    max_genes: int = 8000,
    max_pct_mt: float = 20.0,
) -> ad.AnnData:
    """过滤低质量基因与细胞

    Parameters
    ----------
    adata : AnnData
        原始计数矩阵
    min_cells : int
        基因至少在多少个细胞中表达
    min_genes : int
        细胞至少表达多少基因
    max_genes : int
        细胞最多表达多少基因（过滤 doublet）
    max_pct_mt : float
        线粒体基因百分比上限

    Returns
    -------
    AnnData
        过滤后的 AnnData
    """
    adata = adata.copy()

    # 基础 QC 指标
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # 过滤
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # 过滤 doublet（基因数上限）
    if max_genes < np.inf:
        adata = adata[adata.obs["n_genes_by_counts"] < max_genes].copy()

    if max_pct_mt < 100:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)

    n_after = adata.n_obs
    g_after = adata.n_vars
    print(f"  [filter] {n_before} -> {n_after} cells, {g_after} genes")

    return adata


def normalize_and_log(
    adata: ad.AnnData,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """总量归一化 + log1p 变换

    保留 adata.raw 为原始计数。

    Parameters
    ----------
    adata : AnnData
    target_sum : float
        归一化目标总和

    Returns
    -------
    AnnData
        .X 为 log1p 归一化矩阵，.raw 为原始计数
    """
    adata = adata.copy()
    adata.raw = adata  # 保留原始计数

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    return adata


def select_hvg(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
    flavor: str = "seurat",
    subset: bool = True,
    force_include: list[str] | None = None,
) -> ad.AnnData:
    """选择高变基因

    Parameters
    ----------
    adata : AnnData
        已归一化的 AnnData
    n_top_genes : int
        选择的 HVG 数量
    flavor : str
        'seurat' / 'cell_ranger' / 'seurat_v3'
    subset : bool
        是否只保留 HVG（True 则 adata.X 仅含 HVG）
    force_include : list of str or None
        强制包含在 HVG 中的基因名列表

    Returns
    -------
    AnnData
        .var['highly_variable'] 标记 + 可选子集
    """
    adata = adata.copy()
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor=flavor
    )

    if force_include:
        added = 0
        for g in force_include:
            if g in adata.var_names and not adata.var.loc[g, "highly_variable"]:
                adata.var.loc[g, "highly_variable"] = True
                added += 1
        if added:
            print(f"  [HVG] force-included {added} target genes")

    n_hvg = adata.var["highly_variable"].sum()
    print(f"  [HVG] selected {n_hvg} / {adata.n_vars} genes")

    if subset:
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


def preprocess(
    adata: ad.AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    max_genes: int = 8000,
    max_pct_mt: float = 20.0,
    target_sum: float = 1e4,
    n_top_genes: int = 3000,
    hvg_flavor: str = "seurat",
    scale: bool = False,
    max_value: Optional[float] = 10.0,
    **kwargs,
) -> ad.AnnData:
    """一站式预处理管线

    Steps:
    1. 过滤基因/细胞
    2. 归一化 + log1p（保存 .raw）
    3. HVG 选择
    4. 可选 z-score 缩放

    Parameters
    ----------
    adata : AnnData
        原始计数矩阵
    scale : bool
        是否做 z-score 缩放（H-VAE 通常不需要）
    max_value : float or None
        缩放后的截断值

    Returns
    -------
    AnnData
        预处理后的 AnnData:
        - .X: 归一化+log1p（或缩放后）表达，仅 HVG
        - .raw: 原始计数（全部基因）
        - .var['highly_variable']: 标记
    """
    print("[preprocess] starting...")

    # Step 1: 过滤
    adata = filter_genes_cells(
        adata,
        min_cells=min_cells,
        min_genes=min_genes,
        max_genes=max_genes,
        max_pct_mt=max_pct_mt,
    )

    # Step 2: 归一化
    adata = normalize_and_log(adata, target_sum=target_sum)

    # Step 3: HVG
    force_genes = kwargs.get("force_include_genes", None)
    adata = select_hvg(adata, n_top_genes=n_top_genes, flavor=hvg_flavor,
                       force_include=force_genes)

    # Step 4: 可选缩放
    if scale:
        sc.pp.scale(adata, max_value=max_value)
        print(f"  [scale] z-score with max_value={max_value}")

    print(f"[preprocess] done: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata
