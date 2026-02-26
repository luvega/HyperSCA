#!/usr/bin/env python
"""将 10x .h5 表达矩阵转换为标准化 AnnData (.h5ad)

用法:
    python scripts/prepare_h5ad.py [--modality chromium|visium|xenium|all]

功能:
- Chromium: 读取 filtered_feature_bc_matrix.h5, 合并 cell_metadata.csv 注释
- Visium:   读取 filtered_feature_bc_matrix.h5, 附加空间坐标
- Xenium:   读取 cell_feature_matrix.h5, 附加空间坐标 (cells.parquet)
- 输出标准化 .h5ad 到各数据目录

注意: VisiumHD 暂不纳入阶段1（数据量巨大，需单独策略处理）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CHROMIUM_DIR = DATA_DIR / "Chromium_HumanColon_Oliveira"
VISIUM_DIR = DATA_DIR / "Visium_HumanColon_Oliveira"
XENIUM_DIR = DATA_DIR / "Xenium_HumanColon_Oliveira"


# =========================================================================
# Chromium scRNA-seq
# =========================================================================

def prepare_chromium() -> Path:
    """Chromium: .h5 + cell_metadata.csv → expression.h5ad"""
    print("[Chromium] 开始准备...")

    h5_path = CHROMIUM_DIR / "filtered_feature_bc_matrix.h5"
    meta_path = CHROMIUM_DIR / "cell_metadata.csv"
    out_path = CHROMIUM_DIR / "expression.h5ad"

    if not h5_path.exists():
        raise FileNotFoundError(f"缺少表达矩阵: {h5_path}")

    # 读取 10x .h5
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    print(f"  expression matrix: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # 合并元数据
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        print(f"  元数据: {len(meta)} rows, columns: {list(meta.columns)}")

        # 以 Barcode 为索引对齐
        meta = meta.set_index("Barcode")
        common = adata.obs_names.intersection(meta.index)
        print(f"  匹配细胞数: {len(common)} / {adata.n_obs}")

        if len(common) > 0:
            adata = adata[common].copy()
            for col in meta.columns:
                adata.obs[col] = meta.loc[adata.obs_names, col].values
    else:
        print("  [警告] cell_metadata.csv 不存在，跳过注释合并")

    adata.write(str(out_path))
    print(f"  [OK] saved: {out_path} ({adata.shape[0]} x {adata.shape[1]})")
    return out_path


# =========================================================================
# Visium ST
# =========================================================================

def prepare_visium() -> Path:
    """Visium: .h5 + spatial → expression.h5ad"""
    print("[Visium] 开始准备...")

    h5_path = VISIUM_DIR / "outs" / "filtered_feature_bc_matrix.h5"
    pos_path = VISIUM_DIR / "outs" / "spatial" / "tissue_positions.csv"
    sf_path = VISIUM_DIR / "outs" / "spatial" / "scalefactors_json.json"
    out_path = VISIUM_DIR / "expression.h5ad"

    if not h5_path.exists():
        raise FileNotFoundError(f"缺少表达矩阵: {h5_path}")

    # 手动读取 .h5（sc.read_visium 在缺少 hires_image 时报错）
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    print(f"  read_10x_h5 OK")

    print(f"  expression matrix: {adata.shape[0]} spots x {adata.shape[1]} genes")

    # 确保空间坐标存在
    if "spatial" not in adata.obsm and pos_path.exists():
        pos = pd.read_csv(pos_path)
        # tissue_positions.csv 格式: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
        pos = pos.set_index(pos.columns[0])  # barcode 列
        common = adata.obs_names.intersection(pos.index)
        if len(common) > 0:
            adata = adata[common].copy()
            spatial_cols = [c for c in pos.columns if "pxl" in c.lower()]
            if len(spatial_cols) >= 2:
                adata.obsm["spatial"] = pos.loc[adata.obs_names, spatial_cols].values.astype(np.float64)
                print(f"  空间坐标已附加: {spatial_cols}")

    if "spatial" in adata.obsm:
        print(f"  空间坐标维度: {adata.obsm['spatial'].shape}")
    else:
        print("  [警告] 未找到空间坐标")

    # 标记 in_tissue
    if pos_path.exists():
        pos = pd.read_csv(pos_path)
        pos = pos.set_index(pos.columns[0])
        if "in_tissue" in pos.columns:
            common = adata.obs_names.intersection(pos.index)
            adata.obs["in_tissue"] = 0
            adata.obs.loc[common, "in_tissue"] = pos.loc[common, "in_tissue"].values

    adata.write(str(out_path))
    print(f"  [OK] saved: {out_path} ({adata.shape[0]} x {adata.shape[1]})")
    return out_path


# =========================================================================
# Xenium
# =========================================================================

def prepare_xenium() -> Path:
    """Xenium: cell_feature_matrix.h5 + cells.parquet → expression.h5ad"""
    print("[Xenium] 开始准备...")

    h5_path = XENIUM_DIR / "cell_feature_matrix.h5"
    cells_path = XENIUM_DIR / "cells.parquet"
    out_path = XENIUM_DIR / "expression.h5ad"

    if not h5_path.exists():
        raise FileNotFoundError(f"缺少表达矩阵: {h5_path}")

    # 读取 10x .h5
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    print(f"  expression matrix: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # 附加空间坐标 (cells.parquet)
    if cells_path.exists():
        cells = pd.read_parquet(cells_path)
        print(f"  cells.parquet: {len(cells)} rows, columns: {list(cells.columns)}")

        # Xenium cells.parquet 通常含: cell_id, x_centroid, y_centroid, ...
        coord_cols = []
        for possible_x in ["x_centroid", "x_location"]:
            if possible_x in cells.columns:
                coord_cols.append(possible_x)
                break
        for possible_y in ["y_centroid", "y_location"]:
            if possible_y in cells.columns:
                coord_cols.append(possible_y)
                break

        if len(coord_cols) == 2 and len(cells) == adata.n_obs:
            adata.obsm["spatial"] = cells[coord_cols].values.astype(np.float64)
            print(f"  空间坐标已附加: {coord_cols}")

            # 附加其他元数据
            for col in cells.columns:
                if col not in coord_cols and col != "cell_id":
                    try:
                        adata.obs[col] = cells[col].values
                    except Exception:
                        pass
        elif len(cells) != adata.n_obs:
            print(f"  [警告] cells.parquet ({len(cells)}) 与矩阵 ({adata.n_obs}) 行数不匹配")
    else:
        print("  [警告] cells.parquet 不存在")

    adata.write(str(out_path))
    print(f"  [OK] saved: {out_path} ({adata.shape[0]} x {adata.shape[1]})")
    return out_path


# =========================================================================
# 主入口
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="将 10x .h5 转换为 .h5ad")
    parser.add_argument(
        "--modality",
        choices=["chromium", "visium", "xenium", "all"],
        default="all",
        help="要处理的数据模态（默认: all）",
    )
    args = parser.parse_args()

    results = {}
    modalities = {
        "chromium": prepare_chromium,
        "visium": prepare_visium,
        "xenium": prepare_xenium,
    }

    targets = modalities if args.modality == "all" else {args.modality: modalities[args.modality]}

    for name, func in targets.items():
        try:
            path = func()
            results[name] = ("SUCCESS", str(path))
        except Exception as e:
            results[name] = ("FAILED", str(e))
            print(f"  [ERROR] {e}")
        print()

    # 汇总
    print("=" * 50)
    print("prepare_h5ad 汇总")
    print("=" * 50)
    for name, (status, detail) in results.items():
        print(f"  {name:12s}: {status} — {detail}")

    n_ok = sum(1 for s, _ in results.values() if s == "SUCCESS")
    print(f"\n成功: {n_ok} / {len(results)}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
