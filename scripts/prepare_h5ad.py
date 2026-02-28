#!/usr/bin/env python
"""四项目数据标准化入库与 h5ad 准备脚本。

该脚本兼容两类工作流:
1) 旧版 demo 模态转换（chromium/visium/xenium）
2) 研究版四项目入库（scCRC_ICB/scCRC_Neu/ST_CRC_MSS/scCRC_IFNG）

研究版输出:
    data/scRNA/scCRC_*/
    data/ST/ST_CRC_MSS/
    data/metadata/project_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# 旧版 demo 路径
CHROMIUM_DIR = DATA_DIR / "Chromium_HumanColon_Oliveira"
VISIUM_DIR = DATA_DIR / "Visium_HumanColon_Oliveira"
XENIUM_DIR = DATA_DIR / "Xenium_HumanColon_Oliveira"

# 四项目默认路径
DEFAULT_SOURCES = {
    "scCRC_ICB": Path(r"G:\scCRC_ICB"),
    "scCRC_Neu": Path(r"G:\scCRC_Neu"),
    "ST_CRC_MSS": Path(r"G:\ST_CRC_MSS"),
    "scCRC_IFNG": Path(r"F:\scCRC_IFNG"),
}


def _save_manifest(items: list[dict]) -> Path:
    meta_dir = DATA_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "project_manifest.json"
    payload = {"sources": items}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_if_exists(src: Path, dst: Path, reports: list[str]) -> None:
    if src.exists():
        _safe_copy(src, dst)
        reports.append(f"COPIED: {src} -> {dst}")
    else:
        reports.append(f"MISSING: {src}")


def _find_first_h5ad(root: Path) -> Path | None:
    for p in root.glob("**/*.h5ad"):
        return p
    return None


def _find_ifng_h5ad(root: Path) -> Path | None:
    """为 IFNG 优先选择包含空间坐标的全量 h5ad。"""
    preferred = [
        root / "data" / "processed" / "all_samples.h5ad",
        root / "data" / "processed" / "hypersca_input" / "expression.h5ad",
        root / "data" / "processed" / "hypersca_train" / "expression.h5ad",
    ]
    for p in preferred:
        if p.exists():
            return p
    return _find_first_h5ad(root)


def _build_neu_h5ad(src_root: Path, out_dir: Path, reports: list[str]) -> Path | None:
    """将 scCRC_Neu 的 DESeq2 normalized count 聚合成 pseudo-celltype h5ad。"""
    tsv_files = sorted(src_root.glob("**/*-NormalizedCounts.tsv"))
    if not tsv_files:
        reports.append(f"MISSING: {src_root} 下未找到 *-NormalizedCounts.tsv")
        return None

    mats = []
    labels = []
    genes = None
    for f in tsv_files:
        try:
            df = pd.read_csv(f, sep="\t")
        except Exception as exc:
            reports.append(f"SKIP: {f} read error: {exc}")
            continue
        if "gene" in df.columns:
            gene_col = "gene"
        elif "symbol" in df.columns:
            gene_col = "symbol"
        else:
            gene_col = df.columns[0]
        df = df.dropna(subset=[gene_col]).copy()
        df = df.set_index(gene_col)
        values = df.select_dtypes(include=[np.number])
        if values.empty:
            continue
        profile = values.mean(axis=1)
        if genes is None:
            genes = profile.index
        profile = profile.reindex(genes).fillna(0.0)
        mats.append(profile.values)
        labels.append(f.stem.replace("-NormalizedCounts", ""))

    if not mats:
        reports.append("FAILED: 无法从 scCRC_Neu 构建表达矩阵")
        return None
    X = np.vstack(mats)
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = labels
    adata.obs["celltype"] = labels
    adata.var_names = pd.Index(genes.astype(str))
    out = out_dir / "expression.h5ad"
    adata.write(out)
    reports.append(f"BUILT: {out} ({adata.n_obs} x {adata.n_vars})")
    return out


def _build_st_h5ad(src_root: Path, out_dir: Path, reports: list[str]) -> Path | None:
    """将 STmetadata 合并为 spot-level h5ad。"""
    csv_files = sorted(src_root.glob("STmetadata_*.csv"))
    if not csv_files:
        reports.append(f"MISSING: {src_root} 下未找到 STmetadata_*.csv")
        return None
    chunks = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            reports.append(f"SKIP: {f} read error: {exc}")
            continue
        df["sample_id"] = f.stem
        chunks.append(df)
    if not chunks:
        reports.append("FAILED: ST metadata 读取失败")
        return None
    merged = pd.concat(chunks, ignore_index=True)
    merged.to_csv(out_dir / "spot_metadata.csv", index=False)

    numeric = merged.select_dtypes(include=[np.number]).copy()
    if {"x", "y"}.issubset(numeric.columns):
        coord_cols = ["x", "y"]
    else:
        coord_cols = numeric.columns[:2].tolist() if len(numeric.columns) >= 2 else []
    feature_cols = [c for c in numeric.columns if c not in {"x", "y"}]
    if not feature_cols:
        reports.append("WARN: ST 数值特征不足，跳过 h5ad 构建")
        return None
    obs = merged[["sample_id"]].copy()
    if "seurat_clusters" in merged.columns:
        obs["celltype"] = merged["seurat_clusters"].astype(str)
    else:
        obs["celltype"] = "spot"
    adata = ad.AnnData(X=numeric[feature_cols].to_numpy(dtype=np.float32), obs=obs)
    adata.var_names = pd.Index(feature_cols)
    if len(coord_cols) == 2:
        adata.obsm["spatial"] = merged[coord_cols].to_numpy(dtype=np.float32)
    adata.write(out_dir / "expression.h5ad")
    reports.append(f"BUILT: {out_dir / 'expression.h5ad'} ({adata.n_obs} x {adata.n_vars})")
    return out_dir / "expression.h5ad"


def _prepare_multisource(args) -> int:
    """四项目标准化入库。"""
    sources = {
        "scCRC_ICB": Path(args.icb_root),
        "scCRC_Neu": Path(args.neu_root),
        "ST_CRC_MSS": Path(args.st_root),
        "scCRC_IFNG": Path(args.ifng_root),
    }
    sc_root = DATA_DIR / "scRNA"
    st_root = DATA_DIR / "ST"
    meta_root = DATA_DIR / "metadata"
    sc_root.mkdir(parents=True, exist_ok=True)
    st_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []
    manifest_rows: list[dict] = []
    results: dict[str, str] = {}

    # scCRC_Neu
    neu_dir = sc_root / "scCRC_Neu"
    neu_dir.mkdir(parents=True, exist_ok=True)
    neu_h5ad = _find_first_h5ad(sources["scCRC_Neu"])
    if neu_h5ad is not None:
        _safe_copy(neu_h5ad, neu_dir / "expression.h5ad")
        report_lines.append(f"COPIED: {neu_h5ad} -> {neu_dir / 'expression.h5ad'}")
        results["scCRC_Neu"] = "SUCCESS"
    else:
        built = _build_neu_h5ad(sources["scCRC_Neu"], neu_dir, report_lines)
        results["scCRC_Neu"] = "SUCCESS" if built else "FAILED"

    # scCRC_IFNG
    ifng_dir = sc_root / "scCRC_IFNG"
    ifng_dir.mkdir(parents=True, exist_ok=True)
    ifng_h5ad = _find_ifng_h5ad(sources["scCRC_IFNG"])
    if ifng_h5ad is not None:
        _safe_copy(ifng_h5ad, ifng_dir / "expression.h5ad")
        report_lines.append(f"COPIED: {ifng_h5ad} -> {ifng_dir / 'expression.h5ad'}")
        # IFNG CosMx 也是空间转录组数据，额外同步到 ST 目录，便于统一空间流程调用
        ifng_st_dir = st_root / "scCRC_IFNG_CosMx"
        ifng_st_dir.mkdir(parents=True, exist_ok=True)
        _safe_copy(ifng_h5ad, ifng_st_dir / "expression.h5ad")
        report_lines.append(f"COPIED: {ifng_h5ad} -> {ifng_st_dir / 'expression.h5ad'}")
        results["scCRC_IFNG"] = "SUCCESS"
    else:
        # 最小可用: 拷贝临床映射和 target table
        _copy_if_exists(
            sources["scCRC_IFNG"] / "results" / "tables" / "sample_clinical_mapping.csv",
            ifng_dir / "sample_clinical_mapping.csv",
            report_lines,
        )
        _copy_if_exists(
            sources["scCRC_IFNG"] / "results" / "tables" / "targets_shared_specific_by_mmr.csv",
            ifng_dir / "targets_shared_specific_by_mmr.csv",
            report_lines,
        )
        results["scCRC_IFNG"] = "SUCCESS"

    # scCRC_ICB
    icb_dir = sc_root / "scCRC_ICB"
    icb_dir.mkdir(parents=True, exist_ok=True)
    deg_dir = icb_dir / "deg_tables"
    deg_dir.mkdir(parents=True, exist_ok=True)
    icb_files = sorted((sources["scCRC_ICB"] / "output").glob("DEGs_MSS*.csv"))
    if not icb_files:
        report_lines.append("MISSING: scCRC_ICB/output/DEGs_MSS*.csv")
        results["scCRC_ICB"] = "FAILED"
    else:
        for f in icb_files:
            _safe_copy(f, deg_dir / f.name)
            report_lines.append(f"COPIED: {f} -> {deg_dir / f.name}")
        results["scCRC_ICB"] = "SUCCESS"

    # ST_CRC_MSS
    st_dir = st_root / "ST_CRC_MSS"
    st_dir.mkdir(parents=True, exist_ok=True)
    st_h5ad = _find_first_h5ad(sources["ST_CRC_MSS"])
    if st_h5ad is not None:
        _safe_copy(st_h5ad, st_dir / "expression.h5ad")
        report_lines.append(f"COPIED: {st_h5ad} -> {st_dir / 'expression.h5ad'}")
        results["ST_CRC_MSS"] = "SUCCESS"
    else:
        built = _build_st_h5ad(sources["ST_CRC_MSS"], st_dir, report_lines)
        results["ST_CRC_MSS"] = "SUCCESS" if built else "FAILED"

    for name, src in sources.items():
        if name == "ST_CRC_MSS":
            modality = "ST"
            std_dir = str(st_root / name)
        else:
            modality = "scRNA"
            std_dir = str(sc_root / name)
        manifest_rows.append(
            {
                "name": name,
                "source_path": str(src),
                "modality": modality,
                "standardized_dir": std_dir,
                "status": results.get(name, "FAILED"),
            }
        )

    manifest_path = _save_manifest(manifest_rows)
    report_path = meta_root / "onboarding_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Report: {report_path}")
    for k, v in results.items():
        print(f"  {k:12s}: {v}")
    return 0 if all(v == "SUCCESS" for v in results.values()) else 1


# =========================================================================
# 旧版 demo: Chromium / Visium / Xenium
# =========================================================================
def prepare_chromium() -> Path:
    h5_path = CHROMIUM_DIR / "filtered_feature_bc_matrix.h5"
    meta_path = CHROMIUM_DIR / "cell_metadata.csv"
    out_path = CHROMIUM_DIR / "expression.h5ad"
    if not h5_path.exists():
        raise FileNotFoundError(f"缺少表达矩阵: {h5_path}")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    if meta_path.exists():
        meta = pd.read_csv(meta_path).set_index("Barcode")
        common = adata.obs_names.intersection(meta.index)
        if len(common) > 0:
            adata = adata[common].copy()
            for col in meta.columns:
                adata.obs[col] = meta.loc[adata.obs_names, col].values
    adata.write(str(out_path))
    return out_path


def prepare_visium() -> Path:
    h5_path = VISIUM_DIR / "outs" / "filtered_feature_bc_matrix.h5"
    pos_path = VISIUM_DIR / "outs" / "spatial" / "tissue_positions.csv"
    out_path = VISIUM_DIR / "expression.h5ad"
    if not h5_path.exists():
        raise FileNotFoundError(f"缺少表达矩阵: {h5_path}")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    if pos_path.exists():
        pos = pd.read_csv(pos_path).set_index("barcode")
        common = adata.obs_names.intersection(pos.index)
        if len(common) > 0:
            adata = adata[common].copy()
            cols = [c for c in pos.columns if "pxl" in c.lower()]
            if len(cols) >= 2:
                adata.obsm["spatial"] = pos.loc[adata.obs_names, cols[:2]].values.astype(np.float64)
    adata.write(str(out_path))
    return out_path


def prepare_xenium() -> Path:
    h5_path = XENIUM_DIR / "cell_feature_matrix.h5"
    out_path = XENIUM_DIR / "expression.h5ad"
    if not h5_path.exists():
        raise FileNotFoundError(f"缺少表达矩阵: {h5_path}")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    adata.write(str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare h5ad for HyperSCA")
    parser.add_argument("--mode", choices=["demo", "research"], default="research")
    parser.add_argument("--modality", choices=["chromium", "visium", "xenium", "all"], default="all")
    parser.add_argument("--icb-root", default=str(DEFAULT_SOURCES["scCRC_ICB"]))
    parser.add_argument("--neu-root", default=str(DEFAULT_SOURCES["scCRC_Neu"]))
    parser.add_argument("--st-root", default=str(DEFAULT_SOURCES["ST_CRC_MSS"]))
    parser.add_argument("--ifng-root", default=str(DEFAULT_SOURCES["scCRC_IFNG"]))
    args = parser.parse_args()

    if args.mode == "research":
        return _prepare_multisource(args)

    results = {}
    modalities = {"chromium": prepare_chromium, "visium": prepare_visium, "xenium": prepare_xenium}
    targets = modalities if args.modality == "all" else {args.modality: modalities[args.modality]}
    for name, func in targets.items():
        try:
            path = func()
            results[name] = ("SUCCESS", str(path))
        except Exception as exc:
            results[name] = ("FAILED", str(exc))
    for name, (status, detail) in results.items():
        print(f"{name:12s}: {status} - {detail}")
    n_ok = sum(1 for s, _ in results.values() if s == "SUCCESS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
