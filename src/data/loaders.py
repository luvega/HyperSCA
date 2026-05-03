"""统一数据加载函数。

支持三类加载：
- 元数据/坐标（Phase 0）
- 标准化 AnnData .h5ad（Phase 1+）
- 四项目标准化入库后的一体化 manifest / 表格加载（研究完整版）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_chromium_meta(data_dir: Path) -> pd.DataFrame:
    """加载 Chromium scRNA-seq 细胞元数据

    Returns
    -------
    pd.DataFrame
        列: Barcode, Patient, BC, QCFilter, Level1, Level2
    """
    path = data_dir / "cell_metadata.csv"
    _check(path)
    return pd.read_csv(path)


def load_visium_spatial(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """加载 Visium 空间坐标与缩放因子

    Returns
    -------
    positions : pd.DataFrame
        列: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    scalefactors : dict
    """
    pos_path = data_dir / "outs" / "spatial" / "tissue_positions.csv"
    sf_path = data_dir / "outs" / "spatial" / "scalefactors_json.json"
    _check(pos_path)
    _check(sf_path)
    positions = pd.read_csv(pos_path)
    with open(sf_path) as f:
        scalefactors = json.load(f)
    return positions, scalefactors


def load_visiumhd_geojson(data_dir: Path, layer: str = "cell") -> list[dict]:
    """加载 VisiumHD 细胞/核分割 GeoJSON（流式解析避免大内存）

    Parameters
    ----------
    layer : str
        "cell" 或 "nucleus"

    Returns
    -------
    list[dict]
        每个元素为 GeoJSON Feature
    """
    fname = f"{layer}_segmentations.geojson"
    path = data_dir / "segmented_outputs" / fname
    _check(path)
    with open(path) as f:
        data = json.load(f)
    return data.get("features", [])


def load_xenium_meta(data_dir: Path) -> tuple[dict, list[dict]]:
    """加载 Xenium 实验元信息与基因面板

    Returns
    -------
    experiment : dict
        experiment.xenium 中的元信息
    targets : list[dict]
        gene_panel.json 中的 targets 列表
    """
    exp_path = data_dir / "experiment.xenium"
    panel_path = data_dir / "gene_panel.json"
    _check(exp_path)
    _check(panel_path)
    with open(exp_path) as f:
        experiment = json.load(f)
    with open(panel_path) as f:
        panel = json.load(f)
    targets = panel.get("payload", {}).get("targets", [])
    return experiment, targets


def _check(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"数据文件缺失: {path}")


@dataclass
class ProjectSource:
    """四项目数据源配置。"""

    name: str
    source_path: str
    modality: str
    standardized_dir: str


def load_project_manifest(
    data_root: Optional[Path] = None,
    manifest_name: str = "project_manifest.json",
) -> list[ProjectSource]:
    """读取多项目标准化清单。"""
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data" / "metadata"
    path = Path(data_root) / manifest_name
    _check(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for row in payload.get("sources", []):
        sources.append(
            ProjectSource(
                name=str(row["name"]),
                source_path=str(row["source_path"]),
                modality=str(row["modality"]),
                standardized_dir=str(row["standardized_dir"]),
            )
        )
    return sources


def load_standardized_tables(
    data_root: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """读取 /data/metadata 下标准化表。"""
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data" / "metadata"
    root = Path(data_root)
    out: dict[str, pd.DataFrame] = {}
    for name in ["sample_table.csv", "entity_table.csv", "feature_table.csv", "measure_table.csv"]:
        f = root / name
        if f.exists():
            out[name.replace(".csv", "")] = pd.read_csv(f)
    return out


def load_project_h5ad(
    project_name: str,
    data_root: Optional[Path] = None,
    filename: str = "expression.h5ad",
) -> "anndata.AnnData":
    """按项目名称加载标准化 h5ad。

    project_name:
        scCRC_Neu / scCRC_IFNG / scCRC_ICB / ST_CRC_MSS
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data"
    base = Path(data_root)
    if project_name == "ST_CRC_MSS":
        folder = base / "ST" / project_name
        modality = "visium"
    else:
        folder = base / "scRNA" / project_name
        modality = "chromium"
    return load_h5ad(folder, modality=modality, filename=filename)


# =========================================================================
# AnnData .h5ad 加载（Phase 1+ 模型输入）
# =========================================================================

def load_h5ad(
    data_dir: Path,
    modality: str = "visium",
    filename: str = "expression.h5ad",
) -> "anndata.AnnData":
    """加载标准化 AnnData .h5ad 文件

    Parameters
    ----------
    data_dir : Path
        数据目录（如 data/Visium_HumanColon_Oliveira）
    modality : str
        数据模态，用于自动推断路径和校验：
        "chromium" / "visium" / "xenium"
    filename : str
        .h5ad 文件名（默认 expression.h5ad）

    Returns
    -------
    anndata.AnnData
        包含 .X（表达矩阵）、.obs（细胞注释）、.obsm['spatial']（坐标，如有）
    """
    import anndata

    path = Path(data_dir) / filename
    _check(path)

    adata = anndata.read_h5ad(str(path))

    # 基础校验
    if adata.n_obs == 0:
        raise ValueError(f"AnnData 为空: {path}")
    if adata.n_vars == 0:
        raise ValueError(f"AnnData 无基因: {path}")

    # 模态特异校验
    if modality == "visium" and "spatial" not in adata.obsm:
        import warnings
        warnings.warn(f"Visium AnnData 缺少 obsm['spatial'] 空间坐标: {path}")

    if modality == "chromium":
        expected_cols = {"Level1", "Level2"}
        missing = expected_cols - set(adata.obs.columns)
        if missing:
            import warnings
            warnings.warn(f"Chromium AnnData 缺少注释列: {missing}")

    return adata


def load_visium_h5ad(
    data_dir: Optional[Path] = None,
) -> "anndata.AnnData":
    """快捷方式：加载 Visium .h5ad 并校验空间坐标"""
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data" / "Visium_HumanColon_Oliveira"
    return load_h5ad(data_dir, modality="visium")


def load_chromium_h5ad(
    data_dir: Optional[Path] = None,
) -> "anndata.AnnData":
    """快捷方式：加载 Chromium scRNA-seq .h5ad"""
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data" / "Chromium_HumanColon_Oliveira"
    return load_h5ad(data_dir, modality="chromium")


# =========================================================================
# Reference model loading (data/ref)
# =========================================================================

@dataclass
class ReferenceManifest:
    """Parsed reference_manifest.json."""

    reference_name: str
    version: str
    model_type: str
    model_dir: str
    mappings_dir: str
    reference_h5ad: str
    label_key: str
    n_cells: int
    n_genes: int
    hvg_only: bool
    created: str


def load_reference_manifest(
    data_root: Optional[Path] = None,
    manifest_name: str = "reference_manifest.json",
) -> ReferenceManifest:
    """Read data/ref/manifest/reference_manifest.json."""
    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data"
    path = Path(data_root) / "ref" / "manifest" / manifest_name
    _check(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReferenceManifest(
        reference_name=payload.get("reference_name", ""),
        version=payload.get("version", ""),
        model_type=payload.get("model_type", ""),
        model_dir=payload.get("model_dir", ""),
        mappings_dir=payload.get("mappings_dir", ""),
        reference_h5ad=payload.get("reference_h5ad", ""),
        label_key=payload.get("label_key", ""),
        n_cells=int(payload.get("n_cells", 0)),
        n_genes=int(payload.get("n_genes", 0)),
        hvg_only=bool(payload.get("hvg_only", True)),
        created=payload.get("created", ""),
    )


def load_icb_reference(
    data_root: Optional[Path] = None,
) -> tuple["anndata.AnnData", dict]:
    """Load ICB reference AnnData + label_dict.

    Returns (adata, label_dict) where label_dict maps cell-type → count.
    """
    import anndata

    manifest = load_reference_manifest(data_root)
    ref_path = Path(manifest.reference_h5ad)
    _check(ref_path)
    adata = anndata.read_h5ad(str(ref_path))

    label_path = Path(manifest.mappings_dir) / "label_dict.json"
    label_dict: dict = {}
    if label_path.exists():
        label_dict = json.loads(label_path.read_text(encoding="utf-8"))

    return adata, label_dict
