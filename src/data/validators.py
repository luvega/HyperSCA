"""数据完整性与多项目标准化校验。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> list[str]:
    """通用必需列校验。"""
    req = set(required)
    missing = sorted(req - set(df.columns))
    if not missing:
        return []
    return [f"[{table_name}] 缺失必需列: {missing}"]


def validate_chromium_meta(df: pd.DataFrame) -> list[str]:
    """校验 Chromium 元数据字段完整性，返回问题列表。"""
    issues = validate_required_columns(
        df, {"Barcode", "Patient", "QCFilter", "Level1", "Level2"}, "chromium_meta"
    )
    if "Barcode" in df.columns and df["Barcode"].duplicated().any():
        n = int(df["Barcode"].duplicated().sum())
        issues.append(f"[chromium_meta] Barcode 重复: {n} 条")
    return issues


def validate_visium_positions(df: pd.DataFrame) -> list[str]:
    """校验 Visium 空间坐标。"""
    issues = validate_required_columns(
        df,
        {"barcode", "in_tissue", "pxl_row_in_fullres", "pxl_col_in_fullres"},
        "visium_positions",
    )
    cols = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    if all(c in df.columns for c in cols) and df[cols].isna().any().any():
        issues.append("[visium_positions] 存在 NaN 空间坐标")
    return issues


def validate_geojson_features(features: list[dict], min_count: int = 1) -> list[str]:
    """校验 GeoJSON feature 列表。"""
    issues = []
    if len(features) < min_count:
        issues.append(f"Feature 数量过少: {len(features)} < {min_count}")
    if features and "geometry" not in features[0]:
        issues.append("首条 Feature 缺少 geometry 字段")
    return issues


def check_file_exists(path: Path, label: str = "") -> list[str]:
    """检查文件是否存在。"""
    if not path.exists():
        return [f"文件缺失{' (' + label + ')' if label else ''}: {path}"]
    return []


def validate_multisource_min_fields(df: pd.DataFrame) -> list[str]:
    """校验干湿回写和统一表的最小字段约束。"""
    required = {
        "sample_id",
        "mmr_group",
        "celltype",
        "spot_or_cell_id",
        "x",
        "y",
    }
    return validate_required_columns(df, required, "multisource_min_fields")


def validate_experiment_roundtrip_fields(df: pd.DataFrame) -> list[str]:
    """校验实验回写表字段约束。"""
    required = {
        "sample_id",
        "timepoint",
        "dose",
        "gene",
        "effect_size",
    }
    return validate_required_columns(df, required, "experiment_roundtrip")


def validate_onboarding_tree(data_root: Path) -> list[str]:
    """校验 /data 入库后目录结构。"""
    issues: list[str] = []
    expected_dirs = [
        data_root / "scRNA",
        data_root / "ST",
        data_root / "metadata",
    ]
    for d in expected_dirs:
        if not d.exists():
            issues.append(f"缺少目录: {d}")
    # 四项目目录
    project_dirs = [
        data_root / "scRNA" / "scCRC_Neu",
        data_root / "scRNA" / "scCRC_IFNG",
        data_root / "scRNA" / "scCRC_ICB",
        data_root / "ST" / "ST_CRC_MSS",
    ]
    for d in project_dirs:
        if not d.exists():
            issues.append(f"缺少项目目录: {d}")
    return issues


def validate_reference_tree(data_root: Path, strict: bool = False) -> list[str]:
    """校验 data/ref 参考模型目录结构。

    Parameters
    ----------
    data_root : Path
        data/ 根目录
    strict : bool
        True = 缺失项视为错误；False = 仅报告警告（默认非阻断）
    """
    import json as _json

    ref_root = data_root / "ref"
    issues: list[str] = []
    prefix = "ERROR" if strict else "WARN"

    if not ref_root.exists():
        issues.append(f"{prefix}: data/ref 目录不存在")
        return issues

    manifest_path = ref_root / "manifest" / "reference_manifest.json"
    if not manifest_path.exists():
        issues.append(f"{prefix}: reference_manifest.json 不存在")
        return issues

    try:
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"{prefix}: manifest 解析失败: {exc}")
        return issues

    model_dir = Path(manifest.get("model_dir", ""))
    if not model_dir.exists():
        issues.append(f"{prefix}: 模型目录不存在: {model_dir}")

    mappings_dir = Path(manifest.get("mappings_dir", ""))
    expected_files = ["label_dict.json", "mapping_stats.json"]
    for fname in expected_files:
        fp = mappings_dir / fname
        if not fp.exists():
            issues.append(f"{prefix}: 映射文件缺失: {fp}")

    ref_h5ad = Path(manifest.get("reference_h5ad", ""))
    if not ref_h5ad.exists():
        issues.append(f"{prefix}: reference AnnData 不存在: {ref_h5ad}")

    n_cells = manifest.get("n_cells", 0)
    if n_cells < 100:
        issues.append(f"{prefix}: reference 细胞数过少 ({n_cells})")

    return issues
