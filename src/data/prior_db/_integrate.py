"""
先验数据库加载与整合。

从 data/prior_db/ 缓存读取已下载的 TSV 文件，
提供统一的 DataFrame 加载接口。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.prior_db._config import (
    MANIFEST_FILE,
    PRIOR_DB_DIR,
    SUBDIR_LIANA,
    SUBDIR_NICHENET,
    SUBDIR_OMNIPATH,
)

logger = logging.getLogger(__name__)


def _read_tsv(relpath: str) -> pd.DataFrame:
    """读取 PRIOR_DB_DIR 下的 TSV 文件。"""
    path = PRIOR_DB_DIR / relpath
    if not path.exists():
        raise FileNotFoundError(
            f"先验数据库文件不存在: {path}\n"
            "请先运行: python scripts/download_prior_db.py"
        )
    return pd.read_csv(path, sep="\t")


# ═══════════════════════════════════════════════════════════════
# OmniPath 数据加载
# ═══════════════════════════════════════════════════════════════

def load_lr_interactions() -> pd.DataFrame:
    """
    加载配体-受体互作表（OmniPath ligrecextra）。

    Returns
    -------
    pd.DataFrame
        列包含 source, target, source_genesymbol, target_genesymbol,
        is_directed, is_stimulation, is_inhibition, sources 等。
    """
    return _read_tsv(f"{SUBDIR_OMNIPATH}/lr_interactions.tsv")


def load_tf_targets() -> pd.DataFrame:
    """
    加载 TF-靶基因调控表（OmniPath DoRothEA）。

    Returns
    -------
    pd.DataFrame
        列包含 source_genesymbol (TF), target_genesymbol (target),
        is_stimulation, is_inhibition, dorothea_level, sources 等。
    """
    return _read_tsv(f"{SUBDIR_OMNIPATH}/dorothea_tf_target.tsv")


def load_signaling_network() -> pd.DataFrame:
    """
    加载信号通路 PPI 网络（OmniPath post_translational）。

    Returns
    -------
    pd.DataFrame
        列包含 source_genesymbol, target_genesymbol,
        is_directed, is_stimulation, is_inhibition, sources 等。
    """
    return _read_tsv(f"{SUBDIR_OMNIPATH}/signaling_ppi.tsv")


# ═══════════════════════════════════════════════════════════════
# NicheNet 数据加载
# ═══════════════════════════════════════════════════════════════

def load_nichenet_prior() -> dict[str, pd.DataFrame]:
    """
    加载 NicheNet v2 先验网络。

    Returns
    -------
    dict[str, pd.DataFrame]
        键: 'lr_network', 'sig_network', 'gr_network'
        分别对应配受体对、信号网络、基因调控网络。
    """
    result: dict[str, pd.DataFrame] = {}
    for name in ("lr_network", "signaling_network", "gr_network"):
        path = PRIOR_DB_DIR / SUBDIR_NICHENET / f"{name}.tsv"
        if path.exists():
            result[name] = pd.read_csv(path, sep="\t")
        else:
            logger.warning("NicheNet %s 不存在: %s", name, path)
    if not result:
        raise FileNotFoundError(
            f"NicheNet 先验数据不存在于 {PRIOR_DB_DIR / SUBDIR_NICHENET}\n"
            "请先运行: python scripts/download_prior_db.py"
        )
    return result


# ═══════════════════════════════════════════════════════════════
# LIANA 数据加载
# ═══════════════════════════════════════════════════════════════

def load_liana_resource() -> pd.DataFrame:
    """
    加载 LIANA 共识配受体资源。

    Returns
    -------
    pd.DataFrame
        列包含 ligand, receptor 及来源标注。
    """
    return _read_tsv(f"{SUBDIR_LIANA}/consensus_lr_resource.tsv")


# ═══════════════════════════════════════════════════════════════
# 清单与统计
# ═══════════════════════════════════════════════════════════════

def get_manifest() -> dict[str, Any]:
    """
    读取下载清单 manifest.json。

    Returns
    -------
    dict
        包含 generated_at, cache_dir, sections 等字段。
    """
    path = PRIOR_DB_DIR / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"清单文件不存在: {path}\n"
            "请先运行: python scripts/download_prior_db.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def print_summary() -> None:
    """打印先验数据库缓存摘要。"""
    manifest = get_manifest()
    print(f"生成时间: {manifest['generated_at']}")
    print(f"缓存目录: {manifest['cache_dir']}")
    print(f"总耗时:   {manifest['sections'].get('elapsed_seconds', '?')} 秒")
    print("-" * 60)
    for section, info in manifest["sections"].items():
        if section == "elapsed_seconds":
            continue
        if isinstance(info, dict) and "error" in info:
            print(f"  [{section}] ERROR: {info['error']}")
            continue
        if isinstance(info, dict):
            for name, meta in info.items():
                if isinstance(meta, dict) and "rows" in meta:
                    print(f"  [{section}/{name}] {meta['rows']:>8,} rows × {meta['cols']} cols")
