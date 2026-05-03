"""
先验数据库一键下载逻辑。

download_all(force=False)  统一入口
├── _download_omnipath()   OmniPath REST API → TSV
├── _download_nichenet()   Zenodo RDS → TSV（需 pyreadr）
└── _download_liana()      liana-py 内置 / OmniPath 回退
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from src.data.prior_db._config import (
    HTTP_RETRIES,
    HTTP_TIMEOUT,
    HTTPS_PROXY,
    LIANA_FALLBACK_URL,
    MANIFEST_FILE,
    NICHENET_FILES,
    OMNIPATH_QUERIES,
    PRIOR_DB_DIR,
    SUBDIR_INTEGRATED,
    SUBDIR_LIANA,
    SUBDIR_NICHENET,
    SUBDIR_OMNIPATH,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# HTTP 工具
# ═══════════════════════════════════════════════════════════════

def _get_session() -> requests.Session:
    """创建带代理与重试的 requests Session。"""
    sess = requests.Session()
    if HTTPS_PROXY:
        sess.proxies.update({"https": HTTPS_PROXY, "http": HTTPS_PROXY})
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=HTTP_RETRIES,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
    )
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _download_to_bytes(url: str, desc: str) -> bytes:
    """下载 URL 内容，返回字节，带进度条。"""
    sess = _get_session()
    resp = sess.get(url, timeout=HTTP_TIMEOUT, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue()


def _download_tsv(url: str, desc: str) -> pd.DataFrame:
    """下载 OmniPath REST TSV 端点 → DataFrame。"""
    raw = _download_to_bytes(url, desc)
    return pd.read_csv(io.BytesIO(raw), sep="\t")


def _save_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    logger.info("  saved %s  (%d rows × %d cols)", path.name, *df.shape)


# ═══════════════════════════════════════════════════════════════
# OmniPath
# ═══════════════════════════════════════════════════════════════

def _download_omnipath(out_dir: Path, force: bool) -> dict[str, Any]:
    """下载 OmniPath 三类数据集。"""
    results: dict[str, Any] = {}
    for name, url in OMNIPATH_QUERIES.items():
        dest = out_dir / f"{name}.tsv"
        if dest.exists() and not force:
            logger.info("  [skip] %s 已存在", dest)
            df = pd.read_csv(dest, sep="\t")
        else:
            logger.info("  [download] %s ...", name)
            df = _download_tsv(url, f"OmniPath/{name}")
            _save_tsv(df, dest)
        results[name] = {"rows": len(df), "cols": len(df.columns), "file": str(dest)}
    return results


# ═══════════════════════════════════════════════════════════════
# NicheNet
# ═══════════════════════════════════════════════════════════════

def _download_nichenet(out_dir: Path, force: bool) -> dict[str, Any]:
    """下载 NicheNet v2 先验网络（RDS → TSV）。"""
    results: dict[str, Any] = {}
    try:
        import pyreadr
    except ImportError:
        logger.warning("  pyreadr 未安装，跳过 NicheNet RDS 下载。pip install pyreadr")
        return results

    for name, url in NICHENET_FILES.items():
        dest_tsv = out_dir / f"{name}.tsv"
        if dest_tsv.exists() and not force:
            logger.info("  [skip] %s 已存在", dest_tsv)
            df = pd.read_csv(dest_tsv, sep="\t")
        else:
            logger.info("  [download] NicheNet/%s ...", name)
            raw = _download_to_bytes(url, f"NicheNet/{name}")
            # 写临时 RDS 再读取
            tmp_rds = out_dir / f"{name}.rds"
            tmp_rds.parent.mkdir(parents=True, exist_ok=True)
            tmp_rds.write_bytes(raw)
            rds_data = pyreadr.read_r(str(tmp_rds))
            # RDS 通常只含一个 DataFrame
            df = list(rds_data.values())[0]
            _save_tsv(df, dest_tsv)
            # 保留 RDS 原始文件作为备份，不删除
        results[name] = {"rows": len(df), "cols": len(df.columns), "file": str(dest_tsv)}
    return results


# ═══════════════════════════════════════════════════════════════
# LIANA
# ═══════════════════════════════════════════════════════════════

def _download_liana(out_dir: Path, force: bool) -> dict[str, Any]:
    """获取 LIANA 共识配受体资源。"""
    dest = out_dir / "consensus_lr_resource.tsv"
    results: dict[str, Any] = {}

    if dest.exists() and not force:
        logger.info("  [skip] %s 已存在", dest)
        df = pd.read_csv(dest, sep="\t")
    else:
        # 策略 1: 从 liana-py 包内直接加载（无需网络）
        df = _try_liana_package()
        if df is None:
            # 策略 2: 从 OmniPath REST API 回退
            logger.info("  [fallback] 从 OmniPath REST API 下载 LR 资源 ...")
            df = _download_tsv(LIANA_FALLBACK_URL, "LIANA/fallback")
        _save_tsv(df, dest)

    results["consensus_lr_resource"] = {
        "rows": len(df),
        "cols": len(df.columns),
        "file": str(dest),
    }
    return results


def _try_liana_package() -> pd.DataFrame | None:
    """尝试从 liana-py 包内加载共识 LR 资源。"""
    try:
        from liana.resource import select_resource
        df = select_resource("consensus")
        if df is not None and not df.empty:
            logger.info("  [liana-py] 从包内加载 consensus resource (%d rows)", len(df))
            return df
    except Exception as exc:  # noqa: BLE001
        logger.debug("  liana-py 加载失败: %s", exc)
    return None


# ═══════════════════════════════════════════════════════════════
# 整合与清单
# ═══════════════════════════════════════════════════════════════

def _write_manifest(base: Path, sections: dict[str, Any]) -> Path:
    """生成 manifest.json。"""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(base),
        "sections": sections,
    }
    path = base / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("清单已写入 %s", path)
    return path


# ═══════════════════════════════════════════════════════════════
# 公共入口
# ═══════════════════════════════════════════════════════════════

def download_all(force: bool = False) -> Path:
    """
    一键下载全部先验数据库。

    Parameters
    ----------
    force : bool
        若为 True，强制重新下载（覆盖已有缓存）。

    Returns
    -------
    Path
        manifest.json 路径。
    """
    base = PRIOR_DB_DIR
    base.mkdir(parents=True, exist_ok=True)
    logger.info("先验数据库缓存目录: %s", base)

    sections: dict[str, Any] = {}
    t0 = time.time()

    # ── OmniPath ────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("[1/3] OmniPath（LR 互作 / DoRothEA TF-target / 信号通路 PPI）")
    try:
        sections["omnipath"] = _download_omnipath(base / SUBDIR_OMNIPATH, force)
    except Exception as exc:
        logger.error("OmniPath 下载失败: %s", exc)
        sections["omnipath"] = {"error": str(exc)}

    # ── NicheNet ────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("[2/3] NicheNet（先验配受体 / 信号 / 基因调控网络）")
    try:
        sections["nichenet"] = _download_nichenet(base / SUBDIR_NICHENET, force)
    except Exception as exc:
        logger.error("NicheNet 下载失败: %s", exc)
        sections["nichenet"] = {"error": str(exc)}

    # ── LIANA ───────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("[3/3] LIANA（共识配受体资源）")
    try:
        sections["liana"] = _download_liana(base / SUBDIR_LIANA, force)
    except Exception as exc:
        logger.error("LIANA 下载失败: %s", exc)
        sections["liana"] = {"error": str(exc)}

    # ── 清单 ────────────────────────────────────────────────
    elapsed = time.time() - t0
    sections["elapsed_seconds"] = round(elapsed, 1)
    manifest_path = _write_manifest(base, sections)

    logger.info("=" * 50)
    logger.info("全部完成，耗时 %.1f 秒", elapsed)
    return manifest_path
