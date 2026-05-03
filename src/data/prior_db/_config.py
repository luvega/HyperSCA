"""
先验数据库路径、URL 与网络配置。

环境变量
--------
HYPERSCA_PRIOR_DB_DIR   缓存目录（默认 <project>/data/prior_db）
HYPERSCA_HTTP_TIMEOUT   HTTP 超时秒数（默认 180）
HYPERSCA_HTTP_RETRIES   HTTP 重试次数（默认 3）
HTTPS_PROXY             HTTPS 代理（可选）
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 项目根目录推断 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/data/prior_db → 根

# ── 缓存目录 ────────────────────────────────────────────────────
PRIOR_DB_DIR = Path(
    os.environ.get("HYPERSCA_PRIOR_DB_DIR", str(PROJECT_ROOT / "data" / "prior_db"))
)

# ── 网络配置 ────────────────────────────────────────────────────
HTTP_TIMEOUT: int = int(os.environ.get("HYPERSCA_HTTP_TIMEOUT", "180"))
HTTP_RETRIES: int = int(os.environ.get("HYPERSCA_HTTP_RETRIES", "3"))
HTTPS_PROXY: str = os.environ.get(
    "HYPERSCA_HTTP_PROXY", os.environ.get("HTTPS_PROXY", "")
)

# ── OmniPath REST 端点 ──────────────────────────────────────────
OMNIPATH_BASE = "https://omnipathdb.org"
OMNIPATH_QUERIES: dict[str, str] = {
    "lr_interactions": (
        f"{OMNIPATH_BASE}/interactions"
        "?datasets=ligrecextra&genesymbols=1"
        "&fields=sources,references,curation_effort"
    ),
    "dorothea_tf_target": (
        f"{OMNIPATH_BASE}/interactions"
        "?datasets=dorothea&genesymbols=1"
        "&fields=sources,references,dorothea_level"
    ),
    "signaling_ppi": (
        f"{OMNIPATH_BASE}/interactions"
        "?datasets=omnipath&types=post_translational&genesymbols=1"
        "&fields=sources,references,curation_effort"
    ),
}

# ── NicheNet Zenodo (v2, 2021-12-21 release) ───────────────────
NICHENET_ZENODO = "https://zenodo.org/records/7074291/files"
NICHENET_FILES: dict[str, str] = {
    "lr_network": f"{NICHENET_ZENODO}/lr_network_human_21122021.rds",
    "signaling_network": f"{NICHENET_ZENODO}/signaling_network_human_21122021.rds",
    "gr_network": f"{NICHENET_ZENODO}/gr_network_human_21122021.rds",
}

# ── LIANA 共识配受体资源 ────────────────────────────────────────
# liana-py 包内含 consensus resource，也可从 OmniPath 获取
# 优先使用包内资源（无需网络），其次 REST API
LIANA_FALLBACK_URL = (
    f"{OMNIPATH_BASE}/interactions"
    "?datasets=ligrecextra&genesymbols=1"
    "&fields=sources,references,curation_effort"
)

# ── 子目录布局 ──────────────────────────────────────────────────
SUBDIR_OMNIPATH = "omnipath"
SUBDIR_NICHENET = "nichenet"
SUBDIR_LIANA = "liana"
SUBDIR_INTEGRATED = "integrated"
MANIFEST_FILE = "manifest.json"
