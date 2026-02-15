#!/usr/bin/env python
"""
一键下载 LIANA / NicheNet / OmniPath 先验信号通路数据库。

用法
----
    python scripts/download_prior_db.py          # 首次下载（跳过已有缓存）
    python scripts/download_prior_db.py --force   # 强制重新下载
    python scripts/download_prior_db.py --summary # 仅打印已有缓存摘要

产物目录
--------
    data/prior_db/
    ├── omnipath/
    │   ├── lr_interactions.tsv        配体-受体互作（OmniPath ligrecextra）
    │   ├── dorothea_tf_target.tsv     TF-靶基因调控（DoRothEA）
    │   └── signaling_ppi.tsv          信号通路 PPI（OmniPath post_translational）
    ├── nichenet/
    │   ├── lr_network.tsv             NicheNet 配受体网络
    │   ├── lr_network.rds             NicheNet 原始 RDS
    │   ├── sig_network.tsv            NicheNet 信号网络
    │   ├── sig_network.rds
    │   ├── gr_network.tsv             NicheNet 基因调控网络
    │   └── gr_network.rds
    ├── liana/
    │   └── consensus_lr_resource.tsv  LIANA 共识配受体资源
    └── manifest.json                  下载清单与统计

环境变量
--------
    HYPERSCA_HTTP_TIMEOUT   HTTP 超时秒数（默认 180）
    HYPERSCA_HTTP_RETRIES   HTTP 重试次数（默认 3）
    HYPERSCA_HTTP_PROXY     HTTPS 代理地址（可选）
    HYPERSCA_PRIOR_DB_DIR   自定义缓存目录（默认 data/prior_db）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.prior_db._download import download_all
from src.data.prior_db._integrate import print_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载 LIANA / NicheNet / OmniPath 先验信号通路数据库"
    )
    parser.add_argument(
        "--force", action="store_true", help="强制重新下载（覆盖已有缓存）"
    )
    parser.add_argument(
        "--summary", action="store_true", help="仅打印已有缓存摘要，不下载"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="显示详细日志"
    )
    args = parser.parse_args()

    # 日志配置
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.summary:
        try:
            print_summary()
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        return

    manifest_path = download_all(force=args.force)
    print()
    print(f"[OK] manifest: {manifest_path}")
    print()
    print_summary()


if __name__ == "__main__":
    main()
