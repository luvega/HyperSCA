#!/usr/bin/env python3
"""核对任务 C 公开镜像与基因标识转换，并写外部来源记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_acquisition import (  # noqa: E402
    TaskCAcquisitionError,
    create_task_c_acquisition_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "核对 Zenodo 公开镜像、转换后的 H5AD 和可选的 Figshare 403 记录；"
            "不联网下载，也不把本机文件时间误写成下载时间。"
        )
    )
    parser.add_argument("--mirror-k562-h5ad", type=Path, required=True)
    parser.add_argument("--mirror-rpe1-h5ad", type=Path, required=True)
    parser.add_argument("--converted-k562-h5ad", type=Path, required=True)
    parser.add_argument("--converted-rpe1-h5ad", type=Path, required=True)
    parser.add_argument("--k562-figshare-403-evidence", type=Path)
    parser.add_argument("--rpe1-figshare-403-evidence", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="每次核对的细胞行数；程序仍会按 128 MiB 内存上限自动缩小。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    evidence = {
        context: path
        for context, path in (
            ("k562", args.k562_figshare_403_evidence),
            ("rpe1", args.rpe1_figshare_403_evidence),
        )
        if path is not None
    }
    try:
        summary = create_task_c_acquisition_manifest(
            mirror_paths={
                "k562": args.mirror_k562_h5ad,
                "rpe1": args.mirror_rpe1_h5ad,
            },
            converted_paths={
                "k562": args.converted_k562_h5ad,
                "rpe1": args.converted_rpe1_h5ad,
            },
            output_path=args.output_manifest,
            figshare_403_evidence=evidence,
            requested_chunk_rows=args.chunk_rows,
        )
    except TaskCAcquisitionError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
