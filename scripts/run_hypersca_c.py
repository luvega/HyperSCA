#!/usr/bin/env python3
"""在允许查看的任务 C 文件上运行 HyperSCA-C。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.causal.hypersca_c import HyperSCACError
from src.causal.hypersca_c_run import run_hypersca_c
from src.evaluation.task_c_data import TaskCDataError


class _ChineseArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法运行 HyperSCA-C：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _ChineseArgumentParser(
        description="从允许查看的单细胞干预数据学习 HyperSCA-C 候选基因关系。"
    )
    parser.add_argument(
        "--context",
        action="append",
        help="公开训练文件，格式为 k562=path 或 rpe1=path，可重复提供。",
    )
    parser.add_argument(
        "--profile-input",
        type=Path,
        help="从公开父文件重算并限制基因、细胞数量的统一比较输入。",
    )
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        help="统一比较输入的来源、抽样位置和转换记录。",
    )
    parser.add_argument(
        "--profile-identity-input",
        type=Path,
        help="统一比较输入的稳定记录路径；实际读取可使用已核验的临时副本。",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        try:
            seed = int(args.seed)
        except (TypeError, ValueError) as exc:
            raise HyperSCACError("seed 必须是整数") from exc
        summary = run_hypersca_c(
            context_values=args.context or (),
            profile_input_path=args.profile_input,
            profile_identity_path=args.profile_identity_input,
            profile_manifest_path=args.profile_manifest,
            config_path=args.config,
            gene_list_path=args.gene_list,
            public_manifest_path=args.public_manifest,
            output_dir=args.output_dir,
            seed=seed,
            device=args.device,
        )
    except (HyperSCACError, TaskCDataError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
