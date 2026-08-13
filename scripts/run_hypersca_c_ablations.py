#!/usr/bin/env python3
"""在允许查看的任务 C 文件上运行八项预登记消融。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.causal.hypersca_c import HyperSCACError
from src.causal.hypersca_c_ablation import run_hypersca_c_ablations
from src.evaluation.task_c_data import TaskCDataError


class _ChineseArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法运行 HyperSCA-C 消融：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _ChineseArgumentParser(
        description=(
            "按固定顺序运行八项 HyperSCA-C 消融；输出只表示原始推断，"
            "不表示已经完成正式封存评分。"
        )
    )
    parser.add_argument(
        "--context",
        action="append",
        required=True,
        help="公开训练文件，格式为 k562=path 或 rpe1=path，可重复提供。",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--ablation-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--prior-edges",
        type=Path,
        help="可选的独立外部先验关系 CSV；还必须同时提供来源清单。",
    )
    parser.add_argument(
        "--prior-source-manifest",
        type=Path,
        help="可选的外部先验来源与不重用评分关系的证明清单。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        try:
            seed = int(args.seed)
        except (TypeError, ValueError) as exc:
            raise HyperSCACError("seed 必须是整数") from exc
        result = run_hypersca_c_ablations(
            context_values=args.context,
            config_path=args.config,
            gene_list_path=args.gene_list,
            public_manifest_path=args.public_manifest,
            ablation_registry_path=args.ablation_registry,
            output_root=args.output_root,
            seed=seed,
            device=args.device,
            prior_edges_path=args.prior_edges,
            prior_source_manifest_path=args.prior_source_manifest,
        )
    except (HyperSCACError, TaskCDataError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
