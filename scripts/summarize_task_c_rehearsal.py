#!/usr/bin/env python3
"""只读核对任务 C 预演，并生成尚未获准执行的全量作业草案。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_aggregation import (  # noqa: E402
    TaskCAggregationError,
    summarize_task_c_rehearsal,
)


class _PlainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法汇总任务 C 预演：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _PlainArgumentParser(
        description=(
            "只读核对单随机种子真实数据预演，汇总方法状态和实测资源，"
            "并写出不可直接执行的五份数据划分作业草案。"
            "该结果不比较性能优劣，也不授权启动正式作业。"
        )
    )
    parser.add_argument(
        "--rehearsal-root",
        type=Path,
        required=True,
        help="预演控制器写出的完整证据目录；命令只读取，不改动其中内容。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="保存四份汇总文件的新目录；若目录已存在则停止，不会覆盖。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = summarize_task_c_rehearsal(
            rehearsal_root=args.rehearsal_root,
            output_dir=args.output_dir,
        )
    except (TaskCAggregationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
