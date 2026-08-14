#!/usr/bin/env python3
"""运行任务 C 的四种真实数据预演条件，并保留完整证据记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_rehearsal import (  # noqa: E402
    TaskCRehearsalError,
    run_task_c_rehearsal,
)


class _PlainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法完成任务 C 预演：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _PlainArgumentParser(
        description=(
            "在四种细胞环境条件下进行单随机种子的流程验证。"
            "结果只说明数据隔离、方法运行和评分步骤是否连通，不代表真实数据性能。"
        )
    )
    parser.add_argument(
        "--profile",
        choices=["connection", "comprehensive"],
        required=True,
        help="connection 为 64 基因连接检查；comprehensive 为 256 基因全方法预演。",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="种子 11 的已核验公开数据划分目录。",
    )
    parser.add_argument(
        "--method-assets-root",
        type=Path,
        required=True,
        help="固定版本的外部方法代码和隔离环境记录目录。",
    )
    parser.add_argument(
        "--prepared-identity-sha256",
        help=(
            "正式预演必填：从 prepare_task_c_data.py 输出独立保存的 seed 11 "
            "materialization_identity_sha256；本地重新计算值不是签名或外部信任根。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="保存本次预演记录的新目录；已有目录不会被覆盖。",
    )
    parser.add_argument(
        "--methods",
        required=True,
        help="按登记名称给出方法，使用英文逗号分隔。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "只复用完整重验通过且与外部 resume token 一致的既有预演；"
            "内部记录可被同步改写，不能替代该外部值。"
        ),
    )
    parser.add_argument(
        "--resume-token",
        help="初次运行 stdout 返回并由调用者独立保存的 resume_token。",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="使用小型合成闭环做最小运行检查；仍只属于流程验证。",
    )
    return parser


def _methods(value: str) -> tuple[str, ...]:
    methods = tuple(value.split(","))
    if not methods or any(not method or method != method.strip() for method in methods):
        raise TaskCRehearsalError(
            "--methods 必须是无空项、无额外空格的登记名称列表"
        )
    if len(set(methods)) != len(methods):
        raise TaskCRehearsalError("--methods 不能重复同一方法")
    return methods


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_task_c_rehearsal(
            profile=args.profile,
            prepared_root=args.prepared_root,
            prepared_identity_sha256=args.prepared_identity_sha256,
            method_assets_root=args.method_assets_root,
            output_root=args.output_root,
            method_ids=_methods(args.methods),
            expected_resume_token=args.resume_token,
            resume=args.resume,
            synthetic_smoke=args.synthetic_smoke,
            project_root=ROOT,
        )
    except (TaskCRehearsalError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
