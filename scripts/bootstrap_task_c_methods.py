#!/usr/bin/env python3
"""获取任务 C 比较方法的固定官方代码，并记录隔离环境。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_method_registry import TaskCMethodRegistryError
from src.evaluation.task_c_runtime import TaskCRuntimeError, bootstrap_task_c_methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "获取公平比较所需的固定版本官方代码，创建隔离环境，"
            "并记录哪些论文目前没有可运行的官方实现。"
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="保存外部官方代码、环境清单和不可用状态的新目录。",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/task_c_methods_v1.json",
        help="已经审核并固定版本的方法清单。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = bootstrap_task_c_methods(
            cache_root=arguments.cache_root,
            registry_path=arguments.registry,
            project_root=ROOT,
        )
    except (TaskCRuntimeError, TaskCMethodRegistryError) as exc:
        parser.error(
            f"无法准备比较方法：{exc}。请检查缓存目录、固定版本和隔离环境配置。"
        )
    print(
        "比较方法准备完成："
        f"{summary['source_count']} 份官方代码，"
        f"{summary['environment_count']} 个隔离环境，"
        f"{summary['publication_only_count']} 种方法已明确记录为暂无官方代码。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
