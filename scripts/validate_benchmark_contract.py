"""Validate and optionally snapshot the HyperSCA C/S/D benchmark contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_contract import (
    BenchmarkContractError,
    contract_digest,
    load_benchmark_contract,
    write_preregistration_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "检查任务 C/S/D 的预先固定比较规则是否完整。该命令不会运行模型，"
            "也不会产生方法优越性的结论。"
        )
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "benchmark_contract_v1.json",
        help=(
            "预先固定的比较规则文件，默认 "
            "configs/benchmark_contract_v1.json。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选的规则快照保存目录；不填写时只检查，不写快照。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        contract = load_benchmark_contract(args.contract)
    except BenchmarkContractError as exc:
        parser.error(
            f"无法继续：{exc}。请检查输入文件、字段和参数是否符合文档中的数据规范。"
        )
    summary = {
        "status": "valid",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_count": len(contract["tasks"]),
        "tasks": list(contract["tasks"]),
    }
    if args.output_dir is not None:
        summary.update(write_preregistration_bundle(contract, args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
