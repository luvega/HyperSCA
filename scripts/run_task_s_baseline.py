"""Run one pre-registered Task S simple spatial baseline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_contract import (
    BenchmarkContractError,
    load_benchmark_contract,
)
from src.evaluation.task_c_benchmark import sha256_file
from src.evaluation.task_s_benchmark import TaskSBenchmarkError, run_task_s_baseline


def _current_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "unknown-revision"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在独立验证数据上分别评估自身效应和邻近细胞效应。两个简单对照方法使用"
            "相同的自身效应输入，以便单独判断空间信息是否带来增益。"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help=(
            "独立验证数据 CSV，需包含任务 S 规定的自身效应、邻近效应和空间分组字段。"
        ),
    )
    parser.add_argument(
        "--baseline-id",
        choices=["own_only", "fixed_distance_decay"],
        required=True,
        help=(
            "简单对照方法：own_only 只用自身效应；"
            "fixed_distance_decay 再加入固定距离衰减。"
        ),
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="本次使用的数据集标识，写入结果记录以便追溯。",
    )
    parser.add_argument(
        "--dataset-source",
        required=True,
        help="数据来源说明，写入结果记录以便追溯。",
    )
    parser.add_argument(
        "--data-status",
        choices=["external_benchmark", "synthetic_smoke"],
        required=True,
        help=(
            "external_benchmark 表示独立外部数据；"
            "synthetic_smoke 只表示合成流程检查。"
        ),
    )
    parser.add_argument(
        "--own-effect-source-id",
        required=True,
        help=(
            "自身效应预测的来源标识，用于核对两个简单对照是否使用相同输入。"
        ),
    )
    parser.add_argument(
        "--own-effect-source",
        type=Path,
        required=True,
        help=(
            "自身效应预测来源文件；程序记录其校验值，但不从中读取验证结果。"
        ),
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=None,
        help="固定距离衰减的长度尺度；own_only 不使用该值。",
    )
    parser.add_argument(
        "--length-scale-source-id",
        default=None,
        help=(
            "长度尺度来源的标识；使用 fixed_distance_decay 时按比较规则提供。"
        ),
    )
    parser.add_argument(
        "--length-scale-source",
        type=Path,
        default=None,
        help="记录 length_scale 的 JSON 文件；程序核对数值并记录校验值。",
    )
    parser.add_argument(
        "--attest-own-effect-train-only",
        action="store_true",
        help="确认自身效应预测未使用独立验证集的结果。",
    )
    parser.add_argument(
        "--attest-nonadjacent-spatial-blocks",
        action="store_true",
        help="确认验证空间分区与训练空间分区不相邻。",
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
        "--code-revision",
        default=None,
        help="可选的代码版本标识；不填写时自动读取当前 Git 版本。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=11,
        help="控制可重复计算的随机起点，默认 11。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="保存指标、预测和分析记录的目录。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        holdout = pd.read_csv(args.input_csv)
        contract = load_benchmark_contract(args.contract)
        length_scale_digest = None
        length_scale = args.length_scale
        if args.length_scale_source is not None:
            try:
                length_payload = json.loads(
                    args.length_scale_source.read_text(encoding="utf-8")
                )
                recorded_length_scale = float(length_payload["length_scale"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise TaskSBenchmarkError(
                    "length-scale source must be JSON with a numeric length_scale"
                ) from exc
            if length_scale is None:
                length_scale = recorded_length_scale
            elif not abs(length_scale - recorded_length_scale) <= 1e-12:
                raise TaskSBenchmarkError(
                    "--length-scale does not match the length-scale source artifact"
                )
            length_scale_digest = sha256_file(args.length_scale_source)
        run = run_task_s_baseline(
            holdout=holdout,
            contract=contract,
            baseline_id=args.baseline_id,
            dataset_id=args.dataset_id,
            dataset_source=args.dataset_source,
            data_status=args.data_status,
            input_digest=sha256_file(args.input_csv),
            own_effect_source_id=args.own_effect_source_id,
            own_effect_source_digest=sha256_file(args.own_effect_source),
            length_scale=length_scale,
            length_scale_source_id=args.length_scale_source_id,
            length_scale_source_digest=length_scale_digest,
            train_only_attested=args.attest_own_effect_train_only,
            nonadjacent_blocks_attested=args.attest_nonadjacent_spatial_blocks,
            code_revision=args.code_revision or _current_revision(),
            random_seed=args.random_seed,
            output_dir=args.output_dir,
        )
    except (TaskSBenchmarkError, BenchmarkContractError) as exc:
        parser.error(
            f"无法继续：{exc}。请检查输入文件、字段和参数是否符合文档中的数据规范。"
        )
    summary = {
        "status": run["metrics"]["status"],
        "baseline_id": args.baseline_id,
        "dataset_id": args.dataset_id,
        "data_status": args.data_status,
        "neighbor_effect_rmse": run["metrics"]["neighbor_effect_rmse"],
        "own_effect_rmse": run["metrics"]["own_effect_rmse"],
        "coverage": run["metrics"]["coverage"],
        "abstention_rate": run["metrics"]["abstention_rate"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
