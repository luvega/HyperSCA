"""Run the Task C mean-difference baseline on a CausalBench NPZ cache."""
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
from src.evaluation.task_c_benchmark import (
    TaskCBenchmarkError,
    load_causalbench_npz,
    run_task_c_mean_difference,
    sha256_file,
)


def _current_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision or "unknown-revision"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在单细胞干预数据上运行均值差简单对照方法，用于判断更复杂的因果网络方法"
            "是否真正超过直接利用干预标签的结果。"
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        required=True,
        help="CausalBench 生成的三数组 NPZ 输入文件。",
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
        "--context-id",
        required=True,
        help="本次评估的细胞、组织或实验情境标识。",
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
        "--output-dir",
        type=Path,
        required=True,
        help="保存指标、预测和分析记录的目录。",
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
        "--reference-edges",
        type=Path,
        default=None,
        help="可选的参考关系表 CSV；提供时必须同时提供 --reference-id。",
    )
    parser.add_argument(
        "--reference-id",
        default=None,
        help="参考关系表的来源标识；提供时必须同时提供 --reference-edges。",
    )
    parser.add_argument(
        "--source-column",
        default="source",
        help="参考关系表中起点对象的列名，默认 source。",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="参考关系表中终点对象的列名，默认 target。",
    )
    parser.add_argument(
        "--control-label",
        default="non-targeting",
        help="输入数据中未定向干预对照组的标签，默认 non-targeting。",
    )
    parser.add_argument(
        "--excluded-label",
        default="excluded",
        help="输入数据中不参与评估的样本标签，默认 excluded。",
    )
    parser.add_argument(
        "--min-cells-per-intervention",
        type=int,
        default=5,
        help="每个干预至少需要的细胞数，默认 5；不足时该干预不参与估计。",
    )
    parser.add_argument(
        "--precision-at-k",
        type=int,
        default=1000,
        help="计算前 k 个预测关系精确率时采用的 k，默认 1000。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=11,
        help="控制可重复计算的随机起点，默认 11。",
    )
    parser.add_argument(
        "--code-revision",
        default=None,
        help="可选的代码版本标识；不填写时自动读取当前 Git 版本。",
    )
    return parser


def _load_reference_edges(
    path: Path,
    source_column: str,
    target_column: str,
) -> set[tuple[str, str]]:
    table = pd.read_csv(path)
    missing = {source_column, target_column} - set(table.columns)
    if missing:
        raise TaskCBenchmarkError(
            f"reference edge table is missing columns: {sorted(missing)}"
        )
    selected = table[[source_column, target_column]].dropna()
    return {
        (str(source), str(target))
        for source, target in selected.itertuples(index=False, name=None)
        if str(source) != str(target)
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if (args.reference_edges is None) != (args.reference_id is None):
            raise TaskCBenchmarkError(
                "--reference-edges and --reference-id must be supplied together"
            )
        expression, interventions, gene_names = load_causalbench_npz(args.input_npz)
        reference_edges = None
        reference_digest = None
        if args.reference_edges is not None:
            reference_edges = _load_reference_edges(
                args.reference_edges,
                args.source_column,
                args.target_column,
            )
            reference_digest = sha256_file(args.reference_edges)
        contract = load_benchmark_contract(args.contract)
        run = run_task_c_mean_difference(
            expression=expression,
            interventions=interventions,
            gene_names=gene_names,
            contract=contract,
            dataset_id=args.dataset_id,
            dataset_source=args.dataset_source,
            context_id=args.context_id,
            data_status=args.data_status,
            input_digest=sha256_file(args.input_npz),
            code_revision=args.code_revision or _current_revision(),
            random_seed=args.random_seed,
            output_dir=args.output_dir,
            reference_edges=reference_edges,
            reference_id=args.reference_id,
            reference_digest=reference_digest,
            control_label=args.control_label,
            excluded_label=args.excluded_label,
            min_cells_per_intervention=args.min_cells_per_intervention,
            precision_at_k=args.precision_at_k,
        )
    except (TaskCBenchmarkError, BenchmarkContractError) as exc:
        parser.error(
            f"无法继续：{exc}。请检查输入文件、字段和参数是否符合文档中的数据规范。"
        )
    summary = {
        "status": run["metrics"]["status"],
        "dataset_id": args.dataset_id,
        "context_id": args.context_id,
        "data_status": args.data_status,
        "average_precision": run["metrics"]["average_precision"],
        "coverage": run["metrics"]["coverage"],
        "abstention_rate": run["metrics"]["abstention_rate"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
