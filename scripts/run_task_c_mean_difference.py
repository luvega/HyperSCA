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

from src.evaluation.benchmark_contract import load_benchmark_contract
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
            "Run the deterministic Task C mean-difference baseline using the "
            "three-array NPZ produced by CausalBench CreateDataset."
        )
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument(
        "--data-status",
        choices=["external_benchmark", "synthetic_smoke"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "benchmark_contract_v1.json",
    )
    parser.add_argument("--reference-edges", type=Path, default=None)
    parser.add_argument("--reference-id", default=None)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--excluded-label", default="excluded")
    parser.add_argument("--min-cells-per-intervention", type=int, default=5)
    parser.add_argument("--precision-at-k", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--code-revision", default=None)
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
    args = build_parser().parse_args(argv)
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
