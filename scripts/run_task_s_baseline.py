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

from src.evaluation.benchmark_contract import load_benchmark_contract
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
        description="Run an own-only or fixed-distance-decay Task S baseline."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--baseline-id",
        choices=["own_only", "fixed_distance_decay"],
        required=True,
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument(
        "--data-status",
        choices=["external_benchmark", "synthetic_smoke"],
        required=True,
    )
    parser.add_argument("--own-effect-source-id", required=True)
    parser.add_argument("--own-effect-source", type=Path, required=True)
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--length-scale-source-id", default=None)
    parser.add_argument("--length-scale-source", type=Path, default=None)
    parser.add_argument(
        "--attest-own-effect-train-only",
        action="store_true",
        help="Attest that own-effect predictions were fit without holdout outcomes.",
    )
    parser.add_argument(
        "--attest-nonadjacent-spatial-blocks",
        action="store_true",
        help="Attest that held-out spatial blocks do not touch training blocks.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "benchmark_contract_v1.json",
    )
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
