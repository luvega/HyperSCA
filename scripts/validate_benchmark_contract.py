"""Validate and optionally snapshot the HyperSCA C/S/D benchmark contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_contract import (
    contract_digest,
    load_benchmark_contract,
    write_preregistration_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen HyperSCA Tasks C/S/D benchmark contract."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "benchmark_contract_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for contract_snapshot.json and task_registry.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_benchmark_contract(args.contract)
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
