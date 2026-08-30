#!/usr/bin/env python3
"""Run one fixed Task C null input under the shared runtime supervisor."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method-id", choices=("hypersca_c", "mean_difference"), required=True
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-cells", type=int, required=True)
    parser.add_argument("--hypersca-config", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--scientific-status", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.evaluation.task_c_rehearsal import (
        _scientific_null_predictions,
        _write_new_bytes,
        _write_new_record,
    )

    args = build_parser().parse_args(argv)
    try:
        predictions = _scientific_null_predictions(
            method_id=args.method_id,
            profile_record={
                "input": args.input_npz,
                "manifest": args.profile_manifest,
            },
            seed=args.seed,
            min_cells=args.min_cells,
            hypersca_config_path=args.hypersca_config,
        )
        _write_new_bytes(
            args.output_csv, predictions.to_csv(index=False).encode("utf-8")
        )
        _write_new_record(
            args.scientific_status,
            {
                "schema_version": "1.0",
                "status": "completed",
                "scientific_status": dict(
                    predictions.attrs.get("formal_null_scientific_status", {})
                ),
            },
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
