"""Run the Step2 causal stability and negative-control sidecar."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.causal.stability_audit import run_causal_stability_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperSCA causal stability audit")
    parser.add_argument("--step2-dir", type=Path, default=ROOT / "results" / "step2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-null-controls", type=int, default=0)
    parser.add_argument(
        "--null-modes",
        type=str,
        default=(
            "matrix_permutation,node_label_shuffle,"
            "outgoing_weight_permutation"
        ),
        help="Comma-separated bootstrap-frequency surrogate null modes.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.step2_dir / "causal_audit"
    result = run_causal_stability_audit(
        step2_dir=args.step2_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        n_null_controls=args.n_null_controls,
        null_modes=tuple(
            item.strip() for item in args.null_modes.split(",") if item.strip()
        ),
        random_seed=args.random_seed,
    )
    print(f"Edge stability rows: {len(result['edge_stability'])}")
    for label, path in result["output_paths"].items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
