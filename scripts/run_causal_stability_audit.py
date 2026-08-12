"""Run the Step2 causal stability and negative-control sidecar."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.causal.stability_audit import run_causal_stability_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "通过重复抽样检查因果关系是否稳定，并用随机重排形成零效应对照。"
            "这些结果只提供补充证据，不改变候选靶点排序。"
        )
    )
    parser.add_argument(
        "--step2-dir",
        type=Path,
        default=ROOT / "results" / "step2",
        help="第二步因果分析结果所在目录，默认 results/step2。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "稳定性检查结果的保存目录；不填写时写入第二步结果目录下的 "
            "causal_audit。"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="将关系计为重复出现所需的最低频率，默认 0.5。",
    )
    parser.add_argument(
        "--n-null-controls",
        type=int,
        default=0,
        help="每种随机零效应对照的重复次数；0 表示不运行随机对照。",
    )
    parser.add_argument(
        "--null-modes",
        type=str,
        default=(
            "matrix_permutation,node_label_shuffle,"
            "outgoing_weight_permutation"
        ),
        help=(
            "以逗号分隔的随机零效应对照类型；这些对照只用于检查稳定性，"
            "不改变候选排序。"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="控制重复抽样的随机起点，默认 42；相同输入和数值可复现结果。",
    )
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
