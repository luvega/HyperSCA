"""CLI entrypoint for behavior grammar rule export and virtual tissue simulation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.behavior_grammar.config import BehaviorGrammarConfig, BehaviorGrammarPaths
from src.behavior_grammar.demo import write_demo_discovery_run
from src.behavior_grammar.pipeline import BehaviorGrammarPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperSCA Behavior Grammar Stage 5 Sidecar")
    parser.add_argument(
        "--discovery-manifest",
        type=Path,
        default=None,
        help="Path to results/discovery/target_discovery/<run_id>/manifest.json",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a small synthetic CRC TME discovery run before running the Stage 5 sidecar.",
    )
    parser.add_argument("--step4-dir", type=Path, default=ROOT / "results" / "step4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "behavior_grammar")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-rules", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--sensitivity-delta", type=float, default=0.10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.demo and args.discovery_manifest is not None:
        raise SystemExit("--demo and --discovery-manifest are mutually exclusive")
    if not args.demo and args.discovery_manifest is None:
        raise SystemExit("provide --discovery-manifest or use --demo")

    discovery_manifest = args.discovery_manifest
    if args.demo:
        demo_run_id = args.run_id or "demo_behavior_grammar"
        demo_root = _infer_demo_root(args.output_dir)
        discovery_manifest = write_demo_discovery_run(demo_root, run_id=demo_run_id)

    paths = BehaviorGrammarPaths.default(
        root=ROOT,
        discovery_manifest=discovery_manifest,
        step4_dir=args.step4_dir,
        output_base=args.output_dir,
    )
    config = BehaviorGrammarConfig(
        paths=paths,
        run_id=args.run_id,
        max_rules=args.max_rules,
        time_steps=args.time_steps,
        dt=args.dt,
        sensitivity_delta=args.sensitivity_delta,
    )
    outputs = BehaviorGrammarPipeline(config).run()
    print(f"Run directory: {outputs['run_dir']}")
    print(f"Manifest: {outputs['manifest_path']}")
    print(f"Rules: {outputs['rules_markdown']}")
    print(f"Report: {outputs['simulation_report']}")
    print(f"Figure: {outputs['figure_path']}")
    return 0


def _infer_demo_root(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.name == "behavior_grammar" and output_dir.parent.name == "results":
        return output_dir.parent.parent
    return ROOT


if __name__ == "__main__":
    raise SystemExit(main())
