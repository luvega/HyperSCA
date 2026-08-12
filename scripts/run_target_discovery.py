"""CLI entrypoint for the HyperSCA target discovery pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.discovery.target_discovery.config import DiscoveryPaths, GeometryModeConfig, TargetDiscoveryConfig
from src.discovery.target_discovery.pipeline import TargetDiscoveryPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperSCA Target Discovery")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "discovery" / "target_discovery")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-perturb", type=int, default=50)
    parser.add_argument("--geometry-k", type=int, default=4)
    parser.add_argument("--geometry-blend", type=float, default=0.30)
    parser.add_argument("--platform", choices=["cosmx", "visium", "visiumhd", "all"], default="all")
    parser.add_argument("--hierarchy-levels", type=int, default=3)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--score-profile",
        choices=["evidence_gated", "legacy_full"],
        default="evidence_gated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = DiscoveryPaths.default(root=ROOT, output_base=args.output_dir)
    config = TargetDiscoveryConfig(
        paths=paths,
        geometry=GeometryModeConfig(geometry_k=args.geometry_k, geometry_blend=args.geometry_blend),
        max_perturb=args.max_perturb,
        platform=args.platform,
        hierarchy_levels=args.hierarchy_levels,
        run_id=args.run_id,
        device=args.device,
        skip_figures=args.skip_figures,
        score_profile=args.score_profile,
    )
    outputs = TargetDiscoveryPipeline(config).run()
    print(f"Run directory: {outputs['run_dir']}")
    print(f"Manifest: {outputs['manifest_path']}")
    if "target_discovery_report" in outputs:
        print(f"Report: {outputs['target_discovery_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
