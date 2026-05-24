#!/usr/bin/env python
"""阶段 3 运行入口: 反事实扰动与候选靶点排序。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.config import HyperSCAConfig
from src.pipeline.step3_perturbation import PerturbationPipeline


def _parse_targets(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    arr = [x.strip() for x in raw.split(",") if x.strip()]
    return arr or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HyperSCA Stage 3: Counterfactual Perturbation"
    )
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--step1-dir", type=str, default=None, help="Stage 1 output directory")
    parser.add_argument("--step2-dir", type=str, default=None, help="Stage 2 output directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Stage 3 output directory")
    parser.add_argument("--fig-dir", type=str, default=None, help="Stage 3 figure output directory")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["expression_ko", "hyperbolic_latent_ko", "diffusion_cf"],
        help="Counterfactual method",
    )
    parser.add_argument("--targets", type=str, default=None, help="Comma separated target genes")
    parser.add_argument("--target-top-k", type=int, default=None, help="Top K ranked targets")
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--diffusion-epochs", type=int, default=None)
    parser.add_argument("--diffusion-hidden", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = HyperSCAConfig.from_yaml(args.config) if args.config else HyperSCAConfig()

    overrides = {
        "step3_input_step1_dir": args.step1_dir,
        "step3_input_step2_dir": args.step2_dir,
        "step3_output_dir": args.output_dir,
        "step3_figures_dir": args.fig_dir,
        "step3_method": args.method,
        "step3_target_top_k": args.target_top_k,
        "step3_diffusion_steps": args.diffusion_steps,
        "step3_diffusion_epochs": args.diffusion_epochs,
        "step3_diffusion_hidden": args.diffusion_hidden,
        "device": args.device,
        "seed": args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)
    targets = _parse_targets(args.targets)
    if targets is not None:
        config.step3_target_genes = targets

    print("HyperSCA Stage 3: Counterfactual Perturbation")
    print("=" * 60)
    print(f"  step1_dir:   {config.step3_input_step1_dir}")
    print(f"  step2_dir:   {config.step3_input_step2_dir}")
    print(f"  output_dir:  {config.step3_output_dir}")
    print(f"  figures_dir: {config.step3_figures_dir}")
    print(f"  method:      {config.step3_method}")
    print(f"  targets:     {config.step3_target_genes}")
    print(f"  device:      {config.device}")
    print("=" * 60)

    pipeline = PerturbationPipeline(config)
    pipeline.run()

    print("\n[DONE] Stage 3 completed successfully.")
    print(f"  Metrics: {config.step3_output_dir}/step3_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

