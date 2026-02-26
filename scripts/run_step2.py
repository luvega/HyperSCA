#!/usr/bin/env python
"""阶段 2 运行入口: 空间约束下的因果通讯网络构建

用法:
    python scripts/run_step2.py
    python scripts/run_step2.py --granularity cluster --bootstrap-n 50
    python scripts/run_step2.py --config path.yaml --device cuda

完整参数列表见 src/pipeline/config.py::HyperSCAConfig (step2_* 前缀)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.config import HyperSCAConfig
from src.pipeline.step2_causal import CausalPipeline


def main():
    parser = argparse.ArgumentParser(
        description="HyperSCA Stage 2: Causal Communication Network"
    )
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--step1-dir", type=str, default=None,
                        help="Stage 1 output directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Stage 2 output directory")
    parser.add_argument("--granularity", type=str, default=None,
                        choices=["cluster", "single_cell"],
                        help="Analysis granularity")
    parser.add_argument("--leiden-resolution", type=float, default=None)
    parser.add_argument("--disentangle-dim", type=int, default=None)
    parser.add_argument("--disentangle-epochs", type=int, default=None)
    parser.add_argument("--hsic-alpha", type=float, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--bootstrap-threshold", type=float, default=None)
    parser.add_argument("--cmi-alpha", type=float, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--known-axes", type=str, default=None,
                        help="JSON file with known signaling axes")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = HyperSCAConfig.from_yaml(args.config)
    else:
        config = HyperSCAConfig()

    # CLI 参数覆盖
    overrides = {
        "step2_input_dir": args.step1_dir,
        "step2_output_dir": args.output_dir,
        "step2_granularity": args.granularity,
        "step2_leiden_resolution": args.leiden_resolution,
        "step2_disentangle_dim": args.disentangle_dim,
        "step2_disentangle_epochs": args.disentangle_epochs,
        "step2_hsic_alpha": args.hsic_alpha,
        "step2_bootstrap_n": args.bootstrap_n,
        "step2_bootstrap_threshold": args.bootstrap_threshold,
        "step2_cmi_alpha": args.cmi_alpha,
        "step2_max_cells": args.max_cells,
        "step2_known_axes_file": args.known_axes,
        "device": args.device,
        "seed": args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)

    print("HyperSCA Stage 2: Causal Communication Network")
    print("=" * 60)
    print(f"  step1_dir:    {config.step2_input_dir}")
    print(f"  output_dir:   {config.step2_output_dir}")
    print(f"  granularity:  {config.step2_granularity}")
    print(f"  z_dim:        {config.step2_disentangle_dim}")
    print(f"  epochs:       {config.step2_disentangle_epochs}")
    print(f"  hsic_alpha:   {config.step2_hsic_alpha}")
    print(f"  bootstrap_n:  {config.step2_bootstrap_n}")
    print(f"  threshold:    {config.step2_bootstrap_threshold}")
    print(f"  device:       {config.device}")
    print("=" * 60)

    pipeline = CausalPipeline(config)
    results = pipeline.run()

    print("\n[DONE] Stage 2 completed successfully.")
    print(f"  Output: {config.step2_output_dir}")
    print(f"  Metrics: {config.step2_output_dir}/step2_metrics.json")
    print(f"  Report: {config.step2_output_dir}/interpretation_step2.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
