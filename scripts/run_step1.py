#!/usr/bin/env python
"""阶段 1 运行入口: 双曲流形嵌入

用法:
    python scripts/run_step1.py [--config path.yaml]
    python scripts/run_step1.py --epochs 100 --latent-dim 16 --device cuda

完整参数列表见 src/pipeline/config.py::HyperSCAConfig
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
from src.pipeline.step1_embedding import EmbeddingPipeline


def main():
    parser = argparse.ArgumentParser(description="HyperSCA Stage 1: Hyperbolic Embedding")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--modality", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--spatial-k", type=int, default=None)
    parser.add_argument("--use-topola", dest="use_topola", action="store_true")
    parser.add_argument("--no-topola", dest="use_topola", action="store_false")
    parser.set_defaults(use_topola=None)
    parser.add_argument("--topola-max-nodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = HyperSCAConfig.from_yaml(args.config)
    else:
        config = HyperSCAConfig()

    # CLI 参数覆盖
    overrides = {
        "data_dir": args.data_dir,
        "modality": args.modality,
        "hvae_epochs": args.epochs,
        "hvae_pretrain_epochs": args.pretrain_epochs,
        "hvae_latent_dim": args.latent_dim,
        "n_top_genes": args.n_top_genes,
        "spatial_k": args.spatial_k,
        "use_topola": args.use_topola,
        "topola_max_nodes": args.topola_max_nodes,
        "hvae_lr": args.lr,
        "hvae_beta": args.beta,
        "hvae_gamma": args.gamma,
        "device": args.device,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)

    print("HyperSCA Stage 1: Hyperbolic Embedding")
    print("=" * 60)
    print(f"  data_dir: {config.data_dir}")
    print(f"  modality: {config.modality}")
    print(f"  latent_dim: {config.hvae_latent_dim}")
    print(f"  epochs: {config.hvae_pretrain_epochs} (pretrain) + {config.hvae_epochs} (train)")
    print(f"  device: {config.device}")
    print(f"  output: {config.output_dir}")
    print("=" * 60)

    pipeline = EmbeddingPipeline(config)
    results = pipeline.run()

    print("\n[DONE] Stage 1 completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
