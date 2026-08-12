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
    parser = argparse.ArgumentParser(
        description=(
            "从表达、空间位置和补充机制证据中整理候选靶点。默认排序只由直接证据决定，"
            "并保存分析记录清单（manifest），方便复查输入和输出。"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "discovery" / "target_discovery",
        help="保存分析结果的根目录。默认写入 results/discovery/target_discovery。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="本次分析的名称；不填写时自动生成。",
    )
    parser.add_argument(
        "--max-perturb",
        type=int,
        default=50,
        help="最多评估多少个候选干预对象，默认 50。",
    )
    parser.add_argument(
        "--geometry-k",
        type=int,
        default=4,
        help="描述局部空间关系时，每个位置采用的近邻数量，默认 4。",
    )
    parser.add_argument(
        "--geometry-blend",
        type=float,
        default=0.30,
        help="局部与整体空间信息的合并比例，默认 0.30。",
    )
    parser.add_argument(
        "--platform",
        choices=["cosmx", "visium", "visiumhd", "all"],
        default="all",
        help="选择分析的空间测量平台；all 表示分析全部已配置平台。",
    )
    parser.add_argument(
        "--hierarchy-levels",
        type=int,
        default=3,
        help="空间结构分层的层数，默认 3。",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="只生成数据和报告，不绘制图形。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="执行计算所用设备，默认 cuda；没有可用显卡时可设为 cpu。",
    )
    parser.add_argument(
        "--score-profile",
        choices=["evidence_gated", "legacy_full"],
        default="evidence_gated",
        help=(
            "候选排序规则。evidence_gated 只用直接证据排序；"
            "legacy_full 仅用于复现旧结果。"
        ),
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
    print(f"本次分析目录：{outputs['run_dir']}")
    print(f"分析记录清单：{outputs['manifest_path']}")
    if "target_discovery_report" in outputs:
        print(f"可阅读报告：{outputs['target_discovery_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
