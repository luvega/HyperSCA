#!/usr/bin/env python3
"""用同一份证据规则运行一种 Task C 比较方法。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_method_run import TaskCMethodRunError, run_task_c_method


class _PlainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法运行 Task C 方法：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _PlainArgumentParser(
        description=(
            "运行一种已登记的基因关系方法，保留原始结果，并把结果补齐到共同关系范围。"
        )
    )
    parser.add_argument("--method-id", required=True, help="方法清单中的固定名称。")
    parser.add_argument("--input-npz", type=Path, help="该方法获准读取的单细胞文件。")
    parser.add_argument("--output-dir", type=Path, required=True, help="新结果目录。")
    parser.add_argument("--seed", required=True, help="非负整数随机种子。")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/task_c_methods_v1.json",
        help="已审核并固定版本的方法清单。",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="bootstrap_task_c_methods.py 准备的官方代码和环境记录目录。",
    )
    parser.add_argument(
        "--data-status",
        choices=("external_benchmark", "synthetic_smoke"),
        help="明确区分正式公开数据和仅用于检查流程的合成数据。",
    )
    parser.add_argument("--context-id", help="该输入代表的细胞背景。")
    parser.add_argument("--min-cells", default="2", help="每组至少需要的细胞数。")
    parser.add_argument("--public-manifest", type=Path, help="正式数据的公开文件清单。")
    parser.add_argument(
        "--derived-input-manifest",
        type=Path,
        help="跨环境派生学习文件的来源、变换和输出校验记录。",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="HyperSCA-C 公开训练文件，格式为 k562=path 或 rpe1=path。",
    )
    parser.add_argument("--hypersca-config", type=Path, help="HyperSCA-C 固定设置。")
    parser.add_argument("--gene-list", type=Path, help="HyperSCA-C 固定基因清单。")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", default="86400")
    parser.add_argument(
        "--trial-parameters",
        type=Path,
        help=(
            "运行前固定的候选编号和参数；省略时记录为无需调节的固定设置。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        try:
            seed = int(args.seed)
            min_cells = int(args.min_cells)
            timeout_seconds = float(args.timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise TaskCMethodRunError(
                "seed、min-cells 和 timeout-seconds 必须是有效数字"
            ) from exc
        status = run_task_c_method(
            method_id=args.method_id,
            input_npz=args.input_npz,
            output_dir=args.output_dir,
            seed=seed,
            registry_path=args.registry,
            asset_root=args.asset_root,
            data_status=args.data_status,
            context_id=args.context_id,
            min_cells=min_cells,
            public_manifest_path=args.public_manifest,
            derived_input_manifest_path=args.derived_input_manifest,
            context_values=args.context,
            hypersca_config_path=args.hypersca_config,
            gene_list_path=args.gene_list,
            device=args.device,
            timeout_seconds=timeout_seconds,
            trial_parameters_path=args.trial_parameters,
            project_root=ROOT,
        )
    except (TaskCMethodRunError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(status, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if str(status["status"]).startswith("completed_") or status["status"] == "official_assets_unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
