#!/usr/bin/env python3
"""Audit whether a registered spatial-perturbation predictor can run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "核查空间扰动预测方法是否已有预注册且可执行的接入层（adapter），"
            "并发布不读取实验结局的证据边界记录。"
        ),
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument(
        "--output",
        dest="output_dir",
        metavar="OUTPUT_DIR",
        type=Path,
        required=True,
        help="写入四个终止能力证据文件的输出目录",
    )
    return parser


def main() -> int:
    parser = _arguments()
    options = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    repository_root_text = str(repository_root)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)

    from src.evaluation.safe_declaration_reader import read_safe_declaration
    from src.evaluation.spatial_perturbation_predictor_contract import (
        BridgePredictorContractError,
        audit_bridge_predictor_capability,
    )
    from src.evaluation.spatial_perturbation_runner import (
        publish_spatial_perturbation_run,
    )

    try:
        capability = audit_bridge_predictor_capability(
            read_safe_declaration(options.registry, label="预测方法注册表"),
            read_safe_declaration(options.protocol, label="研究方案"),
            method_id=options.method_id,
        )
        publish_spatial_perturbation_run(capability, output_dir=options.output_dir)
    except (BridgePredictorContractError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
