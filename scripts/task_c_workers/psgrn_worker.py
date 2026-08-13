#!/usr/bin/env python3
"""Run the fixed PSGRN source on an already permitted cell subset."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Sequence

WORKER_DIRECTORY = str(Path(__file__).resolve().parent)
if WORKER_DIRECTORY not in sys.path:
    sys.path.insert(0, WORKER_DIRECTORY)

from causalbench_worker import (
    TRAINING_INFORMATION,
    WorkerContractError,
    load_fixed_npz,
    ranked_edges,
    select_training_cells,
    validate_output_destination,
    write_ranked_csv,
)


EXPECTED_PSGRN_COMMIT = "74aa640f7c472b23a69811f6795bb17678efd344"
EXPECTED_PSGRN_REPOSITORY = "https://github.com/GuanLab/PSGRN.git"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在隔离环境中运行固定版本 PSGRN，分析 HyperSCA 已允许的单细胞表达与干预信息。"
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        required=True,
        help="含表达矩阵、干预标签和固定基因名的单细胞 NPZ 文件。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="新建的前 1,000 条基因关系结果；已有文件不会被覆盖。",
    )
    parser.add_argument(
        "--psgrn-source",
        type=Path,
        required=True,
        help="固定提交且没有本地改动的 PSGRN 官方源码目录。",
    )
    parser.add_argument(
        "--training-information",
        choices=TRAINING_INFORMATION,
        required=True,
        help="方法获准读取的是仅未干预细胞，还是公开的部分干预细胞。",
    )
    parser.add_argument("--seed", type=int, required=True, help="固定随机种子。")
    parser.add_argument(
        "--output-semantics",
        choices=("official_return_order",),
        required=True,
        help="按官方返回顺序赋予递减正分，不改变关系选择。",
    )
    return parser


def _git(source: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("PSGRN source cannot be verified as a fixed Git checkout") from exc
    return completed.stdout.strip()


def validate_psgrn_source(
    source_path: Path, expected_commit: str | None = None
) -> Path:
    """Refuse modified, redirected, or differently versioned external code."""
    if expected_commit is None:
        expected_commit = EXPECTED_PSGRN_COMMIT
    source = Path(source_path).expanduser()
    try:
        metadata = os.lstat(source)
    except OSError as exc:
        raise SystemExit("PSGRN source directory does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit("PSGRN source directory must not be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("PSGRN source must be a directory")
    source = source.resolve(strict=True)
    top_level = Path(_git(source, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != source:
        raise SystemExit("PSGRN source must be the verified repository root")
    if _git(source, "rev-parse", "HEAD") != expected_commit:
        raise SystemExit("PSGRN source revision does not match the registered commit")
    if _git(source, "remote", "get-url", "origin") != EXPECTED_PSGRN_REPOSITORY:
        raise SystemExit("PSGRN source repository does not match the registered source")
    if _git(source, "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("PSGRN source must have a clean working tree")
    entrypoint = source / "src/main.py"
    try:
        entrypoint_metadata = os.lstat(entrypoint)
    except OSError as exc:
        raise SystemExit("PSGRN fixed entrypoint is missing") from exc
    if stat.S_ISLNK(entrypoint_metadata.st_mode) or not stat.S_ISREG(
        entrypoint_metadata.st_mode
    ):
        raise SystemExit("PSGRN fixed entrypoint must be a regular file, not a symbolic link")
    return entrypoint


def _load_custom(entrypoint: Path) -> tuple[object, object]:
    spec = importlib.util.spec_from_file_location("hypersca_fixed_psgrn", entrypoint)
    if spec is None or spec.loader is None:
        raise SystemExit("PSGRN fixed entrypoint could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        custom = module.Custom
        training_regime = module.TrainingRegime
    except Exception as exc:
        raise SystemExit("PSGRN fixed entrypoint is incompatible with this environment") from exc
    return custom, training_regime


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_output_destination(args.output_csv)
        entrypoint = validate_psgrn_source(args.psgrn_source)
        expression, interventions, genes = load_fixed_npz(args.input_npz)
        expression, interventions = select_training_cells(
            expression, interventions, args.training_information
        )
        custom_class, training_regime = _load_custom(entrypoint)
        regime = (
            training_regime.Observational
            if args.training_information == "observational"
            else training_regime.PartialIntervational
        )
        returned = custom_class()(
            expression,
            interventions.tolist(),
            list(genes),
            regime,
            args.seed,
        )
        rows = ranked_edges(returned, genes, limit=1_000)
        write_ranked_csv(args.output_csv, rows)
    except WorkerContractError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
