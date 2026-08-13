#!/usr/bin/env python3
"""Run the fixed PSGRN source on an already permitted cell subset."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
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


@dataclass(frozen=True)
class VerifiedPSGRNSource:
    """Immutable source bytes read from both Git and the opened work-tree file."""

    entrypoint: Path
    source_bytes: bytes


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


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _git_run(source: Path, *arguments: str, text: bool) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=text,
            env=_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("PSGRN source cannot be verified as a fixed Git checkout") from exc


def _git(source: Path, *arguments: str) -> str:
    return _git_run(source, *arguments, text=True).stdout.strip()


def _git_bytes(source: Path, *arguments: str) -> bytes:
    return _git_run(source, *arguments, text=False).stdout


def _assert_no_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SystemExit("PSGRN source directory does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit("PSGRN source path must not use a symbolic link")


def _reject_git_history_rewrites(source: Path) -> None:
    if _git(source, "for-each-ref", "--format=%(refname)", "refs/replace"):
        raise SystemExit("PSGRN source must not contain Git replace refs")
    grafts_text = _git(source, "rev-parse", "--git-path", "info/grafts")
    grafts = Path(grafts_text)
    if not grafts.is_absolute():
        grafts = source / grafts
    if os.path.lexists(grafts):
        raise SystemExit("PSGRN source must not contain Git grafts")


def validate_psgrn_source(
    source_path: Path, expected_commit: str | None = None
) -> VerifiedPSGRNSource:
    """Refuse modified, redirected, or differently versioned external code."""
    if expected_commit is None:
        expected_commit = EXPECTED_PSGRN_COMMIT
    source = Path(os.path.abspath(os.fspath(Path(source_path).expanduser())))
    try:
        metadata = os.lstat(source)
    except OSError as exc:
        raise SystemExit("PSGRN source directory does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit("PSGRN source directory must not be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("PSGRN source must be a directory")
    _assert_no_symlink_components(source)
    source = source.resolve(strict=True)
    _reject_git_history_rewrites(source)
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
    committed_bytes = _git_bytes(source, "show", f"{expected_commit}:src/main.py")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(entrypoint, flags)
    except OSError as exc:
        raise SystemExit("PSGRN fixed entrypoint could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("PSGRN fixed entrypoint must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    working_bytes = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(working_bytes) != before.st_size
    ):
        raise SystemExit("PSGRN fixed entrypoint changed while it was being read")
    if working_bytes != committed_bytes:
        raise SystemExit("PSGRN fixed entrypoint does not match the registered commit")
    return VerifiedPSGRNSource(entrypoint=entrypoint, source_bytes=working_bytes)


def _load_custom(verified: VerifiedPSGRNSource) -> tuple[object, object]:
    module_name = "hypersca_fixed_psgrn"
    module = types.ModuleType(module_name)
    module.__file__ = str(verified.entrypoint)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(
            verified.source_bytes,
            str(verified.entrypoint),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        custom = module.Custom
        training_regime = module.TrainingRegime
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SystemExit("PSGRN fixed entrypoint is incompatible with this environment") from exc
    if sys.modules.get(module_name) is not module:
        sys.modules.pop(module_name, None)
        raise SystemExit("PSGRN fixed entrypoint changed its protected module binding")
    return custom, training_regime


def validate_psgrn_training_information(training_information: str) -> None:
    if training_information != "partial_interventional":
        raise WorkerContractError(
            "requested training information does not match the registered data boundary for PSGRN"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_psgrn_training_information(args.training_information)
        validate_output_destination(args.output_csv)
        verified = validate_psgrn_source(args.psgrn_source)
        expression, interventions, genes = load_fixed_npz(args.input_npz)
        expression, interventions = select_training_cells(
            expression, interventions, args.training_information
        )
        custom_class, training_regime = _load_custom(verified)
        try:
            regime = training_regime.PartialIntervational
            returned = custom_class()(
                expression,
                interventions.tolist(),
                list(genes),
                regime,
                args.seed,
            )
        finally:
            sys.modules.pop("hypersca_fixed_psgrn", None)
        rows = ranked_edges(returned, genes, limit=1_000)
        write_ranked_csv(args.output_csv, rows)
    except WorkerContractError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
