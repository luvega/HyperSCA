#!/usr/bin/env python3
"""Run one fixed CausalBench method on an already permitted cell subset."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import io
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence
import unicodedata
import zipfile


MODEL_NAMES = (
    "random1000",
    "grnboost",
    "pc",
    "ges",
    "gies",
    "gsp",
    "igsp",
    "notears-lin-sparse",
    "DCDI-G",
    "DCDI-DSF",
    "DCDFG-LIN",
    "DCDFG-MLP",
    "sortnregress",
)
TRAINING_INFORMATION = ("observational", "partial_interventional")
MAXIMUM_INPUT_BYTES = 512 * 1024 * 1024
MAXIMUM_ARCHIVE_BYTES = 512 * 1024 * 1024
MAXIMUM_CELLS = 1_000_000
MAXIMUM_GENES = 1_000
CONTROL_LABEL = "non-targeting"


class WorkerContractError(ValueError):
    """The allowed input or an external method result breaks the fixed contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在隔离环境中运行一种单细胞基因关系方法；输入必须是 HyperSCA "
            "已经允许使用的细胞和基因。"
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
        help="新建的基因关系结果；已有文件不会被覆盖。",
    )
    parser.add_argument("--model-name", choices=MODEL_NAMES, required=True)
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


def _regular_snapshot(path: Path) -> bytes:
    source = Path(path).expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise WorkerContractError(
            "input NPZ must be an existing regular file, not a symbolic link"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkerContractError("input NPZ must be a regular file")
        if metadata.st_size <= 0:
            raise WorkerContractError("input NPZ must not be empty")
        if metadata.st_size > MAXIMUM_INPUT_BYTES:
            raise WorkerContractError("input NPZ is unusually large")
        chunks: list[bytes] = []
        remaining = MAXIMUM_INPUT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or len(payload) > MAXIMUM_INPUT_BYTES:
            raise WorkerContractError("input NPZ changed while it was being read")
        return payload
    finally:
        os.close(descriptor)


def _canonical_text_vector(values: Any, name: str) -> tuple[str, ...]:
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 1:
        raise WorkerContractError(f"{name} must be one-dimensional")
    if array.dtype.kind not in {"U", "S"}:
        raise WorkerContractError(f"{name} must contain strings without object data")
    if array.dtype.kind == "S":
        try:
            items = tuple(value.decode("utf-8", errors="strict") for value in array.tolist())
        except UnicodeDecodeError as exc:
            raise WorkerContractError(f"{name} must contain valid UTF-8") from exc
    else:
        items = tuple(array.tolist())
    if any(not item for item in items):
        raise WorkerContractError(f"{name} must contain non-empty strings")
    if any(item != item.strip() for item in items):
        raise WorkerContractError(f"{name} must not contain surrounding whitespace")
    if any(not unicodedata.is_normalized("NFC", item) for item in items):
        raise WorkerContractError(f"{name} must use NFC text")
    return items


def load_fixed_npz(path: Path) -> tuple[Any, Any, tuple[str, ...]]:
    """Read exactly the three arrays used by the frozen Task C data split."""
    import numpy as np

    payload = _regular_snapshot(path)
    buffer = io.BytesIO(payload)
    try:
        with zipfile.ZipFile(buffer) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > MAXIMUM_ARCHIVE_BYTES:
                raise WorkerContractError("input NPZ has an invalid or unusually large archive")
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            expected = {"expression_matrix", "interventions", "var_names"}
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise WorkerContractError(
                    "input NPZ must contain exactly expression_matrix, interventions, and var_names"
                )
            expression = np.asarray(archive["expression_matrix"])
            interventions_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
    except WorkerContractError:
        raise
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise WorkerContractError("input NPZ could not be read safely") from exc

    if expression.ndim != 2:
        raise WorkerContractError("expression_matrix must be two-dimensional")
    rows, columns = expression.shape
    if rows == 0 or rows > MAXIMUM_CELLS:
        raise WorkerContractError("expression_matrix has an invalid number of rows")
    if columns < 2 or columns > MAXIMUM_GENES:
        raise WorkerContractError("expression_matrix has an invalid number of gene columns")
    if expression.dtype.kind not in {"i", "u", "f"}:
        raise WorkerContractError("expression_matrix must contain numeric values")
    if not np.all(np.isfinite(expression)):
        raise WorkerContractError("expression_matrix values must be finite")
    interventions = _canonical_text_vector(interventions_raw, "intervention labels")
    genes = _canonical_text_vector(genes_raw, "gene names")
    if len(interventions) != rows:
        raise WorkerContractError("expression rows must equal intervention labels")
    if len(genes) != columns:
        raise WorkerContractError("expression columns must equal gene names")
    if len(set(genes)) != len(genes):
        raise WorkerContractError("gene names must be unique")
    if CONTROL_LABEL not in interventions:
        raise WorkerContractError("at least one non-targeting cell is required")

    safe_expression = np.array(expression, copy=True)
    safe_interventions = np.asarray(interventions, dtype=str)
    safe_expression.setflags(write=False)
    safe_interventions.setflags(write=False)
    return safe_expression, safe_interventions, genes


def select_training_cells(
    expression: Any,
    interventions: Any,
    training_information: str,
) -> tuple[Any, Any]:
    import numpy as np

    if training_information == "observational":
        selected = np.flatnonzero(interventions == CONTROL_LABEL)
        expression = np.array(expression[selected], copy=True)
        interventions = np.array(interventions[selected], copy=True)
    elif training_information == "partial_interventional":
        expression = np.array(expression, copy=True)
        interventions = np.array(interventions, copy=True)
    else:  # The argument parser should make this unreachable.
        raise WorkerContractError("training information is not registered")
    expression.setflags(write=False)
    interventions.setflags(write=False)
    return expression, interventions


def ranked_edges(raw: object, genes: Sequence[str], *, limit: int | None = None) -> list[tuple[str, str, float]]:
    """Validate a provably ordered edge list and attach decreasing positive scores."""
    if isinstance(raw, (set, frozenset, Mapping)) or not isinstance(raw, (list, tuple)):
        raise WorkerContractError(
            "failed_invalid_output: official return order is not provable"
        )
    chosen = raw if limit is None else raw[:limit]
    fixed_genes = set(genes)
    endpoints: list[tuple[str, str]] = []
    for relation in chosen:
        if not isinstance(relation, (list, tuple)) or len(relation) != 2:
            raise WorkerContractError(
                "failed_invalid_output: every returned relation must have two endpoints"
            )
        source, target = relation
        if type(source) is not str or type(target) is not str:
            raise WorkerContractError(
                "failed_invalid_output: relation endpoints must be gene names"
            )
        if (
            not source
            or not target
            or source != source.strip()
            or target != target.strip()
            or not unicodedata.is_normalized("NFC", source)
            or not unicodedata.is_normalized("NFC", target)
        ):
            raise WorkerContractError(
                "failed_invalid_output: relation endpoints must be canonical gene names"
            )
        if source not in fixed_genes or target not in fixed_genes:
            raise WorkerContractError(
                "failed_invalid_output: relation endpoint is outside the fixed gene set"
            )
        endpoints.append((source, target))
    count = len(endpoints)
    return [
        (source, target, float(count - index))
        for index, (source, target) in enumerate(endpoints)
    ]


def _ensure_safe_parent(destination: Path) -> None:
    parent = destination.parent
    missing: list[Path] = []
    cursor = parent
    while not os.path.lexists(cursor):
        missing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    if os.path.lexists(cursor):
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WorkerContractError("output parent must be a real directory")
    for directory in reversed(missing):
        directory.mkdir()
    cursor = parent
    while cursor != cursor.parent:
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WorkerContractError("output parent must not use symbolic links")
        cursor = cursor.parent


def write_ranked_csv(path: Path, rows: Sequence[tuple[str, str, float]]) -> None:
    destination = Path(path).expanduser()
    validate_output_destination(destination)
    temporary: str | None = None
    linked = False
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("source", "target", "score"))
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        os.unlink(temporary)
        temporary = None
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise WorkerContractError("output CSV must not already exist") from exc
    except WorkerContractError:
        raise
    except OSError as exc:
        if linked:
            try:
                os.unlink(destination)
            except OSError:
                pass
        raise WorkerContractError("output CSV could not be written atomically") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def validate_output_destination(path: Path) -> None:
    """Reject overwrite attempts before an expensive external method is run."""
    destination = Path(path).expanduser()
    if os.path.lexists(destination):
        raise WorkerContractError("output CSV must not already exist")
    _ensure_safe_parent(destination)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_output_destination(args.output_csv)
        # External imports deliberately occur only after argument parsing, so
        # ``--help`` remains available before the dedicated environment exists.
        from causalscbench.models.arboreto_baselines import GRNBoost
        from causalscbench.models.causallearn_models import GES, PC
        from causalscbench.models.dcdi_models import DCDI, DCDFG
        from causalscbench.models.gies import GIES
        from causalscbench.models.notears import NotearsLin
        from causalscbench.models.random_network import RandomWithSize
        from causalscbench.models.sparsest_permutations import (
            GreedySparsestPermutation,
            InterventionalGreedySparsestPermutation,
        )
        from causalscbench.models.training_regimes import TrainingRegime
        from causalscbench.models.varsortability import Sortnregress

        model_builders = {
            "random1000": lambda: RandomWithSize(1000),
            "grnboost": GRNBoost,
            "pc": lambda: PC(missing_value=False),
            "ges": GES,
            "gies": GIES,
            "gsp": GreedySparsestPermutation,
            "igsp": InterventionalGreedySparsestPermutation,
            "notears-lin-sparse": lambda: NotearsLin(lambda1=0.001),
            "DCDI-G": lambda: DCDI("DCDI-G"),
            "DCDI-DSF": lambda: DCDI("DCDI-DSF"),
            "DCDFG-LIN": lambda: DCDFG("linear"),
            "DCDFG-MLP": lambda: DCDFG("mlplr"),
            "sortnregress": Sortnregress,
        }
        expression, interventions, genes = load_fixed_npz(args.input_npz)
        expression, interventions = select_training_cells(
            expression, interventions, args.training_information
        )
        regime = (
            TrainingRegime.Observational
            if args.training_information == "observational"
            else TrainingRegime.PartialIntervational
        )
        model = model_builders[args.model_name]()
        returned = model(
            expression,
            interventions.tolist(),
            list(genes),
            regime,
            args.seed,
        )
        rows = ranked_edges(returned, genes)
        write_ranked_csv(args.output_csv, rows)
    except WorkerContractError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
