#!/usr/bin/env python3
"""Score one complete relation table on sealed intervention-response cells."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import site
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence
import unicodedata
import zipfile


EXPECTED_CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
MAXIMUM_PREDICTION_BYTES = 64 * 1024 * 1024
MAXIMUM_HELDOUT_BYTES = 512 * 1024 * 1024
MAXIMUM_ARCHIVE_BYTES = 512 * 1024 * 1024
MAXIMUM_CELLS = 1_000_000
MAXIMUM_GENES = 256
MAXIMUM_TEXT_ITEM_BYTES = 4 * 1024
MAXIMUM_TOTAL_TEXT_BYTES = 64 * 1024 * 1024
MAXIMUM_JSON_DEPTH = 16
MAXIMUM_JSON_ITEMS = 200_000
MAXIMUM_ARRAY_ITEMS = 100_000
MAXIMUM_SEED = 2**32 - 1
CONTROL_LABEL = "non-targeting"
EXCLUDED_LABEL = "excluded"


class ScoringContractError(ValueError):
    """The sealed input or scoring boundary is unsafe or scientifically invalid."""


class InvalidMetricOutput(ScoringContractError):
    """The official evaluator returned a value that cannot be recorded exactly."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在独立评分步骤读取封存干预细胞，并计算 CausalBench 官方补充指标；"
            "比较方法本身不会收到封存数据路径。"
        )
    )
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        required=True,
        help="一个方法返回的完整有向基因关系表。",
    )
    parser.add_argument(
        "--heldout-npz",
        type=Path,
        required=True,
        help="只供独立评分步骤读取的封存三数组 NPZ。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="新建的官方补充评分记录；已有文件不会被覆盖。",
    )
    parser.add_argument(
        "--seed",
        required=True,
        help="0 至 2^32-1 的固定随机种子。",
    )
    parser.add_argument(
        "--causalbench-source",
        type=Path,
        required=True,
        help="固定提交且没有本地改动的 CausalBench 官方源码目录。",
    )
    return parser


def _load_causalbench_boundary_module() -> Any:
    path = Path(__file__).with_name("causalbench_worker.py")
    spec = importlib.util.spec_from_file_location(
        "_task_c_verified_causalbench_boundary", path
    )
    if spec is None or spec.loader is None:
        raise ScoringContractError("fixed CausalBench source verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _clean_python_environment() -> Iterator[None]:
    """Ignore caller-supplied Python paths while using the fixed evaluator."""

    previous_path = list(sys.path)
    names = ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONNOUSERSITE")
    previous_environment = {name: os.environ.get(name) for name in names}

    def normalized(entry: str) -> Path:
        return Path(entry or os.getcwd()).expanduser().resolve(strict=False)

    injected = {
        normalized(entry)
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    }
    user_site = site.getusersitepackages()
    user_sites = (
        {normalized(user_site)}
        if isinstance(user_site, str)
        else {normalized(entry) for entry in user_site}
    )
    sys.path[:] = [
        entry
        for entry in previous_path
        if entry
        and Path(entry).is_absolute()
        and normalized(entry) not in injected
        and normalized(entry) not in user_sites
    ]
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        os.environ.pop(name, None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        yield
    finally:
        sys.path[:] = previous_path
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_nlink),
    )


def _snapshot(path: Path, label: str, maximum_bytes: int) -> tuple[bytes, tuple[int, int]]:
    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ScoringContractError(
            f"{label} must be an existing regular file, not a symbolic link"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ScoringContractError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise ScoringContractError(f"{label} must not be a hard link")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ScoringContractError(f"{label} has an invalid file size")
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    except ScoringContractError:
        raise
    except OSError as exc:
        raise ScoringContractError(f"{label} could not be read safely") from exc
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise ScoringContractError(f"{label} changed while being read") from exc
    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(current)
    ):
        raise ScoringContractError(f"{label} changed while being read")
    return bytes(payload), (int(after.st_dev), int(after.st_ino))


def _source_file_snapshot(path: Path) -> tuple[bytes, tuple[int, ...]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ScoringContractError("fixed CausalBench source file changed") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > MAXIMUM_PREDICTION_BYTES
        ):
            raise ScoringContractError("fixed CausalBench source file changed")
        payload = bytearray()
        while len(payload) <= MAXIMUM_PREDICTION_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    MAXIMUM_PREDICTION_BYTES + 1 - len(payload),
                ),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    except ScoringContractError:
        raise
    except OSError as exc:
        raise ScoringContractError("fixed CausalBench source file changed") from exc
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise ScoringContractError("fixed CausalBench source file changed") from exc
    if (
        len(payload) != before.st_size
        or len(payload) > MAXIMUM_PREDICTION_BYTES
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(current)
    ):
        raise ScoringContractError("fixed CausalBench source file changed")
    return bytes(payload), _identity(after)


def _expected_commit(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScoringContractError(
            "expected CausalBench commit must be a full lowercase SHA-1"
        )
    return value


def _validate_expected_checkout(
    source: Path, boundary: Any, expected_commit: str
) -> Path:
    expected = _expected_commit(expected_commit)
    try:
        return boundary.validate_causalbench_source(
            source, expected_commit=expected
        )
    except SystemExit as exc:
        raise ScoringContractError(str(exc)) from exc


def _committed_blob(
    source: Path,
    relative: str,
    boundary: Any,
    *,
    expected_commit: str,
) -> bytes:
    expected = _expected_commit(expected_commit)
    environment = boundary._git_environment()
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(source),
                "show",
                f"{expected}:{relative}",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScoringContractError(
            "fixed CausalBench committed source could not be read"
        ) from exc
    return bytes(completed.stdout)


def freeze_causalbench_python_source(
    source: Path,
    boundary: Any,
    *,
    expected_commit: str,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Bind every tracked CausalBench Python file to its committed bytes."""

    expected = _expected_commit(expected_commit)
    source = _validate_expected_checkout(source, boundary, expected)
    try:
        tracked = boundary._git(
            source,
            "ls-tree",
            "-r",
            "--name-only",
            expected,
            "--",
            "causalscbench",
        ).splitlines()
    except SystemExit as exc:
        raise ScoringContractError(str(exc)) from exc
    relatives = sorted(relative for relative in tracked if relative.endswith(".py"))
    evaluation_relative = "causalscbench/evaluation/statistical_evaluation.py"
    if evaluation_relative not in relatives:
        raise ScoringContractError("fixed CausalBench evaluation module is missing")
    frozen: dict[str, tuple[str, tuple[int, ...]]] = {}
    for relative in relatives:
        lexical = Path(relative)
        if lexical.is_absolute() or ".." in lexical.parts:
            raise ScoringContractError("fixed CausalBench source inventory is unsafe")
        path = source / lexical
        payload, identity = _source_file_snapshot(path)
        committed = _committed_blob(
            source,
            relative,
            boundary,
            expected_commit=expected,
        )
        if payload != committed:
            raise ScoringContractError("fixed CausalBench source file changed")
        frozen[relative] = (hashlib.sha256(committed).hexdigest(), identity)
    return frozen


def _make_read_only_tree(root: Path) -> None:
    directories = [root]
    directories.extend(
        path for path in root.rglob("*") if path.is_dir()
    )
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        os.chmod(directory, 0o500, follow_symlinks=False)


def _remove_private_snapshot(root: Path) -> None:
    if not root.exists():
        return
    for directory, subdirectories, _files in os.walk(root, topdown=True):
        os.chmod(directory, 0o700, follow_symlinks=False)
        for name in subdirectories:
            child = Path(directory) / name
            if child.is_symlink():
                raise ScoringContractError(
                    "private CausalBench snapshot unexpectedly contains a symbolic link"
                )
    shutil.rmtree(root)


@contextmanager
def fixed_causalbench_source_snapshot(
    source: Path,
    boundary: Any,
    *,
    expected_commit: str,
) -> Iterator[tuple[Path, dict[str, tuple[str, tuple[int, ...]]]]]:
    """Loadable private copy made only from verified committed Python bytes."""

    expected = _expected_commit(expected_commit)
    source = _validate_expected_checkout(source, boundary, expected)
    live_frozen = freeze_causalbench_python_source(
        source,
        boundary,
        expected_commit=expected,
    )
    snapshot = Path(tempfile.mkdtemp(prefix="hypersca-causalbench-evaluation-"))
    frozen: dict[str, tuple[str, tuple[int, ...]]] = {}
    try:
        for relative, (expected_hash, _live_identity) in live_frozen.items():
            payload = _committed_blob(
                source,
                relative,
                boundary,
                expected_commit=expected,
            )
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ScoringContractError(
                    "fixed CausalBench committed source changed"
                )
            destination = snapshot / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                written = 0
                while written < len(payload):
                    written += os.write(descriptor, payload[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(destination, 0o400, follow_symlinks=False)
            copied, identity = _source_file_snapshot(destination)
            if copied != payload:
                raise ScoringContractError(
                    "private CausalBench snapshot content changed"
                )
            frozen[relative] = (expected_hash, identity)
        _make_read_only_tree(snapshot)
        verify_causalbench_python_source(snapshot, frozen)
        _validate_expected_checkout(source, boundary, expected)
        yield snapshot, frozen
    finally:
        _remove_private_snapshot(snapshot)


def verify_causalbench_python_source(
    source: Path,
    frozen: Mapping[str, tuple[str, tuple[int, ...]]],
) -> None:
    """Reject content, inode or metadata changes after source verification."""

    for relative, expected in frozen.items():
        payload, identity = _source_file_snapshot(source / relative)
        if hashlib.sha256(payload).hexdigest() != expected[0] or identity != expected[1]:
            raise ScoringContractError("fixed CausalBench source file changed")


def verify_loaded_causalbench_modules(
    source: Path,
    frozen: Mapping[str, tuple[str, tuple[int, ...]]],
) -> None:
    """Require imported CausalBench modules to be tracked frozen Python files."""

    found_evaluation = False
    for name, module in tuple(sys.modules.items()):
        if name != "causalscbench" and not name.startswith("causalscbench."):
            continue
        filename = getattr(module, "__file__", None)
        if type(filename) is not str:
            raise ScoringContractError("loaded CausalBench module has no fixed source file")
        try:
            module_path = Path(filename).resolve(strict=True)
            relative = module_path.relative_to(source).as_posix()
        except (OSError, ValueError) as exc:
            raise ScoringContractError(
                "loaded CausalBench module left the fixed source"
            ) from exc
        if relative not in frozen:
            raise ScoringContractError(
                "loaded CausalBench module is not a tracked fixed Python file"
            )
        payload, identity = _source_file_snapshot(module_path)
        expected_hash, expected_identity = frozen[relative]
        if (
            hashlib.sha256(payload).hexdigest() != expected_hash
            or identity != expected_identity
        ):
            raise ScoringContractError("loaded CausalBench module source changed")
        if name == "causalscbench.evaluation.statistical_evaluation":
            found_evaluation = True
    if not found_evaluation:
        raise ScoringContractError("fixed CausalBench evaluation module was not loaded")


def _canonical_text_vector(
    values: Any,
    name: str,
    np: Any,
    *,
    expected_length: int,
    maximum_length: int,
) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ScoringContractError(f"{name} must be a one-dimensional text array")
    if (
        array.shape[0] != expected_length
        or array.shape[0] > maximum_length
        or array.dtype.itemsize <= 0
    ):
        raise ScoringContractError(f"{name} shapes do not match the heldout data")
    if array.dtype.kind == "S":
        try:
            items = tuple(
                value.decode("utf-8", errors="strict") for value in array.tolist()
            )
        except UnicodeError as exc:
            raise ScoringContractError(f"{name} must contain valid UTF-8") from exc
    else:
        items = tuple(array.tolist())
    total_bytes = 0
    for item in items:
        if type(item) is not str or not item or item != item.strip():
            raise ScoringContractError(f"{name} must contain non-empty canonical text")
        if not unicodedata.is_normalized("NFC", item):
            raise ScoringContractError(f"{name} must use NFC text")
        if any(ord(character) < 32 or ord(character) == 127 for character in item):
            raise ScoringContractError(f"{name} contains a control character")
        if item[0] in "=+-@":
            raise ScoringContractError(
                f"{name} must not begin with a spreadsheet formula marker"
            )
        try:
            item_bytes = len(item.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise ScoringContractError(f"{name} must contain valid UTF-8") from exc
        if item_bytes > MAXIMUM_TEXT_ITEM_BYTES:
            raise ScoringContractError(f"{name} contains unusually long text")
        total_bytes += item_bytes
    if total_bytes > MAXIMUM_TOTAL_TEXT_BYTES:
        raise ScoringContractError(f"{name} exceeds the total text limit")
    return items


def _preflight_npz(payload: bytes) -> None:
    expected = {
        "expression_matrix.npy",
        "interventions.npy",
        "var_names.npy",
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ScoringContractError(
                    "heldout NPZ must contain exactly the three registered arrays"
                )
            expanded = 0
            for member in members:
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or Path(member.filename).name != member.filename
                    or member.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or member.file_size <= 0
                ):
                    raise ScoringContractError("heldout NPZ archive is unsafe")
                expanded += int(member.file_size)
                if expanded > MAXIMUM_ARCHIVE_BYTES:
                    raise ScoringContractError("heldout NPZ archive is unusually large")
                if member.compress_size == 0 and member.file_size:
                    raise ScoringContractError("heldout NPZ archive has an invalid member")
                if member.file_size > max(1, member.compress_size) * 10_000:
                    raise ScoringContractError("heldout NPZ archive expands unusually")
    except ScoringContractError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ScoringContractError("heldout NPZ archive could not be read safely") from exc


def load_heldout_npz(payload: bytes, np: Any) -> tuple[Any, Any, tuple[str, ...]]:
    _preflight_npz(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "expression_matrix",
                "interventions",
                "var_names",
            }:
                raise ScoringContractError(
                    "heldout NPZ must contain exactly the three registered arrays"
                )
            expression = np.asarray(archive["expression_matrix"])
            interventions_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
    except ScoringContractError:
        raise
    except (OSError, ValueError, TypeError, EOFError) as exc:
        raise ScoringContractError("heldout NPZ arrays could not be read safely") from exc
    if expression.ndim != 2:
        raise ScoringContractError("heldout expression must be two-dimensional")
    cells, gene_count = expression.shape
    if cells < 2 or cells > MAXIMUM_CELLS:
        raise ScoringContractError("heldout expression has an invalid cell count")
    if gene_count < 2 or gene_count > MAXIMUM_GENES:
        raise ScoringContractError("heldout expression has an invalid gene count")
    if expression.dtype.kind not in {"i", "u", "f"}:
        raise ScoringContractError("heldout expression must contain numeric values")
    if not np.all(np.isfinite(expression)):
        raise ScoringContractError("heldout expression must contain finite values")
    if interventions_raw.ndim != 1 or genes_raw.ndim != 1:
        raise ScoringContractError("heldout array shapes do not agree")
    if interventions_raw.shape[0] != cells or genes_raw.shape[0] != gene_count:
        raise ScoringContractError("heldout array shapes do not agree")
    if interventions_raw.dtype.kind not in {"U", "S"} or genes_raw.dtype.kind not in {
        "U",
        "S",
    }:
        raise ScoringContractError("heldout text arrays must contain strings")
    if interventions_raw.dtype.itemsize <= 0 or genes_raw.dtype.itemsize <= 0:
        raise ScoringContractError("heldout text arrays have an invalid width")
    interventions = _canonical_text_vector(
        interventions_raw,
        "heldout intervention labels",
        np,
        expected_length=cells,
        maximum_length=MAXIMUM_CELLS,
    )
    genes = _canonical_text_vector(
        genes_raw,
        "heldout gene names",
        np,
        expected_length=gene_count,
        maximum_length=MAXIMUM_GENES,
    )
    if len(set(genes)) != len(genes):
        raise ScoringContractError("heldout gene names must be unique")
    eligible_sources = set(interventions) - {CONTROL_LABEL, EXCLUDED_LABEL}
    if CONTROL_LABEL not in interventions or not eligible_sources:
        raise ScoringContractError(
            "heldout cells need controls and at least one tested source gene"
        )
    if not eligible_sources <= set(genes):
        raise ScoringContractError(
            "heldout tested source labels must use the fixed gene names"
        )
    safe_expression = np.array(expression, copy=True)
    safe_interventions = np.asarray(interventions, dtype=str)
    safe_expression.setflags(write=False)
    safe_interventions.setflags(write=False)
    return safe_expression, safe_interventions, genes


def _canonical_endpoint(value: str, genes: set[str]) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or not unicodedata.is_normalized("NFC", value)
        or value not in genes
    ):
        raise ScoringContractError(
            "prediction endpoints must use the fixed heldout gene names"
        )
    return value


def load_predictions(
    payload: bytes, genes: Sequence[str]
) -> list[tuple[str, str, float, bool]]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        rows = csv.reader(io.StringIO(decoded, newline=""), strict=True)
        header = next(rows)
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise ScoringContractError("prediction CSV could not be read safely") from exc
    allowed_headers = (
        ["source", "target", "score"],
        ["source", "target", "score", "returned_by_method"],
    )
    if header not in allowed_headers:
        raise ScoringContractError(
            "prediction CSV must use the registered three or four columns"
        )
    has_returned = len(header) == 4
    gene_set = set(genes)
    relations: set[tuple[str, str]] = set()
    parsed: list[tuple[str, str, float, bool]] = []
    expected_count = len(genes) * (len(genes) - 1)
    try:
        for row in rows:
            if len(row) != len(header):
                raise ScoringContractError("prediction CSV rows have changed columns")
            source = _canonical_endpoint(row[0], gene_set)
            target = _canonical_endpoint(row[1], gene_set)
            if source == target:
                raise ScoringContractError("prediction CSV must not contain self relations")
            relation = (source, target)
            if relation in relations:
                raise ScoringContractError(
                    "prediction CSV must contain unique directed relations"
                )
            try:
                score = float(row[2])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ScoringContractError(
                    "prediction scores must be finite and non-negative"
                ) from exc
            if not math.isfinite(score) or score < 0.0:
                raise ScoringContractError(
                    "prediction scores must be finite and non-negative"
                )
            negative_zero = score == 0.0 and math.copysign(1.0, score) < 0.0
            score = 0.0 if score == 0.0 else score
            if has_returned:
                if row[3] not in {"True", "False"}:
                    raise ScoringContractError(
                        "returned_by_method must contain only True or False"
                    )
                if row[3] == "False" and (score != 0.0 or negative_zero):
                    raise ScoringContractError(
                        "unreturned relations must use positive zero scores"
                    )
                returned = row[3] == "True"
            else:
                returned = True
            relations.add(relation)
            parsed.append((source, target, score, returned))
            if len(parsed) > expected_count:
                raise ScoringContractError(
                    "prediction CSV contains more than the complete relation set"
                )
    except csv.Error as exc:
        raise ScoringContractError("prediction CSV could not be read safely") from exc
    expected = {
        (source, target)
        for source in genes
        for target in genes
        if source != target
    }
    if relations != expected or len(parsed) != expected_count:
        raise ScoringContractError(
            "prediction CSV must contain every directed non-self relation"
        )
    return parsed


def _safe_metric_value(
    value: Any,
    *,
    np: Any,
    depth: int,
    active: set[int],
    counter: list[int],
) -> Any:
    if depth > MAXIMUM_JSON_DEPTH:
        raise InvalidMetricOutput("failed_invalid_output: metrics are too deeply nested")
    counter[0] += 1
    if counter[0] > MAXIMUM_JSON_ITEMS:
        raise InvalidMetricOutput("failed_invalid_output: metrics contain too many values")
    if value is None or type(value) is bool:
        return value
    if isinstance(value, np.generic):
        return _safe_metric_value(
            value.item(), np=np, depth=depth, active=active, counter=counter
        )
    if type(value) is int:
        if abs(value) > 2**63 - 1:
            raise InvalidMetricOutput("failed_invalid_output: metric integer is too large")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidMetricOutput("failed_invalid_output: metric is not finite")
        return value
    if type(value) is str:
        try:
            size = len(value.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise InvalidMetricOutput(
                "failed_invalid_output: metric text is not valid UTF-8"
            ) from exc
        if size > MAXIMUM_TEXT_ITEM_BYTES:
            raise InvalidMetricOutput("failed_invalid_output: metric text is too long")
        return value
    identity = id(value)
    if identity in active:
        raise InvalidMetricOutput("failed_invalid_output: metrics contain a cycle")
    if isinstance(value, np.ndarray):
        if value.ndim > 8 or value.size > MAXIMUM_ARRAY_ITEMS:
            raise InvalidMetricOutput("failed_invalid_output: metric array is too large")
        active.add(identity)
        try:
            return _safe_metric_value(
                value.tolist(),
                np=np,
                depth=depth + 1,
                active=active,
                counter=counter,
            )
        finally:
            active.remove(identity)
    value_type = type(value)
    if value_type.__module__.startswith("torch") and value_type.__name__ == "Tensor":
        try:
            if int(value.numel()) > MAXIMUM_ARRAY_ITEMS:
                raise InvalidMetricOutput(
                    "failed_invalid_output: metric tensor is too large"
                )
            converted = value.detach().cpu().tolist()
        except InvalidMetricOutput:
            raise
        except Exception as exc:
            raise InvalidMetricOutput(
                "failed_invalid_output: metric tensor cannot be recorded"
            ) from exc
        return _safe_metric_value(
            converted,
            np=np,
            depth=depth + 1,
            active=active,
            counter=counter,
        )
    if isinstance(value, Mapping):
        active.add(identity)
        output: dict[str, Any] = {}
        try:
            iterator = iter(value.items())
            for index, item in enumerate(iterator):
                if index >= MAXIMUM_JSON_ITEMS:
                    raise InvalidMetricOutput(
                        "failed_invalid_output: metrics contain too many fields"
                    )
                if type(item) not in {tuple, list} or len(item) != 2:
                    raise InvalidMetricOutput(
                        "failed_invalid_output: metric mapping is malformed"
                    )
                key, child = item
                if type(key) is not str or not key or key in output:
                    raise InvalidMetricOutput(
                        "failed_invalid_output: metric keys must be unique text"
                    )
                output[key] = _safe_metric_value(
                    child,
                    np=np,
                    depth=depth + 1,
                    active=active,
                    counter=counter,
                )
        except InvalidMetricOutput:
            raise
        except Exception as exc:
            raise InvalidMetricOutput(
                "failed_invalid_output: metric mapping cannot be read"
            ) from exc
        finally:
            active.remove(identity)
        return output
    if type(value) in {list, tuple}:
        if len(value) > MAXIMUM_ARRAY_ITEMS:
            raise InvalidMetricOutput("failed_invalid_output: metric list is too large")
        active.add(identity)
        try:
            return [
                _safe_metric_value(
                    child,
                    np=np,
                    depth=depth + 1,
                    active=active,
                    counter=counter,
                )
                for child in value
            ]
        finally:
            active.remove(identity)
    raise InvalidMetricOutput(
        "failed_invalid_output: metric value has an unsupported type"
    )


def make_json_safe(value: Any, np: Any) -> Any:
    """Convert only bounded, finite official results to strict JSON values."""

    return _safe_metric_value(
        value, np=np, depth=0, active=set(), counter=[0]
    )


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_output_parent(path: Path) -> tuple[Path, int, tuple[int, int]]:
    destination = _absolute(path)
    if destination.name in {"", ".", ".."}:
        raise ScoringContractError("output JSON needs a file name")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination.anchor, flags)
    try:
        for component in destination.parent.parts[1:]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        return destination, descriptor, (int(metadata.st_dev), int(metadata.st_ino))
    except OSError as exc:
        os.close(descriptor)
        raise ScoringContractError(
            "output parent must be a real directory without symbolic links"
        ) from exc


def _parent_matches(path: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and (int(metadata.st_dev), int(metadata.st_ino)) == identity
    )


def _exists(descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _unlink_inode(descriptor: int, name: str, identity: tuple[int, int]) -> bool:
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (int(metadata.st_dev), int(metadata.st_ino)) == identity:
            os.unlink(name, dir_fd=descriptor)
            return True
    except OSError:
        pass
    return False


def validate_output_destination(path: Path) -> None:
    destination, descriptor, parent_identity = _open_output_parent(path)
    try:
        if _exists(descriptor, destination.name):
            raise ScoringContractError("output JSON must not already exist")
        if not _parent_matches(destination.parent, parent_identity):
            raise ScoringContractError("output parent directory changed")
    finally:
        os.close(descriptor)


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise ScoringContractError("output JSON could not be encoded strictly") from exc
    destination, descriptor, parent_identity = _open_output_parent(path)
    temporary: str | None = None
    temporary_identity: tuple[int, int] | None = None
    linked = False
    completed = False
    try:
        if _exists(descriptor, destination.name):
            raise ScoringContractError("output JSON must not already exist")
        if not _parent_matches(destination.parent, parent_identity):
            raise ScoringContractError("output parent directory changed")
        temporary = f".{destination.name}.{secrets.token_hex(16)}.tmp"
        file_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=descriptor,
        )
        metadata = os.fstat(file_descriptor)
        temporary_identity = (int(metadata.st_dev), int(metadata.st_ino))
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if not _parent_matches(destination.parent, parent_identity):
            raise ScoringContractError("output parent directory changed")
        os.link(
            temporary,
            destination.name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
            follow_symlinks=False,
        )
        linked = True
        if not _parent_matches(destination.parent, parent_identity):
            raise ScoringContractError("output parent directory changed")
        if temporary_identity is None or not _unlink_inode(
            descriptor, temporary, temporary_identity
        ):
            raise ScoringContractError("temporary output could not be removed safely")
        temporary = None
        os.fsync(descriptor)
        completed = True
    except ScoringContractError:
        raise
    except FileExistsError as exc:
        raise ScoringContractError("output JSON must not already exist") from exc
    except OSError as exc:
        raise ScoringContractError("output JSON could not be written atomically") from exc
    finally:
        if linked and not completed and temporary_identity is not None:
            _unlink_inode(descriptor, destination.name, temporary_identity)
        if temporary is not None and temporary_identity is not None:
            _unlink_inode(descriptor, temporary, temporary_identity)
        os.close(descriptor)


def _fixed_seed(value: str) -> int:
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        raise ScoringContractError(
            f"seed must be a whole number from 0 to {MAXIMUM_SEED}"
        )
    try:
        seed = int(value, 10)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScoringContractError(
            f"seed must be a whole number from 0 to {MAXIMUM_SEED}"
        ) from exc
    if not 0 <= seed <= MAXIMUM_SEED:
        raise ScoringContractError(
            f"seed must be a whole number from 0 to {MAXIMUM_SEED}"
        )
    return seed


def _failure_payload(status: str, seed: int | None, error: str) -> dict[str, Any]:
    return {
        "error": error[:500],
        "schema_version": "1.0",
        "seed": seed,
        "status": status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed: int | None = None
    try:
        validate_output_destination(args.output_json)
        seed = _fixed_seed(args.seed)
        if sys.flags.isolated != 1:
            message = "real sealed scoring requires isolated Python: run with python -I"
            write_json_new(
                args.output_json,
                _failure_payload("failed_runtime_unavailable", seed, message),
            )
            print(f"failed_runtime_unavailable: {message}", file=sys.stderr)
            return 1
        boundary = _load_causalbench_boundary_module()
        with _clean_python_environment():
            try:
                source = boundary.validate_causalbench_source(
                    args.causalbench_source,
                    expected_commit=EXPECTED_CAUSALBENCH_COMMIT,
                )
            except SystemExit as exc:
                raise ScoringContractError(str(exc)) from exc
            import numpy as np

            with fixed_causalbench_source_snapshot(
                source,
                boundary,
                expected_commit=EXPECTED_CAUSALBENCH_COMMIT,
            ) as (snapshot_source, frozen_source):
                with boundary._verified_causalbench_imports(snapshot_source):
                    evaluation_module = importlib.import_module(
                        "causalscbench.evaluation.statistical_evaluation"
                    )
                    boundary._assert_causalbench_modules_are_verified(
                        snapshot_source
                    )
                    verify_causalbench_python_source(
                        snapshot_source, frozen_source
                    )
                    verify_loaded_causalbench_modules(
                        snapshot_source, frozen_source
                    )
                    prediction_payload, prediction_inode = _snapshot(
                        args.prediction_csv,
                        "prediction CSV",
                        MAXIMUM_PREDICTION_BYTES,
                    )
                    heldout_payload, heldout_inode = _snapshot(
                        args.heldout_npz,
                        "heldout NPZ",
                        MAXIMUM_HELDOUT_BYTES,
                    )
                    if prediction_inode == heldout_inode:
                        raise ScoringContractError(
                            "prediction and heldout inputs must be separate files"
                        )
                    expression, interventions, genes = load_heldout_npz(
                        heldout_payload, np
                    )
                    predictions = load_predictions(prediction_payload, genes)
                    eligible_sources = set(interventions.tolist()) - {
                        CONTROL_LABEL,
                        EXCLUDED_LABEL,
                    }
                    ordered_edges = [
                        (source_name, target_name)
                        for source_name, target_name, _score, _returned in sorted(
                            (
                                relation
                                for relation in predictions
                                if relation[0] in eligible_sources and relation[3]
                            ),
                            key=lambda relation: (
                                -relation[2],
                                relation[0],
                                relation[1],
                            ),
                        )
                    ]
                    verify_causalbench_python_source(
                        snapshot_source, frozen_source
                    )
                    verify_loaded_causalbench_modules(
                        snapshot_source, frozen_source
                    )
                    if ordered_edges:
                        evaluator = evaluation_module.Evaluator(
                            expression,
                            interventions,
                            list(genes),
                        )
                        verify_causalbench_python_source(
                            snapshot_source, frozen_source
                        )
                        verify_loaded_causalbench_modules(
                            snapshot_source, frozen_source
                        )
                        raw_metrics = evaluator.evaluate_network(
                            ordered_edges,
                            max_path_length=1,
                            check_false_omission_rate=False,
                            omission_estimation_size=0,
                            seed=seed,
                        )
                        boundary._assert_causalbench_modules_are_verified(
                            snapshot_source
                        )
                        verify_causalbench_python_source(
                            snapshot_source, frozen_source
                        )
                        verify_loaded_causalbench_modules(
                            snapshot_source, frozen_source
                        )
                metrics = (
                    make_json_safe(raw_metrics, np) if ordered_edges else None
                )
                verify_causalbench_python_source(
                    snapshot_source, frozen_source
                )
        if ordered_edges:
            payload = {
                "schema_version": "1.0",
                "status": "supplementary_official_metrics",
                "seed": seed,
                "eligible_source_count": len(eligible_sources),
                "metrics": metrics,
            }
        else:
            payload = {
                "schema_version": "1.0",
                "status": "supplementary_official_metrics_not_applicable",
                "seed": seed,
                "eligible_source_count": len(eligible_sources),
                "metrics": None,
                "reason": "no_returned_relations_for_eligible_sources",
            }
        write_json_new(args.output_json, payload)
        return 0
    except InvalidMetricOutput as exc:
        message = str(exc)
        try:
            write_json_new(
                args.output_json,
                _failure_payload("failed_invalid_output", seed, message),
            )
        except ScoringContractError as write_error:
            print(str(write_error), file=sys.stderr)
            return 1
        print(message, file=sys.stderr)
        return 1
    except (ScoringContractError, OSError, ValueError, TypeError) as exc:
        message = str(exc) or "sealed scoring failed"
        try:
            write_json_new(
                args.output_json,
                _failure_payload("failed_private_scoring", seed, message),
            )
        except ScoringContractError as write_error:
            print(str(write_error), file=sys.stderr)
            return 1
        print(f"failed_private_scoring: {message}", file=sys.stderr)
        return 1
    except Exception:
        message = "official evaluator failed before producing valid metrics"
        try:
            write_json_new(
                args.output_json,
                _failure_payload("failed_private_scoring", seed, message),
            )
        except ScoringContractError as write_error:
            print(str(write_error), file=sys.stderr)
            return 1
        print(f"failed_private_scoring: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
