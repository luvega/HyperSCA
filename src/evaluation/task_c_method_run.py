"""Run one Task C method behind a shared, auditable result contract.

This layer does not decide whether a biological claim is correct.  It fixes which
data a method may inspect, preserves the method's original output, and maps every
runnable method to the same directed gene-relation table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
import unicodedata
import uuid
import zipfile

import numpy as np
import pandas as pd

from src.evaluation.task_c_benchmark import (
    TaskCBenchmarkError,
    score_mean_difference_network,
)
from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistry,
    TaskCMethodRegistryError,
    TaskCMethodSpec,
    load_task_c_method_registry,
)
from src.evaluation.task_c_predictions import (
    TaskCPredictionError,
    normalize_task_c_predictions,
)
from src.evaluation.task_c_runtime import run_isolated_method
from src.evaluation.task_c_runtime import (
    TaskCRuntimeError,
    _validate_source_checkout,
)


SCHEMA_VERSION = "1.0"
MAXIMUM_TASK_C_RUN_GENES = 256
MAXIMUM_INPUT_BYTES = 512 * 1024 * 1024
MAXIMUM_RAW_PREDICTION_BYTES = 64 * 1024 * 1024
MAXIMUM_RECORD_BYTES = 4 * 1024 * 1024
_CONTROL_LABEL = "non-targeting"
_EXCLUDED_LABEL = "excluded"
_DATA_STATUSES = frozenset({"external_benchmark", "synthetic_smoke"})
_COMPLETED_STATUS = "completed_standardized_output"
_FAILED_STATUSES = frozenset(
    {
        "failed_timeout",
        "failed_resource_limit",
        "failed_runtime_unavailable",
        "failed_launch",
        "official_code_incompatible",
        "failed_invalid_output",
    }
)
_PUBLIC_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "train_sources",
        "tune_sources",
        "holdout_source_count",
        "input_sha256",
        "content_sha256",
        "gene_names_sha256",
        "materialization_identity",
        "files",
    }
)
_PUBLIC_TASK_C_PATHS = frozenset(
    f"within/{context}/{partition}.npz"
    for context in ("k562", "rpe1")
    for partition in ("train", "tune", "refit")
) | frozenset(
    f"cross/{direction}/{partition}.npz"
    for direction in ("k562_to_rpe1", "rpe1_to_k562")
    for partition in (
        "source_train",
        "source_tune",
        "source_refit",
        "target_adapt_train",
        "target_adapt_tune",
        "target_adapt_refit",
    )
)


class TaskCMethodRunError(ValueError):
    """A method run cannot satisfy the shared Task C evidence boundary."""


class _InvalidMethodOutput(TaskCMethodRunError):
    """The external method finished but did not return an admissible relation table."""


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    payload: bytes
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    link_count: int

    @property
    def sha256(self) -> str:
        return f"sha256:{hashlib.sha256(self.payload).hexdigest()}"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
        int(stat_result.st_nlink),
    )


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _has_private_component(path: Path) -> bool:
    return any(part.casefold().startswith("private") for part in path.parts)


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise TaskCMethodRunError(f"{label} does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCMethodRunError(f"{label} must not use a symbolic link")


def _capture_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    reject_private: bool = False,
    require_single_link: bool = False,
) -> _Snapshot:
    absolute = _lexical_absolute(path)
    if reject_private and _has_private_component(absolute):
        raise TaskCMethodRunError(f"{label} must not come from a private path")
    _reject_symlink_components(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise TaskCMethodRunError(f"{label} must be an existing regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TaskCMethodRunError(f"{label} must be a regular file")
        if before.st_size <= 0:
            raise TaskCMethodRunError(f"{label} must not be empty")
        if before.st_size > maximum_bytes:
            raise TaskCMethodRunError(f"{label} is too large")
        if require_single_link and before.st_nlink != 1:
            raise TaskCMethodRunError(f"{label} must not be a hard link")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        current = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskCMethodRunError(f"{label} changed while it was read") from exc
    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(current)
    ):
        raise TaskCMethodRunError(f"{label} changed while it was read")
    return _Snapshot(
        path=absolute,
        payload=payload,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        modified_ns=int(after.st_mtime_ns),
        changed_ns=int(after.st_ctime_ns),
        link_count=int(after.st_nlink),
    )


def _verify_snapshot(snapshot: _Snapshot, label: str) -> None:
    current = _capture_file(
        snapshot.path,
        label,
        maximum_bytes=max(snapshot.size, 1),
    )
    if current != snapshot:
        raise TaskCMethodRunError(f"{label} changed during the method run")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskCMethodRunError(f"JSON contains a duplicate field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise TaskCMethodRunError(f"JSON contains a non-finite value: {value}")


def _parse_json(snapshot: _Snapshot, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            snapshot.payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except TaskCMethodRunError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TaskCMethodRunError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TaskCMethodRunError(f"{label} must be a JSON object")
    return payload


def _canonical_text(values: np.ndarray, label: str) -> tuple[str, ...]:
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise TaskCMethodRunError(f"{label} must be a one-dimensional text array")
    try:
        items = tuple(
            value.decode("utf-8", errors="strict") if isinstance(value, bytes) else str(value)
            for value in values.tolist()
        )
    except UnicodeError as exc:
        raise TaskCMethodRunError(f"{label} must use valid UTF-8") from exc
    if any(
        not item
        or item != item.strip()
        or not unicodedata.is_normalized("NFC", item)
        or item[0] in "=+-@"
        or any(ord(character) < 32 or ord(character) == 127 for character in item)
        for item in items
    ):
        raise TaskCMethodRunError(
            f"{label} must contain safe, non-empty, canonical text"
        )
    return items


def _load_fixed_npz(snapshot: _Snapshot) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    buffer = io.BytesIO(snapshot.payload)
    try:
        with zipfile.ZipFile(buffer) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            expected_members = {
                "expression_matrix.npy",
                "interventions.npy",
                "var_names.npy",
            }
            if len(names) != len(expected_members) or set(names) != expected_members:
                raise TaskCMethodRunError(
                    "input NPZ must contain exactly expression_matrix, interventions, and var_names"
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise TaskCMethodRunError("input NPZ must not contain encrypted arrays")
            if sum(member.file_size for member in members) > MAXIMUM_INPUT_BYTES:
                raise TaskCMethodRunError("input NPZ expands beyond the allowed size")
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            if set(archive.files) != {
                "expression_matrix",
                "interventions",
                "var_names",
            }:
                raise TaskCMethodRunError("input NPZ contains an unexpected array")
            expression = np.asarray(archive["expression_matrix"])
            interventions_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
    except TaskCMethodRunError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise TaskCMethodRunError("input NPZ could not be read safely") from exc
    if expression.ndim != 2 or expression.shape[0] == 0 or expression.shape[1] < 2:
        raise TaskCMethodRunError("expression matrix must contain cells and at least two genes")
    if expression.shape[1] > MAXIMUM_TASK_C_RUN_GENES:
        raise TaskCMethodRunError(
            f"a verified Task C run accepts at most {MAXIMUM_TASK_C_RUN_GENES} genes"
        )
    if expression.dtype.kind not in {"i", "u", "f"} or not np.isfinite(expression).all():
        raise TaskCMethodRunError("expression matrix must contain finite numbers")
    interventions = _canonical_text(interventions_raw, "intervention labels")
    genes = _canonical_text(genes_raw, "gene names")
    if len(interventions) != expression.shape[0] or len(genes) != expression.shape[1]:
        raise TaskCMethodRunError("input NPZ array dimensions do not agree")
    if len(set(genes)) != len(genes):
        raise TaskCMethodRunError("gene names must be unique")
    if _CONTROL_LABEL not in interventions:
        raise TaskCMethodRunError("input NPZ needs non-targeting control cells")
    allowed_labels = set(genes) | {_CONTROL_LABEL, _EXCLUDED_LABEL}
    if not set(interventions).issubset(allowed_labels):
        raise TaskCMethodRunError("intervention labels contain a gene outside the fixed gene set")
    safe_expression = np.array(expression, copy=True)
    safe_interventions = np.asarray(interventions, dtype=str)
    safe_expression.setflags(write=False)
    safe_interventions.setflags(write=False)
    return safe_expression, safe_interventions, genes


def _capture_public_input(
    input_path: Path,
    manifest_path: Path,
) -> tuple[_Snapshot, _Snapshot, dict[str, Any], str]:
    manifest = _capture_file(
        manifest_path,
        "public manifest",
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        reject_private=True,
        require_single_link=True,
    )
    payload = _parse_json(manifest, "public manifest")
    if set(payload) != _PUBLIC_MANIFEST_FIELDS or payload.get("schema_version") != "1.0":
        raise TaskCMethodRunError("public manifest does not follow the fixed Task C schema")
    identity_fields = {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "input_sha256",
        "content_sha256",
        "gene_names_sha256",
    }
    materialization_identity = payload.get("materialization_identity")
    if not isinstance(materialization_identity, dict) or set(
        materialization_identity
    ) != identity_fields:
        raise TaskCMethodRunError("public manifest materialization identity is malformed")
    if any(
        materialization_identity.get(field) != payload.get(field)
        for field in identity_fields
    ):
        raise TaskCMethodRunError(
            "public manifest materialization identity disagrees with the split record"
        )
    for integer_field in (
        "seed",
        "min_cells_per_intervention",
        "holdout_source_count",
    ):
        value = payload.get(integer_field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskCMethodRunError(
                f"public manifest {integer_field} must be a non-negative integer"
            )
    if payload["min_cells_per_intervention"] < 1:
        raise TaskCMethodRunError("public manifest minimum cell count must be positive")
    for list_field in ("train_sources", "tune_sources"):
        values = payload.get(list_field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise TaskCMethodRunError(f"public manifest {list_field} is malformed")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != _PUBLIC_TASK_C_PATHS:
        raise TaskCMethodRunError(
            "public manifest must contain the complete public file inventory"
        )
    root = manifest.path.parent.resolve(strict=True)
    requested = _lexical_absolute(input_path)
    if _has_private_component(requested):
        raise TaskCMethodRunError("input NPZ must not come from a private path")
    _reject_symlink_components(requested, "input NPZ")
    requested = requested.resolve(strict=True)
    try:
        requested_relative = requested.relative_to(root).as_posix()
    except ValueError as exc:
        raise TaskCMethodRunError("input NPZ is outside the public manifest directory") from exc
    if requested_relative not in files:
        raise TaskCMethodRunError("input NPZ is not registered by the public manifest")

    snapshots: dict[str, _Snapshot] = {}
    inode_counts: Counter[tuple[int, int]] = Counter()
    for relative, expected_hash in sorted(files.items()):
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or _has_private_component(Path(relative))
        ):
            raise TaskCMethodRunError("public manifest contains an unsafe file record")
        candidate = root / relative
        snapshot = _capture_file(
            candidate,
            f"registered public file {relative}",
            maximum_bytes=MAXIMUM_INPUT_BYTES,
            reject_private=True,
        )
        if snapshot.sha256 != expected_hash:
            raise TaskCMethodRunError(f"registered public file hash changed: {relative}")
        snapshots[relative] = snapshot
        inode_counts[(snapshot.device, snapshot.inode)] += 1
    for relative, snapshot in snapshots.items():
        if snapshot.link_count != inode_counts[(snapshot.device, snapshot.inode)]:
            raise TaskCMethodRunError(
                f"registered public file has an unregistered hard link: {relative}"
            )
    selected = snapshots[requested_relative]
    if selected.path != requested:
        raise TaskCMethodRunError("input NPZ path does not match its public record")
    return selected, manifest, payload, requested_relative


def _context_for_public_path(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) == 3 and parts[0] == "within" and parts[1] in {"k562", "rpe1"}:
        return parts[1]
    if len(parts) == 3 and parts[0] == "cross" and "_to_" in parts[1]:
        source, target = parts[1].split("_to_", 1)
        if source not in {"k562", "rpe1"} or target not in {"k562", "rpe1"}:
            raise TaskCMethodRunError("public path has an unknown biological context")
        if parts[2].startswith("source_"):
            return source
        if parts[2].startswith("target_adapt_"):
            return target
    raise TaskCMethodRunError("public path does not identify one Task C context")


def _build_hypersca_command(
    *,
    project_root: Path,
    context_values: Sequence[str],
    config_path: Path,
    gene_list_path: Path,
    public_manifest_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> tuple[str, ...]:
    context_arguments = tuple(
        item
        for value in context_values
        for item in ("--context", value)
    )
    return (
        sys.executable,
        str(project_root / "scripts/run_hypersca_c.py"),
        *context_arguments,
        "--config",
        str(config_path),
        "--gene-list",
        str(gene_list_path),
        "--public-manifest",
        str(public_manifest_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--device",
        device,
    )


def _capture_synthetic_input(path: Path) -> _Snapshot:
    return _capture_file(
        path,
        "input NPZ",
        maximum_bytes=MAXIMUM_INPUT_BYTES,
        reject_private=True,
        require_single_link=True,
    )


def read_task_c_raw_predictions(
    path: Path,
    gene_names: Sequence[str],
    *,
    maximum_bytes: int = MAXIMUM_RAW_PREDICTION_BYTES,
) -> pd.DataFrame:
    """Read a bounded three-column CSV before it can enter shared scoring."""
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 1:
        raise TaskCMethodRunError("maximum raw output bytes must be a positive integer")
    snapshot = _capture_file(
        path,
        "raw prediction CSV",
        maximum_bytes=maximum_bytes,
        require_single_link=True,
    )
    maximum_rows = max(1_000, 4 * len(gene_names) * max(0, len(gene_names) - 1))
    if snapshot.payload.count(b"\n") > maximum_rows + 1:
        raise TaskCMethodRunError("raw prediction CSV has too many rows")
    try:
        snapshot.payload.decode("utf-8", errors="strict")
        raw = pd.read_csv(
            io.BytesIO(snapshot.payload),
            encoding="utf-8",
            keep_default_na=False,
            on_bad_lines="error",
        )
    except UnicodeError as exc:
        raise TaskCMethodRunError("raw prediction CSV must use valid UTF-8") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise TaskCMethodRunError("raw prediction CSV is malformed") from exc
    if len(raw.columns) != 3 or set(raw.columns) != {"source", "target", "score"}:
        raise TaskCMethodRunError(
            "prediction table must contain exactly source, target, and score"
        )
    try:
        normalize_task_c_predictions(raw, gene_names)
    except TaskCPredictionError as exc:
        raise TaskCMethodRunError(str(exc)) from exc
    _verify_snapshot(snapshot, "raw prediction CSV")
    return raw[["source", "target", "score"]].copy()


def build_task_c_method_command(
    spec: TaskCMethodSpec,
    *,
    input_path: Path,
    output_csv: Path,
    asset_root: Path,
    seed: int,
    project_root: Path,
) -> tuple[str, ...]:
    """Build the fixed worker command without starting an external method."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TaskCMethodRunError("seed must be a non-negative integer")
    if spec.source_kind == "causalbench":
        environment = spec.environment
        if environment is None:
            environment = "hypersca-task-c-causalbench"
        return (
            "conda",
            "run",
            "-n",
            environment,
            "python",
            str(project_root / "scripts/task_c_workers/causalbench_worker.py"),
            "--input-npz",
            str(input_path),
            "--output-csv",
            str(output_csv),
            "--model-name",
            str(spec.command),
            "--causalbench-source",
            str(asset_root / "sources/causalbench"),
            "--training-information",
            spec.training_information,
            "--seed",
            str(seed),
            "--output-semantics",
            spec.output_semantics,
        )
    if spec.source_kind == "git" and spec.method_id == "guanlab_psgrn":
        if spec.environment is None:
            raise TaskCMethodRunError("PSGRN has no registered isolated environment")
        return (
            "conda",
            "run",
            "-n",
            spec.environment,
            "python",
            str(project_root / "scripts/task_c_workers/psgrn_worker.py"),
            "--input-npz",
            str(input_path),
            "--output-csv",
            str(output_csv),
            "--psgrn-source",
            str(asset_root / "sources/guanlab_psgrn"),
            "--training-information",
            spec.training_information,
            "--seed",
            str(seed),
            "--output-semantics",
            spec.output_semantics,
        )
    raise TaskCMethodRunError(f"{spec.method_id} does not use an external Task C worker")


def _safe_command_record(command: Sequence[str]) -> dict[str, object]:
    options: list[str] = []
    executable_names = [Path(command[0]).name]
    path_options = {
        "--input-npz",
        "--output-csv",
        "--causalbench-source",
        "--psgrn-source",
        "--config",
        "--gene-list",
        "--public-manifest",
        "--output-dir",
    }
    template: list[str] = []
    hide_next = False
    context_next = False
    for argument in command:
        if argument.startswith("--"):
            options.append(argument)
            hide_next = argument in path_options
            context_next = argument == "--context"
            template.append(argument)
            continue
        if argument.endswith(".py"):
            executable_names.append(Path(argument).name)
        if hide_next:
            template.append("<fixed-path>")
            hide_next = False
        elif context_next:
            name = argument.split("=", 1)[0]
            template.append(f"{name}=<registered-public-path>")
            context_next = False
        else:
            template.append(argument)
    encoded = json.dumps(template, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "sha256": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "argument_count": len(command),
        "executable_names": executable_names,
        "options": options,
        "template": template,
    }


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_new(path, _json_bytes(payload))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _artifact_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    identities: set[tuple[int, int]] = set()
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "method_status.json":
            continue
        snapshot = _capture_file(
            path,
            f"result artifact {relative}",
            maximum_bytes=MAXIMUM_INPUT_BYTES,
            require_single_link=True,
        )
        inode = (snapshot.device, snapshot.inode)
        if inode in identities:
            raise TaskCMethodRunError("result artifacts must not share hard-link identities")
        identities.add(inode)
        records[relative] = {"sha256": snapshot.sha256, "size_bytes": snapshot.size}
    return records


def _code_snapshots(project_root: Path, spec: TaskCMethodSpec) -> dict[str, _Snapshot]:
    paths = [
        project_root / "scripts/run_task_c_method.py",
        project_root / "src/evaluation/task_c_method_run.py",
        project_root / "src/evaluation/task_c_predictions.py",
    ]
    if spec.method_id == "mean_difference":
        paths.append(project_root / "src/evaluation/task_c_benchmark.py")
    elif spec.source_kind == "causalbench":
        paths.append(project_root / "scripts/task_c_workers/causalbench_worker.py")
    elif spec.source_kind == "git":
        paths.append(project_root / "scripts/task_c_workers/psgrn_worker.py")
    elif spec.method_id == "hypersca_c":
        paths.extend(
            [
                project_root / "scripts/run_hypersca_c.py",
                project_root / "src/causal/hypersca_c_run.py",
                project_root / "src/causal/hypersca_c.py",
            ]
        )
    return {
        path.relative_to(project_root).as_posix(): _capture_file(
            path,
            f"code file {path.name}",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
            require_single_link=True,
        )
        for path in paths
    }


def _registered_method_record(
    spec: TaskCMethodSpec,
    registry: TaskCMethodRegistry,
) -> dict[str, object]:
    repository = spec.repository
    commit = spec.commit
    environment = spec.environment
    if spec.source_kind == "causalbench":
        repository = registry.causalbench["repository"]
        commit = registry.causalbench["commit"]
        environment = registry.causalbench["environment"]
    return {
        "method_id": spec.method_id,
        "role": spec.role,
        "source_kind": spec.source_kind,
        "training_information": spec.training_information,
        "output_semantics": spec.output_semantics,
        "command": spec.command,
        "repository": repository,
        "commit": commit,
        "environment": environment,
        "publication": spec.publication,
        "required_for_core_rehearsal": spec.required_for_core_rehearsal,
    }


def _asset_snapshots(
    asset_root: Path,
    registry: TaskCMethodRegistry,
    spec: TaskCMethodSpec,
) -> dict[str, _Snapshot]:
    if spec.source_kind not in {"causalbench", "git"}:
        return {}
    root = _lexical_absolute(asset_root)
    _reject_symlink_components(root, "method asset root")
    if not root.is_dir():
        raise TaskCMethodRunError("method assets have not been prepared")
    environment = (
        registry.causalbench["environment"]
        if spec.source_kind == "causalbench"
        else spec.environment
    )
    assert environment is not None
    required = {
        "bootstrap_identity.json": root / "bootstrap_identity.json",
        "bootstrap_manifest.json": root / "bootstrap_manifest.json",
        "bootstrap_status.json": root / "bootstrap_status.json",
        f"environment_manifests/{environment}.json": root
        / f"environment_manifests/{environment}.json",
    }
    snapshots = {
        name: _capture_file(
            path,
            f"method asset {name}",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
            require_single_link=True,
        )
        for name, path in required.items()
    }
    status = _parse_json(snapshots["bootstrap_status.json"], "bootstrap status")
    if status.get("status") != "assets_and_environments_recorded":
        raise TaskCMethodRunError("method assets are not in a completed state")
    identity = _parse_json(snapshots["bootstrap_identity.json"], "bootstrap identity")
    expected_registry = identity.get("registry_sha256")
    if not isinstance(expected_registry, str):
        raise TaskCMethodRunError("method asset identity does not record the registry")
    return snapshots


def _external_source_digest(
    asset_root: Path,
    registry: TaskCMethodRegistry,
    spec: TaskCMethodSpec,
) -> str | None:
    if spec.source_kind == "causalbench":
        source = _lexical_absolute(asset_root) / "sources/causalbench"
        expected = {
            "repository": registry.causalbench["repository"],
            "commit": registry.causalbench["commit"],
        }
    elif spec.source_kind == "git":
        assert spec.repository is not None and spec.commit is not None
        source = _lexical_absolute(asset_root) / f"sources/{spec.method_id}"
        expected = {"repository": spec.repository, "commit": spec.commit}
    else:
        return None
    try:
        digest = _validate_source_checkout(source, expected, subprocess.run)
    except TaskCRuntimeError as exc:
        raise TaskCMethodRunError(str(exc)) from exc
    return f"sha256:{digest}"


def _run_identity(
    *,
    spec: TaskCMethodSpec,
    registry: TaskCMethodRegistry,
    registry_snapshot: _Snapshot,
    input_snapshot: _Snapshot | None,
    public_manifest_snapshot: _Snapshot | None,
    code_snapshots: Mapping[str, _Snapshot],
    asset_snapshots: Mapping[str, _Snapshot],
    seed: int,
    data_status: str | None,
    context_id: str | None,
    min_cells: int,
    command_record: Mapping[str, object] | None,
    hypersca_inputs: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method": _registered_method_record(spec, registry),
        "registry_sha256": registry_snapshot.sha256,
        "input_sha256": input_snapshot.sha256 if input_snapshot else None,
        "public_manifest_sha256": (
            public_manifest_snapshot.sha256 if public_manifest_snapshot else None
        ),
        "seed": seed,
        "data_status": data_status,
        "context_id": context_id,
        "min_cells": min_cells,
        "code": {name: snapshot.sha256 for name, snapshot in code_snapshots.items()},
        "assets": {name: snapshot.sha256 for name, snapshot in asset_snapshots.items()},
        "command": dict(command_record) if command_record is not None else None,
        "hypersca_inputs": dict(hypersca_inputs),
    }
    digest = f"sha256:{hashlib.sha256(_json_bytes(identity)).hexdigest()}"
    return identity, digest


def _environment_manifest(
    *,
    spec: TaskCMethodSpec,
    registry: TaskCMethodRegistry,
    registry_snapshot: _Snapshot,
    input_snapshot: _Snapshot | None,
    public_manifest_snapshot: _Snapshot | None,
    code_snapshots: Mapping[str, _Snapshot],
    asset_snapshots: Mapping[str, _Snapshot],
    seed: int,
    data_status: str | None,
    context_id: str | None,
    min_cells: int,
    command_record: Mapping[str, object] | None,
    hypersca_inputs: Mapping[str, str],
) -> dict[str, object]:
    identity, identity_sha256 = _run_identity(
        spec=spec,
        registry=registry,
        registry_snapshot=registry_snapshot,
        input_snapshot=input_snapshot,
        public_manifest_snapshot=public_manifest_snapshot,
        code_snapshots=code_snapshots,
        asset_snapshots=asset_snapshots,
        seed=seed,
        data_status=data_status,
        context_id=context_id,
        min_cells=min_cells,
        command_record=command_record,
        hypersca_inputs=hypersca_inputs,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": spec.method_id,
        "role": spec.role,
        "source_kind": spec.source_kind,
        "training_information": spec.training_information,
        "output_semantics": spec.output_semantics,
        "data_status": data_status,
        "context_id": context_id,
        "seed": seed,
        "min_cells": min_cells,
        "registry_sha256": registry_snapshot.sha256,
        "input": (
            {"sha256": input_snapshot.sha256, "size_bytes": input_snapshot.size}
            if input_snapshot
            else None
        ),
        "public_manifest": (
            {
                "sha256": public_manifest_snapshot.sha256,
                "size_bytes": public_manifest_snapshot.size,
            }
            if public_manifest_snapshot
            else None
        ),
        "registered_method": _registered_method_record(spec, registry),
        "code": {name: snapshot.sha256 for name, snapshot in code_snapshots.items()},
        "assets": {name: snapshot.sha256 for name, snapshot in asset_snapshots.items()},
        "command": dict(command_record) if command_record is not None else None,
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "run_identity": identity,
        "run_identity_sha256": identity_sha256,
    }


def _read_existing_json(path: Path, label: str) -> tuple[dict[str, Any], _Snapshot]:
    snapshot = _capture_file(
        path,
        label,
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        require_single_link=True,
    )
    return _parse_json(snapshot, label), snapshot


def _validate_existing_output(
    output_dir: Path,
    *,
    expected_environment: Mapping[str, object],
    spec: TaskCMethodSpec,
    gene_names: Sequence[str] | None,
    expected_raw: pd.DataFrame | None = None,
) -> dict[str, object]:
    root = _lexical_absolute(output_dir)
    _reject_symlink_components(root, "existing result directory")
    if not root.is_dir():
        raise TaskCMethodRunError("existing result path is not a directory")
    names = {path.name for path in root.iterdir()}
    if not {"method_status.json", "environment_manifest.json"}.issubset(names):
        raise TaskCMethodRunError("existing result is incomplete or unrecognized")
    status, _ = _read_existing_json(root / "method_status.json", "method status")
    environment, _ = _read_existing_json(
        root / "environment_manifest.json", "environment manifest"
    )
    if environment != expected_environment:
        raise TaskCMethodRunError("existing result environment record changed")
    expected_identity_sha256 = str(expected_environment["run_identity_sha256"])
    if environment.get("run_identity_sha256") != expected_identity_sha256:
        raise TaskCMethodRunError("existing result has a different run identity")
    if environment.get("method_id") != spec.method_id or status.get("method_id") != spec.method_id:
        raise TaskCMethodRunError("existing result method identity changed")
    if status.get("run_identity_sha256") != expected_identity_sha256:
        raise TaskCMethodRunError("existing result status identity changed")
    recorded_artifacts = status.get("artifacts")
    actual_artifacts = _artifact_records(root)
    if recorded_artifacts != actual_artifacts:
        raise TaskCMethodRunError("existing result artifact hash or inventory changed")
    existing_status = status.get("status")
    if spec.source_kind == "publication_only":
        if existing_status != "official_assets_unavailable":
            raise TaskCMethodRunError("publication-only result status changed")
        if "raw_predictions.csv" in actual_artifacts or "predictions.csv" in actual_artifacts:
            raise TaskCMethodRunError("publication-only result contains invented predictions")
    elif existing_status == _COMPLETED_STATUS:
        if gene_names is None:
            raise TaskCMethodRunError("completed result has no fixed gene set")
        raw = read_task_c_raw_predictions(root / "raw_predictions.csv", gene_names)
        if expected_raw is not None:
            try:
                pd.testing.assert_frame_equal(
                    expected_raw.reset_index(drop=True),
                    raw.reset_index(drop=True),
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise TaskCMethodRunError(
                    "existing result scientific semantics changed"
                ) from exc
        expected = normalize_task_c_predictions(raw, gene_names)
        predictions_path = root / "predictions.csv"
        prediction_snapshot = _capture_file(
            predictions_path,
            "standardized predictions",
            maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
            require_single_link=True,
        )
        try:
            observed = pd.read_csv(io.BytesIO(prediction_snapshot.payload))
        except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
            raise TaskCMethodRunError("standardized predictions are malformed") from exc
        if list(observed.columns) != [
            "source",
            "target",
            "score",
            "returned_by_method",
        ]:
            raise TaskCMethodRunError("standardized prediction columns changed")
        try:
            pd.testing.assert_frame_equal(expected, observed, check_dtype=False)
        except AssertionError as exc:
            raise TaskCMethodRunError("existing result scientific semantics changed") from exc
    elif existing_status not in _FAILED_STATUSES:
        raise TaskCMethodRunError("existing result has an unsupported status")
    return {**status, "reuse": "verified_existing_output"}


def _publish_bundle(staging: Path, output_dir: Path) -> None:
    destination = _lexical_absolute(output_dir)
    parent = destination.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700)
    _reject_symlink_components(parent, "result parent directory")
    lock = parent / f".{destination.name}.publish.lock"
    lock_descriptor: int | None = None
    try:
        lock_descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        if destination.exists() or destination.is_symlink():
            raise TaskCMethodRunError("result directory appeared before publication")
        os.rename(staging, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        lock.unlink(missing_ok=True)


def _hypersca_genes(snapshot: _Snapshot) -> tuple[str, ...]:
    payload = _parse_json(snapshot, "HyperSCA-C gene list")
    genes_raw = payload.get("genes")
    if not isinstance(genes_raw, list):
        raise TaskCMethodRunError("HyperSCA-C gene list has no ordered genes")
    genes = _canonical_text(np.asarray(genes_raw), "HyperSCA-C genes")
    if len(genes) < 2 or len(genes) > MAXIMUM_TASK_C_RUN_GENES or len(set(genes)) != len(genes):
        raise TaskCMethodRunError("HyperSCA-C gene list must contain 2 to 256 unique genes")
    return genes


def _verify_all_snapshots(
    registry_snapshot: _Snapshot,
    input_snapshot: _Snapshot | None,
    public_manifest_snapshot: _Snapshot | None,
    code_snapshots: Mapping[str, _Snapshot],
    asset_snapshots: Mapping[str, _Snapshot],
    extra_snapshots: Mapping[str, _Snapshot],
) -> None:
    _verify_snapshot(registry_snapshot, "method registry")
    if input_snapshot is not None:
        _verify_snapshot(input_snapshot, "input NPZ")
    if public_manifest_snapshot is not None:
        _verify_snapshot(public_manifest_snapshot, "public manifest")
    for name, snapshot in code_snapshots.items():
        _verify_snapshot(snapshot, f"code file {name}")
    for name, snapshot in asset_snapshots.items():
        _verify_snapshot(snapshot, f"method asset {name}")
    for name, snapshot in extra_snapshots.items():
        _verify_snapshot(snapshot, f"HyperSCA-C input {name}")


def run_task_c_method(
    *,
    method_id: str,
    input_npz: Path | None,
    output_dir: Path,
    seed: int,
    registry_path: Path,
    asset_root: Path,
    data_status: str | None = None,
    context_id: str | None = None,
    min_cells: int = 2,
    public_manifest_path: Path | None = None,
    context_values: Sequence[str] = (),
    hypersca_config_path: Path | None = None,
    gene_list_path: Path | None = None,
    device: str = "cpu",
    timeout_seconds: int | float = 86_400,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Run one registered method and publish a verified, reusable evidence bundle."""
    if not isinstance(method_id, str) or not method_id:
        raise TaskCMethodRunError("method id must be non-empty text")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TaskCMethodRunError("seed must be a non-negative integer")
    if isinstance(min_cells, bool) or not isinstance(min_cells, int) or min_cells < 1:
        raise TaskCMethodRunError("minimum cells must be a positive integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TaskCMethodRunError("timeout must be a positive number")
    if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
        raise TaskCMethodRunError("timeout must be a positive number")
    root = (project_root or Path(__file__).resolve().parents[2]).resolve(strict=True)

    registry_snapshot = _capture_file(
        registry_path,
        "method registry",
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        require_single_link=True,
    )
    try:
        registry = load_task_c_method_registry(registry_snapshot.path)
    except TaskCMethodRegistryError as exc:
        raise TaskCMethodRunError(str(exc)) from exc
    _verify_snapshot(registry_snapshot, "method registry")
    if method_id not in registry.methods:
        raise TaskCMethodRunError(f"method is not registered: {method_id}")
    spec = registry.methods[method_id]
    code_snapshots = _code_snapshots(root, spec)
    asset_snapshots = _asset_snapshots(asset_root, registry, spec)
    if asset_snapshots:
        asset_identity = _parse_json(
            asset_snapshots["bootstrap_identity.json"], "bootstrap identity"
        )
        if asset_identity.get("registry_sha256") != registry_snapshot.sha256.removeprefix(
            "sha256:"
        ) and asset_identity.get("registry_sha256") != registry_snapshot.sha256:
            raise TaskCMethodRunError(
                "prepared method assets belong to a different registry snapshot"
            )
    external_source_sha256 = _external_source_digest(asset_root, registry, spec)
    input_snapshot: _Snapshot | None = None
    public_manifest_snapshot: _Snapshot | None = None
    public_manifest_payload: dict[str, Any] | None = None
    extra_snapshots: dict[str, _Snapshot] = {}
    expression: np.ndarray | None = None
    interventions: np.ndarray | None = None
    genes: tuple[str, ...] | None = None
    hypersca_input_hashes: dict[str, str] = {}
    command: tuple[str, ...] | None = None
    if external_source_sha256 is not None:
        hypersca_input_hashes["official_source_worktree_sha256"] = (
            external_source_sha256
        )

    if spec.source_kind == "publication_only":
        if input_npz is not None or public_manifest_path is not None or context_values:
            raise TaskCMethodRunError("publication-only methods do not inspect Task C data")
        data_status = None
        context_id = None
    elif spec.method_id == "hypersca_c":
        if input_npz is not None:
            raise TaskCMethodRunError(
                "HyperSCA-C needs --context files, not one --input-npz file"
            )
        if data_status != "external_benchmark":
            raise TaskCMethodRunError("HyperSCA-C unified runs require registered public data")
        if (
            public_manifest_path is None
            or hypersca_config_path is None
            or gene_list_path is None
            or not context_values
        ):
            raise TaskCMethodRunError(
                "HyperSCA-C requires public manifest, config, gene list, and context files"
            )
        for name, path in {
            "config": hypersca_config_path,
            "gene_list": gene_list_path,
        }.items():
            snapshot = _capture_file(
                path,
                f"HyperSCA-C {name}",
                maximum_bytes=MAXIMUM_RECORD_BYTES,
                reject_private=True,
                require_single_link=True,
            )
            extra_snapshots[name] = snapshot
            hypersca_input_hashes[name] = snapshot.sha256
        genes = _hypersca_genes(extra_snapshots["gene_list"])
        parsed_contexts: dict[str, Path] = {}
        for raw_context in context_values:
            if not isinstance(raw_context, str) or "=" not in raw_context:
                raise TaskCMethodRunError("HyperSCA-C context must use name=path")
            name, raw_path = raw_context.split("=", 1)
            if name not in {"k562", "rpe1"} or not raw_path or name in parsed_contexts:
                raise TaskCMethodRunError(
                    "HyperSCA-C contexts must be unique k562=path or rpe1=path values"
                )
            parsed_contexts[name] = Path(raw_path)
        for name, context_path in sorted(parsed_contexts.items()):
            selected, captured_manifest, manifest_payload, relative = _capture_public_input(
                context_path,
                public_manifest_path,
            )
            if public_manifest_snapshot is not None and captured_manifest != public_manifest_snapshot:
                raise TaskCMethodRunError("public manifest changed between context checks")
            public_manifest_snapshot = captured_manifest
            public_manifest_payload = manifest_payload
            extra_snapshots[f"context_{name}"] = selected
            hypersca_input_hashes[f"context_{name}_sha256"] = selected.sha256
            hypersca_input_hashes[f"context_{name}_public_path"] = relative
            if _context_for_public_path(relative) != name:
                raise TaskCMethodRunError(
                    "HyperSCA-C context label disagrees with its registered public path"
                )
        assert public_manifest_payload is not None
        manifest_minimum = public_manifest_payload.get("min_cells_per_intervention")
        if isinstance(manifest_minimum, bool) or not isinstance(manifest_minimum, int):
            raise TaskCMethodRunError("public manifest minimum cell count is invalid")
        min_cells = manifest_minimum
        command = _build_hypersca_command(
            project_root=root,
            context_values=context_values,
            config_path=hypersca_config_path,
            gene_list_path=gene_list_path,
            public_manifest_path=public_manifest_path,
            output_dir=_lexical_absolute(output_dir) / "raw_method_output",
            seed=seed,
            device=device,
        )
    else:
        if input_npz is None:
            raise TaskCMethodRunError("this method requires one allowed input NPZ")
        if data_status not in _DATA_STATUSES:
            raise TaskCMethodRunError(
                "choose external_benchmark or synthetic_smoke explicitly for runnable methods"
            )
        if not isinstance(context_id, str) or not context_id.strip():
            raise TaskCMethodRunError("context id must be non-empty text")
        if data_status == "external_benchmark":
            if public_manifest_path is None:
                raise TaskCMethodRunError(
                    "registered public data require the matching public manifest"
                )
            input_snapshot, public_manifest_snapshot, public_manifest_payload, public_relative = (
                _capture_public_input(input_npz, public_manifest_path)
            )
            hypersca_input_hashes["public_relative_path"] = public_relative
            if _context_for_public_path(public_relative) != context_id:
                raise TaskCMethodRunError(
                    "context id disagrees with the registered public path"
                )
        else:
            if public_manifest_path is not None:
                raise TaskCMethodRunError(
                    "synthetic smoke data must not be presented as registered public data"
                )
            input_snapshot = _capture_synthetic_input(input_npz)
        expression, interventions, genes = _load_fixed_npz(input_snapshot)
        if data_status == "external_benchmark":
            assert public_manifest_payload is not None
            manifest_minimum = public_manifest_payload.get("min_cells_per_intervention")
            if manifest_minimum != min_cells:
                raise TaskCMethodRunError(
                    "minimum cells must match the fixed public split manifest"
                )

    staging_parent = _lexical_absolute(output_dir).parent
    if not staging_parent.exists():
        staging_parent.mkdir(parents=True, mode=0o700)
    if output_dir.exists() or output_dir.is_symlink():
        # Build the same identity before accepting reuse.
        if spec.source_kind in {"causalbench", "git"}:
            assert input_snapshot is not None
            command = build_task_c_method_command(
                spec,
                input_path=input_snapshot.path,
                output_csv=_lexical_absolute(output_dir) / "raw_runtime/worker_predictions.csv",
                asset_root=_lexical_absolute(asset_root),
                seed=seed,
                project_root=root,
            )
        command_record = _safe_command_record(command) if command else None
        expected_environment = _environment_manifest(
            spec=spec,
            registry=registry,
            registry_snapshot=registry_snapshot,
            input_snapshot=input_snapshot,
            public_manifest_snapshot=public_manifest_snapshot,
            code_snapshots=code_snapshots,
            asset_snapshots=asset_snapshots,
            seed=seed,
            data_status=data_status,
            context_id=context_id,
            min_cells=min_cells,
            command_record=command_record,
            hypersca_inputs=hypersca_input_hashes,
        )
        expected_raw: pd.DataFrame | None = None
        if spec.method_id == "mean_difference":
            assert expression is not None and interventions is not None and genes is not None
            try:
                expected_result = score_mean_difference_network(
                    expression,
                    interventions,
                    genes,
                    control_label=_CONTROL_LABEL,
                    excluded_label=_EXCLUDED_LABEL,
                    min_cells_per_intervention=min_cells,
                )
            except TaskCBenchmarkError as exc:
                raise TaskCMethodRunError(str(exc)) from exc
            expected_raw = expected_result.scores[["source", "target", "score"]].copy()
        return _validate_existing_output(
            output_dir,
            expected_environment=expected_environment,
            spec=spec,
            gene_names=genes,
            expected_raw=expected_raw,
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_lexical_absolute(output_dir).name}.staging-",
            dir=staging_parent,
        )
    )
    published = False
    try:
        if spec.source_kind in {"causalbench", "git"}:
            assert input_snapshot is not None
            command = build_task_c_method_command(
                spec,
                input_path=input_snapshot.path,
                output_csv=staging / "raw_runtime/worker_predictions.csv",
                asset_root=_lexical_absolute(asset_root),
                seed=seed,
                project_root=root,
            )
        elif spec.method_id == "hypersca_c":
            assert hypersca_config_path is not None
            assert gene_list_path is not None
            assert public_manifest_path is not None
            command = _build_hypersca_command(
                project_root=root,
                context_values=context_values,
                config_path=hypersca_config_path,
                gene_list_path=gene_list_path,
                public_manifest_path=public_manifest_path,
                output_dir=staging / "raw_method_output",
                seed=seed,
                device=device,
            )
        command_record = _safe_command_record(command) if command else None
        environment = _environment_manifest(
            spec=spec,
            registry=registry,
            registry_snapshot=registry_snapshot,
            input_snapshot=input_snapshot,
            public_manifest_snapshot=public_manifest_snapshot,
            code_snapshots=code_snapshots,
            asset_snapshots=asset_snapshots,
            seed=seed,
            data_status=data_status,
            context_id=context_id,
            min_cells=min_cells,
            command_record=command_record,
            hypersca_inputs=hypersca_input_hashes,
        )
        _write_json(staging / "environment_manifest.json", environment)

        status_name: str
        reason: str | None = None
        raw: pd.DataFrame | None = None
        if spec.source_kind == "publication_only":
            status_name = "official_assets_unavailable"
            reason = (
                "The registered report does not provide runnable official code; "
                "no prediction was inferred."
            )
        elif spec.method_id == "mean_difference":
            assert expression is not None and interventions is not None and genes is not None
            if spec.training_information != "partial_interventional":
                raise TaskCMethodRunError(
                    "mean difference is registered for partial intervention information"
                )
            try:
                result = score_mean_difference_network(
                    expression,
                    interventions,
                    genes,
                    control_label=_CONTROL_LABEL,
                    excluded_label=_EXCLUDED_LABEL,
                    min_cells_per_intervention=min_cells,
                )
            except TaskCBenchmarkError as exc:
                raise TaskCMethodRunError(str(exc)) from exc
            raw = result.scores[["source", "target", "score"]].copy()
            status_name = _COMPLETED_STATUS
        else:
            assert command is not None and genes is not None
            inner = run_isolated_method(
                command,
                output_dir=staging / "raw_runtime",
                timeout_seconds=timeout_seconds,
            )
            status_name = str(inner["status"])
            if status_name == "completed_raw_inference":
                raw_source = (
                    staging / "raw_method_output/raw_predictions.csv"
                    if spec.method_id == "hypersca_c"
                    else staging / "raw_runtime/worker_predictions.csv"
                )
                if spec.method_id == "hypersca_c":
                    snapshot = _capture_file(
                        raw_source,
                        "HyperSCA-C original predictions",
                        maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
                        require_single_link=True,
                    )
                    try:
                        original = pd.read_csv(io.BytesIO(snapshot.payload))
                    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
                        raise TaskCMethodRunError(
                            "HyperSCA-C original predictions are malformed"
                        ) from exc
                    if not {"source", "target", "score"}.issubset(original.columns):
                        raise _InvalidMethodOutput(
                            "HyperSCA-C original predictions lack source, target, or score"
                        )
                    projected = original[["source", "target", "score"]]
                    projected_path = staging / ".hypersca-projected.csv"
                    _write_new(projected_path, _csv_bytes(projected))
                    try:
                        raw = read_task_c_raw_predictions(projected_path, genes)
                    except TaskCMethodRunError as exc:
                        raise _InvalidMethodOutput(str(exc)) from exc
                    projected_path.unlink()
                else:
                    try:
                        raw = read_task_c_raw_predictions(raw_source, genes)
                    except TaskCMethodRunError as exc:
                        raise _InvalidMethodOutput(str(exc)) from exc
                status_name = _COMPLETED_STATUS

        if raw is not None:
            assert genes is not None
            try:
                standardized = normalize_task_c_predictions(raw, genes)
            except TaskCPredictionError as exc:
                raise _InvalidMethodOutput(str(exc)) from exc
            else:
                _write_new(staging / "raw_predictions.csv", _csv_bytes(raw))
                _write_new(staging / "predictions.csv", _csv_bytes(standardized))

        _verify_all_snapshots(
            registry_snapshot,
            input_snapshot,
            public_manifest_snapshot,
            code_snapshots,
            asset_snapshots,
            extra_snapshots,
        )
        if external_source_sha256 is not None and _external_source_digest(
            asset_root, registry, spec
        ) != external_source_sha256:
            raise TaskCMethodRunError("official method source changed during the run")
        artifacts = _artifact_records(staging)
        status: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "method_id": spec.method_id,
            "status": status_name,
            "run_identity_sha256": environment["run_identity_sha256"],
            "artifacts": artifacts,
        }
        if reason is not None:
            status["reason"] = reason
        _write_json(staging / "method_status.json", status)
        _publish_bundle(staging, output_dir)
        published = True
        return status
    except (TaskCMethodRunError, TaskCPredictionError) as exc:
        # Invalid method output is evidence too.  Preserve it without inventing a
        # standardized prediction table whenever the run identity is available.
        if staging.exists() and (staging / "environment_manifest.json").is_file():
            (staging / "raw_predictions.csv").unlink(missing_ok=True)
            (staging / "predictions.csv").unlink(missing_ok=True)
            environment_payload = _parse_json(
                _capture_file(
                    staging / "environment_manifest.json",
                    "environment manifest",
                    maximum_bytes=MAXIMUM_RECORD_BYTES,
                    require_single_link=True,
                ),
                "environment manifest",
            )
            status = {
                "schema_version": SCHEMA_VERSION,
                "method_id": spec.method_id,
                "status": (
                    "failed_invalid_output"
                    if isinstance(exc, (_InvalidMethodOutput, TaskCPredictionError))
                    else "failed_runtime_unavailable"
                ),
                "run_identity_sha256": environment_payload["run_identity_sha256"],
                "reason": str(exc),
                "artifacts": _artifact_records(staging),
            }
            _write_json(staging / "method_status.json", status)
            _publish_bundle(staging, output_dir)
            published = True
        raise
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
