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
    _normalized_packages,
    _validate_source_checkout,
)


SCHEMA_VERSION = "1.0"
MAXIMUM_TASK_C_RUN_GENES = 256
MAXIMUM_INPUT_BYTES = 512 * 1024 * 1024
MAXIMUM_RAW_PREDICTION_BYTES = 64 * 1024 * 1024
MAXIMUM_RECORD_BYTES = 4 * 1024 * 1024
MAXIMUM_TUNING_TRIALS = 20
_CONTROL_LABEL = "non-targeting"
_EXCLUDED_LABEL = "excluded"
_DATA_STATUSES = frozenset({"external_benchmark", "synthetic_smoke"})
_SELECTION_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "selected_trial_index",
        "selected_parameters",
        "average_precision",
        "completed_trial_count",
        "external_biological_references_used",
        "final_holdout_used",
        "method_id",
        "condition",
        "profile",
        "stage",
        "context_id",
        "direction",
        "training_and_tuning_inputs_separate",
        "evidence",
        "selection_record_sha256",
    }
)
_SELECTION_EVIDENCE_FIELDS = frozenset(
    {
        "data_status",
        "tune_input_sha256",
        "public_manifest_sha256",
        "profile_manifest_sha256",
        "gene_order_sha256",
        "config_sha256",
        "config",
        "code_sha256",
        "tuning_positive_relation_count",
        "tuning_edges",
        "tuning_edges_sha256",
        "gene_count",
        "training_input_sha256s",
        "training_profile_manifest_sha256s",
        "trials",
        "trial_metrics",
    }
)
_COMPLETED_STATUS = "completed_standardized_output"
_STATUS_SELF_FIELD = "status_content_sha256"
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
        "gene_projection",
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
_RUNTIME_CODE_CLOSURE = (
    "src/evaluation/task_c_method_run.py",
    "src/evaluation/task_c_profile_input.py",
    "scripts/run_task_c_method.py",
    "src/evaluation/task_c_predictions.py",
    "src/evaluation/task_c_runtime.py",
    "src/evaluation/task_c_method_registry.py",
    "src/evaluation/task_c_data.py",
    "src/evaluation/task_c_benchmark.py",
    "src/causal/hypersca_c.py",
    "src/causal/hypersca_c_stability.py",
    "src/causal/hypersca_c_run.py",
    "scripts/run_hypersca_c.py",
    "scripts/task_c_workers/causalbench_worker.py",
    "scripts/task_c_workers/psgrn_worker.py",
)


class TaskCMethodRunError(ValueError):
    """A method run cannot satisfy the shared Task C evidence boundary."""


class _InvalidMethodOutput(TaskCMethodRunError):
    """The external method finished but did not return an admissible relation table."""


def _capture_live_conda_environment(
    environment_name: str,
    *,
    run_command: Any = subprocess.run,
) -> dict[str, object]:
    """Read a bounded, path-free package identity from the named live environment."""

    if (
        not isinstance(environment_name, str)
        or not environment_name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in environment_name
        )
    ):
        raise TaskCMethodRunError("conda environment name is invalid")
    try:
        completed = run_command(
            ["conda", "list", "-n", environment_name, "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        raw = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        packages = _normalized_packages(raw)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
        json.JSONDecodeError,
        TaskCRuntimeError,
    ) as exc:
        raise TaskCMethodRunError(
            "cannot verify the live conda environment package identity"
        ) from exc
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment_name,
        "packages": packages,
    }
    return {
        "environment": environment_name,
        "packages": packages,
        "sha256": f"sha256:{hashlib.sha256(_json_bytes(canonical)).hexdigest()}",
    }


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


@dataclass(frozen=True)
class _DerivedInput:
    input_snapshot: _Snapshot
    manifest_snapshot: _Snapshot
    public_manifest_snapshot: _Snapshot
    public_manifest_payload: dict[str, Any]
    parent_snapshots: tuple[_Snapshot, ...]
    expression: np.ndarray
    interventions: np.ndarray
    genes: tuple[str, ...]
    environment_labels: np.ndarray | None
    run_context_id: str
    condition: str
    profile: str
    stage: str
    context_id: str | None
    direction: str | None


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


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _seal_status(payload: Mapping[str, object]) -> dict[str, object]:
    if _STATUS_SELF_FIELD in payload:
        raise TaskCMethodRunError("method status must be sealed only once")
    sealed = dict(payload)
    sealed[_STATUS_SELF_FIELD] = _canonical_payload_sha256(sealed)
    return sealed


def _validate_status_seal(payload: Mapping[str, object]) -> None:
    recorded = payload.get(_STATUS_SELF_FIELD)
    without_self = dict(payload)
    without_self.pop(_STATUS_SELF_FIELD, None)
    if recorded != _canonical_payload_sha256(without_self):
        raise TaskCMethodRunError("method status self-hash changed")


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


def _trial_candidate(snapshot: _Snapshot | None) -> tuple[int | None, dict[str, Any]]:
    if snapshot is None:
        return None, {}
    payload = _parse_json(snapshot, "trial parameter candidate")
    if set(payload) != {"schema_version", "trial_index", "parameters"}:
        raise TaskCMethodRunError("trial parameter candidate fields changed")
    trial_index = payload.get("trial_index")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or not 0 <= trial_index < MAXIMUM_TUNING_TRIALS
        or not isinstance(payload.get("parameters"), dict)
    ):
        raise TaskCMethodRunError(
            "trial candidate needs index zero to nineteen and one parameter object"
        )
    return trial_index, dict(payload["parameters"])


def _scope_for_public_path(relative: str) -> dict[str, str | None]:
    parts = Path(relative).parts
    if (
        len(parts) == 3
        and parts[0] == "within"
        and parts[1] in {"k562", "rpe1"}
        and parts[2] in {"train.npz", "tune.npz", "refit.npz"}
    ):
        return {
            "condition": "within_environment",
            "profile": "full_public",
            "stage": parts[2].removesuffix(".npz"),
            "context_id": parts[1],
            "direction": None,
        }
    if len(parts) == 3 and parts[0] == "cross" and "_to_" in parts[1]:
        stage = next(
            (
                value
                for value in ("train", "tune", "refit")
                if parts[2] in {f"source_{value}.npz", f"target_adapt_{value}.npz"}
            ),
            None,
        )
        if stage is not None:
            return {
                "condition": "cross_environment",
                "profile": "full_public",
                "stage": stage,
                "context_id": parts[1],
                "direction": parts[1],
            }
    raise TaskCMethodRunError("public input path has no fixed Task C scope")


def _sealed_trial_parameters(
    *,
    method_id: str,
    seed: int,
    trial_index: int | None,
    parameters: Mapping[str, Any],
    scope: Mapping[str, str | None],
    training_input_sha256: str | None,
    profile_manifest_sha256: str | None,
    public_manifest_sha256: str | None,
    gene_order_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trial_index": trial_index,
        "method_id": method_id,
        "condition": scope["condition"],
        "profile": scope["profile"],
        "stage": scope["stage"],
        "context_id": scope["context_id"],
        "direction": scope["direction"],
        "seed": seed,
        "training_input_sha256": training_input_sha256,
        "profile_manifest_sha256": profile_manifest_sha256,
        "public_manifest_sha256": public_manifest_sha256,
        "gene_order_sha256": gene_order_sha256,
        "parameters": dict(parameters),
    }


def _gene_order_sha256(genes: Sequence[str] | None) -> str | None:
    if genes is None:
        return None
    payload = json.dumps(
        list(genes), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validated_selection_record(
    snapshot: _Snapshot,
    *,
    method_id: str,
    scope: Mapping[str, str | None],
    seed: int,
    gene_order_sha256: str | None,
    public_manifest_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _parse_json(snapshot, "selection record")
    if set(record) != _SELECTION_RECORD_FIELDS:
        raise TaskCMethodRunError("selection record fields changed")
    recorded_sha256 = record.get("selection_record_sha256")
    unsigned = dict(record)
    unsigned.pop("selection_record_sha256", None)
    expected_sha256 = f"sha256:{hashlib.sha256(_json_bytes(unsigned)).hexdigest()}"
    evidence = record.get("evidence")
    parameters = record.get("selected_parameters")
    trial_index = record.get("selected_trial_index")
    completed_count = record.get("completed_trial_count")
    average_precision = record.get("average_precision")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or recorded_sha256 != expected_sha256
        or record.get("method_id") != method_id
        or record.get("condition") != scope["condition"]
        or record.get("profile") != scope["profile"]
        or record.get("stage") != "tune"
        or record.get("context_id") != scope["context_id"]
        or record.get("direction") != scope["direction"]
        or record.get("training_and_tuning_inputs_separate") is not True
        or record.get("external_biological_references_used") is not False
        or record.get("final_holdout_used") is not False
        or isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or not 0 <= trial_index < MAXIMUM_TUNING_TRIALS
        or isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or not 1 <= completed_count <= MAXIMUM_TUNING_TRIALS
        or isinstance(average_precision, bool)
        or not isinstance(average_precision, (int, float))
        or not math.isfinite(float(average_precision))
        or not isinstance(parameters, dict)
        or not isinstance(evidence, dict)
        or set(evidence) != _SELECTION_EVIDENCE_FIELDS
    ):
        raise TaskCMethodRunError(
            "selection record identity, policy, or selected parameters are invalid"
        )
    trials = evidence.get("trials")
    training_inputs = evidence.get("training_input_sha256s")
    tune_input = evidence.get("tune_input_sha256")
    if (
        evidence.get("data_status") != "external_benchmark"
        or evidence.get("gene_order_sha256") != gene_order_sha256
        or evidence.get("public_manifest_sha256") != public_manifest_sha256
        or not isinstance(tune_input, str)
        or not tune_input.startswith("sha256:")
        or not isinstance(training_inputs, list)
        or not training_inputs
        or any(
            not isinstance(value, str) or not value.startswith("sha256:")
            for value in training_inputs
        )
        or tune_input in training_inputs
        or not isinstance(trials, list)
        or len(trials) != completed_count
        or any(
            not isinstance(trial, dict)
            or trial.get("seed") != seed
            or trial.get("gene_order_sha256") != gene_order_sha256
            for trial in trials
        )
    ):
        raise TaskCMethodRunError(
            "selection record does not match the refit data, genes, or seed"
        )
    return record, dict(parameters)


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


def _load_fixed_npz(
    snapshot: _Snapshot,
    *,
    include_environment_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray | None]:
    buffer = io.BytesIO(snapshot.payload)
    expected_arrays = {"expression_matrix", "interventions", "var_names"}
    if include_environment_labels:
        expected_arrays.add("environment_labels")
    expected_members = {f"{name}.npy" for name in expected_arrays}
    try:
        with zipfile.ZipFile(buffer) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(expected_members) or set(names) != expected_members:
                arrays_text = ", ".join(sorted(expected_arrays))
                raise TaskCMethodRunError(
                    f"input NPZ must contain exactly {arrays_text}"
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise TaskCMethodRunError("input NPZ must not contain encrypted arrays")
            if sum(member.file_size for member in members) > MAXIMUM_INPUT_BYTES:
                raise TaskCMethodRunError("input NPZ expands beyond the allowed size")
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise TaskCMethodRunError("input NPZ contains an unexpected array")
            expression = np.asarray(archive["expression_matrix"])
            interventions_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
            environment_raw = (
                np.asarray(archive["environment_labels"])
                if include_environment_labels
                else None
            )
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
    safe_environment: np.ndarray | None = None
    if environment_raw is not None:
        environments = _canonical_text(environment_raw, "environment labels")
        if len(environments) != expression.shape[0]:
            raise TaskCMethodRunError(
                "environment labels must match the expression rows"
            )
        if set(environments) != {"k562", "rpe1"}:
            raise TaskCMethodRunError(
                "derived environment labels must contain k562 and rpe1"
            )
        safe_environment = np.asarray(environments, dtype=str)
        safe_environment.setflags(write=False)
    safe_expression.setflags(write=False)
    safe_interventions.setflags(write=False)
    return safe_expression, safe_interventions, genes, safe_environment


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
        "gene_projection",
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


def _standardize_and_concatenate(
    parents: Sequence[tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    if len(parents) != 2:
        raise TaskCMethodRunError("derived input needs exactly two public parents")
    expected_genes = parents[0][3]
    centered: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    environments: list[np.ndarray] = []
    for context_id, expression, labels, genes in parents:
        if genes != expected_genes:
            raise TaskCMethodRunError("derived parents must use the same fixed gene order")
        controls = labels == _CONTROL_LABEL
        if int(np.count_nonzero(controls)) < 2:
            raise TaskCMethodRunError(
                f"derived parent {context_id} needs at least two control cells"
            )
        control_values = np.asarray(expression[controls], dtype=np.float64)
        control_mean = control_values.mean(axis=0)
        control_scale = control_values.std(axis=0, ddof=0)
        safe_scale = np.where(control_scale <= 1e-6, 1.0, control_scale)
        centered.append(
            (np.asarray(expression, dtype=np.float64) - control_mean) / safe_scale
        )
        interventions.append(np.asarray(labels, dtype=str))
        environments.append(np.asarray([context_id] * len(labels), dtype=str))
    expression_out = np.concatenate(centered, axis=0)
    interventions_out = np.concatenate(interventions)
    environment_out = np.concatenate(environments)
    expression_out.setflags(write=False)
    interventions_out.setflags(write=False)
    environment_out.setflags(write=False)
    return expression_out, interventions_out, expected_genes, environment_out


def materialize_task_c_derived_input(
    *,
    public_manifest_path: Path,
    direction: str,
    stage: str,
    output_dir: Path,
    profile: str = "connection",
) -> dict[str, str]:
    """Build a cross-environment profile subset from registered public parents."""

    if stage not in {"train", "tune", "refit"}:
        raise TaskCMethodRunError("derived profile stage must be train, tune, or refit")
    try:
        from src.evaluation.task_c_profile_input import (
            materialize_task_c_profile_input,
        )

        return materialize_task_c_profile_input(
            public_manifest_path=public_manifest_path,
            profile=profile,
            condition="cross_environment",
            direction=direction,
            stage=stage,
            output_dir=output_dir,
        )
    except ValueError as exc:
        raise TaskCMethodRunError(str(exc)) from exc


def _validate_derived_input(
    *,
    input_path: Path,
    derived_manifest_path: Path,
    public_manifest_path: Path,
) -> _DerivedInput:
    try:
        from src.evaluation.task_c_profile_input import (
            validate_task_c_profile_input,
        )

        validated = validate_task_c_profile_input(
            input_path=input_path,
            profile_manifest_path=derived_manifest_path,
            public_manifest_path=public_manifest_path,
        )
    except ValueError as exc:
        raise TaskCMethodRunError(str(exc)) from exc

    def fixed_snapshot(value: object, label: str) -> _Snapshot:
        payload = getattr(value, "payload", None)
        if not isinstance(payload, bytes) or not payload:
            raise TaskCMethodRunError(f"{label} has no fixed validated bytes")
        return _Snapshot(
            path=Path(getattr(value, "path")),
            payload=payload,
            device=int(getattr(value, "device")),
            inode=int(getattr(value, "inode")),
            size=int(getattr(value, "size")),
            modified_ns=int(getattr(value, "modified_ns")),
            changed_ns=int(getattr(value, "changed_ns")),
            link_count=int(getattr(value, "link_count")),
        )

    input_snapshot = fixed_snapshot(validated.input_snapshot, "profile input")
    derived_manifest = fixed_snapshot(
        validated.manifest_snapshot, "profile input manifest"
    )
    public_snapshot = fixed_snapshot(validated.public_snapshot, "public manifest")
    public_payload = _parse_json(public_snapshot, "public manifest")
    captured_parents = tuple(
        fixed_snapshot(snapshot, "profile parent")
        for snapshot in validated.parent_snapshots
    )
    run_context_id = validated.direction or validated.context_id
    assert run_context_id is not None
    return _DerivedInput(
        input_snapshot=input_snapshot,
        manifest_snapshot=derived_manifest,
        public_manifest_snapshot=public_snapshot,
        public_manifest_payload=public_payload,
        parent_snapshots=captured_parents,
        expression=validated.expression,
        interventions=validated.interventions,
        genes=validated.gene_names,
        environment_labels=validated.environment_labels,
        run_context_id=run_context_id,
        condition=validated.condition,
        profile=validated.profile,
        stage=validated.stage,
        context_id=validated.context_id,
        direction=validated.direction,
    )


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
    profile_input_path: Path | None = None,
    profile_manifest_path: Path | None = None,
) -> tuple[str, ...]:
    context_arguments = tuple(
        item
        for value in context_values
        for item in ("--context", value)
    )
    profile_arguments = (
        (
            "--profile-input",
            str(profile_input_path),
            "--profile-manifest",
            str(profile_manifest_path),
        )
        if profile_input_path is not None and profile_manifest_path is not None
        else ()
    )
    return (
        sys.executable,
        str(project_root / "scripts/run_hypersca_c.py"),
        *context_arguments,
        *profile_arguments,
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
        "--profile-input",
        "--profile-manifest",
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
    del spec
    paths = [project_root / relative for relative in _RUNTIME_CODE_CLOSURE]
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
    derived_input_manifest_snapshot: _Snapshot | None,
    public_manifest_snapshot: _Snapshot | None,
    code_snapshots: Mapping[str, _Snapshot],
    asset_snapshots: Mapping[str, _Snapshot],
    seed: int,
    data_status: str | None,
    context_id: str | None,
    min_cells: int,
    command_record: Mapping[str, object] | None,
    hypersca_inputs: Mapping[str, str],
    trial_parameters: Mapping[str, object],
    trial_parameters_sha256: str,
    selection_record: Mapping[str, object] | None,
) -> tuple[dict[str, object], str]:
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method": _registered_method_record(spec, registry),
        "registry_sha256": registry_snapshot.sha256,
        "input_sha256": input_snapshot.sha256 if input_snapshot else None,
        "derived_input_manifest_sha256": (
            derived_input_manifest_snapshot.sha256
            if derived_input_manifest_snapshot
            else None
        ),
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
        "trial_parameters": {
            "sha256": trial_parameters_sha256,
            "content": dict(trial_parameters),
        },
        "selection_record": (
            dict(selection_record) if selection_record is not None else None
        ),
    }
    digest = f"sha256:{hashlib.sha256(_json_bytes(identity)).hexdigest()}"
    return identity, digest


def _environment_manifest(
    *,
    spec: TaskCMethodSpec,
    registry: TaskCMethodRegistry,
    registry_snapshot: _Snapshot,
    input_snapshot: _Snapshot | None,
    derived_input_manifest_snapshot: _Snapshot | None,
    public_manifest_snapshot: _Snapshot | None,
    code_snapshots: Mapping[str, _Snapshot],
    asset_snapshots: Mapping[str, _Snapshot],
    seed: int,
    data_status: str | None,
    context_id: str | None,
    min_cells: int,
    command_record: Mapping[str, object] | None,
    hypersca_inputs: Mapping[str, str],
    trial_parameters: Mapping[str, object],
    trial_parameters_sha256: str,
    selection_record: Mapping[str, object] | None,
) -> dict[str, object]:
    identity, identity_sha256 = _run_identity(
        spec=spec,
        registry=registry,
        registry_snapshot=registry_snapshot,
        input_snapshot=input_snapshot,
        derived_input_manifest_snapshot=derived_input_manifest_snapshot,
        public_manifest_snapshot=public_manifest_snapshot,
        code_snapshots=code_snapshots,
        asset_snapshots=asset_snapshots,
        seed=seed,
        data_status=data_status,
        context_id=context_id,
        min_cells=min_cells,
        command_record=command_record,
        hypersca_inputs=hypersca_inputs,
        trial_parameters=trial_parameters,
        trial_parameters_sha256=trial_parameters_sha256,
        selection_record=selection_record,
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
        "derived_input_manifest": (
            {
                "sha256": derived_input_manifest_snapshot.sha256,
                "size_bytes": derived_input_manifest_snapshot.size,
            }
            if derived_input_manifest_snapshot
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
        "trial_parameters": {
            "sha256": trial_parameters_sha256,
            "content": dict(trial_parameters),
        },
        "selection_record": (
            dict(selection_record) if selection_record is not None else None
        ),
    }


def _read_existing_json(path: Path, label: str) -> tuple[dict[str, Any], _Snapshot]:
    snapshot = _capture_file(
        path,
        label,
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        require_single_link=True,
    )
    return _parse_json(snapshot, label), snapshot


def _validate_hypersca_inner_bundle(
    output_dir: Path,
    *,
    fixed_inputs: Mapping[str, object],
    recompute: bool,
) -> None:
    from src.causal.hypersca_c_run import (
        recompute_hypersca_c_output_bundle,
        validate_hypersca_c_output_bundle,
    )

    validator = (
        recompute_hypersca_c_output_bundle
        if recompute
        else validate_hypersca_c_output_bundle
    )
    validator(output_dir, **fixed_inputs)


def _validate_existing_output(
    output_dir: Path,
    *,
    expected_environment: Mapping[str, object],
    spec: TaskCMethodSpec,
    gene_names: Sequence[str] | None,
    expected_raw: pd.DataFrame | None = None,
    hypersca_validation_inputs: Mapping[str, object] | None = None,
    recompute_hypersca: bool = False,
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
    _validate_status_seal(status)
    required_status_fields = {
        "schema_version",
        "method_id",
        "status",
        "run_identity_sha256",
        "artifacts",
        "inner_status",
        "status_origin",
        "trial_parameters_sha256",
        _STATUS_SELF_FIELD,
    }
    if frozenset(status) not in {
        frozenset(required_status_fields),
        frozenset(required_status_fields | {"reason"}),
    }:
        raise TaskCMethodRunError("existing method status fields changed")
    expected_identity_sha256 = str(expected_environment["run_identity_sha256"])
    if environment.get("run_identity_sha256") != expected_identity_sha256:
        raise TaskCMethodRunError("existing result has a different run identity")
    if environment.get("method_id") != spec.method_id or status.get("method_id") != spec.method_id:
        raise TaskCMethodRunError("existing result method identity changed")
    if status.get("run_identity_sha256") != expected_identity_sha256:
        raise TaskCMethodRunError("existing result status identity changed")
    expected_trial = expected_environment.get("trial_parameters")
    if not isinstance(expected_trial, dict):
        raise TaskCMethodRunError("expected trial parameter identity is malformed")
    trial_snapshot = _capture_file(
        root / "trial_parameters.json",
        "sealed trial parameters",
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        require_single_link=True,
    )
    if (
        trial_snapshot.sha256 != expected_trial.get("sha256")
        or _parse_json(trial_snapshot, "sealed trial parameters")
        != expected_trial.get("content")
        or status.get("trial_parameters_sha256") != trial_snapshot.sha256
    ):
        raise TaskCMethodRunError("sealed trial parameter identity changed")
    recorded_artifacts = status.get("artifacts")
    actual_artifacts = _artifact_records(root)
    if recorded_artifacts != actual_artifacts:
        raise TaskCMethodRunError("existing result artifact hash or inventory changed")
    existing_status = status.get("status")
    top_level = {entry.name for entry in root.iterdir()}
    if spec.source_kind == "publication_only":
        if existing_status != "official_assets_unavailable":
            raise TaskCMethodRunError("publication-only result status changed")
        if "raw_predictions.csv" in actual_artifacts or "predictions.csv" in actual_artifacts:
            raise TaskCMethodRunError("publication-only result contains invented predictions")
        if top_level != {
            "method_status.json",
            "environment_manifest.json",
            "trial_parameters.json",
        }:
            raise TaskCMethodRunError("publication-only result has unexpected files")
        if status.get("inner_status") is not None or status.get(
            "status_origin"
        ) != "publication_record":
            raise TaskCMethodRunError("publication-only status origin changed")
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
        observed["source"] = observed["source"].astype("string")
        observed["target"] = observed["target"].astype("string")
        try:
            pd.testing.assert_frame_equal(expected, observed, check_dtype=True)
        except AssertionError as exc:
            raise TaskCMethodRunError("existing result scientific semantics changed") from exc
        if spec.method_id == "mean_difference":
            expected_top = {
                "method_status.json",
                "environment_manifest.json",
                "raw_predictions.csv",
                "predictions.csv",
                "trial_parameters.json",
            }
        elif spec.method_id == "hypersca_c":
            expected_top = {
                "method_status.json",
                "environment_manifest.json",
                "raw_predictions.csv",
                "predictions.csv",
                "raw_runtime",
                "raw_method_output",
                "trial_parameters.json",
            }
        else:
            expected_top = {
                "method_status.json",
                "environment_manifest.json",
                "raw_predictions.csv",
                "predictions.csv",
                "raw_runtime",
                "trial_parameters.json",
            }
        if top_level != expected_top:
            raise TaskCMethodRunError("completed result file set changed")
        if spec.method_id == "mean_difference" and (
            status.get("inner_status") is not None
            or status.get("status_origin") != "local"
        ):
            raise TaskCMethodRunError("mean-difference status origin changed")
    elif existing_status not in _FAILED_STATUSES:
        raise TaskCMethodRunError("existing result has an unsupported status")
    else:
        if {"raw_predictions.csv", "predictions.csv"} & top_level:
            raise TaskCMethodRunError("failed result must not contain prediction files")
        expected_failed_top = {
            "method_status.json",
            "environment_manifest.json",
            "trial_parameters.json",
        }
        if status.get("inner_status") is not None:
            expected_failed_top.add("raw_runtime")
        if spec.method_id == "hypersca_c" and "raw_method_output" in top_level:
            expected_failed_top.add("raw_method_output")
        if top_level != expected_failed_top:
            raise TaskCMethodRunError("failed result file set changed")
        if spec.method_id == "hypersca_c" and "raw_method_output" in top_level:
            nested_entries = list((root / "raw_method_output").rglob("*"))
            if any(path.is_dir() for path in nested_entries):
                raise TaskCMethodRunError(
                    "failed HyperSCA-C result has unexpected evidence directories"
                )
            nested_names = {
                path.relative_to(root / "raw_method_output").as_posix()
                for path in nested_entries
                if not path.is_dir()
            }
            if not nested_names <= {
                "raw_predictions.csv",
                "fit_summary.json",
                "method_status.json",
                "run_manifest.json",
            }:
                raise TaskCMethodRunError(
                    "failed HyperSCA-C result has unexpected evidence files"
                )
    if spec.source_kind in {"causalbench", "git"} or spec.method_id == "hypersca_c":
        recorded_inner_status = status.get("inner_status")
        if recorded_inner_status is None:
            if not (
                existing_status == "failed_runtime_unavailable"
                and status.get("status_origin") == "post_run_validation"
                and "raw_runtime" not in top_level
            ):
                raise TaskCMethodRunError("outer failure has no inner runtime evidence")
            return {**status, "reuse": "verified_existing_output"}
        runtime_entries = list((root / "raw_runtime").rglob("*"))
        if any(path.is_dir() for path in runtime_entries):
            raise TaskCMethodRunError("inner runtime evidence has unexpected directories")
        runtime_names = {
            path.relative_to(root / "raw_runtime").as_posix()
            for path in runtime_entries
            if not path.is_dir()
        }
        expected_runtime_names = {"method_status.json", "resource_usage.json"}
        if spec.method_id != "hypersca_c":
            if existing_status == _COMPLETED_STATUS:
                expected_runtime_names.add("worker_predictions.csv")
            elif "worker_predictions.csv" in runtime_names:
                expected_runtime_names.add("worker_predictions.csv")
        if runtime_names != expected_runtime_names:
            raise TaskCMethodRunError("inner runtime evidence file set changed")
        inner, _ = _read_existing_json(
            root / "raw_runtime/method_status.json", "inner method status"
        )
        inner_status = inner.get("status")
        if status.get("inner_status") != inner_status:
            raise TaskCMethodRunError("recorded inner status differs from inner evidence")
        if existing_status == _COMPLETED_STATUS:
            if (
                inner_status != "completed_raw_inference"
                or status.get("status_origin") != "standardization"
            ):
                raise TaskCMethodRunError(
                    "completed result requires inner completed raw inference status"
                )
        elif existing_status == "failed_invalid_output":
            if inner_status == "completed_raw_inference":
                expected_origin = "standardization"
            elif inner_status == "failed_invalid_output":
                expected_origin = "inner_runtime"
            else:
                raise TaskCMethodRunError(
                    "invalid output has no corresponding inner status"
                )
            if status.get("status_origin") != expected_origin:
                raise TaskCMethodRunError("invalid output origin differs from inner status")
        elif inner_status == existing_status:
            if status.get("status_origin") != "inner_runtime":
                raise TaskCMethodRunError("outer failure origin differs from inner status")
        elif not (
            existing_status == "failed_runtime_unavailable"
            and inner_status == "completed_raw_inference"
            and status.get("status_origin") == "post_run_validation"
        ):
            raise TaskCMethodRunError("outer failure status differs from inner status")
    if spec.method_id == "hypersca_c" and existing_status == _COMPLETED_STATUS:
        if hypersca_validation_inputs is None:
            raise TaskCMethodRunError(
                "HyperSCA-C validation requires the caller's fixed inputs"
            )
        try:
            _validate_hypersca_inner_bundle(
                root / "raw_method_output",
                fixed_inputs=hypersca_validation_inputs,
                recompute=recompute_hypersca,
            )
        except Exception as exc:
            raise TaskCMethodRunError(
                "HyperSCA-C inner scientific evidence is invalid"
            ) from exc
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


def validate_task_c_method_output_bundle(
    *,
    output_dir: Path,
    input_npz: Path,
    registry_path: Path,
    asset_root: Path,
    public_manifest_path: Path,
    derived_input_manifest_path: Path | None = None,
    hypersca_config_path: Path | None = None,
    gene_list_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Rebuild a completed candidate from its actual fixed training inputs.

    The sealed candidate record supplies only its pre-run index, parameters, method,
    context, and seed.  The caller must supply the real public input files; normal
    reuse validation then recomputes scientific output and every evidence hash.
    """

    root = _lexical_absolute(output_dir)
    sealed_snapshot = _capture_file(
        root / "trial_parameters.json",
        "sealed trial parameters",
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        require_single_link=True,
    )
    sealed = _parse_json(sealed_snapshot, "sealed trial parameters")
    environment_snapshot = _capture_file(
        root / "environment_manifest.json",
        "environment manifest",
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        require_single_link=True,
    )
    environment = _parse_json(environment_snapshot, "environment manifest")
    if set(sealed) != {
        "schema_version",
        "trial_index",
        "method_id",
        "condition",
        "profile",
        "stage",
        "context_id",
        "direction",
        "seed",
        "training_input_sha256",
        "profile_manifest_sha256",
        "public_manifest_sha256",
        "gene_order_sha256",
        "parameters",
    }:
        raise TaskCMethodRunError("sealed trial parameter fields changed")
    if (
        sealed.get("schema_version") != SCHEMA_VERSION
        or sealed.get("stage") != "train"
        or isinstance(sealed.get("trial_index"), bool)
        or not isinstance(sealed.get("trial_index"), int)
        or int(sealed["trial_index"]) < 0
        or not isinstance(sealed.get("parameters"), dict)
        or environment.get("data_status") != "external_benchmark"
    ):
        raise TaskCMethodRunError("completed candidate is not a formal train-stage trial")
    min_cells = environment.get("min_cells")
    if isinstance(min_cells, bool) or not isinstance(min_cells, int):
        raise TaskCMethodRunError("completed candidate minimum cell count changed")
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "trial_index": sealed["trial_index"],
        "parameters": sealed["parameters"],
    }
    temporary_parent = _lexical_absolute(output_dir).parent
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.verify-", dir=temporary_parent
    ) as temporary_name:
        candidate_path = Path(temporary_name) / "trial_candidate.json"
        _write_new(candidate_path, _json_bytes(candidate))
        return run_task_c_method(
            method_id=str(sealed["method_id"]),
            input_npz=input_npz,
            output_dir=root,
            seed=int(sealed["seed"]),
            registry_path=registry_path,
            asset_root=asset_root,
            data_status="external_benchmark",
            context_id=str(sealed["context_id"]),
            min_cells=min_cells,
            public_manifest_path=public_manifest_path,
            derived_input_manifest_path=derived_input_manifest_path,
            hypersca_config_path=hypersca_config_path,
            gene_list_path=gene_list_path,
            device=(
                "cuda"
                if isinstance(environment.get("command"), dict)
                and "cuda" in environment["command"].get("template", [])
                else "cpu"
            ),
            trial_parameters_path=candidate_path,
            project_root=project_root,
        )


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
    derived_input_manifest_path: Path | None = None,
    context_values: Sequence[str] = (),
    hypersca_config_path: Path | None = None,
    gene_list_path: Path | None = None,
    device: str = "cpu",
    timeout_seconds: int | float = 86_400,
    trial_parameters_path: Path | None = None,
    selection_record_path: Path | None = None,
    selection_status_path: Path | None = None,
    selection_tune_input_path: Path | None = None,
    selection_tune_profile_manifest_path: Path | None = None,
    selection_config_path: Path | None = None,
    selection_trial_directories: Sequence[Path] = (),
    selection_trial_input_bindings: Mapping[Path, Path] | None = None,
    selection_trial_profile_bindings: Mapping[Path, Path] | None = None,
    selection_trial_hypersca_configs: Mapping[Path, Path] | None = None,
    selection_trial_gene_lists: Mapping[Path, Path] | None = None,
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
    trial_candidate_snapshot = (
        _capture_file(
            trial_parameters_path,
            "trial parameter candidate",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
            reject_private=True,
            require_single_link=True,
        )
        if trial_parameters_path is not None
        else None
    )
    selection_record_snapshot = (
        _capture_file(
            selection_record_path,
            "selection record",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
            reject_private=True,
            require_single_link=True,
        )
        if selection_record_path is not None
        else None
    )
    if selection_record_snapshot is None and any(
        (
            selection_status_path,
            selection_tune_input_path,
            selection_tune_profile_manifest_path,
            selection_config_path,
            selection_trial_directories,
            selection_trial_input_bindings,
            selection_trial_profile_bindings,
            selection_trial_hypersca_configs,
            selection_trial_gene_lists,
        )
    ):
        raise TaskCMethodRunError("selection replay evidence needs one selection record")
    if trial_candidate_snapshot is not None and selection_record_snapshot is not None:
        raise TaskCMethodRunError(
            "choose either trial parameters or one selection record, not both"
        )
    trial_index, candidate_parameters = _trial_candidate(trial_candidate_snapshot)

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
    live_environment: dict[str, object] | None = None
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
        environment_name = (
            str(registry.causalbench["environment"])
            if spec.source_kind == "causalbench"
            else str(spec.environment)
        )
        expected_environment = _parse_json(
            asset_snapshots[f"environment_manifests/{environment_name}.json"],
            "prepared environment manifest",
        )
        live_environment = _capture_live_conda_environment(environment_name)
        if (
            expected_environment.get("environment") != environment_name
            or expected_environment.get("packages") != live_environment["packages"]
        ):
            raise TaskCMethodRunError(
                "live conda environment differs from the prepared package manifest"
            )
    external_source_sha256 = _external_source_digest(asset_root, registry, spec)
    input_snapshot: _Snapshot | None = None
    public_manifest_snapshot: _Snapshot | None = None
    derived_input_manifest_snapshot: _Snapshot | None = None
    public_manifest_payload: dict[str, Any] | None = None
    extra_snapshots: dict[str, _Snapshot] = {}
    if trial_candidate_snapshot is not None:
        extra_snapshots["trial_parameter_candidate"] = trial_candidate_snapshot
    if selection_record_snapshot is not None:
        extra_snapshots["selection_record"] = selection_record_snapshot
    expression: np.ndarray | None = None
    interventions: np.ndarray | None = None
    genes: tuple[str, ...] | None = None
    hypersca_input_hashes: dict[str, str] = {}
    command: tuple[str, ...] | None = None
    hypersca_validation_inputs: dict[str, object] | None = None
    trial_scope: dict[str, str | None]
    if external_source_sha256 is not None:
        hypersca_input_hashes["official_source_worktree_sha256"] = (
            external_source_sha256
        )
    if live_environment is not None:
        hypersca_input_hashes["live_environment_sha256"] = str(
            live_environment["sha256"]
        )

    if spec.source_kind == "publication_only":
        if (
            input_npz is not None
            or public_manifest_path is not None
            or derived_input_manifest_path is not None
            or context_values
        ):
            raise TaskCMethodRunError("publication-only methods do not inspect Task C data")
        data_status = None
        context_id = None
        trial_scope = {
            "condition": "publication_record",
            "profile": "not_runnable",
            "stage": "not_runnable",
            "context_id": None,
            "direction": None,
        }
    elif spec.method_id == "hypersca_c":
        profile_mode = input_npz is not None or derived_input_manifest_path is not None
        if profile_mode and (
            input_npz is None
            or derived_input_manifest_path is None
            or bool(context_values)
        ):
            raise TaskCMethodRunError(
                "HyperSCA-C profile runs need one input and profile manifest, without contexts"
            )
        if not profile_mode and (input_npz is not None or not context_values):
            raise TaskCMethodRunError(
                "HyperSCA-C needs registered context files or one validated profile input"
            )
        if data_status != "external_benchmark":
            raise TaskCMethodRunError("HyperSCA-C unified runs require registered public data")
        if (
            public_manifest_path is None
            or hypersca_config_path is None
            or gene_list_path is None
        ):
            raise TaskCMethodRunError(
                "HyperSCA-C requires public manifest, config, and gene list"
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
        if profile_mode:
            assert input_npz is not None and derived_input_manifest_path is not None
            derived = _validate_derived_input(
                input_path=input_npz,
                derived_manifest_path=derived_input_manifest_path,
                public_manifest_path=public_manifest_path,
            )
            if genes != derived.genes:
                raise TaskCMethodRunError(
                    "HyperSCA-C gene list must match the profile gene order"
                )
            if context_id != derived.run_context_id:
                raise TaskCMethodRunError(
                    "HyperSCA-C context id must match the profile context or direction"
                )
            input_snapshot = derived.input_snapshot
            derived_input_manifest_snapshot = derived.manifest_snapshot
            public_manifest_snapshot = derived.public_manifest_snapshot
            public_manifest_payload = derived.public_manifest_payload
            for index, parent_snapshot in enumerate(derived.parent_snapshots):
                extra_snapshots[f"profile_parent_{index + 1}"] = parent_snapshot
            extra_snapshots["profile_manifest"] = derived.manifest_snapshot
            hypersca_input_hashes["profile_context"] = derived.run_context_id
            trial_scope = {
                "condition": derived.condition,
                "profile": derived.profile,
                "stage": derived.stage,
                "context_id": derived.run_context_id,
                "direction": derived.direction,
            }
        else:
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
            trial_scope = {
                "condition": "multi_context",
                "profile": "full_public",
                "stage": "refit",
                "context_id": None,
                "direction": None,
            }
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
            profile_input_path=(input_npz if profile_mode else None),
            profile_manifest_path=(
                derived_input_manifest_path if profile_mode else None
            ),
        )
        hypersca_validation_inputs = {
            "context_values": tuple(context_values),
            "profile_input_path": input_npz if profile_mode else None,
            "profile_manifest_path": (
                derived_input_manifest_path if profile_mode else None
            ),
            "config_path": hypersca_config_path,
            "gene_list_path": gene_list_path,
            "public_manifest_path": public_manifest_path,
            "seed": seed,
            "device": device,
        }
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
            if derived_input_manifest_path is not None:
                derived = _validate_derived_input(
                    input_path=input_npz,
                    derived_manifest_path=derived_input_manifest_path,
                    public_manifest_path=public_manifest_path,
                )
                input_snapshot = derived.input_snapshot
                derived_input_manifest_snapshot = derived.manifest_snapshot
                public_manifest_snapshot = derived.public_manifest_snapshot
                public_manifest_payload = derived.public_manifest_payload
                expression = derived.expression
                interventions = derived.interventions
                genes = derived.genes
                for index, parent_snapshot in enumerate(derived.parent_snapshots):
                    extra_snapshots[f"profile_parent_{index + 1}"] = parent_snapshot
                extra_snapshots["derived_input_manifest"] = derived.manifest_snapshot
                hypersca_input_hashes["profile_context"] = derived.run_context_id
                if context_id != derived.run_context_id:
                    raise TaskCMethodRunError(
                        "context id must equal the recorded profile context or direction"
                    )
                trial_scope = {
                    "condition": derived.condition,
                    "profile": derived.profile,
                    "stage": derived.stage,
                    "context_id": derived.run_context_id,
                    "direction": derived.direction,
                }
            else:
                input_snapshot, public_manifest_snapshot, public_manifest_payload, public_relative = (
                    _capture_public_input(input_npz, public_manifest_path)
                )
                hypersca_input_hashes["public_relative_path"] = public_relative
                if _context_for_public_path(public_relative) != context_id:
                    raise TaskCMethodRunError(
                        "context id disagrees with the registered public path"
                    )
                trial_scope = _scope_for_public_path(public_relative)
        else:
            if public_manifest_path is not None or derived_input_manifest_path is not None:
                raise TaskCMethodRunError(
                    "synthetic smoke data must not be presented as registered public data"
                )
            input_snapshot = _capture_synthetic_input(input_npz)
            trial_scope = {
                "condition": "synthetic_smoke",
                "profile": "synthetic_smoke",
                "stage": "synthetic_smoke",
                "context_id": context_id,
                "direction": None,
            }
        if expression is None:
            expression, interventions, genes, environment_labels = _load_fixed_npz(
                input_snapshot
            )
            if environment_labels is not None:
                raise TaskCMethodRunError(
                    "within-environment input must contain exactly three arrays"
                )
        if data_status == "external_benchmark":
            assert public_manifest_payload is not None
            manifest_minimum = public_manifest_payload.get("min_cells_per_intervention")
            if manifest_minimum != min_cells:
                raise TaskCMethodRunError(
                    "minimum cells must match the fixed public split manifest"
                )

    gene_order_sha256 = _gene_order_sha256(genes)
    selection_record_identity: dict[str, object] | None = None
    if selection_record_snapshot is not None:
        if data_status != "external_benchmark" or trial_scope["stage"] != "refit":
            raise TaskCMethodRunError(
                "a selection record may only authorize the public refit stage"
            )
        if (
            selection_status_path is None
            or selection_tune_input_path is None
            or selection_config_path is None
            or not selection_trial_directories
            or selection_trial_input_bindings is None
            or selection_trial_profile_bindings is None
            or public_manifest_path is None
        ):
            raise TaskCMethodRunError(
                "selection record requires completed status, actual tune input, tuning config, and every trial binding"
            )
        try:
            from src.evaluation.task_c_tuning import (
                TaskCTuningError,
                validate_task_c_selection_record,
            )

            validate_task_c_selection_record(
                record_path=selection_record_snapshot.path,
                status_path=selection_status_path,
                tune_input_path=selection_tune_input_path,
                tune_profile_manifest_path=selection_tune_profile_manifest_path,
                public_manifest_path=public_manifest_path,
                config_path=selection_config_path,
                trial_directories=selection_trial_directories,
                trial_input_bindings=selection_trial_input_bindings,
                trial_profile_bindings=selection_trial_profile_bindings,
                registry_path=registry_path,
                asset_root=asset_root,
                trial_hypersca_configs=selection_trial_hypersca_configs,
                trial_gene_lists=selection_trial_gene_lists,
                project_root=root,
            )
        except TaskCTuningError as exc:
            raise TaskCMethodRunError(f"selection record replay failed: {exc}") from exc
        selection_record, candidate_parameters = _validated_selection_record(
            selection_record_snapshot,
            method_id=spec.method_id,
            scope=trial_scope,
            seed=seed,
            gene_order_sha256=gene_order_sha256,
            public_manifest_sha256=(
                public_manifest_snapshot.sha256 if public_manifest_snapshot else None
            ),
        )
        selection_record_identity = {
            "sha256": selection_record_snapshot.sha256,
            "record_sha256": selection_record["selection_record_sha256"],
            "content": selection_record,
        }

    if trial_candidate_snapshot is not None:
        if data_status == "external_benchmark" and trial_scope["stage"] != "train":
            raise TaskCMethodRunError(
                "formal trial candidates must use the public train stage"
            )
        if spec.method_id != "hypersca_c" and candidate_parameters:
            raise TaskCMethodRunError(
                "this fixed comparison method does not accept tunable parameters"
            )
        if spec.method_id == "hypersca_c":
            assert "config" in extra_snapshots
            if candidate_parameters != _parse_json(
                extra_snapshots["config"], "HyperSCA-C config"
            ):
                raise TaskCMethodRunError(
                    "HyperSCA-C trial parameters must exactly equal the run config"
                )
    if selection_record_snapshot is not None:
        if spec.method_id != "hypersca_c" and candidate_parameters:
            raise TaskCMethodRunError(
                "the selected fixed comparison method has no tunable parameters"
            )
        if spec.method_id == "hypersca_c":
            assert "config" in extra_snapshots
            if candidate_parameters != _parse_json(
                extra_snapshots["config"], "HyperSCA-C config"
            ):
                raise TaskCMethodRunError(
                    "selected HyperSCA-C parameters must exactly equal the refit config"
                )
    sealed_trial_parameters = _sealed_trial_parameters(
        method_id=spec.method_id,
        seed=seed,
        trial_index=trial_index,
        parameters=candidate_parameters,
        scope=trial_scope,
        training_input_sha256=input_snapshot.sha256 if input_snapshot else None,
        profile_manifest_sha256=(
            derived_input_manifest_snapshot.sha256
            if derived_input_manifest_snapshot
            else None
        ),
        public_manifest_sha256=(
            public_manifest_snapshot.sha256 if public_manifest_snapshot else None
        ),
        gene_order_sha256=gene_order_sha256,
    )
    sealed_trial_parameters_bytes = _json_bytes(sealed_trial_parameters)
    sealed_trial_parameters_sha256 = (
        f"sha256:{hashlib.sha256(sealed_trial_parameters_bytes).hexdigest()}"
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
            derived_input_manifest_snapshot=derived_input_manifest_snapshot,
            public_manifest_snapshot=public_manifest_snapshot,
            code_snapshots=code_snapshots,
            asset_snapshots=asset_snapshots,
            seed=seed,
            data_status=data_status,
            context_id=context_id,
            min_cells=min_cells,
            command_record=command_record,
            hypersca_inputs=hypersca_input_hashes,
            trial_parameters=sealed_trial_parameters,
            trial_parameters_sha256=sealed_trial_parameters_sha256,
            selection_record=selection_record_identity,
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
        reused_status = _validate_existing_output(
            output_dir,
            expected_environment=expected_environment,
            spec=spec,
            gene_names=genes,
            expected_raw=expected_raw,
            hypersca_validation_inputs=hypersca_validation_inputs,
            recompute_hypersca=(spec.method_id == "hypersca_c"),
        )
        if live_environment is not None:
            refreshed_environment = _capture_live_conda_environment(
                str(live_environment["environment"])
            )
            if refreshed_environment != live_environment:
                raise TaskCMethodRunError(
                    "live conda environment changed during result reuse"
                )
        return reused_status

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_lexical_absolute(output_dir).name}.staging-",
            dir=staging_parent,
        )
    )
    fixed_input_root: Path | None = None
    published = False
    inner_status_name: str | None = None
    status_origin = "local"
    try:
        fixed_input_path: Path | None = None
        execution_hypersca_validation_inputs = hypersca_validation_inputs
        if derived_input_manifest_snapshot is not None:
            assert input_snapshot is not None
            fixed_input_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{_lexical_absolute(output_dir).name}.fixed-inputs-",
                    dir=staging_parent,
                )
            )
            fixed_input_path = fixed_input_root / "profile_input.npz"
            _write_new(fixed_input_path, input_snapshot.payload)
        if spec.source_kind in {"causalbench", "git"}:
            assert input_snapshot is not None
            command = build_task_c_method_command(
                spec,
                input_path=fixed_input_path or input_snapshot.path,
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
                profile_input_path=(fixed_input_path if profile_mode else None),
                profile_manifest_path=(
                    derived_input_manifest_path if profile_mode else None
                ),
            )
            if profile_mode:
                assert fixed_input_path is not None
                execution_hypersca_validation_inputs = {
                    **(hypersca_validation_inputs or {}),
                    "profile_input_path": fixed_input_path,
                }
        command_record = _safe_command_record(command) if command else None
        environment = _environment_manifest(
            spec=spec,
            registry=registry,
            registry_snapshot=registry_snapshot,
            input_snapshot=input_snapshot,
            derived_input_manifest_snapshot=derived_input_manifest_snapshot,
            public_manifest_snapshot=public_manifest_snapshot,
            code_snapshots=code_snapshots,
            asset_snapshots=asset_snapshots,
            seed=seed,
            data_status=data_status,
            context_id=context_id,
            min_cells=min_cells,
            command_record=command_record,
            hypersca_inputs=hypersca_input_hashes,
            trial_parameters=sealed_trial_parameters,
            trial_parameters_sha256=sealed_trial_parameters_sha256,
            selection_record=selection_record_identity,
        )
        _write_new(staging / "trial_parameters.json", sealed_trial_parameters_bytes)
        _write_json(staging / "environment_manifest.json", environment)

        status_name: str
        reason: str | None = None
        raw: pd.DataFrame | None = None
        if spec.source_kind == "publication_only":
            status_name = "official_assets_unavailable"
            status_origin = "publication_record"
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
            status_origin = "local"
        else:
            assert command is not None and genes is not None
            inner = run_isolated_method(
                command,
                output_dir=staging / "raw_runtime",
                timeout_seconds=timeout_seconds,
            )
            status_name = str(inner["status"])
            inner_status_name = status_name
            status_origin = "inner_runtime"
            if status_name == "completed_raw_inference":
                raw_source = (
                    staging / "raw_method_output/raw_predictions.csv"
                    if spec.method_id == "hypersca_c"
                    else staging / "raw_runtime/worker_predictions.csv"
                )
                if spec.method_id == "hypersca_c":
                    try:
                        from src.causal.hypersca_c_run import (
                            validate_hypersca_c_output_bundle,
                        )

                        validate_hypersca_c_output_bundle(
                            staging / "raw_method_output",
                            **(execution_hypersca_validation_inputs or {}),
                        )
                        snapshot = _capture_file(
                            raw_source,
                            "HyperSCA-C original predictions",
                            maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
                            require_single_link=True,
                        )
                        original = pd.read_csv(io.BytesIO(snapshot.payload))
                        if not {"source", "target", "score"}.issubset(
                            original.columns
                        ):
                            raise TaskCMethodRunError(
                                "HyperSCA-C original predictions lack source, target, or score"
                            )
                        projected = original[["source", "target", "score"]]
                        projected_path = staging / ".hypersca-projected.csv"
                        _write_new(projected_path, _csv_bytes(projected))
                        try:
                            raw = read_task_c_raw_predictions(projected_path, genes)
                        finally:
                            projected_path.unlink(missing_ok=True)
                    except Exception as exc:
                        raise _InvalidMethodOutput(
                            f"HyperSCA-C inner scientific evidence is invalid: {exc}"
                        ) from exc
                else:
                    try:
                        raw = read_task_c_raw_predictions(raw_source, genes)
                    except TaskCMethodRunError as exc:
                        raise _InvalidMethodOutput(str(exc)) from exc
                status_name = _COMPLETED_STATUS
                status_origin = "standardization"

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
        if live_environment is not None:
            refreshed_environment = _capture_live_conda_environment(
                str(live_environment["environment"])
            )
            if refreshed_environment != live_environment:
                raise TaskCMethodRunError(
                    "live conda environment changed during the method run"
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
            "inner_status": inner_status_name,
            "status_origin": status_origin,
            "run_identity_sha256": environment["run_identity_sha256"],
            "artifacts": artifacts,
            "trial_parameters_sha256": sealed_trial_parameters_sha256,
        }
        if reason is not None:
            status["reason"] = reason
        status = _seal_status(status)
        _write_json(staging / "method_status.json", status)
        _validate_existing_output(
            staging,
            expected_environment=environment,
            spec=spec,
            gene_names=genes,
            expected_raw=(raw if spec.method_id == "mean_difference" else None),
            hypersca_validation_inputs=hypersca_validation_inputs,
        )
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
            failure_origin = (
                "standardization"
                if isinstance(exc, (_InvalidMethodOutput, TaskCPredictionError))
                else "post_run_validation"
            )
            status = _seal_status({
                "schema_version": SCHEMA_VERSION,
                "method_id": spec.method_id,
                "status": (
                    "failed_invalid_output"
                    if isinstance(exc, (_InvalidMethodOutput, TaskCPredictionError))
                    else "failed_runtime_unavailable"
                ),
                "inner_status": inner_status_name,
                "status_origin": failure_origin,
                "run_identity_sha256": environment_payload["run_identity_sha256"],
                "reason": str(exc),
                "artifacts": _artifact_records(staging),
                "trial_parameters_sha256": sealed_trial_parameters_sha256,
            })
            _write_json(staging / "method_status.json", status)
            _validate_existing_output(
                staging,
                expected_environment=environment_payload,
                spec=spec,
                gene_names=genes,
                hypersca_validation_inputs=hypersca_validation_inputs,
            )
            _publish_bundle(staging, output_dir)
            published = True
        raise
    finally:
        if fixed_input_root is not None and fixed_input_root.exists():
            shutil.rmtree(fixed_input_root)
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
