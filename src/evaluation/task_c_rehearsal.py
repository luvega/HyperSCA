"""Freeze the Task C rehearsal scope and provide small array checks.

The array helpers in this module make the scientific rules easy to test.  A real
rehearsal must still use the comparison-profile input and its analysis record
from :mod:`src.evaluation.task_c_profile_input`; these helpers are not a second
way to prepare real data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
from itertools import islice
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any
import unicodedata
from urllib.parse import unquote, urlparse

import numpy as np

from src.evaluation.task_c_profile_input import (
    CROSS_TRANSFORMATION,
    MAXIMUM_DISTINCT_LABELS,
    MAXIMUM_PARENT_CELLS,
    MAXIMUM_PARENT_GENES,
    MAXIMUM_TEXT_ITEM_BYTES,
    MAXIMUM_TOTAL_TEXT_BYTES,
)


MAXIMUM_CONFIG_BYTES = 64 * 1024
MAXIMUM_JSON_DEPTH = 16
MAXIMUM_SEED = 2**32 - 1
MAXIMUM_METHOD_WORKER_BYTES = 4 * 1024 * 1024
MAXIMUM_METHOD_COMMAND_ARGUMENTS = 512
MAXIMUM_METHOD_BOUNDARY_FILES = 64
MAXIMUM_METHOD_ENVIRONMENT_ENTRIES = 4_096
MAXIMUM_REHEARSAL_JSON_BYTES = 16 * 1024 * 1024
MAXIMUM_REHEARSAL_SUMMARY_JSON_BYTES = 128 * 1024 * 1024
MAXIMUM_REHEARSAL_INPUT_BYTES = 1024 * 1024 * 1024
MAXIMUM_REHEARSAL_REFERENCE_BYTES = 512 * 1024 * 1024
MAXIMUM_REHEARSAL_PREDICTION_BYTES = 64 * 1024 * 1024
MAXIMUM_REHEARSAL_PREDICTION_ROWS = 256 * 255
CONTROL_LABEL = "non-targeting"

FEATURE_SELECTION = "common_expression_genes_train_control_variance_v1"
_CONFIG_FIELDS = (
    "schema_version",
    "seed",
    "promotion_eligible",
    "feature_selection",
    "profiles",
    "null_controls",
    "required_core_methods",
    "required_interventional_method_count",
    "required_artifacts",
    "full_run_seeds",
)
_PROFILE_FIELDS = (
    "maximum_genes",
    "maximum_cells_per_context",
    "timeout_seconds_per_method",
)
_PROFILE_VALUES = {
    "connection": (64, 2_000, 1_800),
    "comprehensive": (256, 20_000, 14_400),
}
_NULL_CONTROL_FIELDS = (
    "repeats",
    "minimum_empirical_advantage",
    "maximum_empirical_p_value",
)
_NULL_CONTROL_VALUES: dict[str, float | int] = {
    "repeats": 20,
    "minimum_empirical_advantage": 0.0,
    "maximum_empirical_p_value": 0.05,
}
_REQUIRED_CORE_METHODS = (
    "hypersca_c",
    "mean_difference",
    "random1000",
    "grnboost",
    "pc",
    "notears_linear",
)
_REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "input_summary.json",
    "metrics.json",
    "predictions.csv",
    "promotion_decision.json",
)
_FULL_RUN_SEEDS = (11, 23, 47, 71, 97)
_PRIVATE_HOLDOUT_PATHS = frozenset(
    {
        "private/within/k562/holdout.npz",
        "private/within/rpe1/holdout.npz",
        "private/cross/k562_to_rpe1/target_holdout.npz",
        "private/cross/rpe1_to_k562/target_holdout.npz",
    }
)
_MATERIALIZATION_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "input_sha256",
        "content_sha256",
        "gene_names_sha256",
        "gene_projection",
        "sealed_holdout_semantic_content_sha256",
    }
)
_PRIVATE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "train_sources",
        "tune_sources",
        "holdout_sources",
        "control_indices",
        "content_sha256",
        "gene_names_sha256",
        "gene_projection",
        "sealed_holdout_semantic_content_sha256",
        "materialization_identity",
        "files",
    }
)
REHEARSAL_CONDITIONS = (
    "within_k562",
    "within_rpe1",
    "k562_to_rpe1",
    "rpe1_to_k562",
)
_FINAL_REHEARSAL_STATUSES = frozenset(
    {
        "passed_real_rehearsal",
        "passed_synthetic_smoke",
        "failed_timeout",
        "failed_resource_limit",
        "failed_runtime_unavailable",
        "failed_launch",
        "failed_invalid_output",
        "failed_private_scoring",
        "failed_null_control",
        "official_code_incompatible",
        "official_assets_unavailable",
    }
)
_INNER_TO_REHEARSAL_STATUS = {
    "completed_standardized_output": "passed_real_rehearsal",
    "failed_timeout": "failed_timeout",
    "failed_resource_limit": "failed_resource_limit",
    "failed_runtime_unavailable": "failed_runtime_unavailable",
    "failed_launch": "failed_launch",
    "failed_invalid_output": "failed_invalid_output",
    "official_code_incompatible": "official_code_incompatible",
    "official_assets_unavailable": "official_assets_unavailable",
}
_REHEARSAL_EXTRA_ARTIFACTS = (
    "method_status.json",
    "resource_usage.json",
    "environment_manifest.json",
)


class TaskCRehearsalError(ValueError):
    """The rehearsal rule or supplied research data are not safe to use."""


@dataclass(frozen=True, slots=True)
class FrozenRehearsalPredictions:
    """One immutable prediction byte sequence shared by scoring and publication."""

    path: Path
    payload: bytes
    sha256: str
    source_path: Path
    source_identity: tuple[int, int, int, int, int]
    frame: Any


@dataclass(frozen=True, slots=True)
class FrozenRehearsalFile:
    """One identity-bound file in the controller's whole-round closure."""

    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class FrozenRehearsalClosure:
    """Controller, model, scoring, reference, and code files fixed for one round."""

    files: Mapping[str, FrozenRehearsalFile]
    external_identity: Mapping[str, str]
    sha256: str


def build_rehearsal_run_id(
    *, profile: str, condition: str, method_id: str, seed: int
) -> str:
    """Build the stable name used for one method-condition rehearsal."""

    if isinstance(seed, bool) or type(seed) is not int or not 0 <= seed <= MAXIMUM_SEED:
        raise TaskCRehearsalError("run identity seed is outside the allowed range")
    safe = (profile, condition, method_id, f"seed-{seed}")
    allowed = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if any(
        type(value) is not str
        or not value
        or value != value.strip()
        or "/" in value
        or "\\" in value
        or ".." in value
        or any(character not in allowed for character in value)
        for value in safe
    ):
        raise TaskCRehearsalError("run identity contains unsafe text")
    return "__".join(safe)


def validate_required_run_artifacts(
    run_dir: str | Path, required_artifacts: Sequence[str]
) -> None:
    """Require the complete, fixed scientific record for a successful run."""

    required = _fixed_text_tuple(
        required_artifacts, "required analysis output files"
    )
    expected = frozenset((*required, *_REHEARSAL_EXTRA_ARTIFACTS))
    destination = Path(run_dir)
    try:
        names = {
            entry.name
            for entry in os.scandir(destination)
            if entry.is_file(follow_symlinks=False)
        }
        non_files = [
            entry.name
            for entry in os.scandir(destination)
            if not entry.is_file(follow_symlinks=False)
        ]
    except OSError as exc:
        raise TaskCRehearsalError(
            "run result directory cannot be inspected"
        ) from exc
    missing = sorted(expected - names)
    extra = sorted(names - expected)
    if missing:
        raise TaskCRehearsalError(
            f"run is missing required artifacts: {missing}"
        )
    if extra or non_files:
        raise TaskCRehearsalError(
            f"run contains unexpected analysis outputs: {sorted(set(extra + non_files))}"
        )


def classify_rehearsal_method_status(inner_status: str) -> str:
    """Translate a method result without weakening or hiding a failure."""

    if type(inner_status) is not str or inner_status not in _INNER_TO_REHEARSAL_STATUS:
        raise TaskCRehearsalError("method returned an unrecognized final status")
    return _INNER_TO_REHEARSAL_STATUS[inner_status]


def _classify_controller_failure(
    error: BaseException | str, inner_status: str | None
) -> str:
    if (
        inner_status in _INNER_TO_REHEARSAL_STATUS
        and inner_status != "completed_standardized_output"
    ):
        return classify_rehearsal_method_status(str(inner_status))
    message = str(error).casefold()
    if "sealed" in message or "private" in message:
        return "failed_private_scoring"
    if "null" in message or "zero-effect" in message:
        return "failed_null_control"
    if "selection" in message or "prediction" in message:
        return "failed_invalid_output"
    return "failed_runtime_unavailable"


def build_rehearsal_execution_plan(
    *, profile: str, method_ids: Sequence[str]
) -> Mapping[str, Mapping[str, object]]:
    """Return the fixed four-condition train, tune and final-fit design."""

    if profile not in _PROFILE_VALUES:
        raise TaskCRehearsalError("profile must be connection or comprehensive")
    methods = _fixed_text_tuple(method_ids, "rehearsal methods")
    if len(set(methods)) != len(methods):
        raise TaskCRehearsalError("rehearsal methods must be unique")
    trial_counts = {
        method: (
            2
            if profile == "connection"
            and method == "hypersca_c"
            else 0
        )
        for method in methods
    }
    selected = tuple(
        method
        for method in ("hypersca_c",)
        if trial_counts.get(method) == 2
    )
    return MappingProxyType(
        {
            condition: MappingProxyType(
                {
                    "stages": ("train", "tune", "refit"),
                    "trial_counts": MappingProxyType(dict(trial_counts)),
                    "selection_bound_refit": selected,
                    "method_stages": MappingProxyType(
                        {
                            method: (
                                ("train", "tune", "refit")
                                if trial_counts[method] == 2
                                else ("refit",)
                            )
                            for method in methods
                        }
                    ),
                }
            )
            for condition in REHEARSAL_CONDITIONS
        }
    )


@dataclass(frozen=True, slots=True)
class MethodWorkerEntrySnapshot:
    """Identity of the one reviewed Python file allowed to start a method."""

    path: Path
    sha256: str
    identity: tuple[int, ...]
    payload: bytes = field(repr=False)


def _fixed_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise TaskCRehearsalError(f"{label} must be a positive whole number")
    return value


@dataclass(frozen=True, slots=True)
class RehearsalProfile:
    """The maximum data size and time allowed for one rehearsal profile."""

    maximum_genes: int
    maximum_cells_per_context: int
    timeout_seconds_per_method: int

    def __post_init__(self) -> None:
        _fixed_positive_integer(self.maximum_genes, "maximum_genes")
        _fixed_positive_integer(
            self.maximum_cells_per_context, "maximum_cells_per_context"
        )
        _fixed_positive_integer(
            self.timeout_seconds_per_method, "timeout_seconds_per_method"
        )


@dataclass(frozen=True, slots=True)
class TaskCRehearsalConfig:
    """Read-only, pre-agreed rules for the Task C real-data rehearsal."""

    schema_version: str
    seed: int
    promotion_eligible: bool
    feature_selection: str
    profiles: Mapping[str, RehearsalProfile]
    null_controls: Mapping[str, float | int]
    required_core_methods: tuple[str, ...]
    required_interventional_method_count: int
    required_artifacts: tuple[str, ...]
    full_run_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "1.0":
            raise TaskCRehearsalError("rehearsal configuration schema changed")
        if type(self.seed) is not int or self.seed != 11:
            raise TaskCRehearsalError("rehearsal seed must remain 11")
        if self.promotion_eligible is not False:
            raise TaskCRehearsalError(
                "rehearsal results cannot be treated as evidence for advancement"
            )
        if (
            type(self.feature_selection) is not str
            or self.feature_selection != FEATURE_SELECTION
        ):
            raise TaskCRehearsalError(
                "gene selection must remain based on training-control variance"
            )

        if not isinstance(self.profiles, Mapping):
            raise TaskCRehearsalError("profiles must contain the two fixed profiles")
        reported_profile_count = len(self.profiles)
        profile_keys = tuple(islice(iter(self.profiles), len(_PROFILE_VALUES) + 1))
        if (
            reported_profile_count != len(_PROFILE_VALUES)
            or len(profile_keys) != len(_PROFILE_VALUES)
            or profile_keys != tuple(_PROFILE_VALUES)
        ):
            raise TaskCRehearsalError(
                "profiles must remain connection then comprehensive"
            )
        copied_profiles: dict[str, RehearsalProfile] = {}
        for name, expected in _PROFILE_VALUES.items():
            profile = self.profiles[name]
            if type(profile) is not RehearsalProfile or (
                profile.maximum_genes,
                profile.maximum_cells_per_context,
                profile.timeout_seconds_per_method,
            ) != expected:
                raise TaskCRehearsalError(
                    f"{name} must keep the fixed profile values"
                )
            copied_profiles[name] = RehearsalProfile(*expected)

        if not isinstance(self.null_controls, Mapping):
            raise TaskCRehearsalError("null controls must keep the fixed checks")
        reported_null_count = len(self.null_controls)
        null_keys = tuple(
            islice(iter(self.null_controls), len(_NULL_CONTROL_FIELDS) + 1)
        )
        if (
            reported_null_count != len(_NULL_CONTROL_FIELDS)
            or len(null_keys) != len(_NULL_CONTROL_FIELDS)
            or null_keys != _NULL_CONTROL_FIELDS
        ):
            raise TaskCRehearsalError(
                "null-control fields or their order changed"
            )
        copied_nulls = {name: self.null_controls[name] for name in null_keys}
        if type(copied_nulls["repeats"]) is not int:
            raise TaskCRehearsalError("null-control repeats must be a whole number")
        for name in (
            "minimum_empirical_advantage",
            "maximum_empirical_p_value",
        ):
            if type(copied_nulls[name]) is not float or not math.isfinite(
                copied_nulls[name]
            ):
                raise TaskCRehearsalError(
                    f"null-control {name} must be a finite decimal number"
                )
        if copied_nulls != _NULL_CONTROL_VALUES:
            raise TaskCRehearsalError("null-control values must remain fixed")

        methods = _fixed_text_tuple(
            self.required_core_methods, "required core methods"
        )
        if methods != _REQUIRED_CORE_METHODS:
            raise TaskCRehearsalError(
                "required core methods or their order changed"
            )
        if (
            isinstance(self.required_interventional_method_count, bool)
            or type(self.required_interventional_method_count) is not int
            or self.required_interventional_method_count != 1
        ):
            raise TaskCRehearsalError(
                "exactly one available intervention-aware comparison is required"
            )
        artifacts = _fixed_text_tuple(
            self.required_artifacts, "required analysis output files"
        )
        if artifacts != _REQUIRED_ARTIFACTS:
            raise TaskCRehearsalError(
                "required analysis output files or their order changed"
            )
        seeds = _fixed_integer_tuple(self.full_run_seeds, "full-run seeds")
        if len(set(seeds)) != len(seeds) or seeds != _FULL_RUN_SEEDS:
            raise TaskCRehearsalError(
                "full-run seeds must remain unique and in their fixed order"
            )

        object.__setattr__(self, "profiles", MappingProxyType(copied_profiles))
        object.__setattr__(self, "null_controls", MappingProxyType(copied_nulls))
        object.__setattr__(self, "required_core_methods", methods)
        object.__setattr__(self, "required_artifacts", artifacts)
        object.__setattr__(self, "full_run_seeds", seeds)


def _fixed_text_tuple(values: object, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TaskCRehearsalError(f"{label} must be an ordered text list")
    reported_count = len(values)
    if reported_count > 64:
        raise TaskCRehearsalError(f"{label} contains too many values")
    copied = tuple(values[index] for index in range(reported_count))
    if not copied or any(type(value) is not str or not value for value in copied):
        raise TaskCRehearsalError(f"{label} must contain only non-empty text")
    return copied


def _fixed_integer_tuple(values: object, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TaskCRehearsalError(f"{label} must be an ordered whole-number list")
    reported_count = len(values)
    if reported_count > 64:
        raise TaskCRehearsalError(f"{label} contains too many values")
    copied = tuple(values[index] for index in range(reported_count))
    if not copied or any(type(value) is not int for value in copied):
        raise TaskCRehearsalError(f"{label} must contain only whole numbers")
    return copied


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskCRehearsalError(
                f"rehearsal configuration contains duplicate field {key}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise TaskCRehearsalError(
        f"rehearsal configuration contains non-finite value {token}"
    )


def _json_depth(value: object, depth: int = 1) -> int:
    if isinstance(value, dict):
        return max(
            [depth, *(_json_depth(item, depth + 1) for item in value.values())]
        )
    if isinstance(value, list):
        return max([depth, *(_json_depth(item, depth + 1) for item in value)])
    return depth


def _reject_nonfinite_numbers(value: object) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if type(current) is float and not math.isfinite(current):
            raise TaskCRehearsalError(
                "rehearsal configuration contains a non-finite number"
            )
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_nlink),
    )


def _read_limited_regular_file(path: Path) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        descriptor = os.open(
            absolute,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration must be a readable regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TaskCRehearsalError(
                "rehearsal configuration must be a regular file"
            )
        if before.st_size > MAXIMUM_CONFIG_BYTES:
            raise TaskCRehearsalError("rehearsal configuration is too large")
        collected = bytearray()
        while len(collected) <= MAXIMUM_CONFIG_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAXIMUM_CONFIG_BYTES + 1 - len(collected)),
            )
            if not chunk:
                break
            collected.extend(chunk)
        after = os.fstat(descriptor)
    except TaskCRehearsalError:
        raise
    except OSError as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration could not be read from its regular file"
        ) from exc
    finally:
        os.close(descriptor)
    if len(collected) > MAXIMUM_CONFIG_BYTES:
        raise TaskCRehearsalError("rehearsal configuration is too large")
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration changed while it was being read"
        ) from exc
    if (
        len(collected) != before.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(current)
    ):
        raise TaskCRehearsalError(
            "rehearsal configuration changed while it was being read"
        )
    return bytes(collected)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _read_limited_regular_file(Path(path))
    try:
        decoded = payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration is not valid UTF-8 text"
        ) from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except TaskCRehearsalError:
        raise
    except RecursionError as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration is too deeply nested"
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration is not valid JSON"
        ) from exc
    try:
        depth = _json_depth(value)
    except RecursionError as exc:
        raise TaskCRehearsalError(
            "rehearsal configuration is too deeply nested"
        ) from exc
    if depth > MAXIMUM_JSON_DEPTH:
        raise TaskCRehearsalError("rehearsal configuration is too deeply nested")
    _reject_nonfinite_numbers(value)
    if not isinstance(value, dict):
        raise TaskCRehearsalError(
            "rehearsal configuration must be one JSON object"
        )
    return value


def _require_field_order(
    payload: Mapping[str, object], fields: tuple[str, ...], label: str
) -> None:
    if tuple(payload) != fields:
        raise TaskCRehearsalError(f"{label} fields or their order changed")


def load_task_c_rehearsal_config(path: str | Path) -> TaskCRehearsalConfig:
    """Read the closed rehearsal policy without allowing local relaxation."""

    payload = _load_json_object(Path(path))
    _require_field_order(payload, _CONFIG_FIELDS, "rehearsal configuration")

    raw_profiles = payload["profiles"]
    if not isinstance(raw_profiles, dict):
        raise TaskCRehearsalError("profiles must be one JSON object")
    if tuple(raw_profiles) != tuple(_PROFILE_VALUES):
        raise TaskCRehearsalError(
            "profiles must remain connection then comprehensive"
        )
    profiles: dict[str, RehearsalProfile] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            raise TaskCRehearsalError(f"{name} profile must be one JSON object")
        _require_field_order(raw_profile, _PROFILE_FIELDS, f"{name} profile")
        profiles[name] = RehearsalProfile(
            maximum_genes=raw_profile["maximum_genes"],
            maximum_cells_per_context=raw_profile["maximum_cells_per_context"],
            timeout_seconds_per_method=raw_profile["timeout_seconds_per_method"],
        )

    raw_nulls = payload["null_controls"]
    if not isinstance(raw_nulls, dict):
        raise TaskCRehearsalError("null controls must be one JSON object")
    _require_field_order(raw_nulls, _NULL_CONTROL_FIELDS, "null-control")

    return TaskCRehearsalConfig(
        schema_version=payload["schema_version"],
        seed=payload["seed"],
        promotion_eligible=payload["promotion_eligible"],
        feature_selection=payload["feature_selection"],
        profiles=profiles,
        null_controls=raw_nulls,
        required_core_methods=payload["required_core_methods"],
        required_interventional_method_count=payload[
            "required_interventional_method_count"
        ],
        required_artifacts=payload["required_artifacts"],
        full_run_seeds=payload["full_run_seeds"],
    )


def _canonical_texts(
    values: object,
    label: str,
    *,
    require_unique: bool,
    maximum_items: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TaskCRehearsalError(f"{label} must be a one-dimensional text list")
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise TaskCRehearsalError(
                f"{label} must be a one-dimensional text list"
            )
        if len(values) > maximum_items:
            raise TaskCRehearsalError(f"{label} contains too many values")
        copied = tuple(values.tolist())
    else:
        if not isinstance(values, Sequence):
            raise TaskCRehearsalError(
                f"{label} must be a one-dimensional text list"
            )
        reported_count = len(values)
        if reported_count > maximum_items:
            raise TaskCRehearsalError(f"{label} contains too many values")
        copied = tuple(values[index] for index in range(reported_count))
    if not copied:
        raise TaskCRehearsalError(f"{label} must not be empty")
    if len(copied) > maximum_items:
        raise TaskCRehearsalError(f"{label} contains too many values")
    total_bytes = 0
    maximum_characters = 0
    for value in copied:
        if type(value) is not str:
            raise TaskCRehearsalError(f"{label} must contain only text")
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise TaskCRehearsalError(f"{label} must use valid UTF-8 text") from exc
        if len(encoded) > MAXIMUM_TEXT_ITEM_BYTES:
            raise TaskCRehearsalError(f"{label} exceeds the per-item text limit")
        total_bytes += len(encoded)
        maximum_characters = max(maximum_characters, len(value))
        projected_array_bytes = (
            len(copied) * maximum_characters * np.dtype("U1").itemsize
        )
        if (
            total_bytes > MAXIMUM_TOTAL_TEXT_BYTES
            or projected_array_bytes > MAXIMUM_TOTAL_TEXT_BYTES
        ):
            raise TaskCRehearsalError(f"{label} exceeds the total text limit")
        if (
            not value
            or value != value.strip()
            or not unicodedata.is_normalized("NFC", value)
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise TaskCRehearsalError(
                f"{label} must contain safe, non-empty NFC text"
            )
    if require_unique and len(set(copied)) != len(copied):
        raise TaskCRehearsalError(f"{label} must be unique")
    return copied


def _immutable_array(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    if contiguous.dtype.hasobject:
        raise TaskCRehearsalError("returned arrays cannot contain Python objects")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


def _immutable_text_array(values: tuple[str, ...]) -> np.ndarray:
    width = max(len(value) for value in values)
    projected = len(values) * width * np.dtype("U1").itemsize
    if projected > MAXIMUM_TOTAL_TEXT_BYTES:
        raise TaskCRehearsalError("returned labels exceed the total text limit")
    return _immutable_array(np.asarray(values, dtype=f"U{width}"))


def _numeric_matrix(
    expression: object,
    label: str,
    *,
    expected_columns: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(expression)
    except (TypeError, ValueError) as exc:
        raise TaskCRehearsalError(f"{label} must be a numeric matrix") from exc
    if (
        raw.ndim != 2
        or raw.shape[0] < 1
        or raw.shape[1] < 2
        or raw.shape[0] > MAXIMUM_PARENT_CELLS
        or raw.shape[1] > MAXIMUM_PARENT_GENES
        or raw.dtype.kind not in {"i", "u", "f"}
    ):
        raise TaskCRehearsalError(
            f"{label} shape must be cells by at least two numeric genes"
        )
    if expected_columns is not None and raw.shape[1] != expected_columns:
        raise TaskCRehearsalError(f"{label} shape does not match the gene names")
    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaskCRehearsalError(f"{label} must be a numeric matrix") from exc
    if not np.isfinite(values).all():
        raise TaskCRehearsalError(f"{label} must contain only finite values")
    return values


def _require_context_keys(contexts: object) -> Mapping[str, object]:
    if not isinstance(contexts, Mapping):
        raise TaskCRehearsalError("contexts must contain k562 and rpe1")
    reported_count = len(contexts)
    keys = tuple(islice(iter(contexts), 3))
    if (
        reported_count != 2
        or len(keys) != 2
        or any(type(key) is not str for key in keys)
        or set(keys) != {"k562", "rpe1"}
    ):
        raise TaskCRehearsalError("contexts must contain exactly k562 and rpe1")
    return contexts


def choose_rehearsal_genes(
    allowed_control_expression: Mapping[str, np.ndarray],
    gene_names: Sequence[str],
    maximum_genes: int,
) -> tuple[str, ...]:
    """Rank shared genes using only registered training-control variability.

    Variance uses ``ddof=0``, matching the comparison-profile builder.  Ties are
    resolved by gene name so input mapping order cannot change the result.
    """

    contexts = _require_context_keys(allowed_control_expression)
    genes = _canonical_texts(
        gene_names,
        "gene names",
        require_unique=True,
        maximum_items=MAXIMUM_PARENT_GENES,
    )
    if (
        isinstance(maximum_genes, bool)
        or type(maximum_genes) is not int
        or maximum_genes < 2
        or maximum_genes > MAXIMUM_PARENT_GENES
    ):
        raise TaskCRehearsalError("maximum_genes must be a whole number of at least two")

    variances: list[np.ndarray] = []
    for context_id in ("k562", "rpe1"):
        values = _numeric_matrix(
            contexts[context_id],
            f"{context_id} control expression",
            expected_columns=len(genes),
        )
        if values.shape[0] < 2:
            raise TaskCRehearsalError(
                f"{context_id} needs at least two control cells"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            variance = values.var(axis=0, ddof=0)
        if not np.isfinite(variance).all():
            raise TaskCRehearsalError(
                f"{context_id} derived gene variance is not finite"
            )
        variances.append(variance)
    with np.errstate(over="ignore", invalid="ignore"):
        mean_variance = np.mean(np.stack(variances, axis=0), axis=0)
    if not np.isfinite(mean_variance).all():
        raise TaskCRehearsalError("mean derived gene variance is not finite")
    order = sorted(
        range(len(genes)),
        key=lambda index: (-float(mean_variance[index]), genes[index]),
    )
    return tuple(genes[index] for index in order[:maximum_genes])


def _stratified_quotas(
    labels: tuple[str, ...],
    maximum_cells: int,
    minimum_cells_per_group: int,
) -> dict[str, int]:
    count_by_label = dict(sorted(Counter(labels).items()))
    if len(count_by_label) > MAXIMUM_DISTINCT_LABELS:
        raise TaskCRehearsalError(
            "distinct cell-label groups exceed the public profile limit"
        )
    if any(count < minimum_cells_per_group for count in count_by_label.values()):
        raise TaskCRehearsalError(
            "every cell-label group must meet the requested minimum"
        )
    required = minimum_cells_per_group * len(count_by_label)
    if required > maximum_cells:
        raise TaskCRehearsalError(
            "maximum_cells is smaller than the reserved group minimums"
        )
    quotas = {label: minimum_cells_per_group for label in count_by_label}
    remaining = maximum_cells - required
    capacities = {
        label: count - minimum_cells_per_group
        for label, count in count_by_label.items()
    }
    capacity_total = sum(capacities.values())
    exact = {
        label: (remaining * capacity / capacity_total if capacity_total else 0.0)
        for label, capacity in capacities.items()
    }
    for label in quotas:
        quotas[label] += min(capacities[label], int(np.floor(exact[label])))
    unassigned = maximum_cells - sum(quotas.values())
    candidates = sorted(
        (label for label in quotas if quotas[label] < count_by_label[label]),
        key=lambda label: (-(exact[label] - np.floor(exact[label])), label),
    )
    for label in candidates[:unassigned]:
        quotas[label] += 1
    if sum(quotas.values()) != maximum_cells:
        raise TaskCRehearsalError(
            "stratified cell selection could not fill the requested size"
        )
    return quotas


def choose_rehearsal_cells(
    interventions: Sequence[str],
    maximum_cells: int,
    seed: int,
    *,
    minimum_cells_per_group: int = 1,
) -> np.ndarray:
    """Choose cells reproducibly while preserving every retained label group."""

    labels = _canonical_texts(
        interventions,
        "cell labels",
        require_unique=False,
        maximum_items=MAXIMUM_PARENT_CELLS,
    )
    if (
        isinstance(maximum_cells, bool)
        or type(maximum_cells) is not int
        or maximum_cells < 1
        or maximum_cells > MAXIMUM_PARENT_CELLS
    ):
        raise TaskCRehearsalError("maximum_cells must be a positive whole number")
    if isinstance(seed, bool) or type(seed) is not int or not 0 <= seed <= MAXIMUM_SEED:
        raise TaskCRehearsalError(
            f"seed must be a whole number from 0 to {MAXIMUM_SEED}"
        )
    if (
        isinstance(minimum_cells_per_group, bool)
        or type(minimum_cells_per_group) is not int
        or minimum_cells_per_group < 1
    ):
        raise TaskCRehearsalError(
            "minimum_cells_per_group must be a positive whole number"
        )

    quotas = _stratified_quotas(
        labels,
        min(maximum_cells, len(labels)),
        minimum_cells_per_group,
    )
    if maximum_cells >= len(labels):
        return _immutable_array(np.arange(len(labels), dtype=np.int64))

    indices_by_label: dict[str, list[int]] = {
        label: [] for label in quotas
    }
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)
    selected: list[int] = []
    for label in sorted(quotas):
        candidates = np.asarray(indices_by_label[label], dtype=np.int64)
        digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        chosen = rng.choice(candidates, size=quotas[label], replace=False)
        selected.extend(int(index) for index in chosen.tolist())
    return _immutable_array(np.asarray(sorted(selected), dtype=np.int64))


def center_and_merge_allowed_contexts(
    contexts: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    control_label: str = CONTROL_LABEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each allowed context from its own controls, then combine rows.

    Control standard deviations use ``ddof=0``.  A value no larger than
    ``1e-6`` is treated as uninformative and replaced by one, matching the
    registered comparison-profile transformation.
    """

    context_map = _require_context_keys(contexts)
    canonical_control = _canonical_texts(
        (control_label,),
        "control label",
        require_unique=True,
        maximum_items=1,
    )[0]
    centered: list[np.ndarray] = []
    labels_out: list[tuple[str, ...]] = []
    environments: list[tuple[str, ...]] = []
    expected_genes: int | None = None
    for context_id in ("k562", "rpe1"):
        entry = context_map[context_id]
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TaskCRehearsalError(
                f"{context_id} context must contain expression and labels"
            )
        raw_expression, raw_labels = entry
        values = _numeric_matrix(raw_expression, f"{context_id} expression")
        context_labels = _canonical_texts(
            raw_labels,
            f"{context_id} labels",
            require_unique=False,
            maximum_items=MAXIMUM_PARENT_CELLS,
        )
        if values.shape[0] != len(context_labels):
            raise TaskCRehearsalError(
                f"{context_id} expression and labels shape do not agree"
            )
        if expected_genes is None:
            expected_genes = values.shape[1]
        elif values.shape[1] != expected_genes:
            raise TaskCRehearsalError("cross-context expression must use the same genes")
        controls = np.fromiter(
            (label == canonical_control for label in context_labels),
            dtype=bool,
            count=len(context_labels),
        )
        if int(np.count_nonzero(controls)) < 2:
            raise TaskCRehearsalError(
                f"{context_id} needs at least two controls"
            )
        control_values = values[controls]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            center = control_values.mean(axis=0)
            scale = control_values.std(axis=0, ddof=0)
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise TaskCRehearsalError(
                f"{context_id} derived control statistics are not finite"
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (values - center) / np.where(
                scale <= 1e-6, 1.0, scale
            )
        if not np.isfinite(standardized).all():
            raise TaskCRehearsalError(
                f"{context_id} standardized expression is not finite"
            )
        centered.append(standardized)
        labels_out.append(context_labels)
        environments.append(tuple(context_id for _ in context_labels))

    merged = np.concatenate(centered, axis=0)
    if not np.isfinite(merged).all():
        raise TaskCRehearsalError("merged standardized expression is not finite")
    merged_labels = tuple(label for group in labels_out for label in group)
    merged_environments = tuple(
        environment for group in environments for environment in group
    )
    return (
        _immutable_array(merged),
        _immutable_text_array(merged_labels),
        _immutable_text_array(merged_environments),
    )


def _real_private_directory(private_root: str | Path) -> Path:
    """Return a private scoring directory without accepting path aliases."""

    try:
        private = Path(
            os.path.abspath(os.fspath(Path(private_root).expanduser()))
        )
    except (TypeError, ValueError, OSError) as exc:
        raise TaskCRehearsalError(
            "private scoring root must be a real directory"
        ) from exc
    cursor = Path(private.anchor)
    try:
        for component in private.parts[1:]:
            cursor /= component
            metadata = os.lstat(cursor)
            if stat.S_ISLNK(metadata.st_mode):
                raise TaskCRehearsalError(
                    "private scoring root must be a real directory without symbolic links"
                )
        metadata = os.lstat(private)
    except TaskCRehearsalError:
        raise
    except OSError as exc:
        raise TaskCRehearsalError(
            "private scoring root must be a real directory"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise TaskCRehearsalError(
            "private scoring root must be a real directory"
        )
    return private.resolve(strict=True)


def _decoded_command_text(value: str) -> str:
    decoded = value
    for _ in range(3):
        expanded = unquote(decoded)
        if expanded == decoded:
            break
        decoded = expanded
    parsed = urlparse(decoded)
    if parsed.scheme.casefold() == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise TaskCRehearsalError(
                "method command contains an unsupported file location"
            )
        return unquote(parsed.path)
    return decoded


def _command_path_candidates(argument: str) -> tuple[str, ...]:
    candidates = [argument]
    if "=" in argument:
        candidates.append(argument.split("=", 1)[1])
    if argument.startswith("-") and not argument.startswith("--") and len(argument) > 2:
        candidates.append(argument[2:])
    try:
        candidates.extend(shlex.split(argument, posix=True))
    except ValueError as exc:
        raise TaskCRehearsalError(
            "method command contains unmatched quoting"
        ) from exc
    decoded = [_decoded_command_text(candidate.strip("'\"")) for candidate in candidates]
    return tuple(dict.fromkeys(candidate for candidate in decoded if candidate))


def _path_is_within(candidate: Path, parent: Path) -> bool:
    return candidate == parent or parent in candidate.parents


def _real_execution_directory(path: str | Path) -> Path:
    try:
        directory = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        metadata = os.lstat(directory)
    except (TypeError, ValueError, OSError) as exc:
        raise TaskCRehearsalError(
            "method execution directory must be a real directory"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise TaskCRehearsalError(
            "method execution directory must be a real directory"
        )
    return directory.resolve(strict=True)


def _method_worker_bytes(path: str | Path) -> tuple[Path, bytes, tuple[int, ...]]:
    try:
        absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    except (TypeError, ValueError, OSError) as exc:
        raise TaskCRehearsalError(
            "method worker entry must be a real regular file"
        ) from exc
    cursor = Path(absolute.anchor)
    try:
        for component in absolute.parts[1:]:
            cursor /= component
            metadata = os.lstat(cursor)
            if stat.S_ISLNK(metadata.st_mode):
                raise TaskCRehearsalError(
                    "method worker entry must not use symbolic links"
                )
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except TaskCRehearsalError:
        raise
    except OSError as exc:
        raise TaskCRehearsalError(
            "method worker entry must be a real regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAXIMUM_METHOD_WORKER_BYTES
        ):
            raise TaskCRehearsalError(
                "method worker entry must be a single-link regular file"
            )
        collected = bytearray()
        while len(collected) <= MAXIMUM_METHOD_WORKER_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    MAXIMUM_METHOD_WORKER_BYTES + 1 - len(collected),
                ),
            )
            if not chunk:
                break
            collected.extend(chunk)
        after = os.fstat(descriptor)
    except TaskCRehearsalError:
        raise
    except OSError as exc:
        raise TaskCRehearsalError(
            "method worker entry could not be read safely"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise TaskCRehearsalError("method worker entry changed") from exc
    if (
        len(collected) != before.st_size
        or len(collected) > MAXIMUM_METHOD_WORKER_BYTES
        or _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(current)
    ):
        raise TaskCRehearsalError("method worker entry changed")
    return absolute.resolve(strict=True), bytes(collected), _file_identity(after)


def freeze_method_worker_entry(path: str | Path) -> MethodWorkerEntrySnapshot:
    """Record the reviewed method entry before the launch-boundary check."""

    resolved, payload, identity = _method_worker_bytes(path)
    if resolved.suffix.casefold() != ".py":
        raise TaskCRehearsalError("method entry must be a Python worker entry")
    return MethodWorkerEntrySnapshot(
        path=resolved,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=identity,
        payload=payload,
    )


def _verify_method_worker_entry(
    path: Path, snapshot: MethodWorkerEntrySnapshot
) -> None:
    resolved, payload, identity = _method_worker_bytes(path)
    if (
        resolved != snapshot.path
        or hashlib.sha256(payload).hexdigest() != snapshot.sha256
        or identity != snapshot.identity
    ):
        raise TaskCRehearsalError("registered method worker entry changed")


def _normalized_python_interpreters(
    values: tuple[str | Path, ...],
) -> set[Path]:
    allowed: set[Path] = set()
    for value in values:
        if not isinstance(value, (str, Path)):
            raise TaskCRehearsalError(
                "allowed Python interpreters must be an ordered path list"
            )
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise TaskCRehearsalError(
                "allowed Python interpreter paths must be absolute"
            )
        try:
            resolved = candidate.resolve(strict=True)
            metadata = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise TaskCRehearsalError(
                "allowed Python interpreter must be a real regular file"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise TaskCRehearsalError(
                "allowed Python interpreter must be a real regular file"
            )
        allowed.add(resolved)
    if not allowed:
        raise TaskCRehearsalError("allowed Python interpreters must not be empty")
    return allowed


def _bounded_sequence_tuple(
    values: object,
    *,
    label: str,
    maximum_items: int,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TaskCRehearsalError(f"{label} must be an ordered list")
    if type(values) is tuple:
        copied = values
    else:
        copied_values: list[Any] = []
        try:
            iterator = iter(values)
            for _index in range(maximum_items + 1):
                try:
                    copied_values.append(next(iterator))
                except StopIteration:
                    break
        except Exception as exc:
            raise TaskCRehearsalError(
                f"{label} could not be copied safely"
            ) from exc
        copied = tuple(copied_values)
    if not copied:
        raise TaskCRehearsalError(f"{label} must not be empty")
    if len(copied) > maximum_items:
        raise TaskCRehearsalError(f"{label} contains too many values")
    return copied


def _strict_command_snapshot(command: object) -> tuple[str, ...]:
    copied = _bounded_sequence_tuple(
        command,
        label="method command",
        maximum_items=MAXIMUM_METHOD_COMMAND_ARGUMENTS,
    )
    if any(type(argument) is not str for argument in copied):
        raise TaskCRehearsalError(
            "method command must contain exact text arguments"
        )
    if any("\x00" in argument for argument in copied):
        raise TaskCRehearsalError("method command text must not contain NUL")
    return copied  # type: ignore[return-value]


def _allowed_interpreter_snapshot(
    values: object,
) -> tuple[str | Path, ...]:
    copied = _bounded_sequence_tuple(
        values,
        label="allowed Python interpreters",
        maximum_items=MAXIMUM_METHOD_BOUNDARY_FILES,
    )
    if any(not isinstance(value, (str, Path)) for value in copied):
        raise TaskCRehearsalError(
            "allowed Python interpreters must contain only paths"
        )
    return copied  # type: ignore[return-value]


def _worker_snapshot_tuple(
    values: object,
) -> tuple[MethodWorkerEntrySnapshot, ...]:
    copied = _bounded_sequence_tuple(
        values,
        label="allowed worker snapshots",
        maximum_items=MAXIMUM_METHOD_BOUNDARY_FILES,
    )
    if any(type(value) is not MethodWorkerEntrySnapshot for value in copied):
        raise TaskCRehearsalError(
            "allowed workers must use frozen entry snapshots"
        )
    return copied  # type: ignore[return-value]


def _environment_snapshot(
    environment: Mapping[str, str] | None,
) -> Mapping[str, str]:
    source: Mapping[str, str] = os.environ if environment is None else environment
    if not isinstance(source, Mapping):
        raise TaskCRehearsalError(
            "method environment must contain safe text names and values"
        )
    copied: dict[str, str] = {}
    try:
        iterator = iter(source.items())
        for _index in range(MAXIMUM_METHOD_ENVIRONMENT_ENTRIES + 1):
            try:
                item = next(iterator)
            except StopIteration:
                break
            if type(item) not in {tuple, list} or len(item) != 2:
                raise TaskCRehearsalError(
                    "method environment contains a malformed entry"
                )
            key, value = item
            if (
                type(key) is not str
                or type(value) is not str
                or not key
                or "\x00" in key
                or "\x00" in value
                or key in copied
            ):
                raise TaskCRehearsalError(
                    "method environment must contain unique safe text names and values"
                )
            copied[key] = value
    except TaskCRehearsalError:
        raise
    except Exception as exc:
        raise TaskCRehearsalError(
            "method environment could not be copied safely"
        ) from exc
    if len(copied) > MAXIMUM_METHOD_ENVIRONMENT_ENTRIES:
        raise TaskCRehearsalError(
            "method environment contains too many entries"
        )
    return MappingProxyType(copied)


def validate_private_scoring_command(
    command: Sequence[str],
    *,
    private_root: str | Path,
    execution_cwd: str | Path | None = None,
    allowed_python_interpreters: Sequence[str | Path] | None = None,
    allowed_worker_snapshots: Sequence[MethodWorkerEntrySnapshot] | None = None,
    allowed_private_inputs: Sequence[str | Path] = (),
) -> None:
    """Preflight one method command against the sealed-data boundary.

    Relative paths, option assignments and symbolic-link aliases are resolved
    before comparison.  Passing this check is not authorization to start a live
    worker: callers must use :func:`run_validated_private_scoring_command`,
    which executes an immutable private copy in the same checked boundary.
    """

    arguments = _strict_command_snapshot(command)

    private = _real_private_directory(private_root)
    if execution_cwd is None:
        raise TaskCRehearsalError(
            "method execution directory is required for path checking"
        )
    cwd = _real_execution_directory(execution_cwd)
    if isinstance(allowed_private_inputs, (str, bytes)) or not isinstance(
        allowed_private_inputs, Sequence
    ):
        raise TaskCRehearsalError(
            "allowed private scoring inputs must be an ordered path list"
        )
    if len(allowed_private_inputs) > MAXIMUM_METHOD_BOUNDARY_FILES:
        raise TaskCRehearsalError("too many private scoring inputs were allowed")
    allowed_private: set[Path] = set()
    for raw_private_input in allowed_private_inputs:
        if not isinstance(raw_private_input, (str, Path)):
            raise TaskCRehearsalError(
                "allowed private scoring inputs must contain only paths"
            )
        candidate = Path(
            os.path.abspath(os.fspath(Path(raw_private_input).expanduser()))
        )
        try:
            metadata = os.lstat(candidate)
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise TaskCRehearsalError(
                "allowed private scoring input must be a real regular file"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not _path_is_within(resolved, private)
        ):
            raise TaskCRehearsalError(
                "allowed private scoring input must be a single-link regular file under the private root"
            )
        allowed_private.add(resolved)
    first_name = Path(arguments[0]).name.casefold()
    if first_name in {
        "sh",
        "bash",
        "zsh",
        "dash",
        "env",
        "sudo",
        "nice",
        "nohup",
        "command",
    }:
        raise TaskCRehearsalError(
            "method command must not use a shell or environment wrapper"
        )
    if first_name.startswith("python") or first_name.startswith("pypy"):
        if len(arguments) > 1 and (
            arguments[1] in {"-c", "-m"}
            or arguments[1].startswith("-c")
            or arguments[1].startswith("-m")
        ):
            raise TaskCRehearsalError(
                "method command must not use dynamic Python execution"
            )
    private_text = os.fspath(private)
    for argument in arguments:
        for candidate_text in _command_path_candidates(argument):
            try:
                candidate = Path(candidate_text).expanduser()
                if not candidate.is_absolute():
                    candidate = cwd / candidate
                lexical = Path(os.path.normpath(os.fspath(candidate)))
                if _path_is_within(lexical, private):
                    try:
                        exact = candidate.resolve(strict=True)
                    except OSError:
                        exact = lexical
                    if exact not in allowed_private:
                        raise TaskCRehearsalError(
                            "method command contains a private scoring path"
                        )
                    continue
                resolved = candidate.resolve(strict=False)
            except TaskCRehearsalError:
                raise
            except (OSError, RuntimeError, ValueError):
                if private_text in candidate_text:
                    raise TaskCRehearsalError(
                        "method command contains a private scoring path"
                    )
                continue
            if resolved == private or private in resolved.parents:
                if resolved not in allowed_private:
                    raise TaskCRehearsalError(
                        "method command contains a private scoring path"
                    )
    interpreter_values = _allowed_interpreter_snapshot(
        allowed_python_interpreters
    )
    worker_snapshots = _worker_snapshot_tuple(allowed_worker_snapshots)
    allowed_interpreters = _normalized_python_interpreters(
        interpreter_values
    )
    try:
        interpreter = Path(arguments[0]).expanduser()
        if not interpreter.is_absolute():
            raise TaskCRehearsalError(
                "method command must use a registered absolute Python interpreter"
            )
        interpreter = interpreter.resolve(strict=True)
    except OSError as exc:
        raise TaskCRehearsalError(
            "method command Python interpreter is unavailable"
        ) from exc
    if interpreter not in allowed_interpreters:
        raise TaskCRehearsalError(
            "method command must use a registered Python interpreter"
        )
    if len(arguments) < 3 or arguments[1] != "-I":
        raise TaskCRehearsalError(
            "method command must use the fixed 'python -I worker.py' form"
        )
    worker_entry = Path(arguments[2]).expanduser()
    if worker_entry.suffix.casefold() != ".py":
        raise TaskCRehearsalError(
            "method command needs a registered Python worker entry"
        )
    if not worker_entry.is_absolute():
        worker_entry = cwd / worker_entry
    requested_worker_entry = worker_entry
    worker_entry, _initial_payload, _initial_identity = _method_worker_bytes(
        requested_worker_entry
    )
    snapshots_by_path: dict[Path, MethodWorkerEntrySnapshot] = {}
    for snapshot in worker_snapshots:
        if snapshot.path in snapshots_by_path:
            raise TaskCRehearsalError(
                "allowed worker snapshots must contain unique entries"
            )
        snapshots_by_path[snapshot.path] = snapshot
    if worker_entry not in snapshots_by_path:
        raise TaskCRehearsalError(
            "method command does not use a registered worker entry"
        )

    _verify_method_worker_entry(
        requested_worker_entry,
        snapshots_by_path[worker_entry],
    )


def _write_worker_bundle_file(
    bundle: Path, snapshot: MethodWorkerEntrySnapshot
) -> Path:
    destination = bundle / snapshot.path.name
    try:
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
            while written < len(snapshot.payload):
                written += os.write(descriptor, snapshot.payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(destination, 0o400, follow_symlinks=False)
    except OSError as exc:
        raise TaskCRehearsalError(
            "private method worker bundle could not be created"
        ) from exc
    copied_path, copied_payload, _identity = _method_worker_bytes(destination)
    if (
        copied_path != destination.resolve(strict=True)
        or copied_payload != snapshot.payload
        or hashlib.sha256(copied_payload).hexdigest() != snapshot.sha256
    ):
        raise TaskCRehearsalError(
            "private method worker bundle content changed"
        )
    return destination


def _remove_worker_bundle(bundle: Path) -> None:
    if not bundle.exists():
        return
    try:
        for path in bundle.iterdir():
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                raise TaskCRehearsalError(
                    "private method worker bundle contains a symbolic link"
                )
            if stat.S_ISREG(metadata.st_mode):
                os.chmod(path, 0o600, follow_symlinks=False)
        os.chmod(bundle, 0o700, follow_symlinks=False)
        shutil.rmtree(bundle)
    except TaskCRehearsalError:
        raise
    except OSError as exc:
        raise TaskCRehearsalError(
            "private method worker bundle could not be removed"
        ) from exc


def run_validated_private_scoring_command(
    command: Sequence[str],
    *,
    private_root: str | Path,
    execution_cwd: str | Path,
    allowed_python_interpreters: Sequence[str | Path],
    allowed_worker_snapshots: Sequence[MethodWorkerEntrySnapshot],
    allowed_private_inputs: Sequence[str | Path] = (),
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Validate and immediately execute a private, read-only worker copy.

    The command entry is replaced with bytes frozen during review.  Reviewed
    sibling Python files are copied beside it for this invocation only.  No live
    worker path is executed after preflight, and the temporary bundle is removed
    after success, failure or timeout.
    """

    arguments = _strict_command_snapshot(command)
    interpreter_values = _allowed_interpreter_snapshot(
        allowed_python_interpreters
    )
    snapshots = _worker_snapshot_tuple(allowed_worker_snapshots)
    process_environment = _environment_snapshot(environment)
    validate_private_scoring_command(
        arguments,
        private_root=private_root,
        execution_cwd=execution_cwd,
        allowed_python_interpreters=interpreter_values,
        allowed_worker_snapshots=snapshots,
        allowed_private_inputs=allowed_private_inputs,
    )
    cwd = _real_execution_directory(execution_cwd)
    interpreters = _normalized_python_interpreters(interpreter_values)
    interpreter = Path(arguments[0]).expanduser().resolve(strict=True)
    if interpreter not in interpreters:
        raise TaskCRehearsalError(
            "method command Python interpreter changed before launch"
        )
    interpreter_before = _file_identity(
        os.stat(interpreter, follow_symlinks=False)
    )

    entry_path = Path(arguments[2]).expanduser()
    if not entry_path.is_absolute():
        entry_path = cwd / entry_path
    entry_path = entry_path.resolve(strict=True)
    snapshots_by_path = {snapshot.path: snapshot for snapshot in snapshots}
    if entry_path not in snapshots_by_path:
        raise TaskCRehearsalError(
            "method command does not use a registered worker entry"
        )
    if len(snapshots_by_path) != len(snapshots):
        raise TaskCRehearsalError(
            "allowed worker snapshots must contain unique entries"
        )
    parent = snapshots_by_path[entry_path].path.parent
    if any(snapshot.path.parent != parent for snapshot in snapshots):
        raise TaskCRehearsalError(
            "reviewed method dependencies must be local sibling files"
        )
    names = [snapshot.path.name for snapshot in snapshots]
    if len(names) != len(set(names)):
        raise TaskCRehearsalError(
            "reviewed method dependencies must have unique file names"
        )
    for snapshot in snapshots:
        _verify_method_worker_entry(snapshot.path, snapshot)

    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise TaskCRehearsalError(
            "method timeout must be a finite positive number of seconds"
        )

    bundle = Path(tempfile.mkdtemp(prefix="hypersca-method-worker-"))
    try:
        copied = {
            snapshot.path: _write_worker_bundle_file(bundle, snapshot)
            for snapshot in snapshots
        }
        os.chmod(bundle, 0o500, follow_symlinks=False)
        launch_command = (
            str(interpreter),
            "-I",
            str(copied[entry_path]),
            *arguments[3:],
        )
        interpreter_after = _file_identity(
            os.stat(interpreter, follow_symlinks=False)
        )
        if interpreter_after != interpreter_before:
            raise TaskCRehearsalError(
                "method command Python interpreter changed before launch"
            )
        return subprocess.run(
            launch_command,
            cwd=cwd,
            env=process_environment,
            capture_output=True,
            text=True,
            timeout=(float(timeout_seconds) if timeout_seconds is not None else None),
            check=False,
        )
    finally:
        _remove_worker_bundle(bundle)


def _strict_json_bytes(payload: object) -> bytes:
    try:
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
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise TaskCRehearsalError(
            "rehearsal record contains a value that cannot be recorded safely"
        ) from exc


def _strict_json_payload(payload_bytes: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = payload_bytes.decode("utf-8", errors="strict")
        payload = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except TaskCRehearsalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TaskCRehearsalError(f"{label} is not valid UTF-8 JSON") from exc
    try:
        depth = _json_depth(payload)
    except RecursionError as exc:
        raise TaskCRehearsalError(f"{label} is too deeply nested") from exc
    if depth > MAXIMUM_JSON_DEPTH:
        raise TaskCRehearsalError(f"{label} is too deeply nested")
    _reject_nonfinite_numbers(payload)
    if not isinstance(payload, dict):
        raise TaskCRehearsalError(f"{label} must contain one JSON object")
    return payload


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    payload_bytes, _identity = _capture_regular_bytes(
        path,
        label=label,
        maximum_bytes=MAXIMUM_REHEARSAL_JSON_BYTES,
    )
    return _strict_json_payload(payload_bytes, label)


def _strict_json_file_with_total_budget(
    path: Path, *, label: str, remaining_bytes: int
) -> tuple[dict[str, Any], int]:
    """Read one retained JSON object without exceeding the snapshot-wide budget."""

    if type(remaining_bytes) is not int or remaining_bytes < 1:
        raise TaskCRehearsalError(
            "rehearsal summary exceeded its total JSON evidence budget"
        )
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise TaskCRehearsalError(f"{label} could not be inspected safely") from exc
    if int(observed.st_size) > remaining_bytes:
        raise TaskCRehearsalError(
            "rehearsal summary exceeded its total JSON evidence budget"
        )
    payload_bytes, _identity = _capture_regular_bytes(
        path,
        label=label,
        maximum_bytes=min(MAXIMUM_REHEARSAL_JSON_BYTES, remaining_bytes),
    )
    return (
        _strict_json_payload(payload_bytes, label),
        remaining_bytes - len(payload_bytes),
    )


def _write_new_record(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    encoded = _strict_json_bytes(payload)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as exc:
        raise TaskCRehearsalError(
            f"analysis output already exists and will not be overwritten: {path.name}"
        ) from exc
    except OSError as exc:
        raise TaskCRehearsalError(
            f"analysis output could not be written safely: {path.name}"
        ) from exc


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise TaskCRehearsalError(f"analysis input cannot be read: {path}") from exc
    return f"sha256:{digest.hexdigest()}"


def _bytes_sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _record_sha256(payload: object) -> str:
    return _bytes_sha256(_strict_json_bytes(payload))


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _has_private_component(path: Path) -> bool:
    return any(part.casefold().startswith("private") for part in path.parts)


def _condition_scope(condition: str) -> dict[str, str | None]:
    if condition == "within_k562":
        return {
            "profile_condition": "within_environment",
            "context_id": "k562",
            "direction": None,
            "target_context": "k562",
        }
    if condition == "within_rpe1":
        return {
            "profile_condition": "within_environment",
            "context_id": "rpe1",
            "direction": None,
            "target_context": "rpe1",
        }
    if condition in {"k562_to_rpe1", "rpe1_to_k562"}:
        return {
            "profile_condition": "cross_environment",
            "context_id": condition,
            "direction": condition,
            "target_context": condition.split("_to_", 1)[1],
        }
    raise TaskCRehearsalError("rehearsal condition is not recognized")


def _safe_failure_reason(error: BaseException | str) -> str:
    value = str(error).strip() or "the rehearsal step did not complete"
    cleaned = " ".join(value.replace("\x00", " ").split())
    return cleaned[:500]


def _tree_inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> dict[str, str]:
    inventory: dict[str, str] = {}
    seen_inodes: set[tuple[int, int]] = set()
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise TaskCRehearsalError(
                f"rehearsal evidence could not be inspected: {relative}"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or int(metadata.st_nlink) != 1
        ):
            raise TaskCRehearsalError(
                "rehearsal evidence must use unique regular files without symbolic "
                f"or hard links: {relative}"
            )
        inode = (int(metadata.st_dev), int(metadata.st_ino))
        if inode in seen_inodes:
            raise TaskCRehearsalError(
                f"rehearsal evidence reuses one file inode: {relative}"
            )
        seen_inodes.add(inode)
        maximum = (
            MAXIMUM_REHEARSAL_PREDICTION_BYTES
            if path.suffix.casefold() == ".csv"
            else MAXIMUM_REHEARSAL_INPUT_BYTES
            if path.suffix.casefold() == ".npz"
            else MAXIMUM_REHEARSAL_JSON_BYTES
        )
        payload, _identity = _capture_regular_bytes(
            path,
            label=f"rehearsal evidence {relative}",
            maximum_bytes=maximum,
        )
        inventory[relative] = _bytes_sha256(payload)
    return inventory


def _reject_output_symbolic_links(path: Path) -> None:
    """Reject an output name reached through any symbolic-link component."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise TaskCRehearsalError(
                "output path components could not be inspected"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCRehearsalError(
                "output root and its parents must not use symbolic links"
            )


def _profile_inputs(
    *, public_manifest: Path, profile: str, staging: Path
) -> dict[str, dict[str, dict[str, Path]]]:
    from src.evaluation.task_c_profile_input import (
        TaskCProfileInputError,
        materialize_task_c_profile_input,
        validate_task_c_profile_input,
    )

    profiles: dict[str, dict[str, dict[str, Path]]] = {}
    for condition in REHEARSAL_CONDITIONS:
        scope = _condition_scope(condition)
        profiles[condition] = {}
        for stage in ("train", "tune", "refit"):
            output = staging / "profiles" / condition / stage
            try:
                created = materialize_task_c_profile_input(
                    public_manifest_path=public_manifest,
                    profile=profile,
                    condition=str(scope["profile_condition"]),
                    stage=stage,
                    context_id=scope["context_id"] if scope["direction"] is None else None,
                    direction=scope["direction"],
                    output_dir=output,
                )
                validated = validate_task_c_profile_input(
                    input_path=Path(created["input_npz"]),
                    profile_manifest_path=Path(created["manifest"]),
                    public_manifest_path=public_manifest,
                )
            except TaskCProfileInputError as exc:
                raise TaskCRehearsalError(
                    f"public profile input could not be verified: {exc}"
                ) from exc
            profiles[condition][stage] = {
                "input": validated.input_path,
                "manifest": validated.manifest_path,
            }
    return profiles


def _read_profile_arrays(profile_record: Mapping[str, Path]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    payload, _identity = _capture_regular_bytes(
        profile_record["input"],
        label="profile input NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            expression = np.asarray(archive["expression_matrix"], dtype=np.float64)
            labels = np.asarray(archive["interventions"], dtype=str)
            genes = tuple(str(value) for value in archive["var_names"].tolist())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TaskCRehearsalError("profile arrays could not be read") from exc
    return expression, labels, genes


def _read_profile_arrays_with_environments(
    profile_record: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray | None]:
    payload, _identity = _capture_regular_bytes(
        profile_record["input"],
        label="profile input NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            expression = np.asarray(archive["expression_matrix"], dtype=np.float64)
            labels = np.asarray(archive["interventions"], dtype=str)
            genes = tuple(str(value) for value in archive["var_names"].tolist())
            environments = (
                np.asarray(archive["environment_labels"], dtype=str)
                if "environment_labels" in archive.files
                else None
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TaskCRehearsalError("profile arrays could not be read") from exc
    return expression, labels, genes, environments


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as exc:
        raise TaskCRehearsalError(
            f"analysis output already exists and will not be overwritten: {path.name}"
        ) from exc
    except OSError as exc:
        raise TaskCRehearsalError(
            f"analysis output could not be written safely: {path.name}"
        ) from exc


def _capture_regular_bytes(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one regular single-link file through a no-follow descriptor."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    _reject_output_symbolic_links(absolute)
    try:
        descriptor = os.open(
            absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise TaskCRehearsalError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or int(before.st_size) < 1
            or int(before.st_size) > maximum_bytes
        ):
            raise TaskCRehearsalError(
                f"{label} must be one bounded regular file with one link"
            )
        chunks: list[bytes] = []
        collected = 0
        while collected <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - collected))
            if not chunk:
                break
            chunks.append(chunk)
            collected += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    payload = b"".join(chunks)
    if after_identity != identity or len(payload) != before.st_size:
        raise TaskCRehearsalError(f"{label} changed while it was being frozen")
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise TaskCRehearsalError(f"{label} changed after it was frozen") from exc
    current_identity = (
        int(current.st_dev),
        int(current.st_ino),
        int(current.st_size),
        int(current.st_mtime_ns),
        int(current.st_ctime_ns),
    )
    if (
        current_identity != identity
        or not stat.S_ISREG(current.st_mode)
        or int(current.st_nlink) != 1
    ):
        raise TaskCRehearsalError(f"{label} changed after it was frozen")
    return payload, identity


def freeze_rehearsal_predictions(
    *, source_path: Path, destination: Path, expected_genes: Sequence[str]
) -> FrozenRehearsalPredictions:
    """Freeze, bound, and parse the sole prediction bytes used downstream."""

    import pandas as pd

    payload, identity = _capture_regular_bytes(
        source_path,
        label="method prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    if payload.count(b"\n") > MAXIMUM_REHEARSAL_PREDICTION_ROWS + 1:
        raise TaskCRehearsalError("method prediction table has too many rows")
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8", on_bad_lines="error")
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise TaskCRehearsalError("method prediction table is not a bounded CSV") from exc
    if list(frame.columns) != [
        "source",
        "target",
        "score",
        "returned_by_method",
    ]:
        raise TaskCRehearsalError("method prediction table columns changed")
    genes = tuple(str(gene) for gene in expected_genes)
    expected_rows = len(genes) * (len(genes) - 1)
    relations = list(
        zip(frame["source"].astype(str), frame["target"].astype(str), strict=True)
    )
    if (
        len(frame) != expected_rows
        or len(set(relations)) != expected_rows
        or set(relations)
        != {(source, target) for source in genes for target in genes if source != target}
    ):
        raise TaskCRehearsalError("method prediction table is not the complete fixed universe")
    try:
        scores = np.asarray(frame["score"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaskCRehearsalError("method prediction scores are invalid") from exc
    if not np.isfinite(scores).all():
        raise TaskCRehearsalError("method prediction scores must be finite")
    _write_new_bytes(destination, payload)
    frozen_payload, _ = _capture_regular_bytes(
        destination,
        label="frozen prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    if frozen_payload != payload:
        raise TaskCRehearsalError("frozen prediction bytes changed during publication")
    return FrozenRehearsalPredictions(
        path=Path(destination),
        payload=payload,
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        source_path=Path(source_path),
        source_identity=identity,
        frame=frame,
    )


def _verify_frozen_predictions(snapshot: FrozenRehearsalPredictions) -> None:
    payload, _ = _capture_regular_bytes(
        snapshot.path,
        label="frozen prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    if payload != snapshot.payload:
        raise TaskCRehearsalError("frozen prediction bytes changed")


def _freeze_rehearsal_closure(
    paths: Mapping[str, Path],
    *,
    external_identity: Mapping[str, str] | None = None,
) -> FrozenRehearsalClosure:
    if not isinstance(paths, Mapping) or not paths:
        raise TaskCRehearsalError("rehearsal closure must contain fixed files")
    frozen: dict[str, FrozenRehearsalFile] = {}
    seen_paths: set[Path] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for label, raw_path in sorted(paths.items()):
        if not isinstance(label, str) or not label:
            raise TaskCRehearsalError("rehearsal closure labels must be non-empty")
        path = Path(raw_path)
        maximum = (
            MAXIMUM_REHEARSAL_REFERENCE_BYTES
            if path.suffix.casefold() == ".csv"
            else MAXIMUM_REHEARSAL_INPUT_BYTES
            if path.suffix.casefold() == ".npz"
            else MAXIMUM_METHOD_WORKER_BYTES
            if path.suffix.casefold() == ".py"
            else MAXIMUM_REHEARSAL_JSON_BYTES
        )
        payload, identity = _capture_regular_bytes(
            path,
            label=f"rehearsal closure file {label}",
            maximum_bytes=maximum,
        )
        absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
        inode = (identity[0], identity[1])
        if absolute in seen_paths or inode in seen_inodes:
            raise TaskCRehearsalError("rehearsal closure files must be unique")
        seen_paths.add(absolute)
        seen_inodes.add(inode)
        frozen[label] = FrozenRehearsalFile(
            path=absolute,
            sha256=_bytes_sha256(payload),
            identity=identity,
        )
    copied_external_identity = dict(external_identity or {})
    if any(
        not isinstance(label, str)
        or not label
        or not _is_sha256_text(value)
        for label, value in copied_external_identity.items()
    ):
        raise TaskCRehearsalError("rehearsal external identity is malformed")
    closure_record = {
        "files": {
            label: {
                "path": os.fspath(snapshot.path),
                "sha256": snapshot.sha256,
                "identity": list(snapshot.identity),
            }
            for label, snapshot in frozen.items()
        },
        "external_identity": dict(sorted(copied_external_identity.items())),
    }
    return FrozenRehearsalClosure(
        files=MappingProxyType(frozen),
        external_identity=MappingProxyType(copied_external_identity),
        sha256=_canonical_sha256(closure_record),
    )


def _verify_frozen_rehearsal_file(label: str, snapshot: FrozenRehearsalFile) -> None:
    maximum = (
        MAXIMUM_REHEARSAL_REFERENCE_BYTES
        if snapshot.path.suffix.casefold() == ".csv"
        else MAXIMUM_REHEARSAL_INPUT_BYTES
        if snapshot.path.suffix.casefold() == ".npz"
        else MAXIMUM_METHOD_WORKER_BYTES
        if snapshot.path.suffix.casefold() == ".py"
        else MAXIMUM_REHEARSAL_JSON_BYTES
    )
    payload, identity = _capture_regular_bytes(
        snapshot.path,
        label=f"rehearsal closure file {label}",
        maximum_bytes=maximum,
    )
    if (
        _bytes_sha256(payload) != snapshot.sha256
        or identity != snapshot.identity
    ):
        raise TaskCRehearsalError(
            f"rehearsal closure changed after launch: {label}"
        )


def _verify_rehearsal_closure(closure: FrozenRehearsalClosure) -> None:
    if not isinstance(closure, FrozenRehearsalClosure):
        raise TaskCRehearsalError("rehearsal closure is invalid")
    for label, snapshot in closure.files.items():
        _verify_frozen_rehearsal_file(label, snapshot)


def _verify_private_rehearsal_closure(
    closure: FrozenRehearsalClosure | None, *, prepared_root: Path
) -> None:
    if not isinstance(closure, FrozenRehearsalClosure):
        raise TaskCRehearsalError("private rehearsal closure is invalid")
    expected = {
        "private_manifest": prepared_root / "private/private_manifest.json",
        **{
            f"private_holdout:{relative}": prepared_root / relative
            for relative in _PRIVATE_HOLDOUT_PATHS
        },
    }
    for label, path in expected.items():
        snapshot = closure.files.get(label)
        if snapshot is None or snapshot.path != Path(
            os.path.abspath(os.fspath(path.expanduser()))
        ):
            raise TaskCRehearsalError(
                "private rehearsal closure lacks one fixed sealed input"
            )
        _verify_frozen_rehearsal_file(label, snapshot)


def materialize_hypersca_profile_contexts(
    *, profile_record: Mapping[str, Path], output_dir: Path
) -> Mapping[str, Path]:
    """Write the two already-frozen cross-environment contexts separately."""

    from src.evaluation.task_c_profile_input import _deterministic_npz

    expression, labels, genes, environments = _read_profile_arrays_with_environments(
        profile_record
    )
    manifest = _strict_json_file(profile_record["manifest"], "profile input record")
    if manifest.get("condition") != "cross_environment" or environments is None:
        raise TaskCRehearsalError(
            "separate HyperSCA-C profile contexts require one cross-environment profile"
        )
    contexts = manifest.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 2:
        raise TaskCRehearsalError(
            "cross-environment profile must record source and target-adapt contexts"
        )
    destination = Path(os.path.abspath(os.fspath(output_dir.expanduser())))
    if destination.exists() or destination.is_symlink():
        raise TaskCRehearsalError("HyperSCA-C context output already exists")
    destination.mkdir(parents=True, mode=0o700)
    created: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    for index, record in enumerate(contexts):
        if not isinstance(record, dict):
            raise TaskCRehearsalError("profile context record is invalid")
        context_id = record.get("context_id")
        role = record.get("role")
        if context_id not in {"k562", "rpe1"} or not isinstance(role, str):
            raise TaskCRehearsalError("profile context identity is invalid")
        selected = environments == context_id
        if not np.any(selected) or context_id in created:
            raise TaskCRehearsalError("profile context cells are incomplete")
        path = destination / (
            "source_profile.npz" if index == 0 else "target_adapt_profile.npz"
        )
        payload = _deterministic_npz(
            {
                "expression_matrix": expression[selected],
                "interventions": labels[selected],
                "var_names": np.asarray(genes),
            }
        )
        _write_new_bytes(path, payload)
        created[str(context_id)] = path
        records.append(
            {
                "context_id": context_id,
                "role": role,
                "file_name": path.name,
                "sha256": _sha256_file(path),
                "cell_count": int(selected.sum()),
                "parent_sha256": record.get("parent_sha256"),
                "selected_sorted_indices": record.get("selected_sorted_indices"),
            }
        )
    _write_new_record(
        destination / "context_manifest.json",
        {
            "schema_version": "1.0",
            "profile_input_sha256": _sha256_file(profile_record["input"]),
            "profile_manifest_sha256": _sha256_file(profile_record["manifest"]),
            "gene_order": list(genes),
            "contexts": records,
        },
    )
    return MappingProxyType(created)


def materialize_sealed_scoring_subset(
    *,
    source_path: Path,
    public_profile_input: Path,
    destination: Path,
    maximum_cells: int,
    seed: int,
) -> Path:
    """Create the private scoring view that matches the public method scope.

    The public profile supplies only the already fixed gene order.  Rows are
    then filtered and, when needed, sampled inside the sealed scoring step; the
    resulting file is never supplied to a comparison method.
    """

    if (
        isinstance(maximum_cells, bool)
        or type(maximum_cells) is not int
        or maximum_cells < 1
    ):
        raise TaskCRehearsalError("sealed scoring cell limit must be positive")
    profile_payload, _profile_identity = _capture_regular_bytes(
        public_profile_input,
        label="public profile NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    source_payload, _source_identity = _capture_regular_bytes(
        source_path,
        label="sealed scoring source NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    try:
        with np.load(io.BytesIO(profile_payload), allow_pickle=False) as profile_archive:
            profile_genes = _canonical_texts(
                np.asarray(profile_archive["var_names"]),
                "public profile genes",
                require_unique=True,
                maximum_items=MAXIMUM_PARENT_GENES,
            )
        with np.load(io.BytesIO(source_payload), allow_pickle=False) as source_archive:
            if set(source_archive.files) != {
                "expression_matrix",
                "interventions",
                "var_names",
            }:
                raise TaskCRehearsalError(
                    "sealed source must contain exactly the three registered arrays"
                )
            expression = np.asarray(source_archive["expression_matrix"])
            labels = np.asarray(source_archive["interventions"])
            source_genes = _canonical_texts(
                np.asarray(source_archive["var_names"]),
                "sealed source genes",
                require_unique=True,
                maximum_items=MAXIMUM_PARENT_GENES,
            )
    except TaskCRehearsalError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise TaskCRehearsalError("sealed scoring arrays could not be read") from exc
    source_by_gene = {gene: index for index, gene in enumerate(source_genes)}
    if any(gene not in source_by_gene for gene in profile_genes):
        raise TaskCRehearsalError(
            "public profile genes are absent from the sealed scoring source"
        )
    canonical_labels = _canonical_texts(
        labels,
        "sealed intervention labels",
        require_unique=False,
        maximum_items=MAXIMUM_PARENT_CELLS,
    )
    values = _numeric_matrix(
        expression,
        "sealed expression",
        expected_columns=len(source_genes),
    )
    profile_gene_set = set(profile_genes)
    retained = np.fromiter(
        (
            label in {CONTROL_LABEL, "excluded"} or label in profile_gene_set
            for label in canonical_labels
        ),
        dtype=bool,
        count=len(canonical_labels),
    )
    if not np.any(retained):
        raise TaskCRehearsalError("sealed scoring scope retained no cells")
    selected_labels = np.asarray(canonical_labels, dtype=str)[retained]
    selected_expression = values[retained][
        :, [source_by_gene[gene] for gene in profile_genes]
    ]
    selected_indices = choose_rehearsal_cells(
        selected_labels,
        min(maximum_cells, len(selected_labels)),
        seed,
        minimum_cells_per_group=1,
    )
    selected_expression = selected_expression[selected_indices]
    selected_labels = selected_labels[selected_indices]
    if CONTROL_LABEL not in set(selected_labels.tolist()):
        raise TaskCRehearsalError("sealed scoring scope retained no control cells")

    destination = Path(os.path.abspath(os.fspath(destination.expanduser())))
    private = _real_private_directory(destination.parent)
    if destination.parent.resolve(strict=True) != private or destination.exists():
        raise TaskCRehearsalError(
            "sealed scoring destination must be a new file in the private root"
        )
    from src.evaluation.task_c_profile_input import _deterministic_npz

    payload = _deterministic_npz(
        {
            "expression_matrix": np.asarray(selected_expression),
            "interventions": np.asarray(selected_labels),
            "var_names": np.asarray(profile_genes),
        }
    )
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise TaskCRehearsalError(
            "sealed scoring subset could not be written safely"
        ) from exc
    return destination


def _synthetic_predictions(
    *, method_id: str, profile_record: Mapping[str, Path], seed: int
) -> Any:
    import pandas as pd

    from src.evaluation.task_c_benchmark import (
        TaskCBenchmarkError,
        score_mean_difference_network,
    )
    from src.evaluation.task_c_predictions import (
        TaskCPredictionError,
        normalize_task_c_predictions,
    )

    expression, labels, genes = _read_profile_arrays(profile_record)
    if method_id in {"hypersca_c", "mean_difference"}:
        try:
            raw = score_mean_difference_network(
                expression,
                labels,
                genes,
                min_cells_per_intervention=1,
            ).scores[["source", "target", "score"]]
        except TaskCBenchmarkError as exc:
            raise TaskCRehearsalError(str(exc)) from exc
    else:
        relations = [
            (source, target)
            for source in genes
            for target in genes
            if source != target
        ]
        rng = np.random.default_rng(seed + int.from_bytes(method_id.encode("utf-8")[:4], "little"))
        order = rng.permutation(len(relations))
        limit = min(1_000, len(relations))
        raw = pd.DataFrame(
            [(*relations[int(index)], float(limit - rank)) for rank, index in enumerate(order[:limit])],
            columns=["source", "target", "score"],
        )
    try:
        return normalize_task_c_predictions(raw, genes)
    except TaskCPredictionError as exc:
        raise TaskCRehearsalError(str(exc)) from exc


def _null_control_records(
    *,
    predictions: Any | None,
    prediction_snapshot: FrozenRehearsalPredictions | None = None,
    profile_record: Mapping[str, Path],
    seed: int,
    synthetic_smoke: bool,
) -> dict[str, object]:
    from sklearn.metrics import average_precision_score

    from src.evaluation.task_c_null_controls import (
        build_control_resampling_null,
        empirical_null_check,
        null_check_to_json_record,
        permute_intervention_labels,
    )
    from src.evaluation.task_c_tuning import (
        TaskCTuningError,
        build_tuning_response_edges,
    )

    expression, labels, genes = _read_profile_arrays(profile_record)
    relations = list(
        zip(
            predictions["source"].astype(str),
            predictions["target"].astype(str),
            strict=True,
        )
    )
    scores = np.asarray(predictions["score"], dtype=float)

    def response_metric(values: np.ndarray, groups: np.ndarray) -> float:
        sources = set(groups.tolist()) - {CONTROL_LABEL, "excluded"}
        try:
            positives = build_tuning_response_edges(
                values,
                groups,
                genes,
                eligible_sources=sources,
                q_value_threshold=0.1,
            )
        except TaskCTuningError:
            return 0.0
        if not positives:
            return 0.0
        truth = np.asarray([relation in positives for relation in relations], dtype=int)
        if int(truth.sum()) in {0, len(truth)}:
            return 0.0
        return float(average_precision_score(truth, scores))

    if synthetic_smoke:
        real_metric = 1.0
        label_metrics = [0.0] * 20
        resampling_metrics = [0.0] * 20
    else:
        real_metric = response_metric(expression, labels)
        label_metrics = []
        resampling_metrics = []
        for repeat in range(20):
            label_seed = seed + repeat + 1
            resampling_seed = seed + repeat + 101
            permuted = permute_intervention_labels(labels, label_seed)
            label_metrics.append(response_metric(expression, permuted))
            sampled_expression, sampled_labels = build_control_resampling_null(
                expression,
                labels,
                resampling_seed,
            )
            resampling_metrics.append(
                response_metric(sampled_expression, sampled_labels)
            )

    result: dict[str, object] = {}
    for name, values, offset, transformation in (
        (
            "label_permutation",
            label_metrics,
            1,
            "intervention labels shuffled while retaining group sizes",
        ),
        (
            "control_resampling",
            resampling_metrics,
            101,
            "all expression rows resampled from control cells",
        ),
    ):
        checked = empirical_null_check(real_metric, values, 0.05, 0.0)
        record: dict[str, object] = dict(null_check_to_json_record(checked))
        record.update(
            {
                "seeds": [seed + repeat + offset for repeat in range(20)],
                "metrics": [float(value) for value in values],
                "transformation": transformation,
            }
        )
        result[name] = record
    return result


def _scientific_null_predictions(
    *,
    method_id: str,
    profile_record: Mapping[str, Path],
    seed: int,
    min_cells: int,
    hypersca_config_path: Path | None,
) -> Any:
    """Run the registered scientific rule on one materialized null input."""

    from src.evaluation.task_c_predictions import normalize_task_c_predictions

    expression, labels, genes, environments = _read_profile_arrays_with_environments(
        profile_record
    )
    if method_id == "mean_difference":
        from src.evaluation.task_c_benchmark import score_mean_difference_network

        raw = score_mean_difference_network(
            expression,
            labels,
            genes,
            control_label=CONTROL_LABEL,
            excluded_label="excluded",
            min_cells_per_intervention=min_cells,
        ).scores[["source", "target", "score"]]
    elif method_id == "hypersca_c":
        if hypersca_config_path is None:
            raise TaskCRehearsalError(
                "HyperSCA-C null inference requires the selected fixed settings"
            )
        from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext
        from src.causal.hypersca_c_stability import fit_stable_hypersca_c

        config = HyperSCACConfig.from_mapping(
            _strict_json_file(hypersca_config_path, "HyperSCA-C null settings")
        )
        manifest = _strict_json_file(
            profile_record["manifest"], "null profile input record"
        )
        contexts_record = manifest.get("contexts")
        if not isinstance(contexts_record, list) or not contexts_record:
            raise TaskCRehearsalError("null profile lacks its fixed contexts")
        context_ids = tuple(
            str(record.get("context_id"))
            for record in contexts_record
            if isinstance(record, dict)
        )
        if not context_ids:
            raise TaskCRehearsalError("null profile lacks its fixed contexts")
        contexts = []
        for context_id in context_ids:
            selected = (
                np.ones(len(labels), dtype=bool)
                if environments is None
                else environments == context_id
            )
            if not np.any(selected):
                raise TaskCRehearsalError(
                    "HyperSCA-C null input lacks one fixed context"
                )
            contexts.append(
                HyperSCACContext(
                    context_id=context_id,
                    expression=expression[selected],
                    interventions=labels[selected],
                    gene_names=genes,
                )
            )
        fitted = fit_stable_hypersca_c(
            contexts,
            config,
            seed=seed,
            device="cpu",
        )
        summary = dict(fitted.summary)
        successful = summary.get("successful_repeats")
        coverage = summary.get("coverage")
        if (
            isinstance(successful, bool)
            or not isinstance(successful, int)
            or successful < 1
            or isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not math.isfinite(float(coverage))
            or float(coverage) <= 0.0
        ):
            raise TaskCRehearsalError(
                "HyperSCA-C formal null fit has no successful, usable coverage"
            )
        usable = fitted.predictions.loc[
            ~fitted.predictions["abstained"].astype(bool)
            & np.isfinite(np.asarray(fitted.predictions["score"], dtype=np.float64))
        ]
        if usable.empty:
            raise TaskCRehearsalError(
                "HyperSCA-C formal null fit returned no usable relations"
            )
        raw = usable[["source", "target", "score"]]
    else:
        raise TaskCRehearsalError(
            "formal null inference is limited to HyperSCA-C and Mean Difference"
        )
    normalized = normalize_task_c_predictions(raw, genes)
    normalized.attrs["formal_null_scientific_status"] = {
        "successful_repeats": int(summary["successful_repeats"]),
        "requested_repeats": int(summary["requested_repeats"]),
        "coverage": float(summary["coverage"]),
    } if method_id == "hypersca_c" else {
        "successful_repeats": 1,
        "requested_repeats": 1,
        "coverage": float(normalized["returned_by_method"].mean()),
    }
    return normalized


def _response_average_precision(
    predictions: Any | None,
    expression: np.ndarray,
    labels: np.ndarray,
    genes: tuple[str, ...],
) -> float:
    from sklearn.metrics import average_precision_score

    from src.evaluation.task_c_tuning import (
        TaskCTuningError,
        build_tuning_response_edges,
    )

    try:
        positives = build_tuning_response_edges(
            expression,
            labels,
            genes,
            eligible_sources=set(labels.tolist()) - {CONTROL_LABEL, "excluded"},
            q_value_threshold=0.1,
        )
    except TaskCTuningError:
        return 0.0
    relations = list(
        zip(
            predictions["source"].astype(str),
            predictions["target"].astype(str),
            strict=True,
        )
    )
    truth = np.asarray([relation in positives for relation in relations], dtype=int)
    if int(truth.sum()) in {0, len(truth)}:
        return 0.0
    return float(
        average_precision_score(
            truth,
            np.asarray(predictions["score"], dtype=np.float64),
        )
    )


def _derive_formal_null_seed(
    *,
    base_seed: int,
    control_index: int,
    repeat: int,
    context_index: int,
    purpose: int,
) -> int:
    coordinates = (control_index, repeat, context_index, purpose)
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in coordinates)
    ):
        raise TaskCRehearsalError("formal null seed coordinates are invalid")
    sequence = np.random.SeedSequence(base_seed, spawn_key=coordinates)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _materialize_formal_null_hypersca_config(
    *, selected_config: Path, destination: Path
) -> Path:
    from src.causal.hypersca_c import HyperSCACConfig

    selected = _strict_json_file(
        selected_config, "selected HyperSCA-C settings for formal null analyses"
    )
    frozen_settings = dict(selected)
    HyperSCACConfig.from_mapping(frozen_settings)
    _write_new_record(destination, frozen_settings)
    return destination


def _contextwise_label_permutation(
    labels: np.ndarray,
    environments: np.ndarray | None,
    seeds: Sequence[int],
) -> np.ndarray:
    from src.evaluation.task_c_null_controls import permute_intervention_labels

    if environments is None:
        if len(seeds) != 1:
            raise TaskCRehearsalError("within-context null requires one derived seed")
        return permute_intervention_labels(labels, int(seeds[0]))
    context_ids = tuple(dict.fromkeys(environments.tolist()))
    if len(seeds) != len(context_ids):
        raise TaskCRehearsalError("cross-context null seeds are incomplete")
    transformed = labels.copy()
    for context_index, context_id in enumerate(context_ids):
        selected = environments == context_id
        transformed[selected] = permute_intervention_labels(
            labels[selected], int(seeds[context_index])
        )
    return transformed


def _contextwise_control_resampling(
    expression: np.ndarray,
    labels: np.ndarray,
    environments: np.ndarray | None,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    from src.evaluation.task_c_null_controls import build_control_resampling_null

    if environments is None:
        if len(seeds) != 1:
            raise TaskCRehearsalError("within-context null requires one derived seed")
        return build_control_resampling_null(expression, labels, int(seeds[0]))
    context_ids = tuple(dict.fromkeys(environments.tolist()))
    if len(seeds) != len(context_ids):
        raise TaskCRehearsalError("cross-context null seeds are incomplete")
    transformed = expression.copy()
    copied_labels = labels.copy()
    for context_index, context_id in enumerate(context_ids):
        selected = environments == context_id
        sampled, sampled_labels = build_control_resampling_null(
            expression[selected], labels[selected], int(seeds[context_index])
        )
        transformed[selected] = sampled
        copied_labels[selected] = sampled_labels
    return transformed, copied_labels


def _directory_written_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    if root.is_file() and not root.is_symlink():
        return int(os.lstat(root).st_size)
    for path in root.rglob("*"):
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise TaskCRehearsalError(
                "null-control output size could not be measured"
            ) from exc
        if stat.S_ISREG(metadata.st_mode):
            total += int(metadata.st_size)
    return total


def _measured_stage_fields(
    *, output_root: Path, elapsed_seconds: float
) -> dict[str, object]:
    peak_rss_bytes: int | None = None
    inner_resource = output_root / "raw_runtime/resource_usage.json"
    if inner_resource.is_file():
        resource = _strict_json_file(inner_resource, "method resource record")
        maximum_resident_kib = resource.get("maximum_resident_kib")
        if (
            isinstance(maximum_resident_kib, int)
            and not isinstance(maximum_resident_kib, bool)
            and maximum_resident_kib >= 0
        ):
            peak_rss_bytes = maximum_resident_kib * 1024
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_gpu_memory_bytes": None,
        "written_disk_bytes": _directory_written_bytes(output_root),
        "measurement_availability": {
            "elapsed_seconds": True,
            "peak_rss_bytes": peak_rss_bytes is not None,
            "peak_gpu_memory_bytes": False,
            "written_disk_bytes": True,
        },
    }


def _run_supervised_null_inference(
    *,
    method_id: str,
    profile_record: Mapping[str, Path],
    seed: int,
    min_cells: int,
    hypersca_config_path: Path | None,
    repeat_root: Path,
    expected_genes: Sequence[str],
    timeout_seconds: float,
) -> tuple[Any | None, dict[str, object]]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise TaskCRehearsalError("formal null timeout must be positive")
    from src.evaluation.task_c_runtime import (
        TaskCRuntimeError,
        run_isolated_method,
    )

    raw_prediction_path = repeat_root / "raw_predictions.csv"
    scientific_status_path = repeat_root / "scientific_status.json"
    baseline_bytes = _directory_written_bytes(repeat_root)
    worker = Path(__file__).resolve().parents[2] / (
        "scripts/task_c_workers/rehearsal_null_worker.py"
    )
    command = [
        sys.executable,
        "-I",
        str(worker),
        "--method-id",
        method_id,
        "--input-npz",
        str(profile_record["input"]),
        "--profile-manifest",
        str(profile_record["manifest"]),
        "--seed",
        str(seed),
        "--min-cells",
        str(min_cells),
        "--output-csv",
        str(raw_prediction_path),
        "--scientific-status",
        str(scientific_status_path),
    ]
    if hypersca_config_path is not None:
        command.extend(("--hypersca-config", str(hypersca_config_path)))
    try:
        runtime = run_isolated_method(
            command,
            output_dir=repeat_root / "supervision",
            timeout_seconds=timeout_seconds,
        )
    except TaskCRuntimeError as exc:
        raise TaskCRehearsalError(
            f"formal null supervisor could not start: {exc}"
        ) from exc
    resource = _strict_json_file(
        repeat_root / "supervision/resource_usage.json",
        "formal null supervisor resource record",
    )
    runtime_status = runtime.get("status")
    status = (
        "completed"
        if runtime_status == "completed_raw_inference"
        else "failed_timeout"
        if runtime_status == "failed_timeout"
        else "failed"
    )
    predictions: Any | None = None
    prediction_sha256: str | None = None
    scientific_status: object = None
    if status == "completed":
        child_status = _strict_json_file(
            scientific_status_path, "formal null scientific status"
        )
        if child_status.get("status") != "completed" or not isinstance(
            child_status.get("scientific_status"), dict
        ):
            status = "failed"
        else:
            scientific_status = child_status["scientific_status"]
    if status == "completed":
        snapshot = freeze_rehearsal_predictions(
            source_path=raw_prediction_path,
            destination=repeat_root / "predictions.csv",
            expected_genes=expected_genes,
        )
        predictions = snapshot.frame
        prediction_sha256 = snapshot.sha256
    written_bytes = max(0, _directory_written_bytes(repeat_root) - baseline_bytes)
    maximum_resident_kib = resource.get("maximum_resident_kib")
    peak_rss = (
        int(maximum_resident_kib) * 1024
        if isinstance(maximum_resident_kib, int)
        and not isinstance(maximum_resident_kib, bool)
        and maximum_resident_kib >= 0
        else None
    )
    record: dict[str, object] = {
        "schema_version": "1.0",
        "component_kind": "null_analysis",
        "stage": "null_control",
        "status": status,
        "return_code": runtime.get("return_code"),
        "timeout_seconds": float(timeout_seconds),
        "elapsed_seconds": resource.get("elapsed_seconds"),
        "peak_rss_bytes": peak_rss,
        "peak_gpu_memory_bytes": None,
        "written_disk_bytes": written_bytes,
        "measurement_availability": {
            "elapsed_seconds": True,
            "peak_rss_bytes": peak_rss is not None,
            "peak_gpu_memory_bytes": False,
            "written_disk_bytes": True,
        },
        "resource_meter": resource.get("resource_meter"),
        "prediction_sha256": prediction_sha256,
        "scientific_status": scientific_status,
    }
    if status != "completed":
        reason = runtime.get("stderr_tail") or runtime_status
        record["reason"] = _safe_failure_reason(str(reason))
    return predictions, record


def _run_formal_null_controls(
    *,
    method_id: str,
    predictions: Any,
    profile_record: Mapping[str, Path],
    seed: int,
    min_cells: int,
    hypersca_config_path: Path | None,
    work_dir: Path,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Materialize and analyze all 40 formal zero-effect inputs."""

    from src.evaluation.task_c_null_controls import (
        empirical_null_check,
        null_check_to_json_record,
    )
    from src.evaluation.task_c_profile_input import _deterministic_npz

    expression, labels, genes, environments = _read_profile_arrays_with_environments(
        profile_record
    )
    real_metric = _response_average_precision(
        predictions, expression, labels, genes
    )
    root = work_dir / "null_controls"
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    formal_hypersca_config = hypersca_config_path
    if method_id == "hypersca_c":
        if hypersca_config_path is None:
            raise TaskCRehearsalError(
                "HyperSCA-C formal null analyses require selected settings"
            )
        formal_hypersca_config = _materialize_formal_null_hypersca_config(
            selected_config=hypersca_config_path,
            destination=root / "hypersca_formal_null_config.json",
        )
    failures: list[dict[str, object]] = []
    used_seeds: set[int] = set()
    result: dict[str, object] = {
        "scope": "formal_scientific_inference_rerun",
        "formal_null_gate_passed": False,
    }
    context_ids = (
        ("within",)
        if environments is None
        else tuple(str(value) for value in dict.fromkeys(environments.tolist()))
    )
    for control_index, control_name in enumerate(
        ("label_permutation", "control_resampling")
    ):
        metrics: list[float] = []
        analyses: list[dict[str, object]] = []
        for repeat in range(20):
            transformation_seeds = tuple(
                _derive_formal_null_seed(
                    base_seed=seed,
                    control_index=control_index,
                    repeat=repeat,
                    context_index=context_index,
                    purpose=0,
                )
                for context_index in range(len(context_ids))
            )
            repeat_seed = _derive_formal_null_seed(
                base_seed=seed,
                control_index=control_index,
                repeat=repeat,
                context_index=0,
                purpose=1,
            )
            current_seeds = {*transformation_seeds, repeat_seed}
            if len(current_seeds) != len(transformation_seeds) + 1 or (
                used_seeds & current_seeds
            ):
                raise TaskCRehearsalError("formal null seed streams overlap")
            used_seeds.update(current_seeds)
            repeat_root = root / control_name / f"repeat_{repeat:02d}"
            repeat_root.mkdir(parents=True, mode=0o700)
            if control_name == "label_permutation":
                transformed_expression = expression
                transformed_labels = _contextwise_label_permutation(
                    labels, environments, transformation_seeds
                )
            else:
                transformed_expression, transformed_labels = (
                    _contextwise_control_resampling(
                        expression, labels, environments, transformation_seeds
                    )
                )
            arrays: dict[str, np.ndarray] = {
                "expression_matrix": transformed_expression,
                "interventions": transformed_labels,
                "var_names": np.asarray(genes),
            }
            if environments is not None:
                arrays["environment_labels"] = environments
            input_path = repeat_root / "input.npz"
            _write_new_bytes(input_path, _deterministic_npz(arrays))
            input_sha256 = _sha256_file(input_path)
            identity = {
                "schema_version": "1.0",
                "method_id": method_id,
                "control": control_name,
                "repeat": repeat,
                "model_seed": repeat_seed,
                "transformation_seeds": {
                    context_id: transformation_seeds[index]
                    for index, context_id in enumerate(context_ids)
                },
                "seed_derivation": "numpy_seed_sequence_v1",
                "parent_profile_input_sha256": _sha256_file(
                    profile_record["input"]
                ),
                "parent_profile_manifest_sha256": _sha256_file(
                    profile_record["manifest"]
                ),
                "input_sha256": input_sha256,
                "hypersca_config_sha256": (
                    _sha256_file(formal_hypersca_config)
                    if formal_hypersca_config is not None
                    else None
                ),
            }
            _write_new_record(repeat_root / "input_identity.json", identity)
            metric: float | None = None
            transformed_profile = {
                "input": input_path,
                "manifest": profile_record["manifest"],
            }
            null_predictions, analysis = _run_supervised_null_inference(
                method_id=method_id,
                profile_record=transformed_profile,
                seed=repeat_seed,
                min_cells=min_cells,
                hypersca_config_path=formal_hypersca_config,
                repeat_root=repeat_root,
                expected_genes=genes,
                timeout_seconds=timeout_seconds,
            )
            if null_predictions is not None:
                metric = _response_average_precision(
                    null_predictions,
                    np.asarray(transformed_expression),
                    np.asarray(transformed_labels),
                    genes,
                )
                metrics.append(metric)
            analysis.update({
                "method_id": method_id,
                "control": control_name,
                "repeat": repeat,
                "seed": repeat_seed,
                "input_sha256": input_sha256,
                "metric": metric,
            })
            if analysis["status"] != "completed":
                failures.append(dict(analysis))
            _write_new_record(repeat_root / "analysis.resource.json", analysis)
            analyses.append(dict(analysis))
        control_record: dict[str, object] = {
            "seeds": [record["seed"] for record in analyses],
            "metrics": metrics,
            "analyses": analyses,
            "completed_analysis_count": sum(
                record["status"] == "completed" for record in analyses
            ),
        }
        if len(metrics) == 20:
            checked = empirical_null_check(real_metric, metrics, 0.05, 0.0)
            control_record.update(null_check_to_json_record(checked))
        result[control_name] = control_record
    if failures:
        _write_new_record(
            root / "null_control_status.json",
            {
                "schema_version": "1.0",
                "status": "failed_null_control",
                "attempted_analysis_count": 40,
                "completed_analysis_count": 40 - len(failures),
                "failures": failures,
            },
        )
        raise TaskCRehearsalError(
            "null-control analyses did not all complete; retained failure records"
        )
    result["formal_null_gate_passed"] = all(
        bool(result[name].get("passed"))  # type: ignore[union-attr]
        for name in ("label_permutation", "control_resampling")
    )
    final_status = (
        "completed_formal_null_controls"
        if result["formal_null_gate_passed"]
        else "failed_null_control"
    )
    _write_new_record(
        root / "null_control_status.json",
        {
            "schema_version": "1.0",
            "status": final_status,
            "attempted_analysis_count": 40,
            "completed_analysis_count": 40,
        },
    )
    if not result["formal_null_gate_passed"]:
        raise TaskCRehearsalError(
            "formal null-control metrics did not pass the fixed empirical gate"
        )
    return result


def _hypersca_gene_list(
    *, profile_record: Mapping[str, Path], destination: Path, profile: str
) -> Path:
    manifest = _strict_json_file(
        profile_record["manifest"], "profile input record"
    )
    selection = manifest.get("gene_selection")
    if not isinstance(selection, dict) or not isinstance(
        selection.get("ordered_genes"), list
    ):
        raise TaskCRehearsalError("profile input record lacks the fixed gene order")
    _write_new_record(
        destination,
        {
            "schema_version": "1.0",
            "selection_id": f"task-c-{profile}-seed-11",
            "selection_basis": (
                "公开训练对照细胞中共同表达基因的变异度排序；"
                "该清单只固定本次对照评估范围。"
            ),
            "genes": selection["ordered_genes"],
        },
    )
    return destination


def _hypersca_trial_configs(
    *, base_config: Path, work_dir: Path
) -> tuple[Path, Path]:
    base = _strict_json_file(base_config, "HyperSCA-C fixed settings")
    second = dict(base)
    shared_l1 = second.get("shared_l1")
    if isinstance(shared_l1, bool) or not isinstance(shared_l1, (int, float)):
        raise TaskCRehearsalError("HyperSCA-C shared_l1 setting is invalid")
    second["shared_l1"] = float(shared_l1) * 2.0
    first_path = work_dir / "hypersca_config_trial_0.json"
    second_path = work_dir / "hypersca_config_trial_1.json"
    _write_new_record(first_path, base)
    _write_new_record(second_path, second)
    return first_path, second_path


def _run_method_bundle(
    *,
    method_id: str,
    profile_record: Mapping[str, Path] | None,
    output_dir: Path,
    seed: int,
    registry_path: Path,
    asset_root: Path,
    public_manifest: Path,
    context_id: str,
    min_cells: int,
    timeout_seconds: int,
    project_root: Path,
    hypersca_config: Path | None = None,
    gene_list: Path | None = None,
    trial_candidate: Path | None = None,
    selection_arguments: Mapping[str, object] | None = None,
    resource_record_path: Path | None = None,
    resource_stage: str | None = None,
) -> dict[str, object]:
    from src.evaluation.task_c_method_run import (
        TaskCMethodRunError,
        run_task_c_method,
    )

    arguments: dict[str, object] = {
        "method_id": method_id,
        "input_npz": profile_record["input"] if profile_record else None,
        "output_dir": output_dir,
        "seed": seed,
        "registry_path": registry_path,
        "asset_root": asset_root,
        "data_status": "external_benchmark" if profile_record else None,
        "context_id": context_id if profile_record else None,
        "min_cells": min_cells,
        "public_manifest_path": public_manifest if profile_record else None,
        "derived_input_manifest_path": (
            profile_record["manifest"] if profile_record else None
        ),
        "hypersca_config_path": hypersca_config,
        "gene_list_path": gene_list,
        "timeout_seconds": timeout_seconds,
        "trial_parameters_path": trial_candidate,
        "project_root": project_root,
    }
    if selection_arguments:
        arguments.update(selection_arguments)
    started = time.monotonic()
    final_status = "raised_before_status"
    try:
        result = run_task_c_method(**arguments)  # type: ignore[arg-type]
        final_status = str(result.get("status", "missing_status"))
        return result
    except TaskCMethodRunError:
        final_status = "raised_task_c_method_error"
        raise
    finally:
        if resource_record_path is not None:
            elapsed = max(0.0, time.monotonic() - started)
            _write_new_record(
                resource_record_path,
                {
                    "schema_version": "1.0",
                    "component_kind": "method_analysis",
                    "method_id": method_id,
                    "stage": resource_stage or "unspecified",
                    "status": final_status,
                    **_measured_stage_fields(
                        output_root=output_dir,
                        elapsed_seconds=elapsed,
                    ),
                },
            )


def _method_bundle_status(path: Path) -> tuple[str | None, str | None]:
    status_path = path / "method_status.json"
    if not status_path.is_file():
        return None, None
    payload = _strict_json_file(status_path, "method result status")
    status = payload.get("status")
    reason = payload.get("reason")
    return (
        status if isinstance(status, str) else None,
        reason if isinstance(reason, str) else None,
    )


def _find_failed_method_bundle(
    work_dir: Path,
) -> tuple[Path | None, str | None, str | None]:
    """Find a recorded inner failure without relabeling it as a generic error."""

    for status_path in sorted(work_dir.rglob("method_status.json")):
        status, reason = _method_bundle_status(status_path.parent)
        if status in _INNER_TO_REHEARSAL_STATUS and status != (
            "completed_standardized_output"
        ):
            return status_path.parent, status, reason
    return None, None, None


def _select_connection_configuration(
    *,
    method_id: str,
    condition: str,
    profiles: Mapping[str, Mapping[str, Path]],
    work_dir: Path,
    seed: int,
    registry_path: Path,
    asset_root: Path,
    public_manifest: Path,
    min_cells: int,
    timeout_seconds: int,
    project_root: Path,
    base_hypersca_config: Path,
) -> tuple[Path, dict[str, object]]:
    if method_id != "hypersca_c":
        raise TaskCRehearsalError(
            "connection selection is reserved for HyperSCA-C; reference methods use fixed settings"
        )
    trial_root = work_dir / "trials"
    trial_root.mkdir(parents=True, mode=0o700)
    gene_list: Path | None = None
    configs: tuple[Path | None, Path | None] = (None, None)
    if method_id == "hypersca_c":
        gene_list = _hypersca_gene_list(
            profile_record=profiles["train"],
            destination=work_dir / "genes.json",
            profile="connection",
        )
        configs = _hypersca_trial_configs(
            base_config=base_hypersca_config,
            work_dir=work_dir,
        )

    trial_dirs: list[Path] = []
    for trial_index in (0, 1):
        candidate = work_dir / f"trial_candidate_{trial_index}.json"
        parameters = (
            _strict_json_file(configs[trial_index], "HyperSCA-C trial settings")
            if configs[trial_index] is not None
            else {}
        )
        _write_new_record(
            candidate,
            {
                "schema_version": "1.0",
                "trial_index": trial_index,
                "parameters": parameters,
            },
        )
        trial_dir = trial_root / f"trial_{trial_index}"
        _run_method_bundle(
            method_id=method_id,
            profile_record=profiles["train"],
            output_dir=trial_dir,
            seed=seed,
            registry_path=registry_path,
            asset_root=asset_root,
            public_manifest=public_manifest,
            context_id=condition.replace("within_", "") if condition.startswith("within_") else condition,
            min_cells=min_cells,
            timeout_seconds=timeout_seconds,
            project_root=project_root,
            hypersca_config=configs[trial_index],
            gene_list=gene_list,
            trial_candidate=candidate,
            resource_record_path=work_dir / f"train_trial_{trial_index}.resource.json",
            resource_stage="train",
        )
        inner_status, _reason = _method_bundle_status(trial_dir)
        if inner_status != "completed_standardized_output":
            raise TaskCRehearsalError(
                f"training trial {trial_index} did not produce a complete relation table"
            )
        from src.evaluation.task_c_method_run import (
            TaskCMethodRunError,
            validate_task_c_method_output_bundle,
        )

        try:
            validate_task_c_method_output_bundle(
                output_dir=trial_dir,
                input_npz=profiles["train"]["input"],
                registry_path=registry_path,
                asset_root=asset_root,
                public_manifest_path=public_manifest,
                derived_input_manifest_path=profiles["train"]["manifest"],
                hypersca_config_path=configs[trial_index],
                gene_list_path=gene_list,
                project_root=project_root,
            )
        except TaskCMethodRunError as exc:
            raise TaskCRehearsalError(
                f"training trial {trial_index} failed reconstruction: {exc}"
            ) from exc
        trial_dirs.append(trial_dir)

    selection = work_dir / "selection_record.json"
    selection_status = Path(f"{selection}.status.json")
    command = [
        sys.executable,
        str(project_root / "scripts/select_task_c_configuration.py"),
        "--tune-npz",
        str(profiles["tune"]["input"]),
        "--profile-manifest",
        str(profiles["tune"]["manifest"]),
        "--public-manifest",
        str(public_manifest),
        "--output-json",
        str(selection),
        "--status-json",
        str(selection_status),
        "--config",
        str(project_root / "configs/task_c_tuning_v1.json"),
        "--registry",
        str(registry_path),
        "--asset-root",
        str(asset_root),
    ]
    for index, trial_dir in enumerate(trial_dirs):
        command.extend(("--trial-dir", str(trial_dir)))
        command.extend(
            ("--trial-input", f"{trial_dir}={profiles['train']['input']}")
        )
        command.extend(
            (
                "--trial-profile-manifest",
                f"{trial_dir}={profiles['train']['manifest']}",
            )
        )
        if method_id == "hypersca_c":
            assert configs[index] is not None and gene_list is not None
            command.extend(
                ("--trial-hypersca-config", f"{trial_dir}={configs[index]}")
            )
            command.extend(("--trial-gene-list", f"{trial_dir}={gene_list}"))
    selection_started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    selection_elapsed = max(0.0, time.monotonic() - selection_started)
    selection_written = sum(
        int(os.lstat(path).st_size)
        for path in (selection, selection_status)
        if path.is_file() and not path.is_symlink()
    )
    _write_new_record(
        work_dir / "tune_selection.resource.json",
        {
            "schema_version": "1.0",
            "component_kind": "configuration_selection",
            "method_id": method_id,
            "stage": "tune",
            "status": "completed" if completed.returncode == 0 else "failed",
            "elapsed_seconds": selection_elapsed,
            "peak_rss_bytes": None,
            "peak_gpu_memory_bytes": None,
            "written_disk_bytes": selection_written,
            "measurement_availability": {
                "elapsed_seconds": True,
                "peak_rss_bytes": False,
                "peak_gpu_memory_bytes": False,
                "written_disk_bytes": True,
            },
        },
    )
    if completed.returncode != 0:
        raise TaskCRehearsalError(
            "public tuning selection failed: " + _safe_failure_reason(completed.stderr)
        )
    selected = _strict_json_file(selection, "selection record")
    selected_parameters = selected.get("selected_parameters")
    if not isinstance(selected_parameters, dict):
        raise TaskCRehearsalError("selection record lacks selected settings")
    refit_config: Path | None = None
    if method_id == "hypersca_c":
        refit_config = work_dir / "selected_refit_config.json"
        _write_new_record(refit_config, selected_parameters)

    refit = work_dir / "refit"
    input_bindings = {
        trial.resolve(): profiles["train"]["input"] for trial in trial_dirs
    }
    profile_bindings = {
        trial.resolve(): profiles["train"]["manifest"] for trial in trial_dirs
    }
    selection_arguments: dict[str, object] = {
        "selection_record_path": selection,
        "selection_status_path": selection_status,
        "selection_tune_input_path": profiles["tune"]["input"],
        "selection_tune_profile_manifest_path": profiles["tune"]["manifest"],
        "selection_config_path": project_root / "configs/task_c_tuning_v1.json",
        "selection_trial_directories": tuple(trial_dirs),
        "selection_trial_input_bindings": input_bindings,
        "selection_trial_profile_bindings": profile_bindings,
    }
    if method_id == "hypersca_c":
        assert gene_list is not None
        selection_arguments["selection_trial_hypersca_configs"] = {
            trial.resolve(): configs[index]
            for index, trial in enumerate(trial_dirs)
        }
        selection_arguments["selection_trial_gene_lists"] = {
            trial.resolve(): gene_list for trial in trial_dirs
        }
    status = _run_method_bundle(
        method_id=method_id,
        profile_record=profiles["refit"],
        output_dir=refit,
        seed=seed,
        registry_path=registry_path,
        asset_root=asset_root,
        public_manifest=public_manifest,
        context_id=condition.replace("within_", "") if condition.startswith("within_") else condition,
        min_cells=min_cells,
        timeout_seconds=timeout_seconds,
        project_root=project_root,
        hypersca_config=refit_config,
        gene_list=gene_list,
        selection_arguments=selection_arguments,
        resource_record_path=work_dir / "refit.resource.json",
        resource_stage="refit",
    )
    return refit, status


def _run_formal_final_method(
    *,
    method_id: str,
    condition: str,
    profile: str,
    profiles: Mapping[str, Mapping[str, Path]],
    work_dir: Path,
    seed: int,
    registry_path: Path,
    asset_root: Path,
    public_manifest: Path,
    min_cells: int,
    timeout_seconds: int,
    project_root: Path,
    source_kind: str,
) -> tuple[Path, dict[str, object]]:
    if profile == "connection" and method_id == "hypersca_c":
        return _select_connection_configuration(
            method_id=method_id,
            condition=condition,
            profiles=profiles,
            work_dir=work_dir,
            seed=seed,
            registry_path=registry_path,
            asset_root=asset_root,
            public_manifest=public_manifest,
            min_cells=min_cells,
            timeout_seconds=timeout_seconds,
            project_root=project_root,
            base_hypersca_config=project_root / "configs/hypersca_c_v1.json",
        )

    refit = work_dir / "refit"
    if source_kind == "publication_only":
        status = _run_method_bundle(
            method_id=method_id,
            profile_record=None,
            output_dir=refit,
            seed=seed,
            registry_path=registry_path,
            asset_root=asset_root,
            public_manifest=public_manifest,
            context_id=condition,
            min_cells=min_cells,
            timeout_seconds=timeout_seconds,
            project_root=project_root,
        )
        return refit, status

    gene_list: Path | None = None
    config: Path | None = None
    if method_id == "hypersca_c":
        gene_list = _hypersca_gene_list(
            profile_record=profiles["refit"],
            destination=work_dir / "genes.json",
            profile=profile,
        )
        config = project_root / "configs/hypersca_c_v1.json"
    status = _run_method_bundle(
        method_id=method_id,
        profile_record=profiles["refit"],
        output_dir=refit,
        seed=seed,
        registry_path=registry_path,
        asset_root=asset_root,
        public_manifest=public_manifest,
        context_id=condition.replace("within_", "") if condition.startswith("within_") else condition,
        min_cells=min_cells,
        timeout_seconds=timeout_seconds,
        project_root=project_root,
        hypersca_config=config,
        gene_list=gene_list,
        resource_record_path=work_dir / "refit.resource.json",
        resource_stage="refit",
    )
    return refit, status


def _verify_formal_final_method_bundle(
    *,
    method_id: str,
    condition: str,
    profile: str,
    profiles: Mapping[str, Mapping[str, Path]],
    work_dir: Path,
    seed: int,
    registry_path: Path,
    asset_root: Path,
    public_manifest: Path,
    min_cells: int,
    timeout_seconds: int,
    project_root: Path,
    source_kind: str,
) -> None:
    """Replay the exact method boundary immediately before outer publication."""

    profile_record: Mapping[str, Path] | None = profiles["refit"]
    if source_kind == "publication_only":
        profile_record = None
    gene_list: Path | None = None
    hypersca_config: Path | None = None
    selection_arguments: dict[str, object] | None = None
    if method_id == "hypersca_c":
        gene_list = work_dir / "genes.json"
        hypersca_config = (
            work_dir / "selected_refit_config.json"
            if profile == "connection"
            else project_root / "configs/hypersca_c_v1.json"
        )
        if profile == "connection":
            trial_dirs = tuple(
                work_dir / "trials" / f"trial_{index}" for index in (0, 1)
            )
            configs = tuple(
                work_dir / f"hypersca_config_trial_{index}.json"
                for index in (0, 1)
            )
            selection = work_dir / "selection_record.json"
            selection_arguments = {
                "selection_record_path": selection,
                "selection_status_path": Path(f"{selection}.status.json"),
                "selection_tune_input_path": profiles["tune"]["input"],
                "selection_tune_profile_manifest_path": profiles["tune"]["manifest"],
                "selection_config_path": project_root / "configs/task_c_tuning_v1.json",
                "selection_trial_directories": trial_dirs,
                "selection_trial_input_bindings": {
                    trial.resolve(): profiles["train"]["input"] for trial in trial_dirs
                },
                "selection_trial_profile_bindings": {
                    trial.resolve(): profiles["train"]["manifest"] for trial in trial_dirs
                },
                "selection_trial_hypersca_configs": {
                    trial.resolve(): configs[index]
                    for index, trial in enumerate(trial_dirs)
                },
                "selection_trial_gene_lists": {
                    trial.resolve(): gene_list for trial in trial_dirs
                },
            }
    result = _run_method_bundle(
        method_id=method_id,
        profile_record=profile_record,
        output_dir=work_dir / "refit",
        seed=seed,
        registry_path=registry_path,
        asset_root=asset_root,
        public_manifest=public_manifest,
        context_id=(
            condition.replace("within_", "")
            if condition.startswith("within_")
            else condition
        ),
        min_cells=min_cells,
        timeout_seconds=timeout_seconds,
        project_root=project_root,
        hypersca_config=hypersca_config,
        gene_list=gene_list,
        selection_arguments=selection_arguments,
        resource_record_path=None,
        resource_stage=None,
    )
    if result.get("status") != "completed_standardized_output":
        raise TaskCRehearsalError(
            "pre-publication method bundle verification did not retain completion"
        )


def _causalbench_python(asset_root: Path, environment_name: str) -> Path:
    try:
        completed = subprocess.run(
            ("conda", "env", "list", "--json"),
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise TaskCRehearsalError(
            "the fixed CausalBench environment cannot be located"
        ) from exc
    environments = payload.get("envs") if isinstance(payload, dict) else None
    if not isinstance(environments, list):
        raise TaskCRehearsalError(
            "the fixed CausalBench environment cannot be located"
        )
    candidates = [
        Path(value) / "bin/python"
        for value in environments
        if isinstance(value, str) and Path(value).name == environment_name
    ]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise TaskCRehearsalError(
            "the fixed CausalBench environment cannot be located"
        )
    del asset_root
    return candidates[0].resolve(strict=True)


def _reference_edges(path: Path, expected_sha256: str) -> set[tuple[str, str]]:
    import csv

    payload, _identity = _capture_regular_bytes(
        path,
        label="reference-relation table",
        maximum_bytes=MAXIMUM_REHEARSAL_REFERENCE_BYTES,
    )
    observed = _bytes_sha256(payload).removeprefix("sha256:")
    if expected_sha256.removeprefix("sha256:") != observed:
        raise TaskCRehearsalError("reference-relation file fingerprint changed")
    try:
        with io.StringIO(payload.decode("utf-8", errors="strict"), newline="") as handle:
            rows = csv.reader(handle)
            if next(rows, None) != ["source", "target"]:
                raise TaskCRehearsalError(
                    "reference-relation table must use source,target columns"
                )
            edges = {(source, target) for source, target in rows}
    except TaskCRehearsalError:
        raise
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise TaskCRehearsalError(
            "reference-relation table could not be read"
        ) from exc
    if not edges:
        raise TaskCRehearsalError("reference-relation table is empty")
    return edges


def _formal_scoring(
    *,
    condition: str,
    predictions: FrozenRehearsalPredictions,
    prepared_root: Path,
    asset_root: Path,
    work_dir: Path,
    seed: int,
    registry: Any,
    project_root: Path,
    public_profile_input: Path,
    maximum_cells: int,
    rehearsal_closure: FrozenRehearsalClosure | None = None,
) -> dict[str, object]:
    if not isinstance(predictions, FrozenRehearsalPredictions):
        raise TaskCRehearsalError(
            "formal scoring requires one frozen prediction snapshot"
        )
    _verify_private_rehearsal_closure(
        rehearsal_closure,
        prepared_root=prepared_root,
    )
    _verify_frozen_predictions(predictions)
    prediction_path = predictions.path
    scope = _condition_scope(condition)
    if scope["direction"] is None:
        source_heldout = prepared_root / "private" / "within" / str(scope["context_id"]) / "holdout.npz"
    else:
        source_heldout = prepared_root / "private" / "cross" / condition / "target_holdout.npz"
    private_root = Path(tempfile.mkdtemp(prefix="private-task-c-scoring-"))
    try:
        heldout = materialize_sealed_scoring_subset(
            source_path=source_heldout,
            public_profile_input=public_profile_input,
            destination=private_root / "heldout-profile.npz",
            maximum_cells=maximum_cells,
            seed=seed,
        )
        metrics = _formal_scoring_subset(
            condition=condition,
            predictions=prediction_path,
            prepared_root=prepared_root,
            asset_root=asset_root,
            work_dir=work_dir,
            seed=seed,
            registry=registry,
            project_root=project_root,
            heldout=heldout,
            private_root=private_root,
        )
        metrics["sealed_scoring_input"] = {
            "source_sha256": _sha256_file(source_heldout),
            "profile_subset_sha256": _sha256_file(heldout),
            "public_profile_input_sha256": _sha256_file(public_profile_input),
        }
        _verify_frozen_predictions(predictions)
        metrics["prediction_sha256"] = predictions.sha256
        return metrics
    finally:
        try:
            _verify_private_rehearsal_closure(
                rehearsal_closure,
                prepared_root=prepared_root,
            )
        finally:
            if private_root.exists() and not private_root.is_symlink():
                shutil.rmtree(private_root)


def _formal_scoring_subset_unrecorded(
    *,
    condition: str,
    predictions: Path,
    prepared_root: Path,
    asset_root: Path,
    work_dir: Path,
    seed: int,
    registry: Any,
    project_root: Path,
    heldout: Path,
    private_root: Path,
) -> dict[str, object]:
    import pandas as pd

    from src.evaluation.task_c_aggregation import (
        TaskCAggregationError,
        evaluate_declared_references,
        task_c_aggregation_to_jsonable,
    )

    scope = _condition_scope(condition)
    prediction_payload, prediction_identity = _capture_regular_bytes(
        predictions,
        label="fixed scoring prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    heldout_payload, heldout_identity = _capture_regular_bytes(
        heldout,
        label="sealed scoring NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    evaluation_worker = project_root / "scripts/task_c_workers/causalbench_evaluation_worker.py"
    boundary_worker = project_root / "scripts/task_c_workers/causalbench_worker.py"
    python = _causalbench_python(
        asset_root, str(registry.causalbench["environment"])
    )
    official_output = work_dir / "sealed_scoring.json"
    command = (
        str(python),
        "-I",
        str(evaluation_worker),
        "--prediction-csv",
        str(predictions),
        "--heldout-npz",
        str(heldout),
        "--output-json",
        str(official_output),
        "--seed",
        str(seed),
        "--causalbench-source",
        str(asset_root / "sources/causalbench"),
    )
    snapshots = (
        freeze_method_worker_entry(evaluation_worker),
        freeze_method_worker_entry(boundary_worker),
    )
    completed = run_validated_private_scoring_command(
        command,
        private_root=private_root,
        execution_cwd=project_root,
        allowed_python_interpreters=(python,),
        allowed_worker_snapshots=snapshots,
        allowed_private_inputs=(heldout,),
        timeout_seconds=1_800,
    )
    official = _strict_json_file(official_output, "sealed scoring result")
    refreshed_predictions, refreshed_prediction_identity = _capture_regular_bytes(
        predictions,
        label="fixed scoring prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    refreshed_heldout, refreshed_heldout_identity = _capture_regular_bytes(
        heldout,
        label="sealed scoring NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    if (
        refreshed_predictions != prediction_payload
        or refreshed_prediction_identity != prediction_identity
        or refreshed_heldout != heldout_payload
        or refreshed_heldout_identity != heldout_identity
    ):
        raise TaskCRehearsalError(
            "sealed scoring inputs changed while the approved worker was running"
        )
    if completed.returncode != 0 or official.get("status") != (
        "supplementary_official_metrics"
    ):
        raise TaskCRehearsalError(
            "sealed scoring did not complete: "
            + _safe_failure_reason(official.get("error", completed.stderr))
        )
    if "metrics" not in official:
        raise TaskCRehearsalError(
            "sealed scoring result lacks the official supplementary metrics"
        )

    provenance_root = prepared_root.parents[1] / "provenance"
    reference = _strict_json_file(
        provenance_root / f"{scope['target_context']}_references.json",
        "reference-relation provenance",
    )
    files = reference.get("files")
    if not isinstance(files, dict):
        raise TaskCRehearsalError("reference-relation provenance lacks files")
    pooled_record = files.get("pooled")
    chip_record = files.get("chipseq")
    if not isinstance(pooled_record, dict) or not isinstance(chip_record, dict):
        raise TaskCRehearsalError("reference-relation provenance lacks file records")
    pooled = _reference_edges(
        Path(str(pooled_record.get("path"))), str(pooled_record.get("sha256"))
    )
    chip = _reference_edges(
        Path(str(chip_record.get("path"))), str(chip_record.get("sha256"))
    )
    try:
        with np.load(io.BytesIO(heldout_payload), allow_pickle=False) as archive:
            labels = np.asarray(archive["interventions"], dtype=str)
        eligible_sources = set(labels.tolist()) - {CONTROL_LABEL, "excluded"}
        scores = pd.read_csv(io.BytesIO(prediction_payload))
        biological = evaluate_declared_references(
            scores,
            pooled_reference=pooled,
            directed_chip_reference=chip,
            eligible_sources=eligible_sources,
            directed_reference_context_match=(scope["target_context"] == "k562"),
            precision_values=(1_000, 5_000),
        )
    except (OSError, ValueError, KeyError, TaskCAggregationError) as exc:
        raise TaskCRehearsalError(
            f"sealed biological-reference scoring failed: {exc}"
        ) from exc
    metrics = task_c_aggregation_to_jsonable(biological)
    final_predictions, final_prediction_identity = _capture_regular_bytes(
        predictions,
        label="fixed scoring prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    final_heldout, final_heldout_identity = _capture_regular_bytes(
        heldout,
        label="sealed scoring NPZ",
        maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
    )
    if (
        final_predictions != prediction_payload
        or final_prediction_identity != prediction_identity
        or final_heldout != heldout_payload
        or final_heldout_identity != heldout_identity
    ):
        raise TaskCRehearsalError(
            "sealed scoring inputs changed during biological-reference scoring"
        )
    assert isinstance(metrics, dict)
    metrics["supplementary_official_metrics"] = official["metrics"]
    metrics["sealed_scoring_status"] = official["status"]
    return metrics


def _formal_scoring_subset(
    *,
    condition: str,
    predictions: Path,
    prepared_root: Path,
    asset_root: Path,
    work_dir: Path,
    seed: int,
    registry: Any,
    project_root: Path,
    heldout: Path,
    private_root: Path,
) -> dict[str, object]:
    """Run sealed scoring and retain its elapsed-time and failure evidence."""

    started = time.monotonic()
    status = "failed"
    try:
        metrics = _formal_scoring_subset_unrecorded(
            condition=condition,
            predictions=predictions,
            prepared_root=prepared_root,
            asset_root=asset_root,
            work_dir=work_dir,
            seed=seed,
            registry=registry,
            project_root=project_root,
            heldout=heldout,
            private_root=private_root,
        )
        status = "completed"
        return metrics
    finally:
        elapsed = max(0.0, time.monotonic() - started)
        _write_new_record(
            work_dir / "sealed_scoring.resource.json",
            {
                "schema_version": "1.0",
                "component_kind": "sealed_scoring",
                "stage": "scoring",
                "status": status,
                **_measured_stage_fields(
                    output_root=work_dir / "sealed_scoring.json",
                    elapsed_seconds=elapsed,
                ),
            },
        )


def _promotion_record() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "workflow_validation_only",
        "claim_level": "workflow_validation_only",
        "promotion_eligible": False,
        "reason": (
            "Single-seed reduced-data rehearsal validates execution and resource readiness only."
        ),
    }


def _outer_environment_record(
    *,
    method_id: str,
    condition: str,
    profile: str,
    synthetic_smoke: bool,
    inner_dir: Path | None,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "method_id": method_id,
        "condition": condition,
        "profile": profile,
        "data_scope": "synthetic_smoke" if synthetic_smoke else "external_benchmark",
        "python": {
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "executable_sha256": _sha256_file(Path(sys.executable).resolve()),
        },
        "inner_method_evidence": (
            str(inner_dir.relative_to(inner_dir.parents[3]))
            if inner_dir is not None and len(inner_dir.parents) > 3
            else None
        ),
        "private_data_received_by_method": False,
    }


def _resource_record(
    work_dir: Path | None, *, used_stages: Sequence[str]
) -> dict[str, object]:
    components: list[dict[str, object]] = []
    if work_dir is not None and work_dir.is_dir():
        for path in sorted(work_dir.rglob("*.resource.json")):
            record = _strict_json_file(path, "rehearsal component resource record")
            components.append(
                {
                    "relative_path": path.relative_to(work_dir).as_posix(),
                    **record,
                }
            )
        for path in sorted(work_dir.rglob("resource_usage.json")):
            components.append(
                {
                    "relative_path": path.relative_to(work_dir).as_posix(),
                    "schema_version": "1.0",
                    "component_kind": "method_runtime_worker",
                    "record": _strict_json_file(path, "method resource record"),
                }
            )
    null_records = [
        record for record in components if record.get("component_kind") == "null_analysis"
    ]
    elapsed = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in components
        if isinstance(record.get("elapsed_seconds", 0.0), (int, float))
        and not isinstance(record.get("elapsed_seconds", 0.0), bool)
    )
    observed_stages = {
        str(record["stage"])
        for record in components
        if isinstance(record.get("stage"), str)
    }
    stage_order = ("train", "tune", "refit", "scoring", "null_control")
    recorded_stages = tuple(
        stage
        for stage in stage_order
        if stage in set(used_stages) | observed_stages
    )
    recorded_stages += tuple(
        stage
        for stage in used_stages
        if stage not in recorded_stages
    )
    phase_resources: dict[str, dict[str, object]] = {}
    for stage in recorded_stages:
        stage_records = [record for record in components if record.get("stage") == stage]
        elapsed_values = [
            float(record["elapsed_seconds"])
            for record in stage_records
            if isinstance(record.get("elapsed_seconds"), (int, float))
            and not isinstance(record.get("elapsed_seconds"), bool)
        ]
        rss_values = [
            int(record["peak_rss_bytes"])
            for record in stage_records
            if isinstance(record.get("peak_rss_bytes"), int)
            and not isinstance(record.get("peak_rss_bytes"), bool)
        ]
        gpu_values = [
            int(record["peak_gpu_memory_bytes"])
            for record in stage_records
            if isinstance(record.get("peak_gpu_memory_bytes"), int)
            and not isinstance(record.get("peak_gpu_memory_bytes"), bool)
        ]
        disk_values = [
            int(record["written_disk_bytes"])
            for record in stage_records
            if isinstance(record.get("written_disk_bytes"), int)
            and not isinstance(record.get("written_disk_bytes"), bool)
        ]
        phase_resources[stage] = {
            "elapsed_seconds": sum(elapsed_values) if elapsed_values else None,
            "peak_rss_bytes": max(rss_values) if rss_values else None,
            "peak_gpu_memory_bytes": max(gpu_values) if gpu_values else None,
            "written_disk_bytes": sum(disk_values) if disk_values else None,
            "measurement_availability": {
                "elapsed_seconds": bool(elapsed_values),
                "peak_rss_bytes": bool(rss_values),
                "peak_gpu_memory_bytes": bool(gpu_values),
                "written_disk_bytes": bool(disk_values),
            },
        }
    total_elapsed = [
        float(record["elapsed_seconds"])
        for record in phase_resources.values()
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    total_rss = [
        int(record["peak_rss_bytes"])
        for record in phase_resources.values()
        if isinstance(record.get("peak_rss_bytes"), int)
    ]
    total_gpu = [
        int(record["peak_gpu_memory_bytes"])
        for record in phase_resources.values()
        if isinstance(record.get("peak_gpu_memory_bytes"), int)
    ]
    total_disk = [
        int(record["written_disk_bytes"])
        for record in phase_resources.values()
        if isinstance(record.get("written_disk_bytes"), int)
    ]
    return {
        "schema_version": "1.0",
        "resource_scope": "single-seed reduced-data rehearsal",
        "used_stages": list(recorded_stages),
        "component_records": components,
        "recorded_elapsed_seconds": elapsed,
        "phase_resources": phase_resources,
        "nonduplicated_totals": {
            "elapsed_seconds": sum(total_elapsed) if total_elapsed else None,
            "peak_rss_bytes": max(total_rss) if total_rss else None,
            "peak_gpu_memory_bytes": max(total_gpu) if total_gpu else None,
            "written_disk_bytes": sum(total_disk) if total_disk else None,
            "measurement_availability": {
                "elapsed_seconds": bool(total_elapsed),
                "peak_rss_bytes": bool(total_rss),
                "peak_gpu_memory_bytes": bool(total_gpu),
                "written_disk_bytes": bool(total_disk),
            },
            "aggregation_rule": (
                "sum non-overlapping phase envelopes; take peak memory across phases"
            ),
        },
        "null_control_analysis_count": len(null_records),
        "null_control_repeat_count_per_type": (
            len(null_records) // 2 if len(null_records) == 40 else 0
        ),
    }


def _publish_outer_success(
    *,
    destination: Path,
    method_id: str,
    condition: str,
    profile: str,
    seed: int,
    predictions: Any | None,
    metrics: Mapping[str, object],
    input_summary: Mapping[str, object],
    inner_dir: Path | None,
    work_dir: Path | None,
    synthetic_smoke: bool,
    required_artifacts: Sequence[str],
    prediction_snapshot: FrozenRehearsalPredictions | None = None,
) -> None:
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists() or destination.exists():
        raise TaskCRehearsalError("run result directory already exists")
    staging.mkdir(parents=True, mode=0o700)
    try:
        prediction_path = staging / "predictions.csv"
        if prediction_snapshot is None:
            if predictions is None:
                raise TaskCRehearsalError("completed run lacks fixed predictions")
            prediction_payload = predictions.to_csv(index=False).encode("utf-8")
        else:
            _verify_frozen_predictions(prediction_snapshot)
            prediction_payload = prediction_snapshot.payload
        _write_new_bytes(prediction_path, prediction_payload)
        prediction_sha256 = _bytes_sha256(prediction_payload)
        metrics_record = dict(metrics)
        recorded_prediction_sha256 = metrics_record.get("prediction_sha256")
        if recorded_prediction_sha256 not in {None, prediction_sha256}:
            raise TaskCRehearsalError(
                "metrics prediction fingerprint differs from the fixed prediction bytes"
            )
        metrics_record["prediction_sha256"] = prediction_sha256
        input_record = dict(input_summary)
        promotion_record = _promotion_record()
        environment_record = _outer_environment_record(
            method_id=method_id,
            condition=condition,
            profile=profile,
            synthetic_smoke=synthetic_smoke,
            inner_dir=inner_dir,
        )
        resource_record = _resource_record(
            work_dir,
            used_stages=(
                ("synthetic_smoke",)
                if synthetic_smoke
                else tuple(input_summary.get("used_stages", ("refit",)))
            ),
        )
        evidence_sha256 = {
            "input_summary.json": _record_sha256(input_record),
            "metrics.json": _record_sha256(metrics_record),
            "predictions.csv": prediction_sha256,
            "promotion_decision.json": _record_sha256(promotion_record),
            "environment_manifest.json": _record_sha256(environment_record),
            "resource_usage.json": _record_sha256(resource_record),
        }
        run_identity = {
            "schema_version": "1.0",
            "profile": profile,
            "condition": condition,
            "method_id": method_id,
            "seed": seed,
            "input_summary_sha256": _canonical_sha256(input_record),
            "prediction_sha256": prediction_sha256,
            "evidence_sha256": evidence_sha256,
        }
        identity_sha256 = _canonical_sha256(run_identity)
        _write_new_record(
            staging / "run_manifest.json",
            {
                **run_identity,
                "run_identity_sha256": identity_sha256,
                "claim_level": "workflow_validation_only",
            },
        )
        _write_new_record(staging / "input_summary.json", input_record)
        _write_new_record(staging / "metrics.json", metrics_record)
        _write_new_record(staging / "promotion_decision.json", promotion_record)
        _write_new_record(staging / "environment_manifest.json", environment_record)
        _write_new_record(staging / "resource_usage.json", resource_record)
        _write_new_record(
            staging / "method_status.json",
            {
                "schema_version": "1.0",
                "method_id": method_id,
                "condition": condition,
                "seed": seed,
                "run_identity_sha256": identity_sha256,
                "status": (
                    "passed_synthetic_smoke"
                    if synthetic_smoke
                    else "passed_real_rehearsal"
                ),
                "controller_validation": (
                    "verified_task_c_synthetic_smoke_bundle_v1"
                    if synthetic_smoke
                    else "verified_task_c_rehearsal_bundle_v1"
                ),
            },
        )
        validate_required_run_artifacts(staging, required_artifacts)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish_outer_failure(
    *,
    destination: Path,
    method_id: str,
    condition: str,
    profile: str,
    seed: int,
    status: str,
    reason: str,
    inner_dir: Path | None,
    work_dir: Path | None,
    synthetic_smoke: bool,
) -> None:
    if status not in _FINAL_REHEARSAL_STATUSES - {
        "passed_real_rehearsal",
        "passed_synthetic_smoke",
    }:
        raise TaskCRehearsalError("failure status is not recognized")
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists() or destination.exists():
        raise TaskCRehearsalError("run result directory already exists")
    staging.mkdir(parents=True, mode=0o700)
    try:
        identity_sha256 = _canonical_sha256(
            {
                "profile": profile,
                "condition": condition,
                "method_id": method_id,
                "seed": seed,
                "failure_status": status,
            }
        )
        _write_new_record(
            staging / "environment_manifest.json",
            _outer_environment_record(
                method_id=method_id,
                condition=condition,
                profile=profile,
                synthetic_smoke=synthetic_smoke,
                inner_dir=inner_dir,
            ),
        )
        _write_new_record(
            staging / "resource_usage.json",
            _resource_record(
                work_dir,
                used_stages=("synthetic_smoke",) if synthetic_smoke else (),
            ),
        )
        _write_new_record(
            staging / "method_status.json",
            {
                "schema_version": "1.0",
                "method_id": method_id,
                "condition": condition,
                "seed": seed,
                "run_identity_sha256": identity_sha256,
                "status": status,
                "controller_validation": "verified_task_c_rehearsal_bundle_v1",
                "reason": _safe_failure_reason(reason),
            },
        )
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _is_sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _validated_private_control_indices(value: object) -> dict[str, dict[str, tuple[int, ...]]]:
    if not isinstance(value, dict) or set(value) != {"k562", "rpe1"}:
        raise TaskCRehearsalError(
            "sealed split control indices must contain exactly k562 and rpe1"
        )
    validated: dict[str, dict[str, tuple[int, ...]]] = {}
    for context in ("k562", "rpe1"):
        partitions = value[context]
        if not isinstance(partitions, dict) or set(partitions) != {
            "train",
            "tune",
            "holdout",
        }:
            raise TaskCRehearsalError(
                "sealed split control partitions must contain train, tune, and holdout"
            )
        checked: dict[str, tuple[int, ...]] = {}
        observed: set[int] = set()
        for partition in ("train", "tune", "holdout"):
            raw = partitions[partition]
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
                raise TaskCRehearsalError(
                    "sealed split control partitions must be non-empty index lists"
                )
            indices = tuple(raw)
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            ) or len(set(indices)) != len(indices):
                raise TaskCRehearsalError(
                    "sealed split control indices must be unique non-negative integers"
                )
            if observed.intersection(indices):
                raise TaskCRehearsalError(
                    "sealed split control partitions must not overlap"
                )
            observed.update(indices)
            checked[partition] = indices
        validated[context] = checked
    return validated


def _validate_private_rehearsal_inputs(
    *, prepared_root: Path, public: Mapping[str, object]
) -> dict[str, Path]:
    """Validate the fixed private split and its four sealed scoring files."""

    from src.evaluation.task_c_data import (
        SealedHoldoutSemanticContentHasher,
        TaskCDataError,
        load_task_c_dataset_from_verified_bytes,
    )

    manifest_path = prepared_root / "private/private_manifest.json"
    manifest_bytes, manifest_identity = _capture_regular_bytes(
        manifest_path,
        label="sealed data record",
        maximum_bytes=MAXIMUM_REHEARSAL_JSON_BYTES,
    )
    private = _strict_json_payload(manifest_bytes, "sealed data record")
    if set(private) != _PRIVATE_MANIFEST_FIELDS:
        if set(private) == _PRIVATE_MANIFEST_FIELDS - {
            "sealed_holdout_semantic_content_sha256"
        }:
            raise TaskCRehearsalError(
                "sealed data record predates its content commitment; rematerialize prepared data"
            )
        raise TaskCRehearsalError("sealed data record fields differ from its schema")
    identity = public.get("materialization_identity")
    if not isinstance(identity, dict) or set(identity) != _MATERIALIZATION_IDENTITY_FIELDS:
        raise TaskCRehearsalError("public materialization identity is incomplete")
    if any(identity.get(field) != public.get(field) for field in _MATERIALIZATION_IDENTITY_FIELDS):
        raise TaskCRehearsalError("public materialization identity changed")
    if private.get("materialization_identity") != identity:
        raise TaskCRehearsalError(
            "sealed and public data must share one materialization identity"
        )
    shared_fields = (
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "train_sources",
        "tune_sources",
        "content_sha256",
        "gene_names_sha256",
        "gene_projection",
        "sealed_holdout_semantic_content_sha256",
    )
    if any(private.get(field) != public.get(field) for field in shared_fields):
        raise TaskCRehearsalError(
            "sealed split semantics disagree with the public split record"
        )
    if (
        private.get("schema_version") != "1.0"
        or private.get("seed") != 11
        or private.get("split_id") != "C-context-intervention-holdout-v1-seed-11"
        or isinstance(private.get("min_cells_per_intervention"), bool)
        or not isinstance(private.get("min_cells_per_intervention"), int)
        or int(private["min_cells_per_intervention"]) <= 0
    ):
        raise TaskCRehearsalError("sealed split identity or schema changed")
    train_sources = _canonical_texts(
        private.get("train_sources"),
        "sealed train sources",
        require_unique=True,
        maximum_items=MAXIMUM_PARENT_GENES,
    )
    tune_sources = _canonical_texts(
        private.get("tune_sources"),
        "sealed tune sources",
        require_unique=True,
        maximum_items=MAXIMUM_PARENT_GENES,
    )
    holdout_sources = _canonical_texts(
        private.get("holdout_sources"),
        "sealed holdout sources",
        require_unique=True,
        maximum_items=MAXIMUM_PARENT_GENES,
    )
    if (
        CONTROL_LABEL in {*train_sources, *tune_sources, *holdout_sources}
        or set(train_sources) & set(tune_sources)
        or set(train_sources) & set(holdout_sources)
        or set(tune_sources) & set(holdout_sources)
        or public.get("holdout_source_count") != len(holdout_sources)
    ):
        raise TaskCRehearsalError("sealed source partitions are incomplete or overlap")
    controls = _validated_private_control_indices(private.get("control_indices"))
    for field in ("input_sha256", "content_sha256"):
        hashes = identity.get(field)
        if not isinstance(hashes, dict) or set(hashes) != {"k562", "rpe1"} or any(
            not _is_sha256_text(value) for value in hashes.values()
        ):
            raise TaskCRehearsalError(
                f"sealed materialization {field} fingerprints are malformed"
            )
    if not _is_sha256_text(identity.get("gene_names_sha256")):
        raise TaskCRehearsalError(
            "sealed materialization gene fingerprint is malformed"
        )
    if not _is_sha256_text(
        identity.get("sealed_holdout_semantic_content_sha256")
    ):
        raise TaskCRehearsalError(
            "sealed holdout semantic content commitment is malformed"
        )
    projection = identity.get("gene_projection")
    if not isinstance(projection, dict) or set(projection) != {
        "projection_rule",
        "common",
        "contexts",
    }:
        raise TaskCRehearsalError("sealed materialization gene projection is malformed")
    common = projection.get("common")
    contexts = projection.get("contexts")
    if (
        projection.get("projection_rule") != "sorted_common_gene_intersection_v1"
        or not isinstance(common, dict)
        or set(common) != {"count", "ordered_genes", "sha256"}
        or not isinstance(contexts, dict)
        or set(contexts) != {"k562", "rpe1"}
    ):
        raise TaskCRehearsalError("sealed materialization gene projection changed")
    genes = _canonical_texts(
        common.get("ordered_genes"),
        "sealed common genes",
        require_unique=True,
        maximum_items=MAXIMUM_PARENT_GENES,
    )
    expected_gene_hash = _bytes_sha256(
        json.dumps(genes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if (
        genes != tuple(sorted(genes))
        or common.get("count") != len(genes)
        or common.get("sha256") != expected_gene_hash
        or identity.get("gene_names_sha256") != expected_gene_hash
    ):
        raise TaskCRehearsalError("sealed common-gene identity changed")
    for context in ("k562", "rpe1"):
        record = contexts[context]
        if not isinstance(record, dict) or set(record) != {
            "original_gene_count",
            "original_gene_names_sha256",
            "selected_original_indices",
            "mapping_sha256",
        }:
            raise TaskCRehearsalError(
                "sealed gene-projection context schema changed"
            )
        original_count = record.get("original_gene_count")
        indices = record.get("selected_original_indices")
        if (
            isinstance(original_count, bool)
            or not isinstance(original_count, int)
            or not len(genes) <= original_count <= MAXIMUM_PARENT_GENES
            or not _is_sha256_text(record.get("original_gene_names_sha256"))
            or isinstance(indices, (str, bytes))
            or not isinstance(indices, Sequence)
            or len(indices) != len(genes)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < original_count
                for index in indices
            )
            or len(set(indices)) != len(indices)
        ):
            raise TaskCRehearsalError(
                "sealed gene-projection context semantics changed"
            )
        mapping = {
            "common_ordered_genes": list(genes),
            "selected_original_indices": list(indices),
        }
        if record.get("mapping_sha256") != _canonical_sha256(mapping):
            raise TaskCRehearsalError(
                "sealed gene-projection context fingerprint changed"
            )

    files = private.get("files")
    if not isinstance(files, dict) or set(files) != _PRIVATE_HOLDOUT_PATHS:
        raise TaskCRehearsalError(
            "sealed data record must inventory exactly the four fixed holdouts"
        )
    snapshots: list[tuple[Path, str, tuple[int, int, int, int, int]]] = []
    paths: dict[str, Path] = {}
    all_sources = train_sources + tune_sources + holdout_sources
    semantic_hasher = SealedHoldoutSemanticContentHasher()
    for relative in sorted(_PRIVATE_HOLDOUT_PATHS):
        expected_hash = files[relative]
        if not _is_sha256_text(expected_hash):
            raise TaskCRehearsalError(
                f"sealed holdout fingerprint is malformed: {relative}"
            )
        path = prepared_root / relative
        payload, file_identity = _capture_regular_bytes(
            path,
            label=f"sealed holdout {relative}",
            maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
        )
        actual_hash = _bytes_sha256(payload)
        if actual_hash != expected_hash:
            raise TaskCRehearsalError(
                f"sealed holdout fingerprint changed: {relative}"
            )
        context = (
            relative.split("/")[2]
            if relative.startswith("private/within/")
            else relative.split("/")[2].split("_to_")[1]
        )
        expected_sources = (
            holdout_sources
            if relative.startswith("private/within/")
            else all_sources
        )
        try:
            dataset = load_task_c_dataset_from_verified_bytes(
                path,
                context_id=context,
                source_bytes=payload,
                source_sha256=actual_hash,
                sealed_holdout_hasher=semantic_hasher,
                logical_artifact=relative,
            )
        except TaskCDataError as exc:
            raise TaskCRehearsalError(
                f"sealed holdout arrays are invalid: {relative}"
            ) from exc
        labels = tuple(str(value) for value in dataset.interventions.tolist())
        counts = Counter(labels)
        if (
            dataset.gene_names != genes
            or set(counts) != {CONTROL_LABEL, *expected_sources}
            or any(
                counts[source] < int(private["min_cells_per_intervention"])
                for source in expected_sources
            )
            or counts[CONTROL_LABEL] != len(controls[context]["holdout"])
        ):
            raise TaskCRehearsalError(
                f"sealed holdout semantics changed: {relative}"
            )
        paths[f"private_holdout:{relative}"] = path
        snapshots.append((path, actual_hash, file_identity))

    if semantic_hasher.sha256() != identity[
        "sealed_holdout_semantic_content_sha256"
    ]:
        raise TaskCRehearsalError(
            "sealed holdout semantic content commitment changed"
        )

    final_manifest_bytes, final_manifest_identity = _capture_regular_bytes(
        manifest_path,
        label="sealed data record",
        maximum_bytes=MAXIMUM_REHEARSAL_JSON_BYTES,
    )
    if final_manifest_bytes != manifest_bytes or final_manifest_identity != manifest_identity:
        raise TaskCRehearsalError("sealed data record changed during validation")
    for path, expected_hash, expected_identity in snapshots:
        final_payload, final_identity = _capture_regular_bytes(
            path,
            label=f"sealed holdout {path.name}",
            maximum_bytes=MAXIMUM_REHEARSAL_INPUT_BYTES,
        )
        if _bytes_sha256(final_payload) != expected_hash or final_identity != expected_identity:
            raise TaskCRehearsalError("sealed holdout changed during validation")
    return paths


def _validate_prepared_rehearsal_inputs(
    *,
    prepared_root: Path,
    method_assets_root: Path,
    synthetic_smoke: bool,
) -> tuple[Path, dict[str, Any]]:
    public_manifest = prepared_root / "public_manifest.json"
    public = _strict_json_file(public_manifest, "public data record")
    if public.get("seed") != 11:
        raise TaskCRehearsalError("rehearsal public data must use fixed seed 11")
    files = public.get("files")
    if not isinstance(files, dict) or not files:
        raise TaskCRehearsalError("public data record lacks its fixed file inventory")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TaskCRehearsalError("public data file inventory is malformed")
        path = prepared_root / relative
        if _sha256_file(path).removeprefix("sha256:") != expected.removeprefix(
            "sha256:"
        ):
            raise TaskCRehearsalError(
                f"public data file fingerprint changed: {relative}"
            )
    if synthetic_smoke:
        return public_manifest, public

    _validate_private_rehearsal_inputs(prepared_root=prepared_root, public=public)
    if prepared_root.name != "seed_11":
        raise TaskCRehearsalError("formal rehearsal must use the seed_11 split directory")
    split_root = prepared_root.parent
    for split_seed in _FULL_RUN_SEEDS:
        sibling = split_root / f"seed_{split_seed}/public_manifest.json"
        sibling_record = _strict_json_file(
            sibling, f"public data record for seed {split_seed}"
        )
        if sibling_record.get("seed") != split_seed:
            raise TaskCRehearsalError(
                f"public data record for seed {split_seed} has the wrong identity"
            )
        sibling_files = sibling_record.get("files")
        if not isinstance(sibling_files, dict) or not sibling_files:
            raise TaskCRehearsalError(
                f"public data record for seed {split_seed} lacks its file inventory"
            )
        for relative, expected in sibling_files.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise TaskCRehearsalError(
                    f"public data file inventory for seed {split_seed} is malformed"
                )
            if _sha256_file(sibling.parent / relative).removeprefix(
                "sha256:"
            ) != expected.removeprefix("sha256:"):
                raise TaskCRehearsalError(
                    f"public data file fingerprint changed for seed {split_seed}: {relative}"
                )
    provenance = prepared_root.parents[1] / "provenance"
    for context in ("k562", "rpe1"):
        _strict_json_file(provenance / f"{context}.json", "expression provenance")
        reference = _strict_json_file(
            provenance / f"{context}_references.json",
            "reference-relation provenance",
        )
        records = reference.get("files")
        if not isinstance(records, dict):
            raise TaskCRehearsalError(
                "reference-relation provenance lacks file records"
            )
        for kind in ("pooled", "chipseq"):
            record = records.get(kind)
            if not isinstance(record, dict):
                raise TaskCRehearsalError(
                    "reference-relation provenance lacks a required file"
                )
            _reference_edges(
                Path(str(record.get("path"))), str(record.get("sha256"))
            )
    for required in (
        method_assets_root / "bootstrap_identity.json",
        method_assets_root / "bootstrap_manifest.json",
    ):
        _strict_json_file(required, "fixed method-asset record")
    return public_manifest, public


def _controller_identity(
    *,
    profile: str,
    methods: tuple[str, ...],
    synthetic_smoke: bool,
    public_manifest: Path,
    registry_path: Path,
    rehearsal_config_path: Path,
    method_assets_root: Path,
    closure: FrozenRehearsalClosure,
    prepared_identity_sha256: str | None,
) -> dict[str, object]:
    asset_identity = None
    if not synthetic_smoke:
        asset_identity = _sha256_file(method_assets_root / "bootstrap_identity.json")
    return {
        "schema_version": "1.0",
        "profile": profile,
        "methods": list(methods),
        "conditions": list(REHEARSAL_CONDITIONS),
        "seed": 11,
        "synthetic_smoke": synthetic_smoke,
        "public_manifest_sha256": _sha256_file(public_manifest),
        "prepared_identity_sha256": prepared_identity_sha256,
        "method_registry_sha256": _sha256_file(registry_path),
        "rehearsal_config_sha256": _sha256_file(rehearsal_config_path),
        "method_assets_identity_sha256": asset_identity,
        "rehearsal_closure_sha256": closure.sha256,
        "rehearsal_closure_files": {
            label: snapshot.sha256
            for label, snapshot in closure.files.items()
        },
        "claim_level": "workflow_validation_only",
        "promotion_eligible": False,
    }


def _rehearsal_closure_paths(
    *,
    project_root: Path,
    prepared_root: Path,
    method_assets_root: Path,
    synthetic_smoke: bool,
) -> dict[str, Path]:
    paths = {
        "controller_config": project_root / "configs/task_c_rehearsal_v1.json",
        "method_registry": project_root / "configs/task_c_methods_v1.json",
        "hypersca_model_config": project_root / "configs/hypersca_c_v1.json",
        "scoring_config": project_root / "configs/task_c_tuning_v1.json",
        "public_manifest": prepared_root / "public_manifest.json",
        "controller_code": project_root / "src/evaluation/task_c_rehearsal.py",
        "method_boundary_code": project_root / "src/evaluation/task_c_method_run.py",
        "runtime_code": project_root / "src/evaluation/task_c_runtime.py",
        "profile_code": project_root / "src/evaluation/task_c_profile_input.py",
        "prediction_code": project_root / "src/evaluation/task_c_predictions.py",
        "null_code": project_root / "src/evaluation/task_c_null_controls.py",
        "scoring_code": project_root / "src/evaluation/task_c_aggregation.py",
        "hypersca_model_code": project_root / "src/causal/hypersca_c.py",
        "hypersca_stability_code": project_root / "src/causal/hypersca_c_stability.py",
        "hypersca_bundle_code": project_root / "src/causal/hypersca_c_run.py",
        "controller_cli": project_root / "scripts/run_task_c_rehearsal.py",
        "selection_cli": project_root / "scripts/select_task_c_configuration.py",
        "sealed_scoring_worker": (
            project_root / "scripts/task_c_workers/causalbench_evaluation_worker.py"
        ),
        "causalbench_worker": (
            project_root / "scripts/task_c_workers/causalbench_worker.py"
        ),
        "formal_null_worker": (
            project_root / "scripts/task_c_workers/rehearsal_null_worker.py"
        ),
    }
    if synthetic_smoke:
        return paths
    paths.update(
        {
            "private_manifest": prepared_root / "private/private_manifest.json",
            "method_asset_identity": method_assets_root / "bootstrap_identity.json",
            "method_asset_manifest": method_assets_root / "bootstrap_manifest.json",
        }
    )
    paths.update(
        {
            f"private_holdout:{relative}": prepared_root / relative
            for relative in _PRIVATE_HOLDOUT_PATHS
        }
    )
    provenance = prepared_root.parents[1] / "provenance"
    for context in ("k562", "rpe1"):
        paths[f"{context}_provenance"] = provenance / f"{context}.json"
        reference_path = provenance / f"{context}_references.json"
        paths[f"{context}_reference_manifest"] = reference_path
        reference = _strict_json_file(
            reference_path, "reference-relation provenance"
        )
        files = reference.get("files")
        if not isinstance(files, dict):
            raise TaskCRehearsalError(
                "reference-relation provenance lacks file records"
            )
        for kind in ("pooled", "chipseq"):
            record = files.get(kind)
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise TaskCRehearsalError(
                    "reference-relation provenance lacks one reference path"
                )
            paths[f"{context}_{kind}_reference"] = Path(str(record["path"]))
    return paths


def _validate_resumed_success_bundle(
    *,
    run_dir: Path,
    status: Mapping[str, object],
    expected_profile: str,
    expected_condition: str,
    expected_method: str,
    expected_seed: int,
    required_artifacts: Sequence[str],
) -> None:
    """Cross-check one published success from its actual evidence bytes."""

    validate_required_run_artifacts(run_dir, required_artifacts)
    manifest = _strict_json_file(run_dir / "run_manifest.json", "run manifest")
    input_summary = _strict_json_file(
        run_dir / "input_summary.json", "run input summary"
    )
    metrics = _strict_json_file(run_dir / "metrics.json", "run metrics")
    environment = _strict_json_file(
        run_dir / "environment_manifest.json", "run environment record"
    )
    resource = _strict_json_file(
        run_dir / "resource_usage.json", "run resource record"
    )
    promotion = _strict_json_file(
        run_dir / "promotion_decision.json", "run promotion decision"
    )
    prediction_payload, _prediction_identity = _capture_regular_bytes(
        run_dir / "predictions.csv",
        label="run prediction table",
        maximum_bytes=MAXIMUM_REHEARSAL_PREDICTION_BYTES,
    )
    prediction_sha256 = _bytes_sha256(prediction_payload)
    evidence_sha256 = {
        "input_summary.json": _record_sha256(input_summary),
        "metrics.json": _record_sha256(metrics),
        "predictions.csv": prediction_sha256,
        "promotion_decision.json": _record_sha256(promotion),
        "environment_manifest.json": _record_sha256(environment),
        "resource_usage.json": _record_sha256(resource),
    }
    expected_identity = {
        "schema_version": "1.0",
        "profile": expected_profile,
        "condition": expected_condition,
        "method_id": expected_method,
        "seed": expected_seed,
        "input_summary_sha256": _canonical_sha256(input_summary),
        "prediction_sha256": prediction_sha256,
        "evidence_sha256": evidence_sha256,
    }
    if any(
        manifest.get(name) != value for name, value in expected_identity.items()
    ) or manifest.get("run_identity_sha256") != _canonical_sha256(expected_identity):
        raise TaskCRehearsalError(
            "existing run evidence changed and disagrees with its manifest"
        )
    if status.get("run_identity_sha256") != manifest.get("run_identity_sha256"):
        raise TaskCRehearsalError(
            "existing method status disagrees with the run manifest"
        )
    if metrics.get("prediction_sha256") != prediction_sha256:
        raise TaskCRehearsalError(
            "existing metrics disagree with the fixed prediction bytes"
        )
    if (
        input_summary.get("profile") != expected_profile
        or input_summary.get("condition") != expected_condition
        or input_summary.get("method_id") != expected_method
    ):
        raise TaskCRehearsalError(
            "existing input summary disagrees with the requested run"
        )
    if (
        environment.get("profile") != expected_profile
        or environment.get("condition") != expected_condition
        or environment.get("method_id") != expected_method
    ):
        raise TaskCRehearsalError(
            "existing environment record disagrees with the requested run"
        )
    if promotion != _promotion_record():
        raise TaskCRehearsalError("existing promotion decision changed")
    if resource.get("schema_version") != "1.0" or not isinstance(
        resource.get("used_stages"), list
    ):
        raise TaskCRehearsalError("existing resource record is incomplete")
    try:
        import pandas as pd

        predictions = pd.read_csv(
            io.BytesIO(prediction_payload), encoding="utf-8", on_bad_lines="error"
        )
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise TaskCRehearsalError("existing prediction table is malformed") from exc
    if list(predictions.columns) != [
        "source",
        "target",
        "score",
        "returned_by_method",
    ]:
        raise TaskCRehearsalError("existing prediction columns changed")
    sources = predictions["source"].astype(str).tolist()
    targets = predictions["target"].astype(str).tolist()
    genes = set(sources) | set(targets)
    relations = set(zip(sources, targets, strict=True))
    expected_relations = {
        (source, target)
        for source in genes
        for target in genes
        if source != target
    }
    try:
        scores = np.asarray(predictions["score"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaskCRehearsalError("existing prediction scores are invalid") from exc
    if (
        len(predictions) != len(expected_relations)
        or relations != expected_relations
        or not np.isfinite(scores).all()
        or bool(np.any(scores < 0.0))
    ):
        raise TaskCRehearsalError(
            "existing prediction table is not the complete fixed relation universe"
        )


def _rebuild_resumed_summary(
    *, output_root: Path, expected_identity: Mapping[str, object], required_artifacts: Sequence[str]
) -> dict[str, object]:
    methods_raw = expected_identity.get("methods")
    conditions_raw = expected_identity.get("conditions")
    profile = expected_identity.get("profile")
    seed = expected_identity.get("seed")
    if (
        not isinstance(methods_raw, list)
        or not isinstance(conditions_raw, list)
        or not isinstance(profile, str)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise TaskCRehearsalError("existing rehearsal identity is malformed")
    methods = tuple(str(value) for value in methods_raw)
    conditions = tuple(str(value) for value in conditions_raw)
    expected_names = {
        build_rehearsal_run_id(
            profile=profile, condition=condition, method_id=method, seed=seed
        ): (condition, method)
        for condition in conditions
        for method in methods
    }
    runs_root = output_root / "runs"
    try:
        entries = tuple(os.scandir(runs_root))
    except OSError as exc:
        raise TaskCRehearsalError("existing rehearsal runs cannot be inspected") from exc
    actual_names = {entry.name for entry in entries}
    if actual_names != set(expected_names) or any(
        not entry.is_dir(follow_symlinks=False) for entry in entries
    ):
        raise TaskCRehearsalError("existing rehearsal run identities are incomplete")
    statuses: dict[str, str] = {}
    for run_name, (condition, method) in expected_names.items():
        run_dir = runs_root / run_name
        status = _strict_json_file(run_dir / "method_status.json", "method status")
        status_name = status.get("status")
        if (
            status.get("schema_version") != "1.0"
            or status.get("method_id") != method
            or status.get("condition") != condition
            or status.get("seed") != seed
            or status_name not in _FINAL_REHEARSAL_STATUSES
        ):
            raise TaskCRehearsalError(
                "existing method status disagrees with its run identity"
            )
        if status_name in {"passed_real_rehearsal", "passed_synthetic_smoke"}:
            _validate_resumed_success_bundle(
                run_dir=run_dir,
                status=status,
                expected_profile=profile,
                expected_condition=condition,
                expected_method=method,
                expected_seed=seed,
                required_artifacts=required_artifacts,
            )
        else:
            expected_failure_files = frozenset(_REHEARSAL_EXTRA_ARTIFACTS)
            try:
                failure_files = {
                    entry.name
                    for entry in os.scandir(run_dir)
                    if entry.is_file(follow_symlinks=False)
                }
            except OSError as exc:
                raise TaskCRehearsalError(
                    "existing failed run cannot be inspected"
                ) from exc
            if failure_files != expected_failure_files:
                raise TaskCRehearsalError("existing failed run evidence is incomplete")
            environment = _strict_json_file(
                run_dir / "environment_manifest.json", "failed run environment record"
            )
            resource = _strict_json_file(
                run_dir / "resource_usage.json", "failed run resource record"
            )
            if (
                environment.get("profile") != profile
                or environment.get("condition") != condition
                or environment.get("method_id") != method
                or resource.get("schema_version") != "1.0"
            ):
                raise TaskCRehearsalError(
                    "existing failed run evidence disagrees with its identity"
                )
        statuses[f"{condition}/{method}"] = str(status_name)
    return {
        "schema_version": "1.0",
        "profile": profile,
        "attempted_methods": list(methods),
        "conditions": list(conditions),
        "attempted_run_count": len(expected_names),
        "status_counts": dict(sorted(Counter(statuses.values()).items())),
        "claim_level": "workflow_validation_only",
        "promotion_eligible": False,
        "resume_status": "verified_existing_output",
    }


def _resume_token_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"resume_status", "resume_token"}
    }


def _rehearsal_resume_token(
    *,
    controller_identity: Mapping[str, object],
    file_inventory: Mapping[str, str],
    rebuilt_summary: Mapping[str, object],
) -> str:
    """Bind actual evidence for a token the caller must retain independently."""
    return _canonical_sha256(
        {
            "controller_identity": dict(controller_identity),
            "file_inventory": dict(file_inventory),
            "rebuilt_summary": _resume_token_summary(rebuilt_summary),
        }
    )


def _resume_verified_rehearsal(
    *,
    output_root: Path,
    expected_identity: Mapping[str, object],
    required_artifacts: Sequence[str],
    expected_resume_token: str,
) -> dict[str, object]:
    manifest_path = output_root / "controller_manifest.json"
    observed = _strict_json_file(manifest_path, "rehearsal controller record")
    if set(observed) != {
        "schema_version",
        "identity",
        "identity_sha256",
        "file_inventory",
        "summary",
        "resume_token",
    } or observed.get("schema_version") != "1.0":
        raise TaskCRehearsalError("existing rehearsal controller record is incomplete")
    if observed.get("identity") != dict(expected_identity):
        raise TaskCRehearsalError(
            "existing rehearsal identity differs from the requested inputs"
        )
    if observed.get("identity_sha256") != _canonical_sha256(expected_identity):
        raise TaskCRehearsalError("existing rehearsal identity fingerprint changed")
    summary = _rebuild_resumed_summary(
        output_root=output_root,
        expected_identity=expected_identity,
        required_artifacts=required_artifacts,
    )
    expected_inventory = observed.get("file_inventory")
    actual_inventory = _tree_inventory(
        output_root, exclude=frozenset({"controller_manifest.json"})
    )
    if expected_inventory != actual_inventory:
        raise TaskCRehearsalError(
            "existing rehearsal output fingerprint changed"
        )
    recorded_summary = observed.get("summary")
    resume_token = _rehearsal_resume_token(
        controller_identity=expected_identity,
        file_inventory=actual_inventory,
        rebuilt_summary=summary,
    )
    expected_recorded_summary = {
        **summary,
        "resume_status": "new_run",
        "resume_token": resume_token,
    }
    if not isinstance(recorded_summary, dict) or recorded_summary != expected_recorded_summary:
        raise TaskCRehearsalError(
            "existing rehearsal summary disagrees with the actual runs"
        )
    if observed.get("resume_token") != resume_token:
        raise TaskCRehearsalError("existing rehearsal resume token changed")
    if expected_resume_token != resume_token:
        raise TaskCRehearsalError(
            "external resume token differs from the fully revalidated rehearsal"
        )
    return {**summary, "resume_token": resume_token}


def inspect_task_c_rehearsal_evidence(
    output_root: str | Path,
    *,
    expected_resume_token: str,
) -> dict[str, object]:
    """Return a read-only snapshot after rechecking one Task C rehearsal.

    The caller must supply the token retained outside the rehearsal directory.
    The controller's own copy is evidence, never the expected value.  This
    function never authorizes or starts the five-seed comparison.
    """

    if not _is_sha256_text(expected_resume_token):
        raise TaskCRehearsalError(
            "external resume token must be one sha256 fingerprint"
        )
    output = Path(os.path.abspath(os.fspath(Path(output_root).expanduser())))
    manifest = _strict_json_file(
        output / "controller_manifest.json", "rehearsal controller record"
    )
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise TaskCRehearsalError(
            "rehearsal controller record lacks a valid identity"
        )
    summary = _resume_verified_rehearsal(
        output_root=output,
        expected_identity=identity,
        required_artifacts=_REQUIRED_ARTIFACTS,
        expected_resume_token=expected_resume_token,
    )
    methods = _fixed_text_tuple(identity.get("methods"), "rehearsal methods")
    conditions = _fixed_text_tuple(
        identity.get("conditions"), "rehearsal conditions"
    )
    profile = identity.get("profile")
    seed = identity.get("seed")
    if (
        conditions != REHEARSAL_CONDITIONS
        or type(profile) is not str
        or type(seed) is not int
    ):
        raise TaskCRehearsalError("rehearsal controller identity is malformed")
    records: list[dict[str, object]] = []
    remaining_json_bytes = MAXIMUM_REHEARSAL_SUMMARY_JSON_BYTES
    for condition in conditions:
        for method in methods:
            run_dir = output / "runs" / build_rehearsal_run_id(
                profile=profile,
                condition=condition,
                method_id=method,
                seed=seed,
            )
            status, remaining_json_bytes = _strict_json_file_with_total_budget(
                run_dir / "method_status.json",
                label="method status",
                remaining_bytes=remaining_json_bytes,
            )
            resource, remaining_json_bytes = _strict_json_file_with_total_budget(
                run_dir / "resource_usage.json",
                label="run resource record",
                remaining_bytes=remaining_json_bytes,
            )
            environment, remaining_json_bytes = _strict_json_file_with_total_budget(
                run_dir / "environment_manifest.json",
                label="run environment record",
                remaining_bytes=remaining_json_bytes,
            )
            record: dict[str, object] = {
                "method_id": method,
                "condition": condition,
                "seed": seed,
                "status": status.get("status"),
                "resource_usage": resource,
                "environment_manifest": environment,
            }
            if status.get("status") in {
                "passed_real_rehearsal",
                "passed_synthetic_smoke",
            }:
                metrics, remaining_json_bytes = _strict_json_file_with_total_budget(
                    run_dir / "metrics.json",
                    label="run metrics",
                    remaining_bytes=remaining_json_bytes,
                )
                input_summary, remaining_json_bytes = (
                    _strict_json_file_with_total_budget(
                        run_dir / "input_summary.json",
                        label="run input summary",
                        remaining_bytes=remaining_json_bytes,
                    )
                )
                record["metrics"] = metrics
                record["input_summary"] = input_summary
            records.append(record)
    inventory = _tree_inventory(
        output, exclude=frozenset({"controller_manifest.json"})
    )
    if manifest.get("file_inventory") != inventory:
        raise TaskCRehearsalError(
            "rehearsal evidence changed while the summary snapshot was collected"
        )
    return {
        "identity": dict(identity),
        "summary": summary,
        "file_inventory": inventory,
        "runs": records,
        "retained_run_json_bytes": (
            MAXIMUM_REHEARSAL_SUMMARY_JSON_BYTES - remaining_json_bytes
        ),
        "validation_scope": (
            "已用预演目录之外独立保存的恢复令牌重新核对任务 C 全部记录；"
            "本次只做汇总，也没有授权启动五份数据划分的正式比较"
        ),
    }


def _outer_input_summary(
    *,
    condition: str,
    profile: str,
    profile_records: Mapping[str, Mapping[str, Path]],
    method_id: str,
    synthetic_smoke: bool,
) -> dict[str, object]:
    stages: dict[str, object] = {}
    used_stages = (
        ("train", "tune", "refit")
        if profile == "connection" and method_id == "hypersca_c"
        else ("refit",)
    )
    for stage in used_stages:
        manifest = _strict_json_file(
            profile_records[stage]["manifest"], "profile input record"
        )
        transformation = manifest.get("transformation")
        standardization = None
        if transformation == CROSS_TRANSFORMATION:
            standardization = {
                "center": "control mean in each environment",
                "control_label": CONTROL_LABEL,
                "low_scale_replacement": 1.0,
                "low_scale_threshold": 1e-6,
                "scale": "control population standard deviation (ddof=0)",
            }
        stages[stage] = {
            "input_sha256": _sha256_file(profile_records[stage]["input"]),
            "profile_manifest_sha256": _sha256_file(
                profile_records[stage]["manifest"]
            ),
            "parent_files": [
                {
                    "public_relative_path": context.get("public_relative_path"),
                    "role": context.get("role"),
                }
                for context in manifest.get("contexts", [])
                if isinstance(context, dict)
            ],
            "transformation": transformation,
            "control_standardization": standardization,
            "environment_labels": manifest.get("environment_labels"),
        }
    return {
        "schema_version": "1.0",
        "profile": profile,
        "condition": condition,
        "method_id": method_id,
        "stages": stages,
        "used_stages": list(used_stages),
        "training_tuning_and_final_fit_are_separate": len(used_stages) == 3,
        "settings_policy": (
            "two public training candidates, separate public tuning, selected public refit"
            if len(used_stages) == 3
            else "fixed no-tuning reference; registered settings used for public refit"
            if method_id == "mean_difference"
            else "registered default settings used for public refit"
        ),
        "private_data_received_by_method": False,
        "data_scope": "synthetic_smoke" if synthetic_smoke else "external_benchmark",
        "interpretation": (
            "This reduced, single-seed run checks execution and evidence boundaries; "
            "it is not a real-data performance conclusion."
        ),
    }


def run_task_c_rehearsal(
    *,
    profile: str,
    prepared_root: Path,
    method_assets_root: Path,
    output_root: Path,
    method_ids: Sequence[str],
    prepared_identity_sha256: str | None = None,
    expected_resume_token: str | None = None,
    resume: bool = False,
    synthetic_smoke: bool = False,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Run the rehearsal using a preparation identity retained outside the bundle.

    Formal runs require ``prepared_identity_sha256`` copied from the preparation
    command's stdout and stored independently; a hash recomputed from the local
    prepared directory is not an external trust anchor or a signature.
    """

    from src.evaluation.task_c_method_registry import (
        TaskCMethodRegistryError,
        load_task_c_method_registry,
    )
    from src.evaluation.task_c_method_run import TaskCMethodRunError

    if type(resume) is not bool or type(synthetic_smoke) is not bool:
        raise TaskCRehearsalError("resume and synthetic_smoke must be true or false")
    if resume:
        if not _is_sha256_text(expected_resume_token):
            raise TaskCRehearsalError(
                "--resume requires the externally retained resume token"
            )
    elif expected_resume_token is not None:
        raise TaskCRehearsalError("resume token may be supplied only with resume")
    if prepared_identity_sha256 is None:
        if not synthetic_smoke:
            raise TaskCRehearsalError(
                "formal rehearsal requires the independently saved prepared identity fingerprint"
            )
    elif not _is_sha256_text(prepared_identity_sha256):
        raise TaskCRehearsalError("prepared identity fingerprint is malformed")
    output = Path(os.path.abspath(os.fspath(output_root.expanduser())))
    _reject_output_symbolic_links(output)
    if resume and not output.exists():
        raise TaskCRehearsalError("resume output root does not exist")
    root = (project_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
    config_path = root / "configs/task_c_rehearsal_v1.json"
    registry_path = root / "configs/task_c_methods_v1.json"
    config = load_task_c_rehearsal_config(config_path)
    if profile not in config.profiles:
        raise TaskCRehearsalError("profile must be connection or comprehensive")
    methods = _fixed_text_tuple(method_ids, "rehearsal methods")
    if len(set(methods)) != len(methods):
        raise TaskCRehearsalError("rehearsal methods must be unique")
    try:
        registry = load_task_c_method_registry(registry_path)
    except TaskCMethodRegistryError as exc:
        raise TaskCRehearsalError(str(exc)) from exc
    unknown = [method for method in methods if method not in registry.methods]
    if unknown:
        raise TaskCRehearsalError(f"methods are not registered: {unknown}")

    prepared = Path(os.path.abspath(os.fspath(prepared_root.expanduser())))
    assets = Path(os.path.abspath(os.fspath(method_assets_root.expanduser())))
    if _has_private_component(prepared) or _has_private_component(output):
        raise TaskCRehearsalError(
            "public rehearsal inputs and outputs must not use a private path"
        )
    public_manifest, public = _validate_prepared_rehearsal_inputs(
        prepared_root=prepared,
        method_assets_root=assets,
        synthetic_smoke=synthetic_smoke,
    )
    current_prepared_identity_sha256 = _canonical_sha256(
        public.get("materialization_identity")
    )
    if (
        prepared_identity_sha256 is not None
        and prepared_identity_sha256 != current_prepared_identity_sha256
    ):
        raise TaskCRehearsalError(
            "external prepared identity differs from the current materialization"
        )
    closure = _freeze_rehearsal_closure(
        _rehearsal_closure_paths(
            project_root=root,
            prepared_root=prepared,
            method_assets_root=assets,
            synthetic_smoke=synthetic_smoke,
        ),
        external_identity=(
            {"prepared_identity_sha256": prepared_identity_sha256}
            if prepared_identity_sha256 is not None
            else None
        ),
    )
    if not synthetic_smoke:
        _validate_private_rehearsal_inputs(prepared_root=prepared, public=public)
        _verify_private_rehearsal_closure(closure, prepared_root=prepared)
    identity = _controller_identity(
        profile=profile,
        methods=methods,
        synthetic_smoke=synthetic_smoke,
        public_manifest=public_manifest,
        registry_path=registry_path,
        rehearsal_config_path=config_path,
        method_assets_root=assets,
        closure=closure,
        prepared_identity_sha256=prepared_identity_sha256,
    )
    if resume:
        if not output.exists():
            raise TaskCRehearsalError("resume output root does not exist")
        return _resume_verified_rehearsal(
            output_root=output,
            expected_identity=identity,
            required_artifacts=config.required_artifacts,
            expected_resume_token=str(expected_resume_token),
        )
    if output.exists() or output.is_symlink():
        raise TaskCRehearsalError(
            "output root already exists; use --resume only for an exact verified run"
        )

    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    published = False
    try:
        profiles = _profile_inputs(
            public_manifest=public_manifest,
            profile=profile,
            staging=staging,
        )
        (staging / "runs").mkdir(mode=0o700)
        (staging / "work").mkdir(mode=0o700)
        statuses: dict[str, str] = {}
        min_cells = public.get("min_cells_per_intervention")
        if isinstance(min_cells, bool) or not isinstance(min_cells, int):
            raise TaskCRehearsalError("public minimum cell count is invalid")
        timeout = config.profiles[profile].timeout_seconds_per_method

        for condition in REHEARSAL_CONDITIONS:
            condition_profiles = profiles[condition]
            for method_id in methods:
                run_id = build_rehearsal_run_id(
                    profile=profile,
                    condition=condition,
                    method_id=method_id,
                    seed=config.seed,
                )
                outer = staging / "runs" / run_id
                work = staging / "work" / condition / method_id
                work.mkdir(parents=True, mode=0o700)
                inner_dir: Path | None = None
                status_name: str | None = None
                reason: str | None = None
                try:
                    spec = registry.methods[method_id]
                    if spec.source_kind == "publication_only":
                        status_name = "official_assets_unavailable"
                        reason = (
                            "The registered publication has no runnable official assets; "
                            "no substitute prediction was created."
                        )
                    elif synthetic_smoke:
                        predictions = _synthetic_predictions(
                            method_id=method_id,
                            profile_record=condition_profiles["refit"],
                            seed=config.seed,
                        )
                        if profile == "connection" and method_id == "hypersca_c":
                            _write_new_record(
                                work / "selection_record.json",
                                {
                                    "schema_version": "1.0",
                                    "status": "synthetic_selection_closure",
                                    "trial_count": 2,
                                    "selected_trial_index": 0,
                                    "train_input_sha256": _sha256_file(
                                        condition_profiles["train"]["input"]
                                    ),
                                    "tune_input_sha256": _sha256_file(
                                        condition_profiles["tune"]["input"]
                                    ),
                                    "refit_input_sha256": _sha256_file(
                                        condition_profiles["refit"]["input"]
                                    ),
                                    "claim_level": "workflow_validation_only",
                                },
                            )
                        metrics: dict[str, object] = {
                            "average_precision": 1.0,
                            "metric_scope": "synthetic workflow closure only",
                        }
                        if method_id in {"hypersca_c", "mean_difference"}:
                            try:
                                metrics["null_controls"] = _null_control_records(
                                    predictions=predictions,
                                    profile_record=condition_profiles["refit"],
                                    seed=config.seed,
                                    synthetic_smoke=True,
                                )
                            except (ValueError, TypeError, OSError) as exc:
                                raise TaskCRehearsalError(
                                    f"null-control workflow failed: {exc}"
                                ) from exc
                            metrics["null_controls"] = {
                                **dict(metrics["null_controls"]),
                                "scope": "synthetic_orchestration_only",
                                "formal_null_gate_passed": False,
                            }
                        _verify_rehearsal_closure(closure)
                        _publish_outer_success(
                            destination=outer,
                            method_id=method_id,
                            condition=condition,
                            profile=profile,
                            seed=config.seed,
                            predictions=predictions,
                            metrics=metrics,
                            input_summary=_outer_input_summary(
                                condition=condition,
                                profile=profile,
                                profile_records=condition_profiles,
                                method_id=method_id,
                                synthetic_smoke=True,
                            ),
                            inner_dir=None,
                            work_dir=work,
                            synthetic_smoke=True,
                            required_artifacts=config.required_artifacts,
                        )
                        status_name = "passed_synthetic_smoke"
                    else:
                        inner_dir, inner = _run_formal_final_method(
                            method_id=method_id,
                            condition=condition,
                            profile=profile,
                            profiles=condition_profiles,
                            work_dir=work,
                            seed=config.seed,
                            registry_path=registry_path,
                            asset_root=assets,
                            public_manifest=public_manifest,
                            min_cells=min_cells,
                            timeout_seconds=timeout,
                            project_root=root,
                            source_kind=spec.source_kind,
                        )
                        inner_name = inner.get("status")
                        if not isinstance(inner_name, str):
                            raise TaskCRehearsalError(
                                "method result lacks a final status"
                            )
                        status_name = classify_rehearsal_method_status(inner_name)
                        if status_name != "passed_real_rehearsal":
                            _inner_status, inner_reason = _method_bundle_status(inner_dir)
                            reason = inner_reason or f"method ended with {inner_name}"
                        else:
                            predictions_path = inner_dir / "predictions.csv"
                            # Connection trials are reconstructed inside the fixed
                            # selector.  The final selected refit is then validated by
                            # run_task_c_method's selection replay before publication.
                            if not predictions_path.is_file():
                                raise TaskCRehearsalError(
                                    "completed method lacks the complete relation table"
                                )
                            _expression, _labels, expected_genes = _read_profile_arrays(
                                condition_profiles["refit"]
                            )
                            prediction_snapshot = freeze_rehearsal_predictions(
                                source_path=predictions_path,
                                destination=work / "fixed_scoring_predictions.csv",
                                expected_genes=expected_genes,
                            )
                            metrics = _formal_scoring(
                                condition=condition,
                                predictions=prediction_snapshot,
                                prepared_root=prepared,
                                asset_root=assets,
                                work_dir=work,
                                seed=config.seed,
                                registry=registry,
                                project_root=root,
                                public_profile_input=condition_profiles["refit"]["input"],
                                maximum_cells=config.profiles[
                                    profile
                                ].maximum_cells_per_context,
                                rehearsal_closure=closure,
                            )
                            if method_id in {"hypersca_c", "mean_difference"}:
                                try:
                                    selected_hypersca_config = (
                                        work / "selected_refit_config.json"
                                        if (work / "selected_refit_config.json").is_file()
                                        else root / "configs/hypersca_c_v1.json"
                                        if method_id == "hypersca_c"
                                        else None
                                    )
                                    metrics["null_controls"] = _run_formal_null_controls(
                                        method_id=method_id,
                                        predictions=prediction_snapshot.frame,
                                        profile_record=condition_profiles["refit"],
                                        seed=config.seed,
                                        min_cells=min_cells,
                                        hypersca_config_path=selected_hypersca_config,
                                        work_dir=work,
                                        timeout_seconds=min(float(timeout), 300.0),
                                    )
                                except (ValueError, TypeError, OSError) as exc:
                                    raise TaskCRehearsalError(
                                        f"null-control workflow failed: {exc}"
                                    ) from exc
                            _verify_formal_final_method_bundle(
                                method_id=method_id,
                                condition=condition,
                                profile=profile,
                                profiles=condition_profiles,
                                work_dir=work,
                                seed=config.seed,
                                registry_path=registry_path,
                                asset_root=assets,
                                public_manifest=public_manifest,
                                min_cells=min_cells,
                                timeout_seconds=timeout,
                                project_root=root,
                                source_kind=spec.source_kind,
                            )
                            _verify_frozen_predictions(prediction_snapshot)
                            _verify_rehearsal_closure(closure)
                            _publish_outer_success(
                                destination=outer,
                                method_id=method_id,
                                condition=condition,
                                profile=profile,
                                seed=config.seed,
                                predictions=None,
                                metrics=metrics,
                                input_summary=_outer_input_summary(
                                    condition=condition,
                                    profile=profile,
                                    profile_records=condition_profiles,
                                    method_id=method_id,
                                    synthetic_smoke=False,
                                ),
                                inner_dir=inner_dir,
                                work_dir=work,
                                synthetic_smoke=False,
                                required_artifacts=config.required_artifacts,
                                prediction_snapshot=prediction_snapshot,
                            )
                    if status_name not in {
                        "passed_real_rehearsal",
                        "passed_synthetic_smoke",
                    }:
                        assert status_name is not None
                        _verify_rehearsal_closure(closure)
                        _publish_outer_failure(
                            destination=outer,
                            method_id=method_id,
                            condition=condition,
                            profile=profile,
                            seed=config.seed,
                            status=status_name,
                            reason=reason or status_name,
                            inner_dir=inner_dir,
                            work_dir=work,
                            synthetic_smoke=synthetic_smoke,
                        )
                except (
                    TaskCMethodRunError,
                    TaskCRehearsalError,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                ) as exc:
                    if outer.exists():
                        raise
                    if inner_dir is None:
                        candidate = work / "refit"
                        inner_dir = candidate if candidate.exists() else None
                    inner_status, inner_reason = (
                        _method_bundle_status(inner_dir)
                        if inner_dir is not None
                        else (None, None)
                    )
                    if inner_status not in _INNER_TO_REHEARSAL_STATUS:
                        found_dir, found_status, found_reason = (
                            _find_failed_method_bundle(work)
                        )
                        if found_dir is not None:
                            inner_dir = found_dir
                            inner_status = found_status
                            inner_reason = found_reason
                    status_name = _classify_controller_failure(exc, inner_status)
                    reason = inner_reason or _safe_failure_reason(exc)
                    _verify_rehearsal_closure(closure)
                    _publish_outer_failure(
                        destination=outer,
                        method_id=method_id,
                        condition=condition,
                        profile=profile,
                        seed=config.seed,
                        status=status_name,
                        reason=reason,
                        inner_dir=inner_dir,
                        work_dir=work,
                        synthetic_smoke=synthetic_smoke,
                    )
                _verify_rehearsal_closure(closure)
                statuses[f"{condition}/{method_id}"] = str(status_name)

        summary: dict[str, object] = {
            "schema_version": "1.0",
            "profile": profile,
            "attempted_methods": list(methods),
            "conditions": list(REHEARSAL_CONDITIONS),
            "attempted_run_count": len(methods) * len(REHEARSAL_CONDITIONS),
            "status_counts": dict(sorted(Counter(statuses.values()).items())),
            "claim_level": "workflow_validation_only",
            "promotion_eligible": False,
            "resume_status": "new_run",
        }
        _verify_rehearsal_closure(closure)
        inventory = _tree_inventory(staging)
        rebuilt_summary = _rebuild_resumed_summary(
            output_root=staging,
            expected_identity=identity,
            required_artifacts=config.required_artifacts,
        )
        if _resume_token_summary(summary) != _resume_token_summary(rebuilt_summary):
            raise TaskCRehearsalError(
                "new rehearsal summary disagrees with the actual published runs"
            )
        resume_token = _rehearsal_resume_token(
            controller_identity=identity,
            file_inventory=inventory,
            rebuilt_summary=rebuilt_summary,
        )
        summary["resume_token"] = resume_token
        _write_new_record(
            staging / "controller_manifest.json",
            {
                "schema_version": "1.0",
                "identity": identity,
                "identity_sha256": _canonical_sha256(identity),
                "file_inventory": inventory,
                "summary": summary,
                "resume_token": resume_token,
            },
        )
        if output.exists() or output.is_symlink():
            raise TaskCRehearsalError("output root appeared before publication")
        os.rename(staging, output)
        published = True
        return summary
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
