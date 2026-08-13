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
REHEARSAL_CONDITIONS = (
    "within_k562",
    "within_rpe1",
    "k562_to_rpe1",
    "rpe1_to_k562",
)
_FINAL_REHEARSAL_STATUSES = frozenset(
    {
        "passed_real_rehearsal",
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
            and method in {"hypersca_c", "mean_difference"}
            else 0
        )
        for method in methods
    }
    selected = tuple(
        method
        for method in ("hypersca_c", "mean_difference")
        if trial_counts.get(method) == 2
    )
    return MappingProxyType(
        {
            condition: MappingProxyType(
                {
                    "stages": ("train", "tune", "refit"),
                    "trial_counts": MappingProxyType(dict(trial_counts)),
                    "selection_bound_refit": selected,
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


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except TaskCRehearsalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TaskCRehearsalError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TaskCRehearsalError(f"{label} must contain one JSON object")
    return payload


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
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        if path.is_symlink() or not path.is_file():
            raise TaskCRehearsalError(
                f"rehearsal output contains an unsupported path: {relative}"
            )
        inventory[relative] = _sha256_file(path)
    return inventory


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
    try:
        with np.load(profile_record["input"], allow_pickle=False) as archive:
            expression = np.asarray(archive["expression_matrix"], dtype=np.float64)
            labels = np.asarray(archive["interventions"], dtype=str)
            genes = tuple(str(value) for value in archive["var_names"].tolist())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TaskCRehearsalError("profile arrays could not be read") from exc
    return expression, labels, genes


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
    try:
        with np.load(public_profile_input, allow_pickle=False) as profile_archive:
            profile_genes = _canonical_texts(
                np.asarray(profile_archive["var_names"]),
                "public profile genes",
                require_unique=True,
                maximum_items=MAXIMUM_PARENT_GENES,
            )
        with np.load(source_path, allow_pickle=False) as source_archive:
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
    predictions: Any,
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
    try:
        return run_task_c_method(**arguments)  # type: ignore[arg-type]
    except TaskCMethodRunError:
        raise


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
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
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
    if profile == "connection" and method_id in {"hypersca_c", "mean_difference"}:
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
    )
    return refit, status


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

    observed = _sha256_file(path).removeprefix("sha256:")
    if expected_sha256.removeprefix("sha256:") != observed:
        raise TaskCRehearsalError("reference-relation file fingerprint changed")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
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
    predictions: Path,
    prepared_root: Path,
    asset_root: Path,
    work_dir: Path,
    seed: int,
    registry: Any,
    project_root: Path,
    public_profile_input: Path,
    maximum_cells: int,
) -> dict[str, object]:
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
        metrics["sealed_scoring_input"] = {
            "source_sha256": _sha256_file(source_heldout),
            "profile_subset_sha256": _sha256_file(heldout),
            "public_profile_input_sha256": _sha256_file(public_profile_input),
        }
        return metrics
    finally:
        if private_root.exists() and not private_root.is_symlink():
            shutil.rmtree(private_root)


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
    import pandas as pd

    from src.evaluation.task_c_aggregation import (
        TaskCAggregationError,
        evaluate_declared_references,
        task_c_aggregation_to_jsonable,
    )

    scope = _condition_scope(condition)
    prediction_sha256 = _sha256_file(predictions)
    heldout_sha256 = _sha256_file(heldout)
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
    if (
        _sha256_file(predictions) != prediction_sha256
        or _sha256_file(heldout) != heldout_sha256
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
        with np.load(heldout, allow_pickle=False) as archive:
            labels = np.asarray(archive["interventions"], dtype=str)
        eligible_sources = set(labels.tolist()) - {CONTROL_LABEL, "excluded"}
        scores = pd.read_csv(predictions)
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
    if (
        _sha256_file(predictions) != prediction_sha256
        or _sha256_file(heldout) != heldout_sha256
    ):
        raise TaskCRehearsalError(
            "sealed scoring inputs changed during biological-reference scoring"
        )
    assert isinstance(metrics, dict)
    metrics["supplementary_official_metrics"] = official["metrics"]
    metrics["sealed_scoring_status"] = official["status"]
    return metrics


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


def _resource_record(inner_dir: Path | None, *, null_repeat_count: int) -> dict[str, object]:
    inner: object = None
    if inner_dir is not None:
        candidates = sorted(inner_dir.rglob("resource_usage.json"))
        if candidates:
            inner = _strict_json_file(candidates[-1], "method resource record")
    return {
        "schema_version": "1.0",
        "resource_scope": "single-seed reduced-data rehearsal",
        "method_resource_record": inner,
        "null_control_repeat_count_per_type": null_repeat_count,
    }


def _publish_outer_success(
    *,
    destination: Path,
    method_id: str,
    condition: str,
    profile: str,
    seed: int,
    predictions: Any,
    metrics: Mapping[str, object],
    input_summary: Mapping[str, object],
    inner_dir: Path | None,
    synthetic_smoke: bool,
    required_artifacts: Sequence[str],
) -> None:
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists() or destination.exists():
        raise TaskCRehearsalError("run result directory already exists")
    staging.mkdir(parents=True, mode=0o700)
    try:
        predictions.to_csv(staging / "predictions.csv", index=False)
        run_identity = {
            "schema_version": "1.0",
            "profile": profile,
            "condition": condition,
            "method_id": method_id,
            "seed": seed,
            "input_summary_sha256": _canonical_sha256(input_summary),
            "prediction_sha256": _sha256_file(staging / "predictions.csv"),
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
        _write_new_record(staging / "input_summary.json", dict(input_summary))
        _write_new_record(staging / "metrics.json", dict(metrics))
        _write_new_record(staging / "promotion_decision.json", _promotion_record())
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
            _resource_record(inner_dir, null_repeat_count=20),
        )
        _write_new_record(
            staging / "method_status.json",
            {
                "schema_version": "1.0",
                "method_id": method_id,
                "condition": condition,
                "seed": seed,
                "run_identity_sha256": identity_sha256,
                "status": "passed_real_rehearsal",
                "controller_validation": "verified_task_c_rehearsal_bundle_v1",
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
    synthetic_smoke: bool,
) -> None:
    if status not in _FINAL_REHEARSAL_STATUSES - {"passed_real_rehearsal"}:
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
            _resource_record(inner_dir, null_repeat_count=0),
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

    private_manifest = prepared_root / "private/private_manifest.json"
    private = _strict_json_file(private_manifest, "sealed data record")
    private_files = private.get("files")
    if not isinstance(private_files, dict) or not private_files:
        raise TaskCRehearsalError("sealed data record lacks its fixed file inventory")
    for relative, expected in private_files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TaskCRehearsalError("sealed data file inventory is malformed")
        path = prepared_root / relative
        if _sha256_file(path).removeprefix("sha256:") != expected.removeprefix(
            "sha256:"
        ):
            raise TaskCRehearsalError(
                f"sealed data file fingerprint changed: {relative}"
            )
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
        "method_registry_sha256": _sha256_file(registry_path),
        "rehearsal_config_sha256": _sha256_file(rehearsal_config_path),
        "method_assets_identity_sha256": asset_identity,
        "claim_level": "workflow_validation_only",
        "promotion_eligible": False,
    }


def _resume_verified_rehearsal(
    *,
    output_root: Path,
    expected_identity: Mapping[str, object],
    required_artifacts: Sequence[str],
) -> dict[str, object]:
    manifest_path = output_root / "controller_manifest.json"
    observed = _strict_json_file(manifest_path, "rehearsal controller record")
    if observed.get("identity") != dict(expected_identity):
        raise TaskCRehearsalError(
            "existing rehearsal identity differs from the requested inputs"
        )
    run_dirs = sorted((output_root / "runs").iterdir())
    for run_dir in run_dirs:
        status = _strict_json_file(run_dir / "method_status.json", "method status")
        if status.get("status") == "passed_real_rehearsal":
            validate_required_run_artifacts(run_dir, required_artifacts)
    expected_inventory = observed.get("file_inventory")
    actual_inventory = _tree_inventory(
        output_root, exclude=frozenset({"controller_manifest.json"})
    )
    if expected_inventory != actual_inventory:
        raise TaskCRehearsalError(
            "existing rehearsal output fingerprint changed"
        )
    summary = observed.get("summary")
    if not isinstance(summary, dict):
        raise TaskCRehearsalError("existing rehearsal summary is missing")
    return {**summary, "resume_status": "verified_existing_output"}


def _outer_input_summary(
    *,
    condition: str,
    profile: str,
    profile_records: Mapping[str, Mapping[str, Path]],
    method_id: str,
    synthetic_smoke: bool,
) -> dict[str, object]:
    stages: dict[str, object] = {}
    for stage in ("train", "tune", "refit"):
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
        "training_tuning_and_final_fit_are_separate": True,
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
    resume: bool = False,
    synthetic_smoke: bool = False,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Run the fixed four-condition rehearsal without raising its claim level."""

    from src.evaluation.task_c_method_registry import (
        TaskCMethodRegistryError,
        load_task_c_method_registry,
    )
    from src.evaluation.task_c_method_run import TaskCMethodRunError

    if type(resume) is not bool or type(synthetic_smoke) is not bool:
        raise TaskCRehearsalError("resume and synthetic_smoke must be true or false")
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
    output = Path(os.path.abspath(os.fspath(output_root.expanduser())))
    if _has_private_component(prepared) or _has_private_component(output):
        raise TaskCRehearsalError(
            "public rehearsal inputs and outputs must not use a private path"
        )
    public_manifest, public = _validate_prepared_rehearsal_inputs(
        prepared_root=prepared,
        method_assets_root=assets,
        synthetic_smoke=synthetic_smoke,
    )
    identity = _controller_identity(
        profile=profile,
        methods=methods,
        synthetic_smoke=synthetic_smoke,
        public_manifest=public_manifest,
        registry_path=registry_path,
        rehearsal_config_path=config_path,
        method_assets_root=assets,
    )
    if output.exists() or output.is_symlink():
        if not resume:
            raise TaskCRehearsalError(
                "output root already exists; use --resume only for an exact verified run"
            )
        return _resume_verified_rehearsal(
            output_root=output,
            expected_identity=identity,
            required_artifacts=config.required_artifacts,
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
                        if profile == "connection" and method_id in {
                            "hypersca_c",
                            "mean_difference",
                        }:
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
                            synthetic_smoke=True,
                            required_artifacts=config.required_artifacts,
                        )
                        status_name = "passed_real_rehearsal"
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
                            import pandas as pd

                            predictions = pd.read_csv(predictions_path)
                            metrics = _formal_scoring(
                                condition=condition,
                                predictions=predictions_path,
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
                            )
                            if method_id in {"hypersca_c", "mean_difference"}:
                                try:
                                    metrics["null_controls"] = _null_control_records(
                                        predictions=predictions,
                                        profile_record=condition_profiles["refit"],
                                        seed=config.seed,
                                        synthetic_smoke=False,
                                    )
                                except (ValueError, TypeError, OSError) as exc:
                                    raise TaskCRehearsalError(
                                        f"null-control workflow failed: {exc}"
                                    ) from exc
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
                                    synthetic_smoke=False,
                                ),
                                inner_dir=inner_dir,
                                synthetic_smoke=False,
                                required_artifacts=config.required_artifacts,
                            )
                    if status_name != "passed_real_rehearsal":
                        assert status_name is not None
                        _publish_outer_failure(
                            destination=outer,
                            method_id=method_id,
                            condition=condition,
                            profile=profile,
                            seed=config.seed,
                            status=status_name,
                            reason=reason or status_name,
                            inner_dir=inner_dir,
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
                    _publish_outer_failure(
                        destination=outer,
                        method_id=method_id,
                        condition=condition,
                        profile=profile,
                        seed=config.seed,
                        status=status_name,
                        reason=reason,
                        inner_dir=inner_dir,
                        synthetic_smoke=synthetic_smoke,
                    )
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
        inventory = _tree_inventory(staging)
        _write_new_record(
            staging / "controller_manifest.json",
            {
                "schema_version": "1.0",
                "identity": identity,
                "identity_sha256": _canonical_sha256(identity),
                "file_inventory": inventory,
                "summary": summary,
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
