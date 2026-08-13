"""Freeze the Task C rehearsal scope and provide small array checks.

The array helpers in this module make the scientific rules easy to test.  A real
rehearsal must still use the comparison-profile input and its analysis record
from :mod:`src.evaluation.task_c_profile_input`; these helpers are not a second
way to prepare real data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import shlex
import stat
from types import MappingProxyType
from typing import Any
import unicodedata
from urllib.parse import unquote, urlparse

import numpy as np

from src.evaluation.task_c_profile_input import (
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


class TaskCRehearsalError(ValueError):
    """The rehearsal rule or supplied research data are not safe to use."""


@dataclass(frozen=True, slots=True)
class MethodWorkerEntrySnapshot:
    """Identity of the one reviewed Python file allowed to start a method."""

    path: Path
    sha256: str
    identity: tuple[int, ...]


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
    values: Sequence[str | Path] | None,
) -> set[Path]:
    if values is None or isinstance(values, (str, bytes)) or not isinstance(
        values, Sequence
    ):
        raise TaskCRehearsalError(
            "allowed Python interpreters must be an ordered path list"
        )
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


def validate_private_scoring_command(
    command: Sequence[str],
    *,
    private_root: str | Path,
    execution_cwd: str | Path | None = None,
    allowed_python_interpreters: Sequence[str | Path] | None = None,
    allowed_worker_snapshots: Sequence[MethodWorkerEntrySnapshot] | None = None,
) -> None:
    """Prove that a method process receives no path into sealed scoring data.

    Relative paths, option assignments and symbolic-link aliases are resolved
    before comparison.  The independently run scoring process is allowed to
    read this directory; a method-training process is not.
    """

    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TaskCRehearsalError("method command must be an ordered list of text")
    try:
        count = len(command)
        arguments = tuple(command[index] for index in range(count))
    except (IndexError, KeyError, TypeError, OverflowError) as exc:
        raise TaskCRehearsalError(
            "method command must be an ordered list of text"
        ) from exc
    if not arguments or any(type(argument) is not str for argument in arguments):
        raise TaskCRehearsalError("method command must be an ordered list of text")
    if any("\x00" in argument for argument in arguments):
        raise TaskCRehearsalError("method command text must not contain NUL")

    private = _real_private_directory(private_root)
    if execution_cwd is None:
        raise TaskCRehearsalError(
            "method execution directory is required for path checking"
        )
    cwd = _real_execution_directory(execution_cwd)
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
    allowed_interpreters = _normalized_python_interpreters(
        allowed_python_interpreters
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
    if (
        allowed_worker_snapshots is None
        or isinstance(allowed_worker_snapshots, (str, bytes))
        or not isinstance(allowed_worker_snapshots, Sequence)
        or not allowed_worker_snapshots
        or any(
            type(snapshot) is not MethodWorkerEntrySnapshot
            for snapshot in allowed_worker_snapshots
        )
    ):
        raise TaskCRehearsalError(
            "allowed workers must use frozen entry snapshots"
        )
    snapshots_by_path: dict[Path, MethodWorkerEntrySnapshot] = {}
    for snapshot in allowed_worker_snapshots:
        if snapshot.path in snapshots_by_path:
            raise TaskCRehearsalError(
                "allowed worker snapshots must contain unique entries"
            )
        snapshots_by_path[snapshot.path] = snapshot
    if worker_entry not in snapshots_by_path:
        raise TaskCRehearsalError(
            "method command does not use a registered worker entry"
        )

    private_text = os.fspath(private)
    for argument in arguments:
        for candidate_text in _command_path_candidates(argument):
            if private_text in candidate_text:
                raise TaskCRehearsalError(
                    "method command contains a private scoring path"
                )
            try:
                candidate = Path(candidate_text).expanduser()
                if not candidate.is_absolute():
                    candidate = cwd / candidate
                lexical = Path(os.path.normpath(os.fspath(candidate)))
                if _path_is_within(lexical, private):
                    raise TaskCRehearsalError(
                        "method command contains a private scoring path"
                    )
                resolved = candidate.resolve(strict=False)
            except TaskCRehearsalError:
                raise
            except (OSError, RuntimeError, ValueError):
                continue
            if resolved == private or private in resolved.parents:
                raise TaskCRehearsalError(
                    "method command contains a private scoring path"
                )
    _verify_method_worker_entry(
        requested_worker_entry,
        snapshots_by_path[worker_entry],
    )
