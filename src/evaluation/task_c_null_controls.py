"""Build Task C zero-effect controls for checking the rehearsal workflow.

These helpers test whether random label relationships can look convincing.  They
do not validate a biological mechanism, and their results are never eligible to
advance a claim to the next evidence stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
import unicodedata

import numpy as np

from src.evaluation.task_c_profile_input import (
    MAXIMUM_DISTINCT_LABELS,
    MAXIMUM_PARENT_CELLS,
    MAXIMUM_PARENT_GENES,
    MAXIMUM_TEXT_ITEM_BYTES,
    MAXIMUM_TOTAL_TEXT_BYTES,
)


CONTROL_LABEL = "non-targeting"
NULL_REPEAT_COUNT = 20
MAXIMUM_EMPIRICAL_P_VALUE = 0.05
MINIMUM_EMPIRICAL_ADVANTAGE = 0.0
MAXIMUM_SEED = 2**32 - 1
_PERMUTATION_ATTEMPTS = 32


class TaskCNullControlError(ValueError):
    """The proposed zero-effect check is not safe or scientifically complete."""


def _validated_seed(seed: object) -> int:
    if type(seed) is not int or not 0 <= seed <= MAXIMUM_SEED:
        raise TaskCNullControlError(
            f"seed must be a whole number from 0 to {MAXIMUM_SEED}"
        )
    return seed


def _safe_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TaskCNullControlError(f"{label} must contain only text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise TaskCNullControlError(f"{label} must use valid UTF-8 text") from exc
    if len(encoded) > MAXIMUM_TEXT_ITEM_BYTES:
        raise TaskCNullControlError(f"{label} exceeds the per-item text limit")
    if not unicodedata.is_normalized("NFC", value):
        raise TaskCNullControlError(f"{label} must use NFC-normalized text")
    if (
        not value
        or value != value.strip()
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise TaskCNullControlError(
            f"{label} must contain safe, non-empty, trimmed text"
        )
    return value


def _copy_label_sequence(labels: object) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise TaskCNullControlError(
            "intervention labels must be a one-dimensional text list"
        )
    if isinstance(labels, np.ndarray):
        if labels.ndim != 1:
            raise TaskCNullControlError(
                "intervention labels must be a one-dimensional text list"
            )
        reported_count = len(labels)
        if reported_count > MAXIMUM_PARENT_CELLS:
            raise TaskCNullControlError("intervention labels contain too many cells")
        raw_values = labels.tolist()
    else:
        if not isinstance(labels, Sequence):
            raise TaskCNullControlError(
                "intervention labels must be a one-dimensional text list"
            )
        try:
            reported_count = len(labels)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TaskCNullControlError(
                "intervention labels must report a safe cell count"
            ) from exc
        if reported_count > MAXIMUM_PARENT_CELLS:
            raise TaskCNullControlError("intervention labels contain too many cells")
        try:
            raw_values = [labels[index] for index in range(reported_count)]
        except (IndexError, KeyError, TypeError) as exc:
            raise TaskCNullControlError(
                "intervention labels changed while being checked"
            ) from exc
    if not raw_values:
        raise TaskCNullControlError("intervention labels must not be empty")

    copied: list[str] = []
    total_bytes = 0
    maximum_characters = 0
    for value in raw_values:
        safe_value = _safe_text(value, "intervention labels")
        encoded_length = len(safe_value.encode("utf-8"))
        total_bytes += encoded_length
        maximum_characters = max(maximum_characters, len(safe_value))
        projected_array_bytes = (
            reported_count * maximum_characters * np.dtype("U1").itemsize
        )
        if (
            total_bytes > MAXIMUM_TOTAL_TEXT_BYTES
            or projected_array_bytes > MAXIMUM_TOTAL_TEXT_BYTES
        ):
            raise TaskCNullControlError(
                "intervention labels exceed the total text limit"
            )
        copied.append(safe_value)
    return tuple(copied)


def _validated_labels(labels: object, control_label: object) -> tuple[str, ...]:
    control = _safe_text(control_label, "control label")
    copied = _copy_label_sequence(labels)
    counts = Counter(copied)
    if len(counts) > MAXIMUM_DISTINCT_LABELS:
        raise TaskCNullControlError(
            "distinct label groups exceed the public rehearsal limit"
        )
    if control not in counts:
        raise TaskCNullControlError(
            "the control label must be present before a zero-effect check"
        )
    if len(counts) < 2:
        raise TaskCNullControlError(
            "at least one intervention group is needed alongside controls"
        )
    return copied


def _immutable_array(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    if contiguous.dtype.hasobject:
        raise TaskCNullControlError("returned arrays cannot contain Python objects")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


def _immutable_labels(values: tuple[str, ...]) -> np.ndarray:
    width = max(len(value) for value in values)
    projected = len(values) * width * np.dtype("U1").itemsize
    if projected > MAXIMUM_TOTAL_TEXT_BYTES:
        raise TaskCNullControlError("returned labels exceed the total text limit")
    return _immutable_array(np.asarray(values, dtype=f"U{width}"))


def permute_intervention_labels(
    labels: Sequence[str], seed: int, *, control_label: str = CONTROL_LABEL
) -> np.ndarray:
    """Shuffle all labels while preserving each observed group size.

    A shuffle that accidentally leaves every row unchanged is retried a bounded
    number of times.  Such an unchanged result is never reported as a valid
    zero-effect dataset.
    """

    checked_seed = _validated_seed(seed)
    copied = _validated_labels(labels, control_label)
    original = np.asarray(copied)
    rng = np.random.default_rng(checked_seed)
    for _ in range(_PERMUTATION_ATTEMPTS):
        permuted = rng.permutation(original)
        if not np.array_equal(permuted, original):
            return _immutable_array(permuted)
    raise TaskCNullControlError(
        "label shuffling repeatedly left every cell unchanged; no valid null was made"
    )


def _numeric_expression(expression: object, expected_cells: int) -> np.ndarray:
    try:
        raw = np.asarray(expression)
    except (TypeError, ValueError) as exc:
        raise TaskCNullControlError(
            "expression must be a finite two-dimensional numeric matrix"
        ) from exc
    if raw.ndim != 2:
        raise TaskCNullControlError("expression must be a two-dimensional matrix")
    if raw.shape[0] != expected_cells:
        raise TaskCNullControlError(
            "expression and labels must describe the same number of cells"
        )
    if raw.shape[1] < 1:
        raise TaskCNullControlError("expression must contain at least one gene")
    if raw.shape[1] > MAXIMUM_PARENT_GENES:
        raise TaskCNullControlError("expression contains too many genes")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TaskCNullControlError("expression must contain numeric values")
    try:
        values = raw.astype(np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TaskCNullControlError("expression must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise TaskCNullControlError("expression must contain only finite values")
    return values


def build_control_resampling_null(
    expression: np.ndarray,
    labels: Sequence[str],
    seed: int,
    control_label: str = CONTROL_LABEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Make fake intervention groups using expression sampled from controls.

    Sampling is with replacement.  Original group labels and group sizes are
    retained so the rehearsal exercises the same downstream workflow.
    """

    checked_seed = _validated_seed(seed)
    control = _safe_text(control_label, "control label")
    copied_labels = _validated_labels(labels, control)
    values = _numeric_expression(expression, len(copied_labels))
    control_rows = np.fromiter(
        (index for index, label in enumerate(copied_labels) if label == control),
        dtype=np.int64,
    )
    rng = np.random.default_rng(checked_seed)
    sampled_rows = rng.choice(
        control_rows, size=len(copied_labels), replace=True
    )
    return (
        _immutable_array(values[sampled_rows]),
        _immutable_labels(copied_labels),
    )


_RESULT_FIELDS = (
    "real_metric",
    "null_median",
    "null_maximum",
    "empirical_advantage",
    "empirical_p_value",
    "repeat_count",
    "minimum_empirical_advantage",
    "maximum_empirical_p_value",
    "passed",
    "evidence_scope",
    "promotion_eligible",
)


@dataclass(frozen=True, slots=True)
class EmpiricalNullCheck(Mapping[str, float | bool | int | str]):
    """Read-only result of the pre-agreed rehearsal-only null comparison."""

    real_metric: float
    null_median: float
    null_maximum: float
    empirical_advantage: float
    empirical_p_value: float
    repeat_count: int
    minimum_empirical_advantage: float
    maximum_empirical_p_value: float
    passed: bool
    evidence_scope: str = "workflow_rehearsal_only"
    promotion_eligible: bool = False

    def __getitem__(self, key: str) -> float | bool | int | str:
        if key not in _RESULT_FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_RESULT_FIELDS)

    def __len__(self) -> int:
        return len(_RESULT_FIELDS)


def _finite_metric(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TaskCNullControlError(f"{label} must be one finite numeric value")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TaskCNullControlError(
            f"{label} must be one finite numeric value"
        ) from exc
    if not math.isfinite(converted):
        raise TaskCNullControlError(f"{label} must be one finite numeric value")
    return converted


def _null_metric_vector(null_metrics: object) -> np.ndarray:
    if isinstance(null_metrics, (str, bytes)):
        raise TaskCNullControlError(
            "exactly twenty finite null metrics are required"
        )
    try:
        raw = np.asarray(null_metrics)
    except (TypeError, ValueError) as exc:
        raise TaskCNullControlError(
            "null metrics must be a one-dimensional numeric list"
        ) from exc
    if raw.ndim != 1:
        raise TaskCNullControlError(
            "null metrics must be a one-dimensional numeric list"
        )
    if len(raw) != NULL_REPEAT_COUNT:
        raise TaskCNullControlError(
            "exactly twenty finite null metrics are required"
        )
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TaskCNullControlError("null metrics must contain numeric values")
    try:
        copied = raw.astype(np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TaskCNullControlError(
            "null metrics must contain numeric values"
        ) from exc
    if not np.isfinite(copied).all():
        raise TaskCNullControlError(
            "exactly twenty finite null metrics are required"
        )
    return copied


def empirical_null_check(
    real_metric: float,
    null_metrics: Sequence[float],
    maximum_p_value: float,
    minimum_empirical_advantage: float = MINIMUM_EMPIRICAL_ADVANTAGE,
) -> EmpiricalNullCheck:
    """Apply the frozen 20-repeat rule used only to rehearse the workflow."""

    real = _finite_metric(real_metric, "real metric")
    null = _null_metric_vector(null_metrics)
    if type(maximum_p_value) is not float:
        raise TaskCNullControlError(
            "maximum_p_value must be the fixed finite decimal value 0.05"
        )
    if not math.isfinite(maximum_p_value) or maximum_p_value != MAXIMUM_EMPIRICAL_P_VALUE:
        raise TaskCNullControlError("maximum_p_value must remain 0.05")
    if type(minimum_empirical_advantage) is not float:
        raise TaskCNullControlError(
            "minimum_empirical_advantage must be the fixed finite decimal value 0.0"
        )
    if (
        not math.isfinite(minimum_empirical_advantage)
        or minimum_empirical_advantage != MINIMUM_EMPIRICAL_ADVANTAGE
    ):
        raise TaskCNullControlError(
            "minimum_empirical_advantage must remain 0.0"
        )

    null_maximum = float(np.max(null))
    ordered_null = np.sort(null)
    middle = NULL_REPEAT_COUNT // 2
    null_median = float(
        ordered_null[middle - 1] / 2.0 + ordered_null[middle] / 2.0
    )
    with np.errstate(over="ignore", invalid="ignore"):
        advantage = float(real - null_maximum)
    if not all(math.isfinite(value) for value in (null_maximum, null_median, advantage)):
        raise TaskCNullControlError(
            "the derived advantage and null summaries must remain finite"
        )
    empirical_p_value = float(
        (1 + np.count_nonzero(null >= real)) / (NULL_REPEAT_COUNT + 1)
    )
    passed = bool(
        advantage > minimum_empirical_advantage
        and empirical_p_value <= maximum_p_value
    )
    return EmpiricalNullCheck(
        real_metric=real,
        null_median=null_median,
        null_maximum=null_maximum,
        empirical_advantage=advantage,
        empirical_p_value=empirical_p_value,
        repeat_count=NULL_REPEAT_COUNT,
        minimum_empirical_advantage=minimum_empirical_advantage,
        maximum_empirical_p_value=maximum_p_value,
        passed=passed,
    )
