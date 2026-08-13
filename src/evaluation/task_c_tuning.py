"""Choose Task C settings from public tuning responses, never final answers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score


CONTROL_LABEL = "non-targeting"
EXCLUDED_LABEL = "excluded"
MINIMUM_CELLS_PER_GROUP = 5
MAXIMUM_TRIALS = 20
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "maximum_trials_per_method",
        "selection_metric",
        "q_value_threshold",
        "tie_break",
        "external_biological_references_allowed",
        "final_holdout_allowed",
    }
)
_REQUIRED_PREDICTION_COLUMNS = ("source", "target", "score")


class TaskCTuningError(ValueError):
    """Tuning evidence does not follow the fixed, holdout-free rules."""


@dataclass(frozen=True)
class TaskCTuningConfig:
    schema_version: str
    maximum_trials_per_method: int
    selection_metric: str
    q_value_threshold: float
    tie_break: tuple[str, str]
    external_biological_references_allowed: bool
    final_holdout_allowed: bool


@dataclass(frozen=True)
class TaskCTuningSelection(Mapping[str, object]):
    """Read-only selection record; ``dict(record)`` is safe for JSON writing."""

    schema_version: str
    selected_trial_index: int
    selected_parameters: Mapping[str, object]
    average_precision: float
    completed_trial_count: int
    external_biological_references_used: bool = False
    final_holdout_used: bool = False

    def _mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "selected_trial_index": self.selected_trial_index,
                "selected_parameters": self.selected_parameters,
                "average_precision": self.average_precision,
                "completed_trial_count": self.completed_trial_count,
                "external_biological_references_used": (
                    self.external_biological_references_used
                ),
                "final_holdout_used": self.final_holdout_used,
            }
        )

    def __getitem__(self, key: str) -> object:
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskCTuningError(f"configuration JSON contains duplicate field {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise TaskCTuningError(f"configuration JSON contains non-finite value {token}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        )
    except TaskCTuningError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskCTuningError("tuning configuration is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TaskCTuningError("tuning configuration must be one JSON object")
    return payload


def load_task_c_tuning_config(path: Path) -> TaskCTuningConfig:
    """Load the closed tuning policy and reject any local relaxation."""

    payload = _read_json_object(Path(path))
    if set(payload) != _CONFIG_FIELDS:
        raise TaskCTuningError("tuning configuration fields changed")
    maximum = payload.get("maximum_trials_per_method")
    q_value = payload.get("q_value_threshold")
    if isinstance(maximum, bool) or maximum != MAXIMUM_TRIALS:
        raise TaskCTuningError("tuning configuration must keep the twenty-trial limit")
    if isinstance(q_value, bool) or not isinstance(q_value, (int, float)):
        raise TaskCTuningError("q-value threshold must be a finite number")
    if not math.isfinite(float(q_value)) or float(q_value) != 0.1:
        raise TaskCTuningError("q-value threshold must be the fixed finite value 0.1")
    if payload.get("schema_version") != "1.0":
        raise TaskCTuningError("tuning configuration schema changed")
    if payload.get("selection_metric") != (
        "average_precision_against_tuning_response_edges"
    ):
        raise TaskCTuningError("tuning selection metric changed")
    if payload.get("tie_break") != [
        "average_precision_descending",
        "trial_index_ascending",
    ]:
        raise TaskCTuningError("tuning tie rule changed")
    if payload.get("external_biological_references_allowed") is not False:
        raise TaskCTuningError("external biological references must remain unavailable")
    if payload.get("final_holdout_allowed") is not False:
        raise TaskCTuningError("final holdout must remain unavailable during tuning")
    return TaskCTuningConfig(
        schema_version="1.0",
        maximum_trials_per_method=MAXIMUM_TRIALS,
        selection_metric=str(payload["selection_metric"]),
        q_value_threshold=0.1,
        tie_break=(
            "average_precision_descending",
            "trial_index_ascending",
        ),
        external_biological_references_allowed=False,
        final_holdout_allowed=False,
    )


def _canonical_text(values: object, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TaskCTuningError(f"{label} must be a sequence of canonical text")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TaskCTuningError(f"{label} must be canonical text") from exc
    if array.ndim != 1 or array.dtype.kind not in {"U", "S", "O"}:
        raise TaskCTuningError(f"{label} must be a one-dimensional text list")
    items: list[str] = []
    for raw in array.tolist():
        if not isinstance(raw, str):
            raise TaskCTuningError(f"{label} must contain text")
        if (
            not raw
            or raw != raw.strip()
            or not unicodedata.is_normalized("NFC", raw)
            or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        ):
            raise TaskCTuningError(f"{label} must contain safe canonical NFC text")
        items.append(raw)
    return tuple(items)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values)
    if values.ndim != 1 or values.size == 0:
        raise TaskCTuningError("multiple-testing correction needs p-values")
    if values.dtype.kind not in {"i", "u", "f"}:
        raise TaskCTuningError("p-values must be numeric")
    values = values.astype(float, copy=False)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise TaskCTuningError("p-values must be finite values between zero and one")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def build_tuning_response_edges(
    expression: np.ndarray,
    interventions: Sequence[str],
    gene_names: Sequence[str],
    *,
    eligible_sources: Iterable[str],
    q_value_threshold: float,
) -> frozenset[tuple[str, str]]:
    """Derive directed response labels from public tuning cells only."""

    raw_values = np.asarray(expression)
    if raw_values.ndim != 2 or raw_values.shape[0] == 0 or raw_values.shape[1] < 2:
        raise TaskCTuningError("expression matrix shape must be cells by at least two genes")
    if raw_values.dtype.kind not in {"i", "u", "f"}:
        raise TaskCTuningError("expression matrix must be numeric")
    values = raw_values.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise TaskCTuningError("expression matrix must contain finite values")
    labels = _canonical_text(interventions, "intervention labels")
    genes = _canonical_text(gene_names, "gene names")
    if len(labels) != values.shape[0] or len(genes) != values.shape[1]:
        raise TaskCTuningError("expression shape and labels do not agree")
    if len(set(genes)) != len(genes):
        raise TaskCTuningError("gene names must be unique")
    if CONTROL_LABEL not in labels:
        raise TaskCTuningError("tuning responses need non-targeting control cells")
    if not set(labels).issubset(set(genes) | {CONTROL_LABEL, EXCLUDED_LABEL}):
        raise TaskCTuningError("intervention labels contain an unknown gene")
    sources = _canonical_text(tuple(eligible_sources), "eligible sources")
    if len(set(sources)) != len(sources):
        raise TaskCTuningError("eligible sources must be unique")
    if not sources or set(sources) & {CONTROL_LABEL, EXCLUDED_LABEL}:
        raise TaskCTuningError("eligible sources cannot be control or excluded labels")
    if not set(sources).issubset(set(genes)):
        raise TaskCTuningError("eligible sources must be a subset of gene names")
    observed = set(labels)
    if not set(sources).issubset(observed):
        raise TaskCTuningError("every eligible source must be observed in tuning cells")
    if isinstance(q_value_threshold, bool) or not isinstance(
        q_value_threshold, (int, float)
    ):
        raise TaskCTuningError("q-value threshold must be the finite fixed value 0.1")
    if not math.isfinite(float(q_value_threshold)) or float(q_value_threshold) != 0.1:
        raise TaskCTuningError("q-value threshold must be the finite fixed value 0.1")

    label_array = np.asarray(labels, dtype=str)
    controls = label_array == CONTROL_LABEL
    if int(controls.sum()) < MINIMUM_CELLS_PER_GROUP:
        raise TaskCTuningError("tuning responses need at least five control cells")
    tests: list[tuple[str, str, float]] = []
    for source in sorted(sources):
        perturbed = label_array == source
        if int(perturbed.sum()) < MINIMUM_CELLS_PER_GROUP:
            raise TaskCTuningError(
                f"eligible source {source} needs at least five tuning cells"
            )
        for target_index, target in enumerate(genes):
            if source == target:
                continue
            control_values = values[controls, target_index]
            perturbed_values = values[perturbed, target_index]
            if np.array_equal(control_values, perturbed_values):
                p_value = 1.0
            else:
                try:
                    result = mannwhitneyu(
                        perturbed_values,
                        control_values,
                        alternative="two-sided",
                        method="auto",
                    )
                    p_value = float(result.pvalue)
                except (TypeError, ValueError, FloatingPointError) as exc:
                    raise TaskCTuningError(
                        f"response comparison failed for {source} to {target}"
                    ) from exc
                if not math.isfinite(p_value):
                    raise TaskCTuningError(
                        f"response comparison was non-finite for {source} to {target}"
                    )
            tests.append((source, target, p_value))
    q_values = _benjamini_hochberg(np.asarray([item[2] for item in tests]))
    edges = frozenset(
        (source, target)
        for (source, target, _), q_value in zip(tests, q_values, strict=True)
        if q_value <= 0.1
    )
    if not edges:
        raise TaskCTuningError(
            "tuning responses contain no positive relation at the fixed q-value"
        )
    return edges


def _deep_freeze_json(value: object, label: str = "parameters") -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskCTuningError(f"{label} must contain finite JSON values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or key in frozen:
                raise TaskCTuningError(f"{label} must use unique text fields")
            frozen[key] = _deep_freeze_json(nested, label)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item, label) for item in value)
    raise TaskCTuningError(f"{label} must contain only strict JSON values")


def thaw_task_c_json(value: object) -> object:
    """Return ordinary JSON containers from an immutable public result."""

    if isinstance(value, Mapping):
        return {str(key): thaw_task_c_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [thaw_task_c_json(nested) for nested in value]
    return value


def _prediction_table(
    predictions: pd.DataFrame,
    expected_pairs: frozenset[tuple[str, str]] | None,
) -> tuple[np.ndarray, frozenset[tuple[str, str]]]:
    if not isinstance(predictions, pd.DataFrame):
        raise TaskCTuningError("trial predictions must be a table")
    columns = tuple(predictions.columns)
    if columns not in {
        _REQUIRED_PREDICTION_COLUMNS,
        (*_REQUIRED_PREDICTION_COLUMNS, "returned_by_method"),
    }:
        raise TaskCTuningError(
            "trial prediction columns must match the fixed three- or four-column table"
        )
    sources = _canonical_text(predictions["source"].to_numpy(), "prediction sources")
    targets = _canonical_text(predictions["target"].to_numpy(), "prediction targets")
    pairs = tuple(zip(sources, targets, strict=True))
    if any(source == target for source, target in pairs):
        raise TaskCTuningError("trial predictions must not contain self relations")
    if len(set(pairs)) != len(pairs):
        raise TaskCTuningError("trial predictions contain duplicate relations")
    genes = set(sources) | set(targets)
    complete = frozenset(
        (source, target) for source in genes for target in genes if source != target
    )
    observed = frozenset(pairs)
    if observed != complete or expected_pairs is not None and observed != expected_pairs:
        raise TaskCTuningError(
            "every trial must contain the same complete directed relation universe"
        )
    scores_raw = predictions["score"].to_numpy()
    if scores_raw.dtype.kind not in {"i", "u", "f"}:
        raise TaskCTuningError("trial scores must be numeric")
    scores = scores_raw.astype(float, copy=False)
    if not np.isfinite(scores).all():
        raise TaskCTuningError("trial scores must be finite")
    if (scores < 0).any():
        raise TaskCTuningError("trial scores must be non-negative")
    if "returned_by_method" in predictions:
        returned = predictions["returned_by_method"].to_numpy()
        if returned.dtype.kind != "b" or not all(
            isinstance(item, (bool, np.bool_)) for item in returned
        ):
            raise TaskCTuningError("returned_by_method must contain only booleans")
    return scores, observed


def select_task_c_configuration(
    trials: Sequence[tuple[int, Mapping[str, object], pd.DataFrame]],
    *,
    tuning_edges: Iterable[tuple[str, str]],
    maximum_trials: int,
    gene_names: Sequence[str] | None = None,
) -> TaskCTuningSelection:
    """Select the highest tuning AP, then the smallest declared trial index."""

    if (
        isinstance(maximum_trials, bool)
        or not isinstance(maximum_trials, int)
        or maximum_trials != MAXIMUM_TRIALS
        or len(trials) > MAXIMUM_TRIALS
    ):
        raise TaskCTuningError("configuration selection allows at most twenty trials")
    if not trials:
        raise TaskCTuningError("at least one completed tuning trial is required")
    declared_indices = [trial[0] for trial in trials]
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in declared_indices
    ):
        raise TaskCTuningError("trial index must be a non-negative integer")
    if len(set(declared_indices)) != len(declared_indices):
        raise TaskCTuningError("trial indices must be unique")
    raw_edges = tuple(tuning_edges)
    if any(not isinstance(edge, (tuple, list)) or len(edge) != 2 for edge in raw_edges):
        raise TaskCTuningError("tuning edges must contain directed positive relations")
    positives = frozenset((edge[0], edge[1]) for edge in raw_edges)
    if not positives or any(
        len(edge) != 2
        or not all(isinstance(item, str) and item for item in edge)
        or edge[0] == edge[1]
        for edge in positives
    ):
        raise TaskCTuningError("tuning edges must contain directed positive relations")
    evaluated: list[tuple[float, int, Mapping[str, object]]] = []
    indices: set[int] = set()
    universe: frozenset[tuple[str, str]] | None = None
    if gene_names is not None:
        genes = _canonical_text(gene_names, "selection gene names")
        if len(genes) < 2 or len(set(genes)) != len(genes):
            raise TaskCTuningError("selection gene names must be unique")
        universe = frozenset(
            (source, target)
            for source in genes
            for target in genes
            if source != target
        )
    for trial_index, parameters, predictions in trials:
        if (
            isinstance(trial_index, bool)
            or not isinstance(trial_index, int)
            or trial_index < 0
        ):
            raise TaskCTuningError("trial index must be a non-negative integer")
        if trial_index in indices:
            raise TaskCTuningError("trial indices must be unique")
        indices.add(trial_index)
        scores, observed = _prediction_table(predictions, universe)
        universe = observed
        if not positives.issubset(observed):
            raise TaskCTuningError("tuning edges must be a subset of the prediction universe")
        labels = np.asarray(
            [pair in positives for pair in zip(predictions["source"], predictions["target"], strict=True)],
            dtype=int,
        )
        if labels.sum() == 0 or labels.sum() == len(labels):
            raise TaskCTuningError("tuning relation universe needs positives and negatives")
        metric = float(average_precision_score(labels, scores))
        if not math.isfinite(metric):
            raise TaskCTuningError("average precision must be finite")
        frozen = _deep_freeze_json(parameters)
        if not isinstance(frozen, Mapping):
            raise TaskCTuningError("trial parameters must be one JSON object")
        evaluated.append((metric, trial_index, frozen))
    metric, trial_index, parameters = sorted(
        evaluated, key=lambda item: (-item[0], item[1])
    )[0]
    return TaskCTuningSelection(
        schema_version="1.0",
        selected_trial_index=trial_index,
        selected_parameters=parameters,
        average_precision=metric,
        completed_trial_count=len(evaluated),
    )
