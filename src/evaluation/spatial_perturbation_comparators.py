"""Frozen comparators for the spatial-perturbation bridge."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from src.evaluation.spatial_perturbation_scoring import BridgePrediction


MAXIMUM_PARAMETER_COUNT = 10_000_000_000
MAXIMUM_TRAINING_COUNT = 10_000_000
MAXIMUM_SEED = 2**63 - 1
MAXIMUM_COMPARATOR_ROWS = 100_000
MAXIMUM_COMPARATOR_COLUMNS = 6
MAXIMUM_ABSOLUTE_EFFECT = 1.0e12
BRIDGE_EFFECT_UNITS = "train_control_standardized_delta"
BRIDGE_PREDICTION_COLUMNS = (
    "unit_id",
    "endpoint",
    "predicted_effect",
    "effect_units",
)
BRIDGE_EVALUATION_COLUMNS = (
    "unit_id",
    "endpoint",
    "predicted_effect",
    "observed_effect",
    "effect_units",
    "effect_identity_sha256",
)
REQUIRED_BRIDGE_COMPARATORS = (
    "matched_euclidean_spatial_causal",
    "hypersca_own_only",
)
_BRIDGE_COMPARATOR_PROMOTION_ROLES = {
    "matched_euclidean_spatial_causal": "required_iut_confirmatory",
    "hypersca_own_only": "required_iut_attribution",
    "fixed_distance_decay": "secondary_audit_only",
    "without_hierarchy_loss": "secondary_audit_only",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METHOD_GEOMETRY = {
    "hypersca": "hyperbolic",
    "matched_euclidean_spatial_causal": "euclidean",
}


class SpatialPerturbationComparatorError(ValueError):
    """A bridge comparator or its frozen budget is invalid."""


def _text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise SpatialPerturbationComparatorError(
            f"{name} must be bounded non-empty built-in NFC text"
        )
    return value


def _sha(value: object, name: str) -> str:
    text = _text(value, name)
    if _SHA256.fullmatch(text) is None:
        raise SpatialPerturbationComparatorError(
            f"{name} must be a lowercase SHA-256"
        )
    return text


def _positive_integer(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise SpatialPerturbationComparatorError(
            f"{name} must be a bounded positive built-in integer"
        )
    return value


def _seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAXIMUM_SEED:
        raise SpatialPerturbationComparatorError(
            "seed must be a bounded non-negative built-in integer"
        )
    return value


@dataclass(frozen=True, slots=True)
class BridgeModelBudget:
    method_id: str
    geometry: str
    parameter_count: int
    optimizer_family: str
    max_updates: int
    early_stopping_patience: int
    tuning_trials: int
    data_identity_sha256: str
    gene_identity_sha256: str
    spatial_graph_identity_sha256: str
    propagation_identity_sha256: str
    seed: int

    def __post_init__(self) -> None:
        method_id = _text(self.method_id, "method_id")
        geometry = _text(self.geometry, "geometry")
        if method_id not in _METHOD_GEOMETRY:
            raise SpatialPerturbationComparatorError(
                "method_id is not a frozen bridge budget role"
            )
        if geometry != _METHOD_GEOMETRY[method_id]:
            raise SpatialPerturbationComparatorError(
                f"method_id/geometry cannot represent the {method_id} role"
            )
        values: tuple[tuple[str, object], ...] = (
            ("method_id", method_id),
            ("geometry", geometry),
            (
                "parameter_count",
                _positive_integer(
                    self.parameter_count,
                    "parameter_count",
                    MAXIMUM_PARAMETER_COUNT,
                ),
            ),
            ("optimizer_family", _text(self.optimizer_family, "optimizer_family")),
            (
                "max_updates",
                _positive_integer(
                    self.max_updates, "max_updates", MAXIMUM_TRAINING_COUNT
                ),
            ),
            (
                "early_stopping_patience",
                _positive_integer(
                    self.early_stopping_patience,
                    "early_stopping_patience",
                    MAXIMUM_TRAINING_COUNT,
                ),
            ),
            (
                "tuning_trials",
                _positive_integer(
                    self.tuning_trials, "tuning_trials", MAXIMUM_TRAINING_COUNT
                ),
            ),
            (
                "data_identity_sha256",
                _sha(self.data_identity_sha256, "data_identity_sha256"),
            ),
            (
                "gene_identity_sha256",
                _sha(self.gene_identity_sha256, "gene_identity_sha256"),
            ),
            (
                "spatial_graph_identity_sha256",
                _sha(
                    self.spatial_graph_identity_sha256,
                    "spatial_graph_identity_sha256",
                ),
            ),
            (
                "propagation_identity_sha256",
                _sha(
                    self.propagation_identity_sha256,
                    "propagation_identity_sha256",
                ),
            ),
            ("seed", _seed(self.seed)),
        )
        for name, value in values:
            object.__setattr__(self, name, value)


def _budget_snapshot(value: object, name: str) -> BridgeModelBudget:
    if type(value) is not BridgeModelBudget:
        raise SpatialPerturbationComparatorError(
            f"{name} must be BridgeModelBudget"
        )
    budget = value
    return BridgeModelBudget(
        method_id=budget.method_id,
        geometry=budget.geometry,
        parameter_count=budget.parameter_count,
        optimizer_family=budget.optimizer_family,
        max_updates=budget.max_updates,
        early_stopping_patience=budget.early_stopping_patience,
        tuning_trials=budget.tuning_trials,
        data_identity_sha256=budget.data_identity_sha256,
        gene_identity_sha256=budget.gene_identity_sha256,
        spatial_graph_identity_sha256=budget.spatial_graph_identity_sha256,
        propagation_identity_sha256=budget.propagation_identity_sha256,
        seed=budget.seed,
    )


def bridge_model_budget_to_mapping(budget: BridgeModelBudget) -> dict[str, object]:
    """Return a JSON-ready, revalidated snapshot of a model budget."""
    frozen = _budget_snapshot(budget, "budget")
    return {
        "method_id": frozen.method_id,
        "geometry": frozen.geometry,
        "parameter_count": frozen.parameter_count,
        "optimizer_family": frozen.optimizer_family,
        "max_updates": frozen.max_updates,
        "early_stopping_patience": frozen.early_stopping_patience,
        "tuning_trials": frozen.tuning_trials,
        "data_identity_sha256": frozen.data_identity_sha256,
        "gene_identity_sha256": frozen.gene_identity_sha256,
        "spatial_graph_identity_sha256": frozen.spatial_graph_identity_sha256,
        "propagation_identity_sha256": frozen.propagation_identity_sha256,
        "seed": frozen.seed,
    }


def validate_bridge_comparator_budgets(
    hypersca: BridgeModelBudget,
    matched_euclidean: BridgeModelBudget,
) -> None:
    """Require a geometry-only matched comparator within a 5% capacity bound."""
    baseline = _budget_snapshot(hypersca, "hypersca")
    comparator = _budget_snapshot(matched_euclidean, "matched_euclidean")
    if baseline.method_id != "hypersca" or baseline.geometry != "hyperbolic":
        raise SpatialPerturbationComparatorError(
            "hypersca budget must have the frozen hypersca hyperbolic role"
        )
    if (
        comparator.method_id != "matched_euclidean_spatial_causal"
        or comparator.geometry != "euclidean"
    ):
        raise SpatialPerturbationComparatorError(
            "matched_euclidean budget must have its frozen Euclidean role"
        )
    if abs(comparator.parameter_count - baseline.parameter_count) * 100 > (
        baseline.parameter_count * 5
    ):
        raise SpatialPerturbationComparatorError(
            "matched_euclidean parameter_count exceeds the frozen 5% tolerance"
        )
    for field in (
        "optimizer_family",
        "max_updates",
        "early_stopping_patience",
        "tuning_trials",
        "data_identity_sha256",
        "gene_identity_sha256",
        "spatial_graph_identity_sha256",
        "propagation_identity_sha256",
        "seed",
    ):
        if getattr(comparator, field) != getattr(baseline, field):
            raise SpatialPerturbationComparatorError(
                f"matched_euclidean must exactly match {field}"
            )


def bridge_comparator_promotion_role(method_id: str) -> str:
    """Return the frozen promotion role; secondary methods never satisfy IUT."""
    method = _text(method_id, "method_id")
    try:
        return _BRIDGE_COMPARATOR_PROMOTION_ROLES[method]
    except KeyError as error:
        raise SpatialPerturbationComparatorError(
            "method_id is not a frozen bridge comparator"
        ) from error


def validate_required_bridge_comparators(comparator_ids: tuple[str, ...]) -> None:
    """Require both necessary IUT comparators in their preregistered order."""
    if type(comparator_ids) is not tuple:
        raise SpatialPerturbationComparatorError(
            "comparator_ids must be a built-in tuple"
        )
    frozen = tuple(
        _text(value, f"comparator_ids[{index}]")
        for index, value in enumerate(comparator_ids)
    )
    roles = tuple(bridge_comparator_promotion_role(value) for value in frozen)
    if "secondary_audit_only" in roles:
        raise SpatialPerturbationComparatorError(
            "secondary_audit_only comparators cannot satisfy a required comparator"
        )
    if frozen != REQUIRED_BRIDGE_COMPARATORS:
        raise SpatialPerturbationComparatorError(
            "required bridge comparators must use the exact frozen order"
        )


def _prediction_snapshot(value: object, index: int) -> BridgePrediction:
    if type(value) is not BridgePrediction:
        raise SpatialPerturbationComparatorError(
            f"predictions[{index}] must be a Task 7 BridgePrediction"
        )
    prediction = value
    try:
        return BridgePrediction(
            prediction.unit_id,
            prediction.endpoint,
            prediction.predicted_delta,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialPerturbationComparatorError(
            f"predictions[{index}] failed Task 7 revalidation: {error}"
        ) from error


def bridge_predictions_to_comparator_frame(
    predictions: tuple[BridgePrediction, ...],
) -> pd.DataFrame:
    """Map Task 7 prediction fields to the frozen comparator artifact columns.

    The mapping is ``predicted_delta -> predicted_effect``; the unit and
    endpoint fields retain their Task 7 meanings and order.
    """
    if type(predictions) is not tuple:
        raise SpatialPerturbationComparatorError(
            "predictions must be a built-in tuple of Task 7 BridgePrediction values"
        )
    if not predictions:
        raise SpatialPerturbationComparatorError("predictions must not be empty")
    if len(predictions) > MAXIMUM_COMPARATOR_ROWS:
        raise SpatialPerturbationComparatorError(
            "predictions exceed the comparator resource limit"
        )
    frozen = tuple(
        _prediction_snapshot(prediction, index)
        for index, prediction in enumerate(predictions)
    )
    keys = tuple((prediction.endpoint, prediction.unit_id) for prediction in frozen)
    if len(set(keys)) != len(keys):
        raise SpatialPerturbationComparatorError(
            "duplicate Task 7 predictions are forbidden"
        )
    endpoints = {prediction.endpoint for prediction in frozen}
    if "neighbor" not in endpoints:
        raise SpatialPerturbationComparatorError(
            "predictions are missing the neighbor endpoint"
        )
    if "own" not in endpoints:
        raise SpatialPerturbationComparatorError(
            "predictions are missing the own endpoint"
        )
    return pd.DataFrame(
        {
            "unit_id": [prediction.unit_id for prediction in frozen],
            "endpoint": [prediction.endpoint for prediction in frozen],
            "predicted_effect": np.asarray(
                [prediction.predicted_delta for prediction in frozen],
                dtype=np.float64,
            ),
            "effect_units": [BRIDGE_EFFECT_UNITS] * len(frozen),
        },
        columns=BRIDGE_PREDICTION_COLUMNS,
    )


def _validate_prediction_frame(predictions: object) -> pd.DataFrame:
    if type(predictions) is not pd.DataFrame:
        raise SpatialPerturbationComparatorError(
            "predictions must be a built-in pandas DataFrame"
        )
    try:
        row_count, column_count = predictions.shape
    except Exception as error:
        raise SpatialPerturbationComparatorError(
            "prediction artifact axes are unsafe"
        ) from error
    if type(row_count) is not int or type(column_count) is not int:
        raise SpatialPerturbationComparatorError(
            "prediction artifact axes must have built-in integer lengths"
        )
    if column_count > MAXIMUM_COMPARATOR_COLUMNS:
        raise SpatialPerturbationComparatorError(
            "prediction artifact exceeds the column resource limit for columns"
        )
    if column_count < len(BRIDGE_PREDICTION_COLUMNS):
        raise SpatialPerturbationComparatorError(
            "prediction artifact does not have exact frozen columns"
        )
    try:
        raw_columns = predictions.columns.tolist()
    except Exception as error:
        raise SpatialPerturbationComparatorError(
            "prediction artifact column labels are unsafe"
        ) from error
    if len(raw_columns) != column_count:
        raise SpatialPerturbationComparatorError(
            "prediction artifact column axis changed during validation"
        )
    columns_list: list[str] = []
    for index, value in enumerate(raw_columns):
        if type(value) is not str:
            raise SpatialPerturbationComparatorError(
                "prediction artifact column labels must be exact built-in text"
            )
        try:
            columns_list.append(_text(value, f"column_labels[{index}]"))
        except SpatialPerturbationComparatorError as error:
            raise SpatialPerturbationComparatorError(
                "prediction artifact column labels must be canonical safe text"
            ) from error
    columns = tuple(columns_list)
    if columns not in (BRIDGE_PREDICTION_COLUMNS, BRIDGE_EVALUATION_COLUMNS):
        if tuple(sorted(columns)) in (
            tuple(sorted(BRIDGE_PREDICTION_COLUMNS)),
            tuple(sorted(BRIDGE_EVALUATION_COLUMNS)),
        ):
            raise SpatialPerturbationComparatorError(
                "prediction artifact has invalid column order"
            )
        raise SpatialPerturbationComparatorError(
            "prediction artifact does not have exact frozen columns"
        )
    if row_count == 0:
        raise SpatialPerturbationComparatorError(
            "prediction artifact must not be empty"
        )
    if row_count > MAXIMUM_COMPARATOR_ROWS:
        raise SpatialPerturbationComparatorError(
            "prediction artifact exceeds the comparator resource limit"
        )
    frame = predictions.copy(deep=True)
    unit_ids = tuple(
        _sha(value, f"unit_id[{index}]")
        for index, value in enumerate(frame["unit_id"].tolist())
    )
    endpoints = tuple(frame["endpoint"].tolist())
    if any(type(value) is not str or value not in ("neighbor", "own") for value in endpoints):
        raise SpatialPerturbationComparatorError(
            "endpoint must contain only exact neighbor or own labels"
        )
    keys = tuple(zip(endpoints, unit_ids))
    if len(set(keys)) != len(keys):
        raise SpatialPerturbationComparatorError(
            "duplicate endpoint/unit_id predictions are forbidden"
        )
    endpoint_set = set(endpoints)
    if "neighbor" not in endpoint_set:
        raise SpatialPerturbationComparatorError(
            "prediction artifact is missing the neighbor endpoint"
        )
    if "own" not in endpoint_set:
        raise SpatialPerturbationComparatorError(
            "prediction artifact is missing the own endpoint"
        )
    numeric_columns = ["predicted_effect"]
    if columns == BRIDGE_EVALUATION_COLUMNS:
        numeric_columns.append("observed_effect")
    for column in numeric_columns:
        if frame[column].dtype != np.dtype("float64"):
            raise SpatialPerturbationComparatorError(
                f"{column} must have exact float64 units"
            )
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        if (
            not np.isfinite(values).all()
            or (np.abs(values) > MAXIMUM_ABSOLUTE_EFFECT).any()
        ):
            raise SpatialPerturbationComparatorError(
                f"{column} must contain finite bounded effects"
            )
    units = tuple(frame["effect_units"].tolist())
    if any(type(value) is not str or value != BRIDGE_EFFECT_UNITS for value in units):
        raise SpatialPerturbationComparatorError(
            f"effect_units must all equal {BRIDGE_EFFECT_UNITS}"
        )
    if columns == BRIDGE_EVALUATION_COLUMNS:
        for index, value in enumerate(frame["effect_identity_sha256"].tolist()):
            _sha(value, f"effect_identity_sha256[{index}]")
    return frame


def predict_bridge_own_only(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep own-effect predictions and replace neighbour predictions by zero."""
    result = _validate_prediction_frame(predictions)
    neighbor = result["endpoint"].eq("neighbor").to_numpy(dtype=bool)
    values = result["predicted_effect"].to_numpy(dtype=np.float64, copy=True)
    values[neighbor] = np.float64(0.0)
    result["predicted_effect"] = values
    return result


def _real_tuple(
    value: object,
    name: str,
    *,
    expected_length: int,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise SpatialPerturbationComparatorError(
            f"{name} must be a built-in tuple"
        )
    if len(value) != expected_length:
        raise SpatialPerturbationComparatorError(
            f"{name} must match the exact frozen row count"
        )
    frozen: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in (int, float) or type(item) is bool:
            raise SpatialPerturbationComparatorError(
                f"{name}[{index}] must be a built-in real number"
            )
        if type(item) is int and (
            item < -MAXIMUM_ABSOLUTE_EFFECT
            or item > MAXIMUM_ABSOLUTE_EFFECT
        ):
            raise SpatialPerturbationComparatorError(
                f"{name} must contain finite bounded values"
            )
        numeric = float(item)
        if (
            not math.isfinite(numeric)
            or abs(numeric) > MAXIMUM_ABSOLUTE_EFFECT
            or (nonnegative and numeric < 0.0)
        ):
            raise SpatialPerturbationComparatorError(
                f"{name} must contain finite bounded values"
            )
        frozen.append(numeric)
    return tuple(frozen)


def predict_bridge_fixed_distance_decay(
    predictions: pd.DataFrame,
    *,
    distances: tuple[float, ...],
    own_effect_predictions: tuple[float, ...],
    length_scale: int | float,
) -> pd.DataFrame:
    """Build the audit-only fixed-distance baseline on frozen Task 8 rows.

    This is the Task S formula adapted to the strict bridge prediction
    artifact: own rows retain their supplied cell-autonomous prediction and
    neighbour rows use ``own_effect * exp(-distance / length_scale)``.
    """
    frame = _validate_prediction_frame(predictions)
    if type(length_scale) not in (int, float) or type(length_scale) is bool:
        raise SpatialPerturbationComparatorError(
            "length_scale must be a finite positive built-in real number"
        )
    if type(length_scale) is int and (
        length_scale < -MAXIMUM_ABSOLUTE_EFFECT
        or length_scale > MAXIMUM_ABSOLUTE_EFFECT
    ):
        raise SpatialPerturbationComparatorError(
            "length_scale must be a finite positive built-in real number"
        )
    numeric_length_scale = float(length_scale)
    if (
        not math.isfinite(numeric_length_scale)
        or not 0.0 < numeric_length_scale <= MAXIMUM_ABSOLUTE_EFFECT
    ):
        raise SpatialPerturbationComparatorError(
            "length_scale must be a finite positive built-in real number"
        )
    frozen_distances = _real_tuple(
        distances,
        "distances",
        expected_length=len(frame),
        nonnegative=True,
    )
    frozen_own = _real_tuple(
        own_effect_predictions,
        "own_effect_predictions",
        expected_length=len(frame),
    )
    endpoints = tuple(frame["endpoint"].tolist())
    for index, (endpoint, distance) in enumerate(zip(endpoints, frozen_distances)):
        if endpoint == "own" and distance != 0.0:
            raise SpatialPerturbationComparatorError(
                f"own distance must be zero at row {index}"
            )
    predicted = np.asarray(
        [
            own_effect
            if endpoint == "own"
            else own_effect * math.exp(-distance / numeric_length_scale)
            for endpoint, distance, own_effect in zip(
                endpoints, frozen_distances, frozen_own
            )
        ],
        dtype=np.float64,
    )
    if not np.isfinite(predicted).all():
        raise SpatialPerturbationComparatorError(
            "fixed_distance_decay produced non-finite predictions"
        )
    frame["predicted_effect"] = predicted
    return frame


def _column_exactly_equal(left: pd.Series, right: pd.Series) -> bool:
    if left.dtype != right.dtype or len(left) != len(right):
        return False
    if left.dtype == np.dtype("float64"):
        return left.to_numpy(copy=False).tobytes() == right.to_numpy(copy=False).tobytes()
    return left.tolist() == right.tolist()


def validate_bridge_comparator_predictions(
    hypersca: pd.DataFrame,
    matched_euclidean: pd.DataFrame,
    hypersca_own_only: pd.DataFrame,
) -> None:
    """Enforce exact common rows and immutable metadata across required methods."""
    baseline = _validate_prediction_frame(hypersca)
    comparators = (
        (
            "matched_euclidean_spatial_causal",
            _validate_prediction_frame(matched_euclidean),
        ),
        ("hypersca_own_only", _validate_prediction_frame(hypersca_own_only)),
    )
    baseline_keys = tuple(zip(baseline["endpoint"], baseline["unit_id"]))
    for method_id, frame in comparators:
        keys = tuple(zip(frame["endpoint"], frame["unit_id"]))
        if keys != baseline_keys:
            raise SpatialPerturbationComparatorError(
                f"{method_id} must use the exact frozen row order"
            )
        if tuple(frame.columns) != tuple(baseline.columns):
            raise SpatialPerturbationComparatorError(
                f"{method_id} must use the exact frozen columns"
            )
        for column in baseline.columns:
            if column == "predicted_effect":
                continue
            if not _column_exactly_equal(baseline[column], frame[column]):
                raise SpatialPerturbationComparatorError(
                    f"{method_id} must exactly preserve {column}"
                )
    expected_own_only = predict_bridge_own_only(baseline)
    if not _column_exactly_equal(
        expected_own_only["predicted_effect"],
        comparators[1][1]["predicted_effect"],
    ):
        raise SpatialPerturbationComparatorError(
            "hypersca_own_only predictions must retain exact own values and positive neighbor zero"
        )
