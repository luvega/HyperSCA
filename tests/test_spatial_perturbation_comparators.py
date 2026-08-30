from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
import importlib
import json
import math
import tracemalloc
from typing import cast

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
import pytest

import src.evaluation.spatial_perturbation_comparators as comparators
from src.evaluation.spatial_perturbation_comparators import (
    BRIDGE_EFFECT_UNITS,
    BridgeModelBudget,
    SpatialPerturbationComparatorError,
    bridge_comparator_promotion_role,
    bridge_model_budget_to_mapping,
    bridge_predictions_to_comparator_frame,
    predict_bridge_fixed_distance_decay,
    predict_bridge_own_only,
    validate_bridge_comparator_budgets,
    validate_bridge_comparator_predictions,
    validate_required_bridge_comparators,
)
from src.evaluation.spatial_perturbation_scoring import BridgePrediction


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["a" * 64, "b" * 64],
            "endpoint": ["own", "neighbor"],
            "predicted_effect": np.asarray([1.25, -0.75], dtype=np.float64),
            "observed_effect": np.asarray([1.0, -0.5], dtype=np.float64),
            "effect_units": [BRIDGE_EFFECT_UNITS, BRIDGE_EFFECT_UNITS],
            "effect_identity_sha256": ["c" * 64, "d" * 64],
        }
    )


def _budget(
    *,
    method_id: str = "hypersca",
    geometry: str = "hyperbolic",
    parameter_count: int = 100,
    optimizer_family: str = "adamw",
    max_updates: int = 20_000,
    early_stopping_patience: int = 500,
    tuning_trials: int = 20,
    data_identity_sha256: str = "a" * 64,
    gene_identity_sha256: str = "b" * 64,
    spatial_graph_identity_sha256: str = "c" * 64,
    propagation_identity_sha256: str = "d" * 64,
    seed: int = 11,
) -> BridgeModelBudget:
    return BridgeModelBudget(
        method_id=method_id,
        geometry=geometry,
        parameter_count=parameter_count,
        optimizer_family=optimizer_family,
        max_updates=max_updates,
        early_stopping_patience=early_stopping_patience,
        tuning_trials=tuning_trials,
        data_identity_sha256=data_identity_sha256,
        gene_identity_sha256=gene_identity_sha256,
        spatial_graph_identity_sha256=spatial_graph_identity_sha256,
        propagation_identity_sha256=propagation_identity_sha256,
        seed=seed,
    )


def _euclidean_budget(**changes: object) -> BridgeModelBudget:
    values: dict[str, object] = {
        "method_id": "matched_euclidean_spatial_causal",
        "geometry": "euclidean",
    }
    values.update(changes)
    return _budget(**values)  # type: ignore[arg-type]


def _three_predictions() -> pd.DataFrame:
    frame = _predictions()
    third = pd.DataFrame(
        {
            "unit_id": ["e" * 64],
            "endpoint": ["neighbor"],
            "predicted_effect": np.asarray([0.5], dtype=np.float64),
            "observed_effect": np.asarray([0.25], dtype=np.float64),
            "effect_units": [BRIDGE_EFFECT_UNITS],
            "effect_identity_sha256": ["f" * 64],
        }
    )
    return pd.concat([frame, third], ignore_index=True)


def test_own_only_sets_neighbor_predictions_to_positive_zero() -> None:
    frame = predict_bridge_own_only(_predictions())

    values = frame.loc[frame["endpoint"] == "neighbor", "predicted_effect"].to_numpy()
    assert np.equal(values, 0.0).all()
    assert not np.signbit(values).any()


def test_fixed_distance_decay_uses_frozen_own_effect_and_distance_semantics() -> None:
    source = _predictions()
    before = source.copy(deep=True)

    result = predict_bridge_fixed_distance_decay(
        source,
        distances=(0.0, 2.0),
        own_effect_predictions=(1.25, 1.25),
        length_scale=2.0,
    )

    assert result["predicted_effect"].tolist() == pytest.approx(
        [1.25, 1.25 * math.exp(-1.0)]
    )
    for column in source.columns:
        if column != "predicted_effect":
            pd.testing.assert_series_equal(result[column], source[column], check_exact=True)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    assert bridge_comparator_promotion_role("fixed_distance_decay") == (
        "secondary_audit_only"
    )


def test_fixed_distance_decay_matches_task_s_for_zero_distance_neighbors() -> None:
    source = _predictions()
    task_s_module = importlib.import_module("src.evaluation.task_s_benchmark")
    task_s_input = pd.DataFrame(
        {
            "unit_id": source["unit_id"],
            "sample_id": ["sample-a", "sample-a"],
            "spatial_block": ["block-a", "block-a"],
            "perturbation_id": ["KO_A", "KO_A"],
            "feature_id": ["gene-a", "gene-a"],
            "distance": [0.0, 0.0],
            "is_perturbed": [True, False],
            "own_effect_prediction": [1.25, 1.25],
            "observed_effect": source["observed_effect"],
        }
    )
    task_s = task_s_module.predict_task_s_baseline(
        task_s_input,
        baseline_id="fixed_distance_decay",
        length_scale=2.0,
    )

    bridge = predict_bridge_fixed_distance_decay(
        source,
        distances=(0.0, 0.0),
        own_effect_predictions=(1.25, 1.25),
        length_scale=2.0,
    )

    np.testing.assert_array_equal(
        bridge["predicted_effect"].to_numpy(),
        task_s.predictions["predicted_effect"].to_numpy(),
    )


@pytest.mark.parametrize(
    "argument",
    ("distances", "own_effect_predictions", "length_scale"),
)
def test_fixed_distance_decay_wraps_huge_integer_conversion(
    argument: str,
) -> None:
    huge = 10**10000
    kwargs: dict[str, object] = {
        "distances": (0.0, 1.0),
        "own_effect_predictions": (1.25, 1.25),
        "length_scale": 2.0,
    }
    if argument == "distances":
        kwargs[argument] = (0.0, huge)
    elif argument == "own_effect_predictions":
        kwargs[argument] = (1.25, huge)
    else:
        kwargs[argument] = huge

    with pytest.raises(SpatialPerturbationComparatorError, match=argument):
        predict_bridge_fixed_distance_decay(
            _predictions(),
            distances=cast(tuple[float, ...], kwargs["distances"]),
            own_effect_predictions=cast(
                tuple[float, ...], kwargs["own_effect_predictions"]
            ),
            length_scale=cast(float, kwargs["length_scale"]),
        )


@pytest.mark.parametrize(
    "argument",
    ("distances", "own_effect_predictions", "length_scale"),
)
def test_fixed_distance_decay_rejects_negative_huge_int_without_abs_allocation(
    argument: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge_negative = -(10**10000)
    kwargs: dict[str, object] = {
        "distances": (0.0, 1.0),
        "own_effect_predictions": (1.25, 1.25),
        "length_scale": 2.0,
    }
    if argument == "distances":
        kwargs[argument] = (0.0, huge_negative)
    elif argument == "own_effect_predictions":
        kwargs[argument] = (1.25, huge_negative)
    else:
        kwargs[argument] = huge_negative

    def forbidden_abs(value: object) -> object:
        if type(value) is int and value.bit_length() > 4096:  # type: ignore[union-attr]
            raise AssertionError("huge integer passed to abs")
        return builtins.abs(value)  # type: ignore[arg-type]

    monkeypatch.setattr(comparators, "abs", forbidden_abs, raising=False)
    with pytest.raises(SpatialPerturbationComparatorError, match=argument):
        predict_bridge_fixed_distance_decay(
            _predictions(),
            distances=cast(tuple[float, ...], kwargs["distances"]),
            own_effect_predictions=cast(
                tuple[float, ...], kwargs["own_effect_predictions"]
            ),
            length_scale=cast(float, kwargs["length_scale"]),
        )


@pytest.mark.parametrize(
    ("distances", "own_effects", "length_scale", "message"),
    (
        ((0.0,), (1.0, 1.0), 1.0, "distances"),
        ((0.0, 1.0), (1.0,), 1.0, "own_effect_predictions"),
        ((1.0, 2.0), (1.0, 1.0), 1.0, "own distance"),
        ((0.0, float("inf")), (1.0, 1.0), 1.0, "distances"),
        ((0.0, 1.0), (1.0, float("nan")), 1.0, "own_effect_predictions"),
        ((0.0, 1.0), (1.0, 1.0), 0.0, "length_scale"),
        ((0.0, 1.0), (1.0, 1.0), float("inf"), "length_scale"),
        ((0.0, 1.0), (1.0, 1.0), True, "length_scale"),
    ),
)
def test_fixed_distance_decay_fails_closed(
    distances: tuple[float, ...],
    own_effects: tuple[float, ...],
    length_scale: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predict_bridge_fixed_distance_decay(
            _predictions(),
            distances=distances,
            own_effect_predictions=own_effects,
            length_scale=length_scale,  # type: ignore[arg-type]
        )


def test_task7_predictions_have_one_unambiguous_comparator_column_mapping() -> None:
    source = (
        BridgePrediction("b" * 64, "neighbor", -0.75),
        BridgePrediction("a" * 64, "own", 1.25),
    )

    frame = bridge_predictions_to_comparator_frame(source)

    assert tuple(frame.columns) == (
        "unit_id",
        "endpoint",
        "predicted_effect",
        "effect_units",
    )
    assert frame.to_dict(orient="records") == [
        {
            "unit_id": source[0].unit_id,
            "endpoint": source[0].endpoint,
            "predicted_effect": source[0].predicted_delta,
            "effect_units": BRIDGE_EFFECT_UNITS,
        },
        {
            "unit_id": source[1].unit_id,
            "endpoint": source[1].endpoint,
            "predicted_effect": source[1].predicted_delta,
            "effect_units": BRIDGE_EFFECT_UNITS,
        },
    ]
    assert frame["predicted_effect"].dtype == np.dtype("float64")


def test_own_only_preserves_own_values_order_observations_and_identities_exactly() -> None:
    source = _predictions().iloc[[1, 0]].copy()
    before = source.copy(deep=True)

    result = predict_bridge_own_only(source)

    pd.testing.assert_frame_equal(source, before, check_exact=True)
    assert result.index.tolist() == source.index.tolist()
    assert result["unit_id"].tolist() == source["unit_id"].tolist()
    assert result["endpoint"].tolist() == source["endpoint"].tolist()
    assert result["observed_effect"].tolist() == source["observed_effect"].tolist()
    assert result["effect_units"].tolist() == source["effect_units"].tolist()
    assert result["effect_identity_sha256"].tolist() == source[
        "effect_identity_sha256"
    ].tolist()
    own = result["endpoint"].eq("own")
    assert result.loc[own, "predicted_effect"].tolist() == source.loc[
        own, "predicted_effect"
    ].tolist()
    result.loc[:, "observed_effect"] = 999.0
    pd.testing.assert_frame_equal(source, before, check_exact=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda frame: frame.drop(columns=["effect_units"]), "columns"),
        (lambda frame: frame.assign(extra="unsafe"), "columns"),
        (
            lambda frame: frame[
                [
                    "endpoint",
                    "unit_id",
                    "predicted_effect",
                    "observed_effect",
                    "effect_units",
                    "effect_identity_sha256",
                ]
            ],
            "column order",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (lambda frame: frame.iloc[[0]].copy(), "neighbor"),
        (lambda frame: frame.iloc[[1]].copy(), "own"),
        (
            lambda frame: frame.assign(endpoint=["own", "transition"]),
            "endpoint",
        ),
        (
            lambda frame: frame.assign(effect_units=[BRIDGE_EFFECT_UNITS, "raw"]),
            "effect_units",
        ),
        (
            lambda frame: frame.assign(predicted_effect=[1.25, float("nan")]),
            "predicted_effect",
        ),
        (
            lambda frame: frame.assign(observed_effect=[1.0, float("inf")]),
            "observed_effect",
        ),
        (
            lambda frame: frame.assign(predicted_effect=[1.25, 10.0**13]),
            "predicted_effect",
        ),
        (
            lambda frame: frame.assign(effect_identity_sha256=["c" * 64, "bad"]),
            "effect_identity_sha256",
        ),
    ),
)
def test_own_only_fails_closed_on_malformed_prediction_artifacts(
    mutation: object, message: str
) -> None:
    assert callable(mutation)
    with pytest.raises(ValueError, match=message):
        predict_bridge_own_only(mutation(_predictions()))


def test_own_only_rejects_non_float_effect_columns() -> None:
    for column in ("predicted_effect", "observed_effect"):
        frame = _predictions()
        frame[column] = [1, -1]
        with pytest.raises(ValueError, match=column):
            predict_bridge_own_only(frame)


def test_own_only_enforces_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparators, "MAXIMUM_COMPARATOR_ROWS", 1)

    with pytest.raises(ValueError, match="resource limit"):
        predict_bridge_own_only(_predictions())


def test_prediction_frame_rejects_500k_empty_columns_before_materialization() -> None:
    hostile = pd.DataFrame(
        np.empty((0, 500_000), dtype=np.float64),
        columns=pd.RangeIndex(500_000),
    )
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="column resource limit"):
            predict_bridge_own_only(hostile)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * 1024 * 1024


@pytest.mark.parametrize(
    "label",
    (
        type("TextSubclass", (str,), {})("unit_id"),
        "unit_id\n",
        "u\u0301nit_id",
        1,
    ),
)
def test_prediction_frame_rejects_non_exact_or_noncanonical_column_labels(
    label: object,
) -> None:
    frame = _predictions()
    columns: list[object] = list(frame.columns)
    columns[0] = label
    frame.columns = cast("pd.Index[str]", pd.Index(columns, dtype=object))

    with pytest.raises(ValueError, match="column labels"):
        predict_bridge_own_only(frame)


def test_prediction_frame_rejects_dataframe_subclasses() -> None:
    class HostileFrame(pd.DataFrame):
        pass

    with pytest.raises(ValueError, match="built-in pandas DataFrame"):
        predict_bridge_own_only(HostileFrame(_predictions()))


def test_prediction_frame_checks_label_type_before_hostile_hashing() -> None:
    class HostileText(str):
        def __hash__(self) -> int:
            raise AssertionError("hostile column label was hashed")

    frame = _predictions()
    columns: list[object] = list(frame.columns)
    columns[0] = HostileText("unit_id")
    frame.columns = cast("pd.Index[str]", pd.Index(columns, dtype=object))

    with pytest.raises(ValueError, match="column labels"):
        predict_bridge_own_only(frame)


def test_adapter_revalidates_mutated_task7_predictions() -> None:
    prediction = BridgePrediction("a" * 64, "own", 1.0)
    object.__setattr__(prediction, "predicted_delta", float("nan"))

    with pytest.raises(ValueError, match="predicted_delta"):
        bridge_predictions_to_comparator_frame((prediction,))


def test_task7_adapter_enforces_resource_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = (
        BridgePrediction("a" * 64, "own", 1.0),
        BridgePrediction("b" * 64, "neighbor", 0.5),
    )
    monkeypatch.setattr(comparators, "MAXIMUM_COMPARATOR_ROWS", 1)

    with pytest.raises(ValueError, match="resource limit"):
        bridge_predictions_to_comparator_frame(predictions)


def test_task7_adapter_rejects_hostile_containers_and_prediction_subclasses() -> None:
    class PredictionSubclass(BridgePrediction):
        pass

    prediction = PredictionSubclass("a" * 64, "own", 1.0)
    with pytest.raises(ValueError, match="built-in tuple"):
        bridge_predictions_to_comparator_frame(
            cast(tuple[BridgePrediction, ...], [prediction])
        )
    with pytest.raises(ValueError, match="Task 7 BridgePrediction"):
        bridge_predictions_to_comparator_frame((prediction,))


def test_own_only_is_permutation_deterministic_and_does_not_use_rng() -> None:
    source = _predictions()
    permuted = source.iloc[[1, 0]].copy()
    np.random.seed(8675309)
    before = repr(np.random.get_state())

    original_result = predict_bridge_own_only(source)
    permuted_result = predict_bridge_own_only(permuted)
    after = repr(np.random.get_state())

    pd.testing.assert_frame_equal(
        original_result,
        permuted_result.iloc[[1, 0]],
        check_exact=True,
    )
    assert before == after


def test_bridge_comparator_promotion_roles_are_frozen() -> None:
    assert bridge_comparator_promotion_role("matched_euclidean_spatial_causal") == (
        "required_iut_confirmatory"
    )
    assert bridge_comparator_promotion_role("hypersca_own_only") == (
        "required_iut_attribution"
    )
    assert bridge_comparator_promotion_role("fixed_distance_decay") == (
        "secondary_audit_only"
    )
    assert bridge_comparator_promotion_role("without_hierarchy_loss") == (
        "secondary_audit_only"
    )


def test_fixed_distance_decay_cannot_satisfy_a_required_comparator() -> None:
    with pytest.raises(ValueError, match="secondary_audit_only"):
        validate_required_bridge_comparators(
            ("matched_euclidean_spatial_causal", "fixed_distance_decay")
        )


def test_required_bridge_comparator_order_is_exact() -> None:
    validate_required_bridge_comparators(
        ("matched_euclidean_spatial_causal", "hypersca_own_only")
    )
    with pytest.raises(ValueError, match="exact frozen order"):
        validate_required_bridge_comparators(
            ("hypersca_own_only", "matched_euclidean_spatial_causal")
        )
    with pytest.raises(ValueError, match="built-in tuple"):
        validate_required_bridge_comparators(
            cast(
                tuple[str, ...],
                ["matched_euclidean_spatial_causal", "hypersca_own_only"],
            )
        )


def test_all_required_methods_share_exact_frozen_rows_and_metadata() -> None:
    hypersca = _three_predictions()
    euclidean = hypersca.copy(deep=True)
    euclidean["predicted_effect"] = np.asarray([2.0, -1.0, 0.75], dtype=np.float64)
    own_only = predict_bridge_own_only(hypersca)

    validate_bridge_comparator_predictions(hypersca, euclidean, own_only)


@pytest.mark.parametrize("mutation", ("deletion", "extra", "permutation"))
def test_method_specific_row_changes_are_forbidden(mutation: str) -> None:
    hypersca = _three_predictions()
    euclidean = hypersca.copy(deep=True)
    own_only = predict_bridge_own_only(hypersca)
    if mutation == "deletion":
        euclidean = euclidean.iloc[:2].copy()
    elif mutation == "extra":
        extra = euclidean.iloc[[0]].copy()
        extra["unit_id"] = "9" * 64
        extra["effect_identity_sha256"] = "8" * 64
        euclidean = pd.concat([euclidean, extra], ignore_index=True)
    else:
        euclidean = euclidean.iloc[[1, 0, 2]].copy()

    with pytest.raises(ValueError, match="exact frozen row order"):
        validate_bridge_comparator_predictions(hypersca, euclidean, own_only)


def test_all_required_methods_must_share_units_and_observed_identity_metadata() -> None:
    hypersca = _three_predictions()
    euclidean = hypersca.copy(deep=True)
    own_only = predict_bridge_own_only(hypersca)
    euclidean["observed_effect"] = (
        euclidean["observed_effect"].to_numpy(dtype=np.float64)
        + np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="observed_effect"):
        validate_bridge_comparator_predictions(hypersca, euclidean, own_only)


def test_own_only_artifact_must_be_derived_from_exact_hypersca_predictions() -> None:
    hypersca = _three_predictions()
    euclidean = hypersca.copy(deep=True)
    own_only = predict_bridge_own_only(hypersca)
    own_only["predicted_effect"] = (
        own_only["predicted_effect"].to_numpy(dtype=np.float64)
        + np.asarray([0.01, 0.0, 0.0], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="hypersca_own_only"):
        validate_bridge_comparator_predictions(hypersca, euclidean, own_only)


def test_budget_is_immutable_and_serializes_to_json_ready_builtins() -> None:
    budget = _budget()

    with pytest.raises(FrozenInstanceError):
        budget.seed = 12  # type: ignore[misc]
    payload = bridge_model_budget_to_mapping(budget)
    assert tuple(payload) == (
        "method_id",
        "geometry",
        "parameter_count",
        "optimizer_family",
        "max_updates",
        "early_stopping_patience",
        "tuning_trials",
        "data_identity_sha256",
        "gene_identity_sha256",
        "spatial_graph_identity_sha256",
        "propagation_identity_sha256",
        "seed",
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("parameter_count", True, "parameter_count"),
        ("parameter_count", 0, "parameter_count"),
        ("parameter_count", -0, "parameter_count"),
        ("parameter_count", 10**100, "parameter_count"),
        ("max_updates", float("inf"), "max_updates"),
        ("early_stopping_patience", float("nan"), "early_stopping_patience"),
        ("tuning_trials", False, "tuning_trials"),
        ("seed", True, "seed"),
        ("seed", -1, "seed"),
        ("seed", 10**100, "seed"),
        ("data_identity_sha256", "A" * 64, "data_identity_sha256"),
        ("gene_identity_sha256", "g" * 64, "gene_identity_sha256"),
        ("spatial_graph_identity_sha256", "c" * 63, "spatial_graph_identity_sha256"),
        ("propagation_identity_sha256", "d" * 65, "propagation_identity_sha256"),
    ),
)
def test_budget_rejects_non_exact_or_unbounded_values(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _budget(**{field: value})  # type: ignore[arg-type]


def test_budget_rejects_integer_subclasses() -> None:
    class CustomInt(int):
        pass

    with pytest.raises(ValueError, match="parameter_count"):
        _budget(parameter_count=CustomInt(100))


@pytest.mark.parametrize(
    "parameter_count",
    (95, 105),
)
def test_euclidean_budget_accepts_exact_five_percent_parameter_boundaries(
    parameter_count: int,
) -> None:
    validate_bridge_comparator_budgets(
        _budget(), _euclidean_budget(parameter_count=parameter_count)
    )


@pytest.mark.parametrize("parameter_count", (94, 106))
def test_euclidean_budget_rejects_parameters_beyond_five_percent(
    parameter_count: int,
) -> None:
    with pytest.raises(ValueError, match="parameter_count"):
        validate_bridge_comparator_budgets(
            _budget(), _euclidean_budget(parameter_count=parameter_count)
        )


def test_parameter_ratio_uses_exact_integer_arithmetic_mutation_counterexample() -> None:
    with pytest.raises(ValueError, match="parameter_count"):
        validate_bridge_comparator_budgets(
            _budget(parameter_count=10_000),
            _euclidean_budget(parameter_count=10_505),
        )


@settings(max_examples=100, deadline=None)
@given(
    hypersca_count=st.integers(min_value=1, max_value=10_000_000_000),
    euclidean_count=st.integers(min_value=1, max_value=10_000_000_000),
)
def test_parameter_ratio_matches_exact_five_percent_integer_inequality(
    hypersca_count: int,
    euclidean_count: int,
) -> None:
    within_tolerance = abs(euclidean_count - hypersca_count) * 100 <= (
        hypersca_count * 5
    )
    if within_tolerance:
        validate_bridge_comparator_budgets(
            _budget(parameter_count=hypersca_count),
            _euclidean_budget(parameter_count=euclidean_count),
        )
    else:
        with pytest.raises(ValueError, match="parameter_count"):
            validate_bridge_comparator_budgets(
                _budget(parameter_count=hypersca_count),
                _euclidean_budget(parameter_count=euclidean_count),
            )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("optimizer_family", "sgd"),
        ("max_updates", 19_999),
        ("early_stopping_patience", 499),
        ("tuning_trials", 19),
        ("data_identity_sha256", "e" * 64),
        ("gene_identity_sha256", "e" * 64),
        ("spatial_graph_identity_sha256", "e" * 64),
        ("propagation_identity_sha256", "e" * 64),
        ("seed", 12),
    ),
)
def test_euclidean_budget_rejects_every_non_geometry_difference(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_bridge_comparator_budgets(
            _budget(), _euclidean_budget(**{field: value})
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"method_id": "matched_euclidean_spatial_causal"}, "geometry"),
        ({"geometry": "euclidean"}, "method_id"),
        ({"method_id": "unknown", "geometry": "hyperbolic"}, "method_id"),
        ({"method_id": "hypersca", "geometry": "spherical"}, "geometry"),
    ),
)
def test_budget_roles_cannot_be_spoofed(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _budget(**changes)  # type: ignore[arg-type]


def test_validator_rechecks_direct_construction_after_identity_mutation() -> None:
    euclidean = _euclidean_budget()
    object.__setattr__(euclidean, "spatial_graph_identity_sha256", "unsafe")

    with pytest.raises(ValueError, match="spatial_graph_identity_sha256"):
        validate_bridge_comparator_budgets(_budget(), euclidean)
    with pytest.raises(ValueError, match="spatial_graph_identity_sha256"):
        bridge_model_budget_to_mapping(euclidean)
