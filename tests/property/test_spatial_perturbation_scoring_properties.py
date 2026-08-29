"""Hypothesis properties for spatial perturbation scoring."""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from src.evaluation.spatial_perturbation_scoring import (
    BridgePrediction,
    SpatialPerturbationScoringError,
    apply_train_control_standardizer,
    bridge_score_to_mapping,
    build_bridge_effect_table,
    fit_train_control_standardizer,
    score_bridge_predictions,
)
from tests.test_spatial_perturbation_scoring import _bridge_case
from tests.test_spatial_perturbation_scoring import (
    _evaluation_predictions,
    _expression_for_eligibility,
    _metadata_with_referenced_section_duplication,
    _standardizer_for_manifest,
)
from tests.test_spatial_perturbation_split import baseline, complete_evidence
from src.evaluation.spatial_perturbation_split import (
    build_pilot_fold,
    evaluate_bridge_eligibility,
)


@given(st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=5, deadline=None)
def test_nondegenerate_train_standardization_is_invariant_to_log_scale(
    multiplier: float,
) -> None:
    """The invariance claim is restricted to scales above the fallback floor."""
    from tests.test_spatial_perturbation_scoring import _manifest_training_case

    manifest, expression, cell_ids, controls = _manifest_training_case()
    logged_scale = np.log1p(expression[list(controls)]).std(axis=0, ddof=0)
    assert bool(np.all(logged_scale > 2e-6))
    assert bool(np.all(logged_scale * multiplier > 2e-6))
    first = fit_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    first_values = apply_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        standardizer=first,
        split_manifest=manifest,
    )

    log_scaled = np.expm1(np.log1p(expression) * multiplier)
    second = fit_train_control_standardizer(
        log_scaled,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    second_values = apply_train_control_standardizer(
        log_scaled,
        gene_names=manifest.gene_names,
        standardizer=second,
        split_manifest=manifest,
    )
    np.testing.assert_allclose(first_values, second_values, rtol=1e-10, atol=1e-10)


@given(st.permutations((0, 1, 2, 3, 4, 5)))
@settings(max_examples=5, deadline=None)
def test_general_prediction_permutation_does_not_change_score(
    permutation: tuple[int, ...]
) -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    chunks = tuple(predictions[index::6] for index in range(6))
    permuted = tuple(item for index in permutation for item in chunks[index])
    first_score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    second_score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=permuted,
    )
    first_mapping = bridge_score_to_mapping(
        first_score,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    second_mapping = bridge_score_to_mapping(
        second_score,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=permuted,
    )
    assert first_mapping == second_mapping


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=5, deadline=None)
def test_unreferenced_cell_or_section_duplication_does_not_change_animal_metric(
    copies: int,
) -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    first = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    duplicated_expression = np.vstack([expression] + [expression[:1]] * copies)
    duplicated_cell_ids = cell_ids + tuple(
        f"unreferenced_copy_{index}" for index in range(copies)
    )
    second = score_bridge_predictions(
        duplicated_expression,
        cell_ids=duplicated_cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    assert first.neighbor_effect_rmse == second.neighbor_effect_rmse
    assert first.animal_level_unit_table == second.animal_level_unit_table
    assert len(second.animal_level_unit_table) == len(first.animal_level_unit_table)


@given(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=5, deadline=None)
def test_finite_bounded_predictions_produce_finite_metrics(delta: float) -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    changed = tuple(
        BridgePrediction(item.unit_id, item.endpoint, float(delta))
        for item in predictions
    )
    result = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=changed,
    )
    assert all(
        math.isfinite(value)
        for value in (
            result.neighbor_effect_rmse,
            result.own_effect_rmse,
            result.neighbor_effect_pcc,
            result.distance_decay_calibration_error,
            result.effect_sign_accuracy,
            result.coverage,
            result.abstention,
        )
    )
    assert -1.0 <= result.neighbor_effect_pcc <= 1.0
    assert 0.0 <= result.effect_sign_accuracy <= 1.0
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.abstention <= 1.0


@given(st.sampled_from([0.0, 2.0, -2.0]))
@settings(max_examples=3, deadline=None)
def test_constant_and_tied_pcc_is_defined(predicted: float) -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    changed = tuple(
        BridgePrediction(item.unit_id, item.endpoint, predicted)
        for item in predictions
    )
    score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=changed,
    )
    assert score.neighbor_effect_pcc == pytest.approx(0.0)


def test_same_frozen_units_are_required_for_every_method() -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    first = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    second = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=tuple(
            BridgePrediction(item.unit_id, item.endpoint, 0.0)
            for item in reversed(predictions)
        ),
    )
    assert tuple(item.unit_id for item in first.effects) == tuple(
        item.unit_id for item in second.effects
    )
    assert first.split_identity_sha256 == second.split_identity_sha256
    assert first.eligibility_identity_sha256 == second.eligibility_identity_sha256


def test_referenced_section_duplication_rebuilds_artifacts_without_changing_hierarchy() -> None:
    base_manifest, base_evidence = baseline()
    base_eligibility = evaluate_bridge_eligibility(base_manifest, base_evidence)
    duplicated_manifest = build_pilot_fold(
        _metadata_with_referenced_section_duplication(), "mouse_1"
    )
    duplicated_unit = next(
        unit
        for unit in duplicated_manifest.primary_units
        if unit.animal_id == "mouse_1"
        and unit.perturbation_id == "guide_0"
        and unit.neighbour_cell_type == "astrocyte"
        and unit.band == "proximal"
    )
    duplicated_context_units = tuple(
        unit
        for unit in duplicated_manifest.primary_units
        if unit.animal_id == "mouse_1"
        and unit.neighbour_cell_type == "astrocyte"
        and unit.band == "proximal"
    )
    duplicated_evidence = complete_evidence(
        duplicated_manifest,
        unit_overrides={unit.unit_id: 60 for unit in duplicated_context_units},
        safe_unit_overrides={unit.unit_id: 60 for unit in duplicated_context_units},
    )
    duplicated_eligibility = evaluate_bridge_eligibility(
        duplicated_manifest, duplicated_evidence
    )
    assert duplicated_eligibility.eligible is True
    duplicated_unit_evidence = next(
        item
        for item in duplicated_evidence.unit_evidence
        if item.unit_id == duplicated_unit.unit_id
    )
    relation_by_id = {
        item.relation_id: item for item in duplicated_manifest.neighbour_relations
    }
    assert {
        relation_by_id[item].section_id
        for item in duplicated_unit_evidence.perturbation_neighbour_relation_ids
    } == {"mouse_1_section", "mouse_1_section_duplicate"}

    def predictions(eligibility: object) -> tuple[BridgePrediction, ...]:
        exact = _evaluation_predictions(eligibility)  # type: ignore[arg-type]
        return tuple(
            BridgePrediction(
                item.unit_id,
                item.endpoint,
                12.0
                if item.endpoint == "neighbor" and item.unit_id == duplicated_unit.unit_id
                else (3.0 if item.endpoint == "neighbor" else 2.0),
            )
            for item in exact
        )

    base_expression, base_cells = _expression_for_eligibility(base_eligibility)
    duplicate_expression, duplicate_cells = _expression_for_eligibility(
        duplicated_eligibility
    )
    base_standardizer = _standardizer_for_manifest(base_manifest)
    duplicate_standardizer = _standardizer_for_manifest(duplicated_manifest)
    base_predictions = predictions(base_eligibility)
    duplicate_predictions = predictions(duplicated_eligibility)
    base_score = score_bridge_predictions(
        base_expression,
        cell_ids=base_cells,
        gene_names=base_manifest.gene_names,
        standardizer=base_standardizer,
        eligibility=base_eligibility,
        predictions=base_predictions,
    )
    duplicate_score = score_bridge_predictions(
        duplicate_expression,
        cell_ids=duplicate_cells,
        gene_names=duplicated_manifest.gene_names,
        standardizer=duplicate_standardizer,
        eligibility=duplicated_eligibility,
        predictions=duplicate_predictions,
    )
    assert duplicate_score.neighbor_effect_rmse == pytest.approx(
        base_score.neighbor_effect_rmse
    )
    assert len(base_score.animal_level_unit_table) == 1
    assert len(duplicate_score.animal_level_unit_table) == 1

    base_table = build_bridge_effect_table(
        base_expression,
        cell_ids=base_cells,
        gene_names=base_manifest.gene_names,
        standardizer=base_standardizer,
        eligibility=base_eligibility,
        predictions=base_predictions,
    )
    duplicate_table = build_bridge_effect_table(
        duplicate_expression,
        cell_ids=duplicate_cells,
        gene_names=duplicated_manifest.gene_names,
        standardizer=duplicate_standardizer,
        eligibility=duplicated_eligibility,
        predictions=duplicate_predictions,
    )

    def raw_relation_weighted_rmse(table: object) -> float:
        effects = {
            item.unit_id: item
            for item in table.effects  # type: ignore[attr-defined]
            if item.endpoint == "neighbor"
        }
        evidence = {
            item.unit_id: item
            for item in table.eligibility.evidence.unit_evidence  # type: ignore[attr-defined]
        }
        weighted = [
            (
                len(evidence[unit_id].perturbation_neighbour_relation_ids),
                (effect.predicted_delta - effect.observed_delta) ** 2,
            )
            for unit_id, effect in effects.items()
        ]
        return math.sqrt(
            sum(weight * error for weight, error in weighted)
            / sum(weight for weight, _ in weighted)
        )

    # Mutation proof: a forbidden raw-relation-weighted scorer changes.
    assert raw_relation_weighted_rmse(duplicate_table) != pytest.approx(
        raw_relation_weighted_rmse(base_table)
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_prediction_is_rejected(bad: float) -> None:
    with pytest.raises(SpatialPerturbationScoringError, match="finite"):
        BridgePrediction("0" * 64, "neighbor", bad)
