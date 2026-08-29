"""Contract tests for train-only spatial perturbation scoring."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
import json
import math
import tracemalloc

import numpy as np
import pytest

from src.evaluation.spatial_perturbation_scoring import (
    BridgeEffect,
    BridgeEffectTable,
    BridgePrediction,
    SpatialPerturbationScoringError,
    TrainControlStandardizer,
    apply_train_control_standardizer,
    bridge_effect_table_to_mapping,
    bridge_score_to_mapping,
    build_bridge_effect_table,
    fit_train_control_standardizer,
    score_bridge_predictions,
    train_control_standardizer_to_mapping,
)
from src.evaluation.spatial_perturbation_split import (
    BridgeBlockAdjacency,
    BridgeEligibilityResult,
    BridgeSplitMetadata,
    BridgeSplitRow,
    BridgeSplitManifest,
    eligibility_result_to_mapping,
    evaluate_bridge_eligibility,
    freeze_bridge_neighbour_relation,
    freeze_bridge_neighbour_table,
)
from src.evaluation.spatial_perturbation_registry import audit_bridge_capability
from tests.test_spatial_perturbation_split import (
    baseline,
    complete_evidence,
    synthetic_metadata,
)


def _manifest_training_case(
    manifest: BridgeSplitManifest | None = None,
) -> tuple[
    BridgeSplitManifest,
    np.ndarray,
    tuple[str, ...],
    tuple[int, ...],
]:
    if manifest is None:
        manifest, _ = baseline()
    train_ids = set(manifest.train_rows)
    selected_rows = tuple(
        row
        for row in manifest.row_provenance
        if row.stable_row_id in train_ids
        and row.cell_role == "safe_source"
        and row.observed_label == manifest.safe_control_label
    )
    cell_ids = tuple(row.cell_id for row in selected_rows)
    controls = tuple(range(len(selected_rows)))
    expression = np.zeros(
        (len(selected_rows), len(manifest.gene_names)), dtype=np.float64
    )
    for offset, row_index in enumerate(controls):
        expression[row_index, :] = 0.0 if offset % 2 == 0 else math.expm1(1.0)
    return manifest, expression, cell_ids, controls


def _fit_manifest_standardizer() -> TrainControlStandardizer:
    manifest, expression, cell_ids, controls = _manifest_training_case()
    return fit_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )


def _expression_for_eligibility(
    eligibility: BridgeEligibilityResult,
) -> tuple[np.ndarray, tuple[str, ...]]:
    manifest = eligibility.manifest
    evidence = eligibility.evidence
    cell_ids = tuple(sorted({
        relation.neighbor_cell_id for relation in manifest.neighbour_relations
    } | {
        cell_id
        for parent in evidence.parent_evidence
        for cell_id in (
            *parent.perturbation_source_cell_ids,
            *parent.safe_source_cell_ids,
        )
    }))
    row_by_cell = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    gene_by_name = {gene: index for index, gene in enumerate(manifest.gene_names)}
    expression = np.zeros((len(cell_ids), len(manifest.gene_names)), dtype=np.float64)
    high = math.expm1(1.0)
    relation_by_id = {
        relation.relation_id: relation for relation in manifest.neighbour_relations
    }
    unit_by_id = {unit.unit_id: unit for unit in manifest.primary_units}
    for unit_evidence in evidence.unit_evidence:
        unit = unit_by_id[unit_evidence.unit_id]
        column = gene_by_name[unit.target_gene]
        for relation_id in unit_evidence.perturbation_neighbour_relation_ids:
            cell_id = relation_by_id[relation_id].neighbor_cell_id
            expression[row_by_cell[cell_id], column] = high
    parent_by_context = {
        (parent.animal_id, parent.perturbation_id): parent
        for parent in manifest.perturbation_parents
    }
    for parent_evidence in evidence.parent_evidence:
        parent = parent_by_context[
            (parent_evidence.animal_id, parent_evidence.perturbation_id)
        ]
        column = gene_by_name[parent.target_gene]
        for cell_id in parent_evidence.perturbation_source_cell_ids:
            expression[row_by_cell[cell_id], column] = high
    return expression, cell_ids


def _standardizer_for_manifest(
    manifest: BridgeSplitManifest,
) -> TrainControlStandardizer:
    _, training, training_cells, controls = _manifest_training_case(manifest)
    return fit_train_control_standardizer(
        training,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=training_cells,
        split_manifest=manifest,
    )


def _metadata_with_referenced_section_duplication() -> BridgeSplitMetadata:
    metadata = synthetic_metadata()
    duplicate_section = "mouse_1_section_duplicate"
    selected_relations = tuple(
        relation
        for relation in metadata.neighbour_relations
        if relation.animal_id == "mouse_1"
        and relation.neighbor_cell_type == "astrocyte"
        and relation.band == "proximal"
    )
    selected_cell_ids = {
        cell_id
        for relation in selected_relations
        for cell_id in (relation.source_cell_id, relation.neighbor_cell_id)
    }
    next_row = max(row.stable_row_id for row in metadata.rows) + 1
    cell_mapping = {
        cell_id: f"{cell_id}_section_copy" for cell_id in selected_cell_ids
    }
    cloned_rows = tuple(
        BridgeSplitRow(
            next_row + offset,
            cell_mapping[row.cell_id],
            row.animal_id,
            duplicate_section,
            row.spatial_block,
            row.context_perturbation_id,
            row.observed_label,
            row.cell_type,
            row.source_cell_type,
            row.cell_role,
            row.distance_band,
        )
        for offset, row in enumerate(metadata.rows)
        if row.cell_id in selected_cell_ids
    )
    cloned_relations = tuple(
        freeze_bridge_neighbour_relation(
            relation.animal_id,
            duplicate_section,
            relation.spatial_block,
            cell_mapping[relation.source_cell_id],
            cell_mapping[relation.neighbor_cell_id],
            relation.source_perturbation_id,
            relation.source_cell_type,
            relation.neighbor_cell_type,
            relation.rank,
            relation.band,
            relation.is_safe_control,
        )
        for relation in selected_relations
    )
    cloned_adjacency = tuple(
        BridgeBlockAdjacency(
            item.animal_id,
            duplicate_section,
            item.first_block,
            item.second_block,
            item.adjacent,
        )
        for item in metadata.block_adjacency
        if item.animal_id == "mouse_1"
        and item.section_id == "mouse_1_section"
    )
    rows = metadata.rows + cloned_rows
    table = freeze_bridge_neighbour_table(
        metadata.neighbour_relations + cloned_relations
    )
    sections = tuple(
        (
            animal,
            section_ids + ((duplicate_section,) if animal == "mouse_1" else ()),
        )
        for animal, section_ids in metadata.candidate.sections_by_specimen
    )
    candidate = replace(metadata.candidate, sections_by_specimen=sections)
    animals = candidate.biological_specimens
    coordinate_counts = tuple(
        (animal, sum(row.animal_id == animal for row in rows)) for animal in animals
    )
    perturbation_counts = tuple(
        (
            animal,
            sum(
                row.animal_id == animal
                and row.observed_label in metadata.perturbations
                for row in rows
            ),
        )
        for animal in animals
    )
    safe_counts = tuple(
        (
            animal,
            sum(
                row.animal_id == animal
                and row.observed_label == metadata.safe_control_label
                for row in rows
            ),
        )
        for animal in animals
    )
    summary = replace(
        metadata.registry_summary,
        sections_by_specimen=sections,
        coordinate_count=len(rows),
        perturbation_label_counts=tuple(
            (
                perturbation,
                sum(row.observed_label == perturbation for row in rows),
            )
            for perturbation in metadata.perturbations
        ),
        safe_control_counts=((metadata.safe_control_label, sum(count for _, count in safe_counts)),),
        barcode_quality_counts=(("valid", len(rows)),),
        label_quality_counts=(("valid", len(rows)),),
        per_specimen_coordinate_counts=coordinate_counts,
        per_specimen_perturbation_counts=perturbation_counts,
        per_specimen_safe_control_counts=safe_counts,
        per_specimen_barcode_valid_counts=coordinate_counts,
        per_specimen_label_valid_counts=coordinate_counts,
    )
    capability = audit_bridge_capability(candidate, summary)
    return BridgeSplitMetadata(
        rows,
        metadata.gene_names,
        metadata.perturbations,
        metadata.neighbour_cell_types,
        metadata.perturbation_targets,
        metadata.block_adjacency + cloned_adjacency,
        metadata.safe_control_label,
        table.relations,
        table.identity_sha256,
        candidate,
        summary,
        capability,
    )


def test_standardizer_controls_are_frozen_train_msafe_rows() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    manifest, expression, cell_ids, controls = _manifest_training_case()
    standardizer = fit_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    assert standardizer.split_identity_sha256 == manifest.split_identity_sha256
    assert standardizer.control_cell_ids == tuple(cell_ids[index] for index in controls)
    assert set(standardizer.control_roles) == {"safe_source:mSafe:train"}
    mapping = train_control_standardizer_to_mapping(standardizer)
    assert mapping["training_artifact"] == {
        "schema": "training_control_replay_v1",
        "artifact_identity_sha256": standardizer.training_artifact.artifact_identity_sha256,
        "expression_shape": list(standardizer.training_shape),
        "expression_sha256": standardizer.training_control_sha256,
        "chunk_count": len(standardizer.training_artifact.expression_chunks),
        "chunk_layout_sha256": module._hash_training_chunk_layout(
            standardizer.training_artifact.expression_chunks,
            standardizer.training_artifact.expression_shape,
        ),
        "cell_ids_sha256": standardizer.training_cell_order_sha256,
        "roles_sha256": module._hash_text_order(
            standardizer.control_roles, b"control_roles_v1\0"
        ),
    }


def test_standardizer_rejects_evaluation_or_non_msafe_control_rows() -> None:
    manifest, _, _, _ = _manifest_training_case()
    row_by_role = {
        "evaluation_safe": next(
            row
            for row in manifest.row_provenance
            if row.stable_row_id in set(manifest.evaluation_rows)
            and row.cell_role == "safe_source"
        ),
        "train_perturbation": next(
            row
            for row in manifest.row_provenance
            if row.stable_row_id in set(manifest.train_rows)
            and row.cell_role == "perturbation_source"
        ),
    }
    expression = np.zeros((1, len(manifest.gene_names)), dtype=np.float64)
    with pytest.raises(SpatialPerturbationScoringError, match="train"):
        fit_train_control_standardizer(
            expression,
            gene_names=manifest.gene_names,
            control_rows=(0,),
            cell_ids=(row_by_role["evaluation_safe"].cell_id,),
            split_manifest=manifest,
        )
    with pytest.raises(SpatialPerturbationScoringError, match="mSafe"):
        fit_train_control_standardizer(
            expression,
            gene_names=manifest.gene_names,
            control_rows=(0,),
            cell_ids=(row_by_role["train_perturbation"].cell_id,),
            split_manifest=manifest,
        )


def test_fit_rejects_unselected_tune_or_evaluation_payload_rows() -> None:
    manifest, _ = baseline()
    rows = manifest.row_provenance
    expression = np.zeros((len(rows), len(manifest.gene_names)), dtype=np.float64)
    cell_ids = tuple(row.cell_id for row in rows)
    controls = tuple(
        index
        for index, row in enumerate(rows)
        if row.stable_row_id in set(manifest.train_rows)
        and row.cell_role == "safe_source"
    )
    with pytest.raises(
        SpatialPerturbationScoringError,
        match="only frozen training control rows",
    ):
        fit_train_control_standardizer(
            expression,
            gene_names=manifest.gene_names,
            control_rows=controls,
            cell_ids=cell_ids,
            split_manifest=manifest,
        )


def test_standardizer_rejects_cell_provenance_or_split_substitution() -> None:
    manifest, expression, cell_ids, controls = _manifest_training_case()
    swapped = list(cell_ids)
    swapped[controls[0]] = "absent_control_cell"
    with pytest.raises(SpatialPerturbationScoringError, match="provenance"):
        fit_train_control_standardizer(
            expression,
            gene_names=manifest.gene_names,
            control_rows=controls,
            cell_ids=tuple(swapped),
            split_manifest=manifest,
        )


def test_training_hash_is_endian_stable_chunked_and_has_bounded_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    _, expression, _, _ = _manifest_training_case()
    tiled = np.tile(expression, (1_600, 1))
    # Exercise the fixed-endian primitive directly at a realistic upper-scale
    # selection without constructing artificial provenance strings.
    rows = tuple(range(0, tiled.shape[0], 2))
    updates: list[int] = []
    real_sha256 = module.hashlib.sha256

    class RecordingHash:
        def __init__(self) -> None:
            self._delegate = real_sha256()

        def update(self, value: bytes) -> None:
            updates.append(len(value))
            self._delegate.update(value)

        def hexdigest(self) -> str:
            return self._delegate.hexdigest()

    monkeypatch.setattr(module.hashlib, "sha256", lambda *args, **kwargs: RecordingHash())
    tracemalloc.start()
    native = module._hash_training_controls(tiled, rows)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    swapped = module._hash_training_controls(tiled.astype(">f8"), rows)
    assert native == swapped
    assert max(updates) <= module.TRAINING_HASH_CHUNK_BYTES
    assert peak < 16 * 1024 * 1024


@pytest.mark.parametrize("logged_half_range", [0.0, 0.5e-6, 1.0e-6])
def test_scale_fallback_is_exactly_one_at_or_below_threshold(
    logged_half_range: float,
) -> None:
    manifest, expression, cell_ids, controls = _manifest_training_case()
    expression = expression.copy()
    expression[list(controls), 0] = np.resize(
        np.asarray([0.0, math.expm1(2.0 * logged_half_range)]),
        len(controls),
    )
    standardizer = fit_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    assert standardizer.scale[0] == 1.0


def test_crossing_scale_fallback_threshold_is_expected_not_invariant() -> None:
    manifest, expression, cell_ids, controls = _manifest_training_case()
    expression = expression.copy()
    expression[list(controls), 0] = np.resize(
        np.asarray([0.0, math.expm1(1.0e-6)]), len(controls)
    )
    first = fit_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    scaled = np.expm1(np.log1p(expression) * 4.0)
    second = fit_train_control_standardizer(
        scaled,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    assert first.scale[0] == 1.0
    assert second.scale[0] > 1.0e-6
    first_values = apply_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        standardizer=first,
        split_manifest=manifest,
    )
    second_values = apply_train_control_standardizer(
        scaled,
        gene_names=manifest.gene_names,
        standardizer=second,
        split_manifest=manifest,
    )
    assert not np.allclose(first_values[:, 0], second_values[:, 0])


@lru_cache(maxsize=1)
def _bridge_case() -> tuple[
    BridgeEligibilityResult,
    np.ndarray,
    tuple[str, ...],
    TrainControlStandardizer,
    tuple[BridgePrediction, ...],
]:
    manifest, evidence = baseline()
    eligibility = evaluate_bridge_eligibility(manifest, evidence)
    standardizer = _fit_manifest_standardizer()

    cell_ids = tuple(sorted({
        cell_id
        for relation in manifest.neighbour_relations
        for cell_id in (relation.neighbor_cell_id,)
    } | {
        cell_id
        for parent in evidence.parent_evidence
        for cell_id in (
            *parent.perturbation_source_cell_ids,
            *parent.safe_source_cell_ids,
        )
    }))
    row_by_cell = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    gene_by_name = {gene: index for index, gene in enumerate(manifest.gene_names)}
    expression = np.zeros((len(cell_ids), len(manifest.gene_names)), dtype=np.float64)
    high = math.expm1(1.0)

    relation_by_id = {relation.relation_id: relation for relation in manifest.neighbour_relations}
    for unit_evidence in evidence.unit_evidence:
        unit = next(unit for unit in manifest.primary_units if unit.unit_id == unit_evidence.unit_id)
        gene_index = gene_by_name[unit.target_gene]
        for relation_id in unit_evidence.perturbation_neighbour_relation_ids:
            cell_id = relation_by_id[relation_id].neighbor_cell_id
            expression[row_by_cell[cell_id], gene_index] = high
    parent_by_context = {
        (parent.animal_id, parent.perturbation_id): parent
        for parent in manifest.perturbation_parents
    }
    for parent_evidence in evidence.parent_evidence:
        parent = parent_by_context[
            (parent_evidence.animal_id, parent_evidence.perturbation_id)
        ]
        gene_index = gene_by_name[parent.target_gene]
        for cell_id in parent_evidence.perturbation_source_cell_ids:
            expression[row_by_cell[cell_id], gene_index] = high

    evaluation_animals = set(manifest.evaluation_animals)
    scoreable_units = {
        unit.unit_id: unit
        for unit in manifest.primary_units
        if unit.animal_id in evaluation_animals
        if unit.unit_id not in eligibility.abstained_unit_ids
    }
    scoreable_parents = {
        parent.parent_id: parent
        for parent in manifest.perturbation_parents
        if parent.animal_id in evaluation_animals
        if parent.parent_id in eligibility.scoreable_parent_ids
    }
    predictions = tuple(
        [
            BridgePrediction(unit_id, "neighbor", 2.0)
            for unit_id in sorted(scoreable_units)
        ]
        + [
            BridgePrediction(parent_id, "own", 2.0)
            for parent_id in sorted(scoreable_parents)
        ]
    )
    return eligibility, expression, cell_ids, standardizer, predictions


def _effect_table(
    predictions: tuple[BridgePrediction, ...] | None = None,
    *,
    expression_transform: object | None = None,
) -> BridgeEffectTable:
    eligibility, expression, cell_ids, standardizer, exact_predictions = _bridge_case()
    if callable(expression_transform):
        expression = expression_transform(expression)
    return build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=exact_predictions if predictions is None else predictions,
    )


def _evaluation_predictions(
    eligibility: BridgeEligibilityResult,
    *,
    neighbor_delta: float = 2.0,
    own_delta: float = 2.0,
) -> tuple[BridgePrediction, ...]:
    evaluation_animals = set(eligibility.manifest.evaluation_animals)
    abstained = set(eligibility.abstained_unit_ids)
    scoreable_units = tuple(
        unit
        for unit in eligibility.manifest.primary_units
        if unit.animal_id in evaluation_animals and unit.unit_id not in abstained
    )
    scoreable_parents = set(eligibility.scoreable_parent_ids)
    parents = tuple(
        parent
        for parent in eligibility.manifest.perturbation_parents
        if parent.animal_id in evaluation_animals
        and parent.parent_id in scoreable_parents
    )
    return tuple(
        [
            BridgePrediction(unit.unit_id, "neighbor", neighbor_delta)
            for unit in scoreable_units
        ]
        + [
            BridgePrediction(parent.parent_id, "own", own_delta)
            for parent in parents
        ]
    )


def _resign_effect_table(
    table: BridgeEffectTable,
    effects: tuple[BridgeEffect, ...],
) -> BridgeEffectTable:
    import src.evaluation.spatial_perturbation_scoring as module

    shell = object.__new__(BridgeEffectTable)
    for name, value in (
        ("effects", effects),
        ("eligibility", table.eligibility),
        ("split_identity_sha256", table.split_identity_sha256),
        ("neighbour_table_identity_sha256", table.neighbour_table_identity_sha256),
        ("eligibility_identity_sha256", table.eligibility_identity_sha256),
        ("standardizer_identity_sha256", table.standardizer_identity_sha256),
    ):
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "effect_table_identity_sha256", "0" * 64)
    identity = module._identity(module._effect_table_unsigned(shell))
    return BridgeEffectTable(
        effects,
        table.eligibility,
        table.split_identity_sha256,
        table.neighbour_table_identity_sha256,
        table.eligibility_identity_sha256,
        table.standardizer_identity_sha256,
        identity,
    )


def test_scoring_contract_contains_only_evaluation_animal_units() -> None:
    eligibility, expression, cell_ids, standardizer, _ = _bridge_case()
    table = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    assert {effect.animal_id for effect in table.effects} == {"mouse_1"}
    assert len(table.effects) == 25


def test_development_animal_prediction_is_rejected_as_extra() -> None:
    eligibility, expression, cell_ids, standardizer, _ = _bridge_case()
    predictions = _evaluation_predictions(eligibility)
    development_unit = next(
        unit
        for unit in eligibility.manifest.primary_units
        if unit.animal_id in eligibility.manifest.development_animals
    )
    with pytest.raises(SpatialPerturbationScoringError, match="extra"):
        build_bridge_effect_table(
            expression,
            cell_ids=cell_ids,
            gene_names=standardizer.genes,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=predictions
            + (BridgePrediction(development_unit.unit_id, "neighbor", 1e6),),
        )


def test_effect_context_is_replayed_from_frozen_unit_not_self_signed() -> None:
    table = _effect_table()
    effect = next(item for item in table.effects if item.endpoint == "neighbor")
    import src.evaluation.spatial_perturbation_scoring as module

    forged = module._make_effect(
        effect.unit_id,
        effect.endpoint,
        "mouse_2",
        effect.perturbation_id,
        effect.gene_name,
        effect.neighbor_cell_type,
        effect.band,
        effect.observed_delta,
        effect.predicted_delta,
    )
    effects = tuple(forged if item.unit_id == effect.unit_id else item for item in table.effects)
    with pytest.raises(SpatialPerturbationScoringError, match="frozen.*context"):
        _resign_effect_table(table, effects)


def test_score_entry_replays_expression_instead_of_accepting_effect_table() -> None:
    eligibility, expression, cell_ids, standardizer, _ = _bridge_case()
    result = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    assert result.neighbor_effect_rmse == pytest.approx(0.0)


def test_one_band_abstention_keeps_primary_sibling_and_only_abstains_calibration() -> None:
    manifest, _ = baseline()
    proximal = next(
        unit
        for unit in manifest.primary_units
        if unit.animal_id == "mouse_1" and unit.band == "proximal"
    )
    evidence = complete_evidence(
        manifest, unit_overrides={proximal.unit_id: 29}
    )
    eligibility = evaluate_bridge_eligibility(manifest, evidence)
    assert eligibility.eligible is True
    _, expression, cell_ids, standardizer, _ = _bridge_case()
    table = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    contexts: dict[tuple[str, str, str, str], set[str]] = {}
    for effect in table.effects:
        if effect.endpoint == "neighbor":
            context = (
                effect.animal_id,
                effect.perturbation_id,
                effect.gene_name,
                effect.neighbor_cell_type,
            )
            contexts.setdefault(context, set()).add(effect.band)
    affected_context = (
        proximal.animal_id,
        proximal.perturbation_id,
        proximal.target_gene,
        proximal.neighbour_cell_type,
    )
    assert contexts[affected_context] == {"local"}
    assert sum(bands == {"proximal", "local"} for bands in contexts.values()) == 9
    score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    assert score.coverage == pytest.approx(0.95)
    assert score.abstention == pytest.approx(0.05)
    assert score.distance_decay_calibration_error == pytest.approx(0.0)
    assert score.distance_calibration_eligible_pairs == 9
    assert score.distance_calibration_total_contexts == 10
    assert score.distance_calibration_coverage == pytest.approx(0.9)
    assert score.distance_calibration_abstention == pytest.approx(0.1)
    mapping = bridge_score_to_mapping(
        score,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    assert mapping["distance_decay_calibration_error"] == pytest.approx(0.0)
    assert mapping["distance_calibration_eligible_pairs"] == 9
    assert mapping["distance_calibration_total_contexts"] == 10
    assert mapping["distance_calibration_coverage"] == pytest.approx(0.9)
    assert mapping["distance_calibration_abstention"] == pytest.approx(0.1)


def test_primary_band_weights_stay_fixed_half_when_one_proximal_unit_abstains() -> None:
    manifest, _ = baseline()
    proximal = next(
        unit
        for unit in manifest.primary_units
        if unit.animal_id == "mouse_1" and unit.band == "proximal"
    )
    eligibility = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(manifest, unit_overrides={proximal.unit_id: 29}),
    )
    expression, cell_ids = _expression_for_eligibility(eligibility)
    standardizer = _standardizer_for_manifest(manifest)
    unit_by_id = {unit.unit_id: unit for unit in manifest.primary_units}
    predictions = tuple(
        BridgePrediction(
            prediction.unit_id,
            prediction.endpoint,
            2.0
            if prediction.endpoint == "own"
            else 2.0 + (
                1.0 if unit_by_id[prediction.unit_id].band == "proximal" else 3.0
            ),
        )
        for prediction in _evaluation_predictions(eligibility)
    )
    score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=manifest.gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    assert score.coverage == pytest.approx(0.95)
    assert score.neighbor_effect_rmse == pytest.approx(math.sqrt(5.0))


def test_fully_abstained_contexts_remain_in_calibration_denominator() -> None:
    manifest, _ = baseline()
    failed = next(
        unit
        for unit in manifest.primary_units
        if unit.animal_id in set(manifest.evaluation_animals)
    )
    eligibility = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(manifest, unit_overrides={failed.unit_id: 0}),
    )
    expression, cell_ids = _expression_for_eligibility(eligibility)
    standardizer = _standardizer_for_manifest(manifest)
    score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=manifest.gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    assert score.coverage == pytest.approx(0.8)
    assert score.distance_calibration_eligible_pairs == 8
    assert score.distance_calibration_total_contexts == 10
    assert score.distance_calibration_coverage == pytest.approx(0.8)
    assert score.distance_calibration_abstention == pytest.approx(0.2)


def test_ineligible_task5_result_cannot_produce_publishable_score() -> None:
    manifest, _ = baseline()
    evaluation_units = tuple(
        unit for unit in manifest.primary_units if unit.animal_id == "mouse_1"
    )
    evidence = complete_evidence(
        manifest,
        unit_overrides={unit.unit_id: 0 for unit in evaluation_units},
    )
    eligibility = evaluate_bridge_eligibility(manifest, evidence)
    assert eligibility.eligible is False
    _, expression, cell_ids, standardizer, _ = _bridge_case()
    with pytest.raises(SpatialPerturbationScoringError, match="evaluation.*coverage"):
        build_bridge_effect_table(
            expression,
            cell_ids=cell_ids,
            gene_names=standardizer.genes,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=_evaluation_predictions(eligibility),
        )


def test_distance_calibration_is_none_when_no_band_pair_is_available() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    table = _effect_table()
    neighbor = tuple(
        effect
        for effect in table.effects
        if effect.endpoint == "neighbor" and effect.band == "proximal"
    )
    calibration = module._distance_calibration(neighbor)
    assert calibration.error is None
    assert calibration.eligible_pairs == 0
    assert calibration.total_contexts == 10
    assert calibration.coverage == 0.0
    assert calibration.abstention == 1.0


def test_development_failures_do_not_block_intact_evaluation_publication() -> None:
    manifest, _ = baseline()
    development_units = tuple(
        unit
        for unit in manifest.primary_units
        if unit.animal_id in set(manifest.development_animals)
    )
    eligibility = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            unit_overrides={unit.unit_id: 0 for unit in development_units},
        ),
    )
    assert eligibility.eligible is False
    expression, cell_ids = _expression_for_eligibility(eligibility)
    standardizer = _standardizer_for_manifest(manifest)
    score = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=manifest.gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    assert score.coverage == 1.0
    assert score.abstention == 0.0


@pytest.mark.parametrize("failed", [1, 2])
def test_evaluation_overall_coverage_gate_is_independent_and_fail_closed(
    failed: int,
) -> None:
    manifest, _ = baseline()
    evaluation_units = tuple(
        unit
        for unit in manifest.primary_units
        if unit.animal_id in set(manifest.evaluation_animals)
    )
    eligibility = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            unit_overrides={unit.unit_id: 0 for unit in evaluation_units[:failed]},
        ),
    )
    expression, cell_ids = _expression_for_eligibility(eligibility)
    standardizer = _standardizer_for_manifest(manifest)
    arguments = dict(
        cell_ids=cell_ids,
        gene_names=manifest.gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=_evaluation_predictions(eligibility),
    )
    if failed == 1:
        score = score_bridge_predictions(expression, **arguments)
        assert score.coverage == pytest.approx(0.8)
        assert score.abstention == pytest.approx(0.2)
    else:
        with pytest.raises(SpatialPerturbationScoringError, match="evaluation.*coverage"):
            score_bridge_predictions(expression, **arguments)


def test_near_constant_anticorrelation_has_frozen_zero_pcc() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    table = _effect_table()
    neighbor = tuple(effect for effect in table.effects if effect.endpoint == "neighbor")
    mutated = tuple(
        module._make_effect(
            effect.unit_id,
            effect.endpoint,
            effect.animal_id,
            effect.perturbation_id,
            effect.gene_name,
            effect.neighbor_cell_type,
            effect.band,
            2.0 + (1e-13 if index % 2 else -1e-13),
            2.0 - (1e-13 if index % 2 else -1e-13),
        )
        for index, effect in enumerate(neighbor)
    )
    assert module._weighted_pcc(mutated) == 0.0


def test_fit_uses_log1p_float64_population_statistics_and_train_controls_only() -> None:
    manifest, expression, cell_ids, controls = _manifest_training_case()
    expression = np.full_like(expression, 10_000, dtype=np.int64)
    for offset, row_index in enumerate(controls):
        expression[row_index, :] = 0 if offset % 2 == 0 else 3
    original = expression.copy()
    standardizer = fit_train_control_standardizer(
        expression,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )

    expected = np.log1p(expression[list(controls)].astype(np.float64))
    assert standardizer.center == pytest.approx(tuple(expected.mean(axis=0)))
    assert standardizer.scale == pytest.approx(tuple(expected.std(axis=0, ddof=0)))
    assert standardizer.control_rows == controls
    assert standardizer.training_shape == expression.shape
    assert standardizer.genes == manifest.gene_names
    assert len(standardizer.training_identity_sha256) == 64
    np.testing.assert_array_equal(expression, original)


def test_constant_training_gene_uses_unit_scale_and_apply_never_refits() -> None:
    manifest, training, cell_ids, controls = _manifest_training_case()
    training = training.copy()
    training[list(controls), 0] = 3.0
    standardizer = fit_train_control_standardizer(
        training,
        gene_names=manifest.gene_names,
        control_rows=controls,
        cell_ids=cell_ids,
        split_manifest=manifest,
    )
    evaluation = np.zeros((2, len(manifest.gene_names)), dtype=np.float64)
    evaluation[:, 0] = 3.0
    evaluation[:, 1] = (8.0, 80.0)
    transformed = apply_train_control_standardizer(
        evaluation,
        gene_names=manifest.gene_names,
        standardizer=standardizer,
        split_manifest=manifest,
    )

    assert standardizer.scale[0] == 1.0
    assert transformed.dtype == np.dtype("float64")
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[1, 0] == pytest.approx(0.0)
    assert transformed[:, 1].mean() != pytest.approx(0.0)


def test_apply_requires_exact_gene_order_and_preserves_input_and_rng() -> None:
    manifest, _, _, _ = _manifest_training_case()
    standardizer = _fit_manifest_standardizer()
    evaluation = np.asarray(
        [[0.0, math.expm1(1.0), 0.0, math.expm1(1.0), 0.0]],
        dtype=np.float32,
    )
    original = evaluation.copy()
    state = np.random.get_state()

    with pytest.raises(SpatialPerturbationScoringError, match="exact gene order"):
        apply_train_control_standardizer(
            evaluation[:, ::-1],
            gene_names=tuple(reversed(manifest.gene_names)),
            standardizer=standardizer,
            split_manifest=manifest,
        )
    transformed = apply_train_control_standardizer(
        evaluation,
        gene_names=manifest.gene_names,
        standardizer=standardizer,
        split_manifest=manifest,
    )

    np.testing.assert_array_equal(evaluation, original)
    np.testing.assert_allclose(
        transformed,
        np.asarray([[-1.0, 1.0, -1.0, 1.0, -1.0]]),
        atol=1e-7,
    )
    new_state = np.random.get_state()
    assert state[0] == new_state[0]  # type: ignore[index]
    np.testing.assert_array_equal(state[1], new_state[1])  # type: ignore[index]
    assert state[2:] == new_state[2:]  # type: ignore[index]


@pytest.mark.parametrize(
    "case, message",
    [
        ("nan", "finite"),
        ("negative", "nonnegative"),
        ("bool", "real numeric"),
        ("complex", "real numeric"),
        ("object", "real numeric"),
        ("empty", "nonempty"),
        ("duplicate", "unique"),
        ("range", "range"),
    ],
)
def test_standardizer_rejects_invalid_domain_shape_dtype_and_rows(
    case: str, message: str
) -> None:
    manifest, expression, cell_ids, controls = _manifest_training_case()
    control_rows = controls
    if case == "nan":
        expression = expression.copy()
        expression[0, 0] = math.nan
    elif case == "negative":
        expression = expression.copy()
        expression[0, 0] = -1.01
    elif case == "bool":
        expression = expression.astype(np.bool_)
    elif case == "complex":
        expression = expression.astype(np.complex128)
    elif case == "object":
        expression = expression.astype(object)
    elif case == "empty":
        control_rows = ()
    elif case == "duplicate":
        control_rows = (controls[0], controls[0])
    elif case == "range":
        control_rows = (len(expression),)
    with pytest.raises(SpatialPerturbationScoringError, match=message):
        fit_train_control_standardizer(
            expression,
            gene_names=manifest.gene_names,
            control_rows=control_rows,
            cell_ids=cell_ids,
            split_manifest=manifest,
        )


def test_standardizer_is_immutable_and_accessors_revalidate_tampering() -> None:
    manifest, _, _, _ = _manifest_training_case()
    standardizer = _fit_manifest_standardizer()
    with pytest.raises(FrozenInstanceError):
        standardizer.center = (0.0,) * len(standardizer.genes)  # type: ignore[misc]
    with pytest.raises(SpatialPerturbationScoringError, match="identity|replay"):
        replace(standardizer, center=(0.0,) * len(standardizer.genes))

    object.__setattr__(standardizer, "scale", (0.0,) * len(standardizer.genes))
    with pytest.raises(SpatialPerturbationScoringError):
        train_control_standardizer_to_mapping(standardizer)
    with pytest.raises(SpatialPerturbationScoringError):
        apply_train_control_standardizer(
            np.zeros((1, len(manifest.gene_names))),
            gene_names=manifest.gene_names,
            standardizer=standardizer,
            split_manifest=manifest,
        )


def test_resigned_standardizer_scale_is_rejected_by_training_expression_replay() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    standardizer = _fit_manifest_standardizer()
    changed_scale = tuple(value + 0.25 for value in standardizer.scale)
    rows_identity = module._hash_integer_order(
        standardizer.control_rows, b"control_rows_v1\0"
    )
    cells_identity = module._hash_text_order(
        standardizer.control_cell_ids, b"control_cell_ids_v1\0"
    )
    roles_identity = module._hash_text_order(
        standardizer.control_roles, b"control_roles_v1\0"
    )
    unsigned = module._standardizer_unsigned(
        standardizer.genes,
        standardizer.center,
        changed_scale,
        standardizer.control_rows,
        standardizer.training_shape,
        standardizer.training_control_sha256,
        standardizer.split_identity_sha256,
        standardizer.training_cell_order_sha256,
        rows_identity,
        cells_identity,
        roles_identity,
        standardizer.training_artifact.artifact_identity_sha256,
    )
    assert "center_hex" not in unsigned
    assert "scale_hex" not in unsigned
    assert unsigned["center_sha256"] == module._hash_float_order(
        standardizer.center, b"standardizer_center_v1\0"
    )
    with pytest.raises(SpatialPerturbationScoringError, match="replay"):
        TrainControlStandardizer(
            standardizer.genes,
            standardizer.center,
            changed_scale,
            module._identity(unsigned),
            standardizer.control_rows,
            standardizer.training_shape,
            standardizer.training_control_sha256,
            standardizer.split_manifest,
            standardizer.split_identity_sha256,
            standardizer.training_cell_ids,
            standardizer.training_cell_order_sha256,
            standardizer.control_cell_ids,
            standardizer.control_roles,
            standardizer.training_artifact,
        )


def test_standardizer_retains_bounded_fixed_endian_replay_artifact() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    standardizer = _fit_manifest_standardizer()
    artifact = standardizer.training_artifact
    assert artifact.expression_shape == standardizer.training_shape
    assert artifact.genes == standardizer.genes
    assert artifact.cell_ids == standardizer.training_cell_ids
    assert artifact.roles == standardizer.control_roles
    assert all(
        0 < len(chunk) <= module.TRAINING_HASH_CHUNK_BYTES
        for chunk in artifact.expression_chunks
    )
    replayed = np.vstack(
        [
            np.frombuffer(chunk, dtype="<f8").reshape(-1, len(artifact.genes))
            for chunk in artifact.expression_chunks
        ]
    )
    assert module._hash_training_artifact_chunks(
        artifact.expression_chunks, artifact.expression_shape
    ) == artifact.expression_sha256
    assert replayed.dtype == np.dtype("float64")


def test_training_artifact_rejects_noncanonical_equivalent_row_chunking() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    shape = (100_000, 1)
    row = np.asarray([1.0], dtype="<f8").tobytes()
    canonical = (row * shape[0],)
    one_row_chunks = (row,) * shape[0]
    assert module._hash_training_artifact_chunks(canonical, shape) == (
        module._hash_training_artifact_chunks(one_row_chunks, shape)
    )
    module._validate_canonical_training_chunks(canonical, shape)
    with pytest.raises(SpatialPerturbationScoringError, match="canonical chunk"):
        module._validate_canonical_training_chunks(one_row_chunks, shape)


def test_task5_and_task6_public_artifacts_build_perfect_zero_rmse_effects() -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    table = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    result = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )

    assert result.neighbor_effect_rmse == pytest.approx(0.0)
    assert result.own_effect_rmse == pytest.approx(0.0)
    assert result.neighbor_effect_pcc == pytest.approx(0.0)
    assert result.distance_decay_calibration_error == pytest.approx(0.0)
    assert result.effect_sign_accuracy == pytest.approx(1.0)
    assert result.coverage == pytest.approx(1.0)
    assert result.abstention == pytest.approx(0.0)
    assert len(result.animal_level_unit_table) == 1
    publication = dict(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    effect_mapping = bridge_effect_table_to_mapping(table, **publication)
    score_mapping = bridge_score_to_mapping(result, **publication)
    assert json.loads(json.dumps(effect_mapping))
    assert json.loads(json.dumps(score_mapping))
    assert score_mapping["distance_calibration_eligible_pairs"] == 10
    assert score_mapping["distance_calibration_total_contexts"] == 10
    assert score_mapping["distance_calibration_coverage"] == 1.0
    assert score_mapping["distance_calibration_abstention"] == 0.0
    assert json.loads(json.dumps(eligibility_result_to_mapping(eligibility)))


def test_primary_bands_are_equal_weight_and_own_is_excluded() -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    unit_by_id = {unit.unit_id: unit for unit in eligibility.manifest.primary_units}
    changed: list[BridgePrediction] = []
    for prediction in predictions:
        if prediction.endpoint == "own":
            changed.append(BridgePrediction(prediction.unit_id, "own", 1_002.0))
        else:
            band = unit_by_id[prediction.unit_id].band
            error = 1.0 if band == "proximal" else 3.0
            changed.append(
                BridgePrediction(prediction.unit_id, "neighbor", 2.0 + error)
            )
    result = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=tuple(changed),
    )

    assert result.neighbor_effect_rmse == pytest.approx(math.sqrt((1.0 + 9.0) / 2.0))
    assert result.own_effect_rmse == pytest.approx(1_000.0)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate"])
def test_missing_extra_or_duplicate_predictions_fail_closed(kind: str) -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    if kind == "missing":
        invalid = predictions[:-1]
    elif kind == "extra":
        invalid = predictions + (BridgePrediction("0" * 64, "own", 0.0),)
    else:
        invalid = predictions + (predictions[0],)

    with pytest.raises(SpatialPerturbationScoringError, match=kind):
        build_bridge_effect_table(
            expression,
            cell_ids=cell_ids,
            gene_names=standardizer.genes,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=invalid,
        )


def test_score_and_effect_table_are_immutable_and_accessors_revalidate() -> None:
    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    table = _effect_table()
    result = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    with pytest.raises(FrozenInstanceError):
        result.coverage = 0.0  # type: ignore[misc]
    with pytest.raises(SpatialPerturbationScoringError):
        replace(result, neighbor_effect_rmse=1.0)

    object.__setattr__(result, "coverage", 0.5)
    with pytest.raises(SpatialPerturbationScoringError):
        bridge_score_to_mapping(
            result,
            expression=expression,
            cell_ids=cell_ids,
            gene_names=standardizer.genes,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=predictions,
        )
    object.__setattr__(table, "effects", ())
    with pytest.raises(SpatialPerturbationScoringError):
        bridge_effect_table_to_mapping(
            table,
            expression=expression,
            cell_ids=cell_ids,
            gene_names=standardizer.genes,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=predictions,
        )


def test_self_signed_arbitrary_effects_and_score_cannot_be_published() -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    eligibility, expression, cell_ids, standardizer, predictions = _bridge_case()
    table = _effect_table()
    forged_effects = tuple(
        module._make_effect(
            effect.unit_id,
            effect.endpoint,
            effect.animal_id,
            effect.perturbation_id,
            effect.gene_name,
            effect.neighbor_cell_type,
            effect.band,
            0.0,
            0.0,
        )
        for effect in table.effects
    )
    forged_table = _resign_effect_table(table, forged_effects)
    forged_score = module._score_effect_table(forged_table)
    publication = dict(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    with pytest.raises(SpatialPerturbationScoringError, match="publication replay"):
        bridge_effect_table_to_mapping(forged_table, **publication)
    with pytest.raises(SpatialPerturbationScoringError, match="publication replay"):
        bridge_score_to_mapping(forged_score, **publication)


def test_resource_cap_is_checked_before_log_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluation.spatial_perturbation_scoring as module

    oversized = np.lib.stride_tricks.as_strided(
        np.zeros(1, dtype=np.float64),
        shape=(module.MAX_EXPRESSION_ELEMENTS // 5 + 1, 5),
        strides=(0, 0),
        writeable=False,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("allocation happened before the resource cap")

    monkeypatch.setattr(module.np, "log1p", forbidden)
    manifest, _, _, _ = _manifest_training_case()
    with pytest.raises(SpatialPerturbationScoringError, match="resource limit"):
        fit_train_control_standardizer(
            oversized,
            gene_names=manifest.gene_names,
            control_rows=(0,),
            cell_ids=(),
            split_manifest=manifest,
        )
