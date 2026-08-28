from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
import hashlib
import inspect
import json
import random

import pytest

from src.evaluation import spatial_perturbation_split as split_module
from src.evaluation.spatial_perturbation_split import (
    MAX_ABSTENTION,
    MIN_BAND_NEIGHBOURS,
    MIN_CELL_TYPE_NEIGHBOURS,
    MIN_COVERAGE,
    MIN_SAFE_SOURCE_CELLS,
    MIN_SOURCE_CELLS,
    MIN_SPATIAL_BLOCKS,
    BridgeBlockAdjacency,
    BridgeEligibilityEvidence,
    BridgeParentEvidence,
    BridgePrimaryUnitEvidence,
    BridgeSplitManifest,
    BridgeSplitMetadata,
    BridgeSplitRow,
    SpatialPerturbationSplitError,
    build_bridge_eligibility_evidence,
    build_pilot_fold,
    eligibility_evidence_to_mapping,
    evaluate_bridge_eligibility,
    split_manifest_to_mapping,
)


PERTURBATIONS = tuple(f"guide_{index}" for index in range(5))
TARGETS = tuple(f"Gene{index}" for index in range(5))
NEIGHBOUR_TYPES = ("astrocyte", "microglia")
BANDS = ("proximal", "local")


def _canonical_sha(mapping: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(mapping, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def synthetic_metadata(
    *, animal_count: int = 3, adjacency: bool = False
) -> BridgeSplitMetadata:
    rows: list[BridgeSplitRow] = []
    row_id = 0
    for animal_index in range(animal_count):
        animal = f"mouse_{animal_index + 1}"
        section = f"{animal}_section"
        for perturbation in PERTURBATIONS:
            for role, label in (
                ("perturbation_source", perturbation),
                ("safe_source", "mSafe"),
            ):
                for index in range(30):
                    rows.append(
                        BridgeSplitRow(
                            row_id, f"cell_{row_id:06d}", animal, section,
                            f"block_{index % 3}", perturbation, label,
                            "source_type", role, "own",
                        )
                    )
                    row_id += 1
            for neighbour_type in NEIGHBOUR_TYPES:
                for band in BANDS:
                    for role in ("perturbation_neighbour", "safe_neighbour"):
                        for index in range(30):
                            rows.append(
                                BridgeSplitRow(
                                    row_id, f"cell_{row_id:06d}", animal, section,
                                    f"block_{index % 3}", perturbation, "unperturbed",
                                    neighbour_type, role, band,
                                )
                            )
                            row_id += 1
    adjacent = tuple(
        BridgeBlockAdjacency(
            f"mouse_{index + 1}", f"mouse_{index + 1}_section", "block_0", "block_1"
        )
        for index in range(animal_count)
    ) if adjacency else ()
    return BridgeSplitMetadata(
        tuple(rows), TARGETS, PERTURBATIONS, NEIGHBOUR_TYPES,
        tuple(zip(PERTURBATIONS, TARGETS)), adjacent, "mSafe",
    )


def _rows_for(
    manifest: BridgeSplitManifest,
    *,
    animal: str,
    perturbation: str,
    role: str,
    cell_type: str | None = None,
    band: str | None = None,
    blocks: tuple[str, ...] | None = None,
) -> tuple[BridgeSplitRow, ...]:
    return tuple(
        row for row in manifest.row_provenance
        if row.animal_id == animal
        and row.context_perturbation_id == perturbation
        and row.cell_role == role
        and (cell_type is None or row.cell_type == cell_type)
        and (band is None or row.distance_band == band)
        and (blocks is None or row.spatial_block in blocks)
    )


def complete_evidence(
    manifest: BridgeSplitManifest,
    *,
    source_overrides: dict[str, int] | None = None,
    safe_source_overrides: dict[str, int] | None = None,
    unit_overrides: dict[str, int] | None = None,
    safe_unit_overrides: dict[str, int] | None = None,
    block_overrides: dict[str, tuple[str, ...]] | None = None,
) -> BridgeEligibilityEvidence:
    source_overrides = source_overrides or {}
    safe_source_overrides = safe_source_overrides or {}
    unit_overrides = unit_overrides or {}
    safe_unit_overrides = safe_unit_overrides or {}
    block_overrides = block_overrides or {}
    parents: list[BridgeParentEvidence] = []
    for parent in manifest.perturbation_parents:
        parent_key = parent.parent_id
        source_rows = _rows_for(
            manifest, animal=parent.animal_id, perturbation=parent.perturbation_id,
            role="perturbation_source", blocks=block_overrides.get(f"{parent_key}:source"),
        )
        safe_rows = _rows_for(
            manifest, animal=parent.animal_id, perturbation=parent.perturbation_id,
            role="safe_source", blocks=block_overrides.get(f"{parent_key}:safe_source"),
        )
        parents.append(
            BridgeParentEvidence(
                parent.animal_id, parent.perturbation_id, parent.target_gene,
                tuple(row.cell_id for row in source_rows[: source_overrides.get(parent_key, 20)]),
                tuple(row.cell_id for row in safe_rows[: safe_source_overrides.get(parent_key, 20)]),
            )
        )
    units: list[BridgePrimaryUnitEvidence] = []
    for unit in manifest.primary_units:
        treatment_rows = _rows_for(
            manifest, animal=unit.animal_id, perturbation=unit.perturbation_id,
            role="perturbation_neighbour", cell_type=unit.neighbour_cell_type,
            band=unit.band, blocks=block_overrides.get(f"{unit.unit_id}:neighbour"),
        )
        safe_rows = _rows_for(
            manifest, animal=unit.animal_id, perturbation=unit.perturbation_id,
            role="safe_neighbour", cell_type=unit.neighbour_cell_type,
            band=unit.band, blocks=block_overrides.get(f"{unit.unit_id}:safe_neighbour"),
        )
        units.append(
            BridgePrimaryUnitEvidence(
                unit.unit_id,
                tuple(row.cell_id for row in treatment_rows[: unit_overrides.get(unit.unit_id, 30)]),
                tuple(row.cell_id for row in safe_rows[: safe_unit_overrides.get(unit.unit_id, 30)]),
            )
        )
    return build_bridge_eligibility_evidence(manifest, tuple(parents), tuple(units))


@lru_cache(maxsize=1)
def baseline() -> tuple[BridgeSplitManifest, BridgeEligibilityEvidence]:
    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    return manifest, complete_evidence(manifest)


def test_manifest_freezes_exact_three_animal_units_and_whole_animal_fold() -> None:
    manifest, _ = baseline()
    assert manifest.development_animals == ("mouse_2", "mouse_3")
    assert manifest.evaluation_animals == ("mouse_1",)
    assert manifest.tune_rows == ()
    assert not set(manifest.development_rows) & set(manifest.evaluation_rows)
    assert len(manifest.perturbation_parents) == 15
    assert len(manifest.primary_units) == 60
    assert {unit.band for unit in manifest.primary_units} == {"proximal", "local"}
    assert {unit.neighbour_cell_type for unit in manifest.primary_units} == set(NEIGHBOUR_TYPES)


@pytest.mark.parametrize("animal_count", (1, 2, 4))
def test_pilot_rejects_any_specimen_count_other_than_three(animal_count: int) -> None:
    with pytest.raises(SpatialPerturbationSplitError, match="exactly three animals"):
        build_pilot_fold(synthetic_metadata(animal_count=animal_count), "mouse_1")


def test_evaluation_requires_revalidated_manifest_and_exact_split_binding() -> None:
    manifest, evidence = baseline()
    assert evaluate_bridge_eligibility(manifest, evidence).eligible is True
    other_manifest = build_pilot_fold(synthetic_metadata(adjacency=True), "mouse_1")
    with pytest.raises(SpatialPerturbationSplitError, match="split identity"):
        evaluate_bridge_eligibility(other_manifest, evidence)
    assert tuple(inspect.signature(evaluate_bridge_eligibility).parameters) == ("manifest", "evidence")


def test_evidence_requires_every_exact_parent_and_unit_without_deletion() -> None:
    manifest, evidence = baseline()
    missing_unit = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, evidence.unit_evidence[:-1]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="exact frozen primary units"):
        evaluate_bridge_eligibility(manifest, missing_unit)
    with pytest.raises(SpatialPerturbationSplitError, match="unique unit evidence"):
        build_bridge_eligibility_evidence(
            manifest, evidence.parent_evidence, evidence.unit_evidence + (evidence.unit_evidence[0],)
        )
    missing_parent = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence[:-1], evidence.unit_evidence
    )
    with pytest.raises(SpatialPerturbationSplitError, match="exact perturbation parents"):
        evaluate_bridge_eligibility(manifest, missing_parent)


def test_arbitrary_animal_perturbation_gene_or_unit_identity_is_rejected() -> None:
    manifest, evidence = baseline()
    parent = evidence.parent_evidence[0]
    for forged in (
        replace(parent, animal_id="invented_mouse"),
        replace(parent, perturbation_id="invented_guide"),
        replace(parent, target_gene="InventedGene"),
    ):
        altered = build_bridge_eligibility_evidence(
            manifest, (forged,) + evidence.parent_evidence[1:], evidence.unit_evidence
        )
        with pytest.raises(SpatialPerturbationSplitError, match="frozen parent context"):
            evaluate_bridge_eligibility(manifest, altered)
    forged_unit = replace(evidence.unit_evidence[0], unit_id="a" * 64)
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, (forged_unit,) + evidence.unit_evidence[1:]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="exact frozen primary units"):
        evaluate_bridge_eligibility(manifest, altered)


def test_all_selected_cells_resolve_to_expected_animal_role_band_and_type() -> None:
    manifest, evidence = baseline()
    first = evidence.unit_evidence[0]
    first_frozen = next(unit for unit in manifest.primary_units if unit.unit_id == first.unit_id)
    other_index = next(
        index for index, item in enumerate(evidence.unit_evidence)
        if next(unit for unit in manifest.primary_units if unit.unit_id == item.unit_id).animal_id
        != first_frozen.animal_id
    )
    other = evidence.unit_evidence[other_index]
    first_cell = first.safe_neighbour_cell_ids[0]
    other_cell = other.safe_neighbour_cell_ids[0]
    forged = replace(first, safe_neighbour_cell_ids=(other_cell,) + first.safe_neighbour_cell_ids[1:])
    forged_other = replace(
        other, safe_neighbour_cell_ids=(first_cell,) + other.safe_neighbour_cell_ids[1:]
    )
    units = list(evidence.unit_evidence)
    units[0] = forged
    units[other_index] = forged_other
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, tuple(units)
    )
    with pytest.raises(SpatialPerturbationSplitError, match="expected animal/context/role/type/band"):
        evaluate_bridge_eligibility(manifest, altered)

    source_cell = evidence.parent_evidence[0].perturbation_source_cell_ids[0]
    forged = replace(first, safe_neighbour_cell_ids=(source_cell,) + first.safe_neighbour_cell_ids[1:])
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, (forged,) + evidence.unit_evidence[1:]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="expected animal/context/role/type/band"):
        evaluate_bridge_eligibility(manifest, altered)


def test_non_adjacent_block_requirement_is_derived_from_frozen_provenance() -> None:
    manifest = build_pilot_fold(synthetic_metadata(adjacency=True), "mouse_1")
    evidence = complete_evidence(manifest)
    result = evaluate_bridge_eligibility(manifest, evidence)
    assert result.eligible is False
    assert "insufficient_spatial_blocks" in result.reasons
    assert "insufficient_safe_control_spatial_blocks" in result.reasons


def test_exact_treatment_and_safe_thresholds_are_derived_from_ids() -> None:
    manifest, _ = baseline()
    parent_ids = tuple(parent.parent_id for parent in manifest.perturbation_parents if parent.animal_id == "mouse_1")
    source_19 = complete_evidence(manifest, source_overrides={parent_ids[0]: 19, parent_ids[1]: 19})
    source_result = evaluate_bridge_eligibility(manifest, source_19)
    assert source_result.reason == "insufficient_perturbation_coverage"
    assert ("mouse_1", 3, 5) in source_result.per_animal_perturbation_coverage
    safe_19 = complete_evidence(manifest, safe_source_overrides={parent_ids[0]: 19, parent_ids[1]: 19})
    assert "insufficient_safe_control_coverage" in evaluate_bridge_eligibility(manifest, safe_19).reasons

    exact = complete_evidence(manifest)
    assert evaluate_bridge_eligibility(manifest, exact).eligible is True
    assert (MIN_SOURCE_CELLS, MIN_SAFE_SOURCE_CELLS, MIN_BAND_NEIGHBOURS) == (20, 20, 50)


def _band_unit_ids(
    manifest: BridgeSplitManifest, parent_ids: tuple[str, ...], band: str
) -> tuple[tuple[str, str], ...]:
    contexts = {
        (parent.animal_id, parent.perturbation_id)
        for parent in manifest.perturbation_parents if parent.parent_id in parent_ids
    }
    grouped: list[tuple[str, str]] = []
    for animal, perturbation in sorted(contexts):
        ids = tuple(
            unit.unit_id for unit in manifest.primary_units
            if (unit.animal_id, unit.perturbation_id) == (animal, perturbation)
            and unit.band == band
        )
        assert len(ids) == 2
        grouped.append((ids[0], ids[1]))
    return tuple(grouped)


def test_exact_paired_band_threshold_49_fails_and_50_passes() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )
    pairs = _band_unit_ids(manifest, mouse_1_parents[:2], "proximal")
    below = {unit_id: count for pair in pairs for unit_id, count in zip(pair, (25, 24))}
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, unit_overrides=below)
    )
    assert result.reason == "insufficient_band_neighbours"

    safe_below = {unit_id: count for pair in pairs for unit_id, count in zip(pair, (25, 24))}
    safe_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, safe_unit_overrides=safe_below)
    )
    assert "insufficient_safe_control_band_neighbours" in safe_result.reasons

    exact_pair = _band_unit_ids(manifest, mouse_1_parents[:1], "proximal")[0]
    exact_50 = {unit_id: 25 for unit_id in exact_pair}
    exact_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, unit_overrides=exact_50)
    )
    assert "insufficient_band_neighbours" not in exact_result.reasons


def test_below_thirty_unit_is_retained_as_abstention_not_immediate_failure() -> None:
    manifest, _ = baseline()
    unit = manifest.primary_units[0]
    evidence = complete_evidence(manifest, unit_overrides={unit.unit_id: 29})
    result = evaluate_bridge_eligibility(manifest, evidence)
    assert result.eligible is True
    assert result.abstained_unit_ids == (unit.unit_id,)
    assert "insufficient_band_neighbours" not in result.reasons
    safe_evidence = complete_evidence(manifest, safe_unit_overrides={unit.unit_id: 29})
    safe_result = evaluate_bridge_eligibility(manifest, safe_evidence)
    assert safe_result.eligible is True
    assert safe_result.abstained_unit_ids == (unit.unit_id,)
    assert MIN_CELL_TYPE_NEIGHBOURS == 30


def test_exact_three_pairwise_non_adjacent_blocks_pass_but_two_fail() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )
    source_blocks: dict[str, tuple[str, ...]] = {
        f"{parent_id}:source": ("block_0", "block_1")
        for parent_id in mouse_1_parents[:2]
    }
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, block_overrides=source_blocks)
    )
    assert result.reason == "insufficient_spatial_blocks"

    safe_blocks: dict[str, tuple[str, ...]] = {
        f"{parent_id}:safe_source": ("block_0", "block_1")
        for parent_id in mouse_1_parents[:2]
    }
    safe_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, block_overrides=safe_blocks)
    )
    assert "insufficient_safe_control_spatial_blocks" in safe_result.reasons
    assert evaluate_bridge_eligibility(manifest, complete_evidence(manifest)).eligible is True
    assert MIN_SPATIAL_BLOCKS == 3


def test_primary_coverage_and_abstention_are_derived_at_exact_boundaries() -> None:
    manifest, _ = baseline()
    unit_ids = tuple(unit.unit_id for unit in manifest.primary_units)
    exact_overrides = {unit_id: 29 for unit_id in unit_ids[:12]}
    exact = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, unit_overrides=exact_overrides)
    )
    assert exact.primary_scoreable == 48
    assert exact.primary_total == 60
    assert exact.abstained == 12
    assert exact.attempted == 60
    assert exact.eligible is True

    below_overrides = {unit_id: 29 for unit_id in unit_ids[:13]}
    below = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, unit_overrides=below_overrides)
    )
    assert below.eligible is False
    assert "insufficient_primary_unit_coverage" in below.reasons
    assert "excessive_abstention" in below.reasons
    assert (MIN_COVERAGE, MAX_ABSTENTION) == (0.80, 0.20)


def test_per_animal_perturbation_coverage_exact_four_fifths_passes() -> None:
    manifest, _ = baseline()
    one_parent_each = {
        next(parent.parent_id for parent in manifest.perturbation_parents if parent.animal_id == animal): 19
        for animal in ("mouse_1", "mouse_2", "mouse_3")
    }
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, source_overrides=one_parent_each)
    )
    assert result.per_animal_perturbation_coverage == (
        ("mouse_1", 4, 5), ("mouse_2", 4, 5), ("mouse_3", 4, 5)
    )
    assert result.eligible is True


def test_controls_must_match_animal_section_source_and_neighbour_strata() -> None:
    manifest, evidence = baseline()
    parent = evidence.parent_evidence[0]
    other_parent_safe = evidence.parent_evidence[5].safe_source_cell_ids[0]
    forged_parent = replace(
        parent, safe_source_cell_ids=(other_parent_safe,) + parent.safe_source_cell_ids[1:]
    )
    altered = build_bridge_eligibility_evidence(
        manifest, (forged_parent,) + evidence.parent_evidence[1:], evidence.unit_evidence
    )
    with pytest.raises(SpatialPerturbationSplitError, match="expected animal/context/role/type/band"):
        evaluate_bridge_eligibility(manifest, altered)


def test_manifest_accessors_revalidate_low_level_mutation() -> None:
    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    object.__setattr__(manifest, "train_rows", manifest.train_rows + (manifest.evaluation_rows[0],))
    with pytest.raises(SpatialPerturbationSplitError, match="whole-animal partition"):
        _ = manifest.development_rows


def test_split_and_evaluation_have_no_seed_or_method_escape_hatches() -> None:
    assert tuple(inspect.signature(build_pilot_fold).parameters) == ("metadata", "evaluation_animal")
    assert "model" not in inspect.signature(evaluate_bridge_eligibility).parameters
    random.seed(1907)
    before = random.getstate()
    build_pilot_fold(synthetic_metadata(), "mouse_1")
    assert random.getstate() == before


def test_direct_records_are_closed_immutable_and_digest_mutations_fail() -> None:
    manifest, evidence = baseline()
    with pytest.raises(FrozenInstanceError):
        evidence.split_identity_sha256 = "a" * 64  # type: ignore[misc]
    raw = eligibility_evidence_to_mapping(evidence)
    raw["split_identity_sha256"] = "a" * 64
    raw.pop("evidence_identity_sha256")
    forged = BridgeEligibilityEvidence(
        **raw, evidence_identity_sha256=_canonical_sha(raw)  # type: ignore[arg-type]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="split identity"):
        evaluate_bridge_eligibility(manifest, forged)
    with pytest.raises(TypeError):
        BridgeParentEvidence("a", "p", "G", (), (), extra=True)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", (True, False, -1, 10**100, 1.0, float("nan"), float("inf")))
def test_stable_row_identity_rejects_boolean_nonfinite_and_huge_values(value: object) -> None:
    row = BridgeSplitRow(
        0, "cell", "mouse", "section", "block", "guide", "guide",
        "source", "perturbation_source", "own",
    )
    with pytest.raises(SpatialPerturbationSplitError):
        replace(row, stable_row_id=value)  # type: ignore[arg-type]


def test_rebound_exports_cannot_change_frozen_science(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, evidence = baseline()
    expected = evaluate_bridge_eligibility(manifest, evidence)
    for name, value in (
        ("MIN_SOURCE_CELLS", 0), ("MIN_SAFE_SOURCE_CELLS", 0),
        ("MIN_BAND_NEIGHBOURS", 0), ("MIN_CELL_TYPE_NEIGHBOURS", 0),
        ("MIN_SPATIAL_BLOCKS", 0), ("MIN_COVERAGE", 0.0),
        ("MAX_ABSTENTION", 1.0), ("_SPLIT_SEED", 99),
        ("_PRIMARY_BANDS", ("forged",)),
    ):
        monkeypatch.setattr(split_module, name, value)
    assert build_pilot_fold(synthetic_metadata(), "mouse_1") == manifest
    assert evaluate_bridge_eligibility(manifest, evidence) == expected


def test_manifest_contains_no_outcomes_effects_or_scores() -> None:
    manifest, _ = baseline()
    mapping = split_manifest_to_mapping(manifest)
    assert not ({"outcome", "effect", "score", "prediction", "rmse"} & set(mapping))
