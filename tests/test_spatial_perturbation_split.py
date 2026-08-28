from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from functools import lru_cache
import hashlib
import inspect
import json
import random
from types import SimpleNamespace
from typing import cast

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


class _HostileSequence:
    def __iter__(self) -> object:
        raise AssertionError("hostile sequence was iterated before validation")


def _canonical_sha(mapping: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(mapping, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def synthetic_metadata(
    *,
    animal_count: int = 3,
    adjacency: bool = False,
    block_count: int = 3,
    neighbour_types: tuple[str, ...] = NEIGHBOUR_TYPES,
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
                for index in range(max(30, block_count * 5)):
                    rows.append(
                        BridgeSplitRow(
                            row_id, f"cell_{row_id:06d}", animal, section,
                            f"block_{index % block_count}", perturbation, label,
                            "source_type", "source_type", role, "own",
                        )
                    )
                    row_id += 1
            neighbour_count = (
                max(50, block_count * 13)
                if len(neighbour_types) == 1
                else max(30, block_count * 10)
            )
            for neighbour_type in neighbour_types:
                for band in BANDS:
                    for role in ("perturbation_neighbour", "safe_neighbour"):
                        for index in range(neighbour_count):
                            rows.append(
                                BridgeSplitRow(
                                    row_id, f"cell_{row_id:06d}", animal, section,
                                    f"block_{index % block_count}", perturbation, "unperturbed",
                                    neighbour_type, "source_type", role, band,
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
        tuple(rows), TARGETS, PERTURBATIONS, neighbour_types,
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
    default_unit_count = 50 if len(manifest.neighbour_cell_types) == 1 else 30
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
                tuple(
                    row.cell_id
                    for row in treatment_rows[
                        : unit_overrides.get(unit.unit_id, default_unit_count)
                    ]
                ),
                tuple(
                    row.cell_id
                    for row in safe_rows[
                        : safe_unit_overrides.get(unit.unit_id, default_unit_count)
                    ]
                ),
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
    assert "insufficient_safe_control_spatial_blocks" not in result.reasons


def test_four_blocks_with_one_adjacent_pair_has_a_valid_three_block_subset() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(
            adjacency=True, block_count=4, neighbour_types=("astrocyte",)
        ),
        "mouse_1",
    )
    result = evaluate_bridge_eligibility(manifest, complete_evidence(manifest))
    assert result.eligible is True


def _cell_ids_with_block_counts(
    manifest: BridgeSplitManifest,
    *,
    animal: str,
    perturbation: str,
    role: str,
    block_counts: tuple[int, ...],
    cell_type: str | None = None,
    band: str | None = None,
) -> tuple[str, ...]:
    selected: list[str] = []
    for block_index, count in enumerate(block_counts):
        rows = _rows_for(
            manifest, animal=animal, perturbation=perturbation, role=role,
            cell_type=cell_type, band=band, blocks=(f"block_{block_index}",),
        )
        selected.extend(row.cell_id for row in rows[:count])
    return tuple(selected)


def test_parent_matching_uses_exact_block_multisets_not_sets() -> None:
    manifest, evidence = baseline()
    parent = evidence.parent_evidence[0]
    assert len(parent.perturbation_source_cell_ids) == 20
    forged_safe = _cell_ids_with_block_counts(
        manifest, animal=parent.animal_id, perturbation=parent.perturbation_id,
        role="safe_source", block_counts=(10, 9, 1),
    )
    forged = replace(parent, safe_source_cell_ids=forged_safe)
    altered = build_bridge_eligibility_evidence(
        manifest, (forged,) + evidence.parent_evidence[1:], evidence.unit_evidence
    )
    with pytest.raises(SpatialPerturbationSplitError, match="multiset"):
        evaluate_bridge_eligibility(manifest, altered)


@pytest.mark.parametrize(
    ("source_count", "safe_count", "expected_reason", "absent_reason"),
    (
        (19, 20, "insufficient_perturbation_coverage", "insufficient_safe_control_coverage"),
        (20, 19, "insufficient_safe_control_coverage", "insufficient_perturbation_coverage"),
    ),
)
def test_asymmetric_parent_thresholds_are_ineligible_before_matching(
    source_count: int,
    safe_count: int,
    expected_reason: str,
    absent_reason: str,
) -> None:
    manifest, _ = baseline()
    parent_ids = tuple(
        parent.parent_id
        for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            source_overrides={parent_id: source_count for parent_id in parent_ids},
            safe_source_overrides={parent_id: safe_count for parent_id in parent_ids},
        ),
    )
    assert expected_reason in result.reasons
    assert absent_reason not in result.reasons


@pytest.mark.parametrize(
    ("treatment_count", "safe_count"),
    ((29, 30), (30, 29)),
)
def test_asymmetric_unit_thresholds_are_retained_as_abstentions(
    treatment_count: int, safe_count: int
) -> None:
    manifest, _ = baseline()
    unit_id = manifest.primary_units[0].unit_id
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            unit_overrides={unit_id: treatment_count},
            safe_unit_overrides={unit_id: safe_count},
        ),
    )
    assert result.eligible is True
    assert result.abstained_unit_ids == (unit_id,)


def test_neighbour_matching_rejects_disjoint_block_multisets() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(block_count=4, neighbour_types=("astrocyte",)), "mouse_1"
    )
    evidence = complete_evidence(manifest)
    item = evidence.unit_evidence[0]
    unit = next(unit for unit in manifest.primary_units if unit.unit_id == item.unit_id)
    treatment = _cell_ids_with_block_counts(
        manifest, animal=unit.animal_id, perturbation=unit.perturbation_id,
        role="perturbation_neighbour", cell_type=unit.neighbour_cell_type,
        band=unit.band, block_counts=(10, 10, 10, 0),
    )
    safe = _cell_ids_with_block_counts(
        manifest, animal=unit.animal_id, perturbation=unit.perturbation_id,
        role="safe_neighbour", cell_type=unit.neighbour_cell_type,
        band=unit.band, block_counts=(0, 10, 10, 10),
    )
    forged = replace(
        item, perturbation_neighbour_cell_ids=treatment, safe_neighbour_cell_ids=safe
    )
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, (forged,) + evidence.unit_evidence[1:]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="multiset"):
        evaluate_bridge_eligibility(manifest, altered)


def test_row_provenance_freezes_source_cell_type() -> None:
    assert "source_cell_type" in {item.name for item in fields(BridgeSplitRow)}


def test_empty_matched_sources_are_threshold_failures_not_provenance_errors() -> None:
    manifest, evidence = baseline()
    empty = tuple(
        replace(
            parent, perturbation_source_cell_ids=(), safe_source_cell_ids=()
        )
        for parent in evidence.parent_evidence[:2]
    )
    altered = build_bridge_eligibility_evidence(
        manifest, empty + evidence.parent_evidence[2:], evidence.unit_evidence
    )
    result = evaluate_bridge_eligibility(manifest, altered)
    assert "insufficient_perturbation_coverage" in result.reasons
    assert "insufficient_safe_control_coverage" in result.reasons


@pytest.mark.parametrize("hostile_field", ("manifest", "evidence"))
def test_public_evaluation_revalidates_hostile_sequences_before_derivation(
    hostile_field: str,
) -> None:
    baseline_manifest, baseline_evidence = baseline()
    manifest = replace(baseline_manifest)
    evidence = replace(baseline_evidence)
    target = manifest if hostile_field == "manifest" else evidence
    field_name = "perturbations" if hostile_field == "manifest" else "parent_evidence"
    object.__setattr__(target, field_name, _HostileSequence())
    with pytest.raises(SpatialPerturbationSplitError):
        evaluate_bridge_eligibility(manifest, evidence)


def test_public_evaluation_normalizes_low_level_empty_perturbations_mutation() -> None:
    baseline_manifest, baseline_evidence = baseline()
    manifest = replace(baseline_manifest)
    object.__setattr__(manifest, "perturbations", ())
    with pytest.raises(SpatialPerturbationSplitError):
        evaluate_bridge_eligibility(manifest, baseline_evidence)


def test_no_module_helper_can_mint_an_arbitrary_trusted_result() -> None:
    manifest, evidence = baseline()
    mutated_manifest = replace(manifest)
    object.__setattr__(mutated_manifest, "perturbations", ("alien",))
    helper = getattr(split_module, "_trusted_eligibility_result", None)
    if helper is not None:
        forged = helper(
            mutated_manifest,
            evidence,
            split_module._DerivedEligibility(
                (), ("alien",), ("alien",), (("alien", 999, 999),),
                999, 999, 0, 999,
            ),
        )
        assert forged.primary_scoreable == 999
        assert forged.scoreable_parent_ids == ("alien",)
        assert forged.manifest.perturbations == ("alien",)
    assert not hasattr(split_module, "_trusted_eligibility_result")


def test_independent_triple_search_has_a_bounded_membership_operation_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_count = 200
    rows = tuple(
        BridgeSplitRow(
            index, f"graph_cell_{index}", "mouse", "section", f"block_{index}",
            "guide", "guide", "source", "source", "perturbation_source", "own",
        )
        for index in range(block_count)
    )
    adjacency = tuple(
        BridgeBlockAdjacency("mouse", "section", f"block_{first}", f"block_{second}")
        for first in range(block_count)
        for second in range(first + 1, block_count)
    )
    real_adjacency = split_module._adjacent_block_pairs

    class BoundedContainsSet(set[frozenset[tuple[str, str, str]]]):
        contains_calls = 0

        def __contains__(self, item: object) -> bool:
            self.contains_calls += 1
            if self.contains_calls > 40_000:
                raise AssertionError("independent-triple search exceeded its operation bound")
            return super().__contains__(item)

    graph_manifest = cast(
        BridgeSplitManifest, SimpleNamespace(block_adjacency=adjacency)
    )
    counted = BoundedContainsSet(real_adjacency(graph_manifest))
    monkeypatch.setattr(split_module, "_adjacent_block_pairs", lambda _: counted)
    assert split_module._has_non_adjacent_block_subset(
        rows, graph_manifest, 3
    ) is False
    assert counted.contains_calls <= 40_000


def test_independent_triple_search_rejects_adversarial_block_cardinality() -> None:
    rows = tuple(
        BridgeSplitRow(
            index, f"huge_graph_cell_{index}", "mouse", "section", f"block_{index}",
            "guide", "guide", "source", "source", "perturbation_source", "own",
        )
        for index in range(5_000)
    )
    graph_manifest = cast(BridgeSplitManifest, SimpleNamespace(block_adjacency=()))
    with pytest.raises(SpatialPerturbationSplitError, match="block graph"):
        split_module._has_non_adjacent_block_subset(
            rows, graph_manifest, 3
        )


def test_exact_treatment_and_safe_thresholds_are_derived_from_ids() -> None:
    manifest, _ = baseline()
    parent_ids = tuple(parent.parent_id for parent in manifest.perturbation_parents if parent.animal_id == "mouse_1")
    below = {parent_ids[0]: 19, parent_ids[1]: 19}
    source_19 = complete_evidence(
        manifest, source_overrides=below, safe_source_overrides=below
    )
    source_result = evaluate_bridge_eligibility(manifest, source_19)
    assert source_result.reason == "insufficient_perturbation_coverage"
    assert ("mouse_1", 3, 5) in source_result.per_animal_perturbation_coverage
    safe_19 = complete_evidence(
        manifest, source_overrides=below, safe_source_overrides=below
    )
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


def locally_abstaining_unit_ids(
    manifest: BridgeSplitManifest, count: int
) -> tuple[str, ...]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for unit in manifest.primary_units:
        grouped.setdefault(
            (unit.animal_id, unit.perturbation_id, unit.band), []
        ).append(unit.unit_id)
    candidates = tuple(sorted(values)[0] for _, values in sorted(grouped.items()))
    return candidates[:count]


def test_exact_paired_band_threshold_49_fails_and_50_passes() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )
    pairs = _band_unit_ids(manifest, mouse_1_parents[:2], "proximal")
    below = {unit_id: count for pair in pairs for unit_id, count in zip(pair, (25, 24))}
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest, unit_overrides=below, safe_unit_overrides=below
        )
    )
    assert result.reason == "insufficient_band_neighbours"

    safe_below = {unit_id: count for pair in pairs for unit_id, count in zip(pair, (25, 24))}
    safe_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest, unit_overrides=safe_below, safe_unit_overrides=safe_below
        )
    )
    assert "insufficient_safe_control_band_neighbours" in safe_result.reasons

    exact_pair = _band_unit_ids(manifest, mouse_1_parents[:1], "proximal")[0]
    exact_50 = {unit_id: 25 for unit_id in exact_pair}
    exact_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest, unit_overrides=exact_50, safe_unit_overrides=exact_50
        )
    )
    assert "insufficient_band_neighbours" not in exact_result.reasons


def test_aggregate_fifty_does_not_rescue_band_with_all_units_abstaining() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    pairs = _band_unit_ids(manifest, mouse_1_parents, "proximal")
    exact_aggregate = {unit_id: 25 for pair in pairs for unit_id in pair}
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest,
            unit_overrides=exact_aggregate,
            safe_unit_overrides=exact_aggregate,
        )
    )
    assert ("mouse_1", 3, 5) in result.per_animal_perturbation_coverage
    assert result.eligible is False
    assert all(unit_id in result.abstained_unit_ids for unit_id in exact_aggregate)


def test_below_thirty_unit_is_retained_as_abstention_not_immediate_failure() -> None:
    manifest, _ = baseline()
    unit = manifest.primary_units[0]
    below = {unit.unit_id: 29}
    evidence = complete_evidence(
        manifest, unit_overrides=below, safe_unit_overrides=below
    )
    result = evaluate_bridge_eligibility(manifest, evidence)
    assert result.eligible is True
    assert result.abstained_unit_ids == (unit.unit_id,)
    assert "insufficient_band_neighbours" not in result.reasons
    safe_evidence = complete_evidence(
        manifest, unit_overrides=below, safe_unit_overrides=below
    )
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
    source_blocks.update(
        {
            f"{parent_id}:safe_source": ("block_0", "block_1")
            for parent_id in mouse_1_parents[:2]
        }
    )
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, block_overrides=source_blocks)
    )
    assert result.reason == "insufficient_spatial_blocks"

    safe_blocks: dict[str, tuple[str, ...]] = {
        f"{parent_id}:source": ("block_0", "block_1")
        for parent_id in mouse_1_parents[:2]
    }
    safe_blocks.update(
        {
            f"{parent_id}:safe_source": ("block_0", "block_1")
            for parent_id in mouse_1_parents[:2]
        }
    )
    safe_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, block_overrides=safe_blocks)
    )
    assert "insufficient_safe_control_spatial_blocks" in safe_result.reasons
    assert evaluate_bridge_eligibility(manifest, complete_evidence(manifest)).eligible is True
    assert MIN_SPATIAL_BLOCKS == 3


def test_primary_coverage_and_abstention_are_derived_at_exact_boundaries() -> None:
    manifest, _ = baseline()
    exact_overrides = {
        unit_id: 29 for unit_id in locally_abstaining_unit_ids(manifest, 12)
    }
    exact = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest,
            unit_overrides=exact_overrides,
            safe_unit_overrides=exact_overrides,
        )
    )
    assert exact.primary_scoreable == 48
    assert exact.primary_total == 60
    assert exact.abstained == 12
    assert exact.attempted == 60
    assert exact.eligible is True

    below_overrides = {
        unit_id: 29 for unit_id in locally_abstaining_unit_ids(manifest, 13)
    }
    below = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest,
            unit_overrides=below_overrides,
            safe_unit_overrides=below_overrides,
        )
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
        manifest,
        complete_evidence(
            manifest,
            source_overrides=one_parent_each,
            safe_source_overrides=one_parent_each,
        ),
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
        "source", "source", "perturbation_source", "own",
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
