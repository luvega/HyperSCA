from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from importlib import import_module
import inspect
import random
import sys
from typing import Any, overload

import pytest
from hypothesis import given, settings, strategies as st

from src.evaluation.spatial_perturbation_split import (
    BridgeParentEvidence,
    BridgeSplitMetadata,
    BridgeSplitRow,
    SpatialPerturbationSplitError,
    build_pilot_fold,
    evaluate_bridge_eligibility,
    split_manifest_to_mapping,
)

_contract_helpers: Any = sys.modules.get("test_spatial_perturbation_split")
if _contract_helpers is None:
    _contract_helpers = import_module("tests.test_spatial_perturbation_split")
baseline: Any = _contract_helpers.baseline
complete_evidence: Any = _contract_helpers.complete_evidence
locally_abstaining_unit_ids: Any = _contract_helpers.locally_abstaining_unit_ids


def _small_metadata(animal_count: int = 3) -> BridgeSplitMetadata:
    return _contract_helpers.synthetic_metadata(animal_count=animal_count)


class _HostileSequence(Sequence[object]):
    @overload
    def __getitem__(self, index: int) -> object: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...
    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        raise AssertionError("custom sequence was accessed")
    def __len__(self) -> int:
        raise AssertionError("custom sequence was measured")
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("custom sequence was iterated")


@settings(max_examples=6, deadline=None)
@given(st.integers(min_value=0, max_value=5))
def test_three_animal_split_identity_is_invariant_to_input_permutation(
    rotation: int,
) -> None:
    metadata = _small_metadata()
    offset = rotation % len(metadata.rows)
    permuted = metadata.rows[offset:] + metadata.rows[:offset]
    relation_offset = rotation % len(metadata.neighbour_relations)
    permuted_relations = (
        metadata.neighbour_relations[relation_offset:]
        + metadata.neighbour_relations[:relation_offset]
    )
    first = build_pilot_fold(metadata, "mouse_2")
    second = build_pilot_fold(
        BridgeSplitMetadata(
            tuple(permuted), tuple(reversed(metadata.gene_names)),
            tuple(reversed(metadata.perturbations)), tuple(reversed(metadata.neighbour_cell_types)),
            tuple(reversed(metadata.perturbation_targets)),
            tuple(reversed(metadata.block_adjacency)), "mSafe", permuted_relations,
            metadata.neighbour_table_identity_sha256, metadata.candidate,
            metadata.registry_summary, metadata.capability_result,
        ),
        "mouse_2",
    )
    assert split_manifest_to_mapping(first) == split_manifest_to_mapping(second)
    assert {row.animal_id for row in first.row_provenance if row.stable_row_id in first.evaluation_rows} == {"mouse_2"}


@settings(max_examples=4, deadline=None)
@given(st.sampled_from((1, 2, 3, 4)))
def test_exactly_three_animals_is_a_mutation_sensitive_property(animal_count: int) -> None:
    if animal_count == 3:
        assert build_pilot_fold(_small_metadata(animal_count), "mouse_1").evaluation_animals == ("mouse_1",)
    else:
        with pytest.raises(SpatialPerturbationSplitError, match="exactly three animals"):
            build_pilot_fold(_small_metadata(animal_count), "mouse_1")


@settings(max_examples=8, deadline=None)
@given(st.integers(min_value=1, max_value=20))
def test_generated_duplicate_cell_id_sequences_are_rejected(index: int) -> None:
    metadata = _small_metadata()
    rows = list(metadata.rows)
    rows[index] = replace(rows[index], cell_id=rows[0].cell_id)
    with pytest.raises(SpatialPerturbationSplitError, match="cell_id"):
        BridgeSplitMetadata(
            rows, metadata.gene_names, metadata.perturbations, metadata.neighbour_cell_types,  # type: ignore[arg-type]
            metadata.perturbation_targets, metadata.block_adjacency, "mSafe",
            metadata.neighbour_relations, metadata.neighbour_table_identity_sha256,
            metadata.candidate, metadata.registry_summary, metadata.capability_result,
        )


@settings(max_examples=5, deadline=None)
@given(st.sampled_from(("e\u0301", " x", "x ", "x\x00", "")))
def test_generated_unsafe_cell_ids_fail_closed(value: str) -> None:
    with pytest.raises(SpatialPerturbationSplitError):
        replace(_small_metadata().rows[0], cell_id=value)


@settings(max_examples=2, deadline=None)
@given(st.sampled_from(("mouse_2", "mouse_3")))
def test_repeated_section_within_animal_is_valid_but_cross_animal_is_not(
    other_animal: str,
) -> None:
    metadata = _small_metadata()
    assert len({row.section_id for row in metadata.rows if row.animal_id == "mouse_1"}) == 1
    rows = list(metadata.rows)
    cross = next(index for index, row in enumerate(rows) if row.animal_id == other_animal)
    rows[cross] = replace(rows[cross], section_id="mouse_1_section")
    with pytest.raises(SpatialPerturbationSplitError, match="section.*two animals"):
        BridgeSplitMetadata(
            rows, metadata.gene_names, metadata.perturbations, metadata.neighbour_cell_types,  # type: ignore[arg-type]
            metadata.perturbation_targets, metadata.block_adjacency, "mSafe",
            metadata.neighbour_relations, metadata.neighbour_table_identity_sha256,
            metadata.candidate, metadata.registry_summary, metadata.capability_result,
        )


@settings(max_examples=10, deadline=None)
@given(st.lists(st.sampled_from(("c1", "c2", "c3")), min_size=2, max_size=8))
def test_generated_parent_cell_sequences_reject_duplicates(values: list[str]) -> None:
    if len(values) == len(set(values)):
        BridgeParentEvidence("a1", "p1", "G1", tuple(values), ("s1",))
    else:
        with pytest.raises(SpatialPerturbationSplitError, match="unique"):
            BridgeParentEvidence("a1", "p1", "G1", tuple(values), ("s1",))


def test_custom_sequences_are_rejected_without_iteration() -> None:
    hostile = _HostileSequence()
    metadata = _small_metadata()
    with pytest.raises(SpatialPerturbationSplitError):
        BridgeSplitMetadata(
            hostile, metadata.gene_names, metadata.perturbations,
            metadata.neighbour_cell_types, metadata.perturbation_targets,
            metadata.block_adjacency, "mSafe", metadata.neighbour_relations,
            metadata.neighbour_table_identity_sha256, metadata.candidate,
            metadata.registry_summary, metadata.capability_result,
        )  # type: ignore[arg-type]


@settings(max_examples=3, deadline=None)
@given(st.sampled_from(("mouse_1", "mouse_2", "mouse_3")))
def test_split_seed_is_independent_and_no_model_seed_exists(evaluation_animal: str) -> None:
    signature = inspect.signature(build_pilot_fold)
    assert tuple(signature.parameters) == ("metadata", "evaluation_animal")
    random.seed(731)
    before = random.getstate()
    assert build_pilot_fold(_small_metadata(), evaluation_animal).split_seed == 11
    assert random.getstate() == before


@settings(max_examples=2, deadline=None)
@given(st.sampled_from((19, 20)))
def test_exact_source_and_safe_thresholds_are_mutation_sensitive(
    count: int,
) -> None:
    manifest, _ = baseline()
    parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    overrides = {parent_id: count for parent_id in parents}
    evidence = complete_evidence(
        manifest,
        source_overrides=overrides,
        safe_source_overrides=overrides,
    )
    result = evaluate_bridge_eligibility(manifest, evidence)
    assert result.eligible is (count == 20)


@settings(max_examples=2, deadline=None)
@given(st.sampled_from((1, 2)))
def test_exact_per_animal_perturbation_coverage_boundary(failed_parents: int) -> None:
    manifest, _ = baseline()
    parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:failed_parents]
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            source_overrides={parent_id: 19 for parent_id in parents},
            safe_source_overrides={parent_id: 19 for parent_id in parents},
        ),
    )
    expected_scoreable = 5 - failed_parents
    assert ("mouse_1", expected_scoreable, 5) in result.per_animal_perturbation_coverage
    assert result.eligible is (failed_parents == 1)


@settings(max_examples=2, deadline=None)
@given(st.sampled_from((12, 13)))
def test_exact_primary_coverage_and_abstention_boundary(abstained: int) -> None:
    manifest, _ = baseline()
    unit_ids = locally_abstaining_unit_ids(manifest, abstained)
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            unit_overrides={unit_id: 29 for unit_id in unit_ids},
            safe_unit_overrides={unit_id: 29 for unit_id in unit_ids},
        ),
    )
    assert result.abstained == abstained
    assert result.primary_scoreable == 60 - abstained
    assert result.eligible is (abstained == 12)


@settings(max_examples=3, deadline=None)
@given(st.sampled_from(("guide_0", "guide_2", "guide_4")))
def test_holdout_only_perturbations_are_structurally_secondary(
    holdout_only: str,
) -> None:
    metadata = _small_metadata()
    rows = tuple(
        row for row in metadata.rows
        if not (row.animal_id in {"mouse_2", "mouse_3"} and row.context_perturbation_id == holdout_only)
    )
    relations = tuple(
        item for item in metadata.neighbour_relations
        if not (
            item.animal_id in {"mouse_2", "mouse_3"}
            and item.matched_perturbation_id == holdout_only
        )
    )
    pruned = BridgeSplitMetadata(
        rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, "mSafe", relations,
        import_module("src.evaluation.spatial_perturbation_split")._neighbour_table_identity(relations),
        metadata.candidate, metadata.registry_summary, metadata.capability_result,
    )
    manifest = build_pilot_fold(pruned, "mouse_1")
    assert holdout_only not in manifest.perturbations
    assert holdout_only in manifest.secondary_perturbations


@settings(max_examples=3, deadline=None)
@given(st.sampled_from(("guide_0", "guide_2", "guide_4")))
def test_development_only_perturbations_are_not_secondary(
    development_only: str,
) -> None:
    metadata = _small_metadata()
    rows = tuple(
        row for row in metadata.rows
        if not (
            row.animal_id == "mouse_1"
            and row.context_perturbation_id == development_only
        )
    )
    relations = tuple(
        item for item in metadata.neighbour_relations
        if not (
            item.animal_id == "mouse_1"
            and item.matched_perturbation_id == development_only
        )
    )
    module = import_module("src.evaluation.spatial_perturbation_split")
    pruned = BridgeSplitMetadata(
        rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label, relations,
        module._neighbour_table_identity(relations), metadata.candidate,
        metadata.registry_summary, metadata.capability_result,
    )
    manifest = build_pilot_fold(pruned, "mouse_1")
    assert development_only in manifest.development_only_perturbations
    assert development_only not in manifest.secondary_perturbations


@settings(max_examples=5, deadline=None)
@given(st.sampled_from(("guide_0", "guide_1", "guide_2", "guide_3", "guide_4")))
def test_registered_perturbation_requires_at_least_one_atomic_source(
    deleted: str,
) -> None:
    metadata = _small_metadata()
    rows = tuple(
        row for row in metadata.rows
        if row.context_perturbation_id != deleted
    )
    relations = tuple(
        item for item in metadata.neighbour_relations
        if item.matched_perturbation_id != deleted
    )
    module = import_module("src.evaluation.spatial_perturbation_split")
    with pytest.raises(SpatialPerturbationSplitError, match="registered.*source"):
        BridgeSplitMetadata(
            rows, metadata.gene_names, metadata.perturbations,
            metadata.neighbour_cell_types, metadata.perturbation_targets,
            metadata.block_adjacency, metadata.safe_control_label, relations,
            module._neighbour_table_identity(relations), metadata.candidate,
            metadata.registry_summary, metadata.capability_result,
        )


@settings(max_examples=3, deadline=None)
@given(st.sampled_from(("guide_0", "guide_1", "guide_4")))
def test_registry_binding_cannot_be_bypassed_by_upstream_perturbation_deletion(
    deleted: str,
) -> None:
    metadata = _small_metadata()
    raw_candidate = metadata.candidate.to_mapping()
    raw_candidate["perturbation_labels"] = [
        item for item in metadata.candidate.perturbation_labels if item != deleted
    ]
    candidate = type(metadata.candidate)(**raw_candidate)
    registry = import_module("src.evaluation.spatial_perturbation_registry")
    capability = registry.audit_bridge_capability(candidate, metadata.registry_summary)
    with pytest.raises(SpatialPerturbationSplitError, match="registry declaration"):
        BridgeSplitMetadata(
            metadata.rows, metadata.gene_names, metadata.perturbations,
            metadata.neighbour_cell_types, metadata.perturbation_targets,
            metadata.block_adjacency, metadata.safe_control_label,
            metadata.neighbour_relations, metadata.neighbour_table_identity_sha256,
            candidate, metadata.registry_summary, capability,
        )


@settings(max_examples=4, deadline=None)
@given(st.integers(min_value=2, max_value=5))
def test_safe_neighbour_cells_may_be_reused_by_distinct_relation_keys(
    reuse_count: int,
) -> None:
    metadata = _small_metadata()
    safe_relations = tuple(
        item for item in metadata.neighbour_relations
        if item.is_safe_control
        and item.animal_id == "mouse_1"
        and item.neighbor_cell_type == metadata.neighbour_cell_types[0]
        and item.band == "proximal"
    )
    reused_id = safe_relations[0].neighbor_cell_id
    reused = tuple(item for item in safe_relations if item.neighbor_cell_id == reused_id)
    assert len(reused[:reuse_count]) == reuse_count
    assert len({item.relation_id for item in reused[:reuse_count]}) == reuse_count
    assert {
        item.matched_perturbation_id for item in reused[:reuse_count]
    } <= set(metadata.perturbations)
    assert {item.source_perturbation_id for item in reused[:reuse_count]} == {"mSafe"}


@settings(max_examples=4, deadline=None)
@given(st.integers(min_value=0, max_value=4))
def test_generic_partition_identity_is_invariant_to_animal_order(rotation: int) -> None:
    module = import_module("src.evaluation.spatial_perturbation_split")
    metadata = _small_metadata(animal_count=5)
    development = ("mouse_1", "mouse_2", "mouse_3")
    evaluation = ("mouse_4", "mouse_5")
    shifted_development = development[rotation % 3:] + development[:rotation % 3]
    shifted_evaluation = evaluation[rotation % 2:] + evaluation[:rotation % 2]
    first = module.build_bridge_partition_manifest(
        metadata, "generic_partition:property", "generic",
        ("mouse_1", "mouse_2"), ("mouse_3",), evaluation,
    )
    second = module.build_bridge_partition_manifest(
        metadata, "generic_partition:property", "generic",
        tuple(item for item in shifted_development if item != "mouse_3"),
        ("mouse_3",), shifted_evaluation,
    )
    assert split_manifest_to_mapping(first) == split_manifest_to_mapping(second)


@settings(max_examples=2, deadline=None)
@given(st.sampled_from(("mouse_1", "mouse_4")))
def test_confirmatory_cohort_mismatch_is_rejected(mutated_tune: str) -> None:
    module = import_module("src.evaluation.spatial_perturbation_split")
    metadata = _contract_helpers.synthetic_metadata(
        animal_count=5, neighbour_types=("astrocyte",), confirmatory=True,
    )
    if mutated_tune == "mouse_1":
        manifest = module.build_bridge_partition_manifest(
            metadata, "confirmatory_partition:property", "confirmatory",
            ("mouse_1", "mouse_2"), ("mouse_3",), ("mouse_4", "mouse_5"),
        )
        assert manifest.capability_result.confirmatory_capable
    else:
        with pytest.raises(SpatialPerturbationSplitError, match="external.*cohort"):
            module.build_bridge_partition_manifest(
                metadata, "confirmatory_partition:property", "confirmatory",
                ("mouse_1", "mouse_2"), (mutated_tune,),
                ("mouse_3", "mouse_5"),
            )


@settings(max_examples=4, deadline=None)
@given(st.integers(min_value=3, max_value=6))
def test_complete_block_graph_has_exact_pair_count(block_count: int) -> None:
    metadata = _contract_helpers.synthetic_metadata(block_count=block_count)
    per_animal_pairs = block_count * (block_count - 1) // 2
    assert len(metadata.block_adjacency) == 3 * per_animal_pairs
    assert all(type(item.adjacent) is bool for item in metadata.block_adjacency)


@settings(max_examples=3, deadline=None)
@given(st.sampled_from((64, 96, 128)))
def test_cartesian_limit_is_checked_by_bounded_integer_arithmetic(
    neighbour_type_count: int,
) -> None:
    module = import_module("src.evaluation.spatial_perturbation_split")
    checker = getattr(module, "_checked_cartesian_size", None)
    assert callable(checker)
    with pytest.raises(SpatialPerturbationSplitError, match="size|limit"):
        checker(3, 256, neighbour_type_count, 2)
