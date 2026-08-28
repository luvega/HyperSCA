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
    rows: list[BridgeSplitRow] = []
    row_id = 0
    for animal_index in range(animal_count):
        animal = f"a{animal_index + 1}"
        for perturbation_index in range(5):
            perturbation = f"p{perturbation_index}"
            for block in range(3):
                for role, label, cell_type, band in (
                    ("perturbation_source", perturbation, "source", "own"),
                    ("safe_source", "mSafe", "source", "own"),
                    ("perturbation_neighbour", "unperturbed", "neighbour", "proximal"),
                    ("safe_neighbour", "unperturbed", "neighbour", "proximal"),
                    ("perturbation_neighbour", "unperturbed", "neighbour", "local"),
                    ("safe_neighbour", "unperturbed", "neighbour", "local"),
                ):
                    rows.append(
                        BridgeSplitRow(
                            row_id, f"c{row_id}", animal, f"{animal}_section", f"b{block}",
                            perturbation, label, cell_type, "source", role, band,
                        )
                    )
                    row_id += 1
    return BridgeSplitMetadata(
        tuple(rows), tuple(f"G{i}" for i in range(5)), tuple(f"p{i}" for i in range(5)),
        ("neighbour",), tuple((f"p{i}", f"G{i}") for i in range(5)), (), "mSafe",
    )


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
@given(st.permutations(_small_metadata().rows))
def test_three_animal_split_identity_is_invariant_to_input_permutation(
    permuted: list[BridgeSplitRow],
) -> None:
    metadata = _small_metadata()
    first = build_pilot_fold(metadata, "a2")
    second = build_pilot_fold(
        BridgeSplitMetadata(
            tuple(permuted), tuple(reversed(metadata.gene_names)),
            tuple(reversed(metadata.perturbations)), tuple(reversed(metadata.neighbour_cell_types)),
            tuple(reversed(metadata.perturbation_targets)), (), "mSafe",
        ),
        "a2",
    )
    assert split_manifest_to_mapping(first) == split_manifest_to_mapping(second)
    assert {row.animal_id for row in first.row_provenance if row.stable_row_id in first.evaluation_rows} == {"a2"}


@settings(max_examples=4, deadline=None)
@given(st.sampled_from((1, 2, 3, 4)))
def test_exactly_three_animals_is_a_mutation_sensitive_property(animal_count: int) -> None:
    if animal_count == 3:
        assert build_pilot_fold(_small_metadata(animal_count), "a1").evaluation_animals == ("a1",)
    else:
        with pytest.raises(SpatialPerturbationSplitError, match="exactly three animals"):
            build_pilot_fold(_small_metadata(animal_count), "a1")


@settings(max_examples=8, deadline=None)
@given(st.integers(min_value=1, max_value=20))
def test_generated_duplicate_cell_id_sequences_are_rejected(index: int) -> None:
    metadata = _small_metadata()
    rows = list(metadata.rows)
    rows[index] = replace(rows[index], cell_id=rows[0].cell_id)
    with pytest.raises(SpatialPerturbationSplitError, match="cell_id"):
        BridgeSplitMetadata(
            rows, metadata.gene_names, metadata.perturbations, metadata.neighbour_cell_types,  # type: ignore[arg-type]
            metadata.perturbation_targets, (), "mSafe",
        )


@settings(max_examples=5, deadline=None)
@given(st.sampled_from(("e\u0301", " x", "x ", "x\x00", "")))
def test_generated_unsafe_cell_ids_fail_closed(value: str) -> None:
    with pytest.raises(SpatialPerturbationSplitError):
        replace(_small_metadata().rows[0], cell_id=value)


@settings(max_examples=2, deadline=None)
@given(st.sampled_from(("a2", "a3")))
def test_repeated_section_within_animal_is_valid_but_cross_animal_is_not(
    other_animal: str,
) -> None:
    metadata = _small_metadata()
    assert len({row.section_id for row in metadata.rows if row.animal_id == "a1"}) == 1
    rows = list(metadata.rows)
    cross = next(index for index, row in enumerate(rows) if row.animal_id == other_animal)
    rows[cross] = replace(rows[cross], section_id="a1_section")
    with pytest.raises(SpatialPerturbationSplitError, match="section.*two animals"):
        BridgeSplitMetadata(
            rows, metadata.gene_names, metadata.perturbations, metadata.neighbour_cell_types,  # type: ignore[arg-type]
            metadata.perturbation_targets, (), "mSafe",
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
    with pytest.raises(SpatialPerturbationSplitError):
        BridgeSplitMetadata(hostile, ("G1",), ("p1",), ("n",), (("p1", "G1"),), (), "mSafe")  # type: ignore[arg-type]


@settings(max_examples=3, deadline=None)
@given(st.sampled_from(("a1", "a2", "a3")))
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
