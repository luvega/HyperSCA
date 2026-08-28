from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import overload

import pytest
from hypothesis import given, settings, strategies as st

from src.evaluation.spatial_perturbation_split import (
    BridgeSplitMetadata,
    BridgeSplitRow,
    SpatialPerturbationSplitError,
    build_pilot_fold,
    evaluate_bridge_eligibility,
    split_manifest_to_mapping,
    unit_counts,
)


def _rows() -> tuple[BridgeSplitRow, ...]:
    return tuple(
        BridgeSplitRow(index, f"cell_{index}", animal, f"{animal}_section", f"block_{index % 3}")
        for index, animal in enumerate(("a1", "a1", "a2", "a2", "a3", "a3"))
    )


class _AdversarialSequence(Sequence[object]):
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


@settings(max_examples=24, deadline=None)
@given(st.permutations(_rows()))
def test_input_permutation_preserves_split_identity(permuted: list[BridgeSplitRow]) -> None:
    first = build_pilot_fold(BridgeSplitMetadata(_rows(), ("G2", "G1"), ("p2", "p1")), "a2")
    second = build_pilot_fold(BridgeSplitMetadata(tuple(permuted), ("G1", "G2"), ("p1", "p2")), "a2")
    assert split_manifest_to_mapping(first) == split_manifest_to_mapping(second)


@settings(max_examples=21, deadline=None)
@given(st.sampled_from((19, 20)), st.sampled_from((49, 50)), st.sampled_from((29, 30)), st.sampled_from((2, 3)))
def test_exact_count_boundaries_are_mutation_sensitive(
    source: int, neighbours: int, cell_type_neighbours: int, blocks: int
) -> None:
    result = evaluate_bridge_eligibility(
        unit_counts(source=source, neighbours=neighbours, cell_type_neighbours=cell_type_neighbours, blocks=blocks)
    )
    assert result.eligible is (
        source >= 20 and neighbours >= 50 and cell_type_neighbours >= 30 and blocks >= 3
    )


@settings(max_examples=12, deadline=None)
@given(st.sampled_from(("a1", "a2", "a3")))
def test_whole_animal_isolation(animal: str) -> None:
    split = build_pilot_fold(BridgeSplitMetadata(_rows(), ("G1",), ("p1",)), animal)
    provenance = {row.stable_row_id: row.animal_id for row in split.row_provenance}
    assert {provenance[row] for row in split.evaluation_rows} == {animal}
    assert animal not in {provenance[row] for row in split.development_rows}


@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=1, max_value=5))
def test_duplicate_cell_ids_are_always_rejected(duplicate_index: int) -> None:
    rows = _rows()
    duplicate = replace(rows[duplicate_index], cell_id=rows[0].cell_id)
    with pytest.raises(SpatialPerturbationSplitError, match="cell_id"):
        BridgeSplitMetadata(rows[:duplicate_index] + (duplicate,) + rows[duplicate_index + 1 :], ("G1",), ("p1",))


@settings(max_examples=16, deadline=None)
@given(st.sampled_from(("e\u0301", " x", "x ", "x\x00", "")))
def test_cell_ids_must_be_unique_safe_nfc_text(value: str) -> None:
    rows = _rows()
    with pytest.raises(SpatialPerturbationSplitError):
        replace(rows[0], cell_id=value)


def test_adversarial_sequences_fail_without_access() -> None:
    hostile = _AdversarialSequence()
    with pytest.raises(SpatialPerturbationSplitError):
        BridgeSplitMetadata(hostile, ("G1",), ("p1",))  # type: ignore[arg-type]


def test_deterministic_identity_changes_for_stable_row_identity_mutation() -> None:
    baseline = build_pilot_fold(BridgeSplitMetadata(_rows(), ("G1",), ("p1",)), "a1")
    rows = _rows()
    changed = replace(rows[-1], stable_row_id=99)
    mutated = build_pilot_fold(BridgeSplitMetadata(rows[:-1] + (changed,), ("G1",), ("p1",)), "a1")
    assert baseline.split_identity_sha256 != mutated.split_identity_sha256
