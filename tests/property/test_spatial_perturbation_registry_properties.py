from __future__ import annotations

from collections.abc import Iterator, Sequence
import math
from pathlib import Path
from typing import overload

import pytest
from hypothesis import given, settings, strategies as st

from src.evaluation.spatial_perturbation_registry import (
    BridgeCandidate,
    MetadataSummary,
    SpatialPerturbationRegistryError,
    audit_bridge_capability,
    metadata_summary_from_mapping,
)


def _candidate(specimens: int = 5) -> BridgeCandidate:
    ids = tuple(f"s{index}" for index in range(1, specimens + 1))
    return BridgeCandidate(
        "candidate", "GSE1", "spatial", ids,
        tuple((value, (f"{value}_sec",)) for value in ids),
        "mSafe", ("p1",), "https://example.test/GSE1", "a" * 64,
    )


def _summary(specimens: int = 5, cohorts: int = 2) -> MetadataSummary:
    ids = tuple(f"s{index}" for index in range(1, specimens + 1))
    cohort_ids = tuple(f"c{index}" for index in range(cohorts))
    return MetadataSummary(
        "candidate", "GSE1", cohort_ids, ids,
        tuple((value, (f"{value}_sec",)) for value in ids), ("b1",), True, True,
        specimens, ("G1",), 1, ("p1",), (("p1", specimens),), (("mSafe", specimens),),
        (("valid", specimens),), (("valid", specimens),),
        tuple((value, cohort_ids[index % cohorts]) for index, value in enumerate(ids)) if cohorts else (),
        (cohort_ids[-1],) if cohorts else (),
        tuple((value, 1) for value in ids), tuple((value, 1) for value in ids),
        tuple((value, 1) for value in ids), tuple((value, 1) for value in ids),
        tuple((value, 1) for value in ids), "CC-BY-4.0", "a" * 64, True,
    )


class _ExplodingSequence(Sequence[object]):
    @overload
    def __getitem__(self, index: int) -> object: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...
    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        raise AssertionError("custom sequence was used")
    def __len__(self) -> int:
        raise AssertionError("custom sequence was measured")
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("custom sequence was iterated")


@settings(max_examples=12, deadline=None)
@given(st.sampled_from((4, 5)), st.sampled_from((1, 2)))
def test_exact_specimen_and_cohort_thresholds(specimens: int, cohorts: int) -> None:
    result = audit_bridge_capability(_candidate(specimens), _summary(specimens, cohorts))
    assert result.confirmatory_capable is (specimens >= 5 and cohorts >= 2)


@settings(max_examples=12, deadline=None)
@given(st.sampled_from((math.nan, math.inf, -math.inf)))
def test_metadata_rejects_non_finite_coverage_values(value: float) -> None:
    raw = _summary().to_mapping()
    raw["coordinate_available"] = value
    with pytest.raises(SpatialPerturbationRegistryError):
        metadata_summary_from_mapping(raw)


@settings(max_examples=12, deadline=None)
@given(st.sampled_from(("\x00", "e\u0301", " x", "x ")))
def test_candidate_rejects_unsafe_text(value: str) -> None:
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate(value, "GSE1", "spatial", (), (), "mSafe", (), "https://example.test", "a" * 64)


def test_custom_sequences_are_rejected_without_iteration() -> None:
    hostile = _ExplodingSequence()
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate("candidate", "GSE1", "spatial", hostile, (), "mSafe", (), "https://example.test", "a" * 64)  # type: ignore[arg-type]


def test_deterministic_identity_and_role_boolean_consistency() -> None:
    first = audit_bridge_capability(_candidate(), _summary())
    second = audit_bridge_capability(_candidate(), _summary())
    assert first.capability_identity_sha256 == second.capability_identity_sha256
    assert first.confirmatory_capable is (first.status == "confirmatory_capable")


def test_duplicate_identifiers_and_forbidden_mapping_keys_are_rejected() -> None:
    raw = _summary().to_mapping()
    raw["biological_specimen_ids"] = ["s1", "s1"]
    with pytest.raises(SpatialPerturbationRegistryError):
        metadata_summary_from_mapping(raw)


def test_registry_loader_bounds_hostile_file_size(tmp_path: Path) -> None:
    from src.evaluation.spatial_perturbation_registry import load_bridge_candidates

    path = tmp_path / "too_large.json"
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(path)
    raw = _summary().to_mapping()
    raw["RMSE"] = 0
    with pytest.raises(SpatialPerturbationRegistryError):
        metadata_summary_from_mapping(raw)
