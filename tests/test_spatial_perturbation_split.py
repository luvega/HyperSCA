from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
import json
import random
from typing import Any

import pytest

from src.evaluation.spatial_perturbation_split import (
    MAX_ABSTENTION,
    MIN_BAND_NEIGHBOURS,
    MIN_CELL_TYPE_NEIGHBOURS,
    MIN_COVERAGE,
    MIN_SAFE_SOURCE_CELLS,
    MIN_SOURCE_CELLS,
    MIN_SPATIAL_BLOCKS,
    BridgeEligibilityEvidence,
    BridgeEligibilityResult,
    BridgeSplitManifest,
    BridgeSplitMetadata,
    BridgeSplitRow,
    SpatialPerturbationSplitError,
    build_pilot_fold,
    eligibility_evidence_to_mapping,
    evaluate_bridge_eligibility,
    split_manifest_to_mapping,
    unit_counts,
)
from src.evaluation import spatial_perturbation_split as split_module


def metadata(rows: tuple[BridgeSplitRow, ...] | None = None) -> BridgeSplitMetadata:
    values = rows or tuple(
        BridgeSplitRow(
            stable_row_id=index,
            cell_id=f"cell_{index}",
            animal_id=animal,
            section_id=section,
            spatial_block=block,
        )
        for index, (animal, section, block) in enumerate(
            (
                ("mouse_1", "mouse_1_s1", "b1"),
                ("mouse_1", "mouse_1_s2", "b2"),
                ("mouse_2", "mouse_2_s1", "b1"),
                ("mouse_2", "mouse_2_s1", "b2"),
                ("mouse_3", "mouse_3_s1", "b1"),
                ("mouse_3", "mouse_3_s1", "b3"),
            )
        )
    )
    return BridgeSplitMetadata(values, ("GeneB", "GeneA"), ("guide_b", "guide_a"))


def _canonical_sha(mapping: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            mapping, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def test_fold_keeps_sections_with_their_animal() -> None:
    split = build_pilot_fold(metadata(), evaluation_animal="mouse_1")

    assert split.development_animals == ("mouse_2", "mouse_3")
    assert split.evaluation_animals == ("mouse_1",)
    assert split.development_rows == (2, 3, 4, 5)
    assert split.evaluation_rows == (0, 1)
    assert split.tune_rows == ()
    assert not set(split.development_rows) & set(split.evaluation_rows)
    assert split.split_seed == 11
    assert split.gene_names == ("GeneA", "GeneB")
    assert split.perturbations == ("guide_a", "guide_b")


def test_split_api_has_no_model_or_variable_seed() -> None:
    parameters = inspect.signature(build_pilot_fold).parameters
    assert tuple(parameters) == ("metadata", "evaluation_animal")
    assert "model_seed" not in parameters
    assert "split_seed" not in parameters


def test_evaluation_animal_must_be_an_exact_registered_member() -> None:
    with pytest.raises(SpatialPerturbationSplitError, match="registered animal"):
        build_pilot_fold(metadata(), evaluation_animal="Mouse_1")
    with pytest.raises(SpatialPerturbationSplitError, match="registered animal"):
        build_pilot_fold(metadata(), evaluation_animal="mouse_4")


def test_repeated_or_cross_animal_sections_and_duplicate_cells_are_rejected() -> None:
    original = metadata().rows
    duplicate_cell = replace(original[-1], cell_id=original[0].cell_id)
    with pytest.raises(SpatialPerturbationSplitError, match="cell_id"):
        BridgeSplitMetadata(original[:-1] + (duplicate_cell,), ("GeneA",), ("p1",))

    ambiguous = replace(original[-1], section_id=original[0].section_id)
    with pytest.raises(SpatialPerturbationSplitError, match="section.*animal"):
        BridgeSplitMetadata(original[:-1] + (ambiguous,), ("GeneA",), ("p1",))

    repeated_row = replace(original[-1], stable_row_id=original[0].stable_row_id)
    with pytest.raises(SpatialPerturbationSplitError, match="stable_row_id"):
        BridgeSplitMetadata(original[:-1] + (repeated_row,), ("GeneA",), ("p1",))


def test_manifest_is_closed_immutable_and_has_no_outcomes() -> None:
    split = build_pilot_fold(metadata(), evaluation_animal="mouse_1")
    expected = {
        "split_id", "split_seed", "development_animals", "evaluation_animals",
        "train_rows", "tune_rows", "evaluation_rows", "gene_names", "perturbations",
        "split_identity_sha256", "row_provenance",
    }
    assert {item.name for item in fields(BridgeSplitManifest)} == expected
    assert set(split_manifest_to_mapping(split)) == expected
    assert not ({"outcome", "effect", "score", "rmse"} & set(split_manifest_to_mapping(split)))
    with pytest.raises(FrozenInstanceError):
        split.split_seed = 12  # type: ignore[misc]
    with pytest.raises(TypeError):
        BridgeSplitManifest(**split_manifest_to_mapping(split), unexpected=True)  # type: ignore[arg-type,call-arg]


def test_direct_manifest_construction_revalidates_provenance_and_digest() -> None:
    split = build_pilot_fold(metadata(), evaluation_animal="mouse_1")
    raw = split_manifest_to_mapping(split)
    raw["split_seed"] = 12
    unsigned = dict(raw)
    unsigned.pop("split_identity_sha256")
    raw["split_identity_sha256"] = _canonical_sha(unsigned)
    with pytest.raises(SpatialPerturbationSplitError, match="seed"):
        BridgeSplitManifest(**raw)  # type: ignore[arg-type]

    raw = split_manifest_to_mapping(split)
    raw["train_rows"] = list(split.train_rows) + [split.evaluation_rows[0]]
    unsigned = dict(raw)
    unsigned.pop("split_identity_sha256")
    raw["split_identity_sha256"] = _canonical_sha(unsigned)
    with pytest.raises(SpatialPerturbationSplitError, match="animal partition"):
        BridgeSplitManifest(**raw)  # type: ignore[arg-type]


def test_split_preserves_input_and_global_random_state() -> None:
    source = metadata()
    original = source
    random.seed(914)
    before = random.getstate()
    build_pilot_fold(source, evaluation_animal="mouse_2")
    after = random.getstate()
    assert source == original
    assert before == after


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"source": 19}, "insufficient_perturbation_coverage"),
        ({"safe_source": 19}, "insufficient_safe_control_coverage"),
        ({"neighbours": 49}, "insufficient_band_neighbours"),
        ({"safe_neighbours": 49}, "insufficient_safe_control_band_neighbours"),
        ({"cell_type_neighbours": 29}, "insufficient_band_neighbours"),
        ({"safe_cell_type_neighbours": 29}, "insufficient_safe_control_band_neighbours"),
        ({"blocks": 2}, "insufficient_spatial_blocks"),
        ({"safe_blocks": 2}, "insufficient_safe_control_spatial_blocks"),
        ({"target_gene_present": False}, "target_gene_not_measurable"),
        ({"perturbation_scoreable": 799_999, "perturbation_total": 1_000_000}, "insufficient_perturbation_coverage"),
        ({"primary_scoreable": 799_999, "primary_total": 1_000_000}, "insufficient_primary_unit_coverage"),
        ({"abstained": 200_001, "attempted": 1_000_000}, "excessive_abstention"),
    ),
)
def test_exact_frozen_thresholds_reject_values_below_or_above_boundary(
    kwargs: dict[str, Any], reason: str
) -> None:
    result = evaluate_bridge_eligibility(unit_counts(**kwargs))
    assert result.eligible is False
    assert result.reason == reason


def test_exact_frozen_threshold_boundaries_pass() -> None:
    assert (
        MIN_SOURCE_CELLS,
        MIN_SAFE_SOURCE_CELLS,
        MIN_BAND_NEIGHBOURS,
        MIN_CELL_TYPE_NEIGHBOURS,
        MIN_SPATIAL_BLOCKS,
        MIN_COVERAGE,
        MAX_ABSTENTION,
    ) == (20, 20, 50, 30, 3, 0.80, 0.20)
    evidence = unit_counts(
        source=20, safe_source=20, neighbours=50, safe_neighbours=50,
        cell_type_neighbours=30, safe_cell_type_neighbours=30,
        blocks=3, safe_blocks=3, perturbation_scoreable=4, perturbation_total=5,
        primary_scoreable=4, primary_total=5, abstained=1, attempted=5,
    )
    result = evaluate_bridge_eligibility(evidence)
    assert result.eligible is True
    assert result.reason is None
    assert result.reasons == ()


def test_both_primary_bands_and_paired_safe_evidence_are_mandatory() -> None:
    evidence = unit_counts()
    object.__setattr__(evidence, "band_evidence", evidence.band_evidence[:1])
    with pytest.raises(SpatialPerturbationSplitError, match="primary bands"):
        evaluate_bridge_eligibility(evidence)

    result = evaluate_bridge_eligibility(unit_counts(safe_neighbours=49))
    assert result.reason == "insufficient_safe_control_band_neighbours"


def test_eligibility_is_method_independent_and_contains_no_outcomes() -> None:
    assert "method" not in inspect.signature(evaluate_bridge_eligibility).parameters
    evidence = unit_counts()
    assert "method" not in {item.name for item in fields(BridgeEligibilityEvidence)}
    result = evaluate_bridge_eligibility(evidence)
    assert "method" not in {item.name for item in fields(BridgeEligibilityResult)}
    mapping = eligibility_evidence_to_mapping(evidence)
    assert not ({"outcome", "effect", "score", "prediction", "rmse"} & set(mapping))


def test_mutated_or_recomputed_evidence_digest_fails_closed() -> None:
    evidence = unit_counts()
    object.__setattr__(evidence, "target_gene", "DifferentGene")
    with pytest.raises(SpatialPerturbationSplitError, match="identity"):
        evaluate_bridge_eligibility(evidence)

    evidence = unit_counts()
    raw = eligibility_evidence_to_mapping(evidence)
    raw["primary_total"] = 0
    raw.pop("evidence_identity_sha256")
    with pytest.raises(SpatialPerturbationSplitError):
        BridgeEligibilityEvidence(
            **raw, evidence_identity_sha256=_canonical_sha(raw)  # type: ignore[arg-type]
        )


def test_result_carries_defensive_evidence_and_rejects_recomputed_forgery() -> None:
    evidence = unit_counts(source=19)
    result = evaluate_bridge_eligibility(evidence)
    assert result.evidence == evidence
    assert result.evidence is not evidence

    unsigned: dict[str, object] = {
        "eligible": True,
        "reason": None,
        "reasons": [],
        "evidence": eligibility_evidence_to_mapping(evidence),
    }
    with pytest.raises(SpatialPerturbationSplitError, match="does not match evidence"):
        BridgeEligibilityResult(
            **unsigned, eligibility_identity_sha256=_canonical_sha(unsigned)  # type: ignore[arg-type]
        )


def test_rebound_scientific_constants_cannot_change_frozen_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_split = build_pilot_fold(metadata(), "mouse_1")
    baseline_result = evaluate_bridge_eligibility(unit_counts(source=19))
    monkeypatch.setattr(split_module, "MIN_SOURCE_CELLS", 0)
    monkeypatch.setattr(split_module, "MIN_SAFE_SOURCE_CELLS", 0)
    monkeypatch.setattr(split_module, "MIN_BAND_NEIGHBOURS", 0)
    monkeypatch.setattr(split_module, "MIN_CELL_TYPE_NEIGHBOURS", 0)
    monkeypatch.setattr(split_module, "MIN_SPATIAL_BLOCKS", 0)
    monkeypatch.setattr(split_module, "MIN_COVERAGE", 0.0)
    monkeypatch.setattr(split_module, "MAX_ABSTENTION", 1.0)
    monkeypatch.setattr(split_module, "_PRIMARY_BANDS", ("forged",))
    monkeypatch.setattr(split_module, "_SPLIT_SEED", 99)

    assert build_pilot_fold(metadata(), "mouse_1") == baseline_split
    assert evaluate_bridge_eligibility(unit_counts(source=19)) == baseline_result


def test_recomputed_digest_cannot_double_count_cells_across_paired_bands() -> None:
    evidence = unit_counts()
    raw = eligibility_evidence_to_mapping(evidence)
    proximal = raw["band_evidence"][0]  # type: ignore[index]
    local = raw["band_evidence"][1]  # type: ignore[index]
    local["perturbation_neighbour_cell_ids"] = list(  # type: ignore[index]
        proximal["perturbation_neighbour_cell_ids"]  # type: ignore[index]
    )
    raw.pop("evidence_identity_sha256")
    with pytest.raises(SpatialPerturbationSplitError, match="across primary bands"):
        BridgeEligibilityEvidence(
            **raw, evidence_identity_sha256=_canonical_sha(raw)  # type: ignore[arg-type]
        )

    raw = eligibility_evidence_to_mapping(evidence)
    first_band = raw["band_evidence"][0]  # type: ignore[index]
    first_band["safe_neighbour_cell_ids"] = list(  # type: ignore[index]
        first_band["perturbation_neighbour_cell_ids"]  # type: ignore[index]
    )
    raw.pop("evidence_identity_sha256")
    with pytest.raises(SpatialPerturbationSplitError, match="treatment and safe"):
        BridgeEligibilityEvidence(
            **raw, evidence_identity_sha256=_canonical_sha(raw)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", (True, False, -1, 10**100, 1.0, float("nan"), float("inf")))
def test_counts_reject_boolean_non_finite_negative_and_huge_values(value: object) -> None:
    with pytest.raises(SpatialPerturbationSplitError):
        unit_counts(source=value)  # type: ignore[arg-type]
