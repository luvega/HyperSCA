from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import cast

import pytest

from src.evaluation.spatial_perturbation_registry import (
    BridgeCandidate,
    BridgeCapabilityResult,
    MetadataSummary,
    SpatialPerturbationRegistryError,
    audit_bridge_capability,
    load_bridge_candidates,
    metadata_summary_from_mapping,
    write_bridge_capability_exclusively,
)
from src.evaluation import spatial_perturbation_registry as registry_module


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/spatial_perturbation_bridge_candidates_v1.json"


def candidate(specimen_count: int = 3) -> BridgeCandidate:
    specimens = tuple(f"mouse_{index}" for index in range(1, specimen_count + 1))
    return BridgeCandidate(
        candidate_id="generic_bridge",
        accession="SYNTEST",
        platform="spatial_perturbation",
        biological_specimens=specimens,
        sections_by_specimen=tuple((item, (f"{item}_section",)) for item in specimens),
        safe_control_label="mSafe",
        perturbation_labels=("guide_a",),
        source_uri="https://example.test/SYNTEST",
        source_identity_sha256="a" * 64,
    )


def metadata_summary(*, animals: int = 5, cohorts: int = 2) -> MetadataSummary:
    specimens = tuple(f"mouse_{index}" for index in range(1, animals + 1))
    return MetadataSummary(
        candidate_id="generic_bridge",
        accession="SYNTEST",
        cohort_ids=tuple(f"cohort_{index}" for index in range(cohorts)),
        biological_specimen_ids=specimens,
        sections_by_specimen=tuple((specimen, (f"{specimen}_section",)) for specimen in specimens),
        block_ids=("block_1",),
        coordinate_available=True,
        coordinate_finite=True,
        coordinate_count=animals,
        measured_gene_names=("GeneA",),
        measured_gene_count=1,
        perturbation_labels=("guide_a",),
        perturbation_label_counts=(("guide_a", animals),),
        safe_control_counts=(("mSafe", animals),),
        barcode_quality_counts=(("valid", animals),),
        label_quality_counts=(("valid", animals),),
        specimen_cohort_assignments=tuple(
            (specimen, f"cohort_{index % cohorts}") for index, specimen in enumerate(specimens)
        ) if cohorts else (),
        external_untouched_cohort_ids=(f"cohort_{cohorts - 1}",) if cohorts else (),
        per_specimen_coordinate_counts=tuple((specimen, 1) for specimen in specimens),
        per_specimen_perturbation_counts=tuple((specimen, 1) for specimen in specimens),
        per_specimen_safe_control_counts=tuple((specimen, 1) for specimen in specimens),
        per_specimen_barcode_valid_counts=tuple((specimen, 1) for specimen in specimens),
        per_specimen_label_valid_counts=tuple((specimen, 1) for specimen in specimens),
        license_identity="CC-BY-4.0",
        source_identity_sha256="a" * 64,
        executable_output_schema_capable=True,
    )


def explicit_metadata_summary(
    *,
    animals: int = 5,
    cohorts: int = 2,
    assignments: tuple[tuple[str, str], ...] | None = None,
    external_cohorts: tuple[str, ...] | None = None,
    covered_specimens: tuple[str, ...] | None = None,
) -> MetadataSummary:
    """A generic, fully structural declaration for the future cohort contract."""
    specimens = tuple(f"mouse_{index}" for index in range(1, animals + 1))
    cohort_ids = tuple(f"cohort_{index}" for index in range(cohorts))
    assignment_values = assignments if assignments is not None else tuple(
        (specimen, cohort_ids[index % len(cohort_ids)]) for index, specimen in enumerate(specimens)
    )
    external_values = external_cohorts if external_cohorts is not None else (cohort_ids[-1],)
    covered = set(covered_specimens if covered_specimens is not None else specimens)

    def evidence() -> tuple[tuple[str, int], ...]:
        return tuple((specimen, int(specimen in covered)) for specimen in specimens)

    total = len(covered)
    return MetadataSummary(
        candidate_id="generic_bridge",
        accession="SYNTEST",
        cohort_ids=cohort_ids,
        biological_specimen_ids=specimens,
        sections_by_specimen=tuple((specimen, (f"{specimen}_section",)) for specimen in specimens),
        block_ids=("block_1",),
        coordinate_available=True,
        coordinate_finite=True,
        coordinate_count=total,
        measured_gene_names=("GeneA",),
        measured_gene_count=1,
        perturbation_labels=("guide_a",),
        perturbation_label_counts=(("guide_a", total),),
        safe_control_counts=(("mSafe", total),),
        barcode_quality_counts=(("valid", total),),
        label_quality_counts=(("valid", total),),
        specimen_cohort_assignments=assignment_values,
        external_untouched_cohort_ids=external_values,
        per_specimen_coordinate_counts=evidence(),
        per_specimen_perturbation_counts=evidence(),
        per_specimen_safe_control_counts=evidence(),
        per_specimen_barcode_valid_counts=evidence(),
        per_specimen_label_valid_counts=evidence(),
        license_identity="CC-BY-4.0",
        source_identity_sha256="a" * 64,
        executable_output_schema_capable=True,
    )


def test_explicit_cohort_assignments_and_external_coverage_are_required() -> None:
    bridge = candidate(5)
    absent_assignments = explicit_metadata_summary(animals=5, cohorts=2, assignments=())
    assert "cohort_identity_coverage_missing" in audit_bridge_capability(bridge, absent_assignments).blocking_reasons

    absent_external = explicit_metadata_summary(animals=5, cohorts=2, external_cohorts=())
    assert "external_cohort_missing" in audit_bridge_capability(bridge, absent_external).blocking_reasons

    orphan_external = explicit_metadata_summary(
        animals=5, cohorts=2, external_cohorts=("cohort_1",),
        assignments=tuple((f"mouse_{index}", "cohort_0") for index in range(1, 6)),
    )
    reasons = audit_bridge_capability(bridge, orphan_external).blocking_reasons
    assert "external_cohort_missing" in reasons

    mixed_external = explicit_metadata_summary(
        animals=5, cohorts=2, external_cohorts=("cohort_1", "invented"),
    )
    assert "external_cohort_missing" in audit_bridge_capability(bridge, mixed_external).blocking_reasons

    all_external = explicit_metadata_summary(
        animals=5, cohorts=2, external_cohorts=("cohort_0", "cohort_1"),
    )
    assert "development_cohort_missing" in audit_bridge_capability(bridge, all_external).blocking_reasons


def test_task5_can_bind_canonical_candidate_and_metadata_identities() -> None:
    bridge = candidate(5)
    summary = explicit_metadata_summary(animals=5, cohorts=2)
    expected_candidate = hashlib.sha256(
        json.dumps(
            bridge.to_mapping(), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_summary = hashlib.sha256(
        json.dumps(
            summary.to_mapping(), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert bridge.candidate_identity_sha256 == expected_candidate
    assert summary.metadata_identity_sha256 == expected_summary

    object.__setattr__(bridge, "perturbation_labels", ())
    assert bridge.candidate_identity_sha256 != expected_candidate


def test_category_specific_aggregate_evidence_cannot_be_hidden_by_other_categories() -> None:
    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    raw["safe_control_counts"] = [["mSafe", 1], ["other_safe", 4]]
    result = audit_bridge_capability(candidate(5), metadata_summary_from_mapping(raw))
    assert result.status == "pilot_audit_only"
    assert "safe_control_coverage_missing" in result.blocking_reasons

    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    raw["barcode_quality_counts"] = [["valid", 1], ["invalid", 4]]
    with pytest.raises(SpatialPerturbationRegistryError, match="valid barcode"):
        metadata_summary_from_mapping(raw)

    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    raw["label_quality_counts"] = [["valid", 1], ["invalid", 4]]
    with pytest.raises(SpatialPerturbationRegistryError, match="valid label"):
        metadata_summary_from_mapping(raw)


@pytest.mark.parametrize(
    "field_reason",
    (
        ("per_specimen_coordinate_counts", "coordinates_unavailable_or_nonfinite"),
        ("per_specimen_perturbation_counts", "perturbation_label_coverage_missing"),
        ("per_specimen_safe_control_counts", "safe_control_coverage_missing"),
        ("per_specimen_barcode_valid_counts", "barcode_quality_coverage_missing"),
        ("per_specimen_label_valid_counts", "label_quality_coverage_missing"),
    ),
)
def test_per_specimen_evidence_cannot_be_replaced_by_a_singleton_aggregate(
    field_reason: tuple[str, str],
) -> None:
    field, reason = field_reason
    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    specimens = tuple(f"mouse_{index}" for index in range(1, 6))
    raw[field] = [[specimens[0], 5]] + [[specimen, 0] for specimen in specimens[1:]]
    result = audit_bridge_capability(candidate(5), metadata_summary_from_mapping(raw))
    assert result.status == "pilot_audit_only"
    assert reason in result.blocking_reasons


def test_coverage_requires_binary_evidence() -> None:
    result = audit_bridge_capability(candidate(), metadata_summary(animals=3))
    coverage = dict(result.coverage)
    coverage["coordinates"] = 5e-324
    unsigned = {
        "candidate_id": result.candidate_id, "status": result.status,
        "confirmatory_capable": result.confirmatory_capable,
        "biological_specimen_count": result.biological_specimen_count,
        "cohort_count": result.cohort_count, "coverage": coverage,
        "blocking_reasons": list(result.blocking_reasons),
    }
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCapabilityResult(**unsigned, capability_identity_sha256=_unsigned_digest(unsigned))  # type: ignore[arg-type]


def test_three_mice_are_pilot_only() -> None:
    result = audit_bridge_capability(candidate(), metadata_summary(animals=3))

    assert result.status == "pilot_audit_only"
    assert result.confirmatory_capable is False
    assert "insufficient_biological_replicates" in result.blocking_reasons
    assert "neighbor_effect_rmse" not in result.to_mapping()


def test_confirmatory_requires_five_specimens_and_two_cohorts() -> None:
    assert not audit_bridge_capability(
        candidate(5), metadata_summary(animals=5, cohorts=1)
    ).confirmatory_capable
    assert audit_bridge_capability(
        candidate(5), metadata_summary(animals=5, cohorts=2)
    ).confirmatory_capable


def test_frozen_registry_declares_only_three_gse274447_mice_without_sections() -> None:
    registry = load_bridge_candidates(REGISTRY_PATH)

    assert tuple(registry) == ("gse274447_msafe_bridge",)
    loaded = registry["gse274447_msafe_bridge"]
    assert loaded.accession == "GSE274447"
    assert loaded.biological_specimens == ("mouse_1", "mouse_2", "mouse_3")
    assert loaded.sections_by_specimen == (
        ("mouse_1", ()),
        ("mouse_2", ()),
        ("mouse_3", ()),
    )
    assert loaded.safe_control_label == "mSafe"


def test_registered_three_mice_cannot_be_upgraded_by_invented_metadata() -> None:
    registry_candidate = load_bridge_candidates(REGISTRY_PATH)["gse274447_msafe_bridge"]
    raw = metadata_summary(animals=5, cohorts=2).to_mapping()
    raw["source_identity_sha256"] = registry_candidate.source_identity_sha256

    result = audit_bridge_capability(registry_candidate, metadata_summary_from_mapping(raw))

    assert result.status == "pilot_audit_only"
    assert result.confirmatory_capable is False
    assert "insufficient_biological_replicates" in result.blocking_reasons
    assert "registered_specimens_mismatch" in result.blocking_reasons


def test_low_level_frozen_candidate_mutation_is_rejected_by_audit_and_writer(tmp_path: Path) -> None:
    frozen = load_bridge_candidates(REGISTRY_PATH)["gse274447_msafe_bridge"]
    specimens = tuple(f"mouse_{index}" for index in range(1, 6))
    for field, value in (
        ("biological_specimens", specimens),
        ("sections_by_specimen", tuple((item, (f"{item}_section",)) for item in specimens)),
        ("perturbation_labels", ("guide_a",)),
    ):
        object.__setattr__(frozen, field, value)
    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    raw.update({
        "candidate_id": frozen.candidate_id,
        "accession": frozen.accession,
        "source_identity_sha256": frozen.source_identity_sha256,
    })
    summary = metadata_summary_from_mapping(raw)
    with pytest.raises(SpatialPerturbationRegistryError, match="declaration"):
        audit_bridge_capability(frozen, summary)
    generic_result = audit_bridge_capability(candidate(), metadata_summary(animals=3))
    with pytest.raises(SpatialPerturbationRegistryError, match="declaration"):
        write_bridge_capability_exclusively(tmp_path / "no.json", generic_result, candidate=frozen)


def test_frozen_source_anchor_alias_cannot_upgrade_or_publish(tmp_path: Path) -> None:
    frozen = load_bridge_candidates(REGISTRY_PATH)["gse274447_msafe_bridge"]
    specimens = tuple(f"mouse_{index}" for index in range(1, 6))
    alias = "alias_bridge"
    for field, value in (
        ("candidate_id", alias),
        ("biological_specimens", specimens),
        ("sections_by_specimen", tuple((item, (f"{item}_section",)) for item in specimens)),
        ("perturbation_labels", ("guide_a",)),
    ):
        object.__setattr__(frozen, field, value)
    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    raw.update({
        "candidate_id": alias,
        "accession": frozen.accession,
        "source_identity_sha256": frozen.source_identity_sha256,
    })
    with pytest.raises(SpatialPerturbationRegistryError, match="declaration"):
        audit_bridge_capability(frozen, metadata_summary_from_mapping(raw))

    unsigned = audit_bridge_capability(candidate(5), metadata_summary(animals=5)).to_mapping()
    unsigned.pop("capability_identity_sha256")
    unsigned["candidate_id"] = alias
    forged = BridgeCapabilityResult(
        **unsigned, capability_identity_sha256=_unsigned_digest(unsigned),
    )  # type: ignore[arg-type]
    output = tmp_path / "alias.json"
    with pytest.raises(SpatialPerturbationRegistryError, match="declaration"):
        write_bridge_capability_exclusively(output, forged, candidate=frozen)
    assert not output.exists()


def test_canonical_gse_source_alias_cannot_upgrade_or_publish(tmp_path: Path) -> None:
    frozen = load_bridge_candidates(REGISTRY_PATH)["gse274447_msafe_bridge"]
    specimens = tuple(f"mouse_{index}" for index in range(1, 6))
    alias = "alias_bridge"
    for field, value in (
        ("candidate_id", alias),
        ("accession", "gse274447"),
        ("source_uri", f"{frozen.source_uri}#same-http-resource"),
        ("source_identity_sha256", "b" * 64),
        ("biological_specimens", specimens),
        ("sections_by_specimen", tuple((item, (f"{item}_section",)) for item in specimens)),
        ("perturbation_labels", ("guide_a",)),
    ):
        object.__setattr__(frozen, field, value)
    raw = explicit_metadata_summary(animals=5, cohorts=2).to_mapping()
    raw.update({"candidate_id": alias, "accession": "gse274447", "source_identity_sha256": "b" * 64})
    with pytest.raises(SpatialPerturbationRegistryError):
        audit_bridge_capability(frozen, metadata_summary_from_mapping(raw))

    unsigned = audit_bridge_capability(candidate(5), metadata_summary(animals=5)).to_mapping()
    unsigned.pop("capability_identity_sha256")
    unsigned["candidate_id"] = alias
    forged = BridgeCapabilityResult(
        **unsigned, capability_identity_sha256=_unsigned_digest(unsigned),
    )  # type: ignore[arg-type]
    output = tmp_path / "canonical-alias.json"
    with pytest.raises(SpatialPerturbationRegistryError):
        write_bridge_capability_exclusively(output, forged, candidate=frozen)
    assert not output.exists()


@pytest.mark.parametrize(
    "accession,source_uri",
    (
        ("gse274447", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447"),
        ("GSE274447", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447#fragment"),
        ("GSE274447", "https://www.ncbi.nlm.nih.gov:443/geo/query/acc.cgi?acc=GSE274447"),
        ("GSE274447", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?x=1&acc=GSE274447"),
        ("SYN1", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447"),
    ),
)
def test_source_identity_rejects_noncanonical_geo_aliases(accession: str, source_uri: str) -> None:
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate(
            "other_candidate", accession, "spatial", (), (), "mSafe", (), source_uri, "a" * 64,
        )


@pytest.mark.parametrize(
    "source_uri",
    (
        "https://example.test/source path",
        "https://example.test/source\u00a0path",
        "https://examplé.test/source",
        "https://example.test./source",
        "https://EXAMPLE.test/source",
        "https://example.test:443/source",
        "https://%65xample.test/source",
        "https://example%2etest/source",
        "https://example.test\\evil/source",
        "https://example.test/{source",
        "https://example.test/a|b",
        "https://example.test/[source]",
    ),
)
def test_source_identity_rejects_noncanonical_generic_https_spellings(source_uri: str) -> None:
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate(
            "synthetic_candidate", "SYN1", "spatial", (), (), "mSafe", (), source_uri, "a" * 64,
        )


def test_writer_binds_result_specimen_count_to_candidate(tmp_path: Path) -> None:
    frozen = load_bridge_candidates(REGISTRY_PATH)["gse274447_msafe_bridge"]
    generic = audit_bridge_capability(candidate(5), metadata_summary(animals=5))
    unsigned = generic.to_mapping()
    unsigned.pop("capability_identity_sha256")
    unsigned["candidate_id"] = frozen.candidate_id
    forged = BridgeCapabilityResult(
        **unsigned,
        capability_identity_sha256=_unsigned_digest(unsigned),
    )  # type: ignore[arg-type]
    output = tmp_path / "forged.json"
    with pytest.raises(SpatialPerturbationRegistryError, match="specimen count"):
        write_bridge_capability_exclusively(output, forged, candidate=frozen)
    assert not output.exists()


def _unsigned_digest(mapping: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(mapping, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def test_result_direct_construction_rejects_semantically_impossible_confirmatory_state() -> None:
    coverage = {key: 1.0 for key in audit_bridge_capability(candidate(), metadata_summary(animals=3)).coverage}
    unsigned = {
        "candidate_id": "gse274447_msafe_bridge", "status": "confirmatory_capable",
        "confirmatory_capable": True, "biological_specimen_count": 3, "cohort_count": 2,
        "coverage": coverage, "blocking_reasons": [],
    }
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCapabilityResult(**unsigned, capability_identity_sha256=_unsigned_digest(unsigned))  # type: ignore[arg-type]


def test_result_direct_construction_requires_every_structural_gate() -> None:
    coverage = {key: 1.0 for key in audit_bridge_capability(candidate(5), metadata_summary(animals=5)).coverage}
    coverage["cohorts"] = 0.0
    unsigned = {
        "candidate_id": "gse274447_msafe_bridge", "status": "confirmatory_capable",
        "confirmatory_capable": True, "biological_specimen_count": 5, "cohort_count": 2,
        "coverage": coverage, "blocking_reasons": [],
    }
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCapabilityResult(**unsigned, capability_identity_sha256=_unsigned_digest(unsigned))  # type: ignore[arg-type]


def test_result_mapping_rejects_mutated_recomputed_impossible_state() -> None:
    result = audit_bridge_capability(candidate(), metadata_summary(animals=3))
    coverage = dict(result.coverage)
    unsigned = {
        "candidate_id": result.candidate_id, "status": "confirmatory_capable",
        "confirmatory_capable": True, "biological_specimen_count": 3, "cohort_count": 2,
        "coverage": coverage, "blocking_reasons": [],
    }
    object.__setattr__(result, "status", "confirmatory_capable")
    object.__setattr__(result, "confirmatory_capable", True)
    object.__setattr__(result, "blocking_reasons", ())
    object.__setattr__(result, "capability_identity_sha256", _unsigned_digest(unsigned))

    with pytest.raises(SpatialPerturbationRegistryError):
        result.to_mapping()


def test_candidate_rejects_global_section_reuse_and_control_label_overlap() -> None:
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate("candidate", "SYN1", "spatial", ("s1", "s2"),
                        (("s1", ("same",)), ("s2", ("same",))), "mSafe", (),
                        "https://example.test/SYN1", "a" * 64)
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate("candidate", "SYN1", "spatial", ("s1",), (("s1", ()),),
                        "mSafe", ("mSafe",), "https://example.test/SYN1", "a" * 64)


def test_metadata_mapping_is_closed_and_outcome_blind() -> None:
    raw = metadata_summary().to_mapping()
    raw["outcome"] = "hostile"

    with pytest.raises(SpatialPerturbationRegistryError, match="unknown"):
        metadata_summary_from_mapping(raw)


def test_result_mapping_is_outcome_blind() -> None:
    mapping = audit_bridge_capability(candidate(), metadata_summary()).to_mapping()
    prohibited = {"expression", "effect", "response", "prediction", "score", "metric", "rmse"}

    assert not any(key.lower() in prohibited for key in mapping)
    assert not any(key.lower() in prohibited for key in mapping["coverage"])


def test_direct_construction_defensively_copies_and_revalidates() -> None:
    raw_specimens = ["mouse_1", "mouse_2", "mouse_3"]
    raw_sections = [("mouse_1", []), ("mouse_2", []), ("mouse_3", [])]
    value = BridgeCandidate(
        "defensive_copy_candidate", "SYNDEF", "spatial_perturbation", raw_specimens,
        raw_sections, "mSafe", ["guide_a"],
        "https://example.test/SYNDEF", "a" * 64,
    )
    raw_specimens.clear()
    raw_sections[0][1].append("changed")

    assert value.biological_specimens == ("mouse_1", "mouse_2", "mouse_3")
    assert value.sections_by_specimen[0] == ("mouse_1", ())
    object.__setattr__(value, "candidate_id", "\x00")
    with pytest.raises(SpatialPerturbationRegistryError):
        audit_bridge_capability(value, metadata_summary())


def test_registry_loader_rejects_duplicate_keys_and_invalid_json_scalars(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1","schema_version":"2"}', encoding="utf-8")
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(duplicate)
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(invalid)


def test_module_has_no_outcome_fields_or_heavy_imports() -> None:
    source = (ROOT / "src/evaluation/spatial_perturbation_registry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"expression", "effect", "response", "prediction", "score", "metric", "rmse"}
    fields = {
        node.arg.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.arg)
    }
    fields.update(
        node.target.id.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )
    assert not fields & forbidden
    probe = subprocess.run(
        [sys.executable, "-c", "import src.evaluation.spatial_perturbation_registry; import sys; assert not set(sys.modules) & {'numpy','pandas','torch','anndata','src.models','src.causal','src.perturbation'}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_cli_help_does_not_import_domain_module() -> None:
    probe = subprocess.run(
        [sys.executable, "scripts/audit_spatial_perturbation_bridge.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert "audit outcome-blind spatial perturbation capability" in probe.stdout


def test_cli_without_assets_publishes_explicit_unavailable_capability(tmp_path: Path) -> None:
    output = tmp_path / "capability.json"
    probe = subprocess.run(
        [
            sys.executable,
            "scripts/audit_spatial_perturbation_bridge.py",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["status"] == "assets_unavailable"
    assert "asset_metadata_unavailable" in published["blocking_reasons"]


def test_cli_default_registry_is_resolved_from_script_root(tmp_path: Path) -> None:
    output = tmp_path / "from_elsewhere.json"
    probe = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_spatial_perturbation_bridge.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_id"] == "gse274447_msafe_bridge"


def test_capability_result_parser_restores_frozen_order_after_canonical_json() -> None:
    expected = audit_bridge_capability(
        candidate(), metadata_summary(animals=3, cohorts=0)
    )
    canonical_json_mapping = json.loads(
        json.dumps(expected.to_mapping(), sort_keys=True)
    )

    with pytest.raises(SpatialPerturbationRegistryError, match="exact ordered"):
        BridgeCapabilityResult(**canonical_json_mapping)

    parsed = registry_module.bridge_capability_result_from_mapping(
        canonical_json_mapping
    )

    assert parsed.to_mapping() == expected.to_mapping()
    assert parsed.capability_identity_sha256 == expected.capability_identity_sha256

    cast(list[str], canonical_json_mapping["blocking_reasons"]).reverse()
    with pytest.raises(SpatialPerturbationRegistryError, match="capability matrix"):
        registry_module.bridge_capability_result_from_mapping(canonical_json_mapping)


@pytest.mark.parametrize(
    "mutation",
    (
        "not_builtin_dict",
        "missing_field",
        "extra_field",
        "bool_specimen_count",
        "integer_capability_flag",
        "missing_coverage_key",
        "extra_coverage_key",
        "integer_coverage_value",
        "tuple_blocking_reasons",
        "wrong_identity",
    ),
)
def test_capability_result_parser_rejects_noncanonical_mappings(mutation: str) -> None:
    raw = audit_bridge_capability(candidate(), metadata_summary(animals=3)).to_mapping()
    if mutation == "not_builtin_dict":
        raw = dict(raw)
        hostile: object = MappingProxyType(raw)
    else:
        hostile = raw
        if mutation == "missing_field":
            raw.pop("status")
        elif mutation == "extra_field":
            raw["extra"] = None
        elif mutation == "bool_specimen_count":
            raw["biological_specimen_count"] = False
        elif mutation == "integer_capability_flag":
            raw["confirmatory_capable"] = 0
        elif mutation == "missing_coverage_key":
            cast(dict[str, object], raw["coverage"]).pop("genes")
        elif mutation == "extra_coverage_key":
            cast(dict[str, object], raw["coverage"])["extra"] = 0.0
        elif mutation == "integer_coverage_value":
            cast(dict[str, object], raw["coverage"])["genes"] = 0
        elif mutation == "tuple_blocking_reasons":
            raw["blocking_reasons"] = tuple(cast(list[str], raw["blocking_reasons"]))
        elif mutation == "wrong_identity":
            raw["capability_identity_sha256"] = "0" * 64

    with pytest.raises(SpatialPerturbationRegistryError):
        registry_module.bridge_capability_result_from_mapping(hostile)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":999999999999999999999999999999999999}',
        b'{"x":' + b"[" * 17 + b"0" + b"]" * 17 + b"}",
        b"\xff\xfe",
    ],
)
def test_registry_loader_rejects_deep_huge_and_invalid_utf8_payloads(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "hostile.json"
    path.write_bytes(payload)
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(path)


def test_registry_loader_rejects_link_and_nonregular_inputs(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(REGISTRY_PATH.read_bytes())
    leaf = tmp_path / "leaf.json"
    leaf.symlink_to(target)
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(leaf)
    linked = tmp_path / "linked.json"
    os.link(target, linked)
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(linked)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "registry.fifo"
        os.mkfifo(fifo)
        with pytest.raises(SpatialPerturbationRegistryError):
            load_bridge_candidates(fifo)


def test_registry_loader_rejects_ancestor_symlink_and_malformed_frozen_candidate(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "registry.json").write_bytes(REGISTRY_PATH.read_bytes())
    link_parent = tmp_path / "linked"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(link_parent / "registry.json")
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    raw["candidates"][0]["candidate_id"] = "changed"
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(malformed)


def test_missing_metadata_through_dangling_symlink_is_a_domain_error(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "metadata_summary.json").symlink_to(asset_root / "missing.json")
    output = tmp_path / "capability.json"
    probe = subprocess.run(
        [sys.executable, "scripts/audit_spatial_perturbation_bridge.py", "--asset-root", str(asset_root), "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert probe.returncode != 0
    assert not output.exists()


def test_exclusive_writer_never_clobbers_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "capability.json"
    output.write_bytes(b"existing")
    result = audit_bridge_capability(candidate(), metadata_summary(animals=3))
    with pytest.raises(SpatialPerturbationRegistryError):
        write_bridge_capability_exclusively(output, result, candidate=candidate())
    assert output.read_bytes() == b"existing"


def test_metadata_and_result_defensively_copy_mutable_mappings() -> None:
    raw_summary = metadata_summary(animals=3).to_mapping()
    summary = metadata_summary_from_mapping(raw_summary)
    raw_summary["biological_specimen_ids"].clear()  # type: ignore[union-attr]
    raw_summary["sections_by_specimen"][0][1].append("forged")  # type: ignore[index,union-attr]
    assert summary.biological_specimen_ids == ("mouse_1", "mouse_2", "mouse_3")
    assert summary.sections_by_specimen[0] == ("mouse_1", ("mouse_1_section",))

    original = audit_bridge_capability(candidate(5), metadata_summary(animals=5))
    raw_result = original.to_mapping()
    copied = BridgeCapabilityResult(**raw_result)  # type: ignore[arg-type]
    raw_result["coverage"]["cohorts"] = 0.0  # type: ignore[index,union-attr]
    raw_result["blocking_reasons"].append("forged")  # type: ignore[union-attr]
    assert copied.capability_identity_sha256 == original.capability_identity_sha256
    assert copied.to_mapping() == original.to_mapping()


def test_registry_ancestor_swap_between_validation_and_leaf_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted" / "nested"
    trusted_parent.mkdir(parents=True)
    registry_path = trusted_parent / "registry.json"
    registry_path.write_bytes(REGISTRY_PATH.read_bytes())
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    (redirected / "registry.json").write_bytes(REGISTRY_PATH.read_bytes())
    holding = tmp_path / "holding"
    real_open = registry_module.os.open
    swapped = False

    def swap_before_leaf(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and str(path).endswith("registry.json"):
            swapped = True
            trusted_parent.rename(holding)
            trusted_parent.symlink_to(redirected, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(registry_module.os, "open", swap_before_leaf)
    with pytest.raises(SpatialPerturbationRegistryError):
        load_bridge_candidates(registry_path)
    assert swapped is True


def test_output_ancestor_swap_is_rejected_without_redirected_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted" / "nested"
    trusted_parent.mkdir(parents=True)
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    output = trusted_parent / "capability.json"
    holding = tmp_path / "holding"
    real_link = registry_module._link_anonymous_noreplace

    def swap_before_publish(anonymous_fd: int, parent_fd: int, target: str) -> None:
        trusted_parent.rename(holding)
        trusted_parent.symlink_to(redirected, target_is_directory=True)
        real_link(anonymous_fd, parent_fd, target)

    monkeypatch.setattr(registry_module, "_link_anonymous_noreplace", swap_before_publish)
    with pytest.raises(SpatialPerturbationRegistryError):
        write_bridge_capability_exclusively(
            output, audit_bridge_capability(candidate(), metadata_summary(animals=3)), candidate=candidate()
        )
    assert not (redirected / "capability.json").exists()


def test_adversarial_replacement_cannot_publish_attacker_staging_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capability.json"
    attacker = tmp_path / "attacker.json"
    attacker_bytes = b'{"attacker":"content"}\n'
    attacker.write_bytes(attacker_bytes)
    real_close = registry_module.os.close
    swapped = False

    def replace_named_staging(descriptor: int) -> None:
        nonlocal swapped
        try:
            staged = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            staged = Path("")
        if not swapped and staged.name.startswith(".capability.json."):
            swapped = True
            real_close(descriptor)
            os.replace(attacker, staged)
            return
        real_close(descriptor)

    monkeypatch.setattr(registry_module.os, "close", replace_named_staging)
    write_bridge_capability_exclusively(
        output, audit_bridge_capability(candidate(), metadata_summary(animals=3)), candidate=candidate()
    )
    assert output.read_bytes() != attacker_bytes
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_id"] == "generic_bridge"


def test_component_walk_closes_descriptor_when_immediate_fstat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "registry.json"
    target.parent.mkdir()
    before = len(os.listdir("/proc/self/fd"))
    real_fstat = registry_module.os.fstat
    injected = False

    def fail_first_component_fstat(descriptor: int):  # type: ignore[no-untyped-def]
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("injected fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(registry_module.os, "fstat", fail_first_component_fstat)
    with pytest.raises(SpatialPerturbationRegistryError):
        registry_module._bound_parent(target, "registry")
    assert injected is True
    assert len(os.listdir("/proc/self/fd")) == before
