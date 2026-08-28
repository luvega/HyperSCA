from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.spatial_perturbation_registry import (
    BridgeCandidate,
    MetadataSummary,
    SpatialPerturbationRegistryError,
    audit_bridge_capability,
    load_bridge_candidates,
    metadata_summary_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/spatial_perturbation_bridge_candidates_v1.json"


def candidate() -> BridgeCandidate:
    return BridgeCandidate(
        candidate_id="gse274447_msafe_bridge",
        accession="GSE274447",
        platform="spatial_perturbation",
        biological_specimens=("mouse_1", "mouse_2", "mouse_3"),
        sections_by_specimen=(("mouse_1", ()), ("mouse_2", ()), ("mouse_3", ())),
        safe_control_label="mSafe",
        perturbation_labels=("guide_a",),
        source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447",
        source_identity_sha256="a" * 64,
    )


def metadata_summary(*, animals: int = 5, cohorts: int = 2) -> MetadataSummary:
    specimens = tuple(f"mouse_{index}" for index in range(animals))
    return MetadataSummary(
        candidate_id="gse274447_msafe_bridge",
        accession="GSE274447",
        cohort_ids=tuple(f"cohort_{index}" for index in range(cohorts)),
        biological_specimen_ids=specimens,
        sections_by_specimen=tuple((specimen, (f"{specimen}_section",)) for specimen in specimens),
        block_ids=("block_1",),
        coordinate_available=True,
        coordinate_finite=True,
        measured_gene_names=("GeneA",),
        measured_gene_count=1,
        perturbation_labels=("guide_a",),
        perturbation_label_counts=(("guide_a", 4),),
        safe_control_counts=(("mSafe", 4),),
        barcode_quality_counts=(("valid", 8),),
        label_quality_counts=(("valid", 8),),
        license_identity="CC-BY-4.0",
        source_identity_sha256="a" * 64,
        executable_output_schema_capable=True,
    )


def test_three_mice_are_pilot_only() -> None:
    result = audit_bridge_capability(candidate(), metadata_summary(animals=3))

    assert result.status == "pilot_audit_only"
    assert result.confirmatory_capable is False
    assert "insufficient_biological_replicates" in result.blocking_reasons
    assert "neighbor_effect_rmse" not in result.to_mapping()


def test_confirmatory_requires_five_specimens_and_two_cohorts() -> None:
    assert not audit_bridge_capability(
        candidate(), metadata_summary(animals=5, cohorts=1)
    ).confirmatory_capable
    assert audit_bridge_capability(
        candidate(), metadata_summary(animals=5, cohorts=2)
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
        "gse274447_msafe_bridge", "GSE274447", "spatial_perturbation", raw_specimens,
        raw_sections, "mSafe", ["guide_a"],
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447", "a" * 64,
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
