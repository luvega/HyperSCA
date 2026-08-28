from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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
        candidate_id="gse274447_msafe_bridge",
        accession="GSE274447",
        platform="spatial_perturbation",
        biological_specimens=specimens,
        sections_by_specimen=tuple((item, (f"{item}_section",)) for item in specimens),
        safe_control_label="mSafe",
        perturbation_labels=("guide_a",),
        source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447",
        source_identity_sha256="a" * 64,
    )


def metadata_summary(*, animals: int = 5, cohorts: int = 2) -> MetadataSummary:
    specimens = tuple(f"mouse_{index}" for index in range(1, animals + 1))
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
        BridgeCandidate("candidate", "GSE1", "spatial", ("s1", "s2"),
                        (("s1", ("same",)), ("s2", ("same",))), "mSafe", (),
                        "https://example.test/GSE1", "a" * 64)
    with pytest.raises(SpatialPerturbationRegistryError):
        BridgeCandidate("candidate", "GSE1", "spatial", ("s1",), (("s1", ()),),
                        "mSafe", ("mSafe",), "https://example.test/GSE1", "a" * 64)


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
    if probe.returncode != 0:
        assert "anonymous exclusive publication" in probe.stderr
        assert not output.exists()
        return
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["status"] == "assets_unavailable"
    assert "asset_metadata_unavailable" in published["blocking_reasons"]


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
        write_bridge_capability_exclusively(output, result)
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
        write_bridge_capability_exclusively(output, audit_bridge_capability(candidate(), metadata_summary(animals=3)))
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
    try:
        write_bridge_capability_exclusively(
            output, audit_bridge_capability(candidate(), metadata_summary(animals=3))
        )
    except SpatialPerturbationRegistryError:
        assert not output.exists()
    else:
        assert output.read_bytes() != attacker_bytes
        assert json.loads(output.read_text(encoding="utf-8"))["candidate_id"] == "gse274447_msafe_bridge"


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
