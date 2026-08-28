from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.evaluation.run_evidence_identity import (
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import RunEvidencePublisher


def valid_identity(*, unit_record: object | None = None) -> RunEvidenceIdentity:
    if unit_record is None:
        unit_record = {"units": ["sample:block-1"]}
    return RunEvidenceIdentity(
        schema_version="1.0",
        protocol_version="hypersca-methods-v2.1",
        protocol_identity="a" * 64,
        claim_id="spatial",
        benchmark_id="osta_colon",
        data_scopes=("train", "tune"),
        data_split_seed=19911,
        model_seed=11,
        data_split_identity_sha256="b" * 64,
        statistical_unit_schema="osta_platform_sample_block_v1",
        statistical_unit_identity_sha256=canonical_sha256(unit_record),
        analysis_identity_sha256="d" * 64,
        input_identity_sha256="e" * 64,
        config_identity_sha256="f" * 64,
        code_identity_sha256="0" * 64,
        evidence_role="pilot_audit_only",
    )


def begin_publisher(
    tmp_path: Path,
    *,
    required_artifacts: tuple[str, ...] = ("metrics.json", "table.csv"),
    maximum_bundle_bytes: int = 4096,
) -> RunEvidencePublisher:
    unit_record = {"units": ["sample:block-1"]}
    return RunEvidencePublisher.begin(
        output_dir=tmp_path / "bundle",
        identity=valid_identity(unit_record=unit_record),
        statistical_unit_record=unit_record,
        required_artifacts=required_artifacts,
        maximum_bundle_bytes=maximum_bundle_bytes,
    )


def test_publisher_stages_bytes_and_streams_one_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")

    publisher = begin_publisher(tmp_path)
    publisher.add_bytes("metrics.json", b"{}\n", media_type="application/json")
    publisher.add_file("table.csv", source, media_type="text/csv")

    assert not (tmp_path / "bundle").exists()
    assert publisher.state == "staging"
    assert publisher.total_artifact_bytes == len(b"{}\n") + source.stat().st_size
    assert tuple(record.relative_path for record in publisher.artifacts) == (
        "metrics.json",
        "table.csv",
    )
    assert publisher.artifacts[1].sha256 == (
        "492d5ea496056f1a6a6592241032fab764c321596317930b4fa0e1e8bc3b7470"
    )


def test_begin_requires_statistical_record_to_match_identity(tmp_path: Path) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        RunEvidencePublisher.begin(
            output_dir=tmp_path / "bundle",
            identity=valid_identity(unit_record={"units": ["expected"]}),
            statistical_unit_record={"units": ["changed"]},
            required_artifacts=(),
            maximum_bundle_bytes=1,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "relative_path",
    [
        "/absolute.json",
        "../escape.json",
        "nested/../../escape.json",
        ".",
        "",
        "bad\x00name.json",
        "bad\nname.json",
        " e.json",
        "e.json ",
        "e\u0301.json",
    ],
)
def test_invalid_artifact_paths_abort_staging(
    tmp_path: Path, relative_path: str
) -> None:
    publisher = begin_publisher(tmp_path, required_artifacts=())
    staging = publisher.staging_path

    with pytest.raises(RunEvidenceError, match="invalid_artifact"):
        publisher.add_bytes(relative_path, b"x", media_type="text/plain")

    assert publisher.state == "aborted"
    assert not staging.exists()
    assert not (tmp_path / "bundle").exists()


def test_duplicate_artifact_path_aborts_staging(tmp_path: Path) -> None:
    publisher = begin_publisher(tmp_path, required_artifacts=("metrics.json",))
    publisher.add_bytes("metrics.json", b"{}", media_type="application/json")

    with pytest.raises(RunEvidenceError, match="invalid_artifact"):
        publisher.add_bytes("metrics.json", b"changed", media_type="application/json")

    assert publisher.state == "aborted"
    assert not publisher.staging_path.exists()


@pytest.mark.parametrize("bad", [True, 0, -1, 1.0, 2**63])
def test_bundle_limit_requires_positive_bounded_exact_int(
    tmp_path: Path, bad: object
) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        RunEvidencePublisher.begin(
            output_dir=tmp_path / "bundle",
            identity=valid_identity(),
            statistical_unit_record={"units": ["sample:block-1"]},
            required_artifacts=(),
            maximum_bundle_bytes=bad,
        )
    assert list(tmp_path.iterdir()) == []


def test_bundle_limit_is_checked_before_bytes_are_written(tmp_path: Path) -> None:
    publisher = begin_publisher(
        tmp_path, required_artifacts=("metrics.json",), maximum_bundle_bytes=3
    )
    staging = publisher.staging_path

    with pytest.raises(RunEvidenceError, match="invalid_artifact"):
        publisher.add_bytes("metrics.json", b"four", media_type="application/json")

    assert publisher.state == "aborted"
    assert not staging.exists()


def test_add_file_rejects_symlink_and_hardlink_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"source")
    symlink = tmp_path / "source-link.csv"
    symlink.symlink_to(source.name)
    publisher = begin_publisher(tmp_path, required_artifacts=("table.csv",))
    with pytest.raises(RunEvidenceError, match="invalid_artifact"):
        publisher.add_file("table.csv", symlink, media_type="text/csv")
    assert publisher.state == "aborted"

    hardlink = tmp_path / "source-hardlink.csv"
    os.link(source, hardlink)
    second = RunEvidencePublisher.begin(
        output_dir=tmp_path / "second-bundle",
        identity=valid_identity(),
        statistical_unit_record={"units": ["sample:block-1"]},
        required_artifacts=("table.csv",),
        maximum_bundle_bytes=4096,
    )
    with pytest.raises(RunEvidenceError, match="invalid_artifact"):
        second.add_file("table.csv", source, media_type="text/csv")
    assert second.state == "aborted"


def test_source_change_during_stream_copy_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"source")
    publisher = begin_publisher(tmp_path, required_artifacts=("table.csv",))

    import src.evaluation.run_evidence_publisher as publisher_module

    original_copy = publisher_module._copy_regular_file

    def changing_copy(*args: object, **kwargs: object):
        result = original_copy(*args, **kwargs)
        source.write_bytes(b"changed")
        return result

    monkeypatch.setattr(publisher_module, "_copy_regular_file", changing_copy)
    with pytest.raises(RunEvidenceError, match="invalid_artifact"):
        publisher.add_file("table.csv", source, media_type="text/csv")

    assert publisher.state == "aborted"
    assert not publisher.staging_path.exists()


def test_context_manager_cleans_staging_on_exception_and_interrupt(
    tmp_path: Path,
) -> None:
    publisher = begin_publisher(tmp_path, required_artifacts=())
    staging = publisher.staging_path
    with pytest.raises(RuntimeError, match="boom"):
        with publisher:
            raise RuntimeError("boom")
    assert publisher.state == "aborted"
    assert not staging.exists()

    second = RunEvidencePublisher.begin(
        output_dir=tmp_path / "second-bundle",
        identity=valid_identity(),
        statistical_unit_record={"units": ["sample:block-1"]},
        required_artifacts=(),
        maximum_bundle_bytes=4096,
    )
    second_staging = second.staging_path
    with pytest.raises(KeyboardInterrupt):
        with second:
            raise KeyboardInterrupt
    assert second.state == "aborted"
    assert not second_staging.exists()


def test_abort_is_idempotent_but_writing_after_abort_is_rejected(tmp_path: Path) -> None:
    publisher = begin_publisher(tmp_path, required_artifacts=())
    publisher.abort()
    publisher.abort()

    with pytest.raises(RunEvidenceError, match="invalid_state_transition"):
        publisher.add_bytes("late.txt", b"late", media_type="text/plain")
    assert publisher.state == "aborted"


def test_existing_or_symlinked_output_is_rejected_without_modification(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RunEvidenceError, match="publication_conflict"):
        begin_publisher(tmp_path, required_artifacts=())
    assert marker.read_text(encoding="utf-8") == "keep"

    output.rmdir() if not any(output.iterdir()) else None
    marker.unlink()
    output.rmdir()
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(RunEvidenceError, match="publication_conflict"):
        begin_publisher(tmp_path, required_artifacts=())
    assert output.is_symlink()
