"""Closure-record tests for the frozen methods protocol v2.1 pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import src.evaluation.methods_protocol_outcome as protocol_outcome
from src.evaluation.methods_protocol_outcome import (
    ProtocolOutcome,
    load_protocol_outcome,
    strict_json,
    write_protocol_outcome_exclusively,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTCOME_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "methods_protocol_v2_1_audit"
    / "protocol_outcome.json"
)


def test_v21_outcome_binds_the_no_release_audit() -> None:
    outcome = load_protocol_outcome(OUTCOME_PATH)

    assert outcome.protocol_version == "hypersca-methods-v2.1"
    assert outcome.status == "pilot_failed_no_release"
    assert outcome.release_authorized is False
    assert outcome.pilot_summary_sha256 == (
        "3fe9e90443f82a911fe02314a540cd8e3383ee016cff9c3dbb46b802490d694c"
    )
    assert len(outcome.run_identity_sha256) == 18
    assert len(set(outcome.run_identity_sha256)) == 18
    assert len(outcome.collection_identity_sha256) == 6


def test_v21_outcome_cannot_be_relabelled_as_release() -> None:
    payload = strict_json(OUTCOME_PATH)
    payload["release_authorized"] = True

    with pytest.raises(ValueError, match="no-release"):
        ProtocolOutcome.from_mapping(payload)


def test_direct_construction_freezes_mutable_identity_inputs() -> None:
    loaded = load_protocol_outcome(OUTCOME_PATH)
    runs = list(loaded.run_identity_sha256)
    collections = list(loaded.collection_identity_sha256)
    reasons = list(loaded.blocking_reasons)

    outcome = ProtocolOutcome(
        protocol_version=loaded.protocol_version,
        protocol_identity_sha256=loaded.protocol_identity_sha256,
        pilot_summary_sha256=loaded.pilot_summary_sha256,
        status=loaded.status,
        release_authorized=loaded.release_authorized,
        run_identity_sha256=runs,
        collection_identity_sha256=collections,
        blocking_reasons=reasons,
    )
    runs.reverse()
    collections.reverse()
    reasons.append("tampered")

    assert outcome.run_identity_sha256 == loaded.run_identity_sha256
    assert outcome.collection_identity_sha256 == loaded.collection_identity_sha256
    assert outcome.blocking_reasons == loaded.blocking_reasons
    assert type(outcome.run_identity_sha256) is tuple
    assert type(outcome.collection_identity_sha256) is tuple
    assert type(outcome.blocking_reasons) is tuple
    with pytest.raises((AttributeError, TypeError)):
        outcome.status = "release_authorized"  # type: ignore[misc]


def test_strict_json_rejects_duplicate_keys_and_unsafe_files(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"a","status":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        strict_json(duplicate)

    regular = tmp_path / "regular.json"
    regular.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.hardlink_to(regular)
    with pytest.raises(ValueError, match="hard-linked"):
        strict_json(linked)


def test_strict_json_rejects_deep_json_after_an_escaped_quote(tmp_path: Path) -> None:
    deep = tmp_path / "deep.json"
    deep.write_bytes(
        b'{"marker":'
        + json.dumps('"').encode("utf-8")
        + b',"deep":'
        + (b"[" * 10_000)
        + b"0"
        + (b"]" * 10_000)
        + b"}"
    )

    with pytest.raises(ValueError, match="too deeply nested"):
        strict_json(deep)


def test_strict_json_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from src.evaluation.methods_protocol_outcome import strict_json; "
                "import sys; "
                "\ntry:\n strict_json(Path(sys.argv[1]))\n"
                "except ValueError:\n raise SystemExit(0)\n"
                "raise SystemExit(1)"
            ),
            str(fifo),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr


def test_freeze_cli_help_does_not_require_evaluation_imports(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "freeze_methods_protocol_outcome.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--pilot-summary" in completed.stdout


def test_outcome_publication_refuses_to_clobber_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)

    write_protocol_outcome_exclusively(output, outcome)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_protocol_outcome_exclusively(output, outcome)


def test_outcome_publication_preserves_preexisting_staging_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_open = protocol_outcome.os.open
    staging_entries: list[Path] = []

    def create_staging_collision(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if flags & os.O_EXCL:
            if dir_fd is None:
                descriptor = original_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            else:
                descriptor = original_open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    mode,
                    dir_fd=dir_fd,
                )
            try:
                os.write(descriptor, b"preexisting staging entry")
            finally:
                os.close(descriptor)
            name = os.fsdecode(path)
            staging_entries.append(
                Path(name) if os.path.isabs(name) else output.parent / name
            )
            raise FileExistsError("simulated staging collision")
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(protocol_outcome.os, "open", create_staging_collision)

    with pytest.raises((FileExistsError, ValueError)):
        write_protocol_outcome_exclusively(output, outcome)

    assert staging_entries
    assert all(entry.read_bytes() == b"preexisting staging entry" for entry in staging_entries)


def test_outcome_publication_rejects_replaced_staging_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_link = protocol_outcome.os.link
    staging: Path | None = None

    def replace_staging_then_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal staging
        source_name = os.fsdecode(source)
        source_dir_fd = kwargs.get("src_dir_fd")
        if source_dir_fd is None:
            staging = Path(source_name)
            os.unlink(source_name)
            os.symlink("/etc/passwd", source_name)
        else:
            assert type(source_dir_fd) is int
            staging = output.parent / source_name
            os.unlink(source_name, dir_fd=source_dir_fd)
            os.symlink("/etc/passwd", source_name, dir_fd=source_dir_fd)
        original_link(source, destination, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(protocol_outcome.os, "link", replace_staging_then_link)

    with pytest.raises(ValueError):
        write_protocol_outcome_exclusively(output, outcome)

    assert staging is not None and staging.is_symlink()
    assert not os.path.lexists(output)


def test_outcome_publication_verifies_the_final_link_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)

    def forge_final_file(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        destination_dir_fd = kwargs.get("dst_dir_fd")
        if destination_dir_fd is None:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        else:
            assert type(destination_dir_fd) is int
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_dir_fd,
            )
        try:
            os.write(descriptor, b"forged publication")
        finally:
            os.close(descriptor)

    monkeypatch.setattr(protocol_outcome.os, "link", forge_final_file)

    with pytest.raises(ValueError, match="inode"):
        write_protocol_outcome_exclusively(output, outcome)

    assert not os.path.lexists(output)


def test_outcome_publication_removes_output_after_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_fsync = protocol_outcome.os.fsync
    fsync_calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("simulated directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(protocol_outcome.os, "fsync", fail_directory_fsync)

    with pytest.raises(ValueError, match="fsync"):
        write_protocol_outcome_exclusively(output, outcome)

    assert fsync_calls >= 2
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
