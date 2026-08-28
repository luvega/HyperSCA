"""Closure-record tests for the frozen methods protocol v2.1 pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
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
    original_rename = getattr(protocol_outcome, "_rename_noreplace", None)
    staging: Path | None = None

    def replace_staging_then_publish(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal staging
        staging = output.parent / source_name
        os.unlink(source_name, dir_fd=parent_fd)
        os.symlink("/etc/passwd", source_name, dir_fd=parent_fd)
        assert original_rename is not None
        original_rename(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        protocol_outcome,
        "_rename_noreplace",
        replace_staging_then_publish,
        raising=False,
    )

    with pytest.raises(ValueError):
        write_protocol_outcome_exclusively(output, outcome)

    assert staging is not None
    assert output.is_symlink()


def test_outcome_publication_verifies_the_final_link_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)

    def forge_final_file(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        descriptor = os.open(
            target_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.write(descriptor, b"forged publication")
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        protocol_outcome, "_rename_noreplace", forge_final_file, raising=False
    )

    with pytest.raises(ValueError, match="inode"):
        write_protocol_outcome_exclusively(output, outcome)

    assert output.read_bytes() == b"forged publication"


def test_outcome_publication_preserves_output_after_directory_fsync_failure(
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
    assert output.exists()


def test_outcome_publication_detects_destination_swap_during_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_fsync = protocol_outcome.os.fsync
    fsync_calls = 0

    def fsync_then_swap_destination(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        original_fsync(descriptor)
        if fsync_calls == 2:
            output.unlink()
            output.write_bytes(b"forged bytes after directory fsync")

    monkeypatch.setattr(protocol_outcome.os, "fsync", fsync_then_swap_destination)

    with pytest.raises(ValueError, match="inode"):
        write_protocol_outcome_exclusively(output, outcome)

    assert fsync_calls == 2
    assert output.read_bytes() == b"forged bytes after directory fsync"


def test_outcome_publication_detects_parent_replacement_after_precommit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "bound"
    parent.mkdir()
    displaced_parent = tmp_path / "displaced"
    output = parent / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_rename = protocol_outcome._rename_noreplace

    def replace_parent_then_publish(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        parent.rename(displaced_parent)
        parent.mkdir()
        original_rename(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        protocol_outcome, "_rename_noreplace", replace_parent_then_publish
    )

    with pytest.raises(ValueError, match="parent changed"):
        write_protocol_outcome_exclusively(output, outcome)

    assert not output.exists()
    assert load_protocol_outcome(displaced_parent / output.name) == outcome


def test_outcome_publication_preserves_staging_replaced_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_rename = getattr(protocol_outcome, "_rename_noreplace", None)

    def replace_staging_then_publish(parent_fd: int, source_name: str, target_name: str) -> None:
        os.unlink(source_name, dir_fd=parent_fd)
        descriptor = os.open(
            source_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.write(descriptor, b"unowned staging replacement")
        finally:
            os.close(descriptor)
        assert original_rename is not None
        original_rename(parent_fd, source_name, target_name)

    monkeypatch.setattr(
        protocol_outcome,
        "_rename_noreplace",
        replace_staging_then_publish,
        raising=False,
    )

    with pytest.raises(ValueError):
        write_protocol_outcome_exclusively(output, outcome)

    assert output.read_bytes() == b"unowned staging replacement"


def test_outcome_publication_preserves_destination_swapped_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_rename = getattr(protocol_outcome, "_rename_noreplace", None)

    def publish_then_replace_destination(
        parent_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        assert original_rename is not None
        original_rename(parent_fd, source_name, target_name)
        os.unlink(target_name, dir_fd=parent_fd)
        descriptor = os.open(
            target_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.write(descriptor, b"unowned destination")
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        protocol_outcome,
        "_rename_noreplace",
        publish_then_replace_destination,
        raising=False,
    )

    with pytest.raises(ValueError):
        write_protocol_outcome_exclusively(output, outcome)

    assert output.read_bytes() == b"unowned destination"


def test_outcome_publication_rechecks_ancestor_symlinks_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = tmp_path / "container"
    ancestor = container / "ancestor"
    terminal = ancestor / "terminal"
    terminal.mkdir(parents=True)
    output = terminal / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_fsync = protocol_outcome.os.fsync
    replacement_done = False

    def insert_equivalent_ancestor_symlink(descriptor: int) -> None:
        nonlocal replacement_done
        original_fsync(descriptor)
        if not replacement_done:
            real_ancestor = container / "real_ancestor"
            ancestor.rename(real_ancestor)
            ancestor.symlink_to(real_ancestor, target_is_directory=True)
            replacement_done = True

    monkeypatch.setattr(protocol_outcome.os, "fsync", insert_equivalent_ancestor_symlink)

    with pytest.raises(ValueError, match="symbolic link"):
        write_protocol_outcome_exclusively(output, outcome)


def test_late_publication_failure_never_rolls_back_without_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "protocol_outcome.json"
    outcome = load_protocol_outcome(OUTCOME_PATH)
    original_parent_match = protocol_outcome._parent_path_matches_identity
    parent_match_calls = 0
    original_fsync = protocol_outcome.os.fsync
    fsync_calls = 0

    def fail_final_parent_check(path: Path, identity: tuple[int, int]) -> bool:
        nonlocal parent_match_calls
        parent_match_calls += 1
        return original_parent_match(path, identity)

    def count_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        original_fsync(descriptor)

    original_close = protocol_outcome.os.close

    def fail_after_commit(descriptor: int) -> None:
        if fsync_calls >= 2 and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated late descriptor cleanup failure")
        original_close(descriptor)

    monkeypatch.setattr(protocol_outcome, "_parent_path_matches_identity", fail_final_parent_check)
    monkeypatch.setattr(protocol_outcome.os, "fsync", count_fsync)
    monkeypatch.setattr(protocol_outcome.os, "close", fail_after_commit)

    with pytest.raises(ValueError, match="committed"):
        write_protocol_outcome_exclusively(output, outcome)

    assert parent_match_calls >= 2
    assert fsync_calls >= 2
    assert load_protocol_outcome(output) == outcome
