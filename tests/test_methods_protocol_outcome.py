"""Closure-record tests for the frozen methods protocol v2.1 pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

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
