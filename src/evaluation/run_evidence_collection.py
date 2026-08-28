"""Paired-seed closure and invalidation records for verified run evidence."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import unicodedata

from src.evaluation.run_evidence_identity import (
    MAX_EXACT_INTEGER,
    RunEvidenceError,
    canonical_json_bytes,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import (
    VerifiedRunEvidence,
    verify_run_evidence_bundle,
)


@dataclass(frozen=True, slots=True)
class PairedEvidenceCollection:
    runs: tuple[VerifiedRunEvidence, ...]
    expected_model_seeds: tuple[int, ...]
    collection_identity_sha256: str
    statistics_allowed: bool


def _fail(message: str) -> None:
    raise RunEvidenceError("paired_identity_mismatch", message)


def _exact_seed_tuple(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or not value or len(value) > 10_000:
        _fail("expected_model_seeds must be a non-empty bounded exact tuple")
    result: list[int] = []
    for seed in value:
        if type(seed) is not int or seed < 0 or seed > MAX_EXACT_INTEGER:
            _fail("model seeds must be non-negative bounded exact ints")
        result.append(seed)
    if len(set(result)) != len(result):
        _fail("expected_model_seeds must be unique")
    return tuple(result)


def _verified_runs_once(value: object) -> tuple[VerifiedRunEvidence, ...]:
    if type(value) is not tuple or not value or len(value) > 10_000:
        _fail("runs must be a non-empty bounded exact tuple")
    runs = tuple(value)
    if any(type(run) is not VerifiedRunEvidence for run in runs):
        _fail("every run must be exact verified evidence")
    replayed: list[VerifiedRunEvidence] = []
    for run in runs:
        try:
            replay = verify_run_evidence_bundle(
                run.output_dir, expected_identity=run.identity
            )
        except RunEvidenceError as exc:
            raise RunEvidenceError(
                "paired_identity_mismatch", "a run no longer replays as verified evidence"
            ) from exc
        if replay != run:
            _fail("a supplied verified run differs from replayed evidence")
        replayed.append(replay)
    return tuple(replayed)


def validate_paired_collection(
    runs: object, *, expected_model_seeds: object
) -> PairedEvidenceCollection:
    """Validate exact paired evidence before any cross-seed statistic is computed."""

    frozen_runs = _verified_runs_once(runs)
    seeds = _exact_seed_tuple(expected_model_seeds)
    by_seed: dict[int, VerifiedRunEvidence] = {}
    for run in frozen_runs:
        seed = run.identity.model_seed
        if seed in by_seed:
            _fail("model seed is duplicated")
        by_seed[seed] = run
    if set(by_seed) != set(seeds) or len(by_seed) != len(seeds):
        _fail("model seeds differ from the preregistered seed set")
    ordered = tuple(by_seed[seed] for seed in seeds)
    first = ordered[0].identity
    paired_fields = (
        "schema_version",
        "protocol_version",
        "protocol_identity",
        "claim_id",
        "benchmark_id",
        "data_scopes",
        "data_split_seed",
        "data_split_identity_sha256",
        "statistical_unit_schema",
        "statistical_unit_identity_sha256",
        "analysis_identity_sha256",
        "evidence_role",
    )
    for run in ordered[1:]:
        for field_name in paired_fields:
            if getattr(run.identity, field_name) != getattr(first, field_name):
                _fail(f"paired identity field differs: {field_name}")
    collection_record = {
        "schema_version": "1.0",
        "expected_model_seeds": list(seeds),
        "runs": [
            {
                "model_seed": run.identity.model_seed,
                "run_identity_sha256": run.identity.run_identity_sha256,
                "bundle_identity_sha256": run.bundle_identity_sha256,
                "terminal_status": run.terminal_status,
            }
            for run in ordered
        ],
    }
    return PairedEvidenceCollection(
        runs=ordered,
        expected_model_seeds=seeds,
        collection_identity_sha256=canonical_sha256(collection_record),
        statistics_allowed=all(
            run.terminal_status == "completed" for run in ordered
        ),
    )


def _safe_reason(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > 4096
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise RunEvidenceError(
            "paired_identity_mismatch", "invalidation reason is invalid"
        )
    return value


def _parent_without_symlinks(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    try:
        metadata = os.lstat(current)
    except OSError as exc:
        raise RunEvidenceError(
            "publication_infrastructure", "invalidation parent is unavailable"
        ) from exc
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise RunEvidenceError(
                "publication_infrastructure", "invalidation parent is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RunEvidenceError(
                "publication_conflict", "invalidation parent contains a symbolic link"
            )
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunEvidenceError(
            "publication_infrastructure", "invalidation parent is not a directory"
        )


def write_invalidation_record(
    path: Path | str,
    runs: object,
    *,
    category: str,
    reason: str,
) -> Path:
    """Exclusively publish a small mismatch record without changing run trees."""

    replayed = _verified_runs_once(runs)
    if type(category) is not str or category != "paired_identity_mismatch":
        raise RunEvidenceError(
            "paired_identity_mismatch", "invalidation category is not registered"
        )
    validated_reason = _safe_reason(reason)
    try:
        output = Path(path).absolute()
    except (TypeError, ValueError, OSError) as exc:
        raise RunEvidenceError(
            "publication_infrastructure", "invalidation path is invalid"
        ) from exc
    if output.name in {"", ".", ".."} or "/" in output.name:
        raise RunEvidenceError(
            "publication_infrastructure", "invalidation filename is invalid"
        )
    _parent_without_symlinks(output.parent)
    record = {
        "schema_version": "1.0",
        "category": category,
        "reason": validated_reason,
        "run_identity_sha256": [
            run.identity.run_identity_sha256 for run in replayed
        ],
        "bundle_identity_sha256": [
            run.bundle_identity_sha256 for run in replayed
        ],
    }
    payload = canonical_json_bytes(record)
    parent_fd = -1
    file_fd = -1
    created = False
    try:
        parent_fd = os.open(
            output.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            file_fd = os.open(
                output.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o400,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise RunEvidenceError(
                "publication_conflict", "invalidation record already exists"
            ) from exc
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("short invalidation write")
            view = view[written:]
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = -1
        os.fsync(parent_fd)
        return output
    except RunEvidenceError:
        raise
    except (OSError, ValueError, OverflowError) as exc:
        if created and parent_fd >= 0:
            try:
                os.unlink(output.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise RunEvidenceError(
            "publication_infrastructure", "invalidation record cannot be published"
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


__all__ = [
    "PairedEvidenceCollection",
    "validate_paired_collection",
    "write_invalidation_record",
]
