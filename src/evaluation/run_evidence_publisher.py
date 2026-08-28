"""Safe staging and publication for immutable run-evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from types import MappingProxyType, TracebackType
from typing import Any, Mapping
import unicodedata

from src.evaluation.run_evidence_identity import (
    MAX_EXACT_INTEGER,
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_sha256,
    validate_strict_json,
)


_COPY_CHUNK_BYTES = 1024 * 1024
_RESERVED_ARTIFACT_PATHS = frozenset({"method_status.json", "run_manifest.json"})
_CONTROL_PATHS = frozenset({"method_status.json", "run_manifest.json"})
_FAILURE_STATUSES = frozenset(
    {
        "failed_runtime",
        "failed_timeout",
        "failed_invalid_output",
        "failed_runtime_unavailable",
        "failed_resource",
        "failed_infrastructure",
    }
)
_MAX_CONTROL_JSON_BYTES = 4 * 1024 * 1024
_RENAME_NOREPLACE = 1


class _PublisherState(Enum):
    STAGING = "staging"
    COMPLETED_PUBLISHED = "completed_published"
    FAILED_PUBLISHED = "failed_published"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    media_type: str

    def to_record(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class VerifiedRunEvidence:
    identity: RunEvidenceIdentity
    terminal_status: str
    artifacts: tuple[ArtifactRecord, ...]
    statistical_unit_record: Mapping[str, object]
    summary: Mapping[str, object] | None
    output_dir: Path
    bundle_identity_sha256: str


def _safe_text(value: object, *, allow_slash: bool = False) -> bool:
    if type(value) is not str or not value or value != value.strip():
        return False
    if unicodedata.normalize("NFC", value) != value:
        return False
    if any(unicodedata.category(character).startswith("C") for character in value):
        return False
    if not allow_slash and ("/" in value or "\\" in value):
        return False
    return True


def _artifact_parts(value: object) -> tuple[str, ...]:
    if not _safe_text(value, allow_slash=True) or "\\" in value:
        raise RunEvidenceError("invalid_artifact", "artifact path is not safe text")
    parsed = PurePosixPath(value)
    parts = parsed.parts
    if parsed.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise RunEvidenceError(
            "invalid_artifact", "artifact path must remain below the bundle root"
        )
    for part in parts:
        if not _safe_text(part) or len(part.encode("utf-8")) > 255:
            raise RunEvidenceError("invalid_artifact", "artifact path component is invalid")
    normalized = "/".join(parts)
    if normalized in _RESERVED_ARTIFACT_PATHS:
        raise RunEvidenceError("invalid_artifact", "artifact path is publisher-reserved")
    if len(normalized.encode("utf-8")) > 4096:
        raise RunEvidenceError("invalid_artifact", "artifact path is too long")
    return tuple(parts)


def _relative_artifact_path(value: object) -> str:
    return "/".join(_artifact_parts(value))


def _assert_existing_directory_without_symlink(path: Path) -> os.stat_result:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except (OSError, ValueError) as exc:
            raise RunEvidenceError(
                "publication_infrastructure", "output parent is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RunEvidenceError(
                "publication_conflict", "output parent contains a symbolic link"
            )
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunEvidenceError(
            "publication_infrastructure", "output parent is not a directory"
        )
    return metadata


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_mode,
        first.st_nlink,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_mode,
        second.st_nlink,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _copy_regular_file(source_fd: int, destination_fd: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
    return total, digest.hexdigest()


def _write_all(destination_fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(destination_fd, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]


def _canonical_control_bytes(value: object) -> bytes:
    from src.evaluation.run_evidence_identity import canonical_json_bytes

    return canonical_json_bytes(value)


def _rename_noreplace(
    parent_fd: int, source_name: str, destination_name: str
) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise RunEvidenceError(
            "publication_infrastructure", "exclusive rename is unavailable"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RunEvidenceError(
            "publication_conflict", "formal output target already exists"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise RunEvidenceError(
            "publication_infrastructure", "exclusive rename is unavailable"
        )
    raise RunEvidenceError(
        "publication_infrastructure", "exclusive publication failed"
    )


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze(child) for child in value)
    return value


def _load_strict_canonical_json(payload: bytes, *, label: str) -> Any:
    if not payload or len(payload) > _MAX_CONTROL_JSON_BYTES:
        raise RunEvidenceError("invalid_artifact", f"{label} has an invalid size")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RunEvidenceError(
                    "invalid_artifact", f"{label} contains a duplicate JSON field"
                )
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
        normalized = validate_strict_json(value)
        if _canonical_control_bytes(normalized) != payload:
            raise RunEvidenceError(
                "invalid_artifact", f"{label} is not canonical JSON"
            )
        return normalized
    except RunEvidenceError as exc:
        if exc.category == "invalid_artifact":
            raise
        raise RunEvidenceError("invalid_artifact", f"{label} is invalid JSON") from exc
    except (UnicodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise RunEvidenceError("invalid_artifact", f"{label} is invalid JSON") from exc


def _open_read_below(directory_fd: int, parts: tuple[str, ...]) -> int:
    current_fd = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        return os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=current_fd,
        )
    finally:
        os.close(current_fd)


def _read_bound_regular_file(
    directory_fd: int, relative_path: str, *, maximum_bytes: int
) -> tuple[bytes, os.stat_result]:
    parts = _artifact_parts(relative_path) if relative_path not in _CONTROL_PATHS else (relative_path,)
    file_fd = -1
    try:
        file_fd = _open_read_below(directory_fd, parts)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise RunEvidenceError(
                "invalid_artifact", "bundle file is not a bounded single-link regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(_COPY_CHUNK_BYTES, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise RunEvidenceError("invalid_artifact", "bundle file is too large")
        after = os.fstat(file_fd)
        if not _same_file_identity(before, after) or total != before.st_size:
            raise RunEvidenceError("invalid_artifact", "bundle file changed while reading")
        return b"".join(chunks), before
    except RunEvidenceError:
        raise
    except (OSError, ValueError, OverflowError) as exc:
        raise RunEvidenceError("invalid_artifact", "bundle file cannot be read safely") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _hash_bound_regular_file(
    directory_fd: int, relative_path: str, *, expected_size: int
) -> tuple[str, os.stat_result]:
    parts = _artifact_parts(relative_path)
    file_fd = -1
    try:
        file_fd = _open_read_below(directory_fd, parts)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise RunEvidenceError(
                "invalid_artifact", "artifact is not the declared single-link regular file"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(file_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > expected_size:
                raise RunEvidenceError("invalid_artifact", "artifact grew while hashing")
        after = os.fstat(file_fd)
        if (
            not _same_file_identity(before, after)
            or total != expected_size
            or after.st_size != expected_size
        ):
            raise RunEvidenceError("invalid_artifact", "artifact changed while hashing")
        return digest.hexdigest(), before
    except RunEvidenceError:
        raise
    except (OSError, ValueError, OverflowError) as exc:
        raise RunEvidenceError("invalid_artifact", "artifact cannot be hashed safely") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _open_destination_below(staging_fd: int, parts: tuple[str, ...]) -> int:
    directory_fd = os.dup(staging_fd)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)


def _stat_name(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _remove_private_tree(path: Path, expected: tuple[int, int]) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != expected:
        raise RunEvidenceError(
            "publication_infrastructure", "staging directory identity changed"
        )
    for root, directory_names, file_names in os.walk(path, topdown=False, followlinks=False):
        root_path = Path(root)
        for name in file_names:
            child = root_path / name
            metadata = os.lstat(child)
            if stat.S_ISDIR(metadata.st_mode):
                raise RunEvidenceError(
                    "publication_infrastructure", "unexpected directory entry type"
                )
            child.unlink()
        for name in directory_names:
            child = root_path / name
            metadata = os.lstat(child)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                child.unlink()
            else:
                child.rmdir()
    path.rmdir()


class RunEvidencePublisher:
    """Single-use artifact staging state machine."""

    def __init__(
        self,
        *,
        output_dir: Path,
        identity: RunEvidenceIdentity,
        statistical_unit_record: Any,
        required_artifacts: tuple[str, ...],
        maximum_bundle_bytes: int,
        parent_fd: int,
        parent_identity: tuple[int, int],
        staging_path: Path,
        staging_fd: int,
        staging_identity: tuple[int, int],
    ) -> None:
        self._output_dir = output_dir
        self._identity = identity
        self._statistical_unit_record = statistical_unit_record
        self._required_artifacts = required_artifacts
        self._maximum_bundle_bytes = maximum_bundle_bytes
        self._parent_fd = parent_fd
        self._parent_identity = parent_identity
        self._staging_path = staging_path
        self._staging_fd = staging_fd
        self._staging_identity = staging_identity
        self._state = _PublisherState.STAGING
        self._artifacts: list[ArtifactRecord] = []
        self._artifact_paths: set[str] = set()
        self._total_artifact_bytes = 0

    @classmethod
    def begin(
        cls,
        *,
        output_dir: Path | str,
        identity: RunEvidenceIdentity,
        statistical_unit_record: object,
        required_artifacts: tuple[str, ...],
        maximum_bundle_bytes: int,
    ) -> "RunEvidencePublisher":
        if type(identity) is not RunEvidenceIdentity:
            raise RunEvidenceError(
                "invalid_identity", "identity must be an exact RunEvidenceIdentity"
            )
        if (
            type(maximum_bundle_bytes) is not int
            or maximum_bundle_bytes <= 0
            or maximum_bundle_bytes > MAX_EXACT_INTEGER
        ):
            raise RunEvidenceError(
                "invalid_identity", "maximum_bundle_bytes must be a positive bounded int"
            )
        if type(required_artifacts) is not tuple:
            raise RunEvidenceError(
                "invalid_identity", "required_artifacts must be an exact tuple"
            )
        normalized_required = tuple(
            _relative_artifact_path(item) for item in required_artifacts
        )
        if len(set(normalized_required)) != len(normalized_required):
            raise RunEvidenceError(
                "invalid_identity", "required_artifacts must be unique"
            )
        normalized_units = validate_strict_json(statistical_unit_record)
        if canonical_sha256(normalized_units) != identity.statistical_unit_identity_sha256:
            raise RunEvidenceError(
                "invalid_identity",
                "statistical unit record does not match the run identity",
            )

        try:
            output = Path(output_dir).absolute()
        except (OSError, TypeError, ValueError) as exc:
            raise RunEvidenceError(
                "publication_infrastructure", "output directory path is invalid"
            ) from exc
        if not _safe_text(output.name):
            raise RunEvidenceError(
                "publication_infrastructure", "output directory name is invalid"
            )
        parent = output.parent
        parent_metadata = _assert_existing_directory_without_symlink(parent)
        try:
            parent_fd = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except (OSError, ValueError) as exc:
            raise RunEvidenceError(
                "publication_infrastructure", "output parent cannot be opened"
            ) from exc
        try:
            bound_parent = os.fstat(parent_fd)
            if (bound_parent.st_dev, bound_parent.st_ino) != (
                parent_metadata.st_dev,
                parent_metadata.st_ino,
            ):
                raise RunEvidenceError(
                    "publication_infrastructure", "output parent identity changed"
                )
            if _stat_name(parent_fd, output.name) is not None:
                raise RunEvidenceError(
                    "publication_conflict", "formal output target already exists"
                )
            staging_name = ""
            for _ in range(64):
                candidate = f".{output.name}.staging-{secrets.token_hex(12)}"
                try:
                    os.mkdir(candidate, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                staging_name = candidate
                break
            if not staging_name:
                raise RunEvidenceError(
                    "publication_infrastructure", "cannot allocate a staging directory"
                )
            staging_fd = os.open(
                staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            staging_metadata = os.fstat(staging_fd)
            return cls(
                output_dir=output,
                identity=identity,
                statistical_unit_record=normalized_units,
                required_artifacts=normalized_required,
                maximum_bundle_bytes=maximum_bundle_bytes,
                parent_fd=parent_fd,
                parent_identity=(bound_parent.st_dev, bound_parent.st_ino),
                staging_path=parent / staging_name,
                staging_fd=staging_fd,
                staging_identity=(staging_metadata.st_dev, staging_metadata.st_ino),
            )
        except BaseException:
            os.close(parent_fd)
            raise

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def identity(self) -> RunEvidenceIdentity:
        return self._identity

    @property
    def staging_path(self) -> Path:
        return self._staging_path

    @property
    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._artifacts)

    @property
    def total_artifact_bytes(self) -> int:
        return self._total_artifact_bytes

    def __enter__(self) -> "RunEvidencePublisher":
        self._require_staging()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exception_type is not None and self._state is _PublisherState.STAGING:
            self.abort()
        return False

    def _require_staging(self) -> None:
        if self._state is not _PublisherState.STAGING:
            raise RunEvidenceError(
                "invalid_state_transition", "publisher is no longer staging"
            )

    def _reserve_artifact(self, relative_path: object, size_bytes: int) -> tuple[str, ...]:
        self._require_staging()
        parts = _artifact_parts(relative_path)
        normalized = "/".join(parts)
        if normalized in self._artifact_paths:
            raise RunEvidenceError("invalid_artifact", "artifact path is duplicated")
        if self._total_artifact_bytes + size_bytes > self._maximum_bundle_bytes:
            raise RunEvidenceError("invalid_artifact", "bundle byte limit exceeded")
        return parts

    @staticmethod
    def _validate_media_type(media_type: object) -> str:
        if (
            not _safe_text(media_type, allow_slash=True)
            or "\\" in media_type
            or media_type.count("/") != 1
            or any(not part for part in media_type.split("/"))
        ):
            raise RunEvidenceError("invalid_artifact", "media_type is invalid")
        return media_type

    def _record_artifact(
        self, parts: tuple[str, ...], size_bytes: int, sha256: str, media_type: str
    ) -> ArtifactRecord:
        relative_path = "/".join(parts)
        record = ArtifactRecord(relative_path, size_bytes, sha256, media_type)
        self._artifacts.append(record)
        self._artifact_paths.add(relative_path)
        self._total_artifact_bytes += size_bytes
        return record

    def _abort_and_raise(self, error: BaseException) -> None:
        try:
            self.abort()
        except RunEvidenceError as cleanup_error:
            raise cleanup_error from error
        raise error

    def add_bytes(
        self, relative_path: str, payload: bytes, *, media_type: str
    ) -> ArtifactRecord:
        try:
            self._require_staging()
            if type(payload) is not bytes:
                raise RunEvidenceError(
                    "invalid_artifact", "artifact payload must be exact bytes"
                )
            validated_media_type = self._validate_media_type(media_type)
            parts = self._reserve_artifact(relative_path, len(payload))
            destination_fd = _open_destination_below(self._staging_fd, parts)
            try:
                _write_all(destination_fd, payload)
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            return self._record_artifact(
                parts,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                validated_media_type,
            )
        except (RunEvidenceError, OSError, UnicodeError, ValueError) as exc:
            error = (
                exc
                if isinstance(exc, RunEvidenceError)
                else RunEvidenceError("invalid_artifact", "artifact bytes cannot be staged")
            )
            self._abort_and_raise(error)
        raise AssertionError("unreachable")

    def add_file(
        self, relative_path: str, source_path: Path | str, *, media_type: str
    ) -> ArtifactRecord:
        source_fd = -1
        destination_fd = -1
        try:
            self._require_staging()
            validated_media_type = self._validate_media_type(media_type)
            source = Path(source_path).absolute()
            _assert_source_path_without_symlink(source)
            source_fd = os.open(
                source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RunEvidenceError(
                    "invalid_artifact", "artifact source must be a single-link regular file"
                )
            parts = self._reserve_artifact(relative_path, before.st_size)
            destination_fd = _open_destination_below(self._staging_fd, parts)
            size_bytes, sha256 = _copy_regular_file(source_fd, destination_fd)
            os.fsync(destination_fd)
            after = os.fstat(source_fd)
            if not _same_file_identity(before, after) or size_bytes != before.st_size:
                raise RunEvidenceError(
                    "invalid_artifact", "artifact source changed during streaming copy"
                )
            return self._record_artifact(
                parts, size_bytes, sha256, validated_media_type
            )
        except (RunEvidenceError, OSError, UnicodeError, ValueError, TypeError) as exc:
            error = (
                exc
                if isinstance(exc, RunEvidenceError)
                else RunEvidenceError("invalid_artifact", "artifact file cannot be staged")
            )
            self._abort_and_raise(error)
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)
            if source_fd >= 0:
                os.close(source_fd)
        raise AssertionError("unreachable")

    def _write_control_file(self, relative_path: str, payload: bytes) -> None:
        if relative_path not in _CONTROL_PATHS:
            raise RunEvidenceError(
                "publication_infrastructure", "unknown publisher control file"
            )
        destination_fd = -1
        try:
            destination_fd = os.open(
                relative_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
                dir_fd=self._staging_fd,
            )
            _write_all(destination_fd, payload)
            os.fsync(destination_fd)
        except RunEvidenceError:
            raise
        except (OSError, ValueError) as exc:
            raise RunEvidenceError(
                "publication_infrastructure", "control file cannot be written"
            ) from exc
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)

    def _terminal_records(
        self,
        *,
        status: str,
        summary: object | None,
        reason: str | None,
    ) -> tuple[bytes, bytes]:
        artifacts = tuple(sorted(self._artifacts, key=lambda item: item.relative_path))
        artifact_records = [artifact.to_record() for artifact in artifacts]
        inventory_sha256 = canonical_sha256(artifact_records)
        normalized_summary = None if summary is None else validate_strict_json(summary)
        if normalized_summary is not None and type(normalized_summary) is not dict:
            raise RunEvidenceError(
                "invalid_state_transition", "completed summary must be a JSON object"
            )
        summary_sha256 = (
            None if normalized_summary is None else canonical_sha256(normalized_summary)
        )
        status_record = {
            "schema_version": "1.0",
            "status": status,
            "run_identity_sha256": self._identity.run_identity_sha256,
            "artifact_inventory_sha256": inventory_sha256,
            "summary_sha256": summary_sha256,
            "reason": reason,
        }
        manifest_record = {
            "schema_version": "1.0",
            "run_identity": self._identity.to_record(),
            "run_identity_sha256": self._identity.run_identity_sha256,
            "statistical_unit_record": self._statistical_unit_record,
            "statistical_unit_identity_sha256": self._identity.statistical_unit_identity_sha256,
            "required_artifacts": list(self._required_artifacts),
            "artifacts": artifact_records,
            "artifact_inventory_sha256": inventory_sha256,
            "summary": normalized_summary,
            "summary_sha256": summary_sha256,
            "terminal_status": status_record,
        }
        status_payload = _canonical_control_bytes(status_record)
        manifest_payload = _canonical_control_bytes(manifest_record)
        if (
            self._total_artifact_bytes + len(status_payload) + len(manifest_payload)
            > self._maximum_bundle_bytes
        ):
            raise RunEvidenceError(
                "invalid_artifact", "bundle byte limit exceeded by control records"
            )
        return status_payload, manifest_payload

    def _publish_terminal(
        self,
        *,
        status: str,
        summary: object | None,
        reason: str | None,
        published_state: _PublisherState,
    ) -> Path:
        published = False
        try:
            self._require_staging()
            status_payload, manifest_payload = self._terminal_records(
                status=status, summary=summary, reason=reason
            )
            self._write_control_file("method_status.json", status_payload)
            self._write_control_file("run_manifest.json", manifest_payload)
            os.fsync(self._staging_fd)
            parent_now = os.fstat(self._parent_fd)
            if (parent_now.st_dev, parent_now.st_ino) != self._parent_identity:
                raise RunEvidenceError(
                    "publication_infrastructure", "output parent identity changed"
                )
            _rename_noreplace(
                self._parent_fd, self._staging_path.name, self._output_dir.name
            )
            published = True
            os.fsync(self._parent_fd)
            self._state = published_state
            os.close(self._staging_fd)
            self._staging_fd = -1
            verified = verify_run_evidence_bundle(
                self._output_dir, expected_identity=self._identity
            )
            if verified.terminal_status != status:
                raise RunEvidenceError(
                    "publication_infrastructure", "published status cannot be replayed"
                )
            return self._output_dir
        except (RunEvidenceError, OSError, ValueError, OverflowError) as exc:
            error = (
                exc
                if isinstance(exc, RunEvidenceError)
                else RunEvidenceError(
                    "publication_infrastructure", "terminal publication failed"
                )
            )
            if not published and self._state is _PublisherState.STAGING:
                try:
                    self.abort()
                except RunEvidenceError as cleanup_error:
                    raise cleanup_error from error
            raise error
        finally:
            if published and self._parent_fd >= 0:
                os.close(self._parent_fd)
                self._parent_fd = -1

    def finalize_completed(self, *, summary: object) -> Path:
        try:
            self._require_staging()
            if set(self._artifact_paths) != set(self._required_artifacts):
                raise RunEvidenceError(
                    "invalid_state_transition",
                    "completed publication requires the exact declared artifacts",
                )
            return self._publish_terminal(
                status="completed",
                summary=summary,
                reason=None,
                published_state=_PublisherState.COMPLETED_PUBLISHED,
            )
        except RunEvidenceError as exc:
            if self._state is _PublisherState.STAGING:
                self._abort_and_raise(exc)
            raise

    def finalize_failure(self, *, status: str, reason: str) -> Path:
        try:
            self._require_staging()
            if type(status) is not str or status not in _FAILURE_STATUSES:
                raise RunEvidenceError(
                    "invalid_state_transition", "failure status is not registered"
                )
            if not _safe_text(reason, allow_slash=True) or len(reason.encode("utf-8")) > 4096:
                raise RunEvidenceError(
                    "invalid_state_transition", "failure reason is invalid"
                )
            if self._artifacts:
                raise RunEvidenceError(
                    "invalid_state_transition",
                    "failure publication cannot contain scientific artifacts",
                )
            return self._publish_terminal(
                status=status,
                summary=None,
                reason=reason,
                published_state=_PublisherState.FAILED_PUBLISHED,
            )
        except RunEvidenceError as exc:
            if self._state is _PublisherState.STAGING:
                self._abort_and_raise(exc)
            raise

    def abort(self) -> None:
        if self._state is _PublisherState.ABORTED:
            return
        self._require_staging()
        os.close(self._staging_fd)
        self._staging_fd = -1
        try:
            _remove_private_tree(self._staging_path, self._staging_identity)
        finally:
            os.close(self._parent_fd)
            self._parent_fd = -1
            self._state = _PublisherState.ABORTED


def _assert_source_path_without_symlink(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except (OSError, ValueError) as exc:
            raise RunEvidenceError("invalid_artifact", "artifact source is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RunEvidenceError(
                "invalid_artifact", "artifact source path contains a symbolic link"
            )


def _collect_bundle_files(directory_fd: int) -> set[str]:
    files: set[str] = set()

    def visit(current_fd: int, prefix: tuple[str, ...]) -> None:
        try:
            entries = tuple(os.scandir(current_fd))
        except OSError as exc:
            raise RunEvidenceError(
                "invalid_artifact", "bundle directory cannot be enumerated"
            ) from exc
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RunEvidenceError(
                    "invalid_artifact", "bundle entry cannot be inspected"
                ) from exc
            relative_parts = (*prefix, entry.name)
            if stat.S_ISDIR(metadata.st_mode) and not entry.is_symlink():
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=current_fd,
                )
                try:
                    visit(child_fd, relative_parts)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode) or entry.is_symlink():
                raise RunEvidenceError(
                    "invalid_artifact", "bundle contains a non-regular file"
                )
            relative = "/".join(relative_parts)
            if relative in files:
                raise RunEvidenceError(
                    "invalid_artifact", "bundle contains a duplicate path"
                )
            files.add(relative)

    try:
        visit(directory_fd, ())
        return files
    except RunEvidenceError:
        raise
    except (OSError, ValueError) as exc:
        raise RunEvidenceError("invalid_artifact", "bundle tree cannot be inspected") from exc


def _require_exact_fields(
    value: object, expected: set[str], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise RunEvidenceError(
            "invalid_artifact", f"{label} fields do not match the schema"
        )
    return value


def _parse_artifact_records(value: object) -> tuple[ArtifactRecord, ...]:
    if type(value) is not list:
        raise RunEvidenceError("invalid_artifact", "artifact inventory must be an array")
    records: list[ArtifactRecord] = []
    paths: set[str] = set()
    for raw_record in value:
        record = _require_exact_fields(
            raw_record,
            {"relative_path", "size_bytes", "sha256", "media_type"},
            label="artifact record",
        )
        relative_path = _relative_artifact_path(record["relative_path"])
        size_bytes = record["size_bytes"]
        sha256 = record["sha256"]
        media_type = record["media_type"]
        if (
            type(size_bytes) is not int
            or size_bytes < 0
            or size_bytes > MAX_EXACT_INTEGER
        ):
            raise RunEvidenceError("invalid_artifact", "artifact size is invalid")
        if (
            type(sha256) is not str
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RunEvidenceError("invalid_artifact", "artifact SHA-256 is invalid")
        validated_media_type = RunEvidencePublisher._validate_media_type(media_type)
        if relative_path in paths:
            raise RunEvidenceError("invalid_artifact", "artifact path is duplicated")
        paths.add(relative_path)
        records.append(
            ArtifactRecord(
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=sha256,
                media_type=validated_media_type,
            )
        )
    if records != sorted(records, key=lambda item: item.relative_path):
        raise RunEvidenceError("invalid_artifact", "artifact inventory is not sorted")
    return tuple(records)


def verify_run_evidence_bundle(
    path: Path | str, *, expected_identity: RunEvidenceIdentity | None = None
) -> VerifiedRunEvidence:
    """Replay a terminal bundle from bytes and return deeply immutable evidence."""

    if expected_identity is not None and type(expected_identity) is not RunEvidenceIdentity:
        raise RunEvidenceError(
            "invalid_identity", "expected_identity must be an exact RunEvidenceIdentity"
        )
    try:
        output = Path(path).absolute()
    except (OSError, TypeError, ValueError) as exc:
        raise RunEvidenceError("invalid_artifact", "bundle path is invalid") from exc
    _assert_existing_directory_without_symlink(output)
    try:
        directory_fd = os.open(
            output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except (OSError, ValueError) as exc:
        raise RunEvidenceError("invalid_artifact", "bundle cannot be opened") from exc
    try:
        directory_before = os.fstat(directory_fd)
        manifest_payload, manifest_metadata = _read_bound_regular_file(
            directory_fd,
            "run_manifest.json",
            maximum_bytes=_MAX_CONTROL_JSON_BYTES,
        )
        status_payload, status_metadata = _read_bound_regular_file(
            directory_fd,
            "method_status.json",
            maximum_bytes=_MAX_CONTROL_JSON_BYTES,
        )
        manifest = _require_exact_fields(
            _load_strict_canonical_json(manifest_payload, label="run manifest"),
            {
                "schema_version",
                "run_identity",
                "run_identity_sha256",
                "statistical_unit_record",
                "statistical_unit_identity_sha256",
                "required_artifacts",
                "artifacts",
                "artifact_inventory_sha256",
                "summary",
                "summary_sha256",
                "terminal_status",
            },
            label="run manifest",
        )
        status_record = _require_exact_fields(
            _load_strict_canonical_json(status_payload, label="method status"),
            {
                "schema_version",
                "status",
                "run_identity_sha256",
                "artifact_inventory_sha256",
                "summary_sha256",
                "reason",
            },
            label="method status",
        )
        if manifest["schema_version"] != "1.0" or status_record["schema_version"] != "1.0":
            raise RunEvidenceError("invalid_artifact", "bundle schema version is invalid")
        try:
            identity = RunEvidenceIdentity.from_record(manifest["run_identity"])
        except RunEvidenceError as exc:
            raise RunEvidenceError("invalid_identity", "run identity cannot be replayed") from exc
        if manifest["run_identity_sha256"] != identity.run_identity_sha256:
            raise RunEvidenceError("invalid_identity", "run identity SHA-256 changed")
        if expected_identity is not None and identity != expected_identity:
            raise RunEvidenceError("invalid_identity", "run identity differs from expected")

        unit_record = validate_strict_json(manifest["statistical_unit_record"])
        if type(unit_record) is not dict:
            raise RunEvidenceError(
                "invalid_artifact", "statistical unit record must be an object"
            )
        if (
            canonical_sha256(unit_record)
            != identity.statistical_unit_identity_sha256
            or manifest["statistical_unit_identity_sha256"]
            != identity.statistical_unit_identity_sha256
        ):
            raise RunEvidenceError(
                "invalid_identity", "statistical unit identity changed"
            )

        required_value = manifest["required_artifacts"]
        if type(required_value) is not list:
            raise RunEvidenceError(
                "invalid_artifact", "required artifacts must be an array"
            )
        required_artifacts = tuple(
            _relative_artifact_path(item) for item in required_value
        )
        if len(set(required_artifacts)) != len(required_artifacts):
            raise RunEvidenceError(
                "invalid_artifact", "required artifacts are duplicated"
            )
        artifacts = _parse_artifact_records(manifest["artifacts"])
        artifact_paths = tuple(record.relative_path for record in artifacts)
        artifact_records = [record.to_record() for record in artifacts]
        inventory_sha256 = canonical_sha256(artifact_records)
        if (
            manifest["artifact_inventory_sha256"] != inventory_sha256
            or status_record["artifact_inventory_sha256"] != inventory_sha256
        ):
            raise RunEvidenceError("invalid_artifact", "artifact inventory changed")

        terminal_status = status_record["status"]
        if type(terminal_status) is not str or terminal_status not in {
            "completed",
            *_FAILURE_STATUSES,
        }:
            raise RunEvidenceError("invalid_artifact", "terminal status is invalid")
        if status_record["run_identity_sha256"] != identity.run_identity_sha256:
            raise RunEvidenceError("invalid_identity", "method status identity changed")
        if manifest["terminal_status"] != status_record:
            raise RunEvidenceError("invalid_artifact", "terminal status records differ")

        summary = manifest["summary"]
        summary_sha256 = manifest["summary_sha256"]
        reason = status_record["reason"]
        if terminal_status == "completed":
            if type(summary) is not dict or summary_sha256 != canonical_sha256(summary):
                raise RunEvidenceError("invalid_artifact", "completed summary changed")
            if status_record["summary_sha256"] != summary_sha256 or reason is not None:
                raise RunEvidenceError("invalid_artifact", "completed status is inconsistent")
            if set(artifact_paths) != set(required_artifacts):
                raise RunEvidenceError(
                    "invalid_artifact", "completed artifact set is incomplete"
                )
        else:
            if summary is not None or summary_sha256 is not None:
                raise RunEvidenceError("invalid_artifact", "failure cannot contain a summary")
            if status_record["summary_sha256"] is not None or not _safe_text(
                reason, allow_slash=True
            ):
                raise RunEvidenceError("invalid_artifact", "failure reason is invalid")
            if artifacts or required_artifacts:
                raise RunEvidenceError(
                    "invalid_artifact", "failure cannot contain scientific artifacts"
                )

        expected_files = set(artifact_paths) | _CONTROL_PATHS
        if _collect_bundle_files(directory_fd) != expected_files:
            raise RunEvidenceError("invalid_artifact", "bundle file set changed")
        observed_inodes = {
            (manifest_metadata.st_dev, manifest_metadata.st_ino),
            (status_metadata.st_dev, status_metadata.st_ino),
        }
        if len(observed_inodes) != 2:
            raise RunEvidenceError("invalid_artifact", "control files share an inode")
        for artifact in artifacts:
            observed_sha256, metadata = _hash_bound_regular_file(
                directory_fd,
                artifact.relative_path,
                expected_size=artifact.size_bytes,
            )
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in observed_inodes:
                raise RunEvidenceError("invalid_artifact", "bundle files share an inode")
            observed_inodes.add(inode)
            if observed_sha256 != artifact.sha256:
                raise RunEvidenceError("invalid_artifact", "artifact content changed")
        directory_after = os.fstat(directory_fd)
        if (
            directory_before.st_dev,
            directory_before.st_ino,
        ) != (directory_after.st_dev, directory_after.st_ino):
            raise RunEvidenceError("invalid_artifact", "bundle directory identity changed")
        try:
            path_after = os.lstat(output)
        except OSError as exc:
            raise RunEvidenceError("invalid_artifact", "bundle path disappeared") from exc
        if (
            stat.S_ISLNK(path_after.st_mode)
            or path_after.st_dev != directory_after.st_dev
            or path_after.st_ino != directory_after.st_ino
        ):
            raise RunEvidenceError("invalid_artifact", "bundle path identity changed")
        return VerifiedRunEvidence(
            identity=identity,
            terminal_status=terminal_status,
            artifacts=artifacts,
            statistical_unit_record=_deep_freeze(unit_record),
            summary=None if summary is None else _deep_freeze(summary),
            output_dir=output,
            bundle_identity_sha256=hashlib.sha256(
                b"run-evidence-bundle-v1\x00"
                + manifest_payload
                + b"\x00"
                + status_payload
            ).hexdigest(),
        )
    except RunEvidenceError:
        raise
    except (OSError, UnicodeError, ValueError, OverflowError, TypeError) as exc:
        raise RunEvidenceError("invalid_artifact", "bundle verification failed") from exc
    finally:
        os.close(directory_fd)


__all__ = [
    "ArtifactRecord",
    "RunEvidencePublisher",
    "VerifiedRunEvidence",
    "verify_run_evidence_bundle",
]
