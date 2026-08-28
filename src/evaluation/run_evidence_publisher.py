"""Safe staging and publication for immutable run-evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from types import TracebackType
from typing import Any, Iterable
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


__all__ = ["ArtifactRecord", "RunEvidencePublisher"]
