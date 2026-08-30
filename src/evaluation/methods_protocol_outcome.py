"""Immutable closure record for the methods protocol v2.1 pilot audit."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, cast

from src.evaluation.run_evidence_identity import canonical_json_bytes, validate_strict_json


PROTOCOL_VERSION = "hypersca-methods-v2.1"
PROTOCOL_IDENTITY_SHA256 = "caa2f9a4aed7e474c123cb815435f65df5011387a4be1181d324a635b1a01613"
PILOT_SUMMARY_SHA256 = "3fe9e90443f82a911fe02314a540cd8e3383ee016cff9c3dbb46b802490d694c"
MAXIMUM_OUTCOME_BYTES = 128 * 1024
_RENAME_NOREPLACE = 1

_OUTCOME_KEYS = frozenset(
    {
        "blocking_reasons",
        "collection_identity_sha256",
        "pilot_summary_sha256",
        "protocol_identity_sha256",
        "protocol_version",
        "release_authorized",
        "run_identity_sha256",
        "status",
    }
)
_FROZEN_RUN_IDENTITIES = (
    "fc088d5a8251c4e7e70fe6613ca3847dd971a36a412d3096760d7cff68426b22",
    "c63a6de584dda5e00ba76a4b962764fb200c5b8c021ea1dead4607093d2ef827",
    "7a7a8ea0273f86773601930884e8e66e744a48bbf4e73b20772f33440f112b1d",
    "bc322a4d4821da931b7df3bcf7e5919b76432e38adb9ef286b8f13e7beed6960",
    "63b854969471aa962131cd8117f8445971cbeba3143205d8c5e35cba0ceb7e26",
    "909adaa309179f583b2c0fed6036777dafa0dde08e4095405d30c26ea835c5f6",
    "51eb97812a2bb53949d82c7f5eb585811ce0c984e24d1de8ea7230568ae8f2f4",
    "10723f903f68a8a17efd16015879972eb5a9c733bc2e8ad27aaa04eab6209a8f",
    "1869270041b3fc086ac64c51ade9fdea7420057c0cca0c6a42432479a56c323b",
    "bc2ec707dd7be4911c76eab87e773dcb94788e655d5c61451256e539441c47e0",
    "535211ce95de353080b05fceb6a99ab2fd52f242128be217b2c09ec8200a4a88",
    "66c6acc450e19d2b56de558d08c2f8d96f088d94cf17df53ce2f5fc87a19d25b",
    "22604c3cc2a3a1df97e4406a340b97d4e965e922b86bc953171b8d4baeb60423",
    "05a1321d0f0d076c97280b066ea9534634afa340d5fec712fd2e851e8d6d11ab",
    "9ecc164e3b6a172d2fc1e78f9a34bbc6f071b2f5bf63c8706c33f001b267fa50",
    "98bd107ed4c86160bfb501de934f869d450dc726974f21d190a887f23f146c2f",
    "14c6a2ad076b05114a2541102323224ccf07c25213a13bafa390364c10152e13",
    "68c373c0f795fa0e0bc85e9575c28b1ee2a864077e7d0b830e130e424d86634b",
)
_FROZEN_COLLECTION_IDENTITIES = (
    "8d076045146a488799d91406c2208ae1c584531098bc076c87b063c5806e2379",
    "48898dd06010932ae17aed3abe59d69be4da45978c67f9528928f606165f8d57",
    "467e5dddb04209aeb0df84c3d2c7341831cdabded4576575b1a8c8b2057837d3",
    "a776be2ebde43e1c501439d7b9ca4e843cbba26030c009050e0f843146ca6350",
    "8f0aec7e95fdaab9f6be34bc269bca6430ff760fddaa590a71be7009596c2b31",
    "3c7b19b6cbe75b82b1f783b2e1f7cfde2f441351f8f277c5ad43c4cda173939e",
)
_FROZEN_BLOCKING_REASONS = (
    "OSTA has one biological sample per platform stratum.",
    "The CausalBench mean-difference comparator has no eligible relations.",
    "The OSTA hierarchy-attribution confidence interval crosses zero.",
    "CausalBench direction-attribution fails for both k562_to_rpe1 and rpe1_to_k562.",
    "The v2.1 redesign has already been used and cannot be silently redesigned again.",
)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_deep_json(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > 32:
                raise ValueError("JSON is too deeply nested")
        elif character in "]}":
            depth -= 1


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _bounded_integer(token: str) -> int:
    if len(token.lstrip("-")) > 19:
        raise ValueError("JSON integer is too large")
    return int(token)


def _bounded_float(token: str) -> float:
    if len(token) > 128:
        raise ValueError("JSON float is too large")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON float is non-finite")
    return value


def _strict_json_from_bytes(payload: bytes, label: str) -> dict[str, object]:
    if not payload or len(payload) > MAXIMUM_OUTCOME_BYTES:
        raise ValueError(f"{label} is empty or unusually large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    _reject_deep_json(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
            parse_int=_bounded_integer,
            parse_float=_bounded_float,
        )
        normalized = validate_strict_json(value)
    except (
        _DuplicateJsonKey,
        UnicodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc
    if type(normalized) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return normalized


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} is not safely accessible") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not contain a symbolic link")


def _read_bounded_regular_file(path: Path, label: str) -> bytes:
    source = Path(path)
    _reject_symlink_components(source, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must not be hard-linked")
        if before.st_size <= 0 or before.st_size > MAXIMUM_OUTCOME_BYTES:
            raise ValueError(f"{label} is empty or unusually large")
        payload = bytearray()
        while len(payload) <= MAXIMUM_OUTCOME_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, MAXIMUM_OUTCOME_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while it was read")
    return bytes(payload)


def strict_json(path: Path) -> dict[str, object]:
    """Read one bounded, single-link JSON object without duplicate keys."""

    return _strict_json_from_bytes(_read_bounded_regular_file(path, "JSON input"), "JSON input")


def _freeze_text_sequence(
    value: list[str] | tuple[str, ...], label: str
) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{label} must be an ordered built-in sequence")
    frozen: tuple[str, ...] = tuple(value)
    if any(type(item) is not str or not item for item in frozen):
        raise ValueError(f"{label} must contain non-empty built-in strings")
    return frozen


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True, init=False)
class ProtocolOutcome:
    """The exact, no-release closure of the audited v2.1 pilot."""

    protocol_version: str
    protocol_identity_sha256: str
    pilot_summary_sha256: str
    status: str
    release_authorized: bool
    run_identity_sha256: tuple[str, ...]
    collection_identity_sha256: tuple[str, ...]
    blocking_reasons: tuple[str, ...]

    def __init__(
        self,
        *,
        protocol_version: str,
        protocol_identity_sha256: str,
        pilot_summary_sha256: str,
        status: str,
        release_authorized: bool,
        run_identity_sha256: list[str] | tuple[str, ...],
        collection_identity_sha256: list[str] | tuple[str, ...],
        blocking_reasons: list[str] | tuple[str, ...],
    ) -> None:
        object.__setattr__(self, "protocol_version", protocol_version)
        object.__setattr__(self, "protocol_identity_sha256", protocol_identity_sha256)
        object.__setattr__(self, "pilot_summary_sha256", pilot_summary_sha256)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "release_authorized", release_authorized)
        object.__setattr__(
            self,
            "run_identity_sha256",
            _freeze_text_sequence(run_identity_sha256, "run_identity_sha256"),
        )
        object.__setattr__(
            self,
            "collection_identity_sha256",
            _freeze_text_sequence(
                collection_identity_sha256,
                "collection_identity_sha256",
            ),
        )
        object.__setattr__(
            self,
            "blocking_reasons",
            _freeze_text_sequence(blocking_reasons, "blocking_reasons"),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not str or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("outcome is not frozen v2.1 evidence")
        if type(self.status) is not str or self.status != "pilot_failed_no_release":
            raise ValueError("v2.1 outcome must remain no-release")
        if type(self.release_authorized) is not bool or self.release_authorized is not False:
            raise ValueError("v2.1 outcome must remain no-release")
        if _require_sha256(self.protocol_identity_sha256, "protocol_identity_sha256") != PROTOCOL_IDENTITY_SHA256:
            raise ValueError("outcome does not bind the frozen v2.1 protocol identity")
        if _require_sha256(self.pilot_summary_sha256, "pilot_summary_sha256") != PILOT_SUMMARY_SHA256:
            raise ValueError("outcome does not bind the audited pilot summary")

        runs = _freeze_text_sequence(self.run_identity_sha256, "run_identity_sha256")
        collections = _freeze_text_sequence(
            self.collection_identity_sha256, "collection_identity_sha256"
        )
        reasons = _freeze_text_sequence(self.blocking_reasons, "blocking_reasons")
        for identity in (*runs, *collections):
            _require_sha256(identity, "identity")
        if len(runs) != 18 or len(set(runs)) != 18:
            raise ValueError("v2.1 outcome must bind 18 unique runs")
        if len(collections) != 6 or len(set(collections)) != 6:
            raise ValueError("v2.1 outcome must bind six paired collections")
        if runs != _FROZEN_RUN_IDENTITIES or collections != _FROZEN_COLLECTION_IDENTITIES:
            raise ValueError("v2.1 outcome identities must retain their frozen ordering")
        if reasons != _FROZEN_BLOCKING_REASONS:
            raise ValueError("v2.1 outcome must retain its five frozen blocking reasons")
        object.__setattr__(self, "run_identity_sha256", runs)
        object.__setattr__(self, "collection_identity_sha256", collections)
        object.__setattr__(self, "blocking_reasons", reasons)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProtocolOutcome":
        if type(value) is not dict or set(value) != _OUTCOME_KEYS:
            raise ValueError("outcome must contain exactly the frozen closure fields")
        return cls(
            protocol_version=cast(str, value["protocol_version"]),
            protocol_identity_sha256=cast(str, value["protocol_identity_sha256"]),
            pilot_summary_sha256=cast(str, value["pilot_summary_sha256"]),
            status=cast(str, value["status"]),
            release_authorized=cast(bool, value["release_authorized"]),
            run_identity_sha256=cast(
                list[str] | tuple[str, ...], value["run_identity_sha256"]
            ),
            collection_identity_sha256=cast(
                list[str] | tuple[str, ...], value["collection_identity_sha256"]
            ),
            blocking_reasons=cast(
                list[str] | tuple[str, ...], value["blocking_reasons"]
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "blocking_reasons": list(self.blocking_reasons),
            "collection_identity_sha256": list(self.collection_identity_sha256),
            "pilot_summary_sha256": self.pilot_summary_sha256,
            "protocol_identity_sha256": self.protocol_identity_sha256,
            "protocol_version": self.protocol_version,
            "release_authorized": self.release_authorized,
            "run_identity_sha256": list(self.run_identity_sha256),
            "status": self.status,
        }


def load_protocol_outcome(path: Path) -> ProtocolOutcome:
    """Load and require the canonical, immutable v2.1 closure record."""

    payload = _read_bounded_regular_file(path, "protocol outcome")
    record = _strict_json_from_bytes(payload, "protocol outcome")
    outcome = ProtocolOutcome.from_mapping(record)
    if payload != canonical_json_bytes(outcome.to_mapping()):
        raise ValueError("protocol outcome is not canonical JSON")
    return outcome


def outcome_from_pilot_summary(path: Path) -> ProtocolOutcome:
    """Verify the approved pilot summary and extract its frozen identities."""

    payload = _read_bounded_regular_file(path, "pilot summary")
    if hashlib.sha256(payload).hexdigest() != PILOT_SUMMARY_SHA256:
        raise ValueError("pilot summary SHA-256 does not match the approved audit")
    summary = _strict_json_from_bytes(payload, "pilot summary")
    if (
        summary.get("protocol_version") != PROTOCOL_VERSION
        or summary.get("protocol_identity_sha256") != PROTOCOL_IDENTITY_SHA256
        or summary.get("run_count") != 18
        or summary.get("completed_run_count") != 18
        or summary.get("promotion_eligible") is not False
    ):
        raise ValueError("pilot summary does not contain the frozen v2.1 no-release audit")
    runs = summary.get("runs")
    collections = summary.get("paired_collections")
    if type(runs) is not list or type(collections) is not list:
        raise ValueError("pilot summary does not contain the audited identities")
    run_identities: list[str] = []
    collection_identities: list[str] = []
    for item in runs:
        if type(item) is not dict or type(item.get("run_identity_sha256")) is not str:
            raise ValueError("pilot summary identity records must be JSON objects")
        run_identities.append(item["run_identity_sha256"])
    for item in collections:
        if (
            type(item) is not dict
            or type(item.get("collection_identity_sha256")) is not str
        ):
            raise ValueError("pilot summary identity records must be JSON objects")
        collection_identities.append(item["collection_identity_sha256"])
    return ProtocolOutcome(
        protocol_version=PROTOCOL_VERSION,
        protocol_identity_sha256=PROTOCOL_IDENTITY_SHA256,
        pilot_summary_sha256=PILOT_SUMMARY_SHA256,
        status="pilot_failed_no_release",
        release_authorized=False,
        run_identity_sha256=run_identities,
        collection_identity_sha256=collection_identities,
        blocking_reasons=_FROZEN_BLOCKING_REASONS,
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (metadata.st_dev, metadata.st_ino)


def _stat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _parent_path_matches_identity(parent: Path, identity: tuple[int, int]) -> bool:
    try:
        observed = os.stat(parent, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(observed.st_mode) and _inode_identity(observed) == identity


def _open_bound_parent_directory(parent: Path) -> tuple[int, tuple[int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise ValueError("protocol outcome parent is missing or unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        identity = _inode_identity(metadata)
        if not stat.S_ISDIR(metadata.st_mode) or not _parent_path_matches_identity(
            parent, identity
        ):
            raise ValueError("protocol outcome parent changed while it was bound")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _verify_bound_parent_directory(parent: Path, identity: tuple[int, int]) -> None:
    """Reject path substitution before a dirfd-relative publication commits."""

    _reject_symlink_components(parent, "protocol outcome parent")
    if not _parent_path_matches_identity(parent, identity):
        raise ValueError("protocol outcome parent changed while it was bound")


def _rename_noreplace(parent_fd: int, source_name: str, target_name: str) -> None:
    """Atomically publish *source_name* only when *target_name* is absent."""

    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
    except (AttributeError, OSError) as exc:
        raise ValueError("exclusive protocol outcome publication is unavailable") from exc
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
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError("refusing to overwrite existing protocol outcome")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise ValueError("exclusive protocol outcome publication is unavailable")
    raise ValueError("protocol outcome atomic publication failed") from OSError(error_number, os.strerror(error_number))


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short protocol outcome write")
        view = view[written:]


def write_protocol_outcome_exclusively(path: Path, outcome: ProtocolOutcome) -> None:
    """Atomically publish a canonical closure record without clobbering output.

    A successful ``renameat2(RENAME_NOREPLACE)`` followed by a successful fsync
    of the already-bound parent directory is the commit point.  On every
    pre-commit or post-publication uncertainty this function raises while
    preserving the involved names: Linux/Python has no inode-conditional unlink
    primitive, so deleting a name after an adversarial replacement is unsafe.
    An uncertain or pre-commit failure can therefore leave a randomized staging
    entry.  An operator must inspect its inode identity before any removal and
    must never blindly delete a matching-looking staging filename.
    """

    destination = Path(path)
    _reject_symlink_components(destination.parent, "protocol outcome parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "protocol outcome parent")
    payload = canonical_json_bytes(outcome.to_mapping())
    parent_fd: int | None = None
    temporary_fd: int | None = None
    parent_identity: tuple[int, int] | None = None
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    publication_committed = False
    operation_failed = False
    try:
        parent_fd, parent_identity = _open_bound_parent_directory(destination.parent)
        if _stat_at(parent_fd, destination.name) is not None:
            raise ValueError("refusing to overwrite existing protocol outcome")

        for _ in range(8):
            candidate = f".{destination.name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            temporary_metadata = os.fstat(temporary_fd)
            temporary_identity = _inode_identity(temporary_metadata)
            observed_temporary = _stat_at(parent_fd, temporary_name)
            if (
                not stat.S_ISREG(temporary_metadata.st_mode)
                or temporary_metadata.st_nlink != 1
                or observed_temporary is None
                or _inode_identity(observed_temporary) != temporary_identity
            ):
                raise ValueError("protocol outcome staging inode is invalid")
            break
        if temporary_fd is None or temporary_name is None or temporary_identity is None:
            raise ValueError("unable to allocate protocol outcome staging file")

        _write_all(temporary_fd, payload)
        os.fsync(temporary_fd)
        temporary_metadata = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
            or _inode_identity(temporary_metadata) != temporary_identity
        ):
            raise ValueError("protocol outcome staging inode changed before publication")

        _verify_bound_parent_directory(destination.parent, parent_identity)
        immediately_before_publish = _stat_at(parent_fd, temporary_name)
        if (
            immediately_before_publish is None
            or not stat.S_ISREG(immediately_before_publish.st_mode)
            or immediately_before_publish.st_nlink != 1
            or _inode_identity(immediately_before_publish) != temporary_identity
        ):
            raise ValueError("protocol outcome staging inode changed before publication")

        _rename_noreplace(parent_fd, temporary_name, destination.name)
        published_metadata = _stat_at(parent_fd, destination.name)
        if published_metadata is None:
            raise ValueError("published protocol outcome inode is missing")
        if (
            not stat.S_ISREG(published_metadata.st_mode)
            or _inode_identity(published_metadata) != temporary_identity
            or published_metadata.st_nlink != 1
        ):
            raise ValueError("published protocol outcome inode is invalid")

        final_fd = os.open(
            destination.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            final_metadata = os.fstat(final_fd)
        finally:
            os.close(final_fd)
        if (
            not stat.S_ISREG(final_metadata.st_mode)
            or _inode_identity(final_metadata) != temporary_identity
            or final_metadata.st_nlink != 1
        ):
            raise ValueError("published protocol outcome inode is invalid")

        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise ValueError(
                "protocol outcome publication directory fsync failed; output is preserved"
            ) from exc
        publication_committed = True
        final_after_commit = _stat_at(parent_fd, destination.name)
        if (
            final_after_commit is None
            or not stat.S_ISREG(final_after_commit.st_mode)
            or _inode_identity(final_after_commit) != temporary_identity
            or final_after_commit.st_nlink != 1
        ):
            raise ValueError("published protocol outcome inode is invalid")
        _verify_bound_parent_directory(destination.parent, parent_identity)
    except OSError as exc:
        operation_failed = True
        raise ValueError("protocol outcome publication failed") from exc
    except ValueError:
        operation_failed = True
        raise
    finally:
        close_error: OSError | None = None
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError as exc:
                close_error = exc
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError as exc:
                close_error = close_error or exc
        if close_error is not None:
            if publication_committed:
                raise ValueError(
                    "protocol outcome publication committed but descriptor cleanup failed"
                ) from close_error
            if not operation_failed:
                raise ValueError("protocol outcome descriptor cleanup failed") from close_error
