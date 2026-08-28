"""Immutable closure record for the methods protocol v2.1 pilot audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping

from src.evaluation.run_evidence_identity import canonical_json_bytes, validate_strict_json


PROTOCOL_VERSION = "hypersca-methods-v2.1"
PROTOCOL_IDENTITY_SHA256 = "caa2f9a4aed7e474c123cb815435f65df5011387a4be1181d324a635b1a01613"
PILOT_SUMMARY_SHA256 = "3fe9e90443f82a911fe02314a540cd8e3383ee016cff9c3dbb46b802490d694c"
MAXIMUM_OUTCOME_BYTES = 128 * 1024

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
            elif character == "\\\\":
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
    except (_DuplicateJsonKey, UnicodeError, json.JSONDecodeError, OverflowError, ValueError) as exc:
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
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
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


def _freeze_text_sequence(value: object, label: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{label} must be an ordered built-in sequence")
    frozen = tuple(value)
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


@dataclass(frozen=True, slots=True)
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
            protocol_version=value["protocol_version"],  # type: ignore[arg-type]
            protocol_identity_sha256=value["protocol_identity_sha256"],  # type: ignore[arg-type]
            pilot_summary_sha256=value["pilot_summary_sha256"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            release_authorized=value["release_authorized"],  # type: ignore[arg-type]
            run_identity_sha256=value["run_identity_sha256"],  # type: ignore[arg-type]
            collection_identity_sha256=value["collection_identity_sha256"],  # type: ignore[arg-type]
            blocking_reasons=value["blocking_reasons"],  # type: ignore[arg-type]
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
    try:
        run_identities = [item["run_identity_sha256"] for item in runs if type(item) is dict]
        collection_identities = [
            item["collection_identity_sha256"] for item in collections if type(item) is dict
        ]
    except KeyError as exc:
        raise ValueError("pilot summary does not contain the audited identities") from exc
    if len(run_identities) != len(runs) or len(collection_identities) != len(collections):
        raise ValueError("pilot summary identity records must be JSON objects")
    return ProtocolOutcome(
        protocol_version=PROTOCOL_VERSION,
        protocol_identity_sha256=PROTOCOL_IDENTITY_SHA256,
        pilot_summary_sha256=PILOT_SUMMARY_SHA256,
        status="pilot_failed_no_release",
        release_authorized=False,
        run_identity_sha256=run_identities,  # type: ignore[arg-type]
        collection_identity_sha256=collection_identities,  # type: ignore[arg-type]
        blocking_reasons=_FROZEN_BLOCKING_REASONS,
    )


def write_protocol_outcome_exclusively(path: Path, outcome: ProtocolOutcome) -> None:
    """Publish one canonical closure record without overwriting any output."""

    destination = Path(path)
    _reject_symlink_components(destination.parent, "protocol outcome parent")
    if destination.exists() or destination.is_symlink():
        raise ValueError("refusing to overwrite existing protocol outcome")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "protocol outcome parent")
    payload = canonical_json_bytes(outcome.to_mapping())
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short protocol outcome write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError("refusing to overwrite existing protocol outcome") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
