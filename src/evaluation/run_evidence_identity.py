"""Immutable identities and canonical JSON for run-evidence bundles.

This module intentionally depends only on the Python standard library.  It is
the lowest-level trust boundary used by run evidence publication and replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import unicodedata
from typing import Any


ERROR_CATEGORIES = frozenset(
    {
        "invalid_identity",
        "invalid_artifact",
        "invalid_state_transition",
        "publication_conflict",
        "publication_infrastructure",
        "paired_identity_mismatch",
    }
)
EVIDENCE_ROLES = frozenset(
    {"pilot_audit_only", "release_candidate", "infrastructure_smoke"}
)
IDENTITY_SCHEMA_VERSION = "1.0"
MAX_EXACT_INTEGER = 2**63 - 1
MAX_JSON_DEPTH = 32
MAX_JSON_ITEMS = 100_000
MAX_CANONICAL_JSON_BYTES = 4 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class RunEvidenceError(ValueError):
    """Stable, non-sensitive domain error for run-evidence operations."""

    def __init__(self, category: str, message: str):
        if type(category) is not str or category not in ERROR_CATEGORIES:
            raise ValueError("unknown run-evidence error category")
        if type(message) is not str or not message:
            raise ValueError("run-evidence error message must be non-empty text")
        self.category = category
        super().__init__(f"{category}: {message}")


def _is_safe_nfc_text(value: object, *, allow_empty: bool) -> bool:
    if type(value) is not str:
        return False
    if not allow_empty and not value:
        return False
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        return False
    return not any(unicodedata.category(character).startswith("C") for character in value)


def _require_exact_nfc_text(value: object, *, field_name: str) -> str:
    if not _is_safe_nfc_text(value, allow_empty=False):
        raise RunEvidenceError(
            "invalid_identity", f"{field_name} must be non-empty exact NFC text"
        )
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RunEvidenceError(
            "invalid_identity", f"{field_name} must be a lowercase SHA-256"
        )
    return value


def _require_bounded_seed(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_EXACT_INTEGER:
        raise RunEvidenceError(
            "invalid_identity",
            f"{field_name} must be a non-negative bounded exact int",
        )
    return value


def _require_data_scopes(value: object, *, evidence_role: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value or len(value) > 16:
        raise RunEvidenceError(
            "invalid_identity", "data_scopes must be a non-empty bounded exact tuple"
        )
    scopes: list[str] = []
    for item in value:
        scopes.append(_require_exact_nfc_text(item, field_name="data_scopes item"))
    if len(set(scopes)) != len(scopes):
        raise RunEvidenceError("invalid_identity", "data_scopes must be unique")
    frozen = tuple(scopes)
    if evidence_role == "pilot_audit_only" and frozen != ("train", "tune"):
        raise RunEvidenceError(
            "invalid_identity",
            "pilot_audit_only data_scopes must equal ('train', 'tune')",
        )
    return frozen


def validate_strict_json(value: object) -> Any:
    """Return an independent built-in JSON tree or fail closed.

    Scalars must be exact built-ins; mappings and sequences must be exact
    ``dict``, ``list``, or ``tuple`` objects.  Restricting containers to exact
    built-ins prevents a mutable or adversarial protocol implementation from
    changing values between validation and canonical serialization.
    """

    item_count = 0

    def visit(item: object, depth: int) -> Any:
        nonlocal item_count
        if depth > MAX_JSON_DEPTH:
            raise RunEvidenceError("invalid_identity", "JSON nesting is too deep")
        item_count += 1
        if item_count > MAX_JSON_ITEMS:
            raise RunEvidenceError("invalid_identity", "JSON item count is too large")

        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if item < -MAX_EXACT_INTEGER or item > MAX_EXACT_INTEGER:
                raise RunEvidenceError(
                    "invalid_identity", "JSON integer is outside the supported range"
                )
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise RunEvidenceError("invalid_identity", "JSON float must be finite")
            return item
        if type(item) is str:
            if not _is_safe_nfc_text(item, allow_empty=True):
                raise RunEvidenceError(
                    "invalid_identity", "JSON text must be exact safe NFC text"
                )
            return item
        if type(item) is dict:
            result: dict[str, Any] = {}
            for key, child in tuple(item.items()):
                if not _is_safe_nfc_text(key, allow_empty=False):
                    raise RunEvidenceError(
                        "invalid_identity",
                        "JSON object keys must be non-empty exact safe NFC text",
                    )
                result[key] = visit(child, depth + 1)
            return result
        if type(item) is list or type(item) is tuple:
            return [visit(child, depth + 1) for child in tuple(item)]
        raise RunEvidenceError(
            "invalid_identity", "value is not an exact built-in JSON type"
        )

    return visit(value, 0)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a strict JSON value to deterministic UTF-8 bytes."""

    normalized = validate_strict_json(value)
    try:
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, UnicodeError, ValueError) as exc:
        raise RunEvidenceError(
            "invalid_identity", "value cannot be serialized as canonical JSON"
        ) from exc
    if len(payload) > MAX_CANONICAL_JSON_BYTES:
        raise RunEvidenceError("invalid_identity", "canonical JSON is too large")
    return payload


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of ``canonical_json_bytes(value)``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class RunEvidenceIdentity:
    """Exact immutable scientific and provenance identity for one run."""

    schema_version: str
    protocol_version: str
    protocol_identity: str
    claim_id: str
    benchmark_id: str
    data_scopes: tuple[str, ...]
    data_split_seed: int
    model_seed: int
    data_split_identity_sha256: str
    statistical_unit_schema: str
    statistical_unit_identity_sha256: str
    analysis_identity_sha256: str
    input_identity_sha256: str
    config_identity_sha256: str
    code_identity_sha256: str
    evidence_role: str
    run_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != IDENTITY_SCHEMA_VERSION:
            raise RunEvidenceError(
                "invalid_identity", "schema_version must equal '1.0'"
            )
        for field_name in (
            "protocol_version",
            "claim_id",
            "benchmark_id",
            "statistical_unit_schema",
        ):
            _require_exact_nfc_text(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "protocol_identity",
            "data_split_identity_sha256",
            "statistical_unit_identity_sha256",
            "analysis_identity_sha256",
            "input_identity_sha256",
            "config_identity_sha256",
            "code_identity_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        if type(self.evidence_role) is not str or self.evidence_role not in EVIDENCE_ROLES:
            raise RunEvidenceError("invalid_identity", "unknown evidence_role")
        _require_bounded_seed(self.data_split_seed, field_name="data_split_seed")
        _require_bounded_seed(self.model_seed, field_name="model_seed")
        frozen_scopes = _require_data_scopes(
            self.data_scopes, evidence_role=self.evidence_role
        )
        object.__setattr__(self, "data_scopes", frozen_scopes)
        object.__setattr__(
            self,
            "run_identity_sha256",
            hashlib.sha256(canonical_json_bytes(self.to_record())).hexdigest(),
        )

    def to_record(self) -> dict[str, object]:
        """Return a fresh strict-JSON record without the derived digest."""

        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "protocol_identity": self.protocol_identity,
            "claim_id": self.claim_id,
            "benchmark_id": self.benchmark_id,
            "data_scopes": list(self.data_scopes),
            "data_split_seed": self.data_split_seed,
            "model_seed": self.model_seed,
            "data_split_identity_sha256": self.data_split_identity_sha256,
            "statistical_unit_schema": self.statistical_unit_schema,
            "statistical_unit_identity_sha256": self.statistical_unit_identity_sha256,
            "analysis_identity_sha256": self.analysis_identity_sha256,
            "input_identity_sha256": self.input_identity_sha256,
            "config_identity_sha256": self.config_identity_sha256,
            "code_identity_sha256": self.code_identity_sha256,
            "evidence_role": self.evidence_role,
        }

    @classmethod
    def from_record(cls, value: object) -> "RunEvidenceIdentity":
        """Construct from an exact record and reject missing or extra fields."""

        normalized = validate_strict_json(value)
        if type(normalized) is not dict:
            raise RunEvidenceError("invalid_identity", "identity record must be an object")
        expected_fields = {
            "schema_version",
            "protocol_version",
            "protocol_identity",
            "claim_id",
            "benchmark_id",
            "data_scopes",
            "data_split_seed",
            "model_seed",
            "data_split_identity_sha256",
            "statistical_unit_schema",
            "statistical_unit_identity_sha256",
            "analysis_identity_sha256",
            "input_identity_sha256",
            "config_identity_sha256",
            "code_identity_sha256",
            "evidence_role",
        }
        if set(normalized) != expected_fields:
            raise RunEvidenceError(
                "invalid_identity", "identity record fields do not match the schema"
            )
        scopes = normalized["data_scopes"]
        if type(scopes) is not list:
            raise RunEvidenceError("invalid_identity", "data_scopes must be an array")
        arguments = dict(normalized)
        arguments["data_scopes"] = tuple(scopes)
        return cls(**arguments)


__all__ = [
    "ERROR_CATEGORIES",
    "EVIDENCE_ROLES",
    "RunEvidenceError",
    "RunEvidenceIdentity",
    "canonical_json_bytes",
    "canonical_sha256",
    "validate_strict_json",
]
