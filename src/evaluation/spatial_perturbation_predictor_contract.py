"""Outcome-blind contract for spatial-perturbation predictor evidence.

This trust boundary is deliberately standard-library only.  It validates
declarations and already-produced prediction bytes; it never discovers or
imports a model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, cast
import unicodedata

from src.evaluation.run_evidence_identity import (
    MAX_EXACT_INTEGER,
    RunEvidenceError,
    canonical_json_bytes,
    canonical_sha256,
    validate_strict_json,
)
from src.methods_protocol_v3_contract import (
    MethodsProtocolV3,
    build_methods_protocol_v3,
    protocol_identity_v3,
    protocol_to_mapping_v3,
)


MAX_PREDICTION_BYTES = 8 * 1024 * 1024
MAX_PREDICTION_ROWS = 100_000
_PREDICTION_METHODS = (
    "hypersca",
    "matched_euclidean_spatial_causal",
    "hypersca_own_only",
)
_PREDICTION_COLUMNS = (
    "unit_id",
    "endpoint",
    "predicted_effect",
    "effect_units",
)
_PREDICTION_FORMAT = "bridge_comparator_predictions_json_v1"


def _fresh_prediction_schema() -> dict[str, object]:
    return {
        "format": _PREDICTION_FORMAT,
        "methods": list(_PREDICTION_METHODS),
        "columns": list(_PREDICTION_COLUMNS),
    }


PREDICTION_METHODS = _PREDICTION_METHODS
PREDICTION_COLUMNS = _PREDICTION_COLUMNS
PREDICTION_SCHEMA = _fresh_prediction_schema()
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FIXED_STATUS = "method_adapter_not_executable"
_FIXED_REASON = "no_preregistered_bridge_predictor_adapter"
_REGISTRY_SCHEMA_VERSION = "spatial_perturbation_bridge_candidates_v1"
_REGISTRY_METHOD_ID = "hypersca"
_REGISTRY_FIELDS = frozenset({"schema_version", "candidates"})
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "accession",
        "platform",
        "biological_specimens",
        "sections_by_specimen",
        "safe_control_label",
        "perturbation_labels",
        "source_uri",
        "source_identity_sha256",
    }
)
_FORBIDDEN_DECLARATION_KEYS = frozenset(
    {
        "callable",
        "factory",
        "factory_path",
        "import",
        "import_path",
        "model_module",
        "module",
        "outcome",
        "outcomes",
        "prediction_path",
    }
)


class BridgePredictorContractError(ValueError):
    """A declaration or immutable prediction bundle is not admissible."""


def _fail(message: str) -> None:
    raise BridgePredictorContractError(message)


def _safe_text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 4096
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        _fail(f"{name} must be bounded non-empty exact safe NFC text")
    return cast(str, value)


def _sha(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{name} must be an exact lowercase SHA-256")
    return cast(str, value)


def _seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_EXACT_INTEGER:
        _fail("model_seed must be a bounded non-negative exact int")
    return cast(int, value)


def _strict_tree(value: object, name: str) -> Any:
    try:
        return validate_strict_json(value)
    except RunEvidenceError as error:
        raise BridgePredictorContractError(f"{name} is not strict JSON") from error


def _reject_unsafe_declaration_keys(value: object, name: str) -> dict[str, Any]:
    normalized = _strict_tree(value, name)
    if type(normalized) is not dict or not normalized:
        _fail(f"{name} must be a non-empty exact built-in declaration")

    def visit(item: object) -> None:
        if type(item) is dict:
            for key, child in item.items():
                if key.casefold() in _FORBIDDEN_DECLARATION_KEYS:
                    _fail(f"{name} contains a forbidden executable or outcome field")
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)

    visit(normalized)
    adapters = normalized.get("adapters")
    if adapters is not None and (type(adapters) is not list or adapters):
        _fail("no predictor adapter is preregistered for the bridge")
    return normalized


def formal_bridge_predictor_registry_declaration_to_mapping(
    value: object,
) -> dict[str, object]:
    """Validate and snapshot the tracked Task4 candidate registry schema."""

    normalized = _strict_tree(value, "registry_declaration")
    if (
        type(normalized) is not dict
        or set(normalized) != _REGISTRY_FIELDS
        or normalized.get("schema_version") != _REGISTRY_SCHEMA_VERSION
        or type(normalized.get("candidates")) is not list
        or not normalized["candidates"]
    ):
        _fail("registry_declaration must equal the exact closed formal registry")
    candidate_ids: set[str] = set()
    for raw_candidate in normalized["candidates"]:
        if type(raw_candidate) is not dict or set(raw_candidate) != _CANDIDATE_FIELDS:
            _fail("registry candidate fields do not match the exact closed schema")
        candidate = cast(dict[str, object], raw_candidate)
        candidate_id = _safe_text(candidate["candidate_id"], "candidate_id")
        if candidate_id in candidate_ids:
            _fail("registry candidate_id values must be unique")
        candidate_ids.add(candidate_id)
        _safe_text(candidate["accession"], "accession")
        if candidate["platform"] != "spatial_perturbation":
            _fail("registry candidate platform must be spatial_perturbation")
        specimens = candidate["biological_specimens"]
        if (
            type(specimens) is not list
            or not specimens
            or any(type(item) is not str for item in specimens)
        ):
            _fail("registry biological_specimens must be a non-empty text array")
        specimen_values = cast(list[object], specimens)
        specimen_names = [
            _safe_text(item, "biological_specimen") for item in specimen_values
        ]
        if len(set(specimen_names)) != len(specimen_names):
            _fail("registry biological_specimens must be unique")
        sections = candidate["sections_by_specimen"]
        if type(sections) is not list or len(sections) != len(specimen_names):
            _fail("registry sections_by_specimen must cover every specimen exactly")
        section_values = cast(list[object], sections)
        section_specimens: list[str] = []
        for raw_sections in section_values:
            if (
                type(raw_sections) is not list
                or len(raw_sections) != 2
                or type(raw_sections[0]) is not str
                or type(raw_sections[1]) is not list
            ):
                _fail("registry section entry must be an exact specimen/list pair")
            section_pair = cast(list[object], raw_sections)
            section_specimens.append(
                _safe_text(section_pair[0], "section specimen")
            )
            section_names = cast(list[object], section_pair[1])
            if any(type(item) is not str for item in section_names):
                _fail("registry section identifiers must be exact strings")
            validated_sections = [
                _safe_text(item, "section_id") for item in section_names
            ]
            if len(set(validated_sections)) != len(validated_sections):
                _fail("registry section identifiers must be unique per specimen")
        if section_specimens != specimen_names:
            _fail("registry sections_by_specimen order must match specimens")
        _safe_text(candidate["safe_control_label"], "safe_control_label")
        perturbations = candidate["perturbation_labels"]
        if type(perturbations) is not list or any(
            type(item) is not str for item in perturbations
        ):
            _fail("registry perturbation_labels must be an exact text array")
        perturbation_values = cast(list[object], perturbations)
        perturbation_names = [
            _safe_text(item, "perturbation_label") for item in perturbation_values
        ]
        if len(set(perturbation_names)) != len(perturbation_names):
            _fail("registry perturbation_labels must be unique")
        _safe_text(candidate["source_uri"], "source_uri")
        _sha(
            candidate["source_identity_sha256"],
            "registry source_identity_sha256",
        )
    return cast(dict[str, object], _strict_tree(normalized, "registry_declaration"))


def formal_protocol_declaration_to_mapping(value: object) -> dict[str, object]:
    """Return an exact snapshot of the frozen Methods v3.0 declaration."""

    try:
        if type(value) is MethodsProtocolV3:
            mapping = protocol_to_mapping_v3(cast(MethodsProtocolV3, value))
        else:
            mapping = _reject_unsafe_declaration_keys(
                value, "protocol_declaration"
            )
        if type(mapping) is not dict:
            _fail("protocol_declaration must be an exact formal declaration")
        version = mapping.get("protocol_version")
        role = mapping.get("bridge_role")
        capability_identity = mapping.get("capability_identity_sha256")
        if version != "hypersca-methods-v3.0" or type(role) is not str:
            _fail("protocol_declaration must use hypersca-methods-v3.0")
        role_text = cast(str, role)
        expected_protocol = build_methods_protocol_v3(
            bridge_role=role_text,
            capability_identity_sha256=cast(str, capability_identity),
        )
        expected = protocol_to_mapping_v3(expected_protocol)
        if mapping != expected:
            _fail("protocol_declaration differs from the frozen Methods v3.0 schema")
        return protocol_to_mapping_v3(expected_protocol)
    except BridgePredictorContractError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise BridgePredictorContractError(
            "protocol_declaration is not an exact Methods v3.0 declaration"
        ) from error


def formal_protocol_declaration_identity_sha256(value: object) -> str:
    mapping = formal_protocol_declaration_to_mapping(value)
    protocol = build_methods_protocol_v3(
        bridge_role=cast(str, mapping["bridge_role"]),
        capability_identity_sha256=cast(
            str, mapping["capability_identity_sha256"]
        ),
    )
    return protocol_identity_v3(protocol)


def _capability_unsigned(
    method_id: str, registry_identity: str, protocol_identity: str
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "method_id": method_id,
        "registry_identity_sha256": registry_identity,
        "protocol_identity_sha256": protocol_identity,
        "status": _FIXED_STATUS,
        "executable": False,
        "adapter_identity_sha256": None,
        "blocking_reasons": [_FIXED_REASON],
    }


@dataclass(frozen=True, slots=True)
class BridgePredictorCapability:
    schema_version: str
    method_id: str
    registry_identity_sha256: str
    protocol_identity_sha256: str
    status: str
    executable: bool
    adapter_identity_sha256: None
    blocking_reasons: tuple[str, ...]
    capability_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "1.0":
            _fail("capability schema_version must equal '1.0'")
        method = _safe_text(self.method_id, "method_id")
        registry_identity = _sha(
            self.registry_identity_sha256, "registry_identity_sha256"
        )
        protocol_identity = _sha(
            self.protocol_identity_sha256, "protocol_identity_sha256"
        )
        if (
            type(self.status) is not str
            or self.status != _FIXED_STATUS
            or type(self.executable) is not bool
            or self.executable is not False
            or self.adapter_identity_sha256 is not None
            or type(self.blocking_reasons) is not tuple
            or self.blocking_reasons != (_FIXED_REASON,)
        ):
            _fail("capability must remain the fixed non-executable production result")
        identity = _sha(
            self.capability_identity_sha256, "capability_identity_sha256"
        )
        expected = canonical_sha256(
            _capability_unsigned(method, registry_identity, protocol_identity)
        )
        if identity != expected:
            _fail("capability identity does not match its fixed declaration")
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "registry_identity_sha256", registry_identity)
        object.__setattr__(self, "protocol_identity_sha256", protocol_identity)
        object.__setattr__(self, "capability_identity_sha256", identity)


def _snapshot_capability(value: object) -> BridgePredictorCapability:
    if type(value) is not BridgePredictorCapability:
        _fail("capability must be an exact BridgePredictorCapability")
    item = cast(BridgePredictorCapability, value)
    return BridgePredictorCapability(
        item.schema_version,
        item.method_id,
        item.registry_identity_sha256,
        item.protocol_identity_sha256,
        item.status,
        item.executable,
        item.adapter_identity_sha256,
        item.blocking_reasons,
        item.capability_identity_sha256,
    )


def audit_bridge_predictor_capability(
    registry_declaration: object,
    protocol_declaration: object,
    *,
    method_id: str,
) -> BridgePredictorCapability:
    """Audit frozen declarations without loading outcomes or model code."""

    registry = formal_bridge_predictor_registry_declaration_to_mapping(
        registry_declaration
    )
    protocol = formal_protocol_declaration_to_mapping(protocol_declaration)
    method = _safe_text(method_id, "method_id")
    if method != _REGISTRY_METHOD_ID:
        _fail("method_id is not the unique formally registered method")
    registry_identity = canonical_sha256(registry)
    protocol_identity = formal_protocol_declaration_identity_sha256(protocol)
    unsigned = _capability_unsigned(method, registry_identity, protocol_identity)
    return BridgePredictorCapability(
        "1.0",
        method,
        registry_identity,
        protocol_identity,
        _FIXED_STATUS,
        False,
        None,
        (_FIXED_REASON,),
        canonical_sha256(unsigned),
    )


def bridge_predictor_capability_to_mapping(
    capability: BridgePredictorCapability,
) -> dict[str, object]:
    frozen = _snapshot_capability(capability)
    mapping = _capability_unsigned(
        frozen.method_id,
        frozen.registry_identity_sha256,
        frozen.protocol_identity_sha256,
    )
    mapping["capability_identity_sha256"] = frozen.capability_identity_sha256
    return mapping


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if type(value) is list:
        return tuple(_deep_freeze(child) for child in value)
    return value


def _strict_json_bytes(payload: object) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_PREDICTION_BYTES:
        _fail("prediction_bytes must be bounded non-empty exact bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail("prediction_bytes contain a duplicate JSON field")
            result[key] = value
        return result

    try:
        exact_payload = cast(bytes, payload)
        value = json.loads(
            exact_payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
        normalized = _strict_tree(value, "prediction_bytes")
        if type(normalized) is not dict or canonical_json_bytes(normalized) != exact_payload:
            _fail("prediction_bytes must be canonical UTF-8 JSON")
        return normalized
    except BridgePredictorContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, OverflowError) as error:
        raise BridgePredictorContractError("prediction_bytes are invalid JSON") from error


def _validate_schema(value: object) -> dict[str, Any]:
    normalized = _strict_tree(value, "prediction_schema")
    if normalized != _fresh_prediction_schema():
        _fail("prediction_schema must equal the frozen bridge schema")
    return normalized


def _schema_record(value: object) -> dict[str, Any]:
    if type(value) is MappingProxyType:
        normalized = {
            key: list(child) if type(child) is tuple else child
            for key, child in value.items()
        }
        if normalized != _fresh_prediction_schema():
            _fail("frozen prediction_schema changed")
        return normalized
    return _validate_schema(value)


def _validate_prediction_payload(payload: object, *, origin: str) -> dict[str, Any]:
    normalized = _strict_json_bytes(payload)
    if set(normalized) != {"schema_version", "origin", "predictions"}:
        _fail("prediction payload fields do not match the frozen schema")
    if normalized["schema_version"] != "1.0" or normalized["origin"] != origin:
        _fail("prediction payload origin or schema changed")
    predictions = normalized["predictions"]
    if type(predictions) is not dict or set(predictions) != set(_PREDICTION_METHODS):
        _fail("prediction payload must contain the exact bridge methods")
    frozen_rows: dict[str, tuple[tuple[str, str, float, str], ...]] = {}
    for method in _PREDICTION_METHODS:
        rows = predictions[method]
        if type(rows) is not list or not rows or len(rows) > MAX_PREDICTION_ROWS:
            _fail("each prediction method requires bounded non-empty rows")
        validated: list[tuple[str, str, float, str]] = []
        for row in rows:
            if type(row) is not dict or set(row) != set(_PREDICTION_COLUMNS):
                _fail("prediction row fields do not match the schema")
            unit_id = _sha(row["unit_id"], "unit_id")
            endpoint = row["endpoint"]
            effect = row["predicted_effect"]
            units = row["effect_units"]
            if type(endpoint) is not str or endpoint not in {"neighbor", "own"}:
                _fail("prediction endpoint is not frozen")
            if type(effect) not in {int, float} or type(effect) is bool:
                _fail("predicted_effect must be an exact built-in real number")
            numeric = float(effect)
            if not math.isfinite(numeric) or abs(numeric) > 1.0e12:
                _fail("predicted_effect must be finite and bounded")
            if units != "train_control_standardized_delta":
                _fail("prediction effect units changed")
            validated.append((unit_id, endpoint, numeric, units))
        keys = tuple((row[0], row[1]) for row in validated)
        if len(set(keys)) != len(keys) or {row[1] for row in validated} != {
            "neighbor",
            "own",
        }:
            _fail("prediction rows must have unique keys and both endpoints")
        frozen_rows[method] = tuple(validated)
    baseline = frozen_rows["hypersca"]
    for method in _PREDICTION_METHODS[1:]:
        candidate = frozen_rows[method]
        if tuple((row[0], row[1], row[3]) for row in candidate) != tuple(
            (row[0], row[1], row[3]) for row in baseline
        ):
            _fail("bridge methods must preserve exact prediction rows and units")
    own_only = frozen_rows["hypersca_own_only"]
    for baseline_row, own_row in zip(baseline, own_only):
        expected = baseline_row[2] if baseline_row[1] == "own" else 0.0
        if own_row[2] != expected or (expected == 0.0 and math.copysign(1.0, own_row[2]) < 0):
            _fail("own-only predictions do not match the frozen comparator")
    return normalized


def _bundle_unsigned(bundle: "BridgePredictionBundle") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "method_id": bundle.method_id,
        "protocol_identity_sha256": bundle.protocol_identity_sha256,
        "data_identity_sha256": bundle.data_identity_sha256,
        "split_identity_sha256": bundle.split_identity_sha256,
        "statistical_unit_identity_sha256": bundle.statistical_unit_identity_sha256,
        "code_identity_sha256": bundle.code_identity_sha256,
        "model_seed": bundle.model_seed,
        "prediction_schema": _schema_record(bundle.prediction_schema),
        "prediction_bytes_sha256": bundle.prediction_bytes_sha256,
        "origin": bundle.origin,
        "evidence_role": bundle.evidence_role,
    }


@dataclass(frozen=True, slots=True)
class BridgePredictionBundle:
    schema_version: str
    method_id: str
    protocol_identity_sha256: str
    data_identity_sha256: str
    split_identity_sha256: str
    statistical_unit_identity_sha256: str
    code_identity_sha256: str
    model_seed: int
    prediction_schema: Mapping[str, object]
    prediction_bytes: bytes
    prediction_bytes_sha256: str
    origin: str
    evidence_role: str
    prediction_bundle_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "1.0":
            _fail("prediction bundle schema_version must equal '1.0'")
        method = _safe_text(self.method_id, "method_id")
        identities = tuple(
            _sha(getattr(self, name), name)
            for name in (
                "protocol_identity_sha256",
                "data_identity_sha256",
                "split_identity_sha256",
                "statistical_unit_identity_sha256",
                "code_identity_sha256",
            )
        )
        seed = _seed(self.model_seed)
        schema = _validate_schema(self.prediction_schema)
        origin = _safe_text(self.origin, "origin")
        role = _safe_text(self.evidence_role, "evidence_role")
        if origin == "synthetic_fixture":
            if role != "synthetic_audit_only":
                _fail("synthetic predictions require synthetic_audit_only evidence")
        elif origin == "production":
            _fail(
                "production prediction evidence is not executable without a preregistered adapter"
            )
        else:
            _fail("prediction origin is not registered")
        _validate_prediction_payload(self.prediction_bytes, origin=origin)
        payload_sha = _sha(self.prediction_bytes_sha256, "prediction_bytes_sha256")
        if payload_sha != hashlib.sha256(self.prediction_bytes).hexdigest():
            _fail("prediction bytes digest changed")
        bundle_identity = _sha(
            self.prediction_bundle_identity_sha256,
            "prediction_bundle_identity_sha256",
        )
        for name, value in zip(
            (
                "protocol_identity_sha256",
                "data_identity_sha256",
                "split_identity_sha256",
                "statistical_unit_identity_sha256",
                "code_identity_sha256",
            ),
            identities,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "model_seed", seed)
        object.__setattr__(self, "prediction_schema", _deep_freeze(schema))
        object.__setattr__(self, "prediction_bytes", bytes(self.prediction_bytes))
        object.__setattr__(self, "prediction_bytes_sha256", payload_sha)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "evidence_role", role)
        if bundle_identity != canonical_sha256(_bundle_unsigned(self)):
            _fail("prediction bundle identity changed")
        object.__setattr__(
            self, "prediction_bundle_identity_sha256", bundle_identity
        )


def _snapshot_bundle(value: object) -> BridgePredictionBundle:
    if type(value) is not BridgePredictionBundle:
        _fail("prediction bundle must be an exact BridgePredictionBundle")
    item = cast(BridgePredictionBundle, value)
    return BridgePredictionBundle(
        item.schema_version,
        item.method_id,
        item.protocol_identity_sha256,
        item.data_identity_sha256,
        item.split_identity_sha256,
        item.statistical_unit_identity_sha256,
        item.code_identity_sha256,
        item.model_seed,
        _schema_record(item.prediction_schema),
        item.prediction_bytes,
        item.prediction_bytes_sha256,
        item.origin,
        item.evidence_role,
        item.prediction_bundle_identity_sha256,
    )


def build_bridge_prediction_bundle(
    *,
    method_id: str,
    protocol_identity_sha256: str,
    data_identity_sha256: str,
    split_identity_sha256: str,
    statistical_unit_identity_sha256: str,
    code_identity_sha256: str,
    model_seed: int,
    prediction_schema: object,
    prediction_bytes: bytes,
    origin: str,
    evidence_role: str,
) -> BridgePredictionBundle:
    schema = _validate_schema(prediction_schema)
    payload = bytes(prediction_bytes) if type(prediction_bytes) is bytes else prediction_bytes
    payload_sha = (
        hashlib.sha256(payload).hexdigest() if type(payload) is bytes else "0" * 64
    )
    shell = object.__new__(BridgePredictionBundle)
    for name, value in (
        ("schema_version", "1.0"),
        ("method_id", method_id),
        ("protocol_identity_sha256", protocol_identity_sha256),
        ("data_identity_sha256", data_identity_sha256),
        ("split_identity_sha256", split_identity_sha256),
        ("statistical_unit_identity_sha256", statistical_unit_identity_sha256),
        ("code_identity_sha256", code_identity_sha256),
        ("model_seed", model_seed),
        ("prediction_schema", schema),
        ("prediction_bytes", payload),
        ("prediction_bytes_sha256", payload_sha),
        ("origin", origin),
        ("evidence_role", evidence_role),
    ):
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "prediction_bundle_identity_sha256", "0" * 64)
    try:
        identity = canonical_sha256(_bundle_unsigned(shell))
    except RunEvidenceError as error:
        raise BridgePredictorContractError(
            "prediction bundle fields are not canonical"
        ) from error
    return BridgePredictionBundle(
        "1.0",
        method_id,
        protocol_identity_sha256,
        data_identity_sha256,
        split_identity_sha256,
        statistical_unit_identity_sha256,
        code_identity_sha256,
        model_seed,
        schema,
        payload,
        payload_sha,
        origin,
        evidence_role,
        identity,
    )


def bridge_prediction_bundle_to_mapping(
    bundle: BridgePredictionBundle,
) -> dict[str, object]:
    frozen = _snapshot_bundle(bundle)
    mapping = _bundle_unsigned(frozen)
    mapping["prediction_bundle_identity_sha256"] = (
        frozen.prediction_bundle_identity_sha256
    )
    return mapping


def prediction_payload_to_mapping(bundle: BridgePredictionBundle) -> dict[str, Any]:
    """Return a fresh exact prediction payload after full bundle replay."""

    frozen = _snapshot_bundle(bundle)
    return _validate_prediction_payload(frozen.prediction_bytes, origin=frozen.origin)


__all__ = [
    "BridgePredictionBundle",
    "BridgePredictorCapability",
    "BridgePredictorContractError",
    "MAX_PREDICTION_BYTES",
    "PREDICTION_COLUMNS",
    "PREDICTION_METHODS",
    "PREDICTION_SCHEMA",
    "audit_bridge_predictor_capability",
    "bridge_prediction_bundle_to_mapping",
    "bridge_predictor_capability_to_mapping",
    "build_bridge_prediction_bundle",
    "formal_protocol_declaration_identity_sha256",
    "formal_protocol_declaration_to_mapping",
    "prediction_payload_to_mapping",
]
