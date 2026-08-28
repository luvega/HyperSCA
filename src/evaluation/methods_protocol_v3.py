"""Immutable v3 evidence protocol for the public HyperSCA methods release."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import cast


BRIDGE_PRIMARY_BANDS = (("proximal", 1, 5), ("local", 6, 15))
BRIDGE_SECONDARY_BANDS = (("transition", 16, 30), ("distal", 31, 60))

_CLAIM_IDS = ("spatial", "intracellular_causal", "bridge")
_PRIMARY_METRICS = (
    "neighborhood_preservation_at_k",
    "directed_edge_average_precision",
    "neighbor_effect_rmse",
)
_PILOT_SEEDS = (11, 23, 47)
_RELEASE_SEEDS = (11, 23, 47, 71, 101)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_TEXT_LENGTH = 256


def _safe_text(value: object, name: str) -> str:
    """Validate bounded, canonical plain text before it enters the protocol."""
    if type(value) is not str or not value or len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{name} must be a bounded non-empty built-in string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must be trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{name} must not contain control text")
    return value


def _frozen_text_tuple(
    value: object, name: str, expected: tuple[str, ...]
) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a {len(expected)}-item built-in list or tuple")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != len(expected):
        raise ValueError(f"{name} must be a {len(expected)}-item built-in list or tuple")
    frozen: tuple[str, ...] = tuple(
        _safe_text(item, f"{name}[{index}]") for index, item in enumerate(items)
    )
    if len(set(frozen)) != len(frozen):
        raise ValueError(f"{name} must not contain duplicates")
    if frozen != expected:
        raise ValueError(f"{name} must match the frozen protocol values")
    return frozen


def _frozen_integer_tuple(
    value: object, name: str, expected: tuple[int, ...]
) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a {len(expected)}-item built-in list or tuple")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != len(expected):
        raise ValueError(f"{name} must be a {len(expected)}-item built-in list or tuple")
    frozen: tuple[object, ...] = tuple(items)
    if any(type(item) is not int for item in frozen):
        raise ValueError(f"{name} must contain built-in integers")
    if len(set(frozen)) != len(frozen):
        raise ValueError(f"{name} must not contain duplicates")
    if frozen != expected:
        raise ValueError(f"{name} must match the preregistered seeds")
    return cast(tuple[int, ...], frozen)


def _sha256(value: object, name: str) -> str:
    text = _safe_text(value, name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True, slots=True)
class MethodsProtocolV3:
    protocol_version: str
    claim_ids: tuple[str, str, str]
    primary_metrics: tuple[str, str, str]
    pilot_seeds: tuple[int, int, int]
    release_seeds: tuple[int, int, int, int, int]
    bootstrap_resamples: int
    confidence: float
    multiple_testing: str
    integrated_gate: str
    bridge_role: str
    capability_identity_sha256: str
    integrated_claim_enabled: bool
    crc_role: str

    def __post_init__(self) -> None:
        protocol_version = _safe_text(self.protocol_version, "protocol_version")
        claim_ids = _frozen_text_tuple(self.claim_ids, "claim_ids", _CLAIM_IDS)
        primary_metrics = _frozen_text_tuple(
            self.primary_metrics, "primary_metrics", _PRIMARY_METRICS
        )
        pilot_seeds = _frozen_integer_tuple(self.pilot_seeds, "pilot_seeds", _PILOT_SEEDS)
        release_seeds = _frozen_integer_tuple(
            self.release_seeds, "release_seeds", _RELEASE_SEEDS
        )
        multiple_testing = _safe_text(self.multiple_testing, "multiple_testing")
        integrated_gate = _safe_text(self.integrated_gate, "integrated_gate")
        bridge_role = _safe_text(self.bridge_role, "bridge_role")
        crc_role = _safe_text(self.crc_role, "crc_role")
        capability_identity_sha256 = _sha256(
            self.capability_identity_sha256, "capability_identity_sha256"
        )

        if protocol_version != "hypersca-methods-v3.0":
            raise ValueError("protocol_version is frozen to hypersca-methods-v3.0")
        if type(self.bootstrap_resamples) is not int or self.bootstrap_resamples != 10_000:
            raise ValueError("bootstrap_resamples is frozen to 10000")
        if type(self.confidence) is not float or not math.isfinite(self.confidence):
            raise ValueError("confidence must be a finite built-in float")
        if self.confidence != 0.95:
            raise ValueError("confidence is frozen to 0.95")
        if multiple_testing != "distinct_families_no_cross_adjustment":
            raise ValueError("multiple_testing is frozen to distinct-family testing")
        if integrated_gate != "intersection_union_all_three":
            raise ValueError("integrated_gate is frozen to all-three intersection-union")
        if bridge_role not in ("pilot_audit_only", "confirmatory"):
            raise ValueError("bridge_role must be pilot_audit_only or confirmatory")
        if type(self.integrated_claim_enabled) is not bool:
            raise ValueError("integrated_claim_enabled must be a built-in boolean")
        if self.integrated_claim_enabled is not (bridge_role == "confirmatory"):
            raise ValueError("integrated_claim_enabled must match the bridge_role")
        if crc_role != "application_only":
            raise ValueError("CRC must remain application_only")

        object.__setattr__(self, "protocol_version", protocol_version)
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "primary_metrics", primary_metrics)
        object.__setattr__(self, "pilot_seeds", pilot_seeds)
        object.__setattr__(self, "release_seeds", release_seeds)
        object.__setattr__(
            self, "capability_identity_sha256", capability_identity_sha256
        )
        object.__setattr__(self, "multiple_testing", multiple_testing)
        object.__setattr__(self, "integrated_gate", integrated_gate)
        object.__setattr__(self, "bridge_role", bridge_role)
        object.__setattr__(self, "crc_role", crc_role)


def build_methods_protocol_v3(
    *, bridge_role: str, capability_identity_sha256: str
) -> MethodsProtocolV3:
    """Build the sole permitted v3 protocol shape for a bridge asset role."""
    return MethodsProtocolV3(
        protocol_version="hypersca-methods-v3.0",
        claim_ids=_CLAIM_IDS,
        primary_metrics=_PRIMARY_METRICS,
        pilot_seeds=_PILOT_SEEDS,
        release_seeds=_RELEASE_SEEDS,
        bootstrap_resamples=10_000,
        confidence=0.95,
        multiple_testing="distinct_families_no_cross_adjustment",
        integrated_gate="intersection_union_all_three",
        bridge_role=bridge_role,
        capability_identity_sha256=capability_identity_sha256,
        integrated_claim_enabled=bridge_role == "confirmatory",
        crc_role="application_only",
    )


def _band_mappings(bands: tuple[tuple[str, int, int], ...]) -> list[dict[str, object]]:
    return [
        {"name": name, "minimum_distance": minimum, "maximum_distance": maximum}
        for name, minimum, maximum in bands
    ]


def protocol_to_mapping_v3(protocol: MethodsProtocolV3) -> dict[str, object]:
    """Return the ordered public contract used to calculate the protocol identity."""
    if type(protocol) is not MethodsProtocolV3:
        raise ValueError("protocol must be MethodsProtocolV3")
    return {
        "schema": "hypersca_methods_protocol_v3",
        "protocol_version": protocol.protocol_version,
        "claims": {
            "spatial": {
                "claim_id": protocol.claim_ids[0],
                "benchmark": "osta_colon",
                "primary_metric": protocol.primary_metrics[0],
                "comparators": {
                    "confirmatory": "matched_euclidean_autoencoder",
                    "attribution": "hypersca_without_hierarchy_loss",
                },
            },
            "intracellular_causal": {
                "claim_id": protocol.claim_ids[1],
                "benchmark": "causalbench_intracellular_interventional_causal_recovery",
                "primary_metric": protocol.primary_metrics[1],
                "comparators": {
                    "confirmatory": "matched_non_hyperbolic_baseline",
                    "attribution": "hypersca_without_hierarchy_loss",
                },
            },
            "bridge": {
                "claim_id": protocol.claim_ids[2],
                "primary_metric": protocol.primary_metrics[2],
                "primary_bands": _band_mappings(BRIDGE_PRIMARY_BANDS),
                "secondary_bands": _band_mappings(BRIDGE_SECONDARY_BANDS),
                "iut": {
                    "comparators": [
                        "matched_euclidean_spatial_causal",
                        "hypersca_own_only",
                    ],
                    "within_iut_multiplicity_adjustment": "none",
                    "claim_p_value": "max_component_p",
                },
            },
        },
        "statistics": {
            "pilot_seeds": list(protocol.pilot_seeds),
            "release_seeds": list(protocol.release_seeds),
            "bootstrap_resamples": protocol.bootstrap_resamples,
            "confidence": protocol.confidence,
            "family_decision_rule": "nominal_one_sided_paired_95_percent_ci_lower_bound_gt_zero",
            "cross_family_adjustment": "none",
            "multiple_testing": protocol.multiple_testing,
        },
        "integration": {
            "gate": protocol.integrated_gate,
            "integrated_claim_enabled": protocol.integrated_claim_enabled,
            "evidence_policy": "all_three_family_gates_required",
            "extra_p_value": "none",
        },
        "governance": {
            "crc_role": protocol.crc_role,
            "crc_may_tune_hyperparameters": False,
            "crc_may_select_baselines": False,
            "crc_may_change_thresholds": False,
            "crc_may_rescue_promotion": False,
        },
        "capability_identity_sha256": protocol.capability_identity_sha256,
    }


def protocol_identity_v3(protocol: MethodsProtocolV3) -> str:
    """Hash the canonical v3 public contract deterministically."""
    payload = json.dumps(
        protocol_to_mapping_v3(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
