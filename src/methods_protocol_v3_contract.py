"""Dependency-free immutable v3 methods protocol contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import cast


def _science() -> dict[str, object]:
    """Return fresh frozen literals; no mutable module state is scientific truth."""
    return {
        "version": "hypersca-methods-v3.0",
        "claim_ids": ("spatial", "intracellular_causal", "bridge"),
        "metrics": (
            "neighborhood_preservation_at_k",
            "directed_edge_average_precision",
            "neighbor_effect_rmse",
        ),
        "pilot_seeds": (11, 23, 47),
        "release_seeds": (11, 23, 47, 71, 101),
    }


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > 256:
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
        raise ValueError(
            f"{name} must be a {len(expected)}-item built-in list or tuple"
        )
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != len(expected):
        raise ValueError(
            f"{name} must be a {len(expected)}-item built-in list or tuple"
        )
    frozen = tuple(
        _safe_text(item, f"{name}[{index}]") for index, item in enumerate(items)
    )
    if len(set(frozen)) != len(frozen) or frozen != expected:
        raise ValueError(f"{name} must match the frozen protocol values")
    return frozen


def _frozen_integer_tuple(
    value: object, name: str, expected: tuple[int, ...]
) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{name} must be a {len(expected)}-item built-in list or tuple"
        )
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != len(expected) or any(type(item) is not int for item in items):
        raise ValueError(f"{name} must contain the frozen built-in integers")
    frozen = cast(tuple[int, ...], tuple(items))
    if len(set(frozen)) != len(frozen) or frozen != expected:
        raise ValueError(f"{name} must match the preregistered seeds")
    return frozen


def _sha256(value: object, name: str) -> str:
    text = _safe_text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
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
        science = _science()
        version = _safe_text(self.protocol_version, "protocol_version")
        claim_ids = _frozen_text_tuple(self.claim_ids, "claim_ids", science["claim_ids"])  # type: ignore[arg-type]
        metrics = _frozen_text_tuple(self.primary_metrics, "primary_metrics", science["metrics"])  # type: ignore[arg-type]
        pilot = _frozen_integer_tuple(self.pilot_seeds, "pilot_seeds", science["pilot_seeds"])  # type: ignore[arg-type]
        release = _frozen_integer_tuple(self.release_seeds, "release_seeds", science["release_seeds"])  # type: ignore[arg-type]
        multiple = _safe_text(self.multiple_testing, "multiple_testing")
        gate = _safe_text(self.integrated_gate, "integrated_gate")
        role = _safe_text(self.bridge_role, "bridge_role")
        crc = _safe_text(self.crc_role, "crc_role")
        capability = _sha256(
            self.capability_identity_sha256, "capability_identity_sha256"
        )
        if version != science["version"]:
            raise ValueError("protocol_version is frozen to hypersca-methods-v3.0")
        if (
            type(self.bootstrap_resamples) is not int
            or self.bootstrap_resamples != 10_000
        ):
            raise ValueError("bootstrap_resamples is frozen to 10000")
        if (
            type(self.confidence) is not float
            or not math.isfinite(self.confidence)
            or self.confidence != 0.95
        ):
            raise ValueError("confidence is frozen to 0.95")
        if (
            multiple != "distinct_families_no_cross_adjustment"
            or gate != "intersection_union_all_three"
        ):
            raise ValueError("multiple_testing and integrated_gate are frozen")
        if role not in ("pilot_audit_only", "confirmatory"):
            raise ValueError("bridge_role must be pilot_audit_only or confirmatory")
        if type(
            self.integrated_claim_enabled
        ) is not bool or self.integrated_claim_enabled is not (role == "confirmatory"):
            raise ValueError("integrated_claim_enabled must match the bridge_role")
        if crc != "application_only":
            raise ValueError("CRC must remain application_only")
        for name, value in (
            ("protocol_version", version),
            ("claim_ids", claim_ids),
            ("primary_metrics", metrics),
            ("pilot_seeds", pilot),
            ("release_seeds", release),
            ("capability_identity_sha256", capability),
            ("multiple_testing", multiple),
            ("integrated_gate", gate),
            ("bridge_role", role),
            ("crc_role", crc),
        ):
            object.__setattr__(self, name, value)


def build_methods_protocol_v3(
    *, bridge_role: str, capability_identity_sha256: str
) -> MethodsProtocolV3:
    science = _science()
    return MethodsProtocolV3(
        protocol_version=science["version"],  # type: ignore[arg-type]
        claim_ids=science["claim_ids"],  # type: ignore[arg-type]
        primary_metrics=science["metrics"],  # type: ignore[arg-type]
        pilot_seeds=science["pilot_seeds"],  # type: ignore[arg-type]
        release_seeds=science["release_seeds"],  # type: ignore[arg-type]
        bootstrap_resamples=10_000,
        confidence=0.95,
        multiple_testing="distinct_families_no_cross_adjustment",
        integrated_gate="intersection_union_all_three",
        bridge_role=bridge_role,
        capability_identity_sha256=capability_identity_sha256,
        integrated_claim_enabled=bridge_role == "confirmatory",
        crc_role="application_only",
    )


def _snapshot(protocol: MethodsProtocolV3) -> MethodsProtocolV3:
    if type(protocol) is not MethodsProtocolV3:
        raise ValueError("protocol must be MethodsProtocolV3")
    return MethodsProtocolV3(
        protocol.protocol_version,
        protocol.claim_ids,
        protocol.primary_metrics,
        protocol.pilot_seeds,
        protocol.release_seeds,
        protocol.bootstrap_resamples,
        protocol.confidence,
        protocol.multiple_testing,
        protocol.integrated_gate,
        protocol.bridge_role,
        protocol.capability_identity_sha256,
        protocol.integrated_claim_enabled,
        protocol.crc_role,
    )


def protocol_to_mapping_v3(protocol: MethodsProtocolV3) -> dict[str, object]:
    snapshot = _snapshot(protocol)
    bands = lambda values: [
        {"name": name, "minimum_distance": low, "maximum_distance": high}
        for name, low, high in values
    ]
    return {
        "schema": "hypersca_methods_protocol_v3",
        "protocol_version": snapshot.protocol_version,
        "bridge_role": snapshot.bridge_role,
        "claims": {
            "spatial": {
                "claim_id": snapshot.claim_ids[0],
                "benchmark": "osta_colon",
                "primary_metric": snapshot.primary_metrics[0],
                "primary_k": 15,
                "comparators": {
                    "confirmatory": "matched_euclidean_autoencoder",
                    "attribution": "hypersca_without_hierarchy_loss",
                },
            },
            "intracellular_causal": {
                "claim_id": snapshot.claim_ids[1],
                "benchmark": "causalbench_intracellular_interventional_causal_recovery",
                "primary_metric": snapshot.primary_metrics[1],
                "comparators": {
                    "confirmatory": "matched_non_hyperbolic_baseline",
                    "attribution": "hypersca_c_shared_only",
                },
            },
            "bridge": {
                "claim_id": snapshot.claim_ids[2],
                "primary_metric": snapshot.primary_metrics[2],
                "primary_bands": bands((("proximal", 1, 5), ("local", 6, 15))),
                "secondary_bands": bands((("transition", 16, 30), ("distal", 31, 60))),
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
            "pilot_seeds": list(snapshot.pilot_seeds),
            "release_seeds": list(snapshot.release_seeds),
            "bootstrap_resamples": snapshot.bootstrap_resamples,
            "confidence": snapshot.confidence,
            "family_decision_rule": "nominal_one_sided_paired_95_percent_ci_lower_bound_gt_zero",
            "cross_family_adjustment": "none",
            "multiple_testing": snapshot.multiple_testing,
        },
        "integration": {
            "gate": snapshot.integrated_gate,
            "integrated_claim_enabled": snapshot.integrated_claim_enabled,
            "evidence_policy": "all_three_family_gates_required",
            "extra_p_value": "none",
        },
        "governance": {
            "crc_role": snapshot.crc_role,
            "crc_may_tune_hyperparameters": False,
            "crc_may_select_baselines": False,
            "crc_may_change_thresholds": False,
            "crc_may_rescue_promotion": False,
        },
        "capability_identity_sha256": snapshot.capability_identity_sha256,
    }


def protocol_identity_v3(protocol: MethodsProtocolV3) -> str:
    payload = json.dumps(
        protocol_to_mapping_v3(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
