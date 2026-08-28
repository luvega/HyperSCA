"""Pure domain policy for admitting HyperSCA scientific claims.

This module intentionally performs no file I/O and has no pandas dependency.
Benchmark adapters must first turn their outputs into immutable ClaimEvidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import InitVar, asdict, dataclass


CLAIM_IDS = frozenset({"spatial", "causal"})
DECISION_STATUSES = frozenset({"not_evaluated", "blocked", "audit_only", "admitted"})
TERMINAL_RUN_STATUSES = frozenset(
    {
        "completed",
        "failed_invalid_input",
        "failed_invalid_output",
        "failed_timeout",
        "failed_resource",
        "failed_runtime",
        "not_applicable",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_JOINT_DERIVATION_PROOF = object()


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty built-in string")
    return value


def _require_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_v3_text(value: object, name: str) -> str:
    """Require a canonical, bounded text atom for the v3 public boundary."""
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty NFC-safe built-in string")
    return value


def _require_v3_identity(value: object, name: str) -> str:
    return _require_sha256(value, name)


def _v3_family_maps(
    policy: "EvidencePolicyV3",
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Return validated immutable-policy contents without trusting mutated slots."""
    policy.__post_init__()
    return dict(policy.family_primary_metrics), dict(policy.required_comparators)


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    """One preregistered paired comparison for a scientific claim."""

    claim_id: str
    protocol_version: str
    benchmark_id: str
    primary_metric: str
    comparator_id: str
    paired_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    adjusted_p_value: float | None
    attempted_units: int
    completed_units: int
    run_statuses: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    artifact_identity: str

    def __post_init__(self) -> None:
        if self.claim_id not in CLAIM_IDS or type(self.claim_id) is not str:
            raise ValueError(f"claim_id must be one of {sorted(CLAIM_IDS)}")
        for name in (
            "protocol_version",
            "benchmark_id",
            "primary_metric",
            "comparator_id",
        ):
            _require_text(getattr(self, name), name)
        if type(self.attempted_units) is not int or self.attempted_units <= 0:
            raise ValueError("attempted_units must be a positive built-in integer")
        if (
            type(self.completed_units) is not int
            or not 0 <= self.completed_units <= self.attempted_units
        ):
            raise ValueError(
                "completed_units must be a built-in integer in [0, attempted_units]"
            )
        if (
            type(self.run_statuses) is not tuple
            or len(self.run_statuses) != self.attempted_units
        ):
            raise ValueError(
                "run_statuses must be a tuple with one terminal status per attempted unit"
            )
        if any(
            type(status) is not str or status not in TERMINAL_RUN_STATUSES
            for status in self.run_statuses
        ):
            raise ValueError("run_statuses contains an unknown or non-terminal status")
        if (
            sum(status == "completed" for status in self.run_statuses)
            != self.completed_units
        ):
            raise ValueError("completed_units does not match run_statuses")
        statistics = (
            self.paired_estimate,
            self.ci_low,
            self.ci_high,
            self.adjusted_p_value,
        )
        if self.completed_units == self.attempted_units:
            estimate = _require_float(self.paired_estimate, "paired_estimate")
            ci_low = _require_float(self.ci_low, "ci_low")
            ci_high = _require_float(self.ci_high, "ci_high")
            adjusted_p_value = _require_float(self.adjusted_p_value, "adjusted_p_value")
            if ci_low > ci_high:
                raise ValueError("ci_low must not exceed ci_high")
            if not 0.0 <= adjusted_p_value <= 1.0:
                raise ValueError("adjusted_p_value must lie in [0, 1]")
        elif any(value is not None for value in statistics):
            raise ValueError("incomplete evidence must not contain summary statistics")
        if type(self.required_artifacts) is not tuple or not self.required_artifacts:
            raise ValueError("required_artifacts must be a non-empty tuple")
        if any(
            type(item) is not str
            or not item
            or item.startswith("/")
            or ".." in item.split("/")
            for item in self.required_artifacts
        ):
            raise ValueError(
                "required_artifacts must contain safe relative artifact names"
            )
        if len(set(self.required_artifacts)) != len(self.required_artifacts):
            raise ValueError("required_artifacts must not contain duplicates")
        _require_sha256(self.artifact_identity, "artifact_identity")


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """Frozen confirmatory rules shared by spatial and causal claims."""

    protocol_version: str
    spatial_primary_metric: str
    causal_primary_metric: str
    spatial_confirmatory_comparator: str
    spatial_attribution_comparator: str
    causal_confirmatory_comparator: str
    causal_attribution_comparator: str
    adjusted_alpha: float
    minimum_lower_bound: float

    def __post_init__(self) -> None:
        for name in (
            "protocol_version",
            "spatial_primary_metric",
            "causal_primary_metric",
            "spatial_confirmatory_comparator",
            "spatial_attribution_comparator",
            "causal_confirmatory_comparator",
            "causal_attribution_comparator",
        ):
            _require_text(getattr(self, name), name)
        comparators = {
            self.spatial_confirmatory_comparator,
            self.spatial_attribution_comparator,
            self.causal_confirmatory_comparator,
            self.causal_attribution_comparator,
        }
        if len(comparators) != 4:
            raise ValueError("claim-specific primary comparators must differ")
        adjusted_alpha = _require_float(self.adjusted_alpha, "adjusted_alpha")
        _require_float(self.minimum_lower_bound, "minimum_lower_bound")
        if not 0.0 < adjusted_alpha < 1.0:
            raise ValueError("adjusted_alpha must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    """Immutable allowed use for one claim boundary."""

    claim_id: str
    status: str
    allowed_use: str
    blocking_reasons: tuple[str, ...]
    evidence_identity: str
    derived_from: tuple[str, ...] = ()
    _joint_derivation_proof: InitVar[object | None] = None

    def __post_init__(self, _joint_derivation_proof: object | None) -> None:
        if type(self.claim_id) is not str or self.claim_id not in {*CLAIM_IDS, "joint"}:
            raise ValueError("claim_id must be spatial, causal, or joint")
        if type(self.status) is not str or self.status not in DECISION_STATUSES:
            raise ValueError("status is not a valid claim decision status")
        _require_text(self.allowed_use, "allowed_use")
        if type(self.blocking_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.blocking_reasons
        ):
            raise ValueError("blocking_reasons must be a tuple of non-empty strings")
        _require_sha256(self.evidence_identity, "evidence_identity")
        if type(self.derived_from) is not tuple or any(
            _SHA256_PATTERN.fullmatch(value) is None
            for value in self.derived_from
            if type(value) is str
        ):
            raise ValueError("derived_from must contain SHA-256 identities")
        if any(type(value) is not str for value in self.derived_from):
            raise ValueError("derived_from must contain SHA-256 identities")
        if self.claim_id == "joint":
            if len(self.derived_from) != 2:
                raise ValueError(
                    "joint decisions require two source decision identities"
                )
            if (
                self.status == "admitted"
                and _joint_derivation_proof is not _JOINT_DERIVATION_PROOF
            ):
                raise ValueError(
                    "admitted joint decisions must be created by derive_joint_claim"
                )
        elif self.derived_from:
            raise ValueError("only joint decisions may contain derived_from identities")


@dataclass(frozen=True, slots=True)
class V3ClaimEvidence:
    """One immutable paired comparison under the three-family v3 protocol."""

    claim_id: str
    protocol_version: str
    primary_metric: str
    comparator_id: str
    paired_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    nominal_p_value: float | None
    attempted_units: int
    completed_units: int
    evidence_role: str
    artifact_identity: str

    def __post_init__(self) -> None:
        if type(self.claim_id) is not str or self.claim_id not in {
            "spatial",
            "intracellular_causal",
            "bridge",
        }:
            raise ValueError("claim_id is not a v3 evidence family")
        for name in (
            "protocol_version",
            "primary_metric",
            "comparator_id",
            "evidence_role",
        ):
            _require_v3_text(getattr(self, name), name)
        if self.evidence_role not in {
            "confirmatory",
            "attribution",
            "application_only",
            "pilot_audit_only",
            "synthetic_audit_only",
            "audit_only",
        }:
            raise ValueError("evidence_role is not permitted by the v3 protocol")
        if type(self.attempted_units) is not int or self.attempted_units <= 0:
            raise ValueError("attempted_units must be a positive built-in integer")
        if (
            type(self.completed_units) is not int
            or self.completed_units < 0
            or self.completed_units > self.attempted_units
        ):
            raise ValueError(
                "completed_units must be a built-in integer in [0, attempted_units]"
            )
        statistics = (
            self.paired_estimate,
            self.ci_low,
            self.ci_high,
            self.nominal_p_value,
        )
        if self.completed_units == self.attempted_units:
            _require_float(self.paired_estimate, "paired_estimate")
            ci_low = _require_float(self.ci_low, "ci_low")
            ci_high = _require_float(self.ci_high, "ci_high")
            p_value = _require_float(self.nominal_p_value, "nominal_p_value")
            if ci_low > ci_high:
                raise ValueError("ci_low must not exceed ci_high")
            if not 0.0 <= p_value <= 1.0:
                raise ValueError("nominal_p_value must lie in [0, 1]")
        elif any(value is not None for value in statistics):
            raise ValueError("incomplete evidence must not contain summary statistics")
        _require_v3_identity(self.artifact_identity, "artifact_identity")

    def to_mapping(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "claim_id": self.claim_id,
            "protocol_version": self.protocol_version,
            "primary_metric": self.primary_metric,
            "comparator_id": self.comparator_id,
            "paired_estimate": self.paired_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "nominal_p_value": self.nominal_p_value,
            "attempted_units": self.attempted_units,
            "completed_units": self.completed_units,
            "evidence_role": self.evidence_role,
            "artifact_identity": self.artifact_identity,
        }


@dataclass(frozen=True, slots=True)
class EvidencePolicyV3:
    """Frozen v3 promotion policy for spatial, causal, and bridge evidence."""

    protocol_version: str
    family_primary_metrics: tuple[tuple[str, str], ...]
    required_comparators: tuple[tuple[str, tuple[str, ...]], ...]
    nominal_alpha: float
    minimum_lower_bound: float
    bridge_role: str
    integrated_claim_enabled: bool

    def __post_init__(self) -> None:
        _require_v3_text(self.protocol_version, "protocol_version")
        if self.protocol_version != "hypersca-methods-v3.0":
            raise ValueError("protocol_version is frozen to hypersca-methods-v3.0")
        expected_metrics = (
            ("spatial", "neighborhood_preservation_at_k"),
            ("intracellular_causal", "directed_edge_average_precision"),
            ("bridge", "neighbor_effect_rmse"),
        )
        if (
            type(self.family_primary_metrics) is not tuple
            or len(self.family_primary_metrics) != len(expected_metrics)
            or any(
                type(item) is not tuple or len(item) != 2
                for item in self.family_primary_metrics
            )
            or self.family_primary_metrics != expected_metrics
        ):
            raise ValueError("family_primary_metrics must equal the frozen v3 families")
        expected_comparators = (
            (
                "spatial",
                (
                    "matched_euclidean_autoencoder",
                    "hypersca_without_hierarchy_loss",
                ),
            ),
            (
                "intracellular_causal",
                ("matched_non_hyperbolic_baseline", "hypersca_c_shared_only"),
            ),
            (
                "bridge",
                ("matched_euclidean_spatial_causal", "hypersca_own_only"),
            ),
        )
        if (
            type(self.required_comparators) is not tuple
            or len(self.required_comparators) != len(expected_comparators)
            or any(
                type(item) is not tuple or len(item) != 2 or type(item[1]) is not tuple
                for item in self.required_comparators
            )
            or self.required_comparators != expected_comparators
        ):
            raise ValueError(
                "required_comparators must equal the frozen v3 comparators"
            )
        for family, metric in self.family_primary_metrics:
            if type(family) is not str or type(metric) is not str:
                raise ValueError("family_primary_metrics must contain built-in tuples")
            _require_v3_text(family, "family")
            _require_v3_text(metric, "primary_metric")
        for family, comparators in self.required_comparators:
            if type(family) is not str or type(comparators) is not tuple:
                raise ValueError("required_comparators must contain built-in tuples")
            if len(comparators) != 2 or len(set(comparators)) != 2:
                raise ValueError(
                    "each v3 family requires exactly two distinct comparators"
                )
            for comparator in comparators:
                _require_v3_text(comparator, "comparator_id")
        alpha = _require_float(self.nominal_alpha, "nominal_alpha")
        lower_bound = _require_float(self.minimum_lower_bound, "minimum_lower_bound")
        if alpha != 0.05:
            raise ValueError("nominal_alpha must equal the frozen v3 value 0.05")
        if lower_bound != 0.0:
            raise ValueError("minimum_lower_bound must equal the frozen v3 value 0.0")
        if type(self.bridge_role) is not str or self.bridge_role not in {
            "pilot_audit_only",
            "confirmatory",
        }:
            raise ValueError("bridge_role must be pilot_audit_only or confirmatory")
        if type(self.integrated_claim_enabled) is not bool:
            raise ValueError("integrated_claim_enabled must be a built-in bool")
        if self.integrated_claim_enabled is not (self.bridge_role == "confirmatory"):
            raise ValueError("integrated_claim_enabled must match the bridge_role")

    def to_mapping(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "protocol_version": self.protocol_version,
            "family_primary_metrics": self.family_primary_metrics,
            "required_comparators": self.required_comparators,
            "nominal_alpha": self.nominal_alpha,
            "minimum_lower_bound": self.minimum_lower_bound,
            "bridge_role": self.bridge_role,
            "integrated_claim_enabled": self.integrated_claim_enabled,
        }


@dataclass(frozen=True, slots=True)
class V3ClaimDecision:
    """Immutable public-use decision for a v3 evidence family or integration."""

    claim_id: str
    protocol_version: str
    status: str
    allowed_use: str
    blocking_reasons: tuple[str, ...]
    evidence_identity: str
    nominal_p_value: float | None
    multiplicity_adjustment: str
    evidence_role: str
    application_evidence_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.claim_id) is not str or self.claim_id not in {
            "spatial",
            "intracellular_causal",
            "bridge",
            "integrated",
        }:
            raise ValueError("claim_id is not a v3 decision family")
        for name in (
            "protocol_version",
            "allowed_use",
            "multiplicity_adjustment",
            "evidence_role",
        ):
            _require_v3_text(getattr(self, name), name)
        if type(self.status) is not str or self.status not in DECISION_STATUSES:
            raise ValueError("status is not a valid claim decision status")
        if type(self.blocking_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.blocking_reasons
        ):
            raise ValueError("blocking_reasons must be a tuple of non-empty strings")
        _require_v3_identity(self.evidence_identity, "evidence_identity")
        if self.nominal_p_value is not None:
            p_value = _require_float(self.nominal_p_value, "nominal_p_value")
            if not 0.0 <= p_value <= 1.0:
                raise ValueError("nominal_p_value must lie in [0, 1]")
        if self.multiplicity_adjustment not in {
            "none_family_specific",
            "none_intersection_union",
            "not_applicable",
        }:
            raise ValueError("multiplicity_adjustment is not valid for v3")
        if self.evidence_role not in {
            "confirmatory",
            "pilot_audit_only",
            "integrated",
        }:
            raise ValueError("evidence_role is not valid for a v3 decision")
        if type(self.application_evidence_identities) is not tuple or any(
            type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None
            for value in self.application_evidence_identities
        ):
            raise ValueError(
                "application_evidence_identities must be SHA-256 identities"
            )
        if len(set(self.application_evidence_identities)) != len(
            self.application_evidence_identities
        ):
            raise ValueError(
                "application_evidence_identities must not contain duplicates"
            )

    def to_mapping(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "claim_id": self.claim_id,
            "protocol_version": self.protocol_version,
            "status": self.status,
            "allowed_use": self.allowed_use,
            "blocking_reasons": self.blocking_reasons,
            "evidence_identity": self.evidence_identity,
            "nominal_p_value": self.nominal_p_value,
            "multiplicity_adjustment": self.multiplicity_adjustment,
            "evidence_role": self.evidence_role,
            "application_evidence_identities": self.application_evidence_identities,
        }


def _blocked_decision(
    claim_id: str, reasons: tuple[str, ...], identity_payload: object
) -> ClaimDecision:
    return ClaimDecision(
        claim_id=claim_id,
        status="blocked",
        allowed_use="not_used",
        blocking_reasons=reasons,
        evidence_identity=_canonical_sha256(identity_payload),
    )


def _evaluate_claim(
    evidence: tuple[ClaimEvidence, ...],
    policy: EvidencePolicy,
    *,
    claim_id: str,
    primary_metric: str,
    admitted_use: str,
) -> ClaimDecision:
    if type(evidence) is not tuple or any(
        type(item) is not ClaimEvidence for item in evidence
    ):
        raise ValueError("evidence must be a tuple of ClaimEvidence values")
    by_comparator: dict[str, ClaimEvidence] = {}
    for item in evidence:
        if item.comparator_id in by_comparator:
            raise ValueError(
                f"duplicate evidence for comparator {item.comparator_id!r}"
            )
        if item.claim_id != claim_id:
            raise ValueError(f"{claim_id} evaluation received {item.claim_id} evidence")
        if item.protocol_version != policy.protocol_version:
            raise ValueError("evidence protocol_version does not match the policy")
        if item.primary_metric != primary_metric:
            raise ValueError("evidence primary_metric does not match the policy")
        by_comparator[item.comparator_id] = item

    required = (
        (
            policy.spatial_confirmatory_comparator,
            policy.spatial_attribution_comparator,
        )
        if claim_id == "spatial"
        else (
            policy.causal_confirmatory_comparator,
            policy.causal_attribution_comparator,
        )
    )
    missing = tuple(
        comparator for comparator in required if comparator not in by_comparator
    )
    identity_payload = {
        "claim_id": claim_id,
        "policy": asdict(policy),
        "evidence": [
            asdict(item)
            for item in sorted(evidence, key=lambda item: item.comparator_id)
        ],
    }
    if missing:
        return _blocked_decision(
            claim_id,
            (f"missing comparator evidence: {', '.join(missing)}",),
            identity_payload,
        )

    failures: list[str] = []
    for comparator in required:
        item = by_comparator[comparator]
        failed_statuses = tuple(
            status for status in item.run_statuses if status != "completed"
        )
        if failed_statuses or item.completed_units != item.attempted_units:
            failures.append(
                f"{comparator} has incomplete terminal evidence: {', '.join(failed_statuses) or 'missing completed units'}"
            )
    if failures:
        return _blocked_decision(claim_id, tuple(failures), identity_payload)

    inconclusive: list[str] = []
    for comparator in required:
        item = by_comparator[comparator]
        assert item.ci_low is not None
        assert item.adjusted_p_value is not None
        if item.ci_low <= policy.minimum_lower_bound:
            inconclusive.append(
                f"{comparator} confidence interval lower bound {item.ci_low} did not exceed {policy.minimum_lower_bound}"
            )
        if item.adjusted_p_value >= policy.adjusted_alpha:
            inconclusive.append(
                f"{comparator} adjusted p-value {item.adjusted_p_value} did not pass {policy.adjusted_alpha}"
            )
    identity = _canonical_sha256(identity_payload)
    if inconclusive:
        return ClaimDecision(
            claim_id=claim_id,
            status="audit_only",
            allowed_use="audit_only",
            blocking_reasons=tuple(inconclusive),
            evidence_identity=identity,
        )
    return ClaimDecision(
        claim_id=claim_id,
        status="admitted",
        allowed_use=admitted_use,
        blocking_reasons=(),
        evidence_identity=identity,
    )


def evaluate_spatial_claim(
    evidence: tuple[ClaimEvidence, ...], policy: EvidencePolicy
) -> ClaimDecision:
    return _evaluate_claim(
        evidence,
        policy,
        claim_id="spatial",
        primary_metric=policy.spatial_primary_metric,
        admitted_use="spatial_representation_claim",
    )


def evaluate_causal_claim(
    evidence: tuple[ClaimEvidence, ...], policy: EvidencePolicy
) -> ClaimDecision:
    return _evaluate_claim(
        evidence,
        policy,
        claim_id="causal",
        primary_metric=policy.causal_primary_metric,
        admitted_use="interventional_recovery_claim",
    )


def derive_joint_claim(spatial: ClaimDecision, causal: ClaimDecision) -> ClaimDecision:
    if type(spatial) is not ClaimDecision or spatial.claim_id != "spatial":
        raise ValueError("spatial must be a spatial ClaimDecision")
    if type(causal) is not ClaimDecision or causal.claim_id != "causal":
        raise ValueError("causal must be a causal ClaimDecision")
    derived_from = (spatial.evidence_identity, causal.evidence_identity)
    identity = _canonical_sha256(
        {
            "claim_id": "joint",
            "spatial": asdict(spatial),
            "causal": asdict(causal),
        }
    )
    reasons: tuple[str, ...]
    if "blocked" in {spatial.status, causal.status}:
        status = "blocked"
        allowed_use = "not_used"
        reasons = (
            "joint claim is blocked because at least one component claim is blocked",
        )
    elif spatial.status == causal.status == "admitted":
        status = "admitted"
        allowed_use = "generalizable_spatial_causal_claim"
        reasons = ()
    else:
        status = "audit_only"
        allowed_use = "audit_only"
        reasons = (
            "joint claim requires both spatial and causal claims to be admitted",
        )
    return ClaimDecision(
        claim_id="joint",
        status=status,
        allowed_use=allowed_use,
        blocking_reasons=reasons,
        evidence_identity=identity,
        derived_from=derived_from,
        _joint_derivation_proof=_JOINT_DERIVATION_PROOF,
    )


def _v3_decision(
    *,
    claim_id: str,
    policy: EvidencePolicyV3,
    status: str,
    allowed_use: str,
    reasons: tuple[str, ...],
    identity_payload: object,
    nominal_p_value: float | None,
    multiplicity_adjustment: str,
    application_evidence_identities: tuple[str, ...],
    evidence_role: str | None = None,
) -> V3ClaimDecision:
    if evidence_role is None:
        evidence_role = policy.bridge_role if claim_id == "bridge" else "confirmatory"
    return V3ClaimDecision(
        claim_id=claim_id,
        protocol_version=policy.protocol_version,
        status=status,
        allowed_use=allowed_use,
        blocking_reasons=reasons,
        evidence_identity=_canonical_sha256(identity_payload),
        nominal_p_value=nominal_p_value,
        multiplicity_adjustment=multiplicity_adjustment,
        evidence_role=evidence_role,
        application_evidence_identities=application_evidence_identities,
    )


def _evaluate_v3_family(
    evidence: tuple[V3ClaimEvidence, ...],
    policy: EvidencePolicyV3,
    claim_id: str,
) -> V3ClaimDecision:
    if type(policy) is not EvidencePolicyV3:
        raise ValueError("policy must be an EvidencePolicyV3")
    family_metrics, comparators_by_family = _v3_family_maps(policy)
    if claim_id not in family_metrics:
        raise ValueError("claim_id is not a v3 evidence family")
    if type(evidence) is not tuple or any(
        type(item) is not V3ClaimEvidence for item in evidence
    ):
        raise ValueError("evidence must be a tuple of V3ClaimEvidence values")
    required = comparators_by_family[claim_id]
    if len(evidence) > len(required) + 6:
        raise ValueError("evidence contains too many records for one v3 family")
    application: list[V3ClaimEvidence] = []
    scientific: list[V3ClaimEvidence] = []
    for item in evidence:
        item.__post_init__()
        if item.claim_id != claim_id:
            raise ValueError(f"{claim_id} evaluation received {item.claim_id} evidence")
        if item.protocol_version != policy.protocol_version:
            raise ValueError("evidence protocol_version does not match the policy")
        if item.primary_metric != family_metrics[claim_id]:
            raise ValueError("evidence primary_metric does not match the policy")
        if item.evidence_role == "application_only":
            application.append(item)
        else:
            scientific.append(item)
    if len(scientific) > len(required):
        raise ValueError("scientific evidence contains extra comparator evidence")
    by_comparator: dict[str, V3ClaimEvidence] = {}
    for item in scientific:
        if item.comparator_id not in required:
            raise ValueError("scientific evidence contains an unregistered comparator")
        if item.comparator_id in by_comparator:
            raise ValueError(
                f"duplicate evidence for comparator {item.comparator_id!r}"
            )
        by_comparator[item.comparator_id] = item

    application_identities = tuple(
        sorted(item.artifact_identity for item in application)
    )
    identity_payload = {
        "claim_id": claim_id,
        "policy": policy.to_mapping(),
        "scientific_evidence": [
            item.to_mapping()
            for item in sorted(scientific, key=lambda item: item.comparator_id)
        ],
        "application_evidence": [
            item.to_mapping()
            for item in sorted(application, key=lambda item: item.artifact_identity)
        ],
    }
    missing = tuple(
        comparator for comparator in required if comparator not in by_comparator
    )
    multiplicity = (
        "none_intersection_union" if claim_id == "bridge" else "none_family_specific"
    )
    paired_counts = {
        (
            item.attempted_units,
            item.completed_units,
        )
        for item in by_comparator.values()
    }
    incomplete = tuple(
        comparator
        for comparator in required
        if comparator in by_comparator
        and by_comparator[comparator].completed_units
        != by_comparator[comparator].attempted_units
    )
    blocking_reasons: list[str] = []
    if missing:
        blocking_reasons.append(f"missing comparator evidence: {', '.join(missing)}")
        if claim_id == "bridge" and by_comparator:
            blocking_reasons.append("bridge evidence was attempted")
    if incomplete or len(paired_counts) > 1:
        detail = "common paired attempted/completed units are required"
        if incomplete:
            detail = f"{detail}; incomplete evidence: {', '.join(incomplete)}"
        blocking_reasons.append(detail)
    if blocking_reasons:
        return _v3_decision(
            claim_id=claim_id,
            policy=policy,
            status="blocked",
            allowed_use="not_used",
            reasons=tuple(blocking_reasons),
            identity_payload=identity_payload,
            nominal_p_value=None,
            multiplicity_adjustment=multiplicity,
            application_evidence_identities=application_identities,
        )

    expected_roles = (
        (policy.bridge_role, policy.bridge_role)
        if claim_id == "bridge"
        else ("confirmatory", "attribution")
    )
    role_failures = tuple(
        f"{comparator} must have evidence role {expected_role}"
        for comparator, expected_role in zip(required, expected_roles)
        if by_comparator[comparator].evidence_role != expected_role
    )
    if role_failures:
        return _v3_decision(
            claim_id=claim_id,
            policy=policy,
            status="audit_only",
            allowed_use="audit_only",
            reasons=role_failures,
            identity_payload=identity_payload,
            nominal_p_value=None,
            multiplicity_adjustment=multiplicity,
            application_evidence_identities=application_identities,
        )

    audit_roles = {"pilot_audit_only", "synthetic_audit_only", "audit_only"}
    restricted = tuple(
        comparator
        for comparator in required
        if by_comparator[comparator].evidence_role in audit_roles
    )
    if restricted:
        return _v3_decision(
            claim_id=claim_id,
            policy=policy,
            status="audit_only",
            allowed_use="audit_only",
            reasons=(
                f"audit-only evidence cannot be promoted: {', '.join(restricted)}",
            ),
            identity_payload=identity_payload,
            nominal_p_value=None,
            multiplicity_adjustment=multiplicity,
            application_evidence_identities=application_identities,
        )

    p_values = tuple(
        by_comparator[comparator].nominal_p_value for comparator in required
    )
    if any(p_value is None for p_value in p_values):
        raise ValueError("completed evidence must contain nominal p-values")
    nominal_p_value = max(p_value for p_value in p_values if p_value is not None)
    assert all(by_comparator[comparator].ci_low is not None for comparator in required)
    failures: list[str] = []
    for comparator in required:
        item = by_comparator[comparator]
        assert item.ci_low is not None
        if item.ci_low <= policy.minimum_lower_bound:
            failures.append(
                f"{comparator} confidence interval lower bound {item.ci_low} did not exceed {policy.minimum_lower_bound}"
            )
        assert item.nominal_p_value is not None
        if item.nominal_p_value >= policy.nominal_alpha:
            failures.append(
                f"{comparator} nominal p-value {item.nominal_p_value} did not pass {policy.nominal_alpha}"
            )
    if failures:
        return _v3_decision(
            claim_id=claim_id,
            policy=policy,
            status="audit_only",
            allowed_use="audit_only",
            reasons=tuple(failures),
            identity_payload=identity_payload,
            nominal_p_value=nominal_p_value,
            multiplicity_adjustment=multiplicity,
            application_evidence_identities=application_identities,
        )
    allowed_uses = {
        "spatial": "spatial_representation_preservation_gain",
        "intracellular_causal": "intracellular_intervention_directed_relation_recovery",
        "bridge": "spatial_neighbour_response_recovery",
    }
    return _v3_decision(
        claim_id=claim_id,
        policy=policy,
        status="admitted",
        allowed_use=allowed_uses[claim_id],
        reasons=(),
        identity_payload=identity_payload,
        nominal_p_value=nominal_p_value,
        multiplicity_adjustment=multiplicity,
        application_evidence_identities=application_identities,
    )


def evaluate_v3_claim(
    evidence: tuple[V3ClaimEvidence, ...], policy: EvidencePolicyV3
) -> V3ClaimDecision:
    """Evaluate either of the standalone v3 evidence families."""
    if type(evidence) is not tuple or not evidence:
        raise ValueError("evidence must contain one v3 family")
    claim_id = evidence[0].claim_id if type(evidence[0]) is V3ClaimEvidence else None
    if claim_id not in {"spatial", "intracellular_causal"}:
        raise ValueError(
            "evaluate_v3_claim only accepts spatial or intracellular_causal"
        )
    return _evaluate_v3_family(evidence, policy, claim_id)


def evaluate_bridge_claim(
    evidence: tuple[V3ClaimEvidence, ...], policy: EvidencePolicyV3
) -> V3ClaimDecision:
    """Evaluate the bridge as an intersection-union test over both comparators."""
    if type(evidence) is not tuple:
        raise ValueError("evidence must be a tuple of V3ClaimEvidence values")
    if evidence and (
        type(evidence[0]) is not V3ClaimEvidence or evidence[0].claim_id != "bridge"
    ):
        raise ValueError("evaluate_bridge_claim only accepts bridge evidence")
    return _evaluate_v3_family(evidence, policy, "bridge")


def _is_purely_missing_bridge_block(decision: V3ClaimDecision) -> bool:
    """Classify only the closed missing-evidence reason codes as unavailable."""
    missing_reasons = {
        "missing comparator evidence: matched_euclidean_spatial_causal",
        "missing comparator evidence: hypersca_own_only",
        "missing comparator evidence: matched_euclidean_spatial_causal, hypersca_own_only",
    }
    return (
        decision.status == "blocked"
        and bool(decision.blocking_reasons)
        and all(reason in missing_reasons for reason in decision.blocking_reasons)
    )


def derive_integrated_claim(
    decisions: tuple[V3ClaimDecision, ...], policy: EvidencePolicyV3
) -> V3ClaimDecision:
    """Allow an integrated claim only when all three v3 families are admitted."""
    if type(policy) is not EvidencePolicyV3:
        raise ValueError("policy must be an EvidencePolicyV3")
    policy.__post_init__()
    if (
        type(decisions) is not tuple
        or len(decisions) != 3
        or any(type(decision) is not V3ClaimDecision for decision in decisions)
    ):
        raise ValueError(
            "decisions must contain exactly one decision per required claim"
        )
    expected = {"spatial", "intracellular_causal", "bridge"}
    by_claim: dict[str, V3ClaimDecision] = {}
    for decision in decisions:
        decision.__post_init__()
        if decision.claim_id in by_claim or decision.claim_id not in expected:
            raise ValueError(
                "decisions must contain exactly one decision per required claim"
            )
        if decision.protocol_version != policy.protocol_version:
            raise ValueError("decision protocol_version does not match the policy")
        by_claim[decision.claim_id] = decision
    if set(by_claim) != expected:
        raise ValueError(
            "decisions must contain exactly one decision per required claim"
        )
    identity_payload = {
        "policy": policy.to_mapping(),
        "decisions": [by_claim[claim].to_mapping() for claim in sorted(expected)],
    }
    component_statuses = {decision.status for decision in by_claim.values()}
    status: str
    allowed_use: str
    reasons: tuple[str, ...]
    bridge_unavailable = (
        by_claim["spatial"].status == "admitted"
        and by_claim["intracellular_causal"].status == "admitted"
        and _is_purely_missing_bridge_block(by_claim["bridge"])
    )
    if bridge_unavailable:
        status, allowed_use, reasons = (
            "audit_only",
            "separate_module_claims_only",
            ("integrated claim requires available bridge evidence",),
        )
    elif "blocked" in component_statuses:
        status, allowed_use, reasons = (
            "blocked",
            "not_used",
            ("integrated claim is blocked because at least one component is blocked",),
        )
    elif (
        policy.integrated_claim_enabled is True
        and all(decision.status == "admitted" for decision in by_claim.values())
        and by_claim["spatial"].allowed_use
        == "spatial_representation_preservation_gain"
        and by_claim["intracellular_causal"].allowed_use
        == "intracellular_intervention_directed_relation_recovery"
        and by_claim["bridge"].allowed_use == "spatial_neighbour_response_recovery"
        and by_claim["bridge"].evidence_role == policy.bridge_role
        and by_claim["bridge"].multiplicity_adjustment == "none_intersection_union"
    ):
        status, allowed_use, reasons = (
            "admitted",
            "integrated_spatial_causal_gain",
            (),
        )
    else:
        status, allowed_use, reasons = (
            "audit_only",
            "separate_module_claims_only",
            (
                "integrated claim requires three admitted confirmatory evidence families",
            ),
        )
    return _v3_decision(
        claim_id="integrated",
        policy=policy,
        status=status,
        allowed_use=allowed_use,
        reasons=reasons,
        identity_payload=identity_payload,
        nominal_p_value=None,
        multiplicity_adjustment="not_applicable",
        application_evidence_identities=tuple(
            sorted(
                {
                    identity
                    for decision in by_claim.values()
                    for identity in decision.application_evidence_identities
                }
            )
        ),
        evidence_role="integrated",
    )


__all__ = [
    "ClaimDecision",
    "ClaimEvidence",
    "EvidencePolicy",
    "EvidencePolicyV3",
    "TERMINAL_RUN_STATUSES",
    "V3ClaimDecision",
    "V3ClaimEvidence",
    "derive_integrated_claim",
    "derive_joint_claim",
    "evaluate_bridge_claim",
    "evaluate_causal_claim",
    "evaluate_spatial_claim",
    "evaluate_v3_claim",
]
