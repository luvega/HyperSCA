"""Pure domain policy for admitting HyperSCA scientific claims.

This module intentionally performs no file I/O and has no pandas dependency.
Benchmark adapters must first turn their outputs into immutable ClaimEvidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
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


__all__ = [
    "ClaimDecision",
    "ClaimEvidence",
    "EvidencePolicy",
    "TERMINAL_RUN_STATUSES",
    "derive_joint_claim",
    "evaluate_causal_claim",
    "evaluate_spatial_claim",
]
