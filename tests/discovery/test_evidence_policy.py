from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.discovery.evidence_policy import (
    ClaimDecision,
    ClaimEvidence,
    EvidencePolicy,
    derive_joint_claim,
    evaluate_causal_claim,
    evaluate_spatial_claim,
)
from src.discovery.target_discovery.admission import build_claim_admission_status


ARTIFACT_IDENTITY = "a" * 64


def _policy() -> EvidencePolicy:
    return EvidencePolicy(
        protocol_version="hypersca-methods-v2.1",
        spatial_primary_metric="neighborhood_preservation_at_k",
        causal_primary_metric="directed_edge_average_precision",
        spatial_confirmatory_comparator="euclidean_autoencoder",
        spatial_attribution_comparator="hypersca_without_hierarchy_loss",
        causal_confirmatory_comparator="mean_difference",
        causal_attribution_comparator="hypersca_c_shared_only",
        adjusted_alpha=0.05,
        minimum_lower_bound=0.0,
    )


def _evidence(
    *,
    claim_id: str,
    comparator_id: str,
    metric: str,
    ci_low: float = 0.01,
    ci_high: float = 0.08,
    statuses: tuple[str, ...] = ("completed", "completed", "completed"),
) -> ClaimEvidence:
    all_completed = all(status == "completed" for status in statuses)
    return ClaimEvidence(
        claim_id=claim_id,
        protocol_version="hypersca-methods-v2.1",
        benchmark_id="osta_colon" if claim_id == "spatial" else "causalbench_k562_rpe1",
        primary_metric=metric,
        comparator_id=comparator_id,
        paired_estimate=0.04 if all_completed else None,
        ci_low=ci_low if all_completed else None,
        ci_high=ci_high if all_completed else None,
        adjusted_p_value=0.02 if all_completed else None,
        attempted_units=len(statuses),
        completed_units=sum(status == "completed" for status in statuses),
        run_statuses=statuses,
        required_artifacts=("primary_metric_units.csv", "primary_metric_summary.json"),
        artifact_identity=ARTIFACT_IDENTITY,
    )


def _spatial_pair(**kwargs: object) -> tuple[ClaimEvidence, ClaimEvidence]:
    return (
        _evidence(
            claim_id="spatial",
            comparator_id="euclidean_autoencoder",
            metric="neighborhood_preservation_at_k",
            **kwargs,
        ),
        _evidence(
            claim_id="spatial",
            comparator_id="hypersca_without_hierarchy_loss",
            metric="neighborhood_preservation_at_k",
            **kwargs,
        ),
    )


def _causal_pair(**kwargs: object) -> tuple[ClaimEvidence, ClaimEvidence]:
    return (
        _evidence(
            claim_id="causal",
            comparator_id="mean_difference",
            metric="directed_edge_average_precision",
            **kwargs,
        ),
        _evidence(
            claim_id="causal",
            comparator_id="hypersca_c_shared_only",
            metric="directed_edge_average_precision",
            **kwargs,
        ),
    )


def test_causal_claim_rejects_spatial_comparator_evidence_after_redesign() -> None:
    wrong = (
        _evidence(
            claim_id="causal",
            comparator_id="euclidean_autoencoder",
            metric="directed_edge_average_precision",
        ),
        _evidence(
            claim_id="causal",
            comparator_id="hypersca_without_hierarchy_loss",
            metric="directed_edge_average_precision",
        ),
    )

    decision = evaluate_causal_claim(wrong, _policy())

    assert decision.status == "blocked"
    assert "mean_difference" in decision.blocking_reasons[0]


def test_spatial_claim_requires_both_frozen_comparators() -> None:
    policy = _policy()
    admitted = evaluate_spatial_claim(_spatial_pair(), policy)
    missing_attribution = evaluate_spatial_claim((_spatial_pair()[0],), policy)

    assert admitted.status == "admitted"
    assert admitted.allowed_use == "spatial_representation_claim"
    assert missing_attribution.status == "blocked"
    assert "missing comparator" in missing_attribution.blocking_reasons[0]


def test_completed_but_inconclusive_evidence_stays_audit_only() -> None:
    decision = evaluate_causal_claim(_causal_pair(ci_low=-0.01), _policy())

    assert decision.status == "audit_only"
    assert decision.allowed_use == "audit_only"
    assert any("confidence interval" in reason for reason in decision.blocking_reasons)


def test_percentile_confidence_interval_may_exclude_the_point_estimate() -> None:
    evidence = list(_spatial_pair())
    first = evidence[0]
    evidence[0] = ClaimEvidence(
        claim_id=first.claim_id,
        protocol_version=first.protocol_version,
        benchmark_id=first.benchmark_id,
        primary_metric=first.primary_metric,
        comparator_id=first.comparator_id,
        paired_estimate=0.04,
        ci_low=0.05,
        ci_high=0.08,
        adjusted_p_value=0.02,
        attempted_units=first.attempted_units,
        completed_units=first.completed_units,
        run_statuses=first.run_statuses,
        required_artifacts=first.required_artifacts,
        artifact_identity=first.artifact_identity,
    )

    assert evaluate_spatial_claim(tuple(evidence), _policy()).status == "admitted"


def test_failed_confirmatory_run_blocks_the_claim_without_fabricating_zero() -> None:
    evidence = _spatial_pair(statuses=("completed", "failed_resource", "completed"))
    decision = evaluate_spatial_claim(evidence, _policy())

    assert decision.status == "blocked"
    assert decision.allowed_use == "not_used"
    assert any("failed_resource" in reason for reason in decision.blocking_reasons)


def test_incomplete_evidence_requires_missing_statistics_instead_of_synthetic_zero() -> (
    None
):
    failed = ClaimEvidence(
        claim_id="spatial",
        protocol_version="hypersca-methods-v1",
        benchmark_id="osta_colon",
        primary_metric="neighborhood_preservation_at_k",
        comparator_id="euclidean_autoencoder",
        paired_estimate=None,
        ci_low=None,
        ci_high=None,
        adjusted_p_value=None,
        attempted_units=3,
        completed_units=2,
        run_statuses=("completed", "failed_timeout", "completed"),
        required_artifacts=("method_status.json",),
        artifact_identity="f" * 64,
    )

    assert failed.paired_estimate is None
    with pytest.raises(ValueError, match="must not contain summary statistics"):
        ClaimEvidence(
            claim_id="spatial",
            protocol_version="hypersca-methods-v1",
            benchmark_id="osta_colon",
            primary_metric="neighborhood_preservation_at_k",
            comparator_id="euclidean_autoencoder",
            paired_estimate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            adjusted_p_value=1.0,
            attempted_units=3,
            completed_units=2,
            run_statuses=("completed", "failed_timeout", "completed"),
            required_artifacts=("method_status.json",),
            artifact_identity="f" * 64,
        )


def test_joint_admission_can_only_be_derived_from_two_admitted_claims() -> None:
    spatial = evaluate_spatial_claim(_spatial_pair(), _policy())
    causal = evaluate_causal_claim(_causal_pair(), _policy())
    joint = derive_joint_claim(spatial, causal)

    assert joint.status == "admitted"
    assert joint.allowed_use == "generalizable_spatial_causal_claim"
    assert joint.derived_from == (spatial.evidence_identity, causal.evidence_identity)
    with pytest.raises(ValueError, match="derive_joint_claim"):
        ClaimDecision(
            claim_id="joint",
            status="admitted",
            allowed_use="generalizable_spatial_causal_claim",
            blocking_reasons=(),
            evidence_identity="b" * 64,
            derived_from=("c" * 64, "d" * 64),
        )


def test_decisions_are_immutable_and_legacy_view_is_derived() -> None:
    spatial = evaluate_spatial_claim(_spatial_pair(), _policy())
    causal = evaluate_causal_claim(_causal_pair(ci_low=-0.01), _policy())
    joint = derive_joint_claim(spatial, causal)

    with pytest.raises(FrozenInstanceError):
        spatial.status = "blocked"  # type: ignore[misc]

    legacy = build_claim_admission_status(spatial, causal, joint).set_index("module")
    assert legacy.loc["hyperbolic_spatial", "status"] == spatial.status
    assert legacy.loc["hyperbolic_spatial", "allowed_use"] == spatial.allowed_use
    assert legacy.loc["hyperbolic_causal", "status"] == causal.status
    assert legacy.loc["hyperbolic_joint", "status"] == joint.status


@pytest.mark.parametrize(
    "field_override",
    [
        {"claim_id": "unknown"},
        {"adjusted_p_value": float("nan")},
        {"attempted_units": True},
        {"run_statuses": ("completed", "invented_status", "completed")},
        {"artifact_identity": "not-a-sha256"},
    ],
)
def test_claim_evidence_rejects_malformed_domain_values(
    field_override: dict[str, object]
) -> None:
    payload = {
        "claim_id": "spatial",
        "protocol_version": "hypersca-methods-v1",
        "benchmark_id": "osta_colon",
        "primary_metric": "neighborhood_preservation_at_k",
        "comparator_id": "euclidean_autoencoder",
        "paired_estimate": 0.04,
        "ci_low": 0.01,
        "ci_high": 0.08,
        "adjusted_p_value": 0.02,
        "attempted_units": 3,
        "completed_units": 3,
        "run_statuses": ("completed", "completed", "completed"),
        "required_artifacts": ("primary_metric_units.csv",),
        "artifact_identity": ARTIFACT_IDENTITY,
    }
    payload.update(field_override)
    with pytest.raises(ValueError):
        ClaimEvidence(**payload)  # type: ignore[arg-type]
