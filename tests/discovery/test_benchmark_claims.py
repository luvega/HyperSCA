from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.discovery.benchmark_claims import (
    BootstrapComparisonEvidence,
    build_holm_adjusted_claim_evidence,
)
from src.discovery.evidence_policy import EvidencePolicy
from src.evaluation.benchmark_evidence import PairedBootstrapSummary


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


def _comparison(
    comparator: str, p_value: float, identity_character: str
) -> BootstrapComparisonEvidence:
    return BootstrapComparisonEvidence(
        comparator_id=comparator,
        summary=PairedBootstrapSummary(
            estimate=0.04,
            ci_low=0.01,
            ci_high=0.08,
            one_sided_p_value=p_value,
            resamples=10_000,
            random_seed=191,
            resampling_scheme="registered_test_scheme",
        ),
        run_statuses=("completed", "completed", "completed"),
        required_artifacts=("primary_metric_units.csv", "primary_metric_summary.json"),
        artifact_identity=identity_character * 64,
    )


def test_claim_adapter_uses_weaker_comparator_then_holm_across_two_claims() -> None:
    policy = _policy()
    spatial = (
        _comparison(policy.spatial_confirmatory_comparator, 0.01, "a"),
        _comparison(policy.spatial_attribution_comparator, 0.04, "b"),
    )
    causal = (
        _comparison(policy.causal_confirmatory_comparator, 0.02, "c"),
        _comparison(policy.causal_attribution_comparator, 0.03, "d"),
    )

    spatial_evidence, causal_evidence = build_holm_adjusted_claim_evidence(
        policy=policy,
        spatial_comparisons=spatial,
        causal_comparisons=causal,
    )

    assert {item.comparator_id for item in spatial_evidence} == {
        policy.spatial_confirmatory_comparator,
        policy.spatial_attribution_comparator,
    }
    assert [item.adjusted_p_value for item in spatial_evidence] == pytest.approx(
        [0.06, 0.06]
    )
    assert [item.adjusted_p_value for item in causal_evidence] == pytest.approx(
        [0.06, 0.06]
    )
    assert all(item.benchmark_id == "osta_colon" for item in spatial_evidence)
    assert all(item.benchmark_id == "causalbench_k562_rpe1" for item in causal_evidence)


def test_claim_adapter_rejects_missing_or_duplicate_frozen_comparators() -> None:
    policy = _policy()
    complete = (
        _comparison(policy.spatial_confirmatory_comparator, 0.01, "a"),
        _comparison(policy.spatial_attribution_comparator, 0.02, "b"),
    )
    causal_complete = (
        _comparison(policy.causal_confirmatory_comparator, 0.01, "c"),
        _comparison(policy.causal_attribution_comparator, 0.02, "d"),
    )
    with pytest.raises(ValueError, match="exactly the frozen comparators"):
        build_holm_adjusted_claim_evidence(
            policy=policy,
            spatial_comparisons=(complete[0],),
            causal_comparisons=causal_complete,
        )
    with pytest.raises(ValueError, match="exactly the frozen comparators"):
        build_holm_adjusted_claim_evidence(
            policy=policy,
            spatial_comparisons=(complete[0], complete[0]),
            causal_comparisons=causal_complete,
        )


def test_claim_adapter_retains_terminal_failures_without_metric_fabrication() -> None:
    policy = _policy()
    failed = BootstrapComparisonEvidence(
        comparator_id=policy.spatial_confirmatory_comparator,
        summary=None,
        run_statuses=("completed", "failed_timeout", "completed"),
        required_artifacts=("method_status.json",),
        artifact_identity="e" * 64,
    )
    passed = _comparison(policy.spatial_attribution_comparator, 0.02, "f")

    spatial_evidence, _ = build_holm_adjusted_claim_evidence(
        policy=policy,
        spatial_comparisons=(failed, passed),
        causal_comparisons=(
            _comparison(policy.causal_confirmatory_comparator, 0.01, "1"),
            _comparison(policy.causal_attribution_comparator, 0.02, "2"),
        ),
    )

    retained = next(
        item
        for item in spatial_evidence
        if item.comparator_id == policy.spatial_confirmatory_comparator
    )
    assert retained.run_statuses == ("completed", "failed_timeout", "completed")
    assert retained.completed_units == 2
    assert retained.ci_low is None
    with pytest.raises(FrozenInstanceError):
        failed.run_statuses = ("completed",)  # type: ignore[misc]
