"""Bridge pure benchmark summaries into immutable scientific claim evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.discovery.evidence_policy import (
    ClaimEvidence,
    EvidencePolicy,
    TERMINAL_RUN_STATUSES,
)
from src.evaluation.benchmark_evidence import (
    PairedBootstrapSummary,
    holm_adjust_two_claims,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class BootstrapComparisonEvidence:
    """One comparator's complete bootstrap result or retained terminal failure."""

    comparator_id: str
    summary: PairedBootstrapSummary | None
    run_statuses: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    artifact_identity: str

    def __post_init__(self) -> None:
        if type(self.comparator_id) is not str or not self.comparator_id.strip():
            raise ValueError("comparator_id must be a non-empty built-in string")
        if type(self.run_statuses) is not tuple or not self.run_statuses:
            raise ValueError("run_statuses must be a non-empty tuple")
        if any(
            type(status) is not str or status not in TERMINAL_RUN_STATUSES
            for status in self.run_statuses
        ):
            raise ValueError(
                "run_statuses must contain only registered terminal statuses"
            )
        all_completed = all(status == "completed" for status in self.run_statuses)
        if all_completed and type(self.summary) is not PairedBootstrapSummary:
            raise ValueError(
                "completed comparison evidence requires a bootstrap summary"
            )
        if not all_completed and self.summary is not None:
            raise ValueError(
                "incomplete comparison evidence must not contain a bootstrap summary"
            )
        if type(self.required_artifacts) is not tuple or not self.required_artifacts:
            raise ValueError("required_artifacts must be a non-empty tuple")
        if (
            type(self.artifact_identity) is not str
            or _SHA256.fullmatch(self.artifact_identity) is None
        ):
            raise ValueError("artifact_identity must be a lowercase SHA-256 digest")


def _ordered_comparisons(
    comparisons: tuple[BootstrapComparisonEvidence, ...],
    policy: EvidencePolicy,
    *,
    claim_id: str,
) -> tuple[BootstrapComparisonEvidence, BootstrapComparisonEvidence]:
    if type(comparisons) is not tuple or any(
        type(item) is not BootstrapComparisonEvidence for item in comparisons
    ):
        raise ValueError(
            "comparisons must be a tuple of BootstrapComparisonEvidence values"
        )
    by_id = {item.comparator_id: item for item in comparisons}
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
    if len(comparisons) != 2 or len(by_id) != 2 or set(by_id) != set(required):
        raise ValueError("each claim requires exactly the frozen comparators")
    return by_id[required[0]], by_id[required[1]]


def _claim_raw_p(
    comparisons: tuple[BootstrapComparisonEvidence, BootstrapComparisonEvidence]
) -> float:
    summaries = tuple(item.summary for item in comparisons)
    if any(summary is None for summary in summaries):
        return 1.0
    return max(
        summary.one_sided_p_value for summary in summaries if summary is not None
    )


def _claim_evidence(
    *,
    claim_id: str,
    benchmark_id: str,
    primary_metric: str,
    comparisons: tuple[BootstrapComparisonEvidence, BootstrapComparisonEvidence],
    adjusted_p_value: float,
    policy: EvidencePolicy,
) -> tuple[ClaimEvidence, ClaimEvidence]:
    evidence: list[ClaimEvidence] = []
    for comparison in comparisons:
        summary = comparison.summary
        evidence.append(
            ClaimEvidence(
                claim_id=claim_id,
                protocol_version=policy.protocol_version,
                benchmark_id=benchmark_id,
                primary_metric=primary_metric,
                comparator_id=comparison.comparator_id,
                paired_estimate=None if summary is None else summary.estimate,
                ci_low=None if summary is None else summary.ci_low,
                ci_high=None if summary is None else summary.ci_high,
                adjusted_p_value=None if summary is None else adjusted_p_value,
                attempted_units=len(comparison.run_statuses),
                completed_units=sum(
                    status == "completed" for status in comparison.run_statuses
                ),
                run_statuses=comparison.run_statuses,
                required_artifacts=comparison.required_artifacts,
                artifact_identity=comparison.artifact_identity,
            )
        )
    return evidence[0], evidence[1]


def build_holm_adjusted_claim_evidence(
    *,
    policy: EvidencePolicy,
    spatial_comparisons: tuple[BootstrapComparisonEvidence, ...],
    causal_comparisons: tuple[BootstrapComparisonEvidence, ...],
) -> tuple[tuple[ClaimEvidence, ClaimEvidence], tuple[ClaimEvidence, ClaimEvidence]]:
    """Apply an intersection-union comparator gate, then Holm across claims.

    A claim's raw p-value is the larger (weaker) of its confirmatory and
    attribution comparator p-values.  Holm correction is then applied to the
    spatial and causal claims.  An incomplete claim is conservatively entered
    into the two-claim family with p=1 while its own statistics remain absent.
    """

    if type(policy) is not EvidencePolicy:
        raise ValueError("policy must be EvidencePolicy")
    spatial = _ordered_comparisons(
        spatial_comparisons, policy, claim_id="spatial"
    )
    causal = _ordered_comparisons(causal_comparisons, policy, claim_id="causal")
    spatial_adjusted, causal_adjusted = holm_adjust_two_claims(
        (_claim_raw_p(spatial), _claim_raw_p(causal))
    )
    return (
        _claim_evidence(
            claim_id="spatial",
            benchmark_id="osta_colon",
            primary_metric=policy.spatial_primary_metric,
            comparisons=spatial,
            adjusted_p_value=spatial_adjusted,
            policy=policy,
        ),
        _claim_evidence(
            claim_id="causal",
            benchmark_id="causalbench_k562_rpe1",
            primary_metric=policy.causal_primary_metric,
            comparisons=causal,
            adjusted_p_value=causal_adjusted,
            policy=policy,
        ),
    )


__all__ = ["BootstrapComparisonEvidence", "build_holm_adjusted_claim_evidence"]
