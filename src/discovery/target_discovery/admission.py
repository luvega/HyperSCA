"""Evidence-module admission gates for target discovery outputs."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.discovery.evidence_policy import ClaimDecision


ADMISSION_COLUMNS = [
    "module",
    "status",
    "required_artifacts",
    "controls_passed",
    "allowed_use",
    "blocking_reason",
]


def build_claim_admission_status(
    spatial: ClaimDecision,
    causal: ClaimDecision,
    joint: ClaimDecision,
) -> pd.DataFrame:
    """Expose immutable claim decisions as the legacy admission-table shape."""

    expected = (("spatial", spatial), ("causal", causal), ("joint", joint))
    if any(
        type(decision) is not ClaimDecision or decision.claim_id != claim_id
        for claim_id, decision in expected
    ):
        raise ValueError(
            "claim decisions do not match spatial, causal, and joint slots"
        )
    rows = [
        {
            "module": f"hyperbolic_{claim_id}",
            "status": decision.status,
            "required_artifacts": "admission_decision.json",
            "controls_passed": decision.status == "admitted",
            "allowed_use": decision.allowed_use,
            "blocking_reason": "; ".join(decision.blocking_reasons),
        }
        for claim_id, decision in expected
    ]
    frame = pd.DataFrame(rows, columns=ADMISSION_COLUMNS)
    frame["controls_passed"] = frame["controls_passed"].map(bool).astype(object)
    return frame


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series([], dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _has_ordinal_rank_display(ranking: pd.DataFrame) -> bool:
    if ranking.empty:
        return False
    basis_ok = "ranking_basis" in ranking and ranking["ranking_basis"].astype(str).eq("tiered_unweighted_evidence").all()
    score_ok = (
        "final_score_method" in ranking
        and ranking["final_score_method"].astype(str).eq("ordinal_rank_display_not_weighted_sum").all()
    )
    return bool(basis_ok and score_ok)


def _n_perturbation_targets(perturbation_scores: Any) -> int:
    if isinstance(perturbation_scores, pd.DataFrame):
        return int(len(perturbation_scores))
    if isinstance(perturbation_scores, dict):
        return int(len(perturbation_scores))
    return 0


def _embedding_row(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {
            "module": "scimilarity_embedding",
            "status": "blocked",
            "required_artifacts": "benchmarks/scimilarity_embedding/embedding_benchmark_summary.json",
            "controls_passed": False,
            "allowed_use": "not_used",
            "blocking_reason": "missing embedding benchmark summary",
        }
    decision = str(summary.get("decision", ""))
    if decision == "scimilarity_signal_detected":
        return {
            "module": "scimilarity_embedding",
            "status": "admitted",
            "required_artifacts": "embedding_benchmark_metrics.csv;embedding_benchmark_summary.json",
            "controls_passed": True,
            "allowed_use": "representation_support",
            "blocking_reason": "",
        }
    if decision == "no_material_advantage_over_euclidean_baseline":
        return {
            "module": "scimilarity_embedding",
            "status": "sidecar_only",
            "required_artifacts": "embedding_benchmark_metrics.csv;embedding_benchmark_summary.json",
            "controls_passed": True,
            "allowed_use": "baseline_comparison_only",
            "blocking_reason": "no material advantage over Euclidean baseline",
        }
    return {
        "module": "scimilarity_embedding",
        "status": "disabled",
        "required_artifacts": "embedding_benchmark_metrics.csv;embedding_benchmark_summary.json",
        "controls_passed": False,
        "allowed_use": "not_used",
        "blocking_reason": str(summary.get("rationale", decision or "no valid scimilarity advantage")),
    }


def _causal_controls_passed(summary: dict[str, Any] | None) -> bool:
    summary = summary or {}
    n_nulls = int(summary.get("n_null_control_matrices", 0) or 0)
    n_negative_pass = int(summary.get("n_negative_control_pass", 0) or 0)
    return n_nulls >= 10 and n_negative_pass > 0


def build_module_admission_status(
    *,
    ranking: pd.DataFrame | None = None,
    spatial_qc: pd.DataFrame | None = None,
    embedding_summary: dict[str, Any] | None = None,
    causal_summary: dict[str, Any] | None = None,
    perturbation_scores: Any = None,
) -> pd.DataFrame:
    """Build the auditable admission table for ranking and sidecar modules."""

    ranking = ranking if ranking is not None else pd.DataFrame()
    spatial_qc = spatial_qc if spatial_qc is not None else pd.DataFrame()
    ordinal = _has_ordinal_rank_display(ranking)
    spatial_pass_edges = int(_bool_series(spatial_qc, "passes").sum())
    causal_passed = _causal_controls_passed(causal_summary)
    n_perturbation_targets = _n_perturbation_targets(perturbation_scores)

    rows: list[dict[str, Any]] = [
        {
            "module": "main_ranking",
            "status": "admitted" if ordinal else "blocked",
            "required_artifacts": "scoring/target_ranking.csv;scoring/evidence_matrix.csv",
            "controls_passed": bool(ordinal),
            "allowed_use": "rank_order" if ordinal else "not_used",
            "blocking_reason": "" if ordinal else "ranking is not tiered_unweighted_evidence with ordinal final_score",
        },
        {
            "module": "final_score",
            "status": "admitted" if ordinal else "blocked",
            "required_artifacts": "scoring/target_ranking.csv",
            "controls_passed": bool(ordinal),
            "allowed_use": "ordinal_display_only" if ordinal else "not_used",
            "blocking_reason": "" if ordinal else "final_score is not marked as ordinal rank display",
        },
        {
            "module": "cell2location_context",
            "status": "admitted" if spatial_pass_edges > 0 else ("blocked" if spatial_qc.empty else "disabled"),
            "required_artifacts": "spatial/spatial_context_qc.csv",
            "controls_passed": spatial_pass_edges > 0,
            "allowed_use": "ranking_tiebreaker" if spatial_pass_edges > 0 else "not_used",
            "blocking_reason": "" if spatial_pass_edges > 0 else "no null-calibrated spatial context edges passed QC",
        },
        _embedding_row(embedding_summary),
        {
            "module": "causal_graph",
            "status": "sidecar_only",
            "required_artifacts": "benchmarks/causal_stability_scimilarity/causal_audit_summary.json",
            "controls_passed": causal_passed,
            "allowed_use": "mechanism_hypothesis_sidecar" if causal_passed else "exploratory_annotation_only",
            "blocking_reason": ""
            if causal_passed
            else "causal null-control threshold not met; keep out of ranking and mechanism conclusions",
        },
        {
            "module": "perturbation_proxy",
            "status": "sidecar_only" if n_perturbation_targets > 0 else "blocked",
            "required_artifacts": "perturbation/target_perturbation_scores.csv",
            "controls_passed": False,
            "allowed_use": "validation_prioritization_only" if n_perturbation_targets > 0 else "not_used",
            "blocking_reason": "proxy propagation is not a true perturbation benchmark",
        },
        {
            "module": "mechanism_prior_lr",
            "status": "sidecar_only",
            "required_artifacts": "mechanism evidence;prior-db audit",
            "controls_passed": False,
            "allowed_use": "mechanism_annotation_only",
            "blocking_reason": "requires prior bias audit before ranking use",
        },
    ]
    frame = pd.DataFrame(rows, columns=ADMISSION_COLUMNS)
    frame["controls_passed"] = frame["controls_passed"].map(lambda value: True if value else False).astype(object)
    return frame
