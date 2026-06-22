from __future__ import annotations

import pandas as pd


def test_build_module_admission_status_enforces_evidence_gates():
    from src.discovery.target_discovery.admission import build_module_admission_status

    ranking = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "ranking_basis": ["tiered_unweighted_evidence", "tiered_unweighted_evidence"],
            "final_score_method": [
                "ordinal_rank_display_not_weighted_sum",
                "ordinal_rank_display_not_weighted_sum",
            ],
        }
    )
    spatial_qc = pd.DataFrame(
        {
            "source": ["Tumor"],
            "target": ["Fibro"],
            "passes": [False],
            "sample_support": [0],
        }
    )
    causal_summary = {"n_null_control_matrices": 0, "n_negative_control_pass": 0}
    perturbation_scores = pd.DataFrame({"gene": ["A"], "gradient_decay_r2": [0.3]})

    status = build_module_admission_status(
        ranking=ranking,
        spatial_qc=spatial_qc,
        embedding_summary={},
        causal_summary=causal_summary,
        perturbation_scores=perturbation_scores,
    )

    assert list(status.columns) == [
        "module",
        "status",
        "required_artifacts",
        "controls_passed",
        "allowed_use",
        "blocking_reason",
    ]
    by_module = status.set_index("module")
    assert by_module.loc["main_ranking", "status"] == "admitted"
    assert by_module.loc["final_score", "allowed_use"] == "ordinal_display_only"
    assert by_module.loc["cell2location_context", "status"] == "disabled"
    assert by_module.loc["cell2location_context", "allowed_use"] == "not_used"
    assert by_module.loc["scimilarity_embedding", "status"] == "blocked"
    assert "missing" in by_module.loc["scimilarity_embedding", "blocking_reason"]
    assert by_module.loc["causal_graph", "status"] == "sidecar_only"
    assert by_module.loc["causal_graph", "controls_passed"] is False
    assert by_module.loc["causal_graph", "allowed_use"] == "exploratory_annotation_only"
    assert by_module.loc["perturbation_proxy", "status"] == "sidecar_only"
    assert by_module.loc["perturbation_proxy", "allowed_use"] == "validation_prioritization_only"
