from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.discovery.target_discovery.artifacts import ArtifactWriter
from src.discovery.target_discovery.heavy_stages import EvidenceScoringStage
from src.discovery.target_discovery.scoring import compare_modes, retain_hubs_and_combos, score_candidates


def _candidate_pool():
    return pd.DataFrame(
        {
            "gene": ["GENE_A", "GENE_B"],
            "cross_queue_count": [2, 1],
            "direction_consistency": [1.0, 0.5],
            "mean_abs_lfc": [1.0, 0.2],
            "celltypes_neu": ["Fibroblast_S1", "Macrophage"],
            "is_anchor": [False, False],
            "is_ifng_target": [False, False],
            "mean_lfc": [1.0, 0.2],
            "min_padj": [0.01, 0.05],
        }
    )


def _step2():
    return {
        "node_labels": ["Fibroblast_S1", "Macrophage"],
        "betweenness": {"Fibroblast_S1": 0.7, "Macrophage": 0.2},
        "flow_edges": [{"source": "GENE_A", "target": "REC_A"}],
        "metrics": {"graph_sparsity": 0.5, "hsic_independence": 0.8, "known_axis_recall": 0.4, "mean_bootstrap_freq": 0.3},
    }


def test_score_candidates_returns_ranked_frame():
    cluster_expr = pd.DataFrame({"GENE_A": [1.0, 2.0]}, index=["Fibroblast_S1", "Macrophage"])
    ranked = score_candidates(_candidate_pool(), _step2(), _step2(), {"GENE_A": {"n_ranked": 3}}, {}, cluster_expr)
    assert list(ranked["rank"]) == [1, 2]
    assert "final_score" in ranked.columns
    assert ranked["ranking_basis"].eq("tiered_unweighted_evidence").all()
    assert ranked["final_score_method"].eq("ordinal_rank_display_not_weighted_sum").all()
    assert ranked.iloc[0]["final_score"] >= ranked.iloc[1]["final_score"]


def test_evidence_gated_ranking_does_not_promote_sidecar_only_signal():
    candidate_pool = pd.DataFrame(
        {
            "gene": ["DIRECT_EVIDENCE", "SIDECAR_ONLY"],
            "cross_queue_count": [2, 1],
            "direction_consistency": [0.8, 1.0],
            "mean_abs_lfc": [0.5, 3.0],
            "min_padj": [0.01, 1e-8],
            "neg_log10_padj": [2.0, 8.0],
            "celltypes_neu": ["Fibroblast_S1", "Macrophage"],
        }
    )
    step2 = {
        "node_labels": ["Fibroblast_S1", "Macrophage"],
        "betweenness": {"Fibroblast_S1": 0.0, "Macrophage": 1.0},
        "flow_edges": [{"source": "SIDECAR_ONLY", "target": "REC"}],
    }
    perturbation = {
        "SIDECAR_ONLY": {
            "n_ranked": 30,
            "spatial_quality": {
                "gradient_decay_r2": 1.0,
                "propagation_depth": 4,
                "moran_i_effect": 1.0,
            },
        }
    }

    ranked = score_candidates(
        candidate_pool,
        step2,
        step2,
        perturbation,
        {},
        pd.DataFrame(
            {"DIRECT_EVIDENCE": [1.0, 0.0], "SIDECAR_ONLY": [0.0, 1.0]},
            index=["Fibroblast_S1", "Macrophage"],
        ),
        score_profile="evidence_gated",
    )

    assert ranked.iloc[0]["gene"] == "DIRECT_EVIDENCE"
    assert ranked.iloc[1]["gene"] == "SIDECAR_ONLY"
    assert ranked["final_score"].tolist() == [1.0, 0.5]
    assert ranked.loc[ranked["gene"] == "SIDECAR_ONLY", "s_causal"].iloc[0] > ranked.loc[
        ranked["gene"] == "DIRECT_EVIDENCE", "s_causal"
    ].iloc[0]
    assert ranked.loc[ranked["gene"] == "SIDECAR_ONLY", "s_spatial"].iloc[0] > ranked.loc[
        ranked["gene"] == "DIRECT_EVIDENCE", "s_spatial"
    ].iloc[0]
    assert ranked["sidecar_only_modules"].str.contains("causal_graph").all()
    assert ranked["rank_rationale"].str.contains("sidecar_not_ranked").all()


def test_legacy_weighted_profile_is_explicitly_marked():
    cluster_expr = pd.DataFrame(
        {"GENE_A": [1.0, 2.0]}, index=["Fibroblast_S1", "Macrophage"]
    )

    ranked = score_candidates(
        _candidate_pool(),
        _step2(),
        _step2(),
        {"GENE_A": {"n_ranked": 3}},
        {},
        cluster_expr,
        score_profile="legacy_full",
    )

    assert ranked["ranking_basis"].eq("legacy_weighted_sum").all()
    assert ranked["final_score_method"].eq("legacy_weighted_sum").all()
    assert ranked["rank_rationale"].eq("legacy_weighted_reproduction_only").all()
    assert ranked["sidecar_only_modules"].eq("").all()


def test_retain_hubs_and_combos_keeps_ranked_rows_without_anchor_priority():
    ranking = _candidate_pool()
    ranking["rank"] = [1, 2]
    step3 = {"GENE_A": {"ranked_targets": pd.DataFrame([{"ligand": "GENE_A", "receptor": "REC_A", "target_priority_score": 0.9}])}}
    hubs, combos = retain_hubs_and_combos(ranking, step3)
    assert list(hubs["gene"]) == ["GENE_A", "GENE_B"]
    assert hubs["is_anchor"].eq(False).all()
    assert len(combos) == 1


def test_compare_modes_returns_geometry_and_step2_summary():
    geom = {"metrics": {"separation": 2.0}}
    comp = compare_modes(geom, geom, _step2(), _step2(), {}, {}, pd.DataFrame({"gene": ["GENE_A"], "final_score": [1.0]}))
    assert comp["geometry"]["hyp_separation"] == 2.0
    assert "hyp_graph_sparsity" in comp["step2"]


def test_evidence_scoring_stage_writes_ranking_policy_and_admission(tmp_path):
    writer = ArtifactWriter(tmp_path, run_id="ranking_gate")
    context = SimpleNamespace(
        config=SimpleNamespace(score_profile="evidence_gated"),
        writer=writer,
    )
    stage = EvidenceScoringStage()
    geom = {"metrics": {"separation": 1.0}}
    step2 = _step2()
    perturbation = {"GENE_A": {"n_ranked": 3}}

    outputs = stage.run(
        context,
        {
            "candidate_pool": _candidate_pool(),
            "cluster_expression": pd.DataFrame(
                {"GENE_A": [1.0, 2.0]}, index=["Fibroblast_S1", "Macrophage"]
            ),
            "geometry_results": {"hyperbolic": geom, "euclidean": geom},
            "causal_results": {"hyperbolic": step2, "euclidean": step2},
            "perturbation_results": {"hyperbolic": perturbation, "euclidean": {}},
        },
    )

    admission = outputs["module_admission"].set_index("module")
    assert admission.loc["main_ranking", "status"] == "admitted"
    assert admission.loc["causal_graph", "status"] == "sidecar_only"
    assert (writer.run_dir / "scoring" / "module_admission.csv").exists()
    assert (writer.run_dir / "scoring" / "ranking_policy.json").exists()
