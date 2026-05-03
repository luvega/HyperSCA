from __future__ import annotations

import pandas as pd

from src.discovery.target_discovery.scoring import compare_modes, retain_hubs_and_combos, score_candidates


def _candidate_pool():
    return pd.DataFrame(
        {
            "gene": ["A", "MFAP2"],
            "cross_queue_count": [2, 1],
            "direction_consistency": [1.0, 0.5],
            "mean_abs_lfc": [1.0, 0.2],
            "celltypes_neu": ["Fibroblast_S1", "Macrophage"],
            "is_anchor": [False, True],
            "is_ifng_target": [False, False],
            "mean_lfc": [1.0, 0.2],
            "min_padj": [0.01, 0.05],
        }
    )


def _step2():
    return {
        "node_labels": ["Fibroblast_S1", "Macrophage"],
        "betweenness": {"Fibroblast_S1": 0.7, "Macrophage": 0.2},
        "flow_edges": [{"source": "A", "target": "B"}],
        "metrics": {"graph_sparsity": 0.5, "hsic_independence": 0.8, "known_axis_recall": 0.4, "mean_bootstrap_freq": 0.3},
    }


def test_score_candidates_returns_ranked_frame():
    cluster_expr = pd.DataFrame({"A": [1.0, 2.0]}, index=["Fibroblast_S1", "Macrophage"])
    ranked = score_candidates(_candidate_pool(), _step2(), _step2(), {"A": {"n_ranked": 3}}, {}, cluster_expr)
    assert list(ranked["rank"]) == [1, 2]
    assert "final_score" in ranked.columns
    assert ranked.iloc[0]["final_score"] >= ranked.iloc[1]["final_score"]


def test_retain_hubs_and_combos_keeps_anchor_and_combo_rows():
    ranking = _candidate_pool()
    ranking["rank"] = [1, 2]
    step3 = {"A": {"ranked_targets": pd.DataFrame([{"ligand": "A", "receptor": "B", "target_priority_score": 0.9}])}}
    hubs, combos = retain_hubs_and_combos(ranking, step3)
    assert "MFAP2" in set(hubs["gene"])
    assert len(combos) == 1


def test_compare_modes_returns_geometry_and_step2_summary():
    geom = {"metrics": {"separation": 2.0}}
    comp = compare_modes(geom, geom, _step2(), _step2(), {}, {}, pd.DataFrame({"gene": ["A"], "final_score": [1.0]}))
    assert comp["geometry"]["hyp_separation"] == 2.0
    assert "hyp_graph_sparsity" in comp["step2"]
