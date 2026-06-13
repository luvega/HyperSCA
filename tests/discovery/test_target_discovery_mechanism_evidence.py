from __future__ import annotations

import pandas as pd


def _flow_edges():
    return [
        {
            "source_layer": 0,
            "target_layer": 1,
            "source": "POSTN",
            "target": "ITGAV",
            "pathway": "Integrin-FAK",
            "causal_edge": "CAF->TAM",
            "weight": 0.9,
            "ligand_expr": 2.0,
            "receptor_expr": 3.0,
        },
        {
            "source_layer": 1,
            "target_layer": 2,
            "source": "ITGAV",
            "target": "SRC",
            "pathway": "Integrin-FAK",
            "causal_edge": "CAF->TAM",
            "weight": 0.7,
        },
        {
            "source_layer": 2,
            "target_layer": 3,
            "source": "SRC",
            "target": "CD163",
            "pathway": "Integrin-FAK",
            "causal_edge": "CAF->TAM",
            "weight": 0.5,
            "target_expr": 4.0,
        },
    ]


def test_mechanism_evidence_builds_complete_lr_tf_target_chain():
    from src.discovery.target_discovery.mechanism_evidence import build_mechanism_evidence

    ranking = pd.DataFrame(
        {
            "gene": ["CD163", "UNRELATED"],
            "rank": [1, 2],
            "final_score": [0.8, 0.2],
            "s_niche": [0.5, 0.1],
        }
    )

    matrix, summary = build_mechanism_evidence(
        ranking=ranking,
        causal_result={"flow_edges": _flow_edges(), "flow_summary": {"relaxed_mode": False}},
        perturbation_results={"CD163": {"spatial_quality": {"gradient_decay_r2": 0.6, "propagation_depth": 2}}},
        top_n=20,
    )

    assert len(matrix) == 1
    row = matrix.iloc[0]
    assert row["target_gene"] == "CD163"
    assert row["ligand"] == "POSTN"
    assert row["receptor"] == "ITGAV"
    assert row["tf"] == "SRC"
    assert row["downstream_target"] == "CD163"
    assert row["s_mechanism"] > 0
    assert summary["n_targets_with_mechanism"] == 1


def test_mechanism_evidence_preserves_multiple_chains_per_pathway():
    from src.discovery.target_discovery.mechanism_evidence import build_mechanism_evidence

    flow_edges = _flow_edges() + [
        {
            "source_layer": 0,
            "target_layer": 1,
            "source": "POSTN",
            "target": "ITGB5",
            "pathway": "Integrin-FAK",
            "causal_edge": "CAF->TAM",
            "weight": 0.8,
            "ligand_expr": 2.0,
            "receptor_expr": 2.5,
        },
        {
            "source_layer": 1,
            "target_layer": 2,
            "source": "ITGB5",
            "target": "FAK",
            "pathway": "Integrin-FAK",
            "causal_edge": "CAF->TAM",
            "weight": 0.6,
        },
        {
            "source_layer": 2,
            "target_layer": 3,
            "source": "FAK",
            "target": "MRC1",
            "pathway": "Integrin-FAK",
            "causal_edge": "CAF->TAM",
            "weight": 0.4,
            "target_expr": 3.5,
        },
    ]
    ranking = pd.DataFrame(
        {
            "gene": ["CD163", "MRC1"],
            "rank": [1, 2],
            "final_score": [0.8, 0.7],
        }
    )

    matrix, summary = build_mechanism_evidence(
        ranking=ranking,
        causal_result={"flow_edges": flow_edges, "flow_summary": {"relaxed_mode": False}},
        top_n=20,
    )

    assert set(matrix["target_gene"]) == {"CD163", "MRC1"}
    assert set(matrix["receptor"]) == {"ITGAV", "ITGB5"}
    assert summary["n_targets_with_mechanism"] == 2


def test_mechanism_scores_are_append_only_and_do_not_change_final_score():
    from src.discovery.target_discovery.mechanism_evidence import append_mechanism_scores

    ranking = pd.DataFrame({"gene": ["CD163"], "final_score": [0.8], "rank": [1]})
    matrix = pd.DataFrame({"target_gene": ["CD163"], "s_mechanism": [0.75]})

    out = append_mechanism_scores(ranking, matrix)

    assert out.loc[0, "final_score"] == 0.8
    assert out.loc[0, "rank"] == 1
    assert out.loc[0, "s_mechanism"] == 0.75
