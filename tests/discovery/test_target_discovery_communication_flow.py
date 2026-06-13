from __future__ import annotations

import numpy as np
import pandas as pd


def _expr():
    return pd.DataFrame(
        {
            "POSTN": [3.0, 0.1],
            "ITGAV": [0.1, 2.0],
            "SRC": [0.2, 2.5],
            "CD163": [0.1, 4.0],
        },
        index=["Fibroblast_S1", "Macrophage"],
    )


def test_distance_constrained_lr_flow_reports_forward_direction():
    from src.discovery.target_discovery.communication_flow import build_communication_flow

    causal_adj = np.array([[0.0, 1.0], [0.0, 0.0]])
    spatial_adj = np.array([[0.0, 0.8], [0.8, 0.0]])
    dist = np.array([[0.0, 0.2], [0.2, 0.0]])

    result = build_communication_flow(
        mode="hyperbolic",
        cluster_expr=_expr(),
        spatial_adjacency=spatial_adj,
        geometry_result={"dist_matrix": dist},
        causal_result={
            "causal_graph": type("Graph", (), {"adjacency": causal_adj})(),
            "type_mapping": {"Fibroblast_S1": "CAF", "Macrophage": "TAM"},
            "node_labels": ["Fibroblast_S1", "Macrophage"],
        },
    )

    edges = result["lr_flow_edges"]
    postn = edges[(edges["ligand"] == "POSTN") & (edges["receptor"] == "ITGAV")].iloc[0]
    assert postn["direction_status"] == "forward"
    assert postn["flow_score"] > 0
    assert result["direction_consistency"]["overall"]["direction_consistency_rate"] == 1.0


def test_direction_consistency_excludes_unresolved_edges_from_denominator():
    from src.discovery.target_discovery.communication_flow import summarize_direction_consistency

    edges = pd.DataFrame(
        [
            {"flow_score": 1.0, "direction_status": "forward", "pathway": "p1"},
            {"flow_score": 1.0, "direction_status": "reverse", "pathway": "p1"},
            {"flow_score": 100.0, "direction_status": "unresolved", "pathway": "p1"},
        ]
    )

    summary = summarize_direction_consistency(edges)

    assert summary["overall"]["resolved_edges"] == 2
    assert summary["overall"]["unresolved_count"] == 1
    assert summary["overall"]["direction_consistency_rate"] == 0.5
