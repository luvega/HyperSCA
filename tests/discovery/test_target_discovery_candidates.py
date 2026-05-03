from __future__ import annotations

import pandas as pd

from src.discovery.target_discovery.candidates import aggregate_candidate_pool
from src.discovery.target_discovery.expression import assemble_cluster_expression
from src.discovery.target_discovery.spatial import build_spatial_adjacency_from_tables


def test_aggregate_candidate_pool_scores_multisource_genes():
    neu = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "celltype_neu": ["Fibroblast_S1", "Macrophage"],
            "lfc_neu": [1.0, -0.7],
            "padj_neu": [0.01, 0.02],
        }
    )
    icb = pd.DataFrame(
        {
            "gene": ["A"],
            "celltype_icb": ["Fibro"],
            "lfc_icb": [0.8],
            "padj_icb": [0.03],
            "source_file": ["unit.csv"],
        }
    )
    ifng = pd.DataFrame(
        {
            "gene": ["C"],
            "celltype_ifng": ["IFNG_focus"],
            "lfc_ifng": [0.5],
            "mmr_group": ["MSS"],
        }
    )

    out = aggregate_candidate_pool(neu, icb, ifng)

    assert out.iloc[0]["gene"] == "A"
    assert int(out[out["gene"] == "A"].iloc[0]["cross_queue_count"]) == 2
    assert "init_score" in out.columns


def test_assemble_cluster_expression_log1p_means():
    tables = {
        "Fibroblast_S1": pd.DataFrame({"s1": [0.0, 3.0], "s2": [2.0, 5.0]}, index=["A", "B"]),
        "Macrophage": pd.DataFrame({"s1": [1.0, 1.0], "s2": [1.0, 3.0]}, index=["A", "B"]),
    }

    expr, labels = assemble_cluster_expression(tables)

    assert labels == ["Fibroblast_S1", "Macrophage"]
    assert expr.shape == (2, 2)
    assert float(expr.loc["Fibroblast_S1", "A"]) > 0


def test_build_spatial_adjacency_from_tables_returns_square_matrix():
    table = pd.DataFrame(
        {
            "Fibro_ADAMDEC1": [1.0, 0.0, 1.0],
            "Mac_M1": [0.0, 1.0, 0.5],
        }
    )

    out = build_spatial_adjacency_from_tables([table], ["Fibroblast_S1", "Macrophage"])

    assert out.shape == (2, 2)
    assert out[0, 0] == 0.0
