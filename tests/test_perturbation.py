"""阶段3扰动模块单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_rank_counterfactual_interaction_targets_with_prior():
    from src.perturbation.target_ranking import PriorKnowledge, rank_counterfactual_interaction_targets

    flow_edges = [
        {
            "source_layer": 0,
            "target_layer": 1,
            "source": "POSTN",
            "target": "ITGAV",
            "weight": 0.9,
            "causal_edge": "CAF→TAM",
            "pathway": "Integrin-FAK",
        },
        {
            "source_layer": 0,
            "target_layer": 1,
            "source": "IL10",
            "target": "IL10RA",
            "weight": 0.8,
            "causal_edge": "TAM→CD8T",
            "pathway": "IL10-STAT3",
        },
    ]
    genes = ["POSTN", "ITGAV", "IL10", "IL10RA"]
    obs = pd.DataFrame([[1.0, 1.2, 1.1, 0.9]], columns=genes, index=["node1"])
    cf = pd.DataFrame([[0.2, 0.3, 1.0, 0.85]], columns=genes, index=["node1"])

    prior = PriorKnowledge(
        lr_pairs={("POSTN", "ITGAV")},
        tf_targets=set(),
        sources={("POSTN", "ITGAV"): {"liana"}},
    )
    ranked = rank_counterfactual_interaction_targets(
        flow_edges=flow_edges,
        observed_expression=obs,
        counterfactual_expression=cf,
        prior=prior,
        node_to_type={"node1": "CAF"},
        min_abs_delta=0.01,
        top_k=None,
    )

    assert not ranked.empty
    assert ranked.iloc[0]["ligand"] == "POSTN"
    assert bool(ranked.iloc[0]["prior_hit"]) is True
    assert ranked.iloc[0]["target_priority_score"] > 0


def test_rank_counterfactual_interaction_targets_grouped_delta():
    from src.perturbation.target_ranking import PriorKnowledge, rank_counterfactual_interaction_targets

    flow_edges = [
        {
            "source_layer": 0,
            "target_layer": 1,
            "source": "POSTN",
            "target": "ITGAV",
            "weight": 1.0,
            "causal_edge": "CAF->TAM",
            "pathway": "Integrin-FAK",
        }
    ]
    genes = ["POSTN", "ITGAV"]
    obs = pd.DataFrame(
        [[1.0, 1.0], [3.0, 3.0]],
        columns=genes,
        index=["caf_node", "tam_node"],
    )
    cf = pd.DataFrame(
        [[0.1, 1.0], [3.0, 0.5]],
        columns=genes,
        index=["caf_node", "tam_node"],
    )

    ranked = rank_counterfactual_interaction_targets(
        flow_edges=flow_edges,
        observed_expression=obs,
        counterfactual_expression=cf,
        prior=PriorKnowledge(lr_pairs=set(), tf_targets=set(), sources={}),
        node_to_type={"caf_node": "CAF", "tam_node": "TAM"},
        min_abs_delta=0.01,
        top_k=None,
    )
    assert not ranked.empty
    row = ranked.iloc[0]
    assert row["source_type"] == "CAF"
    assert row["target_type"] == "TAM"
    # 分组下: ligand 在 CAF 变化 -0.9, receptor 在 TAM 变化 -2.5
    assert abs(float(row["combined_abs_delta"]) - 3.4) < 1e-6


def test_causal_diffusion_cf_minimal_intervention():
    from src.perturbation.diffusion_cf import CausalDiffusionCF, DiffusionConfig

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (64, 4)).astype(np.float32)
    genes = ["A", "B", "C", "D"]
    # A -> B -> C, D isolated
    mask = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )

    model = CausalDiffusionCF(
        input_dim=4,
        causal_mask=mask,
        device="cpu",
        config=DiffusionConfig(n_steps=5, hidden_dim=32, train_epochs=1, batch_size=32),
    )
    out = model.fit(x)
    assert "train_loss" in out

    x_cf = model.generate_counterfactual(
        x_observed=x,
        gene_names=genes,
        intervention={"A": 0.0},
        seed=123,
    )
    assert x_cf.shape == x.shape
    # 干预基因被强制设置
    assert np.allclose(x_cf[:, 0], 0.0, atol=1e-5)
    # D 非后继，应该冻结为观测值
    assert np.allclose(x_cf[:, 3], x[:, 3], atol=1e-4)


def test_step3_target_metrics_contains_dashboard_metrics():
    from src.pipeline.step3_perturbation import PerturbationPipeline

    obs = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.2, 2.1, 2.9]],
        columns=["POSTN", "ITGAV", "GENE_X"],
        index=["n1", "n2"],
    )
    cf = pd.DataFrame(
        [[0.6, 1.5, 3.1], [0.7, 1.6, 3.0]],
        columns=["POSTN", "ITGAV", "GENE_X"],
        index=["n1", "n2"],
    )
    ranked = pd.DataFrame(
        [
            {
                "ligand": "POSTN",
                "receptor": "ITGAV",
                "prior_hit": True,
                "target_priority_score": 0.8,
                "pathway": "Integrin-FAK",
            }
        ]
    )
    m = PerturbationPipeline._compute_target_metrics("POSTN", ranked, obs, cf)
    assert "dashboard_metrics" in m
    for k in ("r2_mean", "pcc_median", "mse", "marker_direction_accuracy"):
        assert k in m["dashboard_metrics"]

