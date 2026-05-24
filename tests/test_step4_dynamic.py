"""Tests for Step4 dynamic intervention pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.config import HyperSCAConfig
from src.pipeline.step4_dynamic_intervention import DynamicInterventionPipeline


def test_step4_dynamic_pipeline_runs(tmp_path: Path):
    step2 = tmp_path / "step2"
    step3 = tmp_path / "step3"
    step2.mkdir()
    step3.mkdir()

    cluster_adj = np.array([[0, 1], [1, 0]], dtype=float)
    cluster_expr = np.array([[2.0, 1.0], [1.5, 0.5]], dtype=float)
    np.save(step2 / "cluster_adj.npy", cluster_adj)
    np.save(step2 / "cluster_expr.npy", cluster_expr)
    (step2 / "node_info.json").write_text(
        json.dumps({"node_labels": ["CAF", "TAM"], "type_mapping": {"CAF": "CAF", "TAM": "TAM"}}),
        encoding="utf-8",
    )
    pd.DataFrame(cluster_expr, index=["CAF", "TAM"], columns=["GENE_A", "GENE_B"]).to_csv(
        step2 / "cluster_expr_df.csv"
    )
    # Step3 presence marker
    pd.DataFrame({"ligand": ["GENE_A"], "receptor": ["REC_A"], "target_priority_score": [0.5]}).to_csv(
        step3 / "interaction_targets_GENE_A.csv", index=False
    )

    cfg = HyperSCAConfig(
        step4_input_step2_dir=str(step2),
        step4_input_step3_dir=str(step3),
        step4_output_dir=str(tmp_path / "step4"),
        step4_hub_genes=["GENE_A", "GENE_B"],
        step4_dose_grid=[0.1, 1.0],
        step4_time_grid=[0.0, 12.0, 24.0],
    )
    summary = DynamicInterventionPipeline(cfg).run()
    assert summary["n_targets"] >= 2
    assert (tmp_path / "step4" / "step4_summary.json").exists()
    assert (tmp_path / "step4" / "combination_ranking.csv").exists()
    combo_df = pd.read_csv(tmp_path / "step4" / "combination_ranking.csv")
    assert set(combo_df["model_type"]) == {"bliss_proxy"}
    assert combo_df["calibrated_by_experiment"].eq(False).all()


def test_step4_requires_cluster_expression_gene_names(tmp_path: Path):
    """Step4 should not silently replace missing gene names with GENE_0 columns."""
    step2 = tmp_path / "step2"
    step3 = tmp_path / "step3"
    step2.mkdir()
    step3.mkdir()

    np.save(step2 / "cluster_adj.npy", np.array([[0, 1], [1, 0]], dtype=float))
    np.save(step2 / "cluster_expr.npy", np.array([[2.0, 1.0], [1.5, 0.5]], dtype=float))
    (step2 / "node_info.json").write_text(
        json.dumps({"node_labels": ["CAF", "TAM"]}),
        encoding="utf-8",
    )

    cfg = HyperSCAConfig(
        step4_input_step2_dir=str(step2),
        step4_input_step3_dir=str(step3),
        step4_output_dir=str(tmp_path / "step4"),
        step4_hub_genes=["GENE_A"],
    )

    with pytest.raises(FileNotFoundError, match="cluster_expr_df.csv"):
        DynamicInterventionPipeline(cfg).load_inputs()


def test_step4_defaults_targets_from_expression_without_hubs(tmp_path: Path):
    step2 = tmp_path / "step2"
    step3 = tmp_path / "step3"
    step2.mkdir()
    step3.mkdir()

    cluster_adj = np.array([[0, 1], [1, 0]], dtype=float)
    cluster_expr = np.array([[5.0, 1.0], [4.0, 2.0]], dtype=float)
    np.save(step2 / "cluster_adj.npy", cluster_adj)
    np.save(step2 / "cluster_expr.npy", cluster_expr)
    (step2 / "node_info.json").write_text(
        json.dumps({"node_labels": ["N1", "N2"], "type_mapping": {"N1": "Type1", "N2": "Type2"}}),
        encoding="utf-8",
    )
    pd.DataFrame(cluster_expr, index=["N1", "N2"], columns=["GENE_A", "GENE_B"]).to_csv(
        step2 / "cluster_expr_df.csv"
    )

    cfg = HyperSCAConfig(
        step4_input_step2_dir=str(step2),
        step4_input_step3_dir=str(step3),
        step4_output_dir=str(tmp_path / "step4"),
        step4_dose_grid=[1.0],
        step4_time_grid=[0.0, 12.0],
    )
    summary = DynamicInterventionPipeline(cfg).run()

    assert HyperSCAConfig().step4_hub_genes == []
    assert summary["targets"][0] == "GENE_A"
    assert summary["n_targets"] >= 1
