"""Tests for Step4 dynamic intervention pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    pd.DataFrame(cluster_expr, index=["CAF", "TAM"], columns=["POSTN", "INHBA"]).to_csv(
        step2 / "cluster_expr_df.csv"
    )
    # Step3 presence marker
    pd.DataFrame({"ligand": ["POSTN"], "receptor": ["ITGAV"], "target_priority_score": [0.5]}).to_csv(
        step3 / "interaction_targets_POSTN.csv", index=False
    )

    cfg = HyperSCAConfig(
        step4_input_step2_dir=str(step2),
        step4_input_step3_dir=str(step3),
        step4_output_dir=str(tmp_path / "step4"),
        step4_hub_genes=["POSTN", "INHBA"],
        step4_dose_grid=[0.1, 1.0],
        step4_time_grid=[0.0, 12.0, 24.0],
    )
    summary = DynamicInterventionPipeline(cfg).run()
    assert summary["n_targets"] >= 2
    assert (tmp_path / "step4" / "step4_summary.json").exists()
    assert (tmp_path / "step4" / "combination_ranking.csv").exists()
