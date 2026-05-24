"""Tests for roundtrip update pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config import HyperSCAConfig
from src.pipeline.roundtrip_update import RoundtripUpdatePipeline
from src.pipeline.step4_dynamic_intervention import DynamicInterventionPipeline


def _prepare_minimal_step2_step3(tmp_path: Path) -> tuple[Path, Path, Path]:
    step2 = tmp_path / "step2"
    step3 = tmp_path / "step3"
    step4 = tmp_path / "step4"
    step2.mkdir()
    step3.mkdir()
    step4.mkdir()
    np.save(step2 / "cluster_adj.npy", np.array([[0, 1], [1, 0]], dtype=float))
    np.save(step2 / "cluster_expr.npy", np.array([[2.0, 1.0], [1.0, 0.5]], dtype=float))
    (step2 / "node_info.json").write_text(
        json.dumps({"node_labels": ["CAF", "TAM"], "type_mapping": {"CAF": "CAF", "TAM": "TAM"}}),
        encoding="utf-8",
    )
    pd.DataFrame([[2.0, 1.0], [1.0, 0.5]], index=["CAF", "TAM"], columns=["GENE_A", "GENE_B"]).to_csv(
        step2 / "cluster_expr_df.csv"
    )
    pd.DataFrame({"ligand": ["GENE_A"], "receptor": ["REC_A"], "target_priority_score": [0.4]}).to_csv(
        step3 / "interaction_targets_GENE_A.csv", index=False
    )
    return step2, step3, step4


def test_roundtrip_update_pipeline(tmp_path: Path):
    step2, step3, step4 = _prepare_minimal_step2_step3(tmp_path)
    exp = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2"],
            "timepoint": [24, 24, 48],
            "dose": [1.0, 3.0, 1.0],
            "gene": ["GENE_A", "GENE_B", "GENE_A"],
            "effect_size": [0.4, 0.6, 0.3],
        }
    )
    exp_path = tmp_path / "experiment_roundtrip.csv"
    exp.to_csv(exp_path, index=False)

    cfg = HyperSCAConfig(
        step4_input_step2_dir=str(step2),
        step4_input_step3_dir=str(step3),
        step4_output_dir=str(step4),
        roundtrip_experiment_file=str(exp_path),
    )
    # baseline step4 first
    DynamicInterventionPipeline(cfg).run()
    report = RoundtripUpdatePipeline(cfg).run()
    assert "calibrated_params" in report
    assert (step4 / "roundtrip_update_report.json").exists()
