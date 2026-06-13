from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_grammar.config import BehaviorGrammarConfig, BehaviorGrammarPaths
from src.behavior_grammar.pipeline import BehaviorGrammarPipeline
from src.behavior_grammar.rule_builder import build_rules_from_discovery


def _write_synthetic_discovery_run(root: Path) -> Path:
    run_dir = root / "results" / "discovery" / "target_discovery" / "synthetic"
    (run_dir / "scoring").mkdir(parents=True, exist_ok=True)
    (run_dir / "causal" / "hyperbolic").mkdir(parents=True, exist_ok=True)
    (run_dir / "niche").mkdir(parents=True, exist_ok=True)
    (run_dir / "expression").mkdir(parents=True, exist_ok=True)

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "synthetic",
                "artifacts": [
                    "scoring/candidate_scores.csv",
                    "causal/hyperbolic/causal_edges.csv",
                    "niche/target_niche_mapping.csv",
                    "expression/cluster_expression.csv",
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "gene": ["TGFB1", "CXCL9", "PDCD1"],
            "cell_type": ["CAF", "TAM", "CD8_T"],
            "score": [0.91, 0.82, 0.77],
            "mean_expression": [2.0, 1.4, 0.9],
        }
    ).to_csv(run_dir / "scoring" / "candidate_scores.csv", index=False)
    pd.DataFrame(
        {
            "source": ["CAF", "TAM", "CD8_T"],
            "target": ["Tumor", "CD8_T", "Tumor"],
            "weight": [0.8, 0.7, 0.6],
            "ligand": ["TGFB1", "CXCL9", "PDCD1"],
            "receptor": ["TGFBR1", "CXCR3", "PDCD1"],
        }
    ).to_csv(run_dir / "causal" / "hyperbolic" / "causal_edges.csv", index=False)
    pd.DataFrame(
        {
            "target": ["TGFB1", "CXCL9", "PDCD1"],
            "niche": ["stromal_invasion", "inflamed_edge", "immune_checkpoint"],
        }
    ).to_csv(run_dir / "niche" / "target_niche_mapping.csv", index=False)
    pd.DataFrame(
        {
            "cell_type": ["CAF", "TAM", "CD8_T", "Tumor"],
            "TGFB1": [2.0, 0.2, 0.1, 0.0],
            "CXCL9": [0.1, 1.4, 0.3, 0.0],
            "PDCD1": [0.0, 0.1, 0.9, 0.2],
        }
    ).to_csv(run_dir / "expression" / "cluster_expression.csv", index=False)
    return run_dir / "manifest.json"


def test_build_rules_from_discovery_keeps_rules_data_driven(tmp_path):
    manifest = _write_synthetic_discovery_run(tmp_path)

    ruleset = build_rules_from_discovery(manifest, max_rules=3)

    assert ruleset.run_id == "synthetic"
    assert [rule.signal for rule in ruleset.rules] == ["TGFB1", "CXCL9", "PDCD1"]
    assert {rule.cell_type for rule in ruleset.rules} == {"CAF", "TAM", "CD8_T"}
    assert all("anchor" not in ref.lower() for rule in ruleset.rules for ref in rule.evidence_refs)


def test_behavior_grammar_pipeline_writes_run_artifacts(tmp_path):
    manifest = _write_synthetic_discovery_run(tmp_path)
    paths = BehaviorGrammarPaths.default(
        root=tmp_path,
        discovery_manifest=manifest,
        output_base=tmp_path / "results" / "behavior_grammar",
    )
    config = BehaviorGrammarConfig(paths=paths, run_id="stage5", max_rules=3, time_steps=5)

    outputs = BehaviorGrammarPipeline(config).run()

    run_dir = tmp_path / "results" / "behavior_grammar" / "stage5"
    assert outputs["run_dir"] == run_dir
    assert (run_dir / "rules" / "rules.json").exists()
    assert (run_dir / "rules" / "rules.md").exists()
    assert (run_dir / "simulation" / "simulation_summary.json").exists()
    assert (run_dir / "simulation" / "simulation_report.md").exists()
    assert (run_dir / "simulation" / "qoi_sensitivity.csv").exists()
    assert (run_dir / "figures" / "population_trajectories.png").exists()

    summary = json.loads((run_dir / "simulation" / "simulation_summary.json").read_text(encoding="utf-8"))
    assert summary["n_rules"] == 3
    assert summary["n_cell_types"] >= 3
    assert np.isfinite(summary["final_total_population"])

    report = (run_dir / "simulation" / "simulation_report.md").read_text(encoding="utf-8")
    assert "Final total population" in report
    assert "Top QoI sensitivity" in report
