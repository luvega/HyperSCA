"""Demo discovery artifacts for visible behavior grammar runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_demo_discovery_run(root: str | Path, *, run_id: str = "demo_behavior_grammar") -> Path:
    """Create a small CRC TME discovery manifest suitable for Stage5 demos."""
    root = Path(root)
    run_dir = root / "results" / "discovery" / "target_discovery" / run_id
    (run_dir / "scoring").mkdir(parents=True, exist_ok=True)
    (run_dir / "causal" / "hyperbolic").mkdir(parents=True, exist_ok=True)
    (run_dir / "niche").mkdir(parents=True, exist_ok=True)
    (run_dir / "expression").mkdir(parents=True, exist_ok=True)

    artifacts = [
        "scoring/candidate_scores.csv",
        "causal/hyperbolic/causal_edges.csv",
        "niche/target_niche_mapping.csv",
        "expression/cluster_expression.csv",
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "gene": ["TGFB1", "CXCL9", "PDCD1", "EGF", "IFNG"],
            "cell_type": ["CAF", "TAM", "CD8_T", "TAM", "CD8_T"],
            "score": [0.91, 0.82, 0.77, 0.68, 0.63],
            "mean_expression": [2.0, 1.4, 0.9, 1.1, 0.8],
        }
    ).to_csv(run_dir / "scoring" / "candidate_scores.csv", index=False)
    pd.DataFrame(
        {
            "source": ["CAF", "TAM", "CD8_T", "TAM", "CD8_T"],
            "target": ["Tumor", "CD8_T", "Tumor", "Tumor", "Tumor"],
            "weight": [0.8, 0.7, 0.6, 0.58, 0.55],
            "ligand": ["TGFB1", "CXCL9", "PDCD1", "EGF", "IFNG"],
            "receptor": ["TGFBR1", "CXCR3", "PDCD1", "EGFR", "IFNGR1"],
        }
    ).to_csv(run_dir / "causal" / "hyperbolic" / "causal_edges.csv", index=False)
    pd.DataFrame(
        {
            "target": ["TGFB1", "CXCL9", "PDCD1", "EGF", "IFNG"],
            "niche": ["stromal_invasion", "inflamed_edge", "immune_checkpoint", "tam_motility", "cytotoxic_edge"],
        }
    ).to_csv(run_dir / "niche" / "target_niche_mapping.csv", index=False)
    pd.DataFrame(
        {
            "cell_type": ["CAF", "TAM", "CD8_T", "Tumor"],
            "TGFB1": [2.0, 0.2, 0.1, 0.0],
            "CXCL9": [0.1, 1.4, 0.3, 0.0],
            "PDCD1": [0.0, 0.1, 0.9, 0.2],
            "EGF": [0.1, 1.1, 0.1, 0.2],
            "IFNG": [0.0, 0.2, 0.8, 0.0],
        }
    ).to_csv(run_dir / "expression" / "cluster_expression.csv", index=False)
    return run_dir / "manifest.json"
