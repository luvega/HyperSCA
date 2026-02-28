"""Roundtrip update pipeline: import experiments -> recalibrate -> rerun Step4."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.experiment_roundtrip import (
    calibrate_pkpd_params,
    load_experiment_results,
    summarize_experiment_effects,
)
from src.pipeline.config import HyperSCAConfig
from src.pipeline.step4_dynamic_intervention import DynamicInterventionPipeline


class RoundtripUpdatePipeline:
    """Run wet-dry roundtrip update and produce comparison report."""

    def __init__(self, config: HyperSCAConfig):
        self.config = config
        self.output_dir = Path(config.step4_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        t0 = time.time()
        baseline_path = self.output_dir / "step4_summary.json"
        baseline = {}
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        exp_df = load_experiment_results(self.config.roundtrip_experiment_file)
        exp_summary = summarize_experiment_effects(exp_df)
        calibrated = calibrate_pkpd_params(
            exp_summary,
            default_ec50=self.config.step4_pd_ec50,
            default_emax=self.config.step4_pd_emax,
        )

        # Update config in-memory
        self.config.step4_pd_ec50 = calibrated["ec50"]
        self.config.step4_pd_emax = calibrated["emax"]

        rerun = DynamicInterventionPipeline(self.config).run()

        report = {
            "baseline_summary": baseline,
            "experiment_rows": int(len(exp_df)),
            "experiment_genes": sorted(exp_df["gene"].astype(str).unique().tolist()),
            "calibrated_params": calibrated,
            "rerun_summary": rerun,
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        (self.output_dir / "roundtrip_update_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report
