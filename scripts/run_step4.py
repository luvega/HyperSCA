#!/usr/bin/env python
"""Stage 4 dynamic intervention and optional roundtrip update."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.config import HyperSCAConfig
from src.pipeline.roundtrip_update import RoundtripUpdatePipeline
from src.pipeline.step4_dynamic_intervention import DynamicInterventionPipeline


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    arr = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(x) for x in arr] if arr else None


def _parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    arr = [x.strip() for x in raw.split(",") if x.strip()]
    return arr or None


def main() -> int:
    parser = argparse.ArgumentParser(description="HyperSCA Stage 4: Dynamic intervention")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--step2-dir", type=str, default=None)
    parser.add_argument("--step3-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hub-genes", type=str, default=None, help="Comma separated")
    parser.add_argument("--dose-grid", type=str, default=None, help="Comma separated float")
    parser.add_argument("--time-grid", type=str, default=None, help="Comma separated float")
    parser.add_argument("--combo-max-size", type=int, default=None)
    parser.add_argument("--experiment-file", type=str, default=None)
    parser.add_argument("--with-roundtrip", action="store_true")
    args = parser.parse_args()

    cfg = HyperSCAConfig.from_yaml(args.config) if args.config else HyperSCAConfig()
    overrides = {
        "step4_input_step2_dir": args.step2_dir,
        "step4_input_step3_dir": args.step3_dir,
        "step4_output_dir": args.output_dir,
        "step4_combo_max_size": args.combo_max_size,
        "roundtrip_experiment_file": args.experiment_file,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    hub = _parse_str_list(args.hub_genes)
    if hub is not None:
        cfg.step4_hub_genes = hub
    dg = _parse_float_list(args.dose_grid)
    if dg is not None:
        cfg.step4_dose_grid = dg
    tg = _parse_float_list(args.time_grid)
    if tg is not None:
        cfg.step4_time_grid = tg

    print("HyperSCA Stage 4: Dynamic Intervention")
    print("=" * 60)
    print(f"  step2_dir: {cfg.step4_input_step2_dir}")
    print(f"  step3_dir: {cfg.step4_input_step3_dir}")
    print(f"  output:    {cfg.step4_output_dir}")
    print(f"  hubs:      {cfg.step4_hub_genes}")
    print("=" * 60)

    summary = DynamicInterventionPipeline(cfg).run()
    print("[DONE] Step4 summary:", summary)

    if args.with_roundtrip:
        report = RoundtripUpdatePipeline(cfg).run()
        print("[DONE] Roundtrip report written:", report.get("elapsed_seconds"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
