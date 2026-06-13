from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_demo_writes_visible_simulation_outputs(tmp_path):
    output_dir = tmp_path / "results" / "behavior_grammar"
    cmd = [
        sys.executable,
        "scripts/run_behavior_grammar_simulation.py",
        "--demo",
        "--output-dir",
        str(output_dir),
        "--run-id",
        "visible_demo",
        "--time-steps",
        "5",
    ]

    completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=True)

    run_dir = output_dir / "visible_demo"
    assert "Report:" in completed.stdout
    assert (run_dir / "rules" / "rules.md").exists()
    assert (run_dir / "simulation" / "simulation_report.md").exists()
    assert (run_dir / "simulation" / "population_trajectory.csv").exists()
    assert (run_dir / "figures" / "population_trajectories.png").exists()
