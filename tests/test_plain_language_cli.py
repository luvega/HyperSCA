from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script, required_phrases",
    [
        ("run_target_discovery.py", ["候选靶点", "直接证据", "分析记录清单"]),
        ("run_causal_stability_audit.py", ["重复抽样", "零效应对照"]),
        ("validate_benchmark_contract.py", ["预先固定的比较规则", "不会运行模型"]),
        ("run_task_c_mean_difference.py", ["干预数据", "简单对照方法"]),
        ("run_task_s_baseline.py", ["自身效应", "邻近细胞效应", "独立验证数据"]),
    ],
)
def test_help_explains_research_purpose(
    script: str,
    required_phrases: list[str],
) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for phrase in required_phrases:
        assert phrase in completed.stdout


def test_task_c_missing_input_gives_actionable_error(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_task_c_mean_difference.py"),
            "--input-npz",
            str(tmp_path / "missing.npz"),
            "--dataset-id",
            "missing",
            "--dataset-source",
            "test",
            "--context-id",
            "test",
            "--data-status",
            "synthetic_smoke",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "无法继续" in completed.stderr
    assert "请检查" in completed.stderr
