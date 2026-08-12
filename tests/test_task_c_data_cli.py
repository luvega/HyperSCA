import json
import subprocess
import sys


def test_export_command_reports_pinned_source_without_downloading(tmp_path):
    data_dir = tmp_path / "raw"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_causalbench_data.py",
            "--data-dir",
            str(data_dir),
            "--describe-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    description = json.loads(result.stdout)
    assert description["repository"] == "https://github.com/causalbench/causalbench.git"
    assert description["commit"] == "1a2143cffdc85f835b41ce8d52034be1bf903e71"
    assert description["datasets"] == ["dataset_k562.npz", "dataset_rpe1.npz"]
    assert description["references"] == [
        "reference_k562_pooled.csv",
        "reference_k562_chipseq.csv",
        "reference_rpe1_pooled.csv",
        "reference_rpe1_chipseq.csv",
    ]
