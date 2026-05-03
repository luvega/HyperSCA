from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.discovery.target_discovery.artifacts import ArtifactWriter, json_default


def test_json_default_serializes_common_science_objects():
    assert json_default(np.int64(3)) == 3
    assert json_default(np.float32(1.5)) == 1.5
    assert json_default(np.array([1, 2])) == [1, 2]
    assert json_default({"A", "B"}) in (["A", "B"], ["B", "A"])
    df_payload = json_default(pd.DataFrame({"x": [1]}))
    assert df_payload == [{"x": 1}]


def test_artifact_writer_creates_sections_and_manifest(tmp_path):
    writer = ArtifactWriter(tmp_path, run_id="unit")
    table_path = writer.write_table("example.csv", pd.DataFrame({"gene": ["A"]}), section="candidates")
    json_path = writer.write_json("metrics.json", {"value": np.float32(2.0)}, section="scoring")
    array_path = writer.write_array("adj.npy", np.eye(2), section="spatial")
    md_path = writer.write_markdown("report.md", "# Report\n", section="reports")
    writer.finalize()

    assert table_path.exists()
    assert json_path.exists()
    assert array_path.exists()
    assert md_path.exists()
    manifest = json.loads((tmp_path / "unit" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit"
    assert "candidates/example.csv" in manifest["artifacts"]


def test_artifact_writer_rejects_path_traversal(tmp_path):
    writer = ArtifactWriter(tmp_path, run_id="unit")
    with pytest.raises(ValueError, match="unsafe artifact path"):
        writer.write_markdown("escape.md", "bad", section="../outside")
    with pytest.raises(ValueError, match="unsafe artifact path"):
        writer.write_markdown("../escape.md", "bad", section="reports")
    assert not (tmp_path / "outside" / "escape.md").exists()


def test_write_json_replaces_non_finite_values_with_null(tmp_path):
    writer = ArtifactWriter(tmp_path, run_id="unit")
    path = writer.write_json(
        "metrics.json",
        {
            "nan": np.nan,
            "inf": np.inf,
            "nested": {"neg_inf": -np.inf},
            "array": np.array([1.0, np.nan]),
            "frame": pd.DataFrame({"x": [1.0, np.nan]}),
        },
        section="scoring",
    )
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["nan"] is None
    assert payload["inf"] is None
    assert payload["nested"]["neg_inf"] is None
    assert payload["array"] == [1.0, None]
    assert payload["frame"] == [{"x": 1.0}, {"x": None}]
