"""Tests for Phase D0 data standardization validators and loaders."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loaders import load_project_manifest
from src.data.validators import (
    validate_experiment_roundtrip_fields,
    validate_multisource_min_fields,
    validate_onboarding_tree,
)


def test_validate_multisource_min_fields_ok():
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "mmr_group": ["MSS"],
            "celltype": ["CAF"],
            "spot_or_cell_id": ["c1"],
            "x": [1.0],
            "y": [2.0],
        }
    )
    assert validate_multisource_min_fields(df) == []


def test_validate_experiment_roundtrip_fields_missing():
    df = pd.DataFrame({"sample_id": ["s1"], "gene": ["POSTN"]})
    issues = validate_experiment_roundtrip_fields(df)
    assert issues
    assert "缺失必需列" in issues[0]


def test_validate_onboarding_tree(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "scRNA" / "scCRC_Neu").mkdir(parents=True)
    (data_root / "scRNA" / "scCRC_IFNG").mkdir(parents=True)
    (data_root / "scRNA" / "scCRC_ICB").mkdir(parents=True)
    (data_root / "ST" / "ST_CRC_MSS").mkdir(parents=True)
    (data_root / "metadata").mkdir(parents=True)
    issues = validate_onboarding_tree(data_root)
    assert issues == []


def test_load_project_manifest(tmp_path: Path):
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    payload = {
        "sources": [
            {
                "name": "scCRC_Neu",
                "source_path": r"G:\scCRC_Neu",
                "modality": "scRNA",
                "standardized_dir": "data/scRNA/scCRC_Neu",
            }
        ]
    }
    (meta / "project_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    out = load_project_manifest(meta)
    assert len(out) == 1
    assert out[0].name == "scCRC_Neu"
