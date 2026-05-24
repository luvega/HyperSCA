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
    df = pd.DataFrame({"sample_id": ["s1"], "gene": ["GENE_A"]})
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


def test_build_canonical_schema_does_not_seed_manual_targets(monkeypatch, tmp_path: Path):
    import scripts.build_canonical_schema as schema

    for name in ("neu", "ifng", "icb", "st"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(schema, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(schema, "NEU_DIR", tmp_path / "neu")
    monkeypatch.setattr(schema, "IFNG_DIR", tmp_path / "ifng")
    monkeypatch.setattr(schema, "ICB_DIR", tmp_path / "icb")
    monkeypatch.setattr(schema, "ST_DIR", tmp_path / "st")
    schema.OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_table = schema.build_feature_table(include_icb=False)

    assert feature_table.empty
