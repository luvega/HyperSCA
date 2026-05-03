"""Acceptance tests for ICB full-data reference integration.

Validates:
1. prepare_h5ad.py ICB full-ingest helpers are importable and correct
2. build_icb_reference.py is importable and has expected functions
3. loaders.py new reference functions work (with/without data/ref)
4. validators.py new validate_reference_tree works
5. Backward compatibility: existing APIs unchanged
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestPrepareH5adICBHelpers:
    """Test _build_icb_h5ad and _extract_icb_metadata_via_r importability."""

    def test_module_importable(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "prepare_h5ad", ROOT / "scripts" / "prepare_h5ad.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "_build_icb_h5ad")
        assert hasattr(mod, "_extract_icb_metadata_via_r")
        assert hasattr(mod, "_prepare_multisource")


class TestBuildICBReference:
    """Test build_icb_reference.py importability and key functions."""

    def test_module_importable(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "build_icb_reference", ROOT / "scripts" / "build_icb_reference.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "preprocess_for_reference")
        assert hasattr(mod, "train_scvi_reference")
        assert hasattr(mod, "train_scanvi_reference")
        assert hasattr(mod, "export_reference")
        assert hasattr(mod, "_resolve_label_key")

    def test_resolve_label_key(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "build_icb_reference", ROOT / "scripts" / "build_icb_reference.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import anndata
        adata = anndata.AnnData(
            X=np.random.rand(50, 10).astype(np.float32),
            obs=pd.DataFrame({
                "MajorCellType": np.random.choice(["A", "B", "C"], 50),
                "other": np.random.rand(50),
            }),
        )
        key = mod._resolve_label_key(adata, None)
        assert key == "MajorCellType"

        key2 = mod._resolve_label_key(adata, "MajorCellType")
        assert key2 == "MajorCellType"

        key3 = mod._resolve_label_key(adata, "nonexistent")
        assert key3 == "MajorCellType"


class TestReferenceLoaders:
    """Test new reference loader functions in loaders.py."""

    def test_reference_manifest_dataclass(self):
        from src.data.loaders import ReferenceManifest
        m = ReferenceManifest(
            reference_name="test", version="v1", model_type="scvi",
            model_dir="/tmp/m", mappings_dir="/tmp/map",
            reference_h5ad="/tmp/ref.h5ad", label_key="ct",
            n_cells=100, n_genes=50, hvg_only=True, created="2025-01-01",
        )
        assert m.reference_name == "test"
        assert m.n_cells == 100

    def test_load_reference_manifest_missing(self, tmp_path):
        from src.data.loaders import load_reference_manifest
        with pytest.raises(FileNotFoundError):
            load_reference_manifest(data_root=tmp_path)

    def test_load_reference_manifest_valid(self, tmp_path):
        from src.data.loaders import load_reference_manifest
        manifest_dir = tmp_path / "ref" / "manifest"
        manifest_dir.mkdir(parents=True)
        payload = {
            "reference_name": "icb_reference",
            "version": "v1",
            "model_type": "scanvi",
            "model_dir": str(tmp_path / "models"),
            "mappings_dir": str(tmp_path / "mappings"),
            "reference_h5ad": str(tmp_path / "ref.h5ad"),
            "label_key": "MajorCellType",
            "n_cells": 1000,
            "n_genes": 3000,
            "hvg_only": True,
            "created": "2025-01-01T00:00:00",
        }
        (manifest_dir / "reference_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
        m = load_reference_manifest(data_root=tmp_path)
        assert m.model_type == "scanvi"
        assert m.n_cells == 1000

    def test_load_icb_reference_missing(self, tmp_path):
        from src.data.loaders import load_icb_reference
        with pytest.raises(FileNotFoundError):
            load_icb_reference(data_root=tmp_path)


class TestReferenceValidators:
    """Test validate_reference_tree."""

    def test_missing_ref_dir(self, tmp_path):
        from src.data.validators import validate_reference_tree
        issues = validate_reference_tree(tmp_path, strict=False)
        assert len(issues) >= 1
        assert any("data/ref" in i for i in issues)

    def test_valid_ref_tree(self, tmp_path):
        from src.data.validators import validate_reference_tree
        ref_root = tmp_path / "ref"
        manifest_dir = ref_root / "manifest"
        manifest_dir.mkdir(parents=True)
        model_dir = tmp_path / "models" / "icb"
        model_dir.mkdir(parents=True)
        map_dir = tmp_path / "mappings" / "icb"
        map_dir.mkdir(parents=True)
        ref_h5ad = map_dir / "reference_adata.h5ad"
        ref_h5ad.write_text("dummy")
        (map_dir / "label_dict.json").write_text("{}")
        (map_dir / "mapping_stats.json").write_text("{}")

        manifest = {
            "model_dir": str(model_dir),
            "mappings_dir": str(map_dir),
            "reference_h5ad": str(ref_h5ad),
            "n_cells": 5000,
        }
        (manifest_dir / "reference_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        issues = validate_reference_tree(tmp_path, strict=False)
        assert len(issues) == 0

    def test_strict_mode(self, tmp_path):
        from src.data.validators import validate_reference_tree
        issues = validate_reference_tree(tmp_path, strict=True)
        assert any("ERROR" in i for i in issues)


class TestBackwardCompatibility:
    """Verify existing APIs are not broken."""

    def test_validate_onboarding_tree_unchanged(self, tmp_path):
        from src.data.validators import validate_onboarding_tree
        issues = validate_onboarding_tree(tmp_path)
        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_existing_loader_functions(self):
        from src.data.loaders import (
            load_project_manifest,
            load_standardized_tables,
            load_project_h5ad,
            load_h5ad,
            load_visium_h5ad,
            load_chromium_h5ad,
        )
        assert callable(load_project_manifest)
        assert callable(load_standardized_tables)
        assert callable(load_project_h5ad)

    def test_existing_validator_functions(self):
        from src.data.validators import (
            validate_required_columns,
            validate_chromium_meta,
            validate_visium_positions,
            validate_onboarding_tree,
        )
        assert callable(validate_required_columns)
        assert callable(validate_chromium_meta)
