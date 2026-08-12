from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.discovery.target_discovery.guardrails import (
    validate_from_scratch_config,
    validate_no_manual_target_lists,
    validate_no_forbidden_result_files,
)


def _config(tmp_path, *, from_scratch: bool = True, focused_genes=()):
    paths = SimpleNamespace(
        from_scratch_de_dir=tmp_path / "from_scratch" / "de",
        from_scratch_expression_dir=tmp_path / "from_scratch" / "expression",
        from_scratch_spatial_dir=tmp_path / "from_scratch" / "spatial",
        icb_dir=tmp_path / "raw_icb",
        ifng_dir=tmp_path / "raw_ifng",
        neu_dir=tmp_path / "raw_neu",
        st_dir=tmp_path / "raw_st",
    )
    for path in (
        paths.from_scratch_de_dir,
        paths.from_scratch_expression_dir,
        paths.from_scratch_spatial_dir,
        paths.icb_dir,
        paths.ifng_dir,
        paths.neu_dir,
        paths.st_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(from_scratch=from_scratch, focused_genes=focused_genes, paths=paths)


def test_validate_from_scratch_config_rejects_manual_target_lists(tmp_path):
    config = _config(tmp_path, focused_genes=("GREM1",))

    with pytest.raises(ValueError, match="focused_genes|manual target"):
        validate_from_scratch_config(config)


def test_validate_no_manual_target_lists_applies_to_every_discovery_mode(tmp_path):
    config = _config(tmp_path, from_scratch=False, focused_genes=("GREM1",))

    with pytest.raises(ValueError, match="focused_genes|manual target"):
        validate_no_manual_target_lists(config)


def test_validate_from_scratch_config_rejects_legacy_output_paths(tmp_path):
    config = _config(tmp_path)
    config.paths.icb_dir = tmp_path / "scCRC_ICB" / "output"

    with pytest.raises(ValueError, match="legacy output source"):
        validate_from_scratch_config(config)


def test_validate_no_forbidden_result_files_rejects_legacy_deg_files(tmp_path):
    source = tmp_path / "raw_icb"
    source.mkdir()
    pd.DataFrame({"gene": ["OLD"]}).to_csv(source / "DEGs_MSS_response_Mid_lfc0.5.csv", index=False)

    with pytest.raises(ValueError, match="forbidden legacy result file"):
        validate_no_forbidden_result_files(source, label="raw_icb")
