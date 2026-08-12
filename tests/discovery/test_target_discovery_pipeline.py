from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import pytest

from src.discovery.target_discovery.config import DiscoveryPaths, TargetDiscoveryConfig
from src.discovery.target_discovery.lightweight_stages import (
    CandidateDiscoveryStage,
    ExpressionAssemblyStage,
    SpatialContextStage,
)
from src.discovery.target_discovery.pipeline import TargetDiscoveryPipeline
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


@dataclass
class FakeStage:
    name: str
    key: str
    value: Any

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        context.writer.write_json(f"{self.name}.json", {"ok": True}, section="fake")
        return {self.key: self.value}


def _config(tmp_path) -> TargetDiscoveryConfig:
    paths = DiscoveryPaths.default(root=tmp_path, output_base=tmp_path / "runs")
    return TargetDiscoveryConfig(paths=paths, run_id="smoke", device="cpu")


def test_pipeline_runs_fake_stages_and_writes_manifest(tmp_path):
    cfg = _config(tmp_path)
    pipeline = TargetDiscoveryPipeline(
        config=cfg,
        stages=[
            FakeStage("one", "a", 1),
            FakeStage("two", "b", 2),
        ],
        icb_data_mode_detector=lambda config: "deg_only",
    )

    outputs = pipeline.run()

    assert outputs["a"] == 1
    assert outputs["b"] == 2
    assert (tmp_path / "runs" / "smoke" / "manifest.json").exists()
    assert (tmp_path / "runs" / "smoke" / "fake" / "one.json").exists()


def test_default_config_uses_evidence_gated_ranking(tmp_path):
    cfg = _config(tmp_path)

    assert cfg.score_profile == "evidence_gated"


def test_target_discovery_cli_does_not_accept_manual_gene_seeds():
    from scripts.run_target_discovery import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--genes", "GREM1"])


def test_default_target_discovery_stages_include_append_only_sidecars():
    from src.discovery.target_discovery.pipeline import default_target_discovery_stages

    names = [stage.name for stage in default_target_discovery_stages()]

    assert "communication_flow" in names
    assert names.index("communication_flow") == names.index("causal_discovery") + 1
    assert "mechanism_evidence" in names
    assert names.index("mechanism_evidence") == names.index("evidence_scoring") + 1


def test_lightweight_stages_can_run_on_synthetic_files(tmp_path):
    paths = DiscoveryPaths(
        root=tmp_path,
        data_dir=tmp_path / "data",
        neu_dir=tmp_path / "neu",
        ifng_dir=tmp_path / "ifng",
        icb_dir=tmp_path / "icb",
        st_dir=tmp_path / "st",
        output_base=tmp_path / "runs",
        icb_h5ad_path=tmp_path / "data" / "scRNA" / "scCRC_ICB" / "expression.h5ad",
        reference_manifest_path=tmp_path / "data" / "ref" / "manifest" / "reference_manifest.json",
    )
    paths.neu_dir.mkdir(parents=True, exist_ok=True)
    paths.icb_dir.mkdir(parents=True, exist_ok=True)
    (paths.ifng_dir / "results" / "tables").mkdir(parents=True, exist_ok=True)
    paths.st_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"symbol": ["A"], "padj": [0.01], "log2FoldChange": [1.0]}).to_csv(
        paths.neu_dir / "Fibroblast_S1-DESeq2_result.tsv", sep="\t", index=False
    )
    pd.DataFrame({"s1": [1.0], "s2": [3.0]}, index=["A"]).to_csv(
        paths.neu_dir / "Fibroblast_S1-NormalizedCounts.tsv", sep="\t"
    )
    pd.DataFrame({"Fibro_ADAMDEC1": [1.0, 0.5]}).to_csv(paths.st_dir / "STmetadata_unit.csv", index=False)

    cfg = TargetDiscoveryConfig(paths=paths, run_id="light", device="cpu")
    pipeline = TargetDiscoveryPipeline(
        cfg,
        stages=[CandidateDiscoveryStage(), ExpressionAssemblyStage(), SpatialContextStage()],
        icb_data_mode_detector=lambda config: "deg_only",
    )
    out = pipeline.run()
    assert not out["candidate_pool"].empty
    assert out["cluster_expression"].shape[0] == 1
    assert out["spatial_adjacency"].shape == (1, 1)


def test_candidate_discovery_does_not_seed_manual_focus_genes(tmp_path):
    paths = DiscoveryPaths(
        root=tmp_path,
        data_dir=tmp_path / "data",
        neu_dir=tmp_path / "neu",
        ifng_dir=tmp_path / "ifng",
        icb_dir=tmp_path / "icb",
        st_dir=tmp_path / "st",
        output_base=tmp_path / "runs",
        icb_h5ad_path=tmp_path / "data" / "scRNA" / "scCRC_ICB" / "expression.h5ad",
        reference_manifest_path=tmp_path / "data" / "ref" / "manifest" / "reference_manifest.json",
    )
    paths.neu_dir.mkdir(parents=True, exist_ok=True)
    paths.icb_dir.mkdir(parents=True, exist_ok=True)
    (paths.ifng_dir / "results" / "tables").mkdir(parents=True, exist_ok=True)

    cfg = TargetDiscoveryConfig(paths=paths, run_id="no_manual_focus", device="cpu")
    pipeline = TargetDiscoveryPipeline(
        cfg,
        stages=[CandidateDiscoveryStage()],
        icb_data_mode_detector=lambda config: "deg_only",
    )

    out = pipeline.run()

    assert out["candidate_pool"].empty
