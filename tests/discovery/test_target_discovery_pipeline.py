from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.discovery.target_discovery.config import DiscoveryPaths, TargetDiscoveryConfig
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
