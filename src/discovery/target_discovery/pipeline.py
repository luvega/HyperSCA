"""Pipeline orchestration for target discovery."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from src.discovery.target_discovery.artifacts import ArtifactWriter
from src.discovery.target_discovery.config import TargetDiscoveryConfig
from src.discovery.target_discovery.stage import DiscoveryStage, TargetDiscoveryRunContext


def default_icb_data_mode_detector(config: TargetDiscoveryConfig) -> str:
    if config.paths.reference_manifest_path.exists():
        return "reference"
    if config.paths.icb_h5ad_path.exists():
        return "h5ad"
    return "deg_only"


class TargetDiscoveryPipeline:
    def __init__(
        self,
        config: TargetDiscoveryConfig,
        stages: Iterable[DiscoveryStage] | None = None,
        icb_data_mode_detector: Callable[[TargetDiscoveryConfig], str] = default_icb_data_mode_detector,
    ):
        self.config = config
        self.stages = list(stages or [])
        self.icb_data_mode_detector = icb_data_mode_detector

    def run(self) -> dict[str, Any]:
        run_id = self.config.resolved_run_id()
        writer = ArtifactWriter(self.config.paths.output_base, run_id=run_id)
        context = TargetDiscoveryRunContext(
            config=self.config,
            writer=writer,
            started_at=time.time(),
            icb_data_mode=self.icb_data_mode_detector(self.config),
        )
        outputs: dict[str, Any] = {}
        try:
            for stage in self.stages:
                started_at = time.time()
                produced = dict(stage.run(context, outputs))
                outputs.update(produced)
                writer.record_stage(stage.name, time.time() - started_at, produced)
            outputs["run_dir"] = writer.run_dir
            outputs["manifest_path"] = writer.finalize()
            return outputs
        except Exception:
            writer.finalize()
            raise
