"""Stage protocol and run context for target discovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from src.discovery.target_discovery.artifacts import ArtifactWriter
from src.discovery.target_discovery.config import TargetDiscoveryConfig


@dataclass
class TargetDiscoveryRunContext:
    config: TargetDiscoveryConfig
    writer: ArtifactWriter
    started_at: float
    icb_data_mode: str


class DiscoveryStage(Protocol):
    name: str

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        ...
