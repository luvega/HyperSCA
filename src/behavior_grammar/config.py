"""Configuration for behavior grammar sidecar runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BehaviorGrammarPaths:
    root: Path
    discovery_manifest: Path | None
    step4_dir: Path | None
    output_base: Path

    @classmethod
    def default(
        cls,
        root: Path | None = None,
        discovery_manifest: Path | None = None,
        step4_dir: Path | None = None,
        output_base: Path | None = None,
    ) -> "BehaviorGrammarPaths":
        root = project_root() if root is None else Path(root)
        return cls(
            root=root,
            discovery_manifest=Path(discovery_manifest) if discovery_manifest else None,
            step4_dir=Path(step4_dir) if step4_dir else root / "results" / "step4",
            output_base=Path(output_base) if output_base else root / "results" / "behavior_grammar",
        )


@dataclass(frozen=True)
class BehaviorGrammarConfig:
    paths: BehaviorGrammarPaths = field(default_factory=BehaviorGrammarPaths.default)
    run_id: str | None = None
    max_rules: int = 30
    time_steps: int = 12
    dt: float = 1.0
    sensitivity_delta: float = 0.10
    random_seed: int = 42

    @classmethod
    def default(cls) -> "BehaviorGrammarConfig":
        return cls()

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return datetime.now().strftime("%Y%m%d_%H%M%S")
