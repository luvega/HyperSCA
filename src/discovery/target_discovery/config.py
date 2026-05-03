"""Configuration for target discovery runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DiscoveryPaths:
    root: Path
    data_dir: Path
    neu_dir: Path
    ifng_dir: Path
    icb_dir: Path
    st_dir: Path
    output_base: Path
    icb_h5ad_path: Path
    reference_manifest_path: Path

    @classmethod
    def default(cls, root: Path | None = None, output_base: Path | None = None) -> "DiscoveryPaths":
        root = project_root() if root is None else Path(root)
        data_dir = root / "data"
        return cls(
            root=root,
            data_dir=data_dir,
            neu_dir=Path(
                r"G:\scCRC_Neu\downstream_analyses_de_analysis"
                r"\0downstream_analyses_de_analysis\de_analysis"
                r"\de_analysis_tumor_mss_msi\deseq2_dgea"
            ),
            ifng_dir=Path(r"F:\scCRC_IFNG"),
            icb_dir=Path(r"G:\scCRC_ICB\output"),
            st_dir=Path(r"G:\ST_CRC_MSS"),
            output_base=Path(output_base) if output_base else root / "results" / "discovery" / "target_discovery",
            icb_h5ad_path=data_dir / "scRNA" / "scCRC_ICB" / "expression.h5ad",
            reference_manifest_path=data_dir / "ref" / "manifest" / "reference_manifest.json",
        )


@dataclass(frozen=True)
class GeometryModeConfig:
    modes: tuple[str, ...] = ("hyperbolic", "euclidean")
    geometry_k: int = 4
    geometry_blend: float = 0.30


@dataclass(frozen=True)
class TargetDiscoveryConfig:
    paths: DiscoveryPaths = field(default_factory=DiscoveryPaths.default)
    geometry: GeometryModeConfig = field(default_factory=GeometryModeConfig)
    max_perturb: int = 50
    platform: str = "all"
    focused_genes: tuple[str, ...] = ()
    hierarchy_levels: int = 3
    run_id: str | None = None
    random_seed: int = 42
    device: str = "cuda"
    skip_figures: bool = False

    @classmethod
    def default(cls) -> "TargetDiscoveryConfig":
        return cls()

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return datetime.now().strftime("%Y%m%d_%H%M%S")
