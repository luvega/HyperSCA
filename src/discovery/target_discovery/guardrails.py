"""Safety checks for data-driven target discovery runs."""
from __future__ import annotations

from pathlib import Path

from src.discovery.target_discovery.config import TargetDiscoveryConfig


FORBIDDEN_SOURCE_PARTS = {"output"}
FORBIDDEN_RESULT_FILE_PREFIXES = (
    "DEGs_",
    "Specific_DEGs",
    "Major_specific_genes",
    "Mid_specific_genes",
    "icb_nonresponse_target",
    "target_ranking",
)


def _has_forbidden_part(path: Path) -> bool:
    return any(part.lower() in FORBIDDEN_SOURCE_PARTS for part in path.parts)


def _has_forbidden_result_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        if child.name.startswith(FORBIDDEN_RESULT_FILE_PREFIXES):
            return True
    return False


def validate_no_manual_target_lists(config: TargetDiscoveryConfig) -> None:
    """Reject manual target seeds in every target-discovery mode."""
    if getattr(config, "focused_genes", ()):
        raise ValueError("target discovery forbids focused_genes or manual target lists")


def validate_from_scratch_config(config: TargetDiscoveryConfig) -> None:
    """Reject prior-target or legacy-result inputs in from-scratch mode."""
    validate_no_manual_target_lists(config)
    if not config.from_scratch:
        return

    paths = config.paths
    required = {
        "from_scratch_de_dir": paths.from_scratch_de_dir,
        "from_scratch_expression_dir": paths.from_scratch_expression_dir,
        "from_scratch_spatial_dir": paths.from_scratch_spatial_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"from-scratch discovery requires recomputed artifact dirs: {', '.join(missing)}")

    source_dirs = {
        "icb_dir": paths.icb_dir,
        "ifng_dir": paths.ifng_dir,
        "neu_dir": paths.neu_dir,
        "st_dir": paths.st_dir,
    }
    for label, path in source_dirs.items():
        path = Path(path)
        if _has_forbidden_part(path):
            raise ValueError(f"from-scratch discovery forbids legacy output source for {label}: {path}")
        if _has_forbidden_result_file(path):
            raise ValueError(f"from-scratch discovery forbids legacy result files under {label}: {path}")


def validate_no_forbidden_result_files(path: Path, *, label: str) -> None:
    """Reject known legacy result files in a recomputed input directory."""
    if _has_forbidden_result_file(Path(path)):
        raise ValueError(f"forbidden legacy result file found in {label}: {path}")
