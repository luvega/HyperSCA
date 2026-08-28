#!/usr/bin/env python3
"""Audit outcome-blind spatial perturbation bridge capability."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="audit outcome-blind spatial perturbation capability")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--candidate-id", default="gse274447_msafe_bridge")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    options = _arguments().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if options.registry is None:
        options.registry = repository_root / "configs" / "spatial_perturbation_bridge_candidates_v1.json"
    repository_root_text = str(repository_root)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)
    from src.evaluation.spatial_perturbation_registry import (
        SpatialPerturbationRegistryError,
        audit_bridge_capability,
        load_asset_metadata,
        load_bridge_candidates,
        unavailable_metadata_summary,
        write_bridge_capability_exclusively,
    )
    try:
        candidates = load_bridge_candidates(options.registry)
        candidate = candidates[options.candidate_id]
        summary = (load_asset_metadata(options.asset_root, candidate) if options.asset_root is not None
                   else unavailable_metadata_summary(candidate))
        write_bridge_capability_exclusively(
            options.output, audit_bridge_capability(candidate, summary), candidate=candidate
        )
    except (KeyError, SpatialPerturbationRegistryError) as exc:
        _arguments().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
