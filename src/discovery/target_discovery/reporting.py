"""Markdown reporting for target discovery."""
from __future__ import annotations

from pathlib import Path


def build_target_discovery_report(context, inputs) -> str:
    ranking = inputs.get("target_ranking")
    comparison = inputs.get("mode_comparison", {})
    hubs = inputs.get("retained_hubs")
    lines = [
        "# Target Discovery Report",
        "",
        f"- Run ID: `{context.writer.run_id}`",
        f"- ICB data mode: `{context.icb_data_mode}`",
        f"- Platform filter: `{context.config.platform}`",
        "",
        "## Summary",
    ]
    if ranking is not None:
        lines.append(f"- Candidate ranking size: {len(ranking)}")
        if not ranking.empty:
            top = ranking.head(10)["gene"].astype(str).tolist()
            lines.append(f"- Top candidates: {', '.join(top)}")
    if hubs is not None:
        lines.append(f"- Retained hubs: {len(hubs)}")
    if comparison:
        geom = comparison.get("geometry", {})
        lines.extend(
            [
                "",
                "## Geometry Comparison",
                f"- Hyperbolic separation: {geom.get('hyp_separation', 0):.4f}",
                f"- Euclidean separation: {geom.get('euc_separation', 0):.4f}",
            ]
        )
    lines.extend(["", "## Outputs", f"- Manifest: `{context.writer.run_dir / 'manifest.json'}`"])
    return "\n".join(lines) + "\n"


def build_migration_notes(run_dir: Path) -> str:
    mappings = {
        "candidate_pool.csv": "candidates/candidate_pool.csv",
        "target_ranking.csv": "scoring/target_ranking.csv",
        "evidence_matrix.csv": "scoring/evidence_matrix.csv",
        "hub_targets_retained.csv": "scoring/hub_targets_retained.csv",
        "spatiotemporal_regulatory_combos.csv": "scoring/spatiotemporal_regulatory_combos.csv",
        "mode_comparison.json": "scoring/mode_comparison.json",
        "target_discovery_report.md": "reports/target_discovery_report.md",
        "comparison_report.md": "reports/migration_notes.md plus scoring/mode_comparison.json",
    }
    lines = [
        "# Target Discovery Migration Notes",
        "",
        "Old root: `results/integration/discovery/`",
        f"New root: `{run_dir}`",
        "",
        "| Old path | New path |",
        "| --- | --- |",
    ]
    for old, new in mappings.items():
        lines.append(f"| `{old}` | `{new}` |")
    return "\n".join(lines) + "\n"
