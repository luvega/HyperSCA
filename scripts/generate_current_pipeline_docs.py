"""Generate commit-friendly project progress docs from local benchmark artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/benchmarks/hyperbolic_spatial_crc_v3_two_candidate_downstream_20260622"
DOC_DIR = ROOT / "docs/research"
FIG_DIR = DOC_DIR / "figures"
REPORT_PATH = DOC_DIR / "hypersca_benchmark_progress_20260622.md"
SNAPSHOT_PATH = DOC_DIR / "hypersca_benchmark_progress_20260622.json"
INVENTORY_PATH = DOC_DIR / "hypersca_project_progress_inventory_20260622.md"
SUBMISSION_PATH = ROOT / "docs/github_submission_20260622.md"
FLOWCHART_PNG = FIG_DIR / "hypersca_current_pipeline_flowchart_20260622.png"
FLOWCHART_SVG = FIG_DIR / "hypersca_current_pipeline_flowchart_20260622.svg"
SUMMARY_FIG = FIG_DIR / "hypersca_two_candidate_downstream_summary_20260622.png"
OVERVIEW_FIG = FIG_DIR / "hypersca_current_pipeline_overview_imagegen_20260622.png"


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(item) for item in row) + " |")
    return "\n".join(lines)


def _metric(metrics: pd.DataFrame, method: str, metric: str) -> float:
    found = metrics[(metrics["method"] == method) & (metrics["metric"] == metric)]
    if found.empty:
        return float("nan")
    return float(found.iloc[0]["mean"])


def _load_inputs() -> dict[str, pd.DataFrame | dict]:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Missing benchmark run directory: {RUN_DIR}")

    manifest = json.loads((RUN_DIR / "two_candidate_downstream_manifest.json").read_text())
    return {
        "manifest": manifest,
        "metrics": pd.read_csv(RUN_DIR / "main_two_candidate_seed_metrics.csv"),
        "targets": pd.read_csv(RUN_DIR / "candidate_target_ablation/candidate_target_ablation_summary.csv"),
        "grid": pd.read_csv(RUN_DIR / "visiumhd_cell2location_vs_rctd_grid_metrics.csv"),
    }


def _write_progress_report(data: dict[str, pd.DataFrame | dict]) -> None:
    manifest = data["manifest"]
    metrics = data["metrics"]
    targets = data["targets"]
    grid = data["grid"]

    full = "hvae_hierarchy_spatial_v3_product"
    no_radial = "hvae_hierarchy_spatial_v3_product__without_radial_depth_loss"
    metric_rows = [
        [
            "Full v3 product",
            _metric(metrics, full, "knn_label_purity"),
            _metric(metrics, full, "distance_label_auc"),
            _metric(metrics, full, "heldout_context_edge_auc"),
            _metric(metrics, full, "celcomen_interaction_energy_auc"),
            _metric(metrics, full, "target_rank_delta_nonzero"),
        ],
        [
            "Without radial-depth loss",
            _metric(metrics, no_radial, "knn_label_purity"),
            _metric(metrics, no_radial, "distance_label_auc"),
            _metric(metrics, no_radial, "heldout_context_edge_auc"),
            _metric(metrics, no_radial, "celcomen_interaction_energy_auc"),
            _metric(metrics, no_radial, "target_rank_delta_nonzero"),
        ],
    ]
    target_rows = [
        [
            row["method"].replace("hvae_hierarchy_spatial_v3_product", "v3_product"),
            int(row["n_cells_used"]),
            int(row["n_genes_used"]),
            int(row["n_scored_target_genes"]),
            int(row["n_changed_ranks"]),
            float(row["max_topk_enrichment_score_delta"]),
            row["quality_gate"],
        ]
        for _, row in targets.iterrows()
    ]
    grid_rows = [
        [
            row["cell_type"],
            float(row["spearman_grid_fraction"]),
            float(row["cell2location_mean_fraction"]),
            float(row["rctd_mean_fraction"]),
        ]
        for _, row in grid.sort_values("spearman_grid_fraction", ascending=False).iterrows()
    ]

    report = f"""# HyperSCA Benchmark Progress Summary 2026-06-22

Generated from local benchmark artifacts at `{RUN_DIR.relative_to(ROOT)}`.

## Current Status

The current GitHub-ready interpretation is conservative: the main method comparison is restricted to two internally trained v3 branches, `hvae_hierarchy_spatial_v3_product` and `hvae_hierarchy_spatial_v3_product__without_radial_depth_loss`. SCimilarity is retained only as an external pretrained appendix reference, not as a main ranking competitor.

The full downstream audit remains `audit_only_no_promotion`. It does not modify the active target ranking because neither candidate produced a non-zero target rank delta or target enrichment improvement. The GPU path was active for the v3 runs, and VisiumHD cell2location full output passed the expected 545,913-row abundance check.

## Main Two-Candidate Metrics

{_markdown_table(["Candidate", "kNN purity", "Label AUC", "Held-out context AUC", "Celcomen energy AUC", "Target rank delta"], metric_rows)}

## Target Discovery Gate

{_markdown_table(["Candidate", "Cells", "Genes", "Scored targets", "Changed ranks", "Top-k enrichment delta", "Gate"], target_rows)}

## VisiumHD RCTD and cell2location Concordance

Dominant grid-label concordance between RCTD and cell2location is `0.827`. The strongest shared spatial abundance patterns are tumor and fibroblast compartments; T-cell and ILC fractions are weakly concordant and should be reviewed visually before biological claims.

{_markdown_table(["Cell type", "Grid Spearman", "cell2location mean", "RCTD mean"], grid_rows)}

## Interpretation

- The two v3 branches are stable enough for continued downstream testing under the same 5k-cell, 6k-gene, 3-seed protocol.
- The radial-depth branch is not yet functionally justified: removing radial depth slightly improves label AUC, while the full model slightly improves held-out context and interaction-energy AUC.
- Prototype and radial hierarchy objectives still behave near chance; these remain algorithm-development targets rather than release-ready evidence.
- Xenium remains panel-aware by design. Targeted-panel gene coverage does not support whole-transcriptome RCTD or cell2location assumptions, so Xenium should be evaluated with panel-compatible spatial annotation and visualization.

## Next Quality Gates

1. Keep the active target ranking unchanged until target rank delta, target enrichment, or spatial niche biology shows reviewable improvement.
2. Use VisiumHD segmented RCTD and cell2location as full downstream references for spatial niche interpretation.
3. Continue loss-component ablation with explicit prototype, teacher, context-edge, Euclidean residual, and radial-depth reporting.
4. Only promote a v3 branch after seed stability and bootstrap intervals support a reproducible gain.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run_dir": str(RUN_DIR.relative_to(ROOT)),
        "quality_gate": manifest.get("quality_gate", "audit_only_no_promotion"),
        "main_ranking_modified": bool(manifest.get("main_ranking_modified", False)),
        "promotion_recommended": bool(manifest.get("promotion_recommended", False)),
        "candidate_metrics": {
            row[0]: {
                "knn_label_purity": row[1],
                "distance_label_auc": row[2],
                "heldout_context_edge_auc": row[3],
                "celcomen_interaction_energy_auc": row[4],
                "target_rank_delta_nonzero": row[5],
            }
            for row in metric_rows
        },
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")


def _status_lines() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _category(path: str) -> str:
    if path == "README.md" or path.startswith("docs/") or path.startswith("reports/"):
        return "Documentation and reports"
    if path.startswith("scripts/generate_") or path.startswith("scripts/run_") or path.startswith("scripts/postprocess_"):
        return "Benchmark and workflow scripts"
    if path.startswith("src/discovery/target_discovery/") or path.startswith("tests/discovery/"):
        return "Target discovery and benchmark tests"
    if path.startswith("src/models/hyperbolic/") or "hyperbolic" in path:
        return "Hyperbolic representation modules"
    if path.startswith("src/perturbation/") or path.startswith("src/evaluation/"):
        return "Perturbation and spatial evaluation"
    if path.startswith("src/data/prior_db/") or path.startswith("configs/") or "onboarding" in path:
        return "Data onboarding and prior database"
    return "Other modified project files"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_project_inventory() -> None:
    grouped: dict[str, list[str]] = {}
    for line in _status_lines():
        path = line[3:]
        grouped.setdefault(_category(path), []).append(f"`{line[:2].strip() or 'M'} {path}`")

    rows = []
    for category, paths in sorted(grouped.items()):
        sample = "<br>".join(paths[:8])
        if len(paths) > 8:
            sample += f"<br>... +{len(paths) - 8} more"
        rows.append([category, len(paths), sample])

    raw_dirs = [
        ROOT / "results/benchmarks/hyperbolic_spatial_crc_v3_two_candidate_downstream_20260622",
        ROOT / "results/benchmarks/unified_spatial_annotation_cell2location_visiumhd_full",
        ROOT / "results/benchmarks/hyperbolic_spatial_crc_v3_visiumhd_loss_ablation_celcomen_energy_20260622",
    ]
    raw_rows = [
        [str(path.relative_to(ROOT)), f"{_dir_size(path) / (1024 * 1024):.1f} MB", "local ignored result source"]
        for path in raw_dirs
        if path.exists()
    ]

    inventory = f"""# HyperSCA Project Progress Inventory 2026-06-22

This inventory summarizes the current local worktree for preparing a GitHub submission. It is intentionally conservative: large raw outputs remain local, and algorithm changes should be reviewed separately from the documentation-only progress commit.

## Local Worktree Categories

{_markdown_table(["Category", "File count", "Representative paths"], rows)}

## Large Local Benchmark Outputs

{_markdown_table(["Path", "Approx. size", "Policy"], raw_rows)}

## Submission Recommendation

- Commit the README, compact progress report, workflow figures, and submission notes first.
- Do not stage raw `results/` outputs unless a reviewer explicitly requests a small derived artifact.
- Review algorithm code separately: target-discovery modules, hyperbolic hierarchy losses, spatial annotation scripts, and corresponding tests have broader behavioral impact than the documentation refresh.
- Keep Xenium panel-aware handling as a distinct branch of the workflow and avoid whole-transcriptome deconvolution claims for targeted-panel data.
"""
    INVENTORY_PATH.write_text(inventory, encoding="utf-8")


def _draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        2.8,
        1.05,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#2f3b46",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + 1.4, y + 0.53, text, ha="center", va="center", fontsize=10.5, color="#16202a", wrap=True)


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.25,
            color="#5b6770",
        )
    )


def _write_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(0.35, 7.55, "HyperSCA Current Analysis Workflow", fontsize=18, weight="bold", color="#102030")
    ax.text(
        0.35,
        7.15,
        "Single-cell and spatial omics benchmark path used for the 2026-06-22 GitHub-ready audit snapshot",
        fontsize=10.5,
        color="#4a5560",
    )

    boxes = [
        ((0.45, 5.8), "Input atlases\nscCRC_ICB reference\nOSTA Visium/HD/Xenium", "#d8edf3"),
        ((4.0, 5.8), "Unified annotation\nRCTD + cell2location GPU\nXenium panel-aware", "#dff0dc"),
        ((7.55, 5.8), "v3 representation\nProduct hyperbolic + Euclidean\nprototype/radial/context losses", "#efe4f4"),
        ((11.1, 5.8), "Benchmark gates\n5k cells / 6k genes\n3 seeds + block holdout", "#f5e4cd"),
        ((2.2, 3.55), "Downstream evidence\nTarget ranking\nTarget/context enrichment", "#e8eef8"),
        ((5.75, 3.55), "Spatial interpretation\nVisiumHD segmented RCTD\ncell2location concordance", "#e4f0e8"),
        ((9.3, 3.55), "Release decision\nAudit-only no promotion\nSCimilarity appendix only", "#f1e5e0"),
        ((5.75, 1.35), "Commit-friendly outputs\nREADME, report, figures\nraw results remain local", "#f2f2f2"),
    ]
    for xy, text, color in boxes:
        _draw_box(ax, xy, text, color)

    _draw_arrow(ax, (3.25, 6.33), (3.95, 6.33))
    _draw_arrow(ax, (6.8, 6.33), (7.5, 6.33))
    _draw_arrow(ax, (10.35, 6.33), (11.05, 6.33))
    _draw_arrow(ax, (5.3, 5.8), (3.55, 4.62))
    _draw_arrow(ax, (8.95, 5.8), (7.15, 4.62))
    _draw_arrow(ax, (12.45, 5.8), (10.7, 4.62))
    _draw_arrow(ax, (4.0, 3.05), (6.05, 2.42))
    _draw_arrow(ax, (7.15, 3.05), (7.15, 2.42))
    _draw_arrow(ax, (10.7, 3.05), (8.25, 2.42))

    ax.text(
        0.45,
        0.45,
        "Promotion condition: non-zero target rank delta, improved target enrichment, or reviewable spatial niche biological gain.",
        fontsize=10.5,
        color="#31404d",
    )
    fig.tight_layout()
    fig.savefig(FLOWCHART_PNG, bbox_inches="tight")
    fig.savefig(FLOWCHART_SVG, bbox_inches="tight")
    plt.close(fig)


def _copy_summary_figure() -> None:
    src = RUN_DIR / "figures/two_candidate_downstream_summary.png"
    if src.exists():
        shutil.copy2(src, SUMMARY_FIG)


def _write_concept_overview() -> None:
    """Create a local README overview asset matching the imagegen concept prompt."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")

    for x in [i * 0.8 for i in range(21)]:
        ax.plot([x, x], [0.4, 8.6], color="#edf1f4", lw=0.6, zorder=0)
    for y in [0.4 + i * 0.8 for i in range(11)]:
        ax.plot([0.2, 15.8], [y, y], color="#edf1f4", lw=0.6, zorder=0)

    # Single-cell atlas clusters.
    cluster_specs = [
        (2.0, 5.9, "#75aadb"),
        (2.75, 5.0, "#67b7a4"),
        (1.45, 4.7, "#d98b73"),
        (2.45, 6.8, "#b996d5"),
    ]
    for cx, cy, color in cluster_specs:
        for i in range(18):
            dx = ((i % 6) - 2.5) * 0.12
            dy = ((i // 6) - 1.0) * 0.14
            ax.scatter(cx + dx, cy + dy, s=36, color=color, edgecolor="white", linewidth=0.6, zorder=3)

    # Spatial tissue map.
    tissue = FancyBboxPatch((4.6, 4.55), 2.15, 2.15, boxstyle="round,pad=0.04,rounding_size=0.35",
                            facecolor="#f4d5c7", edgecolor="#7d5a4f", linewidth=1.1, zorder=2)
    ax.add_patch(tissue)
    for i in range(70):
        x = 4.85 + (i % 10) * 0.17 + (0.04 if i % 2 else 0)
        y = 4.82 + (i // 10) * 0.22
        c = ["#9bc5e8", "#9fd0b8", "#e59a85", "#d6b9e7"][i % 4]
        ax.scatter(x, y, s=22, color=c, edgecolor="white", linewidth=0.45, zorder=3)

    # Hyperbolic manifold and hierarchy graph.
    ax.add_patch(plt.Circle((8.45, 5.65), 1.2, facecolor="#f1e6f7", edgecolor="#6d587a", lw=1.2, zorder=1))
    for r, alpha in [(0.35, 0.55), (0.7, 0.38), (1.05, 0.28)]:
        ax.add_patch(plt.Circle((8.45, 5.65), r, fill=False, edgecolor="#8e6aa5", lw=1.0, alpha=alpha, zorder=2))
    graph_points = [(8.0, 6.2), (8.45, 6.45), (8.85, 6.0), (8.18, 5.55), (8.72, 5.38), (8.36, 4.9)]
    for i, (x1, y1) in enumerate(graph_points[:-1]):
        for x2, y2 in graph_points[i + 1:i + 3]:
            ax.plot([x1, x2], [y1, y2], color="#66536f", lw=1.0, alpha=0.72, zorder=3)
    for x, y in graph_points:
        ax.scatter(x, y, s=58, color="#7b60a8", edgecolor="white", linewidth=0.8, zorder=4)

    # Quality-gated target dashboard.
    panel = FancyBboxPatch((10.75, 4.45), 2.2, 2.45, boxstyle="round,pad=0.04,rounding_size=0.12",
                           facecolor="#fff7e8", edgecolor="#8f7652", linewidth=1.1, zorder=2)
    ax.add_patch(panel)
    for i, height in enumerate([0.45, 0.9, 0.6, 1.25]):
        ax.add_patch(plt.Rectangle((11.05 + i * 0.34, 4.8), 0.18, height, color="#dfa957", zorder=3))
    for y in [6.25, 5.85, 5.45]:
        ax.plot([11.05, 12.45], [y, y], color="#c8b998", lw=1.2, zorder=3)
    for x, y, ok in [(12.45, 6.25, True), (12.45, 5.85, True), (12.45, 5.45, False)]:
        ax.scatter(x, y, s=70, color="#6fac7b" if ok else "#c96a5d", edgecolor="white", linewidth=0.8, zorder=4)

    # Report artifact.
    doc = FancyBboxPatch((13.75, 4.65), 1.25, 1.85, boxstyle="round,pad=0.04,rounding_size=0.08",
                         facecolor="#eef3f6", edgecolor="#52606a", linewidth=1.1, zorder=2)
    ax.add_patch(doc)
    for y in [6.05, 5.7, 5.35, 5.0]:
        ax.plot([14.0, 14.75], [y, y], color="#8fa0aa", lw=1.0, zorder=3)
    ax.add_patch(plt.Rectangle((14.0, 4.78), 0.38, 0.28, color="#75aadb", zorder=3))
    ax.add_patch(plt.Rectangle((14.45, 4.78), 0.28, 0.52, color="#67b7a4", zorder=3))

    # Flow arrows.
    arrow_pairs = [((3.25, 5.65), (4.45, 5.65)), ((6.8, 5.65), (7.2, 5.65)),
                   ((9.75, 5.65), (10.55, 5.65)), ((13.0, 5.65), (13.58, 5.65))]
    for start, end in arrow_pairs:
        _draw_arrow(ax, start, end)

    fig.tight_layout()
    fig.savefig(OVERVIEW_FIG, bbox_inches="tight")
    plt.close(fig)


def _write_submission_notes() -> None:
    body = dedent(
        f"""
        # GitHub Submission Plan 2026-06-22

        ## Recommended Scope

        This repository has a mixed worktree with many algorithm and benchmark changes. Do not stage the whole tree blindly. For the documentation-focused progress update, stage only the files below plus any explicitly reviewed algorithm files from the current benchmark branch.

        ## Suggested Documentation Commit

        ```bash
        git checkout -b docs/current-benchmark-progress-20260622
        git add README.md \\
          scripts/generate_current_pipeline_docs.py \\
          docs/research/hypersca_benchmark_progress_20260622.md \\
          docs/research/hypersca_benchmark_progress_20260622.json \\
          docs/research/hypersca_project_progress_inventory_20260622.md \\
          docs/research/figures/hypersca_current_pipeline_flowchart_20260622.png \\
          docs/research/figures/hypersca_current_pipeline_flowchart_20260622.svg \\
          docs/research/figures/hypersca_two_candidate_downstream_summary_20260622.png \\
          docs/research/figures/hypersca_current_pipeline_overview_imagegen_20260622.png \\
          docs/github_submission_20260622.md
        git commit -m "docs: summarize current HyperSCA benchmark progress"
        ```

        The `gh` CLI is not available in this environment, so PR creation should be done after installing `gh` or from the GitHub web UI.

        ## Draft PR Summary

        ```markdown
        ## Summary
        - regenerate README around the current HyperSCA spatial-hyperbolic benchmark workflow
        - add a 2026-06-22 benchmark progress report for the two internal v3 candidates
        - add reproducible workflow figures and a GitHub-ready submission checklist
        - add a local progress inventory so broad algorithm changes can be reviewed separately from docs

        ## Benchmark Status
        - main comparison is limited to `hvae_hierarchy_spatial_v3_product` and `hvae_hierarchy_spatial_v3_product__without_radial_depth_loss`
        - SCimilarity remains an external pretrained appendix reference
        - quality gate remains audit-only/no-promotion because target rank delta and target enrichment deltas are still zero
        - VisiumHD cell2location full abundance row check remains 545,913 rows

        ## Tests
        - `python scripts/generate_current_pipeline_docs.py`
        - `python -m py_compile scripts/generate_current_pipeline_docs.py`
        - `PYTHONPATH=. pytest tests -q -p no:cacheprovider`
        ```

        ## Raw Artifact Policy

        Keep large raw benchmark outputs in ignored `results/` paths. Commit compact reports, figures, and JSON snapshots under `docs/` or `reports/` only.
        """
    ).strip() + "\n"
    SUBMISSION_PATH.write_text(body, encoding="utf-8")


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_inputs()
    _write_progress_report(data)
    _write_project_inventory()
    _write_flowchart()
    _write_concept_overview()
    _copy_summary_figure()
    _write_submission_notes()
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {SNAPSHOT_PATH.relative_to(ROOT)}")
    print(f"Wrote {INVENTORY_PATH.relative_to(ROOT)}")
    print(f"Wrote {FLOWCHART_PNG.relative_to(ROOT)}")
    print(f"Wrote {FLOWCHART_SVG.relative_to(ROOT)}")
    print(f"Wrote {OVERVIEW_FIG.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_FIG.relative_to(ROOT)}")
    print(f"Wrote {SUBMISSION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
