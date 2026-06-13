"""Reproduce full spatial communication OT/flow sidecar on integration results."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.discovery.target_discovery.communication_flow import build_communication_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the distance-constrained LR communication OT/flow proxy "
            "for precomputed integration discovery results."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "results" / "integration" / "discovery",
        help="Directory containing hyperbolic/ and euclidean/ integration outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "integration" / "discovery" / "communication_ot_reproduction",
        help="Output directory for reproduced communication artifacts.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hyperbolic", "euclidean"],
        help="Geometry modes to reproduce.",
    )
    parser.add_argument(
        "--spatial-source",
        choices=["blended", "adjacency"],
        default="blended",
        help=(
            "Matrix used as the spatial constraint. Older integration runs did not "
            "persist raw spatial_adjacency separately; blended is the default "
            "compatibility choice."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Geometry-distance cost weight.")
    parser.add_argument("--beta", type=float, default=0.5, help="Spatial adjacency cost weight.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Entropic-style distance scale.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode_summaries: list[dict[str, Any]] = []
    all_pathways: list[pd.DataFrame] = []
    top_edges: list[pd.DataFrame] = []
    warnings: list[str] = []

    for mode in args.modes:
        mode_dir = args.source_dir / mode
        try:
            inputs = _load_mode_inputs(mode_dir, mode=mode, spatial_source=args.spatial_source)
        except FileNotFoundError as exc:
            warnings.append(str(exc))
            continue

        result = build_communication_flow(
            mode=mode,
            cluster_expr=inputs["cluster_expr"],
            spatial_adjacency=inputs["spatial_adjacency"],
            geometry_result={"dist_matrix": inputs["dist_matrix"]},
            causal_result={
                "node_labels": inputs["node_labels"],
                "type_mapping": inputs["type_mapping"],
                "causal_adjacency": inputs["causal_adjacency"],
            },
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
        )
        _write_mode_outputs(args.output_dir / mode, result)

        overall = result["direction_consistency"]["overall"]
        baseline = result["baseline_comparison"]
        mode_summaries.append(
            {
                "mode": mode,
                "n_nodes": overall["n_edges"] and baseline.get("n_nodes", len(inputs["node_labels"])),
                "n_lr_flow_edges": overall["n_edges"],
                "resolved_edges": overall["resolved_edges"],
                "forward_count": overall["forward_count"],
                "reverse_count": overall["reverse_count"],
                "ambiguous_count": overall["ambiguous_count"],
                "unresolved_count": overall["unresolved_count"],
                "direction_consistency_rate": overall["direction_consistency_rate"],
                "weighted_direction_consistency_rate": overall["weighted_direction_consistency_rate"],
                "total_flow": float(result["lr_flow_edges"]["flow_score"].sum())
                if not result["lr_flow_edges"].empty
                else 0.0,
                "max_flow": float(result["lr_flow_edges"]["flow_score"].max())
                if not result["lr_flow_edges"].empty
                else 0.0,
                "causal_spatial_consistency": baseline.get("causal_spatial_consistency", 0.0),
                "baseline_spatial_consistency": baseline.get("baseline_spatial_consistency", 0.0),
                "spatial_consistency_gain": baseline.get("spatial_consistency_gain", 0.0),
                "spatial_source": args.spatial_source,
            }
        )

        pathway = result["pathway_summary"].copy()
        if not pathway.empty:
            pathway.insert(0, "mode", mode)
            all_pathways.append(pathway)

        edges = result["lr_flow_edges"].copy()
        if not edges.empty:
            selected = edges.sort_values("flow_score", ascending=False).head(25).copy()
            selected.insert(0, "rank_in_mode", range(1, len(selected) + 1))
            top_edges.append(selected)

    summary = pd.DataFrame(mode_summaries)
    pathway_all = pd.concat(all_pathways, ignore_index=True) if all_pathways else pd.DataFrame()
    top_edges_all = pd.concat(top_edges, ignore_index=True) if top_edges else pd.DataFrame()
    summary.to_csv(args.output_dir / "mode_summary.csv", index=False)
    pathway_all.to_csv(args.output_dir / "pathway_summary_all_modes.csv", index=False)
    top_edges_all.to_csv(args.output_dir / "top_lr_flow_edges_all_modes.csv", index=False)

    figure_path = _plot_reproduction(args.output_dir, summary, pathway_all)
    report_path = _write_report(args.output_dir, summary, pathway_all, top_edges_all, args, warnings)
    manifest = {
        "run_id": "communication_ot_reproduction",
        "elapsed_seconds": time.time() - started_at,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "modes": args.modes,
        "spatial_source": args.spatial_source,
        "alpha": args.alpha,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "artifacts": _relative_artifact_list(args.output_dir),
        "figure": str(figure_path),
        "report": str(report_path),
        "warnings": warnings,
        "interpretation_boundary": (
            "Distance-constrained LR communication flow is an OT-style proxy "
            "for direction-consistency comparison, not causal proof and not COMMOT."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Output directory: {args.output_dir}")
    print(f"Summary: {args.output_dir / 'mode_summary.csv'}")
    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")
    return 0


def _load_mode_inputs(mode_dir: Path, *, mode: str, spatial_source: str) -> dict[str, Any]:
    required = [
        mode_dir / "step2" / "cluster_expr.csv",
        mode_dir / "step2" / "node_info.json",
        mode_dir / "step2" / "causal_adjacency.npy",
        mode_dir / "geometry" / "distance.npy",
        mode_dir / "geometry" / f"{spatial_source}.npy",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing {mode} communication input: {path}")

    node_info = json.loads((mode_dir / "step2" / "node_info.json").read_text(encoding="utf-8"))
    node_labels = [str(label) for label in node_info["node_labels"]]
    cluster_expr = pd.read_csv(mode_dir / "step2" / "cluster_expr.csv")
    first_col = cluster_expr.columns[0]
    if first_col in {"celltype", "cell_type", "index", "Unnamed: 0"}:
        cluster_expr = cluster_expr.set_index(first_col)
    cluster_expr.index = cluster_expr.index.astype(str)
    cluster_expr = cluster_expr.reindex(node_labels).fillna(0.0)
    cluster_expr = cluster_expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return {
        "mode": mode,
        "node_labels": node_labels,
        "type_mapping": {str(k): str(v) for k, v in node_info["type_mapping"].items()},
        "cluster_expr": cluster_expr,
        "causal_adjacency": np.load(mode_dir / "step2" / "causal_adjacency.npy"),
        "dist_matrix": np.load(mode_dir / "geometry" / "distance.npy"),
        "spatial_adjacency": np.load(mode_dir / "geometry" / f"{spatial_source}.npy"),
    }


def _write_mode_outputs(mode_out: Path, result: dict[str, Any]) -> None:
    mode_out.mkdir(parents=True, exist_ok=True)
    result["lr_flow_edges"].to_csv(mode_out / "lr_flow_edges.csv", index=False)
    np.save(mode_out / "flow_matrix.npy", result["flow_matrix"])
    result["pathway_summary"].to_csv(mode_out / "pathway_summary.csv", index=False)
    (mode_out / "direction_consistency.json").write_text(
        json.dumps(result["direction_consistency"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (mode_out / "baseline_comparison.json").write_text(
        json.dumps(result["baseline_comparison"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _plot_reproduction(output_dir: Path, summary: pd.DataFrame, pathway_all: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    colors = {"hyperbolic": "#0072B2", "euclidean": "#D55E00"}
    modes = summary["mode"].tolist() if not summary.empty else []

    ax = axes[0, 0]
    x = np.arange(len(modes))
    width = 0.35
    ax.bar(
        x - width / 2,
        summary["direction_consistency_rate"] if not summary.empty else [],
        width=width,
        color=[colors.get(mode, "#8C8C8C") for mode in modes],
        alpha=0.65,
        label="Unweighted",
    )
    ax.bar(
        x + width / 2,
        summary["weighted_direction_consistency_rate"] if not summary.empty else [],
        width=width,
        color=[colors.get(mode, "#8C8C8C") for mode in modes],
        label="Weighted",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Direction consistency")
    ax.set_title("A. Causal-direction agreement")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    status_cols = ["forward_count", "reverse_count", "ambiguous_count", "unresolved_count"]
    status_colors = ["#009E73", "#CC79A7", "#E69F00", "#8C8C8C"]
    bottom = np.zeros(len(modes))
    for col, color in zip(status_cols, status_colors):
        values = summary[col].to_numpy(dtype=float) if not summary.empty else []
        ax.bar(modes, values, bottom=bottom, color=color, label=col.replace("_count", ""))
        bottom += values
    ax.set_ylabel("LR flow edges")
    ax.set_title("B. Direction-status composition")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    if not pathway_all.empty:
        top = (
            pathway_all.groupby(["mode", "pathway"], as_index=False)["total_flow"]
            .sum()
            .sort_values("total_flow", ascending=False)
            .head(8)
        )
        labels = [f"{row.mode}:{row.pathway}" for row in top.itertuples()]
        ax.barh(np.arange(len(top)), top["total_flow"], color=[colors.get(mode, "#8C8C8C") for mode in top["mode"]])
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    ax.set_xlabel("Total LR flow")
    ax.set_title("C. Top pathway flow")

    ax = axes[1, 1]
    if not summary.empty:
        y = np.arange(len(modes))
        ax.barh(y - 0.18, summary["causal_spatial_consistency"], height=0.35, color="#0072B2", label="Causal graph")
        ax.barh(y + 0.18, summary["baseline_spatial_consistency"], height=0.35, color="#D55E00", label="LR-flow baseline")
        ax.set_yticks(y)
        ax.set_yticklabels(modes)
        ax.set_xlim(0, 1.05)
    ax.set_xlabel("Spatial consistency")
    ax.set_title("D. Spatial consistency baseline")
    ax.legend(frameon=False)

    fig.suptitle("Spatial communication OT/flow reproduction on integration discovery results")
    fig.tight_layout()
    fig_path = output_dir / "spatial_communication_ot_reproduction.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    pathway_all: pd.DataFrame,
    top_edges_all: pd.DataFrame,
    args: argparse.Namespace,
    warnings: list[str],
) -> Path:
    lines = [
        "# Spatial Communication OT/Flow Reproduction",
        "",
        "## Scope",
        "",
        "- Input: precomputed `results/integration/discovery/{hyperbolic,euclidean}` outputs.",
        f"- Spatial constraint: `{args.spatial_source}.npy` used as compatibility input because older integration runs did not persist raw `spatial_adjacency.npy` separately.",
        "- Method: HyperSCA distance-constrained LR communication flow sidecar; this is an OT-style proxy and not a full COMMOT run.",
        "- Interpretation: direction consistency and communication-priority comparison only; not causal proof.",
        "",
        "## Mode Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No mode summary generated._",
        "",
        "## Top Pathways",
        "",
        pathway_all.sort_values(["mode", "total_flow"], ascending=[True, False]).head(20).to_markdown(index=False)
        if not pathway_all.empty
        else "_No pathway rows generated._",
        "",
        "## Top LR Flow Edges",
        "",
        top_edges_all[
            [
                "mode",
                "rank_in_mode",
                "pathway",
                "source_node",
                "target_node",
                "ligand",
                "receptor",
                "tf",
                "target_gene",
                "flow_score",
                "direction_status",
                "missing_genes",
            ]
        ]
        .head(30)
        .to_markdown(index=False)
        if not top_edges_all.empty
        else "_No edge rows generated._",
        "",
        "## Warnings",
        "",
        "\n".join(f"- {item}" for item in warnings) if warnings else "- None",
        "",
    ]
    path = output_dir / "spatial_communication_ot_reproduction_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _relative_artifact_list(output_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(output_dir)).replace("\\", "/") for path in output_dir.rglob("*") if path.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
