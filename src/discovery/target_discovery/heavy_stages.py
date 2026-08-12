"""Stage wrappers for geometry, causal, perturbation, scoring, niche, and reporting."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.discovery.target_discovery.admission import build_module_admission_status
from src.discovery.target_discovery.geometry import blend_adjacencies, compute_geometry
from src.discovery.target_discovery.scoring import (
    compare_modes,
    ranking_policy,
    retain_hubs_and_combos,
    score_candidates,
)
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


class GeometryComparisonStage:
    name = "geometry_comparison"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        expr = inputs["cluster_expression"]
        labels = inputs["node_labels"]
        spatial_adj = inputs["spatial_adjacency"]
        geom_results: dict[str, dict] = {}
        blended: dict[str, Any] = {}
        for mode in context.config.geometry.modes:
            geom = compute_geometry(expr, labels, mode=mode, k=context.config.geometry.geometry_k)
            geom_results[mode] = geom
            blended_adj = blend_adjacencies(spatial_adj, geom["adjacency"], context.config.geometry.geometry_blend)
            blended[mode] = blended_adj
            context.writer.write_table(
                "embedding.csv",
                pd.DataFrame(geom["embedding"], index=labels, columns=["d1", "d2"]).reset_index().rename(columns={"index": "node"}),
                section=f"geometry/{mode}",
            )
            context.writer.write_array("distance.npy", geom["dist_matrix"], section=f"geometry/{mode}")
            context.writer.write_array("adjacency.npy", geom["adjacency"], section=f"geometry/{mode}")
            context.writer.write_array("blended.npy", blended_adj, section=f"geometry/{mode}")
            context.writer.write_json("metrics.json", geom["metrics"], section=f"geometry/{mode}")
        return {"geometry_results": geom_results, "blended_adjacencies": blended}


class EvidenceScoringStage:
    name = "evidence_scoring"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        causal = inputs["causal_results"]
        perturb = inputs["perturbation_results"]
        score_profile = context.config.score_profile
        ranking = score_candidates(
            inputs["candidate_pool"],
            causal["hyperbolic"],
            causal["euclidean"],
            perturb["hyperbolic"],
            perturb["euclidean"],
            inputs["cluster_expression"],
            score_profile=score_profile,
        )
        hubs, combos = retain_hubs_and_combos(ranking, perturb["hyperbolic"])
        comparison = compare_modes(
            inputs["geometry_results"]["hyperbolic"],
            inputs["geometry_results"]["euclidean"],
            causal["hyperbolic"],
            causal["euclidean"],
            perturb["hyperbolic"],
            perturb["euclidean"],
            ranking,
        )
        module_admission = build_module_admission_status(
            ranking=ranking,
            perturbation_scores=perturb["hyperbolic"],
        )
        context.writer.write_table("target_ranking.csv", ranking, section="scoring")
        evidence_cols = [
            col
            for col in [
                "gene",
                "rank",
                "final_score",
                "ranking_basis",
                "final_score_method",
                "evidence_support_tier",
                "evidence_source_count",
                "direction_consistency",
                "neg_log10_padj",
                "mean_abs_lfc",
                "rank_rationale",
                "s_causal",
                "s_spatial",
                "s_consistency",
                "s_actionability",
                "s_niche",
            ]
            if col in ranking
        ]
        context.writer.write_table("evidence_matrix.csv", ranking[evidence_cols], section="scoring")
        context.writer.write_table("module_admission.csv", module_admission, section="scoring")
        context.writer.write_json("ranking_policy.json", ranking_policy(score_profile), section="scoring")
        context.writer.write_table("hub_targets_retained.csv", hubs, section="scoring")
        context.writer.write_table("spatiotemporal_regulatory_combos.csv", combos, section="scoring")
        context.writer.write_json("mode_comparison.json", comparison, section="scoring")
        return {
            "target_ranking": ranking,
            "retained_hubs": hubs,
            "retained_combos": combos,
            "mode_comparison": comparison,
            "module_admission": module_admission,
        }


class ReportAndFigureStage:
    name = "report_and_figure"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        from src.discovery.target_discovery.figures import generate_figure_pack
        from src.discovery.target_discovery.reporting import build_migration_notes, build_target_discovery_report

        figure_paths = [] if context.config.skip_figures else generate_figure_pack(context.writer, inputs)
        report = build_target_discovery_report(context, inputs)
        migration = build_migration_notes(context.writer.run_dir)
        report_path = context.writer.write_markdown("target_discovery_report.md", report, section="reports")
        migration_path = context.writer.write_markdown("migration_notes.md", migration, section="reports")
        return {"figure_paths": figure_paths, "target_discovery_report": report_path, "migration_notes": migration_path}
