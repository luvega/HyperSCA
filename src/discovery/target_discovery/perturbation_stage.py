"""Step3 perturbation screen wrapper for target discovery."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.discovery.target_discovery.constants import ANCHOR_GENES
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


def select_perturbation_targets(candidate_pool: pd.DataFrame, cluster_expr: pd.DataFrame, max_perturb: int) -> list[str]:
    gene_upper = {str(col).upper(): str(col) for col in cluster_expr.columns}
    available = [str(gene) for gene in candidate_pool["gene"] if str(gene).upper() in gene_upper]
    targets: list[str] = []
    for anchor in ANCHOR_GENES:
        if anchor in available and anchor not in targets:
            targets.append(anchor)
    for gene in available:
        if gene not in targets and len(targets) < max_perturb:
            targets.append(gene)
    return targets


def run_perturbation_screen(step2_results: dict, target_genes: list[str], writer, section: str) -> dict:
    from scipy.spatial.distance import cdist
    from sklearn.manifold import MDS

    from src.evaluation.cf_metrics import evaluate_counterfactual
    from src.evaluation.spatial_metrics import evaluate_spatial_propagation
    from src.perturbation.spatial_propagation import propagate_perturbation

    cluster_expr = step2_results["cluster_expr"]
    node_labels = step2_results["node_labels"]
    type_mapping = step2_results["type_mapping"]
    flow_edges = step2_results["flow_edges"]
    causal_adj = step2_results["causal_graph"].adjacency
    cluster_adj = step2_results["cluster_adj"]
    n_nodes = len(node_labels)

    dist_mat = np.maximum(1.0 - cluster_adj, (1.0 - cluster_adj).T)
    np.fill_diagonal(dist_mat, 0)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto").fit_transform(dist_mat)

    gene_upper = {str(col).upper(): str(col) for col in cluster_expr.columns}
    results = {}

    for target_gene in target_genes:
        gene_key = target_gene.upper()
        if gene_key not in gene_upper:
            continue
        col = gene_upper[gene_key]

        observed = cluster_expr.copy()
        counterfactual = observed.copy()
        counterfactual[col] = observed[col] * 0.5

        for edge in flow_edges:
            if edge.get("source_layer") != 0 or str(edge.get("source", "")).upper() != gene_key:
                continue
            receptor = str(edge.get("target", "")).upper()
            if receptor not in gene_upper:
                continue
            causal_edge = str(edge.get("causal_edge", ""))
            if "->" in causal_edge:
                target_type = causal_edge.split("->")[1].strip()
            elif "\u2192" in causal_edge:
                target_type = causal_edge.split("\u2192")[1].strip()
            else:
                target_type = ""
            receptor_col = gene_upper[receptor]
            affected_rows = [idx for idx, celltype in type_mapping.items() if celltype == target_type and idx in counterfactual.index]
            if affected_rows:
                counterfactual.loc[affected_rows, receptor_col] = observed.loc[affected_rows, receptor_col] * 0.75

        try:
            from src.perturbation.target_ranking import rank_counterfactual_interaction_targets

            ranked = rank_counterfactual_interaction_targets(
                flow_edges=flow_edges,
                observed_expression=observed,
                counterfactual_expression=counterfactual,
                node_to_type={label: type_mapping.get(label, label) for label in node_labels},
                min_abs_delta=0.001,
                top_k=30,
            )
        except Exception:
            ranked = pd.DataFrame()

        delta = (counterfactual[col].values - observed[col].values).astype(float)
        abs_delta = np.abs(delta)
        source_nodes = list(np.where(abs_delta >= max(abs_delta.max() * 0.3, 1e-12))[0]) or [int(np.argmax(abs_delta))]
        propagation = propagate_perturbation(
            causal_adj=causal_adj,
            source_nodes=source_nodes,
            source_delta=delta,
            spatial_coords=coords,
            decay_length=150.0,
            max_depth=4,
            convergence_tol=0.01,
        )

        common = [c for c in counterfactual.columns if c in observed.columns]
        cf_quality = evaluate_counterfactual(
            observed=observed[common].values,
            counterfactual=counterfactual[common].values,
            gene_names=common,
            expected_directions={target_gene: -1},
        )

        spatial_quality = {}
        if propagation.get("bfs_layers"):
            try:
                effect = np.asarray(propagation.get("effect", np.zeros(n_nodes)))
                effect_mag = np.abs(effect) if effect.ndim == 1 else np.mean(np.abs(effect), axis=1)
                seed_nodes = propagation["bfs_layers"][0]["nodes"]
                source_distances = cdist(coords, coords[seed_nodes]).min(axis=1)
                spatial_quality = evaluate_spatial_propagation(
                    coords=coords,
                    effect_magnitudes=effect_mag[:n_nodes],
                    source_distances=source_distances,
                    bfs_layers=propagation["bfs_layers"],
                    causal_adj=causal_adj,
                    observed_expr=observed[col].values.astype(float),
                    counterfactual_expr=counterfactual[col].values.astype(float),
                    threshold=0.01,
                )
            except Exception:
                spatial_quality = {}

        results[target_gene] = {
            "n_ranked": len(ranked),
            "cf_quality": cf_quality,
            "spatial_quality": spatial_quality,
            "propagation": {
                "n_layers": len(propagation.get("bfs_layers", [])),
                "fit_params": propagation.get("fit_params", {}),
            },
            "ranked_targets": ranked,
        }
        if not ranked.empty:
            writer.write_table(f"targets_{target_gene}.csv", ranked, section=section)

    summary = {"targets": target_genes, "per_target": {}}
    for target_gene, result in results.items():
        summary["per_target"][target_gene] = {
            "n_ranked": result["n_ranked"],
            "cf_quality": result["cf_quality"],
            "spatial_quality": result["spatial_quality"],
            "propagation": result["propagation"],
        }
    writer.write_json("step3_metrics.json", summary, section=section)
    return results


class PerturbationScreenStage:
    name = "perturbation_screen"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        targets = select_perturbation_targets(
            inputs["candidate_pool"],
            inputs["cluster_expression"],
            context.config.max_perturb,
        )
        results = {}
        for mode, causal_result in inputs["causal_results"].items():
            results[mode] = run_perturbation_screen(
                step2_results=causal_result,
                target_genes=targets,
                writer=context.writer,
                section=f"perturbation/{mode}",
            )
        return {"perturbation_targets": targets, "perturbation_results": results}
