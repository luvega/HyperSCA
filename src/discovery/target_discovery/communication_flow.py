"""Distance-constrained LR communication sidecar for target discovery."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.causal.baseline_communication import compare_spatial_causal_advantage
from src.causal.signaling_flow import CRC_LR_DATABASE


def _as_array(value: Any, shape: tuple[int, int], default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        return np.full(shape, default, dtype=float)
    return arr


def _gene_value(cluster_expr: pd.DataFrame, node: str, gene: str) -> tuple[float, bool]:
    gene_map = {str(col).upper(): col for col in cluster_expr.columns}
    key = str(gene).upper()
    if key not in gene_map or node not in cluster_expr.index:
        return 0.0, False
    value = pd.to_numeric(pd.Series([cluster_expr.loc[node, gene_map[key]]]), errors="coerce").fillna(0.0).iloc[0]
    return float(value), True


def _expr_support(*values: float) -> float:
    scores = [1.0 if float(v) > 0 else 0.0 for v in values]
    return float(np.mean(scores)) if scores else 0.0


def _direction_status(causal_adj: np.ndarray, source_idx: int, target_idx: int) -> tuple[bool, bool, str]:
    forward = bool(causal_adj[source_idx, target_idx] > 0)
    reverse = bool(causal_adj[target_idx, source_idx] > 0)
    if forward and reverse:
        return forward, reverse, "ambiguous"
    if forward:
        return forward, reverse, "forward"
    if reverse:
        return forward, reverse, "reverse"
    return forward, reverse, "unresolved"


def summarize_direction_consistency(
    edges: pd.DataFrame,
    weight_col: str = "flow_score",
    include_pathways: bool = True,
) -> dict[str, Any]:
    """Summarize causal-direction agreement for LR flow edges."""
    statuses = {"forward", "reverse", "ambiguous", "unresolved"}
    if edges.empty or "direction_status" not in edges:
        return {
            "overall": {
                "n_edges": 0,
                "resolved_edges": 0,
                "forward_count": 0,
                "reverse_count": 0,
                "ambiguous_count": 0,
                "unresolved_count": 0,
                "direction_consistency_rate": 0.0,
                "weighted_direction_consistency_rate": 0.0,
            },
            "per_pathway": {},
        }

    frame = edges.copy()
    frame["direction_status"] = frame["direction_status"].where(frame["direction_status"].isin(statuses), "unresolved")
    weights = pd.to_numeric(frame.get(weight_col, 1.0), errors="coerce").fillna(0.0).astype(float)
    resolved = frame["direction_status"].isin(["forward", "reverse", "ambiguous"])
    forward = frame["direction_status"].eq("forward")
    resolved_count = int(resolved.sum())
    weighted_den = float(weights[resolved].sum())
    weighted_num = float(weights[forward].sum())
    overall = {
        "n_edges": int(len(frame)),
        "resolved_edges": resolved_count,
        "forward_count": int(forward.sum()),
        "reverse_count": int(frame["direction_status"].eq("reverse").sum()),
        "ambiguous_count": int(frame["direction_status"].eq("ambiguous").sum()),
        "unresolved_count": int(frame["direction_status"].eq("unresolved").sum()),
        "direction_consistency_rate": float(forward.sum() / max(resolved_count, 1)),
        "weighted_direction_consistency_rate": float(weighted_num / weighted_den) if weighted_den > 0 else 0.0,
    }

    per_pathway: dict[str, Any] = {}
    if include_pathways and "pathway" in frame:
        for pathway, group in frame.groupby("pathway"):
            per_pathway[str(pathway)] = summarize_direction_consistency(
                group,
                weight_col=weight_col,
                include_pathways=False,
            )["overall"]
    return {"overall": overall, "per_pathway": per_pathway}


def build_communication_flow(
    mode: str,
    cluster_expr: pd.DataFrame,
    spatial_adjacency: np.ndarray,
    geometry_result: Mapping[str, Any],
    causal_result: Mapping[str, Any],
    alpha: float = 0.5,
    beta: float = 0.5,
    epsilon: float = 1.0,
) -> dict[str, Any]:
    """Build a lightweight distance-constrained directed LR flow sidecar."""
    labels = list(causal_result.get("node_labels") or cluster_expr.index.astype(str).tolist())
    n_nodes = len(labels)
    shape = (n_nodes, n_nodes)
    cluster_expr = cluster_expr.copy()
    cluster_expr.index = cluster_expr.index.astype(str)
    spatial = _as_array(spatial_adjacency, shape)
    dist = _as_array(geometry_result.get("dist_matrix"), shape)
    causal_graph = causal_result.get("causal_graph")
    causal_adj = _as_array(getattr(causal_graph, "adjacency", causal_result.get("causal_adjacency")), shape)
    type_mapping = dict(causal_result.get("type_mapping", {label: label for label in labels}))

    positive_dist = dist[dist > 0]
    dist_scale = float(positive_dist.max()) if len(positive_dist) else 1.0
    norm_dist = dist / max(dist_scale, 1e-12)
    spatial_norm = np.clip(spatial, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    flow_matrix = np.zeros(shape, dtype=float)

    for entry in CRC_LR_DATABASE:
        source_nodes = [idx for idx, label in enumerate(labels) if type_mapping.get(label, label) == entry["source_type"]]
        target_nodes = [idx for idx, label in enumerate(labels) if type_mapping.get(label, label) == entry["target_type"]]
        for source_idx in source_nodes:
            for target_idx in target_nodes:
                if source_idx == target_idx:
                    continue
                source_node = labels[source_idx]
                target_node = labels[target_idx]
                ligand_expr, ligand_ok = _gene_value(cluster_expr, source_node, entry["ligand"])
                receptor_expr, receptor_ok = _gene_value(cluster_expr, target_node, entry["receptor"])
                tf_expr, tf_ok = _gene_value(cluster_expr, target_node, entry["tf"])
                target_expr, target_ok = _gene_value(cluster_expr, target_node, entry["target"])
                missing = [
                    gene for gene, ok in (
                        (entry["ligand"], ligand_ok),
                        (entry["receptor"], receptor_ok),
                        (entry["tf"], tf_ok),
                        (entry["target"], target_ok),
                    )
                    if not ok
                ]
                spatial_weight = float(spatial_norm[source_idx, target_idx])
                geometry_distance = float(norm_dist[source_idx, target_idx])
                geometry_weight = float(np.exp(-geometry_distance / max(epsilon, 1e-12)))
                cost = alpha * geometry_distance + beta * (1.0 - spatial_weight)
                blended_weight = float(np.exp(-cost / max(epsilon, 1e-12)))
                support = _expr_support(ligand_expr, receptor_expr, tf_expr, target_expr)
                flow_score = float(support * blended_weight)
                causal_forward, causal_reverse, status = _direction_status(causal_adj, source_idx, target_idx)
                flow_matrix[source_idx, target_idx] += flow_score
                rows.append(
                    {
                        "mode": mode,
                        "pathway": entry["pathway"],
                        "ligand": entry["ligand"],
                        "receptor": entry["receptor"],
                        "tf": entry["tf"],
                        "target_gene": entry["target"],
                        "source_type": entry["source_type"],
                        "target_type": entry["target_type"],
                        "source_node": source_node,
                        "target_node": target_node,
                        "ligand_expr": ligand_expr,
                        "receptor_expr": receptor_expr,
                        "tf_expr": tf_expr,
                        "target_expr": target_expr,
                        "missing_genes": ";".join(missing),
                        "spatial_weight": spatial_weight,
                        "geometry_distance": geometry_distance,
                        "geometry_weight": geometry_weight,
                        "blended_weight": blended_weight,
                        "flow_score": flow_score,
                        "normalized_flow": 0.0,
                        "causal_forward": causal_forward,
                        "causal_reverse": causal_reverse,
                        "direction_status": status,
                    }
                )

    edges = pd.DataFrame(rows)
    if not edges.empty:
        max_score = float(edges["flow_score"].max())
        if max_score > 0:
            edges["normalized_flow"] = edges["flow_score"] / max_score
    direction = summarize_direction_consistency(edges)
    pathway_summary = _summarize_pathways(edges)
    flow_binary = (flow_matrix > 0).astype(float)
    baseline = compare_spatial_causal_advantage(
        causal_adj=(causal_adj > 0).astype(float),
        baseline_adj=flow_binary,
        spatial_adj=spatial,
    )
    return {
        "lr_flow_edges": edges,
        "flow_matrix": flow_matrix,
        "pathway_summary": pathway_summary,
        "direction_consistency": direction,
        "baseline_comparison": baseline,
    }


def _summarize_pathways(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "pathway",
                "n_edges",
                "total_flow",
                "mean_flow",
                "max_flow",
                "top_source",
                "top_target",
                "causal_forward_count",
                "causal_reverse_count",
                "ambiguous_count",
                "unresolved_count",
                "direction_consistency_rate",
            ]
        )
    rows: list[dict[str, Any]] = []
    for pathway, group in edges.groupby("pathway"):
        top = group.sort_values("flow_score", ascending=False).iloc[0]
        consistency = summarize_direction_consistency(group)["overall"]
        rows.append(
            {
                "pathway": pathway,
                "n_edges": int(len(group)),
                "total_flow": float(group["flow_score"].sum()),
                "mean_flow": float(group["flow_score"].mean()),
                "max_flow": float(group["flow_score"].max()),
                "top_source": top["source_node"],
                "top_target": top["target_node"],
                "causal_forward_count": consistency["forward_count"],
                "causal_reverse_count": consistency["reverse_count"],
                "ambiguous_count": consistency["ambiguous_count"],
                "unresolved_count": consistency["unresolved_count"],
                "direction_consistency_rate": consistency["direction_consistency_rate"],
            }
        )
    return pd.DataFrame(rows)


class CommunicationFlowStage:
    name = "communication_flow"

    def run(self, context, inputs):
        results = {}
        for mode, causal_result in inputs["causal_results"].items():
            result = build_communication_flow(
                mode=mode,
                cluster_expr=inputs["cluster_expression"],
                spatial_adjacency=inputs["spatial_adjacency"],
                geometry_result=inputs["geometry_results"][mode],
                causal_result=causal_result,
            )
            section = f"communication/{mode}"
            context.writer.write_table("lr_flow_edges.csv", result["lr_flow_edges"], section=section)
            context.writer.write_array("flow_matrix.npy", result["flow_matrix"], section=section)
            context.writer.write_table("pathway_summary.csv", result["pathway_summary"], section=section)
            context.writer.write_json("direction_consistency.json", result["direction_consistency"], section=section)
            context.writer.write_json("baseline_comparison.json", result["baseline_comparison"], section=section)
            results[mode] = result
        return {"communication_results": results}
