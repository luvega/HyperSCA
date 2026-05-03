"""Step2 causal discovery wrapper for target discovery."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.discovery.target_discovery.constants import PRIOR_AXES, TYPE_MAPPING
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


def run_causal_discovery(
    cluster_expr: pd.DataFrame,
    cluster_adj: np.ndarray,
    node_labels: list[str],
    writer,
    section: str,
    device: str,
) -> dict:
    from src.causal.causal_graph import CausalCellGraph, load_known_axes
    from src.causal.cmi_pruning import bootstrap_causal_discovery, threshold_pruning
    from src.causal.disentangle import train_disentangle
    from src.causal.signaling_flow import infer_signaling_flow, summarize_signaling_flows
    from src.evaluation.causal_metrics import evaluate_causal

    n_nodes = len(node_labels)
    expr_np = cluster_expr.values.astype(np.float32)
    type_mapping = {label: TYPE_MAPPING.get(label, label) for label in node_labels}

    rows, cols = np.where(cluster_adj > 0)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(cluster_adj[rows, cols], dtype=torch.float32)
    x = torch.tensor(expr_np, dtype=torch.float32)
    disentangled = train_disentangle(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        z_dim=16,
        hidden_dims=[256, 128],
        epochs=200,
        lr=1e-3,
        hsic_alpha=1.0,
        device=device,
        verbose=True,
    )
    z_int, z_ext = disentangled["z_int"], disentangled["z_ext"]

    freq = bootstrap_causal_discovery(
        data=expr_np.T,
        n_bootstraps=100,
        alpha=0.05,
        max_cond_set=3,
        seed=42,
        verbose=True,
    )
    adjacency, pruned_freq = threshold_pruning(freq, threshold=0.5)

    for source_type, target_type, prior_weight in PRIOR_AXES:
        source_nodes = [i for i, label in enumerate(node_labels) if type_mapping.get(label) == source_type]
        target_nodes = [i for i, label in enumerate(node_labels) if type_mapping.get(label) == target_type]
        for source in source_nodes:
            for target in target_nodes:
                if adjacency[source, target] == 0 and adjacency[target, source] == 0:
                    adjacency[source, target] = prior_weight

    causal_graph = CausalCellGraph(adjacency=adjacency, node_labels=node_labels, bootstrap_freq=pruned_freq)

    rng = np.random.default_rng(42)
    n_samples = max(200, n_nodes * 10)
    idx = rng.choice(n_nodes, size=n_samples, replace=True)
    validation_data = {
        label: z_ext[idx, min(i, z_ext.shape[1] - 1)] + rng.normal(0, 0.01, n_samples)
        for i, label in enumerate(node_labels)
    }
    falsification = causal_graph.validate_structure(pd.DataFrame(validation_data))

    flow_edges = infer_signaling_flow(
        causal_graph_adj=causal_graph.adjacency,
        node_labels=node_labels,
        expression_data=cluster_expr,
        type_mapping=type_mapping,
    )
    flow_summary = summarize_signaling_flows(flow_edges)
    if flow_summary["n_total_flow_edges"] == 0:
        relaxed_adj = np.maximum(causal_graph.adjacency, causal_graph.adjacency.T)
        relaxed_edges = infer_signaling_flow(
            causal_graph_adj=relaxed_adj,
            node_labels=node_labels,
            expression_data=cluster_expr,
            type_mapping=type_mapping,
        )
        relaxed_summary = summarize_signaling_flows(relaxed_edges)
        if relaxed_summary["n_total_flow_edges"] > 0:
            flow_edges, flow_summary = relaxed_edges, relaxed_summary
            flow_summary["relaxed_mode"] = True

    known_axes = load_known_axes(None)
    axis_results = causal_graph.evaluate_known_axes(known_axes=known_axes, type_mapping=type_mapping)
    metrics = evaluate_causal(
        adjacency=causal_graph.adjacency,
        bootstrap_freq=causal_graph.bootstrap_freq,
        z_int=z_int,
        z_ext=z_ext,
        labels=np.arange(n_nodes),
        cluster_adj=(cluster_adj > 0).astype(float),
        known_axis_results=axis_results,
        falsification_results=falsification,
        signaling_flow_summary=flow_summary,
    )

    writer.write_array("causal_adjacency.npy", causal_graph.adjacency, section=section)
    writer.write_array("bootstrap_freq.npy", freq, section=section)
    writer.write_array("z_int.npy", z_int, section=section)
    writer.write_array("z_ext.npy", z_ext, section=section)
    graph_path = writer.section_dir(section) / "causal_graph.graphml"
    causal_graph.to_graphml(graph_path)
    writer._record(graph_path)
    writer.write_json("node_info.json", {"node_labels": node_labels, "type_mapping": type_mapping}, section=section)
    writer.write_json("step2_metrics.json", metrics, section=section)
    writer.write_json("axis_results.json", axis_results, section=section)
    writer.write_json("flow_summary.json", flow_summary, section=section)
    writer.write_json("falsification.json", falsification, section=section)
    writer.write_json("losses.json", disentangled["losses"], section=section)
    writer.write_json("flow_edges.json", {"flow_edges": flow_edges}, section=section)
    writer.write_table("cluster_expr.csv", cluster_expr.reset_index().rename(columns={"index": "celltype"}), section=section)

    import networkx as nx

    graph = nx.from_numpy_array(causal_graph.adjacency, create_using=nx.DiGraph)
    betweenness = nx.betweenness_centrality(graph)
    betweenness_by_label = {node_labels[i]: betweenness.get(i, 0.0) for i in range(n_nodes)}

    return {
        "causal_graph": causal_graph,
        "flow_edges": flow_edges,
        "flow_summary": flow_summary,
        "metrics": metrics,
        "axis_results": axis_results,
        "falsification": falsification,
        "type_mapping": type_mapping,
        "node_labels": node_labels,
        "cluster_expr": cluster_expr,
        "cluster_adj": cluster_adj,
        "z_int": z_int,
        "z_ext": z_ext,
        "betweenness": betweenness_by_label,
        "disentangle_losses": disentangled["losses"],
    }


class CausalDiscoveryStage:
    name = "causal_discovery"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        results = {}
        for mode, adjacency in inputs["blended_adjacencies"].items():
            results[mode] = run_causal_discovery(
                cluster_expr=inputs["cluster_expression"],
                cluster_adj=adjacency,
                node_labels=inputs["node_labels"],
                writer=context.writer,
                section=f"causal/{mode}",
                device=context.config.device,
            )
        return {"causal_results": results}
