"""Lightweight target discovery stages with synthetic-testable behavior."""
from __future__ import annotations

from typing import Any, Mapping

from src.discovery.target_discovery.candidates import aggregate_candidate_pool
from src.discovery.target_discovery.constants import CELLTYPES
from src.discovery.target_discovery.expression import assemble_cluster_expression, read_normalized_count_tables
from src.discovery.target_discovery.loaders import read_icb_deg_tables, read_ifng_tables, read_neu_deg_tables
from src.discovery.target_discovery.spatial import build_spatial_adjacency_from_tables, read_st_metadata_tables
from src.discovery.target_discovery.stage import TargetDiscoveryRunContext


class CandidateDiscoveryStage:
    name = "candidate_discovery"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        del inputs
        paths = context.config.paths
        pool = aggregate_candidate_pool(
            read_neu_deg_tables(paths.neu_dir),
            read_icb_deg_tables(paths.icb_dir),
            read_ifng_tables(paths.ifng_dir, ()),
        )
        context.writer.write_table("candidate_pool.csv", pool, section="candidates")
        return {"candidate_pool": pool}


class ExpressionAssemblyStage:
    name = "expression_assembly"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        del inputs
        tables = read_normalized_count_tables(context.config.paths.neu_dir, CELLTYPES)
        expr, labels = assemble_cluster_expression(tables)
        context.writer.write_table("cluster_expression.csv", expr.reset_index().rename(columns={"index": "celltype"}), section="expression")
        context.writer.write_json("node_labels.json", {"node_labels": labels}, section="expression")
        return {"cluster_expression": expr, "node_labels": labels}


class SpatialContextStage:
    name = "spatial_context"

    def run(self, context: TargetDiscoveryRunContext, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        labels = list(inputs["node_labels"])
        tables = read_st_metadata_tables(context.config.paths.st_dir)
        adj = build_spatial_adjacency_from_tables(tables, labels)
        context.writer.write_array("spatial_adjacency.npy", adj, section="spatial")
        return {"spatial_adjacency": adj}
