"""Unified niche inventory and target mapping helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.discovery.target_discovery.config import DiscoveryPaths
from src.discovery.target_discovery.constants import TYPE_MAPPING


def collect_available_data_inventory(paths: DiscoveryPaths, writer=None) -> dict[str, Any]:
    inventory = {
        "icb_reference": {"path": str(paths.reference_manifest_path), "exists": paths.reference_manifest_path.exists()},
        "icb_h5ad": {"path": str(paths.icb_h5ad_path), "exists": paths.icb_h5ad_path.exists()},
        "st_metadata": {"path": str(paths.st_dir), "n_tables": len(list(paths.st_dir.glob("STmetadata_*.csv"))) if paths.st_dir.exists() else 0},
    }
    if writer is not None:
        writer.write_json("available_data_inventory.json", inventory, section="niche")
    return inventory


def build_unified_niche_definition(
    paths: DiscoveryPaths,
    writer,
    n_clusters: int | None = None,
    k_min: int = 8,
    k_max: int = 18,
    fallback_node_labels: list[str] | None = None,
    platform: str = "all",
) -> dict[str, Any]:
    del paths, n_clusters, k_min, k_max, platform
    labels = list(fallback_node_labels or [])
    rows = []
    for niche_id, label in enumerate(labels):
        rows.append(
            {
                "niche_id": niche_id,
                "niche_label": TYPE_MAPPING.get(label, label),
                "dominant_celltype": label,
                "dominant_broad_type": TYPE_MAPPING.get(label, label),
            }
        )
    definition = pd.DataFrame(rows)
    writer.write_table("unified_niche_definition.csv", definition, section="niche")
    return {"definition": definition, "selected_k": len(definition), "source": "fallback_node_labels"}


def map_targets_to_unified_niches(
    writer,
    ranking: pd.DataFrame,
    cluster_expr: pd.DataFrame,
    node_labels: list[str],
    niche_pack: dict[str, Any],
    combos: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    del niche_pack
    node_to_type = {label: TYPE_MAPPING.get(label, label) for label in node_labels}
    target_rows = []
    for _, row in ranking.head(50).iterrows():
        gene = str(row.get("gene", ""))
        if gene not in cluster_expr.columns:
            continue
        top_node = cluster_expr[gene].astype(float).idxmax()
        target_rows.append(
            {
                "gene": gene,
                "top_node": top_node,
                "broad_type": node_to_type.get(top_node, top_node),
                "rank": row.get("rank"),
                "final_score": row.get("final_score"),
            }
        )
    target_niche = pd.DataFrame(target_rows)
    combo_niche = combos.copy()
    writer.write_table("target_to_unified_niche.csv", target_niche, section="niche")
    writer.write_table("combo_to_unified_niche.csv", combo_niche, section="niche")
    return {"target_niche": target_niche, "combo_niche": combo_niche}


class UnifiedNicheStage:
    name = "unified_niche"

    def run(self, context, inputs):
        inventory = collect_available_data_inventory(context.config.paths, context.writer)
        niche_pack = build_unified_niche_definition(
            paths=context.config.paths,
            writer=context.writer,
            n_clusters=None,
            k_min=8,
            k_max=18,
            fallback_node_labels=inputs["node_labels"],
            platform=context.config.platform,
        )
        niche_map = map_targets_to_unified_niches(
            writer=context.writer,
            ranking=inputs["target_ranking"],
            cluster_expr=inputs["cluster_expression"],
            node_labels=inputs["node_labels"],
            niche_pack=niche_pack,
            combos=inputs["retained_combos"],
        )
        return {
            "available_data_inventory": inventory,
            "niche_pack": niche_pack,
            "target_niche": niche_map.get("target_niche"),
            "combo_niche": niche_map.get("combo_niche"),
        }
