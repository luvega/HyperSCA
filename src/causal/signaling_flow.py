"""多层信号流推断 (Multilayer Signaling Flow)

构建 4 层信号传导链路:
    Ligand → Receptor → TF → Target

基于因果图边 + 配受体先验数据库 + 表达数据。

参考实现（adapter 模式，不直接 import）:
    references/flowsig/ 中 GEM 聚合策略
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# =========================================================================
# CRC TME 配受体先验数据库（内建精选，覆盖路线图中的关键通路）
# =========================================================================

CRC_LR_DATABASE = [
    # CAF → TAM
    {"ligand": "POSTN", "receptor": "ITGAV", "tf": "SRC", "target": "CD163",
     "source_type": "CAF", "target_type": "TAM", "pathway": "Integrin-FAK"},
    {"ligand": "POSTN", "receptor": "ITGB5", "tf": "FAK", "target": "MRC1",
     "source_type": "CAF", "target_type": "TAM", "pathway": "Integrin-FAK"},
    {"ligand": "MFAP2", "receptor": "ITGA5", "tf": "SRC", "target": "CD163",
     "source_type": "CAF", "target_type": "TAM", "pathway": "Integrin-FAK"},
    {"ligand": "MFAP2", "receptor": "ITGB1", "tf": "FAK", "target": "MRC1",
     "source_type": "CAF", "target_type": "TAM", "pathway": "Integrin-FAK"},

    # CAF → Treg
    {"ligand": "INHBA", "receptor": "ACVR1B", "tf": "SMAD2", "target": "FOXP3",
     "source_type": "CAF", "target_type": "Treg", "pathway": "Activin-SMAD"},
    {"ligand": "INHBA", "receptor": "ACVR2A", "tf": "SMAD3", "target": "FOXP3",
     "source_type": "CAF", "target_type": "Treg", "pathway": "Activin-SMAD"},

    # TAM → CD8T (immunosuppression)
    {"ligand": "IL10", "receptor": "IL10RA", "tf": "STAT3", "target": "PDCD1",
     "source_type": "TAM", "target_type": "CD8T", "pathway": "IL10-STAT3"},
    {"ligand": "TGFB1", "receptor": "TGFBR1", "tf": "SMAD3", "target": "HAVCR2",
     "source_type": "TAM", "target_type": "CD8T", "pathway": "TGFb-SMAD"},
    {"ligand": "TGFB1", "receptor": "TGFBR2", "tf": "SMAD2", "target": "LAG3",
     "source_type": "TAM", "target_type": "CD8T", "pathway": "TGFb-SMAD"},

    # Epithelial → TAM (tumor-derived)
    {"ligand": "CSF1", "receptor": "CSF1R", "tf": "SPI1", "target": "CD163",
     "source_type": "Epithelial", "target_type": "TAM", "pathway": "CSF1-SPI1"},

    # CAF → Epithelial (EMT promotion)
    {"ligand": "HGF", "receptor": "MET", "tf": "ETS1", "target": "VIM",
     "source_type": "CAF", "target_type": "Epithelial", "pathway": "HGF-MET-EMT"},
]


def infer_signaling_flow(
    causal_graph_adj: np.ndarray,
    node_labels: list[str],
    expression_data: Optional[pd.DataFrame] = None,
    type_mapping: Optional[dict[str, str]] = None,
    lr_database: Optional[list[dict]] = None,
    min_expression: float = 0.1,
) -> list[dict]:
    """推断多层信号流

    基于因果图的边，结合配受体先验数据库和表达水平，
    构建 Ligand → Receptor → TF → Target 的完整信号链路。

    Parameters
    ----------
    causal_graph_adj : (K, K) 因果邻接矩阵
    node_labels : 节点名
    expression_data : (K, G) 聚合表达矩阵（行=节点，列=基因）
    type_mapping : {node_label: cell_type}
    lr_database : 配受体数据库
    min_expression : 表达量阈值

    Returns
    -------
    list[dict] 信号流边列表，每项包含:
        source_layer, source, target_layer, target, weight,
        causal_edge, pathway, evidence_score
    """
    if lr_database is None:
        lr_database = CRC_LR_DATABASE

    if type_mapping is None:
        type_mapping = {label: label for label in node_labels}

    # 反向映射
    type_to_nodes: dict[str, list[str]] = {}
    for label, ctype in type_mapping.items():
        type_to_nodes.setdefault(ctype, []).append(label)

    flow_edges: list[dict] = []

    for lr_entry in lr_database:
        src_type = lr_entry["source_type"]
        tgt_type = lr_entry["target_type"]

        src_nodes = type_to_nodes.get(src_type, [])
        tgt_nodes = type_to_nodes.get(tgt_type, [])

        if not src_nodes or not tgt_nodes:
            continue

        # 检查因果图中是否存在该方向的边
        has_causal_edge = False
        max_edge_weight = 0.0
        for s in src_nodes:
            for t in tgt_nodes:
                s_idx = node_labels.index(s)
                t_idx = node_labels.index(t)
                if causal_graph_adj[s_idx, t_idx] > 0:
                    has_causal_edge = True
                    max_edge_weight = max(
                        max_edge_weight, causal_graph_adj[s_idx, t_idx]
                    )

        if not has_causal_edge:
            continue

        # 检查配体/受体表达水平
        ligand = lr_entry["ligand"]
        receptor = lr_entry["receptor"]
        tf = lr_entry["tf"]
        target = lr_entry["target"]

        ligand_expr = _get_expression(expression_data, src_nodes, ligand)
        receptor_expr = _get_expression(expression_data, tgt_nodes, receptor)
        tf_expr = _get_expression(expression_data, tgt_nodes, tf)
        target_expr = _get_expression(expression_data, tgt_nodes, target)

        # 综合证据得分
        evidence_score = max_edge_weight
        if expression_data is not None:
            expr_score = np.mean([
                1.0 if ligand_expr > min_expression else 0.3,
                1.0 if receptor_expr > min_expression else 0.3,
                1.0 if tf_expr > min_expression else 0.3,
                1.0 if target_expr > min_expression else 0.3,
            ])
            evidence_score *= expr_score

        # 构建 4 层流边
        flow_edges.append({
            "source_layer": 0, "source": ligand,
            "target_layer": 1, "target": receptor,
            "weight": float(evidence_score),
            "causal_edge": f"{src_type}→{tgt_type}",
            "pathway": lr_entry.get("pathway", ""),
            "ligand_expr": float(ligand_expr),
            "receptor_expr": float(receptor_expr),
        })
        flow_edges.append({
            "source_layer": 1, "source": receptor,
            "target_layer": 2, "target": tf,
            "weight": float(evidence_score * 0.8),
            "causal_edge": f"{src_type}→{tgt_type}",
            "pathway": lr_entry.get("pathway", ""),
            "tf_expr": float(tf_expr),
        })
        flow_edges.append({
            "source_layer": 2, "source": tf,
            "target_layer": 3, "target": target,
            "weight": float(evidence_score * 0.6),
            "causal_edge": f"{src_type}→{tgt_type}",
            "pathway": lr_entry.get("pathway", ""),
            "target_expr": float(target_expr),
        })

    return flow_edges


def _get_expression(
    expression_data: Optional[pd.DataFrame],
    node_labels: list[str],
    gene: str,
) -> float:
    """安全获取基因表达水平"""
    if expression_data is None:
        return 1.0  # 无表达数据时默认有表达
    if gene not in expression_data.columns:
        return 0.0
    # 取这些节点中的最大表达
    vals = []
    for label in node_labels:
        if label in expression_data.index:
            vals.append(float(expression_data.loc[label, gene]))
    return max(vals) if vals else 0.0


def summarize_signaling_flows(flow_edges: list[dict]) -> dict:
    """汇总信号流统计

    Returns
    -------
    dict with: n_complete_flows, n_partial_flows, pathways, ...
    """
    # 统计完整链路（4 层都有）
    pathway_edges: dict[str, list[dict]] = {}
    for e in flow_edges:
        key = e.get("pathway", "unknown")
        pathway_edges.setdefault(key, []).append(e)

    complete = 0
    partial = 0
    pathway_summaries = []

    for pw, edges in pathway_edges.items():
        layers_covered = set(e["source_layer"] for e in edges) | set(
            e["target_layer"] for e in edges
        )
        is_complete = layers_covered == {0, 1, 2, 3}
        if is_complete:
            complete += 1
        else:
            partial += 1
        avg_weight = np.mean([e["weight"] for e in edges])
        pathway_summaries.append({
            "pathway": pw,
            "n_edges": len(edges),
            "complete": is_complete,
            "avg_weight": float(avg_weight),
            "layers_covered": sorted(layers_covered),
        })

    return {
        "n_total_flow_edges": len(flow_edges),
        "n_pathways": len(pathway_edges),
        "n_complete_flows": complete,
        "n_partial_flows": partial,
        "flow_completeness": complete / max(complete + partial, 1),
        "pathways": pathway_summaries,
    }
