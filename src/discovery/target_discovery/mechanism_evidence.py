"""Mechanistic LR-TF-target evidence sidecar for target discovery."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _norm01(value: float, scale: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value / scale)))


def _component_score(value: Any) -> float:
    try:
        return 1.0 if float(value) > 0 else 0.0
    except Exception:
        return 0.0


def _split_causal_edge(causal_edge: str) -> tuple[str, str]:
    if "->" in causal_edge:
        source_type, target_type = causal_edge.split("->", 1)
    elif "→" in causal_edge:
        source_type, target_type = causal_edge.split("→", 1)
    else:
        return causal_edge, ""
    return source_type, target_type


def _extract_chains(flow_edges: Sequence[Mapping[str, Any]], relaxed_mode: bool = False) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = {}
    for edge in flow_edges:
        key = (str(edge.get("pathway", "")), str(edge.get("causal_edge", "")))
        parts = keyed.setdefault(key, {"lr": [], "rtf": [], "tf_target": []})
        if edge.get("source_layer") == 0 and edge.get("target_layer") == 1:
            parts["lr"].append(edge)
        elif edge.get("source_layer") == 1 and edge.get("target_layer") == 2:
            parts["rtf"].append(edge)
        elif edge.get("source_layer") == 2 and edge.get("target_layer") == 3:
            parts["tf_target"].append(edge)

    chains: list[dict[str, Any]] = []
    for (pathway, causal_edge), parts in keyed.items():
        source_type, target_type = _split_causal_edge(causal_edge)
        rtf_by_receptor: dict[str, list[Mapping[str, Any]]] = {}
        for edge in parts["rtf"]:
            rtf_by_receptor.setdefault(str(edge.get("source", "")), []).append(edge)
        tf_target_by_tf: dict[str, list[Mapping[str, Any]]] = {}
        for edge in parts["tf_target"]:
            tf_target_by_tf.setdefault(str(edge.get("source", "")), []).append(edge)

        for lr in parts["lr"]:
            ligand = str(lr.get("source", ""))
            receptor = str(lr.get("target", ""))
            rtf_candidates = rtf_by_receptor.get(receptor, [{}])
            for rtf in rtf_candidates:
                tf = str(rtf.get("target", ""))
                tf_target_candidates = tf_target_by_tf.get(tf, [{}]) if tf else [{}]
                for tf_target in tf_target_candidates:
                    downstream = str(tf_target.get("target", ""))
                    if not (ligand or receptor or tf or downstream):
                        continue
                    chain_parts = [part for part in (lr, rtf, tf_target) if part]
                    chains.append(
                        {
                            "pathway": pathway,
                            "causal_edge": causal_edge,
                            "ligand": ligand,
                            "receptor": receptor,
                            "tf": tf,
                            "downstream_target": downstream,
                            "source_type": source_type,
                            "target_type": target_type,
                            "relaxed_mode": bool(relaxed_mode),
                            "ligand_expr": float(lr.get("ligand_expr", 0.0) or 0.0),
                            "receptor_expr": float(lr.get("receptor_expr", 0.0) or 0.0),
                            "target_expr": float(tf_target.get("target_expr", 0.0) or 0.0),
                            "edge_weight": float(
                                np.mean([float(part.get("weight", 0.0) or 0.0) for part in chain_parts])
                            ),
                        }
                    )
    return chains


def _spatial_support(gene: str, perturbation_results: Mapping[str, Any] | None) -> float:
    if not perturbation_results or gene not in perturbation_results:
        return 0.0
    spatial = perturbation_results[gene].get("spatial_quality", {})
    return float(
        0.5 * _norm01(float(spatial.get("gradient_decay_r2", 0.0) or 0.0))
        + 0.3 * _norm01(float(spatial.get("propagation_depth", 0.0) or 0.0), scale=4.0)
        + 0.2 * _norm01(float(spatial.get("moran_i_effect", 0.0) or 0.0))
    )


def build_mechanism_evidence(
    ranking: pd.DataFrame,
    causal_result: Mapping[str, Any],
    perturbation_results: Mapping[str, Any] | None = None,
    top_n: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build chain-level evidence for already-ranked candidate targets."""
    columns = [
        "target_gene",
        "pathway",
        "source_type",
        "target_type",
        "ligand",
        "receptor",
        "tf",
        "downstream_target",
        "causal_edge",
        "relaxed_mode",
        "prior_source",
        "s_lr_prior",
        "s_expr_ligand",
        "s_expr_receptor",
        "s_expr_tf",
        "s_expr_target",
        "s_causal_edge",
        "s_tf_target_prior",
        "s_spatial",
        "s_niche",
        "s_mechanism",
    ]
    if ranking.empty or "gene" not in ranking:
        return pd.DataFrame(columns=columns), {
            "n_ranked_targets": 0,
            "n_targets_with_mechanism": 0,
            "chain_coverage": 0.0,
            "complete_chain_rate": 0.0,
            "prior_sources": {},
        }

    ranked = ranking.head(top_n).copy()
    relaxed = bool(causal_result.get("flow_summary", {}).get("relaxed_mode", False))
    chains = _extract_chains(causal_result.get("flow_edges", []), relaxed_mode=relaxed)
    rows: list[dict[str, Any]] = []
    for _, target_row in ranked.iterrows():
        gene = str(target_row.get("gene", ""))
        gene_upper = gene.upper()
        for chain in chains:
            members = {
                chain["ligand"].upper(),
                chain["receptor"].upper(),
                chain["tf"].upper(),
                chain["downstream_target"].upper(),
            }
            if gene_upper not in members:
                continue
            s_expr_ligand = _component_score(chain["ligand_expr"])
            s_expr_receptor = _component_score(chain["receptor_expr"])
            s_expr_tf = 1.0 if chain["tf"] else 0.0
            s_expr_target = _component_score(chain["target_expr"])
            s_spatial = _spatial_support(gene, perturbation_results)
            s_niche = float(target_row.get("s_niche", 0.0) or 0.0)
            s_mechanism = float(
                0.20 * 1.0
                + 0.25 * np.mean([s_expr_ligand, s_expr_receptor, s_expr_tf, s_expr_target])
                + 0.25 * _norm01(chain["edge_weight"])
                + 0.15 * 1.0
                + 0.10 * s_spatial
                + 0.05 * _norm01(s_niche)
            )
            rows.append(
                {
                    "target_gene": gene,
                    "pathway": chain["pathway"],
                    "source_type": chain["source_type"],
                    "target_type": chain["target_type"],
                    "ligand": chain["ligand"],
                    "receptor": chain["receptor"],
                    "tf": chain["tf"],
                    "downstream_target": chain["downstream_target"],
                    "causal_edge": chain["causal_edge"],
                    "relaxed_mode": chain["relaxed_mode"],
                    "prior_source": "curated_builtin",
                    "s_lr_prior": 1.0,
                    "s_expr_ligand": s_expr_ligand,
                    "s_expr_receptor": s_expr_receptor,
                    "s_expr_tf": s_expr_tf,
                    "s_expr_target": s_expr_target,
                    "s_causal_edge": _norm01(chain["edge_weight"]),
                    "s_tf_target_prior": 1.0 if chain["tf"] and chain["downstream_target"] else 0.0,
                    "s_spatial": s_spatial,
                    "s_niche": s_niche,
                    "s_mechanism": s_mechanism,
                }
            )

    matrix = pd.DataFrame(rows, columns=columns)
    targets_with_mechanism = set(matrix["target_gene"].astype(str)) if not matrix.empty else set()
    summary = {
        "n_ranked_targets": int(len(ranked)),
        "n_targets_with_mechanism": int(len(targets_with_mechanism)),
        "n_mechanism_chains": int(len(matrix)),
        "chain_coverage": float(len(targets_with_mechanism) / max(len(ranked), 1)),
        "complete_chain_rate": float(1.0 if len(matrix) else 0.0),
        "prior_sources": matrix["prior_source"].value_counts().to_dict() if not matrix.empty else {},
        "relaxed_mode_fraction": float(matrix["relaxed_mode"].mean()) if not matrix.empty else 0.0,
    }
    return matrix, summary


def append_mechanism_scores(ranking: pd.DataFrame, mechanism_matrix: pd.DataFrame) -> pd.DataFrame:
    """Append explanatory mechanism scores without changing rank or final_score."""
    out = ranking.copy()
    if out.empty or mechanism_matrix.empty or "gene" not in out:
        out["s_mechanism"] = 0.0
        return out
    scores = mechanism_matrix.groupby("target_gene")["s_mechanism"].max().to_dict()
    out["s_mechanism"] = out["gene"].astype(str).map(scores).fillna(0.0)
    return out


def build_mechanism_evidence_report(summary: Mapping[str, Any]) -> str:
    """Render a conservative Markdown mechanism evidence report."""
    return "\n".join(
        [
            "# LR-TF-target Mechanism Evidence",
            "",
            f"- Ranked targets reviewed: {summary.get('n_ranked_targets', 0)}",
            f"- Targets with mechanism chains: {summary.get('n_targets_with_mechanism', 0)}",
            f"- Mechanism chain coverage: {summary.get('chain_coverage', 0.0):.3f}",
            "",
            "Curated LR/TF-target resources are used only to annotate candidates already nominated by the data-driven layer. These chains are mechanism hypotheses and computational priorities, not validated treatment conclusions.",
            "",
        ]
    )


class MechanismEvidenceStage:
    name = "mechanism_evidence"

    def run(self, context, inputs):
        ranking = inputs["target_ranking"]
        matrix, summary = build_mechanism_evidence(
            ranking=ranking,
            causal_result=inputs["causal_results"]["hyperbolic"],
            perturbation_results=inputs["perturbation_results"].get("hyperbolic", {}),
            top_n=20,
        )
        ranking_with_mechanism = append_mechanism_scores(ranking, matrix)
        report = build_mechanism_evidence_report(summary)
        context.writer.write_table("mechanism_evidence_matrix.csv", matrix, section="scoring")
        context.writer.write_table("target_ranking_with_mechanism.csv", ranking_with_mechanism, section="scoring")
        context.writer.write_json("mechanism_summary.json", summary, section="scoring")
        report_path = context.writer.write_markdown("mechanism_evidence_report.md", report, section="reports")
        return {
            "mechanism_evidence_matrix": matrix,
            "mechanism_summary": summary,
            "mechanism_evidence_report": report_path,
            "target_ranking_with_mechanism": ranking_with_mechanism,
        }
