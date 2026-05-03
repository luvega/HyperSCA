"""Evidence scoring and mode comparison for target discovery."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.discovery.target_discovery.constants import ANCHOR_GENES, SCORE_WEIGHTS
from src.discovery.target_discovery.utils import minmax


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float).values


def _split_celltypes(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def score_candidates(
    candidate_pool: pd.DataFrame,
    step2_hyp: dict,
    step2_euc: dict,
    step3_hyp: dict,
    step3_euc: dict,
    cluster_expr: pd.DataFrame,
) -> pd.DataFrame:
    del step2_euc, step3_euc
    pool = candidate_pool.copy()
    if pool.empty:
        return pool

    bc_hyp = step2_hyp.get("betweenness", {})
    node_labels = step2_hyp.get("node_labels", [])
    gene_cols = {str(col).upper() for col in cluster_expr.columns}
    flow_edges = step2_hyp.get("flow_edges", [])

    causal_scores: list[float] = []
    spatial_scores: list[float] = []
    action_scores: list[float] = []
    niche_scores: list[float] = []

    for _, row in pool.iterrows():
        gene = str(row.get("gene", ""))
        gene_upper = gene.upper()

        assoc_cts = {ct for ct in _split_celltypes(row.get("celltypes_neu", "")) if ct in node_labels}
        if not assoc_cts and gene_upper in gene_cols:
            assoc_cts = set(node_labels)
        bc_vals = [float(bc_hyp.get(ct, 0.0)) for ct in assoc_cts]
        s_causal = max(bc_vals) if bc_vals else 0.0
        if gene in step3_hyp:
            s_causal += 0.2 * float(step3_hyp[gene].get("n_ranked", 0)) / 30.0
        causal_scores.append(s_causal)

        s_spatial = 0.0
        if gene in step3_hyp:
            spatial_quality = step3_hyp[gene].get("spatial_quality", {})
            s_spatial += float(spatial_quality.get("gradient_decay_r2", 0.0)) * 0.5
            s_spatial += min(float(spatial_quality.get("propagation_depth", 0)) / 4.0, 1.0) * 0.3
            s_spatial += max(0.0, float(spatial_quality.get("moran_i_effect", 0.0))) * 0.2
        spatial_scores.append(s_spatial)

        is_flow = 0.0
        for edge in flow_edges:
            if str(edge.get("source", "")).upper() == gene_upper or str(edge.get("target", "")).upper() == gene_upper:
                is_flow = 1.0
                break
        in_expr = 1.0 if gene_upper in gene_cols else 0.0
        action_scores.append(is_flow * 0.6 + in_expr * 0.4)

        s_niche = 0.0
        if float(row.get("n_celltypes_ifng", 0) or 0) > 0:
            s_niche += 0.4
        if bool(row.get("is_ifng_target", False)):
            s_niche += 0.3
        if float(row.get("cross_queue_count", 0) or 0) >= 3:
            s_niche += 0.3
        niche_scores.append(s_niche)

    cross_queue = _numeric_series(pool, "cross_queue_count")
    direction = _numeric_series(pool, "direction_consistency")
    mean_abs_lfc = _numeric_series(pool, "mean_abs_lfc")

    pool["s_causal"] = minmax(np.array(causal_scores))
    pool["s_spatial"] = minmax(np.array(spatial_scores))
    pool["s_consistency"] = minmax(cross_queue * 2.0 + direction + minmax(mean_abs_lfc))
    pool["s_actionability"] = minmax(np.array(action_scores))
    pool["s_niche"] = minmax(np.array(niche_scores))

    weights = SCORE_WEIGHTS
    pool["final_score"] = (
        weights["causal"] * pool["s_causal"]
        + weights["spatial"] * pool["s_spatial"]
        + weights["consistency"] * pool["s_consistency"]
        + weights["actionability"] * pool["s_actionability"]
        + weights["niche"] * pool["s_niche"]
    )
    pool = pool.sort_values("final_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool


def retain_hubs_and_combos(
    ranking: pd.DataFrame,
    step3_results_hyp: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors_df = ranking[ranking["gene"].isin(ANCHOR_GENES)].copy() if "gene" in ranking else pd.DataFrame()
    new_hubs = ranking[~ranking["gene"].isin(ANCHOR_GENES)].head(30).copy() if "gene" in ranking else pd.DataFrame()
    retained = pd.concat([anchors_df, new_hubs], ignore_index=True)
    if not retained.empty:
        retained = retained.drop_duplicates(subset=["gene"])
        if "rank" in retained:
            retained = retained.sort_values("rank")

    combo_rows: list[dict[str, Any]] = []
    for target, result in step3_results_hyp.items():
        ranked_df = result.get("ranked_targets")
        if ranked_df is None or ranked_df.empty:
            continue
        for _, row in ranked_df.head(20).iterrows():
            combo_rows.append(
                {
                    "trigger_target": target,
                    "ligand": row.get("ligand", ""),
                    "receptor": row.get("receptor", ""),
                    "pathway": row.get("pathway", ""),
                    "causal_edge": row.get("causal_edge", ""),
                    "target_priority_score": row.get("target_priority_score", np.nan),
                }
            )
    combos = pd.DataFrame(combo_rows)
    if not combos.empty:
        combos = combos.sort_values("target_priority_score", ascending=False).reset_index(drop=True)
    return retained.reset_index(drop=True), combos


def compare_modes(
    geom_hyp: dict,
    geom_euc: dict,
    s2_hyp: dict,
    s2_euc: dict,
    s3_hyp: dict,
    s3_euc: dict,
    ranking: pd.DataFrame,
) -> dict:
    comp = {
        "geometry": {
            "hyp_separation": geom_hyp.get("metrics", {}).get("separation", 0),
            "euc_separation": geom_euc.get("metrics", {}).get("separation", 0),
        },
        "step2": {},
        "step3": {},
        "ranking": {},
    }

    for key in ["graph_sparsity", "hsic_independence", "known_axis_recall", "mean_bootstrap_freq", "neighbor_predictivity"]:
        comp["step2"][f"hyp_{key}"] = s2_hyp.get("metrics", {}).get(key, 0)
        comp["step2"][f"euc_{key}"] = s2_euc.get("metrics", {}).get(key, 0)

    if not ranking.empty and "final_score" in ranking and "gene" in ranking:
        comp["ranking"]["top50_genes"] = ranking.nlargest(min(50, len(ranking)), "final_score")["gene"].astype(str).tolist()[:20]
    else:
        comp["ranking"]["top50_genes"] = []

    shared_targets = set(s3_hyp.keys()) & set(s3_euc.keys())
    for target in list(shared_targets)[:5]:
        h_sp = s3_hyp[target].get("spatial_quality", {})
        e_sp = s3_euc[target].get("spatial_quality", {})
        comp["step3"][target] = {
            "hyp_grad_r2": h_sp.get("gradient_decay_r2", 0),
            "euc_grad_r2": e_sp.get("gradient_decay_r2", 0),
            "hyp_depth": h_sp.get("propagation_depth", 0),
            "euc_depth": e_sp.get("propagation_depth", 0),
        }

    return comp
