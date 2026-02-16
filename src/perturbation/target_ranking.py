"""阶段 3: 反事实细胞互作靶点排序。

将阶段 2 的信号流边（Ligand->Receptor）与 prior_db 先验资源融合，
对干预前后（Observed vs Counterfactual）变化进行打分与排序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PriorKnowledge:
    """统一的先验知识容器。"""

    lr_pairs: set[tuple[str, str]]
    tf_targets: set[tuple[str, str]]
    sources: dict[tuple[str, str], set[str]]


def load_prior_knowledge() -> PriorKnowledge:
    """从 src.data.prior_db 读取先验资源并统一映射。

    Returns
    -------
    PriorKnowledge
        合并 OmniPath / LIANA / NicheNet（如可用）的 LR 对与来源信息。
    """
    try:
        from src.data.prior_db import load_liana_resource
    except Exception:  # pragma: no cover
        load_liana_resource = None
    try:
        from src.data.prior_db import load_lr_interactions, load_nichenet_prior, load_tf_targets
    except Exception:  # pragma: no cover
        load_lr_interactions = None
        load_nichenet_prior = None
        load_tf_targets = None

    lr_pairs: set[tuple[str, str]] = set()
    tf_pairs: set[tuple[str, str]] = set()
    sources: dict[tuple[str, str], set[str]] = {}

    def _add_lr(df: Optional[pd.DataFrame], source_name: str) -> None:
        if df is None or df.empty:
            return
        cols = {c.lower(): c for c in df.columns}
        lig_col = None
        rec_col = None
        # 常见命名
        for c in ("ligand", "source_genesymbol", "source"):
            if c in cols:
                lig_col = cols[c]
                break
        for c in ("receptor", "target_genesymbol", "target"):
            if c in cols:
                rec_col = cols[c]
                break
        if lig_col is None or rec_col is None:
            return
        sub = df[[lig_col, rec_col]].dropna()
        for lig, rec in sub.itertuples(index=False):
            key = (str(lig).upper(), str(rec).upper())
            lr_pairs.add(key)
            sources.setdefault(key, set()).add(source_name)

    # OmniPath LR
    if load_lr_interactions is not None:
        try:
            _add_lr(load_lr_interactions(), "omnipath")
        except Exception:
            pass

    # LIANA consensus LR
    if load_liana_resource is not None:
        try:
            _add_lr(load_liana_resource(), "liana")
        except Exception:
            pass

    # NicheNet LR
    if load_nichenet_prior is not None:
        try:
            nichenet = load_nichenet_prior()
            _add_lr(nichenet.get("lr_network"), "nichenet")
        except Exception:
            pass

    # OmniPath TF-target
    if load_tf_targets is not None:
        try:
            tf_df = load_tf_targets()
            cols = {c.lower(): c for c in tf_df.columns}
            tf_col = cols.get("source_genesymbol", cols.get("source"))
            tgt_col = cols.get("target_genesymbol", cols.get("target"))
            if tf_col and tgt_col:
                sub = tf_df[[tf_col, tgt_col]].dropna()
                tf_pairs = {(str(tf).upper(), str(tg).upper()) for tf, tg in sub.itertuples(index=False)}
        except Exception:
            pass

    return PriorKnowledge(lr_pairs=lr_pairs, tf_targets=tf_pairs, sources=sources)


def rank_counterfactual_interaction_targets(
    flow_edges: list[dict],
    observed_expression: pd.DataFrame,
    counterfactual_expression: pd.DataFrame,
    *,
    prior: Optional[PriorKnowledge] = None,
    node_to_type: Optional[dict[str, str]] = None,
    min_abs_delta: float = 0.01,
    top_k: Optional[int] = 50,
) -> pd.DataFrame:
    """对反事实细胞互作靶点进行排序。

    Parameters
    ----------
    flow_edges
        来自阶段2/3的多层流边列表，至少包含 layer0->1（Ligand->Receptor）边。
    observed_expression, counterfactual_expression
        行为节点/细胞群，列为基因的表达矩阵。
    prior
        可选先验。若为空则自动从 prior_db 加载（若不可用则降级为无先验）。
    node_to_type
        节点到细胞类型映射。若提供，则优先使用 source_type/target_type 分组变化。
    min_abs_delta
        对 ligand/receptor 平均变化幅度的最小阈值。
    top_k
        返回前 K 条，None 表示返回全部。
    """
    if observed_expression.shape[1] != counterfactual_expression.shape[1]:
        raise ValueError("Observed/Counterfactual gene dimensions mismatch.")

    if prior is None:
        prior = load_prior_knowledge()

    # 统一基因列
    common_genes = [g for g in observed_expression.columns if g in counterfactual_expression.columns]
    if not common_genes:
        raise ValueError("No shared genes between observed and counterfactual expression.")
    obs = observed_expression[common_genes]
    cf = counterfactual_expression[common_genes]

    # 全局平均变化（无分组信息时回退）
    delta_global = cf.mean(axis=0) - obs.mean(axis=0)
    delta_map = {str(g).upper(): float(v) for g, v in delta_global.items()}

    grouped_delta: dict[tuple[str, str], float] = {}
    if node_to_type:
        obs_idx = [idx for idx in obs.index if idx in node_to_type]
        cf_idx = [idx for idx in cf.index if idx in node_to_type]
        common_idx = [idx for idx in obs_idx if idx in cf_idx]
        if common_idx:
            obs_sub = obs.loc[common_idx]
            cf_sub = cf.loc[common_idx]
            types = sorted({node_to_type[idx] for idx in common_idx})
            for ctype in types:
                rows = [idx for idx in common_idx if node_to_type[idx] == ctype]
                if not rows:
                    continue
                d = cf_sub.loc[rows].mean(axis=0) - obs_sub.loc[rows].mean(axis=0)
                for g, v in d.items():
                    grouped_delta[(ctype, str(g).upper())] = float(v)

    records: list[dict] = []
    for edge in flow_edges:
        if edge.get("source_layer") != 0 or edge.get("target_layer") != 1:
            continue

        ligand = str(edge.get("source", "")).upper()
        receptor = str(edge.get("target", "")).upper()
        if not ligand or not receptor:
            continue

        src_type = None
        tgt_type = None
        causal_edge = str(edge.get("causal_edge", ""))
        if "→" in causal_edge:
            parts = causal_edge.split("→")
            if len(parts) == 2:
                src_type, tgt_type = parts[0].strip(), parts[1].strip()
        elif "->" in causal_edge:
            parts = causal_edge.split("->")
            if len(parts) == 2:
                src_type, tgt_type = parts[0].strip(), parts[1].strip()

        if src_type and (src_type, ligand) in grouped_delta:
            d_l = grouped_delta[(src_type, ligand)]
        else:
            d_l = delta_map.get(ligand, 0.0)
        if tgt_type and (tgt_type, receptor) in grouped_delta:
            d_r = grouped_delta[(tgt_type, receptor)]
        else:
            d_r = delta_map.get(receptor, 0.0)
        combined_delta = abs(d_l) + abs(d_r)
        if combined_delta < min_abs_delta:
            continue

        base_weight = float(edge.get("weight", 0.0))
        prior_hit = (ligand, receptor) in prior.lr_pairs
        prior_bonus = 1.25 if prior_hit else 1.0
        evidence_sources = sorted(prior.sources.get((ligand, receptor), set()))

        # 若有 pathway / causal_edge，可作为解释信息直接输出
        score = base_weight * combined_delta * prior_bonus
        records.append(
            {
                "ligand": ligand,
                "receptor": receptor,
                "causal_edge": causal_edge,
                "source_type": src_type or "",
                "target_type": tgt_type or "",
                "pathway": edge.get("pathway", ""),
                "flow_weight": base_weight,
                "delta_ligand": d_l,
                "delta_receptor": d_r,
                "combined_abs_delta": combined_delta,
                "prior_hit": prior_hit,
                "prior_sources": ",".join(evidence_sources),
                "target_priority_score": float(score),
            }
        )

    if not records:
        cols = [
            "ligand",
            "receptor",
            "causal_edge",
            "source_type",
            "target_type",
            "pathway",
            "flow_weight",
            "delta_ligand",
            "delta_receptor",
            "combined_abs_delta",
            "prior_hit",
            "prior_sources",
            "target_priority_score",
        ]
        return pd.DataFrame(columns=cols)

    ranked = pd.DataFrame(records).sort_values(
        ["target_priority_score", "combined_abs_delta"],
        ascending=[False, False],
    )
    ranked = ranked.reset_index(drop=True)
    if top_k is not None:
        ranked = ranked.head(top_k).copy()
    return ranked

