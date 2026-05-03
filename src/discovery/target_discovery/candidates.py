"""Candidate target discovery from multi-source DEG tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.discovery.target_discovery.constants import ANCHOR_GENES, IFNG_FOCUS_GENES
from src.discovery.target_discovery.utils import minmax


def aggregate_candidate_pool(
    neu_df: pd.DataFrame,
    icb_df: pd.DataFrame,
    ifng_df: pd.DataFrame,
) -> pd.DataFrame:
    all_genes: set[str] = set()
    if not neu_df.empty:
        all_genes |= set(neu_df["gene"].dropna().astype(str).unique())
    if not icb_df.empty:
        all_genes |= set(icb_df["gene"].dropna().astype(str).unique())
    if not ifng_df.empty:
        all_genes |= set(ifng_df["gene"].dropna().astype(str).unique())
    all_genes -= {"", "nan", "None"}

    rows: list[dict] = []
    for gene in sorted(all_genes):
        n_sub = neu_df[neu_df["gene"].astype(str) == gene] if not neu_df.empty else pd.DataFrame()
        i_sub = icb_df[icb_df["gene"].astype(str) == gene] if not icb_df.empty else pd.DataFrame()
        f_sub = ifng_df[ifng_df["gene"].astype(str) == gene] if not ifng_df.empty else pd.DataFrame()

        lfcs: list[float] = []
        for frame, col in [(n_sub, "lfc_neu"), (i_sub, "lfc_icb"), (f_sub, "lfc_ifng")]:
            if not frame.empty and col in frame:
                lfcs.extend(pd.to_numeric(frame[col], errors="coerce").dropna().astype(float).tolist())
        padjs: list[float] = []
        for frame, col in [(n_sub, "padj_neu"), (i_sub, "padj_icb")]:
            if not frame.empty and col in frame:
                padjs.extend(pd.to_numeric(frame[col], errors="coerce").dropna().astype(float).tolist())

        signs = np.sign(lfcs) if lfcs else np.array([])
        majority = np.sign(np.sum(signs)) if signs.size else 0
        direction_consistency = float(np.mean(signs == majority)) if majority != 0 else (0.5 if signs.size else 0.0)

        n_ct_neu = int(n_sub["celltype_neu"].nunique()) if "celltype_neu" in n_sub else 0
        n_ct_icb = int(i_sub["celltype_icb"].nunique()) if "celltype_icb" in i_sub else 0
        n_ct_ifng = int(f_sub["celltype_ifng"].nunique()) if "celltype_ifng" in f_sub else 0
        min_padj = float(np.min(padjs)) if padjs else 1.0

        rows.append(
            {
                "gene": gene,
                "n_celltypes_neu": n_ct_neu,
                "n_celltypes_icb": n_ct_icb,
                "n_celltypes_ifng": n_ct_ifng,
                "cross_queue_count": int(n_ct_neu > 0) + int(n_ct_icb > 0) + int(n_ct_ifng > 0),
                "mean_lfc": float(np.mean(lfcs)) if lfcs else 0.0,
                "mean_abs_lfc": float(np.mean(np.abs(lfcs))) if lfcs else 0.0,
                "direction_consistency": direction_consistency,
                "min_padj": min_padj,
                "neg_log10_padj": -float(np.log10(max(min_padj, 1e-300))),
                "is_anchor": gene in ANCHOR_GENES,
                "is_ifng_target": gene in IFNG_FOCUS_GENES,
                "celltypes_neu": ";".join(sorted(n_sub["celltype_neu"].astype(str).unique())) if "celltype_neu" in n_sub else "",
                "celltypes_icb": ";".join(sorted(i_sub["celltype_icb"].astype(str).unique())) if "celltype_icb" in i_sub else "",
                "celltypes_ifng": ";".join(sorted(f_sub["celltype_ifng"].astype(str).unique())) if "celltype_ifng" in f_sub else "",
            }
        )

    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool
    pool["init_score"] = (
        pool["cross_queue_count"] * 2.0
        + minmax(pool["mean_abs_lfc"].values) * 1.5
        + minmax(pool["neg_log10_padj"].values) * 1.5
        + pool["direction_consistency"] * 1.0
        + minmax(pool["n_celltypes_neu"].values) * 0.5
        + minmax(pool["n_celltypes_ifng"].values) * 0.5
    )
    return pool.sort_values("init_score", ascending=False).reset_index(drop=True)
