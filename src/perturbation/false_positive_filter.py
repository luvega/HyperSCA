"""Step3 假阳性过滤模块。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_empirical_fdr(
    scores: np.ndarray,
    n_perm: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """通过置换估计每个 score 的经验 FDR。"""
    rng = np.random.default_rng(seed)
    if scores.size == 0:
        return np.array([], dtype=float)
    null_scores = []
    centered = scores - np.mean(scores)
    for _ in range(n_perm):
        perm = rng.permutation(centered)
        null_scores.append(np.abs(perm))
    null = np.concatenate(null_scores)
    fdr = np.zeros_like(scores, dtype=float)
    abs_scores = np.abs(scores)
    for i, s in enumerate(abs_scores):
        fp = np.mean(null >= s)
        tp = np.mean(abs_scores >= s)
        fdr[i] = float(fp / max(tp, 1e-8))
    return np.clip(fdr, 0.0, 1.0)


def filter_false_positive_targets(
    ranked: pd.DataFrame,
    *,
    score_col: str = "target_priority_score",
    min_score: float = 0.02,
    max_fdr: float = 0.25,
    require_prior_or_large_effect: bool = True,
) -> pd.DataFrame:
    """对阶段3候选靶点进行假阳性过滤。"""
    if ranked is None or ranked.empty:
        return ranked.copy() if ranked is not None else pd.DataFrame()
    if score_col not in ranked.columns:
        return ranked.copy()

    out = ranked.copy()
    scores = out[score_col].astype(float).values
    out["empirical_fdr"] = estimate_empirical_fdr(scores)
    out["pass_score"] = out[score_col].astype(float) >= min_score
    out["pass_fdr"] = out["empirical_fdr"] <= max_fdr

    if require_prior_or_large_effect:
        prior = out["prior_hit"].astype(bool) if "prior_hit" in out.columns else False
        large_effect = (
            out["combined_abs_delta"].astype(float) >= out["combined_abs_delta"].astype(float).quantile(0.7)
            if "combined_abs_delta" in out.columns
            else True
        )
        out["pass_bio_rule"] = prior | large_effect
    else:
        out["pass_bio_rule"] = True

    out["pass_filter"] = out["pass_score"] & out["pass_fdr"] & out["pass_bio_rule"]
    out = out.sort_values([score_col, "combined_abs_delta"], ascending=[False, False])
    return out
