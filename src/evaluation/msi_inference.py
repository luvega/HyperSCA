"""基于已发表 MSI 基因签名的 MSS/MSI 转录组推断（无临床标注时近似）。

策略：主用免疫/基质签名，辅以 MMR（尤其 MLH1）。MLH1 低表达与散发 MSI（甲基化）相关。

参考文献：
- Yang et al. (2022) Sci Rep: MAP - Tumor microenvironment-aware MSI prediction
- Liu et al. (2022) Front Surg: Microsatellite-Related Transcriptomic Signature
- Lynch/MMR: MLH1 启动子甲基化 → 低 MLH1 表达
"""
from __future__ import annotations

from typing import Any

import numpy as np

# MSI-H 相关：免疫浸润、细胞毒性（MSI-H 通常免疫浸润更高）
MSI_HIGH_GENES = [
    "CD8A", "GZMB", "PRF1", "IFNG", "CXCL10", "CCL5",
    "PDCD1", "LAG3", "CTLA4", "CD274", "CXCL9",
    "STAT1", "IRF1", "GBP1", "GBP5", "IDO1",
]

# MSS 相关：基质、EMT、CAF（MSS 通常基质更丰富）
MSS_HIGH_GENES = [
    "COL1A1", "COL3A1", "FN1", "ACTA2", "POSTN",
    "MFAP2", "INHBA", "TGFB1", "FAP", "DCN",
    "LUM", "COL5A1", "COL6A2", "THBS2",
]

# MMR 辅助：Lynch 相关基因，MLH1 低表达与散发 MSI 关联
MMR_GENES = ["MLH1", "MSH2", "MSH6", "PMS2"]


def _resolve_gene_indices(
    var_names: np.ndarray | list,
    genes: list[str],
) -> tuple[list[int], list[str]]:
    """返回在 var_names 中存在的基因索引及实际匹配的基因名。"""
    if hasattr(var_names, "tolist"):
        var_names = list(var_names)
    vu = {str(v).upper(): (i, str(v)) for i, v in enumerate(var_names)}
    idx_list: list[int] = []
    found: list[str] = []
    for g in genes:
        if g.upper() in vu:
            i, name = vu[g.upper()]
            idx_list.append(i)
            found.append(name)
    return idx_list, found


def compute_msi_score(
    X: np.ndarray,
    var_names: np.ndarray | list,
    msi_genes: list[str] | None = None,
    mss_genes: list[str] | None = None,
    use_log1p: bool = True,
) -> dict[str, Any]:
    """基于 MSI/MSS 基因签名计算样本级 MSI 倾向得分。

    Parameters
    ----------
    X : ndarray (n_spots, n_genes)
        表达矩阵
    var_names : array-like
        基因名（与 X 列对齐）
    msi_genes, mss_genes : list
        签名基因列表
    use_log1p : bool
        是否对表达做 log1p 变换

    Returns
    -------
    dict
        score, mean_msi, mean_mss, n_msi_found, n_mss_found, gene_coverage,
        msi_genes_found, mss_genes_found
    """
    msi_genes = msi_genes or MSI_HIGH_GENES
    mss_genes = mss_genes or MSS_HIGH_GENES

    msi_idx, msi_found = _resolve_gene_indices(var_names, msi_genes)
    mss_idx, mss_found = _resolve_gene_indices(var_names, mss_genes)

    if not msi_idx and not mss_idx:
        return {
            "score": float("nan"),
            "mean_msi": float("nan"),
            "mean_mss": float("nan"),
            "n_msi_found": 0,
            "n_mss_found": 0,
            "gene_coverage": 0.0,
            "msi_genes_found": [],
            "mss_genes_found": [],
        }

    vals = np.asarray(X, dtype=np.float64).copy()
    if use_log1p:
        vals = np.log1p(np.maximum(vals, 0))

    mean_msi = float(np.nanmean(vals[:, msi_idx])) if msi_idx else 0.0
    mean_mss = float(np.nanmean(vals[:, mss_idx])) if mss_idx else 0.0

    all_idx = msi_idx + mss_idx
    pooled = vals[:, all_idx].flatten()
    pooled = pooled[~np.isnan(pooled) & np.isfinite(pooled)]
    std_pool = float(np.std(pooled)) if len(pooled) > 1 else 1.0
    std_pool = max(std_pool, 1e-6)

    score = (mean_msi - mean_mss) / std_pool

    n_total = len(msi_genes) + len(mss_genes)
    n_found = len(msi_found) + len(mss_found)
    coverage = n_found / n_total if n_total else 0.0

    return {
        "score": float(score),
        "mean_msi": mean_msi,
        "mean_mss": mean_mss,
        "n_msi_found": len(msi_found),
        "n_mss_found": len(mss_found),
        "gene_coverage": coverage,
        "msi_genes_found": msi_found,
        "mss_genes_found": mss_found,
    }


def compute_msi_score_combined(
    X: np.ndarray,
    var_names: np.ndarray | list,
    msi_genes: list[str] | None = None,
    mss_genes: list[str] | None = None,
    mmr_genes: list[str] | None = None,
    use_log1p: bool = True,
    alpha_primary: float = 0.85,
    beta_mmr: float = 0.15,
) -> dict[str, Any]:
    """主用免疫/基质签名，辅以 MMR（尤其 MLH1）的综合 MSI 得分。

    MLH1 低表达与散发 MSI（启动子甲基化）相关，作为辅助证据。
    最终得分 = alpha_primary * primary_score + beta_mmr * mmlh1_aux

    Parameters
    ----------
    alpha_primary : float
        免疫/基质主得分权重（默认 0.85）
    beta_mmr : float
        MLH1 辅助权重（默认 0.15）

    Returns
    -------
    dict
        继承 compute_msi_score 的字段，额外：
        score_primary, score_mmr_aux, mean_mlh1, n_mmr_found
    """
    msi_genes = msi_genes or MSI_HIGH_GENES
    mss_genes = mss_genes or MSS_HIGH_GENES
    mmr_genes = mmr_genes or MMR_GENES

    res = compute_msi_score(X, var_names, msi_genes, mss_genes, use_log1p)
    primary = res["score"]

    mmlh1_aux = 0.0
    mean_mlh1 = float("nan")
    mmr_found: list[str] = []

    mrr_idx, mrr_found = _resolve_gene_indices(var_names, mmr_genes)
    if mrr_idx:
        vals = np.asarray(X, dtype=np.float64)
        if use_log1p:
            vals = np.log1p(np.maximum(vals, 0))
        mlh1_i = next((i for i, g in enumerate(mrr_found) if g.upper() == "MLH1"), None)
        if mlh1_i is not None:
            mlh1_vals = vals[:, mrr_idx[mlh1_i]]
            mean_mlh1 = float(np.nanmean(mlh1_vals))
            all_mmr = vals[:, mrr_idx].flatten()
            med = float(np.nanmedian(all_mmr))
            std_all = max(float(np.nanstd(all_mmr)), 1e-6)
            # 低 MLH1 → 正 aux（MSI-like）
            mlh1_z = (mean_mlh1 - med) / std_all
            mmlh1_aux = float(np.clip(-mlh1_z, -1.0, 1.0))
        else:
            mean_mlh1 = float("nan")
        mmr_found = mrr_found

    combined = alpha_primary * primary + beta_mmr * mmlh1_aux
    res["score_primary"] = primary
    res["score_mmr_aux"] = mmlh1_aux
    res["mean_mlh1"] = mean_mlh1
    res["n_mmr_found"] = len(mmr_found)
    res["mmr_genes_found"] = mmr_found
    res["score"] = float(combined)
    return res


def infer_status_from_score(
    score: float,
    threshold_msi: float = 0.3,
    threshold_mss: float = -0.3,
) -> tuple[str, str]:
    """从得分推断 MSS/MSI 及置信度。

    Returns
    -------
    (status, confidence)
        status: "MSI", "MSS", "Unknown"
        confidence: "high", "medium", "low", "none"
    """
    if np.isnan(score):
        return "Unknown", "none"

    if score >= threshold_msi:
        status = "MSI"
        conf = "high" if score >= 0.8 else ("medium" if score >= 0.5 else "low")
    elif score <= threshold_mss:
        status = "MSS"
        conf = "high" if score <= -0.8 else ("medium" if score <= -0.5 else "low")
    else:
        status = "Unknown"
        conf = "low"
    return status, conf
