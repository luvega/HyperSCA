"""
Cross-sample and niche-level evaluation metrics for multi-source CRC integration.

Provides:
  - Niche clustering from ST deconvolution composition vectors
  - Per-niche enrichment of cell types
  - Cross-sample consistency of causal edges
  - MMR-stratified differential statistics
  - Integration into the target_discovery evidence scoring pipeline
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_niches(
    deconv_matrix: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
) -> dict:
    """Cluster spatial spots into niches based on cell-type deconvolution scores.

    Parameters
    ----------
    deconv_matrix : DataFrame
        (n_spots, n_celltypes) deconvolution proportions per spot.
    n_clusters : int
        Number of niche clusters.
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys:
        labels : ndarray (n_spots,) cluster assignments
        centroids : ndarray (n_clusters, n_celltypes)
        silhouette : float
        dominant_types : list[str] per cluster
    """
    X = deconv_matrix.values.astype(float)
    if X.shape[0] < n_clusters:
        n_clusters = max(2, X.shape[0] // 2)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0

    dominant_types = []
    for c in range(n_clusters):
        top_idx = int(np.argmax(centroids[c]))
        dominant_types.append(deconv_matrix.columns[top_idx])

    return {
        "labels": labels,
        "centroids": centroids,
        "silhouette": float(sil),
        "n_clusters": n_clusters,
        "dominant_types": dominant_types,
    }


def niche_enrichment(
    deconv_matrix: pd.DataFrame,
    niche_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute per-niche enrichment of each cell type vs background.

    Returns DataFrame: (n_clusters, n_celltypes) with log2 fold-enrichment values.
    """
    global_mean = deconv_matrix.mean(axis=0)
    records = []
    for c in sorted(set(niche_labels)):
        mask = niche_labels == c
        niche_mean = deconv_matrix.loc[mask].mean(axis=0)
        lfc = np.log2((niche_mean + 1e-6) / (global_mean + 1e-6))
        for ct, val in lfc.items():
            records.append({"niche": c, "celltype": ct, "log2_enrichment": float(val)})
    return pd.DataFrame(records)


def cross_sample_edge_consistency(
    per_sample_adjacencies: dict[str, np.ndarray],
    node_labels: list[str],
    threshold: float = 0.3,
) -> dict:
    """Measure consistency of causal edges across samples.

    Parameters
    ----------
    per_sample_adjacencies : dict
        {sample_id: adjacency_matrix (K, K)}
    node_labels : list[str]
    threshold : float
        Edge presence threshold.

    Returns
    -------
    dict with:
        edge_frequency : ndarray (K, K) fraction of samples with edge
        n_robust_edges : int (present in >50% samples)
        jaccard_mean : float (mean pairwise Jaccard of edge sets)
    """
    K = len(node_labels)
    n_samples = len(per_sample_adjacencies)
    if n_samples == 0:
        return {"edge_frequency": np.zeros((K, K)), "n_robust_edges": 0, "jaccard_mean": 0.0}

    edge_count = np.zeros((K, K))
    edge_sets = []
    for sid, adj in per_sample_adjacencies.items():
        mask = (adj > threshold).astype(int)
        edge_count += mask
        edge_sets.append(set(zip(*np.where(mask > 0))))

    freq = edge_count / n_samples
    n_robust = int((freq > 0.5).sum())

    jaccards = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            union = edge_sets[i] | edge_sets[j]
            inter = edge_sets[i] & edge_sets[j]
            jaccards.append(len(inter) / max(len(union), 1))

    return {
        "edge_frequency": freq,
        "n_robust_edges": n_robust,
        "jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        "n_samples": n_samples,
    }


def mmr_stratified_test(
    expression_df: pd.DataFrame,
    mmr_labels: pd.Series,
    test_genes: list[str],
) -> pd.DataFrame:
    """Perform MMR-stratified differential testing (Wilcoxon rank-sum).

    Parameters
    ----------
    expression_df : DataFrame (n_cells, n_genes)
    mmr_labels : Series of 'pMMR' / 'dMMR' aligned with expression_df index
    test_genes : genes to test

    Returns
    -------
    DataFrame with columns: gene, mmr_comparison, stat, pvalue, mean_pMMR, mean_dMMR, log2FC
    """
    records = []
    groups = mmr_labels.unique()
    if len(groups) < 2:
        return pd.DataFrame(columns=[
            "gene", "mmr_comparison", "stat", "pvalue", "mean_pMMR", "mean_dMMR", "log2FC"
        ])

    g1_name = "pMMR" if "pMMR" in groups else sorted(groups)[0]
    g2_name = "dMMR" if "dMMR" in groups else sorted(groups)[-1]

    mask1 = mmr_labels == g1_name
    mask2 = mmr_labels == g2_name

    for gene in test_genes:
        if gene not in expression_df.columns:
            continue
        x1 = expression_df.loc[mask1, gene].dropna().values
        x2 = expression_df.loc[mask2, gene].dropna().values
        if len(x1) < 3 or len(x2) < 3:
            continue
        stat_val, p_val = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        m1 = float(np.mean(x1))
        m2 = float(np.mean(x2))
        lfc = float(np.log2((m1 + 1e-6) / (m2 + 1e-6)))
        records.append({
            "gene": gene,
            "mmr_comparison": f"{g1_name}_vs_{g2_name}",
            "stat": float(stat_val),
            "pvalue": float(p_val),
            f"mean_{g1_name}": m1,
            f"mean_{g2_name}": m2,
            "log2FC": lfc,
        })

    return pd.DataFrame(records)


def niche_prototype_matching(
    per_sample_prototypes: dict[str, pd.DataFrame],
    method: str = "hungarian",
) -> dict:
    """Match niche prototypes across samples via Hungarian algorithm.

    Parameters
    ----------
    per_sample_prototypes : dict
        {sample_id: DataFrame(n_niches, n_celltypes)} – each row is a
        niche prototype (mean composition vector).
    method : str
        Matching method; currently only ``"hungarian"`` is supported.

    Returns
    -------
    dict with keys:
        n_samples, pairwise_consistency (list of dicts),
        mean_consistency, q25_consistency.
    """
    from scipy.optimize import linear_sum_assignment

    samples = sorted(per_sample_prototypes.keys())
    if len(samples) < 2:
        return {
            "n_samples": len(samples),
            "pairwise_consistency": [],
            "mean_consistency": 0.0,
            "q25_consistency": 0.0,
        }

    pairwise: list[dict] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            s1, s2 = samples[i], samples[j]
            p1 = per_sample_prototypes[s1]
            p2 = per_sample_prototypes[s2]
            n1, n2 = p1.shape[0], p2.shape[0]

            corr = np.zeros((n1, n2))
            for a in range(n1):
                for b in range(n2):
                    v1 = p1.iloc[a].values.astype(float)
                    v2 = p2.iloc[b].values.astype(float)
                    sd1 = float(np.std(v1))
                    sd2 = float(np.std(v2))
                    if sd1 < 1e-12 or sd2 < 1e-12:
                        corr[a, b] = 0.0
                    else:
                        r_val = float(np.corrcoef(v1, v2)[0, 1])
                        corr[a, b] = r_val if np.isfinite(r_val) else 0.0

            cost = 1.0 - corr
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_corrs = [float(corr[r, c]) for r, c in zip(row_ind, col_ind)]

            pairwise.append({
                "sample_a": s1,
                "sample_b": s2,
                "matched_pairs": list(zip(row_ind.tolist(), col_ind.tolist())),
                "matched_correlations": matched_corrs,
                "mean_match_corr": float(np.mean(matched_corrs)) if matched_corrs else 0.0,
                "q25_match_corr": (
                    float(np.quantile(matched_corrs, 0.25))
                    if matched_corrs else 0.0
                ),
            })

    all_corrs = [p["mean_match_corr"] for p in pairwise]
    return {
        "n_samples": len(samples),
        "pairwise_consistency": pairwise,
        "mean_consistency": float(np.mean(all_corrs)) if all_corrs else 0.0,
        "q25_consistency": float(np.quantile(all_corrs, 0.25)) if all_corrs else 0.0,
    }


def niche_target_score(
    niche_enrichment_df: pd.DataFrame,
    target_gene: str,
    niche_labels: np.ndarray,
    gene_expr: np.ndarray,
) -> float:
    """Compute a niche-aware score for a target gene.

    High scores when the gene is differentially expressed across
    niches dominated by different cell types (indicating niche-specificity).
    """
    if len(set(niche_labels)) < 2 or len(gene_expr) != len(niche_labels):
        return 0.0

    niche_means = []
    for c in sorted(set(niche_labels)):
        mask = niche_labels == c
        niche_means.append(float(np.mean(gene_expr[mask])))

    variance = float(np.var(niche_means))
    overall_var = float(np.var(gene_expr)) if np.var(gene_expr) > 0 else 1e-12

    return min(variance / overall_var, 1.0)


def evaluate_cross_sample(
    per_sample_adjacencies: Optional[dict[str, np.ndarray]] = None,
    node_labels: Optional[list[str]] = None,
    deconv_matrix: Optional[pd.DataFrame] = None,
    n_niches: int = 5,
    mmr_labels: Optional[pd.Series] = None,
    expression_df: Optional[pd.DataFrame] = None,
    test_genes: Optional[list[str]] = None,
) -> dict:
    """One-stop cross-sample evaluation aggregating niche, edge consistency, and MMR metrics."""
    results: dict = {}

    if deconv_matrix is not None and not deconv_matrix.empty:
        niche_res = cluster_niches(deconv_matrix, n_clusters=n_niches)
        results["niche_silhouette"] = niche_res["silhouette"]
        results["niche_dominant_types"] = niche_res["dominant_types"]
        results["niche_n_clusters"] = niche_res["n_clusters"]
        enr = niche_enrichment(deconv_matrix, niche_res["labels"])
        results["niche_enrichment_summary"] = {
            "n_enriched": int((enr["log2_enrichment"].abs() > 0.5).sum()),
            "top_enrichment": float(enr["log2_enrichment"].abs().max()),
        }

    if per_sample_adjacencies and node_labels:
        edge_cons = cross_sample_edge_consistency(per_sample_adjacencies, node_labels)
        results["edge_jaccard_mean"] = edge_cons["jaccard_mean"]
        results["n_robust_edges"] = edge_cons["n_robust_edges"]

    if mmr_labels is not None and expression_df is not None and test_genes:
        mmr_res = mmr_stratified_test(expression_df, mmr_labels, test_genes)
        results["mmr_n_significant"] = int((mmr_res["pvalue"] < 0.05).sum()) if not mmr_res.empty else 0
        results["mmr_results"] = mmr_res.to_dict(orient="records") if not mmr_res.empty else []

    return results
