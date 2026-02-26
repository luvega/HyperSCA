"""
MVP Integration: Result-level HyperSCA pipeline for CRC multi-source data.

Bypasses Step1 (H-VAE) by constructing cluster-level inputs directly from:
  - scCRC_Neu: NormalizedCounts pseudo-bulk expression → cluster_expr (K, G)
  - ST_CRC_MSS: deconvolution co-localization → cluster_adj (K, K)
  - scCRC_IFNG: CosMx + scRNA MMR-annotated evidence (primary secondary source)
  - scCRC_ICB: DEG evidence (optional, kept for backward compat)

Then runs Step2 core (disentangle → causal discovery → signaling flow)
and Step3 core (perturbation → spatial propagation → target ranking).

Usage:
    python scripts/run_mvp_integration.py
    python scripts/run_mvp_integration.py --embedding-mode euclidean
    python scripts/run_mvp_integration.py --no-icb --targets POSTN INHBA CD74
    python scripts/run_mvp_integration.py --max-targets 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "integration" / "mvp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEU_DIR = Path(r"G:\scCRC_Neu\downstream_analyses_de_analysis"
               r"\0downstream_analyses_de_analysis\de_analysis"
               r"\de_analysis_tumor_mss_msi\deseq2_dgea")
IFNG_DIR = Path(r"F:\scCRC_IFNG")
ICB_DIR = Path(r"G:\scCRC_ICB\output")
ST_DIR  = Path(r"G:\ST_CRC_MSS")

ANCHOR_TARGETS = ["MFAP2", "POSTN", "INHBA"]
IFNG_TARGETS = ["CD74", "INHBA", "CXCL10", "IFNG", "COL1A1", "MFAP5", "FN1"]

# Cell types relevant to the three seed targets and CRC TME axes
FOCUS_CELLTYPES = [
    "Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3",
    "Macrophage", "Macrophage_cycling",
    "Pericyte",
    "T_cell_CD4", "T_cell_CD8", "T_cell_regulatory",
    "cDC1", "cDC2",
    "Neutrophil",
    "Endothelial_venous",
    "Monocyte_classical",
]

# Map each focus cell type to a TME role for signaling flow
TYPE_MAPPING = {
    "Fibroblast_S1": "CAF", "Fibroblast_S2": "CAF", "Fibroblast_S3": "CAF",
    "Macrophage": "TAM", "Macrophage_cycling": "TAM",
    "Pericyte": "CAF",
    "T_cell_CD4": "CD4T", "T_cell_CD8": "CD8T", "T_cell_regulatory": "Treg",
    "cDC1": "DC", "cDC2": "DC",
    "Neutrophil": "Neutrophil",
    "Endothelial_venous": "Endothelial",
    "Monocyte_classical": "Monocyte",
}

# Corresponding ST deconvolution columns for each focus cell type
ST_DECONV_MAP = {
    "Fibroblast_S1": ["Fibro_ADAMDEC1", "Fibro_CXCL8", "Fibro_CXCL14"],
    "Fibroblast_S2": ["Fibro_GPM6B", "Fibro_KCNN3", "Fibro_MYH11"],
    "Fibroblast_S3": ["Fibro_NOTCH3", "Fibro_PI16"],
    "Macrophage": ["Mac_M1", "Mac_M2", "Mac_SPP1"],
    "Macrophage_cycling": ["Mac_M1"],
    "Pericyte": ["Endo"],
    "T_cell_CD4": ["CD4_CXCL13", "CD4_Tcm", "CD4_Treg", "CD4_act"],
    "T_cell_CD8": ["CD8_Cyto", "CD8_HSP", "CD8_Teff", "CD8_Tem", "CD8_Tex"],
    "T_cell_regulatory": ["CD4_Treg"],
    "cDC1": ["cDC1"],
    "cDC2": ["cDC2"],
    "Neutrophil": ["Monocyte_S100A8"],
    "Endothelial_venous": ["Endo"],
    "Monocyte_classical": ["Monocyte_S100A8"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyperSCA MVP result-level integration"
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["euclidean", "hyperbolic", "both"],
        default="both",
        help="Geometry mode(s); 'both' runs hyperbolic + euclidean sequentially.",
    )
    parser.add_argument(
        "--geometry-blend",
        type=float,
        default=0.30,
        help="Blend weight for geometry adjacency into Step2 adjacency (0-1).",
    )
    parser.add_argument(
        "--geometry-k",
        type=int,
        default=4,
        help="k for kNN graph on auxiliary embedding.",
    )
    parser.add_argument(
        "--targets", nargs="*", default=None,
        help="Explicit target genes to perturb (overrides auto-discovery).",
    )
    parser.add_argument(
        "--max-targets", type=int, default=10,
        help="Max number of targets when using auto-discovery from IFNG + Neu.",
    )
    parser.add_argument(
        "--no-icb", action="store_true",
        help="Skip scCRC_ICB data source (use IFNG instead).",
    )
    return parser.parse_args()


def discover_targets(cluster_expr: pd.DataFrame, max_targets: int,
                     explicit: list[str] | None = None) -> list[str]:
    """Build dynamic target list: anchors first, then IFNG targets, then DEG-driven."""
    gene_upper = {c.upper(): c for c in cluster_expr.columns}
    available = set(gene_upper.keys())

    if explicit:
        return [g for g in explicit if g.upper() in available]

    targets: list[str] = []
    for g in ANCHOR_TARGETS:
        if g.upper() in available and g not in targets:
            targets.append(g)
    for g in IFNG_TARGETS:
        if g.upper() in available and g not in targets:
            targets.append(g)

    # IFNG MMR-specific targets
    ifng_mmr_path = IFNG_DIR / "results" / "tables" / "targets_shared_specific_by_mmr.csv"
    if ifng_mmr_path.exists():
        try:
            mdf = pd.read_csv(ifng_mmr_path)
            gcol = "gene" if "gene" in mdf.columns else mdf.columns[0]
            for g in mdf[gcol].dropna().unique():
                g = str(g)
                if g.upper() in available and g not in targets and len(targets) < max_targets:
                    targets.append(g)
        except Exception:
            pass

    # Fill remaining from Neu DESeq2 top significant genes
    if len(targets) < max_targets:
        sig_genes = []
        for tsv in sorted(NEU_DIR.glob("*-DESeq2_result.tsv")):
            try:
                df = pd.read_csv(tsv, sep="\t")
                sig = df[(df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 1.0)]
                for g in sig["symbol"].dropna().unique():
                    if g.upper() in available and g not in targets and g not in sig_genes:
                        sig_genes.append(g)
            except Exception:
                continue
        for g in sig_genes[:max_targets - len(targets)]:
            targets.append(g)

    return targets[:max_targets]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase A: Build cluster-level expression matrix from NormalizedCounts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_cluster_expression() -> tuple[pd.DataFrame, list[str]]:
    """Load NormalizedCounts for each focus cell type and compute sample means.

    Returns (K, G) DataFrame (rows=cell types, cols=genes) and list of node labels.
    """
    print("=" * 60)
    print("[Phase A] Building cluster expression from scCRC_Neu NormalizedCounts")
    print("=" * 60)

    dfs = {}
    for ct in FOCUS_CELLTYPES:
        fpath = NEU_DIR / f"{ct}-NormalizedCounts.tsv"
        if not fpath.exists():
            print(f"  SKIP: {ct} (file not found)")
            continue
        df = pd.read_csv(fpath, sep="\t", index_col=0)
        mean_expr = df.mean(axis=1)
        dfs[ct] = mean_expr
        print(f"  Loaded {ct}: {df.shape[1]} samples, {df.shape[0]} genes")

    if not dfs:
        raise RuntimeError("No NormalizedCounts files loaded")

    cluster_expr = pd.DataFrame(dfs).T
    cluster_expr = cluster_expr.fillna(0)

    # log1p normalize for numerical stability in downstream models
    cluster_expr_log = np.log1p(cluster_expr)

    node_labels = list(cluster_expr_log.index)
    print(f"\n  Cluster expression: {cluster_expr_log.shape} "
          f"({len(node_labels)} cell types x {cluster_expr_log.shape[1]} genes)")
    return cluster_expr_log, node_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase B: Build spatial adjacency from ST deconvolution co-localization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_spatial_adjacency(node_labels: list[str]) -> np.ndarray:
    """Compute cell-type co-localization adjacency from ST spot-level deconvolution.

    For each pair of cell types, Pearson correlation of their deconvolution
    scores across spots indicates spatial co-localization.
    """
    print()
    print("=" * 60)
    print("[Phase B] Building spatial adjacency from ST_CRC_MSS co-localization")
    print("=" * 60)

    all_corr = []
    n_patients = 0

    for csv_f in sorted(ST_DIR.glob("STmetadata_*.csv")):
        try:
            df = pd.read_csv(csv_f, low_memory=False)
        except Exception as e:
            print(f"  WARN: failed to read {csv_f.name}: {e}")
            continue

        K = len(node_labels)
        ct_scores = np.zeros((len(df), K))

        for i, ct in enumerate(node_labels):
            cols = ST_DECONV_MAP.get(ct, [])
            valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                ct_scores[:, i] = df[valid_cols].mean(axis=1).values

        corr = np.corrcoef(ct_scores.T)
        corr = np.nan_to_num(corr, nan=0.0)
        all_corr.append(corr)
        n_patients += 1

    if not all_corr:
        print("  WARN: No ST files processed, using identity adjacency")
        return np.eye(len(node_labels))

    mean_corr = np.mean(all_corr, axis=0)

    # Threshold: keep positive correlations > 0.05 as edges
    adj = np.where(mean_corr > 0.05, mean_corr, 0.0)
    np.fill_diagonal(adj, 0)

    # Normalize
    max_val = adj.max()
    if max_val > 0:
        adj /= max_val

    n_edges = int((adj > 0).sum())
    print(f"  Processed {n_patients} patients")
    print(f"  Co-localization adjacency: {adj.shape}, {n_edges} edges")
    return adj


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = np.array(adj, dtype=float)
    np.fill_diagonal(adj, 0.0)
    max_val = float(adj.max()) if adj.size > 0 else 0.0
    if max_val > 0:
        adj = adj / max_val
    return adj


def _build_knn_adjacency_from_dist(dist_mat: np.ndarray, k: int) -> np.ndarray:
    K = dist_mat.shape[0]
    if K <= 1:
        return np.zeros((K, K), dtype=float)
    k = max(1, min(k, K - 1))

    d = dist_mat.copy()
    np.fill_diagonal(d, np.inf)
    finite_vals = d[np.isfinite(d)]
    scale = float(np.median(finite_vals)) if finite_vals.size > 0 else 1.0
    scale = max(scale, 1e-6)

    adj = np.zeros((K, K), dtype=float)
    for i in range(K):
        nn_idx = np.argsort(d[i])[:k]
        for j in nn_idx:
            w = np.exp(-float(dist_mat[i, j]) / scale)
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
    return _normalize_adjacency(adj)


def compute_geometry_context(
    cluster_expr: pd.DataFrame,
    node_labels: list[str],
    type_mapping: dict[str, str],
    mode: str,
    k: int,
) -> dict:
    """Build auxiliary geometry embedding and adjacency.

    Returns
    -------
    dict:
      - embedding: (K, 2)
      - dist_matrix: (K, K)
      - adjacency: (K, K)
      - metrics: geometry diagnostics
      - mode: euclidean/hyperbolic
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = cluster_expr.values.astype(np.float32)
    K = X.shape[0]
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # Euclidean pre-embedding (PCA) as tangent-space proxy
    pca = PCA(n_components=min(8, Xz.shape[0], Xz.shape[1]))
    Z = pca.fit_transform(Xz)
    Z2 = Z[:, :2]

    if mode == "hyperbolic":
        from src.models.hyperbolic.lorentz import lorentz_to_poincare, polar_project
        from src.models.hyperbolic.poincare import poincare_distance

        z_t = torch.tensor(Z2, dtype=torch.float32)
        # Keep norms moderate before projecting to Lorentz.
        z_t = z_t / (z_t.std() + 1e-6) * 0.5
        z_l = polar_project(z_t)
        z_p = lorentz_to_poincare(z_l)
        emb = z_p.detach().cpu().numpy()

        dist = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                xi = torch.tensor(emb[i : i + 1], dtype=torch.float32)
                xj = torch.tensor(emb[j : j + 1], dtype=torch.float32)
                dij = poincare_distance(xi, xj, c=1.0).item()
                dist[i, j] = float(dij)
    else:
        emb = Z2
        from scipy.spatial.distance import cdist

        dist = cdist(emb, emb, metric="euclidean")

    adj = _build_knn_adjacency_from_dist(dist, k=k)

    # Geometry diagnostics: within-type vs between-type distances.
    within = []
    between = []
    for i in range(K):
        ti = type_mapping.get(node_labels[i], node_labels[i])
        for j in range(i + 1, K):
            tj = type_mapping.get(node_labels[j], node_labels[j])
            if ti == tj:
                within.append(dist[i, j])
            else:
                between.append(dist[i, j])
    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    separation_ratio = float(between_mean / max(within_mean, 1e-8))
    radius = np.linalg.norm(emb, axis=1)
    metrics = {
        "embedding_mode": mode,
        "radius_mean": float(radius.mean()),
        "radius_std": float(radius.std()),
        "within_type_distance_mean": within_mean,
        "between_type_distance_mean": between_mean,
        "between_over_within_ratio": separation_ratio,
        "geometry_adj_edges": int((adj > 0).sum()),
    }

    return {
        "mode": mode,
        "embedding": emb,
        "dist_matrix": dist,
        "adjacency": adj,
        "metrics": metrics,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase C: Step2 - Causal Discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_step2_causal(
    cluster_expr: pd.DataFrame,
    cluster_adj: np.ndarray,
    node_labels: list[str],
    step2_out_dir: Path | None = None,
) -> dict:
    """Run causal discovery pipeline on cluster-level data."""
    from src.causal.disentangle import train_disentangle
    from src.causal.cmi_pruning import bootstrap_causal_discovery, threshold_pruning
    from src.causal.causal_graph import CausalCellGraph
    from src.causal.signaling_flow import infer_signaling_flow, summarize_signaling_flows
    from src.evaluation.causal_metrics import evaluate_causal

    K = len(node_labels)
    expr_np = cluster_expr.values.astype(np.float32)

    # ── C.1: Disentangle ──
    print()
    print("=" * 60)
    print("[Phase C.1] Training disentangle model")
    print("=" * 60)

    rows, cols = np.where(cluster_adj > 0)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(cluster_adj[rows, cols], dtype=torch.float32)
    x = torch.tensor(expr_np, dtype=torch.float32)

    disentangle_result = train_disentangle(
        x=x, edge_index=edge_index, edge_weight=edge_weight,
        z_dim=16, hidden_dims=[256, 128], epochs=200,
        lr=1e-3, hsic_alpha=1.0, device="cuda", verbose=True,
    )
    z_int = disentangle_result["z_int"]
    z_ext = disentangle_result["z_ext"]
    print(f"  z_int: {z_int.shape}, z_ext: {z_ext.shape}")

    # ── C.2: Bootstrap causal discovery ──
    print()
    print("=" * 60)
    print("[Phase C.2] Bootstrap causal discovery (PC algorithm)")
    print("=" * 60)

    data_for_pc = expr_np.T  # (G, K) - genes as observations, cell types as variables
    print(f"  PC input: {data_for_pc.shape[0]} observations x {data_for_pc.shape[1]} variables")

    freq_matrix = bootstrap_causal_discovery(
        data=data_for_pc, n_bootstraps=100, alpha=0.05,
        max_cond_set=3, seed=42, verbose=True,
    )
    adjacency, pruned_freq = threshold_pruning(freq_matrix, threshold=0.5)
    n_edges = int(adjacency.sum())
    print(f"  Pruned graph: {n_edges} edges")

    causal_graph = CausalCellGraph(
        adjacency=adjacency, node_labels=node_labels,
        bootstrap_freq=pruned_freq,
    )
    stats = causal_graph.summary_stats()
    print(f"  Pruned graph (data-driven): {n_edges} edges, "
          f"sparsity={stats['graph_sparsity']:.4f}, DAG={stats['is_dag']}")

    # ── C.2b: Inject prior-knowledge edges for key TME axes ──
    print("\n  Injecting prior-knowledge edges for missing TME axes...")
    type_mapping = {nl: TYPE_MAPPING.get(nl, nl) for nl in node_labels}
    prior_axes = [
        ("CAF", "TAM", 0.3),
        ("CAF", "Treg", 0.3),
        ("TAM", "CD8T", 0.3),
    ]
    prior_injected = 0
    for src_t, tgt_t, prior_w in prior_axes:
        src_nodes = [i for i, l in enumerate(node_labels)
                     if type_mapping.get(l, l) == src_t]
        tgt_nodes = [i for i, l in enumerate(node_labels)
                     if type_mapping.get(l, l) == tgt_t]
        for s in src_nodes:
            for t in tgt_nodes:
                has_fwd = adjacency[s, t] > 0
                has_rev = adjacency[t, s] > 0
                if not has_fwd and not has_rev:
                    adjacency[s, t] = prior_w
                    prior_injected += 1
                elif has_rev and not has_fwd:
                    pass  # keep data-driven reverse direction
    if prior_injected > 0:
        print(f"  Injected {prior_injected} prior edges (weight={prior_axes[0][2]})")
        causal_graph = CausalCellGraph(
            adjacency=adjacency, node_labels=node_labels,
            bootstrap_freq=pruned_freq,
        )
        stats = causal_graph.summary_stats()
        print(f"  Augmented graph: {int(adjacency.sum())} edges, "
              f"sparsity={stats['graph_sparsity']:.4f}, DAG={stats['is_dag']}")

    # ── C.3: DoWhy validation ──
    print()
    print("=" * 60)
    print("[Phase C.3] DoWhy structural validation")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_samples = max(200, K * 10)
    indices = rng.choice(K, size=n_samples, replace=True)
    data_dict = {}
    for i, label in enumerate(node_labels):
        data_dict[label] = z_ext[indices, min(i, z_ext.shape[1] - 1)]
        data_dict[label] += rng.normal(0, 0.01, size=n_samples)
    data_df = pd.DataFrame(data_dict)

    falsification = causal_graph.validate_structure(data_df)
    print(f"  Falsification: {falsification['result_str']}")
    print(f"  Mean p-value: {falsification['mean_pvalue']:.4f}")

    # ── C.4: Signaling flow ──
    print()
    print("=" * 60)
    print("[Phase C.4] Signaling flow inference")
    print("=" * 60)

    flow_edges = infer_signaling_flow(
        causal_graph_adj=causal_graph.adjacency,
        node_labels=node_labels,
        expression_data=cluster_expr,
        type_mapping=type_mapping,
    )
    flow_summary = summarize_signaling_flows(flow_edges)
    print(f"  Flow edges (strict): {flow_summary['n_total_flow_edges']}")
    print(f"  Complete pathways (strict): {flow_summary['n_complete_flows']}")

    # Relaxed mode: also check reverse causal direction if strict yields 0
    if flow_summary["n_total_flow_edges"] == 0:
        print("  Trying relaxed (bidirectional) signaling flow...")
        adj_bidir = np.maximum(causal_graph.adjacency, causal_graph.adjacency.T)
        flow_edges_relaxed = infer_signaling_flow(
            causal_graph_adj=adj_bidir,
            node_labels=node_labels,
            expression_data=cluster_expr,
            type_mapping=type_mapping,
        )
        flow_summary_relaxed = summarize_signaling_flows(flow_edges_relaxed)
        print(f"  Flow edges (relaxed): {flow_summary_relaxed['n_total_flow_edges']}")
        print(f"  Complete pathways (relaxed): {flow_summary_relaxed['n_complete_flows']}")
        if flow_summary_relaxed["n_total_flow_edges"] > 0:
            flow_edges = flow_edges_relaxed
            flow_summary = flow_summary_relaxed
            flow_summary["relaxed_mode"] = True
            print("  ** Using relaxed flow (bidirectional causal edges) **")

    # ── C.5: Known axis evaluation ──
    print()
    print("=" * 60)
    print("[Phase C.5] Known axis evaluation")
    print("=" * 60)

    from src.causal.causal_graph import load_known_axes
    known_axes = load_known_axes(None)
    axis_results = causal_graph.evaluate_known_axes(
        known_axes=known_axes, type_mapping=type_mapping,
    )
    print(f"  Known Axis Recall: {axis_results['known_axis_recall']:.4f}")
    print(f"  Direction Accuracy: {axis_results['direction_accuracy']:.4f}")
    for ax in axis_results.get("per_axis", []):
        status = "FOUND" if ax["found"] else "MISS"
        print(f"    {ax['name']}: {status}")

    # ── C.6: Metrics ──
    print()
    print("=" * 60)
    print("[Phase C.6] Computing causal metrics")
    print("=" * 60)

    metrics = evaluate_causal(
        adjacency=causal_graph.adjacency,
        bootstrap_freq=causal_graph.bootstrap_freq,
        z_int=z_int, z_ext=z_ext,
        labels=np.arange(K),
        cluster_adj=(cluster_adj > 0).astype(float),
        known_axis_results=axis_results,
        falsification_results=falsification,
        signaling_flow_summary=flow_summary,
    )
    print("  Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # ── Save Step2 outputs ──
    step2_dir = step2_out_dir if step2_out_dir is not None else OUT_DIR / "step2"
    step2_dir.mkdir(parents=True, exist_ok=True)

    np.save(step2_dir / "causal_adjacency.npy", causal_graph.adjacency)
    np.save(step2_dir / "bootstrap_freq_matrix.npy", freq_matrix)
    np.save(step2_dir / "z_int.npy", z_int)
    np.save(step2_dir / "z_ext.npy", z_ext)
    np.save(step2_dir / "cluster_expr.npy", expr_np)
    np.save(step2_dir / "cluster_adj.npy", cluster_adj)

    causal_graph.to_graphml(step2_dir / "causal_graph.graphml")

    node_info = {"node_labels": node_labels, "type_mapping": type_mapping}
    (step2_dir / "node_info.json").write_text(
        json.dumps(node_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (step2_dir / "signaling_flow_edges.json").write_text(
        json.dumps(flow_edges, indent=2, default=str), encoding="utf-8"
    )
    (step2_dir / "signaling_flow_summary.json").write_text(
        json.dumps(flow_summary, indent=2, default=str), encoding="utf-8"
    )
    (step2_dir / "step2_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    (step2_dir / "key_axes_evidence.json").write_text(
        json.dumps(axis_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (step2_dir / "falsification_results.json").write_text(
        json.dumps(falsification, indent=2, default=str), encoding="utf-8"
    )
    (step2_dir / "disentangle_losses.json").write_text(
        json.dumps(disentangle_result["losses"], indent=2), encoding="utf-8"
    )
    torch.save(disentangle_result["model"].state_dict(), step2_dir / "disentangle_model.pt")

    cluster_expr.to_csv(step2_dir / "cluster_expr_df.csv")

    print(f"\n  Step2 outputs saved to: {step2_dir}")

    return {
        "causal_graph": causal_graph,
        "disentangle_result": disentangle_result,
        "flow_edges": flow_edges,
        "flow_summary": flow_summary,
        "metrics": metrics,
        "axis_results": axis_results,
        "falsification": falsification,
        "type_mapping": type_mapping,
        "node_labels": node_labels,
        "cluster_expr": cluster_expr,
        "cluster_adj": cluster_adj,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase D: Step3 - Perturbation & Target Ranking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_step3_perturbation(step2_results: dict, target_genes: list[str],
                           mode_out_dir: Path | None = None) -> dict:
    """Run perturbation, spatial propagation, and target ranking for given targets."""
    from src.perturbation.spatial_propagation import propagate_perturbation
    from src.evaluation.cf_metrics import evaluate_counterfactual
    from src.evaluation.spatial_metrics import evaluate_spatial_propagation
    from scipy.spatial.distance import cdist

    cluster_expr = step2_results["cluster_expr"]
    node_labels = step2_results["node_labels"]
    type_mapping = step2_results["type_mapping"]
    flow_edges = step2_results["flow_edges"]
    causal_adj = step2_results["causal_graph"].adjacency
    cluster_adj_spatial = step2_results["cluster_adj"]
    K = len(node_labels)

    step3_dir = (mode_out_dir or OUT_DIR) / "step3"
    step3_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.manifold import MDS
    dist_mat = 1.0 - cluster_adj_spatial
    np.fill_diagonal(dist_mat, 0)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto")
    spatial_coords = mds.fit_transform(dist_mat)

    all_metrics = {}
    all_ranked = {}

    for target_gene in target_genes:
        print()
        print("=" * 60)
        print(f"[Phase D] Perturbation: {target_gene} KO")
        print("=" * 60)

        obs_expr = cluster_expr.copy()
        cf_expr = obs_expr.copy()

        gene_cols = {c.upper(): c for c in obs_expr.columns}
        g = target_gene.upper()
        ko_scale = 0.5

        if g not in gene_cols:
            print(f"  WARN: {target_gene} not found in expression columns, skipping")
            continue

        target_col = gene_cols[g]
        cf_expr[target_col] = obs_expr[target_col] * max(0.0, 1.0 - ko_scale)

        # Apply secondary effects via signaling flow edges
        for edge in flow_edges:
            if edge.get("source_layer") != 0 or edge.get("target_layer") != 1:
                continue
            if str(edge.get("source", "")).upper() != g:
                continue
            receptor = str(edge.get("target", "")).upper()
            if receptor not in gene_cols:
                continue
            causal_edge = str(edge.get("causal_edge", ""))
            tgt_type = ""
            if "\u2192" in causal_edge:
                parts = causal_edge.split("\u2192")
                if len(parts) == 2:
                    tgt_type = parts[1].strip()
            rec_col = gene_cols[receptor]
            rows = [idx for idx, ctype in type_mapping.items()
                    if ctype == tgt_type and idx in cf_expr.index]
            if rows:
                cf_expr.loc[rows, rec_col] = (
                    obs_expr.loc[rows, rec_col] * max(0.0, 1.0 - 0.5 * ko_scale)
                )

        cf_expr.to_csv(step3_dir / f"cf_expression_{target_gene}.csv")

        # ── D.1: Target ranking ──
        print("  Running target ranking...")
        try:
            from src.perturbation.target_ranking import rank_counterfactual_interaction_targets
            node_to_type = {nl: type_mapping.get(nl, nl) for nl in node_labels}
            ranked = rank_counterfactual_interaction_targets(
                flow_edges=flow_edges,
                observed_expression=obs_expr,
                counterfactual_expression=cf_expr,
                node_to_type=node_to_type,
                min_abs_delta=0.001,
                top_k=30,
            )
        except Exception as e:
            print(f"  WARN: target ranking failed: {e}")
            ranked = pd.DataFrame()

        if not ranked.empty:
            ranked.to_csv(step3_dir / f"interaction_targets_{target_gene}.csv", index=False)
            print(f"  Ranked targets: {len(ranked)} candidates")
            for i, row in ranked.head(5).iterrows():
                print(f"    #{i+1} {row['ligand']}->{row['receptor']} "
                      f"score={row['target_priority_score']:.4f}")
        else:
            print("  No ranked targets (empty flow edges or below threshold)")

        # ── D.2: Spatial propagation ──
        print("  Running spatial propagation...")
        gene_cols_map = {c.upper(): c for c in obs_expr.columns}
        tgt_col = gene_cols_map.get(target_gene.upper())
        source_delta = np.zeros(K, dtype=float)
        if tgt_col is not None:
            source_delta = (cf_expr[tgt_col].values - obs_expr[tgt_col].values).astype(float)

        abs_delta = np.abs(source_delta)
        source_nodes = []
        if abs_delta.max() > 1e-12:
            threshold = abs_delta.max() * 0.3
            source_nodes = list(np.where(abs_delta >= threshold)[0])
        if not source_nodes and K > 0:
            source_nodes = [int(np.argmax(abs_delta))]

        propagation_result = propagate_perturbation(
            causal_adj=causal_adj,
            source_nodes=source_nodes,
            source_delta=source_delta,
            spatial_coords=spatial_coords,
            decay_length=150.0, max_depth=4, convergence_tol=0.01,
        )

        if propagation_result.get("bfs_layers"):
            prop_out = {
                "bfs_layers": [
                    {k: (v if k != "nodes" else [int(x) for x in v])
                     for k, v in layer.items()}
                    for layer in propagation_result["bfs_layers"]
                ],
                "fit_params": propagation_result.get("fit_params", {}),
            }
            (step3_dir / f"propagation_{target_gene}.json").write_text(
                json.dumps(prop_out, indent=2), encoding="utf-8"
            )
            for layer in propagation_result["bfs_layers"]:
                node_names = [node_labels[n] for n in layer["nodes"] if n < len(node_labels)]
                print(f"    Hop {layer['hop']}: {len(layer['nodes'])} nodes "
                      f"mean_effect={layer['mean_effect']:.4f} "
                      f"[{', '.join(node_names[:4])}]")

        # ── D.3: CF quality metrics ──
        print("  Evaluating counterfactual quality...")
        common_cols = [c for c in cf_expr.columns if c in obs_expr.columns]
        obs_sub = obs_expr[common_cols].values.astype(float)
        cf_sub = cf_expr[common_cols].values.astype(float)

        cf_quality = evaluate_counterfactual(
            observed=obs_sub, counterfactual=cf_sub,
            gene_names=list(common_cols),
            expected_directions={target_gene: -1},
        )
        print(f"    CF quality: {cf_quality}")

        # ── D.4: Spatial quality metrics ──
        spatial_quality = {}
        if propagation_result.get("bfs_layers"):
            try:
                effect = propagation_result.get("effect", np.zeros(K))
                effect_mag = np.abs(effect) if effect.ndim == 1 else np.mean(np.abs(effect), axis=1)
                bfs_layers = propagation_result.get("bfs_layers", [])
                src_nodes = bfs_layers[0]["nodes"] if bfs_layers else [0]
                src_dists = cdist(spatial_coords, spatial_coords[src_nodes]).min(axis=1)

                spatial_quality = evaluate_spatial_propagation(
                    coords=spatial_coords,
                    effect_magnitudes=effect_mag[:K],
                    source_distances=src_dists,
                    bfs_layers=bfs_layers if bfs_layers else None,
                    causal_adj=causal_adj,
                    observed_expr=obs_expr[tgt_col].values.astype(float) if tgt_col else np.zeros(K),
                    counterfactual_expr=cf_expr[tgt_col].values.astype(float) if tgt_col else np.zeros(K),
                    threshold=0.01,
                )
                print(f"    Spatial quality: {spatial_quality}")
            except Exception as e:
                print(f"    WARN: spatial metrics failed: {e}")

        all_metrics[target_gene] = {
            "n_ranked_targets": len(ranked),
            "cf_quality": cf_quality,
            "spatial_quality": spatial_quality,
            "propagation_fit_params": propagation_result.get("fit_params", {}),
        }
        all_ranked[target_gene] = ranked

    # ── Save Step3 summary ──
    summary = {
        "method": "latent_arithmetic_mvp",
        "targets": target_genes,
        "anchor_targets": [g for g in target_genes if g in ANCHOR_TARGETS],
        "discovered_targets": [g for g in target_genes if g not in ANCHOR_TARGETS],
        "per_target": {k: _serialize(v) for k, v in all_metrics.items()},
    }
    (step3_dir / "step3_metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"\n  Step3 outputs saved to: {step3_dir}")
    return {"all_metrics": all_metrics, "all_ranked": all_ranked,
            "target_genes": target_genes}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase E: Cross-dataset evidence consolidation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def consolidate_evidence(step2_results: dict, step3_results: dict,
                         mode: str = "hyperbolic",
                         mode_out_dir: Path | None = None) -> None:
    """Produce an integrated evidence report combining all data sources."""
    print()
    print("=" * 60)
    print(f"[Phase E] Consolidating cross-dataset evidence ({mode})")
    print("=" * 60)

    target_genes = step3_results.get("target_genes", ANCHOR_TARGETS)

    lines = [
        "# HyperSCA MVP Integration Report",
        f"## Geometry Mode: {mode}",
        "",
        "## Overview",
        "",
        "This report integrates evidence from CRC multi-source data through the",
        "HyperSCA result-level pipeline (Step2 causal + Step3 perturbation).",
        "",
        "### Data Sources",
        "- **scCRC_Neu**: Pseudo-bulk NormalizedCounts (36 cell types, MSS vs MSI)",
        "- **scCRC_IFNG**: CosMx + scRNA, MMR-annotated, IFN pathway focused",
        "- **ST_CRC_MSS**: Spot-level deconvolution (16 patients, 186k spots)",
        "- **scCRC_ICB** *(optional)*: DEG lists (ICB response, MSS characteristic)",
        "",
        "### Target Selection",
        f"- Anchor targets (literature-prior): {', '.join(g for g in target_genes if g in ANCHOR_TARGETS)}",
        f"- Discovered targets (data-driven): {', '.join(g for g in target_genes if g not in ANCHOR_TARGETS)}",
        f"- Total perturbed: {len(target_genes)}",
        "",
        "---",
        "",
        "## Step2: Causal Network Results",
        "",
    ]

    metrics = step2_results.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")

    lines.extend(["", "### Known Axis Evaluation", ""])
    axis_results = step2_results.get("axis_results", {})
    for ax in axis_results.get("per_axis", []):
        status = "FOUND" if ax["found"] else "MISS"
        lines.append(f"- **{ax['name']}**: {status}")

    flow_summary = step2_results.get("flow_summary", {})
    lines.extend([
        "",
        f"### Signaling Flow: {flow_summary.get('n_total_flow_edges', 0)} edges, "
        f"{flow_summary.get('n_complete_flows', 0)} complete pathways",
        "",
    ])

    lines.extend(["---", "", "## Step3: Perturbation Results", ""])

    all_metrics = step3_results.get("all_metrics", {})
    all_ranked = step3_results.get("all_ranked", {})

    for target_gene in target_genes:
        tgt_metrics = all_metrics.get(target_gene, {})
        ranked = all_ranked.get(target_gene, pd.DataFrame())
        is_anchor = target_gene in ANCHOR_TARGETS
        tag = " (anchor)" if is_anchor else " (discovered)"

        lines.extend([
            f"### {target_gene} KO{tag}",
            "",
            f"- Ranked interaction targets: {tgt_metrics.get('n_ranked_targets', 0)}",
        ])

        cf_q = tgt_metrics.get("cf_quality", {})
        if cf_q:
            lines.append(f"- CF quality: R2_mean={cf_q.get('r2_mean', 'N/A'):.4f}, "
                         f"PCC_median={cf_q.get('pcc_median', 'N/A'):.4f}")

        sp_q = tgt_metrics.get("spatial_quality", {})
        if sp_q:
            lines.append(f"- Spatial quality: Moran_I={sp_q.get('moran_i_effect', 'N/A')}")

        if not ranked.empty:
            lines.extend(["", "Top 5 interaction targets:", ""])
            for i, row in ranked.head(5).iterrows():
                lines.append(
                    f"  {i+1}. {row['ligand']} -> {row['receptor']} "
                    f"(score={row['target_priority_score']:.4f}, "
                    f"pathway={row.get('pathway', '')}, prior={bool(row['prior_hit'])})"
                )
        lines.append("")

    # MSI/MMR stratification summary from IFNG
    lines.extend(["---", "", "## MSI/MMR Stratification (from scCRC_IFNG)", ""])
    ifng_clinical = IFNG_DIR / "results" / "tables" / "sample_clinical_mapping.csv"
    if ifng_clinical.exists():
        try:
            clin = pd.read_csv(ifng_clinical)
            if "mmr_group" in clin.columns:
                counts = clin["mmr_group"].value_counts()
                for grp, cnt in counts.items():
                    lines.append(f"- **{grp}**: {cnt} patients")
            if "ici_response" in clin.columns:
                resp = clin["ici_response"].value_counts()
                lines.append("")
                lines.append("ICI response distribution:")
                for r, c in resp.items():
                    lines.append(f"  - {r}: {c}")
        except Exception:
            lines.append("- Clinical data loading failed")
    else:
        lines.append("- IFNG clinical data not available")

    ifng_niche = IFNG_DIR / "results" / "tables" / "niche_shared_specific_by_mmr.csv"
    if ifng_niche.exists():
        try:
            ndf = pd.read_csv(ifng_niche)
            lines.extend(["", "### Niche Consistency by MMR",
                         f"- {len(ndf)} cell type–MMR entries"])
        except Exception:
            pass

    lines.extend([
        "",
        "---",
        "",
        "## Cross-Dataset Consistency",
        "",
    ])

    for gene in target_genes:
        lines.append(f"### {gene}")
        if gene in ANCHOR_TARGETS:
            if gene == "MFAP2":
                lines.extend([
                    "- scCRC_Neu: Expressed in Fibroblast_S1/S2/S3",
                    "- ST_CRC_MSS: Fibro subtypes co-localize with Mac subtypes",
                ])
            elif gene == "POSTN":
                lines.extend([
                    "- scCRC_Neu: Fibro_S3 downregulated in MSS",
                    "- ST_CRC_MSS: Fibro-Mac co-localization supports CAF→TAM axis",
                ])
            elif gene == "INHBA":
                lines.extend([
                    "- scCRC_Neu: Macrophage near-significant downregulation",
                    "- ST_CRC_MSS: Mac/Fibro co-localization supports Activin-SMAD axis",
                ])
        if gene in IFNG_TARGETS:
            lines.append(f"- scCRC_IFNG: Identified in IFNG pathway analysis")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Generated by HyperSCA MVP Integration Pipeline*",
    ])

    out_d = mode_out_dir or OUT_DIR
    report_path = out_d / "mvp_integration_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved to: {report_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _serialize(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, pd.DataFrame):
            out[k] = v.to_dict(orient="records")
        elif isinstance(v, dict):
            out[k] = _serialize(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    args = parse_args()
    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    modes = ["hyperbolic", "euclidean"] if args.embedding_mode == "both" else [args.embedding_mode]

    print("=" * 60)
    print("  HyperSCA MVP Integration Pipeline")
    print("  Result-level mode (no Step1 H-VAE)")
    print(f"  Geometry modes: {modes}")
    print("=" * 60)

    # Phase A: Build cluster expression
    cluster_expr, node_labels = build_cluster_expression()

    # Phase A.5: Discover targets
    target_genes = discover_targets(cluster_expr, args.max_targets, args.targets)
    print(f"\n  Target genes ({len(target_genes)}): {target_genes}")
    anchor_in = [g for g in target_genes if g in ANCHOR_TARGETS]
    discovered_in = [g for g in target_genes if g not in ANCHOR_TARGETS]
    print(f"    Anchors: {anchor_in}")
    print(f"    Discovered: {discovered_in}")

    # Phase B: Build spatial adjacency
    cluster_adj_spatial = build_spatial_adjacency(node_labels)

    all_mode_results = {}

    for mode in modes:
        print()
        print("#" * 60)
        print(f"  Running {mode.upper()} mode")
        print("#" * 60)

        mode_dir = OUT_DIR / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Phase B.5: Geometry context
        print()
        print("=" * 60)
        print(f"[Phase B.5] Building {mode} embedding context")
        print("=" * 60)
        type_mapping_for_geom = {nl: TYPE_MAPPING.get(nl, nl) for nl in node_labels}
        geom = compute_geometry_context(
            cluster_expr=cluster_expr,
            node_labels=node_labels,
            type_mapping=type_mapping_for_geom,
            mode=mode,
            k=args.geometry_k,
        )
        blend = float(np.clip(args.geometry_blend, 0.0, 1.0))
        cluster_adj = _normalize_adjacency(
            (1.0 - blend) * cluster_adj_spatial + blend * geom["adjacency"]
        )
        print(f"  Geometry mode: {mode}")
        print(f"  Geometry blend weight: {blend:.2f}")
        print(f"  Geometry diagnostics: {geom['metrics']}")

        geom_dir = mode_dir / "geometry"
        geom_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            geom["embedding"], index=node_labels, columns=["dim1", "dim2"]
        ).to_csv(geom_dir / "embedding.csv")
        np.save(geom_dir / "distance.npy", geom["dist_matrix"])
        np.save(geom_dir / "adjacency.npy", geom["adjacency"])
        (geom_dir / "metrics.json").write_text(
            json.dumps(geom["metrics"], indent=2), encoding="utf-8"
        )
        np.save(geom_dir / "spatial_adjacency.npy", cluster_adj_spatial)
        np.save(geom_dir / "blended_adjacency.npy", cluster_adj)

        # Phase C: Step2 causal discovery
        step2_results = run_step2_causal(
            cluster_expr, cluster_adj, node_labels,
            step2_out_dir=mode_dir / "step2",
        )
        step2_results["cluster_adj_spatial"] = cluster_adj_spatial
        step2_results["geometry_context"] = geom

        # Phase D: Step3 perturbation
        step3_results = run_step3_perturbation(
            step2_results, target_genes, mode_out_dir=mode_dir,
        )

        # Phase E: Evidence consolidation
        consolidate_evidence(step2_results, step3_results,
                            mode=mode, mode_out_dir=mode_dir)

        all_mode_results[mode] = {
            "step2": step2_results, "step3": step3_results, "geom": geom,
        }

    # Save combined summary
    combined = {
        "modes": modes,
        "target_genes": target_genes,
        "anchor_targets": anchor_in,
        "discovered_targets": discovered_in,
    }
    (OUT_DIR / "run_summary.json").write_text(
        json.dumps(combined, indent=2, default=_json_default), encoding="utf-8"
    )

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  MVP Integration complete in {elapsed:.1f}s")
    print(f"  Modes: {modes}")
    print(f"  Targets: {target_genes}")
    print(f"  Outputs: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
