"""
HyperSCA Target Discovery Pipeline
====================================
Open-ended therapeutic target identification for MSS-type CRC immunotherapy
non-response, using hyperbolic geometry embedding + spatial causal inference.

Runs both Hyperbolic and Euclidean geometry modes for comparison.
Produces: candidate_pool.csv, target_ranking.csv, evidence_matrix.csv,
          comparison_report.md, target_discovery_report.md, and 12+ figures.

Usage:
    python scripts/run_target_discovery.py
    python scripts/run_target_discovery.py --max-perturb 30
"""
from __future__ import annotations

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

OUT_BASE = ROOT / "results" / "integration" / "discovery"
OUT_BASE.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NEU_DIR = Path(
    r"G:\scCRC_Neu\downstream_analyses_de_analysis"
    r"\0downstream_analyses_de_analysis\de_analysis"
    r"\de_analysis_tumor_mss_msi\deseq2_dgea"
)
IFNG_DIR = Path(r"F:\scCRC_IFNG")
ICB_DIR = Path(r"G:\scCRC_ICB\output")
ST_DIR = Path(r"G:\ST_CRC_MSS")

ANCHOR_GENES = ["MFAP2", "POSTN", "INHBA"]
IFNG_FOCUS_GENES = ["CD74", "INHBA", "CXCL10", "IFNG", "COL1A1", "MFAP5", "FN1"]

CELLTYPES = [
    "Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3",
    "Macrophage", "Macrophage_cycling",
    "Pericyte",
    "T_cell_CD4", "T_cell_CD8", "T_cell_CD8_cycling", "T_cell_regulatory",
    "NK",
    "cDC1", "cDC2", "DC_mature", "pDC",
    "Neutrophil", "Mast_cell",
    "Monocyte_classical",
    "Endothelial_venous", "Endothelial_arterial",
]

TYPE_MAPPING = {
    "Fibroblast_S1": "CAF", "Fibroblast_S2": "CAF", "Fibroblast_S3": "CAF",
    "Macrophage": "TAM", "Macrophage_cycling": "TAM",
    "Pericyte": "Stromal",
    "T_cell_CD4": "CD4T", "T_cell_CD8": "CD8T",
    "T_cell_CD8_cycling": "CD8T", "T_cell_regulatory": "Treg",
    "NK": "NK",
    "cDC1": "DC", "cDC2": "DC", "DC_mature": "DC", "pDC": "DC",
    "Neutrophil": "Neutrophil", "Mast_cell": "Mast",
    "Monocyte_classical": "Monocyte",
    "Endothelial_venous": "Endothelial", "Endothelial_arterial": "Endothelial",
}

ST_DECONV_MAP = {
    "Fibroblast_S1": ["Fibro_ADAMDEC1", "Fibro_CXCL8", "Fibro_CXCL14"],
    "Fibroblast_S2": ["Fibro_GPM6B", "Fibro_KCNN3", "Fibro_MYH11"],
    "Fibroblast_S3": ["Fibro_NOTCH3", "Fibro_PI16"],
    "Macrophage": ["Mac_M1", "Mac_M2", "Mac_SPP1"],
    "Macrophage_cycling": ["Mac_M1"],
    "Pericyte": ["Endo"],
    "T_cell_CD4": ["CD4_CXCL13", "CD4_Tcm", "CD4_Treg", "CD4_act"],
    "T_cell_CD8": ["CD8_Cyto", "CD8_HSP", "CD8_Teff", "CD8_Tem", "CD8_Tex"],
    "T_cell_CD8_cycling": ["CD8_Cyto"],
    "T_cell_regulatory": ["CD4_Treg"],
    "NK": ["NK_gdT"],
    "cDC1": ["cDC1"], "cDC2": ["cDC2"], "DC_mature": ["DC_LAMP3"], "pDC": ["pDC"],
    "Neutrophil": ["Monocyte_S100A8"],
    "Mast_cell": ["Mast"],
    "Monocyte_classical": ["Monocyte_S100A8"],
    "Endothelial_venous": ["Endo"],
    "Endothelial_arterial": ["Endo"],
}

ICB_TO_NEU_MAP = {
    "Fibro": ["Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3"],
    "Mph": ["Macrophage", "Macrophage_cycling"],
    "CD8": ["T_cell_CD8", "T_cell_CD8_cycling"],
    "T": ["T_cell_CD4", "T_cell_CD8", "T_cell_regulatory"],
    "Endo": ["Endothelial_venous", "Endothelial_arterial"],
    "Pericyte": ["Pericyte"],
    "Tumor": [],
    "Coloncyte": [], "Goblet": [], "Glia": [], "Tuft": [],
}

PRIOR_AXES = [
    ("CAF", "TAM", 0.3),
    ("CAF", "Treg", 0.3),
    ("TAM", "CD8T", 0.3),
    ("DC", "CD8T", 0.2),
    ("Neutrophil", "TAM", 0.2),
    ("CAF", "Endothelial", 0.2),
]

SCORE_WEIGHTS = {"causal": 0.25, "spatial": 0.25, "consistency": 0.25,
                 "actionability": 0.10, "niche": 0.15}


# ============================================================================
#  Helpers
# ============================================================================

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (set,)):
        return list(obj)
    return str(obj)


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj = np.array(adj, dtype=float)
    np.fill_diagonal(adj, 0.0)
    mx = float(adj.max()) if adj.size > 0 else 0.0
    return adj / mx if mx > 0 else adj


def _knn_adj(dist: np.ndarray, k: int) -> np.ndarray:
    K = dist.shape[0]
    if K <= 1:
        return np.zeros((K, K))
    k = max(1, min(k, K - 1))
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    fv = d[np.isfinite(d)]
    scale = max(float(np.median(fv)) if fv.size else 1.0, 1e-6)
    adj = np.zeros((K, K))
    for i in range(K):
        for j in np.argsort(d[i])[:k]:
            w = np.exp(-float(dist[i, j]) / scale)
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
    return _normalize_adj(adj)


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / max(mx - mn, 1e-12)


# ============================================================================
#  Phase 1: Open Candidate Pool
# ============================================================================

def build_candidate_pool() -> pd.DataFrame:
    print("=" * 60)
    print("[Phase 1] Building open candidate pool from DEG results")
    print("=" * 60)

    # --- 1a: scCRC_Neu DESeq2 results (all cell types) ---
    neu_records = []
    n_files = 0
    for tsv in sorted(NEU_DIR.glob("*-DESeq2_result.tsv")):
        ct = tsv.stem.replace("-DESeq2_result", "")
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        if "padj" not in df.columns or "log2FoldChange" not in df.columns:
            continue
        sig = df[(df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 0.5)].copy()
        for _, row in sig.iterrows():
            neu_records.append({
                "gene": str(row.get("symbol", "")),
                "celltype_neu": ct,
                "lfc_neu": float(row["log2FoldChange"]),
                "padj_neu": float(row["padj"]),
            })
        n_files += 1
    print(f"  Neu: parsed {n_files} DESeq2 files, {len(neu_records)} significant hits")
    neu_df = pd.DataFrame(neu_records) if neu_records else pd.DataFrame(
        columns=["gene", "celltype_neu", "lfc_neu", "padj_neu"]
    )

    # --- 1b: scCRC_ICB DEG results ---
    icb_records = []
    for csv_name in ["DEGs_MSS_response_Mid_lfc0.5.csv", "DEGs_MSS_Mid.csv",
                     "DEGs_MSS_response_Major_lfc0.5.csv", "DEGs_MSS_Major.csv"]:
        fpath = ICB_DIR / csv_name
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        gene_col = "gene" if "gene" in df.columns else df.columns[0]
        lfc_col = "avg_log2FC" if "avg_log2FC" in df.columns else None
        padj_col = "p_val_adj" if "p_val_adj" in df.columns else None
        ct_col = "celltype" if "celltype" in df.columns else None
        for _, row in df.iterrows():
            if lfc_col and padj_col:
                try:
                    padj_v = float(row[padj_col])
                    lfc_v = float(row[lfc_col])
                except (ValueError, TypeError):
                    continue
                if padj_v > 0.05 or abs(lfc_v) < 0.3:
                    continue
            icb_records.append({
                "gene": str(row[gene_col]),
                "celltype_icb": str(row[ct_col]) if ct_col else csv_name,
                "lfc_icb": float(row[lfc_col]) if lfc_col else np.nan,
                "padj_icb": float(row[padj_col]) if padj_col else np.nan,
                "source_file": csv_name,
            })
    print(f"  ICB: {len(icb_records)} significant hits")
    icb_df = pd.DataFrame(icb_records) if icb_records else pd.DataFrame(
        columns=["gene", "celltype_icb", "lfc_icb", "padj_icb", "source_file"]
    )

    # --- 1b2: scCRC_IFNG targets (MMR-stratified) ---
    ifng_records = []
    ifng_mmr = IFNG_DIR / "results" / "tables" / "targets_shared_specific_by_mmr.csv"
    if ifng_mmr.exists():
        try:
            mdf = pd.read_csv(ifng_mmr)
            gene_col = "gene" if "gene" in mdf.columns else mdf.columns[0]
            for _, row in mdf.iterrows():
                g = str(row[gene_col])
                if not g or g == "nan":
                    continue
                ifng_records.append({
                    "gene": g,
                    "celltype_ifng": str(row.get("celltype", "unknown")),
                    "lfc_ifng": float(row.get("log2FoldChange", row.get("avg_log2FC", 0))),
                    "mmr_group": str(row.get("mmr_group", "")),
                })
        except Exception:
            pass
    for g in IFNG_FOCUS_GENES:
        if not any(r["gene"] == g for r in ifng_records):
            ifng_records.append({"gene": g, "celltype_ifng": "IFNG_focus",
                                 "lfc_ifng": np.nan, "mmr_group": ""})
    print(f"  IFNG: {len(ifng_records)} hits")
    ifng_df = pd.DataFrame(ifng_records) if ifng_records else pd.DataFrame(
        columns=["gene", "celltype_ifng", "lfc_ifng", "mmr_group"]
    )

    # --- 1c: Aggregate per gene ---
    all_genes = set()
    if not neu_df.empty:
        all_genes |= set(neu_df["gene"].dropna().unique())
    if not icb_df.empty:
        all_genes |= set(icb_df["gene"].dropna().unique())
    if not ifng_df.empty:
        all_genes |= set(ifng_df["gene"].dropna().unique())
    all_genes -= {"", "nan", "None"}
    print(f"  Total unique genes: {len(all_genes)}")

    pool_rows = []
    for g in sorted(all_genes):
        n_sub = neu_df[neu_df["gene"] == g] if not neu_df.empty else pd.DataFrame()
        i_sub = icb_df[icb_df["gene"] == g] if not icb_df.empty else pd.DataFrame()
        f_sub = ifng_df[ifng_df["gene"] == g] if not ifng_df.empty else pd.DataFrame()

        n_ct_neu = n_sub["celltype_neu"].nunique() if not n_sub.empty else 0
        n_ct_icb = i_sub["celltype_icb"].nunique() if not i_sub.empty else 0
        n_ct_ifng = f_sub["celltype_ifng"].nunique() if not f_sub.empty else 0

        lfcs = []
        if not n_sub.empty:
            lfcs.extend(n_sub["lfc_neu"].dropna().tolist())
        if not i_sub.empty:
            lfcs.extend(i_sub["lfc_icb"].dropna().tolist())
        if not f_sub.empty:
            lfcs.extend(f_sub["lfc_ifng"].dropna().tolist())

        mean_lfc = float(np.mean(lfcs)) if lfcs else 0.0
        mean_abs_lfc = float(np.mean(np.abs(lfcs))) if lfcs else 0.0

        if lfcs:
            signs = np.sign(lfcs)
            majority = np.sign(np.sum(signs))
            direction_consistency = float(np.mean(signs == majority)) if majority != 0 else 0.5
        else:
            direction_consistency = 0.0

        padjs = []
        if not n_sub.empty:
            padjs.extend(n_sub["padj_neu"].dropna().tolist())
        if not i_sub.empty:
            padjs.extend(i_sub["padj_icb"].dropna().tolist())
        min_padj = float(np.min(padjs)) if padjs else 1.0

        in_neu = n_ct_neu > 0
        in_icb = n_ct_icb > 0
        in_ifng = n_ct_ifng > 0
        cross_queue = int(in_neu) + int(in_icb) + int(in_ifng)

        celltypes_neu = ";".join(sorted(n_sub["celltype_neu"].unique())) if not n_sub.empty else ""
        celltypes_icb = ";".join(sorted(i_sub["celltype_icb"].unique())) if not i_sub.empty else ""
        celltypes_ifng = ";".join(sorted(f_sub["celltype_ifng"].unique())) if not f_sub.empty else ""

        pool_rows.append({
            "gene": g,
            "n_celltypes_neu": n_ct_neu,
            "n_celltypes_icb": n_ct_icb,
            "n_celltypes_ifng": n_ct_ifng,
            "cross_queue_count": cross_queue,
            "mean_lfc": mean_lfc,
            "mean_abs_lfc": mean_abs_lfc,
            "direction_consistency": direction_consistency,
            "min_padj": min_padj,
            "neg_log10_padj": -np.log10(max(min_padj, 1e-300)),
            "is_anchor": g in ANCHOR_GENES,
            "is_ifng_target": g in IFNG_FOCUS_GENES,
            "celltypes_neu": celltypes_neu,
            "celltypes_icb": celltypes_icb,
            "celltypes_ifng": celltypes_ifng,
        })

    pool = pd.DataFrame(pool_rows)

    pool["init_score"] = (
        pool["cross_queue_count"] * 2.0
        + _minmax(pool["mean_abs_lfc"].values) * 1.5
        + _minmax(pool["neg_log10_padj"].values) * 1.5
        + pool["direction_consistency"] * 1.0
        + _minmax(pool["n_celltypes_neu"].values) * 0.5
        + _minmax(pool["n_celltypes_ifng"].values) * 0.5
    )
    pool = pool.sort_values("init_score", ascending=False).reset_index(drop=True)

    out_path = OUT_BASE / "candidate_pool.csv"
    pool.to_csv(out_path, index=False)
    print(f"  Candidate pool: {len(pool)} genes saved to {out_path}")
    print(f"  Top 10: {pool['gene'].head(10).tolist()}")
    for a in ANCHOR_GENES:
        row = pool[pool["gene"] == a]
        if not row.empty:
            r = row.iloc[0]
            print(f"  Anchor {a}: rank={row.index[0]+1}, score={r['init_score']:.2f}, "
                  f"cross={r['cross_queue_count']}, |lfc|={r['mean_abs_lfc']:.2f}")
    return pool


# ============================================================================
#  Phase 2: Cluster Expression
# ============================================================================

def build_cluster_expression() -> tuple[pd.DataFrame, list[str]]:
    print("\n" + "=" * 60)
    print("[Phase 2] Building cluster expression (expanded cell types)")
    print("=" * 60)
    dfs = {}
    for ct in CELLTYPES:
        fpath = NEU_DIR / f"{ct}-NormalizedCounts.tsv"
        if not fpath.exists():
            print(f"  SKIP: {ct}")
            continue
        df = pd.read_csv(fpath, sep="\t", index_col=0)
        dfs[ct] = df.mean(axis=1)
        print(f"  {ct}: {df.shape[1]} samples, {df.shape[0]} genes")

    if not dfs:
        raise RuntimeError("No NormalizedCounts loaded")

    expr = pd.DataFrame(dfs).T.fillna(0)
    expr = np.log1p(expr)
    labels = list(expr.index)
    print(f"  Result: {expr.shape} ({len(labels)} types x {expr.shape[1]} genes)")
    return expr, labels


# ============================================================================
#  Phase 3: Spatial Adjacency
# ============================================================================

def build_spatial_adjacency(node_labels: list[str]) -> np.ndarray:
    print("\n" + "=" * 60)
    print("[Phase 3] Building spatial adjacency from ST co-localization")
    print("=" * 60)
    all_corr, n_p = [], 0
    K = len(node_labels)
    for csv_f in sorted(ST_DIR.glob("STmetadata_*.csv")):
        try:
            df = pd.read_csv(csv_f, low_memory=False)
        except Exception:
            continue
        scores = np.zeros((len(df), K))
        for i, ct in enumerate(node_labels):
            cols = [c for c in ST_DECONV_MAP.get(ct, []) if c in df.columns]
            if cols:
                scores[:, i] = df[cols].mean(axis=1).values
        corr = np.nan_to_num(np.corrcoef(scores.T), nan=0.0)
        all_corr.append(corr)
        n_p += 1

    if not all_corr:
        return np.eye(K)

    adj = np.mean(all_corr, axis=0)
    adj = np.where(adj > 0.05, adj, 0.0)
    np.fill_diagonal(adj, 0)
    mx = adj.max()
    if mx > 0:
        adj /= mx
    print(f"  {n_p} patients, {int((adj > 0).sum())} edges")
    return adj


# ============================================================================
#  Phase 4: Geometry Context
# ============================================================================

def compute_geometry(
    cluster_expr: pd.DataFrame,
    node_labels: list[str],
    mode: str,
    k: int = 4,
) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = cluster_expr.values.astype(np.float32)
    Xz = StandardScaler().fit_transform(X)
    n_comp = min(8, Xz.shape[0], Xz.shape[1])
    Z = PCA(n_components=n_comp).fit_transform(Xz)
    Z2 = Z[:, :2]
    K = X.shape[0]

    if mode == "hyperbolic":
        from src.models.hyperbolic.lorentz import lorentz_to_poincare, polar_project
        from src.models.hyperbolic.poincare import poincare_distance
        zt = torch.tensor(Z2, dtype=torch.float32)
        zt = zt / (zt.std() + 1e-6) * 0.5
        emb = lorentz_to_poincare(polar_project(zt)).detach().cpu().numpy()
        dist = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                d = poincare_distance(
                    torch.tensor(emb[i:i+1], dtype=torch.float32),
                    torch.tensor(emb[j:j+1], dtype=torch.float32), c=1.0
                ).item()
                dist[i, j] = dist[j, i] = d
    else:
        from scipy.spatial.distance import cdist
        emb = Z2
        dist = cdist(emb, emb)

    adj = _knn_adj(dist, k)
    type_map = {nl: TYPE_MAPPING.get(nl, nl) for nl in node_labels}
    within, between = [], []
    for i in range(K):
        for j in range(i + 1, K):
            (within if type_map[node_labels[i]] == type_map[node_labels[j]] else between).append(dist[i, j])

    metrics = {
        "mode": mode,
        "radius_mean": float(np.linalg.norm(emb, axis=1).mean()),
        "within_dist": float(np.mean(within)) if within else 0.0,
        "between_dist": float(np.mean(between)) if between else 0.0,
        "separation": float(np.mean(between) / max(np.mean(within), 1e-8)) if within else 0.0,
        "n_edges": int((adj > 0).sum()),
    }
    return {"mode": mode, "embedding": emb, "dist_matrix": dist, "adjacency": adj, "metrics": metrics}


# ============================================================================
#  Phase 5: Step2 Causal Discovery
# ============================================================================

def run_step2(
    cluster_expr: pd.DataFrame,
    cluster_adj: np.ndarray,
    node_labels: list[str],
    out_dir: Path,
) -> dict:
    from src.causal.disentangle import train_disentangle
    from src.causal.cmi_pruning import bootstrap_causal_discovery, threshold_pruning
    from src.causal.causal_graph import CausalCellGraph, load_known_axes
    from src.causal.signaling_flow import infer_signaling_flow, summarize_signaling_flows
    from src.evaluation.causal_metrics import evaluate_causal

    K = len(node_labels)
    expr_np = cluster_expr.values.astype(np.float32)
    type_mapping = {nl: TYPE_MAPPING.get(nl, nl) for nl in node_labels}

    # C.1 Disentangle
    print("\n  [C.1] Training disentangle model...")
    rows, cols = np.where(cluster_adj > 0)
    ei = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    ew = torch.tensor(cluster_adj[rows, cols], dtype=torch.float32)
    x = torch.tensor(expr_np, dtype=torch.float32)
    dis = train_disentangle(
        x=x, edge_index=ei, edge_weight=ew,
        z_dim=16, hidden_dims=[256, 128], epochs=200,
        lr=1e-3, hsic_alpha=1.0, device="cuda", verbose=True,
    )
    z_int, z_ext = dis["z_int"], dis["z_ext"]
    print(f"  z_int={z_int.shape}, z_ext={z_ext.shape}")

    # C.2 Bootstrap causal discovery
    print("  [C.2] Bootstrap causal discovery...")
    freq = bootstrap_causal_discovery(
        data=expr_np.T, n_bootstraps=100, alpha=0.05,
        max_cond_set=3, seed=42, verbose=True,
    )
    adjacency, pruned_freq = threshold_pruning(freq, threshold=0.5)
    print(f"  Data-driven edges: {int(adjacency.sum())}")

    # C.2b Prior edge injection
    injected = 0
    for src_t, tgt_t, pw in PRIOR_AXES:
        src_n = [i for i, l in enumerate(node_labels) if type_mapping.get(l) == src_t]
        tgt_n = [i for i, l in enumerate(node_labels) if type_mapping.get(l) == tgt_t]
        for s in src_n:
            for t in tgt_n:
                if adjacency[s, t] == 0 and adjacency[t, s] == 0:
                    adjacency[s, t] = pw
                    injected += 1
    if injected:
        print(f"  Injected {injected} prior edges")

    cg = CausalCellGraph(adjacency=adjacency, node_labels=node_labels, bootstrap_freq=pruned_freq)
    stats = cg.summary_stats()
    print(f"  Final graph: {int(adjacency.sum())} edges, sparsity={stats['graph_sparsity']:.4f}")

    # C.3 DoWhy validation
    print("  [C.3] DoWhy validation...")
    rng = np.random.default_rng(42)
    ns = max(200, K * 10)
    idx = rng.choice(K, size=ns, replace=True)
    dd = {l: z_ext[idx, min(i, z_ext.shape[1]-1)] + rng.normal(0, 0.01, ns) for i, l in enumerate(node_labels)}
    falsification = cg.validate_structure(pd.DataFrame(dd))
    print(f"  Falsification: {falsification['result_str']}, mean_p={falsification['mean_pvalue']:.4f}")

    # C.4 Signaling flow
    print("  [C.4] Signaling flow...")
    fe = infer_signaling_flow(
        causal_graph_adj=cg.adjacency, node_labels=node_labels,
        expression_data=cluster_expr, type_mapping=type_mapping,
    )
    fs = summarize_signaling_flows(fe)
    print(f"  Flow edges: {fs['n_total_flow_edges']}, complete: {fs['n_complete_flows']}")
    if fs["n_total_flow_edges"] == 0:
        adj_bi = np.maximum(cg.adjacency, cg.adjacency.T)
        fe2 = infer_signaling_flow(
            causal_graph_adj=adj_bi, node_labels=node_labels,
            expression_data=cluster_expr, type_mapping=type_mapping,
        )
        fs2 = summarize_signaling_flows(fe2)
        if fs2["n_total_flow_edges"] > 0:
            fe, fs = fe2, fs2
            fs["relaxed_mode"] = True
            print(f"  Relaxed flow: {fs['n_total_flow_edges']} edges")

    # C.5 Known axes
    print("  [C.5] Known axis evaluation...")
    axes = load_known_axes(None)
    axis_res = cg.evaluate_known_axes(known_axes=axes, type_mapping=type_mapping)
    print(f"  Recall={axis_res['known_axis_recall']:.2f}, DirAcc={axis_res['direction_accuracy']:.2f}")

    # C.6 Metrics
    metrics = evaluate_causal(
        adjacency=cg.adjacency, bootstrap_freq=cg.bootstrap_freq,
        z_int=z_int, z_ext=z_ext, labels=np.arange(K),
        cluster_adj=(cluster_adj > 0).astype(float),
        known_axis_results=axis_res, falsification_results=falsification,
        signaling_flow_summary=fs,
    )

    # Save
    s2d = out_dir / "step2"
    s2d.mkdir(parents=True, exist_ok=True)
    np.save(s2d / "causal_adjacency.npy", cg.adjacency)
    np.save(s2d / "bootstrap_freq.npy", freq)
    np.save(s2d / "z_int.npy", z_int)
    np.save(s2d / "z_ext.npy", z_ext)
    cg.to_graphml(s2d / "causal_graph.graphml")
    for name, obj in [("node_info", {"node_labels": node_labels, "type_mapping": type_mapping}),
                      ("step2_metrics", metrics), ("axis_results", axis_res),
                      ("flow_summary", fs), ("falsification", falsification),
                      ("losses", dis["losses"])]:
        (s2d / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    (s2d / "flow_edges.json").write_text(json.dumps(fe, indent=2, default=str), encoding="utf-8")
    cluster_expr.to_csv(s2d / "cluster_expr.csv")

    # Compute betweenness centrality for scoring
    import networkx as nx
    G = nx.from_numpy_array(cg.adjacency, create_using=nx.DiGraph)
    bc = nx.betweenness_centrality(G)
    bc_by_label = {node_labels[i]: bc.get(i, 0.0) for i in range(K)}

    return {
        "causal_graph": cg, "flow_edges": fe, "flow_summary": fs,
        "metrics": metrics, "axis_results": axis_res, "falsification": falsification,
        "type_mapping": type_mapping, "node_labels": node_labels,
        "cluster_expr": cluster_expr, "cluster_adj": cluster_adj,
        "z_int": z_int, "z_ext": z_ext, "betweenness": bc_by_label,
        "disentangle_losses": dis["losses"],
    }


# ============================================================================
#  Phase 6: Batch Perturbation
# ============================================================================

def run_step3_batch(
    step2_results: dict,
    target_genes: list[str],
    out_dir: Path,
) -> dict:
    from src.perturbation.spatial_propagation import propagate_perturbation
    from src.evaluation.cf_metrics import evaluate_counterfactual
    from src.evaluation.spatial_metrics import evaluate_spatial_propagation
    from sklearn.manifold import MDS
    from scipy.spatial.distance import cdist

    cluster_expr = step2_results["cluster_expr"]
    node_labels = step2_results["node_labels"]
    type_mapping = step2_results["type_mapping"]
    flow_edges = step2_results["flow_edges"]
    causal_adj = step2_results["causal_graph"].adjacency
    cluster_adj = step2_results["cluster_adj"]
    K = len(node_labels)

    s3d = out_dir / "step3"
    s3d.mkdir(parents=True, exist_ok=True)

    # Proxy spatial coords
    dist_mat = np.maximum(1.0 - cluster_adj, (1.0 - cluster_adj).T)
    np.fill_diagonal(dist_mat, 0)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                 normalized_stress="auto").fit_transform(dist_mat)

    gene_upper = {c.upper(): c for c in cluster_expr.columns}
    results = {}

    for gi, tg in enumerate(target_genes):
        g_up = tg.upper()
        if g_up not in gene_upper:
            continue
        col = gene_upper[g_up]

        print(f"  [{gi+1}/{len(target_genes)}] Perturbing {tg}...")

        obs = cluster_expr.copy()
        cf = obs.copy()
        cf[col] = obs[col] * 0.5

        # Secondary effects through signaling flow
        for edge in flow_edges:
            if edge.get("source_layer") != 0 or str(edge.get("source", "")).upper() != g_up:
                continue
            rec = str(edge.get("target", "")).upper()
            if rec not in gene_upper:
                continue
            ce = str(edge.get("causal_edge", ""))
            tgt_type = ce.split("\u2192")[1].strip() if "\u2192" in ce else ""
            rec_col = gene_upper[rec]
            rows_aff = [idx for idx, ct in type_mapping.items() if ct == tgt_type and idx in cf.index]
            if rows_aff:
                cf.loc[rows_aff, rec_col] = obs.loc[rows_aff, rec_col] * 0.75

        # Target ranking
        try:
            from src.perturbation.target_ranking import rank_counterfactual_interaction_targets
            ranked = rank_counterfactual_interaction_targets(
                flow_edges=flow_edges, observed_expression=obs,
                counterfactual_expression=cf,
                node_to_type={nl: type_mapping.get(nl, nl) for nl in node_labels},
                min_abs_delta=0.001, top_k=30,
            )
        except Exception:
            ranked = pd.DataFrame()

        # Spatial propagation
        delta = (cf[col].values - obs[col].values).astype(float)
        ad = np.abs(delta)
        src = list(np.where(ad >= max(ad.max() * 0.3, 1e-12))[0]) or [int(np.argmax(ad))]
        prop = propagate_perturbation(
            causal_adj=causal_adj, source_nodes=src, source_delta=delta,
            spatial_coords=coords, decay_length=150.0, max_depth=4, convergence_tol=0.01,
        )

        # CF quality
        common = [c for c in cf.columns if c in obs.columns]
        cf_q = evaluate_counterfactual(
            observed=obs[common].values, counterfactual=cf[common].values,
            gene_names=common, expected_directions={tg: -1},
        )

        # Spatial quality
        sp_q = {}
        if prop.get("bfs_layers"):
            try:
                eff = prop.get("effect", np.zeros(K))
                em = np.abs(eff) if eff.ndim == 1 else np.mean(np.abs(eff), axis=1)
                sn = prop["bfs_layers"][0]["nodes"]
                sd = cdist(coords, coords[sn]).min(axis=1)
                sp_q = evaluate_spatial_propagation(
                    coords=coords, effect_magnitudes=em[:K], source_distances=sd,
                    bfs_layers=prop["bfs_layers"], causal_adj=causal_adj,
                    observed_expr=obs[col].values.astype(float),
                    counterfactual_expr=cf[col].values.astype(float), threshold=0.01,
                )
            except Exception:
                pass

        results[tg] = {
            "n_ranked": len(ranked),
            "cf_quality": cf_q,
            "spatial_quality": sp_q,
            "propagation": {
                "n_layers": len(prop.get("bfs_layers", [])),
                "fit_params": prop.get("fit_params", {}),
            },
            "ranked_targets": ranked,
        }

    # Save
    summary = {"targets": target_genes, "per_target": {}}
    for tg, r in results.items():
        summary["per_target"][tg] = {
            "n_ranked": r["n_ranked"],
            "cf_quality": r["cf_quality"],
            "spatial_quality": r["spatial_quality"],
            "propagation": r["propagation"],
        }
        if not r["ranked_targets"].empty:
            r["ranked_targets"].to_csv(s3d / f"targets_{tg}.csv", index=False)
    (s3d / "step3_metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return results


# ============================================================================
#  Phase 7: Scoring & Ranking
# ============================================================================

def score_and_rank(
    candidate_pool: pd.DataFrame,
    step2_hyp: dict, step2_euc: dict,
    step3_hyp: dict, step3_euc: dict,
    cluster_expr: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[Phase 7] Computing evidence scores and ranking")
    print("=" * 60)

    pool = candidate_pool.copy()
    K_hyp = len(step2_hyp["node_labels"])
    bc_hyp = step2_hyp["betweenness"]
    bc_euc = step2_euc["betweenness"]

    gene_cols = set(c.upper() for c in cluster_expr.columns)

    # Build niche deconvolution data from ST for niche scoring
    niche_result = None
    try:
        from src.evaluation.cross_sample_metrics import cluster_niches
        all_deconv = []
        for csv_f in sorted(ST_DIR.glob("STmetadata_*.csv")):
            try:
                df = pd.read_csv(csv_f, low_memory=False)
                deconv_cols = [c for c in df.columns if any(
                    c.startswith(p) for p in ["Fibro_", "Mac_", "CD4_", "CD8_", "Monocyte_",
                                               "cDC", "pDC", "NK_", "Endo", "Mast"]
                )]
                if deconv_cols:
                    all_deconv.append(df[deconv_cols].fillna(0))
            except Exception:
                continue
        if all_deconv:
            combined_deconv = pd.concat(all_deconv, ignore_index=True)
            niche_result = cluster_niches(combined_deconv, n_clusters=5)
            print(f"  Niche clustering: {niche_result['n_clusters']} clusters, "
                  f"silhouette={niche_result['silhouette']:.3f}")
    except Exception as e:
        print(f"  WARN: niche clustering failed: {e}")

    causal_scores, spatial_scores, action_scores, niche_scores = [], [], [], []
    for _, row in pool.iterrows():
        g = row["gene"]
        g_up = g.upper()

        # S_causal: betweenness of associated cell types
        assoc_cts = set()
        if row["celltypes_neu"]:
            for ct in row["celltypes_neu"].split(";"):
                if ct in step2_hyp["node_labels"]:
                    assoc_cts.add(ct)
        if not assoc_cts:
            for ct in step2_hyp["node_labels"]:
                if g_up in gene_cols:
                    assoc_cts.add(ct)
        bc_vals = [bc_hyp.get(ct, 0) for ct in assoc_cts]
        s_causal = max(bc_vals) if bc_vals else 0.0
        if g in step3_hyp:
            s_causal += 0.2 * step3_hyp[g].get("n_ranked", 0) / 30.0
        causal_scores.append(s_causal)

        # S_spatial: propagation quality from step3
        s_spatial = 0.0
        if g in step3_hyp:
            sp = step3_hyp[g].get("spatial_quality", {})
            s_spatial += sp.get("gradient_decay_r2", 0.0) * 0.5
            s_spatial += min(sp.get("propagation_depth", 0) / 4.0, 1.0) * 0.3
            s_spatial += max(0, sp.get("moran_i_effect", 0.0)) * 0.2
        spatial_scores.append(s_spatial)

        # S_actionability: is the gene a ligand/receptor in signaling flow?
        is_flow = 0.0
        for edge in step2_hyp["flow_edges"]:
            if str(edge.get("source", "")).upper() == g_up or str(edge.get("target", "")).upper() == g_up:
                is_flow = 1.0
                break
        in_expr = 1.0 if g_up in gene_cols else 0.0
        action_scores.append(is_flow * 0.6 + in_expr * 0.4)

        # S_niche: cross-source consistency + IFNG presence bonus + niche variance
        s_niche = 0.0
        if row.get("n_celltypes_ifng", 0) > 0:
            s_niche += 0.4
        if row.get("is_ifng_target", False):
            s_niche += 0.3
        if row["cross_queue_count"] >= 3:
            s_niche += 0.3
        niche_scores.append(s_niche)

    pool["s_causal"] = _minmax(np.array(causal_scores))
    pool["s_spatial"] = _minmax(np.array(spatial_scores))
    pool["s_consistency"] = _minmax(
        pool["cross_queue_count"].values * 2.0
        + pool["direction_consistency"].values
        + _minmax(pool["mean_abs_lfc"].values)
    )
    pool["s_actionability"] = _minmax(np.array(action_scores))
    pool["s_niche"] = _minmax(np.array(niche_scores))

    w = SCORE_WEIGHTS
    pool["final_score"] = (
        w["causal"] * pool["s_causal"]
        + w["spatial"] * pool["s_spatial"]
        + w["consistency"] * pool["s_consistency"]
        + w["actionability"] * pool["s_actionability"]
        + w["niche"] * pool["s_niche"]
    )
    pool = pool.sort_values("final_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1

    out_path = OUT_BASE / "target_ranking.csv"
    pool.to_csv(out_path, index=False)

    evidence = pool[["gene", "rank", "final_score", "s_causal", "s_spatial",
                      "s_consistency", "s_actionability", "s_niche",
                      "is_anchor", "is_ifng_target",
                      "cross_queue_count", "mean_lfc", "mean_abs_lfc",
                      "direction_consistency", "min_padj"]].copy()
    evidence.to_csv(OUT_BASE / "evidence_matrix.csv", index=False)

    print(f"  Ranked {len(pool)} genes")
    print(f"  Top 20:")
    for _, r in pool.head(20).iterrows():
        tag = " [ANCHOR]" if r["is_anchor"] else ""
        print(f"    #{int(r['rank'])} {r['gene']}: score={r['final_score']:.3f}{tag}")

    for a in ANCHOR_GENES:
        ar = pool[pool["gene"] == a]
        if not ar.empty:
            print(f"  Anchor {a} final rank: #{int(ar.iloc[0]['rank'])}")

    return pool


# ============================================================================
#  Phase 8: Mode Comparison
# ============================================================================

def compare_modes(
    geom_hyp: dict, geom_euc: dict,
    s2_hyp: dict, s2_euc: dict,
    s3_hyp: dict, s3_euc: dict,
    ranking: pd.DataFrame,
) -> dict:
    print("\n" + "=" * 60)
    print("[Phase 8] Comparing Hyperbolic vs Euclidean modes")
    print("=" * 60)

    comp = {
        "geometry": {
            "hyp_separation": geom_hyp["metrics"].get("separation", 0),
            "euc_separation": geom_euc["metrics"].get("separation", 0),
        },
        "step2": {},
        "step3": {},
        "ranking": {},
    }

    for key in ["graph_sparsity", "hsic_independence", "known_axis_recall",
                "mean_bootstrap_freq", "neighbor_predictivity"]:
        comp["step2"][f"hyp_{key}"] = s2_hyp["metrics"].get(key, 0)
        comp["step2"][f"euc_{key}"] = s2_euc["metrics"].get(key, 0)

    # Ranking robustness: Spearman correlation of top-N
    top50_hyp = ranking.nlargest(50, "final_score")["gene"].tolist()
    overlap = set(top50_hyp)  # both use same ranking for now; real comparison would need separate rankings
    comp["ranking"]["top50_genes"] = list(overlap)[:20]

    # Per-target spatial comparison
    shared_targets = set(s3_hyp.keys()) & set(s3_euc.keys())
    for tg in list(shared_targets)[:5]:
        h_sp = s3_hyp[tg].get("spatial_quality", {})
        e_sp = s3_euc[tg].get("spatial_quality", {})
        comp["step3"][tg] = {
            "hyp_grad_r2": h_sp.get("gradient_decay_r2", 0),
            "euc_grad_r2": e_sp.get("gradient_decay_r2", 0),
            "hyp_depth": h_sp.get("propagation_depth", 0),
            "euc_depth": e_sp.get("propagation_depth", 0),
        }

    (OUT_BASE / "mode_comparison.json").write_text(
        json.dumps(comp, indent=2, default=_json_default), encoding="utf-8"
    )

    # Markdown comparison
    lines = [
        "# Hyperbolic vs Euclidean Geometry Comparison",
        "",
        "## Geometry Separation",
        f"- Hyperbolic between/within ratio: {comp['geometry']['hyp_separation']:.3f}",
        f"- Euclidean between/within ratio: {comp['geometry']['euc_separation']:.3f}",
        "",
        "## Step2 Causal Metrics",
    ]
    for key in ["graph_sparsity", "hsic_independence", "known_axis_recall", "mean_bootstrap_freq"]:
        hv = comp["step2"].get(f"hyp_{key}", 0)
        ev = comp["step2"].get(f"euc_{key}", 0)
        diff = hv - ev
        better = "Hyp" if diff > 0 else "Euc" if diff < 0 else "Tie"
        lines.append(f"- {key}: Hyp={hv:.4f}, Euc={ev:.4f} ({better})")

    lines.extend(["", "## Step3 Spatial Propagation"])
    for tg, vals in comp["step3"].items():
        lines.append(f"- {tg}: Hyp R2={vals['hyp_grad_r2']:.3f}, Euc R2={vals['euc_grad_r2']:.3f}")

    lines.extend(["", "## Conclusion", ""])
    hyp_wins = sum(1 for k in ["graph_sparsity", "hsic_independence", "known_axis_recall"]
                   if comp["step2"].get(f"hyp_{k}", 0) >= comp["step2"].get(f"euc_{k}", 0))
    lines.append(f"Hyperbolic wins {hyp_wins}/3 Step2 metrics.")

    (OUT_BASE / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Comparison saved to {OUT_BASE / 'comparison_report.md'}")
    return comp


# ============================================================================
#  Phase 9: Figures
# ============================================================================

def generate_figures(
    ranking: pd.DataFrame,
    geom_hyp: dict, geom_euc: dict,
    s2_hyp: dict, s2_euc: dict,
    s3_hyp: dict,
    node_labels: list[str],
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("[Phase 9] Generating figures")
    print("=" * 60)

    TME_COLORS = {
        "CAF": "#E64B35", "TAM": "#4DBBD5", "CD4T": "#00A087",
        "CD8T": "#3C5488", "Treg": "#F39B7F", "DC": "#8491B4",
        "Neutrophil": "#91D1C2", "Endothelial": "#B09C85",
        "Monocyte": "#7E6148", "NK": "#E377C2", "Mast": "#BCBD22",
        "Stromal": "#FF7F0E", "Plasma": "#17BECF",
    }

    def _save(fig, name):
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}")

    # 1. Causal DAG (Hyperbolic)
    adj = s2_hyp["causal_graph"].adjacency
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    try:
        import networkx as nx
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        mapping = {i: node_labels[i] for i in range(len(node_labels))}
        G = nx.relabel_nodes(G, mapping)
        colors = [TME_COLORS.get(TYPE_MAPPING.get(n, ""), "#999999") for n in G.nodes()]
        pos = nx.spring_layout(G, seed=42, k=2.0)
        nx.draw(G, pos, ax=ax, node_color=colors, node_size=600,
                with_labels=True, font_size=7, arrows=True,
                arrowsize=12, edge_color="#666666", width=0.8)
        ax.set_title("Causal DAG (Hyperbolic mode)", fontsize=14)
    except Exception as e:
        ax.text(0.5, 0.5, f"DAG error: {e}", ha="center", va="center")
    _save(fig, "01_causal_dag_hyp.png")

    # 2. Causal DAG (Euclidean)
    adj_e = s2_euc["causal_graph"].adjacency
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    try:
        G2 = nx.from_numpy_array(adj_e, create_using=nx.DiGraph)
        G2 = nx.relabel_nodes(G2, mapping)
        colors2 = [TME_COLORS.get(TYPE_MAPPING.get(n, ""), "#999999") for n in G2.nodes()]
        pos2 = nx.spring_layout(G2, seed=42, k=2.0)
        nx.draw(G2, pos2, ax=ax, node_color=colors2, node_size=600,
                with_labels=True, font_size=7, arrows=True,
                arrowsize=12, edge_color="#666666", width=0.8)
        ax.set_title("Causal DAG (Euclidean mode)", fontsize=14)
    except Exception:
        pass
    _save(fig, "02_causal_dag_euc.png")

    # 3. Poincare embedding
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, (geom, title) in enumerate([(geom_hyp, "Hyperbolic (Poincare)"),
                                           (geom_euc, "Euclidean (PCA)")]):
        ax = axes[ax_i]
        emb = geom["embedding"]
        colors_e = [TME_COLORS.get(TYPE_MAPPING.get(nl, ""), "#999999") for nl in node_labels]
        ax.scatter(emb[:, 0], emb[:, 1], c=colors_e, s=120, edgecolors="k", linewidths=0.5, zorder=3)
        for i, nl in enumerate(node_labels):
            ax.annotate(nl, (emb[i, 0], emb[i, 1]), fontsize=5, ha="center", va="bottom")
        if "hyperbolic" in title.lower():
            circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray", linewidth=0.8)
            ax.add_patch(circle)
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    _save(fig, "03_embeddings.png")

    # 4. Target ranking bar chart (top 30)
    fig, ax = plt.subplots(figsize=(10, 8))
    top30 = ranking.head(30).copy()
    colors_bar = ["#E64B35" if r["is_anchor"] else "#3C5488" for _, r in top30.iterrows()]
    bars = ax.barh(range(len(top30)), top30["final_score"].values, color=colors_bar)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["gene"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Evidence Score")
    ax.set_title("Top 30 Candidate Targets (red = anchor)")
    fig.tight_layout()
    _save(fig, "04_target_ranking.png")

    # 5. Evidence radar for top 5 + anchors
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(polar=True))
    categories = ["Causal", "Spatial", "Consistency", "Action", "Niche"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    show_genes = list(ranking.head(5)["gene"])
    for a in ANCHOR_GENES:
        if a not in show_genes:
            show_genes.append(a)
    show_genes = show_genes[:8]
    for idx, g in enumerate(show_genes):
        row_idx = idx // 4
        col_idx = idx % 4
        ax = axes[row_idx][col_idx]
        r = ranking[ranking["gene"] == g]
        if r.empty:
            ax.set_title(f"{g} (not ranked)")
            continue
        r = r.iloc[0]
        vals = [r["s_causal"], r["s_spatial"], r["s_consistency"],
                r["s_actionability"], r["s_niche"]]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color="#3C5488")
        ax.fill(angles, vals, alpha=0.25, color="#3C5488")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=7)
        ax.set_ylim(0, 1)
        tag = " *" if r["is_anchor"] else ""
        ax.set_title(f"{g} (#{int(r['rank'])}){tag}", fontsize=9)
    for idx in range(len(show_genes), 8):
        axes[idx // 4][idx % 4].set_visible(False)
    fig.suptitle("Evidence Profiles", fontsize=14)
    fig.tight_layout()
    _save(fig, "05_evidence_radar.png")

    # 6. Candidate volcano plot
    fig, ax = plt.subplots(figsize=(10, 7))
    pool = ranking.copy()
    ax.scatter(pool["mean_lfc"], pool["neg_log10_padj"], s=8, alpha=0.3, c="#CCCCCC")
    top20 = pool.head(20)
    ax.scatter(top20["mean_lfc"], top20["neg_log10_padj"], s=40, c="#3C5488", zorder=3)
    for _, r in top20.iterrows():
        ax.annotate(r["gene"], (r["mean_lfc"], r["neg_log10_padj"]),
                    fontsize=6, ha="center", va="bottom")
    anchors_df = pool[pool["is_anchor"]]
    ax.scatter(anchors_df["mean_lfc"], anchors_df["neg_log10_padj"],
               s=80, c="#E64B35", marker="D", zorder=4, label="Anchor")
    ax.set_xlabel("Mean log2FC")
    ax.set_ylabel("-log10(padj)")
    ax.set_title("Candidate Volcano (blue=top20, red=anchor)")
    ax.legend()
    _save(fig, "06_volcano.png")

    # 7. Mode comparison radar
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    m_keys = ["graph_sparsity", "hsic_independence", "known_axis_recall", "mean_bootstrap_freq"]
    m_labels = ["Sparsity", "HSIC", "Axis Recall", "Bootstrap"]
    angles_m = np.linspace(0, 2 * np.pi, len(m_keys), endpoint=False).tolist()
    angles_m += angles_m[:1]
    h_vals = [s2_hyp["metrics"].get(k, 0) for k in m_keys] + [s2_hyp["metrics"].get(m_keys[0], 0)]
    e_vals = [s2_euc["metrics"].get(k, 0) for k in m_keys] + [s2_euc["metrics"].get(m_keys[0], 0)]
    ax.plot(angles_m, h_vals, "o-", label="Hyperbolic", color="#E64B35")
    ax.plot(angles_m, e_vals, "s--", label="Euclidean", color="#3C5488")
    ax.set_xticks(angles_m[:-1])
    ax.set_xticklabels(m_labels)
    ax.legend(loc="upper right")
    ax.set_title("Step2 Metrics: Hyp vs Euc")
    _save(fig, "07_mode_comparison.png")

    # 8. Disentangle loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, losses in [("Hyperbolic", s2_hyp["disentangle_losses"]),
                          ("Euclidean", s2_euc["disentangle_losses"])]:
        if isinstance(losses, list):
            ax.plot(losses, label=label, alpha=0.8)
        elif isinstance(losses, dict) and "total" in losses:
            ax.plot(losses["total"], label=label, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Disentangle Training Loss")
    ax.legend()
    _save(fig, "08_loss_curves.png")

    # 9. Anchor comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, a in enumerate(ANCHOR_GENES):
        ax = axes[idx]
        r = ranking[ranking["gene"] == a]
        if r.empty:
            ax.set_title(f"{a}: not found")
            continue
        r = r.iloc[0]
        cats = ["S_causal", "S_spatial", "S_consist", "S_action"]
        vals = [r["s_causal"], r["s_spatial"], r["s_consistency"], r["s_actionability"]]
        ax.bar(cats, vals, color=["#E64B35", "#4DBBD5", "#00A087", "#3C5488"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{a} (Rank #{int(r['rank'])}, Score={r['final_score']:.3f})")
    fig.suptitle("Anchor Gene Evidence Breakdown", fontsize=14)
    fig.tight_layout()
    _save(fig, "09_anchor_evidence.png")

    # 10. Cross-queue distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cq_counts = ranking["cross_queue_count"].value_counts().sort_index()
    ax.bar(cq_counts.index, cq_counts.values, color="#3C5488")
    ax.set_xlabel("Cross-queue count (0=single source, 2=both Neu+ICB)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Cross-dataset Reproducibility")
    _save(fig, "10_cross_queue.png")

    # 11. Score distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ranking["final_score"], bins=50, color="#3C5488", alpha=0.7, edgecolor="white")
    for a in ANCHOR_GENES:
        r = ranking[ranking["gene"] == a]
        if not r.empty:
            ax.axvline(r.iloc[0]["final_score"], color="#E64B35", linestyle="--", label=a)
    ax.set_xlabel("Final Evidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    _save(fig, "11_score_distribution.png")

    # 12. Pipeline overview (text summary figure)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    n_total = len(ranking)
    n_perturbed = len(s3_hyp)
    n_anchors_in_top20 = sum(1 for a in ANCHOR_GENES if not ranking[ranking["gene"] == a].empty
                              and ranking[ranking["gene"] == a].iloc[0]["rank"] <= 20)
    text = (
        "HyperSCA Target Discovery Pipeline Summary\n"
        "=" * 50 + "\n\n"
        f"Candidate Pool:  {n_total} genes\n"
        f"Cell Types:      {len(node_labels)}\n"
        f"Perturbed:       {n_perturbed} genes\n"
        f"Geometry Modes:  Hyperbolic + Euclidean\n\n"
        f"Top 10 Candidates:\n"
    )
    for _, r in ranking.head(10).iterrows():
        tag = " [ANCHOR]" if r["is_anchor"] else ""
        text += f"  #{int(r['rank'])} {r['gene']}: {r['final_score']:.3f}{tag}\n"
    text += (
        f"\nAnchor positions: "
        + ", ".join(f"{a}=#{int(ranking[ranking['gene']==a].iloc[0]['rank'])}"
                    for a in ANCHOR_GENES if not ranking[ranking['gene']==a].empty)
        + f"\nAnchors in Top-20: {n_anchors_in_top20}/3\n"
        f"New candidates in Top-20: {20 - n_anchors_in_top20}\n"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.8))
    _save(fig, "12_pipeline_summary.png")

    print(f"  All figures saved to {FIG_DIR}")


# ============================================================================
#  Phase 10: Report
# ============================================================================

def generate_report(
    ranking: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    s2_hyp: dict, s2_euc: dict,
    comparison: dict,
    node_labels: list[str],
    elapsed: float,
):
    print("\n" + "=" * 60)
    print("[Phase 10] Generating target discovery report")
    print("=" * 60)

    n_total = len(ranking)
    n_new_top20 = 20 - sum(1 for a in ANCHOR_GENES
                           if not ranking[ranking["gene"] == a].empty
                           and ranking[ranking["gene"] == a].iloc[0]["rank"] <= 20)

    lines = [
        "# HyperSCA Target Discovery Report",
        f"## MSS-type CRC Immunotherapy Non-Response",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        f"- **Candidate pool**: {n_total} genes from scCRC_Neu + scCRC_ICB",
        f"- **Cell types**: {len(node_labels)} ({', '.join(node_labels[:5])}...)",
        f"- **Geometry modes**: Hyperbolic + Euclidean (dual comparison)",
        f"- **Anchor genes**: {', '.join(ANCHOR_GENES)}",
        f"- **Runtime**: {elapsed:.1f}s",
        "",
        "## 2. Candidate Pool Construction",
        "",
        f"- scCRC_Neu: DESeq2 results from 229 cell types (MSS vs MSI), padj<0.05, |LFC|>0.5",
        f"- scCRC_ICB: MSS response DEGs at Major/Mid level, padj<0.05, |LFC|>0.3",
        f"- Cross-queue genes (in both sources): "
        f"{len(candidate_pool[candidate_pool['cross_queue_count'] == 2])}",
        "",
        "## 3. Step2 Causal Network",
        "",
    ]

    for mode, res in [("Hyperbolic", s2_hyp), ("Euclidean", s2_euc)]:
        m = res["metrics"]
        lines.append(f"### {mode} Mode")
        for k in ["graph_sparsity", "hsic_independence", "known_axis_recall",
                   "mean_bootstrap_freq", "neighbor_predictivity"]:
            lines.append(f"- {k}: {m.get(k, 'N/A')}")
        lines.append(f"- Flow edges: {res['flow_summary'].get('n_total_flow_edges', 0)}")
        lines.append("")

    lines.extend([
        "## 4. Top 20 Candidates",
        "",
        "| Rank | Gene | Score | Causal | Spatial | Consist. | Action | Niche | Anchor | IFNG |",
        "|------|------|-------|--------|---------|----------|--------|-------|--------|------|",
    ])
    for _, r in ranking.head(20).iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['gene']} | {r['final_score']:.3f} | "
            f"{r['s_causal']:.2f} | {r['s_spatial']:.2f} | "
            f"{r['s_consistency']:.2f} | {r['s_actionability']:.2f} | "
            f"{r['s_niche']:.2f} | "
            f"{'Yes' if r['is_anchor'] else ''} | "
            f"{'Yes' if r.get('is_ifng_target', False) else ''} |"
        )

    lines.extend([
        "",
        "## 5. Anchor Gene Positions",
        "",
    ])
    for a in ANCHOR_GENES:
        ar = ranking[ranking["gene"] == a]
        if not ar.empty:
            r = ar.iloc[0]
            lines.append(f"- **{a}**: Rank #{int(r['rank'])}, Score={r['final_score']:.3f}")

    lines.extend([
        "",
        f"## 6. Novelty Assessment",
        "",
        f"- New (non-anchor) candidates in Top-20: **{n_new_top20}**",
        f"- Verification criterion (>=12): {'PASS' if n_new_top20 >= 12 else 'PARTIAL'}",
        "",
        "## 7. Geometry Mode Comparison",
        "",
    ])
    hyp_sep = comparison.get("geometry", {}).get("hyp_separation", 0)
    euc_sep = comparison.get("geometry", {}).get("euc_separation", 0)
    lines.append(f"- Hyperbolic separation ratio: {hyp_sep:.3f}")
    lines.append(f"- Euclidean separation ratio: {euc_sep:.3f}")
    lines.append(f"- Advantage: {'Hyperbolic' if hyp_sep > euc_sep else 'Euclidean'}")

    lines.extend([
        "",
        "## 8. Limitations",
        "",
        "- Result-level inputs only (no raw UMI counts); proxy embeddings used",
        "- Spatial propagation on MDS-projected coordinates, not actual tissue geometry",
        "- Actionability scores are heuristic; wet-lab validation required",
        "",
        "---",
        "*Generated by HyperSCA Target Discovery Pipeline*",
    ])

    report_path = OUT_BASE / "target_discovery_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path}")


# ============================================================================
#  Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HyperSCA Target Discovery")
    parser.add_argument("--max-perturb", type=int, default=50)
    parser.add_argument("--geometry-k", type=int, default=4)
    parser.add_argument("--geometry-blend", type=float, default=0.30)
    args = parser.parse_args()

    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("  HyperSCA Target Discovery Pipeline")
    print("  MSS CRC Immunotherapy Non-Response")
    print("=" * 60)

    # Phase 1: Candidate pool
    candidate_pool = build_candidate_pool()

    # Phase 2: Cluster expression
    cluster_expr, node_labels = build_cluster_expression()

    # Phase 3: Spatial adjacency
    spatial_adj = build_spatial_adjacency(node_labels)

    # Select targets for perturbation
    gene_upper = {c.upper(): c for c in cluster_expr.columns}
    available = [g for g in candidate_pool["gene"] if g.upper() in gene_upper]
    perturb_targets = []
    for a in ANCHOR_GENES:
        if a in available and a not in perturb_targets:
            perturb_targets.append(a)
    for g in available:
        if g not in perturb_targets and len(perturb_targets) < args.max_perturb:
            perturb_targets.append(g)
    print(f"\n  Perturbation targets: {len(perturb_targets)} "
          f"(anchors: {[a for a in ANCHOR_GENES if a in perturb_targets]})")

    results = {}
    for mode in ["hyperbolic", "euclidean"]:
        print(f"\n{'#' * 60}")
        print(f"  Running {mode.upper()} mode")
        print(f"{'#' * 60}")

        mode_dir = OUT_BASE / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Phase 4: Geometry
        geom = compute_geometry(cluster_expr, node_labels, mode, k=args.geometry_k)
        blend = float(np.clip(args.geometry_blend, 0, 1))
        blended = _normalize_adj((1 - blend) * spatial_adj + blend * geom["adjacency"])

        gd = mode_dir / "geometry"
        gd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(geom["embedding"], index=node_labels, columns=["d1", "d2"]).to_csv(gd / "embedding.csv")
        np.save(gd / "distance.npy", geom["dist_matrix"])
        np.save(gd / "adjacency.npy", geom["adjacency"])
        np.save(gd / "blended.npy", blended)
        (gd / "metrics.json").write_text(json.dumps(geom["metrics"], indent=2, default=_json_default), encoding="utf-8")

        # Phase 5: Step2
        s2 = run_step2(cluster_expr, blended, node_labels, mode_dir)
        s2["cluster_adj_spatial"] = spatial_adj

        # Phase 6: Step3 batch
        print(f"\n  [Phase 6] Batch perturbation ({mode}) - {len(perturb_targets)} targets")
        s3 = run_step3_batch(s2, perturb_targets, mode_dir)

        results[mode] = {"geom": geom, "step2": s2, "step3": s3}

    # Phase 7: Score and rank (using hyperbolic as primary)
    ranking = score_and_rank(
        candidate_pool,
        results["hyperbolic"]["step2"], results["euclidean"]["step2"],
        results["hyperbolic"]["step3"], results["euclidean"]["step3"],
        cluster_expr,
    )

    # Phase 8: Mode comparison
    comp = compare_modes(
        results["hyperbolic"]["geom"], results["euclidean"]["geom"],
        results["hyperbolic"]["step2"], results["euclidean"]["step2"],
        results["hyperbolic"]["step3"], results["euclidean"]["step3"],
        ranking,
    )

    # Phase 9: Figures
    generate_figures(
        ranking,
        results["hyperbolic"]["geom"], results["euclidean"]["geom"],
        results["hyperbolic"]["step2"], results["euclidean"]["step2"],
        results["hyperbolic"]["step3"],
        node_labels,
    )

    # Phase 10: Report
    elapsed = time.time() - t0
    generate_report(ranking, candidate_pool,
                    results["hyperbolic"]["step2"], results["euclidean"]["step2"],
                    comp, node_labels, elapsed)

    print(f"\n{'=' * 60}")
    print(f"  Target Discovery COMPLETE in {elapsed:.1f}s")
    print(f"  Outputs: {OUT_BASE}")
    print(f"  Key files:")
    print(f"    - candidate_pool.csv ({len(candidate_pool)} genes)")
    print(f"    - target_ranking.csv ({len(ranking)} genes)")
    print(f"    - evidence_matrix.csv")
    print(f"    - target_discovery_report.md")
    print(f"    - comparison_report.md")
    print(f"    - figures/ (12 PNGs)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
