#!/usr/bin/env python
"""Generate ST niche and spatial communication figure pack.

Focus:
1) Hyperbolic niche hierarchy advantage (vs Euclidean)
2) Per-ST-input niche spatial distribution
3) Per-ST-input cell-type enrichment by niche
4) Target static spatial score map (niche-informed)
5) Predicted combo effect spatial map
6) Recognized communication flow direction map
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plot_style import (
    CMAP_SPATIAL,
    CMAP_SPATIAL_DIVERGING,
    apply_cns_style,
    get_color_mapping,
    save_figure,
)


def _broad_type_from_deconv_col(col: str) -> str:
    c = str(col)
    if c.startswith("Fibro_"):
        return "CAF"
    if c.startswith("Mac_"):
        return "TAM"
    if c.startswith("CD4_"):
        return "CD4T"
    if c.startswith("CD8_"):
        return "CD8T"
    if c.startswith("cDC") or c.startswith("DC_") or c.startswith("pDC"):
        return "DC"
    if c.startswith("Monocyte_"):
        return "Monocyte"
    if c.startswith("NK_"):
        return "NK"
    if c.startswith("Endo"):
        return "Endothelial"
    if c.startswith("Mast"):
        return "Mast"
    return "Other"


def _safe_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_causal_edge(edge: str) -> tuple[str, str] | None:
    txt = str(edge)
    txt = txt.replace("->", "→")
    if "→" not in txt:
        return None
    left, right = txt.split("→", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    return left, right


def _type_flow_weights(flow_edges: list[dict]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for e in flow_edges:
        parsed = _parse_causal_edge(e.get("causal_edge", ""))
        if parsed is None:
            continue
        key = parsed
        w = float(e.get("weight", 0.0))
        out[key] = out.get(key, 0.0) + w
    return out


def _subsample_idx(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n), size=max_points, replace=False))


def _ensure_dirs(base: Path) -> dict[str, Path]:
    out = {
        "root": base,
        "niche_distribution": base / "niche_distribution",
        "niche_enrichment": base / "niche_enrichment",
        "target_static": base / "target_static",
        "info_flow": base / "info_flow",
        "perturb_flow": base / "perturb_flow",
        "violin": base / "violin",
        "native_celltype_spatial": base / "native_celltype_spatial",
        "colocalization_spatial": base / "colocalization_spatial",
        "colocalization_violin": base / "colocalization_violin",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _plot_global_hierarchy_advantage(
    assignment: pd.DataFrame,
    metrics: dict,
    out_path: Path,
) -> None:
    idx = _subsample_idx(len(assignment), max_points=50000, seed=42)
    sub = assignment.iloc[idx].copy()
    niches = sorted(sub["niche_id"].astype(int).unique().tolist())
    cmap = plt.get_cmap("tab20")
    niche_color = {n: cmap(i % 20) for i, n in enumerate(niches)}
    colors = [niche_color[int(n)] for n in sub["niche_id"].astype(int).values]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes
    ax1.scatter(sub["hyp_x"], sub["hyp_y"], s=3, c=colors, alpha=0.6, edgecolors="none")
    ax1.set_title("Hyperbolic niche embedding")
    ax1.set_xlabel("hyp_x")
    ax1.set_ylabel("hyp_y")
    ax1.set_aspect("equal")
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
    ax1.add_patch(circle)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)

    ax2.scatter(sub["euc_x"], sub["euc_y"], s=3, c=colors, alpha=0.6, edgecolors="none")
    ax2.set_title("Euclidean niche embedding")
    ax2.set_xlabel("euc_x")
    ax2.set_ylabel("euc_y")
    ax2.set_aspect("equal")

    names = ["silhouette_hyperbolic", "silhouette_euclidean", "hierarchy_corr_hyperbolic", "hierarchy_corr_euclidean"]
    vals = [float(metrics.get(k, 0.0)) for k in names]
    bars = ax3.bar(range(len(vals)), vals, color=["#EE6677", "#4477AA", "#EE6677", "#4477AA"], edgecolor="white")
    ax3.set_xticks(range(len(vals)))
    ax3.set_xticklabels(["Sil_hyp", "Sil_euc", "Hier_hyp", "Hier_euc"], rotation=20)
    ax3.set_title("Hyperbolic hierarchy advantage metrics")
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Stage-1 MSS Spatial Niche Hierarchy (Hyperbolic vs Euclidean)")
    save_figure(fig, out_path, dpi=300, config={"chart": "stage1_hierarchy_advantage"})


def _sample_enrichment_heatmap(
    st_dir: Path,
    sample_id: str,
    sample_assign: pd.DataFrame,
    out_path: Path,
) -> None:
    csv_path = st_dir / f"STmetadata_{sample_id}.csv"
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return
    deconv_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in ["Fibro_", "Mac_", "CD4_", "CD8_", "Monocyte_", "cDC", "pDC", "NK_", "Endo", "Mast"])
    ]
    if not deconv_cols:
        return

    valid_idx = sample_assign["sample_spot_idx"].astype(int).values
    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < len(df))]
    if len(valid_idx) == 0:
        return

    sub_expr = df.iloc[valid_idx][deconv_cols].fillna(0.0).astype(float).copy()
    row_sum = sub_expr.sum(axis=1).replace(0, np.nan)
    sub_expr = sub_expr.div(row_sum, axis=0).fillna(0.0)
    niche_ids = sample_assign.iloc[: len(valid_idx)]["niche_id"].astype(int).values
    sub_expr["niche_id"] = niche_ids

    # Aggregate by broad type
    recs = []
    for nid, gdf in sub_expr.groupby("niche_id"):
        mean_col = gdf[deconv_cols].mean(axis=0)
        broad_vals: dict[str, list[float]] = {}
        for col, val in mean_col.items():
            bt = _broad_type_from_deconv_col(col)
            broad_vals.setdefault(bt, []).append(float(val))
        row = {"niche_id": int(nid)}
        for bt, vals in broad_vals.items():
            row[bt] = float(np.mean(vals))
        recs.append(row)
    prof = pd.DataFrame(recs).fillna(0.0).set_index("niche_id").sort_index()
    if prof.empty:
        return
    gmean = prof.mean(axis=0)
    enrich = np.log2((prof + 1e-6) / (gmean + 1e-6))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(enrich.values, aspect="auto", cmap=CMAP_SPATIAL_DIVERGING)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="log2 enrichment")
    ax.set_xticks(range(enrich.shape[1]))
    ax.set_xticklabels(enrich.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(enrich.shape[0]))
    ax.set_yticklabels([f"N{int(i)}" for i in enrich.index], fontsize=8)
    ax.set_title(f"{sample_id} niche cell-type enrichment")
    save_figure(fig, out_path, dpi=300, config={"chart": "sample_niche_enrichment", "sample": sample_id})


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.fillna(0.0).astype(float).copy()
    rs = x.sum(axis=1).replace(0.0, np.nan)
    return x.div(rs, axis=0).fillna(0.0)


def _collect_st_native_celltypes_for_sample(
    st_dir: Path,
    sample_id: str,
    sample_assign: pd.DataFrame,
) -> pd.DataFrame:
    csv_path = st_dir / f"STmetadata_{sample_id}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return pd.DataFrame()
    deconv_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in ["Fibro_", "Mac_", "CD4_", "CD8_", "Monocyte_", "cDC", "pDC", "NK_", "Endo", "Mast"])
    ]
    if not deconv_cols:
        return pd.DataFrame()
    sp_idx = pd.to_numeric(sample_assign["sample_spot_idx"], errors="coerce").fillna(-1).astype(int).values
    valid = (sp_idx >= 0) & (sp_idx < len(df))
    if not np.any(valid):
        return pd.DataFrame()
    use_idx = np.where(valid)[0]
    prof = _normalize_rows(df.iloc[sp_idx[valid]][deconv_cols])

    out = sample_assign.iloc[use_idx][["sample_id", "sample_spot_idx", "x", "y"]].reset_index(drop=True)
    bt_scores: dict[str, np.ndarray] = {}
    for col in deconv_cols:
        bt = _broad_type_from_deconv_col(col)
        bt_scores.setdefault(bt, []).append(pd.to_numeric(prof[col], errors="coerce").fillna(0.0).values)
    bt_cols = []
    for bt, arrs in bt_scores.items():
        cc = np.vstack(arrs)
        vals = np.mean(cc, axis=0)
        cname = f"BT_{bt}"
        out[cname] = vals.astype(float)
        bt_cols.append(cname)
    if bt_cols:
        out["dominant_celltype"] = out[bt_cols].idxmax(axis=1).str.replace("BT_", "", regex=False)
    else:
        out["dominant_celltype"] = "Other"
    out["source_modality"] = "st_visium"
    return out


def _load_cosmx_native_celltypes(cosmx_h5ad: Path, max_cells: int = 140000) -> pd.DataFrame:
    if not cosmx_h5ad.exists():
        return pd.DataFrame()
    try:
        import anndata as ad
        adata = ad.read_h5ad(cosmx_h5ad, backed="r")
    except Exception:
        return pd.DataFrame()
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif {"x", "y"}.issubset(set(adata.obs.columns)):
        coords = adata.obs[["x", "y"]].to_numpy()
    else:
        return pd.DataFrame()
    ct_col = "cell_type" if "cell_type" in adata.obs.columns else ("final_anno" if "final_anno" in adata.obs.columns else None)
    if ct_col is None:
        return pd.DataFrame()
    samples = adata.obs["sample"].astype(str).values if "sample" in adata.obs.columns else np.array(["CosMx_IFNG"] * adata.n_obs)
    cts = adata.obs[ct_col].astype(str).values
    if len(cts) > max_cells:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(np.arange(len(cts)), size=max_cells, replace=False))
        coords = coords[idx]
        samples = samples[idx]
        cts = cts[idx]

    bt_map = {
        "EPI": "Epithelial",
        "FIBROENDOMUSCLE": "CAF",
        "MYELOID": "TAM",
        "T/NK": "CD8T",
        "T_OTHER": "CD4T",
        "PLASMA/B": "Other",
        "MAST": "Mast",
        "TNK": "CD8T",
        "NEUTROPHIL": "Neutrophil",
        "ENDO": "Endothelial",
    }
    broad = []
    for ct in cts:
        key = str(ct).upper()
        broad.append(bt_map.get(key, "Other"))
    out = pd.DataFrame(
        {
            "sample_id": [f"CosMx_{s}" for s in samples],
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
            "dominant_celltype": broad,
            "source_modality": "cosmx",
        }
    )
    return out


def _load_visiumhd_native_celltypes(
    visium_h5: Path,
    visium_pos: Path,
    max_spots: int = 120000,
) -> pd.DataFrame:
    if (not visium_h5.exists()) or (not visium_pos.exists()):
        return pd.DataFrame()
    try:
        import scanpy as sc
        adata = sc.read_10x_h5(str(visium_h5))
        adata.var_names_make_unique()
    except Exception:
        return pd.DataFrame()
    try:
        pos = pd.read_parquet(visium_pos)
    except Exception:
        return pd.DataFrame()
    if "barcode" not in pos.columns:
        return pd.DataFrame()
    pos = pos[pos.get("in_tissue", 1) == 1].set_index("barcode")
    keep = [b for b in adata.obs_names if b in pos.index]
    if not keep:
        return pd.DataFrame()
    adata = adata[keep].copy()
    coords = pos.loc[keep][["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    if adata.n_obs > max_spots:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_spots, replace=False))
        adata = adata[idx].copy()
        coords = coords[idx]

    markers = {
        "CAF": ["POSTN", "COL1A1", "COL1A2", "DCN", "FAP", "PDGFRA", "MFAP2"],
        "TAM": ["CD68", "LST1", "APOE", "C1QA", "C1QB", "FCER1G", "TYROBP"],
        "CD8T": ["CD3D", "CD3E", "CD8A", "CD8B", "NKG7", "GZMB"],
        "CD4T": ["CD3D", "CD3E", "IL7R", "LTB"],
        "DC": ["FCER1A", "CLEC10A", "CLEC9A", "LILRA4"],
        "Endothelial": ["VWF", "KDR", "EMCN", "PECAM1"],
        "Monocyte": ["S100A8", "S100A9", "LYZ", "CTSS"],
        "NK": ["NKG7", "GNLY", "KLRD1"],
        "Mast": ["TPSAB1", "KIT", "MS4A2"],
    }
    vu = {str(v).upper(): str(v) for v in adata.var_names}
    score = {}
    for bt, gs in markers.items():
        real = [vu[g.upper()] for g in gs if g.upper() in vu]
        if not real:
            score[bt] = np.zeros(adata.n_obs, dtype=float)
            continue
        x = adata[:, real].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        score[bt] = np.asarray(x).mean(axis=1).astype(float)
    score_df = pd.DataFrame(score)
    score_df = _normalize_rows(score_df)
    dom = score_df.idxmax(axis=1).astype(str).values
    out = pd.DataFrame(
        {
            "sample_id": "VisiumHD_16um",
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
            "dominant_celltype": dom,
            "source_modality": "visiumhd",
        }
    )
    return out


def _plot_native_celltype_spatial(data: pd.DataFrame, sample_id: str, out_path: Path) -> None:
    if data.empty:
        return
    idx = _subsample_idx(len(data), max_points=50000, seed=42)
    sub = data.iloc[idx].copy()
    labels = sub["dominant_celltype"].astype(str).values
    cmap = get_color_mapping(labels)
    colors = [cmap[l] for l in labels]
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.scatter(sub["x"], sub["y"], s=2, c=colors, alpha=0.78, edgecolors="none")
    ax.set_title(f"{sample_id} native cell-type spatial map")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    uniq = sorted(set(labels))
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=4, markerfacecolor=cmap[u], markeredgecolor="none", label=u)
        for u in uniq
    ]
    ax.legend(handles=handles, title="Cell type", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7)
    save_figure(fig, out_path, dpi=300, config={"chart": "native_celltype_spatial", "sample": sample_id})


def _build_native_celltype_spatial_pack(
    assignment: pd.DataFrame,
    st_dir: Path,
    out_dirs: dict[str, Path],
    cosmx_h5ad: Path,
    visium_h5: Path,
    visium_pos: Path,
) -> dict:
    produced = 0
    sample_stats = []
    # ST_CRC_MSS per sample
    for sid in sorted(set(assignment["sample_id"].astype(str).tolist())):
        sdf = assignment[assignment["sample_id"].astype(str) == sid].copy()
        if sdf.empty:
            continue
        native = _collect_st_native_celltypes_for_sample(st_dir, sid, sdf.sort_values("sample_spot_idx"))
        if native.empty:
            continue
        _plot_native_celltype_spatial(native, sid, out_dirs["native_celltype_spatial"] / f"native_celltype_{sid}.png")
        frac = native["dominant_celltype"].value_counts(normalize=True).to_dict()
        sample_stats.append({"sample_id": sid, "source_modality": "st_visium", "n_points": int(len(native)), "fractions": frac})
        produced += 1
    # CosMx
    cos_df = _load_cosmx_native_celltypes(cosmx_h5ad)
    if not cos_df.empty:
        for sid in sorted(set(cos_df["sample_id"].astype(str).tolist())):
            sdf = cos_df[cos_df["sample_id"].astype(str) == sid].copy()
            if sdf.empty:
                continue
            _plot_native_celltype_spatial(sdf, sid, out_dirs["native_celltype_spatial"] / f"native_celltype_{sid}.png")
            frac = sdf["dominant_celltype"].value_counts(normalize=True).to_dict()
            sample_stats.append({"sample_id": sid, "source_modality": "cosmx", "n_points": int(len(sdf)), "fractions": frac})
            produced += 1
    # VisiumHD
    vis_df = _load_visiumhd_native_celltypes(visium_h5, visium_pos)
    if not vis_df.empty:
        sid = "VisiumHD_16um"
        _plot_native_celltype_spatial(vis_df, sid, out_dirs["native_celltype_spatial"] / f"native_celltype_{sid}.png")
        frac = vis_df["dominant_celltype"].value_counts(normalize=True).to_dict()
        sample_stats.append({"sample_id": sid, "source_modality": "visiumhd", "n_points": int(len(vis_df)), "fractions": frac})
        produced += 1
    return {"native_celltype_maps": produced, "sample_stats": sample_stats}


def _plot_anchor_coloc_tripanel(
    sample_id: str,
    x: np.ndarray,
    y: np.ndarray,
    expr_norm: np.ndarray,
    type_score: np.ndarray,
    gene: str,
    celltype: str,
    out_path: Path,
) -> None:
    coloc = expr_norm * type_score
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    ax1, ax2, ax3 = axes
    sc1 = ax1.scatter(x, y, c=expr_norm, s=2, cmap=CMAP_SPATIAL, edgecolors="none", alpha=0.82)
    fig.colorbar(sc1, ax=ax1, shrink=0.75, pad=0.02, label=f"{gene} expr (scaled)")
    ax1.set_title(f"{sample_id} {gene} expression")
    ax1.invert_yaxis()
    sc2 = ax2.scatter(x, y, c=type_score, s=2, cmap=CMAP_SPATIAL, edgecolors="none", alpha=0.82)
    fig.colorbar(sc2, ax=ax2, shrink=0.75, pad=0.02, label=f"{celltype} score")
    ax2.set_title(f"{sample_id} {celltype} distribution")
    ax2.invert_yaxis()
    sc3 = ax3.scatter(x, y, c=coloc, s=2, cmap=CMAP_SPATIAL, edgecolors="none", alpha=0.85)
    fig.colorbar(sc3, ax=ax3, shrink=0.75, pad=0.02, label="co-localization score")
    ax3.set_title(f"{sample_id} {gene} x {celltype} co-localization")
    ax3.invert_yaxis()
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    save_figure(fig, out_path, dpi=300, config={"chart": "gene_celltype_colocalization_spatial", "sample": sample_id, "gene": gene, "celltype": celltype})


def _build_gene_celltype_colocalization_pack(
    assignment: pd.DataFrame,
    target_niche: pd.DataFrame,
    st_h5ad: Path,
    st_dir: Path,
    out_dirs: dict[str, Path],
    max_genes: int = 12,
    max_samples: int = 12,
    max_spots_per_sample: int = 12000,
) -> dict:
    if assignment.empty or target_niche.empty or (not st_h5ad.exists()):
        return {"colocalization_pairs": 0}
    try:
        import anndata as ad
        adata = ad.read_h5ad(st_h5ad, backed="r")
        adata.var_names_make_unique()
    except Exception:
        return {"colocalization_pairs": 0}
    if "sample_id" not in adata.obs.columns:
        return {"colocalization_pairs": 0}
    obs_sample = adata.obs["sample_id"].astype(str).values
    var_names = [str(v) for v in adata.var_names]
    gene_to_idx: dict[str, int] = {}
    for i, vv in enumerate(var_names):
        up = vv.upper()
        base = up.split("-")[0]
        if base not in gene_to_idx:
            gene_to_idx[base] = i
        if up not in gene_to_idx:
            gene_to_idx[up] = i
    top_genes = (
        target_niche[["target_gene", "global_rank"]]
        .drop_duplicates()
        .sort_values("global_rank")
        .head(max(3, int(max_genes)))["target_gene"]
        .astype(str)
        .str.upper()
        .tolist()
    )
    anchors = ["POSTN", "MFAP2", "INHBA"]
    genes = []
    for g in top_genes + anchors:
        if g not in genes and g in gene_to_idx:
            genes.append(g)
    if not genes:
        return {"colocalization_pairs": 0}

    corr_rows = []
    spatial_saved = 0
    sample_list = sorted(set(assignment["sample_id"].astype(str).tolist()))
    for sid in sample_list[: max(1, int(max_samples))]:
        if not (st_dir / f"STmetadata_{sid}.csv").exists():
            continue
        sdf = assignment[assignment["sample_id"].astype(str) == sid].copy().sort_values("sample_spot_idx")
        native = _collect_st_native_celltypes_for_sample(st_dir, sid, sdf)
        if native.empty:
            continue
        if len(native) > max_spots_per_sample:
            idx_sub = _subsample_idx(len(native), max_points=max_spots_per_sample, seed=42)
            native = native.iloc[idx_sub].copy().reset_index(drop=True)
        obs_idx = np.where(obs_sample == sid)[0]
        if len(obs_idx) == 0:
            continue
        sp_idx = pd.to_numeric(native["sample_spot_idx"], errors="coerce").fillna(-1).astype(int).values
        valid = (sp_idx >= 0) & (sp_idx < len(obs_idx))
        if not np.any(valid):
            continue
        use_obs = obs_idx[sp_idx[valid]]
        x = pd.to_numeric(native.loc[valid, "x"], errors="coerce").fillna(0.0).values.astype(float)
        y = pd.to_numeric(native.loc[valid, "y"], errors="coerce").fillna(0.0).values.astype(float)
        bt_cols = [c for c in native.columns if c.startswith("BT_")]
        if not bt_cols:
            continue
        bt_mat = native.loc[valid, bt_cols].to_numpy(dtype=float)
        bt_types = [c.replace("BT_", "") for c in bt_cols]

        sample_gene_corr = {}
        for g in genes:
            vg_idx = int(gene_to_idx[g])
            xx = adata[use_obs, vg_idx].X
            expr = xx.toarray().reshape(-1) if hasattr(xx, "toarray") else np.asarray(xx).reshape(-1)
            expr = np.asarray(expr, dtype=float)
            if expr.shape[0] != bt_mat.shape[0]:
                nmin = min(expr.shape[0], bt_mat.shape[0])
                expr = expr[:nmin]
                bt_loc = bt_mat[:nmin, :]
                x_loc = x[:nmin]
                y_loc = y[:nmin]
            else:
                bt_loc = bt_mat
                x_loc = x
                y_loc = y
            q1 = float(np.nanquantile(expr, 0.05))
            q2 = float(np.nanquantile(expr, 0.95))
            expr_norm = np.clip((expr - q1) / max(q2 - q1, 1e-12), 0.0, 1.0)
            sample_gene_corr[g] = {}
            for j, bt in enumerate(bt_types):
                vv = bt_loc[:, j]
                if np.nanstd(expr) < 1e-12 or np.nanstd(vv) < 1e-12:
                    cc = 0.0
                else:
                    cc = float(np.corrcoef(expr, vv)[0, 1])
                    if not np.isfinite(cc):
                        cc = 0.0
                corr_rows.append(
                    {
                        "sample_id": sid,
                        "gene": g,
                        "celltype": bt,
                        "corr": cc,
                    }
                )
                sample_gene_corr[g][bt] = cc
            if g in anchors:
                top_bt = sorted(sample_gene_corr[g].items(), key=lambda kv: kv[1], reverse=True)[0][0]
                bt_idx = bt_types.index(top_bt)
                _plot_anchor_coloc_tripanel(
                    sample_id=sid,
                    x=x_loc,
                    y=y_loc,
                    expr_norm=expr_norm,
                    type_score=bt_loc[:, bt_idx],
                    gene=g,
                    celltype=top_bt,
                    out_path=out_dirs["colocalization_spatial"] / f"coloc_spatial_{sid}_{g}_{top_bt}.png",
                )
                spatial_saved += 1

    if not corr_rows:
        return {"colocalization_pairs": 0}
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dirs["root"] / "gene_celltype_colocalization_correlations.csv", index=False)

    # heatmap: mean correlation over samples
    heat = (
        corr_df.groupby(["gene", "celltype"], as_index=False)["corr"]
        .mean()
        .pivot(index="gene", columns="celltype", values="corr")
        .fillna(0.0)
    )
    if not heat.empty:
        vmax = max(float(np.max(np.abs(heat.values))), 1e-6)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(heat))))
        im = ax.imshow(heat.values, aspect="auto", cmap=CMAP_SPATIAL_DIVERGING, vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02, label="mean correlation")
        ax.set_xticks(range(heat.shape[1]))
        ax.set_xticklabels(heat.columns, rotation=30, ha="right")
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels(heat.index)
        ax.set_title("Gene-celltype spatial co-localization heatmap")
        save_figure(fig, out_dirs["root"] / "gene_celltype_colocalization_heatmap.png", dpi=300, config={"chart": "gene_celltype_colocalization_heatmap"})

    # violin: per gene, distribution of correlation across samples per celltype
    violin_genes = anchors + [g for g in heat.index.tolist() if g not in anchors][:7]
    for g in violin_genes:
        sub = corr_df[corr_df["gene"] == g].copy()
        if sub.empty:
            continue
        cells = sorted(sub["celltype"].astype(str).unique().tolist())
        groups = [
            pd.to_numeric(sub[sub["celltype"].astype(str) == c]["corr"], errors="coerce").dropna().values
            for c in cells
        ]
        if not any(len(v) > 0 for v in groups):
            continue
        fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(cells)), 4.6))
        parts = ax.violinplot(
            groups,
            positions=np.arange(1, len(cells) + 1),
            widths=0.85,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#E78F7D")
            pc.set_edgecolor("#D85D5A")
            pc.set_alpha(0.72)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#6FAFC2")
            parts["cmedians"].set_linewidth(1.1)
        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_xticks(np.arange(1, len(cells) + 1))
        ax.set_xticklabels(cells, rotation=30, ha="right")
        ax.set_ylabel("correlation across spots")
        ax.set_title(f"{g} co-localization with cell types (violin)")
        save_figure(fig, out_dirs["colocalization_violin"] / f"coloc_violin_{g}.png", dpi=300, config={"chart": "gene_celltype_coloc_violin", "gene": g})

    return {
        "colocalization_pairs": int(corr_df[["gene", "celltype"]].drop_duplicates().shape[0]),
        "colocalization_spatial_maps": int(spatial_saved),
        "genes_used": sorted(set(corr_df["gene"].astype(str).tolist())),
        "celltypes_used": sorted(set(corr_df["celltype"].astype(str).tolist())),
    }


def _plot_sample_maps(
    sample_id: str,
    sample_df: pd.DataFrame,
    definition: pd.DataFrame,
    target_niche: pd.DataFrame,
    type_flow: dict[tuple[str, str], float],
    out_dirs: dict[str, Path],
    selected_targets: list[str],
    niche_color_map: dict[int, str] | None = None,
) -> None:
    idx = _subsample_idx(len(sample_df), max_points=20000, seed=42)
    sub = sample_df.iloc[idx].copy()
    niches = sorted(sub["niche_id"].astype(int).unique().tolist())
    cmap = plt.get_cmap("tab20")
    niche_color = {}
    for i, n in enumerate(niches):
        if niche_color_map and int(n) in niche_color_map:
            niche_color[n] = niche_color_map[int(n)]
        else:
            niche_color[n] = cmap(i % 20)
    colors = [niche_color[int(n)] for n in sub["niche_id"].astype(int).values]

    # 1) Niche distribution map
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(sub["x"], sub["y"], s=2, c=colors, alpha=0.75, edgecolors="none")
    ax.set_title(f"{sample_id} niche spatial distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    save_figure(fig, out_dirs["niche_distribution"] / f"niche_distribution_{sample_id}.png", dpi=300, config={"sample": sample_id})

    # 2) Static target maps (niche-informed)
    if not target_niche.empty:
        for tg in selected_targets:
            sub_t = target_niche[target_niche["target_gene"].str.upper() == tg.upper()]
            if sub_t.empty:
                continue
            score_map = {
                int(r["niche_id"]): float(r["weighted_expression"])
                for _, r in sub_t.iterrows()
            }
            vals = sub["niche_id"].astype(int).map(score_map).fillna(0.0).values
            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            sc = ax.scatter(sub["x"], sub["y"], c=vals, s=2, cmap=CMAP_SPATIAL, alpha=0.8, edgecolors="none")
            fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, label="niche-informed target score")
            ax.set_title(f"{sample_id} static target distribution: {tg}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.invert_yaxis()
            save_figure(fig, out_dirs["target_static"] / f"target_static_{sample_id}_{tg}.png", dpi=300, config={"sample": sample_id, "target": tg})

    # 3) Info flow direction map
    cent = sample_df.groupby("niche_id", as_index=False)[["x", "y"]].mean()
    niche_dom = {
        int(r["niche_id"]): str(r["dominant_type"])
        for _, r in definition.iterrows()
    }
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for _, r in cent.iterrows():
        nid = int(r["niche_id"])
        ax.scatter(r["x"], r["y"], s=90, c=[niche_color.get(nid, "#999999")], edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(r["x"], r["y"], f"N{nid}", fontsize=8, ha="center", va="center", zorder=4)
    wmax = max([v for v in type_flow.values()] + [1e-6])
    for _, srow in cent.iterrows():
        sid = int(srow["niche_id"])
        st = niche_dom.get(sid, "")
        for _, trow in cent.iterrows():
            tid = int(trow["niche_id"])
            if sid == tid:
                continue
            tt = niche_dom.get(tid, "")
            w = type_flow.get((st, tt), 0.0)
            if w <= 0:
                continue
            ax.annotate(
                "",
                xy=(float(trow["x"]), float(trow["y"])),
                xytext=(float(srow["x"]), float(srow["y"])),
                arrowprops=dict(arrowstyle="->", color="#444444", linewidth=0.4 + 2.0 * (w / wmax), alpha=0.5),
                zorder=2,
            )
    ax.set_title(f"{sample_id} recognized information flow")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    save_figure(fig, out_dirs["info_flow"] / f"info_flow_{sample_id}.png", dpi=300, config={"sample": sample_id})


def _target_niche_heatmap(target_niche: pd.DataFrame, out_path: Path) -> None:
    if target_niche.empty:
        return
    top_targets = (
        target_niche[["target_gene", "global_rank"]]
        .drop_duplicates()
        .sort_values("global_rank")
        .head(24)["target_gene"]
        .tolist()
    )
    mat = (
        target_niche[target_niche["target_gene"].isin(top_targets)]
        .pivot(index="target_gene", columns="niche_name", values="z_score_within_target")
        .fillna(0.0)
    )
    if mat.empty:
        return
    vmax = max(float(np.max(np.abs(mat.values))), 1e-6)
    fig, ax = plt.subplots(figsize=(11, max(5, len(mat) * 0.3)))
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP_SPATIAL_DIVERGING, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="target-level z-score")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title("Target-Niche Matrix (for spatial communication)")
    save_figure(fig, out_path, dpi=300, config={"chart": "target_niche_heatmap_spatial_comm"})


def _top30_target_niche_panels(
    target_niche: pd.DataFrame,
    definition: pd.DataFrame,
    out_dir: Path,
    hierarchy_json: dict | None = None,
) -> None:
    if target_niche.empty:
        return
    top30 = (
        target_niche[["target_gene", "global_rank"]]
        .drop_duplicates()
        .sort_values("global_rank")
        .head(30)["target_gene"]
        .tolist()
    )
    mat = (
        target_niche[target_niche["target_gene"].isin(top30)]
        .pivot(index="target_gene", columns="niche_name", values="z_score_within_target")
        .fillna(0.0)
    )
    if mat.empty:
        return
    ordered_rows = [g for g in top30 if g in mat.index] + [g for g in mat.index if g not in top30]
    mat = mat.loc[ordered_rows]

    # all-niche heatmap
    vmax = max(float(np.max(np.abs(mat.values))), 1e-6)
    fig, ax = plt.subplots(figsize=(12, max(6, len(mat) * 0.28)))
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP_SPATIAL_DIVERGING, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="target-level z-score")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    for tick in ax.get_yticklabels():
        if tick.get_text().upper() in {"POSTN", "MFAP2", "INHBA"}:
            tick.set_color("#D55E00")
            tick.set_fontweight("bold")
    ax.set_title("Top30 Target-Niche Enrichment (All Niches)")
    save_figure(fig, out_dir / "top30_target_niche_heatmap_all.png", dpi=300, config={"chart": "top30_target_niche_heatmap_all"})

    # hierarchy-ordered heatmap
    col_order = list(mat.columns)
    if hierarchy_json and isinstance(hierarchy_json, dict):
        order_ids = hierarchy_json.get("niche_order_by_radius", [])
        id_to_name = {
            int(r["niche_id"]): str(r["niche_name"]) for _, r in definition.iterrows()
        }
        ord_names = [id_to_name.get(int(i), "") for i in order_ids]
        ord_names = [n for n in ord_names if n in mat.columns]
        if ord_names:
            col_order = ord_names + [c for c in mat.columns if c not in ord_names]
    hm = mat[col_order]
    fig, ax = plt.subplots(figsize=(12, max(6, len(hm) * 0.28)))
    im = ax.imshow(hm.values, aspect="auto", cmap=CMAP_SPATIAL_DIVERGING, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="target-level z-score")
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels(hm.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(hm.shape[0]))
    ax.set_yticklabels(hm.index, fontsize=8)
    for tick in ax.get_yticklabels():
        if tick.get_text().upper() in {"POSTN", "MFAP2", "INHBA"}:
            tick.set_color("#D55E00")
            tick.set_fontweight("bold")
    ax.set_title("Top30 Target-Niche Enrichment (Hierarchy Ordered)")
    save_figure(fig, out_dir / "top30_target_niche_heatmap_hierarchy.png", dpi=300, config={"chart": "top30_target_niche_heatmap_hierarchy"})

    # dotplot
    fig, ax = plt.subplots(figsize=(12, max(6, len(mat) * 0.30)))
    for yi, tg in enumerate(mat.index):
        for xi, niche_name in enumerate(mat.columns):
            val = float(mat.loc[tg, niche_name])
            size = 24 + 140 * min(abs(val) / vmax, 1.0)
            color = "#D73027" if val >= 0 else "#4575B4"
            edge = "#D55E00" if tg.upper() in {"POSTN", "MFAP2", "INHBA"} else "white"
            lw = 0.8 if tg.upper() in {"POSTN", "MFAP2", "INHBA"} else 0.3
            ax.scatter(xi, yi, s=size, c=color, alpha=0.8, edgecolors=edge, linewidths=lw)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_title("Top30 Target-Niche Dotplot (anchors highlighted)")
    save_figure(fig, out_dir / "top30_target_niche_dotplot_all.png", dpi=300, config={"chart": "top30_target_niche_dotplot_all"})


def _load_step3_anchor_tables(step3_dir: Path, anchors: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for g in anchors:
        p = step3_dir / f"targets_{g.upper()}.csv"
        if not p.exists():
            p = step3_dir / f"targets_{g}.csv"
        if not p.exists():
            out[g.upper()] = pd.DataFrame()
            continue
        try:
            out[g.upper()] = pd.read_csv(p)
        except Exception:
            out[g.upper()] = pd.DataFrame()
    return out


def _aggregate_type_flow_from_step3(step3_target_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    if step3_target_df.empty:
        return {}
    need_cols = ["source_type", "target_type"]
    if any(c not in step3_target_df.columns for c in need_cols):
        return {}
    w = np.ones(len(step3_target_df), dtype=float)
    if "flow_weight" in step3_target_df.columns:
        w *= pd.to_numeric(step3_target_df["flow_weight"], errors="coerce").fillna(1.0).values
    if "target_priority_score" in step3_target_df.columns:
        w *= np.maximum(pd.to_numeric(step3_target_df["target_priority_score"], errors="coerce").fillna(0.0).values, 0.0)
    if "combined_abs_delta" in step3_target_df.columns:
        w *= np.maximum(pd.to_numeric(step3_target_df["combined_abs_delta"], errors="coerce").fillna(0.0).values, 1e-6)
    sub = step3_target_df.copy()
    sub["_w"] = w
    agg = sub.groupby(["source_type", "target_type"], as_index=False)["_w"].sum()
    return {
        (str(r["source_type"]), str(r["target_type"])): float(r["_w"])
        for _, r in agg.iterrows()
        if float(r["_w"]) > 0
    }


def _anchor_niche_score_map(target_niche: pd.DataFrame, anchor_gene: str) -> dict[int, float]:
    if target_niche.empty:
        return {}
    sub = target_niche[target_niche["target_gene"].astype(str).str.upper() == anchor_gene.upper()].copy()
    if sub.empty:
        return {}
    if "weighted_expression" in sub.columns:
        score_col = "weighted_expression"
    elif "z_score_within_target" in sub.columns:
        score_col = "z_score_within_target"
    else:
        return {}
    return {
        int(r["niche_id"]): float(r[score_col])
        for _, r in sub.iterrows()
    }


def _normalize_for_display(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return np.zeros_like(v, dtype=float)
    lo = float(np.nanquantile(v, 0.05))
    hi = float(np.nanquantile(v, 0.95))
    if hi <= lo:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
    den = max(hi - lo, 1e-12)
    vn = (np.clip(v, lo, hi) - lo) / den
    return vn


def _fallback_niche_group(
    all_niches: list[int],
    score_order: list[int],
    top_n: int = 2,
) -> list[int]:
    if score_order:
        return score_order[: min(top_n, len(score_order))]
    return all_niches[: min(top_n, len(all_niches))]


def _render_perturbation_flow_map(
    sample_id: str,
    sample_df: pd.DataFrame,
    definition: pd.DataFrame,
    target_niche: pd.DataFrame,
    niche_color_map: dict[int, str],
    anchor_gene: str,
    step3_target_df: pd.DataFrame,
    out_path: Path,
) -> None:
    type_flow = _aggregate_type_flow_from_step3(step3_target_df)
    if not type_flow:
        return
    cent = sample_df.groupby("niche_id", as_index=False)[["x", "y"]].mean()
    if cent.empty:
        return
    niche_dom = {
        int(r["niche_id"]): str(r["dominant_type"])
        for _, r in definition.iterrows()
    }
    niche_size = sample_df["niche_id"].astype(int).value_counts().to_dict()
    score_map = _anchor_niche_score_map(target_niche, anchor_gene)
    all_niches = sorted(cent["niche_id"].astype(int).tolist())
    score_order = sorted(all_niches, key=lambda n: float(score_map.get(int(n), 0.0)), reverse=True)

    # expand type-level flow to niche-level weighted edges
    niche_edges: list[tuple[int, int, float]] = []
    for (st, tt), tw in type_flow.items():
        src_n = [int(r["niche_id"]) for _, r in cent.iterrows() if niche_dom.get(int(r["niche_id"]), "") == st]
        tgt_n = [int(r["niche_id"]) for _, r in cent.iterrows() if niche_dom.get(int(r["niche_id"]), "") == tt]
        # Fallback for cross-platform type mismatch (e.g. no explicit Treg-dominant niche).
        if not src_n:
            src_n = _fallback_niche_group(all_niches, score_order, top_n=2)
        if not tgt_n:
            tgt_n = _fallback_niche_group(all_niches, score_order, top_n=2)
        src_sum = float(sum(max(int(niche_size.get(n, 1)), 1) for n in src_n))
        tgt_sum = float(sum(max(int(niche_size.get(n, 1)), 1) for n in tgt_n))
        for s in src_n:
            for t in tgt_n:
                if s == t:
                    continue
                ws = max(int(niche_size.get(s, 1)), 1) / max(src_sum, 1e-12)
                wt = max(int(niche_size.get(t, 1)), 1) / max(tgt_sum, 1e-12)
                niche_edges.append((s, t, float(tw * ws * wt)))
    if not niche_edges:
        return

    id2xy = {int(r["niche_id"]): (float(r["x"]), float(r["y"])) for _, r in cent.iterrows()}
    x = sample_df["x"].astype(float).values
    y = sample_df["y"].astype(float).values
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    gx = np.linspace(x_min, x_max, 42)
    gy = np.linspace(y_min, y_max, 42)
    Xg, Yg = np.meshgrid(gx, gy)
    U = np.zeros_like(Xg, dtype=float)
    V = np.zeros_like(Yg, dtype=float)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    sigma = 0.16 * span
    max_w = max(w for _, _, w in niche_edges)

    for s, t, w in niche_edges:
        if s not in id2xy or t not in id2xy:
            continue
        sx, sy = id2xy[s]
        tx, ty = id2xy[t]
        dx, dy = tx - sx, ty - sy
        dn = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / dn, dy / dn
        amp = float(w / max(max_w, 1e-12))
        g = np.exp(-((Xg - sx) ** 2 + (Yg - sy) ** 2) / (2.0 * sigma * sigma))
        U += amp * ux * g
        V += amp * uy * g

    speed = np.sqrt(U * U + V * V)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    # COMMOT-like layered rendering: target-expression heatmap background + flow field overlay.
    point_vals = sample_df["niche_id"].astype(int).map(score_map).values if score_map else np.zeros(len(sample_df))
    point_vals = _normalize_for_display(np.asarray(point_vals, dtype=float))
    bg = ax.scatter(
        x, y, s=2.0, c=point_vals, cmap=CMAP_SPATIAL, alpha=0.65, edgecolors="none", zorder=0
    )
    cbar_bg = fig.colorbar(bg, ax=ax, shrink=0.72, pad=0.02)
    cbar_bg.set_label(f"{anchor_gene} niche-expression (scaled)")
    ax.contourf(
        Xg, Yg, speed,
        levels=8,
        cmap="Greys",
        alpha=0.12,
        zorder=1,
    )
    try:
        st = ax.streamplot(
            gx, gy, U, V,
            color=speed,
            cmap="Greys",
            density=1.15,
            linewidth=0.4 + 1.4 * np.clip(speed / max(float(np.max(speed)), 1e-12), 0.0, 1.0),
            arrowsize=0.8,
            zorder=2,
        )
        cbar_flow = fig.colorbar(st.lines, ax=ax, shrink=0.72, pad=0.08)
        cbar_flow.set_label("flow speed")
    except Exception:
        skip = (slice(None, None, 2), slice(None, None, 2))
        qv = ax.quiver(
            Xg[skip], Yg[skip], U[skip], V[skip],
            np.sqrt(U[skip] * U[skip] + V[skip] * V[skip]),
            cmap="Greys", alpha=0.8, width=0.002, zorder=2,
        )
        cbar_flow = fig.colorbar(qv, ax=ax, shrink=0.72, pad=0.08)
        cbar_flow.set_label("flow speed")

    for nid, (cx, cy) in id2xy.items():
        ax.scatter(
            cx, cy, s=96,
            c=[niche_color_map.get(int(nid), "#9E9E9E")],
            edgecolors="white", linewidths=0.8, zorder=3,
        )
        ax.text(cx, cy, f"N{int(nid)}", fontsize=8, ha="center", va="center", zorder=4)

    edge_scale = max([w for _, _, w in niche_edges] + [1e-12])
    for s, t, w in niche_edges:
        if s not in id2xy or t not in id2xy:
            continue
        sx, sy = id2xy[s]
        tx, ty = id2xy[t]
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="->",
                color="#303030",
                linewidth=0.35 + 2.2 * float(w / edge_scale),
                alpha=0.35,
            ),
            zorder=2,
        )

    ax.set_title(f"{sample_id} perturbation-driven communication flow ({anchor_gene})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    save_figure(fig, out_path, dpi=300, config={"chart": "perturbation_flow_commot_style", "sample": sample_id, "anchor": anchor_gene})


def _read_spot_expression_from_h5ad(
    h5ad_path: Path,
    assignment: pd.DataFrame,
    genes: list[str],
) -> tuple[pd.DataFrame, set[str]]:
    if not h5ad_path.exists() or assignment.empty or not genes:
        return pd.DataFrame(), set()
    try:
        import anndata as ad
    except Exception:
        return pd.DataFrame(), set()
    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
    except Exception:
        return pd.DataFrame(), set()
    if "sample_id" not in adata.obs.columns:
        return pd.DataFrame(), set()
    obs_sample = adata.obs["sample_id"].astype(str).values
    var_upper = {str(v).upper(): str(v) for v in adata.var_names}
    use_genes = [g for g in genes if g.upper() in var_upper]
    if not use_genes:
        return pd.DataFrame(), set()

    recs: list[pd.DataFrame] = []
    for sid, sdf in assignment.groupby("sample_id"):
        sid = str(sid)
        obs_idx = np.where(obs_sample == sid)[0]
        if len(obs_idx) == 0:
            continue
        sdf2 = sdf.sort_values("sample_spot_idx").copy()
        sp_idx = pd.to_numeric(sdf2["sample_spot_idx"], errors="coerce").fillna(-1).astype(int).values
        valid = (sp_idx >= 0) & (sp_idx < len(obs_idx))
        if not np.any(valid):
            continue
        use_obs = obs_idx[sp_idx[valid]]
        use_nid = sdf2["niche_id"].astype(int).values[valid]
        use_nname = sdf2["niche_name"].astype(str).values[valid]

        for g in use_genes:
            vg = var_upper[g.upper()]
            xx = adata[use_obs, [vg]].X
            vals = xx.toarray().reshape(-1) if hasattr(xx, "toarray") else np.asarray(xx).reshape(-1)
            recs.append(
                pd.DataFrame(
                    {
                        "sample_id": sid,
                        "niche_id": use_nid,
                        "niche_name": use_nname,
                        "target_gene": g.upper(),
                        "value": vals.astype(float),
                        "value_source": "st_expression",
                    }
                )
            )
    if not recs:
        return pd.DataFrame(), set()
    out = pd.concat(recs, axis=0, ignore_index=True)
    return out, set([g.upper() for g in use_genes])


def _fallback_niche_weight_distribution(
    assignment: pd.DataFrame,
    target_niche: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    if assignment.empty or target_niche.empty or not genes:
        return pd.DataFrame()
    base = assignment[["sample_id", "niche_id", "niche_name"]].copy()
    recs = []
    for g in genes:
        sub = target_niche[target_niche["target_gene"].astype(str).str.upper() == g.upper()]
        if sub.empty:
            continue
        score_map = {
            int(r["niche_id"]): float(r["weighted_expression"])
            for _, r in sub.iterrows()
        }
        df = base.copy()
        df["target_gene"] = g.upper()
        df["value"] = df["niche_id"].astype(int).map(score_map).fillna(0.0).astype(float)
        df["value_source"] = "niche_weighted_score"
        recs.append(df)
    if not recs:
        return pd.DataFrame()
    return pd.concat(recs, axis=0, ignore_index=True)


def _draw_single_gene_violin(
    gene: str,
    data: pd.DataFrame,
    niche_order: list[str],
    out_path: Path,
) -> None:
    if data.empty:
        return
    available = [n for n in niche_order if n in set(data["niche_name"].astype(str))]
    if not available:
        return
    groups = [
        pd.to_numeric(
            data[data["niche_name"].astype(str) == n]["value"],
            errors="coerce",
        ).dropna().values
        for n in available
    ]
    if not any(len(g) > 0 for g in groups):
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(available)), 4.6))
    parts = ax.violinplot(
        groups,
        positions=np.arange(1, len(available) + 1),
        widths=0.85,
        showmeans=False,
        showextrema=False,
        showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#5E88C5")
        pc.set_edgecolor("#2A5CAA")
        pc.set_alpha(0.70)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#D95F0E")
        parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks(np.arange(1, len(available) + 1))
    ax.set_xticklabels(available, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("expression / niche score")
    anchor_tag = " (anchor)" if gene.upper() in {"POSTN", "MFAP2", "INHBA"} else ""
    ax.set_title(f"{gene.upper()} by niche{anchor_tag}")
    save_figure(fig, out_path, dpi=300, config={"chart": "violin_gene_by_niche", "gene": gene.upper()})


def _build_violin_pack(
    assignment: pd.DataFrame,
    definition: pd.DataFrame,
    target_niche: pd.DataFrame,
    out_dirs: dict[str, Path],
    st_h5ad: Path,
) -> dict:
    if target_niche.empty or assignment.empty:
        return {"violin_gene_count": 0, "value_source": "none"}
    top30 = (
        target_niche[["target_gene", "global_rank"]]
        .drop_duplicates()
        .sort_values("global_rank")
        .head(30)["target_gene"]
        .astype(str)
        .str.upper()
        .tolist()
    )
    anchors = ["POSTN", "MFAP2", "INHBA"]
    genes = []
    for g in top30 + anchors:
        if g not in genes:
            genes.append(g)
    order_df = definition.sort_values(["hierarchy_level", "niche_id"])[["niche_name"]].drop_duplicates()
    niche_order = order_df["niche_name"].astype(str).tolist()

    st_long, st_found = _read_spot_expression_from_h5ad(st_h5ad, assignment, genes)
    miss = [g for g in genes if g.upper() not in st_found]
    fb_long = _fallback_niche_weight_distribution(assignment, target_niche, miss)
    long_df = pd.concat([st_long, fb_long], axis=0, ignore_index=True) if not st_long.empty or not fb_long.empty else pd.DataFrame()
    if long_df.empty:
        return {"violin_gene_count": 0, "value_source": "none"}

    summary = (
        long_df.groupby(["target_gene", "niche_name"], as_index=False)["value"]
        .agg(
            median="median",
            q25=lambda x: float(np.nanquantile(x, 0.25)),
            q75=lambda x: float(np.nanquantile(x, 0.75)),
            mean="mean",
            n="count",
        )
    )
    summary.to_csv(out_dirs["root"] / "top30_violin_summary.csv", index=False)

    for g in genes:
        sub = long_df[long_df["target_gene"].astype(str).str.upper() == g.upper()].copy()
        if sub.empty:
            continue
        _draw_single_gene_violin(
            gene=g,
            data=sub,
            niche_order=niche_order,
            out_path=out_dirs["violin"] / f"violin_{g.upper()}_by_niche.png",
        )

    anchor_df = long_df[long_df["target_gene"].isin(anchors)].copy()
    if not anchor_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(max(13.5, 0.35 * len(niche_order) * 3), 4.2), sharey=True)
        for ax, g in zip(axes, anchors):
            sub = anchor_df[anchor_df["target_gene"] == g]
            avail = [n for n in niche_order if n in set(sub["niche_name"].astype(str))]
            groups = [
                pd.to_numeric(sub[sub["niche_name"].astype(str) == n]["value"], errors="coerce").dropna().values
                for n in avail
            ]
            if any(len(v) > 0 for v in groups):
                parts = ax.violinplot(
                    groups,
                    positions=np.arange(1, len(avail) + 1),
                    widths=0.8,
                    showmeans=False,
                    showextrema=False,
                    showmedians=True,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor("#7CAFD3")
                    pc.set_edgecolor("#2A5CAA")
                    pc.set_alpha(0.70)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("#D95F0E")
                    parts["cmedians"].set_linewidth(1.1)
            ax.set_xticks(np.arange(1, len(avail) + 1))
            ax.set_xticklabels(avail, rotation=35, ha="right", fontsize=7)
            ax.set_title(g, color="#D95F0E", fontweight="bold")
            if g == anchors[0]:
                ax.set_ylabel("expression / niche score")
        save_figure(fig, out_dirs["root"] / "anchor_violin_panel.png", dpi=300, config={"chart": "anchor_violin_panel"})

    return {
        "violin_gene_count": int(len(set(long_df["target_gene"].astype(str).tolist()))),
        "st_expr_genes": sorted([str(x) for x in st_found]),
        "fallback_genes": sorted([str(x) for x in miss]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate spatial communication figure pack.")
    parser.add_argument("--st-dir", default="G:/ST_CRC_MSS", help="Directory of STmetadata_*.csv")
    parser.add_argument("--niche-dir", default="results/integration/discovery/niche")
    parser.add_argument("--flow-edges-json", default="results/integration/discovery/hyperbolic/step2/flow_edges.json")
    parser.add_argument("--step3-target-dir", default="results/integration/discovery/hyperbolic/step3")
    parser.add_argument("--st-expression-h5ad", default="data/ST/ST_CRC_MSS/expression.h5ad")
    parser.add_argument("--cosmx-h5ad", default="data/ST/scCRC_IFNG_CosMx/expression.h5ad")
    parser.add_argument("--visiumhd-h5", default="data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/filtered_feature_bc_matrix.h5")
    parser.add_argument("--visiumhd-pos", default="data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/spatial/tissue_positions.parquet")
    parser.add_argument("--coloc-max-genes", type=int, default=12)
    parser.add_argument("--coloc-max-samples", type=int, default=12)
    parser.add_argument("--coloc-max-spots-per-sample", type=int, default=12000)
    parser.add_argument("--output-dir", default="results/figures/spatial_comm")
    args = parser.parse_args()

    apply_cns_style()
    st_dir = Path(args.st_dir)
    niche_dir = ROOT / args.niche_dir
    out_dirs = _ensure_dirs(ROOT / args.output_dir)

    assignment_path = niche_dir / "spot_niche_assignment.csv"
    definition_path = niche_dir / "unified_niche_definition.csv"
    target_niche_path = niche_dir / "target_niche_expression.csv"
    metrics_path = niche_dir / "niche_hierarchy_metrics.json"
    hierarchy_path = niche_dir / "niche_hierarchy.json"
    color_path = niche_dir / "niche_color_map.json"

    if not assignment_path.exists() or not definition_path.exists():
        print("[ERROR] missing niche definition outputs. Run target discovery niche phase first.")
        return 1

    assignment = pd.read_csv(assignment_path)
    definition = pd.read_csv(definition_path)
    target_niche = pd.read_csv(target_niche_path) if target_niche_path.exists() else pd.DataFrame()
    flow_edges = _safe_json(ROOT / args.flow_edges_json)
    metrics = _safe_json(metrics_path)
    hierarchy_json = _safe_json(hierarchy_path)
    color_json = _safe_json(color_path)
    niche_color_map = {}
    if isinstance(color_json, dict):
        niche_color_map = {int(k): v for k, v in color_json.items()}

    native_celltype_meta = _build_native_celltype_spatial_pack(
        assignment=assignment,
        st_dir=st_dir,
        out_dirs=out_dirs,
        cosmx_h5ad=ROOT / args.cosmx_h5ad,
        visium_h5=ROOT / args.visiumhd_h5,
        visium_pos=ROOT / args.visiumhd_pos,
    )

    # global hierarchy figure
    _plot_global_hierarchy_advantage(
        assignment=assignment,
        metrics=metrics if isinstance(metrics, dict) else {},
        out_path=out_dirs["root"] / "stage1_hyperbolic_hierarchy_advantage.png",
    )

    # global target-niche matrix for spatial communication
    _target_niche_heatmap(
        target_niche=target_niche,
        out_path=out_dirs["root"] / "target_niche_heatmap_spatial_comm.png",
    )
    _top30_target_niche_panels(
        target_niche=target_niche,
        definition=definition,
        out_dir=out_dirs["root"],
        hierarchy_json=hierarchy_json if isinstance(hierarchy_json, dict) else {},
    )

    # fixed anchor targets for per-sample maps
    selected_targets = ["POSTN", "MFAP2", "INHBA"]
    type_flow = _type_flow_weights(flow_edges if isinstance(flow_edges, list) else [])
    step3_anchor_tables = _load_step3_anchor_tables(ROOT / args.step3_target_dir, selected_targets)

    # per-sample figures
    samples = sorted(assignment["sample_id"].astype(str).unique().tolist())
    for sid in samples:
        sdf = assignment[assignment["sample_id"].astype(str) == sid].copy()
        if sdf.empty:
            continue
        _plot_sample_maps(
            sample_id=sid,
            sample_df=sdf,
            definition=definition,
            target_niche=target_niche,
            type_flow=type_flow,
            out_dirs=out_dirs,
            selected_targets=selected_targets,
            niche_color_map=niche_color_map,
        )
        _sample_enrichment_heatmap(
            st_dir=st_dir,
            sample_id=sid,
            sample_assign=sdf.sort_values("sample_spot_idx"),
            out_path=out_dirs["niche_enrichment"] / f"niche_enrichment_{sid}.png",
        )
        for tg in selected_targets:
            _render_perturbation_flow_map(
                sample_id=sid,
                sample_df=sdf,
                definition=definition,
                target_niche=target_niche,
                niche_color_map=niche_color_map,
                anchor_gene=tg,
                step3_target_df=step3_anchor_tables.get(tg.upper(), pd.DataFrame()),
                out_path=out_dirs["perturb_flow"] / f"perturb_flow_{sid}_{tg}.png",
            )

    violin_meta = _build_violin_pack(
        assignment=assignment,
        definition=definition,
        target_niche=target_niche,
        out_dirs=out_dirs,
        st_h5ad=ROOT / args.st_expression_h5ad,
    )
    try:
        coloc_meta = _build_gene_celltype_colocalization_pack(
            assignment=assignment,
            target_niche=target_niche,
            st_h5ad=ROOT / args.st_expression_h5ad,
            st_dir=st_dir,
            out_dirs=out_dirs,
            max_genes=int(args.coloc_max_genes),
            max_samples=int(args.coloc_max_samples),
            max_spots_per_sample=int(args.coloc_max_spots_per_sample),
        )
    except Exception as e:
        coloc_meta = {"colocalization_pairs": 0, "error": str(e)}

    summary = {
        "n_samples": len(samples),
        "samples": samples,
        "selected_targets": selected_targets,
        "top30_count": int(
            target_niche[["target_gene", "global_rank"]]
            .drop_duplicates()
            .sort_values("global_rank")
            .head(30)
            .shape[0]
        ) if not target_niche.empty else 0,
        "step3_target_dir": str(ROOT / args.step3_target_dir),
        "st_expression_h5ad": str(ROOT / args.st_expression_h5ad),
        "cosmx_h5ad": str(ROOT / args.cosmx_h5ad),
        "visiumhd_h5": str(ROOT / args.visiumhd_h5),
        "visiumhd_pos": str(ROOT / args.visiumhd_pos),
        "coloc_max_genes": int(args.coloc_max_genes),
        "coloc_max_samples": int(args.coloc_max_samples),
        "coloc_max_spots_per_sample": int(args.coloc_max_spots_per_sample),
        "native_celltype_spatial": native_celltype_meta,
        "violin": violin_meta,
        "gene_celltype_colocalization": coloc_meta,
        "output_dir": str(out_dirs["root"]),
    }
    (out_dirs["root"] / "spatial_comm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DONE] spatial_comm figures saved to: {out_dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

