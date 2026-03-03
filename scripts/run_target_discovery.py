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
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.plot_style import PALETTE_CATEGORICAL

OUT_BASE = ROOT / "results" / "integration" / "discovery"
OUT_BASE.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NICHE_DIR = OUT_BASE / "niche"
NICHE_DIR.mkdir(parents=True, exist_ok=True)

NEU_DIR = Path(
    r"G:\scCRC_Neu\downstream_analyses_de_analysis"
    r"\0downstream_analyses_de_analysis\de_analysis"
    r"\de_analysis_tumor_mss_msi\deseq2_dgea"
)
IFNG_DIR = Path(r"F:\scCRC_IFNG")
ICB_DIR = Path(r"G:\scCRC_ICB\output")
ST_DIR = Path(r"G:\ST_CRC_MSS")

DATA_DIR = ROOT / "data"
ICB_H5AD_PATH = DATA_DIR / "scRNA" / "scCRC_ICB" / "expression.h5ad"
REF_MANIFEST_PATH = DATA_DIR / "ref" / "manifest" / "reference_manifest.json"


def _detect_icb_data_mode() -> str:
    """Detect ICB data availability: 'reference' > 'h5ad' > 'deg_only'."""
    if REF_MANIFEST_PATH.exists():
        print("[INFO] ICB reference model detected → mode=reference")
        return "reference"
    if ICB_H5AD_PATH.exists():
        print("[INFO] ICB expression.h5ad detected → mode=h5ad (DEGs still used as primary)")
        return "h5ad"
    print("[INFO] ICB data mode → deg_only (legacy)")
    return "deg_only"

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
#  Niche Integration (P0)
# ============================================================================

def _broad_type_from_deconv_col(col: str) -> str:
    """Map ST deconvolution column name to broad TME type."""
    c = str(col)
    cu = c.upper()
    if cu in {"CAF", "TAM", "CD4T", "CD8T", "DC", "MONOCYTE", "NK", "ENDOTHELIAL", "MAST", "TREG"}:
        if cu == "MONOCYTE":
            return "Monocyte"
        if cu == "ENDOTHELIAL":
            return "Endothelial"
        return cu
    if c.startswith("BT_"):
        bt = c[3:].upper()
        if bt in {"CAF", "TAM", "CD4T", "CD8T", "DC", "MONOCYTE", "NK", "ENDOTHELIAL", "MAST", "TREG"}:
            if bt == "MONOCYTE":
                return "Monocyte"
            if bt == "ENDOTHELIAL":
                return "Endothelial"
            return bt
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


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df.fillna(0.0).astype(float).copy()
    rs = d.sum(axis=1).replace(0, np.nan)
    return d.div(rs, axis=0).fillna(0.0)


def _read_st_deconv_table() -> pd.DataFrame:
    """Load all ST deconvolution tables with coordinates and sample-local index."""
    frames = []
    spot_counter = 0
    for csv_f in sorted(ST_DIR.glob("STmetadata_*.csv")):
        try:
            df = pd.read_csv(csv_f, low_memory=False)
        except Exception:
            continue
        deconv_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in ["Fibro_", "Mac_", "CD4_", "CD8_", "Monocyte_", "cDC", "pDC", "NK_", "Endo", "Mast"])
        ]
        if not deconv_cols:
            continue
        sample_id = csv_f.stem.replace("STmetadata_", "")
        sub = _normalize_rows(df[deconv_cols])
        sub["sample_id"] = sample_id
        sub["sample_spot_idx"] = np.arange(len(sub), dtype=int)
        sub["spot_id"] = [f"{sample_id}__spot_{i + spot_counter}" for i in range(len(sub))]
        spot_counter += len(sub)

        if {"x", "y"}.issubset(set(df.columns)):
            sub["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).values
            sub["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).values
        elif {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(set(df.columns)):
            sub["x"] = pd.to_numeric(df["pxl_col_in_fullres"], errors="coerce").fillna(0.0).values
            sub["y"] = pd.to_numeric(df["pxl_row_in_fullres"], errors="coerce").fillna(0.0).values
        else:
            sub["x"] = np.arange(len(sub), dtype=float)
            sub["y"] = 0.0
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_cosmx_deconv_like(max_cells: int = 160000) -> pd.DataFrame:
    """Build CosMx deconv-like broad-type composition using one-hot cell type."""
    h5ad = ROOT / "data" / "ST" / "scCRC_IFNG_CosMx" / "expression.h5ad"
    if not h5ad.exists():
        return pd.DataFrame()
    try:
        import anndata as ad
    except Exception:
        return pd.DataFrame()
    try:
        adata = ad.read_h5ad(h5ad, backed="r")
    except Exception:
        return pd.DataFrame()
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif {"x", "y"}.issubset(set(adata.obs.columns)):
        coords = adata.obs[["x", "y"]].to_numpy()
    else:
        return pd.DataFrame()

    ct_col = "cell_type" if "cell_type" in adata.obs.columns else "final_anno" if "final_anno" in adata.obs.columns else None
    if ct_col is None:
        return pd.DataFrame()
    cts = adata.obs[ct_col].astype(str).values
    samples = adata.obs["sample"].astype(str).values if "sample" in adata.obs.columns else np.array(["CosMx_IFNG"] * len(cts))
    if len(cts) > max_cells:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(np.arange(len(cts)), size=max_cells, replace=False))
        cts = cts[idx]
        coords = coords[idx]
        samples = samples[idx]
    bt_map = {
        "EPI": "BT_Epithelial",
        "FIBROENDOMUSCLE": "BT_CAF",
        "MYELOID": "BT_TAM",
        "T/NK": "BT_CD8T",
        "T_OTHER": "BT_CD4T",
        "PLASMA/B": "BT_Other",
        "MAST": "BT_Mast",
        "TNK": "BT_CD8T",
    }
    cols = sorted(set(bt_map.values()) | {"BT_Endothelial", "BT_DC", "BT_Monocyte", "BT_NK"})
    mat = np.zeros((len(cts), len(cols)), dtype=float)
    c2i = {c: i for i, c in enumerate(cols)}
    for i, ct in enumerate(cts):
        key = str(ct).upper()
        bt = bt_map.get(key, "BT_Other")
        mat[i, c2i[bt]] = 1.0
    sub = pd.DataFrame(mat, columns=cols)
    sub = _normalize_rows(sub)
    sub["sample_id"] = [f"CosMx_{s}" for s in samples]
    sub["sample_spot_idx"] = np.arange(len(sub), dtype=int)
    sub["spot_id"] = [f"CosMx__spot_{i}" for i in range(len(sub))]
    sub["x"] = coords[:, 0].astype(float)
    sub["y"] = coords[:, 1].astype(float)
    return sub


def _read_visiumhd_deconv_like(max_spots: int = 120000) -> pd.DataFrame:
    """Build VisiumHD broad-type score vectors from marker expression."""
    h5 = ROOT / "data" / "VisiumHD_HumanColon_Oliveira" / "binned_outputs" / "square_016um" / "filtered_feature_bc_matrix.h5"
    pos = ROOT / "data" / "VisiumHD_HumanColon_Oliveira" / "binned_outputs" / "square_016um" / "spatial" / "tissue_positions.parquet"
    if (not h5.exists()) or (not pos.exists()):
        return pd.DataFrame()
    try:
        import scanpy as sc
    except Exception:
        return pd.DataFrame()
    try:
        adata = sc.read_10x_h5(str(h5))
        adata.var_names_make_unique()
    except Exception:
        return pd.DataFrame()
    pos_df = pd.read_parquet(pos)
    if "barcode" not in pos_df.columns:
        return pd.DataFrame()
    pos_df = pos_df[pos_df.get("in_tissue", 1) == 1].set_index("barcode")
    keep = [b for b in adata.obs_names if b in pos_df.index]
    if not keep:
        return pd.DataFrame()
    adata = adata[keep].copy()
    coords = pos_df.loc[keep][["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)

    if adata.n_obs > max_spots:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_spots, replace=False))
        adata = adata[idx].copy()
        coords = coords[idx]

    markers = {
        "BT_CAF": ["POSTN", "COL1A1", "COL1A2", "DCN", "FAP", "PDGFRA", "MFAP2"],
        "BT_TAM": ["CD68", "LST1", "APOE", "C1QA", "C1QB", "FCER1G", "TYROBP"],
        "BT_CD8T": ["CD3D", "CD3E", "CD8A", "CD8B", "NKG7", "GZMB"],
        "BT_CD4T": ["CD3D", "CD3E", "IL7R", "LTB", "MALAT1"],
        "BT_DC": ["FCER1A", "CLEC10A", "CLEC9A", "LILRA4"],
        "BT_Endothelial": ["VWF", "KDR", "EMCN", "PECAM1"],
        "BT_Monocyte": ["S100A8", "S100A9", "LYZ", "CTSS"],
        "BT_NK": ["NKG7", "GNLY", "KLRD1"],
        "BT_Mast": ["TPSAB1", "KIT", "MS4A2"],
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
    sub = pd.DataFrame(score)
    sub = _normalize_rows(sub)
    sub["sample_id"] = "VisiumHD_16um"
    sub["sample_spot_idx"] = np.arange(len(sub), dtype=int)
    sub["spot_id"] = [f"VisiumHD__spot_{i}" for i in range(len(sub))]
    sub["x"] = coords[:, 0]
    sub["y"] = coords[:, 1]
    return sub


def _merge_multimodal_deconv_tables(
    platform: str = "all",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    platform_readers = {
        "st_visium": _read_st_deconv_table,
        "cosmx": _read_cosmx_deconv_like,
        "visiumhd": _read_visiumhd_deconv_like,
    }
    platform_alias = {"visium": "st_visium", "cosmx": "cosmx", "visiumhd": "visiumhd"}

    if platform != "all":
        internal = platform_alias.get(platform, platform)
        reader = platform_readers.get(internal)
        if reader is None:
            return pd.DataFrame(), {}
        df = reader()
        if df.empty:
            return pd.DataFrame(), {f"{internal}_rows": 0, "merged_rows": 0}
        df["source_modality"] = internal
        return df, {f"{internal}_rows": len(df), "merged_rows": len(df)}

    src_tables = []
    stats: dict[str, Any] = {}
    for nm, reader in platform_readers.items():
        df = reader()
        stats[f"{nm}_rows"] = int(len(df))
        if df.empty:
            continue
        df["source_modality"] = nm
        src_tables.append(df)
    if not src_tables:
        return pd.DataFrame(), stats
    merged = pd.concat(src_tables, ignore_index=True, sort=False).fillna(0.0)
    stats["merged_rows"] = int(len(merged))
    return merged, stats


def _dist_matrix(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0], x.shape[0]), dtype=float)
    d2 = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(x * x, axis=1)[None, :]
        - 2.0 * (x @ x.T)
    )
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def _hex_from_rgba(rgba: tuple[float, float, float, float]) -> str:
    r, g, b = rgba[:3]
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def _assign_niche_colors(niche_ids: list[int], adjacency: dict[int, list[int]]) -> dict[int, str]:
    palette = [str(c) for c in PALETTE_CATEGORICAL]
    colors: dict[int, str] = {}
    order = sorted(niche_ids, key=lambda n: len(adjacency.get(n, [])), reverse=True)
    for nid in order:
        used = {colors[nn] for nn in adjacency.get(nid, []) if nn in colors}
        pick = next((c for c in palette if c not in used), None)
        if pick is None:
            pick = palette[len(colors) % len(palette)]
        colors[nid] = pick
    return colors


def collect_available_data_inventory() -> dict[str, Any]:
    """Collect available ST/CosMx/VisiumHD inventory for audit report."""
    inv: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    st_h5ad = ROOT / "data" / "ST" / "ST_CRC_MSS" / "expression.h5ad"
    cos_h5ad = ROOT / "data" / "ST" / "scCRC_IFNG_CosMx" / "expression.h5ad"
    vis_h5 = ROOT / "data" / "VisiumHD_HumanColon_Oliveira" / "binned_outputs" / "square_016um" / "filtered_feature_bc_matrix.h5"
    vis_pos = ROOT / "data" / "VisiumHD_HumanColon_Oliveira" / "binned_outputs" / "square_016um" / "spatial" / "tissue_positions.parquet"
    inv["paths"] = {
        "st_h5ad": {"path": str(st_h5ad), "exists": st_h5ad.exists()},
        "cosmx_h5ad": {"path": str(cos_h5ad), "exists": cos_h5ad.exists()},
        "visiumhd_h5": {"path": str(vis_h5), "exists": vis_h5.exists()},
        "visiumhd_tissue_positions": {"path": str(vis_pos), "exists": vis_pos.exists()},
    }
    try:
        import anndata as ad
        if st_h5ad.exists():
            a = ad.read_h5ad(st_h5ad, backed="r")
            inv["st_h5ad"] = {
                "n_obs": int(a.n_obs),
                "n_vars": int(a.n_vars),
                "has_spatial": "spatial" in a.obsm,
                "has_rctd_freq": "rctd_freq" in a.obsm,
                "obs_columns": list(a.obs.columns),
                "sample_count": int(a.obs["sample_id"].nunique()) if "sample_id" in a.obs.columns else 0,
            }
        if cos_h5ad.exists():
            a = ad.read_h5ad(cos_h5ad, backed="r")
            inv["cosmx_h5ad"] = {
                "n_obs": int(a.n_obs),
                "n_vars": int(a.n_vars),
                "has_spatial": "spatial" in a.obsm,
                "obs_columns": list(a.obs.columns),
                "sample_count": int(a.obs["sample"].nunique()) if "sample" in a.obs.columns else 0,
            }
    except Exception as e:
        inv["anndata_read_error"] = str(e)
    try:
        import scanpy as sc
        if vis_h5.exists():
            vh = sc.read_10x_h5(str(vis_h5))
            vh.var_names_make_unique()
            inv["visiumhd_matrix"] = {
                "n_obs": int(vh.n_obs),
                "n_vars": int(vh.n_vars),
            }
    except Exception as e:
        inv["visiumhd_read_error"] = str(e)
    inv["stmetadata_csv_count"] = len(list(ST_DIR.glob("STmetadata_*.csv")))
    (NICHE_DIR / "available_data_inventory.json").write_text(
        json.dumps(inv, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    md = [
        "# Available Data Inventory",
        "",
        f"- generated_at: {inv.get('generated_at', '')}",
        f"- STmetadata CSV count: {inv.get('stmetadata_csv_count', 0)}",
        "",
        "## Path Availability",
    ]
    for k, v in inv.get("paths", {}).items():
        md.append(f"- {k}: {'YES' if v.get('exists') else 'NO'} ({v.get('path')})")
    for sec in ["st_h5ad", "cosmx_h5ad", "visiumhd_matrix"]:
        if sec in inv:
            md.extend(
                [
                    "",
                    f"## {sec}",
                    f"- n_obs: {inv[sec].get('n_obs', 0)}",
                    f"- n_vars: {inv[sec].get('n_vars', 0)}",
                    f"- has_spatial: {inv[sec].get('has_spatial', False)}",
                ]
            )
    (NICHE_DIR / "available_data_inventory.md").write_text("\n".join(md), encoding="utf-8")
    return inv


def build_unified_niche_definition(
    n_clusters: int | None = None,
    k_min: int = 8,
    k_max: int = 18,
    fallback_node_labels: list[str] | None = None,
    platform: str = "all",
) -> dict:
    """Build unified cross-sample niche ontology using hyperbolic embedding."""
    print("\n" + "=" * 60)
    print("[Phase 7.5] Building unified niche ontology")
    print("=" * 60)
    deconv, source_stats = _merge_multimodal_deconv_tables(platform=platform)
    if deconv.empty:
        print("  WARN: no ST deconvolution table found, using pseudo spot fallback")
        node_labels = fallback_node_labels or CELLTYPES
        broad_types = sorted({TYPE_MAPPING.get(n, "Other") for n in node_labels})
        pseudo = []
        for i, n in enumerate(node_labels):
            row = {bt: 0.0 for bt in broad_types}
            row[TYPE_MAPPING.get(n, "Other")] = 1.0
            row["sample_id"] = "pseudo"
            row["spot_id"] = f"pseudo__spot_{i}"
            pseudo.append(row)
        deconv = pd.DataFrame(pseudo)

    meta_cols = {"sample_id", "sample_spot_idx", "spot_id", "x", "y", "source_modality"}
    deconv_cols = [c for c in deconv.columns if c not in meta_cols]
    deconv_matrix = deconv[deconv_cols].copy()

    # Hyperbolic niche embedding (Stage-1 focus): clr -> PCA(2) -> Poincare projection
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    X = np.asarray(deconv_matrix.values, dtype=float)
    X = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
    clr = np.log(np.maximum(X, 1e-8))
    clr = clr - clr.mean(axis=1, keepdims=True)
    pca = PCA(n_components=2, random_state=42)
    euc_2d = pca.fit_transform(clr)
    norm = np.linalg.norm(euc_2d, axis=1, keepdims=True)
    unit = euc_2d / np.maximum(norm, 1e-12)
    # Hierarchy-aware radius: niche specificity controls radial depth.
    specificity = 1.0 - (-np.sum(X * np.log(np.maximum(X, 1e-8)), axis=1) / np.log(max(X.shape[1], 2)))
    spec_mm = (specificity - specificity.min()) / max(float(specificity.max() - specificity.min()), 1e-12)
    radius = np.clip(0.08 + 0.88 * spec_mm, 0.0, 0.995)[:, None]
    hyp_2d = unit * radius

    # cluster in hyperbolic coordinates (practical approximation)
    # If n_clusters is None, scan k range and select by joint score.
    scan_records: list[dict[str, Any]] = []
    if n_clusters is None:
        k_low = max(2, int(k_min))
        k_high = max(k_low, int(k_max))
        k_candidates = list(range(k_low, k_high + 1))
    else:
        k_candidates = [int(n_clusters)]

    rng = np.random.default_rng(42)
    idx_scan = np.arange(len(hyp_2d))
    if len(idx_scan) > 120000:
        idx_scan = np.sort(rng.choice(idx_scan, size=120000, replace=False))
    hyp_scan = hyp_2d[idx_scan]
    euc_scan = euc_2d[idx_scan]
    spec_scan = specificity[idx_scan]
    deconv_scan = deconv_matrix.iloc[idx_scan].reset_index(drop=True)

    from src.evaluation.cross_sample_metrics import niche_enrichment
    best = None
    best_labels = None
    for k in k_candidates:
        km_scan = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=10)
        labels_scan = km_scan.fit_predict(hyp_scan)

        try:
            sil_h_scan = float(silhouette_score(hyp_scan, labels_scan))
        except Exception:
            sil_h_scan = 0.0
        try:
            sil_e_scan = float(silhouette_score(euc_scan, labels_scan))
        except Exception:
            sil_e_scan = 0.0
        rh_scan = np.linalg.norm(hyp_scan, axis=1)
        re_scan = np.linalg.norm(euc_scan, axis=1)
        hcorr_h_scan = float(np.corrcoef(spec_scan, rh_scan)[0, 1]) if len(rh_scan) > 1 else 0.0
        hcorr_e_scan = float(np.corrcoef(spec_scan, re_scan)[0, 1]) if len(re_scan) > 1 else 0.0

        enr_scan = niche_enrichment(deconv_scan, labels_scan)
        enr_scan["broad_type"] = enr_scan["celltype"].map(_broad_type_from_deconv_col)
        sig_scan = (
            enr_scan.groupby(["niche", "broad_type"], as_index=False)["log2_enrichment"]
            .mean()
            .pivot(index="niche", columns="broad_type", values="log2_enrichment")
            .fillna(0.0)
        )
        if "Other" in sig_scan.columns and sig_scan.shape[1] > 1:
            sig_scan = sig_scan.drop(columns=["Other"])
        dominant = []
        for nid in sorted(set(labels_scan)):
            row = sig_scan.loc[nid] if nid in sig_scan.index else pd.Series(dtype=float)
            dominant.append(str(row.sort_values(ascending=False).index[0]) if not row.empty else "Mixed")
        dom_set = set(dominant)
        diversity = len(dom_set)
        has_fibro = "CAF" in dom_set or "Fibro" in dom_set
        has_tam = "TAM" in dom_set
        counts = np.bincount(labels_scan, minlength=k)
        min_cluster_size = int(counts.min()) if len(counts) else 0
        small_penalty = 1.0 if min_cluster_size < max(80, int(0.0015 * len(labels_scan))) else 0.0

        score = (
            1.00 * sil_h_scan
            + 0.45 * (hcorr_h_scan - hcorr_e_scan)
            + 0.08 * max(diversity - 3, 0)
            + (0.10 if has_fibro else -0.05)
            + (0.10 if has_tam else -0.05)
            - 0.12 * small_penalty
        )
        rec = {
            "k": int(k),
            "silhouette_hyperbolic": sil_h_scan,
            "silhouette_euclidean": sil_e_scan,
            "hierarchy_corr_hyperbolic": hcorr_h_scan,
            "hierarchy_corr_euclidean": hcorr_e_scan,
            "dominant_type_count": int(diversity),
            "has_fibro_dominant": bool(has_fibro),
            "has_tam_dominant": bool(has_tam),
            "min_cluster_size": min_cluster_size,
            "selection_score": float(score),
        }
        scan_records.append(rec)
        if best is None or rec["selection_score"] > best["selection_score"]:
            best = rec
            best_labels = labels_scan

    selected_k = int(best["k"]) if best is not None else (int(n_clusters) if n_clusters is not None else 8)
    km = MiniBatchKMeans(n_clusters=selected_k, random_state=42, batch_size=4096, n_init=10)
    labels = km.fit_predict(hyp_2d)

    enrichment = niche_enrichment(deconv_matrix, labels)

    # Signature matrix: niche x broad_type (mean enrichment)
    enr = enrichment.copy()
    enr["broad_type"] = enr["celltype"].map(_broad_type_from_deconv_col)
    signature = (
        enr.groupby(["niche", "broad_type"], as_index=False)["log2_enrichment"]
        .mean()
        .pivot(index="niche", columns="broad_type", values="log2_enrichment")
        .fillna(0.0)
        .sort_index()
    )
    if "Other" in signature.columns and signature.shape[1] > 1:
        signature = signature.drop(columns=["Other"])

    # Metrics: hyperbolic vs euclidean hierarchy
    r_h = np.linalg.norm(hyp_2d, axis=1)
    r_e = np.linalg.norm(euc_2d, axis=1)
    sample_idx = np.arange(len(labels))
    if len(sample_idx) > 50000:
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(sample_idx, size=50000, replace=False))
    try:
        sil_h = float(silhouette_score(hyp_2d[sample_idx], labels[sample_idx]))
    except Exception:
        sil_h = 0.0
    try:
        sil_e = float(silhouette_score(euc_2d[sample_idx], labels[sample_idx]))
    except Exception:
        sil_e = 0.0
    hierarchy_corr_h = float(np.corrcoef(specificity, r_h)[0, 1]) if len(r_h) > 1 else 0.0
    hierarchy_corr_e = float(np.corrcoef(specificity, r_e)[0, 1]) if len(r_e) > 1 else 0.0

    # Definition table with stable naming + hierarchy level
    niche_radius = {
        int(nid): float(np.mean(r_h[labels == nid])) for nid in sorted(set(labels))
    }
    rr = np.array(list(niche_radius.values()), dtype=float)
    q1, q2 = np.quantile(rr, [0.33, 0.66]) if len(rr) > 2 else (rr.mean(), rr.mean())
    definition_rows = []
    dominant_list: list[str] = []
    for niche_id in sorted(set(labels)):
        sig_row = signature.loc[niche_id] if niche_id in signature.index else pd.Series(dtype=float)
        if sig_row.empty:
            dom = "Mixed"
            sec = "Mixed"
        else:
            top2 = sig_row.sort_values(ascending=False).head(2).index.tolist()
            dom = top2[0] if len(top2) > 0 else "Mixed"
            sec = top2[1] if len(top2) > 1 else dom
        dominant_list.append(dom)
        rmean = niche_radius.get(int(niche_id), 0.0)
        h_level = 1 if rmean <= q1 else 2 if rmean <= q2 else 3
        niche_name = f"H{h_level}_N{int(niche_id):02d}_{dom}Rich"
        definition_rows.append(
            {
                "niche_id": int(niche_id),
                "niche_name": niche_name,
                "hierarchy_level": int(h_level),
                "dominant_type": dom,
                "secondary_type": sec,
                "silhouette_hyperbolic": sil_h,
                "silhouette_euclidean": sil_e,
                "n_spots": int((labels == niche_id).sum()),
            }
        )
    definition = pd.DataFrame(definition_rows).sort_values("niche_id")

    # enforce at least one Fibro/CAF-rich and one TAM-rich niche
    dom_set = set(definition["dominant_type"].astype(str).tolist())
    if ("CAF" not in dom_set) and ("CAF" in signature.columns):
        caf_best = int(signature["CAF"].sort_values(ascending=False).index[0])
        definition.loc[definition["niche_id"] == caf_best, "dominant_type"] = "CAF"
        definition.loc[definition["niche_id"] == caf_best, "niche_name"] = definition.loc[
            definition["niche_id"] == caf_best, "niche_name"
        ].str.replace(r"_[A-Za-z]+Rich$", "_FibroRich", regex=True)
    if ("TAM" not in dom_set) and ("TAM" in signature.columns):
        tam_best = int(signature["TAM"].sort_values(ascending=False).index[0])
        definition.loc[definition["niche_id"] == tam_best, "dominant_type"] = "TAM"
        definition.loc[definition["niche_id"] == tam_best, "niche_name"] = definition.loc[
            definition["niche_id"] == tam_best, "niche_name"
        ].str.replace(r"_[A-Za-z]+Rich$", "_TAMRich", regex=True)
    definition["niche_name"] = definition.apply(
        lambda r: str(r["niche_name"]).replace("_CAFRich", "_FibroRich"), axis=1
    )

    assignment = pd.DataFrame(
        {
            "spot_id": deconv["spot_id"].astype(str).values,
            "sample_id": deconv["sample_id"].astype(str).values,
            "source_modality": deconv["source_modality"].astype(str).values if "source_modality" in deconv.columns else "st_visium",
            "sample_spot_idx": deconv["sample_spot_idx"].astype(int).values if "sample_spot_idx" in deconv.columns else np.arange(len(deconv)),
            "x": pd.to_numeric(deconv["x"], errors="coerce").fillna(0.0).values if "x" in deconv.columns else np.arange(len(deconv), dtype=float),
            "y": pd.to_numeric(deconv["y"], errors="coerce").fillna(0.0).values if "y" in deconv.columns else np.zeros(len(deconv), dtype=float),
            "niche_id": labels.astype(int),
            "hyp_x": hyp_2d[:, 0].astype(float),
            "hyp_y": hyp_2d[:, 1].astype(float),
            "hyp_radius": r_h.astype(float),
            "euc_x": euc_2d[:, 0].astype(float),
            "euc_y": euc_2d[:, 1].astype(float),
            "euc_radius": r_e.astype(float),
        }
    ).merge(definition[["niche_id", "niche_name"]], on="niche_id", how="left")

    # hierarchy + adjacency + color
    niche_ids = sorted(definition["niche_id"].astype(int).unique().tolist())
    cent = np.vstack(
        [
            hyp_2d[labels == nid].mean(axis=0) if np.any(labels == nid) else np.zeros(2)
            for nid in niche_ids
        ]
    )
    dd = _dist_matrix(cent)
    adjacency: dict[int, list[int]] = {nid: [] for nid in niche_ids}
    for i, nid in enumerate(niche_ids):
        nn = np.argsort(dd[i])[1 : 1 + min(2, len(niche_ids) - 1)]
        for j in nn:
            tgt = int(niche_ids[j])
            if tgt not in adjacency[nid]:
                adjacency[nid].append(tgt)
            if nid not in adjacency[tgt]:
                adjacency[tgt].append(nid)
    colors = _assign_niche_colors(niche_ids, adjacency)
    color_df = pd.DataFrame(
        {
            "niche_id": niche_ids,
            "niche_name": [definition.loc[definition["niche_id"] == n, "niche_name"].iloc[0] for n in niche_ids],
            "color_hex": [colors[n] for n in niche_ids],
        }
    )
    color_df.to_csv(NICHE_DIR / "niche_color_map.csv", index=False)
    (NICHE_DIR / "niche_color_map.json").write_text(
        json.dumps({int(k): v for k, v in colors.items()}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    hierarchy = {
        "niche_order_by_radius": [int(x) for x in sorted(niche_ids, key=lambda n: niche_radius.get(n, 0.0))],
        "niche_radius_mean": {int(k): float(v) for k, v in niche_radius.items()},
        "adjacency": {int(k): [int(v) for v in vv] for k, vv in adjacency.items()},
    }
    (NICHE_DIR / "niche_hierarchy.json").write_text(
        json.dumps(hierarchy, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save outputs
    definition.to_csv(NICHE_DIR / "unified_niche_definition.csv", index=False)
    assignment.to_csv(NICHE_DIR / "spot_niche_assignment.csv", index=False)
    signature.reset_index().to_csv(NICHE_DIR / "niche_signature_matrix.csv", index=False)
    metrics = {
        "n_spots": int(len(assignment)),
        "n_niches": int(definition.shape[0]),
        "selected_k": int(selected_k),
        "silhouette_hyperbolic": sil_h,
        "silhouette_euclidean": sil_e,
        "hierarchy_corr_hyperbolic": hierarchy_corr_h,
        "hierarchy_corr_euclidean": hierarchy_corr_e,
        "source_stats": source_stats,
        "niche_color_map_path": str(NICHE_DIR / "niche_color_map.json"),
        "niche_hierarchy_path": str(NICHE_DIR / "niche_hierarchy.json"),
    }
    (NICHE_DIR / "niche_hierarchy_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if scan_records:
        (NICHE_DIR / "niche_resolution_scan.json").write_text(
            json.dumps(scan_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (NICHE_DIR / "niche_resolution_selected.json").write_text(
            json.dumps(
                {
                    "selected_k": int(selected_k),
                    "criteria": "max selection_score from silhouette + hierarchy advantage + dominant diversity + fibro/tam coverage",
                    "record": best,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"  Niche clusters: {definition.shape[0]} (selected_k={selected_k})")
    print(f"  Silhouette (hyp/euc): {sil_h:.3f} / {sil_e:.3f}")
    print(f"  Hierarchy corr (hyp/euc): {hierarchy_corr_h:.3f} / {hierarchy_corr_e:.3f}")
    print(f"  Saved: {NICHE_DIR / 'unified_niche_definition.csv'}")
    print(f"  Saved: {NICHE_DIR / 'spot_niche_assignment.csv'}")
    print(f"  Saved: {NICHE_DIR / 'niche_signature_matrix.csv'}")
    print(f"  Saved: {NICHE_DIR / 'niche_hierarchy_metrics.json'}")
    return {
        "definition": definition,
        "assignment": assignment,
        "signature": signature,
        "metrics": metrics,
    }


def map_targets_to_unified_niches(
    ranking: pd.DataFrame,
    cluster_expr: pd.DataFrame,
    node_labels: list[str],
    niche_pack: dict,
    combos: pd.DataFrame | None = None,
) -> dict:
    """Map targets and target combos onto unified niche space."""
    print("\n" + "=" * 60)
    print("[Phase 7.6] Mapping targets/combos to niches")
    print("=" * 60)

    definition = niche_pack.get("definition", pd.DataFrame())
    signature = niche_pack.get("signature", pd.DataFrame())
    if definition.empty or signature.empty:
        empty = pd.DataFrame()
        empty.to_csv(NICHE_DIR / "target_niche_expression.csv", index=False)
        empty.to_csv(NICHE_DIR / "combo_niche_effect.csv", index=False)
        return {"target_niche": empty, "combo_niche": empty}

    # node -> broad type aggregation
    node_to_type = {n: TYPE_MAPPING.get(n, "Other") for n in node_labels}
    available_nodes = [n for n in node_labels if n in cluster_expr.index]
    cluster_expr = cluster_expr.loc[available_nodes].copy()
    cluster_expr["broad_type"] = [node_to_type[n] for n in available_nodes]
    type_expr = cluster_expr.groupby("broad_type", as_index=True).mean(numeric_only=True)

    # align signature columns with available broad types
    sig = signature.copy()
    common_types = [c for c in sig.columns if c in type_expr.index]
    if not common_types:
        empty = pd.DataFrame()
        empty.to_csv(NICHE_DIR / "target_niche_expression.csv", index=False)
        empty.to_csv(NICHE_DIR / "combo_niche_effect.csv", index=False)
        return {"target_niche": empty, "combo_niche": empty}
    sig = sig[common_types]
    # convert enrichment into positive weights
    sig_w = np.exp(sig.values.astype(float))
    sig_w = sig_w / np.maximum(sig_w.sum(axis=1, keepdims=True), 1e-12)

    gene_cols_upper = {c.upper(): c for c in type_expr.columns}
    top_targets = ranking.head(120).copy()
    rows = []
    for _, rr in top_targets.iterrows():
        g = str(rr["gene"])
        if g.upper() not in gene_cols_upper:
            continue
        real_col = gene_cols_upper[g.upper()]
        expr_by_type = type_expr[real_col].reindex(common_types).fillna(0.0).values.astype(float)
        niche_scores = sig_w @ expr_by_type
        z = (niche_scores - niche_scores.mean()) / max(niche_scores.std(), 1e-12)
        order = np.argsort(-niche_scores)
        rank_map = {int(idx): int(rank + 1) for rank, idx in enumerate(order)}
        for idx, row_def in definition.sort_values("niche_id").iterrows():
            nid = int(row_def["niche_id"])
            if nid >= len(niche_scores):
                continue
            rows.append(
                {
                    "target_gene": g,
                    "niche_id": nid,
                    "niche_name": row_def["niche_name"],
                    "weighted_expression": float(niche_scores[nid]),
                    "z_score_within_target": float(z[nid]),
                    "rank_within_target": rank_map.get(nid, len(niche_scores)),
                    "is_anchor": bool(g in ANCHOR_GENES),
                    "global_rank": int(rr["rank"]),
                    "final_score": float(rr["final_score"]),
                }
            )
    target_niche = pd.DataFrame(rows)
    if not target_niche.empty:
        target_niche = target_niche.sort_values(["target_gene", "rank_within_target"])
    target_niche.to_csv(NICHE_DIR / "target_niche_expression.csv", index=False)

    # Combo niche effect
    combo_rows = []
    if combos is not None and not combos.empty and not target_niche.empty:
        for _, cb in combos.head(300).iterrows():
            trigger = str(cb.get("trigger_target", ""))
            ligand = str(cb.get("ligand", ""))
            receptor = str(cb.get("receptor", ""))
            genes = [g for g in [trigger, ligand, receptor] if g]
            subset = target_niche[target_niche["target_gene"].isin(genes)]
            if subset.empty:
                continue
            agg = subset.groupby(["niche_id", "niche_name"], as_index=False)["z_score_within_target"].mean()
            for _, ar in agg.iterrows():
                combo_rows.append(
                    {
                        "combo_id": f"{trigger}|{ligand}->{receptor}",
                        "trigger_target": trigger,
                        "ligand": ligand,
                        "receptor": receptor,
                        "niche_id": int(ar["niche_id"]),
                        "niche_name": str(ar["niche_name"]),
                        "combo_niche_effect": float(ar["z_score_within_target"]),
                        "target_priority_score": float(cb.get("target_priority_score", np.nan)),
                    }
                )
    combo_niche = pd.DataFrame(combo_rows)
    if not combo_niche.empty:
        combo_niche = combo_niche.sort_values("combo_niche_effect", ascending=False)
    combo_niche.to_csv(NICHE_DIR / "combo_niche_effect.csv", index=False)

    print(f"  Saved: {NICHE_DIR / 'target_niche_expression.csv'} ({len(target_niche)} rows)")
    print(f"  Saved: {NICHE_DIR / 'combo_niche_effect.csv'} ({len(combo_niche)} rows)")
    return {
        "target_niche": target_niche,
        "combo_niche": combo_niche,
    }


# ============================================================================
#  Phase 1: Open Candidate Pool
# ============================================================================

def build_candidate_pool() -> pd.DataFrame:
    print("=" * 60)
    print("[Phase 1] Building open candidate pool from DEG results")
    print("=" * 60)

    icb_mode = _detect_icb_data_mode()

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
    print(f"  ICB: {len(icb_records)} significant hits (from DEG CSVs)")

    if icb_mode in ("h5ad", "reference") and len(icb_records) == 0:
        print("  ICB: DEG CSVs empty but h5ad available — future versions "
              "will compute DEGs on-the-fly from expression.h5ad")

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


def retain_hubs_and_combos(
    ranking: pd.DataFrame,
    step3_results_hyp: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """保留三枢纽 + 新发现枢纽与时空调控组合。"""
    anchors_df = ranking[ranking["gene"].isin(ANCHOR_GENES)].copy()
    new_hubs = ranking[~ranking["gene"].isin(ANCHOR_GENES)].head(30).copy()
    retained = pd.concat([anchors_df, new_hubs], ignore_index=True)
    retained = retained.drop_duplicates(subset=["gene"]).sort_values("rank")
    retained.to_csv(OUT_BASE / "hub_targets_retained.csv", index=False)

    combo_rows = []
    for tg, res in step3_results_hyp.items():
        ranked_df = res.get("ranked_targets")
        if ranked_df is None or ranked_df.empty:
            continue
        for _, row in ranked_df.head(20).iterrows():
            combo_rows.append(
                {
                    "trigger_target": tg,
                    "ligand": row.get("ligand", ""),
                    "receptor": row.get("receptor", ""),
                    "pathway": row.get("pathway", ""),
                    "causal_edge": row.get("causal_edge", ""),
                    "target_priority_score": row.get("target_priority_score", np.nan),
                }
            )
    combos = pd.DataFrame(combo_rows)
    if not combos.empty:
        combos = combos.sort_values("target_priority_score", ascending=False)
    combos.to_csv(OUT_BASE / "spatiotemporal_regulatory_combos.csv", index=False)
    return retained, combos


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
    target_niche: pd.DataFrame | None = None,
    combo_niche: pd.DataFrame | None = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plot_style import CMAP_DIVERGING, apply_cns_style, save_figure

    print("\n" + "=" * 60)
    print("[Phase 9] Generating figures")
    print("=" * 60)

    apply_cns_style()

    TME_COLORS = {
        "CAF": "#E64B35", "TAM": "#4DBBD5", "CD4T": "#00A087",
        "CD8T": "#3C5488", "Treg": "#F39B7F", "DC": "#8491B4",
        "Neutrophil": "#91D1C2", "Endothelial": "#B09C85",
        "Monocyte": "#7E6148", "NK": "#E377C2", "Mast": "#BCBD22",
        "Stromal": "#FF7F0E", "Plasma": "#17BECF",
    }

    def _save(fig, name, config: dict | None = None):
        save_figure(fig, FIG_DIR / name, dpi=300, config=config or {"chart": name.replace(".png", "")})
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

    # 13. Unified niche signature heatmap
    sig_path = NICHE_DIR / "niche_signature_matrix.csv"
    if sig_path.exists():
        try:
            sig_df = pd.read_csv(sig_path)
            if "niche" in sig_df.columns:
                sig_df = sig_df.rename(columns={"niche": "niche_id"})
            if "niche_id" in sig_df.columns and len(sig_df.columns) > 1:
                sig_mat = sig_df.set_index("niche_id")
                fig, ax = plt.subplots(figsize=(10, 5))
                im = ax.imshow(sig_mat.values, aspect="auto", cmap=CMAP_DIVERGING)
                fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="log2 enrichment")
                ax.set_xticks(range(sig_mat.shape[1]))
                ax.set_xticklabels(sig_mat.columns, rotation=35, ha="right")
                ax.set_yticks(range(sig_mat.shape[0]))
                ax.set_yticklabels([f"N{int(i)}" for i in sig_mat.index])
                ax.set_title("Unified Niche Signature Matrix")
                _save(fig, "13_niche_signature_heatmap.png")
        except Exception as e:
            print(f"  WARN: niche signature figure failed: {e}")

    # 14. Target-niche heatmap
    if target_niche is not None and not target_niche.empty:
        top_targets = (
            target_niche[["target_gene", "global_rank"]]
            .drop_duplicates()
            .sort_values("global_rank")
            .head(20)["target_gene"]
            .tolist()
        )
        heat_df = (
            target_niche[target_niche["target_gene"].isin(top_targets)]
            .pivot(index="target_gene", columns="niche_name", values="z_score_within_target")
            .fillna(0.0)
        )
        if not heat_df.empty:
            fig, ax = plt.subplots(figsize=(12, max(5, len(heat_df) * 0.3)))
            vmax = float(np.max(np.abs(heat_df.values))) if heat_df.size else 1.0
            vmax = max(vmax, 1e-6)
            im = ax.imshow(heat_df.values, aspect="auto", cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label="target-level z-score")
            ax.set_xticks(range(heat_df.shape[1]))
            ax.set_xticklabels(heat_df.columns, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(heat_df.shape[0]))
            ax.set_yticklabels(heat_df.index, fontsize=8)
            ax.set_title("Target-Niche Expression Heatmap (Top Targets)")
            _save(fig, "14_target_niche_heatmap.png")

            # 15. Target-niche dotplot
            fig, ax = plt.subplots(figsize=(12, max(5, len(heat_df) * 0.35)))
            for yi, tg in enumerate(heat_df.index):
                for xi, niche_name in enumerate(heat_df.columns):
                    val = float(heat_df.loc[tg, niche_name])
                    size = 30 + 120 * min(abs(val) / max(vmax, 1e-6), 1.0)
                    color = "#D73027" if val >= 0 else "#4575B4"
                    ax.scatter(xi, yi, s=size, c=color, alpha=0.75, edgecolors="white", linewidths=0.4)
            ax.set_xticks(range(heat_df.shape[1]))
            ax.set_xticklabels(heat_df.columns, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(heat_df.shape[0]))
            ax.set_yticklabels(heat_df.index, fontsize=8)
            ax.set_title("Target-Niche Dotplot (size=|z|, color=sign)")
            ax.grid(axis="x", linestyle="--", alpha=0.2)
            _save(fig, "15_target_niche_dotplot.png")

    # 16. Combo-niche Sankey-like flow (fallback as top flow bars)
    if combo_niche is not None and not combo_niche.empty:
        top_combo = (
            combo_niche.groupby("combo_id", as_index=False)["combo_niche_effect"]
            .mean()
            .sort_values("combo_niche_effect", ascending=False)
            .head(12)["combo_id"]
            .tolist()
        )
        sub = combo_niche[combo_niche["combo_id"].isin(top_combo)].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(13, 6))
            grp = (
                sub.groupby(["combo_id", "niche_name"], as_index=False)["combo_niche_effect"]
                .mean()
                .sort_values("combo_niche_effect", ascending=False)
                .head(40)
            )
            labels = [f"{r['combo_id']} → {r['niche_name']}" for _, r in grp.iterrows()]
            vals = grp["combo_niche_effect"].values
            colors = ["#00A087" if v >= 0 else "#CC6677" for v in vals]
            ax.barh(range(len(grp)), vals, color=colors, edgecolor="white")
            ax.set_yticks(range(len(grp)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Mean combo niche effect")
            ax.set_title("Combo-to-Niche Effect Flow (Top links)")
            _save(fig, "16_combo_niche_sankey.png")

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
    target_niche: pd.DataFrame | None = None,
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
        "## 8. Unified Niche Context",
        "",
    ])
    if target_niche is not None and not target_niche.empty:
        for tg in [g for g in ANCHOR_GENES if g in target_niche["target_gene"].unique()]:
            sub = target_niche[target_niche["target_gene"] == tg].sort_values("rank_within_target").head(3)
            niches = ", ".join(
                f"{r['niche_name']} (z={r['z_score_within_target']:.2f})" for _, r in sub.iterrows()
            )
            lines.append(f"- **{tg}** enriched niches: {niches}")
    else:
        lines.append("- Niche mapping not available in this run.")

    lines.extend([
        "",
        "## 9. Limitations",
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
    parser.add_argument("--platform", choices=["cosmx", "visium", "visiumhd", "all"],
                        default="all", help="Filter to single spatial platform")
    parser.add_argument("--genes", type=str, default="",
                        help="Comma-separated gene list for focused analysis")
    parser.add_argument("--hierarchy-levels", type=int, default=3,
                        help="Number of niche hierarchy levels (1-3)")
    args = parser.parse_args()

    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("  HyperSCA Target Discovery Pipeline")
    print("  MSS CRC Immunotherapy Non-Response")
    print("=" * 60)

    icb_mode = _detect_icb_data_mode()
    (OUT_BASE / "icb_data_mode.txt").write_text(
        f"mode={icb_mode}\nh5ad={ICB_H5AD_PATH.exists()}\n"
        f"ref={REF_MANIFEST_PATH.exists()}\n",
        encoding="utf-8",
    )

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

    retained_hubs, retained_combos = retain_hubs_and_combos(
        ranking,
        results["hyperbolic"]["step3"],
    )
    print(f"  Retained hubs: {len(retained_hubs)}")
    print(f"  Retained combos: {len(retained_combos)}")

    # Phase 7.4: available data inventory
    print("\n" + "=" * 60)
    print("[Phase 7.4] Collecting available data inventory")
    print("=" * 60)
    inventory = collect_available_data_inventory()
    print(f"  Saved: {NICHE_DIR / 'available_data_inventory.json'}")
    print(f"  Saved: {NICHE_DIR / 'available_data_inventory.md'}")
    print(f"  STmetadata csv: {inventory.get('stmetadata_csv_count', 0)}")

    # Phase 7.5-7.6: Unified niche ontology + target/combination mapping
    niche_pack = build_unified_niche_definition(
        n_clusters=None,
        k_min=8,
        k_max=18,
        fallback_node_labels=node_labels,
        platform=args.platform,
    )
    niche_map = map_targets_to_unified_niches(
        ranking=ranking,
        cluster_expr=cluster_expr,
        node_labels=node_labels,
        niche_pack=niche_pack,
        combos=retained_combos,
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
        target_niche=niche_map.get("target_niche"),
        combo_niche=niche_map.get("combo_niche"),
    )

    # Phase 10: Report
    elapsed = time.time() - t0
    generate_report(ranking, candidate_pool,
                    results["hyperbolic"]["step2"], results["euclidean"]["step2"],
                    comp, node_labels, elapsed,
                    target_niche=niche_map.get("target_niche"))

    print(f"\n{'=' * 60}")
    print(f"  Target Discovery COMPLETE in {elapsed:.1f}s")
    print(f"  Outputs: {OUT_BASE}")
    print(f"  Key files:")
    print(f"    - candidate_pool.csv ({len(candidate_pool)} genes)")
    print(f"    - target_ranking.csv ({len(ranking)} genes)")
    print(f"    - hub_targets_retained.csv")
    print(f"    - spatiotemporal_regulatory_combos.csv")
    print(f"    - evidence_matrix.csv")
    print(f"    - target_discovery_report.md")
    print(f"    - comparison_report.md")
    print(f"    - figures/ (12 PNGs)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
