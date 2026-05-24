#!/usr/bin/env python
"""Per-platform spatial niche analysis with hyperbolic embedding.

Four-scale co-localization niche hierarchy:
  micro → small → medium → macro
Each scale defined by cell-type co-localization ratios (p_i × p_j)
in progressively wider neighborhoods.

Usage:
    python scripts/run_platform_niche_analysis.py
    python scripts/run_platform_niche_analysis.py --platform visium
    python scripts/run_platform_niche_analysis.py --platform cosmx --genes FN1,COL1A1,CCL18
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.plot_style import (
    apply_cns_style, save_figure, get_color_mapping,
    get_dynamic_point_size, get_dynamic_alpha,
    PALETTE_CATEGORICAL, CMAP_EXPRESSION, CMAP_SPATIAL,
    place_legend_outside, add_watermark,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GENE_ALIAS = {"COLA1": "COL1A1"}

DEFAULT_GENES = ["FN1", "COL1A1", "CCL18", "CXCL8", "SPP1"]

PLATFORM_MAP = {
    "visium": "st_visium",
    "cosmx": "cosmx",
    "visiumhd": "visiumhd",
}

ST_DIR = Path(r"G:\ST_CRC_MSS")
COSMX_H5AD = ROOT / "data" / "ST" / "scCRC_IFNG_CosMx" / "expression.h5ad"
VISIUMHD_H5 = ROOT / "data" / "VisiumHD_HumanColon_Oliveira" / "binned_outputs" / "square_016um" / "filtered_feature_bc_matrix.h5"
VISIUMHD_POS = ROOT / "data" / "VisiumHD_HumanColon_Oliveira" / "binned_outputs" / "square_016um" / "spatial" / "tissue_positions.parquet"
REF_MANIFEST = ROOT / "data" / "ref" / "manifest" / "reference_manifest.json"

AUDIT_DIR = ROOT / "results" / "figures" / "spatial_comm" / "audit"
FIG_BASE = ROOT / "results" / "figures" / "spatial_comm"
NICHE_BASE = ROOT / "results" / "integration" / "discovery" / "niche"

SCALE_ORDER = ["micro", "small", "medium", "macro"]

VISIUM_DECONV_PREFIXES = [
    "Fibro_", "Mac_", "CD4_", "CD8_", "Monocyte_",
    "cDC", "pDC", "NK_", "Endo", "Mast", "B_", "Plasma_",
]
VISIUM_TUMOR_PREFIXES = ["Epi", "Tumor", "Malig", "Malignant", "Cancer", "Epithelial"]
VISIUM_RAW_ANNOTATION_COLS = [
    "orig.ident", "seurat_clusters", "id",
    "Type", "Treatment", "group1", "group2", "leiden",
    "level1", "level2", "level3",
]
COSMX_RAW_ANNOTATION_COLS = [
    "cell_type", "final_anno", "MajorCellType", "SubCellType",
    "major_celltype", "sub_celltype",
]
TUMOR_RELATED_REGEX = r"epi|epithel|tumou?r|malig|cancer|carcin"


def resolve_genes(gene_str: str) -> list[str]:
    raw = [g.strip() for g in gene_str.split(",") if g.strip()]
    return [GENE_ALIAS.get(g, g) for g in raw]


def _to_text_series(arr_like) -> pd.Series:
    """Convert array-like to clean text series with NA normalization."""
    s = pd.Series(arr_like, copy=False)
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    return s


def _dedupe_keep_order(cols: list[str]) -> list[str]:
    """Deduplicate while preserving the original order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _pick_first_nonempty(df: pd.DataFrame, candidates: list[str], default: str = "Unknown") -> pd.Series:
    """Row-wise coalesce over candidate columns."""
    out = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    for c in candidates:
        if c not in df.columns:
            continue
        s = _to_text_series(df[c])
        out = out.where(out.notna(), s)
    return out.fillna(default).astype(str)


def _extract_tumor_related_label(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Extract first tumor/epithelial/malignant related label from candidate columns."""
    out = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    for c in candidates:
        if c not in df.columns:
            continue
        s = _to_text_series(df[c])
        mask = s.str.contains(TUMOR_RELATED_REGEX, case=False, regex=True, na=False)
        fill_mask = out.isna() & mask
        out.loc[fill_mask] = s.loc[fill_mask]
    return out


def _select_visium_deconv_columns(df: pd.DataFrame) -> list[str]:
    """Select deconvolution columns for Visium including epithelial/tumor channels."""
    cols: list[str] = []
    for c in df.columns:
        if any(c.startswith(p) for p in VISIUM_DECONV_PREFIXES):
            cols.append(c)
            continue
        if any(c.startswith(p) for p in VISIUM_TUMOR_PREFIXES):
            cols.append(c)
            continue
        if c in {"Epi", "EPI"}:
            cols.append(c)
            continue
    cols = _dedupe_keep_order(cols)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _map_cosmx_celltype_to_bt(label: str) -> str:
    """Map CosMx raw labels to broad cell-type channels."""
    u = str(label).upper()
    direct = {
        "EPI": "BT_Epithelial",
        "FIBROENDOMUSCLE": "BT_CAF",
        "MYELOID": "BT_TAM",
        "T/NK": "BT_CD8T",
        "T_OTHER": "BT_CD4T",
        "PLASMA/B": "BT_Other",
        "MAST": "BT_Mast",
        "TNK": "BT_CD8T",
    }
    if u in direct:
        return direct[u]
    if any(k in u for k in ["MALIGN", "TUMOR", "CANCER", "CARCIN"]):
        return "BT_MalignantEpithelial"
    if any(k in u for k in ["EPI", "EPCAM", "KRT", "CLDN"]):
        return "BT_Epithelial"
    if any(k in u for k in ["FIBRO", "CAF", "MYH11"]):
        return "BT_CAF"
    if any(k in u for k in ["MYELOID", "MAC", "TAM"]):
        return "BT_TAM"
    if "MONO" in u:
        return "BT_Monocyte"
    if "CD8" in u:
        return "BT_CD8T"
    if "CD4" in u:
        return "BT_CD4T"
    if "DC" in u:
        return "BT_DC"
    if "NK" in u:
        return "BT_NK"
    if "ENDO" in u or "VASC" in u:
        return "BT_Endothelial"
    if "MAST" in u:
        return "BT_Mast"
    return "BT_Other"


# =========================================================================
# Phase 1: Platform readiness audit
# =========================================================================

def audit_platform(platform: str) -> dict:
    """Check data availability for a single platform."""
    result: dict[str, Any] = {
        "platform": platform,
        "timestamp": datetime.now().isoformat(),
        "has_data": False,
        "has_spatial_coords": False,
        "has_native_celltype": False,
        "has_deconv_celltype": False,
        "n_samples": 0,
        "n_spots_or_cells": 0,
        "gene_coverage": {},
        "reference_available": REF_MANIFEST.exists(),
    }

    if platform == "visium":
        csv_files = sorted(ST_DIR.glob("STmetadata_*.csv")) if ST_DIR.exists() else []
        result["n_samples"] = len(csv_files)
        result["has_data"] = len(csv_files) > 0
        if csv_files:
            df0 = pd.read_csv(csv_files[0], nrows=5)
            result["has_spatial_coords"] = {"x", "y"}.issubset(df0.columns) or {"pxl_col_in_fullres"}.issubset(df0.columns)
            deconv = _select_visium_deconv_columns(df0)
            result["has_deconv_celltype"] = len(deconv) > 0
            result["has_native_celltype"] = any(
                c in df0.columns for c in ["level2", "level3", "level1", "seurat_clusters"]
            )
            total = sum(len(pd.read_csv(f, usecols=[0])) for f in csv_files[:3])
            result["n_spots_or_cells"] = total

    elif platform == "cosmx":
        result["has_data"] = COSMX_H5AD.exists()
        if COSMX_H5AD.exists():
            try:
                import anndata as ad
                a = ad.read_h5ad(COSMX_H5AD, backed="r")
                result["n_spots_or_cells"] = a.n_obs
                result["n_samples"] = a.obs["sample"].nunique() if "sample" in a.obs.columns else 1
                result["has_spatial_coords"] = "spatial" in a.obsm or {"x", "y"}.issubset(a.obs.columns)
                ct_col = next((c for c in ["cell_type", "final_anno"] if c in a.obs.columns), None)
                result["has_native_celltype"] = ct_col is not None
                result["has_deconv_celltype"] = ct_col is not None
                for g in DEFAULT_GENES:
                    vu = {str(v).upper() for v in a.var_names}
                    result["gene_coverage"][g] = g.upper() in vu
                a.file.close()
            except Exception as e:
                result["error"] = str(e)

    elif platform == "visiumhd":
        result["has_data"] = VISIUMHD_H5.exists() and VISIUMHD_POS.exists()
        if result["has_data"]:
            try:
                import scanpy as sc
                a = sc.read_10x_h5(str(VISIUMHD_H5))
                a.var_names_make_unique()
                pos = pd.read_parquet(VISIUMHD_POS)
                in_tissue = pos[pos.get("in_tissue", 1) == 1]
                result["n_spots_or_cells"] = len(in_tissue)
                result["n_samples"] = 1
                result["has_spatial_coords"] = True
                result["has_native_celltype"] = False
                result["has_deconv_celltype"] = True
                vu = {str(v).upper() for v in a.var_names}
                for g in DEFAULT_GENES:
                    result["gene_coverage"][g] = g.upper() in vu
            except Exception as e:
                result["error"] = str(e)

    return result


def run_audit(platforms: list[str]) -> dict:
    """Phase 1: generate audit reports for all requested platforms."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for p in platforms:
        print(f"  Auditing {p} ...")
        r = audit_platform(p)
        reports[p] = r
        out = AUDIT_DIR / f"platform_readiness_{p}.json"
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        status = "READY" if r["has_data"] else "MISSING"
        print(f"    {p}: {status} | spots={r['n_spots_or_cells']} | samples={r['n_samples']}")

    summary_lines = ["# Platform Readiness Summary\n",
                     f"Generated: {datetime.now().isoformat()}\n",
                     f"Reference model: {'YES' if REF_MANIFEST.exists() else 'NO'}\n\n",
                     "| Platform | Data | Spatial | Native CT | Deconv CT | Spots/Cells | Samples |\n",
                     "|----------|------|---------|-----------|-----------|-------------|----------|\n"]
    for p, r in reports.items():
        summary_lines.append(
            f"| {p} | {'Y' if r['has_data'] else 'N'} | "
            f"{'Y' if r['has_spatial_coords'] else 'N'} | "
            f"{'Y' if r['has_native_celltype'] else 'N'} | "
            f"{'Y' if r['has_deconv_celltype'] else 'N'} | "
            f"{r['n_spots_or_cells']} | {r['n_samples']} |\n"
        )
    if any(r.get("gene_coverage") for r in reports.values()):
        summary_lines.append("\n## Gene Coverage\n\n| Gene |")
        for p in platforms:
            summary_lines.append(f" {p} |")
        summary_lines.append("\n|------|")
        for _ in platforms:
            summary_lines.append("------|")
        summary_lines.append("\n")
        for g in DEFAULT_GENES:
            summary_lines.append(f"| {g} |")
            for p in platforms:
                gc = reports[p].get("gene_coverage", {})
                summary_lines.append(f" {'Y' if gc.get(g) else '-'} |")
            summary_lines.append("\n")

    (AUDIT_DIR / "platform_readiness_summary.md").write_text("".join(summary_lines), encoding="utf-8")
    print(f"  Audit saved → {AUDIT_DIR}")
    return reports


# =========================================================================
# Data readers (adapted from run_target_discovery, single-platform)
# =========================================================================

def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.values.astype(float)
    rs = vals.sum(axis=1, keepdims=True)
    rs = np.maximum(rs, 1e-12)
    return pd.DataFrame(vals / rs, columns=df.columns, index=df.index)


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    """Column-wise standardization with zero-variance guard."""
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def _to_broad_type(name: str) -> str:
    """Map platform-specific cell labels to broad biological classes."""
    u = str(name).upper()
    if "MALIGN" in u or "TUMOR" in u or "CANCER" in u or "CARCIN" in u:
        return "MalignantEpithelial"
    if "EPI" in u or "EPCAM" in u or "CLDN" in u:
        return "Epithelial"
    if "FIBRO" in u or "CAF" in u or "MYH11" in u:
        return "CAF"
    if "MAC" in u or "TAM" in u or "MYELOID" in u:
        return "TAM"
    if "MONOCYTE" in u or "MONO" in u:
        return "Monocyte"
    if "CD8" in u:
        return "CD8T"
    if "CD4" in u:
        return "CD4T"
    if "NK" in u:
        return "NK"
    if "DC" in u:
        return "DC"
    if "B" in u or "PLASMA" in u:
        return "B"
    if "ENDO" in u or "VWF" in u or "PECAM1" in u:
        return "Endothelial"
    if "MAST" in u:
        return "Mast"
    return "Other"


def _build_spatial_knn(coords: np.ndarray, sample_ids: np.ndarray,
                       n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Build within-sample Euclidean kNN indices/distances."""
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k = max(1, int(n_neighbors))
    knn_idx = np.zeros((n, k), dtype=int)
    knn_dist = np.zeros((n, k), dtype=float)

    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        gidx = np.where(mask)[0]
        if len(gidx) == 0:
            continue
        if len(gidx) == 1:
            knn_idx[gidx[0], :] = gidx[0]
            knn_dist[gidx[0], :] = 0.0
            continue

        c = coords[gidx]
        kk = min(k, len(gidx))
        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
        nn.fit(c)
        d, i = nn.kneighbors(c)
        mapped = gidx[i]

        if kk < k:
            pad_i = np.repeat(mapped[:, -1:], repeats=(k - kk), axis=1)
            pad_d = np.repeat(d[:, -1:], repeats=(k - kk), axis=1)
            mapped = np.concatenate([mapped, pad_i], axis=1)
            d = np.concatenate([d, pad_d], axis=1)

        knn_idx[gidx] = mapped
        knn_dist[gidx] = d

    return knn_idx, knn_dist


def _aggregate_local_composition(
    comp: pd.DataFrame,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
) -> pd.DataFrame:
    """Weighted local composition using Gaussian-like spatial decay."""
    vals = comp.values.astype(float)
    neigh_vals = vals[knn_idx]  # (N, k, C)
    dist = np.asarray(knn_dist, dtype=float)
    dist = np.maximum(dist, 0.0)
    sigma = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    sigma = max(float(sigma), 1e-6)
    w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    agg = np.einsum("nk,nkc->nc", w, neigh_vals)
    out = pd.DataFrame(agg, index=comp.index, columns=comp.columns)
    return _normalize_rows(out)


def _poincare_project(x2d: np.ndarray, min_radius: float = 0.05,
                      max_radius: float = 0.995) -> np.ndarray:
    """Project 2D Euclidean features into Poincare disk."""
    norm = np.linalg.norm(x2d, axis=1, keepdims=True)
    unit = x2d / np.maximum(norm, 1e-12)
    scale = float(np.quantile(norm, 0.95)) if len(norm) > 0 else 1.0
    scale = max(scale, 1e-6)
    radius = np.tanh(norm / scale)
    radius = np.clip(radius, min_radius, max_radius)
    return unit * radius


def _poincare_distance_to_candidates(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute Poincare distance from one point to many candidates."""
    uu = float(np.sum(u * u))
    vv = np.sum(v * v, axis=1)
    diff2 = np.sum((v - u[None, :]) ** 2, axis=1)
    denom = np.maximum((1.0 - uu) * (1.0 - vv), 1e-12)
    z = 1.0 + 2.0 * diff2 / denom
    z = np.maximum(z, 1.0 + 1e-7)
    return np.arccosh(z)


def _build_hyperbolic_knn(
    hyp_2d: np.ndarray,
    sample_ids: np.ndarray,
    n_neighbors: int = 12,
    candidate_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build within-sample kNN in Poincare distance space."""
    from sklearn.neighbors import NearestNeighbors

    n = hyp_2d.shape[0]
    k = max(1, int(n_neighbors))
    candidate_factor = max(2, int(candidate_factor))
    knn_idx = np.zeros((n, k), dtype=int)
    knn_dist = np.zeros((n, k), dtype=float)

    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        gidx = np.where(mask)[0]
        if len(gidx) == 0:
            continue
        if len(gidx) == 1:
            knn_idx[gidx[0], :] = gidx[0]
            knn_dist[gidx[0], :] = 0.0
            continue

        pts = hyp_2d[gidx]
        candidate_k = min(len(gidx), max(k + 1, candidate_factor * k))
        nn = NearestNeighbors(n_neighbors=candidate_k, metric="euclidean")
        nn.fit(pts)
        _, cand_local = nn.kneighbors(pts)

        for i_local, i_global in enumerate(gidx):
            cand_global = gidx[cand_local[i_local]]
            d_h = _poincare_distance_to_candidates(hyp_2d[i_global], hyp_2d[cand_global])
            order = np.argsort(d_h)
            take = min(k, len(order))
            sel_idx = cand_global[order[:take]]
            sel_dist = d_h[order[:take]]
            if take < k:
                pad_len = k - take
                sel_idx = np.concatenate([sel_idx, np.repeat(i_global, pad_len)])
                sel_dist = np.concatenate([sel_dist, np.zeros(pad_len, dtype=float)])
            knn_idx[i_global] = sel_idx
            knn_dist[i_global] = sel_dist

    return knn_idx, knn_dist


def read_platform_deconv(platform: str, max_spots: int = 150000) -> pd.DataFrame:
    """Read deconv-like composition table for a single platform."""
    if platform == "visium":
        return _read_visium(max_spots=max_spots)
    elif platform == "cosmx":
        return _read_cosmx(max_spots)
    elif platform == "visiumhd":
        return _read_visiumhd(max_spots)
    return pd.DataFrame()


def _read_visium(max_spots: int = 150000) -> pd.DataFrame:
    frames = []
    counter = 0
    for csv_f in sorted(ST_DIR.glob("STmetadata_*.csv")):
        try:
            df = pd.read_csv(csv_f, low_memory=False)
        except Exception:
            continue
        deconv_cols = _select_visium_deconv_columns(df)
        if not deconv_cols:
            continue
        sid = csv_f.stem.replace("STmetadata_", "")
        sub = _normalize_rows(df[deconv_cols])
        sub["sample_id"] = sid
        sub["spot_id"] = [f"{sid}__spot_{i + counter}" for i in range(len(sub))]
        counter += len(sub)
        if {"x", "y"}.issubset(df.columns):
            sub["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).values
            sub["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).values
        elif {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(df.columns):
            sub["x"] = pd.to_numeric(df["pxl_col_in_fullres"], errors="coerce").fillna(0.0).values
            sub["y"] = pd.to_numeric(df["pxl_row_in_fullres"], errors="coerce").fillna(0.0).values
        else:
            sub["x"] = np.arange(len(sub), dtype=float)
            sub["y"] = 0.0

        # Preserve all raw annotation layers (especially epithelial/tumor related).
        for c in VISIUM_RAW_ANNOTATION_COLS:
            if c in df.columns:
                safe = c.lower().replace(".", "_")
                sub[f"raw_{safe}"] = _to_text_series(df[c]).values

        sub["native_celltype"] = _pick_first_nonempty(
            df, ["level2", "level3", "level1", "seurat_clusters"]
        ).values
        sub["native_tumor_epi_label"] = _extract_tumor_related_label(
            df, ["level2", "level3", "level1", "Type"]
        ).values
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) > max_spots:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(np.arange(len(merged)), size=max_spots, replace=False))
        merged = merged.iloc[idx].reset_index(drop=True)
    merged["source_modality"] = "st_visium"
    return merged


def _read_cosmx(max_cells: int = 150000) -> pd.DataFrame:
    if not COSMX_H5AD.exists():
        return pd.DataFrame()
    import anndata as ad
    adata = ad.read_h5ad(COSMX_H5AD, backed="r")
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy()
    else:
        adata.file.close()
        return pd.DataFrame()
    ct_col = next((c for c in ["cell_type", "final_anno"] if c in adata.obs.columns), None)
    if ct_col is None:
        adata.file.close()
        return pd.DataFrame()
    cts = _to_text_series(adata.obs[ct_col]).values
    samples = (
        _to_text_series(adata.obs["sample"]).values
        if "sample" in adata.obs.columns
        else np.array(["CosMx_IFNG"] * len(cts))
    )
    fov_col = next((c for c in ["fov", "FOV", "fov_id", "fovID"] if c in adata.obs.columns), None)
    fovs = (
        _to_text_series(adata.obs[fov_col]).fillna("NA").astype(str).values
        if fov_col is not None
        else np.array(["NA"] * len(cts))
    )
    raw_obs: dict[str, np.ndarray] = {}
    for c in COSMX_RAW_ANNOTATION_COLS:
        if c in adata.obs.columns:
            raw_obs[c] = _to_text_series(adata.obs[c]).values
    adata.file.close()

    if len(cts) > max_cells:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(np.arange(len(cts)), size=max_cells, replace=False))
        cts, coords, samples, fovs = cts[idx], coords[idx], samples[idx], fovs[idx]
        for c in list(raw_obs.keys()):
            raw_obs[c] = raw_obs[c][idx]

    cols = sorted({
        "BT_CAF", "BT_CD4T", "BT_CD8T", "BT_DC", "BT_Endothelial",
        "BT_Epithelial", "BT_MalignantEpithelial", "BT_Mast",
        "BT_Monocyte", "BT_NK", "BT_Other", "BT_TAM",
    })
    mat = np.zeros((len(cts), len(cols)), dtype=float)
    c2i = {c: i for i, c in enumerate(cols)}
    for i, ct in enumerate(cts):
        bt = _map_cosmx_celltype_to_bt(ct)
        mat[i, c2i.get(bt, c2i.get("BT_Other", 0))] = 1.0
    sub = pd.DataFrame(mat, columns=cols)
    sub = _normalize_rows(sub)
    sub["sample_id"] = [f"CosMx_{s}" for s in samples]
    sub["fov_id"] = fovs
    sub["spot_id"] = [f"CosMx__cell_{i}" for i in range(len(sub))]
    sub["x"] = coords[:, 0].astype(float)
    sub["y"] = coords[:, 1].astype(float)
    sub["native_celltype"] = cts
    for c, v in raw_obs.items():
        safe = c.lower().replace(" ", "_")
        sub[f"raw_{safe}"] = v
    sub["native_tumor_epi_label"] = _extract_tumor_related_label(
        pd.DataFrame(raw_obs) if raw_obs else pd.DataFrame({"cell_type": cts}),
        [ct_col, "SubCellType", "MajorCellType", "sub_celltype", "major_celltype", "final_anno", "cell_type"],
    ).values
    sub["source_modality"] = "cosmx"
    return sub


def _read_visiumhd(max_spots: int = 120000) -> pd.DataFrame:
    if not VISIUMHD_H5.exists() or not VISIUMHD_POS.exists():
        return pd.DataFrame()
    import scanpy as sc
    adata = sc.read_10x_h5(str(VISIUMHD_H5))
    adata.var_names_make_unique()
    pos_df = pd.read_parquet(VISIUMHD_POS)
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
        "BT_CAF": ["COL1A1", "COL1A2", "DCN", "FAP", "PDGFRA"],
        "BT_TAM": ["CD68", "LST1", "APOE", "C1QA", "C1QB", "FCER1G", "TYROBP"],
        "BT_CD8T": ["CD3D", "CD3E", "CD8A", "CD8B", "NKG7", "GZMB"],
        "BT_CD4T": ["CD3D", "CD3E", "IL7R", "LTB", "MALAT1"],
        "BT_DC": ["FCER1A", "CLEC10A", "CLEC9A", "LILRA4"],
        "BT_Endothelial": ["VWF", "KDR", "EMCN", "PECAM1"],
        "BT_Monocyte": ["S100A8", "S100A9", "LYZ", "CTSS"],
        "BT_NK": ["NKG7", "GNLY", "KLRD1"],
        "BT_Mast": ["TPSAB1", "KIT", "MS4A2"],
        "BT_Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "CLDN4", "MUC1"],
        "BT_MalignantEpithelial": ["CEACAM5", "EPCAM", "KRT19", "KRT17", "MUC1", "PROM1", "SOX9"],
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
    bt_cols = [c for c in sub.columns if c.startswith("BT_")]
    top_bt = sub[bt_cols].idxmax(axis=1).astype(str) if bt_cols else pd.Series(["BT_Other"] * len(sub))
    sub["sample_id"] = "VisiumHD_16um"
    sub["spot_id"] = [f"VisiumHD__spot_{i}" for i in range(len(sub))]
    sub["x"] = coords[:, 0]
    sub["y"] = coords[:, 1]
    sub["native_celltype"] = top_bt.str.replace("BT_", "", regex=False)
    tum = pd.Series([np.nan] * len(sub), index=sub.index, dtype=object)
    tum.loc[top_bt == "BT_MalignantEpithelial"] = "malignant_epithelial"
    tum.loc[(top_bt == "BT_Epithelial") & tum.isna()] = "epithelial"
    sub["native_tumor_epi_label"] = tum.values
    sub["raw_annotation_source"] = "marker_inference"
    sub["source_modality"] = "visiumhd"
    return sub


def read_platform_gene_expression(platform: str, genes: list[str],
                                  max_spots: int = 150000) -> pd.DataFrame | None:
    """Read raw gene expression for target genes on a platform."""
    if platform == "cosmx":
        if not COSMX_H5AD.exists():
            return None
        import anndata as ad
        adata = ad.read_h5ad(COSMX_H5AD)
        if "spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["spatial"])
        elif {"x", "y"}.issubset(adata.obs.columns):
            coords = adata.obs[["x", "y"]].to_numpy()
        else:
            return None
        if adata.n_obs > max_spots:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_spots, replace=False))
            adata = adata[idx].copy()
            coords = coords[idx]
        vu = {str(v).upper(): str(v) for v in adata.var_names}
        ct_col = next((c for c in ["cell_type", "final_anno"] if c in adata.obs.columns), None)
        rows = {"x": coords[:, 0], "y": coords[:, 1]}
        if ct_col:
            rows["celltype"] = adata.obs[ct_col].astype(str).values
        if "sample" in adata.obs.columns:
            rows["sample_id"] = [f"CosMx_{s}" for s in adata.obs["sample"].astype(str).values]
        fov_col = next((c for c in ["fov", "FOV", "fov_id", "fovID"] if c in adata.obs.columns), None)
        if fov_col is not None:
            rows["fov_id"] = _to_text_series(adata.obs[fov_col]).fillna("NA").astype(str).values
        for g in genes:
            real = vu.get(g.upper())
            if real:
                x = adata[:, real].X
                if hasattr(x, "toarray"):
                    x = x.toarray()
                rows[g] = np.asarray(x).flatten().astype(float)
        return pd.DataFrame(rows)

    elif platform == "visium":
        visium_h5ad = ROOT / "data" / "ST" / "ST_CRC_MSS" / "expression.h5ad"
        if not visium_h5ad.exists():
            return None
        import anndata as ad
        adata = ad.read_h5ad(visium_h5ad)
        if "spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["spatial"])
        elif {"x", "y"}.issubset(adata.obs.columns):
            coords = adata.obs[["x", "y"]].to_numpy()
        else:
            return None
        if adata.n_obs > max_spots:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_spots, replace=False))
            adata = adata[idx].copy()
            coords = coords[idx]
        vu = {str(v).upper(): str(v) for v in adata.var_names}
        rows = {"x": coords[:, 0], "y": coords[:, 1]}
        if "sample_id" in adata.obs.columns:
            rows["sample_id"] = adata.obs["sample_id"].astype(str).values
        elif "sample" in adata.obs.columns:
            rows["sample_id"] = adata.obs["sample"].astype(str).values
        ct_col = next((c for c in ["level2", "level3", "level1", "seurat_clusters"] if c in adata.obs.columns), None)
        if ct_col is not None:
            rows["celltype"] = _to_text_series(adata.obs[ct_col]).fillna("Unknown").values
        for g in genes:
            real = vu.get(g.upper())
            if real:
                x = adata[:, real].X
                if hasattr(x, "toarray"):
                    x = x.toarray()
                rows[g] = np.asarray(x).flatten().astype(float)
        return pd.DataFrame(rows)

    elif platform == "visiumhd":
        if not VISIUMHD_H5.exists():
            return None
        import scanpy as sc
        adata = sc.read_10x_h5(str(VISIUMHD_H5))
        adata.var_names_make_unique()
        pos_df = pd.read_parquet(VISIUMHD_POS)
        pos_df = pos_df[pos_df.get("in_tissue", 1) == 1].set_index("barcode")
        keep = [b for b in adata.obs_names if b in pos_df.index]
        if not keep:
            return None
        adata = adata[keep].copy()
        coords = pos_df.loc[keep][["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
        if adata.n_obs > max_spots:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_spots, replace=False))
            adata = adata[idx].copy()
            coords = coords[idx]
        vu = {str(v).upper(): str(v) for v in adata.var_names}
        rows = {"x": coords[:, 0], "y": coords[:, 1], "sample_id": "VisiumHD_16um"}
        for g in genes:
            real = vu.get(g.upper())
            if real:
                x = adata[:, real].X
                if hasattr(x, "toarray"):
                    x = x.toarray()
                rows[g] = np.asarray(x).flatten().astype(float)
        return pd.DataFrame(rows)

    return None


# =========================================================================
# Phase 2: Multi-scale co-localization niche hierarchy
# =========================================================================

def _compute_coloc_profiles(
    comp_values: np.ndarray, knn_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft co-localization: C_ab = mean(p_a * p_b) over kNN neighborhood.

    Returns (coloc_flat, coloc_mat):
        coloc_flat : (N, C*(C+1)/2) upper-triangle vector per spot
        coloc_mat  : (N, C, C) full co-localization matrix per spot
    """
    N, C = comp_values.shape
    k = knn_idx.shape[1]
    neigh = comp_values[knn_idx]                        # (N, k, C)
    coloc_mat = np.einsum("nka,nkb->nab", neigh, neigh) / k  # (N, C, C)
    triu = np.triu_indices(C)
    coloc_flat = coloc_mat[:, triu[0], triu[1]]         # (N, C*(C+1)/2)
    return coloc_flat, coloc_mat


def _name_niche_unique(
    scale: str, nid: int, enr_row: pd.Series, all_enr: pd.DataFrame,
) -> tuple[str, dict]:
    """Generate ``Scale_Nxx_Type1_Type2[_Interface]`` name with evidence."""
    if enr_row.empty or len(enr_row) == 0:
        return f"{scale.capitalize()}_N{nid:02d}_Mixed", {"uniqueness_score": 0.0}

    top = enr_row.sort_values(ascending=False)
    t1 = top.index[0].replace("BT_", "") if len(top) > 0 else "Mixed"
    t2 = top.index[1].replace("BT_", "") if len(top) > 1 else ""

    others = all_enr.drop(nid, errors="ignore")
    if not others.empty and len(others) > 0:
        diff = enr_row - others.mean()
        unique_score = float(diff.abs().mean())
    else:
        diff = enr_row.copy()
        unique_score = 0.0

    immune = {"CD8T", "CD4T", "NK", "DC", "B"}
    stroma = {"CAF", "TAM", "Monocyte", "Endothelial", "Mast"}
    b1 = _to_broad_type(t1)
    b2 = _to_broad_type(t2) if t2 else ""
    is_interface = bool(
        (b1 in immune and b2 in stroma)
        or (b1 in stroma and b2 in immune)
        or (b1 == "Epithelial" and b2 in (stroma | immune))
        or (b2 == "Epithelial" and b1 in (stroma | immune))
        or (b1 == "MalignantEpithelial" and b2 in (stroma | immune))
        or (b2 == "MalignantEpithelial" and b1 in (stroma | immune))
    )

    suffix = "_Interface" if is_interface and t2 else ""
    parts = [t1]
    if t2 and t2 != t1:
        parts.append(t2)
    name = f"{scale.capitalize()}_N{nid:02d}_{'_'.join(parts)}{suffix}"

    evidence = {
        "top_enriched": top.head(3).to_dict(),
        "uniqueness_score": unique_score,
        "unique_top_types": [
            c.replace("BT_", "")
            for c in diff.sort_values(ascending=False).head(2).index.tolist()
        ],
        "is_interface": is_interface,
    }
    return name, evidence


def _annotate_macro_niche(proto_comp: np.ndarray, deconv_cols: list[str]) -> str:
    """Rule-based macro-niche biological annotation."""
    if len(proto_comp) == 0 or proto_comp.sum() < 1e-12:
        return "TransitionZone"
    idx = np.argsort(-proto_comp)[:2]
    dom = deconv_cols[idx[0]].replace("BT_", "")
    sec = deconv_cols[idx[1]].replace("BT_", "") if len(idx) > 1 else dom
    dom_b, sec_b = _to_broad_type(dom), _to_broad_type(sec)
    if dom_b in {"Epithelial", "MalignantEpithelial"}:
        return "CancerNest"
    if dom_b in {"CD8T", "CD4T", "B", "DC"} or sec_b in {"CD8T", "CD4T", "B", "DC"}:
        return "TLS_like"
    if dom_b in {"CAF", "TAM"} and sec_b in {"Epithelial", "MalignantEpithelial", "CD8T", "CD4T"}:
        return "InvasiveFront_like"
    if dom_b in {"CAF", "Endothelial"} and sec_b in {"CAF", "Endothelial", "Mast"}:
        return "StromalReactive"
    return "TransitionZone"


def build_multiscale_niche(
    platform: str, deconv: pd.DataFrame,
    k_min: int = 8, k_max: int = 18,
) -> dict:
    """Build four-scale co-localization niche hierarchy.

    Scales: micro -> small -> medium -> macro, each defined by
    cell-type co-localization ratios in progressively wider neighborhoods.
    Micro niches are clustered on co-localization features in Poincaré space;
    higher scales are produced by hierarchically merging micro niches.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import linkage, fcluster

    out_dir = NICHE_BASE / platform
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = {
        "sample_id", "spot_id", "x", "y",
        "source_modality", "native_celltype", "sample_spot_idx",
    }
    dynamic_meta = {
        c for c in deconv.columns
        if c.startswith("raw_") or c.startswith("native_")
    }
    meta_cols = meta_cols | dynamic_meta
    deconv_cols = [
        c for c in deconv.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(deconv[c])
    ]
    if not deconv_cols:
        raise ValueError(f"{platform}: no deconvolution columns found")
    dm = deconv[deconv_cols].copy()

    coords = deconv[["x", "y"]].to_numpy(dtype=float)
    sample_ids = (
        deconv["sample_id"].astype(str).values
        if "sample_id" in deconv.columns
        else np.array([platform] * len(deconv))
    )

    # --- Spatial neighborhood (Visium: spot+8; others: kNN) ---
    spatial_k = 9 if platform == "visium" else 12
    sp_knn_idx, sp_knn_dist = _build_spatial_knn(coords, sample_ids, spatial_k)
    dm_local = _aggregate_local_composition(dm, sp_knn_idx, sp_knn_dist)

    # --- Co-localization profiles ---
    comp = dm_local.values.astype(float)
    comp = comp / np.maximum(comp.sum(axis=1, keepdims=True), 1e-12)
    coloc_flat, coloc_mat_all = _compute_coloc_profiles(comp, sp_knn_idx)

    # --- Joint hyperbolic embedding ---
    xy_std = _standardize_columns(coords)
    coloc_std = _standardize_columns(coloc_flat)
    joint = np.concatenate([xy_std, coloc_std], axis=1)

    pca = PCA(n_components=min(2, joint.shape[1]), random_state=42)
    joint_2d = pca.fit_transform(joint)
    hyp_2d = _poincare_project(joint_2d)

    hyper_knn_k = 10 if platform == "visium" else 12
    hyp_knn_idx, hyp_knn_dist = _build_hyperbolic_knn(
        hyp_2d, sample_ids, n_neighbors=hyper_knn_k, candidate_factor=4,
    )

    # --- Micro-scale clustering ---
    hyp_knn_mean = hyp_2d[hyp_knn_idx].mean(axis=1)
    coloc_knn_mean = coloc_flat[hyp_knn_idx].mean(axis=1)
    cluster_feat = np.concatenate([hyp_2d, hyp_knn_mean, coloc_knn_mean], axis=1)

    best_score, best_k = -np.inf, max(2, int(k_min))
    scan_records: list[dict] = []
    idx_scan = np.arange(len(cluster_feat))
    if len(idx_scan) > 30000:
        rng = np.random.default_rng(42)
        idx_scan = np.sort(rng.choice(idx_scan, size=30000, replace=False))
    feat_scan = cluster_feat[idx_scan]
    sil_sub = min(10000, len(feat_scan))

    for k in range(max(2, int(k_min)), max(2, int(k_max)) + 1):
        if k >= len(feat_scan):
            break
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=10)
        lab = km.fit_predict(feat_scan)
        try:
            sil = float(silhouette_score(
                feat_scan, lab, sample_size=sil_sub, random_state=42))
        except Exception:
            sil = 0.0
        counts = np.bincount(lab, minlength=k)
        tiny = 0.12 if int(counts.min()) < max(30, int(0.001 * len(feat_scan))) else 0.0
        score = sil - tiny
        scan_records.append({"k": k, "silhouette": sil, "score": score})
        if score > best_score:
            best_score, best_k = score, k

    micro_k = min(best_k, max(2, len(cluster_feat) - 1))
    km_final = MiniBatchKMeans(
        n_clusters=micro_k, random_state=42, batch_size=4096, n_init=10)
    micro_labels = km_final.fit_predict(cluster_feat)

    # --- Micro prototypes -> hierarchical linkage ---
    C = comp.shape[1]
    micro_proto_coloc = np.zeros((micro_k, coloc_flat.shape[1]))
    micro_proto_comp = np.zeros((micro_k, C))
    for nid in range(micro_k):
        mask = micro_labels == nid
        if mask.sum() > 0:
            micro_proto_coloc[nid] = coloc_flat[mask].mean(axis=0)
            micro_proto_comp[nid] = comp[mask].mean(axis=0)

    Z = linkage(micro_proto_coloc, method="ward") if micro_k >= 2 else None

    small_k = max(min(micro_k - 1, micro_k // 2 + 1), 2) if micro_k > 2 else micro_k
    medium_k = max(min(small_k - 1, small_k // 2 + 1), 2) if small_k > 2 else small_k
    macro_k = max(min(medium_k - 1, medium_k // 2 + 1), 2) if medium_k > 2 else medium_k

    if Z is not None and micro_k > 2:
        small_map = fcluster(Z, t=small_k, criterion="maxclust") - 1
        medium_map = fcluster(Z, t=medium_k, criterion="maxclust") - 1
        macro_map = fcluster(Z, t=macro_k, criterion="maxclust") - 1
    else:
        small_map = np.zeros(micro_k, dtype=int)
        medium_map = np.zeros(micro_k, dtype=int)
        macro_map = np.zeros(micro_k, dtype=int)

    all_labels = {
        "micro": micro_labels,
        "small": small_map[micro_labels],
        "medium": medium_map[micro_labels],
        "macro": macro_map[micro_labels],
    }
    all_k = {
        "micro": micro_k,
        "small": int(small_map.max()) + 1 if len(small_map) else 1,
        "medium": int(medium_map.max()) + 1 if len(medium_map) else 1,
        "macro": int(macro_map.max()) + 1 if len(macro_map) else 1,
    }

    # --- Per-scale: enrichment, co-loc matrix, naming, evidence ---
    global_mean = comp.mean(axis=0)
    scales_output: dict[str, dict] = {}

    for scale in SCALE_ORDER:
        labels_s = all_labels[scale]
        nk = all_k[scale]

        proto_comp_s = np.zeros((nk, C))
        coloc_niche = np.zeros((nk, C, C))
        for nid in range(nk):
            mask = labels_s == nid
            if mask.sum() > 0:
                proto_comp_s[nid] = comp[mask].mean(axis=0)
                coloc_niche[nid] = coloc_mat_all[mask].mean(axis=0)

        # Spot-to-niche soft score from cell-type proportion agreement.
        proto_norm = proto_comp_s / np.maximum(proto_comp_s.sum(axis=1, keepdims=True), 1e-12)
        score_matrix = comp @ proto_norm.T  # (N_spots, n_niches), range ~[0,1]
        assigned_score = score_matrix[np.arange(len(comp)), labels_s]

        enrichment_mat = np.log2(
            (proto_comp_s + 1e-6) / (global_mean[None, :] + 1e-6))
        enrichment_df = pd.DataFrame(
            enrichment_mat, columns=deconv_cols, index=list(range(nk)))
        enrichment_df.index.name = "niche_id"

        triu = np.triu_indices(C)
        pair_names = [
            f"{deconv_cols[i]}|{deconv_cols[j]}" for i, j in zip(triu[0], triu[1])
        ]
        coloc_df = pd.DataFrame(
            coloc_niche[:, triu[0], triu[1]],
            columns=pair_names, index=list(range(nk)),
        )
        coloc_df.index.name = "niche_id"

        niche_names_map: dict[int, str] = {}
        evidence_rows: list[dict] = []
        for nid in range(nk):
            enr_row = (
                enrichment_df.loc[nid]
                if nid in enrichment_df.index
                else pd.Series(dtype=float)
            )
            if scale == "macro":
                macro_ann = _annotate_macro_niche(proto_comp_s[nid], deconv_cols)
                name = f"Macro_N{nid:02d}_{macro_ann}"
                evidence: dict[str, Any] = {"macro_annotation": macro_ann}
            else:
                name, evidence = _name_niche_unique(
                    scale, nid, enr_row, enrichment_df)
            niche_names_map[nid] = name
            evidence["niche_id"] = nid
            evidence["niche_name"] = name
            evidence["scale"] = scale
            evidence["n_spots"] = int((labels_s == nid).sum())
            evidence_rows.append(evidence)

        definition = pd.DataFrame(evidence_rows)
        spot_names = [niche_names_map.get(nid, f"{scale}_{nid}") for nid in labels_s]

        assignment = pd.DataFrame({
            "spot_id": (
                deconv["spot_id"].values
                if "spot_id" in deconv.columns
                else np.arange(len(deconv))
            ),
            "sample_id": sample_ids,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "niche_id": labels_s,
            "niche_name": spot_names,
            "niche_score": assigned_score,
            "scale": scale,
        })
        if "fov_id" in deconv.columns:
            assignment["fov_id"] = _to_text_series(deconv["fov_id"]).fillna("NA").values
        anno_cols = [
            c for c in deconv.columns
            if c == "native_celltype" or c.startswith("native_") or c.startswith("raw_")
        ]
        for ac in anno_cols:
            if ac not in assignment.columns:
                assignment[ac] = deconv[ac].values

        scale_dir = out_dir / scale
        scale_dir.mkdir(parents=True, exist_ok=True)
        definition.to_csv(scale_dir / f"{scale}_niche_definition.csv", index=False)
        assignment.to_csv(scale_dir / f"{scale}_spot_assignment.csv", index=False)
        coloc_df.to_csv(scale_dir / f"{scale}_coloc_matrix.csv")
        enrichment_df.to_csv(scale_dir / f"{scale}_enrichment.csv")

        scales_output[scale] = {
            "labels": labels_s,
            "n_niches": nk,
            "enrichment": enrichment_df,
            "coloc_matrix": coloc_df,
            "score_matrix": score_matrix,
            "definition": definition,
            "assignment": assignment,
            "niche_names_map": niche_names_map,
            "proto_comp": proto_comp_s,
        }

    # --- Summaries ---
    r_h = np.linalg.norm(hyp_2d, axis=1)
    colors = get_color_mapping(
        [str(n) for n in sorted(set(micro_labels))], PALETTE_CATEGORICAL)
    niche_colors = {
        int(k_): v for k_, v in zip(sorted(set(micro_labels)), colors.values())
    }

    (out_dir / "niche_resolution_scan.json").write_text(json.dumps(
        {"platform": platform, "micro_k": micro_k, "small_k": all_k["small"],
         "medium_k": all_k["medium"], "macro_k": all_k["macro"],
         "scan": scan_records}, indent=2), encoding="utf-8")
    (out_dir / "niche_color_map.json").write_text(
        json.dumps(niche_colors, indent=2), encoding="utf-8")
    (out_dir / "hyperbolic_knn_summary.json").write_text(json.dumps(
        {"platform": platform, "spatial_k": spatial_k,
         "hyper_knn_k": hyper_knn_k}, indent=2), encoding="utf-8")

    print(f"  {platform}: micro_k={micro_k} small_k={all_k['small']} "
          f"medium_k={all_k['medium']} macro_k={all_k['macro']} "
          f"score={best_score:.3f}")

    return {
        "scales": scales_output,
        "hyp_2d": hyp_2d,
        "hyp_radius": r_h,
        "niche_colors": niche_colors,
        "micro_labels": micro_labels,
        "all_labels": all_labels,
        "all_k": all_k,
        "linkage_Z": Z,
        "deconv_cols": deconv_cols,
        "comp_values": comp,
        "coloc_flat": coloc_flat,
        "coloc_mat_all": coloc_mat_all,
        "sample_ids": sample_ids,
        "coords": coords,
    }


# =========================================================================
# Phase 3: Figure generation helpers
# =========================================================================

def _compute_density_sizes(
    coords: np.ndarray,
    sample_ids: np.ndarray | None = None,
    *,
    min_size: float = 0.8,
    max_size: float = 18.0,
    k: int = 8,
) -> np.ndarray:
    """Per-point adaptive size based on local spatial density (kNN mean dist).

    Dense regions get smaller points, sparse regions get larger ones.
    Computed within each sample to handle multi-sample datasets.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    mean_dist = np.ones(n, dtype=float)
    if sample_ids is None:
        sample_ids = np.zeros(n, dtype=int)
    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        kk = min(k + 1, len(idx))
        nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
        nn.fit(coords[idx])
        dists, _ = nn.kneighbors(coords[idx])
        mean_dist[idx] = dists[:, 1:].mean(axis=1)

    lo, hi = np.percentile(mean_dist, [2, 98])
    if hi - lo < 1e-12:
        return np.full(n, (min_size + max_size) / 2)
    normed = np.clip((mean_dist - lo) / (hi - lo), 0, 1)
    return min_size + normed * (max_size - min_size)


def _apply_sample_count_rescale(
    sizes: np.ndarray,
    sample_ids: np.ndarray | None,
    *,
    target_ref_n: int = 3000,
    min_scale: float = 0.65,
    max_scale: float = 1.10,
) -> np.ndarray:
    """Second-pass size scaling by per-sample spot count."""
    out = np.asarray(sizes, dtype=float).copy()
    if sample_ids is None:
        return out
    sample_ids = np.asarray(sample_ids)
    for sid in pd.unique(sample_ids):
        mask = sample_ids == sid
        n = int(mask.sum())
        if n <= 0:
            continue
        scale = (float(target_ref_n) / float(n)) ** 0.2
        scale = float(np.clip(scale, min_scale, max_scale))
        out[mask] *= scale
    return out


def _sanitize_token(text: str) -> str:
    """Sanitize text for file name tokens."""
    s = str(text).strip().replace("\\", "_").replace("/", "_").replace(" ", "_")
    s = s.replace(":", "_").replace("|", "_")
    return s[:80] if s else "unknown"


def _build_plot_units(
    platform: str,
    n_points: int,
    sample_ids: np.ndarray | None = None,
    fov_ids: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Build plotting units: overall + optional per-sample split."""
    units: list[tuple[str, np.ndarray]] = [("all", np.ones(n_points, dtype=bool))]

    if platform == "visium" and sample_ids is not None:
        sid_arr = np.asarray(sample_ids).astype(str)
        for sid in sorted(pd.unique(sid_arr)):
            units.append((str(sid), sid_arr == sid))
        return units

    if platform == "cosmx" and sample_ids is not None:
        # CosMx maps should be split by sample (no stacked "all" panel).
        units = []
        sid_arr = np.asarray(sample_ids).astype(str)
        for sid in sorted(pd.unique(sid_arr)):
            units.append((str(sid), sid_arr == sid))
        return units

    return units


def _plot_spatial_scatter(coords: np.ndarray, values, title: str, path: Path,
                          cmap=None, categorical: bool = False,
                          color_map: dict | None = None,
                          s: float | np.ndarray = 1.0,
                          highlight_label: str | None = None):
    """Generic spatial scatter: continuous heatmap or categorical.

    If *highlight_label* is given (categorical mode only), only that label
    is drawn in colour; the rest is grey background.
    """
    apply_cns_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sizes = np.broadcast_to(np.asarray(s, dtype=float), len(coords))

    if categorical:
        labels_u = sorted(set(values))
        cm = color_map or get_color_mapping(labels_u)
        vals_arr = np.asarray(values)

        if highlight_label is not None:
            bg = vals_arr != highlight_label
            if bg.any():
                ax.scatter(coords[bg, 0], coords[bg, 1], c="#E0E0E0",
                           s=sizes[bg] * 0.45, alpha=1.0,
                           marker="o", edgecolors="none", linewidths=0.0,
                           rasterized=True)
            fg = vals_arr == highlight_label
            if fg.any():
                ax.scatter(coords[fg, 0], coords[fg, 1],
                           c=cm.get(highlight_label, "#E41A1C"),
                           s=sizes[fg], alpha=1.0,
                           marker="o", edgecolors="none", linewidths=0.0,
                           label=highlight_label,
                           rasterized=True)
            ax.legend(fontsize=8, markerscale=2, loc="upper right", frameon=False)
        else:
            for lab in labels_u:
                mask = vals_arr == lab
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=cm.get(lab, "#999"),
                           s=sizes[mask], alpha=1.0,
                           marker="o", edgecolors="none", linewidths=0.0,
                           label=lab, rasterized=True)
            place_legend_outside(ax, fontsize=6, markerscale=3)
    else:
        vals = np.asarray(values, dtype=float)
        finite = np.isfinite(vals)
        vmin, vmax = np.nanpercentile(vals[finite], [2, 98]) if finite.any() else (0, 1)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals,
                        cmap=cmap or CMAP_SPATIAL,
                        s=sizes, alpha=1.0,
                        marker="o", edgecolors="none", linewidths=0.0,
                        vmin=vmin, vmax=vmax,
                        rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Expression")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    add_watermark(ax)
    save_figure(fig, path)


def _plot_violin(
    data: pd.DataFrame, gene: str, group_col: str, title: str, path: Path,
    log1p: bool = True,
    palette: dict[str, str] | None = None,
):
    """Violin plot of gene expression by group with optional log1p transform."""
    apply_cns_style()
    import seaborn as sns
    if gene not in data.columns or group_col not in data.columns:
        return
    plot_df = data[[gene, group_col]].dropna()
    if plot_df.empty:
        return
    y_label = gene
    if log1p:
        plot_df = plot_df.copy()
        plot_df[gene] = np.log1p(plot_df[gene].astype(float))
        y_label = f"log1p({gene})"
    n_groups = plot_df[group_col].nunique()
    fig, ax = plt.subplots(1, 1, figsize=(max(6, n_groups * 0.8), 5))
    order = sorted(plot_df[group_col].unique())
    use_hue = palette is not None
    sns.violinplot(
        data=plot_df, x=group_col, y=gene, order=order, ax=ax,
        hue=group_col if use_hue else None,
        palette=palette if use_hue else None,
        dodge=False, legend=False,
        inner="box", linewidth=0.6, cut=0, scale="width",
    )
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    add_watermark(ax)
    save_figure(fig, path)


def _plot_dotplot(data: pd.DataFrame, genes: list[str], group_col: str,
                  title: str, path: Path):
    """Dot plot: fraction expressing vs mean expression."""
    apply_cns_style()
    if group_col not in data.columns:
        return
    groups = sorted(data[group_col].unique())
    valid_genes = [g for g in genes if g in data.columns]
    if not valid_genes or not groups:
        return
    frac_mat = np.zeros((len(groups), len(valid_genes)))
    mean_mat = np.zeros_like(frac_mat)
    for i, grp in enumerate(groups):
        sub = data.loc[data[group_col] == grp, valid_genes]
        if sub.empty:
            continue
        frac_mat[i] = (sub > 0).mean().values
        mean_mat[i] = sub.mean().values

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(valid_genes) * 0.9), max(4, len(groups) * 0.5)))
    for i, grp in enumerate(groups):
        for j, gene in enumerate(valid_genes):
            size = frac_mat[i, j] * 200
            color = mean_mat[i, j]
            ax.scatter(j, i, s=size, c=color, cmap=CMAP_EXPRESSION,
                       vmin=0, vmax=max(mean_mat.max(), 1e-3), edgecolors="gray", linewidths=0.3)
    ax.set_xticks(range(len(valid_genes)))
    ax.set_xticklabels(valid_genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, len(valid_genes) - 0.5)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    sm = plt.cm.ScalarMappable(cmap=CMAP_EXPRESSION, norm=plt.Normalize(0, max(mean_mat.max(), 1e-3)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Mean expr")
    add_watermark(ax)
    save_figure(fig, path)


def _build_knn_indices(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """kNN indices on coordinates for local co-localization scoring."""
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    if n <= 1:
        return np.zeros((n, 1), dtype=int)
    kk = max(2, min(int(k), n))
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
    nn.fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx


def _normalize_expr_matrix(expr: np.ndarray) -> np.ndarray:
    """Gene-wise robust normalization to [0,1] by q05-q95."""
    x = np.asarray(expr, dtype=float)
    if x.size == 0:
        return x
    q05 = np.nanquantile(x, 0.05, axis=0)
    q95 = np.nanquantile(x, 0.95, axis=0)
    den = np.maximum(q95 - q05, 1e-12)
    xn = (x - q05[None, :]) / den[None, :]
    return np.clip(xn, 0.0, 1.0)


def _plot_coloc_heatmap(
    mean_mat: np.ndarray,
    genes: list[str],
    groups: list[str],
    title: str,
    path: Path,
):
    """Heatmap for gene-group co-localization scores."""
    if mean_mat.size == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(max(7, 0.5 * len(groups) + 4), max(5, 0.5 * len(genes) + 2)))
    vmax = max(float(np.nanmax(mean_mat)), 1e-6)
    im = ax.imshow(mean_mat, aspect="auto", cmap=CMAP_EXPRESSION, vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Mean co-localization score")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8)
    ax.set_title(title, fontsize=11)
    add_watermark(ax)
    save_figure(fig, path)


def _plot_coloc_dotplot(
    mean_mat: np.ndarray,
    frac_mat: np.ndarray,
    genes: list[str],
    groups: list[str],
    title: str,
    path: Path,
):
    """Dotplot for co-localization: color=mean, size=fraction(high)."""
    if mean_mat.size == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(max(7, 0.5 * len(groups) + 4), max(5, 0.5 * len(genes) + 2)))
    vmax = max(float(np.nanmax(mean_mat)), 1e-6)
    for i, g in enumerate(genes):
        for j, grp in enumerate(groups):
            size = 12 + 180 * float(frac_mat[i, j])
            color = float(mean_mat[i, j])
            ax.scatter(j, i, s=size, c=color, cmap=CMAP_EXPRESSION,
                       vmin=0.0, vmax=vmax, edgecolors="none", marker="o")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8)
    ax.set_title(title, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=CMAP_EXPRESSION, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.7, label="Mean co-localization score")
    add_watermark(ax)
    save_figure(fig, path)


def _plot_coloc_violin_per_gene(
    coloc_tensor: np.ndarray,
    genes: list[str],
    groups: list[str],
    palette: dict[str, str],
    title_prefix: str,
    out_dir: Path,
):
    """Per-gene violin of co-localization scores across groups."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for gi, gene in enumerate(genes):
        arrays = []
        valid_groups = []
        for gj, grp in enumerate(groups):
            vals = coloc_tensor[:, gi, gj]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            arrays.append(vals)
            valid_groups.append(grp)
        if not arrays:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(max(7, 0.5 * len(valid_groups) + 4), 4.6))
        parts = ax.violinplot(
            arrays,
            positions=np.arange(1, len(valid_groups) + 1),
            widths=0.85,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body, grp in zip(parts["bodies"], valid_groups):
            body.set_facecolor(palette.get(grp, "#6FAFC2"))
            body.set_edgecolor("none")
            body.set_alpha(0.9)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#333333")
            parts["cmedians"].set_linewidth(1.0)
        ax.set_xticks(np.arange(1, len(valid_groups) + 1))
        ax.set_xticklabels(valid_groups, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Co-localization score", fontsize=10)
        ax.set_title(f"{title_prefix} — {gene}", fontsize=11)
        add_watermark(ax)
        save_figure(fig, out_dir / f"violin_{_sanitize_token(gene)}.png")


def _map_expr_to_deconv_indices(
    expr_coords: np.ndarray,
    expr_sample: np.ndarray,
    deconv_coords: np.ndarray,
    deconv_sample: np.ndarray,
) -> np.ndarray:
    """Map each expression spot/cell to nearest deconv spot/cell within sample."""
    from sklearn.neighbors import NearestNeighbors
    n = len(expr_coords)
    mapped = np.full(n, -1, dtype=int)
    e_s = np.asarray(expr_sample).astype(str)
    d_s = np.asarray(deconv_sample).astype(str)
    for sid in sorted(set(e_s)):
        em = e_s == sid
        dm = d_s == sid
        eidx = np.where(em)[0]
        didx = np.where(dm)[0]
        if len(eidx) == 0 or len(didx) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(deconv_coords[didx])
        _, ridx = nn.kneighbors(expr_coords[eidx])
        mapped[eidx] = didx[ridx[:, 0]]
    return mapped


def _generate_sample_coloc_plots(
    *,
    sample_id: str,
    expr_coords: np.ndarray,
    expr_values_norm: np.ndarray,
    genes: list[str],
    group_labels: np.ndarray,
    group_name: str,
    out_dir: Path,
    palette: dict[str, str] | None = None,
):
    """Generate per-sample co-localization heatmap/dotplot/violin."""
    labels = _to_text_series(group_labels).fillna("NA").values.astype(str)
    groups = [g for g in sorted(pd.unique(labels)) if g != "NA"]
    if len(groups) == 0 or expr_values_norm.size == 0:
        return

    knn_idx = _build_knn_indices(expr_coords, k=8)
    n = len(labels)
    gk = len(groups)
    local_mat = np.zeros((n, gk), dtype=float)
    for gi, grp in enumerate(groups):
        ind = (labels == grp).astype(float)
        local_mat[:, gi] = ind[knn_idx].mean(axis=1)

    # (N, G, K) = expr_norm(spot,gene) * local_group_score(spot,group)
    coloc_tensor = expr_values_norm[:, :, None] * local_mat[:, None, :]
    mean_mat = coloc_tensor.mean(axis=0)                  # (G, K)
    frac_mat = (coloc_tensor > 0.20).mean(axis=0)         # (G, K)

    out_dir.mkdir(parents=True, exist_ok=True)
    mean_df = pd.DataFrame(mean_mat, index=genes, columns=groups)
    frac_df = pd.DataFrame(frac_mat, index=genes, columns=groups)
    mean_df.to_csv(out_dir / f"coloc_mean_{_sanitize_token(sample_id)}.csv")
    frac_df.to_csv(out_dir / f"coloc_frac_{_sanitize_token(sample_id)}.csv")

    ttl = f"{sample_id} — {group_name} co-localization"
    _plot_coloc_heatmap(
        mean_mat=mean_mat,
        genes=genes,
        groups=groups,
        title=f"{ttl} heatmap",
        path=out_dir / f"heatmap_{_sanitize_token(sample_id)}.png",
    )
    _plot_coloc_dotplot(
        mean_mat=mean_mat,
        frac_mat=frac_mat,
        genes=genes,
        groups=groups,
        title=f"{ttl} dotplot",
        path=out_dir / f"dotplot_{_sanitize_token(sample_id)}.png",
    )

    if palette is None:
        palette = get_color_mapping(groups)
    _plot_coloc_violin_per_gene(
        coloc_tensor=coloc_tensor,
        genes=genes,
        groups=groups,
        palette=palette,
        title_prefix=f"{sample_id} — {group_name}",
        out_dir=out_dir / f"violin_{_sanitize_token(sample_id)}",
    )


def _plot_clustered_enrichment_heatmap(
    enrichment: pd.DataFrame, title: str, path: Path,
):
    """Niche × cell-type enrichment with row+column Ward hierarchical clustering."""
    apply_cns_style()
    import seaborn as sns

    if enrichment is None or enrichment.empty or enrichment.shape[0] < 2:
        if enrichment is not None and not enrichment.empty:
            show = enrichment.copy()
            show.index = [f"N{int(i):02d}" for i in show.index]
            show.columns = [c.replace("BT_", "") for c in show.columns]
            fig, ax = plt.subplots(figsize=(max(7, show.shape[1] * 0.9), 3))
            import seaborn as sns
            sns.heatmap(show, cmap="RdBu_r", center=0.0, linewidths=0.3,
                        linecolor="white", ax=ax,
                        cbar_kws={"label": u"log\u2082 enrichment"})
            ax.set_title(title, fontsize=11)
            add_watermark(ax)
            save_figure(fig, path)
        return

    show = enrichment.copy()
    show.index = [f"N{int(i):02d}" for i in show.index]
    show.columns = [c.replace("BT_", "") for c in show.columns]

    row_cluster = show.shape[0] >= 2
    col_cluster = show.shape[1] >= 2
    n_rows, n_cols = show.shape
    # Keep heatmap output ratio stable for manuscript/dashboard use.
    fig_w = min(max(7.5, n_cols * 0.48 + 3.5), 17.0)
    fig_h = min(max(5.5, n_rows * 0.55 + 3.5), 15.0)
    ratio = fig_w / max(fig_h, 1e-6)
    if ratio < 1.15:
        fig_w = min(17.0, fig_h * 1.15)
    elif ratio > 2.2:
        fig_h = max(5.5, fig_w / 2.2)

    with plt.rc_context({"figure.constrained_layout.use": False}):
        g = sns.clustermap(
            show,
            method="ward",
            metric="euclidean",
            cmap="RdBu_r",
            center=0.0,
            linewidths=0.3,
            linecolor="white",
            figsize=(fig_w, fig_h),
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            cbar_kws={"label": u"log\u2082 enrichment"},
            dendrogram_ratio=(0.12, 0.10),
            xticklabels=True,
            yticklabels=True,
        )
    g.ax_heatmap.set_title(title, fontsize=11, pad=15)
    g.ax_heatmap.set_xlabel("Cell Type", fontsize=10)
    g.ax_heatmap.set_ylabel("Niche", fontsize=10)
    g.ax_heatmap.tick_params(axis="x", labelsize=8, rotation=45)
    g.ax_heatmap.tick_params(axis="y", labelsize=8)
    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_horizontalalignment("right")
    add_watermark(g.ax_heatmap)
    save_figure(g.fig, path)


def _plot_poincare_multiscale(
    hyp_2d: np.ndarray, all_labels: dict, title: str, path: Path,
):
    """Four-panel Poincaré disk colored by micro/small/medium/macro."""
    apply_cns_style()
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    theta = np.linspace(0, 2 * np.pi, 300)

    for ax_idx, scale_name in enumerate(SCALE_ORDER):
        ax = axes[ax_idx]
        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.5)
        for r in [0.25, 0.5, 0.75]:
            ax.plot(
                r * np.cos(theta), r * np.sin(theta),
                color="#DDD", linewidth=0.4, linestyle="--",
            )
        labels_s = all_labels[scale_name]
        unique_labels = sorted(set(labels_s))
        cm = get_color_mapping([str(la) for la in unique_labels])
        for lab in unique_labels:
            m = labels_s == lab
            ax.scatter(
                hyp_2d[m, 0], hyp_2d[m, 1],
                s=get_dynamic_point_size(int(m.sum())),
                c=cm[str(lab)], alpha=0.5,
                edgecolors="none", rasterized=True,
                label=f"N{lab:02d}" if len(unique_labels) <= 12 else "",
            )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"{scale_name.capitalize()} ({len(unique_labels)} niches)", fontsize=11)
        ax.set_xlabel(u"Poincar\u00e9 x")
        ax.set_ylabel(u"Poincar\u00e9 y")
        if len(unique_labels) <= 12:
            ax.legend(fontsize=6, markerscale=2, loc="upper right", frameon=False)
    fig.suptitle(title, fontsize=13, y=1.02)
    add_watermark(axes[-1])
    save_figure(fig, path)


def _plot_niche_dendrogram_from_Z(
    Z, micro_k: int, title: str, path: Path,
):
    """Dendrogram of micro-niche prototypes from precomputed linkage."""
    apply_cns_style()
    from scipy.cluster.hierarchy import dendrogram
    if Z is None or micro_k < 2:
        return
    fig, ax = plt.subplots(1, 1, figsize=(max(6, micro_k * 0.7), 4.5))
    dendrogram(Z, labels=[f"N{i:02d}" for i in range(micro_k)],
               leaf_rotation=45, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Linkage distance")
    add_watermark(ax)
    save_figure(fig, path)


def _plot_cross_sample_heatmap(
    consistency_result: dict, title: str, path: Path,
):
    """Cross-sample niche prototype correlation heatmap."""
    apply_cns_style()
    import seaborn as sns

    pairwise = consistency_result.get("pairwise_consistency", [])
    if not pairwise:
        return
    samples = sorted(set(
        [p["sample_a"] for p in pairwise] + [p["sample_b"] for p in pairwise]
    ))
    n = len(samples)
    mat = np.eye(n)
    s2i = {s: i for i, s in enumerate(samples)}
    for p in pairwise:
        i, j = s2i[p["sample_a"]], s2i[p["sample_b"]]
        mat[i, j] = p["mean_match_corr"]
        mat[j, i] = p["mean_match_corr"]

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.6)))
    sns.heatmap(
        pd.DataFrame(mat, index=samples, columns=samples),
        cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".2f",
        linewidths=0.3, ax=ax,
    )
    ax.set_title(title, fontsize=11)
    add_watermark(ax)
    save_figure(fig, path)


def _plot_consistency_bar(
    consistency_by_scale: dict, title: str, path: Path,
):
    """Bar chart of mean / Q25 consistency across scales."""
    apply_cns_style()
    scales, means, q25s = [], [], []
    for sn in SCALE_ORDER:
        if sn in consistency_by_scale:
            c = consistency_by_scale[sn]
            scales.append(sn.capitalize())
            means.append(c.get("mean_consistency", 0))
            q25s.append(c.get("q25_consistency", 0))
    if not scales:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(scales))
    ax.bar(x - 0.15, means, width=0.3, label="Mean match corr", color="#377EB8")
    ax.bar(x + 0.15, q25s, width=0.3, label="Q25 (robust)", color="#E41A1C", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title, fontsize=11)
    add_watermark(ax)
    save_figure(fig, path)


# =========================================================================
# Phase 3b: generate all figures for a platform
# =========================================================================

_TOPN_TYPE_MAPS = 12
_TUMOR_EPI_PRIORITY_LABELS = {
    "epithelia", "epi_cea", "epi_mki67", "epi_normal",
    "epithelial", "malignant_epithelial", "tumor_center",
    "boundary_neighborhoods", "cancernest",
}


def _select_type_map_labels(values: np.ndarray, topn: int = _TOPN_TYPE_MAPS) -> list[str]:
    """Select labels for per-type spatial maps: tumor/epi priority + TopN by count."""
    from collections import Counter
    counts = Counter(str(v) for v in values if str(v) not in {"NA", "nan", ""})
    priority = [
        lab for lab in counts
        if lab.lower() in _TUMOR_EPI_PRIORITY_LABELS
        or any(k in lab.lower() for k in ("epi", "tumor", "malig", "cancer"))
    ]
    remaining = [
        lab for lab, _ in counts.most_common()
        if lab not in priority
    ]
    selected = priority + remaining
    return selected[:topn]


def generate_platform_figures(
    platform: str, niche_pack: dict,
    deconv: pd.DataFrame, genes: list[str],
    max_spots: int = 150000,
):
    """Generate all per-platform spatial + niche figures."""
    fig_dir = FIG_BASE / platform
    fig_dir.mkdir(parents=True, exist_ok=True)
    fixed_point_size = 3.2

    scales = niche_pack["scales"]
    hyp_2d = niche_pack["hyp_2d"]
    all_labels = niche_pack["all_labels"]
    coords = niche_pack["coords"]

    sample_ids = niche_pack.get("sample_ids")
    density_s = np.full(len(coords), fixed_point_size, dtype=float)

    fov_ids = (
        _to_text_series(deconv["fov_id"]).fillna("NA").astype(str).values
        if "fov_id" in deconv.columns
        else None
    )
    sample_plot_units = _build_plot_units(
        platform=platform,
        n_points=len(coords),
        sample_ids=sample_ids,
        fov_ids=fov_ids,
    )

    # ----------------------------------------------------------------
    # A) Native annotation: 1 overview + per-type highlight maps
    # ----------------------------------------------------------------
    anno_plot_cols = [
        "native_celltype",
        "native_tumor_epi_label",
        "raw_level1", "raw_level2", "raw_level3",
        "raw_seurat_clusters",
    ]
    for ac in anno_plot_cols:
        if ac not in deconv.columns:
            continue
        vals = _to_text_series(deconv[ac]).fillna("NA").values
        uniq = sorted(set(v for v in vals if v != "NA"))
        if len(uniq) < 1:
            continue
        cm = get_color_mapping(uniq)

        for sid_name, smask in sample_plot_units:
            tag = "all" if sid_name == "all" else f"sample_{_sanitize_token(sid_name)}"
            title_tail = "all samples" if sid_name == "all" else f"sample={sid_name}"
            _plot_spatial_scatter(
                coords[smask], vals[smask],
                f"{platform} \u2014 {ac} ({title_tail})",
                fig_dir / f"{ac}_spatial__{tag}.png",
                categorical=True, s=density_s[smask], color_map=cm,
            )

        type_dir = fig_dir / f"{ac}_types"
        type_dir.mkdir(parents=True, exist_ok=True)
        selected = _select_type_map_labels(vals)
        for lab in selected:
            safe = lab.replace("/", "_").replace(" ", "_")[:40]
            for sid_name, smask in sample_plot_units:
                tag = "all" if sid_name == "all" else f"sample_{_sanitize_token(sid_name)}"
                title_tail = "all samples" if sid_name == "all" else f"unit={sid_name}"
                _plot_spatial_scatter(
                    coords[smask], vals[smask],
                    f"{platform} \u2014 {ac}: {lab} ({title_tail})",
                    type_dir / f"{safe}__{tag}.png",
                    categorical=True, s=density_s[smask], color_map=cm,
                    highlight_label=lab,
                )

    # ----------------------------------------------------------------
    # B) Per-scale niche: 1 overview + per-niche highlight maps + heatmap
    # ----------------------------------------------------------------
    for scale_name in SCALE_ORDER:
        sd = scales[scale_name]
        labels_s = sd["labels"]
        enrichment = sd["enrichment"]
        nk = sd["n_niches"]
        niche_names_map = sd["niche_names_map"]
        score_matrix = sd.get("score_matrix")

        spot_names = np.array(
            [niche_names_map.get(nid, f"N{nid}") for nid in labels_s])
        cm_niche = get_color_mapping(sorted(set(spot_names)))
        assigned_score = None
        if score_matrix is not None:
            score_matrix = np.asarray(score_matrix, dtype=float)
            assigned_score = score_matrix[np.arange(len(labels_s)), labels_s]

        for sid_name, smask in sample_plot_units:
            tag = "all" if sid_name == "all" else f"sample_{_sanitize_token(sid_name)}"
            title_tail = "all samples" if sid_name == "all" else f"sample={sid_name}"
            _plot_spatial_scatter(
                coords[smask], spot_names[smask],
                f"{platform} \u2014 {scale_name.capitalize()} Niches (k={nk}, {title_tail})",
                fig_dir / f"{scale_name}_niche_spatial__{tag}.png",
                categorical=True, s=density_s[smask], color_map=cm_niche,
            )
            if assigned_score is not None:
                _plot_spatial_scatter(
                    coords[smask], assigned_score[smask],
                    f"{platform} \u2014 {scale_name.capitalize()} Niche Score ({title_tail})",
                    fig_dir / f"{scale_name}_niche_score_spatial__{tag}.png",
                    categorical=False, cmap=CMAP_EXPRESSION, s=density_s[smask],
                )

        niche_type_dir = fig_dir / f"{scale_name}_niche_types"
        niche_type_dir.mkdir(parents=True, exist_ok=True)
        selected_niches = _select_type_map_labels(spot_names, topn=_TOPN_TYPE_MAPS)
        niche_name_to_id = {v: int(k) for k, v in niche_names_map.items()}
        for nlab in selected_niches:
            safe = nlab.replace("/", "_").replace(" ", "_")[:60]
            nid = niche_name_to_id.get(nlab)
            for sid_name, smask in sample_plot_units:
                tag = "all" if sid_name == "all" else f"sample_{_sanitize_token(sid_name)}"
                title_tail = "all samples" if sid_name == "all" else f"unit={sid_name}"
                if score_matrix is not None and nid is not None:
                    nscore = score_matrix[:, int(nid)]
                    _plot_spatial_scatter(
                        coords[smask], nscore[smask],
                        f"{platform} \u2014 {scale_name}: {nlab} score ({title_tail})",
                        niche_type_dir / f"{safe}__{tag}_score.png",
                        categorical=False, cmap=CMAP_EXPRESSION, s=density_s[smask],
                    )
                else:
                    _plot_spatial_scatter(
                        coords[smask], spot_names[smask],
                        f"{platform} \u2014 {scale_name}: {nlab} ({title_tail})",
                        niche_type_dir / f"{safe}__{tag}.png",
                        categorical=True, s=density_s[smask], color_map=cm_niche,
                        highlight_label=nlab,
                    )

        _plot_clustered_enrichment_heatmap(
            enrichment,
            f"{platform} \u2014 {scale_name.capitalize()} Enrichment",
            fig_dir / f"{scale_name}_enrichment_clustermap.png",
        )

    # ----------------------------------------------------------------
    # C) Poincare + Dendrogram
    # ----------------------------------------------------------------
    _plot_poincare_multiscale(
        hyp_2d, all_labels,
        f"{platform} \u2014 Poincar\u00e9 Multi-Scale",
        fig_dir / "poincare_multiscale.png",
    )
    _plot_niche_dendrogram_from_Z(
        niche_pack.get("linkage_Z"),
        niche_pack["all_k"]["micro"],
        f"{platform} \u2014 Micro Niche Dendrogram",
        fig_dir / "niche_hierarchy_dendrogram.png",
    )

    # ----------------------------------------------------------------
    # D) Gene expression spatial + violin (log1p) + dotplot
    # ----------------------------------------------------------------
    expr_df = read_platform_gene_expression(platform, genes, max_spots=max_spots)
    if expr_df is not None and not expr_df.empty:
        expr_coords = expr_df[["x", "y"]].values
        expr_sample = (
            expr_df["sample_id"].values
            if "sample_id" in expr_df.columns else None
        )
        expr_fov = (
            _to_text_series(expr_df["fov_id"]).fillna("NA").astype(str).values
            if "fov_id" in expr_df.columns
            else None
        )
        gene_density_s = np.full(len(expr_df), fixed_point_size, dtype=float)
        expr_plot_units = _build_plot_units(
            platform=platform,
            n_points=len(expr_df),
            sample_ids=expr_sample,
            fov_ids=expr_fov,
        )

        for gene in genes:
            if gene not in expr_df.columns:
                continue
            gvals = np.log1p(expr_df[gene].astype(float).values)
            for sid_name, smask in expr_plot_units:
                tag = "all" if sid_name == "all" else f"sample_{_sanitize_token(sid_name)}"
                title_tail = "all samples" if sid_name == "all" else f"unit={sid_name}"
                _plot_spatial_scatter(
                    expr_coords[smask], gvals[smask],
                    f"{platform} \u2014 {gene} log1p(expr) ({title_tail})",
                    fig_dir / f"gene_spatial_{gene}__{tag}.png", s=gene_density_s[smask],
                )

        if "celltype" in expr_df.columns:
            for gene in genes:
                if gene not in expr_df.columns:
                    continue
                _plot_violin(
                    expr_df, gene, "celltype",
                    f"{platform} \u2014 {gene} by Cell Type [log1p]",
                    fig_dir / f"violin_{gene}_by_celltype.png",
                    log1p=True,
                )

        micro_labels = niche_pack["micro_labels"]
        if len(expr_df) == len(micro_labels):
            nm = scales["micro"]["niche_names_map"]
            expr_df["niche_name"] = [
                nm.get(nid, f"Micro_N{nid:02d}") for nid in micro_labels
            ]
            micro_color_by_id = niche_pack.get("niche_colors", {})
            uniq_niche_names = sorted(set(expr_df["niche_name"]))
            default_colors = get_color_mapping(uniq_niche_names)
            niche_palette: dict[str, str] = {}
            for nid, nname in nm.items():
                nid_int = int(nid)
                niche_palette[nname] = micro_color_by_id.get(
                    nid_int, default_colors.get(nname, "#666666")
                )
            for nname in uniq_niche_names:
                if nname not in niche_palette:
                    niche_palette[nname] = default_colors.get(nname, "#666666")
            for gene in genes:
                if gene not in expr_df.columns:
                    continue
                _plot_violin(
                    expr_df, gene, "niche_name",
                    f"{platform} \u2014 {gene} by Micro Niche [log1p]",
                    fig_dir / f"violin_{gene}_by_niche.png",
                    log1p=True,
                    palette=niche_palette,
                )
            valid_genes = [g for g in genes if g in expr_df.columns]
            if valid_genes:
                _plot_dotplot(
                    expr_df, valid_genes, "niche_name",
                    f"{platform} \u2014 Genes by Micro Niche",
                    fig_dir / "dotplot_genes_by_niche.png",
                )

        if "celltype" in expr_df.columns:
            valid_genes = [g for g in genes if g in expr_df.columns]
            if valid_genes:
                _plot_dotplot(
                    expr_df, valid_genes, "celltype",
                    f"{platform} \u2014 Genes by Cell Type",
                    fig_dir / "dotplot_genes_by_celltype.png",
                )

        # ----------------------------------------------------------------
        # E) Per-sample spatial co-localization panels:
        #    target-gene expression × native cell type / niche type (all scales)
        # ----------------------------------------------------------------
        valid_genes = [g for g in genes if g in expr_df.columns]
        if valid_genes and "sample_id" in expr_df.columns and "sample_id" in deconv.columns:
            expr_sample_all = _to_text_series(expr_df["sample_id"]).fillna("NA").values
            deconv_sample_all = _to_text_series(deconv["sample_id"]).fillna("NA").values
            mapped_idx = _map_expr_to_deconv_indices(
                expr_coords=expr_df[["x", "y"]].values,
                expr_sample=expr_sample_all,
                deconv_coords=coords,
                deconv_sample=deconv_sample_all,
            )
            valid_map = mapped_idx >= 0
            if np.any(valid_map):
                expr_sub = expr_df.loc[valid_map].reset_index(drop=True)
                mapped_idx = mapped_idx[valid_map]
                expr_norm_all = _normalize_expr_matrix(
                    np.log1p(expr_sub[valid_genes].astype(float).values)
                )
                expr_coords_sub = expr_sub[["x", "y"]].values
                expr_sid_sub = _to_text_series(expr_sub["sample_id"]).fillna("NA").values

                native_col = (
                    "native_celltype" if "native_celltype" in deconv.columns
                    else ("native_tumor_epi_label" if "native_tumor_epi_label" in deconv.columns else None)
                )
                native_palette = None
                if native_col is not None:
                    native_all = _to_text_series(deconv[native_col]).fillna("NA").values
                    native_palette = get_color_mapping([v for v in sorted(set(native_all)) if v != "NA"])

                for sid in sorted(pd.unique(expr_sid_sub)):
                    sm = expr_sid_sub == sid
                    if sm.sum() < 20:
                        continue
                    s_coords = expr_coords_sub[sm]
                    s_expr_norm = expr_norm_all[sm]
                    s_map_idx = mapped_idx[sm]

                    if native_col is not None:
                        s_native = _to_text_series(deconv[native_col].iloc[s_map_idx]).fillna("NA").values
                        _generate_sample_coloc_plots(
                            sample_id=str(sid),
                            expr_coords=s_coords,
                            expr_values_norm=s_expr_norm,
                            genes=valid_genes,
                            group_labels=s_native,
                            group_name="native celltype",
                            out_dir=fig_dir / "coloc_native",
                            palette=native_palette,
                        )

                    for scale_name in SCALE_ORDER:
                        sd = scales[scale_name]
                        nmap = sd["niche_names_map"]
                        labels_scale = np.asarray(sd["labels"])
                        names_all = np.array([nmap.get(int(nid), f"N{int(nid)}") for nid in labels_scale], dtype=object)
                        s_scale = names_all[s_map_idx]
                        scale_palette = get_color_mapping(sorted(set(names_all)))
                        _generate_sample_coloc_plots(
                            sample_id=str(sid),
                            expr_coords=s_coords,
                            expr_values_norm=s_expr_norm,
                            genes=valid_genes,
                            group_labels=s_scale,
                            group_name=f"{scale_name} niche type",
                            out_dir=fig_dir / f"coloc_{scale_name}_niche",
                            palette=scale_palette,
                        )

    elif platform == "visium":
        print(f"    {platform}: gene expression not directly available (spot metadata only)")

    print(f"  Figures saved \u2192 {fig_dir}")


# =========================================================================
# Phase 4: Cross-sample consistency evaluation
# =========================================================================

def run_cross_sample_eval(
    platform: str, niche_pack: dict, deconv: pd.DataFrame,
) -> dict[str, dict]:
    """Evaluate niche prototype consistency across samples."""
    from src.evaluation.cross_sample_metrics import niche_prototype_matching

    fig_dir = FIG_BASE / platform
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = NICHE_BASE / platform
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = niche_pack["sample_ids"]
    unique_samples = pd.unique(sample_ids)
    if len(unique_samples) < 2:
        print(f"  {platform}: only {len(unique_samples)} sample(s), skipping cross-sample eval")
        return {}

    comp = niche_pack["comp_values"]
    deconv_cols = niche_pack["deconv_cols"]
    consistency_by_scale: dict[str, dict] = {}

    for scale in SCALE_ORDER:
        labels_s = niche_pack["all_labels"][scale]
        per_sample: dict[str, pd.DataFrame] = {}
        for sid in unique_samples:
            smask = sample_ids == sid
            sl = labels_s[smask]
            sc = comp[smask]
            nids = sorted(set(sl))
            if len(nids) < 2:
                continue
            proto = np.zeros((len(nids), comp.shape[1]))
            for row, nid in enumerate(nids):
                nmask = sl == nid
                if nmask.sum() > 0:
                    proto[row] = sc[nmask].mean(axis=0)
            per_sample[str(sid)] = pd.DataFrame(proto, columns=deconv_cols)

        if len(per_sample) < 2:
            continue

        result = niche_prototype_matching(per_sample)
        consistency_by_scale[scale] = result

        _plot_cross_sample_heatmap(
            result,
            f"{platform} \u2014 {scale.capitalize()} Cross-Sample Consistency",
            fig_dir / f"{scale}_cross_sample_heatmap.png",
        )

    if consistency_by_scale:
        _plot_consistency_bar(
            consistency_by_scale,
            f"{platform} \u2014 Cross-Sample Consistency Summary",
            fig_dir / "cross_sample_consistency_summary.png",
        )
        (out_dir / "cross_sample_consistency.json").write_text(
            json.dumps(
                {s: {"mean": v.get("mean_consistency", 0),
                     "q25": v.get("q25_consistency", 0),
                     "n_samples": v.get("n_samples", 0)}
                 for s, v in consistency_by_scale.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  {platform}: cross-sample eval done ({len(consistency_by_scale)} scales)")

    return consistency_by_scale


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Per-platform multi-scale niche analysis")
    parser.add_argument("--platform", choices=["cosmx", "visium", "visiumhd", "all"],
                        default="all")
    parser.add_argument("--genes", type=str, default=",".join(DEFAULT_GENES))
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=18)
    parser.add_argument("--max-spots", type=int, default=150000)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    genes = resolve_genes(args.genes)
    platforms = list(PLATFORM_MAP.keys()) if args.platform == "all" else [args.platform]
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("  HyperSCA Multi-Scale Co-localization Niche Analysis")
    print(f"  Platforms: {platforms}")
    print(f"  Genes: {genes}")
    print(f"  Scales: {SCALE_ORDER}")
    print("=" * 60)

    if not args.skip_audit:
        print("\n[Phase 1] Platform readiness audit")
        run_audit(platforms)
    else:
        print("\n[Phase 1] Skipped (--skip-audit)")

    print("\n[Phase 2] Multi-scale co-localization niche hierarchy")
    niche_packs: dict[str, tuple] = {}
    for plat in platforms:
        print(f"\n--- {plat} ---")
        deconv = read_platform_deconv(plat, max_spots=args.max_spots)
        if deconv.empty:
            print(f"  {plat}: no data available, skipping")
            continue
        print(f"  {plat}: {len(deconv)} spots/cells loaded")
        pack = build_multiscale_niche(plat, deconv, k_min=args.k_min, k_max=args.k_max)
        niche_packs[plat] = (pack, deconv)

    print("\n[Phase 3] Generating figures")
    for plat, (pack, deconv) in niche_packs.items():
        print(f"\n--- {plat} figures ---")
        generate_platform_figures(plat, pack, deconv, genes, max_spots=args.max_spots)

    print("\n[Phase 4] Cross-sample consistency evaluation")
    for plat, (pack, deconv) in niche_packs.items():
        print(f"\n--- {plat} cross-sample ---")
        run_cross_sample_eval(plat, pack, deconv)

    print("\n" + "=" * 60)
    print("[DONE] Multi-scale niche analysis complete")
    for plat in niche_packs:
        print(f"  {plat}: {NICHE_BASE / plat}")
        print(f"  {plat} figs: {FIG_BASE / plat}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
