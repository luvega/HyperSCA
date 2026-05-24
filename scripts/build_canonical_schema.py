"""
Build canonical schema tables from CRC multi-source data.

Integrates four possible sources:
  - scCRC_Neu  (G:)  — DESeq2 pseudo-bulk MSS vs MSI
  - scCRC_IFNG (F:)  — CosMx spatial + scRNA, per-patient MMR annotation
  - ST_CRC_MSS (G:)  — Stereo-seq / Visium spot deconvolution
  - scCRC_ICB  (G:)  — ICB response DEGs (optional, kept for backward compat)

Produces tables + alias dictionaries under results/integration/schema/:
  - sample_table.csv   (with mmr_group / ifn_ip / cohort columns)
  - entity_table.csv
  - feature_table.csv
  - measure_table.csv
  - celltype_alias.json

Usage:
    python scripts/build_canonical_schema.py
    python scripts/build_canonical_schema.py --no-icb   # skip ICB source
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "data"
OUT_DIR = DEFAULT_DATA_ROOT / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEU_DIR = Path(
    r"G:\scCRC_Neu\downstream_analyses_de_analysis"
    r"\0downstream_analyses_de_analysis\de_analysis"
    r"\de_analysis_tumor_mss_msi\deseq2_dgea"
)
IFNG_DIR = Path(r"F:\scCRC_IFNG")
ICB_DIR = Path(r"G:\scCRC_ICB\output")
ST_DIR = Path(r"G:\ST_CRC_MSS")

SEED_TARGETS: list[str] = []

IFNG_TARGETS: list[str] = []

ALL_FOCUS_GENES = sorted(set(SEED_TARGETS + IFNG_TARGETS))

# ── Cell type alias mapping ────────────────────────────────────────────────
CELLTYPE_ALIAS: dict[str, dict] = {
    "Fibroblast_S1":       {"canonical": "Fibroblast", "sub": "S1",        "major": "Stromal"},
    "Fibroblast_S2":       {"canonical": "Fibroblast", "sub": "S2",        "major": "Stromal"},
    "Fibroblast_S3":       {"canonical": "Fibroblast", "sub": "S3",        "major": "Stromal"},
    "Fibro":               {"canonical": "Fibroblast", "sub": None,        "major": "Stromal"},
    "Fibroblast":          {"canonical": "Fibroblast", "sub": None,        "major": "Stromal"},
    "Fibro_ADAMDEC1":      {"canonical": "Fibroblast", "sub": "ADAMDEC1",  "major": "Stromal"},
    "Fibro_CXCL8":         {"canonical": "Fibroblast", "sub": "CXCL8",     "major": "Stromal"},
    "Fibro_CXCL14":        {"canonical": "Fibroblast", "sub": "CXCL14",    "major": "Stromal"},
    "Fibro_GPM6B":         {"canonical": "Fibroblast", "sub": "GPM6B",     "major": "Stromal"},
    "Fibro_KCNN3":         {"canonical": "Fibroblast", "sub": "KCNN3",     "major": "Stromal"},
    "Fibro_MYH11":         {"canonical": "Fibroblast", "sub": "MYH11",     "major": "Stromal"},
    "Fibro_NOTCH3":        {"canonical": "Fibroblast", "sub": "NOTCH3",    "major": "Stromal"},
    "Fibro_PI16":          {"canonical": "Fibroblast", "sub": "PI16",      "major": "Stromal"},
    "Macrophage":          {"canonical": "Macrophage", "sub": None,        "major": "Myeloid"},
    "Macrophage_cycling":  {"canonical": "Macrophage", "sub": "cycling",   "major": "Myeloid"},
    "Mph":                 {"canonical": "Macrophage", "sub": None,        "major": "Myeloid"},
    "Mac_M1":              {"canonical": "Macrophage", "sub": "M1",        "major": "Myeloid"},
    "Mac_M2":              {"canonical": "Macrophage", "sub": "M2",        "major": "Myeloid"},
    "Mac_SPP1":            {"canonical": "Macrophage", "sub": "SPP1",      "major": "Myeloid"},
    "Monocyte_classical":  {"canonical": "Monocyte",   "sub": "classical", "major": "Myeloid"},
    "Monocyte_non_classical": {"canonical": "Monocyte", "sub": "non_classical", "major": "Myeloid"},
    "Monocyte_S100A8":     {"canonical": "Monocyte",   "sub": "S100A8",    "major": "Myeloid"},
    "Mye":                 {"canonical": "Myeloid",    "sub": None,        "major": "Myeloid"},
    "Neutrophil":          {"canonical": "Neutrophil", "sub": None,        "major": "Myeloid"},
    "Pericyte":            {"canonical": "Pericyte",   "sub": None,        "major": "Stromal"},
    "Stromal":             {"canonical": "Stromal",    "sub": None,        "major": "Stromal"},
    "Epi":                 {"canonical": "Epithelial", "sub": None,        "major": "Epithelial"},
    "Tumor":               {"canonical": "Tumor",      "sub": None,        "major": "Epithelial"},
    "Endothelial_arterial":   {"canonical": "Endothelial", "sub": "arterial",   "major": "Stromal"},
    "Endothelial_venous":     {"canonical": "Endothelial", "sub": "venous",     "major": "Stromal"},
    "Endothelial_lymphatic":  {"canonical": "Endothelial", "sub": "lymphatic",  "major": "Stromal"},
    "Endo":                   {"canonical": "Endothelial", "sub": None,         "major": "Stromal"},
    "cDC1":  {"canonical": "cDC1",  "sub": None, "major": "Myeloid"},
    "cDC2":  {"canonical": "cDC2",  "sub": None, "major": "Myeloid"},
    "DC3":   {"canonical": "DC3",   "sub": None, "major": "Myeloid"},
    "DC_mature": {"canonical": "DC_mature", "sub": None, "major": "Myeloid"},
    "pDC":   {"canonical": "pDC",   "sub": None, "major": "Myeloid"},
    "T_cell_CD4":        {"canonical": "CD4_T", "sub": None,      "major": "T_cell"},
    "T_cell_CD8":        {"canonical": "CD8_T", "sub": None,      "major": "T_cell"},
    "T_cell_CD4_cycling":{"canonical": "CD4_T", "sub": "cycling", "major": "T_cell"},
    "T_cell_CD8_cycling":{"canonical": "CD8_T", "sub": "cycling", "major": "T_cell"},
    "T_cell_regulatory":  {"canonical": "Treg",  "sub": None,     "major": "T_cell"},
    "NKT":   {"canonical": "NKT",   "sub": None, "major": "T_cell"},
    "ILC":   {"canonical": "ILC",   "sub": None, "major": "Innate_lymphoid"},
    "gamma_delta":  {"canonical": "gdT",  "sub": None, "major": "T_cell"},
    "NK_gdT":       {"canonical": "NK_gdT", "sub": None, "major": "T_cell"},
    "GC_B_cell":    {"canonical": "GC_B",  "sub": None, "major": "B_cell"},
    "B_act":        {"canonical": "B_act", "sub": None, "major": "B_cell"},
    "B_naive":      {"canonical": "B_naive", "sub": None, "major": "B_cell"},
    "Plasmablast":  {"canonical": "Plasmablast", "sub": None, "major": "B_cell"},
    "Plasma_IgA":   {"canonical": "Plasma", "sub": "IgA", "major": "B_cell"},
    "Plasma_IgG":   {"canonical": "Plasma", "sub": "IgG", "major": "B_cell"},
    "Plasma_IgM":   {"canonical": "Plasma", "sub": "IgM", "major": "B_cell"},
    "Mast_cell":    {"canonical": "Mast",   "sub": None, "major": "Myeloid"},
    "Mast":         {"canonical": "Mast",   "sub": None, "major": "Myeloid"},
    "Eosinophil":   {"canonical": "Eosinophil", "sub": None, "major": "Myeloid"},
    "Myeloid_progenitor": {"canonical": "Myeloid_progenitor", "sub": None, "major": "Myeloid"},
    "Schwann_cell": {"canonical": "Schwann", "sub": None, "major": "Stromal"},
    "Enteroendocrine": {"canonical": "Enteroendocrine", "sub": None, "major": "Epithelial"},
    "Tuft":         {"canonical": "Tuft",   "sub": None, "major": "Epithelial"},
    # IFNG project cell types (CosMx annotation level)
    "FibroEndoMuscle": {"canonical": "Fibroblast", "sub": "EndoMuscle", "major": "Stromal"},
    "T/NK":            {"canonical": "T_NK",       "sub": None,         "major": "T_cell"},
    "T_Other":         {"canonical": "T_Other",    "sub": None,         "major": "T_cell"},
    "Plasma/B":        {"canonical": "Plasma_B",   "sub": None,         "major": "B_cell"},
    "Myeloid":         {"canonical": "Myeloid",    "sub": None,         "major": "Myeloid"},
    "NK":              {"canonical": "NK",          "sub": None,         "major": "T_cell"},
}


# ── 1. Sample table ────────────────────────────────────────────────────────
def build_sample_table(*, include_icb: bool = True) -> pd.DataFrame:
    rows = []

    _sample_cols = [
        "dataset", "patient_id", "sample_id", "tissue_region",
        "mss_msi", "mmr_group", "ifn_ip", "cohort",
        "treatment", "modality", "data_level", "source_path",
    ]

    rows.append({
        "dataset": "scCRC_Neu",
        "patient_id": "multi",
        "sample_id": "scCRC_Neu_tumor_mss_msi",
        "tissue_region": "tumor",
        "mss_msi": "MSS_vs_MSI",
        "mmr_group": "",
        "ifn_ip": "",
        "cohort": "scCRC_Neu",
        "treatment": "naive",
        "modality": "scRNA-seq_pseudobulk",
        "data_level": "pseudobulk",
        "source_path": str(NEU_DIR),
    })

    # IFNG project: per-patient rows with MMR annotation
    ifng_clinical = IFNG_DIR / "results" / "tables" / "sample_clinical_mapping.csv"
    if ifng_clinical.exists():
        clin = pd.read_csv(ifng_clinical)
        for _, r in clin.iterrows():
            rows.append({
                "dataset": "scCRC_IFNG",
                "patient_id": str(r.get("sample", "")),
                "sample_id": f"scCRC_IFNG_{r.get('sample', '')}",
                "tissue_region": str(r.get("tissue", "Primary")),
                "mss_msi": str(r.get("mmr_group", "")),
                "mmr_group": str(r.get("mmr_group", "")),
                "ifn_ip": str(r.get("ifn_ip", "")),
                "cohort": str(r.get("cohort", "IFNG")),
                "treatment": str(r.get("ici_treatment", "naive")),
                "modality": "CosMx_scRNA",
                "data_level": "single_cell",
                "source_path": str(IFNG_DIR / "data" / "processed"),
            })
        print(f"  IFNG clinical: {len(clin)} patients loaded")
    else:
        print(f"  WARN: IFNG clinical mapping not found at {ifng_clinical}")

    if include_icb:
        icb_h5ad = DEFAULT_DATA_ROOT / "scRNA" / "scCRC_ICB" / "expression.h5ad"
        icb_has_full = icb_h5ad.exists()
        if icb_has_full:
            print(f"  ICB: full expression.h5ad detected → data_level=single_cell")
        rows.append({
            "dataset": "scCRC_ICB",
            "patient_id": "multi",
            "sample_id": "scCRC_ICB_MSS",
            "tissue_region": "tumor",
            "mss_msi": "MSS",
            "mmr_group": "MSS",
            "ifn_ip": "",
            "cohort": "scCRC_ICB",
            "treatment": "ICB_response",
            "modality": "scRNA-seq" if icb_has_full else "scRNA-seq_DEG",
            "data_level": "single_cell" if icb_has_full else "result",
            "source_path": str(icb_h5ad if icb_has_full else ICB_DIR),
        })

    for csv_f in sorted(ST_DIR.glob("STmetadata_*.csv")):
        patient_id = csv_f.stem.replace("STmetadata_", "").replace("_T_2", "_T2").replace("_T", "")
        rows.append({
            "dataset": "ST_CRC_MSS",
            "patient_id": patient_id,
            "sample_id": csv_f.stem,
            "tissue_region": "tumor",
            "mss_msi": "MSS",
            "mmr_group": "MSS",
            "ifn_ip": "",
            "cohort": "ST_CRC_MSS",
            "treatment": "mixed",
            "modality": "Visium_spot_metadata",
            "data_level": "spot",
            "source_path": str(csv_f),
        })

    df = pd.DataFrame(rows, columns=_sample_cols)
    df.to_csv(OUT_DIR / "sample_table.csv", index=False)
    print(f"[sample_table] {len(df)} rows → {OUT_DIR / 'sample_table.csv'}")
    return df


# ── 2. Feature table (focused on seed targets + context genes) ─────────────
def build_feature_table(*, include_icb: bool = True) -> pd.DataFrame:
    genes_seen: dict[str, set] = {}

    for tsv in NEU_DIR.glob("*-DESeq2_result.tsv"):
        try:
            df = pd.read_csv(tsv, sep="\t", usecols=["symbol"], nrows=0)
        except Exception:
            continue
        genes_seen.setdefault("scCRC_Neu", set())

    for tsv in NEU_DIR.glob("*-DESeq2_result.tsv"):
        for tgt in SEED_TARGETS:
            genes_seen.setdefault("scCRC_Neu", set()).add(tgt)

    # IFNG: target specificity & MMR-shared tables
    ifng_spec = IFNG_DIR / "results" / "tables" / "target_specificity.csv"
    if ifng_spec.exists():
        try:
            spec_df = pd.read_csv(ifng_spec)
            gene_col = "gene" if "gene" in spec_df.columns else spec_df.columns[0]
            for g in spec_df[gene_col].dropna().unique():
                genes_seen.setdefault("scCRC_IFNG", set()).add(g)
        except Exception:
            pass

    ifng_mmr = IFNG_DIR / "results" / "tables" / "targets_shared_specific_by_mmr.csv"
    if ifng_mmr.exists():
        try:
            mmr_df = pd.read_csv(ifng_mmr)
            gene_col = "gene" if "gene" in mmr_df.columns else mmr_df.columns[0]
            for g in mmr_df[gene_col].dropna().unique():
                genes_seen.setdefault("scCRC_IFNG", set()).add(g)
        except Exception:
            pass

    for tgt in IFNG_TARGETS:
        genes_seen.setdefault("scCRC_IFNG", set()).add(tgt)

    if include_icb:
        for csv_f in ICB_DIR.glob("DEGs_MSS_*.csv"):
            try:
                df = pd.read_csv(csv_f, nrows=5000)
                if "gene" in df.columns:
                    for g in df["gene"].unique():
                        genes_seen.setdefault("scCRC_ICB", set()).add(g)
            except Exception:
                continue

    rows = []
    all_genes: set[str] = set()
    for ds, gs in genes_seen.items():
        all_genes.update(gs)
    all_genes.update(ALL_FOCUS_GENES)

    for gene in sorted(all_genes):
        present_in = [ds for ds, gs in genes_seen.items() if gene in gs]
        rows.append({
            "gene_symbol": gene,
            "ensembl_id": "",
            "feature_type": "gene",
            "is_seed_target": gene in SEED_TARGETS,
            "is_ifng_target": gene in IFNG_TARGETS,
            "present_in_datasets": ";".join(present_in),
        })

    df = pd.DataFrame(
        rows,
        columns=[
            "gene_symbol",
            "ensembl_id",
            "feature_type",
            "is_seed_target",
            "is_ifng_target",
            "present_in_datasets",
        ],
    )
    df.to_csv(OUT_DIR / "feature_table.csv", index=False)
    print(f"[feature_table] {len(df)} rows → {OUT_DIR / 'feature_table.csv'}")
    return df


# ── 3. Entity table ────────────────────────────────────────────────────────
def build_entity_table(*, include_icb: bool = True) -> pd.DataFrame:
    rows = []

    _entity_base = {
        "x": np.nan, "y": np.nan, "cluster_label": "",
        "spatial_level1": "", "spatial_level3": "", "mmr_group": "",
    }

    for tsv in sorted(NEU_DIR.glob("*-DESeq2_result.tsv")):
        ct = tsv.stem.replace("-DESeq2_result", "")
        alias_info = CELLTYPE_ALIAS.get(ct, {})
        rows.append({
            **_entity_base,
            "entity_id": f"scCRC_Neu__{ct}",
            "dataset": "scCRC_Neu",
            "entity_type": "pseudobulk_celltype",
            "original_label": ct,
            "canonical_celltype": alias_info.get("canonical", ct),
            "sub_type": alias_info.get("sub", ""),
            "major_lineage": alias_info.get("major", ""),
        })

    # IFNG project: node_info cell types + niche annotation
    ifng_node = IFNG_DIR / "results" / "step2" / "node_info.json"
    if ifng_node.exists():
        import json as _json
        with open(ifng_node) as f:
            node_data = _json.load(f)
        for ct in node_data.get("node_labels", []):
            alias_info = CELLTYPE_ALIAS.get(ct, {})
            rows.append({
                **_entity_base,
                "entity_id": f"scCRC_IFNG__{ct}",
                "dataset": "scCRC_IFNG",
                "entity_type": "cluster_celltype",
                "original_label": ct,
                "canonical_celltype": alias_info.get("canonical", ct),
                "sub_type": alias_info.get("sub", ""),
                "major_lineage": alias_info.get("major", ""),
            })

    ifng_niche = IFNG_DIR / "results" / "tables" / "niche_shared_specific_by_mmr.csv"
    if ifng_niche.exists():
        try:
            ndf = pd.read_csv(ifng_niche)
            ct_col = "celltype" if "celltype" in ndf.columns else ndf.columns[0]
            mmr_col = "mmr_group" if "mmr_group" in ndf.columns else None
            for _, r in ndf.iterrows():
                ct = str(r[ct_col])
                alias_info = CELLTYPE_ALIAS.get(ct, {})
                eid = f"scCRC_IFNG_niche__{ct}"
                if any(row.get("entity_id") == eid for row in rows):
                    continue
                rows.append({
                    **_entity_base,
                    "entity_id": eid,
                    "dataset": "scCRC_IFNG",
                    "entity_type": "niche_celltype",
                    "original_label": ct,
                    "canonical_celltype": alias_info.get("canonical", ct),
                    "sub_type": alias_info.get("sub", ""),
                    "major_lineage": alias_info.get("major", ""),
                    "mmr_group": str(r[mmr_col]) if mmr_col else "",
                })
        except Exception:
            pass

    if include_icb:
        icb_h5ad_path = DEFAULT_DATA_ROOT / "scRNA" / "scCRC_ICB" / "expression.h5ad"
        icb_entity_added = False
        if icb_h5ad_path.exists():
            try:
                import anndata as _ad
                _icb = _ad.read_h5ad(str(icb_h5ad_path), backed="r")
                for lk in ["MajorCellType", "MidCellType", "celltype",
                            "cell_type", "Level1"]:
                    if lk in _icb.obs.columns:
                        vc = _icb.obs[lk].value_counts()
                        for ct_label, cnt in vc.items():
                            ct_str = str(ct_label)
                            alias_info = CELLTYPE_ALIAS.get(ct_str, {})
                            rows.append({
                                **_entity_base,
                                "entity_id": f"scCRC_ICB__{lk}_{ct_str}",
                                "dataset": "scCRC_ICB",
                                "entity_type": "h5ad_celltype",
                                "original_label": ct_str,
                                "canonical_celltype": alias_info.get("canonical", ct_str),
                                "sub_type": alias_info.get("sub", ""),
                                "major_lineage": alias_info.get("major", ""),
                                "cluster_label": lk,
                            })
                        icb_entity_added = True
                        print(f"  ICB entities: {len(vc)} types from {lk}")
                        break
                _icb.file.close()
            except Exception as exc:
                print(f"  WARN: ICB h5ad entity extraction failed: {exc}")

        if not icb_entity_added:
            icb_celltypes = [
                ("Epi", "Major"), ("Mye", "Major"), ("Stromal", "Major"),
                ("Tumor", "Mid"), ("Macrophage", "Mid"), ("Fibroblast", "Mid"),
            ]
            for ct, level in icb_celltypes:
                alias_info = CELLTYPE_ALIAS.get(ct, {})
                rows.append({
                    **_entity_base,
                    "entity_id": f"scCRC_ICB__{level}_{ct}",
                    "dataset": "scCRC_ICB",
                    "entity_type": "deg_celltype",
                    "original_label": ct,
                    "canonical_celltype": alias_info.get("canonical", ct),
                    "sub_type": alias_info.get("sub", ""),
                    "major_lineage": alias_info.get("major", ""),
                    "cluster_label": level,
                })

    st_files = sorted(ST_DIR.glob("STmetadata_*.csv"))
    for csv_f in st_files[:3]:
        df = pd.read_csv(csv_f, nrows=20)
        patient = csv_f.stem.replace("STmetadata_", "")
        for _, r in df.iterrows():
            rows.append({
                **_entity_base,
                "entity_id": f"ST__{patient}__{r.get('index', '')}",
                "dataset": "ST_CRC_MSS",
                "entity_type": "spot",
                "original_label": str(r.get("seurat_clusters", "")),
                "canonical_celltype": "",
                "sub_type": "",
                "major_lineage": "",
                "x": r.get("x", np.nan),
                "y": r.get("y", np.nan),
                "cluster_label": str(r.get("seurat_clusters", "")),
                "spatial_level1": r.get("level1", ""),
                "spatial_level3": r.get("level3", ""),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "entity_table.csv", index=False)
    print(f"[entity_table] {len(df)} rows → {OUT_DIR / 'entity_table.csv'}")
    return df


# ── 4. Measure table (seed-target-focused) ────────────────────────────────
def build_measure_table(*, include_icb: bool = True) -> pd.DataFrame:
    rows = []

    focus_genes = ALL_FOCUS_GENES

    for tgt in focus_genes:
        for tsv in sorted(NEU_DIR.glob("*-DESeq2_result.tsv")):
            ct = tsv.stem.replace("-DESeq2_result", "")
            try:
                df = pd.read_csv(tsv, sep="\t")
                hit = df[df["symbol"] == tgt]
                if hit.empty:
                    continue
                h = hit.iloc[0]
                eid = f"scCRC_Neu__{ct}"
                rows.append({
                    "entity_id": eid, "feature_id": tgt,
                    "value": h["log2FoldChange"], "value_type": "log2FC",
                    "analysis_level": "pseudobulk", "comparison": "MSS_vs_MSI",
                    "p_adj": h["padj"], "dataset": "scCRC_Neu",
                    "mmr_group": "",
                })
                rows.append({
                    "entity_id": eid, "feature_id": tgt,
                    "value": h["baseMean"], "value_type": "baseMean",
                    "analysis_level": "pseudobulk", "comparison": "MSS_vs_MSI",
                    "p_adj": np.nan, "dataset": "scCRC_Neu",
                    "mmr_group": "",
                })
            except Exception:
                continue

    # IFNG: MMR-stratified target measures
    ifng_mmr_tgt = IFNG_DIR / "results" / "tables" / "targets_shared_specific_by_mmr.csv"
    if ifng_mmr_tgt.exists():
        try:
            mdf = pd.read_csv(ifng_mmr_tgt)
            gene_col = "gene" if "gene" in mdf.columns else mdf.columns[0]
            for _, r in mdf.iterrows():
                gene = str(r[gene_col])
                mmr_g = str(r.get("mmr_group", ""))
                fc_val = r.get("log2FoldChange", r.get("avg_log2FC", np.nan))
                p_val = r.get("padj", r.get("p_val_adj", np.nan))
                ct_raw = str(r.get("celltype", "unknown"))
                eid = f"scCRC_IFNG__{ct_raw}"
                rows.append({
                    "entity_id": eid, "feature_id": gene,
                    "value": fc_val, "value_type": "log2FC",
                    "analysis_level": "single_cell", "comparison": f"MMR_{mmr_g}",
                    "p_adj": p_val, "dataset": "scCRC_IFNG",
                    "mmr_group": mmr_g,
                })
        except Exception as e:
            print(f"  WARN: IFNG MMR target measures failed: {e}")

    ifng_spec = IFNG_DIR / "results" / "tables" / "target_specificity.csv"
    if ifng_spec.exists():
        try:
            sdf = pd.read_csv(ifng_spec)
            gene_col = "gene" if "gene" in sdf.columns else sdf.columns[0]
            for _, r in sdf.iterrows():
                gene = str(r[gene_col])
                ct_raw = str(r.get("celltype", "unknown"))
                spec_val = r.get("specificity", r.get("log2FoldChange", np.nan))
                eid = f"scCRC_IFNG__{ct_raw}"
                rows.append({
                    "entity_id": eid, "feature_id": gene,
                    "value": spec_val, "value_type": "specificity",
                    "analysis_level": "single_cell", "comparison": "cell_specificity",
                    "p_adj": np.nan, "dataset": "scCRC_IFNG",
                    "mmr_group": "",
                })
        except Exception:
            pass

    if include_icb:
        icb_files = [
            ("DEGs_MSS_Major.csv", "MSS_characteristic"),
            ("DEGs_MSS_Mid.csv", "MSS_characteristic"),
            ("DEGs_MSS_response_Major.csv", "MSS_ICB_response"),
            ("DEGs_MSS_response_Mid.csv", "MSS_ICB_response"),
        ]
        for fname, comparison in icb_files:
            fpath = ICB_DIR / fname
            if not fpath.exists():
                continue
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            for tgt in focus_genes:
                hits = df[df["gene"] == tgt]
                for _, h in hits.iterrows():
                    ct = h.get("celltype", "unknown")
                    level = "Major" if "Major" in fname else "Mid"
                    eid = f"scCRC_ICB__{level}_{ct}"
                    rows.append({
                        "entity_id": eid, "feature_id": tgt,
                        "value": h["avg_log2FC"], "value_type": "avg_log2FC",
                        "analysis_level": "result", "comparison": comparison,
                        "p_adj": h.get("p_val_adj", np.nan), "dataset": "scCRC_ICB",
                        "mmr_group": "MSS",
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "measure_table.csv", index=False)
    print(f"[measure_table] {len(df)} rows → {OUT_DIR / 'measure_table.csv'}")
    return df


# ── Mapping rules doc ─────────────────────────────────────────────────────
def write_mapping_rules():
    doc = """# Canonical Schema Mapping Rules

## Data Sources
| Source      | Path                                          | Modality          |
|-------------|-----------------------------------------------|-------------------|
| scCRC_Neu   | G:\\scCRC_Neu\\...\\deseq2_dgea                | scRNA pseudo-bulk  |
| scCRC_IFNG  | F:\\scCRC_IFNG                                 | CosMx + scRNA      |
| scCRC_ICB   | G:\\scCRC_ICB\\output  (optional)              | scRNA DEG          |
| ST_CRC_MSS  | G:\\ST_CRC_MSS                                | Visium spot meta   |

## MMR/MSI Stratification
- `mmr_group` derived from scCRC_IFNG `sample_clinical_mapping.csv`
- Values: pMMR (= MSS), dMMR (= MSI-H), or blank if unavailable
- All ST_CRC_MSS patients assumed MSS

## Candidate Genes
- No manual seed or anchor genes are injected.
- Feature and measure tables use genes observed in source result files.
- Final prioritization is handled by the target_discovery pipeline.

## Cell-Type Alias Convention
- See celltype_alias.json for original → canonical mapping
- `major_lineage`: top-level grouping (Stromal, Myeloid, T_cell, B_cell, Epithelial, Innate_lymphoid)
"""
    (OUT_DIR / "mapping_rules.md").write_text(doc, encoding="utf-8")
    print(f"[mapping_rules] → {OUT_DIR / 'mapping_rules.md'}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    global OUT_DIR, NEU_DIR, IFNG_DIR, ICB_DIR, ST_DIR

    parser = argparse.ArgumentParser(description="Build canonical schema")
    parser.add_argument("--no-icb", action="store_true",
                        help="Skip scCRC_ICB source (use IFNG instead)")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--neu-root", type=str, default=str(NEU_DIR))
    parser.add_argument("--ifng-root", type=str, default=str(IFNG_DIR))
    parser.add_argument("--icb-root", type=str, default=str(ICB_DIR))
    parser.add_argument("--st-root", type=str, default=str(ST_DIR))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir) if args.output_dir else data_root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR = out_dir

    NEU_DIR = Path(args.neu_root)
    IFNG_DIR = Path(args.ifng_root)
    ICB_DIR = Path(args.icb_root)
    ST_DIR = Path(args.st_root)

    include_icb = not args.no_icb

    with open(OUT_DIR / "celltype_alias.json", "w", encoding="utf-8") as f:
        json.dump(CELLTYPE_ALIAS, f, indent=2, ensure_ascii=False)
    print(f"[alias] {len(CELLTYPE_ALIAS)} entries → {OUT_DIR / 'celltype_alias.json'}")

    build_sample_table(include_icb=include_icb)
    build_feature_table(include_icb=include_icb)
    build_entity_table(include_icb=include_icb)
    build_measure_table(include_icb=include_icb)
    write_mapping_rules()

    print("\n[DONE] All canonical schema tables written to:", OUT_DIR)


if __name__ == "__main__":
    main()
