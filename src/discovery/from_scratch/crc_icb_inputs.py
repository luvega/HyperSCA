"""Build CRC ICB from-scratch inputs from raw 10x and clinical metadata."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _norm_col(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip()).strip("_").lower()


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    lookup = {_norm_col(col): col for col in frame.columns}
    for candidate in candidates:
        key = _norm_col(candidate)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"missing required column; tried: {', '.join(candidates)}")


def _optional_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    try:
        return _column(frame, *candidates)
    except KeyError:
        return None


def merge_crc_icb_metadata(
    geo_cell_metadata: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    patient_metadata: pd.DataFrame,
    *,
    keep_cancer_types: Iterable[str] = ("CRC",),
) -> pd.DataFrame:
    """Merge cell, sample, and patient metadata for the CRC ICB cohort."""
    geo = geo_cell_metadata.copy()
    sample = sample_metadata.copy()
    patient = patient_metadata.copy()

    barcode_col = _column(geo, "barcode", "barcode_full", "orig.ident")
    sample_col = _column(geo, "sample_id", "Ident", "orig.ident")
    geo["barcode"] = geo[barcode_col].astype(str)
    geo["sample_id"] = geo[sample_col].astype(str)

    sample_id_col = _column(sample, "Sample ID", "sample_id")
    sample_patient_col = _column(sample, "Patient ID", "patient_id")
    sample = sample.rename(columns={sample_id_col: "sample_id", sample_patient_col: "patient_id"})
    sample_keep = ["sample_id", "patient_id"]
    for wanted, aliases in {
        "Treatment.Stage": ("Treatment Stage", "Treatment.Stage", "Treatment_Status"),
        "Treatment.point": ("Treatment point", "Treatment.point"),
        "Biopsy.Site": ("Biopsy Site", "Biopsy.Site"),
        "Sampling.Stage": ("Sampling Stage", "Sampling.Stage"),
    }.items():
        col = _optional_column(sample, *aliases)
        if col:
            sample = sample.rename(columns={col: wanted})
            sample_keep.append(wanted)
    sample = sample[sample_keep].drop_duplicates("sample_id")

    patient_id_col = _column(patient, "Patient ID", "patient_id")
    cancer_col = _column(patient, "Cancer Type", "Cancer.Type", "cancer_type")
    response_col = _column(patient, "Response")
    msi_col = _optional_column(patient, "MSI/MSS", "MSI.MSS", "mss_msi")
    patient = patient.rename(
        columns={
            patient_id_col: "patient_id",
            cancer_col: "Cancer.Type",
            response_col: "Response",
        }
    )
    patient_keep = ["patient_id", "Cancer.Type", "Response"]
    if msi_col:
        patient = patient.rename(columns={msi_col: "MSI.MSS"})
        patient_keep.append("MSI.MSS")
    for wanted, aliases in {
        "TRG.status": ("TRG status", "TRG.status"),
        "Tumor.Regression.Ratio": ("Tumor Regression Ratio", "Tumor.Regression.Ratio"),
        "Treatment.Regimen": ("Treatment Regimen", "Treatment.Regimen"),
    }.items():
        col = _optional_column(patient, *aliases)
        if col:
            patient = patient.rename(columns={col: wanted})
            patient_keep.append(wanted)
    patient = patient[patient_keep].drop_duplicates("patient_id")

    merged = geo.merge(sample, on="sample_id", how="left").merge(patient, on="patient_id", how="left")
    merged = merged[merged["Cancer.Type"].isin(set(keep_cancer_types))].copy()
    merged["binary_response"] = np.where(merged["Response"].astype(str).eq("CR"), "pCR", "non-pCR")
    return merged.reset_index(drop=True)


def read_barcodes(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_features(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    features = pd.DataFrame(rows)
    gene_ids = features.iloc[:, 0].astype(str)
    gene_symbols = features.iloc[:, 1].astype(str) if features.shape[1] > 1 else gene_ids
    return pd.DataFrame({"gene_ids": gene_ids.values, "gene_symbols": gene_symbols.values})


def read_geo_cell_metadata(path: Path) -> pd.DataFrame:
    """Read GEO cell metadata while preserving the barcode row names."""
    frame = pd.read_csv(path, sep=" ", quotechar='"', engine="python", index_col=0)
    frame = frame.reset_index(names="barcode")
    return frame


def stratified_barcode_indices(
    metadata: pd.DataFrame,
    barcodes: list[str],
    *,
    max_cells: int | None,
    strata_columns: tuple[str, ...] = ("MajorCellType", "binary_response", "MSI.MSS"),
    random_seed: int = 42,
) -> list[int]:
    """Return barcode indices, optionally stratified by biological labels."""
    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    present = metadata[metadata["barcode"].isin(barcode_to_idx)].copy()
    if max_cells is None or len(present) <= max_cells:
        return sorted(barcode_to_idx[barcode] for barcode in present["barcode"])

    rng = np.random.default_rng(random_seed)
    strata = [col for col in strata_columns if col in present.columns]
    if not strata:
        chosen = rng.choice(present["barcode"].to_numpy(), size=max_cells, replace=False)
        return sorted(barcode_to_idx[str(barcode)] for barcode in chosen)

    present["_stratum"] = present[strata].astype(str).agg("|".join, axis=1)
    per_group = max(20, max_cells // max(present["_stratum"].nunique(), 1))
    selected: list[str] = []
    for _, group in present.groupby("_stratum"):
        take = min(len(group), per_group)
        selected.extend(rng.choice(group["barcode"].to_numpy(), size=take, replace=False).astype(str).tolist())
    if len(selected) > max_cells:
        selected = rng.choice(np.array(selected), size=max_cells, replace=False).astype(str).tolist()
    elif len(selected) < max_cells:
        remaining = np.array(sorted(set(present["barcode"].astype(str)) - set(selected)))
        extra = min(max_cells - len(selected), len(remaining))
        if extra:
            selected.extend(rng.choice(remaining, size=extra, replace=False).astype(str).tolist())
    return sorted(barcode_to_idx[barcode] for barcode in selected)


def read_mtx_column_subset(matrix_path: Path, keep_indices: list[int], n_genes: int):
    """Stream a 10x Matrix Market file and materialize only selected cells."""
    from scipy.sparse import coo_matrix

    keep = set(keep_indices)
    remap = {old: new for new, old in enumerate(sorted(keep))}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    with gzip.open(matrix_path, "rt") as handle:
        header_consumed = False
        for line in handle:
            if line.startswith("%"):
                continue
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            if not header_consumed:
                header_consumed = True
                continue
            row_idx = int(parts[0]) - 1
            col_idx = int(parts[1]) - 1
            if col_idx not in keep:
                continue
            rows.append(row_idx)
            cols.append(remap[col_idx])
            data.append(float(parts[2]))
    mat = coo_matrix((data, (rows, cols)), shape=(n_genes, len(keep_indices)), dtype=np.float32)
    return mat.tocsr().T


def build_crc_icb_h5ad(
    *,
    matrix_path: Path,
    features_path: Path,
    barcodes_path: Path,
    geo_metadata_path: Path,
    sample_metadata_path: Path,
    patient_metadata_path: Path,
    output_h5ad: Path,
    output_metadata_csv: Path,
    provenance_path: Path,
    max_cells: int | None = None,
    random_seed: int = 42,
) -> Path:
    """Build a CRC-only AnnData object from raw 10x inputs."""
    import anndata as ad

    geo = read_geo_cell_metadata(geo_metadata_path)
    sample = pd.read_csv(sample_metadata_path)
    patient = pd.read_csv(patient_metadata_path)
    metadata = merge_crc_icb_metadata(geo, sample, patient)
    barcodes = read_barcodes(barcodes_path)
    features = read_features(features_path)
    keep_indices = stratified_barcode_indices(metadata, barcodes, max_cells=max_cells, random_seed=random_seed)
    kept_barcodes = [barcodes[idx] for idx in keep_indices]
    matrix = read_mtx_column_subset(matrix_path, keep_indices, n_genes=len(features))

    obs = metadata.set_index("barcode").reindex(kept_barcodes)
    obs.index = kept_barcodes
    var = features.copy()
    var.index = var["gene_symbols"].astype(str)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    metadata.to_csv(output_metadata_csv, index=False)
    provenance = {
        "matrix_path": str(matrix_path),
        "features_path": str(features_path),
        "barcodes_path": str(barcodes_path),
        "geo_metadata_path": str(geo_metadata_path),
        "sample_metadata_path": str(sample_metadata_path),
        "patient_metadata_path": str(patient_metadata_path),
        "output_h5ad": str(output_h5ad),
        "n_cells_metadata_crc": int(len(metadata)),
        "n_cells_written": int(adata.n_obs),
        "cell_use_fraction": float(adata.n_obs / len(metadata)) if len(metadata) else 0.0,
        "n_genes": int(adata.n_vars),
        "max_cells": max_cells,
        "random_seed": random_seed,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output_h5ad
