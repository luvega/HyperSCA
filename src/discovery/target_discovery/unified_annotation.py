"""Unified cell annotation helpers for spatial mapping POCs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize


PREDICTION_COLUMNS = (
    "obs_id",
    "dataset_id",
    "x",
    "y",
    "method",
    "reference",
    "assay_type",
    "unified_level0",
    "unified_level1",
    "unified_level2",
    "confidence",
    "status",
    "source_label",
    "top2_label",
)
ABUNDANCE_METADATA_COLUMNS = ("spot_id", "sample_id", "x", "y", "level3")
UNIFIED_LEVEL1 = (
    "Tumor",
    "Intestinal_Epithelial",
    "T_cells",
    "B_cells",
    "Myeloid",
    "ILC",
    "Endothelial",
    "Fibroblast",
    "Smooth_Muscle",
    "Neuronal",
    "Unknown",
)


@dataclass(frozen=True)
class UnifiedLabel:
    source_label: str
    unified_level0: str
    unified_level1: str
    unified_level2: str


@dataclass(frozen=True)
class MethodStatus:
    dataset_id: str
    method: str
    status: str
    runnable: bool
    message: str = ""


@dataclass
class TransferResult:
    status: str
    predictions: pd.DataFrame | None = None
    abundance: pd.DataFrame | None = None
    qc: dict[str, Any] | None = None
    message: str = ""


def _clean_label(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return "Unknown"
    return text


def _norm(value: Any) -> str:
    return _clean_label(value).lower().replace("_", " ").replace(".", " ").replace("-", " ")


def _level0(level1: str) -> str:
    if level1 in {"Tumor", "Intestinal_Epithelial"}:
        return "tumor_epithelial"
    if level1 in {"T_cells", "B_cells", "Myeloid", "ILC"}:
        return "immune"
    if level1 in {"Endothelial", "Fibroblast", "Smooth_Muscle"}:
        return "stromal"
    if level1 == "Neuronal":
        return "neural_or_other"
    return "unknown"


def _from_text(label: str) -> str:
    text = _norm(label)
    if "qc filtered" in text or text == "unknown":
        return "Unknown"
    if "tumor" in text or "malignant" in text:
        return "Tumor"
    if any(token in text for token in ["t cell", "cd4", "cd8", "cytotoxic"]):
        return "T_cells"
    if any(token in text for token in ["b cell", "plasma", "naiveb", "memb"]):
        return "B_cells"
    if any(token in text for token in ["myeloid", "macrophage", "mono", "mast", "dendritic"]):
        return "Myeloid"
    if "ilc" in text:
        return "ILC"
    if "endothelial" in text or text == "endo":
        return "Endothelial"
    if any(token in text for token in ["fibro", "caf", "myofibro", "stromal"]):
        return "Fibroblast"
    if any(token in text for token in ["smooth muscle", "vsm", "sm stress"]):
        return "Smooth_Muscle"
    if any(token in text for token in ["neuronal", "neuron", "enteric glial", "glial"]):
        return "Neuronal"
    if any(token in text for token in ["epi", "epithelial", "enterocyte", "goblet", "tuft", "coloncyte"]):
        return "Intestinal_Epithelial"
    return "Unknown"


def map_label_to_unified(*, source_system: str, major_label: Any, fine_label: Any | None = None) -> UnifiedLabel:
    """Map scCRC_ICB, OSTA, RCTD, or transferred labels into one taxonomy."""
    major = _clean_label(major_label)
    fine = _clean_label(fine_label if fine_label is not None else major_label)
    source = fine if fine != "Unknown" else major
    system = str(source_system).lower()

    if system == "sccrc_icb":
        major_norm = _norm(major)
        if major_norm == "t":
            level1 = "T_cells"
        elif major_norm == "b":
            level1 = "B_cells"
        elif major_norm == "mye":
            level1 = "Myeloid"
        elif major_norm == "ilc":
            level1 = "ILC"
        elif major_norm == "epi":
            level1 = "Tumor" if "tumor" in _norm(fine) else "Intestinal_Epithelial"
        elif major_norm == "stromal":
            level1 = _from_text(fine)
            if level1 == "Unknown":
                level1 = "Fibroblast"
        else:
            level1 = _from_text(fine if fine != "Unknown" else major)
    else:
        level1 = _from_text(fine if fine != "Unknown" else major)

    return UnifiedLabel(
        source_label=source,
        unified_level0=_level0(level1),
        unified_level1=level1,
        unified_level2=fine,
    )


def build_unified_celltype_dictionary() -> pd.DataFrame:
    descriptions = {
        "Tumor": "Malignant epithelial tumor cells.",
        "Intestinal_Epithelial": "Non-malignant epithelial and differentiated intestinal cells.",
        "T_cells": "T lymphocytes including CD4, CD8, and cytotoxic states.",
        "B_cells": "B lineage cells including plasma cells.",
        "Myeloid": "Macrophage, monocyte, dendritic, and other myeloid states.",
        "ILC": "Innate lymphoid cells.",
        "Endothelial": "Vascular and lymphatic endothelial cells.",
        "Fibroblast": "Fibroblast, CAF, and myofibroblast-like stromal states.",
        "Smooth_Muscle": "Smooth muscle and vascular smooth muscle states.",
        "Neuronal": "Neuronal or enteric glial states.",
        "Unknown": "Unmapped, low-confidence, or filtered labels.",
    }
    return pd.DataFrame(
        {
            "unified_level0": [_level0(label) for label in UNIFIED_LEVEL1],
            "unified_level1": list(UNIFIED_LEVEL1),
            "description": [descriptions[label] for label in UNIFIED_LEVEL1],
        }
    )


def validate_prediction_table(table: pd.DataFrame) -> None:
    missing = [col for col in PREDICTION_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"prediction table missing columns: {missing}")
    pd.to_numeric(table["confidence"], errors="raise")


def validate_abundance_table(table: pd.DataFrame) -> None:
    missing = [col for col in ABUNDANCE_METADATA_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"abundance table missing metadata columns: {missing}")
    numeric_cols = [
        col
        for col in table.columns
        if col not in ABUNDANCE_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(table[col])
    ]
    if not numeric_cols:
        raise ValueError("abundance table has no numeric cell-type columns")


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cell2location_status_for_dataset(
    *,
    dataset_id: str,
    assay_type: str,
    device: str,
    gpu_required: bool,
    cuda_available: bool | None = None,
) -> MethodStatus:
    if str(assay_type) == "targeted_panel":
        return MethodStatus(
            dataset_id=dataset_id,
            method="cell2location",
            status="not_applicable:targeted_panel_cell_level",
            runnable=False,
            message="cell2location whole-transcriptome deconvolution is not used for targeted Xenium panels",
        )
    if str(assay_type) == "visiumhd_segmented_cell":
        return MethodStatus(
            dataset_id=dataset_id,
            method="cell2location",
            status="not_applicable:segmented_cell_level",
            runnable=False,
            message="cell2location spot/bin deconvolution is not used for VisiumHD segmented cell-level observations",
        )
    cuda = _torch_cuda_available() if cuda_available is None else bool(cuda_available)
    if gpu_required and str(device) == "cuda" and not cuda:
        return MethodStatus(dataset_id, "cell2location", "blocked:cuda_unavailable", False)
    if gpu_required and str(device) != "cuda":
        return MethodStatus(dataset_id, "cell2location", "blocked:cuda_required", False)
    return MethodStatus(dataset_id, "cell2location", "ready", True)


def add_unified_labels(adata, *, source_system: str, label_key: str, fine_label_key: str | None = None) -> None:
    if label_key not in adata.obs:
        raise ValueError(f"reference obs missing label_key={label_key!r}")
    fine_values = adata.obs[fine_label_key] if fine_label_key and fine_label_key in adata.obs else adata.obs[label_key]
    mapped = [
        map_label_to_unified(source_system=source_system, major_label=major, fine_label=fine)
        for major, fine in zip(adata.obs[label_key].tolist(), fine_values.tolist(), strict=False)
    ]
    adata.obs["unified_level0"] = [item.unified_level0 for item in mapped]
    adata.obs["unified_level1"] = [item.unified_level1 for item in mapped]
    adata.obs["unified_level2"] = [item.unified_level2 for item in mapped]
    adata.obs["cell2location_label"] = adata.obs["unified_level1"].astype(str)


def downsample_by_label(adata, *, label_key: str, max_cells_per_label: int | None, seed: int):
    if not max_cells_per_label or max_cells_per_label <= 0:
        return adata.copy()
    labels = adata.obs[label_key].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for label in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == label)
        if len(idx) > max_cells_per_label:
            idx = np.sort(rng.choice(idx, size=max_cells_per_label, replace=False))
        keep.extend(idx.tolist())
    return adata[np.array(sorted(keep), dtype=int)].copy()


def subset_obs(adata, *, max_obs: int | None, seed: int):
    if not max_obs or max_obs <= 0 or adata.n_obs <= max_obs:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_obs, replace=False))
    return adata[idx].copy()


def read_reference_from_spec(spec: dict[str, Any], *, max_cells_per_label: int | None, seed: int):
    import anndata as ad

    source = str(spec.get("source", "h5ad"))
    if source == "sccrc_icb_raw":
        from src.discovery.target_discovery.cell2location_context import build_icb_reference_anndata

        adata = build_icb_reference_anndata(
            spec["raw_icb_dir"],
            spec["metadata_path"],
            label_key=str(spec.get("label_key", "MajorCellType")),
            major_label_key=str(spec.get("major_label_key", spec.get("label_key", "MajorCellType"))),
        )
    elif source == "h5ad":
        adata = ad.read_h5ad(spec["h5ad_path"])
    else:
        raise ValueError(f"unsupported reference source: {source}")

    filter_column = spec.get("filter_column")
    filter_value = spec.get("filter_value")
    if filter_column and filter_column in adata.obs and filter_value is not None:
        adata = adata[adata.obs[str(filter_column)].astype(str).eq(str(filter_value))].copy()

    add_unified_labels(
        adata,
        source_system=str(spec.get("source_system", "osta")),
        label_key=str(spec.get("label_key", "Level1")),
        fine_label_key=spec.get("fine_label_key"),
    )
    sampled = downsample_by_label(
        adata,
        label_key="unified_level1",
        max_cells_per_label=max_cells_per_label,
        seed=seed,
    )
    return sampled


def read_query_from_spec(spec: dict[str, Any], *, max_query_cells: int | None, seed: int):
    import anndata as ad

    adata = ad.read_h5ad(spec["h5ad_path"])
    if "spatial" in adata.obsm and not {"x", "y"}.issubset(adata.obs.columns):
        spatial = np.asarray(adata.obsm["spatial"])
        if spatial.ndim == 2 and spatial.shape[1] >= 2:
            adata.obs["x"] = spatial[:, 0]
            adata.obs["y"] = spatial[:, 1]
    return subset_obs(adata, max_obs=max_query_cells, seed=seed)


def _shared_genes(reference, query, *, max_genes: int | None) -> list[str]:
    query_genes = set(map(str, query.var_names))
    genes = [str(gene) for gene in reference.var_names if str(gene) in query_genes]
    if max_genes and len(genes) > max_genes:
        genes = genes[:max_genes]
    return genes


def _dense_matrix(adata, genes: list[str]) -> np.ndarray:
    x = adata[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _scaled_reference_query(reference, query, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ref_x = np.log1p(np.maximum(_dense_matrix(reference, genes), 0.0))
    query_x = np.log1p(np.maximum(_dense_matrix(query, genes), 0.0))
    mean = ref_x.mean(axis=0, keepdims=True)
    std = ref_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (ref_x - mean) / std, (query_x - mean) / std


def _prediction_metadata(query, *, dataset_id: str) -> pd.DataFrame:
    obs = query.obs.copy()
    obs_id = query.obs_names.astype(str)
    x = pd.to_numeric(obs["x"], errors="coerce").fillna(0.0).to_numpy() if "x" in obs else np.zeros(query.n_obs)
    y = pd.to_numeric(obs["y"], errors="coerce").fillna(0.0).to_numpy() if "y" in obs else np.zeros(query.n_obs)
    return pd.DataFrame({"obs_id": obs_id, "dataset_id": dataset_id, "x": x, "y": y})


def predictions_to_abundance(predictions: pd.DataFrame, *, dataset_id: str) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "spot_id": predictions["obs_id"].astype(str),
            "sample_id": dataset_id,
            "x": predictions["x"],
            "y": predictions["y"],
            "level3": "unknown",
        }
    )
    for label in sorted(predictions["unified_level1"].dropna().astype(str).unique()):
        rows[label] = (predictions["unified_level1"].astype(str) == label).astype(float).to_numpy()
    validate_abundance_table(rows)
    return rows


def _is_doublet(value: Any) -> bool:
    return "doublet" in _norm(value)


def run_rctd_existing_annotation(
    query,
    *,
    dataset_id: str,
    reference_name: str,
    assay_type: str,
    label_key: str,
    secondary_label_key: str | None,
    class_key: str | None,
    source_system: str,
) -> TransferResult:
    """Standardize existing RCTD labels/weights from OSTA-style query metadata."""
    if str(assay_type) == "targeted_panel":
        return TransferResult(
            status="not_applicable:targeted_panel_cell_level",
            message="RCTD is not used for targeted Xenium cell-level panels.",
        )
    if label_key not in query.obs:
        return TransferResult(status="not_available:missing_rctd_label", message=f"query obs missing {label_key!r}")

    primary_raw = query.obs[label_key].tolist()
    secondary_raw = query.obs[secondary_label_key].tolist() if secondary_label_key and secondary_label_key in query.obs else primary_raw
    classes = query.obs[class_key].tolist() if class_key and class_key in query.obs else ["singlet"] * query.n_obs
    primary = [
        map_label_to_unified(source_system=source_system, major_label=value, fine_label=value)
        for value in primary_raw
    ]
    secondary = [
        map_label_to_unified(source_system=source_system, major_label=value, fine_label=value)
        for value in secondary_raw
    ]
    doublet = np.asarray([_is_doublet(value) for value in classes], dtype=bool)
    secondary_known = np.asarray([item.unified_level1 != "Unknown" for item in secondary], dtype=bool)
    primary_weight = np.where(doublet & secondary_known, 0.5, 1.0)
    secondary_weight = np.where(doublet & secondary_known, 0.5, 0.0)

    rows = _prediction_metadata(query, dataset_id=dataset_id)
    rows["method"] = "rctd"
    rows["reference"] = reference_name
    rows["assay_type"] = assay_type
    rows["unified_level1"] = [item.unified_level1 for item in primary]
    rows["unified_level0"] = [item.unified_level0 for item in primary]
    rows["unified_level2"] = [item.unified_level2 for item in primary]
    rows["confidence"] = primary_weight
    rows["status"] = ["doublet" if flag else "ok" for flag in doublet]
    rows["source_label"] = [_clean_label(value) for value in primary_raw]
    rows["top2_label"] = [item.unified_level1 for item in secondary]
    rows = rows[list(PREDICTION_COLUMNS)]
    validate_prediction_table(rows)

    abundance = pd.DataFrame(
        {
            "spot_id": rows["obs_id"].astype(str),
            "sample_id": dataset_id,
            "x": rows["x"],
            "y": rows["y"],
            "level3": [_clean_label(value) for value in classes],
        }
    )
    for label in sorted(set(rows["unified_level1"].astype(str)) | {item.unified_level1 for item in secondary}):
        abundance[label] = 0.0
    for i, (p_label, s_label) in enumerate(zip(rows["unified_level1"].astype(str), [item.unified_level1 for item in secondary], strict=False)):
        abundance.at[i, p_label] += float(primary_weight[i])
        if secondary_weight[i] > 0:
            abundance.at[i, s_label] += float(secondary_weight[i])
    validate_abundance_table(abundance)
    return TransferResult(status="ok", predictions=rows, abundance=abundance)


def _softmax_confidence(sim: np.ndarray) -> np.ndarray:
    if sim.shape[1] == 1:
        return np.ones(sim.shape[0], dtype=float)
    stable = sim - sim.max(axis=1, keepdims=True)
    exp = np.exp(stable)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def run_prototype_label_transfer(
    reference,
    query,
    *,
    dataset_id: str,
    reference_name: str,
    assay_type: str,
    method: str,
    max_genes: int | None,
) -> TransferResult:
    genes = _shared_genes(reference, query, max_genes=max_genes)
    if not genes:
        return TransferResult(status="failed:no_shared_genes", message="No shared genes between reference and query.")
    ref_x, query_x = _scaled_reference_query(reference, query, genes)
    labels = reference.obs["unified_level1"].astype(str).to_numpy()
    label_order = sorted(pd.unique(labels))
    centroids = np.vstack([ref_x[labels == label].mean(axis=0) for label in label_order])
    sim = normalize(query_x) @ normalize(centroids).T
    best_idx = sim.argmax(axis=1)
    if len(label_order) > 1:
        top2_idx = np.argsort(-sim, axis=1)[:, 1]
    else:
        top2_idx = best_idx
    confidence = _softmax_confidence(sim)

    rows = _prediction_metadata(query, dataset_id=dataset_id)
    chosen = [label_order[i] for i in best_idx]
    top2 = [label_order[i] for i in top2_idx]
    rows["method"] = method
    rows["reference"] = reference_name
    rows["assay_type"] = assay_type
    rows["unified_level1"] = chosen
    rows["unified_level0"] = [_level0(label) for label in chosen]
    rows["unified_level2"] = chosen
    rows["confidence"] = confidence
    rows["status"] = "ok"
    rows["source_label"] = chosen
    rows["top2_label"] = top2
    rows = rows[list(PREDICTION_COLUMNS)]
    validate_prediction_table(rows)
    abundance = predictions_to_abundance(rows, dataset_id=dataset_id)
    return TransferResult(status="ok", predictions=rows, abundance=abundance)


def run_hvae_label_transfer(
    reference,
    query,
    *,
    dataset_id: str,
    reference_name: str,
    assay_type: str,
    max_genes: int | None,
    epochs: int,
    device: str,
    seed: int,
) -> TransferResult:
    if epochs <= 0:
        return TransferResult(status="skipped:hvae_epochs_zero", message="hvae_epochs is zero.")
    if str(device) == "cuda" and not _torch_cuda_available():
        return TransferResult(status="blocked:cuda_unavailable", message="CUDA device was requested but is not visible.")
    genes = _shared_genes(reference, query, max_genes=max_genes)
    if not genes:
        return TransferResult(status="failed:no_shared_genes", message="No shared genes between reference and query.")
    try:
        from src.discovery.target_discovery.hyperbolic_spatial_benchmark import (
            HyperbolicSpatialBenchmarkConfig,
            _fit_hvae_embedding,
            _knn_edges_from_matrix,
            _scaled_expression,
        )

        ref_x = _dense_matrix(reference, genes)
        query_x = _dense_matrix(query, genes)
        combined = pd.DataFrame(
            np.vstack([ref_x, query_x]),
            index=[*reference.obs_names.astype(str), *query.obs_names.astype(str)],
            columns=genes,
        )
        edge_index, edge_weight = _knn_edges_from_matrix(_scaled_expression(combined), k=8)
        config = HyperbolicSpatialBenchmarkConfig(
            methods=("hvae_expression_knn",),
            embedding_dim=2,
            hvae_latent_dim=2,
            hvae_epochs=int(epochs),
            random_seed=int(seed),
            device=str(device),
        )
        embedding, status, valid = _fit_hvae_embedding(
            combined,
            edge_index_np=edge_index,
            edge_weight_np=edge_weight,
            config=config,
        )
        if not valid or embedding is None:
            return TransferResult(status=status)
        ref_latent = embedding[: reference.n_obs]
        query_latent = embedding[reference.n_obs :]
        labels = reference.obs["unified_level1"].astype(str).to_numpy()
        label_order = sorted(pd.unique(labels))
        centroids = np.vstack([ref_latent[labels == label].mean(axis=0) for label in label_order])
        sim = normalize(query_latent) @ normalize(centroids).T
        best_idx = sim.argmax(axis=1)
        top2_idx = np.argsort(-sim, axis=1)[:, 1] if len(label_order) > 1 else best_idx
        confidence = _softmax_confidence(sim)
        rows = _prediction_metadata(query, dataset_id=dataset_id)
        chosen = [label_order[i] for i in best_idx]
        rows["method"] = "hvae"
        rows["reference"] = reference_name
        rows["assay_type"] = assay_type
        rows["unified_level1"] = chosen
        rows["unified_level0"] = [_level0(label) for label in chosen]
        rows["unified_level2"] = chosen
        rows["confidence"] = confidence
        rows["status"] = "ok"
        rows["source_label"] = chosen
        rows["top2_label"] = [label_order[i] for i in top2_idx]
        rows = rows[list(PREDICTION_COLUMNS)]
        validate_prediction_table(rows)
        return TransferResult(status="ok", predictions=rows, abundance=predictions_to_abundance(rows, dataset_id=dataset_id))
    except Exception as exc:
        return TransferResult(status=f"failed:{type(exc).__name__}", message=str(exc))


def compute_prediction_qc(
    predictions: pd.DataFrame,
    query,
    *,
    truth_label_key: str | None,
    truth_source_system: str = "osta",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_id": str(predictions["dataset_id"].iloc[0]),
        "method": str(predictions["method"].iloc[0]),
        "reference": str(predictions["reference"].iloc[0]),
        "n_obs": int(len(predictions)),
        "mean_confidence": float(pd.to_numeric(predictions["confidence"], errors="coerce").mean()),
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "ari": np.nan,
        "nmi": np.nan,
    }
    if not truth_label_key or truth_label_key not in query.obs:
        return row
    truth = [
        map_label_to_unified(source_system=truth_source_system, major_label=value, fine_label=value).unified_level1
        for value in query.obs.loc[predictions["obs_id"].astype(str), truth_label_key].tolist()
    ]
    pred = predictions["unified_level1"].astype(str).tolist()
    row["accuracy"] = float(np.mean(np.asarray(truth, dtype=object) == np.asarray(pred, dtype=object)))
    row["macro_f1"] = float(f1_score(truth, pred, average="macro", zero_division=0))
    row["ari"] = float(adjusted_rand_score(truth, pred))
    row["nmi"] = float(normalized_mutual_info_score(truth, pred))
    return row


def standardize_cell2location_abundance(table: pd.DataFrame, *, dataset_id: str, source_system: str) -> pd.DataFrame:
    meta = pd.DataFrame()
    meta["spot_id"] = table["spot_id"].astype(str) if "spot_id" in table else table.index.astype(str)
    meta["sample_id"] = table["sample_id"].astype(str) if "sample_id" in table else dataset_id
    meta["x"] = pd.to_numeric(table["x"], errors="coerce").fillna(0.0) if "x" in table else 0.0
    meta["y"] = pd.to_numeric(table["y"], errors="coerce").fillna(0.0) if "y" in table else 0.0
    meta["level3"] = table["level3"].astype(str) if "level3" in table else "unknown"
    out = meta.copy()
    for col in table.columns:
        if col in ABUNDANCE_METADATA_COLUMNS or not pd.api.types.is_numeric_dtype(table[col]):
            continue
        label = map_label_to_unified(source_system=source_system, major_label=col, fine_label=col).unified_level1
        out[label] = out.get(label, 0.0) + pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    validate_abundance_table(out)
    return out


def write_csv_gz(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, compression="infer")
