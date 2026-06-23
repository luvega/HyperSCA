"""Recompute CRC ICB discovery artifacts from raw single-cell and spatial data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from src.discovery.from_scratch.crc_icb_inputs import (
    merge_crc_icb_metadata,
    read_barcodes,
    read_features,
    read_geo_cell_metadata,
    read_mtx_column_subset,
)


def _optional_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _select_crc_tumor_cells(
    metadata: pd.DataFrame,
    barcodes: list[str],
    *,
    max_cells: int | None,
    response_column: str,
    case_label: str,
    control_label: str,
    celltype_col: str,
    random_seed: int,
) -> tuple[pd.DataFrame, list[int]]:
    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    selected = metadata[metadata["barcode"].isin(barcode_to_idx)].copy()
    if "Tissue" in selected.columns:
        selected = selected[selected["Tissue"].astype(str).str.contains("Tumor", case=False, na=False)].copy()
    selected = selected[selected[response_column].astype(str).isin({case_label, control_label})].copy()
    selected = selected[selected[celltype_col].notna()].copy()
    selected = selected[selected[celltype_col].astype(str).str.len() > 0].copy()
    selected = selected[selected["patient_id"].notna()].copy()
    selected["_matrix_idx"] = selected["barcode"].map(barcode_to_idx).astype(int)
    eligible_cells = int(len(selected))

    if max_cells is not None and len(selected) > max_cells:
        rng = np.random.default_rng(random_seed)
        strata = [celltype_col, response_column]
        for col in ("MSI.MSS", "Treatment.Stage"):
            if col in selected.columns:
                strata.append(col)
        selected["_stratum"] = selected[strata].astype(str).agg("|".join, axis=1)
        per_group = max(50, max_cells // max(selected["_stratum"].nunique(), 1))
        chosen: list[int] = []
        for _, group in selected.groupby("_stratum", sort=True):
            take = min(len(group), per_group)
            chosen.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).astype(int).tolist())
        if len(chosen) > max_cells:
            chosen = rng.choice(np.array(chosen), size=max_cells, replace=False).astype(int).tolist()
        elif len(chosen) < max_cells:
            remaining = np.array(sorted(set(selected.index.astype(int)) - set(chosen)))
            extra = min(max_cells - len(chosen), len(remaining))
            if extra:
                chosen.extend(rng.choice(remaining, size=extra, replace=False).astype(int).tolist())
        selected = selected.loc[chosen].copy()

    selected = selected.sort_values("_matrix_idx").reset_index(drop=True)
    selected.attrs["eligible_cells_before_max"] = eligible_cells
    return selected, selected["_matrix_idx"].astype(int).tolist()


def _pseudobulk_counts(
    matrix: sparse.csr_matrix,
    obs: pd.DataFrame,
    *,
    celltype_col: str,
    response_column: str,
    covariates: tuple[str, ...],
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    group_cols = [celltype_col, "patient_id", "sample_id", response_column]
    grouped = obs.groupby(group_cols, dropna=False, sort=True).indices
    if not grouped:
        raise RuntimeError("No pseudobulk groups could be formed")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    group_meta_rows: list[dict] = []
    for group_id, (key, indices) in enumerate(grouped.items()):
        idx = np.asarray(indices, dtype=int)
        rows.append(np.full(len(idx), group_id, dtype=int))
        cols.append(idx)
        meta = dict(zip(group_cols, key))
        meta["n_cells"] = int(len(idx))
        first = obs.iloc[idx[0]]
        for covariate in covariates:
            if covariate in obs.columns:
                meta[covariate] = first[covariate]
        group_meta_rows.append(meta)

    membership = sparse.csr_matrix(
        (
            np.ones(sum(len(item) for item in cols), dtype=np.float32),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(len(group_meta_rows), matrix.shape[0]),
    )
    return (membership @ matrix).tocsr(), pd.DataFrame(group_meta_rows)


def _log_cpm(counts: sparse.csr_matrix) -> np.ndarray:
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    library[library <= 0] = 1.0
    normalized = counts.multiply((1_000_000.0 / library)[:, None])
    return np.log1p(normalized.toarray()).astype(np.float32)


def _make_unique_gene_names(values: pd.Series) -> np.ndarray:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for value in values.astype(str):
        base = value if value and value.lower() != "nan" else "UNKNOWN"
        count = seen.get(base, 0)
        if count == 0:
            unique.append(base)
        else:
            unique.append(f"{base}.{count}")
        seen[base] = count + 1
    return np.asarray(unique, dtype=object)


def _write_response_de(
    log_expr: np.ndarray,
    group_meta: pd.DataFrame,
    genes: np.ndarray,
    *,
    output_path: Path,
    celltype_col: str,
    response_column: str,
    case_label: str,
    control_label: str,
    min_pseudobulk_per_group: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for celltype in sorted(group_meta[celltype_col].astype(str).unique()):
        ct_mask = group_meta[celltype_col].astype(str).to_numpy() == celltype
        case_idx = np.where(ct_mask & (group_meta[response_column].astype(str).to_numpy() == case_label))[0]
        control_idx = np.where(ct_mask & (group_meta[response_column].astype(str).to_numpy() == control_label))[0]
        if len(case_idx) < min_pseudobulk_per_group or len(control_idx) < min_pseudobulk_per_group:
            continue

        case_expr = log_expr[case_idx, :]
        control_expr = log_expr[control_idx, :]
        mean_case = case_expr.mean(axis=0)
        mean_control = control_expr.mean(axis=0)
        _, p_values = ttest_ind(case_expr, control_expr, axis=0, equal_var=False, nan_policy="omit")
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        padj = multipletests(p_values, method="fdr_bh")[1]
        frame = pd.DataFrame(
            {
                "gene": genes,
                "celltype": celltype,
                "mean_case_log_cpm": mean_case,
                "mean_control_log_cpm": mean_control,
                "log2_fold_change": (mean_case - mean_control) / np.log(2.0),
                "p_value": p_values,
                "adjusted_p_value": padj,
                "n_case_pseudobulk": len(case_idx),
                "n_control_pseudobulk": len(control_idx),
            }
        )
        frames.append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        result = pd.DataFrame(
            columns=[
                "gene",
                "celltype",
                "mean_case_log_cpm",
                "mean_control_log_cpm",
                "log2_fold_change",
                "p_value",
                "adjusted_p_value",
                "n_case_pseudobulk",
                "n_control_pseudobulk",
            ]
        )
    else:
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["adjusted_p_value", "celltype", "gene"], kind="mergesort")
    result.to_csv(output_path, index=False)
    return result


def _empty_wilcox_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "gene",
            "celltype",
            "mean_case_log_cpm",
            "mean_control_log_cpm",
            "log2_fold_change",
            "lfc_icb",
            "p_value",
            "adjusted_p_value",
            "padj_icb",
            "detection_fraction",
            "n_case_cells_available",
            "n_control_cells_available",
            "n_case_cells_used",
            "n_control_cells_used",
            "n_case_donors",
            "n_control_donors",
            "cell_use_fraction_case",
            "cell_use_fraction_control",
            "effective_donor_count_case",
            "effective_donor_count_control",
            "max_donor_fraction_case",
            "max_donor_fraction_control",
            "donor_direction_consistency",
            "de_method",
            "evidence_tier",
            "sampling_policy",
            "gene_block_size",
            "is_de_significant",
        ]
    )


def _effective_donor_count(donors: pd.Series) -> tuple[float, float]:
    fractions = donors.astype(str).value_counts(normalize=True)
    if fractions.empty:
        return 0.0, 0.0
    effective = float(1.0 / np.square(fractions.to_numpy(dtype=float)).sum())
    return effective, float(fractions.max())


def _log_cpm_block(matrix: sparse.csr_matrix, cell_idx: np.ndarray, library: np.ndarray, start: int, stop: int) -> np.ndarray:
    block = matrix[cell_idx, start:stop].tocsr()
    scale = 1_000_000.0 / np.maximum(library[cell_idx], 1.0)
    dense = block.multiply(scale[:, None]).toarray().astype(np.float32, copy=False)
    return np.log1p(dense).astype(np.float32, copy=False)


def _mannwhitneyu_pvalues(case_expr: np.ndarray, control_expr: np.ndarray) -> np.ndarray:
    try:
        result = mannwhitneyu(case_expr, control_expr, axis=0, alternative="two-sided", method="asymptotic")
        return np.nan_to_num(result.pvalue, nan=1.0, posinf=1.0, neginf=1.0)
    except TypeError:
        p_values = [
            mannwhitneyu(case_expr[:, idx], control_expr[:, idx], alternative="two-sided").pvalue
            for idx in range(case_expr.shape[1])
        ]
        return np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)


def _donor_means(expr: np.ndarray, donors: pd.Series) -> np.ndarray:
    labels = donors.astype(str).to_numpy()
    rows = [expr[labels == donor].mean(axis=0) for donor in sorted(pd.unique(labels))]
    return np.vstack(rows) if rows else np.empty((0, expr.shape[1]), dtype=np.float32)


def _donor_direction_consistency(case_expr: np.ndarray, control_expr: np.ndarray, case_donors: pd.Series, control_donors: pd.Series, lfc: np.ndarray) -> np.ndarray:
    case_means = _donor_means(case_expr, case_donors)
    control_means = _donor_means(control_expr, control_donors)
    if case_means.size == 0 or control_means.size == 0:
        return np.zeros(case_expr.shape[1], dtype=float)
    pair_diff = case_means[:, None, :] - control_means[None, :, :]
    positive = np.mean(pair_diff > 0, axis=(0, 1))
    negative = np.mean(pair_diff < 0, axis=(0, 1))
    neutral = np.full_like(positive, 0.5, dtype=float)
    return np.where(lfc > 0, positive, np.where(lfc < 0, negative, neutral)).astype(float)


def _write_response_de_wilcox(
    matrix: sparse.csr_matrix,
    obs: pd.DataFrame,
    genes: np.ndarray,
    *,
    output_path: Path,
    celltype_col: str,
    response_column: str,
    case_label: str,
    control_label: str,
    min_cells_per_group: int,
    min_donors_per_group: int,
    min_detection_fraction: float,
    padj_threshold: float,
    lfc_threshold: float,
    donor_direction_consistency_threshold: float,
    min_effective_donor_count: float,
    gene_block_size: int,
    sampling_policy: str,
) -> pd.DataFrame:
    if sampling_policy != "all_available":
        raise ValueError(f"unsupported Wilcoxon sampling_policy: {sampling_policy}")
    if gene_block_size <= 0:
        raise ValueError("gene_block_size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if obs.empty:
        result = _empty_wilcox_frame()
        result.to_csv(output_path, index=False)
        return result

    library = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    library[library <= 0] = 1.0
    labels = obs[celltype_col].astype(str).to_numpy()
    response = obs[response_column].astype(str).to_numpy()

    frames: list[pd.DataFrame] = []
    for celltype in sorted(pd.unique(labels)):
        ct_mask = labels == str(celltype)
        case_idx = np.where(ct_mask & (response == case_label))[0]
        control_idx = np.where(ct_mask & (response == control_label))[0]
        case_donors = obs.iloc[case_idx]["patient_id"].astype(str)
        control_donors = obs.iloc[control_idx]["patient_id"].astype(str)
        n_case_donors = int(case_donors.nunique())
        n_control_donors = int(control_donors.nunique())
        if (
            len(case_idx) < min_cells_per_group
            or len(control_idx) < min_cells_per_group
            or n_case_donors < min_donors_per_group
            or n_control_donors < min_donors_per_group
        ):
            continue

        eff_case, max_frac_case = _effective_donor_count(case_donors)
        eff_control, max_frac_control = _effective_donor_count(control_donors)
        chunk_frames: list[pd.DataFrame] = []
        for start in range(0, matrix.shape[1], gene_block_size):
            stop = min(start + gene_block_size, matrix.shape[1])
            case_expr = _log_cpm_block(matrix, case_idx, library, start, stop)
            control_expr = _log_cpm_block(matrix, control_idx, library, start, stop)
            mean_case = case_expr.mean(axis=0)
            mean_control = control_expr.mean(axis=0)
            lfc = (mean_case - mean_control) / np.log(2.0)
            p_values = _mannwhitneyu_pvalues(case_expr, control_expr)
            detection_fraction = np.maximum((case_expr > 0).mean(axis=0), (control_expr > 0).mean(axis=0))
            donor_consistency = _donor_direction_consistency(case_expr, control_expr, case_donors, control_donors, lfc)
            chunk_frames.append(
                pd.DataFrame(
                    {
                        "gene": genes[start:stop],
                        "celltype": str(celltype),
                        "mean_case_log_cpm": mean_case,
                        "mean_control_log_cpm": mean_control,
                        "log2_fold_change": lfc,
                        "lfc_icb": lfc,
                        "p_value": p_values,
                        "detection_fraction": detection_fraction,
                        "n_case_cells_available": int(len(case_idx)),
                        "n_control_cells_available": int(len(control_idx)),
                        "n_case_cells_used": int(len(case_idx)),
                        "n_control_cells_used": int(len(control_idx)),
                        "n_case_donors": n_case_donors,
                        "n_control_donors": n_control_donors,
                        "cell_use_fraction_case": 1.0,
                        "cell_use_fraction_control": 1.0,
                        "effective_donor_count_case": eff_case,
                        "effective_donor_count_control": eff_control,
                        "max_donor_fraction_case": max_frac_case,
                        "max_donor_fraction_control": max_frac_control,
                        "donor_direction_consistency": donor_consistency,
                        "de_method": "cell_wilcox_balanced",
                        "evidence_tier": "expansion",
                        "sampling_policy": sampling_policy,
                        "gene_block_size": int(gene_block_size),
                    }
                )
            )
        if not chunk_frames:
            continue
        frame = pd.concat(chunk_frames, ignore_index=True)
        frame["adjusted_p_value"] = multipletests(frame["p_value"].to_numpy(dtype=float), method="fdr_bh")[1]
        frame["padj_icb"] = frame["adjusted_p_value"]
        frame["is_de_significant"] = (
            (frame["padj_icb"] <= padj_threshold)
            & (frame["lfc_icb"].abs() >= lfc_threshold)
            & (frame["detection_fraction"] >= min_detection_fraction)
            & (frame["donor_direction_consistency"] >= donor_direction_consistency_threshold)
            & (frame["effective_donor_count_case"] >= min_effective_donor_count)
            & (frame["effective_donor_count_control"] >= min_effective_donor_count)
        )
        frames.append(frame)

    if not frames:
        result = _empty_wilcox_frame()
    else:
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["adjusted_p_value", "celltype", "gene"], kind="mergesort")
    result.to_csv(output_path, index=False)
    return result


def _write_cluster_expression(
    log_expr: np.ndarray,
    group_meta: pd.DataFrame,
    genes: np.ndarray,
    *,
    output_path: Path,
    celltype_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    labels = group_meta[celltype_col].astype(str).to_numpy()
    for celltype in sorted(pd.unique(labels)):
        idx = np.where(labels == celltype)[0]
        values = log_expr[idx, :].mean(axis=0)
        row = {"celltype": celltype}
        row.update({gene: float(value) for gene, value in zip(genes, values)})
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def _derive_celltype_signatures(
    log_expr: np.ndarray,
    group_meta: pd.DataFrame,
    genes: np.ndarray,
    *,
    celltype_col: str,
    genes_per_celltype: int,
) -> dict[str, list[str]]:
    labels = group_meta[celltype_col].astype(str).to_numpy()
    signatures: dict[str, list[str]] = {}
    for celltype in sorted(pd.unique(labels)):
        ct_idx = np.where(labels == celltype)[0]
        other_idx = np.where(labels != celltype)[0]
        if len(ct_idx) == 0 or len(other_idx) == 0:
            continue
        delta = log_expr[ct_idx, :].mean(axis=0) - log_expr[other_idx, :].mean(axis=0)
        order = np.argsort(np.nan_to_num(delta, nan=-np.inf))[::-1]
        picked: list[str] = []
        for gene_idx in order:
            gene = str(genes[gene_idx])
            if gene and gene not in picked and delta[gene_idx] > 0:
                picked.append(gene)
            if len(picked) >= genes_per_celltype:
                break
        if picked:
            signatures[celltype] = picked
    return signatures


def _write_signatures(signatures: dict[str, list[str]], output_path: Path) -> pd.DataFrame:
    rows = [
        {"celltype": celltype, "rank": rank, "gene": gene}
        for celltype, genes in signatures.items()
        for rank, gene in enumerate(genes, start=1)
    ]
    frame = pd.DataFrame(rows, columns=["celltype", "rank", "gene"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _score_gem_file(
    gem_path: Path,
    signatures: dict[str, list[str]],
    *,
    max_spots: int | None,
    random_seed: int,
    chunksize: int,
) -> pd.DataFrame:
    signature_genes = sorted({gene for genes in signatures.values() for gene in genes})
    if not signature_genes:
        return pd.DataFrame()
    wanted = set(signature_genes)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(gem_path, sep="\t", compression="infer", chunksize=chunksize):
        cols = {col.lower(): col for col in chunk.columns}
        gene_col = cols.get("geneid", cols.get("gene", cols.get("genes")))
        x_col = cols.get("x")
        y_col = cols.get("y")
        count_col = cols.get("midcounts", cols.get("count", cols.get("counts")))
        if gene_col is None or x_col is None or y_col is None or count_col is None:
            raise ValueError(f"Unsupported GEM columns in {gem_path}: {list(chunk.columns)}")
        chunk = chunk[chunk[gene_col].astype(str).isin(wanted)]
        if chunk.empty:
            continue
        grouped = (
            chunk.groupby([x_col, y_col, gene_col], sort=False)[count_col]
            .sum()
            .reset_index()
            .rename(columns={x_col: "x", y_col: "y", gene_col: "gene", count_col: "count"})
        )
        chunks.append(grouped)
    if not chunks:
        return pd.DataFrame()

    collapsed = pd.concat(chunks, ignore_index=True)
    collapsed = collapsed.groupby(["x", "y", "gene"], sort=False)["count"].sum().reset_index()
    spot_gene = collapsed.pivot_table(index=["x", "y"], columns="gene", values="count", aggfunc="sum", fill_value=0.0)
    if max_spots is not None and len(spot_gene) > max_spots:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(np.arange(len(spot_gene)), size=max_spots, replace=False)
        spot_gene = spot_gene.iloc[np.sort(keep)]

    log_expr = np.log1p(spot_gene)
    scores = pd.DataFrame(index=spot_gene.index)
    for celltype, genes in signatures.items():
        present = [gene for gene in genes if gene in log_expr.columns]
        if not present:
            scores[celltype] = 0.0
        else:
            scores[celltype] = log_expr[present].mean(axis=1).astype(float)
    scores = scores.reset_index()
    scores.insert(0, "sample", gem_path.stem.replace(".gem", "").replace("STexpression_", ""))
    scores["spot_id"] = scores["sample"].astype(str) + ":" + scores["x"].astype(str) + "_" + scores["y"].astype(str)
    return scores


def _write_spatial_context(
    st_root: Path,
    signatures: dict[str, list[str]],
    *,
    output_path: Path,
    max_spots_per_sample: int | None,
    max_spatial_samples: int | None,
    random_seed: int,
    chunksize: int,
) -> pd.DataFrame:
    gem_files = sorted(Path(st_root).glob("STexpression_*.gem.gz"))
    if max_spatial_samples is not None:
        gem_files = gem_files[:max_spatial_samples]
    frames = [
        _score_gem_file(
            path,
            signatures,
            max_spots=max_spots_per_sample,
            random_seed=random_seed,
            chunksize=chunksize,
        )
        for path in gem_files
    ]
    frames = [frame for frame in frames if not frame.empty]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["sample", "x", "y", "spot_id", *signatures.keys()])
    result.to_csv(output_path, index=False)
    return result


def build_crc_icb_from_scratch_artifacts(
    *,
    icb_root: Path,
    st_root: Path,
    output_root: Path,
    max_cells: int | None = None,
    max_spots_per_sample: int | None = 20_000,
    max_spatial_samples: int | None = None,
    genes_per_celltype: int = 60,
    min_pseudobulk_per_group: int = 2,
    min_wilcox_cells_per_group: int = 50,
    min_wilcox_donors_per_group: int = 2,
    min_wilcox_detection_fraction: float = 0.05,
    wilcox_padj_threshold: float = 0.01,
    wilcox_lfc_threshold: float = 0.25,
    wilcox_donor_direction_consistency_threshold: float = 0.6,
    wilcox_min_effective_donor_count: float = 2.0,
    wilcox_gene_block_size: int = 128,
    wilcox_sampling_policy: str = "all_available",
    response_column: str = "binary_response",
    case_label: str = "non-pCR",
    control_label: str = "pCR",
    celltype_col: str = "MajorCellType",
    covariates: tuple[str, ...] = ("MSI.MSS", "Treatment.Stage"),
    random_seed: int = 42,
    chunksize: int = 1_000_000,
) -> dict[str, Path]:
    """Recompute DE, cluster-expression, and spatial-context inputs."""
    icb_root = Path(icb_root)
    st_root = Path(st_root)
    output_root = Path(output_root)
    input_dir = icb_root / "input"
    matrix_path = _optional_existing(input_dir / "matrix.mtx.gz")
    features_path = _optional_existing(input_dir / "features.tsv.gz")
    barcodes_path = _optional_existing(input_dir / "barcodes.tsv.gz")
    geo_metadata_path = _optional_existing(input_dir / "GSE236581_CRC-ICB_metadata.txt.gz")
    sample_metadata_path = _optional_existing(input_dir / "scCRC_ICB_sample_meta.csv")
    patient_metadata_path = _optional_existing(input_dir / "scCRC_ICB_patient meta.csv")

    geo = read_geo_cell_metadata(geo_metadata_path)
    sample = pd.read_csv(sample_metadata_path)
    patient = pd.read_csv(patient_metadata_path)
    metadata = merge_crc_icb_metadata(geo, sample, patient)
    barcodes = read_barcodes(barcodes_path)
    features = read_features(features_path)
    selected_obs, keep_indices = _select_crc_tumor_cells(
        metadata,
        barcodes,
        max_cells=max_cells,
        response_column=response_column,
        case_label=case_label,
        control_label=control_label,
        celltype_col=celltype_col,
        random_seed=random_seed,
    )
    matrix = read_mtx_column_subset(matrix_path, keep_indices, n_genes=len(features))
    counts, group_meta = _pseudobulk_counts(
        matrix,
        selected_obs,
        celltype_col=celltype_col,
        response_column=response_column,
        covariates=covariates,
    )
    log_expr = _log_cpm(counts)
    genes = _make_unique_gene_names(features["gene_symbols"])

    de_path = output_root / "de" / "response_de.csv"
    wilcox_de_path = output_root / "de" / "response_de_wilcox.csv"
    expression_path = output_root / "expression" / "cluster_expression.csv"
    signature_path = output_root / "expression" / "celltype_signatures.csv"
    spatial_path = output_root / "spatial" / "spatial_context.csv"
    pseudobulk_path = output_root / "expression" / "pseudobulk_metadata.csv"
    selected_meta_path = output_root / "metadata" / "selected_crc_tumor_cells.csv"
    provenance_path = output_root / "provenance.json"

    de = _write_response_de(
        log_expr,
        group_meta,
        genes,
        output_path=de_path,
        celltype_col=celltype_col,
        response_column=response_column,
        case_label=case_label,
        control_label=control_label,
        min_pseudobulk_per_group=min_pseudobulk_per_group,
    )
    wilcox_de = _write_response_de_wilcox(
        matrix,
        selected_obs,
        genes,
        output_path=wilcox_de_path,
        celltype_col=celltype_col,
        response_column=response_column,
        case_label=case_label,
        control_label=control_label,
        min_cells_per_group=min_wilcox_cells_per_group,
        min_donors_per_group=min_wilcox_donors_per_group,
        min_detection_fraction=min_wilcox_detection_fraction,
        padj_threshold=wilcox_padj_threshold,
        lfc_threshold=wilcox_lfc_threshold,
        donor_direction_consistency_threshold=wilcox_donor_direction_consistency_threshold,
        min_effective_donor_count=wilcox_min_effective_donor_count,
        gene_block_size=wilcox_gene_block_size,
        sampling_policy=wilcox_sampling_policy,
    )
    _write_cluster_expression(log_expr, group_meta, genes, output_path=expression_path, celltype_col=celltype_col)
    signatures = _derive_celltype_signatures(
        log_expr,
        group_meta,
        genes,
        celltype_col=celltype_col,
        genes_per_celltype=genes_per_celltype,
    )
    _write_signatures(signatures, signature_path)
    _write_spatial_context(
        st_root,
        signatures,
        output_path=spatial_path,
        max_spots_per_sample=max_spots_per_sample,
        max_spatial_samples=max_spatial_samples,
        random_seed=random_seed,
        chunksize=chunksize,
    )

    pseudobulk_path.parent.mkdir(parents=True, exist_ok=True)
    selected_meta_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    group_meta.to_csv(pseudobulk_path, index=False)
    selected_obs.drop(columns=[col for col in ("_matrix_idx", "_stratum") if col in selected_obs.columns]).to_csv(
        selected_meta_path,
        index=False,
    )
    eligible_cells = int(selected_obs.attrs.get("eligible_cells_before_max", len(selected_obs)))
    cell_use_fraction = float(len(selected_obs) / eligible_cells) if eligible_cells else 0.0
    provenance = {
        "mode": "from_scratch",
        "icb_root": str(icb_root),
        "st_root": str(st_root),
        "matrix_path": str(matrix_path),
        "features_path": str(features_path),
        "barcodes_path": str(barcodes_path),
        "geo_metadata_path": str(geo_metadata_path),
        "sample_metadata_path": str(sample_metadata_path),
        "patient_metadata_path": str(patient_metadata_path),
        "response_column": response_column,
        "case_label": case_label,
        "control_label": control_label,
        "celltype_col": celltype_col,
        "covariates": list(covariates),
        "max_cells": max_cells,
        "eligible_cells": eligible_cells,
        "selected_cells": int(len(selected_obs)),
        "cell_use_fraction": cell_use_fraction,
        "pseudobulk_groups": int(len(group_meta)),
        "genes": int(len(genes)),
        "de_rows": int(len(de)),
        "wilcox_de_rows": int(len(wilcox_de)),
        "wilcox_sampling_policy": wilcox_sampling_policy,
        "wilcox_gene_block_size": int(wilcox_gene_block_size),
        "wilcox_min_cells_per_group": int(min_wilcox_cells_per_group),
        "wilcox_min_donors_per_group": int(min_wilcox_donors_per_group),
        "wilcox_min_effective_donor_count": float(wilcox_min_effective_donor_count),
        "spatial_files_considered": len(sorted(Path(st_root).glob("STexpression_*.gem.gz"))),
        "max_spots_per_sample": max_spots_per_sample,
        "max_spatial_samples": max_spatial_samples,
        "genes_per_celltype_signature": genes_per_celltype,
        "random_seed": random_seed,
        "legacy_result_inputs_used": False,
        "manual_target_genes_used": False,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "de": de_path,
        "wilcox_de": wilcox_de_path,
        "expression": expression_path,
        "signatures": signature_path,
        "spatial": spatial_path,
        "pseudobulk_metadata": pseudobulk_path,
        "selected_metadata": selected_meta_path,
        "provenance": provenance_path,
    }
