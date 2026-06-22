"""cell2location-oriented spatial context inputs and outputs."""
from __future__ import annotations

import gzip
import json
import shlex
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


METADATA_COLUMNS = {"spot_id", "sample_id", "x", "y", "level3", "region", "array_row", "array_col"}


def _read_text_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _first_existing(directory: Path, names: Iterable[str]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of these files exist in {directory}: {', '.join(names)}")


def _read_10x_features(path: Path) -> pd.DataFrame:
    rows = [line.split("\t") for line in _read_text_lines(path) if line]
    if not rows:
        raise ValueError(f"empty features file: {path}")
    max_len = max(len(row) for row in rows)
    padded = [row + [""] * (max_len - len(row)) for row in rows]
    columns = ["gene_id", "gene_name", "feature_type"][:max_len]
    return pd.DataFrame(padded, columns=columns)


def _read_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"} or path.name.endswith(".txt.gz"):
        lines = [line for line in _read_text_lines(path) if line.strip()]
        header = shlex.split(lines[0])
        rows = [shlex.split(line) for line in lines[1:]]
        if rows and len(rows[0]) == len(header) + 1:
            header = ["barcode", *header]
        frame = pd.DataFrame(rows, columns=header)
    else:
        frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(col).strip().strip('"') for col in frame.columns]
    for col in frame.select_dtypes(include="object").columns:
        frame[col] = frame[col].astype(str).str.strip().str.strip('"')
    return frame


def build_icb_reference_anndata(
    raw_icb_dir: str | Path,
    metadata_path: str | Path,
    *,
    label_key: str = "MidCellType",
    major_label_key: str = "MajorCellType",
):
    """Build a raw-count AnnData reference from 10x files and ICB metadata.

    This function intentionally consumes raw 10x files plus metadata only. It
    does not inspect historical DEG/output directories, keeping target
    nomination independent from old result tables.
    """
    import anndata as ad

    raw_icb_dir = Path(raw_icb_dir)
    matrix_path = _first_existing(raw_icb_dir, ["matrix.mtx.gz", "matrix.mtx"])
    features_path = _first_existing(raw_icb_dir, ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"])
    barcodes_path = _first_existing(raw_icb_dir, ["barcodes.tsv.gz", "barcodes.tsv"])

    matrix = mmread(matrix_path).tocsr().T
    features = _read_10x_features(features_path)
    barcodes = [line.strip() for line in _read_text_lines(barcodes_path) if line.strip()]
    if matrix.shape[0] != len(barcodes):
        raise ValueError(f"matrix cell count {matrix.shape[0]} != barcode count {len(barcodes)}")
    if matrix.shape[1] != len(features):
        raise ValueError(f"matrix gene count {matrix.shape[1]} != feature count {len(features)}")

    meta = _read_metadata(Path(metadata_path))
    barcode_col = next((col for col in ["barcode", "orig.ident", "cell", "cell_id"] if col in meta.columns), meta.columns[0])
    meta = meta.copy()
    meta[barcode_col] = meta[barcode_col].astype(str)
    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode")).join(meta.set_index(barcode_col), how="left")
    if obs[label_key].isna().all() if label_key in obs else False:
        suffix_meta = meta.copy()
        suffix_meta["_barcode_suffix"] = suffix_meta[barcode_col].astype(str).str.split("_").str[-1]
        suffix_obs = pd.DataFrame(index=pd.Index([bc.split("_")[-1] for bc in barcodes], name="_barcode_suffix"))
        obs = suffix_obs.join(suffix_meta.set_index("_barcode_suffix"), how="left")
        obs.index = pd.Index(barcodes, name="barcode")
    if label_key not in obs:
        raise ValueError(f"metadata missing label_key={label_key!r}")
    obs["cell2location_label"] = obs[label_key].astype(str).fillna("Unknown")
    obs["cell2location_major"] = obs[major_label_key].astype(str).fillna("Unknown") if major_label_key in obs else obs["cell2location_label"]

    var = pd.DataFrame(index=features["gene_name"].astype(str).values)
    var["gene_id"] = features["gene_id"].astype(str).values
    var["feature_type"] = features["feature_type"].astype(str).values if "feature_type" in features else "Gene Expression"
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["cell2location_reference"] = {
        "label_key": label_key,
        "major_label_key": major_label_key,
        "source": "raw_10x_plus_metadata",
    }
    return adata


def downsample_reference_by_label(
    adata,
    *,
    max_cells_per_label: int | None,
    seed: int = 0,
    label_col: str = "cell2location_label",
):
    """Return a seeded per-label subset for feasible reference model training."""
    if max_cells_per_label is None or int(max_cells_per_label) <= 0:
        sampled = adata.copy()
        sampled.uns["cell2location_reference_sampling"] = {
            "enabled": False,
            "max_cells_per_label": None,
            "seed": int(seed),
            "n_before": int(adata.n_obs),
            "n_after": int(sampled.n_obs),
        }
        return sampled
    if label_col not in adata.obs:
        raise ValueError(f"AnnData obs missing label_col={label_col!r}")

    labels = adata.obs[label_col].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_label: dict[str, dict[str, int]] = {}
    for label in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == label)
        before = int(len(idx))
        if before > int(max_cells_per_label):
            idx = np.sort(rng.choice(idx, size=int(max_cells_per_label), replace=False))
        selected.extend(idx.tolist())
        per_label[str(label)] = {"before": before, "after": int(len(idx))}

    sampled = adata[np.array(sorted(selected), dtype=int)].copy()
    sampled.uns["cell2location_reference_sampling"] = {
        "enabled": True,
        "max_cells_per_label": int(max_cells_per_label),
        "seed": int(seed),
        "label_col": label_col,
        "n_before": int(adata.n_obs),
        "n_after": int(sampled.n_obs),
        "per_label": per_label,
    }
    return sampled


def coerce_nonnegative_integer_counts(matrix):
    """Coerce a count-like matrix to non-negative integer pseudo-counts with audit metadata."""
    is_sparse = sparse.issparse(matrix)
    values = matrix.data.astype(float, copy=True) if is_sparse else np.asarray(matrix, dtype=float).ravel().copy()
    finite = values[np.isfinite(values)]
    if finite.size:
        noninteger = np.abs(finite - np.rint(finite)) > 1e-6
        negative = finite < 0
        audit = {
            "transform": "round_nonnegative",
            "noninteger_fraction": float(noninteger.mean()),
            "negative_value_count": int(negative.sum()),
            "min_before": float(finite.min()),
            "max_before": float(finite.max()),
            "sum_before": float(np.clip(finite, 0, None).sum()),
        }
    else:
        audit = {
            "transform": "round_nonnegative",
            "noninteger_fraction": 0.0,
            "negative_value_count": 0,
            "min_before": 0.0,
            "max_before": 0.0,
            "sum_before": 0.0,
        }
    rounded = np.rint(np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0, None))
    audit["sum_after"] = float(rounded.sum())
    if is_sparse:
        coerced = matrix.copy().tocsr()
        coerced.data = rounded.astype(np.float32, copy=False)
        coerced.eliminate_zeros()
        return coerced, audit
    return rounded.reshape(np.asarray(matrix).shape).astype(np.float32, copy=False), audit


def spatial_gem_to_anndata(gem_path: str | Path, metadata_path: str | Path, *, sample_id: str | None = None):
    """Convert one ST GEM count file plus metadata to raw-count spatial AnnData."""
    import anndata as ad

    gem_path = Path(gem_path)
    metadata_path = Path(metadata_path)
    sample_id = sample_id or gem_path.name.replace("STexpression_", "").replace(".gem.gz", "").replace(".gem", "")
    gem = pd.read_csv(gem_path, sep="\t", compression="infer", low_memory=False)
    required = {"geneID", "x", "y", "MIDCounts"}
    missing = required - set(gem.columns)
    if missing:
        raise ValueError(f"GEM missing columns: {sorted(missing)}")
    gem = gem[["geneID", "x", "y", "MIDCounts"]].copy()
    gem["spot_id"] = gem["x"].astype(str) + "_" + gem["y"].astype(str)
    spot_index = pd.Index(sorted(gem["spot_id"].unique()), name="spot_id")
    gene_index = pd.Index(sorted(gem["geneID"].astype(str).unique()), name="gene")
    spot_codes = pd.Categorical(gem["spot_id"], categories=spot_index).codes
    gene_codes = pd.Categorical(gem["geneID"].astype(str), categories=gene_index).codes
    matrix = sparse.coo_matrix(
        (pd.to_numeric(gem["MIDCounts"], errors="coerce").fillna(0).astype(float).values, (spot_codes, gene_codes)),
        shape=(len(spot_index), len(gene_index)),
    ).tocsr()
    matrix, count_audit = coerce_nonnegative_integer_counts(matrix)

    meta = pd.read_csv(metadata_path, low_memory=False)
    if not {"x", "y"}.issubset(meta.columns):
        raise ValueError(f"metadata missing x/y columns: {metadata_path}")
    meta = meta.copy()
    meta["spot_id"] = meta["x"].astype(str) + "_" + meta["y"].astype(str)
    obs = pd.DataFrame(index=spot_index).join(meta.set_index("spot_id"), how="left")
    obs["sample_id"] = sample_id
    obs["x"] = pd.to_numeric(obs["x"], errors="coerce")
    obs["y"] = pd.to_numeric(obs["y"], errors="coerce")
    var = pd.DataFrame(index=gene_index)
    var["gene_id"] = gene_index.astype(str)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=float)
    adata.uns["spatial_context_method"] = "raw_gem_for_cell2location"
    adata.uns["spatial_count_audit"] = count_audit
    return adata


def read_cell2location_manifest(spatial_dir: str | Path) -> dict:
    manifest_path = Path(spatial_dir) / "deconvolution_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"cell2location deconvolution manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("method") != "cell2location":
        raise ValueError(f"spatial context manifest method is not cell2location: {payload.get('method')!r}")
    return payload


def read_cell2location_abundance_tables(spatial_dir: str | Path) -> list[pd.DataFrame]:
    """Read validated cell2location abundance tables.

    A deconvolution manifest is required so legacy signature-scoring tables do
    not get silently reused as if they were true deconvolution outputs.
    """
    spatial_dir = Path(spatial_dir)
    manifest = read_cell2location_manifest(spatial_dir)
    candidates = sorted(
        set(
            [
                *spatial_dir.glob("cell2location_abundance.csv"),
                *spatial_dir.glob("cell2location_abundance.csv.gz"),
                *spatial_dir.glob("*cell2location*abundance*.csv"),
                *spatial_dir.glob("*cell2location*abundance*.csv.gz"),
            ]
        )
    )
    if not candidates:
        raise FileNotFoundError(f"no cell2location abundance table found in {spatial_dir}")
    tables: list[pd.DataFrame] = []
    for path in candidates:
        table = pd.read_csv(path, compression="infer", low_memory=False)
        abundance_cols = [col for col in table.columns if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(table[col])]
        if not abundance_cols:
            raise ValueError(f"cell2location abundance table has no numeric cell-type columns: {path}")
        table.attrs["spatial_context_method"] = "cell2location"
        table.attrs["cell2location_manifest"] = manifest
        table.attrs["abundance_columns"] = abundance_cols
        tables.append(table)
    return tables


def write_cell2location_manifest(output_dir: str | Path, payload: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"method": "cell2location", **payload}
    path = output_dir / "deconvolution_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def reference_signature_means(adata, *, label_col: str = "cell2location_label") -> pd.DataFrame:
    """Compute hard-coded raw-count mean signatures for diagnostics/fallback."""
    labels = adata.obs[label_col].astype(str)
    rows: list[pd.Series] = []
    for label in sorted(labels.unique()):
        subset = adata[labels == label]
        values = np.asarray(subset.X.mean(axis=0)).ravel()
        rows.append(pd.Series(values, index=adata.var_names.astype(str), name=label))
    return pd.DataFrame(rows).T


def export_cell2location_abundance_from_adata(adata_vis, output_dir: str | Path, *, sample_id: str, detection_alpha: int) -> pd.DataFrame:
    """Export the most useful cell2location abundance posterior from AnnData."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    abundance = None
    for key in ["q05_cell_abundance_w_sf", "means_cell_abundance_w_sf"]:
        if key in adata_vis.obsm:
            abundance = pd.DataFrame(adata_vis.obsm[key], index=adata_vis.obs_names)
            if hasattr(adata_vis, "uns") and "mod" in adata_vis.uns:
                factor_names = adata_vis.uns["mod"].get("factor_names")
                if factor_names is not None and len(factor_names) == abundance.shape[1]:
                    abundance.columns = [str(value) for value in factor_names]
            break
    if abundance is None:
        raise ValueError("cell2location posterior abundance not found in AnnData obsm")
    meta_cols = [col for col in ["sample_id", "x", "y", "level3"] if col in adata_vis.obs]
    meta = adata_vis.obs[meta_cols].copy()
    if "sample_id" not in meta:
        meta["sample_id"] = sample_id
    table = pd.concat([meta.reset_index(names="spot_id"), abundance.reset_index(drop=True)], axis=1)
    table.attrs["detection_alpha"] = int(detection_alpha)
    table.attrs["sample_id"] = sample_id
    table.to_csv(output_dir / f"cell2location_abundance_alpha{detection_alpha}_{sample_id}.csv.gz", index=False)
    return table
