#!/usr/bin/env python
"""Prepare sparse AnnData caches from raw OSTA HumanColon_Oliveira data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_DATASETS = (
    "Chromium_HumanColon_Oliveira",
    "Visium_HumanColon_Oliveira",
    "VisiumHD_HumanColon_Oliveira",
    "VisiumHD_HumanColon_Oliveira_segmented",
    "Xenium_HumanColon_Oliveira",
)


def _decode(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def _read_10x_h5(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(v) for v in matrix["shape"][()])
        data = matrix["data"][()]
        indices = matrix["indices"][()]
        indptr = matrix["indptr"][()]
        barcodes = _decode(matrix["barcodes"][()])
        features = matrix["features"]
        var = pd.DataFrame(index=pd.Index(_decode(features["name"][()]), name="gene_symbol"))
        for key in ("id", "feature_type", "genome"):
            if key in features:
                var[key] = _decode(features[key][()])
    # 10x stores features x barcodes as CSC. AnnData expects obs x var.
    x = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    return x, var, barcodes


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            out.append(name)
        else:
            counts[name] += 1
            out.append(f"{name}-{counts[name]}")
    return out


def _read_10x_h5_unique(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
    x, var, barcodes = _read_10x_h5(path)
    var.index = _make_unique(var.index.astype(str).tolist())
    return x, var, barcodes


def _metadata_indexed(metadata: pd.DataFrame, id_col: str, barcodes: list[str]) -> pd.DataFrame:
    metadata = metadata.copy()
    if id_col not in metadata.columns:
        metadata[id_col] = barcodes[: len(metadata)]
    metadata[id_col] = metadata[id_col].astype(str)
    metadata = metadata.drop_duplicates(subset=[id_col]).set_index(id_col)
    obs = pd.DataFrame(index=pd.Index(barcodes, name=id_col))
    common_cols = [col for col in metadata.columns if col not in obs.columns]
    obs = obs.join(metadata[common_cols], how="left")
    obs[id_col] = obs.index.astype(str)
    return obs


def _add_spatial(adata: ad.AnnData, obs: pd.DataFrame, x_col: str, y_col: str) -> None:
    if x_col in obs.columns and y_col in obs.columns:
        xy = obs[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        adata.obsm["spatial"] = xy
        adata.obs["x"] = xy[:, 0]
        adata.obs["y"] = xy[:, 1]


def _write_cache(adata: ad.AnnData, out_path: Path, dataset_id: str, source_paths: list[Path]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    manifest = {
        "dataset_id": dataset_id,
        "cache_format": "h5ad_sparse",
        "cache_path": str(out_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "nnz": int(adata.X.nnz) if sparse.issparse(adata.X) else None,
        "obs_columns": list(map(str, adata.obs.columns)),
        "has_spatial": "spatial" in adata.obsm,
        "assay_type": str(adata.uns.get("assay_type", "")),
        "source_paths": [str(path) for path in source_paths],
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _chromium(raw_dir: Path) -> tuple[ad.AnnData, list[Path]]:
    h5_path = raw_dir / "filtered_feature_bc_matrix.h5"
    metadata_path = raw_dir / "cell_metadata.csv"
    x, var, barcodes = _read_10x_h5_unique(h5_path)
    metadata = pd.read_csv(metadata_path)
    obs = _metadata_indexed(metadata, "Barcode", barcodes)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    return adata, [h5_path, metadata_path]


def _visium(raw_dir: Path) -> tuple[ad.AnnData, list[Path]]:
    outs = raw_dir / "outs"
    h5_path = outs / "filtered_feature_bc_matrix.h5"
    positions_path = outs / "spatial" / "tissue_positions.csv"
    x, var, barcodes = _read_10x_h5_unique(h5_path)
    positions = pd.read_csv(positions_path)
    if "barcode" not in positions.columns:
        positions = pd.read_csv(
            positions_path,
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
    obs = _metadata_indexed(positions, "barcode", barcodes)
    obs["label"] = obs.get("in_tissue", pd.Series(index=obs.index, dtype=object)).map({1: "in_tissue", 0: "off_tissue"}).fillna("unknown")
    adata = ad.AnnData(X=x, obs=obs, var=var)
    _add_spatial(adata, obs, "pxl_col_in_fullres", "pxl_row_in_fullres")
    return adata, [h5_path, positions_path]


def _xenium(raw_dir: Path) -> tuple[ad.AnnData, list[Path]]:
    h5_path = raw_dir / "cell_feature_matrix.h5"
    metadata_path = raw_dir / "cells.parquet"
    x, var, barcodes = _read_10x_h5_unique(h5_path)
    metadata = pd.read_parquet(metadata_path)
    obs = _metadata_indexed(metadata, "cell_id", barcodes)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    _add_spatial(adata, obs, "x_centroid", "y_centroid")
    return adata, [h5_path, metadata_path]


def _find_first(raw_dir: Path, pattern: str) -> Path:
    hits = sorted(raw_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matched {pattern} under {raw_dir}")
    return hits[0]


def _barcode_from_cell_id(cell_id: Any) -> str:
    text = str(cell_id)
    if text.startswith("cellid_"):
        return text
    return f"cellid_{int(cell_id):09d}-1"


def _polygon_stats(coords: list[list[float]]) -> tuple[float, float, float, int]:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return 0.0, 0.0, 0.0, 0
    if arr.shape[0] > 1 and np.allclose(arr[0, :2], arr[-1, :2]):
        arr = arr[:-1]
    if arr.shape[0] < 3:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean()), 0.0, int(arr.shape[0])
    x = arr[:, 0]
    y = arr[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(cross.sum())
    area = abs(signed_area)
    if area <= 0:
        return float(x.mean()), float(y.mean()), 0.0, int(arr.shape[0])
    cx = float(((x + x_next) * cross).sum() / (6.0 * signed_area))
    cy = float(((y + y_next) * cross).sum() / (6.0 * signed_area))
    return cx, cy, area, int(arr.shape[0])


def _segmentation_metadata(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        if "cell_id" not in props:
            continue
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Polygon" or not coordinates:
            continue
        x, y, area, n_vertices = _polygon_stats(coordinates[0])
        cell_id = props["cell_id"]
        rows.append(
            {
                "barcode": _barcode_from_cell_id(cell_id),
                "cell_id_numeric": int(cell_id) if str(cell_id).isdigit() else np.nan,
                "x": x,
                "y": y,
                "cell_area_px2": area,
                "cell_segmentation_vertices": n_vertices,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["cell_id_numeric", "x", "y", "cell_area_px2", "cell_segmentation_vertices"])
    return pd.DataFrame(rows).drop_duplicates("barcode").set_index("barcode")


def _visium_hd(raw_dir: Path) -> tuple[ad.AnnData, list[Path]]:
    h5_path = _find_first(raw_dir, "**/square_008um/**/filtered_feature_bc_matrix.h5")
    if "square_008um" not in str(h5_path):
        h5_path = _find_first(raw_dir, "**/filtered_feature_bc_matrix.h5")
    x, var, barcodes = _read_10x_h5_unique(h5_path)
    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    source_paths = [h5_path]
    decon_hits = sorted(raw_dir.glob("**/square_008um/deconvolution.csv.gz"))
    if decon_hits:
        decon = pd.read_csv(decon_hits[0])
        barcode_col = decon.columns[1] if len(decon.columns) > 1 else decon.columns[0]
        decon[barcode_col] = decon[barcode_col].astype(str)
        decon = decon.drop_duplicates(subset=[barcode_col]).set_index(barcode_col)
        obs = obs.join(decon, how="left")
        label_cols = [col for col in ("DeconLabel1", "decon_label_1", "label") if col in obs.columns]
        if label_cols:
            obs["label"] = obs[label_cols[0]].astype(str)
        source_paths.append(decon_hits[0])
    position_hits = sorted(raw_dir.glob("**/square_008um/spatial/tissue_positions*"))
    if position_hits:
        positions = pd.read_parquet(position_hits[0]) if position_hits[0].suffix == ".parquet" else pd.read_csv(position_hits[0])
        if "barcode" in positions.columns:
            obs = obs.join(positions.drop_duplicates("barcode").set_index("barcode"), how="left", rsuffix="_position")
            source_paths.append(position_hits[0])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    for x_col, y_col in (
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
        ("x", "y"),
    ):
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            _add_spatial(adata, adata.obs, x_col, y_col)
            break
    return adata, source_paths


def _visium_hd_segmented(raw_dir: Path) -> tuple[ad.AnnData, list[Path]]:
    seg_dir = raw_dir / "segmented_outputs"
    h5_path = seg_dir / "filtered_feature_cell_matrix.h5"
    geojson_path = seg_dir / "cell_segmentations.geojson"
    x, var, barcodes = _read_10x_h5_unique(h5_path)
    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    metadata = _segmentation_metadata(geojson_path)
    obs = obs.join(metadata, how="left")
    obs["assay_type"] = "visiumhd_segmented_cell"
    adata = ad.AnnData(X=x, obs=obs, var=var)
    _add_spatial(adata, obs, "x", "y")
    adata.uns["assay_type"] = "visiumhd_segmented_cell"
    adata.uns["source_dataset_id"] = raw_dir.name
    return adata, [h5_path, geojson_path]


def build_dataset(raw_root: Path, cache_root: Path, dataset_id: str, *, overwrite: bool) -> dict[str, Any]:
    segmented = dataset_id.endswith("_segmented")
    raw_dataset_id = dataset_id.removesuffix("_segmented") if segmented else dataset_id
    raw_dir = raw_root / raw_dataset_id
    if not raw_dir.exists():
        return {"dataset_id": dataset_id, "status": "missing_raw", "raw_dir": str(raw_dir)}
    out_path = cache_root / dataset_id / "benchmark.h5ad"
    if out_path.exists() and not overwrite:
        existing = ad.read_h5ad(out_path, backed="r")
        manifest = {
            "dataset_id": dataset_id,
            "status": "exists",
            "cache_path": str(out_path),
            "n_obs": int(existing.n_obs),
            "n_vars": int(existing.n_vars),
        }
        existing.file.close()
        return manifest
    if dataset_id.startswith("Chromium_"):
        adata, source_paths = _chromium(raw_dir)
    elif raw_dataset_id.startswith("VisiumHD_") and segmented:
        adata, source_paths = _visium_hd_segmented(raw_dir)
    elif raw_dataset_id.startswith("VisiumHD_"):
        adata, source_paths = _visium_hd(raw_dir)
    elif raw_dataset_id.startswith("Visium_"):
        adata, source_paths = _visium(raw_dir)
    elif raw_dataset_id.startswith("Xenium_"):
        adata, source_paths = _xenium(raw_dir)
    else:
        return {"dataset_id": dataset_id, "status": "unsupported", "raw_dir": str(raw_dir)}
    manifest = _write_cache(adata, out_path, dataset_id, source_paths)
    manifest["status"] = "written"
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/home/a/Data/OSTA"))
    parser.add_argument("--cache-root", type=Path, default=Path("/home/a/Data/OSTA/benchmark_cache"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    for dataset_id in datasets:
        print(f"[{dataset_id}] preparing sparse h5ad cache", flush=True)
        try:
            result = build_dataset(args.raw_root, args.cache_root, dataset_id, overwrite=args.overwrite)
        except Exception as exc:
            result = {"dataset_id": dataset_id, "status": "failed", "message": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.cache_root / "prepare_osta_benchmark_cache_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if any(item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
