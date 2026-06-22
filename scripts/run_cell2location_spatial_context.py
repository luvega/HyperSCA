#!/usr/bin/env python
"""Build cell2location spatial context from scCRC_ICB reference and ST_CRC_MSS GEM files."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

from src.discovery.target_discovery.cell2location_context import (
    build_icb_reference_anndata,
    coerce_nonnegative_integer_counts,
    downsample_reference_by_label,
    export_cell2location_abundance_from_adata,
    reference_signature_means,
    spatial_gem_to_anndata,
    write_cell2location_manifest,
)


def _sample_id_from_gem(path: Path) -> str:
    return path.name.replace("STexpression_", "").replace(".gem.gz", "").replace(".gem", "")


def _matching_metadata(st_dir: Path, sample_id: str) -> Path:
    path = st_dir / f"STmetadata_{sample_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing ST metadata for {sample_id}: {path}")
    return path


def _log(message: str) -> None:
    print(f"[cell2location] {message}", flush=True)


def _ensure_csr(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()
    return adata


def _strip_signature_prefix(columns) -> list[str]:
    prefix = "means_per_cluster_mu_fg_"
    return [str(col).replace(prefix, "", 1) if str(col).startswith(prefix) else str(col) for col in columns]


def _extract_reference_signatures(adata_ref) -> pd.DataFrame:
    if "means_per_cluster_mu_fg" in adata_ref.varm:
        signatures = pd.DataFrame(adata_ref.varm["means_per_cluster_mu_fg"], index=adata_ref.var_names).copy()
        signatures.columns = _strip_signature_prefix(signatures.columns)
        return signatures
    signature_cols = [col for col in adata_ref.var.columns if str(col).startswith("means_per_cluster_mu_fg_")]
    if signature_cols:
        signatures = adata_ref.var[signature_cols].copy()
        signatures.columns = _strip_signature_prefix(signatures.columns)
        return signatures
    available = {
        "varm": list(getattr(adata_ref, "varm", {}).keys()),
        "var_prefix_matches": signature_cols,
    }
    raise ValueError(f"cell2location RegressionModel did not export reference signatures: {available}")


def _cell2location_signatures(adata_ref, *, label_key: str, max_epochs: int, batch_key: str | None):
    import cell2location
    from cell2location.models import RegressionModel
    from cell2location.utils.filtering import filter_genes
    import torch

    torch.set_float32_matmul_precision("high")
    _ensure_csr(adata_ref)
    _log(f"filtering reference genes for RegressionModel: cells={adata_ref.n_obs}, genes={adata_ref.n_vars}")
    selected = filter_genes(adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
    adata_ref = adata_ref[:, selected].copy()
    _ensure_csr(adata_ref)
    _log(f"training RegressionModel: cells={adata_ref.n_obs}, genes={adata_ref.n_vars}, epochs={max_epochs}")
    RegressionModel.setup_anndata(adata=adata_ref, labels_key=label_key, batch_key=batch_key)
    model = RegressionModel(adata_ref)
    model.train(max_epochs=max_epochs, train_size=1)
    _log("exporting RegressionModel posterior signatures")
    adata_ref = model.export_posterior(
        adata_ref,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500},
    )
    signatures = _extract_reference_signatures(adata_ref)
    return signatures, adata_ref


def _run_spatial_mapping(
    adata_vis,
    signatures: pd.DataFrame,
    *,
    n_cells_per_location: int,
    detection_alpha: int,
    max_epochs: int,
    max_genes: int | None = None,
    train_batch_size: int | None = 4096,
    posterior_batch_size: int | None = 4096,
    posterior_num_samples: int = 1000,
):
    import cell2location
    from cell2location.models import Cell2location
    import torch

    torch.set_float32_matmul_precision("high")
    shared = [gene for gene in adata_vis.var_names.astype(str) if gene in signatures.index]
    if not shared:
        raise ValueError("no shared genes between spatial AnnData and cell2location signatures")
    if max_genes and int(max_genes) > 0 and len(shared) > int(max_genes):
        shared = shared[: int(max_genes)]
    adata_vis = adata_vis[:, shared].copy()
    _ensure_csr(adata_vis)
    adata_vis.X, count_audit = coerce_nonnegative_integer_counts(adata_vis.X)
    adata_vis.uns["cell2location_count_input_audit"] = count_audit
    signatures = signatures.loc[shared]
    _log(
        "training Cell2location mapping: "
        f"sample={adata_vis.obs['sample_id'].iloc[0] if 'sample_id' in adata_vis.obs else 'unknown'}, "
        f"spots={adata_vis.n_obs}, genes={adata_vis.n_vars}, alpha={detection_alpha}, epochs={max_epochs}, "
        f"max_genes={max_genes}, "
        f"train_batch_size={train_batch_size}, posterior_batch_size={posterior_batch_size}, "
        f"posterior_num_samples={posterior_num_samples}"
    )
    Cell2location.setup_anndata(adata=adata_vis)
    model = Cell2location(
        adata_vis,
        cell_state_df=signatures,
        N_cells_per_location=n_cells_per_location,
        detection_alpha=detection_alpha,
    )
    model.train(max_epochs=max_epochs, batch_size=train_batch_size, train_size=1)
    _log("exporting Cell2location mapping posterior")
    adata_vis = model.export_posterior(
        adata_vis,
        sample_kwargs={"num_samples": int(posterior_num_samples), "batch_size": posterior_batch_size or model.adata.n_obs},
    )
    return adata_vis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cell2location spatial context for HyperSCA")
    parser.add_argument("--raw-icb-dir", type=Path, default=Path("/home/a/Data/scCRC_ICB/input"))
    parser.add_argument("--metadata", type=Path, default=Path("/home/a/Data/scCRC_ICB/input/GSE236581_CRC-ICB_metadata.txt.gz"))
    parser.add_argument("--st-dir", type=Path, default=Path("/home/a/Data/ST_CRC_MSS"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "from_scratch" / "cell2location")
    parser.add_argument("--label-key", type=str, default="SubCellType")
    parser.add_argument("--major-label-key", type=str, default="MajorCellType")
    parser.add_argument("--batch-key", type=str, default=None)
    parser.add_argument("--n-cells-per-location", type=int, default=30)
    parser.add_argument("--detection-alphas", type=str, default="20,200")
    parser.add_argument("--reference-epochs", type=int, default=250)
    parser.add_argument("--mapping-epochs", type=int, default=30000)
    parser.add_argument("--posterior-num-samples", type=int, default=1000)
    parser.add_argument("--max-cells-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reuse-prepared", action="store_true", help="Reuse existing raw reference/spatial h5ad files")
    parser.add_argument("--prepare-only", action="store_true", help="Only export raw reference/spatial inputs and hard-coded signatures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "status": "started",
        "raw_icb_dir": str(args.raw_icb_dir),
        "metadata": str(args.metadata),
        "st_dir": str(args.st_dir),
        "label_key": args.label_key,
        "major_label_key": args.major_label_key,
        "n_cells_per_location": args.n_cells_per_location,
        "detection_alphas": [int(value) for value in args.detection_alphas.split(",") if value.strip()],
        "max_cells_per_label": args.max_cells_per_label,
        "seed": args.seed,
        "reuse_prepared": bool(args.reuse_prepared),
    }
    try:
        import anndata as ad

        ref_path = args.output_dir / "sccrc_icb_reference_raw_counts.h5ad"
        hardcoded_path = args.output_dir / "reference_signatures_hardcoded.csv.gz"
        if args.reuse_prepared and ref_path.exists():
            _log(f"reusing prepared reference: {ref_path}")
            adata_ref = ad.read_h5ad(ref_path)
        else:
            _log("building raw scCRC_ICB reference from 10x matrix and metadata")
            adata_ref = build_icb_reference_anndata(
                args.raw_icb_dir,
                args.metadata,
                label_key=args.label_key,
                major_label_key=args.major_label_key,
            )
            _log(f"writing raw reference h5ad: {ref_path}")
            adata_ref.write_h5ad(ref_path)
        if args.reuse_prepared and hardcoded_path.exists():
            _log(f"reusing diagnostic hard-coded signatures: {hardcoded_path}")
        else:
            _log("computing diagnostic hard-coded mean signatures")
            hardcoded = reference_signature_means(adata_ref)
            hardcoded.to_csv(hardcoded_path)
        manifest["reference_shape"] = [int(adata_ref.n_obs), int(adata_ref.n_vars)]

        if args.reuse_prepared:
            spatial_paths = sorted(str(path) for path in args.output_dir.glob("spatial_raw_*.h5ad"))
            if spatial_paths:
                _log(f"reusing {len(spatial_paths)} prepared spatial h5ad files")
            else:
                _log("no prepared spatial h5ad files found; rebuilding from GEM")
        else:
            spatial_paths = []
        if not spatial_paths:
            spatial_paths = []
            for gem_path in sorted(args.st_dir.glob("STexpression_*.gem.gz")):
                sample_id = _sample_id_from_gem(gem_path)
                _log(f"building raw spatial h5ad: {sample_id}")
                adata_vis = spatial_gem_to_anndata(gem_path, _matching_metadata(args.st_dir, sample_id), sample_id=sample_id)
                out_path = args.output_dir / f"spatial_raw_{sample_id}.h5ad"
                adata_vis.write_h5ad(out_path)
                spatial_paths.append(str(out_path))
        manifest["spatial_h5ad"] = spatial_paths

        if args.prepare_only:
            manifest["status"] = "prepared_only"
            write_cell2location_manifest(args.output_dir, manifest)
            return 0

        adata_ref_train = downsample_reference_by_label(
            adata_ref,
            max_cells_per_label=args.max_cells_per_label,
            seed=args.seed,
            label_col="cell2location_label",
        )
        manifest["reference_training_shape"] = [int(adata_ref_train.n_obs), int(adata_ref_train.n_vars)]
        manifest["reference_sampling"] = adata_ref_train.uns.get("cell2location_reference_sampling", {})
        signatures, adata_ref_fit = _cell2location_signatures(
            adata_ref_train,
            label_key="cell2location_label",
            max_epochs=args.reference_epochs,
            batch_key=args.batch_key,
        )
        signatures.to_csv(args.output_dir / "reference_signatures.csv.gz")
        adata_ref_fit.write_h5ad(args.output_dir / "sccrc_icb_reference_cell2location_fit.h5ad")

        all_tables = []
        for alpha in manifest["detection_alphas"]:
            for h5ad_path in spatial_paths:
                import anndata as ad

                adata_vis = ad.read_h5ad(h5ad_path)
                sample_id = Path(h5ad_path).stem.replace("spatial_raw_", "")
                mapped = _run_spatial_mapping(
                    adata_vis,
                    signatures,
                    n_cells_per_location=args.n_cells_per_location,
                    detection_alpha=alpha,
                    max_epochs=args.mapping_epochs,
                    posterior_num_samples=args.posterior_num_samples,
                )
                mapped.write_h5ad(args.output_dir / f"cell2location_map_alpha{alpha}_{sample_id}.h5ad")
                all_tables.append(
                    export_cell2location_abundance_from_adata(mapped, args.output_dir, sample_id=sample_id, detection_alpha=alpha)
                )
        selected_alpha = manifest["detection_alphas"][0]
        selected = [table for table in all_tables if int(table.attrs.get("detection_alpha", -1)) == int(selected_alpha)]
        if not selected:
            selected = all_tables[: len(spatial_paths)]
        pd.concat(selected, ignore_index=True).to_csv(args.output_dir / "cell2location_abundance.csv.gz", index=False)
        manifest["selected_detection_alpha"] = selected_alpha
        manifest["status"] = "ready"
        write_cell2location_manifest(args.output_dir, manifest)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        write_cell2location_manifest(args.output_dir, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
