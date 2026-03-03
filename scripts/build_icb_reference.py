#!/usr/bin/env python
"""Build ICB reference model from scCRC_ICB expression data.

Trains scVI (unsupervised) or scANVI (semi-supervised with cell-type labels)
on the ICB single-cell data, then exports reference artifacts to data/ref/.

Output layout:
    data/ref/models/icb_reference/v1/   — model dir (scvi-tools save format)
    data/ref/mappings/icb_reference/v1/ — label_dict.json, mapping_stats.json
    data/ref/manifest/reference_manifest.json

Usage:
    python scripts/build_icb_reference.py
    python scripts/build_icb_reference.py --label-key MajorCellType --max-epochs 50
    python scripts/build_icb_reference.py --model scanvi --label-key MajorCellType
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REF_ROOT = DATA_DIR / "ref"
ICB_H5AD = DATA_DIR / "scRNA" / "scCRC_ICB" / "expression.h5ad"

VERSION = "v1"

CANDIDATE_LABEL_KEYS = [
    "MajorCellType", "MidCellType", "celltype",
    "cell_type", "Level1", "annotation",
]


def _resolve_label_key(adata, requested: str | None) -> str | None:
    """Find the best available cell-type label column."""
    if requested and requested in adata.obs.columns:
        return requested
    for key in CANDIDATE_LABEL_KEYS:
        if key in adata.obs.columns:
            n_labels = adata.obs[key].nunique()
            if 2 <= n_labels <= 200:
                return key
    return None


def preprocess_for_reference(adata, min_genes: int = 200, min_cells: int = 3,
                             n_top_genes: int = 3000):
    """Standard preprocessing for scVI/scANVI reference training."""
    print(f"  Raw: {adata.n_obs} cells x {adata.n_vars} genes")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  After filter: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, subset=True, flavor="seurat_v3",
        layer="counts",
    )
    print(f"  After HVG: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def train_scvi_reference(adata, max_epochs: int = 100, batch_key: str | None = None):
    """Train scVI model and return the trained model."""
    import scvi

    scvi.model.SCVI.setup_anndata(
        adata, layer="counts", batch_key=batch_key,
    )
    model = scvi.model.SCVI(
        adata, n_latent=30, n_layers=2, gene_likelihood="nb",
    )
    print(f"  Training scVI (max_epochs={max_epochs}) ...")
    t0 = time.time()
    model.train(max_epochs=max_epochs, early_stopping=True)
    elapsed = time.time() - t0
    print(f"  scVI training done in {elapsed:.1f}s")
    return model, "scvi"


def train_scanvi_reference(adata, label_key: str, max_epochs: int = 100,
                           batch_key: str | None = None):
    """Train scANVI model (semi-supervised) and return the trained model."""
    import scvi

    unlabeled_cat = "Unknown"
    labels = adata.obs[label_key].astype(str).copy()
    labels[labels.isna()] = unlabeled_cat
    adata.obs["_scanvi_labels"] = labels

    scvi.model.SCVI.setup_anndata(
        adata, layer="counts", batch_key=batch_key,
    )
    vae = scvi.model.SCVI(adata, n_latent=30, n_layers=2, gene_likelihood="nb")
    print(f"  Pre-training scVI (max_epochs={max(20, max_epochs // 2)}) ...")
    vae.train(max_epochs=max(20, max_epochs // 2), early_stopping=True)

    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae, unlabeled_category=unlabeled_cat,
        labels_key="_scanvi_labels",
    )
    print(f"  Fine-tuning scANVI (max_epochs={max_epochs}) ...")
    t0 = time.time()
    scanvi.train(max_epochs=max_epochs, early_stopping=True)
    elapsed = time.time() - t0
    print(f"  scANVI training done in {elapsed:.1f}s")
    return scanvi, "scanvi"


def export_reference(model, adata, model_type: str, label_key: str | None,
                     version: str = VERSION):
    """Save model + mappings + manifest to data/ref/."""
    model_dir = REF_ROOT / "models" / "icb_reference" / version
    map_dir = REF_ROOT / "mappings" / "icb_reference" / version
    manifest_dir = REF_ROOT / "manifest"
    for d in [model_dir, map_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir), overwrite=True)
    print(f"  Model saved → {model_dir}")

    label_dict = {}
    mapping_stats = {
        "model_type": model_type,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "label_key": label_key or "",
        "version": version,
        "created": datetime.now().isoformat(),
    }

    if label_key and label_key in adata.obs.columns:
        vc = adata.obs[label_key].value_counts()
        label_dict = {str(k): int(v) for k, v in vc.items()}
        mapping_stats["n_labels"] = len(label_dict)
        mapping_stats["label_counts"] = label_dict

    (map_dir / "label_dict.json").write_text(
        json.dumps(label_dict, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (map_dir / "mapping_stats.json").write_text(
        json.dumps(mapping_stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  Mappings saved → {map_dir}")

    latent = model.get_latent_representation()
    adata.obsm["X_scvi"] = latent

    if model_type == "scanvi":
        preds = model.predict()
        probs = model.predict(soft=True)
        adata.obs["label_ref"] = preds
        adata.obs["label_ref_prob"] = probs.max(axis=1).values
        adata.obs["label_ref_entropy"] = -(
            probs * np.log(probs.clip(1e-10))
        ).sum(axis=1).values

    ref_h5ad_path = map_dir / "reference_adata.h5ad"
    adata.write(ref_h5ad_path)
    print(f"  Reference AnnData → {ref_h5ad_path}")

    manifest = {
        "reference_name": "icb_reference",
        "version": version,
        "model_type": model_type,
        "model_dir": str(model_dir),
        "mappings_dir": str(map_dir),
        "reference_h5ad": str(ref_h5ad_path),
        "label_key": label_key or "",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "hvg_only": True,
        "created": datetime.now().isoformat(),
    }
    manifest_path = manifest_dir / "reference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  Manifest → {manifest_path}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ICB reference model")
    parser.add_argument("--input", type=str, default=str(ICB_H5AD),
                        help="Path to ICB expression.h5ad")
    parser.add_argument("--model", choices=["scvi", "scanvi", "auto"], default="auto",
                        help="Model type (auto = scanvi if labels available)")
    parser.add_argument("--label-key", type=str, default=None,
                        help="obs column for cell-type labels")
    parser.add_argument("--batch-key", type=str, default=None,
                        help="obs column for batch correction")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--version", type=str, default=VERSION)
    parser.add_argument("--subsample", type=int, default=0,
                        help="Subsample to N cells for fast testing (0=all)")
    args = parser.parse_args()

    h5ad_path = Path(args.input)
    if not h5ad_path.exists():
        print(f"[ERROR] ICB h5ad not found: {h5ad_path}")
        print("  Run `python scripts/prepare_h5ad.py --mode research` first.")
        return 1

    print(f"[1/4] Loading {h5ad_path} ...")
    adata = sc.read_h5ad(str(h5ad_path))
    if adata.obs.get("label_original") is None and args.label_key:
        if args.label_key in adata.obs.columns:
            adata.obs["label_original"] = adata.obs[args.label_key].copy()

    if args.subsample > 0 and adata.n_obs > args.subsample:
        sc.pp.subsample(adata, n_obs=args.subsample, random_state=42)
        print(f"  Subsampled to {adata.n_obs} cells")

    label_key = _resolve_label_key(adata, args.label_key)
    if label_key:
        adata.obs["label_original"] = adata.obs[label_key].astype(str)
        print(f"  Label key: {label_key} ({adata.obs[label_key].nunique()} types)")
    else:
        print("  No cell-type labels found, will train unsupervised scVI")

    print("[2/4] Preprocessing ...")
    adata = preprocess_for_reference(adata, n_top_genes=args.n_top_genes)

    model_choice = args.model
    if model_choice == "auto":
        model_choice = "scanvi" if label_key else "scvi"

    print(f"[3/4] Training {model_choice} ...")
    if model_choice == "scanvi" and label_key:
        model, mtype = train_scanvi_reference(
            adata, label_key=label_key, max_epochs=args.max_epochs,
            batch_key=args.batch_key,
        )
    else:
        if model_choice == "scanvi" and not label_key:
            print("  WARN: No labels → falling back to scVI")
        model, mtype = train_scvi_reference(
            adata, max_epochs=args.max_epochs, batch_key=args.batch_key,
        )

    print("[4/4] Exporting reference ...")
    manifest = export_reference(
        model, adata, model_type=mtype,
        label_key=label_key, version=args.version,
    )
    print(f"\n[DONE] Reference built: {manifest['model_type']} "
          f"({manifest['n_cells']} cells, {manifest['n_genes']} genes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
