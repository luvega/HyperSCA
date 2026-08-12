#!/usr/bin/env python
"""Run a unified cell annotation and spatial mapping POC."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import yaml
from scipy import io as scipy_io
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.discovery.target_discovery.unified_annotation import (
    PREDICTION_COLUMNS,
    TransferResult,
    build_unified_celltype_dictionary,
    cell2location_status_for_dataset,
    compute_prediction_qc,
    map_label_to_unified,
    read_query_from_spec,
    read_reference_from_spec,
    run_hvae_label_transfer,
    run_prototype_label_transfer,
    run_rctd_existing_annotation,
    standardize_cell2location_abundance,
    validate_abundance_table,
    validate_prediction_table,
    write_csv_gz,
)


def _parse_methods(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return ["cell2location", "rctd", "hvae"]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _progress_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _initialize_progress(path: Path | None, *, run_id: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    block = (
        "\n".join(
            [
                f"# {run_id} Progress",
                "",
                f"- initialized_at: {_progress_timestamp()}",
                "",
                "## Events",
                "",
            ]
        )
    )
    if path.exists():
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n---\n\n")
            handle.write(block)
            handle.write("\n")
    else:
        path.write_text(block, encoding="utf-8")


def _append_progress(path: Path | None, message: str) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {_progress_timestamp()} {message}\n")


def _r_package_available(package: str) -> bool:
    try:
        result = subprocess.run(
            ["Rscript", "-e", f"quit(status=ifelse(requireNamespace('{package}', quietly=TRUE), 0, 1))"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _rctd_spacexr_doublet_status() -> tuple[str, str]:
    if not _r_package_available("spacexr"):
        return (
            "blocked:spacexr_unavailable",
            "RCTD training requires the R package spacexr; existing RCTD labels were not found for this dataset.",
        )
    return ("ready", "")


def _shared_genes_for_rctd(reference, query, *, max_genes: int | None) -> list[str]:
    query_genes = set(map(str, query.var_names))
    genes = [str(gene) for gene in reference.var_names if str(gene) in query_genes]
    if max_genes and len(genes) > max_genes:
        genes = genes[:max_genes]
    return genes


def _gene_by_obs_counts(adata, genes: list[str]):
    x = adata[:, genes].X.T
    if sparse.issparse(x):
        x = x.tocoo(copy=True)
        x.data = np.rint(np.maximum(x.data, 0)).astype(np.int64)
        return x
    arr = np.rint(np.maximum(np.asarray(x), 0)).astype(np.int64)
    return sparse.coo_matrix(arr)


def _write_lines(path: Path, values) -> None:
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def _write_count_bundle(adata, genes: list[str], output_dir: Path, prefix: str) -> None:
    scipy_io.mmwrite(output_dir / f"{prefix}_counts.mtx", _gene_by_obs_counts(adata, genes))
    _write_lines(output_dir / f"{prefix}_genes.tsv", genes)
    _write_lines(output_dir / f"{prefix}_barcodes.tsv", adata.obs_names.astype(str))


def _export_spacexr_inputs(reference, query, *, output_dir: Path, max_genes: int | None) -> None:
    genes = _shared_genes_for_rctd(reference, query, max_genes=max_genes)
    if not genes:
        raise ValueError("No shared genes between reference and query for RCTD.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_count_bundle(reference, genes, output_dir, "reference")
    _write_count_bundle(query, genes, output_dir, "query")
    pd.DataFrame(
        {
            "barcode": reference.obs_names.astype(str),
            "cell_type": reference.obs["unified_level1"].astype(str).to_numpy(),
        }
    ).to_csv(output_dir / "reference_labels.csv", index=False)
    pd.DataFrame(
        {
            "barcode": query.obs_names.astype(str),
            "x": pd.to_numeric(query.obs["x"], errors="coerce").fillna(0.0) if "x" in query.obs else 0.0,
            "y": pd.to_numeric(query.obs["y"], errors="coerce").fillna(0.0) if "y" in query.obs else 0.0,
        }
    ).to_csv(output_dir / "query_coords.csv", index=False)


def _query_xy(query, obs_ids: pd.Series) -> pd.DataFrame:
    obs = query.obs.copy()
    x = pd.to_numeric(obs["x"], errors="coerce").fillna(0.0) if "x" in obs else pd.Series(0.0, index=obs.index)
    y = pd.to_numeric(obs["y"], errors="coerce").fillna(0.0) if "y" in obs else pd.Series(0.0, index=obs.index)
    return pd.DataFrame(
        {
            "obs_id": obs_ids.to_numpy(),
            "x": x.reindex(obs_ids).fillna(0.0).to_numpy(),
            "y": y.reindex(obs_ids).fillna(0.0).to_numpy(),
        }
    )


def _spacexr_outputs_to_transfer_result(
    *,
    query,
    result_dir: Path,
    dataset_id: str,
    reference_name: str,
    assay_type: str,
) -> TransferResult:
    results_df = pd.read_csv(result_dir / "results_df.csv.gz")
    abundance_raw = pd.read_csv(result_dir / "abundance.csv.gz")
    if results_df.empty:
        return TransferResult(status="failed:empty_rctd_output", message="spacexr produced no RCTD rows.")
    results_df["obs_id"] = results_df["obs_id"].astype(str)
    abundance_raw["obs_id"] = abundance_raw["obs_id"].astype(str)
    xy = _query_xy(query, results_df["obs_id"])
    first = results_df.get("first_type", pd.Series("Unknown", index=results_df.index)).fillna("Unknown").astype(str)
    second = results_df.get("second_type", pd.Series("Unknown", index=results_df.index)).fillna("Unknown").astype(str)
    primary = [map_label_to_unified(source_system="rctd", major_label=value, fine_label=value) for value in first]
    secondary = [map_label_to_unified(source_system="rctd", major_label=value, fine_label=value) for value in second]
    numeric_abundance = abundance_raw.drop(columns=["obs_id"], errors="ignore").select_dtypes(include="number")
    confidence = numeric_abundance.max(axis=1).to_numpy() if not numeric_abundance.empty else np.ones(len(results_df))
    predictions = pd.DataFrame(
        {
            "obs_id": results_df["obs_id"],
            "dataset_id": dataset_id,
            "x": xy["x"],
            "y": xy["y"],
            "method": "rctd",
            "reference": reference_name,
            "assay_type": assay_type,
            "unified_level0": [item.unified_level0 for item in primary],
            "unified_level1": [item.unified_level1 for item in primary],
            "unified_level2": [item.unified_level2 for item in primary],
            "confidence": confidence,
            "status": results_df.get("spot_class", pd.Series("ok", index=results_df.index)).fillna("ok").astype(str),
            "source_label": first,
            "top2_label": [item.unified_level1 for item in secondary],
        }
    )[list(PREDICTION_COLUMNS)]

    abundance_xy = _query_xy(query, abundance_raw["obs_id"])
    status_by_obs = dict(zip(results_df["obs_id"], predictions["status"], strict=False))
    abundance = pd.DataFrame(
        {
            "spot_id": abundance_raw["obs_id"],
            "sample_id": dataset_id,
            "x": abundance_xy["x"],
            "y": abundance_xy["y"],
            "level3": abundance_raw["obs_id"].map(status_by_obs).fillna("unknown"),
        }
    )
    for col in abundance_raw.columns:
        if col == "obs_id" or not pd.api.types.is_numeric_dtype(abundance_raw[col]):
            continue
        label = map_label_to_unified(source_system="rctd", major_label=col, fine_label=col).unified_level1
        abundance[label] = abundance.get(label, 0.0) + pd.to_numeric(abundance_raw[col], errors="coerce").fillna(0.0)
    validate_prediction_table(predictions)
    validate_abundance_table(abundance)
    return TransferResult(status="ok", predictions=predictions, abundance=abundance)


def _spacexr_rctd_ready_run(
    *,
    reference,
    query,
    dataset_spec: dict[str, Any],
    reference_name: str,
    output_dir: Path,
    max_genes: int,
    max_cores: int,
    test_mode: bool,
    doublet_mode: str,
    reference_min_umi: int,
    umi_min: int,
    counts_min: int,
    umi_min_sigma: int,
    cell_min_instance: int,
) -> TransferResult:
    dataset_id = str(dataset_spec["dataset_id"])
    assay_type = str(dataset_spec.get("assay_type", "whole_transcriptome"))
    safe_name = f"{dataset_id}_{reference_name}".replace("/", "_")
    input_dir = output_dir / "_rctd_inputs" / safe_name
    result_dir = output_dir / "_rctd_raw" / safe_name
    try:
        _export_spacexr_inputs(reference, query, output_dir=input_dir, max_genes=max_genes)
        cmd = [
            "Rscript",
            str(ROOT / "scripts" / "run_spacexr_rctd.R"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(result_dir),
            "--doublet-mode",
            str(doublet_mode),
            "--max-cores",
            str(max_cores),
            "--test-mode",
            "true" if test_mode else "false",
            "--reference-min-umi",
            str(reference_min_umi),
            "--umi-min",
            str(umi_min),
            "--counts-min",
            str(counts_min),
            "--umi-min-sigma",
            str(umi_min_sigma),
            "--cell-min-instance",
            str(cell_min_instance),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            message = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
            return TransferResult(status="failed:spacexr_rctd", message=message[:4000])
        result = _spacexr_outputs_to_transfer_result(
            query=query,
            result_dir=result_dir,
            dataset_id=dataset_id,
            reference_name=reference_name,
            assay_type=assay_type,
        )
        result.message = completed.stdout.strip()
        return result
    except Exception as exc:
        return TransferResult(status=f"failed:{type(exc).__name__}", message=str(exc))


def _cell2location_ready_run(
    *,
    reference,
    query,
    dataset_spec: dict[str, Any],
    reference_name: str,
    output_dir: Path,
    reference_epochs: int,
    mapping_epochs: int,
    max_genes: int,
    n_cells_per_location: int,
    detection_alpha: int,
    train_batch_size: int | None = 4096,
    posterior_batch_size: int | None = 4096,
    posterior_num_samples: int = 1000,
    signature_cache: dict[tuple[str, int], pd.DataFrame] | None = None,
) -> tuple[str, Path | None, str]:
    try:
        from scripts.run_cell2location_spatial_context import _cell2location_signatures, _run_spatial_mapping
        from src.discovery.target_discovery.cell2location_context import export_cell2location_abundance_from_adata

        cache_key = (str(reference_name), int(reference_epochs))
        if signature_cache is not None and cache_key in signature_cache:
            signatures = signature_cache[cache_key]
        else:
            signatures, _ = _cell2location_signatures(
                reference.copy(),
                label_key="cell2location_label",
                max_epochs=reference_epochs,
                batch_key=None,
            )
            if signature_cache is not None:
                signature_cache[cache_key] = signatures
        mapped = _run_spatial_mapping(
            query.copy(),
            signatures,
            n_cells_per_location=n_cells_per_location,
            detection_alpha=detection_alpha,
            max_epochs=mapping_epochs,
            max_genes=max_genes,
            train_batch_size=train_batch_size,
            posterior_batch_size=posterior_batch_size,
            posterior_num_samples=posterior_num_samples,
        )
        raw_table = export_cell2location_abundance_from_adata(
            mapped,
            output_dir / "_cell2location_raw",
            sample_id=str(dataset_spec["dataset_id"]),
            detection_alpha=detection_alpha,
        )
        table = standardize_cell2location_abundance(
            raw_table,
            dataset_id=str(dataset_spec["dataset_id"]),
            source_system=str(dataset_spec.get("truth_source_system", "osta")),
        )
        out_path = (
            output_dir
            / "abundance"
            / f"{dataset_spec['dataset_id']}_cell2location_{reference_name}_abundance.csv.gz"
        )
        write_csv_gz(table, out_path)
        return "ok", out_path, ""
    except Exception as exc:
        return f"failed:{type(exc).__name__}", None, str(exc)


def _shared_gene_count(reference, query) -> int:
    reference_genes = set(map(str, reference.var_names))
    query_genes = set(map(str, query.var_names))
    return int(len(reference_genes & query_genes))


def run_from_config(config: dict[str, Any], *, methods: list[str] | None = None, device: str | None = None, prepare_only: bool = False) -> int:
    output_dir = Path(config.get("output_dir", ROOT / "results" / "benchmarks" / "unified_spatial_annotation_poc"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["predictions", "abundance", "qc"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    selected_methods = methods or _parse_methods(config.get("methods"))
    selected_device = device or str(config.get("device", "cuda"))
    seed = int(config.get("random_seed", 42))
    max_cells_per_label = int(config.get("max_cells_per_label", 1000))
    max_query_cells = int(config.get("max_query_cells", 5000))
    max_genes = int(config.get("max_genes", 2000))
    hvae_epochs = int(config.get("hvae_epochs", 5))
    gpu_required = bool(config.get("gpu_required_for_cell2location", True))
    cell2location_train_batch_size = int(config.get("cell2location_train_batch_size", 4096))
    cell2location_posterior_batch_size = int(config.get("cell2location_posterior_batch_size", 4096))
    cell2location_posterior_num_samples = int(config.get("cell2location_posterior_num_samples", 1000))
    progress_path = Path(config["progress_path"]) if config.get("progress_path") else None
    _initialize_progress(progress_path, run_id=str(config.get("run_id", "unified_spatial_annotation_poc")))

    dictionary = build_unified_celltype_dictionary()
    dictionary.to_csv(output_dir / "unified_celltype_dictionary.csv", index=False)

    reference_manifest: list[dict[str, Any]] = []
    references: list[tuple[str, dict[str, Any], Any]] = []
    for spec in config.get("references", []):
        adata = read_reference_from_spec(spec, max_cells_per_label=max_cells_per_label, seed=seed)
        references.append((str(spec["name"]), spec, adata))
        reference_manifest.append(
            {
                "name": str(spec["name"]),
                "source": str(spec.get("source", "h5ad")),
                "source_system": str(spec.get("source_system", "osta")),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "labels": adata.obs["unified_level1"].astype(str).value_counts().to_dict(),
            }
        )
    _write_json(output_dir / "reference_manifest.json", reference_manifest)

    method_runs: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    cell2location_signature_cache: dict[tuple[str, int], pd.DataFrame] = {}
    if prepare_only:
        _write_json(
            output_dir / "annotation_manifest.json",
            {"run_id": config.get("run_id", "unified_spatial_annotation_poc"), "status": "prepared_only", "method_runs": []},
        )
        return 0

    for dataset_spec in config.get("datasets", []):
        dataset_id = str(dataset_spec["dataset_id"])
        assay_type = str(dataset_spec.get("assay_type", "whole_transcriptome"))
        query = read_query_from_spec(dataset_spec, max_query_cells=max_query_cells, seed=seed)
        for reference_name, reference_spec, reference in references:
            for method in selected_methods:
                run_row: dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "method": method,
                    "reference": reference_name,
                    "assay_type": assay_type,
                    "query_n_obs": int(query.n_obs),
                    "query_n_vars": int(query.n_vars),
                    "max_query_cells": int(max_query_cells),
                    "full_query": bool(max_query_cells <= 0),
                    "status": "started",
                    "prediction_path": "",
                    "abundance_path": "",
                    "message": "",
                }
                try:
                    if method == "singler":
                        result = run_prototype_label_transfer(
                            reference,
                            query,
                            dataset_id=dataset_id,
                            reference_name=reference_name,
                            assay_type=assay_type,
                            method="singler",
                            max_genes=max_genes,
                        )
                    elif method == "rctd":
                        label_key = str(dataset_spec.get("rctd_label_key") or "DeconLabel1")
                        rctd_mode = str(dataset_spec.get("rctd_mode", "existing"))
                        if rctd_mode.startswith("spacexr") and label_key not in query.obs:
                            status, message = _rctd_spacexr_doublet_status()
                            if status != "ready":
                                run_row.update({"status": status, "message": message})
                                method_runs.append(run_row)
                                continue
                            result = _spacexr_rctd_ready_run(
                                reference=reference,
                                query=query,
                                dataset_spec=dataset_spec,
                                reference_name=reference_name,
                                output_dir=output_dir,
                                max_genes=max_genes,
                                max_cores=int(config.get("rctd_max_cores", 4)),
                                test_mode=bool(config.get("rctd_test_mode", False)),
                                doublet_mode=str(dataset_spec.get("rctd_doublet_mode", config.get("rctd_doublet_mode", "doublet"))),
                                reference_min_umi=int(config.get("rctd_reference_min_umi", 100)),
                                umi_min=int(config.get("rctd_umi_min", 100)),
                                counts_min=int(config.get("rctd_counts_min", 10)),
                                umi_min_sigma=int(config.get("rctd_umi_min_sigma", 300)),
                                cell_min_instance=int(config.get("rctd_cell_min_instance", 25)),
                            )
                        else:
                            result = run_rctd_existing_annotation(
                                query,
                                dataset_id=dataset_id,
                                reference_name=reference_name,
                                assay_type=assay_type,
                                label_key=label_key,
                                secondary_label_key=dataset_spec.get("rctd_secondary_label_key", "DeconLabel2"),
                                class_key=dataset_spec.get("rctd_class_key", "DeconClass"),
                                source_system=str(dataset_spec.get("truth_source_system", "osta")),
                            )
                    elif method == "hvae":
                        result = run_hvae_label_transfer(
                            reference,
                            query,
                            dataset_id=dataset_id,
                            reference_name=reference_name,
                            assay_type=assay_type,
                            max_genes=max_genes,
                            epochs=hvae_epochs,
                            device=selected_device,
                            seed=seed,
                        )
                    elif method == "cell2location":
                        run_row.update(
                            {
                                "shared_genes": _shared_gene_count(reference, query),
                                "cell2location_reference_epochs": int(config.get("cell2location_reference_epochs", 250)),
                                "cell2location_mapping_epochs": int(config.get("cell2location_mapping_epochs", 30000)),
                                "cell2location_max_genes": int(max_genes),
                                "cell2location_n_cells_per_location": int(config.get("cell2location_n_cells_per_location", 30)),
                                "cell2location_detection_alpha": int(config.get("cell2location_detection_alpha", 20)),
                                "cell2location_train_batch_size": cell2location_train_batch_size,
                                "cell2location_posterior_batch_size": cell2location_posterior_batch_size,
                                "cell2location_posterior_num_samples": cell2location_posterior_num_samples,
                            }
                        )
                        _append_progress(
                            progress_path,
                            (
                                "cell2location preflight "
                                f"dataset={dataset_id} reference={reference_name} "
                                f"query_n_obs={query.n_obs} query_n_vars={query.n_vars} "
                                f"shared_genes={run_row['shared_genes']} max_query_cells={max_query_cells} "
                                f"train_batch_size={cell2location_train_batch_size} "
                                f"posterior_batch_size={cell2location_posterior_batch_size} "
                                f"posterior_num_samples={cell2location_posterior_num_samples}"
                            ),
                        )
                        status = cell2location_status_for_dataset(
                            dataset_id=dataset_id,
                            assay_type=assay_type,
                            device=selected_device,
                            gpu_required=gpu_required,
                        )
                        if not status.runnable:
                            run_row.update({"status": status.status, "message": status.message})
                            _append_progress(
                                progress_path,
                                f"cell2location skipped dataset={dataset_id} reference={reference_name} status={status.status}",
                            )
                            method_runs.append(run_row)
                            continue
                        cell_status, abundance_path, message = _cell2location_ready_run(
                            reference=reference,
                            query=query,
                            dataset_spec=dataset_spec,
                            reference_name=reference_name,
                            output_dir=output_dir,
                            reference_epochs=int(config.get("cell2location_reference_epochs", 250)),
                            mapping_epochs=int(config.get("cell2location_mapping_epochs", 30000)),
                            max_genes=max_genes,
                            n_cells_per_location=int(config.get("cell2location_n_cells_per_location", 30)),
                            detection_alpha=int(config.get("cell2location_detection_alpha", 20)),
                            train_batch_size=cell2location_train_batch_size,
                            posterior_batch_size=cell2location_posterior_batch_size,
                            posterior_num_samples=cell2location_posterior_num_samples,
                            signature_cache=cell2location_signature_cache,
                        )
                        run_row.update({"status": cell_status, "message": message})
                        if abundance_path is not None:
                            run_row["abundance_path"] = str(abundance_path)
                        _append_progress(
                            progress_path,
                            f"cell2location finished dataset={dataset_id} reference={reference_name} status={cell_status}",
                        )
                        method_runs.append(run_row)
                        continue
                    else:
                        run_row.update({"status": f"not_available:{method}", "message": "method is not implemented"})
                        method_runs.append(run_row)
                        continue

                    run_row["status"] = result.status
                    run_row["message"] = result.message
                    if result.predictions is not None:
                        prediction_path = output_dir / "predictions" / f"{dataset_id}_{method}_{reference_name}_labels.csv.gz"
                        write_csv_gz(result.predictions, prediction_path)
                        run_row["prediction_path"] = str(prediction_path)
                        qc_rows.append(
                            compute_prediction_qc(
                                result.predictions,
                                query,
                                truth_label_key=dataset_spec.get("truth_label_key"),
                                truth_source_system=str(dataset_spec.get("truth_source_system", "osta")),
                            )
                        )
                    if result.abundance is not None:
                        abundance_path = output_dir / "abundance" / f"{dataset_id}_{method}_{reference_name}_abundance.csv.gz"
                        write_csv_gz(result.abundance, abundance_path)
                        run_row["abundance_path"] = str(abundance_path)
                    method_runs.append(run_row)
                except Exception as exc:
                    run_row.update(
                        {
                            "status": f"failed:{type(exc).__name__}",
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if method == "cell2location":
                        _append_progress(
                            progress_path,
                            f"cell2location failed dataset={dataset_id} reference={reference_name} error={type(exc).__name__}",
                        )
                    method_runs.append(run_row)

    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(output_dir / "qc" / "unified_annotation_metrics.csv", index=False)
    else:
        pd.DataFrame(
            columns=["dataset_id", "method", "reference", "n_obs", "mean_confidence", "accuracy", "macro_f1", "ari", "nmi"]
        ).to_csv(output_dir / "qc" / "unified_annotation_metrics.csv", index=False)

    manifest = {
        "run_id": config.get("run_id", "unified_spatial_annotation_poc"),
        "status": "ready",
        "device": selected_device,
        "max_query_cells": max_query_cells,
        "cell2location_train_batch_size": cell2location_train_batch_size,
        "cell2location_posterior_batch_size": cell2location_posterior_batch_size,
        "cell2location_posterior_num_samples": cell2location_posterior_num_samples,
        "methods": selected_methods,
        "method_runs": method_runs,
    }
    _write_json(output_dir / "annotation_manifest.json", manifest)
    report = [
        "# Unified Spatial Annotation POC",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Device: `{selected_device}`",
        f"- Methods: `{', '.join(selected_methods)}`",
        f"- References: `{', '.join(item['name'] for item in reference_manifest)}`",
        "",
        "## Method Status",
        "",
        pd.DataFrame(method_runs).to_markdown(index=False) if method_runs else "_No method runs._",
        "",
        "## Notes",
        "",
        "- `rctd` standardizes existing RCTD/deconvolution labels when available, otherwise `rctd_mode: spacexr_*` runs `spacexr::run.RCTD`.",
        "- `rctd_mode: spacexr_doublet` is used for VisiumHD segmented cell-level data when no precomputed RCTD labels are present.",
        "- `cell2location` is gated for targeted panels and CUDA availability before training.",
        "- `hvae` is skipped when `hvae_epochs` is zero; otherwise it trains HyperSCA HVAE and transfers labels in latent space.",
    ]
    (output_dir / "unified_annotation_poc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    methods = _parse_methods(args.methods) if args.methods else None
    return run_from_config(config, methods=methods, device=args.device, prepare_only=args.prepare_only)


if __name__ == "__main__":
    raise SystemExit(main())
