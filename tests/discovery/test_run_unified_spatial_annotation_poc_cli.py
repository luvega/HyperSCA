from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]


def _write_h5ad(path: Path, rows: list[str], genes: list[str], values: list[list[float]], obs: pd.DataFrame) -> None:
    obs = obs.copy()
    obs.index = pd.Index(rows)
    var = pd.DataFrame(index=pd.Index(genes))
    adata = ad.AnnData(X=sparse.csr_matrix(np.asarray(values, dtype=np.float32)), obs=obs, var=var)
    if {"x", "y"}.issubset(obs.columns):
        adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.write_h5ad(path)


def test_rctd_spacexr_doublet_status_reports_missing_dependency(monkeypatch):
    from scripts import run_unified_spatial_annotation_poc as poc

    monkeypatch.setattr(poc, "_r_package_available", lambda package: False)

    status, message = poc._rctd_spacexr_doublet_status()

    assert status == "blocked:spacexr_unavailable"
    assert "spacexr" in message


def test_spacexr_rctd_run_exports_inputs_and_standardizes_outputs(tmp_path, monkeypatch):
    from scripts import run_unified_spatial_annotation_poc as poc

    genes = ["GENE1", "GENE2", "GENE3"]
    reference = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[100, 0, 0], [120, 0, 0], [0, 100, 0], [0, 120, 0]], dtype=np.float32)),
        obs=pd.DataFrame({"unified_level1": ["Tumor", "Tumor", "T_cells", "T_cells"]}, index=["r1", "r2", "r3", "r4"]),
        var=pd.DataFrame(index=genes),
    )
    query = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[80, 0, 0], [40, 40, 0]], dtype=np.float32)),
        obs=pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=["q1", "q2"]),
        var=pd.DataFrame(index=genes),
    )
    calls = []

    def fake_run(cmd, text, capture_output, check):
        calls.append(cmd)
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "obs_id": ["q1", "q2"],
                "spot_class": ["singlet", "doublet_certain"],
                "first_type": ["Tumor", "Tumor"],
                "second_type": ["Tumor", "T_cells"],
            }
        ).to_csv(out_dir / "results_df.csv.gz", index=False)
        pd.DataFrame(
            {
                "obs_id": ["q1", "q2"],
                "Tumor": [1.0, 0.55],
                "T_cells": [0.0, 0.45],
            }
        ).to_csv(out_dir / "abundance.csv.gz", index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(poc.subprocess, "run", fake_run)
    result = poc._spacexr_rctd_ready_run(
        reference=reference,
        query=query,
        dataset_spec={"dataset_id": "toy_visium", "assay_type": "whole_transcriptome", "rctd_mode": "spacexr_deconvolution"},
        reference_name="toy_reference",
        output_dir=tmp_path,
        max_genes=3,
        max_cores=1,
        test_mode=True,
        doublet_mode="doublet",
        reference_min_umi=1,
        umi_min=1,
        counts_min=0,
        umi_min_sigma=1,
        cell_min_instance=1,
    )

    assert result.status == "ok"
    assert calls and calls[0][0] == "Rscript"
    assert (tmp_path / "_rctd_inputs" / "toy_visium_toy_reference" / "reference_counts.mtx").exists()
    assert result.predictions is not None
    assert result.predictions["unified_level1"].tolist() == ["Tumor", "Tumor"]
    assert result.predictions["top2_label"].tolist() == ["Tumor", "T_cells"]
    assert result.abundance is not None
    assert result.abundance.loc[result.abundance["spot_id"].eq("q2"), "T_cells"].iloc[0] == 0.45


def test_cell2location_ready_run_reuses_reference_signatures(tmp_path, monkeypatch):
    from scripts import run_cell2location_spatial_context as c2l
    from scripts import run_unified_spatial_annotation_poc as poc
    from src.discovery.target_discovery import cell2location_context as ctx

    genes = ["GENE1", "GENE2"]
    reference = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[10, 0], [9, 0], [0, 8], [0, 7]], dtype=np.float32)),
        obs=pd.DataFrame({"cell2location_label": ["Tumor", "Tumor", "T_cells", "T_cells"]}, index=["r1", "r2", "r3", "r4"]),
        var=pd.DataFrame(index=genes),
    )
    query = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[5, 1], [0, 5]], dtype=np.float32)),
        obs=pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=["q1", "q2"]),
        var=pd.DataFrame(index=genes),
    )
    signature_calls = []

    def fake_signatures(adata_ref, *, label_key, max_epochs, batch_key):
        signature_calls.append((adata_ref.n_obs, label_key, max_epochs, batch_key))
        return pd.DataFrame({"Tumor": [1.0, 0.0], "T_cells": [0.0, 1.0]}, index=genes), adata_ref

    def fake_mapping(
        adata_vis,
        signatures,
        *,
        n_cells_per_location,
        detection_alpha,
        max_epochs,
        max_genes,
        train_batch_size,
        posterior_batch_size,
        posterior_num_samples,
    ):
        assert signatures.columns.tolist() == ["Tumor", "T_cells"]
        assert max_genes == 3
        assert train_batch_size == 128
        assert posterior_batch_size == 64
        assert posterior_num_samples == 25
        return adata_vis

    def fake_export(adata_vis, output_dir, *, sample_id, detection_alpha):
        return pd.DataFrame(
            {
                "spot_id": adata_vis.obs_names.astype(str),
                "sample_id": sample_id,
                "x": adata_vis.obs["x"].to_numpy(),
                "y": adata_vis.obs["y"].to_numpy(),
                "level3": "unknown",
                "Tumor": [0.8, 0.1],
                "T_cells": [0.2, 0.9],
            }
        )

    monkeypatch.setattr(c2l, "_cell2location_signatures", fake_signatures)
    monkeypatch.setattr(c2l, "_run_spatial_mapping", fake_mapping)
    monkeypatch.setattr(ctx, "export_cell2location_abundance_from_adata", fake_export)

    signature_cache = {}
    common = {
        "reference": reference,
        "reference_name": "toy_reference",
        "output_dir": tmp_path,
        "reference_epochs": 7,
        "mapping_epochs": 11,
        "max_genes": 3,
        "n_cells_per_location": 30,
        "detection_alpha": 20,
        "train_batch_size": 128,
        "posterior_batch_size": 64,
        "posterior_num_samples": 25,
        "signature_cache": signature_cache,
    }
    first = poc._cell2location_ready_run(
        query=query,
        dataset_spec={"dataset_id": "toy_visium_a", "assay_type": "whole_transcriptome"},
        **common,
    )
    second = poc._cell2location_ready_run(
        query=query,
        dataset_spec={"dataset_id": "toy_visium_b", "assay_type": "whole_transcriptome"},
        **common,
    )

    assert first[0] == "ok"
    assert second[0] == "ok"
    assert len(signature_calls) == 1


def test_cell2location_manifest_records_full_query_preflight(tmp_path, monkeypatch):
    from scripts import run_cell2location_spatial_context as c2l
    from scripts import run_unified_spatial_annotation_poc as poc
    from src.discovery.target_discovery import cell2location_context as ctx

    reference_path = tmp_path / "reference.h5ad"
    query_path = tmp_path / "query.h5ad"
    out_dir = tmp_path / "out"
    genes = ["GENE1", "GENE2", "GENE3"]
    _write_h5ad(
        reference_path,
        ["r1", "r2", "r3"],
        genes,
        [[10, 0, 0], [9, 0, 0], [0, 8, 0]],
        pd.DataFrame({"Level1": ["Tumor", "Tumor", "T cells"], "Level2": ["Tumor", "Tumor", "CD4 T cell"]}),
    )
    _write_h5ad(
        query_path,
        ["q1", "q2", "q3"],
        genes,
        [[5, 0, 1], [0, 6, 0], [3, 2, 1]],
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
    )

    def fake_signatures(adata_ref, *, label_key, max_epochs, batch_key):
        return pd.DataFrame({"Tumor": [1.0, 0.0], "T_cells": [0.0, 1.0]}, index=["GENE1", "GENE2"]), adata_ref

    def fake_mapping(
        adata_vis,
        signatures,
        *,
        n_cells_per_location,
        detection_alpha,
        max_epochs,
        max_genes,
        train_batch_size,
        posterior_batch_size,
        posterior_num_samples,
    ):
        assert adata_vis.n_obs == 3
        assert max_genes == 3
        assert train_batch_size == 128
        assert posterior_batch_size == 64
        assert posterior_num_samples == 25
        return adata_vis

    def fake_export(adata_vis, output_dir, *, sample_id, detection_alpha):
        return pd.DataFrame(
            {
                "spot_id": adata_vis.obs_names.astype(str),
                "sample_id": sample_id,
                "x": adata_vis.obs["x"].to_numpy(),
                "y": adata_vis.obs["y"].to_numpy(),
                "level3": "unknown",
                "Tumor": [0.8, 0.1, 0.5],
                "T_cells": [0.2, 0.9, 0.5],
            }
        )

    monkeypatch.setattr(c2l, "_cell2location_signatures", fake_signatures)
    monkeypatch.setattr(c2l, "_run_spatial_mapping", fake_mapping)
    monkeypatch.setattr(ctx, "export_cell2location_abundance_from_adata", fake_export)

    out_dir.mkdir()
    progress_path = out_dir / "progress.md"
    progress_path.write_text("# Existing checkpoint\n\nkeep this note\n", encoding="utf-8")

    config = {
        "run_id": "toy_cell2location_full",
        "output_dir": str(out_dir),
        "progress_path": str(progress_path),
        "random_seed": 7,
        "max_cells_per_label": 10,
        "max_query_cells": 0,
        "max_genes": 3,
        "device": "cpu",
        "gpu_required_for_cell2location": False,
        "methods": ["cell2location"],
        "cell2location_train_batch_size": 128,
        "cell2location_posterior_batch_size": 64,
        "cell2location_posterior_num_samples": 25,
        "references": [
            {
                "name": "toy_reference",
                "source": "h5ad",
                "h5ad_path": str(reference_path),
                "source_system": "osta",
                "label_key": "Level1",
                "fine_label_key": "Level2",
            }
        ],
        "datasets": [
            {
                "dataset_id": "toy_visiumhd",
                "h5ad_path": str(query_path),
                "assay_type": "whole_transcriptome",
            }
        ],
    }

    assert poc.run_from_config(config) == 0

    manifest = json.loads((out_dir / "annotation_manifest.json").read_text(encoding="utf-8"))
    row = manifest["method_runs"][0]
    assert row["status"] == "ok"
    assert row["query_n_obs"] == 3
    assert row["query_n_vars"] == 3
    assert row["max_query_cells"] == 0
    assert row["full_query"] is True
    assert row["shared_genes"] == 3
    assert row["cell2location_train_batch_size"] == 128
    assert row["cell2location_posterior_batch_size"] == 64
    assert row["cell2location_posterior_num_samples"] == 25
    progress = progress_path.read_text(encoding="utf-8")
    assert "keep this note" in progress
    assert "toy_visiumhd" in progress
    assert "query_n_obs=3" in progress
    assert "train_batch_size=128" in progress
    assert "posterior_num_samples=25" in progress


def test_unified_spatial_annotation_poc_cli_writes_manifest_predictions_and_qc(tmp_path):
    reference_path = tmp_path / "reference.h5ad"
    query_path = tmp_path / "query.h5ad"
    out_dir = tmp_path / "out"

    genes = ["EPCAM", "PTPRC", "COL1A1"]
    _write_h5ad(
        reference_path,
        ["r_tumor_1", "r_tumor_2", "r_t_1", "r_t_2"],
        genes,
        [[10, 0, 0], [9, 0, 1], [0, 8, 0], [0, 7, 1]],
        pd.DataFrame(
            {
                "Level1": ["Tumor", "Tumor", "T cells", "T cells"],
                "Level2": ["Tumor III", "Tumor III", "CD4 T cell", "CD8 Cytotoxic T cell"],
            }
        ),
    )
    _write_h5ad(
        query_path,
        ["q_tumor", "q_tcell"],
        genes,
        [[8, 0, 0], [0, 9, 0]],
        pd.DataFrame(
            {
                "truth": ["Tumor III", "CD4 T cell"],
                "DeconClass": ["singlet", "doublet_certain"],
                "DeconLabel1": ["Tumor III", "CD4 T cell"],
                "DeconLabel2": ["CAF", "Macrophage"],
                "x": [1.0, 2.0],
                "y": [1.0, 1.5],
            }
        ),
    )

    config = {
        "run_id": "toy",
        "output_dir": str(out_dir),
        "random_seed": 7,
        "max_cells_per_label": 10,
        "max_query_cells": 10,
        "max_genes": 3,
        "hvae_epochs": 0,
        "device": "cpu",
        "methods": ["rctd", "cell2location", "hvae"],
        "references": [
            {
                "name": "toy_reference",
                "source": "h5ad",
                "h5ad_path": str(reference_path),
                "source_system": "osta",
                "label_key": "Level1",
                "fine_label_key": "Level2",
            }
        ],
        "datasets": [
            {
                "dataset_id": "toy_visium",
                "h5ad_path": str(query_path),
                "assay_type": "whole_transcriptome",
                "truth_label_key": "truth",
                "rctd_label_key": "DeconLabel1",
                "rctd_secondary_label_key": "DeconLabel2",
                "rctd_class_key": "DeconClass",
            },
            {
                "dataset_id": "toy_xenium",
                "h5ad_path": str(query_path),
                "assay_type": "targeted_panel",
                "truth_label_key": "truth",
                "rctd_label_key": "DeconLabel1",
            },
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_unified_spatial_annotation_poc.py"),
            "--config",
            str(config_path),
            "--methods",
            "rctd,cell2location,hvae",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "annotation_manifest.json").read_text(encoding="utf-8"))
    statuses = {(row["dataset_id"], row["method"]): row["status"] for row in manifest["method_runs"]}
    assert statuses[("toy_visium", "rctd")] == "ok"
    assert statuses[("toy_visium", "hvae")] == "skipped:hvae_epochs_zero"
    assert statuses[("toy_xenium", "rctd")] == "not_applicable:targeted_panel_cell_level"
    assert statuses[("toy_xenium", "cell2location")] == "not_applicable:targeted_panel_cell_level"

    predictions = pd.read_csv(out_dir / "predictions" / "toy_visium_rctd_toy_reference_labels.csv.gz")
    assert predictions["unified_level1"].tolist() == ["Tumor", "T_cells"]
    assert predictions["confidence"].min() > 0.0

    abundance = pd.read_csv(out_dir / "abundance" / "toy_visium_rctd_toy_reference_abundance.csv.gz")
    assert {"spot_id", "sample_id", "x", "y", "level3", "Tumor", "T_cells"}.issubset(abundance.columns)

    qc = pd.read_csv(out_dir / "qc" / "unified_annotation_metrics.csv")
    assert {"dataset_id", "method", "reference", "macro_f1", "accuracy"}.issubset(qc.columns)
