from __future__ import annotations

import numpy as np
import pytest
import json
import hashlib
from pathlib import Path

from src.examples.methods_quickstart import make_hypersca_methods_example


def test_spatial_pilot_split_is_deterministic_disjoint_and_buffered() -> None:
    from src.evaluation.methods_pilot import split_spatial_train_tune

    example = make_hypersca_methods_example(seed=11)
    first = split_spatial_train_tune(example.positions, seed=11)
    second = split_spatial_train_tune(example.positions, seed=11)

    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.tune_indices, second.tune_indices)
    assert np.intersect1d(first.train_indices, first.tune_indices).size == 0
    assert len(first.buffer_indices) > 0
    assert len(first.train_indices) + len(first.tune_indices) + len(
        first.buffer_indices
    ) == len(example.positions)
    assert len(set(first.tune_block_ids)) >= 2
    for array in (first.train_indices, first.tune_indices, first.buffer_indices):
        assert array.flags.writeable is False


def test_spatial_hierarchy_uses_nested_coordinate_blocks_only() -> None:
    from src.evaluation.methods_pilot import (
        build_spatial_hierarchy_triplets,
        split_spatial_train_tune,
    )

    example = make_hypersca_methods_example(seed=11)
    split = split_spatial_train_tune(example.positions, seed=11)
    triplets = build_spatial_hierarchy_triplets(
        example.positions, split.train_indices, seed=11
    )

    assert triplets.shape[1] == 3
    assert len(triplets) > 0
    assert set(np.unique(triplets)).issubset(set(split.train_indices.tolist()))
    assert triplets.flags.writeable is False


def test_synthetic_spatial_pilot_fits_three_equal_capacity_models() -> None:
    from src.evaluation.methods_pilot import (
        SpatialPilotConfig,
        fit_spatial_pilot_models,
        split_spatial_train_tune,
    )

    example = make_hypersca_methods_example(seed=11, n_cells=128, n_genes=32)
    split = split_spatial_train_tune(
        example.positions, seed=11, grid_size=4, tune_span=1, buffer_width=0
    )
    result = fit_spatial_pilot_models(
        expression=example.expression,
        positions=example.positions,
        split=split,
        seed=11,
        config=SpatialPilotConfig(
            hidden_dim=16,
            latent_dim=4,
            maximum_epochs=2,
            early_stopping_patience=2,
            hierarchy_weight=0.1,
        ),
        device="cpu",
    )

    assert set(result.embeddings) == {
        "hypersca_hyperbolic",
        "euclidean_autoencoder",
        "hypersca_without_hierarchy_loss",
    }
    assert len(set(result.parameter_counts.values())) == 1
    assert set(result.parameter_counts.values()).pop() > 0
    for embedding in result.embeddings.values():
        assert embedding.shape == (len(split.tune_indices), 4)
        assert np.isfinite(embedding).all()
        assert embedding.flags.writeable is False
    assert result.hierarchy_loss_enabled == {
        "hypersca_hyperbolic": True,
        "euclidean_autoencoder": True,
        "hypersca_without_hierarchy_loss": False,
    }
    assert result.training_scopes == ("train", "tune_evaluation_only")


def test_osta_h5ad_pilot_writes_audit_only_seven_file_bundle(tmp_path) -> None:
    import anndata as ad
    from scipy import sparse

    from src.evaluation.methods_pilot import SpatialPilotConfig, run_osta_pilot_run
    import src.evaluation.run_evidence_publisher as publisher_module
    from src.evaluation.run_evidence_publisher import verify_run_evidence_bundle

    example = make_hypersca_methods_example(seed=11, n_cells=2048, n_genes=32)
    input_path = tmp_path / "benchmark.h5ad"
    dataset = ad.AnnData(
        X=sparse.csr_matrix(example.expression.astype(np.float32)),
    )
    dataset.var_names = list(example.gene_names)
    dataset.obs_names = [f"cell-{index}" for index in range(len(example.expression))]
    dataset.obsm["spatial"] = example.positions.astype(np.float32)
    dataset.write_h5ad(input_path)

    output = tmp_path / "pilot"
    record = run_osta_pilot_run(
        h5ad_path=input_path,
        dataset_id="synthetic_visium",
        platform_id="visium",
        output_dir=output,
        seed=11,
        config=SpatialPilotConfig(
            hidden_dim=16,
            latent_dim=4,
            maximum_epochs=2,
            early_stopping_patience=2,
            hierarchy_weight=0.1,
        ),
        device="cpu",
        maximum_cells=1024,
        maximum_genes=32,
    )

    assert record["status"] == "completed"
    assert record["promotion_eligible"] is False
    assert record["data_scopes"] == ["train", "tune"]
    expected = {
        "run_manifest.json",
        "method_status.json",
        "resource_usage.json",
        "primary_metric_units.csv",
        "primary_metric_summary.json",
        "secondary_metrics.csv",
        "claim_decision.json",
        "embeddings.npz",
    }
    assert {path.name for path in output.iterdir()} == expected
    primary = np.genfromtxt(
        output / "primary_metric_units.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    assert len(primary) >= 4
    assert len(set(primary["unit_id"].tolist())) >= 2
    verified = verify_run_evidence_bundle(output)
    assert verified.identity.data_split_seed == 19_911
    assert verified.identity.model_seed == 11
    assert verified.identity.statistical_unit_schema == (
        "osta_platform_sample_block_v1"
    )
    assert verified.identity.evidence_role == "pilot_audit_only"
    assert verified.statistical_unit_record["units"] == tuple(
        sorted(set(primary["unit_id"].tolist()))
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["split"]["split_seed"] == 19_911
    assert set(manifest["summary"]["code"]) == {
        "methods_pilot.py",
        "methods_protocol.py",
        "benchmark_evidence.py",
        "run_evidence_publisher.py",
    }
    assert all(
        len(record["sha256"]) == 64
        for record in manifest["summary"]["code"].values()
    )
    publisher_sha = hashlib.sha256(
        Path(publisher_module.__file__).read_bytes()
    ).hexdigest()
    assert (
        manifest["summary"]["publisher"]["source_sha256"] == publisher_sha
    )
    assert "release_holdout" not in (output / "run_manifest.json").read_text(
        encoding="utf-8"
    )
    second_output = tmp_path / "pilot_seed_23"
    run_osta_pilot_run(
        h5ad_path=input_path,
        dataset_id="synthetic_visium",
        platform_id="visium",
        output_dir=second_output,
        seed=23,
        config=SpatialPilotConfig(
            hidden_dim=16,
            latent_dim=4,
            maximum_epochs=2,
            early_stopping_patience=2,
            hierarchy_weight=0.1,
        ),
        device="cpu",
        maximum_cells=1024,
        maximum_genes=32,
    )
    second_manifest = json.loads(
        (second_output / "run_manifest.json").read_text(encoding="utf-8")
    )
    second_verified = verify_run_evidence_bundle(second_output)
    assert second_manifest["summary"]["split"] == manifest["summary"]["split"]
    assert (
        second_verified.identity.data_split_identity_sha256
        == verified.identity.data_split_identity_sha256
    )
    assert (
        second_verified.identity.statistical_unit_identity_sha256
        == verified.identity.statistical_unit_identity_sha256
    )
    assert second_verified.identity.run_identity_sha256 != verified.identity.run_identity_sha256
    with np.load(output / "embeddings.npz", allow_pickle=False) as first_arrays:
        first_tune = first_arrays["tune_global_indices"].copy()
    with np.load(second_output / "embeddings.npz", allow_pickle=False) as second_arrays:
        second_tune = second_arrays["tune_global_indices"].copy()
    assert np.array_equal(first_tune, second_tune)
    with pytest.raises(ValueError, match="already exists"):
        run_osta_pilot_run(
            h5ad_path=input_path,
            dataset_id="synthetic_visium",
            platform_id="visium",
            output_dir=output,
            seed=11,
            config=SpatialPilotConfig(maximum_epochs=1),
            device="cpu",
            maximum_cells=1024,
            maximum_genes=32,
        )
