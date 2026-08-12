from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark_contract import contract_digest, load_benchmark_contract
from src.evaluation.task_s_benchmark import (
    TaskSBenchmarkError,
    evaluate_task_s_predictions,
    predict_task_s_baseline,
    run_task_s_baseline,
)


CONTRACT_PATH = Path("configs/benchmark_contract_v1.json")


def _toy_holdout() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["a0", "a1", "a2", "b0", "b1", "b2"],
            "sample_id": ["slice_2"] * 6,
            "spatial_block": ["block_c"] * 3 + ["block_d"] * 3,
            "perturbation_id": ["KO_A"] * 3 + ["KO_B"] * 3,
            "feature_id": ["program_1"] * 6,
            "distance": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "is_perturbed": [True, False, False, True, False, False],
            "own_effect_prediction": [2.0, 2.0, 2.0, -1.0, -1.0, -1.0],
            "observed_effect": [
                2.0,
                2.0 * np.exp(-1.0),
                2.0 * np.exp(-2.0),
                -1.0,
                -1.0 * np.exp(-1.0),
                -1.0 * np.exp(-2.0),
            ],
        }
    )


def test_own_only_and_fixed_decay_separate_own_and_neighbor_predictions() -> None:
    holdout = _toy_holdout()

    own_only = predict_task_s_baseline(holdout, baseline_id="own_only")
    decay = predict_task_s_baseline(
        holdout,
        baseline_id="fixed_distance_decay",
        length_scale=1.0,
    )

    own_mask = holdout["is_perturbed"].to_numpy()
    neighbor_mask = ~own_mask
    assert own_only.predictions.loc[own_mask, "predicted_effect"].tolist() == [
        2.0,
        -1.0,
    ]
    assert own_only.predictions.loc[neighbor_mask, "predicted_effect"].eq(0.0).all()
    np.testing.assert_allclose(
        decay.predictions["predicted_effect"],
        holdout["observed_effect"],
    )
    assert decay.summary["formula"] == "own_effect_prediction * exp(-distance / 1.0)"


def test_both_baselines_share_the_same_abstention_mask() -> None:
    holdout = _toy_holdout()
    holdout.loc[holdout["perturbation_id"] == "KO_B", "own_effect_prediction"] = np.nan

    own_only = predict_task_s_baseline(holdout, baseline_id="own_only")
    decay = predict_task_s_baseline(
        holdout,
        baseline_id="fixed_distance_decay",
        length_scale=1.0,
    )

    assert own_only.predictions["abstained"].tolist() == decay.predictions[
        "abstained"
    ].tolist()
    assert own_only.summary["coverage"] == pytest.approx(0.5)
    assert own_only.summary["abstention_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mutation, baseline_id, length_scale, message",
    [
        (
            lambda frame: frame.drop(columns=["spatial_block"]),
            "own_only",
            None,
            "missing required columns",
        ),
        (
            lambda frame: frame.assign(distance=-1.0),
            "own_only",
            None,
            "non-negative",
        ),
        (
            lambda frame: frame.assign(
                distance=np.where(frame["is_perturbed"], 1.0, frame["distance"])
            ),
            "own_only",
            None,
            "distance zero",
        ),
        (
            lambda frame: frame.assign(
                is_perturbed=[pd.NA, False, False, True, False, False]
            ),
            "own_only",
            None,
            "boolean",
        ),
        (
            lambda frame: frame,
            "fixed_distance_decay",
            None,
            "length_scale",
        ),
        (
            lambda frame: frame,
            "not_a_baseline",
            None,
            "baseline_id",
        ),
    ],
)
def test_task_s_predictions_fail_closed(
    mutation,
    baseline_id: str,
    length_scale: float | None,
    message: str,
) -> None:
    with pytest.raises(TaskSBenchmarkError, match=message):
        predict_task_s_baseline(
            mutation(_toy_holdout()),
            baseline_id=baseline_id,
            length_scale=length_scale,
        )


def test_task_s_metrics_keep_primary_neighbor_endpoint_separate() -> None:
    holdout = _toy_holdout()
    own_only = predict_task_s_baseline(holdout, baseline_id="own_only")
    decay = predict_task_s_baseline(
        holdout,
        baseline_id="fixed_distance_decay",
        length_scale=1.0,
    )

    own_metrics = evaluate_task_s_predictions(own_only.predictions)
    decay_metrics = evaluate_task_s_predictions(decay.predictions)

    assert own_metrics["own_effect_rmse"] == pytest.approx(0.0)
    assert own_metrics["neighbor_effect_rmse"] > 0.0
    assert decay_metrics["neighbor_effect_rmse"] == pytest.approx(0.0)
    assert decay_metrics["neighbor_effect_pcc"] == pytest.approx(1.0)
    assert decay_metrics["distance_decay_calibration_error"] == pytest.approx(0.0)
    assert decay_metrics["primary_metric"] == "neighbor_effect_rmse"


def test_distance_calibration_does_not_cancel_opposite_group_errors() -> None:
    holdout = _toy_holdout()
    result = predict_task_s_baseline(
        holdout,
        baseline_id="fixed_distance_decay",
        length_scale=1.0,
    )
    predictions = result.predictions.copy()
    neighbor = ~predictions["is_perturbed"]
    predictions.loc[
        neighbor & predictions["perturbation_id"].eq("KO_A"),
        "observed_effect",
    ] += 1.0
    predictions.loc[
        neighbor & predictions["perturbation_id"].eq("KO_B"),
        "observed_effect",
    ] -= 1.0

    metrics = evaluate_task_s_predictions(predictions)

    assert metrics["distance_decay_calibration_error"] == pytest.approx(1.0)


def test_task_s_run_writes_fixed_decay_artifact_bundle(tmp_path: Path) -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)

    run = run_task_s_baseline(
        holdout=_toy_holdout(),
        contract=contract,
        baseline_id="fixed_distance_decay",
        dataset_id="spatial_perturb_toy",
        dataset_source="generated:test_task_s_benchmark",
        data_status="synthetic_smoke",
        input_digest="sha256:" + "a" * 64,
        own_effect_source_id="toy_non_spatial_model",
        own_effect_source_digest="sha256:" + "b" * 64,
        length_scale=1.0,
        length_scale_source_id="training_slice_nn_distance",
        length_scale_source_digest="sha256:" + "c" * 64,
        train_only_attested=True,
        nonadjacent_blocks_attested=True,
        code_revision="abc1234",
        random_seed=23,
        output_dir=tmp_path,
    )

    required = contract["shared_design"]["required_run_artifacts"]
    assert all((tmp_path / name).exists() for name in required)
    assert run["metrics"]["neighbor_effect_rmse"] == pytest.approx(0.0)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    decision = json.loads((tmp_path / "promotion_decision.json").read_text())
    assert manifest["contract_sha256"] == contract_digest(contract)
    assert manifest["method_id"] == "fixed_distance_decay"
    assert manifest["input_artifacts"]["length_scale_source"] == (
        "sha256:" + "c" * 64
    )
    assert decision["status"] == "not_applicable_simple_baseline"
    assert decision["synthetic_smoke"] is True


def test_external_task_s_run_requires_split_and_training_attestations(
    tmp_path: Path,
) -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)

    with pytest.raises(TaskSBenchmarkError, match="attest"):
        run_task_s_baseline(
            holdout=_toy_holdout(),
            contract=contract,
            baseline_id="own_only",
            dataset_id="spatial_perturb_external",
            dataset_source="https://example.org/versioned-dataset",
            data_status="external_benchmark",
            input_digest="sha256:" + "a" * 64,
            own_effect_source_id="train_model",
            own_effect_source_digest="sha256:" + "b" * 64,
            train_only_attested=False,
            nonadjacent_blocks_attested=True,
            code_revision="abc1234",
            random_seed=23,
            output_dir=tmp_path,
        )


def test_task_s_cli_runs_one_baseline_from_canonical_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "holdout.csv"
    _toy_holdout().to_csv(input_path, index=False)
    length_source = tmp_path / "training_length_scale.json"
    length_source.write_text('{"length_scale": 1.0}\n')
    own_source = tmp_path / "own_effect_manifest.json"
    own_source.write_text('{"method": "toy"}\n')
    output_dir = tmp_path / "run"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_task_s_baseline.py",
            "--input-csv",
            str(input_path),
            "--baseline-id",
            "fixed_distance_decay",
            "--dataset-id",
            "spatial_perturb_toy",
            "--dataset-source",
            "generated:test_task_s_benchmark",
            "--data-status",
            "synthetic_smoke",
            "--own-effect-source-id",
            "toy_non_spatial_model",
            "--own-effect-source",
            str(own_source),
            "--length-scale",
            "1.0",
            "--length-scale-source-id",
            "training_slice_nn_distance",
            "--length-scale-source",
            str(length_source),
            "--attest-own-effect-train-only",
            "--attest-nonadjacent-spatial-blocks",
            "--code-revision",
            "test-revision",
            "--random-seed",
            "23",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["baseline_id"] == "fixed_distance_decay"
    assert summary["neighbor_effect_rmse"] == pytest.approx(0.0)
    assert summary["data_status"] == "synthetic_smoke"
    assert (output_dir / "predictions.csv").exists()
