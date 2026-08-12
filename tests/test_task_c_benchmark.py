from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.evaluation.benchmark_contract import contract_digest, load_benchmark_contract
from src.evaluation.task_c_benchmark import (
    MeanDifferenceNetworkBaseline,
    TaskCBenchmarkError,
    evaluate_task_c_scores,
    load_causalbench_npz,
    run_task_c_mean_difference,
    score_mean_difference_network,
    sha256_file,
)


CONTRACT_PATH = Path("configs/benchmark_contract_v1.json")


def _toy_inputs() -> tuple[np.ndarray, list[str], list[str]]:
    expression = np.asarray(
        [
            [1.0, 2.0, 5.0],
            [1.0, 2.0, 5.0],
            [0.0, 5.0, 5.0],
            [0.0, 7.0, 5.0],
            [1.0, 2.0, 8.0],
            [1.0, 2.0, 10.0],
        ]
    )
    interventions = [
        "non-targeting",
        "non-targeting",
        "A",
        "A",
        "B",
        "B",
    ]
    return expression, interventions, ["A", "B", "C"]


def test_mean_difference_scores_directed_effects_without_self_edges() -> None:
    expression, interventions, genes = _toy_inputs()

    result = score_mean_difference_network(
        expression,
        interventions,
        genes,
        min_cells_per_intervention=2,
    )

    assert result.summary["n_control_cells"] == 2
    assert result.summary["eligible_sources"] == ["A", "B"]
    assert result.summary["coverage"] == 1.0
    assert set(result.scores.columns) == {
        "source",
        "target",
        "effect",
        "score",
        "n_intervention",
        "n_control",
        "source_rank",
    }
    assert not (result.scores["source"] == result.scores["target"]).any()

    a_edges = result.scores[result.scores["source"] == "A"].reset_index(drop=True)
    assert a_edges["target"].tolist() == ["B", "C"]
    assert a_edges["effect"].tolist() == pytest.approx([4.0, 0.0])
    assert a_edges["score"].tolist() == pytest.approx([4.0, 0.0])
    assert a_edges["source_rank"].tolist() == [1, 2]


def test_sparse_and_dense_mean_difference_scores_match() -> None:
    expression, interventions, genes = _toy_inputs()

    dense = score_mean_difference_network(
        expression,
        interventions,
        genes,
        min_cells_per_intervention=2,
    )
    sparse_result = score_mean_difference_network(
        sparse.csr_matrix(expression),
        interventions,
        genes,
        min_cells_per_intervention=2,
    )

    pd.testing.assert_frame_equal(dense.scores, sparse_result.scores)
    assert dense.summary == sparse_result.summary


@pytest.mark.parametrize(
    "expression, interventions, genes, message",
    [
        (
            np.ones((2, 2)),
            ["A", "A"],
            ["A", "B"],
            "control cells",
        ),
        (
            np.ones((2, 2)),
            ["non-targeting", "A"],
            ["A", "A"],
            "unique",
        ),
        (
            np.asarray([[1.0, np.nan], [1.0, 2.0]]),
            ["non-targeting", "A"],
            ["A", "B"],
            "finite",
        ),
    ],
)
def test_mean_difference_rejects_invalid_inputs(
    expression,
    interventions,
    genes,
    message: str,
) -> None:
    with pytest.raises(TaskCBenchmarkError, match=message):
        score_mean_difference_network(
            expression,
            interventions,
            genes,
            min_cells_per_intervention=1,
        )


def test_causalbench_compatible_callable_returns_fixed_top_k_edges() -> None:
    expression, interventions, genes = _toy_inputs()
    baseline = MeanDifferenceNetworkBaseline(
        top_k_per_source=1,
        min_cells_per_intervention=2,
    )

    edges = baseline(
        expression,
        interventions,
        genes,
        training_regime="ignored-compatible-placeholder",
        seed=23,
    )

    assert edges == [("A", "B"), ("B", "C")]


def test_task_c_metrics_use_complete_directed_score_universe() -> None:
    expression, interventions, genes = _toy_inputs()
    result = score_mean_difference_network(
        expression,
        interventions,
        genes,
        min_cells_per_intervention=2,
    )

    metrics = evaluate_task_c_scores(
        result.scores,
        reference_edges={("A", "B"), ("B", "C")},
        precision_at_k=2,
    )

    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["n_reference_edges_in_universe"] == 2
    assert metrics["n_scored_edges"] == 4


def test_causalbench_npz_loader_and_hash_are_reproducible(tmp_path: Path) -> None:
    expression, interventions, genes = _toy_inputs()
    input_path = tmp_path / "dataset_k562.npz"
    np.savez(
        input_path,
        expression_matrix=expression,
        interventions=np.asarray(interventions),
        var_names=np.asarray(genes),
    )

    loaded_expression, loaded_interventions, loaded_genes = load_causalbench_npz(
        input_path
    )

    np.testing.assert_array_equal(loaded_expression, expression)
    assert loaded_interventions == interventions
    assert loaded_genes == genes
    digest = sha256_file(input_path)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert digest == sha256_file(input_path)


def test_task_c_run_writes_required_baseline_artifacts(tmp_path: Path) -> None:
    expression, interventions, genes = _toy_inputs()
    contract = load_benchmark_contract(CONTRACT_PATH)

    run = run_task_c_mean_difference(
        expression=expression,
        interventions=interventions,
        gene_names=genes,
        contract=contract,
        dataset_id="weissmann_k562_toy",
        dataset_source="generated:test_task_c_benchmark",
        context_id="K562",
        data_status="synthetic_smoke",
        input_digest="sha256:" + "a" * 64,
        code_revision="abc1234",
        random_seed=23,
        output_dir=tmp_path,
        reference_edges={("A", "B"), ("B", "C")},
        reference_id="toy_reference",
        reference_digest="sha256:" + "c" * 64,
        min_cells_per_intervention=2,
        precision_at_k=2,
    )

    required = contract["shared_design"]["required_run_artifacts"]
    assert all((tmp_path / name).exists() for name in required)
    assert run["metrics"]["average_precision"] == pytest.approx(1.0)
    assert run["input_summary"]["data_status"] == "synthetic_smoke"
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    decision = json.loads((tmp_path / "promotion_decision.json").read_text())
    assert manifest["contract_sha256"] == contract_digest(contract)
    assert manifest["method_id"] == "mean_difference"
    assert manifest["method_role"] == "simple_baseline"
    assert manifest["dataset_source"] == "generated:test_task_c_benchmark"
    assert manifest["input_artifacts"]["declared_reference"] == (
        "sha256:" + "c" * 64
    )
    assert decision["status"] == "not_applicable_simple_baseline"
    assert decision["claim_level"] == "baseline_only"
    assert decision["synthetic_smoke"] is True


def test_task_c_run_without_reference_stays_not_evaluated(tmp_path: Path) -> None:
    expression, interventions, genes = _toy_inputs()
    contract = load_benchmark_contract(CONTRACT_PATH)

    run = run_task_c_mean_difference(
        expression=expression,
        interventions=interventions,
        gene_names=genes,
        contract=contract,
        dataset_id="unscored",
        dataset_source="generated:test_task_c_benchmark",
        context_id="unknown",
        data_status="external_benchmark",
        input_digest="sha256:" + "b" * 64,
        code_revision="abc1234",
        random_seed=23,
        output_dir=tmp_path,
        min_cells_per_intervention=2,
    )

    assert run["metrics"]["status"] == "not_evaluated_no_reference"
    assert run["metrics"]["average_precision"] is None
    assert run["metrics"]["reference_id"] is None


def test_task_c_cli_runs_official_causalbench_npz_shape(tmp_path: Path) -> None:
    expression, interventions, genes = _toy_inputs()
    input_path = tmp_path / "dataset_k562.npz"
    np.savez(
        input_path,
        expression_matrix=expression,
        interventions=np.asarray(interventions),
        var_names=np.asarray(genes),
    )
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {"source": ["A", "B"], "target": ["B", "C"]}
    ).to_csv(reference_path, index=False)
    output_dir = tmp_path / "run"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_task_c_mean_difference.py",
            "--input-npz",
            str(input_path),
            "--dataset-id",
            "weissmann_k562_toy",
            "--dataset-source",
            "generated:test_task_c_benchmark",
            "--context-id",
            "K562",
            "--data-status",
            "synthetic_smoke",
            "--output-dir",
            str(output_dir),
            "--reference-edges",
            str(reference_path),
            "--reference-id",
            "toy_reference",
            "--code-revision",
            "test-revision",
            "--random-seed",
            "23",
            "--min-cells-per-intervention",
            "2",
            "--precision-at-k",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["status"] == "evaluated_against_declared_reference"
    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["data_status"] == "synthetic_smoke"
    assert (output_dir / "predictions.csv").exists()
