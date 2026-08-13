from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_tuning import (
    TaskCTuningError,
    _benjamini_hochberg,
    build_tuning_response_edges,
    load_task_c_tuning_config,
    select_task_c_configuration,
)
from src.evaluation.task_c_data import (
    TaskCDataset,
    build_shared_task_c_split,
    materialize_task_c_split,
)
from src.evaluation.task_c_profile_input import materialize_task_c_profile_input
from src.evaluation.task_c_method_run import TaskCMethodRunError, run_task_c_method
from src.evaluation.task_c_predictions import normalize_task_c_predictions
from tests.test_task_c_data import dataset_for_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/task_c_tuning_v1.json"


def _complete_predictions(
    genes: tuple[str, ...], scores: dict[tuple[str, str], float]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "score": scores.get((source, target), 0.1),
                "returned_by_method": True,
            }
            for source in genes
            for target in genes
            if source != target
        ]
    )


def _response_data() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    genes = ("A", "B", "C")
    controls = np.zeros((10, 3), dtype=float)
    a_cells = np.tile([0.0, 8.0, 0.0], (10, 1))
    c_cells = np.tile([0.0, 0.0, 8.0], (10, 1))
    return (
        np.vstack((controls, a_cells, c_cells)),
        np.asarray(["non-targeting"] * 10 + ["A"] * 10 + ["C"] * 10),
        genes,
    )


def test_frozen_tuning_config_has_closed_fixed_policy() -> None:
    config = load_task_c_tuning_config(CONFIG)

    assert config.maximum_trials_per_method == 20
    assert config.q_value_threshold == 0.1
    assert config.final_holdout_allowed is False
    assert config.external_biological_references_allowed is False
    with pytest.raises(FrozenInstanceError):
        config.maximum_trials_per_method = 19  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda text: text.replace('"schema_version": "1.0",', '"schema_version": "1.0",\n  "schema_version": "1.0",'), "duplicate"),
        (lambda text: text.replace('"maximum_trials_per_method": 20', '"maximum_trials_per_method": true'), "twenty"),
        (lambda text: text.replace('"q_value_threshold": 0.1', '"q_value_threshold": NaN'), "finite"),
        (lambda text: text.replace('"final_holdout_allowed": false', '"final_holdout_allowed": true'), "holdout"),
        (lambda text: text.replace('\n}', ',\n  "extra": 1\n}'), "fields"),
    ],
)
def test_tuning_config_rejects_schema_drift(
    tmp_path: Path, mutation: object, message: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(mutation(CONFIG.read_text(encoding="utf-8")), encoding="utf-8")  # type: ignore[operator]

    with pytest.raises(TaskCTuningError, match=message):
        load_task_c_tuning_config(path)


def test_tuning_edges_use_only_allowed_response_cells_and_all_tests() -> None:
    expression, labels, genes = _response_data()

    edges = build_tuning_response_edges(
        expression,
        labels,
        genes,
        eligible_sources={"A"},
        q_value_threshold=0.1,
    )

    assert isinstance(edges, frozenset)
    assert ("A", "B") in edges
    assert all(source == "A" for source, _ in edges)


def test_benjamini_hochberg_uses_the_complete_test_family() -> None:
    adjusted = _benjamini_hochberg(np.asarray([0.01, 0.04, 0.03, 0.002]))

    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.008])


@pytest.mark.parametrize(
    ("expression", "labels", "genes", "eligible", "message"),
    [
        (np.zeros((9, 2)), ["non-targeting"] * 4 + ["A"] * 5, ["A", "B"], {"A"}, "control"),
        (np.zeros((9, 2)), ["non-targeting"] * 5 + ["A"] * 4, ["A", "B"], {"A"}, "five"),
        (np.zeros((10, 2)), ["non-targeting"] * 5 + ["A"] * 5, ["A", "A"], {"A"}, "unique"),
        (np.zeros((10, 2)), ["non-targeting"] * 5 + ["A"] * 5, ["A", "B"], {"excluded"}, "eligible"),
        (np.zeros((10, 2)), ["non-targeting"] * 5 + ["A"] * 5, ["A", "B"], {"B"}, "observed"),
        (np.asarray([[0.0, np.inf]] * 10), ["non-targeting"] * 5 + ["A"] * 5, ["A", "B"], {"A"}, "finite"),
        (np.zeros((10, 2), dtype=object), ["non-targeting"] * 5 + ["A"] * 5, ["A", "B"], {"A"}, "numeric"),
        (np.zeros((10, 3)), ["non-targeting"] * 5 + ["A"] * 5, ["A", "B"], {"A"}, "shape"),
        (np.zeros((10, 2)), ["non-targeting"] * 5 + ["A"] * 5, ["A", "E\u0301"], {"A"}, "canonical"),
    ],
)
def test_tuning_edge_inputs_fail_closed(
    expression: object,
    labels: object,
    genes: object,
    eligible: object,
    message: str,
) -> None:
    with pytest.raises(TaskCTuningError, match=message):
        build_tuning_response_edges(
            expression,  # type: ignore[arg-type]
            labels,  # type: ignore[arg-type]
            genes,  # type: ignore[arg-type]
            eligible_sources=eligible,  # type: ignore[arg-type]
            q_value_threshold=0.1,
        )


def test_constant_responses_are_handled_without_false_positive() -> None:
    expression = np.zeros((10, 2), dtype=float)
    labels = np.asarray(["non-targeting"] * 5 + ["A"] * 5)

    with pytest.raises(TaskCTuningError, match="positive"):
        build_tuning_response_edges(
            expression,
            labels,
            ("A", "B"),
            eligible_sources={"A"},
            q_value_threshold=0.1,
        )


def test_configuration_selection_maximizes_ap_and_breaks_ties_by_index() -> None:
    genes = ("A", "B", "C")
    first = _complete_predictions(genes, {("A", "B"): 0.9})
    second = _complete_predictions(genes, {("A", "B"): 0.9})

    selected = select_task_c_configuration(
        [
            (7, {"lambda": [2, {"value": 1.0}]}, second),
            (2, {"lambda": [1, {"value": 1.0}]}, first),
        ],
        tuning_edges={("A", "B")},
        maximum_trials=20,
    )

    assert selected["selected_trial_index"] == 2
    assert selected["average_precision"] == pytest.approx(1.0)
    assert selected["completed_trial_count"] == 2
    with pytest.raises(TypeError):
        selected.selected_parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        selected.selected_parameters["lambda"][1]["value"] = 2  # type: ignore[index]


def test_exact_score_ties_are_unranked_not_alphabetically_favored() -> None:
    genes = ("A", "B", "C")
    tied = _complete_predictions(genes, {})
    tied["score"] = 1.0

    first = select_task_c_configuration(
        [(0, {}, tied.sample(frac=1.0, random_state=1))],
        tuning_edges={("A", "B"), ("C", "B")},
        maximum_trials=20,
    )
    second = select_task_c_configuration(
        [(0, {}, tied.sample(frac=1.0, random_state=2))],
        tuning_edges={("A", "B"), ("C", "B")},
        maximum_trials=20,
    )

    assert first["average_precision"] == pytest.approx(2 / 6)
    assert first["average_precision"] == second["average_precision"]


@pytest.mark.parametrize(
    ("trials", "maximum", "edges", "message"),
    [
        ([], 20, {("A", "B")}, "at least one"),
        ([(True, {}, pd.DataFrame())], 20, {("A", "B")}, "trial index"),
        ([(0, {}, pd.DataFrame()), (0, {}, pd.DataFrame())], 20, {("A", "B")}, "unique"),
        ([(0, {}, pd.DataFrame())], True, {("A", "B")}, "twenty"),
        ([(0, {}, pd.DataFrame())], 19, {("A", "B")}, "twenty"),
        ([(index, {}, pd.DataFrame()) for index in range(21)], 20, {("A", "B")}, "twenty"),
        (
            [(0, {}, _complete_predictions(("A", "B", "C"), {}))],
            20,
            {("A", "B"), ("X", "Y")},
            "subset",
        ),
    ],
)
def test_selection_rejects_invalid_trial_set(
    trials: object, maximum: object, edges: object, message: str
) -> None:
    with pytest.raises(TaskCTuningError, match=message):
        select_task_c_configuration(
            trials,  # type: ignore[arg-type]
            tuning_edges=edges,  # type: ignore[arg-type]
            maximum_trials=maximum,  # type: ignore[arg-type]
        )


def test_selection_requires_same_complete_relation_universe() -> None:
    complete = _complete_predictions(("A", "B", "C"), {("A", "B"): 1.0})
    incomplete = complete.iloc[:-1].copy()

    with pytest.raises(TaskCTuningError, match="complete"):
        select_task_c_configuration(
            [(0, {}, complete), (1, {}, incomplete)],
            tuning_edges={("A", "B")},
            maximum_trials=20,
        )


def test_selection_rejects_complete_but_unknown_gene_universe() -> None:
    alternate = _complete_predictions(("A", "B", "Z"), {("A", "B"): 1.0})

    with pytest.raises(TaskCTuningError, match="complete"):
        select_task_c_configuration(
            [(0, {}, alternate)],
            tuning_edges={("A", "B")},
            maximum_trials=20,
            gene_names=("A", "B", "C"),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(score=-1.0), "non-negative"),
        (lambda frame: frame.assign(score=np.inf), "finite"),
        (lambda frame: frame.assign(score=True), "numeric"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (lambda frame: frame.assign(extra=1), "columns"),
        (lambda frame: frame.assign(source=lambda value: value["source"].where(value.index != 0, "Z")), "complete"),
        (lambda frame: frame.assign(returned_by_method="yes"), "returned"),
    ],
)
def test_selection_rejects_invalid_prediction_table(
    mutate: object, message: str
) -> None:
    frame = _complete_predictions(("A", "B", "C"), {("A", "B"): 1.0})

    with pytest.raises(TaskCTuningError, match=message):
        select_task_c_configuration(
            [(0, {}, mutate(frame))],  # type: ignore[operator]
            tuning_edges={("A", "B")},
            maximum_trials=20,
        )


def _write_npz(path: Path) -> None:
    expression, labels, genes = _response_data()
    np.savez_compressed(
        path,
        expression_matrix=expression,
        interventions=labels,
        var_names=np.asarray(genes),
    )


def _write_smoke_trial(path: Path, *, index: int, method: str = "demo") -> None:
    path.mkdir()
    _complete_predictions(("A", "B", "C"), {("A", "B"): 0.9}).to_csv(
        path / "predictions.csv", index=False
    )
    (path / "trial_parameters.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "trial_index": index,
                "method_id": method,
                "condition": "within_environment",
                "profile": "connection",
                "parameters": {"strength": index + 1},
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/select_task_c_configuration.py"), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_synthetic_smoke_is_explicit_atomic_and_deterministic(tmp_path: Path) -> None:
    tune = tmp_path / "tune.npz"
    _write_npz(tune)
    trial = tmp_path / "trial"
    _write_smoke_trial(trial, index=0)
    output = tmp_path / "selected.json"

    completed = _run_cli(
        "--tune-npz",
        str(tune),
        "--trial-dir",
        str(trial),
        "--output-json",
        str(output),
        "--config",
        str(CONFIG),
        "--synthetic-smoke",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_trial_index"] == 0
    assert payload["evidence"]["data_status"] == "synthetic_smoke"
    assert payload["external_biological_references_used"] is False
    assert payload["final_holdout_used"] is False
    assert payload["evidence"]["code_sha256"].startswith("sha256:")
    assert payload["selection_record_sha256"].startswith("sha256:")
    assert payload["evidence"]["trials"][0]["predictions_sha256"].startswith(
        "sha256:"
    )
    status = json.loads(Path(f"{output}.status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed_selection"
    assert status["selection_record_sha256"] == payload["selection_record_sha256"]
    rerun = _run_cli(
        "--tune-npz",
        str(tune),
        "--trial-dir",
        str(trial),
        "--output-json",
        str(output),
        "--config",
        str(CONFIG),
        "--synthetic-smoke",
    )
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr


def test_cli_does_not_accept_unregistered_formal_input(tmp_path: Path) -> None:
    tune = tmp_path / "tune.npz"
    _write_npz(tune)
    trial = tmp_path / "trial"
    _write_smoke_trial(trial, index=0)

    completed = _run_cli(
        "--tune-npz",
        str(tune),
        "--trial-dir",
        str(trial),
        "--output-json",
        str(tmp_path / "selected.json"),
        "--config",
        str(CONFIG),
    )

    assert completed.returncode != 0
    assert "public manifest" in completed.stderr


def test_cli_no_positive_relations_writes_identified_failure_status(
    tmp_path: Path,
) -> None:
    tune = tmp_path / "constant-tune.npz"
    np.savez_compressed(
        tune,
        expression_matrix=np.zeros((30, 3), dtype=float),
        interventions=np.asarray(
            ["non-targeting"] * 10 + ["A"] * 10 + ["C"] * 10
        ),
        var_names=np.asarray(("A", "B", "C")),
    )
    trial = tmp_path / "trial"
    _write_smoke_trial(trial, index=0)
    output = tmp_path / "selection.json"

    completed = _run_cli(
        "--tune-npz", str(tune),
        "--trial-dir", str(trial),
        "--output-json", str(output),
        "--config", str(CONFIG),
        "--synthetic-smoke",
    )

    assert completed.returncode != 0
    assert not output.exists()
    status = json.loads(Path(f"{output}.status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_selection"
    assert status["condition"] == "synthetic_smoke"
    assert status["tune_input_sha256"].startswith("sha256:")
    assert status["reason_category"] == "no_public_tuning_relations"


def test_cli_rejects_symlink_hardlink_and_tampered_trial_inputs(tmp_path: Path) -> None:
    tune = tmp_path / "tune.npz"
    _write_npz(tune)
    trial = tmp_path / "trial"
    _write_smoke_trial(trial, index=0)
    outside = tmp_path / "outside.csv"
    outside.write_bytes((trial / "predictions.csv").read_bytes())
    (trial / "predictions.csv").unlink()
    (trial / "predictions.csv").symlink_to(outside)

    symlinked = _run_cli(
        "--tune-npz", str(tune), "--trial-dir", str(trial),
        "--output-json", str(tmp_path / "one.json"), "--config", str(CONFIG),
        "--synthetic-smoke",
    )
    assert symlinked.returncode != 0
    assert "symbolic" in symlinked.stderr

    (trial / "predictions.csv").unlink()
    os.link(outside, trial / "predictions.csv")
    hardlinked = _run_cli(
        "--tune-npz", str(tune), "--trial-dir", str(trial),
        "--output-json", str(tmp_path / "two.json"), "--config", str(CONFIG),
        "--synthetic-smoke",
    )
    assert hardlinked.returncode != 0
    assert "hard link" in hardlinked.stderr


def test_cli_rejects_more_than_twenty_trial_arguments(tmp_path: Path) -> None:
    tune = tmp_path / "tune.npz"
    _write_npz(tune)
    arguments = ["--tune-npz", str(tune), "--output-json", str(tmp_path / "out.json"), "--config", str(CONFIG), "--synthetic-smoke"]
    for index in range(21):
        trial = tmp_path / f"trial-{index}"
        _write_smoke_trial(trial, index=index)
        arguments.extend(("--trial-dir", str(trial)))

    completed = _run_cli(*arguments)

    assert completed.returncode != 0
    assert "twenty" in completed.stderr


def test_cli_rejects_unrecognized_condition_or_profile(tmp_path: Path) -> None:
    tune = tmp_path / "tune.npz"
    _write_npz(tune)
    trial = tmp_path / "trial"
    _write_smoke_trial(trial, index=0)
    parameters_path = trial / "trial_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameters["condition"] = "invented_condition"
    parameters_path.write_text(json.dumps(parameters) + "\n", encoding="utf-8")

    completed = _run_cli(
        "--tune-npz", str(tune), "--trial-dir", str(trial),
        "--output-json", str(tmp_path / "out.json"), "--config", str(CONFIG),
        "--synthetic-smoke",
    )

    assert completed.returncode != 0
    assert "condition" in completed.stderr


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _selection_record_sha256(payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _dataset_with_public_tune_controls(context_id: str) -> TaskCDataset:
    original = dataset_for_split(context_id)
    extra_controls = np.zeros((15, len(original.gene_names)), dtype=np.float32)
    return TaskCDataset(
        expression=np.vstack((extra_controls, original.expression.copy())),
        interventions=np.asarray(
            ["non-targeting"] * 15 + original.interventions.tolist(), dtype=str
        ),
        gene_names=original.gene_names,
        context_id=context_id,
        source_path=Path(f"{context_id}-with-controls.npz"),
        source_sha256=f"sha256:{context_id}-with-controls",
    )


def _run_bound_direct_mean_trial(
    tmp_path: Path,
    *,
    training: Path,
    public: Path,
    trial_index: int,
) -> Path:
    candidate = tmp_path / f"direct-candidate-{trial_index}.json"
    candidate.write_text(
        json.dumps(
            {"schema_version": "1.0", "trial_index": trial_index, "parameters": {}}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / f"direct-trial-{trial_index}"
    run_task_c_method(
        method_id="mean_difference",
        input_npz=training,
        output_dir=output,
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id="k562",
        min_cells=5,
        public_manifest_path=public,
        trial_parameters_path=candidate,
        project_root=ROOT,
    )
    return output


def test_cli_formal_mode_binds_public_tune_and_completed_run(tmp_path: Path) -> None:
    k562 = _dataset_with_public_tune_controls("k562")
    rpe1 = _dataset_with_public_tune_controls("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    tune = Path(bundle["within"]["k562"]["tune"])
    train = Path(bundle["within"]["k562"]["train"])
    public = Path(bundle["public_manifest"])
    trial = _run_bound_direct_mean_trial(
        tmp_path, training=train, public=public, trial_index=0
    )

    completed = _run_cli(
        "--tune-npz", str(tune),
        "--public-manifest", str(public),
        "--trial-dir", str(trial),
        "--trial-input", f"{trial}={train}",
        "--output-json", str(tmp_path / "selected.json"),
        "--config", str(CONFIG),
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "selected.json").read_text(encoding="utf-8"))
    assert result["evidence"]["data_status"] == "external_benchmark"
    assert result["evidence"]["public_manifest_sha256"].startswith("sha256:")

    predictions = pd.read_csv(trial / "predictions.csv")
    predictions.loc[0, "score"] = 0.123
    predictions.to_csv(trial / "predictions.csv", index=False)
    tampered = _run_cli(
        "--tune-npz", str(tune),
        "--public-manifest", str(public),
        "--trial-dir", str(trial),
        "--trial-input", f"{trial}={train}",
        "--output-json", str(tmp_path / "tampered.json"),
        "--config", str(CONFIG),
    )
    assert tampered.returncode != 0
    assert "reconstruction" in tampered.stderr or "artifact" in tampered.stderr
    failed_status = json.loads(
        Path(f"{tmp_path / 'tampered.json'}.status.json").read_text(encoding="utf-8")
    )
    assert failed_status["status"] == "failed_selection"
    assert failed_status["selection_record_sha256"] is None
    assert failed_status["reason_category"] == "invalid_trial_bundle"

    raw = pd.read_csv(trial / "raw_predictions.csv")
    raw.loc[0, "score"] = 7.0
    raw.to_csv(trial / "raw_predictions.csv", index=False)
    normalize_task_c_predictions(raw, tuple(k562.gene_names)).to_csv(
        trial / "predictions.csv", index=False
    )
    status_path = trial / "method_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    for relative in ("raw_predictions.csv", "predictions.csv"):
        payload = (trial / relative).read_bytes()
        status_payload["artifacts"][relative] = {
            "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            "size_bytes": len(payload),
        }
    status_payload.pop("status_content_sha256")
    status_payload["status_content_sha256"] = _canonical_sha256(status_payload)
    status_path.write_text(json.dumps(status_payload) + "\n", encoding="utf-8")
    synchronized = _run_cli(
        "--tune-npz", str(tune),
        "--public-manifest", str(public),
        "--trial-dir", str(trial),
        "--trial-input", f"{trial}={train}",
        "--output-json", str(tmp_path / "synchronized-tamper.json"),
        "--config", str(CONFIG),
    )
    assert synchronized.returncode != 0
    assert "scientific semantics" in synchronized.stderr


def test_cli_formal_mode_rejects_refit_partition_and_unverified_trial(
    tmp_path: Path,
) -> None:
    k562 = _dataset_with_public_tune_controls("k562")
    rpe1 = _dataset_with_public_tune_controls("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    refit = Path(bundle["within"]["k562"]["refit"])
    public = Path(bundle["public_manifest"])
    trial = tmp_path / "trial"
    _write_smoke_trial(trial, index=0)

    completed = _run_cli(
        "--tune-npz", str(refit),
        "--public-manifest", str(public),
        "--trial-dir", str(trial),
        "--output-json", str(tmp_path / "selected.json"),
        "--config", str(CONFIG),
    )

    assert completed.returncode != 0
    assert "tune partition" in completed.stderr


def test_cli_formal_mode_rejects_non_task_c_environment_schema(tmp_path: Path) -> None:
    k562 = _dataset_with_public_tune_controls("k562")
    rpe1 = _dataset_with_public_tune_controls("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    tune = Path(bundle["within"]["k562"]["tune"])
    train = Path(bundle["within"]["k562"]["train"])
    public = Path(bundle["public_manifest"])
    trial = _run_bound_direct_mean_trial(
        tmp_path, training=train, public=public, trial_index=0
    )
    environment_path = trial / "environment_manifest.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["unexpected"] = True
    environment_path.write_text(json.dumps(environment) + "\n", encoding="utf-8")
    status_path = trial / "method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    environment_bytes = environment_path.read_bytes()
    status["artifacts"]["environment_manifest.json"] = {
        "sha256": f"sha256:{hashlib.sha256(environment_bytes).hexdigest()}",
        "size_bytes": len(environment_bytes),
    }
    status.pop("status_content_sha256")
    status["status_content_sha256"] = _canonical_sha256(status)
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

    completed = _run_cli(
        "--tune-npz", str(tune),
        "--public-manifest", str(public),
        "--trial-dir", str(trial),
        "--trial-input", f"{trial}={train}",
        "--output-json", str(tmp_path / "selected.json"),
        "--config", str(CONFIG),
    )

    assert completed.returncode != 0
    assert "environment" in completed.stderr and "changed" in completed.stderr


def _run_bound_mean_trial(
    tmp_path: Path,
    *,
    bundle: dict[str, object],
    profile: dict[str, str],
    context_id: str,
    trial_index: int,
) -> Path:
    candidate = tmp_path / f"candidate-{trial_index}.json"
    candidate.write_text(
        json.dumps(
            {"schema_version": "1.0", "trial_index": trial_index, "parameters": {}}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / f"bound-trial-{trial_index}"
    run_task_c_method(
        method_id="mean_difference",
        input_npz=Path(profile["input_npz"]),
        derived_input_manifest_path=Path(profile["manifest"]),
        output_dir=output,
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id=context_id,
        min_cells=5,
        public_manifest_path=Path(bundle["public_manifest"]),
        trial_parameters_path=candidate,
        project_root=ROOT,
    )
    return output


def test_cli_selects_train_fitted_trials_using_separate_tune_responses(
    tmp_path: Path,
) -> None:
    k562 = _dataset_with_public_tune_controls("k562")
    rpe1 = _dataset_with_public_tune_controls("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    train_profile = materialize_task_c_profile_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="k562",
        stage="train",
        output_dir=tmp_path / "profile-train",
    )
    tune_profile = materialize_task_c_profile_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="k562",
        stage="tune",
        output_dir=tmp_path / "profile-tune",
    )
    first = _run_bound_mean_trial(
        tmp_path,
        bundle=bundle,
        profile=train_profile,
        context_id="k562",
        trial_index=0,
    )
    second = _run_bound_mean_trial(
        tmp_path,
        bundle=bundle,
        profile=train_profile,
        context_id="k562",
        trial_index=1,
    )

    completed = _run_cli(
        "--tune-npz", tune_profile["input_npz"],
        "--profile-manifest", tune_profile["manifest"],
        "--public-manifest", str(bundle["public_manifest"]),
        "--trial-dir", str(second),
        "--trial-dir", str(first),
        "--trial-input", f"{first}={train_profile['input_npz']}",
        "--trial-input", f"{second}={train_profile['input_npz']}",
        "--trial-profile-manifest", f"{first}={train_profile['manifest']}",
        "--trial-profile-manifest", f"{second}={train_profile['manifest']}",
        "--output-json", str(tmp_path / "selected.json"),
        "--config", str(CONFIG),
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "selected.json").read_text(encoding="utf-8"))
    assert result["selected_trial_index"] == 0
    assert result["condition"] == "within_environment"
    assert result["profile"] == "connection"
    assert result["context_id"] == "k562"
    assert result["direction"] is None
    assert result["stage"] == "tune"
    assert result["training_and_tuning_inputs_separate"] is True
    assert result["evidence"]["tune_input_sha256"] != (
        result["evidence"]["training_input_sha256s"][0]
    )
    assert result["evidence"]["training_input_sha256s"] == [
        json.loads(
            (first / "trial_parameters.json").read_text(encoding="utf-8")
        )["training_input_sha256"]
    ]
    assert {trial["seed"] for trial in result["evidence"]["trials"]} == {11}
    assert not any(path.name == "trial_parameters.json" for path in tmp_path.glob("*.json"))

    refit_profile = materialize_task_c_profile_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="k562",
        stage="refit",
        output_dir=tmp_path / "profile-refit",
    )
    refit_output = tmp_path / "selected-refit"
    run_task_c_method(
        method_id="mean_difference",
        input_npz=Path(refit_profile["input_npz"]),
        derived_input_manifest_path=Path(refit_profile["manifest"]),
        output_dir=refit_output,
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id="k562",
        min_cells=5,
        public_manifest_path=Path(bundle["public_manifest"]),
        selection_record_path=tmp_path / "selected.json",
        selection_status_path=Path(f"{tmp_path / 'selected.json'}.status.json"),
        selection_tune_input_path=Path(tune_profile["input_npz"]),
        selection_tune_profile_manifest_path=Path(tune_profile["manifest"]),
        selection_config_path=CONFIG,
        selection_trial_directories=(second, first),
        selection_trial_input_bindings={
            first.resolve(): Path(train_profile["input_npz"]),
            second.resolve(): Path(train_profile["input_npz"]),
        },
        selection_trial_profile_bindings={
            first.resolve(): Path(train_profile["manifest"]),
            second.resolve(): Path(train_profile["manifest"]),
        },
        project_root=ROOT,
    )
    refit_environment = json.loads(
        (refit_output / "environment_manifest.json").read_text(encoding="utf-8")
    )
    assert refit_environment["selection_record"]["content"] == result
    assert refit_environment["run_identity"]["selection_record"] == (
        refit_environment["selection_record"]
    )
    assert json.loads(
        (refit_output / "trial_parameters.json").read_text(encoding="utf-8")
    )["parameters"] == result["selected_parameters"]

    tampered_record = tmp_path / "tampered-selected.json"
    tampered = json.loads(json.dumps(result))
    tampered["average_precision"] = float(tampered["average_precision"]) / 2.0
    tampered.pop("selection_record_sha256")
    tampered["selection_record_sha256"] = _selection_record_sha256(tampered)
    tampered_record.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    tampered_status = tmp_path / "tampered-selected.status.json"
    status_payload = json.loads(
        Path(f"{tmp_path / 'selected.json'}.status.json").read_text(encoding="utf-8")
    )
    status_payload["selection_record_sha256"] = tampered["selection_record_sha256"]
    tampered_status.write_text(json.dumps(status_payload) + "\n", encoding="utf-8")
    with pytest.raises(TaskCMethodRunError, match="replay"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=Path(refit_profile["input_npz"]),
            derived_input_manifest_path=Path(refit_profile["manifest"]),
            output_dir=tmp_path / "tampered-refit",
            seed=11,
            registry_path=ROOT / "configs/task_c_methods_v1.json",
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="k562",
            min_cells=5,
            public_manifest_path=Path(bundle["public_manifest"]),
            selection_record_path=tampered_record,
            selection_status_path=tampered_status,
            selection_tune_input_path=Path(tune_profile["input_npz"]),
            selection_tune_profile_manifest_path=Path(tune_profile["manifest"]),
            selection_config_path=CONFIG,
            selection_trial_directories=(second, first),
            selection_trial_input_bindings={
                first.resolve(): Path(train_profile["input_npz"]),
                second.resolve(): Path(train_profile["input_npz"]),
            },
            selection_trial_profile_bindings={
                first.resolve(): Path(train_profile["manifest"]),
                second.resolve(): Path(train_profile["manifest"]),
            },
            project_root=ROOT,
        )


def test_cli_rejects_k562_trial_for_rpe1_tune_profile(tmp_path: Path) -> None:
    k562 = _dataset_with_public_tune_controls("k562")
    rpe1 = _dataset_with_public_tune_controls("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    k562_profile = materialize_task_c_profile_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="k562",
        stage="train",
        output_dir=tmp_path / "k562-train",
    )
    rpe1_profile = materialize_task_c_profile_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        profile="connection",
        condition="within_environment",
        context_id="rpe1",
        stage="tune",
        output_dir=tmp_path / "rpe1-tune",
    )
    trial = _run_bound_mean_trial(
        tmp_path,
        bundle=bundle,
        profile=k562_profile,
        context_id="k562",
        trial_index=0,
    )

    completed = _run_cli(
        "--tune-npz", rpe1_profile["input_npz"],
        "--profile-manifest", rpe1_profile["manifest"],
        "--public-manifest", str(bundle["public_manifest"]),
        "--trial-dir", str(trial),
        "--trial-input", f"{trial}={k562_profile['input_npz']}",
        "--trial-profile-manifest", f"{trial}={k562_profile['manifest']}",
        "--output-json", str(tmp_path / "selected.json"),
        "--config", str(CONFIG),
    )

    assert completed.returncode != 0
    assert "context" in completed.stderr or "tuning evidence" in completed.stderr


def test_cli_scores_a_sealed_hypersca_train_run_on_separate_tune_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.causal import hypersca_c_run as hypersca_run

    k562 = _dataset_with_public_tune_controls("k562")
    rpe1 = _dataset_with_public_tune_controls("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    bundle = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    profiles = {
        stage: materialize_task_c_profile_input(
            public_manifest_path=Path(bundle["public_manifest"]),
            profile="connection",
            condition="within_environment",
            context_id="k562",
            stage=stage,
            output_dir=tmp_path / f"profile-{stage}",
        )
        for stage in ("train", "tune")
    }
    profile_manifest = json.loads(
        Path(profiles["train"]["manifest"]).read_text(encoding="utf-8")
    )
    genes = profile_manifest["gene_selection"]["ordered_genes"]
    gene_list = tmp_path / "genes.json"
    gene_list.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "selection_id": "tuning-train-profile",
                "selection_basis": "使用统一比较范围中的固定基因顺序",
                "genes": genes,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "hypersca-config.json"
    config.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "hypersca-candidate.json"
    candidate.write_text(
        '{"schema_version":"1.0","trial_index":0,"parameters":{}}\n',
        encoding="utf-8",
    )

    def fake_run(
        command: Sequence[str], *, output_dir: Path, timeout_seconds: object
    ) -> dict[str, object]:
        del timeout_seconds
        runtime = Path(output_dir)
        runtime.mkdir()
        raw_output = Path(command[command.index("--output-dir") + 1])
        raw_output.mkdir()
        (raw_output / "raw_predictions.csv").write_text(
            f"source,target,score\n{genes[0]},{genes[1]},1\n",
            encoding="utf-8",
        )
        inner = {"schema_version": "1.0", "status": "completed_raw_inference"}
        (runtime / "method_status.json").write_text(
            json.dumps(inner) + "\n", encoding="utf-8"
        )
        (runtime / "resource_usage.json").write_text(
            '{"schema_version":"1.0"}\n', encoding="utf-8"
        )
        return inner

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.run_isolated_method", fake_run
    )
    monkeypatch.setattr(
        hypersca_run, "validate_hypersca_c_output_bundle", lambda *args, **kwargs: None
    )
    trial = tmp_path / "hypersca-train-trial"
    run_task_c_method(
        method_id="hypersca_c",
        input_npz=Path(profiles["train"]["input_npz"]),
        derived_input_manifest_path=Path(profiles["train"]["manifest"]),
        output_dir=trial,
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id="k562",
        min_cells=5,
        public_manifest_path=Path(bundle["public_manifest"]),
        hypersca_config_path=config,
        gene_list_path=gene_list,
        trial_parameters_path=candidate,
        project_root=ROOT,
    )

    completed = _run_cli(
        "--tune-npz", profiles["tune"]["input_npz"],
        "--profile-manifest", profiles["tune"]["manifest"],
        "--public-manifest", str(bundle["public_manifest"]),
        "--trial-dir", str(trial),
        "--trial-input", f"{trial}={profiles['train']['input_npz']}",
        "--trial-profile-manifest", f"{trial}={profiles['train']['manifest']}",
        "--trial-hypersca-config", f"{trial}={config}",
        "--trial-gene-list", f"{trial}={gene_list}",
        "--output-json", str(tmp_path / "hypersca-selected.json"),
        "--config", str(CONFIG),
    )

    assert completed.returncode != 0
    assert "reconstruction" in completed.stderr or "scientific evidence" in completed.stderr
