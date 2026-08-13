from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_data import (
    TaskCDataset,
    build_shared_task_c_split,
    materialize_task_c_split,
)
from src.evaluation.task_c_rehearsal import (
    TaskCRehearsalError,
    build_rehearsal_execution_plan,
    build_rehearsal_run_id,
    classify_rehearsal_method_status,
    freeze_method_worker_entry,
    materialize_sealed_scoring_subset,
    validate_private_scoring_command,
    validate_required_run_artifacts,
)
from src.evaluation import task_c_rehearsal as rehearsal_module
from src.evaluation.task_c_aggregation import aggregate_task_c_runs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_task_c_rehearsal.py"
REQUIRED = {
    "run_manifest.json",
    "input_summary.json",
    "metrics.json",
    "predictions.csv",
    "promotion_decision.json",
    "method_status.json",
    "resource_usage.json",
    "environment_manifest.json",
}


def _dataset(tmp_path: Path, context: str, *, seed: int) -> TaskCDataset:
    genes = tuple(f"G{index:02d}" for index in range(40))
    sources = genes[:12]
    labels = ["non-targeting"] * 60 + [
        source for source in sources for _ in range(6)
    ]
    rng = np.random.default_rng(seed)
    expression = rng.normal(scale=0.05, size=(len(labels), len(genes)))
    for row, source in enumerate(labels):
        if source != "non-targeting":
            source_index = genes.index(source)
            expression[row, (source_index + 1) % len(genes)] += 4.0
    path = tmp_path / f"{context}.npz"
    np.savez(
        path,
        expression_matrix=expression.astype(np.float32),
        interventions=np.asarray(labels),
        var_names=np.asarray(genes),
    )
    return TaskCDataset(
        expression=expression.astype(np.float32),
        interventions=np.asarray(labels),
        gene_names=genes,
        context_id=context,
        source_path=path.resolve(),
        source_sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.fixture()
def prepared_root(tmp_path: Path) -> Path:
    k562 = _dataset(tmp_path, "k562", seed=11)
    rpe1 = _dataset(tmp_path, "rpe1", seed=23)
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    root = tmp_path / "prepared" / "splits" / "seed_11"
    materialize_task_c_split(k562, rpe1, split, root)
    return root


def _run_cli(
    prepared_root: Path,
    output_root: Path,
    *extra: str,
    methods: str = "hypersca_c,mean_difference,random1000",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile",
            "connection",
            "--prepared-root",
            str(prepared_root),
            "--method-assets-root",
            str(prepared_root.parent / "method_assets"),
            "--output-root",
            str(output_root),
            "--methods",
            methods,
            "--synthetic-smoke",
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_identity_and_required_files_reject_unsafe_or_incomplete_runs(
    tmp_path: Path,
) -> None:
    assert build_rehearsal_run_id(
        profile="connection",
        condition="within_k562",
        method_id="mean_difference",
        seed=11,
    ) == "connection__within_k562__mean_difference__seed-11"
    with pytest.raises(TaskCRehearsalError, match="unsafe"):
        build_rehearsal_run_id(
            profile="connection",
            condition="../private",
            method_id="mean_difference",
            seed=11,
        )
    run = tmp_path / "run"
    run.mkdir()
    for name in sorted(REQUIRED - {"metrics.json"}):
        (run / name).write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskCRehearsalError, match="metrics.json"):
        validate_required_run_artifacts(run, tuple(REQUIRED - {
            "method_status.json", "resource_usage.json", "environment_manifest.json"
        }))


def test_execution_plan_has_four_conditions_and_real_selection_closure() -> None:
    plan = build_rehearsal_execution_plan(
        profile="connection",
        method_ids=("hypersca_c", "mean_difference", "random1000"),
    )
    assert tuple(plan) == (
        "within_k562",
        "within_rpe1",
        "k562_to_rpe1",
        "rpe1_to_k562",
    )
    for condition in plan.values():
        assert condition["stages"] == ("train", "tune", "refit")
        assert condition["trial_counts"] == {
            "hypersca_c": 2,
            "mean_difference": 2,
            "random1000": 0,
        }
        assert condition["selection_bound_refit"] == (
            "hypersca_c",
            "mean_difference",
        )

    comprehensive = build_rehearsal_execution_plan(
        profile="comprehensive", method_ids=("hypersca_c", "mean_difference")
    )
    assert all(
        value["trial_counts"] == {"hypersca_c": 0, "mean_difference": 0}
        for value in comprehensive.values()
    )


@pytest.mark.parametrize(
    ("inner", "outer"),
    [
        ("completed_standardized_output", "passed_real_rehearsal"),
        ("failed_timeout", "failed_timeout"),
        ("failed_resource_limit", "failed_resource_limit"),
        ("failed_launch", "failed_launch"),
        ("failed_runtime_unavailable", "failed_runtime_unavailable"),
        ("official_code_incompatible", "official_code_incompatible"),
        ("failed_invalid_output", "failed_invalid_output"),
        ("official_assets_unavailable", "official_assets_unavailable"),
    ],
)
def test_method_statuses_are_retained_without_softening_failures(
    inner: str, outer: str
) -> None:
    assert classify_rehearsal_method_status(inner) == outer
    with pytest.raises(TaskCRehearsalError, match="unrecognized"):
        classify_rehearsal_method_status("completed_with_warnings")


def test_failed_training_trial_keeps_its_exact_status(tmp_path: Path) -> None:
    trial = tmp_path / "work/trials/trial_0"
    trial.mkdir(parents=True)
    (trial / "method_status.json").write_text(
        json.dumps(
            {
                "status": "failed_timeout",
                "reason": "time limit reached during the fixed training trial",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    found, status, reason = rehearsal_module._find_failed_method_bundle(
        tmp_path / "work"
    )

    assert found == trial
    assert status == "failed_timeout"
    assert reason == "time limit reached during the fixed training trial"


def test_scoring_failure_is_not_relabelled_as_a_passed_method() -> None:
    assert rehearsal_module._classify_controller_failure(
        TaskCRehearsalError("sealed scoring did not complete"),
        "completed_standardized_output",
    ) == "failed_private_scoring"
    assert rehearsal_module._classify_controller_failure(
        TaskCRehearsalError("outer reporting noticed the inner result"),
        "failed_timeout",
    ) == "failed_timeout"


def test_synthetic_cli_publishes_four_conditions_and_required_scientific_files(
    prepared_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    completed = _run_cli(prepared_root, output)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["claim_level"] == "workflow_validation_only"
    assert summary["promotion_eligible"] is False
    assert summary["profile"] == "connection"
    assert summary["attempted_methods"] == [
        "hypersca_c",
        "mean_difference",
        "random1000",
    ]
    assert summary["conditions"] == [
        "within_k562",
        "within_rpe1",
        "k562_to_rpe1",
        "rpe1_to_k562",
    ]
    run_dirs = sorted((output / "runs").iterdir())
    assert len(run_dirs) == 12
    for run_dir in run_dirs:
        assert {path.name for path in run_dir.iterdir()} == REQUIRED
        status = json.loads((run_dir / "method_status.json").read_text())
        assert status["status"] == "passed_real_rehearsal"
        promotion = json.loads((run_dir / "promotion_decision.json").read_text())
        assert promotion["status"] == "workflow_validation_only"
        assert promotion["promotion_eligible"] is False

    aggregated = aggregate_task_c_runs(run_dirs)
    assert aggregated["verified_completed_run_count"] == 12
    cross_run = next(
        run_dir for run_dir in run_dirs if "__k562_to_rpe1__mean_difference__" in run_dir.name
    )
    cross_input = json.loads((cross_run / "input_summary.json").read_text())
    standardization = cross_input["stages"]["refit"]["control_standardization"]
    assert standardization == {
        "center": "control mean in each environment",
        "control_label": "non-targeting",
        "low_scale_replacement": 1.0,
        "low_scale_threshold": 1e-06,
        "scale": "control population standard deviation (ddof=0)",
    }

    for method in ("hypersca_c", "mean_difference"):
        for run_dir in run_dirs:
            if f"__{method}__" not in run_dir.name:
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text())
            controls = metrics["null_controls"]
            assert controls["label_permutation"]["repeat_count"] == 20
            assert controls["control_resampling"]["repeat_count"] == 20
            assert len(controls["label_permutation"]["seeds"]) == 20
            assert len(controls["control_resampling"]["seeds"]) == 20


def test_resume_reuses_only_exact_verified_identity_and_detects_tampering(
    prepared_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    first = _run_cli(prepared_root, output, methods="mean_difference")
    assert first.returncode == 0, first.stderr
    resumed = _run_cli(
        prepared_root, output, "--resume", methods="mean_difference"
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["resume_status"] == "verified_existing_output"

    different = _run_cli(
        prepared_root, output, "--resume", methods="mean_difference,random1000"
    )
    assert different.returncode != 0
    assert "identity" in different.stderr

    run_dir = next((output / "runs").iterdir())
    (run_dir / "metrics.json").write_text('{"average_precision":0.0}\n')
    tampered = _run_cli(
        prepared_root, output, "--resume", methods="mean_difference"
    )
    assert tampered.returncode != 0
    assert "changed" in tampered.stderr or "fingerprint" in tampered.stderr

    missing_output = tmp_path / "missing-results"
    created = _run_cli(prepared_root, missing_output, methods="random1000")
    assert created.returncode == 0, created.stderr
    missing_run = next((missing_output / "runs").iterdir())
    (missing_run / "metrics.json").unlink()
    missing = _run_cli(
        prepared_root,
        missing_output,
        "--resume",
        methods="random1000",
    )
    assert missing.returncode != 0
    assert "missing" in missing.stderr


def test_existing_output_never_clobbers_without_resume(
    prepared_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    completed = _run_cli(prepared_root, output, methods="mean_difference")

    assert completed.returncode != 0
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not (output / "runs").exists()


def test_output_parent_symbolic_link_is_rejected(
    prepared_root: Path, tmp_path: Path
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "result-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    completed = _run_cli(
        prepared_root,
        alias / "results",
        methods="mean_difference",
    )

    assert completed.returncode != 0
    assert "symbolic" in completed.stderr
    assert not (real_parent / "results").exists()


def test_help_uses_biomedical_language_and_keeps_fixed_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for option in (
        "--profile",
        "--prepared-root",
        "--method-assets-root",
        "--output-root",
        "--methods",
        "--resume",
        "--synthetic-smoke",
    ):
        assert option in completed.stdout
    assert "流程验证" in completed.stdout
    assert "真实数据性能" in completed.stdout


def test_scoring_boundary_allows_only_the_exact_registered_sealed_input(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    heldout = private / "heldout.npz"
    heldout.write_bytes(b"fixed private bytes")
    other = private / "other.npz"
    other.write_bytes(b"other private bytes")
    worker = tmp_path / "worker.py"
    worker.write_text("print('reviewed')\n", encoding="utf-8")
    snapshot = freeze_method_worker_entry(worker)
    command = (
        str(Path(sys.executable).resolve()),
        "-I",
        str(worker),
        "--heldout-npz",
        str(heldout),
    )

    validate_private_scoring_command(
        command,
        private_root=private,
        execution_cwd=tmp_path,
        allowed_python_interpreters=(Path(sys.executable).resolve(),),
        allowed_worker_snapshots=(snapshot,),
        allowed_private_inputs=(heldout,),
    )
    with pytest.raises(TaskCRehearsalError, match="private scoring path"):
        validate_private_scoring_command(
            (*command[:-1], str(other)),
            private_root=private,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(Path(sys.executable).resolve(),),
            allowed_worker_snapshots=(snapshot,),
            allowed_private_inputs=(heldout,),
        )


def test_sealed_scoring_subset_matches_the_publicly_fixed_gene_scope(
    tmp_path: Path,
) -> None:
    source = tmp_path / "heldout.npz"
    np.savez(
        source,
        expression_matrix=np.arange(40, dtype=float).reshape(10, 4),
        interventions=np.asarray(
            ["non-targeting"] * 4 + ["A"] * 2 + ["B"] * 2 + ["C"] * 2
        ),
        var_names=np.asarray(["A", "B", "C", "D"]),
    )
    public_profile = tmp_path / "profile.npz"
    np.savez(
        public_profile,
        expression_matrix=np.zeros((2, 2), dtype=float),
        interventions=np.asarray(["non-targeting", "A"]),
        var_names=np.asarray(["B", "A"]),
    )
    private = tmp_path / "private-score"
    private.mkdir()
    created = materialize_sealed_scoring_subset(
        source_path=source,
        public_profile_input=public_profile,
        destination=private / "heldout-profile.npz",
        maximum_cells=6,
        seed=11,
    )

    with np.load(created, allow_pickle=False) as archive:
        assert archive["var_names"].tolist() == ["B", "A"]
        assert archive["expression_matrix"].shape[1] == 2
        assert archive["expression_matrix"].shape[0] <= 6
        assert set(archive["interventions"].tolist()) <= {
            "non-targeting",
            "A",
            "B",
        }
    second_private = tmp_path / "private-score-second"
    second_private.mkdir()
    repeated = materialize_sealed_scoring_subset(
        source_path=source,
        public_profile_input=public_profile,
        destination=second_private / "heldout-profile.npz",
        maximum_cells=6,
        seed=11,
    )
    assert repeated.read_bytes() == created.read_bytes()


def test_formal_scoring_is_started_only_by_the_validated_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = tmp_path / "prepared" / "splits" / "seed_11"
    provenance = tmp_path / "prepared" / "provenance"
    provenance.mkdir(parents=True)
    heldout_root = tmp_path / "private-scoring"
    heldout_root.mkdir()
    heldout = heldout_root / "heldout.npz"
    np.savez(
        heldout,
        expression_matrix=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 2.0, 0.0]]
        ),
        interventions=np.asarray(["non-targeting", "non-targeting", "A"]),
        var_names=np.asarray(["A", "B", "C"]),
    )
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            (source, target, float(source == "A" and target == "B"), True)
            for source in ("A", "B", "C")
            for target in ("A", "B", "C")
            if source != target
        ],
        columns=["source", "target", "score", "returned_by_method"],
    ).to_csv(predictions, index=False)
    pooled = tmp_path / "pooled.csv"
    pooled.write_text("source,target\nA,B\nB,A\n", encoding="utf-8")
    chip = tmp_path / "chip.csv"
    chip.write_text("source,target\nA,B\n", encoding="utf-8")
    reference = {
        "files": {
            "pooled": {
                "path": str(pooled),
                "sha256": hashlib.sha256(pooled.read_bytes()).hexdigest(),
            },
            "chipseq": {
                "path": str(chip),
                "sha256": hashlib.sha256(chip.read_bytes()).hexdigest(),
            },
        }
    }
    (provenance / "k562_references.json").write_text(
        json.dumps(reference) + "\n", encoding="utf-8"
    )
    work = tmp_path / "work"
    work.mkdir()
    launched: dict[str, object] = {}

    def fake_launcher(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        launched["command"] = command
        launched.update(kwargs)
        command_values = tuple(command)  # type: ignore[arg-type]
        output = Path(command_values[command_values.index("--output-json") + 1])
        output.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "supplementary_official_metrics",
                    "metrics": {"official_check": 0.5},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command_values, 0, "", "")

    monkeypatch.setattr(
        rehearsal_module, "run_validated_private_scoring_command", fake_launcher
    )
    monkeypatch.setattr(
        rehearsal_module,
        "_causalbench_python",
        lambda *_args: Path(sys.executable).resolve(),
    )

    class Registry:
        causalbench = {"environment": "fixed-test"}

    metrics = rehearsal_module._formal_scoring_subset(
        condition="within_k562",
        predictions=predictions,
        prepared_root=prepared,
        asset_root=tmp_path / "assets",
        work_dir=work,
        seed=11,
        registry=Registry(),
        project_root=ROOT,
        heldout=heldout,
        private_root=heldout_root,
    )

    assert metrics["average_precision"] == 1.0
    assert launched["allowed_private_inputs"] == (heldout,)
    command = tuple(launched["command"])  # type: ignore[arg-type]
    assert command[:2] == (str(Path(sys.executable).resolve()), "-I")
    snapshots = launched["allowed_worker_snapshots"]
    assert len(snapshots) == 2  # type: ignore[arg-type]


def test_connection_controller_runs_two_train_trials_then_selected_refit(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles: dict[str, dict[str, Path]] = {}
    for stage in ("train", "tune", "refit"):
        created = rehearsal_module._profile_inputs(
            public_manifest=prepared_root / "public_manifest.json",
            profile="connection",
            staging=tmp_path / f"profile-build-{stage}",
        )["within_k562"][stage]
        profiles[stage] = created
    calls: list[dict[str, object]] = []

    def fake_method_bundle(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        (output / "method_status.json").write_text(
            json.dumps({"status": "completed_standardized_output"}) + "\n",
            encoding="utf-8",
        )
        return {"status": "completed_standardized_output"}

    def fake_selection(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output-json") + 1])
        status = Path(command[command.index("--status-json") + 1])
        output.write_text(
            json.dumps({"selected_parameters": {}, "selected_trial_index": 0})
            + "\n",
            encoding="utf-8",
        )
        status.write_text('{"status":"completed_selection"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(rehearsal_module, "_run_method_bundle", fake_method_bundle)
    monkeypatch.setattr(rehearsal_module.subprocess, "run", fake_selection)
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.validate_task_c_method_output_bundle",
        lambda **_kwargs: {"status": "completed_standardized_output"},
    )
    final, status = rehearsal_module._select_connection_configuration(
        method_id="mean_difference",
        condition="within_k562",
        profiles=profiles,
        work_dir=tmp_path / "work",
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        public_manifest=prepared_root / "public_manifest.json",
        min_cells=5,
        timeout_seconds=60,
        project_root=ROOT,
        base_hypersca_config=ROOT / "configs/hypersca_c_v1.json",
    )

    assert status["status"] == "completed_standardized_output"
    assert final == tmp_path / "work/refit"
    assert [
        json.loads(Path(call["profile_record"]["manifest"]).read_text())["stage"]  # type: ignore[index]
        for call in calls
    ] == ["train", "train", "refit"]
    assert calls[-1]["selection_arguments"] is not None
