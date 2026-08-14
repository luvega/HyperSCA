from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_data import (
    SealedHoldoutSemanticContentHasher,
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


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _record_sha256(payload: object) -> str:
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
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_record(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _external_resume_token(
    identity: object,
    inventory: object,
    summary: dict[str, object],
) -> str:
    rebuilt_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"resume_status", "resume_token"}
    }
    return _canonical_sha256(
        {
            "controller_identity": identity,
            "file_inventory": inventory,
            "rebuilt_summary": rebuilt_summary,
        }
    )


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


def _formal_validation_dependencies(prepared_root: Path, tmp_path: Path) -> Path:
    provenance = prepared_root.parents[1] / "provenance"
    provenance.mkdir(parents=True)
    for context in ("k562", "rpe1"):
        (provenance / f"{context}.json").write_text("{}\n", encoding="utf-8")
        references: dict[str, dict[str, str]] = {}
        for kind in ("pooled", "chipseq"):
            path = tmp_path / f"{context}-{kind}.csv"
            path.write_text("source,target\nG00,G01\n", encoding="utf-8")
            references[kind] = {
                "path": str(path),
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (provenance / f"{context}_references.json").write_text(
            json.dumps({"files": references}) + "\n",
            encoding="utf-8",
        )
    assets = tmp_path / "method-assets"
    assets.mkdir()
    for name in ("bootstrap_identity.json", "bootstrap_manifest.json"):
        (assets / name).write_text("{}\n", encoding="utf-8")
    return assets


def _copy_public_sibling_splits(prepared_root: Path) -> None:
    for seed in (23, 47, 71, 97):
        sibling = prepared_root.parent / f"seed_{seed}"
        shutil.copytree(prepared_root, sibling)
        manifest_path = sibling / "public_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["seed"] = seed
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


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
            "mean_difference": 0,
            "random1000": 0,
        }
        assert condition["selection_bound_refit"] == ("hypersca_c",)
        assert condition["method_stages"] == {
            "hypersca_c": ("train", "tune", "refit"),
            "mean_difference": ("refit",),
            "random1000": ("refit",),
        }

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
        assert status["status"] == "passed_synthetic_smoke"
        assert status["controller_validation"] == (
            "verified_task_c_synthetic_smoke_bundle_v1"
        )
        promotion = json.loads((run_dir / "promotion_decision.json").read_text())
        assert promotion["status"] == "workflow_validation_only"
        assert promotion["promotion_eligible"] is False

    aggregated = aggregate_task_c_runs(run_dirs)
    assert aggregated["verified_completed_run_count"] == 0
    assert aggregated["synthetic_structural_run_count"] == 12
    assert aggregated["not_formally_completed_count"] == 12
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
            assert controls["scope"] == "synthetic_orchestration_only"
            assert controls["formal_null_gate_passed"] is False
            assert controls["label_permutation"]["repeat_count"] == 20
            assert controls["control_resampling"]["repeat_count"] == 20
            assert len(controls["label_permutation"]["seeds"]) == 20
            assert len(controls["control_resampling"]["seeds"]) == 20

    for run_dir in run_dirs:
        resource = json.loads((run_dir / "resource_usage.json").read_text())
        method = json.loads((run_dir / "method_status.json").read_text())["method_id"]
        assert resource["null_control_analysis_count"] == 0
        assert resource["null_control_repeat_count_per_type"] == 0
        if method == "hypersca_c":
            assert resource["used_stages"] == ["synthetic_smoke"]
        else:
            assert resource["used_stages"] == ["synthetic_smoke"]


def test_resume_reuses_only_exact_verified_identity_and_detects_tampering(
    prepared_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    first = _run_cli(prepared_root, output, methods="mean_difference")
    assert first.returncode == 0, first.stderr
    resume_token = json.loads(first.stdout)["resume_token"]
    resumed = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        resume_token,
        methods="mean_difference",
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["resume_status"] == "verified_existing_output"

    different = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        resume_token,
        methods="mean_difference,random1000",
    )
    assert different.returncode != 0
    assert "identity" in different.stderr

    run_dir = next((output / "runs").iterdir())
    (run_dir / "metrics.json").write_text('{"average_precision":0.0}\n')
    tampered = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        resume_token,
        methods="mean_difference",
    )
    assert tampered.returncode != 0
    assert "changed" in tampered.stderr or "fingerprint" in tampered.stderr

    missing_output = tmp_path / "missing-results"
    created = _run_cli(prepared_root, missing_output, methods="random1000")
    assert created.returncode == 0, created.stderr
    missing_resume_token = json.loads(created.stdout)["resume_token"]
    missing_run = next((missing_output / "runs").iterdir())
    (missing_run / "metrics.json").unlink()
    missing = _run_cli(
        prepared_root,
        missing_output,
        "--resume",
        "--resume-token",
        missing_resume_token,
        methods="random1000",
    )
    assert missing.returncode != 0
    assert "missing" in missing.stderr


def test_resume_requires_the_token_returned_by_initial_publication(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "results"
    first = _run_cli(prepared_root, output, methods="mean_difference")
    assert first.returncode == 0, first.stderr
    summary = json.loads(first.stdout)
    token = summary["resume_token"]
    assert len(token) == 71 and token.startswith("sha256:")
    controller = json.loads(
        (output / "controller_manifest.json").read_text(encoding="utf-8")
    )
    assert controller["resume_token"] == token
    assert controller["summary"]["resume_token"] == token

    missing = _run_cli(
        prepared_root,
        output,
        "--resume",
        methods="mean_difference",
    )
    assert missing.returncode != 0
    assert "resume" in missing.stderr and "token" in missing.stderr
    wrong = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        "sha256:" + "0" * 64,
        methods="mean_difference",
    )
    assert wrong.returncode != 0
    resumed = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        token,
        methods="mean_difference",
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["resume_token"] == token


def test_resume_api_rejects_a_missing_output_root_without_creating_it(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing-api-results"

    with pytest.raises(TaskCRehearsalError, match="resume.*does not exist"):
        rehearsal_module.run_task_c_rehearsal(
            profile="connection",
            prepared_root=prepared_root,
            method_assets_root=prepared_root.parent / "method_assets",
            output_root=output,
            method_ids=("mean_difference",),
            expected_resume_token="sha256:" + "0" * 64,
            resume=True,
            synthetic_smoke=True,
            project_root=ROOT,
        )

    assert not output.exists()


def test_resume_cli_rejects_a_missing_output_root_without_creating_it(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing-cli-results"

    completed = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        "sha256:" + "0" * 64,
        methods="mean_difference",
    )

    assert completed.returncode != 0
    assert "resume" in completed.stderr and "does not exist" in completed.stderr
    assert not output.exists()


def test_resume_external_token_rejects_fully_resigned_success_bundle(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "results"
    first = _run_cli(prepared_root, output, methods="mean_difference")
    assert first.returncode == 0, first.stderr
    original_token = json.loads(first.stdout)["resume_token"]
    run_dir = next((output / "runs").iterdir())

    predictions_path = run_dir / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "score"] = float(predictions.loc[0, "score"]) + 0.25
    predictions.to_csv(predictions_path, index=False)
    prediction_sha256 = (
        "sha256:" + hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    )
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["prediction_sha256"] = prediction_sha256
    _write_record(metrics_path, metrics)
    evidence = {
        "input_summary.json": _record_sha256(
            json.loads((run_dir / "input_summary.json").read_text(encoding="utf-8"))
        ),
        "metrics.json": _record_sha256(metrics),
        "predictions.csv": prediction_sha256,
        "promotion_decision.json": _record_sha256(
            json.loads(
                (run_dir / "promotion_decision.json").read_text(encoding="utf-8")
            )
        ),
        "environment_manifest.json": _record_sha256(
            json.loads(
                (run_dir / "environment_manifest.json").read_text(encoding="utf-8")
            )
        ),
        "resource_usage.json": _record_sha256(
            json.loads((run_dir / "resource_usage.json").read_text(encoding="utf-8"))
        ),
    }
    input_summary = json.loads(
        (run_dir / "input_summary.json").read_text(encoding="utf-8")
    )
    run_identity = {
        "schema_version": "1.0",
        "profile": input_summary["profile"],
        "condition": input_summary["condition"],
        "method_id": input_summary["method_id"],
        "seed": 11,
        "input_summary_sha256": _canonical_sha256(input_summary),
        "prediction_sha256": prediction_sha256,
        "evidence_sha256": evidence,
    }
    run_identity_sha256 = _canonical_sha256(run_identity)
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(run_identity)
    manifest["run_identity_sha256"] = run_identity_sha256
    _write_record(run_dir / "run_manifest.json", manifest)
    status = json.loads((run_dir / "method_status.json").read_text(encoding="utf-8"))
    status["run_identity_sha256"] = run_identity_sha256
    _write_record(run_dir / "method_status.json", status)

    controller_path = output / "controller_manifest.json"
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    inventory = {
        path.relative_to(output).as_posix(): (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != controller_path
    }
    changed_token = _external_resume_token(
        controller["identity"], inventory, controller["summary"]
    )
    assert changed_token != original_token
    controller["file_inventory"] = inventory
    controller["resume_token"] = changed_token
    controller["summary"]["resume_token"] = changed_token
    _write_record(controller_path, controller)

    resumed = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        original_token,
        methods="mean_difference",
    )

    assert resumed.returncode != 0
    assert "resume" in resumed.stderr and "token" in resumed.stderr


def test_resume_rejects_hardlinked_evidence(
    prepared_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    first = _run_cli(prepared_root, output, methods="mean_difference")
    assert first.returncode == 0, first.stderr
    resume_token = json.loads(first.stdout)["resume_token"]
    metrics = next((output / "runs").iterdir()) / "metrics.json"
    outside = tmp_path / "outside-metrics.json"
    metrics.replace(outside)
    os.link(outside, metrics)

    resumed = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        resume_token,
        methods="mean_difference",
    )

    assert resumed.returncode != 0
    assert "hard" in resumed.stderr or "link" in resumed.stderr


def test_resume_rejects_changed_metrics_even_when_controller_inventory_is_resigned(
    prepared_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    first = _run_cli(prepared_root, output, methods="mean_difference")
    assert first.returncode == 0, first.stderr
    resume_token = json.loads(first.stdout)["resume_token"]
    run_dir = next((output / "runs").iterdir())
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["average_precision"] = 0.125
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    controller_path = output / "controller_manifest.json"
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    relative = metrics_path.relative_to(output).as_posix()
    controller["file_inventory"][relative] = (
        "sha256:" + hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    )
    controller_path.write_text(
        json.dumps(controller, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resumed = _run_cli(
        prepared_root,
        output,
        "--resume",
        "--resume-token",
        resume_token,
        methods="mean_difference",
    )

    assert resumed.returncode != 0
    assert "metrics" in resumed.stderr or "evidence" in resumed.stderr


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


def test_output_root_and_grandparent_symbolic_links_are_rejected_before_resume(
    prepared_root: Path, tmp_path: Path
) -> None:
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(real_output, target_is_directory=True)
    direct = _run_cli(
        prepared_root,
        output_alias,
        "--resume",
        "--resume-token",
        "sha256:" + "0" * 64,
        methods="mean_difference",
    )
    assert direct.returncode != 0
    assert "symbolic" in direct.stderr

    real_parent = tmp_path / "deep-real"
    real_parent.mkdir()
    grandparent_alias = tmp_path / "grandparent-alias"
    grandparent_alias.symlink_to(real_parent, target_is_directory=True)
    nested = _run_cli(
        prepared_root,
        grandparent_alias / "child" / "results",
        methods="mean_difference",
    )
    assert nested.returncode != 0
    assert "symbolic" in nested.stderr
    assert not (real_parent / "child").exists()


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
        "--resume-token",
        "--prepared-identity-sha256",
        "--synthetic-smoke",
    ):
        assert option in completed.stdout
    assert "流程验证" in completed.stdout
    assert "真实数据性能" in completed.stdout
    assert "独立保存" in completed.stdout
    assert "不是签名" in completed.stdout
    assert "输出根目录必须已经存在" in completed.stdout


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


def test_sealed_scoring_subset_rejects_hardlinked_npz_input(tmp_path: Path) -> None:
    source = tmp_path / "heldout.npz"
    np.savez(
        source,
        expression_matrix=np.arange(12, dtype=float).reshape(4, 3),
        interventions=np.asarray(["non-targeting", "non-targeting", "A", "A"]),
        var_names=np.asarray(["A", "B", "C"]),
    )
    public_profile = tmp_path / "profile.npz"
    np.savez(
        public_profile,
        expression_matrix=np.zeros((2, 2), dtype=float),
        interventions=np.asarray(["non-targeting", "A"]),
        var_names=np.asarray(["A", "B"]),
    )
    os.link(public_profile, tmp_path / "profile-second-name.npz")
    private = tmp_path / "private-score"
    private.mkdir()

    with pytest.raises(TaskCRehearsalError, match="one link"):
        materialize_sealed_scoring_subset(
            source_path=source,
            public_profile_input=public_profile,
            destination=private / "heldout-profile.npz",
            maximum_cells=4,
            seed=11,
        )


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
    scoring_resource = json.loads(
        (work / "sealed_scoring.resource.json").read_text(encoding="utf-8")
    )
    assert scoring_resource["component_kind"] == "sealed_scoring"
    assert scoring_resource["stage"] == "scoring"
    assert scoring_resource["status"] == "completed"
    assert scoring_resource["elapsed_seconds"] >= 0.0
    assert scoring_resource["written_disk_bytes"] > 0
    assert scoring_resource["measurement_availability"]["written_disk_bytes"] is True


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
        method_id="hypersca_c",
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


def test_connection_selector_rejects_fixed_no_tuning_baseline(
    prepared_root: Path, tmp_path: Path
) -> None:
    profiles = rehearsal_module._profile_inputs(
        public_manifest=prepared_root / "public_manifest.json",
        profile="connection",
        staging=tmp_path / "profiles",
    )["within_k562"]

    with pytest.raises(TaskCRehearsalError, match="HyperSCA-C"):
        rehearsal_module._select_connection_configuration(
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


def test_cross_hypersca_profile_contexts_preserve_frozen_rows_and_gene_order(
    prepared_root: Path, tmp_path: Path
) -> None:
    profile = rehearsal_module._profile_inputs(
        public_manifest=prepared_root / "public_manifest.json",
        profile="connection",
        staging=tmp_path / "profiles",
    )["k562_to_rpe1"]["refit"]

    contexts = rehearsal_module.materialize_hypersca_profile_contexts(
        profile_record=profile,
        output_dir=tmp_path / "hypersca-contexts",
    )

    assert tuple(contexts) == ("k562", "rpe1")
    with np.load(profile["input"], allow_pickle=False) as merged:
        genes = merged["var_names"].tolist()
        labels = merged["environment_labels"].astype(str)
        merged_expression = merged["expression_matrix"]
        merged_interventions = merged["interventions"].astype(str)
    for context_id, path in contexts.items():
        with np.load(path, allow_pickle=False) as context:
            assert set(context.files) == {
                "expression_matrix",
                "interventions",
                "var_names",
            }
            assert context["var_names"].tolist() == genes
            selected = labels == context_id
            np.testing.assert_array_equal(
                context["expression_matrix"], merged_expression[selected]
            )
            np.testing.assert_array_equal(
                context["interventions"].astype(str),
                merged_interventions[selected],
            )


def test_hypersca_command_boundary_uses_only_the_verified_profile_record(
    tmp_path: Path,
) -> None:
    from src.evaluation.task_c_method_run import _build_hypersca_command

    command = _build_hypersca_command(
        project_root=ROOT,
        context_values=(),
        config_path=ROOT / "configs/hypersca_c_v1.json",
        gene_list_path=tmp_path / "genes.json",
        public_manifest_path=tmp_path / "public_manifest.json",
        output_dir=tmp_path / "output",
        seed=11,
        device="cpu",
        profile_input_path=tmp_path / "verified-profile.npz",
        profile_manifest_path=tmp_path / "profile_manifest.json",
    )

    assert "--profile-manifest" in command
    assert "--profile-input" in command
    assert "--profile-context" not in command
    assert "--context" not in command


def test_cross_hypersca_orchestration_passes_no_arbitrary_context_paths(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = rehearsal_module._profile_inputs(
        public_manifest=prepared_root / "public_manifest.json",
        profile="connection",
        staging=tmp_path / "profiles",
    )["k562_to_rpe1"]["refit"]
    captured: dict[str, object] = {}

    def fake_run_task_c_method(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "completed_standardized_output"}

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.run_task_c_method",
        fake_run_task_c_method,
    )
    rehearsal_module._run_method_bundle(
        method_id="hypersca_c",
        profile_record=profile,
        output_dir=tmp_path / "method-output",
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        public_manifest=prepared_root / "public_manifest.json",
        context_id="k562_to_rpe1",
        min_cells=5,
        timeout_seconds=60,
        project_root=ROOT,
        hypersca_config=ROOT / "configs/hypersca_c_v1.json",
        gene_list=tmp_path / "genes.json",
    )

    assert "profile_context_values" not in captured
    assert captured["input_npz"] == profile["input"]


def test_formal_fixed_baseline_runs_real_method_orchestration(
    prepared_root: Path, tmp_path: Path
) -> None:
    profiles = rehearsal_module._profile_inputs(
        public_manifest=prepared_root / "public_manifest.json",
        profile="connection",
        staging=tmp_path / "profiles",
    )["within_k562"]

    final, status = rehearsal_module._run_formal_final_method(
        method_id="mean_difference",
        condition="within_k562",
        profile="connection",
        profiles=profiles,
        work_dir=tmp_path / "formal-work",
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        public_manifest=prepared_root / "public_manifest.json",
        min_cells=5,
        timeout_seconds=60,
        project_root=ROOT,
        source_kind="local",
    )

    assert status["status"] == "completed_standardized_output"
    assert (final / "predictions.csv").is_file()
    assert not (tmp_path / "formal-work/trials").exists()
    resource = json.loads(
        (tmp_path / "formal-work/refit.resource.json").read_text(encoding="utf-8")
    )
    assert resource["stage"] == "refit"
    assert resource["elapsed_seconds"] >= 0.0
    assert resource["peak_rss_bytes"] is None
    assert resource["peak_gpu_memory_bytes"] is None
    assert resource["written_disk_bytes"] > 0
    assert resource["measurement_availability"] == {
        "elapsed_seconds": True,
        "peak_rss_bytes": False,
        "peak_gpu_memory_bytes": False,
        "written_disk_bytes": True,
    }


def test_prepublication_verifier_replays_the_complete_inner_bundle_boundary(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = rehearsal_module._profile_inputs(
        public_manifest=prepared_root / "public_manifest.json",
        profile="connection",
        staging=tmp_path / "profiles",
    )["within_k562"]
    captured: dict[str, object] = {}

    def fake_bundle(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "completed_standardized_output", "reuse": "verified"}

    monkeypatch.setattr(rehearsal_module, "_run_method_bundle", fake_bundle)
    rehearsal_module._verify_formal_final_method_bundle(
        method_id="mean_difference",
        condition="within_k562",
        profile="connection",
        profiles=profiles,
        work_dir=tmp_path / "formal-work",
        seed=11,
        registry_path=ROOT / "configs/task_c_methods_v1.json",
        asset_root=tmp_path / "assets",
        public_manifest=prepared_root / "public_manifest.json",
        min_cells=5,
        timeout_seconds=60,
        project_root=ROOT,
        source_kind="local",
    )

    assert captured["output_dir"] == tmp_path / "formal-work/refit"
    assert captured["profile_record"] == profiles["refit"]
    assert captured["resource_record_path"] is None


def test_scoring_and_publication_share_one_immutable_prediction_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inner-predictions.csv"
    original = (
        b"source,target,score,returned_by_method\n"
        b"A,B,1.0,True\nB,A,0.0,False\n"
    )
    source.write_bytes(original)
    snapshot = rehearsal_module.freeze_rehearsal_predictions(
        source_path=source,
        destination=tmp_path / "work/frozen-predictions.csv",
        expected_genes=("A", "B"),
    )
    source.write_bytes(original.replace(b"1.0", b"9.0"))
    output = tmp_path / "runs/run"
    metrics = {"average_precision": 1.0, "prediction_sha256": snapshot.sha256}

    rehearsal_module._publish_outer_success(
        destination=output,
        method_id="mean_difference",
        condition="within_k562",
        profile="connection",
        seed=11,
        predictions=None,
        prediction_snapshot=snapshot,
        metrics=metrics,
        input_summary={"used_stages": ["refit"]},
        inner_dir=None,
        work_dir=None,
        synthetic_smoke=False,
        required_artifacts=(
            "run_manifest.json",
            "input_summary.json",
            "metrics.json",
            "predictions.csv",
            "promotion_decision.json",
        ),
    )

    assert (output / "predictions.csv").read_bytes() == original
    published_metrics = json.loads((output / "metrics.json").read_text())
    assert published_metrics["prediction_sha256"] == snapshot.sha256
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["prediction_sha256"] == snapshot.sha256


def test_formal_scoring_rejects_an_unfrozen_prediction_path(tmp_path: Path) -> None:
    with pytest.raises(TaskCRehearsalError, match="frozen prediction snapshot"):
        rehearsal_module._formal_scoring(
            condition="within_k562",
            predictions=tmp_path / "mutable.csv",
            prepared_root=tmp_path / "prepared",
            asset_root=tmp_path / "assets",
            work_dir=tmp_path / "work",
            seed=11,
            registry=object(),
            project_root=ROOT,
            public_profile_input=tmp_path / "profile.npz",
            maximum_cells=10,
        )


def test_rehearsal_closure_detects_code_change_after_freezing(tmp_path: Path) -> None:
    code = tmp_path / "critical.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    closure = rehearsal_module._freeze_rehearsal_closure({"critical": code})
    code.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(TaskCRehearsalError, match="closure.*changed"):
        rehearsal_module._verify_rehearsal_closure(closure)


def test_rehearsal_closure_binds_external_prepared_identity(tmp_path: Path) -> None:
    code = tmp_path / "critical.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    first = rehearsal_module._freeze_rehearsal_closure(
        {"critical": code},
        external_identity={"prepared_identity_sha256": "sha256:" + "1" * 64},
    )
    second = rehearsal_module._freeze_rehearsal_closure(
        {"critical": code},
        external_identity={"prepared_identity_sha256": "sha256:" + "2" * 64},
    )

    assert first.external_identity == {
        "prepared_identity_sha256": "sha256:" + "1" * 64
    }
    assert first.sha256 != second.sha256


def test_formal_validation_rejects_unrelated_absolute_private_inventory(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    manifest_path = prepared_root / "private/private_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unrelated = tmp_path / "unrelated.npz"
    unrelated.write_bytes(b"not one of the four registered holdouts")
    manifest["files"] = {
        str(unrelated.resolve()): "sha256:"
        + hashlib.sha256(unrelated.read_bytes()).hexdigest()
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (prepared_root / "private/within/k562/holdout.npz").write_bytes(b"")
    monkeypatch.setattr(rehearsal_module, "_FULL_RUN_SEEDS", (11,))

    with pytest.raises(TaskCRehearsalError, match="sealed|private|holdout|inventory"):
        rehearsal_module._validate_prepared_rehearsal_inputs(
            prepared_root=prepared_root,
            method_assets_root=assets,
            synthetic_smoke=False,
        )


def test_formal_validation_accepts_exact_registered_private_split(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    monkeypatch.setattr(rehearsal_module, "_FULL_RUN_SEEDS", (11,))

    manifest_path, public = rehearsal_module._validate_prepared_rehearsal_inputs(
        prepared_root=prepared_root,
        method_assets_root=assets,
        synthetic_smoke=False,
    )

    assert manifest_path == prepared_root / "public_manifest.json"
    assert public["materialization_identity"]["seed"] == 11


def test_formal_validation_requires_rematerialization_for_old_identity(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    for relative in ("public_manifest.json", "private/private_manifest.json"):
        path = prepared_root / relative
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("sealed_holdout_semantic_content_sha256")
        manifest["materialization_identity"].pop(
            "sealed_holdout_semantic_content_sha256"
        )
        path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    monkeypatch.setattr(rehearsal_module, "_FULL_RUN_SEEDS", (11,))

    with pytest.raises(TaskCRehearsalError, match="rematerialize"):
        rehearsal_module._validate_prepared_rehearsal_inputs(
            prepared_root=prepared_root,
            method_assets_root=assets,
            synthetic_smoke=False,
        )


def test_formal_validation_rejects_changed_private_split_semantics(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    manifest_path = prepared_root / "private/private_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["holdout_sources"] = []
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    monkeypatch.setattr(rehearsal_module, "_FULL_RUN_SEEDS", (11,))

    with pytest.raises(TaskCRehearsalError, match="sealed|private|split|semantic"):
        rehearsal_module._validate_prepared_rehearsal_inputs(
            prepared_root=prepared_root,
            method_assets_root=assets,
            synthetic_smoke=False,
        )


def test_formal_validation_rejects_resigned_finite_private_expression_change(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    relative = "private/within/k562/holdout.npz"
    holdout_path = prepared_root / relative
    with np.load(holdout_path, allow_pickle=False) as archive:
        expression = np.asarray(archive["expression_matrix"]).copy()
        interventions = np.asarray(archive["interventions"]).copy()
        var_names = np.asarray(archive["var_names"]).copy()
    expression[0, 0] += np.asarray(0.5, dtype=expression.dtype)
    np.savez_compressed(
        holdout_path,
        expression_matrix=expression,
        interventions=interventions,
        var_names=var_names,
    )
    manifest_path = prepared_root / "private/private_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = (
        "sha256:" + hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    monkeypatch.setattr(rehearsal_module, "_FULL_RUN_SEEDS", (11,))

    with pytest.raises(TaskCRehearsalError, match="commitment|sealed|semantic"):
        rehearsal_module._validate_prepared_rehearsal_inputs(
            prepared_root=prepared_root,
            method_assets_root=assets,
            synthetic_smoke=False,
        )


def test_formal_rehearsal_requires_external_prepared_identity_after_coordinated_resign(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    _copy_public_sibling_splits(prepared_root)
    public_path = prepared_root / "public_manifest.json"
    private_path = prepared_root / "private/private_manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    encoded_identity = json.dumps(
        public["materialization_identity"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    prepared_identity_sha256 = (
        "sha256:" + hashlib.sha256(encoded_identity).hexdigest()
    )

    relative = "private/within/k562/holdout.npz"
    holdout_path = prepared_root / relative
    with np.load(holdout_path, allow_pickle=False) as archive:
        expression = np.asarray(archive["expression_matrix"]).copy()
        interventions = np.asarray(archive["interventions"]).copy()
        var_names = np.asarray(archive["var_names"]).copy()
    expression[0, 0] += np.asarray(0.5, dtype=expression.dtype)
    np.savez_compressed(
        holdout_path,
        expression_matrix=expression,
        interventions=interventions,
        var_names=var_names,
    )
    private = json.loads(private_path.read_text(encoding="utf-8"))
    private["files"][relative] = (
        "sha256:" + hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    )
    hasher = SealedHoldoutSemanticContentHasher()
    for logical_artifact in sorted(rehearsal_module._PRIVATE_HOLDOUT_PATHS):
        with np.load(prepared_root / logical_artifact, allow_pickle=False) as archive:
            hasher.add_arrays(
                logical_artifact,
                np.asarray(archive["expression_matrix"]),
                np.asarray(archive["interventions"]),
                tuple(str(value) for value in archive["var_names"].tolist()),
            )
    changed_commitment = hasher.sha256()
    for path, manifest in ((public_path, public), (private_path, private)):
        manifest["sealed_holdout_semantic_content_sha256"] = changed_commitment
        manifest["materialization_identity"][
            "sealed_holdout_semantic_content_sha256"
        ] = changed_commitment
        path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(TaskCRehearsalError, match="prepared.*identity|external"):
        rehearsal_module.run_task_c_rehearsal(
            profile="connection",
            prepared_root=prepared_root,
            prepared_identity_sha256=prepared_identity_sha256,
            method_assets_root=assets,
            output_root=tmp_path / "results",
            method_ids=("mean_difference",),
            synthetic_smoke=False,
            project_root=ROOT,
        )


def test_formal_rehearsal_requires_independently_saved_prepared_identity(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    output = tmp_path / "existing-results"
    output.mkdir()
    with pytest.raises(TaskCRehearsalError, match="prepared.*identity|independent"):
        rehearsal_module.run_task_c_rehearsal(
            profile="connection",
            prepared_root=prepared_root,
            method_assets_root=assets,
            output_root=output,
            method_ids=("mean_difference",),
            synthetic_smoke=False,
            project_root=ROOT,
        )


def test_formal_validation_rejects_resigned_invalid_gene_projection(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    public_path = prepared_root / "public_manifest.json"
    private_path = prepared_root / "private/private_manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    projection = public["gene_projection"]
    count = projection["common"]["count"]
    projection["contexts"]["k562"]["selected_original_indices"] = [0] * count
    for manifest in (public, private):
        manifest["gene_projection"] = projection
        manifest["materialization_identity"]["gene_projection"] = projection
    public_path.write_text(json.dumps(public) + "\n", encoding="utf-8")
    private_path.write_text(json.dumps(private) + "\n", encoding="utf-8")
    monkeypatch.setattr(rehearsal_module, "_FULL_RUN_SEEDS", (11,))

    with pytest.raises(TaskCRehearsalError, match="projection|gene|semantic"):
        rehearsal_module._validate_prepared_rehearsal_inputs(
            prepared_root=prepared_root,
            method_assets_root=assets,
            synthetic_smoke=False,
        )


def test_rehearsal_closure_freezes_all_four_private_holdouts(
    prepared_root: Path,
    tmp_path: Path,
) -> None:
    assets = _formal_validation_dependencies(prepared_root, tmp_path)
    paths = rehearsal_module._rehearsal_closure_paths(
        project_root=ROOT,
        prepared_root=prepared_root,
        method_assets_root=assets,
        synthetic_smoke=False,
    )
    private_labels = {
        label for label in paths if label.startswith("private_holdout:")
    }
    assert private_labels == {
        "private_holdout:private/within/k562/holdout.npz",
        "private_holdout:private/within/rpe1/holdout.npz",
        "private_holdout:private/cross/k562_to_rpe1/target_holdout.npz",
        "private_holdout:private/cross/rpe1_to_k562/target_holdout.npz",
    }
    closure = rehearsal_module._freeze_rehearsal_closure(paths)
    relative = "private/within/k562/holdout.npz"
    changed = prepared_root / relative
    changed.write_bytes(b"changed after the round was frozen")
    private_manifest = prepared_root / "private/private_manifest.json"
    resigned = json.loads(private_manifest.read_text(encoding="utf-8"))
    resigned["files"][relative] = (
        "sha256:" + hashlib.sha256(changed.read_bytes()).hexdigest()
    )
    private_manifest.write_text(json.dumps(resigned) + "\n", encoding="utf-8")

    with pytest.raises(TaskCRehearsalError, match="closure.*changed"):
        rehearsal_module._verify_rehearsal_closure(closure)


def test_formal_scoring_rechecks_private_holdouts_after_scoring(
    prepared_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_paths = {
        "private_manifest": prepared_root / "private/private_manifest.json",
        **{
            f"private_holdout:{relative}": prepared_root / relative
            for relative in rehearsal_module._PRIVATE_HOLDOUT_PATHS
        },
    }
    closure = rehearsal_module._freeze_rehearsal_closure(private_paths)
    profile = rehearsal_module._profile_inputs(
        public_manifest=prepared_root / "public_manifest.json",
        profile="connection",
        staging=tmp_path / "profiles",
    )["within_k562"]["refit"]
    _expression, _labels, genes = rehearsal_module._read_profile_arrays(profile)
    raw_predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            (source, target, 0.0, False)
            for source in genes
            for target in genes
            if source != target
        ],
        columns=["source", "target", "score", "returned_by_method"],
    ).to_csv(raw_predictions, index=False)
    snapshot = rehearsal_module.freeze_rehearsal_predictions(
        source_path=raw_predictions,
        destination=tmp_path / "frozen-predictions.csv",
        expected_genes=genes,
    )
    changed_holdout = prepared_root / "private/within/k562/holdout.npz"

    def change_private_input_after_scoring(**_kwargs: object) -> dict[str, object]:
        changed_holdout.write_bytes(b"changed during private scoring")
        return {"average_precision": 0.0}

    monkeypatch.setattr(
        rehearsal_module,
        "_formal_scoring_subset",
        change_private_input_after_scoring,
    )

    with pytest.raises(TaskCRehearsalError, match="closure.*changed"):
        rehearsal_module._formal_scoring(
            condition="within_k562",
            predictions=snapshot,
            prepared_root=prepared_root,
            asset_root=tmp_path / "assets",
            work_dir=tmp_path / "work",
            seed=11,
            registry=object(),
            project_root=ROOT,
            public_profile_input=profile["input"],
            maximum_cells=20,
            rehearsal_closure=closure,
        )


def _small_profile_record(tmp_path: Path) -> dict[str, Path]:
    genes = np.asarray(["A", "B", "C", "D"])
    labels = np.asarray(["non-targeting"] * 12 + ["A"] * 6 + ["B"] * 6)
    expression = np.zeros((len(labels), len(genes)), dtype=np.float64)
    expression[:, :] = np.arange(len(labels), dtype=float)[:, None] * 0.001
    expression[labels == "A", 1] += 3.0
    expression[labels == "B", 2] -= 2.0
    input_path = tmp_path / "profile.npz"
    np.savez(
        input_path,
        expression_matrix=expression,
        interventions=labels,
        var_names=genes,
    )
    manifest = tmp_path / "profile_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "condition": "within_environment",
                "context_id": "k562",
                "direction": None,
                "contexts": [{"context_id": "k562", "role": "within_refit"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {"input": input_path, "manifest": manifest}


def test_formal_mean_difference_nulls_retain_all_40_when_empirical_gate_fails(
    tmp_path: Path,
) -> None:
    profile = _small_profile_record(tmp_path)
    predictions = rehearsal_module._scientific_null_predictions(
        method_id="mean_difference",
        profile_record=profile,
        seed=11,
        min_cells=2,
        hypersca_config_path=None,
    )

    with pytest.raises(TaskCRehearsalError, match="empirical gate"):
        rehearsal_module._run_formal_null_controls(
            method_id="mean_difference",
            predictions=predictions,
            profile_record=profile,
            seed=11,
            min_cells=2,
            hypersca_config_path=None,
            work_dir=tmp_path / "null-work",
        )

    resources = sorted((tmp_path / "null-work").rglob("analysis.resource.json"))
    assert len(resources) == 40
    analyses = [json.loads(path.read_text(encoding="utf-8")) for path in resources]
    assert all(record["status"] == "completed" for record in analyses)
    assert all(record["stage"] == "null_control" for record in analyses)
    assert all(record["input_sha256"].startswith("sha256:") for record in analyses)
    assert all(record["prediction_sha256"].startswith("sha256:") for record in analyses)
    assert all(record["timeout_seconds"] == 300.0 for record in analyses)
    assert all(record["measurement_availability"]["elapsed_seconds"] for record in analyses)
    assert all(record["peak_gpu_memory_bytes"] is None for record in analyses)
    assert all(
        record["measurement_availability"]["peak_gpu_memory_bytes"] is False
        for record in analyses
    )
    final_status = json.loads(
        (tmp_path / "null-work/null_controls/null_control_status.json").read_text()
    )
    assert final_status["status"] == "failed_null_control"


def test_formal_null_seed_sequence_has_no_overlap_across_types_and_contexts() -> None:
    seeds = {
        rehearsal_module._derive_formal_null_seed(
            base_seed=11,
            control_index=control_index,
            repeat=repeat,
            context_index=context_index,
            purpose=purpose,
        )
        for control_index in range(2)
        for repeat in range(20)
        for context_index in range(2)
        for purpose in range(2)
    }

    assert len(seeds) == 160
    assert all(0 <= value <= 2**64 - 1 for value in seeds)


def test_formal_null_hypersca_settings_preserve_selected_bootstrap(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.json"
    selected_payload = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    assert selected_payload["bootstrap_repeats"] == 20
    assert selected_payload["bootstrap_success_fraction"] == 0.8
    selected.write_text(json.dumps(selected_payload) + "\n", encoding="utf-8")
    created = rehearsal_module._materialize_formal_null_hypersca_config(
        selected_config=selected,
        destination=tmp_path / "formal-null.json",
    )

    payload = json.loads(created.read_text(encoding="utf-8"))
    assert payload == selected_payload


def test_formal_null_failure_is_retained_and_prevents_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _small_profile_record(tmp_path)
    predictions = rehearsal_module._scientific_null_predictions(
        method_id="mean_difference",
        profile_record=profile,
        seed=11,
        min_cells=2,
        hypersca_config_path=None,
    )
    calls = 0

    def supervised(**_kwargs: object) -> tuple[pd.DataFrame | None, dict[str, object]]:
        nonlocal calls
        calls += 1
        status = "failed" if calls == 1 else "completed"
        record: dict[str, object] = {
            "schema_version": "1.0",
            "component_kind": "null_analysis",
            "stage": "null_control",
            "status": status,
            "return_code": 2 if calls == 1 else 0,
            "timeout_seconds": 30.0,
            "elapsed_seconds": 0.01,
            "peak_rss_bytes": 1024,
            "peak_gpu_memory_bytes": 0,
            "written_disk_bytes": 1,
            "measurement_availability": {
                "elapsed_seconds": True,
                "peak_rss_bytes": True,
                "peak_gpu_memory_bytes": True,
                "written_disk_bytes": True,
            },
            "prediction_sha256": None if calls == 1 else "sha256:fixed",
            "scientific_status": None,
        }
        if calls == 1:
            record["reason"] = "deliberate repeat failure"
            return None, record
        return predictions, record

    monkeypatch.setattr(
        rehearsal_module, "_run_supervised_null_inference", supervised
    )
    with pytest.raises(TaskCRehearsalError, match="null-control analyses"):
        rehearsal_module._run_formal_null_controls(
            method_id="mean_difference",
            predictions=predictions,
            profile_record=profile,
            seed=11,
            min_cells=2,
            hypersca_config_path=None,
            work_dir=tmp_path / "null-work",
        )

    resources = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "null-work").rglob("analysis.resource.json"))
    ]
    assert len(resources) == 40
    failed = [record for record in resources if record["status"] == "failed"]
    assert len(failed) == 1
    assert "deliberate repeat failure" in failed[0]["reason"]
