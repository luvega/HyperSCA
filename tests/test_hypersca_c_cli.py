from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.causal.hypersca_c import HyperSCACError
from src.evaluation.task_c_data import (
    build_shared_task_c_split,
    load_task_c_dataset,
    materialize_task_c_split,
    sha256_path,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = {
    "raw_predictions.csv",
    "fit_summary.json",
    "method_status.json",
    "run_manifest.json",
}


def _write_context(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = ["non-targeting"] * 10 + [
        source for source in ("A", "B", "C", "D", "E") for _ in range(5)
    ]
    expression = rng.normal(size=(len(labels), 5)).astype(np.float32)
    expression[:, 1] = 1.2 * expression[:, 0] + rng.normal(
        scale=0.1, size=len(labels)
    )
    np.savez(
        path,
        expression_matrix=expression,
        interventions=np.asarray(labels),
        var_names=np.asarray(["A", "B", "C", "D", "E"]),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


@pytest.fixture
def prepared_run(tmp_path: Path) -> dict[str, object]:
    raw_k562 = tmp_path / "raw_k562.npz"
    raw_rpe1 = tmp_path / "raw_rpe1.npz"
    _write_context(raw_k562, 11)
    _write_context(raw_rpe1, 23)
    k562 = load_task_c_dataset(raw_k562, context_id="k562")
    rpe1 = load_task_c_dataset(raw_rpe1, context_id="rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    bundle = materialize_task_c_split(
        k562, rpe1, split, tmp_path / "bundle=public"
    )

    config = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    config.update(
        {
            "maximum_epochs": 3,
            "early_stopping_patience": 2,
            "bootstrap_repeats": 2,
        }
    )
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    gene_path = tmp_path / "genes.json"
    _write_json(
        gene_path,
        {
            "schema_version": "1.0",
            "selection_id": "small-check-v1",
            "selection_basis": "预先登记的小型核验基因集合",
            "genes": ["C", "A", "B"],
        },
    )
    return {
        "k562": Path(bundle["within"]["k562"]["refit"]),
        "rpe1": Path(bundle["within"]["rpe1"]["refit"]),
        "k562_train": Path(bundle["within"]["k562"]["train"]),
        "cross_source": Path(
            bundle["cross"]["k562_to_rpe1"]["source_refit"]
        ),
        "cross_target": Path(
            bundle["cross"]["k562_to_rpe1"]["target_adapt_refit"]
        ),
        "cross_target_train": Path(
            bundle["cross"]["k562_to_rpe1"]["target_adapt_train"]
        ),
        "reverse_source": Path(
            bundle["cross"]["rpe1_to_k562"]["source_refit"]
        ),
        "private_holdout": Path(bundle["within"]["k562"]["holdout"]),
        "public_manifest": Path(bundle["public_manifest"]),
        "config": config_path,
        "gene_list": gene_path,
        "output": tmp_path / "run",
    }


def _command(
    prepared: dict[str, object],
    *,
    output: Path | None = None,
    config: Path | None = None,
    gene_list: Path | None = None,
    k562: Path | None = None,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_hypersca_c.py"),
        "--context",
        f"k562={k562 or prepared['k562']}",
        "--context",
        f"rpe1={prepared['rpe1']}",
        "--config",
        str(config or prepared["config"]),
        "--gene-list",
        str(gene_list or prepared["gene_list"]),
        "--public-manifest",
        str(prepared["public_manifest"]),
        "--output-dir",
        str(output or prepared["output"]),
        "--seed",
        "11",
        "--device",
        "cpu",
    ]


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {
        name: ((directory / name).read_bytes(), (directory / name).stat().st_mtime_ns)
        for name in sorted(ARTIFACT_NAMES)
    }


def _rewrite_run_manifest(
    path: Path,
    payload: dict[str, object],
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    payload.pop("run_manifest_content_sha256", None)
    payload["run_manifest_content_sha256"] = run_module._payload_sha256(payload)
    write_json(path, payload)


def _all_failed_stability_result(
    *, context_ids: tuple[str, ...] = ("k562", "rpe1")
) -> object:
    from src.causal.hypersca_c_stability import (
        HyperSCAStabilityResult,
        build_stability_table,
    )

    predictions, summary = build_stability_table(
        [],
        ("C", "A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"C": 1.0, "A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
        expected_contexts=context_ids,
    )
    return HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=("repeat_0:failed", "repeat_1:failed"),
    )


def _one_success_stability_result() -> object:
    from src.causal.hypersca_c_stability import (
        HyperSCAStabilityResult,
        build_stability_table,
    )

    k562 = np.zeros((3, 3), dtype=np.float32)
    rpe1 = np.zeros((3, 3), dtype=np.float32)
    k562[0, 1] = -1.0
    rpe1[0, 1] = 1.0
    predictions, summary = build_stability_table(
        [{"k562": k562, "rpe1": rpe1}],
        ("C", "A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"C": 1.0, "A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
        expected_contexts=("k562", "rpe1"),
    )
    return HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=("repeat_1:failed",),
    )


def _within_refit_condition() -> dict[str, str]:
    return {
        "condition": "within_refit_k562_rpe1",
        "mode": "within",
        "direction": "none",
        "stage": "refit",
    }


def test_shared_run_validator_accepts_exact_one_repeat_context_aggregation() -> None:
    import src.causal.hypersca_c_run as run_module

    result = _one_success_stability_result()
    predictions, _, _ = run_module._validate_run_scientific_result(
        predictions=result.predictions,
        summary=result.summary,
        failures=result.failures,
        context_ids=("k562", "rpe1"),
        gene_names=("C", "A", "B"),
        requested_repeats=2,
        selection_threshold=0.1,
        seed=11,
        condition=_within_refit_condition(),
    )
    pd.testing.assert_frame_equal(predictions, result.predictions)


def test_shared_run_validator_rejects_synchronized_one_repeat_median_forgery() -> None:
    import src.causal.hypersca_c_run as run_module

    result = _one_success_stability_result()
    predictions = result.predictions.copy(deep=True)
    edge = (predictions["source"] == "C") & (predictions["target"] == "A")
    predictions.loc[edge, ["effect", "median_effect"]] = 0.5
    predictions.loc[edge, "direction"] = 1
    predictions.loc[edge, "score"] = 0.25
    predictions = predictions.sort_values(
        ["abstained", "score", "source", "target"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    with pytest.raises(HyperSCACError, match="一次|聚合|中位|selection"):
        run_module._validate_run_scientific_result(
            predictions=predictions,
            summary=result.summary,
            failures=result.failures,
            context_ids=("k562", "rpe1"),
            gene_names=("C", "A", "B"),
            requested_repeats=2,
            selection_threshold=0.1,
            seed=11,
            condition=_within_refit_condition(),
        )


def test_shared_run_validator_does_not_infer_multi_repeat_values_from_context_medians() -> None:
    import src.causal.hypersca_c_run as run_module
    from src.causal.hypersca_c_stability import build_stability_table

    first_k562 = np.zeros((3, 3), dtype=np.float32)
    first_rpe1 = np.zeros((3, 3), dtype=np.float32)
    second_k562 = np.zeros((3, 3), dtype=np.float32)
    second_rpe1 = np.zeros((3, 3), dtype=np.float32)
    first_k562[0, 1], second_k562[0, 1] = -2.0, -1.0
    first_rpe1[0, 1], second_rpe1[0, 1] = 1.0, 2.0
    predictions, summary = build_stability_table(
        [
            {"k562": first_k562, "rpe1": first_rpe1},
            {"k562": second_k562, "rpe1": second_rpe1},
        ],
        ("C", "A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"C": 1.0, "A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
        expected_contexts=("k562", "rpe1"),
    )
    edge = (predictions["source"] == "C") & (predictions["target"] == "A")
    predictions.loc[edge, ["effect", "median_effect"]] = 0.5
    predictions.loc[edge, "direction"] = 1
    predictions.loc[edge, "selection_frequency"] = 0.75
    predictions.loc[edge, "direction_agreement"] = 0.75
    predictions.loc[edge, "score"] = 0.28125
    predictions = predictions.sort_values(
        ["abstained", "score", "source", "target"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    validated, _, _ = run_module._validate_run_scientific_result(
        predictions=predictions,
        summary=summary,
        failures=(),
        context_ids=("k562", "rpe1"),
        gene_names=("C", "A", "B"),
        requested_repeats=2,
        selection_threshold=0.1,
        seed=11,
        condition=_within_refit_condition(),
    )
    assert float(validated.loc[0, "median_effect"]) == pytest.approx(0.5)


def test_hypersca_c_cli_writes_traced_raw_results_and_reuses_exact_run(
    prepared_run: dict[str, object], tmp_path: Path
) -> None:
    command = _command(prepared_run)
    completed = _run(command)
    output = Path(prepared_run["output"])

    assert set(path.name for path in output.iterdir()) == ARTIFACT_NAMES
    predictions = pd.read_csv(output / "raw_predictions.csv")
    assert len(predictions) == 6
    assert set(predictions["source"]) == {"A", "B", "C"}
    assert set(predictions["target"]) == {"A", "B", "C"}

    summary = json.loads((output / "fit_summary.json").read_text(encoding="utf-8"))
    status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == summary
    assert status["status"] == "completed_raw_inference"
    assert status["claim_level"] == "raw_inference_only"
    assert status["failure_count"] == (
        status["requested_bootstraps"] - status["successful_bootstraps"]
    )
    assert status["coverage"] == summary["coverage"]
    assert manifest["status"] == "completed_raw_inference"
    assert manifest["condition"] == "within_refit_k562_rpe1"
    assert manifest["mode"] == "within"
    assert manifest["direction"] is None
    assert manifest["stage"] == "refit"
    for field in ("condition", "mode", "direction", "stage"):
        assert manifest["run_identity"][field] == manifest[field]
    assert manifest["contexts"][0]["context_id"] == "k562"
    assert Path(manifest["contexts"][0]["input_path"]).is_absolute()
    assert manifest["config"]["values"]["prior_discount"] == 0.0
    assert (
        manifest["run_identity"]["config_sha256"]
        == manifest["config"]["sha256"]
    )
    assert manifest["gene_selection"]["ordered_genes"] == ["C", "A", "B"]
    assert manifest["gene_selection"]["gene_count"] == 3
    assert manifest["gene_selection"]["selection_id"] == "small-check-v1"
    assert manifest["gene_selection"]["selection_basis"]
    assert manifest["gene_selection"]["sha256"] == sha256_path(
        Path(prepared_run["gene_list"])
    )
    assert (
        manifest["run_identity"]["gene_list_sha256"]
        == manifest["gene_selection"]["sha256"]
    )
    assert manifest["public_manifest"]["sha256"] == sha256_path(
        Path(prepared_run["public_manifest"])
    )
    assert (
        manifest["run_identity"]["public_manifest_sha256"]
        == manifest["public_manifest"]["sha256"]
    )
    for context in manifest["contexts"]:
        identity_context = next(
            item
            for item in manifest["run_identity"]["contexts"]
            if item["context_id"] == context["context_id"]
        )
        assert identity_context["input_sha256"] == context["input_sha256"]
        assert identity_context["content_sha256"] == context["content_sha256"]
    assert manifest["artifacts"]["raw_predictions.csv"]["sha256"] == sha256_path(
        output / "raw_predictions.csv"
    )
    assert manifest["code"]["git_commit"]
    assert isinstance(manifest["code"]["dirty"], bool)
    assert manifest["code"]["code_state_sha256"].startswith("sha256:")
    assert (
        manifest["run_identity"]["code_state_sha256"]
        == manifest["code"]["code_state_sha256"]
    )
    assert manifest["run_identity"]["code_dirty"] == manifest["code"]["dirty"]
    assert manifest["duration_seconds"] >= 0.0

    before = _snapshot(output)
    reused = _run(command)
    assert json.loads(reused.stdout) == summary
    assert _snapshot(output) == before

    changed_config_payload = json.loads(Path(prepared_run["config"]).read_text())
    changed_config_payload["learning_rate"] = 0.02
    changed_config = tmp_path / "changed_config.json"
    _write_json(changed_config, changed_config_payload)
    rejected_config = _run(
        _command(prepared_run, config=changed_config), check=False
    )
    assert rejected_config.returncode != 0
    assert _snapshot(output) == before

    changed_genes = tmp_path / "changed_genes.json"
    _write_json(
        changed_genes,
        {
            "schema_version": "1.0",
            "selection_id": "different-check-v1",
            "selection_basis": "另一份预先登记集合",
            "genes": ["A", "B", "C"],
        },
    )
    rejected_genes = _run(
        _command(prepared_run, gene_list=changed_genes), check=False
    )
    assert rejected_genes.returncode != 0
    assert _snapshot(output) == before

    changed_input = tmp_path / "changed_input.npz"
    _write_context(changed_input, 97)
    rejected_input = _run(_command(prepared_run, k562=changed_input), check=False)
    assert rejected_input.returncode != 0
    assert _snapshot(output) == before

    raw_path = output / "raw_predictions.csv"
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    tampered = _snapshot(output)
    rejected_tamper = _run(command, check=False)
    assert rejected_tamper.returncode != 0
    assert "已改变" in rejected_tamper.stderr
    assert _snapshot(output) == tampered


def test_public_hypersca_output_validator_reuses_the_frozen_scientific_checks(
    prepared_run: dict[str, object],
    tmp_path: Path,
) -> None:
    from src.causal.hypersca_c_run import (
        run_hypersca_c,
        validate_hypersca_c_output_bundle,
    )

    output = tmp_path / "validated-output"
    run_hypersca_c(
        context_values=(
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ),
        config_path=Path(prepared_run["config"]),
        gene_list_path=Path(prepared_run["gene_list"]),
        public_manifest_path=Path(prepared_run["public_manifest"]),
        output_dir=output,
        seed=11,
        device="cpu",
    )

    validation_inputs = {
        "context_values": (
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ),
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "seed": 11,
        "device": "cpu",
    }
    validated = validate_hypersca_c_output_bundle(output, **validation_inputs)

    assert validated["requested_repeats"] == 2
    assert 0.0 <= validated["coverage"] <= 1.0

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_hash = f"sha256:{'0' * 64}"
    manifest["contexts"][0]["input_sha256"] = forged_hash
    manifest["run_identity"]["contexts"][0]["input_sha256"] = forged_hash
    _rewrite_run_manifest(manifest_path, manifest)

    with pytest.raises(HyperSCACError, match="输入|身份|清单|context"):
        validate_hypersca_c_output_bundle(output, **validation_inputs)


def test_deterministic_recompute_rejects_synchronized_zero_forgery(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import sha256_path, write_json

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _one_success_stability_result(),
    )
    output = tmp_path / "deterministic-recompute"
    arguments = {
        "context_values": (
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ),
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "seed": 11,
        "device": "cpu",
    }
    run_module.run_hypersca_c(output_dir=output, **arguments)

    raw_path = output / "raw_predictions.csv"
    forged = pd.read_csv(raw_path, keep_default_na=False)
    metric_columns = [
        "effect",
        "median_effect",
        "direction",
        "selection_frequency",
        "direction_agreement",
        "context_consistency",
        "effect_k562",
        "effect_rpe1",
        "score",
    ]
    forged.loc[:, metric_columns] = 0.0
    forged = forged.sort_values(
        ["abstained", "score", "source", "target"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    forged.to_csv(raw_path, index=False)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["raw_predictions.csv"]["sha256"] = sha256_path(raw_path)
    _rewrite_run_manifest(manifest_path, manifest)
    run_module.validate_hypersca_c_output_bundle(output, **arguments)

    with pytest.raises(HyperSCACError, match="重新计算|deterministic|一致"):
        run_module.recompute_hypersca_c_output_bundle(output, **arguments)


def test_existing_partial_output_is_rejected_without_changes(
    prepared_run: dict[str, object], tmp_path: Path
) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    marker = partial / "fit_summary.json"
    marker.write_text("partial", encoding="utf-8")
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    completed = _run(_command(prepared_run, output=partial), check=False)
    assert completed.returncode != 0
    assert "无法运行 HyperSCA-C" in completed.stderr
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
    assert set(partial.iterdir()) == {marker}


def test_gene_limit_and_missing_gene_list_fail_before_fit(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module

    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit must not be called")

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", forbidden_fit)
    too_many = tmp_path / "too_many.json"
    _write_json(
        too_many,
        {
            "schema_version": "1.0",
            "selection_id": "over-limit-v1",
            "selection_basis": "用于验证当前核验运行上限",
            "genes": [f"G{index}" for index in range(run_module.MAX_VERIFIED_GENES + 1)],
        },
    )
    with pytest.raises(HyperSCACError, match="当前核验运行上限"):
        run_module.run_hypersca_c(
            context_values=[f"k562={prepared_run['k562']}"],
            config_path=Path(prepared_run["config"]),
            gene_list_path=too_many,
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=tmp_path / "must-not-exist",
            seed=11,
            device="cpu",
        )
    assert not called
    assert not (tmp_path / "must-not-exist").exists()

    spec = importlib.util.spec_from_file_location(
        "run_hypersca_c_script", ROOT / "scripts/run_hypersca_c.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with pytest.raises(SystemExit):
        script.main(
            [
                "--context",
                f"k562={prepared_run['k562']}",
                "--config",
                str(prepared_run["config"]),
                "--public-manifest",
                str(prepared_run["public_manifest"]),
                "--output-dir",
                str(tmp_path / "missing-gene-list"),
                "--seed",
                "11",
            ]
        )
    assert not called
    assert not (tmp_path / "missing-gene-list").exists()


@pytest.mark.parametrize(
    "context_values",
    [
        ["unknown=/tmp/data.npz"],
        ["k562=/tmp/a.npz", "k562=/tmp/b.npz"],
        ["k562="],
    ],
)
def test_context_identifiers_are_strict_before_fit(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_values: list[str],
) -> None:
    import src.causal.hypersca_c_run as run_module

    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit must not be called")

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", forbidden_fit)
    with pytest.raises(HyperSCACError):
        run_module.run_hypersca_c(
            context_values=context_values,
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=tmp_path / "invalid-context",
            seed=11,
            device="cpu",
        )
    assert not called
    assert not (tmp_path / "invalid-context").exists()


@pytest.mark.parametrize(
    "case",
    [
        "within_label_swap",
        "within_mixed_stages",
        "cross_label_swap",
        "cross_target_only",
        "cross_source_only",
        "cross_mixed_directions",
        "cross_mixed_stages",
        "within_cross_mixed",
    ],
)
def test_context_files_must_form_one_complete_registered_condition_before_fit(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    import src.causal.hypersca_c_run as run_module

    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit must not be called")

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", forbidden_fit)
    cases = {
        "within_label_swap": [f"rpe1={prepared_run['k562']}"],
        "within_mixed_stages": [
            f"k562={prepared_run['k562_train']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "cross_label_swap": [
            f"k562={prepared_run['cross_target']}",
            f"rpe1={prepared_run['cross_source']}",
        ],
        "cross_target_only": [f"rpe1={prepared_run['cross_target']}"],
        "cross_source_only": [f"k562={prepared_run['cross_source']}"],
        "cross_mixed_directions": [
            f"k562={prepared_run['cross_source']}",
            f"rpe1={prepared_run['reverse_source']}",
        ],
        "cross_mixed_stages": [
            f"k562={prepared_run['cross_source']}",
            f"rpe1={prepared_run['cross_target_train']}",
        ],
        "within_cross_mixed": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['cross_target']}",
        ],
    }
    with pytest.raises(HyperSCACError, match="condition|组合|匹配|绑定|完整"):
        run_module.run_hypersca_c(
            context_values=cases[case],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=tmp_path / f"invalid-condition-{case}",
            seed=11,
            device="cpu",
        )
    assert not called
    assert not (tmp_path / f"invalid-condition-{case}").exists()


@pytest.mark.parametrize("case", ["within_single", "within_joint", "cross"])
def test_complete_within_and_cross_conditions_reach_fit(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    import src.causal.hypersca_c_run as run_module

    class FitReached(RuntimeError):
        pass

    def reached_fit(*args: object, **kwargs: object) -> object:
        raise FitReached

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", reached_fit)
    cases = {
        "within_single": [f"k562={prepared_run['k562']}"],
        "within_joint": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "cross": [
            f"k562={prepared_run['cross_source']}",
            f"rpe1={prepared_run['cross_target']}",
        ],
    }
    with pytest.raises(FitReached):
        run_module.run_hypersca_c(
            context_values=cases[case],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=tmp_path / f"legal-condition-{case}",
            seed=11,
            device="cpu",
        )
    assert not (tmp_path / f"legal-condition-{case}").exists()


def test_public_only_contract_rejects_private_and_symbolic_link_inputs_before_fit(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module

    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit must not be called")

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", forbidden_fit)
    private_dir = tmp_path / "private"
    private_manifest = private_dir / "public_manifest.json"
    private_dir.mkdir()
    private_manifest.write_bytes(Path(prepared_run["public_manifest"]).read_bytes())
    with pytest.raises(HyperSCACError, match="公开清单"):
        run_module.run_hypersca_c(
            context_values=[f"k562={prepared_run['k562']}"],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=private_manifest,
            output_dir=tmp_path / "private-rejected",
            seed=11,
            device="cpu",
        )

    linked_input = tmp_path / "linked.npz"
    try:
        linked_input.symlink_to(Path(prepared_run["k562"]))
    except OSError:
        pytest.skip("当前文件系统不支持符号链接")
    with pytest.raises(HyperSCACError, match="符号链接"):
        run_module.run_hypersca_c(
            context_values=[f"k562={linked_input}"],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=tmp_path / "link-rejected",
            seed=11,
            device="cpu",
        )
    assert not called


@pytest.mark.parametrize("linked_name", ["config", "gene_list"])
def test_settings_inputs_reject_symbolic_links_before_fit(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_name: str,
) -> None:
    import src.causal.hypersca_c_run as run_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: pytest.fail("fit must not be called"),
    )
    linked = tmp_path / f"linked_{linked_name}.json"
    try:
        linked.symlink_to(Path(prepared_run[linked_name]))
    except OSError:
        pytest.skip("当前文件系统不支持符号链接")
    arguments = {
        "context_values": [f"k562={prepared_run['k562']}"],
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "output_dir": tmp_path / f"linked_{linked_name}_output",
        "seed": 11,
        "device": "cpu",
    }
    arguments[f"{linked_name}_path"] = linked
    with pytest.raises(HyperSCACError, match="符号链接"):
        run_module.run_hypersca_c(**arguments)
    assert not Path(arguments["output_dir"]).exists()


def test_public_manifest_requires_the_exact_public_artifact_inventory(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: pytest.fail("fit must not be called"),
    )
    manifest_path = Path(prepared_run["public_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    unused = next(
        relative
        for relative in payload["files"]
        if relative
        not in {
            "within/k562/refit.npz",
            "within/rpe1/refit.npz",
        }
    )
    digest = payload["files"].pop(unused)
    payload["files"]["unregistered/public_file.npz"] = digest
    corrupt = manifest_path.parent / "corrupt_public_manifest.json"
    _write_json(corrupt, payload)
    with pytest.raises(HyperSCACError, match="完整公开文件清单"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=corrupt,
            output_dir=tmp_path / "corrupt-manifest-output",
            seed=11,
            device="cpu",
        )
    assert not (tmp_path / "corrupt-manifest-output").exists()


def test_public_inventory_rejects_private_hardlink_alias_before_fit(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit must not be called")

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", forbidden_fit)
    public_path = Path(prepared_run["cross_target_train"])
    private_path = Path(prepared_run["private_holdout"])
    public_path.unlink()
    os.link(private_path, public_path)
    manifest_path = Path(prepared_run["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = public_path.relative_to(manifest_path.parent).as_posix()
    manifest["files"][relative] = sha256_path(public_path)
    write_json(manifest_path, manifest)

    with pytest.raises(HyperSCACError, match="硬链接|inode|公开库存"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=manifest_path,
            output_dir=tmp_path / "private-hardlink-rejected",
            seed=11,
            device="cpu",
        )
    assert not called


@pytest.mark.parametrize("tamper", ["target_labels", "gene_hash"])
def test_selected_public_data_must_match_manifest_biology_before_fit(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    called = False

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fit must not be called")

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", forbidden_fit)
    manifest_path = Path(prepared_run["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "target_labels":
        target_path = Path(prepared_run["cross_target"])
        target = load_task_c_dataset(target_path, context_id="rpe1")
        labels = target.interventions.copy()
        labels[0] = "A"
        np.savez_compressed(
            target_path,
            expression_matrix=target.expression,
            interventions=labels,
            var_names=np.asarray(target.gene_names),
        )
        relative = target_path.relative_to(manifest_path.parent).as_posix()
        manifest["files"][relative] = sha256_path(target_path)
    else:
        manifest["gene_names_sha256"] = f"sha256:{'0' * 64}"
        manifest["materialization_identity"]["gene_names_sha256"] = (
            manifest["gene_names_sha256"]
        )
    write_json(manifest_path, manifest)

    with pytest.raises(HyperSCACError, match="干预|基因|语义|gene"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['cross_source']}",
                f"rpe1={prepared_run['cross_target']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=manifest_path,
            output_dir=tmp_path / f"biological-tamper-{tamper}",
            seed=11,
            device="cpu",
        )
    assert not called


def test_fresh_run_hashes_public_inventory_once_per_inode_plus_selected_postcheck(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module
    import src.evaluation.task_c_data as data_module

    manifest_path = Path(prepared_run["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_paths = {
        (manifest_path.parent / relative).resolve()
        for relative in manifest["files"]
    }
    unique_inodes = {
        (path.stat().st_dev, path.stat().st_ino) for path in public_paths
    }
    selected_inodes = {
        (Path(prepared_run[name]).stat().st_dev, Path(prepared_run[name]).stat().st_ino)
        for name in ("k562", "rpe1")
    }
    original_capture = run_module._capture_file_snapshot
    original_load = data_module.np.load
    original_sha256 = data_module.sha256_path
    reads_by_inode: dict[tuple[int, int], int] = {}

    def count_path(path: object) -> None:
        if not isinstance(path, (str, os.PathLike)):
            return
        candidate = Path(path).resolve()
        if candidate not in public_paths:
            return
        stat = candidate.stat()
        inode = (stat.st_dev, stat.st_ino)
        reads_by_inode[inode] = reads_by_inode.get(inode, 0) + 1

    def counted_capture(*args: object, **kwargs: object) -> object:
        result = original_capture(*args, **kwargs)
        count_path(args[0])
        return result

    def counted_load(source: object, *args: object, **kwargs: object) -> object:
        count_path(source)
        return original_load(source, *args, **kwargs)

    def counted_sha256(path: Path | str) -> str:
        count_path(path)
        return original_sha256(path)

    monkeypatch.setattr(run_module, "_capture_file_snapshot", counted_capture)
    monkeypatch.setattr(data_module.np, "load", counted_load)
    monkeypatch.setattr(data_module, "sha256_path", counted_sha256)
    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    run_module.run_hypersca_c(
        context_values=[
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        config_path=Path(prepared_run["config"]),
        gene_list_path=Path(prepared_run["gene_list"]),
        public_manifest_path=manifest_path,
        output_dir=tmp_path / "counted-fresh-run",
        seed=11,
        device="cpu",
    )

    assert set(reads_by_inode) == unique_inodes
    assert sum(reads_by_inode.values()) <= len(unique_inodes) + len(selected_inodes)
    assert all(
        count <= (2 if inode in selected_inodes else 1)
        for inode, count in reads_by_inode.items()
    )


def test_exact_reuse_hashes_public_inventory_at_most_once_per_inode(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module
    import src.evaluation.task_c_data as data_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    output = tmp_path / "counted-reuse"
    arguments = {
        "context_values": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "output_dir": output,
        "seed": 11,
        "device": "cpu",
    }
    run_module.run_hypersca_c(**arguments)

    manifest_path = Path(prepared_run["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_paths = {
        (manifest_path.parent / relative).resolve()
        for relative in manifest["files"]
    }
    unique_inodes = {
        (path.stat().st_dev, path.stat().st_ino) for path in public_paths
    }
    original_capture = run_module._capture_file_snapshot
    original_load = data_module.np.load
    original_sha256 = data_module.sha256_path
    reads_by_inode: dict[tuple[int, int], int] = {}

    def count_path(path: object) -> None:
        if not isinstance(path, (str, os.PathLike)):
            return
        candidate = Path(path).resolve()
        if candidate not in public_paths:
            return
        stat = candidate.stat()
        inode = (stat.st_dev, stat.st_ino)
        reads_by_inode[inode] = reads_by_inode.get(inode, 0) + 1

    def counted_capture(*args: object, **kwargs: object) -> object:
        result = original_capture(*args, **kwargs)
        count_path(args[0])
        return result

    def counted_load(source: object, *args: object, **kwargs: object) -> object:
        count_path(source)
        return original_load(source, *args, **kwargs)

    def counted_sha256(path: Path | str) -> str:
        count_path(path)
        return original_sha256(path)

    monkeypatch.setattr(run_module, "_capture_file_snapshot", counted_capture)
    monkeypatch.setattr(data_module.np, "load", counted_load)
    monkeypatch.setattr(data_module, "sha256_path", counted_sha256)
    run_module.run_hypersca_c(**arguments)

    assert set(reads_by_inode) == unique_inodes
    assert all(count == 1 for count in reads_by_inode.values())


def test_unselected_public_file_changed_during_fit_is_rejected_by_post_stat(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module

    unselected = Path(prepared_run["cross_target_train"])

    def mutate_unselected(*args: object, **kwargs: object) -> object:
        unselected.write_bytes(unselected.read_bytes() + b"\n")
        return _all_failed_stability_result()

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", mutate_unselected)
    output = tmp_path / "unselected-public-changed"
    with pytest.raises(HyperSCACError, match="公开库存|变化|改变"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=output,
            seed=11,
            device="cpu",
        )
    assert not output.exists()


def test_gene_list_is_strict_and_selected_gene_must_exist(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: pytest.fail("fit must not be called"),
    )
    invalid_records = [
        {
            "schema_version": "1.0",
            "selection_id": "duplicate-v1",
            "selection_basis": "重复基因应当被拒绝",
            "genes": ["A", "A"],
        },
        {
            "schema_version": "1.0",
            "selection_id": "missing-v1",
            "selection_basis": "不存在的基因应当被拒绝",
            "genes": ["A", "NOT_PRESENT"],
        },
        {
            "schema_version": "1.0",
            "selection_id": "bad/path",
            "selection_basis": "标识不能像路径",
            "genes": ["A", "B"],
        },
        {
            "schema_version": "1.0",
            "selection_id": "extra-v1",
            "selection_basis": "额外字段应当被拒绝",
            "genes": ["A", "B"],
            "extra": True,
        },
    ]
    for index, record in enumerate(invalid_records):
        path = tmp_path / f"invalid_{index}.json"
        _write_json(path, record)
        with pytest.raises(HyperSCACError):
            run_module.run_hypersca_c(
                context_values=[f"k562={prepared_run['k562']}"],
                config_path=Path(prepared_run["config"]),
                gene_list_path=path,
                public_manifest_path=Path(prepared_run["public_manifest"]),
                output_dir=tmp_path / f"invalid_output_{index}",
                seed=11,
                device="cpu",
            )
        assert not (tmp_path / f"invalid_output_{index}").exists()


@pytest.mark.parametrize(
    "changed_input", ["config", "gene_list", "public_manifest", "context"]
)
def test_input_changed_during_fit_is_rejected_before_any_artifact_is_written(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.causal.hypersca_c_stability import (
        HyperSCAStabilityResult,
        build_stability_table,
    )

    predictions, summary = build_stability_table(
        [],
        ("C", "A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"C": 1.0, "A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
        expected_contexts=("k562", "rpe1"),
    )
    result = HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=("repeat_0:failed", "repeat_1:failed"),
    )
    paths = {
        "config": Path(prepared_run["config"]),
        "gene_list": Path(prepared_run["gene_list"]),
        "public_manifest": Path(prepared_run["public_manifest"]),
        "context": Path(prepared_run["k562"]),
    }

    def mutate_then_return(*args: object, **kwargs: object) -> object:
        target = paths[changed_input]
        target.write_bytes(target.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(run_module, "fit_stable_hypersca_c", mutate_then_return)
    output = tmp_path / f"changed-during-fit-{changed_input}"
    with pytest.raises(HyperSCACError, match="变化|改变|输入"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=output,
            seed=11,
            device="cpu",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "forgery",
    [
        "self_edge",
        "duplicate_edge",
        "wrong_gene",
        "nan_effect",
        "unknown_column",
        "missing_context_effect",
        "wrong_context_effect",
        "wrong_context_consistency",
        "median_outside_context_range",
        "wrong_requested_repeats",
    ],
)
def test_run_boundary_rejects_forged_scientific_results_before_writing(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    import src.causal.hypersca_c_run as run_module

    valid = _all_failed_stability_result()
    predictions = valid.predictions.copy(deep=True)
    summary = dict(valid.summary)
    failures = tuple(valid.failures)
    if forgery == "self_edge":
        predictions.loc[0, "target"] = predictions.loc[0, "source"]
    elif forgery == "duplicate_edge":
        predictions.loc[1, ["source", "target"]] = predictions.loc[
            0, ["source", "target"]
        ].to_numpy()
    elif forgery == "wrong_gene":
        predictions["source"] = predictions["source"].replace("A", "X")
        predictions["target"] = predictions["target"].replace("A", "X")
    elif forgery == "nan_effect":
        predictions.loc[0, "effect"] = np.nan
    elif forgery == "unknown_column":
        predictions["unregistered_metric"] = 0.0
    elif forgery == "missing_context_effect":
        predictions = predictions.drop(columns="effect_rpe1")
    elif forgery == "wrong_context_effect":
        predictions = predictions.rename(columns={"effect_rpe1": "effect_HEK293"})
    elif forgery == "wrong_context_consistency":
        predictions.loc[0, "context_consistency"] = 1.0
    elif forgery == "median_outside_context_range":
        predictions.loc[0, ["effect", "median_effect"]] = 3.0
        predictions.loc[0, ["effect_k562", "effect_rpe1"]] = [1.0, 2.0]
        predictions.loc[0, "direction"] = 1
    elif forgery == "wrong_requested_repeats":
        summary["requested_repeats"] = 3
        failures = (*failures, "repeat_2:failed")
    forged = SimpleNamespace(
        predictions=predictions,
        summary=summary,
        failures=failures,
    )
    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: forged,
    )
    output = tmp_path / f"forged-{forgery}"
    with pytest.raises(HyperSCACError):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=output,
            seed=11,
            device="cpu",
        )
    assert not output.exists()


def test_single_context_effect_must_equal_the_overall_median(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module

    valid = _all_failed_stability_result(context_ids=("k562",))
    predictions = valid.predictions.copy(deep=True)
    predictions.loc[0, "effect_k562"] = 1.0
    forged = SimpleNamespace(
        predictions=predictions,
        summary=dict(valid.summary),
        failures=tuple(valid.failures),
    )
    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: forged,
    )
    output = tmp_path / "single-context-contradiction"
    with pytest.raises(HyperSCACError, match="context|细胞环境|中位"):
        run_module.run_hypersca_c(
            context_values=[f"k562={prepared_run['k562']}"],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=output,
            seed=11,
            device="cpu",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "tampered_artifact", ["predictions", "cross_column", "summary", "status"]
)
def test_reuse_rejects_semantic_tampering_even_with_synchronized_hashes(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_artifact: str,
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.evaluation.task_c_data import write_json

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    output = tmp_path / f"semantic-tamper-{tampered_artifact}"
    arguments = {
        "context_values": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "output_dir": output,
        "seed": 11,
        "device": "cpu",
    }
    run_module.run_hypersca_c(**arguments)

    if tampered_artifact == "predictions":
        artifact_name = "raw_predictions.csv"
        predictions = pd.read_csv(output / artifact_name)
        predictions["unregistered_metric"] = 0.0
        predictions.to_csv(output / artifact_name, index=False)
    elif tampered_artifact == "cross_column":
        artifact_name = "raw_predictions.csv"
        predictions = pd.read_csv(output / artifact_name)
        predictions.loc[0, "context_consistency"] = 1.0
        predictions.to_csv(output / artifact_name, index=False)
    elif tampered_artifact == "summary":
        artifact_name = "fit_summary.json"
        summary = json.loads((output / artifact_name).read_text(encoding="utf-8"))
        summary["requested_repeats"] = 3
        write_json(output / artifact_name, summary)
    else:
        artifact_name = "method_status.json"
        status = json.loads((output / artifact_name).read_text(encoding="utf-8"))
        status["usable_for_ranking"] = not status["usable_for_ranking"]
        write_json(output / artifact_name, status)

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][artifact_name]["sha256"] = sha256_path(
        output / artifact_name
    )
    manifest.pop("run_manifest_content_sha256")
    manifest["run_manifest_content_sha256"] = run_module._payload_sha256(manifest)
    write_json(manifest_path, manifest)

    with pytest.raises(HyperSCACError):
        run_module.run_hypersca_c(**arguments)


@pytest.mark.parametrize(
    "tamper",
    [
        "schema",
        "method",
        "status",
        "seed",
        "device",
        "contexts",
        "config",
        "gene_order",
        "public_manifest",
        "code_commit",
        "condition",
        "extra_field",
        "reverse_time",
        "duration_string",
        "duration_nan",
        "duration_huge_integer",
    ],
)
def test_reuse_cross_checks_complete_run_manifest_semantics(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    import src.causal.hypersca_c_run as run_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    output = tmp_path / f"manifest-static-tamper-{tamper}"
    arguments = {
        "context_values": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "output_dir": output,
        "seed": 11,
        "device": "cpu",
    }
    run_module.run_hypersca_c(**arguments)
    before = _snapshot(output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if tamper == "schema":
        manifest["schema_version"] = "9.0"
    elif tamper == "method":
        manifest["method_id"] = "other_method"
    elif tamper == "status":
        manifest["status"] = "passed_real_rehearsal"
    elif tamper == "seed":
        manifest["seed"] = 12
    elif tamper == "device":
        manifest["device"] = "cuda"
    elif tamper == "contexts":
        manifest["contexts"][0]["content_sha256"] = f"sha256:{'0' * 64}"
    elif tamper == "config":
        manifest["config"]["values"]["learning_rate"] = 0.02
    elif tamper == "gene_order":
        manifest["gene_selection"]["ordered_genes"] = ["X", "Y", "Z"]
    elif tamper == "public_manifest":
        manifest["public_manifest"]["path"] = "/tmp/other-public-manifest.json"
    elif tamper == "code_commit":
        manifest["code"]["git_commit"] = "0" * 40
    elif tamper == "condition":
        manifest["stage"] = "train"
    elif tamper == "extra_field":
        manifest["unregistered_claim"] = "validated"
    elif tamper == "reverse_time":
        manifest["started_utc"] = "2026-08-13T12:00:02Z"
        manifest["completed_utc"] = "2026-08-13T12:00:01Z"
    elif tamper == "duration_string":
        manifest["duration_seconds"] = "NaN"
    elif tamper == "duration_nan":
        manifest["duration_seconds"] = float("nan")
    else:
        manifest["duration_seconds"] = 10**400

    if tamper == "duration_nan":
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        _rewrite_run_manifest(manifest_path, manifest)

    tampered_snapshot = _snapshot(output)
    with pytest.raises(HyperSCACError):
        run_module.run_hypersca_c(**arguments)
    assert tampered_snapshot != before
    assert _snapshot(output) == tampered_snapshot


def test_new_output_rejects_reversed_run_timestamps(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module

    timestamps = iter(
        ["2026-08-13T12:00:02Z", "2026-08-13T12:00:01Z"]
    )
    monkeypatch.setattr(run_module, "_utc_now", lambda: next(timestamps))
    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    output = tmp_path / "reversed-new-run-time"
    with pytest.raises(HyperSCACError, match="时间|UTC|时长"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=output,
            seed=11,
            device="cpu",
        )
    assert not output.exists()


def test_new_output_rejects_a_staging_file_with_external_hardlink(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    original = run_module._write_csv_atomic
    outside = tmp_path / "staging-output-alias.csv"

    def write_and_link(path: Path, predictions: pd.DataFrame) -> None:
        original(path, predictions)
        os.link(path, outside)

    monkeypatch.setattr(run_module, "_write_csv_atomic", write_and_link)
    output = tmp_path / "hardlinked-new-output"
    with pytest.raises(HyperSCACError, match="硬链接|普通文件|link"):
        run_module.run_hypersca_c(
            context_values=[
                f"k562={prepared_run['k562']}",
                f"rpe1={prepared_run['rpe1']}",
            ],
            config_path=Path(prepared_run["config"]),
            gene_list_path=Path(prepared_run["gene_list"]),
            public_manifest_path=Path(prepared_run["public_manifest"]),
            output_dir=output,
            seed=11,
            device="cpu",
        )
    assert not output.exists()


def test_output_reuse_rejects_a_file_with_external_hardlink(
    prepared_run: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as run_module

    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    output = tmp_path / "hardlinked-existing-output"
    arguments = {
        "context_values": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "output_dir": output,
        "seed": 11,
        "device": "cpu",
    }
    run_module.run_hypersca_c(**arguments)
    os.link(output / "raw_predictions.csv", tmp_path / "existing-output-alias.csv")
    with pytest.raises(HyperSCACError, match="硬链接|普通文件|link"):
        run_module.run_hypersca_c(**arguments)


def test_all_bootstrap_failures_remain_visible_and_unusable(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module
    from src.causal.hypersca_c_stability import build_stability_table
    from src.causal.hypersca_c_stability import HyperSCAStabilityResult

    predictions, summary = build_stability_table(
        [],
        ("C", "A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"C": 1.0, "A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
        expected_contexts=("k562", "rpe1"),
    )
    failure_result = HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=("repeat_0:failed", "repeat_1:failed"),
    )
    monkeypatch.setattr(
        run_module, "fit_stable_hypersca_c", lambda *args, **kwargs: failure_result
    )
    output = tmp_path / "all-failed"
    run_module.run_hypersca_c(
        context_values=[
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        config_path=Path(prepared_run["config"]),
        gene_list_path=Path(prepared_run["gene_list"]),
        public_manifest_path=Path(prepared_run["public_manifest"]),
        output_dir=output,
        seed=11,
        device="cpu",
    )
    status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed_raw_inference"
    assert status["successful_bootstraps"] == 0
    assert status["failure_count"] == status["requested_bootstraps"] == 2
    assert status["coverage"] == 0.0
    assert status["usable_for_ranking"] is False


def test_root_untracked_code_changes_identity_and_blocks_existing_output_reuse(
    prepared_run: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.causal.hypersca_c_run as run_module

    repository = tmp_path / "code-repository"
    repository.mkdir()
    tracked = repository / "tracked.txt"
    tracked.write_text("registered\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=HyperSCA Test",
            "-c",
            "user.email=hypersca@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=repository,
        check=True,
    )
    tracked.write_text("already dirty\n", encoding="utf-8")
    sitecustomize = repository / "sitecustomize.py"
    sitecustomize.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(run_module, "_ROOT", repository)

    first_state = run_module._git_state()
    assert first_state["dirty"] is True
    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: _all_failed_stability_result(),
    )
    output = tmp_path / "code-state-output"
    arguments = {
        "context_values": [
            f"k562={prepared_run['k562']}",
            f"rpe1={prepared_run['rpe1']}",
        ],
        "config_path": Path(prepared_run["config"]),
        "gene_list_path": Path(prepared_run["gene_list"]),
        "public_manifest_path": Path(prepared_run["public_manifest"]),
        "output_dir": output,
        "seed": 11,
        "device": "cpu",
    }
    run_module.run_hypersca_c(**arguments)

    sitecustomize.write_text("VERSION = 2\n", encoding="utf-8")
    second_state = run_module._git_state()
    assert second_state["dirty"] is True
    assert second_state["code_state_sha256"] != first_state["code_state_sha256"]
    monkeypatch.setattr(
        run_module,
        "fit_stable_hypersca_c",
        lambda *args, **kwargs: pytest.fail("changed code must not enter fit"),
    )
    before = _snapshot(output)
    with pytest.raises(HyperSCACError, match="代码|已有输出|不能覆盖"):
        run_module.run_hypersca_c(**arguments)
    assert _snapshot(output) == before
