from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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
    assert manifest["contexts"][0]["context_id"] == "k562"
    assert Path(manifest["contexts"][0]["input_path"]).is_absolute()
    assert manifest["config"]["values"]["prior_discount"] == 0.0
    assert manifest["gene_selection"]["ordered_genes"] == ["C", "A", "B"]
    assert manifest["gene_selection"]["gene_count"] == 3
    assert manifest["gene_selection"]["selection_id"] == "small-check-v1"
    assert manifest["gene_selection"]["selection_basis"]
    assert manifest["gene_selection"]["sha256"] == sha256_path(
        Path(prepared_run["gene_list"])
    )
    assert manifest["public_manifest"]["sha256"] == sha256_path(
        Path(prepared_run["public_manifest"])
    )
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
