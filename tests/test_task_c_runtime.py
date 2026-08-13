from __future__ import annotations

from collections.abc import Sequence
import json
import hashlib
import io
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation import task_c_runtime as runtime_module
from src.evaluation import task_c_method_run as method_run_module
from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistryError,
    load_task_c_method_registry,
)
from src.evaluation.task_c_predictions import normalize_task_c_predictions
from src.evaluation.task_c_data import (
    build_shared_task_c_split,
    load_task_c_dataset,
    materialize_task_c_split,
)
from src.evaluation.task_c_runtime import (
    TaskCRuntimeError,
    _run_bounded_command,
    _parse_maximum_resident_kib,
    bootstrap_task_c_methods,
    classify_publication_only_method,
    run_isolated_method,
)
from src.evaluation.task_c_method_run import (
    MAXIMUM_TASK_C_RUN_GENES,
    TaskCMethodRunError,
    _standardize_and_concatenate,
    _capture_live_conda_environment,
    _validate_hypersca_inner_bundle,
    build_task_c_method_command,
    materialize_task_c_derived_input,
    read_task_c_raw_predictions,
    run_task_c_method,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/task_c_methods_v1.json"
TASK_C_RUNTIME_CLOSURE = (
    "src/evaluation/task_c_method_run.py",
    "src/evaluation/task_c_profile_input.py",
    "scripts/run_task_c_method.py",
    "src/evaluation/task_c_predictions.py",
    "src/evaluation/task_c_runtime.py",
    "src/evaluation/task_c_method_registry.py",
    "src/evaluation/task_c_data.py",
    "src/evaluation/task_c_benchmark.py",
    "src/causal/hypersca_c.py",
    "src/causal/hypersca_c_stability.py",
    "src/causal/hypersca_c_run.py",
    "scripts/run_hypersca_c.py",
    "scripts/task_c_workers/causalbench_worker.py",
    "scripts/task_c_workers/psgrn_worker.py",
)


def test_publication_only_method_gets_explicit_unavailable_status() -> None:
    registry = load_task_c_method_registry(REGISTRY)

    status = classify_publication_only_method(registry.methods["betterboost"])

    assert set(status) == {
        "schema_version",
        "method_id",
        "status",
        "publication",
        "reason",
    }
    assert status["status"] == "official_assets_unavailable"
    assert status["publication"].startswith("https://openreview.net/")


def test_hypersca_inner_validation_recomputes_only_for_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.causal.hypersca_c_run as hypersca_run

    calls: list[str] = []
    monkeypatch.setattr(
        hypersca_run,
        "validate_hypersca_c_output_bundle",
        lambda *args, **kwargs: calls.append("validate"),
    )
    monkeypatch.setattr(
        hypersca_run,
        "recompute_hypersca_c_output_bundle",
        lambda *args, **kwargs: calls.append("recompute"),
    )
    fixed_inputs = {
        "config_path": tmp_path / "config.json",
        "gene_list_path": tmp_path / "genes.json",
        "public_manifest_path": tmp_path / "public.json",
        "seed": 11,
        "device": "cpu",
    }

    _validate_hypersca_inner_bundle(
        tmp_path / "nested", fixed_inputs=fixed_inputs, recompute=False
    )
    _validate_hypersca_inner_bundle(
        tmp_path / "nested", fixed_inputs=fixed_inputs, recompute=True
    )

    assert calls == ["validate", "recompute"]


def test_live_conda_environment_identity_is_canonical_and_bounded() -> None:
    calls: list[tuple[tuple[str, ...], object]] = []

    def fake_runner(command: Sequence[str], **kwargs: object) -> object:
        calls.append((tuple(command), kwargs.get("timeout")))
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    {"name": "zlib", "version": "1.3", "build_string": "h1"},
                    {"name": "numpy", "version": "2.0", "build_string": "py310"},
                ]
            ),
            stderr="",
            returncode=0,
        )

    captured = _capture_live_conda_environment(
        "hypersca-task-c-causalbench", run_command=fake_runner
    )

    assert captured["packages"] == [
        {"name": "numpy", "version": "2.0", "build_string": "py310"},
        {"name": "zlib", "version": "1.3", "build_string": "h1"},
    ]
    assert str(captured["sha256"]).startswith("sha256:")
    assert calls == [
        (
            (
                "conda",
                "list",
                "-n",
                "hypersca-task-c-causalbench",
                "--json",
            ),
            60,
        )
    ]
    assert "/" not in json.dumps(captured)


def _live_environment_record(version: str) -> dict[str, object]:
    packages = [{"name": "python", "version": version, "build_string": "h0"}]
    canonical = {
        "schema_version": "1.0",
        "environment": "hypersca-task-c-causalbench",
        "packages": packages,
    }
    encoded = (
        json.dumps(canonical, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        "environment": canonical["environment"],
        "packages": packages,
        "sha256": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    }


def _fake_external_asset_snapshots(
    tmp_path: Path,
    *,
    packages: list[dict[str, str]],
) -> dict[str, object]:
    root = tmp_path / "fake-assets"
    environment_root = root / "environment_manifests"
    environment_root.mkdir(parents=True)
    registry_hash = f"sha256:{hashlib.sha256(REGISTRY.read_bytes()).hexdigest()}"
    records = {
        "bootstrap_identity.json": {"registry_sha256": registry_hash},
        "bootstrap_manifest.json": {"schema_version": "1.0"},
        "bootstrap_status.json": {
            "schema_version": "1.0",
            "status": "assets_and_environments_recorded",
        },
        "environment_manifests/hypersca-task-c-causalbench.json": {
            "schema_version": "1.0",
            "environment": "hypersca-task-c-causalbench",
            "packages": packages,
        },
    }
    snapshots: dict[str, object] = {}
    for relative, payload in records.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        snapshots[relative] = method_run_module._capture_file(
            path,
            f"fake asset {relative}",
            maximum_bytes=method_run_module.MAXIMUM_RECORD_BYTES,
            require_single_link=True,
        )
    return snapshots


def _patch_external_assets_and_run(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: dict[str, object],
) -> None:
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._asset_snapshots",
        lambda *args, **kwargs: snapshots,
    )
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._external_source_digest",
        lambda *args, **kwargs: None,
    )

    def fake_run(
        command: Sequence[str], *, output_dir: Path, timeout_seconds: object
    ) -> dict[str, object]:
        del command, timeout_seconds
        output_dir.mkdir()
        (output_dir / "worker_predictions.csv").write_text(
            "source,target,score\nA,B,1\n", encoding="utf-8"
        )
        inner = {"schema_version": "1.0", "status": "completed_raw_inference"}
        (output_dir / "method_status.json").write_text(
            json.dumps(inner) + "\n", encoding="utf-8"
        )
        (output_dir / "resource_usage.json").write_text(
            '{"schema_version":"1.0"}\n', encoding="utf-8"
        )
        return inner

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.run_isolated_method", fake_run
    )


def _synthetic_external_arguments(tmp_path: Path) -> dict[str, object]:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    return {
        "method_id": "pc",
        "input_npz": input_path,
        "output_dir": tmp_path / "run",
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "unused-assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }


def test_external_run_rejects_live_environment_mismatch_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _live_environment_record("3.10")
    snapshots = _fake_external_asset_snapshots(
        tmp_path, packages=expected["packages"]
    )
    _patch_external_assets_and_run(monkeypatch, snapshots)
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._capture_live_conda_environment",
        lambda name: _live_environment_record("3.11"),
    )

    with pytest.raises(TaskCMethodRunError, match="differs from the prepared"):
        run_task_c_method(**_synthetic_external_arguments(tmp_path))
    assert not (tmp_path / "run").exists()


def test_external_reuse_rejects_a_changed_live_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _live_environment_record("3.10")
    snapshots = _fake_external_asset_snapshots(
        tmp_path, packages=expected["packages"]
    )
    _patch_external_assets_and_run(monkeypatch, snapshots)
    current = {"record": expected}
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._capture_live_conda_environment",
        lambda name: current["record"],
    )
    arguments = _synthetic_external_arguments(tmp_path)
    run_task_c_method(**arguments)
    current["record"] = _live_environment_record("3.11")

    with pytest.raises(TaskCMethodRunError, match="differs from the prepared"):
        run_task_c_method(**arguments)


def test_external_run_rejects_live_environment_change_during_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _live_environment_record("3.10")
    snapshots = _fake_external_asset_snapshots(
        tmp_path, packages=expected["packages"]
    )
    _patch_external_assets_and_run(monkeypatch, snapshots)
    records = iter((expected, _live_environment_record("3.11")))
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._capture_live_conda_environment",
        lambda name: next(records),
    )

    with pytest.raises(TaskCMethodRunError, match="changed during the method run"):
        run_task_c_method(**_synthetic_external_arguments(tmp_path))
    status = json.loads(
        (tmp_path / "run/method_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed_runtime_unavailable"
    assert not (tmp_path / "run/predictions.csv").exists()


def test_timeout_is_not_mislabeled_and_terminates_the_process_group(
    tmp_path: Path,
) -> None:
    child_pid = tmp_path / "child.pid"
    program = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    output = tmp_path / "run"

    result = run_isolated_method(
        [sys.executable, "-c", program],
        output_dir=output,
        timeout_seconds=0.5,
    )

    assert result["status"] == "failed_timeout"
    assert (output / "method_status.json").exists()
    assert child_pid.exists()
    spawned_pid = int(child_pid.read_text(encoding="utf-8"))
    for _ in range(40):
        process_status = Path(f"/proc/{spawned_pid}/status")
        if not process_status.exists() or "State:\tZ" in process_status.read_text(
            encoding="utf-8"
        ):
            break
        time.sleep(0.025)
    else:
        pytest.fail("the timed-out method left a running child process")


def test_completed_leader_cannot_leave_a_background_child(tmp_path: Path) -> None:
    child_pid = tmp_path / "background.pid"
    child_ready = tmp_path / "background.ready"
    child_program = (
        "import pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(child_ready)!r}).write_text('ready'); "
        "time.sleep(30)"
    )
    program = (
        "import pathlib, subprocess, sys, time\n"
        f"child=subprocess.Popen([sys.executable, '-c', {child_program!r}])\n"
        f"ready=pathlib.Path({str(child_ready)!r})\n"
        "deadline=time.monotonic()+5\n"
        "while not ready.exists() and time.monotonic()<deadline:\n"
        "    time.sleep(0.01)\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
    )

    result = run_isolated_method(
        [sys.executable, "-c", program],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert result["status"] == "completed_raw_inference"
    spawned_pid = int(child_pid.read_text(encoding="utf-8"))
    for _ in range(40):
        process_status = Path(f"/proc/{spawned_pid}/status")
        if not process_status.exists() or "State:\tZ" in process_status.read_text(
            encoding="utf-8"
        ):
            break
        time.sleep(0.025)
    else:
        os.kill(spawned_pid, signal.SIGKILL)
        pytest.fail("a successful method leader left a running background process")


def test_resource_parser_does_not_silently_misread_malformed_rss(
    tmp_path: Path,
) -> None:
    report = tmp_path / "resource.txt"
    report.write_text(
        "Maximum resident set size (kbytes): 12,345\n",
        encoding="utf-8",
    )

    assert _parse_maximum_resident_kib(report) is None

    report.write_text(
        f"Maximum resident set size (kbytes): {2**63}\n",
        encoding="utf-8",
    )
    assert _parse_maximum_resident_kib(report) is None


def test_nonzero_exit_records_bounded_stderr_without_private_command_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    private_argument = str(tmp_path / "private-patient-directory" / "input.npz")

    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.buffer.write(b'x' * 20000 + b'bad input'); sys.exit(3)",
            private_argument,
        ],
        output_dir=output,
        timeout_seconds=10,
        maximum_output_bytes=512,
    )

    assert result["status"] == "official_code_incompatible"
    assert "bad input" in result["stderr_tail"]
    assert result["output_was_truncated"] == {"stdout": False, "stderr": True}
    serialized = (output / "method_status.json").read_text(encoding="utf-8")
    assert private_argument not in serialized
    assert set(result["command_trace"]) == {
        "argument_count",
        "command_sha256",
        "executable_name",
    }


def test_invalid_output_marker_is_preserved_and_invalid_utf8_is_decoded(
    tmp_path: Path,
) -> None:
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "os.write(2, b'\\xfffailed_invalid_output: unknown gene'); "
                "sys.exit(2)"
            ),
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert result["status"] == "failed_invalid_output"
    assert "failed_invalid_output" in result["stderr_tail"]
    assert "\ufffd" in result["stderr_tail"]


@pytest.mark.parametrize(
    ("marker", "expected_status"),
    [
        ("failed_invalid_output: unknown gene", "failed_invalid_output"),
        ("MemoryError: allocation failed", "failed_resource_limit"),
    ],
)
def test_failure_marker_before_a_long_log_still_controls_classification(
    tmp_path: Path, marker: str, expected_status: str
) -> None:
    program = (
        "import sys; "
        f"sys.stderr.write({marker!r} + '\\n'); "
        "sys.stderr.write('later detail\\n' * 5000); "
        "sys.exit(3)"
    )

    result = run_isolated_method(
        [sys.executable, "-c", program],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
        maximum_output_bytes=256,
    )

    assert result["status"] == expected_status
    assert marker not in result["stderr_tail"]


def test_command_values_with_spaces_are_removed_from_saved_output(
    tmp_path: Path,
) -> None:
    private_value = str(tmp_path / "patient cohort A" / "input matrix.npz")
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1], file=sys.stderr); sys.exit(3)",
            private_value,
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert private_value not in result["stderr_tail"]
    assert "patient cohort A" not in result["stderr_tail"]
    assert "<command-argument>" in result["stderr_tail"]


def test_option_equals_value_is_redacted_without_hiding_option_name(
    tmp_path: Path,
) -> None:
    private_value = str(tmp_path / "patient-007" / "input.npz")
    argument = f"--input-npz={private_value}"
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1], file=sys.stderr); sys.exit(3)",
            argument,
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert "--input-npz=<command-argument>" in result["stderr_tail"]
    assert "patient-007" not in result["stderr_tail"]
    assert private_value not in result["stderr_tail"]


def test_option_equals_value_is_redacted_when_only_its_value_is_logged(
    tmp_path: Path,
) -> None:
    private_value = str(tmp_path / "patient cohort B" / "input matrix.npz")
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1].split('=', 1)[1], file=sys.stderr); sys.exit(3)",
            f"--input-npz={private_value}",
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert private_value not in result["stderr_tail"]
    assert "patient cohort B" not in result["stderr_tail"]
    assert "<command-argument>" in result["stderr_tail"]


@pytest.mark.parametrize(
    "private_argument",
    [
        "-i={root}/Patient cohort C/input matrix.npz",
        "-I{root}/Patient cohort D/include files",
    ],
)
def test_short_attached_option_paths_are_fully_redacted(
    tmp_path: Path,
    private_argument: str,
) -> None:
    argument = private_argument.format(root=tmp_path)
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1], file=sys.stderr); sys.exit(3)",
            argument,
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert argument not in result["stderr_tail"]
    assert "Patient" not in result["stderr_tail"]
    assert "cohort" not in result["stderr_tail"]
    assert "<command-argument>" in result["stderr_tail"]


def test_short_equals_option_value_is_redacted_when_logged_without_option(
    tmp_path: Path,
) -> None:
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1].split('=', 1)[1], file=sys.stderr); sys.exit(3)",
            "-i=P7",
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert "P7" not in result["stderr_tail"]
    assert "<command-argument>" in result["stderr_tail"]


def test_short_nonsensitive_arguments_do_not_change_method_logs(
    tmp_path: Path,
) -> None:
    expected = "A method 3 stayed visible"
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            f"import sys; print({expected!r}, file=sys.stderr); sys.exit(3)",
            "3",
            "A",
        ],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert result["stderr_tail"] == expected + "\n"
    assert "\x00" not in result["stderr_tail"]


def test_overlapping_sensitive_paths_are_redacted_once_without_nul_markers(
    tmp_path: Path,
) -> None:
    private_root = str(tmp_path / "Patient cohort A")
    private_input = str(Path(private_root) / "nested input matrix.npz")
    output = tmp_path / "run"
    result = run_isolated_method(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1], sys.argv[2], sep='\\n', file=sys.stderr); sys.exit(3)",
            private_input,
            private_root,
            "A",
        ],
        output_dir=output,
        timeout_seconds=10,
    )

    assert private_input not in result["stderr_tail"]
    assert private_root not in result["stderr_tail"]
    assert "Patient" not in result["stderr_tail"]
    assert result["stderr_tail"].count("<command-argument>") == 2
    assert "\x00" not in result["stderr_tail"]
    serialized = (output / "method_status.json").read_text(encoding="utf-8")
    assert "\\u0000" not in serialized
    assert json.loads(serialized) == result


def test_explicit_exit_137_is_not_reported_as_a_terminating_signal(
    tmp_path: Path,
) -> None:
    result = run_isolated_method(
        [sys.executable, "-c", "raise SystemExit(137)"],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert result["return_code"] == 137
    assert result["terminating_signal"] is None


@pytest.mark.parametrize(
    ("return_code", "expected_status"),
    [
        (125, "failed_runtime_unavailable"),
        (126, "failed_launch"),
        (127, "failed_launch"),
    ],
)
def test_runtime_wrapper_exit_codes_are_not_called_method_incompatibility(
    tmp_path: Path, return_code: int, expected_status: str
) -> None:
    result = run_isolated_method(
        [sys.executable, "-c", f"raise SystemExit({return_code})"],
        output_dir=tmp_path / "run",
        timeout_seconds=10,
    )

    assert result["status"] == expected_status
    assert result["terminating_signal"] is None


def test_missing_resource_meter_produces_paired_runtime_unavailable_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "GNU_TIME", tmp_path / "missing-time")
    output = tmp_path / "run"

    result = run_isolated_method(
        [sys.executable, "-c", "print('must not run')"],
        output_dir=output,
        timeout_seconds=10,
    )

    assert result["status"] == "failed_runtime_unavailable"
    resource = json.loads((output / "resource_usage.json").read_text(encoding="utf-8"))
    assert resource["resource_meter"] == "unavailable"
    assert {path.name for path in output.iterdir()} == {
        "method_status.json",
        "resource_usage.json",
    }


def test_unreasonably_long_rss_number_becomes_paired_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HugeRssProcess:
        pid = 99999997
        returncode = 0
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def __init__(self, command: list[str]) -> None:
            report = Path(command[command.index("-o") + 1])
            report.write_text(
                "Maximum resident set size (kbytes): " + "9" * 10000 + "\n",
                encoding="utf-8",
            )

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(
        runtime_module.subprocess,
        "Popen",
        lambda command, **kwargs: HugeRssProcess(command),
    )
    monkeypatch.setattr(
        runtime_module, "_terminate_process_group", lambda process: None
    )
    output = tmp_path / "run"

    result = run_isolated_method(
        [sys.executable, "-c", "pass"],
        output_dir=output,
        timeout_seconds=10,
    )

    assert result["status"] == "failed_runtime_unavailable"
    resource = json.loads((output / "resource_usage.json").read_text(encoding="utf-8"))
    assert resource["resource_meter"] == "unavailable"
    assert resource["maximum_resident_kib"] is None
    assert {path.name for path in output.iterdir()} == {
        "method_status.json",
        "resource_usage.json",
    }


def test_launch_oserror_produces_paired_failed_launch_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_launch(*args: object, **kwargs: object) -> object:
        raise OSError("test-only launch error")

    monkeypatch.setattr(runtime_module.subprocess, "Popen", fail_launch)
    output = tmp_path / "run"

    result = run_isolated_method(
        [sys.executable, "-c", "pass"],
        output_dir=output,
        timeout_seconds=10,
    )

    assert result["status"] == "failed_launch"
    assert "could not be started" in result["stderr_tail"]
    assert {path.name for path in output.iterdir()} == {
        "method_status.json",
        "resource_usage.json",
    }


def test_output_directory_replacement_cannot_redirect_records(tmp_path: Path) -> None:
    output = tmp_path / "run"
    moved = tmp_path / "original-run-inode"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    marker = tmp_path / "method-started"
    program = (
        "import pathlib, time; "
        f"pathlib.Path({str(marker)!r}).write_text('started'); "
        "time.sleep(0.4)"
    )
    result_box: list[dict[str, object]] = []

    thread = threading.Thread(
        target=lambda: result_box.append(
            run_isolated_method(
                [sys.executable, "-c", program],
                output_dir=output,
                timeout_seconds=10,
            )
        )
    )
    thread.start()
    for _ in range(200):
        if marker.exists() and output.exists():
            break
        time.sleep(0.005)
    else:
        pytest.fail("method did not start in time")
    output.rename(moved)
    output.symlink_to(attacker, target_is_directory=True)
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert result_box[0]["status"] == "failed_runtime_unavailable"
    assert not list(attacker.iterdir())
    assert {path.name for path in moved.iterdir()} == {
        "method_status.json",
        "resource_usage.json",
    }
    assert not list(moved.glob("*.tmp"))


def test_keyboard_interrupt_still_cleans_runtime_temporary_files_and_records_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedProcess:
        pid = 99999999
        returncode = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        runtime_module.subprocess,
        "Popen",
        lambda *args, **kwargs: InterruptedProcess(),
    )
    monkeypatch.setattr(
        runtime_module, "_terminate_process_group", lambda process: None
    )
    output = tmp_path / "run"

    with pytest.raises(KeyboardInterrupt):
        run_isolated_method(
            [sys.executable, "-c", "pass"],
            output_dir=output,
            timeout_seconds=10,
        )

    assert {path.name for path in output.iterdir()} == {
        "method_status.json",
        "resource_usage.json",
    }
    status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_runtime_unavailable"


def test_bootstrap_bounded_process_is_terminated_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedProcess:
        pid = 99999998
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    terminated: list[object] = []
    monkeypatch.setattr(
        runtime_module.subprocess,
        "Popen",
        lambda *args, **kwargs: InterruptedProcess(),
    )
    monkeypatch.setattr(
        runtime_module,
        "_terminate_process_group",
        lambda process: terminated.append(process),
    )

    with pytest.raises(KeyboardInterrupt):
        _run_bounded_command(["git", "--version"], timeout=10)

    assert len(terminated) == 1


def test_success_writes_exact_atomic_status_and_resource_records(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"

    result = run_isolated_method(
        [sys.executable, "-c", "print('completed')"],
        output_dir=output,
        timeout_seconds=10,
    )

    assert result["status"] == "completed_raw_inference"
    assert set(result) == {
        "schema_version",
        "status",
        "return_code",
        "terminating_signal",
        "elapsed_seconds",
        "stdout_tail",
        "stderr_tail",
        "output_was_truncated",
        "command_trace",
    }
    assert math.isfinite(result["elapsed_seconds"])
    assert result["elapsed_seconds"] >= 0
    resource = json.loads((output / "resource_usage.json").read_text(encoding="utf-8"))
    assert set(resource) == {
        "schema_version",
        "elapsed_seconds",
        "maximum_resident_kib",
        "resource_meter",
    }
    assert resource["resource_meter"] == "gnu_time_v"
    assert resource["maximum_resident_kib"] is None or (
        type(resource["maximum_resident_kib"]) is int
        and resource["maximum_resident_kib"] >= 0
    )
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize(
    ("command", "timeout", "message"),
    [
        ([], 1, "command"),
        ("python -V", 1, "command"),
        (["python", ""], 1, "command"),
        (["python", "bad\x00argument"], 1, "command"),
        (["python", 3], 1, "command"),
        (["python"], True, "timeout"),
        (["python"], 0, "timeout"),
        (["python"], float("inf"), "timeout"),
    ],
)
def test_runtime_rejects_ambiguous_commands_and_timeouts(
    tmp_path: Path, command: object, timeout: object, message: str
) -> None:
    with pytest.raises(TaskCRuntimeError, match=message):
        run_isolated_method(  # type: ignore[arg-type]
            command,
            output_dir=tmp_path / "run",
            timeout_seconds=timeout,
        )


def test_runtime_keeps_a_hard_upper_bound_on_captured_output(tmp_path: Path) -> None:
    with pytest.raises(TaskCRuntimeError, match="maximum output bytes"):
        run_isolated_method(
            [sys.executable, "-c", "pass"],
            output_dir=tmp_path / "run",
            timeout_seconds=10,
            maximum_output_bytes=2**30,
        )


def test_runtime_rejects_unsafe_or_nonempty_output_locations(tmp_path: Path) -> None:
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "earlier-result.json").write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(populated, target_is_directory=True)
    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")

    for destination in (populated, linked, regular_file):
        with pytest.raises(TaskCRuntimeError, match="output"):
            run_isolated_method(
                [sys.executable, "-c", "pass"],
                output_dir=destination,
                timeout_seconds=10,
            )


def test_missing_executable_is_a_recorded_compatibility_failure(tmp_path: Path) -> None:
    output = tmp_path / "run"

    result = run_isolated_method(
        ["certainly-not-a-real-task-c-executable-2841"],
        output_dir=output,
        timeout_seconds=10,
    )

    assert result["status"] == "failed_launch"
    assert result["return_code"] is None
    assert "could not be started" in result["stderr_tail"]


class _FakeCompleted:
    def __init__(self, *, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class _FakeBootstrapRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.repositories: dict[Path, dict[str, str]] = {}
        self.environments: set[str] = set()
        self.replacements: set[Path] = set()
        self.environment_files_seen: list[Path] = []
        self.environment_bytes_seen: list[bytes] = []
        self.fail_environment_update = False

    def __call__(self, command: list[str], **kwargs: object) -> _FakeCompleted:
        assert command and all(type(part) is str for part in command)
        assert "shell" not in kwargs
        assert kwargs.get("check") is True
        assert isinstance(kwargs.get("timeout"), (int, float))
        assert kwargs["timeout"] > 0  # type: ignore[operator]
        self.calls.append(tuple(command))
        if command[:2] == ["git", "clone"]:
            repository, destination_text = command[2], command[3]
            destination = Path(destination_text)
            destination.mkdir(parents=True)
            (destination / ".git").mkdir()
            (destination / "README.md").write_text(
                "fixed official source\n", encoding="utf-8"
            )
            self.repositories[destination] = {"repository": repository, "commit": ""}
            return _FakeCompleted()
        if command[0:2] == ["git", "-C"]:
            source = Path(command[2])
            operation = command[3:]
            if operation[:2] == ["checkout", "--detach"]:
                self.repositories[source]["commit"] = operation[2]
                return _FakeCompleted()
            if operation == ["remote", "get-url", "origin"]:
                return _FakeCompleted(
                    stdout=self.repositories[source]["repository"] + "\n"
                )
            if operation == ["rev-parse", "HEAD"]:
                return _FakeCompleted(stdout=self.repositories[source]["commit"] + "\n")
            if operation and operation[0] == "status":
                return _FakeCompleted(stdout="")
            if operation == ["replace", "-l"]:
                return _FakeCompleted(
                    stdout="replacement\n" if source in self.replacements else ""
                )
            if operation == ["rev-parse", "--git-path", "info/grafts"]:
                return _FakeCompleted(stdout=str(source / ".git/info/grafts") + "\n")
            if operation[:3] == ["ls-files", "-z", "--cached"]:
                return _FakeCompleted(stdout="README.md\x00")
            if operation[:4] == ["ls-tree", "-r", "-z", "--name-only"]:
                return _FakeCompleted(stdout="README.md\x00")
            if operation[:3] == ["diff", "--no-ext-diff", "--quiet"]:
                return _FakeCompleted(stdout="")
        if command == ["conda", "env", "list", "--json"]:
            return _FakeCompleted(
                stdout=json.dumps(
                    {
                        "envs": [f"/fake/envs/{name}" for name in self.environments],
                        "active_prefix": "/private/local/conda",
                        "conda_version": "25.5.1",
                        "envs_dirs": ["/private/local/conda/envs"],
                    }
                )
            )
        if command[:3] == ["conda", "env", "create"]:
            environment_file = Path(command[command.index("--file") + 1])
            self.environment_files_seen.append(environment_file)
            self.environment_bytes_seen.append(environment_file.read_bytes())
            name = next(
                line.split(":", 1)[1].strip()
                for line in environment_file.read_text(encoding="utf-8").splitlines()
                if line.startswith("name:")
            )
            self.environments.add(name)
            return _FakeCompleted()
        if command[:3] == ["conda", "env", "update"]:
            if self.fail_environment_update:
                raise subprocess.CalledProcessError(1, command)
            environment_file = Path(command[command.index("--file") + 1])
            self.environment_files_seen.append(environment_file)
            self.environment_bytes_seen.append(environment_file.read_bytes())
            self.environments.add(command[4])
            return _FakeCompleted()
        if command[:3] == ["conda", "run", "-n"]:
            return _FakeCompleted(
                stdout=json.dumps(
                    [{"name": "python", "version": "3.10.12", "build_string": "0"}]
                )
            )
        raise AssertionError(f"unexpected bootstrap command: {command}")


class _FailingCloneRunner(_FakeBootstrapRunner):
    def __call__(self, command: list[str], **kwargs: object) -> _FakeCompleted:
        if command[:2] == ["git", "clone"]:
            destination = Path(command[3])
            destination.mkdir(parents=True)
            (destination / "partial-download").write_text(
                "incomplete", encoding="utf-8"
            )
            raise subprocess.CalledProcessError(1, command)
        return super().__call__(command, **kwargs)


def _copy_bootstrap_inputs(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    environment_dir = project / "envs/task_c"
    environment_dir.mkdir(parents=True)
    for name in ("causalbench.yml", "psgrn.yml"):
        (environment_dir / name).write_bytes((ROOT / "envs/task_c" / name).read_bytes())
    registry = tmp_path / "task_c_methods_v1.json"
    registry.write_bytes(REGISTRY.read_bytes())
    return project, registry


def test_bootstrap_uses_fixed_sources_environments_and_explicit_unavailability(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "method-assets"
    runner = _FakeBootstrapRunner()

    summary = bootstrap_task_c_methods(
        cache_root=cache,
        registry_path=REGISTRY,
        project_root=ROOT,
        run_command=runner,
    )

    assert summary["status"] == "assets_and_environments_recorded"
    assert (cache / "sources/causalbench/.git").is_dir()
    assert (cache / "sources/guanlab_psgrn/.git").is_dir()
    for method_id in ("betterboost", "sparse_rc", "catran"):
        status = json.loads(
            (cache / f"status/{method_id}/method_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "official_assets_unavailable"
    manifests = sorted((cache / "environment_manifests").glob("*.json"))
    assert [path.stem for path in manifests] == [
        "hypersca-task-c-causalbench",
        "hypersca-task-c-psgrn",
    ]
    identity = json.loads(
        (cache / "bootstrap_identity.json").read_text(encoding="utf-8")
    )
    assert set(identity) == {
        "schema_version",
        "registry_sha256",
        "sources",
        "environment_files",
    }
    assert all(str(ROOT) not in path.read_text(encoding="utf-8") for path in manifests)
    completion = json.loads(
        (cache / "bootstrap_status.json").read_text(encoding="utf-8")
    )
    overall = cache / "bootstrap_manifest.json"
    assert completion["status"] == "assets_and_environments_recorded"
    assert (
        completion["bootstrap_manifest_sha256"]
        == hashlib.sha256(overall.read_bytes()).hexdigest()
    )
    overall_payload = json.loads(overall.read_text(encoding="utf-8"))
    assert set(overall_payload["environment_manifests"]) == {
        path.name for path in manifests
    }
    assert set(overall_payload["publication_statuses"]) == {
        "betterboost/method_status.json",
        "sparse_rc/method_status.json",
        "catran/method_status.json",
    }
    assert all(path.is_relative_to(cache) for path in runner.environment_files_seen)
    assert not list(cache.rglob("*.tmp"))
    assert not list(cache.glob(".bootstrap-staging-*"))


def test_exact_bootstrap_rerun_reuses_sources_and_updates_environments(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "method-assets"
    runner = _FakeBootstrapRunner()
    arguments = {
        "cache_root": cache,
        "registry_path": REGISTRY,
        "project_root": ROOT,
        "run_command": runner,
    }

    bootstrap_task_c_methods(**arguments)
    clone_count = sum(call[:2] == ("git", "clone") for call in runner.calls)
    bootstrap_task_c_methods(**arguments)

    assert sum(call[:2] == ("git", "clone") for call in runner.calls) == clone_count
    assert any(call[:3] == ("conda", "env", "update") for call in runner.calls)


def test_bootstrap_rejects_wrong_source_version_and_changed_identity(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "method-assets"
    runner = _FakeBootstrapRunner()
    arguments = {
        "cache_root": cache,
        "registry_path": REGISTRY,
        "project_root": ROOT,
        "run_command": runner,
    }
    bootstrap_task_c_methods(**arguments)
    source = cache / "sources/causalbench"
    runner.repositories[source]["commit"] = "0" * 40

    with pytest.raises(TaskCRuntimeError, match="fixed commit"):
        bootstrap_task_c_methods(**arguments)

    runner.repositories[source]["commit"] = load_task_c_method_registry(
        REGISTRY
    ).causalbench["commit"]
    identity_path = cache / "bootstrap_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["registry_sha256"] = "f" * 64
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(TaskCRuntimeError, match="identity"):
        bootstrap_task_c_methods(**arguments)


def test_bootstrap_rejects_semantically_equal_but_rewritten_identity(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "method-assets"
    runner = _FakeBootstrapRunner()
    arguments = {
        "cache_root": cache,
        "registry_path": REGISTRY,
        "project_root": ROOT,
        "run_command": runner,
    }
    bootstrap_task_c_methods(**arguments)
    identity_path = cache / "bootstrap_identity.json"
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    identity_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TaskCRuntimeError, match="exact|bytes|identity"):
        bootstrap_task_c_methods(**arguments)


def test_bootstrap_json_records_reject_duplicate_keys_and_excessive_size(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "method-assets"
    runner = _FakeBootstrapRunner()
    arguments = {
        "cache_root": cache,
        "registry_path": REGISTRY,
        "project_root": ROOT,
        "run_command": runner,
    }
    bootstrap_task_c_methods(**arguments)
    identity_path = cache / "bootstrap_identity.json"
    original = identity_path.read_text(encoding="utf-8")
    identity_path.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(TaskCRuntimeError, match="duplicate"):
        bootstrap_task_c_methods(**arguments)

    identity_path.write_text(original + " " * (2 * 1024 * 1024), encoding="utf-8")
    with pytest.raises(TaskCRuntimeError, match="large"):
        bootstrap_task_c_methods(**arguments)


def test_bootstrap_yaml_requires_a_structural_exact_vcs_pin(tmp_path: Path) -> None:
    project, registry = _copy_bootstrap_inputs(tmp_path)
    expected = load_task_c_method_registry(registry).causalbench
    (project / "envs/task_c/causalbench.yml").write_text(
        "\n".join(
            [
                f"name: {expected['environment']}",
                "channels: [conda-forge]",
                "dependencies:",
                "  - python=3.10",
                "  - pip:",
                "      - example-package==1.0",
                f"# git+{expected['repository']}@{expected['commit']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskCRuntimeError, match="YAML|fixed CausalBench source"):
        bootstrap_task_c_methods(
            cache_root=tmp_path / "cache",
            registry_path=registry,
            project_root=project,
            run_command=_FakeBootstrapRunner(),
        )


def test_bootstrap_yaml_rejects_duplicate_sections_and_extra_vcs_source(
    tmp_path: Path,
) -> None:
    project, registry = _copy_bootstrap_inputs(tmp_path)
    expected = load_task_c_method_registry(registry).causalbench
    environment = project / "envs/task_c/causalbench.yml"
    environment.write_text(
        "\n".join(
            [
                f"name: {expected['environment']}",
                "channels:",
                "  - conda-forge",
                "channels:",
                "  - defaults",
                "dependencies:",
                "  - python=3.10",
                "  - pip:",
                f"      - git+{expected['repository']}@{expected['commit']}",
                "      - git+https://example.invalid/other.git@" + "0" * 40,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskCRuntimeError, match="YAML|VCS|fixed CausalBench source"):
        bootstrap_task_c_methods(
            cache_root=tmp_path / "cache",
            registry_path=registry,
            project_root=project,
            run_command=_FakeBootstrapRunner(),
        )


@pytest.mark.parametrize(
    "environment_text",
    [
        "\n".join(
            [
                "name: hypersca-task-c-causalbench",
                "channels:",
                "  - conda-forge",
                "dependencies:",
                "  - python=3.10",
                "  - pip: {git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71: null}",
            ]
        ),
        "\n".join(
            [
                "name: hypersca-task-c-causalbench",
                "channels:",
                "  - conda-forge",
                "dependencies:",
                "  - python=3.10",
                "  - pip:",
                "      - git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71",
                "  - pip:",
                "      - numpy==1.26.4",
            ]
        ),
        "\n".join(
            [
                "name: hypersca-task-c-causalbench",
                "name : hypersca-task-c-causalbench",
                "channels:",
                "  - conda-forge",
                "dependencies:",
                "  - python=3.10",
                "  - pip:",
                "      - git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71",
            ]
        ),
        "\n".join(
            [
                "name: hypersca-task-c-causalbench",
                "channels:",
                "  - conda-forge",
                "dependencies:",
                "  - python=3.10",
                "  - pip:",
                "      - git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71",
                "      - extra @ git+https://example.invalid/other.git@" + "0" * 40,
            ]
        ),
        "\n".join(
            [
                "name: hypersca-task-c-causalbench",
                "channels:",
                "  - conda-forge",
                "dependencies:",
                "  - python=3.10",
                "  - pip:",
                "      - !!python/object/apply:os.system [echo unsafe]",
                "      - git+https://github.com/causalbench/causalbench.git@1a2143cffdc85f835b41ce8d52034be1bf903e71",
            ]
        ),
    ],
)
def test_bootstrap_safe_yaml_rejects_mapping_pip_duplicate_keys_and_tags(
    tmp_path: Path,
    environment_text: str,
) -> None:
    project, registry = _copy_bootstrap_inputs(tmp_path)
    (project / "envs/task_c/causalbench.yml").write_text(
        environment_text + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskCRuntimeError, match="YAML|pip|environment"):
        bootstrap_task_c_methods(
            cache_root=tmp_path / "cache",
            registry_path=registry,
            project_root=project,
            run_command=_FakeBootstrapRunner(),
        )


def test_bootstrap_deep_yaml_is_reported_as_a_runtime_input_error(
    tmp_path: Path,
) -> None:
    project, registry = _copy_bootstrap_inputs(tmp_path)
    nested = "[" * 2000 + "value" + "]" * 2000
    (project / "envs/task_c/causalbench.yml").write_text(
        f"name: {nested}\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"

    with pytest.raises(TaskCRuntimeError, match="YAML|environment"):
        bootstrap_task_c_methods(
            cache_root=cache,
            registry_path=registry,
            project_root=project,
            run_command=_FakeBootstrapRunner(),
        )

    assert not list(cache.glob(".bootstrap-staging-*"))


def test_bad_registry_never_leaves_staging_and_is_repeatably_rejected(
    tmp_path: Path,
) -> None:
    bad_registry = tmp_path / "bad-registry.json"
    bad_registry.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    cache = tmp_path / "cache"

    for _ in range(2):
        with pytest.raises(
            (TaskCMethodRegistryError, TaskCRuntimeError),
            match="registry|schema_version|fixed method",
        ):
            bootstrap_task_c_methods(
                cache_root=cache,
                registry_path=bad_registry,
                project_root=ROOT,
                run_command=_FakeBootstrapRunner(),
            )
        assert not list(cache.glob(".bootstrap-staging-*")) if cache.exists() else True


def test_leftover_bootstrap_staging_fails_closed_without_creating_another(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    leftover = cache / ".bootstrap-staging-fixed"
    leftover.mkdir()
    (leftover / "unexpected").write_text("partial", encoding="utf-8")

    with pytest.raises(TaskCRuntimeError, match="staging|unrecognized"):
        bootstrap_task_c_methods(
            cache_root=cache,
            registry_path=REGISTRY,
            project_root=ROOT,
            run_command=_FakeBootstrapRunner(),
        )

    assert {path.name for path in cache.iterdir()} == {leftover.name}


class _MutatingInputRunner(_FakeBootstrapRunner):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.mutated = False

    def __call__(self, command: list[str], **kwargs: object) -> _FakeCompleted:
        result = super().__call__(command, **kwargs)
        if command == ["conda", "env", "list", "--json"] and not self.mutated:
            self.path.write_bytes(
                self.path.read_bytes() + b"\n# changed during preparation\n"
            )
            self.mutated = True
        return result


def test_bootstrap_uses_snapshots_and_rejects_inputs_changed_during_run(
    tmp_path: Path,
) -> None:
    project, registry = _copy_bootstrap_inputs(tmp_path)
    changed_environment = project / "envs/task_c/causalbench.yml"
    original_bytes = changed_environment.read_bytes()
    runner = _MutatingInputRunner(changed_environment)

    with pytest.raises(TaskCRuntimeError, match="changed during preparation"):
        bootstrap_task_c_methods(
            cache_root=tmp_path / "cache",
            registry_path=registry,
            project_root=project,
            run_command=runner,
        )

    assert runner.environment_files_seen
    assert all(
        path.is_relative_to(tmp_path / "cache")
        for path in runner.environment_files_seen
    )
    assert original_bytes in runner.environment_bytes_seen
    status = json.loads(
        (tmp_path / "cache/bootstrap_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed_asset_preparation"
    assert not (tmp_path / "cache/bootstrap_manifest.json").exists()


def test_failed_refresh_invalidates_an_older_completion_record(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    runner = _FakeBootstrapRunner()
    arguments = {
        "cache_root": cache,
        "registry_path": REGISTRY,
        "project_root": ROOT,
        "run_command": runner,
    }
    bootstrap_task_c_methods(**arguments)
    assert (cache / "bootstrap_manifest.json").exists()
    runner.fail_environment_update = True

    with pytest.raises(TaskCRuntimeError, match="official asset command failed"):
        bootstrap_task_c_methods(**arguments)

    status = json.loads((cache / "bootstrap_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_asset_preparation"
    assert not (cache / "bootstrap_manifest.json").exists()


def test_bootstrap_rejects_untracked_symlink_replacement_and_grafts(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    runner = _FakeBootstrapRunner()
    arguments = {
        "cache_root": cache,
        "registry_path": REGISTRY,
        "project_root": ROOT,
        "run_command": runner,
    }
    bootstrap_task_c_methods(**arguments)
    source = cache / "sources/causalbench"

    (source / "ignored-secret.tmp").write_text("unexpected", encoding="utf-8")
    with pytest.raises(TaskCRuntimeError, match="untracked|ignored|official source"):
        bootstrap_task_c_methods(**arguments)
    (source / "ignored-secret.tmp").unlink()

    (source / "outside-link").symlink_to(tmp_path / "outside")
    with pytest.raises(TaskCRuntimeError, match="symbolic link|official source"):
        bootstrap_task_c_methods(**arguments)
    (source / "outside-link").unlink()

    runner.replacements.add(source)
    with pytest.raises(TaskCRuntimeError, match="replacement"):
        bootstrap_task_c_methods(**arguments)
    runner.replacements.clear()

    graft = source / ".git/info/grafts"
    graft.parent.mkdir(parents=True)
    graft.write_text("0" * 40 + " " + "1" * 40 + "\n", encoding="utf-8")
    with pytest.raises(TaskCRuntimeError, match="graft"):
        bootstrap_task_c_methods(**arguments)


def test_bootstrap_rejects_symlink_cache_and_inconsistent_environment_name(
    tmp_path: Path,
) -> None:
    real_cache = tmp_path / "real"
    real_cache.mkdir()
    linked_cache = tmp_path / "linked"
    linked_cache.symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(TaskCRuntimeError, match="cache"):
        bootstrap_task_c_methods(
            cache_root=linked_cache,
            registry_path=REGISTRY,
            project_root=ROOT,
            run_command=_FakeBootstrapRunner(),
        )


def test_failed_clone_does_not_leave_a_half_downloaded_source(tmp_path: Path) -> None:
    cache = tmp_path / "method-assets"

    with pytest.raises(TaskCRuntimeError, match="official asset command failed"):
        bootstrap_task_c_methods(
            cache_root=cache,
            registry_path=REGISTRY,
            project_root=ROOT,
            run_command=_FailingCloneRunner(),
        )

    assert not (cache / "sources/causalbench").exists()


def test_bootstrap_rejects_symlinked_managed_subdirectory(tmp_path: Path) -> None:
    cache = tmp_path / "method-assets"
    cache.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (cache / "sources").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(TaskCRuntimeError, match="sources"):
        bootstrap_task_c_methods(
            cache_root=cache,
            registry_path=REGISTRY,
            project_root=ROOT,
            run_command=_FakeBootstrapRunner(),
        )

    assert not list(elsewhere.iterdir())

    project = tmp_path / "project"
    environment_dir = project / "envs/task_c"
    environment_dir.mkdir(parents=True)
    (environment_dir / "causalbench.yml").write_text(
        "name: wrong-environment\n", encoding="utf-8"
    )
    (environment_dir / "psgrn.yml").write_text(
        (ROOT / "envs/task_c/psgrn.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(TaskCRuntimeError, match="environment name"):
        bootstrap_task_c_methods(
            cache_root=tmp_path / "unused",
            registry_path=REGISTRY,
            project_root=project,
            run_command=_FakeBootstrapRunner(),
        )


def test_bootstrap_help_has_no_filesystem_or_environment_side_effects(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/bootstrap_task_c_methods.py"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "官方代码" in completed.stdout
    assert "隔离环境" in completed.stdout
    assert not list(tmp_path.iterdir())


def test_bootstrap_cli_reports_a_bad_registry_without_a_traceback(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/bootstrap_task_c_methods.py"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--registry",
            str(tmp_path / "missing-registry.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "无法准备比较方法" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "cache").exists()


def _write_method_input(path: Path, *, gene_count: int = 2) -> None:
    genes = [chr(ord("A") + index) if index < 26 else f"G{index}" for index in range(gene_count)]
    expression = np.zeros((6, gene_count), dtype=np.float32)
    if gene_count >= 2:
        expression[3:, 1] = np.asarray([1.0, 1.2, 0.8], dtype=np.float32)
    np.savez(
        path,
        expression_matrix=expression,
        interventions=np.asarray(
            ["non-targeting", "non-targeting", "non-targeting", "A", "A", "A"]
        ),
        var_names=np.asarray(genes),
    )


def _materialized_public_bundle(tmp_path: Path) -> dict[str, object]:
    raw_paths: dict[str, Path] = {}
    for context, seed in (("k562", 11), ("rpe1", 23)):
        path = tmp_path / f"raw_{context}.npz"
        rng = np.random.default_rng(seed)
        labels = ["non-targeting"] * 10 + [
            source for source in ("A", "B", "C", "D", "E") for _ in range(5)
        ]
        np.savez(
            path,
            expression_matrix=rng.normal(size=(len(labels), 5)).astype(np.float32),
            interventions=np.asarray(labels),
            var_names=np.asarray(["A", "B", "C", "D", "E"]),
        )
        raw_paths[context] = path
    k562 = load_task_c_dataset(raw_paths["k562"], context_id="k562")
    rpe1 = load_task_c_dataset(raw_paths["rpe1"], context_id="rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    return materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")


def _run_method_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_task_c_method.py"), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_mean_difference_unified_cli_writes_a_complete_verified_bundle(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "allowed_train.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"

    completed = _run_method_cli(
        "--method-id",
        "mean_difference",
        "--input-npz",
        str(input_path),
        "--output-dir",
        str(output),
        "--seed",
        "11",
        "--registry",
        str(REGISTRY),
        "--asset-root",
        str(tmp_path / "method_assets"),
        "--data-status",
        "synthetic_smoke",
        "--context-id",
        "synthetic",
        "--min-cells",
        "2",
    )

    assert completed.returncode == 0, completed.stderr
    raw = pd.read_csv(output / "raw_predictions.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    assert list(raw.columns) == ["source", "target", "score"]
    assert len(predictions) == 2
    assert set(predictions.columns) == {
        "source",
        "target",
        "score",
        "returned_by_method",
    }
    status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    environment = json.loads(
        (output / "environment_manifest.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "completed_standardized_output"
    assert status["method_id"] == "mean_difference"
    assert status["artifacts"]["predictions.csv"]["sha256"].startswith("sha256:")
    assert environment["training_information"] == "partial_interventional"
    assert environment["data_status"] == "synthetic_smoke"
    sealed_parameters = json.loads(
        (output / "trial_parameters.json").read_text(encoding="utf-8")
    )
    assert sealed_parameters["parameters"] == {}
    assert sealed_parameters["trial_index"] is None
    assert sealed_parameters["stage"] == "synthetic_smoke"
    assert environment["trial_parameters"]["content"] == sealed_parameters
    assert status["trial_parameters_sha256"] == environment["trial_parameters"][
        "sha256"
    ]
    assert status["artifacts"]["trial_parameters.json"]["sha256"] == status[
        "trial_parameters_sha256"
    ]
    assert "passed_real_rehearsal" not in (output / "method_status.json").read_text(
        encoding="utf-8"
    )


def test_publication_only_cli_needs_no_data_and_never_writes_predictions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "publication"

    completed = _run_method_cli(
        "--method-id",
        "betterboost",
        "--output-dir",
        str(output),
        "--seed",
        "11",
        "--registry",
        str(REGISTRY),
        "--asset-root",
        str(tmp_path / "missing-assets"),
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    environment = json.loads(
        (output / "environment_manifest.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "official_assets_unavailable"
    assert environment["method_id"] == "betterboost"
    assert environment["registry_sha256"].startswith("sha256:")
    assert not (output / "raw_predictions.csv").exists()
    assert not (output / "predictions.csv").exists()


def test_external_commands_bind_registered_source_environment_and_permissions(
    tmp_path: Path,
) -> None:
    registry = load_task_c_method_registry(REGISTRY)
    input_path = tmp_path / "input.npz"
    output_csv = tmp_path / "raw.csv"
    assets = tmp_path / "assets"

    causalbench = build_task_c_method_command(
        registry.methods["pc"],
        input_path=input_path,
        output_csv=output_csv,
        asset_root=assets,
        seed=17,
        project_root=ROOT,
    )
    psgrn = build_task_c_method_command(
        registry.methods["guanlab_psgrn"],
        input_path=input_path,
        output_csv=output_csv,
        asset_root=assets,
        seed=17,
        project_root=ROOT,
    )

    assert causalbench[:6] == (
        "conda",
        "run",
        "-n",
        "hypersca-task-c-causalbench",
        "python",
        str(ROOT / "scripts/task_c_workers/causalbench_worker.py"),
    )
    assert causalbench[causalbench.index("--causalbench-source") + 1] == str(
        assets / "sources/causalbench"
    )
    assert causalbench[causalbench.index("--training-information") + 1] == "observational"
    assert causalbench[causalbench.index("--output-semantics") + 1] == "official_unranked_edges"
    assert psgrn[psgrn.index("--psgrn-source") + 1] == str(
        assets / "sources/guanlab_psgrn"
    )
    assert psgrn[psgrn.index("--training-information") + 1] == "partial_interventional"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"source,target\nA,B\n", "exactly source, target, and score"),
        (b"source,target,score\nA,UNKNOWN,1\n", "outside the fixed gene set"),
        (b"source,target,score\nA,B,-1\n", "non-negative"),
        (b"source,target,score\nA,B,NaN\n", "finite"),
    ],
)
def test_raw_prediction_reader_rejects_invalid_method_output(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "raw.csv"
    path.write_bytes(payload)

    with pytest.raises(TaskCMethodRunError, match=message):
        read_task_c_raw_predictions(path, ("A", "B"))


def test_raw_prediction_reader_rejects_large_symbolic_and_hard_linked_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.csv"
    target.write_text("source,target,score\nA,B,1\n", encoding="utf-8")
    linked = tmp_path / "linked.csv"
    linked.symlink_to(target)
    with pytest.raises(TaskCMethodRunError, match="symbolic link"):
        read_task_c_raw_predictions(linked, ("A", "B"))

    hardlink = tmp_path / "hardlink.csv"
    os.link(target, hardlink)
    with pytest.raises(TaskCMethodRunError, match="hard link"):
        read_task_c_raw_predictions(target, ("A", "B"))

    target.unlink()
    hardlink.unlink()
    too_large = tmp_path / "large.csv"
    too_large.write_bytes(b"source,target,score\n" + b"A,B,1\n" * 200_000)
    with pytest.raises(TaskCMethodRunError, match="too large"):
        read_task_c_raw_predictions(too_large, ("A", "B"), maximum_bytes=1024)


def test_gene_limit_is_checked_before_mean_difference_computation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "too_many_genes.npz"
    _write_method_input(input_path, gene_count=MAXIMUM_TASK_C_RUN_GENES + 1)
    called = False

    def forbidden_score(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("gene limit must be checked first")

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.score_mean_difference_network",
        forbidden_score,
    )
    with pytest.raises(TaskCMethodRunError, match="at most 256"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=input_path,
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="synthetic_smoke",
            context_id="synthetic",
            min_cells=2,
            project_root=ROOT,
        )
    assert called is False


def test_unified_run_reuses_only_an_exact_scientifically_valid_bundle(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    arguments = {
        "method_id": "mean_difference",
        "input_npz": input_path,
        "output_dir": tmp_path / "run",
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }

    first = run_task_c_method(**arguments)
    second = run_task_c_method(**arguments)
    assert first["status"] == "completed_standardized_output"
    assert second["reuse"] == "verified_existing_output"

    predictions = Path(arguments["output_dir"]) / "predictions.csv"
    predictions.write_text(
        predictions.read_text(encoding="utf-8").replace(",True", ",False", 1),
        encoding="utf-8",
    )
    with pytest.raises(TaskCMethodRunError, match="changed|hash|semantic"):
        run_task_c_method(**arguments)


def test_trial_parameters_are_bound_before_run_and_changed_candidate_cannot_reuse(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        '{"schema_version":"1.0","trial_index":3,"parameters":{}}\n',
        encoding="utf-8",
    )
    arguments = {
        "method_id": "mean_difference",
        "input_npz": input_path,
        "output_dir": tmp_path / "run",
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "trial_parameters_path": candidate,
        "project_root": ROOT,
    }

    run_task_c_method(**arguments)
    record_path = tmp_path / "run/trial_parameters.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["trial_index"] == 3
    assert record["method_id"] == "mean_difference"
    assert record["seed"] == 11
    assert record["training_input_sha256"].startswith("sha256:")
    assert "tune_input_sha256" not in record

    candidate.write_text(
        '{"schema_version":"1.0","trial_index":4,"parameters":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskCMethodRunError, match="environment|identity|parameter"):
        run_task_c_method(**arguments)


def test_formal_trial_candidate_must_train_before_tuning(tmp_path: Path) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        '{"schema_version":"1.0","trial_index":0,"parameters":{}}\n',
        encoding="utf-8",
    )

    with pytest.raises(TaskCMethodRunError, match="train stage"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=Path(bundle["within"]["k562"]["tune"]),
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="k562",
            min_cells=5,
            public_manifest_path=Path(bundle["public_manifest"]),
            trial_parameters_path=candidate,
            project_root=ROOT,
        )


def test_bound_trial_parameter_artifact_cannot_be_replaced_and_resealed(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "mean_difference",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)
    parameters_path = output / "trial_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameters["trial_index"] = 9
    parameters_path.write_text(json.dumps(parameters) + "\n", encoding="utf-8")
    _update_outer_artifact_record(output, "trial_parameters.json")

    with pytest.raises(TaskCMethodRunError, match="parameter|environment|identity"):
        run_task_c_method(**arguments)


def test_partial_output_and_private_or_symbolic_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "method_status.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskCMethodRunError, match="incomplete|unrecognized"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=input_path,
            output_dir=partial,
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="synthetic_smoke",
            context_id="synthetic",
            min_cells=2,
            project_root=ROOT,
        )

    private_dir = tmp_path / "private"
    private_dir.mkdir()
    private_input = private_dir / "input.npz"
    _write_method_input(private_input)
    with pytest.raises(TaskCMethodRunError, match="private"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=private_input,
            output_dir=tmp_path / "private-run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="synthetic_smoke",
            context_id="synthetic",
            min_cells=2,
            project_root=ROOT,
        )

    symlink = tmp_path / "input-link.npz"
    symlink.symlink_to(input_path)
    with pytest.raises(TaskCMethodRunError, match="symbolic link"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=symlink,
            output_dir=tmp_path / "symlink-run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="synthetic_smoke",
            context_id="synthetic",
            min_cells=2,
            project_root=ROOT,
        )


def test_reuse_recomputes_the_mean_difference_scientific_result(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "mean_difference",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)

    raw = pd.read_csv(output / "raw_predictions.csv")
    raw.loc[0, "score"] = 9.0
    raw.to_csv(output / "raw_predictions.csv", index=False)
    normalized = normalize_task_c_predictions(raw, ["A", "B"])
    normalized.to_csv(output / "predictions.csv", index=False)
    status_path = output / "method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    for name in ("raw_predictions.csv", "predictions.csv"):
        payload = (output / name).read_bytes()
        status["artifacts"][name] = {
            "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            "size_bytes": len(payload),
        }
    _write_resealed_outer_status(status_path, status)

    with pytest.raises(TaskCMethodRunError, match="scientific semantics"):
        run_task_c_method(**arguments)


def test_formal_run_rejects_an_incomplete_public_inventory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    input_path = bundle / "train.npz"
    _write_method_input(input_path)
    input_hash = f"sha256:{hashlib.sha256(input_path.read_bytes()).hexdigest()}"
    identity = {
        "schema_version": "1.0",
        "split_id": "test-split",
        "seed": 11,
        "min_cells_per_intervention": 2,
        "input_sha256": {"k562": input_hash, "rpe1": input_hash},
        "content_sha256": {"k562": input_hash, "rpe1": input_hash},
        "gene_names_sha256": input_hash,
        "gene_projection": {
            "rule": "sorted_common_gene_intersection_v1",
            "common_gene_count": 2,
        },
    }
    manifest = {
        **identity,
        "train_sources": ["A"],
        "tune_sources": [],
        "holdout_source_count": 0,
        "materialization_identity": identity,
        "files": {"train.npz": input_hash},
    }
    manifest_path = bundle / "public_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskCMethodRunError, match="complete public file inventory"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=input_path,
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="k562",
            min_cells=2,
            public_manifest_path=manifest_path,
            project_root=ROOT,
        )


def test_reuse_rejects_a_rewritten_environment_record_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "mean_difference",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)

    environment_path = output / "environment_manifest.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["role"] = "rewritten-role"
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = output / "method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    payload = environment_path.read_bytes()
    status["artifacts"]["environment_manifest.json"] = {
        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "size_bytes": len(payload),
    }
    _write_resealed_outer_status(status_path, status)

    with pytest.raises(TaskCMethodRunError, match="environment.*changed"):
        run_task_c_method(**arguments)


def test_formal_run_rejects_a_context_label_that_disagrees_with_public_path(
    tmp_path: Path,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)

    with pytest.raises(TaskCMethodRunError, match="context.*public path"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=Path(bundle["within"]["k562"]["train"]),
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="rpe1",
            min_cells=5,
            public_manifest_path=Path(bundle["public_manifest"]),
            project_root=ROOT,
        )


def test_formal_run_rejects_a_public_manifest_with_split_identity_drift(
    tmp_path: Path,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    manifest_path = Path(bundle["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["materialization_identity"]["seed"] = 23
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskCMethodRunError, match="materialization identity"):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=Path(bundle["within"]["k562"]["train"]),
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="k562",
            min_cells=5,
            public_manifest_path=manifest_path,
            project_root=ROOT,
        )


def test_cross_environment_derived_input_is_recomputed_before_mean_run(
    tmp_path: Path,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    derived = materialize_task_c_derived_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        direction="k562_to_rpe1",
        stage="refit",
        output_dir=tmp_path / "derived",
    )

    status = run_task_c_method(
        method_id="mean_difference",
        input_npz=Path(derived["input_npz"]),
        derived_input_manifest_path=Path(derived["manifest"]),
        output_dir=tmp_path / "run",
        seed=11,
        registry_path=REGISTRY,
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id="k562_to_rpe1",
        min_cells=5,
        public_manifest_path=Path(bundle["public_manifest"]),
        project_root=ROOT,
    )

    assert status["status"] == "completed_standardized_output"
    environment = json.loads(
        (tmp_path / "run/environment_manifest.json").read_text(encoding="utf-8")
    )
    assert environment["run_identity"]["derived_input_manifest_sha256"].startswith(
        "sha256:"
    )
    with np.load(Path(derived["input_npz"]), allow_pickle=False) as archive:
        assert set(archive.files) == {
            "expression_matrix",
            "interventions",
            "var_names",
            "environment_labels",
        }
        assert set(archive["environment_labels"].tolist()) == {"k562", "rpe1"}


def test_cross_environment_formula_uses_control_population_standard_deviation(
) -> None:
    genes = ("A", "B")
    first = np.asarray([[1.0, 2.0], [1.0, 4.0], [3.0, 6.0]])
    second = np.asarray([[5.0, 8.0], [5.0, 12.0], [7.0, 16.0]])
    labels = np.asarray(["non-targeting", "non-targeting", "A"])

    expression, interventions, observed_genes, environments = (
        _standardize_and_concatenate(
            (
                ("k562", first, labels, genes),
                ("rpe1", second, labels, genes),
            )
        )
    )

    np.testing.assert_allclose(
        expression,
        np.asarray(
            [
                [0.0, -1.0],
                [0.0, 1.0],
                [2.0, 3.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        ),
    )
    assert interventions.tolist() == labels.tolist() * 2
    assert observed_genes == genes
    assert environments.tolist() == ["k562"] * 3 + ["rpe1"] * 3


def test_cross_environment_formula_requires_two_controls() -> None:
    with pytest.raises(TaskCMethodRunError, match="at least two|2 control"):
        _standardize_and_concatenate(
            (
                (
                    "k562",
                    np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                    np.asarray(["non-targeting", "A"]),
                    ("A", "B"),
                ),
                (
                    "rpe1",
                    np.asarray([[1.0, 2.0], [1.0, 4.0]]),
                    np.asarray(["non-targeting", "non-targeting"]),
                    ("A", "B"),
                ),
            )
        )


@pytest.mark.parametrize("mutation", ["parent", "direction", "labels", "output"])
def test_cross_environment_derived_input_rejects_identity_or_content_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    derived = materialize_task_c_derived_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        direction="k562_to_rpe1",
        stage="refit",
        output_dir=tmp_path / "derived",
    )
    input_path = Path(derived["input_npz"])
    manifest_path = Path(derived["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "parent":
        manifest["contexts"][0]["parent_sha256"] = "sha256:" + "0" * 64
    elif mutation == "direction":
        manifest["direction"] = "rpe1_to_k562"
    else:
        with np.load(input_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if mutation == "labels":
            arrays["environment_labels"] = np.asarray(
                ["k562"] * len(arrays["environment_labels"])
            )
        else:
            arrays["expression_matrix"] = arrays["expression_matrix"].copy()
            arrays["expression_matrix"][0, 0] += 1.0
        np.savez(input_path, **arrays)
        payload = input_path.read_bytes()
        manifest["output"]["sha256"] = (
            f"sha256:{hashlib.sha256(payload).hexdigest()}"
        )
        manifest["output"]["size_bytes"] = len(payload)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TaskCMethodRunError,
        match="profile|derived|parent|direction|environment",
    ):
        run_task_c_method(
            method_id="mean_difference",
            input_npz=input_path,
            derived_input_manifest_path=manifest_path,
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="k562_to_rpe1",
            min_cells=5,
            public_manifest_path=Path(bundle["public_manifest"]),
            project_root=ROOT,
        )


@pytest.mark.parametrize("changed_relative", TASK_C_RUNTIME_CLOSURE)
def test_runtime_code_closure_is_complete_and_each_dependency_blocks_reuse(
    tmp_path: Path,
    changed_relative: str,
) -> None:
    closure = set(TASK_C_RUNTIME_CLOSURE)
    project = tmp_path / "project"
    for relative in closure:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "mean_difference",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": project,
    }

    run_task_c_method(**arguments)
    environment = json.loads(
        (output / "environment_manifest.json").read_text(encoding="utf-8")
    )
    assert closure <= set(environment["code"])

    changed = project / changed_relative
    changed.write_bytes(changed.read_bytes() + b"\n# changed after the run\n")
    with pytest.raises(TaskCMethodRunError, match="identity|environment.*changed"):
        run_task_c_method(**arguments)


def _patch_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inner_status: str,
    write_raw: bool,
    raw_relation: tuple[str, str] = ("A", "B"),
) -> None:
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._asset_snapshots",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._external_source_digest",
        lambda *args, **kwargs: None,
    )

    def fake_run(
        command: object,
        *,
        output_dir: Path,
        timeout_seconds: object,
    ) -> dict[str, object]:
        del command, timeout_seconds
        destination = Path(output_dir)
        destination.mkdir()
        if write_raw:
            (destination / "worker_predictions.csv").write_text(
                "source,target,score\n"
                f"{raw_relation[0]},{raw_relation[1]},1\n",
                encoding="utf-8",
            )
        status = {"schema_version": "1.0", "status": inner_status}
        (destination / "method_status.json").write_text(
            json.dumps(status, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "resource_usage.json").write_text(
            json.dumps({"schema_version": "1.0"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return status

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.run_isolated_method",
        fake_run,
    )


def _update_outer_artifact_record(output: Path, relative: str) -> None:
    status_path = output / "method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    payload = (output / relative).read_bytes()
    status["artifacts"][relative] = {
        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "size_bytes": len(payload),
    }
    _write_resealed_outer_status(status_path, status)


def _write_resealed_outer_status(
    status_path: Path,
    status: dict[str, object],
) -> None:
    status.pop("status_content_sha256", None)
    encoded = json.dumps(
        status,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    status["status_content_sha256"] = (
        f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    )
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_external_completed_output_requires_inner_completed_raw_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_external_runtime(
        monkeypatch,
        inner_status="completed_raw_inference",
        write_raw=True,
    )
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "pc",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)

    inner_path = output / "raw_runtime/method_status.json"
    inner_path.write_text(
        json.dumps({"schema_version": "1.0", "status": "failed_timeout"}) + "\n",
        encoding="utf-8",
    )
    _update_outer_artifact_record(output, "raw_runtime/method_status.json")
    with pytest.raises(TaskCMethodRunError, match="inner|raw inference|status"):
        run_task_c_method(**arguments)


def test_external_completed_output_rejects_unknown_inner_evidence_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_external_runtime(
        monkeypatch,
        inner_status="completed_raw_inference",
        write_raw=True,
    )
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "pc",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)

    unexpected = output / "raw_runtime/unexplained.bin"
    unexpected.write_bytes(b"not part of the frozen evidence contract")
    _update_outer_artifact_record(output, "raw_runtime/unexplained.bin")
    with pytest.raises(TaskCMethodRunError, match="runtime|evidence|file set"):
        run_task_c_method(**arguments)


def test_external_failed_output_cannot_be_relabelled_or_gain_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_external_runtime(
        monkeypatch,
        inner_status="failed_timeout",
        write_raw=False,
    )
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "pc",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)

    (output / "predictions.csv").write_text(
        "source,target,score,returned_by_method\nA,B,1,True\n",
        encoding="utf-8",
    )
    _update_outer_artifact_record(output, "predictions.csv")
    status_path = output / "method_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "official_code_incompatible"
    _write_resealed_outer_status(status_path, status)
    with pytest.raises(TaskCMethodRunError, match="failed|prediction|inner|status"):
        run_task_c_method(**arguments)


def test_external_failed_output_rejects_unknown_top_level_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_external_runtime(
        monkeypatch,
        inner_status="failed_timeout",
        write_raw=False,
    )
    input_path = tmp_path / "input.npz"
    _write_method_input(input_path)
    output = tmp_path / "run"
    arguments = {
        "method_id": "pc",
        "input_npz": input_path,
        "output_dir": output,
        "seed": 11,
        "registry_path": REGISTRY,
        "asset_root": tmp_path / "assets",
        "data_status": "synthetic_smoke",
        "context_id": "synthetic",
        "min_cells": 2,
        "project_root": ROOT,
    }
    run_task_c_method(**arguments)

    unexpected = output / "unexplained.txt"
    unexpected.write_text("not declared evidence\n", encoding="utf-8")
    _update_outer_artifact_record(output, "unexplained.txt")
    with pytest.raises(TaskCMethodRunError, match="failed|file set"):
        run_task_c_method(**arguments)


def test_external_method_accepts_a_recomputed_formal_cross_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    derived = materialize_task_c_derived_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        direction="k562_to_rpe1",
        stage="refit",
        output_dir=tmp_path / "derived",
    )
    derived_manifest = json.loads(Path(derived["manifest"]).read_text(encoding="utf-8"))
    first_gene, second_gene = derived_manifest["gene_selection"]["ordered_genes"][:2]
    _patch_external_runtime(
        monkeypatch,
        inner_status="completed_raw_inference",
        write_raw=True,
        raw_relation=(first_gene, second_gene),
    )

    status = run_task_c_method(
        method_id="pc",
        input_npz=Path(derived["input_npz"]),
        derived_input_manifest_path=Path(derived["manifest"]),
        output_dir=tmp_path / "run",
        seed=11,
        registry_path=REGISTRY,
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id="k562_to_rpe1",
        min_cells=5,
        public_manifest_path=Path(bundle["public_manifest"]),
        project_root=ROOT,
    )

    assert status["status"] == "completed_standardized_output"
    environment = json.loads(
        (tmp_path / "run/environment_manifest.json").read_text(encoding="utf-8")
    )
    assert environment["command"]["options"] == [
        "--input-npz",
        "--output-csv",
        "--model-name",
        "--causalbench-source",
        "--training-information",
        "--seed",
        "--output-semantics",
    ]


def test_hypersca_dispatches_the_exact_validated_profile_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.evaluation.task_c_profile_input import (
        materialize_task_c_profile_input,
    )

    bundle = _materialized_public_bundle(tmp_path)
    profile = materialize_task_c_profile_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        condition="within_environment",
        context_id="k562",
        output_dir=tmp_path / "profile",
        profile="connection",
    )
    profile_manifest = json.loads(
        Path(profile["manifest"]).read_text(encoding="utf-8")
    )
    gene_list = tmp_path / "genes.json"
    gene_list.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "selection_id": "profile-connection",
                "selection_basis": "统一比较范围记录中的固定基因顺序",
                "genes": profile_manifest["gene_selection"]["ordered_genes"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    captured: dict[str, tuple[str, ...]] = {}

    def fake_run(
        command: Sequence[str],
        *,
        output_dir: Path,
        timeout_seconds: object,
    ) -> dict[str, object]:
        del timeout_seconds
        captured["command"] = tuple(command)
        runtime = Path(output_dir)
        runtime.mkdir()
        inner = {"schema_version": "1.0", "status": "completed_raw_inference"}
        (runtime / "method_status.json").write_text(
            json.dumps(inner) + "\n", encoding="utf-8"
        )
        (runtime / "resource_usage.json").write_text(
            '{"schema_version":"1.0"}\n', encoding="utf-8"
        )
        return inner

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.run_isolated_method",
        fake_run,
    )
    with pytest.raises(TaskCMethodRunError):
        run_task_c_method(
            method_id="hypersca_c",
            input_npz=Path(profile["input_npz"]),
            derived_input_manifest_path=Path(profile["manifest"]),
            output_dir=tmp_path / "run",
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id="k562",
            public_manifest_path=Path(bundle["public_manifest"]),
            hypersca_config_path=config,
            gene_list_path=gene_list,
            project_root=ROOT,
        )

    command = captured["command"]
    dispatched_input = Path(command[command.index("--profile-input") + 1])
    assert dispatched_input != Path(profile["input_npz"])
    assert dispatched_input.name == "profile_input.npz"
    assert "--profile-identity-input" not in command
    assert command[command.index("--profile-manifest") + 1] == str(
        profile["manifest"]
    )
    assert "--context" not in command


def test_external_worker_reads_a_staged_copy_of_validated_profile_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    derived = materialize_task_c_derived_input(
        public_manifest_path=Path(bundle["public_manifest"]),
        direction="k562_to_rpe1",
        stage="refit",
        output_dir=tmp_path / "profile",
    )
    original_bytes = Path(derived["input_npz"]).read_bytes()
    observed: dict[str, object] = {}

    def fake_run(
        command: Sequence[str],
        *,
        output_dir: Path,
        timeout_seconds: object,
    ) -> dict[str, object]:
        del timeout_seconds
        input_path = Path(command[command.index("--input-npz") + 1])
        observed["input_path"] = input_path
        observed["input_bytes"] = input_path.read_bytes()
        with np.load(input_path, allow_pickle=False) as archive:
            first, second = archive["var_names"][:2].tolist()
        runtime = Path(output_dir)
        runtime.mkdir()
        (runtime / "worker_predictions.csv").write_text(
            f"source,target,score\n{first},{second},1\n", encoding="utf-8"
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
        "src.evaluation.task_c_method_run._asset_snapshots", lambda *args: {}
    )
    monkeypatch.setattr(
        "src.evaluation.task_c_method_run._external_source_digest",
        lambda *args: None,
    )
    run_task_c_method(
        method_id="pc",
        input_npz=Path(derived["input_npz"]),
        derived_input_manifest_path=Path(derived["manifest"]),
        output_dir=tmp_path / "run",
        seed=11,
        registry_path=REGISTRY,
        asset_root=tmp_path / "assets",
        data_status="external_benchmark",
        context_id="k562_to_rpe1",
        min_cells=5,
        public_manifest_path=Path(bundle["public_manifest"]),
        project_root=ROOT,
    )

    assert observed["input_bytes"] == original_bytes
    assert ".fixed-inputs-" in Path(observed["input_path"]).parent.name
    assert Path(observed["input_path"]) != Path(derived["input_npz"])


@pytest.mark.parametrize("nested_case", ["missing_raw", "missing_evidence_files"])
def test_hypersca_invalid_nested_output_is_classified_as_invalid_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_case: str,
) -> None:
    bundle = _materialized_public_bundle(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    genes = tmp_path / "genes.json"
    genes.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "selection_id": "nested-check",
                "selection_basis": "核对嵌套结果分类",
                "genes": ["A", "B"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_hypersca_run(
        command: Sequence[str],
        *,
        output_dir: Path,
        timeout_seconds: object,
    ) -> dict[str, object]:
        del timeout_seconds
        runtime = Path(output_dir)
        runtime.mkdir()
        inner_status = {"schema_version": "1.0", "status": "completed_raw_inference"}
        (runtime / "method_status.json").write_text(
            json.dumps(inner_status) + "\n", encoding="utf-8"
        )
        (runtime / "resource_usage.json").write_text(
            '{"schema_version":"1.0"}\n', encoding="utf-8"
        )
        if nested_case == "missing_evidence_files":
            raw_output = Path(command[command.index("--output-dir") + 1])
            raw_output.mkdir()
            (raw_output / "raw_predictions.csv").write_text(
                "source,target,score\nA,B,1\n",
                encoding="utf-8",
            )
        return inner_status

    monkeypatch.setattr(
        "src.evaluation.task_c_method_run.run_isolated_method",
        fake_hypersca_run,
    )
    output = tmp_path / "run"
    with pytest.raises(TaskCMethodRunError):
        run_task_c_method(
            method_id="hypersca_c",
            input_npz=None,
            output_dir=output,
            seed=11,
            registry_path=REGISTRY,
            asset_root=tmp_path / "assets",
            data_status="external_benchmark",
            context_id=None,
            public_manifest_path=Path(bundle["public_manifest"]),
            context_values=(f"k562={bundle['within']['k562']['refit']}",),
            hypersca_config_path=config,
            gene_list_path=genes,
            project_root=ROOT,
        )
    outer = json.loads((output / "method_status.json").read_text(encoding="utf-8"))
    assert outer["status"] == "failed_invalid_output"
    assert not (output / "predictions.csv").exists()
