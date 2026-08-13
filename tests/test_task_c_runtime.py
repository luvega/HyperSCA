from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from src.evaluation.task_c_method_registry import load_task_c_method_registry
from src.evaluation.task_c_runtime import (
    TaskCRuntimeError,
    _parse_maximum_resident_kib,
    bootstrap_task_c_methods,
    classify_publication_only_method,
    run_isolated_method,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/task_c_methods_v1.json"


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

    assert result["status"] == "official_code_incompatible"
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

    def __call__(self, command: list[str], **kwargs: object) -> _FakeCompleted:
        assert command and all(type(part) is str for part in command)
        assert "shell" not in kwargs
        assert kwargs.get("check") is True
        self.calls.append(tuple(command))
        if command[:2] == ["git", "clone"]:
            repository, destination_text = command[2], command[3]
            destination = Path(destination_text)
            destination.mkdir(parents=True)
            (destination / ".git").mkdir()
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
            if operation == ["status", "--porcelain"]:
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
            environment_file = Path(command[4])
            name = next(
                line.split(":", 1)[1].strip()
                for line in environment_file.read_text(encoding="utf-8").splitlines()
                if line.startswith("name:")
            )
            self.environments.add(name)
            return _FakeCompleted()
        if command[:3] == ["conda", "env", "update"]:
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
    assert not list(cache.rglob("*.tmp"))


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
