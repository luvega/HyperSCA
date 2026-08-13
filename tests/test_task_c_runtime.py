from __future__ import annotations

import json
import hashlib
import io
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from src.evaluation import task_c_runtime as runtime_module
from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistryError,
    load_task_c_method_registry,
)
from src.evaluation.task_c_runtime import (
    TaskCRuntimeError,
    _run_bounded_command,
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
