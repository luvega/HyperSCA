"""隔离运行任务 C 比较方法，并保存可核对的失败和资源记录。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
from typing import Any
import uuid

from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistry,
    TaskCMethodSpec,
    load_task_c_method_registry,
)


SCHEMA_VERSION = "1.0"
GNU_TIME = Path("/usr/bin/time")
DEFAULT_MAXIMUM_OUTPUT_BYTES = 64 * 1024
HARD_MAXIMUM_OUTPUT_BYTES = 1024 * 1024
_ALLOWED_CACHE_ENTRIES = {
    "bootstrap_identity.json",
    "environment_manifests",
    "sources",
    "status",
}
_RESOURCE_LIMIT_CODES = {137, 152}
_RESOURCE_LIMIT_SIGNALS = {signal.SIGKILL, signal.SIGXCPU}
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s\"']+)")


class TaskCRuntimeError(ValueError):
    """运行位置、命令或官方资产不符合固定比较规则。"""


class _TailBuffer:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.content = bytearray()
        self.total_bytes = 0

    def add(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        self.content.extend(chunk)
        if len(self.content) > self.maximum_bytes:
            del self.content[: len(self.content) - self.maximum_bytes]

    @property
    def was_truncated(self) -> bool:
        return self.total_bytes > self.maximum_bytes

    def text(self) -> str:
        decoded = bytes(self.content).decode("utf-8", errors="replace")
        return _PRIVATE_PATH.sub("<absolute-path>", decoded)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.exists() and component.is_symlink():
            raise TaskCRuntimeError(f"{label} must not contain a symbolic link")


def _require_safe_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TaskCRuntimeError(f"{label} is missing or unreadable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TaskCRuntimeError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise TaskCRuntimeError(f"{label} must not be a hard-linked file")


def _atomic_create_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        content = _json_bytes(payload)
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise TaskCRuntimeError(
                f"refusing to overwrite existing record {path.name}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_replace_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        _require_safe_regular_file(path, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        content = _json_bytes(payload)
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    _require_safe_regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskCRuntimeError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise TaskCRuntimeError(f"{label} must contain one JSON object")
    return value


def classify_publication_only_method(spec: TaskCMethodSpec) -> dict[str, object]:
    """明确记录只有论文、没有可运行官方代码的方法。"""

    if spec.source_kind != "publication_only" or not spec.publication:
        raise TaskCRuntimeError("method is not a publication-only comparison")
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": spec.method_id,
        "status": "official_assets_unavailable",
        "publication": spec.publication,
        "reason": (
            "The registered primary source describes the method but does not "
            "provide runnable official code."
        ),
    }


def _validated_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TaskCRuntimeError("command must be a non-empty sequence of strings")
    if not command:
        raise TaskCRuntimeError("command must not be empty")
    values: list[str] = []
    for value in command:
        if type(value) is not str or not value or "\x00" in value:
            raise TaskCRuntimeError(
                "command arguments must be non-empty strings without NUL bytes"
            )
        values.append(value)
    return tuple(values)


def _validated_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskCRuntimeError(f"{label} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise TaskCRuntimeError(f"{label} must be a finite positive number")
    return normalized


def _prepare_empty_output_directory(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    _reject_symlink_components(destination, "output directory")
    if destination.exists():
        if not destination.is_dir():
            raise TaskCRuntimeError("output location must be a directory")
        try:
            if any(destination.iterdir()):
                raise TaskCRuntimeError(
                    "output directory must be new or an existing empty directory"
                )
        except OSError as exc:
            raise TaskCRuntimeError(
                "output directory cannot be inspected safely"
            ) from exc
    else:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise TaskCRuntimeError(
                "output directory cannot be created safely"
            ) from exc
    return destination


def _command_trace(command: Sequence[str]) -> dict[str, object]:
    encoded = json.dumps(
        list(command), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "executable_name": Path(command[0]).name,
        "argument_count": len(command),
        "command_sha256": _sha256_bytes(encoded),
    }


def _drain_stream(stream: Any, target: _TailBuffer) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            target.add(chunk)
    finally:
        stream.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    # start_new_session=True makes the leader PID the process-group ID.  Keep
    # using that known ID after the leader exits so background children cannot
    # escape merely by outliving their parent.
    group_id = process.pid
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 0.5
    group_still_exists = True
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            group_still_exists = False
            break
        time.sleep(0.01)
    if group_still_exists:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _parse_maximum_resident_kib(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if "Maximum resident set size" not in line:
            continue
        value_text = line.rsplit(":", 1)[-1].strip()
        if re.fullmatch(r"[0-9]+", value_text) is None:
            return None
        try:
            value = int(value_text)
        except ValueError:
            return None
        return value if value >= 0 else None
    return None


def _executable_is_missing(executable: str) -> bool:
    if "/" in executable:
        candidate = Path(executable)
        return not candidate.is_file() or not os.access(candidate, os.X_OK)
    return shutil.which(executable) is None


def _classify_return(return_code: int, stderr: str) -> str:
    lowered = stderr.lower()
    if "failed_invalid_output" in lowered:
        return "failed_invalid_output"
    if return_code == 0:
        return "completed_raw_inference"
    signal_number = -return_code if return_code < 0 else return_code - 128
    if (
        return_code in _RESOURCE_LIMIT_CODES
        or signal_number in _RESOURCE_LIMIT_SIGNALS
        or "out of memory" in lowered
        or "memoryerror" in lowered
    ):
        return "failed_resource_limit"
    return "official_code_incompatible"


def run_isolated_method(
    command: Sequence[str],
    *,
    output_dir: str | Path,
    timeout_seconds: int | float,
    maximum_output_bytes: int = DEFAULT_MAXIMUM_OUTPUT_BYTES,
) -> dict[str, object]:
    """在独立进程组运行一种方法，并限制保留的输出文字大小。"""

    normalized_command = _validated_command(command)
    timeout = _validated_positive_number(timeout_seconds, "timeout")
    if isinstance(maximum_output_bytes, bool) or type(maximum_output_bytes) is not int:
        raise TaskCRuntimeError("maximum output bytes must be a positive integer")
    if maximum_output_bytes <= 0:
        raise TaskCRuntimeError("maximum output bytes must be a positive integer")
    if maximum_output_bytes > HARD_MAXIMUM_OUTPUT_BYTES:
        raise TaskCRuntimeError(
            f"maximum output bytes cannot exceed {HARD_MAXIMUM_OUTPUT_BYTES}"
        )
    destination = _prepare_empty_output_directory(output_dir)
    resource_temporary = destination / f".resource_usage.{uuid.uuid4().hex}.tmp"
    stdout = _TailBuffer(maximum_output_bytes)
    stderr = _TailBuffer(maximum_output_bytes)
    started = time.monotonic()
    return_code: int | None = None
    status = "official_code_incompatible"
    resource_meter = "gnu_time_v"
    process: subprocess.Popen[bytes] | None = None
    threads: list[threading.Thread] = []
    missing_method = _executable_is_missing(normalized_command[0])
    missing_meter = not GNU_TIME.is_file() or not os.access(GNU_TIME, os.X_OK)
    try:
        if missing_method:
            stderr.add(b"registered method executable could not be started")
        elif missing_meter:
            resource_meter = "unavailable"
            stderr.add(
                b"GNU time resource meter is unavailable; method was not started"
            )
        else:
            timed_command = [
                str(GNU_TIME),
                "-v",
                "-o",
                str(resource_temporary),
                "--",
                *normalized_command,
            ]
            environment = dict(os.environ)
            environment["LC_ALL"] = "C"
            process = subprocess.Popen(
                timed_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                bufsize=0,
                env=environment,
            )
            assert process.stdout is not None and process.stderr is not None
            threads = [
                threading.Thread(
                    target=_drain_stream,
                    args=(process.stdout, stdout),
                    daemon=True,
                ),
                threading.Thread(
                    target=_drain_stream,
                    args=(process.stderr, stderr),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status = "failed_timeout"
                _terminate_process_group(process)
                return_code = None
            else:
                # A method is not allowed to leave background descendants behind.
                _terminate_process_group(process)
            for thread in threads:
                thread.join(timeout=1)
            if status != "failed_timeout":
                assert return_code is not None
                status = _classify_return(return_code, stderr.text())
    except BaseException:
        if process is not None:
            _terminate_process_group(process)
        for thread in threads:
            thread.join(timeout=1)
        resource_temporary.unlink(missing_ok=True)
        raise

    elapsed = float(time.monotonic() - started)
    if not math.isfinite(elapsed) or elapsed < 0:
        resource_temporary.unlink(missing_ok=True)
        raise TaskCRuntimeError("elapsed time could not be measured safely")
    if return_code is not None and return_code < 0:
        terminating_signal: int | None = -return_code
    elif return_code is not None and return_code >= 128:
        terminating_signal = return_code - 128
    else:
        terminating_signal = None
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "return_code": return_code,
        "terminating_signal": terminating_signal,
        "elapsed_seconds": elapsed,
        "stdout_tail": stdout.text(),
        "stderr_tail": stderr.text(),
        "output_was_truncated": {
            "stdout": stdout.was_truncated,
            "stderr": stderr.was_truncated,
        },
        "command_trace": _command_trace(normalized_command),
    }
    resource_values: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "elapsed_seconds": elapsed,
        "maximum_resident_kib": _parse_maximum_resident_kib(resource_temporary),
        "resource_meter": resource_meter,
    }
    resource_temporary.unlink(missing_ok=True)
    created: list[Path] = []
    try:
        for path, record in (
            (destination / "method_status.json", payload),
            (destination / "resource_usage.json", resource_values),
        ):
            _atomic_create_json(path, record)
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return payload


def _environment_name(path: Path) -> str:
    _require_safe_regular_file(path, f"environment file {path.name}")
    names = [
        line.split(":", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("name:")
    ]
    if len(names) != 1 or not names[0]:
        raise TaskCRuntimeError(f"environment name is missing from {path.name}")
    return names[0]


def _source_records(registry: TaskCMethodRegistry) -> dict[str, dict[str, str]]:
    records = {
        "causalbench": {
            "repository": registry.causalbench["repository"],
            "commit": registry.causalbench["commit"],
        }
    }
    for method in registry.methods.values():
        if method.source_kind == "git":
            assert method.repository is not None and method.commit is not None
            records[method.method_id] = {
                "repository": method.repository,
                "commit": method.commit,
            }
    return records


def _environment_records(
    registry: TaskCMethodRegistry,
    project_root: Path,
) -> dict[str, Path]:
    records = {
        registry.causalbench["environment"]: project_root
        / "envs/task_c/causalbench.yml"
    }
    for method in registry.methods.values():
        if method.source_kind == "git":
            assert method.environment is not None
            records[method.environment] = project_root / "envs/task_c/psgrn.yml"
    for expected_name, path in records.items():
        if _environment_name(path) != expected_name:
            raise TaskCRuntimeError(
                f"environment name in {path.name} does not match the method registry"
            )
        content = path.read_text(encoding="utf-8")
        causalbench_pin = (
            f"{registry.causalbench['repository']}@{registry.causalbench['commit']}"
        )
        if causalbench_pin not in content:
            raise TaskCRuntimeError(
                f"environment {expected_name} does not contain the fixed CausalBench source"
            )
    return records


def _bootstrap_identity(
    registry_path: Path,
    sources: Mapping[str, Mapping[str, str]],
    environments: Mapping[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": _sha256_file(registry_path),
        "sources": {key: dict(value) for key, value in sources.items()},
        "environment_files": {
            name: _sha256_file(path) for name, path in environments.items()
        },
    }


def _run_checked(
    runner: Callable[..., Any],
    command: list[str],
    *,
    capture_output: bool = False,
) -> Any:
    kwargs: dict[str, object] = {"check": True}
    if capture_output:
        kwargs.update({"capture_output": True, "text": True})
    try:
        return runner(command, **kwargs)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskCRuntimeError(
            f"official asset command failed: {Path(command[0]).name}"
        ) from exc


def _validate_source_checkout(
    source: Path,
    expected: Mapping[str, str],
    runner: Callable[..., Any],
) -> None:
    if source.is_symlink() or not source.is_dir() or not (source / ".git").is_dir():
        raise TaskCRuntimeError("official source must be a regular Git checkout")
    repository = _run_checked(
        runner,
        ["git", "-C", str(source), "remote", "get-url", "origin"],
        capture_output=True,
    ).stdout.strip()
    revision = _run_checked(
        runner,
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
    ).stdout.strip()
    changes = _run_checked(
        runner,
        ["git", "-C", str(source), "status", "--porcelain"],
        capture_output=True,
    ).stdout.strip()
    if repository != expected["repository"]:
        raise TaskCRuntimeError(
            "official source repository does not match the registry"
        )
    if revision != expected["commit"]:
        raise TaskCRuntimeError("official source is not at the fixed commit")
    if changes:
        raise TaskCRuntimeError("official source contains unreviewed local changes")


def _ensure_source(
    source: Path,
    expected: Mapping[str, str],
    runner: Callable[..., Any],
) -> None:
    if source.exists() or source.is_symlink():
        _validate_source_checkout(source, expected, runner)
        return
    try:
        _run_checked(
            runner,
            ["git", "clone", expected["repository"], str(source)],
        )
        _run_checked(
            runner,
            ["git", "-C", str(source), "checkout", "--detach", expected["commit"]],
        )
        _validate_source_checkout(source, expected, runner)
    except BaseException:
        if source.exists() and not source.is_symlink():
            shutil.rmtree(source)
        raise


def _existing_environment_names(payload: object) -> set[str]:
    if not isinstance(payload, dict) or "envs" not in payload:
        raise TaskCRuntimeError("conda environment list has an unexpected format")
    values = payload["envs"]
    if not isinstance(values, list) or any(type(value) is not str for value in values):
        raise TaskCRuntimeError("conda environment list has an unexpected format")
    return {Path(value).name for value in values}


def _normalized_packages(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        raise TaskCRuntimeError("conda package list has an unexpected format")
    packages: list[dict[str, str]] = []
    names: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise TaskCRuntimeError("conda package list has an unexpected format")
        normalized: dict[str, str] = {}
        for key in ("name", "version", "build_string"):
            value = raw.get(key, "")
            if type(value) is not str or (key != "build_string" and not value):
                raise TaskCRuntimeError("conda package list has an unexpected format")
            normalized[key] = value
        if normalized["name"] in names:
            raise TaskCRuntimeError(
                "conda package list contains duplicate package names"
            )
        names.add(normalized["name"])
        packages.append(normalized)
    return sorted(packages, key=lambda item: item["name"])


def _write_same_or_new(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        if _read_json_object(path, path.name) != dict(payload):
            raise TaskCRuntimeError(f"existing {path.name} has a different identity")
        return
    _atomic_create_json(path, payload)


def _prepare_cache_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    _reject_symlink_components(root, "cache root")
    if root.exists():
        if not root.is_dir():
            raise TaskCRuntimeError("cache root must be a directory")
        unexpected = {entry.name for entry in root.iterdir()} - _ALLOWED_CACHE_ENTRIES
        if unexpected:
            raise TaskCRuntimeError(
                f"cache root contains unrecognized entries: {sorted(unexpected)}"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
    return root


def _ensure_cache_subdirectory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise TaskCRuntimeError(f"cache {label} must not be a symbolic link")
    if path.exists():
        if not path.is_dir():
            raise TaskCRuntimeError(f"cache {label} must be a directory")
    else:
        path.mkdir(exist_ok=False)
    return path


def bootstrap_task_c_methods(
    *,
    cache_root: str | Path,
    registry_path: str | Path,
    project_root: str | Path,
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """获取固定版本的官方代码，建立隔离环境，并留下可重复核对的记录。"""

    registry_file = Path(registry_path)
    registry = load_task_c_method_registry(registry_file)
    project = Path(project_root)
    sources = _source_records(registry)
    environments = _environment_records(registry, project)
    identity = _bootstrap_identity(registry_file, sources, environments)
    root = _prepare_cache_root(cache_root)
    identity_path = root / "bootstrap_identity.json"
    if identity_path.exists() or identity_path.is_symlink():
        if _read_json_object(identity_path, "bootstrap identity") != identity:
            raise TaskCRuntimeError(
                "cache identity differs from the fixed registry or environment files"
            )
    else:
        _atomic_create_json(identity_path, identity)

    source_root = _ensure_cache_subdirectory(root / "sources", "sources")
    for source_id, expected in sources.items():
        _ensure_source(source_root / source_id, expected, run_command)

    status_root = _ensure_cache_subdirectory(root / "status", "status")
    for method in registry.methods.values():
        if method.source_kind == "publication_only":
            method_status_root = _ensure_cache_subdirectory(
                status_root / method.method_id,
                f"status/{method.method_id}",
            )
            _write_same_or_new(
                method_status_root / "method_status.json",
                classify_publication_only_method(method),
            )

    environment_listing = _run_checked(
        run_command,
        ["conda", "env", "list", "--json"],
        capture_output=True,
    )
    try:
        existing_environments = _existing_environment_names(
            json.loads(environment_listing.stdout)
        )
    except json.JSONDecodeError as exc:
        raise TaskCRuntimeError("conda environment list is not valid JSON") from exc
    manifest_root = _ensure_cache_subdirectory(
        root / "environment_manifests", "environment_manifests"
    )
    for environment_name, environment_file in environments.items():
        if environment_name in existing_environments:
            _run_checked(
                run_command,
                [
                    "conda",
                    "env",
                    "update",
                    "--name",
                    environment_name,
                    "--file",
                    str(environment_file),
                    "--prune",
                ],
            )
        else:
            _run_checked(
                run_command,
                ["conda", "env", "create", "--file", str(environment_file)],
            )
        package_listing = _run_checked(
            run_command,
            ["conda", "run", "-n", environment_name, "conda", "list", "--json"],
            capture_output=True,
        )
        try:
            packages = _normalized_packages(json.loads(package_listing.stdout))
        except json.JSONDecodeError as exc:
            raise TaskCRuntimeError("conda package list is not valid JSON") from exc
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "environment": environment_name,
            "specification_sha256": _sha256_file(environment_file),
            "packages": packages,
        }
        _atomic_replace_json(
            manifest_root / f"{environment_name}.json",
            manifest,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "assets_and_environments_recorded",
        "source_count": len(sources),
        "environment_count": len(environments),
        "publication_only_count": sum(
            method.source_kind == "publication_only"
            for method in registry.methods.values()
        ),
    }
