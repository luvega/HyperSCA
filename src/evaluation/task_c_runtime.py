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
import tempfile
import threading
import time
from typing import Any
import uuid

import yaml  # type: ignore[import-untyped]

from src.evaluation.task_c_method_registry import (
    TaskCMethodRegistry,
    TaskCMethodSpec,
    load_task_c_method_registry,
)


SCHEMA_VERSION = "1.0"
GNU_TIME = Path("/usr/bin/time")
DEFAULT_MAXIMUM_OUTPUT_BYTES = 64 * 1024
HARD_MAXIMUM_OUTPUT_BYTES = 1024 * 1024
MAXIMUM_BOOTSTRAP_RECORD_BYTES = 2 * 1024 * 1024
MAXIMUM_BOOTSTRAP_COMMAND_BYTES = 8 * 1024 * 1024
MAXIMUM_BOOTSTRAP_INPUT_BYTES = 1024 * 1024
BOOTSTRAP_COMMAND_TIMEOUT_SECONDS = 600
MAXIMUM_CAUSALBENCH_SOURCE_BLOB_BYTES = 512 * 1024 * 1024
CAUSALBENCH_DASK_PINS = frozenset(
    {"dask[complete]==2024.4.1", "distributed==2024.4.1"}
)
CAUSALBENCH_EXPORT_REQUIRED_FILES = {
    "causalscbench/__init__.py",
    "causalscbench/data_access/__init__.py",
    "causalscbench/data_access/create_dataset.py",
    "causalscbench/data_access/create_evaluation_datasets.py",
    "causalscbench/data_access/data/K562_ChipSeq.csv",
    "causalscbench/data_access/data/Hep_G2_ChipSeq.csv",
}
_ALLOWED_CACHE_ENTRIES = {
    "bootstrap_identity.json",
    "bootstrap_manifest.json",
    "bootstrap_status.json",
    "environment_manifests",
    "sources",
    "status",
}
_RESOURCE_LIMIT_CODES = {137, 152}
_RESOURCE_LIMIT_SIGNALS = {signal.SIGKILL, signal.SIGXCPU}
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s\"']+)")
_PLAIN_OPTION = re.compile(r"(?:--[A-Za-z][A-Za-z0-9_-]*|-[A-Za-z])")
_OPTION_WITH_VALUE = re.compile(r"(-{1,2}[A-Za-z][A-Za-z0-9_-]*)=(.+)", re.DOTALL)
_SHORT_OPTION_WITH_VALUE = re.compile(r"(-[A-Za-z])(.+)", re.DOTALL)
_NEGATIVE_NUMBER = re.compile(r"-(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_COMMAND_ARGUMENT_REDACTION = "<command-argument>"


def _has_private_command_shape(value: str) -> bool:
    return (
        "/" in value or "\\" in value or any(character.isspace() for character in value)
    )


def _redact_command_values(text: str, command: Sequence[str]) -> str:
    replacements: dict[str, str] = {}
    for argument in command:
        if _PLAIN_OPTION.fullmatch(argument) or _NEGATIVE_NUMBER.fullmatch(argument):
            continue
        equals_option = _OPTION_WITH_VALUE.fullmatch(argument)
        attached_option = _SHORT_OPTION_WITH_VALUE.fullmatch(argument)
        if equals_option is not None:
            option, value = equals_option.groups()
            replacements.setdefault(
                argument,
                f"{option}={_COMMAND_ARGUMENT_REDACTION}",
            )
            replacements.setdefault(value, _COMMAND_ARGUMENT_REDACTION)
        elif attached_option is not None:
            option, value = attached_option.groups()
            replacements.setdefault(
                argument,
                f"{option}{_COMMAND_ARGUMENT_REDACTION}",
            )
            replacements.setdefault(value, _COMMAND_ARGUMENT_REDACTION)
        elif _has_private_command_shape(argument):
            replacements.setdefault(argument, _COMMAND_ARGUMENT_REDACTION)
    if not replacements:
        return text
    alternatives = sorted(replacements, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(value) for value in alternatives))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


class TaskCRuntimeError(ValueError):
    """运行位置、命令或官方资产不符合固定比较规则。"""


class _TailBuffer:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.content = bytearray()
        self.total_bytes = 0
        self._evidence_window = b""
        self.saw_invalid_output = False
        self.saw_resource_limit = False

    def add(self, chunk: bytes) -> None:
        evidence = (self._evidence_window + chunk).lower()
        if b"failed_invalid_output" in evidence:
            self.saw_invalid_output = True
        if b"out of memory" in evidence or b"memoryerror" in evidence:
            self.saw_resource_limit = True
        self._evidence_window = evidence[-64:]
        self.total_bytes += len(chunk)
        self.content.extend(chunk)
        if len(self.content) > self.maximum_bytes:
            del self.content[: len(self.content) - self.maximum_bytes]

    @property
    def was_truncated(self) -> bool:
        return self.total_bytes > self.maximum_bytes

    def text(self, command: Sequence[str] = ()) -> str:
        decoded = _redact_command_values(self.raw_text(), command)
        return _PRIVATE_PATH.sub(
            "<absolute-path>",
            decoded,
        )

    def raw_text(self) -> str:
        return bytes(self.content).decode("utf-8", errors="replace")


class _DuplicateJsonKey(ValueError):
    pass


class _UniqueSafeYamlLoader(yaml.SafeLoader):
    pass


def _construct_unique_yaml_mapping(
    loader: _UniqueSafeYamlLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise TaskCRuntimeError(
                "environment YAML contains an invalid mapping key"
            ) from exc
        if duplicate:
            raise TaskCRuntimeError(f"environment YAML contains duplicate key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


class _FileSnapshot:
    def __init__(
        self,
        path: Path,
        payload: bytes,
        identity: tuple[int, int, int, int, int],
    ) -> None:
        self.path = path
        self.payload = payload
        self.identity = identity

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.payload)

    @property
    def mode(self) -> int:
        return self.identity[4]


class _CommandResult:
    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RunDirectory:
    """已打开的结果目录；后续写入不再解析可被替换的路径。"""

    def __init__(
        self,
        path: Path,
        descriptor: int,
        identity: tuple[int, int],
        parent_descriptor: int,
        name: str,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self.parent_descriptor = parent_descriptor
        self.name = name

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_descriptor)

    def path_still_names_open_directory(self) -> bool:
        try:
            metadata = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            path_metadata = self.path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == self.identity
            and stat.S_ISDIR(path_metadata.st_mode)
            and (path_metadata.st_dev, path_metadata.st_ino) == self.identity
        )

    def unlink(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self.descriptor)
        except FileNotFoundError:
            pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_deep_json(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > 32:
                raise TaskCRuntimeError("JSON record is too deeply nested")
        elif character in "]}":
            depth -= 1


def _strict_json_loads(payload: bytes | str, label: str) -> object:
    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
        text = payload
    else:
        encoded = payload
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TaskCRuntimeError(f"{label} is not valid UTF-8") from exc
    if len(encoded) > MAXIMUM_BOOTSTRAP_RECORD_BYTES:
        raise TaskCRuntimeError(f"{label} is unusually large")
    _reject_deep_json(text)
    try:
        return json.loads(text, object_pairs_hook=_json_object_without_duplicates)
    except _DuplicateJsonKey as exc:
        raise TaskCRuntimeError(str(exc)) from exc
    except (json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise TaskCRuntimeError(f"{label} is not valid JSON") from exc


def _snapshot_regular_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int = MAXIMUM_BOOTSTRAP_INPUT_BYTES,
    allow_empty: bool = False,
) -> _FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TaskCRuntimeError(f"{label} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TaskCRuntimeError(f"{label} must be one regular, unlinked file")
        if (before.st_size == 0 and not allow_empty) or before.st_size > maximum_bytes:
            raise TaskCRuntimeError(f"{label} is empty or unusually large")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise TaskCRuntimeError(f"{label} changed while it was read")
    return _FileSnapshot(path, payload, before_identity)


def _verify_snapshot_unchanged(snapshot: _FileSnapshot, label: str) -> None:
    current = _snapshot_regular_file(snapshot.path, label)
    if current.identity != snapshot.identity or current.payload != snapshot.payload:
        raise TaskCRuntimeError(f"{label} changed during preparation")


def _write_snapshot(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _validate_registry_snapshot(snapshot: _FileSnapshot) -> TaskCMethodRegistry:
    # TemporaryFile is anonymous on supported POSIX systems, so validation uses
    # the immutable bytes already read without leaving a named preflight file.
    with tempfile.TemporaryFile(mode="w+b") as registry_file:
        registry_file.write(snapshot.payload)
        registry_file.flush()
        registry_file.seek(0)
        return load_task_c_method_registry(
            Path(f"/proc/self/fd/{registry_file.fileno()}")
        )


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
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _atomic_json_at(
    directory: _RunDirectory,
    name: str,
    payload: Mapping[str, object],
    *,
    replace: bool,
) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory.descriptor,
        )
        content = _json_bytes(payload)
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if replace:
            os.rename(
                temporary,
                name,
                src_dir_fd=directory.descriptor,
                dst_dir_fd=directory.descriptor,
            )
        else:
            try:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=directory.descriptor,
                    dst_dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise TaskCRuntimeError(
                    f"refusing to overwrite existing record {name}"
                ) from exc
            directory.unlink(temporary)
        os.fsync(directory.descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        directory.unlink(temporary)


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
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


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


def _prepare_empty_output_directory(path: str | Path) -> _RunDirectory:
    destination = Path(os.path.abspath(Path(path).expanduser()))
    parts = destination.parts
    if len(parts) < 2 or destination == Path(destination.anchor):
        raise TaskCRuntimeError("output directory must not be the filesystem root")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current_descriptor = os.open(destination.anchor, directory_flags)
    except OSError as exc:
        raise TaskCRuntimeError(
            "output directory root cannot be opened safely"
        ) from exc
    try:
        for component in parts[1:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=current_descriptor)
                os.fsync(current_descriptor)
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_descriptor,
                )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        name = parts[-1]
        try:
            os.mkdir(name, mode=0o755, dir_fd=current_descriptor)
            os.fsync(current_descriptor)
        except FileExistsError:
            pass
        descriptor = os.open(name, directory_flags, dir_fd=current_descriptor)
    except OSError as exc:
        os.close(current_descriptor)
        raise TaskCRuntimeError(
            "output directory cannot be created or opened safely"
        ) from exc
    metadata = os.fstat(descriptor)
    opened = _RunDirectory(
        destination,
        descriptor,
        (metadata.st_dev, metadata.st_ino),
        current_descriptor,
        name,
    )
    if not opened.path_still_names_open_directory():
        opened.close()
        raise TaskCRuntimeError("output directory changed while it was opened")
    try:
        entries = os.listdir(opened.descriptor)
    except OSError as exc:
        opened.close()
        raise TaskCRuntimeError("output directory cannot be inspected safely") from exc
    if entries:
        opened.close()
        raise TaskCRuntimeError(
            "output directory must be new or an existing empty directory"
        )
    return opened


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
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if "Maximum resident set size" not in line:
            continue
        value_text = line.rsplit(":", 1)[-1].strip()
        if re.fullmatch(r"[0-9]{1,20}", value_text) is None:
            return None
        try:
            value = int(value_text)
        except (ValueError, OverflowError):
            return None
        return value if value <= 2**63 - 1 else None
    return None


def _parse_maximum_resident_kib_at(
    directory: _RunDirectory,
    name: str,
) -> tuple[int | None, bool]:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory.descriptor,
        )
    except OSError:
        return None, False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return None, False
        maximum = 64 * 1024
        payload = os.read(descriptor, maximum + 1)
        if len(payload) > maximum:
            return None, False
    finally:
        os.close(descriptor)
    text = payload.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if "Maximum resident set size" not in line:
            continue
        value_text = line.rsplit(":", 1)[-1].strip()
        if re.fullmatch(r"[0-9]{1,20}", value_text) is None:
            return None, False
        try:
            value = int(value_text)
        except (ValueError, OverflowError):
            return None, False
        if value > 2**63 - 1:
            return None, False
        return value, True
    return None, False


def _executable_is_missing(executable: str) -> bool:
    if "/" in executable:
        candidate = Path(executable)
        return not candidate.is_file() or not os.access(candidate, os.X_OK)
    return shutil.which(executable) is None


def _classify_return(return_code: int, stderr: _TailBuffer) -> str:
    if stderr.saw_invalid_output:
        return "failed_invalid_output"
    if return_code == 0:
        return "completed_raw_inference"
    if return_code == 125:
        return "failed_runtime_unavailable"
    if return_code in {126, 127}:
        return "failed_launch"
    signal_number = -return_code if return_code < 0 else None
    if (
        return_code in _RESOURCE_LIMIT_CODES
        or signal_number in _RESOURCE_LIMIT_SIGNALS
        or stderr.saw_resource_limit
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
    resource_temporary = f".resource_usage.{uuid.uuid4().hex}.tmp"
    stdout = _TailBuffer(maximum_output_bytes)
    stderr = _TailBuffer(maximum_output_bytes)
    started = time.monotonic()
    return_code: int | None = None
    status = "official_code_incompatible"
    resource_meter = "gnu_time_v"
    process: subprocess.Popen[bytes] | None = None
    resource_descriptor: int | None = None
    threads: list[threading.Thread] = []
    pending_exception: BaseException | None = None
    missing_method = _executable_is_missing(normalized_command[0])
    missing_meter = not GNU_TIME.is_file() or not os.access(GNU_TIME, os.X_OK)
    try:
        if missing_method:
            stderr.add(b"registered method executable could not be started")
            status = "failed_launch"
            resource_meter = "unavailable"
        elif missing_meter:
            status = "failed_runtime_unavailable"
            resource_meter = "unavailable"
            stderr.add(
                b"GNU time resource meter is unavailable; method was not started"
            )
        else:
            resource_descriptor = os.open(
                resource_temporary,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=destination.descriptor,
            )
            timed_command = [
                str(GNU_TIME),
                "-v",
                "-o",
                f"/proc/self/fd/{resource_descriptor}",
                "--",
                *normalized_command,
            ]
            environment = dict(os.environ)
            environment["LC_ALL"] = "C"
            try:
                process = subprocess.Popen(
                    timed_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    bufsize=0,
                    env=environment,
                    pass_fds=(resource_descriptor,),
                )
            except OSError:
                status = "failed_launch"
                resource_meter = "unavailable"
                stderr.add(b"registered method process could not be started")
            if process is None:
                return_code = None
            else:
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
                    status = _classify_return(return_code, stderr)
    except BaseException as exc:
        if process is not None:
            _terminate_process_group(process)
        for thread in threads:
            thread.join(timeout=1)
        status = "failed_runtime_unavailable"
        resource_meter = "unavailable"
        stderr.add(b"runtime supervision was interrupted")
        pending_exception = exc

    if resource_descriptor is not None:
        os.close(resource_descriptor)
        resource_descriptor = None

    elapsed = float(time.monotonic() - started)
    if not math.isfinite(elapsed) or elapsed < 0:
        destination.unlink(resource_temporary)
        destination.close()
        raise TaskCRuntimeError("elapsed time could not be measured safely")
    if return_code is not None and return_code < 0:
        terminating_signal: int | None = -return_code
    else:
        terminating_signal = None
    maximum_resident_kib, resource_report_valid = _parse_maximum_resident_kib_at(
        destination,
        resource_temporary,
    )
    if not resource_report_valid:
        resource_meter = "unavailable"
        if process is not None and status == "completed_raw_inference":
            status = "failed_runtime_unavailable"
    if not destination.path_still_names_open_directory():
        status = "failed_runtime_unavailable"
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "return_code": return_code,
        "terminating_signal": terminating_signal,
        "elapsed_seconds": elapsed,
        "stdout_tail": stdout.text(normalized_command),
        "stderr_tail": stderr.text(normalized_command),
        "output_was_truncated": {
            "stdout": stdout.was_truncated,
            "stderr": stderr.was_truncated,
        },
        "command_trace": _command_trace(normalized_command),
    }
    resource_values: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "elapsed_seconds": elapsed,
        "maximum_resident_kib": maximum_resident_kib,
        "resource_meter": resource_meter,
    }
    destination.unlink(resource_temporary)
    created: list[str] = []
    try:
        for name, record in (
            ("method_status.json", payload),
            ("resource_usage.json", resource_values),
        ):
            _atomic_json_at(destination, name, record, replace=False)
            created.append(name)
        if not destination.path_still_names_open_directory() and payload["status"] != (
            "failed_runtime_unavailable"
        ):
            payload["status"] = "failed_runtime_unavailable"
            _atomic_json_at(
                destination,
                "method_status.json",
                payload,
                replace=True,
            )
    except BaseException:
        for name in created:
            destination.unlink(name)
        raise
    finally:
        destination.unlink(resource_temporary)
        destination.close()
    if pending_exception is not None:
        raise pending_exception
    return payload


def _validate_environment_snapshot(
    snapshot: _FileSnapshot,
    expected_name: str,
    causalbench_pin: str,
    *,
    require_arboreto_dask: bool,
) -> None:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskCRuntimeError(
            f"environment file {snapshot.path.name} is not valid UTF-8"
        ) from exc
    try:
        documents = list(yaml.load_all(text, Loader=_UniqueSafeYamlLoader))
    except (
        yaml.YAMLError,
        TaskCRuntimeError,
        TypeError,
        ValueError,
        RecursionError,
        OverflowError,
    ) as exc:
        if isinstance(exc, TaskCRuntimeError):
            raise
        raise TaskCRuntimeError(
            f"environment file {snapshot.path.name} is not safe, valid YAML"
        ) from exc
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise TaskCRuntimeError(
            f"environment file {snapshot.path.name} must contain one YAML mapping"
        )
    document = documents[0]
    if "name" in document and document["name"] != expected_name:
        raise TaskCRuntimeError(
            f"environment name in {snapshot.path.name} does not match the method registry"
        )
    if set(document) != {"name", "channels", "dependencies"}:
        raise TaskCRuntimeError(
            f"environment file {snapshot.path.name} fields must be exactly name, channels, dependencies"
        )
    channels = document["channels"]
    if (
        not isinstance(channels, list)
        or not channels
        or any(type(channel) is not str or not channel for channel in channels)
        or len(set(channels)) != len(channels)
    ):
        raise TaskCRuntimeError(
            f"environment {expected_name} channels must be unique names"
        )
    dependencies = document["dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise TaskCRuntimeError(
            f"environment {expected_name} dependencies must be a non-empty list"
        )
    pip_sections: list[list[object]] = []
    for dependency in dependencies:
        if type(dependency) is str and dependency:
            continue
        if (
            isinstance(dependency, dict)
            and set(dependency) == {"pip"}
            and isinstance(dependency["pip"], list)
        ):
            pip_sections.append(dependency["pip"])
            continue
        raise TaskCRuntimeError(
            f"environment {expected_name} contains an unsupported dependency structure"
        )
    if len(pip_sections) != 1:
        raise TaskCRuntimeError(
            f"environment {expected_name} must contain one pip dependency section"
        )
    pip_dependencies = pip_sections[0]
    if any(
        type(dependency) is not str or not dependency for dependency in pip_dependencies
    ):
        raise TaskCRuntimeError(
            f"environment {expected_name} pip dependencies must be package strings"
        )
    pip_strings = [str(dependency) for dependency in pip_dependencies]
    if len(set(pip_strings)) != len(pip_strings):
        raise TaskCRuntimeError(
            f"environment {expected_name} contains duplicate pip dependencies"
        )
    vcs_dependencies = [
        dependency for dependency in pip_strings if "git+" in dependency.casefold()
    ]
    if vcs_dependencies != [causalbench_pin]:
        raise TaskCRuntimeError(
            f"environment {expected_name} does not contain the fixed CausalBench source"
        )
    if require_arboreto_dask:
        dask_dependencies = {
            dependency
            for dependency in pip_strings
            if re.fullmatch(
                r"(?:dask(?:\[[^\]]+\])?|distributed)(?:[<>=!~@ ;].*)?",
                dependency,
                flags=re.IGNORECASE,
            )
        }
        if dask_dependencies != CAUSALBENCH_DASK_PINS:
            raise TaskCRuntimeError(
                f"environment {expected_name} does not contain exactly the fixed Arboreto Dask and distributed versions"
            )


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
) -> dict[str, _FileSnapshot]:
    records = {
        registry.causalbench["environment"]: project_root
        / "envs/task_c/causalbench.yml"
    }
    for method in registry.methods.values():
        if method.source_kind == "git":
            assert method.environment is not None
            records[method.environment] = project_root / "envs/task_c/psgrn.yml"
    causalbench_pin = (
        f"git+{registry.causalbench['repository']}@{registry.causalbench['commit']}"
    )
    snapshots: dict[str, _FileSnapshot] = {}
    for expected_name, path in records.items():
        snapshot = _snapshot_regular_file(path, f"environment file {path.name}")
        _validate_environment_snapshot(
            snapshot,
            expected_name,
            causalbench_pin,
            require_arboreto_dask=path.name == "causalbench.yml",
        )
        snapshots[expected_name] = snapshot
    return snapshots


def _bootstrap_identity(
    registry_snapshot: _FileSnapshot,
    sources: Mapping[str, Mapping[str, str]],
    environments: Mapping[str, _FileSnapshot],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": registry_snapshot.sha256,
        "sources": {key: dict(value) for key, value in sources.items()},
        "environment_files": {
            name: snapshot.sha256 for name, snapshot in environments.items()
        },
    }


def _run_bounded_command(command: list[str], timeout: float) -> _CommandResult:
    stdout = _TailBuffer(MAXIMUM_BOOTSTRAP_COMMAND_BYTES)
    stderr = _TailBuffer(MAXIMUM_BOOTSTRAP_COMMAND_BYTES)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )
    except OSError as exc:
        raise TaskCRuntimeError(
            f"official asset command could not start: {Path(command[0]).name}"
        ) from exc
    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(
            target=_drain_stream, args=(process.stdout, stdout), daemon=True
        ),
        threading.Thread(
            target=_drain_stream, args=(process.stderr, stderr), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise TaskCRuntimeError(
            f"official asset command timed out: {Path(command[0]).name}"
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
    _terminate_process_group(process)
    if stdout.was_truncated or stderr.was_truncated:
        raise TaskCRuntimeError(
            f"official asset command output was unusually large: {Path(command[0]).name}"
        )
    if return_code != 0:
        raise TaskCRuntimeError(
            f"official asset command failed: {Path(command[0]).name}"
        )
    return _CommandResult(stdout.raw_text(), stderr.raw_text(), return_code)


def _run_checked(
    runner: Callable[..., Any],
    command: list[str],
    *,
    capture_output: bool = False,
) -> Any:
    if runner is subprocess.run:
        return _run_bounded_command(
            command,
            float(BOOTSTRAP_COMMAND_TIMEOUT_SECONDS),
        )
    kwargs: dict[str, object] = {
        "check": True,
        "timeout": BOOTSTRAP_COMMAND_TIMEOUT_SECONDS,
    }
    if capture_output:
        kwargs.update({"capture_output": True, "text": True})
    try:
        return runner(command, **kwargs)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TaskCRuntimeError(
            f"official asset command failed: {Path(command[0]).name}"
        ) from exc


def _git_blob_id(payload: bytes, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise TaskCRuntimeError("official source uses an unsupported Git object format")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _parse_fixed_commit_tree(text: str) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    for raw_record in text.split("\x00"):
        if not raw_record:
            continue
        metadata, separator, relative = raw_record.partition("\t")
        parts = metadata.split(" ")
        if (
            separator != "\t"
            or len(parts) != 3
            or parts[0] not in {"100644", "100755"}
            or parts[1] != "blob"
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", parts[2]) is None
            or not relative
            or relative.startswith("/")
            or relative in records
        ):
            raise TaskCRuntimeError("fixed commit contains an unsupported Git tree entry")
        records[relative] = (parts[0], parts[2])
    if not records:
        raise TaskCRuntimeError("fixed commit does not contain ordinary source files")
    return records


def _validate_source_checkout(
    source: Path,
    expected: Mapping[str, str],
    runner: Callable[..., Any],
) -> str:
    git_directory = source / ".git"
    if (
        source.is_symlink()
        or not source.is_dir()
        or git_directory.is_symlink()
        or not git_directory.is_dir()
    ):
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
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        capture_output=True,
    ).stdout.strip()
    if repository != expected["repository"]:
        raise TaskCRuntimeError(
            "official source repository does not match the registry"
        )
    if revision != expected["commit"]:
        raise TaskCRuntimeError("official source is not at the fixed commit")
    if changes:
        raise TaskCRuntimeError(
            "official source contains changed, untracked, or ignored files"
        )
    replacements = _run_checked(
        runner,
        ["git", "-C", str(source), "replace", "-l"],
        capture_output=True,
    ).stdout.strip()
    if replacements:
        raise TaskCRuntimeError("official source uses an unreviewed Git replacement")
    graft_path_text = _run_checked(
        runner,
        ["git", "-C", str(source), "rev-parse", "--git-path", "info/grafts"],
        capture_output=True,
    ).stdout.strip()
    graft_path = Path(graft_path_text)
    if not graft_path.is_absolute():
        graft_path = source / graft_path
    if graft_path.exists() or graft_path.is_symlink():
        graft = _snapshot_regular_file(graft_path, "Git graft file")
        if graft.payload.strip():
            raise TaskCRuntimeError("official source uses an unreviewed Git graft")
    tracked_text = _run_checked(
        runner,
        ["git", "-C", str(source), "ls-files", "-z", "--cached"],
        capture_output=True,
    ).stdout
    committed_text = _run_checked(
        runner,
        ["git", "-C", str(source), "ls-tree", "-r", "-z", "HEAD"],
        capture_output=True,
    ).stdout
    tracked = {item for item in tracked_text.split("\x00") if item}
    committed_tree = _parse_fixed_commit_tree(committed_text)
    committed = set(committed_tree)
    if tracked != committed or not tracked:
        raise TaskCRuntimeError(
            "official source tracked files do not match the fixed commit"
        )
    object_format = _run_checked(
        runner,
        ["git", "-C", str(source), "rev-parse", "--show-object-format"],
        capture_output=True,
    ).stdout.strip()
    if object_format not in {"sha1", "sha256"}:
        raise TaskCRuntimeError("official source uses an unsupported Git object format")
    if any(
        len(blob_id) != (40 if object_format == "sha1" else 64)
        for _, blob_id in committed_tree.values()
    ):
        raise TaskCRuntimeError("fixed commit blob identity has the wrong length")
    _run_checked(
        runner,
        ["git", "-C", str(source), "diff", "--no-ext-diff", "--quiet", "HEAD", "--"],
    )
    worktree_files: set[str] = set()
    digest = hashlib.sha256()
    for current_root, directory_names, file_names in os.walk(
        source,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        if current == source:
            directory_names[:] = [name for name in directory_names if name != ".git"]
        directory_names.sort()
        file_names.sort()
        for directory_name in tuple(directory_names):
            directory = current / directory_name
            if directory.is_symlink():
                raise TaskCRuntimeError("official source contains a symbolic link")
        for file_name in file_names:
            path = current / file_name
            if path.is_symlink():
                raise TaskCRuntimeError("official source contains a symbolic link")
            relative = path.relative_to(source).as_posix()
            if relative not in tracked:
                raise TaskCRuntimeError(
                    "official source contains an untracked worktree file"
                )
            snapshot = _snapshot_regular_file(
                path,
                f"official source file {relative}",
                maximum_bytes=512 * 1024 * 1024,
                allow_empty=(
                    committed_tree[relative][1]
                    == _git_blob_id(b"", object_format)
                ),
            )
            expected_mode, expected_blob_id = committed_tree[relative]
            observed_executable = bool(snapshot.mode & 0o111)
            if observed_executable != (expected_mode == "100755"):
                raise TaskCRuntimeError(
                    f"official source file {relative} mode differs from the fixed commit"
                )
            if _git_blob_id(snapshot.payload, object_format) != expected_blob_id:
                raise TaskCRuntimeError(
                    f"official source file {relative} bytes differ from the fixed commit blob"
                )
            worktree_files.add(relative)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(expected_mode.encode("ascii"))
            digest.update(b"\x00")
            digest.update(expected_blob_id.encode("ascii"))
            digest.update(b"\x00")
            digest.update(bytes.fromhex(snapshot.sha256))
    if worktree_files != tracked:
        raise TaskCRuntimeError(
            "official source contains untracked or missing worktree files"
        )
    return digest.hexdigest()


def _ensure_source(
    source: Path,
    expected: Mapping[str, str],
    runner: Callable[..., Any],
) -> str:
    if source.exists() or source.is_symlink():
        return _validate_source_checkout(source, expected, runner)
    try:
        _run_checked(
            runner,
            ["git", "clone", expected["repository"], str(source)],
        )
        _run_checked(
            runner,
            ["git", "-C", str(source), "checkout", "--detach", expected["commit"]],
        )
        return _validate_source_checkout(source, expected, runner)
    except BaseException:
        if source.exists() and not source.is_symlink():
            shutil.rmtree(source)
        raise


def _canonical_bootstrap_json(path: Path, label: str) -> tuple[_FileSnapshot, dict[str, object]]:
    snapshot = _snapshot_regular_file(
        path,
        label,
        maximum_bytes=MAXIMUM_BOOTSTRAP_RECORD_BYTES,
    )
    payload = _strict_json_loads(snapshot.payload, label)
    if not isinstance(payload, dict) or snapshot.payload != _json_bytes(payload):
        raise TaskCRuntimeError(f"{label} is not one canonical JSON object")
    return snapshot, payload


def _require_sha256_hex(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TaskCRuntimeError(f"{label} is not a SHA-256 identity")
    return value


def _validate_bootstrap_file_hashes(
    root: Path,
    manifest: Mapping[str, object],
) -> list[_FileSnapshot]:
    snapshots: list[_FileSnapshot] = []
    for field, directory in (
        ("environment_manifests", root / "environment_manifests"),
        ("publication_statuses", root / "status"),
    ):
        records = manifest.get(field)
        if not isinstance(records, dict):
            raise TaskCRuntimeError(f"bootstrap manifest {field} is invalid")
        expected_paths: set[Path] = set()
        for relative, expected_sha in records.items():
            if (
                type(relative) is not str
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise TaskCRuntimeError(
                    f"bootstrap manifest {field} contains an unsafe path"
                )
            _require_sha256_hex(expected_sha, f"bootstrap manifest {field} hash")
            candidate = directory / relative
            expected_paths.add(candidate)
            snapshot = _snapshot_regular_file(
                candidate,
                f"bootstrap asset {field}/{relative}",
                maximum_bytes=MAXIMUM_BOOTSTRAP_RECORD_BYTES,
            )
            _strict_json_loads(snapshot.payload, f"bootstrap asset {field}/{relative}")
            if snapshot.sha256 != expected_sha:
                raise TaskCRuntimeError(
                    f"bootstrap asset {field}/{relative} hash changed"
                )
            snapshots.append(snapshot)
        actual_paths = {
            path
            for path in directory.rglob("*.json")
            if path.is_file() and not path.is_symlink()
        }
        if actual_paths != expected_paths:
            raise TaskCRuntimeError(
                f"bootstrap asset {field} file inventory is incomplete"
            )
    return snapshots


def _read_fixed_git_blob(source: Path, blob_id: str, object_format: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), "cat-file", "blob", blob_id],
            check=True,
            capture_output=True,
            timeout=BOOTSTRAP_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TaskCRuntimeError("fixed CausalBench Git blob could not be read") from exc
    payload = completed.stdout
    if (
        not isinstance(payload, bytes)
        or len(payload) > MAXIMUM_CAUSALBENCH_SOURCE_BLOB_BYTES
        or _git_blob_id(payload, object_format) != blob_id
    ):
        raise TaskCRuntimeError("fixed CausalBench Git blob bytes are invalid")
    if len(completed.stderr) > MAXIMUM_BOOTSTRAP_COMMAND_BYTES:
        raise TaskCRuntimeError("fixed CausalBench Git command output is too large")
    return payload


def _causalbench_commit_inventory(
    source: Path,
    commit: str,
) -> tuple[list[tuple[str, str, bytes]], str]:
    tree_text = _run_checked(
        subprocess.run,
        ["git", "-C", str(source), "ls-tree", "-r", "-z", commit],
        capture_output=True,
    ).stdout
    committed_tree = _parse_fixed_commit_tree(tree_text)
    selected = {
        relative: record
        for relative, record in committed_tree.items()
        if relative.startswith("causalscbench/")
    }
    if not CAUSALBENCH_EXPORT_REQUIRED_FILES <= set(selected):
        raise TaskCRuntimeError(
            "fixed CausalBench commit lacks required package code or data resources"
        )
    object_format = _run_checked(
        subprocess.run,
        ["git", "-C", str(source), "rev-parse", "--show-object-format"],
        capture_output=True,
    ).stdout.strip()
    if object_format not in {"sha1", "sha256"}:
        raise TaskCRuntimeError("fixed CausalBench source uses an invalid object format")
    inventory: list[tuple[str, str, bytes]] = []
    digest = hashlib.sha256()
    for relative in sorted(selected):
        mode, blob_id = selected[relative]
        payload = _read_fixed_git_blob(source, blob_id, object_format)
        payload_sha = _sha256_bytes(payload)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(mode.encode("ascii"))
        digest.update(b"\x00")
        digest.update(payload_sha.encode("ascii"))
        digest.update(b"\x00")
        inventory.append((relative, mode, payload))
    return inventory, digest.hexdigest()


def _validated_causalbench_export_assets(
    method_assets_root: str | Path,
    *,
    repository: str,
    commit: str,
) -> tuple[dict[str, str], list[tuple[str, str, bytes]]]:
    root = Path(os.path.abspath(Path(method_assets_root).expanduser()))
    _reject_symlink_components(root, "method assets root")
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != _ALLOWED_CACHE_ENTRIES:
        raise TaskCRuntimeError("method assets root is not one complete bootstrap cache")
    manifest_snapshot, manifest = _canonical_bootstrap_json(
        root / "bootstrap_manifest.json", "bootstrap manifest"
    )
    status_snapshot, status = _canonical_bootstrap_json(
        root / "bootstrap_status.json", "bootstrap status"
    )
    identity_snapshot, identity = _canonical_bootstrap_json(
        root / "bootstrap_identity.json", "bootstrap identity"
    )
    if set(manifest) != {
        "schema_version",
        "bootstrap_identity_sha256",
        "sources",
        "environment_manifests",
        "publication_statuses",
    } or manifest.get("schema_version") != SCHEMA_VERSION:
        raise TaskCRuntimeError("bootstrap manifest schema is invalid")
    if status != {
        "schema_version": SCHEMA_VERSION,
        "status": "assets_and_environments_recorded",
        "bootstrap_manifest_sha256": manifest_snapshot.sha256,
    }:
        raise TaskCRuntimeError("bootstrap status does not bind the manifest")
    if manifest.get("bootstrap_identity_sha256") != identity_snapshot.sha256:
        raise TaskCRuntimeError("bootstrap manifest does not bind bootstrap identity")
    if set(identity) != {
        "schema_version",
        "registry_sha256",
        "sources",
        "environment_files",
    } or identity.get("schema_version") != SCHEMA_VERSION:
        raise TaskCRuntimeError("bootstrap identity schema is invalid")
    _require_sha256_hex(identity.get("registry_sha256"), "bootstrap registry hash")
    sources = manifest.get("sources")
    identity_sources = identity.get("sources")
    if not isinstance(sources, dict) or not isinstance(identity_sources, dict):
        raise TaskCRuntimeError("bootstrap source records are invalid")
    causalbench = sources.get("causalbench")
    expected_source = {"repository": repository, "commit": commit}
    if (
        not isinstance(causalbench, dict)
        or set(causalbench) != {"repository", "commit", "worktree_sha256"}
        or {key: causalbench.get(key) for key in expected_source} != expected_source
        or identity_sources.get("causalbench") != expected_source
    ):
        raise TaskCRuntimeError(
            "bootstrap manifest does not bind the fixed CausalBench source"
        )
    expected_worktree = _require_sha256_hex(
        causalbench.get("worktree_sha256"), "CausalBench worktree hash"
    )
    extra_snapshots = _validate_bootstrap_file_hashes(root, manifest)
    source = root / "sources/causalbench"
    observed_worktree = _validate_source_checkout(
        source,
        expected_source,
        subprocess.run,
    )
    if observed_worktree != expected_worktree:
        raise TaskCRuntimeError("CausalBench worktree hash differs from bootstrap manifest")
    inventory, inventory_sha = _causalbench_commit_inventory(source, commit)
    for snapshot, label in (
        (manifest_snapshot, "bootstrap manifest"),
        (status_snapshot, "bootstrap status"),
        (identity_snapshot, "bootstrap identity"),
    ):
        _verify_snapshot_unchanged(snapshot, label)
    for snapshot in extra_snapshots:
        _verify_snapshot_unchanged(snapshot, f"bootstrap asset {snapshot.path.name}")
    evidence = {
        "repository": repository,
        "commit": commit,
        "bootstrap_manifest_sha256": f"sha256:{manifest_snapshot.sha256}",
        "source_worktree_sha256": f"sha256:{observed_worktree}",
        "private_source_inventory_sha256": f"sha256:{inventory_sha}",
    }
    return evidence, inventory


def validate_causalbench_export_assets(
    method_assets_root: str | Path,
    *,
    repository: str,
    commit: str,
) -> dict[str, str]:
    """Strictly validate the bootstrapped CausalBench source for export."""

    evidence, _ = _validated_causalbench_export_assets(
        method_assets_root,
        repository=repository,
        commit=commit,
    )
    return evidence


def _private_source_inventory_sha256(root: Path) -> str:
    files: list[tuple[str, str, bytes]] = []
    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        current_metadata = current.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(current_metadata.st_mode)
            or stat.S_IMODE(current_metadata.st_mode) != 0o500
        ):
            raise TaskCRuntimeError(
                "private CausalBench snapshot directories must remain read-only"
            )
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            if (current / directory_name).is_symlink():
                raise TaskCRuntimeError("private CausalBench snapshot contains a symlink")
        for file_name in file_names:
            path = current / file_name
            if path.is_symlink():
                raise TaskCRuntimeError("private CausalBench snapshot contains a symlink")
            relative = path.relative_to(root).as_posix()
            snapshot = _snapshot_regular_file(
                path,
                f"private CausalBench source {relative}",
                maximum_bytes=MAXIMUM_CAUSALBENCH_SOURCE_BLOB_BYTES,
                allow_empty=True,
            )
            mode = "100755" if snapshot.mode & 0o111 else "100644"
            required_permissions = 0o500 if mode == "100755" else 0o400
            if stat.S_IMODE(snapshot.mode) != required_permissions:
                raise TaskCRuntimeError(
                    "private CausalBench snapshot files must remain read-only"
                )
            files.append((relative, mode, snapshot.payload))
    if not CAUSALBENCH_EXPORT_REQUIRED_FILES <= {record[0] for record in files}:
        raise TaskCRuntimeError("private CausalBench snapshot inventory is incomplete")
    digest = hashlib.sha256()
    for relative, mode, payload in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(mode.encode("ascii"))
        digest.update(b"\x00")
        digest.update(_sha256_bytes(payload).encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def materialize_causalbench_source_snapshot(
    method_assets_root: str | Path,
    destination: str | Path,
    *,
    repository: str,
    commit: str,
    expected_evidence: Mapping[str, str],
) -> dict[str, str]:
    """Build a private read-only package tree from fixed-commit Git blobs."""

    evidence, inventory = _validated_causalbench_export_assets(
        method_assets_root,
        repository=repository,
        commit=commit,
    )
    if evidence != dict(expected_evidence):
        raise TaskCRuntimeError("CausalBench assets changed after export preflight")
    target = Path(destination)
    descriptor_bound_parent = (
        len(target.parts) >= 5
        and target.parts[:4] == ("/", "proc", "self", "fd")
        and target.parts[4].isdigit()
    )
    if not descriptor_bound_parent:
        _reject_symlink_components(target.parent, "private CausalBench snapshot parent")
    if target.exists() or target.is_symlink():
        raise TaskCRuntimeError("private CausalBench snapshot destination already exists")
    target.mkdir(mode=0o700)
    try:
        for relative, mode, payload in inventory:
            path = target / relative
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o500 if mode == "100755" else 0o400,
            )
            try:
                written = 0
                while written < len(payload):
                    written += os.write(descriptor, payload[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for current_root, directory_names, _ in os.walk(target, topdown=False):
            for directory_name in directory_names:
                (Path(current_root) / directory_name).chmod(0o500)
        target.chmod(0o500)
        _fsync_directory(target)
        if not descriptor_bound_parent:
            _fsync_directory(target.parent)
        verify_causalbench_source_snapshot(target, evidence)
    except BaseException:
        target.chmod(0o700)
        for current_root, directory_names, file_names in os.walk(
            target, topdown=False, followlinks=False
        ):
            current = Path(current_root)
            current.chmod(0o700)
            for directory_name in directory_names:
                (current / directory_name).chmod(0o700)
            for file_name in file_names:
                (current / file_name).chmod(0o600)
        shutil.rmtree(target)
        raise
    return evidence


def verify_causalbench_source_snapshot(
    source_root: str | Path,
    evidence: Mapping[str, str],
) -> None:
    """Recheck the private package inventory without importing it."""

    if set(evidence) != {
        "repository",
        "commit",
        "bootstrap_manifest_sha256",
        "source_worktree_sha256",
        "private_source_inventory_sha256",
    }:
        raise TaskCRuntimeError("CausalBench source evidence schema is invalid")
    expected = evidence.get("private_source_inventory_sha256")
    if type(expected) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
        raise TaskCRuntimeError("CausalBench private source inventory hash is invalid")
    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise TaskCRuntimeError("private CausalBench source is not a directory")
    observed = _private_source_inventory_sha256(root)
    if f"sha256:{observed}" != expected:
        raise TaskCRuntimeError("private CausalBench source snapshot changed")


def remove_causalbench_source_snapshot(
    source_root: str | Path,
    evidence: Mapping[str, str],
) -> None:
    """Verify and remove the private read-only snapshot before publication."""

    root = Path(source_root)
    verify_causalbench_source_snapshot(root, evidence)
    for current_root, directory_names, _ in os.walk(
        root, topdown=False, followlinks=False
    ):
        current = Path(current_root)
        current.chmod(0o700)
        for directory_name in directory_names:
            (current / directory_name).chmod(0o700)
    shutil.rmtree(root)


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
    expected_bytes = _json_bytes(payload)
    if path.exists() or path.is_symlink():
        snapshot = _snapshot_regular_file(
            path,
            path.name,
            maximum_bytes=MAXIMUM_BOOTSTRAP_RECORD_BYTES,
        )
        _strict_json_loads(snapshot.payload, path.name)
        if snapshot.payload != expected_bytes:
            raise TaskCRuntimeError(
                f"existing {path.name} bytes have a different identity"
            )
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


def _unlink_cache_record(path: Path) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    _require_safe_regular_file(path, path.name)
    path.unlink()
    _fsync_directory(path.parent)


def _ensure_cache_subdirectory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise TaskCRuntimeError(f"cache {label} must not be a symbolic link")
    if path.exists():
        if not path.is_dir():
            raise TaskCRuntimeError(f"cache {label} must be a directory")
    else:
        path.mkdir(exist_ok=False)
    return path


def _reject_unexpected_entries(
    directory: Path,
    expected_names: set[str],
    label: str,
) -> None:
    unexpected = {entry.name for entry in directory.iterdir()} - expected_names
    if unexpected:
        raise TaskCRuntimeError(
            f"cache {label} contains unexpected entries: {sorted(unexpected)}"
        )


def bootstrap_task_c_methods(
    *,
    cache_root: str | Path,
    registry_path: str | Path,
    project_root: str | Path,
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """获取固定版本的官方代码，建立隔离环境，并留下可重复核对的记录。"""

    registry_snapshot = _snapshot_regular_file(
        Path(registry_path),
        "method registry",
    )
    # Validate the immutable bytes before creating any cache or staging path.
    registry = _validate_registry_snapshot(registry_snapshot)
    root = _prepare_cache_root(cache_root)
    staging = root / f".bootstrap-staging-{uuid.uuid4().hex}"
    staging_created = False
    try:
        staging.mkdir(mode=0o700)
        staging_created = True
        _fsync_directory(root)
        registry_copy = staging / "task_c_methods_v1.json"
        _write_snapshot(registry_copy, registry_snapshot.payload)
        sources = _source_records(registry)
        project = Path(project_root)
        in_progress = {
            "schema_version": SCHEMA_VERSION,
            "status": "asset_preparation_in_progress",
        }
        _atomic_replace_json(root / "bootstrap_status.json", in_progress)
        _unlink_cache_record(root / "bootstrap_manifest.json")
        environments = _environment_records(registry, project)
        environment_copies: dict[str, Path] = {}
        input_directory = staging / "environment_inputs"
        input_directory.mkdir()
        for environment_name, snapshot in environments.items():
            destination = input_directory / snapshot.path.name
            _write_snapshot(destination, snapshot.payload)
            environment_copies[environment_name] = destination

        identity = _bootstrap_identity(
            registry_snapshot,
            sources,
            environments,
        )
        identity_path = root / "bootstrap_identity.json"
        _write_same_or_new(identity_path, identity)

        source_root = _ensure_cache_subdirectory(root / "sources", "sources")
        _reject_unexpected_entries(source_root, set(sources), "sources")
        source_manifests: dict[str, dict[str, str]] = {}
        for source_id, expected in sources.items():
            worktree_sha256 = _ensure_source(
                source_root / source_id,
                expected,
                run_command,
            )
            source_manifests[source_id] = {
                **dict(expected),
                "worktree_sha256": worktree_sha256,
            }

        publication_payloads = {
            method.method_id: classify_publication_only_method(method)
            for method in registry.methods.values()
            if method.source_kind == "publication_only"
        }
        staged_status_root = staging / "status"
        staged_status_root.mkdir()
        for method_id, payload in publication_payloads.items():
            method_root = staged_status_root / method_id
            method_root.mkdir()
            _atomic_create_json(method_root / "method_status.json", payload)

        environment_listing = _run_checked(
            run_command,
            ["conda", "env", "list", "--json"],
            capture_output=True,
        )
        existing_environments = _existing_environment_names(
            _strict_json_loads(environment_listing.stdout, "conda environment list")
        )
        staged_manifest_root = staging / "environment_manifests"
        staged_manifest_root.mkdir()
        environment_payloads: dict[str, dict[str, object]] = {}
        for environment_name, environment_file in environment_copies.items():
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
                existing_environments.add(environment_name)
            package_listing = _run_checked(
                run_command,
                [
                    "conda",
                    "run",
                    "-n",
                    environment_name,
                    "conda",
                    "list",
                    "--json",
                ],
                capture_output=True,
            )
            packages = _normalized_packages(
                _strict_json_loads(package_listing.stdout, "conda package list")
            )
            manifest: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "environment": environment_name,
                "specification_sha256": environments[environment_name].sha256,
                "packages": packages,
            }
            environment_payloads[environment_name] = manifest
            _atomic_create_json(
                staged_manifest_root / f"{environment_name}.json",
                manifest,
            )

        _verify_snapshot_unchanged(registry_snapshot, "method registry")
        for environment_name, snapshot in environments.items():
            _verify_snapshot_unchanged(
                snapshot,
                f"environment file for {environment_name}",
            )

        status_root = _ensure_cache_subdirectory(root / "status", "status")
        _reject_unexpected_entries(status_root, set(publication_payloads), "status")
        publication_hashes: dict[str, str] = {}
        for method_id, payload in publication_payloads.items():
            method_status_root = _ensure_cache_subdirectory(
                status_root / method_id,
                f"status/{method_id}",
            )
            _reject_unexpected_entries(
                method_status_root,
                {"method_status.json"},
                f"status/{method_id}",
            )
            path = method_status_root / "method_status.json"
            _write_same_or_new(path, payload)
            publication_hashes[f"{method_id}/method_status.json"] = _sha256_bytes(
                _json_bytes(payload)
            )

        manifest_root = _ensure_cache_subdirectory(
            root / "environment_manifests", "environment_manifests"
        )
        _reject_unexpected_entries(
            manifest_root,
            {f"{name}.json" for name in environment_payloads},
            "environment_manifests",
        )
        environment_hashes: dict[str, str] = {}
        for environment_name, payload in environment_payloads.items():
            filename = f"{environment_name}.json"
            _atomic_replace_json(manifest_root / filename, payload)
            environment_hashes[filename] = _sha256_bytes(_json_bytes(payload))

        overall_manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "bootstrap_identity_sha256": _sha256_bytes(_json_bytes(identity)),
            "sources": source_manifests,
            "environment_manifests": environment_hashes,
            "publication_statuses": publication_hashes,
        }
        staged_overall_path = staging / "bootstrap_manifest.json"
        _atomic_create_json(staged_overall_path, overall_manifest)
        staged_overall = _snapshot_regular_file(
            staged_overall_path,
            "staged bootstrap manifest",
            maximum_bytes=MAXIMUM_BOOTSTRAP_RECORD_BYTES,
        )
        if staged_overall.payload != _json_bytes(overall_manifest):
            raise TaskCRuntimeError(
                "staged bootstrap manifest changed before publication"
            )
        parsed_overall = _strict_json_loads(
            staged_overall.payload,
            "staged bootstrap manifest",
        )
        if parsed_overall != overall_manifest:
            raise TaskCRuntimeError("staged bootstrap manifest is inconsistent")
        _atomic_replace_json(root / "bootstrap_manifest.json", overall_manifest)
        overall_hash = _sha256_bytes(_json_bytes(overall_manifest))
        completed_status = {
            "schema_version": SCHEMA_VERSION,
            "status": "assets_and_environments_recorded",
            "bootstrap_manifest_sha256": overall_hash,
        }
        _atomic_replace_json(root / "bootstrap_status.json", completed_status)
    except BaseException:
        if staging_created:
            _unlink_cache_record(root / "bootstrap_manifest.json")
            failed_status = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_asset_preparation",
                "reason": (
                    "Official assets or isolated environments did not satisfy the "
                    "fixed preparation rules."
                ),
            }
            _atomic_replace_json(root / "bootstrap_status.json", failed_status)
        raise
    finally:
        if staging_created and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            _fsync_directory(root)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "assets_and_environments_recorded",
        "source_count": len(sources),
        "environment_count": len(environments),
        "publication_only_count": len(publication_payloads),
    }
