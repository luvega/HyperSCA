"""Safely create or verify the versioned formal Task C export bundle."""

from __future__ import annotations

import csv
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Mapping
import uuid
import zipfile

import numpy as np

from src.evaluation.task_c_acquisition import (
    TaskCAcquisitionError,
    bind_export_sources_against_acquisition,
    load_task_c_acquisition_manifest,
)
from src.evaluation.task_c_data import TASK_C_AUTHORITATIVE_SOURCE_MAXIMUM_GENES
from src.evaluation.task_c_runtime import (
    TaskCRuntimeError,
    materialize_causalbench_source_snapshot,
    remove_causalbench_source_snapshot,
    validate_causalbench_export_assets,
    verify_causalbench_source_snapshot,
)


SCHEMA_VERSION = "2.0"
REPOSITORY = "https://github.com/causalbench/causalbench.git"
COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
CONTEXTS = ("k562", "rpe1")
REFERENCES = (
    "reference_k562_pooled.csv",
    "reference_k562_chipseq.csv",
    "reference_rpe1_pooled.csv",
    "reference_rpe1_chipseq.csv",
)
SUPPORT_FILES = (
    "corum_complexes.txt.zip",
    "human_lr_pair.txt",
    "protein.links.txt.gz",
    "protein.physical.links.txt.gz",
    "protein.info.txt.gz",
)
MAXIMUM_SUPPORT_BYTES = 16 * 1024 * 1024 * 1024
MAXIMUM_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
MAXIMUM_NPZ_EXPANDED_BYTES = 32 * 1024 * 1024 * 1024
MAXIMUM_CELLS = 2_000_000
MAXIMUM_TEXT_ITEM_BYTES = 1024 * 1024
MAXIMUM_TOTAL_TEXT_BYTES = 512 * 1024 * 1024


class TaskCFormalExportError(ValueError):
    """A formal export cannot be safely created or reused."""


class _CausalBenchModuleBinding:
    """Temporarily bind all CausalBench imports to one private source tree."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve(strict=True)
        self._saved_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "causalscbench" or name.startswith("causalscbench.")
        }
        for name in self._saved_modules:
            sys.modules.pop(name, None)
        self._saved_path = list(sys.path)
        self._saved_dont_write_bytecode = sys.dont_write_bytecode
        sys.path.insert(0, str(source_root))
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        self._closed = False
        try:
            dataset_module = importlib.import_module(
                "causalscbench.data_access.create_dataset"
            )
            evaluation_module = importlib.import_module(
                "causalscbench.data_access.create_evaluation_datasets"
            )
            self.CreateDataset = dataset_module.CreateDataset
            self.CreateEvaluationDatasets = (
                evaluation_module.CreateEvaluationDatasets
            )
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self) -> None:
        observed = 0
        for name, module in tuple(sys.modules.items()):
            if name != "causalscbench" and not name.startswith("causalscbench."):
                continue
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise TaskCFormalExportError(
                    "CausalBench imported a module without a bound source file"
                )
            try:
                origin = Path(module_file).resolve(strict=True)
            except OSError as exc:
                raise TaskCFormalExportError(
                    "CausalBench imported module source disappeared"
                ) from exc
            if not origin.is_relative_to(self.source_root):
                raise TaskCFormalExportError(
                    "CausalBench imported code outside the private fixed source"
                )
            observed += 1
        if observed < 4:
            raise TaskCFormalExportError(
                "CausalBench private source did not load the required modules"
            )

    def close(self) -> None:
        if self._closed:
            return
        for name in tuple(sys.modules):
            if name == "causalscbench" or name.startswith("causalscbench."):
                sys.modules.pop(name, None)
        sys.modules.update(self._saved_modules)
        sys.path[:] = self._saved_path
        sys.dont_write_bytecode = self._saved_dont_write_bytecode
        importlib.invalidate_caches()
        self._closed = True


class _BoundFile:
    def __init__(self, path: Path, label: str, maximum_bytes: int) -> None:
        parts = path.parts
        if (
            len(parts) >= 6
            and parts[:4] == ("/", "proc", "self", "fd")
            and parts[4].isdigit()
        ):
            self.path = Path(os.path.abspath(path))
        else:
            self.path = _absolute_without_symlinks(path, label)
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise TaskCFormalExportError(f"{label}缺失或不是普通文件") from exc
        try:
            before = os.fstat(self.descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise TaskCFormalExportError(
                    f"{label}必须是 single-link regular file"
                )
            if before.st_size <= 0 or before.st_size > maximum_bytes:
                raise TaskCFormalExportError(f"{label}为空或超过大小上限")
            digest = hashlib.sha256()
            observed = 0
            while True:
                chunk = os.read(self.descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                digest.update(chunk)
            after = os.fstat(self.descriptor)
            self.identity = _identity(before)
            if self.identity != _identity(after) or observed != before.st_size:
                raise TaskCFormalExportError(f"{label}在核对期间发生变化")
            self.size_bytes = before.st_size
            self.sha256 = f"sha256:{digest.hexdigest()}"
            os.lseek(self.descriptor, 0, os.SEEK_SET)
        except BaseException:
            os.close(self.descriptor)
            raise

    def copy_to(self, directory: Path, name: str) -> None:
        target_descriptor: int | None = None
        try:
            target_descriptor = os.open(
                directory / name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            written = 0
            while True:
                chunk = os.read(self.descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                offset = 0
                while offset < len(chunk):
                    offset += os.write(target_descriptor, chunk[offset:])
                written += len(chunk)
            os.fsync(target_descriptor)
            if written != self.size_bytes or f"sha256:{digest.hexdigest()}" != self.sha256:
                raise TaskCFormalExportError(f"{self.label}安全复制结果不一致")
        finally:
            if target_descriptor is not None:
                os.close(target_descriptor)
            os.lseek(self.descriptor, 0, os.SEEK_SET)

    def verify_unchanged(self) -> None:
        current = os.fstat(self.descriptor)
        if _identity(current) != self.identity or not stat.S_ISREG(current.st_mode):
            raise TaskCFormalExportError(f"{self.label}在正式导出期间发生变化")
        try:
            path_metadata = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise TaskCFormalExportError(f"{self.label}在正式导出后缺失") from exc
        if _identity(path_metadata) != self.identity:
            raise TaskCFormalExportError(f"{self.label}在正式导出期间被替换")

    def close(self) -> None:
        os.close(self.descriptor)


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _absolute_without_symlinks(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TaskCFormalExportError(f"无法读取{label}路径") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCFormalExportError(f"{label}不能经过 symbolic link")
    return absolute


def _artifact_names(use_filter: bool) -> tuple[str, ...]:
    suffix = "_filtered" if use_filter else ""
    return (
        f"dataset_k562{suffix}.npz",
        f"dataset_rpe1{suffix}.npz",
        *REFERENCES,
    )


def _sha256_file(path: Path, label: str, maximum_bytes: int) -> tuple[str, int]:
    bound = _BoundFile(path, label, maximum_bytes)
    try:
        bound.verify_unchanged()
        return bound.sha256, bound.size_bytes
    finally:
        bound.close()


def _npy_header(handle: object, label: str) -> tuple[tuple[int, ...], np.dtype]:
    try:
        version = np.lib.format.read_magic(handle)  # type: ignore[arg-type]
        shape, fortran, dtype = np.lib.format._read_array_header(  # type: ignore[attr-defined]
            handle, version
        )
    except (EOFError, OSError, ValueError) as exc:
        raise TaskCFormalExportError(f"{label} NPY header 无效") from exc
    if fortran or not isinstance(shape, tuple) or dtype.hasobject or dtype.fields:
        raise TaskCFormalExportError(f"{label} NPY layout 无效")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _strict_json_load(payload: bytes, label: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TaskCFormalExportError(f"{label}包含重复 JSON 字段")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    except TaskCFormalExportError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise TaskCFormalExportError(f"{label}不是有效 JSON") from exc


def _stream_numeric_is_finite(
    handle: object, dtype: np.dtype, element_count: int, label: str
) -> None:
    remaining = element_count * dtype.itemsize
    tail = b""
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))  # type: ignore[attr-defined]
        if not chunk:
            raise TaskCFormalExportError(f"{label} NPY payload 截断")
        remaining -= len(chunk)
        data = tail + chunk
        usable = len(data) - len(data) % dtype.itemsize
        if usable:
            values = np.frombuffer(data[:usable], dtype=dtype)
            if not np.isfinite(values).all():
                raise TaskCFormalExportError(f"{label}包含 non-finite 数值")
        tail = data[usable:]
    if tail or handle.read(1):  # type: ignore[attr-defined]
        raise TaskCFormalExportError(f"{label} NPY payload 大小不一致")


def _stream_text_payload(
    handle: object,
    dtype: np.dtype,
    element_count: int,
    label: str,
) -> set[str]:
    remaining = element_count * dtype.itemsize
    tail = b""
    observed: set[str] = set()
    total_text_bytes = 0
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))  # type: ignore[attr-defined]
        if not chunk:
            raise TaskCFormalExportError(f"{label} NPY payload 截断")
        remaining -= len(chunk)
        data = tail + chunk
        usable = len(data) - len(data) % dtype.itemsize
        if usable:
            values = np.frombuffer(data[:usable], dtype=dtype)
            for raw_value in values.tolist():
                if isinstance(raw_value, bytes):
                    try:
                        value = raw_value.decode("utf-8", errors="strict")
                    except UnicodeError as exc:
                        raise TaskCFormalExportError(
                            f"{label}包含无效 UTF-8 文本"
                        ) from exc
                else:
                    value = raw_value
                if (
                    not isinstance(value, str)
                    or not value
                    or value != value.strip()
                    or any(ord(character) < 32 or ord(character) == 127 for character in value)
                ):
                    raise TaskCFormalExportError(f"{label}包含非规范文本")
                encoded_size = len(value.encode("utf-8", errors="strict"))
                if encoded_size > MAXIMUM_TEXT_ITEM_BYTES:
                    raise TaskCFormalExportError(f"{label}包含异常长文本")
                total_text_bytes += encoded_size
                if total_text_bytes > MAXIMUM_TOTAL_TEXT_BYTES:
                    raise TaskCFormalExportError(f"{label}文本总量超过上限")
                observed.add(value)
        tail = data[usable:]
    if tail or handle.read(1):  # type: ignore[attr-defined]
        raise TaskCFormalExportError(f"{label} NPY payload 大小不一致")
    return observed


def _validate_npz(path: Path) -> dict[str, object]:
    bound = _BoundFile(path, path.name, MAXIMUM_ARTIFACT_BYTES)
    expected = {
        "expression_matrix.npy",
        "interventions.npy",
        "var_names.npy",
    }
    try:
        stream = os.fdopen(os.dup(bound.descriptor), "rb")
        with stream, zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != expected:
                raise TaskCFormalExportError(
                    f"{path.name}必须精确包含三个登记数组"
                )
            if sum(member.file_size for member in members) > MAXIMUM_NPZ_EXPANDED_BYTES:
                raise TaskCFormalExportError(f"{path.name}展开大小超过上限")
            headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
            text_values: dict[str, set[str]] = {}
            for member in members:
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or Path(member.filename).name != member.filename
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.file_size <= 0
                ):
                    raise TaskCFormalExportError(f"{path.name} archive 结构不安全")
                with archive.open(member) as handle:
                    shape, dtype = _npy_header(handle, f"{path.name}:{member.filename}")
                    headers[member.filename] = (shape, dtype)
                    if member.filename == "expression_matrix.npy":
                        if (
                            len(shape) != 2
                            or not 1 <= shape[0] <= MAXIMUM_CELLS
                            or shape[1] < 1
                            or shape[1]
                            > TASK_C_AUTHORITATIVE_SOURCE_MAXIMUM_GENES
                            or dtype.kind not in "iuf"
                            or dtype.itemsize <= 0
                        ):
                            raise TaskCFormalExportError(
                                f"{path.name} expression shape 或数值类型无效"
                            )
                        count = shape[0] * shape[1]
                        if count * dtype.itemsize > MAXIMUM_NPZ_EXPANDED_BYTES:
                            raise TaskCFormalExportError(
                                f"{path.name} expression 超过展开上限"
                            )
                        _stream_numeric_is_finite(handle, dtype, count, path.name)
                    else:
                        maximum = (
                            MAXIMUM_CELLS
                            if member.filename == "interventions.npy"
                            else TASK_C_AUTHORITATIVE_SOURCE_MAXIMUM_GENES
                        )
                        if (
                            len(shape) != 1
                            or not 1 <= shape[0] <= maximum
                            or dtype.kind not in "US"
                            or dtype.itemsize <= 0
                        ):
                            raise TaskCFormalExportError(
                                f"{path.name}文本数组 shape 或类型无效"
                            )
                        count = shape[0]
                        if count * dtype.itemsize > MAXIMUM_NPZ_EXPANDED_BYTES:
                            raise TaskCFormalExportError(
                                f"{path.name}文本数组超过展开上限"
                            )
                        text_values[member.filename] = _stream_text_payload(
                            handle,
                            dtype,
                            count,
                            f"{path.name}:{member.filename}",
                        )
            expression_shape, _ = headers["expression_matrix.npy"]
            interventions_shape, interventions_dtype = headers["interventions.npy"]
            genes_shape, genes_dtype = headers["var_names.npy"]
            if (
                len(expression_shape) != 2
                or not 1 <= expression_shape[0] <= MAXIMUM_CELLS
                or expression_shape[1] < 1
                or expression_shape[1]
                > TASK_C_AUTHORITATIVE_SOURCE_MAXIMUM_GENES
                or interventions_shape != (expression_shape[0],)
                or genes_shape != (expression_shape[1],)
                or interventions_dtype.kind not in "US"
                or genes_dtype.kind not in "US"
                or interventions_dtype.itemsize <= 0
                or genes_dtype.itemsize <= 0
            ):
                raise TaskCFormalExportError(f"{path.name}数组 shape 或类型无效")
            genes = text_values["var_names.npy"]
            interventions = text_values["interventions.npy"]
            eligible = interventions - {"non-targeting", "excluded"}
            if (
                len(genes) != expression_shape[1]
                or "non-targeting" not in interventions
                or not eligible
                or not eligible <= genes
            ):
                raise TaskCFormalExportError(
                    f"{path.name}基因或干预标签不符合正式任务 C 约定"
                )
    except TaskCFormalExportError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TaskCFormalExportError(f"{path.name}不是安全 NPZ") from exc
    finally:
        try:
            bound.verify_unchanged()
        finally:
            bound.close()
    return {"shape": [expression_shape[0], expression_shape[1]]}


def _validate_reference(path: Path, *, pooled: bool) -> dict[str, object]:
    bound = _BoundFile(path, path.name, MAXIMUM_ARTIFACT_BYTES)
    edges: set[tuple[str, str]] = set()
    try:
        binary = os.fdopen(os.dup(bound.descriptor), "rb")
        with binary, io.TextIOWrapper(
            binary, encoding="utf-8", newline=""
        ) as handle:
            rows = csv.reader(handle, strict=True)
            if next(rows) != ["source", "target"]:
                raise TaskCFormalExportError(f"{path.name}表头无效")
            for row in rows:
                if len(row) != 2 or any(not value or value != value.strip() for value in row):
                    raise TaskCFormalExportError(f"{path.name}包含无效关系")
                edge = (row[0], row[1])
                if edge[0] == edge[1] or edge in edges:
                    raise TaskCFormalExportError(f"{path.name}包含自关系或重复关系")
                edges.add(edge)
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise TaskCFormalExportError(f"{path.name}不是有效关系 CSV") from exc
    finally:
        try:
            bound.verify_unchanged()
        finally:
            bound.close()
    if pooled and any((target, source) not in edges for source, target in edges):
        raise TaskCFormalExportError(f"{path.name} pooled 关系不是双向展开")
    return {"edge_count": len(edges)}


def _write_edges_exclusive(path: Path, edges: set[tuple[str, str]]) -> int:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["source", "target"])
    writer.writerows(sorted(edges))
    content = output.getvalue().encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(edges)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_artifacts(directory: Path, use_filter: bool) -> dict[str, object]:
    validation: dict[str, object] = {}
    for name in _artifact_names(use_filter):
        path = directory / name
        if name.endswith(".npz"):
            validation[name] = _validate_npz(path)
        else:
            validation[name] = _validate_reference(path, pooled="_pooled" in name)
    return validation


def _validate_existing(
    output: Path,
    use_filter: bool,
    acquisition_reference: Mapping[str, object],
    observed_sources: Mapping[str, object],
    causalbench_source: Mapping[str, str],
) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(output, flags)
    except OSError as exc:
        raise TaskCFormalExportError(
            "existing formal export 不是普通 directory"
        ) from exc
    try:
        metadata = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise TaskCFormalExportError("existing formal export 不是普通 directory")
        observed_path = output.stat(follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (
            observed_path.st_dev,
            observed_path.st_ino,
        ):
            raise TaskCFormalExportError("existing formal export directory 被替换")
        bound_output = Path(f"/proc/self/fd/{directory_descriptor}")
        expected = {*_artifact_names(use_filter), "export_manifest.json"}
        if {entry.name for entry in os.scandir(bound_output)} != expected:
            raise TaskCFormalExportError(
                "existing formal export 不是 complete 七文件目录"
            )
        fresh_validation = _validate_artifacts(bound_output, use_filter)
        manifest_bound = _BoundFile(
            bound_output / "export_manifest.json",
            "existing export manifest",
            4 * 1024 * 1024,
        )
        try:
            payload_bytes = os.read(
                manifest_bound.descriptor, manifest_bound.size_bytes
            )
            manifest = _strict_json_load(
                payload_bytes, "existing export manifest"
            )
        except (TaskCFormalExportError, OSError) as exc:
            raise TaskCFormalExportError("existing export manifest 无效") from exc
        finally:
            try:
                manifest_bound.verify_unchanged()
            finally:
                manifest_bound.close()
        expected_manifest_keys = {
            "schema_version",
            "status",
            "repository",
            "commit",
            "filter",
            "paths",
            "artifact_sha256",
            "artifact_size_bytes",
            "artifact_validation",
            "acquisition_manifest",
            "verified_converted_sources",
            "source_binding",
            "supporting_source_files",
            "downloaded_at_utc",
            "acquisition_time_note",
            "exported_at_utc",
            "dropped_self_edges",
            "causalbench_source",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != expected_manifest_keys
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "formal_export_complete"
            or manifest.get("repository") != REPOSITORY
            or manifest.get("commit") != COMMIT
            or manifest.get("filter") is not bool(use_filter)
            or manifest.get("acquisition_manifest") != dict(acquisition_reference)
            or manifest.get("verified_converted_sources") != dict(observed_sources)
            or manifest.get("source_binding")
            != "verified read-only /proc/self/fd snapshots"
            or manifest.get("downloaded_at_utc") is not None
            or manifest.get("artifact_validation") != fresh_validation
            or manifest.get("causalbench_source") != dict(causalbench_source)
            or manifest.get("acquisition_time_note")
            != "No server download time is claimed; local files were verified snapshots."
            or not isinstance(manifest.get("exported_at_utc"), str)
        ):
            raise TaskCFormalExportError(
                "existing formal export 身份、schema 或来源 stale"
            )
        try:
            exported_at = datetime.fromisoformat(manifest["exported_at_utc"])
        except ValueError as exc:
            raise TaskCFormalExportError(
                "existing formal export 导出时间无效"
            ) from exc
        if exported_at.utcoffset() != timezone.utc.utcoffset(exported_at):
            raise TaskCFormalExportError(
                "existing formal export 导出时间不是 UTC"
            )
        supporting = manifest.get("supporting_source_files")
        if not isinstance(supporting, dict) or set(supporting) != {
            *SUPPORT_FILES,
            *(("summary_stats.xlsx",) if use_filter else ()),
        }:
            raise TaskCFormalExportError("existing formal export 辅助来源不完整")
        for record in supporting.values():
            if (
                not isinstance(record, dict)
                or set(record) != {"sha256", "size_bytes", "origin"}
                or not isinstance(record.get("sha256"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", record["sha256"])
                is None
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
                or record.get("origin")
                not in {
                    "safe_copy_from_source_data_dir",
                    "downloaded_by_pinned_causalbench",
                }
            ):
                raise TaskCFormalExportError(
                    "existing formal export 辅助来源记录无效"
                )
        dropped = manifest.get("dropped_self_edges")
        if not isinstance(dropped, dict) or set(dropped) != set(CONTEXTS):
            raise TaskCFormalExportError("existing formal export 自关系记录无效")
        for counts in dropped.values():
            if (
                not isinstance(counts, dict)
                or set(counts) != {"pooled", "chipseq"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in counts.values()
                )
            ):
                raise TaskCFormalExportError(
                    "existing formal export 自关系计数无效"
                )
        hashes = manifest.get("artifact_sha256")
        sizes = manifest.get("artifact_size_bytes")
        paths = manifest.get("paths")
        if not isinstance(hashes, dict) or set(hashes) != set(
            _artifact_names(use_filter)
        ):
            raise TaskCFormalExportError("existing formal export hash 清单不完整")
        if not isinstance(sizes, dict) or set(sizes) != set(hashes):
            raise TaskCFormalExportError("existing formal export size 清单不完整")
        if not isinstance(paths, dict) or set(paths) != set(hashes):
            raise TaskCFormalExportError("existing formal export path 清单不完整")
        for name in hashes:
            if paths[name] != name or Path(paths[name]).is_absolute():
                raise TaskCFormalExportError("existing formal export path 无效")
            digest, size = _sha256_file(
                bound_output / name, name, MAXIMUM_ARTIFACT_BYTES
            )
            if hashes[name] != digest or sizes[name] != size:
                raise TaskCFormalExportError("existing formal export artifact stale")
        final_path = output.stat(follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (
            final_path.st_dev,
            final_path.st_ino,
        ):
            raise TaskCFormalExportError("existing formal export directory 被替换")
        result = dict(manifest)
        result["reuse_status"] = "verified_existing_formal_export"
        return result
    finally:
        os.close(directory_descriptor)


def _rename_noreplace(parent_descriptor: int, source: str, target: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TaskCFormalExportError("本机缺少排他目录发布所需的 renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise TaskCFormalExportError("formal export publish target already exists")
        raise TaskCFormalExportError(f"formal export publish renameat2 failed: errno {error}")


def _fsync_parent_after_publish(parent_descriptor: int) -> None:
    os.fsync(parent_descriptor)


def _clear_owned_directory(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    for name in os.listdir(descriptor):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                _clear_owned_directory(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _same_directory_path(path: Path, expected: os.stat_result) -> bool:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(observed.st_mode) and (
        observed.st_dev,
        observed.st_ino,
    ) == (expected.st_dev, expected.st_ino)


def export_task_c_formal_bundle(
    *,
    source_data_dir: str | Path,
    output_dir: str | Path,
    acquisition_manifest: str | Path,
    method_assets_root: str | Path,
    use_filter: bool,
) -> dict[str, object]:
    """Build a new formal bundle privately, or fully validate an existing one."""

    source = _absolute_without_symlinks(Path(source_data_dir), "原始数据目录")
    if not source.is_dir():
        raise TaskCFormalExportError("原始数据目录缺失或不是 directory")
    output = _absolute_without_symlinks(Path(output_dir), "正式导出目录")
    if output == source:
        raise TaskCFormalExportError("正式导出目录必须与原始数据目录分开")
    parent = _absolute_without_symlinks(output.parent, "正式导出父目录")
    if not parent.is_dir():
        raise TaskCFormalExportError("正式导出父目录必须预先存在")
    acquisition_bound = _BoundFile(
        Path(acquisition_manifest), "acquisition manifest", 4 * 1024 * 1024
    )
    try:
        acquisition, acquisition_reference = load_task_c_acquisition_manifest(
            acquisition_manifest, require_official_metadata=True
        )
    except TaskCAcquisitionError as exc:
        acquisition_bound.close()
        raise TaskCFormalExportError(f"公开数据获取记录无效：{exc}") from exc
    if (
        acquisition_reference.get("sha256") != acquisition_bound.sha256
        or acquisition_reference.get("size_bytes") != acquisition_bound.size_bytes
    ):
        acquisition_bound.close()
        raise TaskCFormalExportError("acquisition manifest 读取快照不一致")
    try:
        causalbench_source = validate_causalbench_export_assets(
            method_assets_root,
            repository=REPOSITORY,
            commit=COMMIT,
        )
    except TaskCRuntimeError as exc:
        acquisition_bound.close()
        raise TaskCFormalExportError(f"固定 CausalBench 资产无效：{exc}") from exc
    converted_paths = {context: source / f"{context}.h5ad" for context in CONTEXTS}
    support_names = (*SUPPORT_FILES, *(("summary_stats.xlsx",) if use_filter else ()))
    support_bound: dict[str, _BoundFile] = {}
    staging_name: str | None = None
    staging_descriptor: int | None = None
    parent_descriptor: int | None = None
    private_source: Path | None = None
    module_binding: _CausalBenchModuleBinding | None = None
    try:
        with bind_export_sources_against_acquisition(
            acquisition, converted_paths
        ) as bound_sources:
            observed_sources = bound_sources.observed_records
            if output.exists() or output.is_symlink():
                result = _validate_existing(
                    output,
                    bool(use_filter),
                    acquisition_reference,
                    observed_sources,
                    causalbench_source,
                )
                bound_sources.verify_unchanged()
                acquisition_bound.verify_unchanged()
                if validate_causalbench_export_assets(
                    method_assets_root,
                    repository=REPOSITORY,
                    commit=COMMIT,
                ) != causalbench_source:
                    raise TaskCFormalExportError(
                        "CausalBench 资产在正式导出复用核对期间发生变化"
                    )
                return result
            for name in support_names:
                candidate = source / name
                if candidate.exists() or candidate.is_symlink():
                    support_bound[name] = _BoundFile(
                        candidate, f"原始辅助缓存 {name}", MAXIMUM_SUPPORT_BYTES
                    )
            parent_descriptor = os.open(
                parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_metadata = os.fstat(parent_descriptor)
            if not _same_directory_path(parent, parent_metadata):
                raise TaskCFormalExportError("正式导出父目录在绑定时被替换")
            staging_name = f".{output.name}.staging-{uuid.uuid4().hex}"
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
            staging_descriptor = os.open(
                staging_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            staging = Path(f"/proc/self/fd/{staging_descriptor}")
            private_source = staging / ".causalbench-source"
            try:
                materialized_source = materialize_causalbench_source_snapshot(
                    method_assets_root,
                    private_source,
                    repository=REPOSITORY,
                    commit=COMMIT,
                    expected_evidence=causalbench_source,
                )
            except TaskCRuntimeError as exc:
                raise TaskCFormalExportError(
                    f"无法建立固定 CausalBench 私有源码快照：{exc}"
                ) from exc
            if materialized_source != causalbench_source:
                raise TaskCFormalExportError(
                    "CausalBench 私有源码快照身份与预检不一致"
                )
            verify_causalbench_source_snapshot(private_source, causalbench_source)
            for context, descriptor_path in bound_sources.descriptor_paths.items():
                os.symlink(descriptor_path, staging / f"{context}.h5ad")
            for name, bound in support_bound.items():
                bound.copy_to(staging, name)

            module_binding = _CausalBenchModuleBinding(private_source)
            CreateDataset = module_binding.CreateDataset
            CreateEvaluationDatasets = module_binding.CreateEvaluationDatasets
            returned = CreateDataset(str(staging), bool(use_filter)).load()
            dataset_names = _artifact_names(bool(use_filter))[:2]
            if len(returned) != 2:
                raise TaskCFormalExportError("CausalBench 未返回两个 dataset")
            for returned_path, expected_name in zip(returned, dataset_names, strict=True):
                if Path(returned_path).absolute() != (staging / expected_name).absolute():
                    raise TaskCFormalExportError("CausalBench dataset 输出路径越界")
            dropped_self: dict[str, dict[str, int]] = {}
            for context, dataset_name in (
                ("k562", "weissmann_k562"),
                ("rpe1", "weissmann_rpe1"),
            ):
                corum, ligand_receptor, string_network, string_physical, chipseq = (
                    CreateEvaluationDatasets(str(staging), dataset_name).load()
                )
                pooled = set().union(
                    corum, ligand_receptor, string_network, string_physical, chipseq
                )
                pooled_self = {edge for edge in pooled if edge[0] == edge[1]}
                chipseq_set = set(chipseq)
                chipseq_self = {edge for edge in chipseq_set if edge[0] == edge[1]}
                pooled -= pooled_self
                pooled = pooled | {(target, source_name) for source_name, target in pooled}
                _write_edges_exclusive(
                    staging / f"reference_{context}_pooled.csv", pooled
                )
                _write_edges_exclusive(
                    staging / f"reference_{context}_chipseq.csv",
                    chipseq_set - chipseq_self,
                )
                dropped_self[context] = {
                    "pooled": len(pooled_self),
                    "chipseq": len(chipseq_self),
                }

            supporting_records: dict[str, object] = {}
            for name in support_names:
                candidate = staging / name
                digest, size = _sha256_file(
                    candidate, f"实际辅助缓存 {name}", MAXIMUM_SUPPORT_BYTES
                )
                if name in support_bound and (
                    digest != support_bound[name].sha256
                    or size != support_bound[name].size_bytes
                ):
                    raise TaskCFormalExportError(
                        f"CausalBench 改写了只应读取的辅助缓存 {name}"
                    )
                supporting_records[name] = {
                    "sha256": digest,
                    "size_bytes": size,
                    "origin": (
                        "safe_copy_from_source_data_dir"
                        if name in support_bound
                        else "downloaded_by_pinned_causalbench"
                    ),
                }
            bound_sources.verify_unchanged()
            acquisition_bound.verify_unchanged()
            for bound in support_bound.values():
                bound.verify_unchanged()
            assert module_binding is not None
            assert private_source is not None
            module_binding.verify()
            verify_causalbench_source_snapshot(private_source, causalbench_source)
            module_binding.close()
            module_binding = None
            remove_causalbench_source_snapshot(private_source, causalbench_source)
            private_source = None

            descriptor_paths = bound_sources.descriptor_paths
            for context in CONTEXTS:
                staged_h5ad = staging / f"{context}.h5ad"
                metadata = staged_h5ad.lstat()
                if (
                    not stat.S_ISLNK(metadata.st_mode)
                    or os.readlink(staged_h5ad) != str(descriptor_paths[context])
                ):
                    raise TaskCFormalExportError(
                        f"CausalBench 改写了绑定的 {context} H5AD 入口"
                    )
                staged_h5ad.unlink()
            for name in support_names:
                (staging / name).unlink()
            validation = _validate_artifacts(staging, bool(use_filter))
            artifact_sha256: dict[str, str] = {}
            artifact_sizes: dict[str, int] = {}
            paths: dict[str, str] = {}
            for name in _artifact_names(bool(use_filter)):
                digest, size = _sha256_file(staging / name, name, MAXIMUM_ARTIFACT_BYTES)
                artifact_sha256[name] = digest
                artifact_sizes[name] = size
                paths[name] = name
            manifest: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "status": "formal_export_complete",
                "repository": REPOSITORY,
                "commit": COMMIT,
                "filter": bool(use_filter),
                "paths": paths,
                "artifact_sha256": artifact_sha256,
                "artifact_size_bytes": artifact_sizes,
                "artifact_validation": validation,
                "acquisition_manifest": acquisition_reference,
                "verified_converted_sources": observed_sources,
                "source_binding": "verified read-only /proc/self/fd snapshots",
                "supporting_source_files": supporting_records,
                "downloaded_at_utc": None,
                "acquisition_time_note": (
                    "No server download time is claimed; local files were verified snapshots."
                ),
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "dropped_self_edges": dropped_self,
                "causalbench_source": causalbench_source,
            }
            _write_json_exclusive(staging / "export_manifest.json", manifest)
            if {entry.name for entry in os.scandir(staging)} != {
                *_artifact_names(bool(use_filter)),
                "export_manifest.json",
            }:
                raise TaskCFormalExportError("formal staging 不是精确七文件目录")
            for entry in os.scandir(staging):
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise TaskCFormalExportError("formal staging 含非普通或多链接文件")
            bound_sources.verify_unchanged()
            for bound in support_bound.values():
                bound.verify_unchanged()
            try:
                current_causalbench_source = validate_causalbench_export_assets(
                    method_assets_root,
                    repository=REPOSITORY,
                    commit=COMMIT,
                )
            except TaskCRuntimeError as exc:
                raise TaskCFormalExportError(
                    f"CausalBench 资产在发布前核对失败：{exc}"
                ) from exc
            if current_causalbench_source != causalbench_source:
                raise TaskCFormalExportError(
                    "CausalBench 资产在正式导出期间发生变化"
                )
            if not _same_directory_path(parent, parent_metadata):
                raise TaskCFormalExportError("正式导出父目录在发布前被替换")
            os.fsync(staging_descriptor)
            bound_sources.verify_unchanged()
            acquisition_bound.verify_unchanged()
            staging_metadata = os.fstat(staging_descriptor)
            _rename_noreplace(parent_descriptor, staging_name, output.name)
            staging_name = None
            try:
                final_descriptor = os.open(
                    output.name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
                try:
                    final_metadata = os.fstat(final_descriptor)
                finally:
                    os.close(final_descriptor)
                if (
                    not stat.S_ISDIR(final_metadata.st_mode)
                    or (final_metadata.st_dev, final_metadata.st_ino)
                    != (staging_metadata.st_dev, staging_metadata.st_ino)
                ):
                    raise OSError("published directory inode changed")
                _fsync_parent_after_publish(parent_descriptor)
                if not _same_directory_path(parent, parent_metadata):
                    raise OSError("published parent directory path changed")
                bound_sources.verify_unchanged()
                acquisition_bound.verify_unchanged()
                if validate_causalbench_export_assets(
                    method_assets_root,
                    repository=REPOSITORY,
                    commit=COMMIT,
                ) != causalbench_source:
                    raise TaskCFormalExportError(
                        "CausalBench 资产在发布后发生变化"
                    )
            except BaseException as exc:
                assert staging_descriptor is not None
                _clear_owned_directory(staging_descriptor)
                os.rmdir(output.name, dir_fd=parent_descriptor)
                try:
                    os.fsync(parent_descriptor)
                except OSError:
                    pass
                if isinstance(exc, TaskCFormalExportError):
                    raise
                if isinstance(exc, TaskCAcquisitionError):
                    raise TaskCFormalExportError(
                        "acquisition 在发布后变化；已删除正式导出目录"
                    ) from exc
                if isinstance(exc, TaskCRuntimeError):
                    raise TaskCFormalExportError(
                        "CausalBench 资产发布后核对失败；已删除正式导出目录"
                    ) from exc
                raise TaskCFormalExportError(
                    "formal export post-publish check failed; published target removed"
                ) from exc
            result = dict(manifest)
            result["reuse_status"] = "created_new_formal_export"
            return result
    except (TaskCAcquisitionError, TaskCRuntimeError) as exc:
        raise TaskCFormalExportError(str(exc)) from exc
    finally:
        if module_binding is not None:
            module_binding.close()
        if private_source is not None and private_source.exists():
            try:
                remove_causalbench_source_snapshot(
                    private_source, causalbench_source
                )
            except (TaskCRuntimeError, OSError):
                # The owned staging directory cleanup below remains the final fallback.
                pass
        for bound in support_bound.values():
            bound.close()
        acquisition_bound.close()
        if staging_name is not None and parent_descriptor is not None:
            if staging_descriptor is not None:
                _clear_owned_directory(staging_descriptor)
            try:
                os.rmdir(staging_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
