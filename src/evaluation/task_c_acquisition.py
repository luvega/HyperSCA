"""核对任务 C 公开数据的镜像来源和基因标识转换。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping
import uuid

import numpy as np


SCHEMA_VERSION = "1.0"
RECORD_TYPE = "task_c_acquisition"
ZENODO_RECORD_ID = 7041849
ZENODO_DOI = "10.5281/zenodo.7041849"
ZENODO_LICENSE = "CC-BY-4.0"
ZENODO_RECORD_URL = "https://zenodo.org/records/7041849"
MAXIMUM_H5AD_BYTES = 16 * 1024 * 1024 * 1024
MAXIMUM_RECORD_BYTES = 4 * 1024 * 1024
MAXIMUM_EVIDENCE_BYTES = 1024 * 1024
MAXIMUM_MATRIX_CHUNK_BYTES = 128 * 1024 * 1024
MAXIMUM_CELLS = 2_000_000
MAXIMUM_GENES = 100_000
MAXIMUM_METADATA_TEXT_BYTES = 512 * 1024 * 1024
MAXIMUM_TEXT_VALUE_BYTES = 1024 * 1024
DEFAULT_CHUNK_ROWS = 256
_ENSEMBL_GENE_ID = re.compile(r"ENSG[0-9]+(?:\.[0-9]+)?")


class TaskCAcquisitionError(ValueError):
    """公开数据来源或转换结果不能支持正式任务 C 运行。"""


@dataclass(frozen=True)
class AcquisitionFileSpec:
    context_id: str
    file_name: str
    size_bytes: int
    md5: str
    zenodo_content_url: str
    figshare_original_url: str


OFFICIAL_ACQUISITION_FILES: dict[str, AcquisitionFileSpec] = {
    "k562": AcquisitionFileSpec(
        context_id="k562",
        file_name="ReplogleWeissman2022_K562_essential.h5ad",
        size_bytes=1_546_729_675,
        md5="d8cba17576d1a8afc0f7d71b79cad0f7",
        zenodo_content_url=(
            "https://zenodo.org/api/records/7041849/files/"
            "ReplogleWeissman2022_K562_essential.h5ad/content"
        ),
        figshare_original_url="https://plus.figshare.com/ndownloader/files/35773219",
    ),
    "rpe1": AcquisitionFileSpec(
        context_id="rpe1",
        file_name="ReplogleWeissman2022_rpe1.h5ad",
        size_bytes=1_236_886_900,
        md5="cc7f1ec50aeb3a3e1b4a6cfa713d80fa",
        zenodo_content_url=(
            "https://zenodo.org/api/records/7041849/files/"
            "ReplogleWeissman2022_rpe1.h5ad/content"
        ),
        figshare_original_url="https://plus.figshare.com/ndownloader/files/35775606",
    ),
}


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    link_count: int
    sha256: str
    md5: str
    payload: bytes | None = None

    @property
    def identity(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.size_bytes,
            self.modified_ns,
            self.changed_ns,
            self.link_count,
        )


def _absolute_without_symlinks(path: str | Path, label: str) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TaskCAcquisitionError(f"无法读取{label}的路径信息") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCAcquisitionError(f"{label}不能经过符号链接（symbolic link）")
    return absolute


def _capture_regular_file(
    path: str | Path,
    label: str,
    *,
    maximum_bytes: int,
    keep_payload: bool = False,
) -> _FileSnapshot:
    absolute = _absolute_without_symlinks(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise TaskCAcquisitionError(f"{label}缺失或不是安全的普通文件") from exc
    payload_parts: list[bytes] | None = [] if keep_payload else None
    sha256 = hashlib.sha256()
    md5 = hashlib.new("md5", usedforsecurity=False)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TaskCAcquisitionError(f"{label}必须是普通文件（regular file）")
        if before.st_nlink != 1:
            raise TaskCAcquisitionError(f"{label}必须只有一个文件链接")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise TaskCAcquisitionError(f"{label}为空或超过允许大小")
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise TaskCAcquisitionError(f"{label}超过允许大小")
            sha256.update(chunk)
            md5.update(chunk)
            if payload_parts is not None:
                payload_parts.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if before_identity != after_identity or observed != before.st_size:
        raise TaskCAcquisitionError(f"{label}在核对过程中发生变化")
    return _FileSnapshot(
        path=absolute,
        device=before.st_dev,
        inode=before.st_ino,
        size_bytes=before.st_size,
        modified_ns=before.st_mtime_ns,
        changed_ns=before.st_ctime_ns,
        link_count=before.st_nlink,
        sha256=f"sha256:{sha256.hexdigest()}",
        md5=md5.hexdigest(),
        payload=b"".join(payload_parts) if payload_parts is not None else None,
    )


def _verify_path_identity(snapshot: _FileSnapshot, label: str) -> None:
    absolute = _absolute_without_symlinks(snapshot.path, label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise TaskCAcquisitionError(f"{label}在核对后缺失") from exc
    current = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )
    if current != snapshot.identity or not stat.S_ISREG(metadata.st_mode):
        raise TaskCAcquisitionError(f"{label}在核对过程中发生变化")


def _open_snapshot_descriptor(snapshot: _FileSnapshot, label: str) -> int:
    """Reopen the verified inode once, then keep it bound during HDF5 reads."""

    absolute = _absolute_without_symlinks(snapshot.path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise TaskCAcquisitionError(f"{label}在分块核对前缺失或被替换") from exc
    try:
        metadata = os.fstat(descriptor)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_nlink,
        )
        if identity != snapshot.identity or not stat.S_ISREG(metadata.st_mode):
            raise TaskCAcquisitionError(f"{label}在分块核对前发生变化")
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        if not descriptor_path.exists():
            raise TaskCAcquisitionError("本机缺少绑定 H5AD 文件所需的 /proc/self/fd")
        os.set_inheritable(descriptor, False)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_json_load(payload: bytes, label: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskCAcquisitionError(f"{label}不是 UTF-8 文本") from exc
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
                raise TaskCAcquisitionError(f"{label}嵌套过深")
        elif character in "]}":
            depth -= 1

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TaskCAcquisitionError(f"{label}包含重复字段 {key}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise TaskCAcquisitionError(f"{label}不是有效 JSON") from exc


def _consume_text(value: object, remaining: list[int], label: str) -> None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    encoded = str(value).encode("utf-8")
    if len(encoded) > MAXIMUM_TEXT_VALUE_BYTES:
        raise TaskCAcquisitionError(f"{label}包含异常长文本")
    remaining[0] -= len(encoded)
    if remaining[0] < 0:
        raise TaskCAcquisitionError(f"{label}文本总量超过核对预算")


def _check_metadata_text(frame: object, remaining: list[int], label: str) -> None:
    import pandas as pd

    assert isinstance(frame, pd.DataFrame)
    for value in frame.index:
        _consume_text(value, remaining, f"{label} index")
    for column in frame.columns:
        _consume_text(column, remaining, f"{label} columns")
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = series.cat.categories
        elif pd.api.types.is_object_dtype(series.dtype) or isinstance(
            series.dtype, pd.StringDtype
        ):
            values = series
        else:
            continue
        for value in values:
            _consume_text(value, remaining, f"{label}.{column}")


def _matrix_storage(matrix: object) -> str:
    value = getattr(matrix, "format", None)
    return str(value).lower() if value in {"csr", "csc"} else "dense"


def _compare_expression(
    mirror: object,
    converted: object,
    *,
    shape: tuple[int, int],
    requested_chunk_rows: int,
) -> tuple[int, str]:
    from scipy import sparse

    if isinstance(requested_chunk_rows, bool) or not isinstance(
        requested_chunk_rows, int
    ) or requested_chunk_rows <= 0 or requested_chunk_rows > 4096:
        raise TaskCAcquisitionError("chunk rows 必须在 1 到 4096 之间")
    left_storage = _matrix_storage(mirror)
    right_storage = _matrix_storage(converted)
    if left_storage != right_storage:
        raise TaskCAcquisitionError("镜像与转换文件的表达矩阵存储结构不同")
    left_dtype = np.dtype(getattr(mirror, "dtype"))
    right_dtype = np.dtype(getattr(converted, "dtype"))
    if left_dtype != right_dtype or left_dtype.kind not in "biufc":
        raise TaskCAcquisitionError("镜像与转换文件的表达矩阵数值类型不同或无效")
    row_bytes = max(1, shape[1] * left_dtype.itemsize * 2)
    budget_rows = MAXIMUM_MATRIX_CHUNK_BYTES // row_bytes
    if budget_rows < 1:
        raise TaskCAcquisitionError("单行表达矩阵超过分块内存预算")
    chunk_rows = min(requested_chunk_rows, budget_rows)
    for start in range(0, shape[0], chunk_rows):
        stop = min(shape[0], start + chunk_rows)
        left = mirror[start:stop, :]
        right = converted[start:stop, :]
        if left_storage == "dense":
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
                raise TaskCAcquisitionError("表达矩阵必须全部为 finite 数值")
            if not np.array_equal(left_array, right_array):
                raise TaskCAcquisitionError("镜像与转换文件的 expression 数值不同")
            continue
        if not sparse.issparse(left) or not sparse.issparse(right):
            raise TaskCAcquisitionError("稀疏表达矩阵分块不能保持稀疏结构")
        if left.getformat() != left_storage or right.getformat() != right_storage:
            raise TaskCAcquisitionError("稀疏表达矩阵格式发生变化")
        if not np.isfinite(left.data).all() or not np.isfinite(right.data).all():
            raise TaskCAcquisitionError("表达矩阵必须全部为 finite 数值")
        for attribute in ("indptr", "indices", "data"):
            if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
                raise TaskCAcquisitionError(
                    "镜像与转换文件的稀疏 expression 结构或数值不同"
                )
    return chunk_rows, left_storage


def _verify_h5ad_pair(
    mirror_snapshot: _FileSnapshot,
    converted_snapshot: _FileSnapshot,
    *,
    requested_chunk_rows: int,
) -> dict[str, object]:
    mirror_descriptor = _open_snapshot_descriptor(
        mirror_snapshot, "Zenodo 镜像 H5AD"
    )
    try:
        converted_descriptor = _open_snapshot_descriptor(
            converted_snapshot, "转换后 H5AD"
        )
    except BaseException:
        os.close(mirror_descriptor)
        raise
    mirror = None
    converted = None
    try:
        import anndata as ad
        import pandas as pd

        try:
            mirror = ad.read_h5ad(
                Path(f"/proc/self/fd/{mirror_descriptor}"), backed="r"
            )
            converted = ad.read_h5ad(
                Path(f"/proc/self/fd/{converted_descriptor}"), backed="r"
            )
        except Exception as exc:
            raise TaskCAcquisitionError("H5AD 文件无法以只读分块方式打开") from exc
        if mirror.shape != converted.shape:
            raise TaskCAcquisitionError("镜像与转换文件的 shape 不同")
        cells, genes = mirror.shape
        if cells <= 0 or genes <= 0 or cells > MAXIMUM_CELLS or genes > MAXIMUM_GENES:
            raise TaskCAcquisitionError("H5AD shape 超过核对预算")
        remaining_text = [MAXIMUM_METADATA_TEXT_BYTES]
        for frame, label in (
            (mirror.obs, "mirror obs"),
            (converted.obs, "converted obs"),
            (mirror.var, "mirror var"),
            (converted.var, "converted var"),
        ):
            _check_metadata_text(frame, remaining_text, label)
        if not np.array_equal(
            mirror.obs_names.to_numpy(), converted.obs_names.to_numpy()
        ):
            raise TaskCAcquisitionError("镜像与转换文件的 obs names 不同")
        try:
            pd.testing.assert_frame_equal(
                mirror.obs,
                converted.obs,
                check_exact=True,
                check_categorical=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise TaskCAcquisitionError("镜像与转换文件的 obs table 不同") from exc
        if "ensembl_id" not in mirror.var.columns:
            raise TaskCAcquisitionError("镜像 var 缺少 ensembl_id")
        mirror_gene_names = mirror.var_names.astype(str).to_numpy()
        ensembl_ids = mirror.var["ensembl_id"].astype(str).to_numpy()
        if (
            any(not value for value in mirror_gene_names)
            or len(set(mirror_gene_names.tolist())) != genes
            or any(_ENSEMBL_GENE_ID.fullmatch(value) for value in mirror_gene_names)
        ):
            raise TaskCAcquisitionError("镜像 var_names 不是唯一的 gene symbols")
        if (
            any(_ENSEMBL_GENE_ID.fullmatch(value) is None for value in ensembl_ids)
            or len(set(ensembl_ids.tolist())) != genes
        ):
            raise TaskCAcquisitionError("镜像 ensembl_id 不是唯一的 Ensembl gene IDs")
        if not np.array_equal(converted.var_names.astype(str).to_numpy(), ensembl_ids):
            raise TaskCAcquisitionError("转换文件 var_names 未精确使用 mirror ensembl_id")
        if list(converted.var.columns) != [*mirror.var.columns.tolist(), "gene_name"]:
            raise TaskCAcquisitionError("转换文件 var 字段不符合固定转换规则")
        if not np.array_equal(
            converted.var["gene_name"].astype(str).to_numpy(), mirror_gene_names
        ):
            raise TaskCAcquisitionError("转换文件 gene_name 未精确保留 mirror var_names")
        try:
            pd.testing.assert_frame_equal(
                mirror.var.reset_index(drop=True),
                converted.var.drop(columns=["gene_name"]).reset_index(drop=True),
                check_exact=True,
                check_categorical=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise TaskCAcquisitionError("转换文件的其余 var 字段与镜像不同") from exc
        container_keys: dict[str, list[str]] = {}
        for name in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            mirror_keys = list(getattr(mirror, name).keys())
            converted_keys = list(getattr(converted, name).keys())
            if mirror_keys != converted_keys or mirror_keys:
                raise TaskCAcquisitionError(
                    f"{name} 必须在镜像与转换文件中同为空且键一致"
                )
            container_keys[name] = mirror_keys
        if mirror.raw is not None or converted.raw is not None:
            raise TaskCAcquisitionError("镜像与转换文件都不能包含 raw")
        chunk_rows, storage = _compare_expression(
            mirror.X,
            converted.X,
            shape=(cells, genes),
            requested_chunk_rows=requested_chunk_rows,
        )
        matrix_dtype = str(np.dtype(mirror.X.dtype))
    finally:
        if mirror is not None:
            mirror.file.close()
        if converted is not None:
            converted.file.close()
        os.close(mirror_descriptor)
        os.close(converted_descriptor)
    _verify_path_identity(mirror_snapshot, "Zenodo 镜像 H5AD")
    _verify_path_identity(converted_snapshot, "转换后 H5AD")
    return {
        "shape": [int(cells), int(genes)],
        "matrix_storage": storage,
        "matrix_dtype": matrix_dtype,
        "chunk_rows": int(chunk_rows),
        "maximum_matrix_chunk_bytes": MAXIMUM_MATRIX_CHUNK_BYTES,
        "maximum_metadata_text_bytes": MAXIMUM_METADATA_TEXT_BYTES,
        "expression_equal": True,
        "obs_equal": True,
        "var_conversion_equal": True,
        "empty_container_keys": container_keys,
        "raw_absent": True,
    }


def verify_h5ad_conversion(
    mirror_path: str | Path,
    converted_path: str | Path,
    *,
    requested_chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, object]:
    """逐块核对表达矩阵，并核对细胞和基因注释转换。"""

    mirror = _capture_regular_file(
        mirror_path,
        "Zenodo 镜像 H5AD",
        maximum_bytes=MAXIMUM_H5AD_BYTES,
    )
    converted = _capture_regular_file(
        converted_path,
        "转换后 H5AD",
        maximum_bytes=MAXIMUM_H5AD_BYTES,
    )
    return _verify_h5ad_pair(
        mirror,
        converted,
        requested_chunk_rows=requested_chunk_rows,
    )


def _file_record(snapshot: _FileSnapshot) -> dict[str, object]:
    return {
        "file_name": snapshot.path.name,
        "size_bytes": snapshot.size_bytes,
        "md5": snapshot.md5,
        "sha256": snapshot.sha256,
        "filesystem_observed_mtime_ns": snapshot.modified_ns,
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    absolute = _absolute_without_symlinks(path, "acquisition manifest 输出路径")
    if absolute.exists() or absolute.is_symlink():
        raise TaskCAcquisitionError("acquisition manifest 已存在，拒绝 overwrite")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    _absolute_without_symlinks(absolute.parent, "acquisition manifest 输出目录")
    temporary = absolute.parent / f".{absolute.name}.{uuid.uuid4().hex}.tmp"
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, absolute, follow_symlinks=False)
        except FileExistsError as exc:
            raise TaskCAcquisitionError(
                "acquisition manifest 发布时已存在，拒绝 overwrite"
            ) from exc
        temporary.unlink()
        directory = os.open(
            absolute.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def create_task_c_acquisition_manifest(
    *,
    mirror_paths: Mapping[str, str | Path],
    converted_paths: Mapping[str, str | Path],
    output_path: str | Path,
    figshare_403_evidence: Mapping[str, str | Path] | None = None,
    authoritative_files: Mapping[str, AcquisitionFileSpec] = OFFICIAL_ACQUISITION_FILES,
    requested_chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, object]:
    """验证两个公开镜像及转换文件，并排他写入外部来源记录。"""

    contexts = {"k562", "rpe1"}
    if (
        set(mirror_paths) != contexts
        or set(converted_paths) != contexts
        or set(authoritative_files) != contexts
    ):
        raise TaskCAcquisitionError("必须且只能提供 k562 与 rpe1 两组文件")
    evidence_paths = dict(figshare_403_evidence or {})
    if set(evidence_paths) - contexts:
        raise TaskCAcquisitionError("403 证据包含未知细胞环境")
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise TaskCAcquisitionError("acquisition manifest 已存在，拒绝 overwrite")
    datasets: dict[str, object] = {}
    for context in ("k562", "rpe1"):
        spec = authoritative_files[context]
        if spec.context_id != context:
            raise TaskCAcquisitionError("官方来源元数据的细胞环境不一致")
        mirror = _capture_regular_file(
            mirror_paths[context],
            f"{context} Zenodo 镜像 H5AD",
            maximum_bytes=MAXIMUM_H5AD_BYTES,
        )
        converted = _capture_regular_file(
            converted_paths[context],
            f"{context} 转换后 H5AD",
            maximum_bytes=MAXIMUM_H5AD_BYTES,
        )
        if mirror.path.name != spec.file_name:
            raise TaskCAcquisitionError(f"{context} 镜像文件名不是官方文件名")
        if mirror.size_bytes != spec.size_bytes:
            raise TaskCAcquisitionError(f"{context} 镜像 size 与官方记录不一致")
        if mirror.md5 != spec.md5:
            raise TaskCAcquisitionError(f"{context} 镜像 MD5 与官方记录不一致")
        validation = _verify_h5ad_pair(
            mirror,
            converted,
            requested_chunk_rows=requested_chunk_rows,
        )
        evidence_record: dict[str, object] | None = None
        if context in evidence_paths:
            evidence = _capture_regular_file(
                evidence_paths[context],
                f"{context} Figshare 403 证据",
                maximum_bytes=MAXIMUM_EVIDENCE_BYTES,
                keep_payload=True,
            )
            assert evidence.payload is not None
            normalized = evidence.payload.lower()
            if b"403" not in normalized or b"forbidden" not in normalized:
                raise TaskCAcquisitionError(
                    f"{context} Figshare 证据没有明确记录 403 Forbidden"
                )
            evidence_record = _file_record(evidence)
        mirror_record = _file_record(mirror)
        mirror_record.update(
            {
                "official_size_bytes": spec.size_bytes,
                "official_md5": spec.md5,
                "zenodo_content_url": spec.zenodo_content_url,
            }
        )
        datasets[context] = {
            "mirror": mirror_record,
            "converted": _file_record(converted),
            "figshare_original_url": spec.figshare_original_url,
            "figshare_403_evidence": evidence_record,
            "conversion_rule": {
                "mirror_var_names": "gene_symbols",
                "converted_var_names": "exact mirror ensembl_id values",
                "converted_gene_name": "exact mirror var_names values",
                "other_var_fields": "unchanged and in the original order",
            },
            "validation": validation,
        }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "status": "verified_local_acquisition_and_conversion",
        "zenodo": {
            "record_id": ZENODO_RECORD_ID,
            "doi": ZENODO_DOI,
            "license": ZENODO_LICENSE,
            "record_url": ZENODO_RECORD_URL,
        },
        "local_time_semantics": (
            "filesystem_observed_mtime_ns is local file metadata, not server download time"
        ),
        "datasets": datasets,
    }
    _write_exclusive_json(output, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": payload["status"],
        "acquisition_manifest": str(output.absolute()),
        "acquisition_manifest_sha256": (
            "sha256:"
            + hashlib.sha256(
                (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            ).hexdigest()
        ),
    }


def _validate_manifest_payload(
    payload: object,
    *,
    require_official_metadata: bool,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TaskCAcquisitionError("acquisition manifest 顶层必须是对象")
    if set(payload) != {
        "schema_version",
        "record_type",
        "status",
        "zenodo",
        "local_time_semantics",
        "datasets",
    }:
        raise TaskCAcquisitionError("acquisition manifest 顶层 schema 不完整")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("record_type") != RECORD_TYPE
        or payload.get("status") != "verified_local_acquisition_and_conversion"
    ):
        raise TaskCAcquisitionError("acquisition manifest schema 或状态无效")
    zenodo = payload.get("zenodo")
    if not isinstance(zenodo, dict) or zenodo != {
        "record_id": ZENODO_RECORD_ID,
        "doi": ZENODO_DOI,
        "license": ZENODO_LICENSE,
        "record_url": ZENODO_RECORD_URL,
    }:
        raise TaskCAcquisitionError("acquisition manifest 的 Zenodo 来源声明无效")
    if payload.get("local_time_semantics") != (
        "filesystem_observed_mtime_ns is local file metadata, not server download time"
    ):
        raise TaskCAcquisitionError("acquisition manifest 的本机时间含义无效")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"k562", "rpe1"}:
        raise TaskCAcquisitionError("acquisition manifest 必须包含 k562 与 rpe1")
    for context in ("k562", "rpe1"):
        entry = datasets.get(context)
        if not isinstance(entry, dict) or set(entry) != {
            "mirror",
            "converted",
            "figshare_original_url",
            "figshare_403_evidence",
            "conversion_rule",
            "validation",
        }:
            raise TaskCAcquisitionError(
                "acquisition manifest 数据条目 schema 或转换证据无效"
            )
        for role in ("mirror", "converted"):
            record = entry.get(role)
            expected_keys = {
                "file_name",
                "size_bytes",
                "md5",
                "sha256",
                "filesystem_observed_mtime_ns",
            }
            if role == "mirror":
                expected_keys.update(
                    {"official_size_bytes", "official_md5", "zenodo_content_url"}
                )
            if not isinstance(record, dict) or set(record) != expected_keys:
                raise TaskCAcquisitionError("acquisition manifest 文件记录无效")
            if (
                not isinstance(record.get("file_name"), str)
                or not record["file_name"]
                or Path(record["file_name"]).name != record["file_name"]
                or not isinstance(record.get("sha256"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", record["sha256"]) is None
                or not isinstance(record.get("md5"), str)
                or re.fullmatch(r"[0-9a-f]{32}", record["md5"]) is None
                or isinstance(record.get("size_bytes"), bool)
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
                or isinstance(record.get("filesystem_observed_mtime_ns"), bool)
                or not isinstance(record.get("filesystem_observed_mtime_ns"), int)
                or record["filesystem_observed_mtime_ns"] < 0
            ):
                raise TaskCAcquisitionError("acquisition manifest 文件指纹无效")
        evidence = entry.get("figshare_403_evidence")
        if evidence is not None:
            if not isinstance(evidence, dict) or set(evidence) != {
                "file_name",
                "size_bytes",
                "md5",
                "sha256",
                "filesystem_observed_mtime_ns",
            }:
                raise TaskCAcquisitionError("acquisition manifest 的 403 证据无效")
            if (
                not isinstance(evidence.get("file_name"), str)
                or Path(evidence["file_name"]).name != evidence["file_name"]
                or not isinstance(evidence.get("sha256"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", evidence["sha256"]) is None
                or not isinstance(evidence.get("md5"), str)
                or re.fullmatch(r"[0-9a-f]{32}", evidence["md5"]) is None
                or isinstance(evidence.get("size_bytes"), bool)
                or not isinstance(evidence.get("size_bytes"), int)
                or evidence["size_bytes"] <= 0
                or isinstance(evidence.get("filesystem_observed_mtime_ns"), bool)
                or not isinstance(evidence.get("filesystem_observed_mtime_ns"), int)
            ):
                raise TaskCAcquisitionError("acquisition manifest 的 403 证据指纹无效")
        if entry.get("conversion_rule") != {
            "mirror_var_names": "gene_symbols",
            "converted_var_names": "exact mirror ensembl_id values",
            "converted_gene_name": "exact mirror var_names values",
            "other_var_fields": "unchanged and in the original order",
        }:
            raise TaskCAcquisitionError("acquisition manifest 的 conversion 转换规则无效")
        validation = entry.get("validation")
        if not isinstance(validation, dict) or set(validation) != {
            "shape",
            "matrix_storage",
            "matrix_dtype",
            "chunk_rows",
            "maximum_matrix_chunk_bytes",
            "maximum_metadata_text_bytes",
            "expression_equal",
            "obs_equal",
            "var_conversion_equal",
            "empty_container_keys",
            "raw_absent",
        }:
            raise TaskCAcquisitionError("acquisition manifest 的转换验证 schema 无效")
        shape = validation.get("shape")
        empty_keys = validation.get("empty_container_keys")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
            or validation.get("matrix_storage") not in {"dense", "csr", "csc"}
            or not isinstance(validation.get("matrix_dtype"), str)
            or isinstance(validation.get("chunk_rows"), bool)
            or not isinstance(validation.get("chunk_rows"), int)
            or not 1 <= validation["chunk_rows"] <= 4096
            or validation.get("maximum_matrix_chunk_bytes")
            != MAXIMUM_MATRIX_CHUNK_BYTES
            or validation.get("maximum_metadata_text_bytes")
            != MAXIMUM_METADATA_TEXT_BYTES
            or validation.get("expression_equal") is not True
            or validation.get("obs_equal") is not True
            or validation.get("var_conversion_equal") is not True
            or validation.get("raw_absent") is not True
            or not isinstance(empty_keys, dict)
            or set(empty_keys) != {"layers", "obsm", "varm", "obsp", "varp", "uns"}
            or any(value != [] for value in empty_keys.values())
        ):
            raise TaskCAcquisitionError("acquisition manifest 的转换验证内容无效")
        if require_official_metadata:
            spec = OFFICIAL_ACQUISITION_FILES[context]
            mirror = entry["mirror"]
            assert isinstance(mirror, dict)
            if (
                mirror.get("file_name") != spec.file_name
                or mirror.get("size_bytes") != spec.size_bytes
                or mirror.get("md5") != spec.md5
                or mirror.get("official_size_bytes") != spec.size_bytes
                or mirror.get("official_md5") != spec.md5
                or mirror.get("zenodo_content_url") != spec.zenodo_content_url
                or entry.get("figshare_original_url") != spec.figshare_original_url
            ):
                raise TaskCAcquisitionError(
                    "acquisition manifest 与固定 Zenodo/Figshare 来源元数据不一致"
                )
    return payload


def load_task_c_acquisition_manifest(
    path: str | Path,
    *,
    require_official_metadata: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """严格读取一份不可替换的来源记录，并返回它的文件指纹。"""

    snapshot = _capture_regular_file(
        path,
        "acquisition manifest",
        maximum_bytes=MAXIMUM_RECORD_BYTES,
        keep_payload=True,
    )
    assert snapshot.payload is not None
    payload = _validate_manifest_payload(
        _safe_json_load(snapshot.payload, "acquisition manifest"),
        require_official_metadata=require_official_metadata,
    )
    _verify_path_identity(snapshot, "acquisition manifest")
    return payload, {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
    }


def verify_export_sources_against_acquisition(
    acquisition: Mapping[str, object],
    converted_paths: Mapping[str, str | Path],
) -> dict[str, dict[str, object]]:
    """确认导出命令实际读取的两个 H5AD 仍与来源记录相同。"""

    if set(converted_paths) != {"k562", "rpe1"}:
        raise TaskCAcquisitionError("导出来源必须且只能包含 k562 与 rpe1")
    datasets = acquisition.get("datasets")
    if not isinstance(datasets, dict):
        raise TaskCAcquisitionError("acquisition record 缺少数据条目")
    observed: dict[str, dict[str, object]] = {}
    for context in ("k562", "rpe1"):
        entry = datasets.get(context)
        expected = entry.get("converted") if isinstance(entry, dict) else None
        if not isinstance(expected, dict):
            raise TaskCAcquisitionError("acquisition record 缺少转换文件指纹")
        snapshot = _capture_regular_file(
            converted_paths[context],
            f"{context} 导出来源 H5AD",
            maximum_bytes=MAXIMUM_H5AD_BYTES,
        )
        if any(
            (
                snapshot.sha256 != expected.get("sha256"),
                snapshot.md5 != expected.get("md5"),
                snapshot.size_bytes != expected.get("size_bytes"),
            )
        ):
            raise TaskCAcquisitionError(
                f"{context} 导出来源与 acquisition record 不一致"
            )
        observed[context] = _file_record(snapshot)
    return observed
