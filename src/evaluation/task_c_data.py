"""Strict loading and provenance records for the real-data Task C benchmark."""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Integral
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CAUSALBENCH_REPOSITORY = "https://github.com/causalbench/causalbench.git"
CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
CONTROL_LABEL = "non-targeting"
TASK_C_AUTHORITATIVE_SOURCE_MAXIMUM_GENES = 100_000
TASK_C_NPZ_MAXIMUM_NPY_HEADER_BYTES = 64 * 1024
TASK_C_NPZ_MAXIMUM_CELLS = 1_000_000
TASK_C_NPZ_MAXIMUM_GENES = 1_000
TASK_C_NPZ_MAXIMUM_EXPRESSION_ELEMENTS = (
    TASK_C_NPZ_MAXIMUM_CELLS * TASK_C_NPZ_MAXIMUM_GENES
)
TASK_C_NPZ_MAXIMUM_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024


class TaskCDataError(ValueError):
    """Raised when a Task C input cannot meet the benchmark data contract."""


def preflight_task_c_npz_bytes(
    payload: bytes,
    *,
    label: str = "Task C dataset",
    maximum_npy_header_bytes: int = TASK_C_NPZ_MAXIMUM_NPY_HEADER_BYTES,
    maximum_cells: int = TASK_C_NPZ_MAXIMUM_CELLS,
    maximum_genes: int = TASK_C_NPZ_MAXIMUM_GENES,
    maximum_expression_elements: int = TASK_C_NPZ_MAXIMUM_EXPRESSION_ELEMENTS,
    maximum_expanded_bytes: int = TASK_C_NPZ_MAXIMUM_EXPANDED_BYTES,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any], int]]:
    """Reject unsafe fixed-array NPZ structure before NumPy allocates arrays."""

    expected_names = {
        "expression_matrix.npy",
        "interventions.npy",
        "var_names.npy",
    }
    if not isinstance(payload, bytes) or not payload:
        raise TaskCDataError(f"{label} ZIP bytes must be non-empty")
    limits = (
        maximum_npy_header_bytes,
        maximum_cells,
        maximum_genes,
        maximum_expression_elements,
        maximum_expanded_bytes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in limits):
        raise TaskCDataError(f"{label} ZIP limits are invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise TaskCDataError(f"{label} ZIP has duplicate members")
            if set(names) != expected_names or len(names) != len(expected_names):
                raise TaskCDataError(f"{label} ZIP members changed")
            if any(member.is_dir() or member.flag_bits & 0x1 for member in members):
                raise TaskCDataError(
                    f"{label} ZIP members must be regular and unencrypted"
                )
            if sum(int(member.file_size) for member in members) > maximum_expanded_bytes:
                raise TaskCDataError(f"{label} expanded members exceed the byte limit")

            total_array_bytes = 0
            headers: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}
            for member in members:
                with archive.open(member, mode="r") as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(
                            handle,
                            max_header_size=maximum_npy_header_bytes,
                        )
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(
                            handle,
                            max_header_size=maximum_npy_header_bytes,
                        )
                    else:
                        raise TaskCDataError(f"{label} NPY version is not allowed")
                    dtype = np.dtype(dtype)
                    if dtype.hasobject:
                        raise TaskCDataError(f"{label} NPY object arrays are not allowed")
                    element_count = 1
                    for dimension in shape:
                        if (
                            isinstance(dimension, bool)
                            or not isinstance(dimension, int)
                            or dimension < 0
                            or (
                                dimension
                                and element_count
                                > maximum_expression_elements // dimension
                            )
                        ):
                            raise TaskCDataError(
                                f"{label} NPY element count exceeds the limit"
                            )
                        element_count *= dimension
                    if dtype.itemsize < 1 or (
                        element_count > maximum_expanded_bytes // int(dtype.itemsize)
                    ):
                        raise TaskCDataError(
                            f"{label} NPY dtype or array bytes exceed the limit"
                        )
                    array_bytes = element_count * int(dtype.itemsize)
                    if handle.tell() > maximum_npy_header_bytes + 16:
                        raise TaskCDataError(f"{label} NPY header exceeds the limit")
                    if handle.tell() + array_bytes != int(member.file_size):
                        raise TaskCDataError(
                            f"{label} NPY header disagrees with member size"
                        )
                    total_array_bytes += array_bytes
                    if total_array_bytes > maximum_expanded_bytes:
                        raise TaskCDataError(
                            f"{label} expanded array bytes exceed the limit"
                        )
                    headers[member.filename] = (
                        tuple(int(dimension) for dimension in shape),
                        dtype,
                        element_count,
                    )

            expression_shape, expression_dtype, expression_elements = headers[
                "expression_matrix.npy"
            ]
            labels_shape, labels_dtype, labels_elements = headers[
                "interventions.npy"
            ]
            genes_shape, genes_dtype, genes_elements = headers["var_names.npy"]
            if (
                len(expression_shape) != 2
                or expression_dtype.kind not in {"i", "u", "f"}
                or not 1 <= expression_shape[0] <= maximum_cells
                or not 2 <= expression_shape[1] <= maximum_genes
                or expression_elements > maximum_expression_elements
            ):
                raise TaskCDataError(
                    f"{label} expression header has an unsafe shape or dtype"
                )
            if (
                len(labels_shape) != 1
                or labels_dtype.kind not in {"U", "S"}
                or labels_dtype.itemsize < 1
                or labels_shape[0] != expression_shape[0]
                or labels_elements > maximum_cells
            ):
                raise TaskCDataError(
                    f"{label} intervention text header disagrees with expression rows"
                )
            if (
                len(genes_shape) != 1
                or genes_dtype.kind not in {"U", "S"}
                or genes_dtype.itemsize < 1
                or genes_shape[0] != expression_shape[1]
                or genes_elements > maximum_genes
            ):
                raise TaskCDataError(
                    f"{label} gene text header disagrees with expression columns"
                )
            return headers
    except TaskCDataError:
        raise
    except (
        OSError,
        ValueError,
        EOFError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise TaskCDataError(f"{label} ZIP or NPY header is invalid") from exc


@dataclass(frozen=True)
class TaskCDataset:
    expression: np.ndarray
    interventions: np.ndarray
    gene_names: tuple[str, ...]
    context_id: str
    source_path: Path
    source_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_names", tuple(self.gene_names))
        object.__setattr__(self, "content_sha256", _dataset_content_sha256(self))
        self.expression.setflags(write=False)
        self.interventions.setflags(write=False)


@dataclass(frozen=True)
class TaskCSplit:
    schema_version: str
    split_id: str
    seed: int
    train_sources: tuple[str, ...]
    tune_sources: tuple[str, ...]
    holdout_sources: tuple[str, ...]
    control_indices: Mapping[str, Mapping[str, tuple[int, ...]]]
    min_cells_per_intervention: int

    def __post_init__(self) -> None:
        frozen_controls = {
            context: MappingProxyType({
                partition: tuple(indices) for partition, indices in partitions.items()
            })
            for context, partitions in self.control_indices.items()
        }
        object.__setattr__(self, "control_indices", MappingProxyType(frozen_controls))


@dataclass(frozen=True)
class _CommonGeneProjection:
    gene_names: tuple[str, ...]
    column_indices: Mapping[str, tuple[int, ...]]
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_names", tuple(self.gene_names))
        object.__setattr__(
            self,
            "column_indices",
            MappingProxyType(
                {context: tuple(indices) for context, indices in self.column_indices.items()}
            ),
        )
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))


_TASK_C_SPLIT_SEEDS = frozenset({11, 23, 47, 71, 97})

_WITHIN_ARTIFACTS = {
    context: {
        "train": f"within/{context}/train.npz",
        "tune": f"within/{context}/tune.npz",
        "refit": f"within/{context}/refit.npz",
        "holdout": f"private/within/{context}/holdout.npz",
    }
    for context in ("k562", "rpe1")
}
_CROSS_ARTIFACTS = {
    direction: {
        "source_train": f"cross/{direction}/source_train.npz",
        "source_tune": f"cross/{direction}/source_tune.npz",
        "source_refit": f"cross/{direction}/source_refit.npz",
        "target_adapt_train": f"cross/{direction}/target_adapt_train.npz",
        "target_adapt_tune": f"cross/{direction}/target_adapt_tune.npz",
        "target_adapt_refit": f"cross/{direction}/target_adapt_refit.npz",
        "target_holdout": f"private/cross/{direction}/target_holdout.npz",
    }
    for direction in ("k562_to_rpe1", "rpe1_to_k562")
}
_PUBLIC_MANIFEST = "public_manifest.json"
_PRIVATE_MANIFEST = "private/private_manifest.json"
_PUBLIC_ARTIFACT_PATHS = frozenset(
    relative
    for partitions in _WITHIN_ARTIFACTS.values()
    for name, relative in partitions.items()
    if name != "holdout"
) | frozenset(
    relative
    for partitions in _CROSS_ARTIFACTS.values()
    for name, relative in partitions.items()
    if name != "target_holdout"
)
_PRIVATE_ARTIFACT_PATHS = frozenset(
    partitions["holdout"] for partitions in _WITHIN_ARTIFACTS.values()
) | frozenset(
    partitions["target_holdout"] for partitions in _CROSS_ARTIFACTS.values()
)
SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD = (
    "sealed_holdout_semantic_content_sha256"
)


def _update_length_prefixed_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_array_digest_metadata(
    digest: Any,
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
) -> None:
    metadata = json.dumps(
        {"name": name, "dtype": dtype.str, "shape": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _update_length_prefixed_digest(digest, metadata)


def _update_array_digest(digest: Any, name: str, values: np.ndarray) -> None:
    _update_array_digest_metadata(
        digest,
        name,
        dtype=values.dtype,
        shape=values.shape,
    )
    if values.flags.c_contiguous:
        digest.update(memoryview(values).cast("B"))
        return
    iterator = np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=1024 * 1024,
    )
    for chunk in iterator:
        contiguous = np.ascontiguousarray(chunk)
        digest.update(memoryview(contiguous).cast("B"))


def _update_selected_matrix_digest(
    digest: Any,
    name: str,
    values: np.ndarray,
    row_indices: np.ndarray,
    column_indices: Sequence[int],
) -> None:
    columns = np.asarray(tuple(column_indices), dtype=int)
    _update_array_digest_metadata(
        digest,
        name,
        dtype=values.dtype,
        shape=(len(row_indices), len(columns)),
    )
    bytes_per_row = max(1, len(columns) * max(1, values.dtype.itemsize))
    rows_per_chunk = max(1, (1024 * 1024) // bytes_per_row)
    for start in range(0, len(row_indices), rows_per_chunk):
        rows = row_indices[start : start + rows_per_chunk]
        chunk = np.ascontiguousarray(values[np.ix_(rows, columns)])
        digest.update(memoryview(chunk).cast("B"))


def _update_selected_vector_digest(
    digest: Any,
    name: str,
    values: np.ndarray,
    row_indices: np.ndarray,
) -> None:
    _update_array_digest_metadata(
        digest,
        name,
        dtype=values.dtype,
        shape=(len(row_indices),),
    )
    items_per_chunk = max(1, (1024 * 1024) // max(1, values.dtype.itemsize))
    for start in range(0, len(row_indices), items_per_chunk):
        rows = row_indices[start : start + items_per_chunk]
        chunk = np.ascontiguousarray(values[rows])
        digest.update(memoryview(chunk).cast("B"))


def _update_gene_order_digest(digest: Any, gene_names: Sequence[str]) -> None:
    genes = json.dumps(
        tuple(gene_names),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    _update_length_prefixed_digest(digest, genes)


class SealedHoldoutSemanticContentHasher:
    """Stream the fixed four sealed holdouts into one public-safe commitment."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"TaskC-sealed-holdout-semantic-content-v1\0")
        self._ordered_paths = tuple(sorted(_PRIVATE_ARTIFACT_PATHS))
        self._position = 0

    def _add_logical_artifact(self, relative: str) -> None:
        if (
            self._position >= len(self._ordered_paths)
            or relative != self._ordered_paths[self._position]
        ):
            raise TaskCDataError(
                "sealed holdout artifacts must use the fixed complete logical order"
            )
        logical_artifact = json.dumps(
            {"logical_artifact": relative},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _update_length_prefixed_digest(self._digest, logical_artifact)
        self._position += 1

    def add_arrays(
        self,
        relative: str,
        expression: np.ndarray,
        interventions: np.ndarray,
        gene_names: Sequence[str],
    ) -> None:
        """Add one validated artifact while preserving its stored array dtypes."""
        self._add_logical_artifact(relative)
        _update_array_digest(self._digest, "expression", expression)
        _update_array_digest(self._digest, "interventions", interventions)
        _update_gene_order_digest(self._digest, gene_names)

    def add_projected_selection(
        self,
        relative: str,
        dataset: TaskCDataset,
        row_indices: np.ndarray,
        column_indices: Sequence[int],
        gene_names: Sequence[str],
    ) -> None:
        """Add the exact projected arrays that materialization will write."""
        self._add_logical_artifact(relative)
        _update_selected_matrix_digest(
            self._digest,
            "expression",
            dataset.expression,
            row_indices,
            column_indices,
        )
        _update_selected_vector_digest(
            self._digest,
            "interventions",
            dataset.interventions,
            row_indices,
        )
        _update_gene_order_digest(self._digest, gene_names)

    def sha256(self) -> str:
        if self._position != len(self._ordered_paths):
            raise TaskCDataError(
                "sealed holdout commitment requires exactly four logical artifacts"
            )
        return f"sha256:{self._digest.hexdigest()}"


def _dataset_content_sha256(dataset: TaskCDataset) -> str:
    digest = hashlib.sha256()
    digest.update(b"TaskCDataset-content-v1\0")
    _update_array_digest(digest, "expression", dataset.expression)
    _update_array_digest(digest, "interventions", dataset.interventions)
    genes = json.dumps(
        dataset.gene_names,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(genes).to_bytes(8, "big"))
    digest.update(genes)
    return f"sha256:{digest.hexdigest()}"


def validate_task_c_dataset_content(dataset: TaskCDataset) -> None:
    current = _dataset_content_sha256(dataset)
    if current != dataset.content_sha256:
        raise TaskCDataError(f"{dataset.context_id} dataset content changed in memory")
    if dataset.expression.flags.writeable or dataset.interventions.flags.writeable:
        raise TaskCDataError(f"{dataset.context_id} dataset arrays are no longer read-only")


def _validate_task_c_dataset_pair_content(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
) -> None:
    validate_task_c_dataset_content(k562)
    validate_task_c_dataset_content(rpe1)


def _eligible_sources(dataset: TaskCDataset, min_cells: int) -> set[str]:
    counts = Counter(dataset.interventions.tolist())
    genes = set(dataset.gene_names)
    return {label for label, count in counts.items() if label != CONTROL_LABEL and label in genes and count >= min_cells}


def _control_partitions(
    dataset: TaskCDataset,
    seed: int,
    minimum_tune_controls: int,
) -> dict[str, tuple[int, ...]]:
    controls = np.flatnonzero(dataset.interventions == CONTROL_LABEL)
    if len(controls) < minimum_tune_controls + 4:
        raise TaskCDataError(
            "control cells must reserve the public minimum for tuning and "
            "at least two cells for train and holdout"
        )
    shuffled = np.random.default_rng(seed).permutation(controls)
    tune_count = max(int(len(shuffled) * 0.2), minimum_tune_controls)
    holdout_count = max(int(len(shuffled) * 0.2), 2)
    train_end = len(shuffled) - tune_count - holdout_count
    tune_end = train_end + tune_count
    return {
        "train": tuple(sorted(int(i) for i in shuffled[:train_end])),
        "tune": tuple(sorted(int(i) for i in shuffled[train_end:tune_end])),
        "holdout": tuple(sorted(int(i) for i in shuffled[tune_end:])),
    }


def _source_partitions(sources: list[str], seed: int) -> dict[str, tuple[str, ...]]:
    shuffled = np.random.default_rng(seed).permutation(np.asarray(sources, dtype=str))
    train_end = int(len(shuffled) * 0.6)
    tune_end = train_end + int(len(shuffled) * 0.2)
    return {
        "train": tuple(sorted(str(x) for x in shuffled[:train_end])),
        "tune": tuple(sorted(str(x) for x in shuffled[train_end:tune_end])),
        "holdout": tuple(sorted(str(x) for x in shuffled[tune_end:])),
    }


def build_shared_task_c_split(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    *,
    seed: int,
    min_cells: int = 5,
) -> TaskCSplit:
    if k562.context_id != "k562" or rpe1.context_id != "rpe1":
        raise TaskCDataError("datasets must be provided in k562, rpe1 context order")
    if isinstance(min_cells, bool) or not isinstance(min_cells, Integral) or min_cells <= 0:
        raise TaskCDataError("min_cells must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed not in _TASK_C_SPLIT_SEEDS:
        raise TaskCDataError("seed must be one of 11, 23, 47, 71, 97")
    seed = int(seed)
    min_cells = int(min_cells)
    common = sorted(_eligible_sources(k562, min_cells) & _eligible_sources(rpe1, min_cells))
    if len(common) < 5:
        raise TaskCDataError("at least 5 shared eligible intervention sources are required")
    source_parts = _source_partitions(common, seed)
    split = TaskCSplit(
        schema_version="1.0",
        split_id=f"C-context-intervention-holdout-v1-seed-{seed}",
        seed=seed,
        train_sources=source_parts["train"],
        tune_sources=source_parts["tune"],
        holdout_sources=source_parts["holdout"],
        control_indices=MappingProxyType({
            "k562": MappingProxyType(_control_partitions(k562, seed, min_cells)),
            "rpe1": MappingProxyType(_control_partitions(rpe1, seed, min_cells)),
        }),
        min_cells_per_intervention=min_cells,
    )
    validate_task_c_split(split, k562, rpe1)
    return split


def validate_task_c_split(split: TaskCSplit, k562: TaskCDataset, rpe1: TaskCDataset) -> None:
    if split.schema_version != "1.0":
        raise TaskCDataError("unsupported Task C split schema")
    if isinstance(split.seed, bool) or not isinstance(split.seed, Integral) or split.seed not in _TASK_C_SPLIT_SEEDS:
        raise TaskCDataError("split seed is not registered")
    expected_id = f"C-context-intervention-holdout-v1-seed-{int(split.seed)}"
    if split.split_id != expected_id:
        raise TaskCDataError("split id is inconsistent with seed")
    if isinstance(split.min_cells_per_intervention, bool) or not isinstance(split.min_cells_per_intervention, Integral) or split.min_cells_per_intervention <= 0:
        raise TaskCDataError("min_cells_per_intervention must be positive")
    if k562.context_id != "k562" or rpe1.context_id != "rpe1":
        raise TaskCDataError("datasets must have k562, rpe1 context identities")

    source_parts = (split.train_sources, split.tune_sources, split.holdout_sources)
    names = ("train", "tune", "holdout")
    seen: set[str] = set()
    for name, part in zip(names, source_parts):
        if not isinstance(part, tuple) or not part:
            raise TaskCDataError(f"{name} source partition must be a nonempty tuple")
        if any(not isinstance(value, str) for value in part):
            raise TaskCDataError("source partitions must contain strings")
        if len(set(part)) != len(part):
            raise TaskCDataError("source partition contains duplicate sources")
        if seen.intersection(part):
            raise TaskCDataError("source partitions overlap")
        seen.update(part)
    expected_sources = sorted(_eligible_sources(k562, int(split.min_cells_per_intervention)) & _eligible_sources(rpe1, int(split.min_cells_per_intervention)))
    if seen != set(expected_sources):
        raise TaskCDataError("source partition union differs from exact shared eligible sources")
    expected_source_parts = _source_partitions(expected_sources, int(split.seed))
    if (split.train_sources, split.tune_sources, split.holdout_sources) != (
        expected_source_parts["train"], expected_source_parts["tune"], expected_source_parts["holdout"]
    ):
        raise TaskCDataError("source partitions do not match deterministic seed assignment")

    if not isinstance(split.control_indices, Mapping) or set(split.control_indices) != {"k562", "rpe1"}:
        raise TaskCDataError("control indices must contain exactly k562 and rpe1 contexts")
    for context, dataset in (("k562", k562), ("rpe1", rpe1)):
        partitions = split.control_indices[context]
        if not isinstance(partitions, Mapping) or set(partitions) != {"train", "tune", "holdout"}:
            raise TaskCDataError(f"{context} control partitions must contain exactly train, tune, holdout")
        controls = set(np.flatnonzero(dataset.interventions == CONTROL_LABEL).tolist())
        all_indices: set[int] = set()
        for name in ("train", "tune", "holdout"):
            part = partitions[name]
            if not isinstance(part, tuple):
                raise TaskCDataError("control partitions must be tuples")
            values: list[int] = []
            for index in part:
                if isinstance(index, bool) or not isinstance(index, Integral):
                    raise TaskCDataError("control indices must be integers")
                index = int(index)
                if index < 0 or index >= dataset.expression.shape[0]:
                    raise TaskCDataError("control index is out of range")
                if index not in controls:
                    raise TaskCDataError("control indices must point only to control rows")
                values.append(index)
            if len(set(values)) != len(values):
                raise TaskCDataError("control partition contains duplicate indices")
            if all_indices.intersection(values):
                raise TaskCDataError("control partitions overlap")
            all_indices.update(values)
        if all_indices != controls:
            raise TaskCDataError(f"{context} control partition union differs from all controls")
        expected_controls = _control_partitions(
            dataset,
            int(split.seed),
            int(split.min_cells_per_intervention),
        )
        if any(partitions[name] != expected_controls[name] for name in ("train", "tune", "holdout")):
            raise TaskCDataError(f"{context} control partitions do not match deterministic seed assignment")


def sha256_path(path: Path | str, chunked: int = 1024 * 1024) -> str:
    path = Path(path)
    if isinstance(chunked, bool):
        chunked = 1024 * 1024 if chunked else 1024 * 1024
    if chunked <= 0:
        raise TaskCDataError("SHA-256 chunk size must be positive")
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while data := handle.read(chunked):
                digest.update(data)
    except (OSError, ValueError) as exc:
        raise TaskCDataError(f"cannot read file for SHA-256: {path}") from exc
    return f"sha256:{digest.hexdigest()}"


def _text_vector(values: np.ndarray, kind: str) -> tuple[str, ...]:
    if values.ndim != 1:
        raise TaskCDataError(f"{kind} must be a one-dimensional array")
    if values.dtype.kind not in {"U", "S"}:
        raise TaskCDataError(f"{kind} must be one-dimensional Unicode or byte-string arrays")
    result: list[str] = []
    for value in values.tolist():
        if isinstance(value, (list, tuple, dict, set, np.ndarray)):
            raise TaskCDataError(f"{kind} contain non-stringable nested values")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="strict")
        try:
            text = str(value)
        except Exception as exc:
            raise TaskCDataError(f"{kind} contain non-stringable values") from exc
        if not text:
            result.append(text)
            continue
        if text != text.strip():
            raise TaskCDataError(f"{kind} must not contain leading or trailing whitespace")
        result.append(text)
    return tuple(result)


def _load_task_c_dataset_archive(
    archive_source: Path | io.BytesIO,
    *,
    source_path: Path,
    source_sha256: str,
    context_id: str,
    sealed_holdout_hasher: SealedHoldoutSemanticContentHasher | None = None,
    logical_artifact: str | None = None,
) -> TaskCDataset:
    if (sealed_holdout_hasher is None) != (logical_artifact is None):
        raise TaskCDataError(
            "sealed holdout hashing requires both a hasher and logical artifact"
        )
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("context_id must be exactly k562 or rpe1")
    try:
        with np.load(archive_source, allow_pickle=False) as archive:
            required = {"expression_matrix", "interventions", "var_names"}
            missing = required.difference(archive.files)
            if missing:
                raise TaskCDataError(f"missing required array(s): {', '.join(sorted(missing))}")
            expression = np.asarray(archive["expression_matrix"])
            interventions_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
    except TaskCDataError:
        raise
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TaskCDataError(f"cannot load Task C dataset: {source_path}") from exc

    if expression.ndim != 2:
        raise TaskCDataError("expression_matrix must be two-dimensional")
    if expression.shape[0] == 0 or expression.shape[1] < 2:
        raise TaskCDataError("expression_matrix must have rows and at least two gene columns")
    if not np.issubdtype(expression.dtype, np.number) or not np.all(np.isfinite(expression)):
        raise TaskCDataError("expression_matrix values must be finite numeric values")
    try:
        labels = _text_vector(interventions_raw, "intervention labels")
        genes = _text_vector(genes_raw, "gene names")
    except UnicodeError as exc:
        raise TaskCDataError("intervention labels or gene names contain invalid UTF-8") from exc
    if expression.shape[0] != len(labels):
        raise TaskCDataError("expression rows must equal intervention labels")
    if expression.shape[1] != len(genes):
        raise TaskCDataError("expression columns must equal gene names")
    if not genes or any(not gene for gene in genes):
        raise TaskCDataError("gene names must be nonempty")
    if len(set(genes)) != len(genes):
        raise TaskCDataError("gene names must be unique")
    if any(not label for label in labels):
        raise TaskCDataError("intervention labels must be nonempty")
    if CONTROL_LABEL not in labels:
        raise TaskCDataError("at least one non-targeting control is required")
    if sealed_holdout_hasher is not None and logical_artifact is not None:
        sealed_holdout_hasher.add_arrays(
            logical_artifact,
            expression,
            interventions_raw,
            genes,
        )
    return TaskCDataset(
        expression=expression,
        interventions=np.asarray(labels, dtype=str),
        gene_names=genes,
        context_id=context_id,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def load_task_c_dataset(
    path: Path | str,
    *,
    context_id: str,
    sealed_holdout_hasher: SealedHoldoutSemanticContentHasher | None = None,
    logical_artifact: str | None = None,
) -> TaskCDataset:
    source_path = Path(path).expanduser().resolve()
    before = _file_signature(source_path)
    dataset = _load_task_c_dataset_archive(
        source_path,
        source_path=source_path,
        source_sha256=_consistent_sha256(source_path, before),
        context_id=context_id,
        sealed_holdout_hasher=sealed_holdout_hasher,
        logical_artifact=logical_artifact,
    )
    if _file_signature(source_path) != before:
        raise TaskCDataError(f"dataset changed while being loaded: {source_path}")
    return dataset


def load_task_c_dataset_from_verified_bytes(
    path: Path | str,
    *,
    context_id: str,
    source_bytes: bytes,
    source_sha256: str,
    sealed_holdout_hasher: SealedHoldoutSemanticContentHasher | None = None,
    logical_artifact: str | None = None,
) -> TaskCDataset:
    """Parse bytes whose file identity and hash were verified by the caller."""

    source_path = Path(path).expanduser().resolve()
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise TaskCDataError("captured dataset bytes must be nonempty bytes")
    if (
        not isinstance(source_sha256, str)
        or not source_sha256.startswith("sha256:")
        or len(source_sha256) != 71
        or any(character not in "0123456789abcdef" for character in source_sha256[7:])
    ):
        raise TaskCDataError("captured dataset source_sha256 is malformed")
    preflight_task_c_npz_bytes(source_bytes, label="sealed Task C holdout")
    return _load_task_c_dataset_archive(
        io.BytesIO(source_bytes),
        source_path=source_path,
        source_sha256=source_sha256,
        context_id=context_id,
        sealed_holdout_hasher=sealed_holdout_hasher,
        logical_artifact=logical_artifact,
    )


def _file_signature(path: Path) -> tuple[int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise TaskCDataError(f"cannot stat dataset file: {path}") from exc
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _consistent_sha256(path: Path, before: tuple[int, int, int]) -> str:
    digest = sha256_path(path)
    if _file_signature(path) != before:
        raise TaskCDataError(f"dataset changed while being loaded: {path}")
    return digest


def build_task_c_provenance(dataset: TaskCDataset) -> dict[str, Any]:
    counts = Counter(dataset.interventions.tolist())
    n_controls = counts.get(CONTROL_LABEL, 0)
    return {
        "schema_version": "1.0",
        "context": dataset.context_id,
        "context_id": dataset.context_id,
        "repository": CAUSALBENCH_REPOSITORY,
        "causalbench_repository": CAUSALBENCH_REPOSITORY,
        "commit": CAUSALBENCH_COMMIT,
        "causalbench_commit": CAUSALBENCH_COMMIT,
        "figshare_source_url": {
            "k562": "https://plus.figshare.com/ndownloader/files/35773219",
            "rpe1": "https://plus.figshare.com/ndownloader/files/35775606",
        }[dataset.context_id],
        "source_path": str(dataset.source_path),
        "input_sha256": dataset.source_sha256,
        "n_cells": int(dataset.expression.shape[0]),
        "n_genes": int(dataset.expression.shape[1]),
        "control_label": CONTROL_LABEL,
        "n_control_cells": int(n_controls),
        "intervention_counts": dict(sorted(counts.items())),
        "validation": {
            "missing_arrays": [],
            "duplicate_gene_names": False,
            "nonfinite_expression": False,
        },
        "licenses": {"CausalBench": "Apache-2.0", "Replogle": "CC-BY-4.0"},
    }


def _validate_reference(path: Path | str) -> tuple[int, str, set[tuple[str, str]]]:
    path = Path(path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != ["source", "target"]:
                raise TaskCDataError(f"reference CSV must have exactly source,target headers: {path}")
            edges: set[tuple[str, str]] = set()
            count = 0
            for row in reader:
                if len(row) != 2:
                    raise TaskCDataError(f"reference CSV rows must have exactly two fields: {path}")
                source, target = row
                if not source or not target or source != source.strip() or target != target.strip() or source.lower() in {"nan", "inf", "infinity"} or target.lower() in {"nan", "inf", "infinity"}:
                    raise TaskCDataError(f"reference CSV source/target must be finite nonempty text without whitespace padding: {path}")
                if source == target:
                    raise TaskCDataError(f"reference CSV contains self edge: {path}")
                edge = (source, target)
                if edge in edges:
                    raise TaskCDataError(f"reference CSV contains duplicate directed row: {path}")
                edges.add(edge)
                count += 1
    except TaskCDataError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TaskCDataError(f"cannot read reference CSV: {path}") from exc
    if count == 0:
        raise TaskCDataError(f"reference CSV must contain at least one edge: {path}")
    return count, sha256_path(path), edges


def build_task_c_reference_provenance(*, context_id: str, pooled_path: Path | str, chipseq_path: Path | str) -> dict[str, Any]:
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("context_id must be exactly k562 or rpe1")
    pooled_rows, pooled_hash, pooled_edges = _validate_reference(pooled_path)
    chip_rows, chip_hash, _ = _validate_reference(chipseq_path)
    missing_reverse = {(target, source) for source, target in pooled_edges} - pooled_edges
    if missing_reverse:
        raise TaskCDataError("pooled reference must contain every reverse edge")
    return {
        "schema_version": "1.0",
        "context": context_id,
        "context_id": context_id,
        "repository": CAUSALBENCH_REPOSITORY,
        "causalbench_repository": CAUSALBENCH_REPOSITORY,
        "commit": CAUSALBENCH_COMMIT,
        "causalbench_commit": CAUSALBENCH_COMMIT,
        "primary_reference_id": "causalbench_pooled_biological_v1",
        "primary_reference_scope": "pooled biological evidence expanded in both directions",
        "directed_reference_id": "causalbench_chipseq_v1",
        "directed_reference_scope": "K562 ChIP file" if context_id == "k562" else "HepG2 ChIP file bundled by the pinned CausalBench RPE1 branch",
        "primary_evidence": {"id": "causalbench_pooled_biological_v1", "scope": "pooled biological evidence expanded in both directions"},
        "directed_evidence": {"id": "causalbench_chipseq_v1", "scope": "K562 ChIP file" if context_id == "k562" else "bundled HepG2 ChIP file"},
        "files": {
            "pooled": {"path": str(Path(pooled_path).expanduser().resolve()), "sha256": pooled_hash, "row_count": pooled_rows},
            "chipseq": {"path": str(Path(chipseq_path).expanduser().resolve()), "sha256": chip_hash, "row_count": chip_rows},
        },
        "pooled_sha256": pooled_hash,
        "chipseq_sha256": chip_hash,
        "pooled_row_count": pooled_rows,
        "chipseq_row_count": chip_rows,
        "license_mapping": {
            "causalbench_code": "Apache-2.0", "chip_atlas_adapted_data": "CC-BY-SA-4.0",
            "corum": "CC-BY-NC", "string_db": "CC-BY-4.0",
            "ligand_receptor_resource_as_declared_by_causalbench": "GPL-3.0",
        },
    }


def write_json(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError) as exc:
        raise TaskCDataError(f"cannot write provenance JSON: {destination}") from exc


def check_task_c_json_record(path: Path | str, payload: Mapping[str, Any]) -> str:
    """Check whether a provenance record is absent or exactly reusable, without writing."""
    destination = Path(path)
    if destination.is_symlink():
        raise TaskCDataError(f"provenance record must not be a symbolic link: {destination}")
    if not destination.exists():
        return "missing"
    if not destination.is_file():
        raise TaskCDataError(f"provenance record is not a regular file: {destination}")
    existing = _read_manifest(destination)
    if existing != payload:
        raise TaskCDataError(f"existing provenance record differs: {destination}")
    return "reusable"


def write_task_c_json_record(path: Path | str, payload: Mapping[str, Any]) -> str:
    """Write a missing record, or leave an exactly matching record byte-for-byte intact."""
    status = check_task_c_json_record(path, payload)
    if status == "missing":
        write_json(path, payload)
        return "written"
    return status


def _validated_row_indices(
    dataset: TaskCDataset,
    indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
        raise TaskCDataError("row indices must be a one-dimensional integer sequence")
    normalized = np.asarray([int(value) for value in values.tolist()], dtype=int)
    if len(set(normalized.tolist())) != len(normalized):
        raise TaskCDataError("row indices must be unique")
    if np.any(normalized < 0) or np.any(normalized >= dataset.expression.shape[0]):
        raise TaskCDataError("row index is out of range")
    return normalized


def _write_dataset_subset(
    dataset: TaskCDataset,
    indices: Sequence[int] | np.ndarray,
    path: Path,
    *,
    gene_names: tuple[str, ...] | None = None,
    column_indices: Sequence[int] | None = None,
) -> str:
    """Atomically write one validated, self-describing dataset subset."""
    selected = _validated_row_indices(dataset, indices)
    written_genes = dataset.gene_names if gene_names is None else tuple(gene_names)
    written_columns = (
        tuple(range(len(dataset.gene_names)))
        if column_indices is None
        else tuple(column_indices)
    )
    destination = Path(path).expanduser()
    temporary: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".npz",
            dir=destination.parent,
        )
        os.close(fd)
        np.savez_compressed(
            temporary,
            expression_matrix=dataset.expression[
                np.ix_(selected, np.asarray(written_columns, dtype=int))
            ],
            interventions=dataset.interventions[selected],
            var_names=np.asarray(written_genes),
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception as exc:
        raise TaskCDataError(f"cannot atomically write dataset subset: {destination}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return str(destination)


def _atomic_link_or_copy(source: Path, destination: Path) -> str:
    """Reuse an immutable public artifact without recompressing its rows."""
    temporary: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        os.close(fd)
        os.unlink(temporary)
        try:
            os.link(source, temporary)
        except OSError as exc:
            if exc.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                raise
            shutil.copyfile(source, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception as exc:
        raise TaskCDataError(f"cannot atomically reuse dataset subset: {destination}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return str(destination)


def _indices_for_sources(
    dataset: TaskCDataset,
    sources: Sequence[str],
    control_indices: Sequence[int],
) -> np.ndarray:
    if isinstance(sources, (str, bytes)) or any(
        not isinstance(source, str) or not source or source == CONTROL_LABEL
        for source in sources
    ):
        raise TaskCDataError("intervention sources must be nonempty text labels")
    if len(set(sources)) != len(sources):
        raise TaskCDataError("intervention sources must be unique")
    observed = set(dataset.interventions.tolist())
    missing = set(sources) - observed
    if missing:
        raise TaskCDataError("intervention source is absent from the dataset")
    controls = _validated_row_indices(dataset, control_indices)
    if any(dataset.interventions[index] != CONTROL_LABEL for index in controls):
        raise TaskCDataError("control indices must point only to control rows")
    source_mask = np.isin(dataset.interventions, np.asarray(tuple(sources), dtype=str))
    selected = np.concatenate((np.flatnonzero(source_mask), controls))
    return _validated_row_indices(dataset, np.sort(selected))


def _sealed_holdout_semantic_content_sha256_from_split(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    projection: _CommonGeneProjection,
) -> str:
    datasets = {"k562": k562, "rpe1": rpe1}
    all_sources = split.train_sources + split.tune_sources + split.holdout_sources
    selections: dict[str, tuple[TaskCDataset, np.ndarray, tuple[int, ...]]] = {}
    for context, dataset in datasets.items():
        relative = _WITHIN_ARTIFACTS[context]["holdout"]
        selections[relative] = (
            dataset,
            _indices_for_sources(
                dataset,
                split.holdout_sources,
                split.control_indices[context]["holdout"],
            ),
            projection.column_indices[context],
        )
    for source_name, target_name in (("k562", "rpe1"), ("rpe1", "k562")):
        target = datasets[target_name]
        direction = f"{source_name}_to_{target_name}"
        relative = _CROSS_ARTIFACTS[direction]["target_holdout"]
        selections[relative] = (
            target,
            _indices_for_sources(
                target,
                all_sources,
                split.control_indices[target_name]["holdout"],
            ),
            projection.column_indices[target_name],
        )
    if set(selections) != _PRIVATE_ARTIFACT_PATHS:
        raise TaskCDataError("sealed holdout selections are incomplete")

    hasher = SealedHoldoutSemanticContentHasher()
    for relative in sorted(selections):
        dataset, rows, columns = selections[relative]
        hasher.add_projected_selection(
            relative,
            dataset,
            rows,
            columns,
            projection.gene_names,
        )
    return hasher.sha256()


def _private_split_payload(split: TaskCSplit) -> dict[str, Any]:
    return {
        "schema_version": split.schema_version,
        "split_id": split.split_id,
        "seed": split.seed,
        "min_cells_per_intervention": split.min_cells_per_intervention,
        "train_sources": list(split.train_sources),
        "tune_sources": list(split.tune_sources),
        "holdout_sources": list(split.holdout_sources),
        "control_indices": {
            context: {name: list(values) for name, values in parts.items()}
            for context, parts in split.control_indices.items()
        },
    }


def _bundle_path(root: Path, relative: str) -> Path:
    lexical = Path(relative)
    if lexical.is_absolute() or not lexical.parts or ".." in lexical.parts:
        raise TaskCDataError("bundle artifact path must be a safe relative path")
    candidate = root
    for part in lexical.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise TaskCDataError(f"bundle path contains a symbolic link: {relative}")
    return root / lexical


def _artifact_inventory(paths: Iterable[str], root: Path) -> dict[str, str]:
    return {
        relative: sha256_path(_bundle_path(root, relative))
        for relative in sorted(paths)
    }


def _gene_names_sha256(gene_names: tuple[str, ...]) -> str:
    encoded = json.dumps(gene_names, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def task_c_materialization_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Fingerprint an identity for independent retention after data preparation."""
    if not isinstance(identity, Mapping) or not identity:
        raise TaskCDataError("materialization identity must be a non-empty mapping")
    try:
        return _canonical_sha256(dict(identity))
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise TaskCDataError("materialization identity cannot be fingerprinted") from exc


def _common_gene_projection(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
) -> _CommonGeneProjection:
    common_genes = tuple(sorted(set(k562.gene_names) & set(rpe1.gene_names)))
    if len(common_genes) < 2:
        raise TaskCDataError("K562 and RPE1 need at least two common genes")
    contexts: dict[str, dict[str, object]] = {}
    column_indices: dict[str, tuple[int, ...]] = {}
    for dataset in (k562, rpe1):
        lookup = {gene: index for index, gene in enumerate(dataset.gene_names)}
        indices = tuple(int(lookup[gene]) for gene in common_genes)
        column_indices[dataset.context_id] = indices
        mapping_payload = {
            "common_ordered_genes": list(common_genes),
            "selected_original_indices": list(indices),
        }
        contexts[dataset.context_id] = {
            "original_gene_count": len(dataset.gene_names),
            "original_gene_names_sha256": _gene_names_sha256(dataset.gene_names),
            "selected_original_indices": list(indices),
            "mapping_sha256": _canonical_sha256(mapping_payload),
        }
    record = {
        "projection_rule": "sorted_common_gene_intersection_v1",
        "common": {
            "count": len(common_genes),
            "ordered_genes": list(common_genes),
            "sha256": _gene_names_sha256(common_genes),
        },
        "contexts": contexts,
    }
    return _CommonGeneProjection(
        gene_names=common_genes,
        column_indices=column_indices,
        record=record,
    )


def _materialization_identity(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    projection: _CommonGeneProjection,
) -> dict[str, Any]:
    return {
        "schema_version": split.schema_version,
        "split_id": split.split_id,
        "seed": split.seed,
        "min_cells_per_intervention": split.min_cells_per_intervention,
        "input_sha256": {
            "k562": k562.source_sha256,
            "rpe1": rpe1.source_sha256,
        },
        "content_sha256": {
            "k562": k562.content_sha256,
            "rpe1": rpe1.content_sha256,
        },
        "gene_names_sha256": _gene_names_sha256(projection.gene_names),
        "gene_projection": dict(projection.record),
        SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD: (
            _sealed_holdout_semantic_content_sha256_from_split(
                k562,
                rpe1,
                split,
                projection,
            )
        ),
    }


def _public_split_payload(
    split: TaskCSplit,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": split.schema_version,
        "split_id": split.split_id,
        "seed": split.seed,
        "min_cells_per_intervention": split.min_cells_per_intervention,
        "train_sources": list(split.train_sources),
        "tune_sources": list(split.tune_sources),
        "holdout_source_count": len(split.holdout_sources),
        "input_sha256": identity["input_sha256"],
        "content_sha256": identity["content_sha256"],
        "gene_names_sha256": identity["gene_names_sha256"],
        "gene_projection": identity["gene_projection"],
        SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD: identity[
            SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD
        ],
    }


def _materialized_result(root: Path) -> dict[str, Any]:
    within = {
        context: {
            name: str(root / relative)
            for name, relative in _WITHIN_ARTIFACTS[context].items()
        }
        for context in ("k562", "rpe1")
    }
    cross = {
        direction: {
            name: str(root / relative)
            for name, relative in _CROSS_ARTIFACTS[direction].items()
        }
        for direction in ("k562_to_rpe1", "rpe1_to_k562")
    }
    return {
        "within": within,
        "cross": cross,
        "public_manifest": str(root / _PUBLIC_MANIFEST),
        "private_manifest": str(root / _PRIVATE_MANIFEST),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskCDataError(f"cannot verify existing manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise TaskCDataError(f"existing manifest is not a JSON object: {path}")
    return payload


def _verify_artifact_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    expected_paths: set[str],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected_paths:
        raise TaskCDataError("existing manifest has an incomplete artifact hash inventory")
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TaskCDataError("existing artifact inventory is malformed")
        candidate = _bundle_path(root, relative)
        if not candidate.is_file() or sha256_path(candidate) != expected_hash:
            raise TaskCDataError(f"existing artifact hash changed: {relative}")


def _reject_public_private_inode_overlap(root: Path) -> None:
    def identities(paths: Iterable[str]) -> set[tuple[int, int]]:
        result: set[tuple[int, int]] = set()
        for relative in paths:
            stat = _bundle_path(root, relative).stat()
            result.add((stat.st_dev, stat.st_ino))
        return result

    if identities(_PUBLIC_ARTIFACT_PATHS) & identities(_PRIVATE_ARTIFACT_PATHS):
        raise TaskCDataError("public artifact uses a private artifact hard link inode")


def _reject_bundle_symlinks(root: Path) -> None:
    for relative in (
        _PUBLIC_MANIFEST,
        _PRIVATE_MANIFEST,
        *_PUBLIC_ARTIFACT_PATHS,
        *_PRIVATE_ARTIFACT_PATHS,
    ):
        _bundle_path(root, relative)


def _materialized_sealed_holdout_semantic_content_sha256(root: Path) -> str:
    hasher = SealedHoldoutSemanticContentHasher()
    for relative in sorted(_PRIVATE_ARTIFACT_PATHS):
        context = (
            relative.split("/")[2]
            if relative.startswith("private/within/")
            else relative.split("/")[2].split("_to_")[1]
        )
        load_task_c_dataset(
            _bundle_path(root, relative),
            context_id=context,
            sealed_holdout_hasher=hasher,
            logical_artifact=relative,
        )
    return hasher.sha256()


def _reuse_existing_materialization(
    root: Path,
    identity: Mapping[str, Any],
    public_split_payload: Mapping[str, Any],
    private_split_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    public_path = root / _PUBLIC_MANIFEST
    private_path = root / _PRIVATE_MANIFEST
    if not root.exists():
        return None
    _reject_bundle_symlinks(root)
    try:
        has_entries = next(root.iterdir(), None) is not None
    except OSError as exc:
        raise TaskCDataError(f"cannot inspect output directory: {root}") from exc
    if not has_entries:
        return None
    if not public_path.is_file() or not private_path.is_file():
        raise TaskCDataError("existing output is incomplete; choose a new output directory")
    public = _read_manifest(public_path)
    private = _read_manifest(private_path)
    expected_public_fields = set(public_split_payload) | {
        "materialization_identity",
        "files",
    }
    expected_private_fields = set(private_split_payload) | {
        "materialization_identity",
        "files",
    }
    if set(public) != expected_public_fields:
        raise TaskCDataError(
            "existing public manifest schema is obsolete; rematerialize in a new output directory"
        )
    if set(private) != expected_private_fields:
        raise TaskCDataError(
            "existing private manifest schema is obsolete; rematerialize in a new output directory"
        )
    if public.get("materialization_identity") != identity or private.get(
        "materialization_identity"
    ) != identity:
        raise TaskCDataError("existing output has a different materialization identity")
    if any(public.get(key) != value for key, value in public_split_payload.items()):
        raise TaskCDataError("existing public manifest semantic record differs from split")
    if any(private.get(key) != value for key, value in private_split_payload.items()):
        raise TaskCDataError("existing private manifest semantic record differs from split")
    _verify_artifact_inventory(root, public, set(_PUBLIC_ARTIFACT_PATHS))
    _verify_artifact_inventory(root, private, set(_PRIVATE_ARTIFACT_PATHS))
    try:
        actual_commitment = _materialized_sealed_holdout_semantic_content_sha256(root)
    except TaskCDataError as exc:
        raise TaskCDataError(
            "existing sealed holdouts are invalid; rematerialize in a new output directory"
        ) from exc
    if actual_commitment != identity[SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD]:
        raise TaskCDataError(
            "existing sealed holdout semantic content changed; rematerialize in a new output directory"
        )
    _verify_artifact_inventory(root, private, set(_PRIVATE_ARTIFACT_PATHS))
    _reject_public_private_inode_overlap(root)
    return _materialized_result(root)


def _task_c_materialization_records(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    output_dir: str | Path,
    *,
    content_prevalidated: bool,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    _CommonGeneProjection,
]:
    if not content_prevalidated:
        _validate_task_c_dataset_pair_content(k562, rpe1)
    validate_task_c_split(split, k562, rpe1)
    if k562.expression.shape[1] != len(k562.gene_names) or rpe1.expression.shape[
        1
    ] != len(rpe1.gene_names):
        raise TaskCDataError("dataset gene names do not match expression columns")
    projection = _common_gene_projection(k562, rpe1)
    root = Path(output_dir).expanduser().resolve()
    identity = _materialization_identity(k562, rpe1, split, projection)
    public_split_payload = _public_split_payload(split, identity)
    private_split_payload = _private_split_payload(split)
    private_split_payload["content_sha256"] = identity["content_sha256"]
    private_split_payload["gene_names_sha256"] = identity["gene_names_sha256"]
    private_split_payload["gene_projection"] = identity["gene_projection"]
    private_split_payload[SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD] = identity[
        SEALED_HOLDOUT_SEMANTIC_CONTENT_FIELD
    ]
    return root, identity, public_split_payload, private_split_payload, projection


def _check_task_c_materialization(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    output_dir: str | Path,
    *,
    content_prevalidated: bool,
) -> str:
    root, identity, public_payload, private_payload, _ = _task_c_materialization_records(
        k562,
        rpe1,
        split,
        output_dir,
        content_prevalidated=content_prevalidated,
    )
    reused = _reuse_existing_materialization(
        root,
        identity,
        public_payload,
        private_payload,
    )
    return "reusable" if reused is not None else "missing"


def check_task_c_materialization(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    output_dir: str | Path,
) -> str:
    """Check a prospective bundle without creating directories or replacing files."""
    return _check_task_c_materialization(
        k562,
        rpe1,
        split,
        output_dir,
        content_prevalidated=False,
    )


def check_task_c_materializations(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    requests: Sequence[tuple[TaskCSplit, str | Path]],
) -> tuple[str, ...]:
    """Validate dataset content once, then preflight several fixed-seed bundles."""
    _validate_task_c_dataset_pair_content(k562, rpe1)
    return tuple(
        _check_task_c_materialization(
            k562,
            rpe1,
            split,
            output_dir,
            content_prevalidated=True,
        )
        for split, output_dir in requests
    )


def _materialize_task_c_split(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    output_dir: str | Path,
    *,
    content_prevalidated: bool,
) -> dict[str, Any]:
    root, identity, public_split_payload, private_split_payload, projection = (
        _task_c_materialization_records(
            k562,
            rpe1,
            split,
            output_dir,
            content_prevalidated=content_prevalidated,
        )
    )
    reused = _reuse_existing_materialization(
        root,
        identity,
        public_split_payload,
        private_split_payload,
    )
    if reused is not None:
        return reused
    _reject_bundle_symlinks(root)

    datasets = {"k562": k562, "rpe1": rpe1}
    result = _materialized_result(root)

    for context, dataset in datasets.items():
        controls = split.control_indices[context]
        for partition, sources, control_parts in (
            ("train", split.train_sources, ("train",)),
            ("tune", split.tune_sources, ("tune",)),
            ("refit", split.train_sources + split.tune_sources, ("train", "tune")),
            ("holdout", split.holdout_sources, ("holdout",)),
        ):
            control_indices = tuple(
                index for part in control_parts for index in controls[part]
            )
            path = _bundle_path(root, _WITHIN_ARTIFACTS[context][partition])
            _write_dataset_subset(
                dataset,
                _indices_for_sources(dataset, sources, control_indices),
                path,
                gene_names=projection.gene_names,
                column_indices=projection.column_indices[context],
            )

    for source_name, target_name in (("k562", "rpe1"), ("rpe1", "k562")):
        target = datasets[target_name]
        target_controls = split.control_indices[target_name]
        direction = f"{source_name}_to_{target_name}"
        for partition in ("source_train", "source_tune", "source_refit"):
            within_name = partition.removeprefix("source_")
            _atomic_link_or_copy(
                _bundle_path(root, _WITHIN_ARTIFACTS[source_name][within_name]),
                _bundle_path(root, _CROSS_ARTIFACTS[direction][partition]),
            )
        for partition, control_parts in (
            ("target_adapt_train", ("train",)),
            ("target_adapt_tune", ("tune",)),
            ("target_adapt_refit", ("train", "tune")),
        ):
            control_indices = tuple(
                index for part in control_parts for index in target_controls[part]
            )
            _write_dataset_subset(
                target,
                np.sort(_validated_row_indices(target, control_indices)),
                _bundle_path(root, _CROSS_ARTIFACTS[direction][partition]),
                gene_names=projection.gene_names,
                column_indices=projection.column_indices[target_name],
            )
        all_sources = split.train_sources + split.tune_sources + split.holdout_sources
        _write_dataset_subset(
            target,
            _indices_for_sources(target, all_sources, target_controls["holdout"]),
            _bundle_path(root, _CROSS_ARTIFACTS[direction]["target_holdout"]),
            gene_names=projection.gene_names,
            column_indices=projection.column_indices[target_name],
        )

    private_payload = dict(private_split_payload)
    private_payload.update(
        {
            "materialization_identity": identity,
            "files": _artifact_inventory(_PRIVATE_ARTIFACT_PATHS, root),
        }
    )
    public_payload = dict(public_split_payload)
    public_payload.update(
        {
            "materialization_identity": identity,
            "files": _artifact_inventory(_PUBLIC_ARTIFACT_PATHS, root),
        }
    )
    write_json(_bundle_path(root, _PRIVATE_MANIFEST), private_payload)
    write_json(_bundle_path(root, _PUBLIC_MANIFEST), public_payload)
    return result


def materialize_task_c_split(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    split: TaskCSplit,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write allowed model-building data separately from sealed evaluation data."""
    return _materialize_task_c_split(
        k562,
        rpe1,
        split,
        output_dir,
        content_prevalidated=False,
    )


def materialize_task_c_splits(
    k562: TaskCDataset,
    rpe1: TaskCDataset,
    requests: Sequence[tuple[TaskCSplit, str | Path]],
) -> tuple[dict[str, Any], ...]:
    """Preflight all requested bundles, then materialize them after one content check."""
    _validate_task_c_dataset_pair_content(k562, rpe1)
    for split, output_dir in requests:
        _check_task_c_materialization(
            k562,
            rpe1,
            split,
            output_dir,
            content_prevalidated=True,
        )
    return tuple(
        _materialize_task_c_split(
            k562,
            rpe1,
            split,
            output_dir,
            content_prevalidated=True,
        )
        for split, output_dir in requests
    )
