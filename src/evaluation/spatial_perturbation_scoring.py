"""Train-only standardisation and frozen hierarchical bridge scoring.

The module consumes Task 5 eligibility results, which in turn bind Task 6's
canonical neighbour artifact.  It never selects units from prediction output:
every scoreable neighbour and own-cell unit must have exactly one prediction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any, cast
import unicodedata

import numpy as np
from numpy.typing import NDArray

from src.evaluation.spatial_perturbation_split import (
    BridgeEligibilityResult,
    BridgeSplitManifest,
    eligibility_result_to_mapping,
    split_manifest_to_mapping,
)


MAX_EXPRESSION_ELEMENTS = 10_000_000
MAX_EXPRESSION_ROWS = 2_000_000
MAX_TRAINING_CONTROL_ROWS = 100_000
MAX_GENES = 50_000
MAX_EFFECT_UNITS = 100_000
MAX_ABSOLUTE_EFFECT = 1.0e12
TRAINING_HASH_CHUNK_BYTES = 1024 * 1024

_MAX_TEXT = 256
_SHA = re.compile(r"[0-9a-f]{64}")
_PRIMARY_BANDS = ("proximal", "local")
_ENDPOINTS = ("neighbor", "own")
_PCC_VARIANCE_FLOOR = 1.0e-24
_SUPPORTED_DTYPES = {
    np.dtype("int8"), np.dtype("int16"), np.dtype("int32"), np.dtype("int64"),
    np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64"),
    np.dtype("float32"), np.dtype("float64"),
}


class SpatialPerturbationScoringError(ValueError):
    """Scoring input does not match the frozen bridge contract."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialPerturbationScoringError(
            "value cannot be represented as canonical JSON"
        ) from error


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT:
        raise SpatialPerturbationScoringError(
            f"{name} must be bounded non-empty built-in text"
        )
    text = cast(str, value)
    if text != text.strip() or unicodedata.normalize("NFC", text) != text:
        raise SpatialPerturbationScoringError(f"{name} must be trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in text):
        raise SpatialPerturbationScoringError(f"{name} contains unsafe control text")
    return text


def _sha(value: object, name: str) -> str:
    text = _safe_text(value, name)
    if _SHA.fullmatch(text) is None:
        raise SpatialPerturbationScoringError(
            f"{name} must be a lowercase SHA-256 identity"
        )
    return text


def _text_tuple(
    value: object,
    name: str,
    *,
    maximum: int,
    unique: bool = False,
) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise SpatialPerturbationScoringError(
            f"{name} must be a built-in list or tuple"
        )
    raw = cast(list[object] | tuple[object, ...], value)
    if len(raw) > maximum:
        raise SpatialPerturbationScoringError(f"{name} exceeds the resource limit")
    result = tuple(_safe_text(item, f"{name}[{index}]") for index, item in enumerate(raw))
    if unique and len(set(result)) != len(result):
        raise SpatialPerturbationScoringError(f"{name} must contain unique values")
    return result


def _finite_float(value: object, name: str, *, bounded: bool = True) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise SpatialPerturbationScoringError(
            f"{name} must be a built-in real number"
        )
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise SpatialPerturbationScoringError(f"{name} must be finite")
    if bounded and abs(result) > MAX_ABSOLUTE_EFFECT:
        raise SpatialPerturbationScoringError(f"{name} exceeds the numeric bound")
    return result


def _nonnegative_integer(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise SpatialPerturbationScoringError(
            f"{name} must be a bounded nonnegative built-in integer"
        )
    return value


def _validate_expression(
    expression: object,
    *,
    expected_genes: int,
    allow_empty_rows: bool = False,
) -> NDArray[Any]:
    if type(expression) is not np.ndarray:
        raise SpatialPerturbationScoringError("expression must be a numpy.ndarray")
    array = cast(NDArray[Any], expression)
    if array.ndim != 2:
        raise SpatialPerturbationScoringError("expression must be a two-dimensional matrix")
    rows, genes = array.shape
    if genes != expected_genes:
        raise SpatialPerturbationScoringError(
            "expression shape does not match the exact gene order"
        )
    if genes == 0 or genes > MAX_GENES or rows > MAX_EXPRESSION_ROWS:
        raise SpatialPerturbationScoringError("expression exceeds the resource limit")
    if rows == 0 and not allow_empty_rows:
        raise SpatialPerturbationScoringError("expression must contain rows")
    if rows * genes > MAX_EXPRESSION_ELEMENTS:
        raise SpatialPerturbationScoringError("expression exceeds the resource limit")
    if array.dtype.newbyteorder("=") not in _SUPPORTED_DTYPES:
        raise SpatialPerturbationScoringError(
            "expression must have a supported real numeric dtype"
        )
    try:
        finite = bool(np.isfinite(array).all())
        negative = bool((array < 0).any())
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialPerturbationScoringError(
            "expression must have a supported real numeric dtype"
        ) from error
    if not finite:
        raise SpatialPerturbationScoringError("expression must contain finite values")
    if negative:
        raise SpatialPerturbationScoringError(
            "expression must contain nonnegative values before log1p"
        )
    return array


def _control_rows(value: object, row_count: int) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise SpatialPerturbationScoringError(
            "control_rows must be a built-in list or tuple"
        )
    raw = cast(list[object] | tuple[object, ...], value)
    if not raw:
        raise SpatialPerturbationScoringError("control_rows must be nonempty")
    if len(raw) > MAX_TRAINING_CONTROL_ROWS:
        raise SpatialPerturbationScoringError("control_rows exceeds the resource limit")
    result: list[int] = []
    for index, item in enumerate(raw):
        if type(item) is not int:
            raise SpatialPerturbationScoringError(
                f"control_rows[{index}] must be a built-in integer"
            )
        if item < 0 or item >= row_count:
            raise SpatialPerturbationScoringError(
                f"control_rows[{index}] is outside the expression row range"
            )
        result.append(item)
    if len(set(result)) != len(result):
        raise SpatialPerturbationScoringError("control_rows must contain unique rows")
    return tuple(result)


def _standardizer_unsigned(
    genes: tuple[str, ...],
    center: tuple[float, ...],
    scale: tuple[float, ...],
    control_rows: tuple[int, ...],
    training_shape: tuple[int, int],
    training_control_sha256: str,
    split_identity_sha256: str,
    training_cell_order_sha256: str,
    control_rows_sha256: str,
    control_cell_ids_sha256: str,
    control_roles_sha256: str,
    training_artifact_identity_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "train_control_standardizer_v1",
        "genes_sha256": _hash_text_order(genes, b"standardizer_genes_v1\0"),
        "center_sha256": _hash_float_order(center, b"standardizer_center_v1\0"),
        "scale_sha256": _hash_float_order(scale, b"standardizer_scale_v1\0"),
        "control_row_count": len(control_rows),
        "control_rows_sha256": control_rows_sha256,
        "training_shape": list(training_shape),
        "training_control_sha256": training_control_sha256,
        "split_identity_sha256": split_identity_sha256,
        "training_cell_order_sha256": training_cell_order_sha256,
        "control_cell_ids_sha256": control_cell_ids_sha256,
        "control_roles_sha256": control_roles_sha256,
        "training_artifact_identity_sha256": training_artifact_identity_sha256,
    }


def _snapshot_manifest(manifest: BridgeSplitManifest) -> BridgeSplitManifest:
    if type(manifest) is not BridgeSplitManifest:
        raise SpatialPerturbationScoringError(
            "split_manifest must be a Task 5 BridgeSplitManifest"
        )
    try:
        split_manifest_to_mapping(manifest)
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialPerturbationScoringError(
            "Task 5 split manifest failed revalidation"
        ) from error
    return manifest


def _hash_integer_order(values: tuple[int, ...], schema: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(schema)
    for value in values:
        digest.update(struct.pack(">Q", value))
    return digest.hexdigest()


def _hash_text_order(values: tuple[str, ...], schema: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(schema)
    buffer = bytearray()
    for value in values:
        encoded = value.encode("utf-8")
        item = struct.pack(">I", len(encoded)) + encoded
        if buffer and len(buffer) + len(item) > TRAINING_HASH_CHUNK_BYTES:
            digest.update(bytes(buffer))
            buffer.clear()
        buffer.extend(item)
    if buffer:
        digest.update(bytes(buffer))
    return digest.hexdigest()


def _hash_float_order(values: tuple[float, ...], schema: bytes) -> str:
    """Hash canonical little-endian float64 chunks without JSON expansion."""
    digest = hashlib.sha256()
    digest.update(schema)
    digest.update(struct.pack(">Q", len(values)))
    chunk_items = max(1, TRAINING_HASH_CHUNK_BYTES // 8)
    for offset in range(0, len(values), chunk_items):
        chunk = np.asarray(values[offset:offset + chunk_items], dtype=np.dtype("<f8"))
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _control_chunk_size(gene_count: int) -> int:
    return max(1, TRAINING_HASH_CHUNK_BYTES // max(8, gene_count * 8))


def _iter_control_chunks(
    expression: NDArray[Any], control_rows: tuple[int, ...]
) -> Any:
    chunk_size = _control_chunk_size(expression.shape[1])
    for offset in range(0, len(control_rows), chunk_size):
        indices = np.asarray(control_rows[offset:offset + chunk_size], dtype=np.intp)
        yield np.ascontiguousarray(expression[indices], dtype=np.dtype("<f8"))


def _hash_training_controls(
    expression: NDArray[Any], control_rows: tuple[int, ...]
) -> str:
    """Hash fixed little-endian float64 values without a monolithic payload."""
    digest = hashlib.sha256()
    digest.update(b"training_control_values_v2\0")
    digest.update(struct.pack(">QQ", len(control_rows), expression.shape[1]))
    for chunk in _iter_control_chunks(expression, control_rows):
        payload = memoryview(chunk).cast("B")
        for offset in range(0, len(payload), TRAINING_HASH_CHUNK_BYTES):
            digest.update(payload[offset:offset + TRAINING_HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _hash_training_artifact_chunks(
    chunks: tuple[bytes, ...], shape: tuple[int, int]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"training_control_values_v2\0")
    digest.update(struct.pack(">QQ", shape[0], shape[1]))
    for chunk in chunks:
        payload = memoryview(chunk)
        for offset in range(0, len(payload), TRAINING_HASH_CHUNK_BYTES):
            digest.update(payload[offset:offset + TRAINING_HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _validate_canonical_training_chunks(
    chunks: tuple[bytes, ...], shape: tuple[int, int]
) -> None:
    """Require the unique row-aligned layout produced by the frozen chunker."""
    rows, genes = shape
    if (
        type(rows) is not int
        or type(genes) is not int
        or rows <= 0
        or rows > MAX_TRAINING_CONTROL_ROWS
        or genes <= 0
        or genes > MAX_GENES
        or rows * genes > MAX_EXPRESSION_ELEMENTS
    ):
        raise SpatialPerturbationScoringError(
            "canonical chunk shape exceeds the resource limit"
        )
    if type(chunks) is not tuple:
        raise SpatialPerturbationScoringError("canonical chunks must be a tuple")
    rows_per_chunk = _control_chunk_size(genes)
    expected_count = (rows + rows_per_chunk - 1) // rows_per_chunk
    if len(chunks) != expected_count:
        raise SpatialPerturbationScoringError(
            "training artifact does not use the canonical chunk layout"
        )
    row_bytes = genes * 8
    for index, chunk in enumerate(chunks):
        if type(chunk) is not bytes:
            raise SpatialPerturbationScoringError("canonical chunks must contain bytes")
        expected_rows = (
            rows_per_chunk
            if index < expected_count - 1
            else rows - rows_per_chunk * (expected_count - 1)
        )
        if len(chunk) != expected_rows * row_bytes:
            raise SpatialPerturbationScoringError(
                "training artifact does not use the canonical chunk layout"
            )


def _hash_training_chunk_layout(
    chunks: tuple[bytes, ...], shape: tuple[int, int]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"training_control_chunk_layout_v1\0")
    digest.update(struct.pack(">QQQ", shape[0], shape[1], len(chunks)))
    for chunk in chunks:
        digest.update(struct.pack(">Q", len(chunk)))
    return digest.hexdigest()


def _training_artifact_unsigned(
    genes: tuple[str, ...],
    cell_ids: tuple[str, ...],
    roles: tuple[str, ...],
    shape: tuple[int, int],
    expression_sha256: str,
    split_identity_sha256: str,
    chunks: tuple[bytes, ...],
) -> dict[str, object]:
    return {
        "schema": "training_control_replay_v1",
        "genes_sha256": _hash_text_order(genes, b"training_genes_v1\0"),
        "cell_ids_sha256": _hash_text_order(cell_ids, b"training_cell_order_v1\0"),
        "roles_sha256": _hash_text_order(roles, b"control_roles_v1\0"),
        "expression_shape": list(shape),
        "expression_sha256": expression_sha256,
        "split_identity_sha256": split_identity_sha256,
        "chunk_layout_sha256": _hash_training_chunk_layout(chunks, shape),
    }


@dataclass(frozen=True, slots=True)
class TrainingControlArtifact:
    genes: tuple[str, ...]
    cell_ids: tuple[str, ...]
    roles: tuple[str, ...]
    split_manifest: BridgeSplitManifest
    split_identity_sha256: str
    expression_shape: tuple[int, int]
    expression_chunks: tuple[bytes, ...]
    expression_sha256: str
    artifact_identity_sha256: str

    def __post_init__(self) -> None:
        genes = _text_tuple(self.genes, "artifact.genes", maximum=MAX_GENES, unique=True)
        if not genes:
            raise SpatialPerturbationScoringError("artifact genes must be nonempty")
        cells = _text_tuple(
            self.cell_ids,
            "artifact.cell_ids",
            maximum=MAX_TRAINING_CONTROL_ROWS,
            unique=True,
        )
        roles = _text_tuple(
            self.roles, "artifact.roles", maximum=MAX_TRAINING_CONTROL_ROWS
        )
        if type(self.expression_shape) not in (list, tuple) or len(self.expression_shape) != 2:
            raise SpatialPerturbationScoringError("artifact expression_shape is invalid")
        raw_shape = cast(list[object] | tuple[object, ...], self.expression_shape)
        shape = (
            _nonnegative_integer(raw_shape[0], "artifact.expression_shape[0]", MAX_TRAINING_CONTROL_ROWS),
            _nonnegative_integer(raw_shape[1], "artifact.expression_shape[1]", MAX_GENES),
        )
        if shape[0] == 0 or shape[1] != len(genes) or len(cells) != shape[0] or len(roles) != shape[0]:
            raise SpatialPerturbationScoringError(
                "artifact shape does not match genes, cells, and roles"
            )
        if shape[0] * shape[1] > MAX_EXPRESSION_ELEMENTS:
            raise SpatialPerturbationScoringError("artifact exceeds the resource limit")
        if type(self.expression_chunks) not in (list, tuple):
            raise SpatialPerturbationScoringError("artifact chunks must be a sequence")
        raw_chunks = cast(list[object] | tuple[object, ...], self.expression_chunks)
        if not raw_chunks or len(raw_chunks) > shape[0]:
            raise SpatialPerturbationScoringError("artifact chunks exceed the resource limit")
        candidate_chunks = tuple(raw_chunks)
        _validate_canonical_training_chunks(
            cast(tuple[bytes, ...], candidate_chunks), shape
        )
        row_bytes = shape[1] * 8
        chunks: list[bytes] = []
        total_bytes = 0
        for index, value in enumerate(raw_chunks):
            if type(value) is not bytes:
                raise SpatialPerturbationScoringError(
                    f"artifact.expression_chunks[{index}] must be bytes"
                )
            chunk = cast(bytes, value)
            if not chunk or len(chunk) > TRAINING_HASH_CHUNK_BYTES or len(chunk) % row_bytes:
                raise SpatialPerturbationScoringError("artifact chunk boundary is invalid")
            total_bytes += len(chunk)
            if total_bytes > shape[0] * row_bytes:
                raise SpatialPerturbationScoringError("artifact byte count is invalid")
            values = np.frombuffer(chunk, dtype=np.dtype("<f8"))
            if not bool(np.isfinite(values).all()) or bool((values < 0).any()):
                raise SpatialPerturbationScoringError(
                    "artifact expression must be finite and nonnegative"
                )
            chunks.append(chunk)
        if total_bytes != shape[0] * row_bytes:
            raise SpatialPerturbationScoringError("artifact byte count is invalid")
        manifest = _snapshot_manifest(self.split_manifest)
        split_identity = _sha(self.split_identity_sha256, "artifact.split_identity_sha256")
        if split_identity != manifest.split_identity_sha256 or genes != manifest.gene_names:
            raise SpatialPerturbationScoringError("artifact does not match split manifest")
        provenance = {row.cell_id: row for row in manifest.row_provenance}
        train_rows = set(manifest.train_rows)
        for cell_id, role in zip(cells, roles):
            row = provenance.get(cell_id)
            if (
                row is None
                or row.stable_row_id not in train_rows
                or row.cell_role != "safe_source"
                or row.observed_label != manifest.safe_control_label
                or row.context_perturbation_id != manifest.safe_control_label
                or role != "safe_source:mSafe:train"
            ):
                raise SpatialPerturbationScoringError(
                    "artifact rows must be frozen train mSafe controls"
                )
        expression_identity = _sha(self.expression_sha256, "artifact.expression_sha256")
        if expression_identity != _hash_training_artifact_chunks(tuple(chunks), shape):
            raise SpatialPerturbationScoringError("artifact expression hash does not replay")
        identity = _sha(self.artifact_identity_sha256, "artifact_identity_sha256")
        expected = _identity(
            _training_artifact_unsigned(
                genes,
                cells,
                roles,
                shape,
                expression_identity,
                split_identity,
                tuple(chunks),
            )
        )
        if identity != expected:
            raise SpatialPerturbationScoringError("artifact identity does not replay")
        for name, value in (
            ("genes", genes), ("cell_ids", cells), ("roles", roles),
            ("split_manifest", manifest), ("split_identity_sha256", split_identity),
            ("expression_shape", shape), ("expression_chunks", tuple(chunks)),
            ("expression_sha256", expression_identity),
            ("artifact_identity_sha256", identity),
        ):
            object.__setattr__(self, name, value)


def _snapshot_training_artifact(
    artifact: TrainingControlArtifact,
) -> TrainingControlArtifact:
    if type(artifact) is not TrainingControlArtifact:
        raise SpatialPerturbationScoringError(
            "training_artifact must be TrainingControlArtifact"
        )
    return TrainingControlArtifact(
        artifact.genes,
        artifact.cell_ids,
        artifact.roles,
        artifact.split_manifest,
        artifact.split_identity_sha256,
        artifact.expression_shape,
        artifact.expression_chunks,
        artifact.expression_sha256,
        artifact.artifact_identity_sha256,
    )


def _statistics_from_training_artifact(
    artifact: TrainingControlArtifact,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gene_count = artifact.expression_shape[1]
    sums = np.zeros(gene_count, dtype=np.float64)
    for chunk in artifact.expression_chunks:
        values = np.frombuffer(chunk, dtype=np.dtype("<f8")).reshape(-1, gene_count)
        sums += np.log1p(values).sum(axis=0, dtype=np.float64)
    center_array = sums / artifact.expression_shape[0]
    squared = np.zeros(gene_count, dtype=np.float64)
    for chunk in artifact.expression_chunks:
        values = np.frombuffer(chunk, dtype=np.dtype("<f8")).reshape(-1, gene_count)
        logged = np.log1p(values)
        squared += np.square(logged - center_array).sum(axis=0, dtype=np.float64)
    scale_array = np.sqrt(squared / artifact.expression_shape[0], dtype=np.float64)
    scale_array[scale_array <= 1.0e-6] = 1.0
    return tuple(map(float, center_array)), tuple(map(float, scale_array))


def _training_statistics(
    expression: NDArray[Any], control_rows: tuple[int, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gene_count = expression.shape[1]
    sums = np.zeros(gene_count, dtype=np.float64)
    for chunk in _iter_control_chunks(expression, control_rows):
        logged = np.log1p(chunk)
        sums += logged.sum(axis=0, dtype=np.float64)
    center_array = sums / len(control_rows)
    squared = np.zeros(gene_count, dtype=np.float64)
    for chunk in _iter_control_chunks(expression, control_rows):
        logged = np.log1p(chunk)
        squared += np.square(logged - center_array).sum(axis=0, dtype=np.float64)
    scale_array = np.sqrt(squared / len(control_rows), dtype=np.float64)
    scale_array[scale_array <= 1.0e-6] = 1.0
    return (
        tuple(float(item) for item in center_array),
        tuple(float(item) for item in scale_array),
    )


@dataclass(frozen=True, slots=True)
class TrainControlStandardizer:
    genes: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    training_identity_sha256: str
    control_rows: tuple[int, ...]
    training_shape: tuple[int, int]
    training_control_sha256: str
    split_manifest: BridgeSplitManifest
    split_identity_sha256: str
    training_cell_ids: tuple[str, ...]
    training_cell_order_sha256: str
    control_cell_ids: tuple[str, ...]
    control_roles: tuple[str, ...]
    training_artifact: TrainingControlArtifact

    def __post_init__(self) -> None:
        genes = _text_tuple(self.genes, "genes", maximum=MAX_GENES, unique=True)
        if not genes:
            raise SpatialPerturbationScoringError("genes must be nonempty")
        center = _float_sequence(self.center, "center", len(genes), positive=False)
        scale = _float_sequence(self.scale, "scale", len(genes), positive=True)
        if type(self.training_shape) not in (list, tuple) or len(self.training_shape) != 2:
            raise SpatialPerturbationScoringError("training_shape must have two dimensions")
        raw_shape = cast(list[object] | tuple[object, ...], self.training_shape)
        shape = (
            _nonnegative_integer(raw_shape[0], "training_shape[0]", MAX_EXPRESSION_ROWS),
            _nonnegative_integer(raw_shape[1], "training_shape[1]", MAX_GENES),
        )
        if shape[0] == 0 or shape[1] != len(genes):
            raise SpatialPerturbationScoringError(
                "training_shape does not match the frozen gene order"
            )
        rows = _control_rows(self.control_rows, shape[0])
        if rows != tuple(range(shape[0])):
            raise SpatialPerturbationScoringError(
                "standardizer input must contain only frozen training control rows"
            )
        training_control_identity = _sha(
            self.training_control_sha256, "training_control_sha256"
        )
        manifest = _snapshot_manifest(self.split_manifest)
        split_identity = _sha(self.split_identity_sha256, "split_identity_sha256")
        if split_identity != manifest.split_identity_sha256:
            raise SpatialPerturbationScoringError(
                "standardizer split identity does not match its manifest"
            )
        if genes != manifest.gene_names:
            raise SpatialPerturbationScoringError(
                "standardizer genes do not match its split manifest"
            )
        training_cells = _text_tuple(
            self.training_cell_ids,
            "training_cell_ids",
            maximum=MAX_TRAINING_CONTROL_ROWS,
            unique=True,
        )
        if len(training_cells) != shape[0]:
            raise SpatialPerturbationScoringError(
                "training cell provenance does not match training_shape"
            )
        training_cell_order_identity = _sha(
            self.training_cell_order_sha256, "training_cell_order_sha256"
        )
        if training_cell_order_identity != _hash_text_order(
            training_cells, b"training_cell_order_v1\0"
        ):
            raise SpatialPerturbationScoringError(
                "training cell-order identity does not match provenance"
            )
        control_cells = _text_tuple(
            self.control_cell_ids,
            "control_cell_ids",
            maximum=MAX_TRAINING_CONTROL_ROWS,
            unique=True,
        )
        roles = _text_tuple(
            self.control_roles,
            "control_roles",
            maximum=MAX_TRAINING_CONTROL_ROWS,
        )
        if len(control_cells) != len(rows) or len(roles) != len(rows):
            raise SpatialPerturbationScoringError(
                "control provenance lengths do not match control_rows"
            )
        if control_cells != tuple(training_cells[index] for index in rows):
            raise SpatialPerturbationScoringError(
                "control cell provenance does not match control_rows"
            )
        provenance_by_cell = {row.cell_id: row for row in manifest.row_provenance}
        train_rows = set(manifest.train_rows)
        for index, (cell_id, role) in enumerate(zip(control_cells, roles)):
            provenance = provenance_by_cell.get(cell_id)
            if provenance is None:
                raise SpatialPerturbationScoringError(
                    "control cell is absent from frozen provenance"
                )
            if provenance.stable_row_id not in train_rows:
                raise SpatialPerturbationScoringError(
                    "control cell does not belong to frozen train rows"
                )
            if (
                provenance.cell_role != "safe_source"
                or provenance.context_perturbation_id != manifest.safe_control_label
                or provenance.observed_label != manifest.safe_control_label
            ):
                raise SpatialPerturbationScoringError(
                    "control cell must be a frozen mSafe safe_source"
                )
            if role != "safe_source:mSafe:train":
                raise SpatialPerturbationScoringError(
                    f"control_roles[{index}] does not match frozen provenance"
                )
        control_rows_identity = _hash_integer_order(rows, b"control_rows_v1\0")
        control_cells_identity = _hash_text_order(
            control_cells, b"control_cell_ids_v1\0"
        )
        control_roles_identity = _hash_text_order(roles, b"control_roles_v1\0")
        artifact = _snapshot_training_artifact(self.training_artifact)
        if (
            artifact.genes != genes
            or artifact.cell_ids != training_cells
            or artifact.roles != roles
            or artifact.expression_shape != shape
            or artifact.expression_sha256 != training_control_identity
            or artifact.split_identity_sha256 != split_identity
        ):
            raise SpatialPerturbationScoringError(
                "standardizer training artifact does not match provenance"
            )
        replay_center, replay_scale = _statistics_from_training_artifact(artifact)
        if center != replay_center or scale != replay_scale:
            raise SpatialPerturbationScoringError(
                "standardizer center/scale do not match training expression replay"
            )
        identity = _sha(self.training_identity_sha256, "training_identity_sha256")
        expected = _identity(
            _standardizer_unsigned(
                genes,
                center,
                scale,
                rows,
                shape,
                training_control_identity,
                split_identity,
                training_cell_order_identity,
                control_rows_identity,
                control_cells_identity,
                control_roles_identity,
                artifact.artifact_identity_sha256,
            )
        )
        if identity != expected:
            raise SpatialPerturbationScoringError(
                "standardizer identity does not match its training controls"
            )
        for name, value in (
            ("genes", genes), ("center", center), ("scale", scale),
            ("training_identity_sha256", identity), ("control_rows", rows),
            ("training_shape", shape),
            ("training_control_sha256", training_control_identity),
            ("split_manifest", manifest),
            ("split_identity_sha256", split_identity),
            ("training_cell_ids", training_cells),
            ("training_cell_order_sha256", training_cell_order_identity),
            ("control_cell_ids", control_cells),
            ("control_roles", roles),
            ("training_artifact", artifact),
        ):
            object.__setattr__(self, name, value)


def _float_sequence(
    value: object,
    name: str,
    length: int,
    *,
    positive: bool,
) -> tuple[float, ...]:
    if type(value) not in (list, tuple):
        raise SpatialPerturbationScoringError(f"{name} must be a built-in sequence")
    raw = cast(list[object] | tuple[object, ...], value)
    if len(raw) != length:
        raise SpatialPerturbationScoringError(f"{name} length does not match genes")
    result = tuple(_finite_float(item, f"{name}[{index}]", bounded=False) for index, item in enumerate(raw))
    if positive and any(item <= 0.0 for item in result):
        raise SpatialPerturbationScoringError(f"{name} must contain positive values")
    return result


def _snapshot_standardizer(
    standardizer: TrainControlStandardizer,
) -> TrainControlStandardizer:
    if type(standardizer) is not TrainControlStandardizer:
        raise SpatialPerturbationScoringError(
            "standardizer must be TrainControlStandardizer"
        )
    return TrainControlStandardizer(
        standardizer.genes,
        standardizer.center,
        standardizer.scale,
        standardizer.training_identity_sha256,
        standardizer.control_rows,
        standardizer.training_shape,
        standardizer.training_control_sha256,
        standardizer.split_manifest,
        standardizer.split_identity_sha256,
        standardizer.training_cell_ids,
        standardizer.training_cell_order_sha256,
        standardizer.control_cell_ids,
        standardizer.control_roles,
        standardizer.training_artifact,
    )


def fit_train_control_standardizer(
    expression: np.ndarray,
    *,
    gene_names: tuple[str, ...],
    control_rows: tuple[int, ...],
    cell_ids: tuple[str, ...],
    split_manifest: BridgeSplitManifest,
) -> TrainControlStandardizer:
    """Fit log1p mean and population SD using only declared training controls."""
    genes = _text_tuple(gene_names, "gene_names", maximum=MAX_GENES, unique=True)
    if not genes:
        raise SpatialPerturbationScoringError("gene_names must be nonempty")
    array = _validate_expression(expression, expected_genes=len(genes))
    rows = _control_rows(control_rows, array.shape[0])
    if rows != tuple(range(array.shape[0])):
        raise SpatialPerturbationScoringError(
            "fit input must contain only frozen training control rows"
        )
    manifest = _snapshot_manifest(split_manifest)
    if genes != manifest.gene_names:
        raise SpatialPerturbationScoringError(
            "training genes must match the split manifest's exact gene order"
        )
    cells = _text_tuple(
        cell_ids, "cell_ids", maximum=MAX_TRAINING_CONTROL_ROWS, unique=True
    )
    if len(cells) != array.shape[0]:
        raise SpatialPerturbationScoringError(
            "cell_ids length does not match expression rows"
        )
    provenance_by_cell = {item.cell_id: item for item in manifest.row_provenance}
    train_rows = set(manifest.train_rows)
    control_cells: list[str] = []
    for index in rows:
        cell_id = cells[index]
        provenance = provenance_by_cell.get(cell_id)
        if provenance is None:
            raise SpatialPerturbationScoringError(
                "control cell provenance is absent from the split manifest"
            )
        if provenance.stable_row_id not in train_rows:
            raise SpatialPerturbationScoringError(
                "control cells must belong only to frozen train rows"
            )
        if (
            provenance.cell_role != "safe_source"
            or provenance.context_perturbation_id != manifest.safe_control_label
            or provenance.observed_label != manifest.safe_control_label
        ):
            raise SpatialPerturbationScoringError(
                "training controls must be frozen mSafe safe_source rows"
            )
        control_cells.append(cell_id)
    control_cell_tuple = tuple(control_cells)
    roles = ("safe_source:mSafe:train",) * len(rows)
    shape = (int(array.shape[0]), int(array.shape[1]))
    training_cell_order_identity = _hash_text_order(
        cells, b"training_cell_order_v1\0"
    )
    control_rows_identity = _hash_integer_order(rows, b"control_rows_v1\0")
    control_cells_identity = _hash_text_order(
        control_cell_tuple, b"control_cell_ids_v1\0"
    )
    control_roles_identity = _hash_text_order(roles, b"control_roles_v1\0")
    chunks = tuple(
        bytes(memoryview(chunk).cast("B"))
        for chunk in _iter_control_chunks(array, rows)
    )
    training_control_identity = _hash_training_artifact_chunks(chunks, shape)
    artifact_unsigned = _training_artifact_unsigned(
        genes,
        cells,
        roles,
        shape,
        training_control_identity,
        manifest.split_identity_sha256,
        chunks,
    )
    artifact = TrainingControlArtifact(
        genes,
        cells,
        roles,
        manifest,
        manifest.split_identity_sha256,
        shape,
        chunks,
        training_control_identity,
        _identity(artifact_unsigned),
    )
    center, scale = _statistics_from_training_artifact(artifact)
    unsigned = _standardizer_unsigned(
        genes,
        center,
        scale,
        rows,
        shape,
        training_control_identity,
        manifest.split_identity_sha256,
        training_cell_order_identity,
        control_rows_identity,
        control_cells_identity,
        control_roles_identity,
        artifact.artifact_identity_sha256,
    )
    return TrainControlStandardizer(
        genes,
        center,
        scale,
        _identity(unsigned),
        rows,
        shape,
        training_control_identity,
        manifest,
        manifest.split_identity_sha256,
        cells,
        training_cell_order_identity,
        control_cell_tuple,
        roles,
        artifact,
    )


def apply_train_control_standardizer(
    expression: np.ndarray,
    *,
    gene_names: tuple[str, ...],
    standardizer: TrainControlStandardizer,
    split_manifest: BridgeSplitManifest,
) -> NDArray[np.float64]:
    """Apply a frozen standardizer without any tune/evaluation refitting path."""
    frozen = _snapshot_standardizer(standardizer)
    manifest = _snapshot_manifest(split_manifest)
    if manifest.split_identity_sha256 != frozen.split_identity_sha256:
        raise SpatialPerturbationScoringError(
            "application split identity does not match the training split"
        )
    genes = _text_tuple(gene_names, "gene_names", maximum=MAX_GENES, unique=True)
    if genes != frozen.genes:
        raise SpatialPerturbationScoringError(
            "application requires the standardizer's exact gene order"
        )
    array = _validate_expression(
        expression, expected_genes=len(genes), allow_empty_rows=True
    )
    logged = np.log1p(np.asarray(array, dtype=np.float64))
    result = (logged - np.asarray(frozen.center, dtype=np.float64)) / np.asarray(
        frozen.scale, dtype=np.float64
    )
    if not bool(np.isfinite(result).all()):
        raise SpatialPerturbationScoringError(
            "standardized expression must remain finite"
        )
    return cast(NDArray[np.float64], result)


def train_control_standardizer_to_mapping(
    standardizer: TrainControlStandardizer,
) -> dict[str, object]:
    frozen = _snapshot_standardizer(standardizer)
    return {
        "schema": "train_control_standardizer_v1",
        "genes": list(frozen.genes),
        "center": list(frozen.center),
        "scale": list(frozen.scale),
        "control_rows": list(frozen.control_rows),
        "training_shape": list(frozen.training_shape),
        "training_control_sha256": frozen.training_control_sha256,
        "split_identity_sha256": frozen.split_identity_sha256,
        "training_cell_order_sha256": frozen.training_cell_order_sha256,
        "control_row_count": len(frozen.control_rows),
        "control_rows_sha256": _hash_integer_order(
            frozen.control_rows, b"control_rows_v1\0"
        ),
        "control_cell_ids_sha256": _hash_text_order(
            frozen.control_cell_ids, b"control_cell_ids_v1\0"
        ),
        "control_roles_sha256": _hash_text_order(
            frozen.control_roles, b"control_roles_v1\0"
        ),
        "training_artifact": {
            "schema": "training_control_replay_v1",
            "artifact_identity_sha256": frozen.training_artifact.artifact_identity_sha256,
            "expression_shape": list(frozen.training_artifact.expression_shape),
            "expression_sha256": frozen.training_artifact.expression_sha256,
            "chunk_count": len(frozen.training_artifact.expression_chunks),
            "chunk_layout_sha256": _hash_training_chunk_layout(
                frozen.training_artifact.expression_chunks,
                frozen.training_artifact.expression_shape,
            ),
            "cell_ids_sha256": _hash_text_order(
                frozen.training_artifact.cell_ids, b"training_cell_order_v1\0"
            ),
            "roles_sha256": _hash_text_order(
                frozen.training_artifact.roles, b"control_roles_v1\0"
            ),
        },
        "training_identity_sha256": frozen.training_identity_sha256,
    }


@dataclass(frozen=True, slots=True)
class BridgePrediction:
    unit_id: str
    endpoint: str
    predicted_delta: float

    def __post_init__(self) -> None:
        unit = _sha(self.unit_id, "unit_id")
        endpoint = _safe_text(self.endpoint, "endpoint")
        if endpoint not in _ENDPOINTS:
            raise SpatialPerturbationScoringError("prediction endpoint is not frozen")
        predicted = _finite_float(self.predicted_delta, "predicted_delta")
        object.__setattr__(self, "unit_id", unit)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "predicted_delta", predicted)


def _effect_unsigned(effect: "BridgeEffect") -> dict[str, object]:
    return {
        "unit_id": effect.unit_id,
        "endpoint": effect.endpoint,
        "animal_id": effect.animal_id,
        "perturbation_id": effect.perturbation_id,
        "gene_name": effect.gene_name,
        "neighbor_cell_type": effect.neighbor_cell_type,
        "band": effect.band,
        "observed_delta_hex": effect.observed_delta.hex(),
        "predicted_delta_hex": effect.predicted_delta.hex(),
    }


@dataclass(frozen=True, slots=True)
class BridgeEffect:
    unit_id: str
    endpoint: str
    animal_id: str
    perturbation_id: str
    gene_name: str
    neighbor_cell_type: str
    band: str
    observed_delta: float
    predicted_delta: float
    effect_identity_sha256: str

    def __post_init__(self) -> None:
        unit = _sha(self.unit_id, "unit_id")
        endpoint = _safe_text(self.endpoint, "endpoint")
        if endpoint not in _ENDPOINTS:
            raise SpatialPerturbationScoringError("effect endpoint is not frozen")
        text_values = {
            name: _safe_text(getattr(self, name), name)
            for name in (
                "animal_id", "perturbation_id", "gene_name",
                "neighbor_cell_type", "band",
            )
        }
        if endpoint == "neighbor":
            if text_values["band"] not in _PRIMARY_BANDS:
                raise SpatialPerturbationScoringError(
                    "neighbor effects must use a primary band"
                )
            if text_values["neighbor_cell_type"] == "own":
                raise SpatialPerturbationScoringError(
                    "neighbor effects cannot use the own cell type"
                )
        elif (
            text_values["band"] != "own"
            or text_values["neighbor_cell_type"] != "own"
        ):
            raise SpatialPerturbationScoringError(
                "own effects must remain a separate own endpoint"
            )
        observed = _finite_float(self.observed_delta, "observed_delta")
        predicted = _finite_float(self.predicted_delta, "predicted_delta")
        identity = _sha(self.effect_identity_sha256, "effect_identity_sha256")
        for name, value in text_values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "unit_id", unit)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "observed_delta", observed)
        object.__setattr__(self, "predicted_delta", predicted)
        if identity != _identity(_effect_unsigned(self)):
            raise SpatialPerturbationScoringError(
                "effect identity does not match its frozen unit"
            )
        object.__setattr__(self, "effect_identity_sha256", identity)


def _make_effect(
    unit_id: str,
    endpoint: str,
    animal_id: str,
    perturbation_id: str,
    gene_name: str,
    neighbor_cell_type: str,
    band: str,
    observed_delta: float,
    predicted_delta: float,
) -> BridgeEffect:
    shell = object.__new__(BridgeEffect)
    for name, value in (
        ("unit_id", unit_id), ("endpoint", endpoint), ("animal_id", animal_id),
        ("perturbation_id", perturbation_id), ("gene_name", gene_name),
        ("neighbor_cell_type", neighbor_cell_type), ("band", band),
        ("observed_delta", float(observed_delta)),
        ("predicted_delta", float(predicted_delta)),
    ):
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "effect_identity_sha256", "0" * 64)
    identity = _identity(_effect_unsigned(shell))
    return BridgeEffect(
        unit_id, endpoint, animal_id, perturbation_id, gene_name,
        neighbor_cell_type, band, float(observed_delta), float(predicted_delta), identity,
    )


def _snapshot_effect(effect: BridgeEffect) -> BridgeEffect:
    if type(effect) is not BridgeEffect:
        raise SpatialPerturbationScoringError("effects must contain BridgeEffect values")
    return BridgeEffect(
        effect.unit_id, effect.endpoint, effect.animal_id, effect.perturbation_id,
        effect.gene_name, effect.neighbor_cell_type, effect.band,
        effect.observed_delta, effect.predicted_delta, effect.effect_identity_sha256,
    )


def _snapshot_eligibility(result: BridgeEligibilityResult) -> BridgeEligibilityResult:
    if type(result) is not BridgeEligibilityResult:
        raise SpatialPerturbationScoringError(
            "eligibility must be a Task 5 BridgeEligibilityResult"
        )
    try:
        eligibility_result_to_mapping(result)
    except (TypeError, ValueError, OverflowError) as error:
        raise SpatialPerturbationScoringError(
            "Task 5 eligibility artifact failed revalidation"
        ) from error
    return result


def _expected_keys(
    eligibility: BridgeEligibilityResult,
) -> tuple[set[tuple[str, str]], dict[str, Any], dict[str, Any]]:
    manifest = eligibility.manifest
    evaluation_animals = set(manifest.evaluation_animals)
    abstained = set(eligibility.abstained_unit_ids)
    evaluation_units = tuple(
        unit
        for unit in manifest.primary_units
        if unit.animal_id in evaluation_animals
    )
    units = {
        unit.unit_id: unit
        for unit in evaluation_units
        if unit.unit_id not in abstained
    }
    scoreable_parents = set(eligibility.scoreable_parent_ids)
    parents = {
        parent.parent_id: parent
        for parent in manifest.perturbation_parents
        if parent.animal_id in evaluation_animals
        and parent.parent_id in scoreable_parents
    }
    if not evaluation_units:
        raise SpatialPerturbationScoringError(
            "evaluation coverage denominator must be positive"
        )
    if len(units) * 5 < len(evaluation_units) * 4:
        raise SpatialPerturbationScoringError(
            "evaluation overall coverage is below 0.80"
        )
    for animal in sorted(evaluation_animals):
        animal_units = tuple(
            unit for unit in evaluation_units if unit.animal_id == animal
        )
        animal_scoreable = sum(unit.unit_id in units for unit in animal_units)
        if not animal_units or animal_scoreable * 5 < len(animal_units) * 4:
            raise SpatialPerturbationScoringError(
                f"evaluation animal coverage is below 0.80 for {animal}"
            )
        animal_parents = tuple(
            parent
            for parent in manifest.perturbation_parents
            if parent.animal_id == animal
        )
        animal_scoreable_parents = sum(
            parent.parent_id in parents for parent in animal_parents
        )
        if (
            not animal_parents
            or animal_scoreable_parents * 5 < len(animal_parents) * 4
        ):
            raise SpatialPerturbationScoringError(
                f"evaluation animal parent coverage is below 0.80 for {animal}"
            )
    keys = {("neighbor", unit_id) for unit_id in units}
    keys.update(("own", parent_id) for parent_id in parents)
    if not units or not parents:
        raise SpatialPerturbationScoringError(
            "scoring requires scoreable frozen neighbor and own units"
        )
    return keys, units, parents


def _effect_table_unsigned(table: "BridgeEffectTable") -> dict[str, object]:
    return {
        "schema": "bridge_effect_table_v1",
        "split_identity_sha256": table.split_identity_sha256,
        "neighbour_table_identity_sha256": table.neighbour_table_identity_sha256,
        "eligibility_identity_sha256": table.eligibility_identity_sha256,
        "standardizer_identity_sha256": table.standardizer_identity_sha256,
        "effects": [
            {
                **_effect_unsigned(effect),
                "effect_identity_sha256": effect.effect_identity_sha256,
            }
            for effect in table.effects
        ],
    }


@dataclass(frozen=True, slots=True)
class BridgeEffectTable:
    effects: tuple[BridgeEffect, ...]
    eligibility: BridgeEligibilityResult
    split_identity_sha256: str
    neighbour_table_identity_sha256: str
    eligibility_identity_sha256: str
    standardizer_identity_sha256: str
    effect_table_identity_sha256: str

    def __post_init__(self) -> None:
        eligibility = _snapshot_eligibility(self.eligibility)
        if type(self.effects) not in (list, tuple):
            raise SpatialPerturbationScoringError(
                "effects must be a built-in list or tuple"
            )
        if len(self.effects) > MAX_EFFECT_UNITS:
            raise SpatialPerturbationScoringError("effects exceeds the resource limit")
        effects = tuple(
            _snapshot_effect(effect) for effect in cast(tuple[BridgeEffect, ...], self.effects)
        )
        effects = tuple(sorted(effects, key=lambda item: (_ENDPOINTS.index(item.endpoint), item.unit_id)))
        keys = tuple((effect.endpoint, effect.unit_id) for effect in effects)
        if len(set(keys)) != len(keys):
            raise SpatialPerturbationScoringError("duplicate effects are forbidden")
        expected_keys, _, _ = _expected_keys(eligibility)
        if set(keys) != expected_keys:
            raise SpatialPerturbationScoringError(
                "effects do not contain the exact frozen Task 5 units"
            )
        unit_by_id = {
            unit.unit_id: unit
            for unit in eligibility.manifest.primary_units
            if unit.animal_id in set(eligibility.manifest.evaluation_animals)
        }
        parent_by_id = {
            parent.parent_id: parent
            for parent in eligibility.manifest.perturbation_parents
            if parent.animal_id in set(eligibility.manifest.evaluation_animals)
        }
        for effect in effects:
            if effect.endpoint == "neighbor":
                unit = unit_by_id[effect.unit_id]
                expected_context = (
                    unit.animal_id,
                    unit.perturbation_id,
                    unit.target_gene,
                    unit.neighbour_cell_type,
                    unit.band,
                )
            else:
                parent = parent_by_id[effect.unit_id]
                expected_context = (
                    parent.animal_id,
                    parent.perturbation_id,
                    parent.target_gene,
                    "own",
                    "own",
                )
            actual_context = (
                effect.animal_id,
                effect.perturbation_id,
                effect.gene_name,
                effect.neighbor_cell_type,
                effect.band,
            )
            if actual_context != expected_context:
                raise SpatialPerturbationScoringError(
                    "effect fields do not match their frozen Task 5 context"
                )
        split_identity = _sha(self.split_identity_sha256, "split_identity_sha256")
        neighbour_identity = _sha(
            self.neighbour_table_identity_sha256,
            "neighbour_table_identity_sha256",
        )
        eligibility_identity = _sha(
            self.eligibility_identity_sha256, "eligibility_identity_sha256"
        )
        standardizer_identity = _sha(
            self.standardizer_identity_sha256, "standardizer_identity_sha256"
        )
        if (
            split_identity != eligibility.manifest.split_identity_sha256
            or neighbour_identity
            != eligibility.manifest.neighbour_table_identity_sha256
            or eligibility_identity != eligibility.eligibility_identity_sha256
        ):
            raise SpatialPerturbationScoringError(
                "effect table identities do not match Task 5/6 artifacts"
            )
        table_identity = _sha(
            self.effect_table_identity_sha256, "effect_table_identity_sha256"
        )
        for name, value in (
            ("effects", effects), ("eligibility", eligibility),
            ("split_identity_sha256", split_identity),
            ("neighbour_table_identity_sha256", neighbour_identity),
            ("eligibility_identity_sha256", eligibility_identity),
            ("standardizer_identity_sha256", standardizer_identity),
        ):
            object.__setattr__(self, name, value)
        if table_identity != _identity(_effect_table_unsigned(self)):
            raise SpatialPerturbationScoringError(
                "effect table identity does not match its frozen effects"
            )
        object.__setattr__(self, "effect_table_identity_sha256", table_identity)


def _snapshot_effect_table(table: BridgeEffectTable) -> BridgeEffectTable:
    if type(table) is not BridgeEffectTable:
        raise SpatialPerturbationScoringError("effect_table must be BridgeEffectTable")
    return BridgeEffectTable(
        table.effects,
        table.eligibility,
        table.split_identity_sha256,
        table.neighbour_table_identity_sha256,
        table.eligibility_identity_sha256,
        table.standardizer_identity_sha256,
        table.effect_table_identity_sha256,
    )


def _prediction_snapshot(value: BridgePrediction, index: int) -> BridgePrediction:
    if type(value) is not BridgePrediction:
        raise SpatialPerturbationScoringError(
            f"predictions[{index}] must be BridgePrediction"
        )
    return BridgePrediction(value.unit_id, value.endpoint, value.predicted_delta)


def _mean_for_cells(
    standardized: NDArray[np.float64],
    row_by_cell: dict[str, int],
    cell_ids: tuple[str, ...],
    gene_index: int,
    name: str,
) -> float:
    unique = tuple(dict.fromkeys(cell_ids))
    if not unique:
        raise SpatialPerturbationScoringError(f"{name} has no frozen cells")
    missing = tuple(cell_id for cell_id in unique if cell_id not in row_by_cell)
    if missing:
        raise SpatialPerturbationScoringError(
            f"expression is missing frozen cells for {name}"
        )
    total = math.fsum(float(standardized[row_by_cell[cell_id], gene_index]) for cell_id in unique)
    return total / len(unique)


def build_bridge_effect_table(
    expression: np.ndarray,
    *,
    cell_ids: tuple[str, ...],
    gene_names: tuple[str, ...],
    standardizer: TrainControlStandardizer,
    eligibility: BridgeEligibilityResult,
    predictions: tuple[BridgePrediction, ...],
) -> BridgeEffectTable:
    """Build observed treatment-minus-mSafe effects on exact frozen units."""
    frozen_standardizer = _snapshot_standardizer(standardizer)
    frozen_eligibility = _snapshot_eligibility(eligibility)
    genes = _text_tuple(gene_names, "gene_names", maximum=MAX_GENES, unique=True)
    if genes != frozen_standardizer.genes or genes != frozen_eligibility.manifest.gene_names:
        raise SpatialPerturbationScoringError(
            "effect construction requires the exact frozen gene order"
        )
    array = _validate_expression(expression, expected_genes=len(genes))
    cells = _text_tuple(
        cell_ids, "cell_ids", maximum=MAX_EXPRESSION_ROWS, unique=True
    )
    if len(cells) != array.shape[0]:
        raise SpatialPerturbationScoringError(
            "cell_ids length does not match expression rows"
        )
    expected_keys, unit_by_id, parent_by_id = _expected_keys(frozen_eligibility)
    if type(predictions) not in (list, tuple):
        raise SpatialPerturbationScoringError(
            "predictions must be a built-in list or tuple"
        )
    if len(predictions) > MAX_EFFECT_UNITS:
        raise SpatialPerturbationScoringError("predictions exceeds the resource limit")
    frozen_predictions = tuple(
        _prediction_snapshot(item, index)
        for index, item in enumerate(cast(tuple[BridgePrediction, ...], predictions))
    )
    keys = tuple((item.endpoint, item.unit_id) for item in frozen_predictions)
    if len(set(keys)) != len(keys):
        raise SpatialPerturbationScoringError("duplicate predictions are forbidden")
    actual = set(keys)
    missing = expected_keys - actual
    extra = actual - expected_keys
    if missing:
        raise SpatialPerturbationScoringError(
            "missing predictions for frozen scoreable units"
        )
    if extra:
        raise SpatialPerturbationScoringError(
            "extra predictions outside frozen scoreable units"
        )
    prediction_by_key = {
        (item.endpoint, item.unit_id): item.predicted_delta
        for item in frozen_predictions
    }
    standardized = apply_train_control_standardizer(
        array,
        gene_names=genes,
        standardizer=frozen_standardizer,
        split_manifest=frozen_eligibility.manifest,
    )
    row_by_cell = {cell_id: index for index, cell_id in enumerate(cells)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    relation_by_id = {
        relation.relation_id: relation
        for relation in frozen_eligibility.manifest.neighbour_relations
    }
    unit_evidence = {
        item.unit_id: item for item in frozen_eligibility.evidence.unit_evidence
    }
    effects: list[BridgeEffect] = []
    for unit_id, unit in unit_by_id.items():
        primary_evidence = unit_evidence[unit_id]
        treatment_cells = tuple(
            relation_by_id[relation_id].neighbor_cell_id
            for relation_id in primary_evidence.perturbation_neighbour_relation_ids
        )
        safe_cells = tuple(
            relation_by_id[relation_id].neighbor_cell_id
            for relation_id in primary_evidence.safe_neighbour_relation_ids
        )
        column = gene_index[unit.target_gene]
        observed = _mean_for_cells(
            standardized, row_by_cell, treatment_cells, column,
            f"neighbor unit {unit_id}",
        ) - _mean_for_cells(
            standardized, row_by_cell, safe_cells, column,
            f"matched mSafe unit {unit_id}",
        )
        effects.append(
            _make_effect(
                unit_id, "neighbor", unit.animal_id, unit.perturbation_id,
                unit.target_gene, unit.neighbour_cell_type, unit.band, observed,
                prediction_by_key[("neighbor", unit_id)],
            )
        )
    parent_evidence = {
        (item.animal_id, item.perturbation_id): item
        for item in frozen_eligibility.evidence.parent_evidence
    }
    for parent_id, parent in parent_by_id.items():
        own_evidence = parent_evidence[(parent.animal_id, parent.perturbation_id)]
        column = gene_index[parent.target_gene]
        observed = _mean_for_cells(
            standardized,
            row_by_cell,
            own_evidence.perturbation_source_cell_ids,
            column,
            f"own perturbation unit {parent_id}",
        ) - _mean_for_cells(
            standardized,
            row_by_cell,
            own_evidence.safe_source_cell_ids,
            column,
            f"own mSafe unit {parent_id}",
        )
        effects.append(
            _make_effect(
                parent_id, "own", parent.animal_id, parent.perturbation_id,
                parent.target_gene, "own", "own", observed,
                prediction_by_key[("own", parent_id)],
            )
        )
    effects_tuple = tuple(sorted(effects, key=lambda item: (_ENDPOINTS.index(item.endpoint), item.unit_id)))
    shell = object.__new__(BridgeEffectTable)
    for name, value in (
        ("effects", effects_tuple), ("eligibility", frozen_eligibility),
        ("split_identity_sha256", frozen_eligibility.manifest.split_identity_sha256),
        ("neighbour_table_identity_sha256", frozen_eligibility.manifest.neighbour_table_identity_sha256),
        ("eligibility_identity_sha256", frozen_eligibility.eligibility_identity_sha256),
        ("standardizer_identity_sha256", frozen_standardizer.training_identity_sha256),
    ):
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "effect_table_identity_sha256", "0" * 64)
    table_identity = _identity(_effect_table_unsigned(shell))
    return BridgeEffectTable(
        effects_tuple,
        frozen_eligibility,
        frozen_eligibility.manifest.split_identity_sha256,
        frozen_eligibility.manifest.neighbour_table_identity_sha256,
        frozen_eligibility.eligibility_identity_sha256,
        frozen_standardizer.training_identity_sha256,
        table_identity,
    )


def bridge_effect_table_to_mapping(
    table: BridgeEffectTable,
    *,
    expression: np.ndarray,
    cell_ids: tuple[str, ...],
    gene_names: tuple[str, ...],
    standardizer: TrainControlStandardizer,
    eligibility: BridgeEligibilityResult,
    predictions: tuple[BridgePrediction, ...],
) -> dict[str, object]:
    """Serialize only after replaying the complete publication evidence."""
    frozen = _snapshot_effect_table(table)
    replayed = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    if frozen != replayed:
        raise SpatialPerturbationScoringError(
            "effect table does not match publication replay evidence"
        )
    mapping = _effect_table_unsigned(frozen)
    mapping["effects"] = [
        {
            "unit_id": effect.unit_id,
            "endpoint": effect.endpoint,
            "animal_id": effect.animal_id,
            "perturbation_id": effect.perturbation_id,
            "gene_name": effect.gene_name,
            "neighbor_cell_type": effect.neighbor_cell_type,
            "band": effect.band,
            "observed_delta": effect.observed_delta,
            "predicted_delta": effect.predicted_delta,
            "effect_identity_sha256": effect.effect_identity_sha256,
        }
        for effect in frozen.effects
    ]
    mapping["effect_table_identity_sha256"] = frozen.effect_table_identity_sha256
    return mapping


def _hierarchical_means(
    effects: tuple[BridgeEffect, ...],
    value: Callable[[BridgeEffect], float],
    dimensions: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    rows: list[tuple[dict[str, str], float]] = [
        (
            {
                "animal_id": item.animal_id,
                "perturbation_id": item.perturbation_id,
                "neighbor_cell_type": item.neighbor_cell_type,
                "band": item.band,
                "gene_name": item.gene_name,
            },
            value(item),
        )
        for item in effects
    ]
    for dimension in dimensions:
        grouped: dict[tuple[tuple[str, str], ...], list[float]] = {}
        contexts: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        for context, metric in rows:
            remaining = {name: item for name, item in context.items() if name != dimension}
            key = tuple(sorted(remaining.items()))
            grouped.setdefault(key, []).append(metric)
            contexts[key] = remaining
        rows = [
            (contexts[key], math.fsum(values) / len(values))
            for key, values in sorted(grouped.items())
        ]
    animal_values = {
        context["animal_id"]: metric
        for context, metric in rows
        if set(context) == {"animal_id"}
    }
    if not animal_values:
        raise SpatialPerturbationScoringError("hierarchical aggregation has no animals")
    return math.fsum(animal_values.values()) / len(animal_values), animal_values


def _hierarchical_primary_means(
    effects: tuple[BridgeEffect, ...],
    value: Callable[[BridgeEffect], float],
) -> tuple[float, dict[str, float]]:
    """Aggregate each frozen primary band separately, then give each half weight."""
    animals = {effect.animal_id for effect in effects}
    by_band_and_animal: dict[str, dict[str, float]] = {}
    for band in _PRIMARY_BANDS:
        band_effects = tuple(effect for effect in effects if effect.band == band)
        if not band_effects:
            raise SpatialPerturbationScoringError(
                "primary aggregation requires both frozen bands"
            )
        _, values = _hierarchical_means(
            band_effects,
            value,
            ("gene_name", "neighbor_cell_type", "perturbation_id", "band"),
        )
        if set(values) != animals:
            raise SpatialPerturbationScoringError(
                "each evaluation animal requires both frozen primary bands"
            )
        by_band_and_animal[band] = values
    animal_values = {
        animal: math.fsum(
            by_band_and_animal[band][animal] * 0.5 for band in _PRIMARY_BANDS
        )
        for animal in animals
    }
    return math.fsum(animal_values.values()) / len(animal_values), animal_values


def _neighbor_weights(effects: tuple[BridgeEffect, ...]) -> tuple[float, ...]:
    animals = {item.animal_id for item in effects}
    perturbations: dict[tuple[str, str], set[str]] = {}
    cell_types: dict[tuple[str, str, str], set[str]] = {}
    genes: dict[tuple[str, str, str, str], set[str]] = {}
    for item in effects:
        perturbations.setdefault((item.animal_id, item.band), set()).add(
            item.perturbation_id
        )
        cell_types.setdefault(
            (item.animal_id, item.band, item.perturbation_id), set()
        ).add(item.neighbor_cell_type)
        genes.setdefault(
            (
                item.animal_id,
                item.band,
                item.perturbation_id,
                item.neighbor_cell_type,
            ), set()
        ).add(item.gene_name)
    if any(
        (animal, band) not in perturbations
        for animal in animals
        for band in _PRIMARY_BANDS
    ):
        raise SpatialPerturbationScoringError(
            "each evaluation animal requires both frozen primary bands"
        )
    weights = []
    for item in effects:
        weight = 1.0 / len(animals)
        weight *= 0.5
        weight /= len(perturbations[(item.animal_id, item.band)])
        weight /= len(
            cell_types[(item.animal_id, item.band, item.perturbation_id)]
        )
        weight /= len(
            genes[
                (
                    item.animal_id,
                    item.band,
                    item.perturbation_id,
                    item.neighbor_cell_type,
                )
            ]
        )
        weights.append(weight)
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


def _weighted_pcc(effects: tuple[BridgeEffect, ...]) -> float:
    weights = _neighbor_weights(effects)
    observed = tuple(item.observed_delta for item in effects)
    predicted = tuple(item.predicted_delta for item in effects)
    mean_observed = math.fsum(w * x for w, x in zip(weights, observed))
    mean_predicted = math.fsum(w * x for w, x in zip(weights, predicted))
    variance_observed = math.fsum(
        w * (x - mean_observed) ** 2 for w, x in zip(weights, observed)
    )
    variance_predicted = math.fsum(
        w * (x - mean_predicted) ** 2 for w, x in zip(weights, predicted)
    )
    if (
        variance_observed <= _PCC_VARIANCE_FLOOR
        or variance_predicted <= _PCC_VARIANCE_FLOOR
    ):
        return 0.0
    covariance = math.fsum(
        w * (x - mean_observed) * (y - mean_predicted)
        for w, x, y in zip(weights, observed, predicted)
    )
    return min(1.0, max(-1.0, covariance / math.sqrt(variance_observed * variance_predicted)))


@dataclass(frozen=True, slots=True)
class _DistanceCalibration:
    error: float | None
    by_animal: dict[str, float | None]
    eligible_pairs: int
    total_contexts: int
    coverage: float
    abstention: float
    eligible_pairs_by_animal: dict[str, int]
    total_contexts_by_animal: dict[str, int]


def _distance_calibration(
    effects: tuple[BridgeEffect, ...],
    expected_contexts: tuple[tuple[str, str, str, str], ...] | None = None,
) -> _DistanceCalibration:
    by_context: dict[tuple[str, str, str, str], dict[str, BridgeEffect]] = {}
    for effect in effects:
        key = (
            effect.animal_id, effect.perturbation_id,
            effect.neighbor_cell_type, effect.gene_name,
        )
        by_context.setdefault(key, {})[effect.band] = effect
    if expected_contexts is None:
        frozen_contexts = tuple(sorted(by_context))
    else:
        frozen_contexts = tuple(sorted(set(expected_contexts)))
        if set(by_context) - set(frozen_contexts):
            raise SpatialPerturbationScoringError(
                "calibration effects fall outside frozen evaluation contexts"
            )
    pair_effects: list[BridgeEffect] = []
    errors: dict[str, float] = {}
    total_by_animal: dict[str, int] = {}
    eligible_by_animal: dict[str, int] = {}
    for key in frozen_contexts:
        bands = by_context.get(key, {})
        total_by_animal[key[0]] = total_by_animal.get(key[0], 0) + 1
        if set(bands) != set(_PRIMARY_BANDS):
            continue
        eligible_by_animal[key[0]] = eligible_by_animal.get(key[0], 0) + 1
        proximal = bands["proximal"]
        local = bands["local"]
        error = abs(
            (proximal.predicted_delta - local.predicted_delta)
            - (proximal.observed_delta - local.observed_delta)
        )
        synthetic = _make_effect(
            proximal.unit_id,
            "neighbor",
            key[0], key[1], key[3], key[2], "proximal", error, error,
        )
        pair_effects.append(synthetic)
        errors[synthetic.effect_identity_sha256] = error
    total = len(frozen_contexts)
    eligible = len(pair_effects)
    by_animal: dict[str, float | None] = {
        animal: None for animal in total_by_animal
    }
    overall_error: float | None = None
    if pair_effects:
        overall_error, computed_by_animal = _hierarchical_means(
            tuple(pair_effects),
            lambda item: errors[item.effect_identity_sha256],
            ("gene_name", "band", "neighbor_cell_type", "perturbation_id"),
        )
        by_animal.update(computed_by_animal)
    coverage = eligible / total if total else 0.0
    return _DistanceCalibration(
        overall_error,
        by_animal,
        eligible,
        total,
        coverage,
        1.0 - coverage,
        eligible_by_animal,
        total_by_animal,
    )


@dataclass(frozen=True, slots=True)
class BridgeAnimalUnit:
    animal_id: str
    neighbor_effect_rmse: float
    own_effect_rmse: float
    distance_decay_calibration_error: float | None
    distance_calibration_eligible_pairs: int
    distance_calibration_total_contexts: int
    neighbor_unit_count: int
    own_unit_count: int
    unit_identity_sha256: str

    def __post_init__(self) -> None:
        animal = _safe_text(self.animal_id, "animal_id")
        neighbor_rmse = _finite_float(
            self.neighbor_effect_rmse, "neighbor_effect_rmse", bounded=False
        )
        own_rmse = _finite_float(self.own_effect_rmse, "own_effect_rmse", bounded=False)
        calibration = (
            None
            if self.distance_decay_calibration_error is None
            else _finite_float(
                self.distance_decay_calibration_error,
                "distance_decay_calibration_error",
                bounded=False,
            )
        )
        if neighbor_rmse < 0.0 or own_rmse < 0.0 or (
            calibration is not None and calibration < 0.0
        ):
            raise SpatialPerturbationScoringError("animal errors must be nonnegative")
        eligible_pairs = _nonnegative_integer(
            self.distance_calibration_eligible_pairs,
            "distance_calibration_eligible_pairs",
            MAX_EFFECT_UNITS,
        )
        total_contexts = _nonnegative_integer(
            self.distance_calibration_total_contexts,
            "distance_calibration_total_contexts",
            MAX_EFFECT_UNITS,
        )
        if eligible_pairs > total_contexts or (calibration is None) != (eligible_pairs == 0):
            raise SpatialPerturbationScoringError(
                "animal calibration availability does not match eligible pairs"
            )
        neighbor_count = _nonnegative_integer(
            self.neighbor_unit_count, "neighbor_unit_count", MAX_EFFECT_UNITS
        )
        own_count = _nonnegative_integer(
            self.own_unit_count, "own_unit_count", MAX_EFFECT_UNITS
        )
        identity = _sha(self.unit_identity_sha256, "unit_identity_sha256")
        unsigned = {
            "animal_id": animal,
            "neighbor_effect_rmse_hex": neighbor_rmse.hex(),
            "own_effect_rmse_hex": own_rmse.hex(),
            "distance_decay_calibration_error_hex": (
                None if calibration is None else calibration.hex()
            ),
            "distance_calibration_eligible_pairs": eligible_pairs,
            "distance_calibration_total_contexts": total_contexts,
            "neighbor_unit_count": neighbor_count,
            "own_unit_count": own_count,
        }
        if identity != _identity(unsigned):
            raise SpatialPerturbationScoringError(
                "animal unit identity does not match its metrics"
            )
        for name, value in (
            ("animal_id", animal), ("neighbor_effect_rmse", neighbor_rmse),
            ("own_effect_rmse", own_rmse),
            ("distance_decay_calibration_error", calibration),
            ("distance_calibration_eligible_pairs", eligible_pairs),
            ("distance_calibration_total_contexts", total_contexts),
            ("neighbor_unit_count", neighbor_count), ("own_unit_count", own_count),
            ("unit_identity_sha256", identity),
        ):
            object.__setattr__(self, name, value)


def _make_animal_unit(
    animal: str,
    neighbor_rmse: float,
    own_rmse: float,
    calibration: float | None,
    calibration_eligible_pairs: int,
    calibration_total_contexts: int,
    neighbor_count: int,
    own_count: int,
) -> BridgeAnimalUnit:
    unsigned = {
        "animal_id": animal,
        "neighbor_effect_rmse_hex": neighbor_rmse.hex(),
        "own_effect_rmse_hex": own_rmse.hex(),
        "distance_decay_calibration_error_hex": (
            None if calibration is None else calibration.hex()
        ),
        "distance_calibration_eligible_pairs": calibration_eligible_pairs,
        "distance_calibration_total_contexts": calibration_total_contexts,
        "neighbor_unit_count": neighbor_count,
        "own_unit_count": own_count,
    }
    return BridgeAnimalUnit(
        animal, neighbor_rmse, own_rmse, calibration,
        calibration_eligible_pairs, calibration_total_contexts,
        neighbor_count, own_count, _identity(unsigned),
    )


@dataclass(frozen=True, slots=True)
class _ComputedScore:
    neighbor_effect_rmse: float
    own_effect_rmse: float
    neighbor_effect_pcc: float
    distance_decay_calibration_error: float | None
    distance_calibration_eligible_pairs: int
    distance_calibration_total_contexts: int
    distance_calibration_coverage: float
    distance_calibration_abstention: float
    effect_sign_accuracy: float
    coverage: float
    abstention: float
    animal_level_unit_table: tuple[BridgeAnimalUnit, ...]


def _compute_score(table: BridgeEffectTable) -> _ComputedScore:
    neighbor = tuple(item for item in table.effects if item.endpoint == "neighbor")
    own = tuple(item for item in table.effects if item.endpoint == "own")
    neighbor_mse, neighbor_by_animal = _hierarchical_primary_means(
        neighbor,
        lambda item: (item.predicted_delta - item.observed_delta) ** 2,
    )
    own_mse, own_by_animal = _hierarchical_means(
        own,
        lambda item: (item.predicted_delta - item.observed_delta) ** 2,
        ("gene_name", "band", "neighbor_cell_type", "perturbation_id"),
    )
    evaluation_animals = set(table.eligibility.manifest.evaluation_animals)
    calibration_contexts = tuple(
        (
            unit.animal_id,
            unit.perturbation_id,
            unit.neighbour_cell_type,
            unit.target_gene,
        )
        for unit in table.eligibility.manifest.primary_units
        if unit.animal_id in evaluation_animals
    )
    calibration = _distance_calibration(neighbor, calibration_contexts)
    weights = _neighbor_weights(neighbor)
    sign_accuracy = math.fsum(
        weight
        for weight, item in zip(weights, neighbor)
        if (item.observed_delta > 0) - (item.observed_delta < 0)
        == (item.predicted_delta > 0) - (item.predicted_delta < 0)
    )
    eligibility = table.eligibility
    evaluation_animals = set(eligibility.manifest.evaluation_animals)
    evaluation_units = tuple(
        unit for unit in eligibility.manifest.primary_units
        if unit.animal_id in evaluation_animals
    )
    if not evaluation_units:
        raise SpatialPerturbationScoringError("eligibility denominators must be positive")
    evaluation_scoreable = len(neighbor)
    coverage = evaluation_scoreable / len(evaluation_units)
    abstention = (len(evaluation_units) - evaluation_scoreable) / len(evaluation_units)
    animals = sorted(set(neighbor_by_animal) | set(own_by_animal))
    animal_units = tuple(
        _make_animal_unit(
            animal,
            math.sqrt(neighbor_by_animal[animal]),
            math.sqrt(own_by_animal[animal]),
            calibration.by_animal.get(animal),
            calibration.eligible_pairs_by_animal.get(animal, 0),
            calibration.total_contexts_by_animal.get(animal, 0),
            sum(item.animal_id == animal for item in neighbor),
            sum(item.animal_id == animal for item in own),
        )
        for animal in animals
    )
    return _ComputedScore(
        math.sqrt(neighbor_mse),
        math.sqrt(own_mse),
        _weighted_pcc(neighbor),
        calibration.error,
        calibration.eligible_pairs,
        calibration.total_contexts,
        calibration.coverage,
        calibration.abstention,
        min(1.0, max(0.0, sign_accuracy)),
        coverage,
        abstention,
        animal_units,
    )


def _score_unsigned(score: "BridgeScore") -> dict[str, object]:
    return {
        "schema": "bridge_score_v1",
        "neighbor_effect_rmse_hex": score.neighbor_effect_rmse.hex(),
        "own_effect_rmse_hex": score.own_effect_rmse.hex(),
        "neighbor_effect_pcc_hex": score.neighbor_effect_pcc.hex(),
        "distance_decay_calibration_error_hex": (
            None
            if score.distance_decay_calibration_error is None
            else score.distance_decay_calibration_error.hex()
        ),
        "distance_calibration_eligible_pairs": score.distance_calibration_eligible_pairs,
        "distance_calibration_total_contexts": score.distance_calibration_total_contexts,
        "distance_calibration_coverage_hex": score.distance_calibration_coverage.hex(),
        "distance_calibration_abstention_hex": score.distance_calibration_abstention.hex(),
        "effect_sign_accuracy_hex": score.effect_sign_accuracy.hex(),
        "coverage_hex": score.coverage.hex(),
        "abstention_hex": score.abstention.hex(),
        "animal_level_unit_table": [
            item.unit_identity_sha256 for item in score.animal_level_unit_table
        ],
        "split_identity_sha256": score.split_identity_sha256,
        "neighbour_table_identity_sha256": score.neighbour_table_identity_sha256,
        "eligibility_identity_sha256": score.eligibility_identity_sha256,
        "standardizer_identity_sha256": score.standardizer_identity_sha256,
        "effect_table_identity_sha256": score.effect_table_identity_sha256,
    }


@dataclass(frozen=True, slots=True)
class BridgeScore:
    neighbor_effect_rmse: float
    own_effect_rmse: float
    neighbor_effect_pcc: float
    distance_decay_calibration_error: float | None
    distance_calibration_eligible_pairs: int
    distance_calibration_total_contexts: int
    distance_calibration_coverage: float
    distance_calibration_abstention: float
    effect_sign_accuracy: float
    coverage: float
    abstention: float
    animal_level_unit_table: tuple[BridgeAnimalUnit, ...]
    split_identity_sha256: str
    neighbour_table_identity_sha256: str
    eligibility_identity_sha256: str
    standardizer_identity_sha256: str
    effect_table_identity_sha256: str
    effect_table: BridgeEffectTable
    scoring_identity_sha256: str

    def __post_init__(self) -> None:
        table = _snapshot_effect_table(self.effect_table)
        computed = _compute_score(table)
        numeric_names = (
            "neighbor_effect_rmse", "own_effect_rmse", "neighbor_effect_pcc",
            "distance_calibration_coverage", "distance_calibration_abstention",
            "effect_sign_accuracy", "coverage", "abstention",
        )
        numeric = tuple(
            _finite_float(getattr(self, name), name, bounded=False)
            for name in numeric_names
        )
        calibration = (
            None
            if self.distance_decay_calibration_error is None
            else _finite_float(
                self.distance_decay_calibration_error,
                "distance_decay_calibration_error",
                bounded=False,
            )
        )
        calibration_eligible = _nonnegative_integer(
            self.distance_calibration_eligible_pairs,
            "distance_calibration_eligible_pairs",
            MAX_EFFECT_UNITS,
        )
        calibration_total = _nonnegative_integer(
            self.distance_calibration_total_contexts,
            "distance_calibration_total_contexts",
            MAX_EFFECT_UNITS,
        )
        if numeric[0] < 0.0 or numeric[1] < 0.0 or (
            calibration is not None and calibration < 0.0
        ):
            raise SpatialPerturbationScoringError("score errors must be nonnegative")
        if not -1.0 <= numeric[2] <= 1.0:
            raise SpatialPerturbationScoringError("PCC must be within [-1, 1]")
        if any(not 0.0 <= numeric[index] <= 1.0 for index in (3, 4, 5, 6, 7)):
            raise SpatialPerturbationScoringError(
                "accuracy, coverage, and abstention must be within [0, 1]"
            )
        if (
            calibration_eligible > calibration_total
            or (calibration is None) != (calibration_eligible == 0)
            or numeric[3] != (
                calibration_eligible / calibration_total if calibration_total else 0.0
            )
            or numeric[4] != 1.0 - numeric[3]
        ):
            raise SpatialPerturbationScoringError(
                "distance calibration availability and coverage are inconsistent"
            )
        if type(self.animal_level_unit_table) not in (list, tuple):
            raise SpatialPerturbationScoringError(
                "animal_level_unit_table must be a built-in list or tuple"
            )
        animal_units = tuple(
            BridgeAnimalUnit(
                item.animal_id, item.neighbor_effect_rmse, item.own_effect_rmse,
                item.distance_decay_calibration_error,
                item.distance_calibration_eligible_pairs,
                item.distance_calibration_total_contexts,
                item.neighbor_unit_count,
                item.own_unit_count, item.unit_identity_sha256,
            )
            if type(item) is BridgeAnimalUnit
            else cast(BridgeAnimalUnit, item)
            for item in self.animal_level_unit_table
        )
        if any(type(item) is not BridgeAnimalUnit for item in animal_units):
            raise SpatialPerturbationScoringError(
                "animal_level_unit_table must contain BridgeAnimalUnit values"
            )
        animal_units = tuple(sorted(animal_units, key=lambda item: item.animal_id))
        if len({item.animal_id for item in animal_units}) != len(animal_units):
            raise SpatialPerturbationScoringError("animal units must be unique")
        expected_numeric = (
            computed.neighbor_effect_rmse, computed.own_effect_rmse,
            computed.neighbor_effect_pcc,
            computed.distance_calibration_coverage,
            computed.distance_calibration_abstention,
            computed.effect_sign_accuracy, computed.coverage, computed.abstention,
        )
        if (
            numeric != expected_numeric
            or calibration != computed.distance_decay_calibration_error
            or calibration_eligible != computed.distance_calibration_eligible_pairs
            or calibration_total != computed.distance_calibration_total_contexts
            or animal_units != computed.animal_level_unit_table
        ):
            raise SpatialPerturbationScoringError(
                "score metrics do not match hierarchical recomputation"
            )
        identities = (
            _sha(self.split_identity_sha256, "split_identity_sha256"),
            _sha(self.neighbour_table_identity_sha256, "neighbour_table_identity_sha256"),
            _sha(self.eligibility_identity_sha256, "eligibility_identity_sha256"),
            _sha(self.standardizer_identity_sha256, "standardizer_identity_sha256"),
            _sha(self.effect_table_identity_sha256, "effect_table_identity_sha256"),
        )
        if identities != (
            table.split_identity_sha256, table.neighbour_table_identity_sha256,
            table.eligibility_identity_sha256, table.standardizer_identity_sha256,
            table.effect_table_identity_sha256,
        ):
            raise SpatialPerturbationScoringError(
                "score identities do not match the effect table"
            )
        for name, value in zip(numeric_names, numeric):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "distance_decay_calibration_error", calibration)
        object.__setattr__(self, "distance_calibration_eligible_pairs", calibration_eligible)
        object.__setattr__(self, "distance_calibration_total_contexts", calibration_total)
        object.__setattr__(self, "animal_level_unit_table", animal_units)
        for identity_name, identity_value in zip(
            (
                "split_identity_sha256", "neighbour_table_identity_sha256",
                "eligibility_identity_sha256", "standardizer_identity_sha256",
                "effect_table_identity_sha256",
            ), identities,
        ):
            object.__setattr__(self, identity_name, identity_value)
        object.__setattr__(self, "effect_table", table)
        scoring_identity = _sha(
            self.scoring_identity_sha256, "scoring_identity_sha256"
        )
        if scoring_identity != _identity(_score_unsigned(self)):
            raise SpatialPerturbationScoringError(
                "score identity does not match its metrics and unit table"
            )
        object.__setattr__(self, "scoring_identity_sha256", scoring_identity)


def _score_effect_table(effect_table: BridgeEffectTable) -> BridgeScore:
    table = _snapshot_effect_table(effect_table)
    computed = _compute_score(table)
    shell = object.__new__(BridgeScore)
    values: tuple[tuple[str, object], ...] = (
        ("neighbor_effect_rmse", computed.neighbor_effect_rmse),
        ("own_effect_rmse", computed.own_effect_rmse),
        ("neighbor_effect_pcc", computed.neighbor_effect_pcc),
        ("distance_decay_calibration_error", computed.distance_decay_calibration_error),
        ("distance_calibration_eligible_pairs", computed.distance_calibration_eligible_pairs),
        ("distance_calibration_total_contexts", computed.distance_calibration_total_contexts),
        ("distance_calibration_coverage", computed.distance_calibration_coverage),
        ("distance_calibration_abstention", computed.distance_calibration_abstention),
        ("effect_sign_accuracy", computed.effect_sign_accuracy),
        ("coverage", computed.coverage), ("abstention", computed.abstention),
        ("animal_level_unit_table", computed.animal_level_unit_table),
        ("split_identity_sha256", table.split_identity_sha256),
        ("neighbour_table_identity_sha256", table.neighbour_table_identity_sha256),
        ("eligibility_identity_sha256", table.eligibility_identity_sha256),
        ("standardizer_identity_sha256", table.standardizer_identity_sha256),
        ("effect_table_identity_sha256", table.effect_table_identity_sha256),
        ("effect_table", table),
    )
    for name, value in values:
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "scoring_identity_sha256", "0" * 64)
    identity = _identity(_score_unsigned(shell))
    return BridgeScore(
        computed.neighbor_effect_rmse,
        computed.own_effect_rmse,
        computed.neighbor_effect_pcc,
        computed.distance_decay_calibration_error,
        computed.distance_calibration_eligible_pairs,
        computed.distance_calibration_total_contexts,
        computed.distance_calibration_coverage,
        computed.distance_calibration_abstention,
        computed.effect_sign_accuracy,
        computed.coverage,
        computed.abstention,
        computed.animal_level_unit_table,
        table.split_identity_sha256,
        table.neighbour_table_identity_sha256,
        table.eligibility_identity_sha256,
        table.standardizer_identity_sha256,
        table.effect_table_identity_sha256,
        table,
        identity,
    )


def score_bridge_predictions(
    expression: np.ndarray,
    *,
    cell_ids: tuple[str, ...],
    gene_names: tuple[str, ...],
    standardizer: TrainControlStandardizer,
    eligibility: BridgeEligibilityResult,
    predictions: tuple[BridgePrediction, ...],
) -> BridgeScore:
    """Rebuild observed effects and score exact held-out frozen units."""
    table = build_bridge_effect_table(
        expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    return _score_effect_table(table)


def bridge_score_to_mapping(
    score: BridgeScore,
    *,
    expression: np.ndarray,
    cell_ids: tuple[str, ...],
    gene_names: tuple[str, ...],
    standardizer: TrainControlStandardizer,
    eligibility: BridgeEligibilityResult,
    predictions: tuple[BridgePrediction, ...],
) -> dict[str, object]:
    """Return a JSON-ready snapshot after replaying all derived metrics."""
    if type(score) is not BridgeScore:
        raise SpatialPerturbationScoringError("score must be BridgeScore")
    frozen = BridgeScore(
        score.neighbor_effect_rmse,
        score.own_effect_rmse,
        score.neighbor_effect_pcc,
        score.distance_decay_calibration_error,
        score.distance_calibration_eligible_pairs,
        score.distance_calibration_total_contexts,
        score.distance_calibration_coverage,
        score.distance_calibration_abstention,
        score.effect_sign_accuracy,
        score.coverage,
        score.abstention,
        score.animal_level_unit_table,
        score.split_identity_sha256,
        score.neighbour_table_identity_sha256,
        score.eligibility_identity_sha256,
        score.standardizer_identity_sha256,
        score.effect_table_identity_sha256,
        score.effect_table,
        score.scoring_identity_sha256,
    )
    replayed = score_bridge_predictions(
        expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        predictions=predictions,
    )
    if frozen != replayed:
        raise SpatialPerturbationScoringError(
            "score does not match publication replay evidence"
        )
    return {
        "schema": "bridge_score_v1",
        "neighbor_effect_rmse": frozen.neighbor_effect_rmse,
        "own_effect_rmse": frozen.own_effect_rmse,
        "neighbor_effect_pcc": frozen.neighbor_effect_pcc,
        "distance_decay_calibration_error": frozen.distance_decay_calibration_error,
        "distance_calibration_eligible_pairs": frozen.distance_calibration_eligible_pairs,
        "distance_calibration_total_contexts": frozen.distance_calibration_total_contexts,
        "distance_calibration_coverage": frozen.distance_calibration_coverage,
        "distance_calibration_abstention": frozen.distance_calibration_abstention,
        "effect_sign_accuracy": frozen.effect_sign_accuracy,
        "coverage": frozen.coverage,
        "abstention": frozen.abstention,
        "animal_level_unit_table": [
            {
                "animal_id": item.animal_id,
                "neighbor_effect_rmse": item.neighbor_effect_rmse,
                "own_effect_rmse": item.own_effect_rmse,
                "distance_decay_calibration_error": item.distance_decay_calibration_error,
                "distance_calibration_eligible_pairs": item.distance_calibration_eligible_pairs,
                "distance_calibration_total_contexts": item.distance_calibration_total_contexts,
                "neighbor_unit_count": item.neighbor_unit_count,
                "own_unit_count": item.own_unit_count,
                "unit_identity_sha256": item.unit_identity_sha256,
            }
            for item in frozen.animal_level_unit_table
        ],
        "split_identity_sha256": frozen.split_identity_sha256,
        "neighbour_table_identity_sha256": frozen.neighbour_table_identity_sha256,
        "eligibility_identity_sha256": frozen.eligibility_identity_sha256,
        "standardizer_identity_sha256": frozen.standardizer_identity_sha256,
        "effect_table_identity_sha256": frozen.effect_table_identity_sha256,
        "scoring_identity_sha256": frozen.scoring_identity_sha256,
    }
