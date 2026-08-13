"""Build reproducible, size-bounded public inputs for Task C comparisons."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from typing import Any, Mapping, Sequence

import numpy as np


PROFILE_LIMITS: Mapping[str, tuple[int, int]] = {
    "connection": (64, 2_000),
    "comprehensive": (256, 20_000),
}
PROFILE_SELECTION_SEED = 11
PROFILE_SCHEMA = "task_c_profile_subset_v1"
CONTROL_LABEL = "non-targeting"
EXCLUDED_LABEL = "excluded"
WITHIN_TRANSFORMATION = "profile_gene_and_stratified_cell_subset_v1"
CROSS_TRANSFORMATION = "per_environment_control_zscore_then_row_concatenate_v1"
GENE_SELECTION_RULE = (
    "mean_context_control_population_variance_desc_then_gene_lexicographic_v1"
)
CELL_SELECTION_RULE = (
    "label_stratified_minimum_reserved_without_replacement_seed_11_v2"
)
MAXIMUM_FILE_BYTES = 512 * 1024 * 1024
MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
MAXIMUM_TOTAL_PARENT_BYTES = 1024 * 1024 * 1024
MAXIMUM_TOTAL_EXPANDED_PARENT_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_EXPANDED_PARENT_BYTES = 1024 * 1024 * 1024
MAXIMUM_PROFILE_INPUT_BYTES = 512 * 1024 * 1024
MAXIMUM_NPY_HEADER_BYTES = 64 * 1024
MAXIMUM_PARENT_CELLS = 1_000_000
MAXIMUM_PARENT_GENES = 1_000
MAXIMUM_PARENT_EXPRESSION_ELEMENTS = MAXIMUM_PARENT_CELLS * MAXIMUM_PARENT_GENES

_PUBLIC_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "train_sources",
        "tune_sources",
        "holdout_source_count",
        "input_sha256",
        "content_sha256",
        "gene_names_sha256",
        "gene_projection",
        "materialization_identity",
        "files",
    }
)
_PUBLIC_PATHS = frozenset(
    f"within/{context}/{partition}.npz"
    for context in ("k562", "rpe1")
    for partition in ("train", "tune", "refit")
) | frozenset(
    f"cross/{direction}/{partition}.npz"
    for direction in ("k562_to_rpe1", "rpe1_to_k562")
    for partition in (
        "source_train",
        "source_tune",
        "source_refit",
        "target_adapt_train",
        "target_adapt_tune",
        "target_adapt_refit",
    )
)
_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "profile_input_schema",
        "profile",
        "limits",
        "selection_seed",
        "condition",
        "context_id",
        "direction",
        "stage",
        "gene_selection",
        "contexts",
        "environment_labels",
        "transformation",
        "output",
    }
)


class TaskCProfileInputError(ValueError):
    """A profile subset cannot be rebuilt from the registered public parents."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes | None
    sha256_value: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    link_count: int

    @property
    def sha256(self) -> str:
        return self.sha256_value


@dataclass(frozen=True)
class TaskCProfileInput:
    profile: str
    condition: str
    stage: str
    context_id: str | None
    direction: str | None
    expression: np.ndarray
    interventions: np.ndarray
    gene_names: tuple[str, ...]
    environment_labels: np.ndarray | None
    input_path: Path
    manifest_path: Path
    input_sha256: str
    manifest_sha256: str
    public_manifest_sha256: str
    parent_paths: tuple[Path, ...]
    parent_sha256: tuple[str, ...]
    manifest: Mapping[str, object]
    input_snapshot: _FileSnapshot
    manifest_snapshot: _FileSnapshot
    public_snapshot: _FileSnapshot
    parent_snapshots: tuple[_FileSnapshot, ...]


@dataclass(frozen=True)
class _BuiltProfile:
    input_bytes: bytes
    manifest: dict[str, object]
    expression: np.ndarray
    interventions: np.ndarray
    genes: tuple[str, ...]
    environments: np.ndarray | None
    public_snapshot: _FileSnapshot
    parent_snapshots: tuple[_FileSnapshot, ...]


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _has_private_part(path: Path) -> bool:
    return any(part.casefold().startswith("private") for part in path.parts)


def _reject_symlink_parts(path: Path, label: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise TaskCProfileInputError(f"{label} does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCProfileInputError(f"{label} must not use a symbolic link")


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_nlink),
    )


def _capture(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    single_link: bool = False,
    collect_bytes: bool = True,
) -> _FileSnapshot:
    absolute = _absolute(path)
    if _has_private_part(absolute):
        raise TaskCProfileInputError(f"{label} must not use private data")
    _reject_symlink_parts(absolute, label)
    try:
        descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise TaskCProfileInputError(f"{label} must be a regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise TaskCProfileInputError(f"{label} must be a nonempty regular file")
        if before.st_size > maximum_bytes:
            raise TaskCProfileInputError(f"{label} is too large")
        if single_link and before.st_nlink != 1:
            raise TaskCProfileInputError(f"{label} must not be a hard link")
        payload = bytearray() if collect_bytes else None
        digest = hashlib.sha256()
        bytes_read = 0
        while bytes_read <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - bytes_read),
            )
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
            if payload is not None:
                payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise TaskCProfileInputError(f"{label} changed while being read") from exc
    if (
        bytes_read != before.st_size
        or bytes_read > maximum_bytes
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(current)
    ):
        raise TaskCProfileInputError(f"{label} changed while being read")
    return _FileSnapshot(
        path=absolute,
        payload=bytes(payload) if payload is not None else None,
        sha256_value=f"sha256:{digest.hexdigest()}",
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        modified_ns=int(after.st_mtime_ns),
        changed_ns=int(after.st_ctime_ns),
        link_count=int(after.st_nlink),
    )


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskCProfileInputError(f"JSON contains duplicate field {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise TaskCProfileInputError(f"JSON contains non-finite value {value}")


def _json(snapshot: _FileSnapshot, label: str) -> dict[str, Any]:
    if snapshot.payload is None:
        raise TaskCProfileInputError(f"{label} bytes were not retained")
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except TaskCProfileInputError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TaskCProfileInputError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TaskCProfileInputError(f"{label} must be a JSON object")
    return value


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_public_manifest(
    path: Path,
) -> tuple[_FileSnapshot, dict[str, Any], dict[str, _FileSnapshot]]:
    manifest = _capture(
        path, "public manifest", maximum_bytes=MAXIMUM_MANIFEST_BYTES, single_link=True
    )
    payload = _json(manifest, "public manifest")
    if set(payload) != _PUBLIC_MANIFEST_FIELDS or payload.get("schema_version") != "1.0":
        raise TaskCProfileInputError("public manifest schema changed")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != _PUBLIC_PATHS:
        raise TaskCProfileInputError("public manifest inventory is incomplete")
    identity_fields = {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "input_sha256",
        "content_sha256",
        "gene_names_sha256",
        "gene_projection",
    }
    identity = payload.get("materialization_identity")
    if not isinstance(identity, dict) or set(identity) != identity_fields or any(
        identity.get(field) != payload.get(field) for field in identity_fields
    ):
        raise TaskCProfileInputError("public manifest identity changed")
    root = manifest.path.parent
    inventory: dict[str, _FileSnapshot] = {}
    snapshots_by_inode: dict[tuple[int, int], _FileSnapshot] = {}
    inode_counts: Counter[tuple[int, int]] = Counter()
    for relative, expected_hash in sorted(files.items()):
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or _has_private_part(relative_path)
            or not isinstance(expected_hash, str)
        ):
            raise TaskCProfileInputError("public manifest contains an unsafe path")
        candidate = _absolute(root / relative)
        _reject_symlink_parts(candidate, f"registered public file {relative}")
        try:
            metadata = os.lstat(candidate)
        except OSError as exc:
            raise TaskCProfileInputError(
                f"registered public file is missing: {relative}"
            ) from exc
        inode = (int(metadata.st_dev), int(metadata.st_ino))
        canonical = snapshots_by_inode.get(inode)
        if canonical is None:
            canonical = _capture(
                candidate,
                f"registered public file {relative}",
                maximum_bytes=MAXIMUM_FILE_BYTES,
                collect_bytes=False,
            )
            if _identity(metadata) != (
                canonical.device,
                canonical.inode,
                canonical.size,
                canonical.modified_ns,
                canonical.changed_ns,
                canonical.link_count,
            ):
                raise TaskCProfileInputError(
                    f"registered public file changed before capture: {relative}"
                )
            snapshots_by_inode[inode] = canonical
        snapshot = _FileSnapshot(
            path=candidate,
            payload=None,
            sha256_value=canonical.sha256,
            device=canonical.device,
            inode=canonical.inode,
            size=canonical.size,
            modified_ns=canonical.modified_ns,
            changed_ns=canonical.changed_ns,
            link_count=canonical.link_count,
        )
        if snapshot.sha256 != expected_hash:
            raise TaskCProfileInputError(f"registered public hash changed: {relative}")
        inventory[relative] = snapshot
        inode_counts[(snapshot.device, snapshot.inode)] += 1
    for relative, snapshot in inventory.items():
        if snapshot.link_count != inode_counts[(snapshot.device, snapshot.inode)]:
            raise TaskCProfileInputError(
                f"registered public file has an undeclared hard link: {relative}"
            )
    return manifest, payload, inventory


def _text_vector(values: np.ndarray, label: str) -> tuple[str, ...]:
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise TaskCProfileInputError(f"{label} must be a one-dimensional text array")
    try:
        result = tuple(
            value.decode("utf-8", errors="strict") if isinstance(value, bytes) else str(value)
            for value in values.tolist()
        )
    except UnicodeError as exc:
        raise TaskCProfileInputError(f"{label} must use UTF-8") from exc
    if any(
        not value
        or value != value.strip()
        or not unicodedata.is_normalized("NFC", value)
        for value in result
    ):
        raise TaskCProfileInputError(f"{label} contains invalid text")
    return result


def _preflight_parent_archive(payload: bytes) -> None:
    expected_names = {
        "expression_matrix.npy",
        "interventions.npy",
        "var_names.npy",
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise TaskCProfileInputError("public parent ZIP has duplicate members")
            if set(names) != expected_names or len(names) != len(expected_names):
                raise TaskCProfileInputError("public parent ZIP members changed")
            if any(member.is_dir() or member.flag_bits & 0x1 for member in members):
                raise TaskCProfileInputError(
                    "public parent ZIP members must be regular and unencrypted"
                )
            if sum(int(member.file_size) for member in members) > MAXIMUM_EXPANDED_PARENT_BYTES:
                raise TaskCProfileInputError("public parent expanded members are too large")
            total_array_bytes = 0
            headers: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}
            for member in members:
                with archive.open(member, mode="r") as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(
                            handle,
                            max_header_size=MAXIMUM_NPY_HEADER_BYTES,
                        )
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(
                            handle,
                            max_header_size=MAXIMUM_NPY_HEADER_BYTES,
                        )
                    else:
                        raise TaskCProfileInputError(
                            "public parent NPY version is not allowed"
                        )
                    dtype = np.dtype(dtype)
                    if dtype.hasobject:
                        raise TaskCProfileInputError(
                            "public parent NPY object arrays are not allowed"
                        )
                    element_count = 1
                    for dimension in shape:
                        if (
                            isinstance(dimension, bool)
                            or not isinstance(dimension, int)
                            or dimension < 0
                            or (
                                dimension
                                and element_count
                                > MAXIMUM_PARENT_EXPRESSION_ELEMENTS // dimension
                            )
                        ):
                            raise TaskCProfileInputError(
                                "public parent NPY element count exceeds the limit"
                            )
                        element_count *= dimension
                    if dtype.itemsize < 1 or (
                        element_count
                        > MAXIMUM_EXPANDED_PARENT_BYTES // int(dtype.itemsize)
                    ):
                        raise TaskCProfileInputError(
                            "public parent NPY dtype or array bytes exceed the limit"
                        )
                    array_bytes = element_count * int(dtype.itemsize)
                    if handle.tell() > MAXIMUM_NPY_HEADER_BYTES + 16:
                        raise TaskCProfileInputError(
                            "public parent NPY header exceeds the limit"
                        )
                    if handle.tell() + array_bytes != int(member.file_size):
                        raise TaskCProfileInputError(
                            "public parent NPY header disagrees with member size"
                        )
                    total_array_bytes += array_bytes
                    if total_array_bytes > MAXIMUM_EXPANDED_PARENT_BYTES:
                        raise TaskCProfileInputError(
                            "public parent expanded array bytes exceed the limit"
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
                or not 1 <= expression_shape[0] <= MAXIMUM_PARENT_CELLS
                or not 2 <= expression_shape[1] <= MAXIMUM_PARENT_GENES
                or expression_elements > MAXIMUM_PARENT_EXPRESSION_ELEMENTS
            ):
                raise TaskCProfileInputError(
                    "public parent expression header has an unsafe shape or dtype"
                )
            if (
                len(labels_shape) != 1
                or labels_dtype.kind not in {"U", "S"}
                or labels_dtype.itemsize < 1
                or labels_shape[0] != expression_shape[0]
                or labels_elements > MAXIMUM_PARENT_CELLS
            ):
                raise TaskCProfileInputError(
                    "public parent intervention text header disagrees with expression rows"
                )
            if (
                len(genes_shape) != 1
                or genes_dtype.kind not in {"U", "S"}
                or genes_dtype.itemsize < 1
                or genes_shape[0] != expression_shape[1]
                or genes_elements > MAXIMUM_PARENT_GENES
            ):
                raise TaskCProfileInputError(
                    "public parent gene text header disagrees with expression columns"
                )
    except TaskCProfileInputError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise TaskCProfileInputError("public parent ZIP or NPY header is invalid") from exc


def _load_parent(
    snapshot: _FileSnapshot,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if snapshot.payload is None:
        raise TaskCProfileInputError("selected public parent bytes were not retained")
    _preflight_parent_archive(snapshot.payload)
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "expression_matrix",
                "interventions",
                "var_names",
            }:
                raise TaskCProfileInputError("public parent arrays changed")
            expression = np.asarray(archive["expression_matrix"])
            labels_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
    except TaskCProfileInputError:
        raise
    except (OSError, ValueError, TypeError, EOFError) as exc:
        raise TaskCProfileInputError("public parent cannot be read") from exc
    if (
        expression.ndim != 2
        or expression.shape[0] < 1
        or expression.shape[1] < 2
        or expression.dtype.kind not in {"i", "u", "f"}
        or not np.isfinite(expression).all()
    ):
        raise TaskCProfileInputError("public parent expression is invalid")
    labels = _text_vector(labels_raw, "intervention labels")
    genes = _text_vector(genes_raw, "gene names")
    if len(labels) != expression.shape[0] or len(genes) != expression.shape[1]:
        raise TaskCProfileInputError("public parent dimensions disagree")
    if len(set(genes)) != len(genes) or CONTROL_LABEL not in labels:
        raise TaskCProfileInputError("public parent genes or controls are invalid")
    if not set(labels) <= set(genes) | {CONTROL_LABEL, EXCLUDED_LABEL}:
        raise TaskCProfileInputError("public parent has an unknown intervention label")
    return np.asarray(expression), np.asarray(labels, dtype=str), genes


def _gene_names_sha256(genes: Sequence[str]) -> str:
    encoded = json.dumps(list(genes), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _profile_limits(profile: str) -> tuple[int, int]:
    if profile not in PROFILE_LIMITS:
        raise TaskCProfileInputError(
            "profile must be connection or comprehensive"
        )
    return PROFILE_LIMITS[profile]


def _selected_genes(
    inventory: Mapping[str, _FileSnapshot],
    gene_limit: int,
    *,
    cell_limit: int,
    eligible_sources: Sequence[str],
    minimum_cells: int,
    load_parent: Any,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    tuple[dict[str, object], ...],
    tuple[str, ...],
]:
    loaded = []
    parents = []
    for context in ("k562", "rpe1"):
        relative = f"within/{context}/refit.npz"
        snapshot = inventory[relative]
        expression, labels, genes = load_parent(snapshot)
        controls = labels == CONTROL_LABEL
        if int(np.count_nonzero(controls)) < 2:
            raise TaskCProfileInputError(
                f"{context} gene-selection parent needs at least two controls"
            )
        loaded.append((genes, np.asarray(expression[controls], dtype=np.float64)))
        parents.append(
            {
                "context_id": context,
                "public_relative_path": relative,
                "sha256": snapshot.sha256,
            }
        )
    genes = loaded[0][0]
    if loaded[1][0] != genes:
        raise TaskCProfileInputError("gene-selection parents use different gene orders")
    counts_by_context = []
    for context in ("k562", "rpe1"):
        _, labels, _ = load_parent(inventory[f"within/{context}/refit.npz"])
        counts_by_context.append(Counter(labels.tolist()))
    candidates = {
        source
        for source in eligible_sources
        if source in genes
        and all(counts.get(source, 0) >= minimum_cells for counts in counts_by_context)
    }
    if len(candidates) < 2:
        raise TaskCProfileInputError(
            "profile needs at least two shared intervention sources with enough cells"
        )
    mean_variance = np.mean(
        np.stack([values.var(axis=0, ddof=0) for _, values in loaded], axis=0),
        axis=0,
    )
    ranked = sorted(
        (index for index, gene in enumerate(genes) if gene in candidates),
        key=lambda index: (-mean_variance[index], genes[index]),
    )
    maximum_sources_by_cell_budget = cell_limit // minimum_cells - 1
    if maximum_sources_by_cell_budget < 2:
        raise TaskCProfileInputError(
            "cell cap cannot retain controls and at least two intervention sources"
        )
    selection_count = min(
        gene_limit,
        maximum_sources_by_cell_budget,
        len(ranked),
    )
    selected_indices = tuple(int(index) for index in ranked[:selection_count])
    if len(selected_indices) < 2:
        raise TaskCProfileInputError("profile needs at least two selected genes")
    otherwise_selected = ranked[: min(gene_limit, len(ranked))]
    dropped_due_to_cell_budget = tuple(
        genes[index] for index in otherwise_selected[selection_count:]
    )
    return (
        tuple(genes[index] for index in selected_indices),
        selected_indices,
        tuple(parents),
        dropped_due_to_cell_budget,
    )


def _quota_by_label(
    labels: np.ndarray,
    limit: int,
    *,
    minimum_per_label: int,
) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    if minimum_per_label < 1 or any(int(count) < minimum_per_label for count in counts):
        raise TaskCProfileInputError(
            "every retained label needs the public minimum number of cells"
        )
    required = minimum_per_label * len(unique)
    if required > limit:
        raise TaskCProfileInputError(
            "cell cap is smaller than the reserved label minimums"
        )
    count_by_label = {
        str(label): int(count) for label, count in zip(unique, counts)
    }
    quotas = {str(label): minimum_per_label for label in unique}
    remaining = limit - required
    capacities = {
        label: count - minimum_per_label for label, count in count_by_label.items()
    }
    capacity_total = sum(capacities.values())
    exact = {
        label: (remaining * capacity / capacity_total if capacity_total else 0.0)
        for label, capacity in capacities.items()
    }
    for label in quotas:
        quotas[label] += min(capacities[label], int(np.floor(exact[label])))
    unassigned = limit - sum(quotas.values())
    candidates = sorted(
        (label for label in quotas if quotas[label] < count_by_label[label]),
        key=lambda label: (-(exact[label] - np.floor(exact[label])), label),
    )
    for label in candidates[:unassigned]:
        quotas[label] += 1
    if sum(quotas.values()) != limit:
        raise TaskCProfileInputError("stratified cell quotas do not fill the cap")
    return quotas


def _stratified_cell_indices(
    labels: np.ndarray,
    *,
    limit: int,
    minimum_per_label: int = 1,
) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or limit < 1:
        raise TaskCProfileInputError("cell labels and cap are invalid")
    _, counts = np.unique(values, return_counts=True)
    if any(int(count) < minimum_per_label for count in counts):
        raise TaskCProfileInputError(
            "every retained label needs the public minimum number of cells"
        )
    if len(values) <= limit:
        return np.arange(len(values), dtype=np.int64)
    quotas = _quota_by_label(
        values,
        limit,
        minimum_per_label=minimum_per_label,
    )
    selected: list[int] = []
    for label in sorted(quotas):
        candidates = np.flatnonzero(values == label)
        digest = hashlib.sha256(
            f"{PROFILE_SELECTION_SEED}\0{label}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        chosen = rng.choice(candidates, size=quotas[label], replace=False)
        selected.extend(int(index) for index in chosen.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _counts(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def _actual_parent_records(
    *,
    condition: str,
    stage: str,
    context_id: str | None,
    direction: str | None,
) -> tuple[dict[str, str], ...]:
    if stage not in {"tune", "refit"}:
        raise TaskCProfileInputError("profile stage must be tune or refit")
    if condition == "within_environment":
        if context_id not in {"k562", "rpe1"} or direction is not None:
            raise TaskCProfileInputError("within profile needs one k562 or rpe1 context")
        return (
            {
                "role": f"within_{stage}",
                "context_id": context_id,
                "public_relative_path": f"within/{context_id}/{stage}.npz",
            },
        )
    if condition == "cross_environment":
        if direction not in {"k562_to_rpe1", "rpe1_to_k562"} or context_id is not None:
            raise TaskCProfileInputError("cross profile needs one fixed direction")
        source, target = direction.split("_to_", 1)
        return (
            {
                "role": f"source_{stage}",
                "context_id": source,
                "public_relative_path": f"cross/{direction}/source_{stage}.npz",
            },
            {
                "role": f"target_adapt_{stage}",
                "context_id": target,
                "public_relative_path": (
                    f"cross/{direction}/target_adapt_{stage}.npz"
                ),
            },
        )
    raise TaskCProfileInputError(
        "condition must be within_environment or cross_environment"
    )


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asarray(values)))
    return buffer.getvalue()


def _build_profile(
    *,
    public_manifest_path: Path,
    profile: str,
    condition: str,
    stage: str,
    context_id: str | None,
    direction: str | None,
) -> _BuiltProfile:
    gene_limit, cell_limit = _profile_limits(profile)
    parent_specs = _actual_parent_records(
        condition=condition,
        stage=stage,
        context_id=context_id,
        direction=direction,
    )
    public_snapshot, public_manifest, inventory = _load_public_manifest(
        public_manifest_path
    )
    parent_cache: dict[
        tuple[int, int],
        tuple[_FileSnapshot, tuple[np.ndarray, np.ndarray, tuple[str, ...]]],
    ] = {}
    total_parent_bytes = 0
    total_expanded_parent_bytes = 0

    def load_parent(
        inventory_snapshot: _FileSnapshot,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        nonlocal total_parent_bytes, total_expanded_parent_bytes
        key = (inventory_snapshot.device, inventory_snapshot.inode)
        cached = parent_cache.get(key)
        if cached is None:
            captured = _capture(
                inventory_snapshot.path,
                "selected public profile parent",
                maximum_bytes=MAXIMUM_FILE_BYTES,
                collect_bytes=True,
            )
            if (
                captured.sha256 != inventory_snapshot.sha256
                or captured.device != inventory_snapshot.device
                or captured.inode != inventory_snapshot.inode
                or captured.size != inventory_snapshot.size
            ):
                raise TaskCProfileInputError(
                    "selected public profile parent changed after inventory check"
                )
            total_parent_bytes += captured.size
            if total_parent_bytes > MAXIMUM_TOTAL_PARENT_BYTES:
                raise TaskCProfileInputError("selected profile parents are too large together")
            loaded_parent = _load_parent(captured)
            expression, labels, genes = loaded_parent
            expanded_bytes = (
                int(expression.nbytes)
                + int(labels.nbytes)
                + sum(len(gene.encode("utf-8")) for gene in genes)
            )
            total_expanded_parent_bytes += expanded_bytes
            if total_expanded_parent_bytes > MAXIMUM_TOTAL_EXPANDED_PARENT_BYTES:
                raise TaskCProfileInputError(
                    "selected profile parent expanded arrays are too large together"
                )
            parent_cache[key] = (captured, loaded_parent)
            return loaded_parent
        return cached[1]

    eligible_sources = tuple(public_manifest["train_sources"]) + tuple(
        public_manifest["tune_sources"]
    )
    minimum_cells = public_manifest["min_cells_per_intervention"]
    if (
        isinstance(minimum_cells, bool)
        or not isinstance(minimum_cells, int)
        or minimum_cells < 1
    ):
        raise TaskCProfileInputError("public minimum cell count is invalid")
    genes, gene_indices, gene_parents, dropped_due_to_cell_budget = _selected_genes(
        inventory,
        gene_limit,
        cell_limit=cell_limit,
        eligible_sources=eligible_sources,
        minimum_cells=minimum_cells,
        load_parent=load_parent,
    )
    expressions: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    environment_out: list[np.ndarray] = []
    context_records: list[dict[str, object]] = []
    actual_snapshots: list[_FileSnapshot] = []
    for spec in parent_specs:
        snapshot = inventory[spec["public_relative_path"]]
        expression, labels, parent_genes = load_parent(snapshot)
        if any(parent_genes[index] != gene for index, gene in zip(gene_indices, genes)):
            raise TaskCProfileInputError("actual parent gene order differs from selection parents")
        retained = np.isin(labels, np.asarray((CONTROL_LABEL, *genes), dtype=str))
        retained_indices = np.flatnonzero(retained)
        dropped_indices = np.flatnonzero(~retained)
        retained_labels = labels[retained_indices]
        relative_rows = _stratified_cell_indices(
            retained_labels,
            limit=cell_limit,
            minimum_per_label=minimum_cells,
        )
        rows = retained_indices[relative_rows]
        selected_expression = np.asarray(
            expression[np.ix_(rows, np.asarray(gene_indices, dtype=int))],
            dtype=np.float64,
        )
        selected_labels = np.asarray(labels[rows], dtype=str)
        if condition == "cross_environment":
            controls = selected_labels == CONTROL_LABEL
            if int(np.count_nonzero(controls)) < 2:
                raise TaskCProfileInputError(
                    f"{spec['context_id']} profile needs at least two selected controls"
                )
            control_values = selected_expression[controls]
            mean = control_values.mean(axis=0)
            scale = control_values.std(axis=0, ddof=0)
            selected_expression = (selected_expression - mean) / np.where(
                scale <= 1e-6, 1.0, scale
            )
        expressions.append(selected_expression)
        labels_out.append(selected_labels)
        environment_out.append(
            np.asarray([spec["context_id"]] * len(rows), dtype=str)
        )
        context_records.append(
            {
                **spec,
                "parent_sha256": snapshot.sha256,
                "row_filter_rule": (
                    "retain_control_and_selected_gene_interventions_v1"
                ),
                "dropped_original_row_indices": dropped_indices.tolist(),
                "dropped_original_row_count": int(len(dropped_indices)),
                "dropped_by_label": _counts(labels[dropped_indices]),
                "cell_selection_rule": CELL_SELECTION_RULE,
                "minimum_cells_per_retained_label": minimum_cells,
                "selected_sorted_indices": rows.tolist(),
                "label_counts_before": _counts(labels),
                "retained_label_counts_before_sampling": _counts(retained_labels),
                "label_counts_after": _counts(selected_labels),
            }
        )
        actual_snapshots.append(snapshot)
    expression_out = np.concatenate(expressions, axis=0)
    interventions_out = np.concatenate(labels_out)
    final_counts = Counter(interventions_out.tolist())
    required_response_sources = (
        set(genes)
        if stage == "refit"
        else set(genes) & set(public_manifest["tune_sources"])
    )
    if (
        not required_response_sources
        or final_counts.get(CONTROL_LABEL, 0) < minimum_cells
        or any(
            final_counts.get(gene, 0) < minimum_cells
            for gene in required_response_sources
        )
    ):
        raise TaskCProfileInputError(
            "profile output does not retain the public minimum for every selected source"
        )
    arrays: dict[str, np.ndarray] = {
        "expression_matrix": expression_out,
        "interventions": interventions_out,
        "var_names": np.asarray(genes),
    }
    environments: np.ndarray | None = None
    environment_record: dict[str, object] | None = None
    transformation = WITHIN_TRANSFORMATION
    if condition == "cross_environment":
        environments = np.concatenate(environment_out)
        arrays["environment_labels"] = environments
        transformation = CROSS_TRANSFORMATION
        environment_record = {
            "ordered_context_ids": [record["context_id"] for record in context_records],
            "cell_counts": {
                str(record["context_id"]): len(record["selected_sorted_indices"])
                for record in context_records
            },
        }
    input_bytes = _deterministic_npz(arrays)
    if len(input_bytes) > MAXIMUM_PROFILE_INPUT_BYTES:
        raise TaskCProfileInputError("profile output is too large")
    output = {
        "sha256": f"sha256:{hashlib.sha256(input_bytes).hexdigest()}",
        "size_bytes": len(input_bytes),
        "gene_names_sha256": _gene_names_sha256(genes),
        "array_names": list(arrays),
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "profile_input_schema": PROFILE_SCHEMA,
        "profile": profile,
        "limits": {
            "gene_count": gene_limit,
            "cells_per_context": cell_limit,
        },
        "selection_seed": PROFILE_SELECTION_SEED,
        "condition": condition,
        "context_id": context_id,
        "direction": direction,
        "stage": stage,
        "gene_selection": {
            "rule": GENE_SELECTION_RULE,
            "selection_reference_stage": "refit",
            "parents": list(gene_parents),
            "ordered_genes": list(genes),
            "ordered_indices": list(gene_indices),
            "dropped_due_to_cell_budget": list(dropped_due_to_cell_budget),
        },
        "contexts": context_records,
        "environment_labels": environment_record,
        "transformation": transformation,
        "output": output,
    }
    used_inventory_snapshots = [
        *(inventory[record["public_relative_path"]] for record in gene_parents),
        *actual_snapshots,
    ]
    parent_by_path = {
        inventory_snapshot.path: parent_cache[
            (inventory_snapshot.device, inventory_snapshot.inode)
        ][0]
        for inventory_snapshot in used_inventory_snapshots
    }
    for values in (expression_out, interventions_out, environments):
        if values is not None:
            values.setflags(write=False)
    return _BuiltProfile(
        input_bytes=input_bytes,
        manifest=manifest,
        expression=expression_out,
        interventions=interventions_out,
        genes=genes,
        environments=environments,
        public_snapshot=public_snapshot,
        parent_snapshots=tuple(parent_by_path.values()),
    )


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify(snapshot: _FileSnapshot, label: str) -> None:
    current = _capture(snapshot.path, label, maximum_bytes=max(1, snapshot.size))
    if current != snapshot:
        raise TaskCProfileInputError(f"{label} changed during profile materialization")


def materialize_task_c_profile_input(
    *,
    public_manifest_path: Path,
    profile: str,
    condition: str,
    output_dir: Path,
    stage: str = "refit",
    context_id: str | None = None,
    direction: str | None = None,
) -> dict[str, str]:
    """Write one deterministic public profile subset without overwriting outputs."""

    built = _build_profile(
        public_manifest_path=public_manifest_path,
        profile=profile,
        condition=condition,
        stage=stage,
        context_id=context_id,
        direction=direction,
    )
    destination = _absolute(output_dir)
    if destination.exists() or destination.is_symlink():
        raise TaskCProfileInputError("profile output directory already exists")
    parent = destination.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _reject_symlink_parts(parent, "profile output parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    published = False
    try:
        _write_new(staging / "profile_input.npz", built.input_bytes)
        _write_new(staging / "profile_input_manifest.json", _json_bytes(built.manifest))
        _verify(built.public_snapshot, "public manifest")
        for index, snapshot in enumerate(built.parent_snapshots):
            _verify(snapshot, f"profile parent {index + 1}")
        if destination.exists() or destination.is_symlink():
            raise TaskCProfileInputError("profile output appeared before publication")
        os.rename(staging, destination)
        published = True
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    return {
        "input_npz": str(destination / "profile_input.npz"),
        "manifest": str(destination / "profile_input_manifest.json"),
    }


def validate_task_c_profile_input(
    *,
    input_path: Path,
    profile_manifest_path: Path,
    public_manifest_path: Path,
) -> TaskCProfileInput:
    """Rebuild a profile from public parents and compare every recorded value."""

    manifest_snapshot = _capture(
        profile_manifest_path,
        "profile input manifest",
        maximum_bytes=MAXIMUM_MANIFEST_BYTES,
        single_link=True,
    )
    observed = _json(manifest_snapshot, "profile input manifest")
    if set(observed) != _PROFILE_FIELDS:
        raise TaskCProfileInputError("profile manifest fields changed")
    if observed.get("schema_version") != "1.0" or observed.get(
        "profile_input_schema"
    ) != PROFILE_SCHEMA:
        raise TaskCProfileInputError("profile manifest schema changed")
    profile = observed.get("profile")
    condition = observed.get("condition")
    context_id = observed.get("context_id")
    direction = observed.get("direction")
    stage = observed.get("stage")
    if (
        not isinstance(profile, str)
        or not isinstance(condition, str)
        or stage not in {"tune", "refit"}
    ):
        raise TaskCProfileInputError("profile manifest identity is malformed")
    if context_id is not None and not isinstance(context_id, str):
        raise TaskCProfileInputError("profile context identity is malformed")
    if direction is not None and not isinstance(direction, str):
        raise TaskCProfileInputError("profile direction identity is malformed")
    built = _build_profile(
        public_manifest_path=public_manifest_path,
        profile=profile,
        condition=condition,
        stage=stage,
        context_id=context_id,
        direction=direction,
    )
    if observed != built.manifest:
        raise TaskCProfileInputError(
            "profile manifest does not match the recomputed public subset"
        )
    input_snapshot = _capture(
        input_path,
        "profile input NPZ",
        maximum_bytes=MAXIMUM_FILE_BYTES,
        single_link=True,
    )
    if input_snapshot.payload != built.input_bytes:
        raise TaskCProfileInputError(
            "profile input bytes do not match the recomputed public subset"
        )
    _verify(manifest_snapshot, "profile input manifest")
    _verify(input_snapshot, "profile input NPZ")
    _verify(built.public_snapshot, "public manifest")
    for index, parent_snapshot in enumerate(built.parent_snapshots):
        _verify(parent_snapshot, f"profile parent {index + 1}")
    return TaskCProfileInput(
        profile=profile,
        condition=condition,
        stage=stage,
        context_id=context_id,
        direction=direction,
        expression=built.expression,
        interventions=built.interventions,
        gene_names=built.genes,
        environment_labels=built.environments,
        input_path=input_snapshot.path,
        manifest_path=manifest_snapshot.path,
        input_sha256=input_snapshot.sha256,
        manifest_sha256=manifest_snapshot.sha256,
        public_manifest_sha256=built.public_snapshot.sha256,
        parent_paths=tuple(snapshot.path for snapshot in built.parent_snapshots),
        parent_sha256=tuple(snapshot.sha256 for snapshot in built.parent_snapshots),
        manifest=observed,
        input_snapshot=input_snapshot,
        manifest_snapshot=manifest_snapshot,
        public_snapshot=built.public_snapshot,
        parent_snapshots=built.parent_snapshots,
    )
