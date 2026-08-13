"""Strict loading and provenance records for the real-data Task C benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

CAUSALBENCH_REPOSITORY = "https://github.com/causalbench/causalbench.git"
CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
CONTROL_LABEL = "non-targeting"


class TaskCDataError(ValueError):
    """Raised when a Task C input cannot meet the benchmark data contract."""


@dataclass(frozen=True)
class TaskCDataset:
    expression: np.ndarray
    interventions: np.ndarray
    gene_names: tuple[str, ...]
    context_id: str
    source_path: Path
    source_sha256: str


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


_TASK_C_SPLIT_SEEDS = frozenset({11, 23, 47, 71, 97})


def _eligible_sources(dataset: TaskCDataset, min_cells: int) -> set[str]:
    counts = Counter(dataset.interventions.tolist())
    genes = set(dataset.gene_names)
    return {label for label, count in counts.items() if label != CONTROL_LABEL and label in genes and count >= min_cells}


def _control_partitions(dataset: TaskCDataset, seed: int) -> dict[str, tuple[int, ...]]:
    controls = np.flatnonzero(dataset.interventions == CONTROL_LABEL)
    if len(controls) < 5:
        raise TaskCDataError("at least 5 control cells are required")
    shuffled = np.random.default_rng(seed).permutation(controls)
    train_end = int(len(shuffled) * 0.6)
    tune_end = train_end + int(len(shuffled) * 0.2)
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
            "k562": MappingProxyType(_control_partitions(k562, seed)),
            "rpe1": MappingProxyType(_control_partitions(rpe1, seed)),
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
        expected_controls = _control_partitions(dataset, int(split.seed))
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


def load_task_c_dataset(path: Path | str, *, context_id: str) -> TaskCDataset:
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("context_id must be exactly k562 or rpe1")
    source_path = Path(path).expanduser().resolve()
    before = _file_signature(source_path)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
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
    return TaskCDataset(
        expression=expression,
        interventions=np.asarray(labels, dtype=str),
        gene_names=genes,
        context_id=context_id,
        source_path=source_path,
        source_sha256=_consistent_sha256(source_path, before),
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
