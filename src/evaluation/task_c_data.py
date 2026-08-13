"""Strict loading and provenance records for the real-data Task C benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    result: list[str] = []
    for value in values.tolist():
        if isinstance(value, (list, tuple, dict, set, np.ndarray)):
            raise TaskCDataError(f"{kind} contain non-stringable nested values")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="strict")
        try:
            text = str(value).strip()
        except Exception as exc:
            raise TaskCDataError(f"{kind} contain non-stringable values") from exc
        result.append(text)
    return tuple(result)


def load_task_c_dataset(path: Path | str, *, context_id: str) -> TaskCDataset:
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("context_id must be exactly k562 or rpe1")
    source_path = Path(path).expanduser().resolve()
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
    except (OSError, ValueError, TypeError) as exc:
        raise TaskCDataError(f"cannot load Task C dataset: {source_path}") from exc

    if expression.ndim != 2:
        raise TaskCDataError("expression_matrix must be two-dimensional")
    if expression.shape[0] == 0 or expression.shape[1] < 1:
        raise TaskCDataError("expression_matrix must have rows and at least one gene column")
    if not np.issubdtype(expression.dtype, np.number) or not np.all(np.isfinite(expression)):
        raise TaskCDataError("expression_matrix values must be finite numeric values")
    labels = _text_vector(interventions_raw, "intervention labels")
    genes = _text_vector(genes_raw, "gene names")
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
        source_sha256=sha256_path(source_path),
    )


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


def _validate_reference(path: Path | str) -> tuple[int, str]:
    path = Path(path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"source", "target"}.issubset(reader.fieldnames):
                raise TaskCDataError(f"reference CSV must contain source,target headers: {path}")
            edges: set[tuple[str, str]] = set()
            count = 0
            for row in reader:
                source, target = row.get("source"), row.get("target")
                if source is None or target is None:
                    raise TaskCDataError(f"reference CSV has malformed row: {path}")
                source, target = source.strip(), target.strip()
                if not source or not target or source.lower() in {"nan", "inf", "infinity"} or target.lower() in {"nan", "inf", "infinity"}:
                    raise TaskCDataError(f"reference CSV source/target must be finite nonempty text: {path}")
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
    return count, sha256_path(path)


def build_task_c_reference_provenance(*, context_id: str, pooled_path: Path | str, chipseq_path: Path | str) -> dict[str, Any]:
    if context_id not in {"k562", "rpe1"}:
        raise TaskCDataError("context_id must be exactly k562 or rpe1")
    pooled_rows, pooled_hash = _validate_reference(pooled_path)
    chip_rows, chip_hash = _validate_reference(chipseq_path)
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
