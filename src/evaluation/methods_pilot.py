"""Minimal train/tune-only model helpers for the protocol-v2.1 pilot.

This module deliberately contains no benchmark file loading or evidence
publication.  It fits three equal-capacity representation models to arrays
whose train/tune split has already been frozen by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import time
from types import MappingProxyType
from typing import Mapping
import zipfile

import numpy as np

from src.evaluation.run_evidence_identity import (
    RunEvidenceIdentity,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import RunEvidencePublisher


_METHODS = (
    "hypersca_hyperbolic",
    "euclidean_autoencoder",
    "hypersca_without_hierarchy_loss",
)
OSTA_SPLIT_SEED = 19_911


def _readonly(array: np.ndarray, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _coordinates(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] < 16 or raw.shape[1] != 2:
        raise ValueError("positions must have shape (at least 16 cells, 2)")
    if raw.dtype.kind not in "fiu":
        raise ValueError("positions must be numeric")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("positions must be finite")
    return result


def _seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError("seed must be a non-negative built-in 32-bit integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _normalized_coordinates(positions: np.ndarray) -> np.ndarray:
    lower = positions.min(axis=0)
    span = positions.max(axis=0) - lower
    span[span <= 1e-12] = 1.0
    return np.clip((positions - lower) / span, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class SpatialPilotSplit:
    train_indices: np.ndarray
    tune_indices: np.ndarray
    buffer_indices: np.ndarray
    tune_block_ids: tuple[str, ...]
    grid_size: int
    tune_origin: tuple[int, int]
    tune_span: int
    buffer_width: int


def split_spatial_train_tune(
    positions: object,
    *,
    seed: int,
    grid_size: int = 8,
    tune_span: int = 2,
    buffer_width: int = 1,
) -> SpatialPilotSplit:
    """Make one deterministic contiguous spatial holdout with a train buffer."""

    coords = _coordinates(positions)
    seed = _seed(seed)
    grid_size = _positive_integer(grid_size, "grid_size")
    tune_span = _positive_integer(tune_span, "tune_span")
    if type(buffer_width) is not int or buffer_width < 0:
        raise ValueError("buffer_width must be a non-negative built-in integer")
    if tune_span + 2 * buffer_width >= grid_size:
        raise ValueError("grid must leave cells outside the tune block and buffer")

    normalized = _normalized_coordinates(coords)
    bins = np.minimum(
        np.floor(normalized * grid_size).astype(np.int64), grid_size - 1
    )
    rng = np.random.default_rng(seed)
    minimum_origin = buffer_width
    maximum_origin = grid_size - tune_span - buffer_width
    origin_x = int(rng.integers(minimum_origin, maximum_origin + 1))
    origin_y = int(rng.integers(minimum_origin, maximum_origin + 1))
    tune = (
        (bins[:, 0] >= origin_x)
        & (bins[:, 0] < origin_x + tune_span)
        & (bins[:, 1] >= origin_y)
        & (bins[:, 1] < origin_y + tune_span)
    )
    expanded = (
        (bins[:, 0] >= origin_x - buffer_width)
        & (bins[:, 0] < origin_x + tune_span + buffer_width)
        & (bins[:, 1] >= origin_y - buffer_width)
        & (bins[:, 1] < origin_y + tune_span + buffer_width)
    )
    buffer = expanded & ~tune
    train = ~expanded
    train_indices = np.flatnonzero(train)
    tune_indices = np.flatnonzero(tune)
    buffer_indices = np.flatnonzero(buffer)
    if len(train_indices) < 8 or len(tune_indices) < 2:
        raise ValueError("spatial split retained too few train or tune cells")
    tune_blocks = tuple(
        f"g{grid_size}:{int(x)}:{int(y)}" for x, y in bins[tune_indices]
    )
    return SpatialPilotSplit(
        train_indices=_readonly(train_indices, dtype=np.dtype("<i8")),
        tune_indices=_readonly(tune_indices, dtype=np.dtype("<i8")),
        buffer_indices=_readonly(buffer_indices, dtype=np.dtype("<i8")),
        tune_block_ids=tune_blocks,
        grid_size=grid_size,
        tune_origin=(origin_x, origin_y),
        tune_span=tune_span,
        buffer_width=buffer_width,
    )


def build_spatial_hierarchy_triplets(
    positions: object,
    train_indices: object,
    *,
    seed: int,
    maximum_triplets: int = 4096,
) -> np.ndarray:
    """Build train-only triplets from nested 2x2 and 4x4 coordinate blocks."""

    coords = _coordinates(positions)
    seed = _seed(seed)
    maximum_triplets = _positive_integer(maximum_triplets, "maximum_triplets")
    raw_indices = np.asarray(train_indices)
    if (
        raw_indices.ndim != 1
        or raw_indices.dtype.kind not in "iu"
        or len(raw_indices) < 3
    ):
        raise ValueError("train_indices must be a one-dimensional integer array")
    indices = np.asarray(raw_indices, dtype=np.int64)
    if len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(
        indices >= len(coords)
    ):
        raise ValueError("train_indices must be unique in-range cell indices")

    normalized = _normalized_coordinates(coords[indices])
    coarse = np.minimum(np.floor(normalized * 2).astype(np.int64), 1)
    fine = np.minimum(np.floor(normalized * 4).astype(np.int64), 3)
    coarse_ids = coarse[:, 0] * 2 + coarse[:, 1]
    fine_ids = fine[:, 0] * 4 + fine[:, 1]
    by_fine: dict[int, list[int]] = {}
    for local_index, block in enumerate(fine_ids.tolist()):
        by_fine.setdefault(int(block), []).append(local_index)
    rng = np.random.default_rng(seed)
    anchors = rng.permutation(len(indices))
    rows: list[tuple[int, int, int]] = []
    for anchor in anchors:
        same_fine = by_fine[int(fine_ids[anchor])]
        if len(same_fine) < 2:
            continue
        positive_candidates = [item for item in same_fine if item != anchor]
        negative_candidates = np.flatnonzero(coarse_ids != coarse_ids[anchor])
        if not positive_candidates or len(negative_candidates) == 0:
            continue
        positive = int(positive_candidates[int(rng.integers(len(positive_candidates)))])
        negative = int(negative_candidates[int(rng.integers(len(negative_candidates)))])
        rows.append(
            (int(indices[anchor]), int(indices[positive]), int(indices[negative]))
        )
        if len(rows) >= maximum_triplets:
            break
    if not rows:
        raise ValueError("train coordinates do not support nested hierarchy triplets")
    return _readonly(np.asarray(rows), dtype=np.dtype("<i8"))


@dataclass(frozen=True, slots=True)
class SpatialPilotConfig:
    hidden_dim: int = 64
    latent_dim: int = 16
    maximum_epochs: int = 40
    early_stopping_patience: int = 6
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hierarchy_weight: float = 0.1
    hierarchy_margin: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "latent_dim",
            "maximum_epochs",
            "early_stopping_patience",
        ):
            _positive_integer(getattr(self, name), name)
        for name in (
            "learning_rate",
            "weight_decay",
            "hierarchy_weight",
            "hierarchy_margin",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative built-in float")
        if self.learning_rate == 0.0 or self.hierarchy_weight == 0.0:
            raise ValueError("learning_rate and hierarchy_weight must be positive")


@dataclass(frozen=True, slots=True)
class SpatialPilotFit:
    embeddings: Mapping[str, np.ndarray]
    parameter_counts: Mapping[str, int]
    hierarchy_loss_enabled: Mapping[str, bool]
    epochs_completed: Mapping[str, int]
    training_scopes: tuple[str, str]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())
    return output.getvalue()


def _sample_indices(indices: np.ndarray, *, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def _matrix_rows(matrix: object, indices: np.ndarray):
    try:
        return matrix[indices, :]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("H5AD expression rows could not be read") from exc


def _variance(matrix: object) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        second = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        result = second - mean * mean
    else:
        values = np.asarray(matrix, dtype=np.float64)
        result = values.var(axis=0)
    result = np.maximum(np.asarray(result, dtype=np.float64), 0.0)
    if not np.isfinite(result).all():
        raise ValueError("train-only gene variance is not finite")
    return result


def _dense_columns(matrix: object, columns: np.ndarray) -> np.ndarray:
    from scipy import sparse

    selected = matrix[:, columns]
    if sparse.issparse(selected):
        selected = selected.toarray()
    result = np.asarray(selected, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("selected H5AD expression is not finite and non-negative")
    return result


def _unit_rows(units: tuple[object, ...]) -> list[dict[str, object]]:
    return [asdict(unit) for unit in units]


def run_osta_pilot_run(
    *,
    h5ad_path: Path,
    dataset_id: str,
    platform_id: str,
    output_dir: Path,
    seed: int,
    config: SpatialPilotConfig,
    device: str,
    maximum_cells: int = 4096,
    maximum_genes: int = 256,
) -> dict[str, object]:
    """Run one OSTA seed/platform pilot without release data or promotion."""

    import anndata as ad
    import pandas as pd

    from src.evaluation.benchmark_evidence import build_osta_paired_units
    from src.evaluation.methods_protocol import (
        default_methods_protocol,
        protocol_identity,
    )

    seed = _seed(seed)
    for value, name in ((dataset_id, "dataset_id"), (platform_id, "platform_id")):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be non-empty built-in text")
    maximum_cells = _positive_integer(maximum_cells, "maximum_cells")
    maximum_genes = _positive_integer(maximum_genes, "maximum_genes")
    if maximum_cells < 64 or maximum_genes < 2:
        raise ValueError("pilot needs at least 64 cells and two genes")
    source = Path(h5ad_path).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("pilot output directory already exists")

    started = time.monotonic()
    input_sha256 = _sha256_path(source)
    dataset = ad.read_h5ad(source, backed="r")
    try:
        if "spatial" in dataset.obsm:
            positions = np.asarray(dataset.obsm["spatial"], dtype=np.float64)
        elif {"x", "y"}.issubset(dataset.obs.columns):
            positions = dataset.obs[["x", "y"]].to_numpy(dtype=np.float64)
        else:
            raise ValueError("OSTA pilot input has no physical coordinates")
        split = split_spatial_train_tune(positions, seed=OSTA_SPLIT_SEED)
        tune_cap = min(maximum_cells // 4, 1024)
        tune_indices = _sample_indices(
            split.tune_indices, maximum=tune_cap, seed=OSTA_SPLIT_SEED + 101
        )
        train_cap = maximum_cells - len(tune_indices)
        train_indices = _sample_indices(
            split.train_indices, maximum=train_cap, seed=OSTA_SPLIT_SEED + 211
        )
        if len(tune_indices) <= 30 or len(train_indices) < 32:
            raise ValueError("OSTA pilot sampling retained too few train or tune cells")

        train_rows = _matrix_rows(dataset.X, train_indices)
        gene_names = tuple(str(value) for value in dataset.var_names.tolist())
        if len(gene_names) != dataset.n_vars or len(set(gene_names)) != len(gene_names):
            raise ValueError("OSTA pilot genes must be unique text")
        variances = _variance(train_rows)
        gene_order = np.lexsort((np.asarray(gene_names), -variances))
        gene_indices = np.asarray(
            gene_order[: min(maximum_genes, len(gene_order))], dtype=np.int64
        )
        train_expression = _dense_columns(train_rows, gene_indices)
        tune_rows = _matrix_rows(dataset.X, tune_indices)
        tune_expression = _dense_columns(tune_rows, gene_indices)
    finally:
        dataset.file.close()

    expression = np.concatenate([train_expression, tune_expression], axis=0)
    selected_positions = np.concatenate(
        [positions[train_indices], positions[tune_indices]], axis=0
    )
    local_train = np.arange(len(train_indices), dtype=np.int64)
    local_tune = np.arange(
        len(train_indices), len(train_indices) + len(tune_indices), dtype=np.int64
    )
    tune_block_by_global_index = dict(
        zip(split.tune_indices.tolist(), split.tune_block_ids, strict=True)
    )
    local_split = SpatialPilotSplit(
        train_indices=_readonly(local_train, dtype=np.dtype("<i8")),
        tune_indices=_readonly(local_tune, dtype=np.dtype("<i8")),
        buffer_indices=_readonly(np.asarray([], dtype=np.int64), dtype=np.dtype("<i8")),
        tune_block_ids=tuple(
            tune_block_by_global_index[int(index)] for index in tune_indices
        ),
        grid_size=split.grid_size,
        tune_origin=split.tune_origin,
        tune_span=split.tune_span,
        buffer_width=split.buffer_width,
    )
    fit = fit_spatial_pilot_models(
        expression=expression,
        positions=selected_positions,
        split=local_split,
        seed=seed,
        config=config,
        device=device,
    )
    physical = selected_positions[local_tune]
    identities = tuple(dataset_id for _ in local_tune)
    blocks = local_split.tune_block_ids
    platforms = tuple(platform_id for _ in local_tune)
    primary_units: list[object] = []
    secondary_rows: list[dict[str, object]] = []
    comparator_by_method = {
        "euclidean_autoencoder": "euclidean_autoencoder",
        "hypersca_without_hierarchy_loss": "hypersca_without_hierarchy_loss",
    }
    for comparator_method, comparator_id in comparator_by_method.items():
        units = build_osta_paired_units(
            physical_coordinates=physical,
            hypersca_embedding=fit.embeddings["hypersca_hyperbolic"],
            comparator_embedding=fit.embeddings[comparator_method],
            sample_ids=identities,
            block_ids=blocks,
            platform_ids=platforms,
            seed=seed,
            comparator_id=comparator_id,
            k=15,
        )
        primary_units.extend(units)
        for k in (5, 30):
            secondary = build_osta_paired_units(
                physical_coordinates=physical,
                hypersca_embedding=fit.embeddings["hypersca_hyperbolic"],
                comparator_embedding=fit.embeddings[comparator_method],
                sample_ids=identities,
                block_ids=blocks,
                platform_ids=platforms,
                seed=seed,
                comparator_id=comparator_id,
                k=k,
            )
            for row in _unit_rows(secondary):
                secondary_rows.append({"k": k, **row})

    primary_frame = pd.DataFrame(_unit_rows(tuple(primary_units)))
    primary_csv = primary_frame.to_csv(index=False).encode("utf-8")
    secondary_csv = pd.DataFrame(secondary_rows).to_csv(index=False).encode("utf-8")
    summaries: dict[str, dict[str, object]] = {}
    for comparator_id in comparator_by_method.values():
        selected = primary_frame[primary_frame["comparator_id"] == comparator_id]
        summaries[comparator_id] = {
            "attempted_units": int(len(selected)),
            "completed_units": int((selected["status"] == "completed").sum()),
            "mean_paired_difference": float(selected["paired_difference"].mean()),
            "confidence_interval": None,
            "pilot_only": True,
        }
    status_record: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "benchmark_id": "osta_colon",
        "dataset_id": dataset_id,
        "platform_id": platform_id,
        "seed": seed,
        "promotion_eligible": False,
        "data_scopes": ["train", "tune"],
    }
    elapsed = max(0.0, time.monotonic() - started)
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    peak_gpu = None
    if device == "cuda":
        import torch

        peak_gpu = int(torch.cuda.max_memory_allocated())
    resource_record = {
        "schema_version": "1.0",
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "peak_gpu_memory_bytes": peak_gpu,
        "maximum_cells": maximum_cells,
        "maximum_genes": maximum_genes,
    }
    claim_record = {
        "schema_version": "1.0",
        "claim_id": "spatial",
        "status": "audit_only",
        "reason": "three-seed train/tune pilot cannot authorize promotion",
        "promotion_eligible": False,
    }
    embeddings_payload = _deterministic_npz(
        {
            **dict(fit.embeddings),
            "physical_coordinates": physical,
            "train_global_indices": train_indices,
            "tune_global_indices": tune_indices,
            "selected_gene_indices": gene_indices,
        }
    )
    publisher_source = Path(__file__).resolve().with_name(
        "run_evidence_publisher.py"
    )
    code_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("methods_protocol.py"),
        Path(__file__).resolve().with_name("benchmark_evidence.py"),
        publisher_source,
    )
    code_record = {
        path.name: {"sha256": _sha256_path(path)} for path in code_paths
    }
    split_record = {
        "schema_version": "osta_spatial_split_v1",
        "dataset_id": dataset_id,
        "platform_id": platform_id,
        "input_sha256": input_sha256,
        "split_seed": OSTA_SPLIT_SEED,
        "grid_size": split.grid_size,
        "tune_origin": list(split.tune_origin),
        "tune_span": split.tune_span,
        "buffer_width": split.buffer_width,
        "train_global_indices": train_indices.tolist(),
        "tune_global_indices": tune_indices.tolist(),
        "selected_gene_indices": gene_indices.tolist(),
    }
    statistical_unit_record = {
        "schema_version": "osta_platform_sample_block_v1",
        "k": 15,
        "platforms": sorted(set(platforms)),
        "samples": sorted(set(identities)),
        "blocks": sorted(set(blocks)),
        "units": sorted(set(primary_frame["unit_id"].tolist())),
    }
    protocol = default_methods_protocol()
    analysis_record = {
        "schema_version": "osta_analysis_v1",
        "primary_metric": protocol.spatial_primary_metric,
        "primary_k": protocol.spatial_primary_k,
        "confirmatory_comparator": protocol.spatial_confirmatory_comparator,
        "attribution_comparator": protocol.spatial_attribution_comparator,
        "secondary_k": list(protocol.spatial_secondary_k),
    }
    input_record = {
        "schema_version": "osta_input_v1",
        "dataset_id": dataset_id,
        "platform_id": platform_id,
        "sha256": input_sha256,
    }
    config_record = {
        "schema_version": "osta_model_config_v1",
        "model_seed": seed,
        "device": device,
        "maximum_cells": maximum_cells,
        "maximum_genes": maximum_genes,
        "model_config": asdict(config),
    }
    identity = RunEvidenceIdentity(
        schema_version="1.0",
        protocol_version=protocol.protocol_version,
        protocol_identity=protocol_identity(protocol),
        claim_id="spatial",
        benchmark_id="osta_colon",
        data_scopes=("train", "tune"),
        data_split_seed=OSTA_SPLIT_SEED,
        model_seed=seed,
        data_split_identity_sha256=canonical_sha256(split_record),
        statistical_unit_schema="osta_platform_sample_block_v1",
        statistical_unit_identity_sha256=canonical_sha256(statistical_unit_record),
        analysis_identity_sha256=canonical_sha256(analysis_record),
        input_identity_sha256=canonical_sha256(input_record),
        config_identity_sha256=canonical_sha256(config_record),
        code_identity_sha256=canonical_sha256(code_record),
        evidence_role="pilot_audit_only",
    )
    summary = {
        **status_record,
        "split": {
            "split_seed": OSTA_SPLIT_SEED,
            "grid_size": split.grid_size,
            "tune_origin": list(split.tune_origin),
            "tune_span": split.tune_span,
            "buffer_width": split.buffer_width,
            "train_cells": int(len(train_indices)),
            "tune_cells": int(len(tune_indices)),
        },
        "gene_selection": {
            "rule": "train_only_variance_descending_gene_name_tie_break",
            "selected_genes": [gene_names[index] for index in gene_indices],
        },
        "model_config": asdict(config),
        "parameter_counts": dict(fit.parameter_counts),
        "hierarchy_loss_enabled": dict(fit.hierarchy_loss_enabled),
        "epochs_completed": dict(fit.epochs_completed),
        "code": code_record,
        "publisher": {"source_sha256": _sha256_path(publisher_source)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    publisher = RunEvidencePublisher.begin(
        output_dir=output,
        identity=identity,
        statistical_unit_record=statistical_unit_record,
        required_artifacts=(
            "resource_usage.json",
            "primary_metric_units.csv",
            "primary_metric_summary.json",
            "secondary_metrics.csv",
            "claim_decision.json",
            "embeddings.npz",
        ),
        maximum_bundle_bytes=5 * 1024**3,
    )
    with publisher:
        publisher.add_bytes(
            "resource_usage.json",
            _json_bytes(resource_record),
            media_type="application/json",
        )
        publisher.add_bytes(
            "primary_metric_units.csv", primary_csv, media_type="text/csv"
        )
        publisher.add_bytes(
            "primary_metric_summary.json",
            _json_bytes(summaries),
            media_type="application/json",
        )
        publisher.add_bytes(
            "secondary_metrics.csv", secondary_csv, media_type="text/csv"
        )
        publisher.add_bytes(
            "claim_decision.json",
            _json_bytes(claim_record),
            media_type="application/json",
        )
        publisher.add_bytes(
            "embeddings.npz",
            embeddings_payload,
            media_type="application/x-npz",
        )
        publisher.finalize_completed(summary=summary)
    return status_record


def fit_spatial_pilot_models(
    *,
    expression: object,
    positions: object,
    split: SpatialPilotSplit,
    seed: int,
    config: SpatialPilotConfig,
    device: str,
) -> SpatialPilotFit:
    """Fit the three equal-capacity protocol-v2.1 spatial pilot models."""

    import torch
    from torch import nn
    from torch.nn import functional as functional

    values = np.asarray(expression)
    coords = _coordinates(positions)
    seed = _seed(seed)
    if values.ndim != 2 or values.shape[0] != len(coords) or values.shape[1] < 2:
        raise ValueError("expression must be cells by at least two genes")
    if values.dtype.kind not in "fiu":
        raise ValueError("expression must be numeric")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("expression must be finite and non-negative")
    if type(split) is not SpatialPilotSplit:
        raise ValueError("split must be SpatialPilotSplit")
    if type(config) is not SpatialPilotConfig:
        raise ValueError("config must be SpatialPilotConfig")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    train_values = np.log1p(values[split.train_indices])
    tune_values = np.log1p(values[split.tune_indices])
    center = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale <= 1e-6] = 1.0
    train_values = np.clip((train_values - center) / scale, -8.0, 8.0)
    tune_values = np.clip((tune_values - center) / scale, -8.0, 8.0)
    triplets = build_spatial_hierarchy_triplets(
        coords, split.train_indices, seed=seed
    )
    local_by_global = {
        int(global_index): local_index
        for local_index, global_index in enumerate(split.train_indices.tolist())
    }
    local_triplets = np.asarray(
        [[local_by_global[int(item)] for item in row] for row in triplets],
        dtype=np.int64,
    )

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(values.shape[1], config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, values.shape[1]),
            )

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent = self.encoder(inputs)
            return latent, self.decoder(latent)

    torch_device = torch.device(device)
    train_tensor = torch.as_tensor(train_values, dtype=torch.float32, device=torch_device)
    tune_tensor = torch.as_tensor(tune_values, dtype=torch.float32, device=torch_device)
    triplet_tensor = torch.as_tensor(local_triplets, dtype=torch.long, device=torch_device)

    def to_ball(latent: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(latent, dim=1, keepdim=True).clamp_min(1e-8)
        return 0.95 * torch.tanh(norm) * latent / norm

    def poincare_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        difference = ((left - right) ** 2).sum(dim=1)
        left_denominator = (1.0 - (left**2).sum(dim=1)).clamp_min(1e-6)
        right_denominator = (1.0 - (right**2).sum(dim=1)).clamp_min(1e-6)
        argument = 1.0 + 2.0 * difference / (
            left_denominator * right_denominator
        )
        return torch.acosh(argument.clamp_min(1.0 + 1e-6))

    embeddings: dict[str, np.ndarray] = {}
    parameter_counts: dict[str, int] = {}
    epochs_completed: dict[str, int] = {}
    hierarchy_flags = {
        "hypersca_hyperbolic": True,
        "euclidean_autoencoder": True,
        "hypersca_without_hierarchy_loss": False,
    }
    for method in _METHODS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = Autoencoder().to(torch_device)
        parameter_counts[method] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        completed = 0
        for epoch in range(config.maximum_epochs):
            model.train()
            latent, reconstructed = model(train_tensor)
            reconstruction = functional.mse_loss(reconstructed, train_tensor)
            geometry = (
                latent
                if method == "euclidean_autoencoder"
                else to_ball(latent)
            )
            if hierarchy_flags[method]:
                anchor = geometry[triplet_tensor[:, 0]]
                positive = geometry[triplet_tensor[:, 1]]
                negative = geometry[triplet_tensor[:, 2]]
                if method == "euclidean_autoencoder":
                    positive_distance = torch.linalg.vector_norm(
                        anchor - positive, dim=1
                    )
                    negative_distance = torch.linalg.vector_norm(
                        anchor - negative, dim=1
                    )
                else:
                    positive_distance = poincare_distance(anchor, positive)
                    negative_distance = poincare_distance(anchor, negative)
                hierarchy = functional.relu(
                    positive_distance - negative_distance + config.hierarchy_margin
                ).mean()
            else:
                hierarchy = reconstruction.new_zeros(())
            total = reconstruction + config.hierarchy_weight * hierarchy
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed = epoch + 1

            model.eval()
            with torch.no_grad():
                _, tune_reconstructed = model(tune_tensor)
                tune_loss = float(
                    functional.mse_loss(tune_reconstructed, tune_tensor).cpu()
                )
            if tune_loss < best_loss - 1e-8:
                best_loss = tune_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= config.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("pilot model produced no finite training state")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            tangent, _ = model(tune_tensor)
            embedded = (
                tangent
                if method == "euclidean_autoencoder"
                else to_ball(tangent)
            )
        array = embedded.detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError(f"{method} produced non-finite embeddings")
        embeddings[method] = _readonly(array, dtype=np.dtype("<f4"))
        epochs_completed[method] = completed

    return SpatialPilotFit(
        embeddings=MappingProxyType(embeddings),
        parameter_counts=MappingProxyType(parameter_counts),
        hierarchy_loss_enabled=MappingProxyType(hierarchy_flags),
        epochs_completed=MappingProxyType(epochs_completed),
        training_scopes=("train", "tune_evaluation_only"),
    )


__all__ = [
    "SpatialPilotConfig",
    "SpatialPilotFit",
    "SpatialPilotSplit",
    "OSTA_SPLIT_SEED",
    "build_spatial_hierarchy_triplets",
    "fit_spatial_pilot_models",
    "run_osta_pilot_run",
    "split_spatial_train_tune",
]
