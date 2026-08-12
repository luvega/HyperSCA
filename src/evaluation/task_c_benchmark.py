"""Task C adapter and deterministic mean-difference intervention baseline.

The callable baseline follows CausalBench's public model input contract without
depending on the third-party package.  Reference networks are evaluation aids,
not complete causal ground truth, and never affect scoring or edge ranking.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.benchmark_contract import (
    build_run_manifest,
    contract_digest,
    validate_benchmark_contract,
)


SUPPORTED_DATA_STATUSES = {"external_benchmark", "synthetic_smoke"}


class TaskCBenchmarkError(ValueError):
    """Raised when Task C inputs or evidence violate the adapter contract."""


@dataclass(frozen=True)
class MeanDifferenceResult:
    """Complete directed score table and coverage summary."""

    scores: pd.DataFrame
    summary: dict[str, Any]


def _validate_expression_matrix(expression: Any) -> None:
    if not hasattr(expression, "shape") or len(expression.shape) != 2:
        raise TaskCBenchmarkError("expression must be a two-dimensional matrix")
    if expression.shape[0] < 1 or expression.shape[1] < 2:
        raise TaskCBenchmarkError(
            "expression must contain at least one cell and two genes"
        )
    values = expression.data if sparse.issparse(expression) else np.asarray(expression)
    if not np.isfinite(values).all():
        raise TaskCBenchmarkError("expression values must be finite")


def _mean_rows(expression: Any, mask: np.ndarray) -> np.ndarray:
    mean = expression[mask].mean(axis=0)
    return np.asarray(mean).reshape(-1).astype(float, copy=False)


def score_mean_difference_network(
    expression: Any,
    interventions: Sequence[str],
    gene_names: Sequence[str],
    *,
    control_label: str = "non-targeting",
    excluded_label: str = "excluded",
    min_cells_per_intervention: int = 5,
) -> MeanDifferenceResult:
    """Score source-to-target edges by absolute interventional mean change.

    For each eligible perturbed source gene, the signed effect is the perturbed
    expression mean minus the shared non-targeting control mean.  The ranking
    score is its absolute value.  Self edges are excluded and target-name order
    provides deterministic tie breaking.
    """
    _validate_expression_matrix(expression)
    if (
        not isinstance(min_cells_per_intervention, int)
        or isinstance(min_cells_per_intervention, bool)
        or min_cells_per_intervention < 1
    ):
        raise TaskCBenchmarkError(
            "min_cells_per_intervention must be a positive integer"
        )
    labels = np.asarray([str(value) for value in interventions], dtype=str)
    genes = [str(value) for value in gene_names]
    if len(labels) != expression.shape[0]:
        raise TaskCBenchmarkError(
            "interventions length must match expression rows"
        )
    if len(genes) != expression.shape[1]:
        raise TaskCBenchmarkError("gene_names length must match expression columns")
    if any(not gene.strip() for gene in genes):
        raise TaskCBenchmarkError("gene_names must be non-empty strings")
    if len(genes) != len(set(genes)):
        raise TaskCBenchmarkError("gene_names must be unique")
    if not isinstance(control_label, str) or not control_label:
        raise TaskCBenchmarkError("control_label must be a non-empty string")

    control_mask = labels == control_label
    n_control = int(control_mask.sum())
    if n_control < min_cells_per_intervention:
        raise TaskCBenchmarkError(
            f"at least {min_cells_per_intervention} control cells are required"
        )
    control_mean = _mean_rows(expression, control_mask)
    gene_set = set(genes)
    observed_sources = sorted(
        {
            label
            for label in labels.tolist()
            if label not in {control_label, excluded_label}
        }
    )
    source_counts = {
        source: int(np.count_nonzero(labels == source))
        for source in observed_sources
    }
    eligible_set = {
        source
        for source, count in source_counts.items()
        if source in gene_set and count >= min_cells_per_intervention
    }
    eligible_sources = [gene for gene in genes if gene in eligible_set]
    if not eligible_sources:
        raise TaskCBenchmarkError(
            "no perturbation source is both measured and sufficiently replicated"
        )

    frames: list[pd.DataFrame] = []
    for source in eligible_sources:
        intervention_mask = labels == source
        intervention_mean = _mean_rows(expression, intervention_mask)
        effects = intervention_mean - control_mean
        source_frame = pd.DataFrame(
            {
                "source": source,
                "target": genes,
                "effect": effects,
                "score": np.abs(effects),
                "n_intervention": source_counts[source],
                "n_control": n_control,
            }
        )
        source_frame = source_frame[source_frame["target"] != source]
        source_frame = source_frame.sort_values(
            ["score", "target"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        source_frame["source_rank"] = np.arange(1, len(source_frame) + 1)
        frames.append(source_frame)

    scores = pd.concat(frames, ignore_index=True)
    total_sources = len(observed_sources)
    coverage = len(eligible_sources) / total_sources if total_sources else 0.0
    summary = {
        "n_cells": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "n_control_cells": n_control,
        "observed_sources": observed_sources,
        "eligible_sources": eligible_sources,
        "source_cell_counts": source_counts,
        "n_scored_edges": int(len(scores)),
        "coverage": float(coverage),
        "abstention_rate": float(1.0 - coverage),
        "control_label": control_label,
        "excluded_label": excluded_label,
        "min_cells_per_intervention": min_cells_per_intervention,
    }
    return MeanDifferenceResult(scores=scores, summary=summary)


class MeanDifferenceNetworkBaseline:
    """CausalBench-compatible fixed-top-k simple intervention baseline."""

    def __init__(
        self,
        *,
        top_k_per_source: int = 50,
        min_cells_per_intervention: int = 5,
        control_label: str = "non-targeting",
        excluded_label: str = "excluded",
    ) -> None:
        if (
            not isinstance(top_k_per_source, int)
            or isinstance(top_k_per_source, bool)
            or top_k_per_source < 1
        ):
            raise TaskCBenchmarkError("top_k_per_source must be a positive integer")
        self.top_k_per_source = top_k_per_source
        self.min_cells_per_intervention = min_cells_per_intervention
        self.control_label = control_label
        self.excluded_label = excluded_label

    def score(
        self,
        expression_matrix: Any,
        interventions: Sequence[str],
        gene_names: Sequence[str],
    ) -> MeanDifferenceResult:
        return score_mean_difference_network(
            expression_matrix,
            interventions,
            gene_names,
            control_label=self.control_label,
            excluded_label=self.excluded_label,
            min_cells_per_intervention=self.min_cells_per_intervention,
        )

    def __call__(
        self,
        expression_matrix: Any,
        interventions: Sequence[str],
        gene_names: Sequence[str],
        training_regime: Any,
        seed: int = 0,
    ) -> list[tuple[str, str]]:
        """Return directed edges using the public CausalBench model signature."""
        del training_regime, seed
        result = self.score(expression_matrix, interventions, gene_names)
        selected = result.scores[
            result.scores["source_rank"] <= self.top_k_per_source
        ]
        return list(selected[["source", "target"]].itertuples(index=False, name=None))


def _normalize_reference_edges(
    reference_edges: Iterable[tuple[str, str]],
) -> set[tuple[str, str]]:
    normalized: set[tuple[str, str]] = set()
    for edge in reference_edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise TaskCBenchmarkError(
                "reference_edges must contain source-target pairs"
            )
        source, target = str(edge[0]), str(edge[1])
        if source and target and source != target:
            normalized.add((source, target))
    return normalized


def evaluate_task_c_scores(
    scores: pd.DataFrame,
    reference_edges: Iterable[tuple[str, str]],
    *,
    precision_at_k: int = 1000,
) -> dict[str, Any]:
    """Evaluate complete directed scores against a declared reference network."""
    required_columns = {"source", "target", "score"}
    missing = required_columns - set(scores.columns)
    if missing:
        raise TaskCBenchmarkError(
            f"scores are missing required columns: {sorted(missing)}"
        )
    if scores.empty:
        raise TaskCBenchmarkError("scores must not be empty")
    if scores.duplicated(["source", "target"]).any():
        raise TaskCBenchmarkError("scores must contain unique directed edges")
    numeric_scores = pd.to_numeric(scores["score"], errors="coerce").to_numpy()
    if not np.isfinite(numeric_scores).all() or (numeric_scores < 0).any():
        raise TaskCBenchmarkError("scores must be finite and non-negative")
    if (
        not isinstance(precision_at_k, int)
        or isinstance(precision_at_k, bool)
        or precision_at_k < 1
    ):
        raise TaskCBenchmarkError("precision_at_k must be a positive integer")

    reference = _normalize_reference_edges(reference_edges)
    universe = list(scores[["source", "target"]].itertuples(index=False, name=None))
    labels = np.asarray([edge in reference for edge in universe], dtype=int)
    n_positive = int(labels.sum())
    if n_positive == 0:
        raise TaskCBenchmarkError(
            "reference network has no positive edge in the scored universe"
        )
    if n_positive == len(labels):
        raise TaskCBenchmarkError(
            "reference network leaves no negative edge in the scored universe"
        )

    ordered = scores.assign(_reference=labels).sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    k = min(precision_at_k, len(ordered))
    return {
        "average_precision": float(average_precision_score(labels, numeric_scores)),
        "auroc": float(roc_auc_score(labels, numeric_scores)),
        "precision_at_k": float(ordered.head(k)["_reference"].mean()),
        "precision_k": int(k),
        "n_reference_edges_in_universe": n_positive,
        "n_scored_edges": int(len(scores)),
        "reference_edge_prevalence": float(n_positive / len(scores)),
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an input artifact without loading the whole file into memory."""
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise TaskCBenchmarkError(f"could not hash input artifact {source}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def load_causalbench_npz(
    path: str | Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load the three arrays written by CausalBench ``CreateDataset``."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {"expression_matrix", "interventions", "var_names"}
            missing = required - set(archive.files)
            if missing:
                raise TaskCBenchmarkError(
                    f"CausalBench NPZ is missing arrays: {sorted(missing)}"
                )
            expression = np.asarray(archive["expression_matrix"])
            interventions = [str(value) for value in archive["interventions"].tolist()]
            gene_names = [str(value) for value in archive["var_names"].tolist()]
    except (OSError, ValueError) as exc:
        raise TaskCBenchmarkError(
            f"could not load CausalBench NPZ {source}: {exc}"
        ) from exc
    _validate_expression_matrix(expression)
    return expression, interventions, gene_names


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_task_c_mean_difference(
    *,
    expression: Any,
    interventions: Sequence[str],
    gene_names: Sequence[str],
    contract: Mapping[str, Any],
    dataset_id: str,
    dataset_source: str,
    context_id: str,
    data_status: str,
    input_digest: str,
    code_revision: str,
    random_seed: int,
    output_dir: str | Path,
    reference_edges: Iterable[tuple[str, str]] | None = None,
    reference_id: str | None = None,
    reference_digest: str | None = None,
    control_label: str = "non-targeting",
    excluded_label: str = "excluded",
    min_cells_per_intervention: int = 5,
    precision_at_k: int = 1000,
) -> dict[str, Any]:
    """Run the Task C simple baseline and write the required artifact bundle."""
    validate_benchmark_contract(contract)
    if data_status not in SUPPORTED_DATA_STATUSES:
        raise TaskCBenchmarkError(
            f"data_status must be one of {sorted(SUPPORTED_DATA_STATUSES)}"
        )
    if not isinstance(context_id, str) or not context_id.strip():
        raise TaskCBenchmarkError("context_id must be a non-empty string")
    if not isinstance(dataset_source, str) or not dataset_source.strip():
        raise TaskCBenchmarkError("dataset_source must be a non-empty string")
    reference_fields = (reference_edges, reference_id, reference_digest)
    if any(value is None for value in reference_fields) and any(
        value is not None for value in reference_fields
    ):
        raise TaskCBenchmarkError(
            "reference_edges, reference_id, and reference_digest must be supplied together"
        )
    if reference_id is not None and not str(reference_id).strip():
        raise TaskCBenchmarkError("reference_id must be a non-empty string")

    result = score_mean_difference_network(
        expression,
        interventions,
        gene_names,
        control_label=control_label,
        excluded_label=excluded_label,
        min_cells_per_intervention=min_cells_per_intervention,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    normalized_reference = (
        _normalize_reference_edges(reference_edges)
        if reference_edges is not None
        else None
    )
    predictions = result.scores.copy()
    if normalized_reference is None:
        predictions["reference_edge"] = pd.NA
    else:
        predictions["reference_edge"] = [
            (source, target) in normalized_reference
            for source, target in predictions[["source", "target"]].itertuples(
                index=False,
                name=None,
            )
        ]
    predictions.to_csv(destination / "predictions.csv", index=False)

    input_artifacts = {"causalbench_dataset": input_digest}
    if reference_digest is not None:
        input_artifacts["declared_reference"] = reference_digest
    manifest = build_run_manifest(
        contract,
        task_id="C",
        dataset_id=dataset_id,
        method_id="mean_difference",
        method_role="simple_baseline",
        code_revision=code_revision,
        random_seed=random_seed,
        input_artifacts=input_artifacts,
    )
    manifest.update(
        {
            "context_id": context_id,
            "dataset_source": dataset_source,
            "data_status": data_status,
            "causalbench_interface_compatible": True,
            "parameters": {
                "control_label": control_label,
                "excluded_label": excluded_label,
                "min_cells_per_intervention": min_cells_per_intervention,
                "score": "absolute_interventional_mean_difference",
                "tie_break": "target_name_ascending",
            },
        }
    )
    input_summary = {
        **result.summary,
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": "C",
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "context_id": context_id,
        "data_status": data_status,
        "input_digest": input_digest,
        "leakage_checks": {
            "reference_edges_used_for_scoring": False,
            "test_derived_feature_selection": False,
            "test_edges_used_as_priors": False,
        },
    }
    metrics: dict[str, Any] = {
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": "C",
        "method_id": "mean_difference",
        "status": "not_evaluated_no_reference",
        "primary_metric": "average_precision",
        "average_precision": None,
        "auroc": None,
        "precision_at_k": None,
        "precision_k": None,
        "reference_id": reference_id,
        "coverage": result.summary["coverage"],
        "abstention_rate": result.summary["abstention_rate"],
        "external_holdout_passed": False,
        "null_controls_passed": False,
        "null_control_status": "not_run",
        "interpretation": (
            "A declared reference network is incomplete evaluation evidence, "
            "not complete causal ground truth."
        ),
    }
    if normalized_reference is not None:
        metrics.update(
            evaluate_task_c_scores(
                result.scores,
                normalized_reference,
                precision_at_k=precision_at_k,
            )
        )
        metrics["status"] = "evaluated_against_declared_reference"

    promotion_decision = {
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": "C",
        "method_id": "mean_difference",
        "status": "not_applicable_simple_baseline",
        "claim_level": "baseline_only",
        "synthetic_smoke": data_status == "synthetic_smoke",
        "promotion_eligible": False,
        "reason": (
            "This artifact defines a required simple baseline. It cannot promote "
            "itself or establish a HyperSCA superiority claim."
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    _write_json(destination / "run_manifest.json", manifest)
    _write_json(destination / "input_summary.json", input_summary)
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "promotion_decision.json", promotion_decision)
    return {
        "manifest": manifest,
        "input_summary": input_summary,
        "metrics": metrics,
        "promotion_decision": promotion_decision,
        "predictions": predictions,
        "output_dir": destination,
    }
