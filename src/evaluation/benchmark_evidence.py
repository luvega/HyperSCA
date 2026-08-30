"""Pure benchmark adapters for preregistered HyperSCA claim evidence.

The functions in this module score already-produced model outputs.  They do
not train models, select settings, read files, or decide whether a scientific
claim is admitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "failed_invalid_input",
        "failed_invalid_output",
        "failed_timeout",
        "failed_resource",
        "failed_runtime",
        "not_applicable",
    }
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty built-in string")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


@dataclass(frozen=True, slots=True)
class MetricUnitEvidence:
    """One paired benchmark unit, including terminal failures."""

    benchmark_id: str
    metric_id: str
    unit_id: str
    stratum_id: str
    cluster_id: str
    seed: int
    comparator_id: str
    status: str
    hypersca_value: float | None
    comparator_value: float | None
    paired_difference: float | None

    def __post_init__(self) -> None:
        for name in (
            "benchmark_id",
            "metric_id",
            "unit_id",
            "stratum_id",
            "cluster_id",
            "comparator_id",
        ):
            _text(getattr(self, name), name)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative built-in integer")
        if type(self.status) is not str or self.status not in TERMINAL_STATUSES:
            raise ValueError("status must be a registered terminal status")
        values = (self.hypersca_value, self.comparator_value, self.paired_difference)
        if self.status == "completed":
            if any(value is None for value in values):
                raise ValueError("completed units require all metric values")
            hypersca_value = _finite_float(self.hypersca_value, "hypersca_value")
            comparator_value = _finite_float(self.comparator_value, "comparator_value")
            paired_difference = _finite_float(
                self.paired_difference, "paired_difference"
            )
            if not math.isclose(
                paired_difference,
                hypersca_value - comparator_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "paired_difference must equal hypersca_value - comparator_value"
                )
        elif any(value is not None for value in values):
            raise ValueError(
                "failed or not-applicable units must not contain fabricated metric values"
            )


def failed_metric_unit(
    *,
    benchmark_id: str,
    metric_id: str,
    unit_id: str,
    stratum_id: str,
    cluster_id: str,
    seed: int,
    comparator_id: str,
    status: str,
) -> MetricUnitEvidence:
    """Construct a retained terminal failure without a synthetic zero score."""

    if status == "completed":
        raise ValueError("failed_metric_unit cannot construct a completed unit")
    return MetricUnitEvidence(
        benchmark_id=benchmark_id,
        metric_id=metric_id,
        unit_id=unit_id,
        stratum_id=stratum_id,
        cluster_id=cluster_id,
        seed=seed,
        comparator_id=comparator_id,
        status=status,
        hypersca_value=None,
        comparator_value=None,
        paired_difference=None,
    )


def macro_average_completed_units(units: tuple[MetricUnitEvidence, ...]) -> float:
    """Macro-average paired differences while retaining failures in the input."""

    if type(units) is not tuple or any(
        type(unit) is not MetricUnitEvidence for unit in units
    ):
        raise ValueError("units must be a tuple of MetricUnitEvidence values")
    completed = [unit.paired_difference for unit in units if unit.status == "completed"]
    if not completed:
        raise ValueError("at least one completed metric unit is required")
    return float(np.mean(np.asarray(completed, dtype=np.float64)))


def _finite_matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return np.asarray(array, dtype=np.float64)


def _identity_tuple(value: object, name: str, length: int) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != length:
        raise ValueError(
            f"{name} must be a tuple with the same length as the coordinate arrays"
        )
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty built-in strings")
    return value


def _knn_sets(
    coordinates: np.ndarray, sample_ids: tuple[str, ...], k: int
) -> tuple[frozenset[int], ...]:
    neighbors: list[frozenset[int]] = [frozenset() for _ in range(coordinates.shape[0])]
    for sample_id in dict.fromkeys(sample_ids):
        indices = np.flatnonzero(
            np.asarray([value == sample_id for value in sample_ids], dtype=bool)
        )
        if len(indices) <= k:
            raise ValueError("each OSTA sample must contain more cells than k")
        sample = coordinates[indices]
        distances = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(distances, np.inf)
        for local_index, row in enumerate(distances):
            selected = np.argsort(row, kind="stable")[:k]
            neighbors[int(indices[local_index])] = frozenset(
                int(indices[item]) for item in selected
            )
    return tuple(neighbors)


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def build_osta_paired_units(
    *,
    physical_coordinates: object,
    hypersca_embedding: object,
    comparator_embedding: object,
    sample_ids: tuple[str, ...],
    block_ids: tuple[str, ...],
    platform_ids: tuple[str, ...],
    seed: int,
    comparator_id: str,
    k: int = 15,
) -> tuple[MetricUnitEvidence, ...]:
    """Score physical-coordinate versus embedding KNN preservation.

    Per-cell Jaccard scores are macro-averaged within held-out sample/block
    units.  K=15 is the confirmatory setting; callers may use K=5/30 for
    explicitly secondary sensitivity analyses.
    """

    physical = _finite_matrix(physical_coordinates, "physical_coordinates")
    hypersca = _finite_matrix(hypersca_embedding, "hypersca_embedding")
    comparator = _finite_matrix(comparator_embedding, "comparator_embedding")
    if (
        hypersca.shape[0] != physical.shape[0]
        or comparator.shape[0] != physical.shape[0]
    ):
        raise ValueError("all coordinate arrays must have the same number of rows")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive built-in integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative built-in integer")
    _text(comparator_id, "comparator_id")
    sample_ids = _identity_tuple(sample_ids, "sample_ids", physical.shape[0])
    block_ids = _identity_tuple(block_ids, "block_ids", physical.shape[0])
    platform_ids = _identity_tuple(platform_ids, "platform_ids", physical.shape[0])

    physical_neighbors = _knn_sets(physical, sample_ids, k)
    hypersca_neighbors = _knn_sets(hypersca, sample_ids, k)
    comparator_neighbors = _knn_sets(comparator, sample_ids, k)
    hypersca_scores = np.asarray(
        [
            _jaccard(physical_neighbors[index], hypersca_neighbors[index])
            for index in range(physical.shape[0])
        ]
    )
    comparator_scores = np.asarray(
        [
            _jaccard(physical_neighbors[index], comparator_neighbors[index])
            for index in range(physical.shape[0])
        ]
    )

    groups = sorted(set(zip(sample_ids, block_ids)))
    units: list[MetricUnitEvidence] = []
    for sample_id, block_id in groups:
        indices = np.asarray(
            [
                index
                for index, identity in enumerate(zip(sample_ids, block_ids))
                if identity == (sample_id, block_id)
            ],
            dtype=np.int64,
        )
        platforms = {platform_ids[index] for index in indices}
        if len(platforms) != 1:
            raise ValueError(
                "each OSTA sample/block unit must belong to exactly one platform"
            )
        hypersca_value = float(np.mean(hypersca_scores[indices]))
        comparator_value = float(np.mean(comparator_scores[indices]))
        units.append(
            MetricUnitEvidence(
                benchmark_id="osta_colon",
                metric_id="neighborhood_preservation_at_k",
                unit_id=f"{sample_id}:{block_id}",
                stratum_id=next(iter(platforms)),
                cluster_id=sample_id,
                seed=seed,
                comparator_id=comparator_id,
                status="completed",
                hypersca_value=hypersca_value,
                comparator_value=comparator_value,
                paired_difference=float(hypersca_value - comparator_value),
            )
        )
    return tuple(units)


@dataclass(frozen=True, slots=True)
class CausalRelationScore:
    source: str
    target: str
    is_reference_edge: bool
    hypersca_score: float
    comparator_score: float
    hypersca_returned: bool
    comparator_returned: bool

    def __post_init__(self) -> None:
        _text(self.source, "source")
        _text(self.target, "target")
        if self.source == self.target:
            raise ValueError("the scored relation universe excludes self loops")
        for name in ("is_reference_edge", "hypersca_returned", "comparator_returned"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a built-in boolean")
        for name in ("hypersca_score", "comparator_score"):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class CausalBenchmarkResult:
    unit: MetricUnitEvidence
    relation_scores: tuple[CausalRelationScore, ...]
    eligible_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.unit) is not MetricUnitEvidence
            or self.unit.metric_id != "directed_edge_average_precision"
        ):
            raise ValueError(
                "unit must contain directed-edge average precision evidence"
            )
        if type(self.relation_scores) is not tuple or any(
            type(row) is not CausalRelationScore for row in self.relation_scores
        ):
            raise ValueError(
                "relation_scores must be a tuple of CausalRelationScore values"
            )
        if type(self.eligible_sources) is not tuple or not self.eligible_sources:
            raise ValueError("eligible_sources must be a non-empty tuple")


@dataclass(frozen=True, slots=True)
class PairedBootstrapSummary:
    """A deterministic paired bootstrap summary for one comparator."""

    estimate: float
    ci_low: float
    ci_high: float
    one_sided_p_value: float
    resamples: int
    random_seed: int
    resampling_scheme: str

    def __post_init__(self) -> None:
        estimate = _finite_float(self.estimate, "estimate")
        ci_low = _finite_float(self.ci_low, "ci_low")
        ci_high = _finite_float(self.ci_high, "ci_high")
        p_value = _finite_float(self.one_sided_p_value, "one_sided_p_value")
        if ci_low > ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("one_sided_p_value must lie in [0, 1]")
        if type(self.resamples) is not int or not 1 <= self.resamples <= 1_000_000:
            raise ValueError("resamples must be a built-in integer in [1, 1000000]")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative built-in integer")
        _text(self.resampling_scheme, "resampling_scheme")


def _bootstrap_summary(
    estimate: float,
    sampled_differences: np.ndarray,
    *,
    resamples: int,
    random_seed: int,
    scheme: str,
) -> PairedBootstrapSummary:
    ci_low, ci_high = np.quantile(sampled_differences, (0.025, 0.975))
    p_value = (1.0 + float(np.count_nonzero(sampled_differences <= 0.0))) / (
        resamples + 1.0
    )
    return PairedBootstrapSummary(
        estimate=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        one_sided_p_value=float(p_value),
        resamples=resamples,
        random_seed=random_seed,
        resampling_scheme=scheme,
    )


def _bootstrap_arguments(resamples: object, random_seed: object) -> tuple[int, int]:
    if type(resamples) is not int or not 1 <= resamples <= 1_000_000:
        raise ValueError("resamples must be a built-in integer in [1, 1000000]")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a non-negative built-in integer")
    return resamples, random_seed


def bootstrap_osta_units(
    units: tuple[MetricUnitEvidence, ...],
    *,
    resamples: int = 10_000,
    random_seed: int = 19_911,
) -> PairedBootstrapSummary:
    """Average seeds, then resample OSTA samples and nested held-out blocks."""

    resamples, random_seed = _bootstrap_arguments(resamples, random_seed)
    if (
        type(units) is not tuple
        or not units
        or any(type(unit) is not MetricUnitEvidence for unit in units)
    ):
        raise ValueError("units must be a non-empty tuple of MetricUnitEvidence values")
    first = units[0]
    if any(
        unit.benchmark_id != "osta_colon"
        or unit.metric_id != "neighborhood_preservation_at_k"
        or unit.comparator_id != first.comparator_id
        for unit in units
    ):
        raise ValueError(
            "OSTA bootstrap units must share the frozen benchmark, metric, and comparator"
        )
    if any(unit.status != "completed" for unit in units):
        raise ValueError(
            "OSTA confirmatory bootstrap requires every attempted unit to be completed"
        )

    grouped: dict[tuple[str, str, str], dict[int, float]] = {}
    for unit in units:
        key = (unit.stratum_id, unit.cluster_id, unit.unit_id)
        by_seed = grouped.setdefault(key, {})
        if unit.seed in by_seed:
            raise ValueError(
                "OSTA bootstrap contains a duplicate seed for a sample/block unit"
            )
        assert unit.paired_difference is not None
        by_seed[unit.seed] = unit.paired_difference
    seed_sets = {tuple(sorted(by_seed)) for by_seed in grouped.values()}
    if len(seed_sets) != 1:
        raise ValueError(
            "OSTA sample/block units must contain the same frozen seed set"
        )

    hierarchy: dict[str, dict[str, list[float]]] = {}
    for (stratum, cluster, _unit_id), by_seed in sorted(grouped.items()):
        hierarchy.setdefault(stratum, {}).setdefault(cluster, []).append(
            float(np.mean([by_seed[seed] for seed in sorted(by_seed)]))
        )
    estimate = float(
        np.mean(
            [
                np.mean([np.mean(blocks) for blocks in clusters.values()])
                for _stratum, clusters in sorted(hierarchy.items())
            ]
        )
    )
    rng = np.random.default_rng(random_seed)
    sampled = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        stratum_values: list[float] = []
        for _stratum, clusters in sorted(hierarchy.items()):
            cluster_names = tuple(sorted(clusters))
            selected_clusters = rng.integers(
                0, len(cluster_names), size=len(cluster_names)
            )
            cluster_values: list[float] = []
            for selected in selected_clusters:
                blocks = np.asarray(
                    clusters[cluster_names[int(selected)]], dtype=np.float64
                )
                selected_blocks = rng.integers(0, len(blocks), size=len(blocks))
                cluster_values.append(float(np.mean(blocks[selected_blocks])))
            stratum_values.append(float(np.mean(cluster_values)))
        sampled[sample_index] = float(np.mean(stratum_values))
    return _bootstrap_summary(
        estimate,
        sampled,
        resamples=resamples,
        random_seed=random_seed,
        scheme="osta_sample_then_block_seeds_averaged",
    )


def bootstrap_causalbench_contexts(
    results: tuple[CausalBenchmarkResult, ...],
    *,
    resamples: int = 10_000,
    random_seed: int = 22_311,
) -> PairedBootstrapSummary:
    """Average seed scores, then resample eligible sources within contexts."""

    resamples, random_seed = _bootstrap_arguments(resamples, random_seed)
    if (
        type(results) is not tuple
        or not results
        or any(type(result) is not CausalBenchmarkResult for result in results)
    ):
        raise ValueError(
            "results must be a non-empty tuple of CausalBenchmarkResult values"
        )
    first = results[0]
    if any(result.unit.status != "completed" for result in results):
        raise ValueError(
            "CausalBench confirmatory bootstrap requires every attempted context/seed to complete"
        )
    if any(result.unit.comparator_id != first.unit.comparator_id for result in results):
        raise ValueError("CausalBench bootstrap results must use one comparator")

    contexts: dict[str, dict[int, CausalBenchmarkResult]] = {}
    for result in results:
        context = result.unit.stratum_id
        by_seed = contexts.setdefault(context, {})
        if result.unit.seed in by_seed:
            raise ValueError(
                "CausalBench bootstrap contains a duplicate context/seed result"
            )
        by_seed[result.unit.seed] = result
    seed_sets = {tuple(sorted(by_seed)) for by_seed in contexts.values()}
    if len(seed_sets) != 1:
        raise ValueError("CausalBench contexts must contain the same frozen seed set")

    averaged: dict[
        str,
        tuple[tuple[str, ...], tuple[CausalRelationScore, ...], np.ndarray, np.ndarray],
    ] = {}
    context_estimates: list[float] = []
    for context, by_seed in sorted(contexts.items()):
        ordered = tuple(by_seed[seed] for seed in sorted(by_seed))
        template = ordered[0]
        if any(
            result.eligible_sources != template.eligible_sources
            or tuple(
                (row.source, row.target, row.is_reference_edge)
                for row in result.relation_scores
            )
            != tuple(
                (row.source, row.target, row.is_reference_edge)
                for row in template.relation_scores
            )
            for result in ordered[1:]
        ):
            raise ValueError(
                "CausalBench context seeds must share relation and eligible-source identities"
            )
        hypersca_scores = np.mean(
            np.asarray(
                [
                    [row.hypersca_score for row in result.relation_scores]
                    for result in ordered
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        comparator_scores = np.mean(
            np.asarray(
                [
                    [row.comparator_score for row in result.relation_scores]
                    for result in ordered
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        eligible = frozenset(template.eligible_sources)
        indices = np.asarray(
            [
                index
                for index, row in enumerate(template.relation_scores)
                if row.source in eligible
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [template.relation_scores[index].is_reference_edge for index in indices],
            dtype=np.int64,
        )
        context_estimates.append(
            _average_precision(labels, hypersca_scores[indices])
            - _average_precision(labels, comparator_scores[indices])
        )
        averaged[context] = (
            template.eligible_sources,
            template.relation_scores,
            hypersca_scores,
            comparator_scores,
        )

    estimate = float(np.mean(context_estimates))
    rng = np.random.default_rng(random_seed)
    sampled = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        context_differences: list[float] = []
        for _context, (
            eligible_sources,
            rows,
            hypersca_scores,
            comparator_scores,
        ) in sorted(averaged.items()):
            selected_sources = rng.integers(
                0, len(eligible_sources), size=len(eligible_sources)
            )
            selected_indices: list[int] = []
            for selected in selected_sources:
                source = eligible_sources[int(selected)]
                selected_indices.extend(
                    index for index, row in enumerate(rows) if row.source == source
                )
            indices = np.asarray(selected_indices, dtype=np.int64)
            labels = np.asarray(
                [rows[index].is_reference_edge for index in indices], dtype=np.int64
            )
            if int(labels.sum()) == 0:
                context_differences.append(0.0)
            else:
                context_differences.append(
                    _average_precision(labels, hypersca_scores[indices])
                    - _average_precision(labels, comparator_scores[indices])
                )
        sampled[sample_index] = float(np.mean(context_differences))
    return _bootstrap_summary(
        estimate,
        sampled,
        resamples=resamples,
        random_seed=random_seed,
        scheme="causalbench_context_stratified_sources_seeds_averaged",
    )


def holm_adjust_two_claims(p_values: tuple[float, float]) -> tuple[float, float]:
    """Apply Holm's step-down correction to the two confirmatory claims."""

    if type(p_values) is not tuple or len(p_values) != 2:
        raise ValueError("Holm correction is frozen to exactly two confirmatory claims")
    validated = tuple(_finite_float(value, "p_value") for value in p_values)
    if any(not 0.0 <= value <= 1.0 for value in validated):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(2), key=lambda index: validated[index])
    adjusted = [0.0, 0.0]
    first_adjusted = min(1.0, 2.0 * validated[order[0]])
    second_adjusted = min(1.0, max(first_adjusted, validated[order[1]]))
    adjusted[order[0]] = first_adjusted
    adjusted[order[1]] = second_adjusted
    return float(adjusted[0]), float(adjusted[1])


def _genes(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("gene_names must be a non-empty tuple")
    if any(type(gene) is not str or not gene.strip() for gene in value):
        raise ValueError("gene_names must contain non-empty built-in strings")
    if len(set(value)) != len(value):
        raise ValueError("gene_names must be unique")
    return value


def _prediction_map(
    predictions: object, genes: frozenset[str], name: str
) -> dict[tuple[str, str], float]:
    if type(predictions) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    result: dict[tuple[str, str], float] = {}
    for row in predictions:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError(f"{name} rows must be (source, target, score) tuples")
        source, target, score = row
        if (
            type(source) is not str
            or type(target) is not str
            or source not in genes
            or target not in genes
        ):
            raise ValueError(f"{name} contains an endpoint outside gene_names")
        if source == target:
            raise ValueError(f"{name} must not contain self loops")
        if type(score) is not float or not math.isfinite(score) or score < 0.0:
            raise ValueError(
                f"{name} scores must be finite non-negative built-in floats"
            )
        edge = (source, target)
        if edge in result:
            raise ValueError(f"{name} contains a duplicate relation")
        result[edge] = score
    return result


def _reference_set(
    reference_edges: object, genes: frozenset[str]
) -> frozenset[tuple[str, str]]:
    if type(reference_edges) is not tuple:
        raise ValueError("reference_edges must be a tuple")
    result: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for edge in reference_edges:
        if type(edge) is not tuple or len(edge) != 2:
            raise ValueError("reference_edges rows must be (source, target) tuples")
        source, target = edge
        if (
            type(source) is not str
            or type(target) is not str
            or not source.strip()
            or not target.strip()
        ):
            raise ValueError(
                "reference_edges endpoints must be non-empty built-in strings"
            )
        if source == target:
            raise ValueError("reference_edges must not contain self loops")
        if edge in seen:
            raise ValueError("reference_edges contains a duplicate relation")
        seen.add(edge)
        if source in genes and target in genes:
            result.add(edge)
    return frozenset(result)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives <= 0:
        raise ValueError("the eligible relation universe contains no reference edge")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels, dtype=np.int64)
    false_positives = np.cumsum(1 - sorted_labels, dtype=np.int64)
    threshold_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    precision = true_positives[threshold_ends] / (
        true_positives[threshold_ends] + false_positives[threshold_ends]
    )
    recall = true_positives[threshold_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def build_causalbench_paired_unit(
    *,
    gene_names: tuple[str, ...],
    hypersca_predictions: tuple[tuple[str, str, float], ...],
    comparator_predictions: tuple[tuple[str, str, float], ...],
    reference_edges: tuple[tuple[str, str], ...],
    eligible_sources: tuple[str, ...],
    context_id: str,
    seed: int,
    comparator_id: str,
) -> CausalBenchmarkResult:
    """Score the complete p(p-1) directed relation universe for one context."""

    genes = _genes(gene_names)
    gene_set = frozenset(genes)
    if type(eligible_sources) is not tuple or not eligible_sources:
        raise ValueError("eligible_sources must be a non-empty tuple")
    if any(
        type(source) is not str or source not in gene_set for source in eligible_sources
    ):
        raise ValueError("eligible_sources must be unique members of gene_names")
    if len(set(eligible_sources)) != len(eligible_sources):
        raise ValueError("eligible_sources must be unique members of gene_names")
    _text(context_id, "context_id")
    _text(comparator_id, "comparator_id")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative built-in integer")
    hypersca = _prediction_map(hypersca_predictions, gene_set, "hypersca_predictions")
    comparator = _prediction_map(
        comparator_predictions, gene_set, "comparator_predictions"
    )
    references = _reference_set(reference_edges, gene_set)

    rows = tuple(
        CausalRelationScore(
            source=source,
            target=target,
            is_reference_edge=(source, target) in references,
            hypersca_score=float(hypersca.get((source, target), 0.0)),
            comparator_score=float(comparator.get((source, target), 0.0)),
            hypersca_returned=(source, target) in hypersca,
            comparator_returned=(source, target) in comparator,
        )
        for source in genes
        for target in genes
        if source != target
    )
    eligible_set = frozenset(eligible_sources)
    scored = tuple(row for row in rows if row.source in eligible_set)
    labels = np.asarray([row.is_reference_edge for row in scored], dtype=np.int64)
    hypersca_scores = np.asarray(
        [row.hypersca_score for row in scored], dtype=np.float64
    )
    comparator_scores = np.asarray(
        [row.comparator_score for row in scored], dtype=np.float64
    )
    hypersca_ap = _average_precision(labels, hypersca_scores)
    comparator_ap = _average_precision(labels, comparator_scores)
    unit = MetricUnitEvidence(
        benchmark_id="causalbench_k562_rpe1",
        metric_id="directed_edge_average_precision",
        unit_id=context_id,
        stratum_id=context_id,
        cluster_id="eligible_intervention_sources",
        seed=seed,
        comparator_id=comparator_id,
        status="completed",
        hypersca_value=hypersca_ap,
        comparator_value=comparator_ap,
        paired_difference=float(hypersca_ap - comparator_ap),
    )
    return CausalBenchmarkResult(
        unit=unit, relation_scores=rows, eligible_sources=eligible_sources
    )


__all__ = [
    "CausalBenchmarkResult",
    "CausalRelationScore",
    "MetricUnitEvidence",
    "PairedBootstrapSummary",
    "bootstrap_causalbench_contexts",
    "bootstrap_osta_units",
    "build_causalbench_paired_unit",
    "build_osta_paired_units",
    "failed_metric_unit",
    "holm_adjust_two_claims",
    "macro_average_completed_units",
]
