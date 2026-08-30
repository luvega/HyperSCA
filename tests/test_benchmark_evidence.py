from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.evaluation.benchmark_evidence import (
    MetricUnitEvidence,
    PairedBootstrapSummary,
    bootstrap_causalbench_contexts,
    bootstrap_osta_units,
    build_causalbench_paired_unit,
    build_osta_paired_units,
    failed_metric_unit,
    holm_adjust_two_claims,
    macro_average_completed_units,
)


def test_osta_primary_metric_is_physical_to_embedding_knn_jaccard_by_block() -> None:
    physical = np.column_stack((np.arange(24, dtype=float), np.zeros(24, dtype=float)))
    hypersca = physical.copy()
    comparator_order = np.asarray(
        [
            0,
            12,
            1,
            13,
            2,
            14,
            3,
            15,
            4,
            16,
            5,
            17,
            6,
            18,
            7,
            19,
            8,
            20,
            9,
            21,
            10,
            22,
            11,
            23,
        ]
    )
    comparator = physical[comparator_order].copy()
    sample_ids = tuple("sample_a" for _ in range(24))
    block_ids = tuple("left" if index < 12 else "right" for index in range(24))
    platforms = tuple("visium_hd" for _ in range(24))

    units = build_osta_paired_units(
        physical_coordinates=physical,
        hypersca_embedding=hypersca,
        comparator_embedding=comparator,
        sample_ids=sample_ids,
        block_ids=block_ids,
        platform_ids=platforms,
        seed=11,
        comparator_id="euclidean_autoencoder",
        k=3,
    )

    assert [unit.unit_id for unit in units] == ["sample_a:left", "sample_a:right"]
    assert all(unit.metric_id == "neighborhood_preservation_at_k" for unit in units)
    assert all(unit.hypersca_value == pytest.approx(1.0) for unit in units)
    assert all(unit.comparator_value < unit.hypersca_value for unit in units)
    assert all(
        unit.paired_difference
        == pytest.approx(unit.hypersca_value - unit.comparator_value)
        for unit in units
    )
    assert all(unit.stratum_id == "visium_hd" for unit in units)


def test_osta_metric_rejects_cell_level_or_platform_identity_ambiguity() -> None:
    coordinates = np.column_stack((np.arange(8, dtype=float), np.zeros(8, dtype=float)))
    with pytest.raises(ValueError, match="same length"):
        build_osta_paired_units(
            physical_coordinates=coordinates,
            hypersca_embedding=coordinates,
            comparator_embedding=coordinates,
            sample_ids=("s",) * 7,
            block_ids=("b",) * 8,
            platform_ids=("p",) * 8,
            seed=11,
            comparator_id="euclidean_autoencoder",
            k=3,
        )
    with pytest.raises(ValueError, match="one platform"):
        build_osta_paired_units(
            physical_coordinates=coordinates,
            hypersca_embedding=coordinates,
            comparator_embedding=coordinates,
            sample_ids=("s",) * 8,
            block_ids=("b",) * 8,
            platform_ids=("p1",) * 4 + ("p2",) * 4,
            seed=11,
            comparator_id="euclidean_autoencoder",
            k=3,
        )


def test_causalbench_uses_complete_directed_universe_and_missing_scores_are_zero() -> (
    None
):
    result = build_causalbench_paired_unit(
        gene_names=("A", "B", "C"),
        hypersca_predictions=(("A", "B", 0.9), ("B", "C", 0.8)),
        comparator_predictions=(("A", "C", 0.7),),
        reference_edges=(("A", "B"), ("B", "C"), ("C", "A")),
        eligible_sources=("A", "B"),
        context_id="k562",
        seed=11,
        comparator_id="euclidean_autoencoder",
    )

    assert len(result.relation_scores) == 6
    assert all(row.source != row.target for row in result.relation_scores)
    missing = next(
        row for row in result.relation_scores if (row.source, row.target) == ("A", "C")
    )
    assert missing.hypersca_score == 0.0
    assert missing.hypersca_returned is False
    assert result.unit.metric_id == "directed_edge_average_precision"
    assert result.unit.stratum_id == "k562"
    assert result.unit.hypersca_value == pytest.approx(1.0)
    assert result.unit.comparator_value < result.unit.hypersca_value


def test_causalbench_ap_preserves_ties_and_ignores_prediction_row_order() -> None:
    predictions = (("A", "B", 1.0), ("A", "C", 1.0), ("B", "A", 1.0), ("B", "C", 1.0))
    kwargs = dict(
        gene_names=("A", "B", "C"),
        comparator_predictions=predictions,
        reference_edges=(("A", "B"), ("B", "C")),
        eligible_sources=("A", "B"),
        context_id="rpe1",
        seed=23,
        comparator_id="hypersca_without_hierarchy_loss",
    )
    forward = build_causalbench_paired_unit(hypersca_predictions=predictions, **kwargs)
    reversed_rows = build_causalbench_paired_unit(
        hypersca_predictions=predictions[::-1], **kwargs
    )

    assert forward.unit.hypersca_value == pytest.approx(0.5)
    assert reversed_rows.unit.hypersca_value == forward.unit.hypersca_value
    assert reversed_rows.relation_scores == forward.relation_scores


def test_causalbench_rejects_self_loops_duplicates_and_unknown_sources() -> None:
    base = dict(
        gene_names=("A", "B"),
        comparator_predictions=(),
        reference_edges=(("A", "B"),),
        eligible_sources=("A",),
        context_id="k562",
        seed=11,
        comparator_id="euclidean_autoencoder",
    )
    with pytest.raises(ValueError, match="self loops"):
        build_causalbench_paired_unit(hypersca_predictions=(("A", "A", 1.0),), **base)
    with pytest.raises(ValueError, match="duplicate"):
        build_causalbench_paired_unit(
            hypersca_predictions=(("A", "B", 0.4), ("A", "B", 0.5)), **base
        )
    with pytest.raises(ValueError, match="eligible_sources"):
        build_causalbench_paired_unit(
            hypersca_predictions=(), **{**base, "eligible_sources": ("Z",)}
        )


def test_causalbench_reference_network_is_intersected_with_the_frozen_gene_universe() -> (
    None
):
    result = build_causalbench_paired_unit(
        gene_names=("A", "B"),
        hypersca_predictions=(("A", "B", 1.0),),
        comparator_predictions=(),
        reference_edges=(("A", "B"), ("OUTSIDE", "A")),
        eligible_sources=("A",),
        context_id="k562",
        seed=11,
        comparator_id="euclidean_autoencoder",
    )

    assert result.unit.hypersca_value == 1.0
    assert len(result.relation_scores) == 2


def test_failed_units_are_retained_and_never_enter_completed_macro_average() -> None:
    completed = MetricUnitEvidence(
        benchmark_id="osta_colon",
        metric_id="neighborhood_preservation_at_k",
        unit_id="sample:block",
        stratum_id="visium_hd",
        cluster_id="sample",
        seed=11,
        comparator_id="euclidean_autoencoder",
        status="completed",
        hypersca_value=0.8,
        comparator_value=0.7,
        paired_difference=0.1,
    )
    failed = failed_metric_unit(
        benchmark_id="osta_colon",
        metric_id="neighborhood_preservation_at_k",
        unit_id="sample:block2",
        stratum_id="visium_hd",
        cluster_id="sample",
        seed=11,
        comparator_id="euclidean_autoencoder",
        status="failed_resource",
    )

    assert macro_average_completed_units((completed, failed)) == pytest.approx(0.1)
    assert failed.hypersca_value is None
    with pytest.raises(FrozenInstanceError):
        completed.status = "failed_runtime"  # type: ignore[misc]
    with pytest.raises(ValueError, match="completed units require"):
        MetricUnitEvidence(
            benchmark_id="osta_colon",
            metric_id="neighborhood_preservation_at_k",
            unit_id="bad",
            stratum_id="visium_hd",
            cluster_id="sample",
            seed=11,
            comparator_id="euclidean_autoencoder",
            status="completed",
            hypersca_value=None,
            comparator_value=None,
            paired_difference=None,
        )


def test_osta_bootstrap_averages_seeds_before_sample_then_block_resampling() -> None:
    units = tuple(
        MetricUnitEvidence(
            benchmark_id="osta_colon",
            metric_id="neighborhood_preservation_at_k",
            unit_id=f"sample_{sample}:block_{block}",
            stratum_id="visium_hd",
            cluster_id=f"sample_{sample}",
            seed=seed,
            comparator_id="euclidean_autoencoder",
            status="completed",
            hypersca_value=0.8 + 0.01 * seed_index,
            comparator_value=0.5,
            paired_difference=0.3 + 0.01 * seed_index,
        )
        for sample in range(3)
        for block in range(2)
        for seed_index, seed in enumerate((11, 23, 47))
    )

    first = bootstrap_osta_units(units, resamples=500, random_seed=191)
    second = bootstrap_osta_units(units, resamples=500, random_seed=191)

    assert first == second
    assert first.resampling_scheme == "osta_sample_then_block_seeds_averaged"
    assert first.estimate == pytest.approx(0.31)
    assert first.ci_low > 0.0
    assert first.one_sided_p_value == pytest.approx(1.0 / 501.0)
    assert first.resamples == 500


def test_causalbench_bootstrap_resamples_sources_within_context_after_seed_averaging() -> (
    None
):
    results = tuple(
        build_causalbench_paired_unit(
            gene_names=("A", "B", "C"),
            hypersca_predictions=(("A", "B", 0.9), ("B", "C", 0.8)),
            comparator_predictions=(("A", "C", 0.9), ("B", "A", 0.8)),
            reference_edges=(("A", "B"), ("B", "C")),
            eligible_sources=("A", "B"),
            context_id=context,
            seed=seed,
            comparator_id="euclidean_autoencoder",
        )
        for context in ("k562", "rpe1")
        for seed in (11, 23, 47)
    )

    summary = bootstrap_causalbench_contexts(results, resamples=500, random_seed=223)

    assert (
        summary.resampling_scheme
        == "causalbench_context_stratified_sources_seeds_averaged"
    )
    assert summary.estimate > 0.0
    assert summary.ci_low > 0.0
    assert summary.one_sided_p_value == pytest.approx(1.0 / 501.0)


def test_causalbench_bootstrap_retains_zero_positive_source_draws_as_zero_difference() -> (
    None
):
    results = tuple(
        build_causalbench_paired_unit(
            gene_names=("A", "B", "C"),
            hypersca_predictions=(("A", "B", 0.9),),
            comparator_predictions=(("A", "C", 0.9),),
            reference_edges=(("A", "B"),),
            eligible_sources=("A", "B"),
            context_id=context,
            seed=seed,
            comparator_id="euclidean_autoencoder",
        )
        for context in ("k562", "rpe1")
        for seed in (11, 23, 47)
    )

    summary = bootstrap_causalbench_contexts(results, resamples=200, random_seed=7)

    assert summary.estimate > 0.0
    assert summary.ci_low >= 0.0


def test_bootstrap_summaries_do_not_assume_percentile_ci_contains_point_estimate() -> (
    None
):
    summary = PairedBootstrapSummary(
        estimate=0.2,
        ci_low=0.3,
        ci_high=0.4,
        one_sided_p_value=0.01,
        resamples=10_000,
        random_seed=11,
        resampling_scheme="valid_skewed_percentile_bootstrap",
    )

    assert summary.ci_low > summary.estimate


def test_holm_adjustment_is_frozen_to_exactly_two_confirmatory_claims() -> None:
    assert holm_adjust_two_claims((0.01, 0.03)) == pytest.approx((0.02, 0.03))
    assert holm_adjust_two_claims((0.04, 0.01)) == pytest.approx((0.04, 0.02))
    with pytest.raises(ValueError, match="exactly two"):
        holm_adjust_two_claims((0.01,))
