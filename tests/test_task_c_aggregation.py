from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_aggregation import (
    TaskCAggregationError,
    aggregate_task_c_runs,
    evaluate_declared_references,
    paired_cluster_interval,
    task_c_aggregation_to_jsonable,
)


PASSED_FILES = {
    "method_status.json",
    "metrics.json",
    "predictions.csv",
    "run_manifest.json",
    "input_summary.json",
    "promotion_decision.json",
    "resource_usage.json",
    "environment_manifest.json",
}
FAILED_FILES = {
    "method_status.json",
    "resource_usage.json",
    "environment_manifest.json",
}


def _complete_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C", "C"],
            "target": ["B", "C", "A", "C", "A", "B"],
            "score": [0.9, 0.1, 0.2, 0.8, 0.0, 0.0],
            "effect": [1.0, -0.1, -0.2, 0.7, 0.0, 0.0],
        }
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_run(
    root: Path,
    *,
    method_id: str,
    status: str,
    condition: str = "within_k562",
    seed: int = 11,
    identity_character: str = "a",
) -> Path:
    root.mkdir()
    status_payload = {
        "schema_version": "1.0",
        "method_id": method_id,
        "condition": condition,
        "seed": seed,
        "run_identity_sha256": "sha256:" + identity_character * 64,
        "status": status,
        "controller_validation": "verified_task_c_rehearsal_bundle_v1",
    }
    _write_json(root / "method_status.json", status_payload)
    expected = PASSED_FILES if status == "passed_real_rehearsal" else FAILED_FILES
    for name in expected - {"method_status.json"}:
        if name == "metrics.json":
            _write_json(root / name, {"average_precision": 0.2})
        elif name.endswith(".json"):
            _write_json(root / name, {"schema_version": "1.0"})
        else:
            (root / name).write_text(
                "source,target,score\nA,B,0.2\n", encoding="utf-8"
            )
    return root


def test_primary_reference_and_directed_reference_are_reported_separately() -> None:
    metrics = evaluate_declared_references(
        _complete_scores(),
        pooled_reference={("A", "B"), ("B", "A")},
        directed_chip_reference={("A", "B")},
        eligible_sources={"A"},
        directed_reference_context_match=True,
        precision_values=(2, 5),
    )

    assert isinstance(metrics, MappingProxyType)
    assert metrics["primary_reference_id"] == "causalbench_pooled_biological_v1"
    assert metrics["average_precision"] > 0.0
    assert metrics["directed_chip_edge_count"] == 1
    assert metrics["edge_direction_accuracy"] == 1.0
    with pytest.raises(TypeError):
        metrics["average_precision"] = 0.0  # type: ignore[index]


def test_precision_at_k_uses_frozen_stable_relation_order_for_cutoff_ties() -> None:
    scores = _complete_scores()
    scores["score"] = [1.0, 1.0, 1.0, 0.4, 0.3, 0.2]
    metrics = evaluate_declared_references(
        scores,
        pooled_reference={("A", "B")},
        directed_chip_reference={("A", "B")},
        eligible_sources={"A"},
        directed_reference_context_match=True,
        precision_values=(1,),
    )

    assert metrics["precision_at_1"] == pytest.approx(1.0)
    assert metrics["precision_at_1_tie_aware_sensitivity"] == pytest.approx(0.5)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["edge_direction_accuracy"] == pytest.approx(0.0)
    assert metrics["direction_tie_credit"] == pytest.approx(0.0)
    assert task_c_aggregation_to_jsonable(metrics)["precision_at_1"] == 1.0


def test_direction_metric_is_withheld_when_cell_environment_does_not_match() -> None:
    metrics = evaluate_declared_references(
        _complete_scores(),
        pooled_reference={("A", "B")},
        directed_chip_reference={("A", "B")},
        eligible_sources={"A"},
        directed_reference_context_match=False,
        precision_values=(2,),
    )

    assert metrics["directed_chip_edge_count"] == 1
    assert metrics["edge_direction_accuracy"] is None


def test_standard_prediction_schema_without_effect_is_accepted() -> None:
    scores = _complete_scores().drop(columns="effect")
    scores["returned_by_method"] = [True, True, True, True, False, False]

    metrics = evaluate_declared_references(
        scores,
        pooled_reference={("A", "B")},
        directed_chip_reference={("A", "B")},
        eligible_sources={"A"},
        directed_reference_context_match=True,
        precision_values=(2,),
    )

    assert metrics["edge_direction_accuracy"] == 1.0


def test_unreturned_relations_must_keep_positive_zero_scores() -> None:
    scores = _complete_scores()
    scores["returned_by_method"] = [False] * len(scores)
    with pytest.raises(TaskCAggregationError, match="unreturned"):
        evaluate_declared_references(
            scores,
            pooled_reference={("A", "B")},
            directed_chip_reference={("A", "B")},
            eligible_sources={"A"},
            directed_reference_context_match=True,
        )

    scores["score"] = 0.0
    metrics = evaluate_declared_references(
        scores,
        pooled_reference={("A", "B")},
        directed_chip_reference={("A", "B")},
        eligible_sources={"A"},
        directed_reference_context_match=True,
        precision_values=(2,),
    )
    assert metrics["edge_direction_accuracy"] == 0.0


def test_score_and_effect_columns_reject_numeric_text_or_boolean_values() -> None:
    numeric_text = _complete_scores()
    numeric_text["score"] = numeric_text["score"].astype(str)
    with pytest.raises(TaskCAggregationError, match="numeric values"):
        evaluate_declared_references(
            numeric_text,
            pooled_reference={("A", "B")},
            directed_chip_reference=set(),
            eligible_sources={"A"},
            directed_reference_context_match=False,
        )

    boolean_effect = _complete_scores()
    boolean_effect["effect"] = [True] * len(boolean_effect)
    with pytest.raises(TaskCAggregationError, match="numeric values"):
        evaluate_declared_references(
            boolean_effect,
            pooled_reference={("A", "B")},
            directed_chip_reference=set(),
            eligible_sources={"A"},
            directed_reference_context_match=False,
        )


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (lambda table: table.iloc[:-1].copy(), "every directed"),
        (
            lambda table: table.assign(unknown_column=1),
            "column policy",
        ),
        (
            lambda table: table.assign(source=["A", "A", "B", "B", "C", "e\u0301"]),
            "NFC",
        ),
        (
            lambda table: table.assign(score=[0.9, 0.1, 0.2, 0.8, 0.0, float("inf")]),
            "finite",
        ),
        (
            lambda table: table.assign(effect=[1.0, -0.1, -0.2, 0.7, 0.0, float("nan")]),
            "finite",
        ),
    ],
)
def test_score_table_must_be_one_complete_finite_directed_universe(
    changed, message: str
) -> None:
    with pytest.raises(TaskCAggregationError, match=message):
        evaluate_declared_references(
            changed(_complete_scores()),
            pooled_reference={("A", "B")},
            directed_chip_reference=set(),
            eligible_sources={"A"},
            directed_reference_context_match=False,
            precision_values=(2,),
        )


def test_reference_relations_outside_scored_gene_set_are_ignored() -> None:
    metrics = evaluate_declared_references(
        _complete_scores(),
        pooled_reference={("A", "B"), ("A", "D")},
        directed_chip_reference={("A", "B"), ("A", "D")},
        eligible_sources={"A"},
        directed_reference_context_match=True,
        precision_values=(2,),
    )

    assert metrics["n_reference_edges_in_universe"] == 1
    assert metrics["directed_chip_edge_count"] == 1


def test_precision_limit_is_disclosed_as_name_sensitive_when_scores_tie() -> None:
    scores = _complete_scores()
    scores["score"] = 0.0
    metrics = evaluate_declared_references(
        scores,
        pooled_reference={("A", "B")},
        directed_chip_reference=set(),
        eligible_sources={"A"},
        directed_reference_context_match=False,
        precision_values=(1,),
    )

    assert metrics["precision_metric_role"] == "sensitivity_analysis"
    assert metrics["primary_ranking_metric"] == "average_precision"
    assert "gene names" in metrics["precision_tie_limitation"]


def test_eligible_sources_must_stay_inside_scored_gene_set() -> None:
    with pytest.raises(TaskCAggregationError, match="subset"):
        evaluate_declared_references(
            _complete_scores(),
            pooled_reference={("A", "B")},
            directed_chip_reference=set(),
            eligible_sources={"D"},
            directed_reference_context_match=False,
        )
    with pytest.raises(TaskCAggregationError, match="no relation"):
        evaluate_declared_references(
            _complete_scores(),
            pooled_reference={("B", "C"), ("A", "D")},
            directed_chip_reference=set(),
            eligible_sources={"A"},
            directed_reference_context_match=False,
        )


def test_paired_cluster_interval_resamples_seed_condition_pairs() -> None:
    table = pd.DataFrame(
        {
            "seed": [11, 11, 23, 23, 47, 47],
            "condition": ["k562", "rpe1"] * 3,
            "candidate": [0.5, 0.6, 0.55, 0.65, 0.52, 0.62],
            "baseline": [0.4, 0.5, 0.45, 0.55, 0.42, 0.52],
        }
    )
    state_before = np.random.get_state()

    interval = paired_cluster_interval(table, repeats=1000, seed=11)

    state_after = np.random.get_state()
    assert isinstance(interval, MappingProxyType)
    assert interval["estimate"] == pytest.approx(0.1)
    assert interval["ci_lower"] > 0.0
    assert interval["cluster_count"] == 6
    assert interval["attempted_cluster_count"] == 6
    assert interval["dropped_cluster_count"] == 0
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_paired_interval_drops_incomplete_pairs_but_keeps_attempted_denominator() -> None:
    table = pd.DataFrame(
        {
            "seed": [11, 23, 47],
            "condition": ["k562", "k562", "k562"],
            "candidate": [0.5, None, 0.7],
            "baseline": [0.4, 0.3, None],
        }
    )

    interval = paired_cluster_interval(table, repeats=100, seed=23)

    assert interval["estimate"] == pytest.approx(0.1)
    assert interval["attempted_cluster_count"] == 3
    assert interval["cluster_count"] == 1
    assert interval["dropped_cluster_count"] == 2


def test_paired_interval_rejects_duplicate_nonfinite_or_unbounded_clusters() -> None:
    duplicate = pd.DataFrame(
        {
            "seed": [11, 11],
            "condition": ["k562", "k562"],
            "candidate": [0.5, 0.6],
            "baseline": [0.4, 0.5],
        }
    )
    with pytest.raises(TaskCAggregationError, match="unique seed-condition"):
        paired_cluster_interval(duplicate, repeats=100)

    nonfinite = duplicate.iloc[:1].copy()
    nonfinite.loc[0, "candidate"] = float("inf")
    with pytest.raises(TaskCAggregationError, match="finite"):
        paired_cluster_interval(nonfinite, repeats=100)

    with pytest.raises(TaskCAggregationError, match="repeats"):
        paired_cluster_interval(duplicate.iloc[:1], repeats=0)
    with pytest.raises(TaskCAggregationError, match="upper limit"):
        paired_cluster_interval(duplicate.iloc[:1], repeats=1_000_001)

    nonscalar = duplicate.iloc[:1].astype(object)
    nonscalar.at[0, "candidate"] = [0.5]
    with pytest.raises(TaskCAggregationError, match="finite numbers"):
        paired_cluster_interval(nonscalar, repeats=100)

    duplicate_columns = duplicate.iloc[:1].copy()
    duplicate_columns.columns = ["seed", "condition", "candidate", "candidate"]
    with pytest.raises(TaskCAggregationError, match="column names"):
        paired_cluster_interval(duplicate_columns, repeats=100)


def test_failed_runs_remain_in_read_only_method_summary(tmp_path: Path) -> None:
    passed = _make_run(
        tmp_path / "passed",
        method_id="pc",
        status="passed_real_rehearsal",
    )
    failed = _make_run(
        tmp_path / "failed",
        method_id="gies",
        status="failed_timeout",
        identity_character="b",
    )

    summary = aggregate_task_c_runs([passed, failed])

    assert isinstance(summary, MappingProxyType)
    assert summary["attempted_run_count"] == 2
    assert summary["completed_run_count"] == 1
    assert summary["failed_or_unavailable_count"] == 1
    assert summary["status_counts"]["failed_timeout"] == 1
    assert summary["validation_scope"] == (
        "formal completion requires the verified Task C rehearsal bundle "
        "declaration; synthetic smoke and explicit legacy mode are structural only"
    )
    assert isinstance(summary["runs"], tuple)
    assert [run["method_id"] for run in summary["runs"]] == ["pc", "gies"]
    with pytest.raises(TypeError):
        summary["status_counts"]["failed_timeout"] = 2  # type: ignore[index]
    json.dumps(task_c_aggregation_to_jsonable(summary), allow_nan=False)


def test_original_plan_minimal_run_records_are_accepted(tmp_path: Path) -> None:
    passed = tmp_path / "passed"
    failed = tmp_path / "failed"
    passed.mkdir()
    failed.mkdir()
    _write_json(
        passed / "method_status.json",
        {"method_id": "pc", "status": "passed_real_rehearsal"},
    )
    _write_json(passed / "metrics.json", {"average_precision": 0.2})
    _write_json(
        failed / "method_status.json",
        {"method_id": "gies", "status": "failed_timeout"},
    )

    summary = aggregate_task_c_runs(
        [passed, failed], allow_legacy_minimal=True
    )

    assert summary["attempted_run_count"] == 2
    assert summary["completed_run_count"] == 0
    assert summary["verified_completed_run_count"] == 0
    assert summary["legacy_structural_completed_count"] == 1
    assert summary["not_formally_completed_count"] == 2
    assert summary["runs"][0]["controller_validation_status"] == (
        "unverified_legacy_record"
    )
    assert summary["status_counts"]["failed_timeout"] == 1


def test_extended_status_counts_as_controller_validated_completion(tmp_path: Path) -> None:
    passed = _make_run(
        tmp_path / "passed",
        method_id="pc",
        status="passed_real_rehearsal",
    )

    summary = aggregate_task_c_runs([passed])

    assert summary["completed_run_count"] == 1
    assert summary["verified_completed_run_count"] == 1
    assert summary["legacy_structural_completed_count"] == 0
    assert summary["runs"][0]["controller_validation_status"] == (
        "verified_task_c_rehearsal_bundle_v1"
    )


def test_minimal_status_is_rejected_unless_legacy_mode_is_explicit(tmp_path: Path) -> None:
    run = tmp_path / "minimal"
    run.mkdir()
    _write_json(
        run / "method_status.json",
        {"method_id": "pc", "status": "failed_timeout"},
    )

    with pytest.raises(TaskCAggregationError, match="legacy minimal"):
        aggregate_task_c_runs([run])


def test_aggregation_rejects_duplicate_run_identity_or_seed_condition_method(
    tmp_path: Path,
) -> None:
    first = _make_run(
        tmp_path / "first",
        method_id="pc",
        status="failed_timeout",
    )
    second = _make_run(
        tmp_path / "second",
        method_id="pc",
        status="failed_launch",
        identity_character="b",
    )
    with pytest.raises(TaskCAggregationError, match="method-condition-seed"):
        aggregate_task_c_runs([first, second])
    with pytest.raises(TaskCAggregationError, match="duplicate run directory"):
        aggregate_task_c_runs([first, first])


def test_aggregation_requires_minimum_files_and_rejects_failure_metrics(
    tmp_path: Path,
) -> None:
    passed = _make_run(
        tmp_path / "passed",
        method_id="pc",
        status="passed_real_rehearsal",
    )
    (passed / "metrics.json").unlink()
    with pytest.raises(TaskCAggregationError, match="file set"):
        aggregate_task_c_runs([passed])

    failed = _make_run(
        tmp_path / "failed",
        method_id="gies",
        status="failed_timeout",
    )
    _write_json(failed / "metrics.json", {"average_precision": 0.9})
    with pytest.raises(TaskCAggregationError, match="file set"):
        aggregate_task_c_runs([failed])


def test_aggregation_allows_known_controller_files_but_rejects_unknown_extras(
    tmp_path: Path,
) -> None:
    passed = tmp_path / "passed"
    passed.mkdir()
    _write_json(
        passed / "method_status.json",
        {"method_id": "pc", "status": "passed_real_rehearsal"},
    )
    _write_json(passed / "metrics.json", {"average_precision": 0.2})
    _write_json(passed / "run_manifest.json", {"schema_version": "1.0"})
    _write_json(passed / "resource_usage.json", {"schema_version": "1.0"})
    summary = aggregate_task_c_runs([passed], allow_legacy_minimal=True)
    assert summary["legacy_structural_completed_count"] == 1
    assert summary["completed_run_count"] == 0

    (passed / "unregistered-diagnostic.txt").write_text("surprise", encoding="utf-8")
    with pytest.raises(TaskCAggregationError, match="file set"):
        aggregate_task_c_runs([passed], allow_legacy_minimal=True)


def test_aggregation_rejects_untrusted_or_malformed_status_and_metrics(
    tmp_path: Path,
) -> None:
    unvalidated = _make_run(
        tmp_path / "unvalidated",
        method_id="pc",
        status="failed_timeout",
    )
    payload = json.loads((unvalidated / "method_status.json").read_text())
    payload["controller_validation"] = "self_reported"
    _write_json(unvalidated / "method_status.json", payload)
    with pytest.raises(TaskCAggregationError, match="controller validation"):
        aggregate_task_c_runs([unvalidated])

    malformed = _make_run(
        tmp_path / "malformed",
        method_id="pc",
        status="passed_real_rehearsal",
    )
    (malformed / "metrics.json").write_text(
        '{"average_precision":0.2,"average_precision":0.9}\n',
        encoding="utf-8",
    )
    with pytest.raises(TaskCAggregationError, match="duplicate field"):
        aggregate_task_c_runs([malformed])

    nonfinite = _make_run(
        tmp_path / "nonfinite",
        method_id="pc",
        status="passed_real_rehearsal",
    )
    (nonfinite / "metrics.json").write_text(
        '{"average_precision":NaN}\n', encoding="utf-8"
    )
    with pytest.raises(TaskCAggregationError, match="non-finite"):
        aggregate_task_c_runs([nonfinite])

    huge = _make_run(
        tmp_path / "huge",
        method_id="pc",
        status="passed_real_rehearsal",
    )
    (huge / "metrics.json").write_text(
        '{"average_precision":' + "9" * 1_000 + "}\n", encoding="utf-8"
    )
    with pytest.raises(TaskCAggregationError, match="average_precision"):
        aggregate_task_c_runs([huge])


def test_precision_values_numpy_array_gets_clear_domain_error() -> None:
    with pytest.raises(TaskCAggregationError, match="precision values"):
        evaluate_declared_references(
            _complete_scores(),
            pooled_reference={("A", "B")},
            directed_chip_reference=set(),
            eligible_sources={"A"},
            directed_reference_context_match=False,
            precision_values=np.asarray([1, 2]),
        )


def test_aggregation_rejects_symlinked_or_hard_linked_evidence(tmp_path: Path) -> None:
    linked_status = tmp_path / "outside.json"
    _write_json(linked_status, {"not": "a run status"})
    symlink_run = tmp_path / "symlink-run"
    symlink_run.mkdir()
    (symlink_run / "method_status.json").symlink_to(linked_status)
    _write_json(symlink_run / "resource_usage.json", {"schema_version": "1.0"})
    _write_json(
        symlink_run / "environment_manifest.json", {"schema_version": "1.0"}
    )
    with pytest.raises(TaskCAggregationError, match="regular file"):
        aggregate_task_c_runs([symlink_run])

    hardlink_run = _make_run(
        tmp_path / "hardlink-run",
        method_id="pc",
        status="failed_timeout",
    )
    outside_copy = tmp_path / "outside-copy.json"
    os.link(hardlink_run / "resource_usage.json", outside_copy)
    with pytest.raises(TaskCAggregationError, match="hard link"):
        aggregate_task_c_runs([hardlink_run])
