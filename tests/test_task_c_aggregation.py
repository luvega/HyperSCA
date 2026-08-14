from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from src.evaluation import task_c_aggregation as aggregation_module
from src.evaluation import task_c_rehearsal as rehearsal_module
from src.evaluation.task_c_aggregation import (
    TaskCAggregationError,
    aggregate_task_c_runs,
    build_full_run_draft,
    evaluate_declared_references,
    evaluate_rehearsal_readiness,
    paired_cluster_interval,
    task_c_aggregation_to_jsonable,
    thaw_task_c_rehearsal_plan,
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
ROOT = Path(__file__).resolve().parents[1]
REGISTERED_METHODS = {
    "hypersca_c",
    "mean_difference",
    "random1000",
    "grnboost",
    "pc",
    "ges",
    "gies",
    "gsp",
    "igsp",
    "notears_linear",
    "dcdi_g",
    "dcdi_dsf",
    "dcdfg_linear",
    "dcdfg_mlp",
    "sortnregress",
    "guanlab_psgrn",
    "betterboost",
    "sparse_rc",
    "catran",
}
CONDITIONS = (
    "within_k562",
    "within_rpe1",
    "k562_to_rpe1",
    "rpe1_to_k562",
)


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


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _record_sha256(payload: object) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(
    root: Path,
    *,
    method_id: str,
    status: str,
    condition: str = "within_k562",
    seed: int = 11,
    identity_character: str = "a",
    profile: str = "comprehensive",
) -> Path:
    root.mkdir()
    passed = status in {"passed_real_rehearsal", "passed_synthetic_smoke"}
    if passed:
        predictions = (
            b"source,target,score,returned_by_method\n"
            b"A,B,0.2,True\nB,A,0.1,True\n"
        )
        (root / "predictions.csv").write_bytes(predictions)
        prediction_sha256 = "sha256:" + hashlib.sha256(predictions).hexdigest()
        metrics = {
            "average_precision": 0.2,
            "prediction_sha256": prediction_sha256,
        }
        input_summary = {
            "schema_version": "1.0",
            "profile": profile,
            "condition": condition,
            "method_id": method_id,
            "used_stages": ["refit"],
            "training_tuning_and_final_fit_are_separate": False,
            "settings_policy": "registered default settings used for public refit",
            "private_data_received_by_method": False,
            "data_scope": (
                "synthetic_smoke"
                if status == "passed_synthetic_smoke"
                else "external_benchmark"
            ),
        }
        if method_id == "hypersca_c" and profile == "connection":
            input_summary.update(
                {
                    "used_stages": ["train", "tune", "refit"],
                    "training_tuning_and_final_fit_are_separate": True,
                    "settings_policy": (
                        "two public training candidates, separate public tuning, "
                        "selected public refit"
                    ),
                }
            )
        if method_id == "mean_difference":
            input_summary["settings_policy"] = (
                "fixed no-tuning reference; registered settings used for public refit"
            )
        if method_id in {"hypersca_c", "mean_difference"}:
            metrics["null_controls"] = {
                "scope": "formal_zero_effect_reanalysis",
                "formal_null_gate_passed": status == "passed_real_rehearsal",
            }
        promotion = {
            "schema_version": "1.0",
            "status": "workflow_validation_only",
            "claim_level": "workflow_validation_only",
            "promotion_eligible": False,
            "reason": "Single-seed reduced-data rehearsal validates execution and resource readiness only.",
        }
        resource = {
            "schema_version": "1.0",
            "resource_scope": "single-seed reduced-data rehearsal",
            "used_stages": input_summary["used_stages"],
            "nonduplicated_totals": {
                "elapsed_seconds": 10.0,
                "peak_rss_bytes": 1024,
                "peak_gpu_memory_bytes": None,
                "written_disk_bytes": 2048,
                "measurement_availability": {
                    "elapsed_seconds": True,
                    "peak_rss_bytes": True,
                    "peak_gpu_memory_bytes": False,
                    "written_disk_bytes": True,
                },
            },
        }
        environment = {
            "schema_version": "1.0",
            "method_id": method_id,
            "condition": condition,
            "profile": profile,
            "data_scope": input_summary["data_scope"],
            "private_data_received_by_method": False,
        }
        evidence = {
            "input_summary.json": _record_sha256(input_summary),
            "metrics.json": _record_sha256(metrics),
            "predictions.csv": prediction_sha256,
            "promotion_decision.json": _record_sha256(promotion),
            "environment_manifest.json": _record_sha256(environment),
            "resource_usage.json": _record_sha256(resource),
        }
        identity = {
            "schema_version": "1.0",
            "profile": profile,
            "condition": condition,
            "method_id": method_id,
            "seed": seed,
            "input_summary_sha256": _canonical_sha256(input_summary),
            "prediction_sha256": evidence["predictions.csv"],
            "evidence_sha256": evidence,
        }
        run_identity = _canonical_sha256(identity)
        _write_json(
            root / "run_manifest.json",
            {**identity, "run_identity_sha256": run_identity, "claim_level": "workflow_validation_only"},
        )
        _write_json(root / "input_summary.json", input_summary)
        _write_json(root / "metrics.json", metrics)
        _write_json(root / "promotion_decision.json", promotion)
        _write_json(root / "resource_usage.json", resource)
        _write_json(root / "environment_manifest.json", environment)
    else:
        run_identity = "sha256:" + hashlib.sha256(
            f"{method_id}:{condition}:{seed}:{identity_character}".encode("utf-8")
        ).hexdigest()
        _write_json(
            root / "resource_usage.json",
            {
                "schema_version": "1.0",
                "nonduplicated_totals": {
                    "elapsed_seconds": None,
                    "peak_rss_bytes": None,
                    "peak_gpu_memory_bytes": None,
                    "written_disk_bytes": None,
                    "measurement_availability": {
                        "elapsed_seconds": False,
                        "peak_rss_bytes": False,
                        "peak_gpu_memory_bytes": False,
                        "written_disk_bytes": False,
                    },
                },
            },
        )
        _write_json(
            root / "environment_manifest.json",
            {
                "schema_version": "1.0",
                "method_id": method_id,
                "condition": condition,
                "profile": profile,
                "data_scope": "external_benchmark",
                "private_data_received_by_method": False,
            },
        )
    status_payload = {
        "schema_version": "1.0",
        "method_id": method_id,
        "condition": condition,
        "seed": seed,
        "run_identity_sha256": run_identity,
        "status": status,
        "controller_validation": (
            "verified_task_c_synthetic_smoke_bundle_v1"
            if status == "passed_synthetic_smoke"
            else "verified_task_c_rehearsal_bundle_v1"
        ),
    }
    if not passed:
        status_payload["reason"] = status
    _write_json(root / "method_status.json", status_payload)
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


def _formal_method_statuses() -> dict[str, str]:
    statuses = {method: "failed_timeout" for method in REGISTERED_METHODS}
    for method in ("betterboost", "sparse_rc", "catran"):
        statuses[method] = "official_assets_unavailable"
    statuses.update(
        {
            method: "passed_real_rehearsal"
            for method in (
                "hypersca_c",
                "mean_difference",
                "random1000",
                "grnboost",
                "pc",
                "notears_linear",
                "gies",
            )
        }
    )
    return statuses


def test_rehearsal_readiness_requires_core_and_interventional_method() -> None:
    statuses = _formal_method_statuses()
    decision = evaluate_rehearsal_readiness(
        statuses,
        data_checks_passed=True,
        five_splits_reproduced=True,
        null_controls_passed=True,
        tuning_boundary_passed=True,
        project_tests_passed=True,
    )
    assert decision["ready_for_full_run"] is True
    assert decision["claim_level"] == "workflow_validation_only"
    assert decision["authorization_status"] == "not_authorized_to_start"

    statuses["notears_linear"] = "failed_resource_limit"
    assert evaluate_rehearsal_readiness(
        statuses,
        data_checks_passed=True,
        five_splits_reproduced=True,
        null_controls_passed=True,
        tuning_boundary_passed=True,
        project_tests_passed=True,
    )["ready_for_full_run"] is False


def test_rehearsal_readiness_needs_exact_registry_and_real_not_synthetic_passes() -> None:
    statuses = _formal_method_statuses()
    statuses["hypersca_c"] = "passed_synthetic_smoke"
    synthetic = evaluate_rehearsal_readiness(
        statuses,
        data_checks_passed=True,
        five_splits_reproduced=True,
        null_controls_passed=True,
        tuning_boundary_passed=True,
        project_tests_passed=True,
    )
    assert synthetic["ready_for_full_run"] is False
    assert synthetic["checks"]["core_methods"] is False

    missing = dict(statuses)
    missing.pop("catran")
    incomplete = evaluate_rehearsal_readiness(
        missing,
        data_checks_passed=True,
        five_splits_reproduced=True,
        null_controls_passed=True,
        tuning_boundary_passed=True,
        project_tests_passed=True,
    )
    assert incomplete["ready_for_full_run"] is False
    assert incomplete["checks"]["all_methods_classified"] is False


def test_full_run_draft_has_five_seeds_and_never_starts_jobs() -> None:
    draft = build_full_run_draft(
        runnable_methods=["hypersca_c", "mean_difference", "gies"],
        conditions=list(CONDITIONS),
        seeds=[11, 23, 47, 71, 97],
        median_runtime_seconds={
            "hypersca_c": 100.0,
            "mean_difference": 2.0,
            "gies": 20.0,
        },
        maximum_tuning_trials=20,
    )
    assert draft["job_count"] == 1260
    assert draft["tuning_job_count"] == 1200
    assert draft["final_fit_job_count"] == 60
    assert draft["authorization_status"] == "not_authorized_to_start"
    assert draft["runtime_estimate_kind"] == "worst_case_upper_bound"
    assert all("command" not in job for job in draft["jobs"])
    assert isinstance(draft, MappingProxyType)
    assert isinstance(draft["jobs"], tuple)
    assert isinstance(draft["jobs"][0], MappingProxyType)
    with pytest.raises(TypeError):
        draft["authorization_status"] = "authorized"  # type: ignore[index]
    with pytest.raises(TypeError):
        draft["jobs"][0]["phase"] = "run"  # type: ignore[index]

    thawed = thaw_task_c_rehearsal_plan(draft)
    assert isinstance(thawed, dict)
    assert isinstance(thawed["jobs"], list)
    assert json.loads(json.dumps(thawed))["job_count"] == 1260


class _FlipSequence(Sequence[object]):
    def __init__(self, values: Sequence[object]) -> None:
        self._values = tuple(values)
        self.iteration_count = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> object:
        return self._values[index]

    def __iter__(self) -> Iterator[object]:
        self.iteration_count += 1
        if self.iteration_count == 1:
            return iter(self._values)
        return iter(tuple(reversed(self._values)))


class _DuplicateItemsRuntime(Mapping[str, float]):
    def __getitem__(self, key: str) -> float:
        return 1.0

    def __iter__(self) -> Iterator[str]:
        return iter(("hypersca_c",))

    def __len__(self) -> int:
        return 2

    def items(self):  # type: ignore[override]
        return iter((("hypersca_c", 1.0), ("hypersca_c", 2.0)))


class _OnePassRuntime(Mapping[str, float]):
    def __init__(self) -> None:
        self.items_call_count = 0

    def __getitem__(self, key: str) -> float:
        if key != "hypersca_c":
            raise KeyError(key)
        return 10.0

    def __iter__(self) -> Iterator[str]:
        return iter(("hypersca_c",))

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[override]
        self.items_call_count += 1
        if self.items_call_count > 1:
            raise AssertionError("runtime mapping items were read more than once")
        return iter((("hypersca_c", 10.0),))


def test_full_run_draft_copies_each_caller_input_only_once() -> None:
    methods = _FlipSequence(["hypersca_c"])
    conditions = _FlipSequence(CONDITIONS)
    seeds = _FlipSequence([11, 23, 47, 71, 97])
    runtimes = _OnePassRuntime()

    draft = build_full_run_draft(
        runnable_methods=methods,  # type: ignore[arg-type]
        conditions=conditions,  # type: ignore[arg-type]
        seeds=seeds,  # type: ignore[arg-type]
        median_runtime_seconds=runtimes,
        maximum_tuning_trials=20,
    )

    assert draft["job_count"] == 420
    assert methods.iteration_count == 1
    assert conditions.iteration_count == 1
    assert seeds.iteration_count == 1
    assert runtimes.items_call_count == 1


def test_full_run_draft_rejects_mapping_with_duplicate_items() -> None:
    with pytest.raises(TaskCAggregationError, match="runtime"):
        build_full_run_draft(
            runnable_methods=["hypersca_c"],
            conditions=list(CONDITIONS),
            seeds=[11, 23, 47, 71, 97],
            median_runtime_seconds=_DuplicateItemsRuntime(),  # type: ignore[arg-type]
            maximum_tuning_trials=20,
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"seeds": [11]},
        {"conditions": ["within_k562"]},
        {"maximum_tuning_trials": 19},
        {"median_runtime_seconds": {"hypersca_c": float("nan")}},
    ],
)
def test_full_run_draft_rejects_changes_to_frozen_scope(changed: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "runnable_methods": ["hypersca_c"],
        "conditions": list(CONDITIONS),
        "seeds": [11, 23, 47, 71, 97],
        "median_runtime_seconds": {"hypersca_c": 10.0},
        "maximum_tuning_trials": 20,
    }
    arguments.update(changed)
    with pytest.raises(TaskCAggregationError):
        build_full_run_draft(**arguments)  # type: ignore[arg-type]


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "controller_manifest.json"
    }


def _make_rehearsal_root(
    root: Path,
    *,
    statuses: dict[str, str],
    synthetic_smoke: bool = False,
    profile: str = "comprehensive",
) -> tuple[Path, str]:
    runs = root / "runs"
    runs.mkdir(parents=True)
    for condition in CONDITIONS:
        for method_index, (method, status) in enumerate(sorted(statuses.items())):
            _make_run(
                runs / f"{profile}__{condition}__{method}__seed-11",
                method_id=method,
                status=("passed_synthetic_smoke" if synthetic_smoke and status == "passed_real_rehearsal" else status),
                condition=condition,
                identity_character=format(method_index % 16, "x"),
                profile=profile,
            )
    observed_statuses = []
    for path in sorted(runs.iterdir()):
        observed_statuses.append(
            json.loads((path / "method_status.json").read_text(encoding="utf-8"))["status"]
        )
    status_counts = {
        status: observed_statuses.count(status) for status in sorted(set(observed_statuses))
    }
    identity = {
        "schema_version": "1.0",
        "profile": profile,
        "methods": sorted(statuses),
        "conditions": list(CONDITIONS),
        "seed": 11,
        "synthetic_smoke": synthetic_smoke,
        "prepared_identity_sha256": (
            None if synthetic_smoke else "sha256:" + "d" * 64
        ),
        "claim_level": "workflow_validation_only",
        "promotion_eligible": False,
    }
    rebuilt_summary = {
        "schema_version": "1.0",
        "profile": profile,
        "attempted_methods": sorted(statuses),
        "conditions": list(CONDITIONS),
        "attempted_run_count": len(statuses) * len(CONDITIONS),
        "status_counts": status_counts,
        "claim_level": "workflow_validation_only",
        "promotion_eligible": False,
    }
    inventory = _tree_hashes(root)
    resume_token = _canonical_sha256(
        {
            "controller_identity": identity,
            "file_inventory": inventory,
            "rebuilt_summary": rebuilt_summary,
        }
    )
    summary = {
        **rebuilt_summary,
        "resume_status": "new_run",
        "resume_token": resume_token,
    }
    _write_json(
        root / "controller_manifest.json",
        {
            "schema_version": "1.0",
            "identity": identity,
            "identity_sha256": _canonical_sha256(identity),
            "file_inventory": inventory,
            "summary": summary,
            "resume_token": resume_token,
        },
    )
    return root, resume_token


def _run_summary_cli(
    rehearsal_root: Path,
    output: Path,
    resume_token: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_task_c_rehearsal.py"),
            "--rehearsal-root",
            str(rehearsal_root),
            "--output-dir",
            str(output),
            "--resume-token",
            resume_token,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _resign_rehearsal_after_metric_change(root: Path) -> str:
    run = next((root / "runs").iterdir())
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["average_precision"] = 0.91
    _write_json(metrics_path, metrics)

    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_sha256"]["metrics.json"] = _record_sha256(metrics)
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_identity_sha256", "claim_level"}
    }
    changed_run_token = _canonical_sha256(identity)
    manifest["run_identity_sha256"] = changed_run_token
    _write_json(manifest_path, manifest)
    status_path = run / "method_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["run_identity_sha256"] = changed_run_token
    _write_json(status_path, status_payload)

    controller_path = root / "controller_manifest.json"
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["file_inventory"] = _tree_hashes(root)
    rebuilt_summary = {
        key: value
        for key, value in controller["summary"].items()
        if key not in {"resume_status", "resume_token"}
    }
    changed_token = _canonical_sha256(
        {
            "controller_identity": controller["identity"],
            "file_inventory": controller["file_inventory"],
            "rebuilt_summary": rebuilt_summary,
        }
    )
    controller["resume_token"] = changed_token
    controller["summary"]["resume_token"] = changed_token
    _write_json(controller_path, controller)
    return changed_token


@pytest.mark.parametrize("profile", ["connection", "comprehensive"])
def test_summary_accepts_task5_real_profile_specific_stage_records(
    tmp_path: Path, profile: str
) -> None:
    rehearsal, resume_token = _make_rehearsal_root(
        tmp_path / profile,
        statuses={
            "hypersca_c": "passed_real_rehearsal",
            "mean_difference": "passed_real_rehearsal",
        },
        profile=profile,
    )

    completed = _run_summary_cli(
        rehearsal, tmp_path / f"{profile}-summary", resume_token
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (tmp_path / f"{profile}-summary" / "rehearsal_summary.json").read_text()
    )
    assert summary["readiness"]["checks"]["tuning_boundary"] is True


def test_summary_cli_requires_independently_retained_resume_token(
    tmp_path: Path,
) -> None:
    rehearsal, _resume_token = _make_rehearsal_root(
        tmp_path / "rehearsal",
        statuses={"hypersca_c": "passed_real_rehearsal"},
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_task_c_rehearsal.py"),
            "--rehearsal-root",
            str(rehearsal),
            "--output-dir",
            str(tmp_path / "summary"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "resume-token" in completed.stderr
    assert not (tmp_path / "summary").exists()


def test_external_resume_token_rejects_fully_resigned_internal_evidence(
    tmp_path: Path,
) -> None:
    rehearsal, retained_token = _make_rehearsal_root(
        tmp_path / "rehearsal",
        statuses={"hypersca_c": "passed_real_rehearsal"},
    )
    changed_token = _resign_rehearsal_after_metric_change(rehearsal)
    assert changed_token != retained_token

    completed = _run_summary_cli(
        rehearsal, tmp_path / "summary", retained_token
    )

    assert completed.returncode != 0
    assert "external resume token" in completed.stderr
    assert not (tmp_path / "summary").exists()


def test_inspection_limits_total_retained_json_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rehearsal, resume_token = _make_rehearsal_root(
        tmp_path / "rehearsal", statuses=_formal_method_statuses()
    )
    monkeypatch.setattr(
        rehearsal_module, "MAXIMUM_REHEARSAL_SUMMARY_JSON_BYTES", 512
    )

    with pytest.raises(rehearsal_module.TaskCRehearsalError, match="total|budget"):
        rehearsal_module.inspect_task_c_rehearsal_evidence(
            rehearsal, expected_resume_token=resume_token
        )


def test_summary_cli_is_read_only_writes_four_plans_and_keeps_unknown_gates_closed(
    tmp_path: Path,
) -> None:
    rehearsal, resume_token = _make_rehearsal_root(
        tmp_path / "rehearsal", statuses=_formal_method_statuses()
    )
    before = {
        path.relative_to(rehearsal).as_posix(): path.read_bytes()
        for path in sorted(rehearsal.rglob("*"))
        if path.is_file()
    }
    output = tmp_path / "summary"
    completed = _run_summary_cli(rehearsal, output, resume_token)

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert set(stdout) == {"ready_for_full_run", "blocking_items", "output_files"}
    assert stdout["ready_for_full_run"] is False
    assert "five_splits_reproduced" in stdout["blocking_items"]
    assert "project_tests" in stdout["blocking_items"]
    assert set(path.name for path in output.iterdir()) == {
        "rehearsal_summary.json",
        "method_compatibility.csv",
        "resource_estimate.json",
        "full_run_jobs_draft.json",
    }
    after = {
        path.relative_to(rehearsal).as_posix(): path.read_bytes()
        for path in sorted(rehearsal.rglob("*"))
        if path.is_file()
    }
    assert after == before

    summary = json.loads((output / "rehearsal_summary.json").read_text())
    assert summary["readiness"]["checks"]["five_splits_reproduced"] is False
    assert summary["readiness"]["checks"]["project_tests"] is False
    assert summary["readiness"]["authorization_status"] == "not_authorized_to_start"
    assert len(summary["completed_run_metrics"]) == 7 * len(CONDITIONS)
    assert {
        record["method_id"] for record in summary["completed_run_metrics"]
    } == {
        "hypersca_c",
        "mean_difference",
        "random1000",
        "grnboost",
        "pc",
        "notears_linear",
        "gies",
    }
    compatibility = pd.read_csv(output / "method_compatibility.csv")
    assert set(compatibility["method_id"]) == REGISTERED_METHODS
    assert compatibility.loc[
        compatibility["method_id"] == "hypersca_c", "overall_status"
    ].item() == "passed_real_rehearsal"
    resources = json.loads((output / "resource_estimate.json").read_text())
    assert resources["methods"]["hypersca_c"]["median_elapsed_seconds"] == 10.0
    assert resources["methods"]["hypersca_c"]["peak_gpu_memory_bytes"] is None
    assert resources["methods"]["hypersca_c"]["measurement_availability"][
        "peak_gpu_memory_bytes"
    ] is False
    draft = json.loads((output / "full_run_jobs_draft.json").read_text())
    assert draft["authorization_status"] == "not_authorized_to_start"
    assert draft["job_count"] == 2940

    second = _run_summary_cli(rehearsal, output, resume_token)
    assert second.returncode != 0
    assert "不会覆盖" in second.stderr


def test_summary_cli_never_treats_synthetic_smoke_as_real_readiness(
    tmp_path: Path,
) -> None:
    rehearsal, resume_token = _make_rehearsal_root(
        tmp_path / "synthetic",
        statuses=_formal_method_statuses(),
        synthetic_smoke=True,
    )
    output = tmp_path / "summary"
    completed = _run_summary_cli(rehearsal, output, resume_token)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "rehearsal_summary.json").read_text())
    assert summary["readiness"]["ready_for_full_run"] is False
    assert summary["readiness"]["checks"]["core_methods"] is False
    assert json.loads((output / "full_run_jobs_draft.json").read_text())[
        "job_count"
    ] == 0


@pytest.mark.parametrize("hazard", ["symlink", "hardlink", "tamper", "nonfinite", "huge"])
def test_summary_cli_rejects_unsafe_or_changed_rehearsal_evidence(
    tmp_path: Path, hazard: str
) -> None:
    rehearsal, resume_token = _make_rehearsal_root(
        tmp_path / "rehearsal", statuses={"hypersca_c": "passed_real_rehearsal"}
    )
    run = next((rehearsal / "runs").iterdir())
    if hazard == "symlink":
        path = run / "metrics.json"
        saved = tmp_path / "saved.json"
        path.rename(saved)
        path.symlink_to(saved)
    elif hazard == "hardlink":
        os.link(run / "resource_usage.json", tmp_path / "linked.json")
    elif hazard == "tamper":
        (run / "metrics.json").write_text(
            '{"average_precision":0.9}\n', encoding="utf-8"
        )
    elif hazard == "nonfinite":
        path = run / "resource_usage.json"
        path.write_text('{"nonduplicated_totals":{"elapsed_seconds":NaN}}\n')
        controller = json.loads(
            (rehearsal / "controller_manifest.json").read_text(encoding="utf-8")
        )
        controller["file_inventory"] = _tree_hashes(rehearsal)
        _write_json(rehearsal / "controller_manifest.json", controller)
    else:
        path = run / "metrics.json"
        path.write_bytes(
            b'{"average_precision":0.2,"padding":"'
            + b"x" * (16 * 1024 * 1024)
            + b'"}\n'
        )
        controller = json.loads(
            (rehearsal / "controller_manifest.json").read_text(encoding="utf-8")
        )
        controller["file_inventory"] = _tree_hashes(rehearsal)
        _write_json(rehearsal / "controller_manifest.json", controller)

    completed = _run_summary_cli(rehearsal, tmp_path / "summary", resume_token)
    assert completed.returncode != 0
    assert not (tmp_path / "summary").exists()


def test_atomic_directory_publication_never_replaces_an_existing_empty_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "summary"
    staging.mkdir()
    output.mkdir()
    (staging / "rehearsal_summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(TaskCAggregationError, match="already exists"):
        aggregation_module._atomic_publish_directory(staging, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert (staging / "rehearsal_summary.json").is_file()


def test_summary_fsyncs_staging_directory_before_exclusive_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fsynced = False
    real_fsync = aggregation_module.os.fsync
    real_publish = aggregation_module._atomic_publish_directory

    def observed_fsync(descriptor: int) -> None:
        nonlocal directory_fsynced
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsynced = True
        real_fsync(descriptor)

    def checked_publish(staging: Path, output: Path) -> None:
        assert directory_fsynced is True
        real_publish(staging, output)

    monkeypatch.setattr(aggregation_module.os, "fsync", observed_fsync)
    monkeypatch.setattr(
        aggregation_module, "_atomic_publish_directory", checked_publish
    )

    paths = aggregation_module._publish_summary_files(
        tmp_path / "summary", {"rehearsal_summary.json": b"{}\n"}
    )

    assert Path(paths["rehearsal_summary.json"]).is_file()
