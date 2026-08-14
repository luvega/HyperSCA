"""统一汇总任务 C 的评分、区间和真实预演结果。

本模块只比较每个方法都必须返回的完整基因关系表。参考关系是不完整的
生物学证据，而不是完整的因果真值；有向 ChIP 关系也只在细胞环境匹配时
用于补充方向检查。
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from src.evaluation.task_c_benchmark import (
    TaskCBenchmarkError,
    evaluate_task_c_scores,
)


class TaskCAggregationError(ValueError):
    """运行结果不完整，因而不能进行公平汇总。"""


_SCORE_COLUMN_SCHEMAS = frozenset(
    {
        ("source", "target", "score"),
        ("source", "target", "score", "effect"),
        ("source", "target", "score", "returned_by_method"),
        ("source", "target", "score", "effect", "returned_by_method"),
    }
)
MAXIMUM_BOOTSTRAP_REPEATS = 1_000_000
MAXIMUM_RANDOM_SEED = 2**32 - 1
MAXIMUM_STATUS_BYTES = 64 * 1024
MAXIMUM_METRICS_BYTES = 4 * 1024 * 1024
MAXIMUM_EVIDENCE_BYTES = 64 * 1024 * 1024
MAXIMUM_JSON_DEPTH = 32
_PASSED_STATUS = "passed_real_rehearsal"
_SYNTHETIC_STATUS = "passed_synthetic_smoke"
_FAILED_OR_UNAVAILABLE_STATUSES = frozenset(
    {
        "failed_timeout",
        "failed_resource_limit",
        "failed_runtime_unavailable",
        "failed_launch",
        "failed_invalid_output",
        "failed_private_scoring",
        "failed_null_control",
        "official_code_incompatible",
        "official_assets_unavailable",
    }
)
_PASSED_FILES = frozenset(
    {
        "method_status.json",
        "metrics.json",
        "predictions.csv",
        "run_manifest.json",
        "input_summary.json",
        "promotion_decision.json",
        "resource_usage.json",
        "environment_manifest.json",
    }
)
_FAILED_FILES = frozenset(
    {
        "method_status.json",
        "resource_usage.json",
        "environment_manifest.json",
    }
)
_MINIMAL_STATUS_FIELDS = frozenset({"method_id", "status"})
_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "method_id",
        "condition",
        "seed",
        "run_identity_sha256",
        "status",
        "controller_validation",
    }
)
_KNOWN_PASSED_FILES = _PASSED_FILES
_KNOWN_FAILED_FILES = _FAILED_FILES | frozenset(
    {
        "run_manifest.json",
        "input_summary.json",
        "promotion_decision.json",
    }
)
_REHEARSAL_CONDITIONS = frozenset(
    {"within_k562", "within_rpe1", "k562_to_rpe1", "rpe1_to_k562"}
)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def task_c_aggregation_to_jsonable(value: Any) -> Any:
    """将只读汇总结果复制为可安全交给 ``json.dumps`` 的普通容器。"""

    if isinstance(value, Mapping):
        return {
            str(key): task_c_aggregation_to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [task_c_aggregation_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _valid_name(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise TaskCAggregationError(f"{label} must contain non-empty text")
    if unicodedata.normalize("NFC", value) != value:
        raise TaskCAggregationError(f"{label} must use NFC text")
    return value


def _validate_scores(scores: pd.DataFrame) -> tuple[set[str], set[tuple[str, str]]]:
    if type(scores) is not pd.DataFrame:
        raise TaskCAggregationError("scores must be a pandas data table")
    columns = tuple(scores.columns)
    if columns not in _SCORE_COLUMN_SCHEMAS:
        raise TaskCAggregationError(
            "scores must use the fixed source, target, score column policy"
        )
    if scores.empty:
        raise TaskCAggregationError("scores must contain the complete relation table")

    relations: set[tuple[str, str]] = set()
    genes: set[str] = set()
    for source_raw, target_raw in scores[["source", "target"]].itertuples(
        index=False, name=None
    ):
        source = _valid_name(source_raw, "score source names")
        target = _valid_name(target_raw, "score target names")
        if source == target:
            raise TaskCAggregationError("scores must not contain self relations")
        relation = (source, target)
        if relation in relations:
            raise TaskCAggregationError("scores must contain unique directed relations")
        relations.add(relation)
        genes.update(relation)

    if len(genes) < 2:
        raise TaskCAggregationError("scores must contain at least two genes")
    expected = {
        (source, target)
        for source in genes
        for target in genes
        if source != target
    }
    if relations != expected:
        raise TaskCAggregationError(
            "scores must contain every directed non-self relation for one gene set"
        )

    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in scores["score"]
    ):
        raise TaskCAggregationError("scores must contain numeric values")
    numeric_scores = scores["score"].to_numpy(dtype=float)
    if not np.isfinite(numeric_scores).all() or (numeric_scores < 0).any():
        raise TaskCAggregationError("scores must be finite and non-negative")
    if "effect" in scores:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in scores["effect"]
        ):
            raise TaskCAggregationError("effects must contain numeric values")
        effects = scores["effect"].to_numpy(dtype=float)
        if not np.isfinite(effects).all():
            raise TaskCAggregationError("effects must be finite when supplied")
    if "returned_by_method" in scores:
        if any(type(value) not in {bool, np.bool_} for value in scores["returned_by_method"]):
            raise TaskCAggregationError(
                "returned_by_method must contain only true or false values"
            )
        returned = scores["returned_by_method"].to_numpy(dtype=bool)
        unreturned_scores = numeric_scores[~returned]
        if np.any(unreturned_scores != 0.0) or np.any(np.signbit(unreturned_scores)):
            raise TaskCAggregationError(
                "unreturned relations must use positive zero scores"
            )
    return genes, relations


def _normalize_relations(
    values: Iterable[tuple[str, str]],
    *,
    label: str,
) -> set[tuple[str, str]]:
    if isinstance(values, (str, bytes)):
        raise TaskCAggregationError(f"{label} must contain source-target pairs")
    normalized: set[tuple[str, str]] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TaskCAggregationError(
            f"{label} must contain source-target pairs"
        ) from exc
    for raw_edge in iterator:
        if type(raw_edge) not in {tuple, list} or len(raw_edge) != 2:
            raise TaskCAggregationError(f"{label} must contain source-target pairs")
        source = _valid_name(raw_edge[0], f"{label} source names")
        target = _valid_name(raw_edge[1], f"{label} target names")
        if source == target:
            raise TaskCAggregationError(f"{label} must not contain self relations")
        edge = (source, target)
        if edge in normalized:
            raise TaskCAggregationError(f"{label} must contain unique relations")
        normalized.add(edge)
    return normalized


def _normalize_eligible_sources(
    values: Iterable[str], genes: set[str]
) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise TaskCAggregationError("eligible sources must be a non-empty gene set")
    allowed: set[str] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TaskCAggregationError(
            "eligible sources must be a non-empty gene set"
        ) from exc
    for raw_source in iterator:
        source = _valid_name(raw_source, "eligible source names")
        if source in allowed:
            raise TaskCAggregationError("eligible sources must be unique")
        if source not in genes:
            raise TaskCAggregationError(
                "eligible sources must be a subset of the scored genes"
            )
        allowed.add(source)
    if not allowed:
        raise TaskCAggregationError("eligible sources must be a non-empty gene set")
    return allowed


def _stable_precision_at_k(
    scores: pd.DataFrame,
    positives: set[tuple[str, str]],
    requested_k: int,
) -> tuple[float, int]:
    effective_k = min(requested_k, len(scores))
    ordered = scores.sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected = ordered.head(effective_k)
    selected_positive = sum(
        edge in positives
        for edge in selected[["source", "target"]].itertuples(index=False, name=None)
    )
    return float(selected_positive / effective_k), int(effective_k)


def _tie_aware_precision_at_k(
    scores: pd.DataFrame,
    positives: set[tuple[str, str]],
    requested_k: int,
) -> float:
    effective_k = min(requested_k, len(scores))
    score_values = scores["score"].to_numpy(dtype=float)
    cutoff_index = len(score_values) - effective_k
    cutoff = float(np.partition(score_values, cutoff_index)[cutoff_index])
    above = scores[scores["score"] > cutoff]
    tied = scores[scores["score"] == cutoff]
    above_positive = sum(
        edge in positives
        for edge in above[["source", "target"]].itertuples(index=False, name=None)
    )
    tied_positive = sum(
        edge in positives
        for edge in tied[["source", "target"]].itertuples(index=False, name=None)
    )
    remaining = effective_k - len(above)
    expected_positive = above_positive + remaining * tied_positive / len(tied)
    return float(expected_positive / effective_k)


def evaluate_declared_references(
    scores: pd.DataFrame,
    *,
    pooled_reference: Iterable[tuple[str, str]],
    directed_chip_reference: Iterable[tuple[str, str]],
    eligible_sources: Iterable[str],
    directed_reference_context_match: bool,
    precision_values: Sequence[int] = (1000, 5000),
) -> Mapping[str, object]:
    """分别报告汇总生物参考关系和环境匹配的有向补充关系。"""

    genes, scored_relations = _validate_scores(scores)
    pooled = _normalize_relations(pooled_reference, label="pooled reference")
    directed = _normalize_relations(
        directed_chip_reference, label="directed reference"
    )
    allowed_sources = _normalize_eligible_sources(eligible_sources, genes)
    if type(directed_reference_context_match) is not bool:
        raise TaskCAggregationError(
            "directed reference context match must be true or false"
        )
    if isinstance(precision_values, (str, bytes)):
        raise TaskCAggregationError("precision values must be non-empty")
    try:
        precision_count = len(precision_values)
    except (TypeError, OverflowError) as exc:
        raise TaskCAggregationError(
            "precision values must be a finite non-empty sequence"
        ) from exc
    if precision_count < 1:
        raise TaskCAggregationError("precision values must be non-empty")
    checked_k: list[int] = []
    try:
        for value in precision_values:
            if type(value) is not int or value < 1:
                raise TaskCAggregationError(
                    "precision values must be positive whole numbers"
                )
            if value in checked_k:
                raise TaskCAggregationError("precision values must be unique")
            checked_k.append(value)
    except TaskCAggregationError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaskCAggregationError(
            "precision values must be positive whole numbers"
        ) from exc

    scored = scores[scores["source"].isin(allowed_sources)].copy()
    if scored.empty:
        raise TaskCAggregationError(
            "holdout scoring has no eligible source relations"
        )
    pooled_in_scope = {
        edge
        for edge in pooled
        if edge[0] in allowed_sources and edge in scored_relations
    }
    if not pooled_in_scope:
        raise TaskCAggregationError(
            "pooled reference has no relation from an eligible source"
        )
    try:
        metrics = evaluate_task_c_scores(
            scored, pooled_in_scope, precision_at_k=checked_k[0]
        )
    except TaskCBenchmarkError as exc:
        raise TaskCAggregationError(str(exc)) from exc

    for index, requested_k in enumerate(checked_k):
        precision, effective_k = _stable_precision_at_k(
            scored, pooled_in_scope, requested_k
        )
        metrics[f"precision_at_{requested_k}"] = precision
        metrics[f"precision_at_{requested_k}_tie_aware_sensitivity"] = (
            _tie_aware_precision_at_k(scored, pooled_in_scope, requested_k)
        )
        if index == 0:
            metrics["precision_at_k"] = precision
            metrics["precision_k"] = effective_k
    metrics["precision_tie_policy"] = (
        "score descending, then source and target alphabetical order"
    )
    metrics["precision_metric_role"] = "sensitivity_analysis"
    metrics["primary_ranking_metric"] = "average_precision"
    metrics["precision_tie_limitation"] = (
        "ordered precision can depend on gene names at a cutoff tie; the "
        "tie-aware sensitivity reports expected credit within that tie"
    )

    score_map = {
        (source, target): float(score)
        for source, target, score in scores[["source", "target", "score"]].itertuples(
            index=False, name=None
        )
    }
    directed_in_scope = sorted(
        edge
        for edge in directed
        if edge[0] in allowed_sources
        and edge in scored_relations
        and (edge[1], edge[0]) in scored_relations
    )
    comparisons: list[float] = []
    if directed_reference_context_match:
        for source, target in directed_in_scope:
            forward = score_map[(source, target)]
            reverse = score_map[(target, source)]
            comparisons.append(float(forward > reverse))
    direction_accuracy = (
        float(np.mean(comparisons)) if comparisons else None
    )
    metrics.update(
        {
            "primary_reference_id": "causalbench_pooled_biological_v1",
            "primary_reference_scope": (
                "the supplied directed expansion of pooled biological evidence; "
                "incomplete reference, not causal ground truth"
            ),
            "directed_reference_id": "causalbench_chipseq_v1",
            "directed_chip_edge_count": int(len(directed_in_scope)),
            "directed_reference_context_match": directed_reference_context_match,
            "edge_direction_accuracy": direction_accuracy,
            "direction_tie_credit": 0.0,
            "eligible_source_count": int(len(allowed_sources)),
            "scored_edge_count": int(len(scored)),
            "complete_scored_edge_count": int(len(scores)),
        }
    )
    if any(
        isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise TaskCAggregationError("derived metrics must be finite")
    return _deep_freeze(metrics)


def paired_cluster_interval(
    table: pd.DataFrame,
    *,
    candidate_column: str = "candidate",
    baseline_column: str = "baseline",
    repeats: int = 10000,
    seed: int = 11,
) -> Mapping[str, float | int]:
    """按完整的“随机种子 × 生物条件”配对单位估计差值区间。

    候选方法或对照方法缺失的行不进入差值估计，但仍保留在尝试次数中，
    以免失败运行从分母中消失。这里的区间是对已完成配对的重复抽样区间，
    不能代替生物重复所提供的不确定性。
    """

    if type(table) is not pd.DataFrame:
        raise TaskCAggregationError("paired table must be a pandas data table")
    if table.columns.has_duplicates:
        raise TaskCAggregationError("paired table column names must be unique")
    candidate_column = _valid_name(candidate_column, "candidate column")
    baseline_column = _valid_name(baseline_column, "baseline column")
    if candidate_column == baseline_column:
        raise TaskCAggregationError("candidate and baseline columns must differ")
    required = {"seed", "condition", candidate_column, baseline_column}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise TaskCAggregationError(
            "paired table is missing seed, condition, or metric columns"
        )
    if type(repeats) is not int or repeats < 1:
        raise TaskCAggregationError("bootstrap repeats must be a positive whole number")
    if repeats > MAXIMUM_BOOTSTRAP_REPEATS:
        raise TaskCAggregationError("bootstrap repeats exceed the fixed upper limit")
    if type(seed) is not int or not 0 <= seed <= MAXIMUM_RANDOM_SEED:
        raise TaskCAggregationError("bootstrap seed must be a bounded whole number")
    if table.empty:
        raise TaskCAggregationError("paired interval needs attempted matched runs")

    cluster_ids: set[tuple[int, str]] = set()
    differences: list[float] = []
    for raw_seed, raw_condition, candidate, baseline in table[
        ["seed", "condition", candidate_column, baseline_column]
    ].itertuples(index=False, name=None):
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, (int, np.integer)):
            raise TaskCAggregationError("pair seed must be a whole number")
        pair_seed = int(raw_seed)
        if not 0 <= pair_seed <= MAXIMUM_RANDOM_SEED:
            raise TaskCAggregationError("pair seed must be a bounded whole number")
        condition = _valid_name(raw_condition, "pair condition")
        cluster = (pair_seed, condition)
        if cluster in cluster_ids:
            raise TaskCAggregationError(
                "paired table must contain unique seed-condition clusters"
            )
        cluster_ids.add(cluster)

        candidate_missing = candidate is None or candidate is pd.NA or (
            isinstance(candidate, (float, np.floating))
            and math.isnan(float(candidate))
        )
        baseline_missing = baseline is None or baseline is pd.NA or (
            isinstance(baseline, (float, np.floating))
            and math.isnan(float(baseline))
        )
        if candidate_missing or baseline_missing:
            continue
        if (
            isinstance(candidate, (bool, np.bool_))
            or isinstance(baseline, (bool, np.bool_))
            or not isinstance(candidate, (int, float, np.integer, np.floating))
            or not isinstance(baseline, (int, float, np.integer, np.floating))
        ):
            raise TaskCAggregationError("paired metrics must be finite numbers")
        candidate_value = float(candidate)
        baseline_value = float(baseline)
        difference = candidate_value - baseline_value
        if not (
            math.isfinite(candidate_value)
            and math.isfinite(baseline_value)
            and math.isfinite(difference)
        ):
            raise TaskCAggregationError("paired metrics and differences must be finite")
        differences.append(difference)

    if not differences:
        raise TaskCAggregationError("paired interval needs completed matched runs")
    difference_array = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(
            0, len(difference_array), size=len(difference_array)
        )
        estimates[repeat] = float(np.mean(difference_array[indices]))
    result = {
        "estimate": float(np.mean(difference_array)),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
        "cluster_count": int(len(difference_array)),
        "attempted_cluster_count": int(len(cluster_ids)),
        "dropped_cluster_count": int(len(cluster_ids) - len(difference_array)),
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
    }
    if not all(
        math.isfinite(float(result[name]))
        for name in ("estimate", "ci_lower", "ci_upper")
    ):
        raise TaskCAggregationError("paired interval produced non-finite results")
    return _deep_freeze(result)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_nlink),
    )


def _capture_regular_file(path: Path, *, maximum_bytes: int) -> tuple[bytes, tuple[int, int]]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TaskCAggregationError(f"evidence is not a regular file: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise TaskCAggregationError(f"evidence is not a regular file: {path}")
    if before.st_nlink != 1:
        raise TaskCAggregationError(f"evidence file must not be a hard link: {path}")
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise TaskCAggregationError(f"evidence file has an unsafe size: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise TaskCAggregationError(f"evidence is not a regular file: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise TaskCAggregationError(f"evidence file changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = path.lstat()
    except OSError as exc:
        raise TaskCAggregationError(f"evidence file changed while reading: {path}") from exc
    if (
        _file_identity(after) != _file_identity(before)
        or _file_identity(named_after) != _file_identity(before)
    ):
        raise TaskCAggregationError(f"evidence file changed while reading: {path}")
    if len(payload) < 1 or len(payload) > maximum_bytes:
        raise TaskCAggregationError(f"evidence file has an unsafe size: {path}")
    return payload, (int(before.st_dev), int(before.st_ino))


def _reject_json_constant(value: str) -> None:
    raise TaskCAggregationError(f"JSON contains a non-finite number: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskCAggregationError(f"JSON contains a duplicate field: {key}")
        result[key] = value
    return result


def _validate_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > MAXIMUM_JSON_DEPTH:
        raise TaskCAggregationError("JSON evidence is too deeply nested")
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise TaskCAggregationError("JSON object names must be text")
            _validate_json_tree(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise TaskCAggregationError("JSON contains a non-finite number")


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskCAggregationError(f"{label} must use UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except TaskCAggregationError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TaskCAggregationError(f"{label} must contain valid JSON") from exc
    _validate_json_tree(value)
    if type(value) is not dict:
        raise TaskCAggregationError(f"{label} must contain one JSON object")
    return value


def _validate_method_status(status_payload: dict[str, Any]) -> bool:
    fields = frozenset(status_payload)
    extended_fields = {_STATUS_FIELDS, _STATUS_FIELDS | {"reason"}}
    if fields != _MINIMAL_STATUS_FIELDS and fields not in extended_fields:
        raise TaskCAggregationError("method status fields changed")
    method_id = _valid_name(status_payload.get("method_id"), "method identity")
    if (
        len(method_id) > 80
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in method_id)
    ):
        raise TaskCAggregationError("method identity contains unsafe text")
    status = status_payload.get("status")
    if status not in {
        _PASSED_STATUS,
        _SYNTHETIC_STATUS,
        *_FAILED_OR_UNAVAILABLE_STATUSES,
    }:
        raise TaskCAggregationError("method status is not a final rehearsal status")
    if fields == _MINIMAL_STATUS_FIELDS:
        return False

    if status_payload.get("schema_version") != "1.0":
        raise TaskCAggregationError("method status schema changed")
    condition = status_payload.get("condition")
    if condition not in _REHEARSAL_CONDITIONS:
        raise TaskCAggregationError("method status condition is not recognized")
    seed = status_payload.get("seed")
    if type(seed) is not int or not 0 <= seed <= MAXIMUM_RANDOM_SEED:
        raise TaskCAggregationError("method status seed must be a bounded whole number")
    identity = status_payload.get("run_identity_sha256")
    if (
        type(identity) is not str
        or not identity.startswith("sha256:")
        or len(identity) != 71
        or any(character not in "0123456789abcdef" for character in identity[7:])
    ):
        raise TaskCAggregationError("method status run identity is invalid")
    expected_controller = (
        "verified_task_c_synthetic_smoke_bundle_v1"
        if status == _SYNTHETIC_STATUS
        else "verified_task_c_rehearsal_bundle_v1"
    )
    if status_payload.get("controller_validation") != expected_controller:
        raise TaskCAggregationError(
            "method status lacks the rehearsal controller validation declaration"
        )
    reason = status_payload.get("reason")
    if status in {_PASSED_STATUS, _SYNTHETIC_STATUS} and "reason" in status_payload:
        raise TaskCAggregationError("passed method status must not contain a failure reason")
    if "reason" in status_payload:
        _valid_name(reason, "failure reason")
    return True


def _validate_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        raise TaskCAggregationError("completed run metrics must not be empty")
    average_precision = metrics.get("average_precision")
    try:
        finite_average_precision = math.isfinite(float(average_precision))
        in_range = 0.0 <= float(average_precision) <= 1.0
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaskCAggregationError(
            "completed run average_precision must be finite and between zero and one"
        ) from exc
    if (
        isinstance(average_precision, bool)
        or not isinstance(average_precision, (int, float))
        or not finite_average_precision
        or not in_range
    ):
        raise TaskCAggregationError(
            "completed run average_precision must be finite and between zero and one"
        )


def _run_directory_identity(path: Path) -> tuple[int, int, tuple[int, int, int, int, int, int]]:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise TaskCAggregationError(
                f"run directory is unavailable: {absolute}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCAggregationError(
                f"run directory must not use symbolic links: {absolute}"
            )
    metadata = absolute.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise TaskCAggregationError(f"run directory is not a directory: {absolute}")
    return int(metadata.st_dev), int(metadata.st_ino), _file_identity(metadata)


def aggregate_task_c_runs(
    run_directories: Sequence[str | Path],
    *,
    allow_legacy_minimal: bool = False,
) -> Mapping[str, object]:
    """保留成功和失败尝试，并核对固定的预演结果文件集合。

    本函数会检查文件结构和预演控制器写下的验证声明，但不会把一个可重新
    签写的声明当作独立安全证明。后续预演控制器仍须在写入最终状态前核验
    每个方法的原始结果包和运行身份。
    """

    if type(allow_legacy_minimal) is not bool:
        raise TaskCAggregationError("allow_legacy_minimal must be true or false")
    if isinstance(run_directories, (str, bytes)) or not isinstance(
        run_directories, Sequence
    ):
        raise TaskCAggregationError("run directories must be a non-empty sequence")
    if not run_directories:
        raise TaskCAggregationError("run directories must be a non-empty sequence")

    seen_paths: set[str] = set()
    seen_directory_inodes: set[tuple[int, int]] = set()
    seen_file_inodes: set[tuple[int, int]] = set()
    seen_run_identities: set[str] = set()
    seen_method_clusters: set[tuple[str, str, int]] = set()
    run_records: list[dict[str, Any]] = []
    statuses: list[str] = []
    verified_completed = 0
    legacy_structural_completed = 0

    for raw_directory in run_directories:
        if not isinstance(raw_directory, (str, Path)):
            raise TaskCAggregationError("run directory paths must be text or paths")
        run_dir = Path(os.path.abspath(os.fspath(Path(raw_directory).expanduser())))
        run_path = os.fspath(run_dir)
        if run_path in seen_paths:
            raise TaskCAggregationError("duplicate run directory was supplied")
        seen_paths.add(run_path)
        device, inode, directory_before = _run_directory_identity(run_dir)
        directory_inode = (device, inode)
        if directory_inode in seen_directory_inodes:
            raise TaskCAggregationError("duplicate run directory inode was supplied")
        seen_directory_inodes.add(directory_inode)

        status_bytes, status_inode = _capture_regular_file(
            run_dir / "method_status.json", maximum_bytes=MAXIMUM_STATUS_BYTES
        )
        if status_inode in seen_file_inodes:
            raise TaskCAggregationError("run evidence reuses a file inode")
        seen_file_inodes.add(status_inode)
        method_status = _parse_json_object(status_bytes, "method status")
        extended_status = _validate_method_status(method_status)
        if not extended_status and not allow_legacy_minimal:
            raise TaskCAggregationError(
                "legacy minimal method status requires explicit legacy mode"
            )

        status_name = str(method_status["status"])
        if status_name in {_PASSED_STATUS, _SYNTHETIC_STATUS}:
            required_files = frozenset({"method_status.json", "metrics.json"})
            allowed_files = _KNOWN_PASSED_FILES
            if extended_status:
                required_files = _PASSED_FILES
        else:
            required_files = frozenset({"method_status.json"})
            allowed_files = _KNOWN_FAILED_FILES
        try:
            names = {entry.name for entry in os.scandir(run_dir)}
        except OSError as exc:
            raise TaskCAggregationError("run directory cannot be inspected") from exc
        if not required_files <= names or not names <= allowed_files:
            raise TaskCAggregationError(
                f"run file set changed for status {status_name}"
            )

        captured_json: dict[str, dict[str, Any]] = {"method_status.json": method_status}
        for name in sorted(names - {"method_status.json"}):
            maximum = MAXIMUM_METRICS_BYTES if name == "metrics.json" else MAXIMUM_EVIDENCE_BYTES
            payload, file_inode = _capture_regular_file(
                run_dir / name, maximum_bytes=maximum
            )
            if file_inode in seen_file_inodes:
                raise TaskCAggregationError("run evidence reuses a file inode")
            seen_file_inodes.add(file_inode)
            if name.endswith(".json"):
                captured_json[name] = _parse_json_object(payload, name)

        _, _, directory_after = _run_directory_identity(run_dir)
        if directory_after != directory_before:
            raise TaskCAggregationError("run directory changed during aggregation")
        if {entry.name for entry in os.scandir(run_dir)} != names:
            raise TaskCAggregationError("run directory changed during aggregation")

        if extended_status:
            run_identity = str(method_status["run_identity_sha256"])
            if run_identity in seen_run_identities:
                raise TaskCAggregationError("duplicate run identity was supplied")
            seen_run_identities.add(run_identity)
            cluster = (
                str(method_status["method_id"]),
                str(method_status["condition"]),
                int(method_status["seed"]),
            )
            if cluster in seen_method_clusters:
                raise TaskCAggregationError(
                    "duplicate method-condition-seed attempt was supplied"
                )
            seen_method_clusters.add(cluster)

        record = dict(method_status)
        record["controller_validation_status"] = (
            str(method_status["controller_validation"])
            if extended_status
            else "unverified_legacy_record"
        )
        if status_name in {_PASSED_STATUS, _SYNTHETIC_STATUS}:
            metrics = captured_json["metrics.json"]
            _validate_metrics(metrics)
            record["metrics"] = metrics
            if extended_status and status_name == _PASSED_STATUS:
                verified_completed += 1
            elif not extended_status:
                legacy_structural_completed += 1
        run_records.append(record)
        statuses.append(status_name)

    counts = Counter(statuses)
    synthetic_structural = counts.get(_SYNTHETIC_STATUS, 0)
    structural_completed = counts.get(_PASSED_STATUS, 0) + synthetic_structural
    result = {
        "attempted_run_count": len(run_records),
        "completed_run_count": verified_completed,
        "verified_completed_run_count": verified_completed,
        "legacy_structural_completed_count": legacy_structural_completed,
        "synthetic_structural_run_count": synthetic_structural,
        "structural_completed_run_count": structural_completed,
        "not_formally_completed_count": len(run_records) - verified_completed,
        "failed_or_unavailable_count": len(run_records) - structural_completed,
        "status_counts": dict(sorted(counts.items())),
        "runs": run_records,
        "validation_scope": (
            "formal completion requires the verified Task C rehearsal bundle "
            "declaration; synthetic smoke and explicit legacy mode are structural only"
        ),
        "independent_bundle_verification": False,
    }
    return _deep_freeze(result)
