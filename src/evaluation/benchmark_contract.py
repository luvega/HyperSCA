"""Executable benchmark preregistration for HyperSCA Tasks C, S, and D.

The contract freezes comparison conditions before model evaluation.  Promotion
decisions are deliberately conservative and support only benchmark-backed
candidate claims, never clinical or universal superiority claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXPECTED_TASKS = ("C", "S", "D")
EXPECTED_CONTRACT_ID = "hypersca-csd-benchmark-v1"
EXPECTED_RANDOM_SEEDS = (11, 23, 47, 71, 97)
EXPECTED_MAX_TRIALS = 20
EXPECTED_SIMPLE_BASELINES = {
    "C": ("mean_difference",),
    "S": ("own_only", "fixed_distance_decay"),
    "D": ("non_spatial_signature_reversal",),
}
EXPECTED_PRIMARY_METRICS = {
    "C": ("average_precision", "maximize"),
    "S": ("neighbor_effect_rmse", "minimize"),
    "D": ("spearman_response_correlation", "maximize"),
}
SUPPORTED_METHOD_ROLES = {
    "candidate",
    "simple_baseline",
    "external_comparator",
    "null_control",
}


class BenchmarkContractError(ValueError):
    """Raised when a contract or its bound evidence is invalid."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def contract_digest(contract: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical contract representation."""
    payload = _canonical_json(contract).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{context}.{key} must be an object")
    return value


def _require_nonempty_string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{context}.{key} must be a non-empty string")
    return value


def _require_nonempty_string_list(
    parent: Mapping[str, Any],
    key: str,
    context: str,
) -> list[str]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise BenchmarkContractError(
            f"{context}.{key} must be a non-empty list of strings"
        )
    if len(value) != len(set(value)):
        raise BenchmarkContractError(f"{context}.{key} must not contain duplicates")
    return value


def _require_probability(parent: Mapping[str, Any], key: str, context: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkContractError(f"{context}.{key} must be a number in [0, 1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise BenchmarkContractError(f"{context}.{key} must be a number in [0, 1]")
    return numeric


def validate_benchmark_contract(contract: Mapping[str, Any]) -> None:
    """Validate the frozen C/S/D contract, failing closed on omissions."""
    if not isinstance(contract, Mapping):
        raise BenchmarkContractError("contract must be an object")
    if contract.get("schema_version") != "1.0":
        raise BenchmarkContractError("schema_version must be 1.0")
    if contract.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise BenchmarkContractError(
            f"contract.contract_id must be {EXPECTED_CONTRACT_ID}"
        )
    if contract.get("status") != "frozen":
        raise BenchmarkContractError("contract.status must be frozen")
    _require_nonempty_string(contract, "claim_boundary", "contract")

    shared = _require_mapping(contract, "shared_design", "contract")
    seeds = shared.get("random_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise BenchmarkContractError(
            "shared_design.random_seeds must contain at least three unique random seeds"
        )
    if tuple(seeds) != EXPECTED_RANDOM_SEEDS:
        expected_seeds = ", ".join(str(seed) for seed in EXPECTED_RANDOM_SEEDS)
        raise BenchmarkContractError(
            f"shared_design.random_seeds must be frozen as: {expected_seeds}"
        )
    _require_nonempty_string(shared, "feature_policy", "shared_design")
    tuning = _require_mapping(shared, "tuning", "shared_design")
    max_trials = tuning.get("max_trials_per_method")
    if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials < 1:
        raise BenchmarkContractError(
            "shared_design.tuning.max_trials_per_method must be a positive integer"
        )
    if max_trials != EXPECTED_MAX_TRIALS:
        raise BenchmarkContractError(
            "shared_design.tuning.max_trials_per_method must be frozen at 20"
        )
    if tuning.get("same_budget_for_all_methods") is not True:
        raise BenchmarkContractError(
            "shared_design.tuning.same_budget_for_all_methods must be true"
        )
    _require_nonempty_string(tuning, "objective", "shared_design.tuning")
    stopping = _require_mapping(shared, "stopping_rule", "shared_design")
    _require_nonempty_string(stopping, "type", "shared_design.stopping_rule")
    uncertainty = _require_mapping(shared, "uncertainty", "shared_design")
    confidence = uncertainty.get("confidence_level")
    if not isinstance(confidence, (int, float)) or not 0.0 < float(confidence) < 1.0:
        raise BenchmarkContractError(
            "shared_design.uncertainty.confidence_level must be between 0 and 1"
        )
    _require_nonempty_string_list(
        shared,
        "required_run_artifacts",
        "shared_design",
    )

    tasks = _require_mapping(contract, "tasks", "contract")
    if set(tasks) != set(EXPECTED_TASKS):
        raise BenchmarkContractError("contract must define exactly tasks C, S, and D")

    for task_id in EXPECTED_TASKS:
        task = _require_mapping(tasks, task_id, "tasks")
        context = f"tasks.{task_id}"
        _require_nonempty_string(task, "title", context)
        _require_nonempty_string(task, "estimand", context)
        _require_nonempty_string(task, "unit_of_analysis", context)
        split = _require_mapping(task, "split", context)
        _require_nonempty_string(split, "id", f"{context}.split")
        _require_nonempty_string(split, "strategy", f"{context}.split")
        _require_nonempty_string_list(split, "group_keys", f"{context}.split")
        _require_nonempty_string_list(
            split,
            "leakage_forbidden",
            f"{context}.split",
        )
        primary = _require_mapping(task, "primary_metric", context)
        primary_name = _require_nonempty_string(
            primary,
            "name",
            f"{context}.primary_metric",
        )
        if primary.get("direction") not in {"maximize", "minimize"}:
            raise BenchmarkContractError(
                f"{context}.primary_metric.direction must be maximize or minimize"
            )
        if (primary_name, primary.get("direction")) != EXPECTED_PRIMARY_METRICS[task_id]:
            expected_name, expected_direction = EXPECTED_PRIMARY_METRICS[task_id]
            raise BenchmarkContractError(
                f"{context}.primary_metric must be frozen as "
                f"{expected_name}/{expected_direction}"
            )
        _require_nonempty_string_list(task, "secondary_metrics", context)
        baselines = _require_nonempty_string_list(
            task,
            "required_simple_baselines",
            context,
        )
        if tuple(baselines) != EXPECTED_SIMPLE_BASELINES[task_id]:
            expected = ", ".join(EXPECTED_SIMPLE_BASELINES[task_id])
            raise BenchmarkContractError(
                f"{context}.required_simple_baselines must be frozen as: {expected}"
            )
        _require_nonempty_string_list(task, "null_controls", context)
        promotion = _require_mapping(task, "promotion", context)
        _require_probability(promotion, "minimum_coverage", f"{context}.promotion")
        _require_probability(
            promotion,
            "maximum_abstention_rate",
            f"{context}.promotion",
        )
        _require_probability(
            promotion,
            "minimum_seed_success_rate",
            f"{context}.promotion",
        )
        if promotion.get("requires_external_holdout") is not True:
            raise BenchmarkContractError(
                f"{context}.promotion.requires_external_holdout must be true"
            )
        if promotion.get("requires_null_controls") is not True:
            raise BenchmarkContractError(
                f"{context}.promotion.requires_null_controls must be true"
            )
        improvement_threshold = promotion.get("paired_improvement_ci_lower_gt")
        if (
            not isinstance(improvement_threshold, (int, float))
            or isinstance(improvement_threshold, bool)
            or not math.isfinite(float(improvement_threshold))
        ):
            raise BenchmarkContractError(
                f"{context}.promotion.paired_improvement_ci_lower_gt must be finite"
            )


def load_benchmark_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark contract JSON file."""
    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(
            f"could not load benchmark contract {contract_path}: {exc}"
        ) from exc
    validate_benchmark_contract(payload)
    return payload


def build_run_manifest(
    contract: Mapping[str, Any],
    *,
    task_id: str,
    dataset_id: str,
    method_id: str,
    method_role: str,
    code_revision: str,
    random_seed: int,
    input_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Bind one method run to the immutable comparison conditions."""
    validate_benchmark_contract(contract)
    if task_id not in EXPECTED_TASKS:
        raise BenchmarkContractError(f"unsupported task_id: {task_id}")
    for name, value in {
        "dataset_id": dataset_id,
        "method_id": method_id,
        "code_revision": code_revision,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkContractError(f"{name} must be a non-empty string")
    if method_role not in SUPPORTED_METHOD_ROLES:
        raise BenchmarkContractError(
            f"method_role must be one of {sorted(SUPPORTED_METHOD_ROLES)}"
        )
    seeds = contract["shared_design"]["random_seeds"]
    if random_seed not in seeds:
        raise BenchmarkContractError(
            f"random_seed {random_seed} is not pre-registered in the contract"
        )
    if not isinstance(input_artifacts, Mapping) or not input_artifacts:
        raise BenchmarkContractError("input_artifacts must be a non-empty mapping")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(digest, str)
        or not digest.strip()
        for name, digest in input_artifacts.items()
    ):
        raise BenchmarkContractError(
            "input_artifacts keys and digests must be non-empty strings"
        )
    invalid_digests = [
        digest
        for digest in input_artifacts.values()
        if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest) is None
    ]
    if invalid_digests:
        raise BenchmarkContractError(
            "input_artifacts must contain full SHA-256 digests in sha256:<hex> form"
        )

    task = contract["tasks"][task_id]
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": task_id,
        "split_id": task["split"]["id"],
        "dataset_id": dataset_id,
        "method_id": method_id,
        "method_role": method_role,
        "code_revision": code_revision,
        "random_seed": random_seed,
        "input_artifacts": dict(sorted(input_artifacts.items())),
        "primary_metric": task["primary_metric"],
        "promotion_status": "not_evaluated",
        "claim_boundary": contract["claim_boundary"],
    }


def _evidence_probability(evidence: Mapping[str, Any], key: str) -> float:
    value = evidence.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkContractError(f"promotion evidence {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise BenchmarkContractError(
            f"promotion evidence {key} must be between 0 and 1"
        )
    return numeric


def _beats_all_required_baselines(
    required_baselines: list[str],
    paired_improvement: Any,
    threshold: float,
) -> bool:
    if not isinstance(paired_improvement, Mapping):
        return False
    if any(baseline not in paired_improvement for baseline in required_baselines):
        return False
    for baseline in required_baselines:
        interval = paired_improvement.get(baseline)
        if not isinstance(interval, Mapping):
            return False
        values = [interval.get(key) for key in ("estimate", "ci_lower", "ci_upper")]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        ):
            return False
        estimate, ci_lower, ci_upper = (float(value) for value in values)
        if not ci_lower <= estimate <= ci_upper or not ci_lower > threshold:
            return False
    return True


def evaluate_promotion(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate pre-registered promotion gates without changing target ranking."""
    validate_benchmark_contract(contract)
    if not isinstance(evidence, Mapping):
        raise BenchmarkContractError("promotion evidence must be an object")
    if evidence.get("contract_sha256") != contract_digest(contract):
        raise BenchmarkContractError("promotion evidence contract_sha256 mismatch")
    task_id = evidence.get("task_id")
    if task_id not in EXPECTED_TASKS:
        raise BenchmarkContractError(f"unsupported promotion task_id: {task_id}")
    task = contract["tasks"][task_id]
    primary_metric = task["primary_metric"]["name"]
    if evidence.get("primary_metric") != primary_metric:
        raise BenchmarkContractError(
            f"promotion evidence must report primary metric {primary_metric}"
        )

    coverage = _evidence_probability(evidence, "coverage")
    abstention_rate = _evidence_probability(evidence, "abstention_rate")
    successful_seeds = evidence.get("successful_seeds")
    total_seeds = evidence.get("total_seeds")
    if (
        not isinstance(successful_seeds, int)
        or isinstance(successful_seeds, bool)
        or not isinstance(total_seeds, int)
        or isinstance(total_seeds, bool)
        or total_seeds < 1
        or successful_seeds < 0
        or successful_seeds > total_seeds
    ):
        raise BenchmarkContractError(
            "successful_seeds and total_seeds must define a valid count"
        )
    registered_seed_count = len(contract["shared_design"]["random_seeds"])
    if total_seeds != registered_seed_count:
        raise BenchmarkContractError(
            "total_seeds must equal the number of pre-registered random seeds"
        )

    promotion = task["promotion"]
    checks = {
        "minimum_coverage": coverage >= promotion["minimum_coverage"],
        "maximum_abstention_rate": (
            abstention_rate <= promotion["maximum_abstention_rate"]
        ),
        "seed_success_rate": (
            successful_seeds / total_seeds
            >= promotion["minimum_seed_success_rate"]
        ),
        "external_holdout": (
            evidence.get("external_holdout_passed") is True
            if promotion["requires_external_holdout"]
            else True
        ),
        "null_controls": (
            evidence.get("null_controls_passed") is True
            if promotion["requires_null_controls"]
            else True
        ),
        "comparison_parity": evidence.get("same_split_features_and_budget") is True,
        "beats_all_required_simple_baselines": _beats_all_required_baselines(
            task["required_simple_baselines"],
            evidence.get("paired_improvement"),
            float(promotion["paired_improvement_ci_lower_gt"]),
        ),
    }
    promoted = all(checks.values())
    return {
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_digest(contract),
        "task_id": task_id,
        "status": "promoted" if promoted else "not_promoted",
        "claim_level": (
            "benchmark_supported_candidate" if promoted else "hypothesis_only"
        ),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "claim_boundary": contract["claim_boundary"],
    }


def write_preregistration_bundle(
    contract: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write an immutable contract snapshot and a machine-readable task registry."""
    validate_benchmark_contract(contract)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    digest = contract_digest(contract)
    registry = {
        "schema_version": "1.0",
        "contract_id": contract["contract_id"],
        "contract_sha256": digest,
        "tasks": [
            {
                "task_id": task_id,
                "title": contract["tasks"][task_id]["title"],
                "estimand": contract["tasks"][task_id]["estimand"],
                "split_id": contract["tasks"][task_id]["split"]["id"],
                "primary_metric": contract["tasks"][task_id]["primary_metric"],
                "required_simple_baselines": contract["tasks"][task_id][
                    "required_simple_baselines"
                ],
                "promotion_status": "not_evaluated",
            }
            for task_id in EXPECTED_TASKS
        ],
        "claim_boundary": contract["claim_boundary"],
    }
    (destination / "contract_snapshot.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "task_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "contract_id": contract["contract_id"],
        "contract_sha256": digest,
        "output_dir": str(destination),
        "task_count": len(EXPECTED_TASKS),
    }
