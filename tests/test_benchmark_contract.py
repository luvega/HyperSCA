from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.benchmark_contract import (
    BenchmarkContractError,
    build_run_manifest,
    contract_digest,
    evaluate_promotion,
    load_benchmark_contract,
    validate_benchmark_contract,
    write_preregistration_bundle,
)


CONTRACT_PATH = Path("configs/benchmark_contract_v1.json")


def test_default_contract_freezes_three_tasks_and_shared_budget() -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)

    assert contract["contract_id"] == "hypersca-csd-benchmark-v1"
    assert set(contract["tasks"]) == {"C", "S", "D"}
    assert contract["shared_design"]["random_seeds"] == [11, 23, 47, 71, 97]
    assert contract["shared_design"]["tuning"]["max_trials_per_method"] == 20
    assert contract["tasks"]["C"]["required_simple_baselines"] == [
        "mean_difference"
    ]
    assert contract["tasks"]["S"]["required_simple_baselines"] == [
        "own_only",
        "fixed_distance_decay",
    ]
    assert contract["tasks"]["D"]["required_simple_baselines"] == [
        "non_spatial_signature_reversal"
    ]


def test_contract_digest_is_canonical_and_detects_change() -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)
    reordered = json.loads(json.dumps(contract, sort_keys=True))

    assert contract_digest(contract) == contract_digest(reordered)

    changed = copy.deepcopy(contract)
    changed["shared_design"]["tuning"]["max_trials_per_method"] = 21
    assert contract_digest(contract) != contract_digest(changed)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda c: c["tasks"].pop("S"), "exactly tasks C, S, and D"),
        (
            lambda c: c["shared_design"].update(random_seeds=[11]),
            "at least three unique random seeds",
        ),
        (
            lambda c: c["shared_design"].update(
                random_seeds=[12, 24, 48, 72, 98]
            ),
            "frozen as",
        ),
        (
            lambda c: c["shared_design"]["tuning"].update(
                max_trials_per_method=21
            ),
            "frozen at 20",
        ),
        (
            lambda c: c["tasks"]["C"].update(required_simple_baselines=[]),
            "required_simple_baselines",
        ),
        (
            lambda c: c["tasks"]["C"]["primary_metric"].update(name="auroc"),
            "primary_metric must be frozen",
        ),
        (
            lambda c: c["tasks"]["S"]["promotion"].update(
                minimum_coverage=1.2
            ),
            "minimum_coverage",
        ),
    ],
)
def test_invalid_contracts_fail_closed(mutation, message: str) -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)
    mutation(contract)

    with pytest.raises(BenchmarkContractError, match=message):
        validate_benchmark_contract(contract)


def test_run_manifest_binds_method_to_contract_split_and_seed() -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)
    manifest = build_run_manifest(
        contract,
        task_id="C",
        dataset_id="causalbench-example",
        method_id="hypersca-causal",
        method_role="candidate",
        code_revision="abc1234",
        random_seed=23,
        input_artifacts={"expression": "sha256:" + "d" * 64},
    )

    assert manifest["contract_sha256"] == contract_digest(contract)
    assert manifest["split_id"] == contract["tasks"]["C"]["split"]["id"]
    assert manifest["random_seed"] == 23
    assert manifest["promotion_status"] == "not_evaluated"

    with pytest.raises(BenchmarkContractError, match="pre-registered"):
        build_run_manifest(
            contract,
            task_id="C",
            dataset_id="causalbench-example",
            method_id="hypersca-causal",
            method_role="candidate",
            code_revision="abc1234",
            random_seed=999,
            input_artifacts={"expression": "sha256:" + "d" * 64},
        )

    with pytest.raises(BenchmarkContractError, match="SHA-256"):
        build_run_manifest(
            contract,
            task_id="C",
            dataset_id="causalbench-example",
            method_id="hypersca-causal",
            method_role="candidate",
            code_revision="abc1234",
            random_seed=23,
            input_artifacts={"expression": "sha256:deadbeef"},
        )


def _passing_evidence(contract: dict, task_id: str) -> dict:
    baselines = contract["tasks"][task_id]["required_simple_baselines"]
    return {
        "contract_sha256": contract_digest(contract),
        "task_id": task_id,
        "primary_metric": contract["tasks"][task_id]["primary_metric"]["name"],
        "coverage": 0.95,
        "abstention_rate": 0.05,
        "successful_seeds": 5,
        "total_seeds": 5,
        "external_holdout_passed": True,
        "null_controls_passed": True,
        "same_split_features_and_budget": True,
        "paired_improvement": {
            baseline: {
                "estimate": 0.05,
                "ci_lower": 0.01,
                "ci_upper": 0.09,
            }
            for baseline in baselines
        },
    }


def test_promotion_requires_positive_paired_ci_against_every_simple_baseline() -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)
    evidence = _passing_evidence(contract, "S")

    decision = evaluate_promotion(contract, evidence)

    assert decision["status"] == "promoted"
    assert decision["claim_level"] == "benchmark_supported_candidate"
    assert all(decision["checks"].values())

    evidence["paired_improvement"]["own_only"]["ci_lower"] = -0.001
    failed = evaluate_promotion(contract, evidence)
    assert failed["status"] == "not_promoted"
    assert not failed["checks"]["beats_all_required_simple_baselines"]


@pytest.mark.parametrize(
    "field, value, failed_check",
    [
        ("coverage", 0.79, "minimum_coverage"),
        ("abstention_rate", 0.21, "maximum_abstention_rate"),
        ("external_holdout_passed", False, "external_holdout"),
        ("null_controls_passed", False, "null_controls"),
        ("same_split_features_and_budget", False, "comparison_parity"),
        ("successful_seeds", 3, "seed_success_rate"),
    ],
)
def test_promotion_fails_each_reliability_gate(
    field: str,
    value,
    failed_check: str,
) -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)
    evidence = _passing_evidence(contract, "D")
    evidence[field] = value

    decision = evaluate_promotion(contract, evidence)

    assert decision["status"] == "not_promoted"
    assert not decision["checks"][failed_check]


def test_preregistration_bundle_is_audit_ready(tmp_path: Path) -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)

    summary = write_preregistration_bundle(contract, tmp_path)

    assert summary["contract_sha256"] == contract_digest(contract)
    snapshot = json.loads((tmp_path / "contract_snapshot.json").read_text())
    registry = json.loads((tmp_path / "task_registry.json").read_text())
    assert snapshot == contract
    assert [task["task_id"] for task in registry["tasks"]] == ["C", "S", "D"]
    assert all(task["promotion_status"] == "not_evaluated" for task in registry["tasks"])


def test_contract_cli_validates_and_writes_bundle(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_benchmark_contract.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["status"] == "valid"
    assert summary["task_count"] == 3
    assert (tmp_path / "contract_snapshot.json").exists()
    assert (tmp_path / "task_registry.json").exists()
