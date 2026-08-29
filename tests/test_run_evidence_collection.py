from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from src.evaluation.run_evidence_collection import (
    validate_paired_collection,
    write_invalidation_record,
)
from src.evaluation.run_evidence_identity import (
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import (
    RunEvidencePublisher,
    VerifiedRunEvidence,
    verify_run_evidence_bundle,
)


MODEL_SEEDS = (11, 23, 47)


def identity_for(
    *,
    model_seed: int,
    data_split_seed: int = 19911,
    split_sha: str = "b" * 64,
    unit_record: object | None = None,
    analysis_sha: str = "d" * 64,
    protocol_identity: str = "a" * 64,
) -> RunEvidenceIdentity:
    if unit_record is None:
        unit_record = {"units": ["platform:sample:block-1", "platform:sample:block-2"]}
    return RunEvidenceIdentity(
        schema_version="1.0",
        protocol_version="hypersca-methods-v2.1",
        protocol_identity=protocol_identity,
        claim_id="spatial",
        benchmark_id="osta_colon",
        data_scopes=("train", "tune"),
        data_split_seed=data_split_seed,
        model_seed=model_seed,
        data_split_identity_sha256=split_sha,
        statistical_unit_schema="osta_platform_sample_block_v1",
        statistical_unit_identity_sha256=canonical_sha256(unit_record),
        analysis_identity_sha256=analysis_sha,
        input_identity_sha256="e" * 64,
        config_identity_sha256=(f"{model_seed:064x}"),
        code_identity_sha256="0" * 64,
        evidence_role="pilot_audit_only",
    )


def publish_run(
    root: Path,
    *,
    model_seed: int,
    data_split_seed: int = 19911,
    split_sha: str = "b" * 64,
    unit_record: object | None = None,
    analysis_sha: str = "d" * 64,
    protocol_identity: str = "a" * 64,
    failure: bool = False,
) -> VerifiedRunEvidence:
    if unit_record is None:
        unit_record = {"units": ["platform:sample:block-1", "platform:sample:block-2"]}
    identity = identity_for(
        model_seed=model_seed,
        data_split_seed=data_split_seed,
        split_sha=split_sha,
        unit_record=unit_record,
        analysis_sha=analysis_sha,
        protocol_identity=protocol_identity,
    )
    publisher = RunEvidencePublisher.begin(
        output_dir=root / f"seed_{model_seed}",
        identity=identity,
        statistical_unit_record=unit_record,
        required_artifacts=() if failure else ("metrics.json",),
        maximum_bundle_bytes=32_768,
    )
    if failure:
        output = publisher.finalize_failure(
            status="failed_runtime", reason="registered worker failure"
        )
    else:
        publisher.add_bytes(
            "metrics.json",
            json.dumps({"metric": model_seed / 100}).encode("utf-8"),
            media_type="application/json",
        )
        output = publisher.finalize_completed(
            summary={"status": "audit_only", "metric": model_seed / 100}
        )
    return verify_run_evidence_bundle(output, expected_identity=identity)


def valid_runs(tmp_path: Path) -> tuple[VerifiedRunEvidence, ...]:
    return tuple(publish_run(tmp_path, model_seed=seed) for seed in MODEL_SEEDS)


def test_valid_paired_collection_is_seed_ordered_and_deeply_frozen(
    tmp_path: Path,
) -> None:
    runs = valid_runs(tmp_path)

    collection = validate_paired_collection(
        tuple(reversed(runs)), expected_model_seeds=MODEL_SEEDS
    )

    assert tuple(run.identity.model_seed for run in collection.runs) == MODEL_SEEDS
    assert collection.expected_model_seeds == MODEL_SEEDS
    assert collection.statistics_allowed is True
    assert len(collection.collection_identity_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        collection.statistics_allowed = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        collection.runs[0].summary["metric"] = 99  # type: ignore[index]


def test_collection_rejects_model_seed_changing_data_split(tmp_path: Path) -> None:
    runs = tuple(
        publish_run(
            tmp_path,
            model_seed=seed,
            data_split_seed=seed + 1000,
            split_sha=f"{seed:064x}",
        )
        for seed in MODEL_SEEDS
    )
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection(runs, expected_model_seeds=MODEL_SEEDS)


def test_collection_rejects_collapsed_or_changed_statistical_units(
    tmp_path: Path,
) -> None:
    full_units = {"units": ["platform:sample:block-1", "platform:sample:block-2"]}
    collapsed = {"units": ["platform:sample"]}
    runs = (
        publish_run(tmp_path, model_seed=11, unit_record=full_units),
        publish_run(tmp_path, model_seed=23, unit_record=full_units),
        publish_run(tmp_path, model_seed=47, unit_record=collapsed),
    )
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection(runs, expected_model_seeds=MODEL_SEEDS)


@pytest.mark.parametrize(
    "replacement",
    [
        (11, 23),
        (11, 23, 47, 59),
        (11, 23, 23),
        [11, 23, 47],
        (11, True, 47),
    ],
)
def test_expected_seed_contract_is_exact_unique_and_complete(
    tmp_path: Path, replacement: object
) -> None:
    runs = valid_runs(tmp_path)
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection(runs, expected_model_seeds=replacement)


@pytest.mark.parametrize("changed", ["analysis", "protocol", "split_sha"])
def test_collection_rejects_every_nonpaired_identity_axis(
    tmp_path: Path, changed: str
) -> None:
    first = publish_run(tmp_path, model_seed=11)
    second = publish_run(tmp_path, model_seed=23)
    arguments: dict[str, object] = {"model_seed": 47}
    if changed == "analysis":
        arguments["analysis_sha"] = "1" * 64
    elif changed == "protocol":
        arguments["protocol_identity"] = "2" * 64
    else:
        arguments["split_sha"] = "3" * 64
    third = publish_run(tmp_path, **arguments)
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection((first, second, third), expected_model_seeds=MODEL_SEEDS)


def test_terminal_failure_is_retained_but_disables_statistics(tmp_path: Path) -> None:
    runs = (
        publish_run(tmp_path, model_seed=11),
        publish_run(tmp_path, model_seed=23, failure=True),
        publish_run(tmp_path, model_seed=47),
    )

    collection = validate_paired_collection(runs, expected_model_seeds=MODEL_SEEDS)

    assert tuple(run.terminal_status for run in collection.runs) == (
        "completed",
        "failed_runtime",
        "completed",
    )
    assert collection.statistics_allowed is False


def test_directly_modified_verified_record_is_replayed_and_rejected(
    tmp_path: Path,
) -> None:
    runs = valid_runs(tmp_path)
    forged = replace(runs[2], terminal_status="failed_runtime")

    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection(
            (runs[0], runs[1], forged), expected_model_seeds=MODEL_SEEDS
        )


def test_invalidation_record_is_exclusive_and_does_not_modify_runs(
    tmp_path: Path,
) -> None:
    runs = valid_runs(tmp_path)
    before = tuple(run.bundle_identity_sha256 for run in runs)
    output = tmp_path / "invalidation_record.json"

    published = write_invalidation_record(
        output,
        runs,
        category="paired_identity_mismatch",
        reason="model seed changed the registered data split",
    )

    record = json.loads(published.read_text(encoding="utf-8"))
    assert record["category"] == "paired_identity_mismatch"
    assert record["run_identity_sha256"] == [
        run.identity.run_identity_sha256 for run in runs
    ]
    assert tuple(
        verify_run_evidence_bundle(
            run.output_dir, expected_identity=run.identity
        ).bundle_identity_sha256
        for run in runs
    ) == before
    with pytest.raises(RunEvidenceError, match="publication_conflict"):
        write_invalidation_record(
            output,
            runs,
            category="paired_identity_mismatch",
            reason="do not overwrite",
        )


def publish_bridge_run(
    root: Path,
    *,
    model_seed: int,
    split_sha: str = "8" * 64,
    unit_record: object | None = None,
    artifact_paths: tuple[str, ...] = ("primary_metric_summary.json",),
) -> VerifiedRunEvidence:
    if unit_record is None:
        unit_record = {
            "units": [["a" * 64, "neighbor"], ["b" * 64, "own"]]
        }
    identity = RunEvidenceIdentity(
        schema_version="1.0",
        protocol_version="hypersca-methods-v3.0",
        protocol_identity="7" * 64,
        claim_id="bridge",
        benchmark_id="spatial_perturbation_bridge",
        data_scopes=("synthetic",),
        data_split_seed=11,
        model_seed=model_seed,
        data_split_identity_sha256=split_sha,
        statistical_unit_schema="bridge_prediction_unit_v1",
        statistical_unit_identity_sha256=canonical_sha256(unit_record),
        analysis_identity_sha256="6" * 64,
        input_identity_sha256="5" * 64,
        config_identity_sha256=f"{model_seed:064x}",
        code_identity_sha256="4" * 64,
        evidence_role="synthetic_audit_only",
    )
    publisher = RunEvidencePublisher.begin(
        output_dir=root / f"bridge_{model_seed}",
        identity=identity,
        statistical_unit_record=unit_record,
        required_artifacts=artifact_paths,
        maximum_bundle_bytes=16_384,
    )
    for relative_path in artifact_paths:
        publisher.add_bytes(
            relative_path,
            b'{"neighbor_effect_rmse":0.0}',
            media_type=(
                "application/json" if relative_path.endswith(".json") else "text/csv"
            ),
        )
    output = publisher.finalize_completed(
        summary={"claim_id": "bridge", "evidence_role": "synthetic_audit_only"}
    )
    return verify_run_evidence_bundle(output, expected_identity=identity)


def test_incomplete_bridge_artifact_set_cannot_bypass_semantic_replay(
    tmp_path: Path,
) -> None:
    first = publish_bridge_run(tmp_path, model_seed=11)
    second = publish_bridge_run(tmp_path, model_seed=23)

    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection((second, first), expected_model_seeds=(11, 23))


@pytest.mark.parametrize("artifact_count", [10, 12])
def test_bridge_collection_rejects_wrong_artifact_count_even_when_resealed(
    tmp_path: Path, artifact_count: int
) -> None:
    expected = (
        "split_manifest.json",
        "capability_record.json",
        "neighbor_units.csv",
        "predictions_hypersca.csv",
        "predictions_matched_euclidean.csv",
        "predictions_hypersca_own_only.csv",
        "primary_metric_units.csv",
        "primary_metric_summary.json",
        "secondary_metrics.csv",
        "resource_usage.json",
        "claim_decision.json",
    )
    artifact_paths = (
        expected[:artifact_count]
        if artifact_count == 10
        else expected + ("unexpected.json",)
    )
    first = publish_bridge_run(
        tmp_path, model_seed=11, artifact_paths=artifact_paths
    )
    second = publish_bridge_run(
        tmp_path, model_seed=23, artifact_paths=artifact_paths
    )

    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection((first, second), expected_model_seeds=(11, 23))


def test_bridge_collection_still_rejects_changed_statistical_units(
    tmp_path: Path,
) -> None:
    first = publish_bridge_run(tmp_path, model_seed=11)

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed = publish_bridge_run(
        changed_root,
        model_seed=23,
        unit_record={"units": [["a" * 64, "neighbor"]]},
    )
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection((first, changed), expected_model_seeds=(11, 23))


def test_bridge_collection_replays_and_rejects_tampered_evidence(tmp_path: Path) -> None:
    first = publish_bridge_run(tmp_path, model_seed=11)
    second = publish_bridge_run(tmp_path, model_seed=23)
    (second.output_dir / "primary_metric_summary.json").write_bytes(
        b'{"neighbor_effect_rmse":1.0}'
    )

    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection((first, second), expected_model_seeds=(11, 23))
