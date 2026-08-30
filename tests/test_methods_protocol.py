from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from src.evaluation.methods_protocol import (
    ModelBudget,
    ProtocolExecutionState,
    authorize_data_scope,
    default_methods_protocol,
    protocol_identity,
    protocol_to_mapping,
    record_release_attempt,
    transition_protocol,
    validate_comparator_fairness,
    validate_crc_application_request,
)


ROOT = Path(__file__).resolve().parents[1]


def _budget(
    *,
    parameters: int = 1_000_000,
    hierarchy_loss: bool = True,
    max_updates: int = 20_000,
) -> ModelBudget:
    return ModelBudget(
        parameter_count=parameters,
        optimizer_family="adamw",
        max_updates=max_updates,
        early_stopping_patience=500,
        tuning_trials=20,
        data_identity="a" * 64,
        gene_identity="b" * 64,
        seed=11,
        hierarchy_loss_enabled=hierarchy_loss,
    )


def test_default_protocol_freezes_one_confirmatory_metric_per_benchmark() -> None:
    protocol = default_methods_protocol()

    assert protocol.protocol_version == "hypersca-methods-v2.1"
    assert protocol.spatial_primary_metric == "neighborhood_preservation_at_k"
    assert protocol.spatial_primary_k == 15
    assert protocol.spatial_secondary_k == (5, 30)
    assert protocol.causal_primary_metric == "directed_edge_average_precision"
    assert protocol.spatial_confirmatory_comparator == "euclidean_autoencoder"
    assert (
        protocol.spatial_attribution_comparator
        == "hypersca_without_hierarchy_loss"
    )
    assert protocol.causal_confirmatory_comparator == "mean_difference"
    assert protocol.causal_attribution_comparator == "hypersca_c_shared_only"
    assert protocol.pilot_seeds == (11, 23, 47)
    assert protocol.release_seeds == (11, 23, 47, 71, 101)
    assert protocol.bootstrap_resamples == 10_000
    assert protocol.multiple_testing == "holm_two_confirmatory_claims"
    assert protocol.crc_role == "application_only"
    assert protocol.resource_policy.gpu_count == 1
    assert protocol.resource_policy.gpu_memory_gib == 40
    assert protocol.resource_policy.ram_gib == 128
    assert protocol.resource_policy.artifact_gib_per_run == 5
    with pytest.raises(FrozenInstanceError):
        protocol.pilot_seeds = (1,)  # type: ignore[misc]


def test_protocol_v21_records_the_single_pilot_redesign() -> None:
    payload = protocol_to_mapping(default_methods_protocol())

    assert payload["execution"]["redesigns_used"] == 1
    assert payload["execution"]["redesign_reason"] == (
        "causalbench_representation_comparators_were_not_executable"
    )
    assert payload["comparators"]["spatial"] == {
        "confirmatory": "euclidean_autoencoder",
        "attribution": "hypersca_without_hierarchy_loss",
    }
    assert payload["comparators"]["causal"] == {
        "confirmatory": "mean_difference",
        "attribution": "hypersca_c_shared_only",
    }


def test_protocol_identity_is_canonical_and_release_yaml_matches_it() -> None:
    protocol = default_methods_protocol()
    payload = yaml.safe_load(
        (ROOT / "configs" / "hypersca_methods_release.yaml").read_text(encoding="utf-8")
    )

    assert payload == protocol_to_mapping(protocol)
    assert protocol_identity(protocol) == protocol_identity(default_methods_protocol())
    assert len(protocol_identity(protocol)) == 64


def test_pilot_cannot_access_release_holdout_and_release_is_score_only() -> None:
    protocol = default_methods_protocol()

    assert authorize_data_scope(protocol, phase="pilot", scope="train") == "fit_allowed"
    assert (
        authorize_data_scope(protocol, phase="pilot", scope="tune")
        == "selection_allowed"
    )
    with pytest.raises(ValueError, match="sealed"):
        authorize_data_scope(protocol, phase="pilot", scope="release_holdout")
    assert (
        authorize_data_scope(protocol, phase="release", scope="release_holdout")
        == "score_once_only"
    )
    with pytest.raises(ValueError, match="training"):
        authorize_data_scope(
            protocol, phase="release", scope="release_holdout", operation="fit"
        )


def test_comparator_fairness_enforces_capacity_budget_and_single_attribution_change() -> (
    None
):
    hypersca = _budget()
    euclidean = _budget(parameters=1_049_999)
    attribution = _budget(hierarchy_loss=False)

    validate_comparator_fairness(
        hypersca, euclidean, attribution, parameter_tolerance=0.05
    )
    with pytest.raises(ValueError, match="parameter"):
        validate_comparator_fairness(
            hypersca,
            _budget(parameters=1_050_001),
            attribution,
            parameter_tolerance=0.05,
        )
    with pytest.raises(ValueError, match="max_updates"):
        validate_comparator_fairness(
            hypersca,
            _budget(max_updates=19_999),
            attribution,
            parameter_tolerance=0.05,
        )
    with pytest.raises(ValueError, match="only hierarchy_loss_enabled"):
        validate_comparator_fairness(
            hypersca,
            euclidean,
            ModelBudget(
                parameter_count=1_000_000,
                optimizer_family="adamw",
                max_updates=20_000,
                early_stopping_patience=500,
                tuning_trials=19,
                data_identity="a" * 64,
                gene_identity="b" * 64,
                seed=11,
                hierarchy_loss_enabled=False,
            ),
            parameter_tolerance=0.05,
        )


def test_protocol_state_allows_one_redesign_and_same_identity_infrastructure_retry() -> (
    None
):
    state = ProtocolExecutionState(
        phase="pilot", redesigns_used=0, release_identity=None
    )
    state = transition_protocol(state, "redesign_after_pilot_failure")
    assert state.redesigns_used == 1
    with pytest.raises(ValueError, match="one protocol redesign"):
        transition_protocol(state, "redesign_after_pilot_failure")

    ready = transition_protocol(state, "freeze_release")
    retryable = record_release_attempt(
        ready, run_identity="c" * 64, status="failed_runtime"
    )
    assert retryable.phase == "release_retryable"
    with pytest.raises(ValueError, match="same run identity"):
        record_release_attempt(retryable, run_identity="d" * 64, status="completed")
    scored = record_release_attempt(
        retryable, run_identity="c" * 64, status="completed"
    )
    assert scored.phase == "release_scored"
    with pytest.raises(ValueError, match="already been scored"):
        record_release_attempt(scored, run_identity="c" * 64, status="completed")


def test_crc_is_application_only_and_cannot_rescue_or_tune_promotion() -> None:
    protocol = default_methods_protocol()

    validate_crc_application_request(
        protocol,
        tunes_hyperparameters=False,
        selects_baseline=False,
        changes_public_threshold=False,
        attempts_promotion_rescue=False,
    )
    for field in (
        "tunes_hyperparameters",
        "selects_baseline",
        "changes_public_threshold",
        "attempts_promotion_rescue",
    ):
        request = {
            "tunes_hyperparameters": False,
            "selects_baseline": False,
            "changes_public_threshold": False,
            "attempts_promotion_rescue": False,
        }
        request[field] = True
        with pytest.raises(ValueError, match="application-only"):
            validate_crc_application_request(protocol, **request)
