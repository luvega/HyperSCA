from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from src.evaluation.methods_protocol import (
    ModelBudget,
    ProtocolExecutionState,
    record_release_attempt,
    transition_protocol,
    validate_comparator_fairness,
)


def _budget(parameter_count: int, *, hierarchy_loss_enabled: bool) -> ModelBudget:
    return ModelBudget(
        parameter_count=parameter_count,
        optimizer_family="adamw",
        max_updates=20_000,
        early_stopping_patience=500,
        tuning_trials=20,
        data_identity="a" * 64,
        gene_identity="b" * 64,
        seed=11,
        hierarchy_loss_enabled=hierarchy_loss_enabled,
    )


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=900_000, max_value=1_100_000))
def test_parameter_fairness_has_an_exact_five_percent_boundary(
    parameter_count: int,
) -> None:
    hypersca = _budget(1_000_000, hierarchy_loss_enabled=True)
    confirmatory = _budget(parameter_count, hierarchy_loss_enabled=True)
    attribution = _budget(1_000_000, hierarchy_loss_enabled=False)

    if abs(parameter_count - 1_000_000) <= 50_000:
        validate_comparator_fairness(
            hypersca, confirmatory, attribution, parameter_tolerance=0.05
        )
    else:
        with pytest.raises(ValueError, match="parameter"):
            validate_comparator_fairness(
                hypersca, confirmatory, attribution, parameter_tolerance=0.05
            )


@settings(max_examples=60, deadline=None)
@given(
    status=st.sampled_from(
        (
            "completed",
            "failed_invalid_input",
            "failed_invalid_output",
            "failed_timeout",
            "failed_resource",
            "failed_runtime",
            "not_applicable",
        )
    )
)
def test_release_state_never_allows_a_second_identity_or_second_score(
    status: str,
) -> None:
    ready = transition_protocol(
        ProtocolExecutionState(phase="pilot", redesigns_used=0, release_identity=None),
        "freeze_release",
    )
    result = record_release_attempt(ready, run_identity="c" * 64, status=status)

    if result.phase == "release_retryable":
        with pytest.raises(ValueError, match="same run identity"):
            record_release_attempt(result, run_identity="d" * 64, status="completed")
    elif result.phase == "release_scored":
        with pytest.raises(ValueError, match="already been scored"):
            record_release_attempt(result, run_identity="c" * 64, status="completed")
    else:
        with pytest.raises(ValueError, match="release attempts require"):
            record_release_attempt(result, run_identity="c" * 64, status="completed")
