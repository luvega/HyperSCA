from __future__ import annotations

import pytest

from src.behavior_grammar.rules import (
    BehaviorDictionary,
    BehaviorRule,
    RuleParameters,
    SignalDictionary,
    evaluate_response,
)


def test_behavior_rule_parses_human_readable_statement():
    rule = BehaviorRule.from_statement(
        "In CAF cells, ECM increases migration with a Hill response, base 0.2, max 1.0, half-max 0.4, hill 2.",
        evidence_refs=["synthetic:ecm_migration"],
    )

    assert rule.cell_type == "CAF"
    assert rule.signal == "ECM"
    assert rule.direction == "increases"
    assert rule.behavior == "migration"
    assert rule.response_function == "hill"
    assert rule.parameters.base == pytest.approx(0.2)
    assert rule.parameters.saturation == pytest.approx(1.0)
    assert rule.parameters.half_max == pytest.approx(0.4)
    assert rule.parameters.hill_power == pytest.approx(2.0)
    assert rule.evidence_refs == ("synthetic:ecm_migration",)


def test_dictionary_validation_rejects_unknown_signal_and_behavior():
    signals = SignalDictionary.default()
    behaviors = BehaviorDictionary.default()
    rule = BehaviorRule(
        cell_type="TAM",
        signal="unknown_signal",
        direction="increases",
        behavior="motility",
        response_function="hill",
        parameters=RuleParameters(),
    )

    with pytest.raises(ValueError, match="unknown signal"):
        rule.validate(signals=signals, behaviors=behaviors)


def test_response_functions_are_bounded_and_monotonic():
    hill_values = evaluate_response(
        [0.0, 0.5, 1.0],
        RuleParameters(base=0.1, saturation=0.9, half_max=0.5, hill_power=2.0),
        response_function="hill",
        direction="increases",
    )
    step_values = evaluate_response(
        [0.0, 0.5, 1.0],
        RuleParameters(base=0.1, saturation=0.9, half_max=0.5),
        response_function="step",
        direction="decreases",
    )

    assert hill_values[0] >= 0.1
    assert hill_values[-1] <= 0.9
    assert hill_values[0] < hill_values[1] < hill_values[2]
    assert step_values.tolist() == pytest.approx([0.9, 0.1, 0.1])


def test_invalid_response_function_is_rejected():
    rule = BehaviorRule(
        cell_type="CD8_T",
        signal="PD1",
        direction="increases",
        behavior="attack",
        response_function="sigmoid",
        parameters=RuleParameters(),
    )

    with pytest.raises(ValueError, match="response_function"):
        rule.validate()
