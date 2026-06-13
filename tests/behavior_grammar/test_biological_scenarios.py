from __future__ import annotations

import pandas as pd

from src.behavior_grammar.rules import BehaviorRule, RuleParameters, RuleSet
from src.behavior_grammar.simulation import compare_intervention_scenarios, simulate_virtual_tissue


def _rule(cell_type: str, signal: str, behavior: str, *, saturation: float = 0.8) -> BehaviorRule:
    return BehaviorRule(
        cell_type=cell_type,
        signal=signal,
        direction="increases",
        behavior=behavior,
        response_function="hill",
        parameters=RuleParameters(base=0.05, saturation=saturation, half_max=0.3, hill_power=2.0),
    )


def test_caf_ecm_rule_increases_transition_index():
    ruleset = RuleSet(run_id="caf_ecm", rules=(_rule("CAF", "ECM", "transition"),))
    expr = pd.DataFrame({"cell_type": ["CAF"], "ECM": [2.0]})

    trajectory, summary = simulate_virtual_tissue(ruleset, expr, time_steps=4)

    caf = trajectory[trajectory["cell_type"] == "CAF"]
    assert caf["transition_index"].iloc[-1] > caf["transition_index"].iloc[0]
    assert summary["final_total_population"] > caf["population"].iloc[0]


def test_tam_egf_rule_increases_migration_index():
    ruleset = RuleSet(run_id="tam_egf", rules=(_rule("TAM", "EGF", "migration"),))
    expr = pd.DataFrame({"cell_type": ["TAM"], "EGF": [1.5]})

    trajectory, _ = simulate_virtual_tissue(ruleset, expr, time_steps=4)

    tam = trajectory[trajectory["cell_type"] == "TAM"]
    assert tam["migration_index"].iloc[-1] > tam["migration_index"].iloc[0]


def test_cd8_attack_rule_reduces_tumor_population():
    ruleset = RuleSet(run_id="cd8_attack", rules=(_rule("CD8_T", "IFNG", "attack", saturation=1.0),))
    expr = pd.DataFrame(
        {
            "cell_type": ["CD8_T", "Tumor"],
            "IFNG": [2.0, 0.0],
        }
    )

    trajectory, _ = simulate_virtual_tissue(ruleset, expr, time_steps=5)

    tumor = trajectory[trajectory["cell_type"] == "Tumor"]
    assert tumor["population"].iloc[-1] < tumor["population"].iloc[0]


def test_pd1_cd137_combo_improves_tumor_control_over_pd1_only():
    base_rules = RuleSet(
        run_id="combo_base",
        rules=(
            _rule("CD8_T", "PDCD1", "exhaustion", saturation=0.8),
            _rule("CD8_T", "IFNG", "attack", saturation=0.7),
        ),
    )
    expr = pd.DataFrame(
        {
            "cell_type": ["CD8_T", "Tumor"],
            "PDCD1": [1.0, 0.0],
            "IFNG": [1.0, 0.0],
            "CD137_AGONIST": [1.0, 0.0],
        }
    )

    comparison = compare_intervention_scenarios(
        base_rules,
        expr,
        scenarios={
            "pd1_only": {"PDCD1": 0.2},
            "pd1_cd137_combo": {"PDCD1": 0.2, "CD137_AGONIST": 1.8},
        },
        time_steps=5,
    )

    assert comparison.loc["pd1_cd137_combo", "final_tumor_population"] < comparison.loc["pd1_only", "final_tumor_population"]
