"""Lightweight virtual tissue simulation for behavior grammar rule sets."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.behavior_grammar.rules import BehaviorRule, RuleParameters, RuleSet, evaluate_response


_BEHAVIOR_COLUMNS = ("migration", "attack", "exhaustion", "secretion", "transition")


def simulate_virtual_tissue(
    ruleset: RuleSet,
    cluster_expression: pd.DataFrame | None = None,
    *,
    time_steps: int = 12,
    dt: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a deterministic toy virtual tissue simulation from grammar rules."""
    if time_steps < 2:
        raise ValueError("time_steps must be at least 2")
    ruleset.validate()
    expr = _normalize_expression(cluster_expression)
    cell_types = _cell_types(ruleset, expr)
    population = _initial_population(cell_types, expr)
    behavior_state = {name: np.zeros(len(cell_types), dtype=float) for name in _BEHAVIOR_COLUMNS}
    index = {cell_type: i for i, cell_type in enumerate(cell_types)}

    rows: list[dict[str, Any]] = []
    for t in range(time_steps):
        _append_rows(rows, t, cell_types, population, behavior_state)
        if t == time_steps - 1:
            break
        deltas = np.zeros(len(cell_types), dtype=float)
        for rule in ruleset.rules:
            i = index.get(rule.cell_type)
            if i is None:
                continue
            signal = _signal_value(expr, rule.cell_type, rule.signal, population[i])
            effect = float(
                evaluate_response(
                    [signal],
                    rule.parameters,
                    response_function=rule.response_function,
                    direction=rule.direction,
                )[0]
            )
            _apply_rule_effect(rule, effect, i, index, population, deltas, behavior_state)
        population = np.clip(population + deltas * dt, 0.0, None)

    trajectory = pd.DataFrame(rows)
    summary = {
        "run_id": ruleset.run_id,
        "n_rules": int(len(ruleset.rules)),
        "n_cell_types": int(len(cell_types)),
        "time_steps": int(time_steps),
        "final_total_population": float(trajectory[trajectory["time"] == time_steps - 1]["population"].sum()),
        "max_migration_index": float(trajectory["migration_index"].max()) if not trajectory.empty else 0.0,
        "max_attack_index": float(trajectory["attack_index"].max()) if not trajectory.empty else 0.0,
        "max_exhaustion_index": float(trajectory["exhaustion_index"].max()) if not trajectory.empty else 0.0,
    }
    return trajectory, summary


def compute_qoi_sensitivity(
    ruleset: RuleSet,
    cluster_expression: pd.DataFrame | None = None,
    *,
    time_steps: int = 12,
    dt: float = 1.0,
    delta: float = 0.10,
) -> pd.DataFrame:
    """Perturb each rule saturation and report final-population sensitivity."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    _, baseline = simulate_virtual_tissue(ruleset, cluster_expression, time_steps=time_steps, dt=dt)
    baseline_total = max(float(baseline["final_total_population"]), 1e-12)
    rows = []
    for idx, rule in enumerate(ruleset.rules):
        plus_rules = list(ruleset.rules)
        minus_rules = list(ruleset.rules)
        plus_rules[idx] = _perturb_rule(rule, 1.0 + delta)
        minus_rules[idx] = _perturb_rule(rule, max(0.0, 1.0 - delta))
        _, plus = simulate_virtual_tissue(replace(ruleset, rules=tuple(plus_rules)), cluster_expression, time_steps=time_steps, dt=dt)
        _, minus = simulate_virtual_tissue(replace(ruleset, rules=tuple(minus_rules)), cluster_expression, time_steps=time_steps, dt=dt)
        sensitivity = (float(plus["final_total_population"]) - float(minus["final_total_population"])) / (2.0 * delta * baseline_total)
        rows.append(
            {
                "rule_index": idx + 1,
                "cell_type": rule.cell_type,
                "signal": rule.signal,
                "behavior": rule.behavior,
                "baseline_final_total": baseline_total,
                "plus_final_total": plus["final_total_population"],
                "minus_final_total": minus["final_total_population"],
                "sensitivity_index": float(sensitivity),
            }
        )
    return pd.DataFrame(rows)


def compare_intervention_scenarios(
    ruleset: RuleSet,
    cluster_expression: pd.DataFrame,
    *,
    scenarios: dict[str, dict[str, float]],
    time_steps: int = 12,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Compare simple signal-override intervention scenarios.

    Scenario values are multiplicative signal overrides. A CD137 agonist override
    also adds a CD8 attack-boost rule, matching the Stage 5 sidecar's role as a
    transparent hypothesis sandbox rather than a hidden replacement for Step4.
    """
    rows = []
    for name, overrides in scenarios.items():
        expr = _apply_signal_overrides(cluster_expression, overrides)
        scenario_rules = _rules_for_scenario(ruleset, overrides)
        trajectory, summary = simulate_virtual_tissue(scenario_rules, expr, time_steps=time_steps, dt=dt)
        final = trajectory[trajectory["time"] == time_steps - 1]
        tumor = final[final["cell_type"].astype(str).str.contains("tumor|cancer|epithelial", case=False, regex=True)]
        rows.append(
            {
                "scenario": name,
                "final_total_population": summary["final_total_population"],
                "final_tumor_population": float(tumor["population"].sum()) if not tumor.empty else summary["final_total_population"],
                "max_attack_index": summary["max_attack_index"],
                "max_exhaustion_index": summary["max_exhaustion_index"],
                "n_rules": len(scenario_rules.rules),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def _normalize_expression(cluster_expression: pd.DataFrame | None) -> pd.DataFrame:
    if cluster_expression is None or cluster_expression.empty:
        return pd.DataFrame()
    expr = cluster_expression.copy()
    if "cell_type" in expr.columns:
        expr = expr.set_index("cell_type")
    elif "celltype" in expr.columns:
        expr = expr.set_index("celltype")
    elif "index" in expr.columns:
        expr = expr.set_index("index")
    elif "celltype" not in expr.index.names and expr.index.name is None:
        expr.index = expr.index.astype(str)
    return expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _apply_signal_overrides(cluster_expression: pd.DataFrame, overrides: dict[str, float]) -> pd.DataFrame:
    expr = cluster_expression.copy()
    for signal, multiplier in overrides.items():
        if signal in expr.columns:
            expr[signal] = pd.to_numeric(expr[signal], errors="coerce").fillna(0.0) * float(multiplier)
        else:
            expr[signal] = 0.0
            if "cell_type" in expr.columns:
                mask = expr["cell_type"].astype(str).str.contains("CD8|T", case=False, regex=True)
                expr.loc[mask, signal] = float(multiplier)
    return expr


def _rules_for_scenario(ruleset: RuleSet, overrides: dict[str, float]) -> RuleSet:
    extra_rules: list[BehaviorRule] = []
    cd137 = float(overrides.get("CD137_AGONIST", 0.0))
    if cd137 > 1.0:
        extra_rules.append(
            BehaviorRule(
                cell_type="CD8_T",
                signal="CD137_AGONIST",
                direction="increases",
                behavior="attack",
                response_function="hill",
                parameters=RuleParameters(base=0.05, saturation=min(1.0, 0.35 * cd137), half_max=0.3, hill_power=2.0),
                evidence_refs=("intervention:CD137_AGONIST",),
            )
        )
    if not extra_rules:
        return ruleset
    return replace(ruleset, rules=tuple(list(ruleset.rules) + extra_rules))


def _cell_types(ruleset: RuleSet, expr: pd.DataFrame) -> list[str]:
    values = list(dict.fromkeys(rule.cell_type for rule in ruleset.rules if rule.cell_type))
    for label in expr.index.astype(str).tolist():
        if label not in values:
            values.append(label)
    return values or ["TME_cell"]


def _initial_population(cell_types: list[str], expr: pd.DataFrame) -> np.ndarray:
    values = []
    for cell_type in cell_types:
        if not expr.empty and cell_type in expr.index:
            magnitude = float(np.clip(expr.loc[cell_type].astype(float).mean(), 0.0, None))
            values.append(100.0 * (1.0 + magnitude))
        else:
            values.append(100.0)
    return np.asarray(values, dtype=float)


def _signal_value(expr: pd.DataFrame, cell_type: str, signal: str, fallback_population: float) -> float:
    if not expr.empty and cell_type in expr.index and signal in expr.columns:
        value = float(expr.loc[cell_type, signal])
    elif not expr.empty and signal in expr.columns:
        value = float(expr[signal].astype(float).mean())
    else:
        value = float(fallback_population / 100.0)
    return float(np.clip(value / (1.0 + abs(value)), 0.0, 1.0))


def _apply_rule_effect(
    rule: BehaviorRule,
    effect: float,
    i: int,
    index: dict[str, int],
    population: np.ndarray,
    deltas: np.ndarray,
    behavior_state: dict[str, np.ndarray],
) -> None:
    behavior = rule.behavior.lower()
    source_population = population[i]
    if behavior in {"proliferation", "cycle entry"}:
        deltas[i] += 0.05 * effect * source_population
    elif behavior in {"death", "apoptosis", "necrosis"}:
        deltas[i] -= 0.04 * effect * source_population
    elif behavior in {"migration", "motility"}:
        behavior_state["migration"][i] += 0.10 * effect
        deltas[i] += 0.005 * effect * source_population
    elif behavior == "attack":
        behavior_state["attack"][i] += 0.10 * effect
        tumor_idx = _find_tumor_index(index)
        if tumor_idx is not None:
            deltas[tumor_idx] -= 0.03 * effect * source_population
    elif behavior == "exhaustion":
        behavior_state["exhaustion"][i] += 0.10 * effect
        deltas[i] -= 0.01 * effect * source_population
    elif behavior == "secretion":
        behavior_state["secretion"][i] += 0.10 * effect
    elif behavior == "transition":
        behavior_state["transition"][i] += 0.10 * effect
        deltas[i] += 0.002 * effect * source_population


def _find_tumor_index(index: dict[str, int]) -> int | None:
    for label, idx in index.items():
        if "tumor" in label.lower() or "cancer" in label.lower() or "epithelial" in label.lower():
            return idx
    return None


def _append_rows(
    rows: list[dict[str, Any]],
    timepoint: int,
    cell_types: list[str],
    population: np.ndarray,
    behavior_state: dict[str, np.ndarray],
) -> None:
    for i, cell_type in enumerate(cell_types):
        rows.append(
            {
                "time": int(timepoint),
                "cell_type": cell_type,
                "population": float(population[i]),
                "migration_index": float(behavior_state["migration"][i]),
                "attack_index": float(behavior_state["attack"][i]),
                "exhaustion_index": float(behavior_state["exhaustion"][i]),
                "secretion_index": float(behavior_state["secretion"][i]),
                "transition_index": float(behavior_state["transition"][i]),
            }
        )


def _perturb_rule(rule: BehaviorRule, multiplier: float) -> BehaviorRule:
    params = RuleParameters(
        base=rule.parameters.base,
        saturation=rule.parameters.saturation * multiplier,
        half_max=rule.parameters.half_max,
        hill_power=rule.parameters.hill_power,
    )
    return replace(rule, parameters=params)
