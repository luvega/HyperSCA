"""Human-readable cell behavior grammar primitives."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


_VALID_DIRECTIONS = {"increases", "decreases"}
_VALID_RESPONSES = {"hill", "linear", "step"}


@dataclass(frozen=True)
class RuleParameters:
    """Numeric response parameters for one behavior rule."""

    base: float = 0.0
    saturation: float = 1.0
    half_max: float = 0.5
    hill_power: float = 1.0

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"parameter {name} must be finite")
        if self.half_max <= 0:
            raise ValueError("half_max must be positive")
        if self.hill_power <= 0:
            raise ValueError("hill_power must be positive")


@dataclass(frozen=True)
class SignalDictionary:
    """Controlled signal vocabulary for grammar validation."""

    categories: Mapping[str, tuple[str, ...]]

    @classmethod
    def default(cls, extra_signals: Iterable[str] = ()) -> "SignalDictionary":
        values = {
            "ligand": ("TGFB1", "CXCL9", "CXCL10", "IFNG", "EGF", "IL4", "IL10", "PDCD1", "CD274"),
            "cytokine": ("IFNG", "IL4", "IL10", "TNF", "TGFB1"),
            "ecm": ("ECM", "COLLAGEN", "COL1A1", "FN1", "POSTN"),
            "mechanical": ("PRESSURE", "OXYGEN", "DAMAGE"),
            "drug": ("PD1_BLOCKADE", "CD137_AGONIST", "GVAX", "DRUG"),
            "time": ("TIME",),
            "niche": ("NICHE", "STROMAL_INVASION", "INFLAMED_EDGE", "IMMUNE_CHECKPOINT"),
        }
        merged = {key: tuple(dict.fromkeys(items)) for key, items in values.items()}
        extras = tuple(str(signal).strip() for signal in extra_signals if str(signal).strip())
        if extras:
            merged["data_driven"] = extras
        return cls(merged)

    @property
    def all_signals(self) -> set[str]:
        return {signal.upper() for values in self.categories.values() for signal in values}

    def contains(self, signal: str) -> bool:
        return str(signal).upper() in self.all_signals


@dataclass(frozen=True)
class BehaviorDictionary:
    """Controlled cell-behavior vocabulary for grammar validation."""

    behaviors: tuple[str, ...]

    @classmethod
    def default(cls) -> "BehaviorDictionary":
        return cls(
            (
                "proliferation",
                "cycle entry",
                "death",
                "apoptosis",
                "necrosis",
                "migration",
                "motility",
                "secretion",
                "transition",
                "attack",
                "exhaustion",
                "phagocytosis",
            )
        )

    def contains(self, behavior: str) -> bool:
        return str(behavior).lower() in {item.lower() for item in self.behaviors}


@dataclass(frozen=True)
class BehaviorRule:
    """One computable hypothesis statement linking signal to cell behavior."""

    cell_type: str
    signal: str
    direction: str
    behavior: str
    response_function: str
    parameters: RuleParameters = field(default_factory=RuleParameters)
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_statement(
        cls,
        statement: str,
        *,
        evidence_refs: Iterable[str] = (),
    ) -> "BehaviorRule":
        pattern = re.compile(
            r"^In\s+(?P<cell_type>.+?)\s+cells?,\s+"
            r"(?P<signal>.+?)\s+"
            r"(?P<direction>increases|decreases)\s+"
            r"(?P<behavior>.+?)\s+with\s+a\s+"
            r"(?P<response>hill|linear|step)\s+response"
            r"(?P<params>.*)$",
            flags=re.IGNORECASE,
        )
        match = pattern.match(statement.strip().rstrip("."))
        if not match:
            raise ValueError(f"could not parse behavior rule statement: {statement!r}")

        params_text = match.group("params")
        return cls(
            cell_type=match.group("cell_type").strip(),
            signal=match.group("signal").strip(),
            direction=match.group("direction").lower(),
            behavior=match.group("behavior").strip(),
            response_function=match.group("response").lower(),
            parameters=_parse_parameters(params_text),
            evidence_refs=tuple(evidence_refs),
        )

    def validate(
        self,
        *,
        signals: SignalDictionary | None = None,
        behaviors: BehaviorDictionary | None = None,
    ) -> None:
        signals = SignalDictionary.default() if signals is None else signals
        behaviors = BehaviorDictionary.default() if behaviors is None else behaviors
        if not self.cell_type:
            raise ValueError("cell_type is required")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_VALID_DIRECTIONS)}")
        if self.response_function not in _VALID_RESPONSES:
            raise ValueError(f"response_function must be one of {sorted(_VALID_RESPONSES)}")
        if not signals.contains(self.signal):
            raise ValueError(f"unknown signal: {self.signal}")
        if not behaviors.contains(self.behavior):
            raise ValueError(f"unknown behavior: {self.behavior}")
        self.parameters.validate()

    def to_statement(self) -> str:
        return (
            f"In {self.cell_type} cells, {self.signal} {self.direction} {self.behavior} "
            f"with a {self.response_function} response, base {self.parameters.base:g}, "
            f"max {self.parameters.saturation:g}, half-max {self.parameters.half_max:g}, "
            f"hill {self.parameters.hill_power:g}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_type": self.cell_type,
            "signal": self.signal,
            "direction": self.direction,
            "behavior": self.behavior,
            "response_function": self.response_function,
            "parameters": asdict(self.parameters),
            "evidence_refs": list(self.evidence_refs),
            "statement": self.to_statement(),
        }


@dataclass(frozen=True)
class RuleSet:
    """Run-scoped collection of behavior grammar rules."""

    run_id: str
    rules: tuple[BehaviorRule, ...]
    source_manifest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        signals = SignalDictionary.default(extra_signals=[rule.signal for rule in self.rules])
        behaviors = BehaviorDictionary.default()
        for rule in self.rules:
            rule.validate(signals=signals, behaviors=behaviors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_manifest": self.source_manifest,
            "metadata": dict(self.metadata),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Behavior Grammar Rules: {self.run_id}",
            "",
            "These rules are generated from HyperSCA evidence tables and remain editable by a domain expert.",
            "",
            "| # | Cell type | Signal | Direction | Behavior | Response | Evidence |",
            "|---:|---|---|---|---|---|---|",
        ]
        for idx, rule in enumerate(self.rules, start=1):
            evidence = "; ".join(rule.evidence_refs)
            lines.append(
                f"| {idx} | {rule.cell_type} | {rule.signal} | {rule.direction} | "
                f"{rule.behavior} | {rule.response_function} | {evidence} |"
            )
        lines.extend(["", "## Statements", ""])
        lines.extend(f"- {rule.to_statement()}" for rule in self.rules)
        return "\n".join(lines) + "\n"


def _parse_parameters(text: str) -> RuleParameters:
    values = {
        "base": 0.0,
        "saturation": 1.0,
        "half_max": 0.5,
        "hill_power": 1.0,
    }
    aliases = {
        "base": "base",
        "max": "saturation",
        "saturation": "saturation",
        "half-max": "half_max",
        "half max": "half_max",
        "half_max": "half_max",
        "hill": "hill_power",
        "hill power": "hill_power",
        "hill_power": "hill_power",
    }
    for label, number in re.findall(r"(base|max|saturation|half[-_ ]max|hill(?:[_ ]power)?)\s+([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", text, flags=re.I):
        values[aliases[label.lower().replace("_", " ")]] = float(number)
    return RuleParameters(**values)


def evaluate_response(
    signal_values: Iterable[float],
    parameters: RuleParameters,
    *,
    response_function: str,
    direction: str,
) -> np.ndarray:
    """Evaluate one response curve for signal values."""
    parameters.validate()
    if response_function not in _VALID_RESPONSES:
        raise ValueError(f"response_function must be one of {sorted(_VALID_RESPONSES)}")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(_VALID_DIRECTIONS)}")

    values = np.clip(np.asarray(list(signal_values), dtype=float), 0.0, None)
    if response_function == "hill":
        numerator = np.power(values, parameters.hill_power)
        denominator = np.power(parameters.half_max, parameters.hill_power) + numerator + 1e-12
        response = numerator / denominator
    elif response_function == "linear":
        response = np.clip(values / parameters.half_max, 0.0, 1.0)
    else:
        response = (values < parameters.half_max).astype(float) if direction == "decreases" else (values >= parameters.half_max).astype(float)

    if direction == "increases":
        return parameters.base + (parameters.saturation - parameters.base) * response
    if response_function == "step":
        return parameters.base + (parameters.saturation - parameters.base) * response
    return parameters.saturation - (parameters.saturation - parameters.base) * response
