from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from src.evaluation.methods_protocol_v3 import (
    MethodsProtocolV3,
    build_methods_protocol_v3,
    protocol_identity_v3,
)


def _valid_kwargs() -> dict[str, object]:
    return {
        "protocol_version": "hypersca-methods-v3.0",
        "claim_ids": ("spatial", "intracellular_causal", "bridge"),
        "primary_metrics": (
            "neighborhood_preservation_at_k",
            "directed_edge_average_precision",
            "neighbor_effect_rmse",
        ),
        "pilot_seeds": (11, 23, 47),
        "release_seeds": (11, 23, 47, 71, 101),
        "bootstrap_resamples": 10_000,
        "confidence": 0.95,
        "multiple_testing": "distinct_families_no_cross_adjustment",
        "integrated_gate": "intersection_union_all_three",
        "bridge_role": "pilot_audit_only",
        "capability_identity_sha256": "a" * 64,
        "integrated_claim_enabled": False,
        "crc_role": "application_only",
    }


class _IntSubclass(int):
    pass


class _ListSubclass(list[object]):
    pass


@settings(max_examples=20, deadline=None)
@given(st.booleans())
def test_direct_construction_rejects_bool_as_integer(value: bool) -> None:
    kwargs = _valid_kwargs()
    kwargs["bootstrap_resamples"] = value

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=20, deadline=None)
@given(st.sampled_from((11, 23, 47, 71, 101)))
def test_direct_construction_rejects_integer_subclasses(value: int) -> None:
    kwargs = _valid_kwargs()
    kwargs["pilot_seeds"] = (_IntSubclass(value), 23, 47)

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=20, deadline=None)
@given(st.just(_ListSubclass(["spatial", "intracellular_causal", "bridge"])))
def test_direct_construction_rejects_mutable_sequence_subclasses(value: list[object]) -> None:
    kwargs = _valid_kwargs()
    kwargs["claim_ids"] = value

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=20, deadline=None)
@given(st.sampled_from(("\u0000", "\u200b", "e\u0301")))
def test_direct_construction_rejects_unsafe_text(value: str) -> None:
    kwargs = _valid_kwargs()
    kwargs["protocol_version"] = value

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=20, deadline=None)
@given(st.sampled_from((math.inf, -math.inf, math.nan)))
def test_direct_construction_rejects_non_finite_floats(value: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["confidence"] = value

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=20, deadline=None)
@given(st.sampled_from(("spatial", "intracellular_causal", "bridge")))
def test_direct_construction_rejects_duplicate_claim_ids(value: str) -> None:
    kwargs = _valid_kwargs()
    kwargs["claim_ids"] = (value, value, "bridge")

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=20, deadline=None)
@given(st.sampled_from(("A" * 64, "a" * 63, "g" * 64, "a" * 63 + "\n")))
def test_direct_construction_rejects_malformed_capability_identity(value: str) -> None:
    kwargs = _valid_kwargs()
    kwargs["capability_identity_sha256"] = value

    with pytest.raises(ValueError):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


@settings(max_examples=30, deadline=None)
@given(st.sampled_from(("a", "b", "c", "d", "e", "f")))
def test_equal_valid_inputs_have_identical_canonical_identity(digest_character: str) -> None:
    digest = digest_character * 64

    first = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256=digest
    )
    second = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256=digest
    )

    assert protocol_identity_v3(first) == protocol_identity_v3(second)
