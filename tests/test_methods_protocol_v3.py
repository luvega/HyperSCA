from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from src.evaluation.methods_protocol_v3 import (
    MethodsProtocolV3,
    build_methods_protocol_v3,
    protocol_identity_v3,
    protocol_to_mapping_v3,
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


def test_v3_protocol_has_three_claim_families() -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="pilot_audit_only",
        capability_identity_sha256="a" * 64,
    )

    assert protocol.protocol_version == "hypersca-methods-v3.0"
    assert protocol.claim_ids == ("spatial", "intracellular_causal", "bridge")
    assert protocol.primary_metrics == (
        "neighborhood_preservation_at_k",
        "directed_edge_average_precision",
        "neighbor_effect_rmse",
    )
    assert protocol.multiple_testing == "distinct_families_no_cross_adjustment"
    assert protocol.integrated_gate == "intersection_union_all_three"
    assert protocol.integrated_claim_enabled is False


def test_confirmatory_bridge_enables_integrated_claim_only_when_all_gates_pass() -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="confirmatory", capability_identity_sha256="b" * 64
    )

    assert protocol.integrated_claim_enabled is True
    assert protocol.crc_role == "application_only"


def test_v3_mapping_encodes_fixed_comparators_and_bridge_iut_in_order() -> None:
    payload = protocol_to_mapping_v3(
        build_methods_protocol_v3(
            bridge_role="confirmatory", capability_identity_sha256="c" * 64
        )
    )

    assert tuple(payload) == (
        "schema",
        "protocol_version",
        "claims",
        "statistics",
        "integration",
        "governance",
        "capability_identity_sha256",
    )
    claims = cast(Mapping[str, Mapping[str, object]], payload["claims"])
    spatial_claim = claims["spatial"]
    causal_claim = claims["intracellular_causal"]
    bridge_claim = claims["bridge"]

    assert tuple(claims) == ("spatial", "intracellular_causal", "bridge")
    assert spatial_claim["comparators"] == {
        "confirmatory": "matched_euclidean_autoencoder",
        "attribution": "hypersca_without_hierarchy_loss",
    }
    assert causal_claim["benchmark"] == (
        "causalbench_intracellular_interventional_causal_recovery"
    )
    assert causal_claim["comparators"] == {
        "confirmatory": "matched_non_hyperbolic_baseline",
        "attribution": "hypersca_c_shared_only",
    }
    assert bridge_claim["iut"] == {
        "comparators": [
            "matched_euclidean_spatial_causal",
            "hypersca_own_only",
        ],
        "within_iut_multiplicity_adjustment": "none",
        "claim_p_value": "max_component_p",
    }
    assert "holm" not in repr(payload).lower()
    assert "bonferroni" not in repr(payload).lower()
    assert "fdr" not in repr(payload).lower()


def test_direct_construction_freezes_plain_list_inputs() -> None:
    """Characterization: accepted plain lists are copied before caller mutation."""
    kwargs = _valid_kwargs()
    claim_ids = ["spatial", "intracellular_causal", "bridge"]
    kwargs["claim_ids"] = claim_ids

    protocol = MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]
    claim_ids[0] = "changed"

    assert protocol.claim_ids == ("spatial", "intracellular_causal", "bridge")
    with pytest.raises(FrozenInstanceError):
        protocol.bridge_role = "confirmatory"  # type: ignore[misc]


def test_direct_construction_rejects_invalid_integrated_enablement() -> None:
    kwargs = _valid_kwargs()
    kwargs["integrated_claim_enabled"] = True

    with pytest.raises(ValueError, match="integrated_claim_enabled"):
        MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


def test_identity_is_canonical_for_equal_valid_protocols() -> None:
    first = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256="d" * 64
    )
    kwargs = _valid_kwargs()
    kwargs["capability_identity_sha256"] = "d" * 64
    second = MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]

    assert protocol_identity_v3(first) == protocol_identity_v3(second)
    assert len(protocol_identity_v3(first)) == 64
