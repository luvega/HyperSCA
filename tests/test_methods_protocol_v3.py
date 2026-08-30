from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import re
from typing import cast

import pytest

import src.evaluation.methods_protocol_v3 as methods_protocol_v3
from src.evaluation.methods_protocol_v3 import (
    MethodsProtocolV3,
    build_methods_protocol_v3,
    protocol_identity_v3,
    protocol_to_mapping_v3,
)
from src.evaluation.spatial_perturbation_registry import (
    audit_bridge_capability,
    load_bridge_candidates,
    unavailable_metadata_summary,
)
from src.evaluation import spatial_perturbation_registry as spatial_registry


ROOT = Path(__file__).resolve().parents[1]


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


def _expected_mapping(
    *, bridge_role: str, integrated_claim_enabled: bool, capability_identity: str
) -> dict[str, object]:
    return {
        "schema": "hypersca_methods_protocol_v3",
        "protocol_version": "hypersca-methods-v3.0",
        "bridge_role": bridge_role,
        "claims": {
            "spatial": {
                "claim_id": "spatial",
                "benchmark": "osta_colon",
                "primary_metric": "neighborhood_preservation_at_k",
                "primary_k": 15,
                "comparators": {
                    "confirmatory": "matched_euclidean_autoencoder",
                    "attribution": "hypersca_without_hierarchy_loss",
                },
            },
            "intracellular_causal": {
                "claim_id": "intracellular_causal",
                "benchmark": "causalbench_intracellular_interventional_causal_recovery",
                "primary_metric": "directed_edge_average_precision",
                "comparators": {
                    "confirmatory": "matched_non_hyperbolic_baseline",
                    "attribution": "hypersca_c_shared_only",
                },
            },
            "bridge": {
                "claim_id": "bridge",
                "primary_metric": "neighbor_effect_rmse",
                "primary_bands": [
                    {"name": "proximal", "minimum_distance": 1, "maximum_distance": 5},
                    {"name": "local", "minimum_distance": 6, "maximum_distance": 15},
                ],
                "secondary_bands": [
                    {
                        "name": "transition",
                        "minimum_distance": 16,
                        "maximum_distance": 30,
                    },
                    {"name": "distal", "minimum_distance": 31, "maximum_distance": 60},
                ],
                "iut": {
                    "comparators": [
                        "matched_euclidean_spatial_causal",
                        "hypersca_own_only",
                    ],
                    "within_iut_multiplicity_adjustment": "none",
                    "claim_p_value": "max_component_p",
                },
            },
        },
        "statistics": {
            "pilot_seeds": [11, 23, 47],
            "release_seeds": [11, 23, 47, 71, 101],
            "bootstrap_resamples": 10_000,
            "confidence": 0.95,
            "family_decision_rule": "nominal_one_sided_paired_95_percent_ci_lower_bound_gt_zero",
            "cross_family_adjustment": "none",
            "multiple_testing": "distinct_families_no_cross_adjustment",
        },
        "integration": {
            "gate": "intersection_union_all_three",
            "integrated_claim_enabled": integrated_claim_enabled,
            "evidence_policy": "all_three_family_gates_required",
            "extra_p_value": "none",
        },
        "governance": {
            "crc_role": "application_only",
            "crc_may_tune_hyperparameters": False,
            "crc_may_select_baselines": False,
            "crc_may_change_thresholds": False,
            "crc_may_rescue_promotion": False,
        },
        "capability_identity_sha256": capability_identity,
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


def test_confirmatory_bridge_marks_integrated_claim_as_eligible() -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="confirmatory", capability_identity_sha256="b" * 64
    )

    assert protocol.integrated_claim_enabled is True
    assert protocol.crc_role == "application_only"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("integrated_claim_enabled", True),
        ("claim_ids", ("spatial", "bridge", "intracellular_causal")),
        (
            "primary_metrics",
            (
                "neighborhood_preservation_at_k",
                "directed_edge_average_precision",
                "different_metric",
            ),
        ),
        ("confidence", math.nan),
        ("bridge_role", "confirmatory"),
    ),
)
def test_serializers_revalidate_object_setattr_mutations(
    field: str, invalid_value: object
) -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256="a" * 64
    )
    object.__setattr__(protocol, field, invalid_value)

    with pytest.raises(ValueError):
        protocol_to_mapping_v3(protocol)
    with pytest.raises(ValueError):
        protocol_identity_v3(protocol)


def test_protocol_mapping_is_independent_of_rebound_exported_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256="a" * 64
    )
    expected_mapping = protocol_to_mapping_v3(protocol)
    expected_identity = protocol_identity_v3(protocol)

    monkeypatch.setattr(
        methods_protocol_v3, "BRIDGE_PRIMARY_BANDS", (("tampered", 99, 100),)
    )

    assert protocol_to_mapping_v3(protocol) == expected_mapping
    assert protocol_identity_v3(protocol) == expected_identity


@pytest.mark.parametrize(
    ("attribute", "forged_value", "forged_field"),
    (
        ("BRIDGE_PRIMARY_BANDS", (("forged", 99, 100),), None),
        ("BRIDGE_SECONDARY_BANDS", (("forged", 99, 100),), None),
        ("_BRIDGE_PRIMARY_BANDS", (("forged", 99, 100),), None),
        ("_BRIDGE_SECONDARY_BANDS", (("forged", 99, 100),), None),
        (
            "_CLAIM_IDS",
            ("forged_spatial", "forged_causal", "forged_bridge"),
            "claim_ids",
        ),
        (
            "_PRIMARY_METRICS",
            ("forged_spatial", "forged_causal", "forged_bridge"),
            "primary_metrics",
        ),
        ("_PILOT_SEEDS", (3, 5, 7), "pilot_seeds"),
        ("_RELEASE_SEEDS", (3, 5, 7, 11, 13), "release_seeds"),
    ),
)
def test_rebinding_scientific_constants_cannot_forge_v3_protocol(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    forged_value: object,
    forged_field: str | None,
) -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256="a" * 64
    )
    expected_mapping = protocol_to_mapping_v3(protocol)
    expected_identity = protocol_identity_v3(protocol)
    monkeypatch.setattr(
        methods_protocol_v3, attribute, forged_value, raising=False
    )

    assert protocol_to_mapping_v3(protocol) == expected_mapping
    assert protocol_identity_v3(protocol) == expected_identity

    rebuilt = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256="a" * 64
    )
    assert protocol_to_mapping_v3(rebuilt) == expected_mapping
    assert protocol_identity_v3(rebuilt) == expected_identity

    if forged_field is not None:
        kwargs = _valid_kwargs()
        kwargs[forged_field] = forged_value
        with pytest.raises(ValueError):
            MethodsProtocolV3(**kwargs)  # type: ignore[arg-type]


def test_mapping_freezes_spatial_k_and_explicit_bridge_role_in_identity() -> None:
    pilot = build_methods_protocol_v3(
        bridge_role="pilot_audit_only", capability_identity_sha256="a" * 64
    )
    confirmatory = build_methods_protocol_v3(
        bridge_role="confirmatory", capability_identity_sha256="a" * 64
    )
    pilot_mapping = protocol_to_mapping_v3(pilot)
    claims = cast(Mapping[str, Mapping[str, object]], pilot_mapping["claims"])

    assert claims["spatial"]["primary_k"] == 15
    assert pilot_mapping["bridge_role"] == "pilot_audit_only"
    assert protocol_identity_v3(pilot) != protocol_identity_v3(confirmatory)


def test_v3_mapping_encodes_fixed_comparators_and_bridge_iut_in_order() -> None:
    payload = protocol_to_mapping_v3(
        build_methods_protocol_v3(
            bridge_role="confirmatory", capability_identity_sha256="c" * 64
        )
    )

    assert tuple(payload) == (
        "schema",
        "protocol_version",
        "bridge_role",
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


@pytest.mark.parametrize(
    ("bridge_role", "enabled", "digest", "golden_identity"),
    (
        (
            "pilot_audit_only",
            False,
            "a" * 64,
            "c680d8fdf076c5af8ba865753c28fec2a9175a1a3cf94aa87567143930612c3c",
        ),
        (
            "confirmatory",
            True,
            "b" * 64,
            "b58dac79d0f812003445210ca29d1423a5fef6d89cd3d2c89f3707039030e4c5",
        ),
    ),
)
def test_complete_mapping_matches_role_specific_golden(
    bridge_role: str, enabled: bool, digest: str, golden_identity: str
) -> None:
    protocol = build_methods_protocol_v3(
        bridge_role=bridge_role, capability_identity_sha256=digest
    )
    expected = _expected_mapping(
        bridge_role=bridge_role,
        integrated_claim_enabled=enabled,
        capability_identity=digest,
    )

    assert protocol_to_mapping_v3(protocol) == expected
    assert json.dumps(protocol_to_mapping_v3(protocol), separators=(",", ":")) == (
        json.dumps(expected, separators=(",", ":"))
    )
    assert protocol_identity_v3(protocol) == golden_identity


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


def test_task12_missing_external_cohort_stops_before_protocol_freeze_and_pilot() -> None:
    preflight = ROOT / "reports" / "methods_protocol_v3_preflight"
    capability_path = preflight / "bridge_capability.json"
    review_path = preflight / "review.md"

    assert {path.name for path in preflight.iterdir()} == {
        "bridge_capability.json",
        "review.md",
    }
    capability_bytes = capability_path.read_bytes()
    capability = json.loads(capability_bytes)
    assert set(capability) == {
        "candidate_id",
        "status",
        "confirmatory_capable",
        "biological_specimen_count",
        "cohort_count",
        "coverage",
        "blocking_reasons",
        "capability_identity_sha256",
    }
    assert capability["candidate_id"] == "gse274447_msafe_bridge"
    assert capability["status"] == "assets_unavailable"
    assert capability["confirmatory_capable"] is False
    assert "external_cohort_missing" in capability["blocking_reasons"]
    forbidden_scientific_fields = {
        "effect",
        "effects",
        "metric",
        "metrics",
        "prediction",
        "predictions",
        "rmse",
    }
    assert forbidden_scientific_fields.isdisjoint(capability)

    parsed = spatial_registry.bridge_capability_result_from_mapping(capability)
    candidate = load_bridge_candidates(
        ROOT / "configs" / "spatial_perturbation_bridge_candidates_v1.json"
    )["gse274447_msafe_bridge"]
    recomputed = audit_bridge_capability(
        candidate, unavailable_metadata_summary(candidate)
    )
    assert parsed.to_mapping() == recomputed.to_mapping()
    assert (
        parsed.capability_identity_sha256
        == recomputed.capability_identity_sha256
    )
    canonical_bytes = (
        json.dumps(
            recomputed.to_mapping(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    assert capability_bytes == canonical_bytes

    review = " ".join(review_path.read_text(encoding="utf-8").split())
    for required_statement in (
        "pilot_failed_no_release",
        "external_cohort_missing",
        "assets_unavailable",
        "No v3 protocol config was frozen",
        "No predictor capability audit was run",
        "No real bridge pilot was run",
        "no paired scientific collection exists",
        "integrated_claim_enabled=false",
        "separate preregistered design and protocol identity",
    ):
        assert required_statement in review

    assert not (ROOT / "configs" / "hypersca_methods_v3.yaml").exists()
    assert not (preflight / "predictor_capability").exists()
    assert "$SPATIAL_PERTURB_ROOT" in review
    assert re.search(r"/(?:home|Users)/[^/\s`]+/", review) is None
