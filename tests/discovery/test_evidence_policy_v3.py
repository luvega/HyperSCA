from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.discovery.evidence_policy import (
    EvidencePolicyV3,
    V3ClaimDecision,
    V3ClaimEvidence,
    derive_integrated_claim,
    evaluate_bridge_claim,
    evaluate_v3_claim,
)


IDENTITY = "a" * 64


def policy_v3(**overrides: object) -> EvidencePolicyV3:
    payload = {
        "protocol_version": "hypersca-methods-v3.0",
        "family_primary_metrics": (
            ("spatial", "neighborhood_preservation_at_k"),
            ("intracellular_causal", "directed_edge_average_precision"),
            ("bridge", "neighbor_effect_rmse"),
        ),
        "required_comparators": (
            (
                "spatial",
                (
                    "matched_euclidean_autoencoder",
                    "hypersca_without_hierarchy_loss",
                ),
            ),
            (
                "intracellular_causal",
                ("matched_non_hyperbolic_causal", "hypersca_c_shared_only"),
            ),
            (
                "bridge",
                ("matched_euclidean_spatial_causal", "hypersca_own_only"),
            ),
        ),
        "nominal_alpha": 0.05,
        "minimum_lower_bound": 0.0,
        "bridge_role": "confirmatory",
        "integrated_claim_enabled": True,
    }
    payload.update(overrides)
    return EvidencePolicyV3(**payload)  # type: ignore[arg-type]


def evidence(
    claim_id: str,
    comparator_id: str,
    *,
    role: str = "confirmatory",
    ci_low: float | None = 0.01,
    p_value: float | None = 0.01,
    attempted: int = 3,
    completed: int = 3,
    identity: str = IDENTITY,
) -> V3ClaimEvidence:
    is_complete = attempted == completed
    return V3ClaimEvidence(
        claim_id=claim_id,
        protocol_version="hypersca-methods-v3.0",
        primary_metric=dict(policy_v3().family_primary_metrics)[claim_id],
        comparator_id=comparator_id,
        paired_estimate=0.04 if is_complete else None,
        ci_low=ci_low if is_complete else None,
        ci_high=0.08 if is_complete else None,
        nominal_p_value=p_value if is_complete else None,
        attempted_units=attempted,
        completed_units=completed,
        evidence_role=role,
        artifact_identity=identity,
    )


def pair(claim_id: str, *, ci_low: float = 0.01) -> tuple[V3ClaimEvidence, ...]:
    requirements = dict(policy_v3().required_comparators)[claim_id]
    roles = (
        ("confirmatory", "attribution")
        if claim_id != "bridge"
        else ("confirmatory", "confirmatory")
    )
    return tuple(
        evidence(
            claim_id, comparator, role=role, ci_low=ci_low, identity=character * 64
        )
        for comparator, role, character in zip(requirements, roles, "ab")
    )


def bridge_pair(first_p: float, second_p: float) -> tuple[V3ClaimEvidence, ...]:
    items = pair("bridge")
    return (
        evidence("bridge", items[0].comparator_id, p_value=first_p, identity="c" * 64),
        evidence("bridge", items[1].comparator_id, p_value=second_p, identity="d" * 64),
    )


def v3_decision(
    claim_id: str, status: str, *, role: str = "confirmatory"
) -> V3ClaimDecision:
    identities = {"spatial": "a", "intracellular_causal": "b", "bridge": "c"}
    allowed_uses = {
        "spatial": "spatial_representation_preservation_gain",
        "intracellular_causal": "intracellular_intervention_directed_relation_recovery",
        "bridge": "spatial_neighbour_response_recovery",
    }
    return V3ClaimDecision(
        claim_id=claim_id,
        protocol_version="hypersca-methods-v3.0",
        status=status,
        allowed_use=allowed_uses[claim_id] if status == "admitted" else "audit_only",
        blocking_reasons=(),
        evidence_identity=identities[claim_id] * 64,
        nominal_p_value=0.01 if status == "admitted" else None,
        multiplicity_adjustment=(
            "none_intersection_union"
            if claim_id == "bridge"
            else "none_family_specific"
        ),
        evidence_role=role,
        application_evidence_identities=(),
    )


def test_integrated_claim_requires_all_three_components() -> None:
    decisions = (
        v3_decision("spatial", "admitted"),
        v3_decision("intracellular_causal", "admitted"),
        v3_decision("bridge", "audit_only"),
    )
    result = derive_integrated_claim(decisions, policy_v3())
    assert result.status == "audit_only"
    assert result.allowed_use == "separate_module_claims_only"


def test_bridge_uses_intersection_union_without_holm() -> None:
    decision = evaluate_bridge_claim(bridge_pair(0.01, 0.04), policy_v3())
    assert decision.nominal_p_value == 0.04
    assert decision.multiplicity_adjustment == "none_intersection_union"


def test_crc_cannot_rescue_a_missing_bridge() -> None:
    crc = evidence("bridge", "crc", role="application_only", p_value=0.0001)
    decision = evaluate_bridge_claim((crc,), policy_v3())
    assert decision.status == "blocked"
    assert decision.allowed_use == "not_used"
    assert decision.application_evidence_identities == (crc.artifact_identity,)


@pytest.mark.parametrize(
    ("spatial_status", "causal_status", "bridge_status", "expected"),
    [
        ("admitted", "admitted", "admitted", "admitted"),
        ("audit_only", "admitted", "admitted", "audit_only"),
        ("admitted", "audit_only", "admitted", "audit_only"),
        ("admitted", "admitted", "audit_only", "audit_only"),
        ("audit_only", "audit_only", "admitted", "audit_only"),
        ("audit_only", "admitted", "audit_only", "audit_only"),
        ("admitted", "audit_only", "audit_only", "audit_only"),
        ("audit_only", "audit_only", "audit_only", "audit_only"),
        ("blocked", "admitted", "admitted", "blocked"),
    ],
)
def test_integrated_claim_truth_table(
    spatial_status: str, causal_status: str, bridge_status: str, expected: str
) -> None:
    result = derive_integrated_claim(
        (
            v3_decision("spatial", spatial_status),
            v3_decision("intracellular_causal", causal_status),
            v3_decision("bridge", bridge_status),
        ),
        policy_v3(),
    )
    assert result.status == expected


def test_bridge_requires_two_completed_confirmatory_comparators() -> None:
    first, second = bridge_pair(0.01, 0.01)
    incomplete = evidence(
        "bridge", second.comparator_id, attempted=3, completed=2, identity="e" * 64
    )
    assert evaluate_bridge_claim((first, incomplete), policy_v3()).status == "blocked"
    assert evaluate_bridge_claim((first,), policy_v3()).status == "blocked"


def test_bridge_requires_common_paired_units() -> None:
    first, second = bridge_pair(0.01, 0.01)
    unequal = evidence(
        "bridge", second.comparator_id, attempted=4, completed=4, identity="e" * 64
    )
    assert evaluate_bridge_claim((first, unequal), policy_v3()).status == "blocked"


def test_v3_claim_enforces_strict_positive_ci_and_roles() -> None:
    assert (
        evaluate_v3_claim(pair("spatial", ci_low=0.0), policy_v3()).status
        == "audit_only"
    )
    wrong_roles = tuple(
        evidence(
            "spatial",
            item.comparator_id,
            role="confirmatory",
            identity=item.artifact_identity,
        )
        for item in pair("spatial")
    )
    assert evaluate_v3_claim(wrong_roles, policy_v3()).status == "audit_only"


@pytest.mark.parametrize(
    "role", ["pilot_audit_only", "synthetic_audit_only", "audit_only"]
)
def test_audit_evidence_cannot_be_promoted(role: str) -> None:
    first, second = pair("bridge")
    audit_evidence = evidence(
        "bridge", second.comparator_id, role=role, identity="f" * 64
    )
    assert (
        evaluate_bridge_claim((first, audit_evidence), policy_v3()).status
        == "audit_only"
    )


def test_crc_does_not_change_an_admitted_bridge() -> None:
    admitted = evaluate_bridge_claim(bridge_pair(0.01, 0.01), policy_v3())
    crc = evidence(
        "bridge", "crc", role="application_only", p_value=0.0001, identity="e" * 64
    )
    with_crc = evaluate_bridge_claim((*bridge_pair(0.01, 0.01), crc), policy_v3())
    assert with_crc.status == admitted.status == "admitted"
    assert with_crc.nominal_p_value == admitted.nominal_p_value
    assert with_crc.application_evidence_identities == (crc.artifact_identity,)


def test_integrated_claim_rejects_duplicate_extra_and_protocol_mismatched_decisions() -> (
    None
):
    spatial = v3_decision("spatial", "admitted")
    causal = v3_decision("intracellular_causal", "admitted")
    bridge = v3_decision("bridge", "admitted")
    with pytest.raises(ValueError, match="exactly one"):
        derive_integrated_claim((spatial, causal, bridge, bridge), policy_v3())
    tampered = object.__new__(V3ClaimDecision)
    for field in V3ClaimDecision.__dataclass_fields__:
        object.__setattr__(tampered, field, getattr(bridge, field))
    object.__setattr__(tampered, "protocol_version", "wrong")
    with pytest.raises(ValueError, match="protocol"):
        derive_integrated_claim((spatial, causal, tampered), policy_v3())


def test_integrated_claim_requires_exact_claim_wording_and_enabled_policy() -> None:
    spatial = v3_decision("spatial", "admitted")
    causal = v3_decision("intracellular_causal", "admitted")
    bridge = v3_decision("bridge", "admitted")
    wrong_wording = object.__new__(V3ClaimDecision)
    for field in V3ClaimDecision.__dataclass_fields__:
        object.__setattr__(wrong_wording, field, getattr(spatial, field))
    object.__setattr__(wrong_wording, "allowed_use", "some_other_claim")
    assert (
        derive_integrated_claim((wrong_wording, causal, bridge), policy_v3()).status
        == "audit_only"
    )
    assert (
        derive_integrated_claim(
            (spatial, causal, bridge), policy_v3(integrated_claim_enabled=False)
        ).status
        == "audit_only"
    )


def test_direct_mutation_is_detected_at_evaluation_boundary() -> None:
    first, second = pair("spatial")
    object.__setattr__(first, "ci_low", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        evaluate_v3_claim((first, second), policy_v3())


def test_direct_mutation_of_identity_and_policy_is_detected_by_mapping() -> None:
    item = pair("bridge")[0]
    object.__setattr__(item, "artifact_identity", "A" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        item.to_mapping()
    policy = policy_v3()
    object.__setattr__(policy, "integrated_claim_enabled", 1)
    with pytest.raises(ValueError, match="built-in bool"):
        policy.to_mapping()


@pytest.mark.parametrize(
    "override", [{"nominal_alpha": 0.049}, {"minimum_lower_bound": -0.01}]
)
def test_v3_policy_does_not_allow_threshold_tuning(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="frozen v3"):
        policy_v3(**override)


@pytest.mark.parametrize(
    "field_override",
    [
        {"attempted_units": True},
        {"nominal_p_value": float("nan")},
        {"ci_low": 0.09, "ci_high": 0.08},
        {"artifact_identity": "A" * 64},
        {"protocol_version": " methods-v3"},
    ],
)
def test_v3_evidence_rejects_hostile_direct_construction(
    field_override: dict[str, object]
) -> None:
    payload = pair("spatial")[0].to_mapping()
    payload.update(field_override)
    with pytest.raises(ValueError):
        V3ClaimEvidence(**payload)  # type: ignore[arg-type]


def test_v3_public_records_are_frozen() -> None:
    item = pair("bridge")[0]
    with pytest.raises(FrozenInstanceError):
        item.comparator_id = "other"  # type: ignore[misc]
