from __future__ import annotations

from dataclasses import FrozenInstanceError
import re
import subprocess

import pytest

import src.discovery.evidence_policy as evidence_policy_module
from src.discovery.evidence_policy import (
    EvidencePolicyV3,
    V3ClaimDecision,
    V3ClaimEvidence,
    build_evidence_policy_v3,
    derive_integrated_claim,
    evaluate_bridge_claim,
    evaluate_v3_claim,
)
from src.evaluation.methods_protocol_v3 import (
    build_methods_protocol_v3,
    protocol_identity_v3,
    protocol_to_mapping_v3,
)


IDENTITY = "a" * 64


def methods_protocol(bridge_role: str = "confirmatory"):
    return build_methods_protocol_v3(
        bridge_role=bridge_role, capability_identity_sha256="9" * 64
    )


def policy_v3(**overrides: object) -> EvidencePolicyV3:
    role = overrides.pop("bridge_role", "confirmatory")
    payload = build_evidence_policy_v3(methods_protocol(role)).to_mapping()
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
    paired_unit_identity: str = "e" * 64,
    run_statuses: tuple[str, ...] | None = None,
    policy: EvidencePolicyV3 | None = None,
) -> V3ClaimEvidence:
    is_complete = attempted == completed
    statuses = run_statuses or (
        ("completed",) * completed + ("failed_resource",) * (attempted - completed)
    )
    policy = policy or policy_v3()
    return V3ClaimEvidence(
        claim_id=claim_id,
        protocol_version="hypersca-methods-v3.0",
        protocol_identity_sha256=policy.protocol_identity_sha256,
        capability_identity_sha256=policy.capability_identity_sha256,
        primary_metric=dict(policy.family_primary_metrics)[claim_id],
        comparator_id=comparator_id,
        paired_estimate=0.04 if is_complete else None,
        ci_low=ci_low if is_complete else None,
        ci_high=0.08 if is_complete else None,
        nominal_p_value=p_value if is_complete else None,
        attempted_units=attempted,
        completed_units=completed,
        paired_unit_identity_sha256=paired_unit_identity,
        run_statuses=statuses,
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
    policy = policy_v3()
    if claim_id == "bridge":
        if status == "blocked":
            return evaluate_bridge_claim((), policy)
        p_value = 0.01 if status == "admitted" else 0.05
        return evaluate_bridge_claim(bridge_pair(p_value, 0.01), policy)
    if status == "blocked":
        required = dict(policy.required_comparators)[claim_id]
        return evaluate_v3_claim(
            (evidence(claim_id, required[0], attempted=3, completed=2),), policy
        )
    return evaluate_v3_claim(
        pair(claim_id, ci_low=0.01 if status == "admitted" else 0.0), policy
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


def test_real_admitted_modules_with_missing_bridge_are_separate_module_audit_only() -> (
    None
):
    policy = policy_v3()
    spatial = evaluate_v3_claim(pair("spatial"), policy)
    causal = evaluate_v3_claim(pair("intracellular_causal"), policy)
    bridge = evaluate_bridge_claim((), policy)

    result = derive_integrated_claim((spatial, causal, bridge), policy)

    assert bridge.status == "blocked"
    assert result.status == "audit_only"
    assert result.allowed_use == "separate_module_claims_only"


@pytest.mark.parametrize("failed_index", [0, 1])
def test_missing_bridge_comparator_does_not_hide_operational_failure(
    failed_index: int,
) -> None:
    policy = policy_v3()
    required = dict(policy.required_comparators)["bridge"]
    failed = evidence(
        "bridge",
        required[failed_index],
        attempted=3,
        completed=2,
        identity="f" * 64,
    )
    bridge = evaluate_bridge_claim((failed,), policy)

    assert bridge.status == "blocked"
    assert any(
        reason.startswith("missing comparator evidence:")
        for reason in bridge.blocking_reasons
    )
    assert any(
        reason.startswith("common paired attempted/completed units are required")
        for reason in bridge.blocking_reasons
    )
    result = derive_integrated_claim(
        (
            evaluate_v3_claim(pair("spatial"), policy),
            evaluate_v3_claim(pair("intracellular_causal"), policy),
            bridge,
        ),
        policy,
    )
    assert result.status == "blocked"
    assert result.allowed_use == "not_used"


def test_partially_attempted_bridge_is_not_classified_as_unattempted() -> None:
    policy = policy_v3()
    bridge = evaluate_bridge_claim((bridge_pair(0.01, 0.01)[0],), policy)

    assert bridge.status == "blocked"
    assert "bridge evidence was attempted" in bridge.blocking_reasons
    result = derive_integrated_claim(
        (
            evaluate_v3_claim(pair("spatial"), policy),
            evaluate_v3_claim(pair("intracellular_causal"), policy),
            bridge,
        ),
        policy,
    )
    assert result.status == "blocked"


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


def test_incomplete_pilot_bridge_evidence_is_blocked_before_role_downgrade() -> None:
    first, second = bridge_pair(0.01, 0.01)
    incomplete_pilot = evidence(
        "bridge",
        second.comparator_id,
        role="pilot_audit_only",
        attempted=3,
        completed=2,
        identity="f" * 64,
    )

    decision = evaluate_bridge_claim((first, incomplete_pilot), policy_v3())

    assert decision.status == "blocked"
    assert decision.allowed_use == "not_used"
    result = derive_integrated_claim(
        (
            evaluate_v3_claim(pair("spatial"), policy_v3()),
            evaluate_v3_claim(pair("intracellular_causal"), policy_v3()),
            decision,
        ),
        policy_v3(),
    )
    assert result.status == "blocked"
    assert result.allowed_use == "not_used"


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


def test_application_evidence_identities_are_canonical_sorted() -> None:
    first = evidence("bridge", "crc_first", role="application_only", identity="f" * 64)
    second = evidence(
        "bridge", "crc_second", role="application_only", identity="e" * 64
    )
    decision = evaluate_bridge_claim(
        (*bridge_pair(0.01, 0.01), first, second), policy_v3()
    )
    assert decision.application_evidence_identities == ("e" * 64, "f" * 64)


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
    with pytest.raises(ValueError, match="replay"):
        derive_integrated_claim((wrong_wording, causal, bridge), policy_v3())
    assert (
        policy_v3(
            bridge_role="pilot_audit_only", integrated_claim_enabled=False
        ).integrated_claim_enabled
        is False
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


def test_v3_policy_binds_exact_protocol_and_bridge_role_capability() -> None:
    with pytest.raises(ValueError, match="protocol_version"):
        policy_v3(protocol_version="hypersca-methods-v3.1")
    with pytest.raises(ValueError, match="integrated_claim_enabled"):
        policy_v3(bridge_role="pilot_audit_only", integrated_claim_enabled=True)
    pilot = policy_v3(bridge_role="pilot_audit_only", integrated_claim_enabled=False)
    assert pilot.to_mapping()["bridge_role"] == "pilot_audit_only"


def test_policy_mapping_rejects_mutated_protocol_and_bridge_capability() -> None:
    policy = policy_v3()
    object.__setattr__(policy, "protocol_version", "hypersca-methods-v3.1")
    with pytest.raises(ValueError, match="protocol_version"):
        policy.to_mapping()
    policy = policy_v3()
    object.__setattr__(policy, "bridge_role", "pilot_audit_only")
    with pytest.raises(ValueError, match="integrated_claim_enabled"):
        evaluate_bridge_claim(bridge_pair(0.01, 0.01), policy)


def test_v3_policy_uses_the_methods_protocol_causal_comparator() -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="confirmatory", capability_identity_sha256="a" * 64
    )
    claims = protocol_to_mapping_v3(protocol)["claims"]
    causal_comparator = claims["intracellular_causal"]["comparators"]["confirmatory"]  # type: ignore[index]

    assert (
        dict(policy_v3().required_comparators)["intracellular_causal"][0]
        == causal_comparator
    )


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_role", "pilot_audit_only"),
        ("nominal_p_value", 0.99),
        ("allowed_use", "forged_claim"),
        ("blocking_reasons", ("forged",)),
        ("status", "audit_only"),
    ],
)
def test_derive_replays_evidence_and_rejects_mutated_admitted_decisions(
    field: str, value: object
) -> None:
    policy = policy_v3()
    spatial = evaluate_v3_claim(pair("spatial"), policy)
    causal = evaluate_v3_claim(pair("intracellular_causal"), policy)
    bridge = evaluate_bridge_claim(bridge_pair(0.01, 0.02), policy)
    object.__setattr__(spatial, field, value)

    with pytest.raises(ValueError, match="replay"):
        derive_integrated_claim((spatial, causal, bridge), policy)


def test_derive_rejects_directly_forged_admitted_decision() -> None:
    policy = policy_v3()
    causal = evaluate_v3_claim(pair("intracellular_causal"), policy)
    bridge = evaluate_bridge_claim(bridge_pair(0.01, 0.02), policy)
    forged = V3ClaimDecision(
        claim_id="spatial",
        protocol_version=policy.protocol_version,
        status="admitted",
        allowed_use="spatial_representation_preservation_gain",
        blocking_reasons=(),
        evidence_identity="a" * 64,
        nominal_p_value=0.01,
        multiplicity_adjustment="none_family_specific",
        evidence_role="confirmatory",
        application_evidence_identities=(),
        source_evidence=tuple(
            sorted(pair("spatial"), key=lambda item: item.comparator_id)
        ),
    )

    with pytest.raises(ValueError, match="replay"):
        derive_integrated_claim((forged, causal, bridge), policy)


def test_required_comparators_must_share_paired_unit_identity() -> None:
    first, second = bridge_pair(0.01, 0.02)
    mismatched = evidence(
        "bridge",
        second.comparator_id,
        p_value=0.02,
        paired_unit_identity="f" * 64,
        identity=second.artifact_identity,
    )

    decision = evaluate_bridge_claim((first, mismatched), policy_v3())

    assert decision.status == "blocked"
    assert any("paired unit identity" in reason for reason in decision.blocking_reasons)


def test_terminal_failure_cannot_be_reclassified_by_mutating_reasons() -> None:
    policy = policy_v3()
    required = dict(policy.required_comparators)["bridge"]
    failed = evidence(
        "bridge",
        required[0],
        attempted=3,
        completed=2,
        identity="f" * 64,
        run_statuses=("completed", "failed_timeout", "completed"),
    )
    bridge = evaluate_bridge_claim((failed,), policy)
    object.__setattr__(
        bridge,
        "blocking_reasons",
        ("missing comparator evidence: hypersca_own_only",),
    )

    with pytest.raises(ValueError, match="replay"):
        derive_integrated_claim(
            (
                evaluate_v3_claim(pair("spatial"), policy),
                evaluate_v3_claim(pair("intracellular_causal"), policy),
                bridge,
            ),
            policy,
        )


def test_v3_evidence_rejects_estimate_outside_confidence_interval() -> None:
    payload = pair("spatial")[0].to_mapping()
    payload.update({"paired_estimate": 0.09, "ci_low": 0.01, "ci_high": 0.08})
    with pytest.raises(ValueError, match="paired_estimate"):
        V3ClaimEvidence(**payload)  # type: ignore[arg-type]


def test_policy_binds_revalidated_task2_protocol_and_capability_identity() -> None:
    protocol = methods_protocol()
    policy = build_evidence_policy_v3(protocol)
    assert policy.protocol_identity_sha256 == protocol_identity_v3(protocol)
    assert policy.capability_identity_sha256 == protocol.capability_identity_sha256
    payload = policy.to_mapping()
    payload["capability_identity_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="capability_identity"):
        EvidencePolicyV3(**payload)  # type: ignore[arg-type]


def test_canonical_zero_and_evidence_order_produce_same_identity() -> None:
    policy = policy_v3()
    first, second = pair("spatial")
    negative_zero = evidence(
        "spatial",
        first.comparator_id,
        ci_low=-0.0,
        p_value=-0.0,
        identity=first.artifact_identity,
    )
    positive_zero = evidence(
        "spatial",
        first.comparator_id,
        ci_low=0.0,
        p_value=0.0,
        identity=first.artifact_identity,
    )
    assert negative_zero.to_mapping() == positive_zero.to_mapping()
    all_zero = negative_zero.to_mapping()
    all_zero.update({"paired_estimate": -0.0, "ci_low": -0.0, "ci_high": 0.0})
    normalized = V3ClaimEvidence(**all_zero)  # type: ignore[arg-type]
    assert normalized.paired_estimate == 0.0
    assert policy_v3(minimum_lower_bound=-0.0).to_mapping() == policy.to_mapping()
    assert (
        evaluate_v3_claim((negative_zero, second), policy).evidence_identity
        == evaluate_v3_claim((second, positive_zero), policy).evidence_identity
    )


def test_complete_pilot_bridge_retains_iut_maximum_p_value() -> None:
    pilot = policy_v3(bridge_role="pilot_audit_only", integrated_claim_enabled=False)
    first, second = dict(pilot.required_comparators)["bridge"]
    decision = evaluate_bridge_claim(
        (
            evidence(
                "bridge", first, role="pilot_audit_only", p_value=0.01, policy=pilot
            ),
            evidence(
                "bridge", second, role="pilot_audit_only", p_value=0.04, policy=pilot
            ),
        ),
        pilot,
    )
    assert decision.status == "audit_only"
    assert decision.nominal_p_value == 0.04
    assert decision.multiplicity_adjustment == "none_intersection_union"


def test_v3_decision_bounds_reasons_and_application_identities() -> None:
    decision = evaluate_v3_claim(pair("spatial"), policy_v3())
    payload = decision.to_mapping()
    payload["blocking_reasons"] = tuple("reason" for _ in range(17))
    with pytest.raises(ValueError, match="blocking_reasons"):
        V3ClaimDecision(**payload)  # type: ignore[arg-type]
    payload = decision.to_mapping()
    payload["application_evidence_identities"] = tuple(
        f"{index:064x}" for index in range(9)
    )
    with pytest.raises(ValueError, match="application_evidence_identities"):
        V3ClaimDecision(**payload)  # type: ignore[arg-type]


def test_v3_sha_validation_does_not_trust_rebound_v2_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = pair("spatial")[0]
    object.__setattr__(item, "artifact_identity", "not-a-sha256")
    monkeypatch.setattr(evidence_policy_module, "_SHA256_PATTERN", re.compile(".*"))
    with pytest.raises(ValueError, match="SHA-256"):
        item.to_mapping()


def test_integrated_decisions_require_replayable_components_on_construction_and_mapping() -> (
    None
):
    policy = policy_v3()
    forged = V3ClaimDecision(
        claim_id="integrated",
        protocol_version=policy.protocol_version,
        status="admitted",
        allowed_use="integrated_spatial_causal_gain",
        blocking_reasons=(),
        evidence_identity="a" * 64,
        nominal_p_value=None,
        multiplicity_adjustment="not_applicable",
        evidence_role="integrated",
    )
    with pytest.raises(ValueError, match="integrated"):
        forged.to_mapping()

    integrated = derive_integrated_claim(
        (
            evaluate_v3_claim(pair("spatial"), policy),
            evaluate_v3_claim(pair("intracellular_causal"), policy),
            evaluate_bridge_claim(bridge_pair(0.01, 0.02), policy),
        ),
        policy,
    )
    object.__setattr__(integrated, "status", "audit_only")
    object.__setattr__(integrated, "allowed_use", "separate_module_claims_only")
    object.__setattr__(integrated, "evidence_role", "integrated")
    with pytest.raises(ValueError, match="replay"):
        integrated.to_mapping()


def test_integrated_application_identity_bound_is_three_family_union() -> None:
    policy = policy_v3()

    def with_applications(claim_id: str) -> tuple[V3ClaimEvidence, ...]:
        scientific = pair(claim_id) if claim_id != "bridge" else bridge_pair(0.01, 0.02)
        offset = {"spatial": 3, "intracellular_causal": 6, "bridge": 9}[claim_id]
        applications = tuple(
            evidence(
                claim_id,
                f"crc_{claim_id}_{index}",
                role="application_only",
                identity=f"{index + offset:064x}",
            )
            for index in range(3)
        )
        return (*scientific, *applications)

    integrated = derive_integrated_claim(
        (
            evaluate_v3_claim(with_applications("spatial"), policy),
            evaluate_v3_claim(with_applications("intracellular_causal"), policy),
            evaluate_bridge_claim(with_applications("bridge"), policy),
        ),
        policy,
    )
    assert len(integrated.application_evidence_identities) == 9
    assert integrated.application_evidence_identities == tuple(
        sorted(integrated.application_evidence_identities)
    )
    payload = integrated.to_mapping()
    payload["application_evidence_identities"] = tuple(
        f"{index:064x}" for index in range(25)
    )
    with pytest.raises(ValueError, match="application_evidence_identities"):
        V3ClaimDecision(**payload)  # type: ignore[arg-type]


def test_v3_policy_cold_import_does_not_load_numerical_stacks() -> None:
    result = subprocess.run(
        [
            "python3.10",
            "-c",
            "from src.methods_protocol_v3_contract import build_methods_protocol_v3; from src.discovery.evidence_policy import build_evidence_policy_v3; import sys; build_evidence_policy_v3(build_methods_protocol_v3(bridge_role='confirmatory', capability_identity_sha256='a'*64)); assert not set(sys.modules) & {'numpy', 'pandas', 'scipy', 'torch'}",
        ],
        cwd=".",
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("unsafe", ["\u202e", "\u200d", "\u2066"])
def test_v3_text_rejects_unicode_category_c(unsafe: str) -> None:
    with pytest.raises(ValueError, match="NFC-safe"):
        evidence("spatial", f"matched{unsafe}_baseline")
