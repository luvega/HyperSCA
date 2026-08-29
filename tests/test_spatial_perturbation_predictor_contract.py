from __future__ import annotations

from dataclasses import FrozenInstanceError
import builtins
import json
from types import MappingProxyType

import pytest

from src.evaluation.run_evidence_identity import canonical_json_bytes, canonical_sha256
from src.methods_protocol_v3_contract import (
    build_methods_protocol_v3,
    protocol_to_mapping_v3,
)
from src.evaluation.spatial_perturbation_predictor_contract import (
    BridgePredictionBundle,
    BridgePredictorCapability,
    BridgePredictorContractError,
    PREDICTION_SCHEMA,
    audit_bridge_predictor_capability,
    bridge_prediction_bundle_to_mapping,
    bridge_predictor_capability_to_mapping,
    build_bridge_prediction_bundle,
)


METHODS = (
    "hypersca",
    "matched_euclidean_spatial_causal",
    "hypersca_own_only",
)
SCHEMA = {
    "format": "bridge_comparator_predictions_json_v1",
    "methods": list(METHODS),
    "columns": ["unit_id", "endpoint", "predicted_effect", "effect_units"],
}


def declarations() -> tuple[dict[str, object], dict[str, object]]:
    registry = {
        "schema_version": "spatial_perturbation_bridge_candidates_v1",
        "candidates": [
            {
                "candidate_id": "gse274447_msafe_bridge",
                "accession": "GSE274447",
                "platform": "spatial_perturbation",
                "biological_specimens": ["mouse_1", "mouse_2", "mouse_3"],
                "sections_by_specimen": [
                    ["mouse_1", []],
                    ["mouse_2", []],
                    ["mouse_3", []],
                ],
                "safe_control_label": "mSafe",
                "perturbation_labels": [],
                "source_uri": (
                    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447"
                ),
                "source_identity_sha256": (
                    "0e908ba2f21cab2bd222daf31a85ff8369407c8df53f5d9a2424f081528ffa46"
                ),
            }
        ],
    }
    protocol = protocol_to_mapping_v3(
        build_methods_protocol_v3(
            bridge_role="pilot_audit_only",
            capability_identity_sha256="f" * 64,
        )
    )
    return registry, protocol


def prediction_payload(*, origin: str = "synthetic_fixture") -> bytes:
    rows = [
        {
            "unit_id": "a" * 64,
            "endpoint": "own",
            "predicted_effect": 1.0,
            "effect_units": "train_control_standardized_delta",
        },
        {
            "unit_id": "b" * 64,
            "endpoint": "neighbor",
            "predicted_effect": -0.5,
            "effect_units": "train_control_standardized_delta",
        },
    ]
    own_only = [dict(row) for row in rows]
    own_only[1]["predicted_effect"] = 0.0
    return canonical_json_bytes(
        {
            "schema_version": "1.0",
            "origin": origin,
            "predictions": {
                "hypersca": rows,
                "matched_euclidean_spatial_causal": [dict(row) for row in rows],
                "hypersca_own_only": own_only,
            },
        }
    )


def synthetic_bundle(**changes: object) -> BridgePredictionBundle:
    values: dict[str, object] = {
        "method_id": "hypersca",
        "protocol_identity_sha256": "a" * 64,
        "data_identity_sha256": "b" * 64,
        "split_identity_sha256": "c" * 64,
        "statistical_unit_identity_sha256": "d" * 64,
        "code_identity_sha256": "e" * 64,
        "model_seed": 11,
        "prediction_schema": SCHEMA,
        "prediction_bytes": prediction_payload(),
        "origin": "synthetic_fixture",
        "evidence_role": "synthetic_audit_only",
    }
    values.update(changes)
    return build_bridge_prediction_bundle(**values)


def test_outcome_blind_audit_returns_only_fixed_terminal_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, protocol = declarations()

    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("src.models", "src.causal")):
            raise AssertionError("capability audit must not import a model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    capability = audit_bridge_predictor_capability(
        registry, protocol, method_id="hypersca"
    )

    assert type(capability) is BridgePredictorCapability
    assert capability.status == "method_adapter_not_executable"
    assert capability.executable is False
    assert capability.adapter_identity_sha256 is None
    assert capability.blocking_reasons == (
        "no_preregistered_bridge_predictor_adapter",
    )


def test_capability_is_frozen_snapshot_and_every_boundary_revalidates() -> None:
    registry, protocol = declarations()
    capability = audit_bridge_predictor_capability(
        registry, protocol, method_id="hypersca"
    )
    mapping = bridge_predictor_capability_to_mapping(capability)

    assert mapping["registry_identity_sha256"] == canonical_sha256(registry)
    assert mapping["protocol_identity_sha256"] == canonical_sha256(protocol)
    assert mapping["status"] == "method_adapter_not_executable"
    with pytest.raises(FrozenInstanceError):
        capability.executable = True  # type: ignore[misc]
    mapping["status"] = "completed"
    assert capability.status == "method_adapter_not_executable"

    object.__setattr__(capability, "status", "completed")
    with pytest.raises(BridgePredictorContractError):
        bridge_predictor_capability_to_mapping(capability)


@pytest.mark.parametrize(
    "bad_registry",
    [
        lambda: None,
        {"schema_version": "1.0", "factory": "pkg.module:factory"},
        {"schema_version": "1.0", "import_path": "src.models.bridge"},
        {"schema_version": "1.0", "outcomes": [1.0]},
    ],
)
def test_audit_rejects_callables_import_paths_and_outcomes(
    bad_registry: object,
) -> None:
    _, protocol = declarations()
    with pytest.raises(BridgePredictorContractError):
        audit_bridge_predictor_capability(
            bad_registry, protocol, method_id="hypersca"
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda registry: registry.__setitem__("unknown", "accepted-by-blacklist"),
        lambda registry: registry.__setitem__("schema_version", 1.0),
        lambda registry: registry.__setitem__("candidates", []),
        lambda registry: registry.__setitem__(
            "candidates", [dict(registry["candidates"][0], platform="other")]
        ),
        lambda registry: registry["candidates"][0].__setitem__(
            "unknown", "accepted-by-blacklist"
        ),
        lambda registry: registry["candidates"][0].__setitem__(
            "source_identity_sha256", "not-a-sha"
        ),
    ),
)
def test_registry_declaration_is_an_exact_closed_typed_schema(
    mutation: object,
) -> None:
    registry, protocol = declarations()
    mutation(registry)  # type: ignore[operator]

    with pytest.raises(BridgePredictorContractError, match="registry"):
        audit_bridge_predictor_capability(
            registry, protocol, method_id="hypersca"
        )


@pytest.mark.parametrize(
    "field",
    (
        "outcome_response",
        "response_vector",
        "effect_estimate",
        "metric_value",
        "expression_matrix",
        "prediction_payload",
    ),
)
def test_registry_closed_schema_rejects_nested_scientific_data_synonyms(
    field: str,
) -> None:
    registry, protocol = declarations()
    registry["candidates"][0][field] = [0.0]  # type: ignore[index]

    with pytest.raises(BridgePredictorContractError, match="registry"):
        audit_bridge_predictor_capability(
            registry, protocol, method_id="hypersca"
        )


def test_registry_rejects_a_method_not_in_the_unique_formal_declaration() -> None:
    registry, protocol = declarations()

    with pytest.raises(BridgePredictorContractError, match="method"):
        audit_bridge_predictor_capability(
            registry, protocol, method_id="other"
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda protocol: protocol.__setitem__("protocol_version", "hypersca-methods-v3"),
        lambda protocol: protocol["claims"]["bridge"].__setitem__(
            "secondary_bands", []
        ),
    ),
)
def test_audit_rejects_nonformal_protocol_version_or_declaration(
    mutation: object,
) -> None:
    registry, protocol = declarations()
    mutation(protocol)  # type: ignore[operator]

    with pytest.raises(BridgePredictorContractError, match="protocol"):
        audit_bridge_predictor_capability(
            registry, protocol, method_id="hypersca"
        )


def test_prediction_bundle_binds_exact_bytes_and_all_identity_axes() -> None:
    bundle = synthetic_bundle()
    mapping = bridge_prediction_bundle_to_mapping(bundle)

    assert type(bundle) is BridgePredictionBundle
    assert mapping["prediction_bytes_sha256"] == canonical_sha256(
        json.loads(bundle.prediction_bytes)
    )
    assert mapping["prediction_bundle_identity_sha256"] == (
        bundle.prediction_bundle_identity_sha256
    )
    assert mapping["prediction_schema"] == SCHEMA
    mapping["prediction_schema"]["format"] = "changed"
    assert bundle.prediction_schema["format"] == SCHEMA["format"]

    replacements: dict[str, object] = {
        "method_id": "hypersca_revision",
        "protocol_identity_sha256": "1" * 64,
        "data_identity_sha256": "2" * 64,
        "split_identity_sha256": "3" * 64,
        "statistical_unit_identity_sha256": "4" * 64,
        "code_identity_sha256": "5" * 64,
        "model_seed": 23,
    }
    for field, replacement in replacements.items():
        changed = synthetic_bundle(**{field: replacement})
        assert changed.prediction_bundle_identity_sha256 != (
            bundle.prediction_bundle_identity_sha256
        ), field


def test_public_prediction_schema_mutation_is_never_a_trust_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(PREDICTION_SCHEMA, "format", "attacker_schema")

    with pytest.raises(BridgePredictorContractError, match="schema"):
        synthetic_bundle(prediction_schema=PREDICTION_SCHEMA)
    assert synthetic_bundle(prediction_schema=SCHEMA).prediction_schema["format"] == (
        "bridge_comparator_predictions_json_v1"
    )

@pytest.mark.parametrize(
    "field,bad",
    [
        ("model_seed", True),
        ("model_seed", -1),
        ("model_seed", 2**63),
        ("prediction_schema", dict(SCHEMA, callable=lambda: None)),
        ("prediction_schema", MappingProxyType(dict(SCHEMA))),
        ("prediction_bytes", b"{not-json}"),
        ("prediction_bytes", canonical_json_bytes({"origin": "synthetic_fixture"})),
        ("method_id", " e\u0301"),
    ],
)
def test_prediction_bundle_rejects_unsafe_types_schema_and_bytes(
    field: str, bad: object
) -> None:
    with pytest.raises(BridgePredictorContractError):
        synthetic_bundle(**{field: bad})


@pytest.mark.parametrize(
    "origin,evidence_role,payload_origin",
    [
        ("production", "synthetic_audit_only", "production"),
        ("synthetic_fixture", "pilot_audit_only", "synthetic_fixture"),
        ("production", "pilot_audit_only", "synthetic_fixture"),
        ("synthetic_fixture", "release_candidate", "production"),
    ],
)
def test_production_and_synthetic_origins_roles_and_bytes_never_mix(
    origin: str, evidence_role: str, payload_origin: str
) -> None:
    with pytest.raises(BridgePredictorContractError):
        synthetic_bundle(
            origin=origin,
            evidence_role=evidence_role,
            prediction_bytes=prediction_payload(origin=payload_origin),
        )


@pytest.mark.parametrize("evidence_role", ["pilot_audit_only", "release_candidate"])
def test_current_nonexecutable_capability_forbids_consistent_production_bundle(
    evidence_role: str,
) -> None:
    with pytest.raises(
        BridgePredictorContractError, match="production.*not executable"
    ):
        synthetic_bundle(
            origin="production",
            evidence_role=evidence_role,
            prediction_bytes=prediction_payload(origin="production"),
        )


def test_prediction_bundle_tampering_is_rejected_by_accessor_boundary() -> None:
    bundle = synthetic_bundle()
    object.__setattr__(bundle, "prediction_bytes", prediction_payload(origin="production"))

    with pytest.raises(BridgePredictorContractError):
        bridge_prediction_bundle_to_mapping(bundle)
