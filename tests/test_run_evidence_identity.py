from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from src.evaluation.run_evidence_identity import (
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_json_bytes,
    canonical_sha256,
    validate_strict_json,
)


class EvilInt(int):
    pass


class EvilStr(str):
    pass


def valid_identity(**changes: object) -> RunEvidenceIdentity:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": "hypersca-methods-v2.1",
        "protocol_identity": "a" * 64,
        "claim_id": "spatial",
        "benchmark_id": "osta_colon",
        "data_scopes": ("train", "tune"),
        "data_split_seed": 19911,
        "model_seed": 11,
        "data_split_identity_sha256": "b" * 64,
        "statistical_unit_schema": "osta_platform_sample_block_v1",
        "statistical_unit_identity_sha256": "c" * 64,
        "analysis_identity_sha256": "d" * 64,
        "input_identity_sha256": "e" * 64,
        "config_identity_sha256": "f" * 64,
        "code_identity_sha256": "0" * 64,
        "evidence_role": "pilot_audit_only",
    }
    values.update(changes)
    return RunEvidenceIdentity(**values)


def test_split_and_model_seed_are_distinct_identity_fields() -> None:
    first = valid_identity(data_split_seed=19911, model_seed=11)
    second = valid_identity(data_split_seed=19911, model_seed=23)
    changed_split = valid_identity(data_split_seed=23, model_seed=11)

    assert first.run_identity_sha256 != second.run_identity_sha256
    assert first.run_identity_sha256 != changed_split.run_identity_sha256
    assert second.run_identity_sha256 != changed_split.run_identity_sha256


@pytest.mark.parametrize("field", ["data_split_seed", "model_seed"])
@pytest.mark.parametrize("bad", [True, 1.0, EvilInt(11), -1, 2**63])
def test_seed_fields_require_bounded_exact_builtin_ints(
    field: str, bad: object
) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(**{field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("schema_version", "2.0"),
        ("claim_id", " spatial"),
        ("benchmark_id", "osta\ncolon"),
        ("protocol_version", "hypersca-methods-v2.1\x00"),
        ("statistical_unit_schema", "e\u0301"),
        ("claim_id", EvilStr("spatial")),
    ],
)
def test_identity_text_is_exact_safe_nfc_text(field: str, bad: object) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(**{field: bad})


@pytest.mark.parametrize(
    "bad",
    [
        ["train", "tune"],
        ("train", "train"),
        ("train", "refit"),
        (EvilStr("train"), "tune"),
    ],
)
def test_pilot_scopes_are_the_exact_registered_tuple(bad: object) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(data_scopes=bad)


@pytest.mark.parametrize(
    "role",
    ["pilot", "admitted", "audit_only", "confirmed_mechanism", "release"],
)
def test_unknown_or_overclaiming_evidence_roles_are_rejected(role: str) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(evidence_role=role)


@pytest.mark.parametrize(
    "field",
    [
        "protocol_identity",
        "data_split_identity_sha256",
        "statistical_unit_identity_sha256",
        "analysis_identity_sha256",
        "input_identity_sha256",
        "config_identity_sha256",
        "code_identity_sha256",
    ],
)
@pytest.mark.parametrize("bad", ["a" * 63, "A" * 64, "g" * 64, EvilStr("a" * 64)])
def test_identity_sha_fields_require_lowercase_exact_sha256(
    field: str, bad: object
) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(**{field: bad})


def test_identity_is_frozen_and_round_trips_from_its_record() -> None:
    identity = valid_identity()

    with pytest.raises(FrozenInstanceError):
        identity.model_seed = 23  # type: ignore[misc]

    record = identity.to_record()
    assert "run_identity_sha256" not in record
    assert RunEvidenceIdentity.from_record(record) == identity
    record["model_seed"] = 23
    assert identity.model_seed == 11


def test_every_frozen_identity_field_contributes_to_the_sha() -> None:
    identity = valid_identity()
    replacements: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": "hypersca-methods-v2.1-revision",
        "protocol_identity": "1" * 64,
        "claim_id": "causal",
        "benchmark_id": "osta_brain",
        "data_split_seed": 19912,
        "model_seed": 23,
        "data_split_identity_sha256": "2" * 64,
        "statistical_unit_schema": "another_unit_v1",
        "statistical_unit_identity_sha256": "3" * 64,
        "analysis_identity_sha256": "4" * 64,
        "input_identity_sha256": "5" * 64,
        "config_identity_sha256": "6" * 64,
        "code_identity_sha256": "7" * 64,
        "evidence_role": "infrastructure_smoke",
    }
    for field, replacement in replacements.items():
        if field == "schema_version":
            continue
        changed = replace(identity, **{field: replacement})
        assert changed.run_identity_sha256 != identity.run_identity_sha256, field

    smoke = valid_identity(
        evidence_role="infrastructure_smoke", data_scopes=("synthetic",)
    )
    changed_scope = replace(smoke, data_scopes=("synthetic", "train"))
    assert changed_scope.run_identity_sha256 != smoke.run_identity_sha256


def test_canonical_json_is_order_independent_and_utf8() -> None:
    first = {"z": [1, True, None], "a": "结论"}
    second = {"a": "结论", "z": [1, True, None]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_sha256(first) == canonical_sha256(second)
    assert json.loads(canonical_json_bytes(first)) == second


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        EvilInt(1),
        EvilStr("value"),
        {1: "not-a-text-key"},
        {"bad\nkey": "value"},
        {"key": "e\u0301"},
        {"key": object()},
    ],
)
def test_strict_json_rejects_noncanonical_or_unsafe_values(bad: object) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        validate_strict_json(bad)


def test_strict_json_returns_an_independent_builtin_tree() -> None:
    original = {"outer": ({"number": 3, "finite": 1.5},)}
    normalized = validate_strict_json(original)

    assert type(normalized) is dict
    assert type(normalized["outer"]) is list
    assert type(normalized["outer"][0]) is dict
    assert math.isfinite(normalized["outer"][0]["finite"])
    normalized["outer"][0]["number"] = 99
    assert original["outer"][0]["number"] == 3
