from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unicodedata

from hypothesis import assume, given, settings, strategies as st
import pytest

from src.evaluation.run_evidence_identity import (
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import RunEvidencePublisher


SAFE_TEXT = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cc", "Cf", "Cs", "Co", "Cn"),
        blacklist_characters="/\\",
    ),
    min_size=1,
    max_size=20,
).filter(lambda value: value == value.strip())
SAFE_TEXT = SAFE_TEXT.filter(lambda value: unicodedata.normalize("NFC", value) == value)
STRICT_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63 - 1), max_value=2**63 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    SAFE_TEXT,
)


def valid_identity(**changes: object) -> RunEvidenceIdentity:
    unit_record = {"units": ["sample:block-1"]}
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
        "statistical_unit_identity_sha256": canonical_sha256(unit_record),
        "analysis_identity_sha256": "d" * 64,
        "input_identity_sha256": "e" * 64,
        "config_identity_sha256": "f" * 64,
        "code_identity_sha256": "0" * 64,
        "evidence_role": "pilot_audit_only",
    }
    values.update(changes)
    return RunEvidenceIdentity(**values)


@given(st.dictionaries(SAFE_TEXT, STRICT_SCALAR, min_size=1, max_size=12))
def test_canonical_identity_ignores_mapping_insertion_order(
    payload: dict[str, object],
) -> None:
    reversed_payload = dict(reversed(tuple(payload.items())))
    assert canonical_sha256(payload) == canonical_sha256(reversed_payload)


@given(
    st.sampled_from(
        [
            "protocol_identity",
            "data_split_identity_sha256",
            "statistical_unit_identity_sha256",
            "analysis_identity_sha256",
            "input_identity_sha256",
            "config_identity_sha256",
            "code_identity_sha256",
        ]
    ),
    st.sampled_from("123456789abcdef"),
)
def test_every_sha_identity_field_change_changes_run_sha(
    field: str, replacement_character: str
) -> None:
    first = valid_identity()
    replacement = replacement_character * 64
    assume(first.to_record()[field] != replacement)
    changed = replace(first, **{field: replacement})
    assert changed.run_identity_sha256 != first.run_identity_sha256


@given(
    st.lists(SAFE_TEXT, min_size=1, max_size=20, unique=True),
    SAFE_TEXT,
)
def test_any_statistical_unit_addition_changes_unit_identity(
    units: list[str], added: str
) -> None:
    assume(added not in units)
    first = {"units": units}
    changed = {"units": [*units, added]}
    assert canonical_sha256(first) != canonical_sha256(changed)


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_pilot_cannot_reuse_one_integer_as_split_and_model_seed(seed: int) -> None:
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(data_split_seed=seed, model_seed=seed)


@given(
    st.sampled_from(
        [
            "/absolute.json",
            "../escape.json",
            "nested/../../escape.json",
            "bad\x00name.json",
            "bad\nname.json",
            "e\u0301.json",
        ]
    )
)
def test_illegal_relative_path_never_escapes_staging(relative_path: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        publisher = RunEvidencePublisher.begin(
            output_dir=root / "bundle",
            identity=valid_identity(),
            statistical_unit_record={"units": ["sample:block-1"]},
            required_artifacts=(),
            maximum_bundle_bytes=4096,
        )
        with pytest.raises(RunEvidenceError, match="invalid_artifact"):
            publisher.add_bytes(relative_path, b"x", media_type="text/plain")
        assert publisher.state == "aborted"
        assert not (root.parent / "escape.json").exists()
        assert not (root / "bundle").exists()


@settings(max_examples=30, deadline=None)
@given(
    st.lists(
        st.sampled_from(("add", "complete", "failure", "abort")),
        min_size=1,
        max_size=12,
    )
)
def test_publisher_state_machine_publishes_at_most_once(
    operations: list[str],
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        publisher = RunEvidencePublisher.begin(
            output_dir=root / "bundle",
            identity=valid_identity(),
            statistical_unit_record={"units": ["sample:block-1"]},
            required_artifacts=("metrics.json",),
            maximum_bundle_bytes=16_384,
        )
        observed_publications = 0
        for operation in operations:
            try:
                if operation == "add":
                    publisher.add_bytes(
                        "metrics.json", b"{}", media_type="application/json"
                    )
                elif operation == "complete":
                    publisher.finalize_completed(summary={"status": "audit_only"})
                elif operation == "failure":
                    publisher.finalize_failure(
                        status="failed_runtime", reason="generated failure"
                    )
                else:
                    publisher.abort()
            except RunEvidenceError:
                pass
            observed_publications = max(
                observed_publications, int((root / "bundle").exists())
            )
        if publisher.state == "staging":
            publisher.abort()
        assert observed_publications <= 1
        assert sum(path.name == "bundle" for path in root.iterdir()) <= 1
