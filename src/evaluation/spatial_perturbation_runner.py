"""Publication-only runner for frozen spatial-perturbation predictions.

The module has no model-fitting entrypoint.  Heavy scientific dependencies are
loaded only when an already-bound prediction bundle is validated.
"""

from __future__ import annotations

import hashlib
import inspect
import csv
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import CodeType, MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, cast, overload

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from src.methods_protocol_v3_contract import MethodsProtocolV3
    from src.evaluation.spatial_perturbation_comparators import BridgeModelBudget
    from src.evaluation.spatial_perturbation_scoring import TrainControlStandardizer
    from src.evaluation.spatial_perturbation_split import (
        BridgeEligibilityResult,
        BridgeSplitManifest,
    )

from src.evaluation.run_evidence_identity import (
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_json_bytes,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import (
    RunEvidencePublisher,
    VerifiedRunEvidence,
    verify_run_evidence_bundle,
)
from src.evaluation.spatial_perturbation_predictor_contract import (
    PREDICTION_SCHEMA,
    BridgePredictionBundle,
    BridgePredictorCapability,
    BridgePredictorContractError,
    bridge_prediction_bundle_to_mapping,
    bridge_predictor_capability_to_mapping,
    build_bridge_prediction_bundle,
    formal_protocol_declaration_identity_sha256,
    formal_protocol_declaration_to_mapping,
    prediction_payload_to_mapping,
)


_FAILURE_ARTIFACTS = ("capability_record.json", "resource_usage.json")
_SUCCESS_ARTIFACTS = (
    "split_manifest.json",
    "capability_record.json",
    "neighbor_units.csv",
    "predictions_hypersca.csv",
    "predictions_matched_euclidean.csv",
    "predictions_hypersca_own_only.csv",
    "primary_metric_units.csv",
    "primary_metric_summary.json",
    "secondary_metrics.csv",
    "resource_usage.json",
    "claim_decision.json",
)
_METHOD_TO_ARTIFACT = {
    "hypersca": "predictions_hypersca.csv",
    "matched_euclidean_spatial_causal": "predictions_matched_euclidean.csv",
    "hypersca_own_only": "predictions_hypersca_own_only.csv",
}
_MAXIMUM_BUNDLE_BYTES = 32 * 1024 * 1024
_RAW_NEIGHBOR_COLUMNS = (
    "animal_id",
    "section_id",
    "spatial_block",
    "cell_id",
    "perturbation_id",
    "cell_type",
    "x",
    "y",
    "barcode_positive",
)
_RAW_NEIGHBOR_TEXT_COLUMNS = _RAW_NEIGHBOR_COLUMNS[:6]
_OBSERVED_PROJECTION_MAX_EFFECTS = 300_000
_MAX_SOURCE_DEPENDENCY_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_DEPENDENCIES_BYTES = 32 * 1024 * 1024
_MAX_CSV_ROWS = 300_000
_MAX_CSV_COLUMNS = 64
_MAX_CSV_CELLS = 1_000_000
_SYNTHETIC_RAW_NEIGHBOR_INPUT_SHA256 = (
    "05b914d0644ce4fc7f8cd764308ab23dc9072e25378d11434394905fcddb1868"
)


def _synthetic_raw_neighbor_input_sha256(
    observed_payload: bytes | None = None,
) -> str:
    """Return the digest of the code-owned Task9 raw-neighbor fixture."""

    digest_state = hashlib.sha256()
    payload_size = 0

    class DigestWriter:
        def write(self, value: str) -> int:
            nonlocal payload_size
            encoded = value.encode("utf-8")
            payload_size += len(encoded)
            if payload_size > _MAXIMUM_BUNDLE_BYTES:
                raise BridgePredictorContractError(
                    "code-owned raw-neighbor fixture exceeds its resource bound"
                )
            digest_state.update(encoded)
            return len(value)

    writer = csv.writer(DigestWriter(), lineterminator="\n")
    writer.writerow(_RAW_NEIGHBOR_COLUMNS)
    for animal_index in range(3):
        animal = f"mouse_{animal_index + 1}"
        for source_index in range(20):
            section = f"{animal}_section_{source_index:02d}"
            block = f"block_{source_index % 3}"
            for perturbation_index, source_perturbation in enumerate(
                ("guide_0", "guide_1", "mSafe")
            ):
                source_x = float(perturbation_index * 1_000)
                source_id = (
                    f"{animal}_{source_index:02d}_{source_perturbation}_source"
                )
                writer.writerow(
                    (
                        animal, section, block, source_id, source_perturbation,
                        "source_type", source_x, 0.0, True,
                    )
                )
                for rank in range(1, 61 if source_index < 10 else 1):
                    writer.writerow(
                        (
                            animal, section, block,
                            f"{source_id}_neighbor_{rank:02d}", "unperturbed",
                            "astrocyte", source_x + float(rank), 0.0, False,
                        )
                    )
    digest = digest_state.hexdigest()
    if digest != _SYNTHETIC_RAW_NEIGHBOR_INPUT_SHA256:
        raise BridgePredictorContractError(
            "code-owned synthetic raw-neighbor fixture changed"
        )
    if observed_payload is not None and (
        len(observed_payload) != payload_size
        or hashlib.sha256(observed_payload).hexdigest() != digest
    ):
        raise BridgePredictorContractError(
            "raw neighbor input differs from the code-owned synthetic fixture"
        )
    return digest


def _run_input_identity_sha256(
    *, data_identity_sha256: str, observed_projection_identity_sha256: str
) -> str:
    return canonical_sha256(
        {
            "schema_version": "bridge_run_input_v2",
            "data_identity_sha256": data_identity_sha256,
            "observed_effect_projection_identity_sha256": (
                observed_projection_identity_sha256
            ),
        }
    )


def _analysis_contract_record(
    analysis: dict[str, object],
) -> dict[str, object]:
    budgets = analysis.get("comparator_budgets")
    if type(budgets) is not list or any(type(item) is not dict for item in budgets):
        raise BridgePredictorContractError(
            "analysis contract requires exact comparator budget mappings"
        )
    budget_contracts = [
        {key: value for key, value in cast(dict[str, object], item).items() if key != "seed"}
        for item in cast(list[object], budgets)
    ]
    support_contract = {
        "eligibility_identity_sha256": analysis.get("eligibility_identity_sha256"),
        "standardizer_identity_sha256": analysis.get("standardizer_identity_sha256"),
        "comparator_budgets": budget_contracts,
        "applicable_model_seeds": [11, 23, 47],
    }
    return {
        "schema_version": "bridge_analysis_contract_v1",
        "protocol_identity_sha256": analysis.get("protocol_identity_sha256"),
        "data_identity_sha256": analysis.get("data_identity_sha256"),
        "split_identity_sha256": analysis.get("split_identity_sha256"),
        "statistical_unit_identity_sha256": analysis.get(
            "statistical_unit_identity_sha256"
        ),
        "code_identity_sha256": analysis.get("code_identity_sha256"),
        "origin": analysis.get("origin"),
        "evidence_role": analysis.get("evidence_role"),
        "neighbor_max_rank": analysis.get("neighbor_max_rank"),
        "neighbor_bands": analysis.get("neighbor_bands"),
        "neighbor_input_sha256": analysis.get("neighbor_input_sha256"),
        "neighbor_units_sha256": analysis.get("neighbor_units_sha256"),
        "synthetic_fixture_identity_sha256": analysis.get(
            "synthetic_fixture_identity_sha256"
        ),
        "support_contract": support_contract,
        "support_contract_identity_sha256": canonical_sha256(support_contract),
        "aggregation": {
            "primary": "animal_hierarchical_equal_weight",
            "primary_bands": ["proximal", "local"],
            "own_endpoint_separate": True,
            "calibration": "proximal_minus_local_context_error",
        },
    }


def _fresh_synthetic_fixture_declaration() -> dict[str, object]:
    animals = [f"mouse_{index}" for index in range(1, 4)]
    candidate = {
        "candidate_id": "generic_task5_bridge",
        "accession": "SYNTHETIC",
        "platform": "spatial_perturbation",
        "biological_specimens": animals,
        "sections_by_specimen": [
            [
                animal,
                [f"{animal}_section_{index:02d}" for index in range(20)],
            ]
            for animal in animals
        ],
        "perturbation_labels": ["guide_0", "guide_1"],
        "safe_control_label": "mSafe",
        "source_uri": "https://example.test/SYNTHETIC",
        "source_identity_sha256": "a" * 64,
    }
    return {
        "schema_version": "1.0",
        "fixture_id": "task9_spatial_bridge_synthetic_fixture_v1",
        "candidate": candidate,
        "candidate_identity_sha256": canonical_sha256(candidate),
        "metadata_identity_sha256": (
            "a2f46d3be4285edda554ae3ddaf42d2c1fe812645fb0161caf9a13d7c99d18ac"
        ),
        "capability_identity_sha256": (
            "d34536deaf81e83006bdef6893f9033929465dda5b0fe104ea3efada07d42a49"
        ),
        "split_identity_sha256": (
            "a74dc81bca2359e707b60edaf9f49fa256ebb2f7bf0cf3a9f5ece5316b37fe77"
        ),
        "neighbour_table_identity_sha256": (
            "dd2d150ed3f832920d6a120d983a13574bbda42b65215dba5d05cbc7221a041c"
        ),
        "data_identity_sha256": (
            "f6e9e9e39e3004fb0594663b3f30b6c0aea38b89de4a3eef0909526b131ea3e4"
        ),
    }


def _synthetic_fixture_identity_sha256(
    split_mapping: dict[str, object], *, data_identity: str
) -> str:
    expected = _fresh_synthetic_fixture_declaration()
    observed = {
        "schema_version": "1.0",
        "fixture_id": "task9_spatial_bridge_synthetic_fixture_v1",
        "candidate": split_mapping.get("candidate"),
        "candidate_identity_sha256": split_mapping.get(
            "candidate_identity_sha256"
        ),
        "metadata_identity_sha256": split_mapping.get(
            "metadata_identity_sha256"
        ),
        "capability_identity_sha256": split_mapping.get(
            "capability_identity_sha256"
        ),
        "split_identity_sha256": split_mapping.get("split_identity_sha256"),
        "neighbour_table_identity_sha256": cast(
            dict[str, object], split_mapping.get("neighbour_table")
        ).get("identity_sha256")
        if type(split_mapping.get("neighbour_table")) is dict
        else None,
        "data_identity_sha256": data_identity,
    }
    if observed != expected:
        raise BridgePredictorContractError(
            "synthetic fixture provenance differs from the code-owned declaration"
        )
    return canonical_sha256(expected)


def _trusted_code_dependencies() -> tuple[tuple[str, str], ...]:
    """Return the fixed canonical labels that define the published identity."""

    return (
        ("methods_protocol_v3", "methods_protocol_v3_contract.py"),
        ("runner", "evaluation/spatial_perturbation_runner.py"),
        (
            "predictor_contract",
            "evaluation/spatial_perturbation_predictor_contract.py",
        ),
        ("run_evidence_identity", "evaluation/run_evidence_identity.py"),
        ("run_evidence_publisher", "evaluation/run_evidence_publisher.py"),
        ("task4_registry", "evaluation/spatial_perturbation_registry.py"),
        ("task5_split", "evaluation/spatial_perturbation_split.py"),
        ("task6_neighbors", "evaluation/spatial_perturbation_neighbors.py"),
        ("task7_scoring", "evaluation/spatial_perturbation_scoring.py"),
        ("task8_comparators", "evaluation/spatial_perturbation_comparators.py"),
    )


def _trusted_dependency_modules() -> tuple[tuple[str, str], ...]:
    return (
        ("methods_protocol_v3", "src.methods_protocol_v3_contract"),
        ("runner", "src.evaluation.spatial_perturbation_runner"),
        (
            "predictor_contract",
            "src.evaluation.spatial_perturbation_predictor_contract",
        ),
        ("run_evidence_identity", "src.evaluation.run_evidence_identity"),
        ("run_evidence_publisher", "src.evaluation.run_evidence_publisher"),
        ("task4_registry", "src.evaluation.spatial_perturbation_registry"),
        ("task5_split", "src.evaluation.spatial_perturbation_split"),
        ("task6_neighbors", "src.evaluation.spatial_perturbation_neighbors"),
        ("task7_scoring", "src.evaluation.spatial_perturbation_scoring"),
        ("task8_comparators", "src.evaluation.spatial_perturbation_comparators"),
    )


def _trusted_runner_imported_bindings() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "run_evidence_identity",
            (
                "RunEvidenceError",
                "RunEvidenceIdentity",
                "canonical_json_bytes",
                "canonical_sha256",
            ),
        ),
        (
            "run_evidence_publisher",
            (
                "RunEvidencePublisher",
                "VerifiedRunEvidence",
                "verify_run_evidence_bundle",
            ),
        ),
        (
            "predictor_contract",
            (
                "BridgePredictionBundle",
                "BridgePredictorCapability",
                "BridgePredictorContractError",
                "bridge_prediction_bundle_to_mapping",
                "bridge_predictor_capability_to_mapping",
                "formal_protocol_declaration_identity_sha256",
                "formal_protocol_declaration_to_mapping",
                "prediction_payload_to_mapping",
            ),
        ),
    )


_CODE_DEPENDENCIES = _trusted_code_dependencies()
_CODE_DEPENDENCY_MODULES = MappingProxyType(dict(_trusted_dependency_modules()))
_RUNNER_IMPORTED_BINDINGS = MappingProxyType(dict(_trusted_runner_imported_bindings()))


def _split_manifest_json_bytes(
    mapping: object, *, maximum_bytes: int = _MAXIMUM_BUNDLE_BYTES
) -> bytes:
    """Serialize a replayed Task5 manifest without run-evidence JSON limits."""

    if (
        type(mapping) is not dict
        or type(maximum_bytes) is not int
        or not 1 <= maximum_bytes <= _MAXIMUM_BUNDLE_BYTES
    ):
        raise BridgePredictorContractError(
            "split manifest serializer requires an exact mapping and resource bound"
        )
    stack: list[tuple[object, int]] = [(mapping, 0)]
    item_count = 0
    while stack:
        value, depth = stack.pop()
        item_count += 1
        if depth > 64 or item_count > 2_000_000:
            raise BridgePredictorContractError(
                "split manifest exceeds bridge JSON structure limits"
            )
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise BridgePredictorContractError(
                    "split manifest JSON object keys must be exact strings"
                )
            stack.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            stack.extend((child, depth + 1) for child in value)
        elif type(value) is float:
            if not math.isfinite(value):
                raise BridgePredictorContractError(
                    "split manifest JSON numbers must be finite"
                )
        elif value is not None and type(value) not in (str, int, bool):
            raise BridgePredictorContractError(
                "split manifest must contain only exact JSON built-ins"
            )
    try:
        payload = json.dumps(
            mapping,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        decoded = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
        canonical = json.dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise BridgePredictorContractError(
            "split manifest is not canonical strict UTF-8 JSON"
        ) from error
    if (
        not payload
        or len(payload) > maximum_bytes
        or payload != canonical
        or decoded != mapping
        or b"\r" in payload
        or b"\x00" in payload
    ):
        raise BridgePredictorContractError(
            "split manifest JSON exceeds its resource bound or is not canonical"
        )
    return payload


def _source_bytes(
    path: Path, *, remaining_total_bytes: int = _MAX_SOURCE_DEPENDENCIES_BYTES
) -> bytes:
    if (
        type(remaining_total_bytes) is not int
        or not 0 <= remaining_total_bytes <= _MAX_SOURCE_DEPENDENCIES_BYTES
    ):
        raise BridgePredictorContractError(
            "runner source cumulative resource bound is invalid"
        )
    directory_descriptors: list[int] = []
    directory_links: list[tuple[str, tuple[int, ...]]] = []
    file_descriptor = -1
    try:
        absolute = path.absolute()
        parts = absolute.parts

        def identity_axes(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
                item.st_size, item.st_mtime_ns, item.st_ctime_ns,
            )

        root_descriptor = os.open(
            absolute.anchor,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        directory_descriptors.append(root_descriptor)
        for component_name in parts[1:-1]:
            directory_descriptor = directory_descriptors[-1]
            before_component = os.stat(
                component_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(before_component.st_mode)
                or not stat.S_ISDIR(before_component.st_mode)
            ):
                raise BridgePredictorContractError(
                    "runner code dependency path contains a symbolic link"
                )
            next_descriptor = os.open(
                component_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            directory_descriptors.append(next_descriptor)
            opened_component = os.fstat(next_descriptor)
            after_component = os.stat(
                component_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not (
                identity_axes(before_component)
                == identity_axes(opened_component)
                == identity_axes(after_component)
            ):
                raise BridgePredictorContractError(
                    "runner code dependency path changed while it was opened"
                )
            directory_links.append(
                (component_name, identity_axes(opened_component))
            )
        directory_descriptor = directory_descriptors[-1]
        leaf = parts[-1]
        before_path = os.stat(
            leaf, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or before_path.st_nlink != 1
        ):
            raise BridgePredictorContractError(
                "runner source path must be a single-link regular non-symbolic-link file"
            )
        file_descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        before = os.fstat(file_descriptor)
        if (
            before.st_nlink != 1
            or identity_axes(before_path) != identity_axes(before)
            or before.st_size <= 0
            or before.st_size > _MAX_SOURCE_DEPENDENCY_BYTES
            or before.st_size > remaining_total_bytes
        ):
            raise BridgePredictorContractError(
                "runner source changed while its code identity was opened"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise BridgePredictorContractError(
                    "runner source was truncated while read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_descriptor, 1):
            raise BridgePredictorContractError(
                "runner source exceeded its declared resource bound"
            )
        after = os.fstat(file_descriptor)
        after_path = os.stat(
            leaf, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            after.st_nlink != 1
            or identity_axes(before) != identity_axes(after)
            or identity_axes(after) != identity_axes(after_path)
        ):
            raise BridgePredictorContractError(
                "runner source changed while its code identity was read"
            )
        for index, (component_name, expected_identity) in enumerate(
            directory_links
        ):
            parent_descriptor = directory_descriptors[index]
            child_descriptor = directory_descriptors[index + 1]
            linked_component = os.stat(
                component_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            opened_component = os.fstat(child_descriptor)
            if (
                stat.S_ISLNK(linked_component.st_mode)
                or not stat.S_ISDIR(linked_component.st_mode)
                or identity_axes(linked_component) != expected_identity
                or identity_axes(opened_component) != expected_identity
            ):
                raise BridgePredictorContractError(
                    "runner code dependency path changed while it was read"
                )
        return b"".join(chunks)
    except BridgePredictorContractError:
        raise
    except (OSError, ValueError) as error:
        raise BridgePredictorContractError(
            "runner source code identity cannot be read safely"
        ) from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _snapshot_dependency_source_sha256() -> tuple[tuple[str, str], ...]:
    source_dir = Path(__file__).absolute().parents[1]
    snapshots: list[tuple[str, str]] = []
    total = 0
    for label, relative_path in _trusted_code_dependencies():
        payload = _source_bytes(
            source_dir / relative_path,
            remaining_total_bytes=_MAX_SOURCE_DEPENDENCIES_BYTES - total,
        )
        total += len(payload)
        if total > _MAX_SOURCE_DEPENDENCIES_BYTES:
            raise BridgePredictorContractError(
                "runner dependency sources exceed their cumulative resource bound"
            )
        snapshots.append((label, hashlib.sha256(payload).hexdigest()))
    return tuple(snapshots)


_INITIAL_DEPENDENCY_SOURCE_SHA256 = _snapshot_dependency_source_sha256()


def _code_value_bytes(value: object) -> bytes:
    if type(value) is CodeType:
        return b"C" + bytes.fromhex(_code_object_fingerprint(cast(CodeType, value)))
    if value is None:
        return b"N"
    if value is Ellipsis:
        return b"E"
    if type(value) is bool:
        return b"B1" if value else b"B0"
    if type(value) is int:
        return b"I" + str(value).encode("ascii")
    if type(value) is float:
        return b"F" + cast(float, value).hex().encode("ascii")
    if type(value) is complex:
        number = cast(complex, value)
        return (
            b"X"
            + number.real.hex().encode("ascii")
            + b","
            + number.imag.hex().encode("ascii")
        )
    if type(value) is str:
        return b"S" + cast(str, value).encode("utf-8")
    if type(value) is bytes:
        return b"Y" + cast(bytes, value)
    if type(value) is tuple:
        result = bytearray(b"T")
        for child in cast(tuple[object, ...], value):
            payload = _code_value_bytes(child)
            result.extend(len(payload).to_bytes(8, "big"))
            result.extend(payload)
        return bytes(result)
    if type(value) is frozenset:
        children = sorted(
            _code_value_bytes(child) for child in cast(frozenset[object], value)
        )
        return b"R" + b"".join(
            len(child).to_bytes(8, "big") + child for child in children
        )
    raise BridgePredictorContractError(
        "verified source contains an unsupported code constant"
    )


def _code_object_fingerprint(code: CodeType) -> str:
    """Fingerprint loaded/compiled correspondence, never publication identity."""

    digest = hashlib.sha256(b"hypersca-python-code-correspondence-v2\0")
    for numeric_value in (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_firstlineno,
    ):
        digest.update(str(numeric_value).encode("ascii"))
        digest.update(b"\0")
    version_local_components: list[object] = [
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_name,
        code.co_consts,
    ]
    line_table = getattr(code, "co_linetable", None)
    if line_table is None:
        line_table = getattr(code, "co_lnotab", b"")
    version_local_components.append(line_table)
    exception_table = getattr(code, "co_exceptiontable", None)
    if exception_table is not None:
        version_local_components.append(exception_table)
    for component_value in version_local_components:
        payload = _code_value_bytes(component_value)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _compiled_code_fingerprints(code: CodeType) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    for item in code.co_consts:
        if type(item) is CodeType and not item.co_name.startswith("<"):
            declaration = cast(CodeType, item)
            collected.setdefault(declaration.co_name, []).append(
                _code_object_fingerprint(declaration)
            )
    return {
        name: tuple(sorted(fingerprints))
        for name, fingerprints in collected.items()
    }


def _verify_loaded_callables(
    module: ModuleType, *, compiled: CodeType
) -> None:
    missing = object()

    def member_code_objects(value: object) -> tuple[CodeType, ...]:
        if inspect.isfunction(value):
            return (value.__code__,)
        if type(value) in (staticmethod, classmethod):
            return (cast(Any, value).__func__.__code__,)
        if type(value) is property:
            return tuple(
                accessor.__code__
                for accessor in (value.fget, value.fset, value.fdel)
                if accessor is not None
            )
        return ()

    def verify_class(value: object, declaration: CodeType) -> None:
        if (
            not inspect.isclass(value)
            or value.__module__ != module.__name__
            or value.__qualname__ != declaration.co_name
        ):
            raise BridgePredictorContractError(
                "loaded callable class differs from the verified source"
            )
        expected_members: dict[str, list[str]] = {}
        for constant in declaration.co_consts:
            if type(constant) is CodeType and not constant.co_name.startswith("<"):
                expected_members.setdefault(constant.co_name, []).append(
                    _code_object_fingerprint(constant)
                )
        for name, expected_fingerprints in expected_members.items():
            member = vars(value).get(name, missing)
            observed = member_code_objects(member)
            if (
                member is missing
                or sorted(_code_object_fingerprint(code) for code in observed)
                != sorted(expected_fingerprints)
            ):
                raise BridgePredictorContractError(
                    "loaded callable method differs from the verified source"
                )

    declarations: dict[str, list[CodeType]] = {}
    for raw_declaration in compiled.co_consts:
        if (
            type(raw_declaration) is CodeType
            and not raw_declaration.co_name.startswith("<")
        ):
            declaration = cast(CodeType, raw_declaration)
            declarations.setdefault(declaration.co_name, []).append(declaration)
    for declaration_name, candidates in declarations.items():
        binding = vars(module).get(declaration_name, missing)
        if inspect.isfunction(binding):
            if (
                binding.__module__ != module.__name__
                or binding.__qualname__ != declaration_name
                or _code_object_fingerprint(binding.__code__)
                not in {
                    _code_object_fingerprint(candidate)
                    for candidate in candidates
                }
            ):
                raise BridgePredictorContractError(
                    "loaded callable code differs from the verified source"
                )
        else:
            if len(candidates) != 1:
                raise BridgePredictorContractError(
                    "loaded callable declaration is ambiguous"
                )
            verify_class(binding, candidates[0])


def _verify_runner_imported_bindings(
    *,
    label: str,
    module: ModuleType,
    compiled: CodeType,
    binding_names: tuple[str, ...],
    runner_module_name: str,
) -> None:
    if not binding_names:
        return
    runner_module = sys.modules.get(runner_module_name)
    if runner_module is None:
        raise BridgePredictorContractError(
            "runner module is unavailable while verifying callable bindings"
        )
    expected = _compiled_code_fingerprints(compiled)
    missing = object()
    for name in binding_names:
        runner_binding = vars(runner_module).get(name, missing)
        dependency_binding = vars(module).get(name, missing)
        if runner_binding is not dependency_binding or dependency_binding is missing:
            raise BridgePredictorContractError(
                "runner loaded callable binding differs from its dependency"
            )
        if inspect.isfunction(dependency_binding):
            code = dependency_binding.__code__
            declared = (
                dependency_binding.__module__ == module.__name__
                and _code_object_fingerprint(code)
                in expected.get(code.co_name, ())
            )
        elif inspect.isclass(dependency_binding):
            declared = (
                dependency_binding.__module__ == module.__name__
                and dependency_binding.__qualname__ in expected
            )
        else:
            declared = False
        if not declared:
            raise BridgePredictorContractError(
                "runner loaded callable binding is not declared by verified source"
            )


def _verified_source_code(
    *, label: str, path: Path, payload: bytes
) -> CodeType:
    dependency_modules = {
        "methods_protocol_v3": "src.methods_protocol_v3_contract",
        "runner": "src.evaluation.spatial_perturbation_runner",
        "predictor_contract": "src.evaluation.spatial_perturbation_predictor_contract",
        "run_evidence_identity": "src.evaluation.run_evidence_identity",
        "run_evidence_publisher": "src.evaluation.run_evidence_publisher",
        "task4_registry": "src.evaluation.spatial_perturbation_registry",
        "task5_split": "src.evaluation.spatial_perturbation_split",
        "task6_neighbors": "src.evaluation.spatial_perturbation_neighbors",
        "task7_scoring": "src.evaluation.spatial_perturbation_scoring",
        "task8_comparators": "src.evaluation.spatial_perturbation_comparators",
    }
    imported_bindings = {
        "run_evidence_identity": (
            "RunEvidenceError", "RunEvidenceIdentity", "canonical_json_bytes",
            "canonical_sha256",
        ),
        "run_evidence_publisher": (
            "RunEvidencePublisher", "VerifiedRunEvidence",
            "verify_run_evidence_bundle",
        ),
        "predictor_contract": (
            "BridgePredictionBundle", "BridgePredictorCapability",
            "BridgePredictorContractError", "bridge_prediction_bundle_to_mapping",
            "bridge_predictor_capability_to_mapping",
            "formal_protocol_declaration_identity_sha256",
            "formal_protocol_declaration_to_mapping", "prediction_payload_to_mapping",
        ),
    }
    if (
        dict(_CODE_DEPENDENCY_MODULES) != dependency_modules
        or dict(_RUNNER_IMPORTED_BINDINGS) != imported_bindings
        or label not in dependency_modules
    ):
        raise BridgePredictorContractError(
            "runner routing configuration differs from its fixed local trust root"
        )
    try:
        compiled = compile(
            payload,
            next(
                relative_path
                for dependency_label, relative_path in _trusted_code_dependencies()
                if dependency_label == label
            ),
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, UnicodeError, ValueError, OverflowError) as error:
        raise BridgePredictorContractError(
            "runner dependency source cannot be compiled safely"
        ) from error
    module = sys.modules.get(dependency_modules[label])
    if module is not None:
        _verify_loaded_callables(module, compiled=compiled)
        _verify_runner_imported_bindings(
            label=label,
            module=module,
            compiled=compiled,
            binding_names=imported_bindings.get(label, ()),
            runner_module_name=dependency_modules["runner"],
        )
    return compiled


def _source_identity() -> str:
    source_dir = Path(__file__).absolute().parents[1]
    digest = hashlib.sha256(b"hypersca-bridge-runner-source-v2\0")
    trusted_dependencies = (
        ("methods_protocol_v3", "methods_protocol_v3_contract.py"),
        ("runner", "evaluation/spatial_perturbation_runner.py"),
        ("predictor_contract", "evaluation/spatial_perturbation_predictor_contract.py"),
        ("run_evidence_identity", "evaluation/run_evidence_identity.py"),
        ("run_evidence_publisher", "evaluation/run_evidence_publisher.py"),
        ("task4_registry", "evaluation/spatial_perturbation_registry.py"),
        ("task5_split", "evaluation/spatial_perturbation_split.py"),
        ("task6_neighbors", "evaluation/spatial_perturbation_neighbors.py"),
        ("task7_scoring", "evaluation/spatial_perturbation_scoring.py"),
        ("task8_comparators", "evaluation/spatial_perturbation_comparators.py"),
    )
    if (
        _CODE_DEPENDENCIES != trusted_dependencies
        or _trusted_code_dependencies() != trusted_dependencies
        or dict(_CODE_DEPENDENCY_MODULES)
        != dict(_trusted_dependency_modules())
        or dict(_RUNNER_IMPORTED_BINDINGS)
        != dict(_trusted_runner_imported_bindings())
    ):
        raise BridgePredictorContractError(
            "runner routing configuration differs from its fixed trust root"
        )
    expected_source_sha256 = dict(_INITIAL_DEPENDENCY_SOURCE_SHA256)
    if set(expected_source_sha256) != {
        label for label, _ in trusted_dependencies
    }:
        raise BridgePredictorContractError(
            "runner dependency source snapshot is not exact"
        )
    verified_sources: list[tuple[str, Path, bytes]] = []
    source_total = 0
    for label, relative_path in trusted_dependencies:
        path = source_dir / relative_path
        payload = _source_bytes(
            path,
            remaining_total_bytes=_MAX_SOURCE_DEPENDENCIES_BYTES - source_total,
        )
        source_total += len(payload)
        if source_total > _MAX_SOURCE_DEPENDENCIES_BYTES:
            raise BridgePredictorContractError(
                "runner dependency sources exceed their cumulative resource bound"
            )
        if hashlib.sha256(payload).hexdigest() != expected_source_sha256[label]:
            raise BridgePredictorContractError(
                "runner dependency source changed after module initialization"
            )
        verified_sources.append((label, path, payload))
    for label, path, payload in verified_sources:
        _verified_source_code(label=label, path=path, payload=payload)
        relative_path = next(
            item_path
            for item_label, item_path in trusted_dependencies
            if item_label == label
        )
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(relative_path.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def runner_code_identity_sha256() -> str:
    """Return the rebound composite identity of every scientific dependency."""

    return _source_identity()


def _identity(
    *,
    protocol_identity: str,
    split_identity: str,
    unit_record: dict[str, object],
    analysis_record: dict[str, object],
    input_identity: str,
    config_identity: str,
    code_identity: str,
    model_seed: int,
    evidence_role: str,
) -> RunEvidenceIdentity:
    scopes = (
        ("synthetic",)
        if evidence_role == "synthetic_audit_only"
        else (
            ("train", "tune")
            if evidence_role == "pilot_audit_only"
            else ("train", "tune", "holdout")
        )
    )
    return RunEvidenceIdentity(
        schema_version="1.0",
        protocol_version="hypersca-methods-v3.0",
        protocol_identity=protocol_identity,
        claim_id="bridge",
        benchmark_id="spatial_perturbation_bridge",
        data_scopes=scopes,
        data_split_seed=11,
        model_seed=model_seed,
        data_split_identity_sha256=split_identity,
        statistical_unit_schema="bridge_prediction_unit_v1",
        statistical_unit_identity_sha256=canonical_sha256(unit_record),
        analysis_identity_sha256=canonical_sha256(analysis_record),
        input_identity_sha256=input_identity,
        config_identity_sha256=config_identity,
        code_identity_sha256=code_identity,
        evidence_role=evidence_role,
    )


def _verified_twice(
    output: Path, identity: RunEvidenceIdentity
) -> VerifiedRunEvidence:
    first = verify_spatial_perturbation_evidence_bundle(
        output, expected_identity=identity
    )
    second = verify_run_evidence_bundle(
        output, expected_identity=identity
    )
    if first != second:
        raise RunEvidenceError(
            "invalid_artifact", "published bridge evidence changed during replay"
        )
    return second


def _semantic_json(payload: bytes, *, label: str) -> dict[str, object]:
    def unique_pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise RunEvidenceError(
                    "invalid_artifact", f"bridge {label} contains duplicate fields"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
        if type(value) is not dict or canonical_json_bytes(value) != payload:
            raise RunEvidenceError(
                "invalid_artifact", f"bridge {label} is not canonical JSON"
            )
        return cast(dict[str, object], value)
    except RunEvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, OverflowError) as error:
        raise RunEvidenceError(
            "invalid_artifact", f"bridge {label} is invalid JSON"
        ) from error


def _prediction_payload_from_csv(
    payloads: dict[str, bytes], *, origin: str
) -> tuple[bytes, dict[str, list[dict[str, object]]]]:
    predictions: dict[str, list[dict[str, object]]] = {}
    for method_id, relative_path in _METHOD_TO_ARTIFACT.items():
        try:
            text = payloads[relative_path].decode("utf-8")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if tuple(reader.fieldnames or ()) != (
                "unit_id",
                "endpoint",
                "predicted_effect",
                "effect_units",
            ):
                raise ValueError("prediction columns changed")
            rows: list[dict[str, object]] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("prediction row shape changed")
                predicted_effect = float(cast(str, row["predicted_effect"]))
                if not math.isfinite(predicted_effect):
                    raise ValueError("prediction is non-finite")
                rows.append(
                    {
                        "unit_id": row["unit_id"],
                        "endpoint": row["endpoint"],
                        "predicted_effect": predicted_effect,
                        "effect_units": row["effect_units"],
                    }
                )
            if not rows:
                raise ValueError("prediction table is empty")
            keys = tuple((row["unit_id"], row["endpoint"]) for row in rows)
            if len(set(keys)) != len(keys):
                raise ValueError("prediction table contains duplicate units")
            predictions[method_id] = rows
        except (KeyError, UnicodeError, csv.Error, TypeError, ValueError) as error:
            raise RunEvidenceError(
                "invalid_artifact", "bridge prediction artifact is invalid"
            ) from error
    mapping = {
        "schema_version": "1.0",
        "origin": origin,
        "predictions": predictions,
    }
    payload = canonical_json_bytes(mapping)
    return payload, predictions


_MAX_SEMANTIC_ARTIFACT_BYTES = 32 * 1024 * 1024
_SEMANTIC_READ_CHUNK_BYTES = 1024 * 1024


def _open_semantic_directory_bound(
    path: Path,
) -> tuple[int, tuple[tuple[int, int, int, int], ...]]:
    absolute = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    directory_chain: list[tuple[int, int, int, int]] = []
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise OSError("bridge semantic directory path is not exact")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(next_descriptor)
            after = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            axes_before = (
                before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
                before.st_mtime_ns, before.st_ctime_ns,
            )
            axes_opened = (
                opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink,
                opened.st_mtime_ns, opened.st_ctime_ns,
            )
            axes_after = (
                after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
                after.st_mtime_ns, after.st_ctime_ns,
            )
            if not axes_before == axes_opened == axes_after:
                os.close(next_descriptor)
                raise OSError("bridge semantic directory path changed")
            directory_chain.append(
                (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
            )
            os.close(descriptor)
            descriptor = next_descriptor
        result = descriptor
        descriptor = -1
        return result, tuple(directory_chain)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_semantic_artifacts_bound(
    evidence: VerifiedRunEvidence,
) -> dict[str, bytes]:
    """Read the complete published bundle through one bound directory fd."""

    records = {item.relative_path: item for item in evidence.artifacts}
    relative_paths = (*_SUCCESS_ARTIFACTS, "run_manifest.json", "method_status.json")
    for relative_path in _SUCCESS_ARTIFACTS:
        record = records[relative_path]
        if (
            type(record.size_bytes) is not int
            or record.size_bytes < 0
            or record.size_bytes > _MAX_SEMANTIC_ARTIFACT_BYTES
        ):
            raise RunEvidenceError(
                "invalid_artifact", "bridge semantic artifact size exceeds 32 MiB"
            )
    artifact_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    artifact_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd, directory_chain = _open_semantic_directory_bound(
            evidence.output_dir
        )
    except OSError as error:
        raise RunEvidenceError(
            "invalid_artifact", "bridge semantic output directory cannot be bound"
        ) from error
    descriptors: dict[str, int] = {}
    before_stats: dict[str, os.stat_result] = {}
    try:
        declared_total = 0
        for relative_path in relative_paths:
            try:
                artifact_fd = os.open(
                    relative_path, artifact_flags, dir_fd=directory_fd
                )
            except OSError as error:
                raise RunEvidenceError(
                    "invalid_artifact", "bridge semantic artifact cannot be opened"
                ) from error
            descriptors[relative_path] = artifact_fd
            before = os.fstat(artifact_fd)
            before_stats[relative_path] = before
            expected_size = (
                records[relative_path].size_bytes
                if relative_path in records
                else before.st_size
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_size
                or before.st_size < 0
                or before.st_size > _MAX_SEMANTIC_ARTIFACT_BYTES
            ):
                raise RunEvidenceError(
                    "invalid_artifact",
                    "bridge semantic artifact is not an exact regular file",
                )
            declared_total += before.st_size
            if declared_total > _MAX_SEMANTIC_ARTIFACT_BYTES:
                raise RunEvidenceError(
                    "invalid_artifact", "bridge semantic bundle exceeds 32 MiB"
                )
        payloads: dict[str, bytes] = {}
        for relative_path in relative_paths:
            artifact_fd = descriptors[relative_path]
            before = before_stats[relative_path]
            semantic_record = records.get(relative_path)
            try:
                chunks: list[bytes] = []
                remaining = before.st_size
                while remaining:
                    chunk = os.read(
                        artifact_fd, min(_SEMANTIC_READ_CHUNK_BYTES, remaining)
                    )
                    if not chunk:
                        raise RunEvidenceError(
                            "invalid_artifact", "bridge semantic artifact was truncated"
                        )
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if os.read(artifact_fd, 1):
                    raise RunEvidenceError(
                        "invalid_artifact", "bridge semantic artifact grew while read"
                    )
                after = os.fstat(artifact_fd)
                if (
                    (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size)
                    != (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size)
                ):
                    raise RunEvidenceError(
                        "invalid_artifact", "bridge semantic artifact changed while read"
                    )
                after_path = os.stat(
                    relative_path, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    after_path.st_dev, after_path.st_ino, after_path.st_mode,
                    after_path.st_nlink, after_path.st_size,
                ) != (
                    after.st_dev, after.st_ino, after.st_mode,
                    after.st_nlink, after.st_size,
                ):
                    raise RunEvidenceError(
                        "invalid_artifact",
                        "bridge semantic artifact path changed while read",
                    )
                payload = b"".join(chunks)
                if (
                    semantic_record is not None
                    and hashlib.sha256(payload).hexdigest()
                    != semantic_record.sha256
                ):
                    raise RunEvidenceError(
                        "invalid_artifact", "bridge semantic artifact hash changed"
                    )
                payloads[relative_path] = payload
            except OSError as error:
                raise RunEvidenceError(
                    "invalid_artifact", "bridge semantic artifact cannot be read"
                ) from error
        try:
            rebound_fd, rebound_chain = _open_semantic_directory_bound(
                evidence.output_dir
            )
        except OSError as error:
            raise RunEvidenceError(
                "invalid_artifact", "bridge semantic directory path changed"
            ) from error
        try:
            bound = os.fstat(directory_fd)
            rebound = os.fstat(rebound_fd)
            if (
                bound.st_dev, bound.st_ino, bound.st_mode, bound.st_nlink,
            ) != (
                rebound.st_dev, rebound.st_ino, rebound.st_mode, rebound.st_nlink,
            ) or rebound_chain != directory_chain:
                raise RunEvidenceError(
                    "invalid_artifact", "bridge semantic directory path changed"
                )
        finally:
            os.close(rebound_fd)
    finally:
        for artifact_fd in descriptors.values():
            os.close(artifact_fd)
        os.close(directory_fd)
    return payloads


_ANALYSIS_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "claim_id",
        "method_id",
        "protocol_identity_sha256",
        "input_identity_sha256",
        "data_identity_sha256",
        "split_identity_sha256",
        "statistical_unit_identity_sha256",
        "code_identity_sha256",
        "prediction_bytes_sha256",
        "prediction_bundle_identity_sha256",
        "model_seed",
        "origin",
        "evidence_role",
        "neighbor_max_rank",
        "neighbor_bands",
        "neighbor_input_sha256",
        "neighbor_units_sha256",
        "eligibility_identity_sha256",
        "standardizer_identity_sha256",
        "comparator_budget_identity_sha256",
        "comparator_budgets",
        "support_identity_sha256",
        "support_record",
        "observed_effect_projection_identity_sha256",
        "observed_effect_projection",
        "synthetic_fixture_identity_sha256",
        "scoring_identities",
        "artifact_identities_sha256",
        "analysis_contract",
        "analysis_contract_identity_sha256",
    }
)


def _exact_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_analysis_record(
    analysis: dict[str, object], evidence: VerifiedRunEvidence
) -> None:
    sha_fields = (
        "protocol_identity_sha256",
        "input_identity_sha256",
        "data_identity_sha256",
        "split_identity_sha256",
        "statistical_unit_identity_sha256",
        "code_identity_sha256",
        "prediction_bytes_sha256",
        "prediction_bundle_identity_sha256",
        "neighbor_input_sha256",
        "neighbor_units_sha256",
        "eligibility_identity_sha256",
        "standardizer_identity_sha256",
        "comparator_budget_identity_sha256",
        "support_identity_sha256",
        "observed_effect_projection_identity_sha256",
        "analysis_contract_identity_sha256",
        "synthetic_fixture_identity_sha256",
    )
    scoring = analysis.get("scoring_identities")
    if (
        set(analysis) != _ANALYSIS_RECORD_FIELDS
        or analysis.get("schema_version") != "1.0"
        or analysis.get("claim_id") != "bridge"
        or analysis.get("method_id") != "hypersca"
        or analysis.get("origin") != "synthetic_fixture"
        or analysis.get("evidence_role") != "synthetic_audit_only"
        or type(analysis.get("model_seed")) is not int
        or analysis.get("model_seed") != evidence.identity.model_seed
        or type(analysis.get("neighbor_max_rank")) is not int
        or analysis.get("neighbor_max_rank") != 60
        or analysis.get("neighbor_bands")
        != ["proximal", "local", "transition", "distal"]
        or any(not _exact_sha(analysis.get(field)) for field in sha_fields)
        or type(scoring) is not list
        or len(scoring) != len(_METHOD_TO_ARTIFACT)
        or any(not _exact_sha(item) for item in cast(list[object], scoring))
        or analysis.get("protocol_identity_sha256")
        != evidence.identity.protocol_identity
        or analysis.get("input_identity_sha256")
        != evidence.identity.input_identity_sha256
        or analysis.get("split_identity_sha256")
        != evidence.identity.data_split_identity_sha256
        or analysis.get("statistical_unit_identity_sha256")
        != evidence.identity.statistical_unit_identity_sha256
        or analysis.get("code_identity_sha256")
        != evidence.identity.code_identity_sha256
        or analysis.get("prediction_bundle_identity_sha256")
        != evidence.identity.config_identity_sha256
    ):
        raise RunEvidenceError(
            "invalid_identity", "bridge analysis record is not the exact typed schema"
        )


_BUDGET_FIELDS = frozenset(
    {
        "method_id", "geometry", "parameter_count", "optimizer_family",
        "max_updates", "early_stopping_patience", "tuning_trials",
        "data_identity_sha256", "gene_identity_sha256",
        "spatial_graph_identity_sha256", "propagation_identity_sha256", "seed",
    }
)


def _verify_analysis_semantic_axes(
    payloads: dict[str, bytes],
    *,
    split_mapping: dict[str, object],
    analysis: dict[str, object],
) -> None:
    from src.evaluation.spatial_perturbation_comparators import (
        BridgeModelBudget,
        SpatialPerturbationComparatorError,
        bridge_model_budget_to_mapping,
        validate_bridge_comparator_budgets,
    )

    budgets = analysis.get("comparator_budgets")
    if (
        type(budgets) is not list
        or len(budgets) != 2
        or any(type(item) is not dict or set(item) != _BUDGET_FIELDS for item in budgets)
    ):
        raise RunEvidenceError("invalid_identity", "bridge comparator budgets are invalid")
    try:
        reconstructed = tuple(
            BridgeModelBudget(
                method_id=cast(str, item["method_id"]),
                geometry=cast(str, item["geometry"]),
                parameter_count=cast(int, item["parameter_count"]),
                optimizer_family=cast(str, item["optimizer_family"]),
                max_updates=cast(int, item["max_updates"]),
                early_stopping_patience=cast(int, item["early_stopping_patience"]),
                tuning_trials=cast(int, item["tuning_trials"]),
                data_identity_sha256=cast(str, item["data_identity_sha256"]),
                gene_identity_sha256=cast(str, item["gene_identity_sha256"]),
                spatial_graph_identity_sha256=cast(
                    str, item["spatial_graph_identity_sha256"]
                ),
                propagation_identity_sha256=cast(
                    str, item["propagation_identity_sha256"]
                ),
                seed=cast(int, item["seed"]),
            )
            for item in cast(list[dict[str, object]], budgets)
        )
        validate_bridge_comparator_budgets(reconstructed[0], reconstructed[1])
        canonical_budgets = [
            bridge_model_budget_to_mapping(item) for item in reconstructed
        ]
    except (KeyError, TypeError, ValueError, SpatialPerturbationComparatorError) as error:
        raise RunEvidenceError(
            "invalid_identity", "bridge comparator budgets failed Task8 replay"
        ) from error
    support_record = {
        "eligibility_identity_sha256": analysis.get("eligibility_identity_sha256"),
        "standardizer_identity_sha256": analysis.get("standardizer_identity_sha256"),
        "comparator_budgets": canonical_budgets,
    }
    try:
        synthetic_identity = _synthetic_fixture_identity_sha256(
            split_mapping,
            data_identity=cast(str, analysis.get("data_identity_sha256")),
        )
        neighbor_input_identity = _synthetic_raw_neighbor_input_sha256()
        expected_analysis_contract = _analysis_contract_record(analysis)
    except (BridgePredictorContractError, TypeError, ValueError) as error:
        raise RunEvidenceError(
            "invalid_identity", "bridge synthetic fixture provenance changed"
        ) from error
    if (
        canonical_budgets != budgets
        or any(
            item.data_identity_sha256 != analysis.get("data_identity_sha256")
            or item.seed != analysis.get("model_seed")
            for item in reconstructed
        )
        or analysis.get("comparator_budget_identity_sha256")
        != canonical_sha256(canonical_budgets)
        or type(analysis.get("support_record")) is not dict
        or analysis.get("support_record") != support_record
        or analysis.get("support_identity_sha256") != canonical_sha256(support_record)
        or analysis.get("neighbor_units_sha256")
        != hashlib.sha256(payloads["neighbor_units.csv"]).hexdigest()
        or analysis.get("neighbor_input_sha256") != neighbor_input_identity
        or type(analysis.get("analysis_contract")) is not dict
        or analysis.get("analysis_contract") != expected_analysis_contract
        or analysis.get("analysis_contract_identity_sha256")
        != canonical_sha256(expected_analysis_contract)
        or analysis.get("synthetic_fixture_identity_sha256") != synthetic_identity
    ):
        raise RunEvidenceError(
            "invalid_identity", "bridge analysis semantic axes changed"
        )


def _validated_observed_projection(
    analysis: dict[str, object],
) -> dict[tuple[str, str], float]:
    projection = analysis.get("observed_effect_projection")
    projection_fields = {
        "schema_version", "data_identity_sha256", "split_identity_sha256",
        "eligibility_identity_sha256", "standardizer_identity_sha256", "effects",
    }
    if type(projection) is not dict or set(projection) != projection_fields:
        raise RunEvidenceError("invalid_identity", "bridge observed projection is invalid")
    frozen = cast(dict[str, object], projection)
    effects = frozen.get("effects")
    if (
        frozen.get("schema_version") != "bridge_observed_effect_projection_v1"
        or frozen.get("data_identity_sha256") != analysis.get("data_identity_sha256")
        or frozen.get("split_identity_sha256") != analysis.get("split_identity_sha256")
        or frozen.get("eligibility_identity_sha256")
        != analysis.get("eligibility_identity_sha256")
        or frozen.get("standardizer_identity_sha256")
        != analysis.get("standardizer_identity_sha256")
        or type(effects) is not list
        or not effects
        or len(effects) > _OBSERVED_PROJECTION_MAX_EFFECTS
    ):
        raise RunEvidenceError("invalid_identity", "bridge observed projection axes changed")
    observed: dict[tuple[str, str], float] = {}
    canonical_rows: list[dict[str, object]] = []
    try:
        for item in cast(list[object], effects):
            if type(item) is not dict or set(item) != {
                "unit_id", "endpoint", "treatment_mean_hex",
                "safe_control_mean_hex", "observed_delta_hex",
            }:
                raise ValueError("projection row fields changed")
            row = cast(dict[str, object], item)
            unit_id = row.get("unit_id")
            endpoint = row.get("endpoint")
            raw_delta = row.get("observed_delta_hex")
            raw_treatment = row.get("treatment_mean_hex")
            raw_control = row.get("safe_control_mean_hex")
            if (
                not _exact_sha(unit_id)
                or endpoint not in ("neighbor", "own")
                or type(raw_delta) is not str
                or type(raw_treatment) is not str
                or type(raw_control) is not str
            ):
                raise ValueError("projection row types changed")
            delta = float.fromhex(raw_delta)
            treatment = float.fromhex(raw_treatment)
            control = float.fromhex(raw_control)
            if (
                any(
                    not math.isfinite(value) or abs(value) > 1.0e12
                    for value in (delta, treatment, control)
                )
                or delta.hex() != raw_delta
                or treatment.hex() != raw_treatment
                or control.hex() != raw_control
                or (treatment - control).hex() != raw_delta
            ):
                raise ValueError("projection delta is invalid")
            key = (cast(str, unit_id), cast(str, endpoint))
            if key in observed:
                raise ValueError("projection rows are duplicated")
            observed[key] = delta
            canonical_rows.append(cast(dict[str, object], item))
    except (TypeError, ValueError, OverflowError) as error:
        raise RunEvidenceError(
            "invalid_identity", "bridge observed projection rows are invalid"
        ) from error
    if canonical_rows != sorted(
        canonical_rows,
        key=lambda row: (cast(str, row["endpoint"]), cast(str, row["unit_id"])),
    ):
        raise RunEvidenceError("invalid_identity", "bridge observed projection order changed")
    projection_identity = canonical_sha256(frozen)
    if (
        analysis.get("observed_effect_projection_identity_sha256")
        != projection_identity
        or analysis.get("input_identity_sha256")
        != _run_input_identity_sha256(
            data_identity_sha256=cast(str, analysis.get("data_identity_sha256")),
            observed_projection_identity_sha256=projection_identity,
        )
    ):
        raise RunEvidenceError("invalid_identity", "bridge observed projection identity changed")
    return observed


def _parse_exact_csv(
    payload: bytes, *, columns: tuple[str, ...], label: str
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError("columns changed")
        rows: list[dict[str, str]] = []
        for row in reader:
            if (
                len(rows) >= 300_000
                or None in row
                or any(type(value) is not str for value in row.values())
            ):
                raise ValueError("row shape changed")
            rows.append(cast(dict[str, str], row))
        if not rows:
            raise ValueError("table is empty")
        return rows
    except (UnicodeError, csv.Error, TypeError, ValueError) as error:
        raise RunEvidenceError(
            "invalid_artifact", f"bridge {label} CSV is invalid"
        ) from error


def _exact_csv_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise RunEvidenceError("invalid_artifact", f"bridge {label} is invalid") from error
    if parsed < 0 or str(parsed) != value:
        raise RunEvidenceError("invalid_artifact", f"bridge {label} is invalid")
    return parsed


def _exact_csv_float(value: str, *, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RunEvidenceError("invalid_artifact", f"bridge {label} is invalid") from error
    if not math.isfinite(parsed) or abs(parsed) > 1.0e12:
        raise RunEvidenceError("invalid_artifact", f"bridge {label} is invalid")
    return parsed


def _verify_published_score_semantics(
    payloads: dict[str, bytes],
    *,
    split_mapping: dict[str, object],
    analysis: dict[str, object],
    prediction_rows: dict[str, list[dict[str, object]]],
) -> tuple[list[str], dict[str, object]]:
    from src.evaluation.spatial_perturbation_scoring import (
        BridgeEffect,
        SpatialPerturbationScoringError,
        replay_published_bridge_effect_units,
    )

    published_observed = _validated_observed_projection(analysis)
    evaluation_animals = split_mapping.get("evaluation_animals")
    primary_units = split_mapping.get("primary_units")
    parents = split_mapping.get("perturbation_parents")
    neighbour_table = split_mapping.get("neighbour_table")
    if (
        type(evaluation_animals) is not list
        or not evaluation_animals
        or any(type(item) is not str for item in evaluation_animals)
        or type(primary_units) is not list
        or type(parents) is not list
        or type(neighbour_table) is not dict
        or set(cast(dict[object, object], neighbour_table))
        != {"schema", "relation_count", "identity_sha256"}
        or cast(dict[object, object], neighbour_table).get("schema")
        != "bridge_neighbour_artifact_v1"
        or type(
            cast(dict[object, object], neighbour_table).get("relation_count")
        ) is not int
        or not 0
        <= cast(
            int, cast(dict[object, object], neighbour_table).get("relation_count")
        )
        <= 2_000_000
        or not _exact_sha(
            cast(dict[object, object], neighbour_table).get("identity_sha256")
        )
    ):
        raise RunEvidenceError("invalid_artifact", "bridge split score axes are invalid")
    animals = set(cast(list[str], evaluation_animals))
    unit_by_id: dict[str, dict[str, object]] = {}
    for item in cast(list[object], primary_units):
        if type(item) is not dict or set(item) != {
            "unit_id", "animal_id", "perturbation_id", "target_gene",
            "neighbour_cell_type", "band",
        }:
            raise RunEvidenceError("invalid_artifact", "bridge split units are invalid")
        record = cast(dict[str, object], item)
        if record.get("animal_id") in animals:
            unit_id = record.get("unit_id")
            if not _exact_sha(unit_id) or unit_id in unit_by_id:
                raise RunEvidenceError("invalid_artifact", "bridge split units are invalid")
            unit_by_id[cast(str, unit_id)] = record
    parent_by_id: dict[str, dict[str, object]] = {}
    for item in cast(list[object], parents):
        if type(item) is not dict or set(item) != {
            "parent_id", "animal_id", "perturbation_id", "target_gene"
        }:
            raise RunEvidenceError("invalid_artifact", "bridge split parents are invalid")
        record = cast(dict[str, object], item)
        if record.get("animal_id") in animals:
            parent_id = record.get("parent_id")
            if not _exact_sha(parent_id) or parent_id in parent_by_id:
                raise RunEvidenceError("invalid_artifact", "bridge split parents are invalid")
            parent_by_id[cast(str, parent_id)] = record
    if not unit_by_id or not parent_by_id:
        raise RunEvidenceError("invalid_artifact", "bridge split score units are empty")
    contexts = tuple(
        sorted({
            (
                cast(str, unit["animal_id"]),
                cast(str, unit["perturbation_id"]),
                cast(str, unit["neighbour_cell_type"]),
                cast(str, unit["target_gene"]),
            )
            for unit in unit_by_id.values()
        })
    )
    columns = (
        "method_id", "unit_id", "endpoint", "animal_id", "perturbation_id",
        "gene_name", "neighbor_cell_type", "band", "observed_delta",
        "predicted_delta", "effect_identity_sha256",
        "evaluation_neighbor_unit_count", "evaluation_calibration_context_count",
    )
    rows = _parse_exact_csv(
        payloads["primary_metric_units.csv"], columns=columns, label="metric units"
    )
    effects_by_method: dict[str, list[BridgeEffect]] = {
        method: [] for method in _METHOD_TO_ARTIFACT
    }
    declared_counts: set[tuple[int, int]] = set()
    try:
        for row in rows:
            method = row["method_id"]
            if method not in effects_by_method:
                raise SpatialPerturbationScoringError("unknown published method")
            endpoint = row["endpoint"]
            unit_id = row["unit_id"]
            if endpoint == "neighbor":
                frozen_unit = unit_by_id.get(unit_id)
                expected_context = None if frozen_unit is None else (
                    frozen_unit["animal_id"], frozen_unit["perturbation_id"],
                    frozen_unit["target_gene"], frozen_unit["neighbour_cell_type"],
                    frozen_unit["band"],
                )
            elif endpoint == "own":
                frozen_parent = parent_by_id.get(unit_id)
                expected_context = None if frozen_parent is None else (
                    frozen_parent["animal_id"], frozen_parent["perturbation_id"],
                    frozen_parent["target_gene"], "own", "own",
                )
            else:
                expected_context = None
            actual_context = (
                row["animal_id"], row["perturbation_id"], row["gene_name"],
                row["neighbor_cell_type"], row["band"],
            )
            if expected_context is None or actual_context != expected_context:
                raise SpatialPerturbationScoringError(
                    "published effect differs from its split unit"
                )
            observed = _exact_csv_float(row["observed_delta"], label="observed delta")
            predicted = _exact_csv_float(row["predicted_delta"], label="predicted delta")
            effects_by_method[method].append(
                BridgeEffect(
                    unit_id, endpoint, row["animal_id"], row["perturbation_id"],
                    row["gene_name"], row["neighbor_cell_type"], row["band"],
                    observed, predicted, row["effect_identity_sha256"],
                )
            )
            declared_counts.add((
                _exact_csv_int(
                    row["evaluation_neighbor_unit_count"], label="evaluation count"
                ),
                _exact_csv_int(
                    row["evaluation_calibration_context_count"],
                    label="calibration context count",
                ),
            ))
    except (KeyError, TypeError, SpatialPerturbationScoringError) as error:
        raise RunEvidenceError(
            "invalid_artifact", "bridge published effect units are invalid"
        ) from error
    if declared_counts != {(len(unit_by_id), len(contexts))}:
        raise RunEvidenceError("invalid_artifact", "bridge score denominators changed")
    reference_observed: dict[tuple[str, str], float] | None = None
    replay_by_method: dict[str, object] = {}
    try:
        for method in _METHOD_TO_ARTIFACT:
            effects = tuple(effects_by_method[method])
            effect_keys = {(effect.unit_id, effect.endpoint) for effect in effects}
            expected_predictions = {
                (cast(str, row["unit_id"]), cast(str, row["endpoint"])):
                cast(float, row["predicted_effect"])
                for row in prediction_rows[method]
            }
            if effect_keys != set(expected_predictions) or any(
                effect.predicted_delta
                != expected_predictions[(effect.unit_id, effect.endpoint)]
                for effect in effects
            ):
                raise SpatialPerturbationScoringError(
                    "published effects differ from prediction CSV"
                )
            observed_by_unit = {
                (effect.unit_id, effect.endpoint): effect.observed_delta
                for effect in effects
            }
            if reference_observed is None:
                reference_observed = observed_by_unit
            elif observed_by_unit != reference_observed:
                raise SpatialPerturbationScoringError(
                    "observed effects differ between methods"
                )
            replay_by_method[method] = replay_published_bridge_effect_units(
                effects,
                expected_calibration_contexts=contexts,
                evaluation_neighbor_unit_count=len(unit_by_id),
                split_identity_sha256=cast(str, analysis["split_identity_sha256"]),
                neighbour_table_identity_sha256=cast(
                    str, cast(dict[object, object], neighbour_table)["identity_sha256"]
                ),
                eligibility_identity_sha256=cast(
                    str, analysis["eligibility_identity_sha256"]
                ),
                standardizer_identity_sha256=cast(
                    str, analysis["standardizer_identity_sha256"]
                ),
            )
    except (KeyError, TypeError, SpatialPerturbationScoringError) as error:
        raise RunEvidenceError(
            "invalid_artifact", "bridge Task7 metric replay failed"
        ) from error
    if reference_observed != published_observed:
        raise RunEvidenceError(
            "invalid_artifact", "bridge observed effects differ from replay projection"
        )
    primary = _semantic_json(
        payloads["primary_metric_summary.json"], label="primary metric summary"
    )
    method_fields = {
        "neighbor_effect_rmse", "own_effect_rmse", "coverage", "abstention",
        "distance_calibration_eligible_pairs",
        "distance_calibration_total_contexts", "distance_calibration_coverage",
        "distance_calibration_abstention", "effect_table_identity_sha256",
        "scoring_identity_sha256",
    }
    if set(primary) != {"schema_version", *_METHOD_TO_ARTIFACT} or primary.get(
        "schema_version"
    ) != "1.0":
        raise RunEvidenceError("invalid_artifact", "bridge primary summary is invalid")
    scoring_identities: list[str] = []
    for method in _METHOD_TO_ARTIFACT:
        raw_record = primary.get(method)
        replay = replay_by_method[method]
        if type(raw_record) is not dict or set(raw_record) != method_fields:
            raise RunEvidenceError("invalid_artifact", "bridge primary method is invalid")
        record = cast(dict[str, object], raw_record)
        expected = {
            field: getattr(replay, field)
            for field in method_fields
        }
        float_fields = {
            "neighbor_effect_rmse", "own_effect_rmse", "coverage", "abstention",
            "distance_calibration_coverage", "distance_calibration_abstention",
        }
        int_fields = {
            "distance_calibration_eligible_pairs",
            "distance_calibration_total_contexts",
        }
        if (
            any(type(record.get(field)) is not float for field in float_fields)
            or any(type(record.get(field)) is not int for field in int_fields)
            or not _exact_sha(record.get("effect_table_identity_sha256"))
            or not _exact_sha(record.get("scoring_identity_sha256"))
            or record != expected
        ):
            raise RunEvidenceError(
                "invalid_artifact", "bridge primary metrics differ from unit replay"
            )
        scoring_identities.append(cast(str, expected["scoring_identity_sha256"]))
    secondary_rows = _parse_exact_csv(
        payloads["secondary_metrics.csv"],
        columns=("method_id", "metric_id", "value"),
        label="secondary metrics",
    )
    expected_secondary = {
        (method, metric): getattr(replay_by_method[method], metric)
        for method in _METHOD_TO_ARTIFACT
        for metric in (
            "neighbor_effect_pcc",
            "distance_decay_calibration_error",
            "effect_sign_accuracy",
        )
    }
    actual_secondary: dict[tuple[str, str], float | None] = {}
    for row in secondary_rows:
        key = (row["method_id"], row["metric_id"])
        if key not in expected_secondary or key in actual_secondary:
            raise RunEvidenceError("invalid_artifact", "bridge secondary metric key changed")
        value = row["value"]
        actual_secondary[key] = (
            None if value == "" else _exact_csv_float(value, label="secondary metric")
        )
    if actual_secondary != expected_secondary:
        raise RunEvidenceError(
            "invalid_artifact", "bridge secondary metrics differ from unit replay"
        )
    return scoring_identities, primary


def _verify_code_owned_synthetic_replay(
    payloads: dict[str, bytes],
    *,
    split_mapping: dict[str, object],
    analysis: dict[str, object],
    prediction_rows: dict[str, list[dict[str, object]]],
) -> list[str]:
    """Replay Task4-7 from code-owned raw fixture data and published predictions."""

    from src.evaluation.spatial_perturbation_scoring import (
        BridgePrediction,
        SpatialPerturbationScoringError,
        bridge_score_to_mapping,
        score_bridge_predictions,
    )

    fixture = _build_code_owned_synthetic_fixture()
    try:
        expected_fixture_identity = _synthetic_fixture_identity_sha256(
            cast(dict[str, object], fixture["split_mapping"]),
            data_identity=cast(str, fixture["data_identity_sha256"]),
        )
    except (BridgePredictorContractError, TypeError, ValueError) as error:
        raise RunEvidenceError(
            "invalid_identity", "code-owned synthetic fixture replay failed"
        ) from error
    if (
        split_mapping != fixture["split_mapping"]
        or payloads["neighbor_units.csv"] != fixture["neighbor_bytes"]
        or analysis.get("data_identity_sha256")
        != fixture["data_identity_sha256"]
        or analysis.get("neighbor_input_sha256")
        != hashlib.sha256(cast(bytes, fixture["raw_neighbor_bytes"])).hexdigest()
        or analysis.get("neighbor_units_sha256")
        != hashlib.sha256(cast(bytes, fixture["neighbor_bytes"])).hexdigest()
        or analysis.get("eligibility_identity_sha256")
        != cast(dict[str, object], fixture["eligibility_mapping"]).get(
            "eligibility_identity_sha256"
        )
        or analysis.get("standardizer_identity_sha256")
        != cast(dict[str, object], fixture["standardizer_mapping"]).get(
            "training_identity_sha256"
        )
        or analysis.get("observed_effect_projection")
        != fixture["observed_projection"]
        or analysis.get("observed_effect_projection_identity_sha256")
        != canonical_sha256(fixture["observed_projection"])
        or analysis.get("synthetic_fixture_identity_sha256")
        != expected_fixture_identity
    ):
        raise RunEvidenceError(
            "invalid_artifact", "bridge differs from code-owned Task4-7 replay"
        )
    scoring_identities: list[str] = []
    try:
        for method in _METHOD_TO_ARTIFACT:
            predictions = tuple(
                BridgePrediction(
                    cast(str, row["unit_id"]),
                    cast(str, row["endpoint"]),
                    cast(float, row["predicted_effect"]),
                )
                for row in prediction_rows[method]
            )
            score = score_bridge_predictions(
                fixture["expression"],
                cell_ids=fixture["cell_ids"],
                gene_names=fixture["gene_names"],
                standardizer=fixture["standardizer"],
                eligibility=fixture["eligibility"],
                predictions=predictions,
            )
            score_mapping = bridge_score_to_mapping(
                score,
                expression=fixture["expression"],
                cell_ids=fixture["cell_ids"],
                gene_names=fixture["gene_names"],
                standardizer=fixture["standardizer"],
                eligibility=fixture["eligibility"],
                predictions=predictions,
            )
            scoring_identities.append(
                cast(str, score_mapping["scoring_identity_sha256"])
            )
    except (KeyError, TypeError, ValueError, SpatialPerturbationScoringError) as error:
        raise RunEvidenceError(
            "invalid_artifact", "code-owned Task7 prediction replay failed"
        ) from error
    return scoring_identities


def verify_spatial_perturbation_evidence_bundle(
    path: Path | str, *, expected_identity: RunEvidenceIdentity | None = None
) -> VerifiedRunEvidence:
    """Replay generic controls plus the exact Task9 bridge semantics."""

    first = verify_run_evidence_bundle(path, expected_identity=expected_identity)
    if first.terminal_status != "completed":
        return first
    if (
        first.identity.claim_id != "bridge"
        or first.identity.benchmark_id != "spatial_perturbation_bridge"
    ):
        raise RunEvidenceError(
            "invalid_identity", "bridge semantic replay received another claim"
        )
    artifact_records = {item.relative_path: item for item in first.artifacts}
    if set(artifact_records) != set(_SUCCESS_ARTIFACTS):
        raise RunEvidenceError(
            "invalid_artifact", "completed bridge artifact schema is not exact"
        )
    for relative_path in _SUCCESS_ARTIFACTS:
        expected_media = (
            "application/json" if relative_path.endswith(".json") else "text/csv"
        )
        if artifact_records[relative_path].media_type != expected_media:
            raise RunEvidenceError(
                "invalid_artifact", "completed bridge artifact media type changed"
            )
    payloads = _read_semantic_artifacts_bound(first)

    claim = _semantic_json(payloads["claim_decision.json"], label="claim decision")
    if set(claim) != {
        "schema_version",
        "claim_id",
        "decision",
        "synthetic_fixture_identity_sha256",
        "analysis_record",
    }:
        raise RunEvidenceError(
            "invalid_artifact", "bridge claim decision fields are not exact"
        )
    analysis = claim["analysis_record"]
    if type(analysis) is not dict or canonical_sha256(analysis) != first.identity.analysis_identity_sha256:
        raise RunEvidenceError(
            "invalid_identity", "bridge semantic analysis identity changed"
        )
    analysis_record = cast(dict[str, object], analysis)
    _validate_analysis_record(analysis_record, first)
    summary = first.summary
    if (
        type(summary) is not MappingProxyType
        or set(summary) != {
            "claim_id", "evidence_role", "analysis_contract_identity_sha256",
            "prediction_bundle_identity_sha256", "scientific_claim_allowed",
        }
        or summary.get("analysis_contract_identity_sha256")
        != analysis_record.get("analysis_contract_identity_sha256")
    ):
        raise RunEvidenceError(
            "invalid_identity", "bridge completed summary contract changed"
        )
    if (
        claim["schema_version"] != "1.0"
        or claim["claim_id"] != "bridge"
        or claim["decision"] != "synthetic_audit_only_no_scientific_claim"
        or first.identity.evidence_role != "synthetic_audit_only"
        or first.identity.data_scopes != ("synthetic",)
        or analysis_record.get("origin") != "synthetic_fixture"
        or analysis_record.get("evidence_role") != "synthetic_audit_only"
        or claim["synthetic_fixture_identity_sha256"]
        != analysis_record.get("synthetic_fixture_identity_sha256")
    ):
        raise RunEvidenceError(
            "invalid_identity", "bridge claim is not the exact synthetic audit decision"
        )
    artifact_identities = analysis_record.get("artifact_identities_sha256")
    expected_semantic_paths = set(_SUCCESS_ARTIFACTS) - {"claim_decision.json"}
    if (
        type(artifact_identities) is not dict
        or set(artifact_identities) != expected_semantic_paths
        or artifact_identities
        != {
            relative_path: hashlib.sha256(payloads[relative_path]).hexdigest()
            for relative_path in sorted(expected_semantic_paths)
        }
    ):
        raise RunEvidenceError(
            "invalid_artifact", "bridge artifact semantic identities changed"
        )

    capability = _semantic_json(
        payloads["capability_record.json"], label="capability record"
    )
    if set(capability) != {
        "schema_version",
        "status",
        "evidence_role",
        "prediction_bundle_identity_sha256",
        "prediction_bundle",
        "synthetic_fixture_identity_sha256",
    }:
        raise RunEvidenceError(
            "invalid_artifact", "bridge capability record fields are not exact"
        )
    prediction_bundle = capability["prediction_bundle"]
    if type(prediction_bundle) is not dict:
        raise RunEvidenceError(
            "invalid_artifact", "bridge prediction bundle declaration is missing"
        )
    bundle_record = cast(dict[str, object], prediction_bundle)
    prediction_payload, prediction_rows = _prediction_payload_from_csv(
        payloads, origin=cast(str, bundle_record.get("origin"))
    )
    try:
        replayed_bundle = build_bridge_prediction_bundle(
            method_id=cast(str, bundle_record.get("method_id")),
            protocol_identity_sha256=cast(
                str, bundle_record.get("protocol_identity_sha256")
            ),
            data_identity_sha256=cast(
                str, bundle_record.get("data_identity_sha256")
            ),
            split_identity_sha256=cast(
                str, bundle_record.get("split_identity_sha256")
            ),
            statistical_unit_identity_sha256=cast(
                str, bundle_record.get("statistical_unit_identity_sha256")
            ),
            code_identity_sha256=cast(
                str, bundle_record.get("code_identity_sha256")
            ),
            model_seed=cast(int, bundle_record.get("model_seed")),
            prediction_schema=bundle_record.get("prediction_schema"),
            prediction_bytes=prediction_payload,
            origin=cast(str, bundle_record.get("origin")),
            evidence_role=cast(str, bundle_record.get("evidence_role")),
        )
        replayed_bundle_record = bridge_prediction_bundle_to_mapping(replayed_bundle)
    except (BridgePredictorContractError, TypeError, ValueError) as error:
        raise RunEvidenceError(
            "invalid_identity", "bridge prediction bundle cannot be reconstructed"
        ) from error
    bundle_identity = bundle_record.get("prediction_bundle_identity_sha256")
    if (
        set(bundle_record)
        != {
            "schema_version", "method_id", "protocol_identity_sha256",
            "data_identity_sha256", "split_identity_sha256",
            "statistical_unit_identity_sha256", "code_identity_sha256",
            "model_seed", "prediction_schema", "prediction_bytes_sha256",
            "origin", "evidence_role", "prediction_bundle_identity_sha256",
        }
        or bundle_record != replayed_bundle_record
        or bundle_identity != first.identity.config_identity_sha256
        or capability["prediction_bundle_identity_sha256"] != bundle_identity
        or capability["schema_version"] != "1.0"
        or capability["status"] != "synthetic_prediction_bundle_replayed"
        or capability["evidence_role"] != "synthetic_audit_only"
        or capability["synthetic_fixture_identity_sha256"]
        != claim["synthetic_fixture_identity_sha256"]
        or bundle_record.get("protocol_identity_sha256") != first.identity.protocol_identity
        or bundle_record.get("data_identity_sha256")
        != analysis_record.get("data_identity_sha256")
        or bundle_record.get("split_identity_sha256")
        != first.identity.data_split_identity_sha256
        or bundle_record.get("statistical_unit_identity_sha256")
        != first.identity.statistical_unit_identity_sha256
        or bundle_record.get("code_identity_sha256") != first.identity.code_identity_sha256
        or bundle_record.get("model_seed") != first.identity.model_seed
        or bundle_record.get("schema_version") != "1.0"
        or bundle_record.get("method_id") != "hypersca"
        or bundle_record.get("prediction_schema") != PREDICTION_SCHEMA
        or bundle_record.get("origin") != "synthetic_fixture"
        or bundle_record.get("evidence_role") != "synthetic_audit_only"
        or bundle_record.get("prediction_bytes_sha256")
        != hashlib.sha256(prediction_payload).hexdigest()
        or analysis_record.get("prediction_bytes_sha256")
        != bundle_record.get("prediction_bytes_sha256")
    ):
        raise RunEvidenceError(
            "invalid_identity", "bridge prediction bundle semantic identity changed"
        )

    resource = _semantic_json(payloads["resource_usage.json"], label="resource usage")
    if (
        set(resource)
        != {
            "schema_version", "mode", "maximum_bundle_bytes", "model_seed",
            "neighbor_max_rank", "neighbor_units_bytes", "prediction_payload_bytes",
            "model_fitted", "outcomes_used_only_by_task7_scoring_contract",
        }
        or resource.get("schema_version") != "1.0"
        or resource.get("mode") != "orchestration_only"
        or type(resource.get("model_seed")) is not int
        or resource.get("model_seed") != first.identity.model_seed
        or type(resource.get("neighbor_max_rank")) is not int
        or resource.get("neighbor_max_rank") != 60
        or resource.get("neighbor_units_bytes") != len(payloads["neighbor_units.csv"])
        or type(resource.get("maximum_bundle_bytes")) is not int
        or not 1
        <= cast(int, resource.get("maximum_bundle_bytes"))
        <= _MAX_SEMANTIC_ARTIFACT_BYTES
        or sum(len(payloads[relative_path]) for relative_path in (
            *_SUCCESS_ARTIFACTS, "run_manifest.json", "method_status.json"
        ))
        > cast(int, resource.get("maximum_bundle_bytes"))
        or type(resource.get("prediction_payload_bytes")) is not int
        or resource.get("prediction_payload_bytes") != len(prediction_payload)
        or type(resource.get("neighbor_units_bytes")) is not int
        or resource.get("model_fitted") is not False
        or resource.get("outcomes_used_only_by_task7_scoring_contract") is not True
    ):
        raise RunEvidenceError(
            "invalid_artifact", "bridge resource semantic record changed"
        )
    split_mapping = _semantic_json(
        payloads["split_manifest.json"], label="Task5 split manifest"
    )
    if (
        type(split_mapping) is not dict
        or split_mapping.get("split_identity_sha256")
        != first.identity.data_split_identity_sha256
        or _split_manifest_json_bytes(
            split_mapping, maximum_bytes=max(len(payloads["split_manifest.json"]), 1)
        )
        != payloads["split_manifest.json"]
    ):
        raise RunEvidenceError(
            "invalid_artifact", "bridge Task5 split semantic identity changed"
        )
    _verify_analysis_semantic_axes(
        payloads, split_mapping=split_mapping, analysis=analysis_record
    )
    code_owned_scoring_identities = _verify_code_owned_synthetic_replay(
        payloads,
        split_mapping=split_mapping,
        analysis=analysis_record,
        prediction_rows=prediction_rows,
    )
    scoring_identities, _ = _verify_published_score_semantics(
        payloads,
        split_mapping=split_mapping,
        analysis=analysis_record,
        prediction_rows=prediction_rows,
    )
    if (
        analysis_record.get("scoring_identities") != scoring_identities
        or scoring_identities != code_owned_scoring_identities
    ):
        raise RunEvidenceError(
            "invalid_artifact", "bridge scoring semantic identities changed"
        )
    return first


def _publish_failure(
    capability: BridgePredictorCapability,
    *,
    output_dir: Path | str,
    maximum_bundle_bytes: int,
) -> VerifiedRunEvidence:
    initial = bridge_predictor_capability_to_mapping(capability)
    code_identity = _source_identity()
    unit_record: dict[str, object] = {
        "units": ["bridge_predictor_capability_audit"]
    }
    split_identity = canonical_sha256(
        {"schema_version": "1.0", "split": "not_started_without_adapter"}
    )
    analysis_record: dict[str, object] = {
        "schema_version": "1.0",
        "claim_id": "bridge",
        "capability_identity_sha256": capability.capability_identity_sha256,
        "status": "method_adapter_not_executable",
    }
    identity = _identity(
        protocol_identity=capability.protocol_identity_sha256,
        split_identity=split_identity,
        unit_record=unit_record,
        analysis_record=analysis_record,
        input_identity=capability.registry_identity_sha256,
        config_identity=capability.capability_identity_sha256,
        code_identity=code_identity,
        model_seed=0,
        evidence_role="pilot_audit_only",
    )
    resource_record = {
        "schema_version": "1.0",
        "audit": "outcome_blind_predictor_capability",
        "capability_identity_sha256": capability.capability_identity_sha256,
        "registry_identity_sha256": capability.registry_identity_sha256,
        "protocol_identity_sha256": capability.protocol_identity_sha256,
        "code_identity_sha256": code_identity,
        "model_imported": False,
        "outcomes_read": False,
        "scientific_computation_performed": False,
    }
    publisher = RunEvidencePublisher.begin(
        output_dir=output_dir,
        identity=identity,
        statistical_unit_record=unit_record,
        required_artifacts=_FAILURE_ARTIFACTS,
        maximum_bundle_bytes=maximum_bundle_bytes,
    )
    try:
        publisher.add_bytes(
            "capability_record.json",
            canonical_json_bytes(initial),
            media_type="application/json",
        )
        publisher.add_bytes(
            "resource_usage.json",
            canonical_json_bytes(resource_record),
            media_type="application/json",
        )
        if (
            bridge_predictor_capability_to_mapping(capability) != initial
            or _source_identity() != code_identity
        ):
            raise BridgePredictorContractError(
                "capability or runner code identity changed before publication"
            )
        output = publisher.finalize_failure(
            status="method_adapter_not_executable",
            reason="no_preregistered_bridge_predictor_adapter",
        )
    except BaseException:
        if publisher.state == "staging":
            publisher.abort()
        raise
    return _verified_twice(output, identity)


def _validated_data_components(
    *,
    split_manifest: object,
    neighbor_cells: object,
    expression: object,
    cell_ids: object,
    gene_names: object,
) -> tuple[dict[str, object], Any, bytes, str]:
    import numpy as np
    import pandas as pd

    from src.evaluation.spatial_perturbation_neighbors import build_bridge_neighbors
    from src.evaluation.spatial_perturbation_split import (
        BridgeSplitManifest,
        freeze_bridge_neighbour_relation,
        freeze_bridge_neighbour_table,
        split_manifest_to_mapping,
    )

    if type(split_manifest) is not BridgeSplitManifest:
        raise BridgePredictorContractError(
            "split_manifest must be an exact BridgeSplitManifest"
        )
    validated_manifest = cast(BridgeSplitManifest, split_manifest)
    split_mapping = split_manifest_to_mapping(validated_manifest)
    if type(neighbor_cells) is not pd.DataFrame:
        raise BridgePredictorContractError(
            "neighbor_cells must be an exact pandas DataFrame"
        )
    neighbor_input_before = _raw_neighbor_csv_bytes(neighbor_cells)
    first_neighbors = build_bridge_neighbors(neighbor_cells)
    neighbor_input_after_first = _raw_neighbor_csv_bytes(neighbor_cells)
    if neighbor_input_after_first != neighbor_input_before:
        raise BridgePredictorContractError(
            "raw neighbor coordinate input changed during the first Task 6 validation"
        )
    neighbors = build_bridge_neighbors(neighbor_cells)
    neighbor_input_after = _raw_neighbor_csv_bytes(neighbor_cells)
    if (
        neighbor_input_after != neighbor_input_before
        or _csv_bytes(first_neighbors) != _csv_bytes(neighbors)
    ):
        raise BridgePredictorContractError(
            "raw neighbor coordinate input changed during Task 6 validation"
        )
    relations = tuple(
        freeze_bridge_neighbour_relation(*row)
        for row in neighbors.itertuples(index=False, name=None)
    )
    relation_table = freeze_bridge_neighbour_table(relations)
    if (
        relation_table.identity_sha256
        != validated_manifest.neighbour_table_identity_sha256
        or relation_table.relations != validated_manifest.neighbour_relations
    ):
        raise BridgePredictorContractError(
            "neighbor relation table differs from the Task 5 manifest"
        )
    neighbor_bytes = _csv_bytes(neighbors)
    if type(expression) is not np.ndarray or expression.dtype != np.dtype("float64"):
        raise BridgePredictorContractError(
            "expression must be an exact float64 numpy array for Task 7 replay"
        )
    if expression.ndim != 2 or not np.isfinite(expression).all():
        raise BridgePredictorContractError(
            "expression must be a finite two-dimensional Task 7 matrix"
        )
    if type(cell_ids) is not tuple or type(gene_names) is not tuple:
        raise BridgePredictorContractError(
            "cell_ids and gene_names must be exact ordered tuples"
        )
    if expression.shape != (len(cell_ids), len(gene_names)):
        raise BridgePredictorContractError(
            "expression axes must match exact cell and gene identities"
        )
    expression_bytes = np.ascontiguousarray(expression, dtype="<f8").tobytes()
    record = {
        "schema_version": "1.0",
        "split_identity_sha256": split_mapping["split_identity_sha256"],
        "neighbor_max_rank": 60,
        "neighbor_bands": ["proximal", "local", "transition", "distal"],
        "neighbor_input_sha256": hashlib.sha256(
            neighbor_input_before
        ).hexdigest(),
        "neighbor_units_sha256": hashlib.sha256(neighbor_bytes).hexdigest(),
        "expression_shape": list(expression.shape),
        "expression_sha256": hashlib.sha256(expression_bytes).hexdigest(),
        "cell_ids": list(cell_ids),
        "gene_names": list(gene_names),
    }
    data_identity = canonical_sha256(record)
    return split_mapping, neighbors, neighbor_bytes, data_identity


def bridge_run_data_identity_sha256(
    *,
    split_manifest: object,
    neighbor_cells: object,
    expression: object,
    cell_ids: object,
    gene_names: object,
) -> str:
    """Bind exact Task 5/6 inputs and ordered Task 7 matrix axes."""

    return _validated_data_components(
        split_manifest=split_manifest,
        neighbor_cells=neighbor_cells,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
    )[3]


def _build_code_owned_synthetic_fixture() -> dict[str, Any]:
    """Rebuild the frozen Task4-7 synthetic audit fixture from raw declarations."""

    import numpy as np
    import pandas as pd

    from src.evaluation.spatial_perturbation_neighbors import build_bridge_neighbors
    from src.evaluation.spatial_perturbation_registry import (
        BridgeCandidate,
        MetadataSummary,
        audit_bridge_capability,
    )
    from src.evaluation.spatial_perturbation_scoring import (
        build_bridge_observed_effect_projection,
        fit_train_control_standardizer,
        train_control_standardizer_to_mapping,
    )
    from src.evaluation.spatial_perturbation_split import (
        BridgeParentEvidence,
        BridgePrimaryUnitEvidence,
        BridgeSplitMetadata,
        BridgeSplitRow,
        build_bridge_eligibility_evidence,
        build_pilot_fold,
        eligibility_result_to_mapping,
        evaluate_bridge_eligibility,
        freeze_bridge_neighbour_relation,
        freeze_bridge_neighbour_table,
        split_manifest_to_mapping,
    )

    perturbations = ("guide_0", "guide_1")
    genes = ("Gene0", "Gene1")
    animals = ("mouse_1", "mouse_2", "mouse_3")
    raw_rows: list[tuple[object, ...]] = []
    split_rows: list[Any] = []
    row_id = 0
    for animal in animals:
        for source_index in range(20):
            section = f"{animal}_section_{source_index:02d}"
            block = f"block_{source_index % 3}"
            for perturbation_index, source_perturbation in enumerate(
                (*perturbations, "mSafe")
            ):
                source_x = float(perturbation_index * 1_000)
                source_id = f"{animal}_{source_index:02d}_{source_perturbation}_source"
                raw_rows.append(
                    (
                        animal, section, block, source_id, source_perturbation,
                        "source_type", source_x, 0.0, True,
                    )
                )
                split_rows.append(
                    BridgeSplitRow(
                        row_id, source_id, animal, section, block,
                        source_perturbation, source_perturbation,
                        "source_type", "source_type",
                        (
                            "safe_source"
                            if source_perturbation == "mSafe"
                            else "perturbation_source"
                        ),
                        "own",
                    )
                )
                row_id += 1
                for rank in range(1, 61 if source_index < 10 else 1):
                    neighbor_id = f"{source_id}_neighbor_{rank:02d}"
                    raw_rows.append(
                        (
                            animal, section, block, neighbor_id, "unperturbed",
                            "astrocyte", source_x + float(rank), 0.0, False,
                        )
                    )
                    split_rows.append(
                        BridgeSplitRow(
                            row_id, neighbor_id, animal, section, block,
                            "unassigned", "unperturbed", "astrocyte", "astrocyte",
                            "neighbour", "none",
                        )
                    )
                    row_id += 1
    raw_columns = (
        "animal_id", "section_id", "spatial_block", "cell_id",
        "perturbation_id", "cell_type", "x", "y", "barcode_positive",
    )
    cells = pd.DataFrame(raw_rows, columns=raw_columns)
    neighbor_frame = build_bridge_neighbors(cells)
    relations = tuple(
        freeze_bridge_neighbour_relation(*row)
        for row in neighbor_frame.itertuples(index=False, name=None)
    )
    relation_table = freeze_bridge_neighbour_table(relations)
    sections = tuple(
        (
            animal,
            tuple(f"{animal}_section_{index:02d}" for index in range(20)),
        )
        for animal in animals
    )
    candidate = BridgeCandidate(
        "generic_task5_bridge", "SYNTHETIC", "spatial_perturbation",
        animals, sections, "mSafe", perturbations,
        "https://example.test/SYNTHETIC", "a" * 64,
    )
    total_rows = len(split_rows)
    per_animal_rows = tuple(
        (animal, sum(row.animal_id == animal for row in split_rows))
        for animal in animals
    )
    per_animal_sources = tuple(
        (
            animal,
            sum(
                row.animal_id == animal and row.cell_role == "perturbation_source"
                for row in split_rows
            ),
        )
        for animal in animals
    )
    per_animal_safe = tuple(
        (
            animal,
            sum(
                row.animal_id == animal and row.cell_role == "safe_source"
                for row in split_rows
            ),
        )
        for animal in animals
    )
    summary = MetadataSummary(
        "generic_task5_bridge", "SYNTHETIC", ("pilot",), animals, sections,
        ("block_0", "block_1", "block_2"), True, True, total_rows,
        genes, len(genes), perturbations,
        tuple(
            (
                perturbation,
                sum(row.observed_label == perturbation for row in split_rows),
            )
            for perturbation in perturbations
        ),
        (("mSafe", sum(count for _, count in per_animal_safe)),),
        (("valid", total_rows),), (("valid", total_rows),),
        tuple((animal, "pilot") for animal in animals), (), per_animal_rows,
        per_animal_sources, per_animal_safe, per_animal_rows, per_animal_rows,
        "CC-BY-4.0", "a" * 64, True,
    )
    metadata = BridgeSplitMetadata(
        tuple(split_rows), genes, perturbations, ("astrocyte",),
        tuple(zip(perturbations, genes)), (), "mSafe", relation_table.relations,
        relation_table.identity_sha256, candidate, summary,
        audit_bridge_capability(candidate, summary),
    )
    manifest = build_pilot_fold(metadata, "mouse_1")
    parent_evidence: list[Any] = []
    for parent in manifest.perturbation_parents:
        parent_treatment = tuple(
            row
            for row in manifest.row_provenance
            if row.animal_id == parent.animal_id
            and row.context_perturbation_id == parent.perturbation_id
            and row.cell_role == "perturbation_source"
        )
        parent_treatment_sections = {row.section_id for row in parent_treatment}
        parent_safe = tuple(
            row
            for row in manifest.row_provenance
            if row.animal_id == parent.animal_id
            and row.context_perturbation_id == manifest.safe_control_label
            and row.cell_role == "safe_source"
            and row.section_id in parent_treatment_sections
        )
        parent_evidence.append(
            BridgeParentEvidence(
                parent.animal_id, parent.perturbation_id, parent.target_gene,
                tuple(row.cell_id for row in parent_treatment),
                tuple(row.cell_id for row in parent_safe),
            )
        )
    unit_evidence: list[Any] = []
    for unit in manifest.primary_units:
        unit_treatment = tuple(
            relation
            for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.source_perturbation_id == unit.perturbation_id
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and not relation.is_safe_control
        )
        unit_treatment_sections = {
            relation.section_id for relation in unit_treatment
        }
        unit_safe = tuple(
            relation
            for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.source_perturbation_id == manifest.safe_control_label
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and relation.is_safe_control
            and relation.section_id in unit_treatment_sections
        )
        unit_evidence.append(
            BridgePrimaryUnitEvidence(
                unit.unit_id,
                tuple(relation.relation_id for relation in unit_treatment),
                tuple(relation.relation_id for relation in unit_safe),
            )
        )
    evidence = build_bridge_eligibility_evidence(
        manifest, tuple(parent_evidence), tuple(unit_evidence)
    )
    eligibility = evaluate_bridge_eligibility(manifest, evidence)
    if not eligibility.eligible:
        raise BridgePredictorContractError("code-owned synthetic fixture is ineligible")
    cell_ids = tuple(sorted(
        {relation.neighbor_cell_id for relation in manifest.neighbour_relations}
        | {
            cell_id
            for parent in evidence.parent_evidence
            for cell_id in (
                *parent.perturbation_source_cell_ids,
                *parent.safe_source_cell_ids,
            )
        }
    ))
    row_by_cell = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    gene_by_name = {gene: index for index, gene in enumerate(manifest.gene_names)}
    expression = np.zeros((len(cell_ids), len(manifest.gene_names)), dtype=np.float64)
    high = math.expm1(1.0)
    relation_by_id = {
        relation.relation_id: relation for relation in manifest.neighbour_relations
    }
    unit_by_id = {unit.unit_id: unit for unit in manifest.primary_units}
    for unit_item in evidence.unit_evidence:
        unit = unit_by_id[unit_item.unit_id]
        column = gene_by_name[unit.target_gene]
        for relation_id in unit_item.perturbation_neighbour_relation_ids:
            expression[
                row_by_cell[relation_by_id[relation_id].neighbor_cell_id], column
            ] = high
    parent_by_context = {
        (parent.animal_id, parent.perturbation_id): parent
        for parent in manifest.perturbation_parents
    }
    for parent_item in evidence.parent_evidence:
        parent = parent_by_context[
            (parent_item.animal_id, parent_item.perturbation_id)
        ]
        column = gene_by_name[parent.target_gene]
        for cell_id in parent_item.perturbation_source_cell_ids:
            expression[row_by_cell[cell_id], column] = high
    train_ids = set(manifest.train_rows)
    training_rows = tuple(
        row
        for row in manifest.row_provenance
        if row.stable_row_id in train_ids
        and row.cell_role == "safe_source"
        and row.observed_label == manifest.safe_control_label
    )
    training_cell_ids = tuple(row.cell_id for row in training_rows)
    control_rows = tuple(range(len(training_rows)))
    training_expression = np.zeros(
        (len(training_rows), len(manifest.gene_names)), dtype=np.float64
    )
    for offset, row_index in enumerate(control_rows):
        training_expression[row_index, :] = 0.0 if offset % 2 == 0 else high
    standardizer = fit_train_control_standardizer(
        training_expression,
        gene_names=manifest.gene_names,
        control_rows=control_rows,
        cell_ids=training_cell_ids,
        split_manifest=manifest,
    )
    split_mapping, _, neighbor_bytes, data_identity = _validated_data_components(
        split_manifest=manifest,
        neighbor_cells=cells,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
    )
    observed_projection = build_bridge_observed_effect_projection(
        expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
        standardizer=standardizer,
        eligibility=eligibility,
        data_identity_sha256=data_identity,
    )
    fixture = {
        "raw_neighbor_bytes": _raw_neighbor_csv_bytes(cells),
        "split_mapping": split_mapping,
        "neighbor_bytes": neighbor_bytes,
        "data_identity_sha256": data_identity,
        "expression": expression,
        "cell_ids": cell_ids,
        "gene_names": standardizer.genes,
        "standardizer": standardizer,
        "standardizer_mapping": train_control_standardizer_to_mapping(standardizer),
        "eligibility": eligibility,
        "eligibility_mapping": eligibility_result_to_mapping(eligibility),
        "observed_projection": observed_projection,
    }
    return fixture




def _prediction_frames(
    bundle: BridgePredictionBundle,
) -> tuple[tuple[Any, Any, Any], tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]]:
    # These are the existing Task 7/8 adapters.  They are imported lazily so a
    # terminal capability audit and CLI --help remain cold and outcome-blind.
    from src.evaluation.spatial_perturbation_comparators import (
        bridge_predictions_to_comparator_frame,
        validate_bridge_comparator_predictions,
        validate_required_bridge_comparators,
    )
    from src.evaluation.spatial_perturbation_scoring import BridgePrediction

    payload = prediction_payload_to_mapping(bundle)
    methods = payload["predictions"]
    frames: list[Any] = []
    prediction_sets: list[tuple[Any, ...]] = []
    for method_id in _METHOD_TO_ARTIFACT:
        predictions = tuple(
            BridgePrediction(
                row["unit_id"], row["endpoint"], row["predicted_effect"]
            )
            for row in methods[method_id]
        )
        prediction_sets.append(predictions)
        frames.append(bridge_predictions_to_comparator_frame(predictions))
    validate_required_bridge_comparators(
        ("matched_euclidean_spatial_causal", "hypersca_own_only")
    )
    validate_bridge_comparator_predictions(frames[0], frames[1], frames[2])
    return (
        (frames[0], frames[1], frames[2]),
        (prediction_sets[0], prediction_sets[1], prediction_sets[2]),
    )


def _csv_bytes(frame: Any) -> bytes:
    import pandas as pd

    if type(frame) is not pd.DataFrame:
        raise BridgePredictorContractError("CSV frame must be an exact pandas DataFrame")
    rows, columns = frame.shape
    if (
        rows < 0
        or rows > _MAX_CSV_ROWS
        or columns <= 0
        or columns > _MAX_CSV_COLUMNS
        or rows * columns > _MAX_CSV_CELLS
        or any(type(column) is not str for column in frame.columns.tolist())
    ):
        raise BridgePredictorContractError("CSV row or column resource bound exceeded")
    estimated_bytes = sum(len(column.encode("utf-8")) + 1 for column in frame.columns)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series.dtype):
            for value in series.tolist():
                if type(value) is str:
                    estimated_bytes += len(value.encode("utf-8")) + 3
                elif value is None or type(value) in (bool, int, float):
                    estimated_bytes += 32
                else:
                    raise BridgePredictorContractError(
                        "CSV object cells must contain exact primitive values"
                    )
                if estimated_bytes > _MAXIMUM_BUNDLE_BYTES:
                    raise BridgePredictorContractError(
                        "CSV estimated byte resource bound exceeded"
                    )
        else:
            estimated_bytes += rows * 32
        if estimated_bytes > _MAXIMUM_BUNDLE_BYTES:
            raise BridgePredictorContractError(
                "CSV estimated byte resource bound exceeded"
            )
    payload = pd.DataFrame.to_csv(
        frame, index=False, lineterminator="\n"
    ).encode("utf-8")
    if (
        not payload
        or len(payload) > _MAXIMUM_BUNDLE_BYTES
        or b"\r" in payload
        or b"\x00" in payload
    ):
        raise BridgePredictorContractError(
            "prediction table did not serialize as bounded canonical UTF-8 CSV"
        )
    return payload


def _raw_neighbor_csv_bytes(frame: Any) -> bytes:
    """Snapshot exact primitive Task6 input without dispatching user overrides."""

    import numpy as np
    import pandas as pd

    if type(frame) is not pd.DataFrame:
        raise BridgePredictorContractError(
            "neighbor_cells must be an exact pandas DataFrame"
        )
    columns = tuple(frame.columns.tolist())
    if (
        frame.shape[0] > 20_000
        or frame.shape[1] != len(_RAW_NEIGHBOR_COLUMNS)
        or frame.shape[0] * frame.shape[1] > _MAX_CSV_CELLS
    ):
        raise BridgePredictorContractError(
            "neighbor_cells row or column resource bound exceeded"
        )
    if (
        any(type(column) is not str for column in columns)
        or columns != _RAW_NEIGHBOR_COLUMNS
    ):
        raise BridgePredictorContractError(
            "neighbor_cells must use the exact ordered Task 6 columns"
        )
    for column in _RAW_NEIGHBOR_TEXT_COLUMNS:
        if any(type(value) is not str for value in frame[column].tolist()):
            raise BridgePredictorContractError(
                "neighbor_cells text values must be exact built-in strings"
            )
    for column in ("x", "y"):
        series = frame[column]
        if (
            not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            raise BridgePredictorContractError(
                "neighbor_cells coordinates must be exact real numeric columns"
            )
        values = series.to_numpy(copy=True)
        if not bool(np.isfinite(values).all()):
            raise BridgePredictorContractError(
                "neighbor_cells coordinates must be finite"
            )
    barcode = frame["barcode_positive"]
    barcode_values = barcode.tolist()
    if (
        not pd.api.types.is_bool_dtype(barcode.dtype)
        or any(type(value) is not bool for value in barcode_values)
    ):
        raise BridgePredictorContractError(
            "neighbor_cells barcode flags must be exact built-in booleans"
        )
    payload = pd.DataFrame.to_csv(
        frame, index=False, lineterminator="\n"
    ).encode("utf-8")
    if (
        not payload
        or len(payload) > _MAXIMUM_BUNDLE_BYTES
        or b"\r" in payload
        or b"\x00" in payload
    ):
        raise BridgePredictorContractError(
            "raw neighbor table did not serialize as bounded canonical UTF-8 CSV"
        )
    return payload


def _publish_predictions(
    bundle: BridgePredictionBundle,
    *,
    output_dir: Path | str,
    maximum_bundle_bytes: int,
    protocol: object,
    split_manifest: object,
    neighbor_cells: object,
    expression: object,
    cell_ids: object,
    gene_names: object,
    standardizer: object,
    eligibility: object,
    hypersca_budget: object,
    matched_euclidean_budget: object,
) -> VerifiedRunEvidence:
    import numpy as np
    import pandas as pd
    from src.methods_protocol_v3_contract import MethodsProtocolV3
    from src.evaluation.spatial_perturbation_comparators import BridgeModelBudget
    from src.evaluation.spatial_perturbation_scoring import TrainControlStandardizer
    from src.evaluation.spatial_perturbation_split import (
        BridgeEligibilityResult,
        BridgeSplitManifest,
    )

    if (
        type(protocol) is not MethodsProtocolV3
        or type(split_manifest) is not BridgeSplitManifest
        or type(neighbor_cells) is not pd.DataFrame
        or type(expression) is not np.ndarray
        or type(cell_ids) is not tuple
        or type(gene_names) is not tuple
        or type(standardizer) is not TrainControlStandardizer
        or type(eligibility) is not BridgeEligibilityResult
        or type(hypersca_budget) is not BridgeModelBudget
        or type(matched_euclidean_budget) is not BridgeModelBudget
    ):
        raise BridgePredictorContractError(
            "prediction publication requires exact protocol and Task5-8 contract objects"
        )
    protocol = cast(MethodsProtocolV3, protocol)
    split_manifest = cast(BridgeSplitManifest, split_manifest)
    neighbor_cells = cast(pd.DataFrame, neighbor_cells)
    expression = cast(np.ndarray, expression)
    cell_ids = cast(tuple[str, ...], cell_ids)
    gene_names = cast(tuple[str, ...], gene_names)
    standardizer = cast(TrainControlStandardizer, standardizer)
    eligibility = cast(BridgeEligibilityResult, eligibility)
    hypersca_budget = cast(BridgeModelBudget, hypersca_budget)
    matched_euclidean_budget = cast(
        BridgeModelBudget, matched_euclidean_budget
    )
    if eligibility.eligible is not True:
        raise BridgePredictorContractError(
            "bridge scoring requires the complete Task 5 eligibility gate to pass"
        )
    initial = bridge_prediction_bundle_to_mapping(bundle)
    protocol_mapping = formal_protocol_declaration_to_mapping(protocol)
    protocol_identity = formal_protocol_declaration_identity_sha256(protocol)
    if bundle.protocol_identity_sha256 != protocol_identity:
        raise BridgePredictorContractError(
            "prediction protocol identity differs from the exact Methods v3.0 declaration"
        )
    if protocol_mapping["bridge_role"] != "pilot_audit_only":
        raise BridgePredictorContractError(
            "synthetic replay requires the pilot_audit_only Methods v3.0 role"
        )
    statistics = protocol_mapping.get("statistics")
    if (
        type(statistics) is not dict
        or cast(dict[str, object], statistics).get("pilot_seeds") != [11, 23, 47]
        or bundle.model_seed not in (11, 23, 47)
    ):
        raise BridgePredictorContractError(
            "prediction model seed is not an applicable Methods v3.0 pilot seed"
        )
    if bundle.method_id != "hypersca":
        raise BridgePredictorContractError(
            "bridge runner requires the exact hypersca method identity"
        )
    code_identity = _source_identity()
    if bundle.code_identity_sha256 != code_identity:
        raise BridgePredictorContractError(
            "prediction bundle code identity differs from the runner code identity"
        )
    (
        split_mapping,
        neighbors,
        neighbor_bytes,
        data_identity,
    ) = _validated_data_components(
        split_manifest=split_manifest,
        neighbor_cells=neighbor_cells,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
    )
    split_manifest_bytes = _split_manifest_json_bytes(
        split_mapping, maximum_bytes=maximum_bundle_bytes
    )
    if data_identity != bundle.data_identity_sha256:
        raise BridgePredictorContractError(
            "prediction data identity differs from Task 5-7 replay inputs"
        )
    if split_mapping["split_identity_sha256"] != bundle.split_identity_sha256:
        raise BridgePredictorContractError(
            "prediction split identity differs from the Task 5 manifest"
        )
    if (
        protocol_mapping["capability_identity_sha256"]
        != split_mapping["capability_identity_sha256"]
    ):
        raise BridgePredictorContractError(
            "protocol capability identity differs from the Task 5 manifest"
        )
    from src.evaluation.spatial_perturbation_comparators import (
        bridge_model_budget_to_mapping,
        validate_bridge_comparator_budgets,
    )
    from src.evaluation.spatial_perturbation_scoring import (
        bridge_score_to_mapping,
        build_bridge_observed_effect_projection,
        score_bridge_predictions,
        train_control_standardizer_to_mapping,
    )
    from src.evaluation.spatial_perturbation_split import (
        eligibility_result_to_mapping,
        split_manifest_to_mapping,
    )

    validate_bridge_comparator_budgets(hypersca_budget, matched_euclidean_budget)
    budget_mappings = (
        bridge_model_budget_to_mapping(hypersca_budget),
        bridge_model_budget_to_mapping(matched_euclidean_budget),
    )
    if any(
        mapping["data_identity_sha256"] != data_identity
        or mapping["seed"] != bundle.model_seed
        for mapping in budget_mappings
    ):
        raise BridgePredictorContractError(
            "comparator budgets differ from exact data identity or model seed"
        )
    eligibility_mapping = eligibility_result_to_mapping(eligibility)
    if (
        eligibility_mapping["split_identity_sha256"]
        != bundle.split_identity_sha256
    ):
        raise BridgePredictorContractError(
            "eligibility evidence differs from the Task 5 split identity"
        )
    if eligibility_mapping["eligible"] is not True:
        raise BridgePredictorContractError(
            "bridge scoring requires the complete Task 5 eligibility gate to pass"
        )
    synthetic_fixture_identity = _synthetic_fixture_identity_sha256(
        split_mapping, data_identity=data_identity
    )
    standardizer_mapping = train_control_standardizer_to_mapping(standardizer)
    if standardizer_mapping["split_identity_sha256"] != bundle.split_identity_sha256:
        raise BridgePredictorContractError(
            "standardizer differs from the Task 5 split identity"
        )
    eligibility_identity = eligibility_mapping["eligibility_identity_sha256"]
    standardizer_identity = standardizer_mapping["training_identity_sha256"]
    support_record: dict[str, object] = {
        "eligibility_identity_sha256": eligibility_identity,
        "standardizer_identity_sha256": standardizer_identity,
        "comparator_budgets": list(budget_mappings),
    }
    support_identity = canonical_sha256(support_record)

    def current_support_identity() -> str:
        validate_bridge_comparator_budgets(
            hypersca_budget, matched_euclidean_budget
        )
        current_eligibility = eligibility_result_to_mapping(eligibility)
        if current_eligibility["eligible"] is not True:
            raise BridgePredictorContractError(
                "bridge eligibility changed or no longer passes before publication"
            )
        current_eligibility_identity = current_eligibility[
            "eligibility_identity_sha256"
        ]
        del current_eligibility
        current_standardizer = train_control_standardizer_to_mapping(standardizer)
        current_standardizer_identity = current_standardizer[
            "training_identity_sha256"
        ]
        del current_standardizer
        return canonical_sha256(
            {
                "eligibility_identity_sha256": current_eligibility_identity,
                "standardizer_identity_sha256": current_standardizer_identity,
                "comparator_budgets": [
                    bridge_model_budget_to_mapping(hypersca_budget),
                    bridge_model_budget_to_mapping(matched_euclidean_budget),
                ],
            }
        )
    del eligibility_mapping
    del standardizer_mapping
    (hypersca, matched, own_only), prediction_sets = _prediction_frames(bundle)
    unit_record: dict[str, object] = {
        "units": [
            [row.unit_id, row.endpoint] for row in prediction_sets[0]
        ]
    }
    if canonical_sha256(unit_record) != bundle.statistical_unit_identity_sha256:
        raise BridgePredictorContractError(
            "prediction statistical-unit identity does not match exact row order"
        )
    scores: list[Any] = []
    score_mappings: dict[str, dict[str, object]] = {}
    for method_id, predictions in zip(_METHOD_TO_ARTIFACT, prediction_sets):
        score = score_bridge_predictions(
            expression,
            cell_ids=cell_ids,
            gene_names=gene_names,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=predictions,
        )
        scores.append(score)
        score_mappings[method_id] = bridge_score_to_mapping(
            score,
            expression=expression,
            cell_ids=cell_ids,
            gene_names=gene_names,
            standardizer=standardizer,
            eligibility=eligibility,
            predictions=predictions,
        )
    observed_projection = build_bridge_observed_effect_projection(
        expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        standardizer=standardizer,
        eligibility=eligibility,
        data_identity_sha256=data_identity,
    )
    projection_observed = {
        (row["unit_id"], row["endpoint"]): float.fromhex(row["observed_delta_hex"])
        for row in cast(list[dict[str, str]], observed_projection["effects"])
    }
    if any({
        (effect.unit_id, effect.endpoint): effect.observed_delta
        for effect in score.effect_table.effects
    } != projection_observed for score in scores):
        raise BridgePredictorContractError(
            "Task7 observed effects changed between prediction methods"
        )
    observed_projection_identity = canonical_sha256(observed_projection)
    run_input_identity = _run_input_identity_sha256(
        data_identity_sha256=data_identity,
        observed_projection_identity_sha256=observed_projection_identity,
    )
    analysis_record: dict[str, object] = {
        "schema_version": "1.0",
        "claim_id": "bridge",
        "method_id": bundle.method_id,
        "protocol_identity_sha256": bundle.protocol_identity_sha256,
        "input_identity_sha256": run_input_identity,
        "data_identity_sha256": bundle.data_identity_sha256,
        "split_identity_sha256": bundle.split_identity_sha256,
        "statistical_unit_identity_sha256": bundle.statistical_unit_identity_sha256,
        "code_identity_sha256": bundle.code_identity_sha256,
        "prediction_bytes_sha256": bundle.prediction_bytes_sha256,
        "prediction_bundle_identity_sha256": bundle.prediction_bundle_identity_sha256,
        "model_seed": bundle.model_seed,
        "origin": bundle.origin,
        "evidence_role": bundle.evidence_role,
        "neighbor_max_rank": 60,
        "neighbor_bands": ["proximal", "local", "transition", "distal"],
        "neighbor_input_sha256": _synthetic_raw_neighbor_input_sha256(
            _raw_neighbor_csv_bytes(neighbor_cells)
        ),
        "neighbor_units_sha256": hashlib.sha256(neighbor_bytes).hexdigest(),
        "eligibility_identity_sha256": eligibility_identity,
        "standardizer_identity_sha256": standardizer_identity,
        "comparator_budget_identity_sha256": canonical_sha256(
            list(budget_mappings)
        ),
        "comparator_budgets": list(budget_mappings),
        "support_identity_sha256": support_identity,
        "support_record": support_record,
        "observed_effect_projection_identity_sha256": (
            observed_projection_identity
        ),
        "observed_effect_projection": observed_projection,
        "synthetic_fixture_identity_sha256": synthetic_fixture_identity,
        "scoring_identities": [
            score_mappings[method]["scoring_identity_sha256"]
            for method in _METHOD_TO_ARTIFACT
        ],
    }
    analysis_contract = _analysis_contract_record(analysis_record)
    analysis_record["analysis_contract"] = analysis_contract
    analysis_record["analysis_contract_identity_sha256"] = canonical_sha256(
        analysis_contract
    )
    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    primary_summary: dict[str, object] = {"schema_version": "1.0"}
    frozen_eligibility = cast(BridgeEligibilityResult, eligibility)
    evaluation_units = tuple(
        unit
        for unit in frozen_eligibility.manifest.primary_units
        if unit.animal_id in set(frozen_eligibility.manifest.evaluation_animals)
    )
    evaluation_context_count = len({
        (
            unit.animal_id,
            unit.perturbation_id,
            unit.neighbour_cell_type,
            unit.target_gene,
        )
        for unit in evaluation_units
    })
    for method_id, score in zip(_METHOD_TO_ARTIFACT, scores):
        for effect in score.effect_table.effects:
            primary_rows.append(
                {
                    "method_id": method_id,
                    "unit_id": effect.unit_id,
                    "endpoint": effect.endpoint,
                    "animal_id": effect.animal_id,
                    "perturbation_id": effect.perturbation_id,
                    "gene_name": effect.gene_name,
                    "neighbor_cell_type": effect.neighbor_cell_type,
                    "band": effect.band,
                    "observed_delta": effect.observed_delta,
                    "predicted_delta": effect.predicted_delta,
                    "effect_identity_sha256": effect.effect_identity_sha256,
                    "evaluation_neighbor_unit_count": len(evaluation_units),
                    "evaluation_calibration_context_count": evaluation_context_count,
                }
            )
        mapping = score_mappings[method_id]
        primary_summary[method_id] = {
            "neighbor_effect_rmse": mapping["neighbor_effect_rmse"],
            "own_effect_rmse": mapping["own_effect_rmse"],
            "coverage": mapping["coverage"],
            "abstention": mapping["abstention"],
            "distance_calibration_eligible_pairs": mapping[
                "distance_calibration_eligible_pairs"
            ],
            "distance_calibration_total_contexts": mapping[
                "distance_calibration_total_contexts"
            ],
            "distance_calibration_coverage": mapping[
                "distance_calibration_coverage"
            ],
            "distance_calibration_abstention": mapping[
                "distance_calibration_abstention"
            ],
            "effect_table_identity_sha256": mapping[
                "effect_table_identity_sha256"
            ],
            "scoring_identity_sha256": mapping["scoring_identity_sha256"],
        }
        for metric_id in (
            "neighbor_effect_pcc",
            "distance_decay_calibration_error",
            "effect_sign_accuracy",
        ):
            secondary_rows.append(
                {
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "value": mapping[metric_id],
                }
            )
    primary_frame = pd.DataFrame(primary_rows)
    secondary_frame = pd.DataFrame(secondary_rows)
    prediction_artifacts = {
        "predictions_hypersca.csv": _csv_bytes(hypersca),
        "predictions_matched_euclidean.csv": _csv_bytes(matched),
        "predictions_hypersca_own_only.csv": _csv_bytes(own_only),
    }
    artifacts: dict[str, bytes] = {
        "split_manifest.json": split_manifest_bytes,
        "capability_record.json": canonical_json_bytes(
            {
                "schema_version": "1.0",
                "status": "synthetic_prediction_bundle_replayed",
                "evidence_role": bundle.evidence_role,
                "prediction_bundle_identity_sha256": bundle.prediction_bundle_identity_sha256,
                "prediction_bundle": initial,
                "synthetic_fixture_identity_sha256": synthetic_fixture_identity,
            }
        ),
        "neighbor_units.csv": neighbor_bytes,
        **prediction_artifacts,
        "primary_metric_units.csv": _csv_bytes(primary_frame),
        "primary_metric_summary.json": canonical_json_bytes(primary_summary),
        "secondary_metrics.csv": _csv_bytes(secondary_frame),
        "resource_usage.json": canonical_json_bytes(
            {
                "schema_version": "1.0",
                "mode": "orchestration_only",
                "maximum_bundle_bytes": maximum_bundle_bytes,
                "model_seed": bundle.model_seed,
                "neighbor_max_rank": 60,
                "neighbor_units_bytes": len(neighbor_bytes),
                "prediction_payload_bytes": len(bundle.prediction_bytes),
                "model_fitted": False,
                "outcomes_used_only_by_task7_scoring_contract": True,
            }
        ),
    }
    analysis_record["artifact_identities_sha256"] = {
        relative_path: hashlib.sha256(payload).hexdigest()
        for relative_path, payload in sorted(artifacts.items())
    }
    identity = _identity(
        protocol_identity=bundle.protocol_identity_sha256,
        split_identity=bundle.split_identity_sha256,
        unit_record=unit_record,
        analysis_record=analysis_record,
        input_identity=run_input_identity,
        config_identity=bundle.prediction_bundle_identity_sha256,
        code_identity=bundle.code_identity_sha256,
        model_seed=bundle.model_seed,
        evidence_role=bundle.evidence_role,
    )
    artifacts["claim_decision.json"] = canonical_json_bytes(
        {
            "schema_version": "1.0",
            "claim_id": "bridge",
            "decision": "synthetic_audit_only_no_scientific_claim",
            "synthetic_fixture_identity_sha256": synthetic_fixture_identity,
            "analysis_record": analysis_record,
        }
    )
    publisher = RunEvidencePublisher.begin(
        output_dir=output_dir,
        identity=identity,
        statistical_unit_record=unit_record,
        required_artifacts=_SUCCESS_ARTIFACTS,
        maximum_bundle_bytes=maximum_bundle_bytes,
    )
    try:
        for relative_path in _SUCCESS_ARTIFACTS:
            media_type = (
                "application/json" if relative_path.endswith(".json") else "text/csv"
            )
            publisher.add_bytes(
                relative_path, artifacts[relative_path], media_type=media_type
            )
        if (
            bridge_prediction_bundle_to_mapping(bundle) != initial
            or formal_protocol_declaration_to_mapping(protocol) != protocol_mapping
            or _source_identity() != code_identity
            or _split_manifest_json_bytes(
                split_manifest_to_mapping(split_manifest),
                maximum_bytes=maximum_bundle_bytes,
            )
            != split_manifest_bytes
            or bridge_run_data_identity_sha256(
                split_manifest=split_manifest,
                neighbor_cells=neighbor_cells,
                expression=expression,
                cell_ids=cell_ids,
                gene_names=gene_names,
            )
            != data_identity
            or current_support_identity() != support_identity
            or _synthetic_fixture_identity_sha256(
                split_manifest_to_mapping(split_manifest),
                data_identity=data_identity,
            )
            != synthetic_fixture_identity
        ):
            raise BridgePredictorContractError(
                "prediction bundle or runner code identity changed before publication"
            )
        output = publisher.finalize_completed(
            summary={
                "claim_id": "bridge",
                "evidence_role": bundle.evidence_role,
                "analysis_contract_identity_sha256": analysis_record[
                    "analysis_contract_identity_sha256"
                ],
                "prediction_bundle_identity_sha256": bundle.prediction_bundle_identity_sha256,
                "scientific_claim_allowed": False,
            }
        )
    except BaseException:
        if publisher.state == "staging":
            publisher.abort()
        raise
    return _verified_twice(output, identity)


@overload
def publish_spatial_perturbation_run(
    run_input: BridgePredictorCapability,
    *,
    output_dir: Path | str,
    maximum_bundle_bytes: int = _MAXIMUM_BUNDLE_BYTES,
    protocol: None = None,
    split_manifest: None = None,
    neighbor_cells: None = None,
    expression: None = None,
    cell_ids: None = None,
    gene_names: None = None,
    standardizer: None = None,
    eligibility: None = None,
    hypersca_budget: None = None,
    matched_euclidean_budget: None = None,
) -> VerifiedRunEvidence: ...


@overload
def publish_spatial_perturbation_run(
    run_input: BridgePredictionBundle,
    *,
    output_dir: Path | str,
    maximum_bundle_bytes: int = _MAXIMUM_BUNDLE_BYTES,
    protocol: MethodsProtocolV3,
    split_manifest: BridgeSplitManifest,
    neighbor_cells: pd.DataFrame,
    expression: np.ndarray,
    cell_ids: tuple[str, ...],
    gene_names: tuple[str, ...],
    standardizer: TrainControlStandardizer,
    eligibility: BridgeEligibilityResult,
    hypersca_budget: BridgeModelBudget,
    matched_euclidean_budget: BridgeModelBudget,
) -> VerifiedRunEvidence: ...


def publish_spatial_perturbation_run(
    run_input: object,
    *,
    output_dir: Path | str,
    maximum_bundle_bytes: int = _MAXIMUM_BUNDLE_BYTES,
    protocol: object | None = None,
    split_manifest: object | None = None,
    neighbor_cells: object | None = None,
    expression: object | None = None,
    cell_ids: object | None = None,
    gene_names: object | None = None,
    standardizer: object | None = None,
    eligibility: object | None = None,
    hypersca_budget: object | None = None,
    matched_euclidean_budget: object | None = None,
) -> VerifiedRunEvidence:
    """Publish one exact terminal capability or already-produced bundle."""

    if (
        type(maximum_bundle_bytes) is not int
        or not 1 <= maximum_bundle_bytes <= _MAXIMUM_BUNDLE_BYTES
    ):
        raise BridgePredictorContractError(
            "maximum_bundle_bytes must be an exact bounded resource integer"
        )

    if type(run_input) is BridgePredictorCapability:
        if any(
            value is not None
            for value in (
                split_manifest,
                protocol,
                neighbor_cells,
                expression,
                cell_ids,
                gene_names,
                standardizer,
                eligibility,
                hypersca_budget,
                matched_euclidean_budget,
            )
        ):
            raise BridgePredictorContractError(
                "terminal capability publication does not accept scientific inputs"
            )
        return _publish_failure(
            run_input,
            output_dir=output_dir,
            maximum_bundle_bytes=maximum_bundle_bytes,
        )
    if type(run_input) is BridgePredictionBundle:
        return _publish_predictions(
            run_input,
            output_dir=output_dir,
            maximum_bundle_bytes=maximum_bundle_bytes,
            protocol=protocol,
            split_manifest=split_manifest,
            neighbor_cells=neighbor_cells,
            expression=expression,
            cell_ids=cell_ids,
            gene_names=gene_names,
            standardizer=standardizer,
            eligibility=eligibility,
            hypersca_budget=hypersca_budget,
            matched_euclidean_budget=matched_euclidean_budget,
        )
    raise BridgePredictorContractError(
        "run_input must be an exact revalidated prediction bundle or terminal capability"
    )


__all__ = [
    "bridge_run_data_identity_sha256",
    "publish_spatial_perturbation_run",
    "runner_code_identity_sha256",
    "verify_spatial_perturbation_evidence_bundle",
]
