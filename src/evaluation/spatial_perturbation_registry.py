"""Outcome-blind registry and structural capability audit for spatial bridges.

This module deliberately handles identifiers and metadata declarations only.  It
does not open assay matrices or calculate scientific outcomes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import cast
import unicodedata
from urllib.parse import urlsplit


MAXIMUM_REGISTRY_BYTES = 64 * 1024
MAXIMUM_NESTING = 16
MAXIMUM_TEXT_LENGTH = 256
MAXIMUM_ITEMS = 256
MAXIMUM_BLOCK_IDS = 1024
MAXIMUM_GENE_NAMES = 4096
MAXIMUM_COUNT = 10_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GEO_ACCESSION = re.compile(r"GSE[1-9][0-9]*")
_DNS_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))*")
_URI_PATH = re.compile(r"[A-Za-z0-9._~!$&'()*+,;=:@/\-]+")
_RENAME_NOREPLACE = 1
_AT_EMPTY_PATH = 0x1000
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400
_FROZEN_CANDIDATE_ORDER = ("gse274447_msafe_bridge",)
_FROZEN_GSE274447_SOURCE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447"
_FROZEN_GSE274447_SOURCE_IDENTITY = "0e908ba2f21cab2bd222daf31a85ff8369407c8df53f5d9a2424f081528ffa46"


class SpatialPerturbationRegistryError(ValueError):
    """A registry, metadata declaration, or audit publication is unsafe."""


def _safe_text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or len(value) > MAXIMUM_TEXT_LENGTH:
        raise SpatialPerturbationRegistryError(
            f"{name} must be bounded built-in NFC text"
        )
    if not allow_empty and not value:
        raise SpatialPerturbationRegistryError(f"{name} must not be empty")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SpatialPerturbationRegistryError(f"{name} must be trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise SpatialPerturbationRegistryError(f"{name} contains unsafe control text")
    return value


def _sha(value: object, name: str) -> str:
    text = _safe_text(value, name)
    if _SHA256.fullmatch(text) is None:
        raise SpatialPerturbationRegistryError(f"{name} must be a lowercase SHA-256")
    return text


def _canonical_source_uri(accession: str, value: object) -> str:
    """Accept only one spelling for each source resource identity."""
    source_uri = _safe_text(value, "source_uri")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in source_uri):
        raise SpatialPerturbationRegistryError("source_uri must be canonical ASCII HTTPS text")
    try:
        parsed = urlsplit(source_uri)
        port = parsed.port
    except ValueError as exc:
        raise SpatialPerturbationRegistryError("source_uri is not a canonical HTTPS URI") from exc
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not hostname
        or len(hostname) > 253
        or _DNS_HOST.fullmatch(hostname) is None
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.netloc != hostname
        or parsed.fragment
        or "%" in parsed.netloc
        or "%" in hostname
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or "%" in parsed.path
        or _URI_PATH.fullmatch(parsed.path) is None
        or any(piece in ("", ".", "..") for piece in parsed.path.split("/")[1:])
    ):
        raise SpatialPerturbationRegistryError("source_uri is not a canonical HTTPS URI")
    begins_geo = accession[:3].casefold() == "gse"
    is_geo = _GEO_ACCESSION.fullmatch(accession) is not None
    if begins_geo and not is_geo:
        raise SpatialPerturbationRegistryError("GEO accession must be uppercase canonical GSE text")
    is_geo_endpoint = (
        hostname == "www.ncbi.nlm.nih.gov"
        and parsed.path == "/geo/query/acc.cgi"
    )
    if is_geo:
        expected = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        if source_uri != expected:
            raise SpatialPerturbationRegistryError("GEO source_uri must exactly match accession")
    elif is_geo_endpoint:
        raise SpatialPerturbationRegistryError("NCBI GEO endpoint requires a matching GEO accession")
    elif parsed.query or "?" in source_uri:
        raise SpatialPerturbationRegistryError("non-GEO source_uri must not contain a query")
    return source_uri


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0 or value > MAXIMUM_COUNT:
        raise SpatialPerturbationRegistryError(f"{name} must be a bounded nonnegative integer")
    return value


def _flag(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SpatialPerturbationRegistryError(f"{name} must be a built-in boolean")
    return value


def _items(value: object, name: str, *, maximum: int = MAXIMUM_ITEMS) -> tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise SpatialPerturbationRegistryError(f"{name} must be a built-in list or tuple")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) > maximum:
        raise SpatialPerturbationRegistryError(f"{name} exceeds the item limit")
    return tuple(items)


def _text_items(value: object, name: str, *, maximum: int = MAXIMUM_ITEMS) -> tuple[str, ...]:
    frozen = tuple(
        _safe_text(item, f"{name}[{index}]")
        for index, item in enumerate(_items(value, name, maximum=maximum))
    )
    if len(set(frozen)) != len(frozen):
        raise SpatialPerturbationRegistryError(f"{name} must contain unique ordered IDs")
    return frozen


def _text_count_pairs(value: object, name: str) -> tuple[tuple[str, int], ...]:
    pairs: list[tuple[str, int]] = []
    for index, pair in enumerate(_items(value, name)):
        values = _items(pair, f"{name}[{index}]", maximum=2)
        if len(values) != 2:
            raise SpatialPerturbationRegistryError(f"{name}[{index}] must have two items")
        pairs.append(
            (_safe_text(values[0], f"{name}[{index}][0]"), _count(values[1], f"{name}[{index}][1]"))
        )
    labels = tuple(label for label, _ in pairs)
    if len(set(labels)) != len(labels):
        raise SpatialPerturbationRegistryError(f"{name} labels must be unique")
    return tuple(pairs)


def _sections(value: object, name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for index, pair in enumerate(_items(value, name)):
        values = _items(pair, f"{name}[{index}]", maximum=2)
        if len(values) != 2:
            raise SpatialPerturbationRegistryError(f"{name}[{index}] must have two items")
        pairs.append(
            (
                _safe_text(values[0], f"{name}[{index}][0]"),
                _text_items(values[1], f"{name}[{index}][1]"),
            )
        )
    labels = tuple(label for label, _ in pairs)
    if len(set(labels)) != len(labels):
        raise SpatialPerturbationRegistryError(f"{name} specimen IDs must be unique")
    section_ids = tuple(section_id for _, values in pairs for section_id in values)
    if len(set(section_ids)) != len(section_ids):
        raise SpatialPerturbationRegistryError(f"{name} section IDs must be globally unique")
    return tuple(pairs)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _identity(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class BridgeCandidate:
    candidate_id: str
    accession: str
    platform: str
    biological_specimens: tuple[str, ...]
    sections_by_specimen: tuple[tuple[str, tuple[str, ...]], ...]
    safe_control_label: str
    perturbation_labels: tuple[str, ...]
    source_uri: str
    source_identity_sha256: str

    def __post_init__(self) -> None:
        candidate_id = _safe_text(self.candidate_id, "candidate_id")
        accession = _safe_text(self.accession, "accession")
        platform = _safe_text(self.platform, "platform")
        specimens = _text_items(self.biological_specimens, "biological_specimens")
        sections = _sections(self.sections_by_specimen, "sections_by_specimen")
        if tuple(item[0] for item in sections) != specimens:
            raise SpatialPerturbationRegistryError(
                "sections_by_specimen must follow biological_specimens exactly"
            )
        for specimen, section_ids in sections:
            if specimen not in specimens or len(set(section_ids)) != len(section_ids):
                raise SpatialPerturbationRegistryError("invalid specimen-section consistency")
        safe_control_label = _safe_text(self.safe_control_label, "safe_control_label")
        perturbation_labels = _text_items(self.perturbation_labels, "perturbation_labels")
        if safe_control_label in perturbation_labels:
            raise SpatialPerturbationRegistryError("safe_control_label must not be a perturbation label")
        for name, value in (
            ("candidate_id", candidate_id), ("accession", accession),
            ("platform", platform), ("biological_specimens", specimens),
            ("sections_by_specimen", sections),
            ("safe_control_label", safe_control_label),
            ("perturbation_labels", perturbation_labels),
            ("source_uri", _canonical_source_uri(accession, self.source_uri)),
            ("source_identity_sha256", _sha(self.source_identity_sha256, "source_identity_sha256")),
        ):
            object.__setattr__(self, name, value)
        _validated_candidate_declaration(self)

    def to_mapping(self) -> dict[str, object]:
        item = _candidate_snapshot(self)
        return {
            "candidate_id": item.candidate_id, "accession": item.accession,
            "platform": item.platform, "biological_specimens": list(item.biological_specimens),
            "sections_by_specimen": [[label, list(values)] for label, values in item.sections_by_specimen],
            "safe_control_label": item.safe_control_label,
            "perturbation_labels": list(item.perturbation_labels), "source_uri": item.source_uri,
            "source_identity_sha256": item.source_identity_sha256,
        }


def _validated_candidate_declaration(candidate: BridgeCandidate) -> BridgeCandidate:
    """Re-check the immutable declaration whenever it crosses an audit boundary."""
    if type(candidate) is not BridgeCandidate:
        raise SpatialPerturbationRegistryError("candidate must be BridgeCandidate")
    canonical_accession = (
        candidate.accession.casefold() if type(candidate.accession) is str else ""
    )
    canonical_resource = (
        candidate.source_uri.split("#", 1)[0].casefold()
        if type(candidate.source_uri) is str else ""
    )
    canonical_source_identity = (
        candidate.source_identity_sha256.casefold()
        if type(candidate.source_identity_sha256) is str else ""
    )
    frozen_anchor = any((
        candidate.candidate_id == _FROZEN_CANDIDATE_ORDER[0],
        canonical_accession == "gse274447",
        canonical_resource == _FROZEN_GSE274447_SOURCE.casefold(),
        canonical_source_identity == _FROZEN_GSE274447_SOURCE_IDENTITY,
    ))
    exact_frozen_declaration = (
        candidate.candidate_id == _FROZEN_CANDIDATE_ORDER[0]
        and candidate.accession == "GSE274447"
        and candidate.platform == "spatial_perturbation"
        and candidate.biological_specimens == ("mouse_1", "mouse_2", "mouse_3")
        and candidate.sections_by_specimen
        == (("mouse_1", ()), ("mouse_2", ()), ("mouse_3", ()))
        and candidate.safe_control_label == "mSafe"
        and candidate.perturbation_labels == ()
        and candidate.source_uri == _FROZEN_GSE274447_SOURCE
        and candidate.source_identity_sha256 == _FROZEN_GSE274447_SOURCE_IDENTITY
    )
    if frozen_anchor and not exact_frozen_declaration:
        raise SpatialPerturbationRegistryError("GSE274447 candidate declaration must remain frozen")
    return candidate


@dataclass(frozen=True, slots=True)
class MetadataSummary:
    candidate_id: str
    accession: str
    cohort_ids: tuple[str, ...]
    biological_specimen_ids: tuple[str, ...]
    sections_by_specimen: tuple[tuple[str, tuple[str, ...]], ...]
    block_ids: tuple[str, ...]
    coordinate_available: bool
    coordinate_finite: bool
    coordinate_count: int
    measured_gene_names: tuple[str, ...]
    measured_gene_count: int
    perturbation_labels: tuple[str, ...]
    perturbation_label_counts: tuple[tuple[str, int], ...]
    safe_control_counts: tuple[tuple[str, int], ...]
    barcode_quality_counts: tuple[tuple[str, int], ...]
    label_quality_counts: tuple[tuple[str, int], ...]
    specimen_cohort_assignments: tuple[tuple[str, str], ...]
    external_untouched_cohort_ids: tuple[str, ...]
    per_specimen_coordinate_counts: tuple[tuple[str, int], ...]
    per_specimen_perturbation_counts: tuple[tuple[str, int], ...]
    per_specimen_safe_control_counts: tuple[tuple[str, int], ...]
    per_specimen_barcode_valid_counts: tuple[tuple[str, int], ...]
    per_specimen_label_valid_counts: tuple[tuple[str, int], ...]
    license_identity: str
    source_identity_sha256: str
    executable_output_schema_capable: bool

    def __post_init__(self) -> None:
        candidate_id = _safe_text(self.candidate_id, "candidate_id")
        accession = _safe_text(self.accession, "accession")
        cohorts = _text_items(self.cohort_ids, "cohort_ids")
        specimens = _text_items(self.biological_specimen_ids, "biological_specimen_ids")
        sections = _sections(self.sections_by_specimen, "sections_by_specimen")
        if tuple(item[0] for item in sections) != specimens:
            raise SpatialPerturbationRegistryError("sections must follow specimen IDs exactly")
        # Cohort labels are text rather than counts; revalidate their exact pair shape.
        assignment_values = _items(self.specimen_cohort_assignments, "specimen_cohort_assignments")
        assignment_pairs_list: list[tuple[str, str]] = []
        for index, pair in enumerate(assignment_values):
            pair_values = _items(pair, f"specimen_cohort_assignments[{index}]", maximum=2)
            if len(pair_values) != 2:
                raise SpatialPerturbationRegistryError("specimen_cohort_assignments pairs must have two items")
            assignment_pairs_list.append((
                _safe_text(pair_values[0], f"specimen_cohort_assignments[{index}][0]"),
                _safe_text(pair_values[1], f"specimen_cohort_assignments[{index}][1]"),
            ))
        assignment_pairs = tuple(assignment_pairs_list)
        assigned_specimens = tuple(specimen for specimen, _ in assignment_pairs)
        if any(specimen not in specimens for specimen in assigned_specimens):
            raise SpatialPerturbationRegistryError("cohort assignments contain an unknown specimen")
        if assigned_specimens != tuple(specimen for specimen in specimens if specimen in assigned_specimens):
            raise SpatialPerturbationRegistryError("cohort assignments must preserve specimen order")
        if len(set(specimen for specimen, _ in assignment_pairs)) != len(assignment_pairs):
            raise SpatialPerturbationRegistryError("cohort assignments must be unique")
        def specimen_counts(value: object, name: str) -> tuple[tuple[str, int], ...]:
            pairs = _text_count_pairs(value, name)
            if tuple(label for label, _ in pairs) != specimens:
                raise SpatialPerturbationRegistryError(f"{name} must follow specimen IDs exactly")
            return pairs
        coordinate_counts = specimen_counts(self.per_specimen_coordinate_counts, "per_specimen_coordinate_counts")
        perturbation_counts = specimen_counts(self.per_specimen_perturbation_counts, "per_specimen_perturbation_counts")
        safe_counts = specimen_counts(self.per_specimen_safe_control_counts, "per_specimen_safe_control_counts")
        barcode_counts = specimen_counts(self.per_specimen_barcode_valid_counts, "per_specimen_barcode_valid_counts")
        label_valid_counts = specimen_counts(self.per_specimen_label_valid_counts, "per_specimen_label_valid_counts")
        coordinate_count = _count(self.coordinate_count, "coordinate_count")
        if coordinate_count != sum(count for _, count in coordinate_counts):
            raise SpatialPerturbationRegistryError("coordinate_count must equal per-specimen total")
        genes = _text_items(self.measured_gene_names, "measured_gene_names", maximum=MAXIMUM_GENE_NAMES)
        gene_count = _count(self.measured_gene_count, "measured_gene_count")
        if gene_count < len(genes):
            raise SpatialPerturbationRegistryError("measured_gene_count is smaller than named genes")
        labels = _text_items(self.perturbation_labels, "perturbation_labels")
        label_counts = _text_count_pairs(self.perturbation_label_counts, "perturbation_label_counts")
        if tuple(label for label, _ in label_counts) != labels:
            raise SpatialPerturbationRegistryError("perturbation label counts must follow labels exactly")
        safe_control_counts = _text_count_pairs(self.safe_control_counts, "safe_control_counts")
        barcode_quality_counts = _text_count_pairs(self.barcode_quality_counts, "barcode_quality_counts")
        label_quality_counts = _text_count_pairs(self.label_quality_counts, "label_quality_counts")
        if sum(count for _, count in label_counts) != sum(count for _, count in perturbation_counts):
            raise SpatialPerturbationRegistryError("perturbation totals must equal per-specimen total")
        if dict(barcode_quality_counts).get("valid", 0) != sum(count for _, count in barcode_counts):
            raise SpatialPerturbationRegistryError("valid barcode total must equal per-specimen total")
        if dict(label_quality_counts).get("valid", 0) != sum(count for _, count in label_valid_counts):
            raise SpatialPerturbationRegistryError("valid label total must equal per-specimen total")
        normalized_values: tuple[tuple[str, object], ...] = (
            ("candidate_id", candidate_id), ("accession", accession), ("cohort_ids", cohorts),
            ("biological_specimen_ids", specimens), ("sections_by_specimen", sections),
            ("block_ids", _text_items(self.block_ids, "block_ids", maximum=MAXIMUM_BLOCK_IDS)),
            ("coordinate_available", _flag(self.coordinate_available, "coordinate_available")),
            ("coordinate_finite", _flag(self.coordinate_finite, "coordinate_finite")),
            ("coordinate_count", coordinate_count),
            ("measured_gene_names", genes), ("measured_gene_count", gene_count),
            ("perturbation_labels", labels), ("perturbation_label_counts", label_counts),
            ("safe_control_counts", safe_control_counts),
            ("barcode_quality_counts", barcode_quality_counts),
            ("label_quality_counts", label_quality_counts),
            ("specimen_cohort_assignments", assignment_pairs),
            ("external_untouched_cohort_ids", _text_items(self.external_untouched_cohort_ids, "external_untouched_cohort_ids")),
            ("per_specimen_coordinate_counts", coordinate_counts),
            ("per_specimen_perturbation_counts", perturbation_counts),
            ("per_specimen_safe_control_counts", safe_counts),
            ("per_specimen_barcode_valid_counts", barcode_counts),
            ("per_specimen_label_valid_counts", label_valid_counts),
            ("license_identity", _safe_text(self.license_identity, "license_identity")),
            ("source_identity_sha256", _sha(self.source_identity_sha256, "source_identity_sha256")),
            ("executable_output_schema_capable", _flag(self.executable_output_schema_capable, "executable_output_schema_capable")),
        )
        for name, value in normalized_values:
            object.__setattr__(self, name, value)

    def to_mapping(self) -> dict[str, object]:
        item = _summary_snapshot(self)
        pairs = lambda values: [[label, count] for label, count in values]
        return {
            "candidate_id": item.candidate_id, "accession": item.accession,
            "cohort_ids": list(item.cohort_ids), "biological_specimen_ids": list(item.biological_specimen_ids),
            "sections_by_specimen": [[label, list(values)] for label, values in item.sections_by_specimen],
            "block_ids": list(item.block_ids), "coordinate_available": item.coordinate_available,
            "coordinate_finite": item.coordinate_finite, "coordinate_count": item.coordinate_count,
            "measured_gene_names": list(item.measured_gene_names),
            "measured_gene_count": item.measured_gene_count, "perturbation_labels": list(item.perturbation_labels),
            "perturbation_label_counts": pairs(item.perturbation_label_counts),
            "safe_control_counts": pairs(item.safe_control_counts),
            "barcode_quality_counts": pairs(item.barcode_quality_counts),
            "label_quality_counts": pairs(item.label_quality_counts), "license_identity": item.license_identity,
            "specimen_cohort_assignments": [[specimen, cohort] for specimen, cohort in item.specimen_cohort_assignments],
            "external_untouched_cohort_ids": list(item.external_untouched_cohort_ids),
            "per_specimen_coordinate_counts": pairs(item.per_specimen_coordinate_counts),
            "per_specimen_perturbation_counts": pairs(item.per_specimen_perturbation_counts),
            "per_specimen_safe_control_counts": pairs(item.per_specimen_safe_control_counts),
            "per_specimen_barcode_valid_counts": pairs(item.per_specimen_barcode_valid_counts),
            "per_specimen_label_valid_counts": pairs(item.per_specimen_label_valid_counts),
            "source_identity_sha256": item.source_identity_sha256,
            "executable_output_schema_capable": item.executable_output_schema_capable,
        }


_COVERAGE_KEYS = (
    "candidate_identity", "specimen_structure", "cohorts", "coordinates", "genes",
    "perturbation_labels", "safe_controls", "barcode_quality", "label_quality",
    "license", "source_identity", "output_schema", "registered_specimens",
    "registered_sections", "registered_perturbation_labels", "external_cohort",
    "development_cohort",
)

_GATE_REASONS = {
    "candidate_identity": "candidate_identity_mismatch",
    "specimen_structure": "missing_specimen_sections",
    "cohorts": "cohort_identity_coverage_missing",
    "coordinates": "coordinates_unavailable_or_nonfinite",
    "genes": "measured_gene_declaration_missing",
    "perturbation_labels": "perturbation_label_coverage_missing",
    "safe_controls": "safe_control_coverage_missing",
    "barcode_quality": "barcode_quality_coverage_missing",
    "label_quality": "label_quality_coverage_missing",
    "license": "license_identity_missing",
    "source_identity": "source_identity_mismatch",
    "output_schema": "output_schema_capability_missing",
    "registered_specimens": "registered_specimens_mismatch",
    "registered_sections": "registered_sections_mismatch",
    "registered_perturbation_labels": "registered_perturbation_labels_mismatch",
    "external_cohort": "external_cohort_missing",
    "development_cohort": "development_cohort_missing",
}


def _semantic_state(
    specimens: int, cohorts: int, coverage: Mapping[str, float], *, unavailable: bool
) -> tuple[str, bool, tuple[str, ...]]:
    reasons: list[str] = []
    if specimens < 5:
        reasons.append("insufficient_biological_replicates")
    if cohorts < 2:
        reasons.append("insufficient_independent_cohorts")
    for gate, reason in _GATE_REASONS.items():
        if coverage[gate] == 0.0:
            reasons.append(reason)
    if unavailable:
        reasons.append("asset_metadata_unavailable")
    frozen_reasons = tuple(reasons)
    if unavailable:
        return "assets_unavailable", False, frozen_reasons
    if frozen_reasons:
        return "pilot_audit_only", False, frozen_reasons
    return "confirmatory_capable", True, ()


@dataclass(frozen=True, slots=True)
class BridgeCapabilityResult:
    candidate_id: str
    status: str
    confirmatory_capable: bool
    biological_specimen_count: int
    cohort_count: int
    coverage: Mapping[str, float]
    blocking_reasons: tuple[str, ...]
    capability_identity_sha256: str

    def __post_init__(self) -> None:
        candidate_id = _safe_text(self.candidate_id, "candidate_id")
        status = _safe_text(self.status, "status")
        capable = _flag(self.confirmatory_capable, "confirmatory_capable")
        specimens = _count(self.biological_specimen_count, "biological_specimen_count")
        cohorts = _count(self.cohort_count, "cohort_count")
        if type(self.coverage) is not dict or tuple(self.coverage) != _COVERAGE_KEYS:
            raise SpatialPerturbationRegistryError("coverage must be an exact ordered built-in mapping")
        coverage: dict[str, float] = {}
        for key in _COVERAGE_KEYS:
            value = self.coverage[key]
            if type(value) is not float or value not in (0.0, 1.0):
                raise SpatialPerturbationRegistryError("coverage values must be binary built-in floats")
            coverage[key] = 0.0 if value == 0.0 else 1.0
        reasons = _text_items(self.blocking_reasons, "blocking_reasons")
        if status not in ("confirmatory_capable", "pilot_audit_only", "assets_unavailable"):
            raise SpatialPerturbationRegistryError("unknown capability status")
        expected_status, expected_capable, expected_reasons = _semantic_state(
            specimens, cohorts, coverage, unavailable=status == "assets_unavailable"
        )
        if (status, capable, reasons) != (expected_status, expected_capable, expected_reasons):
            raise SpatialPerturbationRegistryError("status, capability flag, and reasons violate the capability matrix")
        expected = _identity(_result_unsigned_mapping(candidate_id, status, capable, specimens, cohorts, coverage, reasons))
        if _sha(self.capability_identity_sha256, "capability_identity_sha256") != expected:
            raise SpatialPerturbationRegistryError("capability identity does not match declared structure")
        for name, value in (("candidate_id", candidate_id), ("status", status), ("confirmatory_capable", capable),
                            ("biological_specimen_count", specimens), ("cohort_count", cohorts),
                            ("coverage", MappingProxyType(coverage)), ("blocking_reasons", reasons),
                            ("capability_identity_sha256", expected)):
            object.__setattr__(self, name, value)

    def to_mapping(self) -> dict[str, object]:
        item = _result_snapshot(self)
        return {
            "candidate_id": item.candidate_id, "status": item.status,
            "confirmatory_capable": item.confirmatory_capable,
            "biological_specimen_count": item.biological_specimen_count,
            "cohort_count": item.cohort_count, "coverage": dict(item.coverage),
            "blocking_reasons": list(item.blocking_reasons),
            "capability_identity_sha256": item.capability_identity_sha256,
        }


def _candidate_snapshot(value: object) -> BridgeCandidate:
    if type(value) is not BridgeCandidate:
        raise SpatialPerturbationRegistryError("candidate must be BridgeCandidate")
    return _validated_candidate_declaration(BridgeCandidate(
        value.candidate_id, value.accession, value.platform, value.biological_specimens,
        value.sections_by_specimen, value.safe_control_label, value.perturbation_labels,
        value.source_uri, value.source_identity_sha256,
    ))


def _summary_snapshot(value: object) -> MetadataSummary:
    if type(value) is not MetadataSummary:
        raise SpatialPerturbationRegistryError("summary must be an immutable MetadataSummary")
    return MetadataSummary(value.candidate_id, value.accession, value.cohort_ids, value.biological_specimen_ids,
                           value.sections_by_specimen, value.block_ids, value.coordinate_available, value.coordinate_finite,
                           value.coordinate_count, value.measured_gene_names, value.measured_gene_count, value.perturbation_labels,
                           value.perturbation_label_counts, value.safe_control_counts, value.barcode_quality_counts,
                           value.label_quality_counts, value.specimen_cohort_assignments,
                           value.external_untouched_cohort_ids, value.per_specimen_coordinate_counts,
                           value.per_specimen_perturbation_counts, value.per_specimen_safe_control_counts,
                           value.per_specimen_barcode_valid_counts, value.per_specimen_label_valid_counts,
                           value.license_identity, value.source_identity_sha256,
                           value.executable_output_schema_capable)


def _result_unsigned_mapping(candidate_id: str, status: str, capable: bool, specimens: int, cohorts: int,
                             coverage: Mapping[str, float], reasons: tuple[str, ...]) -> dict[str, object]:
    return {"candidate_id": candidate_id, "status": status, "confirmatory_capable": capable,
            "biological_specimen_count": specimens, "cohort_count": cohorts,
            "coverage": dict(coverage), "blocking_reasons": list(reasons)}


def _result_snapshot(value: object) -> BridgeCapabilityResult:
    if type(value) is not BridgeCapabilityResult:
        raise SpatialPerturbationRegistryError("result must be BridgeCapabilityResult")
    return BridgeCapabilityResult(value.candidate_id, value.status, value.confirmatory_capable,
                                  value.biological_specimen_count, value.cohort_count, dict(value.coverage),
                                  value.blocking_reasons, value.capability_identity_sha256)


def metadata_summary_from_mapping(raw: object) -> MetadataSummary:
    if type(raw) is not dict:
        raise SpatialPerturbationRegistryError("metadata summary must be an exact JSON object")
    fields = tuple(MetadataSummary.__dataclass_fields__)
    if set(raw) != set(fields):
        raise SpatialPerturbationRegistryError("metadata summary has unknown or missing fields")
    return MetadataSummary(**cast(dict[str, object], raw))  # type: ignore[arg-type]


def _coverage(summary: MetadataSummary, candidate: BridgeCandidate) -> dict[str, float]:
    counts = dict(summary.perturbation_label_counts)
    controls = dict(summary.safe_control_counts)
    barcode = dict(summary.barcode_quality_counts)
    labels = dict(summary.label_quality_counts)
    assignments = dict(summary.specimen_cohort_assignments)
    assigned_cohorts = set(assignments.values())
    every_declared_cohort_is_assigned = bool(summary.cohort_ids) and set(summary.cohort_ids) == assigned_cohorts
    external_cohorts = set(summary.external_untouched_cohort_ids)
    external_is_assigned = bool(external_cohorts) and external_cohorts <= set(summary.cohort_ids) and external_cohorts <= assigned_cohorts
    development_is_assigned = any(
        cohort not in external_cohorts for cohort in assigned_cohorts & set(summary.cohort_ids)
    )
    def every_specimen_has(pairs: tuple[tuple[str, int], ...]) -> bool:
        return tuple(label for label, _ in pairs) == candidate.biological_specimens and all(
            count > 0 for _, count in pairs
        )
    return {
        "candidate_identity": float(summary.candidate_id == candidate.candidate_id and summary.accession == candidate.accession),
        "specimen_structure": float(bool(summary.biological_specimen_ids) and all(sections for _, sections in summary.sections_by_specimen)),
        "cohorts": float(
            every_declared_cohort_is_assigned
            and tuple(assignments) == candidate.biological_specimens
            and all(cohort in summary.cohort_ids for cohort in assignments.values())
        ),
        "coordinates": float(summary.coordinate_available and summary.coordinate_finite and every_specimen_has(summary.per_specimen_coordinate_counts)),
        "genes": float(summary.measured_gene_count > 0 and bool(summary.measured_gene_names)),
        "perturbation_labels": float(bool(summary.perturbation_labels) and all(counts.get(label, 0) > 0 for label in summary.perturbation_labels) and every_specimen_has(summary.per_specimen_perturbation_counts)),
        "safe_controls": float(
            controls.get(candidate.safe_control_label, 0)
            == sum(count for _, count in summary.per_specimen_safe_control_counts)
            and every_specimen_has(summary.per_specimen_safe_control_counts)
        ),
        "barcode_quality": float(
            barcode.get("valid", 0) == sum(count for _, count in summary.per_specimen_barcode_valid_counts)
            and every_specimen_has(summary.per_specimen_barcode_valid_counts)
        ),
        "label_quality": float(
            labels.get("valid", 0) == sum(count for _, count in summary.per_specimen_label_valid_counts)
            and every_specimen_has(summary.per_specimen_label_valid_counts)
        ),
        "license": float(summary.license_identity != "unavailable"),
        "source_identity": float(summary.source_identity_sha256 == candidate.source_identity_sha256),
        "output_schema": float(summary.executable_output_schema_capable),
        "registered_specimens": float(summary.biological_specimen_ids == candidate.biological_specimens),
        "registered_sections": float(summary.sections_by_specimen == candidate.sections_by_specimen),
        "registered_perturbation_labels": float(summary.perturbation_labels == candidate.perturbation_labels),
        "external_cohort": float(external_is_assigned),
        "development_cohort": float(development_is_assigned),
    }


def audit_bridge_capability(candidate: BridgeCandidate, summary: MetadataSummary) -> BridgeCapabilityResult:
    """Classify structural readiness without opening assay values or outcomes."""
    frozen_candidate = _validated_candidate_declaration(_candidate_snapshot(candidate))
    frozen_summary = _summary_snapshot(summary)
    coverage = _coverage(frozen_summary, frozen_candidate)
    unavailable = "asset_metadata_unavailable" in frozen_summary.block_ids
    status, capable, frozen_reasons = _semantic_state(
        len(frozen_candidate.biological_specimens), len(frozen_summary.cohort_ids), coverage,
        unavailable=unavailable,
    )
    identity = _identity(_result_unsigned_mapping(frozen_candidate.candidate_id, status, capable,
                                                  len(frozen_candidate.biological_specimens), len(frozen_summary.cohort_ids),
                                                  coverage, frozen_reasons))
    return BridgeCapabilityResult(frozen_candidate.candidate_id, status, capable,
                                  len(frozen_candidate.biological_specimens), len(frozen_summary.cohort_ids),
                                  coverage, frozen_reasons, identity)


@dataclass(frozen=True, slots=True)
class _BoundDirectories:
    """A root-to-parent directory walk bound only through file descriptors."""

    root_fd: int
    directory_fds: tuple[int, ...]
    component_names: tuple[str, ...]
    component_identities: tuple[tuple[int, int], ...]

    @property
    def parent_fd(self) -> int:
        return self.directory_fds[-1] if self.directory_fds else self.root_fd

    def verify(self, label: str) -> None:
        current_fd = self.root_fd
        for name, expected, next_fd in zip(
            self.component_names, self.component_identities, self.directory_fds
        ):
            try:
                named = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                bound = os.fstat(next_fd)
            except OSError as exc:
                raise SpatialPerturbationRegistryError(
                    f"{label} directory path changed while bound"
                ) from exc
            if (
                not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(bound.st_mode)
                or (named.st_dev, named.st_ino) != expected
                or (bound.st_dev, bound.st_ino) != expected
            ):
                raise SpatialPerturbationRegistryError(
                    f"{label} directory path changed while bound"
                )
            current_fd = next_fd

    def close(self) -> None:
        failures: list[OSError] = []
        for descriptor in reversed((*self.directory_fds, self.root_fd)):
            try:
                os.close(descriptor)
            except OSError as exc:
                failures.append(exc)
        if failures:
            raise SpatialPerturbationRegistryError("bound directory close failed") from failures[0]


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise SpatialPerturbationRegistryError(
            "safe component-wise directory traversal is unavailable"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _bound_parent(path: str | Path, label: str) -> tuple[_BoundDirectories, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    leaf = absolute.name
    if not leaf or leaf in (".", ".."):
        raise SpatialPerturbationRegistryError(f"{label} must name a leaf file")
    names = absolute.parent.parts[1:]
    try:
        root_fd = os.open(os.sep, _directory_flags())
    except OSError as exc:
        raise SpatialPerturbationRegistryError(f"{label} trusted root is unavailable") from exc
    directory_fds: list[int] = []
    identities: list[tuple[int, int]] = []
    current_fd = root_fd
    try:
        for name in names:
            descriptor: int | None = None
            try:
                descriptor = os.open(name, _directory_flags(), dir_fd=current_fd)
                details = os.fstat(descriptor)
                if not stat.S_ISDIR(details.st_mode):
                    raise SpatialPerturbationRegistryError(
                        f"{label} has a non-directory ancestor"
                    )
            except BaseException:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                raise
            assert descriptor is not None
            directory_fds.append(descriptor)
            identities.append((details.st_dev, details.st_ino))
            current_fd = descriptor
        bound = _BoundDirectories(
            root_fd, tuple(directory_fds), tuple(names), tuple(identities)
        )
        bound.verify(label)
        return bound, leaf
    except FileNotFoundError:
        for descriptor in reversed((*directory_fds, root_fd)):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except OSError as exc:
        for descriptor in reversed((*directory_fds, root_fd)):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise SpatialPerturbationRegistryError(
            f"{label} parent is unavailable or unsafe"
        ) from exc
    except BaseException:
        for descriptor in reversed((*directory_fds, root_fd)):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _safe_regular_payload_at(
    bound: _BoundDirectories, leaf: str, label: str, *, maximum: int = MAXIMUM_REGISTRY_BYTES
) -> bytes:
    bound.verify(label)
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=bound.parent_fd,
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SpatialPerturbationRegistryError(f"{label} is not a safe regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size < 1 or before.st_size > maximum:
            raise SpatialPerturbationRegistryError(f"{label} violates safe file limits")
        parts: list[bytes] = []
        observed = 0
        while True:
            block = os.read(descriptor, min(65536, maximum + 1))
            if not block:
                break
            observed += len(block)
            if observed > maximum:
                raise SpatialPerturbationRegistryError(f"{label} exceeds the byte limit")
            parts.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_nlink)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_nlink)
    if identity_before != identity_after or observed != before.st_size:
        raise SpatialPerturbationRegistryError(f"{label} changed while being read")
    bound.verify(label)
    return b"".join(parts)


def _json_payload(payload: bytes, label: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpatialPerturbationRegistryError(f"{label} is not UTF-8") from exc
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if ord(character) < 32 and character not in "\n\r\t":
            raise SpatialPerturbationRegistryError(f"{label} contains control text")
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAXIMUM_NESTING:
                raise SpatialPerturbationRegistryError(f"{label} is nested too deeply")
        elif character in "]}":
            depth -= 1
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SpatialPerturbationRegistryError(f"{label} has duplicate key {key}")
            result[key] = value
        return result
    def reject_number(value: str) -> int:
        if len(value.lstrip("-")) > 12:
            raise SpatialPerturbationRegistryError(f"{label} has an oversized integer")
        return int(value)
    try:
        return json.loads(
            text,
            object_pairs_hook=unique,
            parse_int=reject_number,
            parse_float=lambda value: (_ for _ in ()).throw(
                SpatialPerturbationRegistryError(f"{label} does not permit JSON floats")
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                SpatialPerturbationRegistryError(f"{label} has invalid JSON scalar {value}")
            ),
        )
    except (json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise SpatialPerturbationRegistryError(f"{label} is not valid JSON") from exc


def load_bridge_candidates(path: str | Path) -> Mapping[str, BridgeCandidate]:
    try:
        bound, leaf = _bound_parent(path, "registry")
    except FileNotFoundError as exc:
        raise SpatialPerturbationRegistryError("registry parent is unavailable") from exc
    try:
        try:
            payload = _safe_regular_payload_at(bound, leaf, "registry")
        except FileNotFoundError as exc:
            raise SpatialPerturbationRegistryError("registry is unavailable") from exc
    finally:
        bound.close()
    raw = _json_payload(payload, "registry")
    if type(raw) is not dict or set(raw) != {"schema_version", "candidates"}:
        raise SpatialPerturbationRegistryError("registry has unknown or missing fields")
    if raw["schema_version"] != "spatial_perturbation_bridge_candidates_v1":
        raise SpatialPerturbationRegistryError("registry schema version is not frozen")
    values = _items(raw["candidates"], "candidates")
    if not values:
        raise SpatialPerturbationRegistryError("registry has no candidates")
    candidates: dict[str, BridgeCandidate] = {}
    expected = set(BridgeCandidate.__dataclass_fields__)
    for index, raw_candidate in enumerate(values):
        if type(raw_candidate) is not dict or set(raw_candidate) != expected:
            raise SpatialPerturbationRegistryError(f"candidate {index} has unknown or missing fields")
        item = _validated_candidate_declaration(
            BridgeCandidate(**cast(dict[str, object], raw_candidate))  # type: ignore[arg-type]
        )
        if item.candidate_id in candidates:
            raise SpatialPerturbationRegistryError("candidate IDs must be unique and ordered")
        candidates[item.candidate_id] = item
    if tuple(candidates) != _FROZEN_CANDIDATE_ORDER:
        raise SpatialPerturbationRegistryError("candidate registry order must remain frozen")
    _validated_candidate_declaration(candidates[_FROZEN_CANDIDATE_ORDER[0]])
    return MappingProxyType(candidates)


def unavailable_metadata_summary(candidate: BridgeCandidate) -> MetadataSummary:
    item = _candidate_snapshot(candidate)
    return MetadataSummary(
        candidate_id=item.candidate_id, accession=item.accession, cohort_ids=(),
        biological_specimen_ids=(), sections_by_specimen=(),
        block_ids=("asset_metadata_unavailable",), coordinate_available=False,
        coordinate_finite=False, coordinate_count=0, measured_gene_names=(),
        measured_gene_count=0, perturbation_labels=(), perturbation_label_counts=(),
        safe_control_counts=(), barcode_quality_counts=(), label_quality_counts=(),
        specimen_cohort_assignments=(), external_untouched_cohort_ids=(),
        per_specimen_coordinate_counts=(), per_specimen_perturbation_counts=(),
        per_specimen_safe_control_counts=(), per_specimen_barcode_valid_counts=(),
        per_specimen_label_valid_counts=(), license_identity="unavailable",
        source_identity_sha256=item.source_identity_sha256,
        executable_output_schema_capable=False,
    )


def load_asset_metadata(asset_root: str | Path, candidate: BridgeCandidate) -> MetadataSummary:
    try:
        bound, leaf = _bound_parent(Path(asset_root) / "metadata_summary.json", "asset metadata")
    except FileNotFoundError:
        return unavailable_metadata_summary(candidate)
    try:
        try:
            payload = _safe_regular_payload_at(bound, leaf, "asset metadata")
        except FileNotFoundError:
            return unavailable_metadata_summary(candidate)
    finally:
        bound.close()
    return metadata_summary_from_mapping(_json_payload(payload, "asset metadata"))


def _link_anonymous_noreplace(anonymous_fd: int, parent_fd: int, output: str) -> None:
    """Link the exact anonymous inode into *parent_fd* without replacement."""
    try:
        library = ctypes.CDLL(None, use_errno=True)
        linkat = library.linkat
    except (AttributeError, OSError) as exc:
        raise SpatialPerturbationRegistryError(
            "anonymous exclusive publication is unavailable"
        ) from exc
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    direct = linkat(
        anonymous_fd,
        b"",
        parent_fd,
        os.fsencode(output),
        _AT_EMPTY_PATH,
    )
    if direct == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise SpatialPerturbationRegistryError("refusing to overwrite output")
    unavailable = {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.ENOENT, errno.EPERM}
    if error not in unavailable:
        raise SpatialPerturbationRegistryError("anonymous exclusive publication failed") from OSError(error, os.strerror(error))
    source = os.fsencode(f"/proc/self/fd/{anonymous_fd}")
    fallback = linkat(
        _AT_FDCWD,
        source,
        parent_fd,
        os.fsencode(output),
        _AT_SYMLINK_FOLLOW,
    )
    if fallback == 0:
        return
    fallback_error = ctypes.get_errno()
    if fallback_error in (errno.EEXIST, errno.ENOTEMPTY):
        raise SpatialPerturbationRegistryError("refusing to overwrite output")
    raise SpatialPerturbationRegistryError("anonymous exclusive publication failed") from OSError(
        fallback_error, os.strerror(fallback_error)
    )


def write_bridge_capability_exclusively(
    path: str | Path, result: BridgeCapabilityResult, *, candidate: BridgeCandidate
) -> None:
    """Publish one JSON record with Linux no-clobber rename and durable fsync."""
    frozen_result = _result_snapshot(result)
    frozen_candidate = _validated_candidate_declaration(_candidate_snapshot(candidate))
    if frozen_candidate.candidate_id != frozen_result.candidate_id:
        raise SpatialPerturbationRegistryError("candidate does not match capability result")
    if frozen_result.biological_specimen_count != len(frozen_candidate.biological_specimens):
        raise SpatialPerturbationRegistryError("result specimen count does not match candidate declaration")
    record = frozen_result.to_mapping()
    try:
        bound, output_name = _bound_parent(path, "output")
    except FileNotFoundError as exc:
        raise SpatialPerturbationRegistryError("output parent is unavailable") from exc
    parent_fd = bound.parent_fd
    anonymous_fd: int | None = None
    committed = False
    primary_error: BaseException | None = None
    try:
        bound.verify("output")
        try:
            os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SpatialPerturbationRegistryError("refusing to overwrite output")
        if not hasattr(os, "O_TMPFILE"):
            raise SpatialPerturbationRegistryError("anonymous exclusive publication is unavailable")
        try:
            anonymous_fd = os.open(
                ".",
                os.O_TMPFILE | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise SpatialPerturbationRegistryError(
                "anonymous exclusive publication is unavailable"
            ) from exc
        payload = _canonical_bytes(record) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(anonymous_fd, view)
            if written < 1:
                raise SpatialPerturbationRegistryError("short output write")
            view = view[written:]
        os.fsync(anonymous_fd)
        details = os.fstat(anonymous_fd)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 0:
            raise SpatialPerturbationRegistryError("anonymous output changed")
        bound.verify("output")
        _link_anonymous_noreplace(anonymous_fd, parent_fd, output_name)
        published = os.fstat(anonymous_fd)
        if not stat.S_ISREG(published.st_mode) or published.st_nlink != 1:
            raise SpatialPerturbationRegistryError("published anonymous output changed")
        destination = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(destination.st_mode)
            or destination.st_nlink != 1
            or (destination.st_dev, destination.st_ino)
            != (published.st_dev, published.st_ino)
        ):
            raise SpatialPerturbationRegistryError("published anonymous output changed")
        os.fsync(parent_fd)
        bound.verify("output")
        committed = True
    except OSError as exc:
        primary_error = exc
        raise SpatialPerturbationRegistryError("output publication failed") from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_error: BaseException | None = None
        if anonymous_fd is not None:
            try:
                os.close(anonymous_fd)
            except OSError as exc:
                close_error = exc
        try:
            bound.close()
        except BaseException as exc:
            if close_error is None:
                close_error = exc
        if primary_error is None and close_error is not None:
            raise SpatialPerturbationRegistryError("output cleanup failed") from close_error
        # An unpublished anonymous inode disappears on close.  After publication,
        # uncertainty preserves the destination and never removes a pathname.
        if not committed:
            pass
