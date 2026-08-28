"""Frozen animal-level bridge splits and split-bound eligibility evidence.

The module is outcome blind.  A split manifest owns every statistical unit and
the complete row/block provenance needed to check later cell selections.
Eligibility evidence cannot introduce units, counters, genes, animals, or a
standalone frozen-unit digest.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, cast
import unicodedata

from src.evaluation.spatial_perturbation_registry import (
    BridgeCandidate,
    BridgeCapabilityResult,
    MetadataSummary,
    SpatialPerturbationRegistryError,
    audit_bridge_capability,
    metadata_summary_from_mapping,
)


MIN_SOURCE_CELLS = 20
MIN_SAFE_SOURCE_CELLS = 20
MIN_BAND_NEIGHBOURS = 50
MIN_CELL_TYPE_NEIGHBOURS = 30
MIN_SPATIAL_BLOCKS = 3
MIN_COVERAGE = 0.80
MAX_ABSTENTION = 0.20

# Compatibility exports only.  Fresh literals from _science() remain truth.
_SPLIT_SEED = 11
_PRIMARY_BANDS = ("proximal", "local")

_MAX_TEXT = 256
_MAX_ITEMS = 100_000
_MAX_ROWS = 10_000_000
_MAX_BLOCK_GRAPH_NODES = 4_096
_MAX_FROZEN_CONTEXTS = 50_000
_SHA = re.compile(r"[0-9a-f]{64}")
_ROLES = (
    "perturbation_source",
    "safe_source",
    "neighbour",
)
_REASON_ORDER = (
    "insufficient_perturbation_coverage",
    "insufficient_safe_control_coverage",
    "insufficient_spatial_blocks",
    "insufficient_safe_control_spatial_blocks",
    "insufficient_band_neighbours",
    "insufficient_safe_control_band_neighbours",
    "target_gene_not_measurable",
    "insufficient_primary_unit_coverage",
    "excessive_abstention",
)


def _science() -> dict[str, object]:
    return {
        "split_seed": 11,
        "primary_bands": ("proximal", "local"),
        "minimum_source_cells": 20,
        "minimum_safe_source_cells": 20,
        "minimum_band_neighbours": 50,
        "minimum_cell_type_neighbours": 30,
        "minimum_spatial_blocks": 3,
        "minimum_coverage": Fraction(4, 5),
        "maximum_abstention": Fraction(1, 5),
    }


class SpatialPerturbationSplitError(ValueError):
    """The frozen split or its eligibility evidence is inconsistent."""


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT:
        raise SpatialPerturbationSplitError(
            f"{name} must be bounded non-empty built-in NFC text"
        )
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SpatialPerturbationSplitError(f"{name} must be trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise SpatialPerturbationSplitError(f"{name} contains unsafe control text")
    return value


def _sha(value: object, name: str) -> str:
    text = _safe_text(value, name)
    if _SHA.fullmatch(text) is None:
        raise SpatialPerturbationSplitError(f"{name} must be a lowercase SHA-256")
    return text


def _integer(value: object, name: str, *, maximum: int = _MAX_ROWS) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise SpatialPerturbationSplitError(
            f"{name} must be a bounded nonnegative built-in integer"
        )
    return value


def _items(value: object, name: str, *, maximum: int = _MAX_ITEMS) -> tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise SpatialPerturbationSplitError(f"{name} must be a built-in list or tuple")
    result = cast(list[object] | tuple[object, ...], value)
    if len(result) > maximum:
        raise SpatialPerturbationSplitError(f"{name} exceeds the item limit")
    return tuple(result)


def _checked_complete_graph_size(block_counts: object) -> int:
    counts = _items(block_counts, "block_counts")
    total = 0
    for index, raw_count in enumerate(counts):
        count = _integer(raw_count, f"block_counts[{index}]", maximum=_MAX_ROWS)
        pair_count = count * (count - 1) // 2
        if pair_count > _MAX_ITEMS - total:
            raise SpatialPerturbationSplitError(
                "complete block graph exceeds the safe pair-state limit"
            )
        total += pair_count
    return total


def _text_items(value: object, name: str, *, sort: bool = False) -> tuple[str, ...]:
    result = tuple(
        _safe_text(item, f"{name}[{index}]")
        for index, item in enumerate(_items(value, name))
    )
    if len(set(result)) != len(result):
        raise SpatialPerturbationSplitError(f"{name} must contain unique values")
    return tuple(sorted(result)) if sort else result


def _row_ids(value: object, name: str) -> tuple[int, ...]:
    result = tuple(
        _integer(item, f"{name}[{index}]")
        for index, item in enumerate(_items(value, name, maximum=_MAX_ROWS))
    )
    if len(set(result)) != len(result):
        raise SpatialPerturbationSplitError(f"{name} must contain unique row identities")
    return tuple(sorted(result))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class BridgeSplitRow:
    stable_row_id: int
    cell_id: str
    animal_id: str
    section_id: str
    spatial_block: str
    context_perturbation_id: str
    observed_label: str
    cell_type: str
    source_cell_type: str
    cell_role: str
    distance_band: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stable_row_id", _integer(self.stable_row_id, "stable_row_id")
        )
        for name in (
            "cell_id", "animal_id", "section_id", "spatial_block",
            "context_perturbation_id", "observed_label", "cell_type",
            "source_cell_type", "cell_role", "distance_band",
        ):
            object.__setattr__(self, name, _safe_text(getattr(self, name), name))
        if self.cell_role not in _ROLES:
            raise SpatialPerturbationSplitError("cell_role is not a frozen bridge role")
        bands = cast(tuple[str, str], _science()["primary_bands"])
        if self.cell_role.endswith("source"):
            if self.distance_band != "own":
                raise SpatialPerturbationSplitError("source rows must use the own band")
            if self.source_cell_type != self.cell_type:
                raise SpatialPerturbationSplitError(
                    "source rows must bind source_cell_type to their cell_type"
                )
        elif (
            self.context_perturbation_id != "unassigned"
            or self.observed_label != "unperturbed"
            or self.distance_band != "none"
            or self.source_cell_type != self.cell_type
        ):
            raise SpatialPerturbationSplitError(
                "atomic neighbour rows must not contain relation context"
            )


def _row_from(value: object, name: str) -> BridgeSplitRow:
    if type(value) is BridgeSplitRow:
        row = cast(BridgeSplitRow, value)
        return BridgeSplitRow(
            row.stable_row_id, row.cell_id, row.animal_id, row.section_id,
            row.spatial_block, row.context_perturbation_id, row.observed_label,
            row.cell_type, row.source_cell_type, row.cell_role, row.distance_band,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "stable_row_id", "cell_id", "animal_id", "section_id", "spatial_block",
            "context_perturbation_id", "observed_label", "cell_type", "cell_role",
            "source_cell_type", "distance_band",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeSplitRow(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BridgeSplitRow")


@dataclass(frozen=True, slots=True)
class BridgeBlockAdjacency:
    animal_id: str
    section_id: str
    first_block: str
    second_block: str
    adjacent: bool

    def __post_init__(self) -> None:
        animal = _safe_text(self.animal_id, "animal_id")
        section = _safe_text(self.section_id, "section_id")
        first = _safe_text(self.first_block, "first_block")
        second = _safe_text(self.second_block, "second_block")
        if first == second:
            raise SpatialPerturbationSplitError("a block pair cannot contain itself")
        if type(self.adjacent) is not bool:
            raise SpatialPerturbationSplitError("adjacent must be a built-in boolean")
        first, second = sorted((first, second))
        object.__setattr__(self, "animal_id", animal)
        object.__setattr__(self, "section_id", section)
        object.__setattr__(self, "first_block", first)
        object.__setattr__(self, "second_block", second)


def _adjacency_from(value: object, name: str) -> BridgeBlockAdjacency:
    if type(value) is BridgeBlockAdjacency:
        item = cast(BridgeBlockAdjacency, value)
        return BridgeBlockAdjacency(
            item.animal_id, item.section_id, item.first_block, item.second_block,
            item.adjacent,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        if set(raw) != {"animal_id", "section_id", "first_block", "second_block", "adjacent"}:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeBlockAdjacency(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BridgeBlockAdjacency")


@dataclass(frozen=True, slots=True)
class BridgeNeighbourRelation:
    """Frozen outcome-blind source-to-neighbour membership declaration."""

    relation_id: str
    animal_id: str
    section_id: str
    spatial_block: str
    source_cell_id: str
    neighbor_cell_id: str
    source_perturbation_id: str
    matched_perturbation_id: str
    source_cell_type: str
    neighbor_cell_type: str
    rank: int
    band: str
    contamination: bool
    is_safe_control: bool

    def __post_init__(self) -> None:
        values = {
            name: _safe_text(getattr(self, name), name)
            for name in (
                "animal_id", "section_id", "spatial_block", "source_cell_id",
                "neighbor_cell_id", "source_perturbation_id",
                "matched_perturbation_id", "source_cell_type",
                "neighbor_cell_type", "band",
            )
        }
        rank = _integer(self.rank, "rank", maximum=1_000_000)
        if rank < 1:
            raise SpatialPerturbationSplitError("rank must be positive")
        if values["source_cell_id"] == values["neighbor_cell_id"]:
            raise SpatialPerturbationSplitError(
                "a neighbour relation source and neighbor must be distinct"
            )
        contamination = self.contamination
        if type(contamination) is not bool:
            raise SpatialPerturbationSplitError(
                "contamination must be a built-in boolean"
            )
        if type(self.is_safe_control) is not bool:
            raise SpatialPerturbationSplitError(
                "is_safe_control must be a built-in boolean"
            )
        if values["band"] not in cast(tuple[str, str], _science()["primary_bands"]):
            raise SpatialPerturbationSplitError("relation band is not frozen")
        if (
            values["band"] == "proximal" and rank not in range(1, 6)
        ) or (
            values["band"] == "local" and rank not in range(6, 16)
        ):
            raise SpatialPerturbationSplitError("relation rank does not match its frozen band")
        identity_values: dict[str, object] = {
            **values, "rank": rank, "contamination": contamination,
            "is_safe_control": self.is_safe_control,
        }
        relation_id = _sha(self.relation_id, "relation_id")
        if relation_id != _context_id("bridge_neighbour_relation_v1", identity_values):
            raise SpatialPerturbationSplitError("relation identity does not match edge key")
        for name, value in identity_values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "relation_id", relation_id)


def freeze_bridge_neighbour_relation(
    animal_id: str,
    section_id: str,
    spatial_block: str,
    source_cell_id: str,
    neighbor_cell_id: str,
    source_perturbation_id: str,
    matched_perturbation_id: str,
    source_cell_type: str,
    neighbor_cell_type: str,
    rank: int,
    band: str,
    contamination: bool,
    is_safe_control: bool,
) -> BridgeNeighbourRelation:
    text_values = {
        name: _safe_text(value, name)
        for name, value in (
            ("animal_id", animal_id), ("section_id", section_id),
            ("spatial_block", spatial_block), ("source_cell_id", source_cell_id),
            ("neighbor_cell_id", neighbor_cell_id),
            ("source_perturbation_id", source_perturbation_id),
            ("matched_perturbation_id", matched_perturbation_id),
            ("source_cell_type", source_cell_type),
            ("neighbor_cell_type", neighbor_cell_type), ("band", band),
        )
    }
    rank = _integer(rank, "rank", maximum=1_000_000)
    if type(contamination) is not bool or type(is_safe_control) is not bool:
        raise SpatialPerturbationSplitError(
            "relation control flags must be built-in booleans"
        )
    values: dict[str, object] = {
        **text_values, "rank": rank,
        "contamination": contamination, "is_safe_control": is_safe_control,
    }
    return BridgeNeighbourRelation(
        _context_id("bridge_neighbour_relation_v1", values),
        text_values["animal_id"], text_values["section_id"],
        text_values["spatial_block"], text_values["source_cell_id"],
        text_values["neighbor_cell_id"], text_values["source_perturbation_id"],
        text_values["matched_perturbation_id"], text_values["source_cell_type"],
        text_values["neighbor_cell_type"], rank, text_values["band"],
        contamination, is_safe_control,
    )


def _relation_from(value: object, name: str) -> BridgeNeighbourRelation:
    expected = tuple(BridgeNeighbourRelation.__dataclass_fields__)
    if type(value) is BridgeNeighbourRelation:
        raw = {field: getattr(value, field) for field in expected}
    elif type(value) is dict and set(cast(dict[object, object], value)) == set(expected):
        raw = cast(dict[str, object], value)
    else:
        raise SpatialPerturbationSplitError(f"{name} must be BridgeNeighbourRelation")
    return BridgeNeighbourRelation(**raw)  # type: ignore[arg-type]


def _relation_mapping(item: BridgeNeighbourRelation) -> dict[str, object]:
    snapshot = _relation_from(item, "neighbour_relation")
    return {name: getattr(snapshot, name) for name in BridgeNeighbourRelation.__dataclass_fields__}


def _neighbour_table_identity(relations: tuple[BridgeNeighbourRelation, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b'{"relations":[')
    for index, item in enumerate(relations):
        if index:
            digest.update(b",")
        digest.update(_canonical_bytes(_relation_mapping(item)))
    digest.update(b'],"schema":"bridge_neighbour_table_v1"}')
    return digest.hexdigest()


def _target_pairs(value: object, name: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(_items(value, name)):
        pair = _items(item, f"{name}[{index}]")
        if len(pair) != 2:
            raise SpatialPerturbationSplitError(f"{name}[{index}] must have two items")
        pairs.append(
            (_safe_text(pair[0], f"{name}[{index}][0]"), _safe_text(pair[1], f"{name}[{index}][1]"))
        )
    if len({pair[0] for pair in pairs}) != len(pairs):
        raise SpatialPerturbationSplitError(f"{name} perturbations must be unique")
    return tuple(sorted(pairs))


def _validate_rows(
    rows: tuple[BridgeSplitRow, ...],
    perturbations: tuple[str, ...],
    neighbour_types: tuple[str, ...],
    safe_label: str,
) -> None:
    if not rows:
        raise SpatialPerturbationSplitError("rows must not be empty")
    if len({row.stable_row_id for row in rows}) != len(rows):
        raise SpatialPerturbationSplitError("stable_row_id values must be unique")
    if len({row.cell_id for row in rows}) != len(rows):
        raise SpatialPerturbationSplitError("cell_id values must be unique")
    section_animals: dict[str, str] = {}
    for row in rows:
        prior = section_animals.setdefault(row.section_id, row.animal_id)
        if prior != row.animal_id:
            raise SpatialPerturbationSplitError("a section cannot be assigned to two animals")
        if (
            row.cell_role == "perturbation_source"
            and row.context_perturbation_id not in perturbations
        ):
            raise SpatialPerturbationSplitError("row context perturbation is not registered")
        if row.cell_role == "perturbation_source" and row.observed_label != row.context_perturbation_id:
            raise SpatialPerturbationSplitError("perturbation source label does not match context")
        if row.cell_role == "safe_source" and (
            row.context_perturbation_id != safe_label
            or row.observed_label != safe_label
        ):
            raise SpatialPerturbationSplitError(
                "safe source context/label does not match frozen control"
            )
        if row.cell_role == "neighbour":
            if row.observed_label != "unperturbed" or row.cell_type not in neighbour_types:
                raise SpatialPerturbationSplitError("neighbour row labels are not frozen")


def _snapshot_registry_contract(
    candidate: object,
    summary: object,
    capability: object,
) -> tuple[BridgeCandidate, MetadataSummary, BridgeCapabilityResult]:
    try:
        if type(candidate) is not BridgeCandidate:
            raise SpatialPerturbationRegistryError("candidate type mismatch")
        candidate_snapshot = BridgeCandidate(**candidate.to_mapping())  # type: ignore[arg-type,union-attr]
        if type(summary) is not MetadataSummary:
            raise SpatialPerturbationRegistryError("summary type mismatch")
        summary_snapshot = metadata_summary_from_mapping(summary.to_mapping())  # type: ignore[union-attr]
        if type(capability) is not BridgeCapabilityResult:
            raise SpatialPerturbationRegistryError("capability type mismatch")
        capability_snapshot = BridgeCapabilityResult(**capability.to_mapping())  # type: ignore[arg-type,union-attr]
        expected = audit_bridge_capability(candidate_snapshot, summary_snapshot)
        if capability_snapshot.to_mapping() != expected.to_mapping():
            raise SpatialPerturbationRegistryError(
                "capability is not the audit of the bound candidate and metadata"
            )
        return candidate_snapshot, summary_snapshot, capability_snapshot
    except (SpatialPerturbationRegistryError, TypeError, AttributeError) as exc:
        raise SpatialPerturbationSplitError(
            "invalid upstream registry/capability contract"
        ) from exc


@dataclass(frozen=True, slots=True)
class BridgeSplitMetadata:
    rows: tuple[BridgeSplitRow, ...]
    gene_names: tuple[str, ...]
    perturbations: tuple[str, ...]
    neighbour_cell_types: tuple[str, ...]
    perturbation_targets: tuple[tuple[str, str], ...]
    block_adjacency: tuple[BridgeBlockAdjacency, ...]
    safe_control_label: str
    neighbour_relations: tuple[BridgeNeighbourRelation, ...]
    neighbour_table_identity_sha256: str
    candidate: BridgeCandidate
    registry_summary: MetadataSummary
    capability_result: BridgeCapabilityResult

    def __post_init__(self) -> None:
        rows = tuple(
            _row_from(item, f"rows[{index}]")
            for index, item in enumerate(_items(self.rows, "rows", maximum=_MAX_ROWS))
        )
        rows = tuple(sorted(rows, key=lambda row: row.stable_row_id))
        genes = _text_items(self.gene_names, "gene_names", sort=True)
        perturbations = _text_items(self.perturbations, "perturbations", sort=True)
        neighbour_types = _text_items(
            self.neighbour_cell_types, "neighbour_cell_types", sort=True
        )
        targets = _target_pairs(self.perturbation_targets, "perturbation_targets")
        safe_label = _safe_text(self.safe_control_label, "safe_control_label")
        candidate, summary, capability = _snapshot_registry_contract(
            self.candidate, self.registry_summary, self.capability_result
        )
        if not genes or not perturbations or not neighbour_types:
            raise SpatialPerturbationSplitError("genes, perturbations, and cell types are required")
        if tuple(pair[0] for pair in targets) != perturbations:
            raise SpatialPerturbationSplitError("each perturbation needs one frozen target gene")
        if any(target not in genes for _, target in targets):
            raise SpatialPerturbationSplitError("target gene is not in the measurable gene set")
        _validate_rows(rows, perturbations, neighbour_types, safe_label)
        source_perturbations = tuple(sorted({
            row.context_perturbation_id
            for row in rows
            if row.cell_role == "perturbation_source"
        }))
        if source_perturbations != perturbations:
            raise SpatialPerturbationSplitError(
                "registered perturbations must exactly match atomic source evidence"
            )
        adjacency = tuple(
            _adjacency_from(item, f"block_adjacency[{index}]")
            for index, item in enumerate(_items(self.block_adjacency, "block_adjacency"))
        )
        adjacency = tuple(
            sorted(adjacency, key=lambda item: (item.animal_id, item.section_id, item.first_block, item.second_block))
        )
        keys = {(row.animal_id, row.section_id, row.spatial_block) for row in rows}
        adjacency_keys = {
            (item.animal_id, item.section_id, item.first_block, item.second_block)
            for item in adjacency
        }
        if len(adjacency_keys) != len(adjacency):
            raise SpatialPerturbationSplitError("block adjacency records must be unique")
        for item in adjacency:
            if (
                (item.animal_id, item.section_id, item.first_block) not in keys
                or (item.animal_id, item.section_id, item.second_block) not in keys
            ):
                raise SpatialPerturbationSplitError("block adjacency endpoint is not registered")
        blocks_by_section: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            blocks_by_section.setdefault((row.animal_id, row.section_id), set()).add(
                row.spatial_block
            )
        _checked_complete_graph_size(
            tuple(len(blocks) for blocks in blocks_by_section.values())
        )
        expected_pairs = {
            (animal, section, first, second)
            for (animal, section), blocks in blocks_by_section.items()
            for first_index, first in enumerate(sorted(blocks))
            for second in sorted(blocks)[first_index + 1:]
        }
        if adjacency_keys != expected_pairs:
            raise SpatialPerturbationSplitError(
                "block graph must record an explicit state for every block pair"
            )
        relations = tuple(
            _relation_from(item, f"neighbour_relations[{index}]")
            for index, item in enumerate(
                _items(self.neighbour_relations, "neighbour_relations", maximum=_MAX_ROWS)
            )
        )
        relations = tuple(sorted(relations, key=lambda item: item.relation_id))
        if len({item.relation_id for item in relations}) != len(relations):
            raise SpatialPerturbationSplitError("neighbour relation keys must be unique")
        logical_keys = {
            (
                item.animal_id, item.section_id,
                item.matched_perturbation_id, item.neighbor_cell_id,
            )
            for item in relations
        }
        if len(logical_keys) != len(relations):
            raise SpatialPerturbationSplitError(
                "neighbour relation logical keys must be unique"
            )
        treatment_neighbour_contexts: dict[tuple[str, str, str], str] = {}
        for relation in relations:
            if relation.is_safe_control:
                continue
            key = (relation.animal_id, relation.section_id, relation.neighbor_cell_id)
            prior = treatment_neighbour_contexts.setdefault(
                key, relation.matched_perturbation_id
            )
            if prior != relation.matched_perturbation_id:
                raise SpatialPerturbationSplitError(
                    "a treatment neighbor cannot cross matched perturbations"
                )
        row_by_cell = {row.cell_id: row for row in rows}
        for relation in relations:
            source = row_by_cell.get(relation.source_cell_id)
            neighbor = row_by_cell.get(relation.neighbor_cell_id)
            if source is None or neighbor is None:
                raise SpatialPerturbationSplitError(
                    "neighbour relation endpoint is absent from atomic provenance"
                )
            expected_source_role = "safe_source" if relation.is_safe_control else "perturbation_source"
            expected_source_perturbation = (
                safe_label if relation.is_safe_control
                else relation.matched_perturbation_id
            )
            if (
                relation.matched_perturbation_id not in perturbations
                or relation.source_perturbation_id != expected_source_perturbation
                or source.cell_role != expected_source_role
                or source.animal_id != relation.animal_id
                or source.section_id != relation.section_id
                or source.context_perturbation_id != relation.source_perturbation_id
                or source.source_cell_type != relation.source_cell_type
                or neighbor.cell_role != "neighbour"
                or neighbor.animal_id != relation.animal_id
                or neighbor.section_id != relation.section_id
                or neighbor.spatial_block != relation.spatial_block
                or neighbor.cell_type != relation.neighbor_cell_type
            ):
                raise SpatialPerturbationSplitError(
                    "neighbour relation does not match frozen atomic provenance"
                )
        neighbour_identity = _sha(
            self.neighbour_table_identity_sha256,
            "neighbour_table_identity_sha256",
        )
        if neighbour_identity != _neighbour_table_identity(relations):
            raise SpatialPerturbationSplitError("neighbour-table identity mismatch")
        row_animals = tuple(sorted({row.animal_id for row in rows}))
        row_sections = tuple(
            (animal, tuple(sorted({row.section_id for row in rows if row.animal_id == animal})))
            for animal in row_animals
        )
        if (
            candidate.biological_specimens != row_animals
            or candidate.sections_by_specimen != row_sections
            or summary.biological_specimen_ids != row_animals
            or summary.sections_by_specimen != row_sections
            or candidate.safe_control_label != safe_label
            or dict(summary.safe_control_counts).get(safe_label, 0) <= 0
            or candidate.perturbation_labels != perturbations
            or summary.perturbation_labels != perturbations
            or summary.measured_gene_names != genes
            or set(summary.block_ids) != {row.spatial_block for row in rows}
        ):
            raise SpatialPerturbationSplitError(
                "split metadata does not exactly match the upstream registry declaration"
            )
        for name, value in (
            ("rows", rows), ("gene_names", genes), ("perturbations", perturbations),
            ("neighbour_cell_types", neighbour_types), ("perturbation_targets", targets),
            ("block_adjacency", adjacency), ("safe_control_label", safe_label),
            ("neighbour_relations", relations),
            ("neighbour_table_identity_sha256", neighbour_identity),
            ("candidate", candidate), ("registry_summary", summary),
            ("capability_result", capability),
        ):
            object.__setattr__(self, name, value)


def _context_id(prefix: str, mapping: Mapping[str, object]) -> str:
    return _identity({"schema": prefix, **mapping})


@dataclass(frozen=True, slots=True)
class FrozenPerturbationParent:
    parent_id: str
    animal_id: str
    perturbation_id: str
    target_gene: str

    def __post_init__(self) -> None:
        animal = _safe_text(self.animal_id, "animal_id")
        perturbation = _safe_text(self.perturbation_id, "perturbation_id")
        target = _safe_text(self.target_gene, "target_gene")
        parent_id = _sha(self.parent_id, "parent_id")
        expected = _context_id(
            "bridge_perturbation_parent_v1",
            {"animal_id": animal, "perturbation_id": perturbation, "target_gene": target},
        )
        if parent_id != expected:
            raise SpatialPerturbationSplitError("parent identity does not match context")
        object.__setattr__(self, "animal_id", animal)
        object.__setattr__(self, "perturbation_id", perturbation)
        object.__setattr__(self, "target_gene", target)
        object.__setattr__(self, "parent_id", parent_id)


def _make_parent(animal: str, perturbation: str, target: str) -> FrozenPerturbationParent:
    mapping = {"animal_id": animal, "perturbation_id": perturbation, "target_gene": target}
    return FrozenPerturbationParent(
        _context_id("bridge_perturbation_parent_v1", mapping), animal, perturbation, target
    )


def _parent_from(value: object, name: str) -> FrozenPerturbationParent:
    if type(value) is FrozenPerturbationParent:
        item = cast(FrozenPerturbationParent, value)
        return FrozenPerturbationParent(
            item.parent_id, item.animal_id, item.perturbation_id, item.target_gene
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        if set(raw) != {"parent_id", "animal_id", "perturbation_id", "target_gene"}:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return FrozenPerturbationParent(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be FrozenPerturbationParent")


@dataclass(frozen=True, slots=True)
class FrozenPrimaryUnit:
    unit_id: str
    animal_id: str
    perturbation_id: str
    target_gene: str
    neighbour_cell_type: str
    band: str

    def __post_init__(self) -> None:
        values = {
            "animal_id": _safe_text(self.animal_id, "animal_id"),
            "perturbation_id": _safe_text(self.perturbation_id, "perturbation_id"),
            "target_gene": _safe_text(self.target_gene, "target_gene"),
            "neighbour_cell_type": _safe_text(self.neighbour_cell_type, "neighbour_cell_type"),
            "band": _safe_text(self.band, "band"),
        }
        if values["band"] not in cast(tuple[str, str], _science()["primary_bands"]):
            raise SpatialPerturbationSplitError("primary unit band is not frozen")
        unit_id = _sha(self.unit_id, "unit_id")
        if unit_id != _context_id("bridge_primary_unit_v1", values):
            raise SpatialPerturbationSplitError("unit identity does not match context")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "unit_id", unit_id)


def _make_unit(
    animal: str, perturbation: str, target: str, neighbour_type: str, band: str
) -> FrozenPrimaryUnit:
    mapping = {
        "animal_id": animal, "perturbation_id": perturbation, "target_gene": target,
        "neighbour_cell_type": neighbour_type, "band": band,
    }
    return FrozenPrimaryUnit(
        _context_id("bridge_primary_unit_v1", mapping),
        animal, perturbation, target, neighbour_type, band,
    )


def _unit_from(value: object, name: str) -> FrozenPrimaryUnit:
    if type(value) is FrozenPrimaryUnit:
        item = cast(FrozenPrimaryUnit, value)
        return FrozenPrimaryUnit(
            item.unit_id, item.animal_id, item.perturbation_id, item.target_gene,
            item.neighbour_cell_type, item.band,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "unit_id", "animal_id", "perturbation_id", "target_gene",
            "neighbour_cell_type", "band",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return FrozenPrimaryUnit(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be FrozenPrimaryUnit")


def _row_mapping(row: BridgeSplitRow) -> dict[str, object]:
    return {
        "stable_row_id": row.stable_row_id, "cell_id": row.cell_id,
        "animal_id": row.animal_id, "section_id": row.section_id,
        "spatial_block": row.spatial_block,
        "context_perturbation_id": row.context_perturbation_id,
        "observed_label": row.observed_label, "cell_type": row.cell_type,
        "source_cell_type": row.source_cell_type,
        "cell_role": row.cell_role, "distance_band": row.distance_band,
    }


def _adjacency_mapping(item: BridgeBlockAdjacency) -> dict[str, object]:
    snapshot = _adjacency_from(item, "block_adjacency")
    return {
        "animal_id": snapshot.animal_id, "section_id": snapshot.section_id,
        "first_block": snapshot.first_block, "second_block": snapshot.second_block,
        "adjacent": snapshot.adjacent,
    }


def _parent_mapping(item: FrozenPerturbationParent) -> dict[str, object]:
    snapshot = _parent_from(item, "parent")
    return {
        "parent_id": snapshot.parent_id, "animal_id": snapshot.animal_id,
        "perturbation_id": snapshot.perturbation_id, "target_gene": snapshot.target_gene,
    }


def _unit_mapping(item: FrozenPrimaryUnit) -> dict[str, object]:
    snapshot = _unit_from(item, "unit")
    return {
        "unit_id": snapshot.unit_id, "animal_id": snapshot.animal_id,
        "perturbation_id": snapshot.perturbation_id, "target_gene": snapshot.target_gene,
        "neighbour_cell_type": snapshot.neighbour_cell_type, "band": snapshot.band,
    }


def _manifest_unsigned(manifest: "BridgeSplitManifest") -> dict[str, object]:
    return {
        "split_id": manifest.split_id, "split_role": manifest.split_role,
        "split_seed": manifest.split_seed,
        "development_animals": list(manifest.development_animals),
        "evaluation_animals": list(manifest.evaluation_animals),
        "train_rows": list(manifest.train_rows), "tune_rows": list(manifest.tune_rows),
        "evaluation_rows": list(manifest.evaluation_rows),
        "gene_names": list(manifest.gene_names), "perturbations": list(manifest.perturbations),
        "registered_perturbations": list(manifest.registered_perturbations),
        "secondary_perturbations": list(manifest.secondary_perturbations),
        "development_only_perturbations": list(
            manifest.development_only_perturbations
        ),
        "row_provenance": [_row_mapping(row) for row in manifest.row_provenance],
        "perturbation_parents": [_parent_mapping(item) for item in manifest.perturbation_parents],
        "primary_units": [_unit_mapping(item) for item in manifest.primary_units],
        "neighbour_cell_types": list(manifest.neighbour_cell_types),
        "perturbation_targets": [list(item) for item in manifest.perturbation_targets],
        "block_adjacency": [_adjacency_mapping(item) for item in manifest.block_adjacency],
        "safe_control_label": manifest.safe_control_label,
        "neighbour_table": {
            "schema": "bridge_neighbour_table_v1",
            "relation_count": len(manifest.neighbour_relations),
            "identity_sha256": manifest.neighbour_table_identity_sha256,
        },
        "candidate": manifest.candidate.to_mapping(),
        "registry_summary": manifest.registry_summary.to_mapping(),
        "capability_result": manifest.capability_result.to_mapping(),
        "candidate_identity_sha256": manifest.candidate_identity_sha256,
        "metadata_identity_sha256": manifest.metadata_identity_sha256,
        "capability_identity_sha256": manifest.capability_identity_sha256,
    }


def _checked_cartesian_size(*factors: int) -> int:
    size = 1
    for index, factor in enumerate(factors):
        factor = _integer(factor, f"Cartesian factor[{index}]", maximum=_MAX_ROWS)
        if factor and size > _MAX_FROZEN_CONTEXTS // factor:
            raise SpatialPerturbationSplitError(
                "frozen Cartesian size exceeds the safe context limit"
            )
        size *= factor
    return size


@dataclass(frozen=True, slots=True)
class BridgeSplitManifest:
    split_id: str
    split_role: str
    split_seed: int
    development_animals: tuple[str, ...]
    evaluation_animals: tuple[str, ...]
    train_rows: tuple[int, ...]
    tune_rows: tuple[int, ...]
    evaluation_rows: tuple[int, ...]
    gene_names: tuple[str, ...]
    perturbations: tuple[str, ...]
    registered_perturbations: tuple[str, ...]
    secondary_perturbations: tuple[str, ...]
    development_only_perturbations: tuple[str, ...]
    split_identity_sha256: str
    row_provenance: tuple[BridgeSplitRow, ...]
    perturbation_parents: tuple[FrozenPerturbationParent, ...]
    primary_units: tuple[FrozenPrimaryUnit, ...]
    neighbour_cell_types: tuple[str, ...]
    perturbation_targets: tuple[tuple[str, str], ...]
    block_adjacency: tuple[BridgeBlockAdjacency, ...]
    safe_control_label: str
    neighbour_relations: tuple[BridgeNeighbourRelation, ...]
    neighbour_table_identity_sha256: str
    candidate: BridgeCandidate
    registry_summary: MetadataSummary
    capability_result: BridgeCapabilityResult
    candidate_identity_sha256: str
    metadata_identity_sha256: str
    capability_identity_sha256: str

    def __post_init__(self) -> None:
        split_id = _safe_text(self.split_id, "split_id")
        split_role = _safe_text(self.split_role, "split_role")
        role_namespaces = {
            "pilot": "pilot_leave_one_animal_out:",
            "generic": "generic_partition:",
            "confirmatory": "confirmatory_partition:",
        }
        if split_role not in role_namespaces:
            raise SpatialPerturbationSplitError("split_role is not a frozen split role")
        if (
            split_id == role_namespaces[split_role]
            or not split_id.startswith(role_namespaces[split_role])
            or any(
                split_id.startswith(namespace)
                for role, namespace in role_namespaces.items()
                if role != split_role
            )
        ):
            raise SpatialPerturbationSplitError(
                "split_id namespace does not match split_role"
            )
        split_seed = _integer(self.split_seed, "split_seed")
        if split_seed != _science()["split_seed"]:
            raise SpatialPerturbationSplitError("split seed is frozen to 11")
        development = _text_items(self.development_animals, "development_animals", sort=True)
        evaluation = _text_items(self.evaluation_animals, "evaluation_animals", sort=True)
        if not development or not evaluation or set(development) & set(evaluation):
            raise SpatialPerturbationSplitError(
                "development and evaluation animals must be nonempty and disjoint"
            )
        if split_role == "confirmatory" and len(development) + len(evaluation) < 5:
            raise SpatialPerturbationSplitError(
                "confirmatory role requires at least five animals"
            )
        train = _row_ids(self.train_rows, "train_rows")
        tune = _row_ids(self.tune_rows, "tune_rows")
        held_out = _row_ids(self.evaluation_rows, "evaluation_rows")
        if not train:
            raise SpatialPerturbationSplitError(
                "a legal split requires a nonempty train partition"
            )
        if split_role == "confirmatory" and (not tune or not held_out):
            raise SpatialPerturbationSplitError(
                "confirmatory role requires three nonempty train, tune, and evaluation partitions"
            )
        genes = _text_items(self.gene_names, "gene_names", sort=True)
        perturbations = _text_items(self.perturbations, "perturbations", sort=True)
        registered = _text_items(
            self.registered_perturbations, "registered_perturbations", sort=True
        )
        secondary = _text_items(
            self.secondary_perturbations, "secondary_perturbations", sort=True
        )
        development_only = _text_items(
            self.development_only_perturbations,
            "development_only_perturbations", sort=True,
        )
        if split_role == "pilot" and (
            len(development) != 2
            or len(evaluation) != 1
            or tune
            or split_id != f"pilot_leave_one_animal_out:{evaluation[0]}"
        ):
            raise SpatialPerturbationSplitError(
                "pilot role requires exact three-animal leave-one-out with empty tune"
            )
        neighbour_types = _text_items(
            self.neighbour_cell_types, "neighbour_cell_types", sort=True
        )
        targets = _target_pairs(self.perturbation_targets, "perturbation_targets")
        safe_label = _safe_text(self.safe_control_label, "safe_control_label")
        rows = tuple(
            _row_from(item, f"row_provenance[{index}]")
            for index, item in enumerate(_items(self.row_provenance, "row_provenance", maximum=_MAX_ROWS))
        )
        rows = tuple(sorted(rows, key=lambda row: row.stable_row_id))
        metadata = BridgeSplitMetadata(
            rows, genes, registered, neighbour_types, targets,
            self.block_adjacency, safe_label, self.neighbour_relations,
            self.neighbour_table_identity_sha256, self.candidate,
            self.registry_summary, self.capability_result,
        )
        rows = metadata.rows
        if split_role == "confirmatory":
            if (
                metadata.capability_result.status != "confirmatory_capable"
                or metadata.capability_result.confirmatory_capable is not True
            ):
                raise SpatialPerturbationSplitError(
                    "confirmatory role requires confirmatory_capable upstream status"
                )
            cohort_by_animal = dict(
                metadata.registry_summary.specimen_cohort_assignments
            )
            external_cohorts = set(
                metadata.registry_summary.external_untouched_cohort_ids
            )
            external_animals = {
                animal for animal, cohort in cohort_by_animal.items()
                if cohort in external_cohorts
            }
            nonexternal_animals = set(cohort_by_animal) - external_animals
            if (
                set(evaluation) != external_animals
                or set(development) != nonexternal_animals
            ):
                raise SpatialPerturbationSplitError(
                    "confirmatory evaluation animals must exactly equal untouched external cohort animals"
                )
        animals = tuple(sorted({row.animal_id for row in rows}))
        if tuple(sorted(development + evaluation)) != animals:
            raise SpatialPerturbationSplitError("manifest animals do not match row provenance")
        row_by_id = {row.stable_row_id: row for row in rows}
        all_ids = set(row_by_id)
        if set(train) | set(tune) | set(held_out) != all_ids or (
            set(train) & set(tune) or set(train) & set(held_out) or set(tune) & set(held_out)
        ):
            raise SpatialPerturbationSplitError(
                "rows do not match the exact whole-animal partition"
            )
        train_animals = {row_by_id[item].animal_id for item in train}
        tune_animals = {row_by_id[item].animal_id for item in tune}
        held_animals = {row_by_id[item].animal_id for item in held_out}
        if (
            train_animals & tune_animals
            or train_animals | tune_animals != set(development)
            or held_animals != set(evaluation)
            or train != tuple(row.stable_row_id for row in rows if row.animal_id in train_animals)
            or tune != tuple(row.stable_row_id for row in rows if row.animal_id in tune_animals)
            or held_out != tuple(row.stable_row_id for row in rows if row.animal_id in held_animals)
        ):
            raise SpatialPerturbationSplitError("rows do not match the whole-animal partition")
        target_map = dict(targets)
        if tuple(target_map) != registered or any(value not in genes for value in target_map.values()):
            raise SpatialPerturbationSplitError("manifest target genes are not frozen/measurable")
        development_perturbations = {
            row.context_perturbation_id for row in rows
            if row.animal_id in development and row.cell_role == "perturbation_source"
        }
        evaluation_perturbations = {
            row.context_perturbation_id for row in rows
            if row.animal_id in evaluation and row.cell_role == "perturbation_source"
        }
        expected_primary = tuple(sorted(development_perturbations & evaluation_perturbations))
        expected_secondary = tuple(sorted(evaluation_perturbations - development_perturbations))
        expected_development_only = tuple(
            sorted(development_perturbations - evaluation_perturbations)
        )
        if (
            perturbations != expected_primary
            or secondary != expected_secondary
            or development_only != expected_development_only
            or not perturbations
        ):
            raise SpatialPerturbationSplitError(
                "primary, secondary, and development-only perturbation roles are inconsistent"
            )
        _checked_cartesian_size(len(animals), len(perturbations))
        _checked_cartesian_size(
            len(animals), len(perturbations), len(neighbour_types),
            len(cast(tuple[str, str], _science()["primary_bands"])),
        )
        parents = tuple(
            _parent_from(item, f"perturbation_parents[{index}]")
            for index, item in enumerate(_items(self.perturbation_parents, "perturbation_parents"))
        )
        parents = tuple(sorted(parents, key=lambda item: item.parent_id))
        expected_parents = tuple(
            sorted(
                (_make_parent(animal, perturbation, target_map[perturbation])
                 for animal in animals for perturbation in perturbations),
                key=lambda item: item.parent_id,
            )
        )
        if parents != expected_parents:
            raise SpatialPerturbationSplitError("manifest must contain every exact perturbation parent")
        units = tuple(
            _unit_from(item, f"primary_units[{index}]")
            for index, item in enumerate(_items(self.primary_units, "primary_units"))
        )
        units = tuple(sorted(units, key=lambda item: item.unit_id))
        bands = cast(tuple[str, str], _science()["primary_bands"])
        expected_units = tuple(
            sorted(
                (_make_unit(animal, perturbation, target_map[perturbation], cell_type, band)
                 for animal in animals for perturbation in perturbations
                 for cell_type in neighbour_types for band in bands),
                key=lambda item: item.unit_id,
            )
        )
        if units != expected_units:
            raise SpatialPerturbationSplitError("manifest must contain every exact frozen primary unit")
        adjacency = metadata.block_adjacency
        candidate_identity = _sha(self.candidate_identity_sha256, "candidate_identity_sha256")
        metadata_identity = _sha(self.metadata_identity_sha256, "metadata_identity_sha256")
        capability_identity = _sha(self.capability_identity_sha256, "capability_identity_sha256")
        if (
            candidate_identity != metadata.candidate.candidate_identity_sha256
            or metadata_identity != metadata.registry_summary.metadata_identity_sha256
            or capability_identity != metadata.capability_result.capability_identity_sha256
        ):
            raise SpatialPerturbationSplitError("upstream registry identities do not match")
        identity = _sha(self.split_identity_sha256, "split_identity_sha256")
        for name, value in (
            ("split_id", split_id), ("split_role", split_role),
            ("split_seed", split_seed),
            ("development_animals", development), ("evaluation_animals", evaluation),
            ("train_rows", train), ("tune_rows", tune), ("evaluation_rows", held_out),
            ("gene_names", genes), ("perturbations", perturbations),
            ("registered_perturbations", registered),
            ("secondary_perturbations", secondary),
            ("development_only_perturbations", development_only),
            ("row_provenance", rows), ("perturbation_parents", parents),
            ("primary_units", units), ("neighbour_cell_types", neighbour_types),
            ("perturbation_targets", targets), ("block_adjacency", adjacency),
            ("safe_control_label", safe_label), ("split_identity_sha256", identity),
            ("neighbour_relations", metadata.neighbour_relations),
            ("neighbour_table_identity_sha256", metadata.neighbour_table_identity_sha256),
            ("candidate", metadata.candidate), ("registry_summary", metadata.registry_summary),
            ("capability_result", metadata.capability_result),
            ("candidate_identity_sha256", candidate_identity),
            ("metadata_identity_sha256", metadata_identity),
            ("capability_identity_sha256", capability_identity),
        ):
            object.__setattr__(self, name, value)
        if identity != _identity(_manifest_unsigned(self)):
            raise SpatialPerturbationSplitError("split identity does not match manifest")

    @property
    def development_rows(self) -> tuple[int, ...]:
        snapshot = _snapshot_manifest(self)
        return tuple(sorted(snapshot.train_rows + snapshot.tune_rows))

    @property
    def primary_unit_ids(self) -> tuple[str, ...]:
        snapshot = _snapshot_manifest(self)
        return tuple(item.unit_id for item in snapshot.primary_units)


def _snapshot_manifest(manifest: BridgeSplitManifest) -> BridgeSplitManifest:
    if type(manifest) is not BridgeSplitManifest:
        raise SpatialPerturbationSplitError("manifest must be BridgeSplitManifest")
    return BridgeSplitManifest(
        manifest.split_id, manifest.split_role, manifest.split_seed,
        manifest.development_animals,
        manifest.evaluation_animals, manifest.train_rows, manifest.tune_rows,
        manifest.evaluation_rows, manifest.gene_names, manifest.perturbations,
        manifest.registered_perturbations, manifest.secondary_perturbations,
        manifest.development_only_perturbations,
        manifest.split_identity_sha256, manifest.row_provenance,
        manifest.perturbation_parents, manifest.primary_units,
        manifest.neighbour_cell_types, manifest.perturbation_targets,
        manifest.block_adjacency, manifest.safe_control_label,
        manifest.neighbour_relations, manifest.neighbour_table_identity_sha256,
        manifest.candidate, manifest.registry_summary, manifest.capability_result,
        manifest.candidate_identity_sha256, manifest.metadata_identity_sha256,
        manifest.capability_identity_sha256,
    )


def split_manifest_to_mapping(manifest: BridgeSplitManifest) -> dict[str, object]:
    snapshot = _snapshot_manifest(manifest)
    mapping = _manifest_unsigned(snapshot)
    mapping["neighbour_relations"] = [
        _relation_mapping(item) for item in snapshot.neighbour_relations
    ]
    mapping["split_identity_sha256"] = snapshot.split_identity_sha256
    return mapping


def build_bridge_partition_manifest(
    metadata: BridgeSplitMetadata,
    split_id: str,
    split_role: str,
    train_animals: tuple[str, ...],
    tune_animals: tuple[str, ...],
    evaluation_animals: tuple[str, ...],
) -> BridgeSplitManifest:
    """Freeze a whole-animal bridge partition from registered metadata."""
    if type(metadata) is not BridgeSplitMetadata:
        raise SpatialPerturbationSplitError("metadata must be BridgeSplitMetadata")
    snapshot = BridgeSplitMetadata(
        metadata.rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label,
        metadata.neighbour_relations, metadata.neighbour_table_identity_sha256,
        metadata.candidate, metadata.registry_summary, metadata.capability_result,
    )
    animals = tuple(sorted({row.animal_id for row in snapshot.rows}))
    train_animals = _text_items(train_animals, "train_animals", sort=True)
    tune_animals = _text_items(tune_animals, "tune_animals", sort=True)
    evaluation_animals = _text_items(
        evaluation_animals, "evaluation_animals", sort=True
    )
    development = tuple(sorted(train_animals + tune_animals))
    development_perturbations = {
        row.context_perturbation_id for row in snapshot.rows
        if row.animal_id in development and row.cell_role == "perturbation_source"
    }
    evaluation_perturbations = {
        row.context_perturbation_id for row in snapshot.rows
        if row.animal_id in evaluation_animals
        and row.cell_role == "perturbation_source"
    }
    primary = tuple(sorted(development_perturbations & evaluation_perturbations))
    secondary = tuple(sorted(evaluation_perturbations - development_perturbations))
    development_only = tuple(
        sorted(development_perturbations - evaluation_perturbations)
    )
    if not primary:
        raise SpatialPerturbationSplitError("partition has no shared primary perturbation")
    target_map = dict(snapshot.perturbation_targets)
    _checked_cartesian_size(len(animals), len(primary))
    _checked_cartesian_size(
        len(animals), len(primary), len(snapshot.neighbour_cell_types),
        len(cast(tuple[str, str], _science()["primary_bands"])),
    )
    parents = tuple(
        sorted(
            (_make_parent(animal, perturbation, target_map[perturbation])
             for animal in animals for perturbation in primary),
            key=lambda item: item.parent_id,
        )
    )
    units = tuple(
        sorted(
            (_make_unit(animal, perturbation, target_map[perturbation], cell_type, band)
             for animal in animals for perturbation in primary
             for cell_type in snapshot.neighbour_cell_types
             for band in cast(tuple[str, str], _science()["primary_bands"])),
            key=lambda item: item.unit_id,
        )
    )
    unsigned: dict[str, object] = {
        "split_id": _safe_text(split_id, "split_id"),
        "split_role": _safe_text(split_role, "split_role"),
        "split_seed": _science()["split_seed"],
        "development_animals": development,
        "evaluation_animals": evaluation_animals,
        "train_rows": tuple(
            row.stable_row_id for row in snapshot.rows
            if row.animal_id in train_animals
        ),
        "tune_rows": tuple(
            row.stable_row_id for row in snapshot.rows
            if row.animal_id in tune_animals
        ),
        "evaluation_rows": tuple(
            row.stable_row_id for row in snapshot.rows
            if row.animal_id in evaluation_animals
        ),
        "gene_names": list(snapshot.gene_names), "perturbations": list(primary),
        "registered_perturbations": list(snapshot.perturbations),
        "secondary_perturbations": list(secondary),
        "development_only_perturbations": list(development_only),
        "row_provenance": [_row_mapping(row) for row in snapshot.rows],
        "perturbation_parents": [_parent_mapping(item) for item in parents],
        "primary_units": [_unit_mapping(item) for item in units],
        "neighbour_cell_types": list(snapshot.neighbour_cell_types),
        "perturbation_targets": [list(item) for item in snapshot.perturbation_targets],
        "block_adjacency": [_adjacency_mapping(item) for item in snapshot.block_adjacency],
        "safe_control_label": snapshot.safe_control_label,
        "neighbour_relations": snapshot.neighbour_relations,
        "neighbour_table_identity_sha256": snapshot.neighbour_table_identity_sha256,
        "candidate": snapshot.candidate.to_mapping(),
        "registry_summary": snapshot.registry_summary.to_mapping(),
        "capability_result": snapshot.capability_result.to_mapping(),
        "candidate_identity_sha256": snapshot.candidate.candidate_identity_sha256,
        "metadata_identity_sha256": snapshot.registry_summary.metadata_identity_sha256,
        "capability_identity_sha256": snapshot.capability_result.capability_identity_sha256,
    }
    constructor_values: dict[str, object] = dict(unsigned)
    constructor_values.update({
        "row_provenance": snapshot.rows,
        "perturbation_parents": parents,
        "primary_units": units,
        "block_adjacency": snapshot.block_adjacency,
        "neighbour_relations": snapshot.neighbour_relations,
        "candidate": snapshot.candidate,
        "registry_summary": snapshot.registry_summary,
        "capability_result": snapshot.capability_result,
    })
    unsigned_mapping = _manifest_unsigned(cast(
        BridgeSplitManifest, SimpleNamespace(**constructor_values)
    ))
    constructor = cast(Any, BridgeSplitManifest)
    return constructor(
        **constructor_values,
        split_identity_sha256=_identity(unsigned_mapping),
    )


def build_pilot_fold(
    metadata: BridgeSplitMetadata, evaluation_animal: str
) -> BridgeSplitManifest:
    if type(metadata) is not BridgeSplitMetadata:
        raise SpatialPerturbationSplitError("metadata must be BridgeSplitMetadata")
    animals = tuple(sorted({row.animal_id for row in metadata.rows}))
    if len(animals) != 3:
        raise SpatialPerturbationSplitError("pilot requires exactly three animals")
    evaluation = _safe_text(evaluation_animal, "evaluation_animal")
    if evaluation not in animals:
        raise SpatialPerturbationSplitError(
            "evaluation_animal must be an exact registered animal"
        )
    return build_bridge_partition_manifest(
        metadata,
        f"pilot_leave_one_animal_out:{evaluation}",
        "pilot",
        tuple(animal for animal in animals if animal != evaluation),
        (),
        (evaluation,),
    )


@dataclass(frozen=True, slots=True)
class BridgeParentEvidence:
    animal_id: str
    perturbation_id: str
    target_gene: str
    perturbation_source_cell_ids: tuple[str, ...]
    safe_source_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("animal_id", "perturbation_id", "target_gene"):
            object.__setattr__(self, name, _safe_text(getattr(self, name), name))
        object.__setattr__(
            self, "perturbation_source_cell_ids",
            _text_items(self.perturbation_source_cell_ids, "perturbation_source_cell_ids", sort=True),
        )
        object.__setattr__(
            self, "safe_source_cell_ids",
            _text_items(self.safe_source_cell_ids, "safe_source_cell_ids", sort=True),
        )


def _parent_evidence_from(value: object, name: str) -> BridgeParentEvidence:
    if type(value) is BridgeParentEvidence:
        item = cast(BridgeParentEvidence, value)
        return BridgeParentEvidence(
            item.animal_id, item.perturbation_id, item.target_gene,
            item.perturbation_source_cell_ids, item.safe_source_cell_ids,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "animal_id", "perturbation_id", "target_gene",
            "perturbation_source_cell_ids", "safe_source_cell_ids",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeParentEvidence(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BridgeParentEvidence")


@dataclass(frozen=True, slots=True)
class BridgePrimaryUnitEvidence:
    unit_id: str
    perturbation_neighbour_relation_ids: tuple[str, ...]
    safe_neighbour_relation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _sha(self.unit_id, "unit_id"))
        object.__setattr__(
            self, "perturbation_neighbour_relation_ids",
            _text_items(
                self.perturbation_neighbour_relation_ids,
                "perturbation_neighbour_relation_ids", sort=True,
            ),
        )
        object.__setattr__(
            self, "safe_neighbour_relation_ids",
            _text_items(
                self.safe_neighbour_relation_ids,
                "safe_neighbour_relation_ids", sort=True,
            ),
        )
        if set(self.perturbation_neighbour_relation_ids) & set(self.safe_neighbour_relation_ids):
            raise SpatialPerturbationSplitError(
                "treatment and safe neighbour relation IDs must be disjoint"
            )


def _unit_evidence_from(value: object, name: str) -> BridgePrimaryUnitEvidence:
    if type(value) is BridgePrimaryUnitEvidence:
        item = cast(BridgePrimaryUnitEvidence, value)
        return BridgePrimaryUnitEvidence(
            item.unit_id, item.perturbation_neighbour_relation_ids,
            item.safe_neighbour_relation_ids,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "unit_id", "perturbation_neighbour_relation_ids",
            "safe_neighbour_relation_ids",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgePrimaryUnitEvidence(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BridgePrimaryUnitEvidence")


def _parent_evidence_mapping(item: BridgeParentEvidence) -> dict[str, object]:
    snapshot = _parent_evidence_from(item, "parent_evidence")
    return {
        "animal_id": snapshot.animal_id, "perturbation_id": snapshot.perturbation_id,
        "target_gene": snapshot.target_gene,
        "perturbation_source_cell_ids": list(snapshot.perturbation_source_cell_ids),
        "safe_source_cell_ids": list(snapshot.safe_source_cell_ids),
    }


def _unit_evidence_mapping(item: BridgePrimaryUnitEvidence) -> dict[str, object]:
    snapshot = _unit_evidence_from(item, "unit_evidence")
    return {
        "unit_id": snapshot.unit_id,
        "perturbation_neighbour_relation_ids": list(
            snapshot.perturbation_neighbour_relation_ids
        ),
        "safe_neighbour_relation_ids": list(snapshot.safe_neighbour_relation_ids),
    }


def _evidence_unsigned(evidence: "BridgeEligibilityEvidence") -> dict[str, object]:
    return {
        "split_identity_sha256": evidence.split_identity_sha256,
        "neighbour_table_identity_sha256": evidence.neighbour_table_identity_sha256,
        "parent_evidence": [_parent_evidence_mapping(item) for item in evidence.parent_evidence],
        "unit_evidence": [_unit_evidence_mapping(item) for item in evidence.unit_evidence],
    }


@dataclass(frozen=True, slots=True)
class BridgeEligibilityEvidence:
    split_identity_sha256: str
    neighbour_table_identity_sha256: str
    parent_evidence: tuple[BridgeParentEvidence, ...]
    unit_evidence: tuple[BridgePrimaryUnitEvidence, ...]
    evidence_identity_sha256: str

    def __post_init__(self) -> None:
        split_identity = _sha(self.split_identity_sha256, "split_identity_sha256")
        neighbour_identity = _sha(
            self.neighbour_table_identity_sha256,
            "neighbour_table_identity_sha256",
        )
        parents = tuple(
            _parent_evidence_from(item, f"parent_evidence[{index}]")
            for index, item in enumerate(_items(self.parent_evidence, "parent_evidence"))
        )
        parent_keys = tuple((item.animal_id, item.perturbation_id) for item in parents)
        if len(set(parent_keys)) != len(parent_keys):
            raise SpatialPerturbationSplitError("parent evidence contexts must be unique")
        parents = tuple(sorted(parents, key=lambda item: (item.animal_id, item.perturbation_id)))
        units = tuple(
            _unit_evidence_from(item, f"unit_evidence[{index}]")
            for index, item in enumerate(_items(self.unit_evidence, "unit_evidence"))
        )
        if len({item.unit_id for item in units}) != len(units):
            raise SpatialPerturbationSplitError("unique unit evidence is required")
        units = tuple(sorted(units, key=lambda item: item.unit_id))
        treatment_ids = tuple(
            relation_id
            for item in units
            for relation_id in item.perturbation_neighbour_relation_ids
        )
        safe_ids = tuple(
            relation_id for item in units
            for relation_id in item.safe_neighbour_relation_ids
        )
        if len(set(treatment_ids)) != len(treatment_ids) or len(set(safe_ids)) != len(safe_ids):
            raise SpatialPerturbationSplitError(
                "a neighbour relation cannot appear in two frozen units"
            )
        if set(treatment_ids) & set(safe_ids):
            raise SpatialPerturbationSplitError("treatment and safe neighbour evidence must be disjoint")
        identity = _sha(self.evidence_identity_sha256, "evidence_identity_sha256")
        object.__setattr__(self, "split_identity_sha256", split_identity)
        object.__setattr__(self, "neighbour_table_identity_sha256", neighbour_identity)
        object.__setattr__(self, "parent_evidence", parents)
        object.__setattr__(self, "unit_evidence", units)
        object.__setattr__(self, "evidence_identity_sha256", identity)
        if identity != _identity(_evidence_unsigned(self)):
            raise SpatialPerturbationSplitError("eligibility evidence identity does not match evidence")


def _snapshot_evidence(evidence: BridgeEligibilityEvidence) -> BridgeEligibilityEvidence:
    if type(evidence) is not BridgeEligibilityEvidence:
        raise SpatialPerturbationSplitError("evidence must be BridgeEligibilityEvidence")
    return BridgeEligibilityEvidence(
        evidence.split_identity_sha256, evidence.neighbour_table_identity_sha256,
        evidence.parent_evidence,
        evidence.unit_evidence, evidence.evidence_identity_sha256,
    )


def eligibility_evidence_to_mapping(evidence: BridgeEligibilityEvidence) -> dict[str, object]:
    snapshot = _snapshot_evidence(evidence)
    mapping = _evidence_unsigned(snapshot)
    mapping["evidence_identity_sha256"] = snapshot.evidence_identity_sha256
    return mapping


def build_bridge_eligibility_evidence(
    manifest: BridgeSplitManifest,
    parent_evidence: tuple[BridgeParentEvidence, ...],
    unit_evidence: tuple[BridgePrimaryUnitEvidence, ...],
) -> BridgeEligibilityEvidence:
    snapshot = _snapshot_manifest(manifest)
    parents = tuple(
        _parent_evidence_from(item, f"parent_evidence[{index}]")
        for index, item in enumerate(_items(parent_evidence, "parent_evidence"))
    )
    units = tuple(
        _unit_evidence_from(item, f"unit_evidence[{index}]")
        for index, item in enumerate(_items(unit_evidence, "unit_evidence"))
    )
    unsigned = {
        "split_identity_sha256": snapshot.split_identity_sha256,
        "neighbour_table_identity_sha256": snapshot.neighbour_table_identity_sha256,
        "parent_evidence": [_parent_evidence_mapping(item) for item in sorted(parents, key=lambda item: (item.animal_id, item.perturbation_id))],
        "unit_evidence": [_unit_evidence_mapping(item) for item in sorted(units, key=lambda item: item.unit_id)],
    }
    return BridgeEligibilityEvidence(
        **unsigned, evidence_identity_sha256=_identity(unsigned)  # type: ignore[arg-type]
    )


def _evidence_from(value: object, name: str) -> BridgeEligibilityEvidence:
    if type(value) is BridgeEligibilityEvidence:
        return _snapshot_evidence(cast(BridgeEligibilityEvidence, value))
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "split_identity_sha256", "neighbour_table_identity_sha256",
            "parent_evidence", "unit_evidence",
            "evidence_identity_sha256",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeEligibilityEvidence(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BridgeEligibilityEvidence")


def _block_key(row: BridgeSplitRow) -> tuple[str, str, str]:
    return (row.animal_id, row.section_id, row.spatial_block)


def _adjacent_block_pairs(
    manifest: BridgeSplitManifest,
) -> set[frozenset[tuple[str, str, str]]]:
    return {
        frozenset(
            (
                (item.animal_id, item.section_id, item.first_block),
                (item.animal_id, item.section_id, item.second_block),
            )
        )
        for item in manifest.block_adjacency
        if item.adjacent
    }


def _has_non_adjacent_block_subset(
    rows: tuple[BridgeSplitRow, ...],
    manifest: BridgeSplitManifest,
    minimum: int,
    *,
    adjacent_pairs: set[frozenset[tuple[str, str, str]]] | None = None,
) -> bool:
    blocks = tuple(sorted({_block_key(row) for row in rows}))
    if minimum != 3:
        raise SpatialPerturbationSplitError(
            "the independent block threshold is frozen to three"
        )
    if len(blocks) > _MAX_BLOCK_GRAPH_NODES:
        raise SpatialPerturbationSplitError("selected block graph exceeds the safe bound")
    if len(blocks) < minimum:
        return False
    block_index = {block: index for index, block in enumerate(blocks)}
    adjacency_masks = [0] * len(blocks)
    adjacent = (
        _adjacent_block_pairs(manifest)
        if adjacent_pairs is None
        else adjacent_pairs
    )
    for edge in adjacent:
        endpoints = tuple(edge)
        if len(endpoints) != 2:
            raise SpatialPerturbationSplitError("frozen block adjacency is malformed")
        first = block_index.get(endpoints[0])
        second = block_index.get(endpoints[1])
        if first is None or second is None:
            continue
        adjacency_masks[first] |= 1 << second
        adjacency_masks[second] |= 1 << first

    all_blocks_mask = (1 << len(blocks)) - 1
    for first in range(len(blocks) - 2):
        later_than_first = all_blocks_mask & ~((1 << (first + 1)) - 1)
        first_non_neighbours = later_than_first & ~adjacency_masks[first]
        candidates = first_non_neighbours
        while candidates:
            second_bit = candidates & -candidates
            second = second_bit.bit_length() - 1
            later_than_second = first_non_neighbours & ~((1 << (second + 1)) - 1)
            if later_than_second & ~adjacency_masks[second]:
                return True
            candidates ^= second_bit
    return False


def _distinct_block_count(rows: tuple[BridgeSplitRow, ...]) -> int:
    return len({_block_key(row) for row in rows})


def _source_strata(
    rows: tuple[BridgeSplitRow, ...],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        (
            row.section_id,
            row.spatial_block,
            row.source_cell_type,
        )
        for row in rows
    )


def _neighbour_strata(
    relations: tuple[BridgeNeighbourRelation, ...],
) -> Counter[tuple[str, str, str, str, str]]:
    return Counter(
        (
            item.section_id,
            item.spatial_block,
            item.source_cell_type,
            item.neighbor_cell_type,
            item.band,
        )
        for item in relations
    )


def _resolve_relations(
    relation_ids: tuple[str, ...],
    relation_by_id: dict[str, BridgeNeighbourRelation],
) -> tuple[BridgeNeighbourRelation, ...]:
    missing = tuple(item for item in relation_ids if item not in relation_by_id)
    if missing:
        raise SpatialPerturbationSplitError(
            "evidence relation ID is absent from frozen neighbour table"
        )
    result = tuple(relation_by_id[item] for item in relation_ids)
    if len({item.neighbor_cell_id for item in result}) != len(result):
        raise SpatialPerturbationSplitError(
            "evidence cannot count one neighbour cell more than once per unit"
        )
    return result


def _require_relations(
    relations: tuple[BridgeNeighbourRelation, ...],
    *,
    animal: str,
    perturbation: str,
    cell_type: str,
    band: str,
    safe: bool,
) -> None:
    if any(item.contamination for item in relations):
        raise SpatialPerturbationSplitError(
            "contaminated neighbour relations cannot enter eligibility evidence"
        )
    if any(
        item.animal_id != animal
        or item.matched_perturbation_id != perturbation
        or item.neighbor_cell_type != cell_type
        or item.band != band
        or item.is_safe_control is not safe
        for item in relations
    ):
        raise SpatialPerturbationSplitError(
            "evidence relation does not match expected animal/context/role/type/band/control"
        )


def _resolve_rows(
    cell_ids: tuple[str, ...], row_by_cell: dict[str, BridgeSplitRow]
) -> tuple[BridgeSplitRow, ...]:
    missing = tuple(cell for cell in cell_ids if cell not in row_by_cell)
    if missing:
        raise SpatialPerturbationSplitError("evidence cell ID is absent from manifest provenance")
    return tuple(row_by_cell[cell] for cell in cell_ids)


def _require_rows(
    rows: tuple[BridgeSplitRow, ...],
    *,
    animal: str,
    perturbation: str,
    role: str,
    label: str,
    cell_type: str | None = None,
    band: str,
) -> None:
    if any(
        row.animal_id != animal
        or row.context_perturbation_id != perturbation
        or row.cell_role != role
        or row.observed_label != label
        or row.distance_band != band
        or (cell_type is not None and row.cell_type != cell_type)
        for row in rows
    ):
        raise SpatialPerturbationSplitError(
            "evidence cell does not match expected animal/context/role/type/band"
        )


@dataclass(frozen=True, slots=True)
class _DerivedEligibility:
    reasons: tuple[str, ...]
    abstained_unit_ids: tuple[str, ...]
    scoreable_parent_ids: tuple[str, ...]
    per_animal_perturbation_coverage: tuple[tuple[str, int, int], ...]
    primary_scoreable: int
    primary_total: int
    abstained: int
    attempted: int


def _derive_eligibility(
    manifest: BridgeSplitManifest, evidence: BridgeEligibilityEvidence
) -> _DerivedEligibility:
    if evidence.split_identity_sha256 != manifest.split_identity_sha256:
        raise SpatialPerturbationSplitError("evidence split identity does not match manifest")
    if evidence.neighbour_table_identity_sha256 != manifest.neighbour_table_identity_sha256:
        raise SpatialPerturbationSplitError(
            "evidence neighbour-table identity does not match manifest"
        )
    frozen_parent_by_context = {
        (item.animal_id, item.perturbation_id): item for item in manifest.perturbation_parents
    }
    evidence_parent_by_context = {
        (item.animal_id, item.perturbation_id): item for item in evidence.parent_evidence
    }
    for context, item in evidence_parent_by_context.items():
        frozen = frozen_parent_by_context.get(context)
        if frozen is None or item.target_gene != frozen.target_gene:
            raise SpatialPerturbationSplitError("evidence does not match frozen parent context/target gene")
    if set(evidence_parent_by_context) != set(frozen_parent_by_context):
        raise SpatialPerturbationSplitError("evidence must contain exact perturbation parents")
    frozen_unit_by_id = {item.unit_id: item for item in manifest.primary_units}
    evidence_unit_by_id = {item.unit_id: item for item in evidence.unit_evidence}
    if set(evidence_unit_by_id) != set(frozen_unit_by_id):
        raise SpatialPerturbationSplitError("evidence must contain exact frozen primary units")
    row_by_cell = {row.cell_id: row for row in manifest.row_provenance}
    relation_by_id = {item.relation_id: item for item in manifest.neighbour_relations}
    minimum_source = cast(int, _science()["minimum_source_cells"])
    minimum_safe_source = cast(int, _science()["minimum_safe_source_cells"])
    minimum_band = cast(int, _science()["minimum_band_neighbours"])
    minimum_cell_type = cast(int, _science()["minimum_cell_type_neighbours"])
    minimum_blocks = cast(int, _science()["minimum_spatial_blocks"])
    adjacent_block_pairs = _adjacent_block_pairs(manifest)
    frozen_source_strata = {
        context: {
            (row.section_id, row.source_cell_type)
            for row in manifest.row_provenance
            if row.animal_id == context[0]
            and row.context_perturbation_id == context[1]
            and row.cell_role == "perturbation_source"
        }
        for context in frozen_parent_by_context
    }
    parent_failures: dict[tuple[str, str], set[str]] = {}
    for context, frozen in frozen_parent_by_context.items():
        item = evidence_parent_by_context[context]
        treatment = _resolve_rows(item.perturbation_source_cell_ids, row_by_cell)
        safe = _resolve_rows(item.safe_source_cell_ids, row_by_cell)
        _require_rows(
            treatment, animal=frozen.animal_id, perturbation=frozen.perturbation_id,
            role="perturbation_source", label=frozen.perturbation_id, band="own",
        )
        _require_rows(
            safe, animal=frozen.animal_id,
            perturbation=manifest.safe_control_label,
            role="safe_source", label=manifest.safe_control_label, band="own",
        )
        treatment_meets_threshold = len(treatment) >= minimum_source
        safe_meets_threshold = len(safe) >= minimum_safe_source
        if (
            treatment_meets_threshold
            and safe_meets_threshold
            and _source_strata(treatment) != _source_strata(safe)
        ):
            raise SpatialPerturbationSplitError(
                "safe source controls do not match the frozen stratum multiset"
            )
        failures: set[str] = set()
        if not treatment_meets_threshold:
            failures.add("insufficient_perturbation_coverage")
        if not safe_meets_threshold:
            failures.add("insufficient_safe_control_coverage")
        if not _has_non_adjacent_block_subset(
            treatment,
            manifest,
            minimum_blocks,
            adjacent_pairs=adjacent_block_pairs,
        ):
            failures.add("insufficient_spatial_blocks")
        if _distinct_block_count(safe) < minimum_blocks:
            failures.add("insufficient_safe_control_spatial_blocks")
        if frozen.target_gene not in manifest.gene_names:
            failures.add("target_gene_not_measurable")
        parent_failures[context] = failures
    unit_scoreable: dict[str, bool] = {}
    unit_failures: dict[str, set[str]] = {}
    unit_rows: dict[str, tuple[tuple[BridgeSplitRow, ...], tuple[BridgeSplitRow, ...]]] = {}
    unit_relations: dict[
        str,
        tuple[tuple[BridgeNeighbourRelation, ...], tuple[BridgeNeighbourRelation, ...]],
    ] = {}
    for unit_id, frozen_unit in frozen_unit_by_id.items():
        unit_evidence = evidence_unit_by_id[unit_id]
        treatment_relations = _resolve_relations(
            unit_evidence.perturbation_neighbour_relation_ids, relation_by_id
        )
        safe_relations = _resolve_relations(
            unit_evidence.safe_neighbour_relation_ids, relation_by_id
        )
        _require_relations(
            treatment_relations, animal=frozen_unit.animal_id,
            perturbation=frozen_unit.perturbation_id,
            cell_type=frozen_unit.neighbour_cell_type, band=frozen_unit.band,
            safe=False,
        )
        _require_relations(
            safe_relations, animal=frozen_unit.animal_id,
            perturbation=frozen_unit.perturbation_id,
            cell_type=frozen_unit.neighbour_cell_type, band=frozen_unit.band,
            safe=True,
        )
        treatment = tuple(row_by_cell[item.neighbor_cell_id] for item in treatment_relations)
        safe = tuple(row_by_cell[item.neighbor_cell_id] for item in safe_relations)
        treatment_meets_threshold = len(treatment) >= minimum_cell_type
        safe_meets_threshold = len(safe) >= minimum_cell_type
        if (
            treatment_meets_threshold
            and safe_meets_threshold
            and _neighbour_strata(treatment_relations) != _neighbour_strata(safe_relations)
        ):
            raise SpatialPerturbationSplitError(
                "matched safe neighbours do not match the frozen stratum multiset"
            )
        source_strata = frozen_source_strata[
            (frozen_unit.animal_id, frozen_unit.perturbation_id)
        ]
        if any(
            (item.section_id, item.source_cell_type) not in source_strata
            for item in treatment_relations + safe_relations
        ):
            raise SpatialPerturbationSplitError(
                "neighbour source_cell_type is not frozen by its perturbation parent"
            )
        unit_local_failures: set[str] = set()
        if not treatment_meets_threshold:
            unit_local_failures.add("insufficient_band_neighbours")
        if not safe_meets_threshold:
            unit_local_failures.add("insufficient_safe_control_band_neighbours")
        if _distinct_block_count(treatment) < minimum_blocks:
            unit_local_failures.add("insufficient_spatial_blocks")
        if _distinct_block_count(safe) < minimum_blocks:
            unit_local_failures.add("insufficient_safe_control_spatial_blocks")
        unit_failures[unit_id] = unit_local_failures
        unit_scoreable[unit_id] = not unit_local_failures
        unit_rows[unit_id] = (treatment, safe)
        unit_relations[unit_id] = (treatment_relations, safe_relations)
    bands = cast(tuple[str, str], _science()["primary_bands"])
    for context in frozen_parent_by_context:
        animal, perturbation = context
        for band in bands:
            unit_ids = tuple(
                item.unit_id for item in manifest.primary_units
                if item.animal_id == animal
                and item.perturbation_id == perturbation
                and item.band == band
            )
            treatment = tuple(row for unit_id in unit_ids for row in unit_rows[unit_id][0])
            safe = tuple(row for unit_id in unit_ids for row in unit_rows[unit_id][1])
            treatment_total = len({row.cell_id for row in treatment})
            safe_total = len({row.cell_id for row in safe})
            if treatment_total < minimum_band:
                parent_failures[context].add("insufficient_band_neighbours")
            if safe_total < minimum_band:
                parent_failures[context].add("insufficient_safe_control_band_neighbours")
            treatment_strata = _neighbour_strata(tuple(
                item for unit_id in unit_ids for item in unit_relations[unit_id][0]
            ))
            safe_strata = _neighbour_strata(tuple(
                item for unit_id in unit_ids for item in unit_relations[unit_id][1]
            ))
            matched_neighbours = sum(
                min(treatment_strata[key], safe_strata[key])
                for key in treatment_strata.keys() | safe_strata.keys()
            )
            if (
                treatment_total >= minimum_band
                and safe_total >= minimum_band
                and matched_neighbours < minimum_band
            ):
                parent_failures[context].add(
                    "insufficient_safe_control_band_neighbours"
                )
            if not any(unit_scoreable[unit_id] for unit_id in unit_ids):
                for unit_id in unit_ids:
                    parent_failures[context].update(unit_failures[unit_id])
    parent_scoreable = {
        context: not failures for context, failures in parent_failures.items()
    }
    for unit_id, frozen_unit in frozen_unit_by_id.items():
        unit_scoreable[unit_id] = unit_scoreable[unit_id] and parent_scoreable[
            (frozen_unit.animal_id, frozen_unit.perturbation_id)
        ]
    coverage = tuple(
        (
            animal,
            sum(parent_scoreable[(animal, perturbation)] for perturbation in manifest.perturbations),
            len(manifest.perturbations),
        )
        for animal in tuple(sorted(manifest.development_animals + manifest.evaluation_animals))
    )
    failing_animals = {
        animal for animal, scoreable, total in coverage
        if Fraction(scoreable, total) < cast(Fraction, _science()["minimum_coverage"])
    }
    reasons: set[str] = set()
    for context, failures in parent_failures.items():
        if context[0] in failing_animals:
            reasons.update(failures)
    abstained_ids = tuple(sorted(unit_id for unit_id, scoreable in unit_scoreable.items() if not scoreable))
    attempted = len(unit_scoreable)
    abstained = len(abstained_ids)
    primary_scoreable = attempted - abstained
    if Fraction(primary_scoreable, attempted) < cast(Fraction, _science()["minimum_coverage"]):
        reasons.add("insufficient_primary_unit_coverage")
    if Fraction(abstained, attempted) > cast(Fraction, _science()["maximum_abstention"]):
        reasons.add("excessive_abstention")
    ordered_reasons = tuple(reason for reason in _REASON_ORDER if reason in reasons)
    scoreable_parent_ids = tuple(
        sorted(
            frozen_parent_by_context[context].parent_id
            for context, scoreable in parent_scoreable.items() if scoreable
        )
    )
    return _DerivedEligibility(
        ordered_reasons, abstained_ids, scoreable_parent_ids, coverage,
        primary_scoreable, attempted, abstained, attempted,
    )


def _result_unsigned(result: "BridgeEligibilityResult") -> dict[str, object]:
    return {
        "eligible": result.eligible, "reason": result.reason,
        "reasons": list(result.reasons),
        "abstained_unit_ids": list(result.abstained_unit_ids),
        "scoreable_parent_ids": list(result.scoreable_parent_ids),
        "per_animal_perturbation_coverage": [list(item) for item in result.per_animal_perturbation_coverage],
        "primary_scoreable": result.primary_scoreable, "primary_total": result.primary_total,
        "abstained": result.abstained, "attempted": result.attempted,
        "split_identity_sha256": result.manifest.split_identity_sha256,
        "evidence_identity_sha256": result.evidence.evidence_identity_sha256,
    }


@dataclass(frozen=True, slots=True)
class BridgeEligibilityResult:
    eligible: bool
    reason: str | None
    reasons: tuple[str, ...]
    abstained_unit_ids: tuple[str, ...]
    scoreable_parent_ids: tuple[str, ...]
    per_animal_perturbation_coverage: tuple[tuple[str, int, int], ...]
    primary_scoreable: int
    primary_total: int
    abstained: int
    attempted: int
    manifest: BridgeSplitManifest
    evidence: BridgeEligibilityEvidence
    eligibility_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise SpatialPerturbationSplitError("eligible must be a built-in boolean")
        manifest = _snapshot_manifest(self.manifest)
        evidence = _snapshot_evidence(self.evidence)
        derived = _derive_eligibility(manifest, evidence)
        reasons = _text_items(self.reasons, "reasons")
        reason = None if self.reason is None else _safe_text(self.reason, "reason")
        abstained_ids = _text_items(self.abstained_unit_ids, "abstained_unit_ids", sort=True)
        scoreable_ids = _text_items(self.scoreable_parent_ids, "scoreable_parent_ids", sort=True)
        coverage: list[tuple[str, int, int]] = []
        for index, raw_item in enumerate(_items(self.per_animal_perturbation_coverage, "per_animal_perturbation_coverage")):
            item = _items(raw_item, f"per_animal_perturbation_coverage[{index}]")
            if len(item) != 3:
                raise SpatialPerturbationSplitError("coverage records must have three items")
            coverage.append(
                (_safe_text(item[0], "coverage animal"), _integer(item[1], "coverage scoreable"), _integer(item[2], "coverage total"))
            )
        coverage_tuple = tuple(sorted(coverage))
        counts = (
            _integer(self.primary_scoreable, "primary_scoreable"),
            _integer(self.primary_total, "primary_total"),
            _integer(self.abstained, "abstained"),
            _integer(self.attempted, "attempted"),
        )
        expected = (
            not derived.reasons, derived.reasons[0] if derived.reasons else None,
            derived.reasons, derived.abstained_unit_ids, derived.scoreable_parent_ids,
            derived.per_animal_perturbation_coverage, derived.primary_scoreable,
            derived.primary_total, derived.abstained, derived.attempted,
        )
        actual = (
            self.eligible, reason, reasons, abstained_ids, scoreable_ids,
            coverage_tuple, *counts,
        )
        if actual != expected:
            raise SpatialPerturbationSplitError("eligibility result does not match frozen evidence")
        identity = _sha(self.eligibility_identity_sha256, "eligibility_identity_sha256")
        for name, value in (
            ("reason", reason), ("reasons", reasons),
            ("abstained_unit_ids", abstained_ids), ("scoreable_parent_ids", scoreable_ids),
            ("per_animal_perturbation_coverage", coverage_tuple),
            ("primary_scoreable", counts[0]), ("primary_total", counts[1]),
            ("abstained", counts[2]), ("attempted", counts[3]),
            ("manifest", manifest), ("evidence", evidence),
            ("eligibility_identity_sha256", identity),
        ):
            object.__setattr__(self, name, value)
        if identity != _identity(_result_unsigned(self)):
            raise SpatialPerturbationSplitError("eligibility identity does not match result")


def _snapshot_result(result: BridgeEligibilityResult) -> BridgeEligibilityResult:
    if type(result) is not BridgeEligibilityResult:
        raise SpatialPerturbationSplitError("result must be BridgeEligibilityResult")
    return BridgeEligibilityResult(
        result.eligible, result.reason, result.reasons, result.abstained_unit_ids,
        result.scoreable_parent_ids, result.per_animal_perturbation_coverage,
        result.primary_scoreable, result.primary_total, result.abstained,
        result.attempted, result.manifest, result.evidence,
        result.eligibility_identity_sha256,
    )


def eligibility_result_to_mapping(result: BridgeEligibilityResult) -> dict[str, object]:
    """Return a publication-boundary snapshot after full manifest/evidence replay."""
    snapshot = _snapshot_result(result)
    mapping = _result_unsigned(snapshot)
    mapping["manifest"] = split_manifest_to_mapping(snapshot.manifest)
    mapping["evidence"] = eligibility_evidence_to_mapping(snapshot.evidence)
    mapping["eligibility_identity_sha256"] = snapshot.eligibility_identity_sha256
    return mapping


def evaluate_bridge_eligibility(
    manifest: BridgeSplitManifest, evidence: BridgeEligibilityEvidence
) -> BridgeEligibilityResult:
    manifest_snapshot = _snapshot_manifest(manifest)
    evidence_snapshot = _snapshot_evidence(evidence)
    derived = _derive_eligibility(manifest_snapshot, evidence_snapshot)
    identity_mapping: dict[str, object] = {
        "eligible": not derived.reasons,
        "reason": derived.reasons[0] if derived.reasons else None,
        "reasons": list(derived.reasons),
        "abstained_unit_ids": list(derived.abstained_unit_ids),
        "scoreable_parent_ids": list(derived.scoreable_parent_ids),
        "per_animal_perturbation_coverage": [
            list(item) for item in derived.per_animal_perturbation_coverage
        ],
        "primary_scoreable": derived.primary_scoreable,
        "primary_total": derived.primary_total,
        "abstained": derived.abstained,
        "attempted": derived.attempted,
        "split_identity_sha256": manifest_snapshot.split_identity_sha256,
        "evidence_identity_sha256": evidence_snapshot.evidence_identity_sha256,
    }
    result = object.__new__(BridgeEligibilityResult)
    for name, value in (
        ("eligible", not derived.reasons),
        ("reason", derived.reasons[0] if derived.reasons else None),
        ("reasons", derived.reasons),
        ("abstained_unit_ids", derived.abstained_unit_ids),
        ("scoreable_parent_ids", derived.scoreable_parent_ids),
        ("per_animal_perturbation_coverage", derived.per_animal_perturbation_coverage),
        ("primary_scoreable", derived.primary_scoreable),
        ("primary_total", derived.primary_total),
        ("abstained", derived.abstained),
        ("attempted", derived.attempted),
        ("manifest", manifest_snapshot),
        ("evidence", evidence_snapshot),
        ("eligibility_identity_sha256", _identity(identity_mapping)),
    ):
        object.__setattr__(result, name, value)
    return result


__all__ = [
    "MAX_ABSTENTION", "MIN_BAND_NEIGHBOURS", "MIN_CELL_TYPE_NEIGHBOURS",
    "MIN_COVERAGE", "MIN_SAFE_SOURCE_CELLS", "MIN_SOURCE_CELLS",
    "MIN_SPATIAL_BLOCKS", "BridgeBlockAdjacency", "BridgeEligibilityEvidence",
    "BridgeEligibilityResult", "BridgeParentEvidence", "BridgePrimaryUnitEvidence",
    "BridgeNeighbourRelation", "BridgeSplitManifest", "BridgeSplitMetadata",
    "BridgeSplitRow",
    "FrozenPerturbationParent", "FrozenPrimaryUnit", "SpatialPerturbationSplitError",
    "build_bridge_eligibility_evidence", "build_pilot_fold",
    "eligibility_evidence_to_mapping", "eligibility_result_to_mapping",
    "evaluate_bridge_eligibility",
    "split_manifest_to_mapping",
]
