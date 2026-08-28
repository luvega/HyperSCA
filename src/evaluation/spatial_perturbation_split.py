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
from typing import cast
import unicodedata


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
_SHA = re.compile(r"[0-9a-f]{64}")
_ROLES = (
    "perturbation_source",
    "safe_source",
    "perturbation_neighbour",
    "safe_neighbour",
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
        elif self.distance_band not in bands:
            raise SpatialPerturbationSplitError("neighbour rows must use a primary band")


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

    def __post_init__(self) -> None:
        animal = _safe_text(self.animal_id, "animal_id")
        section = _safe_text(self.section_id, "section_id")
        first = _safe_text(self.first_block, "first_block")
        second = _safe_text(self.second_block, "second_block")
        if first == second:
            raise SpatialPerturbationSplitError("a block cannot be adjacent to itself")
        first, second = sorted((first, second))
        object.__setattr__(self, "animal_id", animal)
        object.__setattr__(self, "section_id", section)
        object.__setattr__(self, "first_block", first)
        object.__setattr__(self, "second_block", second)


def _adjacency_from(value: object, name: str) -> BridgeBlockAdjacency:
    if type(value) is BridgeBlockAdjacency:
        item = cast(BridgeBlockAdjacency, value)
        return BridgeBlockAdjacency(
            item.animal_id, item.section_id, item.first_block, item.second_block
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        if set(raw) != {"animal_id", "section_id", "first_block", "second_block"}:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeBlockAdjacency(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BridgeBlockAdjacency")


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
        if row.context_perturbation_id not in perturbations:
            raise SpatialPerturbationSplitError("row context perturbation is not registered")
        if row.cell_role == "perturbation_source" and row.observed_label != row.context_perturbation_id:
            raise SpatialPerturbationSplitError("perturbation source label does not match context")
        if row.cell_role == "safe_source" and row.observed_label != safe_label:
            raise SpatialPerturbationSplitError("safe source label does not match frozen control")
        if row.cell_role.endswith("neighbour"):
            if row.observed_label != "unperturbed" or row.cell_type not in neighbour_types:
                raise SpatialPerturbationSplitError("neighbour row labels are not frozen")


@dataclass(frozen=True, slots=True)
class BridgeSplitMetadata:
    rows: tuple[BridgeSplitRow, ...]
    gene_names: tuple[str, ...]
    perturbations: tuple[str, ...]
    neighbour_cell_types: tuple[str, ...]
    perturbation_targets: tuple[tuple[str, str], ...]
    block_adjacency: tuple[BridgeBlockAdjacency, ...]
    safe_control_label: str

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
        if not genes or not perturbations or not neighbour_types:
            raise SpatialPerturbationSplitError("genes, perturbations, and cell types are required")
        if tuple(pair[0] for pair in targets) != perturbations:
            raise SpatialPerturbationSplitError("each perturbation needs one frozen target gene")
        if any(target not in genes for _, target in targets):
            raise SpatialPerturbationSplitError("target gene is not in the measurable gene set")
        _validate_rows(rows, perturbations, neighbour_types, safe_label)
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
        for name, value in (
            ("rows", rows), ("gene_names", genes), ("perturbations", perturbations),
            ("neighbour_cell_types", neighbour_types), ("perturbation_targets", targets),
            ("block_adjacency", adjacency), ("safe_control_label", safe_label),
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
        "split_id": manifest.split_id, "split_seed": manifest.split_seed,
        "development_animals": list(manifest.development_animals),
        "evaluation_animals": list(manifest.evaluation_animals),
        "train_rows": list(manifest.train_rows), "tune_rows": list(manifest.tune_rows),
        "evaluation_rows": list(manifest.evaluation_rows),
        "gene_names": list(manifest.gene_names), "perturbations": list(manifest.perturbations),
        "row_provenance": [_row_mapping(row) for row in manifest.row_provenance],
        "perturbation_parents": [_parent_mapping(item) for item in manifest.perturbation_parents],
        "primary_units": [_unit_mapping(item) for item in manifest.primary_units],
        "neighbour_cell_types": list(manifest.neighbour_cell_types),
        "perturbation_targets": [list(item) for item in manifest.perturbation_targets],
        "block_adjacency": [_adjacency_mapping(item) for item in manifest.block_adjacency],
        "safe_control_label": manifest.safe_control_label,
    }


@dataclass(frozen=True, slots=True)
class BridgeSplitManifest:
    split_id: str
    split_seed: int
    development_animals: tuple[str, ...]
    evaluation_animals: tuple[str, ...]
    train_rows: tuple[int, ...]
    tune_rows: tuple[int, ...]
    evaluation_rows: tuple[int, ...]
    gene_names: tuple[str, ...]
    perturbations: tuple[str, ...]
    split_identity_sha256: str
    row_provenance: tuple[BridgeSplitRow, ...]
    perturbation_parents: tuple[FrozenPerturbationParent, ...]
    primary_units: tuple[FrozenPrimaryUnit, ...]
    neighbour_cell_types: tuple[str, ...]
    perturbation_targets: tuple[tuple[str, str], ...]
    block_adjacency: tuple[BridgeBlockAdjacency, ...]
    safe_control_label: str

    def __post_init__(self) -> None:
        split_id = _safe_text(self.split_id, "split_id")
        split_seed = _integer(self.split_seed, "split_seed")
        if split_seed != _science()["split_seed"]:
            raise SpatialPerturbationSplitError("split seed is frozen to 11")
        development = _text_items(self.development_animals, "development_animals", sort=True)
        evaluation = _text_items(self.evaluation_animals, "evaluation_animals", sort=True)
        if len(evaluation) != 1 or set(development) & set(evaluation):
            raise SpatialPerturbationSplitError("one exact evaluation animal is required")
        if split_id != f"pilot_leave_one_animal_out:{evaluation[0]}":
            raise SpatialPerturbationSplitError("split_id does not bind evaluation animal")
        train = _row_ids(self.train_rows, "train_rows")
        tune = _row_ids(self.tune_rows, "tune_rows")
        held_out = _row_ids(self.evaluation_rows, "evaluation_rows")
        if tune:
            raise SpatialPerturbationSplitError("audit-only tune_rows must remain empty")
        genes = _text_items(self.gene_names, "gene_names", sort=True)
        perturbations = _text_items(self.perturbations, "perturbations", sort=True)
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
        _validate_rows(rows, perturbations, neighbour_types, safe_label)
        animals = tuple(sorted({row.animal_id for row in rows}))
        if len(animals) != 3:
            raise SpatialPerturbationSplitError("pilot requires exactly three animals")
        if tuple(sorted(development + evaluation)) != animals:
            raise SpatialPerturbationSplitError("manifest animals do not match row provenance")
        expected_train = tuple(row.stable_row_id for row in rows if row.animal_id in development)
        expected_held_out = tuple(row.stable_row_id for row in rows if row.animal_id == evaluation[0])
        if train != expected_train or held_out != expected_held_out:
            raise SpatialPerturbationSplitError("rows do not match the whole-animal partition")
        target_map = dict(targets)
        if tuple(target_map) != perturbations or any(value not in genes for value in target_map.values()):
            raise SpatialPerturbationSplitError("manifest target genes are not frozen/measurable")
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
        adjacency = tuple(
            _adjacency_from(item, f"block_adjacency[{index}]")
            for index, item in enumerate(_items(self.block_adjacency, "block_adjacency"))
        )
        adjacency = tuple(
            sorted(adjacency, key=lambda item: (item.animal_id, item.section_id, item.first_block, item.second_block))
        )
        block_keys = {(row.animal_id, row.section_id, row.spatial_block) for row in rows}
        if len(set(adjacency)) != len(adjacency):
            raise SpatialPerturbationSplitError("block adjacency records must be unique")
        for item in adjacency:
            if (
                (item.animal_id, item.section_id, item.first_block) not in block_keys
                or (item.animal_id, item.section_id, item.second_block) not in block_keys
            ):
                raise SpatialPerturbationSplitError("block adjacency endpoint is not registered")
        identity = _sha(self.split_identity_sha256, "split_identity_sha256")
        for name, value in (
            ("split_id", split_id), ("split_seed", split_seed),
            ("development_animals", development), ("evaluation_animals", evaluation),
            ("train_rows", train), ("tune_rows", tune), ("evaluation_rows", held_out),
            ("gene_names", genes), ("perturbations", perturbations),
            ("row_provenance", rows), ("perturbation_parents", parents),
            ("primary_units", units), ("neighbour_cell_types", neighbour_types),
            ("perturbation_targets", targets), ("block_adjacency", adjacency),
            ("safe_control_label", safe_label), ("split_identity_sha256", identity),
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
        manifest.split_id, manifest.split_seed, manifest.development_animals,
        manifest.evaluation_animals, manifest.train_rows, manifest.tune_rows,
        manifest.evaluation_rows, manifest.gene_names, manifest.perturbations,
        manifest.split_identity_sha256, manifest.row_provenance,
        manifest.perturbation_parents, manifest.primary_units,
        manifest.neighbour_cell_types, manifest.perturbation_targets,
        manifest.block_adjacency, manifest.safe_control_label,
    )


def split_manifest_to_mapping(manifest: BridgeSplitManifest) -> dict[str, object]:
    snapshot = _snapshot_manifest(manifest)
    mapping = _manifest_unsigned(snapshot)
    mapping["split_identity_sha256"] = snapshot.split_identity_sha256
    return mapping


def build_pilot_fold(
    metadata: BridgeSplitMetadata, evaluation_animal: str
) -> BridgeSplitManifest:
    if type(metadata) is not BridgeSplitMetadata:
        raise SpatialPerturbationSplitError("metadata must be BridgeSplitMetadata")
    snapshot = BridgeSplitMetadata(
        metadata.rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label,
    )
    animals = tuple(sorted({row.animal_id for row in snapshot.rows}))
    if len(animals) != 3:
        raise SpatialPerturbationSplitError("pilot requires exactly three animals")
    evaluation = _safe_text(evaluation_animal, "evaluation_animal")
    if evaluation not in animals:
        raise SpatialPerturbationSplitError("evaluation_animal must be an exact registered animal")
    development = tuple(animal for animal in animals if animal != evaluation)
    target_map = dict(snapshot.perturbation_targets)
    parents = tuple(
        sorted(
            (_make_parent(animal, perturbation, target_map[perturbation])
             for animal in animals for perturbation in snapshot.perturbations),
            key=lambda item: item.parent_id,
        )
    )
    units = tuple(
        sorted(
            (_make_unit(animal, perturbation, target_map[perturbation], cell_type, band)
             for animal in animals for perturbation in snapshot.perturbations
             for cell_type in snapshot.neighbour_cell_types
             for band in cast(tuple[str, str], _science()["primary_bands"])),
            key=lambda item: item.unit_id,
        )
    )
    unsigned: dict[str, object] = {
        "split_id": f"pilot_leave_one_animal_out:{evaluation}",
        "split_seed": _science()["split_seed"],
        "development_animals": list(development), "evaluation_animals": [evaluation],
        "train_rows": [row.stable_row_id for row in snapshot.rows if row.animal_id in development],
        "tune_rows": [],
        "evaluation_rows": [row.stable_row_id for row in snapshot.rows if row.animal_id == evaluation],
        "gene_names": list(snapshot.gene_names), "perturbations": list(snapshot.perturbations),
        "row_provenance": [_row_mapping(row) for row in snapshot.rows],
        "perturbation_parents": [_parent_mapping(item) for item in parents],
        "primary_units": [_unit_mapping(item) for item in units],
        "neighbour_cell_types": list(snapshot.neighbour_cell_types),
        "perturbation_targets": [list(item) for item in snapshot.perturbation_targets],
        "block_adjacency": [_adjacency_mapping(item) for item in snapshot.block_adjacency],
        "safe_control_label": snapshot.safe_control_label,
    }
    return BridgeSplitManifest(
        **unsigned, split_identity_sha256=_identity(unsigned)  # type: ignore[arg-type]
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
    perturbation_neighbour_cell_ids: tuple[str, ...]
    safe_neighbour_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _sha(self.unit_id, "unit_id"))
        object.__setattr__(
            self, "perturbation_neighbour_cell_ids",
            _text_items(self.perturbation_neighbour_cell_ids, "perturbation_neighbour_cell_ids", sort=True),
        )
        object.__setattr__(
            self, "safe_neighbour_cell_ids",
            _text_items(self.safe_neighbour_cell_ids, "safe_neighbour_cell_ids", sort=True),
        )
        if set(self.perturbation_neighbour_cell_ids) & set(self.safe_neighbour_cell_ids):
            raise SpatialPerturbationSplitError("treatment and safe neighbour IDs must be disjoint")


def _unit_evidence_from(value: object, name: str) -> BridgePrimaryUnitEvidence:
    if type(value) is BridgePrimaryUnitEvidence:
        item = cast(BridgePrimaryUnitEvidence, value)
        return BridgePrimaryUnitEvidence(
            item.unit_id, item.perturbation_neighbour_cell_ids, item.safe_neighbour_cell_ids
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {"unit_id", "perturbation_neighbour_cell_ids", "safe_neighbour_cell_ids"}
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
        "perturbation_neighbour_cell_ids": list(snapshot.perturbation_neighbour_cell_ids),
        "safe_neighbour_cell_ids": list(snapshot.safe_neighbour_cell_ids),
    }


def _evidence_unsigned(evidence: "BridgeEligibilityEvidence") -> dict[str, object]:
    return {
        "split_identity_sha256": evidence.split_identity_sha256,
        "parent_evidence": [_parent_evidence_mapping(item) for item in evidence.parent_evidence],
        "unit_evidence": [_unit_evidence_mapping(item) for item in evidence.unit_evidence],
    }


@dataclass(frozen=True, slots=True)
class BridgeEligibilityEvidence:
    split_identity_sha256: str
    parent_evidence: tuple[BridgeParentEvidence, ...]
    unit_evidence: tuple[BridgePrimaryUnitEvidence, ...]
    evidence_identity_sha256: str

    def __post_init__(self) -> None:
        split_identity = _sha(self.split_identity_sha256, "split_identity_sha256")
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
            cell for item in units for cell in item.perturbation_neighbour_cell_ids
        )
        safe_ids = tuple(cell for item in units for cell in item.safe_neighbour_cell_ids)
        if len(set(treatment_ids)) != len(treatment_ids) or len(set(safe_ids)) != len(safe_ids):
            raise SpatialPerturbationSplitError("a neighbour cell cannot appear in two frozen units")
        if set(treatment_ids) & set(safe_ids):
            raise SpatialPerturbationSplitError("treatment and safe neighbour evidence must be disjoint")
        identity = _sha(self.evidence_identity_sha256, "evidence_identity_sha256")
        object.__setattr__(self, "split_identity_sha256", split_identity)
        object.__setattr__(self, "parent_evidence", parents)
        object.__setattr__(self, "unit_evidence", units)
        object.__setattr__(self, "evidence_identity_sha256", identity)
        if identity != _identity(_evidence_unsigned(self)):
            raise SpatialPerturbationSplitError("eligibility evidence identity does not match evidence")


def _snapshot_evidence(evidence: BridgeEligibilityEvidence) -> BridgeEligibilityEvidence:
    if type(evidence) is not BridgeEligibilityEvidence:
        raise SpatialPerturbationSplitError("evidence must be BridgeEligibilityEvidence")
    return BridgeEligibilityEvidence(
        evidence.split_identity_sha256, evidence.parent_evidence,
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
            "split_identity_sha256", "parent_evidence", "unit_evidence",
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
    rows: tuple[BridgeSplitRow, ...],
) -> Counter[tuple[str, str, str, str, str]]:
    return Counter(
        (
            row.section_id,
            row.spatial_block,
            row.source_cell_type,
            row.cell_type,
            row.distance_band,
        )
        for row in rows
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
            safe, animal=frozen.animal_id, perturbation=frozen.perturbation_id,
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
    for unit_id, frozen_unit in frozen_unit_by_id.items():
        unit_evidence = evidence_unit_by_id[unit_id]
        treatment = _resolve_rows(unit_evidence.perturbation_neighbour_cell_ids, row_by_cell)
        safe = _resolve_rows(unit_evidence.safe_neighbour_cell_ids, row_by_cell)
        _require_rows(
            treatment, animal=frozen_unit.animal_id, perturbation=frozen_unit.perturbation_id,
            role="perturbation_neighbour", label="unperturbed",
            cell_type=frozen_unit.neighbour_cell_type, band=frozen_unit.band,
        )
        _require_rows(
            safe, animal=frozen_unit.animal_id, perturbation=frozen_unit.perturbation_id,
            role="safe_neighbour", label="unperturbed",
            cell_type=frozen_unit.neighbour_cell_type, band=frozen_unit.band,
        )
        treatment_meets_threshold = len(treatment) >= minimum_cell_type
        safe_meets_threshold = len(safe) >= minimum_cell_type
        if (
            treatment_meets_threshold
            and safe_meets_threshold
            and _neighbour_strata(treatment) != _neighbour_strata(safe)
        ):
            raise SpatialPerturbationSplitError(
                "matched safe neighbours do not match the frozen stratum multiset"
            )
        source_strata = frozen_source_strata[
            (frozen_unit.animal_id, frozen_unit.perturbation_id)
        ]
        if any(
            (row.section_id, row.source_cell_type) not in source_strata
            for row in treatment + safe
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
            treatment_strata = _neighbour_strata(treatment)
            safe_strata = _neighbour_strata(safe)
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
    "BridgeSplitManifest", "BridgeSplitMetadata", "BridgeSplitRow",
    "FrozenPerturbationParent", "FrozenPrimaryUnit", "SpatialPerturbationSplitError",
    "build_bridge_eligibility_evidence", "build_pilot_fold",
    "eligibility_evidence_to_mapping", "evaluate_bridge_eligibility",
    "split_manifest_to_mapping",
]
