"""Frozen animal-level bridge splits and outcome-blind eligibility evidence."""

from __future__ import annotations

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

_SPLIT_SEED = 11
_PRIMARY_BANDS = ("proximal", "local")
_MAX_TEXT_LENGTH = 256
_MAX_ROWS = 10_000_000
_MAX_EVIDENCE_ITEMS = 10_000
_MAX_COUNT = 1_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _science() -> dict[str, object]:
    """Return fresh literals so rebound module exports cannot alter the contract."""
    return {
        "split_seed": 11,
        "primary_bands": ("proximal", "local"),
        "minimum_source_cells": 20,
        "minimum_safe_source_cells": 20,
        "minimum_band_neighbours": 50,
        "minimum_cell_type_neighbours": 30,
        "minimum_spatial_blocks": 3,
    }


class SpatialPerturbationSplitError(ValueError):
    """Split metadata or eligibility evidence is unsafe or inconsistent."""


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT_LENGTH:
        raise SpatialPerturbationSplitError(
            f"{name} must be bounded non-empty built-in NFC text"
        )
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SpatialPerturbationSplitError(f"{name} must be trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise SpatialPerturbationSplitError(f"{name} contains unsafe control text")
    return value


def _sha256(value: object, name: str) -> str:
    text = _safe_text(value, name)
    if _SHA256.fullmatch(text) is None:
        raise SpatialPerturbationSplitError(f"{name} must be a lowercase SHA-256")
    return text


def _integer(value: object, name: str, *, maximum: int = _MAX_COUNT) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise SpatialPerturbationSplitError(
            f"{name} must be a bounded nonnegative built-in integer"
        )
    return value


def _items(
    value: object, name: str, *, maximum: int = _MAX_EVIDENCE_ITEMS
) -> tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise SpatialPerturbationSplitError(f"{name} must be a built-in list or tuple")
    values = cast(list[object] | tuple[object, ...], value)
    if len(values) > maximum:
        raise SpatialPerturbationSplitError(f"{name} exceeds the item limit")
    return tuple(values)


def _text_items(
    value: object,
    name: str,
    *,
    maximum: int = _MAX_EVIDENCE_ITEMS,
    sort: bool = False,
) -> tuple[str, ...]:
    frozen = tuple(
        _safe_text(item, f"{name}[{index}]")
        for index, item in enumerate(_items(value, name, maximum=maximum))
    )
    if len(set(frozen)) != len(frozen):
        raise SpatialPerturbationSplitError(f"{name} must contain unique values")
    return tuple(sorted(frozen)) if sort else frozen


def _integer_items(
    value: object, name: str, *, maximum_items: int = _MAX_EVIDENCE_ITEMS
) -> tuple[int, ...]:
    frozen = tuple(
        _integer(item, f"{name}[{index}]", maximum=_MAX_ROWS)
        for index, item in enumerate(_items(value, name, maximum=maximum_items))
    )
    if len(set(frozen)) != len(frozen):
        raise SpatialPerturbationSplitError(f"{name} must contain unique row identities")
    return tuple(sorted(frozen))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pairwise_disjoint(values: tuple[tuple[str, ...], ...]) -> bool:
    seen: set[str] = set()
    for items in values:
        if seen.intersection(items):
            return False
        seen.update(items)
    return True


@dataclass(frozen=True, slots=True)
class BridgeSplitRow:
    """Stable, outcome-free provenance for one input row."""

    stable_row_id: int
    cell_id: str
    animal_id: str
    section_id: str
    spatial_block: str

    def __post_init__(self) -> None:
        row_id = _integer(self.stable_row_id, "stable_row_id", maximum=_MAX_ROWS)
        for name, value in (
            ("cell_id", self.cell_id),
            ("animal_id", self.animal_id),
            ("section_id", self.section_id),
            ("spatial_block", self.spatial_block),
        ):
            object.__setattr__(self, name, _safe_text(value, name))
        object.__setattr__(self, "stable_row_id", row_id)


def _split_row_from_object(value: object, name: str) -> BridgeSplitRow:
    if type(value) is BridgeSplitRow:
        row = cast(BridgeSplitRow, value)
        return BridgeSplitRow(
            row.stable_row_id,
            row.cell_id,
            row.animal_id,
            row.section_id,
            row.spatial_block,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "stable_row_id", "cell_id", "animal_id", "section_id", "spatial_block"
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeSplitRow(
            raw["stable_row_id"],  # type: ignore[arg-type]
            raw["cell_id"],  # type: ignore[arg-type]
            raw["animal_id"],  # type: ignore[arg-type]
            raw["section_id"],  # type: ignore[arg-type]
            raw["spatial_block"],  # type: ignore[arg-type]
        )
    raise SpatialPerturbationSplitError(f"{name} must be BridgeSplitRow")


def _validate_row_provenance(rows: tuple[BridgeSplitRow, ...]) -> None:
    if not rows:
        raise SpatialPerturbationSplitError("rows must not be empty")
    row_ids = tuple(row.stable_row_id for row in rows)
    cell_ids = tuple(row.cell_id for row in rows)
    if len(set(row_ids)) != len(row_ids):
        raise SpatialPerturbationSplitError("stable_row_id values must be unique")
    if len(set(cell_ids)) != len(cell_ids):
        raise SpatialPerturbationSplitError("cell_id values must be unique")
    section_animals: dict[str, str] = {}
    for row in rows:
        prior = section_animals.setdefault(row.section_id, row.animal_id)
        if prior != row.animal_id:
            raise SpatialPerturbationSplitError(
                "a section cannot be assigned to two animals"
            )


@dataclass(frozen=True, slots=True)
class BridgeSplitMetadata:
    """Defensive snapshot of the outcome-free inputs used to build a fold."""

    rows: tuple[BridgeSplitRow, ...]
    gene_names: tuple[str, ...]
    perturbations: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = tuple(
            _split_row_from_object(value, f"rows[{index}]")
            for index, value in enumerate(_items(self.rows, "rows", maximum=_MAX_ROWS))
        )
        rows = tuple(sorted(rows, key=lambda row: row.stable_row_id))
        _validate_row_provenance(rows)
        genes = _text_items(self.gene_names, "gene_names", sort=True)
        perturbations = _text_items(self.perturbations, "perturbations", sort=True)
        if not genes or not perturbations:
            raise SpatialPerturbationSplitError(
                "gene_names and perturbations must not be empty"
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "gene_names", genes)
        object.__setattr__(self, "perturbations", perturbations)


def _row_mapping(row: BridgeSplitRow) -> dict[str, object]:
    snapshot = _split_row_from_object(row, "row_provenance")
    return {
        "stable_row_id": snapshot.stable_row_id,
        "cell_id": snapshot.cell_id,
        "animal_id": snapshot.animal_id,
        "section_id": snapshot.section_id,
        "spatial_block": snapshot.spatial_block,
    }


def _manifest_unsigned_mapping(manifest: "BridgeSplitManifest") -> dict[str, object]:
    return {
        "split_id": manifest.split_id,
        "split_seed": manifest.split_seed,
        "development_animals": list(manifest.development_animals),
        "evaluation_animals": list(manifest.evaluation_animals),
        "train_rows": list(manifest.train_rows),
        "tune_rows": list(manifest.tune_rows),
        "evaluation_rows": list(manifest.evaluation_rows),
        "gene_names": list(manifest.gene_names),
        "perturbations": list(manifest.perturbations),
        "row_provenance": [_row_mapping(row) for row in manifest.row_provenance],
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

    def __post_init__(self) -> None:
        split_id = _safe_text(self.split_id, "split_id")
        split_seed = _integer(self.split_seed, "split_seed")
        if split_seed != _science()["split_seed"]:
            raise SpatialPerturbationSplitError("split seed is frozen to 11")
        development = _text_items(
            self.development_animals, "development_animals", sort=True
        )
        evaluation = _text_items(
            self.evaluation_animals, "evaluation_animals", sort=True
        )
        if len(evaluation) != 1 or set(development) & set(evaluation):
            raise SpatialPerturbationSplitError(
                "evaluation animal must be one isolated registered animal"
            )
        if split_id != f"pilot_leave_one_animal_out:{evaluation[0]}":
            raise SpatialPerturbationSplitError("split_id does not bind evaluation animal")
        train = _integer_items(self.train_rows, "train_rows", maximum_items=_MAX_ROWS)
        tune = _integer_items(self.tune_rows, "tune_rows", maximum_items=_MAX_ROWS)
        held_out = _integer_items(
            self.evaluation_rows, "evaluation_rows", maximum_items=_MAX_ROWS
        )
        if tune:
            raise SpatialPerturbationSplitError(
                "pilot fold keeps each development animal wholly in train_rows"
            )
        if set(train) & set(held_out):
            raise SpatialPerturbationSplitError("animal partitions must be disjoint")
        genes = _text_items(self.gene_names, "gene_names", sort=True)
        perturbations = _text_items(self.perturbations, "perturbations", sort=True)
        if not genes or not perturbations:
            raise SpatialPerturbationSplitError(
                "gene_names and perturbations must not be empty"
            )
        provenance = tuple(
            _split_row_from_object(value, f"row_provenance[{index}]")
            for index, value in enumerate(
                _items(self.row_provenance, "row_provenance", maximum=_MAX_ROWS)
            )
        )
        provenance = tuple(sorted(provenance, key=lambda row: row.stable_row_id))
        _validate_row_provenance(provenance)
        animals = tuple(sorted({row.animal_id for row in provenance}))
        if tuple(sorted(development + evaluation)) != animals:
            raise SpatialPerturbationSplitError(
                "manifest animal declarations do not match row provenance"
            )
        expected_train = tuple(
            row.stable_row_id for row in provenance if row.animal_id in development
        )
        expected_evaluation = tuple(
            row.stable_row_id for row in provenance if row.animal_id == evaluation[0]
        )
        if train != expected_train or held_out != expected_evaluation:
            raise SpatialPerturbationSplitError(
                "rows do not match the whole-animal partition in provenance"
            )
        identity = _sha256(self.split_identity_sha256, "split_identity_sha256")
        for name, value in (
            ("split_id", split_id),
            ("split_seed", split_seed),
            ("development_animals", development),
            ("evaluation_animals", evaluation),
            ("train_rows", train),
            ("tune_rows", tune),
            ("evaluation_rows", held_out),
            ("gene_names", genes),
            ("perturbations", perturbations),
            ("row_provenance", provenance),
            ("split_identity_sha256", identity),
        ):
            object.__setattr__(self, name, value)
        if identity != _identity(_manifest_unsigned_mapping(self)):
            raise SpatialPerturbationSplitError("split identity does not match manifest")

    @property
    def development_rows(self) -> tuple[int, ...]:
        return tuple(sorted(self.train_rows + self.tune_rows))


def _snapshot_manifest(manifest: BridgeSplitManifest) -> BridgeSplitManifest:
    if type(manifest) is not BridgeSplitManifest:
        raise SpatialPerturbationSplitError("manifest must be BridgeSplitManifest")
    return BridgeSplitManifest(
        manifest.split_id,
        manifest.split_seed,
        manifest.development_animals,
        manifest.evaluation_animals,
        manifest.train_rows,
        manifest.tune_rows,
        manifest.evaluation_rows,
        manifest.gene_names,
        manifest.perturbations,
        manifest.split_identity_sha256,
        manifest.row_provenance,
    )


def split_manifest_to_mapping(manifest: BridgeSplitManifest) -> dict[str, object]:
    snapshot = _snapshot_manifest(manifest)
    mapping = _manifest_unsigned_mapping(snapshot)
    mapping["split_identity_sha256"] = snapshot.split_identity_sha256
    return mapping


def build_pilot_fold(
    metadata: BridgeSplitMetadata, evaluation_animal: str
) -> BridgeSplitManifest:
    if type(metadata) is not BridgeSplitMetadata:
        raise SpatialPerturbationSplitError("metadata must be BridgeSplitMetadata")
    snapshot = BridgeSplitMetadata(
        metadata.rows, metadata.gene_names, metadata.perturbations
    )
    evaluation = _safe_text(evaluation_animal, "evaluation_animal")
    animals = tuple(sorted({row.animal_id for row in snapshot.rows}))
    if evaluation not in animals:
        raise SpatialPerturbationSplitError(
            "evaluation_animal must be an exact registered animal"
        )
    development = tuple(animal for animal in animals if animal != evaluation)
    unsigned = {
        "split_id": f"pilot_leave_one_animal_out:{evaluation}",
        "split_seed": _science()["split_seed"],
        "development_animals": list(development),
        "evaluation_animals": [evaluation],
        "train_rows": [
            row.stable_row_id for row in snapshot.rows if row.animal_id in development
        ],
        "tune_rows": [],
        "evaluation_rows": [
            row.stable_row_id for row in snapshot.rows if row.animal_id == evaluation
        ],
        "gene_names": list(snapshot.gene_names),
        "perturbations": list(snapshot.perturbations),
        "row_provenance": [_row_mapping(row) for row in snapshot.rows],
    }
    return BridgeSplitManifest(
        **unsigned, split_identity_sha256=_identity(unsigned)  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class BandEligibilityEvidence:
    band: str
    perturbation_neighbour_cell_ids: tuple[str, ...]
    safe_neighbour_cell_ids: tuple[str, ...]
    perturbation_block_ids: tuple[str, ...]
    safe_block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        band = _safe_text(self.band, "band")
        primary_bands = cast(tuple[str, str], _science()["primary_bands"])
        if band not in primary_bands:
            raise SpatialPerturbationSplitError("band must be proximal or local")
        for name in (
            "perturbation_neighbour_cell_ids", "safe_neighbour_cell_ids",
            "perturbation_block_ids", "safe_block_ids",
        ):
            object.__setattr__(self, name, _text_items(getattr(self, name), name, sort=True))
        object.__setattr__(self, "band", band)


@dataclass(frozen=True, slots=True)
class CellTypeBandEligibilityEvidence:
    unit_id: str
    neighbour_cell_type: str
    band: str
    perturbation_neighbour_cell_ids: tuple[str, ...]
    safe_neighbour_cell_ids: tuple[str, ...]
    perturbation_block_ids: tuple[str, ...]
    safe_block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        unit_id = _safe_text(self.unit_id, "unit_id")
        cell_type = _safe_text(self.neighbour_cell_type, "neighbour_cell_type")
        band = _safe_text(self.band, "band")
        primary_bands = cast(tuple[str, str], _science()["primary_bands"])
        if band not in primary_bands:
            raise SpatialPerturbationSplitError("cell-type band must be primary")
        for name in (
            "perturbation_neighbour_cell_ids", "safe_neighbour_cell_ids",
            "perturbation_block_ids", "safe_block_ids",
        ):
            object.__setattr__(self, name, _text_items(getattr(self, name), name, sort=True))
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "neighbour_cell_type", cell_type)
        object.__setattr__(self, "band", band)


@dataclass(frozen=True, slots=True)
class AnimalCoverageEvidence:
    animal_id: str
    scoreable_perturbations: int
    registered_perturbations: int

    def __post_init__(self) -> None:
        animal = _safe_text(self.animal_id, "animal_id")
        scoreable = _integer(self.scoreable_perturbations, "scoreable_perturbations")
        registered = _integer(self.registered_perturbations, "registered_perturbations")
        if registered == 0 or scoreable > registered:
            raise SpatialPerturbationSplitError(
                "animal perturbation coverage counts are inconsistent"
            )
        object.__setattr__(self, "animal_id", animal)
        object.__setattr__(self, "scoreable_perturbations", scoreable)
        object.__setattr__(self, "registered_perturbations", registered)


def _band_from_object(value: object, name: str) -> BandEligibilityEvidence:
    if type(value) is BandEligibilityEvidence:
        item = cast(BandEligibilityEvidence, value)
        return BandEligibilityEvidence(
            item.band,
            item.perturbation_neighbour_cell_ids,
            item.safe_neighbour_cell_ids,
            item.perturbation_block_ids,
            item.safe_block_ids,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "band", "perturbation_neighbour_cell_ids", "safe_neighbour_cell_ids",
            "perturbation_block_ids", "safe_block_ids",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BandEligibilityEvidence(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be BandEligibilityEvidence")


def _cell_type_band_from_object(
    value: object, name: str
) -> CellTypeBandEligibilityEvidence:
    if type(value) is CellTypeBandEligibilityEvidence:
        item = cast(CellTypeBandEligibilityEvidence, value)
        return CellTypeBandEligibilityEvidence(
            item.unit_id,
            item.neighbour_cell_type,
            item.band,
            item.perturbation_neighbour_cell_ids,
            item.safe_neighbour_cell_ids,
            item.perturbation_block_ids,
            item.safe_block_ids,
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "unit_id", "neighbour_cell_type", "band",
            "perturbation_neighbour_cell_ids", "safe_neighbour_cell_ids",
            "perturbation_block_ids", "safe_block_ids",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return CellTypeBandEligibilityEvidence(
            **cast(dict[str, object], raw)  # type: ignore[arg-type]
        )
    raise SpatialPerturbationSplitError(
        f"{name} must be CellTypeBandEligibilityEvidence"
    )


def _coverage_from_object(value: object, name: str) -> AnimalCoverageEvidence:
    if type(value) is AnimalCoverageEvidence:
        item = cast(AnimalCoverageEvidence, value)
        return AnimalCoverageEvidence(
            item.animal_id, item.scoreable_perturbations, item.registered_perturbations
        )
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {"animal_id", "scoreable_perturbations", "registered_perturbations"}
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return AnimalCoverageEvidence(**cast(dict[str, object], raw))  # type: ignore[arg-type]
    raise SpatialPerturbationSplitError(f"{name} must be AnimalCoverageEvidence")


def _band_mapping(item: BandEligibilityEvidence) -> dict[str, object]:
    return {
        "band": item.band,
        "perturbation_neighbour_cell_ids": list(item.perturbation_neighbour_cell_ids),
        "safe_neighbour_cell_ids": list(item.safe_neighbour_cell_ids),
        "perturbation_block_ids": list(item.perturbation_block_ids),
        "safe_block_ids": list(item.safe_block_ids),
    }


def _cell_type_band_mapping(item: CellTypeBandEligibilityEvidence) -> dict[str, object]:
    mapping = _band_mapping(
        BandEligibilityEvidence(
            item.band,
            item.perturbation_neighbour_cell_ids,
            item.safe_neighbour_cell_ids,
            item.perturbation_block_ids,
            item.safe_block_ids,
        )
    )
    return {
        "unit_id": item.unit_id,
        "neighbour_cell_type": item.neighbour_cell_type,
        **mapping,
    }


def _evidence_unsigned_mapping(evidence: "BridgeEligibilityEvidence") -> dict[str, object]:
    return {
        "perturbation_source_cell_ids": list(evidence.perturbation_source_cell_ids),
        "perturbation_source_block_ids": list(evidence.perturbation_source_block_ids),
        "safe_source_cell_ids": list(evidence.safe_source_cell_ids),
        "safe_source_block_ids": list(evidence.safe_source_block_ids),
        "band_evidence": [_band_mapping(item) for item in evidence.band_evidence],
        "cell_type_band_evidence": [
            _cell_type_band_mapping(item) for item in evidence.cell_type_band_evidence
        ],
        "target_gene": evidence.target_gene,
        "measurable_gene_names": list(evidence.measurable_gene_names),
        "per_animal_perturbation_coverage": [
            {
                "animal_id": item.animal_id,
                "scoreable_perturbations": item.scoreable_perturbations,
                "registered_perturbations": item.registered_perturbations,
            }
            for item in evidence.per_animal_perturbation_coverage
        ],
        "primary_scoreable": evidence.primary_scoreable,
        "primary_total": evidence.primary_total,
        "abstained": evidence.abstained,
        "attempted": evidence.attempted,
        "frozen_primary_units_sha256": evidence.frozen_primary_units_sha256,
    }


@dataclass(frozen=True, slots=True)
class BridgeEligibilityEvidence:
    perturbation_source_cell_ids: tuple[str, ...]
    perturbation_source_block_ids: tuple[str, ...]
    safe_source_cell_ids: tuple[str, ...]
    safe_source_block_ids: tuple[str, ...]
    band_evidence: tuple[BandEligibilityEvidence, ...]
    cell_type_band_evidence: tuple[CellTypeBandEligibilityEvidence, ...]
    target_gene: str
    measurable_gene_names: tuple[str, ...]
    per_animal_perturbation_coverage: tuple[AnimalCoverageEvidence, ...]
    primary_scoreable: int
    primary_total: int
    abstained: int
    attempted: int
    frozen_primary_units_sha256: str
    evidence_identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "perturbation_source_cell_ids", "perturbation_source_block_ids",
            "safe_source_cell_ids", "safe_source_block_ids",
        ):
            object.__setattr__(self, name, _text_items(getattr(self, name), name, sort=True))
        bands = tuple(
            _band_from_object(item, f"band_evidence[{index}]")
            for index, item in enumerate(_items(self.band_evidence, "band_evidence"))
        )
        primary_bands = cast(tuple[str, str], _science()["primary_bands"])
        bands = tuple(sorted(bands, key=lambda item: primary_bands.index(item.band)))
        if tuple(item.band for item in bands) != primary_bands:
            raise SpatialPerturbationSplitError(
                "eligibility evidence must contain both primary bands exactly once"
            )
        if not _pairwise_disjoint(
            tuple(item.perturbation_neighbour_cell_ids for item in bands)
        ) or not _pairwise_disjoint(tuple(item.safe_neighbour_cell_ids for item in bands)):
            raise SpatialPerturbationSplitError(
                "neighbour cell IDs must not be counted across primary bands"
            )
        if any(
            set(item.perturbation_neighbour_cell_ids).intersection(
                item.safe_neighbour_cell_ids
            )
            for item in bands
        ):
            raise SpatialPerturbationSplitError(
                "treatment and safe neighbour cell IDs must be disjoint"
            )
        cell_type_bands = tuple(
            _cell_type_band_from_object(item, f"cell_type_band_evidence[{index}]")
            for index, item in enumerate(
                _items(self.cell_type_band_evidence, "cell_type_band_evidence")
            )
        )
        cell_type_bands = tuple(sorted(cell_type_bands, key=lambda item: item.unit_id))
        unit_ids = tuple(item.unit_id for item in cell_type_bands)
        if not cell_type_bands or len(set(unit_ids)) != len(unit_ids):
            raise SpatialPerturbationSplitError(
                "cell_type_band_evidence must contain unique frozen units"
            )
        if set(item.band for item in cell_type_bands) != set(primary_bands):
            raise SpatialPerturbationSplitError(
                "cell-type evidence must cover both primary bands"
            )
        if not _pairwise_disjoint(
            tuple(item.perturbation_neighbour_cell_ids for item in cell_type_bands)
        ) or not _pairwise_disjoint(
            tuple(item.safe_neighbour_cell_ids for item in cell_type_bands)
        ):
            raise SpatialPerturbationSplitError(
                "cell-type neighbour IDs must not be counted across frozen units"
            )
        if any(
            set(item.perturbation_neighbour_cell_ids).intersection(
                item.safe_neighbour_cell_ids
            )
            for item in cell_type_bands
        ):
            raise SpatialPerturbationSplitError(
                "treatment and safe cell-type neighbour IDs must be disjoint"
            )
        if set(self.perturbation_source_cell_ids).intersection(self.safe_source_cell_ids):
            raise SpatialPerturbationSplitError(
                "treatment and safe source cell IDs must be disjoint"
            )
        target = _safe_text(self.target_gene, "target_gene")
        genes = _text_items(self.measurable_gene_names, "measurable_gene_names", sort=True)
        if not genes:
            raise SpatialPerturbationSplitError("measurable_gene_names must not be empty")
        coverage = tuple(
            _coverage_from_object(item, f"per_animal_perturbation_coverage[{index}]")
            for index, item in enumerate(
                _items(
                    self.per_animal_perturbation_coverage,
                    "per_animal_perturbation_coverage",
                )
            )
        )
        coverage = tuple(sorted(coverage, key=lambda item: item.animal_id))
        if not coverage or len({item.animal_id for item in coverage}) != len(coverage):
            raise SpatialPerturbationSplitError(
                "per-animal perturbation coverage must have unique animals"
            )
        primary_scoreable = _integer(self.primary_scoreable, "primary_scoreable")
        primary_total = _integer(self.primary_total, "primary_total")
        abstained = _integer(self.abstained, "abstained")
        attempted = _integer(self.attempted, "attempted")
        if (
            primary_total == 0
            or attempted == 0
            or primary_scoreable > primary_total
            or abstained > attempted
        ):
            raise SpatialPerturbationSplitError(
                "primary coverage or abstention counts are inconsistent"
            )
        frozen_units = _sha256(
            self.frozen_primary_units_sha256, "frozen_primary_units_sha256"
        )
        identity = _sha256(self.evidence_identity_sha256, "evidence_identity_sha256")
        for name, value in (
            ("band_evidence", bands),
            ("cell_type_band_evidence", cell_type_bands),
            ("target_gene", target),
            ("measurable_gene_names", genes),
            ("per_animal_perturbation_coverage", coverage),
            ("primary_scoreable", primary_scoreable),
            ("primary_total", primary_total),
            ("abstained", abstained),
            ("attempted", attempted),
            ("frozen_primary_units_sha256", frozen_units),
            ("evidence_identity_sha256", identity),
        ):
            object.__setattr__(self, name, value)
        if identity != _identity(_evidence_unsigned_mapping(self)):
            raise SpatialPerturbationSplitError(
                "eligibility evidence identity does not match evidence"
            )


def _snapshot_evidence(evidence: BridgeEligibilityEvidence) -> BridgeEligibilityEvidence:
    if type(evidence) is not BridgeEligibilityEvidence:
        raise SpatialPerturbationSplitError(
            "evidence must be BridgeEligibilityEvidence"
        )
    return BridgeEligibilityEvidence(
        evidence.perturbation_source_cell_ids,
        evidence.perturbation_source_block_ids,
        evidence.safe_source_cell_ids,
        evidence.safe_source_block_ids,
        evidence.band_evidence,
        evidence.cell_type_band_evidence,
        evidence.target_gene,
        evidence.measurable_gene_names,
        evidence.per_animal_perturbation_coverage,
        evidence.primary_scoreable,
        evidence.primary_total,
        evidence.abstained,
        evidence.attempted,
        evidence.frozen_primary_units_sha256,
        evidence.evidence_identity_sha256,
    )


def eligibility_evidence_to_mapping(
    evidence: BridgeEligibilityEvidence,
) -> dict[str, object]:
    snapshot = _snapshot_evidence(evidence)
    mapping = _evidence_unsigned_mapping(snapshot)
    mapping["evidence_identity_sha256"] = snapshot.evidence_identity_sha256
    return mapping


@dataclass(frozen=True, slots=True)
class BridgeEligibilityResult:
    eligible: bool
    reason: str | None
    reasons: tuple[str, ...]
    evidence: BridgeEligibilityEvidence
    eligibility_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise SpatialPerturbationSplitError("eligible must be a built-in boolean")
        reasons = _text_items(self.reasons, "reasons")
        reason = self.reason
        if reason is not None:
            reason = _safe_text(reason, "reason")
        evidence = _evidence_from_object(self.evidence, "evidence")
        expected_reasons = _eligibility_reasons(evidence)
        if (
            reasons != expected_reasons
            or self.eligible is not (not expected_reasons)
            or reason != (expected_reasons[0] if expected_reasons else None)
        ):
            raise SpatialPerturbationSplitError(
                "eligibility decision does not match evidence"
            )
        result_identity = _sha256(
            self.eligibility_identity_sha256, "eligibility_identity_sha256"
        )
        unsigned = {
            "eligible": self.eligible,
            "reason": reason,
            "reasons": list(reasons),
            "evidence": eligibility_evidence_to_mapping(evidence),
        }
        if result_identity != _identity(unsigned):
            raise SpatialPerturbationSplitError(
                "eligibility result identity does not match result"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "eligibility_identity_sha256", result_identity)

    @property
    def evidence_identity_sha256(self) -> str:
        return self.evidence.evidence_identity_sha256


def _evidence_from_object(value: object, name: str) -> BridgeEligibilityEvidence:
    if type(value) is BridgeEligibilityEvidence:
        return _snapshot_evidence(cast(BridgeEligibilityEvidence, value))
    if type(value) is dict:
        raw = cast(dict[object, object], value)
        expected = {
            "perturbation_source_cell_ids", "perturbation_source_block_ids",
            "safe_source_cell_ids", "safe_source_block_ids", "band_evidence",
            "cell_type_band_evidence", "target_gene", "measurable_gene_names",
            "per_animal_perturbation_coverage", "primary_scoreable", "primary_total",
            "abstained", "attempted", "frozen_primary_units_sha256",
            "evidence_identity_sha256",
        }
        if set(raw) != expected:
            raise SpatialPerturbationSplitError(f"{name} has unexpected fields")
        return BridgeEligibilityEvidence(
            **cast(dict[str, object], raw)  # type: ignore[arg-type]
        )
    raise SpatialPerturbationSplitError(f"{name} must be BridgeEligibilityEvidence")


def _below(count: int, total: int, threshold: Fraction) -> bool:
    return Fraction(count, total) < threshold


def _eligibility_reasons(
    snapshot: BridgeEligibilityEvidence,
) -> tuple[str, ...]:
    science = _science()
    minimum_source_cells = cast(int, science["minimum_source_cells"])
    minimum_safe_source_cells = cast(int, science["minimum_safe_source_cells"])
    minimum_band_neighbours = cast(int, science["minimum_band_neighbours"])
    minimum_cell_type_neighbours = cast(int, science["minimum_cell_type_neighbours"])
    minimum_spatial_blocks = cast(int, science["minimum_spatial_blocks"])
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if len(snapshot.perturbation_source_cell_ids) < minimum_source_cells:
        add("insufficient_perturbation_coverage")
    if len(snapshot.safe_source_cell_ids) < minimum_safe_source_cells:
        add("insufficient_safe_control_coverage")
    if len(snapshot.perturbation_source_block_ids) < minimum_spatial_blocks:
        add("insufficient_spatial_blocks")
    if len(snapshot.safe_source_block_ids) < minimum_spatial_blocks:
        add("insufficient_safe_control_spatial_blocks")
    for item in snapshot.band_evidence:
        if len(item.perturbation_neighbour_cell_ids) < minimum_band_neighbours:
            add("insufficient_band_neighbours")
        if len(item.safe_neighbour_cell_ids) < minimum_band_neighbours:
            add("insufficient_safe_control_band_neighbours")
        if len(item.perturbation_block_ids) < minimum_spatial_blocks:
            add("insufficient_spatial_blocks")
        if len(item.safe_block_ids) < minimum_spatial_blocks:
            add("insufficient_safe_control_spatial_blocks")
    for cell_type_item in snapshot.cell_type_band_evidence:
        if len(cell_type_item.perturbation_neighbour_cell_ids) < minimum_cell_type_neighbours:
            add("insufficient_band_neighbours")
        if len(cell_type_item.safe_neighbour_cell_ids) < minimum_cell_type_neighbours:
            add("insufficient_safe_control_band_neighbours")
        if len(cell_type_item.perturbation_block_ids) < minimum_spatial_blocks:
            add("insufficient_spatial_blocks")
        if len(cell_type_item.safe_block_ids) < minimum_spatial_blocks:
            add("insufficient_safe_control_spatial_blocks")
    if snapshot.target_gene not in snapshot.measurable_gene_names:
        add("target_gene_not_measurable")
    if any(
        _below(
            item.scoreable_perturbations,
            item.registered_perturbations,
            Fraction(4, 5),
        )
        for item in snapshot.per_animal_perturbation_coverage
    ):
        add("insufficient_perturbation_coverage")
    if _below(snapshot.primary_scoreable, snapshot.primary_total, Fraction(4, 5)):
        add("insufficient_primary_unit_coverage")
    if Fraction(snapshot.abstained, snapshot.attempted) > Fraction(1, 5):
        add("excessive_abstention")
    return tuple(reasons)


def evaluate_bridge_eligibility(
    evidence: BridgeEligibilityEvidence,
) -> BridgeEligibilityResult:
    snapshot = _snapshot_evidence(evidence)
    frozen_reasons = _eligibility_reasons(snapshot)
    unsigned = {
        "eligible": not frozen_reasons,
        "reason": frozen_reasons[0] if frozen_reasons else None,
        "reasons": list(frozen_reasons),
        "evidence": eligibility_evidence_to_mapping(snapshot),
    }
    return BridgeEligibilityResult(
        **unsigned, eligibility_identity_sha256=_identity(unsigned)  # type: ignore[arg-type]
    )


def _generated_ids(prefix: str, count: object) -> tuple[str, ...]:
    value = _integer(count, prefix, maximum=_MAX_EVIDENCE_ITEMS)
    return tuple(f"{prefix}_{index:05d}" for index in range(value))


def unit_counts(
    *,
    source: int = MIN_SOURCE_CELLS,
    safe_source: int = MIN_SAFE_SOURCE_CELLS,
    neighbours: int = MIN_BAND_NEIGHBOURS,
    safe_neighbours: int = MIN_BAND_NEIGHBOURS,
    cell_type_neighbours: int = MIN_CELL_TYPE_NEIGHBOURS,
    safe_cell_type_neighbours: int = MIN_CELL_TYPE_NEIGHBOURS,
    blocks: int = MIN_SPATIAL_BLOCKS,
    safe_blocks: int = MIN_SPATIAL_BLOCKS,
    target_gene_present: bool = True,
    perturbation_scoreable: int = 4,
    perturbation_total: int = 5,
    primary_scoreable: int = 4,
    primary_total: int = 5,
    abstained: int = 1,
    attempted: int = 5,
) -> BridgeEligibilityEvidence:
    """Build synthetic unique-ID evidence for tests without accepting raw counts."""
    if type(target_gene_present) is not bool:
        raise SpatialPerturbationSplitError(
            "target_gene_present must be a built-in boolean"
        )
    perturbation_coverage = AnimalCoverageEvidence(
        "animal_1", perturbation_scoreable, perturbation_total
    )
    primary_scoreable_value = _integer(primary_scoreable, "primary_scoreable")
    primary_total_value = _integer(primary_total, "primary_total")
    abstained_value = _integer(abstained, "abstained")
    attempted_value = _integer(attempted, "attempted")
    perturbation_blocks = _generated_ids("perturbation_block", blocks)
    control_blocks = _generated_ids("safe_block", safe_blocks)
    bands = tuple(
        BandEligibilityEvidence(
            band,
            _generated_ids(f"{band}_perturbation_neighbour", neighbours),
            _generated_ids(f"{band}_safe_neighbour", safe_neighbours),
            perturbation_blocks,
            control_blocks,
        )
        for band in cast(tuple[str, str], _science()["primary_bands"])
    )
    cell_type_bands = tuple(
        CellTypeBandEligibilityEvidence(
            f"animal_1:perturbation_1:cell_type_1:{band}",
            "cell_type_1",
            band,
            _generated_ids(f"{band}_cell_type_perturbation_neighbour", cell_type_neighbours),
            _generated_ids(f"{band}_cell_type_safe_neighbour", safe_cell_type_neighbours),
            perturbation_blocks,
            control_blocks,
        )
        for band in cast(tuple[str, str], _science()["primary_bands"])
    )
    cell_type_bands = tuple(sorted(cell_type_bands, key=lambda item: item.unit_id))
    unsigned: dict[str, object] = {
        "perturbation_source_cell_ids": list(_generated_ids("source_cell", source)),
        "perturbation_source_block_ids": list(perturbation_blocks),
        "safe_source_cell_ids": list(_generated_ids("safe_source_cell", safe_source)),
        "safe_source_block_ids": list(control_blocks),
        "band_evidence": [_band_mapping(item) for item in bands],
        "cell_type_band_evidence": [
            _cell_type_band_mapping(item) for item in cell_type_bands
        ],
        "target_gene": "TargetGene",
        "measurable_gene_names": ["TargetGene"] if target_gene_present else ["OtherGene"],
        "per_animal_perturbation_coverage": [
            {
                "animal_id": perturbation_coverage.animal_id,
                "scoreable_perturbations": perturbation_coverage.scoreable_perturbations,
                "registered_perturbations": perturbation_coverage.registered_perturbations,
            }
        ],
        "primary_scoreable": primary_scoreable_value,
        "primary_total": primary_total_value,
        "abstained": abstained_value,
        "attempted": attempted_value,
        "frozen_primary_units_sha256": _identity(
            {"contract": "frozen_primary_bridge_units_v1", "total": primary_total_value}
        ),
    }
    return BridgeEligibilityEvidence(
        **unsigned, evidence_identity_sha256=_identity(unsigned)  # type: ignore[arg-type]
    )


__all__ = [
    "MAX_ABSTENTION",
    "MIN_BAND_NEIGHBOURS",
    "MIN_CELL_TYPE_NEIGHBOURS",
    "MIN_COVERAGE",
    "MIN_SAFE_SOURCE_CELLS",
    "MIN_SOURCE_CELLS",
    "MIN_SPATIAL_BLOCKS",
    "AnimalCoverageEvidence",
    "BandEligibilityEvidence",
    "BridgeEligibilityEvidence",
    "BridgeEligibilityResult",
    "BridgeSplitManifest",
    "BridgeSplitMetadata",
    "BridgeSplitRow",
    "CellTypeBandEligibilityEvidence",
    "SpatialPerturbationSplitError",
    "build_pilot_fold",
    "eligibility_evidence_to_mapping",
    "evaluate_bridge_eligibility",
    "split_manifest_to_mapping",
    "unit_counts",
]
