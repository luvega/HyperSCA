"""Deterministic, bounded spatial neighborhoods for perturbation bridges.

The builder is outcome blind.  It ranks cells only by their within-section
coordinates, keeps barcode-negative neighbors, and removes physical neighbors
whose cross-perturbation reuse touches a primary distance band.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
import heapq
from itertools import chain
import unicodedata
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


_INPUT_COLUMNS = (
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
_OUTPUT_COLUMNS = (
    "animal_id",
    "section_id",
    "spatial_block",
    "source_cell_id",
    "neighbor_cell_id",
    "perturbation_id",
    "source_cell_type",
    "neighbor_cell_type",
    "rank",
    "band",
    "is_safe_control",
)
_TEXT_COLUMNS = (
    "animal_id",
    "section_id",
    "spatial_block",
    "cell_id",
    "perturbation_id",
    "cell_type",
)
_MAX_TEXT_LENGTH = 256
_MAX_INPUT_ROWS = 250_000
_MAX_PAIR_EVALUATIONS = 5_000_000
_MAX_DISTANCE_KEY_BYTES = 64 * 1024 * 1024
_DISTANCE_KEY_FIXED_BYTES = 128
# Task 5's public freezer accepts at most 50,000 relations.  Keeping the same
# upper bound means every successful result can be handed to it directly.
_MAX_OUTPUT_ROWS = 50_000


class SpatialPerturbationNeighborError(ValueError):
    """The atomic coordinate table cannot define a frozen neighborhood."""


class _Relation(TypedDict):
    animal_id: str
    section_id: str
    spatial_block: str
    source_cell_id: str
    neighbor_cell_id: str
    perturbation_id: str
    source_cell_type: str
    neighbor_cell_type: str
    rank: int
    band: str
    is_safe_control: bool


@dataclass(frozen=True, slots=True)
class _CoordinateColumn:
    values: NDArray[Any]
    integer: bool


_Dyadic = tuple[int, int]


@total_ordering
@dataclass(frozen=True, slots=True)
class _DistanceKey:
    """Canonical nonnegative dyadic value used only for exact ordering."""

    numerator: int
    exponent: int

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _DistanceKey):
            return NotImplemented
        if self.numerator == 0:
            return other.numerator != 0
        if other.numerator == 0:
            return False
        self_magnitude = self.numerator.bit_length() + self.exponent
        other_magnitude = other.numerator.bit_length() + other.exponent
        if self_magnitude != other_magnitude:
            return self_magnitude < other_magnitude
        if self.exponent == other.exponent:
            return self.numerator < other.numerator
        if self.exponent > other.exponent:
            return self.numerator << (self.exponent - other.exponent) < other.numerator
        return self.numerator < other.numerator << (other.exponent - self.exponent)


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT_LENGTH:
        raise SpatialPerturbationNeighborError(
            f"{name} must contain bounded non-empty built-in text"
        )
    text = cast(str, value)
    if text != text.strip() or unicodedata.normalize("NFC", text) != text:
        raise SpatialPerturbationNeighborError(f"{name} must contain trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in text):
        raise SpatialPerturbationNeighborError(f"{name} contains unsafe control text")
    return text


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{name: pd.Series(dtype="object") for name in _OUTPUT_COLUMNS[:-3]},
            "rank": pd.Series(dtype="int64"),
            "band": pd.Series(dtype="object"),
            "is_safe_control": pd.Series(dtype="bool"),
        },
        columns=_OUTPUT_COLUMNS,
    )


def _validate_schema(cells: pd.DataFrame) -> None:
    columns = tuple(cells.columns)
    if len(columns) != len(_INPUT_COLUMNS) or set(columns) != set(_INPUT_COLUMNS):
        raise SpatialPerturbationNeighborError(
            "cells must have exactly the frozen atomic coordinate schema"
        )


def _validate_and_extract(
    cells: pd.DataFrame,
) -> tuple[
    dict[str, list[str]],
    _CoordinateColumn,
    _CoordinateColumn,
    list[bool],
]:
    _validate_schema(cells)

    text: dict[str, list[str]] = {}
    for column in _TEXT_COLUMNS:
        values = cells[column].tolist()
        text[column] = [
            _safe_text(value, f"{column}[{index}]")
            for index, value in enumerate(values)
        ]
    if len(set(text["cell_id"])) != len(text["cell_id"]):
        raise SpatialPerturbationNeighborError("cell_id values must be globally unique")

    coordinates: list[_CoordinateColumn] = []
    for column in ("x", "y"):
        series = cells[column]
        integer = pd.api.types.is_integer_dtype(series.dtype)
        if (
            not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            raise SpatialPerturbationNeighborError(f"{column} must be real numeric data")
        try:
            coordinate_values = series.to_numpy(copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise SpatialPerturbationNeighborError(
                f"{column} must be real numeric data"
            ) from error
        try:
            finite = bool(np.isfinite(coordinate_values).all())
        except TypeError as error:
            raise SpatialPerturbationNeighborError(
                f"{column} must be real numeric data"
            ) from error
        if not finite:
            raise SpatialPerturbationNeighborError(f"{column} must contain finite values")
        coordinates.append(_CoordinateColumn(coordinate_values, integer))

    barcode_series = cells["barcode_positive"]
    if not pd.api.types.is_bool_dtype(barcode_series.dtype):
        raise SpatialPerturbationNeighborError(
            "barcode_positive must contain built-in booleans"
        )
    raw_barcodes = barcode_series.tolist()
    if any(type(value) is not bool for value in raw_barcodes):
        raise SpatialPerturbationNeighborError(
            "barcode_positive must contain built-in booleans"
        )
    return text, coordinates[0], coordinates[1], cast(list[bool], raw_barcodes)


def _canonical_dyadic(numerator: int, exponent: int) -> _Dyadic:
    if numerator == 0:
        return (0, 0)
    magnitude = abs(numerator)
    trailing_zeros = (magnitude & -magnitude).bit_length() - 1
    return (numerator >> trailing_zeros, exponent + trailing_zeros)


def _as_dyadic(value: np.floating[Any]) -> _Dyadic:
    numerator, denominator = value.as_integer_ratio()
    return _canonical_dyadic(numerator, -(denominator.bit_length() - 1))


def _prepare_dyadic_coordinates(
    column: _CoordinateColumn,
    indices: list[int],
) -> list[_Dyadic]:
    values = column.values[indices]
    if column.integer:
        return [_canonical_dyadic(int(value), 0) for value in values]
    return [_as_dyadic(cast(np.floating[Any], value)) for value in values]


def _estimated_distance_key_bytes(
    x: list[_Dyadic],
    y: list[_Dyadic],
) -> int:
    minimum_exponent: int | None = None
    maximum_significant_bit: int | None = None
    for numerator, exponent in chain(x, y):
        if numerator == 0:
            continue
        significant_bit = abs(numerator).bit_length() + exponent
        minimum_exponent = (
            exponent
            if minimum_exponent is None
            else min(minimum_exponent, exponent)
        )
        maximum_significant_bit = (
            significant_bit
            if maximum_significant_bit is None
            else max(maximum_significant_bit, significant_bit)
        )
    if minimum_exponent is None or maximum_significant_bit is None:
        key_bits = 1
    else:
        difference_bits = max(
            1, maximum_significant_bit - minimum_exponent + 1
        )
        key_bits = 2 * difference_bits + 1
    bytes_per_key = _DISTANCE_KEY_FIXED_BYTES + (key_bits + 7) // 8
    return len(x) * bytes_per_key


def _subtract_dyadics(first: _Dyadic, second: _Dyadic) -> _Dyadic:
    exponent = min(first[1], second[1])
    numerator = (
        (first[0] << (first[1] - exponent))
        - (second[0] << (second[1] - exponent))
    )
    return _canonical_dyadic(numerator, exponent)


def _add_nonnegative_dyadics(first: _Dyadic, second: _Dyadic) -> _Dyadic:
    exponent = min(first[1], second[1])
    numerator = (
        (first[0] << (first[1] - exponent))
        + (second[0] << (second[1] - exponent))
    )
    return _canonical_dyadic(numerator, exponent)


def _pair_squared_distance_key(
    source_x: _Dyadic,
    source_y: _Dyadic,
    neighbor_x: _Dyadic,
    neighbor_y: _Dyadic,
) -> _DistanceKey:
    x_delta = _subtract_dyadics(neighbor_x, source_x)
    y_delta = _subtract_dyadics(neighbor_y, source_y)
    x_square = (x_delta[0] * x_delta[0], 2 * x_delta[1])
    y_square = (y_delta[0] * y_delta[0], 2 * y_delta[1])
    numerator, exponent = _add_nonnegative_dyadics(x_square, y_square)
    return _DistanceKey(numerator, exponent)


def _squared_distances(
    x: list[_Dyadic],
    y: list[_Dyadic],
    source_index: int,
) -> list[_DistanceKey]:
    """Build exact section-length dyadic keys without a distance matrix."""
    source_x = x[source_index]
    source_y = y[source_index]
    return [
        _pair_squared_distance_key(source_x, source_y, neighbor_x, neighbor_y)
        for neighbor_x, neighbor_y in zip(x, y)
    ]


def _nearest_indices(
    section_size: int,
    source_index: int,
    distances: list[_DistanceKey],
    cell_ids: list[str],
    max_rank: int,
) -> list[int]:
    """Return at most ``max_rank`` non-self indices with deterministic ties."""
    return heapq.nsmallest(
        min(max_rank, section_size - 1),
        (index for index in range(section_size) if index != source_index),
        key=lambda index: (distances[index], cell_ids[index]),
    )


def _band(rank: int) -> str:
    if rank <= 5:
        return "proximal"
    if rank <= 15:
        return "local"
    if rank <= 30:
        return "transition"
    return "distal"


def _deduplicate(relations: list[_Relation]) -> list[_Relation]:
    selected: dict[tuple[str, str, str, str], _Relation] = {}
    for relation in relations:
        key = (
            relation["animal_id"],
            relation["section_id"],
            relation["perturbation_id"],
            relation["neighbor_cell_id"],
        )
        incumbent = selected.get(key)
        if incumbent is None or (
            relation["rank"], relation["source_cell_id"]
        ) < (
            incumbent["rank"], incumbent["source_cell_id"]
        ):
            selected[key] = relation
    return list(selected.values())


def _exclude_primary_cross_perturbation_contamination(
    relations: list[_Relation],
) -> list[_Relation]:
    perturbations: dict[tuple[str, str, str], set[str]] = {}
    has_primary: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (
            relation["animal_id"],
            relation["section_id"],
            relation["neighbor_cell_id"],
        )
        perturbations.setdefault(key, set()).add(relation["perturbation_id"])
        if relation["rank"] <= 15:
            has_primary.add(key)
    contaminated = {
        key
        for key, source_perturbations in perturbations.items()
        if key in has_primary and len(source_perturbations) > 1
    }
    return [
        relation
        for relation in relations
        if (
            relation["animal_id"],
            relation["section_id"],
            relation["neighbor_cell_id"],
        )
        not in contaminated
    ]


def build_bridge_neighbors(
    cells: pd.DataFrame,
    max_rank: int = 60,
    safe_control_label: str = "mSafe",
) -> pd.DataFrame:
    """Build the canonical within-section bridge-neighbor relation table.

    Every barcode-positive row is treated as a source.  Its complete local
    order includes barcode-positive cells, but only barcode-negative cells are
    emitted, so excluded positive cells can leave gaps in ``rank``.
    """
    if type(cells) is not pd.DataFrame:
        raise TypeError("cells must be a pandas DataFrame")
    reported_row_count = len(cells)
    axis_row_count = int(cells.axes[0].size)
    if axis_row_count > _MAX_INPUT_ROWS:
        raise SpatialPerturbationNeighborError("input row count exceeds the safe limit")
    if reported_row_count != axis_row_count:
        raise SpatialPerturbationNeighborError(
            "input row count is inconsistent with its row axis"
        )
    row_count = axis_row_count
    if type(max_rank) is not int or not 1 <= max_rank <= 60:
        raise SpatialPerturbationNeighborError(
            "max_rank must be a built-in integer between 1 and 60"
        )
    safe_control_label = _safe_text(safe_control_label, "safe_control_label")

    if row_count == 0:
        _validate_schema(cells)
        return _empty_output()
    text, x, y, barcode_positive = _validate_and_extract(cells)

    sections: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(text["animal_id"], text["section_id"])):
        sections.setdefault(key, []).append(index)
    source_indices = [index for index, positive in enumerate(barcode_positive) if positive]

    maximum_relation_count = sum(
        min(max_rank, len(sections[(text["animal_id"][index], text["section_id"][index])]) - 1)
        for index in source_indices
    )
    if maximum_relation_count > _MAX_OUTPUT_ROWS:
        raise SpatialPerturbationNeighborError(
            "possible output relation count exceeds the safe limit"
        )
    pair_evaluations = sum(
        len(section_indices)
        * sum(barcode_positive[index] for index in section_indices)
        for section_indices in sections.values()
    )
    if pair_evaluations > _MAX_PAIR_EVALUATIONS:
        raise SpatialPerturbationNeighborError(
            "pair evaluation count exceeds the safe limit"
        )

    relations: list[_Relation] = []
    for section_key in sorted(sections):
        global_indices = sections[section_key]
        local_cell_ids = [text["cell_id"][index] for index in global_indices]
        local_sources = sorted(
            (
                local_index
                for local_index, global_index in enumerate(global_indices)
                if barcode_positive[global_index]
            ),
            key=lambda local_index: local_cell_ids[local_index],
        )
        if not local_sources:
            continue
        local_x = _prepare_dyadic_coordinates(x, global_indices)
        local_y = _prepare_dyadic_coordinates(y, global_indices)
        if _estimated_distance_key_bytes(local_x, local_y) > _MAX_DISTANCE_KEY_BYTES:
            raise SpatialPerturbationNeighborError(
                "distance key memory budget exceeds the safe limit"
            )
        for local_source_index in local_sources:
            source_index = global_indices[local_source_index]
            distances = _squared_distances(local_x, local_y, local_source_index)
            nearest = _nearest_indices(
                len(global_indices),
                local_source_index,
                distances,
                local_cell_ids,
                max_rank,
            )
            del distances
            for rank, local_neighbor_index in enumerate(nearest, start=1):
                neighbor_index = global_indices[local_neighbor_index]
                if barcode_positive[neighbor_index]:
                    continue
                perturbation_id = text["perturbation_id"][source_index]
                relations.append(
                    {
                        "animal_id": text["animal_id"][source_index],
                        "section_id": text["section_id"][source_index],
                        "spatial_block": text["spatial_block"][source_index],
                        "source_cell_id": text["cell_id"][source_index],
                        "neighbor_cell_id": text["cell_id"][neighbor_index],
                        "perturbation_id": perturbation_id,
                        "source_cell_type": text["cell_type"][source_index],
                        "neighbor_cell_type": text["cell_type"][neighbor_index],
                        "rank": rank,
                        "band": _band(rank),
                        "is_safe_control": perturbation_id == safe_control_label,
                    }
                )

    relations = _deduplicate(relations)
    relations = _exclude_primary_cross_perturbation_contamination(relations)
    relations.sort(
        key=lambda item: (
            item["animal_id"],
            item["section_id"],
            item["perturbation_id"],
            item["rank"],
            item["neighbor_cell_id"],
            item["source_cell_id"],
        )
    )
    if not relations:
        return _empty_output()
    return pd.DataFrame.from_records(relations, columns=_OUTPUT_COLUMNS)


__all__ = ["SpatialPerturbationNeighborError", "build_bridge_neighbors"]
