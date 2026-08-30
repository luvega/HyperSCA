from __future__ import annotations

import random
import tracemalloc
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.evaluation import spatial_perturbation_neighbors as neighbors_module
from src.evaluation.spatial_perturbation_neighbors import build_bridge_neighbors
from src.evaluation.spatial_perturbation_split import freeze_bridge_neighbour_table


INPUT_COLUMNS = [
    "animal_id",
    "section_id",
    "spatial_block",
    "cell_id",
    "perturbation_id",
    "cell_type",
    "x",
    "y",
    "barcode_positive",
]
OUTPUT_COLUMNS = [
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
]


def _frame(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def _cell(
    cell_id: str,
    x: float,
    *,
    animal: str = "mouse_1",
    section: str = "section_1",
    block: str = "block_1",
    perturbation: str = "unperturbed",
    cell_type: str = "astrocyte",
    positive: bool = False,
    y: float = 0.0,
) -> tuple[object, ...]:
    return (
        animal,
        section,
        block,
        cell_id,
        perturbation,
        cell_type,
        x,
        y,
        positive,
    )


def test_exact_output_schema_and_all_four_disjoint_rank_bands() -> None:
    cells = [
        _cell(
            "source",
            0.0,
            perturbation="guide_A",
            cell_type="oligodendrocyte",
            positive=True,
        )
    ]
    cells.extend(_cell(f"neighbor_{rank:02d}", float(rank)) for rank in range(1, 61))

    result = build_bridge_neighbors(_frame(cells))

    assert type(result) is pd.DataFrame
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["rank"].tolist() == list(range(1, 61))
    expected = (
        ["proximal"] * 5
        + ["local"] * 10
        + ["transition"] * 15
        + ["distal"] * 30
    )
    assert result["band"].tolist() == expected
    assert not result["is_safe_control"].any()
    assert set(zip(result["rank"], result["band"])) == {
        (rank, band)
        for band, start, stop in (
            ("proximal", 1, 5),
            ("local", 6, 15),
            ("transition", 16, 30),
            ("distal", 31, 60),
        )
        for rank in range(start, stop + 1)
    }


def test_empty_atomic_table_returns_an_empty_exact_schema() -> None:
    result = build_bridge_neighbors(pd.DataFrame(columns=INPUT_COLUMNS))

    assert result.empty
    assert list(result.columns) == OUTPUT_COLUMNS


def test_positive_cells_are_excluded_but_still_leave_full_rank_gaps() -> None:
    cells = _frame(
        [
            _cell("source_a", 0.0, perturbation="guide_A", positive=True),
            _cell("source_b", 1.0, perturbation="guide_A", positive=True),
            _cell("z_neighbor", 2.0),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=2)

    assert result[["source_cell_id", "neighbor_cell_id", "rank"]].to_dict("records") == [
        {
            "source_cell_id": "source_a",
            "neighbor_cell_id": "z_neighbor",
            "rank": 2,
        }
    ]
    assert not set(result["neighbor_cell_id"]) & {"source_a", "source_b"}


def test_search_is_isolated_by_animal_and_section() -> None:
    cells = _frame(
        [
            _cell("a1_source", 0.0, animal="a1", section="shared", perturbation="g1", positive=True),
            _cell("a1_neighbor", 100.0, animal="a1", section="shared"),
            _cell("a2_source", 0.0, animal="a2", section="shared", perturbation="g2", positive=True),
            _cell("a2_neighbor", 1.0, animal="a2", section="shared"),
            _cell("a1s2_source", 0.0, animal="a1", section="second", perturbation="g3", positive=True),
            _cell("a1s2_neighbor", 2.0, animal="a1", section="second"),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=1)

    assert set(
        zip(result["source_cell_id"], result["neighbor_cell_id"])
    ) == {
        ("a1_source", "a1_neighbor"),
        ("a2_source", "a2_neighbor"),
        ("a1s2_source", "a1s2_neighbor"),
    }


def test_ties_and_logical_dedup_use_cell_ids_deterministically() -> None:
    cells = _frame(
        [
            _cell("source_b", 2.0, perturbation="guide_A", positive=True),
            _cell("shared", 1.0),
            _cell("source_a", 0.0, perturbation="guide_A", positive=True),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=1)

    assert result.to_dict("records") == [
        {
            "animal_id": "mouse_1",
            "section_id": "section_1",
            "spatial_block": "block_1",
            "source_cell_id": "source_a",
            "neighbor_cell_id": "shared",
            "perturbation_id": "guide_A",
            "source_cell_type": "astrocyte",
            "neighbor_cell_type": "astrocyte",
            "rank": 1,
            "band": "proximal",
            "is_safe_control": False,
        }
    ]


def test_logical_dedup_keeps_the_minimum_rank_across_same_guide_sources() -> None:
    cells = _frame(
        [
            _cell("a_source_near", 0.0, perturbation="guide_A", positive=True),
            _cell("shared", 1.0),
            *[_cell(f"private_{x}", float(x)) for x in range(95, 100)],
            _cell("z_source_far", 100.0, perturbation="guide_A", positive=True),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=6)
    shared = result.loc[result["neighbor_cell_id"] == "shared"]

    assert shared[["source_cell_id", "rank", "band"]].to_dict("records") == [
        {"source_cell_id": "a_source_near", "rank": 1, "band": "proximal"}
    ]


def test_ordinary_input_permutations_do_not_change_output() -> None:
    rows = [
        _cell("source", 0.0, perturbation="guide_A", positive=True),
        _cell("n_b", 1.0, y=1.0, cell_type="microglia"),
        _cell("n_a", -1.0, y=-1.0, cell_type="neuron"),
        _cell("n_c", 3.0),
    ]
    baseline = build_bridge_neighbors(_frame(rows), max_rank=3)

    shuffled = _frame(rows).sample(frac=1.0, random_state=981).reset_index(drop=True)
    pdt.assert_frame_equal(build_bridge_neighbors(shuffled, max_rank=3), baseline)
    assert baseline["neighbor_cell_id"].tolist()[:2] == ["n_a", "n_b"]


def test_extreme_finite_coordinates_keep_mathematical_distance_order() -> None:
    cells = _frame(
        [
            _cell("source", 1.0e308, perturbation="guide_A", positive=True),
            _cell("a_farther", -1.0e308),
            _cell("z_nearer", 0.0),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=1)

    assert result["neighbor_cell_id"].tolist() == ["z_nearer"]


def test_int64_coordinates_above_float_exactness_keep_cell_id_tie_break() -> None:
    source_coordinate = 2**53
    cells = _frame(
        [
            _cell("source", source_coordinate, perturbation="guide_A", positive=True),
            _cell("a_minus", source_coordinate - 1),
            _cell("z_plus", source_coordinate + 1),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=1)

    assert result["neighbor_cell_id"].tolist() == ["a_minus"]


def test_float128_extremes_are_ranked_without_squared_distance_overflow() -> None:
    maximum = np.finfo(np.longdouble).max
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("a_farther", 0.0),
            _cell("z_nearer", 0.0),
        ]
    )
    cells["x"] = np.array(
        [0, maximum / np.longdouble(8), maximum / np.longdouble(16)],
        dtype=np.longdouble,
    )
    cells["y"] = np.zeros(3, dtype=np.longdouble)
    assert np.isfinite(cells[["x", "y"]].to_numpy()).all()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_bridge_neighbors(cells, max_rank=1)

    assert not any(item.category is RuntimeWarning for item in caught)
    assert result["neighbor_cell_id"].tolist() == ["z_nearer"]


def test_mixed_float128_dynamic_range_does_not_underflow_tiny_distances() -> None:
    limits = np.finfo(np.longdouble)
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("a_farther", 0.0),
            _cell("z_nearer", 0.0),
            _cell("zz_huge", 0.0),
        ]
    )
    cells["x"] = np.array(
        [0, limits.tiny * 2, limits.tiny, limits.max / np.longdouble(8)],
        dtype=np.longdouble,
    )
    cells["y"] = np.zeros(4, dtype=np.longdouble)
    assert np.isfinite(cells[["x", "y"]].to_numpy()).all()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_bridge_neighbors(cells, max_rank=1)

    assert not any(item.category is RuntimeWarning for item in caught)
    assert result["neighbor_cell_id"].tolist() == ["z_nearer"]


def test_huge_equal_x_does_not_erase_tiny_orthogonal_distance_order() -> None:
    limits = np.finfo(np.longdouble)
    huge = limits.max / np.longdouble(8)
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("a_farther", 0.0),
            _cell("z_nearer", 0.0),
        ]
    )
    cells["x"] = np.full(3, huge, dtype=np.longdouble)
    cells["y"] = np.array(
        [0, limits.tiny * 2, limits.tiny], dtype=np.longdouble
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_bridge_neighbors(cells, max_rank=1)

    assert not any(item.category is RuntimeWarning for item in caught)
    assert result["neighbor_cell_id"].tolist() == ["z_nearer"]


def test_equal_squared_distances_with_different_dyadic_terms_tie_by_cell_id() -> None:
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("a_diagonal", 3 / 8, y=4 / 8),
            _cell("z_axis", 5 / 8, y=0.0),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=1)

    assert result["neighbor_cell_id"].tolist() == ["a_diagonal"]


def test_secondary_only_physical_neighbor_overlap_is_retained_and_task5_freezes_it() -> None:
    rows = [
        _cell("source_A", 0.0, perturbation="guide_A", positive=True),
        _cell("source_B", 1000.0, perturbation="guide_B", positive=True),
    ]
    rows.extend(_cell(f"a_private_{index:02d}", float(index)) for index in range(1, 16))
    rows.extend(_cell(f"b_private_{index:02d}", float(1000 - index)) for index in range(1, 16))
    rows.append(_cell("shared_transition", 500.0, cell_type="microglia"))

    result = build_bridge_neighbors(_frame(rows), max_rank=16)
    shared = result[result["neighbor_cell_id"] == "shared_transition"]

    assert set(zip(shared["perturbation_id"], shared["rank"], shared["band"])) == {
        ("guide_A", 16, "transition"),
        ("guide_B", 16, "transition"),
    }
    frozen = freeze_bridge_neighbour_table(result.to_dict("records"))
    assert frozen.relation_count == len(result)


def test_any_primary_cross_perturbation_overlap_removes_all_shared_associations() -> None:
    cells = _frame(
        [
            _cell("source_A", 0.0, perturbation="guide_A", positive=True),
            _cell("only_A", 1.0),
            _cell("shared", 50.0),
            _cell("only_B", 99.0),
            _cell("source_B", 100.0, perturbation="guide_B", positive=True),
        ]
    )

    result = build_bridge_neighbors(cells, max_rank=2)

    assert "shared" not in set(result["neighbor_cell_id"])
    assert set(zip(result["perturbation_id"], result["neighbor_cell_id"])) == {
        ("guide_A", "only_A"),
        ("guide_B", "only_B"),
    }
    freeze_bridge_neighbour_table(result.to_dict("records"))


def test_msafe_is_a_normal_source_with_control_flag_and_participates_in_contamination() -> None:
    clean = _frame(
        [
            _cell("safe", 0.0, perturbation="mSafe", positive=True),
            _cell("safe_neighbor", 1.0),
            _cell("treated", 100.0, perturbation="guide_A", positive=True),
            _cell("treated_neighbor", 99.0),
        ]
    )
    clean_result = build_bridge_neighbors(clean, max_rank=1)
    flags = dict(zip(clean_result["perturbation_id"], clean_result["is_safe_control"]))
    assert flags == {"guide_A": False, "mSafe": True}

    conflicting = _frame(
        [
            _cell("safe", 0.0, perturbation="mSafe", positive=True),
            _cell("safe_only", 1.0),
            _cell("shared", 50.0),
            _cell("treated_only", 99.0),
            _cell("treated", 100.0, perturbation="guide_A", positive=True),
        ]
    )
    conflict_result = build_bridge_neighbors(conflicting, max_rank=2)
    assert "shared" not in set(conflict_result["neighbor_cell_id"])
    assert set(conflict_result["perturbation_id"]) == {"guide_A", "mSafe"}
    freeze_bridge_neighbour_table(conflict_result.to_dict("records"))


@pytest.mark.parametrize("bad_value", [True, False, 0, 61, -1, 1.0, np.int64(5)])
def test_max_rank_requires_a_bounded_positive_builtin_integer(bad_value: object) -> None:
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("neighbor", 1.0),
        ]
    )
    with pytest.raises((TypeError, ValueError), match="max_rank"):
        build_bridge_neighbors(cells, max_rank=bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns=["y"]),
        lambda frame: frame.assign(extra="x"),
        lambda frame: frame.rename(columns={"cell_id": "barcode"}),
        lambda frame: frame.assign(cell_id=[1, "neighbor"]),
        lambda frame: frame.assign(x=["0", "1"]),
        lambda frame: frame.assign(barcode_positive=[1, 0]),
        lambda frame: frame.assign(perturbation_id=["", "unperturbed"]),
    ],
)
def test_invalid_atomic_schema_and_types_are_rejected(mutator: Any) -> None:
    valid = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("neighbor", 1.0),
        ]
    )
    with pytest.raises((TypeError, ValueError)):
        build_bridge_neighbors(mutator(valid))

    with pytest.raises((TypeError, ValueError), match="DataFrame"):
        build_bridge_neighbors(valid.to_dict("records"))  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_are_rejected(coordinate: float) -> None:
    cells = _frame(
        [
            _cell("source", coordinate, perturbation="guide_A", positive=True),
            _cell("neighbor", 1.0),
        ]
    )
    with pytest.raises(ValueError, match="finite"):
        build_bridge_neighbors(cells)


def test_cell_ids_must_be_globally_unique_not_only_unique_within_section() -> None:
    cells = _frame(
        [
            _cell("duplicate", 0.0, animal="a1", perturbation="g1", positive=True),
            _cell("n1", 1.0, animal="a1"),
            _cell("duplicate", 0.0, animal="a2", perturbation="g2", positive=True),
            _cell("n2", 1.0, animal="a2"),
        ]
    )
    with pytest.raises(ValueError, match="cell_id.*unique|unique.*cell_id"):
        build_bridge_neighbors(cells)


class _OversizedHostileFrame(pd.DataFrame):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: object) -> Any:
        raise AssertionError("oversized input was materialized before its cap check")


class _UnderreportingHostileFrame(pd.DataFrame):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: object) -> Any:
        raise AssertionError("underreporting input was accessed before row validation")


class _ForgedRowCountHostileFrame(pd.DataFrame):
    _metadata = ["accesses"]
    accesses: list[str]

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.accesses = []

    def __len__(self) -> int:
        self.accesses.append("len")
        return 1

    @property
    def axes(self) -> list[pd.Index]:
        self.accesses.append("axes")
        return [pd.RangeIndex(1), pd.Index(INPUT_COLUMNS)]

    def __getitem__(self, key: object) -> Any:
        self.accesses.append("getitem")
        raise AssertionError("forged row counts bypassed the input cap")


def test_dataframe_subclass_is_rejected_before_any_overrideable_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _ForgedRowCountHostileFrame(
        _frame(
            [
                _cell("source", 0.0, perturbation="guide_A", positive=True),
                _cell("neighbor", 1.0),
            ]
        )
    )
    monkeypatch.setattr(neighbors_module, "_MAX_INPUT_ROWS", 1)

    with pytest.raises(TypeError, match="DataFrame"):
        build_bridge_neighbors(hostile)

    assert hostile.accesses == []


def test_input_row_cap_is_checked_before_column_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = pd.DataFrame({"placeholder": [1, 2]})

    def forbidden_extract(*args: object, **kwargs: object) -> object:
        raise AssertionError("input was extracted before its row cap check")

    monkeypatch.setattr(neighbors_module, "_MAX_INPUT_ROWS", 1)
    monkeypatch.setattr(neighbors_module, "_validate_and_extract", forbidden_extract)
    with pytest.raises(ValueError, match="row|limit|large"):
        build_bridge_neighbors(cells)


def test_oversized_dataframe_subclass_is_rejected_before_column_access() -> None:
    hostile = _OversizedHostileFrame({"placeholder": [1]})

    with pytest.raises(TypeError, match="DataFrame"):
        build_bridge_neighbors(hostile)


def test_underreporting_dataframe_subclass_is_rejected_before_column_access() -> None:
    hostile = _UnderreportingHostileFrame(
        _frame(
            [
                _cell("source", 0.0, perturbation="guide_A", positive=True),
                _cell("neighbor", 1.0),
            ]
        )
    )

    with pytest.raises(TypeError, match="DataFrame"):
        build_bridge_neighbors(hostile)


def test_output_preallocation_cap_is_checked_before_distance_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("n1", 1.0),
            _cell("n2", 2.0),
        ]
    )
    monkeypatch.setattr(neighbors_module, "_MAX_OUTPUT_ROWS", 1)

    def fail_if_called(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("distance calculation ran before the output cap check")

    monkeypatch.setattr(neighbors_module, "_squared_distances", fail_if_called)
    with pytest.raises(ValueError, match="output|relation|limit|large"):
        build_bridge_neighbors(cells, max_rank=2)


def test_output_safety_cap_matches_task5_public_freezer_capacity() -> None:
    assert neighbors_module._MAX_OUTPUT_ROWS == 50_000


def test_pair_evaluation_cap_precedes_any_distance_key_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _frame(
        [
            _cell("source_a", 0.0, perturbation="guide_A", positive=True),
            _cell("source_b", 1.0, perturbation="guide_A", positive=True),
            _cell("neighbor", 2.0),
        ]
    )
    monkeypatch.setattr(neighbors_module, "_MAX_PAIR_EVALUATIONS", 5, raising=False)

    def fail_if_called(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("distance keys were materialized before the pair cap")

    monkeypatch.setattr(neighbors_module, "_squared_distances", fail_if_called)
    with pytest.raises(ValueError, match="pair|evaluation|limit|large"):
        build_bridge_neighbors(cells)


def test_float_dyadic_coordinates_are_converted_once_per_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _frame(
        [
            _cell("source_a", 0.25, y=0.5, perturbation="guide_A", positive=True),
            _cell("source_b", 1.25, y=1.5, perturbation="guide_A", positive=True),
            _cell("neighbor", 2.25, y=2.5),
        ]
    )
    original = neighbors_module._as_dyadic
    call_count = 0

    def observe(value: object) -> object:
        nonlocal call_count
        call_count += 1
        return original(value)

    monkeypatch.setattr(neighbors_module, "_as_dyadic", observe)
    build_bridge_neighbors(cells)

    assert call_count == 2 * len(cells)


def test_extreme_distance_key_memory_budget_fails_before_key_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 128
    rows = [
        _cell("source", 0.0, perturbation="guide_A", positive=True),
        *(_cell(f"neighbor_{index:03d}", 0.0) for index in range(1, count)),
    ]
    cells = _frame(rows)
    limits = np.finfo(np.longdouble)
    cells["x"] = np.array(
        [
            0,
            *(
                limits.max / np.longdouble(8)
                if index % 2
                else limits.tiny
                for index in range(1, count)
            ),
        ],
        dtype=np.longdouble,
    )
    cells["y"] = np.zeros(count, dtype=np.longdouble)
    monkeypatch.setattr(
        neighbors_module, "_MAX_DISTANCE_KEY_BYTES", 200_000, raising=False
    )

    def fail_if_called(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("distance keys were materialized before memory budgeting")

    monkeypatch.setattr(neighbors_module, "_squared_distances", fail_if_called)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="memory|budget|key|large"):
            build_bridge_neighbors(cells)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000


def test_integer_coordinates_never_use_longdouble_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames: list[pd.DataFrame] = []
    for dtype, farthest in (
        (np.int64, np.iinfo(np.int64).max),
        (np.uint64, np.iinfo(np.uint64).max),
    ):
        cells = _frame(
            [
                _cell("source", 0.0, perturbation="guide_A", positive=True),
                _cell("a_minus", 0.0),
                _cell("z_plus", 0.0),
                _cell("zz_farthest", 0.0),
            ]
        )
        cells["x"] = np.array(
            [2**53, 2**53 - 1, 2**53 + 1, farthest], dtype=dtype
        )
        cells["y"] = np.zeros(4, dtype=dtype)
        frames.append(cells)

    def forbidden_longdouble(*args: object, **kwargs: object) -> object:
        raise AssertionError("integer coordinates entered a floating conversion path")

    monkeypatch.setattr(neighbors_module.np, "longdouble", forbidden_longdouble)
    for cells in frames:
        result = build_bridge_neighbors(cells, max_rank=1)
        assert result["neighbor_cell_id"].tolist() == ["a_minus"]


def test_each_source_allocates_only_a_section_length_distance_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _frame(
        [
            _cell("a_source", 0.0, animal="a", section="s1", perturbation="g1", positive=True),
            _cell("a_neighbor", 1.0, animal="a", section="s1"),
            _cell("b_source", 0.0, animal="b", section="s2", perturbation="g2", positive=True),
            _cell("b_neighbor", 1.0, animal="b", section="s2"),
        ]
    )
    observed_lengths: list[int] = []
    original = neighbors_module._squared_distances

    def observe(
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        source_index: int,
    ) -> np.ndarray[Any, Any]:
        observed_lengths.append(len(x))
        return original(x, y, source_index)

    monkeypatch.setattr(neighbors_module, "_squared_distances", observe)
    build_bridge_neighbors(cells)

    assert observed_lengths == [2, 2]


def test_large_section_uses_bounded_linear_memory_not_an_all_cell_matrix() -> None:
    count = 5_000
    rows = [_cell("source", 0.0, perturbation="guide_A", positive=True)]
    rows.extend(_cell(f"neighbor_{index:05d}", float(index + 1)) for index in range(count))
    cells = _frame(rows)

    tracemalloc.start()
    try:
        result = build_bridge_neighbors(cells)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(result) == 60
    assert peak < 32_000_000


def test_two_source_extreme_sections_release_each_distance_key_list_before_next() -> None:
    count = 4_000
    rows = [
        _cell("source_a", 0.0, perturbation="guide_A", positive=True),
        _cell("source_b", 0.0, perturbation="guide_A", positive=True),
    ]
    rows.extend(_cell(f"neighbor_{index:05d}", 0.0) for index in range(2, count))
    cells = _frame(rows)
    limits = np.finfo(np.longdouble)
    cells["x"] = np.array(
        [
            limits.max / np.longdouble(8),
            limits.max / np.longdouble(8),
            *([limits.tiny] * (count - 2)),
        ],
        dtype=np.longdouble,
    )
    cells["y"] = np.zeros(count, dtype=np.longdouble)

    tracemalloc.start()
    try:
        build_bridge_neighbors(cells)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 64 * 1024 * 1024


def test_builder_does_not_read_or_mutate_python_or_numpy_global_rng_state() -> None:
    cells = _frame(
        [
            _cell("source", 0.0, perturbation="guide_A", positive=True),
            _cell("neighbor", 1.0),
        ]
    )
    random.seed(891)
    np.random.seed(812)
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    build_bridge_neighbors(cells)

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
