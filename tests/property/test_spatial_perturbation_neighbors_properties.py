from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import random
import tracemalloc

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from hypothesis import example, given, settings, strategies as st

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


@dataclass(frozen=True)
class GeneratedSection:
    frame: pd.DataFrame
    expected_neighbor_ids: tuple[str, ...]
    expected_ranks: tuple[int, ...]
    max_rank: int


@dataclass(frozen=True)
class DyadicOracleCase:
    frame: pd.DataFrame
    expected_neighbor_ids: tuple[str, ...]


@st.composite
def generated_single_source_sections(draw: st.DrawFn) -> GeneratedSection:
    count = draw(st.integers(min_value=1, max_value=35))
    max_rank = draw(st.integers(min_value=1, max_value=min(30, count)))
    coordinates = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=-8, max_value=8),
                st.integers(min_value=-8, max_value=8),
            ),
            min_size=count,
            max_size=count,
        )
    )
    cell_types = draw(
        st.lists(
            st.sampled_from(("astrocyte", "microglia", "neuron")),
            min_size=count,
            max_size=count,
        )
    )
    source_x = draw(st.integers(min_value=-3, max_value=3))
    source_y = draw(st.integers(min_value=-3, max_value=3))
    rows: list[tuple[object, ...]] = [
        (
            "animal_generated",
            "section_generated",
            "block_source",
            "source",
            "guide_generated",
            "oligodendrocyte",
            source_x,
            source_y,
            True,
        )
    ]
    distances: list[tuple[int, str]] = []
    for index, ((x, y), cell_type) in enumerate(zip(coordinates, cell_types)):
        cell_id = f"neighbor_{index:03d}"
        rows.append(
            (
                "animal_generated",
                "section_generated",
                f"block_{index % 3}",
                cell_id,
                "unperturbed",
                cell_type,
                x,
                y,
                False,
            )
        )
        distances.append(((x - source_x) ** 2 + (y - source_y) ** 2, cell_id))
    expected = sorted(distances)[:max_rank]
    return GeneratedSection(
        pd.DataFrame(rows, columns=INPUT_COLUMNS),
        tuple(cell_id for _, cell_id in expected),
        tuple(range(1, max_rank + 1)),
        max_rank,
    )


def _fraction_dyadic(numerator: int, exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(numerator << exponent, 1)
    return Fraction(numerator, 1 << -exponent)


@st.composite
def generated_dyadic_oracle_sections(draw: st.DrawFn) -> DyadicOracleCase:
    huge_exponent = draw(st.integers(min_value=8_000, max_value=10_000))
    tiny_exponent = draw(st.integers(min_value=-10_000, max_value=-8_000))
    tiny_numerator = draw(st.integers(min_value=1, max_value=7))
    extra_count = draw(st.integers(min_value=0, max_value=4))
    extra_terms = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=-8, max_value=8),
                st.sampled_from((-9_000, -100, -1, 0, 1, 100, 9_000)),
            ),
            min_size=extra_count,
            max_size=extra_count,
        )
    )
    source_x = _fraction_dyadic(1, huge_exponent)
    coordinates: list[tuple[str, Fraction, Fraction]] = [
        (
            "a_farther",
            source_x,
            _fraction_dyadic(2 * tiny_numerator, tiny_exponent),
        ),
        (
            "z_nearer",
            source_x,
            _fraction_dyadic(tiny_numerator, tiny_exponent),
        ),
    ]
    coordinates.extend(
        (
            f"extra_{index}",
            source_x,
            _fraction_dyadic(numerator, exponent),
        )
        for index, (numerator, exponent) in enumerate(extra_terms)
    )
    rows: list[tuple[object, ...]] = [
        ("a", "s", "b", "source", "guide", "source", 0, 0, True)
    ]
    rows.extend(
        ("a", "s", "b", cell_id, "none", "neighbor", 0, 0, False)
        for cell_id, _, _ in coordinates
    )
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    all_coordinates = [(source_x, Fraction(0)), *[(x, y) for _, x, y in coordinates]]
    frame["x"] = np.array(
        [np.ldexp(np.longdouble(value.numerator), -(value.denominator.bit_length() - 1)) for value, _ in all_coordinates],
        dtype=np.longdouble,
    )
    frame["y"] = np.array(
        [np.ldexp(np.longdouble(value.numerator), -(value.denominator.bit_length() - 1)) for _, value in all_coordinates],
        dtype=np.longdouble,
    )
    expected = tuple(
        cell_id
        for cell_id, _, _ in sorted(
            coordinates,
            key=lambda item: (
                (item[1] - source_x) ** 2 + item[2] ** 2,
                item[0],
            ),
        )
    )
    return DyadicOracleCase(frame, expected)


@settings(max_examples=35, deadline=None)
@given(generated_single_source_sections())
def test_generated_coordinates_match_exact_squared_distance_and_id_order(
    case: GeneratedSection,
) -> None:
    result = build_bridge_neighbors(case.frame, max_rank=case.max_rank)

    assert tuple(result["neighbor_cell_id"]) == case.expected_neighbor_ids
    assert tuple(result["rank"]) == case.expected_ranks
    assert set(result["source_cell_id"]) == {"source"}
    assert set(result["spatial_block"]) == {"block_source"}


@settings(max_examples=20, deadline=None)
@given(generated_dyadic_oracle_sections())
def test_public_builder_matches_fraction_oracle_across_binary_dynamic_ranges(
    case: DyadicOracleCase,
) -> None:
    result = build_bridge_neighbors(case.frame, max_rank=min(60, len(case.frame) - 1))

    assert tuple(result["neighbor_cell_id"]) == case.expected_neighbor_ids
    assert tuple(result["rank"]) == tuple(range(1, len(case.expected_neighbor_ids) + 1))


@settings(max_examples=20, deadline=None)
@given(st.integers(min_value=1, max_value=60))
def test_generated_ranks_have_exactly_one_of_the_four_frozen_bands(max_rank: int) -> None:
    rows: list[tuple[object, ...]] = [
        ("a", "s", "source_block", "source", "guide", "source", 0, 0, True)
    ]
    rows.extend(
        ("a", "s", "neighbor_block", f"n_{rank:02d}", "none", "neighbor", rank, 0, False)
        for rank in range(1, 61)
    )
    result = build_bridge_neighbors(pd.DataFrame(rows, columns=INPUT_COLUMNS), max_rank=max_rank)
    expected_band = {
        rank: (
            "proximal"
            if rank <= 5
            else "local"
            if rank <= 15
            else "transition"
            if rank <= 30
            else "distal"
        )
        for rank in range(1, max_rank + 1)
    }
    assert dict(zip(result["rank"], result["band"])) == expected_band
    assert len(result) == len(set(zip(result["neighbor_cell_id"], result["band"])))


@settings(max_examples=30, deadline=None)
@given(generated_single_source_sections(), st.data())
def test_generated_multiclass_sections_are_permutation_invariant(
    case: GeneratedSection,
    data: st.DataObject,
) -> None:
    order = data.draw(st.permutations(tuple(range(len(case.frame)))))
    permuted = case.frame.iloc[order].reset_index(drop=True)

    first = build_bridge_neighbors(case.frame, max_rank=case.max_rank)
    second = build_bridge_neighbors(permuted, max_rank=case.max_rank)

    pdt.assert_frame_equal(first, second)
    assert set(first["neighbor_cell_type"]).issubset(
        {"astrocyte", "microglia", "neuron"}
    )


@settings(max_examples=30, deadline=None)
@example(near_sorts_first=True, order=list(range(8)))
@given(
    near_sorts_first=st.booleans(),
    order=st.permutations(tuple(range(8))),
)
def test_generated_permutations_keep_minimum_rank_same_guide_source(
    near_sorts_first: bool,
    order: list[int],
) -> None:
    near_id = "a_source_near" if near_sorts_first else "z_source_near"
    far_id = "z_source_far" if near_sorts_first else "a_source_far"
    rows: list[tuple[object, ...]] = [
        ("a", "s", "b", near_id, "guide_A", "source", 0, 0, True),
        ("a", "s", "b", "shared", "none", "neighbor", 1, 0, False),
    ]
    rows.extend(
        ("a", "s", "b", f"private_{x}", "none", "neighbor", x, 0, False)
        for x in range(95, 100)
    )
    rows.append(
        ("a", "s", "b", far_id, "guide_A", "source", 100, 0, True)
    )
    frame = pd.DataFrame([rows[index] for index in order], columns=INPUT_COLUMNS)

    result = build_bridge_neighbors(frame, max_rank=6)
    shared = result.loc[result["neighbor_cell_id"] == "shared"]

    assert shared[["source_cell_id", "rank", "band"]].to_dict("records") == [
        {"source_cell_id": near_id, "rank": 1, "band": "proximal"}
    ]


@settings(max_examples=25, deadline=None)
@given(
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=12),
)
def test_generated_animal_section_grids_never_cross_physical_boundaries(
    animal_count: int,
    sections_per_animal: int,
    neighbor_count: int,
) -> None:
    rows: list[tuple[object, ...]] = []
    expected: set[tuple[str, str, str, str]] = set()
    for animal_index in range(animal_count):
        animal = f"animal_{animal_index}"
        for section_index in range(sections_per_animal):
            section = f"section_{section_index}"
            source = f"source_{animal_index}_{section_index}"
            rows.append(
                (
                    animal,
                    section,
                    "source_block",
                    source,
                    "guide_A",
                    "source_type",
                    0.0,
                    0.0,
                    True,
                )
            )
            for neighbor_index in range(neighbor_count):
                neighbor = f"neighbor_{animal_index}_{section_index}_{neighbor_index}"
                rows.append(
                    (
                        animal,
                        section,
                        "neighbor_block",
                        neighbor,
                        "unperturbed",
                        "astrocyte",
                        float(neighbor_index + 1),
                        0.0,
                        False,
                    )
                )
                expected.add((animal, section, source, neighbor))
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)

    result = build_bridge_neighbors(frame, max_rank=min(12, neighbor_count))

    observed = set(
        zip(
            result["animal_id"],
            result["section_id"],
            result["source_cell_id"],
            result["neighbor_cell_id"],
        )
    )
    assert observed == expected


@settings(max_examples=15, deadline=None)
@given(
    st.sampled_from(("x", "y")),
    st.sampled_from((float("nan"), float("inf"), float("-inf"))),
)
def test_generated_nonfinite_coordinates_always_fail_closed(
    column: str,
    value: float,
) -> None:
    frame = pd.DataFrame(
        [
            ("a", "s", "b", "source", "guide", "source", 0.0, 0.0, True),
            ("a", "s", "b", "neighbor", "unperturbed", "neighbor", 1.0, 1.0, False),
        ],
        columns=INPUT_COLUMNS,
    )
    frame.loc[1, column] = value

    with pytest.raises(ValueError, match="finite"):
        build_bridge_neighbors(frame)


@settings(max_examples=15, deadline=None)
@given(st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5))
def test_generated_global_duplicate_cell_ids_fail_closed(
    first_animal: int,
    second_section: int,
) -> None:
    frame = pd.DataFrame(
        [
            (f"a_{first_animal}", "s_0", "b", "duplicate", "g1", "source", 0, 0, True),
            (f"a_{first_animal}", "s_0", "b", "n1", "none", "neighbor", 1, 0, False),
            ("other_animal", f"s_{second_section}", "b", "duplicate", "g2", "source", 0, 0, True),
            ("other_animal", f"s_{second_section}", "b", "n2", "none", "neighbor", 1, 0, False),
        ],
        columns=INPUT_COLUMNS,
    )
    with pytest.raises(ValueError, match="cell_id.*unique|unique.*cell_id"):
        build_bridge_neighbors(frame)


@settings(max_examples=8, deadline=None)
@given(st.integers(min_value=15, max_value=24))
def test_generated_secondary_only_overlap_roundtrips_through_task5(
    private_count: int,
) -> None:
    rows: list[tuple[object, ...]] = [
        ("a", "s", "b_a", "source_A", "guide_A", "source", 0, 0, True),
        ("a", "s", "b_b", "source_B", "guide_B", "source", 1000, 0, True),
    ]
    rows.extend(
        ("a", "s", "n", f"a_{index:02d}", "none", "astrocyte", index, 0, False)
        for index in range(1, private_count + 1)
    )
    rows.extend(
        ("a", "s", "n", f"b_{index:02d}", "none", "microglia", 1000 - index, 0, False)
        for index in range(1, private_count + 1)
    )
    rows.append(("a", "s", "n", "shared", "none", "neuron", 500, 0, False))
    max_rank = private_count + 1

    result = build_bridge_neighbors(pd.DataFrame(rows, columns=INPUT_COLUMNS), max_rank=max_rank)
    shared = result[result["neighbor_cell_id"] == "shared"]

    assert set(shared["perturbation_id"]) == {"guide_A", "guide_B"}
    assert set(shared["band"]).issubset({"transition", "distal"})
    frozen = freeze_bridge_neighbour_table(result.to_dict("records"))
    assert frozen.relation_count == len(result)


@settings(max_examples=8, deadline=None)
@given(st.sampled_from(("primary_primary", "primary_secondary")))
def test_generated_any_primary_overlap_removes_every_cross_perturbation_association(
    overlap_kind: str,
) -> None:
    if overlap_kind == "primary_primary":
        rows: list[tuple[object, ...]] = [
            ("a", "s", "a", "source_A", "guide_A", "source", 0, 0, True),
            ("a", "s", "n", "only_A", "none", "neighbor", 1, 0, False),
            ("a", "s", "n", "shared", "none", "neighbor", 50, 0, False),
            ("a", "s", "n", "only_B", "none", "neighbor", 99, 0, False),
            ("a", "s", "b", "source_B", "guide_B", "source", 100, 0, True),
        ]
        max_rank = 2
    else:
        rows = [
            ("a", "s", "a", "source_A", "guide_A", "source", 0, 0, True),
            ("a", "s", "b", "source_B", "guide_B", "source", 1000, 0, True),
            ("a", "s", "n", "shared", "none", "neighbor", 10, 0, False),
        ]
        rows.extend(
            ("a", "s", "n", f"b_{index:02d}", "none", "neighbor", 1000 - index, 0, False)
            for index in range(1, 16)
        )
        max_rank = 16
    result = build_bridge_neighbors(pd.DataFrame(rows, columns=INPUT_COLUMNS), max_rank=max_rank)

    assert "shared" not in set(result["neighbor_cell_id"])
    freeze_bridge_neighbour_table(result.to_dict("records"))


@settings(max_examples=4, deadline=None)
@given(st.integers(min_value=0, max_value=3))
def test_generated_row_caps_precede_materialization(
    cap: int,
) -> None:
    frame = pd.DataFrame({"unused": range(4)})

    def forbidden_extract(*args: object, **kwargs: object) -> object:
        raise AssertionError("input was extracted before its row cap check")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(neighbors_module, "_MAX_INPUT_ROWS", cap)
        patcher.setattr(neighbors_module, "_validate_and_extract", forbidden_extract)
        with pytest.raises(ValueError, match="row|limit|large"):
            build_bridge_neighbors(frame)


@settings(max_examples=4, deadline=None)
@given(st.integers(min_value=1, max_value=3))
def test_generated_output_caps_precede_distance_allocation(
    cap: int,
) -> None:
    rows: list[tuple[object, ...]] = [
        ("a", "s", "b", "source", "guide", "source", 0, 0, True)
    ]
    rows.extend(
        ("a", "s", "b", f"n_{index}", "none", "neighbor", index, 0, False)
        for index in range(1, 6)
    )
    def fail_if_called(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("distance vector was allocated before the cap")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(neighbors_module, "_MAX_OUTPUT_ROWS", cap)
        patcher.setattr(neighbors_module, "_squared_distances", fail_if_called)
        with pytest.raises(ValueError, match="output|relation|limit|large"):
            build_bridge_neighbors(pd.DataFrame(rows, columns=INPUT_COLUMNS), max_rank=5)


@settings(max_examples=4, deadline=None)
@given(st.integers(min_value=2_500, max_value=3_500))
def test_generated_large_sections_have_no_quadratic_distance_matrix(cell_count: int) -> None:
    rows: list[tuple[object, ...]] = [
        ("a", "s", "b", "source", "guide", "source", 0, 0, True)
    ]
    rows.extend(
        ("a", "s", "b", f"n_{index:05d}", "none", "neighbor", index, 0, False)
        for index in range(1, cell_count + 1)
    )
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    tracemalloc.start()
    try:
        result = build_bridge_neighbors(frame)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(result) == 60
    assert peak < 24_000_000


@settings(max_examples=8, deadline=None)
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_generated_builds_leave_local_and_global_rng_states_unchanged(seed: int) -> None:
    frame = pd.DataFrame(
        [
            ("a", "s", "b", "source", "guide", "source", 0, 0, True),
            ("a", "s", "b", "neighbor", "none", "neighbor", 1, 0, False),
        ],
        columns=INPUT_COLUMNS,
    )
    local_python = random.Random(seed)
    local_numpy = np.random.default_rng(seed)
    local_python_before = local_python.getstate()
    local_numpy_before = local_numpy.bit_generator.state
    random.seed(seed)
    np.random.seed(seed)
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    build_bridge_neighbors(frame)

    assert local_python.getstate() == local_python_before
    assert local_numpy.bit_generator.state == local_numpy_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
