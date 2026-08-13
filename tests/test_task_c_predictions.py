from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
import pytest

from src.evaluation.task_c_predictions import (
    MAXIMUM_TASK_C_GENES,
    TaskCPredictionError,
    normalize_task_c_predictions,
)


def test_sparse_output_is_completed_in_fixed_gene_order_with_stable_types() -> None:
    raw = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "score": [0.8, 0.0],
        }
    )

    completed = normalize_task_c_predictions(raw, ["B", "A", "C"])

    assert list(completed.columns) == [
        "source",
        "target",
        "score",
        "returned_by_method",
    ]
    assert list(zip(completed["source"], completed["target"], strict=True)) == [
        ("A", "B"),
        ("B", "A"),
        ("B", "C"),
        ("A", "C"),
        ("C", "B"),
        ("C", "A"),
    ]
    assert completed["score"].tolist() == pytest.approx([0.8, 0, 0, 0, 0, 0])
    assert completed["returned_by_method"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert str(completed["source"].dtype) == "string"
    assert str(completed["target"].dtype) == "string"
    assert completed["score"].dtype == np.dtype("float64")
    assert completed["returned_by_method"].dtype == np.dtype("bool")


def test_duplicate_edges_keep_the_highest_score_without_changing_input() -> None:
    raw = pd.DataFrame(
        {
            "target": ["B", "B", "A"],
            "score": [0.2, 0.7, 0.1],
            "source": ["A", "A", "B"],
        }
    )
    before = raw.copy(deep=True)

    completed = normalize_task_c_predictions(raw, ("A", "B"))

    pd.testing.assert_frame_equal(raw, before)
    assert completed["score"].tolist() == pytest.approx([0.7, 0.1])
    assert completed["returned_by_method"].tolist() == [True, True]


def test_empty_method_output_marks_every_completed_edge_as_not_returned() -> None:
    raw = pd.DataFrame(columns=["source", "target", "score"])

    completed = normalize_task_c_predictions(raw, ["A", "B"])

    assert completed["score"].tolist() == [0.0, 0.0]
    assert completed["returned_by_method"].tolist() == [False, False]


@pytest.mark.parametrize(
    "gene_names",
    [
        [],
        ["A", "A"],
        ["A", 2],
        ["A", ""],
        ["A", " B"],
        ["A", "B "],
        ["A", unicodedata.normalize("NFD", "É")],
    ],
)
def test_gene_names_must_be_unique_canonical_strings(gene_names: list[object]) -> None:
    raw = pd.DataFrame(columns=["source", "target", "score"])

    with pytest.raises(TaskCPredictionError):
        normalize_task_c_predictions(raw, gene_names)  # type: ignore[arg-type]


def test_gene_limit_is_checked_before_quadratic_completion() -> None:
    raw = pd.DataFrame(columns=["source", "target", "score"])
    gene_names = [f"G{index}" for index in range(MAXIMUM_TASK_C_GENES + 1)]

    with pytest.raises(TaskCPredictionError, match="at most"):
        normalize_task_c_predictions(raw, gene_names)


def test_raw_row_limit_fails_before_any_prediction_column_is_accessed() -> None:
    class ColumnAccessProbe(pd.DataFrame):
        def __getitem__(self, key: object) -> object:
            raise AssertionError(f"column was accessed before row limit check: {key}")

    raw = ColumnAccessProbe(
        index=range(1_001),
        columns=["source", "target", "score"],
    )

    with pytest.raises(TaskCPredictionError, match="at most 1000 rows"):
        normalize_task_c_predictions(raw, ["A", "B"])


def test_top_1000_output_is_allowed_for_a_small_gene_set() -> None:
    raw = pd.DataFrame(
        {
            "source": ["A"] * 1_000,
            "target": ["B"] * 1_000,
            "score": [0.1] * 1_000,
        }
    )

    completed = normalize_task_c_predictions(raw, ["A", "B"])

    assert completed["score"].tolist() == pytest.approx([0.1, 0.0])
    assert completed["returned_by_method"].tolist() == [True, False]


@pytest.mark.parametrize(
    "raw",
    [
        pd.DataFrame({"source": ["A"], "target": ["B"]}),
        pd.DataFrame(
            {"source": ["A"], "target": ["B"], "score": [1.0], "rank": [1]}
        ),
        pd.DataFrame(
            [["A", "B", 1.0, 0.5]],
            columns=["source", "target", "score", "score"],
        ),
    ],
)
def test_prediction_table_requires_exactly_the_three_contract_columns(
    raw: pd.DataFrame,
) -> None:
    with pytest.raises(TaskCPredictionError, match="exactly"):
        normalize_task_c_predictions(raw, ["A", "B"])


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (None, "B"),
        ("A", np.nan),
        (1, "B"),
        ("A", 2),
        ("", "B"),
        (" A", "B"),
        ("A", "B "),
        (unicodedata.normalize("NFD", "É"), "B"),
        ("A", "Z"),
    ],
)
def test_relation_endpoints_must_be_known_canonical_strings(
    source: object,
    target: object,
) -> None:
    raw = pd.DataFrame({"source": [source], "target": [target], "score": [1.0]})

    with pytest.raises(TaskCPredictionError):
        normalize_task_c_predictions(raw, ["A", "B", "É"])


def test_self_relations_are_excluded_without_losing_valid_relations() -> None:
    raw = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["A", "B"],
            "score": [0.9, 0.4],
        }
    )

    completed = normalize_task_c_predictions(raw, ["A", "B"])

    assert list(zip(completed["source"], completed["target"], strict=True)) == [
        ("A", "B"),
        ("B", "A"),
    ]
    assert completed["score"].tolist() == pytest.approx([0.4, 0.0])
    assert completed["returned_by_method"].tolist() == [True, False]


@pytest.mark.parametrize(
    "score",
    [True, np.bool_(False), np.nan, np.inf, -np.inf, -0.1, "0.7"],
)
def test_scores_must_be_real_finite_non_negative_numbers(score: object) -> None:
    raw = pd.DataFrame({"source": ["A"], "target": ["B"], "score": [score]})

    with pytest.raises(TaskCPredictionError, match="scores"):
        normalize_task_c_predictions(raw, ["A", "B"])


def test_negative_zero_is_returned_but_all_output_zeros_are_positive() -> None:
    raw = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "score": [-0.0]}
    )

    completed = normalize_task_c_predictions(raw, ["A", "B"])

    assert completed["score"].tolist() == [0.0, 0.0]
    assert not np.signbit(completed["score"].to_numpy()).any()
    assert completed["returned_by_method"].tolist() == [True, False]


def test_one_gene_has_a_well_typed_empty_relation_scope() -> None:
    raw = pd.DataFrame(columns=["source", "target", "score"])

    completed = normalize_task_c_predictions(raw, ["A"])

    assert completed.empty
    assert str(completed["source"].dtype) == "string"
    assert str(completed["target"].dtype) == "string"
    assert completed["score"].dtype == np.dtype("float64")
    assert completed["returned_by_method"].dtype == np.dtype("bool")
