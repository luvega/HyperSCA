from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.evaluation.task_c_null_controls import (
    EmpiricalNullCheck,
    TaskCNullControlError,
    build_control_resampling_null,
    empirical_null_check,
    permute_intervention_labels,
)


def _valid_labels() -> np.ndarray:
    return np.asarray(["non-targeting"] * 10 + ["A"] * 5 + ["B"] * 5)


def _is_bytes_backed(values: np.ndarray) -> bool:
    owner: object = values
    for _ in range(4):
        owner = getattr(owner, "base", None)
        if isinstance(owner, bytes):
            return True
        if owner is None:
            return False
    return False


def _assert_frozen_array(values: np.ndarray) -> None:
    assert not values.flags.writeable
    assert not values.flags.owndata
    assert _is_bytes_backed(values)
    with pytest.raises(ValueError):
        values.setflags(write=True)


def _random_state() -> tuple[object, ...]:
    state = np.random.get_state()
    return (state[0], state[1].copy(), *state[2:])


def _assert_random_state_equal(
    before: tuple[object, ...], after: tuple[object, ...]
) -> None:
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_permutation_keeps_group_sizes_and_returns_a_real_null() -> None:
    labels = _valid_labels()
    before = labels.copy()
    random_before = _random_state()

    first = permute_intervention_labels(labels, seed=11)
    second = permute_intervention_labels(labels.tolist(), seed=11)

    assert first.tolist() == second.tolist()
    assert {
        label: int(np.count_nonzero(first == label)) for label in set(labels)
    } == {
        label: int(np.count_nonzero(labels == label)) for label in set(labels)
    }
    assert not np.array_equal(labels, first)
    _assert_frozen_array(first)
    np.testing.assert_array_equal(labels, before)
    _assert_random_state_equal(random_before, _random_state())


def test_control_resampling_uses_only_control_rows_and_keeps_fake_group_sizes() -> None:
    expression = np.arange(60, dtype=float).reshape(20, 3)
    labels = _valid_labels()
    expression_before = expression.copy()
    labels_before = labels.copy()
    control_rows = {tuple(row) for row in expression[:10]}
    random_before = _random_state()

    null_expression, null_labels = build_control_resampling_null(
        expression, labels, seed=11
    )

    assert null_expression.shape == expression.shape
    assert all(tuple(row) in control_rows for row in null_expression)
    assert null_labels.tolist() == labels.tolist()
    assert {
        label: int(np.count_nonzero(null_labels == label)) for label in set(labels)
    } == {"A": 5, "B": 5, "non-targeting": 10}
    assert not np.shares_memory(null_expression, expression)
    assert not np.shares_memory(null_labels, labels)
    _assert_frozen_array(null_expression)
    _assert_frozen_array(null_labels)
    np.testing.assert_array_equal(expression, expression_before)
    np.testing.assert_array_equal(labels, labels_before)
    _assert_random_state_equal(random_before, _random_state())


def test_control_resampling_is_reproducible_and_seed_sensitive() -> None:
    expression = np.arange(60, dtype=float).reshape(20, 3)
    labels = _valid_labels()

    first, _ = build_control_resampling_null(expression, labels, seed=11)
    repeated, _ = build_control_resampling_null(expression, labels, seed=11)
    changed, _ = build_control_resampling_null(expression, labels, seed=12)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (["non-targeting"] * 5 + ["A"] * 4, "at least five"),
        (["non-targeting"] * 10, "intervention group"),
        (["control"] * 5 + ["A"] * 5, "control label"),
        (["non-targeting"] * 5 + [""] * 5, "safe"),
        (["non-targeting"] * 5 + [" A"] * 5, "safe"),
        (["non-targeting"] * 5 + ["A\n"] * 5, "safe"),
        (["non-targeting"] * 5 + ["e\u0301"] * 5, "NFC"),
        (["non-targeting"] * 5 + ["\ud800"] * 5, "UTF-8"),
        (["non-targeting"] * 5 + [1] * 5, "only text"),
    ],
)
def test_label_permutation_rejects_invalid_rehearsal_groups(
    labels: list[object], message: str
) -> None:
    with pytest.raises(TaskCNullControlError, match=message):
        permute_intervention_labels(labels, seed=11)  # type: ignore[arg-type]


def test_null_controls_limit_group_and_cell_counts_before_copying() -> None:
    too_many_groups = [
        label
        for group in range(1_003)
        for label in ["non-targeting" if group == 0 else f"G{group}"] * 5
    ]
    with pytest.raises(TaskCNullControlError, match="distinct label groups"):
        permute_intervention_labels(too_many_groups, seed=11)

    class OversizedLabels(Sequence[str]):
        def __len__(self) -> int:
            return 1_000_001

        def __getitem__(self, index: int) -> str:
            raise AssertionError("oversized labels must be rejected before copying")

    with pytest.raises(TaskCNullControlError, match="too many cells"):
        permute_intervention_labels(OversizedLabels(), seed=11)


@pytest.mark.parametrize("seed", [True, -1, 1.5, np.int64(11)])
def test_null_controls_require_an_exact_nonnegative_integer_seed(seed: object) -> None:
    with pytest.raises(TaskCNullControlError, match="seed"):
        permute_intervention_labels(_valid_labels(), seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (np.ones(20), "two-dimensional"),
        (np.ones((19, 2)), "same number of cells"),
        (np.ones((20, 0)), "at least one gene"),
        (np.ones((20, 1_001)), "too many genes"),
        (np.full((20, 2), np.nan), "finite"),
        (np.ones((20, 2), dtype=bool), "numeric"),
        (np.ones((20, 2), dtype=complex), "numeric"),
        (np.asarray([["x", "y"]] * 20), "numeric"),
    ],
)
def test_control_resampling_rejects_invalid_expression(
    expression: np.ndarray, message: str
) -> None:
    with pytest.raises(TaskCNullControlError, match=message):
        build_control_resampling_null(expression, _valid_labels(), seed=11)


def test_custom_control_label_must_be_safe_and_present() -> None:
    labels = np.asarray(["vehicle"] * 10 + ["A"] * 5 + ["B"] * 5)
    values, returned = build_control_resampling_null(
        np.arange(40, dtype=float).reshape(20, 2),
        labels,
        seed=11,
        control_label="vehicle",
    )
    assert values.shape == (20, 2)
    assert returned.tolist() == labels.tolist()

    with pytest.raises(TaskCNullControlError, match="control label"):
        build_control_resampling_null(
            np.ones((20, 2)), labels, seed=11, control_label="missing"
        )
    with pytest.raises(TaskCNullControlError, match="safe"):
        build_control_resampling_null(
            np.ones((20, 2)), labels, seed=11, control_label=" vehicle"
        )


def test_empirical_null_check_requires_real_metric_to_exceed_all_twenty_nulls() -> None:
    passed = empirical_null_check(
        0.5,
        np.linspace(0.1, 0.4, 20),
        maximum_p_value=0.05,
        minimum_empirical_advantage=0.0,
    )
    failed = empirical_null_check(
        0.3, np.linspace(0.1, 0.4, 20), maximum_p_value=0.05
    )

    assert isinstance(passed, Mapping)
    assert isinstance(passed, EmpiricalNullCheck)
    assert passed["passed"] is True
    assert passed["empirical_p_value"] == 1.0 / 21.0
    assert passed["empirical_advantage"] == pytest.approx(0.1)
    assert passed["repeat_count"] == 20
    assert passed["evidence_scope"] == "workflow_rehearsal_only"
    assert passed["promotion_eligible"] is False
    assert failed["passed"] is False
    with pytest.raises(TypeError):
        passed["passed"] = False  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        passed.passed = False  # type: ignore[misc]
    assert all(
        np.isfinite(value)
        for key, value in passed.items()
        if key
        in {
            "real_metric",
            "null_median",
            "null_maximum",
            "empirical_advantage",
            "empirical_p_value",
            "minimum_empirical_advantage",
            "maximum_empirical_p_value",
        }
    )


def test_empirical_null_check_counts_ties_conservatively() -> None:
    result = empirical_null_check(
        0.4,
        [0.4] + [0.1] * 19,
        maximum_p_value=0.05,
    )
    assert result["empirical_p_value"] == 2.0 / 21.0
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("real_metric", "null_metrics", "maximum", "minimum", "message"),
    [
        (np.nan, [0.1] * 20, 0.05, 0.0, "real metric"),
        (True, [0.1] * 20, 0.05, 0.0, "real metric"),
        (0.5, [0.1] * 19, 0.05, 0.0, "exactly twenty"),
        (0.5, [0.1] * 19 + [np.inf], 0.05, 0.0, "finite"),
        (0.5, [False] * 20, 0.05, 0.0, "numeric"),
        (0.5, [[0.1] * 20], 0.05, 0.0, "one-dimensional"),
        (0.5, [0.1] * 20, True, 0.0, "maximum_p_value"),
        (0.5, [0.1] * 20, 0.1, 0.0, "remain 0.05"),
        (0.5, [0.1] * 20, 0.05, True, "minimum_empirical_advantage"),
        (0.5, [0.1] * 20, 0.05, -0.1, "remain 0.0"),
    ],
)
def test_empirical_null_check_rejects_changed_or_nonfinite_rules(
    real_metric: object,
    null_metrics: object,
    maximum: object,
    minimum: object,
    message: str,
) -> None:
    with pytest.raises(TaskCNullControlError, match=message):
        empirical_null_check(
            real_metric,  # type: ignore[arg-type]
            null_metrics,  # type: ignore[arg-type]
            maximum_p_value=maximum,  # type: ignore[arg-type]
            minimum_empirical_advantage=minimum,  # type: ignore[arg-type]
        )


def test_empirical_null_check_rejects_nonfinite_derived_advantage() -> None:
    with pytest.raises(TaskCNullControlError, match="derived advantage"):
        empirical_null_check(
            1e308,
            [-1e308] * 20,
            maximum_p_value=0.05,
        )
