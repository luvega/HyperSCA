from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext, HyperSCACError
from src.causal import hypersca_c_stability as stability_module
from src.causal.hypersca_c_stability import (
    HyperSCAStabilityResult,
    build_stability_table,
    fit_stable_hypersca_c,
    stratified_bootstrap_context,
)


ROOT = Path(__file__).resolve().parents[1]


class DuplicateItems(Mapping[str, object]):
    def __init__(self, pairs: tuple[tuple[str, object], ...]) -> None:
        self._pairs = pairs
        self._keys = tuple(dict.fromkeys(key for key, _ in pairs))

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._pairs:
            if candidate == key:
                return value
        raise KeyError(key)

    def items(self) -> object:
        return self._pairs


def default_config(**changes: object) -> HyperSCACConfig:
    payload = json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )
    payload.update(changes)
    return HyperSCACConfig.from_mapping(payload)


def make_context(context_id: str = "k562") -> HyperSCACContext:
    return HyperSCACContext(
        context_id=context_id,
        expression=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 0.0],
                [3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0],
                [5.0, 4.0, 3.0],
            ],
            dtype=np.float32,
        ),
        interventions=np.asarray(
            ["non-targeting", "non-targeting", "A", "A", "B", "B"]
        ),
        gene_names=("A", "B", "C"),
    )


def valid_table() -> tuple[pd.DataFrame, dict[str, object]]:
    return build_stability_table(
        [{"k562": np.asarray([[0.0, 2.0], [0.0, 0.0]])}],
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=1,
        minimum_success_fraction=1.0,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )


def test_stability_score_uses_frozen_product_formula() -> None:
    genes = ("A", "B")
    matrices = [
        {"k562": np.asarray([[0.0, 2.0], [0.0, 0.0]])},
        {"k562": np.asarray([[0.0, 4.0], [0.0, 0.0]])},
        {"k562": np.asarray([[0.0, -3.0], [0.0, 0.0]])},
    ]
    table, summary = build_stability_table(
        matrices,
        genes,
        selection_threshold=0.1,
        requested_repeats=3,
        minimum_success_fraction=0.8,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    ab = table[(table["source"] == "A") & (table["target"] == "B")].iloc[0]
    assert ab["median_effect"] == pytest.approx(2.0)
    assert ab["selection_frequency"] == pytest.approx(1.0)
    assert ab["direction_agreement"] == pytest.approx(2.0 / 3.0)
    assert ab["context_consistency"] == pytest.approx(1.0)
    assert ab["score"] == pytest.approx(4.0 / 3.0)
    assert len(table) == 2
    assert summary["successful_repeats"] == 3


def test_insufficient_success_marks_every_source_as_abstained() -> None:
    table, summary = build_stability_table(
        [{"k562": np.zeros((2, 2))}],
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    assert table["abstained"].all()
    assert summary["coverage"] == 0.0


def test_bootstrap_preserves_intervention_group_sizes() -> None:
    context = HyperSCACContext(
        context_id="k562",
        expression=np.arange(18, dtype=np.float32).reshape(6, 3),
        interventions=np.asarray(
            ["non-targeting", "non-targeting", "A", "A", "B", "B"]
        ),
        gene_names=("A", "B", "C"),
    )
    sampled = stratified_bootstrap_context(context, np.random.default_rng(11))
    labels, counts = np.unique(sampled.interventions, return_counts=True)
    assert dict(zip(labels, counts)) == {"A": 2, "B": 2, "non-targeting": 2}


def test_bootstrap_is_deterministic_group_local_and_does_not_alias_input() -> None:
    context = HyperSCACContext(
        context_id="环境甲",
        expression=np.arange(16, dtype=np.float32).reshape(8, 2),
        interventions=np.asarray(["β", "β", "对照", "对照", "对照", "α", "α", "α"]),
        gene_names=("基因甲", "基因乙"),
    )
    before_expression = context.expression.copy()
    before_interventions = context.interventions.copy()

    first = stratified_bootstrap_context(context, np.random.default_rng(47))
    second = stratified_bootstrap_context(context, np.random.default_rng(47))

    assert first.context_id == context.context_id
    assert first.gene_names == context.gene_names
    assert np.array_equal(first.expression, second.expression)
    assert np.array_equal(first.interventions, second.interventions)
    assert first.interventions.tolist()[:2] == ["β", "β"]
    assert first.interventions.tolist()[2:5] == ["对照", "对照", "对照"]
    assert first.interventions.tolist()[5:] == ["α", "α", "α"]
    assert not np.shares_memory(first.expression, context.expression)
    assert np.array_equal(context.expression, before_expression)
    assert np.array_equal(context.interventions, before_interventions)


@pytest.mark.parametrize(
    ("context", "rng", "match"),
    [
        (object(), np.random.default_rng(1), "context"),
        (make_context(), object(), "rng"),
    ],
)
def test_bootstrap_rejects_wrong_input_types(
    context: object, rng: object, match: str
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        stratified_bootstrap_context(context, rng)  # type: ignore[arg-type]


def test_stability_table_flattens_equal_context_repeat_units() -> None:
    matrices = [
        {
            "k562": np.asarray([[0.0, 2.0], [0.0, 0.0]]),
            "rpe1": np.asarray([[0.0, -4.0], [0.0, 0.0]]),
        },
        {
            "k562": np.asarray([[0.0, 4.0], [0.0, 0.0]]),
            "rpe1": np.asarray([[0.0, -2.0], [0.0, 0.0]]),
        },
    ]
    table, _ = build_stability_table(
        matrices,
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=1.0,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    edge = table.query("source == 'A' and target == 'B'").iloc[0]
    assert edge["median_effect"] == pytest.approx(0.0)
    assert edge["selection_frequency"] == pytest.approx(1.0)
    assert edge["direction_agreement"] == pytest.approx(0.5)
    assert edge["effect_k562"] == pytest.approx(3.0)
    assert edge["effect_rpe1"] == pytest.approx(-3.0)
    assert edge["context_consistency"] == pytest.approx(0.5)
    assert edge["score"] == pytest.approx(0.0)


def test_zero_values_are_not_selected_or_assigned_a_direction() -> None:
    table, _ = build_stability_table(
        [{"k562": np.zeros((2, 2))}],
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=1,
        minimum_success_fraction=1.0,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    assert (table["selection_frequency"] == 0.0).all()
    assert (table["direction_agreement"] == 0.0).all()
    assert (table["context_consistency"] == 0.0).all()
    assert (table["direction"] == 0).all()


def test_stability_table_is_complete_fixed_and_ranks_abstentions_last() -> None:
    table, summary = build_stability_table(
        [{"k562": np.asarray([[0.0, 9.0], [0.0, 0.0]])}],
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=1,
        minimum_success_fraction=1.0,
        source_variance={"A": 0.0, "B": 1.0},
        minimum_source_variance=1e-8,
    )
    assert table.columns.tolist() == [
        "source",
        "target",
        "effect",
        "median_effect",
        "direction",
        "selection_frequency",
        "direction_agreement",
        "context_consistency",
        "effect_k562",
        "score",
        "abstained",
        "abstention_reason",
    ]
    assert table[["source", "target"]].values.tolist() == [["B", "A"], ["A", "B"]]
    assert table["score"].tolist() == [0.0, 9.0]
    assert table["abstained"].tolist() == [False, True]
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["abstention_rate"] == pytest.approx(0.5)


def test_zero_successes_still_produce_every_directed_relationship() -> None:
    table, summary = build_stability_table(
        [],
        ("A", "B", "C"),
        selection_threshold=0.1,
        requested_repeats=3,
        minimum_success_fraction=0.8,
        source_variance={"A": 1.0, "B": 1.0, "C": 1.0},
        minimum_source_variance=1e-8,
    )
    assert len(table) == 6
    assert table["abstained"].all()
    assert set(table["abstention_reason"]) == {
        "insufficient_successful_bootstraps"
    }
    assert summary["successful_repeats"] == 0
    assert summary["repeat_success_fraction"] == 0.0


def test_zero_successes_preserve_declared_context_columns() -> None:
    table, _ = build_stability_table(
        [],
        ("A", "B"),
        selection_threshold=0.1,
        requested_repeats=2,
        minimum_success_fraction=0.8,
        source_variance={"A": 1.0, "B": 1.0},
        minimum_source_variance=1e-8,
        expected_contexts=("k562", "rpe1"),
    )
    assert table[["effect_k562", "effect_rpe1"]].to_numpy().tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_successful_repeats_must_match_declared_contexts() -> None:
    with pytest.raises(HyperSCACError, match="expected_contexts"):
        build_stability_table(
            [{"k562": np.zeros((2, 2))}],
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance={"A": 1.0, "B": 1.0},
            minimum_source_variance=1e-8,
            expected_contexts=("rpe1",),
        )


@pytest.mark.parametrize(
    ("expected_contexts", "match"),
    [
        ((), "at least one"),
        (("k562", "k562"), "unique"),
        (("bad context",), "whitespace"),
        ("k562", "sequence"),
    ],
)
def test_declared_contexts_must_be_nonempty_unique_text(
    expected_contexts: object, match: str
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        build_stability_table(
            [],
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance={"A": 1.0, "B": 1.0},
            minimum_source_variance=1e-8,
            expected_contexts=expected_contexts,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("gene_names", "match"),
    [
        (("A",), "two genes"),
        (("A", "A"), "unique"),
        (("A", "gene B"), "whitespace"),
        (("A", 1), "text"),
    ],
)
def test_stability_table_rejects_invalid_gene_names(
    gene_names: object, match: str
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        build_stability_table(
            [],
            gene_names,  # type: ignore[arg-type]
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance={"A": 1.0, "B": 1.0},
            minimum_source_variance=1e-8,
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"selection_threshold": 0.0}, "selection_threshold"),
        ({"selection_threshold": float("nan")}, "selection_threshold"),
        ({"requested_repeats": True}, "requested_repeats"),
        ({"requested_repeats": 0}, "requested_repeats"),
        ({"minimum_success_fraction": 0.0}, "minimum_success_fraction"),
        ({"minimum_success_fraction": float("inf")}, "minimum_success_fraction"),
        ({"minimum_source_variance": 0.0}, "minimum_source_variance"),
    ],
)
def test_stability_table_rejects_invalid_thresholds_and_counts(
    changes: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "selection_threshold": 0.1,
        "requested_repeats": 1,
        "minimum_success_fraction": 0.8,
        "source_variance": {"A": 1.0, "B": 1.0},
        "minimum_source_variance": 1e-8,
    }
    arguments.update(changes)
    with pytest.raises(HyperSCACError, match=match):
        build_stability_table([], ("A", "B"), **arguments)  # type: ignore[arg-type]


def test_stability_table_rejects_more_successes_than_requested() -> None:
    matrix = {"k562": np.zeros((2, 2))}
    with pytest.raises(HyperSCACError, match="successful"):
        build_stability_table(
            [matrix, matrix],
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance={"A": 1.0, "B": 1.0},
            minimum_source_variance=1e-8,
        )


@pytest.mark.parametrize(
    ("source_variance", "match"),
    [
        ({"A": 1.0}, "exactly"),
        ({"A": 1.0, "B": 1.0, "C": 1.0}, "exactly"),
        ({"A": -1.0, "B": 1.0}, "non-negative"),
        ({"A": float("nan"), "B": 1.0}, "finite"),
    ],
)
def test_stability_table_requires_exact_finite_source_variance(
    source_variance: object, match: str
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        build_stability_table(
            [],
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance=source_variance,  # type: ignore[arg-type]
            minimum_source_variance=1e-8,
        )


def test_stability_table_rejects_duplicate_source_variance_items() -> None:
    class DuplicateVariance(Mapping[str, float]):
        def __iter__(self) -> Iterator[str]:
            return iter(("A", "B"))

        def __len__(self) -> int:
            return 2

        def __getitem__(self, key: str) -> float:
            return 1.0

        def items(self) -> object:
            return (("A", 0.0), ("A", 1.0), ("B", 1.0))

    with pytest.raises(HyperSCACError, match="duplicate"):
        build_stability_table(
            [],
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance=DuplicateVariance(),
            minimum_source_variance=1e-8,
        )


@pytest.mark.parametrize(
    ("matrices", "match"),
    [
        ([{}], "at least one context"),
        ([{"k562": np.zeros((2, 2))}, {"rpe1": np.zeros((2, 2))}], "same contexts"),
        ([[np.zeros((2, 2))]], "mapping"),
        ([{"k562": np.zeros((2, 3))}], "shape"),
        ([{"k562": np.asarray([[0.0, np.nan], [0.0, 0.0]])}], "finite"),
        ([{"k562": np.eye(2)}], "diagonal"),
        ([{"bad context": np.zeros((2, 2))}], "whitespace"),
    ],
)
def test_stability_table_rejects_incomparable_repeat_matrices(
    matrices: object, match: str
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        build_stability_table(
            matrices,  # type: ignore[arg-type]
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=2,
            minimum_success_fraction=0.8,
            source_variance={"A": 1.0, "B": 1.0},
            minimum_source_variance=1e-8,
        )


def test_stability_table_rejects_duplicate_context_items() -> None:
    class DuplicateContexts(Mapping[str, np.ndarray]):
        def __iter__(self) -> Iterator[str]:
            return iter(("k562",))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> np.ndarray:
            return np.zeros((2, 2))

        def items(self) -> object:
            return (
                ("k562", np.zeros((2, 2))),
                ("k562", np.zeros((2, 2))),
            )

    with pytest.raises(HyperSCACError, match="duplicate"):
        build_stability_table(
            [DuplicateContexts()],
            ("A", "B"),
            selection_threshold=0.1,
            requested_repeats=1,
            minimum_success_fraction=0.8,
            source_variance={"A": 1.0, "B": 1.0},
            minimum_source_variance=1e-8,
        )


def test_result_owns_predictions_and_deep_freezes_json_safe_summary() -> None:
    table, summary = valid_table()
    summary["details"] = {"labels": ["stable", "screened"]}
    result = HyperSCAStabilityResult(table, summary, ())

    table.loc[0, "score"] = 999.0
    summary["details"]["labels"].append("changed")  # type: ignore[index,union-attr]
    assert result.predictions.loc[0, "score"] != 999.0
    assert result.summary["details"]["labels"] == ("stable", "screened")  # type: ignore[index]
    thawed = stability_module.thaw_json_record(result.summary)
    assert json.loads(json.dumps(thawed, ensure_ascii=False))["details"] == {
        "labels": ["stable", "screened"]
    }
    thawed["details"]["labels"].append("local change")  # type: ignore[index,union-attr]
    assert result.summary["details"]["labels"] == ("stable", "screened")  # type: ignore[index]
    with pytest.raises(TypeError):
        result.summary["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.summary["details"]["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        dict.__setitem__(result.summary, "bypass", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dict.update(result.summary, {"bypass": True})  # type: ignore[arg-type]
    result.predictions.loc[0, "score"] = 5.0
    assert result.predictions.loc[0, "score"] == 5.0


def test_result_rejects_duplicate_top_level_summary_items() -> None:
    table, summary = valid_table()
    duplicate = DuplicateItems(
        (*tuple(summary.items()), ("coverage", summary["coverage"]))
    )
    with pytest.raises(HyperSCACError, match="duplicate"):
        HyperSCAStabilityResult(table, duplicate, ())


def test_result_rejects_duplicate_nested_summary_items() -> None:
    table, summary = valid_table()
    summary["details"] = DuplicateItems(
        (("labels", ["stable"]), ("labels", ["changed"]))
    )
    with pytest.raises(HyperSCACError, match="duplicate"):
        HyperSCAStabilityResult(table, summary, ())


@pytest.mark.parametrize(
    "unsafe",
    ({"bad": float("nan")}, {"bad": ["mutable"]}, {1: "non-text key"}),
)
def test_thaw_json_record_rejects_values_outside_frozen_json_semantics(
    unsafe: object,
) -> None:
    with pytest.raises(HyperSCACError, match="frozen JSON"):
        stability_module.thaw_json_record(unsafe)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        ("missing", "required columns"),
        ("duplicate", "unique"),
        ("self", "self"),
        ("infinite", "finite"),
        ("non_bool", "bool"),
    ],
)
def test_result_rejects_invalid_prediction_tables(edit: str, match: str) -> None:
    table, summary = valid_table()
    if edit == "missing":
        table = table.drop(columns="score")
    elif edit == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif edit == "self":
        table.loc[0, "target"] = table.loc[0, "source"]
    elif edit == "infinite":
        table.loc[0, "score"] = np.inf
    else:
        table["abstained"] = table["abstained"].astype(int)
    with pytest.raises(HyperSCACError, match=match):
        HyperSCAStabilityResult(table, summary, ())


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("direction", -1, "direction.*sign"),
        ("score", 123.0, "score.*formula"),
    ],
)
def test_result_recomputes_frozen_row_contracts(
    column: str, value: object, match: str
) -> None:
    table, summary = valid_table()
    table.loc[0, column] = value
    with pytest.raises(HyperSCACError, match=match):
        HyperSCAStabilityResult(table, summary, ())


def test_result_requires_every_nonself_relationship() -> None:
    table, summary = build_stability_table(
        [{"k562": np.zeros((3, 3))}],
        ("A", "B", "C"),
        selection_threshold=0.1,
        requested_repeats=1,
        minimum_success_fraction=1.0,
        source_variance={"A": 1.0, "B": 1.0, "C": 1.0},
        minimum_source_variance=1e-8,
    )
    incomplete = table[
        ~((table["source"] == "A") & (table["target"] == "B"))
    ]
    with pytest.raises(HyperSCACError, match="complete"):
        HyperSCAStabilityResult(incomplete, summary, ())


def test_result_requires_equal_source_and_target_gene_sets() -> None:
    table, summary = build_stability_table(
        [{"k562": np.zeros((3, 3))}],
        ("A", "B", "C"),
        selection_threshold=0.1,
        requested_repeats=1,
        minimum_success_fraction=1.0,
        source_variance={"A": 1.0, "B": 1.0, "C": 1.0},
        minimum_source_variance=1e-8,
    )
    unequal = table.loc[table["source"] != "C"]
    with pytest.raises(HyperSCACError, match="source and target gene sets"):
        HyperSCAStabilityResult(unequal, summary, ())


def test_result_rejects_unknown_abstention_reason() -> None:
    table, summary = valid_table()
    table.loc[0, "abstained"] = True
    table.loc[0, "abstention_reason"] = "invented_reason"
    with pytest.raises(HyperSCACError, match="abstention_reason"):
        HyperSCAStabilityResult(table, summary, ())


def test_result_requires_the_frozen_ranking_order() -> None:
    table, summary = valid_table()
    reversed_rows = table.iloc[::-1]
    with pytest.raises(HyperSCACError, match="sorted"):
        HyperSCAStabilityResult(reversed_rows, summary, ())


def test_result_rejects_inconsistent_summary_and_failure_counts() -> None:
    table, summary = valid_table()
    summary["repeat_success_fraction"] = 0.5
    with pytest.raises(HyperSCACError, match="repeat_success_fraction"):
        HyperSCAStabilityResult(table, summary, ())

    _, summary = valid_table()
    summary["coverage"] = 0.0
    summary["abstention_rate"] = 1.0
    with pytest.raises(HyperSCACError, match="coverage"):
        HyperSCAStabilityResult(table, summary, ())

    _, summary = valid_table()
    with pytest.raises(HyperSCACError, match="failure"):
        HyperSCAStabilityResult(table, summary, ("unexpected failure",))


@pytest.mark.parametrize(
    "unsafe_value",
    [{"bad": {"not", "json"}}, {1: "non-text key"}, {"bad": float("nan")}],
)
def test_result_rejects_summary_values_that_are_not_json_safe(
    unsafe_value: object,
) -> None:
    table, summary = valid_table()
    summary["unsafe"] = unsafe_value
    with pytest.raises(HyperSCACError, match="JSON"):
        HyperSCAStabilityResult(table, summary, ())


def test_stable_fit_records_expected_failures_seeds_and_group_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = [make_context("k562"), make_context("rpe1")]
    config = default_config(
        bootstrap_repeats=3,
        bootstrap_success_fraction=0.66,
    )
    calls: list[tuple[int, list[dict[str, int]]]] = []

    def fake_fit(
        sampled_contexts: list[HyperSCACContext],
        supplied_config: HyperSCACConfig,
        *,
        seed: int,
        device: str,
        prior_mask: np.ndarray | None,
    ) -> SimpleNamespace:
        assert supplied_config is config
        assert device == "cpu"
        assert prior_mask is None
        group_sizes = []
        for context in sampled_contexts:
            labels, counts = np.unique(context.interventions, return_counts=True)
            group_sizes.append(dict(zip(labels.tolist(), counts.tolist())))
        calls.append((seed, group_sizes))
        if seed == 31:
            raise HyperSCACError("planned optimizer rejection")
        return SimpleNamespace(
            context_adjacencies={
                context.context_id: np.zeros((3, 3), dtype=np.float32)
                for context in sampled_contexts
            }
        )

    monkeypatch.setattr(stability_module, "fit_hypersca_c_once", fake_fit)
    np.random.seed(123)
    expected_first = np.random.random()
    expected_second = np.random.random()
    np.random.seed(123)
    assert np.random.random() == expected_first

    result = fit_stable_hypersca_c(
        contexts, config, seed=31, device="cpu", prior_mask=None
    )

    assert np.random.random() == expected_second
    assert [seed for seed, _ in calls] == [31, 32, 33]
    assert all(
        group == {"A": 2, "B": 2, "non-targeting": 2}
        for _, repeats in calls
        for group in repeats
    )
    assert result.summary["successful_repeats"] == 2
    assert result.summary["repeat_success_fraction"] == pytest.approx(2.0 / 3.0)
    assert len(result.failures) == 1
    assert result.failures[0].startswith("repeat_0:")
    assert not result.predictions["abstained"].any()


def test_stable_fit_all_expected_failures_trigger_full_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = default_config(bootstrap_repeats=2)

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise HyperSCACError("expected numerical rejection")

    monkeypatch.setattr(stability_module, "fit_hypersca_c_once", fail_fit)
    result = fit_stable_hypersca_c(
        [make_context("k562"), make_context("rpe1")],
        config,
        seed=11,
        device="cpu",
    )
    assert result.predictions["abstained"].all()
    assert (result.predictions[["effect_k562", "effect_rpe1"]] == 0.0).all().all()
    assert result.summary["successful_repeats"] == 0
    assert result.summary["coverage"] == 0.0
    assert len(result.failures) == 2


def test_cpu_stable_fit_uses_one_torch_thread_and_restores_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    config = default_config(bootstrap_repeats=1)
    state = {"threads": 8}
    observed: list[int] = []

    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(
        torch, "set_num_threads", lambda value: state.__setitem__("threads", value)
    )

    def fake_fit(
        sampled_contexts: list[HyperSCACContext], *args: object, **kwargs: object
    ) -> SimpleNamespace:
        observed.append(torch.get_num_threads())
        return SimpleNamespace(
            context_adjacencies={
                sampled_contexts[0].context_id: np.zeros((3, 3), dtype=np.float32)
            }
        )

    monkeypatch.setattr(stability_module, "fit_hypersca_c_once", fake_fit)
    fit_stable_hypersca_c([make_context()], config, seed=11, device="cpu")

    assert observed == [1]
    assert state["threads"] == 8


def test_stable_fit_propagates_unexpected_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = default_config(bootstrap_repeats=1)

    def broken_fit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("programming defect")

    monkeypatch.setattr(stability_module, "fit_hypersca_c_once", broken_fit)
    with pytest.raises(RuntimeError, match="programming defect"):
        fit_stable_hypersca_c([make_context()], config, seed=11, device="cpu")


def test_stable_fit_uses_raw_control_population_variance_for_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = HyperSCACContext(
        context_id="k562",
        expression=np.asarray(
            [[5.0, 0.0, 1.0], [5.0, 2.0, 3.0], [8.0, 3.0, 4.0], [9.0, 4.0, 5.0]],
            dtype=np.float32,
        ),
        interventions=np.asarray(["non-targeting", "non-targeting", "A", "A"]),
        gene_names=("A", "B", "C"),
    )
    config = default_config(bootstrap_repeats=1)

    def fake_fit(
        sampled_contexts: list[HyperSCACContext], *args: object, **kwargs: object
    ) -> SimpleNamespace:
        return SimpleNamespace(
            context_adjacencies={
                sampled_contexts[0].context_id: np.zeros((3, 3), dtype=np.float32)
            }
        )

    monkeypatch.setattr(stability_module, "fit_hypersca_c_once", fake_fit)
    result = fit_stable_hypersca_c([context], config, seed=7, device="cpu")
    by_source = result.predictions.groupby("source", sort=True)["abstained"].first()
    assert by_source.to_dict() == {"A": True, "B": False, "C": False}
    a_reasons = result.predictions.loc[
        result.predictions["source"] == "A", "abstention_reason"
    ]
    assert set(a_reasons) == {"source_has_no_control_variation"}


def test_stable_fit_validates_inputs_before_first_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_fit(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(stability_module, "fit_hypersca_c_once", fake_fit)
    config = default_config(bootstrap_repeats=2)
    with pytest.raises(HyperSCACError, match="context"):
        fit_stable_hypersca_c([], config, seed=1, device="cpu")
    with pytest.raises(HyperSCACError, match="seed"):
        fit_stable_hypersca_c(
            [make_context()], config, seed=2**64 - 1, device="cpu"
        )
    with pytest.raises(HyperSCACError, match="prior"):
        fit_stable_hypersca_c(
            [make_context()],
            config,
            seed=1,
            device="cpu",
            prior_mask=np.zeros((2, 2)),
        )
    with pytest.raises(HyperSCACError, match="selection_threshold"):
        fit_stable_hypersca_c(
            [make_context()],
            default_config(selection_threshold=0.0, bootstrap_repeats=1),
            seed=1,
            device="cpu",
        )
    assert calls == 0
