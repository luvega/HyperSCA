from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.evaluation.task_c_rehearsal import (
    RehearsalProfile,
    TaskCRehearsalConfig,
    TaskCRehearsalError,
    center_and_merge_allowed_contexts,
    choose_rehearsal_cells,
    choose_rehearsal_genes,
    load_task_c_rehearsal_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/task_c_rehearsal_v1.json"

EXPECTED_METHODS = (
    "hypersca_c",
    "mean_difference",
    "random1000",
    "grnboost",
    "pc",
    "notears_linear",
)
EXPECTED_ARTIFACTS = (
    "run_manifest.json",
    "input_summary.json",
    "metrics.json",
    "predictions.csv",
    "promotion_decision.json",
)
EXPECTED_SEEDS = (11, 23, 47, 71, 97)


def _valid_payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_rehearsal_profiles_are_frozen_and_never_promotion_eligible() -> None:
    config = load_task_c_rehearsal_config(CONFIG_PATH)

    assert tuple(config.profiles) == ("connection", "comprehensive")
    assert config.profiles["connection"] == RehearsalProfile(64, 2_000, 1_800)
    assert config.profiles["comprehensive"] == RehearsalProfile(
        256, 20_000, 14_400
    )
    assert config.seed == 11
    assert config.promotion_eligible is False
    assert config.feature_selection == (
        "common_expression_genes_train_control_variance_v1"
    )
    assert config.required_core_methods == EXPECTED_METHODS
    assert config.required_artifacts == EXPECTED_ARTIFACTS
    assert config.full_run_seeds == EXPECTED_SEEDS
    assert config.null_controls == {
        "repeats": 20,
        "minimum_empirical_advantage": 0.0,
        "maximum_empirical_p_value": 0.05,
    }
    assert isinstance(config.profiles, MappingProxyType)
    assert isinstance(config.null_controls, MappingProxyType)
    with pytest.raises(TypeError):
        config.profiles["connection"] = RehearsalProfile(2, 2, 2)  # type: ignore[index]
    with pytest.raises(TypeError):
        config.null_controls["repeats"] = 2  # type: ignore[index]


def test_direct_construction_copies_nested_inputs_and_rejects_relaxation() -> None:
    profiles = {
        "connection": RehearsalProfile(64, 2_000, 1_800),
        "comprehensive": RehearsalProfile(256, 20_000, 14_400),
    }
    original_connection = profiles["connection"]
    null_controls: dict[str, float | int] = {
        "repeats": 20,
        "minimum_empirical_advantage": 0.0,
        "maximum_empirical_p_value": 0.05,
    }
    config = TaskCRehearsalConfig(
        schema_version="1.0",
        seed=11,
        promotion_eligible=False,
        feature_selection="common_expression_genes_train_control_variance_v1",
        profiles=profiles,
        null_controls=null_controls,
        required_core_methods=EXPECTED_METHODS,
        required_interventional_method_count=1,
        required_artifacts=EXPECTED_ARTIFACTS,
        full_run_seeds=EXPECTED_SEEDS,
    )
    profiles.clear()
    null_controls["repeats"] = 1
    assert tuple(config.profiles) == ("connection", "comprehensive")
    assert config.null_controls["repeats"] == 20
    assert not hasattr(config, "__dict__")
    assert not hasattr(config.profiles["connection"], "__dict__")

    object.__setattr__(original_connection, "maximum_genes", 999)
    assert config.profiles["connection"].maximum_genes == 64

    relaxed = dict(config.profiles)
    relaxed["connection"] = RehearsalProfile(65, 2_000, 1_800)
    with pytest.raises(TaskCRehearsalError, match="fixed profile values"):
        TaskCRehearsalConfig(
            schema_version="1.0",
            seed=11,
            promotion_eligible=False,
            feature_selection="common_expression_genes_train_control_variance_v1",
            profiles=relaxed,
            null_controls=config.null_controls,
            required_core_methods=EXPECTED_METHODS,
            required_interventional_method_count=1,
            required_artifacts=EXPECTED_ARTIFACTS,
            full_run_seeds=EXPECTED_SEEDS,
        )


def test_direct_construction_rejects_scalar_subclasses_and_duplicate_mapping_keys() -> None:
    config = load_task_c_rehearsal_config(CONFIG_PATH)

    class DisguisedInteger(int):
        pass

    class DisguisedText(str):
        pass

    with pytest.raises(TaskCRehearsalError, match="maximum_genes"):
        RehearsalProfile(DisguisedInteger(64), 2_000, 1_800)
    with pytest.raises(TaskCRehearsalError, match="schema"):
        replace(config, schema_version=DisguisedText("1.0"))
    with pytest.raises(TaskCRehearsalError, match="gene selection"):
        replace(config, feature_selection=DisguisedText(config.feature_selection))

    class DuplicateProfileMapping(Mapping[str, RehearsalProfile]):
        def __getitem__(self, key: str) -> RehearsalProfile:
            return config.profiles[key]

        def __iter__(self) -> Iterator[str]:
            return iter(("connection", "connection", "comprehensive"))

        def __len__(self) -> int:
            return 3

    with pytest.raises(TaskCRehearsalError, match="connection then comprehensive"):
        replace(config, profiles=DuplicateProfileMapping())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("promotion_eligible", True),
        ("feature_selection", "mean_control_variance_across_contexts"),
        ("required_core_methods", list(reversed(EXPECTED_METHODS))),
        ("required_artifacts", list(reversed(EXPECTED_ARTIFACTS))),
        ("full_run_seeds", [11, 23, 47, 71, 71]),
    ],
)
def test_config_rejects_changed_values_and_order(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _valid_payload()
    payload[field] = value
    path = tmp_path / "changed.json"
    _write_payload(path, payload)

    with pytest.raises(TaskCRehearsalError):
        load_task_c_rehearsal_config(path)


def test_config_rejects_unknown_reordered_duplicate_deep_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["unknown"] = 1
    unknown = tmp_path / "unknown.json"
    _write_payload(unknown, payload)
    with pytest.raises(TaskCRehearsalError, match="fields or their order changed"):
        load_task_c_rehearsal_config(unknown)

    reordered = tmp_path / "reordered.json"
    pairs = list(_valid_payload().items())
    _write_payload(reordered, dict(reversed(pairs)))
    with pytest.raises(TaskCRehearsalError, match="fields or their order changed"):
        load_task_c_rehearsal_config(reordered)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1.0","schema_version":"1.0"}\n')
    with pytest.raises(TaskCRehearsalError, match="duplicate field"):
        load_task_c_rehearsal_config(duplicate)

    deep = tmp_path / "deep.json"
    deep.write_text('{"x":' + "[" * 40 + "0" + "]" * 40 + "}\n")
    with pytest.raises(TaskCRehearsalError, match="too deeply nested"):
        load_task_c_rehearsal_config(deep)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version": NaN}\n')
    with pytest.raises(TaskCRehearsalError, match="non-finite"):
        load_task_c_rehearsal_config(nonfinite)

    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"schema_version": 1e400}\n')
    with pytest.raises(TaskCRehearsalError, match="non-finite"):
        load_task_c_rehearsal_config(overflow)


def test_config_rejects_oversized_input_before_json_decoding(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (65_536 + 1))

    with pytest.raises(TaskCRehearsalError, match="too large"):
        load_task_c_rehearsal_config(path)


def test_config_rejects_symlink_special_file_huge_integer_and_parser_recursion(
    tmp_path: Path,
) -> None:
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(CONFIG_PATH)
    with pytest.raises(TaskCRehearsalError, match="regular file"):
        load_task_c_rehearsal_config(symlink)

    with pytest.raises(TaskCRehearsalError, match="regular file"):
        load_task_c_rehearsal_config(Path("/dev/zero"))

    huge_integer = tmp_path / "huge-integer.json"
    huge_integer.write_text('{"value":' + "9" * 5_000 + "}\n")
    with pytest.raises(TaskCRehearsalError, match="valid JSON"):
        load_task_c_rehearsal_config(huge_integer)

    recursive = tmp_path / "recursive.json"
    recursive.write_text("[" * 2_000 + "0" + "]" * 2_000)
    with pytest.raises(TaskCRehearsalError, match="deeply nested"):
        load_task_c_rehearsal_config(recursive)

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"{\xff}")
    with pytest.raises(TaskCRehearsalError, match="UTF-8"):
        load_task_c_rehearsal_config(invalid_utf8)


def test_gene_selection_uses_population_variance_and_gene_name_ties() -> None:
    k562 = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 5.0, 2.0], [2.0, 10.0, 4.0]]
    )
    rpe1 = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 4.0, 1.0], [4.0, 8.0, 2.0]]
    )
    before = {"k562": k562.copy(), "rpe1": rpe1.copy()}

    selected = choose_rehearsal_genes(
        {"rpe1": rpe1, "k562": k562},
        gene_names=["A", "B", "C"],
        maximum_genes=3,
    )

    assert selected == ("B", "A", "C")
    np.testing.assert_array_equal(k562, before["k562"])
    np.testing.assert_array_equal(rpe1, before["rpe1"])


def test_gene_selection_rejects_nonfinite_derived_variance() -> None:
    overflowing = np.asarray(
        [[-1e308, 1.0], [1e308, 2.0]], dtype=np.float64
    )
    with pytest.raises(TaskCRehearsalError, match="derived gene variance"):
        choose_rehearsal_genes(
            {"k562": overflowing, "rpe1": overflowing.copy()},
            gene_names=["A", "B"],
            maximum_genes=2,
        )


@pytest.mark.parametrize(
    ("controls", "genes", "maximum", "message"),
    [
        (
            {"k562": np.ones((2, 2)), "rpe1": np.ones((2, 2))},
            ["A", 2],
            2,
            "gene names",
        ),
        (
            {"k562": np.ones((2, 2)), "rpe1": np.ones((2, 2))},
            ["A", "A"],
            2,
            "unique",
        ),
        (
            {"k562": np.ones((2, 2)), "rpe1": np.ones((2, 2))},
            ["A", "e\u0301"],
            2,
            "NFC",
        ),
        (
            {"k562": np.ones((2, 2)), "rpe1": np.asarray([[1.0, np.inf]] * 2)},
            ["A", "B"],
            2,
            "finite",
        ),
        (
            {"k562": np.ones((1, 2)), "rpe1": np.ones((2, 2))},
            ["A", "B"],
            2,
            "two control",
        ),
        (
            {"k562": np.ones((2, 2)), "rpe1": np.ones((2, 2))},
            ["A", "B"],
            True,
            "maximum_genes",
        ),
    ],
)
def test_gene_selection_rejects_unsafe_or_inconsistent_inputs(
    controls: dict[str, np.ndarray],
    genes: list[object],
    maximum: object,
    message: str,
) -> None:
    with pytest.raises(TaskCRehearsalError, match=message):
        choose_rehearsal_genes(
            controls,
            gene_names=genes,  # type: ignore[arg-type]
            maximum_genes=maximum,  # type: ignore[arg-type]
        )


def test_gene_selection_requires_the_two_registered_contexts() -> None:
    with pytest.raises(TaskCRehearsalError, match="k562 and rpe1"):
        choose_rehearsal_genes(
            {"k562": np.ones((2, 2))},
            gene_names=["A", "B"],
            maximum_genes=2,
        )


def test_gene_and_cell_text_rejects_unencodable_or_oversized_values() -> None:
    controls = {"k562": np.ones((2, 2)), "rpe1": np.ones((2, 2))}
    with pytest.raises(TaskCRehearsalError, match="UTF-8"):
        choose_rehearsal_genes(
            controls,
            gene_names=["A", "\ud800"],
            maximum_genes=2,
        )
    with pytest.raises(TaskCRehearsalError, match="text limit"):
        choose_rehearsal_cells(
            ["non-targeting", "X" * 5_000],
            maximum_cells=2,
            seed=11,
        )
    with pytest.raises(TaskCRehearsalError, match="UTF-8"):
        choose_rehearsal_cells(
            ["non-targeting", "\ud800"],
            maximum_cells=2,
            seed=11,
        )
    with pytest.raises(TaskCRehearsalError, match="one-dimensional text list"):
        choose_rehearsal_cells(
            (label for label in ("non-targeting", "A")),  # type: ignore[arg-type]
            maximum_cells=2,
            seed=11,
        )


def test_cell_selection_matches_registered_stratified_quota_and_is_reproducible() -> None:
    labels = np.asarray(["non-targeting"] * 6 + ["A"] * 6 + ["B"] * 6)
    before = labels.copy()
    first = choose_rehearsal_cells(
        labels,
        maximum_cells=9,
        seed=11,
        minimum_cells_per_group=2,
    )
    second = choose_rehearsal_cells(
        labels.tolist(),
        maximum_cells=9,
        seed=11,
        minimum_cells_per_group=2,
    )

    assert first.tolist() == second.tolist()
    assert np.all(first[:-1] < first[1:])
    with pytest.raises(ValueError):
        first.setflags(write=True)
    selected_labels = labels[first]
    assert {
        label: int((selected_labels == label).sum())
        for label in set(labels.tolist())
    } == {"A": 3, "B": 3, "non-targeting": 3}
    np.testing.assert_array_equal(labels, before)


def test_cell_selection_reserves_the_minimum_and_does_not_create_small_groups() -> None:
    labels = ["non-targeting"] * 10 + ["A"] * 4 + ["B"] * 4
    selected = choose_rehearsal_cells(
        labels,
        maximum_cells=12,
        seed=11,
        minimum_cells_per_group=3,
    )
    counts = {
        label: int(np.count_nonzero(np.asarray(labels)[selected] == label))
        for label in set(labels)
    }
    assert all(count >= 3 for count in counts.values())

    with pytest.raises(TaskCRehearsalError, match="minimum"):
        choose_rehearsal_cells(
            labels,
            maximum_cells=8,
            seed=11,
            minimum_cells_per_group=3,
        )


def test_cell_selection_limits_distinct_groups_to_the_public_profile_universe() -> None:
    labels = [f"G{index}" for index in range(1_003)]
    with pytest.raises(TaskCRehearsalError, match="distinct cell-label groups"):
        choose_rehearsal_cells(
            labels,
            maximum_cells=len(labels),
            seed=11,
        )


@pytest.mark.parametrize(
    ("labels", "maximum", "seed", "minimum", "message"),
    [
        (["A", 2], 2, 11, 1, "cell labels"),
        (["A", "e\u0301"], 2, 11, 1, "NFC"),
        (["A", "A"], 0, 11, 1, "maximum_cells"),
        (["A", "A"], True, 11, 1, "maximum_cells"),
        (["A", "A"], 2, -1, 1, "seed"),
        (["A", "A"], 2, True, 1, "seed"),
        (["A", "A"], 2, 11, True, "minimum_cells_per_group"),
    ],
)
def test_cell_selection_rejects_invalid_text_and_numeric_boundaries(
    labels: list[object],
    maximum: object,
    seed: object,
    minimum: object,
    message: str,
) -> None:
    with pytest.raises(TaskCRehearsalError, match=message):
        choose_rehearsal_cells(
            labels,  # type: ignore[arg-type]
            maximum_cells=maximum,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
            minimum_cells_per_group=minimum,  # type: ignore[arg-type]
        )


def test_cross_context_merge_uses_population_control_zscore_and_fixed_order() -> None:
    k562 = np.asarray(
        [[1.0, 2.0, 4.0], [3.0, 4.0, 4.0], [9.0, 8.0, 5.0]],
        dtype=np.float64,
    )
    rpe1 = np.asarray(
        [[10.0, 20.0, 7.0], [14.0, 24.0, 7.0], [18.0, 28.0, 8.0]],
        dtype=np.float64,
    )
    k562_before = k562.copy()
    rpe1_before = rpe1.copy()
    random_before = np.random.get_state()

    merged, labels, environments = center_and_merge_allowed_contexts(
        {
            "rpe1": (
                rpe1,
                np.asarray(["non-targeting", "non-targeting", "B"]),
            ),
            "k562": (
                k562,
                np.asarray(["non-targeting", "non-targeting", "A"]),
            ),
        }
    )

    assert environments.tolist() == ["k562"] * 3 + ["rpe1"] * 3
    for environment in ("k562", "rpe1"):
        controls = (environments == environment) & (labels == "non-targeting")
        np.testing.assert_allclose(merged[controls].mean(axis=0), 0.0, atol=1e-12)
        assert merged[controls, 2].std(ddof=0) == 0.0
        np.testing.assert_allclose(
            merged[controls, :2].std(axis=0, ddof=0), 1.0, atol=1e-12
        )
    assert not merged.flags.writeable
    assert not labels.flags.writeable
    assert not environments.flags.writeable
    for values in (merged, labels, environments):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    np.testing.assert_array_equal(k562, k562_before)
    np.testing.assert_array_equal(rpe1, rpe1_before)
    random_after = np.random.get_state()
    assert random_before[0] == random_after[0]
    np.testing.assert_array_equal(random_before[1], random_after[1])
    assert random_before[2:] == random_after[2:]


def test_cross_context_merge_rejects_nonfinite_derived_statistics() -> None:
    expression = np.asarray(
        [[1e308, 1.0], [1e308, 2.0], [1e308, 3.0]], dtype=np.float64
    )
    labels = np.asarray(["non-targeting", "non-targeting", "A"])
    with pytest.raises(TaskCRehearsalError, match="derived control statistics"):
        center_and_merge_allowed_contexts(
            {
                "k562": (expression, labels),
                "rpe1": (expression.copy(), labels.copy()),
            }
        )


def test_cross_context_merge_rejects_duplicate_context_iteration() -> None:
    entry = (np.ones((2, 2)), np.asarray(["non-targeting"] * 2))

    class DuplicateContextMapping(
        Mapping[str, tuple[np.ndarray, np.ndarray]]
    ):
        def __getitem__(self, key: str) -> tuple[np.ndarray, np.ndarray]:
            if key not in {"k562", "rpe1"}:
                raise KeyError(key)
            return entry

        def __iter__(self) -> Iterator[str]:
            return iter(("k562", "k562", "rpe1"))

        def __len__(self) -> int:
            return 3

    with pytest.raises(TaskCRehearsalError, match="exactly k562 and rpe1"):
        center_and_merge_allowed_contexts(DuplicateContextMapping())


@pytest.mark.parametrize(
    ("contexts", "message"),
    [
        (
            {"k562": (np.ones((2, 2)), np.asarray(["non-targeting"] * 2))},
            "k562 and rpe1",
        ),
        (
            {
                "k562": (np.ones((2, 2)), np.asarray(["non-targeting"] * 2)),
                "rpe1": (np.ones((2, 3)), np.asarray(["non-targeting"] * 2)),
            },
            "same genes",
        ),
        (
            {
                "k562": (np.ones((2, 2)), np.asarray(["non-targeting"])),
                "rpe1": (np.ones((2, 2)), np.asarray(["non-targeting"] * 2)),
            },
            "shape",
        ),
        (
            {
                "k562": (
                    np.asarray([[1.0, np.nan], [2.0, 3.0]]),
                    np.asarray(["non-targeting"] * 2),
                ),
                "rpe1": (np.ones((2, 2)), np.asarray(["non-targeting"] * 2)),
            },
            "finite",
        ),
        (
            {
                "k562": (np.ones((2, 2)), np.asarray(["non-targeting", 2], dtype=object)),
                "rpe1": (np.ones((2, 2)), np.asarray(["non-targeting"] * 2)),
            },
            "labels",
        ),
        (
            {
                "k562": (np.ones((2, 2)), np.asarray(["non-targeting", "A"])),
                "rpe1": (np.ones((2, 2)), np.asarray(["non-targeting"] * 2)),
            },
            "two controls",
        ),
    ],
)
def test_cross_context_merge_rejects_unsafe_or_inconsistent_inputs(
    contexts: dict[str, tuple[np.ndarray, np.ndarray]], message: str
) -> None:
    with pytest.raises(TaskCRehearsalError, match=message):
        center_and_merge_allowed_contexts(contexts)
