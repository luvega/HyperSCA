from __future__ import annotations

import csv
import importlib.util
import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import stat
import sys
import textwrap
from types import MappingProxyType
import uuid

import numpy as np
import pytest

from src.evaluation.task_c_rehearsal import (
    RehearsalProfile,
    TaskCRehearsalConfig,
    TaskCRehearsalError,
    center_and_merge_allowed_contexts,
    choose_rehearsal_cells,
    choose_rehearsal_genes,
    freeze_method_worker_entry,
    load_task_c_rehearsal_config,
    run_validated_private_scoring_command,
    validate_private_scoring_command,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/task_c_rehearsal_v1.json"
EVALUATION_WORKER = (
    ROOT / "scripts/task_c_workers/causalbench_evaluation_worker.py"
)

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


def _write_python(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _load_evaluation_worker() -> object:
    module_name = f"test_task_c_evaluation_worker_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, EVALUATION_WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_evaluation_source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "causalbench-fixed-source"
    _write_python(source / "causalscbench/__init__.py", "")
    _write_python(source / "causalscbench/evaluation/__init__.py", "")
    _write_python(
        source / "causalscbench/evaluation/statistical_evaluation.py",
        """
        from collections.abc import Mapping
        import json
        import os
        import numpy as np

        if os.environ.get("FAKE_SOURCE_MUTATION") == "during_import":
            with open(__file__, "a", encoding="utf-8") as handle:
                handle.write("# changed during import\\n")

        class DuplicateMetrics(Mapping):
            def __len__(self):
                return 2
            def __iter__(self):
                return iter(("metric", "metric"))
            def __getitem__(self, key):
                return 1.0
            def items(self):
                return (("metric", 1.0), ("metric", 2.0))

        class Evaluator:
            def __init__(self, expression, interventions, genes):
                self.record = {
                    "shape": list(expression.shape),
                    "interventions": list(interventions),
                    "genes": list(genes),
                }
                if os.environ.get("FAKE_SOURCE_MUTATION") == "during_init":
                    with open(__file__, "a", encoding="utf-8") as handle:
                        handle.write("# changed during evaluator setup\\n")

            def evaluate_network(self, edges, **kwargs):
                self.record["edges"] = [list(edge) for edge in edges]
                self.record["kwargs"] = kwargs
                with open(os.environ["FAKE_EVALUATION_RECORD"], "w", encoding="utf-8") as handle:
                    json.dump(self.record, handle)
                kind = os.environ.get("FAKE_METRIC_KIND", "valid")
                if kind == "nonfinite":
                    return {"false_omission_rate": float("nan")}
                if kind == "duplicate_mapping":
                    return DuplicateMetrics()
                if kind == "deep":
                    value = 1.0
                    for _ in range(40):
                        value = [value]
                    return {"too_deep": value}
                if kind == "raises":
                    raise RuntimeError("simulated official evaluator failure")
                return {
                    "false_discovery_rate": np.float64(0.25),
                    "counts": np.asarray([2, 1], dtype=np.int64),
                }
        """,
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test only"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "remote",
            "add",
            "origin",
            "https://github.com/causalbench/causalbench.git",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "causalscbench"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "test evaluator"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, revision


def _write_heldout_and_predictions(tmp_path: Path) -> tuple[Path, Path]:
    heldout = tmp_path / "sealed holdout.npz"
    np.savez(
        heldout,
        expression_matrix=np.asarray(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 2.0, 1.0], [3.0, 1.0, 0.0]]
        ),
        interventions=np.asarray(["non-targeting", "A", "excluded", "B"]),
        var_names=np.asarray(["A", "B", "C"]),
    )
    predictions = tmp_path / "predictions.csv"
    rows = [
        ("C", "B", 0.0, False),
        ("A", "C", 0.7, True),
        ("B", "A", 0.9, True),
        ("A", "B", 0.9, True),
        ("C", "A", 0.0, False),
        ("B", "C", 0.0, False),
    ]
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source", "target", "score", "returned_by_method"))
        writer.writerows(rows)
    return heldout, predictions


def _run_evaluation_worker(
    tmp_path: Path,
    *,
    metric_kind: str = "valid",
    output: Path | None = None,
) -> tuple[int, str, Path, Path]:
    source, revision = _fake_evaluation_source(tmp_path)
    heldout, predictions = _write_heldout_and_predictions(tmp_path)
    output_path = output or tmp_path / "official metrics.json"
    record = tmp_path / "evaluation-call.json"
    completed = _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output_path,
        seed="17",
        environment={
            "FAKE_EVALUATION_RECORD": str(record),
            "FAKE_METRIC_KIND": metric_kind,
        },
    )
    return completed.returncode, completed.stderr, output_path, record


def _invoke_evaluation_worker(
    tmp_path: Path,
    *,
    source: Path,
    revision: str,
    heldout: Path,
    predictions: Path,
    output: Path,
    seed: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    runner = tmp_path / f"isolated-worker-{uuid.uuid4().hex}.py"
    runner.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import sys

            path = {str(EVALUATION_WORKER)!r}
            spec = importlib.util.spec_from_file_location("isolated_evaluation_worker", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.EXPECTED_CAUSALBENCH_COMMIT = {revision!r}
            raise SystemExit(module.main(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    process_environment = os.environ.copy()
    if environment:
        process_environment.update(environment)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(runner),
            "--prediction-csv",
            str(predictions),
            "--heldout-npz",
            str(heldout),
            "--output-json",
            str(output),
            "--seed",
            seed,
            "--causalbench-source",
            str(source),
        ],
        cwd=ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
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


def test_config_rejects_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "configuration.pipe"
    os.mkfifo(fifo)
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from src.evaluation.task_c_rehearsal import (",
            "    TaskCRehearsalError, load_task_c_rehearsal_config,",
            ")",
            "try:",
            "    load_task_c_rehearsal_config(Path(sys.argv[1]))",
            "except TaskCRehearsalError:",
            "    print('rejected without waiting')",
            "else:",
            "    raise SystemExit('FIFO was accepted')",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(fifo)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=2,
        check=True,
    )
    assert completed.stdout.strip() == "rejected without waiting"


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


@pytest.mark.parametrize(
    "argument_builder",
    [
        lambda private, _alias, _cwd: [str(private / "test.npz")],
        lambda private, _alias, _cwd: [f"--input={private / 'test.npz'}"],
        lambda private, _alias, _cwd: [f"-I{private / 'test.npz'}"],
        lambda private, _alias, cwd: [
            os.path.relpath(private / "test.npz", cwd)
        ],
        lambda _private, alias, _cwd: [str(alias / "test.npz")],
    ],
)
def test_method_command_cannot_receive_private_paths(
    tmp_path: Path, argument_builder
) -> None:
    private_root = tmp_path / "private scoring data"
    private_root.mkdir()
    (private_root / "test.npz").write_bytes(b"private")
    alias = tmp_path / "innocent alias"
    alias.symlink_to(private_root, target_is_directory=True)
    worker = tmp_path / "worker.py"
    worker.write_text("# fixed worker\n", encoding="utf-8")
    interpreter = Path(sys.executable).resolve(strict=True)
    worker_snapshot = freeze_method_worker_entry(worker)

    with pytest.raises(TaskCRehearsalError, match="private scoring path"):
        validate_private_scoring_command(
            [
                str(interpreter),
                "-I",
                str(worker),
                *argument_builder(private_root, alias, tmp_path),
            ],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(worker_snapshot,),
        )


def test_private_command_check_rejects_ambiguous_inputs_and_bad_private_root(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    with pytest.raises(TaskCRehearsalError, match="ordered list"):
        validate_private_scoring_command(
            "python worker.py",  # type: ignore[arg-type]
            private_root=private_root,
            execution_cwd=tmp_path,
        )
    with pytest.raises(TaskCRehearsalError, match="NUL"):
        validate_private_scoring_command(
            ["python", "worker.py\x00--input"],
            private_root=private_root,
            execution_cwd=tmp_path,
        )

    alias = tmp_path / "private-alias"
    alias.symlink_to(private_root, target_is_directory=True)
    with pytest.raises(TaskCRehearsalError, match="real directory"):
        validate_private_scoring_command(
            ["python", "worker.py"],
            private_root=alias,
            execution_cwd=tmp_path,
        )


def test_method_command_without_private_paths_is_accepted(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    public_input = tmp_path / "public input.npz"
    worker = tmp_path / "worker.py"
    worker.write_text("# fixed worker\n", encoding="utf-8")
    interpreter = Path(sys.executable).resolve(strict=True)
    worker_snapshot = freeze_method_worker_entry(worker)

    validate_private_scoring_command(
        [str(interpreter), "-I", str(worker), "--input", str(public_input)],
        private_root=private_root,
        execution_cwd=tmp_path,
        allowed_python_interpreters=(interpreter,),
        allowed_worker_snapshots=(worker_snapshot,),
    )


@pytest.mark.parametrize(
    "command",
    [
        ["bash", "-lc", "python worker.py --input safe.npz"],
        ["/usr/bin/env", "python", "worker.py"],
        ["python", "-c", "open('/tmp/input')"],
        ["python", "-m", "unregistered.worker"],
    ],
)
def test_private_command_check_rejects_shell_and_dynamic_python(
    tmp_path: Path, command: list[str]
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    with pytest.raises(TaskCRehearsalError, match="dynamic|wrapper"):
        validate_private_scoring_command(
            command,
            private_root=private_root,
            execution_cwd=tmp_path,
        )


def test_private_command_check_decodes_file_uri_and_uses_execution_cwd(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private scoring data"
    private_root.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("# fixed worker\n", encoding="utf-8")
    interpreter = Path(sys.executable).resolve(strict=True)
    worker_snapshot = freeze_method_worker_entry(worker)
    encoded = "file://" + str(private_root / "sealed.npz").replace(" ", "%20")
    with pytest.raises(TaskCRehearsalError, match="private scoring path"):
        validate_private_scoring_command(
            [str(interpreter), "-I", str(worker), f"--input={encoded}"],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(worker_snapshot,),
        )

    with pytest.raises(TaskCRehearsalError, match="private scoring path"):
        validate_private_scoring_command(
            [
                str(interpreter),
                "-I",
                str(worker),
                "--input",
                "private scoring data/sealed.npz",
            ],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(worker_snapshot,),
        )


def test_private_command_check_requires_registered_worker_when_requested(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    registered = tmp_path / "registered.py"
    unregistered = tmp_path / "unregistered.py"
    registered.write_text("# registered\n", encoding="utf-8")
    unregistered.write_text("# unregistered\n", encoding="utf-8")
    interpreter = Path(sys.executable).resolve(strict=True)
    registered_snapshot = freeze_method_worker_entry(registered)

    with pytest.raises(TaskCRehearsalError, match="registered worker"):
        validate_private_scoring_command(
            [
                str(interpreter),
                "-I",
                str(unregistered),
                str(registered),
            ],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(registered_snapshot,),
        )


def test_private_command_check_binds_entry_extension_and_frozen_identity(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    worker = tmp_path / "registered.py"
    worker.write_text("# version one\n", encoding="utf-8")
    snapshot = freeze_method_worker_entry(worker)
    interpreter = Path(sys.executable).resolve(strict=True)

    extensionless = tmp_path / "entry"
    extensionless.write_text(worker.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TaskCRehearsalError, match="Python worker entry"):
        validate_private_scoring_command(
            [str(interpreter), "-I", str(extensionless), str(worker)],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(snapshot,),
        )

    linked = tmp_path / "linked.py"
    linked.symlink_to(worker)
    with pytest.raises(TaskCRehearsalError, match="symbolic links"):
        validate_private_scoring_command(
            [str(interpreter), "-I", str(linked)],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(snapshot,),
        )

    worker.write_text("# changed after approval\n", encoding="utf-8")
    with pytest.raises(TaskCRehearsalError, match="worker.*changed"):
        validate_private_scoring_command(
            [str(interpreter), "-I", str(worker)],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(snapshot,),
        )


def test_validated_launcher_executes_frozen_worker_bundle_and_cleans_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    worker = tmp_path / "worker.py"
    dependency = tmp_path / "reviewed_dependency.py"
    dependency.write_text("VALUE = 'frozen dependency'\n", encoding="utf-8")
    worker.write_text(
        "from pathlib import Path\n"
        "dependency = Path(__file__).with_name('reviewed_dependency.py')\n"
        "print(Path(__file__).resolve())\n"
        "print(dependency.read_text(encoding='utf-8').strip())\n",
        encoding="utf-8",
    )
    entry_snapshot = freeze_method_worker_entry(worker)
    dependency_snapshot = freeze_method_worker_entry(dependency)
    interpreter = Path(sys.executable).resolve(strict=True)
    real_run = subprocess.run
    live_marker = tmp_path / "live-worker-executed"

    def replace_live_worker_then_run(command, **kwargs):
        worker.write_text(
            f"from pathlib import Path\nPath({str(live_marker)!r}).write_text('unsafe')\n",
            encoding="utf-8",
        )
        return real_run(command, **kwargs)

    monkeypatch.setattr(
        "src.evaluation.task_c_rehearsal.subprocess.run",
        replace_live_worker_then_run,
    )
    completed = run_validated_private_scoring_command(
        [str(interpreter), "-I", str(worker)],
        private_root=private_root,
        execution_cwd=tmp_path,
        allowed_python_interpreters=(interpreter,),
        allowed_worker_snapshots=(entry_snapshot, dependency_snapshot),
    )

    assert completed.returncode == 0, completed.stderr
    output_lines = completed.stdout.splitlines()
    frozen_entry = Path(output_lines[0])
    assert output_lines[1] == "VALUE = 'frozen dependency'"
    assert frozen_entry != worker
    assert "hypersca-method-worker-" in str(frozen_entry)
    assert not frozen_entry.parent.exists()
    assert not live_marker.exists()


def test_validated_launcher_rejects_worker_change_before_snapshot(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("print('reviewed')\n", encoding="utf-8")
    snapshot = freeze_method_worker_entry(worker)
    interpreter = Path(sys.executable).resolve(strict=True)
    worker.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(TaskCRehearsalError, match="worker.*changed"):
        run_validated_private_scoring_command(
            [str(interpreter), "-I", str(worker)],
            private_root=private_root,
            execution_cwd=tmp_path,
            allowed_python_interpreters=(interpreter,),
            allowed_worker_snapshots=(snapshot,),
        )


def test_official_evaluation_worker_exposes_help_without_external_imports(
    tmp_path: Path,
) -> None:
    blockers = tmp_path / "blocked_external_imports"
    blockers.mkdir()
    for module_name in ("numpy", "pandas", "causalscbench"):
        (blockers / f"{module_name}.py").write_text(
            f"raise RuntimeError('{module_name} imported during help')\n",
            encoding="utf-8",
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(blockers)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/task_c_workers/causalbench_evaluation_worker.py",
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "封存" in completed.stdout
    assert "--prediction-csv" in completed.stdout
    assert "--causalbench-source" in completed.stdout


def test_official_evaluation_worker_requires_isolated_python_for_real_scoring(
    tmp_path: Path,
) -> None:
    worker = _load_evaluation_worker()
    output = tmp_path / "runtime-status.json"

    return_code = worker.main(
        [
            "--prediction-csv",
            str(tmp_path / "not-read.csv"),
            "--heldout-npz",
            str(tmp_path / "not-read.npz"),
            "--output-json",
            str(output),
            "--seed",
            "17",
            "--causalbench-source",
            str(tmp_path / "not-read-source"),
        ]
    )

    assert return_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_runtime_unavailable"
    assert "python -I" in payload["error"]


def test_official_evaluation_worker_uses_fixed_signature_and_stable_order(
    tmp_path: Path,
) -> None:
    return_code, stderr, output, record_path = _run_evaluation_worker(tmp_path)

    assert return_code == 0, stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == [
        "eligible_source_count",
        "metrics",
        "schema_version",
        "seed",
        "status",
    ]
    assert payload == {
        "eligible_source_count": 2,
        "metrics": {"counts": [2, 1], "false_discovery_rate": 0.25},
        "schema_version": "1.0",
        "seed": 17,
        "status": "supplementary_official_metrics",
    }
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["shape"] == [4, 3]
    assert record["genes"] == ["A", "B", "C"]
    assert record["edges"] == [
        ["A", "B"],
        ["B", "A"],
        ["A", "C"],
    ]
    assert record["kwargs"] == {
        "max_path_length": 1,
        "check_false_omission_rate": False,
        "omission_estimation_size": 0,
        "seed": 17,
    }
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_three_column_prediction_table_treats_every_relation_as_returned(
    tmp_path: Path,
) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    heldout, predictions = _write_heldout_and_predictions(tmp_path)
    rows = list(csv.reader(predictions.read_text(encoding="utf-8").splitlines()))
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows([row[:3] for row in rows])
    output = tmp_path / "metrics.json"
    record = tmp_path / "call.json"

    completed = _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
        environment={"FAKE_EVALUATION_RECORD": str(record)},
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(record.read_text(encoding="utf-8"))["edges"] == [
        ["A", "B"],
        ["B", "A"],
        ["A", "C"],
        ["B", "C"],
    ]


def test_isolated_worker_ignores_pythonpath_sitecustomize_injection(
    tmp_path: Path,
) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    heldout, predictions = _write_heldout_and_predictions(tmp_path)
    injected = tmp_path / "injected"
    marker = tmp_path / "sitecustomize-ran"
    _write_python(
        injected / "sitecustomize.py",
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
    )
    _write_python(injected / "numpy.py", "raise RuntimeError('injected numpy')")
    output = tmp_path / "metrics.json"
    record = tmp_path / "call.json"

    completed = _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
        environment={
            "PYTHONPATH": str(injected),
            "FAKE_EVALUATION_RECORD": str(record),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert not marker.exists()


def test_verified_source_snapshot_detects_a_change_before_import(tmp_path: Path) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    worker = _load_evaluation_worker()
    boundary = worker._load_causalbench_boundary_module()
    verified = boundary.validate_causalbench_source(source, revision)
    frozen = worker.freeze_causalbench_python_source(
        verified, boundary, expected_commit=revision
    )
    evaluation_path = source / "causalscbench/evaluation/statistical_evaluation.py"
    evaluation_path.write_text(
        evaluation_path.read_text(encoding="utf-8") + "# changed before import\n",
        encoding="utf-8",
    )

    with pytest.raises(worker.ScoringContractError, match="changed"):
        worker.verify_causalbench_python_source(verified, frozen)


def test_official_import_uses_read_only_snapshot_not_changed_live_checkout(
    tmp_path: Path,
) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    worker = _load_evaluation_worker()
    boundary = worker._load_causalbench_boundary_module()
    verified = boundary.validate_causalbench_source(source, revision)
    live_evaluation = source / "causalscbench/evaluation/statistical_evaluation.py"
    marker = tmp_path / "live-module-executed"
    snapshot_path: Path | None = None

    with worker.fixed_causalbench_source_snapshot(
        verified, boundary, expected_commit=revision
    ) as (snapshot, frozen):
        snapshot_path = snapshot
        snapshot_evaluation = (
            snapshot / "causalscbench/evaluation/statistical_evaluation.py"
        )
        assert stat.S_IMODE(snapshot.stat().st_mode) & stat.S_IWUSR == 0
        assert stat.S_IMODE(snapshot_evaluation.stat().st_mode) & stat.S_IWUSR == 0
        live_evaluation.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
            encoding="utf-8",
        )
        with boundary._verified_causalbench_imports(snapshot):
            module = importlib.import_module(
                "causalscbench.evaluation.statistical_evaluation"
            )
            worker.verify_causalbench_python_source(snapshot, frozen)
            worker.verify_loaded_causalbench_modules(snapshot, frozen)
        assert Path(module.__file__).resolve().is_relative_to(snapshot)
        assert not marker.exists()

    assert snapshot_path is not None
    assert not snapshot_path.exists()


def test_official_snapshot_uses_explicit_commit_not_a_new_head(tmp_path: Path) -> None:
    source, revision_a = _fake_evaluation_source(tmp_path)
    evaluation = source / "causalscbench/evaluation/statistical_evaluation.py"
    marker = tmp_path / "revision-b-executed"
    evaluation.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(source), "add", "causalscbench"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "revision B"],
        check=True,
    )
    revision_b = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision_b != revision_a
    worker = _load_evaluation_worker()
    boundary = worker._load_causalbench_boundary_module()

    with pytest.raises((SystemExit, worker.ScoringContractError), match="revision|commit"):
        with worker.fixed_causalbench_source_snapshot(
            source,
            boundary,
            expected_commit=revision_a,
        ):
            raise AssertionError("a snapshot from revision B must not be yielded")
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["during_import", "during_init"])
def test_official_worker_rejects_source_changes_during_import_or_setup(
    tmp_path: Path, mutation: str
) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    heldout, predictions = _write_heldout_and_predictions(tmp_path)
    output = tmp_path / "failed.json"
    record = tmp_path / "call.json"

    completed = _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
        environment={
            "FAKE_SOURCE_MUTATION": mutation,
            "FAKE_EVALUATION_RECORD": str(record),
        },
    )

    assert completed.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "failed_private_scoring"
    )
    assert not record.exists()


@pytest.mark.parametrize("metric_kind", ["nonfinite", "duplicate_mapping", "deep"])
def test_official_evaluation_worker_records_invalid_metrics_without_partial_success(
    tmp_path: Path, metric_kind: str
) -> None:
    return_code, stderr, output, _ = _run_evaluation_worker(
        tmp_path, metric_kind=metric_kind
    )

    assert return_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "failed_invalid_output"
    assert payload["seed"] == 17
    assert "metrics" not in payload
    assert "failed_invalid_output" in stderr
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_official_evaluation_worker_records_an_evaluator_exception(
    tmp_path: Path,
) -> None:
    return_code, stderr, output, _ = _run_evaluation_worker(
        tmp_path, metric_kind="raises"
    )

    assert return_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_private_scoring"
    assert "simulated official evaluator failure" not in payload["error"]
    assert "Traceback" not in stderr


def test_official_evaluation_worker_never_overwrites_an_existing_result(
    tmp_path: Path,
) -> None:
    output = tmp_path / "already exists.json"
    output.write_text("keep this result\n", encoding="utf-8")
    return_code, stderr, returned, record = _run_evaluation_worker(
        tmp_path / "run", output=output
    )

    assert return_code != 0
    assert "already exist" in stderr
    assert returned.read_text(encoding="utf-8") == "keep this result\n"
    assert not record.exists()


def test_official_evaluation_worker_rejects_incomplete_or_false_nonzero_predictions(
    tmp_path: Path,
) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    heldout, predictions = _write_heldout_and_predictions(tmp_path)
    lines = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    output = tmp_path / "failed.json"
    completed = _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
    )
    assert completed.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "failed_private_scoring"
    )

    third_root = tmp_path / "third"
    third_root.mkdir()
    source, revision = _fake_evaluation_source(third_root)
    heldout, predictions = _write_heldout_and_predictions(third_root)
    predictions.write_text(
        predictions.read_text(encoding="utf-8").replace(
            "C,B,0.0,False", "C,B,-0.0,False"
        ),
        encoding="utf-8",
    )
    output = third_root / "failed.json"
    assert _invoke_evaluation_worker(
        third_root,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
    ).returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "failed_private_scoring"
    )

    second_root = tmp_path / "second"
    second_root.mkdir()
    source, revision = _fake_evaluation_source(second_root)
    heldout, predictions = _write_heldout_and_predictions(second_root)
    content = predictions.read_text(encoding="utf-8").replace(
        "C,B,0.0,False", "C,B,0.1,False"
    )
    predictions.write_text(content, encoding="utf-8")
    output = second_root / "failed.json"
    assert _invoke_evaluation_worker(
        second_root,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
    ).returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "failed_private_scoring"
    )


def test_official_evaluation_worker_rejects_input_aliases_and_bad_seed(
    tmp_path: Path,
) -> None:
    source, revision = _fake_evaluation_source(tmp_path)
    heldout, predictions = _write_heldout_and_predictions(tmp_path)
    linked = tmp_path / "linked-heldout.npz"
    linked.symlink_to(heldout)
    for seed in ("-1", "true", str(2**32)):
        output = tmp_path / f"failed-seed-{seed}.json"
        assert _invoke_evaluation_worker(
            tmp_path,
            source=source,
            revision=revision,
            heldout=heldout,
            predictions=predictions,
            output=output,
            seed=seed,
        ).returncode != 0
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
            "failed_private_scoring"
        )

    output = tmp_path / "failed-link.json"
    assert _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=linked,
        predictions=predictions,
        output=output,
        seed="17",
    ).returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "failed_private_scoring"
    )

    hardlink = tmp_path / "hardlinked-heldout.npz"
    os.link(heldout, hardlink)
    output = tmp_path / "failed-hardlink.json"
    assert _invoke_evaluation_worker(
        tmp_path,
        source=source,
        revision=revision,
        heldout=heldout,
        predictions=predictions,
        output=output,
        seed="17",
    ).returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "failed_private_scoring"
    )


def test_heldout_text_shapes_are_rejected_before_large_vector_conversion(
    tmp_path: Path,
) -> None:
    heldout = tmp_path / "bad-text-shape.npz"
    np.savez(
        heldout,
        expression_matrix=np.zeros((2, 2), dtype=np.float32),
        interventions=np.asarray([b"A"] * 1_000_000, dtype="S1"),
        var_names=np.asarray(["A", "B"]),
    )
    worker = _load_evaluation_worker()
    payload = heldout.read_bytes()

    with pytest.raises(worker.ScoringContractError, match="shapes"):
        worker.load_heldout_npz(payload, np)
