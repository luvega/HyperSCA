from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unicodedata

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CAUSALBENCH_WORKER = ROOT / "scripts/task_c_workers/causalbench_worker.py"
PSGRN_WORKER = ROOT / "scripts/task_c_workers/psgrn_worker.py"
EXPECTED_CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
EXPECTED_PSGRN_COMMIT = "74aa640f7c472b23a69811f6795bb17678efd344"
PROVEN_ORDER_MODELS = {"grnboost"}
UNPROVEN_ORDER_MODELS = {
    "random1000",
    "pc",
    "ges",
    "gies",
    "gsp",
    "igsp",
    "notears-lin-sparse",
    "DCDI-G",
    "DCDI-DSF",
    "DCDFG-LIN",
    "DCDFG-MLP",
    "sortnregress",
}


def _write_input(
    path: Path,
    *,
    expression: object | None = None,
    interventions: object | None = None,
    gene_names: object | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    arrays = {
        "expression_matrix": np.asarray(
            expression
            if expression is not None
            else [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            dtype=np.float64,
        ),
        "interventions": np.asarray(
            interventions
            if interventions is not None
            else ["non-targeting", "A", "non-targeting"]
        ),
        "var_names": np.asarray(
            gene_names if gene_names is not None else ["A", "B"]
        ),
    }
    if extra:
        arrays.update(extra)
    np.savez(path, **arrays)


def _write_python(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _fake_causalbench(tmp_path: Path) -> Path:
    package = tmp_path / "fake_modules"
    models = package / "causalscbench/models"
    _write_python(package / "causalscbench/__init__.py", "")
    _write_python(models / "__init__.py", "")
    _write_python(
        models / "training_regimes.py",
        """
        from enum import Enum

        class TrainingRegime(Enum):
            Observational = "observational"
            PartialIntervational = "partial_interventional"
            Interventional = "interventional"
        """,
    )
    _write_python(
        models / "fake_base.py",
        """
        import json
        import numpy as np
        import os

        class Base:
            def __init__(self, *args, **kwargs):
                self.init_args = list(args)
                self.init_kwargs = kwargs

            def __call__(self, expression, interventions, genes, regime, seed):
                with open(os.environ["FAKE_CALL_RECORD"], "w", encoding="utf-8") as handle:
                    json.dump({
                        "class_name": type(self).__name__,
                        "init_args": self.init_args,
                        "init_kwargs": self.init_kwargs,
                        "shape": list(expression.shape),
                        "expression_writeable": bool(expression.flags.writeable),
                        "interventions": interventions,
                        "genes": genes,
                        "regime_name": regime.name,
                        "regime_value": regime.value,
                        "seed": seed,
                    }, handle)
                edges = [("A", "B"), ("B", "A"), ("A", "A")]
                kind = os.environ.get("FAKE_RETURN_KIND", "list")
                if kind == "set":
                    return set(edges)
                if kind == "mapping":
                    return {edge: index for index, edge in enumerate(edges)}
                if kind == "generator":
                    return (edge for edge in edges)
                if kind == "unknown_gene":
                    return [("A", "NOT_IN_FIXED_GENES")]
                if kind == "numpy_string":
                    return [(np.str_("A"), np.str_("B"))]
                return edges
        """,
    )
    modules = {
        "arboreto_baselines.py": ["GRNBoost"],
        "causallearn_models.py": ["GES", "PC"],
        "dcdi_models.py": ["DCDI", "DCDFG"],
        "gies.py": ["GIES"],
        "notears.py": ["NotearsLin"],
        "random_network.py": ["RandomWithSize"],
        "sparsest_permutations.py": [
            "GreedySparsestPermutation",
            "InterventionalGreedySparsestPermutation",
        ],
        "varsortability.py": ["Sortnregress"],
    }
    for filename, class_names in modules.items():
        classes = "\n".join(f"class {name}(Base):\n    pass" for name in class_names)
        _write_python(
            models / filename,
            f"from causalscbench.models.fake_base import Base\n\n{classes}\n",
        )
    return package


def _run_causalbench(
    tmp_path: Path,
    *,
    model_name: str = "grnboost",
    training_information: str = "observational",
    input_path: Path | None = None,
    output_path: Path | None = None,
    return_kind: str = "list",
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_modules = _fake_causalbench(tmp_path)
    source = input_path or tmp_path / "input.npz"
    if input_path is None:
        _write_input(source)
    destination = output_path or tmp_path / "predictions.csv"
    record = tmp_path / "call.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_modules)
    env["FAKE_CALL_RECORD"] = str(record)
    env["FAKE_RETURN_KIND"] = return_kind
    completed = subprocess.run(
        [
            sys.executable,
            str(CAUSALBENCH_WORKER),
            "--input-npz",
            str(source),
            "--output-csv",
            str(destination),
            "--model-name",
            model_name,
            "--training-information",
            training_information,
            "--seed",
            "17",
            "--output-semantics",
            "official_return_order",
        ],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed, destination, record


def _load_worker(path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_psgrn_repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "PSGRN-test-interface-only"
    source.mkdir()
    _write_python(
        source / "src/main.py",
        """
        from enum import Enum
        import json
        import os

        class TrainingRegime(Enum):
            Observational = "observational"
            PartialIntervational = "partial_interventional"

        class Custom:
            def __call__(self, expression, interventions, genes, regime, seed):
                with open(os.environ["FAKE_CALL_RECORD"], "w", encoding="utf-8") as handle:
                    json.dump({
                        "shape": list(expression.shape),
                        "expression_writeable": bool(expression.flags.writeable),
                        "interventions": interventions,
                        "genes": genes,
                        "regime_name": regime.name,
                        "seed": seed,
                    }, handle)
                edges = [("A", "B"), ("B", "A")] * 600
                kind = os.environ.get("FAKE_RETURN_KIND", "list")
                if kind == "generator":
                    return (edge for edge in edges)
                if kind == "mapping":
                    return {edge: 1 for edge in edges}
                if kind == "set":
                    return set(edges)
                if kind == "unknown_gene":
                    return [("UNKNOWN", "A")]
                return edges
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
            "https://github.com/GuanLab/PSGRN.git",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "src/main.py"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "test interface"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, revision


def test_external_workers_expose_plain_language_help_without_external_imports(
    tmp_path: Path,
) -> None:
    blockers = tmp_path / "blocked_external_imports"
    _write_python(blockers / "numpy.py", "raise RuntimeError('numpy imported during help')")
    _write_python(blockers / "pandas.py", "raise RuntimeError('pandas imported during help')")
    _write_python(
        blockers / "causalscbench/__init__.py",
        "raise RuntimeError('causalscbench imported during help')",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(blockers)

    for worker in (CAUSALBENCH_WORKER, PSGRN_WORKER):
        completed = subprocess.run(
            [sys.executable, str(worker), "--help"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "单细胞" in completed.stdout
        assert "--input-npz" in completed.stdout
        assert "--output-csv" in completed.stdout


@pytest.mark.parametrize(
    ("model_name", "expected_class", "expected_args", "expected_kwargs"),
    [
        ("random1000", "RandomWithSize", [1000], {}),
        ("grnboost", "GRNBoost", [], {}),
        ("pc", "PC", [], {"missing_value": False}),
        ("ges", "GES", [], {}),
        ("gies", "GIES", [], {}),
        ("gsp", "GreedySparsestPermutation", [], {}),
        ("igsp", "InterventionalGreedySparsestPermutation", [], {}),
        ("notears-lin-sparse", "NotearsLin", [], {"lambda1": 0.001}),
        ("DCDI-G", "DCDI", ["DCDI-G"], {}),
        ("DCDI-DSF", "DCDI", ["DCDI-DSF"], {}),
        ("DCDFG-LIN", "DCDFG", ["linear"], {}),
        ("DCDFG-MLP", "DCDFG", ["mlplr"], {}),
        ("sortnregress", "Sortnregress", [], {}),
    ],
)
def test_causalbench_uses_the_pinned_official_constructor_contract(
    tmp_path: Path,
    model_name: str,
    expected_class: str,
    expected_args: list[object],
    expected_kwargs: dict[str, object],
) -> None:
    completed, destination, record_path = _run_causalbench(
        tmp_path,
        model_name=model_name,
        training_information=(
            "partial_interventional"
            if model_name in {"gies", "igsp", "DCDI-G", "DCDI-DSF", "DCDFG-LIN", "DCDFG-MLP"}
            else "observational"
        ),
        check=False,
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["class_name"] == expected_class
    assert record["init_args"] == expected_args
    assert record["init_kwargs"] == expected_kwargs
    if model_name in PROVEN_ORDER_MODELS:
        assert completed.returncode == 0
        assert destination.exists()
    else:
        assert model_name in UNPROVEN_ORDER_MODELS
        assert completed.returncode != 0
        assert "failed_invalid_output" in completed.stderr
        assert not destination.exists()


def test_only_grnboost_has_a_provable_ranked_return_in_the_pinned_adapter() -> None:
    worker = _load_worker(CAUSALBENCH_WORKER, "test_causalbench_order_boundary")

    assert set(worker.PROVEN_OFFICIAL_RETURN_ORDER) == PROVEN_ORDER_MODELS
    assert set(worker.MODEL_NAMES) == PROVEN_ORDER_MODELS | UNPROVEN_ORDER_MODELS


@pytest.mark.parametrize("model_name", sorted(UNPROVEN_ORDER_MODELS))
def test_causalbench_rejects_unproven_order_even_when_the_object_is_a_list(
    tmp_path: Path, model_name: str
) -> None:
    completed, destination, _ = _run_causalbench(
        tmp_path,
        model_name=model_name,
        training_information=(
            "partial_interventional"
            if model_name
            in {"gies", "igsp", "DCDI-G", "DCDI-DSF", "DCDFG-LIN", "DCDFG-MLP"}
            else "observational"
        ),
        return_kind="list",
        check=False,
    )

    assert completed.returncode != 0
    assert "failed_invalid_output" in completed.stderr
    assert "not proven" in completed.stderr
    assert not destination.exists()


def test_observational_worker_uses_only_controls_and_preserves_official_order(
    tmp_path: Path,
) -> None:
    _, destination, record_path = _run_causalbench(tmp_path)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["shape"] == [2, 2]
    assert record["interventions"] == ["non-targeting", "non-targeting"]
    assert record["regime_name"] == "Observational"
    assert record["seed"] == 17
    assert record["expression_writeable"] is False
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["source", "target", "score"]
    assert [(row["source"], row["target"]) for row in rows] == [
        ("A", "B"),
        ("B", "A"),
        ("A", "A"),
    ]
    assert [float(row["score"]) for row in rows] == [3.0, 2.0, 1.0]
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_partial_interventional_worker_keeps_every_allowed_cell(tmp_path: Path) -> None:
    completed, destination, record_path = _run_causalbench(
        tmp_path,
        model_name="gies",
        training_information="partial_interventional",
        check=False,
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["shape"] == [3, 2]
    assert record["interventions"] == ["non-targeting", "A", "non-targeting"]
    assert record["regime_name"] == "PartialIntervational"
    assert completed.returncode != 0
    assert "failed_invalid_output" in completed.stderr
    assert not destination.exists()


@pytest.mark.parametrize("return_kind", ["set", "mapping", "generator"])
def test_ambiguous_causalbench_return_order_fails_closed(
    tmp_path: Path, return_kind: str
) -> None:
    completed, destination, _ = _run_causalbench(
        tmp_path,
        return_kind=return_kind,
        check=False,
    )

    assert completed.returncode != 0
    assert "failed_invalid_output" in completed.stderr
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_unknown_causalbench_endpoint_is_rejected_before_writing(tmp_path: Path) -> None:
    completed, destination, _ = _run_causalbench(
        tmp_path,
        return_kind="unknown_gene",
        check=False,
    )

    assert completed.returncode != 0
    assert "fixed gene set" in completed.stderr
    assert not destination.exists()


def test_numpy_string_endpoints_are_converted_and_validated(tmp_path: Path) -> None:
    completed, destination, _ = _run_causalbench(
        tmp_path,
        model_name="grnboost",
        return_kind="numpy_string",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"source": "A", "target": "B", "score": "1.0"}]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_key", "exactly"),
        ("wrong_rows", "rows"),
        ("wrong_columns", "columns"),
        ("nonfinite", "finite"),
        ("duplicate_genes", "unique"),
        ("empty_label", "non-empty"),
        ("nfd_gene", "NFC"),
        ("missing_control", "non-targeting"),
    ],
)
def test_npz_contract_rejects_invalid_or_ambiguous_arrays(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source = tmp_path / "invalid.npz"
    kwargs: dict[str, object] = {}
    if mutation == "extra_key":
        kwargs["extra"] = {"unexpected": np.asarray([1])}
    elif mutation == "wrong_rows":
        kwargs["interventions"] = ["non-targeting", "A"]
    elif mutation == "wrong_columns":
        kwargs["gene_names"] = ["A", "B", "C"]
    elif mutation == "nonfinite":
        kwargs["expression"] = [[0.0, np.nan], [1.0, 2.0], [3.0, 4.0]]
    elif mutation == "duplicate_genes":
        kwargs["gene_names"] = ["A", "A"]
    elif mutation == "empty_label":
        kwargs["interventions"] = ["non-targeting", "", "A"]
    elif mutation == "nfd_gene":
        kwargs["gene_names"] = ["A", unicodedata.normalize("NFD", "É")]
    else:
        kwargs["interventions"] = ["A", "A", "A"]
    _write_input(source, **kwargs)

    completed, destination, _ = _run_causalbench(
        tmp_path,
        input_path=source,
        check=False,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not destination.exists()


def test_npz_contract_rejects_missing_array_object_dtype_and_non_regular_paths(
    tmp_path: Path,
) -> None:
    fake_modules = _fake_causalbench(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_modules)
    env["FAKE_CALL_RECORD"] = str(tmp_path / "unused.json")
    cases: list[Path] = []
    missing = tmp_path / "missing.npz"
    np.savez(missing, expression_matrix=np.ones((1, 2)), var_names=np.asarray(["A", "B"]))
    cases.append(missing)
    object_dtype = tmp_path / "object.npz"
    np.savez(
        object_dtype,
        expression_matrix=np.ones((1, 2)),
        interventions=np.asarray(["non-targeting"], dtype=object),
        var_names=np.asarray(["A", "B"]),
    )
    cases.append(object_dtype)
    directory = tmp_path / "a_directory"
    directory.mkdir()
    cases.append(directory)
    valid = tmp_path / "valid.npz"
    _write_input(valid)
    symlink = tmp_path / "linked.npz"
    symlink.symlink_to(valid)
    cases.append(symlink)

    for index, source in enumerate(cases):
        destination = tmp_path / f"output_{index}.csv"
        completed = subprocess.run(
            [
                sys.executable,
                str(CAUSALBENCH_WORKER),
                "--input-npz",
                str(source),
                "--output-csv",
                str(destination),
                "--model-name",
                "pc",
                "--training-information",
                "observational",
                "--seed",
                "1",
                "--output-semantics",
                "official_return_order",
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert not destination.exists()


def test_oversized_sparse_input_is_rejected_before_numpy_reads_it(tmp_path: Path) -> None:
    source = tmp_path / "oversized.npz"
    with source.open("wb") as handle:
        handle.truncate(512 * 1024 * 1024 + 1)

    completed, destination, _ = _run_causalbench(
        tmp_path,
        input_path=source,
        check=False,
    )

    assert completed.returncode != 0
    assert "unusually large" in completed.stderr
    assert not destination.exists()


def test_output_must_not_preexist_or_be_a_symbolic_link(tmp_path: Path) -> None:
    existing = tmp_path / "existing.csv"
    existing.write_text("keep me", encoding="utf-8")
    completed, _, record = _run_causalbench(
        tmp_path / "existing_case",
        output_path=existing,
        check=False,
    )
    assert completed.returncode != 0
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert not record.exists()

    target = tmp_path / "target.csv"
    target.write_text("keep target", encoding="utf-8")
    linked = tmp_path / "linked.csv"
    linked.symlink_to(target)
    completed, _, record = _run_causalbench(
        tmp_path / "symlink_case",
        output_path=linked,
        check=False,
    )
    assert completed.returncode != 0
    assert linked.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep target"
    assert not record.exists()


def test_psgrn_source_constants_and_environment_match_the_registry() -> None:
    source = PSGRN_WORKER.read_text(encoding="utf-8")
    environment = (ROOT / "envs/task_c/psgrn.yml").read_text(encoding="utf-8")
    registry = json.loads(
        (ROOT / "configs/task_c_methods_v1.json").read_text(encoding="utf-8")
    )

    assert EXPECTED_PSGRN_COMMIT in source
    assert registry["methods"]["guanlab_psgrn"]["commit"] == EXPECTED_PSGRN_COMMIT
    assert "name: hypersca-task-c-psgrn" in environment
    assert "python=3.10" in environment
    assert f"causalbench.git@{EXPECTED_CAUSALBENCH_COMMIT}" in environment
    assert registry["causalbench"]["commit"] == EXPECTED_CAUSALBENCH_COMMIT


def test_psgrn_uses_clean_git_source_and_limits_output_to_first_1000(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, revision = _make_psgrn_repository(tmp_path)
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "psgrn.csv"
    record_path = tmp_path / "psgrn_call.json"
    _write_input(input_path)
    worker = _load_worker(PSGRN_WORKER, "test_psgrn_worker_contract")
    monkeypatch.setattr(worker, "EXPECTED_PSGRN_COMMIT", revision)
    monkeypatch.setenv("FAKE_CALL_RECORD", str(record_path))

    worker.main(
        [
            "--input-npz",
            str(input_path),
            "--output-csv",
            str(output_path),
            "--psgrn-source",
            str(source),
            "--training-information",
            "partial_interventional",
            "--seed",
            "23",
            "--output-semantics",
            "official_return_order",
        ]
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["shape"] == [3, 2]
    assert record["interventions"] == ["non-targeting", "A", "non-targeting"]
    assert record["regime_name"] == "PartialIntervational"
    assert record["seed"] == 23
    assert record["expression_writeable"] is False
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1_000
    assert float(rows[0]["score"]) == 1_000.0
    assert float(rows[-1]["score"]) == 1.0


def test_psgrn_rejects_wrong_revision_dirty_tree_and_source_symlink(
    tmp_path: Path,
) -> None:
    source, revision = _make_psgrn_repository(tmp_path)
    worker = _load_worker(PSGRN_WORKER, "test_psgrn_worker_source_checks")

    with pytest.raises(SystemExit, match="revision"):
        worker.validate_psgrn_source(source, "0" * 40)

    (source / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(SystemExit, match="clean"):
        worker.validate_psgrn_source(source, revision)
    (source / "untracked.txt").unlink()

    linked = tmp_path / "linked_psgrn"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(SystemExit, match="symbolic link"):
        worker.validate_psgrn_source(linked, revision)


@pytest.mark.parametrize("return_kind", ["set", "mapping", "generator", "unknown_gene"])
def test_psgrn_invalid_output_fails_closed_without_partial_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, return_kind: str
) -> None:
    source, revision = _make_psgrn_repository(tmp_path)
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "psgrn.csv"
    _write_input(input_path)
    worker = _load_worker(PSGRN_WORKER, f"test_psgrn_invalid_{return_kind}")
    monkeypatch.setattr(worker, "EXPECTED_PSGRN_COMMIT", revision)
    monkeypatch.setenv("FAKE_CALL_RECORD", str(tmp_path / "call.json"))
    monkeypatch.setenv("FAKE_RETURN_KIND", return_kind)

    with pytest.raises(SystemExit, match="failed_invalid_output"):
        worker.main(
            [
                "--input-npz",
                str(input_path),
                "--output-csv",
                str(output_path),
                "--psgrn-source",
                str(source),
                "--training-information",
                "partial_interventional",
                "--seed",
                "23",
                "--output-semantics",
                "official_return_order",
            ]
        )
    assert not output_path.exists()
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
