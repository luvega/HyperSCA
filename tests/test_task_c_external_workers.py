from __future__ import annotations

import csv
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unicodedata
import uuid

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CAUSALBENCH_WORKER = ROOT / "scripts/task_c_workers/causalbench_worker.py"
PSGRN_WORKER = ROOT / "scripts/task_c_workers/psgrn_worker.py"
EXPECTED_CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
EXPECTED_PSGRN_COMMIT = "74aa640f7c472b23a69811f6795bb17678efd344"
PROVEN_ORDER_MODELS = {"grnboost"}
OFFICIAL_UNRANKED_MODELS = {
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
OBSERVATIONAL_MODELS = {
    "random1000",
    "grnboost",
    "pc",
    "ges",
    "gsp",
    "notears-lin-sparse",
    "sortnregress",
}
PARTIAL_INTERVENTIONAL_MODELS = {
    "gies",
    "igsp",
    "DCDI-G",
    "DCDI-DSF",
    "DCDFG-LIN",
    "DCDFG-MLP",
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


def _fake_causalbench(
    tmp_path: Path, *, failing_modules: set[str] | None = None
) -> tuple[Path, str]:
    package = tmp_path / "causalbench-fixed-test-source"
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
                if kind == "list_reversed":
                    return list(reversed(edges))
                if kind == "tuple":
                    return tuple(edges)
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
        if failing_modules and filename in failing_modules:
            _write_python(
                models / filename,
                f"raise RuntimeError('unselected module imported: {filename}')\n",
            )
            continue
        classes = "\n".join(f"class {name}(Base):\n    pass" for name in class_names)
        _write_python(
            models / filename,
            f"from causalscbench.models.fake_base import Base\n\n{classes}\n",
        )
    subprocess.run(["git", "init", "-q", str(package)], check=True)
    subprocess.run(
        ["git", "-C", str(package), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(package), "config", "user.name", "Test only"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(package),
            "remote",
            "add",
            "origin",
            "https://github.com/causalbench/causalbench.git",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(package), "add", "causalscbench"], check=True)
    subprocess.run(
        ["git", "-C", str(package), "commit", "-q", "-m", "test interface"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(package), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return package, revision


def _run_causalbench(
    tmp_path: Path,
    *,
    model_name: str = "grnboost",
    training_information: str = "observational",
    input_path: Path | None = None,
    output_path: Path | None = None,
    return_kind: str = "list",
    output_semantics: str | None = None,
    failing_modules: set[str] | None = None,
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    causalbench_source, revision = _fake_causalbench(
        tmp_path, failing_modules=failing_modules
    )
    source = input_path or tmp_path / "input.npz"
    if input_path is None:
        if model_name == "random1000":
            random_genes = ["A", "B", *[f"G{index}" for index in range(31)]]
            _write_input(
                source,
                expression=np.zeros((3, len(random_genes))),
                gene_names=random_genes,
            )
        else:
            _write_input(source)
    destination = output_path or tmp_path / "predictions.csv"
    record = tmp_path / "call.json"
    worker = _load_worker(
        CAUSALBENCH_WORKER,
        f"test_causalbench_worker_{uuid.uuid4().hex}",
    )
    worker.EXPECTED_CAUSALBENCH_COMMIT = revision
    if output_semantics is None:
        output_semantics = (
            "official_return_order"
            if model_name in PROVEN_ORDER_MODELS
            else "official_unranked_edges"
        )
    arguments = [
        "--input-npz",
        str(source),
        "--output-csv",
        str(destination),
        "--model-name",
        model_name,
        "--causalbench-source",
        str(causalbench_source),
        "--training-information",
        training_information,
        "--seed",
        "17",
        "--output-semantics",
        output_semantics,
    ]
    old_record = os.environ.get("FAKE_CALL_RECORD")
    old_return = os.environ.get("FAKE_RETURN_KIND")
    os.environ["FAKE_CALL_RECORD"] = str(record)
    os.environ["FAKE_RETURN_KIND"] = return_kind
    stderr = io.StringIO()
    return_code = 0
    try:
        with redirect_stderr(stderr):
            try:
                worker.main(arguments)
            except SystemExit as exc:
                return_code = int(exc.code) if isinstance(exc.code, int) else 1
                if not isinstance(exc.code, int) and exc.code:
                    stderr.write(f"{exc.code}\n")
    finally:
        if old_record is None:
            os.environ.pop("FAKE_CALL_RECORD", None)
        else:
            os.environ["FAKE_CALL_RECORD"] = old_record
        if old_return is None:
            os.environ.pop("FAKE_RETURN_KIND", None)
        else:
            os.environ["FAKE_RETURN_KIND"] = old_return
    completed = subprocess.CompletedProcess(
        args=[str(CAUSALBENCH_WORKER), *arguments],
        returncode=return_code,
        stdout="",
        stderr=stderr.getvalue(),
    )
    if check and return_code:
        raise subprocess.CalledProcessError(
            return_code,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed, destination, record


def _load_worker(path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_psgrn_repository(
    tmp_path: Path, *, source_text: str | None = None
) -> tuple[Path, str]:
    source = tmp_path / "PSGRN-test-interface-only"
    source.mkdir()
    _write_python(
        source / "src/main.py",
        source_text
        or """
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
        if worker == CAUSALBENCH_WORKER:
            assert "--causalbench-source" in completed.stdout
        else:
            assert "--psgrn-source" in completed.stdout


@pytest.mark.parametrize(
    ("model_name", "wrong_training_information"),
    [
        *[(model, "partial_interventional") for model in sorted(OBSERVATIONAL_MODELS)],
        *[(model, "observational") for model in sorted(PARTIAL_INTERVENTIONAL_MODELS)],
    ],
)
def test_causalbench_rejects_training_information_mismatch_before_input_or_model(
    tmp_path: Path, model_name: str, wrong_training_information: str
) -> None:
    missing_input = tmp_path / "must-not-be-read.npz"
    completed, destination, record = _run_causalbench(
        tmp_path,
        model_name=model_name,
        training_information=wrong_training_information,
        input_path=missing_input,
        check=False,
    )

    assert completed.returncode != 0
    assert "training information" in completed.stderr
    assert "does not match" in completed.stderr
    assert not record.exists()
    assert not destination.exists()


def test_psgrn_rejects_observational_request_before_source_or_input(
    tmp_path: Path,
) -> None:
    worker = _load_worker(PSGRN_WORKER, "test_psgrn_training_boundary")
    output_path = tmp_path / "output.csv"

    with pytest.raises(SystemExit, match="training information.*does not match"):
        worker.main(
            [
                "--input-npz",
                str(tmp_path / "missing.npz"),
                "--output-csv",
                str(output_path),
                "--psgrn-source",
                str(tmp_path / "missing-source"),
                "--training-information",
                "observational",
                "--seed",
                "1",
                "--output-semantics",
                "official_return_order",
            ]
        )
    assert not output_path.exists()


def test_causalbench_source_must_be_fixed_clean_official_and_not_a_symlink(
    tmp_path: Path,
) -> None:
    source, revision = _fake_causalbench(tmp_path)
    worker = _load_worker(CAUSALBENCH_WORKER, "test_causalbench_source_checks")

    assert worker.validate_causalbench_source(source, revision) == source.resolve()
    with pytest.raises(SystemExit, match="revision"):
        worker.validate_causalbench_source(source, "0" * 40)

    (source / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(SystemExit, match="clean"):
        worker.validate_causalbench_source(source, revision)
    (source / "untracked.txt").unlink()

    subprocess.run(
        ["git", "-C", str(source), "remote", "set-url", "origin", "https://example.invalid/fork.git"],
        check=True,
    )
    with pytest.raises(SystemExit, match="registered source"):
        worker.validate_causalbench_source(source, revision)

    linked = tmp_path / "linked-causalbench"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(SystemExit, match="symbolic link"):
        worker.validate_causalbench_source(linked, revision)


def test_causalbench_rejects_a_tracked_source_file_symlink(tmp_path: Path) -> None:
    source, _ = _fake_causalbench(tmp_path)
    selected_module = source / "causalscbench/models/causallearn_models.py"
    selected_module.unlink()
    selected_module.symlink_to("fake_base.py")
    subprocess.run(["git", "-C", str(source), "add", "causalscbench"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "tracked source symlink"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worker = _load_worker(CAUSALBENCH_WORKER, "test_causalbench_tracked_symlink")

    with pytest.raises(SystemExit, match="source.*symbolic link"):
        worker.validate_causalbench_source(source, revision)


@pytest.mark.parametrize("injection", ["replace", "grafts"])
def test_causalbench_rejects_git_history_rewrite_mechanisms(
    tmp_path: Path, injection: str
) -> None:
    source, revision = _fake_causalbench(tmp_path)
    worker = _load_worker(CAUSALBENCH_WORKER, f"test_causalbench_{injection}")
    if injection == "replace":
        subprocess.run(
            ["git", "-C", str(source), "update-ref", f"refs/replace/{revision}", revision],
            check=True,
        )
    else:
        grafts = source / ".git/info/grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(f"{revision}\n", encoding="ascii")

    with pytest.raises(SystemExit, match=injection):
        worker.validate_causalbench_source(source, revision)


def test_causalbench_git_checks_disable_replace_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, revision = _fake_causalbench(tmp_path)
    worker = _load_worker(CAUSALBENCH_WORKER, "test_causalbench_git_environment")
    real_run = subprocess.run
    calls: list[tuple[list[str], dict[str, str]]] = []

    def recording_run(command: list[str], **kwargs: object) -> object:
        calls.append((command, dict(kwargs.get("env", {}))))
        return real_run(command, **kwargs)

    monkeypatch.setattr(worker.subprocess, "run", recording_run)
    worker.validate_causalbench_source(source, revision)

    assert calls
    assert all("--no-replace-objects" in command for command, _ in calls)
    assert all(env.get("GIT_NO_REPLACE_OBJECTS") == "1" for _, env in calls)


def test_causalbench_imports_only_the_selected_method_module(tmp_path: Path) -> None:
    unselected = {
        "arboreto_baselines.py",
        "dcdi_models.py",
        "gies.py",
        "notears.py",
        "random_network.py",
        "sparsest_permutations.py",
        "varsortability.py",
    }
    completed, destination, _ = _run_causalbench(
        tmp_path,
        model_name="pc",
        failing_modules=unselected,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert destination.exists()


def test_causalbench_ignores_pythonpath_and_user_site_package_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malicious = tmp_path / "malicious-pythonpath"
    _write_python(
        malicious / "causalscbench/__init__.py",
        "raise RuntimeError('unverified causalscbench was imported')",
    )
    monkeypatch.setenv("PYTHONPATH", str(malicious))
    monkeypatch.syspath_prepend(str(malicious))

    completed, destination, _ = _run_causalbench(
        tmp_path / "run",
        model_name="pc",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert destination.exists()


def test_causalbench_run_keeps_the_verified_checkout_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, revision = _fake_causalbench(tmp_path)
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "output.csv"
    record_path = tmp_path / "call.json"
    _write_input(input_path)
    worker = _load_worker(CAUSALBENCH_WORKER, "test_causalbench_no_bytecode")
    monkeypatch.setattr(worker, "EXPECTED_CAUSALBENCH_COMMIT", revision)
    monkeypatch.setenv("FAKE_CALL_RECORD", str(record_path))

    worker.main(
        [
            "--input-npz",
            str(input_path),
            "--output-csv",
            str(output_path),
            "--model-name",
            "pc",
            "--causalbench-source",
            str(source),
            "--training-information",
            "observational",
            "--seed",
            "17",
            "--output-semantics",
            "official_unranked_edges",
        ]
    )

    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""


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
    assert completed.returncode == 0, completed.stderr
    assert destination.exists()
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if model_name in OFFICIAL_UNRANKED_MODELS:
        assert {float(row["score"]) for row in rows} == {1.0}
    else:
        assert model_name in PROVEN_ORDER_MODELS


def test_worker_has_a_fixed_ranked_and_unranked_semantics_boundary() -> None:
    worker = _load_worker(CAUSALBENCH_WORKER, "test_causalbench_order_boundary")

    assert worker.MODEL_OUTPUT_SEMANTICS == {
        **{model: "official_return_order" for model in PROVEN_ORDER_MODELS},
        **{
            model: "official_unranked_edges"
            for model in OFFICIAL_UNRANKED_MODELS
        },
    }
    assert set(worker.MODEL_NAMES) == PROVEN_ORDER_MODELS | OFFICIAL_UNRANKED_MODELS


@pytest.mark.parametrize("model_name", sorted(OFFICIAL_UNRANKED_MODELS))
def test_causalbench_unranked_methods_assign_equal_positive_scores(
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

    assert completed.returncode == 0, completed.stderr
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [float(row["score"]) for row in rows] == [1.0, 1.0, 1.0]


def test_unranked_scores_do_not_depend_on_the_official_list_permutation(
    tmp_path: Path,
) -> None:
    _, forward_path, _ = _run_causalbench(
        tmp_path / "forward", model_name="pc", return_kind="list"
    )
    _, reverse_path, _ = _run_causalbench(
        tmp_path / "reverse", model_name="pc", return_kind="list_reversed"
    )

    with forward_path.open(encoding="utf-8", newline="") as handle:
        forward = list(csv.DictReader(handle))
    with reverse_path.open(encoding="utf-8", newline="") as handle:
        reverse = list(csv.DictReader(handle))
    assert {
        (row["source"], row["target"], float(row["score"])) for row in forward
    } == {
        (row["source"], row["target"], float(row["score"])) for row in reverse
    }


def test_unranked_tuple_relations_receive_the_same_positive_score(
    tmp_path: Path,
) -> None:
    completed, destination, _ = _run_causalbench(
        tmp_path,
        model_name="pc",
        return_kind="tuple",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["score"]) for row in rows] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("model_name", "output_semantics"),
    [
        ("grnboost", "official_unranked_edges"),
        ("pc", "official_return_order"),
    ],
)
def test_semantics_mismatch_fails_before_the_model_is_called(
    tmp_path: Path, model_name: str, output_semantics: str
) -> None:
    completed, destination, record = _run_causalbench(
        tmp_path,
        model_name=model_name,
        output_semantics=output_semantics,
        check=False,
    )

    assert completed.returncode != 0
    assert "failed_invalid_output" in completed.stderr
    assert "does not match" in completed.stderr
    assert not record.exists()
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
    assert completed.returncode == 0, completed.stderr
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {float(row["score"]) for row in rows} == {1.0}


@pytest.mark.parametrize("model_name", ["grnboost", "pc"])
@pytest.mark.parametrize("return_kind", ["set", "mapping", "generator"])
def test_ambiguous_causalbench_return_order_fails_closed(
    tmp_path: Path, return_kind: str, model_name: str
) -> None:
    completed, destination, _ = _run_causalbench(
        tmp_path,
        model_name=model_name,
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
    ("field", "unsafe_value", "message"),
    [
        ("gene", "B\x00hidden", "control character"),
        ("gene", "B\nnext-row", "control character"),
        ("gene", "=SUM(A1)", "spreadsheet formula"),
        ("gene", "+cmd", "spreadsheet formula"),
        ("gene", "-cmd", "spreadsheet formula"),
        ("gene", "@cmd", "spreadsheet formula"),
        ("intervention", "A\rnext-row", "control character"),
        ("intervention", "=A", "spreadsheet formula"),
    ],
)
def test_npz_text_rejects_control_characters_and_spreadsheet_formula_prefixes(
    tmp_path: Path, field: str, unsafe_value: str, message: str
) -> None:
    source = tmp_path / "unsafe-text.npz"
    if field == "gene":
        _write_input(source, gene_names=["A", unsafe_value])
    else:
        _write_input(
            source,
            interventions=["non-targeting", unsafe_value, "A"],
        )

    completed, destination, record = _run_causalbench(
        tmp_path / "run",
        input_path=source,
        check=False,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not record.exists()
    assert not destination.exists()


def test_random1000_rejects_a_gene_scope_with_fewer_than_1000_relations(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "too-few-genes.npz"
    _write_input(input_path)
    completed, destination, record = _run_causalbench(
        tmp_path / "run",
        model_name="random1000",
        input_path=input_path,
        check=False,
    )

    assert completed.returncode != 0
    assert "at least 1000" in completed.stderr
    assert not record.exists()
    assert not destination.exists()


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


def test_worker_accepts_the_validated_cross_environment_fourth_array(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "derived.npz"
    _write_input(
        input_path,
        extra={
            "environment_labels": np.asarray(["k562", "rpe1", "rpe1"]),
        },
    )

    completed, destination, record = _run_causalbench(
        tmp_path / "run-derived",
        input_path=input_path,
        model_name="grnboost",
    )

    assert completed.returncode == 0
    assert destination.exists()
    assert record.exists()


def test_npz_contract_rejects_missing_array_object_dtype_and_non_regular_paths(
    tmp_path: Path,
) -> None:
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
        completed, _, _ = _run_causalbench(
            tmp_path / f"case_{index}",
            input_path=source,
            output_path=destination,
            model_name="pc",
            check=False,
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


def test_atomic_output_is_bound_to_the_opened_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _load_worker(CAUSALBENCH_WORKER, "test_bound_output_directory")
    parent = tmp_path / "output-parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-original-parent"
    destination = parent / "predictions.csv"
    real_link = os.link
    replaced = False

    def replace_parent_then_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(moved_parent)
            parent.mkdir()
            (parent / "attacker-existing.txt").write_text("keep", encoding="utf-8")
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(worker.os, "link", replace_parent_then_link)
    with pytest.raises(worker.WorkerContractError, match="parent directory changed"):
        worker.write_ranked_csv(destination, [("A", "B", 1.0)])

    assert (parent / "attacker-existing.txt").read_text(encoding="utf-8") == "keep"
    assert not destination.exists()
    assert list(moved_parent.iterdir()) == []


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


def test_causalbench_environment_pins_import_compatible_scientific_packages() -> None:
    environment = (ROOT / "envs/task_c/causalbench.yml").read_text(
        encoding="utf-8"
    )

    for requirement in (
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "pgmpy==1.0.0",
        "umap-learn==0.5.7",
        "setuptools<81",
    ):
        assert requirement in environment


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


@pytest.mark.parametrize("injection", ["replace", "grafts"])
def test_psgrn_rejects_git_history_rewrite_mechanisms(
    tmp_path: Path, injection: str
) -> None:
    source, revision = _make_psgrn_repository(tmp_path)
    worker = _load_worker(PSGRN_WORKER, f"test_psgrn_{injection}")
    if injection == "replace":
        subprocess.run(
            ["git", "-C", str(source), "update-ref", f"refs/replace/{revision}", revision],
            check=True,
        )
    else:
        grafts = source / ".git/info/grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(f"{revision}\n", encoding="ascii")

    with pytest.raises(SystemExit, match=injection):
        worker.validate_psgrn_source(source, revision)


def test_psgrn_git_checks_disable_replace_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, revision = _make_psgrn_repository(tmp_path)
    worker = _load_worker(PSGRN_WORKER, "test_psgrn_git_environment")
    real_run = subprocess.run
    calls: list[tuple[list[str], dict[str, str]]] = []

    def recording_run(command: list[str], **kwargs: object) -> object:
        calls.append((command, dict(kwargs.get("env", {}))))
        return real_run(command, **kwargs)

    monkeypatch.setattr(worker.subprocess, "run", recording_run)
    worker.validate_psgrn_source(source, revision)

    assert calls
    assert all("--no-replace-objects" in command for command, _ in calls)
    assert all(env.get("GIT_NO_REPLACE_OBJECTS") == "1" for _, env in calls)


def test_psgrn_loads_the_verified_commit_bytes_after_path_changes(
    tmp_path: Path,
) -> None:
    source, revision = _make_psgrn_repository(tmp_path)
    worker = _load_worker(PSGRN_WORKER, "test_psgrn_bound_source_bytes")
    verified = worker.validate_psgrn_source(source, revision)
    (source / "src/main.py").write_text(
        "raise RuntimeError('changed after verification')\n",
        encoding="utf-8",
    )

    custom, training_regime = worker._load_custom(verified)

    assert custom.__name__ == "Custom"
    assert training_regime.PartialIntervational.value == "partial_interventional"


def test_psgrn_failed_source_execution_does_not_leave_a_module(
    tmp_path: Path,
) -> None:
    source, revision = _make_psgrn_repository(
        tmp_path,
        source_text="raise RuntimeError('broken fixed source')\n",
    )
    worker = _load_worker(PSGRN_WORKER, "test_psgrn_failed_module_cleanup")
    verified = worker.validate_psgrn_source(source, revision)
    sys.modules.pop("hypersca_fixed_psgrn", None)

    with pytest.raises(SystemExit, match="incompatible"):
        worker._load_custom(verified)

    assert "hypersca_fixed_psgrn" not in sys.modules


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
