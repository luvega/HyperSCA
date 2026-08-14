import json
import importlib.util
import subprocess
import sys
import types
import hashlib
from pathlib import Path

import numpy as np

from src.evaluation.task_c_data import build_task_c_reference_provenance, sha256_path


ROOT = Path(__file__).resolve().parents[1]


def test_export_command_reports_pinned_source_without_downloading(tmp_path):
    data_dir = tmp_path / "raw"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_causalbench_data.py"),
            "--data-dir",
            str(data_dir),
            "--describe-only",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    description = json.loads(result.stdout)
    assert description["repository"] == "https://github.com/causalbench/causalbench.git"
    assert description["commit"] == "1a2143cffdc85f835b41ce8d52034be1bf903e71"
    assert description["datasets"] == ["dataset_k562.npz", "dataset_rpe1.npz"]
    assert description["references"] == [
        "reference_k562_pooled.csv",
        "reference_k562_chipseq.csv",
        "reference_rpe1_pooled.csv",
        "reference_rpe1_chipseq.csv",
    ]
    assert not data_dir.exists()


def test_describe_only_reports_filtered_dataset_names(tmp_path):
    data_dir = tmp_path / "filtered"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_causalbench_data.py"),
            "--data-dir",
            str(data_dir),
            "--filter",
            "--describe-only",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    description = json.loads(result.stdout)
    assert description["datasets"] == [
        "dataset_k562_filtered.npz",
        "dataset_rpe1_filtered.npz",
    ]
    assert description["filter"] is True
    assert not data_dir.exists()


def test_operational_export_uses_stubs_and_records_reproducible_artifacts(
    tmp_path, monkeypatch
):
    script_path = ROOT / "scripts" / "export_causalbench_data.py"
    spec = importlib.util.spec_from_file_location("export_causalbench_data", script_path)
    exporter = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(exporter)

    calls = {"dataset": [], "evaluations": []}

    class FakeDataset:
        def __init__(self, data_dir, use_filter):
            calls["dataset"].append((data_dir, use_filter))
            self.data_dir = Path(data_dir)
            self.use_filter = use_filter

        def load(self):
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / "k562.h5ad").write_bytes(b"fake-k562")
            (self.data_dir / "rpe1.h5ad").write_bytes(b"fake-rpe1")
            paths = []
            for name in ("k562", "rpe1"):
                suffix = "_filtered" if self.use_filter else ""
                path = self.data_dir / f"dataset_{name}{suffix}.npz"
                path.write_bytes(f"{name}:{self.use_filter}".encode())
                paths.append(str(path))
            return paths

    class FakeEvaluations:
        def __init__(self, data_dir, dataset_name):
            calls["evaluations"].append((data_dir, dataset_name))

        def load(self):
            return (
                {("a", "b"), ("a", "a")},
                {("b", "c")},
                {("c", "d")},
                {("d", "e")},
                {("z", "a"), ("b", "a"), ("z", "z")},
            )

    package = types.ModuleType("causalscbench")
    data_access = types.ModuleType("causalscbench.data_access")
    dataset_module = types.ModuleType("causalscbench.data_access.create_dataset")
    evaluations_module = types.ModuleType(
        "causalscbench.data_access.create_evaluation_datasets"
    )
    dataset_module.CreateDataset = FakeDataset
    evaluations_module.CreateEvaluationDatasets = FakeEvaluations
    monkeypatch.setitem(sys.modules, "causalscbench", package)
    monkeypatch.setitem(sys.modules, "causalscbench.data_access", data_access)
    monkeypatch.setitem(sys.modules, dataset_module.__name__, dataset_module)
    monkeypatch.setitem(sys.modules, evaluations_module.__name__, evaluations_module)

    data_dir = tmp_path / "raw"
    assert exporter.main(["--data-dir", str(data_dir), "--filter"]) == 0

    assert calls["dataset"] == [(str(data_dir), True)]
    assert [name for _, name in calls["evaluations"]] == [
        "weissmann_k562",
        "weissmann_rpe1",
    ]
    assert (data_dir / "dataset_k562_filtered.npz").exists()
    assert (data_dir / "dataset_rpe1_filtered.npz").exists()
    assert (data_dir / "reference_k562_pooled.csv").read_text() == (
        "source,target\n"
        "a,b\n"
        "a,z\n"
        "b,a\n"
        "b,c\n"
        "c,b\n"
        "c,d\n"
        "d,c\n"
        "d,e\n"
        "e,d\n"
        "z,a\n"
    )
    assert (data_dir / "reference_k562_chipseq.csv").read_text() == (
        "source,target\n"
        "b,a\n"
        "z,a\n"
    )
    assert "a,a" not in (data_dir / "reference_k562_pooled.csv").read_text()
    assert "z,z" not in (data_dir / "reference_k562_chipseq.csv").read_text()
    manifest = json.loads((data_dir / "export_manifest.json").read_text())
    assert manifest["datasets"] == [
        "dataset_k562_filtered.npz",
        "dataset_rpe1_filtered.npz",
    ]
    assert manifest["downloaded_at_utc"] is None
    assert manifest["dropped_self_edges"]["k562"] == {"pooled": 2, "chipseq": 1}
    assert manifest["dropped_self_edges"]["rpe1"] == {"pooled": 2, "chipseq": 1}
    assert manifest["source_sha256"]["k562.h5ad"] == "sha256:" + hashlib.sha256(b"fake-k562").hexdigest()
    assert manifest["source_sha256"]["rpe1.h5ad"] == "sha256:" + hashlib.sha256(b"fake-rpe1").hexdigest()
    assert manifest["exported_at_utc"]
    assert set(manifest["sha256"]) == {
        "dataset_k562_filtered.npz",
        "dataset_rpe1_filtered.npz",
        "reference_k562_pooled.csv",
        "reference_k562_chipseq.csv",
        "reference_rpe1_pooled.csv",
        "reference_rpe1_chipseq.csv",
    }
    assert all(value.startswith("sha256:") for value in manifest["sha256"].values())
    assert all(not Path(path).is_absolute() for path in manifest["sha256"])
    for context in ("k562", "rpe1"):
        build_task_c_reference_provenance(
            context_id=context,
            pooled_path=data_dir / f"reference_{context}_pooled.csv",
            chipseq_path=data_dir / f"reference_{context}_chipseq.csv",
        )


def _write_cli_dataset(path: Path) -> None:
    genes = np.asarray(["A", "B", "C", "D", "E", "Z"])
    labels = ["non-targeting"] * 10
    for gene in genes[:5]:
        labels.extend([str(gene)] * 5)
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    np.savez(
        path,
        expression_matrix=expression,
        interventions=np.asarray(labels),
        var_names=genes,
    )


def _prepare_cli_inputs(tmp_path: Path):
    k562 = tmp_path / "k562.npz"
    rpe1 = tmp_path / "rpe1.npz"
    _write_cli_dataset(k562)
    _write_cli_dataset(rpe1)
    references = {}
    for context in ("k562", "rpe1"):
        for reference_id in ("pooled", "chipseq"):
            path = tmp_path / f"reference_{context}_{reference_id}.csv"
            rows = (
                "source,target\nA,B\nB,A\n"
                if reference_id == "pooled"
                else "source,target\nA,B\n"
            )
            path.write_text(rows, encoding="utf-8")
            references[f"{context}_{reference_id}"] = path
    output = tmp_path / "prepared"
    command = [
        sys.executable,
        str(ROOT / "scripts/prepare_task_c_data.py"),
        "--k562-npz",
        str(k562),
        "--rpe1-npz",
        str(rpe1),
        "--k562-pooled-reference",
        str(references["k562_pooled"]),
        "--k562-chipseq-reference",
        str(references["k562_chipseq"]),
        "--rpe1-pooled-reference",
        str(references["rpe1_pooled"]),
        "--rpe1-chipseq-reference",
        str(references["rpe1_chipseq"]),
        "--output-dir",
        str(output),
        "--min-cells-per-intervention",
        "5",
    ]
    return k562, rpe1, references, output, command


def _output_snapshot(output: Path) -> dict[str, bytes]:
    return {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }


def test_prepare_cli_writes_five_reproducible_splits(tmp_path: Path) -> None:
    _, _, references, output, command = _prepare_cli_inputs(tmp_path)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "prepared"
    assert [entry["seed"] for entry in summary["splits"]] == [11, 23, 47, 71, 97]
    for entry in summary["splits"]:
        public = json.loads(Path(entry["public_manifest"]).read_text(encoding="utf-8"))
        encoded_identity = json.dumps(
            public["materialization_identity"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert entry["materialization_identity_sha256"] == (
            "sha256:" + hashlib.sha256(encoded_identity).hexdigest()
        )
    for seed in (11, 23, 47, 71, 97):
        assert (output / "splits" / f"seed_{seed}" / "public_manifest.json").exists()
    assert (output / "provenance" / "k562.json").exists()
    assert (output / "provenance" / "rpe1.json").exists()
    assert (output / "provenance" / "k562_references.json").exists()
    assert (output / "provenance" / "rpe1_references.json").exists()
    for context in ("k562", "rpe1"):
        provenance = json.loads(
            (output / "provenance" / f"{context}_references.json").read_text(
                encoding="utf-8"
            )
        )
        assert provenance["context"] == context
        assert provenance["context_id"] == context
        for reference_id in ("pooled", "chipseq"):
            source = references[f"{context}_{reference_id}"].resolve()
            assert provenance["files"][reference_id]["path"] == str(source)
            assert provenance["files"][reference_id]["sha256"] == sha256_path(source)


def test_prepare_cli_changed_dataset_does_not_modify_existing_output(tmp_path: Path) -> None:
    k562, _, _, output, command = _prepare_cli_inputs(tmp_path)
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    before = _output_snapshot(output)
    with np.load(k562, allow_pickle=False) as archive:
        expression = archive["expression_matrix"].copy()
        interventions = archive["interventions"].copy()
        genes = archive["var_names"].copy()
    expression[0, 0] += 1
    np.savez(
        k562,
        expression_matrix=expression,
        interventions=interventions,
        var_names=genes,
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert _output_snapshot(output) == before


def test_prepare_cli_changed_reference_does_not_modify_existing_output(tmp_path: Path) -> None:
    _, _, references, output, command = _prepare_cli_inputs(tmp_path)
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    before = _output_snapshot(output)
    references["k562_pooled"].write_text(
        "source,target\nA,B\nB,A\nC,D\nD,C\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert _output_snapshot(output) == before
