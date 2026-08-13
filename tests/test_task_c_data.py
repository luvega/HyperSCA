import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.task_c_data import (
    CAUSALBENCH_COMMIT,
    TaskCDataError,
    build_task_c_provenance,
    build_task_c_reference_provenance,
    load_task_c_dataset,
    write_json,
)


def write_dataset(path: Path, genes: list[str], labels: list[str]) -> None:
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    np.savez(path, expression_matrix=expression,
             interventions=np.asarray(labels), var_names=np.asarray(genes))


def write_reference(path: Path, rows: str = "source,target\nA,B\nB,C\n") -> None:
    path.write_text(rows, encoding="utf-8")


def test_valid_k562_dataset_loads(tmp_path):
    path = tmp_path / "dataset.npz"
    write_dataset(path, ["A", "B", "C"], ["non-targeting", "A", "B"])
    dataset = load_task_c_dataset(path, context_id="k562")
    assert dataset.expression.shape == (3, 3)
    assert dataset.gene_names == ("A", "B", "C")
    assert dataset.interventions.tolist() == ["non-targeting", "A", "B"]


def test_duplicate_genes_are_rejected(tmp_path):
    path = tmp_path / "dataset.npz"
    write_dataset(path, ["A", "A"], ["non-targeting"])
    with pytest.raises(TaskCDataError, match="gene names must be unique"):
        load_task_c_dataset(path, context_id="k562")


def test_provenance_is_pinned_and_json_safe(tmp_path):
    path = tmp_path / "dataset.npz"
    write_dataset(path, ["A"], ["non-targeting"])
    dataset = load_task_c_dataset(path, context_id="k562")
    provenance = build_task_c_provenance(dataset)
    assert provenance["context"] == "k562"
    assert provenance["commit"] == CAUSALBENCH_COMMIT
    assert provenance["input_sha256"].startswith("sha256:")
    json.dumps(provenance, allow_nan=False)


def test_reference_provenance_distinguishes_primary_and_directed_sources(tmp_path):
    pooled = tmp_path / "pooled.csv"
    chipseq = tmp_path / "chipseq.csv"
    write_reference(pooled)
    write_reference(chipseq)
    provenance = build_task_c_reference_provenance(
        context_id="k562", pooled_path=pooled, chipseq_path=chipseq
    )
    assert provenance["primary_evidence"]["id"] == "causalbench_pooled_biological_v1"
    assert provenance["directed_evidence"]["id"] == "causalbench_chipseq_v1"
    assert provenance["files"]["pooled"]["sha256"].startswith("sha256:")
    assert provenance["files"]["chipseq"]["sha256"].startswith("sha256:")


def test_missing_array_is_rejected(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, expression_matrix=np.ones((1, 1)), var_names=np.asarray(["A"]))
    with pytest.raises(TaskCDataError, match="interventions"):
        load_task_c_dataset(path, context_id="k562")


def test_nonfinite_expression_is_rejected(tmp_path):
    path = tmp_path / "nonfinite.npz"
    np.savez(path, expression_matrix=np.asarray([[np.nan]]),
             interventions=np.asarray(["non-targeting"]), var_names=np.asarray(["A"]))
    with pytest.raises(TaskCDataError, match="finite"):
        load_task_c_dataset(path, context_id="k562")


def test_empty_intervention_label_is_rejected(tmp_path):
    path = tmp_path / "empty-label.npz"
    write_dataset(path, ["A"], ["non-targeting"])
    np.savez(path, expression_matrix=np.ones((2, 1)),
             interventions=np.asarray(["non-targeting", ""]), var_names=np.asarray(["A"]))
    with pytest.raises(TaskCDataError, match="intervention labels"):
        load_task_c_dataset(path, context_id="k562")


def test_malformed_reference_csv_is_rejected(tmp_path):
    pooled = tmp_path / "pooled.csv"
    chipseq = tmp_path / "chipseq.csv"
    write_reference(pooled, "source,target\nA,A\n")
    write_reference(chipseq)
    with pytest.raises(TaskCDataError, match="self"):
        build_task_c_reference_provenance(
            context_id="k562", pooled_path=pooled, chipseq_path=chipseq
        )


def test_write_json_is_sorted_and_utf8(tmp_path):
    path = tmp_path / "nested" / "provenance.json"
    write_json(path, {"z": "终", "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "z": "终"}
    assert path.read_text(encoding="utf-8").endswith("\n")
