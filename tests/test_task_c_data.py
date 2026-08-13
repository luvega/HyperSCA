import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.evaluation.task_c_data import (
    CAUSALBENCH_COMMIT,
    TaskCDataError,
    TaskCDataset,
    TaskCSplit,
    build_shared_task_c_split,
    validate_task_c_split,
    build_task_c_provenance,
    build_task_c_reference_provenance,
    load_task_c_dataset,
    write_json,
)


def dataset_for_split(context_id: str) -> TaskCDataset:
    genes = ("A", "B", "C", "D", "E", "F", "Z")
    labels = ["non-targeting"] * 10
    for source in ("A", "B", "C", "D", "E"):
        labels.extend([source] * 5)
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    return TaskCDataset(
        expression=expression,
        interventions=np.asarray(labels, dtype=str),
        gene_names=genes,
        context_id=context_id,
        source_path=Path(f"{context_id}.npz"),
        source_sha256="sha256:test",
    )


def test_shared_split_is_reproducible_disjoint_and_validated():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    first = build_shared_task_c_split(k562, rpe1, seed=11)
    second = build_shared_task_c_split(k562, rpe1, seed=11)
    assert first == second
    assert (len(first.train_sources), len(first.tune_sources), len(first.holdout_sources)) == (3, 1, 1)
    validate_task_c_split(first, k562, rpe1)


def test_shared_split_rejects_overlapping_source_partitions():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    corrupt = replace(split, tune_sources=(split.train_sources[0],))
    with pytest.raises(TaskCDataError, match="source partitions overlap"):
        validate_task_c_split(corrupt, k562, rpe1)


def test_shared_split_holdout_sources_are_observed_in_both_contexts():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=23)
    assert set(split.holdout_sources) <= set(k562.interventions)
    assert set(split.holdout_sources) <= set(rpe1.interventions)


def test_shared_split_rejects_invalid_seed():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    with pytest.raises(TaskCDataError, match="seed"):
        build_shared_task_c_split(k562, rpe1, seed=13)


def test_shared_split_rejects_wrong_context_order():
    with pytest.raises(TaskCDataError, match="context"):
        build_shared_task_c_split(dataset_for_split("rpe1"), dataset_for_split("k562"), seed=11)


def test_shared_split_rejects_duplicate_control_index():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    controls = dict(split.control_indices)
    controls["k562"] = dict(controls["k562"])
    controls["k562"]["train"] = (controls["k562"]["train"][0],) * 2
    with pytest.raises(TaskCDataError, match="duplicate|control partitions"):
        validate_task_c_split(replace(split, control_indices=controls), k562, rpe1)


def test_shared_split_rejects_intervention_row_in_controls():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    controls = dict(split.control_indices)
    controls["k562"] = dict(controls["k562"])
    controls["k562"]["train"] = (10,) + controls["k562"]["train"][1:]
    with pytest.raises(TaskCDataError, match="control|intervention"):
        validate_task_c_split(replace(split, control_indices=controls), k562, rpe1)


def test_shared_split_nested_control_mappings_are_immutable():
    split = build_shared_task_c_split(dataset_for_split("k562"), dataset_for_split("rpe1"), seed=11)
    with pytest.raises(TypeError):
        split.control_indices["k562"]["train"] = ()
    with pytest.raises(TypeError):
        split.control_indices["k562"] = {}


def test_task_c_split_defensively_copies_replace_control_mapping():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    original = {context: dict(values) for context, values in split.control_indices.items()}
    replaced = replace(split, control_indices=original)
    validate_task_c_split(replaced, k562, rpe1)
    original["k562"]["train"] = ()
    assert replaced.control_indices["k562"]["train"] == split.control_indices["k562"]["train"]
    with pytest.raises(TypeError):
        replaced.control_indices["k562"]["train"] = ()


def test_task_c_split_constructor_defensively_copies_control_mapping():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    original = {context: dict(values) for context, values in split.control_indices.items()}
    constructed = TaskCSplit(
        schema_version=split.schema_version,
        split_id=split.split_id,
        seed=split.seed,
        train_sources=split.train_sources,
        tune_sources=split.tune_sources,
        holdout_sources=split.holdout_sources,
        control_indices=original,
        min_cells_per_intervention=split.min_cells_per_intervention,
    )
    validate_task_c_split(constructed, k562, rpe1)
    original["rpe1"]["holdout"] = ()
    assert constructed.control_indices["rpe1"]["holdout"] == split.control_indices["rpe1"]["holdout"]
    with pytest.raises(TypeError):
        constructed.control_indices["rpe1"] = {}


def test_shared_split_rejects_source_reassignment_preserving_union():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    train = list(split.train_sources)
    holdout = list(split.holdout_sources)
    train[0], holdout[0] = holdout[0], train[0]
    corrupt = replace(split, train_sources=tuple(train), holdout_sources=tuple(holdout))
    with pytest.raises(TaskCDataError, match="deterministic|expected source"):
        validate_task_c_split(corrupt, k562, rpe1)


def test_shared_split_rejects_control_reassignment_preserving_union():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    partitions = {context: dict(values) for context, values in split.control_indices.items()}
    train = list(partitions["k562"]["train"])
    holdout = list(partitions["k562"]["holdout"])
    train[0], holdout[0] = holdout[0], train[0]
    partitions["k562"] = {"train": tuple(train), "tune": partitions["k562"]["tune"], "holdout": tuple(holdout)}
    corrupt = replace(split, control_indices=MappingProxyType({
        context: MappingProxyType(values) for context, values in partitions.items()
    }))
    with pytest.raises(TaskCDataError, match="deterministic|expected control"):
        validate_task_c_split(corrupt, k562, rpe1)


def test_shared_split_rejects_fewer_than_five_common_sources():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    rpe1 = replace(rpe1, interventions=np.asarray(["non-targeting"] * 10 + [source for source in ("A", "B", "C", "D") for _ in range(5)]))
    with pytest.raises(TaskCDataError, match="5 shared"):
        build_shared_task_c_split(k562, rpe1, seed=11)


def test_shared_split_rejects_fewer_than_five_controls():
    k562, rpe1 = dataset_for_split("k562"), dataset_for_split("rpe1")
    k562 = replace(k562, interventions=np.asarray(["non-targeting"] * 4 + [source for source in ("A", "B", "C", "D", "E") for _ in range(5)]))
    with pytest.raises(TaskCDataError, match="5 control"):
        build_shared_task_c_split(k562, rpe1, seed=11)


def write_dataset(path: Path, genes: list[str], labels: list[str]) -> None:
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    np.savez(path, expression_matrix=expression,
             interventions=np.asarray(labels), var_names=np.asarray(genes))


def write_reference(path: Path, rows: str = "source,target\nA,B\nB,A\n") -> None:
    path.write_text(rows, encoding="utf-8")


def test_reference_header_only_is_rejected(tmp_path):
    pooled = tmp_path / "pooled.csv"
    chipseq = tmp_path / "chipseq.csv"
    write_reference(pooled, "source,target\n")
    write_reference(chipseq)
    with pytest.raises(TaskCDataError, match="at least one"):
        build_task_c_reference_provenance(context_id="k562", pooled_path=pooled, chipseq_path=chipseq)


def test_reference_extra_fields_are_rejected(tmp_path):
    pooled = tmp_path / "pooled.csv"
    chipseq = tmp_path / "chipseq.csv"
    write_reference(pooled, "source,target,extra\nA,B,C\n")
    write_reference(chipseq)
    with pytest.raises(TaskCDataError, match="exactly"):
        build_task_c_reference_provenance(context_id="k562", pooled_path=pooled, chipseq_path=chipseq)


def test_reference_whitespace_endpoint_is_rejected(tmp_path):
    pooled = tmp_path / "pooled.csv"
    chipseq = tmp_path / "chipseq.csv"
    write_reference(pooled, "source,target\nA ,B\nB,A\n")
    write_reference(chipseq)
    with pytest.raises(TaskCDataError, match="whitespace"):
        build_task_c_reference_provenance(context_id="k562", pooled_path=pooled, chipseq_path=chipseq)


def test_asymmetric_pooled_reference_is_rejected(tmp_path):
    pooled = tmp_path / "pooled.csv"
    chipseq = tmp_path / "chipseq.csv"
    write_reference(pooled, "source,target\nA,B\n")
    write_reference(chipseq)
    with pytest.raises(TaskCDataError, match="reverse"):
        build_task_c_reference_provenance(context_id="k562", pooled_path=pooled, chipseq_path=chipseq)


def test_bad_npz_archive_is_rejected(tmp_path):
    path = tmp_path / "truncated.npz"
    path.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(TaskCDataError, match="load|archive"):
        load_task_c_dataset(path, context_id="k562")


@pytest.mark.parametrize("field", ["interventions", "var_names"])
def test_numeric_metadata_is_rejected(tmp_path, field):
    path = tmp_path / f"numeric-{field}.npz"
    arrays = {"expression_matrix": np.ones((1, 2)), "interventions": np.asarray(["non-targeting"]), "var_names": np.asarray(["A", "B"])}
    arrays[field] = np.asarray([1, 2])
    np.savez(path, **arrays)
    with pytest.raises(TaskCDataError, match="Unicode|byte-string"):
        load_task_c_dataset(path, context_id="k562")


def test_metadata_whitespace_is_rejected(tmp_path):
    path = tmp_path / "whitespace.npz"
    np.savez(path, expression_matrix=np.ones((1, 2)), interventions=np.asarray([" non-targeting"]), var_names=np.asarray(["A", "B"]))
    with pytest.raises(TaskCDataError, match="whitespace"):
        load_task_c_dataset(path, context_id="k562")


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
    write_dataset(path, ["A", "B"], ["non-targeting"])
    dataset = load_task_c_dataset(path, context_id="k562")
    provenance = build_task_c_provenance(dataset)
    assert provenance["context"] == "k562"
    assert provenance["commit"] == CAUSALBENCH_COMMIT
    assert provenance["input_sha256"].startswith("sha256:")
    json.dumps(provenance, allow_nan=False)


def test_one_gene_dataset_is_rejected(tmp_path):
    path = tmp_path / "one-gene.npz"
    write_dataset(path, ["A"], ["non-targeting"])
    with pytest.raises(TaskCDataError, match="at least two gene columns"):
        load_task_c_dataset(path, context_id="k562")


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
    np.savez(path, expression_matrix=np.asarray([[np.nan, 0.0]]),
             interventions=np.asarray(["non-targeting"]), var_names=np.asarray(["A", "B"]))
    with pytest.raises(TaskCDataError, match="finite"):
        load_task_c_dataset(path, context_id="k562")


def test_empty_intervention_label_is_rejected(tmp_path):
    path = tmp_path / "empty-label.npz"
    write_dataset(path, ["A", "B"], ["non-targeting", "A"])
    np.savez(path, expression_matrix=np.ones((2, 2)),
             interventions=np.asarray(["non-targeting", ""]), var_names=np.asarray(["A", "B"]))
    with pytest.raises(TaskCDataError, match="intervention labels"):
        load_task_c_dataset(path, context_id="k562")


def test_invalid_utf8_intervention_bytes_are_rejected(tmp_path):
    path = tmp_path / "bad-intervention-encoding.npz"
    np.savez(path, expression_matrix=np.ones((1, 2)),
             interventions=np.asarray([b"non-targeting\xff"]),
             var_names=np.asarray(["A", "B"]))
    with pytest.raises(TaskCDataError, match="intervention labels"):
        load_task_c_dataset(path, context_id="k562")


def test_invalid_utf8_gene_bytes_are_rejected(tmp_path):
    path = tmp_path / "bad-gene-encoding.npz"
    np.savez(path, expression_matrix=np.ones((1, 2)),
             interventions=np.asarray(["non-targeting"]),
             var_names=np.asarray([b"A\xff", b"B"]))
    with pytest.raises(TaskCDataError, match="gene names"):
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
