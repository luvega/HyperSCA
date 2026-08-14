import errno
import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.evaluation.task_c_data import (
    CAUSALBENCH_COMMIT,
    SealedHoldoutSemanticContentHasher,
    TaskCDataError,
    TaskCDataset,
    TaskCSplit,
    build_shared_task_c_split,
    validate_task_c_split,
    build_task_c_provenance,
    build_task_c_reference_provenance,
    load_task_c_dataset,
    materialize_task_c_split,
    sha256_path,
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


def dataset_with_four_cell_f(context_id: str) -> TaskCDataset:
    genes = ("A", "B", "C", "D", "E", "F", "Z")
    labels = ["non-targeting"] * 10
    for source in ("A", "B", "C", "D", "E"):
        labels.extend([source] * 5)
    labels.extend(["F"] * 4)
    expression = np.arange(len(labels) * len(genes), dtype=np.float32).reshape(
        len(labels), len(genes)
    )
    return TaskCDataset(
        expression=expression,
        interventions=np.asarray(labels, dtype=str),
        gene_names=genes,
        context_id=context_id,
        source_path=Path(f"{context_id}.npz"),
        source_sha256=f"sha256:{context_id}-four-cell-f",
    )


@pytest.mark.parametrize("array_name", ["expression", "interventions"])
def test_task_c_dataset_arrays_are_read_only(array_name: str) -> None:
    dataset = dataset_for_split("k562")
    array = getattr(dataset, array_name)
    with pytest.raises(ValueError, match="read-only|writeable"):
        array.flat[0] = array.flat[0]


def test_materialization_rejects_in_memory_expression_tampering(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    k562.expression.setflags(write=True)
    k562.expression[0, 0] += 1

    with pytest.raises(TaskCDataError, match="content|changed|integrity"):
        materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")


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


def _walk_text_values(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_text_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_text_values(nested)
    elif isinstance(value, str):
        yield value


def test_materialized_training_files_exclude_holdout_sources(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")

    train = load_task_c_dataset(result["within"]["k562"]["train"], context_id="k562")
    assert not (set(train.interventions.tolist()) & set(split.holdout_sources))

    public = json.loads(Path(result["public_manifest"]).read_text(encoding="utf-8"))
    assert "holdout_sources" not in public
    assert "control_indices" not in public
    assert all("private" not in value for value in _walk_text_values(public))
    public_without_gene_provenance = dict(public)
    public_without_gene_provenance.pop("gene_projection")
    public_without_gene_provenance["materialization_identity"] = dict(
        public_without_gene_provenance["materialization_identity"]
    )
    public_without_gene_provenance["materialization_identity"].pop(
        "gene_projection"
    )
    assert not set(split.holdout_sources) & set(
        _walk_text_values(public_without_gene_provenance)
    )

    private = json.loads(Path(result["private_manifest"]).read_text(encoding="utf-8"))
    assert private["holdout_sources"] == list(split.holdout_sources)
    assert private["control_indices"]["k562"]["holdout"] == list(
        split.control_indices["k562"]["holdout"]
    )


def test_cross_context_adaptation_contains_only_target_controls(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")

    adapt = load_task_c_dataset(
        result["cross"]["k562_to_rpe1"]["target_adapt_refit"],
        context_id="rpe1",
    )
    assert set(adapt.interventions.tolist()) == {"non-targeting"}
    source_train = load_task_c_dataset(
        result["cross"]["k562_to_rpe1"]["source_train"],
        context_id="k562",
    )
    source_tune = load_task_c_dataset(
        result["cross"]["k562_to_rpe1"]["source_tune"],
        context_id="k562",
    )
    assert not (set(source_train.interventions.tolist()) & set(split.tune_sources))
    assert not (set(source_tune.interventions.tolist()) & set(split.train_sources))


def test_materialized_manifests_use_relative_paths_and_verified_hashes(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=23)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)

    public = json.loads(Path(result["public_manifest"]).read_text(encoding="utf-8"))
    private = json.loads(Path(result["private_manifest"]).read_text(encoding="utf-8"))
    assert public["min_cells_per_intervention"] == split.min_cells_per_intervention
    assert private["min_cells_per_intervention"] == split.min_cells_per_intervention
    assert public["content_sha256"] == {
        "k562": k562.content_sha256,
        "rpe1": rpe1.content_sha256,
    }
    assert private["content_sha256"] == public["content_sha256"]
    commitment = public["sealed_holdout_semantic_content_sha256"]
    assert commitment.startswith("sha256:")
    assert len(commitment) == 71
    assert private["sealed_holdout_semantic_content_sha256"] == commitment
    assert public["materialization_identity"][
        "sealed_holdout_semantic_content_sha256"
    ] == commitment
    assert private["materialization_identity"] == public["materialization_identity"]
    assert public["materialization_identity"]["content_sha256"] == public[
        "content_sha256"
    ]
    assert public["files"]
    assert private["files"]
    assert set(public["files"]).isdisjoint(private["files"])
    for manifest in (public, private):
        for relative_path, digest in manifest["files"].items():
            assert not Path(relative_path).is_absolute()
            assert sha256_path(root / relative_path) == digest
    expected_public = {
        Path(path).resolve().relative_to(root.resolve()).as_posix()
        for partitions in result["within"].values()
        for name, path in partitions.items()
        if name != "holdout"
    }
    expected_public.update(
        Path(path).resolve().relative_to(root.resolve()).as_posix()
        for partitions in result["cross"].values()
        for name, path in partitions.items()
        if name != "target_holdout"
    )
    expected_private = {
        Path(partitions["holdout"]).resolve().relative_to(root.resolve()).as_posix()
        for partitions in result["within"].values()
    }
    expected_private.update(
        Path(partitions["target_holdout"])
        .resolve()
        .relative_to(root.resolve())
        .as_posix()
        for partitions in result["cross"].values()
    )
    assert set(public["files"]) == expected_public
    assert set(private["files"]) == expected_private


def test_materialization_reuse_rejects_resigned_private_expression_change(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    relative = "private/within/k562/holdout.npz"
    holdout_path = root / relative
    with np.load(holdout_path, allow_pickle=False) as archive:
        expression = np.asarray(archive["expression_matrix"]).copy()
        interventions = np.asarray(archive["interventions"]).copy()
        var_names = np.asarray(archive["var_names"]).copy()
    expression[0, 0] += np.asarray(0.5, dtype=expression.dtype)
    np.savez_compressed(
        holdout_path,
        expression_matrix=expression,
        interventions=interventions,
        var_names=var_names,
    )
    manifest_path = Path(result["private_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = sha256_path(holdout_path)
    write_json(manifest_path, manifest)

    with pytest.raises(TaskCDataError, match="identity|commitment|content|rematerialize"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_sealed_holdout_commitment_covers_array_metadata_bytes_and_gene_order() -> None:
    paths = (
        "private/cross/k562_to_rpe1/target_holdout.npz",
        "private/cross/rpe1_to_k562/target_holdout.npz",
        "private/within/k562/holdout.npz",
        "private/within/rpe1/holdout.npz",
    )

    def commitment(
        expression: np.ndarray,
        interventions: np.ndarray,
        genes: tuple[str, ...],
    ) -> str:
        hasher = SealedHoldoutSemanticContentHasher()
        for relative in paths:
            hasher.add_arrays(relative, expression, interventions, genes)
        return hasher.sha256()

    expression = np.arange(6, dtype=np.float32).reshape(3, 2)
    interventions = np.asarray(["non-targeting", "A", "A"], dtype="U13")
    genes = ("A", "B")
    observed = {
        commitment(expression, interventions, genes),
        commitment(expression.astype(np.float64), interventions, genes),
        commitment(expression.reshape(2, 3), interventions, genes),
        commitment(expression + np.float32(1.0), interventions, genes),
        commitment(expression, interventions.astype("U14"), genes),
        commitment(expression, interventions.reshape(1, 3), genes),
        commitment(expression, np.asarray(["non-targeting", "B", "A"]), genes),
        commitment(expression, interventions, tuple(reversed(genes))),
    }
    assert len(observed) == 8


def test_materialization_reuse_requires_rematerialization_for_old_identity(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    for name in ("public_manifest", "private_manifest"):
        path = Path(result[name])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("sealed_holdout_semantic_content_sha256")
        manifest["materialization_identity"].pop(
            "sealed_holdout_semantic_content_sha256"
        )
        write_json(path, manifest)

    with pytest.raises(TaskCDataError, match="obsolete.*rematerialize"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_projects_both_contexts_to_sorted_common_genes(
    tmp_path: Path,
) -> None:
    k562_base = dataset_for_split("k562")
    rpe1_base = dataset_for_split("rpe1")
    k562_genes = ("K_ONLY", "B", "A", "C", "D", "E", "F", "Z")
    rpe1_genes = ("Z", "R_ONLY", "F", "E", "D", "C", "B", "A")
    k562_expression = np.column_stack(
        (
            np.full(len(k562_base.interventions), -1.0),
            k562_base.expression[:, 1],
            k562_base.expression[:, 0],
            k562_base.expression[:, 2],
            k562_base.expression[:, 3],
            k562_base.expression[:, 4],
            k562_base.expression[:, 5],
            k562_base.expression[:, 6],
        )
    )
    rpe1_expression = np.column_stack(
        (
            rpe1_base.expression[:, 6],
            np.full(len(rpe1_base.interventions), -2.0),
            rpe1_base.expression[:, 5],
            rpe1_base.expression[:, 4],
            rpe1_base.expression[:, 3],
            rpe1_base.expression[:, 2],
            rpe1_base.expression[:, 1],
            rpe1_base.expression[:, 0],
        )
    )
    k562 = replace(k562_base, expression=k562_expression, gene_names=k562_genes)
    rpe1 = replace(rpe1_base, expression=rpe1_expression, gene_names=rpe1_genes)
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")

    common = ("A", "B", "C", "D", "E", "F", "Z")
    k562_refit = load_task_c_dataset(
        result["within"]["k562"]["refit"], context_id="k562"
    )
    rpe1_refit = load_task_c_dataset(
        result["within"]["rpe1"]["refit"], context_id="rpe1"
    )
    assert k562_refit.gene_names == common
    assert rpe1_refit.gene_names == common
    assert "K_ONLY" not in k562_refit.gene_names
    assert "R_ONLY" not in rpe1_refit.gene_names
    refit_sources = split.train_sources + split.tune_sources
    k562_rows = np.sort(
        np.concatenate(
            (
                np.flatnonzero(np.isin(k562.interventions, refit_sources)),
                np.asarray(
                    split.control_indices["k562"]["train"]
                    + split.control_indices["k562"]["tune"]
                ),
            )
        )
    )
    rpe1_rows = np.sort(
        np.concatenate(
            (
                np.flatnonzero(np.isin(rpe1.interventions, refit_sources)),
                np.asarray(
                    split.control_indices["rpe1"]["train"]
                    + split.control_indices["rpe1"]["tune"]
                ),
            )
        )
    )
    np.testing.assert_array_equal(
        k562_refit.expression, k562_expression[np.ix_(k562_rows, [2, 1, 3, 4, 5, 6, 7])]
    )
    np.testing.assert_array_equal(
        rpe1_refit.expression, rpe1_expression[np.ix_(rpe1_rows, [7, 6, 5, 4, 3, 2, 0])]
    )
    public = json.loads(Path(result["public_manifest"]).read_text(encoding="utf-8"))
    projection = public["gene_projection"]
    assert projection["projection_rule"] == "sorted_common_gene_intersection_v1"
    assert projection["common"]["ordered_genes"] == list(common)
    assert projection["contexts"]["k562"]["selected_original_indices"] == [
        2,
        1,
        3,
        4,
        5,
        6,
        7,
    ]
    assert projection["contexts"]["rpe1"]["selected_original_indices"] == [
        7,
        6,
        5,
        4,
        3,
        2,
        0,
    ]
    assert k562.gene_names == k562_genes
    assert rpe1.gene_names == rpe1_genes
    np.testing.assert_array_equal(k562.expression, k562_expression)
    np.testing.assert_array_equal(rpe1.expression, rpe1_expression)


def test_materialization_reuse_rejects_tampered_gene_projection(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    manifest_path = Path(result["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gene_projection"]["contexts"]["k562"][
        "selected_original_indices"
    ] = list(reversed(manifest["gene_projection"]["contexts"]["k562"][
        "selected_original_indices"
    ]))
    manifest["materialization_identity"]["gene_projection"] = manifest[
        "gene_projection"
    ]
    write_json(manifest_path, manifest)

    with pytest.raises(TaskCDataError, match="identity|semantic|projection"):
        materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")


def test_materialization_reuses_matching_bundle_and_rejects_changed_identity(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=47)
    root = tmp_path / "bundle"
    first = materialize_task_c_split(k562, rpe1, split, root)
    original_manifest = Path(first["public_manifest"]).read_bytes()

    second = materialize_task_c_split(k562, rpe1, split, root)
    assert Path(second["public_manifest"]).read_bytes() == original_manifest

    changed = replace(k562, source_sha256="sha256:changed")
    with pytest.raises(TaskCDataError, match="existing|identity|different"):
        materialize_task_c_split(changed, rpe1, split, root)


def test_materialization_identity_rejects_changed_minimum_cell_threshold(
    tmp_path: Path,
) -> None:
    k562 = dataset_with_four_cell_f("k562")
    rpe1 = dataset_with_four_cell_f("rpe1")
    root = tmp_path / "bundle"
    five_cell_split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=5)
    four_cell_split = build_shared_task_c_split(k562, rpe1, seed=11, min_cells=4)
    five_cell_sources = (
        five_cell_split.train_sources
        + five_cell_split.tune_sources
        + five_cell_split.holdout_sources
    )
    four_cell_sources = (
        four_cell_split.train_sources
        + four_cell_split.tune_sources
        + four_cell_split.holdout_sources
    )
    assert "F" not in five_cell_sources
    assert "F" in four_cell_sources
    materialize_task_c_split(k562, rpe1, five_cell_split, root)

    with pytest.raises(TaskCDataError, match="identity|different|existing"):
        materialize_task_c_split(k562, rpe1, four_cell_split, root)


def test_materialization_identity_rejects_changed_content_with_same_source_hash(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    root = tmp_path / "bundle"
    materialize_task_c_split(k562, rpe1, split, root)
    changed_expression = k562.expression.copy()
    changed_expression[0, 0] += 1
    changed = replace(k562, expression=changed_expression)

    with pytest.raises(TaskCDataError, match="identity|content|different"):
        materialize_task_c_split(changed, rpe1, split, root)


def test_materialization_rejects_tampered_public_semantic_fields(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=23)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    manifest_path = Path(result["public_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_sources"] = list(reversed(manifest["train_sources"]))
    write_json(manifest_path, manifest)

    with pytest.raises(TaskCDataError, match="manifest|semantic|record"):
        materialize_task_c_split(k562, rpe1, split, root)


@pytest.mark.parametrize("field", ["holdout_sources", "control_indices"])
def test_materialization_rejects_tampered_private_semantic_fields(
    tmp_path: Path,
    field: str,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=47)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    manifest_path = Path(result["private_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "holdout_sources":
        manifest[field] = []
    else:
        manifest[field]["k562"]["holdout"] = []
    write_json(manifest_path, manifest)

    with pytest.raises(TaskCDataError, match="manifest|semantic|record"):
        materialize_task_c_split(k562, rpe1, split, root)


@pytest.mark.parametrize(
    "manifest_name,extra_key,extra_value",
    [
        ("public_manifest", "holdout_sources", ["secret"]),
        ("public_manifest", "private_path", "private/holdout.npz"),
        ("public_manifest", "unexpected", True),
        ("private_manifest", "unexpected", True),
    ],
)
def test_materialization_rejects_unknown_manifest_fields(
    tmp_path: Path,
    manifest_name: str,
    extra_key: str,
    extra_value: object,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=71)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    manifest_path = Path(result[manifest_name])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[extra_key] = extra_value
    write_json(manifest_path, manifest)

    with pytest.raises(TaskCDataError, match="manifest|schema|field"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_rejects_tampered_existing_artifact(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=71)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    Path(result["within"]["k562"]["train"]).write_bytes(b"changed")

    with pytest.raises(TaskCDataError, match="hash|changed|existing"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_rejects_public_symlink_to_private_artifact(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    public_artifact = Path(result["within"]["k562"]["train"])
    private_artifact = Path(result["within"]["k562"]["holdout"])
    public_artifact.unlink()
    public_artifact.symlink_to(private_artifact)
    public_manifest_path = Path(result["public_manifest"])
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    relative = public_artifact.relative_to(root).as_posix()
    public_manifest["files"][relative] = sha256_path(private_artifact)
    write_json(public_manifest_path, public_manifest)

    with pytest.raises(TaskCDataError, match="symbolic|symlink"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_rejects_public_hardlink_to_private_artifact(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=23)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    public_artifact = Path(result["within"]["k562"]["train"])
    private_artifact = Path(result["within"]["k562"]["holdout"])
    public_artifact.unlink()
    os.link(private_artifact, public_artifact)
    public_manifest_path = Path(result["public_manifest"])
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    relative = public_artifact.relative_to(root).as_posix()
    public_manifest["files"][relative] = sha256_path(private_artifact)
    write_json(public_manifest_path, public_manifest)

    with pytest.raises(TaskCDataError, match="hard link|inode|private"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_rejects_symlinked_manifest(tmp_path: Path) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=47)
    root = tmp_path / "bundle"
    result = materialize_task_c_split(k562, rpe1, split, root)
    manifest_path = Path(result["public_manifest"])
    outside = tmp_path / "outside-public-manifest.json"
    outside.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(outside)

    with pytest.raises(TaskCDataError, match="symbolic|symlink"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_rejects_symlinked_bundle_parent_before_writing(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=71)
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "within").symlink_to(outside, target_is_directory=True)

    with pytest.raises(TaskCDataError, match="symbolic|symlink"):
        materialize_task_c_split(k562, rpe1, split, root)


def test_materialization_preflight_does_not_create_missing_output(tmp_path: Path) -> None:
    from src.evaluation.task_c_data import check_task_c_materialization

    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=71)
    output = tmp_path / "missing"

    assert check_task_c_materialization(k562, rpe1, split, output) == "missing"
    assert not output.exists()


def test_cross_source_files_reuse_public_within_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=97)
    original_savez = np.savez_compressed
    calls = []

    def count_savez(*args, **kwargs):
        calls.append(args[0])
        return original_savez(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", count_savez)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")

    assert len(calls) == 16
    for context, direction in (
        ("k562", "k562_to_rpe1"),
        ("rpe1", "rpe1_to_k562"),
    ):
        for within_name, cross_name in (
            ("train", "source_train"),
            ("tune", "source_tune"),
            ("refit", "source_refit"),
        ):
            within_stat = Path(result["within"][context][within_name]).stat()
            cross_stat = Path(result["cross"][direction][cross_name]).stat()
            assert (within_stat.st_dev, within_stat.st_ino) == (
                cross_stat.st_dev,
                cross_stat.st_ino,
            )

    public_manifest = json.loads(
        Path(result["public_manifest"]).read_text(encoding="utf-8")
    )
    private_manifest = json.loads(
        Path(result["private_manifest"]).read_text(encoding="utf-8")
    )

    def inode(relative: str) -> tuple[int, int]:
        stat = (tmp_path / "bundle" / relative).stat()
        return stat.st_dev, stat.st_ino

    public_inodes = {
        inode(path) for path in public_manifest["files"]
    }
    private_inodes = {
        inode(path) for path in private_manifest["files"]
    }
    assert public_inodes.isdisjoint(private_inodes)


def test_cross_source_copy_fallback_preserves_bytes_and_cleans_temporary_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)

    def cross_device_link(source, destination):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", cross_device_link)
    result = materialize_task_c_split(k562, rpe1, split, tmp_path / "bundle")
    within = Path(result["within"]["k562"]["train"])
    cross = Path(result["cross"]["k562_to_rpe1"]["source_train"])
    assert cross.read_bytes() == within.read_bytes()
    assert sha256_path(cross) == sha256_path(within)
    assert (cross.stat().st_dev, cross.stat().st_ino) != (
        within.stat().st_dev,
        within.stat().st_ino,
    )
    assert not [
        path
        for path in (tmp_path / "bundle").rglob("*")
        if path.name.startswith(".source_")
    ]


def test_output_directory_symlink_is_resolved_once_but_inner_symlinks_are_not(
    tmp_path: Path,
) -> None:
    k562 = dataset_for_split("k562")
    rpe1 = dataset_for_split("rpe1")
    split = build_shared_task_c_split(k562, rpe1, seed=11)
    real_root = tmp_path / "real-bundle"
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(real_root, target_is_directory=True)

    result = materialize_task_c_split(k562, rpe1, split, alias)
    assert Path(result["public_manifest"]).parent == real_root.resolve()


def test_failed_npz_write_leaves_no_final_or_temporary_file(tmp_path, monkeypatch):
    import src.evaluation.task_c_data as task_c_data

    dataset = dataset_for_split("k562")
    destination = tmp_path / "part.npz"

    def fail_after_partial_write(path, **arrays):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(TaskCDataError, match="write"):
        task_c_data._write_dataset_subset(dataset, np.asarray([0, 1]), destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_runtime_error_during_npz_write_still_removes_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.evaluation.task_c_data as task_c_data

    dataset = dataset_for_split("k562")
    destination = tmp_path / "part.npz"

    def fail_after_partial_write(path, **arrays):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated library failure")

    monkeypatch.setattr(np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(TaskCDataError, match="write"):
        task_c_data._write_dataset_subset(dataset, np.asarray([0, 1]), destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_keyboard_interrupt_during_npz_write_propagates_after_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.evaluation.task_c_data as task_c_data

    dataset = dataset_for_split("k562")
    destination = tmp_path / "part.npz"

    def interrupt_after_partial_write(path, **arrays):
        Path(path).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez_compressed", interrupt_after_partial_write)
    with pytest.raises(KeyboardInterrupt):
        task_c_data._write_dataset_subset(dataset, np.asarray([0, 1]), destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


_MATERIALIZED_CASES = [
    *(
        ("within", context, partition)
        for context in ("k562", "rpe1")
        for partition in ("train", "tune", "refit", "holdout")
    ),
    *(
        ("cross", direction, partition)
        for direction in ("k562_to_rpe1", "rpe1_to_k562")
        for partition in (
            "source_train",
            "source_tune",
            "source_refit",
            "target_adapt_train",
            "target_adapt_tune",
            "target_adapt_refit",
            "target_holdout",
        )
    ),
]


@pytest.mark.parametrize("family,scope,partition", _MATERIALIZED_CASES)
def test_every_materialized_file_contains_exact_original_rows(
    tmp_path: Path,
    family: str,
    scope: str,
    partition: str,
) -> None:
    datasets = {
        "k562": dataset_for_split("k562"),
        "rpe1": dataset_for_split("rpe1"),
    }
    split = build_shared_task_c_split(datasets["k562"], datasets["rpe1"], seed=97)
    result = materialize_task_c_split(
        datasets["k562"], datasets["rpe1"], split, tmp_path / "bundle"
    )

    if family == "within":
        dataset = datasets[scope]
        source_names = {
            "train": split.train_sources,
            "tune": split.tune_sources,
            "refit": split.train_sources + split.tune_sources,
            "holdout": split.holdout_sources,
        }[partition]
        control_parts = {
            "train": ("train",),
            "tune": ("tune",),
            "refit": ("train", "tune"),
            "holdout": ("holdout",),
        }[partition]
        artifact_path = result[family][scope][partition]
        control_context = scope
    else:
        source_context, target_context = scope.split("_to_")
        if partition.startswith("source_"):
            dataset = datasets[source_context]
            source_names = {
                "source_train": split.train_sources,
                "source_tune": split.tune_sources,
                "source_refit": split.train_sources + split.tune_sources,
            }[partition]
            control_parts = {
                "source_train": ("train",),
                "source_tune": ("tune",),
                "source_refit": ("train", "tune"),
            }[partition]
            control_context = source_context
        else:
            dataset = datasets[target_context]
            source_names = (
                split.train_sources + split.tune_sources + split.holdout_sources
                if partition == "target_holdout"
                else ()
            )
            control_parts = (
                ("holdout",)
                if partition == "target_holdout"
                else {
                    "target_adapt_train": ("train",),
                    "target_adapt_tune": ("tune",),
                    "target_adapt_refit": ("train", "tune"),
                }[partition]
            )
            control_context = target_context
        artifact_path = result[family][scope][partition]

    controls = {
        index
        for control_part in control_parts
        for index in split.control_indices[control_context][control_part]
    }
    sources = set(source_names)
    expected_indices = np.asarray(
        [
            index
            for index, label in enumerate(dataset.interventions.tolist())
            if label in sources or index in controls
        ],
        dtype=int,
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["expression_matrix"], dataset.expression[expected_indices]
        )
        np.testing.assert_array_equal(
            archive["interventions"], dataset.interventions[expected_indices]
        )
        np.testing.assert_array_equal(
            archive["var_names"], np.asarray(dataset.gene_names)
        )
