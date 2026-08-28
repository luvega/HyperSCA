from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from functools import lru_cache
import hashlib
import inspect
import json
import random
from types import SimpleNamespace
from typing import cast

import pytest

from src.evaluation import spatial_perturbation_split as split_module
from src.evaluation.spatial_perturbation_split import (
    MAX_ABSTENTION,
    MIN_BAND_NEIGHBOURS,
    MIN_CELL_TYPE_NEIGHBOURS,
    MIN_COVERAGE,
    MIN_SAFE_SOURCE_CELLS,
    MIN_SOURCE_CELLS,
    MIN_SPATIAL_BLOCKS,
    BridgeBlockAdjacency,
    BridgeEligibilityEvidence,
    BridgeNeighbourRelation,
    BridgeParentEvidence,
    BridgePrimaryUnitEvidence,
    BridgeSplitManifest,
    BridgeSplitMetadata,
    BridgeSplitRow,
    SpatialPerturbationSplitError,
    build_bridge_eligibility_evidence,
    build_pilot_fold,
    eligibility_evidence_to_mapping,
    evaluate_bridge_eligibility,
    split_manifest_to_mapping,
)
from src.evaluation.spatial_perturbation_registry import (
    BridgeCandidate,
    MetadataSummary,
    audit_bridge_capability,
)


PERTURBATIONS = tuple(f"guide_{index}" for index in range(5))
TARGETS = tuple(f"Gene{index}" for index in range(5))
NEIGHBOUR_TYPES = ("astrocyte", "microglia")
BANDS = ("proximal", "local")


class _HostileSequence:
    def __iter__(self) -> object:
        raise AssertionError("hostile sequence was iterated before validation")


def _canonical_sha(mapping: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(mapping, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def synthetic_metadata(
    *,
    animal_count: int = 3,
    adjacency: bool = False,
    block_count: int = 3,
    neighbour_types: tuple[str, ...] = NEIGHBOUR_TYPES,
    neighbour_count: int | None = None,
    confirmatory: bool = False,
) -> BridgeSplitMetadata:
    rows: list[BridgeSplitRow] = []
    relations: list[BridgeNeighbourRelation] = []
    row_id = 0
    source_cells: dict[tuple[str, str, str], list[str]] = {}
    for animal_index in range(animal_count):
        animal = f"mouse_{animal_index + 1}"
        section = f"{animal}_section"
        for source_perturbation, role in (
            *((perturbation, "perturbation_source") for perturbation in PERTURBATIONS),
            ("mSafe", "safe_source"),
        ):
            for index in range(max(30, block_count * 5)):
                block = f"block_{index % block_count}"
                cell_id = f"cell_{row_id:06d}"
                rows.append(
                    BridgeSplitRow(
                        row_id, cell_id, animal, section, block,
                        source_perturbation, source_perturbation,
                        "source_type", "source_type", role, "own",
                    )
                )
                source_cells.setdefault((animal, source_perturbation, block), []).append(cell_id)
                row_id += 1
        frozen_neighbour_count = (
            neighbour_count
            if neighbour_count is not None
            else (
                max(50, block_count * 13)
                if len(neighbour_types) == 1
                else max(30, block_count * 10)
            )
        )
        safe_neighbor_ids: dict[tuple[str, str], list[str]] = {}
        for neighbour_type in neighbour_types:
            for band in BANDS:
                safe_ids: list[str] = []
                for index in range(frozen_neighbour_count):
                    block = f"block_{index % block_count}"
                    cell_id = f"cell_{row_id:06d}"
                    rows.append(BridgeSplitRow(
                        row_id, cell_id, animal, section, block, "unassigned",
                        "unperturbed", neighbour_type, neighbour_type, "neighbour", "none",
                    ))
                    safe_ids.append(cell_id)
                    row_id += 1
                safe_neighbor_ids[(neighbour_type, band)] = safe_ids
                for index, neighbor_id in enumerate(safe_ids):
                    block = f"block_{index % block_count}"
                    sources = source_cells[(animal, "mSafe", block)]
                    source_id = sources[(index // block_count) % len(sources)]
                    relations.append(split_module.freeze_bridge_neighbour_relation(
                        animal, section, block, source_id, neighbor_id, "mSafe",
                        "source_type", neighbour_type,
                        ((index % 5) + 1 if band == "proximal" else (index % 10) + 6),
                        band, False, True,
                    ))
        for perturbation in PERTURBATIONS:
            for neighbour_type in neighbour_types:
                for band in BANDS:
                    treatment_ids: list[str] = []
                    for index in range(frozen_neighbour_count):
                        block = f"block_{index % block_count}"
                        cell_id = f"cell_{row_id:06d}"
                        rows.append(BridgeSplitRow(
                            row_id, cell_id, animal, section, block, "unassigned",
                            "unperturbed", neighbour_type, neighbour_type,
                            "neighbour", "none",
                        ))
                        treatment_ids.append(cell_id)
                        row_id += 1
                    for index, neighbor_id in enumerate(treatment_ids):
                        block = f"block_{index % block_count}"
                        sources = source_cells[(animal, perturbation, block)]
                        source_id = sources[(index // block_count) % len(sources)]
                        relations.append(split_module.freeze_bridge_neighbour_relation(
                            animal, section, block, source_id, neighbor_id,
                            perturbation, "source_type", neighbour_type,
                            ((index % 5) + 1 if band == "proximal" else (index % 10) + 6),
                            band, False, False,
                        ))
    adjacent = tuple(
        BridgeBlockAdjacency(
            f"mouse_{animal_index + 1}", f"mouse_{animal_index + 1}_section",
            f"block_{first}", f"block_{second}", adjacency and first == 0 and second == 1,
        )
        for animal_index in range(animal_count)
        for first in range(block_count)
        for second in range(first + 1, block_count)
    )
    animals = tuple(f"mouse_{index + 1}" for index in range(animal_count))
    sections = tuple((animal, (f"{animal}_section",)) for animal in animals)
    candidate = BridgeCandidate(
        "generic_task5_bridge", "SYNTHETIC", "spatial_perturbation",
        animals, sections, "mSafe", PERTURBATIONS,
        "https://example.test/SYNTHETIC", "a" * 64,
    )
    total_rows = len(rows)
    per_animal_rows = tuple((animal, sum(row.animal_id == animal for row in rows)) for animal in animals)
    per_animal_sources = tuple(
        (animal, sum(row.animal_id == animal and row.cell_role == "perturbation_source" for row in rows))
        for animal in animals
    )
    per_animal_safe = tuple(
        (animal, sum(row.animal_id == animal and row.cell_role == "safe_source" for row in rows))
        for animal in animals
    )
    cohort_ids = ("development", "external") if confirmatory else ("pilot",)
    assignments = tuple(
        (animal, "external" if confirmatory and index >= 3 else (
            "development" if confirmatory else "pilot"
        ))
        for index, animal in enumerate(animals)
    )
    summary = MetadataSummary(
        "generic_task5_bridge", "SYNTHETIC", cohort_ids, animals, sections,
        tuple(f"block_{index}" for index in range(block_count)), True, True,
        total_rows, TARGETS, len(TARGETS), PERTURBATIONS,
        tuple((perturbation, sum(row.observed_label == perturbation for row in rows)) for perturbation in PERTURBATIONS),
        (("mSafe", sum(count for _, count in per_animal_safe)),),
        (("valid", total_rows),), (("valid", total_rows),),
        assignments, (("external",) if confirmatory else ()), per_animal_rows,
        per_animal_sources, per_animal_safe, per_animal_rows, per_animal_rows,
        "CC-BY-4.0", "a" * 64, True,
    )
    capability = audit_bridge_capability(candidate, summary)
    relation_table = split_module.freeze_bridge_neighbour_table(tuple(relations))
    frozen_relations = relation_table.relations
    return BridgeSplitMetadata(
        tuple(rows), TARGETS, PERTURBATIONS, neighbour_types,
        tuple(zip(PERTURBATIONS, TARGETS)), adjacent, "mSafe", frozen_relations,
        relation_table.identity_sha256,
        candidate, summary, capability,
    )


def test_review_primary_excludes_holdout_only_perturbations() -> None:
    metadata = synthetic_metadata()
    rows = tuple(
        row for row in metadata.rows
        if not (
            row.animal_id in {"mouse_2", "mouse_3"}
            and row.context_perturbation_id == "guide_0"
        )
    )
    pruned = BridgeSplitMetadata(
        rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label,
        tuple(
            item for item in metadata.neighbour_relations
            if not (
                item.animal_id in {"mouse_2", "mouse_3"}
                and item.source_perturbation_id == "guide_0"
            )
        ),
        split_module.freeze_bridge_neighbour_table(tuple(
            item for item in metadata.neighbour_relations
            if not (
                item.animal_id in {"mouse_2", "mouse_3"}
                and item.source_perturbation_id == "guide_0"
            )
        )).identity_sha256,
        metadata.candidate, metadata.registry_summary, metadata.capability_result,
    )
    manifest = build_pilot_fold(pruned, "mouse_1")

    assert manifest.perturbations == PERTURBATIONS[1:]
    assert manifest.secondary_perturbations == ("guide_0",)
    assert evaluate_bridge_eligibility(manifest, complete_evidence(manifest)).eligible


def test_review_metadata_contract_binds_task4_registry_objects() -> None:
    names = {field.name for field in fields(BridgeSplitMetadata)}
    assert {"candidate", "registry_summary", "capability_result"} <= names

    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    object.__setattr__(manifest.candidate, "perturbation_labels", PERTURBATIONS[1:])
    with pytest.raises(SpatialPerturbationSplitError):
        split_manifest_to_mapping(manifest)


def test_review_neighbour_provenance_is_a_separate_reusable_relation_table() -> None:
    relation_type = getattr(split_module, "BridgeNeighbourRelation", None)
    assert relation_type is not None
    relation_fields = {field.name for field in fields(relation_type)}
    assert {
        "relation_id", "source_cell_id", "neighbor_cell_id", "rank", "band",
        "contamination", "source_perturbation_id", "source_cell_type",
        "neighbor_cell_type", "is_safe_control",
    } <= relation_fields
    assert "perturbation_id" not in relation_fields
    metadata_names = {field.name for field in fields(BridgeSplitMetadata)}
    assert {"neighbour_relations", "neighbour_table_identity_sha256"} <= metadata_names

    relation = synthetic_metadata().neighbour_relations[0]
    with pytest.raises(SpatialPerturbationSplitError, match="contamination"):
        replace(relation, contamination=0.0)
    with pytest.raises(SpatialPerturbationSplitError, match="rank.*band"):
        replace(relation, rank=15 if relation.band == "proximal" else 1)
    contaminated = split_module.freeze_bridge_neighbour_relation(
        relation.animal_id, relation.section_id, relation.spatial_block,
        relation.source_cell_id, relation.neighbor_cell_id,
        relation.source_perturbation_id, relation.source_cell_type,
        relation.neighbor_cell_type, relation.rank, relation.band, True,
        relation.is_safe_control,
    )
    with pytest.raises(SpatialPerturbationSplitError, match="contaminated"):
        split_module._require_relations(
            (contaminated,), animal=relation.animal_id,
            perturbation=(
                "guide_0" if relation.is_safe_control
                else relation.source_perturbation_id
            ),
            cell_type=relation.neighbor_cell_type, band=relation.band,
            safe=relation.is_safe_control,
        )


def test_review_relations_may_cross_blocks_but_not_sections() -> None:
    metadata = synthetic_metadata()
    relation = metadata.neighbour_relations[0]
    source = next(
        row for row in metadata.rows
        if row.animal_id == relation.animal_id
        and row.section_id == relation.section_id
        and row.context_perturbation_id == relation.source_perturbation_id
        and row.cell_role == ("safe_source" if relation.is_safe_control else "perturbation_source")
        and row.spatial_block != relation.spatial_block
    )
    cross_block = split_module.freeze_bridge_neighbour_relation(
        relation.animal_id, relation.section_id, relation.spatial_block,
        source.cell_id, relation.neighbor_cell_id,
        relation.source_perturbation_id, relation.source_cell_type,
        relation.neighbor_cell_type, relation.rank,
        relation.band, relation.contamination, relation.is_safe_control,
    )
    cross_block_relations = tuple(
        cross_block if item.relation_id == relation.relation_id else item
        for item in metadata.neighbour_relations
    )
    accepted = BridgeSplitMetadata(
        metadata.rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label,
        cross_block_relations,
        split_module.freeze_bridge_neighbour_table(
            cross_block_relations
        ).identity_sha256,
        metadata.candidate, metadata.registry_summary, metadata.capability_result,
    )
    assert cross_block in accepted.neighbour_relations

    cross_section = split_module.freeze_bridge_neighbour_relation(
        relation.animal_id, "alien_section", relation.spatial_block,
        relation.source_cell_id, relation.neighbor_cell_id,
        relation.source_perturbation_id, relation.source_cell_type,
        relation.neighbor_cell_type, relation.rank, relation.band,
        relation.contamination, relation.is_safe_control,
    )
    cross_section_relations = tuple(
        cross_section if item.relation_id == relation.relation_id else item
        for item in metadata.neighbour_relations
    )
    with pytest.raises(SpatialPerturbationSplitError, match="atomic provenance"):
        BridgeSplitMetadata(
            metadata.rows, metadata.gene_names, metadata.perturbations,
            metadata.neighbour_cell_types, metadata.perturbation_targets,
            metadata.block_adjacency, metadata.safe_control_label,
            cross_section_relations,
            split_module.freeze_bridge_neighbour_table(
                cross_section_relations
            ).identity_sha256,
            metadata.candidate, metadata.registry_summary,
            metadata.capability_result,
        )


def test_review_block_graph_records_every_pair_state() -> None:
    adjacency_fields = {field.name for field in fields(BridgeBlockAdjacency)}
    assert "adjacent" in adjacency_fields
    metadata = synthetic_metadata()
    # One section and three blocks per animal: all 3 choose 2 states are frozen.
    assert len(metadata.block_adjacency) == 9
    assert all(type(item.adjacent) is bool for item in metadata.block_adjacency)


def test_review_generic_manifest_does_not_encode_pilot_cardinality() -> None:
    assert "split_role" in {field.name for field in fields(BridgeSplitManifest)}
    implementation = inspect.getsource(BridgeSplitManifest.__post_init__)
    assert "pilot requires exactly three animals" not in implementation
    assert "one exact evaluation animal is required" not in implementation
    pilot = build_pilot_fold(synthetic_metadata(), "mouse_1")
    with pytest.raises(SpatialPerturbationSplitError, match="five animals"):
        replace(
            pilot, split_role="confirmatory",
            split_id="confirmatory_partition:too_small",
        )


def test_review_development_only_perturbations_are_not_secondary() -> None:
    metadata = synthetic_metadata()
    rows = tuple(
        row for row in metadata.rows
        if not (
            row.animal_id == "mouse_1"
            and row.context_perturbation_id == "guide_0"
        )
    )
    relations = tuple(
        item for item in metadata.neighbour_relations
        if not (
            item.animal_id == "mouse_1"
            and item.source_perturbation_id == "guide_0"
        )
    )
    pruned = BridgeSplitMetadata(
        rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label, relations,
        split_module.freeze_bridge_neighbour_table(relations).identity_sha256,
        metadata.candidate,
        metadata.registry_summary, metadata.capability_result,
    )
    manifest = build_pilot_fold(pruned, "mouse_1")
    assert "guide_0" not in manifest.secondary_perturbations
    assert manifest.development_only_perturbations == ("guide_0",)


def test_review_registered_perturbation_cannot_have_all_sources_deleted() -> None:
    metadata = synthetic_metadata()
    rows = tuple(
        row for row in metadata.rows
        if row.context_perturbation_id != "guide_0"
    )
    relations = tuple(
        item for item in metadata.neighbour_relations
        if item.source_perturbation_id != "guide_0"
    )
    with pytest.raises(SpatialPerturbationSplitError, match="registered.*source"):
        BridgeSplitMetadata(
            rows, metadata.gene_names, metadata.perturbations,
            metadata.neighbour_cell_types, metadata.perturbation_targets,
            metadata.block_adjacency, metadata.safe_control_label, relations,
            split_module.freeze_bridge_neighbour_table(relations).identity_sha256,
            metadata.candidate, metadata.registry_summary,
            metadata.capability_result,
        )


def test_review_unit_evidence_public_fields_are_relation_ids() -> None:
    evidence_fields = {field.name for field in fields(BridgePrimaryUnitEvidence)}
    assert {
        "perturbation_neighbour_relation_ids", "safe_neighbour_relation_ids",
    } <= evidence_fields
    assert not {
        "perturbation_neighbour_cell_ids", "safe_neighbour_cell_ids",
    } & evidence_fields


def test_review_generic_manifest_accepts_five_animal_whole_partitions() -> None:
    metadata = synthetic_metadata(animal_count=5, neighbour_types=("astrocyte",))
    animals = tuple(f"mouse_{index}" for index in range(1, 6))
    development = animals[:3]
    evaluation = animals[3:]
    train_animals = set(animals[:2])
    tune_animals = {animals[2]}
    manifest = split_module.build_bridge_partition_manifest(
        metadata, "generic_partition:synthetic", "generic",
        tuple(train_animals), tuple(tune_animals), evaluation,
    )

    assert manifest.development_animals == development
    assert manifest.evaluation_animals == evaluation
    assert {row.animal_id for row in manifest.row_provenance if row.stable_row_id in manifest.tune_rows} == tune_animals
    with pytest.raises(SpatialPerturbationSplitError, match="nonempty train"):
        replace(
            manifest, train_rows=(),
            tune_rows=tuple(sorted(manifest.train_rows + manifest.tune_rows)),
        )
    with pytest.raises(SpatialPerturbationSplitError, match="pilot role"):
        replace(
            manifest, split_role="pilot",
            split_id="pilot_leave_one_animal_out:mouse_4",
        )
    with pytest.raises(SpatialPerturbationSplitError, match="namespace"):
        replace(manifest, split_id="pilot_leave_one_animal_out:mouse_4")
    with pytest.raises(SpatialPerturbationSplitError, match="namespace"):
        replace(manifest, split_id="generic_partition:")
    with pytest.raises(SpatialPerturbationSplitError, match="confirmatory_capable"):
        replace(
            manifest, split_role="confirmatory",
            split_id="confirmatory_partition:unauthorized",
        )


def test_review_public_relation_freezer_and_partition_builder_exist() -> None:
    freezer = getattr(split_module, "freeze_bridge_neighbour_relation", None)
    builder = getattr(split_module, "build_bridge_partition_manifest", None)
    assert callable(freezer)
    assert callable(builder)
    assert "relation_id" not in inspect.signature(freezer).parameters
    assert tuple(inspect.signature(builder).parameters) == (
        "metadata", "split_id", "split_role", "train_animals",
        "tune_animals", "evaluation_animals",
    )
    with pytest.raises(SpatialPerturbationSplitError):
        freezer(
            "mouse_1", "section_1", "block_1", _HostileSequence(),
            "neighbor_1", "guide_0", "source_type",
            "astrocyte", 1, "proximal", False, False,
        )


def test_review_relation_table_identity_is_streamed() -> None:
    metadata = synthetic_metadata()
    first_table = split_module.freeze_bridge_neighbour_table(
        metadata.neighbour_relations
    )
    second_table = split_module.freeze_bridge_neighbour_table(
        tuple(reversed(metadata.neighbour_relations))
    )
    assert first_table.identity_sha256 == second_table.identity_sha256
    manifest = build_pilot_fold(metadata, "mouse_1")
    mapping = split_manifest_to_mapping(manifest)
    split_identity = mapping.pop("split_identity_sha256")
    descriptor = cast(dict[str, object], mapping["neighbour_table"])
    assert descriptor == {
        "schema": "bridge_neighbour_table_v1",
        "relation_count": len(manifest.neighbour_relations),
        "identity_sha256": manifest.neighbour_table_identity_sha256,
    }
    assert _canonical_sha(mapping) == split_identity


def test_final_review_raw_relation_identity_has_no_matched_guide() -> None:
    relation_fields = {field.name for field in fields(BridgeNeighbourRelation)}
    assert "matched_perturbation_id" not in relation_fields
    assert "source_perturbation_id" in relation_fields
    signature = inspect.signature(split_module.freeze_bridge_neighbour_relation)
    assert "matched_perturbation_id" not in signature.parameters
    first = split_module.freeze_bridge_neighbour_relation(
        "mouse_1", "section_1", "block_1", "safe_source_1", "neighbor_1",
        "mSafe", "source_type", "astrocyte", 1, "proximal", False, True,
    )
    second = split_module.freeze_bridge_neighbour_relation(
        "mouse_1", "section_1", "block_1", "safe_source_1", "neighbor_1",
        "mSafe", "source_type", "astrocyte", 1, "proximal", False, True,
    )
    assert first.relation_id == second.relation_id


def test_final_review_public_table_freezer_owns_descriptor_and_identity() -> None:
    table_freezer = getattr(split_module, "freeze_bridge_neighbour_table", None)
    assert callable(table_freezer)
    assert {
        "freeze_bridge_neighbour_relation",
        "freeze_bridge_neighbour_table",
        "build_bridge_partition_manifest",
    } <= set(split_module.__all__)
    relation = split_module.freeze_bridge_neighbour_relation(
        "mouse_1", "section_1", "block_1", "source_1", "neighbor_1",
        "guide_0", "source_type", "astrocyte", 1, "proximal", False, False,
    )
    table = table_freezer((relation,))
    assert table.schema == "bridge_neighbour_table_v1"
    assert table.relation_count == 1
    assert table.relations == (relation,)
    assert len(table.identity_sha256) == 64
    assert not hasattr(split_module, "_neighbour_table_identity")


def test_final_review_manifest_serializer_references_external_relation_artifact() -> None:
    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    mapping = split_manifest_to_mapping(manifest)
    assert "neighbour_relations" not in mapping
    assert mapping["neighbour_table"] == {
        "schema": "bridge_neighbour_table_v1",
        "relation_count": len(manifest.neighbour_relations),
        "identity_sha256": manifest.neighbour_table_identity_sha256,
    }
    table = split_module.freeze_bridge_neighbour_table(
        manifest.neighbour_relations
    )
    serializer = getattr(split_module, "iter_bridge_neighbour_table_json", None)
    assert callable(serializer)
    chunks = tuple(serializer(table))
    assert chunks
    assert max(map(len, chunks)) < 4096
    serialized = b"".join(chunks)
    assert hashlib.sha256(serialized).hexdigest() == table.identity_sha256
    payload = json.loads(serialized)
    assert payload["schema"] == "bridge_neighbour_table_v1"
    assert len(payload["relations"]) == table.relation_count
    object.__setattr__(table, "relations", _HostileSequence())
    with pytest.raises(SpatialPerturbationSplitError):
        serializer(table)


def test_final_review_manifest_row_cap_precedes_json_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = synthetic_metadata()

    def forbidden_mapping(*args: object) -> object:
        raise AssertionError("rows were materialized before the manifest cap")

    monkeypatch.setattr(split_module, "_MAX_MANIFEST_ROWS", 1, raising=False)
    monkeypatch.setattr(split_module, "_row_mapping", forbidden_mapping)
    with pytest.raises(SpatialPerturbationSplitError, match="manifest.*row.*limit"):
        build_pilot_fold(metadata, "mouse_1")
    monkeypatch.undo()
    assert split_module._checked_manifest_row_count(
        split_module._MAX_MANIFEST_ROWS
    ) == split_module._MAX_MANIFEST_ROWS
    with pytest.raises(SpatialPerturbationSplitError, match="manifest.*row.*limit"):
        split_module._checked_manifest_row_count(
            split_module._MAX_MANIFEST_ROWS + 1
        )


def test_review_confirmatory_partition_is_bound_to_capability_and_cohorts() -> None:
    metadata = synthetic_metadata(
        animal_count=5, neighbour_types=("astrocyte",), confirmatory=True,
    )
    assert metadata.capability_result.status == "confirmatory_capable"
    manifest = split_module.build_bridge_partition_manifest(
        metadata, "confirmatory_partition:synthetic", "confirmatory",
        ("mouse_1", "mouse_2"), ("mouse_3",), ("mouse_4", "mouse_5"),
    )
    assert manifest.evaluation_animals == ("mouse_4", "mouse_5")

    with pytest.raises(SpatialPerturbationSplitError, match="nonempty.*tune|three.*nonempty"):
        split_module.build_bridge_partition_manifest(
            metadata, "confirmatory_partition:no_tune", "confirmatory",
            ("mouse_1", "mouse_2", "mouse_3"), (), ("mouse_4", "mouse_5"),
        )
    with pytest.raises(SpatialPerturbationSplitError, match="external.*cohort"):
        split_module.build_bridge_partition_manifest(
            metadata, "confirmatory_partition:cohort_mismatch", "confirmatory",
            ("mouse_1", "mouse_2"), ("mouse_4",), ("mouse_3", "mouse_5"),
        )
    pilot_only = synthetic_metadata(animal_count=5, neighbour_types=("astrocyte",))
    with pytest.raises(SpatialPerturbationSplitError, match="confirmatory_capable"):
        split_module.build_bridge_partition_manifest(
            pilot_only, "confirmatory_partition:unauthorized", "confirmatory",
            ("mouse_1", "mouse_2"), ("mouse_3",), ("mouse_4", "mouse_5"),
        )


def _metadata_with_relations(
    metadata: BridgeSplitMetadata,
    relations: tuple[BridgeNeighbourRelation, ...],
) -> BridgeSplitMetadata:
    frozen = tuple(sorted(relations, key=lambda item: item.relation_id))
    table = split_module.freeze_bridge_neighbour_table(frozen)
    return BridgeSplitMetadata(
        metadata.rows, metadata.gene_names, metadata.perturbations,
        metadata.neighbour_cell_types, metadata.perturbation_targets,
        metadata.block_adjacency, metadata.safe_control_label, frozen,
        table.identity_sha256, metadata.candidate,
        metadata.registry_summary, metadata.capability_result,
    )


def test_review_relation_logical_keys_reject_duplicates_and_cross_guide_treatment() -> None:
    metadata = synthetic_metadata(neighbour_types=("astrocyte",))
    treatment = next(
        item for item in metadata.neighbour_relations
        if not item.is_safe_control and item.source_perturbation_id == "guide_0"
    )
    guide_one_source = next(
        row for row in metadata.rows
        if row.animal_id == treatment.animal_id
        and row.section_id == treatment.section_id
        and row.context_perturbation_id == "guide_1"
        and row.cell_role == "perturbation_source"
    )
    cross_guide = split_module.freeze_bridge_neighbour_relation(
        treatment.animal_id, treatment.section_id, treatment.spatial_block,
        guide_one_source.cell_id, treatment.neighbor_cell_id,
        "guide_1", treatment.source_cell_type,
        treatment.neighbor_cell_type, treatment.rank, treatment.band,
        False, False,
    )
    with pytest.raises(SpatialPerturbationSplitError, match="treatment neighbor.*source perturbation"):
        _metadata_with_relations(
            metadata, metadata.neighbour_relations + (cross_guide,),
        )

    safe = next(item for item in metadata.neighbour_relations if item.is_safe_control)
    other_safe_source = next(
        row for row in metadata.rows
        if row.animal_id == safe.animal_id
        and row.section_id == safe.section_id
        and row.context_perturbation_id == "mSafe"
        and row.cell_role == "safe_source"
        and row.cell_id != safe.source_cell_id
    )
    logical_duplicate = split_module.freeze_bridge_neighbour_relation(
        safe.animal_id, safe.section_id, safe.spatial_block,
        other_safe_source.cell_id, safe.neighbor_cell_id, "mSafe",
        safe.source_cell_type, safe.neighbor_cell_type,
        2 if safe.band == "proximal" else 7,
        safe.band, False, True,
    )
    with pytest.raises(SpatialPerturbationSplitError, match="logical.*unique"):
        _metadata_with_relations(
            metadata, metadata.neighbour_relations + (logical_duplicate,),
        )


def test_review_safe_source_and_relation_are_reusable_across_matched_guides() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(neighbour_types=("astrocyte",)), "mouse_1"
    )
    evidence = complete_evidence(manifest)
    unit_by_id = {item.unit_id: item for item in manifest.primary_units}
    selected = tuple(
        item for item in evidence.unit_evidence
        if unit_by_id[item.unit_id].animal_id == "mouse_1"
        and unit_by_id[item.unit_id].neighbour_cell_type == "astrocyte"
        and unit_by_id[item.unit_id].band == "proximal"
    )
    assert len(selected) == len(PERTURBATIONS)
    assert len({item.safe_neighbour_relation_ids for item in selected}) == 1
    reused_ids = selected[0].safe_neighbour_relation_ids
    assert len(reused_ids) == len(set(reused_ids))


def test_review_deleting_any_block_pair_state_is_rejected() -> None:
    metadata = synthetic_metadata(adjacency=True)
    incomplete = metadata.block_adjacency[1:]
    with pytest.raises(SpatialPerturbationSplitError, match="every block pair"):
        BridgeSplitMetadata(
            metadata.rows, metadata.gene_names, metadata.perturbations,
            metadata.neighbour_cell_types, metadata.perturbation_targets,
            incomplete, metadata.safe_control_label, metadata.neighbour_relations,
            metadata.neighbour_table_identity_sha256, metadata.candidate,
            metadata.registry_summary, metadata.capability_result,
        )


def test_review_cartesian_size_is_rejected_before_unit_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = synthetic_metadata()

    def forbidden_materialization(*args: object) -> object:
        raise AssertionError("primary units were materialized before the size check")

    monkeypatch.setattr(split_module, "_make_unit", forbidden_materialization)
    monkeypatch.setattr(split_module, "_MAX_FROZEN_CONTEXTS", 1)
    with pytest.raises(SpatialPerturbationSplitError, match="Cartesian|cartesian|size|limit"):
        build_pilot_fold(metadata, "mouse_1")

    graph_checker = getattr(split_module, "_checked_complete_graph_size", None)
    assert callable(graph_checker)
    with pytest.raises(SpatialPerturbationSplitError, match="graph.*limit"):
        graph_checker((449, 1, 1))


def test_review_result_serializer_revalidates_low_level_mutation() -> None:
    serializer = getattr(split_module, "eligibility_result_to_mapping", None)
    assert callable(serializer)
    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    result = evaluate_bridge_eligibility(manifest, complete_evidence(manifest))
    object.__setattr__(result, "primary_scoreable", 999)
    with pytest.raises(SpatialPerturbationSplitError):
        serializer(result)


def _rows_for(
    manifest: BridgeSplitManifest,
    *,
    animal: str,
    perturbation: str,
    role: str,
    cell_type: str | None = None,
    band: str | None = None,
    blocks: tuple[str, ...] | None = None,
) -> tuple[BridgeSplitRow, ...]:
    return tuple(
        row for row in manifest.row_provenance
        if row.animal_id == animal
        and row.context_perturbation_id == perturbation
        and row.cell_role == role
        and (cell_type is None or row.cell_type == cell_type)
        and (band is None or row.distance_band == band)
        and (blocks is None or row.spatial_block in blocks)
    )


def complete_evidence(
    manifest: BridgeSplitManifest,
    *,
    source_overrides: dict[str, int] | None = None,
    safe_source_overrides: dict[str, int] | None = None,
    unit_overrides: dict[str, int] | None = None,
    safe_unit_overrides: dict[str, int] | None = None,
    block_overrides: dict[str, tuple[str, ...]] | None = None,
) -> BridgeEligibilityEvidence:
    source_overrides = source_overrides or {}
    safe_source_overrides = safe_source_overrides or {}
    unit_overrides = unit_overrides or {}
    safe_unit_overrides = safe_unit_overrides or {}
    block_overrides = block_overrides or {}
    parents: list[BridgeParentEvidence] = []
    for parent in manifest.perturbation_parents:
        parent_key = parent.parent_id
        source_rows = _rows_for(
            manifest, animal=parent.animal_id, perturbation=parent.perturbation_id,
            role="perturbation_source", blocks=block_overrides.get(f"{parent_key}:source"),
        )
        safe_rows = _rows_for(
            manifest, animal=parent.animal_id,
            perturbation=manifest.safe_control_label,
            role="safe_source", blocks=block_overrides.get(f"{parent_key}:safe_source"),
        )
        parents.append(
            BridgeParentEvidence(
                parent.animal_id, parent.perturbation_id, parent.target_gene,
                tuple(row.cell_id for row in source_rows[: source_overrides.get(parent_key, 20)]),
                tuple(row.cell_id for row in safe_rows[: safe_source_overrides.get(parent_key, 20)]),
            )
        )
    units: list[BridgePrimaryUnitEvidence] = []
    default_unit_count = 50 if len(manifest.neighbour_cell_types) == 1 else 30
    for unit in manifest.primary_units:
        def matching_relations(safe: bool, key: str) -> tuple[BridgeNeighbourRelation, ...]:
            blocks = block_overrides.get(key)
            return tuple(sorted((
                item for item in manifest.neighbour_relations
                if item.animal_id == unit.animal_id
                and (
                    item.is_safe_control
                    or item.source_perturbation_id == unit.perturbation_id
                )
                and item.neighbor_cell_type == unit.neighbour_cell_type
                and item.band == unit.band
                and item.is_safe_control is safe
                and (blocks is None or item.spatial_block in blocks)
            ), key=lambda item: (item.rank, item.spatial_block, item.relation_id)))
        treatment_rows = matching_relations(False, f"{unit.unit_id}:neighbour")
        safe_rows = matching_relations(True, f"{unit.unit_id}:safe_neighbour")
        units.append(
            BridgePrimaryUnitEvidence(
                unit.unit_id,
                tuple(
                    row.relation_id
                    for row in treatment_rows[
                        : unit_overrides.get(unit.unit_id, default_unit_count)
                    ]
                ),
                tuple(
                    row.relation_id
                    for row in safe_rows[
                        : safe_unit_overrides.get(unit.unit_id, default_unit_count)
                    ]
                ),
            )
        )
    return build_bridge_eligibility_evidence(manifest, tuple(parents), tuple(units))


@lru_cache(maxsize=1)
def baseline() -> tuple[BridgeSplitManifest, BridgeEligibilityEvidence]:
    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    return manifest, complete_evidence(manifest)


def test_manifest_freezes_exact_three_animal_units_and_whole_animal_fold() -> None:
    manifest, _ = baseline()
    assert manifest.development_animals == ("mouse_2", "mouse_3")
    assert manifest.evaluation_animals == ("mouse_1",)
    assert manifest.tune_rows == ()
    assert not set(manifest.development_rows) & set(manifest.evaluation_rows)
    assert len(manifest.perturbation_parents) == 15
    assert len(manifest.primary_units) == 60
    assert {unit.band for unit in manifest.primary_units} == {"proximal", "local"}
    assert {unit.neighbour_cell_type for unit in manifest.primary_units} == set(NEIGHBOUR_TYPES)


@pytest.mark.parametrize("animal_count", (1, 2, 4))
def test_pilot_rejects_any_specimen_count_other_than_three(animal_count: int) -> None:
    with pytest.raises(SpatialPerturbationSplitError, match="exactly three animals"):
        build_pilot_fold(synthetic_metadata(animal_count=animal_count), "mouse_1")


def test_evaluation_requires_revalidated_manifest_and_exact_split_binding() -> None:
    manifest, evidence = baseline()
    assert evaluate_bridge_eligibility(manifest, evidence).eligible is True
    other_manifest = build_pilot_fold(synthetic_metadata(adjacency=True), "mouse_1")
    with pytest.raises(SpatialPerturbationSplitError, match="split identity"):
        evaluate_bridge_eligibility(other_manifest, evidence)
    assert tuple(inspect.signature(evaluate_bridge_eligibility).parameters) == ("manifest", "evidence")


def test_evidence_requires_every_exact_parent_and_unit_without_deletion() -> None:
    manifest, evidence = baseline()
    missing_unit = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, evidence.unit_evidence[:-1]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="exact frozen primary units"):
        evaluate_bridge_eligibility(manifest, missing_unit)
    with pytest.raises(SpatialPerturbationSplitError, match="unique unit evidence"):
        build_bridge_eligibility_evidence(
            manifest, evidence.parent_evidence, evidence.unit_evidence + (evidence.unit_evidence[0],)
        )
    missing_parent = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence[:-1], evidence.unit_evidence
    )
    with pytest.raises(SpatialPerturbationSplitError, match="exact perturbation parents"):
        evaluate_bridge_eligibility(manifest, missing_parent)


def test_arbitrary_animal_perturbation_gene_or_unit_identity_is_rejected() -> None:
    manifest, evidence = baseline()
    parent = evidence.parent_evidence[0]
    for forged in (
        replace(parent, animal_id="invented_mouse"),
        replace(parent, perturbation_id="invented_guide"),
        replace(parent, target_gene="InventedGene"),
    ):
        altered = build_bridge_eligibility_evidence(
            manifest, (forged,) + evidence.parent_evidence[1:], evidence.unit_evidence
        )
        with pytest.raises(SpatialPerturbationSplitError, match="frozen parent context"):
            evaluate_bridge_eligibility(manifest, altered)
    forged_unit = replace(evidence.unit_evidence[0], unit_id="a" * 64)
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, (forged_unit,) + evidence.unit_evidence[1:]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="exact frozen primary units"):
        evaluate_bridge_eligibility(manifest, altered)


def test_all_selected_cells_resolve_to_expected_animal_role_band_and_type() -> None:
    manifest, evidence = baseline()
    first = evidence.unit_evidence[0]
    first_frozen = next(unit for unit in manifest.primary_units if unit.unit_id == first.unit_id)
    other_index = next(
        index for index, item in enumerate(evidence.unit_evidence)
        if next(unit for unit in manifest.primary_units if unit.unit_id == item.unit_id).animal_id
        != first_frozen.animal_id
    )
    other = evidence.unit_evidence[other_index]
    first_cell = first.safe_neighbour_relation_ids[0]
    other_cell = other.safe_neighbour_relation_ids[0]
    forged = replace(
        first,
        safe_neighbour_relation_ids=(other_cell,) + first.safe_neighbour_relation_ids[1:],
    )
    forged_other = replace(
        other,
        safe_neighbour_relation_ids=(first_cell,) + other.safe_neighbour_relation_ids[1:],
    )
    units = list(evidence.unit_evidence)
    units[0] = forged
    units[other_index] = forged_other
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, tuple(units)
    )
    with pytest.raises(SpatialPerturbationSplitError, match="expected animal/context/role/type/band|absent from frozen neighbour table"):
        evaluate_bridge_eligibility(manifest, altered)

    source_cell = evidence.parent_evidence[0].perturbation_source_cell_ids[0]
    forged = replace(
        first,
        safe_neighbour_relation_ids=(source_cell,) + first.safe_neighbour_relation_ids[1:],
    )
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, (forged,) + evidence.unit_evidence[1:]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="expected animal/context/role/type/band|absent from frozen neighbour table"):
        evaluate_bridge_eligibility(manifest, altered)


def test_non_adjacent_block_requirement_is_derived_from_frozen_provenance() -> None:
    manifest = build_pilot_fold(synthetic_metadata(adjacency=True), "mouse_1")
    evidence = complete_evidence(manifest)
    result = evaluate_bridge_eligibility(manifest, evidence)
    assert result.eligible is False
    assert "insufficient_spatial_blocks" in result.reasons
    assert "insufficient_safe_control_spatial_blocks" not in result.reasons


def test_four_blocks_with_one_adjacent_pair_has_a_valid_three_block_subset() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(
            adjacency=True, block_count=4, neighbour_types=("astrocyte",)
        ),
        "mouse_1",
    )
    result = evaluate_bridge_eligibility(manifest, complete_evidence(manifest))
    assert result.eligible is True


def _cell_ids_with_block_counts(
    manifest: BridgeSplitManifest,
    *,
    animal: str,
    perturbation: str,
    role: str,
    block_counts: tuple[int, ...],
    cell_type: str | None = None,
    band: str | None = None,
) -> tuple[str, ...]:
    selected: list[str] = []
    for block_index, count in enumerate(block_counts):
        if role in ("perturbation_neighbour", "safe_neighbour"):
            relations = sorted((
                item for item in manifest.neighbour_relations
                if item.animal_id == animal
                and (
                    item.is_safe_control
                    or item.source_perturbation_id == perturbation
                )
                and item.is_safe_control is (role == "safe_neighbour")
                and (cell_type is None or item.neighbor_cell_type == cell_type)
                and (band is None or item.band == band)
                and item.spatial_block == f"block_{block_index}"
            ), key=lambda item: (item.rank, item.relation_id))
            selected.extend(item.relation_id for item in relations[:count])
            continue
        rows = _rows_for(
            manifest, animal=animal,
            perturbation=(
                manifest.safe_control_label if role == "safe_source"
                else perturbation
            ),
            role=role,
            cell_type=cell_type, band=band, blocks=(f"block_{block_index}",),
        )
        selected.extend(row.cell_id for row in rows[:count])
    return tuple(selected)


def test_parent_matching_uses_exact_block_multisets_not_sets() -> None:
    manifest, evidence = baseline()
    parent = evidence.parent_evidence[0]
    assert len(parent.perturbation_source_cell_ids) == 20
    forged_safe = _cell_ids_with_block_counts(
        manifest, animal=parent.animal_id, perturbation=parent.perturbation_id,
        role="safe_source", block_counts=(10, 9, 1),
    )
    forged = replace(parent, safe_source_cell_ids=forged_safe)
    altered = build_bridge_eligibility_evidence(
        manifest, (forged,) + evidence.parent_evidence[1:], evidence.unit_evidence
    )
    with pytest.raises(SpatialPerturbationSplitError, match="multiset"):
        evaluate_bridge_eligibility(manifest, altered)


@pytest.mark.parametrize(
    ("source_count", "safe_count", "expected_reason", "absent_reason"),
    (
        (19, 20, "insufficient_perturbation_coverage", "insufficient_safe_control_coverage"),
        (20, 19, "insufficient_safe_control_coverage", "insufficient_perturbation_coverage"),
    ),
)
def test_asymmetric_parent_thresholds_are_ineligible_before_matching(
    source_count: int,
    safe_count: int,
    expected_reason: str,
    absent_reason: str,
) -> None:
    manifest, _ = baseline()
    parent_ids = tuple(
        parent.parent_id
        for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            source_overrides={parent_id: source_count for parent_id in parent_ids},
            safe_source_overrides={parent_id: safe_count for parent_id in parent_ids},
        ),
    )
    assert expected_reason in result.reasons
    assert absent_reason not in result.reasons


@pytest.mark.parametrize(
    ("treatment_count", "safe_count"),
    ((29, 30), (30, 29)),
)
def test_asymmetric_unit_thresholds_are_retained_as_abstentions(
    treatment_count: int, safe_count: int
) -> None:
    manifest, _ = baseline()
    unit_id = manifest.primary_units[0].unit_id
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            unit_overrides={unit_id: treatment_count},
            safe_unit_overrides={unit_id: safe_count},
        ),
    )
    assert result.eligible is True
    assert result.abstained_unit_ids == (unit_id,)


def test_neighbour_matching_rejects_disjoint_block_multisets() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(block_count=4, neighbour_types=("astrocyte",)), "mouse_1"
    )
    evidence = complete_evidence(manifest)
    item = evidence.unit_evidence[0]
    unit = next(unit for unit in manifest.primary_units if unit.unit_id == item.unit_id)
    treatment = _cell_ids_with_block_counts(
        manifest, animal=unit.animal_id, perturbation=unit.perturbation_id,
        role="perturbation_neighbour", cell_type=unit.neighbour_cell_type,
        band=unit.band, block_counts=(10, 10, 10, 0),
    )
    safe = _cell_ids_with_block_counts(
        manifest, animal=unit.animal_id, perturbation=unit.perturbation_id,
        role="safe_neighbour", cell_type=unit.neighbour_cell_type,
        band=unit.band, block_counts=(0, 10, 10, 10),
    )
    forged = replace(
        item, perturbation_neighbour_relation_ids=treatment,
        safe_neighbour_relation_ids=safe,
    )
    altered = build_bridge_eligibility_evidence(
        manifest, evidence.parent_evidence, (forged,) + evidence.unit_evidence[1:]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="multiset"):
        evaluate_bridge_eligibility(manifest, altered)


def test_row_provenance_freezes_source_cell_type() -> None:
    assert "source_cell_type" in {item.name for item in fields(BridgeSplitRow)}


def test_empty_matched_sources_are_threshold_failures_not_provenance_errors() -> None:
    manifest, evidence = baseline()
    empty = tuple(
        replace(
            parent, perturbation_source_cell_ids=(), safe_source_cell_ids=()
        )
        for parent in evidence.parent_evidence[:2]
    )
    altered = build_bridge_eligibility_evidence(
        manifest, empty + evidence.parent_evidence[2:], evidence.unit_evidence
    )
    result = evaluate_bridge_eligibility(manifest, altered)
    assert "insufficient_perturbation_coverage" in result.reasons
    assert "insufficient_safe_control_coverage" in result.reasons


@pytest.mark.parametrize("hostile_field", ("manifest", "evidence"))
def test_public_evaluation_revalidates_hostile_sequences_before_derivation(
    hostile_field: str,
) -> None:
    baseline_manifest, baseline_evidence = baseline()
    manifest = replace(baseline_manifest)
    evidence = replace(baseline_evidence)
    target = manifest if hostile_field == "manifest" else evidence
    field_name = "perturbations" if hostile_field == "manifest" else "parent_evidence"
    object.__setattr__(target, field_name, _HostileSequence())
    with pytest.raises(SpatialPerturbationSplitError):
        evaluate_bridge_eligibility(manifest, evidence)


def test_public_evaluation_normalizes_low_level_empty_perturbations_mutation() -> None:
    baseline_manifest, baseline_evidence = baseline()
    manifest = replace(baseline_manifest)
    object.__setattr__(manifest, "perturbations", ())
    with pytest.raises(SpatialPerturbationSplitError):
        evaluate_bridge_eligibility(manifest, baseline_evidence)


def test_no_module_helper_can_mint_an_arbitrary_trusted_result() -> None:
    manifest, evidence = baseline()
    mutated_manifest = replace(manifest)
    object.__setattr__(mutated_manifest, "perturbations", ("alien",))
    helper = getattr(split_module, "_trusted_eligibility_result", None)
    if helper is not None:
        forged = helper(
            mutated_manifest,
            evidence,
            split_module._DerivedEligibility(
                (), ("alien",), ("alien",), (("alien", 999, 999),),
                999, 999, 0, 999,
            ),
        )
        assert forged.primary_scoreable == 999
        assert forged.scoreable_parent_ids == ("alien",)
        assert forged.manifest.perturbations == ("alien",)
    assert not hasattr(split_module, "_trusted_eligibility_result")


def test_independent_triple_search_has_a_bounded_membership_operation_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_count = 200
    rows = tuple(
        BridgeSplitRow(
            index, f"graph_cell_{index}", "mouse", "section", f"block_{index}",
            "guide", "guide", "source", "source", "perturbation_source", "own",
        )
        for index in range(block_count)
    )
    adjacency = tuple(
        BridgeBlockAdjacency(
            "mouse", "section", f"block_{first}", f"block_{second}", True
        )
        for first in range(block_count)
        for second in range(first + 1, block_count)
    )
    real_adjacency = split_module._adjacent_block_pairs

    class CountedEdgeSet(set[frozenset[tuple[str, str, str]]]):
        iterated_edges = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            for item in super().__iter__():
                self.iterated_edges += 1
                if self.iterated_edges > len(self):
                    raise AssertionError("adjacency edges were rescanned")
                yield item

    graph_manifest = cast(
        BridgeSplitManifest, SimpleNamespace(block_adjacency=adjacency)
    )
    counted = CountedEdgeSet(real_adjacency(graph_manifest))
    assert split_module._has_non_adjacent_block_subset(
        rows, graph_manifest, 3, adjacent_pairs=counted,
    ) is False
    assert counted.iterated_edges == len(counted)
    assert "combinations" not in inspect.getsource(split_module._has_non_adjacent_block_subset)


def test_independent_triple_search_rejects_adversarial_block_cardinality() -> None:
    rows = tuple(
        BridgeSplitRow(
            index, f"huge_graph_cell_{index}", "mouse", "section", f"block_{index}",
            "guide", "guide", "source", "source", "perturbation_source", "own",
        )
        for index in range(5_000)
    )
    graph_manifest = cast(BridgeSplitManifest, SimpleNamespace(block_adjacency=()))
    with pytest.raises(SpatialPerturbationSplitError, match="block graph"):
        split_module._has_non_adjacent_block_subset(
            rows, graph_manifest, 3
        )


def test_exact_treatment_and_safe_thresholds_are_derived_from_ids() -> None:
    manifest, _ = baseline()
    parent_ids = tuple(parent.parent_id for parent in manifest.perturbation_parents if parent.animal_id == "mouse_1")
    below = {parent_ids[0]: 19, parent_ids[1]: 19}
    source_19 = complete_evidence(
        manifest, source_overrides=below, safe_source_overrides=below
    )
    source_result = evaluate_bridge_eligibility(manifest, source_19)
    assert source_result.reason == "insufficient_perturbation_coverage"
    assert ("mouse_1", 3, 5) in source_result.per_animal_perturbation_coverage
    safe_19 = complete_evidence(
        manifest, source_overrides=below, safe_source_overrides=below
    )
    assert "insufficient_safe_control_coverage" in evaluate_bridge_eligibility(manifest, safe_19).reasons

    exact = complete_evidence(manifest)
    assert evaluate_bridge_eligibility(manifest, exact).eligible is True
    assert (MIN_SOURCE_CELLS, MIN_SAFE_SOURCE_CELLS, MIN_BAND_NEIGHBOURS) == (20, 20, 50)


def _band_unit_ids(
    manifest: BridgeSplitManifest, parent_ids: tuple[str, ...], band: str
) -> tuple[tuple[str, str], ...]:
    contexts = {
        (parent.animal_id, parent.perturbation_id)
        for parent in manifest.perturbation_parents if parent.parent_id in parent_ids
    }
    grouped: list[tuple[str, str]] = []
    for animal, perturbation in sorted(contexts):
        ids = tuple(
            unit.unit_id for unit in manifest.primary_units
            if (unit.animal_id, unit.perturbation_id) == (animal, perturbation)
            and unit.band == band
        )
        assert len(ids) == 2
        grouped.append((ids[0], ids[1]))
    return tuple(grouped)


def locally_abstaining_unit_ids(
    manifest: BridgeSplitManifest, count: int
) -> tuple[str, ...]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for unit in manifest.primary_units:
        grouped.setdefault(
            (unit.animal_id, unit.perturbation_id, unit.band), []
        ).append(unit.unit_id)
    candidates = tuple(sorted(values)[0] for _, values in sorted(grouped.items()))
    return candidates[:count]


def test_exact_paired_band_threshold_49_fails_and_50_passes() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )
    pairs = _band_unit_ids(manifest, mouse_1_parents[:2], "proximal")
    below = {unit_id: count for pair in pairs for unit_id, count in zip(pair, (25, 24))}
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest, unit_overrides=below, safe_unit_overrides=below
        )
    )
    assert result.reason == "insufficient_band_neighbours"

    safe_below = {unit_id: count for pair in pairs for unit_id, count in zip(pair, (25, 24))}
    safe_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest, unit_overrides=safe_below, safe_unit_overrides=safe_below
        )
    )
    assert "insufficient_safe_control_band_neighbours" in safe_result.reasons

    exact_pair = _band_unit_ids(manifest, mouse_1_parents[:1], "proximal")[0]
    exact_50 = {unit_id: 25 for unit_id in exact_pair}
    exact_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest, unit_overrides=exact_50, safe_unit_overrides=exact_50
        )
    )
    assert "insufficient_band_neighbours" not in exact_result.reasons


def test_aggregate_fifty_does_not_rescue_band_with_all_units_abstaining() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    pairs = _band_unit_ids(manifest, mouse_1_parents, "proximal")
    exact_aggregate = {unit_id: 25 for pair in pairs for unit_id in pair}
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest,
            unit_overrides=exact_aggregate,
            safe_unit_overrides=exact_aggregate,
        )
    )
    assert ("mouse_1", 3, 5) in result.per_animal_perturbation_coverage
    assert result.eligible is False
    assert all(unit_id in result.abstained_unit_ids for unit_id in exact_aggregate)


def _cross_cell_type_compensation_evidence(
    manifest: BridgeSplitManifest,
    *,
    matched_oligodendrocytes: int,
) -> BridgeEligibilityEvidence:
    parents = tuple(
        parent for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    treatment_overrides: dict[str, int] = {}
    safe_overrides: dict[str, int] = {}
    for parent in parents:
        units = {
            unit.neighbour_cell_type: unit.unit_id
            for unit in manifest.primary_units
            if unit.animal_id == parent.animal_id
            and unit.perturbation_id == parent.perturbation_id
            and unit.band == "proximal"
        }
        treatment_overrides.update(
            {
                units["astrocyte"]: 29,
                units["microglia"]: 0,
                units["oligodendrocyte"]: matched_oligodendrocytes,
            }
        )
        safe_overrides.update(
            {
                units["astrocyte"]: 0,
                units["microglia"]: 29,
                units["oligodendrocyte"]: matched_oligodendrocytes,
            }
        )
    return complete_evidence(
        manifest,
        unit_overrides=treatment_overrides,
        safe_unit_overrides=safe_overrides,
    )


def test_cross_cell_type_compensation_does_not_satisfy_matched_band_coverage() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(
            neighbour_types=("astrocyte", "microglia", "oligodendrocyte")
        ),
        "mouse_1",
    )
    result = evaluate_bridge_eligibility(
        manifest,
        _cross_cell_type_compensation_evidence(
            manifest, matched_oligodendrocytes=30
        ),
    )
    assert ("mouse_1", 3, 5) in result.per_animal_perturbation_coverage
    assert result.eligible is False
    assert "insufficient_band_neighbours" not in result.reasons
    assert "insufficient_safe_control_band_neighbours" in result.reasons


@pytest.mark.parametrize(("matched", "eligible"), ((49, False), (50, True)))
def test_exact_matched_band_coverage_boundary(matched: int, eligible: bool) -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(
            neighbour_types=("astrocyte", "microglia", "oligodendrocyte"),
            neighbour_count=50,
        ),
        "mouse_1",
    )
    result = evaluate_bridge_eligibility(
        manifest,
        _cross_cell_type_compensation_evidence(
            manifest, matched_oligodendrocytes=matched
        ),
    )
    expected_scoreable = 5 if eligible else 3
    assert ("mouse_1", expected_scoreable, 5) in result.per_animal_perturbation_coverage
    assert result.eligible is eligible
    assert (
        "insufficient_safe_control_band_neighbours" in result.reasons
    ) is (not eligible)


def test_matched_reason_applies_only_after_both_raw_band_thresholds() -> None:
    manifest = build_pilot_fold(
        synthetic_metadata(
            neighbour_types=("astrocyte", "microglia", "oligodendrocyte")
        ),
        "mouse_1",
    )
    treatment_overrides: dict[str, int] = {}
    safe_overrides: dict[str, int] = {}
    parents = tuple(
        parent for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )[:2]
    for parent in parents:
        units = {
            unit.neighbour_cell_type: unit.unit_id
            for unit in manifest.primary_units
            if unit.animal_id == parent.animal_id
            and unit.perturbation_id == parent.perturbation_id
            and unit.band == "proximal"
        }
        treatment_overrides.update(
            {units["astrocyte"]: 19, units["microglia"]: 0}
        )
        safe_overrides.update(
            {units["astrocyte"]: 20, units["microglia"]: 0}
        )
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            unit_overrides=treatment_overrides,
            safe_unit_overrides=safe_overrides,
        ),
    )
    assert ("mouse_1", 3, 5) in result.per_animal_perturbation_coverage
    assert "insufficient_band_neighbours" in result.reasons
    assert "insufficient_safe_control_band_neighbours" not in result.reasons


def test_below_thirty_unit_is_retained_as_abstention_not_immediate_failure() -> None:
    manifest, _ = baseline()
    unit = manifest.primary_units[0]
    below = {unit.unit_id: 29}
    evidence = complete_evidence(
        manifest, unit_overrides=below, safe_unit_overrides=below
    )
    result = evaluate_bridge_eligibility(manifest, evidence)
    assert result.eligible is True
    assert result.abstained_unit_ids == (unit.unit_id,)
    assert "insufficient_band_neighbours" not in result.reasons
    safe_evidence = complete_evidence(
        manifest, unit_overrides=below, safe_unit_overrides=below
    )
    safe_result = evaluate_bridge_eligibility(manifest, safe_evidence)
    assert safe_result.eligible is True
    assert safe_result.abstained_unit_ids == (unit.unit_id,)
    assert MIN_CELL_TYPE_NEIGHBOURS == 30


def test_exact_three_pairwise_non_adjacent_blocks_pass_but_two_fail() -> None:
    manifest, _ = baseline()
    mouse_1_parents = tuple(
        parent.parent_id for parent in manifest.perturbation_parents
        if parent.animal_id == "mouse_1"
    )
    source_blocks: dict[str, tuple[str, ...]] = {
        f"{parent_id}:source": ("block_0", "block_1")
        for parent_id in mouse_1_parents[:2]
    }
    source_blocks.update(
        {
            f"{parent_id}:safe_source": ("block_0", "block_1")
            for parent_id in mouse_1_parents[:2]
        }
    )
    result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, block_overrides=source_blocks)
    )
    assert result.reason == "insufficient_spatial_blocks"

    safe_blocks: dict[str, tuple[str, ...]] = {
        f"{parent_id}:source": ("block_0", "block_1")
        for parent_id in mouse_1_parents[:2]
    }
    safe_blocks.update(
        {
            f"{parent_id}:safe_source": ("block_0", "block_1")
            for parent_id in mouse_1_parents[:2]
        }
    )
    safe_result = evaluate_bridge_eligibility(
        manifest, complete_evidence(manifest, block_overrides=safe_blocks)
    )
    assert "insufficient_safe_control_spatial_blocks" in safe_result.reasons
    assert evaluate_bridge_eligibility(manifest, complete_evidence(manifest)).eligible is True
    assert MIN_SPATIAL_BLOCKS == 3


def test_primary_coverage_and_abstention_are_derived_at_exact_boundaries() -> None:
    manifest, _ = baseline()
    exact_overrides = {
        unit_id: 29 for unit_id in locally_abstaining_unit_ids(manifest, 12)
    }
    exact = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest,
            unit_overrides=exact_overrides,
            safe_unit_overrides=exact_overrides,
        )
    )
    assert exact.primary_scoreable == 48
    assert exact.primary_total == 60
    assert exact.abstained == 12
    assert exact.attempted == 60
    assert exact.eligible is True

    below_overrides = {
        unit_id: 29 for unit_id in locally_abstaining_unit_ids(manifest, 13)
    }
    below = evaluate_bridge_eligibility(
        manifest, complete_evidence(
            manifest,
            unit_overrides=below_overrides,
            safe_unit_overrides=below_overrides,
        )
    )
    assert below.eligible is False
    assert "insufficient_primary_unit_coverage" in below.reasons
    assert "excessive_abstention" in below.reasons
    assert (MIN_COVERAGE, MAX_ABSTENTION) == (0.80, 0.20)


def test_per_animal_perturbation_coverage_exact_four_fifths_passes() -> None:
    manifest, _ = baseline()
    one_parent_each = {
        next(parent.parent_id for parent in manifest.perturbation_parents if parent.animal_id == animal): 19
        for animal in ("mouse_1", "mouse_2", "mouse_3")
    }
    result = evaluate_bridge_eligibility(
        manifest,
        complete_evidence(
            manifest,
            source_overrides=one_parent_each,
            safe_source_overrides=one_parent_each,
        ),
    )
    assert result.per_animal_perturbation_coverage == (
        ("mouse_1", 4, 5), ("mouse_2", 4, 5), ("mouse_3", 4, 5)
    )
    assert result.eligible is True


def test_controls_must_match_animal_section_source_and_neighbour_strata() -> None:
    manifest, evidence = baseline()
    parent = evidence.parent_evidence[0]
    other_parent_safe = evidence.parent_evidence[5].safe_source_cell_ids[0]
    forged_parent = replace(
        parent, safe_source_cell_ids=(other_parent_safe,) + parent.safe_source_cell_ids[1:]
    )
    altered = build_bridge_eligibility_evidence(
        manifest, (forged_parent,) + evidence.parent_evidence[1:], evidence.unit_evidence
    )
    with pytest.raises(SpatialPerturbationSplitError, match="expected animal/context/role/type/band"):
        evaluate_bridge_eligibility(manifest, altered)


def test_manifest_accessors_revalidate_low_level_mutation() -> None:
    manifest = build_pilot_fold(synthetic_metadata(), "mouse_1")
    object.__setattr__(manifest, "train_rows", manifest.train_rows + (manifest.evaluation_rows[0],))
    with pytest.raises(SpatialPerturbationSplitError, match="whole-animal partition"):
        _ = manifest.development_rows


def test_split_and_evaluation_have_no_seed_or_method_escape_hatches() -> None:
    assert tuple(inspect.signature(build_pilot_fold).parameters) == ("metadata", "evaluation_animal")
    assert "model" not in inspect.signature(evaluate_bridge_eligibility).parameters
    random.seed(1907)
    before = random.getstate()
    build_pilot_fold(synthetic_metadata(), "mouse_1")
    assert random.getstate() == before


def test_direct_records_are_closed_immutable_and_digest_mutations_fail() -> None:
    manifest, evidence = baseline()
    with pytest.raises(FrozenInstanceError):
        evidence.split_identity_sha256 = "a" * 64  # type: ignore[misc]
    raw = eligibility_evidence_to_mapping(evidence)
    raw["split_identity_sha256"] = "a" * 64
    raw.pop("evidence_identity_sha256")
    forged = BridgeEligibilityEvidence(
        **raw, evidence_identity_sha256=_canonical_sha(raw)  # type: ignore[arg-type]
    )
    with pytest.raises(SpatialPerturbationSplitError, match="split identity"):
        evaluate_bridge_eligibility(manifest, forged)
    with pytest.raises(TypeError):
        BridgeParentEvidence("a", "p", "G", (), (), extra=True)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", (True, False, -1, 10**100, 1.0, float("nan"), float("inf")))
def test_stable_row_identity_rejects_boolean_nonfinite_and_huge_values(value: object) -> None:
    row = BridgeSplitRow(
        0, "cell", "mouse", "section", "block", "guide", "guide",
        "source", "source", "perturbation_source", "own",
    )
    with pytest.raises(SpatialPerturbationSplitError):
        replace(row, stable_row_id=value)  # type: ignore[arg-type]


def test_rebound_exports_cannot_change_frozen_science(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, evidence = baseline()
    expected = evaluate_bridge_eligibility(manifest, evidence)
    for name, value in (
        ("MIN_SOURCE_CELLS", 0), ("MIN_SAFE_SOURCE_CELLS", 0),
        ("MIN_BAND_NEIGHBOURS", 0), ("MIN_CELL_TYPE_NEIGHBOURS", 0),
        ("MIN_SPATIAL_BLOCKS", 0), ("MIN_COVERAGE", 0.0),
        ("MAX_ABSTENTION", 1.0), ("_SPLIT_SEED", 99),
        ("_PRIMARY_BANDS", ("forged",)),
    ):
        monkeypatch.setattr(split_module, name, value)
    assert build_pilot_fold(synthetic_metadata(), "mouse_1") == manifest
    assert evaluate_bridge_eligibility(manifest, evidence) == expected


def test_manifest_contains_no_outcomes_effects_or_scores() -> None:
    manifest, _ = baseline()
    mapping = split_manifest_to_mapping(manifest)
    assert not ({"outcome", "effect", "score", "prediction", "rmse"} & set(mapping))
