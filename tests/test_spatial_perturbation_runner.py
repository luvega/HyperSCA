from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from functools import lru_cache
import importlib.util
import os
import shutil
from collections import UserDict
from pathlib import Path
import subprocess
import sys
from types import FrameType, MappingProxyType

import numpy as np
import pandas as pd
import pytest
import yaml

from src.methods_protocol_v3_contract import (
    MethodsProtocolV3,
    build_methods_protocol_v3,
    protocol_identity_v3,
    protocol_to_mapping_v3,
)

from src.evaluation.run_evidence_identity import (
    RunEvidenceError,
    RunEvidenceIdentity,
    canonical_json_bytes,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import (
    RunEvidencePublisher,
    VerifiedRunEvidence,
    verify_run_evidence_bundle,
)
from src.evaluation.spatial_perturbation_predictor_contract import (
    PREDICTION_SCHEMA,
    BridgePredictionBundle,
    BridgePredictorContractError,
    audit_bridge_predictor_capability,
    build_bridge_prediction_bundle,
)
from src.evaluation.spatial_perturbation_runner import (
    bridge_run_data_identity_sha256,
    publish_spatial_perturbation_run,
    runner_code_identity_sha256,
)
from src.evaluation.spatial_perturbation_comparators import BridgeModelBudget
from src.evaluation.spatial_perturbation_neighbors import build_bridge_neighbors
from src.evaluation.spatial_perturbation_registry import audit_bridge_capability
from src.evaluation.spatial_perturbation_scoring import (
    BridgeEffect,
    BridgePrediction,
    replay_published_bridge_effect_units,
)
from src.evaluation.spatial_perturbation_split import (
    BridgeParentEvidence,
    BridgePrimaryUnitEvidence,
    BridgeSplitMetadata,
    BridgeSplitRow,
    build_bridge_eligibility_evidence,
    build_pilot_fold,
    evaluate_bridge_eligibility,
    freeze_bridge_neighbour_relation,
    freeze_bridge_neighbour_table,
)
from tests.test_spatial_perturbation_scoring import (
    _evaluation_predictions,
    _expression_for_eligibility,
    _standardizer_for_manifest,
)
from tests.test_spatial_perturbation_split import synthetic_metadata


SCIENTIFIC_ARTIFACTS = {
    "split_manifest.json",
    "capability_record.json",
    "neighbor_units.csv",
    "predictions_hypersca.csv",
    "predictions_matched_euclidean.csv",
    "predictions_hypersca_own_only.csv",
    "primary_metric_units.csv",
    "primary_metric_summary.json",
    "secondary_metrics.csv",
    "resource_usage.json",
    "claim_decision.json",
}


def declarations() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "schema_version": "spatial_perturbation_bridge_candidates_v1",
            "candidates": [
                {
                    "candidate_id": "gse274447_msafe_bridge",
                    "accession": "GSE274447",
                    "platform": "spatial_perturbation",
                    "biological_specimens": ["mouse_1", "mouse_2", "mouse_3"],
                    "sections_by_specimen": [
                        ["mouse_1", []], ["mouse_2", []], ["mouse_3", []]
                    ],
                    "safe_control_label": "mSafe",
                    "perturbation_labels": [],
                    "source_uri": (
                        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447"
                    ),
                    "source_identity_sha256": (
                        "0e908ba2f21cab2bd222daf31a85ff8369407c8df53f5d9a2424f081528ffa46"
                    ),
                }
            ],
        },
        protocol_to_mapping_v3(
            build_methods_protocol_v3(
                bridge_role="pilot_audit_only",
                capability_identity_sha256="f" * 64,
            )
        ),
    )


def capability():
    registry, protocol = declarations()
    return audit_bridge_predictor_capability(
        registry, protocol, method_id="hypersca"
    )


def prediction_bytes(
    predictions: tuple[BridgePrediction, ...], *, own_only_neighbor: float = 0.0
) -> bytes:
    baseline = [
        {
            "unit_id": item.unit_id,
            "endpoint": item.endpoint,
            "predicted_effect": item.predicted_delta,
            "effect_units": "train_control_standardized_delta",
        }
        for item in predictions
    ]
    own_only = [dict(row) for row in baseline]
    for row in own_only:
        if row["endpoint"] == "neighbor":
            row["predicted_effect"] = own_only_neighbor
    return canonical_json_bytes(
        {
            "schema_version": "1.0",
            "origin": "synthetic_fixture",
            "predictions": {
                "hypersca": baseline,
                "matched_euclidean_spatial_causal": [dict(row) for row in baseline],
                "hypersca_own_only": own_only,
            },
        }
    )


def unrelated_neighbor_cells() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "mouse_1",
                "mouse_1_section",
                "block_1",
                "source",
                "guide_0",
                "source_type",
                0.0,
                0.0,
                True,
            ),
            (
                "mouse_1",
                "mouse_1_section",
                "block_1",
                "neighbor_a",
                "unperturbed",
                "astrocyte",
                1.0,
                0.0,
                False,
            ),
            (
                "mouse_1",
                "mouse_1_section",
                "block_1",
                "neighbor_b",
                "unperturbed",
                "microglia",
                2.0,
                0.0,
                False,
            ),
        ],
        columns=(
            "animal_id",
            "section_id",
            "spatial_block",
            "cell_id",
            "perturbation_id",
            "cell_type",
            "x",
            "y",
            "barcode_positive",
        ),
    )


def budgets(
    data_identity: str, *, seed: int = 11
) -> tuple[BridgeModelBudget, BridgeModelBudget]:
    common = {
        "parameter_count": 100,
        "optimizer_family": "adamw",
        "max_updates": 100,
        "early_stopping_patience": 10,
        "tuning_trials": 2,
        "data_identity_sha256": data_identity,
        "gene_identity_sha256": "1" * 64,
        "spatial_graph_identity_sha256": "2" * 64,
        "propagation_identity_sha256": "3" * 64,
        "seed": seed,
    }
    return (
        BridgeModelBudget(method_id="hypersca", geometry="hyperbolic", **common),
        BridgeModelBudget(
            method_id="matched_euclidean_spatial_causal",
            geometry="euclidean",
            **common,
        ),
    )


@lru_cache(maxsize=1)
def _coherent_case() -> dict[str, object]:
    base = synthetic_metadata(neighbour_types=("astrocyte",))
    selected_perturbations = base.perturbations[:2]
    selected_genes = base.gene_names[:2]
    raw_rows: list[tuple[object, ...]] = []
    split_rows: list[BridgeSplitRow] = []
    row_id = 0
    for animal_index in range(3):
        animal = f"mouse_{animal_index + 1}"
        for source_index in range(20):
            section = f"{animal}_section_{source_index:02d}"
            block = f"block_{source_index % 3}"
            for perturbation_index, source_perturbation in enumerate(
                (*selected_perturbations, "mSafe")
            ):
                source_x = float(perturbation_index * 1_000)
                source_id = (
                    f"{animal}_{source_index:02d}_{source_perturbation}_source"
                )
                raw_rows.append(
                    (
                        animal, section, block, source_id, source_perturbation,
                        "source_type", source_x, 0.0, True,
                    )
                )
                split_rows.append(
                    BridgeSplitRow(
                        row_id, source_id, animal, section, block,
                        source_perturbation, source_perturbation,
                        "source_type", "source_type",
                        "safe_source" if source_perturbation == "mSafe" else "perturbation_source",
                        "own",
                    )
                )
                row_id += 1
                for rank in range(1, 61 if source_index < 10 else 1):
                    neighbor_id = f"{source_id}_neighbor_{rank:02d}"
                    raw_rows.append(
                        (
                            animal, section, block, neighbor_id, "unperturbed",
                            "astrocyte", source_x + float(rank), 0.0, False,
                        )
                    )
                    split_rows.append(
                        BridgeSplitRow(
                            row_id, neighbor_id, animal, section, block,
                            "unassigned", "unperturbed", "astrocyte", "astrocyte",
                            "neighbour", "none",
                        )
                    )
                    row_id += 1
    cells = pd.DataFrame(
        raw_rows,
        columns=(
            "animal_id", "section_id", "spatial_block", "cell_id",
            "perturbation_id", "cell_type", "x", "y", "barcode_positive",
        ),
    )
    task6_table = build_bridge_neighbors(cells)
    relations = tuple(
        freeze_bridge_neighbour_relation(*row)
        for row in task6_table.itertuples(index=False, name=None)
    )
    relation_table = freeze_bridge_neighbour_table(relations)
    animals = tuple(f"mouse_{index + 1}" for index in range(3))
    sections = tuple(
        (
            animal,
            tuple(sorted({
                row.section_id
                for row in split_rows
                if row.animal_id == animal and row.cell_role.endswith("source")
            })),
        )
        for animal in animals
    )
    from dataclasses import replace

    candidate = replace(
        base.candidate,
        sections_by_specimen=sections,
        perturbation_labels=selected_perturbations,
    )
    total_rows = len(split_rows)
    per_animal_rows = tuple(
        (animal, sum(row.animal_id == animal for row in split_rows))
        for animal in animals
    )
    per_animal_sources = tuple(
        (
            animal,
            sum(row.animal_id == animal and row.cell_role == "perturbation_source" for row in split_rows),
        )
        for animal in animals
    )
    per_animal_safe = tuple(
        (
            animal,
            sum(row.animal_id == animal and row.cell_role == "safe_source" for row in split_rows),
        )
        for animal in animals
    )
    summary = replace(
        base.registry_summary,
        sections_by_specimen=sections,
        coordinate_count=total_rows,
        measured_gene_names=selected_genes,
        measured_gene_count=len(selected_genes),
        perturbation_labels=selected_perturbations,
        perturbation_label_counts=tuple(
            (
                perturbation,
                sum(row.observed_label == perturbation for row in split_rows),
            )
            for perturbation in selected_perturbations
        ),
        safe_control_counts=(("mSafe", sum(count for _, count in per_animal_safe)),),
        barcode_quality_counts=(("valid", total_rows),),
        label_quality_counts=(("valid", total_rows),),
        per_specimen_coordinate_counts=per_animal_rows,
        per_specimen_perturbation_counts=per_animal_sources,
        per_specimen_safe_control_counts=per_animal_safe,
        per_specimen_barcode_valid_counts=per_animal_rows,
        per_specimen_label_valid_counts=per_animal_rows,
    )
    metadata = BridgeSplitMetadata(
        tuple(split_rows), selected_genes, selected_perturbations,
        base.neighbour_cell_types, base.perturbation_targets[:2], (), "mSafe",
        relation_table.relations, relation_table.identity_sha256,
        candidate, summary, audit_bridge_capability(candidate, summary),
    )
    manifest = build_pilot_fold(metadata, "mouse_1")
    parents = []
    for parent in manifest.perturbation_parents:
        treatment = tuple(
            row for row in manifest.row_provenance
            if row.animal_id == parent.animal_id
            and row.context_perturbation_id == parent.perturbation_id
            and row.cell_role == "perturbation_source"
        )
        treatment_sections = {row.section_id for row in treatment}
        safe = tuple(
            row for row in manifest.row_provenance
            if row.animal_id == parent.animal_id
            and row.context_perturbation_id == manifest.safe_control_label
            and row.cell_role == "safe_source"
            and row.section_id in treatment_sections
        )
        parents.append(
            BridgeParentEvidence(
                parent.animal_id, parent.perturbation_id, parent.target_gene,
                tuple(row.cell_id for row in treatment),
                tuple(row.cell_id for row in safe),
            )
        )
    units = []
    for unit in manifest.primary_units:
        treatment = tuple(
            relation for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.source_perturbation_id == unit.perturbation_id
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and not relation.is_safe_control
        )
        treatment_sections = {relation.section_id for relation in treatment}
        safe = tuple(
            relation for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.source_perturbation_id == manifest.safe_control_label
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and relation.is_safe_control
            and relation.section_id in treatment_sections
        )
        units.append(
            BridgePrimaryUnitEvidence(
                unit.unit_id,
                tuple(relation.relation_id for relation in treatment),
                tuple(relation.relation_id for relation in safe),
            )
        )
    evidence = build_bridge_eligibility_evidence(
        manifest, tuple(parents), tuple(units)
    )
    eligibility = evaluate_bridge_eligibility(manifest, evidence)
    assert eligibility.eligible
    expression, cell_ids = _expression_for_eligibility(eligibility)
    standardizer = _standardizer_for_manifest(manifest)
    predictions = _evaluation_predictions(eligibility)
    data_identity = bridge_run_data_identity_sha256(
        split_manifest=manifest,
        neighbor_cells=cells,
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
    )
    hypersca_budget, euclidean_budget = budgets(data_identity)
    return {
        "eligibility": eligibility,
        "expression": expression,
        "cell_ids": cell_ids,
        "gene_names": standardizer.genes,
        "standardizer": standardizer,
        "predictions": predictions,
        "neighbor_cells": cells,
        "data_identity": data_identity,
        "hypersca_budget": hypersca_budget,
        "matched_euclidean_budget": euclidean_budget,
    }


def run_case() -> dict[str, object]:
    return dict(_coherent_case())


def run_case_for_seed(seed: int) -> dict[str, object]:
    case = run_case()
    baseline, comparator = budgets(case["data_identity"], seed=seed)
    case["hypersca_budget"] = baseline
    case["matched_euclidean_budget"] = comparator
    return case


def formal_protocol(case: dict[str, object]) -> MethodsProtocolV3:
    return build_methods_protocol_v3(
        bridge_role="pilot_audit_only",
        capability_identity_sha256=(
            case["eligibility"].manifest.capability_identity_sha256
        ),
    )


def bundle(case: dict[str, object], **changes: object) -> BridgePredictionBundle:
    predictions = case["predictions"]
    unit_record = {
        "units": [
            [item.unit_id, item.endpoint]
            for item in predictions
        ]
    }
    values: dict[str, object] = {
        "method_id": "hypersca",
        "protocol_identity_sha256": protocol_identity_v3(formal_protocol(case)),
        "data_identity_sha256": case["data_identity"],
        "split_identity_sha256": case["eligibility"].manifest.split_identity_sha256,
        "statistical_unit_identity_sha256": canonical_sha256(unit_record),
        "code_identity_sha256": runner_code_identity_sha256(),
        "model_seed": 11,
        "prediction_schema": PREDICTION_SCHEMA,
        "prediction_bytes": prediction_bytes(predictions),
        "origin": "synthetic_fixture",
        "evidence_role": "synthetic_audit_only",
    }
    values.update(changes)
    return build_bridge_prediction_bundle(**values)


def publish_case(tmp_path: Path, **bundle_changes: object) -> VerifiedRunEvidence:
    case = run_case()
    return publish_case_inputs(
        case, tmp_path / "synthetic", **bundle_changes
    )


def publish_case_inputs(
    case: dict[str, object], output_dir: Path, **bundle_changes: object
) -> VerifiedRunEvidence:
    return publish_spatial_perturbation_run(
        bundle(case, **bundle_changes),
        output_dir=output_dir,
        protocol=formal_protocol(case),
        split_manifest=case["eligibility"].manifest,
        neighbor_cells=case["neighbor_cells"],
        expression=case["expression"],
        cell_ids=case["cell_ids"],
        gene_names=case["gene_names"],
        standardizer=case["standardizer"],
        eligibility=case["eligibility"],
        hypersca_budget=case["hypersca_budget"],
        matched_euclidean_budget=case["matched_euclidean_budget"],
    )


@pytest.fixture(scope="module")
def published_bridge_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> VerifiedRunEvidence:
    return publish_case_inputs(
        run_case(), tmp_path_factory.mktemp("bridge-baseline") / "bundle"
    )


def copy_published_bridge_baseline(
    baseline: VerifiedRunEvidence, destination: Path
) -> VerifiedRunEvidence:
    shutil.copytree(baseline.output_dir, destination)
    return replace(baseline, output_dir=destination)


def publish_failed_bridge_seed(
    baseline: VerifiedRunEvidence, destination: Path, *, model_seed: int
) -> VerifiedRunEvidence:
    identity = replace(
        baseline.identity,
        model_seed=model_seed,
        config_identity_sha256=f"{model_seed:064x}",
    )
    publisher = RunEvidencePublisher.begin(
        output_dir=destination,
        identity=identity,
        statistical_unit_record={
            "units": [
                [item.unit_id, item.endpoint]
                for item in run_case()["predictions"]
            ]
        },
        required_artifacts=(),
        maximum_bundle_bytes=32 * 1024 * 1024,
    )
    output = publisher.finalize_failure(
        status="failed_runtime", reason="registered worker failure"
    )
    return verify_run_evidence_bundle(output, expected_identity=identity)


def mutable_run_case() -> dict[str, object]:
    case = run_case()
    case["expression"] = case["expression"].copy()
    case["neighbor_cells"] = case["neighbor_cells"].copy(deep=True)
    data_identity = bridge_run_data_identity_sha256(
        split_manifest=case["eligibility"].manifest,
        neighbor_cells=case["neighbor_cells"],
        expression=case["expression"],
        cell_ids=case["cell_ids"],
        gene_names=case["gene_names"],
    )
    case["data_identity"] = data_identity
    case["hypersca_budget"], case["matched_euclidean_budget"] = budgets(
        data_identity
    )
    return case


def relabelled_formal_run_case() -> dict[str, object]:
    from dataclasses import replace

    case = run_case()
    manifest = case["eligibility"].manifest
    source_uri = (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123456"
    )
    candidate = replace(
        manifest.candidate,
        candidate_id="formal_candidate",
        accession="GSE123456",
        source_uri=source_uri,
        source_identity_sha256="9" * 64,
    )
    summary = replace(
        manifest.registry_summary,
        candidate_id="formal_candidate",
        accession="GSE123456",
        source_identity_sha256="9" * 64,
    )
    capability_result = audit_bridge_capability(candidate, summary)
    metadata = BridgeSplitMetadata(
        manifest.row_provenance,
        manifest.gene_names,
        manifest.registered_perturbations,
        manifest.neighbour_cell_types,
        manifest.perturbation_targets,
        manifest.block_adjacency,
        manifest.safe_control_label,
        manifest.neighbour_relations,
        manifest.neighbour_table_identity_sha256,
        candidate,
        summary,
        capability_result,
    )
    changed_manifest = build_pilot_fold(metadata, "mouse_1")
    evidence = build_bridge_eligibility_evidence(
        changed_manifest,
        case["eligibility"].evidence.parent_evidence,
        case["eligibility"].evidence.unit_evidence,
    )
    eligibility = evaluate_bridge_eligibility(changed_manifest, evidence)
    assert eligibility.eligible
    expression, cell_ids = _expression_for_eligibility(eligibility)
    standardizer = _standardizer_for_manifest(changed_manifest)
    changed: dict[str, object] = dict(case)
    changed.update(
        {
            "eligibility": eligibility,
            "expression": expression,
            "cell_ids": cell_ids,
            "standardizer": standardizer,
            "predictions": _evaluation_predictions(eligibility),
        }
    )
    data_identity = bridge_run_data_identity_sha256(
        split_manifest=changed_manifest,
        neighbor_cells=changed["neighbor_cells"],
        expression=expression,
        cell_ids=cell_ids,
        gene_names=standardizer.genes,
    )
    changed["data_identity"] = data_identity
    changed["hypersca_budget"], changed["matched_euclidean_budget"] = budgets(
        data_identity
    )
    return changed


def test_runner_requires_exact_v3_protocol_and_bound_identity(tmp_path: Path) -> None:
    case = run_case()
    formal = formal_protocol(case)
    object.__setattr__(formal, "protocol_version", "hypersca-methods-v3")
    with pytest.raises((BridgePredictorContractError, ValueError), match="protocol"):
        publish_spatial_perturbation_run(
            bundle(case),
            output_dir=tmp_path / "bad-version",
            protocol=formal,
            split_manifest=case["eligibility"].manifest,
            neighbor_cells=case["neighbor_cells"],
            expression=case["expression"],
            cell_ids=case["cell_ids"],
            gene_names=case["gene_names"],
            standardizer=case["standardizer"],
            eligibility=case["eligibility"],
            hypersca_budget=case["hypersca_budget"],
            matched_euclidean_budget=case["matched_euclidean_budget"],
        )
    with pytest.raises(BridgePredictorContractError, match="protocol identity"):
        publish_spatial_perturbation_run(
            bundle(case, protocol_identity_sha256="a" * 64),
            output_dir=tmp_path / "bad-identity",
            protocol=formal_protocol(case),
            split_manifest=case["eligibility"].manifest,
            neighbor_cells=case["neighbor_cells"],
            expression=case["expression"],
            cell_ids=case["cell_ids"],
            gene_names=case["gene_names"],
            standardizer=case["standardizer"],
            eligibility=case["eligibility"],
            hypersca_budget=case["hypersca_budget"],
            matched_euclidean_budget=case["matched_euclidean_budget"],
        )


def test_runner_rejects_partially_scoreable_ineligible_evidence_before_output(
    tmp_path: Path,
) -> None:
    case = run_case()
    manifest = case["eligibility"].manifest
    original_evidence = case["eligibility"].evidence
    target_unit = next(
        unit
        for unit in manifest.primary_units
        if unit.animal_id == "mouse_2" and unit.band == "local"
    )
    relation_by_id = {
        relation.relation_id: relation for relation in manifest.neighbour_relations
    }
    unit_evidence = []
    for item in original_evidence.unit_evidence:
        if item.unit_id != target_unit.unit_id:
            unit_evidence.append(item)
            continue
        unit_evidence.append(
            BridgePrimaryUnitEvidence(
                item.unit_id,
                tuple(
                    relation_id
                    for relation_id in item.perturbation_neighbour_relation_ids
                    if relation_by_id[relation_id].section_id.endswith(
                        ("_00", "_01")
                    )
                ),
                item.safe_neighbour_relation_ids,
            )
        )
    evidence = build_bridge_eligibility_evidence(
        manifest,
        original_evidence.parent_evidence,
        tuple(unit_evidence),
    )
    ineligible = evaluate_bridge_eligibility(manifest, evidence)
    assert not ineligible.eligible
    assert ineligible.reason == "insufficient_spatial_blocks"
    assert 0 < ineligible.primary_scoreable < ineligible.primary_total

    expression, cell_ids = _expression_for_eligibility(ineligible)
    case["eligibility"] = ineligible
    case["expression"] = expression
    case["cell_ids"] = cell_ids
    case["predictions"] = _evaluation_predictions(ineligible)
    case["data_identity"] = bridge_run_data_identity_sha256(
        split_manifest=manifest,
        neighbor_cells=case["neighbor_cells"],
        expression=expression,
        cell_ids=cell_ids,
        gene_names=case["gene_names"],
    )
    case["hypersca_budget"], case["matched_euclidean_budget"] = budgets(
        case["data_identity"]
    )
    output = tmp_path / "ineligible"
    with pytest.raises(BridgePredictorContractError, match="eligib"):
        publish_case_inputs(case, output)
    assert not output.exists()


def test_runner_does_not_rehash_full_eligibility_mapping_with_bundle_json_limit(
    tmp_path: Path,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module
    import src.evaluation.spatial_perturbation_split as split_module

    case = run_case()
    mapping_replays = 0

    class ScoringBoundaryReached(Exception):
        pass

    eligibility_code = split_module.eligibility_result_to_mapping.__code__
    canonical_code = runner_module.canonical_sha256.__code__
    prediction_frame_code = runner_module._prediction_frames.__code__

    def observe(frame: FrameType, event: str, argument: object) -> None:
        nonlocal mapping_replays
        if event != "call":
            return
        if (
            frame.f_code is eligibility_code
            and frame.f_locals.get("result") is case["eligibility"]
        ):
            mapping_replays += 1
        elif frame.f_code is canonical_code:
            candidate = frame.f_locals.get("value")
            if type(candidate) is dict and "eligibility" in candidate:
                candidate = candidate["eligibility"]
            if (
                type(candidate) is dict
                and "eligible" in candidate
                and "manifest" in candidate
                and "evidence" in candidate
            ):
                raise AssertionError(
                    "full eligibility mapping reached the 4 MiB bundle JSON hasher"
                )
        elif frame.f_code is prediction_frame_code:
            raise ScoringBoundaryReached

    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        with pytest.raises(ScoringBoundaryReached):
            publish_case_inputs(case, tmp_path / "never-staged")
    finally:
        sys.setprofile(previous_profile)
    assert mapping_replays == 1
    assert not (tmp_path / "never-staged").exists()


def test_runner_does_not_hash_full_replayed_split_with_run_evidence_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module
    import src.evaluation.spatial_perturbation_split as split_module

    case = run_case()
    real_replay = split_module.split_manifest_to_mapping
    replay_count = 0

    def replay(value: object) -> dict[str, object]:
        nonlocal replay_count
        replay_count += 1
        return real_replay(value)

    original_sha = runner_module.canonical_sha256

    def guarded_sha(value: object) -> str:
        if (
            type(value) is dict
            and "split_identity_sha256" in value
            and "neighbour_table" in value
            and "row_provenance" in value
        ):
            raise AssertionError(
                "full split mapping reached the 4 MiB bundle JSON hasher"
            )
        return original_sha(value)

    monkeypatch.setattr(split_module, "split_manifest_to_mapping", replay)
    monkeypatch.setattr(runner_module, "canonical_sha256", guarded_sha)

    assert bridge_run_data_identity_sha256(
        split_manifest=case["eligibility"].manifest,
        neighbor_cells=case["neighbor_cells"],
        expression=case["expression"],
        cell_ids=case["cell_ids"],
        gene_names=case["gene_names"],
    ) == case["data_identity"]
    assert replay_count == 1


def test_bridge_split_serializer_is_canonical_above_four_mib_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    def forbidden_helper(value: object) -> bytes:
        raise AssertionError("run-evidence canonical helper was called")

    monkeypatch.setattr(runner_module, "canonical_json_bytes", forbidden_helper)
    mapping = {
        "padding": "x" * (4 * 1024 * 1024),
        "schema_version": "1.0",
        "split_identity_sha256": "a" * 64,
    }

    payload = runner_module._split_manifest_json_bytes(
        mapping, maximum_bytes=8 * 1024 * 1024
    )
    assert len(payload) > 4 * 1024 * 1024
    assert json.loads(payload) == mapping
    with pytest.raises(BridgePredictorContractError, match="size|resource"):
        runner_module._split_manifest_json_bytes(
            mapping, maximum_bytes=4 * 1024 * 1024
        )


def test_terminal_capability_publishes_only_two_audit_artifacts(tmp_path: Path) -> None:
    result = publish_spatial_perturbation_run(
        capability(), output_dir=tmp_path / "failure"
    )

    assert type(result) is VerifiedRunEvidence
    assert result.terminal_status == "method_adapter_not_executable"
    assert result.identity.claim_id == "bridge"
    assert result.identity.evidence_role == "pilot_audit_only"
    assert tuple(item.relative_path for item in result.artifacts) == (
        "capability_record.json",
        "resource_usage.json",
    )
    assert {item.name for item in result.output_dir.iterdir()} == {
        "capability_record.json",
        "resource_usage.json",
        "method_status.json",
        "run_manifest.json",
    }
    assert verify_run_evidence_bundle(
        result.output_dir, expected_identity=result.identity
    ) == result


def test_terminal_failure_returns_committed_evidence_after_external_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_input = capability()
    output = tmp_path / "post-failure-finalize-mutation"
    original_fsync = os.fsync
    mutated = False

    def fsync_then_mutate(file_descriptor: int) -> None:
        nonlocal mutated
        original_fsync(file_descriptor)
        if output.exists() and not mutated:
            mutated = True
            object.__setattr__(result_input, "registry_identity_sha256", "a" * 64)

    monkeypatch.setattr(os, "fsync", fsync_then_mutate)
    result = publish_spatial_perturbation_run(result_input, output_dir=output)

    assert mutated
    assert result.terminal_status == "method_adapter_not_executable"
    assert verify_run_evidence_bundle(
        output, expected_identity=result.identity
    ) == result


def test_runner_rejects_bundle_resource_limit_above_frozen_maximum(
    tmp_path: Path,
) -> None:
    with pytest.raises((RunEvidenceError, BridgePredictorContractError), match="byte|resource"):
        publish_spatial_perturbation_run(
            capability(),
            output_dir=tmp_path / "oversized-limit",
            maximum_bundle_bytes=32 * 1024 * 1024 + 1,
        )
    assert not (tmp_path / "oversized-limit").exists()


@pytest.mark.parametrize("bad_limit", (True, 0, -1, 1.5))
def test_runner_resource_limit_requires_an_exact_positive_integer(
    tmp_path: Path, bad_limit: object
) -> None:
    with pytest.raises(BridgePredictorContractError, match="resource"):
        publish_spatial_perturbation_run(
            capability(),
            output_dir=tmp_path / "bad-resource",
            maximum_bundle_bytes=bad_limit,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "bad-resource").exists()


def test_runner_rejects_hostile_protocol_mapping_subclasses(tmp_path: Path) -> None:
    case = run_case()
    hostile = UserDict(protocol_to_mapping_v3(formal_protocol(case)))

    with pytest.raises(BridgePredictorContractError, match="protocol"):
        publish_spatial_perturbation_run(
            bundle(case),
            output_dir=tmp_path / "hostile-protocol",
            protocol=hostile,
        )
    assert not (tmp_path / "hostile-protocol").exists()


def test_raw_neighbor_subclass_is_rejected_before_to_csv_is_called() -> None:
    case = run_case()

    class HostileFrame(pd.DataFrame):
        def to_csv(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("hostile to_csv executed before exact type validation")

    hostile = HostileFrame(case["neighbor_cells"])
    with pytest.raises(BridgePredictorContractError, match="DataFrame"):
        bridge_run_data_identity_sha256(
            split_manifest=case["eligibility"].manifest,
            neighbor_cells=hostile,
            expression=case["expression"],
            cell_ids=case["cell_ids"],
            gene_names=case["gene_names"],
        )


def test_exact_raw_frame_cannot_override_trusted_snapshot_serializer() -> None:
    case = run_case()
    cells = case["neighbor_cells"].copy(deep=True)

    def hostile_to_csv(*args: object, **kwargs: object) -> str:
        raise AssertionError("user to_csv override executed")

    cells.to_csv = hostile_to_csv
    assert bridge_run_data_identity_sha256(
        split_manifest=case["eligibility"].manifest,
        neighbor_cells=cells,
        expression=case["expression"],
        cell_ids=case["cell_ids"],
        gene_names=case["gene_names"],
    ) == case["data_identity"]


def test_full_publication_never_dispatches_raw_frame_to_csv_override(
    tmp_path: Path,
) -> None:
    case = mutable_run_case()
    called = False

    def hostile_to_csv(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        raise AssertionError("user to_csv override executed during publication")

    case["neighbor_cells"].to_csv = hostile_to_csv
    result = publish_case_inputs(case, tmp_path / "trusted-raw-serializer")

    assert result.terminal_status == "completed"
    assert not called


def test_runner_explicitly_rejects_tampered_nonsynthetic_success(
    tmp_path: Path,
) -> None:
    case = run_case()
    source = bundle(case)
    object.__setattr__(source, "origin", "production")
    object.__setattr__(source, "evidence_role", "pilot_audit_only")

    with pytest.raises(BridgePredictorContractError):
        publish_spatial_perturbation_run(
            source,
            output_dir=tmp_path / "production-bypass",
            protocol=formal_protocol(case),
        )
    assert not (tmp_path / "production-bypass").exists()


def test_synthetic_origin_cannot_relabel_a_formal_candidate_and_data(
    tmp_path: Path,
) -> None:
    case = relabelled_formal_run_case()
    source = bundle(case)
    assert source.origin == "synthetic_fixture"
    assert source.evidence_role == "synthetic_audit_only"
    assert case["eligibility"].manifest.candidate.accession == "GSE123456"

    output = tmp_path / "relabelled-formal-data"
    with pytest.raises(BridgePredictorContractError, match="synthetic fixture"):
        publish_case_inputs(case, output)
    assert not output.exists()


@pytest.mark.parametrize("target", ("expression", "neighbors", "support"))
def test_runner_aborts_when_support_inputs_mutate_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    case = mutable_run_case()
    original_fsync = os.fsync
    mutated = False

    def fsync_then_mutate(file_descriptor: int) -> None:
        nonlocal mutated
        original_fsync(file_descriptor)
        if not mutated:
            mutated = True
            if target == "expression":
                case["expression"][0, 0] += 0.25
            elif target == "neighbors":
                case["neighbor_cells"].loc[0, "x"] += 0.25
            else:
                object.__setattr__(case["hypersca_budget"], "seed", 23)

    monkeypatch.setattr(os, "fsync", fsync_then_mutate)
    output = tmp_path / f"mutated-{target}"
    with pytest.raises((RunEvidenceError, BridgePredictorContractError, ValueError)):
        publish_case_inputs(case, output)
    assert mutated
    assert not output.exists()


def test_runner_snapshots_raw_neighbors_before_first_task6_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed_case = mutable_run_case()
    original_cells = sealed_case["neighbor_cells"].copy(deep=True)
    sealed_case["neighbor_cells"].loc[0, "x"] += 0.001
    data_identity = bridge_run_data_identity_sha256(
        split_manifest=sealed_case["eligibility"].manifest,
        neighbor_cells=sealed_case["neighbor_cells"],
        expression=sealed_case["expression"],
        cell_ids=sealed_case["cell_ids"],
        gene_names=sealed_case["gene_names"],
    )
    sealed_case["data_identity"] = data_identity
    (
        sealed_case["hypersca_budget"],
        sealed_case["matched_euclidean_budget"],
    ) = budgets(data_identity)
    run_input = bundle(sealed_case)
    calls = 0
    original_to_csv = pd.DataFrame.to_csv

    def serialize_then_mutate(
        cells: pd.DataFrame, *args: object, **kwargs: object
    ) -> str:
        nonlocal calls
        payload = original_to_csv(cells, *args, **kwargs)
        calls += 1
        if calls == 1:
            cells.loc[0, "x"] += 0.001
        return payload

    monkeypatch.setattr(pd.DataFrame, "to_csv", serialize_then_mutate)
    output = tmp_path / "first-task6-mutation"
    with pytest.raises(BridgePredictorContractError, match="neighbor|changed"):
        publish_spatial_perturbation_run(
            run_input,
            output_dir=output,
            protocol=formal_protocol(sealed_case),
            split_manifest=sealed_case["eligibility"].manifest,
            neighbor_cells=original_cells,
            expression=sealed_case["expression"],
            cell_ids=sealed_case["cell_ids"],
            gene_names=sealed_case["gene_names"],
            standardizer=sealed_case["standardizer"],
            eligibility=sealed_case["eligibility"],
            hypersca_budget=sealed_case["hypersca_budget"],
            matched_euclidean_budget=sealed_case[
                "matched_euclidean_budget"
            ],
        )
    assert calls >= 1
    assert not output.exists()


def test_runner_returns_committed_success_after_external_post_finalize_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = mutable_run_case()
    output = tmp_path / "post-finalize-mutation"
    original_fsync = os.fsync
    mutated = False

    def fsync_then_mutate(file_descriptor: int) -> None:
        nonlocal mutated
        original_fsync(file_descriptor)
        if output.exists() and not mutated:
            mutated = True
            case["expression"][0, 0] += 0.25

    monkeypatch.setattr(os, "fsync", fsync_then_mutate)
    result = publish_case_inputs(case, output)

    assert mutated
    assert result.terminal_status == "completed"
    assert result.output_dir == output
    assert verify_run_evidence_bundle(
        output, expected_identity=result.identity
    ) == result


@pytest.mark.parametrize(
    "bad",
    [lambda: None, "predictions.csv", Path("predictions.csv"), {}, object()],
)
def test_runner_accepts_only_exact_contract_objects(tmp_path: Path, bad: object) -> None:
    with pytest.raises((RunEvidenceError, BridgePredictorContractError)):
        publish_spatial_perturbation_run(bad, output_dir=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_synthetic_bundle_publishes_exact_eleven_scientific_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_neighbors as neighbor_module
    import src.evaluation.spatial_perturbation_runner as runner_module
    import src.evaluation.spatial_perturbation_scoring as scoring_module
    import src.evaluation.spatial_perturbation_split as split_module
    import src.evaluation.spatial_perturbation_comparators as comparator_module

    counts = {"split": 0, "neighbors": 0, "score": 0, "comparators": 0}
    def counted(name: str, function: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            counts[name] += 1
            return function(*args, **kwargs)  # type: ignore[operator]

        return wrapper

    case = run_case()
    prevalidated_code_identity = runner_code_identity_sha256()
    monkeypatch.setattr(
        runner_module, "_source_identity", lambda: prevalidated_code_identity
    )
    monkeypatch.setattr(
        split_module,
        "split_manifest_to_mapping",
        counted("split", split_module.split_manifest_to_mapping),
    )
    monkeypatch.setattr(
        neighbor_module,
        "build_bridge_neighbors",
        counted("neighbors", neighbor_module.build_bridge_neighbors),
    )
    monkeypatch.setattr(
        scoring_module,
        "score_bridge_predictions",
        counted("score", scoring_module.score_bridge_predictions),
    )
    monkeypatch.setattr(
        comparator_module,
        "validate_bridge_comparator_predictions",
        counted(
            "comparators", comparator_module.validate_bridge_comparator_predictions
        ),
    )
    result = publish_case_inputs(case, tmp_path / "synthetic")

    assert result.terminal_status == "completed"
    assert result.identity.claim_id == "bridge"
    assert result.identity.data_scopes == ("synthetic",)
    assert result.identity.evidence_role == "synthetic_audit_only"
    assert {item.relative_path for item in result.artifacts} == SCIENTIFIC_ARTIFACTS
    assert {item.name for item in result.output_dir.iterdir()} == SCIENTIFIC_ARTIFACTS | {
        "method_status.json",
        "run_manifest.json",
    }
    decision = json.loads(
        (result.output_dir / "claim_decision.json").read_text(encoding="utf-8")
    )
    assert set(decision) == {
        "analysis_record",
        "claim_id",
        "decision",
        "schema_version",
        "synthetic_fixture_identity_sha256",
    }
    assert decision["claim_id"] == "bridge"
    assert decision["decision"] == "synthetic_audit_only_no_scientific_claim"
    assert canonical_sha256(decision["analysis_record"]) == (
        result.identity.analysis_identity_sha256
    )
    analysis = decision["analysis_record"]
    assert len(analysis["comparator_budgets"]) == 2
    assert analysis["support_record"]["comparator_budgets"] == (
        analysis["comparator_budgets"]
    )
    assert canonical_sha256(analysis["support_record"]) == (
        analysis["support_identity_sha256"]
    )
    assert canonical_sha256(analysis["observed_effect_projection"]) == (
        analysis["observed_effect_projection_identity_sha256"]
    )
    for row in analysis["observed_effect_projection"]["effects"]:
        assert (
            float.fromhex(row["treatment_mean_hex"])
            - float.fromhex(row["safe_control_mean_hex"])
        ).hex() == row["observed_delta_hex"]
    assert analysis["input_identity_sha256"] == result.identity.input_identity_sha256
    assert analysis["input_identity_sha256"] != analysis["data_identity_sha256"]
    summary = json.loads(
        (result.output_dir / "primary_metric_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["hypersca"]["neighbor_effect_rmse"] == pytest.approx(0.0)
    assert summary["hypersca"]["own_effect_rmse"] == pytest.approx(0.0)
    assert summary["hypersca"]["coverage"] == pytest.approx(1.0)
    resources = json.loads(
        (result.output_dir / "resource_usage.json").read_text(encoding="utf-8")
    )
    assert resources["maximum_bundle_bytes"] == 32 * 1024 * 1024
    assert type(resources["maximum_bundle_bytes"]) is int
    assert resources["neighbor_max_rank"] == 60
    assert type(resources["neighbor_max_rank"]) is int
    assert resources["model_seed"] == 11
    assert type(resources["model_seed"]) is int
    assert resources["prediction_payload_bytes"] == len(bundle(case).prediction_bytes)
    primary_units = pd.read_csv(result.output_dir / "primary_metric_units.csv")
    assert tuple(primary_units.columns) == (
        "method_id",
        "unit_id",
        "endpoint",
        "animal_id",
        "perturbation_id",
        "gene_name",
        "neighbor_cell_type",
        "band",
        "observed_delta",
        "predicted_delta",
        "effect_identity_sha256",
        "evaluation_neighbor_unit_count",
        "evaluation_calibration_context_count",
    )
    assert set(primary_units["endpoint"]) == {"neighbor", "own"}
    published_neighbors = pd.read_csv(result.output_dir / "neighbor_units.csv")
    expected_neighbors = build_bridge_neighbors(case["neighbor_cells"])
    pd.testing.assert_frame_equal(published_neighbors, expected_neighbors)
    assert set(published_neighbors["band"]) == {
        "proximal", "local", "transition", "distal"
    }
    assert counts["split"] >= 1
    assert counts["neighbors"] >= 1
    assert counts["score"] >= 3
    assert counts["comparators"] >= 1


def test_bridge_collection_requires_the_complete_preregistered_pilot_seed_set(
    tmp_path: Path,
) -> None:
    from src.evaluation.run_evidence_collection import validate_paired_collection

    runs = tuple(
        publish_case_inputs(
            run_case_for_seed(seed), tmp_path / f"synthetic-seed-{seed}",
            model_seed=seed,
        )
        for seed in (11, 23, 47)
    )
    complete = validate_paired_collection(
        tuple(reversed(runs)), expected_model_seeds=(11, 23, 47)
    )
    assert complete.statistics_allowed is True
    assert tuple(run.identity.model_seed for run in complete.runs) == (11, 23, 47)
    assert len(
        {
            run.identity.analysis_identity_sha256
            for run in runs
        }
    ) == 3
    assert len(
        {
            run.summary["analysis_contract_identity_sha256"]
            for run in runs
            if run.summary is not None
        }
    ) == 1
    with pytest.raises(RunEvidenceError, match="preregistered seed set"):
        validate_paired_collection(
            runs[:2], expected_model_seeds=(11, 23)
        )
    with pytest.raises(RunEvidenceError, match="preregistered seed set"):
        validate_paired_collection(
            runs, expected_model_seeds=(11, 23, 47, 59)
        )


def test_mixed_status_bridge_collection_cannot_shrink_the_seed_contract(
    tmp_path: Path, published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    from src.evaluation.run_evidence_collection import validate_paired_collection

    failed_23 = publish_failed_bridge_seed(
        published_bridge_baseline, tmp_path / "failed-23", model_seed=23
    )
    with pytest.raises(RunEvidenceError, match="preregistered seed set"):
        validate_paired_collection(
            (published_bridge_baseline, failed_23),
            expected_model_seeds=(11, 23),
        )

    failed_47 = publish_failed_bridge_seed(
        published_bridge_baseline, tmp_path / "failed-47", model_seed=47
    )
    complete = validate_paired_collection(
        (published_bridge_baseline, failed_23, failed_47),
        expected_model_seeds=(11, 23, 47),
    )
    assert complete.statistics_allowed is False


def test_bridge_publication_performs_one_fresh_code_owned_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.evaluation import spatial_perturbation_runner as runner_module

    case = run_case()
    run_bundle = bundle(case)
    protocol = formal_protocol(case)
    code_identity = run_bundle.code_identity_sha256
    real_builder = runner_module._build_code_owned_synthetic_fixture
    replays = 0

    def counted_builder() -> dict[str, object]:
        nonlocal replays
        replays += 1
        return real_builder()

    monkeypatch.setattr(
        runner_module, "_build_code_owned_synthetic_fixture", counted_builder
    )
    monkeypatch.setattr(runner_module, "_source_identity", lambda: code_identity)
    publish_spatial_perturbation_run(
        run_bundle,
        output_dir=tmp_path / "single-fresh-replay",
        protocol=protocol,
        split_manifest=case["eligibility"].manifest,
        neighbor_cells=case["neighbor_cells"],
        expression=case["expression"],
        cell_ids=case["cell_ids"],
        gene_names=case["gene_names"],
        standardizer=case["standardizer"],
        eligibility=case["eligibility"],
        hypersca_budget=case["hypersca_budget"],
        matched_euclidean_budget=case["matched_euclidean_budget"],
    )
    assert replays == 1


def test_code_owned_synthetic_builder_has_a_bounded_local_runtime() -> None:
    import time

    from src.evaluation.spatial_perturbation_runner import (
        _build_code_owned_synthetic_fixture,
    )

    started = time.monotonic()
    _build_code_owned_synthetic_fixture()
    assert time.monotonic() - started < 30.0


def test_nonpreregistered_pilot_seed_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    case = run_case_for_seed(13)
    with pytest.raises(BridgePredictorContractError, match="seed|protocol"):
        publish_case_inputs(
            case, tmp_path / "synthetic-seed-13", model_seed=13
        )
    assert not (tmp_path / "synthetic-seed-13").exists()


def test_bridge_semantic_replay_rejects_forged_resealed_artifact(
    tmp_path: Path, published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = published_bridge_baseline
    for artifact_name in (
        "claim_decision.json",
        "primary_metric_summary.json",
        "predictions_hypersca.csv",
        "resource_usage.json",
    ):
        forged = tmp_path / f"forged-{artifact_name.replace('.', '-')}"
        shutil.copytree(original.output_dir, forged)
        artifact_path = forged / artifact_name
        if artifact_name == "claim_decision.json":
            value = json.loads(artifact_path.read_text(encoding="utf-8"))
            value["decision"] = "promoted_scientific_claim"
            payload = canonical_json_bytes(value)
        elif artifact_name == "primary_metric_summary.json":
            value = json.loads(artifact_path.read_text(encoding="utf-8"))
            value["hypersca"]["neighbor_effect_rmse"] = 999.0
            payload = canonical_json_bytes(value)
        elif artifact_name == "resource_usage.json":
            value = json.loads(artifact_path.read_text(encoding="utf-8"))
            value["model_fitted"] = True
            payload = canonical_json_bytes(value)
        else:
            text = artifact_path.read_text(encoding="utf-8")
            lines = text.splitlines()
            first_row = lines[1].split(",")
            first_row[2] = "999.0"
            lines[1] = ",".join(first_row)
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            assert payload != text.encode("utf-8")
        artifact_path.write_bytes(payload)

        manifest_path = forged / "run_manifest.json"
        status_path = forged / "method_status.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        artifact_record = next(
            item
            for item in manifest["artifacts"]
            if item["relative_path"] == artifact_name
        )
        artifact_record["size_bytes"] = len(payload)
        artifact_record["sha256"] = hashlib.sha256(payload).hexdigest()
        inventory_identity = canonical_sha256(manifest["artifacts"])
        manifest["artifact_inventory_sha256"] = inventory_identity
        status["artifact_inventory_sha256"] = inventory_identity
        manifest["terminal_status"] = status
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        status_path.write_bytes(canonical_json_bytes(status))
        generic = verify_run_evidence_bundle(
            forged, expected_identity=original.identity
        )

        with pytest.raises(RunEvidenceError, match="semantic|bridge|claim|artifact"):
            runner_module.verify_spatial_perturbation_evidence_bundle(
                forged, expected_identity=original.identity
            )
        with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
            from src.evaluation.run_evidence_collection import validate_paired_collection

            validate_paired_collection(
                (generic,), expected_model_seeds=(original.identity.model_seed,)
            )


def _synchronously_reseal_bridge_controls(directory: Path) -> RunEvidenceIdentity:
    import src.evaluation.spatial_perturbation_runner as runner_module

    manifest_path = directory / "run_manifest.json"
    status_path = directory / "method_status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    claim = json.loads((directory / "claim_decision.json").read_text(encoding="utf-8"))
    analysis = claim["analysis_record"]
    analysis_contract = runner_module._analysis_contract_record(analysis)
    analysis["analysis_contract"] = analysis_contract
    analysis["analysis_contract_identity_sha256"] = canonical_sha256(
        analysis_contract
    )
    manifest["summary"]["analysis_contract_identity_sha256"] = analysis[
        "analysis_contract_identity_sha256"
    ]
    manifest["summary_sha256"] = canonical_sha256(manifest["summary"])
    manifest["terminal_status"]["summary_sha256"] = manifest["summary_sha256"]
    status["summary_sha256"] = manifest["summary_sha256"]
    analysis["artifact_identities_sha256"] = {
        item["relative_path"]: hashlib.sha256(
            (directory / item["relative_path"]).read_bytes()
        ).hexdigest()
        for item in manifest["artifacts"]
        if item["relative_path"] != "claim_decision.json"
    }
    (directory / "claim_decision.json").write_bytes(canonical_json_bytes(claim))
    identity_record = manifest["run_identity"]
    identity_record["analysis_identity_sha256"] = canonical_sha256(analysis)
    identity = RunEvidenceIdentity.from_record(identity_record)
    manifest["run_identity"] = identity.to_record()
    manifest["run_identity_sha256"] = identity.run_identity_sha256
    status["run_identity_sha256"] = identity.run_identity_sha256
    for artifact in manifest["artifacts"]:
        payload = (directory / artifact["relative_path"]).read_bytes()
        artifact["size_bytes"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    inventory_identity = canonical_sha256(manifest["artifacts"])
    manifest["artifact_inventory_sha256"] = inventory_identity
    status["artifact_inventory_sha256"] = inventory_identity
    manifest["terminal_status"] = status
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    status_path.write_bytes(canonical_json_bytes(status))
    return identity


def test_bridge_semantic_replay_rejects_synchronized_semantic_resealing(
    tmp_path: Path, published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = published_bridge_baseline
    for forgery in (
        "prediction",
        "metric",
        "resource",
        "synthetic_contrasts",
        "neighbor_rank_identity",
        "development_animal_order",
        "observed",
        "budget_identities",
        "resource_cap",
        "analysis",
        "neighbor_input_identity",
        "neighbor_identity",
        "synthetic_identity",
    ):
        forged = tmp_path / f"synchronized-{forgery}"
        shutil.copytree(original.output_dir, forged)
        claim_path = forged / "claim_decision.json"
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        manifest_path = forged / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if forgery == "prediction":
            prediction_path = forged / "predictions_hypersca.csv"
            lines = prediction_path.read_text(encoding="utf-8").splitlines()
            values = lines[1].split(",")
            values[2] = "999.0"
            lines[1] = ",".join(values)
            prediction_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
            payloads = {
                name: (forged / name).read_bytes()
                for name in (
                    "predictions_hypersca.csv",
                    "predictions_matched_euclidean.csv",
                    "predictions_hypersca_own_only.csv",
                )
            }
            prediction_payload, _ = runner_module._prediction_payload_from_csv(
                payloads, origin="synthetic_fixture"
            )
            capability_path = forged / "capability_record.json"
            capability = json.loads(capability_path.read_text(encoding="utf-8"))
            bundle_record = capability["prediction_bundle"]
            bundle_record["prediction_bytes_sha256"] = hashlib.sha256(
                prediction_payload
            ).hexdigest()
            unsigned = dict(bundle_record)
            unsigned.pop("prediction_bundle_identity_sha256")
            bundle_identity = canonical_sha256(unsigned)
            bundle_record["prediction_bundle_identity_sha256"] = bundle_identity
            capability["prediction_bundle_identity_sha256"] = bundle_identity
            capability_path.write_bytes(canonical_json_bytes(capability))
            resource_path = forged / "resource_usage.json"
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            resource["prediction_payload_bytes"] = len(prediction_payload)
            resource_path.write_bytes(canonical_json_bytes(resource))
            claim["analysis_record"]["prediction_bytes_sha256"] = hashlib.sha256(
                prediction_payload
            ).hexdigest()
            claim["analysis_record"][
                "prediction_bundle_identity_sha256"
            ] = bundle_identity
            manifest["run_identity"]["config_identity_sha256"] = bundle_identity
            manifest["summary"][
                "prediction_bundle_identity_sha256"
            ] = bundle_identity
            manifest["summary_sha256"] = canonical_sha256(manifest["summary"])
            manifest["terminal_status"]["summary_sha256"] = manifest[
                "summary_sha256"
            ]
            status = json.loads((forged / "method_status.json").read_text("utf-8"))
            status["summary_sha256"] = manifest["summary_sha256"]
            (forged / "method_status.json").write_bytes(canonical_json_bytes(status))
            manifest_path.write_bytes(canonical_json_bytes(manifest))
        elif forgery == "metric":
            metric_path = forged / "primary_metric_summary.json"
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            metric["hypersca"]["neighbor_effect_rmse"] = 999.0
            metric_path.write_bytes(canonical_json_bytes(metric))
        elif forgery == "resource":
            resource_path = forged / "resource_usage.json"
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            resource["prediction_payload_bytes"] += 1
            resource_path.write_bytes(canonical_json_bytes(resource))
        elif forgery == "development_animal_order":
            split_path = forged / "split_manifest.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            split["development_animals"] = list(
                reversed(split["development_animals"])
            )
            split_path.write_bytes(canonical_json_bytes(split))
        elif forgery == "neighbor_rank_identity":
            neighbors_path = forged / "neighbor_units.csv"
            neighbors = pd.read_csv(neighbors_path)
            neighbors.loc[0, "rank"] = int(neighbors.loc[0, "rank"]) + 1
            neighbor_payload = runner_module._csv_bytes(neighbors)
            neighbors_path.write_bytes(neighbor_payload)
            claim["analysis_record"]["neighbor_units_sha256"] = hashlib.sha256(
                neighbor_payload
            ).hexdigest()
        elif forgery in ("observed", "synthetic_contrasts"):
            units_path = forged / "primary_metric_units.csv"
            units = pd.read_csv(units_path)
            units["observed_delta"] = 0.0
            units["effect_identity_sha256"] = [
                canonical_sha256(
                    {
                        "unit_id": row.unit_id,
                        "endpoint": row.endpoint,
                        "animal_id": row.animal_id,
                        "perturbation_id": row.perturbation_id,
                        "gene_name": row.gene_name,
                        "neighbor_cell_type": row.neighbor_cell_type,
                        "band": row.band,
                        "observed_delta_hex": float(row.observed_delta).hex(),
                        "predicted_delta_hex": float(row.predicted_delta).hex(),
                    }
                )
                for row in units.itertuples(index=False)
            ]
            units_path.write_bytes(runner_module._csv_bytes(units))
            split = json.loads((forged / "split_manifest.json").read_text("utf-8"))
            evaluation_animals = set(split["evaluation_animals"])
            evaluation_units = tuple(
                unit
                for unit in split["primary_units"]
                if unit["animal_id"] in evaluation_animals
            )
            contexts = tuple(sorted({
                (
                    unit["animal_id"],
                    unit["perturbation_id"],
                    unit["neighbour_cell_type"],
                    unit["target_gene"],
                )
                for unit in evaluation_units
            }))
            primary: dict[str, object] = {"schema_version": "1.0"}
            secondary_rows: list[dict[str, object]] = []
            scoring_identities: list[str] = []
            for method in (
                "hypersca",
                "matched_euclidean_spatial_causal",
                "hypersca_own_only",
            ):
                method_rows = units.loc[units["method_id"] == method]
                effects = tuple(
                    BridgeEffect(
                        row.unit_id,
                        row.endpoint,
                        row.animal_id,
                        row.perturbation_id,
                        row.gene_name,
                        row.neighbor_cell_type,
                        row.band,
                        float(row.observed_delta),
                        float(row.predicted_delta),
                        row.effect_identity_sha256,
                    )
                    for row in method_rows.itertuples(index=False)
                )
                replay = replay_published_bridge_effect_units(
                    effects,
                    expected_calibration_contexts=contexts,
                    evaluation_neighbor_unit_count=len(evaluation_units),
                    split_identity_sha256=claim["analysis_record"][
                        "split_identity_sha256"
                    ],
                    neighbour_table_identity_sha256=split["neighbour_table"][
                        "identity_sha256"
                    ],
                    eligibility_identity_sha256=claim["analysis_record"][
                        "eligibility_identity_sha256"
                    ],
                    standardizer_identity_sha256=claim["analysis_record"][
                        "standardizer_identity_sha256"
                    ],
                )
                primary[method] = {
                    "neighbor_effect_rmse": replay.neighbor_effect_rmse,
                    "own_effect_rmse": replay.own_effect_rmse,
                    "coverage": replay.coverage,
                    "abstention": replay.abstention,
                    "distance_calibration_eligible_pairs": (
                        replay.distance_calibration_eligible_pairs
                    ),
                    "distance_calibration_total_contexts": (
                        replay.distance_calibration_total_contexts
                    ),
                    "distance_calibration_coverage": (
                        replay.distance_calibration_coverage
                    ),
                    "distance_calibration_abstention": (
                        replay.distance_calibration_abstention
                    ),
                    "effect_table_identity_sha256": (
                        replay.effect_table_identity_sha256
                    ),
                    "scoring_identity_sha256": replay.scoring_identity_sha256,
                }
                scoring_identities.append(replay.scoring_identity_sha256)
                for metric in (
                    "neighbor_effect_pcc",
                    "distance_decay_calibration_error",
                    "effect_sign_accuracy",
                ):
                    secondary_rows.append(
                        {
                            "method_id": method,
                            "metric_id": metric,
                            "value": getattr(replay, metric),
                        }
                    )
            (forged / "primary_metric_summary.json").write_bytes(
                canonical_json_bytes(primary)
            )
            (forged / "secondary_metrics.csv").write_bytes(
                runner_module._csv_bytes(pd.DataFrame(secondary_rows))
            )
            claim["analysis_record"]["scoring_identities"] = scoring_identities
            if forgery == "synthetic_contrasts":
                projection = claim["analysis_record"][
                    "observed_effect_projection"
                ]
                for row in projection["effects"]:
                    row["treatment_mean_hex"] = 0.0.hex()
                    row["safe_control_mean_hex"] = 0.0.hex()
                    row["observed_delta_hex"] = 0.0.hex()
                projection_identity = canonical_sha256(projection)
                claim["analysis_record"][
                    "observed_effect_projection_identity_sha256"
                ] = projection_identity
                input_identity = canonical_sha256(
                    {
                        "schema_version": "bridge_run_input_v2",
                        "data_identity_sha256": claim["analysis_record"][
                            "data_identity_sha256"
                        ],
                        "observed_effect_projection_identity_sha256": (
                            projection_identity
                        ),
                    }
                )
                claim["analysis_record"]["input_identity_sha256"] = input_identity
                manifest["run_identity"]["input_identity_sha256"] = input_identity
                manifest_path.write_bytes(canonical_json_bytes(manifest))
        elif forgery == "budget_identities":
            claim["analysis_record"]["comparator_budget_identity_sha256"] = "0" * 64
            claim["analysis_record"]["support_identity_sha256"] = "0" * 64
        elif forgery == "resource_cap":
            resource_path = forged / "resource_usage.json"
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            artifact_total = sum(
                (forged / item["relative_path"]).stat().st_size
                for item in manifest["artifacts"]
            )
            resource["maximum_bundle_bytes"] = artifact_total + 128
            resource_path.write_bytes(canonical_json_bytes(resource))
        elif forgery == "analysis":
            claim["analysis_record"]["unknown_axis"] = "forged"
        elif forgery == "neighbor_input_identity":
            claim["analysis_record"]["neighbor_input_sha256"] = "0" * 64
        elif forgery == "neighbor_identity":
            claim["analysis_record"]["neighbor_units_sha256"] = "0" * 64
        else:
            claim["synthetic_fixture_identity_sha256"] = "0" * 64
            claim["analysis_record"]["synthetic_fixture_identity_sha256"] = "0" * 64
            capability_path = forged / "capability_record.json"
            capability = json.loads(capability_path.read_text(encoding="utf-8"))
            capability["synthetic_fixture_identity_sha256"] = "0" * 64
            capability_path.write_bytes(canonical_json_bytes(capability))
        claim_path.write_bytes(canonical_json_bytes(claim))
        identity = _synchronously_reseal_bridge_controls(forged)
        generic = verify_run_evidence_bundle(forged, expected_identity=identity)

        with pytest.raises(
            RunEvidenceError, match="bridge|semantic|metric|artifact|identity|analysis"
        ):
            runner_module.verify_spatial_perturbation_evidence_bundle(
                forged, expected_identity=identity
            )
        from src.evaluation.run_evidence_collection import validate_paired_collection

        with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
            validate_paired_collection(
                (generic,), expected_model_seeds=(identity.model_seed,)
            )


def test_bridge_semantic_replay_never_uses_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = copy_published_bridge_baseline(
        published_bridge_baseline, tmp_path / "safe-semantic-read"
    )

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unsafe Path.read_bytes attempted for {self.name}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    replay = runner_module.verify_spatial_perturbation_evidence_bundle(
        original.output_dir, expected_identity=original.identity
    )
    assert replay == original


def test_bridge_semantic_replay_rejects_oversize_before_artifact_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = copy_published_bridge_baseline(
        published_bridge_baseline, tmp_path / "oversize-semantic-read"
    )
    oversized = replace(
        original,
        artifacts=(
            replace(original.artifacts[0], size_bytes=32 * 1024 * 1024 + 1),
            *original.artifacts[1:],
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "verify_run_evidence_bundle",
        lambda path, *, expected_identity=None: oversized,
    )

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"oversize artifact was read: {self.name}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    with pytest.raises(RunEvidenceError, match="resource|size|32"):
        runner_module.verify_spatial_perturbation_evidence_bundle(
            original.output_dir, expected_identity=original.identity
        )


def test_bridge_semantic_replay_rejects_generic_to_semantic_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = published_bridge_baseline
    swapped = tmp_path / "swapped"
    shutil.copytree(original.output_dir, swapped)
    external = tmp_path / "external-summary.json"
    summary = swapped / "primary_metric_summary.json"
    external.write_bytes(summary.read_bytes())
    generic_verify = verify_run_evidence_bundle
    swapped_once = False

    def generic_then_swap(
        path: Path | str, *, expected_identity: object = None
    ) -> VerifiedRunEvidence:
        nonlocal swapped_once
        replay = generic_verify(path, expected_identity=expected_identity)  # type: ignore[arg-type]
        if not swapped_once:
            summary.unlink()
            summary.symlink_to(external)
            swapped_once = True
        return replay

    monkeypatch.setattr(runner_module, "verify_run_evidence_bundle", generic_then_swap)
    with pytest.raises(RunEvidenceError, match="artifact|link|semantic"):
        runner_module.verify_spatial_perturbation_evidence_bundle(
            swapped, expected_identity=original.identity
        )


def test_bridge_semantic_replay_rejects_output_directory_replaced_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = copy_published_bridge_baseline(
        published_bridge_baseline, tmp_path / "directory-swap"
    )
    moved = tmp_path / "directory-swap-moved"
    original_read = os.read
    generic_verify = verify_run_evidence_bundle
    armed = False
    swapped = False

    def generic_then_arm(
        path: Path | str, *, expected_identity: object = None
    ) -> VerifiedRunEvidence:
        nonlocal armed
        replay = generic_verify(path, expected_identity=expected_identity)  # type: ignore[arg-type]
        armed = True
        return replay

    def swap_directory_then_read(file_descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if armed and not swapped:
            swapped = True
            original.output_dir.rename(moved)
            original.output_dir.mkdir()
        return original_read(file_descriptor, size)

    monkeypatch.setattr(runner_module, "verify_run_evidence_bundle", generic_then_arm)
    monkeypatch.setattr(os, "read", swap_directory_then_read)
    with pytest.raises(RunEvidenceError, match="directory|path|artifact|semantic"):
        runner_module.verify_spatial_perturbation_evidence_bundle(
            original.output_dir, expected_identity=original.identity
        )
    assert swapped


def test_bridge_semantic_replay_rejects_parent_directory_replaced_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_bridge_baseline: VerifiedRunEvidence,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    parent = tmp_path / "bound-parent"
    parent.mkdir()
    original = copy_published_bridge_baseline(
        published_bridge_baseline, parent / "bundle"
    )
    moved_parent = tmp_path / "bound-parent-moved"
    original_read = os.read
    generic_verify = verify_run_evidence_bundle
    armed = False
    swapped = False

    def generic_then_arm(
        path: Path | str, *, expected_identity: object = None
    ) -> VerifiedRunEvidence:
        nonlocal armed
        replay = generic_verify(path, expected_identity=expected_identity)  # type: ignore[arg-type]
        armed = True
        return replay

    def swap_parent_then_read(file_descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if armed and not swapped:
            swapped = True
            parent.rename(moved_parent)
            parent.mkdir()
            (moved_parent / "bundle").rename(parent / "bundle")
        return original_read(file_descriptor, size)

    monkeypatch.setattr(runner_module, "verify_run_evidence_bundle", generic_then_arm)
    monkeypatch.setattr(os, "read", swap_parent_then_read)
    with pytest.raises(RunEvidenceError, match="directory|path|artifact|semantic"):
        runner_module.verify_spatial_perturbation_evidence_bundle(
            original.output_dir, expected_identity=original.identity
        )
    assert swapped


def test_runner_source_reader_rejects_oversize_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    source = tmp_path / "oversize.py"
    with source.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024 + 1)

    def forbidden_read(file_descriptor: int, size: int) -> bytes:
        raise AssertionError("oversize source was read")

    monkeypatch.setattr(os, "read", forbidden_read)
    with pytest.raises(BridgePredictorContractError, match="source|resource|size"):
        runner_module._source_bytes(source)


def test_runner_source_reader_rejects_cumulative_bound_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    source = tmp_path / "cumulative.py"
    source.write_bytes(b"ok\n")

    def forbidden_read(file_descriptor: int, size: int) -> bytes:
        raise AssertionError("source beyond the cumulative budget was read")

    monkeypatch.setattr(os, "read", forbidden_read)
    with pytest.raises(BridgePredictorContractError, match="source|resource|bound"):
        runner_module._source_bytes(source, remaining_total_bytes=2)


def test_csv_serializers_reject_oversize_shape_before_to_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    oversized = pd.DataFrame({"value": range(300_001)})

    def forbidden_to_csv(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversize frame was serialized")

    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden_to_csv)
    with pytest.raises(BridgePredictorContractError, match="resource|row|size"):
        runner_module._csv_bytes(oversized)


def test_task6_neighbors_must_equal_the_task5_manifest_relation_table(
    tmp_path: Path,
) -> None:
    case = run_case()
    rank_15 = build_bridge_neighbors(case["neighbor_cells"], max_rank=15)
    rank_15_table = freeze_bridge_neighbour_table(tuple(
        freeze_bridge_neighbour_relation(*row)
        for row in rank_15.itertuples(index=False, name=None)
    ))
    assert rank_15_table.identity_sha256 != (
        case["eligibility"].manifest.neighbour_table_identity_sha256
    )
    case["neighbor_cells"] = unrelated_neighbor_cells()

    with pytest.raises(BridgePredictorContractError, match="neighbor.*manifest"):
        publish_spatial_perturbation_run(
            bundle(case),
            output_dir=tmp_path / "disconnected",
            protocol=formal_protocol(case),
            split_manifest=case["eligibility"].manifest,
            neighbor_cells=case["neighbor_cells"],
            expression=case["expression"],
            cell_ids=case["cell_ids"],
            gene_names=case["gene_names"],
            standardizer=case["standardizer"],
            eligibility=case["eligibility"],
            hypersca_budget=case["hypersca_budget"],
            matched_euclidean_budget=case["matched_euclidean_budget"],
        )
    assert not (tmp_path / "disconnected").exists()


def test_runner_revalidates_direct_bundle_tampering_before_publication(
    tmp_path: Path,
) -> None:
    case = run_case()
    source = bundle(case)
    object.__setattr__(source, "data_identity_sha256", "9" * 64)

    with pytest.raises(BridgePredictorContractError):
        publish_spatial_perturbation_run(
            source,
            output_dir=tmp_path / "tampered",
            protocol=formal_protocol(case),
            split_manifest=case["eligibility"].manifest,
            neighbor_cells=case["neighbor_cells"],
            expression=case["expression"],
            cell_ids=case["cell_ids"],
            gene_names=case["gene_names"],
            standardizer=case["standardizer"],
            eligibility=case["eligibility"],
            hypersca_budget=case["hypersca_budget"],
            matched_euclidean_budget=case["matched_euclidean_budget"],
        )
    assert not (tmp_path / "tampered").exists()


def test_runner_rejects_code_identity_change_and_never_clobbers(tmp_path: Path) -> None:
    case = run_case()
    with pytest.raises(BridgePredictorContractError, match="code identity"):
        publish_spatial_perturbation_run(
            bundle(case, code_identity_sha256="f" * 64),
            output_dir=tmp_path / "bad-code",
            protocol=formal_protocol(case),
            split_manifest=case["eligibility"].manifest,
            neighbor_cells=case["neighbor_cells"],
            expression=case["expression"],
            cell_ids=case["cell_ids"],
            gene_names=case["gene_names"],
            standardizer=case["standardizer"],
            eligibility=case["eligibility"],
            hypersca_budget=case["hypersca_budget"],
            matched_euclidean_budget=case["matched_euclidean_budget"],
        )
    assert not (tmp_path / "bad-code").exists()

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RunEvidenceError, match="publication_conflict"):
        publish_spatial_perturbation_run(capability(), output_dir=output)
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "changed_name",
    ("spatial_perturbation_registry.py", "methods_protocol_v3_contract.py"),
)
def test_runner_code_identity_rejects_dependency_bytes_changed_after_import(
    monkeypatch: pytest.MonkeyPatch, changed_name: str,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    runner_module.runner_code_identity_sha256()
    original = runner_module._source_bytes
    seen: set[str] = set()

    def changed(
        path: Path, *, remaining_total_bytes: int = 32 * 1024 * 1024
    ) -> bytes:
        seen.add(path.name)
        payload = original(path, remaining_total_bytes=remaining_total_bytes)
        if path.name == changed_name:
            return payload + b"\n# review-byte-change\n"
        return payload

    monkeypatch.setattr(runner_module, "_source_bytes", changed)
    with pytest.raises(BridgePredictorContractError, match="source|changed|code"):
        runner_module.runner_code_identity_sha256()
    assert changed_name in seen


def test_runner_source_reader_rejects_hardlinked_dependency(
    tmp_path: Path,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    source = tmp_path / "dependency.py"
    source.write_bytes(b"trusted = True\n")
    os.link(source, tmp_path / "dependency-alias.py")

    with pytest.raises(BridgePredictorContractError, match="link|regular"):
        runner_module._source_bytes(source)


def test_runner_code_identity_uses_only_canonical_relative_sources() -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    source_root = Path(runner_module.__file__).absolute().parents[1]
    digest = hashlib.sha256(b"hypersca-bridge-runner-source-v2\0")
    for label, relative_path in runner_module._trusted_code_dependencies():
        payload = runner_module._source_bytes(source_root / relative_path)
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(relative_path.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    assert runner_code_identity_sha256() == digest.hexdigest()


def test_runner_routing_declarations_are_immutable() -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    with pytest.raises(TypeError):
        runner_module._CODE_DEPENDENCY_MODULES["task7_scoring"] = "attacker"  # type: ignore[index]
    with pytest.raises(TypeError):
        runner_module._RUNNER_IMPORTED_BINDINGS["predictor_contract"] = ()  # type: ignore[index]


def test_runner_rejects_synchronized_callable_and_route_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module
    import src.evaluation.spatial_perturbation_scoring as scoring_module

    original = scoring_module.score_bridge_predictions

    def rebound(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    replacement_routes = dict(runner_module._CODE_DEPENDENCY_MODULES)
    replacement_routes["task7_scoring"] = "src.evaluation.spatial_perturbation_registry"
    monkeypatch.setattr(
        runner_module,
        "_CODE_DEPENDENCY_MODULES",
        MappingProxyType(replacement_routes),
    )
    monkeypatch.setattr(scoring_module, "score_bridge_predictions", rebound)

    with pytest.raises(BridgePredictorContractError, match="routing|trust|callable"):
        runner_module.runner_code_identity_sha256()


def test_runner_code_identity_executes_on_real_python310() -> None:
    python310 = "python3.10"
    if importlib.util.find_spec("src.evaluation.spatial_perturbation_runner") is None:
        pytest.skip("runner module is unavailable")
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            python310,
            "-c",
            (
                "from src.evaluation.spatial_perturbation_runner import "
                "runner_code_identity_sha256; print(runner_code_identity_sha256())"
            ),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.strip()) == 64


def test_runner_code_identity_rejects_loaded_callable_disk_divergence(
) -> None:
    import src.evaluation.spatial_perturbation_neighbors as neighbor_module
    import src.evaluation.spatial_perturbation_runner as runner_module

    path = Path(neighbor_module.__file__).absolute()
    imported_payload = runner_module._source_bytes(path)
    disk_new_payload = imported_payload.replace(
        b"between 1 and 60", b"between 1 and 61", 1
    )
    assert disk_new_payload != imported_payload
    with pytest.raises(
        BridgePredictorContractError, match="loaded|callable|code|source|changed"
    ):
        runner_module._verified_source_code(
            label="task6_neighbors", path=path, payload=disk_new_payload
        )


def test_runner_code_identity_rejects_actual_imported_callable_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = runner_module.verify_run_evidence_bundle

    def rebound(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "verify_run_evidence_bundle", rebound)

    with pytest.raises(
        BridgePredictorContractError, match="loaded|callable|binding|code"
    ):
        runner_module.runner_code_identity_sha256()


def test_runner_code_identity_rejects_loaded_scientific_callable_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module
    import src.evaluation.spatial_perturbation_scoring as scoring_module

    original = scoring_module.score_bridge_predictions

    def rebound(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    monkeypatch.setattr(scoring_module, "score_bridge_predictions", rebound)

    with pytest.raises(
        BridgePredictorContractError, match="loaded|callable|binding|code"
    ):
        runner_module.runner_code_identity_sha256()


def test_runner_code_identity_rejects_loaded_publisher_method_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    original = RunEvidencePublisher.finalize_failure

    def rebound(
        publisher: RunEvidencePublisher, *args: object, **kwargs: object
    ) -> Path:
        return original(publisher, *args, **kwargs)

    monkeypatch.setattr(RunEvidencePublisher, "finalize_failure", rebound)

    with pytest.raises(
        BridgePredictorContractError, match="loaded|callable|binding|code"
    ):
        runner_module.runner_code_identity_sha256()


def test_runner_source_reader_rejects_ancestor_replaced_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.spatial_perturbation_runner as runner_module

    dependency_dir = tmp_path / "dependency"
    dependency_dir.mkdir()
    moved_dir = tmp_path / "dependency-moved"
    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    source = dependency_dir / "source.py"
    source.write_bytes(b"trusted = True\n")
    (replacement_dir / "source.py").write_bytes(b"trusted = False\n")
    original_read = os.read
    swapped = False

    def swap_ancestor_then_read(file_descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            dependency_dir.rename(moved_dir)
            dependency_dir.symlink_to(replacement_dir, target_is_directory=True)
        return original_read(file_descriptor, size)

    monkeypatch.setattr(os, "read", swap_ancestor_then_read)
    with pytest.raises(BridgePredictorContractError, match="safely|changed"):
        runner_module._source_bytes(source)
    assert swapped


def test_runner_rejects_resealed_comparator_attack_before_output(tmp_path: Path) -> None:
    case = run_case()
    with pytest.raises(BridgePredictorContractError):
        bundle(
            case,
            prediction_bytes=prediction_bytes(
                case["predictions"], own_only_neighbor=0.25
            ),
        )
    assert not (tmp_path / "attack").exists()


def test_cli_help_is_cold_and_cli_publishes_terminal_failure(tmp_path: Path) -> None:
    script = Path("scripts/validate_spatial_perturbation_predictor.py").resolve()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys\n"
                "sys.argv=['validate_spatial_perturbation_predictor.py','--help']\n"
                f"try:\n runpy.run_path({str(script)!r},run_name='__main__')\n"
                "except SystemExit as e:\n"
                " assert e.code == 0\n"
                "forbidden=('numpy','pandas','torch','src.models','src.causal',"
                "'src.evaluation.spatial_perturbation_scoring',"
                "'src.evaluation.spatial_perturbation_comparators')\n"
                "assert not any(name.startswith(forbidden) for name in sys.modules), sorted(sys.modules)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], text=True, capture_output=True
    )
    assert "--output OUTPUT_DIR" in help_result.stdout
    assert "目录" in help_result.stdout

    registry, protocol = declarations()
    registry_path = tmp_path / "spatial_perturbation_bridge_candidates_v1.json"
    protocol_path = tmp_path / "hypersca_methods_v3.yaml"
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            "python3.10",
            str(script),
            "--registry",
            str(registry_path),
            "--protocol",
            str(protocol_path),
            "--method-id",
            "hypersca",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert verify_run_evidence_bundle(output).terminal_status == (
        "method_adapter_not_executable"
    )


@pytest.mark.parametrize(
    ("suffix", "payload"),
    (
        (".json", b'{"schema_version":"a","schema_version":"b"}'),
        (".yaml", b"schema_version: a\nschema_version: b\n"),
    ),
)
def test_cli_declaration_loader_rejects_duplicate_keys(
    tmp_path: Path, suffix: str, payload: bytes
) -> None:
    script = Path("scripts/validate_spatial_perturbation_predictor.py").resolve()
    specification = importlib.util.spec_from_file_location("task9_cli_duplicates", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    declaration = tmp_path / f"declaration{suffix}"
    declaration.write_bytes(payload)

    with pytest.raises(ValueError, match="重复|duplicate"):
        module._load_json(declaration, label="声明")


def test_cli_rejects_abbreviated_long_options(tmp_path: Path) -> None:
    script = Path("scripts/validate_spatial_perturbation_predictor.py").resolve()
    registry, protocol = declarations()
    registry_path = tmp_path / "registry.json"
    protocol_path = tmp_path / "protocol.json"
    registry_path.write_bytes(canonical_json_bytes(registry))
    protocol_path.write_bytes(canonical_json_bytes(protocol))
    output = tmp_path / "abbreviated-output"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--reg",
            str(registry_path),
            "--protocol",
            str(protocol_path),
            "--method-id",
            "hypersca",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert not output.exists()


def test_cli_rejects_declaration_below_symlinked_ancestor(tmp_path: Path) -> None:
    script = Path("scripts/validate_spatial_perturbation_predictor.py").resolve()
    real = tmp_path / "real"
    real.mkdir()
    registry, protocol = declarations()
    (real / "registry.json").write_bytes(canonical_json_bytes(registry))
    (real / "protocol.json").write_bytes(canonical_json_bytes(protocol))
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--registry", str(linked / "registry.json"),
            "--protocol", str(linked / "protocol.json"),
            "--method-id", "hypersca",
            "--output", str(tmp_path / "unsafe-output"),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "unsafe-output").exists()


def test_cli_rejects_regular_file_replacement_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path("scripts/validate_spatial_perturbation_predictor.py").resolve()
    specification = importlib.util.spec_from_file_location("task9_cli", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    declaration = tmp_path / "registry.json"
    replacement = tmp_path / "replacement.json"
    declaration.write_bytes(canonical_json_bytes({"before": True}))
    replacement.write_bytes(canonical_json_bytes({"after": True}))
    original_open = os.open
    swapped = False

    def racing_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and Path(path).name == declaration.name:
            swapped = True
            declaration.unlink()
            replacement.replace(declaration)
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(ValueError, match="变化|安全"):
        module._load_json(declaration, label="注册表")
    assert swapped


def test_cli_rejects_ancestor_replaced_while_declaration_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path("scripts/validate_spatial_perturbation_predictor.py").resolve()
    specification = importlib.util.spec_from_file_location(
        "task9_cli_ancestor_swap", script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    declarations_dir = tmp_path / "declarations"
    declarations_dir.mkdir()
    moved_dir = tmp_path / "declarations-moved"
    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    registry, protocol = declarations()
    registry_path = declarations_dir / "registry.json"
    registry_path.write_bytes(canonical_json_bytes(registry))
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(protocol))
    (replacement_dir / "registry.json").write_bytes(
        canonical_json_bytes(registry)
    )
    output = tmp_path / "ancestor-swap-output"
    original_read = os.read
    swapped = False

    def swap_ancestor_then_read(file_descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            declarations_dir.rename(moved_dir)
            declarations_dir.symlink_to(replacement_dir, target_is_directory=True)
        return original_read(file_descriptor, size)

    monkeypatch.setattr(os, "read", swap_ancestor_then_read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--registry", str(registry_path),
            "--protocol", str(protocol_path),
            "--method-id", "hypersca",
            "--output", str(output),
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert swapped
    assert not output.exists()


def test_cli_rejects_factory_prediction_and_outcome_escape_hatches() -> None:
    help_result = subprocess.run(
        [sys.executable, "scripts/validate_spatial_perturbation_predictor.py", "--help"],
        text=True,
        capture_output=True,
    )
    assert help_result.returncode == 0
    for forbidden in ("factory", "import-path", "prediction-path", "outcome"):
        assert f"--{forbidden}" not in help_result.stdout


def test_task12_plan_uses_predictor_bundle_directory() -> None:
    plan = Path(
        "docs/superpowers/plans/2026-08-28-hypersca-methods-v3-bridge.md"
    ).read_text(encoding="utf-8")
    assert "reports/methods_protocol_v3_preflight/predictor_capability/" in plan
    for name in (
        "capability_record.json",
        "resource_usage.json",
        "run_manifest.json",
        "method_status.json",
    ):
        assert name in plan
