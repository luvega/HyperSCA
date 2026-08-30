from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_evidence_identity import (
    RunEvidenceIdentity,
    canonical_json_bytes,
    canonical_sha256,
)
from src.evaluation.run_evidence_publisher import (
    RunEvidencePublisher,
    VerifiedRunEvidence,
    verify_run_evidence_bundle,
)

from src.discovery.evidence_policy import (
    EvidencePolicyV3,
    V3ClaimDecision,
    V3ClaimEvidence,
    build_evidence_policy_v3,
    derive_integrated_claim,
    evaluate_bridge_claim,
    evaluate_v3_claim,
)
from src.evaluation.methods_protocol_v3 import build_methods_protocol_v3
from src.evaluation.spatial_perturbation_comparators import (
    BridgeModelBudget,
    bridge_predictions_to_comparator_frame,
    predict_bridge_own_only,
    validate_bridge_comparator_budgets,
    validate_bridge_comparator_predictions,
    validate_required_bridge_comparators,
)
from src.evaluation.spatial_perturbation_neighbors import build_bridge_neighbors
from src.evaluation.spatial_perturbation_registry import (
    BridgeCandidate,
    MetadataSummary,
    audit_bridge_capability,
)
from src.evaluation.spatial_perturbation_scoring import (
    BridgePrediction,
    SpatialPerturbationScoringError,
    TrainControlStandardizer,
    apply_train_control_standardizer,
    build_bridge_effect_table,
    fit_train_control_standardizer,
    score_bridge_predictions,
)
from src.evaluation.spatial_perturbation_split import (
    BridgeBlockAdjacency,
    BridgeEligibilityResult,
    BridgeParentEvidence,
    BridgePrimaryUnitEvidence,
    BridgeSplitManifest,
    BridgeSplitMetadata,
    BridgeSplitRow,
    build_bridge_eligibility_evidence,
    build_bridge_partition_manifest,
    evaluate_bridge_eligibility,
    freeze_bridge_neighbour_relation,
    freeze_bridge_neighbour_table,
)


ANIMALS = tuple(f"mouse_{index}" for index in range(1, 6))
PERTURBATIONS = ("KO_A", "KO_B")
GENES = ("GeneA", "GeneB")
TARGETS = tuple(zip(PERTURBATIONS, GENES))
OWN_DELTA = {"KO_A": 1.0, "KO_B": -0.8}
MODEL_SEEDS = (11, 23, 47)


@dataclass(frozen=True, slots=True)
class FiveAnimalFixture:
    raw_cells: pd.DataFrame
    manifest: BridgeSplitManifest
    eligibility: BridgeEligibilityResult
    expression: np.ndarray
    cell_ids: tuple[str, ...]
    standardizer: TrainControlStandardizer
    dgp: str


class _DevelopmentInput(Protocol):
    development_expression: np.ndarray
    development_cell_ids: tuple[str, ...]
    development_animals: tuple[str, ...]
    development_rows: tuple[tuple[str, str, str, str], ...]
    development_relations: tuple[tuple[str, str, str, int, bool], ...]


@dataclass(frozen=True, slots=True)
class DevelopmentAdapterInput:
    development_expression: np.ndarray
    development_cell_ids: tuple[str, ...]
    development_animals: tuple[str, ...]
    development_rows: tuple[tuple[str, str, str, str], ...]
    development_relations: tuple[tuple[str, str, str, int, bool], ...]


@dataclass(frozen=True, slots=True)
class EvaluationPredictionUnits:
    neighbour_units: tuple[tuple[str, str, tuple[int, ...]], ...]
    own_units: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FittedDevelopmentAdapter:
    method_id: str
    adapter_mode: str
    model_seed: int
    own_effects: tuple[tuple[str, float], ...]
    neighbour_parameters: tuple[tuple[str, str, float, float], ...]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    status: str
    evidence_role: str
    scientific_completed: bool
    promotion_authorized: bool
    split_identity_sha256: str
    neighbour_table_identity_sha256: str
    hypersca_rmse: float | None
    matched_euclidean_rmse: float | None
    own_only_rmse: float | None
    model_seed: int
    output_dir: Path
    bundle_identity_sha256: str
    artifact_identity_sha256: str
    run_identity_sha256: str
    prediction_identity_sha256: str


@dataclass(frozen=True, slots=True)
class ScenarioComputation:
    status: str
    hypersca_rmse: float | None
    matched_euclidean_rmse: float | None
    own_only_rmse: float | None
    predictions: tuple[tuple[str, BridgePrediction], ...]


def _sections() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (animal, (f"{animal}_section_1", f"{animal}_section_2")) for animal in ANIMALS
    )


_RAW_COLUMNS = (
    "animal_id",
    "section_id",
    "spatial_block",
    "cell_id",
    "perturbation_id",
    "cell_type",
    "x",
    "y",
    "barcode_positive",
)


@lru_cache(maxsize=1)
def _raw_cell_records() -> tuple[tuple[object, ...], ...]:
    """Cache only immutable atomic records; callers always receive fresh frames."""
    records: list[tuple[object, ...]] = []
    for animal_index, animal in enumerate(ANIMALS):
        for section_index in range(2):
            section = f"{animal}_section_{section_index + 1}"
            for perturbation_index, perturbation in enumerate(
                (*PERTURBATIONS, "mSafe")
            ):
                for block_index in range(5):
                    block = f"block_{block_index}"
                    center_x = float(
                        animal_index * 10_000_000
                        + section_index * 1_000_000
                        + perturbation_index * 100_000
                        + block_index * 10_000
                    )
                    for rank in range(1, 61):
                        records.append(
                            (
                                animal,
                                section,
                                block,
                                f"{animal}_{section}_{perturbation}_a_neighbor_"
                                f"block_{block_index}_rank_{rank:02d}",
                                "unperturbed",
                                "astrocyte",
                                center_x,
                                0.0,
                                False,
                            )
                        )
                for source_in_section in range(10):
                    block_index = source_in_section // 2
                    block = f"block_{block_index}"
                    source_id = (
                        f"{animal}_{section}_{perturbation}_z_source_"
                        f"{source_in_section:02d}"
                    )
                    center_x = float(
                        animal_index * 10_000_000
                        + section_index * 1_000_000
                        + perturbation_index * 100_000
                        + block_index * 10_000
                    )
                    records.append(
                        (
                            animal,
                            section,
                            block,
                            source_id,
                            perturbation,
                            "source_type",
                            center_x,
                            0.0,
                            True,
                        )
                    )
    return tuple(records)


def _raw_cells(row_order_seed: int) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(_raw_cell_records(), columns=_RAW_COLUMNS)
    if row_order_seed:
        frame = frame.sample(frac=1.0, random_state=row_order_seed)
    return frame.reset_index(drop=True)


def _metadata(raw_cells: pd.DataFrame) -> BridgeSplitMetadata:
    neighbors = build_bridge_neighbors(
        raw_cells, max_rank=60, safe_control_label="mSafe"
    )
    relations = tuple(
        freeze_bridge_neighbour_relation(
            animal_id=str(row.animal_id),
            section_id=str(row.section_id),
            spatial_block=str(row.spatial_block),
            source_cell_id=str(row.source_cell_id),
            neighbor_cell_id=str(row.neighbor_cell_id),
            perturbation_id=str(row.perturbation_id),
            source_cell_type=str(row.source_cell_type),
            neighbor_cell_type=str(row.neighbor_cell_type),
            rank=int(cast(Any, row.rank)),
            band=str(row.band),
            is_safe_control=bool(row.is_safe_control),
        )
        for row in neighbors.itertuples(index=False)
    )
    table = freeze_bridge_neighbour_table(relations)
    canonical_cells = raw_cells.sort_values("cell_id", kind="mergesort")
    rows = tuple(
        BridgeSplitRow(
            stable_row_id=index,
            cell_id=str(row.cell_id),
            animal_id=str(row.animal_id),
            section_id=str(row.section_id),
            spatial_block=str(row.spatial_block),
            context_perturbation_id=(
                str(row.perturbation_id) if bool(row.barcode_positive) else "unassigned"
            ),
            observed_label=(
                str(row.perturbation_id)
                if bool(row.barcode_positive)
                else "unperturbed"
            ),
            cell_type=str(row.cell_type),
            source_cell_type=str(row.cell_type),
            cell_role=(
                "safe_source"
                if row.perturbation_id == "mSafe"
                else (
                    "perturbation_source" if bool(row.barcode_positive) else "neighbour"
                )
            ),
            distance_band="own" if bool(row.barcode_positive) else "none",
        )
        for index, row in enumerate(canonical_cells.itertuples(index=False))
    )
    adjacency = tuple(
        BridgeBlockAdjacency(
            animal, section, f"block_{first}", f"block_{second}", False
        )
        for animal, animal_sections in _sections()
        for section in animal_sections
        for first in range(5)
        for second in range(first + 1, 5)
    )
    candidate = BridgeCandidate(
        candidate_id="synthetic_five_animal_bridge",
        accession="SYNTHETIC5",
        platform="spatial_perturbation",
        biological_specimens=ANIMALS,
        sections_by_specimen=_sections(),
        safe_control_label="mSafe",
        perturbation_labels=PERTURBATIONS,
        source_uri="https://example.test/SYNTHETIC5",
        source_identity_sha256="a" * 64,
    )
    per_animal_rows = tuple(
        (animal, sum(row.animal_id == animal for row in rows)) for animal in ANIMALS
    )
    per_animal_perturbations = tuple(
        (
            animal,
            sum(
                row.animal_id == animal and row.cell_role == "perturbation_source"
                for row in rows
            ),
        )
        for animal in ANIMALS
    )
    per_animal_safe = tuple(
        (
            animal,
            sum(
                row.animal_id == animal and row.cell_role == "safe_source"
                for row in rows
            ),
        )
        for animal in ANIMALS
    )
    total_rows = len(rows)
    summary = MetadataSummary(
        candidate_id=candidate.candidate_id,
        accession=candidate.accession,
        cohort_ids=("development", "external"),
        biological_specimen_ids=ANIMALS,
        sections_by_specimen=_sections(),
        block_ids=tuple(f"block_{index}" for index in range(5)),
        coordinate_available=True,
        coordinate_finite=True,
        coordinate_count=total_rows,
        measured_gene_names=GENES,
        measured_gene_count=len(GENES),
        perturbation_labels=PERTURBATIONS,
        perturbation_label_counts=tuple(
            (
                perturbation,
                sum(row.observed_label == perturbation for row in rows),
            )
            for perturbation in PERTURBATIONS
        ),
        safe_control_counts=(("mSafe", sum(count for _, count in per_animal_safe)),),
        barcode_quality_counts=(("valid", total_rows),),
        label_quality_counts=(("valid", total_rows),),
        specimen_cohort_assignments=tuple(
            (animal, "development" if index < 3 else "external")
            for index, animal in enumerate(ANIMALS)
        ),
        external_untouched_cohort_ids=("external",),
        per_specimen_coordinate_counts=per_animal_rows,
        per_specimen_perturbation_counts=per_animal_perturbations,
        per_specimen_safe_control_counts=per_animal_safe,
        per_specimen_barcode_valid_counts=per_animal_rows,
        per_specimen_label_valid_counts=per_animal_rows,
        license_identity="CC-BY-4.0",
        source_identity_sha256=candidate.source_identity_sha256,
        executable_output_schema_capable=True,
    )
    capability = audit_bridge_capability(candidate, summary)
    assert capability.status == "confirmatory_capable"
    return BridgeSplitMetadata(
        rows=rows,
        gene_names=GENES,
        perturbations=PERTURBATIONS,
        neighbour_cell_types=("astrocyte",),
        perturbation_targets=TARGETS,
        block_adjacency=adjacency,
        safe_control_label="mSafe",
        neighbour_relations=table.relations,
        neighbour_table_identity_sha256=table.identity_sha256,
        candidate=candidate,
        registry_summary=summary,
        capability_result=capability,
    )


def _eligibility(
    manifest: BridgeSplitManifest, *, low_coverage: bool = False
) -> BridgeEligibilityResult:
    rows_by_context: dict[tuple[str, str], list[BridgeSplitRow]] = {}
    for row in manifest.row_provenance:
        rows_by_context.setdefault(
            (row.animal_id, row.context_perturbation_id), []
        ).append(row)
    parents = []
    for parent in manifest.perturbation_parents:
        treatment = tuple(
            row.cell_id
            for row in rows_by_context[(parent.animal_id, parent.perturbation_id)]
            if row.cell_role == "perturbation_source"
        )
        safe = tuple(
            row.cell_id
            for row in rows_by_context[(parent.animal_id, "mSafe")]
            if row.cell_role == "safe_source"
        )
        if (
            low_coverage
            and parent.animal_id == "mouse_4"
            and parent.perturbation_id == "KO_A"
        ):
            treatment = treatment[:19]
        parents.append(
            BridgeParentEvidence(
                parent.animal_id,
                parent.perturbation_id,
                parent.target_gene,
                treatment,
                safe,
            )
        )
    units = []
    for unit in manifest.primary_units:
        treatment = tuple(
            relation.relation_id
            for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.source_perturbation_id == unit.perturbation_id
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and not relation.is_safe_control
        )
        safe = tuple(
            relation.relation_id
            for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and relation.is_safe_control
        )
        units.append(BridgePrimaryUnitEvidence(unit.unit_id, treatment, safe))
    evidence = build_bridge_eligibility_evidence(manifest, tuple(parents), tuple(units))
    return evaluate_bridge_eligibility(manifest, evidence)


def _expression(
    raw_cells: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    cells = tuple(sorted(str(value) for value in raw_cells["cell_id"].tolist()))
    expression = np.empty((len(cells), len(GENES)), dtype=np.float64)
    for row_index, cell_id in enumerate(cells):
        for gene_index, gene in enumerate(GENES):
            noise_seed = int.from_bytes(
                hashlib.sha256(f"20260828:{cell_id}:{gene}".encode()).digest()[:8],
                "big",
            )
            noise = float(np.random.default_rng(noise_seed).normal(0.0, 0.01))
            delta = 0.0
            for perturbation, target in TARGETS:
                if gene != target or f"_{perturbation}_" not in cell_id:
                    continue
                if "_a_neighbor_" in cell_id:
                    rank = int(cell_id.rsplit("_", 1)[1])
                    delta = OWN_DELTA[perturbation] * math.exp(-rank / 8.0)
                elif "_z_source_" in cell_id:
                    delta = OWN_DELTA[perturbation]
            control_expression = 5.0
            expression[row_index, gene_index] = control_expression + delta + noise
    return expression, cells


def _fit_standardizer(
    manifest: BridgeSplitManifest,
    expression: np.ndarray,
    cell_ids: tuple[str, ...],
    *,
    leak_holdout: bool = False,
) -> TrainControlStandardizer:
    row_by_cell = {row.cell_id: row for row in manifest.row_provenance}
    train_ids = tuple(
        cell_id
        for cell_id in cell_ids
        if row_by_cell[cell_id].stable_row_id in set(manifest.train_rows)
        and row_by_cell[cell_id].cell_role == "safe_source"
    )
    if leak_holdout:
        leaked = next(
            cell_id
            for cell_id in cell_ids
            if row_by_cell[cell_id].stable_row_id in set(manifest.evaluation_rows)
            and row_by_cell[cell_id].cell_role == "safe_source"
        )
        train_ids = (*train_ids, leaked)
    expression_by_cell = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    training = np.asarray(
        expression[[expression_by_cell[cell_id] for cell_id in train_ids]],
        dtype=np.float64,
    )
    return fit_train_control_standardizer(
        training,
        gene_names=GENES,
        control_rows=tuple(range(len(train_ids))),
        cell_ids=train_ids,
        split_manifest=manifest,
    )


def _build_bridge_structure(
    raw_cells: pd.DataFrame,
) -> tuple[BridgeSplitManifest, BridgeEligibilityResult]:
    metadata = _metadata(raw_cells)
    manifest = build_bridge_partition_manifest(
        metadata,
        split_id="confirmatory_partition:five_animal_synthetic",
        split_role="confirmatory",
        train_animals=("mouse_1", "mouse_2"),
        tune_animals=("mouse_3",),
        evaluation_animals=("mouse_4", "mouse_5"),
    )
    eligibility = _eligibility(manifest)
    assert eligibility.eligible is True
    return manifest, eligibility


@lru_cache(maxsize=1)
def _immutable_bridge_structure() -> (
    tuple[BridgeSplitManifest, BridgeEligibilityResult]
):
    return _build_bridge_structure(_raw_cells(0))


@lru_cache(maxsize=1)
def _immutable_fixture_components() -> tuple[
    BridgeSplitManifest,
    BridgeEligibilityResult,
    bytes,
    tuple[int, int],
    tuple[str, ...],
    TrainControlStandardizer,
]:
    raw_cells = _raw_cells(0)
    manifest, eligibility = _immutable_bridge_structure()
    expression, cell_ids = _expression(raw_cells)
    standardizer = _fit_standardizer(manifest, expression, cell_ids)
    return (
        manifest,
        eligibility,
        expression.tobytes(),
        cast(tuple[int, int], expression.shape),
        cell_ids,
        standardizer,
    )


def build_five_animal_fixture(row_order_seed: int = 0) -> FiveAnimalFixture:
    raw_cells = _raw_cells(row_order_seed)
    (
        manifest,
        eligibility,
        expression_bytes,
        expression_shape,
        cell_ids,
        standardizer,
    ) = _immutable_fixture_components()
    if row_order_seed:
        manifest, eligibility = _build_bridge_structure(raw_cells)
    expression = np.frombuffer(expression_bytes, dtype=np.float64).reshape(
        expression_shape
    )
    return FiveAnimalFixture(
        raw_cells,
        manifest,
        eligibility,
        expression,
        cell_ids,
        standardizer,
        "exponential",
    )


def _zero_predictions(
    eligibility: BridgeEligibilityResult,
) -> tuple[BridgePrediction, ...]:
    evaluation = set(eligibility.manifest.evaluation_animals)
    scoreable_parents = set(eligibility.scoreable_parent_ids)
    return tuple(
        [
            BridgePrediction(unit.unit_id, "neighbor", 0.0)
            for unit in eligibility.manifest.primary_units
            if unit.animal_id in evaluation
            and unit.unit_id not in set(eligibility.abstained_unit_ids)
        ]
        + [
            BridgePrediction(parent.parent_id, "own", 0.0)
            for parent in eligibility.manifest.perturbation_parents
            if parent.animal_id in evaluation and parent.parent_id in scoreable_parents
        ]
    )


def _development_adapter_input(fixture: FiveAnimalFixture) -> DevelopmentAdapterInput:
    row_by_cell = {row.cell_id: row for row in fixture.manifest.row_provenance}
    development = set(fixture.manifest.development_animals)
    indices = tuple(
        index
        for index, cell_id in enumerate(fixture.cell_ids)
        if row_by_cell[cell_id].animal_id in development
    )
    standardized = apply_train_control_standardizer(
        np.array(fixture.expression[list(indices)], copy=True),
        gene_names=GENES,
        standardizer=fixture.standardizer,
        split_manifest=fixture.manifest,
    )
    return DevelopmentAdapterInput(
        development_expression=standardized,
        development_cell_ids=tuple(fixture.cell_ids[index] for index in indices),
        development_animals=fixture.manifest.development_animals,
        development_rows=tuple(
            (row.cell_id, row.animal_id, row.observed_label, row.cell_role)
            for row in fixture.manifest.row_provenance
            if row.animal_id in development
        ),
        development_relations=tuple(
            (
                relation.animal_id,
                relation.neighbor_cell_id,
                relation.source_perturbation_id,
                relation.rank,
                relation.is_safe_control,
            )
            for relation in fixture.manifest.neighbour_relations
            if relation.animal_id in development
        ),
    )


def _development_effects(
    development: _DevelopmentInput,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str, int], float],
]:
    standardized = development.development_expression
    index_by_cell = {
        cell_id: index for index, cell_id in enumerate(development.development_cell_ids)
    }
    rows = {
        cell_id: (animal_id, observed_label, cell_role)
        for cell_id, animal_id, observed_label, cell_role in development.development_rows
    }
    own: dict[tuple[str, str], float] = {}
    neighbours: dict[tuple[str, str, int], float] = {}
    for animal in development.development_animals:
        for perturbation, target in TARGETS:
            gene_index = GENES.index(target)
            treated_sources = [
                index_by_cell[cell_id]
                for cell_id in development.development_cell_ids
                if rows[cell_id][0] == animal
                and rows[cell_id][1] == perturbation
                and rows[cell_id][2] == "perturbation_source"
            ]
            safe_sources = [
                index_by_cell[cell_id]
                for cell_id in development.development_cell_ids
                if rows[cell_id][0] == animal
                and rows[cell_id][1] == "mSafe"
                and rows[cell_id][2] == "safe_source"
            ]
            own[(animal, perturbation)] = float(
                standardized[treated_sources, gene_index].mean()
                - standardized[safe_sources, gene_index].mean()
            )
            for rank in range(1, 61):
                treated_neighbours = [
                    index_by_cell[neighbor_cell_id]
                    for (
                        relation_animal,
                        neighbor_cell_id,
                        source_perturbation,
                        relation_rank,
                        is_safe,
                    ) in development.development_relations
                    if relation_animal == animal
                    and source_perturbation == perturbation
                    and relation_rank == rank
                    and not is_safe
                ]
                safe_neighbours = [
                    index_by_cell[neighbor_cell_id]
                    for (
                        relation_animal,
                        neighbor_cell_id,
                        _source_perturbation,
                        relation_rank,
                        is_safe,
                    ) in development.development_relations
                    if relation_animal == animal and relation_rank == rank and is_safe
                ]
                neighbours[(animal, perturbation, rank)] = float(
                    standardized[treated_neighbours, gene_index].mean()
                    - standardized[safe_neighbours, gene_index].mean()
                )
    return own, neighbours


def _fit_development_only_adapter(
    development: _DevelopmentInput,
    *,
    method_id: str,
    model_seed: int,
    adapter_mode: str = "fitted",
) -> FittedDevelopmentAdapter:
    if method_id not in {"hypersca", "matched_euclidean_spatial_causal"}:
        raise ValueError("unknown test-only adapter")
    allowed_modes = (
        {"fitted", "misfit_constant", "no_neighbor"}
        if method_id == "hypersca"
        else {"fitted"}
    )
    if adapter_mode not in allowed_modes:
        raise ValueError("adapter mode is not available for this method")
    own_by_animal, neighbour_by_animal = _development_effects(development)
    animals = development.development_animals
    rng = np.random.default_rng(model_seed)
    bootstrap_weights = rng.dirichlet(np.ones(len(animals), dtype=np.float64))
    own_effects: list[tuple[str, float]] = []
    parameters: list[tuple[str, str, float, float]] = []
    for perturbation in PERTURBATIONS:
        own_estimate = float(
            np.dot(
                bootstrap_weights,
                [own_by_animal[(animal, perturbation)] for animal in animals],
            )
        )
        own_effects.append((perturbation, own_estimate))
        rank_values = np.arange(1, 61, dtype=np.float64)
        observed = np.asarray(
            [
                np.dot(
                    bootstrap_weights,
                    [
                        neighbour_by_animal[(animal, perturbation, rank)]
                        for animal in animals
                    ],
                )
                for rank in range(1, 61)
            ],
            dtype=np.float64,
        )
        if adapter_mode == "no_neighbor":
            intercept, slope = 0.0, 0.0
        elif adapter_mode == "misfit_constant":
            intercept, slope = float(observed[-1]), 0.0
        elif method_id == "hypersca":
            fit_slice = slice(0, 30)
            signed = np.sign(own_estimate) * observed[fit_slice]
            usable = signed > 1e-6
            slope, intercept = np.polyfit(
                rank_values[fit_slice][usable],
                np.log(signed[usable]),
                1,
            )
        else:
            slope, intercept = np.polyfit(rank_values, observed, 1)
        parameters.append((perturbation, method_id, float(intercept), float(slope)))
    return FittedDevelopmentAdapter(
        method_id,
        adapter_mode,
        model_seed,
        tuple(own_effects),
        tuple(parameters),
    )


def _evaluation_prediction_units(
    manifest: BridgeSplitManifest,
    eligibility: BridgeEligibilityResult,
) -> EvaluationPredictionUnits:
    evaluation = set(manifest.evaluation_animals)
    abstained = set(eligibility.abstained_unit_ids)
    scoreable_parents = set(eligibility.scoreable_parent_ids)
    neighbour_units: list[tuple[str, str, tuple[int, ...]]] = []
    for unit in manifest.primary_units:
        if unit.animal_id not in evaluation or unit.unit_id in abstained:
            continue
        ranks = tuple(
            relation.rank
            for relation in manifest.neighbour_relations
            if relation.animal_id == unit.animal_id
            and relation.source_perturbation_id == unit.perturbation_id
            and relation.neighbor_cell_type == unit.neighbour_cell_type
            and relation.band == unit.band
            and not relation.is_safe_control
        )
        neighbour_units.append((unit.unit_id, unit.perturbation_id, ranks))
    own_units = tuple(
        (parent.parent_id, parent.perturbation_id)
        for parent in manifest.perturbation_parents
        if parent.animal_id in evaluation and parent.parent_id in scoreable_parents
    )
    return EvaluationPredictionUnits(tuple(neighbour_units), own_units)


def _predict_evaluation_units(
    adapter: FittedDevelopmentAdapter,
    *,
    evaluation_units: EvaluationPredictionUnits,
) -> tuple[BridgePrediction, ...]:
    own = dict(adapter.own_effects)
    parameters = {
        perturbation: (intercept, slope)
        for perturbation, _method, intercept, slope in adapter.neighbour_parameters
    }
    predictions: list[BridgePrediction] = []
    for unit_id, perturbation, ranks in evaluation_units.neighbour_units:
        intercept, slope = parameters[perturbation]
        if adapter.adapter_mode == "no_neighbor":
            predicted = 0.0
        elif adapter.adapter_mode == "misfit_constant":
            predicted = intercept
        elif adapter.method_id == "hypersca":
            predicted = float(
                np.sign(own[perturbation])
                * np.mean(np.exp(intercept + slope * np.asarray(ranks)))
            )
        else:
            predicted = float(
                np.mean(intercept + slope * np.asarray(ranks, dtype=np.float64))
            )
        predictions.append(BridgePrediction(unit_id, "neighbor", predicted))
    for parent_id, perturbation in evaluation_units.own_units:
        predictions.append(BridgePrediction(parent_id, "own", own[perturbation]))
    return tuple(predictions)


def _predictions_from_frame(frame: pd.DataFrame) -> tuple[BridgePrediction, ...]:
    return tuple(
        BridgePrediction(
            str(row.unit_id),
            str(row.endpoint),
            float(cast(Any, row.predicted_effect)),
        )
        for row in frame.itertuples(index=False)
    )


def _validate_matched_budgets(fixture: FiveAnimalFixture, model_seed: int) -> None:
    data_identity = hashlib.sha256(
        fixture.expression.tobytes() + "\0".join(fixture.cell_ids).encode()
    ).hexdigest()
    gene_identity = hashlib.sha256("\0".join(GENES).encode()).hexdigest()
    validate_bridge_comparator_budgets(
        BridgeModelBudget(
            "hypersca",
            "hyperbolic",
            1_000,
            "adamw",
            2_000,
            100,
            10,
            data_identity,
            gene_identity,
            fixture.manifest.neighbour_table_identity_sha256,
            "d" * 64,
            model_seed,
        ),
        BridgeModelBudget(
            "matched_euclidean_spatial_causal",
            "euclidean",
            1_000,
            "adamw",
            2_000,
            100,
            10,
            data_identity,
            gene_identity,
            fixture.manifest.neighbour_table_identity_sha256,
            "d" * 64,
            model_seed,
        ),
    )
    validate_required_bridge_comparators(
        ("matched_euclidean_spatial_causal", "hypersca_own_only")
    )


def _prediction_record(
    scenario: str,
    model_seed: int,
    predictions: tuple[tuple[str, BridgePrediction], ...],
) -> dict[str, object]:
    return {
        "schema": "test_only_bridge_predictions_v1",
        "scenario": scenario,
        "model_seed": model_seed,
        "methods": [
            "hypersca",
            "hypersca_own_only",
            "matched_euclidean_spatial_causal",
        ],
        "predictions": sorted(
            [
                {
                    "method_id": method_id,
                    "unit_id": prediction.unit_id,
                    "endpoint": prediction.endpoint,
                    "predicted_delta": prediction.predicted_delta,
                }
                for method_id, prediction in predictions
            ],
            key=lambda row: (
                cast(str, row["method_id"]),
                cast(str, row["endpoint"]),
                cast(str, row["unit_id"]),
            ),
        ),
    }


def _statistical_unit_record(fixture: FiveAnimalFixture) -> dict[str, object]:
    evaluation_animals = set(fixture.manifest.evaluation_animals)
    return {
        "schema": "five_animal_bridge_evaluation_units_v1",
        "units": sorted(
            [
                unit.unit_id
                for unit in fixture.manifest.primary_units
                if unit.animal_id in evaluation_animals
            ]
            + [
                parent.parent_id
                for parent in fixture.manifest.perturbation_parents
                if parent.animal_id in evaluation_animals
            ]
        ),
    }


def _run_identity(
    fixture: FiveAnimalFixture, scenario: str, model_seed: int
) -> RunEvidenceIdentity:
    policy = _policy()
    statistical_units = _statistical_unit_record(fixture)
    return RunEvidenceIdentity(
        schema_version="1.0",
        protocol_version=policy.protocol_version,
        protocol_identity=policy.protocol_identity_sha256,
        claim_id="bridge",
        benchmark_id="synthetic_five_animal_spatial_perturbation_bridge",
        data_scopes=("synthetic",),
        data_split_seed=20260828,
        model_seed=model_seed,
        data_split_identity_sha256=fixture.manifest.split_identity_sha256,
        statistical_unit_schema="five_animal_bridge_evaluation_units_v1",
        statistical_unit_identity_sha256=canonical_sha256(statistical_units),
        analysis_identity_sha256=canonical_sha256(
            {
                "adapters": ["development_exponential", "development_linear"],
                "matched_budget": True,
                "scoring": "production_bridge_scoring",
            }
        ),
        input_identity_sha256=hashlib.sha256(
            fixture.expression.tobytes() + "\0".join(fixture.cell_ids).encode()
        ).hexdigest(),
        config_identity_sha256=canonical_sha256(
            {"scenario": scenario, "dgp": fixture.dgp, "model_seed": model_seed}
        ),
        code_identity_sha256=_adapter_code_identity(),
        evidence_role="synthetic_audit_only",
    )


def _publish_scenario_result(
    *,
    output_root: Path,
    fixture: FiveAnimalFixture,
    scenario: str,
    model_seed: int,
    status: str,
    hypersca_rmse: float | None,
    matched_euclidean_rmse: float | None,
    own_only_rmse: float | None,
    predictions: tuple[tuple[str, BridgePrediction], ...],
) -> ScenarioResult:
    prediction_record = _prediction_record(scenario, model_seed, predictions)
    prediction_identity = canonical_sha256(prediction_record)
    result_record: dict[str, object] = {
        "schema": "test_only_five_animal_scenario_result_v1",
        "scenario": scenario,
        "status": status,
        "evidence_role": "synthetic_audit_only",
        "scientific_completed": False,
        "promotion_authorized": False,
        "split_identity_sha256": fixture.manifest.split_identity_sha256,
        "neighbour_table_identity_sha256": (
            fixture.manifest.neighbour_table_identity_sha256
        ),
        "hypersca_rmse": hypersca_rmse,
        "matched_euclidean_rmse": matched_euclidean_rmse,
        "own_only_rmse": own_only_rmse,
        "model_seed": model_seed,
        "prediction_identity_sha256": prediction_identity,
    }
    statistical_units = _statistical_unit_record(fixture)
    identity = _run_identity(fixture, scenario, model_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_root / "bundle"
    result_bytes = canonical_json_bytes(result_record)
    prediction_bytes = canonical_json_bytes(prediction_record)
    with RunEvidencePublisher.begin(
        output_dir=bundle_dir,
        identity=identity,
        statistical_unit_record=statistical_units,
        required_artifacts=("predictions.json", "scenario_result.json"),
        maximum_bundle_bytes=64 * 1024,
    ) as publisher:
        publisher.add_bytes(
            "predictions.json", prediction_bytes, media_type="application/json"
        )
        publisher.add_bytes(
            "scenario_result.json", result_bytes, media_type="application/json"
        )
        publisher.finalize_completed(summary=result_record)
    verified = verify_run_evidence_bundle(bundle_dir)
    return _replay_published_scenario(verified)


def _adapter_code_identity() -> str:
    return canonical_sha256(
        {
            name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for name, function in (
                ("development_input", _development_adapter_input),
                ("development_effects", _development_effects),
                ("evaluation_units", _evaluation_prediction_units),
                ("fit", _fit_development_only_adapter),
                ("predict", _predict_evaluation_units),
                ("expected_predictions", _expected_method_predictions),
                ("scenario_computation", _immutable_scenario_computation),
            )
        }
    )


def _read_verified_json_artifact(
    verified: VerifiedRunEvidence, relative_path: str
) -> dict[str, object]:
    artifact = next(
        item for item in verified.artifacts if item.relative_path == relative_path
    )
    assert artifact.media_type == "application/json"
    payload = (verified.output_dir / relative_path).read_bytes()
    assert len(payload) == artifact.size_bytes
    assert hashlib.sha256(payload).hexdigest() == artifact.sha256
    parsed = json.loads(payload)
    assert type(parsed) is dict
    assert canonical_json_bytes(parsed) == payload
    return cast(dict[str, object], parsed)


def _validate_published_prediction_record(record: dict[str, object]) -> None:
    assert set(record) == {
        "schema",
        "scenario",
        "model_seed",
        "methods",
        "predictions",
    }
    assert record["schema"] == "test_only_bridge_predictions_v1"
    _validate_scenario(cast(str, record["scenario"]))
    assert record["model_seed"] in MODEL_SEEDS
    methods = [
        "hypersca",
        "hypersca_own_only",
        "matched_euclidean_spatial_causal",
    ]
    assert record["methods"] == methods
    rows = record["predictions"]
    assert type(rows) is list
    keys = []
    for item in cast(list[object], rows):
        assert type(item) is dict
        row = cast(dict[str, object], item)
        assert set(row) == {
            "method_id",
            "unit_id",
            "endpoint",
            "predicted_delta",
        }
        assert row["method_id"] in methods
        assert row["endpoint"] in {"neighbor", "own"}
        assert type(row["unit_id"]) is str
        assert type(row["predicted_delta"]) is float
        assert math.isfinite(cast(float, row["predicted_delta"]))
        keys.append((row["method_id"], row["endpoint"], row["unit_id"]))
    assert len(keys) == len(set(keys))


def _predictions_from_published_record(
    record: dict[str, object], method_id: str
) -> tuple[BridgePrediction, ...]:
    rows = cast(list[object], record["predictions"])
    return tuple(
        BridgePrediction(
            cast(str, row["unit_id"]),
            cast(str, row["endpoint"]),
            cast(float, row["predicted_delta"]),
        )
        for item in rows
        for row in [cast(dict[str, object], item)]
        if row["method_id"] == method_id
    )


def _recompute_published_scenario(
    fixture: FiveAnimalFixture,
    scenario: str,
    prediction_record: dict[str, object],
) -> tuple[str, float | None, float | None, float | None]:
    if scenario == "coverage_low":
        assert prediction_record["predictions"] == []
        eligibility = _eligibility(fixture.manifest, low_coverage=True)
        return (
            eligibility.reason or "insufficient_perturbation_coverage",
            None,
            None,
            None,
        )
    if scenario == "holdout_leak":
        assert prediction_record["predictions"] == []
        with pytest.raises(SpatialPerturbationScoringError):
            _fit_standardizer(
                fixture.manifest,
                fixture.expression,
                fixture.cell_ids,
                leak_holdout=True,
            )
        return "failed_invalid_input", None, None, None
    scores = {}
    for method_id in (
        "hypersca",
        "hypersca_own_only",
        "matched_euclidean_spatial_causal",
    ):
        scores[method_id] = score_bridge_predictions(
            fixture.expression,
            cell_ids=fixture.cell_ids,
            gene_names=GENES,
            standardizer=fixture.standardizer,
            eligibility=fixture.eligibility,
            predictions=_predictions_from_published_record(
                prediction_record, method_id
            ),
        )
    hypersca_rmse = scores["hypersca"].neighbor_effect_rmse
    euclidean_rmse = scores["matched_euclidean_spatial_causal"].neighbor_effect_rmse
    own_only_rmse = scores["hypersca_own_only"].neighbor_effect_rmse
    passed = (
        euclidean_rmse - hypersca_rmse > 0.0 and own_only_rmse - hypersca_rmse > 0.0
    )
    return (
        "audit_metric_gate_passed" if passed else "audit_metric_gate_failed",
        hypersca_rmse,
        euclidean_rmse,
        own_only_rmse,
    )


def _replay_published_scenario(verified: VerifiedRunEvidence) -> ScenarioResult:
    assert verified.terminal_status == "completed"
    assert {artifact.relative_path for artifact in verified.artifacts} == {
        "predictions.json",
        "scenario_result.json",
    }
    prediction_record = _read_verified_json_artifact(verified, "predictions.json")
    replayed = _read_verified_json_artifact(verified, "scenario_result.json")
    _validate_published_prediction_record(prediction_record)
    assert set(replayed) == {
        "schema",
        "scenario",
        "status",
        "evidence_role",
        "scientific_completed",
        "promotion_authorized",
        "split_identity_sha256",
        "neighbour_table_identity_sha256",
        "hypersca_rmse",
        "matched_euclidean_rmse",
        "own_only_rmse",
        "model_seed",
        "prediction_identity_sha256",
    }
    assert replayed["schema"] == "test_only_five_animal_scenario_result_v1"
    assert dict(verified.summary or {}) == replayed
    assert prediction_record["scenario"] == replayed["scenario"]
    assert prediction_record["model_seed"] == replayed["model_seed"]
    assert prediction_record["methods"] == [
        "hypersca",
        "hypersca_own_only",
        "matched_euclidean_spatial_causal",
    ]
    prediction_identity = canonical_sha256(prediction_record)
    assert prediction_identity == replayed["prediction_identity_sha256"]
    scenario = cast(str, replayed["scenario"])
    model_seed = cast(int, replayed["model_seed"])
    fixture = build_five_animal_fixture()
    expected_predictions = _expected_method_predictions(
        fixture, scenario, model_seed
    )
    assert prediction_record == _prediction_record(
        scenario, model_seed, expected_predictions
    )
    assert canonical_json_bytes(dict(verified.statistical_unit_record)) == (
        canonical_json_bytes(_statistical_unit_record(fixture))
    )
    assert verified.identity == _run_identity(fixture, scenario, model_seed)
    status, hypersca_rmse, euclidean_rmse, own_only_rmse = (
        _recompute_published_scenario(
            fixture, scenario, prediction_record
        )
    )
    assert replayed["status"] == status
    assert replayed["hypersca_rmse"] == hypersca_rmse
    assert replayed["matched_euclidean_rmse"] == euclidean_rmse
    assert replayed["own_only_rmse"] == own_only_rmse
    assert replayed["split_identity_sha256"] == fixture.manifest.split_identity_sha256
    assert replayed["neighbour_table_identity_sha256"] == (
        fixture.manifest.neighbour_table_identity_sha256
    )
    assert replayed["evidence_role"] == "synthetic_audit_only"
    assert replayed["scientific_completed"] is False
    assert replayed["promotion_authorized"] is False
    return ScenarioResult(
        status=str(replayed["status"]),
        evidence_role=str(replayed["evidence_role"]),
        scientific_completed=bool(replayed["scientific_completed"]),
        promotion_authorized=bool(replayed["promotion_authorized"]),
        split_identity_sha256=str(replayed["split_identity_sha256"]),
        neighbour_table_identity_sha256=str(
            replayed["neighbour_table_identity_sha256"]
        ),
        hypersca_rmse=cast(float | None, replayed["hypersca_rmse"]),
        matched_euclidean_rmse=cast(float | None, replayed["matched_euclidean_rmse"]),
        own_only_rmse=cast(float | None, replayed["own_only_rmse"]),
        model_seed=int(replayed["model_seed"]),
        output_dir=verified.output_dir,
        bundle_identity_sha256=verified.bundle_identity_sha256,
        artifact_identity_sha256=next(
            artifact.sha256
            for artifact in verified.artifacts
            if artifact.relative_path == "scenario_result.json"
        ),
        run_identity_sha256=verified.identity.run_identity_sha256,
        prediction_identity_sha256=prediction_identity,
    )


def _validate_scenario(scenario: str) -> None:
    if scenario not in {
        "valid",
        "euclidean_wins",
        "own_only_ties",
        "coverage_low",
        "holdout_leak",
    }:
        raise ValueError("unknown five-animal bridge scenario")


def _expected_method_predictions(
    fixture: FiveAnimalFixture, scenario: str, model_seed: int
) -> tuple[tuple[str, BridgePrediction], ...]:
    if scenario in {"coverage_low", "holdout_leak"}:
        return ()
    _validate_matched_budgets(fixture, model_seed)
    development = _development_adapter_input(fixture)
    evaluation_units = _evaluation_prediction_units(
        fixture.manifest, fixture.eligibility
    )
    hypersca_predictions = _predict_evaluation_units(
        _fit_development_only_adapter(
            development,
            method_id="hypersca",
            model_seed=model_seed,
            adapter_mode={
                "valid": "fitted",
                "euclidean_wins": "misfit_constant",
                "own_only_ties": "no_neighbor",
            }[scenario],
        ),
        evaluation_units=evaluation_units,
    )
    euclidean_predictions = _predict_evaluation_units(
        _fit_development_only_adapter(
            development,
            method_id="matched_euclidean_spatial_causal",
            model_seed=model_seed,
        ),
        evaluation_units=evaluation_units,
    )
    hypersca_frame = bridge_predictions_to_comparator_frame(hypersca_predictions)
    euclidean_frame = bridge_predictions_to_comparator_frame(euclidean_predictions)
    own_only_frame = predict_bridge_own_only(hypersca_frame)
    validate_bridge_comparator_predictions(
        hypersca_frame, euclidean_frame, own_only_frame
    )
    own_only_predictions = _predictions_from_frame(own_only_frame)
    return tuple(
        [("hypersca", prediction) for prediction in hypersca_predictions]
        + [
            ("matched_euclidean_spatial_causal", prediction)
            for prediction in euclidean_predictions
        ]
        + [("hypersca_own_only", prediction) for prediction in own_only_predictions]
    )


@lru_cache(maxsize=15)
def _immutable_scenario_computation(
    scenario: str, model_seed: int
) -> ScenarioComputation:
    _validate_scenario(scenario)
    fixture = build_five_animal_fixture()
    if scenario == "coverage_low":
        eligibility = _eligibility(fixture.manifest, low_coverage=True)
        return ScenarioComputation(
            eligibility.reason or "insufficient_perturbation_coverage",
            None,
            None,
            None,
            (),
        )
    if scenario == "holdout_leak":
        try:
            _fit_standardizer(
                fixture.manifest,
                fixture.expression,
                fixture.cell_ids,
                leak_holdout=True,
            )
        except SpatialPerturbationScoringError:
            return ScenarioComputation("failed_invalid_input", None, None, None, ())
        raise AssertionError("held-out control leakage was not rejected")

    predictions = _expected_method_predictions(fixture, scenario, model_seed)
    hypersca_predictions = tuple(
        prediction for method, prediction in predictions if method == "hypersca"
    )
    euclidean_predictions = tuple(
        prediction
        for method, prediction in predictions
        if method == "matched_euclidean_spatial_causal"
    )
    own_only_predictions = tuple(
        prediction
        for method, prediction in predictions
        if method == "hypersca_own_only"
    )
    hypersca_score = score_bridge_predictions(
        fixture.expression,
        cell_ids=fixture.cell_ids,
        gene_names=GENES,
        standardizer=fixture.standardizer,
        eligibility=fixture.eligibility,
        predictions=hypersca_predictions,
    )
    euclidean_score = score_bridge_predictions(
        fixture.expression,
        cell_ids=fixture.cell_ids,
        gene_names=GENES,
        standardizer=fixture.standardizer,
        eligibility=fixture.eligibility,
        predictions=euclidean_predictions,
    )
    own_only_score = score_bridge_predictions(
        fixture.expression,
        cell_ids=fixture.cell_ids,
        gene_names=GENES,
        standardizer=fixture.standardizer,
        eligibility=fixture.eligibility,
        predictions=own_only_predictions,
    )
    passed = (
        euclidean_score.neighbor_effect_rmse - hypersca_score.neighbor_effect_rmse > 0.0
        and own_only_score.neighbor_effect_rmse - hypersca_score.neighbor_effect_rmse
        > 0.0
    )
    return ScenarioComputation(
        "audit_metric_gate_passed" if passed else "audit_metric_gate_failed",
        hypersca_score.neighbor_effect_rmse,
        euclidean_score.neighbor_effect_rmse,
        own_only_score.neighbor_effect_rmse,
        predictions,
    )


def run_fixture_scenario(
    scenario: str,
    output_dir: Path,
    *,
    row_order_seed: int = 0,
    model_seed: int = 11,
) -> ScenarioResult:
    _validate_scenario(scenario)
    if model_seed not in MODEL_SEEDS:
        raise ValueError("model seed is not preregistered")
    fixture = build_five_animal_fixture(row_order_seed)
    computation = _immutable_scenario_computation(scenario, model_seed)
    return _publish_scenario_result(
        output_root=output_dir,
        fixture=fixture,
        scenario=scenario,
        model_seed=model_seed,
        status=computation.status,
        hypersca_rmse=computation.hypersca_rmse,
        matched_euclidean_rmse=computation.matched_euclidean_rmse,
        own_only_rmse=computation.own_only_rmse,
        predictions=computation.predictions,
    )


def _policy() -> EvidencePolicyV3:
    return build_evidence_policy_v3(
        build_methods_protocol_v3(
            bridge_role="confirmatory", capability_identity_sha256="9" * 64
        )
    )


def _claim_evidence(
    claim_id: str,
    comparator_id: str,
    policy: EvidencePolicyV3,
    *,
    role: str,
    identity_character: str,
) -> V3ClaimEvidence:
    return V3ClaimEvidence(
        claim_id=claim_id,
        protocol_version=policy.protocol_version,
        protocol_identity_sha256=policy.protocol_identity_sha256,
        capability_identity_sha256=policy.capability_identity_sha256,
        primary_metric=dict(policy.family_primary_metrics)[claim_id],
        comparator_id=comparator_id,
        paired_estimate=0.1,
        ci_low=0.01,
        ci_high=0.2,
        nominal_p_value=0.01,
        attempted_units=3,
        completed_units=3,
        paired_unit_identity_sha256="e" * 64,
        run_statuses=("completed",) * 3,
        evidence_role=role,
        artifact_identity=identity_character * 64,
    )


def _admitted_module(claim_id: str, policy: EvidencePolicyV3) -> V3ClaimDecision:
    comparators = dict(policy.required_comparators)[claim_id]
    roles = ("confirmatory", "attribution")
    return evaluate_v3_claim(
        tuple(
            _claim_evidence(
                claim_id,
                comparator,
                policy,
                role=role,
                identity_character=identity,
            )
            for comparator, role, identity in zip(comparators, roles, "ab")
        ),
        policy,
    )


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("valid", "audit_metric_gate_passed"),
        ("euclidean_wins", "audit_metric_gate_failed"),
        ("own_only_ties", "audit_metric_gate_failed"),
        ("coverage_low", "insufficient_perturbation_coverage"),
        ("holdout_leak", "failed_invalid_input"),
    ],
)
def test_bridge_scenarios(scenario: str, expected: str, tmp_path: Path) -> None:
    result = run_fixture_scenario(scenario, tmp_path)
    assert result.status == expected
    assert result.evidence_role == "synthetic_audit_only"
    assert result.scientific_completed is False
    assert result.promotion_authorized is False
    if scenario == "valid":
        assert result.hypersca_rmse is not None
        assert result.matched_euclidean_rmse is not None
        assert result.own_only_rmse is not None
        assert result.hypersca_rmse < result.matched_euclidean_rmse
        assert result.hypersca_rmse < result.own_only_rmse
    elif scenario == "euclidean_wins":
        assert result.hypersca_rmse is not None
        assert result.matched_euclidean_rmse is not None
        assert result.matched_euclidean_rmse < result.hypersca_rmse
    elif scenario == "own_only_ties":
        assert result.hypersca_rmse == result.own_only_rmse


def test_fixture_uses_five_animals_two_sections_and_frozen_signal() -> None:
    fixture = build_five_animal_fixture()
    assert len(fixture.raw_cells) == 9_300
    assert len(fixture.manifest.neighbour_relations) == 9_000
    assert fixture.manifest.development_animals == ("mouse_1", "mouse_2", "mouse_3")
    assert fixture.manifest.evaluation_animals == ("mouse_4", "mouse_5")
    assert fixture.manifest.candidate.sections_by_specimen == _sections()
    assert set(fixture.manifest.registered_perturbations) == set(PERTURBATIONS)
    assert len({row.spatial_block for row in fixture.manifest.row_provenance}) == 5
    relation_by_id = {
        relation.relation_id: relation
        for relation in fixture.manifest.neighbour_relations
    }
    primary_unit_by_id = {
        unit.unit_id: unit for unit in fixture.manifest.primary_units
    }
    assert all(
        len(parent.perturbation_source_cell_ids) == 20
        and len(parent.safe_source_cell_ids) == 20
        for parent in fixture.eligibility.evidence.parent_evidence
    )
    assert all(
        len(unit.perturbation_neighbour_relation_ids)
        == {"proximal": 50, "local": 100}[
            primary_unit_by_id[unit.unit_id].band
        ]
        and len(unit.safe_neighbour_relation_ids)
        == {"proximal": 50, "local": 100}[
            primary_unit_by_id[unit.unit_id].band
        ]
        and len(
            {
                relation_by_id[relation_id].spatial_block
                for relation_id in unit.perturbation_neighbour_relation_ids
            }
        )
        >= 3
        for unit in fixture.eligibility.evidence.unit_evidence
    )
    for perturbation, expected in OWN_DELTA.items():
        source = next(
            row
            for row in fixture.raw_cells.itertuples(index=False)
            if row.perturbation_id == perturbation and row.barcode_positive
        )
        gene_index = GENES.index(dict(TARGETS)[perturbation])
        source_index = fixture.cell_ids.index(str(source.cell_id))
        source_observed = fixture.expression[source_index, gene_index]
        source_seed = int.from_bytes(
            hashlib.sha256(
                f"20260828:{source.cell_id}:{GENES[gene_index]}".encode()
            ).digest()[:8],
            "big",
        )
        source_noise = float(np.random.default_rng(source_seed).normal(0.0, 0.01))
        assert source_observed - 5.0 - source_noise == pytest.approx(expected)
        for rank in (1, 5, 6, 15, 16, 30, 31, 60):
            neighbor_id = next(
                relation.neighbor_cell_id
                for relation in fixture.manifest.neighbour_relations
                if relation.source_cell_id == source.cell_id and relation.rank == rank
            )
            neighbor_index = fixture.cell_ids.index(neighbor_id)
            neighbor_observed = fixture.expression[neighbor_index, gene_index]
            neighbor_seed = int.from_bytes(
                hashlib.sha256(
                    f"20260828:{neighbor_id}:{GENES[gene_index]}".encode()
                ).digest()[:8],
                "big",
            )
            neighbor_noise = float(
                np.random.default_rng(neighbor_seed).normal(0.0, 0.01)
            )
            assert neighbor_observed - 5.0 - neighbor_noise == pytest.approx(
                expected * math.exp(-rank / 8.0)
            )


def test_positive_crc_cannot_rescue_missing_bridge_or_integrated_claim() -> None:
    policy = _policy()
    crc = _claim_evidence(
        "bridge",
        "crc_positive",
        policy,
        role="application_only",
        identity_character="f",
    )
    bridge = evaluate_bridge_claim((crc,), policy)
    integrated = derive_integrated_claim(
        (
            _admitted_module("spatial", policy),
            _admitted_module("intracellular_causal", policy),
            bridge,
        ),
        policy,
    )
    assert bridge.status == "blocked"
    assert integrated.status == "audit_only"
    assert integrated.allowed_use == "separate_module_claims_only"
    assert crc.artifact_identity in integrated.application_evidence_identities


def test_synthetic_metric_pass_cannot_become_scientific_promotion(
    tmp_path: Path,
) -> None:
    result = run_fixture_scenario("valid", tmp_path)
    policy = _policy()
    comparators = dict(policy.required_comparators)["bridge"]
    synthetic = tuple(
        _claim_evidence(
            "bridge",
            comparator,
            policy,
            role="synthetic_audit_only",
            identity_character=identity,
        )
        for comparator, identity in zip(comparators, "cd")
    )
    crc = _claim_evidence(
        "bridge",
        "crc_positive",
        policy,
        role="application_only",
        identity_character="f",
    )
    bridge = evaluate_bridge_claim((*synthetic, crc), policy)
    integrated = derive_integrated_claim(
        (
            _admitted_module("spatial", policy),
            _admitted_module("intracellular_causal", policy),
            bridge,
        ),
        policy,
    )
    assert result.status == "audit_metric_gate_passed"
    assert bridge.status == "audit_only"
    assert integrated.status == "audit_only"
    assert integrated.allowed_use == "separate_module_claims_only"
    assert crc.artifact_identity in integrated.application_evidence_identities


def test_adapter_contract_has_no_evaluation_or_oracle_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "_observed_predictions" not in globals()
    adapter = globals().get("_fit_development_only_adapter")
    assert callable(adapter)
    parameter_names = tuple(inspect.signature(adapter).parameters)
    assert "evaluation_expression" not in parameter_names
    assert "evaluation_cell_ids" not in parameter_names
    fixture = build_five_animal_fixture()
    development = _development_adapter_input(fixture)

    class ForbiddenEvaluationView:
        development_expression = development.development_expression
        development_cell_ids = development.development_cell_ids
        development_animals = development.development_animals
        development_rows = development.development_rows
        development_relations = development.development_relations

        def __getattr__(self, name: str) -> object:
            if name in {"evaluation_expression", "evaluation_cell_ids"}:
                raise AssertionError(f"forbidden evaluation access: {name}")
            raise AttributeError(name)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("adapter attempted to read evaluation outcomes")

    monkeypatch.setattr(
        __import__(__name__, fromlist=["build_bridge_effect_table"]),
        "build_bridge_effect_table",
        forbidden,
    )
    for method_id in ("hypersca", "matched_euclidean_spatial_causal"):
        fitted = _fit_development_only_adapter(
            ForbiddenEvaluationView(), method_id=method_id, model_seed=11
        )
        predictions = _predict_evaluation_units(
            fitted,
            evaluation_units=_evaluation_prediction_units(
                fixture.manifest, fixture.eligibility
            ),
        )
        assert predictions


def test_task6_relations_cover_every_frozen_rank_with_matching_signal() -> None:
    fixture = build_five_animal_fixture()
    relation_counts: dict[str, int] = {}
    for relation in fixture.manifest.neighbour_relations:
        relation_counts[relation.source_cell_id] = (
            relation_counts.get(relation.source_cell_id, 0) + 1
        )
    assert len(relation_counts) == 150
    assert set(relation_counts.values()) == {60}
    assert {relation.rank for relation in fixture.manifest.neighbour_relations} == set(
        range(1, 61)
    )
    source_by_id = {
        str(row.cell_id): row for row in fixture.raw_cells.itertuples(index=False)
    }
    assert all(
        relation.neighbor_cell_id
        == (
            f"{source_by_id[relation.source_cell_id].animal_id}_"
            f"{source_by_id[relation.source_cell_id].section_id}_"
            f"{source_by_id[relation.source_cell_id].perturbation_id}_"
            f"a_neighbor_{source_by_id[relation.source_cell_id].spatial_block}_"
            f"rank_{relation.rank:02d}"
        )
        for relation in fixture.manifest.neighbour_relations
    )
    expression_by_cell = {
        cell_id: fixture.expression[index]
        for index, cell_id in enumerate(fixture.cell_ids)
    }
    for relation in fixture.manifest.neighbour_relations:
        if relation.is_safe_control:
            continue
        perturbation = relation.source_perturbation_id
        assert f"_{perturbation}_z_source_" in relation.source_cell_id
        gene = dict(TARGETS)[perturbation]
        noise_seed = int.from_bytes(
            hashlib.sha256(
                f"20260828:{relation.neighbor_cell_id}:{gene}".encode()
            ).digest()[:8],
            "big",
        )
        noise = float(np.random.default_rng(noise_seed).normal(0.0, 0.01))
        observed = expression_by_cell[relation.neighbor_cell_id][GENES.index(gene)]
        assert observed - 5.0 - noise == pytest.approx(
            OWN_DELTA[perturbation] * math.exp(-relation.rank / 8.0)
        )


def test_manifest_relations_are_exactly_the_one_full_frame_task6_result() -> None:
    fixture = build_five_animal_fixture()
    fresh = build_bridge_neighbors(
        fixture.raw_cells, max_rank=60, safe_control_label="mSafe"
    )
    expected = {
        (
            str(row.source_cell_id),
            str(row.neighbor_cell_id),
            int(cast(Any, row.rank)),
            str(row.perturbation_id),
        )
        for row in fresh.itertuples(index=False)
    }
    observed = {
        (
            relation.source_cell_id,
            relation.neighbor_cell_id,
            relation.rank,
            relation.source_perturbation_id,
        )
        for relation in fixture.manifest.neighbour_relations
    }
    assert len(expected) == len(fresh) == len(fixture.manifest.neighbour_relations)
    assert observed == expected


def test_every_source_nearest_sixty_is_its_shared_context_block_pool() -> None:
    fixture = build_five_animal_fixture()
    for (_, _), section in fixture.raw_cells.groupby(
        ["animal_id", "section_id"], sort=False
    ):
        rows = tuple(section.itertuples(index=False))
        for source in (row for row in rows if row.barcode_positive):
            nearest = sorted(
                (row for row in rows if row.cell_id != source.cell_id),
                key=lambda row: (
                    (float(row.x) - float(source.x)) ** 2
                    + (float(row.y) - float(source.y)) ** 2,
                    str(row.cell_id),
                ),
            )[:60]
            assert tuple(str(row.cell_id) for row in nearest) == tuple(
                f"{source.animal_id}_{source.section_id}_"
                f"{source.perturbation_id}_a_neighbor_"
                f"{source.spatial_block}_rank_{rank:02d}"
                for rank in range(1, 61)
            )


def test_scenario_is_reconstructed_from_verified_publisher_bundle(
    tmp_path: Path,
) -> None:
    result = run_fixture_scenario("valid", tmp_path)
    assert result.output_dir.is_dir()
    assert (result.output_dir / "run_manifest.json").is_file()
    assert (result.output_dir / "method_status.json").is_file()
    assert (result.output_dir / "scenario_result.json").is_file()
    assert (result.output_dir / "predictions.json").is_file()
    prediction_record = json.loads(
        (result.output_dir / "predictions.json").read_bytes()
    )
    assert {row["method_id"] for row in prediction_record["predictions"]} == {
        "hypersca",
        "hypersca_own_only",
        "matched_euclidean_spatial_causal",
    }
    assert all(
        set(row) == {"method_id", "unit_id", "endpoint", "predicted_delta"}
        for row in prediction_record["predictions"]
    )
    assert (
        result.bundle_identity_sha256
        == verify_run_evidence_bundle(result.output_dir).bundle_identity_sha256
    )


def test_semantic_replay_rejects_synchronized_prediction_tamper(
    tmp_path: Path,
) -> None:
    result = run_fixture_scenario("valid", tmp_path / "original")
    verified = verify_run_evidence_bundle(result.output_dir)
    prediction_record = _read_verified_json_artifact(verified, "predictions.json")
    result_record = _read_verified_json_artifact(verified, "scenario_result.json")
    rows = cast(list[dict[str, object]], prediction_record["predictions"])
    prediction = next(
        row
        for row in rows
        if row["method_id"] == "hypersca" and row["endpoint"] == "neighbor"
    )
    prediction["predicted_delta"] = (
        cast(float, prediction["predicted_delta"]) + 1.0
    )
    status, hypersca_rmse, euclidean_rmse, own_only_rmse = (
        _recompute_published_scenario(
            build_five_animal_fixture(), "valid", prediction_record
        )
    )
    result_record.update(
        {
            "status": status,
            "hypersca_rmse": hypersca_rmse,
            "matched_euclidean_rmse": euclidean_rmse,
            "own_only_rmse": own_only_rmse,
        }
    )
    result_record["prediction_identity_sha256"] = canonical_sha256(
        prediction_record
    )
    identities = (
        verified.identity,
        replace(verified.identity, config_identity_sha256="f" * 64),
    )
    for index, identity in enumerate(identities):
        tampered_dir = tmp_path / f"tampered_{index}" / "bundle"
        tampered_dir.parent.mkdir(parents=True)
        with RunEvidencePublisher.begin(
            output_dir=tampered_dir,
            identity=identity,
            statistical_unit_record=dict(verified.statistical_unit_record),
            required_artifacts=("predictions.json", "scenario_result.json"),
            maximum_bundle_bytes=64 * 1024,
        ) as publisher:
            publisher.add_bytes(
                "predictions.json",
                canonical_json_bytes(prediction_record),
                media_type="application/json",
            )
            publisher.add_bytes(
                "scenario_result.json",
                canonical_json_bytes(result_record),
                media_type="application/json",
            )
            publisher.finalize_completed(summary=result_record)
        tampered = verify_run_evidence_bundle(tampered_dir)
        with pytest.raises(AssertionError):
            _replay_published_scenario(tampered)


def test_all_scenarios_share_the_one_frozen_exponential_dgp() -> None:
    assert "_scenario_dgp" not in globals()
    fixture = build_five_animal_fixture()
    assert fixture.dgp == "exponential"


def test_published_code_identity_is_bound_to_adapter_source(tmp_path: Path) -> None:
    result = run_fixture_scenario("valid", tmp_path)
    verified = verify_run_evidence_bundle(result.output_dir)
    expected = canonical_sha256(
        {
            name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for name, function in (
                ("development_input", _development_adapter_input),
                ("development_effects", _development_effects),
                ("evaluation_units", _evaluation_prediction_units),
                ("fit", _fit_development_only_adapter),
                ("predict", _predict_evaluation_units),
                ("expected_predictions", _expected_method_predictions),
                ("scenario_computation", _immutable_scenario_computation),
            )
        }
    )
    assert verified.identity.code_identity_sha256 == expected


def test_adapter_source_mutation_changes_code_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _adapter_code_identity()
    original = _predict_evaluation_units

    def mutated_adapter(
        adapter: FittedDevelopmentAdapter,
        *,
        evaluation_units: EvaluationPredictionUnits,
    ) -> tuple[BridgePrediction, ...]:
        return original(adapter, evaluation_units=evaluation_units)

    monkeypatch.setattr(
        __import__(__name__, fromlist=["_predict_evaluation_units"]),
        "_predict_evaluation_units",
        mutated_adapter,
    )
    assert _adapter_code_identity() != baseline


def test_fixture_cache_never_shares_mutable_frames_or_arrays() -> None:
    first = build_five_animal_fixture()
    second = build_five_animal_fixture()
    assert first is not second
    assert first.raw_cells is not second.raw_cells
    assert first.expression is not second.expression
    original_id = str(second.raw_cells.loc[0, "cell_id"])
    original_value = float(second.expression[0, 0])
    first.raw_cells.loc[0, "cell_id"] = "mutated_test_cell"
    with pytest.raises(ValueError, match="read-only"):
        first.expression[0, 0] = -999.0
    rebuilt = build_five_animal_fixture()
    assert str(rebuilt.raw_cells.loc[0, "cell_id"]) == original_id
    assert rebuilt.expression[0, 0] == original_value


def test_permuted_fixture_replays_task6_from_the_actual_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_five_animal_fixture()
    observed_orders: list[tuple[str, ...]] = []
    production_builder = build_bridge_neighbors

    def spy(cells: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        observed_orders.append(tuple(cast(str, value) for value in cells["cell_id"]))
        return production_builder(cells, **kwargs)

    monkeypatch.setattr(
        __import__(__name__, fromlist=["build_bridge_neighbors"]),
        "build_bridge_neighbors",
        spy,
    )
    permuted = build_five_animal_fixture(7)
    assert len(observed_orders) == 1
    assert observed_orders[0] == tuple(
        cast(str, value) for value in permuted.raw_cells["cell_id"]
    )


def test_model_seed_changes_predictions_artifact_and_run_identity(
    tmp_path: Path,
) -> None:
    results = tuple(
        run_fixture_scenario(
            "valid", tmp_path / f"seed-{model_seed}", model_seed=model_seed
        )
        for model_seed in MODEL_SEEDS
    )
    assert len({result.prediction_identity_sha256 for result in results}) == 3
    assert len({result.artifact_identity_sha256 for result in results}) == 3
    assert len({result.run_identity_sha256 for result in results}) == 3
    assert {result.status for result in results} == {"audit_metric_gate_passed"}
