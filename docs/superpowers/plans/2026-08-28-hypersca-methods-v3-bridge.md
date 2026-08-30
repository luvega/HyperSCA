# HyperSCA Methods v3 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the v2.1 no-release result, implement the v3 three-family evidence policy, and build the outcome-blind, publisher-backed spatial-perturbation bridge foundation through synthetic validation and a real-data predictor-capability audit. Stop without running a real pilot when no preregistered executable HyperSCA bridge predictor exists.

**Architecture:** Preserve all audited v2.1 APIs and artifacts. Add v3-specific immutable protocol and evidence types, then implement the bridge as registry, capability, split, neighbour, scoring, comparator, and runner modules with one-way dependencies. Reuse `RunEvidencePublisher` and paired collections rather than duplicating publication or replay logic.

**Tech Stack:** Python 3.10, NumPy, pandas, PyTorch, pytest, Hypothesis, Import Linter, existing HyperSCA evidence publisher and Task S metrics.

---

## Scope and file map

This plan is Phase A and stops before any real bridge pilot or confirmatory
release. GSE274447 has three mice and could support only an audit-only pilot,
but the repository does not yet contain a frozen production predictor from
spatial perturbation inputs to neighbour-expression effects. The outcome-blind
capability audit must therefore publish `method_adapter_not_executable` and
stop. A real pilot requires a separately approved predictor-adapter protocol.
Confirmatory execution additionally remains blocked until an untouched external
cohort passes the outcome-blind audit before a future protocol identity is
frozen.

New source modules:

- `src/evaluation/methods_protocol_v3.py`: immutable v3 protocol and identity.
- `src/evaluation/methods_protocol_outcome.py`: immutable v2.1 closure.
- `src/evaluation/spatial_perturbation_registry.py`: registry and capability audit.
- `src/evaluation/spatial_perturbation_split.py`: animal splits and eligibility.
- `src/evaluation/spatial_perturbation_neighbors.py`: rank-band neighbourhoods.
- `src/evaluation/spatial_perturbation_scoring.py`: standardisation and RMSE.
- `src/evaluation/spatial_perturbation_comparators.py`: comparator contracts.
- `src/evaluation/spatial_perturbation_predictor_contract.py`: formal predictor
  capability and validated prediction-bundle boundary.
- `src/evaluation/spatial_perturbation_runner.py`: publisher orchestration for
  validated prediction bundles and terminal capability failures.

New CLIs, configuration, and evidence:

- `scripts/freeze_methods_protocol_outcome.py`
- `scripts/audit_spatial_perturbation_bridge.py`
- `scripts/validate_spatial_perturbation_predictor.py`
- `configs/spatial_perturbation_bridge_candidates_v1.json`
- `configs/hypersca_methods_v3.yaml`, generated after pre-freeze audit
- `.importlinter`
- `reports/methods_protocol_v2_1_audit/`
- `reports/methods_protocol_v3_preflight/`

## Task 1: Freeze the v2.1 no-release outcome

**Files:**
- Create: `src/evaluation/methods_protocol_outcome.py`
- Create: `scripts/freeze_methods_protocol_outcome.py`
- Create: `tests/test_methods_protocol_outcome.py`
- Create: `reports/methods_protocol_v2_1_audit/protocol_outcome.json`
- Create: `reports/methods_protocol_v2_1_audit/review.md`

- [ ] **Step 1: Write the failing closure tests**

```python
def test_v21_outcome_binds_the_no_release_audit() -> None:
    outcome = load_protocol_outcome(OUTCOME_PATH)
    assert outcome.protocol_version == "hypersca-methods-v2.1"
    assert outcome.status == "pilot_failed_no_release"
    assert outcome.release_authorized is False
    assert outcome.pilot_summary_sha256 == (
        "3fe9e90443f82a911fe02314a540cd8e3383ee016cff9c3dbb46b802490d694c"
    )
    assert len(outcome.run_identity_sha256) == 18
    assert len(set(outcome.run_identity_sha256)) == 18
    assert len(outcome.collection_identity_sha256) == 6


def test_v21_outcome_cannot_be_relabelled_as_release() -> None:
    payload = strict_json(OUTCOME_PATH)
    payload["release_authorized"] = True
    with pytest.raises(ValueError, match="no-release"):
        ProtocolOutcome.from_mapping(payload)
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_methods_protocol_outcome.py -q -p no:cacheprovider
```

Expected: collection fails because `methods_protocol_outcome` does not exist.

- [ ] **Step 3: Implement the immutable closure type**

```python
@dataclass(frozen=True, slots=True)
class ProtocolOutcome:
    protocol_version: str
    protocol_identity_sha256: str
    pilot_summary_sha256: str
    status: str
    release_authorized: bool
    run_identity_sha256: tuple[str, ...]
    collection_identity_sha256: tuple[str, ...]
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.protocol_version != "hypersca-methods-v2.1":
            raise ValueError("outcome is not frozen v2.1 evidence")
        if self.status != "pilot_failed_no_release" or self.release_authorized is not False:
            raise ValueError("v2.1 outcome must remain no-release")
        if len(self.run_identity_sha256) != 18 or len(set(self.run_identity_sha256)) != 18:
            raise ValueError("v2.1 outcome must bind 18 unique runs")
        if len(self.collection_identity_sha256) != 6:
            raise ValueError("v2.1 outcome must bind six paired collections")
```

Validate every identity as lowercase SHA-256 and use strict bounded JSON. The
thin script verifies the exact summary SHA, extracts identities, writes
canonical JSON with exclusive publication, and refuses an existing output.

- [ ] **Step 4: Generate records and run GREEN**

```bash
python scripts/freeze_methods_protocol_outcome.py --pilot-summary /home/a/Data/HyperSCA/results/methods_pilot_v22_publisher/pilot_audit_summary.json --output reports/methods_protocol_v2_1_audit/protocol_outcome.json
pytest tests/test_methods_protocol_outcome.py -q -p no:cacheprovider
```

Expected: PASS; JSON has 18 runs, six collections, and no release authority.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/methods_protocol_outcome.py scripts/freeze_methods_protocol_outcome.py tests/test_methods_protocol_outcome.py reports/methods_protocol_v2_1_audit
git commit -m "docs: freeze methods v2.1 audit outcome"
```

## Task 2: Add the immutable v3 protocol schema

**Files:**
- Create: `src/evaluation/methods_protocol_v3.py`
- Create: `tests/test_methods_protocol_v3.py`
- Create: `tests/property/test_methods_protocol_v3_properties.py`

- [ ] **Step 1: Write RED tests for the exact contract**

```python
def test_v3_protocol_has_three_claim_families() -> None:
    protocol = build_methods_protocol_v3(
        bridge_role="pilot_audit_only",
        capability_identity_sha256="a" * 64,
    )
    assert protocol.protocol_version == "hypersca-methods-v3.0"
    assert protocol.claim_ids == ("spatial", "intracellular_causal", "bridge")
    assert protocol.primary_metrics == (
        "neighborhood_preservation_at_k",
        "directed_edge_average_precision",
        "neighbor_effect_rmse",
    )
    assert protocol.multiple_testing == "distinct_families_no_cross_adjustment"
    assert protocol.integrated_gate == "intersection_union_all_three"
    assert protocol.integrated_claim_enabled is False
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_methods_protocol_v3.py tests/property/test_methods_protocol_v3_properties.py -q -p no:cacheprovider
```

Expected: collection fails because `methods_protocol_v3` is absent.

- [ ] **Step 3: Implement v3 dataclasses and canonical identity**

```python
BRIDGE_PRIMARY_BANDS = (("proximal", 1, 5), ("local", 6, 15))
BRIDGE_SECONDARY_BANDS = (("transition", 16, 30), ("distal", 31, 60))


@dataclass(frozen=True, slots=True)
class MethodsProtocolV3:
    protocol_version: str
    claim_ids: tuple[str, str, str]
    primary_metrics: tuple[str, str, str]
    pilot_seeds: tuple[int, int, int]
    release_seeds: tuple[int, int, int, int, int]
    bootstrap_resamples: int
    confidence: float
    multiple_testing: str
    integrated_gate: str
    bridge_role: str
    capability_identity_sha256: str
    integrated_claim_enabled: bool
    crc_role: str
```

`build_methods_protocol_v3()` accepts only `pilot_audit_only` or
`confirmatory`, freezes seeds `(11, 23, 47)` and `(11, 23, 47, 71, 101)`, uses
10,000 resamples and confidence 0.95, and sets integrated enablement only for a
confirmatory bridge. Add `protocol_to_mapping_v3()` and
`protocol_identity_v3()`; the mapping must contain no Holm field.

- [ ] **Step 4: Add property tests**

Use Hypothesis to reject bool-as-int, integer subclasses, mutable sequences,
unsafe text, non-finite floats, duplicate claim IDs, and malformed capability
identities. Equal inputs must produce the same canonical identity.

- [ ] **Step 5: Run GREEN plus v2 regression**

```bash
pytest tests/test_methods_protocol.py tests/test_methods_protocol_v3.py tests/property/test_methods_protocol_properties.py tests/property/test_methods_protocol_v3_properties.py -q -p no:cacheprovider
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/methods_protocol_v3.py tests/test_methods_protocol_v3.py tests/property/test_methods_protocol_v3_properties.py
git commit -m "feat: define methods v3 evidence protocol"
```

## Task 3: Extend EvidencePolicy without changing v2 decisions

**Files:**
- Modify: `src/discovery/evidence_policy.py`
- Create: `tests/discovery/test_evidence_policy_v3.py`
- Modify: `tests/discovery/test_evidence_policy.py`

- [ ] **Step 1: Write RED three-family tests**

```python
def test_integrated_claim_requires_all_three_components() -> None:
    decisions = (
        v3_decision("spatial", "admitted"),
        v3_decision("intracellular_causal", "admitted"),
        v3_decision("bridge", "audit_only"),
    )
    result = derive_integrated_claim(decisions, policy_v3())
    assert result.status == "audit_only"
    assert result.allowed_use == "separate_module_claims_only"


def test_bridge_uses_intersection_union_without_holm() -> None:
    decision = evaluate_bridge_claim(bridge_pair(0.01, 0.04), policy_v3())
    assert decision.nominal_p_value == 0.04
    assert decision.multiplicity_adjustment == "none_intersection_union"
```

Also test that CRC cannot rescue a missing bridge.

- [ ] **Step 2: Run RED**

```bash
pytest tests/discovery/test_evidence_policy_v3.py -q -p no:cacheprovider
```

- [ ] **Step 3: Add v3 types beside unchanged v2 types**

```python
@dataclass(frozen=True, slots=True)
class V3ClaimEvidence:
    claim_id: str
    protocol_version: str
    primary_metric: str
    comparator_id: str
    paired_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    nominal_p_value: float | None
    attempted_units: int
    completed_units: int
    evidence_role: str
    artifact_identity: str


@dataclass(frozen=True, slots=True)
class EvidencePolicyV3:
    protocol_version: str
    family_primary_metrics: tuple[tuple[str, str], ...]
    required_comparators: tuple[tuple[str, tuple[str, ...]], ...]
    nominal_alpha: float
    minimum_lower_bound: float
    bridge_role: str
    integrated_claim_enabled: bool
```

Implement `evaluate_v3_claim()`, `evaluate_bridge_claim()`, and
`derive_integrated_claim()`. Only three admitted decisions plus a confirmatory
bridge can create `integrated_spatial_causal_gain`. Keep all v2 exported names
and identities compatible.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest tests/discovery/test_evidence_policy.py tests/discovery/test_evidence_policy_v3.py tests/discovery/test_benchmark_claims.py -q -p no:cacheprovider
git add src/discovery/evidence_policy.py tests/discovery/test_evidence_policy.py tests/discovery/test_evidence_policy_v3.py
git commit -m "feat: gate integrated claims on three evidence families"
```

## Task 4: Add the outcome-blind candidate registry and capability audit

**Files:**
- Create: `src/evaluation/spatial_perturbation_registry.py`
- Create: `configs/spatial_perturbation_bridge_candidates_v1.json`
- Create: `scripts/audit_spatial_perturbation_bridge.py`
- Create: `tests/test_spatial_perturbation_registry.py`
- Create: `tests/property/test_spatial_perturbation_registry_properties.py`

- [ ] **Step 1: Write RED tests that exclude outcomes**

```python
def test_three_mice_are_pilot_only() -> None:
    result = audit_bridge_capability(candidate(), metadata_summary(animals=3))
    assert result.status == "pilot_audit_only"
    assert result.confirmatory_capable is False
    assert "insufficient_biological_replicates" in result.blocking_reasons
    assert "neighbor_effect_rmse" not in result.to_mapping()


def test_confirmatory_requires_five_specimens_and_two_cohorts() -> None:
    assert not audit_bridge_capability(candidate(), metadata_summary(animals=5, cohorts=1)).confirmatory_capable
    assert audit_bridge_capability(candidate(), metadata_summary(animals=5, cohorts=2)).confirmatory_capable
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_spatial_perturbation_registry.py tests/property/test_spatial_perturbation_registry_properties.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement immutable registry and audit records**

```python
@dataclass(frozen=True, slots=True)
class BridgeCandidate:
    candidate_id: str
    accession: str
    platform: str
    biological_specimens: tuple[str, ...]
    sections_by_specimen: tuple[tuple[str, tuple[str, ...]], ...]
    safe_control_label: str
    perturbation_labels: tuple[str, ...]
    source_uri: str
    source_identity_sha256: str


@dataclass(frozen=True, slots=True)
class BridgeCapabilityResult:
    candidate_id: str
    status: str
    confirmatory_capable: bool
    biological_specimen_count: int
    cohort_count: int
    coverage: float
    blocking_reasons: tuple[str, ...]
    capability_identity_sha256: str
```

The audit accepts a metadata-only summary with IDs, counts, coordinate
availability, measured genes, and label-quality counts. Its public API must not
accept an expression matrix, prediction, response effect, or metric.

- [ ] **Step 4: Add strict input and CLI tests**

Cover duplicate JSON keys, deep nesting, non-finite and huge numbers, symlinks,
non-regular files, mutable mappings, exclusive output, and exact candidate
order. `--help` imports no NumPy, pandas, torch, or model code.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest tests/test_spatial_perturbation_registry.py tests/property/test_spatial_perturbation_registry_properties.py -q -p no:cacheprovider
python scripts/audit_spatial_perturbation_bridge.py --help
git add src/evaluation/spatial_perturbation_registry.py configs/spatial_perturbation_bridge_candidates_v1.json scripts/audit_spatial_perturbation_bridge.py tests/test_spatial_perturbation_registry.py tests/property/test_spatial_perturbation_registry_properties.py
git commit -m "feat: audit spatial perturbation capability"
```

## Task 5: Freeze animal-level splits and eligibility

**Files:**
- Create: `src/evaluation/spatial_perturbation_split.py`
- Create: `tests/test_spatial_perturbation_split.py`
- Create: `tests/property/test_spatial_perturbation_split_properties.py`

- [ ] **Step 1: Write RED split and threshold tests**

```python
def test_fold_keeps_sections_with_their_animal() -> None:
    split = build_pilot_fold(metadata(), evaluation_animal="mouse_1")
    assert split.development_animals == ("mouse_2", "mouse_3")
    assert split.evaluation_animals == ("mouse_1",)
    assert not set(split.development_rows) & set(split.evaluation_rows)


def test_fixed_source_threshold_rejects_nineteen_cells() -> None:
    result = evaluate_bridge_eligibility(unit_counts(source=19, neighbours=50, blocks=3))
    assert result.eligible is False
    assert result.reason == "insufficient_perturbation_coverage"
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_spatial_perturbation_split.py tests/property/test_spatial_perturbation_split_properties.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement exact frozen thresholds**

```python
MIN_SOURCE_CELLS = 20
MIN_SAFE_SOURCE_CELLS = 20
MIN_BAND_NEIGHBOURS = 50
MIN_CELL_TYPE_NEIGHBOURS = 30
MIN_SPATIAL_BLOCKS = 3
MIN_COVERAGE = 0.80
MAX_ABSTENTION = 0.20


@dataclass(frozen=True, slots=True)
class BridgeSplitManifest:
    split_id: str
    split_seed: int
    development_animals: tuple[str, ...]
    evaluation_animals: tuple[str, ...]
    train_rows: tuple[int, ...]
    tune_rows: tuple[int, ...]
    evaluation_rows: tuple[int, ...]
    gene_names: tuple[str, ...]
    perturbations: tuple[str, ...]
    split_identity_sha256: str
```

The split function does not accept model seed. It uses split seed 11 and keeps
whole animals together. Hypothesis tests cover row-order changes, repeated
sections, adversarial sequences, duplicate cell IDs, and a section assigned to
two animals.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest tests/test_spatial_perturbation_split.py tests/property/test_spatial_perturbation_split_properties.py -q -p no:cacheprovider
git add src/evaluation/spatial_perturbation_split.py tests/test_spatial_perturbation_split.py tests/property/test_spatial_perturbation_split_properties.py
git commit -m "feat: freeze animal-level bridge splits"
```

## Task 6: Build deterministic spatial neighbour bands

**Files:**
- Create: `src/evaluation/spatial_perturbation_neighbors.py`
- Create: `tests/test_spatial_perturbation_neighbors.py`
- Create: `tests/property/test_spatial_perturbation_neighbors_properties.py`

- [ ] **Step 1: Write RED rank and contamination tests**

```python
def test_primary_bands_are_exact_and_barcode_negative() -> None:
    units = build_bridge_neighbors(synthetic_coordinates(), max_rank=60)
    assert set(units.loc[units.band == "proximal", "rank"]) == set(range(1, 6))
    assert set(units.loc[units.band == "local", "rank"]) == set(range(6, 16))
    assert not units.neighbor_barcode_positive.any()


def test_cross_perturbation_contamination_is_excluded() -> None:
    units = build_bridge_neighbors(overlapping_sources(), max_rank=60)
    assert units.query("rank <= 15 and competing_perturbation").empty
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_spatial_perturbation_neighbors.py tests/property/test_spatial_perturbation_neighbors_properties.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement bounded within-section kNN**

Return exact columns:

```python
NEIGHBOR_COLUMNS = (
    "animal_id", "section_id", "spatial_block", "source_cell_id",
    "neighbor_cell_id", "perturbation_id", "source_cell_type",
    "neighbor_cell_type", "rank", "band", "is_safe_control",
)
```

Use squared distance and cell ID as the deterministic tie breaker. Query at
most 61 neighbours per section; never allocate an all-cell distance matrix.
Deduplicate by animal, section, perturbation, and neighbour cell, retaining the
minimum rank. Property tests cover disjoint bands, section isolation, input
permutation, duplicate IDs, non-finite coordinates, and bounded row counts.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest tests/test_spatial_perturbation_neighbors.py tests/property/test_spatial_perturbation_neighbors_properties.py -q -p no:cacheprovider
git add src/evaluation/spatial_perturbation_neighbors.py tests/test_spatial_perturbation_neighbors.py tests/property/test_spatial_perturbation_neighbors_properties.py
git commit -m "feat: construct frozen bridge neighborhoods"
```

## Task 7: Implement train-only effects and hierarchical RMSE

**Files:**
- Create: `src/evaluation/spatial_perturbation_scoring.py`
- Create: `tests/test_spatial_perturbation_scoring.py`
- Create: `tests/property/test_spatial_perturbation_scoring_properties.py`

- [ ] **Step 1: Write RED metric tests**

```python
def test_perfect_prediction_has_zero_primary_rmse() -> None:
    result = score_bridge_predictions(effect_table(predicted_equals_observed=True))
    assert result.neighbor_effect_rmse == pytest.approx(0.0)


def test_primary_bands_are_equal_weight() -> None:
    balanced = effect_table(proximal_rows=10, local_rows=10)
    duplicated = effect_table(proximal_rows=10_000, local_rows=10)
    assert score_bridge_predictions(balanced).neighbor_effect_rmse == (
        score_bridge_predictions(duplicated).neighbor_effect_rmse
    )
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_spatial_perturbation_scoring.py tests/property/test_spatial_perturbation_scoring_properties.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement train-only standardisation**

```python
@dataclass(frozen=True, slots=True)
class TrainControlStandardizer:
    genes: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    training_identity_sha256: str


def fit_train_control_standardizer(
    expression: np.ndarray,
    *,
    gene_names: tuple[str, ...],
    control_rows: tuple[int, ...],
) -> TrainControlStandardizer:
    values = np.log1p(expression[np.asarray(control_rows)])
    center = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64, ddof=0)
    scale[scale <= 1e-6] = 1.0
    return freeze_standardizer(gene_names, center, scale, control_rows)
```

The fit API accepts training control rows only. Applying the result checks exact
gene order and cannot refit from tune or evaluation data.

- [ ] **Step 4: Implement observed effects and exact aggregation**

Observed effect is eligible-neighbour mean minus matched mSafe-neighbour mean.
Aggregate in this order: gene, band, neighbour cell type, perturbation, animal.
Only proximal and local enter the primary; they are equally weighted. Return an
immutable `BridgeScore` with RMSE, own RMSE, PCC, distance calibration, sign
accuracy, coverage, abstention, and the animal-level unit table.

- [ ] **Step 5: Add property tests**

Cover train-standardised scale invariance, cell or section duplication not
increasing animal count, own effects excluded from the primary, identical units
for all methods, missing predictions failing rather than becoming zero, and
finite bounded numeric inputs.

- [ ] **Step 6: Run GREEN and commit**

```bash
pytest tests/test_spatial_perturbation_scoring.py tests/property/test_spatial_perturbation_scoring_properties.py tests/test_task_s_benchmark.py -q -p no:cacheprovider
git add src/evaluation/spatial_perturbation_scoring.py tests/test_spatial_perturbation_scoring.py tests/property/test_spatial_perturbation_scoring_properties.py
git commit -m "feat: score spatial perturbation bridge effects"
```

## Task 8: Freeze bridge comparators and fair budgets

**Files:**
- Create: `src/evaluation/spatial_perturbation_comparators.py`
- Create: `tests/test_spatial_perturbation_comparators.py`

- [ ] **Step 1: Write RED comparator tests**

```python
def test_own_only_sets_neighbor_predictions_to_positive_zero() -> None:
    frame = predict_bridge_own_only(own_predictions())
    values = frame.loc[frame.endpoint == "neighbor", "predicted_effect"].to_numpy()
    assert np.equal(values, 0.0).all()
    assert not np.signbit(values).any()


def test_euclidean_budget_must_share_spatial_graph() -> None:
    with pytest.raises(ValueError, match="spatial_graph_identity"):
        validate_bridge_comparator_budgets(hypersca_budget(), changed_graph_budget())
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_spatial_perturbation_comparators.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement comparator contracts**

```python
@dataclass(frozen=True, slots=True)
class BridgeModelBudget:
    method_id: str
    geometry: str
    parameter_count: int
    optimizer_family: str
    max_updates: int
    early_stopping_patience: int
    tuning_trials: int
    data_identity_sha256: str
    gene_identity_sha256: str
    spatial_graph_identity_sha256: str
    propagation_identity_sha256: str
    seed: int
```

Matched Euclidean may change only geometry and parameter count within 5%.
`hypersca_own_only` retains the exact own-effect predictions and writes positive
zero for every neighbour. Adapt existing `fixed_distance_decay` for secondary
output only; it never satisfies a required bridge comparison.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest tests/test_spatial_perturbation_comparators.py tests/test_task_s_benchmark.py tests/test_methods_protocol.py -q -p no:cacheprovider
git add src/evaluation/spatial_perturbation_comparators.py tests/test_spatial_perturbation_comparators.py
git commit -m "feat: freeze bridge comparator contracts"
```

## Task 9: Publish validated bridge predictions through RunEvidencePublisher

**Files:**
- Create: `src/evaluation/spatial_perturbation_predictor_contract.py`
- Create: `src/evaluation/spatial_perturbation_runner.py`
- Create: `scripts/validate_spatial_perturbation_predictor.py`
- Create: `tests/test_spatial_perturbation_predictor_contract.py`
- Create: `tests/test_spatial_perturbation_runner.py`
- Modify: `tests/test_run_evidence_collection.py`

- [ ] **Step 1: Write RED capability and publication tests**

```python
def test_missing_production_predictor_is_a_terminal_capability_result() -> None:
    result = audit_bridge_predictor_capability(
        registry=production_registry_without_bridge_adapter(),
        protocol=pilot_only_protocol(),
    )
    assert result.status == "method_adapter_not_executable"
    assert result.executable is False
    assert result.adapter_identity_sha256 is None


def test_runner_accepts_only_a_validated_prediction_bundle(tmp_path: Path) -> None:
    bundle = validated_synthetic_prediction_bundle()
    result = publish_spatial_perturbation_evidence(
        predictions=bundle,
        protocol=pilot_only_protocol(),
        output_dir=tmp_path / "run",
    )
    verified = verify_run_evidence_bundle(result.output_dir)
    assert verified.identity.claim_id == "bridge"
    assert verified.identity.evidence_role == "synthetic_audit_only"
    assert verified.identity.data_scopes == ("synthetic",)


def test_failure_bundle_contains_no_fabricated_primary_metric(tmp_path: Path) -> None:
    verified = publish_and_verify_capability_failure(
        tmp_path,
        status="method_adapter_not_executable",
    )
    assert verified.terminal_status == "method_adapter_not_executable"
    assert "primary_metric_summary.json" not in verified.artifact_paths
```

- [ ] **Step 2: Run RED**

```bash
pytest tests/test_spatial_perturbation_predictor_contract.py tests/test_spatial_perturbation_runner.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement the production predictor gate**

Add immutable `BridgePredictorCapability` and `BridgePredictionBundle` types.
The capability audit accepts only a preregistered production adapter with a
source identity, exact input/output schema, model configuration identity, and a
successful outcome-blind executable-interface probe. It never imports a model,
loads outcome values, or fabricates predictions. The current repository has no
such adapter, so its required result is:

```json
{
  "status": "method_adapter_not_executable",
  "executable": false,
  "adapter_identity_sha256": null,
  "blocking_reasons": ["no_preregistered_bridge_predictor_adapter"]
}
```

`BridgePredictionBundle` binds method, protocol, data, split, unit, code, model
seed, prediction schema, and prediction bytes. Direct mappings are defensively
copied and deeply frozen. A production bundle cannot claim synthetic origin,
and a synthetic bundle cannot enter a production evidence role.

- [ ] **Step 4: Implement thin publication orchestration**

The runner accepts a validated prediction bundle, not predictor callables. It
calls split, neighbours, comparators, scoring, and the publisher; it contains
none of their scientific algorithms and never fits a model. Stage and publish:

```text
split_manifest.json
capability_record.json
neighbor_units.csv
predictions_hypersca.csv
predictions_matched_euclidean.csv
predictions_hypersca_own_only.csv
primary_metric_units.csv
primary_metric_summary.json
secondary_metrics.csv
resource_usage.json
claim_decision.json
```

The identity binds ordered animals, sections, blocks, perturbations, cell types,
bands, genes, standardisation, aggregation, contamination, model budgets, code,
capability record, and the exact prediction bundle. Recheck every input and code
path before finalisation. A capability failure publishes only capability,
status, identity, and resource evidence; it contains no primary metric.

- [ ] **Step 5: Keep the CLI thin**

The capability CLI accepts registry, protocol, method id, and output path. It
does not accept a model factory, arbitrary Python import path, prediction path,
or outcome data. `--help` imports no NumPy, pandas, torch, models, or scoring
modules.

- [ ] **Step 6: Run GREEN and commit**

```bash
pytest tests/test_spatial_perturbation_predictor_contract.py tests/test_spatial_perturbation_runner.py tests/test_run_evidence_publisher.py tests/test_run_evidence_collection.py tests/property/test_run_evidence_properties.py -q -p no:cacheprovider
git add src/evaluation/spatial_perturbation_predictor_contract.py src/evaluation/spatial_perturbation_runner.py scripts/validate_spatial_perturbation_predictor.py tests/test_spatial_perturbation_predictor_contract.py tests/test_spatial_perturbation_runner.py tests/test_run_evidence_collection.py
git commit -m "feat: gate spatial perturbation prediction evidence"
```

## Task 10: Enforce eight architectural boundaries

**Files:**
- Modify: `.importlinter`
- Modify: `pyproject.toml`
- Create: `tests/test_v3_import_boundaries.py`
- Create: `src/evaluation/safe_declaration_reader.py`
- Modify: `scripts/audit_spatial_perturbation_bridge.py`
- Modify: `scripts/validate_spatial_perturbation_predictor.py`
- Modify: `tests/test_spatial_perturbation_runner.py`

- [ ] **Step 1: Add explicit development dependencies**

```toml
dev = [
  "pytest>=7,<9",
  "tomli>=2; python_version < '3.11'",
  "hypothesis>=6,<7",
  "import-linter>=2,<3",
]
```

- [ ] **Step 2: Write a test that is RED before the contracts exist**

```python
def test_import_linter_contracts_pass() -> None:
    result = subprocess.run(
        ["lint-imports", "--config", ".importlinter"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 3: Add the contracts**

```ini
[importlinter]
root_package = src

[importlinter:contract:evidence-policy-domain-only]
name = Evidence policy is domain-only
type = forbidden
source_modules = src.discovery.evidence_policy
forbidden_modules =
    src.models
    src.causal
    src.perturbation
    src.data
    src.evaluation.methods_pilot
    src.evaluation.methods_causal_pilot
    src.evaluation.task_s_benchmark
    src.evaluation.task_c_benchmark
    src.evaluation.task_c_method_run
    src.evaluation.task_c_aggregation
    src.evaluation.task_c_rehearsal
    src.evaluation.task_c_runtime
    src.evaluation.task_c_data
    src.evaluation.task_c_acquisition
    src.evaluation.task_c_profile_input
    src.evaluation.task_c_formal_export
    src.evaluation.task_c_predictions
    src.evaluation.task_c_tuning
    src.evaluation.safe_declaration_reader
    src.evaluation.spatial_perturbation_runner

[importlinter:contract:publisher-model-free]
name = Publisher is model-free
type = forbidden
source_modules = src.evaluation.run_evidence_publisher
forbidden_modules =
    src.models
    src.causal
    src.perturbation

[importlinter:contract:capability-audit-outcome-blind]
name = Bridge capability audit is outcome-blind
type = forbidden
source_modules = src.evaluation.spatial_perturbation_registry
forbidden_modules =
    src.models
    src.causal
    src.perturbation
    src.data
    src.discovery
    src.evaluation.msi_inference
    src.evaluation.benchmark_evidence
    src.evaluation.spatial_perturbation_scoring
    src.evaluation.causal_metrics
    src.evaluation.cf_metrics
    src.evaluation.cross_sample_metrics
    src.evaluation.embedding_metrics
    src.evaluation.spatial_metrics
    src.evaluation.spatial_perturbation_runner
    src.evaluation.task_s_benchmark
    src.evaluation.task_c_benchmark
    src.evaluation.task_c_aggregation
    src.evaluation.task_c_tuning
    src.evaluation.task_c_null_controls
    src.evaluation.task_c_profile_input
    src.evaluation.task_c_data
    src.evaluation.task_c_acquisition
    src.evaluation.task_c_formal_export
    src.evaluation.task_c_predictions
    src.evaluation.task_c_runtime
    src.evaluation.task_c_rehearsal
    src.evaluation.task_c_method_registry
    src.evaluation.task_c_method_run
    src.evaluation.methods_protocol_outcome
    src.evaluation.methods_pilot
    src.evaluation.methods_causal_pilot

[importlinter:contract:split-model-free]
name = Bridge split is model-free
type = forbidden
source_modules = src.evaluation.spatial_perturbation_split
forbidden_modules =
    src.models
    src.discovery.evidence_policy

[importlinter:contract:scoring-policy-free]
name = Bridge scoring cannot decide claims
type = forbidden
source_modules = src.evaluation.spatial_perturbation_scoring
forbidden_modules =
    src.discovery.evidence_policy
    src.evaluation.run_evidence_publisher

[importlinter:contract:predictor-contract-model-free]
name = Prediction contract cannot fit models
type = forbidden
source_modules = src.evaluation.spatial_perturbation_predictor_contract
forbidden_modules =
    src.models
    src.causal
    src.perturbation

[importlinter:contract:crc-promotion-isolated]
name = CRC application code cannot decide methods promotion
type = forbidden
source_modules =
    src.discovery.from_scratch.crc_icb_inputs
    src.discovery.from_scratch.crc_icb_artifacts
forbidden_modules =
    src.discovery.evidence_policy
    src.evaluation.methods_protocol_v3
```

These seven Import Linter contracts plus one AST contract form the eight frozen
boundaries. Because `scripts/` is outside `root_package`, the eighth contract is
an AST test over the three new CLIs. It uses structural and call allowlists for
imports, the repository-root constant, argument parsing (`parse_args` or the
legacy `_arguments` spelling), and `main`; it rejects scientific calculations,
model construction, dataframe/array operations, and dynamic or import-time
scientific dependencies. Bounded JSON/YAML declaration reading remains under
`src/evaluation/`, not in a CLI helper.

The same AST test freezes the predictor-contract source with a recursively
canonicalized, Python 3.10/3.13-stable manifest.  Its exact keys comprise the
module declarations and every top-level function or class; each value is a
per-declaration digest over all semantic AST fields.  A deliberate contract
change therefore requires an explicit, locally reviewable manifest update,
while ordinary location metadata and empty version-only `type_params` are
ignored. `TypeIgnore` is the exception: both its tag and line binding remain
semantic so moving a suppression requires review.

- [ ] **Step 4: Run GREEN and commit**

```bash
lint-imports --config .importlinter
pytest tests/test_v3_import_boundaries.py -q -p no:cacheprovider
git add .importlinter pyproject.toml tests/test_v3_import_boundaries.py \
  src/evaluation/safe_declaration_reader.py \
  scripts/audit_spatial_perturbation_bridge.py \
  scripts/validate_spatial_perturbation_predictor.py \
  tests/test_spatial_perturbation_runner.py \
  docs/superpowers/plans/2026-08-28-hypersca-methods-v3-bridge.md
git commit -m "test: enforce methods v3 architecture boundaries"
```

## Task 11: Add a five-animal synthetic end-to-end bridge

**Files:**
- Create: `tests/integration/test_spatial_perturbation_bridge.py`
- Modify: `tests/property/test_run_evidence_properties.py`

- [ ] **Step 1: Build one deterministic fixture**

Use five animals, two sections per animal, mSafe and two perturbations, at least
three spatial blocks, and this frozen signal:

```python
own_delta = {"KO_A": 1.0, "KO_B": -0.8}
neighbor_delta = own_delta[perturbation] * np.exp(-rank / 8.0)
observed = control_expression + own_or_neighbor_delta + seeded_noise
```

The fixture meets production thresholds without bypass flags.

- [ ] **Step 2: Write RED scenarios**

```python
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
    assert run_fixture_scenario(scenario, tmp_path).status == expected
```

Add a positive CRC record and assert that missing bridge evidence still blocks
integrated promotion.

- [ ] **Step 3: Run RED, implement test adapters, and run GREEN**

```bash
pytest tests/integration/test_spatial_perturbation_bridge.py -q -p no:cacheprovider
```

Initial expected failure: scenario adapters are missing. Implement them only in
test fixtures; never add a synthetic shortcut to production. Even the positive
fixture remains `synthetic_audit_only` and cannot become completed scientific
evidence or authorize promotion.

- [ ] **Step 4: Run the complete v3 focused suite**

```bash
pytest tests/test_methods_protocol_outcome.py tests/test_methods_protocol_v3.py tests/discovery/test_evidence_policy_v3.py tests/test_spatial_perturbation_registry.py tests/test_spatial_perturbation_split.py tests/test_spatial_perturbation_neighbors.py tests/test_spatial_perturbation_scoring.py tests/test_spatial_perturbation_comparators.py tests/test_spatial_perturbation_runner.py tests/integration/test_spatial_perturbation_bridge.py tests/property/test_methods_protocol_v3_properties.py tests/property/test_spatial_perturbation_registry_properties.py tests/property/test_spatial_perturbation_split_properties.py tests/property/test_spatial_perturbation_neighbors_properties.py tests/property/test_spatial_perturbation_scoring_properties.py -q -p no:cacheprovider
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_spatial_perturbation_bridge.py tests/property/test_run_evidence_properties.py
git commit -m "test: cover methods v3 bridge end to end"
```

## Task 12: Run pre-freeze audits, freeze the foundation identity, and stop

**Files:**
- Create after audit: `configs/hypersca_methods_v3.yaml`
- Create after audit: `reports/methods_protocol_v3_preflight/bridge_capability.json`
- Create after audit: `reports/methods_protocol_v3_preflight/predictor_capability/`
  containing exactly `capability_record.json`, `resource_usage.json`,
  `run_manifest.json`, and `method_status.json`
- Create after audit: `reports/methods_protocol_v3_preflight/review.md`
- Modify: `tests/test_methods_protocol_v3.py`

- [ ] **Step 1: Verify the public asset root**

```bash
SPATIAL_PERTURB_ROOT=/home/a/Data/SpatialPerturbSeq/GSE274447
test -d "$SPATIAL_PERTURB_ROOT"
find "$SPATIAL_PERTURB_ROOT" -maxdepth 2 -type f -print
```

If absent, publish `external_cohort_missing` and stop. Do not substitute
simulation or create a confirmatory config.

- [ ] **Step 2: Run the metadata-only audit**

```bash
python scripts/audit_spatial_perturbation_bridge.py --registry configs/spatial_perturbation_bridge_candidates_v1.json --candidate-id gse274447 --asset-root "$SPATIAL_PERTURB_ROOT" --output reports/methods_protocol_v3_preflight/bridge_capability.json
```

Expected for GSE274447: three specimens, `pilot_audit_only`,
`confirmatory_capable: false`, and no effect or RMSE fields.

- [ ] **Step 3: Generate and verify the exact v3 YAML**

Build `MethodsProtocolV3` with `bridge_role="pilot_audit_only"` and the audited
asset-capability identity, `integrated_claim_enabled=false`, and no executable
bridge predictor identity. Serialize its canonical mapping to
`configs/hypersca_methods_v3.yaml`. Add a test that YAML equals
`protocol_to_mapping_v3()` exactly and cannot be changed to a runnable bridge
protocol without a new protocol version and identity.

- [ ] **Step 4: Audit the production predictor interface without outcomes**

```bash
python scripts/validate_spatial_perturbation_predictor.py --registry configs/spatial_perturbation_bridge_candidates_v1.json --protocol configs/hypersca_methods_v3.yaml --method-id hypersca --output reports/methods_protocol_v3_preflight/predictor_capability/
```

Expected for the current repository:
`status="method_adapter_not_executable"`, no adapter identity, no predictions,
and no metric artifacts. This is a capability result, not an algorithm failure.
Read that status from
`reports/methods_protocol_v3_preflight/predictor_capability/capability_record.json`;
the `--output` argument is the evidence-bundle directory, not a JSON filename.
Do not call `propagate_perturbation()` as a substitute and do not construct a
new predictor from HyperSCA-C output inside this task.

- [ ] **Step 5: Verify the stop decision**

Write `review.md` stating all of the following:

- v2.1 remains `pilot_failed_no_release`;
- the v3 bridge asset is pilot-only because it has three mice;
- the formal HyperSCA bridge predictor adapter is not executable;
- no real bridge pilot was run and no paired scientific collection exists;
- `integrated_claim_enabled` is false;
- a future adapter requires a separate preregistered design and protocol
  identity before it may see bridge outcomes.

- [ ] **Step 6: Run final verification**

```bash
pytest tests/test_methods_protocol.py tests/test_methods_protocol_v3.py tests/discovery/test_evidence_policy.py tests/discovery/test_evidence_policy_v3.py tests/test_run_evidence_publisher.py tests/test_run_evidence_collection.py tests/test_task_s_benchmark.py tests/test_methods_pilot.py tests/test_methods_causal_pilot.py -q -p no:cacheprovider
pytest tests -q -p no:cacheprovider
python3.10 -m py_compile src/evaluation/methods_protocol_v3.py src/evaluation/spatial_perturbation_registry.py src/evaluation/spatial_perturbation_split.py src/evaluation/spatial_perturbation_neighbors.py src/evaluation/spatial_perturbation_scoring.py src/evaluation/spatial_perturbation_comparators.py src/evaluation/spatial_perturbation_predictor_contract.py src/evaluation/spatial_perturbation_runner.py scripts/audit_spatial_perturbation_bridge.py scripts/validate_spatial_perturbation_predictor.py
lint-imports --config .importlinter
git diff --check
```

Record actual test counts; do not predict them.

- [ ] **Step 7: Commit tracked preflight evidence only**

```bash
git add configs/hypersca_methods_v3.yaml reports/methods_protocol_v3_preflight tests/test_methods_protocol_v3.py
git commit -m "chore: freeze methods v3 pilot identity"
```

No real pilot result tree is created in this phase.

## Final stop gate

After Task 12, stop. Do not run the three-animal real bridge pilot and do not
implement or run a confirmatory bridge release. A future real pilot first needs
a separately reviewed and preregistered production predictor adapter with an
outcome-blind capability record and a new protocol identity. The approved
backup concept is to evaluate an adapter that combines a frozen HyperSCA-C
cell-autonomous effect with a preregistered spatial propagation rule; it is
`backup_not_authorized` here and receives no code in this plan. Confirmatory
execution additionally requires an untouched external cohort, present before
that future freeze, with at least five total biological specimens and at least
two cohorts or studies. Until both prerequisites exist, the scientifically
correct outcome is no bridge evidence and no integrated spatial-causal
promotion.
