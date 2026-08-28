# HyperSCA Methods v3 Spatial-Perturbation Bridge Design

**Date:** 2026-08-28

**Status:** Approved design; implementation not started

**Target protocol:** `hypersca-methods-v3.0`

## 1. Purpose

HyperSCA methods protocol v2.1 evaluated two different parts of the system on
two different benchmarks:

- OSTA evaluated preservation of spatial neighbourhood structure;
- CausalBench evaluated recovery of intracellular directed gene relations from
  dissociated Perturb-seq data.

Those results cannot be concatenated into evidence for an integrated spatial-
causal effect. CausalBench contains real interventions and is well matched to
the intracellular causal-network module, but it does not preserve tissue
coordinates, spatial neighbourhoods, or non-cell-autonomous responses.

Version 3 therefore introduces a third, spatial-perturbation bridge benchmark.
The bridge must observe perturbation identity, spatial coordinates, expression,
and biological replicates in the same experiment. Until such a bridge is
confirmatory-capable and passes its frozen test, the strongest permitted claim
is that the spatial-representation and intracellular-causal modules received
separate support.

## 2. v2.1 Closure

The existing v2.1 pilot remains an immutable no-release audit. Its 18 published
run bundles are not modified, relabelled, or reused by v3.

- pilot summary SHA-256:
  `3fe9e90443f82a911fe02314a540cd8e3383ee016cff9c3dbb46b802490d694c`
- protocol identity:
  `caa2f9a4aed7e474c123cb815435f65df5011387a4be1181d324a635b1a01613`
- outcome: `pilot_failed_no_release`
- release authorised: false

The audit reason records all of the following:

1. OSTA contained one biological sample per platform stratum, so its apparent
   overall benefit could not establish cross-platform generalisation.
2. The CausalBench mean-difference comparator returned no eligible relations
   and therefore was not a valid confirmatory comparator.
3. The OSTA hierarchy attribution interval crossed zero.
4. Both CausalBench direction-level attribution results failed the frozen
   promotion condition.
5. Protocol v2.1 had already used its one allowed redesign and cannot silently
   redesign again.

The v2.1 closure is stored as a Git-anchored JSON record plus a human-readable
review. Git is the independent anchor; the pilot output directory alone is not
treated as an external trust root.

## 3. Evidence Architecture and Claims

Version 3 has three preregistered confirmatory benchmark families and one
application-only evaluation:

| Family | Benchmark | Primary metric | Maximum family-specific claim |
|---|---|---|---|
| Spatial representation | OSTA | `neighborhood_preservation_at_k`, K=15 | Cross-sample spatial-representation preservation gain |
| Intracellular causal recovery | CausalBench | `directed_edge_average_precision` | Directed relation recovery gain under single-cell intervention |
| Spatial-perturbation bridge | Qualified spatial perturbation data | Standardised `neighbor_effect_rmse` | Spatially indexed non-cell-autonomous response recovery gain |
| Application | CRC | Frozen application endpoints | Application-only stability and association |

The formal CausalBench claim name is
`intracellular_interventional_causal_recovery`. It is not called a spatial
causal benchmark.

The integrated claim is an intersection:

```text
OSTA passes
AND CausalBench passes
AND the spatial-perturbation bridge is confirmatory-capable and passes
```

Only this conjunction permits `integrated_spatial_causal_promoted`. OSTA and
CausalBench passing without the bridge permits two separate module-level
claims, not an integrated claim. A positive CRC result cannot rescue a failed
or unavailable public benchmark.

### 3.1 Multiplicity

The three benchmark families are preregistered scientific questions. No Holm,
Bonferroni, or FDR adjustment is applied across them. Each family has exactly
one confirmatory primary metric and a one-sided paired 95% confidence interval.
The manuscript must state that the family-specific intervals are nominal and
not adjusted across benchmark families.

The integrated claim is an intersection-union test: all three family gates
must pass. It does not use an additional p-value and does not select whichever
family performed best. Statements based on "at least one significant primary
metric" are forbidden.

## 4. Staged Bridge Admission

The bridge enters v3 through a staged admission process.

1. `candidate_registered`
2. `capability_passed` or `capability_failed`
3. `pilot_audit_only`
4. `external_cohort_verified`
5. `confirmatory_frozen`
6. `completed_positive`, `completed_negative`, or `operational_failure`

Capability audit is outcome-blind. It may inspect file identity, licence,
animal and section identifiers, coordinates, barcode quality, cell types,
control labels, counts, spatial coverage, and executable output contracts. It
must not calculate perturbation effects, primary metrics, or comparator ranks.

The initial candidate is Spatial Perturb-Seq GSE274447, which contains three
chips from three mice and internally multiplexed mSafe controls. It is eligible
for capability audit and an audit-only pilot. Three mice are not sufficient for
confirmatory promotion.

Confirmatory admission requires:

- at least five independent biological specimens;
- at least two independent batches, cohorts, or studies;
- an untouched external outcome cohort;
- spatial coordinates, expression, perturbation identity, and internal safe
  controls;
- a frozen cell-type mapping and common measurable gene contract;
- executable confirmatory and attribution comparators.

If these conditions are not met, the terminal status is
`bridge_pilot_only` or a more specific capability failure. Thresholds are not
lowered after examining response effects.

## 5. Statistical Units

The top-level independent unit is the biological specimen or animal. Cells,
spatial spots, spatial blocks, perturbations, and sections are nested units.

```text
animal
  -> section
    -> spatial block
      -> perturbation
        -> perturbed source and non-perturbed neighbour cells
```

For GSE274447, the two sections from Mouse 1 are aggregated within Mouse 1 and
do not create two independent replicates. Model seeds are averaged within an
animal-level unit and never count as biological replication.

The within-animal estimand is indexed by:

```text
animal x perturbation x neighbour cell type x preregistered distance band
```

The final confidence interval resamples animals at the top level and spatial
blocks within animal. Cell-level or section-level confidence intervals are
forbidden.

## 6. Data Splits

### 6.1 Audit-only three-animal pilot

The pilot uses preregistered leave-one-animal-out folds:

| Fold | Development animals | Audit evaluation animal |
|---|---|---|
| 1 | Mouse 2, Mouse 3 | Mouse 1 |
| 2 | Mouse 1, Mouse 3 | Mouse 2 |
| 3 | Mouse 1, Mouse 2 | Mouse 3 |

All sections, cells, and blocks from one animal remain in one partition.
Bridge-specific architecture and primary hyperparameters are frozen before the
pilot. Non-adjacent blocks inside development animals may be used for early
stopping or numerical calibration. An audit evaluation animal cannot determine
genes, standardisation, distance scales, filtering, or model selection.

The three-animal pilot can expose implementation, coverage, and modelling
failures, but cannot produce a promotion confidence interval.

### 6.2 Confirmatory split

Confirmatory evaluation uses:

```text
development cohort
  -> train animals
  -> tune animals
external confirmatory cohort
  -> completely held-out animals
```

Animals whose outcomes were examined during the pilot may later be development
data, but can never become confirmatory holdout data. The external cohort is
opened for outcomes only after the protocol, model, preprocessing, thresholds,
comparators, and code identities are frozen.

External mSafe observations are used only to construct the scoring target. They
cannot refit or calibrate the model. Missing perturbations follow frozen
coverage and abstention rules and are never removed after observing their
effects.

The primary analysis tests perturbations represented in both development and
independent evaluation animals. Generalisation to completely unseen
perturbations is a named secondary endpoint and is not mixed into the primary
metric.

## 7. Spatial Neighbour Contract

Neighbourhoods are defined by within-section spatial neighbour rank rather than
a learned or platform-specific physical radius.

| Band | Rank | Role |
|---|---:|---|
| proximal | 1-5 | primary |
| local | 6-15 | primary |
| transition | 16-30 | secondary |
| distal | 31-60 | negative-control secondary |
| own | source cell | separate secondary endpoint |

Proximal and local bands contribute equal weight to the primary metric.
Transition and distal bands cannot improve the primary result.

Neighbour construction must:

- search only within one section;
- exclude the source cell;
- exclude every barcode-positive neighbour;
- count a neighbour linked to multiple same-perturbation sources once, using
  its minimum rank;
- exclude a primary-band neighbour exposed to different perturbation sources
  and record the exclusion as contamination;
- reject non-finite coordinates, duplicate cell identities, and unsafe
  cross-boundary edges;
- use the identical algorithm for mSafe sources.

mSafe matching is within animal and is stratified by section, spatial block,
source-cell type, neighbour-cell type, and distance band. Controls are never
borrowed from a different animal.

## 8. Eligibility and Coverage

Each `animal x perturbation` primary unit requires:

- at least 20 unique, confidently single-barcode source cells;
- source cells in at least three non-adjacent spatial blocks;
- at least 20 unique mSafe source cells in at least three blocks;
- at least 50 unique non-perturbed neighbours in each primary distance band;
- at least 50 corresponding matched mSafe neighbours in each primary band;
- a target gene in the frozen measurable-gene set.

Each `animal x perturbation x neighbour-cell-type x band` unit requires at
least 30 unique neighbours from at least three blocks, with the same minimum for
matched mSafe neighbours. An ineligible cell-type unit abstains rather than
receiving an imputed zero.

At least 80% of preregistered perturbations per animal and at least 80% of all
frozen primary units must be scoreable. Overall abstention must not exceed 20%.
All methods are evaluated on the same frozen paired units. Method-specific
deletion is forbidden.

## 9. Primary Estimand and Metric

Expression is standardised using means and standard deviations derived only
from training-animal control cells. The observed neighbour response is:

```text
mean standardised expression among eligible non-perturbed neighbours
minus
mean standardised expression among matched mSafe neighbours
```

This is calculated for each animal, perturbation, neighbour-cell type, band,
and frozen gene. The confirmatory metric is:

```text
neighbor_effect_rmse = sqrt(mean((predicted_delta - observed_delta)^2))
```

Aggregation proceeds in this order:

```text
genes -> distance band -> neighbour cell type -> perturbation -> animal
```

Each level is aggregated before the next level, and proximal/local bands are
equally weighted. Large sections, abundant cell types, and high-cell-count
perturbations cannot dominate by raw cell count.

Secondary endpoints are:

- `own_effect_rmse`;
- `neighbor_effect_pcc`;
- `distance_decay_calibration_error`;
- `effect_sign_accuracy`;
- coverage and abstention;
- perturbation, cell-type, distance-band, and animal heterogeneity.

Own-cell success cannot rescue neighbour-effect failure.

## 10. Bridge Comparators and Promotion

The bridge uses one primary metric and two necessary paired comparisons.

### 10.1 Matched Euclidean confirmatory comparator

`matched_euclidean_spatial_causal` uses the same inputs, frozen genes, spatial
graph, propagation operator, training epochs, random seeds, and matched
parameter and training budgets as HyperSCA. Its latent geometry is Euclidean.

### 10.2 Spatial attribution comparator

`hypersca_own_only` keeps HyperSCA representation, causal graph, and own-cell
response prediction, while setting every non-perturbed-neighbour prediction to
zero. It is not refitted as an alternative propagation model.

For each animal, positive improvement is defined as:

```text
comparator_neighbor_effect_rmse - hypersca_neighbor_effect_rmse
```

The bridge passes only if the paired one-sided 95% confidence-interval lower
bound is greater than zero for both comparators. This is an intersection-union
gate. Its claim-level p-value is the larger of the two component p-values; the
two comparisons are not multiplicity-adjusted against each other.

Secondary comparators include `fixed_distance_decay`,
`without_hierarchy_loss`, and any external method that passes a frozen
capability audit. Coordinate and neighbour-identity permutations are null
controls, not competitive methods.

The three-animal pilot reports point estimates, animal-level direction, unit
coverage, and operational failures. It never evaluates a promotion CI.

## 11. Components and Dependency Direction

The bridge is implemented as small modules around the existing evidence
infrastructure:

```text
BridgeCandidateRegistry
  -> BridgeCapabilityAudit
    -> BridgeSplitManifest
      -> BridgeRunAdapter
        -> RunEvidencePublisher
          -> BridgePairedCollection
            -> EvidencePolicy
```

Responsibilities:

- registry: immutable asset, licence, identity, and independence metadata;
- capability audit: outcome-blind structural and count checks;
- split manifest: immutable animals, units, genes, bands, and abstentions;
- run adapter: execute one frozen method without making promotion decisions;
- publisher: validate and atomically publish evidence bundles;
- paired collection: prove exact comparable units and identities;
- policy: decide family-specific and integrated claims.

### 11.1 Import Linter contracts

1. EvidencePolicy cannot import models, runners, or data readers.
2. RunEvidencePublisher cannot import `src.models`, `src.causal`, or
   `src.perturbation`.
3. BridgeCapabilityAudit cannot import models, scoring, or response metrics.
4. BridgeSplit cannot import models or EvidencePolicy.
5. BridgeScoring cannot import publisher or promotion policy.
6. BridgeRunAdapter may depend on models and publisher but not CRC.
7. CRC application code cannot import promotion transition interfaces.
8. CLI scripts contain argument parsing and error translation only; scientific
   logic remains under `src/`.

## 12. Failure Semantics

The following terminal or blocking states remain distinct:

- `insufficient_biological_replicates`;
- `insufficient_safe_control`;
- `insufficient_perturbation_coverage`;
- `spatial_contamination_exceeds_limit`;
- `external_cohort_missing`;
- `bridge_pilot_only`;
- `completed_negative`;
- `operational_failure`.

Capability failure does not mean algorithm failure. Operational failure cannot
be represented by zero effect or zero metric. Scientific negative results are
recorded only after all capability, identity, coverage, and execution gates
pass. No OSTA, CausalBench, or CRC result can change these states.

## 13. Testing Strategy

### 13.1 Hypothesis properties

Property tests cover:

- animal-level partition isolation and seed-independent splits;
- row-order invariance of splits and neighbour sets;
- mutually exclusive neighbour bands;
- no cross-section edges, barcode-positive neighbours, or duplicate counting;
- deterministic contamination exclusion;
- equal proximal/local weighting regardless of cell abundance;
- RMSE zero for perfect predictions;
- failure of own-only under true non-zero neighbour effects;
- invariance to train-standardised global expression scaling;
- no artificial increase in biological sample count or CI precision after cell
  or section duplication;
- common-unit enforcement across methods;
- exact immutable state transitions and protocol identities;
- inability of pilot or CRC evidence to trigger promotion;
- integrated promotion if and only if all three family gates pass.

### 13.2 Integration fixture

A small synthetic fixture contains five animals, multiple sections per animal,
mSafe plus two perturbations, a cell-autonomous response, and a known
distance-decaying neighbour response. It exercises:

- one valid bridge pass;
- failure versus matched Euclidean;
- failure versus own-only;
- insufficient coverage;
- held-out-animal leakage;
- positive CRC evidence with promotion still blocked.

### 13.3 Evidence integrity

Tests prove that changes to data, split, model config, code, comparator, or
statistical-unit identity change the run identity. v2.1 and v3 bundles cannot
be reused across protocols. Pilot and confirmatory roles remain distinct.
Publisher reuse revalidates predictions, metrics, status, resources, and paired
collection contents.

## 14. Delivery Sequence

1. Commit the v2.1 no-release outcome record and review.
2. Implement the bridge candidate registry and outcome-blind capability audit.
3. Implement the animal-level split and neighbour contracts with Import Linter
   and Hypothesis tests first.
4. Implement frozen standardisation and neighbour-effect RMSE.
5. Implement matched Euclidean and own-only bridge comparators.
6. Integrate RunEvidencePublisher, paired collections, and EvidencePolicy.
7. Run synthetic smoke tests.
8. Before freezing v3, run outcome-blind asset and capability audits for OSTA,
   the CausalBench comparator, and every bridge candidate.
9. Freeze the exact v3 protocol identity. If no untouched external bridge
   cohort is already capable at this point, freeze the bridge role as
   `pilot_audit_only` and permanently disable the integrated claim for v3.
10. Run the three-animal Spatial Perturb-Seq audit-only pilot under that frozen
    identity. Pilot results cannot alter the protocol or admit a later cohort.
11. After interfaces are frozen, split large benchmark runners where needed.
12. Run eligible five-seed module releases once. A bridge confirmatory release
    is run only if its external cohort was registered before v3 freeze. No
    post-pilot model or protocol redesign is permitted in this study.

## 15. Expected Scientific Outcomes

The permitted interpretations are deliberately narrow:

- OSTA alone supports spatial-representation preservation.
- CausalBench alone supports intracellular intervention-based relation
  recovery.
- A qualified bridge alone supports spatial neighbour-response recovery.
- OSTA plus CausalBench without the bridge supports two separately validated
  modules.
- All three passing supports an integrated spatial-causal gain.
- CRC remains application-only under every combination.

If the bridge never becomes confirmatory-capable, the methods paper reports the
missing bridge as an explicit validation limitation rather than replacing it
with CausalBench, CRC, simulations, or favourable secondary analyses.

## 16. Primary Data References

- CausalBench: <https://www.nature.com/articles/s42003-025-07764-y>
- Spatial Perturb-Seq: <https://www.nature.com/articles/s41467-026-69677-6>
- GSE274447: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274447>
- Perturb-map: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8992964/>
