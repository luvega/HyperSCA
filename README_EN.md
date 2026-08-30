<p align="center">
  <img src="docs/Logo_high%20res.png" alt="HyperSCA Logo" width="280" />
</p>

[中文](README.md) | **[English](README_EN.md)**

[![CI](https://github.com/luvega/HyperSCA/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/luvega/HyperSCA/actions/workflows/ci.yml)

HyperSCA (Hyperbolic Spatiotemporal Causal Analysis) is a multi-step pipeline for joint single-cell and spatial-omics analysis. It first organizes cell states and their hierarchy with hyperbolic embeddings, then infers candidate causal graphs and evaluates candidate targets with counterfactual perturbation simulations. The outputs are intended to form testable mechanistic hypotheses and intervention candidates; they cannot by themselves establish a drug mechanism or treatment effect. HyperSCA supports scRNA-seq, spatial transcriptomics, and clinical or phenotypic stratification in research settings including tumor immunity, autoimmunity, chronic inflammation, infection, and tissue repair.

## Current Release Status

`v0.7.0` is a research release for auditable method comparison and the spatial-perturbation bridge. It continues the rule that conclusion strength must follow direct evidence and freezes comparison contracts, data isolation, coverage, abstention, and evidence-publication paths as replayable artifacts. HyperSCA remains alpha research software: passing software checks is not biological or clinical validation and is not a state-of-the-art performance claim.

- Methods v3 freezes the comparison roles, statistical units, resource limits, coverage, abstention, and conservative promotion conditions for Tasks C/S/D.
- The spatial-perturbation bridge uses whole-animal-isolated splits and connects candidate registration, neighborhoods, effect scoring, simple comparators, and evidence publication in a replayable workflow.
- Run evidence must pass identity, input, artifact, and collection checks; failure or insufficient-data states cannot be rewritten as success by a summary artifact.
- Import-architecture and property tests constrain dependency directions among CLI, scientific, and evidence modules, reducing opportunities to bypass frozen contracts.

The real GSE274447 spatial-perturbation pilot was not run because the registered external cohort root was absent from the execution environment. v0.7.0 publishes only this stop-gate fact; it does not fabricate pilot outcomes, predictor-capability artifacts, or promotion claims.

- [v0.7.0 release notes](docs/releases/v0.7.0.md)
- [Historical v0.6.0 release notes](docs/releases/v0.6.0.md)
- [HyperSCA progress and research-landscape report](reports/research/hypersca_causal_spatial_drug_landscape_20260810.md)
- [Bear supporting and opposing evidence synthesis](reports/research/bear_hypersca_spatial_causal_20260810/report.md)
- [Current method comparison table](reports/research/bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv)
- [Innovation claim and evidence register](reports/research/bear_hypersca_spatial_causal_20260810/innovation_claim_register.tsv)
- [Pre-registered Task C/S/D benchmark contract](docs/research/benchmark_contract_v1.md)
- [Project terminology and writing guide](docs/style/plain_language_terminology.md), which explains how reader-friendly wording preserves scientific evidence boundaries.

Before running a new method comparison, validate that the pre-registered contract is complete:

```bash
python scripts/validate_benchmark_contract.py
```

The contract fixes Tasks C/S/D, five random seeds, data splits and feature rules, equal hyperparameter-trial limits, simple baselines, null controls, and conservative evidence-promotion criteria. It also requires reporting coverage and abstention. A valid contract does not mean that a comparison has been completed; every task remains `not_evaluated` until external holdouts and mandatory simple comparators have run.

Task C includes a reproducible mean-difference baseline compatible with the three-array NPZ format and model-call convention published by CausalBench. See [Task C mean-difference baseline v1](docs/research/task_c_mean_difference_baseline_v1.md).

Task S provides paired `own_only` and `fixed_distance_decay` spatial baselines. They share the same upstream own-effect predictions and report own and neighbor endpoints separately. See [Task S simple spatial baselines v1](docs/research/task_s_simple_baselines_v1.md).

## Project and Method Overview

The complete HyperSCA research workflow has six consecutive stages that can be adapted to the cohort and research question:

- Phase D0 (Data Onboarding): normalize four projects into a shared schema and validate fields.
- Stage 1 (Embedding): learn cell-state representations on Lorentz/Poincare hyperbolic manifolds.
- Stage 2 (Causal): discover candidate causal structure and infer signaling flow on disentangled latent variables.
- Stage 3 (Counterfactual): simulate gene perturbation and spatial propagation in latent space, rank targets, and filter likely false positives.
- Stage 4 (Dynamic Intervention): evaluate time-dependent propagation and multi-target combinations under PK/PD constraints, with roundtrip updates after experimental results.
- Stage 5 (Behavior Grammar / Virtual Tissue): translate target-discovery and Stage 4 evidence into readable cell-behavior rules and run a lightweight virtual-tissue simulation. This is an optional sidecar and does not replace Stages 1–4.

## Design Overview

![HyperSCA v0.7.0 design overview: from multi-omics inputs to evidence-gated target discovery](docs/hypersca_design_overview_v0.7.0.png)

## Benchmark Evaluation and Module Selection

Benchmark evaluation is a module-selection sidecar used to compare candidate spatial annotation, spatial deconvolution, hyperbolic embedding, and downstream target-discovery methods before they can enter the main analysis. It does not replace Phases D0–5 and does not directly change the active target ranking. Candidate modules advance only when they show a non-zero `target rank delta`, improved `target enrichment`, or a reviewable biological gain in spatial niches.

The conservative conclusions of the 2026-06-22 benchmark snapshot remain:

- The primary comparison includes only two internally trained v3 branches: `hvae_hierarchy_spatial_v3_product` and `hvae_hierarchy_spatial_v3_product__without_radial_depth_loss`.
- SCimilarity is an external pretrained appendix reference, not a primary ranking competitor.
- Both v3 branches remain `audit_only_no_promotion`: `target rank delta` is still 0, `target enrichment` has not improved, and prototype/radial hierarchy supervision remains near chance.
- Full-scale VisiumHD cell2location passed validation of a 545,913-row abundance output; segmentation-based RCTD is the near-single-cell-resolution spatial comparator.
- Xenium remains panel-aware; targeted-panel data do not use whole-transcriptome RCTD/cell2location assumptions.

The current audit material is retained as compact reports and figures:

- [Benchmark progress report](docs/research/hypersca_benchmark_progress_20260622.md)
- [Benchmark JSON snapshot](docs/research/hypersca_benchmark_progress_20260622.json)
- [Project progress inventory](docs/research/hypersca_project_progress_inventory_20260622.md)
- [GitHub submission notes](docs/github_submission_20260622.md)
- [Current workflow figure](docs/research/figures/hypersca_current_pipeline_flowchart_20260622.png)
- [Two-candidate downstream summary figure](docs/research/figures/hypersca_two_candidate_downstream_summary_20260622.png)

## Core Methods

### 1) Cell-state representation in hyperbolic space

- Key modules: `src/models/hyperbolic/lorentz.py`, `src/models/hyperbolic/poincare.py`, `src/models/hyperbolic/wrapped_normal.py`, `src/models/hyperbolic/hvae.py`
- Purpose: preserve cell-state hierarchy and relative structure with less geometric distortion than a Euclidean representation.

### 2) Candidate causal relations and signaling flow

- Key modules: `src/causal/disentangle.py`, `src/causal/cmi_pruning.py`, `src/causal/causal_graph.py`, `src/causal/signaling_flow.py`
- Method: disentangled `z_int/z_ext` variables, PC conditional-independence tests, bootstrap stability, DoWhy structural checks, and layered L-R-TF-Target signaling flow.

### 3) Counterfactual perturbation and spatial propagation

- Key modules: `src/perturbation/latent_arithmetic.py`, `src/perturbation/spatial_propagation.py`, `src/perturbation/diffusion_cf.py`, `src/perturbation/target_ranking.py`
- Method: latent-space virtual knockout, causal-graph-constrained diffusion, spatial-gradient decay fitting, and intervention-oriented target ranking.

### 4) Spatial niches and cross-sample stratification

- Key module: `src/evaluation/cross_sample_metrics.py`
- Method: niche clustering, cross-sample edge consistency, and clinical or phenotypic stratification differences included in the final evidence matrix.

### 5) Composable target discovery

- Entry point: `scripts/run_target_discovery.py` is a thin CLI that parses arguments, constructs `TargetDiscoveryConfig`, and starts the pipeline.
- Core package: `src/discovery/target_discovery/`
  - `config.py`, `pipeline.py`, `stage.py`, and `artifacts.py` define configuration, orchestration, stage contracts, and the run manifest.
  - `loaders.py`, `candidates.py`, `expression.py`, and `spatial.py` build lightweight data inputs.
  - `geometry.py`, `causal_stage.py`, `perturbation_stage.py`, `scoring.py`, `niche.py`, `reporting.py`, and `figures.py` implement geometry comparison, Stage 2/3 wrappers, evidence ranking, niche mapping, reports, and figures.
- Output root: `results/discovery/target_discovery/<run_id>/`, partitioned into `candidates/`, `expression/`, `spatial/`, `geometry/{mode}/`, `causal/{mode}/`, `perturbation/{mode}/`, `scoring/`, `niche/`, `reports/`, and `figures/`, plus `manifest.json` and `reports/migration_notes.md`.

### 6) Behavior grammar and virtual-tissue simulation

- Entry point: `scripts/run_behavior_grammar_simulation.py`
- Core package: `src/behavior_grammar/`
  - `rules.py` defines `BehaviorRule`, `SignalDictionary`, `BehaviorDictionary`, `RuleSet`, and Hill, linear, and step responses.
  - `rule_builder.py` builds data-driven rules from `results/discovery/target_discovery/<run_id>/manifest.json`, score tables, causal edges, niche mappings, and expression matrices.
  - `simulation.py` runs a deterministic toy virtual-tissue simulation and emits QoI sensitivity and combination-intervention comparisons.
  - `pipeline.py` reuses the run-scoped artifact manifest to write rules, trajectories, summaries, sensitivity tables, and animations.
- Output root: `results/behavior_grammar/<run_id>/`, including `rules/rules.json`, `rules/rules.md`, `simulation/population_trajectory.csv`, `simulation/simulation_summary.json`, `simulation/qoi_sensitivity.csv`, and `figures/population_trajectories.png`.

## Example Data

Common example inputs are external local data roots and are not version-controlled. The paths below are de-identified placeholders:

- `<PATH_TO_scRNA_REFERENCE>`
  - Example files: `*-NormalizedCounts.tsv`, `*-DE_result.tsv`
  - Use: cluster-level expression matrices, candidate differential-gene pools, and cell-state priors.
- `<PATH_TO_SPATIAL_OMICS>`
  - Example files: `STmetadata_*.csv`, `spot_annotations.*`
  - Use: spatial deconvolution, cell-colocation adjacency, propagation gradients, and niche structure.
- `<PATH_TO_CLINICAL_OR_PHENOTYPE>`
  - Example files: `sample_clinical_mapping.csv`, `group_labels.csv`
  - Use: clinical or phenotypic stratification such as immune subtype, disease stage, or treatment response, plus cross-sample comparison.

Example standardized outputs:

- `results/integration/schema/sample_table.csv`
- `results/integration/schema/entity_table.csv`
- `results/integration/schema/feature_table.csv`
- `results/integration/schema/measure_table.csv`

## Installation

### 1) Create the Conda environment

```bash
conda create -n hypersca python=3.10 -y
conda activate hypersca
```

### 2) Install the core CPU environment

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

`pyproject.toml` enables editable installation while retaining the existing `src.*` import paths. Core CPU dependencies are recorded in `requirements-core.txt`. For the extended single-cell, spatial, causal, and notebook stack, run:

```bash
python -m pip install -r requirements.txt
```

For GPU use, install the PyTorch build matching the local CUDA runtime and then use the [PyG compatibility table](https://data.pyg.org/whl/) for matching compiled extensions. Extensions are recorded in `requirements-gpu.txt` and are excluded from CPU and continuous-integration dependencies. Historical perturbation baselines are optional; the core pipeline does not depend on `scgen`:

```bash
pip install -r requirements-optional-baselines.txt
```

### 3) Validate the environment

```bash
python scripts/validate_env.py --profile core-cpu
pytest tests -q
```

Validation profiles correspond to dependency boundaries:

- `core-cpu`: CI and accelerator-free development; CUDA and compiled PyG extensions are not checked.
- `gpu`: core dependencies plus CUDA and compiled PyG extensions.
- `full`: complete research stack plus GPU/PyG; this remains the compatibility default when `--profile` is omitted.

`scgen` is checked only as an optional historical baseline in the `full` profile. A missing or incompatible `scvi-tools` import produces a warning and does not independently block environment validation.

## Quick Start

### Multi-omics integration example (recommended)

Six interactive notebooks demonstrate the HyperSCA multi-omics workflow and its main figures:

- `notebooks/example_multiomics_integration/README.md`
- `00_data_landscape` → `01_hyperbolic_vs_euclidean` → `02_multiscale_niche` → `03_causal_network` → `04_target_discovery` → `05_summary`

Core comparison snapshot:

| Metric | scRNA-only + Euclidean | Multi-omics + Hyperbolic | Change |
|--------|------------------------|--------------------------|--------|
| Niche Silhouette | 0.417 | **0.710** | **+70%** |
| Hierarchy Correlation | −0.569 | **+1.000** | reversal to perfect |
| Evidence dimensions | 3 | **5** (+spatial, +niche) | +2 independent dimensions |

Scale: 485K spatial locations across three spatial platforms plus three scRNA-seq cohorts. Target discovery is data-driven and uses no preselected anchor target.

### Step-by-step scCRC_ICB example

For a main-workflow walkthrough using scRNA-seq data:

- `notebooks/example_sccrc_icb_step_by_step/README.md`
- `notebooks/example_sccrc_icb_step_by_step/00_environment_and_data_check.ipynb` through `05_step4_dynamic_intervention_and_summary.ipynb`

### A. Build the canonical schema

```bash
python scripts/build_canonical_schema.py
```

### A0. Onboard multiple cohorts into `/data`

The CLI retains the historical argument names `icb/neu/st/ifng`, but each may map to a disease-appropriate external data root.

```bash
python scripts/run_data_onboarding.py \
  --icb-root <PATH_TO_COHORT_A> \
  --neu-root <PATH_TO_COHORT_B> \
  --st-root <PATH_TO_SPATIAL_OMICS> \
  --ifng-root <PATH_TO_COHORT_D>
```

### B. Run Stage 1: hyperbolic embedding

```bash
python scripts/run_step1.py \
  --data-dir data/ST/<YOUR_SPATIAL_PROJECT> \
  --modality visium \
  --output-dir results/step1
```

### C. Run Stage 2: spatial causal inference

```bash
python scripts/run_step2.py \
  --input-dir results/step1 \
  --output-dir results/step2
```

After Stage 2, run the additive stability audit without overwriting existing results. The default `--n-null-controls 0` preserves historical behavior; frequency or network-structure null controls require explicit count, modes, and seed:

```bash
python scripts/run_causal_stability_audit.py \
  --step2-dir results/step2 \
  --n-null-controls 100 \
  --null-modes matrix_permutation,node_label_shuffle,outgoing_weight_permutation \
  --random-seed 42
```

Outputs include `null_control_manifest.json` and content fingerprints. This audit permutes the saved bootstrap-frequency matrix; it does not refit after permuting raw cells, treatments, coordinates, or priors. Its results are supplemental evidence for candidate causal relations, not proof of interventional causal effects.

### D. Run Stage 3: counterfactual perturbation

```bash
python scripts/run_step3.py \
  --input-step1 results/step1 \
  --input-step2 results/step2 \
  --output-dir results/step3
```

### E. Run target discovery and retain network hubs

```bash
python scripts/run_target_discovery.py \
  --run-id demo_target_discovery \
  --max-perturb 10 \
  --geometry-k 4 \
  --geometry-blend 0.30 \
  --platform all \
  --score-profile evidence_gated \
  --skip-figures
```

The default output is `results/discovery/target_discovery/<run_id>/`. Historical precomputed discovery outputs remain under `results/integration/discovery/` for notebooks and earlier README figures.

`evidence_gated` is the default and the only policy permitted for interpretation of the primary ranking. It stratifies first by the number of independent differential-expression sources, then compares directional consistency, significance, and effect size. `final_score` is a display order, not a weighted evidence score. Causal graphs, spatial-propagation proxies, and mechanism priors are recorded as audit columns but do not alter the ranking. Each run writes `scoring/ranking_policy.json` and `scoring/module_admission.csv`. `legacy_full` exists only to reproduce the historical weighted ranking and cannot raise evidence strength. The CLI does not accept manually selected genes or target seeds.

### F. Run dynamic intervention (Stage 4) and roundtrip updates

```bash
python scripts/run_step4.py --with-roundtrip \
  --experiment-file data/metadata/experiment_roundtrip.csv
```

### G. Run the behavior-grammar sidecar (Stage 5)

To inspect behavior-rule outputs without a real target-discovery manifest, run the demo:

```bash
python scripts/run_behavior_grammar_simulation.py \
  --demo \
  --run-id demo_behavior_grammar \
  --time-steps 8
```

For a real target-discovery run, provide its manifest:

```bash
python scripts/run_behavior_grammar_simulation.py \
  --discovery-manifest results/discovery/target_discovery/<run_id>/manifest.json \
  --step4-dir results/step4 \
  --run-id <run_id>
```

The sidecar reads the target-discovery manifest and optional Stage 4 context, then emits readable rules, virtual-tissue trajectories, QoI sensitivity, and animations. It does not modify the Stage 1–4 command or output contracts.

### H. Generate CNS-style Stage 1/2/3 figures

```bash
python scripts/generate_step1_figures.py
python scripts/generate_step2_figures.py
python scripts/generate_step3_figures.py
```

## Primary Outputs

- Canonical data schema and metadata: `data/metadata/`, `results/integration/schema/`
- Stage 1: `results/step1/` (`adata_embedded.h5ad`, `embedding_benchmark.json`)
- Stage 2: `results/step2/` (candidate causal graphs, stability metrics, and simple-comparator results)
- Stage 3: `results/step3/` (perturbation results and targets/combinations after likely false-positive filtering)
- Target discovery: `results/discovery/target_discovery/<run_id>/` (run manifest, candidate pool, geometry comparison, Stage 2/3 wrapper outputs, score tables, niche mappings, reports, and migration notes)
- Historical precomputed notebook reports: `results/integration/discovery/`
- Stage 4: `results/step4/` (`pkpd_summary.json`, `combination_ranking.csv`, `roundtrip_update_report.json`)
- Behavior grammar: `results/behavior_grammar/<run_id>/` (readable rules, simulation summary, QoI sensitivity, and virtual-tissue trajectories)
- CNS-style figures: `results/figures/step1/`, `results/figures/step2/`, `results/figures/step3/`

## Repository Layout

See [docs/project_inventory.md](docs/project_inventory.md) for local directory boundaries, tracked-file rules, validation code, output directories, update history, and current project status.

## Tests

```bash
pytest tests -q -p no:cacheprovider
pytest tests/discovery -q
pytest tests/behavior_grammar -q
```

## License

MIT License.
