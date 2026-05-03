# Target Discovery Pipeline Redesign Spec

Date: 2026-05-03
Status: approved design draft
Scope: restructure `scripts/run_target_discovery.py` into a reusable target discovery pipeline.

## Goal

Refactor the current monolithic target discovery research script into a formal pipeline package that can be called from CLI, notebooks, and future orchestration code. The redesign prioritizes clear stage boundaries, structured artifacts, reproducible run records, and testable components.

This refactor may change CLI parameters and output paths. Any changes must be documented in a migration note so existing users can map old commands and outputs to the new layout.

## Current Problems

`scripts/run_target_discovery.py` currently mixes these responsibilities in one file:

- External data paths and biological constants.
- Candidate gene aggregation from Neu, ICB, and IFNG sources.
- Cluster-level expression assembly.
- Spatial co-localization adjacency construction.
- Hyperbolic and Euclidean geometry comparison.
- Step2 causal discovery invocation.
- Step3 perturbation screening invocation.
- Evidence scoring, hub retention, and combination extraction.
- Unified niche construction and target-to-niche mapping.
- Figure generation and markdown reporting.
- Direct writes to global output directories.

The file is over 2,000 lines, has many implicit global dependencies, and is hard to test without running heavy downstream stages.

## Chosen Approach

Use a redesigned discovery pipeline rather than a purely mechanical file split.

The new package will live under:

```text
src/discovery/target_discovery/
```

The script:

```text
scripts/run_target_discovery.py
```

will become a thin CLI entrypoint that parses command-line arguments, builds `TargetDiscoveryConfig`, runs `TargetDiscoveryPipeline`, and prints the final output location.

## Non-Goals

- Do not rewrite the Step2 causal algorithm.
- Do not rewrite the Step3 perturbation algorithm.
- Do not require output path compatibility with the old script.
- Do not move ignored data or generated results into version control.
- Do not change the scientific scoring semantics unless needed to remove global coupling.

## Architecture

### Configuration

Create `src/discovery/target_discovery/config.py`.

Primary dataclasses:

```python
@dataclass
class DiscoveryPaths:
    root: Path
    data_dir: Path
    neu_dir: Path
    ifng_dir: Path
    icb_dir: Path
    st_dir: Path
    output_base: Path
    icb_h5ad_path: Path
    reference_manifest_path: Path


@dataclass
class GeometryModeConfig:
    modes: tuple[str, ...] = ("hyperbolic", "euclidean")
    geometry_k: int = 4
    geometry_blend: float = 0.30


@dataclass
class TargetDiscoveryConfig:
    paths: DiscoveryPaths
    geometry: GeometryModeConfig
    max_perturb: int = 50
    platform: str = "all"
    focused_genes: tuple[str, ...] = ()
    hierarchy_levels: int = 3
    run_id: str | None = None
    random_seed: int = 42
    device: str = "cuda"
    skip_figures: bool = False
```

Move constants into this package:

- `ANCHOR_GENES`
- `IFNG_FOCUS_GENES`
- `CELLTYPES`
- `TYPE_MAPPING`
- `ST_DECONV_MAP`
- `ICB_TO_NEU_MAP`
- `PRIOR_AXES`
- `SCORE_WEIGHTS`

Constants should remain importable from `config.py` or `constants.py`.

### Run Context

Create a structured run context:

```python
@dataclass
class TargetDiscoveryRunContext:
    config: TargetDiscoveryConfig
    writer: ArtifactWriter
    started_at: float
    icb_data_mode: str
```

The context is passed to all stages. Stages should not depend on module-level output directories.

### Stage Interface

Create `src/discovery/target_discovery/stage.py`.

```python
class DiscoveryStage(Protocol):
    name: str

    def run(
        self,
        context: TargetDiscoveryRunContext,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...
```

Each stage accepts named inputs and returns named artifacts. Stages should be small enough to test independently with synthetic data.

### Artifact Writer

Create `src/discovery/target_discovery/artifacts.py`.

Responsibilities:

- Create a run directory.
- Save tables, JSON, numpy arrays, markdown, and figures.
- Record stage outputs in `manifest.json`.
- Record warnings and stage timings.
- Generate migration notes for old vs new paths.

Suggested API:

```python
class ArtifactWriter:
    def write_table(self, name: str, df: pd.DataFrame, section: str) -> Path: ...
    def write_json(self, name: str, payload: Mapping[str, Any], section: str) -> Path: ...
    def write_array(self, name: str, arr: np.ndarray, section: str) -> Path: ...
    def write_markdown(self, name: str, text: str, section: str) -> Path: ...
    def write_figure(self, name: str, fig: Figure, section: str, metadata: Mapping[str, Any] | None = None) -> Path: ...
```

The writer owns JSON serialization for numpy, pandas, and torch-compatible values.

## Pipeline Stages

Default stage order:

1. `CandidateDiscoveryStage`
2. `ExpressionAssemblyStage`
3. `SpatialContextStage`
4. `GeometryComparisonStage`
5. `CausalDiscoveryStage`
6. `PerturbationScreenStage`
7. `EvidenceScoringStage`
8. `UnifiedNicheStage`
9. `ReportAndFigureStage`

### CandidateDiscoveryStage

Inputs:

- `TargetDiscoveryConfig`
- ICB data mode from context

Outputs:

- `candidate_pool`

Responsibilities:

- Read Neu DESeq2 result files.
- Read ICB DEG CSV files.
- Read IFNG target tables.
- Aggregate per-gene evidence.
- Compute initial candidate score.
- Save `candidates/candidate_pool.csv`.

Existing logic comes from `build_candidate_pool()`.

### ExpressionAssemblyStage

Outputs:

- `cluster_expression`
- `node_labels`

Responsibilities:

- Read `*-NormalizedCounts.tsv` files for configured cell types.
- Aggregate sample columns to cluster-level mean expression.
- Apply `log1p`.
- Save `expression/cluster_expression.csv`.
- Save `expression/node_labels.json`.

Existing logic comes from `build_cluster_expression()`.

### SpatialContextStage

Inputs:

- `node_labels`

Outputs:

- `spatial_adjacency`

Responsibilities:

- Read ST metadata tables.
- Map deconvolution columns using `ST_DECONV_MAP`.
- Compute cross-cell-type co-localization adjacency.
- Normalize adjacency.
- Save `spatial/spatial_adjacency.npy`.

Existing logic comes from `build_spatial_adjacency()`.

### GeometryComparisonStage

Inputs:

- `cluster_expression`
- `node_labels`
- `spatial_adjacency`

Outputs:

- `geometry_results`
- `blended_adjacencies`

Responsibilities:

- Compute hyperbolic and Euclidean embeddings.
- Compute pairwise distances and geometry-specific kNN adjacency.
- Blend spatial and geometry adjacency using `geometry_blend`.
- Save each mode under `geometry/<mode>/`.

Existing logic comes from `compute_geometry()` plus blending in `main()`.

### CausalDiscoveryStage

Inputs:

- `cluster_expression`
- `node_labels`
- `blended_adjacencies`

Outputs:

- `causal_results`

Responsibilities:

- For each geometry mode, invoke the existing Step2 causal discovery logic.
- Train disentanglement model.
- Run bootstrap causal discovery and threshold pruning.
- Inject prior edges from `PRIOR_AXES`.
- Validate structure with DoWhy wrapper.
- Infer signaling flow.
- Evaluate causal metrics.
- Save mode-specific outputs under `causal/<mode>/`.

Existing logic comes from `run_step2()`.

Step2 algorithm behavior should stay unchanged in this refactor except for path handling and configuration injection.

### PerturbationScreenStage

Inputs:

- `candidate_pool`
- `cluster_expression`
- `causal_results`

Outputs:

- `perturbation_results`
- `perturbation_targets`

Responsibilities:

- Select perturbation targets from available candidate genes.
- For each mode, run batch perturbation.
- Save mode-specific outputs under `perturbation/<mode>/`.

Existing logic comes from target selection in `main()` and `run_step3_batch()`.

Step3 algorithm behavior should stay unchanged except for path handling and configuration injection.

### EvidenceScoringStage

Inputs:

- `candidate_pool`
- `causal_results`
- `perturbation_results`
- `cluster_expression`

Outputs:

- `target_ranking`
- `retained_hubs`
- `retained_combos`
- `mode_comparison`

Responsibilities:

- Compute final evidence scores.
- Save `scoring/target_ranking.csv`.
- Save `scoring/evidence_matrix.csv`.
- Save retained hub targets.
- Save spatiotemporal regulatory combinations.
- Compare hyperbolic and Euclidean modes.
- Save `scoring/mode_comparison.json`.

Existing logic comes from `score_and_rank()`, `retain_hubs_and_combos()`, and `compare_modes()`.

### UnifiedNicheStage

Inputs:

- `target_ranking`
- `cluster_expression`
- `node_labels`
- `retained_combos`

Outputs:

- `available_data_inventory`
- `niche_pack`
- `target_niche`
- `combo_niche`

Responsibilities:

- Collect available data inventory.
- Merge multi-platform deconvolution tables.
- Build unified niche definitions.
- Map targets and combinations to niches.
- Save all niche outputs under `niche/`.

Existing logic comes from:

- `collect_available_data_inventory()`
- `_merge_multimodal_deconv_tables()`
- `build_unified_niche_definition()`
- `map_targets_to_unified_niches()`

### ReportAndFigureStage

Inputs:

- All final artifacts.

Outputs:

- Report paths.
- Figure paths.

Responsibilities:

- Generate figure pack unless `skip_figures=True`.
- Generate target discovery report.
- Generate migration notes.
- Save `reports/target_discovery_report.md`.
- Save `reports/migration_notes.md`.

Existing logic comes from `generate_figures()` and `generate_report()`.

## Output Layout

New output layout:

```text
results/discovery/target_discovery/<run_id>/
|-- manifest.json
|-- candidates/
|-- expression/
|-- spatial/
|-- geometry/
|   |-- hyperbolic/
|   `-- euclidean/
|-- causal/
|   |-- hyperbolic/
|   `-- euclidean/
|-- perturbation/
|   |-- hyperbolic/
|   `-- euclidean/
|-- scoring/
|-- niche/
|-- figures/
`-- reports/
```

The old output root was:

```text
results/integration/discovery/
```

The migration note must include a table mapping commonly used old outputs to new outputs.

## CLI Behavior

`scripts/run_target_discovery.py` remains the main command but can change parameters.

Required CLI options:

- `--output-dir`
- `--run-id`
- `--max-perturb`
- `--geometry-k`
- `--geometry-blend`
- `--platform`
- `--genes`
- `--hierarchy-levels`
- `--skip-figures`
- `--device`

The CLI should print:

- Resolved run directory.
- ICB data mode.
- Number of candidates.
- Number of perturbation targets.
- Final report path.
- Manifest path.

## Testing Plan

### Unit Tests

Create tests under:

```text
tests/discovery/test_target_discovery_*.py
```

Initial unit test coverage:

- JSON serialization handles numpy scalars, arrays, pandas dataframes, and sets.
- Candidate aggregation handles empty sources and multi-source gene evidence.
- Geometry helper returns symmetric adjacency and stable metrics on small synthetic data.
- Score normalization handles constant arrays without NaNs.
- Evidence scoring handles synthetic candidate and mock stage outputs.
- Target-to-niche mapping handles missing genes and valid weighted niche output.

### Stage Tests

Use synthetic temporary inputs to test:

- `CandidateDiscoveryStage`
- `ExpressionAssemblyStage`
- `SpatialContextStage`
- `GeometryComparisonStage`
- `EvidenceScoringStage`
- `UnifiedNicheStage` fallback path

Heavy Step2 and Step3 stages can be tested with mocks first.

### Pipeline Smoke Test

Add one smoke test that:

- Builds a synthetic `TargetDiscoveryConfig`.
- Replaces heavy causal and perturbation stages with fake stages.
- Runs `TargetDiscoveryPipeline`.
- Verifies `manifest.json`, `reports/migration_notes.md`, and section directories are written.

## Migration Notes Requirements

Create:

```text
reports/migration_notes.md
```

It must include:

- Old CLI command examples and new examples.
- Old output root and new output root.
- A table mapping old files to new files.
- A note that CLI/output compatibility was intentionally not guaranteed for this refactor.
- A note that Step2 and Step3 algorithm behavior was preserved during the first structural refactor.

## Acceptance Criteria

- `scripts/run_target_discovery.py` is a thin entrypoint.
- New target discovery package exists under `src/discovery/target_discovery/`.
- The pipeline runs through the same conceptual stages as the old script.
- Existing scientific logic is preserved where the spec marks it as preserved.
- CLI and output path changes are documented.
- Unit tests cover pure helpers and lightweight stages.
- Pipeline smoke test verifies manifest and run directory creation.
- The refactor does not revert existing unrelated working-tree changes.

## Risks

- Heavy stages may be slow or GPU-dependent; tests should mock them unless explicitly testing integration behavior.
- Existing notebooks may assume old output paths; migration notes must document the new layout.
- Some old functions write files internally. During implementation, writes should be moved to `ArtifactWriter` without changing algorithm outputs.
- Windows temp directory permissions can affect pytest fixtures in this workspace. Verification commands may need an approved temp directory strategy or targeted tests that avoid `tmp_path` where practical.
