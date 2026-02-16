# HyperSCA

HyperSCA (Hyperbolic Spatiotemporal Causal Analysis) is a computational framework for dissecting the colorectal cancer tumor microenvironment by integrating single-cell RNA sequencing and spatial transcriptomics. The project combines hyperbolic representation learning, causal structure discovery, and counterfactual perturbation to support mechanism-oriented analysis rather than correlation-only description.

## Scientific Motivation

Tumor microenvironment analysis typically faces three limitations:

1. Euclidean embeddings do not represent hierarchical and tree-like biological structures efficiently.
2. Correlation-based interaction analysis cannot reliably separate direct and indirect effects.
3. Static marker comparison does not answer intervention questions such as which signaling axis to perturb and how the effect propagates in space.

HyperSCA addresses these limitations with a three-stage design:

- Stage 1 learns geometry-aware latent representations on a hyperbolic manifold.
- Stage 2 infers causal relations with disentanglement and conditional-independence-guided graph pruning.
- Stage 3 performs counterfactual perturbation and spatial propagation analysis with quantitative evaluation.

## Design Principles

HyperSCA is built around four principles:

- Geometric fidelity: preserve latent hierarchy and geodesic structure in embedding space.
- Causal interpretability: favor explicit graph structures and intervention-aware validation.
- Spatial consistency: model neighborhood diffusion and distance-decay effects after perturbation.
- Reproducibility: provide testable modules, script-level pipelines, and structured result artifacts.

## Architecture Overview

### Stage 1. Hyperbolic Embedding

- Core modules: `src/models/hyperbolic/lorentz.py`, `src/models/hyperbolic/poincare.py`, `src/models/hyperbolic/wrapped_normal.py`, `src/models/hyperbolic/hvae.py`
- Data and graph support: `src/data/preprocessing.py`, `src/data/spatial_graph.py`
- Pipeline: `src/pipeline/step1_embedding.py`
- Evaluation: `src/evaluation/embedding_metrics.py`

This stage uses an H-VAE with graph-aware encoding and manifold-aware decoding to produce latent representations that better capture hierarchical cellular organization.

### Stage 2. Causal Network Discovery

- Core modules: `src/causal/disentangle.py`, `src/causal/cmi_pruning.py`, `src/causal/causal_graph.py`, `src/causal/signaling_flow.py`
- Pipeline: `src/pipeline/step2_causal.py`
- Evaluation: `src/evaluation/causal_metrics.py`

This stage separates external and internal factors, discovers directed dependencies with CI-based pruning, and organizes evidence into interpretable causal graphs and signaling flow summaries.

### Stage 3. Counterfactual Perturbation

- Core modules: `src/perturbation/latent_arithmetic.py`, `src/perturbation/spatial_propagation.py`, `src/perturbation/diffusion_cf.py`, `src/perturbation/target_ranking.py`
- Pipeline: `src/pipeline/step3_perturbation.py`
- Evaluation: `src/evaluation/cf_metrics.py`, `src/evaluation/spatial_metrics.py`
- Visualization: `src/visualization/perturbation.py`

This stage applies intervention vectors in latent space, simulates neighborhood propagation, and quantifies counterfactual plausibility and spatial coherence.

## Repository Structure

Key directories in the current implementation:

- `src/`: core framework modules for embedding, causality, perturbation, evaluation, and pipelines
- `tests/`: unit and integration tests across all three stages
- `notebooks/`: reproducible analysis notebooks (`example_01_metadata.ipynb`, `example_02_spatial_graph.ipynb`, `example_03_segmentation.ipynb`, `example_04_xenium.ipynb`)
- `docs/`: technical roadmap, engineering blueprint, evaluation suite, risk plan, and example guide
- `scripts/`: entry points for examples, environment validation, and prior-database download

## Data Modalities

HyperSCA supports multiple modalities and platform-specific workflows:

- Chromium scRNA-seq
- Visium spatial transcriptomics
- Xenium in situ spatial transcriptomics
- VisiumHD is supported in the data pipeline and can be extended in staged workflows

## Evaluation Coverage

The framework includes evaluation at multiple levels:

- Embedding quality: distortion, clustering consistency, neighborhood preservation
- Causal credibility: independence criteria, graph sparsity, bootstrap support, predictive consistency
- Counterfactual quality: expression statistics consistency, directionality checks, marker-level agreement
- Spatial consistency: Moran-based metrics, propagation depth, decay fitting, spatial-causal coupling

Current test coverage includes 79 pytest cases across geometry, model behavior, causal discovery, perturbation modules, and end-to-end stage pipelines.

## Reproducibility and Execution

### Environment

- Recommended conda environment: `hypersca`
- Python: 3.10
- GPU support: CUDA-compatible PyTorch setup

### Installation

```bash
pip install -r requirements.txt
```

### Validation

```bash
python scripts/validate_env.py
```

### Stage Pipelines

```bash
python scripts/run_step1.py
python scripts/run_step2.py
python scripts/run_step3.py
```

### Tests

```bash
pytest tests/ -v
```

## Outputs

By default, generated artifacts are organized under:

- `results/step1/`: latent embeddings, model checkpoints, adjacency and training records
- `results/step2/`: causal graphs, metrics, signaling summaries, interpretation reports
- `results/step3/`: perturbation outputs, propagation summaries, counterfactual and spatial metrics
- `results/figures/`: stage-specific visualization panels
- `results/examples/`: notebook or example-oriented outputs

## Documentation

For full technical details:

- `docs/technical_roadmap.md`
- `docs/engineering_blueprint.md`
- `docs/evaluation_suite.md`
- `docs/priority_and_risks.md`
- `docs/examples_guide.md`

## Contribution Guidelines

- Follow Python 3.10 and PEP 8 conventions.
- Keep module responsibilities explicit and test-covered.
- Prefer reproducible scripts and deterministic output formats.
- Do not commit raw data or generated large artifacts.

## License

This project is released under the `MIT License`.
