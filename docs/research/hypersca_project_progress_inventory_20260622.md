# HyperSCA Project Progress Inventory 2026-06-22

This inventory summarizes the current local worktree for preparing a GitHub submission. It is intentionally conservative: large raw outputs remain local, and algorithm changes should be reviewed separately from the documentation-only progress commit.

## Local Worktree Categories

| Category | File count | Representative paths |
| --- | --- | --- |
| Benchmark and workflow scripts | 26 | `M scripts/generate_spatial_combo_comm_figures.py`<br>`M scripts/generate_spatial_comm_figures.py`<br>`M scripts/run_causal_stability_audit.py`<br>`M scripts/run_data_onboarding.py`<br>`M scripts/run_platform_niche_analysis.py`<br>`M scripts/run_step3.py`<br>`M scripts/run_target_discovery.py`<br>`?? scripts/generate_current_pipeline_docs.py`<br>... +18 more |
| Data onboarding and prior database | 5 | `M src/data/prior_db/__init__.py`<br>`M src/data/prior_db/_config.py`<br>`M src/data/prior_db/_download.py`<br>`M src/data/prior_db/_integrate.py`<br>`?? configs/` |
| Documentation and reports | 10 | `M README.md`<br>`?? docs/github_submission_20260622.md`<br>`?? docs/research/figures/hypersca_current_pipeline_flowchart_20260622.png`<br>`?? docs/research/figures/hypersca_current_pipeline_flowchart_20260622.svg`<br>`?? docs/research/figures/hypersca_current_pipeline_overview_imagegen_20260622.png`<br>`?? docs/research/figures/hypersca_two_candidate_downstream_summary_20260622.png`<br>`?? docs/research/hypersca_benchmark_progress_20260622.json`<br>`?? docs/research/hypersca_benchmark_progress_20260622.md`<br>... +2 more |
| Hyperbolic representation modules | 4 | `?? scripts/search_scimilarity_hyperbolic_literature.py`<br>`?? src/models/hyperbolic/hierarchy_losses.py`<br>`?? tests/test_hyperbolic_hierarchy_losses.py`<br>`?? tests/test_scimilarity_hyperbolic_literature_search.py` |
| Other modified project files | 16 | `M requirements.txt`<br>`M scripts/build_canonical_schema.py`<br>`M scripts/build_integration_notebooks.py`<br>`M scripts/download_osta_colon_data.R`<br>`M scripts/prepare_h5ad.py`<br>`M src/causal/stability_audit.py`<br>`M tests/test_causal_stability_audit.py`<br>`M tests/test_step3_pipeline.py`<br>... +8 more |
| Perturbation and spatial evaluation | 3 | `M src/evaluation/spatial_metrics.py`<br>`M src/perturbation/spatial_propagation.py`<br>`M src/perturbation/target_ranking.py` |
| Target discovery and benchmark tests | 50 | `M src/discovery/target_discovery/candidates.py`<br>`M src/discovery/target_discovery/causal_stage.py`<br>`M src/discovery/target_discovery/config.py`<br>`M src/discovery/target_discovery/constants.py`<br>`M src/discovery/target_discovery/expression.py`<br>`M src/discovery/target_discovery/figures.py`<br>`M src/discovery/target_discovery/geometry.py`<br>`M src/discovery/target_discovery/heavy_stages.py`<br>... +42 more |

## Large Local Benchmark Outputs

| Path | Approx. size | Policy |
| --- | --- | --- |
| results/benchmarks/hyperbolic_spatial_crc_v3_two_candidate_downstream_20260622 | 29.3 MB | local ignored result source |
| results/benchmarks/unified_spatial_annotation_cell2location_visiumhd_full | 70.9 MB | local ignored result source |
| results/benchmarks/hyperbolic_spatial_crc_v3_visiumhd_loss_ablation_celcomen_energy_20260622 | 88.7 MB | local ignored result source |

## Submission Recommendation

- Commit the README, compact progress report, workflow figures, and submission notes first.
- Do not stage raw `results/` outputs unless a reviewer explicitly requests a small derived artifact.
- Review algorithm code separately: target-discovery modules, hyperbolic hierarchy losses, spatial annotation scripts, and corresponding tests have broader behavioral impact than the documentation refresh.
- Keep Xenium panel-aware handling as a distinct branch of the workflow and avoid whole-transcriptome deconvolution claims for targeted-panel data.
