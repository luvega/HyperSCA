# GitHub Submission Plan 2026-06-22

## Recommended Scope

This repository has a mixed worktree with many algorithm and benchmark changes. Do not stage the whole tree blindly. For the documentation-focused progress update, stage only the files below plus any explicitly reviewed algorithm files from the current benchmark branch.

## Suggested Documentation Commit

```bash
git checkout -b docs/current-benchmark-progress-20260622
git add README.md \
  scripts/generate_current_pipeline_docs.py \
  docs/research/hypersca_benchmark_progress_20260622.md \
  docs/research/hypersca_benchmark_progress_20260622.json \
  docs/research/hypersca_project_progress_inventory_20260622.md \
  docs/research/figures/hypersca_current_pipeline_flowchart_20260622.png \
  docs/research/figures/hypersca_current_pipeline_flowchart_20260622.svg \
  docs/research/figures/hypersca_two_candidate_downstream_summary_20260622.png \
  docs/research/figures/hypersca_current_pipeline_overview_imagegen_20260622.png \
  docs/github_submission_20260622.md
git commit -m "docs: summarize current HyperSCA benchmark progress"
```

The `gh` CLI is not available in this environment, so PR creation should be done after installing `gh` or from the GitHub web UI.

## Draft PR Summary

```markdown
## Summary
- regenerate README around the current HyperSCA spatial-hyperbolic benchmark workflow
- add a 2026-06-22 benchmark progress report for the two internal v3 candidates
- add reproducible workflow figures and a GitHub-ready submission checklist
- add a local progress inventory so broad algorithm changes can be reviewed separately from docs

## Benchmark Status
- main comparison is limited to `hvae_hierarchy_spatial_v3_product` and `hvae_hierarchy_spatial_v3_product__without_radial_depth_loss`
- SCimilarity remains an external pretrained appendix reference
- quality gate remains audit-only/no-promotion because target rank delta and target enrichment deltas are still zero
- VisiumHD cell2location full abundance row check remains 545,913 rows

## Tests
- `python scripts/generate_current_pipeline_docs.py`
- `python -m py_compile scripts/generate_current_pipeline_docs.py`
- `PYTHONPATH=. pytest tests -q -p no:cacheprovider`
```

## Raw Artifact Policy

Keep large raw benchmark outputs in ignored `results/` paths. Commit compact reports, figures, and JSON snapshots under `docs/` or `reports/` only.
