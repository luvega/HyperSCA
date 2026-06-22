# HyperSCA Benchmark Progress Summary 2026-06-22

Generated from local benchmark artifacts at `results/benchmarks/hyperbolic_spatial_crc_v3_two_candidate_downstream_20260622`.

## Current Status

The current GitHub-ready interpretation is conservative: the main method comparison is restricted to two internally trained v3 branches, `hvae_hierarchy_spatial_v3_product` and `hvae_hierarchy_spatial_v3_product__without_radial_depth_loss`. SCimilarity is retained only as an external pretrained appendix reference, not as a main ranking competitor.

The full downstream audit remains `audit_only_no_promotion`. It does not modify the active target ranking because neither candidate produced a non-zero target rank delta or target enrichment improvement. The GPU path was active for the v3 runs, and VisiumHD cell2location full output passed the expected 545,913-row abundance check.

## Main Two-Candidate Metrics

| Candidate | kNN purity | Label AUC | Held-out context AUC | Celcomen energy AUC | Target rank delta |
| --- | --- | --- | --- | --- | --- |
| Full v3 product | 0.671 | 0.746 | 0.654 | 0.716 | 0.000 |
| Without radial-depth loss | 0.673 | 0.754 | 0.648 | 0.710 | 0.000 |

## Target Discovery Gate

| Candidate | Cells | Genes | Scored targets | Changed ranks | Top-k enrichment delta | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| v3_product__without_radial_depth_loss | 5000 | 6000 | 5433 | 0 | 0.000 | audit_only_no_promotion |
| v3_product | 5000 | 6000 | 5433 | 0 | 0.000 | audit_only_no_promotion |

## VisiumHD RCTD and cell2location Concordance

Dominant grid-label concordance between RCTD and cell2location is `0.827`. The strongest shared spatial abundance patterns are tumor and fibroblast compartments; T-cell and ILC fractions are weakly concordant and should be reviewed visually before biological claims.

| Cell type | Grid Spearman | cell2location mean | RCTD mean |
| --- | --- | --- | --- |
| Tumor | 0.844 | 0.283 | 0.442 |
| Fibroblast | 0.751 | 0.228 | 0.317 |
| Myeloid | 0.663 | 0.143 | 0.072 |
| B_cells | 0.606 | 0.115 | 0.015 |
| Intestinal_Epithelial | 0.532 | 0.143 | 0.050 |
| T_cells | 0.107 | 0.044 | 0.011 |
| ILC | 0.079 | 0.045 | 0.002 |

## Interpretation

- The two v3 branches are stable enough for continued downstream testing under the same 5k-cell, 6k-gene, 3-seed protocol.
- The radial-depth branch is not yet functionally justified: removing radial depth slightly improves label AUC, while the full model slightly improves held-out context and interaction-energy AUC.
- Prototype and radial hierarchy objectives still behave near chance; these remain algorithm-development targets rather than release-ready evidence.
- Xenium remains panel-aware by design. Targeted-panel gene coverage does not support whole-transcriptome RCTD or cell2location assumptions, so Xenium should be evaluated with panel-compatible spatial annotation and visualization.

## Next Quality Gates

1. Keep the active target ranking unchanged until target rank delta, target enrichment, or spatial niche biology shows reviewable improvement.
2. Use VisiumHD segmented RCTD and cell2location as full downstream references for spatial niche interpretation.
3. Continue loss-component ablation with explicit prototype, teacher, context-edge, Euclidean residual, and radial-depth reporting.
4. Only promote a v3 branch after seed stability and bootstrap intervals support a reproducible gain.
