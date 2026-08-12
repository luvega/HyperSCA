# Target Discovery Ranking Policy v1

Status: active audit policy for post-v0.6 development.

## Estimand boundary

The main ranking orders data-derived target candidates. It does not estimate a
validated causal treatment effect or a spatial drug mechanism. Causal graphs,
spatial propagation, ligand-receptor priors, and mechanism-chain scores remain
sidecars until their own external intervention gates pass.

## Default policy: `evidence_gated`

Candidates are sorted lexicographically, with stable tie handling, by:

1. number of independent DE sources;
2. direction consistency across observed effects;
3. `-log10` adjusted P value;
4. mean absolute log-fold change;
5. gene symbol, as a deterministic final tie-breaker.

No weighted sum is used. `final_score = (n - rank + 1) / n` is an ordinal
display value only. The output must mark every row with
`ranking_basis=tiered_unweighted_evidence` and
`final_score_method=ordinal_rank_display_not_weighted_sum`.

The runtime writes the exact policy to `scoring/ranking_policy.json` and the
gate decision to `scoring/module_admission.csv`. A partially or inconsistently
marked ranking is blocked.

## Sidecar policy

The following may be reported for audit and validation prioritization but may
not change main rank order:

- causal graph centrality or flow;
- perturbation and spatial propagation proxy metrics;
- ligand-receptor and mechanism priors;
- other representation or niche sidecars without a passed external gate.

The `legacy_full` profile preserves the historical weighted sum for
reproduction only and is explicitly blocked from main-ranking admission.
Manual gene or target seeds are rejected at the pipeline boundary.
