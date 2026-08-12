# Causal Stability Null-Control Policy v1

Status: active audit policy for post-v0.6 development.

## Scope

The current implementation generates surrogate null matrices from the saved
Step2 bootstrap-frequency matrix. It measures whether reported edge frequency
and topology are unusually strong relative to controlled rearrangements of the
same matrix. It does not rerun causal discovery on altered cell-level data and
therefore cannot establish identifiability, remove hidden confounding, or prove
an intervention mechanism.

## Reproducible controls

Each run freezes `n_null_controls`, `null_modes`, and `random_seed`. Supported
modes are:

- `matrix_permutation`: permute all off-diagonal frequencies;
- `node_label_shuffle`: jointly permute row and column labels;
- `outgoing_weight_permutation`: permute each source row's off-diagonal weight
  multiset while preserving that multiset.

The historical name `degree_preserving` is accepted as an alias for
`outgoing_weight_permutation`; it does not claim exact preservation of the full
directed degree sequence.

The audit writes `null_control_manifest.json` with the canonical modes, seed,
requested/generated counts, scope, and SHA-256 digest of generated matrices.
Generated matrices are reproducible from the saved input matrix and manifest.

## Gate

Fewer than 10 null matrices can never pass the negative-control gate. With at
least 10 matrices, an edge must exceed its null 95th percentile and pass
Benjamini-Hochberg FDR at the configured alpha. Passing edges remain
`sidecar_only`; the audit does not change target ranking.

Future Task C work must add data-level nulls—such as intervention-label
shuffle, observation permutation, and prior-off reruns—and external
interventional scoring before causal promotion is considered.
