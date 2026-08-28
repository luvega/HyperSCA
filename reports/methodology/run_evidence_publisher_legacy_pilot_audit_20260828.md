# RunEvidencePublisher legacy pilot audit

Date: 2026-08-28
Protocol: `hypersca-methods-v2.1`
Audit mode: read-only; public train/tune evidence only

## Scope and result

This audit inspected the manifests and terminal status records of the 18
previously accepted pilot runs without rewriting their directories:

- 12 OSTA runs under
  `results/methods_pilot_v21/osta_fixed_split_19911_blocks`;
- 6 CausalBench runs under
  `results/methods_pilot_v21/causalbench_fixed_split_11`.

All 18 legacy manifests report protocol v2.1, model seeds 11/23/47 and public
`train,tune` scopes. OSTA reports the fixed spatial split seed 19911 across four
dataset/platform units. CausalBench reports fixed public split seed 11 through
its public-manifest evidence, two registered directions, 15 eligible sources
per direction, and 138/269 public tune-response edges respectively.

The conclusion is deliberately conservative: these directories remain legacy
audit evidence. They are not silently upgraded to publisher-verified evidence.

## Field compatibility

The following new identity inputs can be reconstructed for an offline
compatibility analysis:

- protocol version and recorded protocol SHA-256;
- benchmark, claim, public scopes and model seed;
- OSTA split seed and spatial split geometry;
- CausalBench public-manifest SHA-256, direction, eligible sources and tune
  reference edges;
- recorded input, configuration and execution-code hashes;
- OSTA K=15 unit identifiers from `primary_metric_units.csv`;
- the CausalBench complete eligible-source relation universe from ordered genes
  and eligible sources.

They cannot be treated as originally sealed publisher identities because the
legacy bundles do not contain:

- separate canonical `data_split_identity_sha256` and
  `statistical_unit_identity_sha256` fields fixed before publication;
- a canonical `run_identity_sha256` covering every frozen field;
- artifact sizes and media types in the new exact inventory schema;
- cross-bound canonical `method_status.json` and `run_manifest.json` records;
- a replay-derived bundle identity;
- the shared `run_evidence_publisher.py` source SHA.

Recomputing those values today would describe a migration calculation, not
prove that the values were frozen when the legacy runs were produced. The new
verifier is therefore intentionally non-retroactive.

## Invalid roots remain invalid

The audit preserves all three earlier invalidations:

1. `results/methods_pilot_v21/osta` — model seeds also changed the spatial
   split, so runs were not paired;
2. `results/methods_pilot_v21/osta_fixed_split_19911` — four held-out spatial
   blocks were collapsed into one statistical unit, giving a degenerate
   interval;
3. `results/methods_pilot_v21/causalbench` — model seeds also changed the
   public intervention split, genes, eligible sources and relation universe.

No invalid directory was deleted, rewritten or relabeled.

## Scientific authorization

- Legacy bundles remain `audit_only` evidence.
- New publisher verification is not retroactive authorization.
- `promotion_eligible` remains false.
- `release_authorized` remains false.
- No private holdout, refit or release input was read for this audit.
- CRC remains application-only and cannot repair a failed public gate.

The next scientifically valid action is to use the new publisher for a newly
authorized public pilot or later release candidate. This audit does not itself
authorize either action.
