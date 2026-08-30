# Methods v3 pre-freeze stop review

The previously frozen v2.1 audit remains `pilot_failed_no_release`; this
pre-freeze check does not alter that decision.

The registered GSE274447 bridge candidate declares three mice, but its expected
public asset root (`$SPATIAL_PERTURB_ROOT`) was absent at the pre-freeze check.
The metadata-only capability record therefore has
`status=assets_unavailable`, is not confirmatory-capable, and includes
`external_cohort_missing`. No assay values or outcomes were opened, and no
effect, prediction, metric, or RMSE result was produced.

No v3 protocol config was frozen. No predictor capability audit was run. No
formal predictor adapter was evaluated. No real bridge pilot was run. In
addition, no paired scientific collection exists. The evidence policy remains
`integrated_claim_enabled=false`; CRC application-only evidence cannot repair
the missing bridge evidence or authorize an integrated spatial-causal claim.

Work stops at this missing-cohort gate. A future bridge attempt requires the
external cohort to be present before freezing, plus a separate preregistered
design and protocol identity before any production predictor adapter may see
bridge outcomes. The backup propagation concept remains unauthorized.
