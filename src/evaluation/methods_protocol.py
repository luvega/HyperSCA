"""Frozen scientific protocol for HyperSCA method-development evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass


_SHA256 = re.compile(r"[0-9a-f]{64}")
_TERMINAL_STATUSES = (
    "completed",
    "failed_invalid_input",
    "failed_invalid_output",
    "failed_timeout",
    "failed_resource",
    "failed_runtime",
    "not_applicable",
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty built-in string")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    gpu_count: int
    gpu_memory_gib: int
    ram_gib: int
    artifact_gib_per_run: int
    pilot_osta_gpu_hours: int
    pilot_causalbench_gpu_hours: int
    release_multiplier: float
    release_max_hours: int

    def __post_init__(self) -> None:
        for name in (
            "gpu_count",
            "gpu_memory_gib",
            "ram_gib",
            "artifact_gib_per_run",
            "pilot_osta_gpu_hours",
            "pilot_causalbench_gpu_hours",
            "release_max_hours",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive built-in integer")
        if type(self.release_multiplier) is not float or not math.isfinite(
            self.release_multiplier
        ):
            raise ValueError("release_multiplier must be a finite built-in float")
        if self.release_multiplier < 1.0:
            raise ValueError("release_multiplier must be at least one")


@dataclass(frozen=True, slots=True)
class MethodsProtocol:
    protocol_version: str
    spatial_primary_metric: str
    spatial_primary_k: int
    spatial_secondary_k: tuple[int, int]
    causal_primary_metric: str
    spatial_confirmatory_comparator: str
    spatial_attribution_comparator: str
    causal_confirmatory_comparator: str
    causal_attribution_comparator: str
    secondary_comparators: tuple[str, ...]
    pilot_seeds: tuple[int, int, int]
    release_seeds: tuple[int, int, int, int, int]
    bootstrap_resamples: int
    bootstrap_confidence: float
    multiple_testing: str
    parameter_tolerance: float
    maximum_protocol_redesigns: int
    crc_role: str
    terminal_statuses: tuple[str, ...]
    required_evidence_artifacts: tuple[str, ...]
    resource_policy: ResourcePolicy

    def __post_init__(self) -> None:
        for name in (
            "protocol_version",
            "spatial_primary_metric",
            "causal_primary_metric",
            "spatial_confirmatory_comparator",
            "spatial_attribution_comparator",
            "causal_confirmatory_comparator",
            "causal_attribution_comparator",
            "multiple_testing",
            "crc_role",
        ):
            _text(getattr(self, name), name)
        if self.spatial_primary_k != 15 or type(self.spatial_primary_k) is not int:
            raise ValueError("spatial_primary_k is frozen to 15")
        if self.spatial_secondary_k != (5, 30):
            raise ValueError("spatial_secondary_k is frozen to (5, 30)")
        if self.pilot_seeds != (11, 23, 47):
            raise ValueError("pilot_seeds are frozen to (11, 23, 47)")
        if self.release_seeds != (11, 23, 47, 71, 101):
            raise ValueError("release_seeds are frozen to five preregistered seeds")
        if (
            type(self.bootstrap_resamples) is not int
            or self.bootstrap_resamples != 10_000
        ):
            raise ValueError("bootstrap_resamples is frozen to 10000")
        if (
            type(self.bootstrap_confidence) is not float
            or self.bootstrap_confidence != 0.95
        ):
            raise ValueError("bootstrap_confidence is frozen to 0.95")
        if self.multiple_testing != "holm_two_confirmatory_claims":
            raise ValueError(
                "multiple_testing must be Holm correction over the two claims"
            )
        if (
            type(self.parameter_tolerance) is not float
            or self.parameter_tolerance != 0.05
        ):
            raise ValueError("parameter_tolerance is frozen to 0.05")
        if (
            type(self.maximum_protocol_redesigns) is not int
            or self.maximum_protocol_redesigns != 1
        ):
            raise ValueError("maximum_protocol_redesigns is frozen to one")
        if self.crc_role != "application_only":
            raise ValueError("CRC must remain application_only")
        if type(self.secondary_comparators) is not tuple or len(
            set(self.secondary_comparators)
        ) != len(self.secondary_comparators):
            raise ValueError("secondary_comparators must be a unique tuple")
        if any(
            type(item) is not str or not item for item in self.secondary_comparators
        ):
            raise ValueError(
                "secondary_comparators must contain non-empty built-in strings"
            )
        primary_comparators = {
            self.spatial_confirmatory_comparator,
            self.spatial_attribution_comparator,
            self.causal_confirmatory_comparator,
            self.causal_attribution_comparator,
        }
        if len(primary_comparators) != 4:
            raise ValueError("claim-specific primary comparators must be distinct")
        if primary_comparators.intersection(self.secondary_comparators):
            raise ValueError(
                "confirmatory and attribution comparators must not be relabeled secondary"
            )
        if self.terminal_statuses != _TERMINAL_STATUSES:
            raise ValueError("terminal_statuses must match the frozen closed set")
        if (
            type(self.required_evidence_artifacts) is not tuple
            or len(self.required_evidence_artifacts) != 7
        ):
            raise ValueError(
                "required_evidence_artifacts must contain the seven frozen bundle records"
            )
        if type(self.resource_policy) is not ResourcePolicy:
            raise ValueError("resource_policy must be ResourcePolicy")


def default_methods_protocol() -> MethodsProtocol:
    return MethodsProtocol(
        protocol_version="hypersca-methods-v2.1",
        spatial_primary_metric="neighborhood_preservation_at_k",
        spatial_primary_k=15,
        spatial_secondary_k=(5, 30),
        causal_primary_metric="directed_edge_average_precision",
        spatial_confirmatory_comparator="euclidean_autoencoder",
        spatial_attribution_comparator="hypersca_without_hierarchy_loss",
        causal_confirmatory_comparator="mean_difference",
        causal_attribution_comparator="hypersca_c_shared_only",
        secondary_comparators=(
            "euclidean_pca",
            "external_registered_methods",
            "best_observed_method",
        ),
        pilot_seeds=(11, 23, 47),
        release_seeds=(11, 23, 47, 71, 101),
        bootstrap_resamples=10_000,
        bootstrap_confidence=0.95,
        multiple_testing="holm_two_confirmatory_claims",
        parameter_tolerance=0.05,
        maximum_protocol_redesigns=1,
        crc_role="application_only",
        terminal_statuses=_TERMINAL_STATUSES,
        required_evidence_artifacts=(
            "run_manifest.json",
            "method_status.json",
            "resource_usage.json",
            "primary_metric_units.csv",
            "primary_metric_summary.json",
            "secondary_metrics.csv",
            "claim_decision.json",
        ),
        resource_policy=ResourcePolicy(
            gpu_count=1,
            gpu_memory_gib=40,
            ram_gib=128,
            artifact_gib_per_run=5,
            pilot_osta_gpu_hours=2,
            pilot_causalbench_gpu_hours=4,
            release_multiplier=1.5,
            release_max_hours=12,
        ),
    )


def protocol_to_mapping(protocol: MethodsProtocol) -> dict[str, object]:
    if type(protocol) is not MethodsProtocol:
        raise ValueError("protocol must be MethodsProtocol")
    return {
        "schema": "hypersca_methods_protocol_v2",
        "protocol_version": protocol.protocol_version,
        "claims": {
            "spatial": {
                "benchmark": "osta_colon",
                "primary_metric": protocol.spatial_primary_metric,
                "primary_k": protocol.spatial_primary_k,
                "secondary_k": list(protocol.spatial_secondary_k),
                "split": "contiguous_heldout_blocks_with_buffer",
                "aggregation": "sample_macro_with_platform_strata",
            },
            "causal": {
                "benchmark": "causalbench_k562_rpe1",
                "primary_metric": protocol.causal_primary_metric,
                "relation_universe": "complete_p_times_p_minus_1_missing_scores_zero",
                "contexts": ["k562", "rpe1"],
                "aggregation": "context_macro_eligible_holdout_sources_only",
            },
        },
        "comparators": {
            "spatial": {
                "confirmatory": protocol.spatial_confirmatory_comparator,
                "attribution": protocol.spatial_attribution_comparator,
            },
            "causal": {
                "confirmatory": protocol.causal_confirmatory_comparator,
                "attribution": protocol.causal_attribution_comparator,
            },
            "secondary": list(protocol.secondary_comparators),
            "parameter_tolerance": protocol.parameter_tolerance,
            "same_budget_fields": [
                "data_identity",
                "gene_identity",
                "seed",
                "optimizer_family",
                "max_updates",
                "early_stopping_patience",
                "tuning_trials",
            ],
        },
        "statistics": {
            "bootstrap_resamples": protocol.bootstrap_resamples,
            "bootstrap_confidence": protocol.bootstrap_confidence,
            "multiple_testing": protocol.multiple_testing,
            "minimum_ci_lower_bound": 0.0,
        },
        "execution": {
            "pilot_seeds": list(protocol.pilot_seeds),
            "release_seeds": list(protocol.release_seeds),
            "pilot_scopes": ["train", "tune"],
            "release_holdout": "score_once_only",
            "maximum_protocol_redesigns": protocol.maximum_protocol_redesigns,
            "redesigns_used": 1,
            "redesign_reason": (
                "causalbench_representation_comparators_were_not_executable"
            ),
            "infrastructure_retry": "same_run_identity_only",
            "terminal_statuses": list(protocol.terminal_statuses),
        },
        "resources": asdict(protocol.resource_policy),
        "evidence_bundle": list(protocol.required_evidence_artifacts),
        "crc": {
            "role": protocol.crc_role,
            "allowed_analyses": [
                "patient_timepoint_resampling_stability",
                "rank_change",
                "response_association",
            ],
            "may_rescue_public_promotion": False,
        },
    }


def protocol_identity(protocol: MethodsProtocol) -> str:
    payload = json.dumps(
        protocol_to_mapping(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelBudget:
    parameter_count: int
    optimizer_family: str
    max_updates: int
    early_stopping_patience: int
    tuning_trials: int
    data_identity: str
    gene_identity: str
    seed: int
    hierarchy_loss_enabled: bool

    def __post_init__(self) -> None:
        for name in (
            "parameter_count",
            "max_updates",
            "early_stopping_patience",
            "tuning_trials",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive built-in integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative built-in integer")
        _text(self.optimizer_family, "optimizer_family")
        _sha256(self.data_identity, "data_identity")
        _sha256(self.gene_identity, "gene_identity")
        if type(self.hierarchy_loss_enabled) is not bool:
            raise ValueError("hierarchy_loss_enabled must be a built-in boolean")


def validate_comparator_fairness(
    hypersca: ModelBudget,
    confirmatory: ModelBudget,
    attribution: ModelBudget,
    *,
    parameter_tolerance: float,
) -> None:
    if any(
        type(value) is not ModelBudget
        for value in (hypersca, confirmatory, attribution)
    ):
        raise ValueError("all model budgets must be ModelBudget values")
    if type(parameter_tolerance) is not float or not 0.0 <= parameter_tolerance <= 1.0:
        raise ValueError("parameter_tolerance must be a built-in float in [0, 1]")
    relative_difference = (
        abs(confirmatory.parameter_count - hypersca.parameter_count)
        / hypersca.parameter_count
    )
    if relative_difference > parameter_tolerance:
        raise ValueError(
            "confirmatory comparator parameter count exceeds the frozen tolerance"
        )
    shared_fields = (
        "optimizer_family",
        "max_updates",
        "early_stopping_patience",
        "tuning_trials",
        "data_identity",
        "gene_identity",
        "seed",
    )
    for field in shared_fields:
        if getattr(confirmatory, field) != getattr(hypersca, field):
            raise ValueError(f"confirmatory comparator must match {field}")
    attribution_fields = ("parameter_count",) + shared_fields
    if (
        hypersca.hierarchy_loss_enabled is not True
        or attribution.hierarchy_loss_enabled is not False
        or any(
            getattr(attribution, field) != getattr(hypersca, field)
            for field in attribution_fields
        )
    ):
        raise ValueError(
            "attribution comparator may change only hierarchy_loss_enabled"
        )


def authorize_data_scope(
    protocol: MethodsProtocol,
    *,
    phase: str,
    scope: str,
    operation: str = "score",
) -> str:
    if type(protocol) is not MethodsProtocol:
        raise ValueError("protocol must be MethodsProtocol")
    for value, name in ((phase, "phase"), (scope, "scope"), (operation, "operation")):
        _text(value, name)
    if phase == "pilot":
        if scope == "train":
            return "fit_allowed"
        if scope == "tune":
            return "selection_allowed"
        if scope == "release_holdout":
            raise ValueError("release holdout is sealed during the pilot")
    elif phase == "release":
        if scope == "train":
            return "fit_allowed"
        if scope == "tune":
            return "frozen_selection_evidence_only"
        if scope == "release_holdout":
            if operation != "score":
                raise ValueError(
                    "release holdout is never available for training or selection"
                )
            return "score_once_only"
    raise ValueError("phase/scope combination is not registered by the protocol")


@dataclass(frozen=True, slots=True)
class ProtocolExecutionState:
    phase: str
    redesigns_used: int
    release_identity: str | None

    def __post_init__(self) -> None:
        if type(self.phase) is not str or self.phase not in {
            "pilot",
            "release_ready",
            "release_retryable",
            "release_scored",
            "release_failed",
        }:
            raise ValueError("phase is not a registered protocol execution phase")
        if type(self.redesigns_used) is not int or self.redesigns_used not in (0, 1):
            raise ValueError("redesigns_used must be zero or one")
        if self.release_identity is not None:
            _sha256(self.release_identity, "release_identity")
        if (
            self.phase in {"release_retryable", "release_scored", "release_failed"}
            and self.release_identity is None
        ):
            raise ValueError("a release attempt phase requires release_identity")


def transition_protocol(
    state: ProtocolExecutionState, event: str
) -> ProtocolExecutionState:
    if type(state) is not ProtocolExecutionState:
        raise ValueError("state must be ProtocolExecutionState")
    _text(event, "event")
    if event == "redesign_after_pilot_failure":
        if state.phase != "pilot" or state.redesigns_used >= 1:
            raise ValueError("only one protocol redesign is allowed during the pilot")
        return ProtocolExecutionState(
            phase="pilot", redesigns_used=1, release_identity=None
        )
    if event == "freeze_release":
        if state.phase != "pilot":
            raise ValueError("only the pilot can transition to release_ready")
        return ProtocolExecutionState(
            phase="release_ready",
            redesigns_used=state.redesigns_used,
            release_identity=None,
        )
    raise ValueError("event is not registered by the protocol state machine")


def record_release_attempt(
    state: ProtocolExecutionState,
    *,
    run_identity: str,
    status: str,
) -> ProtocolExecutionState:
    if type(state) is not ProtocolExecutionState:
        raise ValueError("state must be ProtocolExecutionState")
    _sha256(run_identity, "run_identity")
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("status must be a registered terminal status")
    if state.phase == "release_scored":
        raise ValueError("release holdout has already been scored")
    if state.phase not in {"release_ready", "release_retryable"}:
        raise ValueError(
            "release attempts require release_ready or release_retryable state"
        )
    if state.release_identity is not None and run_identity != state.release_identity:
        raise ValueError("infrastructure retries must use the same run identity")
    if status == "completed":
        phase = "release_scored"
    elif status in {"failed_timeout", "failed_resource", "failed_runtime"}:
        phase = "release_retryable"
    else:
        phase = "release_failed"
    return ProtocolExecutionState(
        phase=phase,
        redesigns_used=state.redesigns_used,
        release_identity=run_identity,
    )


def validate_crc_application_request(
    protocol: MethodsProtocol,
    *,
    tunes_hyperparameters: bool,
    selects_baseline: bool,
    changes_public_threshold: bool,
    attempts_promotion_rescue: bool,
) -> None:
    if type(protocol) is not MethodsProtocol or protocol.crc_role != "application_only":
        raise ValueError("protocol must preserve the CRC application-only role")
    values = (
        tunes_hyperparameters,
        selects_baseline,
        changes_public_threshold,
        attempts_promotion_rescue,
    )
    if any(type(value) is not bool for value in values) or any(values):
        raise ValueError(
            "CRC is application-only and cannot tune, select, change thresholds, or rescue promotion"
        )


__all__ = [
    "MethodsProtocol",
    "ModelBudget",
    "ProtocolExecutionState",
    "ResourcePolicy",
    "authorize_data_scope",
    "default_methods_protocol",
    "protocol_identity",
    "protocol_to_mapping",
    "record_release_attempt",
    "transition_protocol",
    "validate_comparator_fairness",
    "validate_crc_application_request",
]
