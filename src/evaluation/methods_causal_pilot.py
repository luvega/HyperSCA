"""Train/tune-only CausalBench pilot helpers for methods protocol v2.1.

The pilot intentionally has no refit or private-holdout entry point.  It fits
the registered public-data models elsewhere in this module and turns their
train-stage relation scores into paired evidence against labels derived only
from the matching public tune stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.causal.hypersca_c import HyperSCACContext
from src.evaluation.task_c_predictions import normalize_task_c_predictions


_COMPARATORS = ("mean_difference", "hypersca_c_shared_only")
_CONTEXTS = ("k562", "rpe1")
CAUSALBENCH_SPLIT_SEED = 11


@dataclass(frozen=True, slots=True)
class CausalPilotEvidence:
    metrics: Mapping[str, Mapping[str, object]]
    relations: pd.DataFrame
    promotion_eligible: bool
    data_scopes: tuple[str, str]


def _texts(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an ordered text sequence")
    copied = tuple(values)
    if not copied or any(not isinstance(value, str) or not value for value in copied):
        raise ValueError(f"{label} must contain non-empty built-in strings")
    return tuple(str(value) for value in copied)


def split_hypersca_contexts(
    *,
    expression: object,
    interventions: Sequence[str],
    gene_names: Sequence[str],
    environment_labels: Sequence[str],
) -> tuple[HyperSCACContext, HyperSCACContext]:
    """Copy one validated cross profile into the two registered contexts."""

    values = np.asarray(expression)
    labels = np.asarray(_texts(interventions, "interventions"), dtype=str)
    genes = _texts(gene_names, "gene_names")
    environments = np.asarray(
        _texts(environment_labels, "environment_labels"), dtype=str
    )
    if (
        values.ndim != 2
        or values.shape != (len(labels), len(genes))
        or len(environments) != len(labels)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise ValueError("cross profile arrays are malformed or non-finite")
    if set(environments.tolist()) != set(_CONTEXTS):
        raise ValueError("environment_labels must contain exactly k562 and rpe1")
    contexts = tuple(
        HyperSCACContext(
            context_id=context_id,
            expression=values[environments == context_id],
            interventions=labels[environments == context_id],
            gene_names=genes,
        )
        for context_id in _CONTEXTS
    )
    return contexts  # type: ignore[return-value]


def _metric(
    normalized: pd.DataFrame,
    *,
    positives: frozenset[tuple[str, str]],
    eligible_sources: frozenset[str],
) -> float:
    selected = normalized[normalized["source"].isin(eligible_sources)]
    labels = np.fromiter(
        (
            (str(source), str(target)) in positives
            for source, target in selected[["source", "target"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=np.int64,
        count=len(selected),
    )
    if int(labels.sum()) == 0 or int(labels.sum()) == len(labels):
        raise ValueError("tune relation universe needs positives and negatives")
    value = float(average_precision_score(labels, selected["score"].to_numpy()))
    if not math.isfinite(value):
        raise ValueError("directed-edge average precision is not finite")
    return value


def build_causal_pilot_evidence(
    *,
    gene_names: Sequence[str],
    eligible_sources: Sequence[str],
    tuning_edges: frozenset[tuple[str, str]],
    hypersca_predictions: pd.DataFrame,
    comparator_predictions: Mapping[str, pd.DataFrame],
    direction: str,
    seed: int,
) -> CausalPilotEvidence:
    """Build both preregistered paired comparisons on one public tune scope."""

    genes = _texts(gene_names, "gene_names")
    if len(set(genes)) != len(genes):
        raise ValueError("gene_names must be unique")
    sources = _texts(eligible_sources, "eligible_sources")
    if len(set(sources)) != len(sources) or not set(sources).issubset(genes):
        raise ValueError("eligible_sources must be unique selected genes")
    if direction not in {"k562_to_rpe1", "rpe1_to_k562"}:
        raise ValueError("direction must be a registered cross-environment direction")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative built-in integer")
    if set(comparator_predictions) != set(_COMPARATORS):
        raise ValueError("the causal pilot requires its two frozen comparators")
    if not tuning_edges or any(
        type(edge) is not tuple
        or len(edge) != 2
        or edge[0] not in sources
        or edge[1] not in genes
        or edge[0] == edge[1]
        for edge in tuning_edges
    ):
        raise ValueError("tuning_edges must be eligible directed relations")

    hypersca = normalize_task_c_predictions(hypersca_predictions, genes)
    normalized = {
        comparator: normalize_task_c_predictions(raw, genes)
        for comparator, raw in comparator_predictions.items()
    }
    eligible = frozenset(sources)
    hypersca_ap = _metric(
        hypersca, positives=tuning_edges, eligible_sources=eligible
    )
    metric_records: dict[str, Mapping[str, object]] = {}
    relation_frames: list[pd.DataFrame] = []
    universe = pd.DataFrame(
        [(source, target) for source in genes for target in genes if source != target],
        columns=["source", "target"],
    )
    for comparator in _COMPARATORS:
        other = normalized[comparator]
        comparator_ap = _metric(
            other, positives=tuning_edges, eligible_sources=eligible
        )
        metric_records[comparator] = MappingProxyType(
            {
                "benchmark_id": "causalbench_k562_rpe1",
                "metric_id": "directed_edge_average_precision",
                "direction": direction,
                "seed": seed,
                "hypersca_value": hypersca_ap,
                "comparator_value": comparator_ap,
                "paired_difference": float(hypersca_ap - comparator_ap),
                "eligible_source_count": len(sources),
                "positive_edge_count": len(tuning_edges),
            }
        )
        left = universe.merge(
            hypersca,
            on=["source", "target"],
            how="left",
            validate="one_to_one",
        ).rename(
            columns={
                "score": "hypersca_score",
                "returned_by_method": "hypersca_returned",
            }
        )
        paired = left.merge(
            other,
            on=["source", "target"],
            how="left",
            validate="one_to_one",
        ).rename(
            columns={
                "score": "comparator_score",
                "returned_by_method": "comparator_returned",
            }
        )
        paired.insert(0, "comparator_id", comparator)
        paired["is_reference_edge"] = [
            (source, target) in tuning_edges
            for source, target in paired[["source", "target"]].itertuples(
                index=False, name=None
            )
        ]
        paired["eligible_source"] = paired["source"].isin(eligible)
        relation_frames.append(paired)
    return CausalPilotEvidence(
        metrics=MappingProxyType(metric_records),
        relations=pd.concat(relation_frames, ignore_index=True),
        promotion_eligible=False,
        data_scopes=("train", "tune"),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _plain(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _prediction_table(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.loc[:, ["source", "target", "score"]].copy()


def validate_causal_pilot_public_split(path: Path) -> Mapping[str, object]:
    """Require all model seeds to reuse the same preregistered public split."""

    source = Path(path).resolve(strict=True)

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("public manifest contains a duplicate field")
            result[key] = value
        return result

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("public manifest could not be read strictly") from exc
    if not isinstance(payload, dict):
        raise ValueError("public manifest must contain one JSON object")
    if (
        type(payload.get("seed")) is not int
        or payload.get("seed") != CAUSALBENCH_SPLIT_SEED
        or payload.get("split_id")
        != "C-context-intervention-holdout-v1-seed-11"
    ):
        raise ValueError("causal pilot requires fixed data split seed 11")
    return MappingProxyType(payload)


def run_causalbench_pilot_run(
    *,
    public_manifest_path: Path,
    direction: str,
    output_dir: Path,
    seed: int,
    hypersca_config_path: Path,
    ablation_registry_path: Path,
    protocol_path: Path,
    protocol_identity_value: str,
    device: str,
) -> dict[str, object]:
    """Run one cross-direction public train/tune pilot and publish audit evidence."""

    import torch
    import yaml

    from src.causal.hypersca_c import HyperSCACConfig
    from src.causal.hypersca_c_ablation import (
        fit_hypersca_c_ablation,
        load_hypersca_c_ablations,
    )
    from src.evaluation.task_c_benchmark import score_mean_difference_network
    from src.evaluation.task_c_profile_input import (
        materialize_task_c_profile_input,
        validate_task_c_profile_input,
    )
    from src.evaluation.task_c_tuning import (
        CONTROL_LABEL,
        EXCLUDED_LABEL,
        build_tuning_response_edges,
    )

    if direction not in {"k562_to_rpe1", "rpe1_to_k562"}:
        raise ValueError("direction must be a registered cross-environment direction")
    if type(seed) is not int or seed not in {11, 23, 47}:
        raise ValueError("causal pilot seed must be one of 11, 23, or 47")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if (
        type(protocol_identity_value) is not str
        or len(protocol_identity_value) != 64
        or any(character not in "0123456789abcdef" for character in protocol_identity_value)
    ):
        raise ValueError("protocol identity must be one lowercase SHA-256 value")

    public_manifest = Path(public_manifest_path).resolve(strict=True)
    validate_causal_pilot_public_split(public_manifest)
    config_path = Path(hypersca_config_path).resolve(strict=True)
    registry_path = Path(ablation_registry_path).resolve(strict=True)
    frozen_protocol_path = Path(protocol_path).resolve(strict=True)
    destination = Path(os.path.abspath(os.fspath(Path(output_dir).expanduser())))
    if destination.exists() or destination.is_symlink():
        raise ValueError("causal pilot output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)

    protocol_payload = yaml.safe_load(frozen_protocol_path.read_text(encoding="utf-8"))
    if (
        not isinstance(protocol_payload, dict)
        or protocol_payload.get("protocol_version") != "hypersca-methods-v2.1"
        or protocol_payload.get("execution", {}).get("pilot_scopes") != ["train", "tune"]
        or protocol_payload.get("comparators", {}).get("causal")
        != {
            "confirmatory": "mean_difference",
            "attribution": "hypersca_c_shared_only",
        }
    ):
        raise ValueError("protocol does not contain the frozen causal pilot contract")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = HyperSCACConfig.from_mapping(config_payload)
    ablations = load_hypersca_c_ablations(registry_path)

    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        profiles = {}
        for stage in ("train", "tune"):
            record = materialize_task_c_profile_input(
                public_manifest_path=public_manifest,
                profile="connection",
                condition="cross_environment",
                stage=stage,
                direction=direction,
                output_dir=staging / f"{stage}_profile",
            )
            profiles[stage] = validate_task_c_profile_input(
                input_path=Path(record["input_npz"]),
                profile_manifest_path=Path(record["manifest"]),
                public_manifest_path=public_manifest,
            )
        train = profiles["train"]
        tune = profiles["tune"]
        if train.gene_names != tune.gene_names or train.environment_labels is None:
            raise ValueError("train and tune profiles do not share one cross gene scope")
        contexts = split_hypersca_contexts(
            expression=train.expression,
            interventions=train.interventions,
            gene_names=train.gene_names,
            environment_labels=train.environment_labels,
        )
        primary = fit_hypersca_c_ablation(
            contexts,
            config,
            "primary",
            seed=seed,
            device=device,
            registry=ablations,
        )
        shared = fit_hypersca_c_ablation(
            contexts,
            config,
            "shared_only",
            seed=seed,
            device=device,
            registry=ablations,
        )
        mean = score_mean_difference_network(
            train.expression,
            train.interventions,
            train.gene_names,
            min_cells_per_intervention=5,
        )
        eligible_sources = tuple(
            sorted(
                set(tune.interventions.tolist()) - {CONTROL_LABEL, EXCLUDED_LABEL}
            )
        )
        tuning_edges = build_tuning_response_edges(
            tune.expression,
            tune.interventions,
            tune.gene_names,
            eligible_sources=eligible_sources,
            q_value_threshold=0.1,
        )
        evidence = build_causal_pilot_evidence(
            gene_names=train.gene_names,
            eligible_sources=eligible_sources,
            tuning_edges=tuning_edges,
            hypersca_predictions=_prediction_table(primary.predictions),
            comparator_predictions={
                "mean_difference": _prediction_table(mean.scores),
                "hypersca_c_shared_only": _prediction_table(shared.predictions),
            },
            direction=direction,
            seed=seed,
        )
        prediction_tables = {
            "predictions_hypersca_c.csv": normalize_task_c_predictions(
                _prediction_table(primary.predictions), train.gene_names
            ),
            "predictions_mean_difference.csv": normalize_task_c_predictions(
                _prediction_table(mean.scores), train.gene_names
            ),
            "predictions_hypersca_c_shared_only.csv": normalize_task_c_predictions(
                _prediction_table(shared.predictions), train.gene_names
            ),
        }
        for name, table in prediction_tables.items():
            table.to_csv(staging / name, index=False)
        evidence.relations.to_csv(staging / "relation_scores.csv", index=False)
        metrics = {key: dict(value) for key, value in evidence.metrics.items()}
        pd.DataFrame(metrics.values()).to_csv(
            staging / "primary_metric_units.csv", index=False
        )
        _write_json(staging / "primary_metric_summary.json", metrics)
        _write_json(
            staging / "fit_summary.json",
            {
                "hypersca_c": {
                    "summary": primary.summary,
                    "failures": primary.failures,
                },
                "hypersca_c_shared_only": {
                    "summary": shared.summary,
                    "failures": shared.failures,
                },
            },
        )
        elapsed = max(0.0, time.monotonic() - started)
        _write_json(
            staging / "resource_usage.json",
            {
                "schema_version": "1.0",
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * 1024,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
                ),
            },
        )
        status_record: dict[str, object] = {
            "schema_version": "1.0",
            "status": "completed",
            "benchmark_id": "causalbench_k562_rpe1",
            "direction": direction,
            "seed": seed,
            "data_scopes": ["train", "tune"],
            "promotion_eligible": False,
        }
        _write_json(staging / "method_status.json", status_record)
        _write_json(
            staging / "claim_decision.json",
            {
                "schema_version": "1.0",
                "claim_id": "causal",
                "status": "audit_only",
                "reason": "three-seed public train/tune pilot cannot authorize promotion",
                "promotion_eligible": False,
            },
        )
        # Rebuild the public inputs after fitting so long-running input changes fail.
        for stage in ("train", "tune"):
            validated = profiles[stage]
            validate_task_c_profile_input(
                input_path=validated.input_path,
                profile_manifest_path=validated.manifest_path,
                public_manifest_path=public_manifest,
            )
        code_paths = tuple(
            Path(path).resolve()
            for path in (
                __file__,
                Path(__file__).with_name("task_c_profile_input.py"),
                Path(__file__).parents[1] / "causal/hypersca_c.py",
                Path(__file__).parents[1] / "causal/hypersca_c_ablation.py",
                Path(__file__).parents[1] / "causal/hypersca_c_stability.py",
                Path(__file__).with_name("task_c_benchmark.py"),
                Path(__file__).with_name("task_c_predictions.py"),
                Path(__file__).with_name("task_c_tuning.py"),
            )
        )
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
        artifact_paths = tuple(
            sorted(
                path
                for path in staging.rglob("*")
                if path.is_file() and path.name != "run_manifest.json"
            )
        )
        _write_json(
            staging / "run_manifest.json",
            {
                "schema_version": "1.0",
                "protocol_version": "hypersca-methods-v2.1",
                "protocol_identity": protocol_identity_value,
                "protocol": {
                    "path": str(frozen_protocol_path),
                    "sha256": _sha256_path(frozen_protocol_path),
                },
                "execution_code": {
                    "git_commit": completed.stdout.strip(),
                    "files": {
                        str(path): {"sha256": _sha256_path(path)}
                        for path in code_paths
                    },
                },
                "benchmark_id": "causalbench_k562_rpe1",
                "direction": direction,
                "seed": seed,
                "data_scopes": ["train", "tune"],
                "public_manifest": {
                    "path": str(public_manifest),
                    "sha256": _sha256_path(public_manifest),
                },
                "hypersca_config": {
                    "path": str(config_path),
                    "sha256": _sha256_path(config_path),
                },
                "ablation_registry": {
                    "path": str(registry_path),
                    "sha256": _sha256_path(registry_path),
                },
                "eligible_sources": list(eligible_sources),
                "tuning_edges": [list(edge) for edge in sorted(tuning_edges)],
                "artifacts": {
                    str(path.relative_to(staging)): {"sha256": _sha256_path(path)}
                    for path in artifact_paths
                },
            },
        )
        if destination.exists() or destination.is_symlink():
            raise ValueError("causal pilot output appeared before publication")
        os.rename(staging, destination)
        published = True
        return status_record
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


__all__ = [
    "CausalPilotEvidence",
    "CAUSALBENCH_SPLIT_SEED",
    "build_causal_pilot_evidence",
    "run_causalbench_pilot_run",
    "split_hypersca_contexts",
    "validate_causal_pilot_public_split",
]
