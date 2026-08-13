#!/usr/bin/env python3
"""只用公开调节细胞选择 Task C 方法设置，不读取最终参考关系。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_method_run import (  # noqa: E402
    MAXIMUM_RAW_PREDICTION_BYTES,
    MAXIMUM_RECORD_BYTES,
    MAXIMUM_INPUT_BYTES,
    TaskCMethodRunError,
    _capture_file,
    _capture_public_input,
    _capture_synthetic_input,
    _load_fixed_npz,
    _parse_json,
    _validate_status_seal,
    validate_task_c_method_output_bundle,
)
from src.evaluation.task_c_profile_input import (  # noqa: E402
    TaskCProfileInputError,
    validate_task_c_profile_input,
)
from src.evaluation.task_c_tuning import (  # noqa: E402
    CONTROL_LABEL,
    EXCLUDED_LABEL,
    MAXIMUM_TRIALS,
    TaskCTuningError,
    build_tuning_response_edges,
    load_task_c_tuning_config,
    select_task_c_configuration,
    thaw_task_c_json,
)


_SMOKE_TRIAL_FIELDS = frozenset(
    {"schema_version", "trial_index", "method_id", "condition", "profile", "parameters"}
)
_FORMAL_TRIAL_FIELDS = _SMOKE_TRIAL_FIELDS | frozenset(
    {
        "stage",
        "context_id",
        "direction",
        "seed",
        "training_input_sha256",
        "public_manifest_sha256",
        "profile_manifest_sha256",
        "gene_order_sha256",
    }
)
_MAXIMUM_TRIAL_JSON_BYTES = 1024 * 1024
_CONDITIONS = frozenset({"within_environment", "cross_environment"})
_PROFILES = frozenset({"connection", "comprehensive", "full_public"})
_COMPLETED_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "method_id",
        "status",
        "run_identity_sha256",
        "artifacts",
        "inner_status",
        "status_origin",
        "status_content_sha256",
        "trial_parameters_sha256",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        "method_id",
        "role",
        "source_kind",
        "training_information",
        "output_semantics",
        "data_status",
        "context_id",
        "seed",
        "min_cells",
        "registry_sha256",
        "input",
        "derived_input_manifest",
        "public_manifest",
        "registered_method",
        "code",
        "assets",
        "command",
        "python",
        "run_identity",
        "run_identity_sha256",
        "trial_parameters",
        "selection_record",
    }
)


class _PlainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"无法选择 Task C 设置：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _PlainArgumentParser(
        description=(
            "根据公开调节干预的细胞响应，从最多二十组候选设置中选择一组；"
            "该命令没有最终参考关系或留出集入口。"
        )
    )
    parser.add_argument("--tune-npz", required=True, type=Path, help="公开调节细胞文件。")
    parser.add_argument(
        "--trial-dir",
        required=True,
        action="append",
        type=Path,
        help="一次已完成方法运行及其候选设置记录，可重复但最多二十次。",
    )
    parser.add_argument(
        "--trial-input",
        action="append",
        default=[],
        metavar="TRIAL_DIR=TRAIN_NPZ",
        help="每个正式候选实际读取的公开 train 文件；每个候选恰好提供一次。",
    )
    parser.add_argument(
        "--trial-profile-manifest",
        action="append",
        default=[],
        metavar="TRIAL_DIR=PROFILE_JSON",
        help="候选使用小型派生输入时，对应的 train 来源记录。",
    )
    parser.add_argument(
        "--trial-hypersca-config",
        action="append",
        default=[],
        metavar="TRIAL_DIR=CONFIG_JSON",
        help="HyperSCA-C 候选实际使用的固定设置。",
    )
    parser.add_argument(
        "--trial-gene-list",
        action="append",
        default=[],
        metavar="TRIAL_DIR=GENES_JSON",
        help="HyperSCA-C 候选实际使用的固定基因清单。",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/task_c_methods_v1.json",
        help="产生候选运行时使用的方法清单。",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=ROOT / "results/task_c_method_assets",
        help="外部方法的固定代码和环境证据目录。",
    )
    parser.add_argument("--output-json", required=True, type=Path, help="新的选择记录文件。")
    parser.add_argument(
        "--status-json",
        type=Path,
        help="选择状态；默认写在 output-json 后加 .status.json。",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="固定的调节规则。",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        help="正式公开数据的文件清单；正式选择时必需。",
    )
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        help="若调节文件是公开清单派生的小型版本，提供其来源记录。",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="明确标记仅检查流程的合成数据，不能作为正式结果。",
    )
    return parser


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _snapshot(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    require_single_link: bool = True,
):
    try:
        return _capture_file(
            path,
            label,
            maximum_bytes=maximum_bytes,
            reject_private=True,
            require_single_link=require_single_link,
        )
    except TaskCMethodRunError as exc:
        raise TaskCTuningError(str(exc)) from exc


def _strict_json(snapshot: object, label: str) -> dict[str, Any]:
    try:
        return _parse_json(snapshot, label)  # type: ignore[arg-type]
    except TaskCMethodRunError as exc:
        raise TaskCTuningError(str(exc)) from exc


def _safe_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise TaskCTuningError(f"{label} does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TaskCTuningError(f"{label} must not contain a symbolic link")
    if not absolute.is_dir():
        raise TaskCTuningError(f"{label} must be a directory")
    return absolute


def _path_bindings(values: Sequence[str], label: str) -> dict[Path, Path]:
    bindings: dict[Path, Path] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise TaskCTuningError(f"{label} must use TRIAL_DIR=PATH")
        raw_trial, raw_path = value.split("=", 1)
        if not raw_trial or not raw_path:
            raise TaskCTuningError(f"{label} must use TRIAL_DIR=PATH")
        trial = Path(os.path.abspath(os.fspath(Path(raw_trial).expanduser())))
        path = Path(os.path.abspath(os.fspath(Path(raw_path).expanduser())))
        if trial in bindings:
            raise TaskCTuningError(f"{label} repeats one trial directory")
        bindings[trial] = path
    return bindings


def _verify_method_artifacts(
    root: Path, status: Mapping[str, object]
) -> tuple[object, ...]:
    snapshots: list[object] = []
    observed: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "method_status.json":
            continue
        snapshot = _snapshot(
            path,
            f"trial artifact {relative}",
            maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
        )
        snapshots.append(snapshot)
        observed[relative] = {
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size,
        }
    if status.get("artifacts") != observed:
        raise TaskCTuningError("trial artifact inventory or hash changed")
    return tuple(snapshots)


def _read_predictions(snapshot: object, expected_rows: int) -> pd.DataFrame:
    payload = snapshot.payload  # type: ignore[attr-defined]
    if payload.count(b"\n") != expected_rows + 1:
        raise TaskCTuningError("trial prediction row count differs from the complete universe")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise TaskCTuningError("trial predictions are not a valid bounded CSV table") from exc
    if len(frame) != expected_rows:
        raise TaskCTuningError("trial prediction row count differs from the complete universe")
    return frame


def _trial_record(
    trial_dir: Path,
    *,
    synthetic_smoke: bool,
    expected_rows: int,
    evidence_hashes: Mapping[str, str | None],
    expected_scope: Mapping[str, str | None],
) -> tuple[
    int,
    Mapping[str, object],
    pd.DataFrame,
    dict[str, object],
    tuple[object, ...],
]:
    root = _safe_directory(trial_dir, "trial directory")
    top = {path.name for path in root.iterdir()}
    required = {"predictions.csv", "trial_parameters.json"}
    if not required.issubset(top):
        raise TaskCTuningError("trial directory lacks predictions or parameter record")
    if synthetic_smoke:
        if top != required:
            raise TaskCTuningError("synthetic trial directory must contain exactly two files")
    elif not {"method_status.json", "environment_manifest.json"}.issubset(top):
        raise TaskCTuningError("formal trial needs completed method status and environment evidence")
    elif top not in (
        {
            "method_status.json",
            "environment_manifest.json",
            "trial_parameters.json",
            "raw_predictions.csv",
            "predictions.csv",
        },
        {
            "method_status.json",
            "environment_manifest.json",
            "trial_parameters.json",
            "raw_predictions.csv",
            "predictions.csv",
            "raw_runtime",
        },
        {
            "method_status.json",
            "environment_manifest.json",
            "trial_parameters.json",
            "raw_predictions.csv",
            "predictions.csv",
            "raw_runtime",
            "raw_method_output",
        },
    ):
        raise TaskCTuningError("formal trial has an unrecognized completed result layout")
    prediction_snapshot = _snapshot(
        root / "predictions.csv",
        "trial predictions",
        maximum_bytes=MAXIMUM_RAW_PREDICTION_BYTES,
    )
    parameter_snapshot = _snapshot(
        root / "trial_parameters.json",
        "trial parameters",
        maximum_bytes=_MAXIMUM_TRIAL_JSON_BYTES,
    )
    parameters = _strict_json(parameter_snapshot, "trial parameters")
    expected_fields = _SMOKE_TRIAL_FIELDS if synthetic_smoke else _FORMAL_TRIAL_FIELDS
    if set(parameters) != expected_fields or parameters.get("schema_version") != "1.0":
        raise TaskCTuningError("trial parameter fields changed")
    trial_index = parameters.get("trial_index")
    method = parameters.get("method_id")
    condition = parameters.get("condition")
    profile = parameters.get("profile")
    if (
        isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or trial_index < 0
        or not isinstance(method, str)
        or not method
        or condition not in _CONDITIONS
        or profile not in _PROFILES
        or not isinstance(parameters.get("parameters"), dict)
    ):
        raise TaskCTuningError(
            "trial index, method, condition, profile, or parameter identity is malformed"
        )
    snapshots: list[object] = [prediction_snapshot, parameter_snapshot]
    if not synthetic_smoke:
        stage = parameters.get("stage")
        record_context = parameters.get("context_id")
        direction = parameters.get("direction")
        record_seed = parameters.get("seed")
        training_input_sha256 = parameters.get("training_input_sha256")
        gene_order_sha256 = parameters.get("gene_order_sha256")
        if (
            stage != "train"
            or isinstance(record_seed, bool)
            or not isinstance(record_seed, int)
            or record_seed < 0
            or not isinstance(training_input_sha256, str)
            or not training_input_sha256.startswith("sha256:")
            or training_input_sha256 == evidence_hashes["tune_input_sha256"]
            or gene_order_sha256 != evidence_hashes["gene_order_sha256"]
            or record_context != expected_scope["context_id"]
            or direction != expected_scope["direction"]
            or condition != expected_scope["condition"]
            or profile != expected_scope["profile"]
        ):
            raise TaskCTuningError(
                "trial must use a separate train input with the same genes, context, direction, condition, and profile"
            )
        status_snapshot = _snapshot(
            root / "method_status.json",
            "trial method status",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
        )
        environment_snapshot = _snapshot(
            root / "environment_manifest.json",
            "trial environment record",
            maximum_bytes=MAXIMUM_RECORD_BYTES,
        )
        status = _strict_json(status_snapshot, "trial method status")
        environment = _strict_json(environment_snapshot, "trial environment record")
        try:
            _validate_status_seal(status)
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc
        if set(status) != _COMPLETED_STATUS_FIELDS:
            raise TaskCTuningError("completed method status fields changed")
        if set(environment) != _ENVIRONMENT_FIELDS:
            raise TaskCTuningError("completed method environment fields changed")
        run_identity = environment.get("run_identity")
        if not isinstance(run_identity, dict) or environment.get(
            "run_identity_sha256"
        ) != _sha256(_json_bytes(run_identity)):
            raise TaskCTuningError("completed method run identity changed")
        context_id = environment.get("context_id")
        expected_context = (
            context_id in {"k562", "rpe1"}
            if condition == "within_environment"
            else context_id in {"k562_to_rpe1", "rpe1_to_k562"}
        )
        if (
            environment.get("schema_version") != "1.0"
            or status.get("schema_version") != "1.0"
            or status.get("status") != "completed_standardized_output"
            or status.get("method_id") != method
            or environment.get("method_id") != method
            or environment.get("data_status") != "external_benchmark"
            or not expected_context
            or status.get("run_identity_sha256")
            != environment.get("run_identity_sha256")
        ):
            raise TaskCTuningError("formal trial is not the recorded completed method run")
        environment_trial = environment.get("trial_parameters")
        identity_trial = run_identity.get("trial_parameters")
        environment_input = environment.get("input")
        environment_profile = environment.get("derived_input_manifest")
        environment_public = environment.get("public_manifest")
        if (
            not isinstance(environment_trial, dict)
            or environment_trial.get("sha256") != parameter_snapshot.sha256
            or environment_trial.get("content") != parameters
            or identity_trial != environment_trial
            or environment.get("selection_record") is not None
            or run_identity.get("selection_record") is not None
            or status.get("trial_parameters_sha256") != parameter_snapshot.sha256
            or not isinstance(status.get("artifacts"), dict)
            or status["artifacts"].get("trial_parameters.json")
            != {
                "sha256": parameter_snapshot.sha256,
                "size_bytes": parameter_snapshot.size,
            }
        ):
            raise TaskCTuningError(
                "formal trial parameters were not sealed before the method run"
            )
        if (
            not isinstance(environment_input, dict)
            or environment_input.get("sha256") != training_input_sha256
            or run_identity.get("input_sha256") != training_input_sha256
            or parameters.get("public_manifest_sha256")
            != evidence_hashes["public_manifest_sha256"]
            or not isinstance(environment_public, dict)
            or environment_public.get("sha256")
            != parameters.get("public_manifest_sha256")
            or run_identity.get("public_manifest_sha256")
            != parameters.get("public_manifest_sha256")
            or (
                environment_profile.get("sha256")
                if isinstance(environment_profile, dict)
                else None
            )
            != parameters.get("profile_manifest_sha256")
            or run_identity.get("derived_input_manifest_sha256")
            != parameters.get("profile_manifest_sha256")
            or (
                parameters.get("profile_manifest_sha256")
                == evidence_hashes["profile_manifest_sha256"]
                and parameters.get("profile_manifest_sha256") is not None
            )
        ):
            raise TaskCTuningError(
                "trial training evidence is not separate from the tuning evidence"
            )
        snapshots.extend((status_snapshot, environment_snapshot))
        artifacts = status.get("artifacts")
        if not isinstance(artifacts, dict) or "predictions.csv" not in artifacts:
            raise TaskCTuningError("formal trial status lacks standardized predictions")
        predicted = artifacts["predictions.csv"]
        if (
            not isinstance(predicted, dict)
            or predicted.get("sha256") != prediction_snapshot.sha256
            or predicted.get("size_bytes") != prediction_snapshot.size
        ):
            raise TaskCTuningError("formal trial predictions differ from completed run evidence")
        snapshots.extend(_verify_method_artifacts(root, status))
    frame = _read_predictions(prediction_snapshot, expected_rows)
    identity = {
        "method_id": method,
        "condition": condition,
        "profile": profile,
        "context_id": parameters.get("context_id") if not synthetic_smoke else None,
        "direction": parameters.get("direction") if not synthetic_smoke else None,
        "seed": parameters.get("seed") if not synthetic_smoke else None,
        "training_input_sha256": (
            parameters.get("training_input_sha256") if not synthetic_smoke else None
        ),
        "training_profile_manifest_sha256": (
            parameters.get("profile_manifest_sha256") if not synthetic_smoke else None
        ),
        "gene_order_sha256": (
            parameters.get("gene_order_sha256") if not synthetic_smoke else None
        ),
        "trial_index": trial_index,
        "parameters_sha256": parameter_snapshot.sha256,
        "predictions_sha256": prediction_snapshot.sha256,
        "run_identity_sha256": (
            environment.get("run_identity_sha256") if not synthetic_smoke else None
        ),
        "artifacts": status.get("artifacts") if not synthetic_smoke else None,
        "trial_directory": str(root),
    }
    return (
        trial_index,
        parameters["parameters"],
        frame,
        identity,
        tuple(snapshots),
    )


def _load_tune_input(args: argparse.Namespace):
    if args.synthetic_smoke:
        if args.public_manifest is not None or args.profile_manifest is not None:
            raise TaskCTuningError(
                "synthetic smoke data must not use a public or profile manifest"
            )
        try:
            input_snapshot = _capture_synthetic_input(args.tune_npz)
            expression, labels, genes, environments = _load_fixed_npz(input_snapshot)
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc
        return (
            expression,
            labels,
            genes,
            frozenset(set(labels) - {CONTROL_LABEL, EXCLUDED_LABEL}),
            {
                "tune_input_sha256": input_snapshot.sha256,
                "public_manifest_sha256": None,
                "profile_manifest_sha256": None,
                "gene_order_sha256": None,
            },
            (input_snapshot,),
            "synthetic_smoke",
            {
                "condition": "synthetic_smoke",
                "profile": "synthetic_smoke",
                "stage": "synthetic_smoke",
                "context_id": "synthetic",
                "direction": None,
            },
        )
    if args.public_manifest is None:
        raise TaskCTuningError("formal selection requires the matching public manifest")
    if args.profile_manifest is None:
        try:
            input_snapshot, public_snapshot, public, relative = _capture_public_input(
                args.tune_npz, args.public_manifest
            )
            expression, labels, genes, environments = _load_fixed_npz(input_snapshot)
        except TaskCMethodRunError as exc:
            raise TaskCTuningError(str(exc)) from exc
        if not (
            relative.startswith("within/") and relative.endswith("/tune.npz")
        ):
            raise TaskCTuningError("formal tuning input must be the registered tune partition")
        if public.get("min_cells_per_intervention", 0) < 5:
            raise TaskCTuningError("public tuning groups must keep at least five cells")
        sources = frozenset(public["tune_sources"])
        if set(labels) - {CONTROL_LABEL, EXCLUDED_LABEL} != set(sources):
            raise TaskCTuningError(
                "registered tune cells must contain exactly the public tune sources"
            )
        snapshots = (input_snapshot, public_snapshot)
        profile_hash = None
        context = Path(relative).parts[1]
        scope = {
            "condition": "within_environment",
            "profile": "full_public",
            "stage": "tune",
            "context_id": context,
            "direction": None,
        }
    else:
        try:
            validated = validate_task_c_profile_input(
                input_path=args.tune_npz,
                profile_manifest_path=args.profile_manifest,
                public_manifest_path=args.public_manifest,
            )
        except TaskCProfileInputError as exc:
            raise TaskCTuningError(str(exc)) from exc
        if validated.manifest.get("stage") != "tune":
            raise TaskCTuningError("formal profile input must record the tune stage")
        expression = validated.expression
        labels = validated.interventions
        genes = validated.gene_names
        sources = frozenset(set(labels) - {CONTROL_LABEL, EXCLUDED_LABEL})
        input_snapshot = validated.input_snapshot
        public_snapshot = validated.public_snapshot
        snapshots = (
            input_snapshot,
            validated.manifest_snapshot,
            public_snapshot,
            *validated.parent_snapshots,
        )
        profile_hash = validated.manifest_sha256
        scope = {
            "condition": validated.condition,
            "profile": validated.profile,
            "stage": validated.stage,
            "context_id": validated.direction or validated.context_id,
            "direction": validated.direction,
        }
    if not sources.issubset(set(labels)):
        raise TaskCTuningError("public tune sources differ from the tuning cell labels")
    return (
        expression,
        labels,
        genes,
        sources,
        {
            "tune_input_sha256": input_snapshot.sha256,
            "public_manifest_sha256": public_snapshot.sha256,
            "profile_manifest_sha256": profile_hash,
            "gene_order_sha256": _sha256(
                json.dumps(
                    list(genes), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ),
        },
        snapshots,
        "external_benchmark",
        scope,
    )


def _verify_snapshots(snapshots: Sequence[object]) -> None:
    for index, snapshot in enumerate(snapshots):
        label = f"fixed tuning input {index + 1}"
        current = _snapshot(
            Path(getattr(snapshot, "path")),
            label,
            maximum_bytes=max(1, int(getattr(snapshot, "size"))),
            require_single_link=False,
        )
        attributes = (
            "path",
            "payload",
            "device",
            "inode",
            "size",
            "modified_ns",
            "changed_ns",
            "link_count",
        )
        if any(getattr(current, name) != getattr(snapshot, name) for name in attributes):
            raise TaskCTuningError(f"{label} changed during configuration selection")


def _publish_json(path: Path, payload: bytes) -> None:
    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    parent = destination.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _safe_directory(parent, "output parent")
    if destination.exists() or destination.is_symlink():
        raise TaskCTuningError("output JSON already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise TaskCTuningError("output JSON already exists") from exc
        temporary.unlink()
        published = True
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            temporary.unlink(missing_ok=True)


def _selection_failure_category(message: str) -> str:
    lowered = message.lower()
    if "positive" in lowered or "relation" in lowered and "tuning" in lowered:
        return "no_public_tuning_relations"
    if "trial" in lowered or "candidate" in lowered or "prediction" in lowered:
        return "invalid_trial_bundle"
    return "invalid_tuning_input"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    status_path = args.status_json or Path(f"{args.output_json}.status.json")
    failure_condition: str | None = None
    failure_tune_sha256: str | None = None
    try:
        if status_path.exists() or status_path.is_symlink():
            raise TaskCTuningError("selection status JSON already exists")
        if len(args.trial_dir) > MAXIMUM_TRIALS:
            raise TaskCTuningError("configuration selection allows at most twenty trials")
        config_snapshot = _snapshot(
            args.config, "tuning configuration", maximum_bytes=MAXIMUM_RECORD_BYTES
        )
        config = load_task_c_tuning_config(config_snapshot.path)
        (
            expression,
            labels,
            genes,
            sources,
            evidence_hashes,
            tune_snapshots,
            data_status,
            tune_scope,
        ) = _load_tune_input(args)
        failure_condition = str(tune_scope["condition"])
        failure_tune_sha256 = evidence_hashes["tune_input_sha256"]
        tuning_edges = build_tuning_response_edges(
            expression,
            labels,
            genes,
            eligible_sources=sources,
            q_value_threshold=config.q_value_threshold,
        )
        trials = []
        trial_snapshots: list[object] = []
        identities: list[dict[str, object]] = []
        trial_inputs = _path_bindings(args.trial_input, "trial input binding")
        trial_profiles = _path_bindings(
            args.trial_profile_manifest, "trial profile manifest binding"
        )
        trial_configs = _path_bindings(
            args.trial_hypersca_config, "trial HyperSCA-C config binding"
        )
        trial_genes = _path_bindings(
            args.trial_gene_list, "trial gene-list binding"
        )
        expected_trial_roots = {
            Path(os.path.abspath(os.fspath(path.expanduser()))) for path in args.trial_dir
        }
        for bindings, label in (
            (trial_inputs, "trial input"),
            (trial_profiles, "trial profile manifest"),
            (trial_configs, "trial HyperSCA-C config"),
            (trial_genes, "trial gene list"),
        ):
            if not set(bindings) <= expected_trial_roots:
                raise TaskCTuningError(f"{label} binding names an unknown trial")
        if not args.synthetic_smoke and set(trial_inputs) != expected_trial_roots:
            raise TaskCTuningError(
                "formal selection requires exactly one actual train input for every trial"
            )
        if args.synthetic_smoke and any(
            (trial_inputs, trial_profiles, trial_configs, trial_genes)
        ):
            raise TaskCTuningError("synthetic smoke trials cannot use formal input bindings")
        expected_rows = len(genes) * (len(genes) - 1)
        for trial_dir in args.trial_dir:
            trial_root = Path(
                os.path.abspath(os.fspath(Path(trial_dir).expanduser()))
            )
            if not args.synthetic_smoke:
                try:
                    validate_task_c_method_output_bundle(
                        output_dir=trial_root,
                        input_npz=trial_inputs[trial_root],
                        registry_path=args.registry,
                        asset_root=args.asset_root,
                        public_manifest_path=args.public_manifest,
                        derived_input_manifest_path=trial_profiles.get(trial_root),
                        hypersca_config_path=trial_configs.get(trial_root),
                        gene_list_path=trial_genes.get(trial_root),
                        project_root=ROOT,
                    )
                except TaskCMethodRunError as exc:
                    raise TaskCTuningError(
                        f"trial bundle failed reconstruction from its actual train input: {exc}"
                    ) from exc
            index, parameters, frame, identity, snapshots = _trial_record(
                trial_dir,
                synthetic_smoke=args.synthetic_smoke,
                expected_rows=expected_rows,
                evidence_hashes=evidence_hashes,
                expected_scope=tune_scope,
            )
            trials.append((index, parameters, frame))
            if not args.synthetic_smoke:
                train_snapshot = _snapshot(
                    trial_inputs[trial_root],
                    "actual trial train input",
                    maximum_bytes=MAXIMUM_INPUT_BYTES,
                    require_single_link=False,
                )
                trial_snapshots.append(train_snapshot)
                identity["training_input_binding_sha256"] = train_snapshot.sha256
                profile_path = trial_profiles.get(trial_root)
                if profile_path is None:
                    identity["training_profile_binding_sha256"] = None
                else:
                    profile_snapshot = _snapshot(
                        profile_path,
                        "actual trial profile manifest",
                        maximum_bytes=MAXIMUM_RECORD_BYTES,
                    )
                    trial_snapshots.append(profile_snapshot)
                    identity["training_profile_binding_sha256"] = profile_snapshot.sha256
            identities.append(identity)
            trial_snapshots.extend(snapshots)
        shared_identity_fields = (
            "method_id",
            "condition",
            "profile",
            "context_id",
            "direction",
            "seed",
            "gene_order_sha256",
        )
        if any(
            any(identity[field] != identities[0][field] for field in shared_identity_fields)
            for identity in identities[1:]
        ):
            raise TaskCTuningError(
                "all trials must use the same method, genes, context, condition, profile, direction, and seed"
            )
        selection = select_task_c_configuration(
            trials,
            tuning_edges=tuning_edges,
            maximum_trials=config.maximum_trials_per_method,
            gene_names=genes,
        )
        trial_metrics = [
            {
                "trial_index": trial_index,
                "average_precision": float(
                    select_task_c_configuration(
                        [(trial_index, parameters, frame)],
                        tuning_edges=tuning_edges,
                        maximum_trials=config.maximum_trials_per_method,
                        gene_names=genes,
                    )["average_precision"]
                ),
                "parameters": thaw_task_c_json(parameters),
            }
            for trial_index, parameters, frame in trials
        ]
        code_snapshots = tuple(
            _snapshot(path, "tuning code", maximum_bytes=MAXIMUM_RECORD_BYTES)
            for path in (
                ROOT / "src/evaluation/task_c_tuning.py",
                ROOT / "scripts/select_task_c_configuration.py",
            )
        )
        code_identity = _sha256(
            b"".join(snapshot.payload for snapshot in code_snapshots)
        )
        result = thaw_task_c_json(dict(selection))
        assert isinstance(result, dict)
        result["method_id"] = identities[0]["method_id"]
        result.update(tune_scope)
        result["training_and_tuning_inputs_separate"] = True
        training_inputs = sorted(
            {str(identity["training_input_sha256"]) for identity in identities}
        )
        training_profiles = sorted(
            {
                str(identity["training_profile_manifest_sha256"])
                for identity in identities
                if identity["training_profile_manifest_sha256"] is not None
            }
        )
        result["evidence"] = {
            "data_status": data_status,
            **evidence_hashes,
            "config_sha256": config_snapshot.sha256,
            "config": _strict_json(config_snapshot, "tuning configuration"),
            "code_sha256": code_identity,
            "tuning_positive_relation_count": len(tuning_edges),
            "tuning_edges": [list(edge) for edge in sorted(tuning_edges)],
            "tuning_edges_sha256": _sha256(
                _json_bytes([list(edge) for edge in sorted(tuning_edges)])
            ),
            "gene_count": len(genes),
            "training_input_sha256s": training_inputs,
            "training_profile_manifest_sha256s": training_profiles,
            "trials": sorted(identities, key=lambda item: int(item["trial_index"])),
            "trial_metrics": sorted(
                trial_metrics, key=lambda item: int(item["trial_index"])
            ),
        }
        result["selection_record_sha256"] = _sha256(_json_bytes(result))
        all_snapshots = (
            config_snapshot,
            *tune_snapshots,
            *trial_snapshots,
            *code_snapshots,
        )
        _verify_snapshots(all_snapshots)
        _publish_json(args.output_json, _json_bytes(result))
        _publish_json(
            status_path,
            _json_bytes(
                {
                    "schema_version": "1.0",
                    "status": "completed_selection",
                    "condition": result["condition"],
                    "tune_input_sha256": result["evidence"]["tune_input_sha256"],
                    "selection_record_sha256": result["selection_record_sha256"],
                    "reason_category": None,
                    "reason": None,
                }
            ),
        )
    except (
        TaskCTuningError,
        TaskCMethodRunError,
        TaskCProfileInputError,
        OSError,
        UnicodeError,
    ) as exc:
        if not status_path.exists() and not status_path.is_symlink():
            tune_sha256 = failure_tune_sha256
            try:
                tune_sha256 = _snapshot(
                    args.tune_npz,
                    "failed tuning input",
                    maximum_bytes=MAXIMUM_INPUT_BYTES,
                ).sha256
            except Exception:
                pass
            try:
                _publish_json(
                    status_path,
                    _json_bytes(
                        {
                            "schema_version": "1.0",
                            "status": "failed_selection",
                            "condition": failure_condition,
                            "tune_input_sha256": tune_sha256,
                            "selection_record_sha256": None,
                            "reason_category": _selection_failure_category(str(exc)),
                            "reason": str(exc),
                        }
                    ),
                )
            except Exception:
                pass
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
