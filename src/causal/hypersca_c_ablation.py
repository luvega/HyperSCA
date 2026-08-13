"""预先登记并安全运行 HyperSCA-C 候选贡献消融。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.causal import hypersca_c as _core
from src.causal import hypersca_c_run as _run
from src.causal.hypersca_c import (
    HyperSCACConfig,
    HyperSCACContext,
    HyperSCACError,
    fit_hypersca_c_once,
)
from src.causal.hypersca_c_stability import (
    HyperSCAStabilityResult,
    build_stability_table,
    fit_stable_hypersca_c,
    stratified_bootstrap_context,
)
from src.evaluation.task_c_data import (
    load_task_c_dataset_from_verified_bytes,
    sha256_path,
    write_json,
)


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REGISTRY = _ROOT / "configs/hypersca_c_ablations_v1.json"
ABLATION_IDS = (
    "primary",
    "shared_only",
    "separate_contexts",
    "observational_only",
    "no_stability_weighting",
    "acyclicity_off",
    "acyclicity_strong",
    "prior_on_secondary",
)
_EXPECTED_ABLATIONS: dict[str, dict[str, object]] = {
    "primary": {"mode": "joint", "configuration_changes": {}},
    "shared_only": {
        "mode": "joint",
        "configuration_changes": {"enable_context_adjustments": False},
    },
    "separate_contexts": {
        "mode": "one_context_per_fit",
        "configuration_changes": {},
    },
    "observational_only": {
        "mode": "control_cells_only",
        "configuration_changes": {},
    },
    "no_stability_weighting": {
        "mode": "joint",
        "configuration_changes": {"bootstrap_repeats": 1},
    },
    "acyclicity_off": {
        "mode": "joint",
        "configuration_changes": {"acyclicity_weight": 0.0},
    },
    "acyclicity_strong": {
        "mode": "joint",
        "configuration_changes": {"acyclicity_weight": 0.1},
    },
    "prior_on_secondary": {
        "mode": "joint_with_external_prior",
        "configuration_changes": {"prior_discount": 0.5},
    },
}
_PRIOR_MANIFEST_FIELDS = {
    "schema_version",
    "prior_id",
    "source_uri",
    "source_description",
    "prior_edges_sha256",
    "relation_fingerprint_schema",
    "relation_fingerprint",
    "scoring_reference_fingerprints",
    "independence_attestation",
}
_RELATION_FINGERPRINT_SCHEMA = "directed_edge_set_v1"
_ITEM_ARTIFACTS = {
    "raw_predictions.csv",
    "fit_summary.json",
    "method_status.json",
    "run_manifest.json",
}
_BATCH_MANIFEST = "ablation_batch_manifest.json"
_COMPLETED_STATUS_FIELDS = {
    "schema_version",
    "method_id",
    "claim_level",
    "formal_score_status",
    "ablation_id",
    "ablation_mode",
    "configuration_changes",
    "interpretation",
    "seed",
    "contexts",
    "condition",
    "condition_mode",
    "direction",
    "stage",
    "status",
    "reason",
    "requested_bootstraps",
    "successful_bootstraps",
    "failure_count",
    "failures",
    "coverage",
    "usable_for_ranking",
}
_FAILURE_SUMMARY_FIELDS = {
    "schema_version",
    "ablation_id",
    "status",
    "reason",
    "failures",
}
_ITEM_MANIFEST_FIELDS = {
    "schema_version",
    "method_id",
    "status",
    "claim_level",
    "formal_score_status",
    "ablation_id",
    "run_identity",
    "started_utc",
    "completed_utc",
    "duration_seconds",
    "artifacts",
    "run_manifest_content_sha256",
}
_ITEM_IDENTITY_FIELDS = {
    "batch_run_identity",
    "ablation_id",
    "ablation_mode",
    "configuration_changes",
    "effective_config",
    "registered_prior",
}
_BATCH_MANIFEST_FIELDS = {
    "schema_version",
    "method_id",
    "claim_level",
    "formal_score_status",
    "ablation_order",
    "run_identity",
    "ablations",
    "started_utc",
    "completed_utc",
    "duration_seconds",
    "batch_manifest_content_sha256",
}
_BATCH_ITEM_FIELDS = {
    "ablation_id",
    "status",
    "reason",
    "usable_for_ranking",
    "output_directory",
    "run_manifest_sha256",
}


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _deep_freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _canonical_compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_registry_with_snapshot(
    path: str | Path,
) -> tuple[Mapping[str, Mapping[str, object]], Path, _run._FileSnapshot]:
    absolute = _run._regular_file(
        Path(path), "HyperSCA-C 消融登记表", reject_symlink=True
    )
    payload, snapshot = _run._read_strict_json_snapshot(
        absolute, "HyperSCA-C 消融登记表"
    )
    if set(payload) != {"schema_version", "ablations"}:
        raise HyperSCACError("消融登记表字段不符合固定格式")
    if payload["schema_version"] != "1.0":
        raise HyperSCACError("消融登记表 schema_version 必须是 1.0")
    ablations = payload["ablations"]
    if not isinstance(ablations, dict) or tuple(ablations) != ABLATION_IDS:
        raise HyperSCACError("消融登记表必须保留八项固定登记顺序")
    if _canonical_compact(ablations) != _canonical_compact(_EXPECTED_ABLATIONS):
        raise HyperSCACError("消融登记表的固定模式或设置发生了变化")
    frozen = _deep_freeze(ablations)
    assert isinstance(frozen, Mapping)
    return frozen, absolute, snapshot  # type: ignore[return-value]


def load_hypersca_c_ablations(
    path: str | Path,
) -> Mapping[str, Mapping[str, object]]:
    """严格读取固定八项消融，并返回不能原地改写的登记内容。"""

    registry, _, _ = _load_registry_with_snapshot(path)
    return registry


def apply_hypersca_c_ablation(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    ablation_id: str,
    *,
    registry: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[tuple[HyperSCACContext, ...], HyperSCACConfig]:
    """按登记内容改变数据范围或设置，同时保留原始输入。"""

    if not isinstance(config, HyperSCACConfig):
        raise HyperSCACError("config 必须是已经核验的 HyperSCACConfig")
    if isinstance(contexts, (str, bytes)):
        raise HyperSCACError("contexts 必须是细胞环境序列")
    original_contexts = tuple(contexts)
    if not original_contexts or not all(
        isinstance(context, HyperSCACContext) for context in original_contexts
    ):
        raise HyperSCACError("contexts 必须含有至少一个已核验细胞环境")
    chosen_registry = registry or load_hypersca_c_ablations(_DEFAULT_REGISTRY)
    if ablation_id not in chosen_registry:
        raise HyperSCACError(f"未登记的 HyperSCA-C 消融：{ablation_id}")
    spec = chosen_registry[ablation_id]
    transformed = original_contexts
    if spec["mode"] == "control_cells_only":
        control_only: list[HyperSCACContext] = []
        for context in original_contexts:
            selected = context.interventions == config.control_label
            control_only.append(
                HyperSCACContext(
                    context_id=context.context_id,
                    expression=context.expression[selected],
                    interventions=context.interventions[selected],
                    gene_names=context.gene_names,
                )
            )
        transformed = tuple(control_only)
    return transformed, _effective_ablation_config(config, spec)


def _effective_ablation_config(
    config: HyperSCACConfig, spec: Mapping[str, object]
) -> HyperSCACConfig:
    values = asdict(config)
    changes = spec["configuration_changes"]
    assert isinstance(changes, Mapping)
    values.update(dict(changes))
    return HyperSCACConfig.from_mapping(values)


def _source_variance(
    contexts: Sequence[HyperSCACContext], config: HyperSCACConfig
) -> dict[str, float]:
    variances: list[np.ndarray] = []
    for context in contexts:
        controls = context.interventions == config.control_label
        if int(controls.sum()) < 2:
            raise HyperSCACError(f"细胞环境 {context.context_id} 至少需要两个对照细胞")
        variances.append(
            context.expression[controls]
            .astype(np.float64, copy=False)
            .var(axis=0, ddof=0)
        )
    return {
        gene: float(np.mean([values[index] for values in variances]))
        for index, gene in enumerate(contexts[0].gene_names)
    }


def _compact_failure(error: HyperSCACError) -> str:
    message = " ".join(str(error).split()) or error.__class__.__name__
    if len(message) > 200:
        message = message[:197] + "..."
    return message


def fit_separate_contexts_stable(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None = None,
) -> HyperSCAStabilityResult:
    """每个细胞背景独立拟合，再按相同稳定性规则合并完整关系表。"""

    normalized_contexts, genes = _core._validated_contexts(contexts)
    normalized_seed = _core._validated_seed(seed)
    target_device = _core._validated_device(device)
    _core._prior_weights(
        prior_mask,
        dimension=len(genes),
        prior_discount=config.prior_discount,
        device=target_device,
    )
    if normalized_seed + config.bootstrap_repeats - 1 > 2**64 - 1:
        raise HyperSCACError("seed 与重复次数之和超出支持范围")

    matrices: list[Mapping[str, np.ndarray]] = []
    failures: list[str] = []
    for repeat in range(config.bootstrap_repeats):
        repeat_seed = normalized_seed + repeat
        rng = np.random.default_rng(repeat_seed)
        sampled = [
            stratified_bootstrap_context(context, rng)
            for context in normalized_contexts
        ]
        current: dict[str, np.ndarray] = {}
        context_failures: list[str] = []
        for context in sampled:
            try:
                fitted = fit_hypersca_c_once(
                    (context,),
                    config,
                    seed=repeat_seed,
                    device=device,
                    prior_mask=prior_mask,
                )
                current[context.context_id] = fitted.context_adjacencies[
                    context.context_id
                ]
            except HyperSCACError as exc:
                context_failures.append(f"{context.context_id}:{_compact_failure(exc)}")
        if context_failures:
            failures.append(f"repeat_{repeat}:{';'.join(context_failures)}")
            continue
        matrices.append(current)

    predictions, summary = build_stability_table(
        matrices,
        genes,
        selection_threshold=config.selection_threshold,
        requested_repeats=config.bootstrap_repeats,
        minimum_success_fraction=config.bootstrap_success_fraction,
        source_variance=_source_variance(normalized_contexts, config),
        minimum_source_variance=config.minimum_source_variance,
        expected_contexts=tuple(context.context_id for context in normalized_contexts),
    )
    return HyperSCAStabilityResult(
        predictions=predictions,
        summary=summary,
        failures=tuple(failures),
    )


def fit_hypersca_c_ablation(
    contexts: Sequence[HyperSCACContext],
    config: HyperSCACConfig,
    ablation_id: str,
    *,
    seed: int,
    device: str,
    prior_mask: np.ndarray | None = None,
    registry: Mapping[str, Mapping[str, object]] | None = None,
) -> HyperSCAStabilityResult:
    """运行一个已登记消融；主分析沿用完全相同的稳定性拟合入口。"""

    chosen_registry = registry or load_hypersca_c_ablations(_DEFAULT_REGISTRY)
    transformed, transformed_config = apply_hypersca_c_ablation(
        contexts,
        config,
        ablation_id,
        registry=chosen_registry,
    )
    mode = chosen_registry[ablation_id]["mode"]
    if ablation_id != "prior_on_secondary" and prior_mask is not None:
        raise HyperSCACError("只有 prior_on_secondary 可以读取外部先验")
    if ablation_id == "prior_on_secondary" and prior_mask is None:
        raise HyperSCACError("prior_on_secondary 缺少独立登记先验")
    if mode == "one_context_per_fit":
        return fit_separate_contexts_stable(
            transformed,
            transformed_config,
            seed=seed,
            device=device,
            prior_mask=prior_mask,
        )
    return fit_stable_hypersca_c(
        transformed,
        transformed_config,
        seed=seed,
        device=device,
        prior_mask=prior_mask,
    )


def _strict_csv_rows(raw_bytes: bytes, path: Path) -> list[tuple[str, str]]:
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames != ["source", "target"]:
            raise HyperSCACError("先验关系 CSV 必须恰好包含 source,target 两列")
        rows: list[tuple[str, str]] = []
        for row in reader:
            if None in row or set(row) != {"source", "target"}:
                raise HyperSCACError("先验关系 CSV 的列数不符合固定格式")
            source = _run._required_no_whitespace(row["source"], "先验 source")
            target = _run._required_no_whitespace(row["target"], "先验 target")
            rows.append((source, target))
    except HyperSCACError:
        raise
    except (UnicodeError, csv.Error) as exc:
        raise HyperSCACError(f"无法严格读取先验关系 CSV：{path}") from exc
    if not rows:
        raise HyperSCACError("先验关系 CSV 至少需要一条关系")
    if len(set(rows)) != len(rows):
        raise HyperSCACError("先验关系 CSV 不能含有重复关系")
    return rows


def _directed_relation_fingerprint(rows: Sequence[tuple[str, str]]) -> str:
    """生成与 CSV 排列无关的有向关系集合指纹。"""

    ordered = sorted(
        set(rows), key=lambda edge: (edge[0].encode("utf-8"), edge[1].encode("utf-8"))
    )
    canonical = _RELATION_FINGERPRINT_SCHEMA.encode("ascii") + b"\0"
    canonical += json.dumps(
        [[source, target] for source, target in ordered],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_registered_prior(
    prior_edges_path: str | Path,
    prior_source_manifest_path: str | Path,
    gene_names: Sequence[str],
) -> tuple[np.ndarray, Mapping[str, object], tuple[_run._FileSnapshot, ...]]:
    """只接受来源明确且不复用任务 C 评分关系的外部先验。"""

    genes = tuple(gene_names)
    if len(genes) < 2 or len(set(genes)) != len(genes):
        raise HyperSCACError("先验核验需要至少两个不重复的有序基因")
    edges_path = _run._regular_file(
        Path(prior_edges_path), "外部先验关系", reject_symlink=True
    )
    manifest_path = _run._regular_file(
        Path(prior_source_manifest_path), "外部先验来源清单", reject_symlink=True
    )
    manifest, manifest_snapshot = _run._read_strict_json_snapshot(
        manifest_path, "外部先验来源清单"
    )
    if set(manifest) != _PRIOR_MANIFEST_FIELDS:
        raise HyperSCACError("外部先验来源清单字段不符合固定格式")
    if manifest["schema_version"] != "1.0":
        raise HyperSCACError("外部先验来源清单 schema_version 必须是 1.0")
    prior_id = _run._required_no_whitespace(manifest["prior_id"], "prior_id")
    source_uri = _run._required_plain_text(manifest["source_uri"], "source_uri")
    source_description = _run._required_plain_text(
        manifest["source_description"], "source_description"
    )
    edges_snapshot, raw_bytes = _run._capture_file_snapshot(
        edges_path, "外部先验关系", collect_bytes=True
    )
    assert raw_bytes is not None
    rows = _strict_csv_rows(raw_bytes, edges_path)
    registered_sha = _run._sha256_text(
        manifest["prior_edges_sha256"], "prior_edges_sha256"
    )
    if manifest["relation_fingerprint_schema"] != _RELATION_FINGERPRINT_SCHEMA:
        raise HyperSCACError("外部先验关系指纹规则不符合固定格式")
    relation_fingerprint = _run._sha256_text(
        manifest["relation_fingerprint"], "relation_fingerprint"
    )
    expected_relation_fingerprint = _directed_relation_fingerprint(rows)
    if registered_sha != edges_snapshot.sha256 or relation_fingerprint != (
        expected_relation_fingerprint
    ):
        raise HyperSCACError(
            "外部先验关系 SHA 或关系指纹与来源清单不一致，无法排除重叠或重用"
        )

    references = manifest["scoring_reference_fingerprints"]
    expected_references = {"pooled_essentiality", "chip_directional_reference"}
    if not isinstance(references, dict) or set(references) != expected_references:
        raise HyperSCACError("来源清单必须完整登记两类任务 C 评分参考指纹")
    reference_hashes = {
        name: _run._sha256_text(value, f"评分参考指纹 {name}")
        for name, value in references.items()
    }
    attestation = manifest["independence_attestation"]
    expected_attestation = {
        "reuses_pooled_essentiality",
        "reuses_chip_directional_reference",
    }
    if not isinstance(attestation, dict) or set(attestation) != expected_attestation:
        raise HyperSCACError("来源清单缺少先验独立性声明")
    if any(type(attestation[name]) is not bool for name in expected_attestation):
        raise HyperSCACError("先验独立性声明必须使用真假值")
    if any(bool(attestation[name]) for name in expected_attestation) or (
        relation_fingerprint in set(reference_hashes.values())
    ):
        raise HyperSCACError("外部先验重用或重叠了任务 C 评分参考关系")

    gene_index = {gene: index for index, gene in enumerate(genes)}
    mask = np.zeros((len(genes), len(genes)), dtype=np.int8)
    for source, target in rows:
        if source not in gene_index or target not in gene_index:
            raise HyperSCACError("先验关系含有当前有序基因清单之外的基因")
        if source == target:
            raise HyperSCACError("先验关系不能含有基因自身关系")
        mask[gene_index[source], gene_index[target]] = 1
    immutable_mask = np.frombuffer(mask.tobytes(order="C"), dtype=mask.dtype).reshape(
        mask.shape
    )
    record = MappingProxyType(
        {
            "prior_id": prior_id,
            "source_uri": source_uri,
            "source_description": source_description,
            "prior_edges_path": str(edges_path),
            "prior_source_manifest_path": str(manifest_path),
            "prior_edges_sha256": edges_snapshot.sha256,
            "prior_source_manifest_sha256": manifest_snapshot.sha256,
            "relation_fingerprint_schema": _RELATION_FINGERPRINT_SCHEMA,
            "relation_fingerprint": relation_fingerprint,
            "scoring_reference_fingerprints": MappingProxyType(reference_hashes),
            "edge_count": len(rows),
        }
    )
    return immutable_mask, record, (edges_snapshot, manifest_snapshot)


@dataclass(frozen=True)
class _PreparedBatch:
    contexts: tuple[HyperSCACContext, ...]
    context_records: tuple[Mapping[str, object], ...]
    context_snapshots: tuple[tuple[str, _run._FileSnapshot, str], ...]
    config: HyperSCACConfig
    config_values: Mapping[str, object]
    config_path: Path
    config_snapshot: _run._FileSnapshot
    gene_selection: Mapping[str, object]
    gene_path: Path
    gene_snapshot: _run._FileSnapshot
    public_manifest_path: Path
    public_manifest_snapshot: _run._FileSnapshot
    public_files: Mapping[str, str]
    public_inventory: Mapping[str, _run._FileSnapshot]
    condition: Mapping[str, object]
    seed: int
    device: str
    code: Mapping[str, object]


def _prepare_batch_inputs(
    *,
    context_values: Sequence[str],
    config_path: Path,
    gene_list_path: Path,
    public_manifest_path: Path,
    seed: int,
    device: str,
) -> _PreparedBatch:
    """复用主运行的完整公开文件、语义、基因上限和代码身份检查。"""

    parsed_contexts = _run._parse_context_values(context_values)
    gene_path, gene_selection, gene_snapshot = _run._load_gene_selection(gene_list_path)
    config_file, config, config_values, config_snapshot = _run._load_config(config_path)
    normalized_seed = _run._validated_seed(seed)
    normalized_device = _run._validated_device(device)
    (
        manifest_path,
        public_manifest,
        public_files,
        manifest_snapshot,
        public_inventory,
        selected_bytes,
    ) = _run._load_public_manifest(
        public_manifest_path,
        selected_paths=[path for _, path in parsed_contexts],
    )
    matched = []
    for context_id, raw_path in parsed_contexts:
        input_path, relative, snapshot, input_bytes = _run._match_public_input(
            raw_path,
            manifest_path=manifest_path,
            files=public_files,
            inventory=public_inventory,
            selected_bytes=selected_bytes,
        )
        matched.append((context_id, input_path, relative, snapshot, input_bytes))
    condition = _run._condition_record(matched)

    contexts: list[HyperSCACContext] = []
    records: list[Mapping[str, object]] = []
    snapshots: list[tuple[str, _run._FileSnapshot, str]] = []
    raw_genes: tuple[str, ...] | None = None
    selected_genes = tuple(gene_selection["genes"])
    for context_id, input_path, relative, snapshot, input_bytes in matched:
        dataset = load_task_c_dataset_from_verified_bytes(
            input_path,
            context_id=context_id,
            source_bytes=input_bytes,
            source_sha256=snapshot.sha256,
        )
        if dataset.source_sha256 != snapshot.sha256:
            raise HyperSCACError(f"{context_id} context 输入在加载期间发生变化")
        _run._verify_file_snapshot_stat(snapshot, f"{context_id} context 输入文件")
        _run._validate_selected_dataset_semantics(
            dataset,
            relative=relative,
            manifest=public_manifest,
        )
        if raw_genes is None:
            raw_genes = dataset.gene_names
        elif dataset.gene_names != raw_genes:
            raise HyperSCACError("所有 context 原始文件必须使用相同的基因顺序")
        gene_index = {gene: index for index, gene in enumerate(dataset.gene_names)}
        missing = [gene for gene in selected_genes if gene not in gene_index]
        if missing:
            raise HyperSCACError(f"基因清单含有数据中不存在的基因：{missing}")
        columns = np.asarray([gene_index[gene] for gene in selected_genes], dtype=int)
        contexts.append(
            HyperSCACContext(
                context_id=context_id,
                expression=dataset.expression[:, columns],
                interventions=dataset.interventions,
                gene_names=selected_genes,
            )
        )
        records.append(
            MappingProxyType(
                {
                    "context_id": context_id,
                    "input_path": str(input_path),
                    "input_sha256": snapshot.sha256,
                    "content_sha256": dataset.content_sha256,
                    "public_relative_path": relative,
                }
            )
        )
        snapshots.append((context_id, snapshot, dataset.content_sha256))
    selected_bytes.clear()
    return _PreparedBatch(
        contexts=tuple(contexts),
        context_records=tuple(records),
        context_snapshots=tuple(snapshots),
        config=config,
        config_values=MappingProxyType(dict(config_values)),
        config_path=config_file,
        config_snapshot=config_snapshot,
        gene_selection=MappingProxyType(dict(gene_selection)),
        gene_path=gene_path,
        gene_snapshot=gene_snapshot,
        public_manifest_path=manifest_path,
        public_manifest_snapshot=manifest_snapshot,
        public_files=MappingProxyType(dict(public_files)),
        public_inventory=MappingProxyType(dict(public_inventory)),
        condition=MappingProxyType(dict(condition)),
        seed=normalized_seed,
        device=normalized_device,
        code=MappingProxyType(_run._git_state()),
    )


def _batch_identity(
    prepared: _PreparedBatch,
    *,
    registry_path: Path,
    registry_snapshot: _run._FileSnapshot,
    prior_record: Mapping[str, object] | None,
    prior_arguments: Mapping[str, object],
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": "1.0",
        "method_id": "hypersca_c_ablations",
        "contexts": [_plain(record) for record in prepared.context_records],
        "ordered_genes": list(prepared.contexts[0].gene_names),
        "base_config_values": dict(prepared.config_values),
        "config_path": str(prepared.config_path),
        "config_sha256": prepared.config_snapshot.sha256,
        "gene_list_path": str(prepared.gene_path),
        "gene_list_sha256": prepared.gene_snapshot.sha256,
        "public_manifest_path": str(prepared.public_manifest_path),
        "public_manifest_sha256": prepared.public_manifest_snapshot.sha256,
        "ablation_registry_path": str(registry_path),
        "ablation_registry_sha256": registry_snapshot.sha256,
        "seed": prepared.seed,
        "device": prepared.device,
        "condition": prepared.condition["condition"],
        "condition_mode": prepared.condition["mode"],
        "direction": prepared.condition["direction"],
        "stage": prepared.condition["stage"],
        "git_commit": prepared.code["git_commit"],
        "code_dirty": prepared.code["dirty"],
        "code_state_sha256": prepared.code["code_state_sha256"],
        "prior_arguments": _plain(prior_arguments),
        "registered_prior": _plain(prior_record) if prior_record is not None else None,
    }
    return identity


def _capture_optional_prior_argument(
    path: Path | None,
    description: str,
) -> tuple[dict[str, object], _run._FileSnapshot | None]:
    if path is None:
        return {
            "provided": False,
            "path": None,
            "sha256": None,
            "capture_status": "not_provided",
        }, None
    lexical = _run._lexical_absolute(path)
    try:
        snapshot, _ = _run._capture_file_snapshot(
            lexical,
            description,
            collect_bytes=False,
        )
    except HyperSCACError:
        return {
            "provided": True,
            "path": str(lexical),
            "sha256": None,
            "capture_status": "unavailable_or_unsafe",
        }, None
    return {
        "provided": True,
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "capture_status": "captured",
    }, snapshot


def _empty_predictions(
    genes: Sequence[str], context_ids: Sequence[str]
) -> pd.DataFrame:
    del genes
    return pd.DataFrame(
        columns=[
            "source",
            "target",
            "effect",
            "median_effect",
            "direction",
            "selection_frequency",
            "direction_agreement",
            "context_consistency",
            *[f"effect_{context_id}" for context_id in context_ids],
            "score",
            "abstained",
            "abstention_reason",
        ]
    )


def _base_status(
    *,
    ablation_id: str,
    spec: Mapping[str, object],
    prepared: _PreparedBatch,
) -> dict[str, object]:
    interpretation = (
        "单次拟合，不使用跨重复的稳定性权重"
        if ablation_id == "no_stability_weighting"
        else "按预先登记设置运行"
    )
    return {
        "schema_version": "1.0",
        "method_id": "hypersca_c",
        "claim_level": "ablation_raw_inference_only",
        "formal_score_status": "not_sealed",
        "ablation_id": ablation_id,
        "ablation_mode": spec["mode"],
        "configuration_changes": _plain(spec["configuration_changes"]),
        "interpretation": interpretation,
        "seed": prepared.seed,
        "contexts": [context.context_id for context in prepared.contexts],
        "condition": prepared.condition["condition"],
        "condition_mode": prepared.condition["mode"],
        "direction": prepared.condition["direction"],
        "stage": prepared.condition["stage"],
    }


def _failure_payloads(
    *,
    status_name: str,
    reason: str,
    ablation_id: str,
    spec: Mapping[str, object],
    prepared: _PreparedBatch,
    failure: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    failures = [failure] if failure else []
    status = {
        **_base_status(ablation_id=ablation_id, spec=spec, prepared=prepared),
        "status": status_name,
        "reason": reason,
        "requested_bootstraps": (
            asdict(prepared.config) | dict(spec["configuration_changes"])
        )["bootstrap_repeats"],
        "successful_bootstraps": 0,
        "failure_count": len(failures),
        "failures": failures,
        "coverage": 0.0,
        "usable_for_ranking": False,
    }
    summary = {
        "schema_version": "1.0",
        "ablation_id": ablation_id,
        "status": status_name,
        "reason": reason,
        "failures": failures,
    }
    return (
        _empty_predictions(
            prepared.contexts[0].gene_names,
            tuple(context.context_id for context in prepared.contexts),
        ),
        summary,
        status,
    )


def _completed_payloads(
    result: HyperSCAStabilityResult,
    *,
    ablation_id: str,
    spec: Mapping[str, object],
    transformed_config: HyperSCACConfig,
    prepared: _PreparedBatch,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    predictions, summary, primary_status = _run._validate_run_scientific_result(
        predictions=result.predictions,
        summary=result.summary,
        failures=result.failures,
        context_ids=tuple(context.context_id for context in prepared.contexts),
        gene_names=prepared.contexts[0].gene_names,
        requested_repeats=transformed_config.bootstrap_repeats,
        seed=prepared.seed,
        condition=prepared.condition,
    )
    status = {
        **_base_status(ablation_id=ablation_id, spec=spec, prepared=prepared),
        "status": "completed_raw_inference",
        "reason": None,
        "requested_bootstraps": primary_status["requested_bootstraps"],
        "successful_bootstraps": primary_status["successful_bootstraps"],
        "failure_count": primary_status["failure_count"],
        "failures": primary_status["failures"],
        "coverage": primary_status["coverage"],
        "usable_for_ranking": primary_status["usable_for_ranking"],
    }
    return predictions, summary, status


def _write_item(
    directory: Path,
    *,
    predictions: pd.DataFrame,
    summary: Mapping[str, object],
    status: Mapping[str, object],
    batch_identity: Mapping[str, object],
    ablation_id: str,
    spec: Mapping[str, object],
    config: HyperSCACConfig,
    prior_record: Mapping[str, object] | None,
    started_utc: str,
    completed_utc: str,
    duration_seconds: float,
) -> dict[str, object]:
    directory.mkdir()
    raw_path = directory / "raw_predictions.csv"
    predictions.to_csv(raw_path, index=False)
    summary_path = directory / "fit_summary.json"
    status_path = directory / "method_status.json"
    write_json(summary_path, dict(summary))
    write_json(status_path, dict(status))
    item_identity = {
        "batch_run_identity": dict(batch_identity),
        "ablation_id": ablation_id,
        "ablation_mode": spec["mode"],
        "configuration_changes": _plain(spec["configuration_changes"]),
        "effective_config": asdict(config),
        "registered_prior": _plain(prior_record) if prior_record is not None else None,
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "method_id": "hypersca_c",
        "status": status["status"],
        "claim_level": "ablation_raw_inference_only",
        "formal_score_status": "not_sealed",
        "ablation_id": ablation_id,
        "run_identity": item_identity,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "duration_seconds": duration_seconds,
        "artifacts": {
            "raw_predictions.csv": {"sha256": sha256_path(raw_path)},
            "fit_summary.json": {"sha256": sha256_path(summary_path)},
            "method_status.json": {"sha256": sha256_path(status_path)},
            "run_manifest.json": {
                "hash_scope": "canonical_json_without_run_manifest_content_sha256"
            },
        },
    }
    manifest["run_manifest_content_sha256"] = _run._payload_sha256(manifest)
    write_json(directory / "run_manifest.json", manifest)
    return {
        "ablation_id": ablation_id,
        "status": status["status"],
        "reason": status["reason"],
        "usable_for_ranking": status["usable_for_ranking"],
        "output_directory": ablation_id,
        "run_manifest_sha256": sha256_path(directory / "run_manifest.json"),
    }


def _validate_time_record(payload: Mapping[str, object], description: str) -> None:
    started = _run._parse_utc_z(payload.get("started_utc"), "started_utc")
    completed = _run._parse_utc_z(payload.get("completed_utc"), "completed_utc")
    if completed < started:
        raise HyperSCACError(f"{description}完成时间不能早于开始时间")
    duration = payload.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise HyperSCACError(f"{description}时长必须是有限的非负秒数")
    normalized = float(duration)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise HyperSCACError(f"{description}时长必须是有限的非负秒数")
    wall_duration = (completed - started).total_seconds()
    if abs(normalized - wall_duration) > max(1.0, wall_duration * 0.01):
        raise HyperSCACError(f"{description}时长与 UTC 时间不一致")


def _verify_item(
    directory: Path,
    expected_identity: Mapping[str, object],
    expected_ablation_id: str,
) -> str:
    if directory.name != expected_ablation_id:
        raise HyperSCACError("已有单项消融目录与固定登记身份不一致")
    if directory.is_symlink() or not directory.is_dir():
        raise HyperSCACError("已有消融输出必须是普通目录")
    entries = tuple(directory.iterdir())
    if {entry.name for entry in entries} != _ITEM_ARTIFACTS:
        raise HyperSCACError("已有单项消融输出不完整或含额外文件")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise HyperSCACError("已有单项消融输出必须由四个普通文件组成")
    manifest = _run._read_output_json(directory / "run_manifest.json", "消融运行清单")
    if set(manifest) != _ITEM_MANIFEST_FIELDS:
        raise HyperSCACError("已有单项消融运行清单字段不符合固定格式")
    if (
        manifest["schema_version"] != "1.0"
        or manifest["method_id"] != "hypersca_c"
        or manifest["claim_level"] != "ablation_raw_inference_only"
        or manifest["formal_score_status"] != "not_sealed"
    ):
        raise HyperSCACError("已有单项消融运行清单用途说明不符合固定格式")
    _validate_time_record(manifest, "已有单项消融运行清单")
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict) or identity.get("batch_run_identity") != dict(
        expected_identity
    ):
        raise HyperSCACError("已有单项消融输出对应另一组输入或设置")
    ablation_id = manifest.get("ablation_id")
    if (
        ablation_id != expected_ablation_id
        or not isinstance(identity, dict)
        or set(identity) != _ITEM_IDENTITY_FIELDS
    ):
        raise HyperSCACError("已有单项消融清单的登记身份无效")
    spec = _EXPECTED_ABLATIONS[expected_ablation_id]
    expected_config = dict(expected_identity.get("base_config_values", {}))
    expected_config.update(spec["configuration_changes"])  # type: ignore[arg-type]
    if identity.get("ablation_id") != expected_ablation_id:
        raise HyperSCACError("已有单项消融身份与清单不一致")
    if identity.get("ablation_mode") != spec["mode"]:
        raise HyperSCACError("已有单项消融模式与固定登记不一致")
    if identity.get("configuration_changes") != spec["configuration_changes"]:
        raise HyperSCACError("已有单项消融设置变化与固定登记不一致")
    if identity.get("effective_config") != expected_config:
        raise HyperSCACError("已有单项消融 effective config 设置已改变")
    expected_prior = (
        expected_identity.get("registered_prior")
        if expected_ablation_id == "prior_on_secondary"
        else None
    )
    if identity.get("registered_prior") != expected_prior:
        raise HyperSCACError("已有单项消融外部先验身份已改变")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _ITEM_ARTIFACTS:
        raise HyperSCACError("已有单项消融输出缺少校验记录")
    if artifacts.get("run_manifest.json") != {
        "hash_scope": "canonical_json_without_run_manifest_content_sha256"
    }:
        raise HyperSCACError("已有单项消融运行清单自身校验规则无效")
    for name in _ITEM_ARTIFACTS - {"run_manifest.json"}:
        record = artifacts[name]
        if not isinstance(record, dict) or set(record) != {"sha256"}:
            raise HyperSCACError("已有单项消融文件校验记录无效")
        if sha256_path(directory / name) != _run._sha256_text(
            record["sha256"], f"{name} SHA-256"
        ):
            raise HyperSCACError(f"已有消融输出 {name} 已改变")
    without_self = dict(manifest)
    recorded = without_self.pop("run_manifest_content_sha256", None)
    if _run._payload_sha256(without_self) != recorded:
        raise HyperSCACError("已有消融运行清单内容已改变")
    status = _run._read_output_json(directory / "method_status.json", "消融状态")
    if status.get("status") != manifest.get("status"):
        raise HyperSCACError("已有消融状态与运行清单不一致")
    if set(status) != _COMPLETED_STATUS_FIELDS:
        raise HyperSCACError("已有消融状态字段不符合固定格式")
    if (
        status["schema_version"] != "1.0"
        or status["method_id"] != "hypersca_c"
        or status["claim_level"] != "ablation_raw_inference_only"
        or status["formal_score_status"] != "not_sealed"
    ):
        raise HyperSCACError("已有消融状态用途说明不符合固定格式")
    if status.get("ablation_id") != expected_ablation_id:
        raise HyperSCACError("已有消融状态与登记项目不一致")
    if (
        status.get("ablation_mode") != spec["mode"]
        or status.get("configuration_changes") != spec["configuration_changes"]
    ):
        raise HyperSCACError("已有消融状态的模式或设置变化已改变")
    context_records = expected_identity.get("contexts")
    if not isinstance(context_records, list):
        raise HyperSCACError("已有消融批次缺少 context 身份")
    context_ids = [record["context_id"] for record in context_records]
    if status.get("contexts") != context_ids:
        raise HyperSCACError("已有消融状态与固定 context 不一致")
    expected_static_status = {
        "interpretation": (
            "单次拟合，不使用跨重复的稳定性权重"
            if expected_ablation_id == "no_stability_weighting"
            else "按预先登记设置运行"
        ),
        "seed": expected_identity.get("seed"),
        "condition": expected_identity.get("condition"),
        "condition_mode": expected_identity.get("condition_mode"),
        "direction": expected_identity.get("direction"),
        "stage": expected_identity.get("stage"),
    }
    if any(
        status.get(field) != value for field, value in expected_static_status.items()
    ):
        raise HyperSCACError("已有消融状态与批次身份不一致")

    summary = _run._read_output_json(directory / "fit_summary.json", "消融拟合摘要")
    try:
        predictions = pd.read_csv(
            directory / "raw_predictions.csv", keep_default_na=False
        )
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise HyperSCACError("已有消融原始关系表无法重新读取") from exc
    status_name = str(status["status"])
    if status_name == "completed_raw_inference":
        expected_summary_fields = _run._CORE_SUMMARY_FIELDS | {"failures"}
        if set(summary) != expected_summary_fields:
            raise HyperSCACError("已有已完成消融拟合摘要字段不符合固定格式")
        core_summary = {
            key: value for key, value in summary.items() if key != "failures"
        }
        _, validated_summary, primary_status = _run._validate_run_scientific_result(
            predictions=predictions,
            summary=core_summary,
            failures=summary["failures"],
            context_ids=context_ids,
            gene_names=expected_identity["ordered_genes"],  # type: ignore[arg-type]
            requested_repeats=int(expected_config["bootstrap_repeats"]),
            seed=int(expected_identity["seed"]),
            condition={
                "condition": expected_identity["condition"],
                "mode": expected_identity["condition_mode"],
                "direction": expected_identity["direction"],
                "stage": expected_identity["stage"],
            },
        )
        if validated_summary != summary:
            raise HyperSCACError("已有已完成消融拟合摘要与关系表不一致")
        for field in (
            "requested_bootstraps",
            "successful_bootstraps",
            "failure_count",
            "failures",
            "coverage",
            "usable_for_ranking",
        ):
            if status[field] != primary_status[field]:
                raise HyperSCACError("已有已完成消融状态与科学结果不一致")
        if status["reason"] is not None:
            raise HyperSCACError("已有已完成消融不能登记失败原因")
    elif status_name in {"failed_ablation", "official_assets_unavailable"}:
        if set(summary) != _FAILURE_SUMMARY_FIELDS:
            raise HyperSCACError("已有未完成消融摘要字段不符合固定格式")
        expected_columns = _empty_predictions(
            expected_identity["ordered_genes"],  # type: ignore[arg-type]
            context_ids,
        ).columns.tolist()
        if predictions.columns.tolist() != expected_columns or len(predictions) != 0:
            raise HyperSCACError("已有未完成消融必须保留精确空关系表")
        if (
            summary["ablation_id"] != expected_ablation_id
            or summary["status"] != status_name
        ):
            raise HyperSCACError("已有未完成消融摘要与状态不一致")
        if (
            summary["reason"] != status["reason"]
            or summary["failures"] != status["failures"]
        ):
            raise HyperSCACError("已有未完成消融原因或失败记录不一致")
        failures = status["failures"]
        if not isinstance(failures, list) or any(
            not isinstance(failure, str) or not failure for failure in failures
        ):
            raise HyperSCACError("已有未完成消融失败记录格式无效")
        expected_repeats = int(expected_config["bootstrap_repeats"])
        if (
            isinstance(status["requested_bootstraps"], bool)
            or not isinstance(status["requested_bootstraps"], int)
            or status["requested_bootstraps"] != expected_repeats
            or isinstance(status["successful_bootstraps"], bool)
            or not isinstance(status["successful_bootstraps"], int)
            or status["successful_bootstraps"] != 0
            or isinstance(status["failure_count"], bool)
            or not isinstance(status["failure_count"], int)
            or status["failure_count"] != len(failures)
        ):
            raise HyperSCACError("已有未完成消融重复计数与固定设置不一致")
        if (
            isinstance(status["coverage"], bool)
            or not isinstance(status["coverage"], (int, float))
            or float(status["coverage"]) != 0.0
            or status["usable_for_ranking"] is not False
        ):
            raise HyperSCACError("已有未完成消融不能用于排序")
        if status_name == "failed_ablation" and (
            status.get("reason") != "ablation_fit_failed" or len(failures) != 1
        ):
            raise HyperSCACError("已有失败消融的原因或失败计数不符合固定格式")
        if status_name == "official_assets_unavailable" and status.get("reason") != (
            "no_nonoverlapping_preregistered_prior"
        ):
            raise HyperSCACError("已有外部资源不可用原因不符合固定格式")
        if status_name == "official_assets_unavailable" and len(failures) > 1:
            raise HyperSCACError("已有外部资源不可用记录的失败计数无效")
    else:
        raise HyperSCACError("已有消融状态不在固定范围内")
    return status_name


def _reuse_batch(
    output_root: Path,
    identity: Mapping[str, object],
) -> dict[str, str] | None:
    root = _run._lexical_absolute(output_root)
    if root.is_symlink():
        raise HyperSCACError("消融输出根目录不能是符号链接")
    if not root.exists():
        return None
    if not root.is_dir():
        raise HyperSCACError("消融输出位置已存在但不是目录")
    entries = tuple(root.iterdir())
    if not entries:
        return None
    if {entry.name for entry in entries} != set(ABLATION_IDS) | {_BATCH_MANIFEST}:
        raise HyperSCACError("已有消融批次不完整或含额外内容，不能覆盖")
    batch = _run._read_output_json(root / _BATCH_MANIFEST, "消融批次清单")
    if set(batch) != _BATCH_MANIFEST_FIELDS:
        raise HyperSCACError("已有消融批次清单字段不符合固定格式")
    if (
        batch["schema_version"] != "1.0"
        or batch["method_id"] != "hypersca_c_ablations"
        or batch["claim_level"] != "ablation_raw_inference_only"
        or batch["formal_score_status"] != "not_sealed"
        or batch["ablation_order"] != list(ABLATION_IDS)
    ):
        raise HyperSCACError("已有消融批次清单用途或登记顺序不符合固定格式")
    _validate_time_record(batch, "已有消融批次清单")
    if batch.get("run_identity") != dict(identity):
        raise HyperSCACError("已有消融批次对应另一组输入或设置，不能覆盖")
    without_self = dict(batch)
    recorded = without_self.pop("batch_manifest_content_sha256", None)
    if _run._payload_sha256(without_self) != recorded:
        raise HyperSCACError("已有消融批次清单内容已改变")
    records = batch.get("ablations")
    if not isinstance(records, list) or [
        record.get("ablation_id") if isinstance(record, dict) else None
        for record in records
    ] != list(ABLATION_IDS):
        raise HyperSCACError("已有消融批次没有完整保留八项登记顺序")
    statuses: dict[str, str] = {}
    for record, ablation_id in zip(records, ABLATION_IDS, strict=True):
        assert isinstance(record, dict)
        if set(record) != _BATCH_ITEM_FIELDS:
            raise HyperSCACError("已有消融批次项目字段不符合固定格式")
        if record["output_directory"] != ablation_id:
            raise HyperSCACError("已有消融批次项目目录与登记顺序不一致")
        directory = root / ablation_id
        if sha256_path(directory / "run_manifest.json") != record.get(
            "run_manifest_sha256"
        ):
            raise HyperSCACError("已有消融批次清单与单项清单不一致")
        status = _verify_item(directory, identity, ablation_id)
        item_manifest = _run._read_output_json(
            directory / "run_manifest.json", "消融运行清单"
        )
        batch_started = _run._parse_utc_z(batch["started_utc"], "started_utc")
        batch_completed = _run._parse_utc_z(batch["completed_utc"], "completed_utc")
        item_started = _run._parse_utc_z(item_manifest["started_utc"], "started_utc")
        item_completed = _run._parse_utc_z(
            item_manifest["completed_utc"], "completed_utc"
        )
        if item_started < batch_started or item_completed > batch_completed:
            raise HyperSCACError("已有单项消融时间超出批次运行范围")
        if status != record.get("status"):
            raise HyperSCACError("已有消融批次状态与单项状态不一致")
        status_payload = _run._read_output_json(
            directory / "method_status.json", "消融状态"
        )
        if (
            record["reason"] != status_payload["reason"]
            or record["usable_for_ranking"] != status_payload["usable_for_ranking"]
        ):
            raise HyperSCACError("已有消融批次摘要与单项状态不一致")
        statuses[ablation_id] = status
    return statuses


def run_hypersca_c_ablations(
    *,
    context_values: Sequence[str],
    config_path: Path,
    gene_list_path: Path,
    public_manifest_path: Path,
    ablation_registry_path: Path,
    output_root: Path,
    seed: int,
    device: str,
    prior_edges_path: Path | None = None,
    prior_source_manifest_path: Path | None = None,
) -> dict[str, str]:
    """按固定顺序尝试八项消融，并把失败完整保留在原始结果中。"""

    registry, registry_path, registry_snapshot = _load_registry_with_snapshot(
        ablation_registry_path
    )
    prepared = _prepare_batch_inputs(
        context_values=context_values,
        config_path=config_path,
        gene_list_path=gene_list_path,
        public_manifest_path=public_manifest_path,
        seed=seed,
        device=device,
    )
    prior_mask: np.ndarray | None = None
    prior_record: Mapping[str, object] | None = None
    prior_unavailable_detail: str | None = None
    prior_edges_argument, prior_edges_snapshot = _capture_optional_prior_argument(
        prior_edges_path,
        "外部先验关系参数",
    )
    prior_manifest_argument, prior_manifest_snapshot = _capture_optional_prior_argument(
        prior_source_manifest_path,
        "外部先验来源清单参数",
    )
    prior_argument_snapshots = tuple(
        snapshot
        for snapshot in (prior_edges_snapshot, prior_manifest_snapshot)
        if snapshot is not None
    )
    prior_arguments: dict[str, object] = {
        "pair_complete": (
            prior_edges_path is not None and prior_source_manifest_path is not None
        ),
        "edges": prior_edges_argument,
        "source_manifest": prior_manifest_argument,
    }
    if not prior_arguments["pair_complete"] and (
        prior_edges_path is not None or prior_source_manifest_path is not None
    ):
        prior_unavailable_detail = "外部先验关系和来源清单必须同时提供"
    if prior_edges_path is not None and prior_source_manifest_path is not None:
        try:
            prior_mask, prior_record, validated_snapshots = load_registered_prior(
                prior_edges_path,
                prior_source_manifest_path,
                prepared.contexts[0].gene_names,
            )
            if (
                prior_edges_snapshot is None
                or prior_manifest_snapshot is None
                or validated_snapshots
                != (prior_edges_snapshot, prior_manifest_snapshot)
            ):
                raise HyperSCACError("外部先验参数在运行准备期间发生变化")
        except HyperSCACError as exc:
            prior_mask = None
            prior_record = None
            prior_unavailable_detail = " ".join(str(exc).split())[:200]

    identity = _batch_identity(
        prepared,
        registry_path=registry_path,
        registry_snapshot=registry_snapshot,
        prior_record=prior_record,
        prior_arguments=prior_arguments,
    )
    existing = _reuse_batch(output_root, identity)
    if existing is not None:
        return existing

    started_utc = _run._utc_now()
    started_clock = time.monotonic()
    root = _run._lexical_absolute(output_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    records: list[dict[str, object]] = []
    statuses: dict[str, str] = {}
    try:
        for ablation_id in ABLATION_IDS:
            spec = registry[ablation_id]
            item_started_utc = _run._utc_now()
            item_clock = time.monotonic()
            transformed_config = _effective_ablation_config(prepared.config, spec)
            try:
                _, applied_config = apply_hypersca_c_ablation(
                    prepared.contexts,
                    prepared.config,
                    ablation_id,
                    registry=registry,
                )
                if applied_config != transformed_config:
                    raise HyperSCACError("消融变换产生了未登记的设置")
                if ablation_id == "prior_on_secondary" and prior_mask is None:
                    predictions, summary, status = _failure_payloads(
                        status_name="official_assets_unavailable",
                        reason="no_nonoverlapping_preregistered_prior",
                        ablation_id=ablation_id,
                        spec=spec,
                        prepared=prepared,
                        failure=prior_unavailable_detail,
                    )
                else:
                    result = fit_hypersca_c_ablation(
                        prepared.contexts,
                        prepared.config,
                        ablation_id,
                        seed=prepared.seed,
                        device=prepared.device,
                        prior_mask=(
                            prior_mask if ablation_id == "prior_on_secondary" else None
                        ),
                        registry=registry,
                    )
                    predictions, summary, status = _completed_payloads(
                        result,
                        ablation_id=ablation_id,
                        spec=spec,
                        transformed_config=transformed_config,
                        prepared=prepared,
                    )
            except HyperSCACError as exc:
                message = " ".join(str(exc).split()) or exc.__class__.__name__
                predictions, summary, status = _failure_payloads(
                    status_name="failed_ablation",
                    reason="ablation_fit_failed",
                    ablation_id=ablation_id,
                    spec=spec,
                    prepared=prepared,
                    failure=message[:200],
                )
            item_completed_utc = _run._utc_now()
            record = _write_item(
                staging / ablation_id,
                predictions=predictions,
                summary=summary,
                status=status,
                batch_identity=identity,
                ablation_id=ablation_id,
                spec=spec,
                config=transformed_config,
                prior_record=(
                    prior_record if ablation_id == "prior_on_secondary" else None
                ),
                started_utc=item_started_utc,
                completed_utc=item_completed_utc,
                duration_seconds=max(0.0, time.monotonic() - item_clock),
            )
            records.append(record)
            statuses[ablation_id] = str(status["status"])

        _run._verify_run_input_snapshots(
            config_snapshot=prepared.config_snapshot,
            gene_snapshot=prepared.gene_snapshot,
            public_manifest_snapshot=prepared.public_manifest_snapshot,
            public_files=prepared.public_files,
            public_inventory=prepared.public_inventory,
            context_snapshots=prepared.context_snapshots,
            code=prepared.code,
        )
        _run._verify_file_snapshot(registry_snapshot, "HyperSCA-C 消融登记表")
        for snapshot in prior_argument_snapshots:
            _run._verify_file_snapshot(snapshot, "外部先验参数")

        completed_utc = _run._utc_now()
        batch_manifest: dict[str, object] = {
            "schema_version": "1.0",
            "method_id": "hypersca_c_ablations",
            "claim_level": "ablation_raw_inference_only",
            "formal_score_status": "not_sealed",
            "ablation_order": list(ABLATION_IDS),
            "run_identity": identity,
            "ablations": records,
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "duration_seconds": max(0.0, time.monotonic() - started_clock),
        }
        batch_manifest["batch_manifest_content_sha256"] = _run._payload_sha256(
            batch_manifest
        )
        write_json(staging / _BATCH_MANIFEST, batch_manifest)

        if root.exists():
            if root.is_symlink() or not root.is_dir() or next(root.iterdir(), None):
                raise HyperSCACError("消融输出目录在运行期间发生变化，不能覆盖")
            root.rmdir()
        os.replace(staging, root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return statuses
