"""Run HyperSCA-C on registered public Task C training material.

The command-facing helpers in this module keep the scientific fitting code separate
from input permission checks, trace records, and safe output reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.causal.hypersca_c import HyperSCACConfig, HyperSCACContext, HyperSCACError
from src.causal.hypersca_c_stability import (
    HyperSCAStabilityResult,
    fit_stable_hypersca_c,
    thaw_json_record,
)
from src.evaluation.task_c_data import (
    TaskCDataset,
    TaskCDataError,
    load_task_c_dataset,
    sha256_path,
    write_json,
)


MAX_VERIFIED_GENES = 256

_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_CONTEXTS = frozenset({"k562", "rpe1"})
_GENE_LIST_FIELDS = {
    "schema_version",
    "selection_id",
    "selection_basis",
    "genes",
}
_PUBLIC_MANIFEST_FIELDS = {
    "schema_version",
    "split_id",
    "seed",
    "min_cells_per_intervention",
    "train_sources",
    "tune_sources",
    "holdout_source_count",
    "input_sha256",
    "content_sha256",
    "gene_names_sha256",
    "materialization_identity",
    "files",
}
_OUTPUT_NAMES = {
    "raw_predictions.csv",
    "fit_summary.json",
    "method_status.json",
    "run_manifest.json",
}
_PUBLIC_TASK_C_PATHS = frozenset(
    f"within/{context}/{partition}.npz"
    for context in ("k562", "rpe1")
    for partition in ("train", "tune", "refit")
) | frozenset(
    f"cross/{direction}/{partition}.npz"
    for direction in ("k562_to_rpe1", "rpe1_to_k562")
    for partition in (
        "source_train",
        "source_tune",
        "source_refit",
        "target_adapt_train",
        "target_adapt_tune",
        "target_adapt_refit",
    )
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_C_CONTROL_LABEL = "non-targeting"
_FROZEN_BASE_PREDICTION_COLUMNS = frozenset(
    {
        "source",
        "target",
        "effect",
        "median_effect",
        "direction",
        "selection_frequency",
        "direction_agreement",
        "context_consistency",
        "score",
        "abstained",
        "abstention_reason",
    }
)
_CORE_SUMMARY_FIELDS = frozenset(
    {
        "requested_repeats",
        "successful_repeats",
        "repeat_success_fraction",
        "coverage",
        "abstention_rate",
        "score_formula",
    }
)


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    link_count: int


def _required_no_whitespace(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise HyperSCACError(f"{name} 必须是非空且不含空白的文字")
    return value


def _required_plain_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HyperSCACError(f"{name} 必须是无首尾空白的非空通俗文字")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise HyperSCACError(f"{name} 不能含有控制字符")
    return value


def _pairs_to_unique_dict(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HyperSCACError(f"JSON 含有重复字段：{key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise HyperSCACError(f"JSON 不能含有非有限数字：{value}")


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_nlink),
    )


def _capture_file_snapshot(
    path: Path,
    description: str,
    *,
    collect_bytes: bool,
) -> tuple[_FileSnapshot, bytes | None]:
    absolute = _regular_file(path, description, reject_symlink=True)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if collect_bytes else None
    try:
        with absolute.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
        current = absolute.stat()
    except OSError as exc:
        raise HyperSCACError(f"无法固定{description}的输入快照：{path}") from exc
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(current):
        raise HyperSCACError(f"{description}在读取期间发生变化")
    snapshot = _FileSnapshot(
        path=absolute,
        sha256=f"sha256:{digest.hexdigest()}",
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        modified_ns=int(after.st_mtime_ns),
        changed_ns=int(after.st_ctime_ns),
        link_count=int(after.st_nlink),
    )
    return snapshot, b"".join(chunks) if chunks is not None else None


def _verify_file_snapshot(snapshot: _FileSnapshot, description: str) -> None:
    current, _ = _capture_file_snapshot(
        snapshot.path,
        description,
        collect_bytes=False,
    )
    if current != snapshot:
        raise HyperSCACError(f"{description}在拟合期间发生变化")


def _read_strict_json_snapshot(
    path: Path,
    description: str,
) -> tuple[dict[str, object], _FileSnapshot]:
    snapshot, raw_bytes = _capture_file_snapshot(
        path,
        description,
        collect_bytes=True,
    )
    assert raw_bytes is not None
    try:
        raw = raw_bytes.decode("utf-8", errors="strict")
        payload = json.loads(
            raw,
            object_pairs_hook=_pairs_to_unique_dict,
            parse_constant=_reject_json_constant,
        )
    except HyperSCACError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HyperSCACError(f"无法严格读取{description}：{path}") from exc
    if not isinstance(payload, dict):
        raise HyperSCACError(f"{description}顶层必须是 JSON 对象")
    return payload, snapshot


def _read_strict_json(path: Path, description: str) -> dict[str, object]:
    payload, _ = _read_strict_json_snapshot(path, description)
    return payload


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _contains_private_component(path: Path) -> bool:
    return any(part.lower() == "private" for part in path.parts)


def _ensure_no_symlink_component(path: Path, description: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise HyperSCACError(f"{description}不能使用符号链接：{path}")


def _regular_file(path: Path, description: str, *, reject_symlink: bool) -> Path:
    absolute = _lexical_absolute(path)
    if reject_symlink:
        _ensure_no_symlink_component(absolute, description)
    if not absolute.exists() or not absolute.is_file():
        raise HyperSCACError(f"{description}必须是已存在的普通文件：{path}")
    return absolute


def _sha256_text(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HyperSCACError(f"{name} 必须是有效的 SHA-256 记录")
    return value


def _validated_text_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise HyperSCACError(f"{name} 必须是文字列表")
    normalized = [_required_no_whitespace(item, name) for item in value]
    if len(set(normalized)) != len(normalized):
        raise HyperSCACError(f"{name} 不能含有重复值")
    return normalized


def _load_gene_selection(
    path: Path,
) -> tuple[Path, dict[str, object], _FileSnapshot]:
    absolute = _regular_file(path, "基因清单", reject_symlink=True)
    payload, snapshot = _read_strict_json_snapshot(absolute, "基因清单")
    if set(payload) != _GENE_LIST_FIELDS:
        missing = sorted(_GENE_LIST_FIELDS - set(payload))
        extra = sorted(set(payload) - _GENE_LIST_FIELDS)
        raise HyperSCACError(
            f"基因清单字段不符合固定格式；缺少={missing}，额外={extra}"
        )
    if payload["schema_version"] != "1.0":
        raise HyperSCACError("基因清单 schema_version 必须是 1.0")
    selection_id = _required_no_whitespace(payload["selection_id"], "selection_id")
    if "/" in selection_id or "\\" in selection_id:
        raise HyperSCACError("selection_id 不能含有路径字符")
    selection_basis = _required_plain_text(
        payload["selection_basis"], "selection_basis"
    )
    genes = _validated_text_list(payload["genes"], "genes")
    if len(genes) < 2:
        raise HyperSCACError("基因清单至少需要 2 个基因")
    if len(genes) > MAX_VERIFIED_GENES:
        raise HyperSCACError(
            f"基因清单超过当前核验运行上限 {MAX_VERIFIED_GENES}；"
            "该上限用于控制本轮核验，不表示科学上最优的基因数"
        )
    return (
        absolute,
        {
            "schema_version": "1.0",
            "selection_id": selection_id,
            "selection_basis": selection_basis,
            "genes": genes,
        },
        snapshot,
    )


def _parse_context_values(values: Sequence[str]) -> list[tuple[str, Path]]:
    if isinstance(values, (str, bytes)) or not values:
        raise HyperSCACError("至少需要一个 --context name=path")
    parsed: dict[str, Path] = {}
    for raw in values:
        if not isinstance(raw, str) or "=" not in raw:
            raise HyperSCACError("--context 必须使用 name=path")
        name, raw_path = raw.split("=", 1)
        if not name or not raw_path:
            raise HyperSCACError("--context 的 name 和 path 都不能为空")
        if name not in _ALLOWED_CONTEXTS:
            raise HyperSCACError("--context 名称只允许 k562 或 rpe1")
        if name in parsed:
            raise HyperSCACError(f"--context 不能重复提供 {name}")
        parsed[name] = Path(raw_path)
    return [(name, parsed[name]) for name in ("k562", "rpe1") if name in parsed]


def _validate_public_manifest_record(payload: Mapping[str, object]) -> dict[str, str]:
    if set(payload) != _PUBLIC_MANIFEST_FIELDS:
        raise HyperSCACError("公开清单字段不符合 Task C 固定格式")
    if payload["schema_version"] != "1.0":
        raise HyperSCACError("公开清单 schema_version 必须是 1.0")
    _required_no_whitespace(payload["split_id"], "公开清单 split_id")
    for name in ("seed", "min_cells_per_intervention", "holdout_source_count"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HyperSCACError(f"公开清单 {name} 必须是非负整数")
    if int(payload["min_cells_per_intervention"]) < 1:
        raise HyperSCACError("公开清单 min_cells_per_intervention 必须大于 0")
    _validated_text_list(payload["train_sources"], "公开清单 train_sources")
    _validated_text_list(payload["tune_sources"], "公开清单 tune_sources")
    for field in ("input_sha256", "content_sha256"):
        hashes = payload[field]
        if not isinstance(hashes, dict) or set(hashes) != _ALLOWED_CONTEXTS:
            raise HyperSCACError(f"公开清单 {field} 必须恰好包含 k562 和 rpe1")
        for context, value in hashes.items():
            _sha256_text(value, f"公开清单 {field}.{context}")
    _sha256_text(payload["gene_names_sha256"], "公开清单 gene_names_sha256")

    identity = payload["materialization_identity"]
    identity_fields = {
        "schema_version",
        "split_id",
        "seed",
        "min_cells_per_intervention",
        "input_sha256",
        "content_sha256",
        "gene_names_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise HyperSCACError("公开清单 materialization_identity 不符合固定格式")
    for field in identity_fields:
        if identity[field] != payload[field]:
            raise HyperSCACError(
                f"公开清单 materialization_identity.{field} 与清单正文不一致"
            )

    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise HyperSCACError("公开清单 files 必须是非空文件校验表")
    if set(files) != _PUBLIC_TASK_C_PATHS:
        raise HyperSCACError("公开清单必须包含 Task C 的完整公开文件清单")
    normalized: dict[str, str] = {}
    for raw_relative, raw_hash in files.items():
        if not isinstance(raw_relative, str) or not raw_relative:
            raise HyperSCACError("公开清单 files 路径必须是非空文字")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or _contains_private_component(relative)
            or relative.as_posix() != raw_relative
        ):
            raise HyperSCACError("公开清单 files 只能记录安全的公开相对路径")
        normalized[raw_relative] = _sha256_text(
            raw_hash, f"公开清单 files.{raw_relative}"
        )
    return normalized


def _capture_public_inventory(
    manifest_path: Path,
    files: Mapping[str, str],
) -> dict[str, _FileSnapshot]:
    """Verify every registered public file without opening any private path."""

    preflight: dict[str, tuple[Path, os.stat_result]] = {}
    inode_counts: Counter[tuple[int, int]] = Counter()
    for relative in sorted(files):
        candidate = manifest_path.parent / Path(relative)
        absolute = _regular_file(
            candidate,
            f"公开库存文件 {relative}",
            reject_symlink=True,
        )
        try:
            stat = absolute.stat()
        except OSError as exc:
            raise HyperSCACError(f"无法检查公开库存 inode：{relative}") from exc
        preflight[relative] = (absolute, stat)
        inode_counts[(int(stat.st_dev), int(stat.st_ino))] += 1
    for relative, (_, stat) in preflight.items():
        registered_links = inode_counts[(int(stat.st_dev), int(stat.st_ino))]
        if int(stat.st_nlink) != registered_links:
            raise HyperSCACError(
                f"公开库存文件存在包外硬链接或未登记 inode 别名：{relative}"
            )

    # Only after the inode inventory is closed do we open public paths for hashes.
    inventory: dict[str, _FileSnapshot] = {}
    for relative, (absolute, preflight_stat) in preflight.items():
        snapshot, _ = _capture_file_snapshot(
            absolute,
            f"公开库存文件 {relative}",
            collect_bytes=False,
        )
        if _stat_identity(preflight_stat) != (
            snapshot.device,
            snapshot.inode,
            snapshot.size,
            snapshot.modified_ns,
            snapshot.changed_ns,
            snapshot.link_count,
        ):
            raise HyperSCACError(f"公开库存 inode 在核验期间发生变化：{relative}")
        if snapshot.sha256 != files[relative]:
            raise HyperSCACError(f"公开库存文件 SHA-256 不一致：{relative}")
        inventory[relative] = snapshot
    return inventory


def _verify_public_inventory(
    inventory: Mapping[str, _FileSnapshot],
    files: Mapping[str, str],
) -> None:
    if set(inventory) != set(files):
        raise HyperSCACError("公开库存快照不完整")
    inode_counts: Counter[tuple[int, int]] = Counter()
    for relative, snapshot in inventory.items():
        _verify_file_snapshot(snapshot, f"公开库存文件 {relative}")
        if snapshot.sha256 != files[relative]:
            raise HyperSCACError(f"公开库存记录在拟合期间发生变化：{relative}")
        inode_counts[(snapshot.device, snapshot.inode)] += 1
    for relative, snapshot in inventory.items():
        if snapshot.link_count != inode_counts[(snapshot.device, snapshot.inode)]:
            raise HyperSCACError(f"公开库存硬链接语义在拟合期间发生变化：{relative}")


def _load_public_manifest(
    path: Path,
) -> tuple[
    Path,
    dict[str, object],
    dict[str, str],
    _FileSnapshot,
    dict[str, _FileSnapshot],
]:
    lexical = _lexical_absolute(path)
    if _contains_private_component(lexical):
        raise HyperSCACError("公开清单路径不能包含 private 目录")
    absolute = _regular_file(lexical, "公开清单", reject_symlink=True)
    payload, snapshot = _read_strict_json_snapshot(absolute, "公开清单")
    files = _validate_public_manifest_record(payload)
    inventory = _capture_public_inventory(absolute, files)
    return absolute, payload, files, snapshot, inventory


def _match_public_input(
    path: Path,
    *,
    manifest_path: Path,
    files: Mapping[str, str],
    inventory: Mapping[str, _FileSnapshot],
) -> tuple[Path, str, _FileSnapshot]:
    absolute = _regular_file(path, "context 输入文件", reject_symlink=True)
    root = manifest_path.parent.resolve(strict=True)
    resolved = absolute.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise HyperSCACError("context 输入文件不在公开清单目录内") from exc
    if _contains_private_component(Path(relative)):
        raise HyperSCACError("context 输入文件不能来自 private 目录")
    if relative not in files:
        raise HyperSCACError("context 输入文件未登记在公开清单 files 中")
    snapshot = inventory[relative]
    if snapshot.path != resolved:
        raise HyperSCACError("context 输入文件路径与公开库存快照不一致")
    _verify_file_snapshot(snapshot, "context 输入文件")
    if snapshot.sha256 != files[relative]:
        raise HyperSCACError("context 输入文件与公开清单 SHA-256 不一致")
    return resolved, relative, snapshot


def _condition_record(
    matched_inputs: Sequence[tuple[str, Path, str, _FileSnapshot]],
) -> dict[str, object]:
    """Bind context labels to one complete public Task C training condition."""

    kinds = {Path(relative).parts[0] for _, _, relative, _ in matched_inputs}
    if kinds == {"within"}:
        stages: set[str] = set()
        context_names: list[str] = []
        for context_id, _, relative, _ in matched_inputs:
            parts = Path(relative).parts
            if len(parts) != 3 or parts[0] != "within":
                raise HyperSCACError("within condition 文件路径不符合固定格式")
            registered_context = parts[1]
            filename = parts[2]
            if registered_context != context_id:
                raise HyperSCACError("within condition 文件与 context 名称没有正确绑定")
            if not filename.endswith(".npz"):
                raise HyperSCACError("within condition 文件名不符合固定格式")
            stage = filename[: -len(".npz")]
            if stage not in {"train", "tune", "refit"}:
                raise HyperSCACError("within condition stage 不在固定范围内")
            stages.add(stage)
            context_names.append(context_id)
        if len(stages) != 1:
            raise HyperSCACError("联合 within condition 必须使用相同 stage")
        stage = next(iter(stages))
        return {
            "condition": f"within_{stage}_{'_'.join(context_names)}",
            "mode": "within",
            "direction": None,
            "stage": stage,
        }

    if kinds != {"cross"}:
        raise HyperSCACError("context 文件必须组成单一 within 或 cross condition")
    if len(matched_inputs) != 2:
        raise HyperSCACError("cross condition 必须包含完整的来源和目标适配文件")

    parsed: list[tuple[str, str, str, str]] = []
    for context_id, _, relative, _ in matched_inputs:
        parts = Path(relative).parts
        if len(parts) != 3 or parts[0] != "cross":
            raise HyperSCACError("cross condition 文件路径不符合固定格式")
        direction = parts[1]
        filename = parts[2]
        match = re.fullmatch(
            r"(source|target_adapt)_(train|tune|refit)\.npz",
            filename,
        )
        if direction not in {"k562_to_rpe1", "rpe1_to_k562"} or match is None:
            raise HyperSCACError("cross condition 文件名不符合固定格式")
        role, stage = match.groups()
        source_context, target_context = direction.split("_to_", 1)
        expected_context = source_context if role == "source" else target_context
        if context_id != expected_context:
            raise HyperSCACError("cross condition 文件与 context 名称没有正确绑定")
        parsed.append((direction, role, stage, context_id))

    directions = {item[0] for item in parsed}
    roles = {item[1] for item in parsed}
    stages = {item[2] for item in parsed}
    if len(directions) != 1 or roles != {"source", "target_adapt"} or len(stages) != 1:
        raise HyperSCACError("cross condition 必须是同一方向、同一 stage 的完整组合")
    direction = next(iter(directions))
    stage = next(iter(stages))
    return {
        "condition": f"cross_{stage}_{direction}",
        "mode": "cross",
        "direction": direction,
        "stage": stage,
    }


def _gene_names_sha256(gene_names: Sequence[str]) -> str:
    encoded = json.dumps(
        tuple(gene_names),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _expected_interventions(
    relative: str,
    manifest: Mapping[str, object],
) -> set[str]:
    train_sources = set(manifest["train_sources"])
    tune_sources = set(manifest["tune_sources"])
    parts = Path(relative).parts
    filename = parts[2]
    if parts[0] == "within":
        stage = filename.removesuffix(".npz")
        role = "within"
    else:
        match = re.fullmatch(
            r"(source|target_adapt)_(train|tune|refit)\.npz",
            filename,
        )
        if match is None:  # pragma: no cover - condition validation already checked
            raise HyperSCACError("公开数据文件名无法解释干预语义")
        role, stage = match.groups()
    if role == "target_adapt":
        return {_TASK_C_CONTROL_LABEL}
    stage_sources = {
        "train": train_sources,
        "tune": tune_sources,
        "refit": train_sources | tune_sources,
    }[stage]
    return stage_sources | {_TASK_C_CONTROL_LABEL}


def _validate_selected_dataset_semantics(
    dataset: TaskCDataset,
    *,
    relative: str,
    manifest: Mapping[str, object],
) -> None:
    gene_names = tuple(dataset.gene_names)
    if _gene_names_sha256(gene_names) != manifest["gene_names_sha256"]:
        raise HyperSCACError("公开数据基因顺序与 manifest gene_names_sha256 不一致")
    observed = set(dataset.interventions.tolist())
    expected = _expected_interventions(relative, manifest)
    if observed != expected:
        raise HyperSCACError(
            f"公开数据干预标签语义不匹配：{relative}；"
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )


def _load_config(
    path: Path,
) -> tuple[Path, HyperSCACConfig, dict[str, object], _FileSnapshot]:
    absolute = _regular_file(path, "HyperSCA-C 设置文件", reject_symlink=True)
    payload, snapshot = _read_strict_json_snapshot(
        absolute,
        "HyperSCA-C 设置文件",
    )
    config = HyperSCACConfig.from_mapping(payload)
    if config.prior_discount != 0.0:
        raise HyperSCACError("本轮主分析不开放先验折扣，prior_discount 必须是 0")
    return absolute, config, asdict(config), snapshot


def _validated_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**64 - 1:
        raise HyperSCACError("seed 必须是 torch 支持的非负整数")
    return int(seed)


def _validated_device(device: object) -> str:
    if device not in {"cpu", "cuda"}:
        raise HyperSCACError("device 只允许 cpu 或 cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise HyperSCACError("已要求 CUDA，但当前环境不可用")
    return str(device)


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked_diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
            ],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HyperSCACError("无法记录当前代码版本") from exc
    if not commit:
        raise HyperSCACError("当前代码版本记录为空")
    digest = hashlib.sha256()
    digest.update(b"HyperSCA-code-state-v1\0")
    digest.update(len(tracked_diff).to_bytes(8, "big"))
    digest.update(tracked_diff)
    untracked_paths = sorted(path for path in untracked_output.split(b"\0") if path)
    for raw_relative in untracked_paths:
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        candidate = _ROOT / relative
        digest.update(len(raw_relative).to_bytes(8, "big"))
        digest.update(raw_relative)
        try:
            if candidate.is_symlink():
                link_target = os.readlink(candidate).encode(
                    "utf-8", errors="surrogateescape"
                )
                digest.update(b"symlink\0")
                digest.update(len(link_target).to_bytes(8, "big"))
                digest.update(link_target)
            elif candidate.is_file():
                digest.update(b"file\0")
                with candidate.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
            else:
                raise HyperSCACError(f"无法核验未登记代码文件：{relative}")
        except OSError as exc:
            raise HyperSCACError(f"无法核验未登记代码文件：{relative}") from exc
    return {
        "git_commit": commit,
        "dirty": bool(dirty_output.strip()),
        "runtime_dirty": bool(tracked_diff or untracked_paths),
        "code_state_sha256": f"sha256:{digest.hexdigest()}",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HyperSCACError("运行记录含有不能写入 JSON 的内容") from exc
    return (text + "\n").encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()}"


def _read_output_json(path: Path, description: str) -> dict[str, object]:
    payload = _read_strict_json(path, description)
    if path.read_bytes() != _canonical_json_bytes(payload):
        raise HyperSCACError(f"已有{description}的字节格式已改变，不能复用")
    return payload


def _build_identity(
    *,
    context_records: Sequence[Mapping[str, object]],
    config_path: Path,
    gene_path: Path,
    public_manifest_path: Path,
    seed: int,
    device: str,
    code: Mapping[str, object],
    condition: Mapping[str, object],
    config_sha256: str,
    gene_sha256: str,
    public_manifest_sha256: str,
) -> dict[str, object]:
    identity = {
        "schema_version": "1.0",
        "method_id": "hypersca_c",
        "contexts": [
            {
                "context_id": record["context_id"],
                "input_path": record["input_path"],
                "input_sha256": record["input_sha256"],
                "content_sha256": record["content_sha256"],
                "public_relative_path": record["public_relative_path"],
            }
            for record in context_records
        ],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "gene_list_path": str(gene_path),
        "gene_list_sha256": gene_sha256,
        "public_manifest_path": str(public_manifest_path),
        "public_manifest_sha256": public_manifest_sha256,
        "seed": seed,
        "device": device,
        "git_commit": code["git_commit"],
        "code_dirty": code["dirty"],
        "code_state_sha256": code["code_state_sha256"],
    }
    identity.update(condition)
    return identity


def _verify_run_input_snapshots(
    *,
    config_snapshot: _FileSnapshot,
    gene_snapshot: _FileSnapshot,
    public_manifest_snapshot: _FileSnapshot,
    public_files: Mapping[str, str],
    public_inventory: Mapping[str, _FileSnapshot],
    context_snapshots: Sequence[tuple[str, _FileSnapshot, str]],
    code: Mapping[str, object],
) -> None:
    for snapshot, description in (
        (config_snapshot, "HyperSCA-C 设置文件"),
        (gene_snapshot, "基因清单"),
        (public_manifest_snapshot, "公开清单"),
    ):
        _verify_file_snapshot(snapshot, description)
    _verify_public_inventory(public_inventory, public_files)
    for context_id, snapshot, content_sha256 in context_snapshots:
        _verify_file_snapshot(snapshot, f"{context_id} context 输入文件")
        dataset = load_task_c_dataset(snapshot.path, context_id=context_id)
        if (
            dataset.source_sha256 != snapshot.sha256
            or dataset.content_sha256 != content_sha256
        ):
            raise HyperSCACError(f"{context_id} context 输入内容在拟合期间发生变化")
        _verify_file_snapshot(snapshot, f"{context_id} context 输入文件")
    if _git_state() != dict(code):
        raise HyperSCACError("运行代码在拟合期间发生变化")


def _validate_run_scientific_result(
    *,
    predictions: object,
    summary: object,
    failures: object,
    context_ids: Sequence[str],
    gene_names: Sequence[str],
    requested_repeats: int,
    seed: int,
    condition: Mapping[str, object],
    method_status: object | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Validate scientific output independently at both write and reuse boundaries."""

    try:
        validated = HyperSCAStabilityResult(
            predictions=predictions,  # type: ignore[arg-type]
            summary=summary,  # type: ignore[arg-type]
            failures=failures,  # type: ignore[arg-type]
        )
    except HyperSCACError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise HyperSCACError("HyperSCA-C 科学结果无法完成独立深度核验") from exc

    contexts = tuple(context_ids)
    genes = tuple(gene_names)
    expected_columns = _FROZEN_BASE_PREDICTION_COLUMNS | {
        f"effect_{context_id}" for context_id in contexts
    }
    if set(validated.predictions.columns) != expected_columns:
        raise HyperSCACError("原始关系表列集合与当前实际 context 不精确匹配")
    if set(validated.predictions["source"]) != set(genes) or set(
        validated.predictions["target"]
    ) != set(genes):
        raise HyperSCACError("原始关系表基因集合必须恰好等于固定 gene-list")
    expected_rows = len(genes) * (len(genes) - 1)
    if len(validated.predictions) != expected_rows:
        raise HyperSCACError("原始关系表必须包含 gene-list 的全部非自身有向关系")

    core_summary = thaw_json_record(validated.summary)
    if set(core_summary) != _CORE_SUMMARY_FIELDS:
        raise HyperSCACError("拟合摘要字段集合与冻结格式不一致")
    if int(core_summary["requested_repeats"]) != requested_repeats:
        raise HyperSCACError("拟合摘要 requested_repeats 与固定设置不一致")
    disk_summary = {**core_summary, "failures": list(validated.failures)}
    successful = int(core_summary["successful_repeats"])
    coverage = float(core_summary["coverage"])
    expected_status: dict[str, object] = {
        "schema_version": "1.0",
        "method_id": "hypersca_c",
        "status": "completed_raw_inference",
        "claim_level": "raw_inference_only",
        "seed": seed,
        "contexts": list(contexts),
        **condition,
        "requested_bootstraps": requested_repeats,
        "successful_bootstraps": successful,
        "failure_count": len(validated.failures),
        "failures": list(validated.failures),
        "coverage": coverage,
        "usable_for_ranking": successful > 0 and coverage > 0.0,
    }
    if method_status is not None:
        if not isinstance(method_status, Mapping) or dict(method_status) != expected_status:
            raise HyperSCACError("方法状态与关系表、拟合摘要或失败记录不一致")
    return validated.predictions, disk_summary, expected_status


def _reuse_existing_output(
    output_dir: Path,
    identity: Mapping[str, object],
    *,
    context_ids: Sequence[str],
    gene_names: Sequence[str],
    requested_repeats: int,
    seed: int,
    condition: Mapping[str, object],
) -> dict[str, object] | None:
    output = _lexical_absolute(output_dir)
    if output.is_symlink():
        raise HyperSCACError("输出目录不能是符号链接")
    if not output.exists():
        return None
    if not output.is_dir():
        raise HyperSCACError("输出位置已存在但不是目录")
    entries = tuple(output.iterdir())
    if not entries:
        return None
    if {entry.name for entry in entries} != _OUTPUT_NAMES:
        raise HyperSCACError("已有输出不完整或含额外文件，不能覆盖")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise HyperSCACError("已有输出必须由四个普通文件组成")

    manifest_path = output / "run_manifest.json"
    manifest = _read_output_json(manifest_path, "运行清单")
    if manifest.get("run_identity") != identity:
        raise HyperSCACError("已有输出对应另一组输入或设置，不能覆盖")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _OUTPUT_NAMES:
        raise HyperSCACError("已有运行清单缺少完整的四文件校验记录")
    for name in _OUTPUT_NAMES - {"run_manifest.json"}:
        record = artifacts.get(name)
        if not isinstance(record, dict) or set(record) != {"sha256"}:
            raise HyperSCACError(f"已有运行清单的 {name} 校验记录无效")
        expected = _sha256_text(record["sha256"], f"已有 {name} SHA-256")
        if sha256_path(output / name) != expected:
            raise HyperSCACError(f"已有输出 {name} 已改变，不能复用")
    self_record = artifacts.get("run_manifest.json")
    if not isinstance(self_record, dict) or self_record != {
        "hash_scope": "canonical_json_without_run_manifest_content_sha256"
    }:
        raise HyperSCACError("已有运行清单自身校验规则无效")
    recorded_self_hash = manifest.get("run_manifest_content_sha256")
    _sha256_text(recorded_self_hash, "运行清单自身 SHA-256")
    without_self = dict(manifest)
    without_self.pop("run_manifest_content_sha256", None)
    if _payload_sha256(without_self) != recorded_self_hash:
        raise HyperSCACError("已有运行清单内容已改变，不能复用")

    try:
        predictions = pd.read_csv(
            output / "raw_predictions.csv",
            keep_default_na=False,
        )
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise HyperSCACError("已有原始关系表无法重新读取") from exc
    summary = _read_output_json(output / "fit_summary.json", "拟合摘要")
    status = _read_output_json(output / "method_status.json", "方法状态")
    if set(summary) != _CORE_SUMMARY_FIELDS | {"failures"}:
        raise HyperSCACError("已有拟合摘要字段集合与冻结格式不一致")
    failures = summary["failures"]
    core_summary = {
        key: value for key, value in summary.items() if key != "failures"
    }
    _, validated_summary, _ = _validate_run_scientific_result(
        predictions=predictions,
        summary=core_summary,
        failures=failures,
        context_ids=context_ids,
        gene_names=gene_names,
        requested_repeats=requested_repeats,
        seed=seed,
        condition=condition,
        method_status=status,
    )
    return validated_summary


def _write_csv_atomic(path: Path, predictions: pd.DataFrame) -> None:
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        predictions.to_csv(temporary, index=False)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise HyperSCACError("无法原子写入原始关系表") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _write_new_output(
    output_dir: Path,
    *,
    predictions: pd.DataFrame,
    summary: Mapping[str, object],
    method_status: Mapping[str, object],
    run_manifest: dict[str, object],
) -> None:
    output = _lexical_absolute(output_dir)
    parent = output.parent
    staging: Path | None = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
        raw_path = staging / "raw_predictions.csv"
        summary_path = staging / "fit_summary.json"
        status_path = staging / "method_status.json"
        _write_csv_atomic(raw_path, predictions)
        write_json(summary_path, dict(summary))
        write_json(status_path, dict(method_status))

        run_manifest["artifacts"] = {
            "raw_predictions.csv": {"sha256": sha256_path(raw_path)},
            "fit_summary.json": {"sha256": sha256_path(summary_path)},
            "method_status.json": {"sha256": sha256_path(status_path)},
            "run_manifest.json": {
                "hash_scope": "canonical_json_without_run_manifest_content_sha256"
            },
        }
        run_manifest["run_manifest_content_sha256"] = _payload_sha256(run_manifest)
        write_json(staging / "run_manifest.json", run_manifest)

        if output.exists():
            if output.is_symlink() or not output.is_dir() or next(output.iterdir(), None):
                raise HyperSCACError("输出目录在运行期间发生变化，不能写入")
            output.rmdir()
        os.replace(staging, output)
        staging = None
    except (TaskCDataError, OSError) as exc:
        raise HyperSCACError(f"无法安全写入运行结果：{exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def run_hypersca_c(
    *,
    context_values: Sequence[str],
    config_path: Path,
    gene_list_path: Path,
    public_manifest_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Validate one registered run, fit it, and safely materialize four artifacts."""

    parsed_contexts = _parse_context_values(context_values)
    gene_path, gene_selection, gene_snapshot = _load_gene_selection(gene_list_path)
    config_file, config, config_values, config_snapshot = _load_config(config_path)
    normalized_seed = _validated_seed(seed)
    normalized_device = _validated_device(device)
    (
        manifest_path,
        public_manifest_payload,
        public_files,
        public_manifest_snapshot,
        public_inventory,
    ) = _load_public_manifest(public_manifest_path)

    matched_inputs: list[tuple[str, Path, str, _FileSnapshot]] = []
    for context_id, raw_path in parsed_contexts:
        input_path, relative, input_snapshot = _match_public_input(
            raw_path,
            manifest_path=manifest_path,
            files=public_files,
            inventory=public_inventory,
        )
        matched_inputs.append((context_id, input_path, relative, input_snapshot))
    condition = _condition_record(matched_inputs)

    datasets = []
    context_records: list[dict[str, object]] = []
    context_snapshots: list[tuple[str, _FileSnapshot, str]] = []
    expected_raw_genes: tuple[str, ...] | None = None
    selected_genes = tuple(gene_selection["genes"])
    for context_id, input_path, relative, input_snapshot in matched_inputs:
        dataset = load_task_c_dataset(input_path, context_id=context_id)
        if dataset.source_sha256 != input_snapshot.sha256:
            raise HyperSCACError(f"{context_id} context 输入在加载期间发生变化")
        _verify_file_snapshot(input_snapshot, f"{context_id} context 输入文件")
        _validate_selected_dataset_semantics(
            dataset,
            relative=relative,
            manifest=public_manifest_payload,
        )
        if expected_raw_genes is None:
            expected_raw_genes = dataset.gene_names
        elif dataset.gene_names != expected_raw_genes:
            raise HyperSCACError("所有 context 原始文件必须使用相同的基因顺序")
        gene_index = {gene: index for index, gene in enumerate(dataset.gene_names)}
        missing = [gene for gene in selected_genes if gene not in gene_index]
        if missing:
            raise HyperSCACError(f"基因清单含有数据中不存在的基因：{missing}")
        columns = np.asarray([gene_index[gene] for gene in selected_genes], dtype=int)
        datasets.append(
            HyperSCACContext(
                context_id=context_id,
                expression=dataset.expression[:, columns],
                interventions=dataset.interventions,
                gene_names=selected_genes,
            )
        )
        context_records.append(
            {
                "context_id": context_id,
                "input_path": str(input_path),
                "input_sha256": input_snapshot.sha256,
                "content_sha256": dataset.content_sha256,
                "public_relative_path": relative,
            }
        )
        context_snapshots.append(
            (context_id, input_snapshot, dataset.content_sha256)
        )

    code = _git_state()
    identity = _build_identity(
        context_records=context_records,
        config_path=config_file,
        gene_path=gene_path,
        public_manifest_path=manifest_path,
        seed=normalized_seed,
        device=normalized_device,
        code=code,
        condition=condition,
        config_sha256=config_snapshot.sha256,
        gene_sha256=gene_snapshot.sha256,
        public_manifest_sha256=public_manifest_snapshot.sha256,
    )
    context_ids = tuple(record["context_id"] for record in context_records)
    existing = _reuse_existing_output(
        output_dir,
        identity,
        context_ids=context_ids,
        gene_names=selected_genes,
        requested_repeats=config.bootstrap_repeats,
        seed=normalized_seed,
        condition=condition,
    )
    if existing is not None:
        return existing

    started_utc = _utc_now()
    started_clock = time.monotonic()
    result = fit_stable_hypersca_c(
        datasets,
        config,
        seed=normalized_seed,
        device=normalized_device,
    )
    _verify_run_input_snapshots(
        config_snapshot=config_snapshot,
        gene_snapshot=gene_snapshot,
        public_manifest_snapshot=public_manifest_snapshot,
        public_files=public_files,
        public_inventory=public_inventory,
        context_snapshots=context_snapshots,
        code=code,
    )
    predictions, summary, method_status = _validate_run_scientific_result(
        predictions=result.predictions,
        summary=result.summary,
        failures=result.failures,
        context_ids=context_ids,
        gene_names=selected_genes,
        requested_repeats=config.bootstrap_repeats,
        seed=normalized_seed,
        condition=condition,
    )
    completed_utc = _utc_now()
    duration = max(0.0, time.monotonic() - started_clock)

    gene_record: dict[str, object] = {
        "path": str(gene_path),
        "sha256": gene_snapshot.sha256,
        "selection_id": gene_selection["selection_id"],
        "selection_basis": gene_selection["selection_basis"],
        "gene_count": len(selected_genes),
        "ordered_genes": list(selected_genes),
    }
    run_manifest: dict[str, object] = {
        "schema_version": "1.0",
        "method_id": "hypersca_c",
        "status": "completed_raw_inference",
        "seed": normalized_seed,
        "device": normalized_device,
        "contexts": context_records,
        **condition,
        "config": {
            "path": str(config_file),
            "sha256": config_snapshot.sha256,
            "values": config_values,
        },
        "gene_selection": gene_record,
        "public_manifest": {
            "path": str(manifest_path),
            "sha256": public_manifest_snapshot.sha256,
        },
        "code": dict(code),
        "run_identity": identity,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "duration_seconds": duration,
    }
    _write_new_output(
        output_dir,
        predictions=predictions,
        summary=summary,
        method_status=method_status,
        run_manifest=run_manifest,
    )
    return summary
