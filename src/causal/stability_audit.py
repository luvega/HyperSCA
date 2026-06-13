"""Sidecar audit utilities for causal graph stability and negative controls."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _as_square_matrix(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return arr


def _edge_key(labels: Sequence[str], i: int, j: int) -> str:
    return f"{labels[i]}->{labels[j]}"


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_mask = np.asarray(a) > 0
    b_mask = np.asarray(b) > 0
    np.fill_diagonal(a_mask, False)
    np.fill_diagonal(b_mask, False)
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_mask, b_mask).sum() / union)


def _bh_qvalues(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / float(idx + 1))
        q_ranked[idx] = running
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q.tolist()


def summarize_group_consistency(
    group_freqs: Mapping[str, np.ndarray] | None,
    node_labels: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Summarize consensus and group-specific directed edges."""
    if not group_freqs:
        return {
            "groups": [],
            "n_groups": 0,
            "per_group_metrics": {},
            "edge_jaccard_matrix": [],
            "consensus_edges": [],
            "group_specific_edges": [],
            "missing_group_warning": "No group frequency matrices were supplied.",
        }

    labels = list(node_labels)
    group_items = [(str(name), _as_square_matrix(freq, f"group_freqs[{name}]")) for name, freq in group_freqs.items()]
    groups = [name for name, _ in group_items]
    masks = {name: (freq >= threshold).astype(float) for name, freq in group_items}
    for mask in masks.values():
        np.fill_diagonal(mask, 0.0)

    per_group = {
        name: {"n_edges": int(mask.sum()), "edge_density": float(mask.sum() / max(mask.size - len(labels), 1))}
        for name, mask in masks.items()
    }
    jaccard_matrix = [
        [_jaccard(masks[left], masks[right]) for right in groups]
        for left in groups
    ]

    consensus_edges: list[dict[str, Any]] = []
    group_specific_edges: list[dict[str, Any]] = []
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            if i == j:
                continue
            present = [name for name in groups if masks[name][i, j] > 0]
            if len(present) == len(groups):
                consensus_edges.append({"source": source, "target": target, "groups": present})
            elif present:
                group_specific_edges.append({"source": source, "target": target, "groups": present})

    return {
        "groups": groups,
        "n_groups": len(groups),
        "per_group_metrics": per_group,
        "edge_jaccard_matrix": jaccard_matrix,
        "consensus_edges": consensus_edges,
        "group_specific_edges": group_specific_edges,
        "missing_group_warning": None,
    }


def build_edge_stability_table(
    adjacency: np.ndarray,
    bootstrap_freq: np.ndarray,
    node_labels: Sequence[str],
    type_mapping: Mapping[str, str] | None = None,
    stability_freqs: Sequence[np.ndarray] | None = None,
    null_freqs: Sequence[np.ndarray] | None = None,
    group_freqs: Mapping[str, np.ndarray] | None = None,
    causal_input_metadata: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
    fdr_alpha: float = 0.10,
) -> pd.DataFrame:
    """Build an all-directed-edge stability table for Step2 causal outputs."""
    adj = _as_square_matrix(adjacency, "adjacency")
    freq = _as_square_matrix(bootstrap_freq, "bootstrap_freq")
    if adj.shape != freq.shape:
        raise ValueError("adjacency and bootstrap_freq must have the same shape")
    labels = list(node_labels)
    if len(labels) != adj.shape[0]:
        raise ValueError("node_labels length must match adjacency shape")

    type_mapping = dict(type_mapping or {})
    stability = [_as_square_matrix(m, "stability_freq") for m in (stability_freqs or [freq])]
    nulls = [_as_square_matrix(m, "null_freq") for m in (null_freqs or [])]
    groups = {str(name): _as_square_matrix(m, f"group_freqs[{name}]") for name, m in (group_freqs or {}).items()}
    base_mask = (adj > 0).astype(float)
    np.fill_diagonal(base_mask, 0.0)

    rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            if i == j:
                continue
            edge_values = np.array([m[i, j] for m in stability], dtype=float)
            null_values = np.array([m[i, j] for m in nulls], dtype=float)
            mean_freq = float(edge_values.mean()) if len(edge_values) else float(freq[i, j])
            null_mean = float(null_values.mean()) if len(null_values) else 0.0
            null_p95 = float(np.quantile(null_values, 0.95)) if len(null_values) else 0.0
            empirical_p = float((np.sum(null_values >= mean_freq) + 1) / (len(null_values) + 1)) if len(null_values) else 1.0
            pvalues.append(empirical_p)
            group_presence = [name for name, matrix in groups.items() if matrix[i, j] >= threshold]
            group_support_rate = len(group_presence) / max(len(groups), 1) if groups else 0.0
            edge_mask = np.zeros_like(base_mask)
            edge_mask[i, j] = 1.0 if mean_freq >= threshold else 0.0
            rows.append(
                {
                    "source_node": source,
                    "target_node": target,
                    "source_type": type_mapping.get(source, source),
                    "target_type": type_mapping.get(target, target),
                    "base_edge": bool(adj[i, j] > 0),
                    "base_freq": float(freq[i, j]),
                    "mean_freq": mean_freq,
                    "sd_freq": float(edge_values.std(ddof=0)) if len(edge_values) else 0.0,
                    "seed_support": float(np.mean(edge_values >= threshold)) if len(edge_values) else 0.0,
                    "threshold_support": float(np.mean([mean_freq >= t for t in (0.3, 0.5, 0.7)])),
                    "group_support_rate": float(group_support_rate),
                    "group_presence": ";".join(group_presence),
                    "edge_jaccard_vs_base": _jaccard(base_mask, edge_mask),
                    "null_mean_freq": null_mean,
                    "null_p95_freq": null_p95,
                    "empirical_pvalue": empirical_p,
                    "fdr_qvalue": 1.0,
                    "negative_control_pass": False,
                    "stability_class": "unstable_candidate",
                    "evidence_level": (
                        "exploratory_cluster_graph"
                        if (causal_input_metadata or {}).get("observation_unit") == "gene_proxy"
                        else "causal_candidate"
                    ),
                }
            )

    qvalues = _bh_qvalues(pvalues)
    for row, qvalue in zip(rows, qvalues):
        row["fdr_qvalue"] = float(qvalue)
        enough_nulls_for_fdr = len(nulls) >= 10
        row["negative_control_pass"] = bool(
            row["mean_freq"] > row["null_p95_freq"]
            and (row["fdr_qvalue"] <= fdr_alpha or not enough_nulls_for_fdr)
        )
        if row["negative_control_pass"] and row["seed_support"] >= 0.8:
            row["stability_class"] = "stable_candidate"
        elif row["mean_freq"] <= row["null_p95_freq"]:
            row["stability_class"] = "null_like_edge"
        elif row["seed_support"] >= 0.5 or row["group_support_rate"] > 0:
            row["stability_class"] = "context_specific_candidate"
        else:
            row["stability_class"] = "unstable_candidate"

    return pd.DataFrame(rows, dtype=object)


def build_negative_control_report(edge_stability: pd.DataFrame, metadata: Mapping[str, Any] | None = None) -> str:
    """Render a conservative Markdown report for causal negative controls."""
    metadata = dict(metadata or {})
    n_edges = int(edge_stability["base_edge"].sum()) if "base_edge" in edge_stability else 0
    n_pass = int(edge_stability["negative_control_pass"].sum()) if "negative_control_pass" in edge_stability else 0
    level = metadata.get("interpretation", "exploratory causal-candidate audit")
    lines = [
        "# Causal Stability and Negative Control Audit",
        "",
        f"- Evidence level: `{level}`",
        f"- Base directed edges: {n_edges}",
        f"- Edges passing negative controls: {n_pass}",
        "",
        "These results are stability diagnostics for mechanism hypotheses and computational priorities; they are not treatment conclusions or sample-level causal proof.",
    ]
    return "\n".join(lines) + "\n"


def load_step2_audit_inputs(step2_dir: str | Path) -> dict[str, Any]:
    """Load the stable subset of Step2 artifacts needed for causal audit."""
    root = Path(step2_dir)
    adjacency_path = root / "causal_adjacency.npy"
    if not adjacency_path.exists():
        raise FileNotFoundError(f"Missing Step2 artifact: {adjacency_path}")
    freq_path = root / "bootstrap_freq_matrix.npy"
    if not freq_path.exists():
        freq_path = root / "bootstrap_freq.npy"
    if not freq_path.exists():
        raise FileNotFoundError(f"Missing Step2 bootstrap artifact under: {root}")

    adjacency = np.load(adjacency_path)
    bootstrap_freq = np.load(freq_path)
    node_info_path = root / "node_info.json"
    if node_info_path.exists():
        node_info = json.loads(node_info_path.read_text(encoding="utf-8"))
        node_labels = list(node_info.get("node_labels", []))
        type_mapping = dict(node_info.get("type_mapping", {}))
    else:
        node_labels = [f"node_{idx}" for idx in range(adjacency.shape[0])]
        type_mapping = {label: label for label in node_labels}

    metadata_path = root / "causal_input_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "adjacency": adjacency,
        "bootstrap_freq": bootstrap_freq,
        "node_labels": node_labels,
        "type_mapping": type_mapping,
        "causal_input_metadata": metadata,
    }


def write_causal_stability_outputs(
    output_dir: str | Path,
    edge_stability: pd.DataFrame,
    group_consistency: Mapping[str, Any],
    negative_control_report: str,
) -> dict[str, Path]:
    """Write causal audit sidecar artifacts to a plain output directory."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    edge_path = root / "edge_stability.csv"
    group_path = root / "platform_consistency.json"
    summary_path = root / "causal_audit_summary.json"
    report_path = root / "negative_control_report.md"
    edge_stability.to_csv(edge_path, index=False)
    group_path.write_text(json.dumps(group_consistency, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "n_edges": int(edge_stability["base_edge"].sum()) if "base_edge" in edge_stability else 0,
        "n_negative_control_pass": int(edge_stability["negative_control_pass"].sum())
        if "negative_control_pass" in edge_stability
        else 0,
        "stability_classes": edge_stability["stability_class"].value_counts().to_dict()
        if "stability_class" in edge_stability
        else {},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(negative_control_report, encoding="utf-8")
    return {
        "edge_stability": edge_path,
        "platform_consistency": group_path,
        "causal_audit_summary": summary_path,
        "negative_control_report": report_path,
    }


def run_causal_stability_audit(
    step2_dir: str | Path,
    output_dir: str | Path | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run a lightweight audit from existing Step2 artifacts."""
    inputs = load_step2_audit_inputs(step2_dir)
    edge_table = build_edge_stability_table(
        adjacency=inputs["adjacency"],
        bootstrap_freq=inputs["bootstrap_freq"],
        node_labels=inputs["node_labels"],
        type_mapping=inputs["type_mapping"],
        causal_input_metadata=inputs["causal_input_metadata"],
        threshold=threshold,
    )
    group_consistency = summarize_group_consistency(None, inputs["node_labels"], threshold=threshold)
    report = build_negative_control_report(edge_table, metadata=inputs["causal_input_metadata"])
    paths = (
        write_causal_stability_outputs(output_dir, edge_table, group_consistency, report)
        if output_dir is not None
        else {}
    )
    return {
        "edge_stability": edge_table,
        "platform_consistency": group_consistency,
        "negative_control_report": report,
        "output_paths": paths,
    }
