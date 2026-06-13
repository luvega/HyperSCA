"""Build behavior grammar rules from HyperSCA discovery artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.behavior_grammar.rules import BehaviorRule, RuleParameters, RuleSet


@dataclass(frozen=True)
class DiscoveryTables:
    run_id: str
    manifest_path: Path
    ranking: pd.DataFrame
    causal_edges: pd.DataFrame
    niche_mapping: pd.DataFrame
    cluster_expression: pd.DataFrame


def load_discovery_tables(manifest_path: str | Path) -> DiscoveryTables:
    """Load the discovery tables needed to construct behavior rules."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    artifacts = [str(item) for item in manifest.get("artifacts", [])]

    ranking = _read_first_table(
        run_dir,
        artifacts,
        [
            "scoring/target_ranking.csv",
            "scoring/candidate_scores.csv",
            "candidates/candidate_pool.csv",
        ],
    )
    causal_edges = _read_causal_edges(run_dir, artifacts)
    niche_mapping = _read_first_table(
        run_dir,
        artifacts,
        [
            "niche/target_to_unified_niche.csv",
            "niche/target_niche_mapping.csv",
            "niche/combo_to_unified_niche.csv",
        ],
        required=False,
    )
    cluster_expression = _read_first_table(
        run_dir,
        artifacts,
        [
            "expression/cluster_expression.csv",
            "causal/hyperbolic/cluster_expr.csv",
            "causal/euclidean/cluster_expr.csv",
        ],
        required=False,
    )
    return DiscoveryTables(
        run_id=str(manifest.get("run_id") or run_dir.name),
        manifest_path=manifest_path,
        ranking=ranking,
        causal_edges=causal_edges,
        niche_mapping=niche_mapping,
        cluster_expression=cluster_expression,
    )


def build_rules_from_discovery(manifest_path: str | Path, *, max_rules: int = 30) -> RuleSet:
    """Generate data-driven behavior rules from a discovery run manifest."""
    tables = load_discovery_tables(manifest_path)
    ranking = _normalize_ranking(tables.ranking, tables.niche_mapping)
    edge_index = _index_edges(tables.causal_edges)

    rules: list[BehaviorRule] = []
    for _, row in ranking.head(max_rules).iterrows():
        gene = str(row.get("gene", "") or row.get("target", "")).strip()
        if not gene:
            continue
        edge = edge_index.get(gene.upper(), {})
        cell_type = _select_cell_type(row, edge)
        behavior = _infer_behavior(gene, row, edge)
        direction = _infer_direction(gene, behavior)
        score = _coerce_float(row.get("final_score", row.get("score", 0.5)), default=0.5)
        half_max = max(0.05, 1.0 - min(score, 0.95))
        parameters = RuleParameters(
            base=0.05 if direction == "increases" else 0.15,
            saturation=max(0.2, min(1.0, 0.35 + score)),
            half_max=half_max,
            hill_power=2.0,
        )
        evidence_refs = tuple(
            item
            for item in [
                f"manifest:{tables.manifest_path.name}",
                f"score:{score:.3f}",
                _edge_evidence(edge),
                _niche_evidence(row),
            ]
            if item
        )
        rules.append(
            BehaviorRule(
                cell_type=cell_type,
                signal=gene,
                direction=direction,
                behavior=behavior,
                response_function="hill",
                parameters=parameters,
                evidence_refs=evidence_refs,
            )
        )

    ruleset = RuleSet(
        run_id=tables.run_id,
        source_manifest=str(tables.manifest_path),
        rules=tuple(rules),
        metadata={
            "source": "hypersca_target_discovery",
            "n_ranking_rows": int(len(tables.ranking)),
            "n_causal_edges": int(len(tables.causal_edges)),
            "max_rules": int(max_rules),
        },
    )
    ruleset.validate()
    return ruleset


def _read_first_table(
    run_dir: Path,
    artifacts: Iterable[str],
    candidates: list[str],
    *,
    required: bool = True,
) -> pd.DataFrame:
    artifact_set = {Path(item).as_posix(): run_dir / item for item in artifacts}
    for rel in candidates:
        path = artifact_set.get(Path(rel).as_posix(), run_dir / rel)
        if path.exists():
            return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"none of the expected discovery tables exist: {candidates}")
    return pd.DataFrame()


def _read_causal_edges(run_dir: Path, artifacts: Iterable[str]) -> pd.DataFrame:
    csv_edges = _read_first_table(
        run_dir,
        artifacts,
        [
            "causal/hyperbolic/causal_edges.csv",
            "causal/hyperbolic/flow_edges.csv",
            "causal/euclidean/causal_edges.csv",
        ],
        required=False,
    )
    if not csv_edges.empty:
        return csv_edges
    for rel in ["causal/hyperbolic/flow_edges.json", "causal/euclidean/flow_edges.json"]:
        path = run_dir / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        edges = payload.get("flow_edges", payload if isinstance(payload, list) else [])
        return pd.DataFrame(edges)
    return pd.DataFrame()


def _normalize_ranking(ranking: pd.DataFrame, niche_mapping: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    if "target" in out and "gene" not in out:
        out = out.rename(columns={"target": "gene"})
    if "gene" not in out:
        out["gene"] = []
    if "final_score" not in out:
        out["final_score"] = pd.to_numeric(out.get("score", 0.5), errors="coerce").fillna(0.5)
    if not niche_mapping.empty:
        right = niche_mapping.copy()
        if "target" in right and "gene" not in right:
            right = right.rename(columns={"target": "gene"})
        keep = [col for col in ["gene", "niche", "niche_label", "top_node", "broad_type"] if col in right]
        if "gene" in keep:
            out = out.merge(right[keep].drop_duplicates("gene"), on="gene", how="left")
    return out.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)


def _index_edges(edges: pd.DataFrame) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if edges.empty:
        return indexed
    for _, row in edges.iterrows():
        values = {str(key): value for key, value in row.items()}
        candidates = [
            values.get("ligand"),
            values.get("receptor"),
            values.get("gene"),
            values.get("source"),
            values.get("target"),
        ]
        for candidate in candidates:
            if candidate is None or pd.isna(candidate):
                continue
            indexed.setdefault(str(candidate).upper(), values)
    return indexed


def _select_cell_type(row: pd.Series, edge: dict[str, Any]) -> str:
    for key in ["cell_type", "celltype", "top_node", "broad_type", "source"]:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).split(";")[0].strip()
    for key in ["source_celltype", "source", "cell_type"]:
        value = edge.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    value = row.get("celltypes_neu")
    if value is not None and not pd.isna(value) and str(value).strip():
        return str(value).split(";")[0].strip()
    return "TME_cell"


def _infer_behavior(gene: str, row: pd.Series, edge: dict[str, Any]) -> str:
    gene_upper = gene.upper()
    text = " ".join(str(value).upper() for value in list(row.values) + list(edge.values()) if value is not None)
    if gene_upper in {"PDCD1", "CTLA4", "LAG3", "TIGIT"} or "CHECKPOINT" in text:
        return "exhaustion"
    if gene_upper in {"GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10"} or "CD8" in text:
        return "attack" if gene_upper in {"GZMB", "PRF1", "IFNG"} else "migration"
    if gene_upper in {"TGFB1", "POSTN", "FN1", "COL1A1", "COL1A2"} or "STROMAL" in text:
        return "transition"
    if gene_upper in {"EGF", "EREG", "AREG", "HGF", "VEGFA"}:
        return "migration"
    if gene_upper in {"MKI67", "TOP2A", "PCNA"}:
        return "proliferation"
    return "secretion" if edge else "transition"


def _infer_direction(gene: str, behavior: str) -> str:
    if behavior == "exhaustion":
        return "increases"
    if gene.upper() in {"PDCD1", "CTLA4"}:
        return "decreases"
    return "increases"


def _edge_evidence(edge: dict[str, Any]) -> str:
    if not edge:
        return ""
    source = edge.get("source", edge.get("source_celltype", ""))
    target = edge.get("target", edge.get("target_celltype", ""))
    if source or target:
        return f"causal:{source}->{target}"
    return "causal:flow_edge"


def _niche_evidence(row: pd.Series) -> str:
    for key in ["niche", "niche_label", "broad_type"]:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return f"niche:{value}"
    return ""


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed
