"""Run VisiumHD 8um COMMOT reproduction and HyperSCA OT-flow comparison.

The raw 8um VisiumHD matrix has hundreds of thousands of bins. COMMOT's public
API constructs or consumes an ``n_obs x n_obs`` distance matrix, so this script
uses the full bin-level data to build cell-type aggregates and spatial contact
constraints, then runs COMMOT and the HyperSCA LR flow sidecar on the same
aggregated nodes.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.causal.signaling_flow import CRC_LR_DATABASE
from src.discovery.target_discovery.communication_flow import build_communication_flow


DEFAULT_DATA_DIR = (
    ROOT
    / "data"
    / "VisiumHD_HumanColon_Oliveira"
    / "binned_outputs"
    / "square_008um"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "visiumhd8um_commot_ot_comparison"
DEFAULT_REPORT = ROOT / "docs" / "research" / "visiumhd8um_commot_ot_comparison_report.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use full VisiumHD 8um bins to reproduce a cell-type-level COMMOT "
            "baseline and compare it with the HyperSCA distance-constrained "
            "LR OT-flow sidecar."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--block-size", type=int, default=4000)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--min-node-bins", type=int, default=100)
    parser.add_argument("--commot-nitermax", type=int, default=2000)
    parser.add_argument("--cot-eps-p", type=float, default=0.1)
    parser.add_argument("--cot-rho", type=float, default=10.0)
    parser.add_argument("--hypersca-alpha", type=float, default=0.5)
    parser.add_argument("--hypersca-beta", type=float, default=0.5)
    parser.add_argument("--hypersca-epsilon", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "commot").mkdir(exist_ok=True)
    (output_dir / "hypersca_ot").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_path = args.data_dir / "filtered_feature_bc_matrix.h5"
    positions_path = args.data_dir / "spatial" / "tissue_positions.parquet"
    decon_path = args.data_dir / "deconvolution.csv.gz"
    clustering_path = args.data_dir / "clustering.csv.gz"
    for path in [matrix_path, positions_path, decon_path, clustering_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    lr_entries = _unique_lr_entries(CRC_LR_DATABASE)
    selected_genes = sorted(
        {
            gene
            for entry in CRC_LR_DATABASE
            for gene in [entry["ligand"], entry["receptor"], entry["tf"], entry["target"]]
        }
    )
    print(f"Loading selected genes from 10x H5: {len(selected_genes)} requested genes")
    expr_counts, total_counts, barcodes, present_genes, missing_genes, h5_shape = _load_selected_gene_counts(
        matrix_path,
        selected_genes,
        block_size=args.block_size,
    )
    print(f"Loaded expression matrix: {expr_counts.shape[0]} bins x {expr_counts.shape[1]} genes")

    obs = _load_observation_metadata(
        barcodes=barcodes,
        positions_path=positions_path,
        decon_path=decon_path,
        clustering_path=clustering_path,
    )
    obs["node"] = obs["DeconLabel1"].map(_harmonize_label)
    obs.loc[obs["node"].isna(), "node"] = "Unknown"
    obs["node"] = obs["node"].astype(str)

    expr_norm = _log_cp10k(expr_counts, total_counts)
    node_tables = _build_node_tables(
        obs=obs,
        expr_norm=expr_norm,
        genes=present_genes,
        min_node_bins=args.min_node_bins,
    )
    cluster_expr = node_tables["cluster_expr"]
    node_order = cluster_expr.index.astype(str).tolist()
    print(f"Aggregated nodes: {len(node_order)}")

    spatial = _build_spatial_contact(
        obs=obs,
        node_order=node_order,
        knn_k=args.knn_k,
    )
    dist_matrix = cdist(
        node_tables["node_geometry"][["x_centroid", "y_centroid"]].to_numpy(dtype=float),
        node_tables["node_geometry"][["x_centroid", "y_centroid"]].to_numpy(dtype=float),
    )

    _write_input_artifacts(
        output_dir=output_dir,
        cluster_expr=cluster_expr,
        node_tables=node_tables,
        spatial=spatial,
        present_genes=present_genes,
        missing_genes=missing_genes,
        obs=obs,
    )

    print("Running COMMOT on full-data cell-type aggregates")
    commot_result = _run_commot(
        cluster_expr=cluster_expr,
        node_geometry=node_tables["node_geometry"],
        dist_matrix=dist_matrix,
        lr_entries=lr_entries,
        nitermax=args.commot_nitermax,
        cot_eps_p=args.cot_eps_p,
        cot_rho=args.cot_rho,
    )
    commot_all = commot_result["all_edges"]
    commot_expected = _filter_crc_expected_edges(commot_all, lr_entries, set(node_order))
    commot_pathways = _summarize_edge_table(
        commot_expected,
        score_col="commot_score",
        norm_col="commot_normalized",
    )
    commot_all.to_csv(output_dir / "commot" / "commot_all_node_lr_edges.csv", index=False)
    commot_expected.to_csv(output_dir / "commot" / "commot_crc_expected_edges.csv", index=False)
    commot_pathways.to_csv(output_dir / "commot" / "commot_pathway_summary.csv", index=False)

    print("Running HyperSCA OT-flow sidecar on the same aggregates")
    hypersca_result = build_communication_flow(
        mode="visiumhd8um_full_aggregate",
        cluster_expr=cluster_expr,
        spatial_adjacency=spatial["adjacency"],
        geometry_result={"dist_matrix": dist_matrix},
        causal_result={
            "node_labels": node_order,
            "type_mapping": {node: node for node in node_order},
            "causal_adjacency": np.zeros((len(node_order), len(node_order)), dtype=float),
        },
        alpha=args.hypersca_alpha,
        beta=args.hypersca_beta,
        epsilon=args.hypersca_epsilon,
    )
    hypersca_edges = hypersca_result["lr_flow_edges"].copy()
    hypersca_pathways = hypersca_result["pathway_summary"].copy()
    hypersca_edges.to_csv(output_dir / "hypersca_ot" / "lr_flow_edges.csv", index=False)
    np.save(output_dir / "hypersca_ot" / "flow_matrix.npy", hypersca_result["flow_matrix"])
    hypersca_pathways.to_csv(output_dir / "hypersca_ot" / "pathway_summary.csv", index=False)
    (output_dir / "hypersca_ot" / "baseline_comparison.json").write_text(
        json.dumps(hypersca_result["baseline_comparison"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "hypersca_ot" / "direction_consistency.json").write_text(
        json.dumps(hypersca_result["direction_consistency"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paired = _build_paired_comparison(commot_expected, hypersca_edges)
    method_summary = _build_method_summary(
        paired=paired,
        commot_edges=commot_expected,
        hypersca_edges=hypersca_edges,
        node_order=node_order,
        obs=obs,
        present_genes=present_genes,
        missing_genes=missing_genes,
    )
    paired.to_csv(output_dir / "comparison" / "paired_lr_edge_comparison.csv", index=False)
    method_summary.to_csv(output_dir / "comparison" / "method_summary.csv", index=False)

    figure_path = _plot_comparison(
        output_dir=output_dir / "figures",
        paired=paired,
        commot_pathways=commot_pathways,
        hypersca_edges=hypersca_edges,
        commot_edges=commot_expected,
        node_order=node_order,
    )
    report_figure_path = _copy_report_figure(figure_path, args.report_path.parent / "figures")
    report_path = _write_report(
        report_path=args.report_path,
        output_dir=output_dir,
        figure_path=report_figure_path,
        method_summary=method_summary,
        paired=paired,
        commot_pathways=commot_pathways,
        hypersca_pathways=hypersca_pathways,
        node_tables=node_tables,
        h5_shape=h5_shape,
        present_genes=present_genes,
        missing_genes=missing_genes,
        elapsed_seconds=time.time() - started,
        args=args,
    )
    manifest = {
        "run_id": "visiumhd8um_commot_ot_comparison",
        "elapsed_seconds": time.time() - started,
        "data_dir": str(args.data_dir),
        "h5_shape_genes_by_barcodes": h5_shape,
        "n_expression_barcodes": int(len(barcodes)),
        "n_nodes": int(len(node_order)),
        "present_genes": present_genes,
        "missing_genes": missing_genes,
        "artifacts": _relative_artifacts(output_dir),
        "report": str(report_path),
        "figure": str(figure_path),
        "report_figure": str(report_figure_path),
        "raw_bin_level_commot_boundary": (
            "COMMOT's public API uses an n_obs x n_obs distance matrix; raw 545k-bin "
            "COMMOT is not memory-feasible, so this run uses all bins to form "
            "cell-type aggregates before COMMOT."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")
    return 0


def _decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _unique_lr_entries(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    rows: list[dict[str, str]] = []
    for entry in entries:
        key = (
            str(entry["ligand"]),
            str(entry["receptor"]),
            str(entry.get("pathway", "")),
            str(entry["source_type"]),
            str(entry["target_type"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({k: str(entry[k]) for k in ["ligand", "receptor", "pathway", "source_type", "target_type"]})
    return rows


def _load_selected_gene_counts(
    matrix_path: Path,
    selected_genes: list[str],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], list[int]]:
    with h5py.File(matrix_path, "r") as handle:
        shape = [int(x) for x in handle["matrix/shape"][:].tolist()]
        n_genes, n_barcodes = shape
        gene_names = _decode(handle["matrix/features/name"][:])
        barcodes = _decode(handle["matrix/barcodes"][:])
        upper_to_idx = {name.upper(): idx for idx, name in enumerate(gene_names)}
        selected = []
        selected_indices = []
        missing = []
        for gene in selected_genes:
            idx = upper_to_idx.get(gene.upper())
            if idx is None:
                missing.append(gene)
            else:
                selected.append(gene_names[idx])
                selected_indices.append(idx)

        expr = np.zeros((n_barcodes, len(selected)), dtype=np.float32)
        totals = np.zeros(n_barcodes, dtype=np.float32)
        gene_pos = np.full(n_genes, -1, dtype=np.int32)
        for pos, idx in enumerate(selected_indices):
            gene_pos[idx] = pos

        indptr = handle["matrix/indptr"][:]
        indices_ds = handle["matrix/indices"]
        data_ds = handle["matrix/data"]
        for start_col in range(0, n_barcodes, block_size):
            end_col = min(start_col + block_size, n_barcodes)
            start = int(indptr[start_col])
            end = int(indptr[end_col])
            if start == end:
                continue
            idx = indices_ds[start:end]
            vals = data_ds[start:end].astype(np.float32)
            lengths = np.diff(indptr[start_col : end_col + 1]).astype(np.int64)
            local_cols = np.repeat(np.arange(end_col - start_col, dtype=np.int32), lengths)
            cols = local_cols + start_col
            np.add.at(totals, cols, vals)
            mapped = gene_pos[idx]
            mask = mapped >= 0
            if np.any(mask):
                np.add.at(expr, (cols[mask], mapped[mask]), vals[mask])
            if (start_col // block_size) % 25 == 0:
                print(f"  processed columns {start_col:,}-{end_col:,} / {n_barcodes:,}")
    return expr, totals, barcodes, selected, missing, shape


def _load_observation_metadata(
    barcodes: list[str],
    positions_path: Path,
    decon_path: Path,
    clustering_path: Path,
) -> pd.DataFrame:
    obs = pd.DataFrame({"barcode": barcodes})
    positions = pd.read_parquet(
        positions_path,
        columns=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ).set_index("barcode")
    decon = pd.read_csv(
        decon_path,
        usecols=["barcode", "DeconClass", "DeconLabel1", "DeconLabel2"],
    ).set_index("barcode")
    clustering = pd.read_csv(clustering_path, usecols=["barcode", "cluster"]).set_index("barcode")
    obs = obs.join(positions, on="barcode")
    obs = obs.join(decon, on="barcode")
    obs = obs.join(clustering, on="barcode")
    obs["x"] = pd.to_numeric(obs["pxl_col_in_fullres"], errors="coerce")
    obs["y"] = pd.to_numeric(obs["pxl_row_in_fullres"], errors="coerce")
    return obs


def _harmonize_label(label: Any) -> str:
    if pd.isna(label):
        return "Unknown"
    text = str(label).strip()
    lower = text.lower()
    if not text:
        return "Unknown"
    if "caf" in lower or "fibroblast" in lower or "myofibroblast" in lower:
        return "CAF"
    if "macrophage" in lower or "tam" in lower or "mono" in lower:
        return "TAM"
    if "cd8" in lower:
        return "CD8T"
    if "treg" in lower or "regulatory t" in lower:
        return "Treg"
    if any(token in lower for token in ["tumor", "epithelial", "enterocyte", "goblet", "tuft", "neuroendocrine"]):
        return "Epithelial"
    if "endothelial" in lower:
        return "Endothelial"
    if any(token in lower for token in ["pericyte", "smooth muscle", "vsm", "unknown iii", "sm stress"]):
        return "Stromal"
    if "cd4" in lower:
        return "CD4T"
    if any(token in lower for token in ["plasma", "b cell", "memory b", "mature b"]):
        return "B"
    if "neutrophil" in lower:
        return "Neutrophil"
    if any(token in lower for token in ["mast", "nk", "dc", "pdc", "cdc"]):
        return "Other_immune"
    return text.replace(" ", "_")


def _log_cp10k(expr_counts: np.ndarray, total_counts: np.ndarray) -> np.ndarray:
    scale = np.zeros_like(expr_counts, dtype=np.float32)
    nonzero = total_counts > 0
    scale[nonzero] = expr_counts[nonzero] / total_counts[nonzero, None] * 1e4
    return np.log1p(scale).astype(np.float32, copy=False)


def _build_node_tables(
    obs: pd.DataFrame,
    expr_norm: np.ndarray,
    genes: list[str],
    min_node_bins: int,
) -> dict[str, pd.DataFrame]:
    valid = obs["x"].notna() & obs["y"].notna() & obs["node"].notna()
    counts = obs.loc[valid, "node"].value_counts().rename_axis("node").reset_index(name="n_bins")
    included = counts.loc[counts["n_bins"] >= min_node_bins, "node"].astype(str).tolist()
    included_set = set(included)
    node_mask = valid & obs["node"].isin(included_set)

    expr_frame = pd.DataFrame(expr_norm[node_mask.to_numpy()], columns=genes)
    expr_frame["node"] = obs.loc[node_mask, "node"].to_numpy()
    cluster_expr = expr_frame.groupby("node", sort=False)[genes].mean()
    preferred = ["Epithelial", "CAF", "TAM", "CD8T", "Treg"]
    ordered = [node for node in preferred if node in cluster_expr.index]
    ordered += [node for node in cluster_expr.index.astype(str).tolist() if node not in ordered]
    cluster_expr = cluster_expr.reindex(ordered).fillna(0.0)

    node_geometry = (
        obs.loc[node_mask, ["node", "x", "y"]]
        .groupby("node", sort=False)
        .agg(x_centroid=("x", "mean"), y_centroid=("y", "mean"))
        .reindex(ordered)
        .reset_index()
    )
    counts = counts.set_index("node").reindex(ordered).fillna(0).reset_index()
    counts["n_bins"] = counts["n_bins"].astype(int)
    raw_label_counts = (
        obs.loc[valid, "DeconLabel1"].fillna("Unknown").value_counts().rename_axis("raw_label").reset_index(name="n_bins")
    )
    return {
        "cluster_expr": cluster_expr,
        "node_counts": counts,
        "node_geometry": node_geometry,
        "raw_label_counts": raw_label_counts,
    }


def _build_spatial_contact(obs: pd.DataFrame, node_order: list[str], knn_k: int) -> dict[str, np.ndarray | pd.DataFrame]:
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    valid = obs["x"].notna() & obs["y"].notna() & obs["node"].isin(set(node_to_idx))
    coords = obs.loc[valid, ["x", "y"]].to_numpy(dtype=float)
    node_idx = obs.loc[valid, "node"].map(node_to_idx).to_numpy(dtype=np.int32)
    n_nodes = len(node_order)
    counts = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if len(coords) <= 1:
        adjacency = counts.copy()
    else:
        n_neighbors = min(knn_k + 1, len(coords))
        model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
        model.fit(coords)
        _, nn = model.kneighbors(coords, return_distance=True)
        nn = nn[:, 1:]
        src = np.repeat(np.arange(len(coords), dtype=np.int64), nn.shape[1])
        tgt = nn.reshape(-1)
        np.add.at(counts, (node_idx[src], node_idx[tgt]), 1.0)
        row_sum = counts.sum(axis=1, keepdims=True)
        adjacency = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
    return {
        "counts": counts,
        "adjacency": adjacency,
        "counts_table": pd.DataFrame(counts, index=node_order, columns=node_order),
        "adjacency_table": pd.DataFrame(adjacency, index=node_order, columns=node_order),
        "n_spots_used": int(len(coords)),
    }


def _write_input_artifacts(
    output_dir: Path,
    cluster_expr: pd.DataFrame,
    node_tables: dict[str, pd.DataFrame],
    spatial: dict[str, Any],
    present_genes: list[str],
    missing_genes: list[str],
    obs: pd.DataFrame,
) -> None:
    cluster_expr.to_csv(output_dir / "selected_gene_expression_by_node.csv")
    node_tables["node_counts"].to_csv(output_dir / "celltype_counts.csv", index=False)
    node_tables["raw_label_counts"].to_csv(output_dir / "raw_decon_label_counts.csv", index=False)
    node_tables["node_geometry"].to_csv(output_dir / "node_geometry.csv", index=False)
    spatial["counts_table"].to_csv(output_dir / "spatial_contact_counts.csv")
    spatial["adjacency_table"].to_csv(output_dir / "spatial_contact_adjacency.csv")
    manifest = {
        "n_expression_bins": int(len(obs)),
        "n_bins_with_position_and_node": int(spatial["n_spots_used"]),
        "present_genes": present_genes,
        "missing_genes": missing_genes,
        "node_counts": node_tables["node_counts"].to_dict(orient="records"),
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_commot(
    cluster_expr: pd.DataFrame,
    node_geometry: pd.DataFrame,
    dist_matrix: np.ndarray,
    lr_entries: list[dict[str, str]],
    nitermax: int,
    cot_eps_p: float,
    cot_rho: float,
) -> dict[str, pd.DataFrame]:
    import commot as ct

    df_ligrec = pd.DataFrame(
        [[entry["ligand"], entry["receptor"], entry["pathway"]] for entry in lr_entries],
        columns=["ligand", "receptor", "pathway"],
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(cluster_expr.to_numpy(dtype=np.float64)),
        obs=pd.DataFrame(index=cluster_expr.index.astype(str)),
        var=pd.DataFrame(index=cluster_expr.columns.astype(str)),
    )
    adata.obsm["spatial"] = node_geometry[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    adata.obsp["spatial_distance"] = dist_matrix
    nonzero = dist_matrix[dist_matrix > 0]
    dis_thr = float(nonzero.max() * 1.001) if len(nonzero) else 1.0
    ct.tl.spatial_communication(
        adata,
        database_name="crc_lr",
        df_ligrec=df_ligrec,
        pathway_sum=True,
        heteromeric=False,
        dis_thr=dis_thr,
        cot_eps_p=cot_eps_p,
        cot_rho=cot_rho,
        cot_nitermax=nitermax,
    )
    rows: list[dict[str, Any]] = []
    nodes = cluster_expr.index.astype(str).tolist()
    for entry in lr_entries:
        key = f"commot-crc_lr-{entry['ligand']}-{entry['receptor']}"
        if key not in adata.obsp:
            continue
        matrix = adata.obsp[key]
        arr = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        max_score = float(np.nanmax(arr)) if arr.size else 0.0
        for i, source_node in enumerate(nodes):
            for j, target_node in enumerate(nodes):
                if i == j:
                    continue
                score = float(arr[i, j])
                rows.append(
                    {
                        "pathway": entry["pathway"],
                        "ligand": entry["ligand"],
                        "receptor": entry["receptor"],
                        "expected_source_type": entry["source_type"],
                        "expected_target_type": entry["target_type"],
                        "source_node": source_node,
                        "target_node": target_node,
                        "commot_score": score,
                        "commot_normalized": score / max_score if max_score > 0 else 0.0,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        total_max = float(frame["commot_score"].max())
        frame["commot_global_normalized"] = frame["commot_score"] / total_max if total_max > 0 else 0.0
    return {"all_edges": frame}


def _filter_crc_expected_edges(
    commot_all: pd.DataFrame,
    lr_entries: list[dict[str, str]],
    available_nodes: set[str],
) -> pd.DataFrame:
    if commot_all.empty:
        return commot_all.copy()
    allowed = {
        (entry["ligand"], entry["receptor"], entry["source_type"], entry["target_type"])
        for entry in lr_entries
        if entry["source_type"] in available_nodes and entry["target_type"] in available_nodes
    }
    mask = commot_all.apply(
        lambda row: (
            row["ligand"],
            row["receptor"],
            row["source_node"],
            row["target_node"],
        )
        in allowed,
        axis=1,
    )
    return commot_all.loc[mask].copy()


def _summarize_edge_table(pd_edges: pd.DataFrame, score_col: str, norm_col: str) -> pd.DataFrame:
    if pd_edges.empty:
        return pd.DataFrame(columns=["pathway", "n_edges", "total_score", "mean_score", "max_score", "top_edge"])
    rows: list[dict[str, Any]] = []
    for pathway, group in pd_edges.groupby("pathway"):
        top = group.sort_values(score_col, ascending=False).iloc[0]
        rows.append(
            {
                "pathway": pathway,
                "n_edges": int(len(group)),
                "total_score": float(group[score_col].sum()),
                "mean_score": float(group[score_col].mean()),
                "max_score": float(group[score_col].max()),
                "mean_normalized": float(group[norm_col].mean()) if norm_col in group else 0.0,
                "top_edge": f"{top['source_node']}→{top['target_node']}:{top['ligand']}-{top['receptor']}",
            }
        )
    return pd.DataFrame(rows).sort_values("total_score", ascending=False)


def _build_paired_comparison(commot_expected: pd.DataFrame, hypersca_edges: pd.DataFrame) -> pd.DataFrame:
    keys = ["pathway", "ligand", "receptor", "source_node", "target_node"]
    if commot_expected.empty or hypersca_edges.empty:
        return pd.DataFrame(columns=keys + ["commot_score", "commot_global_normalized", "flow_score", "normalized_flow"])
    right = hypersca_edges.rename(columns={"target_gene": "hypersca_target_gene"})
    keep_cols = keys + [
        "flow_score",
        "normalized_flow",
        "spatial_weight",
        "geometry_distance",
        "blended_weight",
        "hypersca_target_gene",
        "tf",
    ]
    merged = commot_expected.merge(right[keep_cols], on=keys, how="outer")
    for col in ["commot_score", "commot_global_normalized", "flow_score", "normalized_flow"]:
        if col in merged:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged.sort_values(["flow_score", "commot_score"], ascending=False)


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    if len(x) < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    return float(pearsonr(x, y)[0])


def _build_method_summary(
    paired: pd.DataFrame,
    commot_edges: pd.DataFrame,
    hypersca_edges: pd.DataFrame,
    node_order: list[str],
    obs: pd.DataFrame,
    present_genes: list[str],
    missing_genes: list[str],
) -> pd.DataFrame:
    rows = [
        {"metric": "n_expression_bins", "value": int(len(obs))},
        {"metric": "n_nodes", "value": int(len(node_order))},
        {"metric": "n_selected_genes_present", "value": int(len(present_genes))},
        {"metric": "n_selected_genes_missing", "value": int(len(missing_genes))},
        {"metric": "n_commot_expected_edges", "value": int(len(commot_edges))},
        {"metric": "n_hypersca_edges", "value": int(len(hypersca_edges))},
        {"metric": "total_commot_score", "value": float(commot_edges.get("commot_score", pd.Series(dtype=float)).sum())},
        {"metric": "total_hypersca_flow", "value": float(hypersca_edges.get("flow_score", pd.Series(dtype=float)).sum())},
        {
            "metric": "paired_spearman_global_norm",
            "value": _safe_corr(
                paired.get("commot_global_normalized", pd.Series(dtype=float)),
                paired.get("normalized_flow", pd.Series(dtype=float)),
                "spearman",
            ),
        },
        {
            "metric": "paired_pearson_global_norm",
            "value": _safe_corr(
                paired.get("commot_global_normalized", pd.Series(dtype=float)),
                paired.get("normalized_flow", pd.Series(dtype=float)),
                "pearson",
            ),
        },
    ]
    return pd.DataFrame(rows)


def _edge_matrix(edges: pd.DataFrame, node_order: list[str], score_col: str) -> np.ndarray:
    matrix = np.zeros((len(node_order), len(node_order)), dtype=float)
    idx = {node: i for i, node in enumerate(node_order)}
    if edges.empty or score_col not in edges:
        return matrix
    for row in edges.itertuples(index=False):
        source = getattr(row, "source_node")
        target = getattr(row, "target_node")
        if source in idx and target in idx:
            matrix[idx[source], idx[target]] += float(getattr(row, score_col))
    if matrix.max() > 0:
        matrix = matrix / matrix.max()
    return matrix


def _plot_comparison(
    output_dir: Path,
    paired: pd.DataFrame,
    commot_pathways: pd.DataFrame,
    hypersca_edges: pd.DataFrame,
    commot_edges: pd.DataFrame,
    node_order: list[str],
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    hyper_path = _summarize_edge_table(hypersca_edges, "flow_score", "normalized_flow")
    pathway = (
        commot_pathways[["pathway", "total_score"]]
        .rename(columns={"total_score": "COMMOT"})
        .merge(hyper_path[["pathway", "total_score"]].rename(columns={"total_score": "HyperSCA OT"}), on="pathway", how="outer")
        .fillna(0.0)
    )
    if not pathway.empty:
        for col in ["COMMOT", "HyperSCA OT"]:
            max_value = float(pathway[col].max())
            pathway[col] = pathway[col] / max_value if max_value > 0 else 0.0
        pathway = pathway.sort_values(["HyperSCA OT", "COMMOT"], ascending=False).head(6)
        y = np.arange(len(pathway))
        ax.barh(y - 0.18, pathway["COMMOT"], height=0.34, color="#4C78A8", label="COMMOT")
        ax.barh(y + 0.18, pathway["HyperSCA OT"], height=0.34, color="#F58518", label="HyperSCA OT")
        ax.set_yticks(y)
        ax.set_yticklabels(pathway["pathway"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
    ax.set_xlabel("Within-method normalized pathway flow")
    ax.set_title("A. Pathway-level agreement")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    if not paired.empty:
        x = paired["commot_global_normalized"].to_numpy(dtype=float)
        y = paired["normalized_flow"].to_numpy(dtype=float)
        ax.scatter(x, y, s=24, color="#2F4B7C", alpha=0.72, edgecolor="white", linewidth=0.3)
        limit = max(float(np.nanmax(x)) if len(x) else 1.0, float(np.nanmax(y)) if len(y) else 1.0, 1.0)
        ax.plot([0, limit], [0, limit], color="#7F7F7F", lw=0.8, ls="--")
        rho = _safe_corr(paired["commot_global_normalized"], paired["normalized_flow"], "spearman")
        ax.text(
            0.03,
            0.92,
            f"Spearman rho = {rho:.2f}",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
    ax.set_xlabel("COMMOT normalized LR flow")
    ax.set_ylabel("HyperSCA OT normalized flow")
    ax.set_title("B. Matched LR edge scores")

    heatmap_image = None
    for panel_ax, title, edges, score_col in [
        (axes[1, 0], "C. COMMOT node-to-node flow", commot_edges, "commot_score"),
        (axes[1, 1], "D. HyperSCA OT node-to-node flow", hypersca_edges, "flow_score"),
    ]:
        matrix = _edge_matrix(edges, node_order, score_col)
        heatmap_image = panel_ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        panel_ax.set_xticks(range(len(node_order)))
        panel_ax.set_yticks(range(len(node_order)))
        panel_ax.set_xticklabels(node_order, rotation=45, ha="right")
        panel_ax.set_yticklabels(node_order)
        panel_ax.tick_params(axis="both", labelsize=7)
        panel_ax.set_title(title)
        panel_ax.set_xlabel("Receiver")
        panel_ax.set_ylabel("Sender")
    if heatmap_image is not None:
        fig.colorbar(
            heatmap_image,
            ax=[axes[1, 0], axes[1, 1]],
            fraction=0.025,
            pad=0.018,
            label="Normalized flow",
        )

    png_path = output_dir / "visiumhd8um_commot_ot_comparison.png"
    svg_path = output_dir / "visiumhd8um_commot_ot_comparison.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _copy_report_figure(figure_path: Path, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / figure_path.name
    shutil.copy2(figure_path, target)
    svg_source = figure_path.with_suffix(".svg")
    if svg_source.exists():
        shutil.copy2(svg_source, figure_dir / svg_source.name)
    return target


def _top_rows(frame: pd.DataFrame, n: int = 8) -> str:
    if frame.empty:
        return "_No rows._"
    try:
        return frame.head(n).to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.head(n).to_string(index=False) + "\n```"


def _write_report(
    report_path: Path,
    output_dir: Path,
    figure_path: Path,
    method_summary: pd.DataFrame,
    paired: pd.DataFrame,
    commot_pathways: pd.DataFrame,
    hypersca_pathways: pd.DataFrame,
    node_tables: dict[str, pd.DataFrame],
    h5_shape: list[int],
    present_genes: list[str],
    missing_genes: list[str],
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> Path:
    summary = {str(row.metric): row.value for row in method_summary.itertuples(index=False)}
    report = f"""# VisiumHD 8um COMMOT 与 HyperSCA OT-flow 对比

## 运行边界

- 输入矩阵：`{args.data_dir / "filtered_feature_bc_matrix.h5"}`，shape 为 `{h5_shape[0]:,} genes × {h5_shape[1]:,} barcodes`。
- 本轮使用全部 8um expression barcodes 聚合 cell-type nodes；空间约束由全部可定位 bins 的 kNN 接触图估计。
- COMMOT 原始 bin 级全量运行不可直接执行：其公开接口需要 `n_obs × n_obs` 距离矩阵，`545,913 × 545,913` dense 距离矩阵单项即超过 TB 级内存。
- 因此，本报告中的“全量复现”指：全量 bins 参与表达聚合与空间接触统计，COMMOT 和 HyperSCA OT-flow 在同一 cell-type 聚合图上对照。

## 数据规模

| 指标 | 数值 |
|---|---:|
| expression bins | {int(summary.get("n_expression_bins", 0)):,} |
| 聚合 nodes | {int(summary.get("n_nodes", 0)):,} |
| CRC LR/TF/target 基因命中 | {len(present_genes)} |
| 缺失基因 | {len(missing_genes)} |
| COMMOT expected LR edges | {int(summary.get("n_commot_expected_edges", 0)):,} |
| HyperSCA OT edges | {int(summary.get("n_hypersca_edges", 0)):,} |
| paired Spearman rho | {float(summary.get("paired_spearman_global_norm", 0.0)):.3f} |
| paired Pearson r | {float(summary.get("paired_pearson_global_norm", 0.0)):.3f} |
| elapsed seconds | {elapsed_seconds:.1f} |

## Cell-type nodes

{_top_rows(node_tables["node_counts"], 20)}

## Pathway comparison

COMMOT pathway summary:

{_top_rows(commot_pathways, 10)}

HyperSCA OT-flow pathway summary:

{_top_rows(hypersca_pathways, 10)}

## Paired LR edge comparison

{_top_rows(paired[["pathway", "ligand", "receptor", "source_node", "target_node", "commot_global_normalized", "normalized_flow"]], 12) if not paired.empty else "_No paired edges._"}

## Figure

![VisiumHD 8um COMMOT OT comparison]({_markdown_path(figure_path, report_path.parent)})

## 输出文件

- `{output_dir / "input_manifest.json"}`
- `{output_dir / "selected_gene_expression_by_node.csv"}`
- `{output_dir / "spatial_contact_adjacency.csv"}`
- `{output_dir / "commot" / "commot_crc_expected_edges.csv"}`
- `{output_dir / "hypersca_ot" / "lr_flow_edges.csv"}`
- `{output_dir / "comparison" / "paired_lr_edge_comparison.csv"}`
- `{output_dir / "comparison" / "method_summary.csv"}`

## 解释口径

本结果是空间通信与机制假设的计算对照，不是因果证明或治疗结论。COMMOT 分数反映基于表达与空间代价的 collective optimal transport 通信强度；HyperSCA OT-flow 分数反映 LR 先验、表达支持、空间接触和几何距离约束下的轻量 sidecar 对照。
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _relative_artifacts(output_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())


def _markdown_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
