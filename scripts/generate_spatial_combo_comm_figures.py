#!/usr/bin/env python
"""Generate multi-dataset spatial target + communication figures.

Datasets:
1) ST_CRC_MSS (standardized h5ad)
2) VisiumHD_HumanColon_Oliveira (16um binned output)
3) scCRC_IFNG CosMx (all_samples.h5ad)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_style import CMAP_SPATIAL, apply_cns_style

JOURNAL_SINGLE_COL_WIDTH = 3.35
JOURNAL_DOUBLE_COL_WIDTH = 7.0
SPATIAL_CONT_CMAP = CMAP_SPATIAL


def _safe_load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _discover_combo_genes(
    combos_csv: Path,
    step4_combo_csv: Path,
    max_genes: int = 3,
) -> list[str]:
    genes: list[str] = []

    if step4_combo_csv.exists():
        step4_df = pd.read_csv(step4_combo_csv)
        pair_rows = step4_df[step4_df["size"] >= 2]
        if not pair_rows.empty:
            pair = str(pair_rows.iloc[0]["combo"])
            genes.extend([g.strip().upper() for g in pair.split("+") if g.strip()])

    if combos_csv.exists():
        combo_df = pd.read_csv(combos_csv)
        if "trigger_target" in combo_df.columns:
            genes.extend(combo_df["trigger_target"].astype(str).str.upper().tolist())
        if "ligand" in combo_df.columns:
            genes.extend(combo_df["ligand"].astype(str).str.upper().tolist())

    # Keep order and uniqueness
    unique = []
    for g in genes:
        if g and g not in unique:
            unique.append(g)
    return unique[:max_genes]


def _subsample_idx(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _extract_gene_matrix(
    adata,
    genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    var_upper = {str(g).upper(): str(g) for g in adata.var_names}
    found = [var_upper[g] for g in genes if g in var_upper]
    if not found:
        return np.zeros((adata.n_obs, 0), dtype=np.float32), []

    x = adata[:, found].X
    if sparse.issparse(x):
        mat = x.toarray()
    else:
        mat = np.asarray(x)
    return mat, found


def _dataset_from_h5ad(
    name: str,
    h5ad_path: Path,
    genes: list[str],
    max_points: int,
    seed: int,
) -> dict:
    a = ad.read_h5ad(h5ad_path, backed="r")
    if "spatial" in a.obsm:
        coords = np.asarray(a.obsm["spatial"])
    elif {"x", "y"}.issubset(set(a.obs.columns)):
        coords = a.obs[["x", "y"]].to_numpy()
    else:
        raise ValueError(f"{name}: no spatial coordinates found.")

    mat, found_genes = _extract_gene_matrix(a, genes)
    if mat.shape[1] == 0:
        score = np.zeros(a.n_obs, dtype=np.float32)
    else:
        score = mat.mean(axis=1).astype(np.float32)

    idx = _subsample_idx(len(score), max_points=max_points, seed=seed)
    smat = mat[idx] if mat.shape[1] > 0 else np.zeros((len(idx), 0), dtype=np.float32)
    summary = []
    for j, g in enumerate(found_genes):
        gv = smat[:, j]
        summary.append(
            {
                "dataset": name,
                "gene": g.upper(),
                "mean_expr": float(np.mean(gv)),
                "pct_expr": float(np.mean(gv > 0)),
            }
        )

    return {
        "name": name,
        "coords": coords[idx],
        "score": score[idx],
        "found_genes": [g.upper() for g in found_genes],
        "summary": summary,
        "n_total": int(a.n_obs),
        "n_plot": int(len(idx)),
    }


def _dataset_from_visiumhd(
    name: str,
    h5_path: Path,
    tissue_pos_parquet: Path,
    genes: list[str],
    max_points: int,
    seed: int,
) -> dict:
    adata = sc.read_10x_h5(str(h5_path))
    # 10x h5 可能存在重复基因名，需先唯一化再做列切片
    adata.var_names_make_unique()
    pos = pd.read_parquet(tissue_pos_parquet)
    pos = pos[pos["in_tissue"] == 1].copy()
    pos = pos.set_index("barcode")

    coords_df = pos.reindex(adata.obs_names)[["pxl_col_in_fullres", "pxl_row_in_fullres"]]
    coords = coords_df.to_numpy()
    valid = np.isfinite(coords).all(axis=1)

    adata = adata[valid].copy()
    coords = coords[valid]

    mat, found_genes = _extract_gene_matrix(adata, genes)
    if mat.shape[1] == 0:
        score = np.zeros(adata.n_obs, dtype=np.float32)
    else:
        score = mat.mean(axis=1).astype(np.float32)

    idx = _subsample_idx(len(score), max_points=max_points, seed=seed)
    smat = mat[idx] if mat.shape[1] > 0 else np.zeros((len(idx), 0), dtype=np.float32)
    summary = []
    for j, g in enumerate(found_genes):
        gv = smat[:, j]
        summary.append(
            {
                "dataset": name,
                "gene": g.upper(),
                "mean_expr": float(np.mean(gv)),
                "pct_expr": float(np.mean(gv > 0)),
            }
        )

    return {
        "name": name,
        "coords": coords[idx],
        "score": score[idx],
        "found_genes": [g.upper() for g in found_genes],
        "summary": summary,
        "n_total": int(adata.n_obs),
        "n_plot": int(len(idx)),
    }


def _load_combo_edges(flow_edges_json: Path, combo_genes: list[str], top_k: int = 18) -> pd.DataFrame:
    edges = _safe_load_json(flow_edges_json)
    if not edges:
        return pd.DataFrame(columns=["source", "target", "weight", "pathway", "causal_edge"])

    df = pd.DataFrame(edges)
    for c in ["source_layer", "target_layer", "source", "target", "weight", "pathway", "causal_edge"]:
        if c not in df.columns:
            df[c] = np.nan

    # Focus on ligand->receptor communication for current combo ligands
    sub = df[(df["source_layer"] == 0) & (df["target_layer"] == 1)].copy()
    sub["source_u"] = sub["source"].astype(str).str.upper()
    sub = sub[sub["source_u"].isin(combo_genes)]
    if sub.empty:
        sub = df[(df["source_layer"] == 0) & (df["target_layer"] == 1)].copy()
    sub = sub.sort_values("weight", ascending=False).head(top_k)
    return sub


def _style_spatial_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(labelsize=8, width=0.6, length=2)


def _plot_communication_panel(ax: plt.Axes, edge_df: pd.DataFrame) -> None:
    ax.grid(False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    if edge_df.empty:
        ax.text(0.5, 0.5, "No communication edges available", ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ligands = list(dict.fromkeys(edge_df["source"].astype(str).tolist()))
    receptors = list(dict.fromkeys(edge_df["target"].astype(str).tolist()))
    pathways = list(dict.fromkeys(edge_df["pathway"].astype(str).tolist()))

    y_l = np.linspace(0.1, 0.9, max(len(ligands), 1))
    y_r = np.linspace(0.1, 0.9, max(len(receptors), 1))
    lig_pos = {g: y_l[i] for i, g in enumerate(ligands)}
    rec_pos = {g: y_r[i] for i, g in enumerate(receptors)}

    # Low-saturation categorical palette
    pathway_palette = [
        "#7AA6C2", "#B6A2C8", "#8FBF9F", "#D2B48C", "#A9A9A9", "#C29F80", "#9FB4D1"
    ]
    pmap = {p: pathway_palette[i % len(pathway_palette)] for i, p in enumerate(pathways)}

    wmax = max(float(edge_df["weight"].max()), 1e-8)
    for _, row in edge_df.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        pw = str(row["pathway"])
        w = float(row["weight"])
        ax.plot(
            [0.15, 0.85],
            [lig_pos[s], rec_pos[t]],
            color=pmap[pw],
            linewidth=0.5 + 0.5 * (w / wmax),
            alpha=0.65,
            zorder=1,
        )

    ax.scatter([0.15] * len(ligands), [lig_pos[g] for g in ligands], s=28, c="#D95F5F", edgecolors="none", zorder=3)
    ax.scatter([0.85] * len(receptors), [rec_pos[g] for g in receptors], s=28, c="#5F8DD9", edgecolors="none", zorder=3)

    texts = []
    for g in ligands:
        texts.append(ax.text(0.11, lig_pos[g], g, ha="right", va="center", fontsize=8))
    for g in receptors:
        texts.append(ax.text(0.89, rec_pos[g], g, ha="left", va="center", fontsize=8))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666", alpha=0.6))

    handles = [
        plt.Line2D([0], [0], color=pmap[p], lw=1.0, label=p) for p in pathways
    ]
    ax.legend(
        handles=handles,
        title="Pathway",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        markerscale=1.8,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Ligand -> Receptor", fontsize=10)


def _plot_dot_panel(ax: plt.Axes, summary_df: pd.DataFrame, combo_genes: list[str], fig: plt.Figure) -> None:
    if summary_df.empty:
        ax.text(0.5, 0.5, "No expression summary", ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x_levels = ["ST_CRC_MSS", "VisiumHD_HumanColon_Oliveira", "CosMx_scCRC_IFNG"]
    y_levels = combo_genes
    x_map = {k: i for i, k in enumerate(x_levels)}
    y_map = {k: i for i, k in enumerate(y_levels)}

    plot_df = summary_df.copy()
    plot_df["x"] = plot_df["dataset"].map(x_map)
    plot_df["y"] = plot_df["gene"].str.upper().map(y_map)
    plot_df = plot_df.dropna(subset=["x", "y"])

    sizes = np.maximum(plot_df["pct_expr"].values * 420.0, 8.0)  # area proportional to pct_expr
    dots = ax.scatter(
        plot_df["x"].values,
        plot_df["y"].values,
        s=sizes,
        c=plot_df["mean_expr"].values,
        cmap=SPATIAL_CONT_CMAP,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xticks(list(range(len(x_levels))))
    ax.set_xticklabels(x_levels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(list(range(len(y_levels))))
    ax.set_yticklabels(y_levels, fontsize=8)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(False)
    cb = fig.colorbar(dots, ax=ax, fraction=0.08, pad=0.02)
    cb.ax.tick_params(labelsize=8, width=0.6, length=2)
    cb.set_label("Mean expression", fontsize=10)


def _save_single_spatial_panel(
    ds: dict,
    combo_genes: list[str],
    out_path: Path,
    vmin: float,
    vmax: float,
    panel_label: str,
) -> None:
    def _draw(fig_w: float, fig_h: float, save_path: Path) -> None:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        sca = ax.scatter(
            ds["coords"][:, 0],
            ds["coords"][:, 1],
            c=ds["score_plot"],
            cmap=SPATIAL_CONT_CMAP,
            s=1.0,
            alpha=0.62,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        _style_spatial_ax(ax)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_title(
            f"{panel_label}  {ds['name']} (n={ds['n_plot']:,}/{ds['n_total']:,})",
            fontsize=13,
            fontweight="bold",
            pad=7,
        )
        cb = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=8, width=0.6, length=2)
        label = combo_genes[0] if len(combo_genes) == 1 else " + ".join(combo_genes)
        cb.set_label(f"Target score (normalized): {label}", fontsize=10)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    _draw(6.2, 5.4, out_path)
    stem = out_path.stem
    _draw(
        JOURNAL_SINGLE_COL_WIDTH,
        JOURNAL_SINGLE_COL_WIDTH * 0.88,
        out_path.with_name(f"{stem}_singlecol.png"),
    )
    _draw(
        JOURNAL_DOUBLE_COL_WIDTH,
        JOURNAL_DOUBLE_COL_WIDTH * 0.86,
        out_path.with_name(f"{stem}_doublecol.png"),
    )


def _save_single_comm_panel(edge_df: pd.DataFrame, out_path: Path) -> None:
    def _draw(fig_w: float, fig_h: float, save_path: Path) -> None:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        _plot_communication_panel(ax, edge_df)
        ax.set_title("D  Cell communication constrained by selected targets", fontsize=13, fontweight="bold", pad=6)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    _draw(8.8, 5.8, out_path)
    stem = out_path.stem
    _draw(
        JOURNAL_SINGLE_COL_WIDTH,
        JOURNAL_SINGLE_COL_WIDTH * 0.85,
        out_path.with_name(f"{stem}_singlecol.png"),
    )
    _draw(
        JOURNAL_DOUBLE_COL_WIDTH,
        JOURNAL_DOUBLE_COL_WIDTH * 0.80,
        out_path.with_name(f"{stem}_doublecol.png"),
    )


def _save_single_dot_panel(summary_df: pd.DataFrame, combo_genes: list[str], out_path: Path) -> None:
    def _draw(fig_w: float, fig_h: float, save_path: Path) -> None:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        _plot_dot_panel(ax, summary_df, combo_genes, fig)
        ax.set_title("E  Anchor target expression dot plot", fontsize=13, fontweight="bold", pad=6)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    _draw(6.8, 5.2, out_path)
    stem = out_path.stem
    _draw(
        JOURNAL_SINGLE_COL_WIDTH,
        JOURNAL_SINGLE_COL_WIDTH * 0.78,
        out_path.with_name(f"{stem}_singlecol.png"),
    )
    _draw(
        JOURNAL_DOUBLE_COL_WIDTH,
        JOURNAL_DOUBLE_COL_WIDTH * 0.76,
        out_path.with_name(f"{stem}_doublecol.png"),
    )


def _normalize_scores_for_plot(datasets: list[dict]) -> tuple[float, float]:
    """稳健归一化，避免全图接近黑色。"""
    all_scores = np.concatenate([np.asarray(d["score"], dtype=float) for d in datasets])
    all_scores = np.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)
    # log1p 增强弱信号可见性
    all_scores = np.log1p(np.clip(all_scores, a_min=0.0, a_max=None))
    lo = float(np.percentile(all_scores, 2))
    hi = float(np.percentile(all_scores, 98))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    for d in datasets:
        raw = np.asarray(d["score"], dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.log1p(np.clip(raw, a_min=0.0, a_max=None))
        normed = (raw - lo) / (hi - lo)
        d["score_plot"] = np.clip(normed, 0.0, 1.0)

    return 0.0, 1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate multi-dataset spatial target and communication figure."
    )
    parser.add_argument("--combos-csv", default="results/integration/discovery/spatiotemporal_regulatory_combos.csv")
    parser.add_argument("--step4-combo-csv", default="results/step4/combination_ranking.csv")
    parser.add_argument("--flow-edges-json", default="results/integration/discovery/hyperbolic/step2/flow_edges.json")

    parser.add_argument("--st-h5ad", default="data/ST/ST_CRC_MSS/expression.h5ad")
    parser.add_argument(
        "--visiumhd-h5",
        default="data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/filtered_feature_bc_matrix.h5",
    )
    parser.add_argument(
        "--visiumhd-tissue-pos",
        default="data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/spatial/tissue_positions.parquet",
    )
    parser.add_argument("--ifng-cosmx-h5ad", default="F:/scCRC_IFNG/data/processed/all_samples.h5ad")

    parser.add_argument("--max-points-per-dataset", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/figures/spatial_comm")
    args = parser.parse_args()

    apply_cns_style()

    combos_csv = PROJECT_ROOT / args.combos_csv
    step4_combo_csv = PROJECT_ROOT / args.step4_combo_csv
    flow_json = PROJECT_ROOT / args.flow_edges_json
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_genes = _discover_combo_genes(combos_csv, step4_combo_csv)
    if not target_genes:
        print("[error] no data-driven targets found in combo inputs")
        return 1
    print(f"[info] data-driven target genes: {target_genes}")

    all_summary_rows: list[dict] = []
    datasets_for_layout = None
    first_vmin, first_vmax = 0.0, 1.0
    for gi, gene in enumerate(target_genes):
        ds_st = _dataset_from_h5ad(
            name="ST_CRC_MSS",
            h5ad_path=PROJECT_ROOT / args.st_h5ad,
            genes=[gene],
            max_points=args.max_points_per_dataset,
            seed=args.seed + gi * 10,
        )
        ds_visiumhd = _dataset_from_visiumhd(
            name="VisiumHD_HumanColon_Oliveira",
            h5_path=PROJECT_ROOT / args.visiumhd_h5,
            tissue_pos_parquet=PROJECT_ROOT / args.visiumhd_tissue_pos,
            genes=[gene],
            max_points=args.max_points_per_dataset,
            seed=args.seed + 1 + gi * 10,
        )
        ds_cosmx = _dataset_from_h5ad(
            name="CosMx_scCRC_IFNG",
            h5ad_path=Path(args.ifng_cosmx_h5ad),
            genes=[gene],
            max_points=args.max_points_per_dataset,
            seed=args.seed + 2 + gi * 10,
        )
        datasets = [ds_st, ds_visiumhd, ds_cosmx]
        vmin, vmax = _normalize_scores_for_plot(datasets)
        if datasets_for_layout is None:
            datasets_for_layout = datasets
            first_vmin, first_vmax = vmin, vmax

        _save_single_spatial_panel(
            ds_st, [gene],
            out_dir / f"panel_A_spatial_ST_CRC_MSS_{gene}.png",
            vmin=vmin, vmax=vmax, panel_label="A",
        )
        _save_single_spatial_panel(
            ds_visiumhd, [gene],
            out_dir / f"panel_B_spatial_VisiumHD_HumanColon_Oliveira_{gene}.png",
            vmin=vmin, vmax=vmax, panel_label="B",
        )
        _save_single_spatial_panel(
            ds_cosmx, [gene],
            out_dir / f"panel_C_spatial_CosMx_scCRC_IFNG_{gene}.png",
            vmin=vmin, vmax=vmax, panel_label="C",
        )
        all_summary_rows.extend(ds_st["summary"] + ds_visiumhd["summary"] + ds_cosmx["summary"])

    edge_df = _load_combo_edges(flow_json, combo_genes=target_genes, top_k=18)
    summary_df = pd.DataFrame(all_summary_rows)

    _save_single_comm_panel(
        edge_df,
        out_dir / "panel_D_communication_target_edges.png",
    )
    _save_single_dot_panel(
        summary_df,
        target_genes,
        out_dir / "panel_E_dotplot_target_expression.png",
    )

    # ===== Figure layout (GridSpec strict alignment) =====
    if datasets_for_layout is None:
        print("[error] no datasets loaded for layout figure")
        return 1
    datasets = datasets_for_layout
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(
        nrows=2, ncols=3, figure=fig,
        height_ratios=[1.0, 0.82],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.16, wspace=0.18,
    )

    spatial_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_comm = fig.add_subplot(gs[1, 0:2])
    ax_dot = fig.add_subplot(gs[1, 2])

    labels = ["A", "B", "C"]
    for i, (ax, ds) in enumerate(zip(spatial_axes, datasets)):
        sca = ax.scatter(
            ds["coords"][:, 0],
            ds["coords"][:, 1],
            c=ds["score_plot"],
            cmap=SPATIAL_CONT_CMAP,
            s=1.0,
            alpha=0.62,
            edgecolors="none",
            vmin=first_vmin,
            vmax=first_vmax,
            rasterized=True,
        )
        _style_spatial_ax(ax)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_title(
            f"{labels[i]}  {ds['name']} (n={ds['n_plot']:,}/{ds['n_total']:,})",
            fontsize=13,
            fontweight="bold",
            pad=7,
        )

    cbar = fig.colorbar(sca, ax=spatial_axes, fraction=0.022, pad=0.01)
    cbar.ax.tick_params(labelsize=8, width=0.6, length=2)
    cbar.set_label(f"Target score (normalized): {target_genes[0]}", fontsize=10)

    _plot_communication_panel(ax_comm, edge_df)
    ax_comm.set_title("D  Cell communication constrained by selected targets", fontsize=13, fontweight="bold", pad=6)

    _plot_dot_panel(ax_dot, summary_df, target_genes, fig)
    ax_dot.set_title("E  Selected target expression dot plot", fontsize=13, fontweight="bold", pad=6)

    fig.suptitle(
        "Spatial Distribution and Communication of Selected Targets",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig_path = out_dir / "cns_spatial_target_communication_multidataset.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = {
        "target_genes": target_genes,
        "datasets": {
            d["name"]: {
                "n_total": d["n_total"],
                "n_plot": d["n_plot"],
                "found_genes": d["found_genes"],
            }
            for d in datasets
        },
        "communication_edges_plotted": int(len(edge_df)),
        "figure_combined": str(fig_path),
        "figures_single_panels": [
            *[str(out_dir / f"panel_A_spatial_ST_CRC_MSS_{gene}.png") for gene in target_genes],
            *[str(out_dir / f"panel_B_spatial_VisiumHD_HumanColon_Oliveira_{gene}.png") for gene in target_genes],
            *[str(out_dir / f"panel_C_spatial_CosMx_scCRC_IFNG_{gene}.png") for gene in target_genes],
            str(out_dir / "panel_D_communication_target_edges.png"),
            str(out_dir / "panel_E_dotplot_target_expression.png"),
        ],
    }
    (out_dir / "cns_spatial_target_communication_multidataset.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] saved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
