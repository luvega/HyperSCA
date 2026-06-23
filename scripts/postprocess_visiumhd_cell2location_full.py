#!/usr/bin/env python
"""Postprocess full VisiumHD cell2location benchmark output."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.discovery.target_discovery.unified_annotation import ABUNDANCE_METADATA_COLUMNS
from src.utils.plot_style import apply_cns_style, save_figure


CELLTYPE_ORDER = [
    "Tumor",
    "Intestinal_Epithelial",
    "T_cells",
    "B_cells",
    "Myeloid",
    "ILC",
    "Endothelial",
    "Fibroblast",
    "Smooth_Muscle",
    "Neuronal",
    "Unknown",
]
CELLTYPE_COLORS = {
    "Tumor": "#D55E00",
    "Intestinal_Epithelial": "#CC79A7",
    "T_cells": "#0072B2",
    "B_cells": "#56B4E9",
    "Myeloid": "#009E73",
    "ILC": "#F0E442",
    "Endothelial": "#999999",
    "Fibroblast": "#E69F00",
    "Smooth_Muscle": "#8A6BBE",
    "Neuronal": "#4D4D4D",
    "Unknown": "#BDBDBD",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _celltype_columns(table: pd.DataFrame) -> list[str]:
    cols = [
        col
        for col in table.columns
        if col not in ABUNDANCE_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(table[col])
    ]
    ordered = [col for col in CELLTYPE_ORDER if col in cols]
    ordered.extend(col for col in cols if col not in ordered)
    if not ordered:
        raise ValueError("abundance table has no numeric cell-type columns")
    return ordered


def _load_abundance(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing abundance file: {path}")
    table = pd.read_csv(path)
    missing = [col for col in ABUNDANCE_METADATA_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"{path} missing abundance metadata columns: {missing}")
    for col in ["x", "y"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


def _dominant_labels(table: pd.DataFrame, celltypes: list[str]) -> pd.Series:
    values = table[celltypes].fillna(0).clip(lower=0)
    return values.idxmax(axis=1)


def _row_fractions(table: pd.DataFrame, celltypes: list[str]) -> pd.DataFrame:
    values = table[celltypes].fillna(0).clip(lower=0).astype(float)
    totals = values.sum(axis=1).replace(0, np.nan)
    return values.div(totals, axis=0).fillna(0.0)


def _composition(table: pd.DataFrame, celltypes: list[str]) -> pd.Series:
    totals = table[celltypes].fillna(0).clip(lower=0).sum(axis=0)
    denom = float(totals.sum())
    if denom <= 0:
        return totals * 0.0
    return totals / denom


def _summarize_abundance(table: pd.DataFrame, celltypes: list[str]) -> dict:
    values = table[celltypes].fillna(0).astype(float)
    totals = values.clip(lower=0).sum(axis=1)
    dominant = values.idxmax(axis=1)
    return {
        "n_rows": int(len(table)),
        "n_celltype_columns": int(len(celltypes)),
        "celltype_columns": celltypes,
        "missing_x": int(table["x"].isna().sum()),
        "missing_y": int(table["y"].isna().sum()),
        "negative_values": int((values < 0).sum().sum()),
        "zero_total_rows": int((totals <= 0).sum()),
        "total_abundance_mean": float(totals.mean()),
        "total_abundance_median": float(totals.median()),
        "total_abundance_p05": float(totals.quantile(0.05)),
        "total_abundance_p95": float(totals.quantile(0.95)),
        "dominant_counts": {str(k): int(v) for k, v in dominant.value_counts().sort_index().items()},
        "composition": {str(k): float(v) for k, v in _composition(table, celltypes).items()},
    }


def _grid_aggregate(table: pd.DataFrame, celltypes: list[str], *, x0: float, y0: float, grid_size: float) -> pd.DataFrame:
    frac = _row_fractions(table, celltypes)
    work = pd.DataFrame(
        {
            "gx": np.floor((table["x"].astype(float).to_numpy() - x0) / grid_size).astype(int),
            "gy": np.floor((table["y"].astype(float).to_numpy() - y0) / grid_size).astype(int),
        }
    )
    for col in celltypes:
        work[col] = frac[col].to_numpy()
    out = work.groupby(["gx", "gy"], observed=True)[celltypes].mean().reset_index()
    counts = work.groupby(["gx", "gy"], observed=True).size().reset_index(name="n_obs")
    return out.merge(counts, on=["gx", "gy"], how="left")


def _spatial_grid_metrics(
    cell2location: pd.DataFrame,
    rctd: pd.DataFrame,
    celltypes: list[str],
    *,
    grid_size: float,
) -> tuple[pd.DataFrame, dict]:
    x0 = float(min(cell2location["x"].min(), rctd["x"].min()))
    y0 = float(min(cell2location["y"].min(), rctd["y"].min()))
    c2l_grid = _grid_aggregate(cell2location, celltypes, x0=x0, y0=y0, grid_size=grid_size)
    rctd_grid = _grid_aggregate(rctd, celltypes, x0=x0, y0=y0, grid_size=grid_size)
    merged = c2l_grid.merge(rctd_grid, on=["gx", "gy"], how="inner", suffixes=("_cell2location", "_rctd"))
    rows = []
    for col in celltypes:
        left = merged[f"{col}_cell2location"]
        right = merged[f"{col}_rctd"]
        corr = left.corr(right, method="spearman") if len(merged) >= 3 else np.nan
        rows.append(
            {
                "cell_type": col,
                "spearman_grid_fraction": float(corr) if pd.notna(corr) else np.nan,
                "cell2location_mean_fraction": float(left.mean()) if len(merged) else np.nan,
                "rctd_mean_fraction": float(right.mean()) if len(merged) else np.nan,
            }
        )
    if len(merged):
        c2l_dom = merged[[f"{col}_cell2location" for col in celltypes]].idxmax(axis=1).str.replace("_cell2location", "")
        rctd_dom = merged[[f"{col}_rctd" for col in celltypes]].idxmax(axis=1).str.replace("_rctd", "")
        dominant_concordance = float((c2l_dom == rctd_dom).mean())
    else:
        dominant_concordance = np.nan
    summary = {
        "grid_size": float(grid_size),
        "cell2location_grid_count": int(len(c2l_grid)),
        "rctd_grid_count": int(len(rctd_grid)),
        "shared_grid_count": int(len(merged)),
        "dominant_grid_concordance": float(dominant_concordance) if pd.notna(dominant_concordance) else np.nan,
    }
    return pd.DataFrame(rows), summary


def _sample_for_plot(table: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(table) <= max_points:
        return table
    return table.sample(n=max_points, random_state=seed)


def _plot_dominant_side_by_side(
    cell2location: pd.DataFrame,
    rctd: pd.DataFrame,
    celltypes: list[str],
    output_dir: Path,
    *,
    max_points: int,
    seed: int,
) -> Path:
    apply_cns_style()
    c2l_plot = _sample_for_plot(cell2location.copy(), max_points, seed)
    rctd_plot = _sample_for_plot(rctd.copy(), max_points, seed + 1)
    c2l_plot["dominant"] = _dominant_labels(c2l_plot, celltypes)
    rctd_plot["dominant"] = _dominant_labels(rctd_plot, celltypes)
    labels = [label for label in CELLTYPE_ORDER if label in set(c2l_plot["dominant"]) | set(rctd_plot["dominant"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharex=True, sharey=True)
    for ax, table, title in [
        (axes[0], c2l_plot, "cell2location full VisiumHD"),
        (axes[1], rctd_plot, "RCTD segmented VisiumHD"),
    ]:
        for label in labels:
            subset = table[table["dominant"].eq(label)]
            if subset.empty:
                continue
            ax.scatter(
                subset["x"],
                subset["y"],
                s=0.25,
                c=CELLTYPE_COLORS.get(label, "#BDBDBD"),
                label=label,
                linewidths=0,
                alpha=0.8,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    handles, labels_out = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=8)
    return save_figure(
        fig,
        output_dir / "visiumhd_cell2location_vs_rctd_dominant_spatial.png",
        dpi=300,
        config={"max_points_per_method": max_points},
    )


def _plot_composition(
    c2l_comp: pd.Series,
    rctd_comp: pd.Series,
    output_dir: Path,
) -> Path:
    apply_cns_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    celltypes = [col for col in CELLTYPE_ORDER if col in c2l_comp.index or col in rctd_comp.index]
    x = np.arange(len(celltypes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.set_layout_engine(None)
    ax.bar(x - width / 2, [c2l_comp.get(col, 0.0) for col in celltypes], width, label="cell2location", color="#0072B2")
    ax.bar(x + width / 2, [rctd_comp.get(col, 0.0) for col in celltypes], width, label="RCTD", color="#D55E00")
    ax.set_ylabel("Fraction of total abundance")
    ax.set_xticks(x)
    ax.set_xticklabels(celltypes, rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.34, top=0.96)
    return save_figure(fig, output_dir / "visiumhd_celltype_composition_cell2location_vs_rctd.png", dpi=300)


def _plot_grid_correlations(grid_metrics: pd.DataFrame, output_dir: Path) -> Path:
    apply_cns_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    data = grid_metrics.dropna(subset=["spearman_grid_fraction"]).copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.set_layout_engine(None)
    ax.bar(
        data["cell_type"],
        data["spearman_grid_fraction"],
        color=[CELLTYPE_COLORS.get(label, "#777777") for label in data["cell_type"]],
    )
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Spearman rho across spatial grid", fontsize=9)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["cell_type"], rotation=45, ha="right", fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.96)
    return save_figure(fig, output_dir / "visiumhd_grid_abundance_spearman_cell2location_vs_rctd.png", dpi=300)


def _plot_celltype_spatial_panels(
    cell2location: pd.DataFrame,
    rctd: pd.DataFrame,
    celltypes: list[str],
    output_dir: Path,
    *,
    max_points: int,
    seed: int,
) -> Path:
    apply_cns_style()
    selected = [ct for ct in ["Tumor", "T_cells", "Myeloid", "Fibroblast"] if ct in celltypes]
    if not selected:
        selected = celltypes[: min(4, len(celltypes))]
    c2l_plot = _sample_for_plot(cell2location.copy(), max_points, seed)
    rctd_plot = _sample_for_plot(rctd.copy(), max_points, seed + 1)
    c2l_frac = _row_fractions(c2l_plot, celltypes)
    rctd_frac = _row_fractions(rctd_plot, celltypes)
    fig, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6.0), sharex=True, sharey=True)
    if len(selected) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col_idx, ct in enumerate(selected):
        for row_idx, (table, frac, method) in enumerate(
            [(c2l_plot, c2l_frac, "cell2location"), (rctd_plot, rctd_frac, "RCTD")]
        ):
            ax = axes[row_idx, col_idx]
            sc = ax.scatter(
                table["x"],
                table["y"],
                c=frac[ct],
                s=0.25,
                cmap="viridis",
                vmin=0,
                vmax=max(float(frac[ct].quantile(0.99)), 1e-6),
                linewidths=0,
                rasterized=True,
            )
            ax.set_title(f"{method}: {ct}")
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    return save_figure(fig, output_dir / "visiumhd_selected_celltype_fraction_spatial_panels.png", dpi=300)


def _append_progress(progress_path: Path, line: str) -> None:
    if not progress_path:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {datetime.now(timezone.utc).isoformat(timespec='seconds')} {line}\n")


def postprocess(args: argparse.Namespace) -> dict:
    c2l_dir = args.cell2location_dir
    output_dir = args.output_dir or c2l_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "qc"
    c2l_abundance_path = args.cell2location_abundance or (
        c2l_dir / "abundance" / "osta_visiumhd_humancolon_oliveira_cell2location_scCRC_ICB_abundance.csv.gz"
    )
    rctd_abundance_path = args.rctd_abundance or (
        args.rctd_dir
        / "abundance"
        / "osta_visiumhd_segmented_humancolon_oliveira_rctd_scCRC_ICB_abundance.csv.gz"
    )
    manifest_path = c2l_dir / "annotation_manifest.json"

    c2l = _load_abundance(c2l_abundance_path)
    rctd = _load_abundance(rctd_abundance_path)
    c2l_celltypes = _celltype_columns(c2l)
    rctd_celltypes = _celltype_columns(rctd)
    common_celltypes = [ct for ct in CELLTYPE_ORDER if ct in c2l_celltypes and ct in rctd_celltypes]
    common_celltypes.extend(ct for ct in c2l_celltypes if ct in rctd_celltypes and ct not in common_celltypes)
    if not common_celltypes:
        raise ValueError("cell2location and RCTD abundance tables have no shared cell-type columns")

    row_count_ok = int(len(c2l)) == int(args.expected_rows)
    c2l_summary = _summarize_abundance(c2l, c2l_celltypes)
    rctd_summary = _summarize_abundance(rctd, rctd_celltypes)
    grid_metrics, grid_summary = _spatial_grid_metrics(c2l, rctd, common_celltypes, grid_size=args.grid_size)

    tables_dir.mkdir(parents=True, exist_ok=True)
    grid_metrics_path = tables_dir / "visiumhd_cell2location_vs_rctd_grid_metrics.csv"
    grid_metrics.to_csv(grid_metrics_path, index=False)
    composition = pd.DataFrame(
        {
            "cell_type": common_celltypes,
            "cell2location_fraction": [_composition(c2l, common_celltypes).get(ct, 0.0) for ct in common_celltypes],
            "rctd_fraction": [_composition(rctd, common_celltypes).get(ct, 0.0) for ct in common_celltypes],
        }
    )
    composition_path = tables_dir / "visiumhd_cell2location_vs_rctd_composition.csv"
    composition.to_csv(composition_path, index=False)

    figures = [
        _plot_dominant_side_by_side(c2l, rctd, common_celltypes, figures_dir, max_points=args.max_plot_points, seed=args.seed),
        _plot_composition(
            pd.Series(dict(zip(composition["cell_type"], composition["cell2location_fraction"]))),
            pd.Series(dict(zip(composition["cell_type"], composition["rctd_fraction"]))),
            figures_dir,
        ),
        _plot_grid_correlations(grid_metrics, figures_dir),
        _plot_celltype_spatial_panels(c2l, rctd, common_celltypes, figures_dir, max_points=args.max_plot_points, seed=args.seed),
    ]

    metrics = {
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cell2location_abundance_path": str(c2l_abundance_path),
        "rctd_abundance_path": str(rctd_abundance_path),
        "expected_cell2location_rows": int(args.expected_rows),
        "cell2location_row_count_ok": bool(row_count_ok),
        "cell2location": c2l_summary,
        "rctd": rctd_summary,
        "shared_celltype_columns": common_celltypes,
        "grid_comparison": grid_summary,
        "grid_metrics_path": str(grid_metrics_path),
        "composition_path": str(composition_path),
        "figures": [str(path) for path in figures],
    }
    metrics_path = tables_dir / "visiumhd_cell2location_full_postprocess_metrics.json"
    metrics["metrics_path"] = str(metrics_path)
    _write_json(metrics_path, metrics)

    manifest = _read_json(manifest_path)
    manifest["postprocess"] = {
        "status": "ok" if row_count_ok else "failed:row_count_mismatch",
        "validated_at": metrics["validated_at"],
        "expected_rows": int(args.expected_rows),
        "observed_rows": int(len(c2l)),
        "metrics_path": str(metrics_path),
        "grid_metrics_path": str(grid_metrics_path),
        "composition_path": str(composition_path),
        "figures": [str(path) for path in figures],
    }
    _write_json(manifest_path, manifest)

    summary_path = output_dir / "visiumhd_cell2location_full_summary.md"
    summary = [
        "# VisiumHD Full cell2location Benchmark Summary",
        "",
        f"- Validated at: `{metrics['validated_at']}`",
        f"- cell2location abundance rows: `{len(c2l):,}` / expected `{int(args.expected_rows):,}`",
        f"- RCTD segmented abundance rows: `{len(rctd):,}`",
        f"- Shared cell-type columns: `{', '.join(common_celltypes)}`",
        f"- Shared spatial grid cells: `{grid_summary['shared_grid_count']:,}` at grid size `{args.grid_size}`",
        f"- Dominant grid concordance: `{grid_summary['dominant_grid_concordance']:.4f}`",
        "",
        "## Outputs",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Grid metrics: `{grid_metrics_path}`",
        f"- Composition: `{composition_path}`",
    ]
    summary.extend(f"- Figure: `{path}`" for path in figures)
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    _append_progress(
        args.progress_path,
        (
            "postprocess complete "
            f"cell2location_rows={len(c2l)} expected_rows={int(args.expected_rows)} "
            f"row_count_ok={row_count_ok} shared_grid_count={grid_summary['shared_grid_count']}"
        ),
    )
    if not row_count_ok:
        raise ValueError(f"cell2location abundance row count {len(c2l)} != expected {int(args.expected_rows)}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell2location-dir",
        type=Path,
        default=ROOT / "results/benchmarks/unified_spatial_annotation_cell2location_visiumhd_full",
    )
    parser.add_argument(
        "--rctd-dir",
        type=Path,
        default=ROOT / "results/benchmarks/unified_spatial_annotation_segmented_rctd_full_sccrc_singlecore_vectorized",
    )
    parser.add_argument("--cell2location-abundance", type=Path, default=None)
    parser.add_argument("--rctd-abundance", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=545_913)
    parser.add_argument("--grid-size", type=float, default=512.0)
    parser.add_argument("--max-plot-points", type=int, default=220_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=ROOT / "results/benchmarks/visiumhd_cell2location_full_progress.md",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = postprocess(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "cell2location_rows": metrics["cell2location"]["n_rows"],
                "expected_rows": metrics["expected_cell2location_rows"],
                "row_count_ok": metrics["cell2location_row_count_ok"],
                "metrics_path": metrics["metrics_path"],
                "figures": metrics["figures"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
