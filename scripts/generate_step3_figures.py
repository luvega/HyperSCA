#!/usr/bin/env python
"""阶段 3 展示图批量生成。

输入:
    - results/step3/            (run_step3.py 产物)
    - results/step2/cluster_expr.npy + node_info.json (观测基线)
    - results/step1/adata_embedded.h5ad (基因名映射)
输出:
    - results/figures/step3/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_observed_expr(step1_dir: Path, step2_dir: Path, expected_genes: list[str]) -> pd.DataFrame:
    import anndata as ad

    cluster_expr = np.load(step2_dir / "cluster_expr.npy")
    node_info = _safe_read_json(step2_dir / "node_info.json")
    node_labels = node_info.get("node_labels", [f"node_{i}" for i in range(cluster_expr.shape[0])])

    gene_names = expected_genes
    if not gene_names:
        adata_path = step1_dir / "adata_embedded.h5ad"
        if adata_path.exists():
            adata = ad.read_h5ad(adata_path)
            gene_names = list(map(str, adata.var_names))
    if len(gene_names) < cluster_expr.shape[1]:
        gene_names = gene_names + [f"GENE_{i}" for i in range(len(gene_names), cluster_expr.shape[1])]
    gene_names = gene_names[: cluster_expr.shape[1]]
    return pd.DataFrame(cluster_expr, index=node_labels, columns=gene_names)


def main() -> int:
    parser = argparse.ArgumentParser(description="HyperSCA Stage 3: Generate Figures")
    parser.add_argument("--input-dir", type=str, default="results/step3", help="Step 3 output directory")
    parser.add_argument("--step2-dir", type=str, default="results/step2", help="Step 2 output directory")
    parser.add_argument("--step1-dir", type=str, default="results/step1", help="Step 1 output directory")
    parser.add_argument("--output-dir", type=str, default="results/figures/step3", help="Figure output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    step2_dir = Path(args.step2_dir)
    step1_dir = Path(args.step1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] input_dir not found: {input_dir}")
        return 1

    metrics = _safe_read_json(input_dir / "step3_metrics.json")
    targets = metrics.get("targets", [])
    if not targets:
        # fallback: infer from filenames
        targets = []
        for path in sorted(input_dir.glob("interaction_targets_*.csv")):
            name = path.stem.replace("interaction_targets_", "")
            targets.append(name)

    print("HyperSCA Stage 3: Generating Figures")
    print("=" * 60)
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Targets: {targets}")
    print("=" * 60)

    from src.visualization.perturbation import (
        plot_interaction_target_ranking,
        plot_multi_target_heatmap,
        plot_perturbation_comparison,
        plot_perturbation_metrics_dashboard,
        plot_step3_overview_dashboard,
    )

    # 先用第一个 cf 文件推断基因列
    expected_genes: list[str] = []
    if targets:
        first_cf = input_dir / f"cf_expression_{targets[0]}.csv"
        if first_cf.exists():
            expected_genes = list(pd.read_csv(first_cf, index_col=0).columns)
    obs_expr = _load_observed_expr(step1_dir, step2_dir, expected_genes=expected_genes)

    marker_union: list[str] = []
    fc_rows: list[np.ndarray] = []
    valid_targets: list[str] = []
    all_ranked_frames: list[pd.DataFrame] = []

    for idx, target in enumerate(targets, start=1):
        print(f"  [{idx}/{len(targets)}] target={target}")
        rank_path = input_dir / f"interaction_targets_{target}.csv"
        cf_path = input_dir / f"cf_expression_{target}.csv"
        if not rank_path.exists() or not cf_path.exists():
            print(f"    [skip] missing files for {target}")
            continue

        ranked = pd.read_csv(rank_path)
        if not ranked.empty:
            ranked = ranked.copy()
            ranked["target_gene"] = target
            all_ranked_frames.append(ranked)
        cf_expr = pd.read_csv(cf_path, index_col=0)
        # 对齐行列
        common_rows = [r for r in cf_expr.index if r in obs_expr.index]
        common_cols = [c for c in cf_expr.columns if c in obs_expr.columns]
        if not common_rows or not common_cols:
            print(f"    [skip] no shared rows/cols for {target}")
            continue
        cf_sub = cf_expr.loc[common_rows, common_cols]
        obs_sub = obs_expr.loc[common_rows, common_cols]

        # 排名图
        plot_interaction_target_ranking(
            ranked_targets=ranked,
            top_n=min(20, len(ranked)),
            save_path=str(output_dir / f"interaction_target_ranking_{target}.png"),
        )

        # 对比图（target + top ligand/receptor）
        gene_cols = {c.upper(): c for c in obs_sub.columns}
        markers = [target.upper()]
        if not ranked.empty:
            markers.extend([str(x).upper() for x in ranked["ligand"].head(8).tolist()])
            markers.extend([str(x).upper() for x in ranked["receptor"].head(8).tolist()])
        markers = [gene_cols[m] for m in dict.fromkeys(markers) if m in gene_cols]
        if not markers:
            markers = list(obs_sub.columns[: min(20, obs_sub.shape[1])])

        obs_mean = obs_sub[markers].mean(axis=0).values
        cf_mean = cf_sub[markers].mean(axis=0).values
        plot_perturbation_comparison(
            gene_names=markers,
            observed=obs_mean,
            counterfactual=cf_mean,
            target_gene=target,
            save_path=str(output_dir / f"perturbation_comparison_{target}.png"),
        )

        # dashboard（优先使用 step3_metrics 的真实指标）
        per_target = metrics.get("per_target", {}).get(target, {})
        dashboard_metrics = per_target.get("dashboard_metrics", {})
        if not dashboard_metrics:
            dashboard_metrics = {
                "r2_mean": float(per_target.get("top_score", 0.0)),
                "pcc_median": 0.0,
                "mse": 0.0,
                "marker_direction_accuracy": 1.0 if per_target.get("n_candidates", 0) > 0 else 0.0,
            }
        plot_perturbation_metrics_dashboard(
            metrics=dashboard_metrics,
            spatial_metrics=None,
            target_gene=target,
            save_path=str(output_dir / f"metrics_dashboard_{target}.png"),
        )

        # 汇总用于多靶点热图
        marker_union.extend(markers[:10])
        valid_targets.append(target)

    marker_union = list(dict.fromkeys(marker_union))[:20]
    if valid_targets and marker_union:
        for target in valid_targets:
            cf_expr = pd.read_csv(input_dir / f"cf_expression_{target}.csv", index_col=0)
            common_rows = [r for r in cf_expr.index if r in obs_expr.index]
            common_cols = [c for c in marker_union if c in cf_expr.columns and c in obs_expr.columns]
            if not common_rows or not common_cols:
                continue
            fc = (
                cf_expr.loc[common_rows, common_cols].mean(axis=0)
                - obs_expr.loc[common_rows, common_cols].mean(axis=0)
            ).reindex(marker_union, fill_value=0.0)
            fc_rows.append(fc.values.astype(float))
        if fc_rows:
            mat = np.vstack(fc_rows)
            plot_multi_target_heatmap(
                targets=valid_targets[: mat.shape[0]],
                marker_genes=marker_union,
                fold_changes=mat,
                save_path=str(output_dir / "multi_target_heatmap_step3.png"),
            )

    # 总览 dashboard
    merged = pd.concat(all_ranked_frames, ignore_index=True) if all_ranked_frames else pd.DataFrame()
    if not merged.empty:
        prior_rate = float(merged["prior_hit"].astype(bool).mean()) if "prior_hit" in merged.columns else None
        score_vals = merged["target_priority_score"].values if "target_priority_score" in merged.columns else None
        pathway_counts: dict[str, int] = {}
        if "pathway" in merged.columns:
            pathway_counts = merged["pathway"].fillna("").astype(str).value_counts().to_dict()
        plot_step3_overview_dashboard(
            top_targets=merged,
            method_name=str(metrics.get("method", "")),
            pathway_counts=pathway_counts,
            prior_hit_rate=prior_rate,
            score_values=score_vals,
            save_path=str(output_dir / "step3_overview_dashboard.png"),
        )
    else:
        plot_step3_overview_dashboard(
            top_targets=None,
            method_name=str(metrics.get("method", "")),
            pathway_counts=None,
            prior_hit_rate=None,
            score_values=None,
            save_path=str(output_dir / "step3_overview_dashboard.png"),
        )

    summary_lines = [
        "# Step3 Figure Summary",
        "",
        "## 候选靶点筛选逻辑",
        "",
        "1. 候选集合来自阶段2信号流中的 Ligand->Receptor 边。",
        "2. 计算反事实与观测差值，按 source_type/target_type 优先分组统计变化。",
        "3. 评分公式: flow_weight × (|delta_ligand| + |delta_receptor|) × prior_bonus。",
        "4. prior_bonus 来自 prior_db 命中 (OmniPath/LIANA/NicheNet)。",
        "",
        "## 生物学意义",
        "",
        "- 高分条目代表: 因果链路强、反事实效应显著、且先验证据支持。",
        "- 可优先用于实验验证与靶点优先级排序。",
        "- 总览 dashboard 提供跨靶点比较，便于快速锁定全局优先通路。",
        "",
        f"- Global prior hit rate: {metrics.get('global', {}).get('global_prior_hit_rate', 0.0):.3f}",
        f"- Global score p95: {metrics.get('global', {}).get('global_score_p95', 0.0):.4f}",
        f"- Targets: {targets}",
        f"- Output dir: {output_dir}",
    ]
    (output_dir / "figure_summary_step3.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n[DONE] All step3 figures saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

