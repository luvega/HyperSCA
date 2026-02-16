"""阶段 3 流水线: 反事实扰动、靶点排序与可视化。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config import HyperSCAConfig


class PerturbationPipeline:
    """阶段 3 端到端流程。"""

    def __init__(self, config: HyperSCAConfig):
        self.config = config
        self.step1_dir = Path(config.step3_input_step1_dir)
        self.step2_dir = Path(config.step3_input_step2_dir)
        self.output_dir = Path(config.step3_output_dir)
        self.fig_dir = Path(config.step3_figures_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(config.seed)

    def load_inputs(self) -> dict:
        """加载阶段1/2产物。"""
        import anndata as ad

        adata_path = self.step1_dir / "adata_embedded.h5ad"
        if not adata_path.exists():
            raise FileNotFoundError(f"Missing Stage1 adata: {adata_path}")
        adata = ad.read_h5ad(adata_path)

        cluster_expr_path = self.step2_dir / "cluster_expr.npy"
        node_info_path = self.step2_dir / "node_info.json"
        flow_path = self.step2_dir / "signaling_flow_edges.json"
        causal_adj_path = self.step2_dir / "causal_adjacency.npy"

        if not cluster_expr_path.exists():
            raise FileNotFoundError(f"Missing Stage2 cluster expression: {cluster_expr_path}")
        if not node_info_path.exists():
            raise FileNotFoundError(f"Missing Stage2 node info: {node_info_path}")

        cluster_expr = np.load(cluster_expr_path)
        node_info = json.loads(node_info_path.read_text(encoding="utf-8"))
        node_labels = node_info.get("node_labels", [])
        type_mapping = node_info.get("type_mapping", {})

        flow_edges = []
        if flow_path.exists():
            flow_edges = json.loads(flow_path.read_text(encoding="utf-8"))
        causal_adj = None
        if causal_adj_path.exists():
            causal_adj = np.load(causal_adj_path)

        gene_names = list(map(str, adata.var_names))
        if cluster_expr.shape[1] < len(gene_names):
            gene_names = gene_names[: cluster_expr.shape[1]]
        elif cluster_expr.shape[1] > len(gene_names):
            gene_names = gene_names + [f"GENE_{i}" for i in range(len(gene_names), cluster_expr.shape[1])]

        obs_expr = pd.DataFrame(cluster_expr, index=node_labels, columns=gene_names)

        # 加载空间坐标（如有）
        spatial_coords = None
        if "spatial" in adata.obsm:
            spatial_coords = np.asarray(adata.obsm["spatial"])
        elif "X_spatial" in adata.obsm:
            spatial_coords = np.asarray(adata.obsm["X_spatial"])

        return {
            "adata": adata,
            "obs_expr": obs_expr,
            "node_labels": node_labels,
            "type_mapping": type_mapping,
            "flow_edges": flow_edges,
            "causal_adj": causal_adj,
            "gene_names": gene_names,
            "spatial_coords": spatial_coords,
        }

    def _latent_arithmetic_cf(
        self,
        observed_expr: pd.DataFrame,
        target_gene: str,
        flow_edges: list[dict],
        node_to_type: dict[str, str],
    ) -> pd.DataFrame:
        """Latent KO：按 source/target 细胞类型差异化施加变化。

        当阶段1嵌入可用时使用 LatentArithmetic（真正双曲空间扰动），
        否则回退到表达空间简化版 KO。
        """
        cf = observed_expr.copy()
        g = target_gene.upper()
        gene_cols = {c.upper(): c for c in observed_expr.columns}
        if g not in gene_cols:
            return cf

        target_col = gene_cols[g]
        ko_scale = float(self.config.step3_latent_ko_scale)
        # 先全局 KO
        cf[target_col] = observed_expr[target_col] * max(0.0, 1.0 - ko_scale)

        # 再按 flow 中 target_gene 作为 ligand 的边，对对应 target_type 下 receptor 施加次级变化
        for edge in flow_edges:
            if edge.get("source_layer") != 0 or edge.get("target_layer") != 1:
                continue
            if str(edge.get("source", "")).upper() != g:
                continue
            receptor = str(edge.get("target", "")).upper()
            if receptor not in gene_cols:
                continue

            causal_edge = str(edge.get("causal_edge", ""))
            tgt_type = ""
            if "→" in causal_edge:
                parts = causal_edge.split("→")
                if len(parts) == 2:
                    tgt_type = parts[1].strip()
            elif "->" in causal_edge:
                parts = causal_edge.split("->")
                if len(parts) == 2:
                    tgt_type = parts[1].strip()

            rec_col = gene_cols[receptor]
            rows = [idx for idx, ctype in node_to_type.items() if ctype == tgt_type and idx in cf.index]
            if rows:
                cf.loc[rows, rec_col] = observed_expr.loc[rows, rec_col] * max(0.0, 1.0 - 0.5 * ko_scale)

        return cf

    @staticmethod
    def _build_gene_causal_mask(flow_edges: list[dict], gene_names: list[str]) -> np.ndarray:
        idx = {g.upper(): i for i, g in enumerate(gene_names)}
        mask = np.zeros((len(gene_names), len(gene_names)), dtype=float)
        for edge in flow_edges:
            s = str(edge.get("source", "")).upper()
            t = str(edge.get("target", "")).upper()
            if s in idx and t in idx:
                mask[idx[s], idx[t]] = 1.0
        return mask

    def _diffusion_cf(
        self,
        observed_expr: pd.DataFrame,
        target_gene: str,
        flow_edges: list[dict],
        gene_names: list[str],
    ) -> pd.DataFrame:
        from src.perturbation.diffusion_cf import CausalDiffusionCF, DiffusionConfig

        mask = self._build_gene_causal_mask(flow_edges, gene_names)
        model = CausalDiffusionCF(
            input_dim=len(gene_names),
            causal_mask=mask,
            device=self.config.device,
            config=DiffusionConfig(
                n_steps=self.config.step3_diffusion_steps,
                hidden_dim=self.config.step3_diffusion_hidden,
                train_epochs=self.config.step3_diffusion_epochs,
            ),
        )
        model.fit(observed_expr.values.astype(np.float32))
        x_cf = model.generate_counterfactual(
            x_observed=observed_expr.values.astype(np.float32),
            gene_names=gene_names,
            intervention={target_gene: float(self.config.step3_intervention_value)},
            seed=self.config.seed,
        )
        return pd.DataFrame(x_cf, index=observed_expr.index, columns=observed_expr.columns)

    def _rank_targets(
        self,
        flow_edges: list[dict],
        obs_expr: pd.DataFrame,
        cf_expr: pd.DataFrame,
        node_to_type: dict[str, str],
    ) -> pd.DataFrame:
        from src.perturbation.target_ranking import rank_counterfactual_interaction_targets

        ranked = rank_counterfactual_interaction_targets(
            flow_edges=flow_edges,
            observed_expression=obs_expr,
            counterfactual_expression=cf_expr,
            node_to_type=node_to_type,
            min_abs_delta=0.01,
            top_k=self.config.step3_target_top_k,
        )
        return ranked

    def _plot_for_target(
        self,
        target_gene: str,
        obs_expr: pd.DataFrame,
        cf_expr: pd.DataFrame,
        ranked: pd.DataFrame,
    ) -> None:
        from src.visualization.perturbation import (
            plot_interaction_target_ranking,
            plot_perturbation_comparison,
        )

        # top markers for comparison: include target + ranked ligand/receptor
        marker_candidates = [target_gene.upper()]
        if not ranked.empty:
            marker_candidates.extend(list(ranked["ligand"].head(10)))
            marker_candidates.extend(list(ranked["receptor"].head(10)))
        marker_candidates = list(dict.fromkeys(marker_candidates))
        gene_cols = {c.upper(): c for c in obs_expr.columns}
        marker_cols = [gene_cols[g] for g in marker_candidates if g in gene_cols]
        if not marker_cols:
            marker_cols = list(obs_expr.columns[: min(20, obs_expr.shape[1])])

        obs_mean = obs_expr[marker_cols].mean(axis=0).values
        cf_mean = cf_expr[marker_cols].mean(axis=0).values

        plot_perturbation_comparison(
            gene_names=marker_cols,
            observed=obs_mean,
            counterfactual=cf_mean,
            target_gene=target_gene,
            save_path=str(self.fig_dir / f"perturbation_comparison_{target_gene}.png"),
        )
        plot_interaction_target_ranking(
            ranked_targets=ranked,
            top_n=min(len(ranked), 20) if not ranked.empty else 20,
            save_path=str(self.fig_dir / f"interaction_target_ranking_{target_gene}.png"),
        )

    def _write_target_report(self, target_gene: str, ranked: pd.DataFrame) -> None:
        """输出包含筛选逻辑与生物意义说明的文字报告。"""
        lines = [
            f"# 阶段3候选靶点报告：{target_gene} KO",
            "",
            "## 筛选逻辑",
            "",
            "1. 以阶段2信号流中的 Ligand->Receptor 边作为候选互作集合。",
            "2. 计算反事实与观测表达差值，优先使用 source_type/target_type 分组变化：",
            "   - ligand 变化使用 source_type 对应细胞群均值差",
            "   - receptor 变化使用 target_type 对应细胞群均值差",
            "3. 目标分数 = flow_weight × (|delta_ligand| + |delta_receptor|) × prior_bonus。",
            "4. prior_bonus: 命中 prior_db (OmniPath/LIANA/NicheNet) 的 LR 对额外加权。",
            "",
            "## 生物学意义解释",
            "",
            "- 分数高表示该互作在因果链路强、反事实变化大、且有先验证据支持。",
            "- source_type/target_type 分组有助于避免全局均值稀释真实细胞群信号。",
            "- 可优先挑选 Top 条目进入后续实验验证（如 CRISPR/Perturb-seq）。",
            "",
            "## Top 候选",
            "",
        ]

        if ranked.empty:
            lines.append("- 当前阈值下无候选，请放宽 min_abs_delta 或检查输入表达矩阵。")
        else:
            for i, row in ranked.head(15).iterrows():
                lines.append(
                    f"- #{i+1} {row['ligand']}→{row['receptor']} "
                    f"(score={row['target_priority_score']:.4f}, prior={bool(row['prior_hit'])}, "
                    f"pathway={row.get('pathway', '')}, edge={row.get('causal_edge', '')})"
                )

        out = self.output_dir / f"interpretation_targets_{target_gene}.md"
        out.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _compute_target_metrics(
        target_gene: str,
        ranked: pd.DataFrame,
        obs_expr: pd.DataFrame,
        cf_expr: pd.DataFrame,
    ) -> dict:
        """计算每个靶点的摘要与dashboard指标。"""
        gene_cols = {c.upper(): c for c in obs_expr.columns}
        tgt_col = gene_cols.get(target_gene.upper())

        # 观测与反事实差异（无真实干预标签时，作为稳定性/变化幅度代理）
        common_rows = [r for r in cf_expr.index if r in obs_expr.index]
        common_cols = [c for c in cf_expr.columns if c in obs_expr.columns]
        obs_sub = obs_expr.loc[common_rows, common_cols]
        cf_sub = cf_expr.loc[common_rows, common_cols]
        obs_mean = obs_sub.mean(axis=0).values.astype(float)
        cf_mean = cf_sub.mean(axis=0).values.astype(float)

        mse = float(np.mean((cf_mean - obs_mean) ** 2))
        denom = float(np.sum((obs_mean - obs_mean.mean()) ** 2))
        if denom > 1e-12:
            r2_like = float(1.0 - np.sum((cf_mean - obs_mean) ** 2) / denom)
        else:
            r2_like = 0.0
        if np.std(obs_mean) > 1e-12 and np.std(cf_mean) > 1e-12:
            pcc_like = float(np.corrcoef(obs_mean, cf_mean)[0, 1])
        else:
            pcc_like = 0.0

        if tgt_col is not None:
            delta_target = float(cf_expr[tgt_col].mean() - obs_expr[tgt_col].mean())
            marker_direction_accuracy = 1.0 if delta_target <= 0 else 0.0
        else:
            delta_target = 0.0
            marker_direction_accuracy = 0.0

        if ranked.empty:
            top_score = 0.0
            prior_hit_rate = 0.0
            score_p95 = 0.0
            score_median = 0.0
            top_pathway = ""
        else:
            scores = ranked["target_priority_score"].astype(float).values
            top_score = float(scores.max())
            prior_hit_rate = float(ranked["prior_hit"].astype(bool).mean())
            score_p95 = float(np.percentile(scores, 95))
            score_median = float(np.median(scores))
            top_pathway = str(
                ranked.sort_values("target_priority_score", ascending=False).iloc[0].get("pathway", "")
            )

        return {
            "n_candidates": int(len(ranked)),
            "top_score": top_score,
            "prior_hit_rate": prior_hit_rate,
            "score_p95": score_p95,
            "score_median": score_median,
            "top_pathway": top_pathway,
            "delta_target_gene_mean_expr": delta_target,
            "dashboard_metrics": {
                "r2_mean": r2_like,
                "pcc_median": pcc_like,
                "mse": mse,
                "marker_direction_accuracy": marker_direction_accuracy,
            },
        }

    def _run_spatial_propagation(
        self,
        target_gene: str,
        obs_expr: pd.DataFrame,
        cf_expr: pd.DataFrame,
        causal_adj: np.ndarray | None,
        spatial_coords: np.ndarray | None,
    ) -> dict:
        """运行空间传播模拟并返回结果。"""
        if causal_adj is None:
            return {}

        from src.perturbation.spatial_propagation import propagate_perturbation

        K = causal_adj.shape[0]
        gene_cols = {c.upper(): c for c in obs_expr.columns}
        tgt_col = gene_cols.get(target_gene.upper())

        # 构建初始效应向量
        source_delta = np.zeros(K, dtype=float)
        if tgt_col is not None:
            delta_vals = (cf_expr[tgt_col].values - obs_expr[tgt_col].values).astype(float)
            source_delta[:len(delta_vals)] = delta_vals[:K]

        # 找 source 节点（表达变化最大的节点）
        source_nodes = []
        abs_delta = np.abs(source_delta)
        if abs_delta.max() > 1e-12:
            threshold = abs_delta.max() * 0.3
            source_nodes = list(np.where(abs_delta >= threshold)[0])
        if not source_nodes and K > 0:
            source_nodes = [int(np.argmax(abs_delta))]

        result = propagate_perturbation(
            causal_adj=causal_adj,
            source_nodes=source_nodes,
            source_delta=source_delta,
            spatial_coords=spatial_coords[:K] if spatial_coords is not None and spatial_coords.shape[0] >= K else None,
            decay_length=self.config.step3_spatial_decay_length,
            max_depth=self.config.step3_propagation_max_depth,
            convergence_tol=self.config.step3_propagation_threshold,
        )
        return result

    def _evaluate_cf_metrics(
        self,
        obs_expr: pd.DataFrame,
        cf_expr: pd.DataFrame,
        gene_names: list[str],
        target_gene: str,
    ) -> dict:
        """使用 cf_metrics 评估反事实质量。"""
        from src.evaluation.cf_metrics import evaluate_counterfactual

        common_rows = [r for r in cf_expr.index if r in obs_expr.index]
        common_cols = [c for c in cf_expr.columns if c in obs_expr.columns]
        obs_sub = obs_expr.loc[common_rows, common_cols].values.astype(float)
        cf_sub = cf_expr.loc[common_rows, common_cols].values.astype(float)

        # KO 基因预期方向: 下调
        expected_directions = {target_gene: -1}

        return evaluate_counterfactual(
            observed=obs_sub,
            counterfactual=cf_sub,
            gene_names=list(common_cols),
            expected_directions=expected_directions,
        )

    def _evaluate_spatial_metrics(
        self,
        obs_expr: pd.DataFrame,
        cf_expr: pd.DataFrame,
        target_gene: str,
        spatial_coords: np.ndarray | None,
        propagation_result: dict,
        causal_adj: np.ndarray | None,
    ) -> dict:
        """使用 spatial_metrics 评估空间传播一致性。"""
        if spatial_coords is None:
            return {}

        from src.evaluation.spatial_metrics import evaluate_spatial_propagation

        gene_cols = {c.upper(): c for c in obs_expr.columns}
        tgt_col = gene_cols.get(target_gene.upper())
        if tgt_col is None:
            return {}

        K = obs_expr.shape[0]
        coords = spatial_coords[:K] if spatial_coords.shape[0] >= K else spatial_coords

        effect = propagation_result.get("effect", np.zeros(K))
        if effect.ndim == 2:
            effect_mag = np.mean(np.abs(effect), axis=1)
        else:
            effect_mag = np.abs(effect)

        # 到 source 最近距离
        bfs_layers = propagation_result.get("bfs_layers", [])
        source_nodes = bfs_layers[0]["nodes"] if bfs_layers else [0]
        from scipy.spatial.distance import cdist
        if coords.shape[0] > 0 and len(source_nodes) > 0:
            src_dists = cdist(coords, coords[source_nodes]).min(axis=1)
        else:
            src_dists = np.zeros(K)

        return evaluate_spatial_propagation(
            coords=coords,
            effect_magnitudes=effect_mag[:K],
            source_distances=src_dists,
            bfs_layers=bfs_layers if bfs_layers else None,
            causal_adj=causal_adj,
            observed_expr=obs_expr[tgt_col].values.astype(float),
            counterfactual_expr=cf_expr[tgt_col].values.astype(float),
            threshold=self.config.step3_propagation_threshold,
        )

    def run(self) -> dict:
        t0 = time.time()
        data = self.load_inputs()
        obs_expr = data["obs_expr"]
        node_to_type = data["type_mapping"]
        flow_edges = data["flow_edges"]
        gene_names = data["gene_names"]
        causal_adj = data.get("causal_adj")
        spatial_coords = data.get("spatial_coords")

        all_ranked = {}
        all_metrics = {}
        fold_change_rows = []

        for target_gene in self.config.step3_target_genes:
            print("=" * 60)
            print(f"[Step3] Target={target_gene} | method={self.config.step3_method}")
            print("=" * 60)

            if self.config.step3_method == "diffusion_cf":
                cf_expr = self._diffusion_cf(
                    observed_expr=obs_expr,
                    target_gene=target_gene,
                    flow_edges=flow_edges,
                    gene_names=gene_names,
                )
            else:
                cf_expr = self._latent_arithmetic_cf(
                    observed_expr=obs_expr,
                    target_gene=target_gene,
                    flow_edges=flow_edges,
                    node_to_type=node_to_type,
                )

            # 保存反事实表达
            cf_expr.to_csv(self.output_dir / f"cf_expression_{target_gene}.csv")

            ranked = self._rank_targets(
                flow_edges=flow_edges,
                obs_expr=obs_expr,
                cf_expr=cf_expr,
                node_to_type=node_to_type,
            )
            ranked.to_csv(self.output_dir / f"interaction_targets_{target_gene}.csv", index=False)
            ranked.to_json(
                self.output_dir / f"interaction_targets_{target_gene}.json",
                orient="records",
                force_ascii=False,
                indent=2,
            )
            self._plot_for_target(target_gene, obs_expr, cf_expr, ranked)
            self._write_target_report(target_gene, ranked)

            # 空间传播
            propagation_result = self._run_spatial_propagation(
                target_gene=target_gene,
                obs_expr=obs_expr,
                cf_expr=cf_expr,
                causal_adj=causal_adj,
                spatial_coords=spatial_coords,
            )
            if propagation_result:
                prop_out = {
                    "bfs_layers": [
                        {k: (v if k != "nodes" else [int(x) for x in v])
                         for k, v in layer.items()}
                        for layer in propagation_result.get("bfs_layers", [])
                    ],
                    "fit_params": propagation_result.get("fit_params", {}),
                }
                (self.output_dir / f"propagation_{target_gene}.json").write_text(
                    json.dumps(prop_out, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            # 反事实质量评估
            cf_quality = self._evaluate_cf_metrics(
                obs_expr=obs_expr,
                cf_expr=cf_expr,
                gene_names=gene_names,
                target_gene=target_gene,
            )
            print(f"  CF metrics: {cf_quality}")

            # 空间一致性评估
            spatial_quality = self._evaluate_spatial_metrics(
                obs_expr=obs_expr,
                cf_expr=cf_expr,
                target_gene=target_gene,
                spatial_coords=spatial_coords,
                propagation_result=propagation_result,
                causal_adj=causal_adj,
            )
            if spatial_quality:
                print(f"  Spatial metrics: {spatial_quality}")

            # 汇总指标
            target_metrics = self._compute_target_metrics(
                target_gene=target_gene,
                ranked=ranked,
                obs_expr=obs_expr,
                cf_expr=cf_expr,
            )
            target_metrics["method"] = self.config.step3_method
            target_metrics["cf_quality"] = cf_quality
            target_metrics["spatial_quality"] = spatial_quality
            if propagation_result:
                target_metrics["propagation_fit_params"] = propagation_result.get("fit_params", {})
            all_metrics[target_gene] = target_metrics
            all_ranked[target_gene] = ranked

            # multi-target heatmap 使用：每个 target 对 top marker 的全局 FC
            top_markers = list(dict.fromkeys([target_gene.upper()] + list(ranked["ligand"].head(6)) + list(ranked["receptor"].head(6))))
            gene_cols = {c.upper(): c for c in obs_expr.columns}
            markers = [gene_cols[g] for g in top_markers if g in gene_cols]
            if markers:
                fc = (cf_expr[markers].mean(axis=0) - obs_expr[markers].mean(axis=0)).values
                fold_change_rows.append((target_gene, markers, fc))

        # 跨靶点热图（对齐 marker 联合集）
        if fold_change_rows:
            from src.visualization.perturbation import plot_multi_target_heatmap

            marker_union = []
            for _, markers, _ in fold_change_rows:
                marker_union.extend(markers)
            marker_union = list(dict.fromkeys(marker_union))[:20]
            mat = np.zeros((len(fold_change_rows), len(marker_union)), dtype=float)
            targets = []
            for i, (tgt, markers, _) in enumerate(fold_change_rows):
                targets.append(tgt)
                cf = None
                obs = None
                # 重新读取对应表达，避免缓存全部 target 的 cf_expr
                cf_path = self.output_dir / f"cf_expression_{tgt}.csv"
                cf = pd.read_csv(cf_path, index_col=0)
                obs = obs_expr
                for j, g in enumerate(marker_union):
                    if g in cf.columns and g in obs.columns:
                        mat[i, j] = float(cf[g].mean() - obs[g].mean())
            plot_multi_target_heatmap(
                targets=targets,
                marker_genes=marker_union,
                fold_changes=mat,
                save_path=str(self.fig_dir / "multi_target_heatmap_step3.png"),
            )

        # 全局汇总
        all_scores = []
        all_prior = []
        for tgt in all_ranked:
            df = all_ranked[tgt]
            if df is None or df.empty:
                continue
            all_scores.extend(df["target_priority_score"].astype(float).tolist())
            all_prior.extend(df["prior_hit"].astype(bool).tolist())

        summary = {
            "method": self.config.step3_method,
            "targets": self.config.step3_target_genes,
            "per_target": all_metrics,
            "global": {
                "n_total_candidates": int(sum(m.get("n_candidates", 0) for m in all_metrics.values())),
                "global_prior_hit_rate": float(np.mean(all_prior)) if all_prior else 0.0,
                "global_score_p95": float(np.percentile(all_scores, 95)) if all_scores else 0.0,
                "global_score_median": float(np.median(all_scores)) if all_scores else 0.0,
            },
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        (self.output_dir / "step3_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\n[Pipeline] Stage 3 total time: {summary['elapsed_seconds']}s")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Figure dir: {self.fig_dir}")
        return summary


def _json_default(obj):
    """JSON 序列化辅助：处理 numpy 类型。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

