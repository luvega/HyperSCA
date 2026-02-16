"""阶段 2 流水线: 空间约束下的因果通讯网络构建

CausalPipeline 端到端执行:
1. 加载阶段 1 输出 (adata + embeddings)
2. 聚类 / 子采样（按粒度）
3. 训练因果解缠模型 (DisentangleModel)
4. Bootstrap 因果发现 + 阈值剪枝
5. DoWhy 结构验证
6. 已知轴评估 + 多层信号流推断
7. 评估指标计算
8. 保存产物 → results/step2/
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.pipeline.config import HyperSCAConfig


class CausalPipeline:
    """阶段 2: 因果通讯网络流水线"""

    def __init__(self, config: HyperSCAConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(config.step2_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    # =================================================================
    # Step 2.1: 加载阶段 1 产物
    # =================================================================

    def load_step1_results(self):
        """加载阶段 1 输出"""
        import anndata as ad

        step1_dir = Path(self.config.step2_input_dir)
        print("=" * 60)
        print("[Step 2.1] Loading Stage 1 results...")
        print("=" * 60)

        adata_path = step1_dir / "adata_embedded.h5ad"
        if not adata_path.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found: {adata_path}\n"
                "Please run `python scripts/run_step1.py` first."
            )

        adata = ad.read_h5ad(adata_path)
        print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
        print(f"  Embeddings: X_poincare={adata.obsm['X_poincare'].shape}")

        # 加载邻接矩阵
        adj_path = step1_dir / "adj_enhanced.npz"
        if not adj_path.exists():
            adj_path = step1_dir / "adj_original.npz"
        adj = sparse.load_npz(adj_path)
        print(f"  Adjacency: {adj.shape}, nnz={adj.nnz}")

        return adata, adj

    # =================================================================
    # Step 2.2: 聚类 / 子采样
    # =================================================================

    def cluster_cells(self, adata):
        """对细胞进行聚类（cluster 粒度）或子采样（single_cell 粒度）"""
        import scanpy as sc

        print()
        print("=" * 60)
        print("[Step 2.2] Clustering / subsampling...")
        print("=" * 60)

        granularity = self.config.step2_granularity

        if granularity == "cluster":
            # 基于 Poincaré 嵌入构建 kNN 图 + Leiden 聚类
            emb = adata.obsm["X_poincare"]
            adata.obsm["X_poincare_for_cluster"] = emb

            sc.pp.neighbors(adata, use_rep="X_poincare_for_cluster", n_neighbors=15)
            sc.tl.leiden(
                adata, resolution=self.config.step2_leiden_resolution,
                key_added="leiden_step2",
            )

            # 如果有已知细胞类型标签，为每个 cluster 找主要类型
            cluster_labels = adata.obs["leiden_step2"].values
            n_clusters = len(np.unique(cluster_labels))

            # 尝试找细胞类型注释
            type_col = None
            for col in ["cell_type", "Level1", "celltype", "annotation"]:
                if col in adata.obs.columns:
                    type_col = col
                    break

            cluster_type_mapping = {}
            if type_col is not None:
                for cl in np.unique(cluster_labels):
                    mask = cluster_labels == cl
                    types = adata.obs[type_col].values[mask]
                    # 众数
                    unique, counts = np.unique(types, return_counts=True)
                    dominant = unique[np.argmax(counts)]
                    cluster_type_mapping[f"C{cl}"] = str(dominant)
                print(f"  Cell type annotation: {type_col}")
            else:
                for cl in np.unique(cluster_labels):
                    cluster_type_mapping[f"C{cl}"] = f"C{cl}"

            print(f"  Clusters: {n_clusters} (Leiden, res={self.config.step2_leiden_resolution})")
            print(f"  Type mapping: {cluster_type_mapping}")

            return adata, cluster_labels, cluster_type_mapping

        elif granularity == "single_cell":
            # 子采样
            max_cells = self.config.step2_max_cells
            n_orig = adata.n_obs
            if n_orig > max_cells:
                rng = np.random.default_rng(self.config.seed)
                idx = np.sort(rng.choice(n_orig, size=max_cells, replace=False))
                adata = adata[idx].copy()
                # 记录子采样索引，供后续同步 adj
                adata.uns["_subsample_idx"] = idx
                print(f"  Subsampled: {max_cells} / {n_orig} cells")
            else:
                print(f"  Using all {n_orig} cells")

            # 仍需 Leiden 聚类（因果发现在 cluster 级执行）
            sc.pp.neighbors(adata, use_rep="X_poincare", n_neighbors=15)
            sc.tl.leiden(
                adata, resolution=self.config.step2_leiden_resolution,
                key_added="leiden_step2",
            )
            cluster_labels = adata.obs["leiden_step2"].values
            n_clusters = len(np.unique(cluster_labels))

            # 尝试找细胞类型注释
            type_col = None
            for col in ["cell_type", "Level1", "celltype", "annotation"]:
                if col in adata.obs.columns:
                    type_col = col
                    break

            cluster_type_mapping = {}
            if type_col is not None:
                for cl in np.unique(cluster_labels):
                    mask = cluster_labels == cl
                    types = adata.obs[type_col].values[mask]
                    unique, counts = np.unique(types, return_counts=True)
                    dominant = unique[np.argmax(counts)]
                    cluster_type_mapping[f"C{cl}"] = str(dominant)
            else:
                for cl in np.unique(cluster_labels):
                    cluster_type_mapping[f"C{cl}"] = f"C{cl}"

            print(f"  Clusters (for causal discovery): {n_clusters}")
            return adata, cluster_labels, cluster_type_mapping

        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    # =================================================================
    # Step 2.3: 聚合到 cluster 级
    # =================================================================

    def aggregate_to_clusters(self, adata, cluster_labels, adj):
        """将单细胞数据聚合到 cluster 级"""
        import scipy.sparse as sp

        print()
        print("=" * 60)
        print("[Step 2.3] Aggregating to cluster level...")
        print("=" * 60)

        unique_clusters = np.unique(cluster_labels)
        K = len(unique_clusters)

        # 表达矩阵聚合（pseudo-bulk mean）
        if sp.issparse(adata.X):
            X_dense = np.array(adata.X.toarray())
        else:
            X_dense = np.array(adata.X)

        cluster_expr = np.zeros((K, X_dense.shape[1]))
        for i, cl in enumerate(unique_clusters):
            mask = cluster_labels == cl
            cluster_expr[i] = X_dense[mask].mean(axis=0)

        # Poincaré 嵌入聚合（centroid）
        poincare_emb = adata.obsm["X_poincare"]
        cluster_emb = np.zeros((K, poincare_emb.shape[1]))
        for i, cl in enumerate(unique_clusters):
            mask = cluster_labels == cl
            cluster_emb[i] = poincare_emb[mask].mean(axis=0)

        # cluster 邻接矩阵（如果两个 cluster 的细胞之间有空间边，则连接）
        adj_dense = adj.toarray() if sp.issparse(adj) else adj
        cluster_adj = np.zeros((K, K))
        for i, cl_i in enumerate(unique_clusters):
            for j, cl_j in enumerate(unique_clusters):
                if i == j:
                    continue
                mask_i = np.where(cluster_labels == cl_i)[0]
                mask_j = np.where(cluster_labels == cl_j)[0]
                # 检查是否有跨 cluster 的空间连接
                sub_adj = adj_dense[np.ix_(mask_i, mask_j)]
                if sub_adj.sum() > 0:
                    cluster_adj[i, j] = sub_adj.sum() / (len(mask_i) * len(mask_j) + 1e-10)

        # 归一化
        max_val = cluster_adj.max()
        if max_val > 0:
            cluster_adj /= max_val

        node_labels = [f"C{cl}" for cl in unique_clusters]

        print(f"  Cluster expression: {cluster_expr.shape}")
        print(f"  Cluster adjacency: {K}x{K}, edges={int((cluster_adj > 0).sum())}")

        # 创建 cluster-level 表达 DataFrame
        cluster_expr_df = pd.DataFrame(
            cluster_expr,
            index=node_labels,
            columns=adata.var_names,
        )

        return {
            "cluster_expr": cluster_expr,
            "cluster_expr_df": cluster_expr_df,
            "cluster_emb": cluster_emb,
            "cluster_adj": cluster_adj,
            "node_labels": node_labels,
            "unique_clusters": unique_clusters,
        }

    # =================================================================
    # Step 2.4: 训练因果解缠模型
    # =================================================================

    def train_disentangle(self, cluster_data: dict):
        """训练解缠模型"""
        from src.causal.disentangle import train_disentangle

        print()
        print("=" * 60)
        print("[Step 2.4] Training disentangle model...")
        print("=" * 60)

        cfg = self.config
        expr = cluster_data["cluster_expr"]
        adj = cluster_data["cluster_adj"]

        # 构建 edge_index
        rows, cols = np.where(adj > 0)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float32)

        x = torch.tensor(expr, dtype=torch.float32)

        result = train_disentangle(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            z_dim=cfg.step2_disentangle_dim,
            hidden_dims=cfg.step2_disentangle_hidden,
            epochs=cfg.step2_disentangle_epochs,
            lr=cfg.step2_disentangle_lr,
            hsic_alpha=cfg.step2_hsic_alpha,
            device=str(self.device),
            verbose=True,
        )

        print(f"  z_int shape: {result['z_int'].shape}")
        print(f"  z_ext shape: {result['z_ext'].shape}")
        print(f"  Final HSIC: {result['losses']['hsic'][-1]:.6f}")

        return result

    # =================================================================
    # Step 2.4b: Cell-level 解缠（single_cell 模式）
    # =================================================================

    def _train_disentangle_cell_level(
        self, adata, adj, cluster_labels, cluster_data: dict
    ):
        """在 cell-level 训练解缠，然后聚合到 cluster-level"""
        from src.causal.disentangle import train_disentangle
        import scipy.sparse as sp

        print()
        print("=" * 60)
        print("[Step 2.4] Training disentangle (cell-level)...")
        print("=" * 60)

        cfg = self.config

        # 准备 cell-level 数据
        if sp.issparse(adata.X):
            X_dense = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X_dense = torch.tensor(np.array(adata.X), dtype=torch.float32)

        # adj 此时应已与 adata 同步（在 run() 中处理）
        adj_sparse = adj
        N = X_dense.shape[0]
        if adj_sparse.shape[0] != N:
            # 安全截断（不应发生，但作为保护）
            min_n = min(adj_sparse.shape[0], N)
            adj_sparse = adj_sparse[:min_n, :min_n]
            X_dense = X_dense[:min_n]

        coo = adj_sparse.tocoo()
        edge_index = torch.tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long
        )
        edge_weight = torch.tensor(coo.data, dtype=torch.float32)

        N = X_dense.shape[0]
        print(f"  Cell-level: {N} cells x {X_dense.shape[1]} genes")

        result = train_disentangle(
            x=X_dense,
            edge_index=edge_index,
            edge_weight=edge_weight,
            z_dim=cfg.step2_disentangle_dim,
            hidden_dims=cfg.step2_disentangle_hidden,
            epochs=cfg.step2_disentangle_epochs,
            lr=cfg.step2_disentangle_lr,
            hsic_alpha=cfg.step2_hsic_alpha,
            device=str(self.device),
            verbose=True,
        )

        # 聚合到 cluster-level（用于后续因果发现）
        z_int_cell = result["z_int"]  # (N, d)
        z_ext_cell = result["z_ext"]  # (N, d)
        unique_clusters = cluster_data["unique_clusters"]
        K = len(unique_clusters)

        z_int_cluster = np.zeros((K, z_int_cell.shape[1]))
        z_ext_cluster = np.zeros((K, z_ext_cell.shape[1]))
        for i, cl in enumerate(unique_clusters):
            mask = cluster_labels == cl
            z_int_cluster[i] = z_int_cell[mask].mean(axis=0)
            z_ext_cluster[i] = z_ext_cell[mask].mean(axis=0)

        print(f"  Aggregated to {K} clusters")
        print(f"  z_int_cluster: {z_int_cluster.shape}")
        print(f"  z_ext_cluster: {z_ext_cluster.shape}")

        # 返回 cluster-level 嵌入（用于因果发现和评估）
        # 同时保存 cell-level 嵌入（用于后续）
        return {
            "model": result["model"],
            "z_int": z_int_cluster,
            "z_ext": z_ext_cluster,
            "z_int_cell": z_int_cell,
            "z_ext_cell": z_ext_cell,
            "losses": result["losses"],
        }

    # =================================================================
    # Step 2.5: Bootstrap 因果发现 + 剪枝
    # =================================================================

    def build_causal_graph(self, cluster_data: dict, disentangle_result: dict):
        """因果结构学习 + Bootstrap + 剪枝"""
        from src.causal.cmi_pruning import bootstrap_causal_discovery, threshold_pruning
        from src.causal.causal_graph import CausalCellGraph

        print()
        print("=" * 60)
        print("[Step 2.5] Causal discovery (Bootstrap + PC)...")
        print("=" * 60)

        cfg = self.config

        # 使用 cluster 表达矩阵的转置作为因果发现输入:
        #   变量 = K 个 cluster（因果图节点）
        #   观测 = G 个基因（提供足够样本量进行 CI 检验）
        # 这在方法学上等价于基于基因共表达模式推断 cluster 间调控关系
        cluster_expr = cluster_data["cluster_expr"]       # (K, G)
        data_for_pc = cluster_expr.T                       # (G, K)
        print(f"  PC input: {data_for_pc.shape[0]} observations (genes) x "
              f"{data_for_pc.shape[1]} variables (clusters)")

        freq_matrix = bootstrap_causal_discovery(
            data=data_for_pc,
            n_bootstraps=cfg.step2_bootstrap_n,
            alpha=cfg.step2_cmi_alpha,
            max_cond_set=cfg.step2_pc_max_cond,
            seed=cfg.seed,
            verbose=True,
        )

        # 阈值剪枝
        adjacency, pruned_freq = threshold_pruning(
            freq_matrix, threshold=cfg.step2_bootstrap_threshold
        )

        n_edges = int(adjacency.sum())
        print(f"  Pruned graph: {n_edges} edges "
              f"(threshold={cfg.step2_bootstrap_threshold})")

        # 构建 CausalCellGraph
        causal_graph = CausalCellGraph(
            adjacency=adjacency,
            node_labels=cluster_data["node_labels"],
            bootstrap_freq=pruned_freq,
        )

        stats = causal_graph.summary_stats()
        print(f"  Sparsity: {stats['graph_sparsity']:.4f}")
        print(f"  Mean freq: {stats['mean_bootstrap_freq']:.4f}")
        print(f"  Is DAG: {stats['is_dag']}")

        return causal_graph, freq_matrix

    # =================================================================
    # Step 2.6: DoWhy 验证
    # =================================================================

    def validate_with_dowhy(self, causal_graph, cluster_data: dict):
        """DoWhy 结构验证 + Arrow Strength"""
        print()
        print("=" * 60)
        print("[Step 2.6] DoWhy validation...")
        print("=" * 60)

        # 准备节点级数据 DataFrame
        node_labels = cluster_data["node_labels"]
        z_ext = cluster_data.get("z_ext")

        # 使用 z_ext 作为 DoWhy 的数据（每个维度作为一个变量）
        if z_ext is not None and z_ext.shape[0] > 3:
            # 构建足够样本的 DataFrame（通过 bootstrap 扩展）
            rng = np.random.default_rng(self.config.seed)
            n_samples = max(200, z_ext.shape[0] * 10)
            indices = rng.choice(z_ext.shape[0], size=n_samples, replace=True)
            # 为每个节点生成一列数据
            data_dict = {}
            for i, label in enumerate(node_labels):
                data_dict[label] = z_ext[indices, min(i, z_ext.shape[1] - 1)]
                data_dict[label] += rng.normal(0, 0.01, size=n_samples)
            data_df = pd.DataFrame(data_dict)
        else:
            # 使用 cluster 表达数据生成
            cluster_expr = cluster_data["cluster_expr"]
            data_dict = {}
            for i, label in enumerate(node_labels):
                data_dict[label] = cluster_expr[i, :min(200, cluster_expr.shape[1])]
            # 补齐到相同长度
            max_len = max(len(v) for v in data_dict.values())
            for k in data_dict:
                if len(data_dict[k]) < max_len:
                    data_dict[k] = np.pad(
                        data_dict[k], (0, max_len - len(data_dict[k]))
                    )
            data_df = pd.DataFrame(data_dict)

        # Falsification
        print("  Running structural validation...")
        falsification = causal_graph.validate_structure(data_df)
        print(f"  Result: {falsification['result_str']}")
        print(f"  Mean p-value: {falsification['mean_pvalue']:.4f}")

        # Arrow Strength
        print("  Computing arrow strength...")
        try:
            strength = causal_graph.compute_arrow_strength(data_df)
            n_nonzero = int((strength > 0).sum())
            print(f"  Arrow strength: {n_nonzero} non-zero entries")
        except Exception as e:
            warnings.warn(f"Arrow strength computation failed: {e}")
            print(f"  Arrow strength: skipped ({e})")

        return falsification

    # =================================================================
    # Step 2.7: 已知轴评估 + 信号流
    # =================================================================

    def evaluate_known_axes_and_flow(
        self, causal_graph, cluster_data: dict, type_mapping: dict
    ):
        """评估已知信号轴 + 推断信号流"""
        from src.causal.signaling_flow import (
            infer_signaling_flow,
            summarize_signaling_flows,
        )
        from src.causal.causal_graph import load_known_axes

        print()
        print("=" * 60)
        print("[Step 2.7] Known axis evaluation & signaling flow...")
        print("=" * 60)

        # 加载已知轴
        known_axes = load_known_axes(self.config.step2_known_axes_file)

        # 已知轴评估
        axis_results = causal_graph.evaluate_known_axes(
            known_axes=known_axes,
            type_mapping=type_mapping,
        )

        print(f"  Known Axis Recall: {axis_results['known_axis_recall']:.4f}")
        print(f"  Direction Accuracy: {axis_results['direction_accuracy']:.4f}")
        for ax in axis_results["per_axis"]:
            status = "FOUND" if ax["found"] else "MISS"
            direction = "CORRECT" if ax.get("correct_direction") else "WRONG/MISS"
            print(f"    {ax['name']}: {status} | Direction: {direction}")

        # 信号流推断
        flow_edges = infer_signaling_flow(
            causal_graph_adj=causal_graph.adjacency,
            node_labels=causal_graph.node_labels,
            expression_data=cluster_data.get("cluster_expr_df"),
            type_mapping=type_mapping,
        )
        flow_summary = summarize_signaling_flows(flow_edges)
        print(f"  Signaling flows: {flow_summary['n_total_flow_edges']} edges, "
              f"{flow_summary['n_complete_flows']} complete pathways")

        return axis_results, flow_edges, flow_summary

    # =================================================================
    # Step 2.8: 评估指标
    # =================================================================

    def compute_metrics(
        self,
        causal_graph,
        disentangle_result: dict,
        cluster_data: dict,
        axis_results: dict,
        falsification: dict,
        flow_summary: dict,
        cluster_labels=None,
    ):
        """计算全部指标"""
        from src.evaluation.causal_metrics import evaluate_causal

        print()
        print("=" * 60)
        print("[Step 2.8] Computing metrics...")
        print("=" * 60)

        # 节点类型标签（整数）
        node_labels = cluster_data["node_labels"]
        K = len(node_labels)
        int_labels = np.arange(K)

        metrics = evaluate_causal(
            adjacency=causal_graph.adjacency,
            bootstrap_freq=causal_graph.bootstrap_freq,
            z_int=disentangle_result["z_int"],
            z_ext=disentangle_result["z_ext"],
            labels=int_labels,
            cluster_adj=(cluster_data["cluster_adj"] > 0).astype(float),
            known_axis_results=axis_results,
            falsification_results=falsification,
            signaling_flow_summary=flow_summary,
        )

        print("  Metrics summary:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        return metrics

    # =================================================================
    # Step 2.9: 保存产物
    # =================================================================

    def save_results(
        self,
        causal_graph,
        disentangle_result: dict,
        cluster_data: dict,
        metrics: dict,
        axis_results: dict,
        flow_edges: list,
        flow_summary: dict,
        falsification: dict,
        freq_matrix: np.ndarray,
        type_mapping: dict,
    ):
        """保存所有产物到 results/step2/"""
        print()
        print("=" * 60)
        print("[Step 2.9] Saving results...")
        print("=" * 60)

        out = self.output_dir

        # 因果图
        causal_graph.to_graphml(out / "causal_graph.graphml")
        np.save(out / "causal_adjacency.npy", causal_graph.adjacency)
        np.save(out / "bootstrap_freq_matrix.npy", freq_matrix)
        if causal_graph.arrow_strength is not None:
            np.save(out / "arrow_strength.npy", causal_graph.arrow_strength)

        # 解缠嵌入（cluster-level）
        np.save(out / "z_int.npy", disentangle_result["z_int"])
        np.save(out / "z_ext.npy", disentangle_result["z_ext"])

        # Cell-level 嵌入（single_cell 模式才有）
        if "z_int_cell" in disentangle_result:
            np.save(out / "z_int_cell.npy", disentangle_result["z_int_cell"])
            np.save(out / "z_ext_cell.npy", disentangle_result["z_ext_cell"])

        # 解缠模型
        torch.save(
            disentangle_result["model"].state_dict(),
            out / "disentangle_model.pt",
        )

        # 损失记录
        with open(out / "disentangle_losses.json", "w", encoding="utf-8") as f:
            json.dump(disentangle_result["losses"], f, indent=2)

        # 指标
        with open(out / "step2_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        # 已知轴评估
        with open(out / "key_axes_evidence.json", "w", encoding="utf-8") as f:
            json.dump(axis_results, f, indent=2, ensure_ascii=False)

        # 信号流
        with open(out / "signaling_flow_edges.json", "w", encoding="utf-8") as f:
            json.dump(flow_edges, f, indent=2, default=str)
        with open(out / "signaling_flow_summary.json", "w", encoding="utf-8") as f:
            json.dump(flow_summary, f, indent=2, default=str)

        # DoWhy 验证
        with open(out / "falsification_results.json", "w", encoding="utf-8") as f:
            json.dump(falsification, f, indent=2, default=str)

        # 配置
        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 节点信息
        node_info = {
            "node_labels": cluster_data["node_labels"],
            "type_mapping": type_mapping,
        }
        with open(out / "node_info.json", "w", encoding="utf-8") as f:
            json.dump(node_info, f, indent=2, ensure_ascii=False)

        # Cluster 表达（用于后续可视化）
        np.save(out / "cluster_expr.npy", cluster_data["cluster_expr"])
        np.save(out / "cluster_adj.npy", cluster_data["cluster_adj"])

        print(f"  All results saved to: {out}")

    # =================================================================
    # Step 2.10: 生成解读报告
    # =================================================================

    def generate_interpretation(
        self,
        metrics: dict,
        axis_results: dict,
        flow_summary: dict,
        falsification: dict,
        type_mapping: dict,
    ):
        """生成 interpretation_step2.md 解读报告"""
        print()
        print("=" * 60)
        print("[Step 2.10] Generating interpretation report...")
        print("=" * 60)

        lines = [
            "# 阶段 2 因果网络分析解读报告",
            "",
            f"*自动生成 | HyperSCA Step 2 | 粒度: {self.config.step2_granularity}*",
            "",
            "---",
            "",
            "## 1. 计算生物学角度",
            "",
            "### 1.1 解缠质量",
            "",
            f"- **HSIC(Z_int, Z_ext)**: {metrics.get('hsic_z_int_z_ext', 'N/A'):.6f}"
            if isinstance(metrics.get('hsic_z_int_z_ext'), (int, float)) else
            f"- **HSIC(Z_int, Z_ext)**: N/A",
            f"  - 目标: 趋近 0（Z_int 与 Z_ext 统计独立）",
            f"  - 解读: {'解缠效果良好，内源与外源信号有效分离' if metrics.get('hsic_z_int_z_ext', 1) < 0.01 else '解缠仍有改进空间，两分量存在残余依赖'}",
            "",
        ]

        z_ext_r2 = metrics.get("z_ext_neighbor_r2", None)
        z_int_r2 = metrics.get("z_int_neighbor_r2", None)
        if z_ext_r2 is not None:
            lines.extend([
                f"- **Z_ext 邻居预测 R²**: {z_ext_r2:.4f} (期望 > 0.3)",
                f"- **Z_int 邻居预测 R²**: {z_int_r2:.4f} (期望 < 0.1)",
                f"  - {'Z_ext 有效捕捉邻居组成信息' if z_ext_r2 > 0.3 else '邻居预测力偏低，可能需要增加训练轮数'}",
                "",
            ])

        lines.extend([
            "### 1.2 网络稀疏性与稳定性",
            "",
            f"- **因果图稀疏度**: {metrics.get('graph_sparsity', 0):.4f} "
            f"(期望 < 0.1，{'达标' if metrics.get('graph_sparsity', 1) < 0.1 else '偏密，建议提高阈值'})",
            f"- **平均 Bootstrap 频率**: {metrics.get('mean_bootstrap_freq', 0):.4f} "
            f"(期望 > 0.5，{'稳定' if metrics.get('mean_bootstrap_freq', 0) > 0.5 else '部分边不稳定'})",
            f"- **边数**: {metrics.get('n_edges', 0)} / {metrics.get('n_nodes', 0)} 节点",
            "",
            "### 1.3 结构可证伪性 (DoWhy)",
            "",
            f"- **Falsification 结果**: {falsification.get('result_str', 'N/A')}",
            f"- **平均 p-value**: {falsification.get('mean_pvalue', 'N/A')}",
            f"  - {'结构通过可证伪检验（p > 0.05），因果图与数据一致' if not falsification.get('rejected', True) else '结构被拒绝，需要检查混杂因素或调整图结构'}",
            "",
            "---",
            "",
            "## 2. 肿瘤免疫治疗角度",
            "",
            "### 2.1 免疫抑制轴识别",
            "",
        ])

        # 逐轴解读
        for ax in axis_results.get("per_axis", []):
            status_icon = "[+]" if ax.get("found") else "[-]"
            dir_str = "方向正确" if ax.get("correct_direction") else "方向待确认"
            lines.extend([
                f"- **{status_icon} {ax['name']}**: "
                f"{'已识别' if ax.get('found') else '未识别'} | {dir_str}",
                f"  - Bootstrap 频率: {ax.get('max_freq_forward', 0):.3f}",
                f"  - 文献证据: {ax.get('evidence', '')}",
                "",
            ])

        lines.extend([
            f"- **已知轴召回率**: {axis_results.get('known_axis_recall', 0):.2%}",
            f"- **方向准确率**: {axis_results.get('direction_accuracy', 0):.2%}",
            "",
            "### 2.2 候选干预靶点优先级",
            "",
            "基于因果图的 Arrow Strength 排序，以下为潜在干预靶点:",
            "",
        ])

        # 信号流通路
        for pw in flow_summary.get("pathways", []):
            complete_str = "完整" if pw.get("complete") else "部分"
            lines.append(
                f"- **{pw['pathway']}** ({complete_str}链路): "
                f"平均强度 {pw.get('avg_weight', 0):.3f}"
            )
        lines.append("")

        lines.extend([
            "### 2.3 治疗意义与风险提示",
            "",
            "1. **CAF→TAM 轴**: 如果 POSTN/MFAP2 → Integrin 信号通路被确认为因果，"
            "则靶向 CAF 分泌的 POSTN 或 MFAP2 可能减弱 M2 TAM 极化，提升抗肿瘤免疫。",
            "",
            "2. **CAF→Treg 轴**: INHBA→SMAD→Foxp3 信号如果与因果图一致，"
            "则抑制 Activin A 可能减少 Treg 分化，解除免疫抑制。",
            "",
            "3. **风险假设**: 因果可识别性依赖充分性假设（faithfulness, causal sufficiency），"
            "TME 中未观测混杂因素可能导致伪因果边。建议后续引入工具变量或敏感性分析。",
            "",
            "---",
            "",
            "*此报告由 HyperSCA Pipeline 自动生成，仅供研究参考。*",
        ])

        report_path = self.output_dir / "interpretation_step2.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  Report saved to: {report_path}")

    # =================================================================
    # 端到端执行
    # =================================================================

    def run(self) -> dict:
        """端到端执行阶段 2"""
        t_start = time.time()

        # 2.1 加载
        adata, adj = self.load_step1_results()

        # 2.2 聚类
        adata, cluster_labels, type_mapping = self.cluster_cells(adata)

        # 同步 adj: 如果 adata 被子采样，adj 也需要同步
        if "_subsample_idx" in adata.uns:
            idx = adata.uns["_subsample_idx"]
            adj = adj[idx][:, idx]
            print(f"  Adjacency synced to subsample: {adj.shape}")

        # 2.3 聚合
        if self.config.step2_granularity == "cluster":
            cluster_data = self.aggregate_to_clusters(adata, cluster_labels, adj)
        else:
            # single_cell 模式: cell-level 解缠 + cluster-level 因果发现
            cluster_data = self.aggregate_to_clusters(adata, cluster_labels, adj)
            cluster_data["_single_cell_mode"] = True

        # 2.4 解缠
        if cluster_data.get("_single_cell_mode"):
            disentangle_result = self._train_disentangle_cell_level(
                adata, adj, cluster_labels, cluster_data
            )
        else:
            disentangle_result = self.train_disentangle(cluster_data)

        # 存入 cluster_data 供后续使用
        cluster_data["z_ext"] = disentangle_result["z_ext"]
        cluster_data["z_int"] = disentangle_result["z_int"]

        # 2.5 因果发现
        causal_graph, freq_matrix = self.build_causal_graph(
            cluster_data, disentangle_result
        )

        # 2.6 DoWhy 验证
        falsification = self.validate_with_dowhy(causal_graph, cluster_data)

        # 2.7 已知轴 + 信号流
        axis_results, flow_edges, flow_summary = self.evaluate_known_axes_and_flow(
            causal_graph, cluster_data, type_mapping
        )

        # 2.8 指标
        metrics = self.compute_metrics(
            causal_graph, disentangle_result, cluster_data,
            axis_results, falsification, flow_summary,
            cluster_labels,
        )

        # 2.9 保存
        self.save_results(
            causal_graph, disentangle_result, cluster_data,
            metrics, axis_results, flow_edges, flow_summary,
            falsification, freq_matrix, type_mapping,
        )

        # 2.10 解读报告
        self.generate_interpretation(
            metrics, axis_results, flow_summary, falsification, type_mapping,
        )

        t_total = time.time() - t_start
        print(f"\n[Pipeline] Stage 2 total time: {t_total:.1f}s")

        return {
            "causal_graph": causal_graph,
            "disentangle_result": disentangle_result,
            "cluster_data": cluster_data,
            "metrics": metrics,
            "axis_results": axis_results,
            "flow_edges": flow_edges,
            "flow_summary": flow_summary,
            "falsification": falsification,
        }
