"""阶段 1 流水线: 双曲流形嵌入

EmbeddingPipeline 端到端执行:
1. 加载 .h5ad → 预处理
2. 构建空间图 + TopoLa
3. 训练 H-VAE
4. 保存嵌入、模型、指标到 results/step1/
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import sparse

from src.pipeline.config import HyperSCAConfig


class EmbeddingPipeline:
    """阶段 1: 双曲嵌入流水线"""

    def __init__(self, config: HyperSCAConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    def load_data(self):
        """加载并预处理数据"""
        from src.data.loaders import load_h5ad
        from src.data.preprocessing import preprocess

        print("=" * 60)
        print("[Step 1.1] Loading data...")
        print("=" * 60)

        adata = load_h5ad(
            Path(self.config.data_dir),
            modality=self.config.modality,
            filename=self.config.h5ad_filename,
        )
        print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

        # 对 Visium: 只保留 in_tissue spots
        if self.config.modality == "visium" and "in_tissue" in adata.obs.columns:
            n_before = adata.n_obs
            adata = adata[adata.obs["in_tissue"] == 1].copy()
            print(f"  in_tissue filter: {n_before} -> {adata.n_obs}")

        force_genes = getattr(self.config, "step3_target_genes", None)
        adata = preprocess(
            adata,
            min_cells=self.config.min_cells,
            min_genes=self.config.min_genes,
            max_genes=self.config.max_genes,
            max_pct_mt=self.config.max_pct_mt,
            target_sum=self.config.target_sum,
            n_top_genes=self.config.n_top_genes,
            hvg_flavor=self.config.hvg_flavor,
            force_include_genes=force_genes,
        )

        return adata

    def build_graph(self, adata):
        """构建空间图 + TopoLa 增强"""
        from src.data.spatial_graph import build_spatial_graph

        print()
        print("=" * 60)
        print("[Step 1.2] Building spatial graph...")
        print("=" * 60)

        coords = adata.obsm["spatial"]
        adj_orig, adj_enh = build_spatial_graph(
            coords,
            method=self.config.spatial_method,
            k=self.config.spatial_k,
            use_topola=self.config.use_topola,
            topola_lambda=self.config.topola_lambda,
            topola_components=self.config.topola_components,
            topola_max_nodes=self.config.topola_max_nodes,
        )

        return adj_orig, adj_enh

    def _sparse_to_edge_index(self, adj: sparse.csr_matrix):
        """稀疏矩阵 → edge_index + edge_weight (torch)"""
        coo = adj.tocoo()
        edge_index = torch.tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long
        )
        edge_weight = torch.tensor(coo.data, dtype=torch.float32)
        return edge_index, edge_weight

    def train_hvae(self, adata, adj_orig, adj_enh):
        """训练 H-VAE

        Returns
        -------
        dict with embeddings, model, losses
        """
        from src.models.hyperbolic.hvae import HyperbolicVAE

        print()
        print("=" * 60)
        print("[Step 1.3] Training H-VAE...")
        print("=" * 60)

        cfg = self.config

        # 准备数据
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            x_input = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            x_input = torch.tensor(np.array(adata.X), dtype=torch.float32)

        # 原始计数用于 NB 重建损失
        if adata.raw is not None:
            raw = adata.raw.to_adata()
            # 取 HVG 子集
            common_genes = adata.var_names.intersection(raw.var_names)
            raw_sub = raw[:, common_genes]
            if sp.issparse(raw_sub.X):
                x_raw = torch.tensor(raw_sub.X.toarray(), dtype=torch.float32)
            else:
                x_raw = torch.tensor(np.array(raw_sub.X), dtype=torch.float32)
        else:
            x_raw = x_input.clone()
        # 稳定性保护：极端计数会放大 NB 对数似然梯度
        x_raw = torch.clamp(x_raw, min=0.0, max=1e4)

        # 图 → edge_index
        adj_for_gcn = adj_enh if adj_enh is not None else adj_orig
        edge_index, edge_weight = self._sparse_to_edge_index(adj_for_gcn)

        # 移到设备
        x_input = x_input.to(self.device)
        x_raw = x_raw.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)

        N, G = x_input.shape
        print(f"  Input: {N} cells x {G} genes, device={self.device}")
        print(f"  Edges: {edge_index.shape[1]}")

        # 初始化模型
        model = HyperbolicVAE(
            input_dim=G,
            latent_dim=cfg.hvae_latent_dim,
            encoder_layers=cfg.hvae_encoder_layers,
            decoder_layers=cfg.hvae_decoder_layers,
            gcn_layers=cfg.hvae_gcn_layers,
            beta=cfg.hvae_beta,
            gamma=cfg.hvae_gamma,
            use_zinb=cfg.hvae_use_zinb,
            dropout=cfg.hvae_dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hvae_lr)

        # 训练记录
        losses = {"total": [], "recon": [], "kl": [], "topo": []}
        total_epochs = cfg.hvae_pretrain_epochs + cfg.hvae_epochs

        print(f"  Pretrain: {cfg.hvae_pretrain_epochs} epochs (beta=0, gamma=0)")
        print(f"  Train: {cfg.hvae_epochs} epochs (beta={cfg.hvae_beta}, gamma={cfg.hvae_gamma})")

        t0 = time.time()

        for epoch in range(total_epochs):
            model.train()

            # 预训练阶段: 只用重建损失
            is_pretrain = epoch < cfg.hvae_pretrain_epochs
            if is_pretrain:
                saved_beta, saved_gamma = model.beta, model.gamma
                model.beta, model.gamma = 0.0, 0.0

            out = model(x_input, edge_index, edge_weight)
            loss_dict = model.loss_function(
                x_raw, out, edge_index, N,
                kl_samples=cfg.hvae_kl_samples,
            )

            optimizer.zero_grad()
            loss_val = loss_dict["total"]
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"  [WARN] NaN/Inf loss at epoch {epoch+1}, skipping update")
                continue
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if is_pretrain:
                model.beta, model.gamma = saved_beta, saved_gamma

            # 记录
            for k in losses:
                losses[k].append(float(loss_dict[k]))

            if (epoch + 1) % 50 == 0 or epoch == 0:
                phase = "pretrain" if is_pretrain else "train"
                print(
                    f"  [{phase}] epoch {epoch+1}/{total_epochs}: "
                    f"total={losses['total'][-1]:.2f} "
                    f"recon={losses['recon'][-1]:.2f} "
                    f"kl={losses['kl'][-1]:.2f} "
                    f"topo={losses['topo'][-1]:.4f}"
                )

        elapsed = time.time() - t0
        print(f"  Training completed in {elapsed:.1f}s")

        # 提取嵌入
        poincare_emb, lorentz_emb = model.get_embeddings(
            x_input, edge_index, edge_weight
        )

        return {
            "poincare_emb": poincare_emb,
            "lorentz_emb": lorentz_emb,
            "model": model,
            "losses": losses,
        }

    def save_results(self, adata, adj_orig, adj_enh, train_results):
        """保存所有产物"""
        print()
        print("=" * 60)
        print("[Step 1.4] Saving results...")
        print("=" * 60)

        out = self.output_dir

        # 嵌入
        np.save(out / "embeddings_poincare.npy", train_results["poincare_emb"])
        np.save(out / "embeddings_lorentz.npy", train_results["lorentz_emb"])
        print(f"  Saved embeddings: {train_results['poincare_emb'].shape}")

        # 邻接矩阵
        sparse.save_npz(out / "adj_original.npz", adj_orig)
        if adj_enh is not None:
            sparse.save_npz(out / "adj_enhanced.npz", adj_enh)

        # 模型
        torch.save(
            train_results["model"].state_dict(),
            out / "hvae_model.pt",
        )

        # 训练损失
        with open(out / "training_losses.json", "w") as f:
            json.dump(train_results["losses"], f, indent=2)

        # 配置
        with open(out / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 预处理后的 AnnData（附加嵌入）
        adata.obsm["X_poincare"] = train_results["poincare_emb"]
        adata.obsm["X_lorentz"] = train_results["lorentz_emb"]
        adata.write(out / "adata_embedded.h5ad")

        # 嵌入质量基准（Hyperbolic vs UMAP）
        try:
            from src.evaluation.embedding_metrics import evaluate_embedding
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import LabelEncoder
            import scanpy as sc

            label_col = None
            for c in ["cell_type", "Level1", "celltype", "annotation", "leiden"]:
                if c in adata.obs.columns:
                    label_col = c
                    break
            metrics = {"hyperbolic": {}, "umap": {}, "label_col": label_col}
            if label_col is not None:
                y = LabelEncoder().fit_transform(adata.obs[label_col].astype(str).values)
                metrics["hyperbolic"] = evaluate_embedding(
                    train_results["poincare_emb"],
                    labels=y,
                    dist_original=None,
                )
                adata_eval = adata.copy()
                sc.tl.pca(adata_eval, n_comps=50)
                sc.pp.neighbors(adata_eval, n_pcs=50)
                sc.tl.umap(adata_eval)
                umap_emb = adata_eval.obsm["X_umap"]
                metrics["umap"] = {
                    "n_cells": int(umap_emb.shape[0]),
                    "embedding_dim": int(umap_emb.shape[1]),
                    "silhouette": float(silhouette_score(umap_emb, y)),
                }
            with open(out / "embedding_benchmark.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"  [WARN] embedding benchmark skipped: {exc}")

        print(f"  All results saved to: {out}")

    def run(self) -> dict:
        """端到端执行"""
        t_start = time.time()

        # 1. 加载数据
        adata = self.load_data()

        # 2. 构建空间图
        adj_orig, adj_enh = self.build_graph(adata)

        # 3. 训练 H-VAE
        train_results = self.train_hvae(adata, adj_orig, adj_enh)

        # 4. 保存
        self.save_results(adata, adj_orig, adj_enh, train_results)

        t_total = time.time() - t_start
        print(f"\n[Pipeline] Total time: {t_total:.1f}s")

        return {
            "adata": adata,
            "adj_orig": adj_orig,
            "adj_enh": adj_enh,
            **train_results,
        }
