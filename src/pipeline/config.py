"""HyperSCA 全局配置"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class HyperSCAConfig:
    """全局超参数配置

    可通过 YAML 文件加载或直接实例化。
    """

    # --- Data ---
    data_dir: str = "data/Visium_HumanColon_Oliveira"
    modality: str = "visium"                # chromium / visium / xenium
    h5ad_filename: str = "expression.h5ad"

    # --- Preprocessing ---
    min_cells: int = 10
    min_genes: int = 200
    max_genes: int = 8000
    max_pct_mt: float = 20.0
    target_sum: float = 1e4
    n_top_genes: int = 3000
    hvg_flavor: str = "seurat"

    # --- Spatial Graph ---
    spatial_k: int = 6
    spatial_method: str = "knn"             # knn / delaunay
    use_topola: bool = True
    topola_lambda: float = 1e-3
    topola_components: Optional[int] = None

    # --- H-VAE ---
    hvae_latent_dim: int = 32
    hvae_encoder_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    hvae_decoder_layers: List[int] = field(default_factory=lambda: [128, 256, 512])
    hvae_gcn_layers: int = 2
    hvae_beta: float = 1.0                  # KL weight
    hvae_gamma: float = 10.0                # Topo regularization weight
    hvae_lr: float = 1e-3
    hvae_epochs: int = 300
    hvae_pretrain_epochs: int = 50          # MLP-only pretrain
    hvae_batch_size: int = 0                # 0 = full batch
    hvae_use_zinb: bool = False
    hvae_dropout: float = 0.1
    hvae_kl_samples: int = 5

    # --- Step 2: Causal Disentanglement ---
    step2_input_dir: str = "results/step1"
    step2_output_dir: str = "results/step2"
    step2_granularity: str = "cluster"          # cluster / single_cell
    step2_n_clusters: Optional[int] = None      # None = Leiden 自动决定
    step2_leiden_resolution: float = 1.0
    step2_disentangle_dim: int = 16             # z_int / z_ext 各自维度
    step2_disentangle_hidden: List[int] = field(default_factory=lambda: [256, 128])
    step2_disentangle_epochs: int = 200
    step2_disentangle_lr: float = 1e-3
    step2_hsic_alpha: float = 1.0               # HSIC 惩罚权重
    step2_bootstrap_n: int = 100                # Bootstrap 重复次数
    step2_bootstrap_threshold: float = 0.5      # 边频率保留阈值
    step2_cmi_alpha: float = 0.05               # CMI 条件独立性检验 α
    step2_pc_max_cond: int = 3                  # PC 算法最大条件集大小
    step2_max_cells: int = 5000                 # single_cell 模式子采样上限
    step2_known_axes_file: Optional[str] = None # 外部先验边集 JSON（可选）

    # --- Step 3: Counterfactual Perturbation ---
    step3_input_step1_dir: str = "results/step1"
    step3_input_step2_dir: str = "results/step2"
    step3_output_dir: str = "results/step3"
    step3_figures_dir: str = "results/figures/step3"
    step3_target_genes: List[str] = field(default_factory=lambda: ["INHBA", "POSTN", "MFAP2"])
    step3_method: str = "latent_arithmetic"      # latent_arithmetic / diffusion_cf
    step3_intervention_value: float = 0.0
    step3_latent_ko_scale: float = 0.5
    step3_enable_target_ranking: bool = True
    step3_target_top_k: int = 30
    step3_diffusion_steps: int = 40
    step3_diffusion_epochs: int = 10
    step3_diffusion_hidden: int = 256
    step3_spatial_decay_length: float = 150.0
    step3_propagation_max_depth: int = 4
    step3_propagation_threshold: float = 0.01

    # --- General ---
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "results/step1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HyperSCAConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """转为字典"""
        from dataclasses import asdict
        return asdict(self)
