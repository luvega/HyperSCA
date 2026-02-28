"""Hyperbolic VAE (H-VAE) 主模型

双分支编码器（GCN + MLP）+ NB/ZINB 解码器，在 Lorentz 流形上训练。

架构:
    Branch 1: GCN (torch_geometric.nn.GCNConv) on Ã — 提取空间拓扑
    Branch 2: MLP on expression X — 提取表达特征
    拼接 → Linear → polar_project → Lorentz loc + scale → WrappedNormal
    解码器: Lorentz z → LogMap → MLP → NB(mean, disp) [+ pi for ZINB]

损失:
    L = recon_loss(NB) + beta * KL(WN || WN_prior) + gamma * topo_reg

Adapter 模式: 参考 references/scDHMap/src/scDHMap/scDHMap.py,
              禁止直接 import，完全重写。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.models.hyperbolic.lorentz import (
    polar_project,
    lorentz_to_poincare,
    lorentz_origin,
    lorentzian_inner,
    lorentz_distance,
    project_to_lorentz,
)
from src.models.hyperbolic.wrapped_normal import WrappedNormal


# =========================================================================
# 辅助层
# =========================================================================

class MeanAct(nn.Module):
    """Softplus 激活 (NB mean)，带 clamp"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Softplus 激活 (NB dispersion)，带 clamp"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


def build_mlp(
    layer_dims: list[int],
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = True,
    last_activation: bool = False,
) -> nn.Sequential:
    """构建 MLP 网络"""
    act_fn = {"relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
    act_class = act_fn.get(activation, nn.ReLU)

    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        is_last = i == len(layer_dims) - 2
        if not is_last or last_activation:
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.append(act_class())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


# =========================================================================
# 双分支编码器
# =========================================================================

class DualBranchEncoder(nn.Module):
    """GCN + MLP 双分支编码器

    Branch 1 (GCN): edge_index + edge_weight → GCN layers → h_graph
    Branch 2 (MLP): expression X → MLP layers → h_expr
    拼接 → Linear → (mu_euclidean, log_sigma)
    mu_euclidean → polar_project → Lorentz mu
    log_sigma → Softplus → sigma (tangent space std)

    Parameters
    ----------
    input_dim : int
        基因数 G
    latent_dim : int
        双曲潜空间维度 d（Lorentz 维度为 d+1）
    hidden_dims : list[int]
        隐藏层维度列表
    gcn_layers : int
        GCN 层数
    dropout : float
        Dropout 率
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        gcn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Branch 1: GCN
        gcn_dims = [input_dim] + hidden_dims[:gcn_layers]
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        for i in range(len(gcn_dims) - 1):
            self.gcn_convs.append(GCNConv(gcn_dims[i], gcn_dims[i + 1]))
            self.gcn_bns.append(nn.BatchNorm1d(gcn_dims[i + 1]))
        self.gcn_out_dim = gcn_dims[-1]

        # Branch 2: MLP
        mlp_dims = [input_dim] + hidden_dims
        self.expr_mlp = build_mlp(mlp_dims, dropout=dropout)
        self.mlp_out_dim = hidden_dims[-1]

        # 融合层
        fused_dim = self.gcn_out_dim + self.mlp_out_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
        )

        # 输出头
        self.enc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.enc_log_sigma = nn.Linear(hidden_dims[-1], latent_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (N, G) 表达矩阵
        edge_index : (2, E) 边索引
        edge_weight : (E,) 边权重

        Returns
        -------
        mu : (N, d+1) Lorentz 流形上的均值
        sigma : (N, d) 切空间标准差
        """
        # Branch 1: GCN
        h_gcn = x
        for conv, bn in zip(self.gcn_convs, self.gcn_bns):
            h_gcn = conv(h_gcn, edge_index, edge_weight)
            h_gcn = bn(h_gcn)
            h_gcn = F.relu(h_gcn)
            h_gcn = self.dropout(h_gcn)

        # Branch 2: MLP
        h_mlp = self.expr_mlp(x)

        # 融合
        h_fused = torch.cat([h_gcn, h_mlp], dim=-1)
        h_fused = self.fuse(h_fused)

        # 参数化
        mu_euclidean = self.enc_mu(h_fused)      # (N, d)
        log_sigma = self.enc_log_sigma(h_fused)   # (N, d)

        # 极坐标投影到 Lorentz
        mu_lorentz = polar_project(mu_euclidean)   # (N, d+1)

        # Softplus + clamp 得到标准差
        sigma = torch.clamp(F.softplus(log_sigma), min=1e-6, max=15.0)

        return mu_lorentz, sigma


# =========================================================================
# NB 解码器
# =========================================================================

class NBDecoder(nn.Module):
    """负二项分布解码器

    输入 Lorentz z (d+1 维) → MLP → NB(mean, disp) [+ pi for ZINB]

    Parameters
    ----------
    latent_dim : int
        双曲维度 d（输入维度为 d+1）
    output_dim : int
        基因数 G
    hidden_dims : list[int]
        解码器隐藏层
    use_zinb : bool
        是否使用 Zero-Inflated NB
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [128, 256, 512],
        use_zinb: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_zinb = use_zinb

        # 主干: Lorentz z (d+1) → hidden
        dec_dims = [latent_dim + 1] + hidden_dims
        self.decoder = build_mlp(dec_dims, dropout=dropout)

        # 输出头
        self.dec_mean = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            MeanAct(),
        )
        self.dec_disp = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            DispAct(),
        )
        if use_zinb:
            self.dec_pi = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim),
                nn.Sigmoid(),
            )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        z : (N, d+1) Lorentz 坐标

        Returns
        -------
        mean : (N, G) NB 均值
        disp : (N, G) NB 逆离散参数
        pi : (N, G) or None — ZINB dropout 概率
        """
        h = self.decoder(z)
        mean = self.dec_mean(h)
        disp = self.dec_disp(h)
        pi = self.dec_pi(h) if self.use_zinb else None
        return mean, disp, pi


# =========================================================================
# NB / ZINB 损失
# =========================================================================

def nb_loss(
    x: torch.Tensor,
    mean: torch.Tensor,
    disp: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """负二项分布负对数似然

    NB(x; mu, theta) = Gamma(x+theta) / (Gamma(x+1)*Gamma(theta))
                        * (theta/(theta+mu))^theta * (mu/(theta+mu))^x

    Parameters
    ----------
    x : (N, G) 原始计数
    mean : (N, G) NB 均值 mu
    disp : (N, G) 逆离散参数 theta

    Returns
    -------
    (N,) 每个细胞的平均负对数似然
    """
    log_theta_mu_eps = torch.log(disp + mean + eps)
    nll = (
        torch.lgamma(x + disp)
        - torch.lgamma(x + 1.0)
        - torch.lgamma(disp)
        + disp * (torch.log(disp + eps) - log_theta_mu_eps)
        + x * (torch.log(mean + eps) - log_theta_mu_eps)
    )
    return -nll.sum(dim=-1)  # (N,)


def zinb_loss(
    x: torch.Tensor,
    mean: torch.Tensor,
    disp: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Zero-Inflated NB 负对数似然"""
    # NB 部分
    softplus_pi = F.softplus(-pi)  # log(sigmoid(-pi)) = -softplus(pi)
    log_theta_mu = torch.log(disp + mean + eps)

    nb_case = (
        torch.lgamma(x + disp)
        - torch.lgamma(x + 1.0)
        - torch.lgamma(disp)
        + disp * (torch.log(disp + eps) - log_theta_mu)
        + x * (torch.log(mean + eps) - log_theta_mu)
        - softplus_pi
    )

    # Zero 部分
    zero_nb = disp * (torch.log(disp + eps) - log_theta_mu) - softplus_pi
    zero_case = -F.softplus(-zero_nb + torch.log(torch.exp(F.softplus(-pi)) - 1 + eps))

    # 根据 x 是否为零选择
    is_zero = (x < 1e-8).float()
    nll = is_zero * zero_case + (1 - is_zero) * nb_case

    return -nll.sum(dim=-1)  # (N,)


# =========================================================================
# 拓扑正则化
# =========================================================================

def topo_regularization(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
    t: float = 1.0,
) -> torch.Tensor:
    """Cauchy 核拓扑正则化（类 t-SNE 吸引/斥力）

    在邻接矩阵连接的节点之间施加 Cauchy 核吸引力，
    在所有节点对之间施加 Cauchy 核斥力。

    Parameters
    ----------
    z : (N, d+1) Lorentz 嵌入
    edge_index : (2, E)
    n_nodes : int
    t : float
        Cauchy 核自由度

    Returns
    -------
    scalar 正则化损失
    """
    # 吸引力: 连接的节点之间
    src, tgt = edge_index[0], edge_index[1]
    d_connected = lorentz_distance(z[src], z[tgt])
    attract = torch.log(1 + d_connected ** 2 / t).mean()

    # 斥力: 随机采样节点对
    n_repel = min(edge_index.shape[1], n_nodes * 5)
    idx_a = torch.randint(0, n_nodes, (n_repel,), device=z.device)
    idx_b = torch.randint(0, n_nodes, (n_repel,), device=z.device)
    d_random = lorentz_distance(z[idx_a], z[idx_b])
    repel = -torch.log(1 + d_random ** 2 / t).mean()

    return attract + repel


# =========================================================================
# H-VAE 主模型
# =========================================================================

class HyperbolicVAE(nn.Module):
    """Hyperbolic VAE 主模型

    Parameters
    ----------
    input_dim : int
        基因数 G
    latent_dim : int
        双曲潜空间维度 d
    encoder_layers : list[int]
        编码器隐藏层
    decoder_layers : list[int]
        解码器隐藏层
    gcn_layers : int
        GCN 分支层数
    beta : float
        KL 散度权重
    gamma : float
        拓扑正则化权重
    use_zinb : bool
        是否使用 ZINB（默认 NB）
    dropout : float
        Dropout 率
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_layers: list[int] = [512, 256, 128],
        decoder_layers: list[int] = [128, 256, 512],
        gcn_layers: int = 2,
        beta: float = 1.0,
        gamma: float = 10.0,
        use_zinb: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.use_zinb = use_zinb

        self.encoder = DualBranchEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_layers,
            gcn_layers=gcn_layers,
            dropout=dropout,
        )

        self.decoder = NBDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_layers,
            use_zinb=use_zinb,
            dropout=dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[WrappedNormal, torch.Tensor]:
        """编码

        Returns
        -------
        q_z : WrappedNormal 后验分布
        mu : (N, d+1) Lorentz 均值
        """
        mu, sigma = self.encoder(x, edge_index, edge_weight)
        q_z = WrappedNormal(mu, sigma)
        return q_z, mu

    def decode(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """解码

        Returns
        -------
        mean, disp, pi
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> dict:
        """前向传播

        Parameters
        ----------
        x : (N, G) 表达矩阵（log1p 归一化后）
        edge_index : (2, E) 边索引
        edge_weight : (E,) 边权重

        Returns
        -------
        dict with keys:
            z : (N, d+1) Lorentz 嵌入
            z_poincare : (N, d) Poincare 嵌入
            mu : (N, d+1) Lorentz 均值
            q_z : WrappedNormal 后验
            recon_mean : (N, G)
            recon_disp : (N, G)
            recon_pi : (N, G) or None
        """
        q_z, mu = self.encode(x, edge_index, edge_weight)
        z = q_z.rsample()
        z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
        z = project_to_lorentz(z)

        mean, disp, pi = self.decode(z)
        mean = torch.nan_to_num(mean, nan=1e-5, posinf=1e6, neginf=1e-5)
        disp = torch.nan_to_num(disp, nan=1e-4, posinf=1e4, neginf=1e-4)
        if pi is not None:
            pi = torch.nan_to_num(pi, nan=0.5, posinf=1.0, neginf=0.0)
            pi = torch.clamp(pi, min=1e-6, max=1 - 1e-6)

        return {
            "z": z,
            "z_poincare": lorentz_to_poincare(z),
            "mu": mu,
            "q_z": q_z,
            "recon_mean": mean,
            "recon_disp": disp,
            "recon_pi": pi,
        }

    def loss_function(
        self,
        x_raw: torch.Tensor,
        forward_out: dict,
        edge_index: torch.Tensor,
        n_nodes: int,
        kl_samples: int = 5,
    ) -> dict:
        """计算损失

        Parameters
        ----------
        x_raw : (N, G) 原始计数（用于 NB 重建损失）
        forward_out : forward() 的输出 dict
        edge_index : (2, E)
        n_nodes : int
        kl_samples : int
            KL 散度 MC 采样数

        Returns
        -------
        dict with keys: total, recon, kl, topo
        """
        q_z = forward_out["q_z"]
        mean = forward_out["recon_mean"]
        disp = forward_out["recon_disp"]
        pi = forward_out["recon_pi"]
        z = forward_out["z"]

        # 重建损失
        if self.use_zinb and pi is not None:
            recon = zinb_loss(x_raw, mean, disp, pi).mean()
        else:
            recon = nb_loss(x_raw, mean, disp).mean()
        recon = torch.nan_to_num(recon, nan=1e6, posinf=1e6, neginf=1e6)

        # KL 散度
        kl = q_z.kl_divergence(n_samples=kl_samples).mean()
        kl = torch.nan_to_num(kl, nan=0.0, posinf=1e4, neginf=0.0)
        kl = torch.clamp(kl, min=0.0, max=1e4)

        # 拓扑正则化
        topo = topo_regularization(z, edge_index, n_nodes)
        topo = torch.nan_to_num(topo, nan=0.0, posinf=1e3, neginf=-1e3)
        topo = torch.clamp(topo, min=-1e3, max=1e3)

        total = recon + self.beta * kl + self.gamma * topo

        return {
            "total": total,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "topo": topo.detach(),
        }

    @torch.no_grad()
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取嵌入（eval 模式）

        Returns
        -------
        poincare_emb : (N, d) Poincare 球坐标
        lorentz_emb : (N, d+1) Lorentz 坐标
        """
        self.eval()
        mu, _ = self.encoder(x, edge_index, edge_weight)
        poincare = lorentz_to_poincare(mu).cpu().numpy()
        lorentz = mu.cpu().numpy()
        return poincare, lorentz
