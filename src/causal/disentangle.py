"""因果解缠模型 (Causal Disentanglement)

参考 Celcomen 算法，将细胞潜变量分解为:
- z_int: 内源性状态（细胞自身转录调控）
- z_ext: 外源性影响（空间邻居通过配受体相互作用的因果决定）

结构因果模型 (SCM):
    z_ext_i = g({z_j}_{j in N(i)}, epsilon_i)     # GCN 消息传递
    z_int_i = h(x_i)                                # MLP 编码
    x_hat_i = decoder(z_int_i, z_ext_i)             # 联合解码

训练目标:
    L = L_recon(X | z_int, z_ext) + alpha * HSIC(z_int, z_ext)

参考实现（adapter 模式，不直接 import）:
    references/celcomen/celcomen/models/celcomen.py
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# =========================================================================
# HSIC (Hilbert-Schmidt Independence Criterion)
# =========================================================================

def rbf_kernel_matrix(X: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF 核矩阵 K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2*sigma^2))"""
    sq_dists = torch.cdist(X, X, p=2).pow(2)
    return torch.exp(-sq_dists / (2 * sigma ** 2 + 1e-10))


def compute_hsic_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> torch.Tensor:
    """可微分 HSIC（用于训练损失）

    Parameters
    ----------
    X : (N, d1) 第一组变量
    Y : (N, d2) 第二组变量
    sigma_x, sigma_y : RBF 核带宽（None 则用 median heuristic）

    Returns
    -------
    HSIC 标量
    """
    N = X.shape[0]

    if sigma_x is None:
        with torch.no_grad():
            dists_x = torch.cdist(X, X, p=2)
            sigma_x = float(torch.median(dists_x[dists_x > 0]).item()) + 1e-5
    if sigma_y is None:
        with torch.no_grad():
            dists_y = torch.cdist(Y, Y, p=2)
            sigma_y = float(torch.median(dists_y[dists_y > 0]).item()) + 1e-5

    K = rbf_kernel_matrix(X, sigma_x)
    L = rbf_kernel_matrix(Y, sigma_y)

    # 中心化: H = I - 1/N
    H = torch.eye(N, device=X.device) - 1.0 / N

    # HSIC = tr(KHLH) / (N-1)^2
    hsic = torch.trace(K @ H @ L @ H) / ((N - 1) ** 2 + 1e-10)
    return hsic


# =========================================================================
# 解缠模型
# =========================================================================

class DisentangleModel(nn.Module):
    """因果解缠模型

    双分支架构:
        - 内源性分支 (MLP): x → z_int
        - 外源性分支 (GCN): x + graph → z_ext
        - 联合解码器: (z_int, z_ext) → x_hat

    Parameters
    ----------
    input_dim : int
        输入特征维度（基因数或嵌入维度）
    z_dim : int
        z_int 和 z_ext 各自的维度
    hidden_dims : list[int]
        隐藏层维度
    gcn_layers : int
        GCN 分支层数
    dropout : float
        Dropout 率
    """

    def __init__(
        self,
        input_dim: int,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
        gcn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.z_dim = z_dim

        # --- 内源性分支 (MLP): x → z_int ---
        int_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            int_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        int_layers.append(nn.Linear(prev_dim, z_dim))
        self.intrinsic_encoder = nn.Sequential(*int_layers)

        # --- 外源性分支 (GCN): x + graph → z_ext ---
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        prev_dim = input_dim
        gcn_hidden = hidden_dims[0] if hidden_dims else 128
        for i in range(gcn_layers):
            out_dim = gcn_hidden if i < gcn_layers - 1 else z_dim
            self.gcn_convs.append(GCNConv(prev_dim, out_dim, add_self_loops=False))
            if i < gcn_layers - 1:
                self.gcn_bns.append(nn.BatchNorm1d(out_dim))
            prev_dim = out_dim
        self.gcn_dropout = nn.Dropout(dropout)

        # --- 联合解码器: (z_int, z_ext) → x_hat ---
        dec_layers = []
        prev_dim = z_dim * 2  # concat z_int + z_ext
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        dec_layers.append(nn.Softplus())  # 非负输出
        self.decoder = nn.Sequential(*dec_layers)

    def encode_intrinsic(self, x: torch.Tensor) -> torch.Tensor:
        """编码内源性分量"""
        return self.intrinsic_encoder(x)

    def encode_extrinsic(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码外源性分量（GCN 消息传递）"""
        h = x
        for i, conv in enumerate(self.gcn_convs):
            h = conv(h, edge_index, edge_weight)
            if i < len(self.gcn_bns):
                h = self.gcn_bns[i](h)
                h = F.relu(h)
                h = self.gcn_dropout(h)
        return h

    def decode(self, z_int: torch.Tensor, z_ext: torch.Tensor) -> torch.Tensor:
        """联合解码"""
        z_cat = torch.cat([z_int, z_ext], dim=-1)
        return self.decoder(z_cat)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> dict:
        """前向传播

        Returns
        -------
        dict with: z_int, z_ext, x_hat
        """
        z_int = self.encode_intrinsic(x)
        z_ext = self.encode_extrinsic(x, edge_index, edge_weight)
        x_hat = self.decode(z_int, z_ext)
        return {"z_int": z_int, "z_ext": z_ext, "x_hat": x_hat}

    @torch.no_grad()
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """提取嵌入"""
        self.eval()
        z_int = self.encode_intrinsic(x).cpu().numpy()
        z_ext = self.encode_extrinsic(x, edge_index, edge_weight).cpu().numpy()
        return z_int, z_ext


# =========================================================================
# 训练函数
# =========================================================================

def train_disentangle(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    z_dim: int = 16,
    hidden_dims: list[int] | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    hsic_alpha: float = 1.0,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """训练解缠模型

    Parameters
    ----------
    x : (N, G) 表达矩阵
    edge_index : (2, E) 空间图边
    edge_weight : (E,) 边权重
    z_dim : z_int / z_ext 维度
    hidden_dims : 隐藏层维度
    epochs : 训练轮数
    lr : 学习率
    hsic_alpha : HSIC 惩罚权重
    device : 设备
    verbose : 是否打印训练进度

    Returns
    -------
    dict with: model, z_int, z_ext, losses
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = DisentangleModel(
        input_dim=x.shape[1],
        z_dim=z_dim,
        hidden_dims=hidden_dims,
    ).to(dev)

    x = x.to(dev)
    edge_index = edge_index.to(dev)
    if edge_weight is not None:
        edge_weight = edge_weight.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = {"total": [], "recon": [], "hsic": []}

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        out = model(x, edge_index, edge_weight)

        # 重建损失 (MSE on log1p-normalized expression)
        recon_loss = F.mse_loss(out["x_hat"], x)

        # HSIC 独立性惩罚
        hsic_loss = compute_hsic_torch(out["z_int"], out["z_ext"])

        total_loss = recon_loss + hsic_alpha * hsic_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses["total"].append(float(total_loss))
        losses["recon"].append(float(recon_loss))
        losses["hsic"].append(float(hsic_loss))

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(
                f"  [disentangle] epoch {epoch+1}/{epochs}: "
                f"total={losses['total'][-1]:.4f} "
                f"recon={losses['recon'][-1]:.4f} "
                f"hsic={losses['hsic'][-1]:.6f}"
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"  Disentangle training completed in {elapsed:.1f}s")

    # 提取嵌入
    z_int, z_ext = model.get_embeddings(x, edge_index, edge_weight)

    return {
        "model": model,
        "z_int": z_int,
        "z_ext": z_ext,
        "losses": losses,
    }
