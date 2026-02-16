"""阶段 3: 基于扩散的反事实最小原型（P1.5）。

目标：
- 提供可运行的条件扩散反事实接口，避免长期停留在文档层。
- 默认使用轻量 MLP 去噪器，支持小规模数据快速验证。
- 提供基础因果掩码约束：非后继基因保持接近观测值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class _TinyDenoiser(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        # t_norm: (B, 1) in [0, 1]
        return self.net(torch.cat([x_t, t_norm], dim=1))


@dataclass
class DiffusionConfig:
    n_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 256
    lr: float = 1e-3
    train_epochs: int = 10
    batch_size: int = 256


class CausalDiffusionCF:
    """最小可运行扩散反事实原型。

    Parameters
    ----------
    input_dim
        基因维度。
    causal_mask
        (G, G) 二值矩阵，causal_mask[i, j]=1 表示 i->j。
    device
        "cuda" 或 "cpu"。
    config
        扩散配置。
    """

    def __init__(
        self,
        input_dim: int,
        causal_mask: Optional[np.ndarray] = None,
        device: str = "cpu",
        config: Optional[DiffusionConfig] = None,
    ):
        self.input_dim = int(input_dim)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.config = config or DiffusionConfig()

        self.model = _TinyDenoiser(self.input_dim, self.config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # beta schedule
        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.n_steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.causal_mask = None
        if causal_mask is not None:
            arr = np.asarray(causal_mask, dtype=float)
            if arr.shape != (self.input_dim, self.input_dim):
                raise ValueError("causal_mask must have shape (input_dim, input_dim)")
            self.causal_mask = arr

    def fit(self, x: np.ndarray) -> dict[str, float]:
        """训练去噪器（噪声预测目标）。"""
        x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        n = x_t.shape[0]
        if n == 0:
            raise ValueError("Empty input data.")

        losses: list[float] = []
        self.model.train()
        for _ in range(self.config.train_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.config.batch_size):
                idx = perm[i : i + self.config.batch_size]
                xb = x_t[idx]
                bsz = xb.shape[0]
                t_idx = torch.randint(0, self.config.n_steps, (bsz,), device=self.device)
                t_norm = t_idx.float().unsqueeze(1) / max(self.config.n_steps - 1, 1)

                noise = torch.randn_like(xb)
                a_bar = self.alpha_bars[t_idx].unsqueeze(1)
                x_noisy = torch.sqrt(a_bar) * xb + torch.sqrt(1.0 - a_bar) * noise

                pred_noise = self.model(x_noisy, t_norm)
                loss = torch.mean((pred_noise - noise) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

        return {"train_loss": float(np.mean(losses)) if losses else 0.0}

    def generate_counterfactual(
        self,
        x_observed: np.ndarray,
        gene_names: list[str],
        intervention: dict[str, float],
        *,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """生成反事实表达矩阵（最小原型）。

        intervention 示例: {"POSTN": 0.0} 表示将该基因压到低表达。
        """
        if len(gene_names) != self.input_dim:
            raise ValueError("gene_names length mismatch with input_dim")

        x_obs_np = np.asarray(x_observed, dtype=np.float32)
        x_obs = torch.as_tensor(x_obs_np, dtype=torch.float32, device=self.device)
        if x_obs.shape[1] != self.input_dim:
            raise ValueError("x_observed shape mismatch with input_dim")

        gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
        intv_idx: list[int] = []
        intv_val: list[float] = []
        for g, v in intervention.items():
            key = str(g).upper()
            if key in gene_to_idx:
                intv_idx.append(gene_to_idx[key])
                intv_val.append(float(v))
        if not intv_idx:
            raise ValueError("No valid intervention genes found in gene_names.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model.eval()
        with torch.no_grad():
            x_t = x_obs + torch.randn_like(x_obs) * 0.2
            frozen_mask = self._frozen_mask(intv_idx)
            for t in reversed(range(self.config.n_steps)):
                t_norm = torch.full((x_t.shape[0], 1), t / max(self.config.n_steps - 1, 1), device=self.device)
                pred_noise = self.model(x_t, t_norm)
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                beta_t = self.betas[t]
                # DDPM reverse mean
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
                )
                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(beta_t) * noise
                else:
                    x_t = mean

                # 强制施加干预值
                for j, v in zip(intv_idx, intv_val):
                    x_t[:, j] = v

                # 因果一致性: 非后继维度回拉到观测值
                if frozen_mask is not None:
                    x_t[:, frozen_mask] = x_obs[:, frozen_mask]

        return x_t.detach().cpu().numpy()

    def _frozen_mask(self, intervention_idx: list[int]) -> Optional[np.ndarray]:
        """返回需冻结（不应变化）的基因掩码。"""
        if self.causal_mask is None:
            return None
        affected = np.zeros(self.input_dim, dtype=bool)
        for idx in intervention_idx:
            affected[idx] = True
            affected |= self._descendants(idx)
        return ~affected

    def _descendants(self, start_idx: int) -> np.ndarray:
        """在 causal_mask 上做 BFS，找后继节点。"""
        visited = np.zeros(self.input_dim, dtype=bool)
        queue = [start_idx]
        while queue:
            cur = queue.pop(0)
            children = np.where(self.causal_mask[cur] > 0)[0]
            for c in children:
                if not visited[c]:
                    visited[c] = True
                    queue.append(int(c))
        return visited

