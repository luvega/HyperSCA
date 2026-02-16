"""阶段 3: 双曲潜空间扰动算术 (Latent Space Arithmetic)。

在 Lorentz 流形上实现基因特异性 KO 扰动向量的估计与施加:
    δ_g = E_treated[Log_o(z)] - E_control[Log_o(z)]
    z_pred = Exp_{z_obs}(PT_{o→z_obs}(-δ_g))

参考思想: CPA / scGen（adapter 模式，不直接 import references/）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.models.hyperbolic.lorentz import (
    exp_map,
    log_map,
    lorentz_origin,
    parallel_transport,
    project_to_lorentz,
)


class LatentArithmetic:
    """双曲潜空间中的扰动向量估计与施加。

    Parameters
    ----------
    curvature : float
        双曲曲率 K（默认 1.0）。当前实现假设 K=1。
    device : str
        "cpu" 或 "cuda"。
    """

    def __init__(self, curvature: float = 1.0, device: str = "cpu"):
        self.curvature = curvature
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

    def compute_perturbation_vector(
        self,
        treated_z: np.ndarray,
        control_z: np.ndarray,
    ) -> np.ndarray:
        """估计扰动向量 δ_g（切空间中）。

        δ_g = mean(Log_o(z_treated)) - mean(Log_o(z_control))

        Parameters
        ----------
        treated_z : (N_t, d+1) Lorentz 坐标（处理组嵌入）
        control_z : (N_c, d+1) Lorentz 坐标（对照组嵌入）

        Returns
        -------
        (d+1,) 切向量（在原点处）
        """
        treated = torch.as_tensor(treated_z, dtype=torch.float32, device=self.device)
        control = torch.as_tensor(control_z, dtype=torch.float32, device=self.device)

        dim = treated.shape[-1] - 1
        o = lorentz_origin(dim, batch_size=1, device=self.device)  # (1, d+1)

        # 映射到原点切空间取均值
        log_t = log_map(treated, o.expand_as(treated))  # (N_t, d+1)
        log_c = log_map(control, o.expand_as(control))  # (N_c, d+1)

        delta = log_t.mean(dim=0) - log_c.mean(dim=0)  # (d+1,)
        return delta.detach().cpu().numpy()

    def apply_perturbation(
        self,
        z_obs: np.ndarray,
        delta: np.ndarray,
        *,
        direction: float = -1.0,
    ) -> np.ndarray:
        """对观测嵌入施加扰动（默认 KO = 减去扰动向量）。

        z_pred_i = Exp_{z_obs_i}(PT_{o→z_obs_i}(direction * δ))

        Parameters
        ----------
        z_obs : (N, d+1) 观测 Lorentz 嵌入
        delta : (d+1,) 切向量（在原点处）
        direction : float
            -1.0 表示 knockout（减去），+1.0 表示 overexpression（加上）

        Returns
        -------
        (N, d+1) 反事实 Lorentz 嵌入
        """
        z = torch.as_tensor(z_obs, dtype=torch.float32, device=self.device)
        d_vec = torch.as_tensor(
            delta * direction, dtype=torch.float32, device=self.device
        )

        dim = z.shape[-1] - 1
        o = lorentz_origin(dim, batch_size=1, device=self.device)

        # 将 δ 从原点平行传输到每个 z_obs_i
        d_expanded = d_vec.unsqueeze(0).expand_as(z)
        o_expanded = o.expand_as(z)
        v_at_z = parallel_transport(d_expanded, o_expanded, z)

        # 指数映射回流形
        z_pred = exp_map(v_at_z, z)
        return z_pred.detach().cpu().numpy()

    def decode_counterfactual(
        self,
        z_cf: np.ndarray,
        decoder: torch.nn.Module,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """用 H-VAE 解码器将反事实嵌入映射回基因表达空间。

        Parameters
        ----------
        z_cf : (N, d+1) 反事实 Lorentz 嵌入
        decoder : 训练好的 H-VAE 解码器模块
        edge_index, edge_weight : 图结构（如果解码器需要）

        Returns
        -------
        (N, G) 反事实基因表达矩阵
        """
        z_t = torch.as_tensor(z_cf, dtype=torch.float32, device=self.device)
        decoder = decoder.to(self.device)
        decoder.eval()
        with torch.no_grad():
            x_hat = decoder(z_t)
        return x_hat.detach().cpu().numpy()

    def virtual_knockout(
        self,
        z_obs: np.ndarray,
        treated_z: np.ndarray,
        control_z: np.ndarray,
        decoder: Optional[torch.nn.Module] = None,
    ) -> dict:
        """端到端虚拟敲除。

        Returns
        -------
        dict with keys:
            delta : (d+1,) 扰动向量
            z_cf : (N, d+1) 反事实嵌入
            x_cf : (N, G) 反事实表达（若提供 decoder）
        """
        delta = self.compute_perturbation_vector(treated_z, control_z)
        z_cf = self.apply_perturbation(z_obs, delta, direction=-1.0)
        result = {"delta": delta, "z_cf": z_cf}
        if decoder is not None:
            result["x_cf"] = self.decode_counterfactual(z_cf, decoder)
        return result
