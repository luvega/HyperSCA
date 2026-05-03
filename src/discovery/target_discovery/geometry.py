"""Geometry comparison helpers for target discovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.discovery.target_discovery.constants import TYPE_MAPPING
from src.discovery.target_discovery.utils import knn_adjacency, normalize_adjacency


def compute_geometry(
    cluster_expr: pd.DataFrame,
    node_labels: list[str],
    mode: str,
    k: int = 4,
) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = cluster_expr.values.astype(np.float32)
    xz = StandardScaler().fit_transform(x)
    n_comp = min(8, xz.shape[0], xz.shape[1])
    z = PCA(n_components=n_comp).fit_transform(xz)
    z2 = z[:, :2]
    if z2.shape[1] < 2:
        z2 = np.pad(z2, ((0, 0), (0, 2 - z2.shape[1])), mode="constant")
    n_nodes = x.shape[0]

    if mode == "hyperbolic":
        from src.models.hyperbolic.lorentz import lorentz_to_poincare, polar_project
        from src.models.hyperbolic.poincare import poincare_distance

        zt = torch.tensor(z2, dtype=torch.float32)
        zt = zt / (zt.std() + 1e-6) * 0.5
        emb = lorentz_to_poincare(polar_project(zt)).detach().cpu().numpy()
        dist = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = poincare_distance(
                    torch.tensor(emb[i : i + 1], dtype=torch.float32),
                    torch.tensor(emb[j : j + 1], dtype=torch.float32),
                    c=1.0,
                ).item()
                dist[i, j] = dist[j, i] = d
    else:
        from scipy.spatial.distance import cdist

        emb = z2
        dist = cdist(emb, emb)

    adj = knn_adjacency(dist, k)
    type_map = {label: TYPE_MAPPING.get(label, label) for label in node_labels}
    within: list[float] = []
    between: list[float] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if type_map[node_labels[i]] == type_map[node_labels[j]]:
                within.append(float(dist[i, j]))
            else:
                between.append(float(dist[i, j]))

    metrics = {
        "mode": mode,
        "radius_mean": float(np.linalg.norm(emb, axis=1).mean()),
        "within_dist": float(np.mean(within)) if within else 0.0,
        "between_dist": float(np.mean(between)) if between else 0.0,
        "separation": float(np.mean(between) / max(np.mean(within), 1e-8)) if within else 0.0,
        "n_edges": int((adj > 0).sum()),
    }
    return {"mode": mode, "embedding": emb, "dist_matrix": dist, "adjacency": adj, "metrics": metrics}


def blend_adjacencies(spatial_adj: np.ndarray, geometry_adj: np.ndarray, blend: float) -> np.ndarray:
    blend = float(np.clip(blend, 0.0, 1.0))
    return normalize_adjacency((1.0 - blend) * spatial_adj + blend * geometry_adj)
