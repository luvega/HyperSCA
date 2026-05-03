"""Spatial co-localization context for target discovery."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.discovery.target_discovery.constants import ST_DECONV_MAP
from src.discovery.target_discovery.utils import normalize_adjacency


def read_st_metadata_tables(st_dir: Path) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for path in sorted(Path(st_dir).glob("STmetadata_*.csv")):
        try:
            tables.append(pd.read_csv(path, low_memory=False))
        except Exception:
            continue
    return tables


def build_spatial_adjacency_from_tables(tables: list[pd.DataFrame], node_labels: list[str]) -> np.ndarray:
    all_corr: list[np.ndarray] = []
    n_nodes = len(node_labels)
    if n_nodes <= 1:
        return np.zeros((n_nodes, n_nodes), dtype=float)
    for table in tables:
        scores = np.zeros((len(table), n_nodes), dtype=float)
        for i, celltype in enumerate(node_labels):
            cols = [col for col in ST_DECONV_MAP.get(celltype, ()) if col in table.columns]
            if cols:
                scores[:, i] = table[cols].mean(axis=1).values
        corr = np.nan_to_num(np.corrcoef(scores.T), nan=0.0)
        if corr.ndim == 0:
            corr = np.zeros((n_nodes, n_nodes), dtype=float)
        all_corr.append(corr)
    if not all_corr:
        return np.eye(n_nodes)
    adj = np.mean(all_corr, axis=0)
    adj = np.where(adj > 0.05, adj, 0.0)
    return normalize_adjacency(adj)
