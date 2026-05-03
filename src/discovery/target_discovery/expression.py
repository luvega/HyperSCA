"""Cluster-level expression assembly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_normalized_count_tables(neu_dir: Path, celltypes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for celltype in celltypes:
        path = Path(neu_dir) / f"{celltype}-NormalizedCounts.tsv"
        if path.exists():
            tables[celltype] = pd.read_csv(path, sep="\t", index_col=0)
    return tables


def assemble_cluster_expression(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    if not tables:
        raise RuntimeError("No NormalizedCounts loaded")
    series = {celltype: table.mean(axis=1) for celltype, table in tables.items()}
    expr = pd.DataFrame(series).T.fillna(0.0)
    expr = np.log1p(expr)
    labels = list(expr.index)
    return expr, labels
