from __future__ import annotations

import importlib
import sys


def test_cell2location_cli_forces_noninteractive_matplotlib_backend(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    sys.modules.pop("scripts.run_cell2location_spatial_context", None)

    importlib.import_module("scripts.run_cell2location_spatial_context")

    import os

    assert os.environ["MPLBACKEND"] == "Agg"


def test_extract_reference_signatures_accepts_cell2location_varm_output():
    import anndata as ad
    import numpy as np
    import pandas as pd

    from scripts.run_cell2location_spatial_context import _extract_reference_signatures

    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["GENE_A", "GENE_B"]),
    )
    adata.varm["means_per_cluster_mu_fg"] = pd.DataFrame(
        {
            "means_per_cluster_mu_fg_Tumor": [3.0, 0.5],
            "means_per_cluster_mu_fg_Macro": [1.0, 2.0],
        },
        index=adata.var_names,
    )

    signatures = _extract_reference_signatures(adata)

    assert signatures.index.tolist() == ["GENE_A", "GENE_B"]
    assert signatures.columns.tolist() == ["Tumor", "Macro"]
    assert signatures.loc["GENE_A", "Tumor"] == 3.0
