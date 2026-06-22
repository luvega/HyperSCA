from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite


def test_build_icb_reference_anndata_from_raw_10x_and_metadata(tmp_path):
    from src.discovery.target_discovery.cell2location_context import build_icb_reference_anndata

    raw = tmp_path / "raw_icb"
    raw.mkdir()
    mmwrite(raw / "matrix.mtx", sparse.coo_matrix([[1, 0, 3], [0, 5, 0]], dtype=int))
    (raw / "features.tsv").write_text("ENSG1\tGENE_A\tGene Expression\nENSG2\tGENE_B\tGene Expression\n", encoding="utf-8")
    (raw / "barcodes.tsv").write_text("CELL1\nCELL2\nCELL3\n", encoding="utf-8")
    meta = tmp_path / "metadata.tsv"
    meta.write_text(
        "barcode\tMidCellType\tMajorCellType\n"
        "CELL1\tTumor\tEpi\n"
        "CELL2\tMacrophage\tMye\n"
        "CELL3\tTumor\tEpi\n",
        encoding="utf-8",
    )
    (tmp_path / "DEGs_MSS_response_Major.csv").write_text("gene\nFORBIDDEN\n", encoding="utf-8")

    adata = build_icb_reference_anndata(raw, meta, label_key="MidCellType", major_label_key="MajorCellType")

    assert adata.shape == (3, 2)
    assert adata.var_names.tolist() == ["GENE_A", "GENE_B"]
    assert adata.obs["cell2location_label"].tolist() == ["Tumor", "Macrophage", "Tumor"]
    assert adata.obs["cell2location_major"].tolist() == ["Epi", "Mye", "Epi"]


def test_spatial_gem_to_anndata_preserves_raw_counts_and_regions(tmp_path):
    from src.discovery.target_discovery.cell2location_context import spatial_gem_to_anndata

    meta = tmp_path / "STmetadata_SAMPLE_T.csv"
    meta.write_text(
        "x,y,level3\n"
        "1,1,tumor_center\n"
        "1,2,stroma\n"
        "2,1,boundary\n",
        encoding="utf-8",
    )
    gem = tmp_path / "STexpression_SAMPLE_T.gem.gz"
    with gzip.open(gem, "wt", encoding="utf-8") as handle:
        handle.write("geneID\tx\ty\tMIDCounts\n")
        handle.write("GENE_A\t1\t1\t4\n")
        handle.write("GENE_B\t1\t1\t2\n")
        handle.write("GENE_A\t1\t2\t3\n")
        handle.write("GENE_B\t2\t1\t5\n")

    adata = spatial_gem_to_anndata(gem, meta, sample_id="SAMPLE_T")

    assert adata.shape == (3, 2)
    assert adata.obs["sample_id"].tolist() == ["SAMPLE_T", "SAMPLE_T", "SAMPLE_T"]
    assert adata.obs["level3"].tolist() == ["tumor_center", "stroma", "boundary"]
    assert int(adata.X.sum()) == 14


def test_spatial_gem_to_anndata_coerces_noninteger_counts_and_records_audit(tmp_path):
    from src.discovery.target_discovery.cell2location_context import spatial_gem_to_anndata

    meta = tmp_path / "STmetadata_SAMPLE_T.csv"
    meta.write_text(
        "x,y,level3\n"
        "1,1,tumor_center\n"
        "1,2,stroma\n",
        encoding="utf-8",
    )
    gem = tmp_path / "STexpression_SAMPLE_T.gem.gz"
    with gzip.open(gem, "wt", encoding="utf-8") as handle:
        handle.write("geneID\tx\ty\tMIDCounts\n")
        handle.write("GENE_A\t1\t1\t2.4\n")
        handle.write("GENE_B\t1\t1\t3.6\n")
        handle.write("GENE_A\t1\t2\t-1.2\n")

    adata = spatial_gem_to_anndata(gem, meta, sample_id="SAMPLE_T")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X).ravel()

    assert np.all(values >= 0)
    assert np.allclose(values, np.rint(values))
    assert adata.uns["spatial_count_audit"]["transform"] == "round_nonnegative"
    assert adata.uns["spatial_count_audit"]["noninteger_fraction"] > 0
    assert adata.uns["spatial_count_audit"]["negative_value_count"] == 1


def test_downsample_reference_by_label_is_seeded_and_records_manifest():
    import anndata as ad
    import numpy as np

    from src.discovery.target_discovery.cell2location_context import downsample_reference_by_label

    adata = ad.AnnData(
        X=np.arange(24).reshape(8, 3),
        obs=pd.DataFrame(
            {"cell2location_label": ["Tumor"] * 5 + ["T"] * 2 + ["B"]},
            index=[f"cell{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=["GENE_A", "GENE_B", "GENE_C"]),
    )

    sampled = downsample_reference_by_label(adata, max_cells_per_label=2, seed=7)
    sampled_again = downsample_reference_by_label(adata, max_cells_per_label=2, seed=7)

    assert sampled.obs_names.tolist() == sampled_again.obs_names.tolist()
    assert sampled.obs["cell2location_label"].value_counts().to_dict() == {"Tumor": 2, "T": 2, "B": 1}
    assert sampled.uns["cell2location_reference_sampling"]["max_cells_per_label"] == 2
    assert sampled.uns["cell2location_reference_sampling"]["n_before"] == 8
    assert sampled.uns["cell2location_reference_sampling"]["n_after"] == 5


def test_read_cell2location_abundance_requires_manifest_and_rejects_signature_scores(tmp_path):
    from src.discovery.target_discovery.cell2location_context import read_cell2location_abundance_tables

    spatial = tmp_path / "spatial"
    spatial.mkdir()
    pd.DataFrame({"Tumor": [0.5], "Macrophage": [0.2]}).to_csv(spatial / "spatial_context.csv", index=False)

    with pytest.raises(ValueError, match="cell2location|manifest"):
        read_cell2location_abundance_tables(spatial)

    (spatial / "deconvolution_manifest.json").write_text(
        json.dumps({"method": "cell2location", "selected_detection_alpha": 20}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "spot_id": ["s1", "s2"],
            "sample_id": ["S", "S"],
            "x": [1, 2],
            "y": [1, 1],
            "level3": ["tumor_center", "stroma"],
            "Tumor": [8.0, 1.0],
            "Macrophage": [0.2, 3.0],
        }
    ).to_csv(spatial / "cell2location_abundance.csv.gz", index=False)

    tables = read_cell2location_abundance_tables(spatial)

    assert len(tables) == 1
    assert tables[0].attrs["spatial_context_method"] == "cell2location"
    assert tables[0]["Tumor"].tolist() == [8.0, 1.0]
