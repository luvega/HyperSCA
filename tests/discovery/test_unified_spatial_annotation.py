from __future__ import annotations

import pandas as pd
import pytest
import anndata as ad
import numpy as np

from src.discovery.target_discovery.unified_annotation import (
    ABUNDANCE_METADATA_COLUMNS,
    PREDICTION_COLUMNS,
    build_unified_celltype_dictionary,
    cell2location_status_for_dataset,
    map_label_to_unified,
    run_hvae_label_transfer,
    run_rctd_existing_annotation,
    validate_abundance_table,
    validate_prediction_table,
)


def test_map_sc_crc_and_osta_labels_to_unified_levels():
    t_cell = map_label_to_unified(source_system="sccrc_icb", major_label="T", fine_label="c01_CD4_Tn_CCR7")
    assert t_cell.unified_level0 == "immune"
    assert t_cell.unified_level1 == "T_cells"
    assert t_cell.unified_level2 == "c01_CD4_Tn_CCR7"

    tumor = map_label_to_unified(source_system="osta", major_label="Tumor III", fine_label="Tumor III")
    assert tumor.unified_level0 == "tumor_epithelial"
    assert tumor.unified_level1 == "Tumor"
    assert tumor.unified_level2 == "Tumor III"

    caf = map_label_to_unified(source_system="osta", major_label="CAF", fine_label="CAF")
    assert caf.unified_level0 == "stromal"
    assert caf.unified_level1 == "Fibroblast"


def test_build_unified_celltype_dictionary_has_required_columns():
    dictionary = build_unified_celltype_dictionary()

    assert {"unified_level0", "unified_level1", "description"}.issubset(dictionary.columns)
    assert {"Tumor", "T_cells", "Fibroblast"}.issubset(set(dictionary["unified_level1"]))
    assert dictionary["unified_level1"].is_unique


def test_prediction_and_abundance_schema_validation():
    predictions = pd.DataFrame(
        {
            "obs_id": ["spot1"],
            "dataset_id": ["toy_visium"],
            "x": [1.0],
            "y": [2.0],
            "method": ["singler"],
            "reference": ["scCRC_ICB"],
            "assay_type": ["whole_transcriptome"],
            "unified_level0": ["tumor_epithelial"],
            "unified_level1": ["Tumor"],
            "unified_level2": ["Tumor III"],
            "confidence": [0.8],
            "status": ["ok"],
            "source_label": ["Tumor"],
            "top2_label": ["Fibroblast"],
        }
    )
    validate_prediction_table(predictions)

    abundance = pd.DataFrame(
        {
            "spot_id": ["spot1"],
            "sample_id": ["toy"],
            "x": [1.0],
            "y": [2.0],
            "level3": ["unknown"],
            "Tumor": [1.0],
            "Fibroblast": [0.0],
        }
    )
    validate_abundance_table(abundance)

    with pytest.raises(ValueError, match="prediction table missing columns"):
        validate_prediction_table(predictions.drop(columns=[PREDICTION_COLUMNS[0]]))
    with pytest.raises(ValueError, match="abundance table has no numeric cell-type columns"):
        validate_abundance_table(abundance[list(ABUNDANCE_METADATA_COLUMNS)])


def test_cell2location_status_handles_panel_and_cuda_gate():
    panel = cell2location_status_for_dataset(
        dataset_id="xenium",
        assay_type="targeted_panel",
        device="cuda",
        gpu_required=True,
        cuda_available=True,
    )
    assert panel.status == "not_applicable:targeted_panel_cell_level"
    assert not panel.runnable

    segmented = cell2location_status_for_dataset(
        dataset_id="visiumhd_segmented",
        assay_type="visiumhd_segmented_cell",
        device="cuda",
        gpu_required=True,
        cuda_available=True,
    )
    assert segmented.status == "not_applicable:segmented_cell_level"
    assert not segmented.runnable

    blocked = cell2location_status_for_dataset(
        dataset_id="visiumhd",
        assay_type="whole_transcriptome",
        device="cuda",
        gpu_required=True,
        cuda_available=False,
    )
    assert blocked.status == "blocked:cuda_unavailable"
    assert not blocked.runnable

    runnable = cell2location_status_for_dataset(
        dataset_id="visium",
        assay_type="whole_transcriptome",
        device="cuda",
        gpu_required=True,
        cuda_available=True,
    )
    assert runnable.status == "ready"
    assert runnable.runnable


def test_rctd_existing_annotation_uses_deconvolution_labels_and_weights():
    query = ad.AnnData(
        X=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "DeconClass": ["singlet", "doublet_certain"],
                "DeconLabel1": ["Tumor III", "CAF"],
                "DeconLabel2": ["CD4 T cell", "Macrophage"],
            },
            index=["spot1", "spot2"],
        ),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )

    result = run_rctd_existing_annotation(
        query,
        dataset_id="toy_visiumhd",
        reference_name="scCRC_ICB",
        assay_type="whole_transcriptome",
        label_key="DeconLabel1",
        secondary_label_key="DeconLabel2",
        class_key="DeconClass",
        source_system="osta",
    )

    assert result.status == "ok"
    assert result.predictions is not None
    assert result.predictions["method"].unique().tolist() == ["rctd"]
    assert result.predictions["unified_level1"].tolist() == ["Tumor", "Fibroblast"]
    assert result.predictions["top2_label"].tolist() == ["T_cells", "Myeloid"]
    assert result.abundance is not None
    assert {"spot_id", "Tumor", "Fibroblast", "T_cells", "Myeloid"}.issubset(result.abundance.columns)
    assert result.abundance.loc[result.abundance["spot_id"].eq("spot2"), "Fibroblast"].iloc[0] == pytest.approx(0.5)


def test_rctd_existing_annotation_marks_targeted_panel_not_applicable():
    query = ad.AnnData(
        X=np.asarray([[1.0]], dtype=np.float32),
        obs=pd.DataFrame({"DeconLabel1": ["Tumor"]}, index=["cell1"]),
        var=pd.DataFrame(index=["GENE1"]),
    )

    result = run_rctd_existing_annotation(
        query,
        dataset_id="toy_xenium",
        reference_name="scCRC_ICB",
        assay_type="targeted_panel",
        label_key="DeconLabel1",
        secondary_label_key="DeconLabel2",
        class_key="DeconClass",
        source_system="osta",
    )

    assert result.status == "not_applicable:targeted_panel_cell_level"
    assert result.predictions is None


def test_hvae_blocks_cuda_request_when_cuda_is_unavailable(monkeypatch):
    reference = ad.AnnData(
        X=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "unified_level1": ["Tumor", "T_cells"],
            },
            index=["r1", "r2"],
        ),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )
    query = ad.AnnData(
        X=np.asarray([[1.0, 0.0]], dtype=np.float32),
        obs=pd.DataFrame({"x": [0.0], "y": [0.0]}, index=["q1"]),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )
    monkeypatch.setattr("src.discovery.target_discovery.unified_annotation._torch_cuda_available", lambda: False)

    result = run_hvae_label_transfer(
        reference,
        query,
        dataset_id="toy",
        reference_name="ref",
        assay_type="whole_transcriptome",
        max_genes=2,
        epochs=1,
        device="cuda",
        seed=1,
    )

    assert result.status == "blocked:cuda_unavailable"
    assert result.predictions is None
