from __future__ import annotations

import gzip

import pandas as pd

from src.discovery.from_scratch.crc_icb_artifacts import build_crc_icb_from_scratch_artifacts
from src.discovery.from_scratch.crc_icb_inputs import merge_crc_icb_metadata, read_geo_cell_metadata


def test_merge_crc_icb_metadata_filters_crc_and_adds_binary_response():
    geo = pd.DataFrame(
        {
            "orig.ident": ["CRC01-T-I_CELL1", "CRC11-T-I_CELL2", "CRC02-T-I_CELL3"],
            "Ident": ["CRC01-T-I", "CRC11-T-I", "CRC02-T-I"],
            "Tissue": ["Tumor", "Tumor", "Tumor"],
            "MajorCellType": ["Epi", "Epi", "Mye"],
        }
    )
    sample = pd.DataFrame(
        {
            "Sample ID": ["CRC01-T-I", "CRC11-T-I", "CRC02-T-I"],
            "Patient ID": ["CRC01", "CRC11", "CRC02"],
            "Treatment Stage": ["Pre", "Pre", "Pre"],
        }
    )
    patient = pd.DataFrame(
        {
            "Patient ID": ["CRC01", "CRC11", "CRC02"],
            "Cancer Type": ["CRC", "Duodenal carcinoma", "CRC"],
            "MSI/MSS": ["MSS", "MSI", "MSS"],
            "Response": ["CR", "CR", "SD"],
        }
    )

    out = merge_crc_icb_metadata(geo, sample, patient)

    assert out["barcode"].tolist() == ["CRC01-T-I_CELL1", "CRC02-T-I_CELL3"]
    assert out["patient_id"].tolist() == ["CRC01", "CRC02"]
    assert out["binary_response"].tolist() == ["pCR", "non-pCR"]
    assert out["MSI.MSS"].tolist() == ["MSS", "MSS"]


def test_read_geo_cell_metadata_preserves_barcode_row_names(tmp_path):
    path = tmp_path / "geo.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write('"orig.ident" "nCount_RNA" "Ident" "Tissue" "MajorCellType"\n')
        handle.write('"CRC01-T-I_CELL1" "SeuratProject" 10 "CRC01-T-I" "Tumor" "Epi"\n')

    out = read_geo_cell_metadata(path)

    assert out.loc[0, "barcode"] == "CRC01-T-I_CELL1"
    assert out.loc[0, "Ident"] == "CRC01-T-I"


def test_build_crc_icb_from_scratch_artifacts_from_raw_mini_inputs(tmp_path):
    icb_input = tmp_path / "icb" / "input"
    st_root = tmp_path / "st"
    icb_input.mkdir(parents=True)
    st_root.mkdir()

    barcodes = [f"S{i}_CELL" for i in range(1, 9)]
    with gzip.open(icb_input / "barcodes.tsv.gz", "wt") as handle:
        handle.write("\n".join(barcodes) + "\n")
    with gzip.open(icb_input / "features.tsv.gz", "wt") as handle:
        handle.write("\n".join([f"G{i}\tG{i}\tGene Expression" for i in range(1, 5)]) + "\n")
    entries = [
        (1, 1, 10), (2, 1, 1), (1, 2, 8), (2, 2, 1),
        (1, 3, 1), (2, 3, 8), (1, 4, 1), (2, 4, 9),
        (3, 5, 7), (4, 5, 1), (3, 6, 6), (4, 6, 1),
        (3, 7, 1), (4, 7, 6), (3, 8, 1), (4, 8, 7),
    ]
    with gzip.open(icb_input / "matrix.mtx.gz", "wt") as handle:
        handle.write("%%MatrixMarket matrix coordinate integer general\n")
        handle.write("%\n")
        handle.write(f"4 8 {len(entries)}\n")
        for row, col, value in entries:
            handle.write(f"{row} {col} {value}\n")
    with gzip.open(icb_input / "GSE236581_CRC-ICB_metadata.txt.gz", "wt") as handle:
        handle.write('"orig.ident" "nCount_RNA" "Ident" "Tissue" "MajorCellType"\n')
        for idx, barcode in enumerate(barcodes, start=1):
            celltype = "Epi" if idx <= 4 else "Mye"
            handle.write(f'"{barcode}" "SeuratProject" 100 "S{idx}" "Tumor" "{celltype}"\n')
    pd.DataFrame(
        {
            "Sample ID": [f"S{i}" for i in range(1, 9)],
            "Patient ID": [f"P{i}" for i in range(1, 9)],
            "Treatment Stage": ["Pre"] * 8,
        }
    ).to_csv(icb_input / "scCRC_ICB_sample_meta.csv", index=False)
    pd.DataFrame(
        {
            "Patient ID": [f"P{i}" for i in range(1, 9)],
            "Cancer Type": ["CRC"] * 8,
            "MSI/MSS": ["MSS", "MSS", "MSI", "MSI", "MSS", "MSS", "MSI", "MSI"],
            "Response": ["SD", "SD", "CR", "CR", "SD", "SD", "CR", "CR"],
        }
    ).to_csv(icb_input / "scCRC_ICB_patient meta.csv", index=False)
    with gzip.open(st_root / "STexpression_SAMPLE.gem.gz", "wt") as handle:
        handle.write("geneID\tx\ty\tMIDCounts\n")
        handle.write("G1\t1\t1\t5\nG2\t1\t1\t1\nG3\t2\t2\t4\nG4\t2\t2\t1\n")

    outputs = build_crc_icb_from_scratch_artifacts(
        icb_root=tmp_path / "icb",
        st_root=st_root,
        output_root=tmp_path / "out",
        max_cells=None,
        max_spots_per_sample=None,
        genes_per_celltype=2,
        min_pseudobulk_per_group=2,
        min_wilcox_cells_per_group=2,
        min_wilcox_donors_per_group=2,
    )

    assert outputs["de"].exists()
    assert outputs["wilcox_de"].exists()
    assert outputs["expression"].exists()
    assert outputs["spatial"].exists()
    wilcox = pd.read_csv(outputs["wilcox_de"])
    assert set(["de_method", "evidence_tier", "n_case_cells_used", "n_control_cells_used"]).issubset(wilcox.columns)
    epi = wilcox[wilcox["celltype"].eq("Epi")]
    assert not epi.empty
    assert epi["n_case_cells_used"].eq(2).all()
    assert epi["n_case_cells_available"].eq(epi["n_case_cells_used"]).all()
    provenance = outputs["provenance"].read_text(encoding="utf-8")
    assert '"legacy_result_inputs_used": false' in provenance
    assert '"manual_target_genes_used": false' in provenance
    assert '"cell_use_fraction": 1.0' in provenance
