#!/usr/bin/env python
"""Build CRC ICB from-scratch AnnData inputs from raw 10x files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.discovery.from_scratch.crc_icb_inputs import build_crc_icb_h5ad


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CRC ICB from-scratch input AnnData")
    parser.add_argument("--icb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "from_scratch" / "crc_icb")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.icb_root / "input"
    output_dir = args.output_dir
    output_h5ad = output_dir / "crc_icb_raw_counts.h5ad"
    metadata_csv = output_dir / "crc_icb_cell_metadata.csv"
    provenance_json = output_dir / "crc_icb_input_provenance.json"

    build_crc_icb_h5ad(
        matrix_path=input_dir / "matrix.mtx.gz",
        features_path=input_dir / "features.tsv.gz",
        barcodes_path=input_dir / "barcodes.tsv.gz",
        geo_metadata_path=input_dir / "GSE236581_CRC-ICB_metadata.txt.gz",
        sample_metadata_path=input_dir / "scCRC_ICB_sample_meta.csv",
        patient_metadata_path=input_dir / "scCRC_ICB_patient meta.csv",
        output_h5ad=output_h5ad,
        output_metadata_csv=metadata_csv,
        provenance_path=provenance_json,
        max_cells=args.max_cells,
        random_seed=args.random_seed,
    )
    print(f"AnnData: {output_h5ad}")
    print(f"Metadata: {metadata_csv}")
    print(f"Provenance: {provenance_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
