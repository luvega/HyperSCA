"""Recompute CRC ICB target-discovery inputs from raw data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.discovery.from_scratch.crc_icb_artifacts import build_crc_icb_from_scratch_artifacts


def _parse_covariates(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute CRC ICB from-scratch artifacts")
    parser.add_argument("--icb-root", type=Path, default=Path("/home/a/Data/scCRC_ICB"))
    parser.add_argument("--st-root", type=Path, default=Path("/home/a/Data/ST_CRC_MSS"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "from_scratch")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-spots-per-sample", type=int, default=20_000)
    parser.add_argument("--max-spatial-samples", type=int, default=None)
    parser.add_argument("--genes-per-celltype", type=int, default=60)
    parser.add_argument("--min-pseudobulk-per-group", type=int, default=2)
    parser.add_argument("--min-wilcox-cells-per-group", type=int, default=50)
    parser.add_argument("--min-wilcox-donors-per-group", type=int, default=2)
    parser.add_argument("--min-wilcox-detection-fraction", type=float, default=0.05)
    parser.add_argument("--wilcox-padj-threshold", type=float, default=0.01)
    parser.add_argument("--wilcox-lfc-threshold", type=float, default=0.25)
    parser.add_argument("--wilcox-donor-direction-consistency-threshold", type=float, default=0.6)
    parser.add_argument("--wilcox-min-effective-donor-count", type=float, default=2.0)
    parser.add_argument("--wilcox-gene-block-size", type=int, default=128)
    parser.add_argument("--wilcox-sampling-policy", type=str, default="all_available")
    parser.add_argument("--response-column", type=str, default="binary_response")
    parser.add_argument("--case-label", type=str, default="non-pCR")
    parser.add_argument("--control-label", type=str, default="pCR")
    parser.add_argument("--celltype-col", type=str, default="MajorCellType")
    parser.add_argument("--covariates", type=str, default="MSI.MSS,Treatment.Stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = build_crc_icb_from_scratch_artifacts(
        icb_root=args.icb_root,
        st_root=args.st_root,
        output_root=args.output_root,
        max_cells=args.max_cells,
        max_spots_per_sample=args.max_spots_per_sample,
        max_spatial_samples=args.max_spatial_samples,
        genes_per_celltype=args.genes_per_celltype,
        min_pseudobulk_per_group=args.min_pseudobulk_per_group,
        min_wilcox_cells_per_group=args.min_wilcox_cells_per_group,
        min_wilcox_donors_per_group=args.min_wilcox_donors_per_group,
        min_wilcox_detection_fraction=args.min_wilcox_detection_fraction,
        wilcox_padj_threshold=args.wilcox_padj_threshold,
        wilcox_lfc_threshold=args.wilcox_lfc_threshold,
        wilcox_donor_direction_consistency_threshold=args.wilcox_donor_direction_consistency_threshold,
        wilcox_min_effective_donor_count=args.wilcox_min_effective_donor_count,
        wilcox_gene_block_size=args.wilcox_gene_block_size,
        wilcox_sampling_policy=args.wilcox_sampling_policy,
        response_column=args.response_column,
        case_label=args.case_label,
        control_label=args.control_label,
        celltype_col=args.celltype_col,
        covariates=_parse_covariates(args.covariates),
        random_seed=args.seed,
        chunksize=args.chunksize,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
