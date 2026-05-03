"""External data loading helpers for target discovery."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_neu_deg_tables(neu_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted(Path(neu_dir).glob("*-DESeq2_result.tsv")):
        celltype = path.stem.replace("-DESeq2_result", "")
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            continue
        if "padj" not in df.columns or "log2FoldChange" not in df.columns:
            continue
        sig = df[(df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 0.5)].copy()
        for _, row in sig.iterrows():
            records.append(
                {
                    "gene": str(row.get("symbol", "")),
                    "celltype_neu": celltype,
                    "lfc_neu": float(row["log2FoldChange"]),
                    "padj_neu": float(row["padj"]),
                }
            )
    return pd.DataFrame(records, columns=["gene", "celltype_neu", "lfc_neu", "padj_neu"])


def read_icb_deg_tables(icb_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for name in ["DEGs_MSS_response_Mid_lfc0.5.csv", "DEGs_MSS_Mid.csv", "DEGs_MSS_response_Major_lfc0.5.csv", "DEGs_MSS_Major.csv"]:
        path = Path(icb_dir) / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        gene_col = "gene" if "gene" in df.columns else df.columns[0]
        lfc_col = "avg_log2FC" if "avg_log2FC" in df.columns else None
        padj_col = "p_val_adj" if "p_val_adj" in df.columns else None
        ct_col = "celltype" if "celltype" in df.columns else None
        for _, row in df.iterrows():
            if lfc_col and padj_col:
                try:
                    padj = float(row[padj_col])
                    lfc = float(row[lfc_col])
                except (TypeError, ValueError):
                    continue
                if padj > 0.05 or abs(lfc) < 0.3:
                    continue
            records.append(
                {
                    "gene": str(row[gene_col]),
                    "celltype_icb": str(row[ct_col]) if ct_col else name,
                    "lfc_icb": float(row[lfc_col]) if lfc_col else float("nan"),
                    "padj_icb": float(row[padj_col]) if padj_col else float("nan"),
                    "source_file": name,
                }
            )
    return pd.DataFrame(records, columns=["gene", "celltype_icb", "lfc_icb", "padj_icb", "source_file"])


def read_ifng_tables(ifng_dir: Path, focus_genes: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict] = []
    path = Path(ifng_dir) / "results" / "tables" / "targets_shared_specific_by_mmr.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            gene_col = "gene" if "gene" in df.columns else df.columns[0]
            for _, row in df.iterrows():
                gene = str(row[gene_col])
                if gene and gene != "nan":
                    records.append(
                        {
                            "gene": gene,
                            "celltype_ifng": str(row.get("celltype", "unknown")),
                            "lfc_ifng": float(row.get("log2FoldChange", row.get("avg_log2FC", 0))),
                            "mmr_group": str(row.get("mmr_group", "")),
                        }
                    )
        except Exception:
            records = []
    for gene in focus_genes:
        if not any(row["gene"] == gene for row in records):
            records.append({"gene": gene, "celltype_ifng": "IFNG_focus", "lfc_ifng": float("nan"), "mmr_group": ""})
    return pd.DataFrame(records, columns=["gene", "celltype_ifng", "lfc_ifng", "mmr_group"])
