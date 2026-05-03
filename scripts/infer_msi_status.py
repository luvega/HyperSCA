#!/usr/bin/env python
"""基于已发表 MSI 基因签名推断 CRC 样本的 MSS/MSI 状态（无临床标注时近似推断）。

参考文献与基因集来源：
- Yang et al. (2022) Sci Rep: Tumor microenvironment-aware MSI prediction (MAP)
- Liu et al. (2022) Front Surg: Microsatellite-Related Transcriptomic Signature (MSRS)
- Immune/stromal 分化：MSI-H 通常免疫浸润高（CD8A/GZMB 等），MSS 通常基质丰富（COL1A1/FN1 等）

用法：
    python scripts/infer_msi_status.py --input data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/filtered_feature_bc_matrix.h5 --input-type 10x_h5 --positions data/VisiumHD_HumanColon_Oliveira/binned_outputs/square_016um/spatial/tissue_positions.parquet
    python scripts/infer_msi_status.py --input data/ST/ST_CRC_MSS/expression.h5ad --input-type h5ad --sample-id ST_CRC_MSS
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.msi_inference import (
    MMR_GENES,
    MSI_HIGH_GENES,
    MSS_HIGH_GENES,
    compute_msi_score,
    compute_msi_score_combined,
    infer_status_from_score,
)


def _extract_expression_matrix(adata, max_obs: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """从 AnnData 提取表达矩阵与 var_names。稀疏矩阵时先降采样再取列，避免 OOM。"""
    from scipy import sparse

    var_names = np.asarray(adata.var_names)
    needed = set(MSI_HIGH_GENES + MSS_HIGH_GENES + MMR_GENES)
    vu = {str(v).upper(): (i, str(v)) for i, v in enumerate(var_names)}
    col_idx = [vu[g][0] for g in needed if g in vu]
    if not col_idx:
        # 无签名基因时取全部（小数据）或失败
        col_idx = list(range(adata.n_vars))

    n_obs = adata.n_obs
    if n_obs > max_obs:
        rng = np.random.default_rng(42)
        row_idx = np.sort(rng.choice(n_obs, size=max_obs, replace=False))
    else:
        row_idx = np.arange(n_obs)

    X_full = adata.X
    if sparse.issparse(X_full):
        X = X_full[row_idx, :][:, col_idx]
        if sparse.issparse(X):
            X = X.toarray()
    else:
        X = np.asarray(X_full[row_idx, :][:, col_idx], dtype=np.float64)

    X = np.asarray(X, dtype=np.float64)
    var_subset = var_names[col_idx]
    return X, var_subset


def run_inference_h5ad(
    h5ad_path: Path,
    sample_id: str = "",
    max_obs: int = 100_000,
) -> dict[str, Any]:
    """从 h5ad 执行 MSI 推断。"""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    sample_id = sample_id or Path(h5ad_path).stem
    X, var_names = _extract_expression_matrix(adata, max_obs=max_obs)
    res = compute_msi_score_combined(X, var_names)

    status, confidence = infer_status_from_score(res["score"])
    out = {
        "sample_id": sample_id,
        "input_type": "h5ad",
        "input_path": str(h5ad_path),
        "n_obs_used": int(X.shape[0]),
        "n_genes_total": int(X.shape[1]),
        "inferred_status": status,
        "confidence": confidence,
        **res,
    }
    return out


def run_inference_10x_h5(
    h5_path: Path,
    positions_path: Path | None,
    sample_id: str = "",
    max_obs: int = 100_000,
) -> dict[str, Any]:
    """从 10x filtered_feature_bc_matrix.h5 执行 MSI 推断。"""
    import scanpy as sc

    adata = sc.read_10x_h5(str(h5_path))
    if positions_path and positions_path.exists():
        pos = pd.read_parquet(positions_path) if str(positions_path).endswith(".parquet") else pd.read_csv(positions_path)
        # 若有 barcode + in_tissue，仅保留组织内 spot（提高信号质量）
        if "barcode" in pos.columns and "in_tissue" in pos.columns:
            bc_in = set(pos[pos["in_tissue"] == 1]["barcode"].tolist())
            if bc_in:
                adata = adata[adata.obs_names.isin(bc_in)].copy()

    sample_id = sample_id or Path(h5_path).parent.name
    X, var_names = _extract_expression_matrix(adata, max_obs=max_obs)
    res = compute_msi_score_combined(X, var_names)

    status, confidence = infer_status_from_score(res["score"])
    out = {
        "sample_id": sample_id,
        "input_type": "10x_h5",
        "input_path": str(h5_path),
        "n_obs_used": int(X.shape[0]),
        "n_genes_total": int(X.shape[1]),
        "inferred_status": status,
        "confidence": confidence,
        **res,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="基于 MSI 基因签名推断 CRC MSS/MSI 状态",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="h5ad 或 10x .h5 路径")
    parser.add_argument(
        "--input-type",
        choices=["h5ad", "10x_h5"],
        default="h5ad",
        help="输入类型",
    )
    parser.add_argument(
        "--positions",
        help="10x 空间坐标（tissue_positions.parquet 或 .csv），仅 input-type=10x_h5 时需要",
    )
    parser.add_argument("--sample-id", help="样本 ID，用于输出标识")
    parser.add_argument(
        "--max-obs",
        type=int,
        default=100_000,
        help="最大 spot 数（超则随机抽样，保证可重复）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="输出 JSON 路径，默认 results/integration/msi_inference/<sample_id>.json",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 输入不存在: {inp}", file=sys.stderr)
        return 1

    if args.input_type == "h5ad":
        result = run_inference_h5ad(inp, sample_id=args.sample_id or "", max_obs=args.max_obs)
    else:
        pos_path = Path(args.positions) if args.positions else None
        result = run_inference_10x_h5(
            inp,
            pos_path,
            sample_id=args.sample_id or "",
            max_obs=args.max_obs,
        )

    print("=" * 60)
    print("MSI 推断结果（主免疫/基质 + 辅 MMR/MLH1，非临床金标准）")
    print("=" * 60)
    print(f"  样本: {result['sample_id']}")
    print(f"  推断状态: {result['inferred_status']} (置信度: {result['confidence']})")
    print(f"  综合得分: {result['score']:.4f} (正值偏 MSI，负值偏 MSS)")
    prim = result.get("score_primary", result["score"])
    mmr = result.get("score_mmr_aux", 0.0)
    print(f"    - 主得分(免疫/基质): {prim:.4f}")
    print(f"    - MMR 辅助(MLH1低→MSI-like): {mmr:.4f}")
    if "mean_mlh1" in result and not (isinstance(result["mean_mlh1"], float) and np.isnan(result["mean_mlh1"])):
        print(f"  MLH1 平均表达(log1p): {result['mean_mlh1']:.4f}")
    print(f"  基因覆盖: MSI {result['n_msi_found']}/{len(MSI_HIGH_GENES)}, MSS {result['n_mss_found']}/{len(MSS_HIGH_GENES)}, MMR {result.get('n_mmr_found', 0)}/{len(MMR_GENES)}")
    print("=" * 60)

    out_path = args.output
    if not out_path:
        out_dir = ROOT / "results" / "integration" / "msi_inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{result['sample_id']}_msi_inference.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入前将 numpy 转为 Python 原生类型
    def _todict(obj):
        if isinstance(obj, dict):
            return {k: _todict(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload = _todict(result)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {out_path}")

    # 同时保存 MD 报告
    md_path = out_path.with_suffix(".md")
    _write_md_report(result, md_path)
    print(f"  已保存: {md_path}")

    return 0


def _write_md_report(result: dict[str, Any], path: Path) -> None:
    """将 MSI 推断结果写入 Markdown 文件。"""
    sid = result["sample_id"]
    status = result["inferred_status"]
    conf = result["confidence"]
    score = result["score"]
    prim = result.get("score_primary", score)
    mmr = result.get("score_mmr_aux", 0.0)
    mean_mlh1 = result.get("mean_mlh1", float("nan"))
    mlh1_str = f"{mean_mlh1:.4f}" if not (isinstance(mean_mlh1, float) and np.isnan(mean_mlh1)) else "N/A"

    msi_found = result.get("msi_genes_found", [])
    mss_found = result.get("mss_genes_found", [])
    mmr_found = result.get("mmr_genes_found", [])

    content = f"""# MSI 推断报告：{sid}

> 基于转录组签名（主免疫/基质 + 辅 MMR/MLH1），**非临床金标准**，仅供参考。

## 推断结果

| 项目 | 值 |
|------|-----|
| **推断状态** | **{status}** |
| 置信度 | {conf} |
| 综合得分 | {score:.4f} |
| 主得分（免疫/基质） | {prim:.4f} |
| MMR 辅助（MLH1 低→MSI-like） | {mmr:.4f} |
| MLH1 平均表达 (log1p) | {mlh1_str} |

> 得分说明：正值偏 MSI，负值偏 MSS。

## 基因覆盖

| 签名 | 命中 | 总数 |
|------|------|------|
| MSI 相关 | {len(msi_found)} | {len(MSI_HIGH_GENES)} |
| MSS 相关 | {len(mss_found)} | {len(MSS_HIGH_GENES)} |
| MMR 辅助 | {len(mmr_found)} | {len(MMR_GENES)} |

## 输入信息

- 输入类型: {result.get("input_type", "N/A")}
- 输入路径: {result.get("input_path", "N/A")}
- 观测数（spot/cell）: {result.get("n_obs_used", "N/A")}
- 基因总数: {result.get("n_genes_total", "N/A")}
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
