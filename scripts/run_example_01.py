"""Example 01: Chromium 元数据 QC 分析（增强版）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_chromium_meta
from src.data.validators import validate_chromium_meta
from src.examples.config import CHROMIUM_DIR, RESULTS_DIR
from src.examples.metadata_qc import (
    celltype_summary,
    filter_kept_cells,
    patient_summary,
    plot_celltype_bar,
    plot_celltype_nested_bar,
    plot_celltype_sunburst,
    plot_patient_qc_bar,
)


def main():
    out_dir = RESULTS_DIR / "example01"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    print("[Example01] 加载 Chromium 元数据 ...")
    df = load_chromium_meta(CHROMIUM_DIR)
    print(f"  总行数: {len(df)}")

    # 校验
    issues = validate_chromium_meta(df)
    if issues:
        print(f"  ⚠ 校验问题: {issues}")
    else:
        print("  校验通过")

    # 过滤
    kept = filter_kept_cells(df)
    print(f"  QC 保留: {len(kept)} / {len(df)}")

    # 细胞类型统计
    ct = celltype_summary(kept)
    ct.to_csv(out_dir / "summary.csv", index=False)
    print(f"  细胞类型数 (Level2): {ct['Level2'].nunique()}")

    # 患者统计
    pt = patient_summary(df)
    pt.to_csv(out_dir / "patient_summary.csv", index=False)

    # --- 绘图 ---
    # 1. 基础 Level1 条形图（向后兼容）
    plot_celltype_bar(ct, save_path=str(out_dir / "fig_celltype.png"))

    # 2. 嵌套条形图 (Level1 × Level2)
    plot_celltype_nested_bar(ct, save_path=str(out_dir / "fig_celltype_nested.png"))

    # 3. Sunburst 环形层级图
    plot_celltype_sunburst(ct, save_path=str(out_dir / "fig_celltype_sunburst.png"))

    # 4. 患者 QC 堆叠条形图
    plot_patient_qc_bar(pt, save_path=str(out_dir / "fig_patient_qc.png"))

    print(f"  输出 -> {out_dir}")


if __name__ == "__main__":
    main()
