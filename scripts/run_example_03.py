"""Example 03: VisiumHD 分割统计（增强版）"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_visiumhd_geojson
from src.data.validators import validate_geojson_features
from src.examples.config import VISIUMHD_DIR, RESULTS_DIR
from src.examples.segmentation_stats import (
    compute_areas,
    plot_area_hist,
    plot_segmentation_quality,
    plot_area_scatter,
    segmentation_summary,
)


def main():
    out_dir = RESULTS_DIR / "example03"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    print("[Example03] 加载 VisiumHD 分割数据 ...")
    cell_feats = load_visiumhd_geojson(VISIUMHD_DIR, layer="cell")
    nuc_feats = load_visiumhd_geojson(VISIUMHD_DIR, layer="nucleus")
    print(f"  Cell features: {len(cell_feats)}")
    print(f"  Nucleus features: {len(nuc_feats)}")

    # 校验
    for label, feats in [("cell", cell_feats), ("nucleus", nuc_feats)]:
        issues = validate_geojson_features(feats)
        if issues:
            print(f"  ⚠ {label} 校验: {issues}")

    # 计算面积
    cell_areas = compute_areas(cell_feats)
    nuc_areas = compute_areas(nuc_feats)

    # 汇总
    stats = segmentation_summary(cell_areas, nuc_areas)
    print(f"  分割统计: {json.dumps(stats, indent=2)}")

    # 保存
    cell_areas.to_csv(out_dir / "cell_areas.csv", index=False)
    nuc_areas.to_csv(out_dir / "nucleus_areas.csv", index=False)
    with open(out_dir / "segmentation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- 绘图 ---
    # 1. 基础面积直方图（向后兼容）
    plot_area_hist(cell_areas, nuc_areas, save_path=str(out_dir / "area_hist.png"))

    # 2. 增强 3-panel 分割质量图（含 NC Ratio）
    plot_segmentation_quality(cell_areas, nuc_areas,
                              save_path=str(out_dir / "segmentation_quality.png"))
    print("  生成: segmentation_quality.png")

    # 3. Cell vs Nucleus 联合散点图
    plot_area_scatter(cell_areas, nuc_areas,
                      save_path=str(out_dir / "area_scatter.png"))
    print("  生成: area_scatter.png")

    print(f"  输出 -> {out_dir}")


if __name__ == "__main__":
    main()
