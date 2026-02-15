"""Example 02: Visium 空间邻域图构建（增强版）"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_visium_spatial, load_chromium_meta
from src.data.validators import validate_visium_positions
from src.examples.config import VISIUM_DIR, CHROMIUM_DIR, RESULTS_DIR, SPATIAL_K
from src.examples.spatial_graph import (
    build_knn_edges,
    filter_in_tissue,
    graph_stats,
    plot_spatial_graph,
    plot_spatial_graph_colored,
    plot_edge_distance_distribution,
)


def main():
    out_dir = RESULTS_DIR / "example02"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    print("[Example02] 加载 Visium 空间数据 ...")
    positions, scalefactors = load_visium_spatial(VISIUM_DIR)
    print(f"  总 spot 数: {len(positions)}")
    print(f"  缩放因子: {scalefactors}")

    # 校验
    issues = validate_visium_positions(positions)
    if issues:
        print(f"  ⚠ 校验问题: {issues}")
    else:
        print("  校验通过")

    # 过滤
    tissue = filter_in_tissue(positions)
    print(f"  in_tissue spot: {len(tissue)}")

    # 构建 kNN 图
    coords = tissue[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
    edges = build_knn_edges(coords, k=SPATIAL_K)
    edges.to_csv(out_dir / "knn_edges.csv", index=False)

    # 统计
    stats = graph_stats(edges, n_nodes=len(tissue))
    print(f"  图统计: {stats}")
    with open(out_dir / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- 绘图 ---
    # 1. 基础空间图（向后兼容）
    plot_spatial_graph(coords, edges, save_path=str(out_dir / "spatial_graph.png"))

    # 2. 按 array_row 区域着色（模拟组织区分层）
    if "array_row" in tissue.columns:
        # 将 array_row 分为若干区域
        row_vals = tissue["array_row"].values
        n_bins = 5
        bins = np.linspace(row_vals.min(), row_vals.max() + 1, n_bins + 1)
        region_labels = np.digitize(row_vals, bins)
        region_names = np.array([f"Region {r}" for r in region_labels])
        plot_spatial_graph_colored(
            coords, edges, region_names,
            label_name="Tissue Region",
            save_path=str(out_dir / "spatial_graph_region.png"),
        )
        print("  生成: spatial_graph_region.png")

    # 3. 边距离分布
    plot_edge_distance_distribution(edges, save_path=str(out_dir / "edge_distance_dist.png"))
    print("  生成: edge_distance_dist.png")

    print(f"  输出 -> {out_dir}")


if __name__ == "__main__":
    main()
