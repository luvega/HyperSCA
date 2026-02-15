"""Example 04: Xenium 基因面板摘要（增强版）"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_xenium_meta
from src.examples.config import XENIUM_DIR, RESULTS_DIR
from src.examples.gene_panel_summary import (
    generate_report_md,
    panel_stats,
    parse_experiment_info,
    parse_targets,
    plot_descriptor_donut,
    plot_source_bar,
    plot_panel_composition_dashboard,
)


def main():
    out_dir = RESULTS_DIR / "example04"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    print("[Example04] 加载 Xenium 实验数据 ...")
    experiment, targets = load_xenium_meta(XENIUM_DIR)
    print(f"  Targets 数: {len(targets)}")

    # 解析
    exp_info = parse_experiment_info(experiment)
    targets_df = parse_targets(targets)
    stats = panel_stats(targets_df)
    print(f"  实验: {exp_info['run_name']} | 细胞数: {exp_info['num_cells']:,}")
    print(f"  Panel: {stats['gene_targets']} gene targets, {stats['other_targets']} other")

    # 保存
    targets_df.to_csv(out_dir / "panel_summary.csv", index=False)
    with open(out_dir / "experiment_info.json", "w") as f:
        json.dump(exp_info, f, indent=2)

    report = generate_report_md(exp_info, stats)
    (out_dir / "experiment_report.md").write_text(report, encoding="utf-8")

    # --- 绘图 ---
    # 1. Descriptor 环形图
    plot_descriptor_donut(targets_df, save_path=str(out_dir / "descriptor_donut.png"))
    print("  生成: descriptor_donut.png")

    # 2. Source panel 条形图
    plot_source_bar(targets_df, save_path=str(out_dir / "source_bar.png"))
    print("  生成: source_bar.png")

    # 3. 面板组成综合 Dashboard
    plot_panel_composition_dashboard(
        targets_df, exp_info=exp_info,
        save_path=str(out_dir / "panel_composition.png"),
    )
    print("  生成: panel_composition.png")

    print(f"  输出 -> {out_dir}")


if __name__ == "__main__":
    main()
