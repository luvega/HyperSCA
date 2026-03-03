#!/usr/bin/env python
"""阶段 4 展示图生成（Dynamic Intervention）。

输入:
    - results/step4/pkpd_summary.json
    - results/step4/dynamic_target_effects.json
    - results/step4/combination_ranking.csv
    - results/step4/roundtrip_update_report.json (可选)

输出:
    - results/figures/step4/dynamic_target_trajectories.png
    - results/figures/step4/combination_effects_synergy.png
    - results/figures/step4/roundtrip_before_after.png
    - results/figures/step4/pkpd_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plot_style import CMAP_EXPRESSION, apply_cns_style, save_figure


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_pkpd_curves(step4_dir: Path, out_dir: Path) -> None:
    payload = _load_json(step4_dir / "pkpd_summary.json")
    curves = payload.get("pk_curves", {})
    pd_summary = payload.get("pd_summary", {})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    if not curves:
        for ax in axes:
            ax.text(0.5, 0.5, "No PK/PD data", ha="center", va="center", transform=ax.transAxes)
        save_figure(fig, out_dir / "pkpd_curves.png", dpi=300, config={"chart": "pkpd_curves"})
        return

    for dose_s, vals in curves.items():
        y = np.asarray(vals, dtype=float)
        x = np.arange(len(y))
        ax1.plot(x, y, marker="o", linewidth=1.5, label=f"dose={dose_s}")
    ax1.set_title("PK concentration curves")
    ax1.set_xlabel("time index")
    ax1.set_ylabel("concentration")
    ax1.legend(fontsize=8)

    for dose_s, meta in pd_summary.items():
        y = np.asarray(meta.get("effect_curve", []), dtype=float)
        x = np.arange(len(y))
        ax2.plot(x, y, marker="o", linewidth=1.5, label=f"dose={dose_s}")
    ax2.set_title("PD effect curves")
    ax2.set_xlabel("time index")
    ax2.set_ylabel("effect")
    ax2.legend(fontsize=8)

    fig.suptitle("Step4 PK/PD Dynamics")
    save_figure(fig, out_dir / "pkpd_curves.png", dpi=300, config={"chart": "pkpd_curves"})


def _plot_dynamic_trajectories(step4_dir: Path, out_dir: Path) -> None:
    payload = _load_json(step4_dir / "dynamic_target_effects.json")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    if not payload:
        for ax in axes:
            ax.text(0.5, 0.5, "No dynamic target effects", ha="center", va="center", transform=ax.transAxes)
        save_figure(fig, out_dir / "dynamic_target_trajectories.png", dpi=300, config={"chart": "dynamic_target_trajectories"})
        return

    # left panel: mean trajectory per target
    for tg, meta in payload.items():
        arr = np.asarray(meta.get("temporal_effect", []), dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            continue
        mean_t = arr.mean(axis=1)
        ax1.plot(np.arange(len(mean_t)), mean_t, marker="o", linewidth=1.8, label=tg)
    ax1.set_title("Per-target mean temporal effect")
    ax1.set_xlabel("time index")
    ax1.set_ylabel("mean effect")
    ax1.legend(fontsize=8)

    # right panel: heatmap target x time
    targets = []
    stack = []
    for tg, meta in payload.items():
        arr = np.asarray(meta.get("temporal_effect", []), dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            continue
        targets.append(tg)
        stack.append(arr.mean(axis=1))
    if stack:
        mat = np.vstack(stack)
        im = ax2.imshow(mat, aspect="auto", cmap=CMAP_EXPRESSION)
        fig.colorbar(im, ax=ax2, shrink=0.75, pad=0.02, label="mean effect")
        ax2.set_yticks(range(len(targets)))
        ax2.set_yticklabels(targets)
        ax2.set_xticks(range(mat.shape[1]))
        ax2.set_xticklabels([str(i) for i in range(mat.shape[1])], fontsize=8)
    ax2.set_title("Target x Time effect heatmap")
    ax2.set_xlabel("time index")

    fig.suptitle("Step4 Dynamic Target Trajectories")
    save_figure(fig, out_dir / "dynamic_target_trajectories.png", dpi=300, config={"chart": "dynamic_target_trajectories"})


def _plot_combo_synergy(step4_dir: Path, out_dir: Path) -> None:
    path = step4_dir / "combination_ranking.csv"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    if not path.exists():
        for ax in axes:
            ax.text(0.5, 0.5, "No combination ranking", ha="center", va="center", transform=ax.transAxes)
        save_figure(fig, out_dir / "combination_effects_synergy.png", dpi=300, config={"chart": "combination_effects_synergy"})
        return

    df = pd.read_csv(path)
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Empty combination ranking", ha="center", va="center", transform=ax.transAxes)
        save_figure(fig, out_dir / "combination_effects_synergy.png", dpi=300, config={"chart": "combination_effects_synergy"})
        return

    top = df.head(15).copy()
    ax1.barh(np.arange(len(top)), top["effect"].values, color="#4477AA", edgecolor="white")
    ax1.set_yticks(np.arange(len(top)))
    ax1.set_yticklabels(top["combo"].values, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("effect")
    ax1.set_title("Top combination effects")

    ax2.barh(np.arange(len(top)), top["synergy_bliss"].values, color="#00A087", edgecolor="white")
    ax2.set_yticks(np.arange(len(top)))
    ax2.set_yticklabels(top["combo"].values, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Bliss synergy")
    ax2.set_title("Top combination synergy")

    fig.suptitle("Step4 Combination Effects and Synergy")
    save_figure(fig, out_dir / "combination_effects_synergy.png", dpi=300, config={"chart": "combination_effects_synergy"})


def _plot_roundtrip(step4_dir: Path, out_dir: Path) -> None:
    payload = _load_json(step4_dir / "roundtrip_update_report.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    if not payload:
        for ax in axes:
            ax.text(0.5, 0.5, "No roundtrip report", ha="center", va="center", transform=ax.transAxes)
        save_figure(fig, out_dir / "roundtrip_before_after.png", dpi=300, config={"chart": "roundtrip_before_after"})
        return

    baseline = payload.get("baseline_summary", {})
    rerun = payload.get("rerun_summary", {})
    b_eff = float(baseline.get("top_combo", {}).get("effect", 0.0))
    r_eff = float(rerun.get("top_combo", {}).get("effect", 0.0))
    ax1.bar(["baseline", "rerun"], [b_eff, r_eff], color=["#CC6677", "#44AA99"], edgecolor="white")
    ax1.set_title("Top combo effect before/after roundtrip")
    ax1.set_ylabel("effect")

    calib = payload.get("calibrated_params", {})
    ec50 = float(calib.get("ec50", 0.0))
    emax = float(calib.get("emax", 0.0))
    ax2.bar(["ec50", "emax"], [ec50, emax], color=["#4477AA", "#EE6677"], edgecolor="white")
    ax2.set_title("Calibrated PK/PD parameters")
    ax2.set_ylabel("value")

    fig.suptitle("Step4 Roundtrip Before/After")
    save_figure(fig, out_dir / "roundtrip_before_after.png", dpi=300, config={"chart": "roundtrip_before_after"})


def main() -> int:
    parser = argparse.ArgumentParser(description="HyperSCA Step4: Generate figures")
    parser.add_argument("--input-dir", type=str, default="results/step4", help="Step4 output directory")
    parser.add_argument("--output-dir", type=str, default="results/figures/step4", help="Figure output directory")
    args = parser.parse_args()

    step4_dir = ROOT / args.input_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_cns_style()
    print("HyperSCA Step4: Generating figures")
    print("=" * 60)
    print(f"  Input:  {step4_dir}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    _plot_pkpd_curves(step4_dir, out_dir)
    _plot_dynamic_trajectories(step4_dir, out_dir)
    _plot_combo_synergy(step4_dir, out_dir)
    _plot_roundtrip(step4_dir, out_dir)

    print(f"[DONE] Step4 figures saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

