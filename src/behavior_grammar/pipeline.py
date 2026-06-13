"""Pipeline for behavior grammar rule export and virtual tissue simulation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.behavior_grammar.config import BehaviorGrammarConfig
from src.behavior_grammar.rule_builder import build_rules_from_discovery, load_discovery_tables
from src.behavior_grammar.simulation import compute_qoi_sensitivity, simulate_virtual_tissue
from src.discovery.target_discovery.artifacts import ArtifactWriter


class BehaviorGrammarPipeline:
    """Optional Stage 5 sidecar that exports behavior rules and toy dynamics."""

    def __init__(self, config: BehaviorGrammarConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        if self.config.paths.discovery_manifest is None:
            raise ValueError("BehaviorGrammarPipeline requires a discovery manifest path")
        manifest_path = Path(self.config.paths.discovery_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing discovery manifest: {manifest_path}")

        run_id = self.config.resolved_run_id()
        writer = ArtifactWriter(self.config.paths.output_base, run_id=run_id)
        started_at = time.time()
        outputs: dict[str, Any] = {}
        try:
            stage_started = time.time()
            ruleset = build_rules_from_discovery(manifest_path, max_rules=self.config.max_rules)
            writer.write_json("rules.json", ruleset.to_dict(), section="rules")
            rules_md_path = writer.write_markdown("rules.md", ruleset.to_markdown(), section="rules")
            writer.record_stage("rule_export", time.time() - stage_started, {"rules": len(ruleset.rules)})

            tables = load_discovery_tables(manifest_path)
            stage_started = time.time()
            trajectory, summary = simulate_virtual_tissue(
                ruleset,
                tables.cluster_expression,
                time_steps=self.config.time_steps,
                dt=self.config.dt,
            )
            sensitivity = compute_qoi_sensitivity(
                ruleset,
                tables.cluster_expression,
                time_steps=self.config.time_steps,
                dt=self.config.dt,
                delta=self.config.sensitivity_delta,
            )
            summary["step4_context"] = _load_step4_context(self.config.paths.step4_dir)
            writer.write_table("population_trajectory.csv", trajectory, section="simulation")
            writer.write_json("simulation_summary.json", summary, section="simulation")
            writer.write_table("qoi_sensitivity.csv", sensitivity, section="simulation")
            report_path = writer.write_markdown(
                "simulation_report.md",
                _format_simulation_report(ruleset, trajectory, summary, sensitivity),
                section="simulation",
            )
            writer.record_stage(
                "virtual_tissue_simulation",
                time.time() - stage_started,
                {"trajectory_rows": len(trajectory), "sensitivity_rows": len(sensitivity)},
            )

            stage_started = time.time()
            fig = _plot_population_trajectories(trajectory)
            fig_path = writer.write_figure(
                "population_trajectories.png",
                fig,
                section="figures",
                metadata={"run_id": run_id, "source_manifest": str(manifest_path)},
            )
            plt.close(fig)
            writer.record_stage("figures", time.time() - stage_started, {"figures": 1})

            outputs.update(
                {
                    "run_dir": writer.run_dir,
                    "manifest_path": writer.finalize(),
                    "rules_markdown": rules_md_path,
                    "simulation_report": report_path,
                    "figure_path": fig_path,
                    "summary": summary,
                    "elapsed_seconds": round(time.time() - started_at, 2),
                }
            )
            return outputs
        except Exception:
            writer.finalize()
            raise


def _load_step4_context(step4_dir: Path | None) -> dict[str, Any]:
    if step4_dir is None:
        return {"available": False}
    step4_dir = Path(step4_dir)
    summary_path = step4_dir / "step4_summary.json"
    combo_path = step4_dir / "combination_ranking.csv"
    context = {
        "available": step4_dir.exists(),
        "summary_path": str(summary_path),
        "combination_ranking_path": str(combo_path),
        "summary_exists": summary_path.exists(),
        "combination_ranking_exists": combo_path.exists(),
    }
    if summary_path.exists():
        try:
            context["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            context["summary_error"] = "invalid_json"
    return context


def _format_simulation_report(ruleset, trajectory, summary, sensitivity) -> str:
    final = trajectory[trajectory["time"] == trajectory["time"].max()] if not trajectory.empty else trajectory
    top_sensitivity = sensitivity.copy()
    if not top_sensitivity.empty and "sensitivity_index" in top_sensitivity:
        top_sensitivity = top_sensitivity.reindex(
            top_sensitivity["sensitivity_index"].abs().sort_values(ascending=False).index
        ).head(5)

    lines = [
        f"# Behavior Grammar Simulation Report: {ruleset.run_id}",
        "",
        "## Summary",
        "",
        f"- Rules: {summary['n_rules']}",
        f"- Cell types: {summary['n_cell_types']}",
        f"- Time steps: {summary['time_steps']}",
        f"- Final total population: {summary['final_total_population']:.3f}",
        f"- Max migration index: {summary['max_migration_index']:.3f}",
        f"- Max attack index: {summary['max_attack_index']:.3f}",
        f"- Max exhaustion index: {summary['max_exhaustion_index']:.3f}",
        "",
        "## Final Cell-State Proxies",
        "",
        _markdown_table(
            final,
            ["cell_type", "population", "migration_index", "attack_index", "exhaustion_index", "transition_index"],
            max_rows=12,
        ),
        "",
        "## Top QoI sensitivity",
        "",
        _markdown_table(
            top_sensitivity,
            ["rule_index", "cell_type", "signal", "behavior", "sensitivity_index"],
            max_rows=5,
        ),
        "",
        "## Generated Files",
        "",
        "- `rules/rules.md`",
        "- `rules/rules.json`",
        "- `simulation/population_trajectory.csv`",
        "- `simulation/simulation_summary.json`",
        "- `simulation/qoi_sensitivity.csv`",
        "- `figures/population_trajectories.png`",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(df, columns: list[str], *, max_rows: int) -> str:
    if df.empty:
        return "_No rows._"
    available = [col for col in columns if col in df.columns]
    if not available:
        return "_No matching columns._"
    rows = df[available].head(max_rows)
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for _, row in rows.iterrows():
        values = [_format_cell(row[col]) for col in available]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _plot_population_trajectories(trajectory):
    fig, ax = plt.subplots(figsize=(7, 4))
    if trajectory.empty:
        ax.set_title("Population trajectories")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        return fig
    for cell_type, sub in trajectory.groupby("cell_type"):
        ax.plot(sub["time"], sub["population"], marker="o", linewidth=1.8, label=str(cell_type))
    ax.set_title("Behavior Grammar Virtual Tissue Trajectories")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Population proxy")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
