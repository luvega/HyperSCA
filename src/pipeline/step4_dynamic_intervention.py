"""Stage 4 pipeline: dynamic intervention (PK/PD + temporal-spatial propagation)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config import HyperSCAConfig


class DynamicInterventionPipeline:
    """Dynamic intervention simulation for retained hubs and combinations."""

    def __init__(self, config: HyperSCAConfig):
        self.config = config
        self.step2_dir = Path(config.step4_input_step2_dir)
        self.step3_dir = Path(config.step4_input_step3_dir)
        self.output_dir = Path(config.step4_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(config.seed)

    def load_inputs(self) -> dict:
        """Load required outputs from Step2/Step3."""
        cluster_adj_path = self.step2_dir / "cluster_adj.npy"
        cluster_expr_path = self.step2_dir / "cluster_expr.npy"
        node_info_path = self.step2_dir / "node_info.json"
        if not cluster_adj_path.exists():
            raise FileNotFoundError(f"Missing Step2 cluster adjacency: {cluster_adj_path}")
        if not cluster_expr_path.exists():
            raise FileNotFoundError(f"Missing Step2 cluster expression: {cluster_expr_path}")
        if not node_info_path.exists():
            raise FileNotFoundError(f"Missing Step2 node info: {node_info_path}")

        cluster_adj = np.load(cluster_adj_path)
        cluster_expr = np.load(cluster_expr_path)
        node_info = json.loads(node_info_path.read_text(encoding="utf-8"))
        node_labels = node_info.get("node_labels", [f"N{i}" for i in range(cluster_expr.shape[0])])

        expr_df_path = self.step2_dir / "cluster_expr_df.csv"
        if expr_df_path.exists():
            cluster_expr_df = pd.read_csv(expr_df_path, index_col=0)
        else:
            raise FileNotFoundError(
                f"Missing Step2 cluster expression gene names: {expr_df_path}. "
                "Step4 requires a gene-labeled cluster_expr_df.csv so target profiles "
                "are not silently replaced by generic GENE_0 columns."
            )

        targets = list(self.config.step4_hub_genes)
        retained_hubs_path = self.step3_dir.parent / "integration" / "discovery" / "hub_targets_retained.csv"
        if retained_hubs_path.exists():
            try:
                retained = pd.read_csv(retained_hubs_path)
                for g in retained["gene"].dropna().astype(str).tolist():
                    if g not in targets:
                        targets.append(g)
            except Exception:
                pass
        # Also scan step3 target outputs
        for p in sorted(self.step3_dir.glob("interaction_targets_*.csv")):
            g = p.stem.replace("interaction_targets_", "")
            if g and g not in targets:
                targets.append(g)
        if not targets:
            targets = self._derive_targets_from_expression(cluster_expr_df)
        return {
            "cluster_adj": cluster_adj,
            "cluster_expr_df": cluster_expr_df,
            "node_labels": node_labels,
            "targets": targets,
        }

    @staticmethod
    def _derive_targets_from_expression(cluster_expr_df: pd.DataFrame, max_targets: int = 5) -> list[str]:
        numeric = cluster_expr_df.apply(pd.to_numeric, errors="coerce")
        if numeric.empty:
            return []
        scores = numeric.abs().mean(axis=0).fillna(0.0) + numeric.var(axis=0).fillna(0.0)
        return scores.sort_values(ascending=False).index.astype(str).tolist()[:max_targets]

    def _target_node_profile(self, gene: str, cluster_expr_df: pd.DataFrame) -> np.ndarray:
        cols_up = {c.upper(): c for c in cluster_expr_df.columns}
        if gene.upper() not in cols_up:
            return np.zeros(cluster_expr_df.shape[0], dtype=float)
        col = cols_up[gene.upper()]
        v = cluster_expr_df[col].values.astype(float)
        vmax = np.max(np.abs(v))
        if vmax < 1e-12:
            return np.zeros_like(v)
        return np.clip(v / vmax, 0.0, None)

    def run(self) -> dict:
        from src.causal.temporal_causal import infer_temporal_causal_graph
        from src.perturbation.combinatorial_intervention import (
            generate_target_combinations,
            rank_combinations,
        )
        from src.perturbation.dose_response import summarize_dose_response
        from src.perturbation.pharmacokinetics import simulate_pk_grid
        from src.perturbation.temporal_spatial_propagation import simulate_temporal_spatial_propagation

        t0 = time.time()
        data = self.load_inputs()
        cluster_adj = data["cluster_adj"]
        cluster_expr_df = data["cluster_expr_df"]
        targets = data["targets"]
        node_labels = data["node_labels"]

        time_grid = list(self.config.step4_time_grid)
        dose_grid = list(self.config.step4_dose_grid)

        pk_curves = simulate_pk_grid(
            dose_grid=dose_grid,
            time_grid=time_grid,
            ka=self.config.step4_pk_ka,
            ke=self.config.step4_pk_ke,
            vd=self.config.step4_pk_vd,
            bioavailability=self.config.step4_pk_f,
        )
        pd_summary = summarize_dose_response(
            pk_curves,
            emax=self.config.step4_pd_emax,
            ec50=self.config.step4_pd_ec50,
            hill=self.config.step4_pd_hill,
        )

        # Per-target dynamic trajectories
        per_target_effects: dict[str, dict] = {}
        single_effect_final: dict[str, float] = {}
        temporal_stack = []
        dose_ref = max(dose_grid) if dose_grid else 1.0
        dose_gain = pd_summary.get(float(dose_ref), {}).get("mean_effect", 0.0)

        for g in targets:
            base = self._target_node_profile(g, cluster_expr_df) * float(dose_gain)
            temporal = simulate_temporal_spatial_propagation(
                base_effect=base,
                time_grid=time_grid,
                causal_adj=cluster_adj,
                diffusion=self.config.step4_temporal_diffusion,
                temporal_decay=self.config.step4_temporal_decay,
            )
            temporal_stack.append(temporal)
            final_effect = float(np.mean(temporal[-1])) if len(temporal) > 0 else 0.0
            single_effect_final[g] = final_effect
            active_nodes = np.where(temporal[-1] > np.quantile(temporal[-1], 0.7))[0].tolist() if len(temporal) > 0 else []
            per_target_effects[g] = {
                "final_mean_effect": final_effect,
                "active_nodes": [node_labels[i] for i in active_nodes if i < len(node_labels)],
                "temporal_effect": temporal.tolist(),
            }

        # Combo evaluation
        combo_effects: dict[tuple[str, ...], float] = {}
        combos = generate_target_combinations(targets, max_size=self.config.step4_combo_max_size)
        for combo in combos:
            if len(combo) == 1:
                combo_effects[combo] = single_effect_final.get(combo[0], 0.0)
                continue
            # Independent combined effect as proxy
            prod = 1.0
            for g in combo:
                eg = float(np.clip(single_effect_final.get(g, 0.0), 0.0, 1.0))
                prod *= (1.0 - eg)
            combo_effects[combo] = float(np.clip(1.0 - prod, 0.0, 1.0))
        combo_ranked = rank_combinations(combo_effects, single_effect_final)
        for row in combo_ranked:
            row["model_type"] = "bliss_proxy"
            row["calibrated_by_experiment"] = False

        # Temporal causal on averaged intervention response
        if temporal_stack:
            mean_temporal = np.mean(np.stack(temporal_stack, axis=0), axis=0)  # (T, K)
            t_adj, t_lag = infer_temporal_causal_graph(mean_temporal, max_lag=2, threshold=0.2)
        else:
            mean_temporal = np.zeros((len(time_grid), len(node_labels)))
            t_adj = np.zeros((len(node_labels), len(node_labels)))
            t_lag = np.zeros_like(t_adj, dtype=int)

        # Save outputs
        (self.output_dir / "pkpd_summary.json").write_text(
            json.dumps({"pk_curves": {str(k): v.tolist() for k, v in pk_curves.items()}, "pd_summary": pd_summary}, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "dynamic_target_effects.json").write_text(
            json.dumps(per_target_effects, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(combo_ranked).to_csv(self.output_dir / "combination_ranking.csv", index=False)
        np.save(self.output_dir / "temporal_causal_adjacency.npy", t_adj)
        np.save(self.output_dir / "temporal_causal_best_lag.npy", t_lag)
        np.save(self.output_dir / "mean_temporal_effect.npy", mean_temporal)

        summary = {
            "targets": targets,
            "n_targets": int(len(targets)),
            "top_combo": combo_ranked[0] if combo_ranked else {},
            "n_combo": int(len(combo_ranked)),
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        (self.output_dir / "step4_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
