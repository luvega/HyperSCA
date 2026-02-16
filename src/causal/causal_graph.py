"""因果细胞图数据结构与验证

功能:
- CausalCellGraph: NetworkX DiGraph 封装
- DoWhy 结构验证（falsification / arrow strength）
- 已知信号轴评估（recall / direction accuracy）
- 图统计与导出

参考 API:
    dowhy.gcm.refute_causal_structure()
    dowhy.gcm.arrow_strength()
    dowhy.gcm.falsify.falsify_graph()
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd


# =========================================================================
# CRC TME 已知信号轴先验（内建，可通过 JSON 覆盖）
# =========================================================================

CRC_KNOWN_AXES = [
    {
        "name": "CAF→TAM (POSTN)",
        "source_type": "CAF",
        "target_type": "TAM",
        "ligands": ["POSTN"],
        "receptors": ["ITGAV", "ITGB5"],
        "tfs": [],
        "targets": ["CD163", "MRC1"],
        "evidence": (
            "CAF-secreted POSTN binds integrin αVβ5 on TAMs, promoting "
            "M2 polarization and CD163/MRC1 upregulation in CRC TME."
        ),
    },
    {
        "name": "CAF→TAM (MFAP2)",
        "source_type": "CAF",
        "target_type": "TAM",
        "ligands": ["MFAP2"],
        "receptors": ["ITGA5", "ITGB1"],
        "tfs": [],
        "targets": ["CD163", "MRC1"],
        "evidence": (
            "MFAP2 from CAFs activates FAK/Src signaling in TAMs "
            "via integrin α5β1, driving immunosuppressive phenotype."
        ),
    },
    {
        "name": "CAF→Treg (INHBA)",
        "source_type": "CAF",
        "target_type": "Treg",
        "ligands": ["INHBA"],
        "receptors": ["ACVR1B", "ACVR2A"],
        "tfs": ["SMAD2", "SMAD3"],
        "targets": ["FOXP3"],
        "evidence": (
            "INHBA→ACVR1B/2A→SMAD2/3→Foxp3 axis: CAF-derived "
            "activin A promotes regulatory T cell differentiation "
            "and immunosuppression in CRC."
        ),
    },
    {
        "name": "TAM→CD8T (immunosuppression)",
        "source_type": "TAM",
        "target_type": "CD8T",
        "ligands": ["IL10", "TGFB1"],
        "receptors": ["IL10RA", "TGFBR1"],
        "tfs": [],
        "targets": ["PDCD1", "HAVCR2", "LAG3"],
        "evidence": (
            "M2 TAMs secrete IL-10 and TGF-β1, promoting CD8+ T cell "
            "exhaustion (PD-1/TIM-3/LAG-3 upregulation) in CRC TME."
        ),
    },
]


def load_known_axes(path: Optional[str] = None) -> list[dict]:
    """加载已知信号轴

    Parameters
    ----------
    path : JSON 文件路径（None 则使用内建 CRC 先验）

    Returns
    -------
    list of axis dicts
    """
    if path is not None:
        with open(path) as f:
            return json.load(f)
    return CRC_KNOWN_AXES


# =========================================================================
# 因果细胞图
# =========================================================================

class CausalCellGraph:
    """因果细胞通讯网络

    封装 NetworkX DiGraph，附带 bootstrap 频率、arrow strength
    等边属性，以及 DoWhy 验证方法。

    Parameters
    ----------
    adjacency : (K, K) 二值邻接矩阵
    node_labels : 长度 K 的节点名称
    bootstrap_freq : (K, K) bootstrap 出现频率
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        node_labels: list[str],
        bootstrap_freq: Optional[np.ndarray] = None,
    ):
        self.adjacency = adjacency.copy()
        self.node_labels = list(node_labels)
        self.K = len(node_labels)
        self.bootstrap_freq = (
            bootstrap_freq.copy() if bootstrap_freq is not None
            else adjacency.astype(float)
        )
        self.arrow_strength: Optional[np.ndarray] = None
        self.falsification_results: Optional[dict] = None

        # 构建 NetworkX DiGraph
        self.graph = nx.DiGraph()
        for i, label in enumerate(node_labels):
            self.graph.add_node(label, idx=i)
        for i in range(self.K):
            for j in range(self.K):
                if adjacency[i, j] > 0:
                    self.graph.add_edge(
                        node_labels[i], node_labels[j],
                        weight=float(self.bootstrap_freq[i, j]),
                    )

    @classmethod
    def from_freq_matrix(
        cls,
        freq_matrix: np.ndarray,
        node_labels: list[str],
        threshold: float = 0.5,
    ) -> "CausalCellGraph":
        """从频率矩阵创建（自动阈值剪枝）"""
        adjacency = (freq_matrix >= threshold).astype(float)
        return cls(adjacency, node_labels, bootstrap_freq=freq_matrix)

    # ----- 图统计 -----

    def summary_stats(self) -> dict:
        """图统计摘要"""
        n_edges = int(self.adjacency.sum())
        max_possible = self.K * (self.K - 1)
        sparsity = n_edges / max(max_possible, 1)
        freq_vals = self.bootstrap_freq[self.adjacency > 0]

        stats = {
            "n_nodes": self.K,
            "n_edges": n_edges,
            "max_possible_edges": max_possible,
            "graph_sparsity": float(sparsity),
            "mean_bootstrap_freq": float(freq_vals.mean()) if len(freq_vals) > 0 else 0.0,
            "median_bootstrap_freq": float(np.median(freq_vals)) if len(freq_vals) > 0 else 0.0,
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }
        return stats

    # ----- DoWhy 验证 -----

    def _make_dag(self) -> nx.DiGraph:
        """将图转为 DAG（通过移除使环最小化的边）"""
        dag = self.graph.copy()
        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag)
                # 移除环中 bootstrap 频率最低的边
                min_edge = min(
                    cycle,
                    key=lambda e: self.graph[e[0]][e[1]].get("weight", 0),
                )
                dag.remove_edge(min_edge[0], min_edge[1])
            except nx.NetworkXNoCycle:
                break
        return dag

    def compute_arrow_strength(
        self,
        data: pd.DataFrame,
        num_samples: int = 2000,
    ) -> np.ndarray:
        """使用 DoWhy 计算 arrow strength

        Parameters
        ----------
        data : DataFrame，列名对应 node_labels

        Returns
        -------
        (K, K) arrow strength 矩阵
        """
        try:
            import dowhy.gcm as gcm

            # DoWhy 需要 DAG，先去环
            dag = self._make_dag()
            causal_model = gcm.ProbabilisticCausalModel(dag)

            # 为每个节点自动分配因果机制
            gcm.auto.assign_causal_mechanisms(causal_model, data)
            gcm.fit(causal_model, data)

            strength_matrix = np.zeros((self.K, self.K))
            for target_label in self.node_labels:
                parents = list(dag.predecessors(target_label))
                if not parents:
                    continue
                try:
                    strengths = gcm.arrow_strength(
                        causal_model, target_label,
                        num_samples_conditional=num_samples,
                    )
                    for (src, tgt), val in strengths.items():
                        src_idx = self.node_labels.index(src)
                        tgt_idx = self.node_labels.index(tgt)
                        strength_matrix[src_idx, tgt_idx] = float(val)
                except Exception as e:
                    warnings.warn(f"Arrow strength failed for {target_label}: {e}")

            self.arrow_strength = strength_matrix
            return strength_matrix

        except ImportError:
            warnings.warn("dowhy not installed, skipping arrow strength")
            self.arrow_strength = self.bootstrap_freq.copy()
            return self.arrow_strength

    def validate_structure(
        self,
        data: pd.DataFrame,
        significance_level: float = 0.05,
    ) -> dict:
        """使用 DoWhy 进行因果结构可证伪检验

        Parameters
        ----------
        data : DataFrame，列名对应 node_labels
        significance_level : 显著性水平

        Returns
        -------
        dict with: rejected (bool), pvalue (float), details
        """
        try:
            from dowhy.gcm import refute_causal_structure

            dag = self._make_dag()
            result, summary = refute_causal_structure(
                dag,
                data,
                significance_level=significance_level,
            )

            # 提取 p-value 信息
            pvalues = []
            for node, tests in summary.items():
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and "p_value" in test_result:
                        pvalues.append(test_result["p_value"])

            min_pvalue = min(pvalues) if pvalues else 1.0
            mean_pvalue = float(np.mean(pvalues)) if pvalues else 1.0

            self.falsification_results = {
                "rejected": str(result).lower().find("rejected") >= 0
                            and str(result).lower().find("not_rejected") < 0,
                "result_str": str(result),
                "min_pvalue": min_pvalue,
                "mean_pvalue": mean_pvalue,
                "n_tests": len(pvalues),
                "details": {str(k): str(v) for k, v in summary.items()},
            }
            return self.falsification_results

        except ImportError:
            warnings.warn("dowhy not installed, skipping structural validation")
            self.falsification_results = {
                "rejected": False,
                "result_str": "skipped (dowhy not available)",
                "min_pvalue": 1.0,
                "mean_pvalue": 1.0,
                "n_tests": 0,
                "details": {},
            }
            return self.falsification_results

        except Exception as e:
            warnings.warn(f"DoWhy validation error: {e}")
            self.falsification_results = {
                "rejected": False,
                "result_str": f"error: {e}",
                "min_pvalue": float("nan"),
                "mean_pvalue": float("nan"),
                "n_tests": 0,
                "details": {},
            }
            return self.falsification_results

    # ----- 已知轴评估 -----

    def evaluate_known_axes(
        self,
        known_axes: Optional[list[dict]] = None,
        type_mapping: Optional[dict[str, str]] = None,
    ) -> dict:
        """评估因果图对已知信号轴的召回与方向准确度

        Parameters
        ----------
        known_axes : 已知轴列表
        type_mapping : {node_label: cell_type} 映射

        Returns
        -------
        dict with: recall, direction_accuracy, per_axis_results
        """
        if known_axes is None:
            known_axes = CRC_KNOWN_AXES

        if type_mapping is None:
            # 假设 node_labels 本身就是 cell type
            type_mapping = {label: label for label in self.node_labels}

        # 反向映射: cell_type → list of node_labels
        type_to_nodes: dict[str, list[str]] = {}
        for label, ctype in type_mapping.items():
            type_to_nodes.setdefault(ctype, []).append(label)

        per_axis = []
        total_found = 0
        total_correct_direction = 0
        total_axes = 0

        for axis in known_axes:
            src_type = axis["source_type"]
            tgt_type = axis["target_type"]

            src_nodes = type_to_nodes.get(src_type, [])
            tgt_nodes = type_to_nodes.get(tgt_type, [])

            if not src_nodes or not tgt_nodes:
                per_axis.append({
                    "name": axis["name"],
                    "found": False,
                    "correct_direction": False,
                    "reason": f"Missing node type: src={src_type}, tgt={tgt_type}",
                })
                continue

            total_axes += 1

            # 检查是否存在 src→tgt 的边（任意 src/tgt 节点组合）
            found_forward = False
            found_reverse = False
            max_freq_forward = 0.0
            max_freq_reverse = 0.0

            for s in src_nodes:
                for t in tgt_nodes:
                    s_idx = self.node_labels.index(s)
                    t_idx = self.node_labels.index(t)
                    if self.adjacency[s_idx, t_idx] > 0:
                        found_forward = True
                        max_freq_forward = max(
                            max_freq_forward, self.bootstrap_freq[s_idx, t_idx]
                        )
                    if self.adjacency[t_idx, s_idx] > 0:
                        found_reverse = True
                        max_freq_reverse = max(
                            max_freq_reverse, self.bootstrap_freq[t_idx, s_idx]
                        )

            found = found_forward or found_reverse
            correct_direction = found_forward and not found_reverse

            if found:
                total_found += 1
            if correct_direction:
                total_correct_direction += 1

            strength = 0.0
            if self.arrow_strength is not None:
                for s in src_nodes:
                    for t in tgt_nodes:
                        s_idx = self.node_labels.index(s)
                        t_idx = self.node_labels.index(t)
                        strength = max(
                            strength, self.arrow_strength[s_idx, t_idx]
                        )

            per_axis.append({
                "name": axis["name"],
                "source_type": src_type,
                "target_type": tgt_type,
                "found": found,
                "found_forward": found_forward,
                "found_reverse": found_reverse,
                "correct_direction": correct_direction,
                "max_freq_forward": float(max_freq_forward),
                "max_freq_reverse": float(max_freq_reverse),
                "arrow_strength": float(strength),
                "evidence": axis.get("evidence", ""),
            })

        recall = total_found / max(total_axes, 1)
        direction_acc = total_correct_direction / max(total_found, 1)

        return {
            "known_axis_recall": float(recall),
            "direction_accuracy": float(direction_acc),
            "n_axes_tested": total_axes,
            "n_axes_found": total_found,
            "n_correct_direction": total_correct_direction,
            "per_axis": per_axis,
        }

    # ----- 导出 -----

    def to_graphml(self, path: str | Path) -> None:
        """导出为 GraphML（仅保留可序列化属性）"""
        # 创建干净的图副本（去除 DoWhy 内部对象等不可序列化属性）
        clean_graph = nx.DiGraph()
        for node, data in self.graph.nodes(data=True):
            clean_attrs = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float, str, bool))
            }
            clean_graph.add_node(node, **clean_attrs)

        for i in range(self.K):
            for j in range(self.K):
                if self.adjacency[i, j] > 0:
                    src, tgt = self.node_labels[i], self.node_labels[j]
                    attrs = {
                        "bootstrap_freq": float(self.bootstrap_freq[i, j]),
                    }
                    if self.arrow_strength is not None:
                        attrs["arrow_strength"] = float(self.arrow_strength[i, j])
                    clean_graph.add_edge(src, tgt, **attrs)

        nx.write_graphml(clean_graph, str(path))

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典"""
        return {
            "node_labels": self.node_labels,
            "adjacency": self.adjacency.tolist(),
            "bootstrap_freq": self.bootstrap_freq.tolist(),
            "arrow_strength": (
                self.arrow_strength.tolist()
                if self.arrow_strength is not None
                else None
            ),
            "stats": self.summary_stats(),
        }
