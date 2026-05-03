"""MSI 基因签名推断模块单元测试。"""
import numpy as np
import pytest

from src.evaluation.msi_inference import (
    MSI_HIGH_GENES,
    MSS_HIGH_GENES,
    compute_msi_score,
    infer_status_from_score,
)


def test_infer_status_from_score() -> None:
    assert infer_status_from_score(0.9) == ("MSI", "high")
    assert infer_status_from_score(0.5) == ("MSI", "medium")
    assert infer_status_from_score(0.35) == ("MSI", "low")
    assert infer_status_from_score(0.0) == ("Unknown", "low")
    assert infer_status_from_score(-0.35) == ("MSS", "low")
    assert infer_status_from_score(-0.6) == ("MSS", "medium")
    assert infer_status_from_score(-1.0) == ("MSS", "high")
    assert infer_status_from_score(np.nan) == ("Unknown", "none")


def test_compute_msi_score_ideal() -> None:
    """MSI-like 表达：MSI 基因高、MSS 基因低 → 正得分。"""
    n_obs, n_genes = 100, 50
    var_names = list(MSI_HIGH_GENES[:5]) + list(MSS_HIGH_GENES[:5]) + [f"G{i}" for i in range(40)]
    X = np.random.rand(n_obs, n_genes) * 2
    # MSI 基因列 (0-4) 抬高
    X[:, :5] += 3.0
    # MSS 基因列 (5-9) 压低
    X[:, 5:10] *= 0.3

    res = compute_msi_score(X, var_names)
    assert res["n_msi_found"] >= 1
    assert res["n_mss_found"] >= 1
    assert res["score"] > 0
    assert res["mean_msi"] > res["mean_mss"]


def test_compute_msi_score_mss_like() -> None:
    """MSS-like 表达：MSS 基因高、MSI 基因低 → 负得分。"""
    n_obs, n_genes = 100, 50
    var_names = list(MSI_HIGH_GENES[:5]) + list(MSS_HIGH_GENES[:5]) + [f"G{i}" for i in range(40)]
    X = np.random.rand(n_obs, n_genes) * 2
    X[:, :5] *= 0.2
    X[:, 5:10] += 4.0

    res = compute_msi_score(X, var_names)
    assert res["score"] < 0
    assert res["mean_mss"] > res["mean_msi"]


def test_compute_msi_score_no_genes() -> None:
    """无匹配基因时返回 nan。"""
    X = np.random.rand(10, 5)
    var_names = ["A", "B", "C", "D", "E"]
    res = compute_msi_score(X, var_names)
    assert np.isnan(res["score"])
    assert res["n_msi_found"] == 0
    assert res["n_mss_found"] == 0


def test_gene_signature_constants() -> None:
    assert len(MSI_HIGH_GENES) >= 10
    assert len(MSS_HIGH_GENES) >= 10
    assert "CD8A" in MSI_HIGH_GENES
    assert "COL1A1" in MSS_HIGH_GENES
