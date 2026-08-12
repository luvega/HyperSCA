from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_DOCS = [
    "README.md",
    "docs/research/target_discovery_ranking_policy_v1.md",
    "docs/research/causal_null_control_policy_v1.md",
    "docs/research/benchmark_contract_v1.md",
    "docs/research/task_c_mean_difference_baseline_v1.md",
    "docs/research/task_s_simple_baselines_v1.md",
]


def test_agents_requires_plain_language_for_user_facing_text() -> None:
    policy = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "面向读者的语言规范" in policy
    assert "先说明研究含义" in policy
    assert "不得改变科学结论" in policy
    assert "docs/style/plain_language_terminology.md" in policy


def test_plain_language_guide_preserves_interfaces_and_claim_boundaries() -> None:
    guide = (ROOT / "docs/style/plain_language_terminology.md").read_text(
        encoding="utf-8"
    )
    assert "通俗名称（标准术语）" in guide
    assert "不重命名" in guide
    assert "不得把相关改写成因果" in guide
    assert "预先固定的比较规则" in guide
    assert "独立的补充分析" in guide


def test_readme_links_the_plain_language_guide() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/style/plain_language_terminology.md" in readme
    assert "项目术语与表达指南" in readme


def test_maintained_documents_keep_conservative_scientific_language() -> None:
    combined = "\n".join(
        (ROOT / path).read_text(encoding="utf-8") for path in MAINTAINED_DOCS
    )
    assert "不能据此声称临床疗效" in combined
    assert "不改变候选靶点排序" in combined
    assert "只证明分析流程可以运行" in combined
    assert "own effect" in combined
    assert "neighbor effect" in combined


def test_stable_interface_names_remain_documented() -> None:
    combined = "\n".join(
        (ROOT / path).read_text(encoding="utf-8") for path in MAINTAINED_DOCS
    )
    for stable_name in (
        "promotion_status",
        "contract_sha256",
        "run_manifest.json",
        "input_summary.json",
        "metrics.json",
        "predictions.csv",
        "promotion_decision.json",
    ):
        assert stable_name in combined
