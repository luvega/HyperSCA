from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
