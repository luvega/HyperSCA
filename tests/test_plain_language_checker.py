from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.quality.plain_language import check_document, strip_non_prose


ROOT = Path(__file__).resolve().parents[1]
RULE = {
    "term": "benchmark contract",
    "aliases": ["benchmark contract", "benchmark contracts"],
    "preferred_phrases": ["预先固定的比较规则"],
}


def test_strip_non_prose_ignores_code_and_link_targets() -> None:
    text = "正文。`benchmark contract`\n```bash\nbenchmark contract\n```\n[链接](benchmark contract)"
    cleaned = strip_non_prose(text)
    assert "benchmark contract" not in cleaned
    assert "正文" in cleaned


def test_first_prose_use_requires_plain_explanation_in_same_paragraph() -> None:
    issues = check_document("The benchmark contract is fixed.", [RULE], "x.md")
    assert len(issues) == 1
    assert issues[0].term == "benchmark contract"


def test_alias_plural_is_also_checked() -> None:
    issues = check_document("Benchmark contracts are fixed.", [RULE], "x.md")
    assert len(issues) == 1


def test_explained_first_use_passes_and_later_short_use_is_allowed() -> None:
    text = (
        "采用预先固定的比较规则（benchmark contract），防止临时改变标准。\n\n"
        "The benchmark contract remains frozen."
    )
    assert check_document(text, [RULE], "x.md") == []


def test_ignored_code_block_does_not_shift_reported_line_number() -> None:
    text = (
        "开头。\n\n"
        "```bash\n"
        "benchmark contract\n"
        "第二行代码\n"
        "```\n\n"
        "The benchmark contract is fixed.\n"
    )
    issues = check_document(text, [RULE], "x.md")
    assert len(issues) == 1
    assert issues[0].line == 8


def test_checker_cli_passes_repository_scope() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_plain_language.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "通俗术语检查通过" in completed.stdout


def test_configuration_is_versioned_and_has_explicit_scope() -> None:
    config = json.loads(
        (ROOT / "configs/plain_language_terms.json").read_text(encoding="utf-8")
    )
    assert config["schema_version"] == "1.0"
    assert "README.md" in config["files"]
    assert len(config["rules"]) >= 12
