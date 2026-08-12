from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PlainLanguageIssue:
    path: str
    term: str
    line: int
    message: str


def strip_non_prose(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return text


def _paragraphs_with_lines(text: str) -> list[tuple[int, str]]:
    paragraphs: list[tuple[int, str]] = []
    start = 1
    buffer: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not buffer:
                start = line_number
            buffer.append(line)
        elif buffer:
            paragraphs.append((start, "\n".join(buffer)))
            buffer = []
    if buffer:
        paragraphs.append((start, "\n".join(buffer)))
    return paragraphs


def check_document(
    text: str,
    rules: Sequence[Mapping[str, Any]],
    path: str,
) -> list[PlainLanguageIssue]:
    prose = strip_non_prose(text)
    paragraphs = _paragraphs_with_lines(prose)
    issues: list[PlainLanguageIssue] = []
    for rule in rules:
        term = str(rule["term"])
        aliases = [str(value) for value in rule.get("aliases", [term])]
        alternatives = "|".join(re.escape(value) for value in aliases)
        pattern = re.compile(
            rf"(?<![\w-])(?:{alternatives})(?![\w-])",
            re.IGNORECASE,
        )
        for line, paragraph in paragraphs:
            if not pattern.search(paragraph):
                continue
            preferred = [str(value) for value in rule["preferred_phrases"]]
            if not any(phrase in paragraph for phrase in preferred):
                issues.append(
                    PlainLanguageIssue(
                        path=path,
                        term=term,
                        line=line,
                        message=(
                            f"首次使用 {term!r} 时，请在同一段说明："
                            + " / ".join(preferred)
                        ),
                    )
                )
            break
    return issues


def check_configured_documents(
    root: Path,
    config_path: Path,
) -> list[PlainLanguageIssue]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0":
        raise ValueError("术语检查配置的 schema_version 必须为 1.0。")
    issues: list[PlainLanguageIssue] = []
    for relative in config["files"]:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"维护范围中的文档不存在：{relative}")
        issues.extend(
            check_document(path.read_text(encoding="utf-8"), config["rules"], relative)
        )
    return issues
