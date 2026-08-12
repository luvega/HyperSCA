"""检查当前维护文档中的标准术语是否在首次出现时得到通俗解释。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.quality.plain_language import check_configured_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查当前维护文档是否先解释研究含义，再使用标准术语。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "plain_language_terms.json",
        help="术语规则和待检查文档的清单。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    issues = check_configured_documents(ROOT, args.config)
    if issues:
        for issue in issues:
            print(f"{issue.path}:{issue.line}: {issue.message}")
        return 1
    print("通俗术语检查通过：所有维护文档均在首次使用时解释了标准术语。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
