# HyperSCA Plain-Language Terminology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HyperSCA's maintained user-facing text understandable to mixed biomedical readers while preserving standard terminology, scientific claim boundaries, and all stable code interfaces.

**Architecture:** Add one repository-wide writing rule in `AGENTS.md`, one human-readable terminology guide, and one small checker that verifies the first prose use of selected computing terms is accompanied by a plain-language explanation. Rewrite only maintained entry points: README, five current research-policy documents, and five command-line programs; preserve historical reports, citations, identifiers, arguments, filenames, and output fields.

**Tech Stack:** Markdown, Python 3.10, `argparse`, JSON configuration, `pytest`, standard-library `pathlib`, `json`, and `re`.

---

## File map

**Create**

- `AGENTS.md` — repository rules plus the permanent reader-facing language policy.
- `docs/style/plain_language_terminology.md` — authoritative terminology guide, examples, and scientific claim safeguards.
- `configs/plain_language_terms.json` — maintained file list and machine-readable term-to-explanation rules.
- `src/quality/__init__.py` — public exports for documentation checks.
- `src/quality/plain_language.py` — prose extraction and first-use explanation checker.
- `scripts/check_plain_language.py` — thin command-line entry point for the checker.
- `tests/test_plain_language_policy.py` — policy, glossary, compatibility, and maintained-document assertions.
- `tests/test_plain_language_checker.py` — checker unit and command-line tests.
- `tests/test_plain_language_cli.py` — command help and friendly error tests.

**Modify**

- `README.md` — plain-language project overview, release status, comparison explanation, and links.
- `docs/research/target_discovery_ranking_policy_v1.md` — explain direct-evidence ranking and supplementary evidence.
- `docs/research/causal_null_control_policy_v1.md` — explain repeated sampling and zero-effect controls.
- `docs/research/benchmark_contract_v1.md` — explain pre-fixed comparison rules and evidence-stage decisions.
- `docs/research/task_c_mean_difference_baseline_v1.md` — explain intervention-network comparison in biomedical terms.
- `docs/research/task_s_simple_baselines_v1.md` — explain own-cell and neighboring-cell effects separately.
- `scripts/run_target_discovery.py` — Chinese research-oriented help for each existing argument and output label.
- `scripts/run_causal_stability_audit.py` — explain stability checks and zero-effect controls.
- `scripts/validate_benchmark_contract.py` — explain validation of pre-fixed comparison rules.
- `scripts/run_task_c_mean_difference.py` — explain external intervention data, reference relationships, and smoke-only runs.
- `scripts/run_task_s_baseline.py` — explain independent spatial validation, own/neighbor effects, and leakage attestations.
- `tests/test_project_configuration.py` — require the plain-language check in CI.
- `.github/workflows/ci.yml` — run the checker before the full test suite.

**Explicitly unchanged**

- All `src/` function, class, module, parameter, and dictionary-key names except for adding the isolated `src/quality/` package.
- Existing command-line option names.
- Existing JSON/CSV field names, file names, directory names, and contract digest content.
- `docs/releases/`, Bear reports, paper titles, quotations, and generated reports.

---

### Task 1: Add the permanent repository rule and terminology guide

**Files:**

- Create: `AGENTS.md`
- Create: `docs/style/plain_language_terminology.md`
- Modify: `README.md`
- Create: `tests/test_plain_language_policy.py`

- [ ] **Step 1: Write failing policy-presence tests**

Add `tests/test_plain_language_policy.py` with these initial tests:

```python
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
```

- [ ] **Step 2: Run the tests and verify they fail because the files are absent**

Run:

```bash
pytest tests/test_plain_language_policy.py -q -p no:cacheprovider
```

Expected: three failures caused by missing `AGENTS.md`, missing terminology guide, and missing README link.

- [ ] **Step 3: Create `AGENTS.md` without dropping the existing repository guidance**

Copy the repository structure, build/test, coding style, testing, commit/PR, and security sections from the user-provided repository guidance. Append this exact policy section:

```markdown
## 面向读者的语言规范

README、研究文档、命令行帮助、错误提示、运行摘要和图注应优先面向生物医学、临床和实验读者。先说明研究含义、用途和证据边界，再在首次出现时用括号保留标准术语，例如“未参与模型建立的独立验证数据（holdout）”。

避免只使用软件工程或机器学习行话。`pipeline`、`artifact`、`sidecar`、`benchmark contract`、`promotion gate`、`coverage` 和 `abstention` 等词首次出现在用户可见文字中时，必须给出通俗解释。代码名、命令参数、文件名和输出字段保持不变，并在附近解释其含义。

通俗化不得改变科学结论：不得把相关写成因果，不得把预测或计算模拟写成实验验证，不得把候选靶点写成有效治疗靶点，也不得提高证据等级。完整规则和首选表达见 `docs/style/plain_language_terminology.md`。
```

- [ ] **Step 4: Create the full terminology guide**

Create `docs/style/plain_language_terminology.md` with these sections and content:

```markdown
# HyperSCA 项目术语与表达指南

## 基本写法

正文采用“通俗名称（标准术语）”。先回答这一步研究什么、为什么需要、结果能说明到什么程度。标准术语供检索和专业沟通使用，后文在不产生歧义时可只用通俗名称。

## 兼容性边界

不重命名 Python 名称、命令参数、JSON/CSV 字段、文件或目录。文献标题、引文和第三方软件正式名称保持原文。

## 科学边界

不得把相关改写成因果，不得把预测或模拟写成实验验证，不得把候选写成有效治疗，也不得把补充证据写成决定性证据。
```

Then include the complete term table from the approved design for: `pipeline`, `workflow`, `benchmark`, `benchmark contract`, `baseline`, `holdout`, `artifact`, `manifest`, `sidecar`, `adapter`, `promotion gate`, `evidence-gated ranking`, `coverage`, `abstention`, `null control`, `bootstrap`, `split`, `tuning budget`, `smoke test`, `fail closed`, `CLI`, `schema`, and `digest/hash`.

End with these positive and negative examples:

```markdown
## 示例

推荐：因果稳定性检查会重复抽样，观察同一条关系是否反复出现。结果作为独立的补充分析（sidecar）保存，不改变候选靶点排序。

不推荐：Causal sidecar 生成 bootstrap artifact，不进入 ranking pipeline。

推荐：该最小运行检查（smoke test）只证明分析流程可以运行，不代表方法已经得到真实数据验证。

不推荐：Smoke test passed，因此模型有效。
```

- [ ] **Step 5: Link the guide from README**

Under the release/report links in `README.md`, add:

```markdown
- [项目术语与表达指南](docs/style/plain_language_terminology.md)：说明本文档如何用通俗文字解释标准术语，并保持科学证据边界。
```

- [ ] **Step 6: Run the policy tests**

Run:

```bash
pytest tests/test_plain_language_policy.py -q -p no:cacheprovider
```

Expected: 3 passed.

- [ ] **Step 7: Commit the policy and guide**

```bash
git add AGENTS.md docs/style/plain_language_terminology.md README.md tests/test_plain_language_policy.py
git commit -m "docs: add plain-language terminology policy"
```

---

### Task 2: Add a focused first-use terminology checker

**Files:**

- Create: `configs/plain_language_terms.json`
- Create: `src/quality/__init__.py`
- Create: `src/quality/plain_language.py`
- Create: `scripts/check_plain_language.py`
- Create: `tests/test_plain_language_checker.py`
- Modify: `tests/test_project_configuration.py`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Write failing unit tests for prose extraction and first-use checks**

Create `tests/test_plain_language_checker.py`:

```python
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
```

- [ ] **Step 2: Run the checker tests and verify import/config failures**

Run:

```bash
pytest tests/test_plain_language_checker.py -q -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError: src.quality`.

- [ ] **Step 3: Create the machine-readable rules**

Create `configs/plain_language_terms.json` with this structure:

```json
{
  "schema_version": "1.0",
  "files": [
    "README.md",
    "docs/research/target_discovery_ranking_policy_v1.md",
    "docs/research/causal_null_control_policy_v1.md",
    "docs/research/benchmark_contract_v1.md",
    "docs/research/task_c_mean_difference_baseline_v1.md",
    "docs/research/task_s_simple_baselines_v1.md"
  ],
  "rules": [
    {"term": "benchmark contract", "aliases": ["benchmark contract", "benchmark contracts"], "preferred_phrases": ["预先固定的比较规则"]},
    {"term": "promotion gate", "preferred_phrases": ["进入下一证据阶段的条件"]},
    {"term": "evidence-gated", "preferred_phrases": ["由直接证据决定"]},
    {"term": "holdout", "aliases": ["holdout", "holdouts"], "preferred_phrases": ["独立验证数据"]},
    {"term": "artifact", "aliases": ["artifact", "artifacts"], "preferred_phrases": ["分析输出文件", "可复查的输出"]},
    {"term": "manifest", "aliases": ["manifest", "manifests"], "preferred_phrases": ["分析记录清单"]},
    {"term": "sidecar", "aliases": ["sidecar", "sidecars"], "preferred_phrases": ["独立的补充分析"]},
    {"term": "baseline", "aliases": ["baseline", "baselines"], "preferred_phrases": ["简单对照方法"]},
    {"term": "coverage", "preferred_phrases": ["可作出判断的比例"]},
    {"term": "abstention", "preferred_phrases": ["暂不判断的比例", "暂不判断"]},
    {"term": "smoke test", "preferred_phrases": ["最小运行检查"]},
    {"term": "pipeline", "preferred_phrases": ["分析流程"]},
    {"term": "adapter", "aliases": ["adapter", "adapters"], "preferred_phrases": ["接入层"]}
  ]
}
```

- [ ] **Step 4: Implement prose extraction and first-use validation**

Create `src/quality/__init__.py`:

```python
from src.quality.plain_language import (
    PlainLanguageIssue,
    check_configured_documents,
    check_document,
    strip_non_prose,
)

__all__ = [
    "PlainLanguageIssue",
    "check_configured_documents",
    "check_document",
    "strip_non_prose",
]
```

Create `src/quality/plain_language.py` with these public interfaces and behavior:

```python
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
```

- [ ] **Step 5: Add the thin checker command**

Create `scripts/check_plain_language.py`:

```python
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
```

- [ ] **Step 6: Run unit tests and observe repository-scope failures**

Run:

```bash
pytest tests/test_plain_language_checker.py -q -p no:cacheprovider
```

Expected: extraction and synthetic first-use tests pass; `test_checker_cli_passes_repository_scope` fails and prints the maintained documents that still need rewriting.

- [ ] **Step 7: Require the checker in project configuration and CI tests**

Append to `tests/test_project_configuration.py`:

```python
def test_ci_runs_plain_language_check_before_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    check = "python scripts/check_plain_language.py"
    tests = "pytest tests -q"
    assert check in workflow
    assert workflow.index(check) < workflow.index(tests)
```

In `.github/workflows/ci.yml`, insert this step after environment validation and before tests:

```yaml
      - name: Check reader-facing terminology
        run: python scripts/check_plain_language.py
```

- [ ] **Step 8: Keep the checker changes uncommitted until the maintained documents pass**

Do not commit or publish an intentionally failing repository-scope check. Proceed directly to Task 3; its final commit includes the checker, rewritten maintained documents, tests, and CI wiring as one green change.

---

### Task 3: Rewrite the maintained README and research-policy documents

**Files:**

- Modify: `README.md`
- Modify: `docs/research/target_discovery_ranking_policy_v1.md`
- Modify: `docs/research/causal_null_control_policy_v1.md`
- Modify: `docs/research/benchmark_contract_v1.md`
- Modify: `docs/research/task_c_mean_difference_baseline_v1.md`
- Modify: `docs/research/task_s_simple_baselines_v1.md`
- Modify: `tests/test_plain_language_policy.py`

- [ ] **Step 1: Add failing claim-boundary and first-use assertions**

Append to `tests/test_plain_language_policy.py`:

```python
MAINTAINED_DOCS = [
    "README.md",
    "docs/research/target_discovery_ranking_policy_v1.md",
    "docs/research/causal_null_control_policy_v1.md",
    "docs/research/benchmark_contract_v1.md",
    "docs/research/task_c_mean_difference_baseline_v1.md",
    "docs/research/task_s_simple_baselines_v1.md",
]


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
```

Run:

```bash
pytest tests/test_plain_language_policy.py -q -p no:cacheprovider
```

Expected: the exact new conservative phrases are absent and the first test fails.

- [ ] **Step 2: Rewrite the README's maintained user-facing prose**

Apply these concrete changes throughout `README.md` while leaving commands, paths, and identifiers unchanged:

| Existing wording | First plain-language use |
|---|---|
| research pipeline | 多步骤分析流程（pipeline） |
| audit baseline | 用于复查现有证据的固定版本 |
| benchmark contract | 预先固定的比较规则（benchmark contract） |
| baseline | 简单对照方法（baseline） |
| holdout | 未参与模型建立的独立验证数据（holdout） |
| sidecar | 独立的补充分析（sidecar） |
| artifact | 可复查的分析输出文件（artifact） |
| manifest | 分析记录清单（manifest） |
| promotion | 进入下一证据阶段 |

Translate user-facing headings such as `Current Release Status`, `Project and Algorithm Overview`, `Benchmark Sidecar and Module Selection`, `Core Algorithms`, and `Installation Guide` into clear Chinese, with the English term only where it aids search. Preserve the release links, code blocks, filenames, and all cautionary claims.

- [ ] **Step 3: Rewrite the five current research documents using a common opening**

Each document must open with four short items in this order:

```markdown
## 这份文件解决什么问题

[One or two sentences in biomedical language.]

## 通俗解释

[Explain the standard term on first use.]

## 不能据此得出什么结论

[Preserve the existing scientific boundary.]

## 机器读取名称

[List stable filenames and fields without renaming them.]
```

Use these document-specific required meanings:

- Ranking policy: direct expression evidence determines order; geometry, causal, spatial, perturbation, and mechanism results remain supplementary and do not change candidate rank.
- Causal null policy: repeated sampling asks whether a relationship reappears; zero-effect controls ask whether random rearrangements appear equally convincing; matrix surrogates are not intervention-level evidence.
- C/S/D contract: rules are fixed before results; C tests intervention networks, S separates own-cell and neighbor effects, D evaluates drug-response association with exposure and target-engagement evidence.
- Task C: mean difference is a deliberately simple intervention-aware comparison floor; reference networks are incomplete evaluation evidence, not complete causal truth.
- Task S: both simple methods share the same own-effect input; the first predicts no neighbor effect, and the second applies a fixed distance decline; good performance does not identify a mechanism.

Include these exact boundary sentences across the relevant documents:

```text
这些补充结果不改变候选靶点排序。
最小运行检查只证明分析流程可以运行，不代表方法已在真实数据中得到验证。
即使达到进入下一证据阶段的条件，也不能据此声称临床疗效、已验证药物机制或普适算法优势。
```

- [ ] **Step 4: Run the terminology checker and document tests**

Run:

```bash
python scripts/check_plain_language.py
pytest tests/test_plain_language_policy.py tests/test_plain_language_checker.py -q -p no:cacheprovider
```

Expected: checker exits 0; all policy/checker tests pass.

- [ ] **Step 5: Review the diff for scientific drift**

Run:

```bash
git diff --word-diff=plain -- README.md docs/research/target_discovery_ranking_policy_v1.md docs/research/causal_null_control_policy_v1.md docs/research/benchmark_contract_v1.md docs/research/task_c_mean_difference_baseline_v1.md docs/research/task_s_simple_baselines_v1.md
```

Confirm all of the following before committing:

- no `promoted`, `validated`, `causal`, or treatment claim became stronger;
- no numeric threshold, metric, seed, split rule, or method formula changed;
- no code block, path, filename, field, DOI, or external link changed;
- own-cell and neighbor effects remain separate;
- synthetic runs remain smoke-only.

- [ ] **Step 6: Commit the maintained-document rewrite**

```bash
git add \
  README.md \
  docs/research/target_discovery_ranking_policy_v1.md \
  docs/research/causal_null_control_policy_v1.md \
  docs/research/benchmark_contract_v1.md \
  docs/research/task_c_mean_difference_baseline_v1.md \
  docs/research/task_s_simple_baselines_v1.md \
  configs/plain_language_terms.json \
  src/quality \
  scripts/check_plain_language.py \
  tests/test_plain_language_policy.py \
  tests/test_plain_language_checker.py \
  tests/test_project_configuration.py \
  .github/workflows/ci.yml
git commit -m "docs: explain active research terms in plain language"
```

---

### Task 4: Rewrite command help and present actionable errors

**Files:**

- Modify: `scripts/run_target_discovery.py`
- Modify: `scripts/run_causal_stability_audit.py`
- Modify: `scripts/validate_benchmark_contract.py`
- Modify: `scripts/run_task_c_mean_difference.py`
- Modify: `scripts/run_task_s_baseline.py`
- Create: `tests/test_plain_language_cli.py`

- [ ] **Step 1: Write failing command-help tests**

Create `tests/test_plain_language_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script, required_phrases",
    [
        ("run_target_discovery.py", ["候选靶点", "直接证据", "分析记录清单"]),
        ("run_causal_stability_audit.py", ["重复抽样", "零效应对照"]),
        ("validate_benchmark_contract.py", ["预先固定的比较规则", "不会运行模型"]),
        ("run_task_c_mean_difference.py", ["干预数据", "简单对照方法"]),
        ("run_task_s_baseline.py", ["自身效应", "邻近细胞效应", "独立验证数据"]),
    ],
)
def test_help_explains_research_purpose(script: str, required_phrases: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for phrase in required_phrases:
        assert phrase in completed.stdout


def test_task_c_missing_input_gives_actionable_error(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_task_c_mean_difference.py"),
            "--input-npz",
            str(tmp_path / "missing.npz"),
            "--dataset-id",
            "missing",
            "--dataset-source",
            "test",
            "--context-id",
            "test",
            "--data-status",
            "synthetic_smoke",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "无法继续" in completed.stderr
    assert "请检查" in completed.stderr
```

- [ ] **Step 2: Run the tests and verify the current English/technical help fails**

Run:

```bash
pytest tests/test_plain_language_cli.py -q -p no:cacheprovider
```

Expected: six failures because the required Chinese explanations and actionable wrapper are absent.

- [ ] **Step 3: Rewrite parser descriptions and every argument help string**

Use these descriptions exactly:

```python
# run_target_discovery.py
description=(
    "从表达、空间位置和补充机制证据中整理候选靶点。默认排序只由直接证据决定，"
    "并保存分析记录清单（manifest），方便复查输入和输出。"
)

# run_causal_stability_audit.py
description=(
    "通过重复抽样检查因果关系是否稳定，并用随机重排形成零效应对照。"
    "这些结果只提供补充证据，不改变候选靶点排序。"
)

# validate_benchmark_contract.py
description=(
    "检查任务 C/S/D 的预先固定比较规则是否完整。该命令不会运行模型，"
    "也不会产生方法优越性的结论。"
)

# run_task_c_mean_difference.py
description=(
    "在单细胞干预数据上运行均值差简单对照方法，用于判断更复杂的因果网络方法"
    "是否真正超过直接利用干预标签的结果。"
)

# run_task_s_baseline.py
description=(
    "在独立验证数据上分别评估自身效应和邻近细胞效应。两个简单对照方法使用"
    "相同的自身效应输入，以便单独判断空间信息是否带来增益。"
)
```

Add a Chinese `help=` to every existing argument. Each help string must state at least one of: research meaning, units, default behavior, evidence effect, or whether it is only a file location. Do not rename an option or change any default, choice, type, or action.

Use the following argument explanations. Keep the English option names so existing commands remain valid.

| Script | Option | Exact `help=` text |
|---|---|---|
| `run_target_discovery.py` | `--output-dir` | `保存分析结果的根目录。默认写入 results/discovery/target_discovery。` |
|  | `--run-id` | `本次分析的名称；不填写时自动生成。` |
|  | `--max-perturb` | `最多评估多少个候选干预对象，默认 50。` |
|  | `--geometry-k` | `描述局部空间关系时，每个位置采用的近邻数量，默认 4。` |
|  | `--geometry-blend` | `局部与整体空间信息的合并比例，默认 0.30。` |
|  | `--platform` | `选择分析的空间测量平台；all 表示分析全部已配置平台。` |
|  | `--hierarchy-levels` | `空间结构分层的层数，默认 3。` |
|  | `--skip-figures` | `只生成数据和报告，不绘制图形。` |
|  | `--device` | `执行计算所用设备，默认 cuda；没有可用显卡时可设为 cpu。` |
|  | `--score-profile` | `候选排序规则。evidence_gated 只用直接证据排序；legacy_full 仅用于复现旧结果。` |
| `run_causal_stability_audit.py` | `--step2-dir` | `第二步因果分析结果所在目录，默认 results/step2。` |
|  | `--output-dir` | `稳定性检查结果的保存目录；不填写时写入第二步结果目录下的 causal_audit。` |
|  | `--threshold` | `将关系计为重复出现所需的最低频率，默认 0.5。` |
|  | `--n-null-controls` | `每种随机零效应对照的重复次数；0 表示不运行随机对照。` |
|  | `--null-modes` | `以逗号分隔的随机零效应对照类型；这些对照只用于检查稳定性，不改变候选排序。` |
|  | `--random-seed` | `控制重复抽样的随机起点，默认 42；相同输入和数值可复现结果。` |
| `validate_benchmark_contract.py` | `--contract` | `预先固定的比较规则文件，默认 configs/benchmark_contract_v1.json。` |
|  | `--output-dir` | `可选的规则快照保存目录；不填写时只检查，不写快照。` |
| `run_task_c_mean_difference.py` | `--input-npz` | `CausalBench 生成的三数组 NPZ 输入文件。` |
|  | `--dataset-id` | `本次使用的数据集标识，写入结果记录以便追溯。` |
|  | `--dataset-source` | `数据来源说明，写入结果记录以便追溯。` |
|  | `--context-id` | `本次评估的细胞、组织或实验情境标识。` |
|  | `--data-status` | `external_benchmark 表示独立外部数据；synthetic_smoke 只表示合成流程检查。` |
|  | `--output-dir` | `保存指标、预测和分析记录的目录。` |
|  | `--contract` | `预先固定的比较规则文件，默认 configs/benchmark_contract_v1.json。` |
|  | `--reference-edges` | `可选的参考关系表 CSV；提供时必须同时提供 --reference-id。` |
|  | `--reference-id` | `参考关系表的来源标识；提供时必须同时提供 --reference-edges。` |
|  | `--source-column` | `参考关系表中起点对象的列名，默认 source。` |
|  | `--target-column` | `参考关系表中终点对象的列名，默认 target。` |
|  | `--control-label` | `输入数据中未定向干预对照组的标签，默认 non-targeting。` |
|  | `--excluded-label` | `输入数据中不参与评估的样本标签，默认 excluded。` |
|  | `--min-cells-per-intervention` | `每个干预至少需要的细胞数，默认 5；不足时该干预不参与估计。` |
|  | `--precision-at-k` | `计算前 k 个预测关系精确率时采用的 k，默认 1000。` |
|  | `--random-seed` | `控制可重复计算的随机起点，默认 11。` |
|  | `--code-revision` | `可选的代码版本标识；不填写时自动读取当前 Git 版本。` |
| `run_task_s_baseline.py` | `--input-csv` | `独立验证数据 CSV，需包含任务 S 规定的自身效应、邻近效应和空间分组字段。` |
|  | `--baseline-id` | `简单对照方法：own_only 只用自身效应；fixed_distance_decay 再加入固定距离衰减。` |
|  | `--dataset-id` | `本次使用的数据集标识，写入结果记录以便追溯。` |
|  | `--dataset-source` | `数据来源说明，写入结果记录以便追溯。` |
|  | `--data-status` | `external_benchmark 表示独立外部数据；synthetic_smoke 只表示合成流程检查。` |
|  | `--own-effect-source-id` | `自身效应预测的来源标识，用于核对两个简单对照是否使用相同输入。` |
|  | `--own-effect-source` | `自身效应预测来源文件；程序记录其校验值，但不从中读取验证结果。` |
|  | `--length-scale` | `固定距离衰减的长度尺度；own_only 不使用该值。` |
|  | `--length-scale-source-id` | `长度尺度来源的标识；使用 fixed_distance_decay 时按比较规则提供。` |
|  | `--length-scale-source` | `记录 length_scale 的 JSON 文件；程序核对数值并记录校验值。` |
|  | `--attest-own-effect-train-only` | `确认自身效应预测未使用独立验证集的结果。` |
|  | `--attest-nonadjacent-spatial-blocks` | `确认验证空间分区与训练空间分区不相邻。` |
|  | `--contract` | `预先固定的比较规则文件，默认 configs/benchmark_contract_v1.json。` |
|  | `--code-revision` | `可选的代码版本标识；不填写时自动读取当前 Git 版本。` |
|  | `--random-seed` | `控制可重复计算的随机起点，默认 11。` |
|  | `--output-dir` | `保存指标、预测和分析记录的目录。` |

Change output labels without changing dictionary keys:

```python
print(f"本次分析目录：{outputs['run_dir']}")
print(f"分析记录清单：{outputs['manifest_path']}")
print(f"可阅读报告：{outputs['target_discovery_report']}")
```

For JSON summaries, preserve keys and values; do not translate machine-readable fields.

- [ ] **Step 4: Add actionable error wrappers to the three validation commands**

In `validate_benchmark_contract.py`, `run_task_c_mean_difference.py`, and `run_task_s_baseline.py`, first replace `args = build_parser().parse_args(argv)` with:

```python
parser = build_parser()
args = parser.parse_args(argv)
```

Import `BenchmarkContractError` beside the existing contract functions. Then make these exact changes:

1. In `validate_benchmark_contract.py`, put only `contract = load_benchmark_contract(args.contract)` inside this wrapper, leaving summary construction after it:

```python
try:
    contract = load_benchmark_contract(args.contract)
except BenchmarkContractError as exc:
    parser.error(
        f"无法继续：{exc}。请检查输入文件、字段和参数是否符合文档中的数据规范。"
    )
```

2. In `run_task_c_mean_difference.py`, start a `try` immediately before the paired `--reference-edges`/`--reference-id` check. Indent that check, `load_causalbench_npz(...)`, optional reference loading, contract loading, and `run_task_c_mean_difference(...)` through the assignment to `run`. Put the following wrapper before construction of `summary`:

```python
except (TaskCBenchmarkError, BenchmarkContractError) as exc:
    parser.error(
        f"无法继续：{exc}。请检查输入文件、字段和参数是否符合文档中的数据规范。"
    )
```

3. In `run_task_s_baseline.py`, start a `try` immediately before `holdout = pd.read_csv(args.input_csv)`. Indent the CSV loading, contract loading, optional length-scale validation, and `run_task_s_baseline(...)` through the assignment to `run`. Put the following wrapper before construction of `summary`:

```python
except (TaskSBenchmarkError, BenchmarkContractError) as exc:
    parser.error(
        f"无法继续：{exc}。请检查输入文件、字段和参数是否符合文档中的数据规范。"
    )
```

Do not catch `KeyboardInterrupt`, `SystemExit`, `OSError`, parser exits, or unexpected programming errors. The existing inner `OSError` handling for the length-scale JSON remains part of its conversion to `TaskSBenchmarkError`.

- [ ] **Step 5: Run command-help and existing command tests**

Run:

```bash
pytest tests/test_plain_language_cli.py tests/test_task_c_benchmark.py tests/test_task_s_benchmark.py tests/test_benchmark_contract.py tests/test_causal_stability_audit.py -q -p no:cacheprovider
```

Expected: all tests pass. Existing assertions prove argument names, defaults, fields, and artifact behavior remain compatible.

- [ ] **Step 6: Commit the command-line rewrite**

```bash
git add scripts/run_target_discovery.py scripts/run_causal_stability_audit.py scripts/validate_benchmark_contract.py scripts/run_task_c_mean_difference.py scripts/run_task_s_baseline.py tests/test_plain_language_cli.py
git commit -m "docs: clarify command help and errors"
```

---

### Task 5: Complete consistency and compatibility verification

**Files:**

- Modify only if verification exposes a wording or checker defect.

- [ ] **Step 1: Run the terminology checker directly**

```bash
python scripts/check_plain_language.py
```

Expected output:

```text
通俗术语检查通过：所有维护文档均在首次使用时解释了标准术语。
```

- [ ] **Step 2: Capture command interfaces before and after if an origin-main comparison is needed**

Run the current branch help commands and inspect option names:

```bash
for script in \
  run_target_discovery.py \
  run_causal_stability_audit.py \
  validate_benchmark_contract.py \
  run_task_c_mean_difference.py \
  run_task_s_baseline.py; do
  python "scripts/$script" --help >/tmp/"$script".help
done
```

Then verify the option sets against `origin/main` using:

```bash
git diff --unified=0 origin/main -- scripts | rg '^[+-]\s*parser\.add_argument' || true
```

Expected: no added, deleted, or renamed `parser.add_argument` option line. Help text additions may change surrounding lines only.

- [ ] **Step 3: Verify the frozen comparison contract did not change**

Run:

```bash
git diff --exit-code origin/main -- configs/benchmark_contract_v1.json
python scripts/validate_benchmark_contract.py
```

Expected: no diff; contract status is valid and its SHA-256 remains `718569e91ef92fa6e642c69cbabd091e24df9996470e70e9246b8781c857e187`.

- [ ] **Step 4: Run focused reader-facing tests**

```bash
pytest \
  tests/test_plain_language_policy.py \
  tests/test_plain_language_checker.py \
  tests/test_plain_language_cli.py \
  tests/test_project_configuration.py \
  -q -p no:cacheprovider
```

Expected: all focused tests pass.

- [ ] **Step 5: Run the full regression suite**

```bash
pytest tests -q -p no:cacheprovider
```

Expected: all tests pass; pre-existing dependency deprecation warnings are allowed, test failures are not.

- [ ] **Step 6: Check formatting and scope**

```bash
git diff --check origin/main...HEAD
git status --short
git diff --stat origin/main...HEAD
```

Expected: no whitespace errors; only the files listed in this plan are changed; the worktree is clean after commits.

- [ ] **Step 7: Perform the four-sample human review**

Read these exact samples in rendered order:

```bash
sed -n '1,140p' README.md
sed -n '1,180p' docs/research/causal_null_control_policy_v1.md
sed -n '1,220p' docs/research/task_s_simple_baselines_v1.md
python scripts/run_task_c_mean_difference.py --help
```

Confirm each sample states purpose, method, and limitation before implementation jargon. Confirm no text implies clinical validation, verified mechanism, or universal superiority.

- [ ] **Step 8: Commit any verification-only wording corrections**

If Step 7 required corrections, commit only those exact files:

```bash
git add \
  README.md \
  docs/research/target_discovery_ranking_policy_v1.md \
  docs/research/causal_null_control_policy_v1.md \
  docs/research/benchmark_contract_v1.md \
  docs/research/task_c_mean_difference_baseline_v1.md \
  docs/research/task_s_simple_baselines_v1.md \
  scripts/run_target_discovery.py \
  scripts/run_causal_stability_audit.py \
  scripts/validate_benchmark_contract.py \
  scripts/run_task_c_mean_difference.py \
  scripts/run_task_s_baseline.py
git commit -m "docs: refine plain-language claim boundaries"
```

If no corrections were needed, do not create an empty commit.

- [ ] **Step 9: Push a review branch and open one documentation-focused PR**

```bash
git push --set-upstream origin HEAD:refs/heads/docs/plain-language-terminology
gh pr create \
  --base main \
  --head docs/plain-language-terminology \
  --title "docs: explain HyperSCA terminology in plain language" \
  --body "## Purpose

Explain maintained HyperSCA documentation and commands for mixed biomedical readers while retaining standard terms once for search and professional communication.

## Scope and compatibility

- adds the permanent AGENTS.md language rule and terminology guide
- rewrites README and the five active ranking/null/C-S-D policy documents
- clarifies five command-line help surfaces and actionable errors
- keeps Python names, command options, output fields, contract contents, historical reports, and evidence levels unchanged

## Verification

- plain-language terminology check: passed
- focused reader-facing tests: passed
- full test suite: passed
- frozen contract digest: 718569e91ef92fa6e642c69cbabd091e24df9996470e70e9246b8781c857e187"
```

The PR body must report the terminology-check result, focused/full test counts, unchanged contract digest, unchanged argument names, maintained-document scope, and the explicit statement that historical reports and scientific evidence levels were not rewritten.

- [ ] **Step 10: Merge only after protected Python 3.10 CI passes**

Use a squash merge, then wait for the post-merge `main` CI run. Report the merge commit, CI URL, and remaining scope: future documents are governed by `AGENTS.md`; archival documents remain unchanged.
