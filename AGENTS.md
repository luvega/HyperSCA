# Repository Guidelines

## Project Structure & Module Organization

HyperSCA is a Python/R research pipeline for single-cell, spatial omics, causal inference, perturbation analysis, and target discovery. Core Python packages live in `src/`, with major domains under `src/models/`, `src/causal/`, `src/perturbation/`, `src/discovery/target_discovery/`, `src/behavior_grammar/`, and `src/data/`. CLI and workflow entrypoints are in `scripts/`. Tests mirror source areas under `tests/`, including focused suites such as `tests/discovery/` and `tests/behavior_grammar/`. Example notebooks are in `notebooks/`; documentation and figures are in `docs/`; generated analyses belong under `results/` or `reports/`.

## Build, Test, and Development Commands

Create the main environment and install dependencies:

```bash
conda create -n hypersca python=3.10 -y
conda activate hypersca
pip install -r requirements.txt
```

Validate the runtime with `python scripts/validate_env.py`. Run tests with `pytest tests -q -p no:cacheprovider`, or target a subsystem with `pytest tests/discovery -q` and `pytest tests/behavior_grammar -q`. Run target discovery locally with:

```bash
python scripts/run_target_discovery.py --run-id demo_target_discovery --max-perturb 10 --skip-figures
```

## Coding Style & Naming Conventions

Use Python 3.10-compatible code, 4-space indentation, descriptive `snake_case` names for functions and modules, and `PascalCase` for classes. Keep CLI scripts thin: parse arguments in `scripts/`, and place reusable logic in `src/`. Prefer typed dataclasses and explicit artifact paths for pipeline interfaces. Keep generated outputs out of source modules.

## Testing Guidelines

Use `pytest`. Add tests close to the affected subsystem, following `test_*.py` naming. For pipeline changes, include focused unit tests plus at least one integration-style test for manifest or artifact behavior. Avoid tests that require private datasets unless guarded by small synthetic fixtures.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style messages, such as `feat: ...`, `docs: ...`, `refactor: ...`, and `chore: ...`. PRs should describe the scientific or pipeline motivation, list changed entrypoints or artifacts, mention required data or environment assumptions, and include the exact tests run. Attach screenshots only for figure or notebook changes.

## Security & Configuration Tips

Do not commit private patient-level data, large generated result folders, local model weights, or machine-specific paths. Keep credentials and external dataset roots outside version control; use `configs/` only for shareable defaults.

## 面向读者的语言规范

README、研究文档、命令行帮助、错误提示、运行摘要和图注应优先面向生物医学、临床和实验读者。先说明研究含义、用途和证据边界，再在首次出现时用括号保留标准术语，例如“未参与模型建立的独立验证数据（holdout）”。

避免只使用软件工程或机器学习行话。`pipeline`、`artifact`、`sidecar`、`benchmark contract`、`promotion gate`、`coverage` 和 `abstention` 等词首次出现在用户可见文字中时，必须给出通俗解释。代码名、命令参数、文件名和输出字段保持不变，并在附近解释其含义。

通俗化不得改变科学结论：不得把相关写成因果，不得把预测或计算模拟写成实验验证，不得把候选靶点写成有效治疗靶点，也不得提高证据等级。完整规则和首选表达见 `docs/style/plain_language_terminology.md`。
