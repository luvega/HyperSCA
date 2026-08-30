# HyperSCA Bear Report Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Bear/SciMaster 的真实检索结果补强 HyperSCA 现有进度与研究版图报告，并保留可复核的查询、原始结果、证据账本和 BibTeX。

**Architecture:** 原始检索产物与正文编辑解耦：15 条查询先写入独立 `raw/` 目录，再经查询账本和证据账本去重、分层，生成 Bear Markdown/HTML 附件，最后只把通过证据门控的结论融合进主报告。主报告继续保留本地工程审计事实和 E0–E3 因果语言，不从 Bear 结果推导未经干预验证的机制结论。

**Tech Stack:** `scimaster-cli 0.3.15`、JSON、BibTeX、TSV、Markdown、单文件 HTML、Bash、`jq`、`rg`、Git。

---

## File Map

- Create: `reports/research/bear_hypersca_spatial_causal_20260810/raw/` — 15 次 `sci search` 的 JSON/BibTeX 原始结果。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/query_manifest.tsv` — 查询账本。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/evidence_ledger.json` — 去重后的证据账本与分类判断。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv` — 升级方向、比较器、公平性约束、主终点与状态。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/innovation_claim_register.tsv` — 框架/评价/算法创新的证据要求与允许措辞。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/report.md` — Bear 综合证据附件。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/report.html` — 自包含可视化附件。
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/references.bib` — 附件实际引用的去重 BibTeX。
- Modify: `reports/research/hypersca_causal_spatial_drug_landscape_20260810.md` — 融合撞车、安静区、挑战、查询透明度与新证据。

### Task 1: Freeze Scope and Verify the Search Runtime

**Files:**
- Reference: `docs/superpowers/specs/2026-08-10-hypersca-bear-report-enhancement-design.md`
- Reference: `reports/research/hypersca_causal_spatial_drug_landscape_20260810.md`
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/raw/`

- [ ] **Step 1: Verify CLI, authentication, quota, and available modes**

Run:

```bash
sci --version
sci usage
sci model --show
```

Expected: version `0.3.15`; authentication succeeds; quota is printed. The current default may be `high`, but all batch commands below must pass an explicit `--mode ultra_low` or `--mode low`.

- [ ] **Step 2: Create the evidence workspace**

Run:

```bash
mkdir -p reports/research/bear_hypersca_spatial_causal_20260810/raw
```

Expected: the directory exists and the main HyperSCA report remains unchanged.

- [ ] **Step 3: Record the pre-search report checksum and Git status**

Run:

```bash
sha256sum reports/research/hypersca_causal_spatial_drug_landscape_20260810.md
git status --short -- reports/research/hypersca_causal_spatial_drug_landscape_20260810.md reports/research/bear_hypersca_spatial_causal_20260810
```

Expected: one checksum line; no Bear artifact exists before collection. Preserve the checksum in the execution log.

### Task 2: Run the Collision Scan

**Files:**
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/raw/q01-*` through `q11-*`

- [ ] **Step 1: Run six orthogonal `ultra_low` scans**

Run each command independently and retain both generated files:

```bash
sci search "single-cell spatial transcriptomics causal inference counterfactual drug target discovery integrated platform" --mode ultra_low --limit 20 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q01-literal
sci search "causal graph counterfactual spatial propagation perturbation modeling single-cell transcriptomics" --mode ultra_low --limit 20 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q02-method
sci search "predict cell-autonomous and neighborhood perturbation effects spatial transcriptomics" --mode ultra_low --limit 20 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q03-problem
sci search "spatial niche specific drug mechanism target prioritization transcriptomics" --mode ultra_low --limit 20 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q04-conclusion
sci search "virtual tissue digital twin multicellular response drug perturbation spatial omics" --mode ultra_low --limit 20 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q05-adjacent
sci search "2025 2026 spatial perturbation causal single-cell drug response preprint" --mode ultra_low --limit 20 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q06-frontier
```

Expected: q01–q06 each produce one timestamped JSON and one timestamped BibTeX file. Empty results are retained and recorded rather than replaced.

- [ ] **Step 2: Re-run the five decision-critical angles with `low` ranking**

Run:

```bash
sci search "single-cell spatial transcriptomics causal inference counterfactual drug target discovery integrated platform" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q07-literal-low
sci search "causal graph counterfactual spatial propagation perturbation modeling single-cell transcriptomics" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q08-method-low
sci search "predict cell-autonomous and neighborhood perturbation effects spatial transcriptomics" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q09-problem-low
sci search "spatial niche specific drug mechanism target prioritization transcriptomics" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q10-conclusion-low
sci search "2025 2026 spatial perturbation causal single-cell drug response preprint" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q11-frontier-low
```

Expected: q07–q11 each produce JSON and BibTeX; no command uses `mid` or `high`.

- [ ] **Step 3: Confirm the first 11 query pairs are complete and valid**

Run:

```bash
find reports/research/bear_hypersca_spatial_causal_20260810/raw -maxdepth 1 -type f -name 'q*.json' | sort
find reports/research/bear_hypersca_spatial_causal_20260810/raw -maxdepth 1 -type f -name 'q*.bib' | sort
jq empty reports/research/bear_hypersca_spatial_causal_20260810/raw/q*.json
```

Expected: 11 JSON files, 11 BibTeX files, and `jq` exits 0.

### Task 3: Search the Quiet Zones and Challenges

**Files:**
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/raw/q12-*` through `q15-*`

- [ ] **Step 1: Search support for evidence-gated joint own/neighbor evaluation**

Run:

```bash
sci search "evidence-gated benchmarking joint cell-autonomous neighborhood perturbation effects target prioritization spatial single-cell" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q12-quiet-evidence-gate
```

Expected: q12 JSON/BibTeX exist. Papers supporting only one component are classified as partial, not direct support.

- [ ] **Step 2: Search support for pharmacological spatial validation**

Run:

```bash
sci search "spatially resolved pharmacological perturbation dose time target engagement neighborhood drug mechanism transcriptomics" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q13-quiet-pharmacology
```

Expected: q13 JSON/BibTeX exist; purely genetic perturbation studies cannot be labeled direct pharmacological support.

- [ ] **Step 3: Search direct contradictions and failure evidence**

Run:

```bash
sci search "failure poor generalization spatial transcriptomics drug response prediction causal mechanism perturbation" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q14-challenge-contradiction
```

Expected: q14 JSON/BibTeX exist; null or weak evidence is retained as a result.

- [ ] **Step 4: Search methodological criticism**

Run:

```bash
sci search "identifiability confounding data leakage benchmark criticism causal inference counterfactual single-cell perturbation spatial transcriptomics" --mode low --limit 15 --out reports/research/bear_hypersca_spatial_causal_20260810/raw --prefix q15-challenge-method
```

Expected: q15 JSON/BibTeX exist and contains candidate evidence on identifiability, generalization, leakage, or validation limits.

- [ ] **Step 5: Verify all 15 searches and record post-search quota**

Run:

```bash
jq empty reports/research/bear_hypersca_spatial_causal_20260810/raw/q*.json
sci usage
```

Expected: 15 valid JSON files and a successful quota response. Report the balance change without claiming that `ultra_low` was cost-free if the displayed balance changed.

### Task 4: Build the Query and Evidence Ledgers

**Files:**
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/query_manifest.tsv`
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/evidence_ledger.json`
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv`
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/innovation_claim_register.tsv`

- [ ] **Step 1: Write the query manifest**

Write a tab-separated file with this exact header:

```text
query_id	phase	label	query	mode	result_count	useful_count	json_file	bib_file
```

Add q01–q15 in order. `result_count` comes from JSON array length; `useful_count` is the number admitted to the evidence ledger after abstract-level screening. Paths are relative to the Bear artifact directory.

- [ ] **Step 2: Deduplicate and classify candidate evidence**

Create `evidence_ledger.json` as a JSON object containing `topic`, `generated_at`, `queries`, and `evidence`. Each evidence record must contain:

```json
{
  "id": "E1",
  "title": "A title returned by this session's sci search",
  "authors": ["Author names returned by sci search"],
  "year": 2025,
  "doi": "normalized DOI or empty string",
  "url": "source URL or empty string",
  "abstract": "abstract returned by sci search or empty string",
  "query_ids": ["q01", "q07"],
  "collision_layer": "direct_collision | method_twin | problem_twin | neighbor | none",
  "evidence_role": "quiet_support | challenge | context",
  "strength": "strong | medium | weak",
  "admission_reason": "one sentence tied to the HyperSCA idea",
  "response_action": "benchmark, negative control, promotion gate, limitation, or empty string"
}
```

Deduplicate first by lowercase DOI stripped of URL prefixes, then by normalized title when DOI is absent. Do not merge records with materially different years or authors solely because their titles are similar.

- [ ] **Step 3: Validate the ledgers**

Before validation, write `comparison_matrix.tsv` with one or more rows for Task C, single-cell counterfactuals, Task S, and Task D. Its exact header is:

```text
task	upgrade_target	comparator	dataset_or_holdout	fairness_constraints	primary_endpoint	reliability_requirements	current_status	evidence_ids
```

Write `innovation_claim_register.tsv` with this exact header:

```text
claim_id	claim_type	claim_text	closest_prior_work	comparator	primary_endpoint	evidence_required	current_status	allowed_wording
```

`claim_type` is `framework`, `evaluation`, or `algorithm`. An algorithm row may use wording stronger than “候选创新” only when `current_status` records completed same-task comparison, external holdout, multi-seed uncertainty, ablation, negative control, calibration, and failure analysis.

Run:

```bash
jq empty reports/research/bear_hypersca_spatial_causal_20260810/evidence_ledger.json
awk -F '\t' 'NR == 1 { if (NF != 9) exit 1 } NR > 1 { if (NF != 9) exit 1 } END { print NR-1 " queries" }' reports/research/bear_hypersca_spatial_causal_20260810/query_manifest.tsv
awk -F '\t' 'NR > 1 { seen[$1] = 1; if (NF != 9) exit 1 } END { if (!("Task C" in seen) || !("single-cell counterfactual" in seen) || !("Task S" in seen) || !("Task D" in seen)) exit 1 }' reports/research/bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv
awk -F '\t' 'NR > 1 { if (NF != 9) exit 1; seen[$2] = 1 } END { if (!("framework" in seen) || !("evaluation" in seen) || !("algorithm" in seen)) exit 1 }' reports/research/bear_hypersca_spatial_causal_20260810/innovation_claim_register.tsv
```

Expected: JSON validation succeeds, the query manifest prints `15 queries`, all four comparison tasks exist, and all three innovation types exist.

### Task 5: Produce the Bear Markdown, HTML, and BibTeX Artifacts

**Files:**
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/report.md`
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/report.html`
- Create: `reports/research/bear_hypersca_spatial_causal_20260810/references.bib`

- [ ] **Step 1: Write `report.md` using the fixed Bear structure**

Use YAML front matter with `skill: bear-propose`, the original topic, UTC `generated_at`, `query_count: 15`, computed result counts, all 15 query records, and output filenames. Use the fixed sections `## 1.` through `## 7.`. In `## 2. 核心发现`, include the nearest collision, four collision layers, relative quiet zones, quiet-zone support ladder, two challenge types, and a 3–5 sentence integrated judgment that does not decide whether the project is “worth doing.” Add a dual-track table that links each evidence gap to a reversible HyperSCA upgrade and a fair same-task comparison.

- [ ] **Step 2: Merge only cited BibTeX records**

Build `references.bib` from raw q01–q15 `.bib` files. Preserve SciMaster keys, deduplicate by DOI/title, and include only E-records cited in `report.md`.

- [ ] **Step 3: Write the self-contained `report.html`**

Use accent `#d97706`, no external fonts/scripts/CDNs, and include: three-part verdict, four metrics, collision radar, three tabs, primary evidence cards, translated Chinese abstracts, DOI links, all query coverage, and empty-result reporting. Use only integer font weights `400/600/700/800`; do not use the contradictory nonstandard `650` values shown in the generic template.

- [ ] **Step 4: Validate the Bear artifacts**

Run:

```bash
test -s reports/research/bear_hypersca_spatial_causal_20260810/report.md
test -s reports/research/bear_hypersca_spatial_causal_20260810/report.html
test -s reports/research/bear_hypersca_spatial_causal_20260810/references.bib
rg -n '^## [1-7]\.' reports/research/bear_hypersca_spatial_causal_20260810/report.md
rg -n 'https://doi.org/' reports/research/bear_hypersca_spatial_causal_20260810/report.html
rg -n 'https?://[^" ]+\.(js|css)' reports/research/bear_hypersca_spatial_causal_20260810/report.html && exit 1 || true
```

Expected: all three files are non-empty; seven Markdown sections exist; DOI links exist; no external JS/CSS dependency is present.

### Task 6: Integrate Bear Evidence into the Main HyperSCA Report

**Files:**
- Modify: `reports/research/hypersca_causal_spatial_drug_landscape_20260810.md`

- [ ] **Step 1: Update metadata, abstract, and search methods**

Add a revision marker and link to the Bear appendix. State the 15-query design, explicit modes, result/admission counts, DOI/title deduplication, and the evidence boundary. Add only one abstract sentence summarizing collision risk, quiet zones, and the highest threat.

- [ ] **Step 2: Add the Bear integrated evidence section**

Insert a section after the external research matrices containing: four-layer collision table, crowded versus quiet zones, quiet-zone support table, challenge-response matrix, and an interpretation paragraph. Every row must cite an E-ID from the Bear appendix and a DOI/source link when available.

Add a separate “创新性与可靠性主张边界” table. Distinguish framework, evaluation, and algorithm contributions; show closest prior work, comparator, primary endpoint, evidence status, and wording currently permitted.

- [ ] **Step 3: Propagate accepted evidence into gaps, roadmap, and promotion gates**

Convert admitted high-threat challenges into specific additions to Task C/S/D, P0–P3, or the promotion policy. Keep pharmacological spatial validation distinct from genetic spatial perturbation and preserve `audit_only_no_promotion` unless new E3 evidence directly changes it.

For every proposed upgrade, state the matching comparator, shared data split, preprocessing/feature panel, tuning budget, primary endpoint, uncertainty requirement, ablation, negative control, calibration, failure analysis, and rollback condition. Prioritize data/benchmark adapters before adding model complexity.

- [ ] **Step 4: Update limitations, claim-evidence map, disclosure, and references**

Add limitations for relative quiet-zone inference, database/index coverage, SciMaster ranking, and query-language bias. Add every new major claim to the claim-evidence table. Disclose `bear-propose`, `scimaster-cli 0.3.15`, modes, query count, and AI-assisted synthesis. Add only referenced papers to the bibliography.

- [ ] **Step 5: Reverse-outline the revised sections**

For the abstract, Bear integrated section, roadmap changes, and conclusion, record: thesis, each paragraph topic sentence, and evidence IDs. Remove or weaken any paragraph that cannot map topic sentence → thesis and evidence → topic sentence.

### Task 7: Run Final Evidence and Markdown QA

**Files:**
- Verify: `reports/research/hypersca_causal_spatial_drug_landscape_20260810.md`
- Verify: `reports/research/bear_hypersca_spatial_causal_20260810/*`

- [ ] **Step 1: Validate JSON, query count, modes, and artifact presence**

Run:

```bash
jq empty reports/research/bear_hypersca_spatial_causal_20260810/raw/q*.json reports/research/bear_hypersca_spatial_causal_20260810/evidence_ledger.json
test "$(find reports/research/bear_hypersca_spatial_causal_20260810/raw -maxdepth 1 -name 'q*.json' | wc -l)" -eq 15
test "$(find reports/research/bear_hypersca_spatial_causal_20260810/raw -maxdepth 1 -name 'q*.bib' | wc -l)" -eq 15
! rg -n $'\thigh\t|\tmid\t' reports/research/bear_hypersca_spatial_causal_20260810/query_manifest.tsv
test -s reports/research/bear_hypersca_spatial_causal_20260810/comparison_matrix.tsv
test -s reports/research/bear_hypersca_spatial_causal_20260810/innovation_claim_register.tsv
```

Expected: all commands exit 0.

- [ ] **Step 2: Validate Markdown structure and local links**

Run `git diff --check` on the two Markdown reports. Check paired code fences, duplicate DOI values, empty evidence anchors, and all local relative links. Any failure blocks completion.

- [ ] **Step 3: Validate claim-evidence alignment**

Confirm every newly added claim appears in `evidence_ledger.json`, every cited E-ID appears in the Bear report, and every Bear report reference exists in `references.bib`. Downgrade unsupported causal language before completion.

- [ ] **Step 4: Run the five-dimension adversarial self-review**

Answer in the main report: contribution clarity, writing clarity, evidence strength, evaluation completeness, and method-design soundness. Resolve every high-risk issue or state it explicitly as a limitation.

- [ ] **Step 5: Inspect the final diff without committing report changes**

Run:

```bash
git diff --stat -- reports/research/hypersca_causal_spatial_drug_landscape_20260810.md reports/research/bear_hypersca_spatial_causal_20260810
git status --short -- reports/research/hypersca_causal_spatial_drug_landscape_20260810.md reports/research/bear_hypersca_spatial_causal_20260810
```

Expected: only the approved report and Bear artifact directory are changed. Do not commit these deliverables unless the user separately requests it.
