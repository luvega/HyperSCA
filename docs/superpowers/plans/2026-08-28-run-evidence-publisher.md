# RunEvidencePublisher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one standard-library evidence publisher that atomically seals run artifacts, separates data-split and model-seed identities, and rejects unpaired cross-seed collections before scientific aggregation.

**Architecture:** Split the implementation into immutable identity primitives, publication/replay, and paired-collection validation, while re-exporting the public API from `src.evaluation.run_evidence_publisher`. Runners provide benchmark-specific statistical-unit records; the publisher hashes and seals them without importing scientific/model code. OSTA and CausalBench adapters consume the same committed publisher source, but no release or private holdout is executed.

**Tech Stack:** Python 3.10 standard library, pytest, Hypothesis, Import Linter, Linux `renameat2(RENAME_NOREPLACE)` through `ctypes`.

---

## File map

- Create `src/evaluation/run_evidence_identity.py`: strict JSON validation, canonical hashes, immutable `RunEvidenceIdentity`.
- Create `src/evaluation/run_evidence_publisher.py`: public facade, artifact staging, terminal publication, replay verifier.
- Create `src/evaluation/run_evidence_collection.py`: paired-seed closure and invalidation records.
- Create `tests/test_run_evidence_identity.py`: exact-type and identity tests.
- Create `tests/test_run_evidence_publisher.py`: state machine, filesystem and replay tests.
- Create `tests/test_run_evidence_collection.py`: paired closure and pilot-regression tests.
- Create `tests/property/test_run_evidence_properties.py`: Hypothesis identity/path/state properties.
- Modify `.importlinter`: publisher independence contracts.
- Modify `src/evaluation/methods_pilot.py`: OSTA publication adapter.
- Modify `tests/test_methods_pilot.py`: OSTA integration evidence.
- Modify design-worktree `src/evaluation/methods_causal_pilot.py`: CausalBench publication adapter after cherry-pick.
- Modify design-worktree `tests/test_methods_causal_pilot.py`: CausalBench integration evidence.
- Create `reports/methodology/run_evidence_publisher_legacy_pilot_audit_20260828.md`: read-only compatibility report.

### Task 1: Immutable run identity

**Files:**
- Create: `src/evaluation/run_evidence_identity.py`
- Test: `tests/test_run_evidence_identity.py`

- [ ] **Step 1: Write failing exact-identity tests**

```python
class EvilInt(int):
    """Regression input proving int subclasses cannot cross identity boundaries."""


def valid_identity(**changes):
    values = {
        "schema_version": "1.0",
        "protocol_version": "hypersca-methods-v2.1",
        "protocol_identity": "a" * 64,
        "claim_id": "spatial",
        "benchmark_id": "osta_colon",
        "data_scopes": ("train", "tune"),
        "data_split_seed": 19911,
        "model_seed": 11,
        "data_split_identity_sha256": "b" * 64,
        "statistical_unit_schema": "osta_platform_sample_block_v1",
        "statistical_unit_identity_sha256": "c" * 64,
        "analysis_identity_sha256": "d" * 64,
        "input_identity_sha256": "e" * 64,
        "config_identity_sha256": "f" * 64,
        "code_identity_sha256": "0" * 64,
        "evidence_role": "pilot_audit_only",
    }
    values.update(changes)
    return RunEvidenceIdentity(**values)


def test_split_and_model_seed_are_distinct_identity_fields():
    first = valid_identity(data_split_seed=19911, model_seed=11)
    second = valid_identity(data_split_seed=19911, model_seed=23)
    changed_split = valid_identity(data_split_seed=23, model_seed=11)
    assert first.run_identity_sha256 != second.run_identity_sha256
    assert first.run_identity_sha256 != changed_split.run_identity_sha256


@pytest.mark.parametrize("bad", [True, 1.0, EvilInt(11)])
def test_seed_fields_require_exact_builtin_ints(bad):
    with pytest.raises(RunEvidenceError, match="invalid_identity"):
        valid_identity(model_seed=bad)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run --no-capture-output -n hypersca \
  python -m pytest tests/test_run_evidence_identity.py -q -p no:cacheprovider
```

Expected: collection error because `src.evaluation.run_evidence_identity` does not exist.

- [ ] **Step 3: Implement strict canonical identity**

Implement these public definitions:

```python
class RunEvidenceError(ValueError):
    def __init__(self, category: str, message: str):
        if category not in ERROR_CATEGORIES:
            raise ValueError("unknown run-evidence error category")
        self.category = category
        super().__init__(f"{category}: {message}")


def canonical_json_bytes(value: object) -> bytes:
    normalized = validate_strict_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RunEvidenceIdentity:
    schema_version: str
    protocol_version: str
    protocol_identity: str
    claim_id: str
    benchmark_id: str
    data_scopes: tuple[str, ...]
    data_split_seed: int
    model_seed: int
    data_split_identity_sha256: str
    statistical_unit_schema: str
    statistical_unit_identity_sha256: str
    analysis_identity_sha256: str
    input_identity_sha256: str
    config_identity_sha256: str
    code_identity_sha256: str
    evidence_role: str
    run_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        text_fields = (
            "schema_version",
            "protocol_version",
            "claim_id",
            "benchmark_id",
            "statistical_unit_schema",
            "evidence_role",
        )
        sha_fields = (
            "protocol_identity",
            "data_split_identity_sha256",
            "statistical_unit_identity_sha256",
            "analysis_identity_sha256",
            "input_identity_sha256",
            "config_identity_sha256",
            "code_identity_sha256",
        )
        for name in text_fields:
            require_exact_nfc_text(getattr(self, name), field_name=name)
        for name in sha_fields:
            require_sha256(getattr(self, name), field_name=name)
        require_exact_unique_text_tuple(self.data_scopes, field_name="data_scopes")
        if type(self.data_split_seed) is not int or self.data_split_seed < 0:
            raise RunEvidenceError(
                "invalid_identity", "data_split_seed must be a non-negative exact int"
            )
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise RunEvidenceError(
                "invalid_identity", "model_seed must be a non-negative exact int"
            )
        if self.evidence_role not in {
            "pilot_audit_only",
            "release_candidate",
            "infrastructure_smoke",
        }:
            raise RunEvidenceError("invalid_identity", "unknown evidence_role")
        object.__setattr__(
            self,
            "run_identity_sha256",
            hashlib.sha256(canonical_json_bytes(self.to_record())).hexdigest(),
        )
```

`validate_strict_json` must defensively copy mappings/sequences once, normalize no values, reject duplicate/lying iterables, reject non-NFC/control text, reject non-built-in scalar subclasses, and reject non-finite floats.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all identity tests pass.

- [ ] **Step 5: Commit identity primitives**

```bash
git add src/evaluation/run_evidence_identity.py tests/test_run_evidence_identity.py
git commit -m "feat: freeze run evidence identities"
```

### Task 2: Artifact staging and state machine

**Files:**
- Create: `src/evaluation/run_evidence_publisher.py`
- Test: `tests/test_run_evidence_publisher.py`

- [ ] **Step 1: Write failing staging/state tests**

```python
def test_publisher_stages_bytes_and_streams_one_regular_file(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")
    publisher = RunEvidencePublisher.begin(
        output_dir=tmp_path / "bundle",
        identity=valid_identity(),
        statistical_unit_record={"units": ["sample:block-1"]},
        required_artifacts=("metrics.json", "table.csv"),
        maximum_bundle_bytes=4096,
    )
    publisher.add_bytes("metrics.json", b"{}\n", media_type="application/json")
    publisher.add_file("table.csv", source, media_type="text/csv")
    assert not (tmp_path / "bundle").exists()


def test_finalized_publisher_cannot_write_or_finalize_again(tmp_path):
    publisher = completed_publisher(tmp_path)
    with pytest.raises(RunEvidenceError, match="invalid_state_transition"):
        publisher.add_bytes("late.txt", b"late", media_type="text/plain")
    with pytest.raises(RunEvidenceError, match="invalid_state_transition"):
        publisher.finalize_completed(summary={"status": "completed"})
```

Add tests rejecting absolute paths, `..`, NUL/control text, duplicate paths, symlink sources, multi-link sources, source changes during copy, bundle-size overflow, and exceptions/`KeyboardInterrupt` cleaning staging.

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n hypersca \
  python -m pytest tests/test_run_evidence_publisher.py -q -p no:cacheprovider
```

Expected: import failure for `RunEvidencePublisher`.

- [ ] **Step 3: Implement `begin`, `add_bytes`, and `add_file`**

Use this state enum and internal artifact record:

```python
class _PublisherState(Enum):
    STAGING = "staging"
    COMPLETED_PUBLISHED = "completed_published"
    FAILED_PUBLISHED = "failed_published"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    media_type: str
```

`begin` creates `.<name>.staging-<random>` with mode 0700 in the bound output parent. `add_file` opens with `O_NOFOLLOW`, requires regular/single-link, copies in 1 MiB chunks, hashes while copying, fsyncs the destination, and compares source `fstat` before/after. `abort()` is idempotent only from STAGING and recursively removes only the exact bound staging inode.

- [ ] **Step 4: Run publisher tests and verify GREEN**

Run the Step 2 command. Expected: staging/state tests pass.

- [ ] **Step 5: Commit staging state machine**

```bash
git add src/evaluation/run_evidence_publisher.py tests/test_run_evidence_publisher.py
git commit -m "feat: stage run evidence safely"
```

### Task 3: Terminal publication and replay verification

**Files:**
- Modify: `src/evaluation/run_evidence_publisher.py`
- Modify: `tests/test_run_evidence_publisher.py`

- [ ] **Step 1: Write failing completion/failure/replay tests**

```python
def test_completed_bundle_has_cross_bound_status_manifest_and_inventory(tmp_path):
    output = publish_completed_bundle(tmp_path)
    verified = verify_run_evidence_bundle(output)
    assert verified.identity == valid_identity()
    assert verified.terminal_status == "completed"
    assert set(verified.artifacts) == {"metrics.json", "table.csv"}


def test_failure_bundle_has_no_scientific_summary(tmp_path):
    publisher = empty_publisher(tmp_path, required_artifacts=())
    output = publisher.finalize_failure(
        status="failed_runtime", reason="worker exited before metrics"
    )
    verified = verify_run_evidence_bundle(output)
    assert verified.terminal_status == "failed_runtime"
    assert verified.summary is None
```

Add tamper tests for manifest, status, artifact bytes, artifact inventory, hardlinks, duplicate inode, extra file, missing file, symlink component, `10**400` size/duration, and synchronized JSON resealing without the original run identity.

- [ ] **Step 2: Run the new tests and verify RED**

Run the Task 2 test command. Expected: missing finalize/verifier APIs.

- [ ] **Step 3: Implement terminal records and exclusive publish**

Completed publication must generate:

```python
status = {
    "schema_version": "1.0",
    "status": "completed",
    "run_identity_sha256": identity.run_identity_sha256,
    "artifact_inventory_sha256": inventory_sha,
    "summary_sha256": canonical_sha256(summary),
}
manifest = {
    "schema_version": "1.0",
    "run_identity": identity.to_record(),
    "run_identity_sha256": identity.run_identity_sha256,
    "statistical_unit_record": statistical_unit_record,
    "artifacts": artifact_records,
    "terminal_status": status,
}
```

Failure publication records `summary=null`, an empty or explicitly permitted diagnostic artifact set, and one registered failure status. Fsync all files and staging directory, call libc `renameat2(..., RENAME_NOREPLACE)`, then fsync parent. Unsupported `renameat2` raises `publication_infrastructure`; it must not fall back to overwrite-capable rename.

- [ ] **Step 4: Implement `verify_run_evidence_bundle`**

Verifier requirements:

```python
def verify_run_evidence_bundle(path: Path) -> VerifiedRunEvidence:
    # bind directory fd/inode; reject symlink components
    # strict bounded JSON with duplicate-key rejection
    # reconstruct RunEvidenceIdentity and exact canonical SHA
    # require exact completed/failure file sets
    # open every artifact via dir_fd + O_NOFOLLOW
    # require regular, nlink == 1, distinct inode, exact size/SHA
    # cross-check status/manifest/inventory/summary
    # recheck directory and files before returning deep immutable evidence
```

- [ ] **Step 5: Run publisher tests and verify GREEN**

Run Task 2 test command. Expected: all publication/replay tests pass.

- [ ] **Step 6: Commit terminal publication**

```bash
git add src/evaluation/run_evidence_publisher.py tests/test_run_evidence_publisher.py
git commit -m "feat: publish and replay run evidence"
```

### Task 4: Paired collection and invalidation evidence

**Files:**
- Create: `src/evaluation/run_evidence_collection.py`
- Modify: `src/evaluation/run_evidence_publisher.py`
- Create: `tests/test_run_evidence_collection.py`

- [ ] **Step 1: Write failing pilot-regression tests**

```python
def test_collection_rejects_model_seed_changing_data_split():
    runs = (
        verified_run(model_seed=11, data_split_seed=11, split_sha="a" * 64),
        verified_run(model_seed=23, data_split_seed=23, split_sha="b" * 64),
        verified_run(model_seed=47, data_split_seed=47, split_sha="c" * 64),
    )
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection(runs, expected_model_seeds=(11, 23, 47))


def test_collection_rejects_collapsed_or_changed_statistical_units():
    runs = (
        verified_run(model_seed=11, unit_sha="a" * 64),
        verified_run(model_seed=23, unit_sha="a" * 64),
        verified_run(model_seed=47, unit_sha="b" * 64),
    )
    with pytest.raises(RunEvidenceError, match="paired_identity_mismatch"):
        validate_paired_collection(runs, expected_model_seeds=(11, 23, 47))
```

Add tests for duplicate/missing/extra seeds, protocol/analysis/scope mismatch, retained terminal failures yielding `statistics_allowed=False`, and deep immutability.

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n hypersca \
  python -m pytest tests/test_run_evidence_collection.py -q -p no:cacheprovider
```

Expected: module/API missing.

- [ ] **Step 3: Implement paired closure**

```python
@dataclass(frozen=True, slots=True)
class PairedEvidenceCollection:
    runs: tuple[VerifiedRunEvidence, ...]
    expected_model_seeds: tuple[int, ...]
    collection_identity_sha256: str
    statistics_allowed: bool


def validate_paired_collection(
    runs: Sequence[VerifiedRunEvidence], *, expected_model_seeds: Sequence[int]
) -> PairedEvidenceCollection:
    # `tuple_once` performs one bounded iteration and rejects an iterator that
    # changes length or values when its stable snapshot is inspected.
    frozen_runs = tuple_once(runs)
    # `exact_int_tuple_once` additionally requires `type(seed) is int`,
    # non-negative values, uniqueness, and the caller-provided order.
    seeds = exact_int_tuple_once(expected_model_seeds)
    # This helper compares the run model-seed set with `seeds`, rejecting
    # missing, extra, or repeated seeds before any scientific aggregation.
    require_unique_exact_seed_set(frozen_runs, seeds)
    # This helper reads only the already-frozen run identities and requires
    # every listed field to equal the first run's field exactly.
    require_equal_fields(
        frozen_runs,
        fields=(
            "protocol_identity",
            "claim_id",
            "benchmark_id",
            "data_scopes",
            "data_split_seed",
            "data_split_identity_sha256",
            "statistical_unit_schema",
            "statistical_unit_identity_sha256",
            "analysis_identity_sha256",
        ),
    )
    collection_record = {
        "schema_version": "1.0",
        "expected_model_seeds": list(seeds),
        "run_identity_sha256": [
            run.identity.run_identity_sha256 for run in frozen_runs
        ],
    }
    return PairedEvidenceCollection(
        runs=tuple(sorted(frozen_runs, key=lambda run: run.identity.model_seed)),
        expected_model_seeds=seeds,
        collection_identity_sha256=canonical_sha256(collection_record),
        statistics_allowed=all(
            run.terminal_status == "completed" for run in frozen_runs
        ),
    )
```

- [ ] **Step 4: Implement atomic invalidation record**

`write_invalidation_record(path, runs, category, reason)` accepts only verified runs, records their run identities and the mismatch category, uses strict JSON and an exclusive single-file `O_EXCL` write with fsync, and never mutates run directories.

- [ ] **Step 5: Run collection tests and verify GREEN**

Run Step 2 command. Expected: all collection tests pass.

- [ ] **Step 6: Commit paired closure**

```bash
git add src/evaluation/run_evidence_collection.py \
  src/evaluation/run_evidence_publisher.py tests/test_run_evidence_collection.py
git commit -m "feat: validate paired run evidence"
```

### Task 5: Property tests and import boundaries

**Files:**
- Create: `tests/property/test_run_evidence_properties.py`
- Modify: `.importlinter`

- [ ] **Step 1: Write failing Hypothesis properties**

```python
@given(st.dictionaries(safe_keys, strict_json_scalars, min_size=1, max_size=12))
def test_canonical_identity_ignores_mapping_insertion_order(payload):
    reversed_payload = dict(reversed(tuple(payload.items())))
    assert canonical_sha256(payload) == canonical_sha256(reversed_payload)


@given(st.sampled_from(IDENTITY_FIELDS), safe_replacement)
def test_every_identity_field_change_changes_run_sha(field, replacement):
    first = valid_identity()
    changed = replace_identity_field(first, field, replacement)
    assume(changed.to_record()[field] != first.to_record()[field])
    assert changed.run_identity_sha256 != first.run_identity_sha256


@given(state_machine_operations())
def test_publisher_state_machine_publishes_at_most_once(operations):
    result = execute_operations(operations)
    assert result.formal_publish_count <= 1
```

In the same test module define `safe_keys` as NFC non-empty text without
controls, `strict_json_scalars` as exact `None`/bool/int/finite-float/text,
`safe_replacement` as a field-aware composite strategy, and
`state_machine_operations()` as lists drawn from
`add_bytes/add_file/finalize_completed/finalize_failure/abort`. Define
`execute_operations()` to use a fresh `tmp_path` publisher, count only
successful terminal publications, and convert `RunEvidenceError` into a
recorded rejected transition. Add explicit example tests for illegal paths,
non-NFC/control text, huge integers, NaN/Infinity, malicious Mapping/Sequence
single-iteration objects, and one-field split/unit drift; no generated case may
read outside `tmp_path`.

- [ ] **Step 2: Run properties and verify RED**

```bash
conda run --no-capture-output -n hypersca \
  python -m pytest tests/property/test_run_evidence_properties.py -q -p no:cacheprovider
```

Expected: at least one unsupported/missing behavior fails before production changes.

- [ ] **Step 3: Add Import Linter contracts**

Append contracts equivalent to:

```ini
[importlinter:contract:run-evidence-independent]
name = Run evidence has no model, causal, discovery, or dataframe dependencies
type = forbidden
source_modules =
    src.evaluation.run_evidence_identity
    src.evaluation.run_evidence_publisher
    src.evaluation.run_evidence_collection
forbidden_modules =
    src.models
    src.causal
    src.discovery
    pandas
    numpy
    torch
    anndata
```

- [ ] **Step 4: Make minimal validation fixes and run GREEN**

Run:

```bash
conda run --no-capture-output -n hypersca \
  python -m pytest tests/test_run_evidence_*.py \
  tests/property/test_run_evidence_properties.py -q -p no:cacheprovider
lint-imports
```

Expected: all publisher/property tests and Import Linter pass.

- [ ] **Step 5: Commit properties and boundaries**

```bash
git add .importlinter tests/property/test_run_evidence_properties.py \
  src/evaluation/run_evidence_*.py
git commit -m "test: harden run evidence properties"
```

### Task 6: OSTA pilot adapter

**Files:**
- Modify: `src/evaluation/methods_pilot.py`
- Modify: `tests/test_methods_pilot.py`

- [ ] **Step 1: Write failing integration test**

Extend the existing synthetic H5AD test:

```python
verified = verify_run_evidence_bundle(output)
assert verified.identity.data_split_seed == OSTA_SPLIT_SEED
assert verified.identity.model_seed == 11
assert verified.identity.statistical_unit_schema == "osta_platform_sample_block_v1"
assert verified.statistical_unit_record["blocks"] == sorted(
    set(primary["unit_id"].tolist())
)
assert manifest["publisher"]["source_sha256"] == sha256_file(
    Path(run_evidence_publisher.__file__)
)
```

Also run seed 23 on the same synthetic H5AD and assert equal split/unit identity but different run identity.

- [ ] **Step 2: Run the integration test and verify RED**

```bash
conda run --no-capture-output -n hypersca \
  python -m pytest tests/test_methods_pilot.py -q -p no:cacheprovider
```

Expected: old bundle lacks publisher identity and replay contract.

- [ ] **Step 3: Adapt OSTA runner**

Construct `RunEvidenceIdentity` after freezing input/config/code identities. Derive the OSTA statistical-unit record from the actual K=15 primary units before publication. Replace the local tempfile/write/`os.rename` block with `RunEvidencePublisher.add_bytes/add_file/finalize_completed`. Preserve `claim_decision.status="audit_only"` and never set release role.

- [ ] **Step 4: Run OSTA and policy regression**

```bash
conda run --no-capture-output -n hypersca python -m pytest \
  tests/test_methods_pilot.py tests/test_methods_protocol.py \
  tests/discovery/test_evidence_policy.py \
  tests/discovery/test_benchmark_claims.py -q -p no:cacheprovider
```

Expected: all tests pass; no real OSTA H5AD is run.

- [ ] **Step 5: Commit OSTA integration**

```bash
git add src/evaluation/methods_pilot.py tests/test_methods_pilot.py
git commit -m "refactor: publish OSTA pilot evidence uniformly"
```

### Task 7: CausalBench adapter in the design worktree

**Files:**
- Cherry-pick publisher commits into `/home/a/.config/superpowers/worktrees/HyperSCA/real-data-readiness-design`
- Modify there: `src/evaluation/methods_causal_pilot.py`
- Modify there: `tests/test_methods_causal_pilot.py`

- [ ] **Step 1: Cherry-pick the exact publisher commits**

```bash
SOURCE_BRANCH=archive/crc-icb-migration-20260812
DESIGN_TREE=/home/a/.config/superpowers/worktrees/HyperSCA/real-data-readiness-design
for SUBJECT in \
  "feat: freeze run evidence identities" \
  "feat: stage run evidence safely" \
  "feat: publish and replay run evidence" \
  "feat: validate paired run evidence" \
  "test: harden run evidence properties"
do
  COMMIT_SHA="$(git log "$SOURCE_BRANCH" --format=%H --fixed-strings \
    --grep="^${SUBJECT}$" -1)"
  test -n "$COMMIT_SHA"
  git -C "$DESIGN_TREE" cherry-pick "$COMMIT_SHA"
done
```

Verify `sha256sum src/evaluation/run_evidence_*.py` is identical in both worktrees.

- [ ] **Step 2: Write failing causal integration test**

Use a small synthetic cross profile and monkeypatch only `fit_hypersca_c_ablation` at the model boundary. Assert:

```python
verified = verify_run_evidence_bundle(output)
assert verified.identity.data_split_seed == 11
assert verified.identity.model_seed == 23
assert verified.identity.statistical_unit_schema == (
    "causalbench_direction_source_relation_v1"
)
assert verified.statistical_unit_record["eligible_sources"] == expected_sources
assert verified.statistical_unit_record["relation_universe_sha256"] == expected_sha
assert verified.identity.evidence_role == "pilot_audit_only"
```

- [ ] **Step 3: Run and verify RED**

```bash
conda run --no-capture-output -n hypersca python -m pytest \
  tests/test_methods_causal_pilot.py -q -p no:cacheprovider
```

Expected: old causal runner lacks publisher identity.

- [ ] **Step 4: Adapt causal runner and run GREEN**

Build the unit record from direction, ordered genes, eligible sources, and sorted tune edges/complete relation universe. Replace local publication with the shared publisher. Do not add refit/private arguments and keep fixed public split seed 11.

Run:

```bash
conda run --no-capture-output -n hypersca python -m pytest \
  tests/test_methods_causal_pilot.py tests/test_task_c_profile_input.py \
  tests/test_hypersca_c_ablation.py tests/test_hypersca_c_stability.py \
  -q -p no:cacheprovider
```

Expected: all focused tests pass; no real Task C run is executed.

- [ ] **Step 5: Commit causal integration**

```bash
git add src/evaluation/methods_causal_pilot.py tests/test_methods_causal_pilot.py
git commit -m "refactor: publish causal pilot evidence uniformly"
```

### Task 8: Legacy pilot audit and final verification

**Files:**
- Create: `reports/methodology/run_evidence_publisher_legacy_pilot_audit_20260828.md`
- Modify: `results/methods_pilot_v21/pilot_summary.json` only if adding the report path; never alter numerical results or authorization.

- [ ] **Step 1: Run a read-only compatibility audit**

Inspect the 12 valid OSTA and 6 valid CausalBench legacy bundles. Record which new identity fields can be reconstructed and which bundles require legacy status. Do not rewrite any bundle.

- [ ] **Step 2: Write the report**

The report must state:

```text
legacy bundles remain audit evidence only;
new publisher verification is not retroactive authorization;
three invalid roots remain invalid;
release_authorized remains false;
no private/refit/release input was read.
```

- [ ] **Step 3: Run final verification**

```bash
conda run --no-capture-output -n hypersca python -m pytest \
  tests/test_run_evidence_*.py tests/property/test_run_evidence_properties.py \
  tests/test_methods_pilot.py tests/test_methods_protocol.py \
  tests/discovery/test_evidence_policy.py tests/discovery/test_benchmark_claims.py \
  -q -p no:cacheprovider
lint-imports
conda run --no-capture-output -n hypersca python -m py_compile \
  src/evaluation/run_evidence_identity.py \
  src/evaluation/run_evidence_publisher.py \
  src/evaluation/run_evidence_collection.py
git diff --check
```

In the design worktree run the Task 7 focused command and `git diff --check`.

- [ ] **Step 4: Verify authorization is unchanged**

```python
summary = json.loads(
    Path("results/methods_pilot_v21/pilot_summary.json").read_text()
)
assert summary["status"] == "audit_only"
assert summary["promotion_eligible"] is False
assert summary["release_authorized"] is False
```

- [ ] **Step 5: Commit audit documentation**

```bash
git add reports/methodology/run_evidence_publisher_legacy_pilot_audit_20260828.md
git commit -m "docs: audit legacy pilot evidence"
```

No step in this plan runs a real release, accesses private holdout data, or changes scientific comparators and thresholds.
