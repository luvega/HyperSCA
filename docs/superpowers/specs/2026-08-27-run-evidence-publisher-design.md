# RunEvidencePublisher 设计规格

- 状态：已确认设计，待实现计划
- 日期：2026-08-27
- 适用协议：`hypersca-methods-v2.1`
- 科学状态：pilot `audit_only`，release 未授权

## 1. 背景与目标

真实 3-seed pilot 暴露了三类不能由单个 benchmark runner 自己可靠避免的错误：model seed 意外改变 data split；多个真实统计块被合并成一个 unit；不同 CausalBench split 的 genes、eligible sources 和 relation universe 被错误地当成配对数据。这些错误都能产生表面完整的运行目录，却不能形成合法的配对统计证据。

`RunEvidencePublisher` 的目标是提供一个只负责证据身份、工件封存、终止状态和跨 seed 配对闭包的公共深接口。它不训练模型，不计算科学指标，不决定 `admitted`，也不修改 protocol、threshold、comparator 或数据范围。

## 2. 明确不做的事情

- 不启动 5-seed release，不访问 sealed holdout。
- 不因为 pilot 结果未通过而修改 HyperSCA/HyperSCA-C。
- 不把 publisher 变成 pandas、PyTorch、anndata 或 benchmark-specific 代码的依赖中心。
- 不在本阶段拆分大型 benchmark runner。
- 不删除或覆盖既有无效 pilot 目录；失效由独立 invalidation evidence 表示。
- 不提供 CRC 结果补救 public gate 的接口。

## 3. 模块边界

新模块位于 `src/evaluation/run_evidence_publisher.py`，只依赖 Python 标准库。通过 Import Linter 固化以下边界：

1. publisher 不导入 `src.models`；
2. publisher 不导入 `src.causal`；
3. publisher 不导入 `src.discovery`；
4. publisher 不导入 pandas、NumPy、PyTorch、anndata；
5. benchmark runner 可以依赖 publisher，publisher 不能反向依赖 runner；
6. scientific `EvidencePolicy` 只能消费已验证 manifest，publisher 不能导入或构造 claim decision。

主工作树与 `design/real-data-readiness` worktree 必须使用逐字节相同的 publisher 模块。实现先形成独立提交，再 cherry-pick 到设计分支；两个 runner 的适配提交分别完成，但 manifest 必须记录相同 publisher source SHA。分支整合前不得运行 release。

## 4. 不可变身份模型

### 4.1 `RunEvidenceIdentity`

冻结字段如下：

- `schema_version`；
- `protocol_version` 与 `protocol_identity`；
- `claim_id` 与 `benchmark_id`；
- `data_scopes`，pilot 必须精确为 `("train", "tune")`；
- `data_split_seed`；
- `model_seed`；
- `data_split_identity_sha256`；
- `statistical_unit_schema`；
- `statistical_unit_identity_sha256`；
- `analysis_identity_sha256`，覆盖指标、comparator 集合和所有非 seed 科学设置；
- `input_identity_sha256`、`config_identity_sha256`、`code_identity_sha256`；
- `evidence_role`，限定为 `pilot_audit_only`、`release_candidate` 或 `infrastructure_smoke`。

构造函数只接受 exact built-in JSON scalar/tuple 类型，拒绝 bool 冒充 int、非有限值、重复项、Unicode 非规范文本和不合法 SHA。规范 JSON 使用 UTF-8、排序字段、紧凑分隔符和 `allow_nan=False`，生成 `run_identity_sha256`。`data_split_seed` 与 `model_seed` 是两个独立字段，不能通过一个通用 `seed` 参数代替。

### 4.2 统计单位身份

publisher 不解释 benchmark 科学含义，但要求 runner 提供规范化 unit record 并记录其 SHA：

- OSTA record 至少包含 platform、sample、held-out block 的有序集合及 K；
- CausalBench record 至少包含 direction、ordered genes、eligible sources、完整 relation universe identity 及 tune-reference identity。

record 自身写入 manifest，SHA 写入 `RunEvidenceIdentity`。因此 unit 记录的任何合并、遗漏、重排语义变化或 relation/source 变化都会改变身份。

## 5. Publisher API 与状态机

### 5.1 API

最小公共 API：

```python
publisher = RunEvidencePublisher.begin(
    output_dir=...,
    identity=...,
    statistical_unit_record=...,
    required_artifacts=(...),
    maximum_bundle_bytes=...,
)
publisher.add_bytes(relative_path, payload, media_type=...)
publisher.add_file(relative_path, source_path, media_type=...)
publisher.finalize_completed(summary=...)
# 或
publisher.finalize_failure(status="failed_runtime", reason=...)
```

`add_bytes` 适合 JSON/CSV 小工件；`add_file` 通过 `O_NOFOLLOW` 打开来源，使用 `fstat` 绑定 inode/size/time/link count，流式复制并在复制后复核，避免把大型 embeddings 全部读入内存。所有相对路径必须无 `..`、无绝对路径、无控制字符且唯一。

### 5.2 状态转换

状态严格为：

```text
NEW -> STAGING -> COMPLETED_PUBLISHED
                 -> FAILED_PUBLISHED
                 -> ABORTED
```

- `begin` 只创建同一父目录内的私有 staging，不创建正式目标。
- 同一实例只能 finalize 一次；完成后所有写接口失效。
- completed 必须具有精确 required artifact 集；failure 不能伪造科学 summary。
- 异常或 `KeyboardInterrupt` 清理 staging，但不删除任何已发布或既有目录。
- 终止状态沿用已登记集合，不把异常归类为 completed。

## 6. 安全与原子发布

- output parent、source file 和 staging 逐级拒绝 symlink。
- 所有正式工件必须是 single-link regular file；bundle 内 inode 不得重复。
- manifest 包含每个工件的相对路径、字节数、SHA-256、media type。
- `method_status.json` 与 `run_manifest.json` 最后生成，并交叉绑定 `run_identity_sha256`、artifact inventory SHA 和 terminal status。
- 文件与 staging directory 在发布前 `fsync`。
- Linux 使用 `renameat2(RENAME_NOREPLACE)` 排他发布；能力缺失时 fail closed，不退化为会覆盖的 `os.rename`。
- 正式目标已存在、悬空链接或竞态出现时一律拒绝，不修改目标。
- 发布后重新打开正式目录并核验 inode、manifest、status 和 inventory；失败则报告基础设施错误，不把目录视为 verified evidence。

## 7. 跨 seed 配对闭包

`validate_paired_collection(manifests, expected_model_seeds)` 返回深不可变 `PairedEvidenceCollection`，并要求：

- model seed 精确、唯一且等于预注册集合；
- protocol、claim、benchmark、data scopes 相同；
- `data_split_seed` 与 `data_split_identity_sha256` 相同；
- `statistical_unit_schema` 与 identity 相同；
- `analysis_identity_sha256` 相同；
- 每个 run 都是已验证 terminal bundle；
- 任一 failure 保留但使 collection 不可产生 summary statistics；
- completed 集合不得用成功 seed 替换失败 seed。

OSTA 与 CausalBench 的 bootstrap adapter 只能接受该 collection，不能直接接收目录列表。collection 验证失败时生成独立 `invalidation_record.json`，记录所有输入 run identity 与拒绝原因；原运行目录保持不变。

## 8. 错误模型

公共异常统一为 `RunEvidenceError`，包含稳定的错误类别：

- `invalid_identity`；
- `invalid_artifact`；
- `invalid_state_transition`；
- `publication_conflict`；
- `publication_infrastructure`；
- `paired_identity_mismatch`。

异常文本不得包含完整命令、凭据或 private 路径。底层 `OSError`、`UnicodeError`、JSON 错误、整数溢出和不支持的原子原语均归一为领域错误。

## 9. 测试策略

### 9.1 单元与集成测试

- bytes/file 工件成功发布及重放验证；
- completed/failure 精确文件集合；
- 已有目标、悬空链接、父路径链接、hardlink、重复 inode、文件变化和目录竞态拒绝；
- staging 在异常与中断后清理；
- 发布后 manifest/status/artifact 交叉篡改拒绝；
- OSTA 与 CausalBench runner 各一个小型 synthetic 集成测试，证明实际调用统一 publisher。

### 9.2 Hypothesis 属性测试

- JSON mapping 插入顺序不改变 identity；
- 任一 identity 字段变化必改变 `run_identity_sha256`；
- `data_split_seed` 和 `model_seed` 交换或共用必被拒绝；
- 任一统计 unit/source/relation 的添加、删除或替换必改变 unit identity；
- 任意跨 seed split/unit identity 漂移使 collection fail closed；
- 任意状态调用序列至多发布一次，finalize 后不能再次写入；
- 任意非法相对路径不能逃出 staging root；
- 超大整数、NaN/Infinity、Unicode 非规范文本和恶意 Mapping/Sequence 被归一拒绝。

## 10. 分阶段实施

1. TDD 实现 immutable identity 与 canonical hashing。
2. TDD 实现 artifact staging、状态机和排他发布。
3. TDD 实现 bundle 重放 verifier 与 paired collection validator。
4. 增加 Import Linter 边界和 Hypothesis 属性测试。
5. 让 OSTA pilot runner 使用 publisher；用小型 synthetic fixture 验证，不重跑真实 pilot。
6. 将 publisher 提交 cherry-pick 到 Task C 设计 worktree，让 causal pilot runner 使用同一模块；同样只跑 synthetic/focused 回归。
7. 对既有有效 pilot bundle 做只读迁移审计：不重写旧目录，只生成 legacy-to-publisher compatibility report。

实现完成后仍保持 release 未授权。是否进行新的 pilot 或 5-seed release需要单独、显式的科学决定。

## 11. 成功标准

- 两个 runner 记录相同 publisher source SHA；
- pilot 中出现过的三类 identity 错误都有先 RED 后 GREEN 的回归测试；
- publisher 模块通过 Import Linter 边界；
- Hypothesis 属性测试覆盖 identity、状态机和配对闭包；
- Python 3.10 focused 与相关 benchmark 回归通过；
- 不修改任何已发布 pilot bundle，不读取 private/release holdout；
- `pilot_summary.json` 仍为 `audit_only` 且 `release_authorized=false`。
