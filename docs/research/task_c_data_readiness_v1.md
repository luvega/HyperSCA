# 任务 C 数据准备与封存规则

任务 C 使用 K562 和 RPE1 单细胞基因干预数据，检验方法能否从部分已见干预推广到未见干预。数据来源于固定版本 CausalBench；HyperSCA 不重新定义原始实验标签，也不把计算结果改写为实验结论。

## 数据和许可

- CausalBench 代码采用 Apache-2.0 许可。
- Replogle Perturb-seq 数据由 CausalBench 记录为 CC-BY-4.0 许可。
- 固定代码提交为 `1a2143cffdc85f835b41ce8d52034be1bf903e71`。
- 本任务同时准备 K562 和 RPE1 两种细胞背景。每次准备都会记录输入文件的内容指纹，用于确认数据没有被悄悄替换。

汇总生物关系还涉及 ChIP-Atlas、CORUM、STRING 和配体—受体资源。各来源的许可随参考关系记录保存；后续分享或再发布这些文件时，应按各来源许可分别核对，不能只依据 CausalBench 代码许可。

## 共享干预划分

任务 C 预先固定五个划分，随机种子分别为 11、23、47、71 和 97。满足细胞数要求、且在 K562 和 RPE1 中都出现的干预基因，在两种细胞背景中使用相同的学习、参数调节和最终检验归属。这样可减少因两种细胞背景划分不同而造成的比较偏差。未干预对照细胞也分别分配给学习、参数调节和最终检验，不能跨组重复使用。

## 数据隔离

模型运行只接收公开记录清单 `public_manifest.json` 列出的学习文件。公开清单会记录可用于学习和参数调节的干预基因，以及最终检验基因的数量，但不会公开最终检验基因名称、细胞位置或文件路径。

未参与模型建立的独立验证数据（holdout）位于 `private/`。这些文件及其完整清单只由最终评分步骤读取，不能用于选基因、选择方法、调整参数或决定何时停止训练。跨细胞背景评估时，目标细胞背景的干预细胞同样保持封存；方法只能使用该背景中公开的未干预对照细胞进行适配。

## 本机准备

以下命令先创建固定版本 CausalBench 的独立运行环境，再导出官方数据和参考关系，最后生成五个共享划分。数据根目录放在项目目录之外，可避免误把大型数据或封存文件提交到版本库。

```bash
conda env create -f envs/task_c/causalbench.yml
TASK_C_DATA_ROOT=/home/a/Data/HyperSCA_external/task_c
mkdir -p "$TASK_C_DATA_ROOT/official"
conda run -n hypersca-task-c-causalbench python scripts/export_causalbench_data.py \
  --data-dir "$TASK_C_DATA_ROOT/official"
python scripts/prepare_task_c_data.py \
  --k562-npz "$TASK_C_DATA_ROOT/official/dataset_k562.npz" \
  --rpe1-npz "$TASK_C_DATA_ROOT/official/dataset_rpe1.npz" \
  --k562-pooled-reference "$TASK_C_DATA_ROOT/official/reference_k562_pooled.csv" \
  --k562-chipseq-reference "$TASK_C_DATA_ROOT/official/reference_k562_chipseq.csv" \
  --rpe1-pooled-reference "$TASK_C_DATA_ROOT/official/reference_rpe1_pooled.csv" \
  --rpe1-chipseq-reference "$TASK_C_DATA_ROOT/official/reference_rpe1_chipseq.csv" \
  --output-dir "$TASK_C_DATA_ROOT/prepared"
```

如需改变每个干预至少包含的细胞数，可在准备命令末尾加入 `--min-cells-per-intervention`。默认值为 5；改变该值会改变哪些干预能够进入五个划分，因此正式比较前必须固定并记录。

## 参考关系如何用于评分

导出命令同时保存 CausalBench 汇总的生物关系和染色质免疫沉淀（ChIP）有向关系。双向展开的汇总关系用于主要平均精确率（average precision, AP）评分；ChIP 关系只用于补充判断预测方向是否一致，不改变主要评分。

固定提交的 RPE1 分支实际使用随包提供的 HepG2 ChIP 文件，而不是 RPE1 特异的 ChIP 文件。两种细胞背景并不相同，因此该结果只能作为方向性补充证据；报告必须明确这一局限，不能把它解释为 RPE1 中已得到直接验证。

## 运行注意

准备脚本会先检查五个划分是否可以生成，再逐个写入结果。不过，整个五划分生成过程不是一次不可分的操作：如果中途因断电、磁盘空间不足或其他错误而停止，不要手工拼接旧文件和新文件。应保留原目录用于排查，另选空目录重新运行；如需复用原目录，必须先确认现有记录清单和文件内容指纹均通过脚本检查。

同一个 `--output-dir` 不能同时运行两个准备进程。并发写入可能使一个进程读到另一个进程尚未完成的文件，从而得到无法可靠复查的结果。不同实验若需同时准备，应使用彼此独立的输出目录。

## 结论边界

这些文件只用于研究评测。主要 AP、方向性补充结果和跨细胞背景表现，均属于预先规定数据上的计算评估。封存检验通过前，不得声称方法已在真实数据上得到独立验证；即使封存检验通过，也不得据此声称某条因果关系、药物作用机制或候选靶点已被实验确认。
