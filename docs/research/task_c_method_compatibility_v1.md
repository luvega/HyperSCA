# 任务 C 比较方法兼容性与证据边界 v1

## 这份说明解决什么问题

任务 C 比较的是：不同方法能否从同一批允许查看的单细胞数据中找回有向基因关系。方法依赖的软件、可用信息和返回形式不同，因此“程序能运行”不等于“比较公平”。本说明固定每种方法如何接入、哪些数据可以读取、哪些结果可以进入共同评分，并如实保留暂时不能运行的方法。

“全面比较清单”指预先登记的全部 19 种候选、简单对照、无效应对照和外部方法；“核心预演清单”是其中必须优先跑通的较小集合。外部方法失败时，其他方法可以继续运行，但失败方法仍保留在全面清单和最终报告中，不会因失败而被删除。

## 共同运行规则

统一命令是 `scripts/run_task_c_method.py`。它先核对方法登记、允许读取的数据、固定基因顺序和软件身份，再保存方法原始结果。只有原始结果满足三列 `source,target,score`、基因名属于固定清单、分数有限且非负时，才会生成共同关系表 `predictions.csv`。

共同关系表包含固定基因集合内所有非自身的有向关系。方法没有返回的关系补为零分，并用 `returned_by_method=false` 标明。这是为了让评价范围一致，不表示方法明确判断该关系不存在。重复关系只保留最高分；标准表本身没有重复关系。

关系强弱必须有官方依据。固定版本中，只有 GRNBoost 的返回顺序可作为强弱顺序，因而按原顺序换成递减正分。其他 CausalBench 方法只提供未排序的关系集合，所有已返回关系并列为 1 分，不把容器位置解释成新的排名。PSGRN 按其官方返回顺序保存前 1,000 条关系。

每次成功运行保留：

- `raw_predictions.csv`：投影后的三列原始关系分数；
- `predictions.csv`：补齐后的共同关系表；
- `method_status.json`：成功、失败或暂无官方代码的真实状态；
- `environment_manifest.json`：登记表、输入、固定代码、软件环境和命令模板的 SHA-256 记录。

外部程序的进程状态和资源记录放在 `raw_runtime/`，避免覆盖统一命令自己的状态。HyperSCA-C 的完整原始输出先保存在 `raw_method_output/`，然后才投影出共同三列。统一命令最多处理 256 个基因；已有目录不会被覆盖。只有真实数据控制器完成封存评分和强制文件核验后，状态才可以升级为 `passed_real_rehearsal`；本命令本身绝不写入该状态。

统一状态还记录内层运行状态、失败发生在运行阶段还是标准化阶段，并带有不含自身字段时计算的内容 SHA-256。重新使用结果时，外层状态必须与 `raw_runtime/method_status.json` 一致：成功标准化要求内层已经完成原始推断；外部程序自身失败时，两层失败名称必须相同；内层成功但三列或 HyperSCA-C 四文件证据无效时，外层才记录为 `failed_invalid_output`。

## 允许读取的信息

“观察信息方法”只能读取标签为 `non-targeting` 的未干预细胞。登记为观察信息的方法包括 Random1000、GRNBoost、PC、GES、GSP、NOTEARS Linear 和 Sortnregress。

“部分干预信息方法”可以读取公开学习文件中的未干预细胞和已公开的干预细胞，但不能读取独立验证数据。登记为部分干预信息的方法包括 HyperSCA-C、均值差简单对照、GIES、IGSP、DCDI-G、DCDI-DSF、DCDFG-LIN、DCDFG-MLP 和 PSGRN。

正式运行必须同时提供 Task C 的完整公开文件清单，并核对所选文件已经登记、SHA-256 一致、没有符号链接或未登记的硬链接。合成数据只能显式标为 `synthetic_smoke`，意思是“仅检查分析流程能否运行”；它不能被当作真实数据证据。任何方法都不得读取名称或路径含 `private` 的独立验证文件。

## 跨环境派生学习文件

单环境比较直接读取公开清单中的三数组文件。跨环境比较不能把两份文件路径悄悄交给各方法自行拼接，而是先由 `materialize_task_c_derived_input` 生成一份可以重算的四数组文件，并在统一命令中同时提供 `--public-manifest` 和 `--derived-input-manifest`。

派生记录的格式名称固定为 `task_c_derived_input_v1`，含以下研究含义：

- `condition=cross_environment`、固定方向和 `stage=refit`；
- 两个父文件恰好是同一方向的 `source_refit.npz` 与 `target_adapt_refit.npz`，并分别记录公开相对路径和 SHA-256；
- 变换固定为 `per_environment_control_center_then_row_concatenate_v1`：每个环境先减去各自未定向干预对照细胞的逐基因均值，再按来源环境、目标适应环境的顺序合并细胞行；
- 输出含 `expression_matrix`、`interventions`、`var_names` 和 `environment_labels` 四个数组；环境标签只允许对应方向的两个细胞背景，并记录各自细胞数；
- 输出文件的 SHA-256、字节数和固定基因顺序 SHA-256 写入派生记录。

统一运行层会重新读取两个公开父文件、重复上述变换，并逐数组比较表达、干预标签、基因顺序和环境标签。父文件、方向、环境标签、输出内容或派生记录任一项变化都会停止运行。派生输入只允许用于正式公开数据，不允许用于合成数据，也不读取独立验证数据。

## CausalBench 方法

CausalBench 官方仓库固定为 <https://github.com/causalbench/causalbench>，提交固定为 `1a2143cffdc85f835b41ce8d52034be1bf903e71`，隔离环境名为 `hypersca-task-c-causalbench`。

当前接入的方法为 Random1000、GRNBoost、PC、GES、GIES、GSP、IGSP、NOTEARS Linear、DCDI-G、DCDI-DSF、DCDFG-LIN、DCDFG-MLP 和 Sortnregress。统一命令把登记的方法名称、允许的信息类型和关系返回含义原样传给固定工作进程，不在主环境中重新实现这些方法。

Varsortability 在该固定提交中是“数据中的方差顺序可能帮助方向判断”的诊断概念；实际可以运行的网络估计器是 `Sortnregress`。登记表只运行 Sortnregress，不把同一实现再以 Varsortability 名义计为第二种独立方法。

## PSGRN

PSGRN 官方仓库固定为 <https://github.com/GuanLab/PSGRN>，提交固定为 `74aa640f7c472b23a69811f6795bb17678efd344`，隔离环境名为 `hypersca-task-c-psgrn`。运行前核对仓库来源、提交和本地文件没有改动；方法只读取公开的部分干预学习文件。

## 当前没有可运行官方资产的方法

BetterBoost、SparseRC 和 CATRAN 当前登记的主要来源是公开方法报告，尚未找到可按本项目固定规则运行的官方代码资产。它们必须记录为 `official_assets_unavailable`，不得用项目自行编写的相似版本替代，也不得生成伪造的关系分数。

对应报告：

- BetterBoost：<https://openreview.net/forum?id=gpDOOAOmMe>
- SparseRC：<https://openreview.net/forum?id=TOaPl9tXlmD>
- CATRAN：<https://openreview.net/forum?id=Wf0QRYUkhwV>

## 如何解释失败与可靠性

`completed_standardized_output` 只表示该次原始结果通过格式、基因范围和数值核验，并成功补齐到共同关系范围；它不表示生物学结论已经得到验证。超时、资源不足、软件不兼容、输出无效和官方资产缺失分别记录，不互相替代。

重新使用已有结果时，不只核对文件 SHA-256。均值差简单对照还会根据固定输入重新计算，以确认数值仍符合“干预组平均表达减去未定向干预对照组平均表达，再取绝对值”的规则；共同关系表也会从原始三列重新生成并逐项比较。这些检查提高可复查性，但仍不能把不完整参考网络解释成完整因果真值。

运行身份逐文件记录统一入口、标准化、公开数据核验、运行监督、方法登记、HyperSCA-C 模型及稳定性核验、两个外部工作进程等固定代码闭包。HyperSCA-C 设置和基因清单、外部隔离环境记录及官方源码状态作为对应方法的附加输入单独记录；任一固定依赖在运行后或复用前变化，已有结果都不能复用。
