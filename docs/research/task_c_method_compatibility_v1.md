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

## 允许读取的信息

“观察信息方法”只能读取标签为 `non-targeting` 的未干预细胞。登记为观察信息的方法包括 Random1000、GRNBoost、PC、GES、GSP、NOTEARS Linear 和 Sortnregress。

“部分干预信息方法”可以读取公开学习文件中的未干预细胞和已公开的干预细胞，但不能读取独立验证数据。登记为部分干预信息的方法包括 HyperSCA-C、均值差简单对照、GIES、IGSP、DCDI-G、DCDI-DSF、DCDFG-LIN、DCDFG-MLP 和 PSGRN。

正式运行必须同时提供 Task C 的完整公开文件清单，并核对所选文件已经登记、SHA-256 一致、没有符号链接或未登记的硬链接。合成数据只能显式标为 `synthetic_smoke`，意思是“仅检查分析流程能否运行”；它不能被当作真实数据证据。任何方法都不得读取名称或路径含 `private` 的独立验证文件。

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

