# Thought–Action 覆盖判断数据处理方案

日期：2026-10-13。依据 `docs/10.13思路整理.md`，将下一阶段的主任务定义为：判断当前评测窗口内的原子 Intent 是否被原始 Action 调用**完整覆盖**，再将覆盖关系聚合为 Thought–Action 的 `faithful`、`partial`、`not_executed`。这里的 `not_executed` 只表示未覆盖，不表示工具没有实际运行。本阶段不验证执行是否成功。

200 条模型审核、30 条人工复核界面、字段清单和评测命令见 `docs/2026-10-13-ta-coverage-pilot-runbook.md`。

## 1. 数据单位与来源

- 原始单位是一个评测窗口：保留原始 Thought、当前 Action 调用列表、轨迹/步骤 ID、任务和来源。一个 Thought 可有多个 due-now Intent，一个窗口也可有多个 Action。
- 训练单位是一个 `(atomic Intent, raw Action)` pair。对于同一窗口的 `m` 个可评测 Intent 和 `n` 个 Action，生成最多 `m × n` 个待标注 pair；数量上限不是自动正负标签。
- ClawGym 用独立 train 轨迹构造第一批数据，SWE-Agent trajectories 按同一 schema 单独转换并作为跨域增量。SWE 的 `target=true` 只说明最终任务结果，不能证明每一步的 Intent–Action 覆盖关系。
- 现有 ClawGym 训练集有 1,973 **检索三元组行**；其中 `is_known_positive`、Silver 正例及既有 GQQ“安全负例”建议均不能直接当作覆盖二分类 Gold。旧数据可用于抽样、难例候选和召回 baseline，入新任务前须重新审核 pair 标签。

## 2. 轨迹切分先于标注扩充

先固定 train/dev/test 的来源边界，再从各 split 内生成窗口和 pair。同一轨迹不得跨 split；SWE 还要按 `instance_id`/issue 分组，近重复任务模板也须同组。跨源检查归一化 Thought、Intent 和 Action 的近重复，避免同一任务或等价命令泄漏。dev 用于模型与阈值选择；独立 Gold test 冻结版本和哈希，不参与训练、提示词调试或阈值选择。现有检索 dev/test 不能无审核地当作新任务 Gold。

## 3. Intent 提取与可评测性

1. 从 Thought 提取当前窗口的 due-now 原子目标；未来计划单独标记。每个 Intent 保留原文、Thought 引文、时间属性和原子化版本，不从 Action 反推补写 Intent。
2. 独立判断目标、操作、范围和关键约束是否明确。记录 `clear`、`context_resolved`、`ambiguous`、`complex_joint_coverage` 及理由。`context_resolved` 只有在评测时实际可见、可复现的前文中能唯一解析指代时才可用；第一版只输入 Intent 与 Action 文本的模型不使用该类，除非提供同一明确上下文输入。
3. 模糊 Intent 保留在审计数据中，不生成训练标签。如果一个窗口仅部分 due-now Intent 可评测，保留局部 pair 信息，但不进入严格 TA 三分类主评测。必须由多个 Action 联合完成且无法继续原子化的 Intent 放入复杂覆盖队列，不强制贴二分类标签。

## 4. Action 保留与 pair 标注

模型输入保留**原始工具调用文本**，包括工具名、命令、路径、URL、参数及多调用边界。可另存规范化/解析字段用于去重、检索和质量检查，但不能只保留简化后的结构，也不能把长命令静默截断为另一个标签对象。超过模型输入长度的 Action 单列处理或使用明确的长文本策略，并报告排除率。

对每个 pair 标 `covered=1` 当且仅当 Action 在操作语义上完整覆盖该 Intent 的明确目标、对象和必要参数；否则标 `covered=0`，同时保留 `negative_type`，例如 `different_operation`、`wrong_target`、`wrong_parameter`、`preparation_only`、`partial_fulfillment`、`equivalent_or_uncertain`。后一类不得强制作为负例。执行成功/失败与覆盖标签分开记录。

候选优先覆盖同对象不同操作、同操作错目标、缺少关键参数或输出、准备动作、语义相关但未完成。所有同窗口 pair 都须独立判断；同轨迹近邻、随机 Action、旧模型 Top-K 只能生成**待审核候选**，不能默认负例。若一个 Intent 可由多个等价 Action 覆盖，均标正例，不能用单一已知正例排斥其他 Action。

## 5. GQQ 辅助审核与 Gold

GQQ 对每个候选先只看 Intent 判断清晰度和目标证据，再核对正例/候选 Action 是否完整覆盖，最后检查不同命令是否可能产生等价结果。输出逐 pair 标签、证据、信心与原因，并保留模型版本、提示词版本和原始来源 ID。模型判为模糊或不确定时不自动入训。

已发现 GQQ 曾把“the input files”误判清晰，把 `xxd`、`od -c` 误判为检查空白格式的安全负例；修订提示词在这两个样本上通过定向复测，但不能证明批量准确。训练集采用规则高置信候选加 GQQ 辅助审核，并对通过与拒绝样本分层人工抽检和纠错；dev/test Gold 由人工独立复核，报告标注者一致性及争议处置。GQQ 输出属于弱监督，不记为人工 Gold。

## 6. 聚合和实验

对窗口内每个可评测 due-now Intent，若任一当前 Action 的 pair 为 `covered=1`，该 Intent 视为已覆盖。设可评测 Intent 数为 `m`、已覆盖数为 `K`：`K=m` 为 `faithful`，`0<K<m` 为 `partial`，`K=0` 为 `not_executed`。`m=0`、含不可评测 due-now Intent 或有不可拆联合覆盖的窗口不进入严格三分类主集。第一版不单独识别完成后又执行相反操作的情形。

第一轮目标为约 2,000–5,000 个经质量控制的 **pair**，同时报告独立轨迹、窗口、Intent、Action、正负 pair 数及各难例类别，避免用重复负例堆行数。以相同 split 比较监督微调 BGE 双塔和 Cross-Encoder；BGE + Code Action 塔可在前两者结果确定后单列实验。分别在 dev 校准二分类阈值；最终报告 pair-level Precision/Recall/F1、TA 三分类 Macro-F1、难例错误率、延迟和资源消耗。若每窗口 Action 数量较少，直接评估全部 pair；只有候选量大时才引入双塔筛选，并先测候选覆盖率。

## 7. 产物与验收

新版本目录应分别保存 `windows.jsonl`、`intents.jsonl`、`actions.jsonl`、`pair_candidates.jsonl`、`pair_reviews.jsonl`、`pair_labels.jsonl`、`ta_gold.jsonl` 和 `manifest.json`。每个 pair 有稳定 ID、源轨迹/步骤/Intent/Action ID、原始文本、审核状态、标签及证据；manifest 记录输入哈希、提示词/模型版本、切分统计、排除原因和去重统计。构建器拒绝跨 split 来源、重复或冲突标签、缺失审核依据、把模糊 Intent 的 pair 当作确定训练标签，以及用 dev/test 审计错误回填 train。

当前 `review_exec_candidates_gqq.py` 和 20 条试跑结果仍属于**旧检索候选审核**；它们不能直接产出这里定义的 pair Gold。实施顺序是先冻结新任务切分和 schema，再完成一个包含多 Intent/多 Action 的小规模人工 Gold 试点，验证覆盖聚合和标注一致性，之后批量转换 ClawGym、试接 SWE-Agent 并训练模型。
