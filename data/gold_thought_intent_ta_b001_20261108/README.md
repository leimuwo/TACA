# Thought → Intent → TA 对人工金标核心集（B001）

这是一组真正能把两阶段拼起来的人工确认数据：同一批 100 个窗口同时有人工审核的 Thought→Current Intent 提取，以及用户明确确认的 Intent→Action 操作匹配。

- 100 个完整窗口
- 100 条非空 Thought
- 108 条当前 Intent
- 108 条人工确认 TA 对：104 `yes`、4 `no`
- 23 条 Future/排除 Intent 单独保存，不进入当前 TA 对
- 每个窗口保留完整 Action 原文、动作 ID、原始文件、消息位置和轨迹 ID

文件：

- `units.jsonl`：完整 Thought、当前 Intent、完整 Action 和窗口级来源。
- `pairs.jsonl`：人工确认的 Intent–Action 操作匹配。
- `future_or_excluded_intents.jsonl`：被标为 Future 或排除的 Intent，避免误当 Current。
- `manifest.json`：哈希和来源核查结果。

## 这组数据可以称为什么

可以称为：**operation-only 的 Thought→Intent→TA pair 人工金标核心集**。

它不能称为：

- 完整参数/选项/限制条件一致性金标；
- 任务成功或环境结果金标；
- 独立验证集或测试集；
- 包含 target_relation（same/precondition/different）的 v1.5 金标；
- 全量数据的人工金标。

原因是当时的人工 TA 页面采用 `operation_only_v1`，人工确认字段是操作是否匹配；4 条 `no` 是 `misalign`。目标关系、参数值、命令条件、任务成功没有作为本批人工确认字段统一导出。该包是暴露的开发校准池，不能和同源窗口随机拆成训练/验证/测试，否则会造成泄漏。

## 来源

- `outputs/ta_consistency_v1_20261023/B001_operation_review_gqq_v1/frozen_units.jsonl`
- `outputs/ta_consistency_v1_20261023/B001_operation_confirmed_v1/intent_extraction_labels.jsonl`
- `outputs/ta_consistency_v1_20261023/B001_operation_confirmed_v1/operation_labels.jsonl`
- `outputs/ta_consistency_v1_20261023/B001_operation_confirmed_v1/human_confirmation_audit.jsonl`

`manifest.json` 中的检查确认：Thought 非空、窗口完整、Action 可回连、Intent 为 accepted Current、TA 对均来自用户明确确认，且每个当前 Intent 都有对应关系标签。

更大的 250 条 Intent 人工复核和 140 窗口 v1.5 TA 人工关系不能直接与此合称金标：140 窗口使用的是 `frozen_teacher_v1.3_not_human_gold` Intent，250 条 Intent 复核与 TA 关系没有逐窗逐 Intent 完整交集。它们应作为扩展复核集或候选集。
