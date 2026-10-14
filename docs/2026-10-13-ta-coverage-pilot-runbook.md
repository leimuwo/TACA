# 2026-10-13 TA 覆盖判断试点

本试点依据 `docs/10.13思路整理.md`，评估窗口定义为同一轨迹的一次 assistant Thought 及其**当次**工具调用。来源 split 沿用 `stage3_5_parent_split_complete_925_v1` 并写入 `split_assignments.jsonl`；647 条 train、139 条 dev、139 条 test 轨迹，构建器拒绝轨迹或任务模板跨 split。200 条审核窗口只从 train 中抽取，覆盖 161 条轨迹、746 个 due-now Intent–Action pair。原始 Thought、Intent、工具调用文本和逐调用工具返回均保留；抽样中的 417 个 Action 均关联到工具返回。构建器没有导入旧检索 qrels 或 `is_known_positive` 标签。

## 冻结字段

| 文件 | 必填字段 | 标签状态 |
| --- | --- | --- |
| `windows.jsonl` | `window_id`, `source`, `split`, `trajectory_id`, `thought_step`, `thought`, `user_request`, `intents`, `actions`, `pairs`, `ta_gold`, `gold_review_status`, `review_notes` | `ta_gold=null` |
| `intents.jsonl` | `intent_id`, `source_intent_id`, `window_id`, `text`, `source_quote`, `temporal_scope`, `assessability`, `assessability_reason` | `assessability=unreviewed` |
| `actions.jsonl` | `action_id`, `window_id`, `call_index`, `raw_call`, `action_text`, `execution_result`, `result_status` | 调用事实，无覆盖标签 |
| `pairs.jsonl` | `pair_id`, `window_id`, `intent_id`, `action_id`, `covered`, `review_status`, `review_reason` | `covered=null` |
| `ta_gold.jsonl` | `window_id`, `gold_label`, `review_status`, `review_notes` | `gold_label=null` |

ID 由来源轨迹、步骤及原始 Intent/Action ID 稳定生成。独立人工复核导出的 JSONL 另含 `gold_label`, `review_reason`, `disputes`, 每个 Intent 的 `due_now`/`assessable` 及每个 pair 的 `yes`/`no`/`uncertain`。严格三分类只接受 due-now Intent 全部可评测且所有相关 pair 已明确判断的窗口；整步标签由程序聚合：全部覆盖为 faithful，部分覆盖为 partial，零覆盖为 not_executed。无 due-now Intent 或任一 due-now Intent 不可评测为 unassessable。人工填写的整步标签只用于一致性校验，不能覆盖程序结果。

## 已运行流程

```bash
python scripts/build_ta_coverage_pilot.py --output data/clawgym/ta_coverage_pilot_20261013
python scripts/review_ta_coverage_gqq.py --windows data/clawgym/ta_coverage_pilot_20261013/windows.jsonl --output data/clawgym/ta_coverage_pilot_20261013/model_reviews_v2.jsonl --workers 6
python scripts/render_ta_coverage_review.py --windows data/clawgym/ta_coverage_pilot_20261013/windows.jsonl --model-reviews data/clawgym/ta_coverage_pilot_20261013/model_reviews_v2.jsonl --output data/clawgym/ta_coverage_review30_v3_20261013
python scripts/predict_ta_coverage_tfidf.py --windows data/clawgym/ta_coverage_pilot_20261013/windows.jsonl --output data/clawgym/ta_coverage_pilot_20261013/tfidf_predictions.jsonl --threshold 0.2
```

旧版模型审核文件保留原始的 `model_raw_ta_label` 和 `aggregation_disagrees`，程序按逐 pair 判断计算 `suggested_ta_label`；新版提示词不再要求模型直接给整步标签。在生成 30 条人工复核队列时，200 条中有 195 条通过结构校验，5 条因模型输出额外 JSON 而失败；39 条有效结果的模型整步标签与其逐 pair 结果冲突。该批通过结果的程序聚合分布是 faithful 137、partial 17、not_executed 39、unassessable 2；5 条失败后来已重试成功，当前完整结果见下方版本说明。**以上均是模型建议，不是 Gold。** 30 条人工复核队列覆盖 30 条不同轨迹，优先包含多 Intent/多 Action、聚合冲突、模糊和模型失败案例。

打开 `data/clawgym/ta_coverage_review30_v3_20261013/review.html` 即可复核；界面直接展示 Thought、due-now Intent、原始 Action 和工具返回。进度暂存浏览器本地，完成后用“导出标注 JSON”保存文件；模型建议折叠在末尾，以降低锚定效应。评测器会拒绝未完成标注或与 pair 覆盖不一致的 Gold。人工复核前不能报告真实性能。

当前导出的 `ta_gold_human_review.jsonl` 有 **30 行人工整步三分类 Gold**（faithful 19、partial 3、not_executed 8）；此前两条空标签已根据人工复核补为 faithful 和 not_executed。整步 Gold 可直接用于步骤级评测；不要求同时填写逐 Intent/pair 字段或判断依据。另有 28 行缺 `review_reason`、78 个 Intent 状态为空、120 个 pair 状态为空，所以当前不能对多数案例做严格的逐项覆盖归因，也不能检验其整步标签是否与逐项覆盖一致。不自动用模型建议填充这些逐项字段。

## 独立预测与评测

本机无 `torch`/`transformers`，因此已运行的 `tfidf_predictions.jsonl` 仅用于验证独立预测文件链路，阈值 0.2 未在 Gold dev 上校准，不能用于模型选型。GPU 环境可运行 BGE-M3 相似度 baseline（阈值先用独立 dev Gold 选择）：

```bash
python scripts/predict_ta_coverage_bge.py --windows data/clawgym/ta_coverage_pilot_20261013/windows.jsonl --threshold <dev确定的阈值> --output outputs/ta_bge_predictions.jsonl --device cuda
```

用现有人工整步 Gold 计算 Accuracy、Macro-F1、混淆矩阵及按工具类型分组的准确率：

```bash
python scripts/evaluate_ta_coverage_pilot.py --windows data/clawgym/ta_coverage_pilot_20261013/windows.jsonl --human data/clawgym/ta_coverage_review30_v3_20261013/ta_gold_human_review.jsonl --predictions data/clawgym/ta_coverage_pilot_20261013/tfidf_predictions.jsonl --gold-source human-overall --output outputs/ta_coverage_review30_overall_metrics_complete.json
```

这次得到 30 条的 Accuracy 0.367、Macro-F1 0.354；旧的 28 条报告保留在 `outputs/ta_coverage_review30_overall_metrics.json`。默认 `--gold-source pair-derived` 仍要求完整逐项人工判断，并校验整步标签与程序聚合一致。这 30 条是 train 轨迹中的定向错误分析队列，**不能充当独立 test 指标**。下一步应从冻结的 dev/test 来源分别建立人工 Gold；dev 校准阈值和开发规则，test 在规则固定后只评一次。步骤级 Macro-F1 经过可靠性验证后，再扩大标注、转换 SWE-Agent 并训练 Cross-Encoder。

200 条窗口的 746 个 pair 全部保留。超过 10 pair 的 14 个窗口已有专门原子化抽查，详见 `docs/2026-10-13-large-window-intent-atomicity-audit.md`；优化目标是减少无效模型审核调用，不在 Gold 候选阶段删掉可能覆盖 Intent 的 Action。

30 条整步人工 Gold 与 GQQ 建议有 8 条冲突；另从未进入人工整步复核的窗口按正负、工具和窗口复杂度分层抽查了 16 个 pair。逐案结论、待裁定项和时间属性导致的缺失 pair 见 `docs/2026-10-13-ta-gqq-conflict-and-pair-audit.md`；暂不入训的候选 ID 见 `data/clawgym/ta_coverage_pilot_20261013/ta_pair_audit_quarantine_20261013.jsonl`。

**审核结果版本说明**：最早的 `model_reviews.jsonl` 有 47 个 `verdict=null`，包含模型整步标签与 pair 聚合不一致而被旧校验器拒绝的情况。`model_reviews_v2.jsonl` 采用追加重试，223 行对应 200 个窗口，按窗口取最后一条时有 5 个格式失败。剩余 5 条已定向重试成功；再叠加问题轨迹的定向复审，`model_reviews_current.jsonl` 是**每窗口一行、200 行、0 个 null verdict** 的当前模型建议。每行 `review_source` 说明来源版本。原始文件保留审计，`windows.jsonl`、`pairs.jsonl` 和 `ta_gold.jsonl` 的 null 标签仍表示**尚无人类 Gold**，不能视为 GQQ 审核失败。
