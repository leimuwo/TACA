# 2026-10-13 清晰 Intent TA dev/test 评测

此任务的单位是一个 Thought 及当步 Action 的窗口。脚本从冻结轨迹 split 重建窗口，按 `(trajectory_id, thought_step, source_intent_id)` 对齐已有清晰 Intent 筛选。**只有窗口内全部 `current` Intent 都清晰**，才保留该窗口；一条模糊或未筛查的当前 Intent 会排除整个窗口。`future` Intent 不参与本步预测。使用同一步全部原始 Action，不使用旧检索 qrels 作为 TA Gold。

已生成待评测集：`data/clawgym/ta_clear_dev_eval_20261013/` 有 573 个 dev 窗口；`data/clawgym/ta_clear_test_eval_20261013/` 有 516 个 test 窗口。各目录的 `windows.jsonl` 含 Thought、Intent、Action 和 pair；`ta_gold_template.jsonl` 的空标签需要人工审核。`report.json` 记录筛选计数。清晰筛选来自既有模型辅助审核，仍应抽检筛选质量。

使用默认 Cross-Encoder checkpoint 生成预测（当前没有 TA dev/test 人工 Gold，因此不会生成 Accuracy/F1）：

```bash
python scripts/evaluate_ta_clear_dev.py --checkpoint outputs/ta_pair_cross_encoder_v1/best --output outputs/ta_clear_dev_cross_encoder_v1 --device cuda
python scripts/evaluate_ta_clear_test.py --checkpoint outputs/ta_pair_cross_encoder_v1/best --output outputs/ta_clear_test_cross_encoder_v1 --device cuda
```

比较 BGE-M3 checkpoint 时，仅替换 `--checkpoint` 和输出目录。每个 `predictions.jsonl` 行包含整步 `predicted_label`、每个 pair 的 logit、阈值和已覆盖 Intent 数；一个 Intent 的任一 pair `logit >= 0` 就视作覆盖。所有当前 Intent 覆盖为 `faithful`，部分为 `partial`，零覆盖为 `not_executed`。默认阈值 0 来自训练脚本；只能在人工 dev Gold 上调阈值，固定后再评 test。

人工对**每个选中窗口**填写 `ta_gold_template.jsonl` 的 `gold_label`（`faithful`、`partial`、`not_executed`，无法可靠判断时可填 `unassessable`），另存为独立 JSONL。正式计算指标时提供 `--human-gold`：

```bash
python scripts/evaluate_ta_clear_dev.py --checkpoint outputs/ta_pair_cross_encoder_v1/best --human-gold data/clawgym/ta_clear_dev_human_gold.jsonl --output outputs/ta_clear_dev_cross_encoder_scored --device cuda
python scripts/evaluate_ta_clear_test.py --checkpoint outputs/ta_pair_cross_encoder_v1/best --human-gold data/clawgym/ta_clear_test_human_gold.jsonl --output outputs/ta_clear_test_cross_encoder_scored --device cuda
```

Gold 必须逐窗口完整覆盖评测集且无重复；`unassessable` 会单列跳过。`report.json` 才会包含步骤级 Accuracy、Macro-F1、混淆矩阵和按工具类型的准确率。建议 dev 用于阈值选择，test 在阈值和模型固定后评估。现有 30 条人工 Gold 来自 train，不能代替这里的 dev/test Gold。
