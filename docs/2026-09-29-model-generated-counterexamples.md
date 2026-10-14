# 模型生成训练反例建议（2026-09-29）

当前 train split 有 5,393 条初判 `exact_match`。阶段 4 已生成 216 条诊断样本，尚不足以作为主要训练反例。优先补充真实数据中较少的 `object_mismatch` 和 `type_mismatch`。

## 建议规模

| 批次 | train 生成量 | 对象错误 | 操作类型错误 |
|---|---:|---:|---:|
| 首批验证 | 2,000 | 1,200 | 800 |
| 验证通过后累计 | 4,000–5,000 | 按首批质量调整 | 按首批质量调整 |

dev/test 各保留约 100–200 条生成样本用于诊断。`partial_match` 单独作为分级标签处理；`no_action` 用于拒识评估，不混入普通二分类负例。

## 生成与验收

1. 从同 split 的初判 `exact_match` 出发，保留完整 raw Action 和来源 ID；donor 不得跨 split。
2. 每条只改变预定维度，并重新计算 Action 哈希与结构审计字段。
3. 模型复判目标标签；不符合目标、证据不可观测或修改超过一个维度的样本剔除。
4. 先抽查首批的标签准确性和各工具族覆盖，再决定是否扩量。

模型生成结果标记为 Silver；当前跳过人工 Gold 全审，正式指标不得将这些样本称为 Gold。

## 2026-09-29 实际产出

已完成 train 负例 5,000 条：`object_mismatch` 3,000 条，`type_mismatch` 2,000 条。对象错误要求原目标文件名在 Intent 中明确出现；操作类型错误仅在 `read` 与 `write` 等真实工具间切换，并保持目标路径。每条均由 GQQ 复判。训练文件 `data/clawgym/bge_train_5000_silver_v1/dataset/train.jsonl` 有 5,000 行，使用 `[INTENT]` / `[ACTION]` 前缀和 `query` / `pos` / `neg` 字段。另有按来源 Action 合并的 `train_grouped.jsonl`（2,401 行），以及独立的 dev/test 正例对文件。数量、哈希、改动维度及 split 隔离见同目录 `manifest.json`。这些数据仍为 Silver，dev/test 正例也来自原有模型标注。
