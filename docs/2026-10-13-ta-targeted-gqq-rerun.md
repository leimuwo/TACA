# 2026-10-13 问题轨迹 GQQ 定向重审

针对原子化审计发现的 3 个问题窗口，以及 GQQ 整步标签与逐 pair 聚合冲突的 2 个窗口，使用 `review_ta_coverage_gqq.py --window-id` 定向重跑。旧结果保留在 `data/clawgym/ta_coverage_pilot_20261013/model_reviews_v2.jsonl`；第一轮新结果在 `model_reviews_targeted_v3.jsonl`，发现多 Action 联合覆盖问题后对两条再次运行，结果在 `model_reviews_targeted_v4.jsonl`。全部是模型审核建议，未修改 `windows/intents/actions/pairs/ta_gold` 或人工 Gold。

| 窗口 ID 前缀 | 原模型原始标签 / 程序聚合 | 最新模型原始标签 / 程序聚合 | 结果与剩余问题 |
| --- | --- | --- | --- |
| `431df21a` | not_executed / not_executed | unassessable / unassessable | 18 个 Thought 内部计算目标不再被当作未执行工具调用；仍需人工确认这些目标的任务边界 |
| `44ec3e01` | partial / partial | unassessable / unassessable | “读取 evolution 文件们”和“探索技能目录”都缺可唯一判定的完整目标；需人工决定是否拆成具体文件 Intent |
| `58405a66` | partial / partial | partial / partial | 5 个写文件目标各由一个 Action 覆盖；“分类全部消息”没有单个 Action 完整覆盖。模型仍标其为可评测，**原子性争议未解决**：需要人工决定拆分为可独立判断的结果目标，或标记不可评测。不能直接把 `partial` 当 Gold |
| `58f2510f` | partial / faithful（冲突） | faithful / faithful | 4 个命名文件读取各有一个对应 Action，聚合冲突消失 |
| `b46016ec` | partial / faithful（冲突） | faithful / faithful | 4 个命名文件读取各有一个对应 Action，聚合冲突消失 |

5 条定向重审均通过 ID/布尔值及 pair 聚合结构校验；其中 `44ec3e01` 和 `58405a66` 在第二轮加入“**单个 Action 必须完整覆盖整个 Intent**”的规则。`58405a66` 说明提示词不能替代原子化人工裁定，特别是一个目标需要多个 Action 联合证明时。新模型建议不回填原始 200 条审核文件，也不参与训练或指标计算，待人工审核后另存修订版标签。
