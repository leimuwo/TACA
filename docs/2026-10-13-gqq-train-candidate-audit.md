# 2026-10-13 GQQ train 候选审核试跑

`scripts/review_exec_candidates_gqq.py` 从 ClawGym 的 train 正例池和同轨迹近邻队列取候选，先排除已入训的 `(Intent, Action)`、dev/test 轨迹以及与同 Intent 已知正例相同的负例，再调用 GQQ 分别判断 Intent 清晰度、正例匹配和每个负例的安全性。脚本读取上级 `.env` 中的 `INF_API_KEY`，支持续跑，不改训练集。审核文件保留来源 edge ID、轨迹 ID、Action 文本与模型理由，不保存密钥。

本地试跑命令：

```bash
python scripts/review_exec_candidates_gqq.py \
  --dataset data/clawgym/bge_dataset_split_20260930_v2_clear_reviewed_exec_20261013 \
  --output data/clawgym/gqq_exec_review_batch20_20261013.jsonl --limit 20
```

20 条候选均获得响应。GQQ 建议 15 个正例对可继续审核、26 个负例安全；标出 3 条模糊 Intent 和 3 条正例不匹配。**这些不是已确认训练样本。** 人工核查发现它曾将“Check if the input files exist”判为清晰，也曾将 `xxd` / `od -c` 错判为“检查空白格式”的负例。调整提示后后者已不再放行，但前者仍漏判，因此不能按 GQQ 的布尔值自动入训。`gqq_exec_review_pilot_20261013.jsonl` 与 `gqq_exec_review_pilot_v2_20261013.jsonl` 保留了这两轮试跑结果。

扩量时应按来源轨迹分批处理；对 GQQ 建议入训的记录复核 Thought、Intent、正例和每个负例，尤其检查模糊指代、等价命令、部分完成、错目标和漏参数。人工确认后再写入审核清单，由 `build_reviewed_exec_train.py` 做去重与来源校验。SWE-Agent 轨迹还需先转换为同一候选格式，并以 issue/轨迹为单位防止 split 泄漏；本脚本当前只处理 ClawGym。

针对首批误判，已将 prompt 改为分步判定：先仅从 Intent 原文提取目标证据，再判断正例是否满足全部明确条件，最后按操作结果判断负例是否可能等价。模糊 Intent 的负例必须全部标为不安全，程序也会校验这一点。定向重审后，“Check if the input files exist” 被判为模糊，`xxd` 与 `od -c` 均不再作为检查空白格式的安全负例。旧审核文件不作原地覆盖；这些结果只证明两个已知误判样本得到纠正，不能替代更大规模的人工抽检。
