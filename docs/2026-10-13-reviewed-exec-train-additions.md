# 2026-10-13 train 轨迹 exec 样本补充

基于 `bge_dataset_split_20260930_v2_clear_exec_constraints_20261013`，只从 `positive_pool.jsonl` 中 `split=train`、`eligibility=candidate` 的真实轨迹边选取新增正例。先按 `(Intent, 规范化 Action)` 与现有 train 去重，再核对同轨迹近邻 Action、原始 Thought 摘录和 Action 参数。新增记录的审核理由见 `data/clawgym/bge_m3_exec_train_reviews_20261013.json`，来源 ID 与实际训练三元组见新数据集的 `reviewed_additions.jsonl`。

本批纳入 5 个独立 exec 正例对和 6 个负例，包括：漏报告文件重定向、运行脚本与查看帮助的操作差异、下载 PDF 与读取 PDF 的操作差异、相同搜索工具但错误主题、执行 wrapper 与只看日志的差异。训练行从 1968 增至 1973；exec 正例行从 259 增至 264，独立 exec 正例对从 107 增至 112。`dev` 和 `test` 逐文件哈希保持一致。

本批为**人工文本审核，不是命令执行验证**。现有模糊 Intent 筛查只覆盖已入选训练样本，新增 5 条没有既有筛查记录，因此逐条记录了明确目标、操作和参数的文本审核依据。没有将其余未审核近邻、dev/test 错例或仅凭相似度推断的负例加入训练。候选池中也发现 Intent 与 Action 不符的边，例如“读取草稿文件”对应 `ls` 目录命令；这类边未纳入。

构建与复核：

```bash
python scripts/build_reviewed_exec_train.py --output data/clawgym/bge_dataset_split_20260930_v2_clear_reviewed_exec_20261013
python -m unittest tests/test_reviewed_exec_train.py
```

产物已经生成在 `data/clawgym/bge_dataset_split_20260930_v2_clear_reviewed_exec_20261013`。这批规模很小，不能预设指标提升；继续扩充应优先增加独立且经过审核的 train 轨迹，并先完成新增 Intent 的模糊性筛查。GPU 训练时将 `scripts/train_bge_m3_dual_tower.py` 的 `--dataset` 指向此目录，再按原评测协议比较 dev 指标。
