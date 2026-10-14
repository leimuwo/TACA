# BGE reranker 训练

从 `Project` 根目录运行。脚本默认使用本地 `bge-reranker-v2-m3`，以现有 v2 Silver 训练集的 `query` 为 Intent、`pos` 和显式 `neg` 为结构化 Action。Action 输入将工具名、路径、URL 和命令移到长内容前，联合编码时优先截断 Action；原始数据不改。单 logit 使用 BCE 损失。未标注的 dev 候选不会自动当作负例。

```bash
pip install -e '.[training]'
python scripts/train_reranker_clawgym.py --output outputs/bge_reranker_v2_m3 --dry-run
python scripts/train_reranker_clawgym.py --output outputs/bge_reranker_v2_m3 --device cuda
```

默认 dev 输入是 `outputs/bge_clawgym_silver_v2_dev_top20_20261009/rankings.jsonl` 的固定 Top-20，正例来自 `data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl`。每轮报告召回候选覆盖率、重排前后 Top-1/5/10 和截断 MRR，按重排后 MRR 选择 `best/`；保存 `best_dev_rankings.jsonl` 与 `training_report.json`。这些是 Silver 已知正例指标，候选之外的正例无法靠重排恢复。test 不用于训练或 checkpoint 选择。

默认 batch 4、梯度累积 8、3 轮、学习率 1e-5、最大联合长度 512；显存不足可减小 batch 与评估 batch。训练会加载约 568M 参数模型，需要配置足够的 GPU 显存。`--dry-run` 仅校验数据和 Top-20 合约，不加载模型。

训练每 25 个微批次打印损失、已用时间和本轮预计剩余时间；可用 `--log-every` 调整。dev 评估每 100 个批次报告进度。日志从修改后的新进程开始生效，已经运行的进程不会动态加载脚本改动。

## 下一轮对照实验

先保持原 Top-20 候选不变，只比较训练目标。`--loss pairwise` 对每组已验证的正负例优化 `softplus(negative_score - positive_score)`；`--exec-weight 3` 提高正例为 `exec` 的组的权重，不会把未标注候选当负例。组 batch 2 通常包含 4–8 个编码对，梯度累积 16；从原始本地基座开始训练，输出到新目录：

```bash
python -u scripts/train_reranker_clawgym.py --output outputs/bge_reranker_pairwise_top20 --device cuda --loss pairwise --exec-weight 3 --batch-size 2 --accumulation 16 --epochs 2
```

无论 pairwise 是否更好，都可单独考察扩大候选池。先使用**同一个**已微调 BGE 召回模型生成 Top-50 排名，再用两个已训练 checkpoint 分别评估；所有输出目录必须不存在：

```bash
python -u scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split dev --model outputs/bge_clawgym_silver_v2/best --qrels data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl --top-k 50 --output outputs/bge_clawgym_silver_v2_dev_top50 --device cuda
python -u scripts/train_reranker_clawgym.py --eval-only --model outputs/bge_reranker_v2_m3/best --output outputs/bge_reranker_bce_dev_top50 --device cuda --rankings outputs/bge_clawgym_silver_v2_dev_top50/rankings.jsonl --top-k 50
python -u scripts/train_reranker_clawgym.py --eval-only --model outputs/bge_reranker_pairwise_top20/best --output outputs/bge_reranker_pairwise_dev_top50 --device cuda --rankings outputs/bge_clawgym_silver_v2_dev_top50/rankings.jsonl --top-k 50
```

比较训练目标时使用相同 Top-20；比较候选覆盖率时才切换 Top-50。Top-50 的 dev 评估规模约 53,900 个编码对，明显比 Top-20 慢。扩大候选池不改变训练标签，且 Silver qrels 仍不足以判定所有未标注候选是否相关。若计划补充难负例，应先逐对审核，再写入训练集。

## 清晰 test 评估

清晰 test 先在完整 test 候选池中生成召回排名，再按 `../data/clawgym/ambiguous/ambiguous_test.jsonl` 排除 270 个模糊 query，保留 804 个清晰 query。筛选记录的 `source_query_id` 对应检索数据的 `query_id`。训练仍只使用 dev；test 只在方案固定后评估。以下从 Project 根目录运行：

```bash
python -u scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split test --model outputs/bge_clawgym_silver_v2/best --qrels data/clawgym/bge_test_qrels_exact_text_20261008/qrels_exact_text_closed.jsonl --top-k 20 --output outputs/bge_clawgym_silver_v2_test_top20_exact --device cuda
python -u scripts/train_reranker_clawgym.py --eval-only --split test --screening ../data/clawgym/ambiguous/ambiguous_test.jsonl --model outputs/bge_reranker_v2_m3/best --rankings outputs/bge_clawgym_silver_v2_test_top20_exact/rankings.jsonl --qrels data/clawgym/bge_test_qrels_exact_text_20261008/qrels_exact_text_closed.jsonl --top-k 20 --output outputs/bge_reranker_v2_m3_clear_test_top20 --device cuda
```

`metrics.json` 中的 `retrieval` 是不重排的清晰 test Top-20，`reranked` 是同一候选池经重排后的结果。两者共享 804 个 query、相同 Action 池和修正后的 qrels。评估目录必须事先不存在；指标仍是 Silver known-positive 指标。
