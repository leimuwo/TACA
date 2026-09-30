# BGE Silver 数据划分与运行方式

从 Project 根目录运行。`train/train.jsonl` 是 `query` / `pos` / `neg` 训练数据；`train_grouped.jsonl` 保留来源和负例类别。dev 用于每轮选择最佳 checkpoint；test 仅在训练方案固定后评一次。两个评测 split 的 `queries.jsonl`、`actions.jsonl` 和 `qrels.jsonl` 分别是意图、固定 Action 候选池和已知正例关系。

## 安装依赖

在支持训练的机器上创建 Python 3.10+ 环境，安装 `pip install -e '.[training]'`。GPU 机器须安装与本机 CUDA 匹配的 PyTorch；本项目基线使用本地模型 `/inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5`；当前数据的意图和 Action 主要为英文。

## 离线检查

```bash
python scripts/train_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --output outputs/bge_clawgym_silver_v2 --base-model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --dry-run
python scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split dev --model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --output outputs/bge_base_dev --validate-only
python scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split test --model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --output outputs/bge_base_test --validate-only
```

## 训练、选择 checkpoint、最终评测

```bash
python scripts/train_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --output outputs/bge_clawgym_silver_v2 --base-model /inspire/hdd/project/control-technology/public/model_hub/bge-base-en-v1.5 --device auto
python scripts/evaluate_bge_clawgym.py --dataset data/clawgym/bge_dataset_split_20260930_v2 --split test --model outputs/bge_clawgym_silver_v2/best --output outputs/bge_clawgym_silver_v2_test --device auto
```

训练默认 3 轮、batch 8、梯度累积 4、学习率 2e-5、最大长度 512。每轮评 dev，按 `known_positive_mrr` 保存 `best/`。训练日志和 dev 分数写入 `training_report.json`；test 的 `metrics.json`、`rankings.jsonl` 写入指定评测目录。所有输出目录须事先不存在。若显存不足，可调低 `--batch-size` 或 `--eval-batch-size`，并适当增加 `--accumulation`。训练前的 dev 基线已保存在 `outputs/bge_base_dev/metrics.json`；与训练后的 dev 指标比较时须使用同一模型、数据、长度及池化设置。

训练使用显式负例和 batch 内 Action 负例，并屏蔽同一 query 的已知正例。`qrels` 仅列出已有模型初标的 `exact_match` 正例；未列出的 Action 是**未判定**。`known_positive_*` 指标只能用于 Silver 诊断与模型比较，不能作为 Gold Recall/MRR。正式评测需要补齐人工相关性标注。`manifest.json` 记录划分数量和输入、输出 SHA-256。
