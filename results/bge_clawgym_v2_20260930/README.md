# ClawGym BGE v2 结果

数据位于 `data/clawgym/bge_dataset_split_20260930_v2/`，逐条低排名审核位于 `data/clawgym/bge_v2_error_audit_20260930/`。训练和评测脚本分别为 `scripts/train_bge_clawgym.py`、`scripts/evaluate_bge_clawgym.py`。

| 目录 | 内容 |
| --- | --- |
| `baseline_dev/` | 原始 `bge-base-en-v1.5` 在 dev 的指标和逐 query 排名 |
| `trained_v2/` | 3 轮训练报告；最佳为第 3 轮，dev 已知正例 MRR 0.516342 |
| `trained_v2_test/` | 上述最佳模型在 test 的指标和逐 query 排名；已知正例 MRR 0.536151 |
| `trained_e5/` | 5 轮重新训练的报告；最佳为第 3 轮，dev 已知正例 MRR 0.523721；尚无该模型的 test 结果 |

原始模型 dev 已知正例 MRR 为 0.369393。所有分数只使用 Silver 已知正例，未列出的 Action 未经判定，不能作为 Gold 检索分数。`trained_e5/` 与 `trained_v2/` 是两次独立训练，不能把第 5 轮实验理解为接续训练。

两个最佳 checkpoint 未放入 Git 对象，分别作为 [GitHub Release `bge-clawgym-v2-20260930`](https://github.com/leimuwo/TACA/releases/tag/bge-clawgym-v2-20260930) 的 `bge_clawgym_silver_v2_best.tar.gz` 和 `bge_clawgym_silver_v2_e5_best.tar.gz` 附件。每个包解压后含 `config.json`、`model.safetensors` 与 tokenizer 文件。`artifact_manifest.json` 记录权重、数据和结果文件 SHA-256。
