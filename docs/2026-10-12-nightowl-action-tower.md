# NightOwl Action 塔实验

使用 `Shuu12121/NightOwl-CodeEmbedding` 作为 Action 塔，现有微调 BGE-base checkpoint 作为 Intent 塔。NightOwl 输出 768 维 CLS 向量，与 BGE-base 相同；两个模型原本不共享向量空间，所以脚本冻结 NightOwl、预编码训练和 dev Action，重新训练 BGE Intent 塔对齐该空间。训练仅使用现有 train 的明确正负例；dev 选 checkpoint，test 不参与。Action JSON 中工具名、命令、路径和 URL 排在长正文前，最大长度 512。模型 revision 固定为 `cfc3c6d172a93c79826db380ce82b6fee377ae1c`。

NightOwl 已下载至 `/inspire/hdd/project/control-technology/public/model_hub/NightOwl-CodeEmbedding`，脚本默认从该目录离线加载。ModernBERT 需要较新的 Transformers，建议 `transformers>=4.56.2`。从 Project 根目录运行：

```bash
python -m pip install -U 'transformers>=4.56.2' torch
python -u scripts/train_nightowl_action_tower.py --output outputs/nightowl_action_tower_v1 --dry-run
python -u scripts/train_nightowl_action_tower.py --output outputs/nightowl_action_tower_v1 --device cuda
```

报告同时给出全体 1,078 条 dev 与已知正例包含 `exec` 的 437 条 dev 查询的 Top-1/5/10、MRR@20。默认按后者 MRR@20 选 `best/query_model/`。原 BGE 对照在相同 qrels 和 Top-20 上为：全体 Top-1 44.43%、MRR@20 0.5309；`exec` 子集 Top-1 22.20%、MRR@20 0.3327。要判断 NightOwl 是否有帮助，先比较这两个子集指标，再看全体指标是否退化。

这属于异构双塔训练实验，并非直接替换模型文件后沿用旧向量索引。部署时所有 Action 必须用 NightOwl 重新编码，Intent 用新训练的 BGE 查询塔编码。模型的公开基准是代码检索；本数据的 `exec` 多为 shell 命令和 JSON 工具调用，因此实际提升需要本项目的 dev/test 实证。

## 线性投影验证

前一实验的最佳轮全体 dev MRR@20 为 0.0393，`exec` 子集为 0.0131，低于原 BGE 基线。下一步先冻结两个编码器，仅训练 NightOwl 后的 768 x 768 无偏置线性投影。训练同时最小化已有样本的检索交叉熵和投影向量相对 BGE Action 向量的余弦距离；第一轮不加入全库采样负例。可以用 `--train-query-tower` 额外微调 Intent 塔，但应在只训练投影的结果出来后再尝试。

在 GPU 环境从 Project 根目录运行：

```bash
python -u scripts/train_nightowl_projection.py --output outputs/nightowl_projection_v1 --device cuda
```

输出 `best/projection.pt`、`best_dev_rankings.jsonl` 和 `training_report.json`。默认按 dev `exec` 正例 MRR@20 选择最佳轮；报告同时附原 BGE 基线。推理时 Intent 仍用原 BGE checkpoint，Action 使用 NightOwl 的 `action_input()` 序列化与 CLS/L2 向量，再依次通过线性投影、L2 归一化。若启用 `--train-query-tower`，Intent 改用输出中的 `best/query_model/`。仍须对所有 Action 重建索引；此投影 checkpoint 不能直接用于原 BGE Action 索引。
