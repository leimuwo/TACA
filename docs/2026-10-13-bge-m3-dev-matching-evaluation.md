# BGE-M3 双塔当前 dev 匹配与评估流程

日期：2026-10-13

本文说明 `outputs/bge_m3_dual_tower_v1/` 的现有 dev 结果如何产生。依据是 `scripts/train_bge_m3_dual_tower.py`、`scripts/train_nightowl_action_tower.py` 中复用的 `evaluate()`，以及该次训练的 `training_report.json`。这是**全库 Action 召回评估**，并非给定一对 Intent/Action 的二分类，也没有先按原轨迹、工具类型或步骤筛选候选。

## 1. 读取数据与标签

该次训练配置使用 `data/clawgym/bge_dataset_split_20260930_v2/dev/`：`queries.jsonl` 有 1,078 条 Intent，`actions.jsonl` 有 1,944 个候选 Action。每条查询由 `query_id` 标识，每个 Action 由 `action_id` 标识。另从 `data/clawgym/bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl` 读取正例关系：一个查询可对应多个已知正确 Action ID。程序校验 ID 唯一、qrels 引用存在、每条查询至少有一个已知正例。

qrels 是 Silver 已知正例集合。未出现在某条查询 qrels 中的 Action 状态是**未判定**，不能从 `is_known_positive: false` 推出它一定错误。已知正例可以在候选池中，但未必进入模型保存的 Top-20。

## 2. 编码与打分

每个 epoch 结束时，Intent 用微调后的 BGE-M3 Intent 塔编码；Action 用独立的 BGE-M3 Action 塔编码，并经过训练得到的线性投影层。两侧分别取最后一层的第一个 token（CLS）向量并做 L2 归一化；投影后的 Action 向量再归一化。该次配置最多保留 Intent 256、Action 512 个 token，Action 输入是序列化的 `[ACTION]` JSON 文本。

对每条 Intent，模型计算它与**全部 1,944 个 dev Action**的点积。由于两侧向量均已归一化，该 `score` 是余弦相似度，不是“正确概率”，没有固定命中阈值。按分数从高到低排序；相同分数按 `action_id` 字典序打破平局。每条查询只保存前 20 个 Action，包含其 ID、分数，以及是否在该查询的 qrels 中。训练中的 batch 内正负例和已知正例屏蔽只用于损失计算，dev 评估时不把训练负例当成 dev 标签，也不先限制候选池。

## 3. 命中与指标

对每条查询，从 Top-20 的第 1 位往后找**第一个**出现在 qrels 中的 Action。如果找到，记录其名次 `r`；如果 20 位都没有，则本次 `r = None`。有多个已知正例时，任意一个进入即可，取最靠前者。

`known_positive_in_candidates` 是 `r` 存在的查询数；`known_positive_at_1/5/10` 分别是 `r <= 1/5/10` 的查询比例；`known_positive_mrr_at_20` 是每条命中查询的 `1/r` 之和除以**全部**该组查询数，未进入 Top-20 的查询贡献 0。所有指标都有 `overall` 与 `exec_positive` 两组：后者指 qrels 中至少有一个 `exec` 正例的查询，不表示候选池只包含 `exec` Action，也不要求排第一的 Action 为 `exec`。

当前最佳轮为第 3 轮，按 dev `exec_positive` 的已知正例 MRR@20 选出。全体 1,078 条中，797 条已知正例进入 Top-20，Top-1 比例 0.4184，MRR@20 为 0.5026；`exec_positive` 437 条中，280 条进入 Top-20，Top-1 为 103 条（0.2357），MRR@20 为 0.3387。因此 `exec` 有 157 条**已知正例**未进入 Top-20。这些数字描述的是当前 qrels 下的全库检索表现，不等同于对所有候选的人工正确性判断。

## 4. 输出文件与任务边界

- `outputs/bge_m3_dual_tower_v1/training_report.json`：各 epoch 的 dev 指标、选模依据和训练配置；其中的 BGE baseline 是从另一份固定排名文件按同一 qrels 重新统计，并非参与 BGE-M3 的候选筛选。
- `outputs/bge_m3_dual_tower_v1/best_dev_rankings.jsonl`：最佳轮每条查询的 Top-20 ID、分数和已知正例标记；原文件不含 Intent/Action 全文。
- `outputs/bge_m3_dual_tower_v1/dev_rankings_review.html`：将排名 ID 与 dev 文本关联的离线审阅页，可同时查看 Intent、已知正例和 Top-20 Action 全文。

当前流程适合评估“从共享 Action 库召回正确 Action”。如果上线任务实际只在当前轨迹或当前步骤给出的若干 Action 中选择，应另外固定**上线时真实可获得**的候选生成规则，建立局部候选评测，并报告其候选覆盖率、Top-1 与 MRR。局部候选至少要包含可信的易混淆 Action；只把唯一正确 Action 与 Intent 配对无法检验区分能力。局部评测和本页全库评测回答不同问题，应分别保留；不能仅因全库数值较低就判断打分或排序实现有 bug。修订漏标时，应把经审核的等价 Action 写入新版本 dev qrels，保持 test 和 train 隔离。
