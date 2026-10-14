# 10.13 改进思路实施记录

日期：2026-10-13。对应 `docs/10.13改进思路.md`。本轮以旧 BGE-M3 最佳 checkpoint 做离线评测，不改写原始 dev/test，也不把 dev 错例回流 train。

## P0：清晰 Intent 与候选池

沿用已有模糊性筛查 `../data/clawgym/ambiguous/` 的完整决策，严格按 `source_query_id` 对齐原 dev 查询；只筛选，不补全 Intent。`data/clawgym/bge_m3_clear_dev_20261013/intent_clear.jsonl` 含 825 条清晰 Intent，剔除 253 条。其余原始文本和 Action 池保留。

`scripts/evaluate_bge_m3_local.py` 实现四种候选规则：同轨迹且同正例工具（**仅诊断，使用标签信息**）、同轨迹邻近步骤、旧全库语义 Top-K、邻近步骤与语义召回混合。邻近规则排除查询自己的 Thought step；全部 1,078 条原 dev 查询都有同一步已知正例，若把同一步 Action 直接入池会导致明显的标签泄漏。混合规则不使用 qrels 的正例工具。K 可取 5/10/20，输出 `candidates.jsonl`、`local_report.json`；GPU 环境可重新为局部候选打分并输出 `rankings.jsonl`。

目前离线可复用旧排名分数计算语义 Top-K 的**清晰子集**结果：

| 候选 | 已知正例覆盖 | Top-1 | MRR@K | `exec` 覆盖 | `exec` Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 语义 Top-5 | 69.9% | 49.8% | 0.5744 | 57.9% | 29.6% |
| 语义 Top-10 | 79.8% | 49.8% | 0.5878 | 70.4% | 29.6% |
| 语义 Top-20 | 84.1% | 49.8% | 0.5910 | 77.2% | 29.6% |

同轨迹邻近 + 语义的混合候选覆盖率分别为 41.2%、71.9%、82.9%，K=20 仍低于文档提出的 90% 门槛。邻近步骤单独构池仅覆盖 1.5%；原因是已知正例基本来自查询当步，排除当步后需要由独立候选生成器找回。**当前不能将这个混合规则定为上线候选方案。** 若上线时当前步骤的待选 Action 集由外部系统给出，应以那份实际候选集重算主指标；若上线要从共享库检索，应继续改进召回，保留全库指标。

离线重现语义结果：

```bash
python scripts/evaluate_bge_m3_local.py --method semantic --k 20 --score-from-rankings --output outputs/bge_m3_local_clear_semantic20_new
```

GPU 上对混合候选重新编码打分，输出目录必须不存在：

```bash
python -u scripts/evaluate_bge_m3_local.py --method hybrid --k 20 --output outputs/bge_m3_local_clear_hybrid20_scored --device cuda
```

全库 dev 评测及旧结果继续保留在 `outputs/bge_m3_dual_tower_v1/`，两种口径分别报告，不直接比较不同候选池下的 MRR。

## P1：难负例与 exec 训练

`data/clawgym/bge_dataset_split_20260930_v2_clear_exec_constraints_20261013/` 是清晰训练集：由已有错脚本、错文件、错操作反事实和经文本审核的 train 近邻组成，再剔除模糊 Intent。2,895 行中保留 1,968 行，删去 911 条模糊行及 16 条未经过现有筛查的新增 Intent 行；dev/test 原文件完全不变。保留的 `exec` 为 259 行、107 个不同的 Intent/正例对。设置 `--exec-repeat 4` 会使每轮采样 1,036 个 `exec` 行，**这不是 1,036 个独立正例，也未达到 600+ 新正例目标**。

`data/clawgym/bge_m3_hard_negatives_clear_20261013/hard_negatives.jsonl` 把现有显式负例按 Intent/正例对归并，含 1,816 个组、5,263 个负例对；204 组少于 3 个负例。类别仅描述 Action 序列化差异，不能当成人工判定的错误原因。未审近邻不自动入训；后续从 train 轨迹补满 3–10 个可靠难负例时，应先确认同义 Action、目标与参数，再按来源 ID 留审核记录。不用随机负例冒充难负例。

新训练可在 GPU 环境运行，最佳轮按**清晰 dev** 的 `exec` 已知正例 MRR@20 选择，输出目录必须不存在：

```bash
python -u scripts/train_bge_m3_dual_tower.py --dataset data/clawgym/bge_dataset_split_20260930_v2_clear_exec_constraints_20261013 --clear-dev-screen-input ../data/clawgym/ambiguous/screen_input.jsonl --clear-dev-decisions ../data/clawgym/ambiguous/decisions.jsonl --output outputs/bge_m3_clear_exec_constraints_20261013 --exec-repeat 4 --device cuda --gradient-checkpointing
```

训练尚未在本环境运行；当前只有数据 dry-run，不能报告新 checkpoint 的效果。训练前仍需复核新引入的 Silver 反事实，尤其长命令的等价副作用。旧 checkpoint 与新 checkpoint 必须在同一清晰 dev 和同一候选规则下比较。

## P1：步骤级一致性与三方审核

`data/clawgym/bge_m3_step_consistency_tri_review_20261013.jsonl` 已导出 825 条清晰 Intent 与旧模型 Top-1 Action 全文，并关联原 Thought、用户请求及同一步抽取 Intent；字段 `due_now_review`、`gold_label`、`predicted_label`、`review_reason` 待填。抽取 Intent 不自动等同 due-now Gold。分类口径：

- `faithful`：这一步的 Action 完整执行当前应做 Intent 的目标、对象和关键约束；允许实现细节不同。
- `partial`：完成部分目标或准备工作，但缺少必需对象、操作、输出或参数。
- `not_executed`：目标/对象明显不同，或 Action 没有执行该 Intent。

人工审核须查看 Thought 原文、当前 due-now Intent、Action 与执行结果；非平行 Intent 不能强行作为单选标签，准备动作不能当作完成，只读操作不能凭工具名判为“额外动作”，字段误标先修复再入训练。现有 `docs/9.28intent_action_matching_rules_v1.md` 已规定同一步多对多覆盖及 due-now 口径，应在审核中沿用。对模糊 Intent 保持剔除，不从后续 Thought/Action 反推补全。

只有完成 Gold 标签并提供独立的三分类预测后，才能运行：

```bash
python scripts/evaluate_step_consistency.py --reviews <已填三分类文件.jsonl> --output outputs/bge_m3_step_consistency_metrics.json
```

脚本计算 Accuracy、每类 Precision/Recall/F1、Macro-F1、混淆矩阵和按 Action 工具分组的准确率。旧双塔只输出相似度排序，没有三分类头；`is_known_positive` 不能替代 `faithful/partial/not_executed` 真值或预测。因此本轮**尚无步骤级 F1**。人工复核的 dev 结论只用于评测标签修订和错误归因，训练样本须从 train 轨迹独立构造。
