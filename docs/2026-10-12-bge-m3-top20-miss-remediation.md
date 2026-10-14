# BGE-M3 exec Top-20 漏召回处理

日期：2026-10-12

## 当前基线与可核实范围

旧 BGE-M3 双塔 `outputs/bge_m3_dual_tower_v1/best/` 的 dev 集有 437 条已知 `exec` 正例查询：103 条 Top-1 命中，177 条正例在 Top-20 内但未排首位，157 条正例未入 Top-20。已将全部 334 条非 Top-1 查询导出到 `data/clawgym/bge_m3_exec_error_audit_full_20261012/review_queue.jsonl`，其中 157 条为 `top20_miss`。这 157 条首位工具为 `exec` 的有 61 条，其他工具 96 条。旧审计只人工抽查过 20 条，不能将抽查比例外推。

旧排名文件只保存 Top-20，**无法从中推算 Top-50/100 覆盖**。当前本地 Python 没有 PyTorch/Transformers，因此尚未重新编码 1,944 个 Action，也没有新的全库名次数字。

## 1. 全库名次与候选数

在 GPU 环境的 Project 根目录运行，使用旧最佳 checkpoint 和训练时一致的 CLS、L2 归一化、Action 投影、长度和排序规则：

```bash
python -u scripts/audit_bge_m3_full_ranks.py --checkpoint outputs/bge_m3_dual_tower_v1/best --output outputs/bge_m3_dual_tower_v1_full_rank_audit_20261012 --device cuda
```

`summary.json` 给出 `exec_at_20/50/100` 和原 157 条漏召回中首个已知正例位于 21–50、51–100、100 以后的数量；`rankings_top100.jsonl` 给出所有 dev 查询的 Top-100；`exec_top20_miss_review.jsonl` 附有 157 条的 Intent、已知正例及 Top-10 Action，标签初始均为 `unjudged`。先确认 `exec_at_20 == 280`、`original_top20_misses == 157`，否则应核对 checkpoint、qrels 和编码环境。若 21–100 的数量可观，可用 Top-50/100 跑现有 `bge-reranker-v2-m3` 做固定候选对照：

```bash
python -u scripts/train_reranker_clawgym.py --eval-only --dataset data/clawgym/bge_dataset_split_20260930_v2_exec_supervision_20261012 --rankings outputs/bge_m3_dual_tower_v1_full_rank_audit_20261012/rankings_top100.jsonl --top-k 50 --model outputs/bge_reranker_v2_m3/best --output outputs/bge_m3_old_recall50_rerank_20261012 --device cuda
```

同一命令改为 `--top-k 100` 并更换输出目录可测 Top-100。扩大候选只提高已知正例可达上限，重排后的 Top-1/MRR 仍须实测；未标注等价 Action 不会自动计入已知正例。

## 2. 操作类型监督

`scripts/build_bge_m3_operation_supervision.py` 从 **train** 的明确操作构造 91 条反事实负例：77 条“列出指定脚本，却没有运行”，14 条“列出指定文件，却没有读取内容”。生成时只替换命令，保持 Action 其余参数和目标路径；对同 Intent 的已知正例排除碰撞。明细在 `data/clawgym/bge_dataset_split_20260930_v2_exec_operation_20261012/new_operation_counterfactuals.jsonl`。

这是可复现的字面规则监督，**未经执行效果审核，不是 Gold 标注**。它只覆盖明确指向脚本或内容的 Intent；笼统的 “read input files” 在原 train 中有 `ls` 正例，因此不据此自动标注 `ls` 为负例。若希望对这类模糊 Intent 补负例，须先查看源轨迹和文件内容。

## 3. 完整命令约束

从已有 train 轨迹的待审核近邻队列里挑出 16 条真实 Action，经 Intent 与命令文本逐条核对后记录在 `data/clawgym/bge_m3_train_constraint_reviews_20261012.json`。它们包含错输入/输出路径、错 URL、运行却未保存指定输出、缺少明确要求的输入输出标志等。`scripts/apply_bge_m3_train_reviews.py` 校验来源 edge 属于 train、Action 与正例池一致、同轨迹及正负例不冲突，然后生成 `data/clawgym/bge_dataset_split_20260930_v2_exec_constraints_20261012/`，共 2,895 行、360 行 `exec` 正例。这 16 行此前未在精选训练集中，因而也是 16 个新正例对；仍属于文本审核的 Silver 数据，未验证执行结果。改变 Python 版本但目标相同等可能等价项未纳入。

路径、URL、数值参数和多步骤缺项在现有精选 train 中样本稀少，不能声称本轮已系统覆盖。后续应继续审核 train 近邻队列的明确目标冲突，记录拒绝与接受理由，并对 `exec` 和全体指标做消融比较。

使用最终数据集重新从本地 BGE-M3 基座训练：

```bash
python -u scripts/train_bge_m3_dual_tower.py --dataset data/clawgym/bge_dataset_split_20260930_v2_exec_constraints_20261012 --output outputs/bge_m3_dual_exec_constraints_20261012 --exec-repeat 4 --device cuda --gradient-checkpointing
```

该配置每轮 3,975 个采样行，其中 `exec` 1,440 行（36.2%）。重复采样不增加独立样本数，也可能放大 Silver 标签噪声；应与上一版 `--exec-repeat 4` 数据集对照，并分别检查全体和 `exec` 的 Top-20 覆盖、Top-1、MRR@20。新模型尚未训练，不能宣称已改善召回。

## 4. dev 标签复核

对 `exec_top20_miss_review.jsonl` 逐项核实 Top-10 未标注 Action：是否执行同一操作、命中同一目标、满足全部输出/参数要求；模糊 Intent 需回看源轨迹。只有确定等价时把该行设为 `"review_label": "equivalent"`，填写 `"equivalent_action_ids": ["..."]` 和 `"review_reason"`；其余保持 `unjudged` 或记录其他审核类别。确认后用 `scripts/apply_bge_m3_dev_reviews.py --reviews <已审核文件> --output <新的评测目录>` 生成新版 dev qrels。脚本拒绝未审核、无理由、重复或未知 Action ID。原 qrels 不覆盖，**dev 审计样本不回流 train**。
