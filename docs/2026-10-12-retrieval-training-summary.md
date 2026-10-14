# Intent–Action 召回与重排实验总结

日期：2026-10-12

## 目标与评估口径

目标是从 Action 库中检索与 Intent 对应的 Action，尤其改善 `exec` 命令、路径、URL 和参数的区分。当前 train 是模型辅助生成的 Silver 正例及已验证显式负例；同一 Intent 可能有多个等价 Action，未标注 Action 不能直接认定为负例。

本文主要比较 1,078 条 dev 查询、1,944 个 dev Action 和修正了完全相同 Action 文本漏标的 qrels。其中 437 条查询至少有一个已知 `exec` 正例。“正例进入 Top-20”与 MRR@20 均只针对**已知**正例；Silver 漏标会使这些数值偏低。test 不用于以下 NightOwl 实验的选模。早期 BGE 报告和单独的 `metrics.json` 可能使用不同 qrels 或不限 Top-20 的 MRR，不应与下表直接相减。

## 已做训练方法

| 阶段 | 模型与训练方式 | 作用和限制 |
| --- | --- | --- |
| BGE 双塔 | 同一个微调后的 `bge-base-en-v1.5` 编码 Intent 与 Action；使用 train 正例、显式负例和批内候选做对比学习，训练时屏蔽同 Intent 的其他已知正例。两侧都产生梯度，但共享同一套 BGE 权重，并非训练两个独立的塔。 | 当前召回基线；仍有 `exec` 同工具不同目标、命令和步骤混淆。 |
| BGE Cross-Encoder 重排 | `bge-reranker-v2-m3` 联合编码 Intent 和 Action；初版用 BCE，后续比较 pairwise、`exec` 权重及补充已验证 `exec` 负例。只对固定 BGE Top-20 候选重排。 | 改善已召回候选的顺序；无法找回候选池外的正例。 |
| NightOwl 直接替换 Action 塔 | 冻结 `NightOwl-CodeEmbedding` Action 塔，预编码 Action，只训练 BGE Intent 塔；每条训练样本比较一个正例和 1–3 个显式负例。 | 两塔原本不共享向量空间，训练目标也远小于全库检索任务；结果严重退化。 |
| NightOwl 线性投影（当前方法三） | 冻结 NightOwl 与 BGE 两个编码器，在 NightOwl 的 768 维 Action 向量后训练一个 768×768 无偏置线性层。损失为显式正负例检索交叉熵，加投影 Action 对 BGE Action 向量的余弦对齐损失；默认不训练 Intent 塔。 | 比直接替换显著改善，但仍只用每条样本的 1–3 个显式负例；尚未加入全库采样负例。 |

### 同口径 dev 结果

| 方法 | 全体 Top-20 覆盖 | 全体 Top-1 | 全体 MRR@20 | `exec` Top-20 覆盖 | `exec` Top-1 | `exec` MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 BGE 双塔 | 814/1078（75.5%） | 44.43% | 0.5309 | 275/437（62.9%） | 22.20% | 0.3327 |
| NightOwl 直接替换，最佳第 3 轮 | 137/1078（12.7%） | 1.95% | 0.0393 | 20/437（4.6%） | 0.69% | 0.0131 |
| NightOwl 线性投影，最佳第 3 轮 | 585/1078（54.3%） | 21.34% | 0.2892 | 144/437（33.0%） | 7.55% | 0.1242 |

在线性投影实验中，`exec` MRR@20 按轮次为 0.0895、0.1194、0.1242；训练平均 loss 为 1.0341、0.4190、0.3579。三轮仍在改善，尚不能仅凭这三轮断言过拟合。重排实验采用**原 BGE 的固定 Top-20**：候选覆盖为 814/1078，未重排 MRR@20 为 0.5309，BCE 最佳第 1 轮重排后为 0.5762；pairwise 最佳为 0.5692，补充 `exec` 负例的实验最佳为 0.5733。这些重排分数不代表 NightOwl 召回的表现，也不能解决其 Top-20 漏召回。

## 当前问题

1. **NightOwl 投影后的召回覆盖仍不足。** 全体 493/1078、`exec` 293/437 条查询的已知正例未进 Top-20。即使重排器完美，这些正例也无法在 Top-20 内找回。当前结果不足以替换原 BGE 召回。
2. **训练候选与评估候选规模不匹配。** 投影脚本每条样本只对已有的少量显式负例计算检索 loss；dev 则对 1,944 个 Action 排序。投影的 BGE 向量对齐项缓解空间不匹配，但不能保证所有相似 Action 的全库排序。
3. **难负例有假负例风险。** 完全相同 Action 文本曾因不同 ID 漏标；`exec` 也可能有不同命令实现同一 Intent。把未标注的同工具高分候选直接作为负例，会压低真实等价 Action。
4. **Action 语义与输入格式存在上限。** NightOwl 的代码检索能力不等于对 shell 命令、多步工具调用和 Action JSON 的适配能力。当前将工具名和关键参数前置、截断至 512 token，仍需核查长 `exec` 中决定行为的片段是否被截掉。
5. **数据与评价仍依赖 Silver 标签。** 模糊 Intent、跨步骤等价行为和原正例本身有问题的样本，都会影响负例构造与指标解释。dev 用于选模；方案确定后才评估 test。

## 依次改进的训练计划

### 训练前：核查漏召回样本

从投影模型未命中 Top-20 的 `exec` 查询抽样，查看已知正例的完整命令、Top-20 候选、Intent 上下文及 Action 截断情况。分别标记“确实漏召回”“可能等价但未标注”“Intent 不足以判定”“原正例有误”，避免把标签问题当成训练问题。对比原 BGE 是否能召回相同正例，确认 NightOwl 丢失了哪些信息。

### 1. 冻结双塔，加入受控的训练池负例

从已有投影 checkpoint 继续训练，只更新线性投影；BGE Intent 塔和 NightOwl Action 塔都冻结。每轮用当前检索分数从训练集 Action 池选择负例：同工具，且 Intent 和正例同时明确写出目标路径、URL 或脚本，而候选写出与之不同的目标。先屏蔽同 Intent 的全部已知正例与显式负例，再从符合约束的候选中选高分项和随机项。原先审核过的显式负例照常使用；未判定的高相似候选不进入 loss。第一版脚本不会自动使用 dev Action 当训练负例，也不会把模糊 Intent 的所有未标注 Action 当负例。

离线校验显示 2,711 个训练组中有 655 组能找到这样的候选，其中 `exec` 为 81 组；这仍不足以覆盖全部执行场景，且当前采样不能称为对全库任意未标注候选的监督。即使有明确目标冲突，也需要抽查命令是否具有等价的副作用；无法保证零误标。每轮应抽查选出的高分负例，继续记录其来源。以同一 dev qrels 的全体和 `exec` Top-20 覆盖、Top-1、MRR@20 判断收益，而非只看训练 loss。

### 2. 在相同负例策略下微调 BGE Intent 塔

从第 1 阶段最佳 checkpoint 初始化，继续训练投影和 BGE Intent 塔，NightOwl Action 塔仍冻结。Intent 编码器使用低于投影层的学习率；训练池采样方式不变，因此能单独看解冻 Intent 塔的增益。若 `exec` 提升但全体明显倒退，应优先检查是否破坏了原 BGE 语义空间，不直接将该 checkpoint 用于下一阶段。

### 3. 再解冻 NightOwl Action 塔上层

从第 2 阶段最佳 checkpoint 初始化，继续训练投影、BGE Intent 塔和 NightOwl 的最后两层。Action 候选每步重新编码以传递梯度；每轮重新编码训练 Action 用于负例挖掘，每轮评估前重新编码 dev Action。编码器学习率均低于投影层。若上层解冻有效，再用独立实验考虑全量解冻；训练样本量只有 2,711 组，全量解冻的过拟合与显存风险更高。每个阶段分别与原 BGE 和上一阶段比较，不把阶段 3 的参数量增加视作必然收益。

### 运行顺序

以下命令从 Project 根目录运行，三个输出目录必须事先不存在。每步检查 `training_report.json` 的全体和 `exec` 指标，并审核采样难负例，再继续下一步。第 3 阶段的显存开销明显高于前两步，使用 batch 2、累积 16；如仍超显存，先降低 `--action-length` 或负例数并记录配置差异。

```bash
python -u scripts/train_nightowl_progressive.py --stage 1 --init-checkpoint outputs/nightowl_projection_v1/best --output outputs/nightowl_progressive_stage1 --device cuda --dry-run
python -u scripts/train_nightowl_progressive.py --stage 1 --init-checkpoint outputs/nightowl_projection_v1/best --output outputs/nightowl_progressive_stage1 --device cuda
python -u scripts/train_nightowl_progressive.py --stage 2 --init-checkpoint outputs/nightowl_progressive_stage1/best --output outputs/nightowl_progressive_stage2 --device cuda
python -u scripts/train_nightowl_progressive.py --stage 3 --init-checkpoint outputs/nightowl_progressive_stage2/best --output outputs/nightowl_progressive_stage3 --device cuda --action-unfreeze-layers 2 --batch-size 2 --accumulation 16
```

阶段 3 的 `--action-unfreeze-layers -1` 是可选的全量 NightOwl 微调对照，应使用新输出目录。所有阶段训练只使用 train Action 池，dev 仅选最佳轮，test 留给最终固定方案。若目标只是提高整体系统效果，也可另行评估“原 BGE 召回 + NightOwl 独立候选/特征 + 重排”的候选融合。

## 复现入口

- BGE 双塔：`scripts/train_bge_clawgym.py`；已选模型：`outputs/bge_clawgym_silver_v2/best/`。
- BGE 重排：`scripts/train_reranker_clawgym.py`；固定候选及训练说明：`docs/2026-10-09-bge-reranker-training.md`。
- NightOwl 直接替换：`scripts/train_nightowl_action_tower.py`；报告：`outputs/nightowl_action_tower_v1/training_report.json`。
- NightOwl 线性投影：`scripts/train_nightowl_projection.py`；报告：`outputs/nightowl_projection_v1/training_report.json`；最佳投影：`outputs/nightowl_projection_v1/best/projection.pt`。
- 三阶段继续训练：`scripts/train_nightowl_progressive.py`；每阶段分别保存 `best/projection.pt`、最佳 dev 排名和训练报告，第二阶段起保存 `best/query_model/`，第三阶段保存 `best/action_model/`。

当前投影实验命令：

```bash
python -u scripts/train_nightowl_projection.py --output outputs/nightowl_projection_v1 --device cuda
```

上面的命令复现的是旧投影实验；三阶段改进应使用“运行顺序”中的新命令。所有输出目录必须事先不存在。

## BGE-M3 双塔重新训练（2026-10-12）

新增独立实验 `scripts/train_bge_m3_dual_tower.py`：Intent 塔与 Action 塔都从本地 `model_hub/bge-m3` 初始化，分别全量微调；Action 塔之后加一个 1024×1024、单位矩阵初始化的可训练线性层。与早期共享权重的 BGE 双塔不同，这里保存两套独立权重。两塔都用 CLS 向量和 L2 归一化，使用已有 Silver 正例、显式负例及批内候选进行对比学习，并屏蔽同 Intent 的其他已知正例。此版本仅用 BGE-M3 的 dense 表征，不训练其 sparse/ColBERT 分支，也未加入上述三阶段脚本的受控训练池负例。

在 GPU 环境从 Project 根目录运行：

```bash
python -u scripts/train_bge_m3_dual_tower.py --output outputs/bge_m3_dual_tower_v1 --dry-run
python -u scripts/train_bge_m3_dual_tower.py --output outputs/bge_m3_dual_tower_v1 --device cuda --gradient-checkpointing
```

默认 batch 2、累积 16、3 轮；H200 使用 BF16 自动混合精度。最佳轮按 dev `exec` 已知正例 MRR@20 选择，报告同时列出全体指标及同口径原 BGE 基线。最佳 checkpoint 下的 `query_model/`、`action_model/` 和 `projection.pt` 必须配套推理，所有 Action 需要重新编码和建索引。训练环境需要 PyTorch 与 Transformers；当前工作环境只有离线数据校验，尚无该实验 GPU 结果。
