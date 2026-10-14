# BGE-M3 双塔 exec 监督增强

日期：2026-10-12

## 数据与边界

原始 v2 train 有 2,711 行，其中 176 行的正例 Action 是 `exec`。已有的 `bge_dataset_split_20260930_v2_exec_augmented_v2` 先加入经验证的显式负例，得到 2,716 行、181 行 `exec` 正例。本轮以该增强集为输入，只在 **train** 正例上构造 72 行反事实负例，生成 `data/clawgym/bge_dataset_split_20260930_v2_exec_supervision_20261012/`：70 行错脚本、2 行错文件，涉及 72 个查询。新增行复用已有正例，并不是 72 个新的独立正例。最终 train 共 2,788 行、253 行 `exec` 正例行。

`scripts/build_bge_m3_exec_supervision.py` 只在 Intent 字面明确指定目标且目标存在于正例命令时替换它；替换名从 train 正例中取得，保持工具、命令其余部分和其他参数不变。按同一 Intent 的全部已知正例排除碰撞，并经 `load_train` 检查重复三元组和正负例冲突。`new_exec_counterfactuals.jsonl` 保存每条的原目标与替换目标，便于复核。逐条检查目标摘要，并用程序核对 72 条均为 `exec`、目标出现在 Intent、替换目标未出现在 Intent、其他 Action 参数不变；70 条脚本命令只替换被执行的脚本名，2 条文件命令只替换文件路径。这是**字面规则和结构核验**，不能代替对每个脚本运行效果的人工审核。

本轮没有满足严格条件的错 URL 或数值参数样本，不能将这两类计入新增量。错文件样本也只有 2 条。要继续补充这些类别，应从新的 train 标注中挑出明确写有 URL、路径或参数值的 Intent，人工确认反事实 Action 不会以其他步骤满足同一目标，再保存审核结论及来源；不能把未标注的相似 Action 自动当负例。此前 `docs/2026-10-12-bge-m3-exec-error-audit.md` 中的 dev 错例仅用于归纳模式，**没有进入本轮 train**。

dev/test 的 queries、actions、qrels 文件与输入增强集的 SHA-256 完全相同。现有 dev qrels 仍是 Silver 已知正例口径；测试集不参与训练和选模。

## 训练与比较

`scripts/train_bge_m3_dual_tower.py` 新增 `--exec-repeat`，默认 1 保持旧采样方式。设为 4 时每轮将每个 `exec` 正例行采样 4 次，再统一打乱；总计 3,547 个采样行，`exec` 1,012 行，占 28.5%。同 Intent 的其他已知正例仍从批内负例中屏蔽。重复采样改变曝光比例，不增加独立样本数，可能放大 Silver 标签噪声。

在 Project 根目录、具有 PyTorch/Transformers 和 CUDA 的环境运行；输出目录必须尚不存在：

```bash
python -u scripts/train_bge_m3_dual_tower.py --dataset data/clawgym/bge_dataset_split_20260930_v2_exec_supervision_20261012 --output outputs/bge_m3_dual_exec_supervision_20261012 --exec-repeat 4 --device cuda --gradient-checkpointing
```

可先在任意环境用同样参数加 `--dry-run` 核验行数和评估输入。脚本从本地 BGE-M3 基座初始化两塔及单位矩阵投影，不从旧 checkpoint 继续；日志每 25 个 micro-batch 输出进度，每轮输出 dev 全体及 `exec` 指标，并按 dev `exec` 已知正例 MRR@20 保存最佳轮。

比较时用同一 dev qrels、Action 库、Top-20、随机种子和训练超参数，同时记录全体与 `exec` 的 Top-20 覆盖、Top-1 和 MRR@20。为了拆分负例与采样比例的作用，分别跑“原增强集 + `--exec-repeat 1`”、“新集 + `--exec-repeat 1`”和“新集 + `--exec-repeat 4`”，每次使用新输出目录。当前环境只完成离线校验，尚无新模型训练指标，不能断言准确率已提升。
