# 2026-10-13 TA 一万候选对扩容

## 现有产物

`data/clawgym/ta_candidate_pool_10k_20261013/` 含 **10,000 个待审核 Intent–Action pair**、4,613 个窗口、651 条轨迹。其中 ClawGym 冻结 train 轨迹贡献 9,085 对，SWE-Agent 贡献 915 对。旧 600 窗口的修订版本优先保留；其余 ClawGym 窗口从同一冻结 train 来源确定性抽取。按规范化 Intent+Action 文本去重后有 9,932 种组合；68 条重复候选不算新增独立训练样本。

SWE-Agent 数据来自本地 `SWE-agent-trajectories/test_data/thought_intent/` 中已完成 Intent 提取的 25 条 SWE 轨迹，而不是将原始 80,036 条轨迹直接转标签。按 `instance_id` 的稳定哈希冻结切分：20 个 issue 进入 train，4 个留 dev，1 个留 test；仅 train 的 915 个 pair 进入这批候选。`target=true/false` 只表示最终 issue 是否解决，**不用于 TA pair 标签**。原始 Action 命令保留在 `raw_call.args.command`，Intent 与 Action 仍需逐对审核。

这 **10,000 对是候选数，不是训练标签数**。`model_reviews_new.jsonl` 是 GQQ 追加审核日志；失败记录可用 `--retry-failed` 重试，审核分歧、模糊 Intent、联合覆盖及不确定项不得强行贴负例。旧版 600 窗口数据含 845 个 GQQ 弱监督可用 pair（782 train、63 内部验证）。

## 2026-10-13 审核结果

继续使用 `deepseek-v4-flash-gqq`。新增 4,013 个窗口完成首轮审核，并对 287 个失败窗口执行一次定向重试；按每个窗口最后一条记录统计，仍有 103 个失败，1,486 个有效且无 `uncertainties`。与已有审核合并并应用隔离、去重规则后，生成 `data/clawgym/ta_pair_weak_train_10k_pool_20261013/`：

| 切分 | pair | 正例 | 负例 | ClawGym | SWE-Agent |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,780 | 1,829 | 951 | 2,453 | 327 |
| 内部验证 | 261 | 171 | 90 | 220 | 41 |

主要损耗来自 2,754 个不确定窗口及 110 个缺少有效审核的窗口（含上述失败项）；另去掉 50 个重复文本 pair。内部验证按轨迹留出，但仍是 GQQ 弱监督，不能替代独立人工 Gold。要达到一万条**可训练** pair，需要新增并审核更多独立轨迹，尤其是完成更多 SWE issue 的 Intent 提取；不能放宽不确定项过滤以凑数。

## 不确定项的保守修订

原始 GQQ 日志保持不变。`scripts/resolve_ta_uncertainties.py` 将可审计的 pair 决策写入 `uncertainty_resolutions_v5.jsonl`：明确写着“No uncertainty”的字段污染直接清理；明确属于准备步骤且所有相关 pair 已判负的说明清理；明确说不影响覆盖结论且没有残余风险措辞的说明清理。其他说明只有能够唯一定位到 Intent 时才隔离该 Intent 的 pair，无法定位时继续隔离整个窗口。不能把“可能是准备步骤”这一宽泛类别整体删除；实际只有少量说明满足自动清理条件。

合并三份审核后的 2,762 个不确定窗口共含 3,211 条说明：81 条字段污染（新增候选池此前统计为约 63 条）、37 条结论不变的准备步骤说明、23 条其他明确不影响结论且无残余风险的说明、999 条可定位到 Intent、2,071 条仍无法安全定位。最后一类及受影响的 pair 均未入训。此规则仅用于探索性弱监督，新增 pair 应再抽样人工复核。

修订版数据位于 `data/clawgym/ta_pair_weak_train_10k_pool_resolved_v5_20261013/`：**3,809 train + 461 内部验证**，train 含 2,104 正例、1,705 负例。相对原版增加 1,029 条 train pair。决策文件共清除 1,245 个受不确定项影响窗口中的 pair，最终训练数还受时间属性、不可评测、隔离清单、去重及按轨迹切分影响。

## GQQ 定向复审

对 v5 决策文件中仍含被隔离 pair 的 **2,584 个窗口**调用同一 `deepseek-v4-flash-gqq` 复审。复审逐 pair 输出 `covered` / `not_covered` / `uncertain`、证据充分标志及残余疑点。只释放复审证据充分、无残余疑点、与原标签一致且 Intent 可评估的 pair。若原疑点涉及目标缺失、复数范围或从 Action 推断目标，正例继续隔离；明确的负例可保留。原始审核与复审日志均未改写。

合并结果：**3,758 对释放**；259 对标签冲突、438 对仍不确定、350 对正例目标范围有风险、660 对 Intent 不符合当前训练条件，另有 5 个窗口复审输出仍未通过格式或 ID 校验。新决策文件为 `data/clawgym/ta_candidate_pool_10k_20261013/uncertainty_resolutions_targeted_v2.jsonl`，每个 pair 的状态与两轮理由记录在 `targeted_audit`。

最终探索性弱监督数据位于 `data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013/`：**7,243 train + 761 内部验证**；train 为 3,146 正例、4,097 负例，其中 3,463 对有两轮 GQQ 一致判断。内部验证仍是模型弱标签，应使用独立人工 Gold 报告真实效果。该扩充量大，尤其要抽查正例的目标路径、复数 Intent 和工具上下文推断。

```bash
python scripts/train_ta_pair_cross_encoder.py --data data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013 --output outputs/ta_pair_cross_encoder_10k_targeted_v3 --device cuda
```

```bash
python scripts/resolve_ta_uncertainties.py --windows data/clawgym/ta_candidate_pool_10k_20261013/windows.jsonl --review data/clawgym/ta_coverage_pilot_20261013/model_reviews_current.jsonl --review data/clawgym/ta_coverage_train_v2_20261013/model_reviews_new.jsonl --review data/clawgym/ta_candidate_pool_10k_20261013/model_reviews_new.jsonl --output data/clawgym/ta_candidate_pool_10k_20261013/uncertainty_resolutions_v5.jsonl
python scripts/prepare_ta_pair_train.py --windows data/clawgym/ta_candidate_pool_10k_20261013/windows.jsonl --review data/clawgym/ta_coverage_pilot_20261013/model_reviews_current.jsonl --review data/clawgym/ta_coverage_train_v2_20261013/model_reviews_new.jsonl --review data/clawgym/ta_candidate_pool_10k_20261013/model_reviews_new.jsonl --resolutions data/clawgym/ta_candidate_pool_10k_20261013/uncertainty_resolutions_v5.jsonl --output data/clawgym/ta_pair_weak_train_10k_pool_resolved_v5_20261013
```

## 审核与训练集

新增窗口审核命令：

```bash
python scripts/review_ta_coverage_gqq.py --windows data/clawgym/ta_candidate_pool_10k_20261013/review_needed.jsonl --output data/clawgym/ta_candidate_pool_10k_20261013/model_reviews_new.jsonl --workers 12 --timeout 120
```

审核完成后，按旧试点、600 窗口版本、新候选池的优先级合并有效记录。训练集构建会过滤旧隔离清单、模型不确定、不可评测、时间属性冲突及文本重复/冲突标签；随后按轨迹留出内部验证：

```bash
python scripts/prepare_ta_pair_train.py --windows data/clawgym/ta_candidate_pool_10k_20261013/windows.jsonl --review data/clawgym/ta_coverage_pilot_20261013/model_reviews_current.jsonl --review data/clawgym/ta_coverage_train_v2_20261013/model_reviews_new.jsonl --review data/clawgym/ta_candidate_pool_10k_20261013/model_reviews_new.jsonl --output data/clawgym/ta_pair_weak_train_10k_pool_20261013
```

训练时显式指定新数据路径：

```bash
python scripts/train_ta_pair_cross_encoder.py --data data/clawgym/ta_pair_weak_train_10k_pool_resolved_v5_20261013 --output outputs/ta_pair_cross_encoder_10k_pool_resolved_v5 --device cuda
```

`manifest.json` 的 `train_pairs` 才是可用于探索性训练的数量。此处内部验证仍是 train 来源的 GQQ 弱监督；独立 dev/test 需要人工 Gold。若目标是**一万条经质量检查的训练 pair**，必须继续扩充并审核来源，尤其是更多独立 SWE issue，而不能把本候选池直接称为一万训练样本。
