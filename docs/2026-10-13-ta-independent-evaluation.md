# TA 独立 dev/test Gold 标注队列

日期：2026-10-13。

## 当前状态

已建立 `data/clawgym/ta_independent_evaluation_20261013/`，沿用冻结的 647/139/139 条 train/dev/test 轨迹及模板簇划分，拒绝轨迹或模板跨 split。dev、test 各抽 100 个窗口，各来自 100 条不同轨迹；另从两者各抽 15 条组成 30 条逐 Intent/pair 子集，共 142 个 pair。与 train 轨迹重叠为 0。

**这些是待人工标注的独立队列，尚不是完成的 Gold，尚无泛化指标。** 没有把 GQQ、旧 qrels、已有 train 人工标注或模型预测写作 Gold。现有 train review30 继续只用于错误分析和流程验证。

抽样先按 seed 与轨迹 ID 的哈希选择轨迹，再按窗口哈希取该轨迹一个窗口；不参考模型建议、预测标签或错误类型。每条轨迹机会相同，指标对应轨迹均匀抽样，不是所有步骤频率加权。每个窗口必须有抽取 Intent 和实际 Action；零抽取 Intent、零 Action 窗口不在本版评测范围。30 条子集也不按类别配额抽样，不能保证三分类均衡。

保留完整 Thought、之前的可观察上下文、抽取 Intent 与证据、所有当步调用及工具返回。对所有抽取 Intent（含上游 future）与所有当步 Action 生成笛卡尔积 pair，避免人工纠正 due-now 后缺 pair。来源 split、Intent 输入、所选原始轨迹和输出 JSONL 均记录 SHA-256。

## 标注入口

- `dev/review.html`：100 条整步盲审，用于阈值校准和开发规则。
- `test/review.html`：100 条整步盲审，仅用于协议固定后的最终测试。
- `pair30/review.html`：30 条逐 Intent/pair 盲审，dev/test 各 15 条。

页面无模型建议，三个队列使用独立浏览器暂存空间。填写标注者、整步标签、依据；逐项队列还需为每个 Intent 填 due-now、可评测性，为每个 pair 填 yes/no/uncertain。点击“导出标注 JSON”导出 JSONL。模板 `annotation_template.jsonl` 可供离线标注；不要改冻结的源文件。

标注只评调用语义，不评执行成功。依据 Thought 与此前上下文确定当前目标，不从实际 Action 或后续结果反推 Intent。未来计划不参与整步覆盖；任何当前目标不可评测、缺少抽取目标、不可拆分联合覆盖或无法裁定的部分执行问题，都应标 `unassessable` 并记录争议。不得删除模糊目标后判 faithful。faithful 为全部原子目标完整覆盖，partial 为部分原子目标完整覆盖，not_executed 为零完整覆盖。无 due-now 目标亦不可评测。

30 条逐项聚合与对应整步标签不一致时，须复核并保留裁定依据，不能自动用模型建议补齐。建议由第二位标注者独立复核这 30 条，报告标签一致性并裁定争议。已完成的单人标签不自动证明标注可靠性。

test 的逐项审核用于最终标注质控，不向模型或规则开发反馈具体案例。若已经用 test 案例修改提示词、规则或阈值，应记录测试污染并另建未使用的最终测试集；不得继续称原 test 为最终泛化验证。

## 复现与冻结 Gold

在 Project 根目录运行；输出目录必须不存在：

```bash
python scripts/build_ta_independent_evaluation.py \
  --output data/clawgym/ta_independent_evaluation_20261013

python scripts/finalize_ta_independent_gold.py \
  --dataset data/clawgym/ta_independent_evaluation_20261013 \
  --dev-review <dev人工导出.jsonl> \
  --test-review <test人工导出.jsonl> \
  --pair-review <30条逐项人工导出.jsonl> \
  --output data/clawgym/ta_independent_gold_20261013
```

冻结器检查源文件哈希、来源 split、标注 ID 完整性、标注者与理由、逐项完整性、存疑 pair、覆盖聚合冲突，以及逐项 Gold 与整步 Gold 是否一致。未完成标注直接拒绝；人工身份字段是来源记录，程序无法验证填写者身份。输出 `dev_gold.jsonl`、`test_gold.jsonl`、`pair30_gold.jsonl` 和冻结 manifest。

先在 dev 选择阈值并固定模型/规则，再用独立预测文件分别评测 dev/test；使用现有 `evaluate_ta_coverage_pilot.py --gold-source human-overall`，30 条子集可用 `--gold-source pair-derived`。报告总数、不可评测数、可评测三分类指标及工具分组；不可评测预测不能作为正确预测，应另报拒判率。当前评测器会跳过不可评测预测，其输出仅是可判定样本指标，不能当无拒判的全量准确率。

## 验证

6 项新增检查与 10 项现有 TA 测试通过；三个盲审页面通过 Node JavaScript 语法检查。已检查各队列轨迹唯一、逐窗口 pair 矩阵完整。未运行模型推理，也未报告泛化指标。

## 30 条 GQQ 补跑结果

后续已按用户要求调用 `deepseek-v4-flash-gqq`，沿用 `ta_coverage_review_v4_20261013` 提示词，没有根据 test 案例调整提示词。首轮 27 条通过、3 条返回多余 JSON；仅重试失败条目后 30 条均通过 ID、布尔标签和覆盖聚合校验。追加运行日志共 33 行，当前去重结果每窗口一行。

结果单独保存在 `pair30/`：

- `gqq_reviews_v4.jsonl`：原始审核与失败重试记录。
- `gqq_reviews_current.jsonl`：30 条当前有效模型审核，含判断理由和不确定性。
- `gqq_intent_decisions.jsonl`：67 条 Intent 判断，含 due-now、可评测性及理由。
- `gqq_pair_decisions.jsonl`：142 条 pair 判断，含完整覆盖与否及理由，其中 34 条建议覆盖。
- `gqq_aggregated_labels.jsonl`：30 条程序聚合标签。
- `gqq_summary.json`：数量、分 split 分布、提示词与输入输出哈希。

聚合结果为 faithful 18、partial 3、not_executed 4、unassessable 5。dev 分别为 9/2/1/3；test 分别为 9/1/3/2。以上是模型输出分布，不是准确率，也不是人工 Gold。所有冻结数据和人工盲审页面的哈希保持一致，未将模型建议展示在盲审页面或填入人工模板。

本次原样沿用现有审核输入，模型仅看到 Thought、Intent、Action 与 pair，不传 `preceding_context`、用户任务或执行返回。因此 GQQ 可评测性判断采用既有较严格的文本口径，人工利用此前上下文审核时可能不同；这一限制须在后续对比中说明。test 结果不用于阈值或规则开发。

后续按用户要求，已将 30 条当前 GQQ 结果整合到 `pair30/review.html` 的默认折叠“GQQ 模型建议”区域，展示整步聚合、逐 Intent due-now/可评测判断、逐 pair 覆盖判断、理由和不确定性。模型区使用 I/A 序号对应原文；不自动填人工标签。原页面浏览器暂存 key 保持一致，已存标注仍可继续；dev/test 整步页面和冻结 JSONL 未修改。30 条页面渲染及 JavaScript 语法验证通过。

复现/续跑命令（已有文件会跳过成功条目）：

```bash
python scripts/review_ta_coverage_gqq.py \
  --windows data/clawgym/ta_independent_evaluation_20261013/pair30/windows.jsonl \
  --output data/clawgym/ta_independent_evaluation_20261013/pair30/gqq_reviews_v4.jsonl \
  --workers 6 --retry-failed
```
