# 2026-10-13 GLM-5.2 清晰 Intent TA dev/test 复审

## 来源与审核

从冻结的 ClawGym 清晰 Intent dev/test 来源各抽取 200 个窗口。抽样仅参考输入工具类型及窗口内 pair 数，不使用模型预测或真实标签；train、内部验证、dev、test 的轨迹互不交叉。使用 `../.env` 中的 `GLM5_2_URL` 与 `GLM5_2_API` 调用 `glm-5.2`，审核日志保存在 `data/clawgym/ta_glm52_clear_eval_20261013/{dev,test}/model_reviews.jsonl`。含不确定说明的窗口另外逐 pair 复审，日志为同目录的 `pair_reviews.jsonl`。两组各有 6 个窗口因 HTTP 504 最终未审成，已隔离。

最终只接收可评估且 due-now 的 Intent、两轮一致且证据充分的疑难 pair；冲突、剩余疑点、模糊正例以及与 train 或另一评测 split 重复的文本 pair 均进入 `review_queue.jsonl`。审核日志及训练数据没有覆盖。

完整弱标签集：`data/clawgym/ta_glm52_clear_eval_labels_v2_20261013/`。完整窗口的标签由程序根据 due-now Intent 的 pair 覆盖聚合；原始模型建议保存在 `model_suggested_label`。

| Split | pair | 正 / 负 | 正例率 | 完整窗口 | 窗口标签 |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 304 | 162 / 142 | 53.3% | 135 | faithful 132, not_executed 3 |
| test | 389 | 178 / 211 | 45.8% | 129 | faithful 119, not_executed 10 |

`pairs.jsonl` 可包含来自未完整通过的窗口的可靠 pair；`window_labels.jsonl` 仅包含所有 pair 均通过的完整窗口。步骤级标签没有 `partial`，因此这批数据**不适合可靠的三分类 F1 结论**。所有标签均为 GLM 弱标签，不是人工 Gold；评测报告须明确这一来源。

## 与 train 的分布

参考训练集为 `data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013/train.jsonl`：7,243 对，正例率 43.4%，`exec` 占 27.3%。完整审核集保持自然分布，适合作为主评测集。另提供固定种子的 pair 级分层对照视图：`data/clawgym/ta_glm52_stratified_pair_eval_v4_20261013/`。它按工具组及 GLM 标签选样，不改变原始审核结果。

| 数据 | pair | 正 / 负 | 正例率 | exec 占比 | 工具分布 L1 | 工具×标签 L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7,243 | 3,146 / 4,097 | 43.4% | 27.3% | 0 | 0 |
| 分层 dev | 200 | 91 / 109 | 45.5% | 27.5% | 0.129 | 0.618 |
| 分层 test | 250 | 109 / 141 | 43.6% | 27.2% | 0.106 | 0.405 |

L1 为各类别比例绝对差之和，越小越接近。虽然正例率与工具占比接近 train，联合分布仍有明显偏差：分层 dev 的 `exec` 负例只有 4 对，test 只有 15 对，而 train 的 `exec` 负例有 1,046 对。已审核完整池分别也仅有 4、15 对，重采样无法补足。train 还含 SWE-Agent 数据，而本批 dev/test 全部是 ClawGym，来源分布也不相同。分层集改变了评测流行率，仅供对照；不能代替完整审核集，也不能推断三分类效果。

## 复现与后续

```bash
cd /inspire/hdd/project/control-technology/public/workspace/wx/Project
python scripts/build_ta_glm_eval_sample.py --output data/clawgym/ta_glm52_clear_eval_20261013
python scripts/review_ta_coverage_glm.py --windows data/clawgym/ta_glm52_clear_eval_20261013/dev/windows.jsonl --output data/clawgym/ta_glm52_clear_eval_20261013/dev/model_reviews.jsonl --retry-failed
python scripts/review_ta_coverage_glm.py --windows data/clawgym/ta_glm52_clear_eval_20261013/test/windows.jsonl --output data/clawgym/ta_glm52_clear_eval_20261013/test/model_reviews.jsonl --retry-failed
for split in dev test; do
  root=data/clawgym/ta_glm52_clear_eval_20261013/$split
  python scripts/build_ta_glm_pair_queue.py --windows "$root/windows.jsonl" --reviews "$root/model_reviews.jsonl" --output "$root/pair_review_queue.jsonl"
  python scripts/rereview_ta_uncertain_pairs_gqq.py --provider glm --windows "$root/windows.jsonl" --resolution "$root/pair_review_queue.jsonl" --review "$root/model_reviews.jsonl" --output "$root/pair_reviews.jsonl" --retry-failed
done
python scripts/finalize_ta_glm_eval.py --root data/clawgym/ta_glm52_clear_eval_20261013 --train data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013/train.jsonl --pair-reviews --output data/clawgym/ta_glm52_clear_eval_labels_v2_20261013
python scripts/match_ta_glm_eval_pairs.py --train data/clawgym/ta_pair_weak_train_10k_pool_targeted_v3_20261013/train.jsonl --eval-root data/clawgym/ta_glm52_clear_eval_labels_v2_20261013 --dev-count 200 --test-count 250 --output data/clawgym/ta_glm52_stratified_pair_eval_v4_20261013
```

上述命令是步骤示例；已有输出目录需改为新的版本路径。各 `manifest.json` 记录源文件 SHA-256 和筛选统计。

下一批应从尚未使用的冻结 dev/test 轨迹定向补 `exec` 错路径、错参数、同命令错目标等难负例，再由 GLM 审核并抽样人工复核。正式报告中的关键失败 case 与最终 Gold 仍须人工确认。
