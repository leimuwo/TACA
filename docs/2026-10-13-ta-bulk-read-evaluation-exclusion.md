# 批量文件阅读 Intent 评测剔除

按用户要求，清晰 dev/test 主评测额外剔除未拆分的批量文件内容目标，例如 `Read all input files`、`Read the HTML files in input/pages/`、`Review the skill package files`。该规则只使用 Intent 文本，不参考模型预测、工具返回或 Gold。实现于 `scripts/ta_evaluation_scope.py`，策略版本 `exclude_bulk_file_read_v1`。

当前目标中任一 Intent 命中规则时剔除整个窗口，不删掉目标后对剩余目标判 faithful。单个明确文件、目录列举/结构探索和编写读取文件的脚本保留。规则是保守文本筛选，不能声称识别了全部非原子 Intent；未命中但仍需联合覆盖的案例继续由人工可评测性审核处理。未来计划中的批量阅读不触发当前窗口剔除。

已在原清晰评测结果上复用相同 pair logits 和阈值，不重跑 GPU，不修改旧输出：

| split | 原窗口 | 剔除窗口 | 新窗口 | 输出 |
| --- | ---: | ---: | ---: | --- |
| dev | 573 | 5 | 568 | `outputs/ta_clear_dev_cross_encoder_atomic_v2/` |
| test | 516 | 6 | 510 | `outputs/ta_clear_test_cross_encoder_atomic_v2/` |

各目录包含筛选后窗口、预测、Gold 模板、报告及 `bulk_read_exclusions.jsonl`。隔离清单记录该 split 中所有命中的来源 Intent（含早已因模糊筛查排除的窗口），报告计数只统计原清晰集新增剔除数。尚无人类 Gold，`metrics_ready=false` 仍是正常状态，不能据预测分布宣称性能改善。

后续 `evaluate_ta_clear_dev.py` / `evaluate_ta_clear_test.py` 默认应用规则。复用既有结果时使用 `--reuse-predictions <旧 predictions.jsonl>`；要求窗口 pair ID 和阈值匹配。该模式直接继承旧预测，`--checkpoint` 仅记录声明，须由调用者保证旧预测来自所声明 checkpoint。

冻结的独立 100/100 标注队列及 30 条 HTML 保留原始样本，不修改来源哈希。人工遇到此类目标应标不可评测，退出严格三分类主指标。四项清晰评测测试通过，包括批量阅读排除、目录列举保留和整步排除。
