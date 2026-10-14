# BGE-M3 双塔 `exec` 错误抽查

日期：2026-10-12

## 口径

模型为 `outputs/bge_m3_dual_tower_v1/best/`，使用 1,078 条 dev 查询、1,944 个 Action 和 `bge_dev_qrels_exact_text_20261009/qrels_review_closed.jsonl`。`exec` 子集定义为查询至少有一个已知 `exec` 正例，共 437 条；其中 103 条 Top-1 命中已知正例，177 条已知正例在 Top-20 但不在首位，157 条已知正例不在 Top-20。

用固定种子 `20261012` 分层抽取 20 条：候选内错排按已知正例排名 2–5、6–10、11–20 各 4 条；Top-20 漏召回按首位为 `exec` 或其他工具各 4 条。此抽样刻意平衡错误类型，**不能把样本比例外推到全部 334 条错误**。完整 Intent、Action、来源轨迹与分数保存在 `data/clawgym/bge_m3_exec_error_audit_20261012/review_queue.jsonl`；可用 `scripts/audit_bge_m3_exec_errors.py` 复现。

以下为基于 Intent 字面、Action 参数和来源轨迹的初审：

- **明确错误**：首位 Action 缺少必要操作或目标明显不符。
- **可能等价**：首位从字面上也可能满足 Intent，需确认执行效果后补正例。
- **原正例不完整**：当前标注的正例缺少 Intent 明确要求，而首位更符合要求；需复核原标签。
- **待核上下文**：Intent 未点明对象，或需原任务/执行结果才能判定。

## 抽查结果

| 分组 | 明确错误 | 可能等价 | 原正例不完整 | 待核上下文 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-20 漏召回 | 5 | 1 | 0 | 2 | 8 |
| 候选内错排 | 5 | 4 | 2 | 1 | 12 |
| 合计 | 10 | 5 | 2 | 3 | 20 |

### Top-20 漏召回

| query_id | 初审 | 依据 |
| --- | --- | --- |
| `1ec180410fc085c862520dea` | 明确错误 | 要用完整 Python、输入、输出路径运行重构脚本；首位只运行 `python3 run.py`，同轨迹进入 Top-20 的也仅是检查 Python 版本。 |
| `2a12175bd4717b7faed16fac` | 待核上下文 | 要数 essay 字数；首位对 `drafts/pitch_email_rewritten.md` 数词，正例对 `output/humanized/essay.txt` 数词。需核原任务指的 essay；Top-20 同轨迹候选只在后一个文件上做引用/格式检查。 |
| `80b375b9971321a1067ea08d` | 明确错误 | 要读取输入文件；首位 `ls -la input/` 只列目录，正例包含 `cat input/state.json`。 |
| `e7f405b78299166dc5ce2623` | 明确错误 | 要设置复数的输出目录；首位只创建 `outputs`，正例创建 `output/brief` 等多个子目录。 |
| `120fcf4e6a8a9ae22b8b7f17` | 明确错误 | 要验证完整输出树；首位 `read` 单个 `compliance_report.csv`，正例 `find output -type f` 枚举整棵树。 |
| `4ad93f2bb030257037207a79` | 可能等价 | 要探索项目现有文件与结构；首位读取 `docs/architecture_overview.txt`，正例用 `find` 列出 `.py` 和 `.csv`。两者可能提供结构信息，需看文档内容与任务要求。 |
| `b388c869fa55c015d54cb43f` | 明确错误 | 要检查有哪些输入文件；首位只读一份 `survey_responses.csv`，正例 `ls input/` 才列出现有文件。 |
| `c8060c2f8fe1adad1581b6e6` | 待核上下文 | Intent 仅说“验证 CSV 文件”，首位读取一份日程 CSV，正例用 Python 检查 `output/summary.csv`。需确认所指文件及“验证”的具体标准。 |

### 候选内错排

| query_id | 正例排名 | 初审 | 依据 |
| --- | ---: | --- | --- |
| `04793ee049c240b95e982a55` | 13 | 原正例不完整 | Intent 明确提到 `config/`、`drafts/`；首位执行 `ls -la config/ drafts/`，正例只在工作区执行 `ls -la`，看不到这两个目录的内容。两者同轨迹，应复核是否补首位为正例。 |
| `1411e3ac6b576ff92431a0f4` | 17 | 明确错误 | 要确认 HTML 文件存在；首位对另一轨迹的单个 `remote_id_page.html` 用 `sed` 查看正文，正例列出本任务的 `data/web/`。 |
| `9bc938ef9749bc2ab1af158f` | 15 | 待核上下文 | 要检查 `yaml` 模块是否可用；首位运行 YAML 工作流验证脚本，正例直接 `import yaml`，失败时还安装依赖。验证脚本是否能证明模块可用，需看脚本和执行结果。 |
| `bb1b1405a4910af3aa24a7a7` | 17 | 明确错误 | 要创建 `out` 目录；首位是 `write` 长文本文件，正例为 `mkdir -p .../out`。 |
| `a5cd6a093327f9503d4379f5` | 3 | 原正例不完整 | Intent 要运行合规检查、写日志并追加退出码；首位包含追加 `EXIT_CODE` 的命令，正例只把输出送入 `tee`，没有追加退出码。首位 `$?` 位于管道之后，可能是 `tee` 的退出码；补标前需核执行结果。 |
| `db6d0c5fd7ca3506f393a4a8` | 5 | 明确错误 | 要搜索 Rebuilding Together San Francisco；首位搜索 HomeAid Orange County，正例批量搜索中包含指定机构。 |
| `f5b0cb8e3ba27cbcafcbadc9` | 4 | 明确错误 | Intent 明确指定正则 `.*\d+P\d*$`；首位靠有限的 `P1`–`P20` 后缀列表过滤，不能覆盖任意数字组合，正例使用指定正则。 |
| `fe5ee7be28b299891d46ddd9` | 4 | 可能等价 | 首位和正例都直接运行 `expense_analysis.py`；主要差别是绝对路径与在工作区 `cd` 后用相对路径及 `pty`。需核脚本对工作目录的依赖。 |
| `0ffbc007c59d9740db65a1af` | 9 | 可能等价 | Intent 只说直接测试 Serper 搜索脚本，未限定查询词；首位和正例都直接调用该脚本，但搜索主题不同。需核原任务是否隐含特定主题。 |
| `1532d77f225dface6e80363c` | 6 | 可能等价 | 首位与正例运行同一 `family_planner.py`，配置、输入与输出路径在同一工作区下等价，形式是绝对路径与相对路径。 |
| `632b94e12532bfc09cd60a90` | 8 | 可能等价 | 首位与正例都直接运行同一 `scripts/pipeline.py`，工作区相同，差别主要是绝对路径与相对路径。 |
| `8eb12478bb2594430985cf28` | 6 | 明确错误 | 要搜索 Habitat for Humanity of Broward；首位搜索 HomeAid Orange County，正例批量搜索中包含指定机构。 |

## 结论与处理

这 20 条中确有 10 条清晰的检索或排序错误，集中在错对象、错操作、漏关键参数。另有 5 条首位可能等价、2 条原正例不完整、3 条需上下文；因此当前 Silver 命中率会低估部分实际可用结果，但不能据此推算全体 334 条的修正后分数。

优先复核 `04793ee049c240b95e982a55` 与 `a5cd6a093327f9503d4379f5` 的原正例，再检查 5 条可能等价案例的执行结果；确认后按 Action ID 补 dev qrels，并保存新版本，不直接覆盖现用评测集。明确错误可用于归纳训练集的同工具错对象、错命令和错参数模式，但 **dev 样本本身不回流训练**。对模糊 Intent 应核原始用户任务与步骤上下文，无法判定的保持未审，不自动给未标注 Action 负例标签。
