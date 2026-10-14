# BGE v2 低排名样例抽查（2026-09-30）

## 范围与方法

从 `bge_dataset_split_20260930_v2/test` 的 1,074 个 Intent 中，按固定随机种子 `20260930` 抽取 36 条已知正例排名大于 10 的样例：`exec` 排名 11–100 和大于 100 各 10 条；`write` 的已知正例序列化长度大于 2,000 字符，排名 11–100 和大于 100 各 8 条。逐条比较 Intent、全部已知正例和前 3 个检索 Action。完整文本见 `data/clawgym/bge_v2_error_audit_20260930/sample_evidence.jsonl`，逐条判断见同目录的 `reviewed_cases.jsonl`。抽样故意偏向失败案例，以下数量不是总体错误率。

## 结果

| 主要归因 | `exec` 20 条 | 长 `write` 16 条 | 合计 |
| --- | ---: | ---: | ---: |
| Intent 缺少足以区分 Action 的上下文 | 10 | 2 | 12 |
| 模型选错工具或操作 | 6 | 5 | 11 |
| 模型选错参数、路径或内容 | 1 | 9 | 10 |
| 疑似 Silver 漏标，需结合完整任务确认 | 3 | 0 | 3 |

### 模型选错工具或操作

- S21、S29：Intent 要**创建** Python 脚本，第一名却是 `exec` **运行**同名脚本。路径词匹配到了，但动作阶段错了。
- S08：Intent 要检查已下载的本地 PDF，第一名却抓取 WADA 网页。S17 要检查超时后本地文件是否存在，第一名却重新下载该文件。
- S28：Intent 要创建完整 manifest，第一名是对同路径文件做 `edit`；路径相同不代表操作等价。

### 模型选错参数、路径或内容

- S22、S24：Intent 指定 `output/README.md`，第一名写入其他文件。
- S27：Intent 指定 `data/search_log.jsonl` 及 JSONL 格式，第一名写的是 `output/search_log.md`。
- S35：已知正例是 `output/REVIEW.md`，第一名写 `output/review.md` 且内容不同。在大小写敏感的文件系统中是不同路径。
- S25、S26：第一名分别写关联的 `workflow.yaml`、`run.sh`，可作为支持步骤，却没有创建 Intent 指定的目标文件。不能把这些部分履行或中间步骤补成正例。

### Intent 缺上下文

- S02、S04、S07：Intent 只说“remaining IPs”，没有 IP 列表；每个 Intent 对应 5 个抓取单一 IP 的已知正例 Action。单条 Action 也未独立完成“抓取剩余 IP”的复数目标。
- S13、S20：只说“all the files”或“output files”，没有文件清单；第一名读取其他任务的单个文件，是否相关无法从 Intent 判断。
- S18、S32、S36：分别只说“output directories”“the script”“the report”，没有足够目标信息。在跨轨迹候选池中，不可能只凭这些文字稳定定位某一条日志 Action。

### 疑似 Silver 漏标

- S01：已知正例是查询 FDA 安全通信页的 Wayback CDX；第 2 名也在 Wayback 查同主题的其他 URL，按当前 Intent 字面语义可能有效。
- S05：Intent 只要求用 Python 直接运行 serper 搜索脚本；第 1 名确实运行同一脚本，但搜索主题不同。原始任务可能要求 WADA，Intent 却没写出。
- S09：Intent 只说搜索另一条 Nelson Mandela 名言；第 2 名也做了 Mandela 名言搜索，但无法从 Intent 确定“另一条”的参照。

这 3 条是**待人工确认的候选**，不是已证明的漏标。尤其 S05、S09 也体现 Intent 缺上下文：若补充原始任务主题或前一条名言，它们可能变成明确负例。其他未标注 Action 不应自动视为负例。

## 后续处理顺序

1. 先补充模糊 Intent 的目标路径、文件列表、查询主题及必要的前序状态；复数目标要决定是否拆成多个单独 Intent。修订后再重新评测，避免把不可判定查询算作模型能力问题。
2. 对明确的“创建 vs 运行”“检查 vs 下载”“write vs edit”及同工具错路径案例补难负例，并保留中间/支持步骤排除规则。
3. 人工核对 S01、S05、S09 的原始任务上下文后，再决定是否补 qrels。未经核对不改 test 标注。

本审查只分析 v2 的 Silver 排名和对应 Action；未改变训练、dev、test 文件，也未重训模型。
