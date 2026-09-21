# Thought–Intent–Action Retrieval 项目进展汇报

**汇报日期：** 2026-09-21  
**当前阶段：** Phase 1 provisional negative 与训练可行性实现
**当前分支：** `feature/phase1-dataset`  
**最新已提交基线：** `04a89f4 docs: design provisional retrieval pilot`

## 一、项目目标

本项目研究工具型 Agent 中可观察 Thought、原子执行 Intent 与实际工具 Action 的一致性。当前目标是训练一个共享参数的双编码向量模型，将自然语言 Intent 与短工具 Action 映射到同一向量空间，并从候选 Actions 中检索真正匹配的 Action。

第一阶段严格限定为：

```text
1 atomic Intent → 1 short Action
```

多 Intent、多 Action 集合履行及 Kimi collective Action 暂不进入第一阶段训练，以避免混合不同监督信号。

## 二、当前技术路线

整体流程如下：

```text
原始轨迹
  → 筛选可用 SWE 轨迹
  → 从 observable Thought 抽取原子 Intent
  → 标准化并过滤短 Action
  → 生成 Intent–Action 候选
  → 人工标注 direct_match / no_match / unfulfilled 等关系
  → 按 trajectory/template 隔离切分
  → 构造 hard negatives
  → 训练和评测 bi-encoder
```

当前已经完成 provisional pilot 的离线工程实现；在线负样本批量生成等待
`INF_API_KEY` 安全注入。所有结果均为 feasibility-only，不替代人工 Gold 数据。

## 三、已完成工作

### 3.1 独立研究仓库

已建立独立 Python 研究仓库，不依赖原 TACA 代码和历史。Git 仅管理代码、配置、Prompt、文档、manifest 和少量样本；大规模轨迹和生成结果保存在外部数据目录。

当前主要目录：

```text
Project/
├── configs/                 # 路径配置示例
├── data/
│   ├── manifests/           # 外部数据快照和校验信息
│   └── samples/             # 可进入 Git 的小型样本
├── docs/
│   ├── designs/             # 数据设计
│   ├── plans/               # 实施计划
│   └── superpowers/         # 仓库与负样本设计文档
├── prompts/                 # Thought → Intent Prompt
├── scripts/                 # 数据筛选、抽取和构建 CLI
├── src/thought_action_retrieval/
│   ├── data/                # 数据路径、manifest、SWE 筛选
│   ├── intent/              # Intent 自动抽取
│   └── matching/            # Phase 1 数据集构建
└── tests/                   # 单元及真实语料集成测试
```

远端仓库已配置为：

```text
git@github.com:leimuwo/TACA.git
```

当前 Phase 1 工作位于 `feature/phase1-dataset`，尚未合并到 `main`。

### 3.2 SWE 500 条轨迹筛选

已从完整 SWE 数据中确定性筛选 500 条轨迹，并保存 manifest 和三个小型样本。

| 指标 | 数量 |
|---|---:|
| 轨迹文件 | 500 |
| 唯一 `case_id` | 500 |
| 唯一原始任务 `instance_id` | 320 |
| 总 Thought–Action 步骤 | 16,197 |
| `swe-agent-llama-405b` | 50 |
| `swe-agent-llama-70b` | 350 |
| `swe-agent-llama-8b` | 100 |

`instance_id` 少于 500 是因为同一任务可由不同模型产生配对轨迹，不是文件重复。数据目录约 111 MB。

### 3.3 Thought → Intent 自动抽取管线

已经实现：

- 从 SWE 和 Kimi 轨迹提取 observable Thought；
- 当前 Action 不进入 Prompt，避免标签泄漏；
- 使用用户请求和前序上下文解析代词；
- 严格 JSON 输出和 schema 校验；
- `source_quote` 必须是 Thought 的原文子串；
- 逐步骤原子保存；
- 自动重试和断点续跑；
- 记录模型、Prompt hash、token usage 和原始响应。

现有已正式抽取的试验语料：

| 来源 | 轨迹 | 步骤 |
|---|---:|---:|
| SWE | 25 | 1,012 |
| Kimi | 25 | 154 |
| 合计 | 50 | 1,166 |

其中 1,110 步抽取成功，56 步为空 Thought 跳过，共得到 1,250 个 Intent。累计模型用量约 461 万 tokens。

### 3.4 Phase 1 Intent–Action 数据构建

已实现完整的 Phase 1 数据构建模块和 CLI：

- Action 解析、确定性序列化和长度过滤；
- 仅 SWE 进入一对一候选，Kimi 只参与审计；
- 只保留“一个 Intent、一个短 Action”的候选；
- 排除 `edit`、多行代码、heredoc、复合 shell、赋值碎片和长 payload；
- 生成稳定候选 ID；
- 输出人工审核 JSONL/CSV；
- 导入人工标注；
- 只有 `direct_match` 可进入训练正样本；
- trajectory 和任务模板分组切分；
- finalize 采用暂存与失败回滚，防止半成品发布；
- Phase 0 manifest、trajectory、step schema 采用 fail-closed 校验。

当前正式 Phase 1 产物来自上述 50 条试验轨迹：

| 指标 | 数量 |
|---|---:|
| 输入轨迹 | 50 |
| 输入步骤 | 1,166 |
| SWE 步骤 | 1,012 |
| Kimi 审计步骤 | 154 |
| 唯一 SWE 标注候选 | 609 |
| 未履行候选 | 0 |
| 人工审核 CSV 行数 | 609 |

主要排除项：

| 排除原因 | 数量 |
|---|---:|
| Action 不适合短语义匹配 | 258 |
| Intent 数量不等于 1 | 145 |
| Kimi 非 Phase 1 来源 | 154 |

Action 过滤命中包括 256 个 `edit`、256 个多行 Action、100 个长 payload、20 个整体过长 Action，以及少量赋值碎片和复合 shell。

### 3.5 Provisional negative pilot

已基于 609 条 SWE 自动候选固化独立 pilot：

| 指标 | 数量 |
|---|---:|
| provisional pairs | 609 |
| generation requests | 609 |
| train | 425 |
| validation | 92 |
| test | 92 |
| Kimi | 0 |

已实现：

- Inspire OpenAI-style endpoint 客户端、代理绕过、重试和 preflight；
- API key 只从 `INF_API_KEY` 读取，不写入状态、日志或仓库；
- 同工具错误参数、错误工具同对象、严格同轨迹无关 Action 三类校验；
- 每个正样本最多 5 个 negatives，按 3+1+1 稳定选择；
- request/prompt/endpoint/model/source manifest 共同决定 resume cache key；
- 原子状态文件、连续失败熔断、拒绝项审计；
- 只有全部 609 请求成功且一致时才允许 finalize；
- 每条 relation 永久标记 `provisional_auto_pair`、`feasibility_only`、
  `human_reviewed: false`。

### 3.6 基线与训练代码

已实现：

- dependency-free TF-IDF 和 BM25；
- Recall@1/3、MRR、Mean Rank；
- hard-negative pairwise accuracy 与 mean similarity margin；
- shared bi-encoder 的 InfoNCE + margin loss；
- frozen/fine-tuned encoder 统一评测接口；
- 训练配置、水印校验、checkpoint metadata 和 dry-run；
- 本地微型共享 encoder 单 batch 训练烟雾测试，不依赖模型下载。

训练入口已通过 CPU 单 batch smoke test。当前机器无可用 GPU，因此不在本环境
运行完整 frozen BGE 下载、评测或多 epoch 训练；这些步骤移交到 GPU 环境执行。
如需在当前 Inspire 镜像运行 venv，必须使用 `env -u LD_LIBRARY_PATH`，避免系统
Python 3.12 CUDA torch 覆盖 venv 中的 Python 3.13 CPU torch。

### 3.7 工程验证

当前 Phase 1 提交已通过：

- 116 项基础环境自动化测试通过，3 项按环境条件跳过；
- `.venv` 中额外 8 项 bi-encoder 测试通过，包括两个真实 PyTorch smoke tests；
- Python `compileall`；
- 仓库秘密、绝对路径、大文件和生成目录审计；
- Git diff 格式检查；
- 真实 50 条语料对账；
- 609 个候选 ID 唯一性检查；
- JSON、JSONL 和 CSV 完整解析；
- 零条符合条件的 `edit`、多行脚本及 heredoc Action。

## 四、Intent 抽取方案审计结论

### 4.1 当前分布

对现有 25 条 SWE、1,012 个步骤统计：

| 每步 Intent 数 | 步骤数 | 比例 |
|---|---:|---:|
| 0 | 14 | 1.4% |
| 1 | 867 | 85.7% |
| 2 | 120 | 11.9% |
| 3 | 9 | 0.9% |
| 4 | 2 | 0.2% |

其中 609 步同时满足单 Intent 和短 Action 条件，占全部 SWE 步骤的 60.2%，占单 Intent 步骤的 70.2%。

### 4.2 合理之处

- 不读取当前 Action，避免 Intent 标签受到答案污染；
- 只抽取 Thought 中显式选择或承诺执行的动作；
- 使用 `source_quote` 支持人工追溯；
- 保留自然语言 Intent，适合后续向量训练；
- 能较好处理搜索、打开、运行测试、删除、跳转等短操作；
- 使用列表即可表达当前任务，无需一开始引入树或图。

### 4.3 主要风险

1. **完整 Intent 抽取与 Phase 1 一对一训练存在边界差异。** Thought 可能同时承诺“修改脚本”和“运行测试”，但当前步骤只有一个 Action。这类多 Intent 步骤目前全部排除。
2. **Coding 场景原子性不完全稳定。** 例如 `ls path` 有时被拆成“进入目录”和“列出文件”；“创建文件并写代码”有时被合并。
3. **模糊执行表达判断不稳定。** `go back`、`review`、`we can search` 等表达在相同 Thought 中偶尔出现 0/1 Intent 不一致。
4. **上下文消耗较高。** 当前平均每步包含约 4.4k 字符用户任务和 5.4k 字符前序上下文，其中包含大量重复的 SWE 操作说明。
5. **批量调用保护仍需加强。** 正式执行前应增加 API preflight、连续失败熔断和 Prompt/模型配置变化时的 resume 隔离。

## 五、完整 500 条抽取准备情况

完整 500 条 SWE 已完成无 API 的 dry-run：

| 指标 | 数量 |
|---|---:|
| 输出轨迹文件 | 500 |
| 唯一轨迹 ID | 500 |
| 待抽取 Thought | 16,197 |
| dry-run 步骤 | 16,197 |
| 当前正式 Intent | 0 |

专用输出目录已经建立，约 205 MB，可用于断点续跑，且不会与现有 25 SWE + 25 Kimi 的试验结果混合。

按现有 25 条 SWE 的真实用量估算，直接执行当前方案约需要：

| Token 类型 | 估算数量 |
|---|---:|
| Prompt tokens | 约 6,037 万 |
| Completion tokens | 约 883 万 |
| 总 tokens | 约 6,920 万 |

因此目前尚未启动完整付费抽取。建议先发布 Thought → Intent v1.1，并进行约 300 步 pilot。

## 六、当前未完成事项

### P0：Intent 抽取定型

- 精简 USER REQUEST，只保留 Issue 主体；
- 将前序上下文控制在最近完整步骤约 3,000 字符；
- 增加 Coding 原子性 few-shot 示例；
- 统一 `review`、`go back`、`we can` 等表达的标注规则；
- 增加 API preflight 和全局错误熔断；
- 抽取约 300 个 pilot 步骤并人工审查至少 100 步；
- pilot 达标后再运行完整 16,197 步。

### P1：人工标注与 Gold 数据

- 对 Intent–Action 候选标注：`direct_match`、`partial_match`、`no_match`、`unfulfilled`、`ambiguous`；
- 冻结人工确认的测试集；
- 仅导出 `direct_match` 为训练正样本；
- 目标获得 500～1,000 个确认的一对一正样本；
- 独立人工测试集至少 200 个 Intent。

### P2：Negative 样本

离线生成、验证和 finalize 代码已经完成。当前剩余工作是安全设置
`INF_API_KEY` 后运行全部 609 个 request，并审查覆盖率。每个正样本构造
3～5 个 hard negatives，优先级为：

1. 同工具、错误参数；
2. 相似操作、错误工具；
3. 自动规则确认的同轨迹无关 Action；
4. in-batch 随机负样本。

必须排除可能部分履行 Intent、supporting/intermediate Action，以及仅因时序不同而被误判为负样本的 Action。

### P3：基线与模型训练

- TF-IDF 与 BM25 已实现，等待 finalized negatives 后运行；
- frozen BGE 与 fine-tuned shared bi-encoder 代码已实现；
- 未微调通用 embedding；
- LLM pairwise judge；
- 共享参数 bi-encoder；
- bi-encoder + 参数级 hard negatives。

pilot 默认使用 `BAAI/bge-small-en-v1.5` 以控制 CPU 成本，并报告 Recall@1、
Recall@3、MRR、Mean Rank、hard-negative pairwise accuracy 和 similarity
margin。fulfillment detection 仍需未履行样本后再加入。

## 七、下一阶段建议

建议按以下顺序继续：

1. 在 shell 中安全导出 `INF_API_KEY`，先执行 1 条真实 preflight/generation；
2. 协议确认后断点生成全部 609 条 negative proposals；
3. finalize provisional records，并统计每类 negative 覆盖率和拒绝原因；
4. 当前环境运行 TF-IDF、BM25；
5. 在 GPU 环境运行 frozen BGE 和最多 3 epoch 的 shared bi-encoder feasibility training；
6. 同时继续人工 Gold 标注，后续从基础 checkpoint 重新正式训练；
7. 完整 500 条 Thought → Intent 抽取仍作为扩大数据规模的独立工作流。

## 八、当前阶段结论

项目已经完成从原始轨迹筛选、Thought → Intent、Phase 1 候选、provisional
negative 生成入口、稀疏基线到 shared bi-encoder 训练入口的工程闭环。当前直接
阻塞不是代码，而是在线生成进程没有可用的 `INF_API_KEY` 环境变量，因此尚未
产生可训练的完整 609 条 negative records。

现阶段可概括为：

> 609 条 SWE provisional positives 已准备完成，negative 校验、基线和训练代码
> 已就绪；安全注入推理密钥后即可运行生成→finalize→baseline→bi-encoder，所有
> 结果只作为 feasibility pilot，后续仍须以人工 Gold 数据重训。
