# Thought–Intent–Action 项目两日工作简报

**汇报日期：** 2026 年 9 月 23 日
**工作时间：** 2026 年 9 月 22 日—23 日

## 一、这两天完成的主要工作

### 1. 完成 ClawGym 数据接入

- 下载并整理了 ClawGym 轨迹数据，当前 `data/clawgym/` 约 7.3 GB。
- 从原始数据中固定抽取了 1,000 条轨迹，转换为项目现有的逐轨迹 JSON 格式，并保存来源行号和哈希值，后续可以复现和核验。
- 扩展了现有 Thought → Intent 抽取代码，使其能够读取 ClawGym/Kimi 格式的轨迹，而不只支持 SWE-agent 数据。

### 2. 增加 Intent 的时序分类

之前只抽取 Thought 中包含哪些执行 Intent，这两天进一步给每个 Intent 增加了时序标签：

- `current`：下一批工具调用准备立即执行；
- `future`：需要在后续步骤执行；
- `conditional`：满足条件后才执行；
- `uncertain`：当前无法可靠确定执行时机。

这样可以避免把 Thought 中提到的完整计划全部错误地与当前 Action 匹配。只有 `current` Intent 才进入当前步骤的 Intent–Action 匹配。

围绕计划列表、`Let me ...` 表达、并行操作、条件动作和复合操作，连续进行了 v1—v5 五轮 Prompt 调整。五轮实验均使用相同的 5 条 ClawGym 轨迹和 297 个 Thought 步骤，全部请求执行成功。

最新 v5 共抽取 463 个 Intent，其中：

| 时序类型 | 数量 |
|---|---:|
| current | 362 |
| future | 94 |
| conditional | 2 |
| uncertain | 5 |

与最初版本相比，v5 将更多“后续计划”从 `current` 中分离出来；满足“恰好 1 个 current Intent + 1 个实际 Action”的步骤由 198 个增加到 223 个。该数字说明一对一匹配候选有所增加，但目前仍属于规则筛选结果，需要人工抽查后才能确定 Prompt 的最终版本。

### 3. 完成第二轮 hard negative 生成

在原有 609 条 SWE Intent–Action provisional 正样本上，重新进行了第二轮困难负样本生成和本地校验：

| 指标 | 第一轮 | 第二轮 |
|---|---:|---:|
| 正样本 | 609 | 609 |
| 最终保留 negatives | 1,599 | 1,600 |
| 同工具、错误参数 | 982 | 983 |
| 错误工具、相同对象 | 307 | 314 |
| 同轨迹无关 Action | 310 | 303 |
| 没有显式 negative 的样本 | 30 | 26 |
| API 请求失败 | 0 | 0 |

第二轮结果整体稳定，并将零显式 negative 的样本从 30 条减少到 26 条。所有数据仍标记为 `provisional_auto_pair`，尚未经过人工 Gold 审核。

### 4. 完成向量模型微调和结果判断

使用 BGE-base 共享双编码器，在 H200 GPU 上进行了 3 个 epoch 的 Intent–Action 匹配训练。训练使用 BF16，并采用 InfoNCE 与 hard-negative margin loss。

测试集包含 92 条 Intent。与未微调的 frozen BGE 相比：

| Test 指标 | Frozen BGE | 微调后 BGE | 变化 |
|---|---:|---:|---:|
| Recall@1 | 0.772 | 0.870 | +0.098 |
| MRR | 0.877 | 0.929 | +0.053 |
| Mean Rank | 1.283 | 1.163 | -0.120 |
| Hard-negative pairwise accuracy | 0.890 | 0.935 | +0.045 |
| 正负相似度 margin | 0.053 | 0.135 | +0.083 |

训练 loss 从 1.318 下降到 0.604，validation MRR 从第一轮的 0.922 提升到第三轮的 0.938，最佳 checkpoint 为第 3 个 epoch。按照当前固定阈值判断器，该实验结果为 `STRONG_PASS`。

## 二、典型实际案例

以下内容均直接摘自实际生成文件。

### Case 1：顺序计划中的 current / future Intent

来源：`clawgym-L62-8a15ba70f020`，Thought Step 2。

**Thought**

```text
Now I need to:
1. Create the output directories
2. For each species, fetch the IUCN Red List assessment page
3. Parse the HTML to extract the required fields
4. Create the consolidated CSV
5. Filter and rank for the shortlist
6. Create the summary JSON

Let me start by creating the directories and then fetch each species page from IUCN.
```

**Action**

```json
{
  "function": "exec",
  "args": {
    "command": "mkdir -p /root/.openclaw/workspace/data/raw_html /root/.openclaw/workspace/data/parsed /root/.openclaw/workspace/data/reports"
  }
}
```

**Intent**

```json
[
  {"intent_id":"T2-I1","intent_text":"Create the output directories.","source_quote":"Create the output directories","temporal_scope":"current"},
  {"intent_id":"T2-I2","intent_text":"Fetch the IUCN Red List assessment page for each species.","source_quote":"fetch each species page from IUCN","temporal_scope":"future"},
  {"intent_id":"T2-I3","intent_text":"Parse the HTML to extract the required fields.","source_quote":"Parse the HTML to extract the required fields","temporal_scope":"future"},
  {"intent_id":"T2-I4","intent_text":"Create the consolidated CSV.","source_quote":"Create the consolidated CSV","temporal_scope":"future"},
  {"intent_id":"T2-I5","intent_text":"Filter and rank for the shortlist.","source_quote":"Filter and rank for the shortlist","temporal_scope":"future"},
  {"intent_id":"T2-I6","intent_text":"Create the summary JSON.","source_quote":"Create the summary JSON","temporal_scope":"future"}
]
```

### Case 2：同一批并行 Action

来源：`clawgym-L242-e821f22628a9`，Thought Step 1。

**Thought**

```text
The user wants me to set up a trend check for Android security risks. Let me break down the tasks:

1. Download the most recent three Android Security Bulletin pages from source.android.com
2. Create a manifest of downloaded files
3. Parse the HTML files using the provided input files
4. Create a trend table
5. Detect month-over-month changes
6. Create a parent-friendly report

First, let me read the input files to understand the structure:
- input/risk_mapping.json
- input/components.txt
- input/report_template.md

Then I'll need to find and download the Android Security Bulletins from source.android.com.
```

**Action**

```json
[
  {"function":"read","args":{"path":"/root/.openclaw/workspace/input/risk_mapping.json"}},
  {"function":"read","args":{"path":"/root/.openclaw/workspace/input/components.txt"}},
  {"function":"read","args":{"path":"/root/.openclaw/workspace/input/report_template.md"}}
]
```

**Intent**

```json
[
  {"intent_id":"T1-I1","intent_text":"Read the input file input/risk_mapping.json.","source_quote":"let me read the input files","temporal_scope":"current"},
  {"intent_id":"T1-I2","intent_text":"Read the input file input/components.txt.","source_quote":"let me read the input files","temporal_scope":"current"},
  {"intent_id":"T1-I3","intent_text":"Read the input file input/report_template.md.","source_quote":"let me read the input files","temporal_scope":"current"},
  {"intent_id":"T1-I4","intent_text":"Find and download the Android Security Bulletins from source.android.com.","source_quote":"find and download the Android Security Bulletins from source.android.com","temporal_scope":"future"}
]
```

### Case 3：条件 Intent

来源：`clawgym-L14817-51eba14f8aaf`，Thought Step 38。

**Thought**

```text
Let me try a different approach. Since I already have items, let me focus on downloading what we can and compiling the data. Let me check on this and try a different method if needed.
```

**Action**

```json
{
  "function": "process",
  "args": {"action":"poll","sessionId":"brisk-seaslug","timeoutMs":"20000"}
}
```

**Intent**

```json
[
  {"intent_id":"T38-I1","intent_text":"Check on the status of the wget download of the arhivi.gov.lv item.","source_quote":"Let me check on this","temporal_scope":"current"},
  {"intent_id":"T38-I2","intent_text":"Try a different download method for the arhivi.gov.lv item if the current one fails.","source_quote":"and try a different method if needed.","temporal_scope":"conditional"}
]
```

### Case 4：无工具执行 Intent

来源：`clawgym-L62-8a15ba70f020`，Thought Step 58。

**Thought**

```text
All files are in place. Let me provide a final summary of what was created.
```

**Action**

```json
[]
```

**Intent**

```json
{
  "execution_intents": [],
  "no_intent_reason": "The Thought only plans to provide a natural-language final summary of what was created, with no concrete tool or environment operation named."
}
```

### Case 5：Hard negative 生成

来源：`swe:CASE-002:T11-I1`。

**Thought**

```text
The script has been updated to use `nipype.Memory` for caching. Now, let's run the script to verify that the fix works.
```

**Action（Positive）**

```text
[ACTION] python(command="reproduce.py")
```

**Intent**

```text
Run the reproduction script to verify that the fix works.
```

**生成的 Negative Actions**

```text
[same_tool_wrong_parameter]
[ACTION] python(command="reproduce_crash.py")

[same_tool_wrong_parameter]
[ACTION] python(command="test_verify2.py")

[wrong_tool_same_object]
[ACTION] create(path="reproduce.py")

[same_trajectory_unrelated]
[ACTION] find_file(args="model.py nipype/interfaces/fsl")
```

## 三、当前阶段判断

这两天主要完成了两件事：

1. 将项目从 SWE-agent 单一数据源扩展到 ClawGym，并开始解决 Thought 中“当前动作”和“未来计划”混在一起的问题；
2. 跑通了 provisional 数据上的 BGE-base 微调，初步证明 hard-negative 训练能够提升 Intent 与正确 Action 的区分能力。

但当前结果仍是**可行性验证，不是正式模型结论**，主要原因是：

- Intent–Action 正样本和 negative 尚未完成人工 Gold 审核；
- 当前向量模型评测是每条 Intent 在其局部 hard-negative 候选中排序，还不是在完整 Action 库中检索；
- 测试集只有 92 条 Intent，来自 7 条独立轨迹，规模仍较小；
- ClawGym temporal Prompt 的 v5 结果尚需人工抽查，不能仅根据自动统计确定最终版本。

## 四、下一步工作

1. 人工抽查 ClawGym v4/v5 的 Intent 与时序标签，确定最终 Prompt；
2. 将确认后的 ClawGym `current` Intent 接入 Intent–Action 候选构建；
3. 人工审核一批 SWE 正负样本，建立独立 Gold validation/test 集；
4. 统一向量模型与 TF-IDF/BM25 的全局候选池，进行可直接比较的检索评测；
5. 输出逐样本排名和错误案例，分析模型主要混淆的是工具类型、目标对象还是参数。

## 五、关键产物位置

- ClawGym 1,000 条轨迹：`data/clawgym/selected_1000_json/`
- ClawGym temporal v1—v5：`data/clawgym/thought_intent_temporal_trial_5*/`
- 第二轮 hard negatives：`data/SWE-agent-trajectories/test_data/provisional_intent_action_negative_trial_v2/`
- BGE-base 训练结果：`data/SWE-agent-trajectories/experiments/provisional_biencoder_h200_bge_base_20260922T064148Z/`
- ClawGym 适配代码分支：`feature/clawgym-intent-adaptation`
- 模型结果判断代码分支：`feature/biencoder-report-judgment`
