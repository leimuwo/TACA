# Thought–Action 匹配项目近期进展

日期：2026-10-09

## 一、近期完成的工作

近期主要围绕 ClawGym 轨迹中的 Thought、Intent 和 Action，完成了从数据解析、Intent 提取、匹配标注、负例构造到向量模型训练与误差分析的一套流程。

### 1. ClawGym 数据分析与 Intent 提取

- ClawGym 主数据共 24,497 条长轨迹，当前本地数据中可观察到的轨迹生成模型标记均为 MiniMax-M2.5。
- 任务主要覆盖文件处理、数据分析、网页检索、代码执行、报告生成和自动化工作流等复杂 Agent 场景。
- 已从其中选取 1,000 条代表性轨迹进行完整时序 Intent 提取，共处理 11,996 个 Thought step，得到 16,191 个执行 Intent，其中：
  - 当前应执行的 Intent：12,626 个；
  - 未来计划 Intent：3,386 个；
  - 条件或不确定 Intent：179 个。
- 在提取过程中保留 `preceding_context`，即当前 Thought 之前的用户请求、历史 Thought、Action 和工具返回，用于区分“当前要执行的动作”和“后续计划”。

### 2. Intent–Action 匹配数据构建

- 对 925 条完整轨迹进行了 Action 结构化、Intent 校验及 Intent–Action 匹配，共包含 13,011 个 Action。
- 当前匹配采用多对多形式：一个 Intent 可以对应多个等价 Action，一个 Action 也可以同时覆盖多个当前 Intent。
- 匹配标签包括 `exact_match`、`partial_match`、`object_mismatch`、`type_mismatch`、`no_intent_match` 和 `ambiguous`，当前结果属于模型辅助生成的 Silver 数据，不作为人工 Gold 标注。
- 已按照轨迹和模板簇划分训练、验证和测试集，分别为 647、139、139 条轨迹，避免同一任务模板跨集合泄漏。

### 3. Negative Case 构造

- 围绕正确的 Intent–Action 对构造了两类主要难负例：
  - P1：工具相同，但路径、URL、实体或关键参数错误；
  - P2：目标相近，但操作类型错误，例如“读取文件”和“写入文件”、“创建脚本”和“运行脚本”。
- 共整理 5,465 条正例边，其中 4,070 条通过原子性、文本长度和标注质量筛选。
- 最终得到 6,907 条通过规则及模型复判的负例候选，包括 4,221 条 P1 和 2,686 条 P2。
- 形成 2,711 行训练数据、6,802 个有效训练三元组；另有 1,322 条正例因为缺少足够可靠的显式负例而暂不进入训练。

### 4. BGE 向量模型微调与评测

- 基于 `bge-base-en-v1.5` 完成 Intent–Action 双塔检索模型微调。
- 在验证集上，基础模型 Top-1 为 27.46%、MRR 为 36.94%；微调后最佳 Top-1 为 42.39%、MRR 为 52.37%，说明模型已明显学到工具类型、目标对象及操作语义之间的对应关系。
- 在原始测试集上，微调模型取得：
  - Top-1：43.39%；
  - Top-5：65.74%；
  - Top-10：73.18%；
  - MRR：53.62%。

## 二、最近的数据质量修正

### 1. 模糊 Intent 筛选

使用模型对训练、验证和测试中的 4,460 条去重 Intent 进行模糊性检查，共发现 1,049 条模糊 Intent，主要问题是使用了“the file”“remaining items”“the report”等缺少明确目标的指代表达。

- 清晰 Intent：3,411 条；
- 模糊 Intent：1,049 条；
- 其中测试集排除 270 条模糊 Intent，保留 804 条清晰 Intent。

在清晰测试子集上，原始 Top-1 为 52.36%，Top-5 为 76.62%，Top-10 为 83.46%。

### 2. 等价 Action 漏标修正

检查发现部分 Action 的序列化文本完全相同，但 Action ID 不同，原评测只按 ID 判断，导致实际正确结果被计为错误。

- 测试集补充 191 条完全相同文本的等价正例边，影响 70 个查询；
- 清晰测试集有 14 条 Top-1 因此得到修正；
- 修正后清晰测试集 Top-1 从 52.36% 提升至 54.10%；
- 验证集也已完成相同处理，新增 278 条等价正例边，影响 119 个查询。

## 三、当前发现的主要问题

对 Top-1 错排样例进行人工抽查后，当前错误主要集中在以下几类：

1. **相邻执行步骤混淆**：Intent 要求“创建脚本”，模型却检索到后续的“运行脚本”；要求“检查文件”，模型却检索到“重新下载文件”。
2. **`exec` 命令语义区分不足**：命令主题相似，但 URL、文件路径、保存位置或关键参数不同。
3. **同工具、错目标**：例如都使用 `write`，但写入了另一个文件；都使用 `web_fetch`，但访问了不同网页。
4. **Intent 本身缺少上下文**：没有明确写出目标文件、查询主题或前序对象，导致多个 Action 在字面上都可能成立。
5. **Silver 标注存在漏标**：部分未标注 Action 实际可能是等价实现，需要结合原始任务和执行结果进一步人工复核。

目前的结果说明，单纯依赖主题词和工具名已经不足，后续重点需要加强对“操作阶段、目标对象、关键参数和执行结果”的建模。

### 失败 Case 总体分布

以下统计采用修正完全相同 Action 文本漏标后的清晰测试集。共评测 804 条 Intent，Top-1 正确 435 条，失败 369 条，当前 Top-1 错误率为 45.90%。由于评测标签仍是 Silver，369 条表示“首位没有命中当前已知正例”，其中仍可能包含尚未补标的等价 Action。

#### 1. 已知正例所在排名

| 已知正例排名 | 失败数 | 占全部失败比例 |
| --- | ---: | ---: |
| 2–5 | 181 | 49.1% |
| 6–10 | 55 | 14.9% |
| 11–100 | 86 | 23.3% |
| 100 名以后 | 47 | 12.7% |
| 合计 | 369 | 100.0% |

共有 236 条失败的已知正例仍位于 Top-10，占失败 Case 的 64.0%。这说明多数失败并不是正确 Action 完全没有被模型识别，而是相关 Action 之间的细粒度顺序没有排对。

#### 2. 按正确 Action 工具分布

| 正确 Action 工具 | 失败数 | 占全部失败比例 |
| --- | ---: | ---: |
| `exec` | 228 | 61.8% |
| `write` | 91 | 24.7% |
| `read` | 15 | 4.1% |
| `web_fetch` | 14 | 3.8% |
| `process` | 8 | 2.2% |
| `browser` | 8 | 2.2% |
| `edit` | 5 | 1.4% |

`exec` 和 `write` 合计占失败 Case 的 86.5%。其中 `exec` 参数结构复杂、可能封装多步操作；`write` 则经常包含相同主题但不同路径或内容的大段文本，因此是目前最主要的两个难点。

#### 3. 错误 Top-1 工具分布

| 模型排在首位的错误工具 | 失败数 | 占全部失败比例 |
| --- | ---: | ---: |
| `exec` | 151 | 40.9% |
| `write` | 93 | 25.2% |
| `web_fetch` | 64 | 17.3% |
| `read` | 34 | 9.2% |
| `edit` | 26 | 7.0% |
| `session_status` | 1 | 0.3% |

#### 4. 主要工具混淆组合

| 正确工具 → 错误 Top-1 工具 | 数量 | 占全部失败比例 | 典型表现 |
| --- | ---: | ---: | --- |
| `exec → exec` | 123 | 33.3% | 命令相关，但脚本、URL、路径或关键参数错误 |
| `write → write` | 60 | 16.3% | 写入主题相近，但目标文件或具体内容错误 |
| `exec → web_fetch` | 45 | 12.2% | 应搜索、下载或检查本地文件，却直接抓取网页 |
| `exec → write` | 31 | 8.4% | 应执行命令，却选择生成文件内容 |
| `exec → read` | 21 | 5.7% | 应执行处理或检查命令，却只读取相关文件 |
| `write → exec` | 15 | 4.1% | 应创建文件，却选择运行相关脚本 |
| `write → edit` | 14 | 3.8% | 应完整写入目标文件，却选择局部修改 |

全部失败中，210 条是“正确工具与错误 Top-1 工具相同”，占 56.9%；159 条为跨工具混淆，占 43.1%。因此问题不能只通过工具分类解决，更关键的是区分同一工具内部的目标、参数和操作效果。

#### 5. 人工定向抽查结果

针对最主要的 `exec → exec` 和 `exec → web_fetch` 两组，各分层抽取 22 条，共人工初审 44 条：

| 初审结论 | 数量 | 占抽查样本比例 |
| --- | ---: | ---: |
| 明确不满足 Intent | 29 | 65.9% |
| 字面上可能等价 | 7 | 15.9% |
| 需要核对任务或执行结果 | 7 | 15.9% |
| 原已知正例本身有明显问题 | 1 | 2.3% |

该抽样专门针对两类高频错误，不能直接外推为 369 条失败的总体比例。不过它表明两点：多数确实是模型错排；同时约三分之一仍涉及等价实现、上下文不足或 Silver 标签质量，需要人工复核后才能最终定性。

### 模糊 Intent 分布

在全量 4,460 条去重 Intent 中，共筛出 1,049 条模糊 Intent，占 23.5%。其问题分布如下：

| 模糊类型 | 数量 | 占模糊 Intent 比例 |
| --- | ---: | ---: |
| 模糊指代 `vague_reference` | 908 | 86.6% |
| 范围不明确 `underspecified_scope` | 77 | 7.3% |
| 缺少目标 `missing_target` | 40 | 3.8% |
| 操作不清楚 `unclear_operation` | 22 | 2.1% |
| 多个 Action 都可能成立 | 2 | 0.2% |

其中绝大多数问题来自“the file”“the outputs”“remaining items”“the same issue”等依赖上下文的表达。按数据划分，模糊 Intent 在 train/dev/test 中分别为 526、253、270 条。当前清晰测试集已排除 test 中的 270 条，但训练集中的模糊 Intent 仍需要改写、补充上下文或降低监督权重。

## 四、常见 Bad Case

以下案例均来自清晰测试集的实际 Top-1 错排结果。

### Case 1：检查目录与创建目录混淆

Intent：

```text
Check the directory structure to verify all outputs are in place.
```

正确 Action：

```text
exec: cd /root/.openclaw/workspace/workspace &&
      find . -type f -name "*.csv" -o -name "*.html" -o -name "*.md" | sort
```

错误 Top-1：

```text
exec: mkdir -p /root/.openclaw/workspace/outputs
```

问题：两者都涉及输出目录，但 Intent 要求检查已有结果，错误 Action 执行的是创建目录。模型匹配到了相同对象，却没有区分“检查”和“创建”的操作阶段。

### Case 2：运行指定脚本与检查 Python 环境混淆

Intent：

```text
Run the script with python3 instead of python.
```

正确 Action：

```text
exec: python3 scripts/score_prospects.py input/prospects.csv
      input/baseline_injury_days.csv input/device_effects.csv --outdir output
```

错误 Top-1：

```text
exec: python3 --version
```

问题：错误 Action 包含高度相关的关键词 `python3`，但没有运行指定脚本。说明模型容易关注命令表面词汇，而忽略脚本路径、输入文件和预期操作。

### Case 3：遗漏决定行为的关键参数

Intent：

```text
Fetch https://worldbank.org/robots.txt with curl following redirects
and save the response body to output/raw/world-bank_robots.txt.
```

正确 Action：

```text
exec: curl -sL -o output/raw/world-bank_robots.txt \
      -w "%{http_code}" https://worldbank.org/robots.txt
```

错误 Top-1：

```text
exec: curl -s -o output/raw/world-bank_robots.txt \
      -w "%{http_code}" https://worldbank.org/robots.txt
```

问题：两个命令的 URL 和输出路径完全相同，但错误 Action 缺少跟随重定向所需的 `-L`。这是典型的“同工具、同目标、错关键参数”难例。

### Case 4：检查本地文件与重新访问网页混淆

Intent：

```text
Verify that downloads/satnam_singh_nba.html contains the string
'satnam' (case-insensitive).
```

正确 Action：

```text
exec: grep -i "satnam" downloads/satnam_singh_nba.html | head -3
```

错误 Top-1：

```text
web_fetch: https://www.nba.com/players/satnam_singh
```

问题：错误 Action 与主题实体 Satnam Singh 高度相关，但 Intent 明确要求验证已经下载的本地文件，而不是重新抓取远端页面。

### Case 5：搜索与直接抓取目标页混淆

Intent：

```text
Search for MSHA conveyor safety information using the serper skill.
```

正确 Action：

```text
exec: python3 scripts/search.py
      -q "MSHA conveyor safety site:msha.gov" --limit 10
```

错误 Top-1：

```text
web_fetch: https://www.msha.gov/safety-and-health/hazards-by-topic/conveyor-safety
```

问题：错误 Action 打开了主题相关页面，但没有执行 Intent 指定的搜索过程。搜索和直接抓取在结果主题上接近，在工具和过程要求上并不等价。

### Case 6：抓取网页与带参数落盘混淆

Intent：

```text
Fetch the UNICEF homepage with a user agent.
```

正确 Action：

```text
exec: curl -sL -A "Mozilla/5.0 ..." \
      -o output/raw/unicef_homepage.html https://unicef.org
```

错误 Top-1：

```text
web_fetch: https://unicef.org
```

问题：错误 Action 能读取同一个网页，但没有设置 User-Agent，也没有将原始页面保存到指定路径，只完成了部分语义。

这些案例说明，当前模型已经能够识别大致主题和相关工具，但对操作类型、执行阶段、目标位置及关键参数的区分仍然不足。这也是下一轮难负例构造的重点。

## 五、下一步计划

1. 继续复核清晰测试集中排名靠前但未标注的候选 Action，完善等价正例集合。
2. 针对“创建/运行/检查”“搜索/抓取/下载”“相同工具但错误路径或 URL”等错误补充高质量难负例。
3. 对模糊 Intent 补充必要的 `preceding_context` 信息，或改写为目标明确、可独立判断的 Intent。
4. 建立一小批人工 Gold 测试集，避免仅依赖 Silver 标签评价模型效果。
5. 在固定评测集上重新训练和对比，重点观察 Top-1、MRR 以及 `exec`、`write` 等主要工具类别的分项表现。

## 六、阶段性结论

目前已经打通“ClawGym 轨迹解析—时序 Intent 提取—Intent/Action 匹配—难负例生成—BGE 微调—错误分析”的完整实验链路。微调模型相比基础模型已有明显提升，但当前性能仍同时受到 Intent 信息不足、相邻步骤干扰和 Silver 正例漏标的影响。近期工作的重点已经从单纯扩充数据，转向提高 Intent 表达和评测标注的可靠性，为下一轮训练及人工 Gold 评测做准备。
