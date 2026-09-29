# BGE Intent–Action 训练数据方案 v2（2026-09-29）

## 目标与现状

训练单位是一个 `due_now` Intent 与其**完整满足该 Intent**的 Action，输出一正多负样本。一个 Intent 可有多个正确 Action，一个 Action 也可对应多个 Intent；正例按 `(trajectory_id, thought_step, intent_id, action_id)` 建立边，不强制一对一。

当前 train 有 5,393 条初判 `exact_match` Action 记录，其中 `exec` 1,809、`read` 1,657、`write` 1,436、`web_fetch` 286、`edit` 144，其余 61。旧版 5,000 条负例只覆盖 2,409 个来源 Action，平均约 2.1 条负例。旧数据保留作基线，不直接充当 v2 的完整训练集。原始匹配仍为模型初标 Silver，不能称作人工 Gold。

## 1. 先建正例池

1. 从现有 train split 的 `exact_match` 匹配取所有 `selected_intent_ids`，逐条展开成 Intent–Action 边；优先检查 `revised_intents`，保留原始 Action、原始哈希、来源 ID、split 和模板簇。将 `ambiguous`、`partial_match`、`no_intent_match` 及结构不清的 Action 放入复核队列。
2. 按工具定义关键语义字段，而非统一要求 `path/content`：文件工具检查操作、目标及内容或编辑范围；`web_fetch` 检查 URL；`exec` 检查命令语义、目标文件及关键参数；其他工具制定对应字段表。仅有相似措辞不足以成为正例。
3. 对每个 Intent 汇总**所有已知正确 Action**，区分“任一条即可完成”和“多步合起来完成”。后者若单条 Action 只起支持或中间作用，不作为该 Intent 的完整正例。未知或依赖隐藏状态的边暂缓。
4. 对长 Action 保存完整 raw JSON 及哈希；另生成训练用语义序列化文本。若模型 token 限制会截掉判别所需字段，该样本不进入本轮训练。不得用截断文本替代原始证据。

先报告各工具族的正例数与可用率。目标是覆盖能明确判定的正例，**不因负例暂时不足而从正例池删除**；未达到训练门槛的正例单列 `positive_only`，用于后续补充。

## 2. 分层构造负例

首批每个正例优先配 **3 条显式负例：P1 两条、P2 一条**；训练时再从 batch 内采样 1 条经过冲突过滤的 P4，形成约 4 个有效负例。若有合格候选，可补到 4 条显式负例；不足时记录缺额，不为凑数加入可疑样本。首批不强制 `P1:P2:P3:P4 = 2:1:1:1`，P3 留作后续实验。P4 动态采样，**不写成固定候选**。

| 类别 | 候选构造 | 必须验证 |
|---|---|---|
| P1 同工具错参数 | 保持真实工具名及参数模式，仅替换目标路径、URL、实体、范围，或改动决定结果的一个参数/内容片段 | Intent 明确约束该字段；替换后不能仍完成 Intent；保持其余关键字段与工具 schema 有效 |
| P2 错操作 | 在数据里确有、可执行且语义相近的工具/操作间构造，例如 `read`、`write`、`edit`；`exec` 可在命令级构造相近操作 | 目标与任务上下文保留，参数符合**新**工具 schema；操作实际不同且不能完成 Intent。不能凭空使用 `append/update/patch` 工具名 |
| P3 语义近邻（后续可选） | 仅当首批评测显示模型仍常被真实相似 Action 混淆时，再从 train 内真实 Action 用 BM25 或简单关键词检索少量候选 | 逐个排除其他正确 Action、同义实现、部分履行和支持步骤；检索相似本身不构成负例标签 |
| P4 in-batch | 从 batch 内其他 query 的正例 Action 动态提供 | batch 按 Intent/目标等价类去重；已知可互为正例的 Action 不得互当负例；冲突时 mask 掉该负例 |

旧版通过复核的负例可进入 P1/P2 候选池，但须重新按 v2 规则验证。P1/P2 构造时每次只改变预定语义维度；跨工具变换允许按新 schema 同步调整必需字段，这种调整不算额外语义错误。若日后启用 P3，其候选也只能来自 train，dev/test 不参与训练候选挖掘。

### GQQ 辅助生成

首批采用“规则候选 + GQQ 补充候选”：先用可检查的参数替换生成 P1/P2；对缺额的正例，再将当前 Intent、原始 Action、同 step 已知正例及工具 schema 交给 GQQ，请它最多提出少量 P1/P2 候选，并返回 `category`、完整 `candidate_action`、`changed_fields`、`why_incorrect`。允许返回空列表，不能要求模型凑数。生成时只提供 train 数据，不让模型调用工具或执行所提出的 Action。

GQQ 的提议先过程序校验：工具必须真实存在；P1 只能改变指定参数；P2 必须保留目标并符合新工具 schema；与任何已知正例或候选文本重复的剔除。之后使用**独立的复判请求与提示词**检查是否完整不匹配、是否属于目标类别、是否只是部分履行或中间步骤。同一个 GQQ 模型的两次调用仍属于模型复核，不能当成人工或独立模型一致性证据；争议样本进人工队列。保存生成提示版本、模型版本、原始响应、规则结果和复判理由，以便追溯。

## 3. 排除与复核

一律排除：已知正例或等价实现；`partial_match`；supporting/intermediate Action；目标不明确以致替换后仍可能满足 Intent；仅靠时序不同被误判、但语义上可以完成 Intent 的 Action；无效工具调用；关键判别字段在训练文本中被截断的样本。

时序只作为证据，不作为自动负例标签。若相同 Intent 下某 Action 的语义确实错误，保留并记录错误证据。判断时只使用当时可见的 Intent、Action 与上下文，不根据后续执行结果臆断隐藏状态。

规则校验检查 schema、原始/生成哈希、修改维度、同 split、重复文本和正负碰撞；GQQ 用独立请求逐条复判正负关系与类别。模型拒绝、意见冲突、跨工具、`exec` 和长文本进入人工抽查/争议队列。人工确认的样本标 `gold_reviewed`；其余明确标 `silver_model_validated`，不混称 Gold。先抽查每类和每个主要工具族的错误率，再决定扩量。

## 4. 数据格式与采样

主文件一行一个正例边：`query`、`pos`（可含多个等价正例）、`neg`（每条带 P1/P2 类别）、`intent_id`、`action_id`、`trajectory_id`、`split`、`raw_action_sha256`、`supervision_tier`、`validation_evidence`。训练器可选一个正例及 3–4 个显式负例，并对 P4 做 false-negative mask。保留候选级 JSONL 与排除队列，便于追溯。

同一 Intent 或等价 Intent 若有多个正例，训练时采用多正例目标或将其他正例从负例集合中 mask；不能在普通 InfoNCE 中把它们当作负例。按来源边均匀采样，限制高频模板、重复 Action 和单一工具族主导训练。

## 5. 验收和测评

先做小批试产，报告：正例池覆盖率、P1/P2 的生成/接受率、平均每正例显式负例数、达到 3 条和 4 条的覆盖率、工具族分布、人工抽查误标率、被排除原因和模型输入截断率。显式负例数量只按**合格落盘样本**计算，不拿 P4 凑数。

使用现有轨迹与模板簇隔离的 dev/test。dev 用于选模型和阈值；test 固定候选池、复核所有可能正例及高排名候选后只评一次。主指标为 Recall@1/5、MRR，另报 P1/P2 分层 pairwise accuracy、无匹配拒识误报率；部分匹配单独评价，不混入完整匹配的二分类负例。先在**相同正例池、相同训练步数**下比较旧负例与 v2 负例，以隔离负例设计的效果；再比较扩展正例池后的完整 v2，并做去掉 P1/P2 的消融。若真实近邻错误仍突出，再单独增加 P3 实验。

## 执行顺序

`正例池与工具 schema → 多正例/排除索引 → 规则生成 P1/P2 → GQQ 补缺 → 程序校验与独立 GQQ 复判 → 小批人工抽查 → 完整训练文件及 manifest → 固定 dev/test 对照评测`。P3 只在测评暴露相应短板后启动。

旧版产物和脚本不被覆盖；v2 数据使用独立版本目录。

## 2026-09-29 首批执行结果

数据位于 `data/clawgym/bge_negatives_v2_20260929/`。共建立 5,465 条 train 正例边；其中 4,070 条通过原子性、复核标记和训练文本长度筛选。得到 6,900 条通过规则和 GQQ 复判的 P1/P2 候选；1,940 条去重后的正例拥有完整的 `2×P1+1×P2`，对应 `train.jsonl` 中 5,820 个互不重复的训练三元组。另有 2,098 条合格正例边未配齐三负例，保存在 `positive_only.jsonl`；未通过正例筛选的边仍保存在 `positive_pool.jsonl`。

训练行的工具覆盖为 `read` 1,143、`write` 640、`exec` 96、`edit` 61。已有 4,061 条旧版 GQQ 复判候选经 v2 规则重新验收；其他接受的候选由规则生成并经 GQQ 复判。GQQ 新提出的补缺候选在本次严格规则下没有被接受，未为凑数量放宽约束。P4 尚未实现为训练采样器，本次产物只包含显式 P1/P2。全部标签仍为 Silver，未做人工 Gold 审核。数量、来源和输入哈希见 `manifest.json`。

完整 JSONL 数据留在工作区的外部数据目录，不提交 Git。Git 跟踪生成脚本及 `data/manifests/clawgym-bge-negatives-v2/manifest.json`，其中记录输入哈希与各输出文件哈希。重跑时给脚本传入 `--matches`、`--actions`、`--intents`、`--old-negatives`、`--output`；GQQ 凭据由 `INTENT_API_KEY` 提供，兼容接口由 `INTENT_API_BASE_URL` 指定，模型名可通过 `INTENT_MODEL_NAME` 设置。脚本支持从已有 `validated_candidates.jsonl` 续跑；调用 GQQ 时绕过代理。
