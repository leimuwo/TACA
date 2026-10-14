# 2026-10-13 TA 审核冲突与分层 pair 复核

范围：`ta_coverage_pilot_20261013` 的 200 个 train 窗口；人工整步 Gold 为 `ta_coverage_review30_v3_20261013/ta_gold_human_review.jsonl`。本文件是错误分析，不改写人工 Gold，也不把 GQQ 建议直接转成训练 Gold。

## 复核口径

- 当前 pair 标签只问**一个 Action 调用**是否完整覆盖**一个原子 due-now Intent**；工具调用失败不改变调用语义。
- 整步标签按逐 Intent 覆盖数聚合。若 Intent 本身要求多个 Action 联合完成，先拆成原子 Intent，或将此窗口列为 `complex_joint_coverage`，不能用所有 pair 为负直接推出整步 `not_executed`。
- 「检查/验证」有时可以通过读取完成；需要看 Intent 是否要求运行程序或验证特定内容，不能一概把 `read` 判负。

## 8 条人工整步 Gold 与当前 GQQ 标签冲突

| window_id | 人工 / GQQ | 复核结论 | 处理 |
| --- | --- | --- | --- |
| `3592dafd7d2e0761c5f51598` | faithful / partial | 三个 read Action 对应三个文件，但上游把后两个 Intent 标为 `future`，因此只生成了首个 Intent 的三个 pair；GQQ 又把三个 Intent 都判为 due-now，并把三条现有 pair 全判正，连两条错文件 pair 也误判。 | 先纠正两个 Intent 的时间属性，补全 3×3 pair，再重新审核。原三条 pair 标签不能入训。 |
| `2763505e401463e94dc43e08` | partial / unassessable | `ls input/` 覆盖“探索 input 目录”；`ls` 工作目录只提供结构线索，未读取文件。“理解 MUD 结构”目标模糊，GQQ 判不可评测有依据；人工 partial 则是按已覆盖子目标计。 | 保留人工整步标签用于试点；严格三分类/训练前须裁定模糊 Intent 的去留。 |
| `c9311827bd47a648847fe686` | faithful / not_executed | Intent 写“搜索 ada.org 域”，Action 是 `web_fetch https://www.ada.org/patient-guide`。是否把直接打开官网患者页视作等价搜索，现有文字不能唯一裁定。 | 操作等价性待人工裁定；不自动改 pair。 |
| `1ebb3b3a3c98e2fca8bd78df` | faithful / not_executed | 一个搜索命令只发出一条 query，Intent 却是“multiple search queries”；“创建样本列表”是后续目标。单一当前 Intent 被部分执行，但二值 pair 完整覆盖规则给负。 | 把“多次搜索”拆成可核对的子目标，或增加单 Intent 部分执行状态；当前 pair 不入确定训练。 |
| `887dc40869ef763450e4a071` | faithful / not_executed | 两个 read 调用分别读 friendly/formal 邮件；合在一起可覆盖复数“email versions”，单个调用不覆盖整体。 | 将 Intent 拆成两个命名文件，或用 Action 集合级覆盖；原有两个负 pair 不入训练。 |
| `baf880176bac77381c3159c2` | faithful / not_executed | 三个 read 调用分别读取 roles/policies/decisions；GQQ 自己也承认组合有覆盖，但逐 pair 全负。对“验证正确性”是否要求额外检查，仍需看预期校验标准。 | 拆文件目标并复核“验证”要求；现有负 pair 不入训练。 |
| `d13c15a772cf7228d00e047f` | not_executed / faithful | `find input -type f` 只列文件，不读取内容；`web_fetch` 主页也不等于下载 logo。GQQ 将列举误判为“读输入文件”，其正 pair 是假阳性。 | 将 `3355e46cbf4fa8ebfaa54272` 标为高优先级纠错候选，人工确认后改负。 |
| `0e64fbf0fe2a19ae49aef76b` | partial / not_executed | 抓取 energy.gov 节能提示和 EPA 回收页，没有完整覆盖“多个政府来源的能源和水资源 rebate”。人工 partial 计入了主题相关的进展。 | 当前二值完整覆盖下两个 pair 均负；若要认可进展，需单独建“部分执行”定义。 |

8 条中，1 条暴露上游时间属性和缺失 pair 问题，3 条集中暴露多 Action 联合覆盖或原子化问题，2 条涉及“部分执行”和“完整覆盖”口径，1 条明显假阳性，1 条存在搜索与定向抓取的等价性争议。上述分类是审计归因，不是新的 Gold 数量。

额外核查：200 个窗口有 120 个 Intent 未配 pair，它们在上游均非 `current`，所以构建器按现有规则跳过。GQQ 却在其中 4 个 Intent（分布于 3 个窗口）判为 due-now；逐条复核后确认 3 个应改为 `current` 并补 pair，另 1 个属于“先写 CSV、再计算”的后续目标，维持 `future`。修订版本见 `docs/2026-10-13-ta-train-v2.md`。不能把缺失 pair 当作负例。

## 正负 pair 分层抽样

从其余 170 个未作整步人工复核的窗口，按 GQQ 正/负、工具（exec/read/web_fetch/write）、单/多 Action 窗口分层；每层按 `SHA256(pair_id)` 排序取首条，共 16 条。下表的“确认”仅指本次文本复核与 GQQ 一致；仍属于小样本质量检查。

| GQQ / 工具 / 窗口 | pair_id | 复核 | 依据 |
| --- | --- | --- | --- |
| 负 / exec / 多 | `397fd317dadc5ac80c3940af` | 确认负 | 查找 bingo 可执行文件，与读取 `input/spec.yaml` 不同。 |
| 负 / exec / 单 | `e26641ad6650103a0b86a2cd` | 确认负 | `mkdir -p output` 仅准备目录，未生成三个产物。 |
| 负 / read / 多 | `160b53803b6680f1eaea75ca` | 确认负 | 读取 `company_story.md`，目标是 `context.json`。 |
| 负 / read / 单 | `0eb044c5e656041939904ac9` | **待裁定** | Intent 是检查 `run_fetch.sh` 的路径，Action 正好读取该脚本。GQQ 认为“读不等于验证”，但读取脚本可能就是核查操作；需确定要求的是发起核查还是证明路径正确。 |
| 负 / web_fetch / 多 | `ad88800e3eb7a9362bed5e2f` | 确认负 | 抓取全国目录页，不是指定 OTP 目录页。 |
| 负 / web_fetch / 单 | `f9497b959bac2c7fdb6c792c` | 确认负 | 打开一个 Wikimedia 文件页，不是搜索公有领域图片。 |
| 负 / write / 多 | `636a4ba776605c44dd9006d6` | 确认负 | 写入 handoff YAML，目标是 `output/TEAM.md`。 |
| 负 / write / 单 | `bc2389ddb8958559c8bd156f` | **待裁定** | 已写入目标 recap；GQQ 的理由一面承认约 150 词落在 120–200 范围，一面又以“未显式验证词数”否定。需核对两条引用是否来自所要求的提取结果。 |
| 正 / exec / 多 | `bd0c5a53b6eb1e134362f911` | 确认正 | 搜索 query 明确涵盖 EU AI Act 系统性风险指定标准。 |
| 正 / exec / 单 | `196d32c2802970da54fc90b6` | 确认正 | `ls -la` 工作目录可查看 notes 文件状态。 |
| 正 / read / 多 | `f4a523c9dba059df2f55c798` | 确认正 | 精确读取指定政策文件。 |
| 正 / read / 单 | `6f857c751ec4f8a110cbbcbf` | 确认正 | 精确读取 `input/pipeline.yaml`。 |
| 正 / web_fetch / 多 | `12dace213eb0678a54018df0` | 确认正 | 抓取指定球队的 fixtures 页面。 |
| 正 / web_fetch / 单 | `164e0faf66abe55625467141` | 确认正 | 导航到 NUCC 1500 Claim Form 页面；返回 404 属执行结果，不改变调用意图。 |
| 正 / write / 多 | `6ef7c5baba985ebce305c22f` | 确认正 | 写入目标 `output/summary.json`。 |
| 正 / write / 单 | `5b9140356f55b05e85721166` | 确认正 | 写入目标 `meeting_candidates.csv` 且含三条候选时段。 |

这 16 条中有 14 条与 GQQ 一致、2 条待裁定；它们不是总体准确率估计，因为分层抽样和相邻 pair 存在相关性。负例中的 `read` 核查语义、`write` 的细约束是下一批抽查重点。

## 入训前处理

1. 先复核 GQQ 判为 due-now 但上游标为 `future` 的 4 个 Intent；对确认应立即执行者修正时间属性，重建相关窗口的全部 pair 并重审。保留原文件作为审计记录。
2. 对上述 `complex_joint_coverage`、部分执行和待裁定样本建隔离清单，不将其现有负 pair 直接用于训练。
3. 新增“列出文件 ≠ 读取文件”“读取脚本核查 ≠ 运行脚本”的操作约束检查；对 write 的长度、引用等细约束用可验证证据再裁定。
4. 扩大 train 人工 pair 抽检，并从冻结的 dev/test 轨迹建立独立 Gold 后再给出泛化指标。
