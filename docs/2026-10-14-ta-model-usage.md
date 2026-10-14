# TA 覆盖模型使用入口

模型：`outputs/ta_pair_cross_encoder_10k_v3/best`。入口：`scripts/use_ta_coverage.py`。这是可调用的判断组件，尚未连接到某个具体 Agent 的执行循环；支持离线 JSONL 和本地 HTTP 两种调用方式。HTTP 是单进程顺序处理，适用于第一阶段本地接入。

## 在 GPU 环境启动

从 Project 根目录运行，与训练使用同一 Python 环境：

```bash
python scripts/use_ta_coverage.py --serve --device cuda --port 8765
```

加载一次 checkpoint 后常驻。`GET /health` 检查就绪，`POST /judge` 接收一个窗口。

```bash
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/judge \
  -H 'Content-Type: application/json' \
  --data-binary @configs/ta_coverage_example.jsonl
```

批量推理：

```bash
python scripts/use_ta_coverage.py \
  --input configs/ta_coverage_example.jsonl \
  --output outputs/ta_usage_example/predictions.jsonl \
  --device cuda
```

默认 checkpoint 可通过 `--checkpoint` 指定。默认 logit 阈值 0，与已报告评测一致；也可通过 `--threshold` 使用 dev 确定的阈值。batch-size 16，max-length 512；Action 序列化、关键字段前置和截断方式与 pair 评测一致。

## 接入流程与输入协议

上游从当前可观察 Thought 和此前上下文提取原子 Intent，确定 `temporal_scope`（current/future/conditional）及 `assessable`。将当步拟调用或已调用的 Action 传入接口，读取输出；可作为旁路记录或人工复核信号。接口本身不执行工具，不修改 Agent 的调用，也不自动阻断。Thought 字段只作为调用方记录，模型实际输入为 Intent 与 Action。

每个 Intent 必须包含非空 `text`、显式时间属性；只有 `current` 参与判断。上游确认可评测时填 `assessable:true`；未确认、不可评测或批量阅读多文件的当前目标使整步输出 `unassessable`，不删目标后聚合。接口只对明确字符串与范围做验证，无法自动保证上游 Intent 清晰或完整。

Action 支持 `{action_id,function,args}` 或 `{action_id,raw_call:{function,args,...}}`，与 ClawGym 调用结构一致。OpenAI 风格 `{function:{name,arguments}}` 应在调用方转换为函数名字符串与参数对象。ID 可省略，自动使用 I1/A1 等；显式 ID 不得重复。不要把工具返回作为 Action；执行成功与否不在当前判断范围。

输出包括 `label`、`pairs`（Intent/Action ID、logit、coverage_score、covered）、`intent_coverage`（覆盖的 Action ID）、`excluded_intents` 与阈值。coverage_score 为 logit 的 sigmoid，未经概率校准，不能解释为可靠的真实正确概率。完整覆盖全部目标为 faithful，覆盖部分为 partial，零覆盖为 not_executed；无当前目标为 unassessable，有明确当前目标但无调用为 not_executed。额外违背性动作及不可拆分的联合覆盖尚不支持。

## 验证状态

当前 CPU 工作环境没有 torch/transformers，未执行真实 checkpoint 推理或 GPU 服务启动。五项推理逻辑测试覆盖多目标聚合、未来目标、不可评测、批量阅读、重复 ID 和异常分数；GPU 命令用于在已有训练环境验收。test GLM pair 指标 ACC 0.9480 / PRE 0.9444 / REC 0.9358 / F1 0.9401，不直接代表在线整步准确率。
