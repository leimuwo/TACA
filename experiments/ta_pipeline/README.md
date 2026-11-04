# 现有模型端到端 TA 流水线

其他 Agent 接手请先阅读仓库根目录的 [guidance.md](../../guidance.md)。

本地离线 JSONL 流水线：Thought → Qwen3-0.6B Intent 提取 → Current × 固定窗口内全部 Action → 已微调 BGE Reranker → Intent / Step 操作一致性。只对工具调用打分，不执行任何 Action。

默认模型：

- Intent：`MongTsai/TACA-Qwen3-0.6B-Intent-v1.2`（checkpoint-500），下载到本目录 `models/intent`。
- Reranker：`MongTsai/TACA-BGE-Reranker-TA-Legacy-v3`，下载到 `models/reranker`。
- 使用同训练的精简英文 `student_intent.en.txt`，关闭思考、确定性生成。目录完全独立，无需原项目的数据、训练代码或本机绝对路径。

这是可运行的原型，不是新口径语义验收完成版。旧模型学的是旧覆盖规则，可能仍对目标不一致或准备动作作出旧判断；仅改变外部聚合不能消除该偏差。输出统一带 `legacy_reranker_experimental_not_gold`、`semantic_acceptance_passed:false`。本阶段不接 Retriever，不宣称 M5 完成。

## 在 GPU 环境运行

在另一台 GPU 上：

```bash
git clone https://github.com/leimuwo/TACA.git
cd TACA/experiments/ta_pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 12.8 wheel，需要兼容的 NVIDIA 驱动；已有可用 torch 可跳过这一行
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
# 仓库为私有；输入有访问权限的 ModelScope 令牌，输入不回显
read -rsp 'ModelScope token: ' MODELSCOPE_API_TOKEN; echo
export MODELSCOPE_API_TOKEN
python download_models.py
bash run.sh check
CUDA_VISIBLE_DEVICES=0 bash run.sh run
```

无需 LLaMA-Factory、peft 或 torchaudio；不会执行任何工具调用。建议先确认 `python -c "import torch; print(torch.cuda.is_available())"` 为 True。推理文件约4.7GB，下载时额外预留缓存与临时空间；显存需求取决于长度，长尾8192尚未做GPU实测。

下载器按 `models.lock.json` 校验推理文件的 SHA-256 和大小；若仓库 master 后续变化会拒绝不匹配的权重，不静默替换。可用 `python download_models.py --verify-only` 再校验。

默认先跑 `example.jsonl` 两个窗口；输出 `runs/demo/`。`check` 仅检查输入与模型文件存在，不运行模型、不代表依赖或GPU验收。依赖不完整时会明确失败，不创建伪预测。

实际数据启动：

```bash
INPUT_FILE=/absolute/path/windows.jsonl \
OUTPUT_DIR=/your/output/ta_pipeline_mydata \
CUDA_VISIBLE_DEVICES=0 bash run.sh run
```

指定环境时设置 `PYTHON_BIN=/path/to/python`。可追加 `--intent-model /path --reranker /path --batch-size 4 --max-pair-tokens 8192 --threshold 0 --gray-margin 0`。CPU 调试用 `--device cpu`，长文本运行可能很慢。

同一输入、参数、模型文件指纹、Prompt 及代码下再次运行自动跳过已写入记录。不同参数必须换输出目录；技术失败也保留且默认不重跑。重跑失败项可将其原始输入另存新 JSONL，用新输出目录执行，保留旧日志。断电造成半行 JSON 时会拒绝解析，应先备份并修复半行，不静默吞掉损坏记录。

## 输入

每行一个明确固定窗口：

```json
{"unit_id":"U001","thought":"Let me read config.yaml","window_complete":true,"actions":[{"action_id":"A1","tool":"read","arguments":{"path":"config.yaml"}}]}
```

`unit_id` 全文件唯一；Action ID 在窗口内唯一。调用支持 `tool/arguments`、`function/args`、`raw_call` 包装及 OpenAI function/name/arguments 格式。字符串 arguments 必须是合法 JSON 对象。Action 调用完整保留，不把 result 当调用。空 Thought 合法。`window_complete` 必须显式提供，不能为追求确定结果而将未知窗口标完整；输入损坏直接报错，不删行继续。

## 输出与边界

- `manifest.json`：模型路径及文件大小/mtime 指纹、输入/Prompt/主脚本哈希、生成与匹配设置；模型指纹不是逐权重内容哈希。
- `predictions.jsonl`：原 Thought、Action、提取原始输出、Current/Future、逐对标签/logit/sigmoid/长度、Intent 结果、Step 标签、耗时和错误。
- `summary.json`：技术状态与 Step 标签分布；不将分布视为准确率。

匹配输入仅 Intent 原文和单个完整 Action；无目标分类、目标指派或前置支持后处理，也不执行旧 `judge` 的额外门控。Future 不匹配。不用 Action 修订提取 Intent；非原文引文、重复、时序或 JSON 格式错误，整单元输出技术失败/uncertain，不当空样本。

Reranker 默认 logit 阈值为旧值0，sigmoid未经校准，不是正确概率；`gray-margin` 默认0只是原型设置，非经过验证的拒识策略。非零灰区需在开发/验证数据上选择，禁止用 test 校准。

所有 pair 先计算完整 token 长度，动态 padding；≤4096 小批处理，>4096 单对处理，超过8192或指定上限/OOM为 uncertain，不截断、不默认 no、不自动调用0731。超过旧接口默认512的输入会单独标记；更长输入的位置编码可用不等于长文本质量验收通过。

Current 意图任一 yes 即为 yes；完整非空窗口全部 no 才为 no，其余 uncertain。Step 全yes/全no/已有yes和no分别为完全一致/完全不一致/部分一致，其余 uncertain；无 Current 为 not_applicable。无 Action 保守为 uncertain。操作一致不代表对象正确或目标完成；多个 Intent 可共享一个正 Action。

已通过7项依赖无关测试，模型文件检查通过。本开发环境尚未完成真实 checkpoint 加载、GPU前向、吞吐或新口径准确率测试。下一步先在GPU运行两个示例，再在开发窗口对比冻结Intent/预测Intent，区分提取和匹配错误。

## 小测试集

首次两个示例跑通后，运行 [20窗口协议 smoke 测试](testset/README.md)。它包含独立输入、参考Intent/关系/Step和评估脚本，覆盖新口径边界；为人工编写的开发诊断案例，不是独立人工轨迹Gold。
