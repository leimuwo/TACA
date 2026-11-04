# TACA 流水线：Agent 接手指南

更新：2026-11-04。接手目标是在另一台 GPU 上实际运行已有模型的端到端 TA 流水线，确认技术可用性并记录错误；暂不启动训练、全量教师重标或修改冻结标注规则。

## 1. 入口与当前完成状态

- GitHub：<https://github.com/leimuwo/TACA>，分支 `main`。
- 部署目录：`experiments/ta_pipeline/`，是独立运行包；不需要原机器的数据、实验目录或绝对路径。
- 首个部署提交：`b5465f65d15500db73191c9027073c6c8a132461`。本指南在后续提交补充，以最新 `main` 为接手入口。
- 已完成：部署代码上传、两个私有 ModelScope 模型上传、远端大小与 SHA-256 核验、7 项独立逻辑测试、Shell 语法和 checkpoint 文件存在性检查。
- 未完成：本版真实 GPU 加载/前向、显存与延迟测量、新规则语义质量评估。不要将 `check` 或单元测试通过写成 GPU 推理通过。

链路：

```text
Thought → Qwen3-0.6B → Current/Future Intent
Current × 当步固定窗口全部 Action → 微调 BGE Reranker → Intent / Step 聚合
```

当前不经过 Retriever，也不执行 Action 中的命令。M5 Retriever/Reranker 训练交付尚未完成。

## 2. 模型与凭据

| 角色 | ModelScope ID | 本地默认目录 |
| --- | --- | --- |
| Intent 提取 | `MongTsai/TACA-Qwen3-0.6B-Intent-v1.2` | `experiments/ta_pipeline/models/intent` |
| 旧微调 Reranker | `MongTsai/TACA-BGE-Reranker-TA-Legacy-v3` | `experiments/ta_pipeline/models/reranker` |

Intent 是 v1.2 compact 全参训练的最佳 `checkpoint-500`。Reranker 基于 BGE-reranker-v2-m3，用7,243对旧GQQ弱标签微调，输出单logit。

模型为私有，需要对上述仓库有权限的账号令牌。通过环境变量 `MODELSCOPE_API_TOKEN` 或交互式不回显方式注入；不要将令牌写入代码、Git、日志或报告。没有凭据/权限时先完成本地逻辑检查，再向用户说明缺少访问权限；不要自行将模型改为公开。

`models.lock.json` 固定推理文件大小及 SHA-256；远端 master 更新后若校验不符，应查明版本，不修改锁文件以绕过检查。两模型推理文件合计约4.7GB，另为下载缓存、Python环境和输出预留空间；不需要 optimizer.pt 等训练状态。

## 3. 在 GPU 上按顺序运行

若已有仓库，先检查 `git status`；保留本地改动，不强制重置。若无仓库：

```bash
git clone https://github.com/leimuwo/TACA.git
cd TACA
```

从仓库根目录进入部署目录：

```bash
cd experiments/ta_pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 已有可用 PyTorch 环境时可跳过安装 torch；下列 wheel 需要兼容的 NVIDIA 驱动
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
python -m unittest discover -s . -p test_pipeline.py
```

不需要安装 LLaMA-Factory、peft 或 torchaudio。若 GPU 不可用，检查驱动与 PyTorch wheel；不要假装已在 GPU 执行。

在 Bash 中注入令牌并下载：

```bash
read -rsp 'ModelScope token: ' MODELSCOPE_API_TOKEN; echo
export MODELSCOPE_API_TOKEN
python download_models.py
python download_models.py --verify-only
bash run.sh check
CUDA_VISIBLE_DEVICES=0 bash run.sh run
```

这会先跑两个示例，输出 `runs/demo/`。只有两个 checkpoint 实际加载且预测写出后，才算完成首次真实推理。若技术失败，先看 `predictions.jsonl` 的 status/error；不要只看进程退出码或文件存在。

运行用户数据：

```bash
INPUT_FILE=/absolute/path/windows.jsonl \
OUTPUT_DIR=/absolute/path/new_run_directory \
CUDA_VISIBLE_DEVICES=0 bash run.sh run
```

可追加 `--batch-size 1` 降低显存需求；修改参数必须换输出目录。指定 Python 用 `PYTHON_BIN=/path/to/python`。替换模型用 `--intent-model /path --reranker /path`，并记录来源及版本；不要将未经校验的其他模型冒充默认模型。

## 4. 数据契约

一行一个固定窗口：

```json
{"unit_id":"U001","thought":"Let me read config.yaml","window_complete":true,"actions":[{"action_id":"A1","tool":"read","arguments":{"path":"config.yaml"}}]}
```

- `unit_id` 全输入唯一，Action ID 窗口内唯一；Thought 是字符串，actions 是列表。
- 必须显式声明真实的 `window_complete`，不能把未知/截断窗口伪装成完整。
- Action 可用 tool/arguments、function/args、raw_call 包装或 OpenAI function 对象。arguments 字符串须为合法 JSON 对象。保留完整命令/参数；不把工具返回当 Action。
- 原始 CTB/ClawGym 轨迹不能直接当上述输入；若需转换，另写可审计适配器，保持 Thought 与其实际当步 Action 的边界，保留来源 ID，不臆造缺失步骤。
- 只用 Thought 提取连续原文 Intent，保留 current/future。不从 Action 反推或修正 Intent。

## 5. 当前规则与已知模型偏差

当前协议 `operation-pair-v2`：独立输入单个 Intent 引文和单个完整 Action，判断操作类型是否相符。取消目标指派、前置支持及目标分类损失。

- 修改配置 vs 修改脚本：同为修改，可为 yes，不要求目标相同。
- 修改文件 vs 仅读取文件：no，准备操作不等于实际修改。
- 搜索代码 vs shell grep：可为 yes，按实际操作，不按工具名。
- 将测试代码写入文件不等于执行测试；复合操作不清楚时保留 uncertain，不能拍脑袋补规则。
- 多个 Intent 可以匹配同一 Action，多个 Action 也可以匹配同一 Intent。Future 不参与当前匹配/聚合。

**旧 Reranker 仍学过目标/内容覆盖与前置支持口径。外部新聚合不能修正它的语义偏差。** 所有输出带 `legacy_reranker_experimental_not_gold` 和 `semantic_acceptance_passed:false`；不要删除标识或把模型候选升级为人工 Gold。旧 GLM 弱标签250对F1=94.01%不是本版新规则准确率。旧模型还见过部分当前评估轨迹，不能直接以其成绩宣称独立泛化。

## 6. 聚合、长度与异常

- 每个 Current 任一 yes → 意图 yes；完整非空窗口全no → 意图 no；其余 uncertain。
- Step 全yes/全no/已有yes和no → fully_consistent/fully_inconsistent/partially_consistent；其余 uncertain；无Current → not_applicable。无Action保守为uncertain。
- 这些标签仅代表操作匹配，不代表文件目标正确、工具成功或任务完成。
- 默认 logit 阈值0、灰区宽度0沿用旧设置，尚未按新规则校准；sigmoid不是可靠正确概率。不用test选择阈值。
- 输入不静默截断。≤4096动态padding小批，4096–8192单对；超过上限或OOM为uncertain。旧接口默认512之外的样本会标记；支持位置长度不等于长文本质量已通过。
- 提取JSON错误、非原文引文、重复/顺序异常、生成截断等为技术失败/uncertain，不当作空Intent。无效输入直接报错，不删行凑成功率。
- 不自动调用0731或其他API作为回退，不自动执行工具，不上传用户输入数据。

## 7. 输出、续跑及验收交付

三个输出：`manifest.json`、`predictions.jsonl`、`summary.json`。manifest保存输入/Prompt/主脚本哈希、参数及模型文件指纹；模型内容完整性另由下载锁校验。

同配置、同输入续跑自动跳过已保存单元，技术失败也会跳过。重跑失败项应从原输入按 unit_id 生成新输入，用新输出目录执行，保留原失败。出现半行JSON先备份再修复，禁止悄悄丢失记录。不要覆盖原输出或悄悄混合不同阈值/模型结果。

接手 Agent 应交付一份短报告，至少记录：

1. Git commit、GPU型号、驱动、Python/PyTorch/Transformers版本，实际命令及模型锁验证结果。
2. 输入窗口数、Current/Future数、待匹配pair数、各status/Step分布、失败原因和未决pair；若有uncertain，应解释来源而非强改no。
3. checkpoint加载与真实推理是否成功、运行耗时；测量显存/延迟时注明样本长度、批量和是否包含加载时间。
4. 输出绝对路径和可复现命令；区分技术跑通与语义验收。没有人工参考标签时只报告分布/案例，不报告准确率。

优先完成示例GPU smoke，再处理小批真实开发窗口。不擅自改Prompt、重标全量数据、训练模型、改变公开性或扩大发布范围。若要修复兼容问题，保留可复现日志、单独提交修改，并使用新输出目录重新验收。
