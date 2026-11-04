# 20窗口协议 smoke 测试集

先用这份小集验证异地GPU流水线，不直接跑全量训练数据。20窗口、21参考pair（13yes/7no/1uncertain），Step覆盖9完全一致、5完全不一致、1部分一致、3不确定、2不适用。

这是编码Agent人工编写的受控案例及期望标签，不是真实轨迹抽样、独立人工Gold或泛化Benchmark。参考Intent按原文边界填写，未由模型填充。参考与输入分文件，推理脚本只读取 `windows.jsonl`，不把参考标签传入模型。

覆盖：读取、搜索、修改、删除、执行测试；同操作不同文件/URL、读取不等于修改、mkdir不等于测试、写入/打印测试命令不等于执行、Future、单Action多Intent、多Action单Intent、复合命令、部分一致、不透明工具、无Action、无Intent和不完整窗口。标注以operation-pair-v2为准；旧模型语义偏差应如实记录。

从部署目录运行：

```bash
INPUT_FILE="$PWD/testset/windows.jsonl" \
OUTPUT_DIR="$PWD/runs/smoke20_v1" \
CUDA_VISIBLE_DEVICES=0 bash run.sh run
python evaluate_smoke.py \
  --predictions runs/smoke20_v1/predictions.jsonl \
  --output runs/smoke20_v1/evaluation.json
```

参考标签不是硬性要求模型逐字复现的唯一语义表达；不同合法引文会影响严格提取和pair对齐，需检查逐例。评测按引文+时序+Action ID对齐，不按局部I1/I2盲对齐。报告提取整窗exact、Step一致率、pair各类P/R/F1及缺失/额外pair；缺失/提取失败不从分母删除。指标是对本开发smoke期望的符合程度，不是独立测试准确率。

先看technical_failure、OOM、超长等技术问题，再区分Intent边界/时序错误与Reranker操作误判。该集不覆盖真实长Thought、8192长尾、跨Agent分布、实际工具返回；通过也不能宣称生产验收。后续另建真实开发小集，保留原始窗口与任务来源，按新口径审核；不沿用旧目标指派/前置支持标签，不将当前20条混入训练后继续称测试。
