# 清晰 test 的 exec 错排抽查（2026-10-09）

## 口径

以 BGE v2 清晰 test 子集和 2026-10-08 修正过相同 Action 文本 ID 的 qrels 为准。Top-1 仍失败 369/804 条。其中已知正例为 `exec`、首位也为 `exec` 的有 123 条；已知正例为 `exec`、首位为 `web_fetch` 的有 45 条。

对两组分别用固定种子 `20261009`，按原已知正例排名 2–5、6–10、11–100、100 名后分层抽取 6/4/6/6 条，各 22 条。查看 Intent、正例和首位 Action 的完整参数、工具、目标及来源轨迹。以下是**44 条失败定向样本**的人工初审，不代表两组总体错误率。`待复核`只表示当前证据不足，不自动改 qrels。

## 抽查汇总

| 首位/正例 | 明确不满足 Intent | 字面上可能等价 | 待核任务或结果 | 原正例本身有明显问题 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exec` / `exec` | 15 | 4 | 3 | 0 | 22 |
| `web_fetch` / `exec` | 14 | 3 | 4 | 1 | 22 |

### exec → exec（22 条）

| query_id | 判定 | 关键依据 |
| --- | --- | --- |
| `6b5c299481fbca1b4f3db5d3` | 可能等价 | 两者均运行 `python3 tool/agenda_tool.py`；首位多一条成功提示。 |
| `d68962c20537fcfb0137413e` | 待复核 | 同 URL、输出 PDF；`curl -v` 与 `-w` 输出不同诊断信息，是否满足“more verbose”需按任务要求判定。 |
| `7615de7e6985faf247ec0a1b` | 明确错误 | Intent 要跟随重定向；首位 `curl -s` 缺 `-L`。 |
| `2399e41d102fe5618a0b76ff` | 明确错误 | Intent 要检查 BeautifulSoup 是否可用；首位执行安装并查看 Python 路径。 |
| `5ee934a8c519c7e304dc81f4` | 可能等价 | 两者均运行 `generate_notes.py` 捕获输出，调用形式不同。 |
| `1a5f6d1513129756ced0e2f1` | 可能等价 | 同 URL、同保存路径；首位仅增加 User-Agent。 |
| `a1f6eb13790ac186442edc5e` | 待复核 | 两者批量抓取 Adele 等页面；首位循环生成的文件名和实际结果需核对，单凭命令主题不能补正例。 |
| `c6dc53c9d7b7ba57ab8b6ed3` | 明确错误 | 要计算本地 zip 的 SHA512；首位只解析官方 `.sha512`。 |
| `f7d19fbe4d642551f4580227` | 明确错误 | 首位运行另一轨迹的 `validate.py`，不是本任务的 `validate_replies.py`。 |
| `baf34a8a3697cc892cedd514` | 明确错误 | 要生成 initial report；首位指定了 fixed report 输出。 |
| `8768b66751cd97615cc43e20` | 明确错误 | 要用 `-L` 检查 WADA 下载的最终 URL；首位检查 Wikimedia 的重定向。 |
| `2b2fdbe333bbdaa111e804fa` | 明确错误 | 要为各 article 重新下载 pageviews；首位只测试少量 URL 模式。 |
| `1a832a6bb2310199440bc9e5` | 明确错误 | 要列出本地 `data/`；首位抓取远端 USADA 页面。 |
| `f9a7548df8057b470e04325e` | 明确错误 | 同为 `curl -L`，但 URL 属于无关的 FDA 页面，不是 Wikimedia API。 |
| `3ae4626fd9cb543e0b288076` | 明确错误 | 要检查可用工具；首位抓取 FDA 网页。 |
| `a9549a83af0f5f526f038a02` | 可能等价 | 两者尝试 archive.org 上同主题的不同 URL；Intent 本身允许“different URLs”。 |
| `b065ed60320e7230d6a0ada1` | 明确错误 | 要改用 `python3` 运行具体脚本；首位只打印 `python3 --version`。 |
| `ac4570c56eab1db6febc3421` | 明确错误 | 要用下划线重试 article 下载；首位测试 URL 模式，没有批量下载。 |
| `d9d13658c8d4ecac350903a7` | 明确错误 | 要检查现有目录结构；首位创建目录。 |
| `b9db17ebb6053dd6189c1521` | 明确错误 | 要运行 `generate_notes.py`；首位运行另一脚本。 |
| `dc39157fdfc6e7b1d7431817` | 明确错误 | 要创建 `outputs`；首位创建 `output`（少一个 s）并执行其他任务。 |
| `bbf8d43037b4d83a1cdcf239` | 待复核 | 要搜索 pageviews API 路径；首位查看 API 根目录并 grep，可能是合理探索步骤，但能否给出具体路径未知。 |

### exec → web_fetch（22 条）

| query_id | 判定 | 关键依据 |
| --- | --- | --- |
| `fca03487521022e433d52493` | 明确错误 | Intent 指定 Serper 搜索；首位直接抓 MSHA 页面。 |
| `2ee2ab8fb510c19ca6a38c0d` | 明确错误 | 要用 `curl` 下载指定 GitHub URL 到本地；首位抓另一域名且不保存原始 HTML。 |
| `defd7eb52d5fd89eb3bf8d7b` | 明确错误 | Intent 指定 Serper 搜索 `nba.com`；首位直接抓球员页。 |
| `9ced01ee4ef2b6c75564e699` | 明确错误 | 要检查本地下载文件是否含 `satnam`；首位抓远端页面。 |
| `9355a60dfd5353cd4cb9d483` | 明确错误 | 要用 `curl` 下载指定 PDF；首位是另一份封面作物 PDF，且未保存。 |
| `3a1dea33e85f252943c367c7` | 明确错误 | 要带 User-Agent 抓首页并落盘；首位 `web_fetch` 没有该参数和保存动作。 |
| `7113bf98f6ac3affb4da5f62` | 明确错误 | 要搜索 Cleveland 组织；首位只打开一个组织主页。 |
| `5c8c5ab3e3bfa47accddad16` | 可能等价 | 要找到 FDA Safety Communications 页 URL；首位直接打开目标 URL，正例则用 Python 打印 URL 和标题。 |
| `23c8f49b45ae021e0ae1f582` | 明确错误 | 指定 Zenodo URL；首位访问 GitHub。 |
| `3cccb1a5f72c08fdebde60c7` | 明确错误 | 要尝试 legacy MediaWiki Analytics API 路径；首位只开 API 根路径。 |
| `7494fe861090d6f85a16716b` | 待复核 | Intent 允许“Serper 或其他搜索方法”；首位打开 NRC VAD 的 GitHub 候选页，需核实是否正确来源。 |
| `b3082ea133ac33f58762d3fc` | 明确错误 | 要按顺序尝试 data.worldbank.org 的首页；首位抓 UNICEF。 |
| `2fcc2be549e4b8f617a3af9d` | 待复核 | 同为 MSHA 传送带法规主题，但 URL 不同；需看实际页面内容是否满足 intent。 |
| `1f9bf8379e3f986c6d159122` | 可能等价 | 要寻找 MSHA 传送带安全页的正确 URL；首位访问同主题另一 URL。 |
| `1a8fed1ffbb182824dffc2ba` | 明确错误 | 要找 FDA 页面新的位置或 URL 结构；首位只访问先前的原 URL。 |
| `81eb7a5d655e09c16e5ea270` | 明确错误 | 要找缓存或替代版；首位抓原页面。 |
| `5452d37f14906648f8b8daa8` | 待复核 | 首位 Apple URL 与正例不同；须核实该页面是否也是 TLS 1.0/1.1 弃用公告。 |
| `66671570aac2313dcd49b3e4` | 明确错误 | 要查看已下载 legacy API 数据格式；首位访问在线 API 文档。 |
| `91e6a5fe46ab8df614078904` | 原标注有问题 | Intent 指定 `web_fetch` 且保存原始 HTML；首位确实用 `web_fetch` 抓同 URL，但不保存；正例是 `exec curl | head`，也未保存。两者均不能完整满足字面 Intent。 |
| `32b8fb591767b6368b847cbd` | 待复核 | 要核查 `source2` 页中诗歌标题；首位抓作者主页，是否同一页面或含目标标题需核对。 |
| `54c8f368f6ebf7a7d147b9ec` | 可能等价 | 要搜索 NRC VAD Lexicon 的位置；首位是候选 GitHub 页面，正例访问 NRC 研究出版物页；需确认 GitHub 内容真实相关。 |
| `d5fa2d65eb2642a96c0417b2` | 明确错误 | 要搜索组织名称中的特定兴趣词；首位抓无关 arXiv 检索页。 |

## 判断与后续

1. 这两组的主体确为模型错误，尤其是命令目的、URL、保存路径和指定工具不匹配。`exec→web_fetch` 不应因都能联网而视为等价：Serper 搜索、指定 `curl`、本地落盘和已有文件验证都有额外操作要求。
2. 抽样中有 7 条字面上可能等价，另 7 条需要结果或原任务上下文核查。这些是 **Silver 漏标候选**，暂不自动加入 qrels。若确认等价，应同时修订 train/dev/test 的等价标注规则，并在固定评测集上重新计分。
3. `91e6a5fe46ab8df614078904` 应优先复核原 Intent–Action 关系；原正例未保存文件，不能以它为标准把首位直接标负。这个问题属于标签质量，不是普通工具混淆。
4. 对明确错误，可在 train 内加入“同命令错参数/错 URL”和“抓网页 vs 搜索/下载/验证”的高质量难负例。不得用 test 样例回流训练。
