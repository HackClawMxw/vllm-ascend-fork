# 发布机分档并发探测

主力机提供 `tools/flashmla_concurrency_probe.py`，发布机按当地 skill 拉起服务后执行。只依赖 Python 标准库，不启动或修改服务，不安装算子，不自动拉取代码。

## 固定版本和分工

早期小流量候选为 `698d00e7c` / `bc83ce0ed`，诊断版为 `f78fe02e8`。团队审查已修复 eager 不同形状永久占用缓冲的问题；放大并发前切到修复候选 `ae92cb43e971f9e7586855b77658101f1f248acd` 并先复测小流量，不能在一次测试中途无记录切换。客户端脚本可单独拷贝到发布机，不要求为运行脚本更换服务版本。

发布机先记录服务实际使用的 checkout SHA、镜像 ID、包/schema/import 路径、启动命令和当地 skill 版本；`--candidate-sha` 只是填写报告，客户端无法验证远端服务版本。随后按当地流程验证 eager，再验证图，最后 DSpark，每个配置使用独立输出目录。主力机根据证据分析、改代码和提交修复；发布机继续使用其已有启动/部署流程复现，环境修复可在发布机就地处理；必要的运行代码修复可在发布机独立分支验证，再回主力机整合，避免两机同时改共享开发分支。

## 使用

以下命令在服务已启动之后运行。把 URL、模型名和 prompt 文件换成发布机已有配置；API root 包含 `/v1`。首轮采用能产生足够 Decode 的已知 prompt。不要把生成长度固定值当作实际生成 token 数，模型可能提前结束。

```bash
python tools/flashmla_concurrency_probe.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL \
  --candidate-sha ae92cb43e971f9e7586855b77658101f1f248acd \
  --prompt-file /path/to/known-prompt.txt \
  --concurrency 1,4,16,32 \
  --requests-per-stage 64 \
  --max-tokens 128 \
  --timeout 120 \
  --output-dir /path/to/evidence/eager-run-001
```

需要 chat 接口时增加 `--api chat/completions`。认证可通过已有 `OPENAI_API_KEY` 提供，或通过 `--api-key-env` 指定其他环境变量名；不要把 token 放在 URL。每档请求数至少等于最大并发档位。线程池上限是客户端同时在途请求数，不保证服务形成同等 batch size，也不能据此证明设备访存并发正确。

默认请求使用同一个 prompt；长短输入混合可用 `--prompts-json /path/to/prompts.json` 替代 `--prompt-file`，JSON 内容为非空字符串数组，按请求编号轮转。日志只记录 `prompt_index`，不记录内容。请求采用非流式、`temperature=0`、`n=1`，没有自动重试或重定向。重复 prompt 可能命中 prefix cache；即便轮转长短输入，也不是随机长度、请求取消/churn 或高负载的完整覆盖。需要定位长度问题时，对已有长/短 prompt 分别另开一次运行，并记录服务的 prefix cache、chunked prefill 和 scheduler 配置。

## 输出与失败现场

- `requests.jsonl`：run 配置；每个请求的 run/request ID、UTC 开始/结束、HTTP 状态、异常类型、耗时、finish reason、是否有文本及基本 usage；每档统计。每条完成后立即 flush。
- `summary.json`：各档统计、最后完成请求 ID/时间、报告的服务 SHA。每档结束更新；整次正常退出再写结束时间。
- 控制台只输出每档统计。输出目录必须不存在，防止覆盖上一轮证据。

请求携带 `X-Request-ID` 供相关联日志使用，但服务是否读取/打印该头需要发布机确认；若没有，可用 UTC 时间窗和进程/rank 日志关联。脚本默认不落 prompt、生成文本、完整响应、HTTP 错误 body 或认证值，异常只落类型，避免异常消息泄露请求内容。usage 只保留非负整数 token 计数。

遇到 HTTP/网络/响应结构错误会完成当前已提交档位，保存现场并停止升档，返回退出码 1。合法 terminal finish reason 的空文本（如立即 EOS、生成预算结束或 content filter）单独记为 `empty_terminal_response`，不伪报协议失败，也不算已证明 Decode 覆盖。工具调用必须带有对应输出。未知 finish reason、缺失 choices、无效 JSON 等视为结构失败。退出码 0 仅说明这些基本结构检查通过。

`--timeout` 是 socket 超时，**不是整个运行的硬性截止时间**；服务持续缓慢发送内容或等待中的多批请求可能使总耗时更长。请求完成前没有完成行；卡住时先保存现有 JSONL、服务日志和各 rank 堆栈，再按发布机流程结束本次运行。中断时 summary 可能只有上一档结果，JSONL 是已经完成请求的主要记录。

## 配合路径、图和 tiling 证据

脚本不证明 FlashMLA 或 tiling 下沉命中。服务验证仍需 [诊断手册](flashmla_diagnostics.md) 的 FIA Prefill / Flash Decode / metadata 标记。首次标记不证明每次 graph replay；图测试必须结合真实 profiler 事件、metadata refresh 与消费者事件顺序，以及相同档位改变 lengths/block table 后的结果。

并发增加后若失败，发布机保留失败档位与最后完成请求时间、所有 rank 日志、错误前后 trace，必要时直接采集 Python 全线程堆栈、打断点/打桩和 profiler。主力机据现场选择检查 request 分流、padding/cu/used_q/slot、共享 cache stride/offset、metadata 更新与 reuse fence、target/draft replay。开启逐轮诊断或断点会扰动时序；修复后要关闭诊断重复原配置验证，不能把诊断模式耗时作为性能结论。

数值验收仍按发布机已有 FIA/eager 对照方法和精度阈值执行；这个客户端不比较生成内容，也不自动给出吞吐或性能合格结论。
