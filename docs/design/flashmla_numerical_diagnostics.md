# FlashMLA eager 数值诊断

本诊断用于旧主线 A 线的连续 `!` 问题，与新主线/KDA 迁移 B 线分开。
基线 `8327fd485453e9475c4af4bae6b8aaeb8f480bad` 的 graph 与 eager D 均已复现；
eager 仅三个顺序 smoke 成功，尚无 eager C8 通过证据。
本地 CPU 检查只证明诊断接入和保存逻辑，不证明真实 NPU 计算正确。

**当前发布状态：暂停，等待用户人工检查。** 主力机可以继续本地开发，发布机不得因为
候选代码或本文更新自动拉起/发负载。机器已交给其他人排查；以下实验步骤只有在用户明确
检查通过并授权恢复验证后才执行。该限制优先于此前自动推进或服务待命安排。

## 为什么增加数值诊断

原 `VLLM_ASCEND_FLASH_MLA_TRACE` 记录 metadata、FIA/Flash 路由、地址和图提交。
它没有 raw/processed logits、最终 sampled token 或中间张量的数值证据。
PR 记录曾明确 TRACE 未开启，后续 eager/graph 回执也未提交数值取证。
不能把代码里存在日志、HTTP 200 或 `decode_external` 当成数值验证版已运行。

本版观察 MRv2 真实 `compute_logits`、`Sampler.sample` 和最终 `SamplerOutput`，
不修改返回张量、不替换采样算法、不改变 Prefill FIA/Decode FlashMLA 或非连续缓存协议。
仅限 eager、无 speculative decoding 的匹配接口；不支持的配置应显式报错。
默认关闭。开启后克隆、归约、CPU 保存引入同步，结果不能当性能数据。
这些同步也可能改变调度或掩盖时序问题；开诊断后不再复现不能直接宣称修复，
应保留无诊断的原失败证据并另行缩小观测干预。

每个 DP 的 TP0 独立记录，以免只观察 DP0 漏掉其他副本的错误。
不同 rank 的 step 是进程内编号，必须结合请求 ID、位置和时间关联，不能仅按 step 数对齐。

## 开关与输出

将以下设置加入发布机已有 eager 启动流程，并确保四节点的 worker 继承这些变量。
输出路径使用该轮 run 的独立目录，不能跨运行复用身份。

```bash
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR=/absolute/path/to/run/sample-diag
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS=128
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS=2
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK=-1
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK=0
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS=0
```

DIR 默认空表示关闭；STEPS 默认 64，范围 1–128；ROWS 默认 2，范围 1–8。
DP_RANK=-1 表示所有 DP，TP_RANK 默认 0；TOKEN_IDS 为 1–16 个非负 token ID。
实际监视 ID 应按 tokenizer 核实，以上 0 仅用于检查 sampler 回退候选，不代表 `!`。
首次复现直接跑 D，不先跑一串 smoke 消耗预算。

输出子目录包含 host、PID、rank 和运行唯一标识；`events.jsonl` 保存摘要，
逐步 `.pt` 保存有限行数的 CPU 张量。摘要可能包含请求 ID、token ID、位置和 logits，
按本地测试数据保存，不把整个原始目录直接贴到公开 PR。
诊断完成应取消这些设置；原 TRACE 开关仍是独立的路径日志，不等于本数值开关。

## 必须验收的证据

发布机启动前核对候选完整 SHA、实际导入路径、pinned vLLM
`84030bbe3d74d99bad477a3d2e37a973ccd8865c`、算子包和诊断输出目录。
启动与请求后按以下要求验收；文件缺失时先检查是否命中诊断入口，不重复声称已验证。

- 每个选中的 DP/TP 都出现诊断 armed 标记，并标明 host、PID 和 rank。
- 真实请求出现 batch/end 记录，关联 request ID、logits 行、位置和最终 sampled token。
- 保存 raw 与 processed logits 的 NaN、正负无穷、有限值范围及 top-k。
- 区分采样器内部 token 和最终 token；`num_sampled=0` 的未完成 chunked Prefill 不算用户已收到 token。
- 保存有限数量 CPU 张量快照，记录对应行和阶段，用于离线检查。
- 输出预算耗尽必须有 exhausted 标记；异常发生在窗口之外，结论是未捕获，不能写成数值正常。

监视 token 0 不意味着已经证明 `!` 对应 token 0。
发布机必须用实际模型 tokenizer 核实，并关联 SSE 中的 request 与服务内部 request ID。
`!` 也可能由多个 token 组合产生，不能仅凭字符串推断无效概率回退。

## 下一次启动的实验顺序

1. 先准备好明确的诊断 SHA、输出目录和原 D payload。模型、算子包和采样参数保持原配置。
2. 使用发布机已有拉起流程启动 eager，确认诊断 armed 和 FIA/Flash 路由；不同时启动 B 线。
3. 先跑原 D 的 C60 / max_tokens=32 / stream=true，沿用发布机的 SSE 全局止损探针。
   请求窗口和预算以该轮 PR 执行指令为准；开启同步诊断后延迟变化不能作为性能退化结论。
4. 连续异常先停客户端并取消本轮请求，保存现场、确认排空，再检查每个 DP 的诊断文件。
   不因出现 `!` 立刻杀掉健康服务；按 PR 的服务待命/资源释放规则处理。
5. 同一服务可以继续做只读文件分析、tokenizer 核对；若已有可控制的 profiler 接口，
   也可按既定方案短时采集，不假设所有 profiler 都能在启动后无配置开启。
   新负载必须由该轮实验分支明确指定；缓存疑似越界/污染时不能把后续运行当干净对照。
6. 需要修改 Python 代码、包或启动参数时才安排下一次重启。
   不在同一轮混入 sampler 更换、KDA、DSpark、PD 或全量 GPQA。

eager C8 是后续待补的并发 8 对照，不是模型权重格式。
它应使用明确匹配的题目、采样参数和干净状态；本次先做 D 取证，
不因看到这条待办就自动发出 C8。是否在同一服务补测取决于状态可信度、剩余诊断预算及该轮指令。

仓库 `tools/flashmla_concurrency_probe.py` 使用非流式请求，不能替代已经保存 payload
且支持 SSE/全局止损的发布机探针。D 的完整 payload 哈希与提示源哈希应分别核对。

## 如何选择下一层 golden

| 首个证据 | 下一项检查 | 可以得出的结论 |
| --- | --- | --- |
| raw logits 已有非有限值 | LM head 输入 hidden，然后向前定位首个异常层 | 不能先归因 sampler；第一输出 token 常来自 FIA Prefill |
| raw 正常、processed 异常 | grammar、penalty、temperature、min-p、top-k/top-p 的处理阶段 | 注意 mask 后的部分 `-inf` 正常，全行无有效候选才是异常 |
| 两者有限但内容错 | 真实 sampled ID、top-k 和该 token 分数；同输入张量的参考计算 | 有限不等于正确；随机采样不能要求两个实现逐 token 一致 |
| sampled ID 与 SSE 不一致 | 请求映射、tokenizer 和输出后处理 | 暂停扩大 attention 排查 |
| 第一个错误位于 KV writer | 实际 slot 的 576 维 KV 与本次归一化/RoPE 输入比较 | 单独验证 writer，不能用合成缓存读测试代替 |
| history gather 开始错 | 实际 block table/offset/stride 直接索引，与 gather 比较 | 找到差异先停，不把坏输入归因 FIA |
| FIA history/merge 开始错 | 相同 query/KV 的分块参考；零历史行合并应等于非空分支 | 空历史 LSE=`-inf` 可以合法，检查最终合并结果 |
| Flash 活跃输出开始错 | 保存真实 Q、实际页表/length/used 和对应 KV，做参考 attention | 使用生产页 stride；只比较有效 token，不检查未定义 padding 输出 |

第一版保存的是采样边界快照，**尚不是 attention/KV golden 抓取器**。
只有前述证据指向某层后，才增加该层小规模快照，避免一开始 dump 全模型/全 KV。
比较 attention 时固定输入、scale、mask、dtype 和有效 token；报告最大绝对/相对误差、
非有限值和合理容差。不要把另一份有已知精度问题的 PR 输出作为 golden。

## Profiling 与 Python 堆栈的用途

- Profiler：确认实际 kernel、metadata 与主算子次序、通信和同步等待；使用发布机已有工具，
  限定最初异常的少量 step。不能单凭 kernel 执行记录证明数值正确。
- Python 全线程堆栈：用于无进度、等待或超时。当前 SSE 仍在持续输出 `!`，
  单张 host 堆栈通常不能解释错误数值，因此不作为本轮主要检查。
- 断点：先缩小复现；多卡暂停会影响通信，必须记录干预。
  图模式需另行设计固定设备快照，不能把本 eager 诊断直接塞进 capture。

发布机回执应包含 armed/首次真实 batch/end、输出文件索引、首次异常 request/rank/step、
原始 SSE、预算是否耗尽及是否有诊断失败。只有这些证据齐全才算本验证版实际运行。
