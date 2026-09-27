# FlashMLA Decode 接入设计 v1：逐项实施与验收

设计日期：2026-09-27。基线为 `a583897e`，参考为 `de31c53d`，版本和需求以[会话结论](flashmla_session_decisions.md)为准。本文是一版待外部包合约补齐的实施设计；下面的步骤均未实现，拟新增文件/符号不是已存在的 API。

当前架构决定：Prefill 保留 FIA，Decode 接指定外部 FlashMLA，两个阶段共用 2 号非连续 cache。首轮按未量化、PCP=1、DCP=1、无 speculative 的混部路径推进，具体芯片、head、block、dtype 和 TP 配置由包文档及实际模型确定。DSpark/DCP 在基础路径之后独立验收；PD 分离暂不推进。

## 1. 数据流与模块职责

```text
scheduler / runner 提供真实阶段、token 顺序、设备长度与 block table
  → MLA builder：划分 Prefill/Decode，保留 FIA metadata，准备 Flash metadata
  → runner / 既有 executor：提交设备 metadata 任务，管理等待与复用
  → MLA forward：Q/KV 投影与原 writer 写入同一份 2 号 cache
      ├─ Prefill：原展开 K/V → FIA（含历史 context 合并）
      └─ Decode：absorbed Q → 外部 FlashMLA → V 升维
  → 两部分恢复原输出位置 → gate / O 投影 → 原通信与 output buffer
```

metadata 流与计算流可以重叠，但必须分别满足输入就绪、schedule 就绪、KV 写入可见以及下一轮 buffer 可复用的依赖。此图表示职责关系，不规定两个设备 stream 的串行执行顺序。

| 位置 | 负责什么 | 必须保留的语义 |
| --- | --- | --- |
| runner / batch preparation | 阶段来源、请求排序、context 生命周期、capture 外更新 | 请求/token 对应关系、采样/positions/slot/table 一致 |
| MLA builder / metadata | 两条 attention 路径的参数、Flash 稳定缓冲与 task provider | FIA Prefill 所需 CPU/device 信息仍有效 |
| MLA impl | cache 别名、Q/KV 前处理、阶段调用、V/O/gate | 同一 token 只写一次 KV；投影和 gate 不重复 |
| 外部包 adapter（拟新增） | 包发现、能力检查、schema 参数转换、Meta/执行调用 | 不拥有 allocator，不重排持久 cache，不承担 runner 生命周期 |
| `DeviceMetadataExecutor` | 既有 stream/event/fence 协议 | 直接复用，不复制 1 号整份 worker |

adapter 拟放在 `vllm_ascend/ops/flashmla.py`；最终命名先查仓库惯例和包名称。需要新增开关时集中在 `envs.py`，默认关闭，记录支持范围。开关选择的是 MLA Decode，不联动 GQA 或 Prefill。未选择该路径时不要求安装新包。

建议错误策略：显式启用但缺包、静态合约不支持或首轮尚未支持的配置，在执行前报出具体原因；动态 Prefill 始终走 FIA。不要用捕获所有异常后回退来掩盖 Flash 执行失败。此策略须在步骤 01 冻结并写测试。

## 2. 分步提交清单

顺序是依赖顺序。每步只修改对应范围并补齐相关测试，验证结果写入记录后再进入下一步。测试失败先修当前步；遇到缺少包或设备等外部输入，记录未完成项，继续不依赖它的设计工作。不能把未测功能描述为已通过。

| 步骤 | 主要改动 | 对应差异项 | 本步完成标志 |
| --- | --- | --- | --- |
| 01 合约与 adapter | 外部包/schema、能力检查、输入输出映射 | F01–F02、F06 | 真实包导入/Meta/小输入验证，支持矩阵完整 |
| 02 阶段分流 | 真实 Prefill 标记与请求顺序、MLA 分组规则 | F01、F21 | 短 Prefill 和 mixed 路由测试通过 |
| 03 共享 cache | 保留 fused Tensor 与原逻辑 slices，核对 writer/reader 协议 | F04、F21–F22 | 非连续 cache 写入、读取、alias/guard 验证通过 |
| 04 Decode eager | Q 适配、Flash 调用、V 升维、输出合并 | F03、F10–F12 | FIA 基线与 Flash 模块数值一致，Prefill 调用仍是 FIA |
| 05 metadata provider | 稳定缓冲、设备真实长度、容量和 task | F05–F07、F12 | 合约/容量测试及真实 metadata 生成通过 |
| 06 runner 生命周期 | MRv2 target 提交/等待/release，MRv1 复用 | F08–F09 | 连续迭代、异常清理和 buffer 复用通过 |
| 07 图模式 | MRv2 capture/replay，再 MRv1 external events | F13–F15 | 同图变长度/table/顺序的多次 replay 通过 |
| 08 DSpark | context writer、draft query、独立 metadata/图 | F17–F18 | rejection/positions/noncausal 与图组合通过 |
| 09 并行与同步 | 目标配置 DCP/TP/SP/PCP；有证据地减少同步 | F16、F19–F20 | 并行数值、通信语义与性能证据完整 |
| 10 部署与收敛 | 真实启动流程、结果报告、两个 draft 同步 | F25–F26 范围检查 | 旧基线服务验收后再计划新主线迁移 |

### 01：冻结外部包合约，建立最小 adapter

**改哪里：** 拟新增 adapter、对应 UT；必要的 `envs.py`/能力检查入口。先填写[总体分析第 8 节](flashmla_tiling_oldmain.md)参数表，每项注明文档版本和真实 schema 证据。

**怎么改：** adapter 分开提供 metadata 与 attention 调用；只转换已经确认的 dtype/layout/参数。检查芯片、dtype、Q heads、KV heads、block、NoPE/RoPE 维度及 graph 能力。输出轴、latent 维度与 LSE 定义均来自文档，不硬编码参考私仓的接口。按 worker 设备初始化顺序加载包，避免未启用路径受包缺失影响。

**怎么验：** 缺包、版本不符、不支持组合有明确错误；真实包导入/schema/Meta/最小输入执行成功。mock 只检查参数和错误路径。若新包不能消费 2 号 stride/offset，先记录不兼容项并解决合约，不能用复制 cache 作为通过标志。

### 02：先确定真实阶段，再切换 attention

**改哪里：** `attention/mla_v1.py` 的 builder/reorder/forward 路由，必要时修改 `attention/utils.py` 的受限调用方式及 MRv1/MRv2 阶段信息传递；检查共同 batch 排序的其他消费者。

**现有问题：** `reorder_batch` 按 `num_scheduled_tokens <= decode_threshold` 分类；MLA builder 在 PCP=1/DCP=1 时调用 `split_decodes_and_prefills(..., treat_short_extends_as_decodes=True)`。短 Prefill 因而可能落入 `_forward_decode`，目前仍然调用 FIA。只替换该函数会违背“Prefill 用 FIA”的要求。

**怎么改：** 优先复用现有 `is_prefilling` 的真实阶段标记，并沿已存在的 runner 数据链核实含义。首选维持“真实 Decode 在前、Prefill 在后”的边界，使已有两段 slicing 可以复用；必须同时对齐排序和 builder 判定。仅设 `treat_short_extends_as_decodes=False` 不足以证明完成，因为 splitter 假设输入已按阶段排好序。不要全局修改其他 backend 的默认分类规则。

若共同排序不能满足所有 backend，需把显式 token/request 索引及输出恢复作为该步骤的独立变更，并分析额外 gather 和图稳定地址的成本；不得把散落的请求误当连续切片。缺少真实阶段信息时不能退回“长度即阶段”的推断。已有阶段标记与 prefix/chunked/speculative 的对应关系需由实际 vLLM 版本验证。

**怎么验：** query 长度为 1 的 Prefill、prefix 命中后的短尾块、chunked 最后一个 token、纯 Decode、请求交错的 mixed batch、padding/空请求。给出每个请求的真实阶段、选中算子和输出位置。此步可先以两个 FIA 分支验证重排数值等价，再启用 Flash；要求真实 Prefill 的 Flash 调用计数为零。纯 Prefill 保持原算子，不代表阶段分组周边完全无需修改。

### 03：保留 2 号 cache，把两种视图接到正确的消费者

**改哪里：** `attention/mla_v1.py` 的 `forward` cache 入口和传参；writer 仅在发现必要差异时修改。对 `worker/utils.py`、MRv1/MRv2 allocator、zeroer 和 COW 主要做回归。

**怎么改：** 在现有 tuple 切片前保留原 fused Tensor 引用；FIA/writer 使用 latent/positional slices，Flash 使用文档支持的原 fused view 或合法零拷贝轴变换。沿用 `exec_kv_prefill` 和普通 Decode writer，先不启用额外 MLAPO/C8。不会新增 Prefill attention 后的 cache reshape/repack，也不把 component-major 强拼成 fused cache。

**怎么验：** 非连续页、非零 offset、manager/kernel block ratio、跨页 slots、有效/无效 slot、零长度 positional 分量按支持矩阵覆盖。先用唯一 token 标记验证实际写入位置与非目标区域，再用真实 writer→reader 验证。alias 检查应比较底层 storage 归属、shape/stride 和各自 offset；不同 slice 的 `data_ptr()` 可以不同，不能以所有指针相等作为别名标准。精确比较 payload 和未写区域；另回归 zero/COW，不能宣称 COW 复制了全部 padding。

### 04：只接通 Decode eager 计算与输出

**改哪里：** `mla_preprocess_decode`、Decode attention 分支、adapter；复用现有 `_q_proj_and_k_up_proj`、`_v_up_proj` 等适用函数，避免重复做 absorption。

**怎么改：** 根据现有 Decode 预处理结果组成新包 Q；保留模型的 NoPE/RoPE 语义和实际本地 head 数。KV 通过步骤 03 的 writer 写入一次，再交给 Flash。包输出规范化成现有 V 升维可消费的轴顺序，恢复到每 token 的 `num_heads * v_head_dim`，和 FIA Prefill 输出写回同一 `o_proj_input`。gate/O projection 仍只执行一次，保留原 TP/SP 语义。

本步先用 adapter 的真实 metadata 生成在当前计算流上同步排队，形成便于定位的 eager 基线；不引入设备到 host 的长度同步。该临时调度方式不开放 FULL graph，步骤 05–07 将它接入正式生命周期并消除双重生成。不能为了演示运行跳过必需 schedule 或使用伪造 metadata。

**怎么验：** 用相同初始 cache 和输入，分别运行 FIA 基线与候选，防止共享测试 cache 导致两次重复写入。比较 attention/投影后输出、有效 cache payload、当前 KV 可见性和调用次数；覆盖多轮 Prefill→Decode、chunked/prefix Prefill 后 Decode、mixed batch。精度容差按包和模型规范填写；不能只检查输出 shape。

### 05：实现 Flash metadata provider 与稳定缓冲

**改哪里：** MLA builder/metadata、adapter；实现已有 `DeviceMetadataTaskProvider` 协议。无需先把所有 Flash metadata 放进通用 `attention_v1.py`，只有确实共享的部分才抽取。

**怎么改：** 将 query boundaries、设备长度、block table 等输入映射到包的 metadata 合约。根据真实 Meta/容量约束预分配 schedule 等缓冲；容量来源、每轮有效范围、padding/零长度规则均显式记录。由 builder 输出 `ATTENTION` task 和 group 标识，consumer 按相同 group 等待。task 不能偷偷依赖尚未完成的 KV 写入；若包 metadata 确有这种依赖，须相应调整提交点。

**怎么验：** 长度变化、容量边界、空行、padding、无效 slots、不同 group；验证 schedule 对应本轮输入，缓冲地址符合图要求。调用方不能直接持有输出缓冲时，核对包允许的生成/拷贝方式及 stream 依赖；不能把临时输出地址交给长期图。动态长度保持在设备，CPU 上界不能冒充真实可见长度。

### 06：接入已有 executor 的 runner 生命周期

**改哪里：** MRv2 `model_runner.py`、`attn_utils.py` 的 build/forward 外层；MRv1 先复用既有 provider 收集与 context。不得整体替换 runner。

**怎么改：** 有 Flash task 时提交，在消费前等待，消费者入队之后 release；异常路径也要正确释放已提交状态。参考 MRv2 的图外 submit/context 模式，target executor 属于 worker，缓冲属于相应 builder/group。没有 task 的纯 Prefill 不调用要求非空 task 的 `submit`，也不执行未提交的 wait/release。步骤 04 的临时 metadata 调用在本步移除。

**依赖顺序：** 本轮输入就绪 → metadata stream 等待输入和上一轮 reuse fence → 生成 schedule → attention stream 等待 schedule → 读取已写 KV 并执行 attention → 记录可复用 fence。release 记录队列顺序，不等于在 CPU 等待设备执行结束。

**怎么验：** 一次提交/一次释放、空任务、多 group、连续迭代、异常路径、复用缓冲不得跨轮覆盖。CPU mock 检查生命周期，真实 NPU stream 测试检查因果顺序；不能以 mock 的调用顺序代替设备执行证据。

### 07：单独接 MRv2 图，再验证 MRv1 图

**改哪里：** MRv2 `aclgraph_utils.py` 的 capture/replay scope、MLA 图 updater；MRv1 的既有 descriptor/stage/group 机制。

**怎么改：** replay 前刷新稳定地址的动态输入，在图外提交 metadata，并保证实际消费者依赖。仅对已改用 Flash 的 Decode 部分绕过对应 FIA 更新；保留 Prefill、其他层和 fallback 的 FIA workspace/handle/update。保留 2 号 UpdatableGraph 等既有能力。MRv1 使用现有 ExternalEvent/frontier 协议，不能照搬 MRv2 的普通 event 语义。

阶段路由可能改变图形状和捕获分支：明确每种图是否支持 mixed batch，以及 capture key 是否区分所需阶段结构。若现有 FULL 图不支持 mixed，按框架允许方式选择 eager/piecewise 并报告，不能把 pure-decode 图拿来执行真实 Prefill，也不能宣称 mixed FULL 已通过。

**怎么验：** pure Decode、支持范围内的 mixed/prefill 模式、同档不同 lengths/table/请求顺序、请求增删、padding、档位切换、重复 replay、异常后恢复；比较 eager/graph。首次 capture 正确而后续读取旧 metadata 的情况必须能被测试发现。未完成该步时，显式阻止未验证的 Flash FULL 配置。

### 08：DSpark 单独接线与验收

**改哪里：** 先审查 `models/kimi_k3_dspark.py` 的既有 context writer，再接 `worker/v2/spec_decode/dspark/speculator.py` 及适用 draft graph 文件。

**怎么改：** 区分 context KV 写入和 draft query attention；复用 writer 即可时不改模型。target/draft 分别持有 metadata/executor/config 作用域。draft 是否为 MLA、query 是否 noncausal、长度如何受 rejection 修正均按实际架构确认。不要把 draft query 自动等同普通主模型 Decode，也不引入无关 GQA FlashAttn。

**怎么验：** acceptance/rejection、短 query、多 query、positions、真实设备长度、padding、cache ownership、target/draft eager 与图组合。未验收前不能将支持普通 Decode 等同支持 speculative。

### 09：目标并行组合与同步优化

**改哪里：** 按部署需求逐个处理 `attention/context_parallel/mla_cp.py`、runner 及投影/通信入口。

**怎么改：** DCP 核对历史/当前 KV 拆分、local lengths、LSE 合并、空 rank、当前 token 恰计一次；TP/SP 保持现有 head 和输出归属。每次只加一个所需组合。确认所有消费者后，再减少 Flash Decode 已不需要的 CPU 长度同步，Prefill FIA 所需信息继续保留。

**怎么验：** 各 rank 的输入和结果、模型数值、通信与 stream trace、端到端 ITL/吞吐/显存。分别报告 metadata、kernel 和整步耗时；性能比较保持模型、并行度、输入、精度和测量方式一致。`.item()` 列表推导仍可能逐项同步，设备 Tensor 的 Python 条件也可能触发同步，不能把语法改写当作优化证据。

### 10：部署验收、回退和新主线迁移

**改哪里：** [交接流程](flashmla_handoff.md)、测试脚本/报告及最终最小代码增量。

**怎么改：** 另一台机器先读取本地启动测试 skill，记录实际版本和命令；先跑 2 号基线，再跑候选。同一分支推送更新两个 draft，保留 PR 的 vLLM 版本字段。继续保持可关闭 Flash 的旧 FIA 路径，确认回退不会改动或重建缓存协议。

**怎么验：** 服务日志中提供实际路由证据；短 Prefill、长 Prefill、prefix/chunked、mixed、多轮 Decode、图和已声明支持的并行/speculative 组合均有对应结果。任何失败标明基线是否同样失败。旧基线验收后才拆出新主线迁移任务，重新核对 API/cache/graph，不将旧环境结果直接沿用。

## 3. 每步的提交与记录格式

建议每步一个可独立审查的功能提交，测试与功能同提交；必要的修复追加单独 commit。提交使用 Conventional Commits 和 sign-off；不在步骤之间夹带格式化全仓、无关优化或底座迁移。依据 `AGENTS.md` 对 env、patch、runner 变更补齐架构说明与所需验证。

每次推进后追加以下记录，并将相同摘要展示给用户：

| 字段 | 必填内容 |
| --- | --- |
| 步骤与提交 | 步骤编号、SHA、关联 F 编号 |
| 改动 | 文件/函数及行为变化 |
| 原因 | 解决的具体接口/生命周期/数值问题 |
| 验证 | 命令、环境、输入、预期、实际结果、日志路径 |
| 已知边界 | 未支持和未测试项，缺少的外部输入 |
| 下一步 | 当前是否满足完成条件，接着处理哪一项 |

初始记录：步骤 01–10 均处于“设计完成、实现未开始”。会话结论归档提交为 `dccb903d4`；本设计随下一条独立文档提交保存。当前没有新算子的性能、精度或部署成功声明。
