# FlashMLA tiling 下沉：旧主线接入分析与实施计划

状态：2026-09-27，源码分析草稿。当前交付仅含分析、交接文档及差异索引，没有接入新算子，也没有 NPU 验证结果。外部算子文档、包及其版本尚待提供；本文中的参考接口不是最终接口约定。

补充入口：[逐项差异与改动清单](flashmla_change_matrix.md)、[另一台机器的执行交接流程](flashmla_handoff.md)、[完整文件差异索引](flashmla_diff_inventory.json)。当前交付均为分析/交接资料，没有 runtime 改动。

会话中的最终需求、撤回的假设及 Prefill/reshape/cache 写入顺序统一保存在[会话结论](flashmla_session_decisions.md)。

**当前确定的执行范围：PD 混部，prefill 使用 FIA，decode 接入外部 FlashMLA；非连续缓存继续使用 2 号方案。PD 分离暂不推进。** 本文对 1 号的描述是参考分析，不能覆盖这项用户要求；此前拟将 absorbed prefill 一起接入 FlashMLA 的计划已取消。

必须满足的路由约束：纯 prefill、chunked prefill、prefix-hit prefill 均保留 FIA；纯 decode 使用新 FlashMLA；mixed batch 分别处理 prefill/decode 部分后按原 token 顺序写回。两条路径读写同一份 2 号 cache，验证 prefill 写入后 decode 能正确读取。不能按一个全局开关直接把整个 `forward` 跳转到 1 号 `_forward_flash`，也不能仅按 Q 长度将短 prefill 当作 Flash decode。具体阶段判定应核对 scheduler/attention metadata 的实际语义。

保留 FIA prefill 也意味着不能照搬 1 号全局 `use_fia=False`、清空 CPU 长度或跳过所有 FIA 图更新的逻辑。必须保留 prefill 及其他 FIA 消费者所需的 metadata/同步，仅对已证明独立的 Flash decode 部分做下沉。mixed batch 的 query boundaries、输出位置和 gate/O 投影次数也要按两部分接口对齐。

## 1. 固定基线与提交关系

| 用途 | 固定版本 | 处理原则 |
| --- | --- | --- |
| 1 号实现参考 | [私仓 PR #3][plan1]，`de31c53dc5b94ff246b17aa198404a082162c2f9` | 逐项参考执行链和 metadata 生命周期，不整体合并 |
| 1 号原始开发基线 | `990243c4c5b0304248b8ccf01213849e957e1301` | 与 2 号历史基线不同 |
| 2 号布局与开发基线 | [上游 PR #16456][plan2]，`HackClawMxw/vllm-ascend-fork:0913main`，`a583897e0c67d9c23288686728124b04b511a462` | 新分支直接建立于此，保留分配、视图、清零及 COW 语义 |
| 2 号实际历史分叉点 | `7f6c984b1204a3e2f11bf018f881193a6d7203b7` | 由首个功能提交的父提交及与当前 upstream/main 的 merge-base 核对 |
| 配套 vLLM | `84030bbe3d74d99bad477a3d2e37a973ccd8865c` | 两个参考 PR 均注明此版本；执行验证时仍须核对环境 |

新开发分支：`codex/flashmla-tiling-oldmain-v2`。先向 2 号仓库的 `0913main` 创建新 draft，同时在个人 fork 保存相同 head 的 draft。个人 fork 使用固定在 `a583897e` 的专用 base 分支，避免相对个人 `main` 混入无关历史差异。已有 draft #1–#5 保留，不作为本次实现或验证结果。

最终目标仍是 VA 新主线；本阶段先形成相对旧基线的可验证增量，再单独适配新主线。GitHub PR 的 `baseRefOid` 会随目标分支移动，不能把查询时的值当成原始开发基线。1 号 PR 描述中的已验收交付 SHA 也不同于本次拉取的 head，不能将其性能和验收记录直接当成本候选的证据。

## 2. FlashMLA 替换的是什么

当前目标：仅在选中的 dense MLA decode 路径内，FlashMLA 接替 FIA 的 attention 核心计算；prefill 保留 FIA。两条路径的前后处理和共享缓存仍必须正确接通。下面分析 1 号时涉及其更广的 prefill 替换，不代表本次采用。

2 号当前 `AscendMLAImpl.forward` 将 fused cache 零拷贝切成 NoPE/RoPE logical views，沿已有预处理、FIA 和输出路径执行。decode 核心调用是 `torch_npu.npu_fused_infer_attention_score_v2`；prefill 也有 FIA 调用。缓存已经 fused，不代表 attention 已经切到 FlashMLA。

1 号 `AscendMLAImpl.forward` 在 `VLLM_ASCEND_ENABLE_FLASH_MLA` 开启时进入 `_forward_flash`。普通 BF16 absorbed 路径依次执行：

1. Q/KV 投影与归一化；按模型语义处理 RoPE。MLAPO 是满足条件时的前处理优化，不是 FlashMLA 本身。
2. 将 Q 的 NoPE 部分吸收到 latent 空间，得到 512 维，再与 64 维 positional 部分组成 576 维 Q；使用实际本地 Q head 数。
3. 按 slot mapping 将压缩 KV 写入 paged cache。注意 scatter 的 key/value 参数在这里承载 latent/RoPE 两部分，并非普通 MHA 的独立 K/V。
4. metadata 算子根据设备上的长度生成调度信息；FlashMLA 消费 Q、paged KV、block table、长度、mask 和 schedule。
5. 获得 512 维 latent attention 输出，再执行 V 升维、可选 output gate、O 投影及原有 TP/SP 通信语义，写入调用方 output buffer。

不能把这一过程实现成仅替换一个 FIA 函数名。FIA 的 NoPE/RoPE 分离参数、FlashMLA 的融合 Q/KV 参数以及输出布局存在差异。

也不能推断全模型所有 attention 都使用 FlashMLA：KDA、稀疏 attention、GQA 各有独立路径。1 号开关还联动 GQA 的 FlashAttn 与缓存形状；本次 MLA 接入需要避免把这个全局联动原样复制。未选择 FlashMLA 的层仍须保持已有 backend。

## 3. 1 号方案各环节与本次取舍

| 环节 | 1 号当前实现 | 本次接入原则 |
| --- | --- | --- |
| 路由和能力检查 | `platform.py`、`envs.py`、MLA impl 检查 A5、dense MLA、dtype 和维度等 | 按新包文档定义能力边界，明确失败或回退策略；不能从开关推导所有层都兼容 |
| 缓存 | 自有 Flash 路径消费 `[P,S,576]` 并以 `unsqueeze(1)` 传 `PA_BNBD` | 保持 2 号 BBND 协议，不引入 1 号 allocator |
| metadata builder | 稳定的 Q、schedule、长度、table、slot、positions、live mask；Meta 算子查询 schedule 形状 | 参考生命周期，用外部包自己的 schema/Meta/容量规范 |
| metadata 调度 | `DeviceMetadataExecutor` 独立 NPU stream，输入就绪、消费等待、复用 fence；该文件与 2 号基线完全相同 | 复用已有 executor，补 MLA provider 与 MRv2 调用链，保证无跨轮覆盖 |
| attention | 自带 `torch.ops._C_ascend.flash_mla_with_kvcache` 及 metadata binding | 接用户指定的外部包；不移植私仓 native kernel 或硬编码其 metadata 格式 |
| prefill | 新增 192/128 非吸收 FlashAttn、分块历史展开及 online merge；保留 absorbed 路径 | 保留 2 号 FIA prefill，不引入 absorbed FlashMLA prefill 或非吸收 FlashAttn prefill |
| decode | BF16 FlashMLA、可选 MLAPO；另有 C8 分支 | 首轮聚焦文档支持的基础精度，量化与融合优化独立验收 |
| DCP | 本地历史 KV、Q 汇聚或复制、当前块、output/LSE 合并 | 后续独立里程碑；不能只将 DCP size 传给算子就认为完成 |
| DSpark | 独立 executor、context KV 写入、query metadata 与 capture 路径 | 分开审查 target 和 draft；按实际 MLA/GQA 架构处理 |

1 号当前已经不是“纯外部包接入”的最小样例：MLA 使用 VA 内置 `_C_ascend` binding，部分 FlashAttn 路径仍使用 `cann_ops_transformer`。这与待接的新包是否同源、同版本、同 ABI，必须以新文档核对。

## 4. tiling 下沉具体下沉哪一段

源码显示：1 号 metadata binding 调用 `aclnnFlashMlaWithKvcacheMetadata`，L0 通过 AICPU launcher 执行；AICPU kernel 的 `Compute` 包含 `Prepare`、`BalanceSchedule`、容量检查和 `GenMetadata`。

因此这里主要是把依赖动态序列长度的调度信息生成放到设备侧 AICPU，让 FlashMLA kernel 消费生成的 schedule。Python builder 仍负责准备输入与缓冲，host binding 和静态属性处理仍存在。“tiling 下沉”不能等同于“完全没有 host tiling”，也不能等同于“模型所有 CPU/NPU 同步都消失”。

长度来源同样关键：1 号使用设备上的 `seq_lens` 和 query boundaries，包括 speculative rejection 修正后的值；CPU 的长度上界不一定是本轮真实可见 KV 长度。新实现不可为了得到 Python list/max 值在 decode 热路径加入 `.cpu()`、`.item()` 或 `.tolist()`。

1 号 metadata builder 用 Meta 调用得到 schedule 的大小，再为实际设备分配稳定目标缓冲；每轮实际 schedule 更新到这个缓冲。其 native binding 含芯片核数和对齐常量，本次应依赖新包的接口，不把这些常量抄进 VA Python。

## 5. 图模式：必须分别处理 MRv1 与 MRv2

2 号现有 FIA 图路径会保存 attention 参数、workspace、handle 和 ExternalEvent，并通过图任务更新接口刷新参数。1 号 Flash MLA 的 `update_graph_params` 提前返回，由独立的 metadata 生命周期提供新输入。

但只有稳定地址刷新、执行顺序、生命周期全部接通后，才能绕过 FIA 更新；直接将 updater 改成空函数会让 replay 读取旧长度或旧 table。

| 路径 | 1 号源码中的依赖关系 | 需要验证 |
| --- | --- | --- |
| MRv1 FULL | 提交携带 batch descriptor，executor 按 descriptor/stage/group 管理 ExternalEvent；attention 消费处 wait/reset | 首次 capture、同档多次 replay、档位切换、frontier 一致性 |
| MRv2 eager/FULL | `build_attn_metadata` 明确在 capture 外 submit，并用普通 event 等待；target/draft/capture 外层 `device_metadata_context` 负责 release | metadata 在 replay 之前刷新，release 在消费者入队之后；异常路径正确清理 |
| 缓冲复用 | 下一轮 metadata stream 等待上一轮消费 stream 记录的 reusable fence | 不允许上一轮 attention 尚未读取完就覆盖长度、schedule 或 table |

FULL replay 不会重新执行被捕获的 Python forward。因此 replay 前的图外刷新不是可选优化，而是正确性要求。验证时必须在同一 capture 档内改变长度、请求顺序、block table 和有效请求数，不能只用同一组输入反复 replay。

1 号为 padding 增加零有效长度的额外 request，使用 `slot=-1`、`token_live` 和输出 mask。新包是否允许零长度行、重复累计长度、空历史及 padding 形状，要逐项确认。仅验证首次非空 eager batch 不够。

ACLGraph 与 torch.compile/static kernel 是不同层面的能力。算子可 capture 不代表 Meta/Fake、编译、所有 graph mode 或 DSpark 图已经通过。

## 6. 继续保留 2 号非连续布局

2 号当前 MRv2 为具备 `MLA_FLASH` 能力且 Q heads 属于 `{8,12,64,96}` 的路径建立 token-fused cache：`[P,B,N,576]`，即 BBND。每个 token 的最后一维是 `[latent512 | positional64]`，页首 stride 可以大于页 payload，storage offset 也可能非零。

上述描述还有前置条件：不配置 KV transfer、普通未量化 dense MLA、匹配 spec/layer 类型等。当前先按无 KV transfer 的混部场景验证，不扩大该条件的支持范围。

对于 A3/FIA 等路径，2 号保留 component-major NoPE/RoPE views；同一 raw backing 内两部分分别组织，padding 体现在页跨度。不能将这种布局伪装成 token-fused tensor。

1 号传给算子的 `PA_BNBD` 是 `[P,N,B,D]`；本次需要优先检查新包是否直接支持 `PA_BBND`。如果必须换轴，也只能在包明确接受对应 stride 时构造存储别名，不能假定一次 `view` 或改 layout 字符串即可适配。

即使 MLA 的 KV head 常为 1，单例维度使数值上看似相同，也不意味着轴语义、stride 检查和 binding 合约相同。写缓存和读 attention 必须共同遵循 2 号物理协议。

需要保持的约束：allocator ownership、raw backing、shape/stride/storage offset、manager block 到 kernel block 的映射、slot/table 编号，以及 2 号现有 zero/COW 语义。更精确地说，V1 zeroer 按物理页建立清零元数据；COW 用 `unflatten` 保留 stride，复制 manager block 覆盖的各 kernel block 的逻辑 payload，不应宣称它复制了所有 padding 字节。MRv2 使用独立的 tuple-aware zeroer，也要单独回归。不得通过对持久 cache 调用 `.contiguous()` 掩盖不兼容；这会产生新存储并破坏写回别名及图地址。Q 或临时张量的必要连续化应单独判断。

## 7. DSpark、DCP 和容易遗漏的边界

DSpark 至少有两条链需要核对：context hidden states 的 KV 预计算/写入，以及 draft query 的 attention。1 号 K3 draft 的 context 写入调用 `exec_kv_prefill`，只改主模型 forward 会漏掉它。query 组的 causal 属性由 draft metadata 提供，不能全部强制为主模型的 causal decode。

target 和 draft 需要独立的 metadata 缓冲、executor 和 config 作用域。验证 rejection、实际 query 数、padding、positions 和下一轮目标长度时，要检查设备真实值，不能直接依赖 scheduler 的 CPU 上界。也不能因为模型叫 DSpark 就假定 draft 一定是 MLA；1 号还有单独的 GQA draft 路径。

DCP 模式下，1 号把 causal attention 拆为历史和当前块：历史使用各 rank 的本地 KV 长度；当前块按相应复制策略计算；之后用 output/LSE 合并。直接平均 output 是错误的，重复计入当前 token 也会错误。Q head ownership、TP shard、当前块暂存 cache、通信输出和空 rank 都需要验证。

PCP、PD/KV transfer、KVPP、量化 C8、非吸收 prefill、Q replication、融合 gate/O-proj 等都可能与接入相交，但不应未经依赖分析整包引入。首轮选择 PCP=1/DCP=1 基础路径是实施顺序，不是对新算子能力的断言。其余配置必须显式路由或报错，不能静默使用未经验证的路径。

PD 分离不在当前阶段范围；后续用户恢复此范围时再使用保留的调查资料。本轮先证明混部中的 FIA prefill → 同一非连续 cache → FlashMLA decode 链正确。

## 8. 收到算子文档后的参数核对表

| 合约项目 | 待确认内容 |
| --- | --- |
| 安装与发现 | 包名、导出符号、版本、设备/CANN/torch-npu 依赖，worker 选定设备后的加载时机 |
| Q | shape、dtype、实际 head 数、stride，TND/NTD，QK 是否融合、NoPE 模型语义 |
| KV | `PA_BBND`/`PA_BNBD`/NZ 支持，paged fused 输入，首维/其他维 stride，非零 offset，block size |
| 长度 | `cache_seqlens`、`cu_seqlens_q`、`seqused_q` 的 dtype/device，inclusive 与历史长度定义，padding/空行规则 |
| 最大长度 | `max_seqlen_q/kv` 的含义、是否要求 `-1`、是否接受容量上界；1 号当前传的是容量上界，不能直接沿用 |
| mask | causal/noncausal 与 mask_mode 对应、mask dtype/shape，长上下文及多 query 的对齐规则 |
| metadata | 生成接口、Meta/Fake 支持、容量、输出 dtype/device、是否允许 caller-owned output、异步/捕获约束 |
| 输出 | V=512 的 latent 输出、TND/NTD 轴顺序、LSE shape/dtype/数值定义、空 KV 输出语义 |
| 写缓存 | 现有 scatter/前处理算子是否接受 2 号 strided views，是否正确处理 `slot=-1` |
| 图与 workspace | capture 支持、动态长度更新方式、workspace 生命周期、binding 是否偷偷连续化或同步到 host |

## 9. 分步骤交付与验收

1. **本次：基线和分析。** 拉取固定版本，建立新开发分支及两个 draft；记录执行链、差异与待确认合约。本次只交付文档。
2. **接口适配。** 对照新文档逐项填写参数表，实现最小外部包 adapter 和能力检查；用真实包验证导入、schema、Meta 和参数路由。模拟测试只能证明 host 合约，不能证明算子可用。
3. **混部基础 eager。** 保留 FIA prefill，在 2 号布局上接通 decode 的 Q/KV 准备、cache 写入、metadata、FlashMLA、V/O 投影。验证纯 prefill 仍调用 FIA、纯 decode 调用 FlashMLA、prefill→decode 共享 cache，以及 prefix-hit/chunked/短 prefill 和 mixed batch 分流；先使用 PCP=1/DCP=1、无 speculative 的基准配置。
4. **图模式。** 先 MRv2 target 的 capture/replay，再独立处理 MRv1 的事件语义。验证稳定地址、跨轮 fence、不同 batch 档、padding、请求增删/重排；保留非 Flash 层的现有图更新。
5. **DSpark。** 接通 context 写入与 draft query/capture；分别验证 target/draft metadata、noncausal 语义、接受/拒绝 token 后的下一轮长度，再组合 FULL 图验证。
6. **混部并行与性能。** 按目标部署需求扩展 DCP/TP/SP 和所需 PCP；prefill 始终保持 FIA，PD 分离另行恢复范围后再处理。测量 metadata、host 同步、decode attention kernel 及整步耗时，不以 kernel 降幅替代端到端 ITL。
7. **旧基线收敛与新主线迁移。** 旧基线测试通过后整理最小增量；再适配 VA 新主线，重新核对 vLLM/VA API、cache contract 和图执行流程，不把旧基线通过视为新主线通过。

测试至少覆盖：文档支持的 head/dtype/block 组合、不同 Q/KV 长度、非连续页、非零 offset、zero/COW 与相邻页保护、当前 KV 可见性、同图变输入、空/填充行及 speculative rejection。精度阈值从算子文档与模型验证约定确定，cache 写入和保护区比较要求精确。长上下文和特殊维度是否支持须有真实包证据。

## 10. 源码阅读入口

以下路径以表中的固定 SHA 为准，后续修改不要用移动分支替代引用：

| 主题 | 源码入口 |
| --- | --- |
| 1 号 metadata 与容量 | [attention_v1.py][ref-meta]：`AscendFlashAttentionMetadata`、`_flash_attention_schedule`、`_build_flash_attention_metadata` |
| 1 号 MLA 执行 | [mla_v1.py][ref-mla]：builder、`_forward_flash`、`_forward_flash_full_prefill`、`exec_kv_prefill`、`update_graph_params` |
| 1 号 stream/event ownership | [device_metadata.py][ref-executor]：`submit`、`wait`、`release` |
| 1 号 MRv2 图外提交 | [worker/v2/attn_utils.py][ref-runner]：`device_metadata_context`、`build_attn_metadata`；另见 `aclgraph_utils.py` 的 capture scope |
| 1 号 AICPU 调度 | `csrc/attention/flash_mla_with_kvcache_metadata/op_host/op_api/l0_flash_mla_with_kvcache_metadata.cpp` 与 `op_kernel_aicpu/flash_mla_with_kvcache_metadata_aicpu.cpp` |
| 1 号 native binding | `csrc/attention/flash_mla_with_kvcache/flash_mla_torch_adpt.h` |
| 2 号布局 | [worker/v2/attn_utils.py][base-layout]：`reshape_kv_cache`；另见 `worker/model_runner_v1.py`、`worker/utils.py`、`worker/v2/utils.py` |
| 2 号原 attention | [mla_v1.py][base-mla]：`forward`、`_forward_decode`、FIA capture/update 路径 |
| 2 号现有回归 | `tests/ut/worker/test_fused_mla_cache_lifecycle.py`、`test_attn_utils_v2.py`、`test_model_runner_v1.py` |

[plan1]: https://github.com/maoxx241/vllm-ascend-rfc16468-private/pull/3
[plan2]: https://github.com/vllm-project/vllm-ascend/pull/16456
[ref-meta]: https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/attention/attention_v1.py#L210
[ref-mla]: https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/attention/mla_v1.py#L2803
[ref-executor]: https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/worker/device_metadata.py#L45
[ref-runner]: https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/worker/v2/attn_utils.py#L425
[base-layout]: https://github.com/HackClawMxw/vllm-ascend-fork/blob/a583897e0c67d9c23288686728124b04b511a462/vllm_ascend/worker/v2/attn_utils.py#L1170
[base-mla]: https://github.com/HackClawMxw/vllm-ascend-fork/blob/a583897e0c67d9c23288686728124b04b511a462/vllm_ascend/attention/mla_v1.py#L2100
