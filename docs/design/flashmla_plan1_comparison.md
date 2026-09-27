# 1 号参考与当前候选：CPU length、tiling 和图更新复核

复核日期：2026-09-27。参考为本地 1 号 `de31c53dc5b94ff246b17aa198404a082162c2f9`；当前运行代码为 `698d00e7c86eb8a9175c92d384497b28f5b798eb`，基于 2 号 `a583897e0c67d9c23288686728124b04b511a462`。`dccb903d4` 是会话归档文档提交，不是 1 号代码。本次提交只补充复核说明，不改变运行代码。

结论：当前已接入设备长度驱动的 FlashMLA metadata/Decode，Prefill 保留 FIA；没有照搬 1 号全局 Flash 路由，也没有完成 1 号特定 K3 路径的 CPU 同步消除优化。下文的“已核对”指源码调用链检查，不表示 NPU、异步 rejection 或图 replay 已验收。

## 一、逐项区别

| 项目 | 1 号固定参考 | 当前候选及原因 |
| --- | --- | --- |
| attention 路由 | 开关开启后，MLA `forward` 整体进入 `_forward_flash`；含 absorbed FlashMLA / expanded FlashAttn Prefill 分支 | 只替换 Decode attention；Prefill 调原 `_forward_prefill` 的 FIA，满足用户最终需求 |
| 短 Prefill | builder 调长度 splitter，`treat_short_extends_as_decodes=True` | 读取真实 `is_prefilling`，MRv2 在构造输入前按阶段稳定排序；短 prompt 尾段仍走 FIA |
| 外部入口 | MLA 经 `torch.ops._C_ascend.flash_mla_with_kvcache*` native binding；部分 Prefill 使用外部 FlashAttn | 两个 MLA 算子均经 `cann_ops_transformer.ops` 公开导出，未新增/复制 C++ binding |
| KV 布局 | 普通 Flash MLA 路径使用 PA_BNBD；C8 history 等分支也有 PA_BBND，不能概括为全部 BNBD | 模型只接 2 号 fused PA_BBND 非连续 view；不重建持久 cache，不复制 allocator/zero/COW |
| 写缓存和投影 | Flash 专用 forward 整理 Q/KV、计算 attention 和输出 | 保留 2 号前处理/writer；拼 absorbed Q 的临时输入，Flash 输出 NTD latent512 后接原 V 升维、gate/O。Prefill 写缓存发生在 FIA 前 |
| 长度上限属性 | schedule/main 使用 `flash.max_query_len`、`flash.max_seq_len` 等容量上界 | metadata/main 均显式传 `max_seqlen_q=-1`、`max_seqlen_kv=-1`，按本次外部包目标合约 |
| metadata 容量与生命周期 | 用 Meta 定容、稳定 schedule、设备长度刷新、现有 executor | 沿用这一机制；独立 `flashmla_metadata.py`，保留与 2 号 runner 的边界 |
| 缓冲 key / padding | key 含 batch、tokens、table columns、causal 等，额外零 used 请求承接 padding | 纯 Decode 按 token 容量预留 request rows；排除尾部空请求，固定 cu/used_q/lengths/table/slots/positions/schedule，padding slot=-1 |
| CPU length | Flash MLA metadata 的 `seq_lens_cpu=None`；MRv2 特定 K3/PCP=1 组合可关闭长度 D2H/sync | FIA Prefill 需要 CPU 长度；保留原 runner 镜像、`use_fia` 与 speculative 同步。Decode 算子实际长度仍只读设备值 |
| `update_graph_params` | Flash 开关开启时 MLA updater 全局直接 return | 仅过滤 `external_flashmla` 层，剩余 FIA MLA 层保留 updater；没有 FIA MLA 层时在读取 attn_params 前 return |
| MRv2 executor 绑定 | 模块级 ContextVar 保存当前 executor | scope 绑定到当前 MLA builder 实例，退出恢复；target/draft 分别持有 executor，未新增全局可变状态 |
| 图路由 | 与整条 Flash forward 的路由配套 | 真正 Prefill 不复用 Decode FULL 图：MRv1 排除 FULL；MRv2 若选中 FULL 则转 eager，原 PIECEWISE 选择保留 |
| DSpark | Flash 专用 context/query、causal 和图流程 | 保留 2 号 context writer，并兼容直接传入 fused Tensor；query 独立 metadata/executor，跳过原 FIA query-length 改写 |
| 扩展范围 | 另有 C8、DCP/current-KV 合并、Prefill 等更广分支 | 本批未移植；PCP/DCP 必须为 1，无 KV transfer，未量化 BBND；实际支持范围见开发记录 |

## 二、CPU length 逐层追踪

| 数据 | 当前来源和消费者 | 源码复核结果 |
| --- | --- | --- |
| 请求阶段、CPU query boundaries | `common.is_prefilling`、`query_start_loc_cpu` → `split_flashmla_requests` | 只用于阶段划分/物理 token 范围；不是 FlashMLA 的 KV 可见长度 |
| Decode KV 长度 | `common.seq_lens` 设备 tensor → `flash.cache_lens` → metadata/main | 每轮在 refresh task 内读取，未用 `seq_lens_cpu_upper_bound` 替代；零 used 行长度清零 |
| Decode query 长度 | 设备 `query_start_loc` → cu 差分 → used_q | 与 CPU token 范围裁剪配合；无效请求和图 padding 不写 cache，主算子使用同一组设备 cu/used_q |
| Decode 旧 FIA 参数 | `decode.seq_lens_list=[]`、`actual_seq_lengths_q=None` | external 路由不进入 `_forward_decode` 的 FIA，也不进入该层 FIA graph updater；这些旧参数不会传给 FlashMLA |
| Prefill CPU 长度 | 优先 `common._seq_lens_cpu`，否则 `common.seq_lens_cpu` → 原 `build_prefill_metadata` / `build_chunked_metadata` | 为 context 分块、累计长度和 FIA host 参数保留；缺失时报错，不在新 builder 中临时 D2H |
| MRv1 CPU 镜像 | 原 runner 传 `optimistic_seq_lens_cpu`；async 时公开 `seq_lens_cpu` 可为 None，但 `_seq_lens_cpu` 保留 | 沿用 2 号来源，不把 optimistic 镜像宣称为每个 Decode rejection 后的精确值；Flash Decode 不消费它。发布机须覆盖 async mixed/prefix/chunked |
| MRv2 CPU 镜像 | `_update_seq_lens_cpu`；speculative 时 `_copy_num_computed_tokens_to_cpu` + event.synchronize | 保留 rejection 修正及 FIA 所需镜像；该处仍有 host 同步，未达到 1 号特定路径的去同步程度 |
| host 容量上界 | `max_seq_len`、`seq_lens_cpu_upper_bound` 仍可能由通用 runner/其他 backend 使用 | 不作为 Flash metadata/main 的实际长度，亦不替代要求为 -1 的属性 |

1 号 `_use_device_seq_lens` 受 executor 启用、Kimi K3 target、PCP=1 等条件限制；不是开启 Flash 就对所有模型删除 CPU mirror。当前也不能在保留 FIA Prefill、混合 backend、speculative 修正的情况下照搬该开关。

**性能边界：** “设备生成 Flash schedule”已经接线；“整个 runner 不再等待 CPU length”尚未实现。仍保留的 D2H/sync、host 请求处理和外部包 launch 开销，可能影响总体收益。若要继续消除同步，应作为下一项代码优化，先列清 FIA Prefill/其他 backend/拒绝修正的全部消费者，再由发布机提供 profile 和正确性结果。

## 三、update_graph_params 和 replay 时序

### 1. 为什么 FlashMLA 不走 FIA updater

原 updater 使用 `decode.seq_lens_list` / `actual_seq_lengths_q`，在 `graph_task_update_begin/end` 之间重发 `npu_fused_infer_attention_score_v2.out`。这是 FIA 捕获任务的更新协议，不能拿来更新外部 FlashMLA。

当前 target 和 draft 的 `attn_keys` 都排除 `metadata.external_flashmla is not None` 的层。过滤后为空立即返回；剩余 FIA MLA 层继续使用原逻辑。因此 Flash Decode 的空 CPU list 不会被该 updater 消费，也不会用 FIA 覆盖捕获的 Flash 调用。Prefill 仍 FIA，但含 Prefill 的批次已从 Decode FULL 图路由排除。

这不是让 FlashMLA 复用旧 schedule：其更新发生在下述设备 task 路径。

### 2. 每轮的真实更新顺序

```text
runner 准备本轮设备 lengths / query boundaries / block table / positions
  → builder 收集 refresh task（复用稳定缓冲）
  → executor 等待输入就绪、上一轮 buffer reuse fence
  → refresh lengths / cu / used_q / slots / positions / rope / schedule
  → compute stream 等待 metadata ready
  → 原 writer 写当前 KV → FlashMLA attention → V/O/gate
  → 记录 buffer reusable，下一轮再更新
```

- MRv1：复用原 executor；FULL descriptor 使用 ExternalEvent。`forward` 在 writer/rope 读 slots/positions 前等待，attention 处的重复 wait 由 executor 去重；forward 完成或异常由原 finally release。
- MRv2：`build_attn_metadata` 在 capture/replay 外 submit/wait；若在 capture 内尝试提交则报错。scope 在模型消费者入队后 release，下一次 metadata build 先结束上一次 submission。
- DSpark：独立 executor 和 capture/propose scope；draft query 的 causal 决定 mask，context writer 不是 Prefill attention。仍需实测 rejection、padding、下一轮长度和 target/draft 交替。
- 2 号 `UpdatableGraph` 与其他 backend 的机制保持。当前 `use_updatable_graph` 针对 `AscendAttentionBackend`；不能把所有 MLA 图概括为都走 UpdatableGraph。混合多种可执行 attention backend 的通用更新仍受原 `_get_graph_update_backend` 选择首个可执行 backend 的边界约束，本次不宣称覆盖任意混合组合。

### 3. 核对范围和发布机验证点

已核对：外部长度来源、-1 属性、旧 FIA list 的消费者隔离、target/draft updater 过滤、metadata task 收集/submit/wait/release、Prefill FULL 路由隔离，以及 shared executor/cache 核心文件相对 2 号未改变。本次没有发现上述调用链需要立即改动的接线问题。

尚未证明：二进制真实执行、零长度行/padding 支持、实际异步可见性、固定地址的连续 replay、非连续第 1 轴支持、FIA 与 Flash 数值一致或性能提升。源码复核不能替代这些证据。

发布机重点留下三组结果：

1. 异步/DSpark rejection 后，设备 lengths 与主算子输入相符；CPU upper bound 即使不同，也没有成为 Flash 可见长度；原 FIA Prefill 的 context 长度正确。
2. 同一捕获档位改变 lengths/table/请求顺序，多次 replay 的 metadata 内容更新、地址不变、输出与 eager 一致。Python 首次日志不能替代 profiler 的算子事件。
3. 纯 Prefill、短 Prefill、mixed 中的 Prefill 实际进入 FIA；纯 Decode 实际进入外部 FlashMLA，FIA updater 没有为它更新 task。分别统计 metadata kernel、attention kernel 和残留 CPU/D2H/sync 时间。

## 四、可追溯源码

- 当前 [MLA builder / forward / updater](https://github.com/Henry-Avery/vllm-ascend/blob/698d00e7c86eb8a9175c92d384497b28f5b798eb/vllm_ascend/attention/mla_v1.py)、[metadata refresh](https://github.com/Henry-Avery/vllm-ascend/blob/698d00e7c86eb8a9175c92d384497b28f5b798eb/vllm_ascend/attention/flashmla_metadata.py)、[adapter](https://github.com/Henry-Avery/vllm-ascend/blob/698d00e7c86eb8a9175c92d384497b28f5b798eb/vllm_ascend/attention/flashmla.py)。
- 当前 [MRv2 CPU mirror](https://github.com/Henry-Avery/vllm-ascend/blob/698d00e7c86eb8a9175c92d384497b28f5b798eb/vllm_ascend/worker/v2/model_runner.py)、[MRv2 executor scope](https://github.com/Henry-Avery/vllm-ascend/blob/698d00e7c86eb8a9175c92d384497b28f5b798eb/vllm_ascend/worker/v2/attn_utils.py)。
- 1 号私仓 [MLA](https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/attention/mla_v1.py)、[共用 Flash metadata](https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/attention/attention_v1.py)、[MRv2 长度优化](https://github.com/maoxx241/vllm-ascend-rfc16468-private/blob/de31c53dc5b94ff246b17aa198404a082162c2f9/vllm_ascend/worker/v2/model_runner.py)，需要私仓权限。
- [开发进度和验证标记](flashmla_development_status.md)、[发布交接](flashmla_handoff.md)。
