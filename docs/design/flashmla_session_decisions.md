# FlashMLA 会话结论与接手约束

整理日期：2026-09-27。本文提炼本次会话中可执行的需求、源码结论和纠正项，供另一台机器拉取 PR 后接手。它不是聊天逐字稿，也不是新算子实现或实机验收报告。后续需求变化应同步更新本文、设计和交接流程，避免只留在聊天中。

## 1. 已确定的目标

| 项目 | 当前决定 |
| --- | --- |
| 最终目标 | 合入 VA 新主线；旧基线验收后另做迁移 |
| 本阶段目标 | 基于 2 号 PR #16456 的固定 head 开发，先向其 fork 的 `0913main` 提交 draft |
| 执行范围 | PD 混部，即同一部署同时处理 prefill/decode；PD 分离暂不推进 |
| Prefill | 保留 FIA；包含纯 prefill、chunked、prefix-hit 和短 prefill |
| Decode | 在明确支持并选中的 dense MLA 路径接入外部 FlashMLA |
| KV cache | 保留 2 号非连续布局、分配、页映射、zero/COW 和存储别名；两个阶段共享同一份持久 cache |
| 参考方式 | 参考 1 号 decode 前后处理、metadata 下沉及图生命周期；逐项适配，不整体移植 |
| 算子来源 | 用户稍后提供的外部包与文档；不能将 1 号仓内 native binding 当成该包接口 |
| 实施方式 | 一点一部分，完成对应验证和记录后再改下一处；每个可独立审查的改动单独提交 |
| 当前交付 | 源码分析、方案设计、另一台机器的交接材料；尚无 runtime 改动 |

## 2. 仓库与版本

- 1 号：[私仓 PR #3](https://github.com/maoxx241/vllm-ascend-rfc16468-private/pull/3)，固定 `de31c53dc5b94ff246b17aa198404a082162c2f9`，读取需要私仓权限。
- 2 号：[上游 PR #16456](https://github.com/vllm-project/vllm-ascend/pull/16456)，固定 `a583897e0c67d9c23288686728124b04b511a462`。
- 配套 vLLM 参考版本：`84030bbe3d74d99bad477a3d2e37a973ccd8865c`；部署前仍须检查实际环境。
- 共享 head：`Henry-Avery/vllm-ascend:codex/flashmla-tiling-oldmain-v2`。
- 目标 draft：[HackClawMxw PR #6](https://github.com/HackClawMxw/vllm-ascend-fork/pull/6)，base=`0913main`。
- 保存 draft：[Henry-Avery PR #10](https://github.com/Henry-Avery/vllm-ascend/pull/10)，base=`codex/flashmla-oldmain-base-a583897e`。
- 用户已确认新建这两个 draft；后续向共享 head 推送，两个 PR 一起更新。已有旧 draft 不作为本次实现基础。

固定引用和差异统计见[总体分析](flashmla_tiling_oldmain.md)与[完整索引](flashmla_diff_inventory.json)。本地私仓参考 worktree、`.git/research` 文件及聊天历史不会随普通 PR 拉取而出现；接手所需结论必须保存在跟踪文件中。

## 3. 已纠正的假设

1. 之前计划涉及 absorbed FlashMLA prefill，用户已明确取消。不能把“接 FlashMLA”解释为 P/D 都使用 FlashMLA。
2. 用户曾要求调查 PD 分离，随后明确暂不推进。当前只保留 eligibility 边界：2 号新增 single-backing 布局排除 KV transfer 配置；这不等于整个分支没有旧 PD 功能。
3. fused cache 只表示存储组织，不表示 attention 已切换到 FlashMLA；2 号仍使用 FIA。
4. 1 号包含大量 K3/A5、GQA、量化、prefill 及通信优化。两分支底座不同，head diff 不是待移植清单。
5. `DeviceMetadataExecutor` 已存在于 2 号，且两个参考 head 中内容相同。应补 MLA provider/runner 接线，不能重复建设 executor。
6. 2 号现有 decode/prefill 分组包含按计算规模划分的语义，不能直接当作用户要求的真实生成阶段。短 prefill 必须单独审查。

## 4. Prefill、reshape 与写缓存的准确顺序

2 号的正常预处理链为：

```text
Q/KV 投影、归一化、按模型处理 RoPE
  → 将当前 token 的压缩 KV 按 slot_mapping 写入持久 cache
  → kv_b_proj 展开本次 Prefill 所需的临时 K/V
  → FIA 计算当前块
  → 有历史上下文时读取历史 cache、展开 K/V、FIA 计算并合并
  → attention 输出展平 head 维
  → gate（如启用）和 O 投影
```

这里的 KV 写入发生在 attention 之前；并不存在“FIA 算完，再把 attention 输出排成 KV cache”的步骤。`mla_preprocess_prefill` 先调用 `exec_kv_prefill`，再调用 `kv_b_proj`。

| 操作 | 处理对象与作用 | 本次决定 |
| --- | --- | --- |
| `make_page_strided_cache_view` / `as_strided` | 初始化时建立带 page stride 和 offset 的存储视图 | 保留 2 号实现 |
| `_exec_kv_no_rope` / `reshape_and_cache` | K3 无旋转分支：归一化 latent，连同 positional 分量按 slot 写入两个逻辑 cache views | 复用现有 writer；不另加整个 cache 的排布转换 |
| `exec_kv_prefill` 中融合 norm/RoPE/cache 算子 | 带 RoPE 的适用分支中合并预处理和 cache write | 按实际模型分支验证；不能假定全部走 scatter |
| `kv_b_proj(...).view(...).split(...)` | 压缩 latent 升维成 Prefill FIA 临时 K/V | 保留；不是持久 cache 的物理布局 |
| `_forward_prefill` 尾部 `attn_output.reshape` | 合并 attention 输出的 head 维，为 O 投影准备输入 | 保留；不负责写 KV |
| connector 的 written/save hook | 通知缓存就绪或触发 connector 行为 | 不能将函数名误认为实际 KV write 的位置 |

在 token-fused 路径上，`forward` 将原 fused Tensor 切成 latent/positional 两个零拷贝 views；两者仍指向同一 backing。Flash Decode 需要原 fused view，旧 FIA/writer 继续使用对应逻辑 views。component-major 布局不能仅凭两个分量具有相同 shape 就伪装成 token-fused 布局。

缓存里保存的是压缩 KV；FIA Prefill 消费的是展开后的临时 K/V。对临时 Q/K/V 的必要连续化和对持久 cache 的 `.contiguous()` 必须区分，后者会新建存储，破坏写回别名和图地址。

源码证据以 2 号固定 SHA 为准：

- [cache view 构造](https://github.com/HackClawMxw/vllm-ascend-fork/blob/a583897e0c67d9c23288686728124b04b511a462/vllm_ascend/worker/utils.py#L112)。
- [Prefill 预处理与写入顺序](https://github.com/HackClawMxw/vllm-ascend-fork/blob/a583897e0c67d9c23288686728124b04b511a462/vllm_ascend/attention/mla_v1.py#L1956)。
- [FIA 与输出 reshape](https://github.com/HackClawMxw/vllm-ascend-fork/blob/a583897e0c67d9c23288686728124b04b511a462/vllm_ascend/attention/mla_v1.py#L1332)。
- [短 extend 分流语义](https://github.com/HackClawMxw/vllm-ascend-fork/blob/a583897e0c67d9c23288686728124b04b511a462/vllm_ascend/attention/utils.py#L358)。

## 5. 还不能下结论的项目

| 缺失输入 | 需要它解决的问题 |
| --- | --- |
| 外部包文档、版本、导出 schema | Q/KV/输出、mask、长度、schedule、Meta/Fake 的真实合约 |
| stride/offset 与 layout 约束 | 是否可直接读取 2 号 BBND、页间 padding、非零 offset；是否允许存储别名换轴 |
| 部署机器的启动测试 skill | 实际工作目录、容器/依赖、启动顺序、日志位置与验收命令 |
| 首轮模型、硬件与部署参数 | runner、dtype、heads、block、TP/DP/PP、speculative、graph mode 的测试组合 |
| 实机验证 | writer/gather/Flash reader 是否正确共享非连续 cache；eager/graph 数值、稳定性和性能 |

源码中的 API 和 shape 示例均不能代替新包文档。文档未到前可完成架构设计、调用链核对与测试设计，不能以猜测接口宣称已经接入。PD 分离、C8、额外 Q replication、融合 gate/O-proj 等不自动进入当前范围。

## 6. 接手原则

先读[交接流程](flashmla_handoff.md)，再按[逐项差异](flashmla_change_matrix.md)定位代码。每项记录“改动、原因、验证、结果、未覆盖项”，完成后再进入下一项；mock、NPU 算子、模块、图与服务验证分开报告。

当前只有文档验证和源码结论。现有 PR 保持 draft，不把参考 PR 的性能记录或旧 draft 的运行结果写成本分支验收结果。
