# FlashMLA 另一台机器的执行交接流程

当前阶段只确定流程与差异。外部算子说明、安装包、模型/硬件信息以及启动测试 skill 在另一台机器，后续由用户补充。不在本机猜测安装步骤、设备地址、模型路径或启动命令。

**接手机器必须遵守的当前需求：PD 混部；prefill 保留 FIA，decode 接外部 FlashMLA；沿用 2 号非连续缓存。PD 分离暂不推进。** 禁止按 1 号全局 Flash 分支将 prefill/decode 一起切换；mixed batch 按实际阶段分别执行，并共用同一份缓存。此前 absorbed FlashMLA prefill 的计划已取消。

## 接手入口

- 工作仓库：`Henry-Avery/vllm-ascend`。
- 工作分支：`codex/flashmla-tiling-oldmain-v2`；先核对远端最新 SHA，读取已有提交，不从旧 draft 复制未验实现。
- 旧基线：`a583897e0c67d9c23288686728124b04b511a462`，目标 `HackClawMxw/vllm-ascend-fork:0913main`。
- 目标 draft：[PR #6](https://github.com/HackClawMxw/vllm-ascend-fork/pull/6)；个人备份：[PR #10](https://github.com/Henry-Avery/vllm-ascend/pull/10)。两个 PR 共享 head，继续推送同一分支即可更新。
- 1 号固定参考：`maoxx241/vllm-ascend-rfc16468-private` 的 `de31c53dc5b94ff246b17aa198404a082162c2f9`；需要该私仓读取权限。不能直接替换成当时最新 head。
- 阅读顺序：[会话结论](flashmla_session_decisions.md) → [总体分析](flashmla_tiling_oldmain.md) → [逐项清单](flashmla_change_matrix.md) → 外部算子文档 → 当地启动测试 skill。

当前使用 `kv_transfer_config=None` 的混部场景，不扩大 2 号新布局的 transfer 支持范围。prefill FIA 与 decode FlashMLA 必须消费同一缓存协议，不能靠阶段切换时重新分配/复制持久 cache 来隐藏不兼容。

先读取目标 checkout 的 `AGENTS.md` 和用户提供的 skill，记录 skill 路径/版本。保留目标机器已有工作，使用合适的独立 checkout；本文不要求移动、覆盖或清理已有运行环境。本机分析未套用某台机器的固定目录或服务命令。

## 阶段与完成条件

| 阶段 | 输入与动作 | 必须留下的结果 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| P0 环境核对 | VA/vLLM SHA、硬件、CANN、torch/torch-npu、算子包、模型配置、当地 skill | 环境清单；现有服务/checkout 状态；实际路径 | 包/文档版本可对应，基线明确 |
| P1 合约冻结 | 填总体分析第 8 节参数表；核对 decode 的 BBND、stride、offset、mask、长度和 Meta；确认 prefill FIA 与其共享缓存 | 参数映射表、阶段路由规则、支持矩阵、未知项、真实包导入/schema 结果 | 能用 2 号缓存协议完成两个阶段的读写；不兼容项有明确处理方案 |
| P2 基线测量 | 按当地 skill 启动未接新算子的 2 号基线 | 精确启动命令、配置、输出、精度和性能基准；失败记录 | 基线问题与候选问题可区分 |
| P3 adapter/eager | 按 F01–F12 接通 decode FlashMLA，保留 prefill FIA；CPU 合约检查 + NPU 算子/模块比较 | 参数/alias、数值、cache guards；纯 prefill=FIA、纯 decode=FlashMLA、mixed 分流、prefill→decode 交接、prefix/chunked/短 prefill 结果 | 路由和数值同时符合需求；不能仅凭模型输出正常验收 |
| P4 图 | 按 F13–F15 分别接 MRv2 和所需 MRv1；同档改变输入多次 replay | 地址稳定、event/fence 顺序、graph/eager 一致性及变输入证据 | 无旧 metadata、越界、死锁或 buffer 覆盖 |
| P5 speculative | F17–F18：context KV、draft query、target/draft 独立生命周期 | acceptance/rejection、noncausal、padding、下一轮长度、图组合测试 | context writer 与 query attention 均通过 |
| P6 并行/同步 | 按混部部署范围做 DCP/TP/SP 和所需 PCP，之后移除已证实多余的 host sync；prefill 仍走 FIA | rank/shard/LSE/current KV 正确性；同步点与性能证据 | 功能通过后再报告性能；PD 分离不在当前阶段 |
| P7 发布草稿结果 | 更新代码、文档、实际命令和报告，按仓库/skill 做检查，推共享 head | 两个 draft 对应同一 SHA；已测/未测边界 | 旧基线验收后，另行做新主线迁移 |

步骤是推进顺序，不是对新包支持能力的预判。P0/P1 缺少资料时，可以继续源码分析，但不将猜测接口写成最终实现。基础阶段暂用 PCP=1/DCP=1；如果用户的首轮验收要求包含其他配置，应调整阶段范围并记录。

## 验证证据的记录格式

每个用例记录：候选 SHA、基线 SHA、包版本、模型/并行配置、执行命令、输入规模、预期值、实际结果、日志/产物路径、是否重复验证。

至少区分以下证据，不能互相替代：

1. 文档/schema/Meta 合约检查。
2. CPU mock 或 Tensor alias/参数路由测试。
3. NPU 真实算子正确性，包括 noncontiguous/offset/无效 slot。
4. MLA 模块端到端，包括投影、cache write、attention、output。
5. capture/replay，尤其同图不同 lengths/table/request order。
6. 实际服务的模型精度、稳定性及性能。

精度阈值和实际启动/测试命令以外部文档及当地 skill 确定。持久 cache 的正确写入与非目标区域保护应做精确比较。CPU 测试通过不能写成 NPU 或服务通过；参考 PR 的历史测量不能写成本候选结果。

当前环境只运行了文档检查，不具备本任务的 NPU 验收证据。本机 `format.sh ci` 因缺少 `pre-commit` 未能运行，接手机器需要按仓库和 skill 补齐适用检查。

验收增加路由证据：确认 prefill 请求实际调用 FIA，decode 实际调用 FlashMLA；mixed batch 两部分均有正确调用。特别检查短 prefill 的阶段识别，不能仅依据 query length/decode threshold 推断阶段。DSpark context 写入不等于 prefill attention，draft query 的路由按独立语义核对。

## 可交给另一台机器的任务说明

> 继续 `Henry-Avery/vllm-ascend:codex/flashmla-tiling-oldmain-v2` 上的 FlashMLA tiling 下沉任务。当前先验证 PD 混部：prefill 必须保留 FIA，decode 接指定外部 FlashMLA，mixed batch 按阶段分流并共用 2 号非连续缓存；不要照搬 1 号将整个 forward 切到 FlashMLA，PD 分离暂不推进。先读取 `docs/design/flashmla_handoff.md`、总体分析和逐项清单，再读取我提供的算子文档与本机启动测试 skill。以 `a583897e` 的 2 号缓存协议为基线，参考固定 `de31c53d` 的 1 号 decode/metadata 流程。复用已有 DeviceMetadataExecutor，逐项适配 MLA/MRv2/图/DSpark，不覆盖 allocator，不将两条分支的底座差异当成必需改动。先冻结合约和验证基线，再按阶段实现与测试，必须提供 FIA prefill→共享 cache→FlashMLA decode 的路由和正确性证据；将真实结果、未覆盖项和最小代码增量推到同一分支，更新 PR #6 和个人备份 PR #10。最终再适配 VA 新主线。
