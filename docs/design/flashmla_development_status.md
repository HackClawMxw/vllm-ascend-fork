# 主力机 / 发布机协同开发记录

本机为主力机，负责代码实现、静态检查和提交；发布机负责环境、真实算子、图和服务验证。用户最新要求：以代码开发为主，逐模块完成代码后集中验证，避免主力机扩展过多测试。已写的接口检查保留，不能将 CPU 合约通过写成 NPU 通过。

## 固定输入

- 代码基线：2 号 `a583897e0c67d9c23288686728124b04b511a462`。
- 实现参考：1 号私仓 `de31c53dc5b94ff246b17aa198404a082162c2f9`。`dccb903d4` 是会话归档文档提交，不是 1 号代码。
- 新增输入：[PR #10 算子合约留言](https://github.com/Henry-Avery/vllm-ascend/pull/10#issuecomment-5856540746)。
- 算子源码/文档：CANN ops-transformer `09ef2cd6de0e6cfaf2c061ea138a283ef0b4c73c` 的 `attention/flash_mla_with_kvcache/docs/torchapi_flash_mla_with_kvcache.md` 和同目录 `torch_extension/flash_mla_with_kvcache.py`，主力机已通过 Git 拉取核对。
- 发布镜像：`sha256:33ab5c4a19544b3e91afe65e339c21d250094bb1205b9bac873162b992cd13ed`。发布机须核对本地实际 ID；镜像 ID 不证明包已安装。
- GitHub 协作入口：[PR #10](https://github.com/Henry-Avery/vllm-ascend/pull/10)，代码推送共享 head，同步更新[目标 PR #6](https://github.com/HackClawMxw/vllm-ascend-fork/pull/6)。发布机按候选 SHA 回报，避免两机覆盖开发分支。

## 已冻结的接口映射

公开导出为 `cann_ops_transformer.ops.flash_mla_with_kvcache` 和 `flash_mla_with_kvcache_metadata`。Q 为 TND、576 维，输出为 NTD、512 维；BF16/FP16，本地 Q heads 为 64/96、KV heads 为 1、block size 为 128。metadata 和主算子均显式传 `max_seqlen_q=-1`、`max_seqlen_kv=-1`、一致的头数/长度/mask。TND 主动提供 int32 的 `cu_seqlens_q` 和 metadata，`seqused_q` 可选。mask=0 不传 mask；mask=3 传 int8 2048×2048 上三角为 1、其余为 0 的 mask。

adapter 支持明确指定 `PA_BBND` 或文档拼写 `PA_NZ`；模型接入保留 2 号 BBND，不将持久 cache 转成 NZ。长度、block table、metadata 由调用者保证内容正确；接口层仅做不触发设备同步的结构检查。

**待发布机核实：** 用户目标允许 KV 第 0、1 轴非连续，固定文档的 BBND 只承诺第 0 轴。adapter 原样传递 Tensor/stride/offset，由真实包验证，失败不会连续化或静默改走 FIA。Meta 源码会查询当前 NPU 核数，必须选定设备后调用，不照搬其容量常量。

## 逐项进度

| 部分 | 主力机结果 | 发布机待验证 |
| --- | --- | --- |
| 01 外部接口 | `attention/flashmla.py`：延迟加载、固定参数、结构检查、保留缓存别名；29 项 CPU 合约检查通过 | 实际包/schema、Meta、非连续两轴/offset、输出与 LSE |
| 02–07 模型与图 | 已写候选，见下方逐提交记录；尚未 NPU 验收 | Prefill/Decode 分流、cache、metadata、capture/replay |
| 08–09 扩展配置 | DSpark 候选已写；DCP/PCP 暂显式拒绝 | DSpark、DCP 等实际部署组合 |

探测脚本 `tools/flashmla_probe.py` 仅用于合成输入的包预检，不证明服务进入新分支。`--execute` 才实际运行算子；不提供数值容差时只报告误差，不宣称精度通过。服务验证还需要路由与图生命周期证据，待完整代码后按发布机 skill 集中执行。

主力机 CPU 环境为独立 `.git/flashmla-dev-venv`，不随 PR 分发，也不代表部署镜像。相应命令：`python -m pytest --confcutdir=tests/ut/attention tests/ut/attention/test_flashmla_contract.py -q`。该命令隔离引擎初始化，只测试 adapter，不属于完整 UT/ST。

## 当前代码候选（运行验收尚未完成）

用户已将流程调整为主力机集中完成代码、发布机随后集中验证。以下是代码落地记录，不替代设计文档的验收条件。

| 提交 | 单独完成的部分 | 作用与边界 |
| --- | --- | --- |
| `9ef74992a` | 外部包 adapter、合成输入 probe | 显式传接口参数，不复制持久 KV，不静默回退 |
| `d8786a95a` | 开关与真实阶段分流 | `VLLM_ASCEND_ENABLE_FLASH_MLA=1`；默认 0；短 Prefill 仍是 Prefill |
| `378615ec4` | builder 持有的稳定缓冲与 metadata task | 用包的 Meta 获取容量；设备真实长度生成 schedule；复用现有 executor |
| `4408d8f1c` | Decode 计算、共享 cache、MRv1 | 原 Q/KV 前处理及 writer → FlashMLA → 原 V/O/gate；MRv1 等待发生在 writer/rope 之前 |
| `0a6d1a320` | MRv2 target 与图生命周期 | capture/replay 外刷新 metadata；消费者结束后记录复用 fence；含 Prefill 的请求不得误命中 Decode FULL 图 |
| `a8da46c20` | DSpark draft 生命周期 | 独立 executor；按 draft causal 选择 mask；保留 context writer；不套用 FIA 的长度改写 |

**支持范围：** 本轮代码为混部、PCP=1、DCP=1、无 KV transfer、未量化 BF16/FP16、BBND、block128、local Q heads64/96、KV heads1、latent512+positional64。硬件须具备 MLA_FLASH capability。超出范围显式报错。Adapter 的 PA_NZ 合约检查不代表模型已经接入 NZ。TP 配置必须满足本地头数约束；TP/SP、DSpark 实际效果仍待发布机验证。DCP/PCP 扩展、PD 分离和新主线迁移不在这批候选中。

**相对 1 号的主动差异：** Prefill 保留 2 号 FIA 和 chunked/prefix 历史读取；保留 2 号单 backing/BBND strided view、zero/COW；使用公开外部包接口和本任务 `max_seqlen_*=-1` 合约；不复制 1 号 native binding 或整份 runner。现有 FIA host mirrors 保留，未在没有性能证据时删除同步。

**Prefill 与 cache 顺序：** 仍由 `mla_preprocess_prefill` 调 `exec_kv_prefill` 先写缓存，再执行 FIA。attention 之后的 reshape 是输出投影所需，不是重新排布 KV。DSpark 直接调用 writer 时也将 fused Tensor 暴露为原逻辑 slices。

**图路由：** MRv1 在 dispatch 排除含真实 Prefill 的 FULL 图；MRv2 若该批被选中 Decode FULL 图则退到 eager，已有 PIECEWISE 选择保持。纯 Decode 使用固定 token 容量的缓冲，补齐请求 `used_q=0`、长度 0、slot=-1，末尾 cu 等于物理 T。此实现仍需通过真实算子的零长度/padding 与重复 replay 验证。

## 发布机：证明实际进入了新代码

先记录候选 `git rev-parse HEAD`、镜像完整 ID、包版本/schema、实际 import 文件路径，按本机启动 skill 启动。显式设置开关后检查下列标记；仅看到服务启动成功或文本正常不够。

| 标记/观察点 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| `[FlashMLA] configured` | 模型实例启用且通过静态门槛 | 真实 attention 已执行 |
| `[FlashMLA] metadata_external` | 进入新 metadata refresh；`deferred=True` 表示经 executor 提交 | 生成内容正确或每轮都更新 |
| `[FlashMLA] prefill_fia: layer=...` | 此层实际经过保留的 FIA Prefill 分支 | 每个 Prefill 用例都覆盖 |
| `[FlashMLA] decode_external: layer=... stride=... offset=...` | Python 路径已调用外部主算子；可对照原 cache 的 stride/offset | 异步 NPU 已成功完成、数值正确或每次 replay 都执行 |

标记按实例/层首次输出，避免热路径逐 token 日志；图 replay 通常不会再次执行 Python。需要证明 replay 命中时，使用发布机 profiler 的真实算子事件，并在同一个图档位改变 lengths、block table、请求顺序后比较 eager/graph 结果。不得用一次初始化日志代替 replay 证据。

若定位失败，完成代码后的临时打桩/断点依次设在：

1. `split_flashmla_requests`：真实阶段、CPU 边界与分流计数；含一 token 的 prompt 尾段。
2. `FlashMLAMetadataBuilder.build.refresh`：设备长度/cu/used_q、schedule 地址；确认每轮更新、下一轮等待上一轮复用 fence。
3. `exec_kv_prefill` / `exec_kv_decode`：原 backing 的指针、stride、offset、slot=-1；确认一次写入且非目标区域不变。
4. `_forward_external_flashmla` / `FlashMLAAdapter.attention`：Q[T,H,576]、原 BBND cache、输出[H,T,512]；与 FIA 基线比较 V 升维和 O 投影结果。
5. MRv2 `build_attn_metadata`、MRv1 executor wait、target/draft graph replay：确认 metadata 在 capture 外刷新且事件顺序正确。

调试打印设备内容、同步或断点仅在发布机定位时临时启用，不提交到生产热路径。先跑 eager 的 Prefill→Decode、短 Prefill、prefix/chunked、mixed，再跑同图变输入，最后 DSpark；具体模型路径、启动命令、精度阈值按当地 skill，不在主力机猜测。

本机已有 31 项 CPU 接口/阶段检查通过；新增运行路径仅做静态检查，未在 NPU、服务或图上验证。`tools/flashmla_probe.py` 只能证明合成输入接口，不能代替上述服务证据。包对第 1 轴非连续的支持仍是必须实测的阻断风险。

提交前检查：全部改动 Python 的 AST 与定向 Ruff 检查通过，31 项 CPU 检查通过。已运行仓库要求的 `bash format.sh ci`：其他 hooks 通过，整体未通过；原因是本机缺少 shellcheck，且 ruff-format 修改了基线 `worker/utils.py` 的两处格式。该无关格式修改已恢复，不混入本功能。没有修改 shell 文件，不将该结果写成全量 CI 通过。
