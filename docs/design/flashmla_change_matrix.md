# 1 号 / 2 号方案逐项对照与本次改动清单

本文是接入前的源码分析，不是已经实现的改动列表。固定 SHA、调用链和参数待确认项见 [总体分析](flashmla_tiling_oldmain.md)，执行顺序见 [交接流程](flashmla_handoff.md)。

## 比较口径

1 号是大范围 K3/A5 集成，2 号 PR 自身聚焦缓存协议；2 号所在底座本来就有 FIA、DCP、DSpark、图模式等功能。因此“2 号 PR 没有新增”不等于“2 号分支没有这个能力”。

逐项判定使用三组比较：

| 比较 | 范围 | 用途 |
| --- | --- | --- |
| 1 号自身增量 | `990243c4 → de31c53d`，761 个文件 | 区分参考 PR 新增的内容与其原有底座 |
| 2 号自身增量 | `7f6c984b → a583897e`，13 个文件 | 明确本次必须保留的缓存改动 |
| 两个 head | `a583897e → de31c53d` | 定位实际差异，不能直接整体移植 |

默认 Git rename detection 下两个 head 有 1,474 个文件差异；[完整索引](flashmla_diff_inventory.json)为便于按路径读取，固定使用 `--no-renames`，将重命名拆成删除/新增，共 1,530 条 head 比较记录。另包含 761 条 1 号自身增量和 13 条 2 号自身增量。索引覆盖全部文件，但不是“已对所有 native kernel 和无关模块完成语义审查”的声明。

## 2 号自身的 13 个文件

下表的“保留”指新接入应在这些语义之上扩展，不代表这些文件后续绝对不能修改。

| 文件 | 2 号增量 | 本次处理 |
| --- | --- | --- |
| `vllm_ascend/device/hardware_profile.py` | `MLA_FLASH` 硬件能力标识 | 保留；与新包可用性检查分开 |
| `vllm_ascend/attention/utils.py` | 本地 Q head 集合 `{8,12,64,96}` | 保留现有布局选择；算子支持范围以新文档为准 |
| `vllm_ascend/attention/mla_v1.py` | 单 Tensor fused cache 转为 NoPE/RoPE 零拷贝 slices；限制既有 fused prolog 路径 | 在 Flash 路由处分支；保持 FIA 原有 slices 与保护条件 |
| `vllm_ascend/worker/model_runner_v1.py` | 单 raw MLA 分配与 fused/component views | 保留 MRv1 布局及物理归属 |
| `vllm_ascend/worker/utils.py` | 单 backing 判定、页 stride view、strided COW、共享页 zeroer 判定 | 直接复用；不要换成 1 号的页组织 |
| `vllm_ascend/worker/v2/attn_utils.py` | MRv2 allocator/reshape 使用 BBND fused 或 component-major views | 保留为 Flash 输入的来源 |
| `vllm_ascend/worker/v2/model_runner.py` | 接入 tuple-aware zeroer | 保留初始化与生命周期 |
| `vllm_ascend/worker/v2/utils.py` | `AscendV2KVBlockZeroer` 遍历逻辑组件 | 保留，单独验证组件及相邻存储 |
| `tests/ut/device/test_hardware_profile.py` | 硬件能力断言 | 保留回归 |
| `tests/ut/worker/test_attn_utils_v2.py` | 分配、布局、stride 等覆盖 | 保留并追加真实包消费检查 |
| `tests/ut/worker/test_fused_mla_cache_lifecycle.py` | fused/component zeroer 元数据与 COW payload 检查 | 保留；不将 CPU 元数据断言当成 NPU 清零证明 |
| `tests/ut/worker/test_model_runner_v1.py` | MRv1 fused cache 回归 | 保留 |
| `tests/ut/worker/test_model_runner_v2_mamba.py` | hybrid runner 测试状态适配 | 保留 |

## 非连续布局：具体哪里不同

约定 P=kernel page 数、B=每页 token 数、N=KV head 数、D=最后一维。

| 项目 | 2 号当前分支 | 1 号 Flash 路径 | 本次决定 |
| --- | --- | --- | --- |
| BF16 缓存外观 | `[P,B,N,576]` BBND | backend 声明 BNBD；MLA forward 常消费 `[P,B,576]`，再加 N 轴传算子 | 延续 2 号 BBND，等待包确认 `PA_BBND` |
| token 内组织 | fused 为每 token 的 512 维 latent 接 64 维 positional | BF16 同样融合 latent/positional 数据 | 数据语义可参考，轴顺序不可照搬 |
| 非 Flash/FIA 布局 | component-major NoPE/RoPE 分量，同页 backing | 全局 Flash 开关会改变其他 attention cache 路由 | 保持 2 号按能力/head 数选择的协议 |
| 页跨度和起点 | `make_page_strided_cache_view` 显式保留 page stride 与 storage offset | `flash_kv_cache.py` 有自有 kernel-page view 变换 | 不复制 1 号变换；读写都从 2 号 views 出发 |
| manager/kernel block | 以既有 ratio 划分物理 slot | 有面向 BLHNC/LBHNC 的描述符变换 | 验证编号、ratio 和地址，不替换 allocator |
| V1 清零 | fused/shared-component 按物理页元数据处理 | 有 page stride 与 owned payload 分离的 Flash 清零路径 | 保留 2 号，补相邻槽和 padding 测试 |
| COW | 对 dim0 `unflatten`，复制每个 kernel block 的逻辑 payload | 不能用 1 号整体 worker 覆盖 2 号 | 保留；不宣称复制所有 padding 字节 |
| MRv2 清零 | tuple-aware wrapper | 参考分支有不同 Flash/cache 处理 | 按 MRv2 自己的 zeroer 验证 |
| 量化 | 本任务基线的 BF16 fused/component 合约 | C8 另有 FP8 latent 与 BF16 positional 分区、640-byte 容量 view | 首轮不导入 C8 合约 |

N=1 时 BBND/BNBD 的单例轴可能使元素地址看似等价，但 shape、stride 与 binding 检查仍有区别。只改字符串不是完成适配；对持久 cache 做 `.contiguous()` 也不是允许的解决方案。

## 执行链逐项差异和拟改动

处理标记：“接入”表示本任务必须落实；“复用”表示已有能力；“分阶段”表示相关但需要单独验证；“暂不导入”表示不属于当前最小接入。最终参数由外部算子文档确定。

| ID | 2 号现状 | 1 号对应实现 | 本次拟改动 / 处理 | 主要文件 |
| --- | --- | --- | --- | --- |
| F01 | MLA 已有 FIA backend；缓存 capability 不等于选中 Flash | 全局 Flash 开关联动 platform、MLA 与 GQA | 接入外部包能力检查和 MLA 路由；避免误改其他 backend | `platform.py`、`envs.py`、`attention/mla_v1.py` |
| F02 | 没有本次新包的 FlashMLA adapter | MLA 使用 `_C_ascend` 自带 binding | 新建最小 adapter，直接遵守外部包 schema；实际文件名实施时确定 | adapter、包加载入口 |
| F03 | 原 FIA 预处理得到分离 Q NoPE/positional | absorbed Q 拼成 576 维；使用实际 head 数 | 接通 Q absorption 与输入 layout，不复制额外 head replication | `attention/mla_v1.py` |
| F04 | 原路径经 slices 写 fused/component KV | Flash 路径 scatter 写融合 cache 的两个分量 | 适配新路径 KV 写入，验证 stride、offset、无效 slot 与 NoPE/RoPE | `attention/mla_v1.py` |
| F05 | FIA metadata 包含 CPU 长度/list 等 | Flash 稳定 buffers、设备长度、schedule | 接入 Flash 专用 metadata；共享现有基础结构 | `attention/attention_v1.py`、`attention/mla_v1.py` |
| F06 | 没有新包的 metadata 生成调用 | Meta 查询容量，实际调用生成 schedule 并刷新固定目标 | 用新包 Meta/容量规则；核对 max length、dtype、空行 | adapter、MLA builder |
| F07 | **已有 `DeviceMetadataExecutor`** | **同一文件、相同 Git blob** | **复用，不新建、不整体覆盖**；补 MLA task provider | `worker/device_metadata.py`、MLA builder |
| F08 | MRv1 已有 provider 收集、submit、forward context 与 release | Flash builder 挂接同一机制 | 扩展 MLA provider；保留 2 号已有生命周期修复 | `worker/model_runner_v1.py` |
| F09 | MRv2 当前没有参考分支的 Flash executor/context 接线 | target executor、builder 收集 task、图外 submit/wait、finally release | 接入 MRv2 target 生命周期，兼顾异常和 buffer fence | `worker/v2/model_runner.py`、`attn_utils.py` |
| F10 | MLA decode 调 FIA v2 | FlashMLA 读 paged KV 与 schedule | 替换选中路径核心 attention；传参按包验证 | `attention/mla_v1.py` |
| F11 | 既有 V 升维、gate、O projection | Flash 输出按 TND/NTD 和 merge 需求分流 | 对齐输出轴、V=512 latent 与现有投影；保留 TP 语义 | `attention/mla_v1.py` |
| F12 | 既有 padding 和 FIA 长度约束 | 额外零长度 request、`token_live`、`slot=-1`、输出 mask | 接入新包允许的 padding 规则，验证空行和残留数据 | builders、输出处理 |
| F13 | FIA graph 参数/handle/workspace 更新 | Flash MLA updater 跳过，稳定 buffer 提供新输入 | 在前置刷新完成后仅跳过选中 Flash 路径；保留其他 backend updater | MLA updater、`compilation/acl_graph.py` |
| F14 | MRv1 FULL 已有 executor/event 基础设施 | descriptor/stage/group 对应 ExternalEvent | 复用并验证 MLA capture/replay，不移植整份 runner | MRv1、executor、forward context |
| F15 | MRv2 capture/replay 还承载既有图实现 | metadata 图外更新，capture 外层进入 context | 插入 Flash 生命周期，保留 2 号已有 UpdatableGraph/PCP 支持 | `worker/v2/aclgraph_utils.py` |
| F16 | speculative 后拷贝 computed lengths 到 CPU，并等待 event | K3 + Flash + PCP1 使用设备真实长度，CPU 留上界 | 分阶段移除已证明多余的同步；先核对所有消费者 | `worker/v2/model_runner.py` |
| F17 | K3 DSpark context 预计算与 `exec_kv_prefill` 调用已有 | Flash 分支支持 context KV 写入 | 扩展 writer 即可时不改模型；检查 target/draft cache ownership | `models/kimi_k3_dspark.py`、MLA writer |
| F18 | DSpark query/capture 已有通路 | draft 独立 executor、positions、causal/query metadata 接线 | 接入 MLA draft 的 metadata 与 capture；不能硬设 causal | `worker/v2/spec_decode/dspark/speculator.py`、`dflash/aclgraph.py` |
| F19 | 基线已有 MLA DCP，包括 FIA history/current 路径 | Flash history/current + output/LSE exchange/merge | 分阶段迁移计算接口，复用分布式语义；验证当前 KV 恰计一次 | `attention/context_parallel/mla_cp.py`、`common_cp.py` |
| F20 | 既有 TP head ownership 与 projection | 额外 Q replication、通信 overlap、融合 gate/O 投影 | 暂不导入性能优化；先保持当前 TP 输出一致 | `models/kimi_k3.py`、MLA impl、Triton helpers |
| F21 | prefill 用已有 FIA/展开路径 | absorbed FlashMLA 和额外 192/128 非吸收 FlashAttn | 先按包支持范围接 absorbed；非吸收路径独立评估 | `attention/mla_v1.py`、`mla_prefill.py` |
| F22 | 已有 MLAPO/其他 fused preprocess policy；fused cache 有保护 | 1 号扩展 NoPE/原生前处理及权重路径 | 首轮普通前处理验证正确，后续按 stride 合约启用融合 | `attention/mla_v1.py`、`ops/mla.py` |
| F23 | 其他 attention 使用自身 backend/cache | 1 号 Flash 开关也接 GQA FlashAttn/head-slot packing | 暂不导入；GQA draft 是否需要属于另一个明确范围 | `attention/attention_v1.py`、platform、draft |
| F24 | 本次尚无外部 C8 合约 | C8 cache、量化准备、native metadata 与输出格式 | 暂不导入，避免改变 2 号 cache 存储协议 | `worker/flash_kv_cache.py`、C8 ops/native |
| F25 | 旧基线已有 PCP/PD 等实现 | 1 号另有通信、传输、启动顺序和相关优化 | 按部署需求逐项验证，不把整套运行环境搬入最小接入 | transfer、runner、platform |
| F26 | 原有 KDA/MoE/Mamba/PP 代码 | 1 号含大量独立优化及修复 | 暂不导入，除非后续能证明是接入的必要依赖 | 对应模块与 native kernels |

F07 的同一 blob 为 `0f5eed70fb957384fb62c769408889a5effc92c4`。F17 的 context 写入调用在两边都存在；需要改变的是算子路径对 cache 的消费方式，不能把整条 DSpark 功能记成新引入。

## 底座差异不能作为移植清单

两个 head 的直接 diff 中可以看到 `compilation/updatable_graph.py`、`dsa_v41.py` 等文件在 1 号一侧缺失。它们与 1 号自身增量的关系必须单独核对；不得据此删除 2 号已有功能。

实现时每个改动应标注 F 编号、依赖、验证证据和是否改变 cache contract；如果真实包要求改动 2 号非连续布局，应先记录具体不兼容项，不能悄悄引入 copy/repack 作为完成标志。

当前结论是：非连续布局已经做了针对性对照；attention 前后处理、metadata、图、DSpark、DCP 和同步点也已完成以上源码层面的分析。尚未完成的是新包合约核对、运行实现及真实设备验证。
