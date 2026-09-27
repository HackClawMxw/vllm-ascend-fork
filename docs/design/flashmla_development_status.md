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
| 02–07 模型与图 | 开发中；完成后补具体提交和范围 | Prefill/Decode 分流、cache、metadata、capture/replay |
| 08–09 扩展配置 | 基础混部之后逐项处理，不能假设支持 | DSpark、DCP 等实际部署组合 |

探测脚本 `tools/flashmla_probe.py` 仅用于合成输入的包预检，不证明服务进入新分支。`--execute` 才实际运行算子；不提供数值容差时只报告误差，不宣称精度通过。服务验证还需要路由与图生命周期证据，待完整代码后按发布机 skill 集中执行。

主力机 CPU 环境为独立 `.git/flashmla-dev-venv`，不随 PR 分发，也不代表部署镜像。相应命令：`python -m pytest --confcutdir=tests/ut/attention tests/ut/attention/test_flashmla_contract.py -q`。该命令隔离引擎初始化，只测试 adapter，不属于完整 UT/ST。
