# 发布机固定候选与可选诊断版

## 先验证的版本

首轮固定 `698d00e7c86eb8a9175c92d384497b28f5b798eb`；也可使用只多了对比文档的 `bc83ce0ed666b69eadaf27c3be740655547d2fca`，两者运行代码相同。不要在首轮过程中自动跟随分支最新提交。记录实际 SHA、镜像 ID、真实包/schema/导入路径、启动命令及结果。

本诊断提交在上述候选之后单独增加观测点，默认关闭。失败需要定位时，在独立 checkout 切到包含本文件和 `VLLM_ASCEND_FLASH_MLA_TRACE` 的诊断提交，重启对应 worker 后开启：

```bash
export VLLM_ASCEND_ENABLE_FLASH_MLA=1
export VLLM_ASCEND_FLASH_MLA_TRACE=1
```

实际启动命令、模型和设备配置按发布机已有 skill。诊断完成关闭 TRACE 后重启再测性能；本开关会逐 metadata build 和逐次 Python attention 调用记录日志，影响主机耗时。它不读取 tensor 内容，不额外 `.cpu()` / `.item()` / synchronize，不改变 layout、mask、lengths、算子选择或缓存写入。日志含本地设备地址、层名和形状，不含请求文本/token 内容。

## 标记与证据边界

| event | 所在位置 | 可以判断 | 不代表 |
| --- | --- | --- | --- |
| `metadata_prepared` | builder 收集本轮任务 | builder/step、真实 Decode 数、mixed、物理 T、rows、mask 和稳定缓冲地址 | task 已运行 |
| `metadata_refresh_begin` | executor 执行 refresh 的入口 | 本轮 task 进入；可与 prepared 的 builder/step 配对 | 设备已经完成 |
| `metadata_enqueued` | 外部 metadata 调用及 schedule copy 返回后 | 元数据生成与拷贝已入队；记录设备源 lengths/cu 地址和目标 schedule 地址 | schedule 内容正确或 metadata kernel 已成功完成 |
| `decode_call` | 外部主算子调用前 | layer、schedule/Q/cache 指针、stride/offset；可关联 metadata | 主算子调用成功 |
| `decode_enqueued` | 外部主算子返回后 | Python dispatch 已返回，输出形状可见 | 异步 NPU 无错误或数值通过 |
| `prefill_enqueued` | 原 FIA Prefill 返回后 | 此层本次 Prefill 仍调用原实现 | 设备完成或历史 context 正确 |
| `target_replay_submit` | MRv2 target replay 入口 | 即将提交哪个图 descriptor；应在该轮 metadata 入队/wait 后 | 图内每个算子实际执行 |
| `draft_replay_submit` | MRv2 DFlash/DSpark 图入口 | draft metadata rebuild 后即将提交 replay | DSpark acceptance/rejection 正确 |

已有 `configured`、`metadata_external`、`prefill_fia`、`decode_external` 首次日志继续保留。builder/step 是进程内诊断编号；不同 TP worker 的地址和编号不能相互比较，先按进程/设备区分。不同 graph token 容量、causal 或 builder 可以使用不同缓冲；同一档位、同一 builder 才要求 schedule/Q/lengths 等地址复用。

metadata refresh 发生在图外，所以 replay 时仍应出现本轮 metadata 标记。FULL replay 不重新执行 Python attention，`decode_call` / `decode_enqueued` 不会每轮出现，这是正常行为。MRv1 复用已有 ExternalEvent 协议，本次没有修改通用 executor 或加入 MRv1 replay logger；结合 metadata 标记、原图日志和 profiler 检查。

**验收需要真实设备证据：** profiler 应看到外部 FlashMLA metadata 和主算子事件，实际名字以安装包/profiler 为准；同步后的运行结果需无错误、数值符合阈值。同图改变 lengths/table/请求顺序后与 eager 比较。Python 日志和地址稳定单独不能证明 tiling 下沉正确，更不能证明 CPU 同步开销已清除。

## 必要时的断点和临时打桩

优先拿固定候选重现，再开启本诊断开关；仍不能定位时，仅在发布机的最小 eager 用例中增加内容 dump/断点。不要在正式 capture 内打交互断点或临时插入设备同步；多卡断点可能让其他 rank 等待通信，先缩小复现配置。

| 症状 | 首个检查位置 | 对照数据 |
| --- | --- | --- |
| Prefill 错走 Flash | `split_flashmla_requests`，MRv1 dispatch / MRv2 `gather_batch_req_state` | CPU is_prefilling、query boundaries、排序后计数；一 token prompt 尾段 |
| metadata 未更新或长度不对 | `FlashMLAMetadataBuilder.build.refresh` | 设备 seq_lens、cu、used_q；rejection 后长度；源 tensor 的 stream 依赖；mask/-1 属性 |
| 非连续错误 / 缓存污染 | `exec_kv_prefill` / `exec_kv_decode` / `decode_call` | 原 backing、stride、offset、有效 slot 与 -1 padding；非目标区域 guard |
| eager 对、graph 错 | MRv2 `build_attn_metadata` / MRv1 executor submit/wait/release | 同档位指针稳定、schedule 内容变化、ready/reuse fence、FIA updater 没有覆盖 Flash task |
| DSpark 错 | draft metadata build、context writer、draft replay | target/draft 独立状态、noncausal mask、padding、rejection 后下一轮长度 |

设备内容检查应只 dump 最小长度/slot/table/schedule 摘要；临时同步仅用于定位，结果不能当性能数据。主力机收到失败证据后单独提交修复，不让诊断日志与功能修复混在同一个提交。

## 两机回报约定

发布机在 PR #10 回报：实际 SHA、是否 TRACE、环境/包版本、命令、用例与预期、最后一个成功标记、首次错误和完整 traceback、日志/profiler 路径。先区分包导入、Meta、真实 kernel、eager 模块、图 replay、服务六个层次。首轮无需等待本诊断提交；没有失败也要回报实际命中和数值证据。

主力机不把后续诊断版或修复版自动算作首轮已通过版本；每次需要重测时明确新 SHA 和受影响用例。首轮支持范围及剩余 CPU 同步见[开发记录](flashmla_development_status.md)和[1 号对比](flashmla_plan1_comparison.md)。

用户已明确授权：定位需要时可直接使用断点、临时打桩、profiler 和 Python 堆栈，不必仅依赖已有日志，也无需为这些诊断动作反复确认。具体工具和启动方式使用发布机已安装工具及当地 skill；不要假设 py-spy/debugpy 等已安装。卡住时优先保存各 worker 的 Python 全线程堆栈，识别 host event.synchronize、包加载、metadata submit/wait、图更新或通信等待的位置，再与设备 profiler 时间线对照。需要信号触发堆栈时须先确认进程已注册对应 handler，不能向未注册的进程盲发信号。对多卡进程的 attach/断点记录暂停时间，避免将诊断导致的等待误判为原问题。
