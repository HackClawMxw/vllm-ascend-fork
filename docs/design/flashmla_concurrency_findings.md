# 并发审查发现与首轮修补

本次 agent team 检查的是运行候选 `698d00e7c` 及诊断版 `f78fe02e8`。发现和修复均按源码与 CPU 回归说明；尚无本候选的真实 NPU 并发验收结果。

## P1：普通 eager Decode 的不同形状永久占用缓冲

来源是主力机接入提交 `378615ec4` 的 `FlashMLAMetadataBuilder.build`。参考 1 号按形状保留缓冲以保证图地址稳定，但接入时没有把普通 eager 的临时形状与图形状区分：对每个纯 Decode `(tokens, rows, columns, causal)` 都向 `self.buffers` 登记，没有淘汰。这是新 adapter metadata 层的生命周期问题，不是 2 号 KV allocator、持久 KV cache 或外部包内部缓存问题。

不是每个请求都新增一份；是首次遇到的新形状留下 Q、schedule、table 等引用。以 H=64、BF16、D=576 为例，Q 每 token 占 72 KiB；若一个 eager 进程依次经历 T=1..1024，累计 Q 约 36 GiB。这是说明风险的理论上界例子，不是已观测的发布机 OOM。NPU allocator 的 reserved/cache 与仍被 Python 引用的 live buffers 也需要区分。

最小修复：

- `state.build` 新增显式 `retain_for_graph`；普通 eager 新形状不登记持久 buffers，由本轮 metadata/task 持有。
- 捕获入口登记稳定缓冲；MRv1 显式 FULL execution 也保留，以保持 ExternalEvent 的 schedule frontier。普通 replay/eager 命中已有图键时继续复用。
- mixed batch 继续使用临时缓冲；不做带淘汰的图地址缓存，不改变 KV backing、算子参数或流等待协议。
- MRv1/MRv2 target 与 draft 的标准 FULL capture 入口已核对。固定 vLLM 的正常 capture warmup 是 NONE，不依赖“所有 warmup 都是 FULL”的假设。

新增实际 builder 生命周期检查 8 项，与现有 32 项合跑 40 passed。新增回归用 eager token 数1..16验证 buffers 不累积；显式保留一个图档位后，其余 eager 形状不留存且命中图档位继续复用。已临时加载 Git HEAD 旧生产模块反证：纯 eager 不累积测试在旧代码因 buffers 非空失败，不是因为新参数不兼容而失败。

发布机验证：先 eager、无 speculative，逐档提高再降低活跃请求数，记录真实 batch T、buffer key 数和 live Q 字节、NPU allocated/reserved。修复后持久 key 数应由图档位决定，不随 eager 历史形状增长；随后确认 FULL 同档位变请求/长度仍地址稳定并读取新 schedule。TRACE 中同一 eager 未捕获形状的地址变化允许；图档位的稳定要求不变。

首轮旧候选仍可用于包/小流量功能定位，但长时间遍历大量 eager 形状前应切换此修复后的完整 SHA。测试报告不得把旧候选与修复版结果混在一起。

## P2：DSpark FULL 路径重复构建 metadata 的性能缺口

固定 vLLM 的 DFlash `propose` 先构建 draft metadata，随后本地 `DFlashAclGraphManager.run_fullgraph` 又调用 `build_draft_attn_metadatas`，使 Flash metadata task 重复提交。该入口结构原本服务 FIA 图参数更新，新 Flash 接入后还会重复生成设备 schedule。

目前没有证明它导致数值错误；本批不在 P1 修复中合并改动。发布机 profile 应记录一个 draft forward 对应的 metadata 调用次数，不能把两次 metadata 当成两个 Decode step。后续优化候选是复用第一次 metadata，但必须确认它已使用与 FULL descriptor 一致的 padding/shape，不能直接删除第二次调用。由主力机根据日志与实际对象生命周期另做最小修复和回归。

## 并发验证职责

主力机负责源码修补、定向回归、工具和明确的新 SHA；发布机按当地 skill 拉起服务，提供真实包、事件、长稳与数值证据。参见[双机分工](flashmla_team_workflow.md)、[并发客户端](flashmla_concurrency_probe.md)与[诊断手册](flashmla_diagnostics.md)。
