# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=4 -m pytest -xq <this file>.

Real external operators, strided writer, HCCL exchange and graph replay. The
synthetic layer does not qualify K3 checkpoint loading or the full MRV2 runner.
"""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

import vllm_ascend.attention.mla_v1 as mla_v1
from tests.e2e.nightly.single_node.ops.singlecard_ops.test_flash_mla_with_kvcache import (
    ATOL,
    RTOL,
    _make_layer,
    _make_strided_cache,
    _reference,
    _write_reference_cache,
)
from vllm_ascend import envs
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.flash_mla import build_flash_mla_metadata
from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor
from vllm_ascend.worker.v2.attn_utils import device_metadata_context


@pytest.fixture(scope="module")
def ranks():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size not in (2, 4, 8):
        pytest.skip("requires torchrun with 2, 4 or 8 A5 ranks")
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    if not is_950():
        pytest.skip("external FlashMLA is scoped to A5")
    init_device_properties_triton()
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        yield dist.get_rank(), size
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("heads", [64, 96])
@pytest.mark.parametrize("interleave", [1, 16])
@torch.inference_mode()
def test_dcp_layer_eager_and_graph(monkeypatch, ranks, heads, interleave):
    rank, size = ranks
    torch.manual_seed(20260917)
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    monkeypatch.setattr(mla_v1, "notify_kv_cache_written", lambda *_: None)
    monkeypatch.setattr(mla_v1, "record_attention_compute_start", lambda: None)
    base = _make_layer(gate=True, bias=True, fused=True, heads=heads)
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.__dict__.update(base.__dict__)
    impl.use_mla_rope = False  # K3 MLA-NoPE; single-card cases cover RoPE.
    impl.num_heads = heads // size
    impl.dcp_size, impl.dcp_rank, impl.dcp_device_group = size, rank, dist.group.WORLD
    impl.q_proj.qrep_active = True
    impl.dcp_W_UK_T = base.W_UK_T
    start, end = rank * impl.num_heads, (rank + 1) * impl.num_heads
    impl.W_UV = base.W_UV[start:end].contiguous()
    impl.g_proj.weight = torch.nn.Parameter(base.g_proj.weight[start * 128 : end * 128].clone(), requires_grad=False)
    impl.o_proj.weight = torch.nn.Parameter(base.o_proj.weight[:, start * 128 : end * 128].clone(), requires_grad=False)
    cache, backing, protected = _make_strided_cache()
    guards = backing.cpu()[protected].clone()
    table = torch.tensor([[5, 1, 6, 0], [7, 2, 4, 3]], dtype=torch.int32, device="npu")
    builder = SimpleNamespace(
        flash_num_heads=heads,
        flash_dcp_size=size,
        flash_dcp_rank=rank,
        flash_interleave_size=interleave,
        kernel_block_size=128,
        kv_cache_spec=SimpleNamespace(block_size=768, dtype=torch.bfloat16),
        device=cache.device,
        decode_threshold=1,
        _flash_buffers={},
        _flash_attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1),
        _device_metadata_enabled=True,
        _device_metadata_tasks=(),
    )
    executor = DeviceMetadataExecutor()
    hidden = torch.empty(8, 64, dtype=torch.bfloat16, device="npu")
    output = torch.empty_like(hidden)
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=8,
        num_actual_tokens=0,
        query_start_loc=torch.zeros(3, dtype=torch.int32, device="npu"),
        seq_lens=torch.zeros(2, dtype=torch.int32, device="npu"),
        block_table_tensor=table,
        slot_mapping=torch.full((8,), -1, dtype=torch.int64, device="npu"),
        positions=torch.zeros(8, dtype=torch.int64, device="npu"),
        max_query_len=1,
        causal=True,
    )

    def prepare():
        flash = build_flash_mla_metadata(builder, common)
        executor.submit(builder._device_metadata_tasks)
        for task in builder._device_metadata_tasks:
            executor.wait(task.stage, task.group_id)
        return SimpleNamespace(flash=flash)

    graph = None
    graph_flash = None
    # Initial prefill (all history empty), mixed batch, then same decode bucket
    # crossing physical pages and toggling an inactive request.
    cases = [
        ([0, 0], [4, 3]),
        ([128 * size - 1, 17], [1, 3]),
        ([128 * size, 18], [1, 1]),
        ([128 * size + 1, 19], [1, 0]),
        ([128 * size + 2, 20], [1, 1]),
    ]
    for step, (history_lens, query_lens) in enumerate(cases):
        hidden.copy_(torch.randn_like(hidden) * 0.1)
        count = sum(query_lens)
        common.num_actual_tokens, common.max_query_len = count, max(query_lens)
        common.query_start_loc.copy_(torch.tensor([0, query_lens[0], count], dtype=torch.int32, device="npu"))
        lengths = [h + q if q else 0 for h, q in zip(history_lens, query_lens)]
        common.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        if step == 4:
            table.copy_(table.flip(0))  # New physical page mapping, same pointers.
        physical = table.cpu().tolist()
        q_c, kv = impl.fused_qkv_a_proj(hidden)[0].split([64, 576], dim=-1)
        q_nope, q_pe = impl._q_proj_and_k_up_proj(impl.q_a_layernorm(q_c))
        query = torch.cat((q_nope, q_pe), dim=-1)
        ckv, kpe = kv.view(8, 1, 576).split([512, 64], dim=-1)
        ckv = impl.kv_a_layernorm(ckv.contiguous())
        projected_kv = torch.cat((ckv, kpe), dim=-1).cpu()
        # Independent logical KV oracle; scatter writes only this rank's slots.
        global_cache = torch.randn(2, 128 * size + 16, 1, 576, dtype=torch.bfloat16) * 0.1
        local_cache = cache.cpu().clone()
        slots = [-1] * 8
        offset = 0
        for request, (h, q) in enumerate(zip(history_lens, query_lens)):
            global_cache[request, h : h + q] = projected_kv[offset : offset + q]
            for position in range(h + q):
                if (position // interleave) % size != rank:
                    continue
                local_position = position // (size * interleave) * interleave + position % interleave
                page, within = divmod(local_position, 128)
                if position < h:
                    local_cache[physical[request][page], within] = global_cache[request, position]
                else:
                    slots[offset + position - h] = physical[request][page] * 128 + within
            offset += q
        cache.copy_(local_cache)
        common.slot_mapping.copy_(torch.tensor(slots, dtype=torch.int64, device="npu"))
        expected_cache = _write_reference_cache(cache, ckv, kpe, common.slot_mapping)

        with device_metadata_context(executor):
            metadata = prepare()
            impl._forward_flash("synthetic_dcp", hidden, cache, metadata, output)
        torch.npu.synchronize()
        eager_output = output.clone()
        torch.testing.assert_close(cache.cpu().float(), expected_cache, atol=0, rtol=0)
        torch.testing.assert_close(backing.cpu()[protected], guards, atol=0, rtol=0)
        offset = 0
        for request, q in enumerate(query_lens):
            if not q:
                continue
            # One logical page per request for the CPU oracle; no DCP merge.
            latent = _reference(
                query[offset : offset + q, start:end],
                global_cache[request : request + 1],
                [q],
                [lengths[request]],
                torch.tensor([[0]]),
                causal=True,
                scale=impl.scale,
            )
            latent = latent.to(torch.bfloat16).float()
            projected = torch.einsum("htl,hlv->thv", latent, impl.W_UV.cpu().float()).to(torch.bfloat16).reshape(q, -1)
            gate = F.linear(hidden[offset : offset + q].cpu().float(), impl.g_proj.weight.cpu().float()).to(
                torch.bfloat16
            )
            projected *= torch.sigmoid(gate)
            expected = F.linear(projected.float(), impl.o_proj.weight.cpu().float(), impl.o_proj.bias.cpu().float()).to(
                torch.bfloat16
            )
            torch.testing.assert_close(eager_output[offset : offset + q].cpu(), expected, atol=ATOL, rtol=RTOL)
            offset += q
        assert torch.count_nonzero(eager_output[count:]) == 0
        if step == 2:
            # Warm up the real collective before capture. Metadata is outside
            # capture and reuses BOTH schedule addresses on replay.
            for _ in range(2):
                with device_metadata_context(executor):
                    metadata = prepare()
                    impl._forward_flash("synthetic_dcp", hidden, cache, metadata, output)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with device_metadata_context(executor):
                metadata = prepare()
                graph_flash = metadata.flash
                pointers = [
                    t.data_ptr()
                    for t in (graph_flash.schedule, graph_flash.current.schedule, graph_flash.current.cache)
                ]
                with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                    impl._forward_flash("synthetic_dcp", hidden, cache, metadata, output)
        if step >= 3:
            with device_metadata_context(executor):
                metadata = prepare()
                assert metadata.flash is graph_flash
                assert pointers == [
                    t.data_ptr()
                    for t in (graph_flash.schedule, graph_flash.current.schedule, graph_flash.current.cache)
                ]
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, eager_output, atol=ATOL, rtol=RTOL)
        dist.barrier()
