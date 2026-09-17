# SPDX-License-Identifier: Apache-2.0
"""torchrun --standalone --nproc-per-node=4 -m pytest -xq <this file>.

Real projection weights/loaders, NPU operators and TP group; backend registration
is isolated for the small layer. Not a K3 checkpoint or full MRV2 runner smoke.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.distributed import parallel_state

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_flash_mla_with_kvcache import (
    ATOL,
    RTOL,
    _cu_seqlens,
    _make_layer,
    _make_strided_cache,
    _reference,
    _slots,
    _write_reference_cache,
)
from vllm_ascend import envs
from vllm_ascend.attention import flash_mla, mla_v1
from vllm_ascend.device.device_config import is_950
from vllm_ascend.models import kimi_k3
from vllm_ascend.ops import linear as ascend_linear
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module")
def ranks():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size != 4 or not is_950():
        pytest.skip("requires torchrun with four A5 ranks")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    enable_custom_op()
    parallel_state.init_distributed_environment(
        world_size=size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="hccl",
    )
    group = parallel_state.init_model_parallel_group(
        [list(range(size))],
        local_rank=local_rank,
        backend="hccl",
        group_name="replicated_mla_test",
        use_device_communicator=False,
    )
    try:
        yield group, rank, size
    finally:
        group.destroy()
        parallel_state.destroy_distributed_environment()


def _loaded_layer(monkeypatch, fused):
    def wrapper(*args):
        return SimpleNamespace(mla_attn=SimpleNamespace(num_heads=args[1], impl=SimpleNamespace()))

    # Use production construction/linear loaders; avoid needing a full model
    # config and layer registration just to exercise a 64-wide synthetic input.
    monkeypatch.setattr(kimi_k3, "AscendKimiK3MultiHeadLatentAttention", wrapper)
    for name in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear"):
        monkeypatch.setattr(kimi_k3, name, getattr(ascend_linear, f"Ascend{name}"))
    layer = kimi_k3.AscendKimiMLAAttention(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        hidden_size=64,
        num_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=64 if fused else None,
        kv_lora_rank=512,
        use_output_gate=True,
        use_rope=False,
        prefix="model.layers.1.self_attn",
    ).to(device="npu", dtype=torch.bfloat16)
    weights = {}
    for name, parameter in layer.named_parameters():
        weight = (torch.randn(parameter.shape) * 0.02).to(torch.bfloat16)
        if "layernorm" in name:
            weight.fill_(1)
        loader = getattr(parameter, "weight_loader", None)
        if loader is None:
            parameter.copy_(weight.to(parameter))
        else:
            loader(parameter, weight.to(parameter))
        weights[name] = weight.float()
        torch.testing.assert_close(parameter.cpu(), weight, atol=0, rtol=0)
    assert layer.mla_attn.mla_attn.num_heads == layer.num_local_heads == 96
    for name in ("q_b_proj" if fused else "q_proj", "kv_b_proj", "g_proj", "o_proj"):
        assert getattr(layer, name).tp_size == 1
    assert not layer.o_proj.reduce_results
    impl = _make_layer(gate=True, bias=False, fused=fused, heads=96)
    for name in ("fused_qkv_a_proj", "kv_a_proj_with_mqa", "q_a_layernorm", "kv_a_layernorm", "g_proj", "o_proj"):
        setattr(impl, name, getattr(layer, name, None))
    impl.q_proj = layer.q_b_proj if fused else layer.q_proj
    # This fixture isolates weight processing; full wrapper post-load processing
    # remains part of the required real-model test.
    kv_weight = layer.kv_b_proj.weight.view(96, 256, 512)
    impl.W_UK_T = kv_weight[:, :128].contiguous()
    impl.W_UV = kv_weight[:, 128:].transpose(1, 2).contiguous()
    impl.use_mla_rope = False
    return layer, impl, weights


def _cpu_inputs(hidden, weights, fused):
    def linear(x, name):
        return F.linear(x.float(), weights[f"{name}.weight"]).to(torch.bfloat16).float()

    def norm(x):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()

    if fused:
        q_c, kv = linear(hidden, "fused_qkv_a_proj").split([64, 576], -1)
        query = linear(norm(q_c), "q_b_proj")
    else:
        kv = linear(hidden, "kv_a_proj_with_mqa")
        query = linear(hidden, "q_proj")
    query = query.view(-1, 96, 192)
    k_up = weights["kv_b_proj.weight"].view(96, 256, 512)[:, :128]
    q_abs = torch.einsum("thd,hdl->thl", query[..., :128], k_up).to(torch.bfloat16).float()
    return torch.cat((q_abs, query[..., 128:]), -1), norm(kv[:, :512]).unsqueeze(1), kv[:, 512:].unsqueeze(1)


@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_replicated_mla_eager_and_sp(monkeypatch, ranks, fused):
    group, rank, size = ranks
    monkeypatch.setattr(parallel_state, "_TP", group)
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    torch.manual_seed(20260917)
    layer, impl, weights = _loaded_layer(monkeypatch, fused)
    cache, backing, protected = _make_strided_cache()
    guard = backing.cpu()[protected].clone()
    identity = cache.data_ptr(), cache.stride(), cache.storage_offset()
    builder = SimpleNamespace(
        device=cache.device,
        kernel_block_size=128,
        decode_threshold=1,
        kv_cache_spec=SimpleNamespace(block_size=768, dtype=torch.bfloat16),
        _device_metadata_enabled=False,
    )
    flash_mla.init_flash_mla_metadata(builder, impl)
    monkeypatch.setattr(mla_v1, "notify_kv_cache_written", lambda *_: None)
    monkeypatch.setattr(mla_v1, "record_attention_compute_start", lambda: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("replicated MLA must not perform a TP collective")

    monkeypatch.setattr(group, "all_reduce", forbidden)
    monkeypatch.setattr(group, "all_gather", forbidden)
    monkeypatch.setattr(group, "reduce_scatter", forbidden)
    table = torch.tensor([[5, 1, 6, 0], [7, 2, 4, 3]], dtype=torch.int32, device="npu")
    cases = [
        ([4, 3], [4, 3]),
        ([1, 1], [5, 4]),
        ([3, 1], [8, 5]),
        ([1, 1], [129, 128]),
        ([1, 1], [130, 129]),
        ([1, 0], [131, 129]),
    ]
    for query_lens, cache_lens in cases:
        actual = sum(query_lens)
        tokens = actual + 1
        hidden = (torch.randn(tokens, 64) * 0.1).to(torch.bfloat16)
        slots = _slots(query_lens, cache_lens, table)
        positions = torch.cat(
            [torch.arange(kv - q, kv, device="npu", dtype=torch.int64) for q, kv in zip(query_lens, cache_lens)]
        )
        common = SimpleNamespace(
            num_reqs=2,
            num_input_tokens=tokens,
            num_actual_tokens=actual,
            query_start_loc=_cu_seqlens(query_lens),
            max_query_len=max(query_lens),
            seq_lens=torch.tensor(cache_lens, dtype=torch.int32, device="npu"),
            block_table_tensor=table,
            slot_mapping=F.pad(slots, (0, 1), value=-1),
            positions=F.pad(positions, (0, 1)),
            causal=True,
        )
        flash = flash_mla.build_flash_mla_metadata(builder, common)
        assert flash.query.shape == (tokens, 96, 576)
        assert flash.contract.num_heads_q == 96 and not flash.contract.return_softmax_lse
        q, ckv, kpe = _cpu_inputs(hidden[:actual].float(), weights, fused)
        expected_cache = _write_reference_cache(cache, ckv, kpe, slots)
        output = torch.empty_like(hidden, device="npu")
        impl._forward_flash("replicated_mla_fixture", hidden.npu(), cache, SimpleNamespace(flash=flash), output)
        torch.npu.synchronize()
        # Norm/GEMM rounding is checked numerically; writer exactness is covered
        # by the single-card scatter spy, using its actual production inputs.
        torch.testing.assert_close(cache.float().cpu(), expected_cache, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(backing.cpu()[protected], guard, atol=0, rtol=0)
        assert identity == (cache.data_ptr(), cache.stride(), cache.storage_offset())
        latent = _reference(q, expected_cache, query_lens, cache_lens, table, causal=True, scale=impl.scale)
        v_up = weights["kv_b_proj.weight"].view(96, 256, 512)[:, 128:].transpose(1, 2)
        projected = torch.einsum("htl,hlv->thv", latent.to(torch.bfloat16).float(), v_up)
        projected = projected.to(torch.bfloat16).reshape(actual, -1)
        gate = F.linear(hidden[:actual].float(), weights["g_proj.weight"]).to(torch.bfloat16)
        expected = F.linear((projected * gate.sigmoid()).float(), weights["o_proj.weight"]).to(torch.bfloat16)
        torch.testing.assert_close(output[:actual].cpu(), expected, atol=ATOL, rtol=RTOL)
        assert torch.count_nonzero(output[actual:]) == 0
        copies = [torch.empty_like(output) for _ in range(size)]
        dist.all_gather(copies, output)  # Test-only comparison, outside MLA.
        for copy in copies:
            torch.testing.assert_close(copy, output, atol=ATOL, rtol=RTOL)
        decoder = SimpleNamespace(use_sequence_parallel=True, self_attn=layer)
        sharded = kimi_k3.AscendKimiDecoderLayer._finish_attention_output(decoder, output[:actual])
        padded = F.pad(output[:actual], (0, 0, 0, (-actual) % size))
        torch.testing.assert_close(sharded, padded.chunk(size)[rank], atol=0, rtol=0)
