# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

import vllm_ascend.attention.mla_v1 as mla_v1
import vllm_ascend.ops.rotary_embedding as rotary_embedding
from vllm_ascend import envs
from vllm_ascend.attention.flash_mla import (
    FlashMLAContract,
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_metadata,
    validate_flash_mla_kv_cache,
)
from vllm_ascend.device.device_config import is_950
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor
from vllm_ascend.worker.v2.attn_utils import device_metadata_context

PAGES = 8
BLOCK_SIZE = 128
HEADS_Q = 64
QK_DIM = 576
V_DIM = 512
PAGE_GAP = 64
STORAGE_OFFSET = 13
ATOL = 2e-2
RTOL = 2e-2


def _write_reference_cache(cache, ckv, kpe, slots):
    """Update a CPU oracle by physical page/offset, independently of scatter."""
    expected = cache.float().cpu().clone()
    ckv, kpe = ckv.float().cpu(), kpe.float().cpu()
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            page, offset = divmod(slot, cache.shape[1])
            expected[page, offset, 0, :V_DIM] = ckv[row, 0]
            expected[page, offset, 0, V_DIM:] = kpe[row, 0]
    return expected


def _make_strided_cache(block_size=BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_elements = block_size * QK_DIM
    page_stride = page_elements + PAGE_GAP
    required = STORAGE_OFFSET + (PAGES - 1) * page_stride + page_elements
    backing = torch.full((required,), 7.25, dtype=torch.bfloat16, device="npu")
    cache = torch.as_strided(
        backing,
        (PAGES, block_size, 1, QK_DIM),
        (page_stride, QK_DIM, QK_DIM, 1),
        STORAGE_OFFSET,
    )
    cache.copy_(torch.randn_like(cache) * 0.1)

    protected = torch.ones(required, dtype=torch.bool)
    for page in range(PAGES):
        start = STORAGE_OFFSET + page * page_stride
        protected[start : start + page_elements] = False
    return cache, backing, protected


def _cu_seqlens(query_lens: list[int]) -> torch.Tensor:
    boundaries = [0]
    for length in query_lens:
        boundaries.append(boundaries[-1] + length)
    return torch.tensor(boundaries, dtype=torch.int32, device="npu")


def _slots(
    query_lens: list[int],
    cache_lens: list[int],
    block_table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    table = block_table.cpu()
    result: list[int] = []
    for request, (query_len, cache_len) in enumerate(zip(query_lens, cache_lens)):
        for logical_position in range(cache_len - query_len, cache_len):
            logical_block, offset = divmod(logical_position, block_size)
            physical_block = int(table[request, logical_block])
            result.append(physical_block * block_size + offset)
    return torch.tensor(result, dtype=torch.int64, device="npu")


def _reference(
    query: torch.Tensor,
    cache: torch.Tensor,
    query_lens: list[int],
    cache_lens: list[int],
    block_table: torch.Tensor,
    *,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    query_cpu = query.float().cpu()
    cache_cpu = cache.float().cpu()
    table_cpu = block_table.cpu()
    outputs = []
    token_start = 0
    for request, (query_len, cache_len) in enumerate(zip(query_lens, cache_lens)):
        keys = []
        for logical_position in range(cache_len):
            logical_block, offset = divmod(logical_position, cache.shape[1])
            physical_block = int(table_cpu[request, logical_block])
            keys.append(cache_cpu[physical_block, offset, 0])
        key = torch.stack(keys)
        q = query_cpu[token_start : token_start + query_len]
        scores = torch.einsum("qnd,kd->qnk", q, key) * scale
        if causal:
            q_index = torch.arange(query_len).view(query_len, 1)
            k_index = torch.arange(cache_len).view(1, cache_len)
            visible = k_index <= cache_len - query_len + q_index
            scores.masked_fill_(~visible.unsqueeze(1), float("-inf"))
        probability = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qnk,kv->qnv", probability, key[:, :V_DIM]))
        token_start += query_len
    return torch.cat(outputs, dim=0).transpose(0, 1)


def _run_case(
    cache: torch.Tensor,
    backing: torch.Tensor,
    protected: torch.Tensor,
    block_table: torch.Tensor,
    query_lens: list[int],
    cache_lens: list[int],
    *,
    causal: bool,
    heads_q: int = HEADS_Q,
) -> None:
    """Compare real two-stage attention and scatter with an independent CPU oracle."""
    total_query = sum(query_lens)
    query = (
        torch.randn(
            total_query,
            heads_q,
            QK_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.1
    )
    current_ckv = (
        torch.randn(
            total_query,
            1,
            V_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.1
    )
    current_kpe = (
        torch.randn(
            total_query,
            1,
            QK_DIM - V_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.1
    )
    slot_mapping = _slots(query_lens, cache_lens, block_table, block_size=cache.shape[1])
    protected_before = backing.cpu()[protected].clone()
    expected_cache = _write_reference_cache(cache, current_ckv, current_kpe, slot_mapping)

    torch_npu.npu_scatter_pa_kv_cache(
        key=current_ckv.contiguous(),
        value=current_kpe.contiguous(),
        key_cache=cache[..., :V_DIM],
        value_cache=cache[..., V_DIM:],
        slot_mapping=slot_mapping,
        cache_mode="Norm",
    )

    cu_seqlens_q = _cu_seqlens(query_lens)
    seqused_q = torch.tensor(query_lens, dtype=torch.int32, device="npu")
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device="npu")
    mask_mode = 3 if causal else 0
    attn_mask = (
        torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8, device="npu"),
            diagonal=1,
        )
        if causal
        else None
    )
    scale = 1.0 / math.sqrt(QK_DIM)
    contract = FlashMLAContract(
        num_heads_q=heads_q,
        mask_mode=mask_mode,
        softmax_scale=scale,
    )
    metadata = flash_mla_with_kvcache_metadata(
        cache_seqlens,
        contract.num_heads_q,
        contract.num_heads_kv,
        **contract.metadata_kwargs(cu_seqlens_q, seqused_q),
    )
    actual, _ = flash_mla_with_kvcache(
        query,
        cache,
        **contract.attention_kwargs(
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            attn_mask=attn_mask,
            metadata=metadata,
        ),
    )
    torch.npu.synchronize()

    assert actual.shape == (heads_q, total_query, V_DIM)
    torch.testing.assert_close(cache.float().cpu(), expected_cache, atol=0, rtol=0)
    expected = _reference(
        query,
        expected_cache,
        query_lens,
        cache_lens,
        block_table,
        causal=causal,
        scale=scale,
    )
    torch.testing.assert_close(actual.float().cpu(), expected, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        backing.cpu()[protected],
        protected_before,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize(
    ("query_lens", "cache_lens", "causal"),
    [
        ([1, 1], [129, 9], True),
        ([4, 3], [132, 13], True),
        ([1, 5], [130, 133], True),
        ([3, 2], [131, 10], False),
    ],
    ids=["decode-cross-page", "prefill", "mixed", "no-mask"],
)
@pytest.mark.parametrize("heads_q", [64, 96])
def test_external_flash_mla_strided_pa_bbnd(
    query_lens: list[int],
    cache_lens: list[int],
    causal: bool,
    block_size: int,
    heads_q: int,
):
    if not is_950():
        pytest.skip("external FlashMLA is scoped to A5")
    torch.manual_seed(20260916)
    cache, backing, protected = _make_strided_cache(block_size)
    validate_flash_mla_kv_cache(
        cache,
        expected_dtype=torch.bfloat16,
        expected_device=cache.device,
    )
    block_table = torch.tensor(
        [[5, 1, 6, 0], [7, 2, 4, 3]],
        dtype=torch.int32,
        device="npu",
    )
    _run_case(
        cache,
        backing,
        protected,
        block_table,
        query_lens,
        cache_lens,
        causal=causal,
        heads_q=heads_q,
    )


@pytest.mark.parametrize("block_size", [128])
def test_external_flash_mla_multistep_length_change(block_size):
    if not is_950():
        pytest.skip("external FlashMLA is scoped to A5")
    torch.manual_seed(20260917)
    cache, backing, protected = _make_strided_cache(block_size)
    block_table = torch.tensor(
        [[5, 1, 6, 0], [7, 2, 4, 3]],
        dtype=torch.int32,
        device="npu",
    )
    _run_case(
        cache,
        backing,
        protected,
        block_table,
        [2, 1],
        [block_size + 1, 8],
        causal=True,
    )
    _run_case(
        cache,
        backing,
        protected,
        block_table,
        [1, 3],
        [block_size + 2, 11],
        causal=True,
    )


class _TupleLinear(torch.nn.Module):
    """Small unquantized layer fixture; no TP collectives or real model loading."""

    def __init__(self, in_features, out_features, *, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16, device="npu") * 0.02,
            requires_grad=False,
        )
        self.bias = (
            torch.nn.Parameter(
                torch.randn(out_features, dtype=torch.bfloat16, device="npu") * 0.02, requires_grad=False
            )
            if bias
            else None
        )
        # This fixture implements unquantized F.linear directly. Only the
        # method identity is needed by the production fast-path predicate.
        self.quant_method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
        self.input_is_parallel = True
        self.reduce_results = False
        self.custom_op = None
        self.calls = []

    def forward(self, x, *, is_prefill=False):
        self.calls.append(is_prefill)
        return F.linear(x, self.weight, self.bias), None


def _make_layer(*, gate, bias, fused, heads=HEADS_Q):
    impl = mla_v1.AscendMLAImpl.__new__(mla_v1.AscendMLAImpl)
    impl.num_heads = heads
    impl.num_kv_heads = 1
    impl.kv_lora_rank = V_DIM
    impl.qk_nope_head_dim = 128
    impl.qk_rope_head_dim = 64
    impl.qk_head_dim = 192
    impl.v_head_dim = 128
    impl.q_lora_rank = 64 if fused else None
    impl.scale = impl.qk_head_dim**-0.5
    impl.use_mla_rope = True
    impl.use_output_gate = gate
    impl.fused_qkv_a_proj = _TupleLinear(64, 64 + QK_DIM) if fused else None
    impl.kv_a_proj_with_mqa = None if fused else _TupleLinear(64, QK_DIM)
    impl.q_proj = _TupleLinear(64, heads * impl.qk_head_dim)
    impl.q_a_layernorm = torch.nn.RMSNorm(64, eps=1e-6, dtype=torch.bfloat16, device="npu")
    impl.kv_a_layernorm = torch.nn.RMSNorm(V_DIM, eps=1e-6, dtype=torch.bfloat16, device="npu")
    impl.W_UK_T = torch.randn(heads, 128, V_DIM, dtype=torch.bfloat16, device="npu") * 0.02
    impl.W_UV = torch.randn(heads, V_DIM, 128, dtype=torch.bfloat16, device="npu") * 0.02
    impl.g_proj = _TupleLinear(64, heads * 128) if gate else None
    impl.o_proj = _TupleLinear(heads * 128, 64, bias=bias)
    return impl


@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("fused", [True, False], ids=["fused-qkv", "separate-q-kv"])
@pytest.mark.parametrize("heads", [64, 96])
@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "nope"])
@pytest.mark.parametrize(
    ("query_lens", "gate", "bias", "input_pad", "output_pad", "expect_fast"),
    [
        ([1, 1], True, False, 0, 0, True),
        ([1, 1], True, False, 2, 0, True),
        ([1, 1], True, False, 0, 2, False),
        ([1, 1], False, False, 0, 0, False),
        ([1, 1], True, True, 0, 0, False),
        ([4, 3], True, False, 0, 0, False),
        ([1, 5], True, False, 0, 0, False),
    ],
    ids=["decode-fast", "input-padding", "output-padding", "no-gate", "bias", "prefill", "mixed"],
)
@torch.inference_mode()
def test_external_flash_mla_layer_forward(
    monkeypatch, block_size, fused, heads, use_rope, query_lens, gate, bias, input_pad, output_pad, expect_fast
):
    """Real schedule/scatter/attention/V-up/gate/output; synthetic layer, not a model smoke."""
    if not is_950():
        pytest.skip("external FlashMLA is scoped to A5")
    torch.manual_seed(20260918)
    impl = _make_layer(gate=gate, bias=bias, fused=fused, heads=heads)
    impl.use_mla_rope = use_rope
    cache, backing, protected = _make_strided_cache(block_size)
    protected_before = backing.cpu()[protected].clone()
    cache_identity = cache.data_ptr(), cache.stride(), cache.storage_offset()
    cache_lens = [block_size + 4, 13]
    table = torch.tensor([[5, 1, 6, 0], [7, 2, 4, 3]], dtype=torch.int32, device="npu")
    actual_tokens = sum(query_lens)
    tokens = actual_tokens + input_pad
    slots = _slots(query_lens, cache_lens, table, block_size=block_size)
    positions = torch.cat(
        [torch.arange(kv - q, kv, dtype=torch.int64, device="npu") for q, kv in zip(query_lens, cache_lens)]
    )
    if input_pad:
        slots = torch.cat((slots, slots.new_full((input_pad,), -1)))
        positions = torch.cat((positions, positions.new_zeros(input_pad)))
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=tokens,
        num_actual_tokens=actual_tokens,
        query_start_loc=_cu_seqlens(query_lens),
        query_start_loc_cpu=_cu_seqlens(query_lens).cpu(),
        max_query_len=max(query_lens),
        seq_lens=torch.tensor(cache_lens, dtype=torch.int32, device="npu"),
        block_table_tensor=table,
        slot_mapping=slots,
        positions=positions,
        causal=True,
        attn_state=mla_v1.AscendAttentionState.ChunkedPrefill,
        context_parallel_metadata=None,
    )
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    builder.flash_num_heads = heads
    builder.kernel_block_size = block_size
    builder.kv_cache_spec = SimpleNamespace(block_size=block_size * 6, dtype=torch.bfloat16)
    builder.device = cache.device
    builder.decode_threshold = 1
    builder._flash_buffers = {}
    builder._flash_attn_mask = mask
    builder._device_metadata_enabled = False
    builder._device_metadata_tasks = ()
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: mask)
    builder.metadata_cls = mla_v1.AscendMLAMetadata
    metadata = builder.build(0, common)
    flash = metadata.flash
    assert flash is not None
    assert flash.schedule is not None
    assert flash.schedule.device == cache.device
    assert flash.contract.max_seqlen_q == -1
    assert flash.query_capacity == tokens
    assert flash.contract.max_seqlen_kv == -1
    assert flash.kv_capacity == table.shape[1] * block_size

    # Seed the real RoPE lookup table; no projection/RMSNorm/RoPE operator is mocked.
    freq = torch.outer(torch.arange(max(cache_lens) + 1).float(), 10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
    monkeypatch.setattr(rotary_embedding, "_cos_cache", torch.cat((freq.cos(), freq.cos()), -1).to(cache))
    monkeypatch.setattr(rotary_embedding, "_sin_cache", torch.cat((freq.sin(), freq.sin()), -1).to(cache))
    monkeypatch.setattr(rotary_embedding, "_cos_mla", None)
    # This non-PD layer fixture has no service context or transfer fences.
    monkeypatch.setattr(mla_v1, "notify_kv_cache_written", lambda *_: None)
    monkeypatch.setattr(mla_v1, "record_attention_compute_start", lambda: None)
    captured = {}
    real_scatter = torch_npu.npu_scatter_pa_kv_cache
    real_attention = mla_v1.flash_mla_with_kvcache
    real_gate = mla_v1.flash_attention_gate

    def scatter_spy(**kwargs):
        assert kwargs["key_cache"].stride(0) == cache.stride(0)
        assert kwargs["value_cache"].data_ptr() == cache.data_ptr() + V_DIM * cache.element_size()
        captured["expected_cache"] = _write_reference_cache(
            cache, kwargs["key"], kwargs["value"], kwargs["slot_mapping"]
        )
        return real_scatter(**kwargs)

    def attention_spy(query, passed_cache, **kwargs):
        assert passed_cache is cache
        assert kwargs["metadata"] is flash.schedule
        assert kwargs["layout_kv"] == "PA_BBND"
        assert kwargs["layout_out"] == "NTD"
        captured["query"] = query[:actual_tokens].float().cpu()
        return real_attention(query, passed_cache, **kwargs)

    def gate_spy(*args):
        captured["fast"] = True
        return real_gate(*args)

    monkeypatch.setattr(torch_npu, "npu_scatter_pa_kv_cache", scatter_spy)
    monkeypatch.setattr(mla_v1, "flash_mla_with_kvcache", attention_spy)
    monkeypatch.setattr(mla_v1, "flash_attention_gate", gate_spy)
    hidden = torch.randn(tokens, 64, dtype=torch.bfloat16, device="npu") * 0.1
    output = torch.full((tokens + output_pad, 64), float("nan"), dtype=torch.bfloat16, device="npu")
    result = impl.forward("synthetic_mla", hidden, cache, metadata, output)
    torch.npu.synchronize()
    assert result is output
    assert (cache.data_ptr(), cache.stride(), cache.storage_offset()) == cache_identity
    assert captured.get("fast", False) is expect_fast
    assert impl.o_proj.calls == ([] if expect_fast else [flash.is_prefill])
    torch.testing.assert_close(cache.float().cpu(), captured["expected_cache"], atol=0, rtol=0)
    torch.testing.assert_close(backing.cpu()[protected], protected_before, atol=0, rtol=0)

    # Independent CPU attention/postprocessing oracle, fed by the real
    # preprocessing outputs. Preprocessing numerics and model loading are
    # deliberately not claimed as independently validated by this fixture.
    latent = (
        _reference(
            captured["query"], captured["expected_cache"], query_lens, cache_lens, table, causal=True, scale=impl.scale
        )
        .to(torch.bfloat16)
        .float()
    )
    projected = torch.einsum("htl,hlv->thv", latent, impl.W_UV.float().cpu())
    projected = projected.to(torch.bfloat16).reshape(actual_tokens, -1)
    if gate:
        gate_values = F.linear(hidden[:actual_tokens].float().cpu(), impl.g_proj.weight.float().cpu())
        gate_values = gate_values.to(torch.bfloat16)
        projected = projected * torch.sigmoid(gate_values)
    expected = F.linear(
        projected.float(),
        impl.o_proj.weight.float().cpu(),
        impl.o_proj.bias.float().cpu() if bias else None,
    ).to(torch.bfloat16)
    torch.testing.assert_close(output[:actual_tokens].cpu(), expected, atol=ATOL, rtol=RTOL)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[actual_tokens:]) == 0


@torch.inference_mode()
def test_external_flash_mla_layer_graph_refresh(monkeypatch):
    """Same decode bucket, new lengths/pages/live rows; real graph versus eager.

    This uses a synthetic layer and the MRV2 executor ownership mechanism,
    not a complete model runner or TP service.
    """
    if not is_950():
        pytest.skip("external FlashMLA is scoped to A5")
    torch.manual_seed(20260919)
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    monkeypatch.setattr(mla_v1, "notify_kv_cache_written", lambda *_: None)
    monkeypatch.setattr(mla_v1, "record_attention_compute_start", lambda: None)
    impl = _make_layer(gate=True, bias=False, fused=True)
    cache, backing, protected = _make_strided_cache()
    protected_before = backing.cpu()[protected].clone()
    freq = torch.outer(torch.arange(512).float(), 10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
    monkeypatch.setattr(rotary_embedding, "_cos_cache", torch.cat((freq.cos(), freq.cos()), -1).to(cache))
    monkeypatch.setattr(rotary_embedding, "_sin_cache", torch.cat((freq.sin(), freq.sin()), -1).to(cache))
    monkeypatch.setattr(rotary_embedding, "_cos_mla", None)
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    builder.device = cache.device
    builder.flash_num_heads = HEADS_Q
    builder.kernel_block_size = BLOCK_SIZE
    builder.kv_cache_spec = SimpleNamespace(block_size=BLOCK_SIZE * 6, dtype=torch.bfloat16)
    builder.decode_threshold = 1
    builder._flash_buffers = {}
    builder._flash_attn_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    builder._device_metadata_enabled = True
    builder._device_metadata_tasks = ()
    builder.metadata_cls = mla_v1.AscendMLAMetadata
    table = torch.tensor([[5, 1, 6, 0], [7, 2, 4, 3]], dtype=torch.int32, device="npu")
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=4,
        num_actual_tokens=2,
        query_start_loc=_cu_seqlens([1, 1]),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([129, 9], dtype=torch.int32, device="npu"),
        block_table_tensor=table,
        slot_mapping=_slots([1, 1], [129, 9], table),
        positions=torch.tensor([128, 8], dtype=torch.int64, device="npu"),
        max_query_len=1,
        causal=True,
        context_parallel_metadata=None,
        attn_state=mla_v1.AscendAttentionState.DecodeOnly,
    )
    executor = DeviceMetadataExecutor()
    hidden = torch.randn(4, 64, dtype=torch.bfloat16, device="npu") * 0.1
    output = torch.empty_like(hidden)

    def prepare():
        assert not torch.npu.is_current_stream_capturing()
        metadata = builder.build(0, common)
        tasks = builder.take_device_metadata_tasks()
        executor.submit(tasks)
        for task in tasks:
            executor.wait(task.stage, task.group_id)
        return metadata

    for _ in range(3):
        with device_metadata_context(executor):
            metadata = prepare()
            impl.forward("synthetic_mla", hidden, cache, metadata, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with device_metadata_context(executor):
        metadata = prepare()
        flash = metadata.flash
        pointers = [getattr(flash, name).data_ptr() for name in ("query", "schedule", "cu", "cache_lens", "slots")]
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            impl.forward("synthetic_mla", hidden, cache, metadata, output)

    for step, lengths in enumerate(([130, 10], [131, 0], [132, 11])):
        # Replace physical pages without changing the bucket or Tensor addresses.
        if step == 2:
            common.block_table_tensor.copy_(table.flip(0))
        common.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        common.positions.copy_(torch.tensor([lengths[0] - 1, max(lengths[1] - 1, 0)], dtype=torch.int64, device="npu"))
        slots = _slots([1, 1], [lengths[0], max(lengths[1], 1)], common.block_table_tensor)
        common.slot_mapping.copy_(slots)
        hidden.copy_(torch.randn_like(hidden) * 0.1)
        with device_metadata_context(executor):
            metadata = prepare()
            assert metadata.flash is flash
            assert pointers == [
                getattr(flash, name).data_ptr() for name in ("query", "schedule", "cu", "cache_lens", "slots")
            ]
            graph.replay()
        torch.npu.synchronize()
        replay_output = output.clone()
        with device_metadata_context(executor):
            metadata = prepare()
            eager_output = torch.empty_like(output)
            impl.forward("synthetic_mla", hidden, cache, metadata, eager_output)
        torch.npu.synchronize()
        torch.testing.assert_close(replay_output, eager_output, atol=ATOL, rtol=RTOL)
        assert torch.isfinite(replay_output).all()
        assert torch.count_nonzero(replay_output[2:]) == 0
        if lengths[1] == 0:
            assert torch.count_nonzero(replay_output[1]) == 0
        torch.testing.assert_close(backing.cpu()[protected], protected_before, atol=0, rtol=0)
