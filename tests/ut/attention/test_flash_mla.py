# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.flash_mla as flash_mla
import vllm_ascend.attention.mla_v1 as mla_v1
import vllm_ascend.platform as ascend_platform
from vllm_ascend import envs as ascend_envs


def _common(*, causal=True, num_input_tokens=6):
    return SimpleNamespace(
        num_reqs=2,
        num_input_tokens=num_input_tokens,
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([5, 2], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3, 1], [2, 0]], dtype=torch.int32),
        slot_mapping=torch.tensor([384, 385, 256, 257, 99, 98], dtype=torch.int64),
        positions=torch.tensor([3, 4, 0, 1, 91, 92], dtype=torch.int64),
        max_query_len=2,
        causal=causal,
    )


def _builder(*, heads=64, deferred=False):
    return SimpleNamespace(
        flash_num_heads=heads,
        kernel_block_size=128,
        kv_cache_spec=SimpleNamespace(block_size=768, dtype=torch.bfloat16),
        device=torch.device("cpu"),
        decode_threshold=1,
        _flash_buffers={},
        _flash_attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1),
        _device_metadata_enabled=deferred,
        _device_metadata_tasks=(),
    )


@pytest.fixture
def schedule_calls(monkeypatch):
    calls = []

    def metadata(lengths, heads, kv_heads, **kwargs):
        calls.append((lengths, heads, kv_heads, kwargs))
        return torch.full((8,), len(calls), dtype=torch.int32, device=lengths.device)

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache_metadata", metadata)
    return calls


def test_metadata_padding_and_documented_attributes(schedule_calls):
    flash = flash_mla.build_flash_mla_metadata(_builder(), _common())
    assert flash.cu.tolist() == [0, 2, 4, 6]
    assert flash.used_q.tolist() == [2, 2, 0]
    assert flash.cache_lens.tolist() == [5, 2, 0]
    assert flash.slots.tolist() == [384, 385, 256, 257, -1, -1]
    assert flash.positions.tolist() == [3, 4, 0, 1, 0, 0]
    assert flash.token_live.tolist() == [True, True, True, True, False, False]
    assert (flash.query_capacity, flash.kv_capacity) == (6, 256)
    assert flash.query.shape == (6, 64, 576)
    assert len(schedule_calls) == 2  # Meta sizing followed by real scheduling.
    assert schedule_calls[0][0].device.type == "meta"
    assert schedule_calls[1][0] is flash.cache_lens
    kwargs = flash.contract.attention_kwargs(
        block_table=flash.block_table,
        cache_seqlens=flash.cache_lens,
        cu_seqlens_q=flash.cu,
        seqused_q=flash.used_q,
        attn_mask=flash.attn_mask,
        metadata=flash.schedule,
    )
    for call in schedule_calls:
        assert call[1:3] == (64, 1)
        assert call[3]["max_seqlen_q"] == kwargs["max_seqlen_q"] == -1
        assert call[3]["max_seqlen_kv"] == kwargs["max_seqlen_kv"] == -1
    assert kwargs["metadata"] is flash.schedule


def test_stable_buffers_refresh_device_lengths_and_visibility(schedule_calls):
    builder, common = _builder(), _common(causal=False)
    first = flash_mla.build_flash_mla_metadata(builder, common)
    pointers = [
        getattr(first, name).data_ptr() for name in ("query", "schedule", "cu", "used_q", "cache_lens", "block_table")
    ]
    original_slots = common.slot_mapping.clone()
    common.seq_lens.copy_(torch.tensor([20, 0], dtype=torch.int32))
    common.query_start_loc.copy_(torch.tensor([0, 1, 4], dtype=torch.int32))
    common.slot_mapping[0] = -1
    second = flash_mla.build_flash_mla_metadata(builder, common)
    assert first is second
    assert pointers == [
        getattr(second, name).data_ptr() for name in ("query", "schedule", "cu", "used_q", "cache_lens", "block_table")
    ]
    assert second.cache_lens.tolist() == [20, 0, 0]
    assert second.used_q.tolist() == [1, 0, 0]
    assert second.token_live.tolist() == [True, False, False, False, False, False]
    assert second.slots.tolist() == [-1] * 6
    torch.testing.assert_close(common.slot_mapping[1:], original_slots[1:])
    assert second.schedule.tolist() == [3] * 8
    assert len(schedule_calls) == 3  # Same bucket does not invoke Meta again.


def test_prefill_buffers_are_batch_owned(schedule_calls):
    builder, common = _builder(), _common()
    first = flash_mla.build_flash_mla_metadata(builder, common)
    second = flash_mla.build_flash_mla_metadata(builder, common)
    assert first is not second
    assert not builder._flash_buffers


def test_deferred_task_populates_buffers_before_consumer(schedule_calls):
    builder, common = _builder(deferred=True), _common(causal=False)
    flash = flash_mla.build_flash_mla_metadata(builder, common)
    assert len(schedule_calls) == 1
    (task,) = builder._device_metadata_tasks
    assert task.group_id == id(flash.schedule)
    common.seq_lens[0] = 19
    task.run()
    assert flash.cache_lens.tolist() == [19, 2, 0]
    assert len(schedule_calls) == 2


def test_empty_tokens_skip_both_meta_and_real_operator(schedule_calls):
    common = _common(num_input_tokens=0)
    common.num_reqs = common.num_actual_tokens = common.max_query_len = 0
    common.query_start_loc = torch.tensor([0], dtype=torch.int32)
    flash = flash_mla.build_flash_mla_metadata(_builder(), common)
    assert flash.query.shape[0] == 0
    assert flash.schedule.numel() == 0
    assert not schedule_calls


def test_all_inactive_rows_are_refreshed_without_host_scalar_branch(schedule_calls):
    common = _common()
    common.seq_lens.zero_()
    flash = flash_mla.build_flash_mla_metadata(_builder(), common)
    assert flash.used_q.tolist() == [0, 0, 0]
    assert flash.slots.tolist() == [-1] * 6
    assert not flash.token_live.any()
    assert len(schedule_calls) == 2


@pytest.mark.parametrize("heads", [64, 96])
def test_supported_actual_q_heads(heads):
    assert flash_mla.FlashMLAContract(num_heads_q=heads, mask_mode=3).num_heads_q == heads


@pytest.mark.parametrize("heads", [0, 16, 32, 128])
def test_unsupported_actual_q_heads_fail(heads):
    with pytest.raises(ValueError, match="64 or 96"):
        flash_mla.FlashMLAContract(num_heads_q=heads, mask_mode=3)


@pytest.mark.parametrize("name", ["max_seqlen_q", "max_seqlen_kv"])
def test_capacity_must_not_leak_into_external_attrs(name):
    with pytest.raises(ValueError, match="must be -1"):
        flash_mla.FlashMLAContract(num_heads_q=64, mask_mode=3, **{name: 128})


def test_kernel_size_is_not_manager_size(schedule_calls):
    builder = _builder()
    assert flash_mla.build_flash_mla_metadata(builder, _common()).kv_capacity == 256
    builder.kernel_block_size = 16
    with pytest.raises(ValueError, match="kernel block size 128"):
        flash_mla.build_flash_mla_metadata(builder, _common())


def test_missing_meta_support_fails_without_schedule_guess(monkeypatch):
    def missing(*args, **kwargs):
        raise NotImplementedError("Meta kernel is absent")

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache_metadata", missing)
    builder = _builder()
    with pytest.raises(RuntimeError, match="Meta support"):
        flash_mla.build_flash_mla_metadata(builder, _common(causal=False))
    assert not builder._flash_buffers


def test_real_schedule_shape_must_match_meta(monkeypatch):
    def mismatch(lengths, *args, **kwargs):
        return torch.ones(8 if lengths.device.type == "meta" else 9, dtype=torch.int32, device=lengths.device)

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache_metadata", mismatch)
    with pytest.raises(RuntimeError, match="shape differs"):
        flash_mla.build_flash_mla_metadata(_builder(), _common())


def test_public_package_loader_and_no_fallback(monkeypatch):
    calls = []
    ops = SimpleNamespace(flash_mla_with_kvcache=object(), flash_mla_with_kvcache_metadata=object())

    def imported(name):
        calls.append(name)
        return ops

    flash_mla._get_flash_mla_ops.cache_clear()
    try:
        monkeypatch.setattr(flash_mla, "import_module", imported)
        assert flash_mla._get_flash_mla_ops() == (ops.flash_mla_with_kvcache, ops.flash_mla_with_kvcache_metadata)
        assert calls == ["cann_ops_transformer.ops"]
        flash_mla._get_flash_mla_ops.cache_clear()
        monkeypatch.setattr(flash_mla, "import_module", lambda _: SimpleNamespace())
        with pytest.raises(RuntimeError, match="matching cann_ops_transformer.ops"):
            flash_mla._get_flash_mla_ops()
    finally:
        flash_mla._get_flash_mla_ops.cache_clear()


def test_strided_pa_bbnd_cache_is_valid_without_repacking():
    pages, block_size, dim = 4, 128, 576
    page_stride, offset = block_size * dim + 64, 13
    backing = torch.empty(offset + (pages - 1) * page_stride + block_size * dim, dtype=torch.bfloat16)
    cache = torch.as_strided(backing, (pages, block_size, 1, dim), (page_stride, dim, dim, 1), offset)
    flash_mla.validate_flash_mla_kv_cache(cache, expected_dtype=cache.dtype, expected_device=cache.device)
    assert cache.storage_offset() == offset
    overlapping = torch.as_strided(backing, cache.shape, (block_size * dim - 1, dim, dim, 1), offset)
    with pytest.raises(ValueError, match="overlap"):
        flash_mla.validate_flash_mla_kv_cache(overlapping, expected_dtype=cache.dtype, expected_device=cache.device)


def _valid_platform_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            enforce_eager=True,
            use_mla=True,
            is_hybrid=False,
            dtype=torch.bfloat16,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={},
    )


def test_platform_contract_rejects_non_mrv2_and_quantized_cache(monkeypatch):
    monkeypatch.setitem(
        ascend_envs.env_variables,
        "VLLM_ASCEND_ENABLE_FLASH_MLA",
        lambda: True,
    )
    monkeypatch.setattr("vllm_ascend.device.device_config.is_950", lambda: True)
    monkeypatch.setattr(ascend_platform, "model_uses_sfa_sparse", lambda config: False)

    ascend_platform._validate_flash_mla_config(_valid_platform_config())

    invalid = _valid_platform_config()
    invalid.use_v2_model_runner = False
    invalid.cache_config.cache_dtype = "fp8"
    with pytest.raises(ValueError, match="MRV2.*KV cache dtype"):
        ascend_platform._validate_flash_mla_config(invalid)


@pytest.mark.parametrize("model_v_dim", [128, 256, 512])
@pytest.mark.parametrize("heads", [64, 96])
def test_flash_mla_model_dimensions_are_not_absorbed_dimensions(monkeypatch, model_v_dim, heads):
    config = _valid_platform_config()
    config.model_config.runner_type = "generate"
    monkeypatch.setitem(ascend_envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    monkeypatch.setattr(mla_v1, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mla_v1, "get_ascend_config", lambda: SimpleNamespace(enable_kv_nz=False))
    monkeypatch.setattr(mla_v1, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    monkeypatch.setattr(mla_v1, "enabling_mlapo", lambda _: False)
    monkeypatch.setattr(mla_v1, "enable_fa_quant", lambda *_: False)
    impl = mla_v1.AscendMLAImpl(
        num_heads=heads,
        head_size=576,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=model_v_dim,
        rotary_emb=None,
        q_b_proj=None,
        kv_b_proj=None,
        o_proj=None,
    )
    assert impl.qk_head_dim == 192
    assert impl.v_head_dim == model_v_dim
    assert impl.kv_lora_rank == 512


def test_flash_mla_rejects_truncated_output_before_execution():
    impl = mla_v1.AscendMLAImpl.__new__(mla_v1.AscendMLAImpl)
    metadata = SimpleNamespace(flash=SimpleNamespace(query=torch.empty(2, 64, 576)))
    with pytest.raises(ValueError, match="one hidden row"):
        impl._forward_flash("test", torch.empty(2, 64), None, metadata, torch.empty(1, 64))


def test_platform_allows_graph_mode(monkeypatch):
    monkeypatch.setitem(ascend_envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    monkeypatch.setattr("vllm_ascend.device.device_config.is_950", lambda: True)
    monkeypatch.setattr(ascend_platform, "model_uses_sfa_sparse", lambda config: False)
    config = _valid_platform_config()
    config.model_config.enforce_eager = False
    ascend_platform._validate_flash_mla_config(config)


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_hybrid_flash_mla_requires_dcp_one(monkeypatch, dcp):
    monkeypatch.setitem(ascend_envs.env_variables, "VLLM_ASCEND_ENABLE_FLASH_MLA", lambda: True)
    monkeypatch.setattr("vllm_ascend.device.device_config.is_950", lambda: True)
    monkeypatch.setattr(ascend_platform, "model_uses_sfa_sparse", lambda _: False)
    config = _valid_platform_config()
    config.model_config.is_hybrid = True
    config.parallel_config.decode_context_parallel_size = dcp
    if dcp == 1:
        ascend_platform._validate_flash_mla_config(config)
    else:
        with pytest.raises(ValueError, match="DCP size must be 1"):
            ascend_platform._validate_flash_mla_config(config)
