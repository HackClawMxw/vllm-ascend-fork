# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""External MLA adapter; stable-buffer lifecycle follows PR #16468 (48c8a870).

Only the non-DCP, absorbed MLA path is selected here. Cache storage remains
owned by MRV2 (#16456); this module never allocates or repacks persistent KV.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch

from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask


@dataclass(frozen=True)
class FlashMLAContract:
    """Shared external API attributes, not internal buffer capacities."""

    num_heads_q: int
    mask_mode: int
    num_heads_kv: int = 1
    head_dim_qk: int = 576
    head_dim_v: int = 512
    layout_q: str = "TND"
    layout_kv: str = "PA_BBND"
    layout_out: str = "NTD"
    softmax_scale: float = 1.0
    return_softmax_lse: bool = False
    max_seqlen_q: int = -1
    max_seqlen_kv: int = -1

    def __post_init__(self) -> None:
        if self.num_heads_q not in (64, 96) or self.num_heads_kv != 1:
            raise ValueError("External FlashMLA requires 64 or 96 actual Q heads and one KV head.")
        if (self.head_dim_qk, self.head_dim_v) != (576, 512):
            raise ValueError("FlashMLA requires head_dim_qk=576 and head_dim_v=512.")
        if (self.max_seqlen_q, self.max_seqlen_kv) != (-1, -1):
            raise ValueError("External FlashMLA max_seqlen_q/kv must be -1; capacities are internal only.")
        if self.mask_mode not in (0, 3):
            raise ValueError("FlashMLA mask_mode must be 0 (none) or 3 (causal).")
        if (self.layout_q, self.layout_kv, self.layout_out) != ("TND", "PA_BBND", "NTD"):
            raise ValueError("This integration requires TND/PA_BBND/NTD layouts.")

    def with_softmax_scale(self, softmax_scale: float) -> "FlashMLAContract":
        return replace(self, softmax_scale=float(softmax_scale))

    def metadata_kwargs(self, cu_seqlens_q: torch.Tensor, seqused_q: torch.Tensor) -> dict[str, Any]:
        return {
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_v": self.head_dim_v,
            "mask_mode": self.mask_mode,
            "layout_q": self.layout_q,
        }

    def attention_kwargs(
        self,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor,
        attn_mask: torch.Tensor | None,
        metadata: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "attn_mask": attn_mask,
            "metadata": metadata,
            "head_dim_v": self.head_dim_v,
            "softmax_scale": self.softmax_scale,
            "mask_mode": self.mask_mode,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "layout_q": self.layout_q,
            "layout_kv": self.layout_kv,
            "layout_out": self.layout_out,
            "return_softmax_lse": self.return_softmax_lse,
        }


@dataclass
class FlashMLAMetadata:
    """Stable inputs updated outside the model graph, as in reference #1."""

    query: torch.Tensor
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    live_boundaries: torch.Tensor
    token_live: torch.Tensor
    positions: torch.Tensor
    attn_mask: torch.Tensor | None
    query_capacity: int
    kv_capacity: int
    is_prefill: bool
    contract: FlashMLAContract


@lru_cache
def _get_flash_mla_ops() -> tuple[Callable, Callable]:
    """Use the documented public wrappers, never an in-tree fallback."""
    try:
        namespace = import_module("cann_ops_transformer.ops")
        return namespace.flash_mla_with_kvcache, namespace.flash_mla_with_kvcache_metadata
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "FlashMLA requires flash_mla_with_kvcache and flash_mla_with_kvcache_metadata "
            "from the matching cann_ops_transformer.ops package."
        ) from exc


def ensure_flash_mla_ops_loaded() -> None:
    _get_flash_mla_ops()


def flash_mla_with_kvcache_metadata(*args: Any, **kwargs: Any) -> torch.Tensor:
    _, metadata_op = _get_flash_mla_ops()
    return metadata_op(*args, **kwargs)


def flash_mla_with_kvcache(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    attention_op, _ = _get_flash_mla_ops()
    return attention_op(*args, **kwargs)


def _flash_mla_schedule(flash: FlashMLAMetadata, *, meta: bool = False) -> torch.Tensor:
    """Ask the same operator for sizing and real scheduling; do not guess core counts."""

    def tensor(value):
        return torch.empty_like(value, device="meta") if meta else value

    result = flash_mla_with_kvcache_metadata(
        tensor(flash.cache_lens),
        flash.contract.num_heads_q,
        flash.contract.num_heads_kv,
        **flash.contract.metadata_kwargs(tensor(flash.cu), tensor(flash.used_q)),
    )
    if result.dtype != torch.int32 or result.ndim != 1 or result.numel() == 0:
        raise RuntimeError("External FlashMLA metadata must be a non-empty one-dimensional int32 tensor.")
    expected_device = torch.device("meta") if meta else flash.cache_lens.device
    if result.device != expected_device:
        raise RuntimeError(f"FlashMLA metadata must be on {expected_device}, got {result.device}.")
    return result


def init_flash_mla_metadata(builder: Any, impl: Any) -> None:
    ensure_flash_mla_ops_loaded()
    # Validate the actual per-rank Q head count; never replace it with a
    # model-global count or pad Q to make an unsupported geometry pass.
    FlashMLAContract(num_heads_q=impl.num_heads, mask_mode=3)
    builder.flash_num_heads = impl.num_heads
    builder._flash_buffers = {}
    builder._flash_attn_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=builder.device), diagonal=1)


def build_flash_mla_metadata(builder: Any, common: Any) -> FlashMLAMetadata:
    """Reference #1's shared eager/graph lifecycle, adapted only at the API boundary.

    The extra zero-used row owns token padding. Device lengths remain
    authoritative; no device-to-host scalar reads or per-step Python maxima.
    Runtime input values retain MRV2's producer contract. Shape/dtype mistakes
    fail before dispatch; numerical/lifecycle acceptance still requires NPU tests.
    """
    batch = common.num_reqs
    tokens = max(common.num_actual_tokens, common.num_input_tokens)
    table = common.block_table_tensor[:batch]
    block_size = builder.kernel_block_size or builder.kv_cache_spec.block_size
    if block_size != 128:
        raise ValueError("External FlashMLA requires kernel block size 128, not the manager/interleave size.")
    if batch < 0 or batch + 1 >= 65536 or common.num_actual_tokens < 0:
        raise ValueError("Invalid FlashMLA batch size.")
    if table.ndim != 2 or table.shape[0] != batch or table.shape[1] <= 0:
        raise ValueError("FlashMLA requires the current group's non-empty kernel block table.")
    if common.positions is None or common.positions.numel() < common.num_actual_tokens:
        raise ValueError("FlashMLA requires real MRV2 positions for all actual tokens.")
    if common.query_start_loc.numel() < batch + 1 or common.seq_lens.numel() < batch:
        raise ValueError("FlashMLA requires query boundaries and cache lengths for every request.")
    if common.slot_mapping.numel() < common.num_actual_tokens:
        raise ValueError("FlashMLA requires write slots for all actual tokens.")
    if (
        common.query_start_loc.dtype != torch.int32
        or common.seq_lens.dtype != torch.int32
        or table.dtype != torch.int32
    ):
        raise ValueError("FlashMLA MRV2 query boundaries, cache lengths and block table must be int32.")

    # Keep reference #1's buffer ownership policy. Eager prefill shapes do
    # not accumulate in the builder; decode/graph shapes keep stable addresses.
    key = batch, tokens, table.shape[1], common.causal
    is_eager_prefill = common.causal and common.max_query_len > builder.decode_threshold
    buffers = {} if is_eager_prefill else builder._flash_buffers
    if key not in buffers:
        rows = batch + 1
        int_args = {"dtype": torch.int32, "device": builder.device}
        buffers[key] = FlashMLAMetadata(
            query=torch.empty(
                (tokens, builder.flash_num_heads, 576), dtype=builder.kv_cache_spec.dtype, device=builder.device
            ),
            schedule=torch.empty(0, **int_args),
            cu=torch.zeros(rows + 1, **int_args),
            used_q=torch.zeros(rows, **int_args),
            cache_lens=torch.zeros(rows, **int_args),
            block_table=torch.zeros((rows, table.shape[1]), **int_args),
            slots=torch.full((tokens,), -1, dtype=torch.int64, device=builder.device),
            live_boundaries=torch.zeros(tokens + 1, **int_args),
            token_live=torch.zeros(tokens, dtype=torch.bool, device=builder.device),
            positions=torch.zeros(tokens, dtype=torch.int64, device=builder.device),
            attn_mask=builder._flash_attn_mask if common.causal else None,
            query_capacity=tokens,
            kv_capacity=table.shape[1] * block_size,
            is_prefill=common.max_query_len > builder.decode_threshold,
            contract=FlashMLAContract(num_heads_q=builder.flash_num_heads, mask_mode=3 if common.causal else 0),
        )
        flash = buffers[key]
        if tokens:
            try:
                flash.schedule = torch.empty_like(_flash_mla_schedule(flash, meta=True), device=builder.device)
            except (NotImplementedError, RuntimeError) as exc:
                del buffers[key]
                raise RuntimeError(
                    "The external FlashMLA package must provide matching metadata Meta support."
                ) from exc
    flash = buffers[key]
    flash.is_prefill = common.max_query_len > builder.decode_threshold

    def build_metadata() -> None:
        flash.cu[: batch + 1].copy_(common.query_start_loc[: batch + 1])
        flash.cu[batch + 1].fill_(tokens)
        flash.used_q[:batch].copy_(flash.cu[1 : batch + 1] - flash.cu[:batch])
        flash.used_q[:batch].masked_fill_(common.seq_lens[:batch] <= 0, 0)
        flash.used_q[batch:].zero_()
        flash.cache_lens[:batch].copy_(common.seq_lens[:batch])
        flash.cache_lens[batch:].zero_()
        flash.block_table[:batch].copy_(table)
        flash.block_table[batch:].zero_()
        flash.slots.fill_(-1)
        slots = common.slot_mapping[:tokens]
        flash.slots[: slots.shape[0]].copy_(slots)
        flash.live_boundaries.zero_()
        live_rows = (flash.used_q > 0).to(torch.int32)
        flash.live_boundaries.scatter_add_(0, flash.cu[:-1].long(), live_rows)
        flash.live_boundaries.scatter_add_(0, (flash.cu[:-1] + flash.used_q).long(), -live_rows)
        flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)
        flash.slots.masked_fill_(~flash.token_live, -1)
        flash.positions.zero_()
        positions = common.positions[: common.num_actual_tokens]
        flash.positions[: positions.shape[0]].copy_(positions)
        if tokens:
            schedule = _flash_mla_schedule(flash)
            if schedule.shape != flash.schedule.shape:
                raise RuntimeError("FlashMLA metadata shape differs from its Meta-sized stable buffer.")
            flash.schedule.copy_(schedule)

    if builder._device_metadata_enabled:
        builder._device_metadata_tasks = (
            DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_metadata, id(flash.schedule)),
        )
    else:
        build_metadata()
    return flash


def validate_flash_mla_kv_cache(
    k_cache: torch.Tensor, *, expected_dtype: torch.dtype, expected_device: torch.device
) -> None:
    """Validate #2's PA_BBND view without repairing or copying storage."""
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("FlashMLA requires one fused KV cache Tensor.")
    if k_cache.ndim != 4 or k_cache.shape[1:] != (128, 1, 576):
        raise ValueError(f"FlashMLA PA_BBND cache must have shape [P, 128, 1, 576], got {tuple(k_cache.shape)}.")
    if k_cache.stride()[1:] != (576, 576, 1):
        raise ValueError(f"FlashMLA cache must keep PA_BBND inner strides (576, 576, 1), got {k_cache.stride()}.")
    if k_cache.stride(0) < 128 * 576:
        raise ValueError("FlashMLA physical pages overlap.")
    if k_cache.dtype != expected_dtype or k_cache.dtype != torch.bfloat16:
        raise ValueError(f"FlashMLA cache must be BF16 and match Q, got {k_cache.dtype} and {expected_dtype}.")
    if k_cache.device != expected_device:
        raise ValueError(f"FlashMLA cache must be on {expected_device}, got {k_cache.device}.")
