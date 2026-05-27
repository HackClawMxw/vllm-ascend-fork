# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend NPU TurboQuant attention backend.

Independent attention backend for TurboQuant KV cache compression on Ascend NPU.
Activated via --kv-cache-dtype turboquant_* on the command line.

Cache layout (no leading 2 dimension, K+V combined per slot):
  (num_blocks, block_size, num_kv_heads, slot_size_aligned)

Does NOT modify AscendAttentionBackend, AscendC8AttentionBackendImpl,
AscendMLABackend, or AscendSFABackend — all existing paths remain untouched.
"""

import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch_npu

from vllm.config import get_current_vllm_config
from vllm_ascend.attention.tq_config import TurboQuantConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm_ascend.attention.ops.turboquant_decode import (
    npu_turboquant_decode_attention,
    npu_turboquant_full_dequant_kv,
)
from vllm_ascend.attention.ops.turboquant_store import npu_turboquant_store

# Module-level diagnostic flag (survives graph capture warmup)
_diag_store_roundtrip_done = [False]


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device)."""
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class AscendTurboQuantBackend(AttentionBackend):
    """TurboQuant attention backend for Ascend NPU."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["AscendTurboQuantImpl"]:
        return AscendTurboQuantImpl

    @staticmethod
    def get_builder_cls() -> type["AscendTurboQuantMetadataBuilder"]:
        return AscendTurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Layout: (num_blocks, block_size, num_kv_heads, slot_size_aligned)
        """
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0


@dataclass
class AscendTurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention on Ascend NPU."""

    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0


class AscendTurboQuantMetadataBuilder(
    AttentionMetadataBuilder[AscendTurboQuantMetadata]
):
    """Builds AscendTurboQuantMetadata from scheduler output."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> AscendTurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata

        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return AscendTurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class AscendTurboQuantImpl(AttentionImpl[AscendTurboQuantMetadata]):
    """TurboQuant attention implementation for Ascend NPU.

    Uses pure PyTorch/NPU operations for quantization and dequantization,
    then delegates attention computation to npu_fused_infer_attention_score.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self._val_data_bytes = math.ceil(head_size * self.tq_config.effective_value_quant_bits / 8)
        self._causal_mask_cache = None

    def _get_causal_mask(self, device: torch.device):
        """Return the standard 2048x2048 upper-triangular causal mask."""
        cached = self._causal_mask_cache
        if cached is None:
            cached = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1,
            ).to(device)
            self._causal_mask_cache = cached
        elif cached.device != device:
            cached = cached.to(device)
        return cached

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints)."""
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H

            # Centroids are normally registered as a buffer by
            # Attention._init_turboquant_buffers (vllm GPU).  The deployed
            # v0.19.1 base does not have that method, so we compute them
            # lazily on first use.
            if not hasattr(layer, "_tq_centroids"):
                from vllm_ascend.attention.tq_centroids import get_centroids
                layer.register_buffer(
                    "_tq_centroids",
                    get_centroids(D, self.tq_config.centroid_bits).to(device),
                )

            c = layer._tq_centroids.to(device=device, dtype=torch.float32)
            c_sorted, _ = c.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache."""
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        # Extract the single combined TQ cache tensor from the tuple
        if isinstance(kv_cache, (tuple, list)):
            kv_cache = kv_cache[0]

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

        # ---- TQ DIAGNOSTIC: store round-trip verification ----
        # Use module-level flag to survive graph capture warmup.
        # Only run for real prefill (N > 1), skip graph capture dummy calls (N=1).
        if N > 1 and not _diag_store_roundtrip_done:
            _diag_store_roundtrip_done[0] = True  # type: ignore
            self._diag_store_roundtrip(
                k, v, kv_cache, slot_mapping, layer, N,
            )
        # ---- END TQ DIAGNOSTIC ----

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | tuple[torch.Tensor, ...],
        attn_metadata: AscendTurboQuantMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        # Extract the single combined TQ cache tensor from the tuple
        if isinstance(kv_cache, (tuple, list)):
            kv_cache = kv_cache[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        self._ensure_on_device(layer, q.device)

        # Store compressed K/V into cache (inside forward to work with
        # torch.compile — matching standard AscendAttention pattern).
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, attn_metadata.slot_mapping, layer)

        Pi = layer._tq_Pi
        PiT = layer._tq_PiT
        centroids = layer._tq_centroids

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, centroids, Pi
            )
        elif num_decodes == 0:
            attn_out = self._prefill_attention(
                q, k, v, kv_cache, attn_metadata, centroids, Pi, layer
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch_threshold)
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size,
                device=q.device, dtype=q.dtype,
            )

            decode_meta = AscendTurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[:num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, centroids, Pi
            )

            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
            prefill_qsl = attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            prefill_meta = AscendTurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                centroids,
                Pi,
                layer,
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store-side round-trip diagnostic                                    #
    # ------------------------------------------------------------------ #
    def _diag_store_roundtrip(self, k, v, kv_cache, slot_mapping, layer, N):
        """Verify store→unpack round-trip for the first valid token."""
        import math as _math
        from vllm_ascend.attention.ops.turboquant_decode import (
            _unpack_mse_key, _unpack_fp8_key, _unpack_value,
        )

        tq = self.tq_config
        block_size = kv_cache.shape[1]
        mse_bits = tq.key_mse_bits
        key_fp8 = tq.key_fp8
        mse_bytes = _math.ceil(self.head_size * mse_bits / 8) if not key_fp8 else 0
        val_data_bytes = _math.ceil(
            self.head_size * tq.effective_value_quant_bits / 8
        )
        centroids = layer._tq_centroids
        Pi = layer._tq_Pi

        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return

        first_idx = valid_mask.nonzero()[0, 0].item()
        slot_idx = slot_mapping[first_idx].item()
        bid = slot_idx // block_size
        pid = slot_idx % block_size

        print(f"[TQ-DIAG-STORE] token={first_idx} slot={slot_idx} "
              f"block={bid} pos={pid} block_size={block_size} "
              f"slot_size={kv_cache.shape[-1]}")
        print(f"[TQ-DIAG-STORE] mse_bits={mse_bits} mse_bytes={mse_bytes} "
              f"key_packed_size={tq.key_packed_size} val_data_bytes={val_data_bytes} "
              f"key_fp8={key_fp8} norm_corr={tq.norm_correction}")
        print(f"[TQ-DIAG-STORE] centroids[:4]={centroids[:4].tolist()}")

        for h in range(min(self.num_kv_heads, 2)):
            slot_h = kv_cache[bid, pid, h, :]  # (slot_size,) uint8

            # --- Dequantize key ---
            if key_fp8:
                k_deq = _unpack_fp8_key(
                    slot_h.unsqueeze(0), self.head_size
                ).squeeze(0)
            else:
                k_deq = _unpack_mse_key(
                    slot_h.unsqueeze(0), centroids, mse_bits, mse_bytes,
                    self.head_size, tq.norm_correction,
                ).squeeze(0)
                if Pi is not None:
                    k_deq = (k_deq.float() @ Pi).to(torch.float16)

            # --- Dequantize value ---
            v_deq = _unpack_value(
                slot_h.unsqueeze(0), tq.key_packed_size,
                tq.effective_value_quant_bits, val_data_bytes,
                self.head_size,
            ).squeeze(0)

            k_orig = k[first_idx, h, :].detach().float()
            v_orig = v[first_idx, h, :].detach().float()
            k_deq_f = k_deq.float()
            v_deq_f = v_deq.float()

            k_err = (k_orig - k_deq_f).abs()
            v_err = (v_orig - v_deq_f).abs()

            # vec_norm vs original norm
            if not key_fp8:
                nb = slot_h[mse_bytes:mse_bytes + 2].contiguous()
                vn = nb.view(torch.uint16).view(torch.float16).to(torch.float32).item()
                no = k_orig.norm().item()
                print(f"[TQ-DIAG-STORE] head={h} vec_norm={vn:.6f} "
                      f"orig_norm={no:.6f} ratio={vn / (no + 1e-8):.4f}")

            # Cosine similarity (direction quality)
            cos_sim = torch.nn.functional.cosine_similarity(
                k_orig.unsqueeze(0), k_deq_f.unsqueeze(0)
            ).item()

            print(f"[TQ-DIAG-STORE] head={h} key_err max={k_err.max().item():.6f} "
                  f"mean={k_err.mean().item():.6f} cos_sim={cos_sim:.6f}")
            print(f"[TQ-DIAG-STORE] head={h} val_err max={v_err.max().item():.6f} "
                  f"mean={v_err.mean().item():.6f}")

            if h == 0:
                print(f"[TQ-DIAG-STORE] k_orig[:8] ={k_orig[:8].tolist()}")
                print(f"[TQ-DIAG-STORE] k_deq[:8]  ={k_deq_f[:8].tolist()}")
                print(f"[TQ-DIAG-STORE] v_orig[:8] ={v_orig[:8].tolist()}")
                print(f"[TQ-DIAG-STORE] v_deq[:8]  ={v_deq_f[:8].tolist()}")
                # Raw slot hex dump (first 16 bytes)
                print(f"[TQ-DIAG-STORE] slot_raw[:16]={slot_h[:16].tolist()}")

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache                                      #
    # ------------------------------------------------------------------ #
    def _store_kv(self, key, value, kv_cache, slot_mapping, layer):
        npu_turboquant_store(
            key, value, kv_cache, slot_mapping,
            layer._tq_PiT, layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
        )

    # ------------------------------------------------------------------ #
    #  Prefill attention                                                   #
    # ------------------------------------------------------------------ #
    def _prefill_attention(self, q, k, v, kv_cache, attn_metadata,
                           centroids, Pi, layer):
        N, Hq, D = q.shape

        # First-chunk prefill: all K/V are in the current batch, no cached KV
        if attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(q, k, v, attn_metadata)

        # Continuation prefill: need to attend to previously cached KV
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        query_start_loc = attn_metadata.query_start_loc

        # Dequantize all cached KV
        key_dequant, value_dequant = npu_turboquant_full_dequant_kv(
            kv_cache, block_table, seq_lens.tolist(),
            self.num_kv_heads, D,
            self.tq_config.key_fp8,
            self.tq_config.key_mse_bits,
            self.tq_config.key_packed_size,
            self.tq_config.effective_value_quant_bits,
            self._val_data_bytes,
            centroids,
            self.tq_config.norm_correction,
            Pi,
        )

        # Concatenate dequantized cached KV with current batch KV
        # and run standard flash attention
        all_keys = []
        all_values = []
        all_queries = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]

        for i in range(len(seq_lens)):
            q_len = query_start_loc[i + 1] - query_start_loc[i]
            kv_len = seq_lens[i]

            # Current batch KV for this request
            q_start = query_start_loc[i].item()
            q_end = query_start_loc[i + 1].item()

            all_queries.append(q[q_start:q_end])

            # Cached KV for this request (before current batch)
            cached_kv_len = kv_len - q_len
            if cached_kv_len > 0 and key_dequant.numel() > 0:
                # Find start of this request in dequantized KV
                kv_start = sum(seq_lens[:i].tolist()) if i > 0 else 0
                # Only take the previously cached portion (not current batch)
                cached_k = key_dequant[kv_start:kv_start + cached_kv_len]
                cached_v = value_dequant[kv_start:kv_start + cached_kv_len]
                all_keys.append(torch.cat([cached_k, k[q_start:q_end]], dim=0))
                all_values.append(torch.cat([cached_v, v[q_start:q_end]], dim=0))
            else:
                all_keys.append(k[q_start:q_end])
                all_values.append(v[q_start:q_end])

            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + kv_len)

        cat_q = torch.cat(all_queries, dim=0)
        cat_k = torch.cat(all_keys, dim=0)
        cat_v = torch.cat(all_values, dim=0)

        # Derive individual (non-cumulative) sequence lengths
        q_lens = [cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                  for i in range(len(cu_seqlens_q) - 1)]
        k_lens = [cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                  for i in range(len(cu_seqlens_k) - 1)]
        attn_mask = self._get_causal_mask(cat_q.device)

        output, _ = torch_npu.npu_fused_infer_attention_score(
            cat_q, cat_k, cat_v,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            scale=self.scale,
            sparse_mode=3,
            atten_mask=attn_mask,
            actual_seq_lengths=q_lens,
            actual_seq_lengths_kv=k_lens,
        )

        return output

    def _flash_attn_varlen(self, q, k, v, attn_metadata):
        """First-chunk prefill using NPU fused attention."""
        seq_lens = attn_metadata.seq_lens.tolist()
        attn_mask = self._get_causal_mask(q.device)

        output, _ = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            scale=self.scale,
            sparse_mode=3,
            atten_mask=attn_mask,
            actual_seq_lengths=seq_lens,
            actual_seq_lengths_kv=seq_lens,
        )

        return output

    # ------------------------------------------------------------------ #
    #  Decode attention                                                    #
    # ------------------------------------------------------------------ #
    def _decode_attention(self, q, kv_cache, attn_metadata, centroids, Pi):
        return npu_turboquant_decode_attention(
            query=q,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens.tolist(),
            scale=self.scale,
            attn_metadata=attn_metadata,
            layer=None,
            key_fp8=self.tq_config.key_fp8,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            head_dim=self.head_size,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            centroids=centroids,
            norm_correction=self.tq_config.norm_correction,
            Pi=Pi,
        )
