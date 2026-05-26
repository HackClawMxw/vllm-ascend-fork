# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU (PyTorch) implementation of TurboQuant decode attention.

Replaces the Triton decode kernels with pure PyTorch operations
that run on Ascend NPU via torch_npu.

Strategy for Phase 0: dequantize KV from TQ cache to FP16,
then use npu_fused_infer_attention_score for standard attention.

This is the simplest correct implementation. Performance optimization
(fused bit-unpack + attention) can be added later as a C++ custom op.
"""

import math

import torch
import torch_npu


def _unpack_mse_key(
    slot_data: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    mse_bytes: int,
    head_dim: int,
    norm_correction: bool = False,
) -> torch.Tensor:
    """Unpack MSE key indices and reconstruct key vector.

    Args:
        slot_data: (..., slot_size) uint8 slot data for one position/head.
        centroids: (n_centroids,) float32 centroids.
        mse_bits: 3 or 4 bits.
        mse_bytes: bytes for packed MSE indices.
        head_dim: attention head dimension.
        norm_correction: whether to normalize centroid vectors.

    Returns:
        key: (..., head_dim) float16 reconstructed key.
    """
    n_centroids = 1 << mse_bits

    # Unpack bit-packed indices
    mse_raw = slot_data[..., :mse_bytes].to(torch.int32)

    if mse_bits == 4:
        # Two 4-bit indices per byte
        idx_lo = mse_raw & 0xF
        idx_hi = (mse_raw >> 4) & 0xF
        indices = torch.stack([idx_lo, idx_hi], dim=-1).reshape(
            *mse_raw.shape[:-1], head_dim
        )
    elif mse_bits == 3:
        # Eight 3-bit indices per 3 bytes
        n_groups = head_dim // 8
        raw = slot_data[..., : n_groups * 3].to(torch.int32)
        # Read 2 bytes per group for 16-bit window
        raw_16 = torch.zeros(*raw.shape[:-1], n_groups, 4, dtype=torch.int32, device=raw.device)
        for g in range(n_groups):
            raw_16[..., g, 0] = raw[..., g * 3]
            raw_16[..., g, 1] = raw[..., g * 3 + 1]
            raw_16[..., g, 2] = raw[..., g * 3 + 2]

        # For each of the 8 indices in a group
        indices_list = []
        for b in range(8):
            byte_idx = (b * 3) // 8
            bit_off = (b * 3) % 8
            if bit_off <= 5:
                idx = (raw_16[..., byte_idx] >> bit_off) & 0x7
            else:
                idx = ((raw_16[..., byte_idx] >> bit_off) |
                       (raw_16[..., byte_idx + 1] * (1 << (8 - bit_off)))) & 0x7
            indices_list.append(idx)
        indices = torch.stack(indices_list, dim=-1).reshape(
            *slot_data.shape[:-1], head_dim
        )
    else:
        raise ValueError(f"Unsupported mse_bits: {mse_bits}")

    indices = indices.clamp(0, n_centroids - 1)

    # Gather centroids
    key = centroids[indices]

    # Norm correction: re-normalize centroid vector to unit norm
    if norm_correction:
        c_norm_sq = (key * key).sum(dim=-1, keepdim=True)
        c_inv_norm = 1.0 / torch.sqrt(c_norm_sq + 1e-16)
        key = key * c_inv_norm

    # Load and apply vec_norm (fp16 at MSE_BYTES offset, little-endian 2 bytes)
    # Use view instead of << to avoid NPU __lshift__ segfault during graph replay
    norm_bytes = slot_data[..., mse_bytes:mse_bytes + 2].contiguous()
    vec_norm = norm_bytes.view(torch.uint16).view(torch.float16).to(torch.float32)
    vec_norm = vec_norm.unsqueeze(-1)  # broadcast over head_dim

    key = vec_norm * key

    return key.to(torch.float16)


def _unpack_fp8_key(
    slot_data: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Unpack FP8 key from slot data.

    Args:
        slot_data: (..., slot_size) uint8 slot data.
        head_dim: attention head dimension.

    Returns:
        key: (..., head_dim) float16 reconstructed key.
    """
    k_bytes = slot_data[..., :head_dim]
    k_fp8 = k_bytes.view(torch.float8_e4m3fn)
    return k_fp8.to(torch.float16)


def _unpack_value(
    slot_data: torch.Tensor,
    key_packed_size: int,
    value_quant_bits: int,
    val_data_bytes: int,
    head_dim: int,
) -> torch.Tensor:
    """Unpack quantized value from slot data.

    Args:
        slot_data: (..., slot_size) uint8 slot data.
        key_packed_size: offset to value data.
        value_quant_bits: 3 or 4 bits.
        val_data_bytes: bytes for packed value data.
        head_dim: attention head dimension.

    Returns:
        value: (..., head_dim) float16 reconstructed value.
    """
    val_raw = slot_data[..., key_packed_size : key_packed_size + val_data_bytes].to(
        torch.int32
    )

    if value_quant_bits == 4:
        idx_lo = val_raw & 0xF
        idx_hi = (val_raw >> 4) & 0xF
        v_indices = torch.stack([idx_lo, idx_hi], dim=-1).reshape(
            *val_raw.shape[:-1], head_dim
        )
    elif value_quant_bits == 3:
        n_groups = head_dim // 8
        raw = slot_data[..., key_packed_size : key_packed_size + n_groups * 3].to(
            torch.int32
        )
        raw_16 = torch.zeros(*raw.shape[:-1], n_groups, 4, dtype=torch.int32, device=raw.device)
        for g in range(n_groups):
            raw_16[..., g, 0] = raw[..., g * 3]
            raw_16[..., g, 1] = raw[..., g * 3 + 1]
            raw_16[..., g, 2] = raw[..., g * 3 + 2]

        indices_list = []
        for b in range(8):
            byte_idx = (b * 3) // 8
            bit_off = (b * 3) % 8
            if bit_off <= 5:
                idx = (raw_16[..., byte_idx] >> bit_off) & 0x7
            else:
                idx = ((raw_16[..., byte_idx] >> bit_off) |
                       (raw_16[..., byte_idx + 1] * (1 << (8 - bit_off)))) & 0x7
            indices_list.append(idx)
        v_indices = torch.stack(indices_list, dim=-1).reshape(
            *slot_data.shape[:-1], head_dim
        )
    else:
        raise ValueError(f"Unsupported value_quant_bits: {value_quant_bits}")

    v_indices = v_indices.to(torch.float32)

    # Load scale and zero (fp16, 4 bytes total at val_data_bytes offset)
    # Use view instead of << to avoid NPU __lshift__ segfault during graph replay
    sc_base = key_packed_size + val_data_bytes
    sc_bytes = slot_data[..., sc_base:sc_base + 2].contiguous()
    v_scale = sc_bytes.view(torch.uint16).view(torch.float16).to(torch.float32)

    zr_bytes = slot_data[..., sc_base + 2:sc_base + 4].contiguous()
    v_zero = zr_bytes.view(torch.uint16).view(torch.float16).to(torch.float32)

    # Dequantize: value = index * scale + zero
    value = v_indices * v_scale.unsqueeze(-1) + v_zero.unsqueeze(-1)
    return value.to(torch.float16)


def _gather_and_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    key_fp8: bool,
    mse_bits: int,
    mse_bytes: int,
    key_packed_size: int,
    value_quant_bits: int,
    val_data_bytes: int,
    centroids: torch.Tensor,
    norm_correction: bool,
    Pi: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV blocks and dequantize to FP16.

    Args:
        kv_cache: (num_blocks, block_size, Hk, slot_size) uint8 TQ cache.
        block_table: (B, max_num_blocks) int32.
        seq_lens: list of sequence lengths.
        num_kv_heads: number of KV heads.
        head_dim: attention head dimension.
        (remaining args: TQ config parameters)

    Returns:
        key: (total_tokens, Hk, D) float16 dequantized keys.
        value: (total_tokens, Hk, D) float16 dequantized values.
    """
    block_size = kv_cache.shape[1]
    B = len(seq_lens)
    max_seq_len = max(seq_lens) if seq_lens else 0
    max_blocks = block_table.shape[1]

    all_keys = []
    all_values = []

    for b in range(B):
        seq_len = seq_lens[b]
        if seq_len <= 0:
            continue

        # Gather blocks for this sequence
        block_ids = block_table[b, : (seq_len + block_size - 1) // block_size]
        gathered = kv_cache[block_ids]  # (n_blocks, block_size, Hk, slot_size)
        # Flatten to (seq_len_padded, Hk, slot_size)
        gathered = gathered.reshape(-1, num_kv_heads, kv_cache.shape[3])
        gathered = gathered[:seq_len]  # (seq_len, Hk, slot_size)

        # Dequantize key per head
        k_list = []
        v_list = []
        for h in range(num_kv_heads):
            slot_h = gathered[:, h, :]  # (seq_len, slot_size)

            if key_fp8:
                k = _unpack_fp8_key(slot_h, head_dim)
            else:
                k = _unpack_mse_key(
                    slot_h, centroids, mse_bits, mse_bytes, head_dim, norm_correction
                )

            # Inverse Hadamard rotation for MSE keys: k_orig = k_recon @ Pi
            if not key_fp8 and Pi is not None:
                k = (k.float() @ Pi).to(torch.float16)

            v = _unpack_value(slot_h, key_packed_size, value_quant_bits, val_data_bytes, head_dim)

            k_list.append(k)
            v_list.append(v)

        # Stack heads: (seq_len, Hk, D)
        keys_b = torch.stack(k_list, dim=1)
        values_b = torch.stack(v_list, dim=1)
        all_keys.append(keys_b)
        all_values.append(values_b)

    if not all_keys:
        empty = torch.zeros(0, num_kv_heads, head_dim, dtype=torch.float16, device=kv_cache.device)
        return empty, empty

    key = torch.cat(all_keys, dim=0)
    value = torch.cat(all_values, dim=0)
    return key, value


def npu_turboquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    scale: float,
    attn_metadata: object,
    layer: object,
    key_fp8: bool,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    head_dim: int,
    num_kv_heads: int,
    num_heads: int,
    centroids: torch.Tensor,
    norm_correction: bool,
    Pi: torch.Tensor | None,
) -> torch.Tensor:
    """Decode attention for TurboQuant on NPU.

    Strategy: dequantize KV from compressed cache into a temporary
    paged FP16 cache, then call npu_fused_infer_attention_score with
    block_table (paged attention).  This matches the pattern used by
    the standard AscendAttention backend for decode.

    NPU FIA non-paged mode does NOT support different Q/KV lengths,
    which is required for decode (1 query token vs many KV tokens).
    Paged attention with block_table is the only working decode path.

    Returns:
        output: (num_decode_tokens, Hq, D) float16.
    """
    B = block_table.shape[0]
    D = head_dim
    block_size = kv_cache.shape[1]
    device = query.device

    mse_bytes = math.ceil(D * mse_bits / 8) if not key_fp8 else 0
    val_data_bytes = math.ceil(D * value_quant_bits / 8)

    # Collect unique block IDs referenced by active sequences
    active_ids = []
    for b in range(B):
        sl = seq_lens[b]
        if sl <= 0:
            continue
        n_blocks = (sl + block_size - 1) // block_size
        active_ids.append(block_table[b, :n_blocks])

    if not active_ids:
        return query.new_zeros(query.shape[0], num_heads, D)

    used_blocks = torch.unique(torch.cat(active_ids))
    n_used = used_blocks.shape[0]

    # Build reverse mapping: original block_id -> index in partial cache
    max_id = used_blocks.max().item()
    id_to_idx = torch.full((max_id + 1,), -1, dtype=torch.int32, device=device)
    for i in range(n_used):
        id_to_idx[used_blocks[i].item()] = i

    # Dequantize used blocks into paged FP16 format:
    #   (n_used, block_size, num_kv_heads * head_dim) — flattened heads,
    #   same layout as standard AscendAttention key_cache.
    key_cache_fp16 = query.new_empty(n_used, block_size, num_kv_heads * D)
    value_cache_fp16 = query.new_empty(n_used, block_size, num_kv_heads * D)

    for i in range(n_used):
        bid = used_blocks[i].item()
        block_data = kv_cache[bid]  # (block_size, Hk, slot_size)

        for h in range(num_kv_heads):
            slot_h = block_data[:, h, :]  # (block_size, slot_size)

            if key_fp8:
                k = _unpack_fp8_key(slot_h, D)
            else:
                k = _unpack_mse_key(
                    slot_h, centroids, mse_bits, mse_bytes, D, norm_correction
                )
                if Pi is not None:
                    k = (k.float() @ Pi).to(torch.float16)

            v = _unpack_value(
                slot_h, key_packed_size, value_quant_bits, val_data_bytes, D
            )

            key_cache_fp16[i, :, h * D : (h + 1) * D] = k
            value_cache_fp16[i, :, h * D : (h + 1) * D] = v

    # Remap block_table: original block IDs -> partial cache indices
    bt_clamped = block_table.clamp(min=0, max=max_id)
    remapped_bt = id_to_idx[bt_clamped].clamp(min=0).to(block_table.dtype)

    # Paged attention via FIA — same pattern as standard AscendAttention
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query[:B],
        key_cache_fp16,
        value_cache_fp16,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        input_layout="TND",
        block_table=remapped_bt,
        block_size=block_size,
        sparse_mode=0,
        actual_seq_lengths=[1] * B,
        actual_seq_lengths_kv=seq_lens,
    )

    return output


def npu_turboquant_full_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    key_fp8: bool,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    val_data_bytes: int,
    centroids: torch.Tensor,
    norm_correction: bool,
    Pi: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full dequantization of all cached KV for continuation prefill.

    Returns per-request batched K, V tensors for use with flash attention.

    Returns:
        key: (total_tokens, Hk, D) float16.
        value: (total_tokens, Hk, D) float16.
    """
    mse_bytes = math.ceil(head_dim * mse_bits / 8) if not key_fp8 else 0

    return _gather_and_dequant_kv(
        kv_cache, block_table, seq_lens,
        num_kv_heads, head_dim,
        key_fp8, mse_bits, mse_bytes,
        key_packed_size, value_quant_bits, val_data_bytes,
        centroids, norm_correction, Pi,
    )
