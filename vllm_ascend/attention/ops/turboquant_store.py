# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU (PyTorch) implementation of TurboQuant KV store.

Replaces the Triton kernels from vllm with pure PyTorch operations
that run on Ascend NPU via torch_npu.

Two paths:
1. FP8 key mode: key cast to FP8 + value uniform quantization.
2. MSE key mode: normalize + Hadamard rotate + Lloyd-Max bucketize + pack.

Both paths write into the combined TQ cache slot layout:
  [key_packed | value_packed]
"""

import math

import torch


def _quantize_values_batched(
    values: torch.Tensor,
    value_quant_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized uniform affine quantization for a batch of value vectors.

    Args:
        values: (NH, D) float32 value vectors.
        value_quant_bits: 3 or 4 bits.

    Returns:
        packed: (NH, packed_bytes) uint8 packed quantized values.
        scales: (NH,) fp16 scale factors.
        zeros: (NH,) fp16 zero-points.
    """
    NH, D = values.shape
    n_levels = (1 << value_quant_bits) - 1  # 7 or 15

    val_min = values.min(dim=-1).values  # (NH,)
    val_max = values.max(dim=-1).values  # (NH,)
    v_scale = (val_max - val_min) / n_levels
    v_scale = v_scale.clamp(min=1e-8)

    # (NH, D)
    q_vals = ((values - val_min.unsqueeze(-1)) / v_scale.unsqueeze(-1) + 0.5).int().clamp(0, n_levels)

    if value_quant_bits == 4:
        q_pairs = q_vals.view(NH, -1, 2)
        packed = ((q_pairs[:, :, 0] & 0xF) | ((q_pairs[:, :, 1] & 0xF) * 16)).to(torch.uint8)
        packed = packed.reshape(NH, -1)
    elif value_quant_bits == 3:
        n_groups = D // 8
        q_groups = q_vals.view(NH, n_groups, 8)
        shifts = torch.arange(8, dtype=torch.int32, device=values.device) * 3
        # Use power-of-2 multiplication instead of << for NPU graph replay compat
        p2 = 2 ** shifts  # [1, 8, 64, 512, ...]
        packed_24 = (q_groups * p2.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        b0 = (packed_24 % 256).to(torch.uint8)
        b1 = ((packed_24 // 256) % 256).to(torch.uint8)
        b2 = ((packed_24 // 65536) % 256).to(torch.uint8)
        packed = torch.stack([b0, b1, b2], dim=-1).reshape(NH, -1)
    else:
        raise ValueError(f"Unsupported value_quant_bits: {value_quant_bits}")

    return packed, v_scale.to(torch.float16), val_min.to(torch.float16)


def _fp16_to_bytes_batched(x: torch.Tensor) -> torch.Tensor:
    """Convert (NH,) fp16 tensor to (NH, 2) uint8 bytes (little-endian).

    Uses bitcast to uint16, then extracts low/high bytes.
    NPU requires int32 for bitwise ops (uint16 & scalar fails).
    """
    u16 = x.view(torch.uint16).to(torch.int32)  # int32 for NPU bitwise compat
    lo = (u16 % 256).to(torch.uint8)
    hi = (u16 // 256).to(torch.uint8)
    return torch.stack([lo, hi], dim=-1)  # (NH, 2)


def _scatter_to_cache(
    slot_data: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    N: int,
    H: int,
) -> None:
    """Scatter packed slot data into paged KV cache.

    Args:
        slot_data: (N*H, slot_size) uint8 data to write.
        kv_cache: (num_blocks, block_size, Hk, slot_size) uint8 cache.
        slot_mapping: (N,) int32 slot indices per token.
        N: number of tokens.
        H: number of KV heads.
    """
    block_size = kv_cache.shape[1]

    slot_mapping_expanded = slot_mapping.unsqueeze(1).expand(N, H).reshape(N * H)
    head_indices = torch.arange(H, device=slot_data.device).unsqueeze(0).expand(N, H).reshape(N * H)

    valid_mask = slot_mapping_expanded >= 0
    valid_slots = slot_mapping_expanded[valid_mask]
    valid_heads = head_indices[valid_mask]
    valid_data = slot_data[valid_mask]

    block_indices = valid_slots // block_size
    pos_indices = valid_slots % block_size

    kv_cache[block_indices, pos_indices, valid_heads] = valid_data


def _store_fp8_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_packed_size: int,
    value_quant_bits: int,
    val_data_bytes: int,
) -> None:
    """FP8 key cast + value uniform quantization + scatter to cache."""
    N, H, D = key.shape
    NH = N * H

    k_flat = key.reshape(NH, D)
    v_flat = value.float().reshape(NH, D)

    # FP8 key cast
    k_fp8 = k_flat.to(torch.float8_e4m3fn)
    k_bytes = k_fp8.view(torch.uint8)  # (NH, D)

    # Vectorized value quantization
    val_packed, scales, zeros = _quantize_values_batched(v_flat, value_quant_bits)

    # Pack scale + zero as fp16 bytes: (NH, 2) each
    scale_bytes = _fp16_to_bytes_batched(scales)  # (NH, 2)
    zero_bytes = _fp16_to_bytes_batched(zeros)    # (NH, 2)
    metadata = torch.cat([scale_bytes, zero_bytes], dim=-1)  # (NH, 4)

    # Combine: [key_bytes | val_packed | scale_bytes | zero_bytes]
    slot_data = torch.cat([k_bytes, val_packed, metadata], dim=1)  # (NH, slot_size)

    _scatter_to_cache(slot_data, kv_cache, slot_mapping, N, H)


def _store_mse_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    PiT: torch.Tensor,
    midpoints: torch.Tensor,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    val_data_bytes: int,
    norm_correction: bool = False,
) -> None:
    """MSE key quantization (normalize + rotate + bucketize + pack) + value quantization."""
    N, H, D = key.shape
    NH = N * H
    n_centroids = 1 << mse_bits
    mse_bytes = math.ceil(D * mse_bits / 8)

    # Normalize + rotate (GEMM on NPU)
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)  # (NH, 1)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ PiT  # (NH, D)

    # Binary search bucketize using searchsorted
    idx = torch.searchsorted(midpoints, y, right=True)  # (NH, D)
    idx = idx.clamp(0, n_centroids - 1)

    # Pack MSE indices
    if mse_bits == 4:
        idx_pairs = idx.view(NH, -1, 2)
        packed_mse = ((idx_pairs[:, :, 0] & 0xF) |
                      ((idx_pairs[:, :, 1] & 0xF) << 4)).to(torch.uint8)
        packed_mse = packed_mse.reshape(NH, mse_bytes)
    elif mse_bits == 3:
        n_groups = D // 8
        idx_groups = idx.view(NH, n_groups, 8)
        shifts = torch.arange(8, dtype=torch.int32, device=key.device) * 3
        p2 = 2 ** shifts
        packed_24 = (idx_groups * p2.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        b0 = (packed_24 % 256).to(torch.uint8)
        b1 = ((packed_24 // 256) % 256).to(torch.uint8)
        b2 = ((packed_24 // 65536) % 256).to(torch.uint8)
        packed_mse = torch.stack([b0, b1, b2], dim=-1).reshape(NH, mse_bytes)
    else:
        raise ValueError(f"Unsupported mse_bits: {mse_bits}")

    # Norms as fp16 bytes (2 bytes per token-head)
    norms_fp16 = norms.squeeze(1).to(torch.float16)  # (NH,)
    norm_bytes = _fp16_to_bytes_batched(norms_fp16)  # (NH, 2)

    # Vectorized value quantization
    v_flat = value.float().reshape(NH, D)
    val_packed, scales, zeros = _quantize_values_batched(v_flat, value_quant_bits)

    # Pack scale + zero as fp16 bytes
    scale_bytes = _fp16_to_bytes_batched(scales)  # (NH, 2)
    zero_bytes = _fp16_to_bytes_batched(zeros)    # (NH, 2)
    metadata = torch.cat([scale_bytes, zero_bytes], dim=-1)  # (NH, 4)

    # Combine: [mse_indices | norm_bytes | val_packed | scale_bytes | zero_bytes]
    slot_data = torch.cat([packed_mse, norm_bytes, val_packed, metadata], dim=1)  # (NH, slot_size)

    _scatter_to_cache(slot_data, kv_cache, slot_mapping, N, H)


def npu_turboquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    PiT: torch.Tensor | None,
    midpoints: torch.Tensor | None,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
) -> None:
    """Main entry point: quantize K/V and store to TQ cache on NPU.

    Args:
        key: (N, Hk, D) raw keys.
        value: (N, Hk, D) raw values.
        kv_cache: (num_blocks, block_size, Hk, slot_size) uint8.
        slot_mapping: (N,) int32 slot indices.
        PiT: (D, D) Hadamard transpose (None for FP8 mode).
        midpoints: (n_centroids-1,) Lloyd-Max midpoints (None for FP8 mode).
        mse_bits: MSE quantization bits (0 for FP8).
        key_packed_size: packed key size in bytes.
        value_quant_bits: value quantization bits (3 or 4).
        key_fp8: whether keys use FP8 mode.
        norm_correction: whether to apply norm correction.
    """
    N = slot_mapping.shape[0]
    if N <= 0:
        return

    val_data_bytes = math.ceil(key.shape[-1] * value_quant_bits / 8)

    if key_fp8:
        _store_fp8_key_value(
            key, value, kv_cache, slot_mapping,
            key_packed_size, value_quant_bits, val_data_bytes,
        )
    else:
        assert PiT is not None and midpoints is not None, \
            "PiT and midpoints required for MSE mode"
        _store_mse_key_value(
            key, value, kv_cache, slot_mapping, PiT, midpoints,
            mse_bits, key_packed_size, value_quant_bits, val_data_bytes,
            norm_correction,
        )
