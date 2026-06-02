# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU (PyTorch) implementation of TurboQuant decode attention.

Replaces the Triton decode kernels with pure PyTorch operations
that run on Ascend NPU via torch_npu.

Strategy: dequantize KV from TQ cache to FP16,
then use npu_fused_infer_attention_score for standard attention.

All per-block / per-head Python loops have been replaced with
vectorized tensor operations for performance on NPU.
"""

import ctypes
import math
import os

import torch
import torch_npu

# Try to load the fused Ascend C decode operator
_tq_fused_decode_loaded = False
try:
    _build_dir = os.path.join(os.path.dirname(__file__),
                               "tq_fused_decode", "build")
    _ops_path = os.path.join(_build_dir, "libtq_fused_decode_ops.so")
    _kern_path = os.path.join(_build_dir, "lib",
                               "libtq_fused_decode_kernels.so")
    if os.path.exists(_ops_path):
        # Preload kernel .so into global scope so ops .so can resolve it
        if os.path.exists(_kern_path):
            ctypes.CDLL(_kern_path, mode=ctypes.RTLD_GLOBAL)
            print(f"[TQ] Preloaded kernel: {_kern_path}")
        else:
            print(f"[TQ] WARNING: kernel .so not found at {_kern_path}")

        # Try ctypes first for better error messages
        try:
            ctypes.CDLL(_ops_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            print(f"[TQ] ctypes load error: {e}")
            raise

        # Load via torch for op registration
        torch.ops.load_library(_ops_path)
        _tq_fused_decode_loaded = True
        print(f"[TQ] Fused decode kernel loaded: {_ops_path}")
    else:
        print(f"[TQ] Fused decode kernel NOT found at: {_ops_path}")
except Exception as e:
    print(f"[TQ] Failed to load fused decode kernel: {e}")


# ============================================================================
# Diagnostic configuration (env-controlled)
# ============================================================================
# TQ_DIAG_DUAL_PATH=1 (default): enable the diagnostic branch when the fused
#   kernel is eligible. Within this branch, TQ_DIAG_SAMPLE controls frequency.
#   Set to 0 to bypass entirely and use fused-only (the goal state once the
#   kernel is verified correct).
# TQ_DIAG_SAMPLE=N: only run dual-path on every Nth call (1-indexed).
#   0 (default) = run dual-path EVERY call (safe+slow; always returns fallback).
#   N>0          = run dual-path every Nth call; other calls return fused
#                  directly. With N=10 you get ~10% diagnostic overhead and
#                  ~90% fast fused-only calls. Recommended AFTER the kernel
#                  fix is verified; if logs show consistent "ok" you can
#                  graduate to TQ_DIAG_DUAL_PATH=0.
# TQ_DIAG_LOG_LIMIT=N: only log first N DIAGNOSTIC calls per process (default
#   30). Diagnostic calls past this limit still run both paths (and still
#   return fallback for safety on the diagnostic call) but stop printing.
# TQ_DIAG_DIFF_THRESHOLD: log "GARBLED" if max_diff > threshold (default 1e-2).
_TQ_DIAG_DUAL_PATH = os.environ.get("TQ_DIAG_DUAL_PATH", "1") not in ("0", "false", "False")
_TQ_DIAG_SAMPLE = int(os.environ.get("TQ_DIAG_SAMPLE", "0"))
_TQ_DIAG_LOG_LIMIT = int(os.environ.get("TQ_DIAG_LOG_LIMIT", "30"))
_TQ_DIAG_DIFF_THRESHOLD = float(os.environ.get("TQ_DIAG_DIFF_THRESHOLD", "1e-2"))

# Per-process diagnostic state (initialized lazily inside the call path)
_TQ_DIAG_STATE = {
    "total_count": 0,    # all fused-eligible calls (diagnostic + fast)
    "diag_count": 0,     # calls that actually ran dual-path
    "garbled_count": 0,
    "max_diff_seen": 0.0,
}


def _tq_diag_log(*args, **kwargs):
    """Always-flushed print so logs survive a server crash."""
    print(*args, **kwargs, flush=True)


def _unpack_mse_key(
    slot_data: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    mse_bytes: int,
    head_dim: int,
    norm_correction: bool = False,
    key_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack MSE key indices and reconstruct key vector.

    Args:
        slot_data: (..., slot_size) uint8 slot data.
        centroids: (n_centroids,) float32 centroids.
        mse_bits: 3 or 4 bits.
        mse_bytes: bytes for packed MSE indices.
        head_dim: attention head dimension.
        norm_correction: whether to normalize centroid vectors.
        key_lut: (256, 2) float32 precomputed LUT for 4-bit mode.
            Maps byte → [centroid[lo], centroid[hi]], avoiding 5+ kernel
            launches for bit-unpack + centroid gather.

    Returns:
        key: (..., head_dim) float16 reconstructed key.
    """
    n_centroids = 1 << mse_bits

    # Fast path: 4-bit LUT (2 kernels instead of 7 for bit-unpack + gather)
    if mse_bits == 4 and key_lut is not None:
        byte_data = slot_data[..., :mse_bytes].long()
        key_pairs = key_lut[byte_data]                      # (..., 64, 2) f32
        key = key_pairs.reshape(*slot_data.shape[:-1], head_dim)  # (..., 128) f32
    elif mse_bits == 4:
        # Fallback without LUT
        mse_raw = slot_data[..., :mse_bytes].to(torch.int32)
        idx_lo = mse_raw & 0xF
        idx_hi = (mse_raw >> 4) & 0xF
        indices = torch.stack([idx_lo, idx_hi], dim=-1).reshape(
            *mse_raw.shape[:-1], head_dim
        )
        indices = indices.clamp(0, n_centroids - 1)
        key = centroids[indices]
    elif mse_bits == 3:
        # Eight 3-bit indices per 3 bytes
        n_groups = head_dim // 8
        raw = slot_data[..., : n_groups * 3].to(torch.int32)
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
        indices = torch.stack(indices_list, dim=-1).reshape(
            *slot_data.shape[:-1], head_dim
        )
        indices = indices.clamp(0, n_centroids - 1)
        key = centroids[indices]
    else:
        raise ValueError(f"Unsupported mse_bits: {mse_bits}")

    # Norm correction: re-normalize centroid vector to unit norm
    if norm_correction:
        c_norm_sq = (key * key).sum(dim=-1, keepdim=True)
        c_inv_norm = 1.0 / torch.sqrt(c_norm_sq + 1e-16)
        key = key * c_inv_norm

    # Load and apply vec_norm (fp16 at MSE_BYTES offset, little-endian 2 bytes)
    norm_bytes = slot_data[..., mse_bytes:mse_bytes + 2].contiguous()
    vec_norm = norm_bytes.view(torch.uint16).view(torch.float16).to(torch.float32)

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
    val_idx_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack quantized value from slot data.

    Args:
        slot_data: (..., slot_size) uint8 slot data.
        key_packed_size: offset to value data.
        value_quant_bits: 3 or 4 bits.
        val_data_bytes: bytes for packed value data.
        head_dim: attention head dimension.
        val_idx_lut: (256, 2) float32 precomputed LUT for 4-bit mode.
            Maps byte → [float(lo), float(hi)], avoiding 5+ kernel launches.

    Returns:
        value: (..., head_dim) float16 reconstructed value.
    """
    # Fast path: 4-bit LUT (2 kernels instead of 7)
    if value_quant_bits == 4 and val_idx_lut is not None:
        byte_data = slot_data[..., key_packed_size : key_packed_size + val_data_bytes].long()
        v_pairs = val_idx_lut[byte_data]                           # (..., 64, 2) f32
        v_indices = v_pairs.reshape(*slot_data.shape[:-1], head_dim)  # (..., 128) f32
    elif value_quant_bits == 4:
        val_raw = slot_data[..., key_packed_size : key_packed_size + val_data_bytes].to(
            torch.int32
        )
        idx_lo = val_raw & 0xF
        idx_hi = (val_raw >> 4) & 0xF
        v_indices = torch.stack([idx_lo, idx_hi], dim=-1).reshape(
            *val_raw.shape[:-1], head_dim
        )
        v_indices = v_indices.to(torch.float32)
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
        v_indices = v_indices.to(torch.float32)
    else:
        raise ValueError(f"Unsupported value_quant_bits: {value_quant_bits}")

    # Load scale and zero (fp16, 4 bytes total at val_data_bytes offset)
    sc_base = key_packed_size + val_data_bytes
    sc_bytes = slot_data[..., sc_base:sc_base + 2].contiguous()
    v_scale = sc_bytes.view(torch.uint16).view(torch.float16).to(torch.float32)

    zr_bytes = slot_data[..., sc_base + 2:sc_base + 4].contiguous()
    v_zero = zr_bytes.view(torch.uint16).view(torch.float16).to(torch.float32)

    # Dequantize: value = index * scale + zero
    value = v_indices * v_scale + v_zero
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
    target_dtype: torch.dtype = torch.float16,
    key_lut: torch.Tensor | None = None,
    val_idx_lut: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV blocks and dequantize (vectorized).

    Builds flat indices for all valid tokens across sequences, gathers
    from the flattened cache in a single operation, then batch-dequantizes
    all keys and values simultaneously.

    Keys are returned in Hadamard-rotated space (matching the compressed
    cache format). The caller is responsible for rotating Q (and any
    current-batch K) into the same space before calling attention.

    Returns:
        key: (total_tokens, Hk, D) dequantized keys (Hadamard space).
        value: (total_tokens, Hk, D) dequantized values.
    """
    block_size = kv_cache.shape[1]
    slot_size = kv_cache.shape[3]
    device = kv_cache.device
    B = len(seq_lens)

    # Build flat indices for all valid tokens, preserving sequence order.
    # Each index maps to (block_id * block_size + pos_in_block) in the
    # flattened cache view.
    index_parts = []
    for b in range(B):
        s = seq_lens[b]
        if s <= 0:
            continue
        n_blocks = (s + block_size - 1) // block_size
        bids = block_table[b, :n_blocks]
        positions = torch.arange(s, device=device)
        flat_idx = bids[positions // block_size] * block_size + positions % block_size
        index_parts.append(flat_idx)

    if not index_parts:
        empty = torch.zeros(0, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        return empty, empty

    all_indices = torch.cat(index_parts)  # (total_tokens,)

    # Gather from flattened cache: (total_tokens, Hk, slot_size)
    kv_flat = kv_cache.reshape(-1, num_kv_heads, slot_size)
    valid_data = kv_flat[all_indices]

    # Flatten to (total_tokens * Hk, slot_size) for batch unpacking
    flat_data = valid_data.reshape(-1, slot_size)

    # Vectorized key dequantization
    if key_fp8:
        all_keys = _unpack_fp8_key(flat_data, head_dim)
    else:
        all_keys = _unpack_mse_key(
            flat_data, centroids, mse_bits, mse_bytes, head_dim, norm_correction,
            key_lut=key_lut,
        )

    # Vectorized value dequantization
    all_values = _unpack_value(
        flat_data, key_packed_size, value_quant_bits, val_data_bytes, head_dim,
        val_idx_lut=val_idx_lut,
    )

    # Reshape to (total_tokens, Hk, D) and cast to target dtype
    key = all_keys.reshape(-1, num_kv_heads, head_dim).to(target_dtype)
    value = all_values.reshape(-1, num_kv_heads, head_dim).to(target_dtype)
    return key, value


def _decode_fallback_impl(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    scale: float,
    key_fp8: bool,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    head_dim: int,
    num_kv_heads: int,
    num_heads: int,
    centroids: torch.Tensor,
    norm_correction: bool,
    target_dtype: torch.dtype,
    key_lut: torch.Tensor | None,
    val_idx_lut: torch.Tensor | None,
) -> torch.Tensor:
    """Python dequant + npu_fused_infer_attention_score decode path.

    Extracted as a standalone helper so the dual-path diagnostic can call it
    alongside the fused kernel without duplicating logic.
    """
    B = block_table.shape[0]
    D = head_dim
    block_size = kv_cache.shape[1]
    slot_size = kv_cache.shape[3]
    device = query.device

    mse_bytes = math.ceil(D * mse_bits / 8) if not key_fp8 else 0
    val_data_bytes = math.ceil(D * value_quant_bits / 8)

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

    all_block_data = kv_cache[used_blocks]
    flat_data = all_block_data.reshape(-1, slot_size)

    if key_fp8:
        all_keys = _unpack_fp8_key(flat_data, D)
    else:
        all_keys = _unpack_mse_key(
            flat_data, centroids, mse_bits, mse_bytes, D, norm_correction,
            key_lut=key_lut,
        )

    all_values = _unpack_value(
        flat_data, key_packed_size, value_quant_bits, val_data_bytes, D,
        val_idx_lut=val_idx_lut,
    )

    key_cache = all_keys.reshape(n_used, block_size, num_kv_heads * D).to(target_dtype)
    value_cache = all_values.reshape(n_used, block_size, num_kv_heads * D).to(target_dtype)

    max_id = used_blocks.max().item()
    id_to_idx = torch.full((max_id + 1,), -1, dtype=torch.int32, device=device)
    id_to_idx[used_blocks] = torch.arange(n_used, dtype=torch.int32, device=device)

    bt_clamped = block_table.clamp(min=0, max=max_id)
    remapped_bt = id_to_idx[bt_clamped].clamp(min=0).to(block_table.dtype)

    # Match standard AscendAttention decode convention:
    # Q seq_lens are cumulative, KV seq_lens are individual.
    cum_seq_lens_q = list(range(1, B + 1))

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query[:B],
        key_cache,
        value_cache,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        input_layout="TND",
        block_table=remapped_bt,
        block_size=block_size,
        sparse_mode=0,
        actual_seq_lengths=cum_seq_lens_q,
        actual_seq_lengths_kv=seq_lens,
    )
    return output


def _call_fused_kernel(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    head_dim: int,
    block_size: int,
    centroids: torch.Tensor,
    norm_correction: bool,
    target_dtype: torch.dtype,
):
    """Invoke the fused Ascend C decode kernel and cast to target_dtype."""
    device = query.device
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    # Kernel expects fp16 query; convert from bf16/fp32 if needed.
    B = block_table.shape[0]
    q_in = query[:B].to(torch.float16)
    out = torch.ops.npu.tq_fused_decode(
        q_in,
        kv_cache,
        block_table,
        seq_lens_t,
        centroids.to(torch.float32),
        sm_scale=scale,
        mse_bytes=math.ceil(head_dim * mse_bits / 8),
        key_packed_size=key_packed_size,
        val_data_bytes=math.ceil(head_dim * value_quant_bits / 8),
        head_dim=head_dim,
        block_size=block_size,
        norm_correction=norm_correction,
    )
    return out.to(target_dtype)


def _log_decode_diff(
    out_fused: torch.Tensor,
    out_fb: torch.Tensor,
    seq_lens: list[int],
    head_dim: int,
    num_heads: int,
) -> None:
    """Compare fused vs fallback outputs and log a single summary line.

    Logs are rate-limited by _TQ_DIAG_LOG_LIMIT (counted against diagnostic
    calls only, not total calls). After the limit is reached, this function
    becomes a no-op (the caller still returns the fallback tensor, so
    correctness is preserved — we just stop spamming logs).
    """
    state = _TQ_DIAG_STATE
    state["diag_count"] += 1
    n = state["diag_count"]
    if n > _TQ_DIAG_LOG_LIMIT:
        return

    # Float diff for stable comparison regardless of target dtype.
    diff = (out_fused.float() - out_fb.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    has_nan = bool(torch.isnan(out_fused).any().item() or torch.isinf(out_fused).any().item())
    has_garbled = max_diff > _TQ_DIAG_DIFF_THRESHOLD or has_nan

    if max_diff > state["max_diff_seen"]:
        state["max_diff_seen"] = max_diff
    if has_garbled:
        state["garbled_count"] += 1

    # Locate the worst element: (batch, head, dim)
    flat_idx = int(diff.argmax().item())
    total_per_batch = num_heads * head_dim
    b_bad = flat_idx // total_per_batch
    h_bad = (flat_idx % total_per_batch) // head_dim
    d_bad = (flat_idx % total_per_batch) % head_dim

    seq_lens_str = ",".join(str(x) for x in seq_lens[:min(8, len(seq_lens))])
    if len(seq_lens) > 8:
        seq_lens_str += f",...(+{len(seq_lens) - 8})"

    tag = "GARBLED" if has_garbled else "ok"
    _tq_diag_log(
        f"[TQ-DIAG #{n}] {tag} max_diff={max_diff:.4e} mean={mean_diff:.4e} "
        f"B={out_fused.shape[0]} seq=[{seq_lens_str}] "
        f"worst@(b={b_bad},h={h_bad},d={d_bad}) "
        f"nan/inf={'yes' if has_nan else 'no'} "
        f"running(garbled={state['garbled_count']}/{n}, peak={state['max_diff_seen']:.4e})"
    )

    if has_garbled and n <= 5:
        # Show the first head's first 8 dims so we can see bit-pattern corruption.
        fused_sample = out_fused[0, 0, :8].detach().cpu().float().tolist()
        fb_sample = out_fb[0, 0, :8].detach().cpu().float().tolist()
        _tq_diag_log(f"  fused[0,0,:8]    = {[f'{v:+.3e}' for v in fused_sample]}")
        _tq_diag_log(f"  fallback[0,0,:8] = {[f'{v:+.3e}' for v in fb_sample]}")
        # Full 128-dim dump of head 0 to expose the exact repeating pattern.
        fused_full = out_fused[0, 0, :].detach().cpu().float()
        unique_vals = torch.unique(fused_full)
        _tq_diag_log(
            f"  fused[0,0,:] unique={unique_vals.tolist()[:16]} "
            f"(count={unique_vals.numel()})"
        )

    # Kernel launch diagnostic: check for the -42.0 sentinel and blockIdx values.
    if n <= 3:
        fused_all = out_fused.detach().cpu().float()
        all_unique = torch.unique(fused_all)
        _tq_diag_log(
            f"  KERNEL-LAUNCH-CHECK: fused all-heads unique="
            f"{all_unique.tolist()[:32]} (count={all_unique.numel()})"
        )


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
    target_dtype: torch.dtype = torch.float16,
    key_lut: torch.Tensor | None = None,
    val_idx_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention for TurboQuant on NPU.

    Three execution modes (controlled by env vars at import time):

    1. Fused kernel unavailable (or wrong dtype): runs Python fallback only.
    2. Fused kernel available, TQ_DIAG_DUAL_PATH=1 (default):
       runs BOTH fused kernel and Python fallback, logs max_diff,
       returns fallback for safety. Use this to diagnose garbled output.
    3. Fused kernel available, TQ_DIAG_DUAL_PATH=0:
       runs fused kernel only (original behavior).
    """
    B = block_table.shape[0]
    D = head_dim
    block_size = kv_cache.shape[1]
    device = query.device

    fused_eligible = _tq_fused_decode_loaded and not key_fp8 and mse_bits == 4

    # One-time reason log when fused kernel is not being used at all.
    if not fused_eligible:
        if not getattr(npu_turboquant_decode_attention, "_diag_printed", False):
            npu_turboquant_decode_attention._diag_printed = True
            reasons = []
            if not _tq_fused_decode_loaded:
                reasons.append("kernel not loaded")
            if key_fp8:
                reasons.append("key_fp8=True")
            if mse_bits != 4:
                reasons.append(f"mse_bits={mse_bits} (!=4)")
            _tq_diag_log(
                f"[TQ] Fallback-only mode. Reasons: {', '.join(reasons)}; "
                f"dual_path_diag={_TQ_DIAG_DUAL_PATH} (inactive)"
            )
        return _decode_fallback_impl(
            query, kv_cache, block_table, seq_lens, scale,
            key_fp8, mse_bits, key_packed_size, value_quant_bits,
            head_dim, num_kv_heads, num_heads, centroids, norm_correction,
            target_dtype, key_lut, val_idx_lut,
        )

    # Fused kernel is eligible. Decide between dual-path diagnostic and fast path.
    if _TQ_DIAG_DUAL_PATH:
        # Sample the dual-path diagnostic to avoid paying the fallback cost
        # on every decode call. TQ_DIAG_SAMPLE=0 → every call (safe+slow).
        # TQ_DIAG_SAMPLE=N>0 → every Nth call diagnostic, others return fused.
        state = _TQ_DIAG_STATE
        state["total_count"] += 1
        do_diag = (
            _TQ_DIAG_SAMPLE <= 0
            or (state["total_count"] % _TQ_DIAG_SAMPLE == 0)
        )

        if do_diag:
            # Run both paths and compare. The fused result is throwaway for
            # correctness (we return fallback), but the diff tells us exactly
            # where and how badly the kernel diverges.
            out_fused = _call_fused_kernel(
                query, kv_cache, block_table, seq_lens, scale,
                mse_bits, key_packed_size, value_quant_bits,
                D, block_size, centroids, norm_correction,
                target_dtype,
            )
            out_fb = _decode_fallback_impl(
                query, kv_cache, block_table, seq_lens, scale,
                key_fp8, mse_bits, key_packed_size, value_quant_bits,
                head_dim, num_kv_heads, num_heads, centroids, norm_correction,
                target_dtype, key_lut, val_idx_lut,
            )
            _log_decode_diff(out_fused, out_fb, seq_lens, head_dim=D, num_heads=num_heads)
            return out_fb

        # Fast lane: skip the fallback and return fused directly. This is
        # the target steady state; sampling just keeps an occasional
        # verification ping in the logs.
        return _call_fused_kernel(
            query, kv_cache, block_table, seq_lens, scale,
            mse_bits, key_packed_size, value_quant_bits,
            D, block_size, centroids, norm_correction,
            target_dtype,
        )

    # Pure fused path (original behavior — used after the kernel is fixed
    # and dual-path diagnostic is no longer needed).
    return _call_fused_kernel(
        query, kv_cache, block_table, seq_lens, scale,
        mse_bits, key_packed_size, value_quant_bits,
        D, block_size, centroids, norm_correction,
        target_dtype,
    )


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
    target_dtype: torch.dtype = torch.float16,
    key_lut: torch.Tensor | None = None,
    val_idx_lut: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full dequantization of all cached KV for continuation prefill.

    Keys are returned in Hadamard-rotated space. The caller must rotate
    Q and current-batch K into the same space before calling attention.

    Returns:
        key: (total_tokens, Hk, D) in target_dtype (Hadamard space).
        value: (total_tokens, Hk, D) in target_dtype.
    """
    mse_bytes = math.ceil(head_dim * mse_bits / 8) if not key_fp8 else 0

    return _gather_and_dequant_kv(
        kv_cache, block_table, seq_lens,
        num_kv_heads, head_dim,
        key_fp8, mse_bits, mse_bytes,
        key_packed_size, value_quant_bits, val_data_bytes,
        centroids, norm_correction, target_dtype,
        key_lut=key_lut, val_idx_lut=val_idx_lut,
    )
