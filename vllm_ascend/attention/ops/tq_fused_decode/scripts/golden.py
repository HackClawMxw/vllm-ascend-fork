"""Python reference implementation for golden comparison.

Reuses the existing _unpack_mse_key and _unpack_value from turboquant_decode.py
to compute the expected attention output.
"""
import math
import numpy as np
import torch

MSE_BITS = 4
D = 128
MSE_BYTES = 64
KEY_PACKED_SIZE = 66
VAL_BITS = 4
VAL_DATA_BYTES = 64
SLOT_SIZE = 134


def fp16_bytes_to_float(data, offset):
    """Read 2 bytes as fp16 little-endian, return float32."""
    raw = int(data[offset]) | (int(data[offset + 1]) << 8)
    h = np.frombuffer(np.uint16(raw).tobytes(), dtype=np.float16)[0]
    return float(h)


def unpack_key(slot_data):
    """Unpack MSE key from a single slot (SLOT_SIZE bytes)."""
    key = np.zeros(D, dtype=np.float32)
    for byte_idx in range(MSE_BYTES):
        packed = slot_data[byte_idx]
        lo_idx = packed & 0xF
        hi_idx = (packed >> 4) & 0xF
        key[byte_idx * 2 + 0] = lo_idx
        key[byte_idx * 2 + 1] = hi_idx
    return key  # raw indices, not yet looked up in centroids


def compute_attention_python(query_rot_np, kv_cache_np, block_table_np,
                             seq_lens_np, centroids_np, sm_scale,
                             norm_correction=True):
    """Compute attention using the same logic as the Ascend C kernel."""
    B, Hq, _ = query_rot_np.shape
    Hk = kv_cache_np.shape[2]
    gqa_ratio = Hq // Hk

    output = np.zeros((B, Hq, D), dtype=np.float32)

    for b in range(B):
        seq_len = int(seq_lens_np[b])
        for qh in range(Hq):
            q = query_rot_np[b, qh].astype(np.float32)
            kv_head = qh // gqa_ratio

            # Online softmax state
            running_max = -3.4e38
            running_sum = 0.0
            acc = np.zeros(D, dtype=np.float32)

            for token_pos in range(seq_len):
                virtual_block = token_pos // 16
                offset_in_block = token_pos % 16
                physical_block = int(block_table_np[b, virtual_block])

                slot = kv_cache_np[physical_block, offset_in_block, kv_head]

                # Compute score
                raw_score = 0.0
                norm_sq = 0.0
                for byte_idx in range(MSE_BYTES):
                    packed = int(slot[byte_idx])
                    lo_idx = packed & 0xF
                    hi_idx = (packed >> 4) & 0xF
                    c_lo = centroids_np[lo_idx]
                    c_hi = centroids_np[hi_idx]
                    raw_score += q[byte_idx * 2] * c_lo + q[byte_idx * 2 + 1] * c_hi
                    if norm_correction:
                        norm_sq += c_lo * c_lo + c_hi * c_hi

                if norm_correction:
                    inv_norm = 1.0 / math.sqrt(norm_sq + 1e-16)
                    raw_score *= inv_norm

                vec_norm = fp16_bytes_to_float(slot, MSE_BYTES)
                score = raw_score * vec_norm * sm_scale

                # Online softmax
                new_max = max(running_max, score)
                alpha = math.exp(running_max - new_max)
                weight = math.exp(score - new_max)
                running_sum = running_sum * alpha + weight
                acc = acc * alpha

                # Dequantize value
                val = np.zeros(D, dtype=np.float32)
                for byte_idx in range(VAL_DATA_BYTES):
                    packed = int(slot[KEY_PACKED_SIZE + byte_idx])
                    val[byte_idx * 2] = float(packed & 0xF)
                    val[byte_idx * 2 + 1] = float((packed >> 4) & 0xF)

                v_scale = fp16_bytes_to_float(slot, KEY_PACKED_SIZE + VAL_DATA_BYTES)
                v_zero = fp16_bytes_to_float(slot, KEY_PACKED_SIZE + VAL_DATA_BYTES + 2)
                val = val * v_scale + v_zero

                acc += weight * val
                running_max = new_max

            output[b, qh] = acc / running_sum

    return output.astype(np.float16)
