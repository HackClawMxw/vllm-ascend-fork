"""Generate test data for direct-invoke verification."""
import math
import os
import struct
import numpy as np
import torch

# Test parameters
B, Hq, Hk, D = 1, 32, 8, 128
BLOCK_SIZE = 16
SEQ_LEN = 32
N_CENTROIDS = 16
MSE_BITS = 4
MSE_BYTES = math.ceil(D * MSE_BITS / 8)  # 64
KEY_PACKED_SIZE = MSE_BYTES + 2  # 66
VAL_BITS = 4
VAL_DATA_BYTES = math.ceil(D * VAL_BITS / 8)  # 64
SLOT_SIZE = KEY_PACKED_SIZE + VAL_DATA_BYTES + 4  # 134 (key_packed + val_data + scale + zero)

MAX_BLOCKS = math.ceil(SEQ_LEN / BLOCK_SIZE)
NUM_BLOCKS = MAX_BLOCKS
GQA_RATIO = Hq // Hk

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

np.random.seed(42)

# 1. Query (B, Hq, D) fp16
query = np.random.randn(B, Hq, D).astype(np.float16)
with open("input/query.bin", "wb") as f:
    f.write(query.tobytes())

# 2. Centroids (16,) fp32
centroids = np.random.randn(N_CENTROIDS).astype(np.float32)
with open("input/centroids.bin", "wb") as f:
    f.write(centroids.tobytes())

# 3. Block table (B, max_blocks) int32
block_table = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(1, MAX_BLOCKS)
with open("input/block_table.bin", "wb") as f:
    f.write(block_table.tobytes())

# 4. Seq lens (B,) int32
seq_lens = np.array([SEQ_LEN], dtype=np.int32)
with open("input/seq_lens.bin", "wb") as f:
    f.write(seq_lens.tobytes())

# 5. KV cache (num_blocks, block_size, Hk, slot_size) uint8
# Pack synthetic data into slot format:
#   [mse_indices(64B) | vec_norm(2B) | val_indices(64B) | val_scale(2B) | val_zero(2B)]
kv_cache = np.zeros((NUM_BLOCKS, BLOCK_SIZE, Hk, SLOT_SIZE), dtype=np.uint8)

for block in range(NUM_BLOCKS):
    for slot in range(BLOCK_SIZE):
        for head in range(Hk):
            offset = 0
            # MSE indices: random 4-bit packed bytes
            mse_indices = np.random.randint(0, 16, size=D, dtype=np.uint8)
            for byte_idx in range(MSE_BYTES):
                lo = mse_indices[byte_idx * 2] & 0xF
                hi = mse_indices[byte_idx * 2 + 1] & 0xF
                packed = int(lo) | (int(hi) << 4)
                kv_cache[block, slot, head, offset] = packed
                offset += 1

            # vec_norm: random fp16 as 2 bytes (little-endian)
            vec_norm = np.float16(np.random.randn())
            norm_bytes = np.frombuffer(vec_norm.tobytes(), dtype=np.uint8)
            kv_cache[block, slot, head, offset:offset+2] = norm_bytes
            offset += 2

            # Value indices: random 4-bit packed bytes
            val_indices = np.random.randint(0, 16, size=D, dtype=np.uint8)
            for byte_idx in range(VAL_DATA_BYTES):
                lo = val_indices[byte_idx * 2] & 0xF
                hi = val_indices[byte_idx * 2 + 1] & 0xF
                packed = int(lo) | (int(hi) << 4)
                kv_cache[block, slot, head, offset] = packed
                offset += 1

            # val_scale: random fp16
            v_scale = np.float16(abs(np.random.randn()) + 0.1)
            scale_bytes = np.frombuffer(v_scale.tobytes(), dtype=np.uint8)
            kv_cache[block, slot, head, offset:offset+2] = scale_bytes
            offset += 2

            # val_zero: random fp16
            v_zero = np.float16(np.random.randn())
            zero_bytes = np.frombuffer(v_zero.tobytes(), dtype=np.uint8)
            kv_cache[block, slot, head, offset:offset+2] = zero_bytes
            offset += 2

with open("input/kv_cache.bin", "wb") as f:
    f.write(kv_cache.tobytes())

print(f"Generated test data: B={B}, Hq={Hq}, Hk={Hk}, D={D}, seq_len={SEQ_LEN}")
print(f"  block_size={BLOCK_SIZE}, num_blocks={NUM_BLOCKS}, slot_size={SLOT_SIZE}")
print(f"  kv_cache shape: {kv_cache.shape}, size: {kv_cache.nbytes} bytes")
