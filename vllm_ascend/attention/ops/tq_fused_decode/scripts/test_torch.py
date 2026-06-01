"""PyTorch path test: load .so and call tq_fused_decode via torch.ops."""
import math
import os
import sys
import numpy as np
import torch

# Add parent directory to path for golden import
sys.path.insert(0, os.path.dirname(__file__))
from gen_data import *
from golden import compute_attention_python

# Build .so path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
so_path = os.path.join(script_dir, "build", "libtq_fused_decode_ops.so")

if not os.path.exists(so_path):
    print(f"Shared library not found at {so_path}")
    print("Build first with: cd build && cmake .. && make")
    sys.exit(1)

torch.ops.load_library(so_path)

# Generate test data on NPU
device = torch.device("npu:0")

query = torch.randn(B, Hq, D, dtype=torch.float16, device=device)
centroids_t = torch.randn(N_CENTROIDS, dtype=torch.float32, device=device)
block_table_t = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device).reshape(1, MAX_BLOCKS)
seq_lens_t = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)

# Generate synthetic packed KV cache
kv_cache_t = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, Hk, SLOT_SIZE, dtype=torch.uint8, device=device)
# Fill with random packed data
kv_cache_t.copy_(torch.randint(0, 256, (NUM_BLOCKS, BLOCK_SIZE, Hk, SLOT_SIZE),
                                dtype=torch.uint8))

sm_scale = 1.0 / math.sqrt(D)

# Call operator
output = torch.ops.npu.tq_fused_decode(
    query, kv_cache_t, block_table_t, seq_lens_t, centroids_t,
    sm_scale=sm_scale,
    mse_bytes=MSE_BYTES,
    key_packed_size=KEY_PACKED_SIZE,
    val_data_bytes=VAL_DATA_BYTES,
    head_dim=D,
    block_size=BLOCK_SIZE,
    norm_correction=True,
)

print(f"Output shape: {output.shape}, dtype: {output.dtype}")
print(f"Output[0, 0, :8]: {output[0, 0, :8]}")

# Compute golden reference on CPU
query_cpu = query.cpu().numpy()
kv_cache_cpu = kv_cache_t.cpu().numpy()
block_table_cpu = block_table_t.cpu().numpy()
seq_lens_cpu = seq_lens_t.cpu().numpy()
centroids_cpu = centroids_t.cpu().numpy()

golden = compute_attention_python(
    query_cpu, kv_cache_cpu, block_table_cpu, seq_lens_cpu,
    centroids_cpu, sm_scale, norm_correction=True
)

# Compare
output_cpu = output.cpu().numpy()
diff = np.abs(output_cpu.astype(np.float32) - golden.astype(np.float32))
cos_sim = float(np.dot(output_cpu.flatten(), golden.flatten())) / (
    np.linalg.norm(output_cpu.flatten()) * np.linalg.norm(golden.flatten()) + 1e-16)

print(f"Max diff: {diff.max():.6f}, Cosine sim: {cos_sim:.6f}")
if cos_sim > 0.999:
    print("PASS")
else:
    print("FAIL (cosine sim < 0.999)")
