"""Verify direct-invoke output against Python golden reference."""
import numpy as np
from gen_data import *
from golden import compute_attention_python

# Load inputs
query = np.fromfile("input/query.bin", dtype=np.float16).reshape(B, Hq, D)
kv_cache = np.fromfile("input/kv_cache.bin", dtype=np.uint8).reshape(
    NUM_BLOCKS, BLOCK_SIZE, Hk, SLOT_SIZE)
block_table = np.fromfile("input/block_table.bin", dtype=np.int32).reshape(B, MAX_BLOCKS)
seq_lens = np.fromfile("input/seq_lens.bin", dtype=np.int32)
centroids = np.fromfile("input/centroids.bin", dtype=np.float32)

# Load kernel output
if os.path.exists("output/output.bin"):
    kernel_output = np.fromfile("output/output.bin", dtype=np.float16).reshape(B, Hq, D)
else:
    print("No kernel output found, skipping verification")
    exit(0)

# Compute golden reference
sm_scale = 1.0 / math.sqrt(D)
golden_output = compute_attention_python(
    query, kv_cache, block_table, seq_lens, centroids, sm_scale,
    norm_correction=True
)

# Compare
diff = np.abs(kernel_output.astype(np.float32) - golden_output.astype(np.float32))
max_diff = diff.max()
mean_diff = diff.mean()

# Cosine similarity
def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-16)

cos_sim = cosine_sim(kernel_output, golden_output)

print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")

if cos_sim > 0.999:
    print("PASS: Cosine similarity > 0.999")
else:
    print("FAIL: Cosine similarity < 0.999")
    # Print first few values for debugging
    print(f"  Kernel output[0,0,:8]: {kernel_output[0, 0, :8]}")
    print(f"  Golden output[0,0,:8]: {golden_output[0, 0, :8]}")
