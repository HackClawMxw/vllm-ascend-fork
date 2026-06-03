#include "ops.h"
#include "../op_kernel/tq_fused_decode_tiling.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <cstdio>

// The CCE compiler generates tq_fused_decode_kernel as an empty stub (single ret
// instruction). The ACTUAL launch function is aclrtlaunch_tq_fused_decode_kernel,
// which packages arguments into void*[] and calls rtKernelLaunchWithHandle.
extern "C" void aclrtlaunch_tq_fused_decode_kernel(
    uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    void* queryRot, void* kvCache, void* blockTable,
    void* seqLens, void* centroids, void* output, void* tiling);

namespace ascend_kernel {

torch::Tensor tq_fused_decode_torch(
    const torch::Tensor& query_rot,
    const torch::Tensor& kv_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    const torch::Tensor& centroids,
    double sm_scale,
    int64_t mse_bytes,
    int64_t key_packed_size,
    int64_t val_data_bytes,
    int64_t head_dim,
    int64_t block_size,
    bool norm_correction) {
    auto q = query_rot.contiguous();
    auto kv = kv_cache.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto ct = centroids.contiguous();

    // Diagnostics: verify all tensors are on NPU with expected shapes
    printf("[TQ-HOST] q: dev=%d shape=[%ld,%ld,%ld] ptr=%p\n",
           static_cast<int>(q.device().type()),
           q.size(0), q.size(1), q.size(2), q.data_ptr());
    printf("[TQ-HOST] kv: dev=%d shape=[%ld,%ld,%ld,%ld] ptr=%p\n",
           static_cast<int>(kv.device().type()),
           kv.size(0), kv.size(1), kv.size(2), kv.size(3), kv.data_ptr());
    printf("[TQ-HOST] ct: dev=%d numel=%ld ptr=%p\n",
           static_cast<int>(ct.device().type()),
           ct.numel(), ct.data_ptr());
    fflush(stdout);

    int64_t B = q.size(0);
    int64_t Hq = q.size(1);
    int64_t D = q.size(2);
    int64_t Hk = kv.size(2);
    int64_t slot_size = kv.size(3);
    int64_t max_blocks = bt.size(1);

    // DIAGNOSTIC: pre-fill with sentinel to detect if kernel actually writes.
    // If output is still -42.0 after kernel launch, the kernel did NOT run.
    auto output = at::full({B, Hq, D}, -42.0, q.options().dtype(at::kHalf));

    // Get NPU stream
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Compute tiling
    TqFusedDecodeTilingData tiling;
    memset(&tiling, 0, sizeof(tiling));
    tiling.batchSize = static_cast<uint32_t>(B);
    tiling.numQueryHeads = static_cast<uint32_t>(Hq);
    tiling.numKvHeads = static_cast<uint32_t>(Hk);
    tiling.gqaRatio = static_cast<uint32_t>(Hq / Hk);
    tiling.gridSize = static_cast<uint32_t>(B * Hq);
    tiling.blockSize = static_cast<uint32_t>(block_size);
    tiling.headDim = static_cast<uint32_t>(head_dim);
    tiling.mseBytes = static_cast<uint32_t>(mse_bytes);
    tiling.keyPackedSize = static_cast<uint32_t>(key_packed_size);
    tiling.valDataBytes = static_cast<uint32_t>(val_data_bytes);
    tiling.slotSize = static_cast<uint32_t>(slot_size);
    tiling.maxBlocksPerSeq = static_cast<uint32_t>(max_blocks);
    tiling.blockStride = static_cast<uint64_t>(block_size) * Hk * slot_size;
    tiling.slotStride = static_cast<uint64_t>(Hk) * slot_size;
    tiling.normCorrection = norm_correction ? 1u : 0u;
    tiling.smScale = static_cast<float>(sm_scale);

    // Use gridSize as blockDim: each core handles one (batch, head) pair.
    // Extra cores exit early via GetBlockIdx() >= gridSize check in kernel.
    uint32_t blockDim = tiling.gridSize;
    if (blockDim == 0) blockDim = 1;

    // Copy tiling to device via temporary tensor.
    // The kernel's DataCopy reads TILING_BUF_ALIGNED bytes (ceil to 32-byte),
    // so the device buffer must be at least that large to avoid MTE overread.
    constexpr int64_t kTilingBufAligned =
        ((sizeof(TqFusedDecodeTilingData) + 31) / 32) * 32;
    auto tilingTensor = at::from_blob(
        &tiling, {kTilingBufAligned},
        at::kByte).to(q.device()).clone();

    // Launch kernel via direct C function call (not <<<>>> syntax)
    printf("[TQ-HOST] Launching kernel blockDim=%d gridSize=%d B=%ld Hq=%ld "
           "headDim=%d blockSize=%d slotSize=%d stream=%p\n",
           blockDim, tiling.gridSize, B, Hq,
           tiling.headDim, tiling.blockSize, tiling.slotSize,
           (void*)aclStream);
    fflush(stdout);
    aclrtlaunch_tq_fused_decode_kernel(
        blockDim, nullptr, aclStream,
        q.data_ptr(), kv.data_ptr(), bt.data_ptr(),
        sl.data_ptr(), ct.data_ptr(), output.data_ptr(),
        tilingTensor.data_ptr());
    printf("[TQ-HOST] aclrtlaunch returned\n");
    fflush(stdout);

    // Synchronize to force kernel completion and capture device-side errors.
    // If this returns non-zero, the kernel failed on the device.
    aclError aclRet = aclrtSynchronizeStream(aclStream);
    printf("[TQ-HOST] aclrtSynchronizeStream = %d (0=SUCCESS)\n", (int)aclRet);
    fflush(stdout);

    // Read back first element to verify if kernel wrote anything.
    auto check = output[0][0][0].item<at::Half>();
    printf("[TQ-HOST] output[0][0][0] = %f (expected 1.0 if kernel ran)\n",
           static_cast<float>(check));
    fflush(stdout);

    return output;
}

} // namespace ascend_kernel
