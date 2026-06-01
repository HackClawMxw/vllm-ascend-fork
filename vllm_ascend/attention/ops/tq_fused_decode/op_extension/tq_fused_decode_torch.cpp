#include "ops.h"
#include "../op_kernel/tq_fused_decode_tiling.h"

#include <acl/acl.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/NPUGuard.h>

#include <cstdint>
#include <cstdio>

// Kernel entry declared with __global__ __aicore__ in .asc file.
// When compiled into the shared library, it becomes a C-linkage function
// with signature: (blockDim, l2Ctrl, stream, ...gm_args...)
extern "C" void tq_fused_decode_kernel(
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

    int64_t B = q.size(0);
    int64_t Hq = q.size(1);
    int64_t D = q.size(2);
    int64_t Hk = kv.size(2);
    int64_t slot_size = kv.size(3);
    int64_t max_blocks = bt.size(1);

    auto output = at::empty({B, Hq, D}, q.options().dtype(at::kHalf));

    // Clear NPU stream (required before direct kernel call)
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Query vector core count
    uint32_t coreCount = 0;
    int32_t deviceId = c10_npu::current_device();
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreCount);

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

    uint32_t gridSize = tiling.gridSize;
    uint32_t usedCores = (coreCount < gridSize) ? coreCount : gridSize;
    if (usedCores == 0) usedCores = 1;

    // Copy tiling to device via temporary tensor
    auto tilingTensor = at::from_blob(
        &tiling, {static_cast<int64_t>(sizeof(TqFusedDecodeTilingData))},
        at::kByte).to(q.device()).clone();

    // Launch kernel via direct C function call (not <<<>>> syntax)
    tq_fused_decode_kernel(
        usedCores, nullptr, aclStream,
        q.data_ptr(), kv.data_ptr(), bt.data_ptr(),
        sl.data_ptr(), ct.data_ptr(), output.data_ptr(),
        tilingTensor.data_ptr());

    return output;
}

} // namespace ascend_kernel
