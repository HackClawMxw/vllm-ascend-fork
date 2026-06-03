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
    // Kernel reads query as half (fp16). Caller may pass float32.
    auto q = query_rot.contiguous();
    printf("[TQ-HOST] q before conv: dtype=%d\n", static_cast<int>(q.scalar_type()));
    fflush(stdout);
    if (q.scalar_type() != at::kHalf) {
        q = q.toType(at::kHalf);
    }
    printf("[TQ-HOST] q after conv: dtype=%d\n", static_cast<int>(q.scalar_type()));
    fflush(stdout);
    auto kv = kv_cache.contiguous();
    // Ensure correct device + dtype: kernel reads block_table/seq_lens as int32,
    // centroids as float32. Guard against host-side int64 or CPU tensors that
    // would cause MTE "DDR address out of range" (error 507057).
    auto bt = block_table.contiguous().to(q.device(), at::kInt);
    auto sl = seq_lens.contiguous().to(q.device(), at::kInt);
    torch::Tensor ct;
    if (centroids.scalar_type() != at::kFloat) {
        ct = centroids.contiguous().toType(at::kFloat).to(q.device());
    } else {
        ct = centroids.contiguous().to(q.device());
    }

    // Diagnostics: verify all tensors are on NPU with expected shapes/dtypes
    printf("[TQ-HOST] q: dev=%d dtype=%d shape=[%ld,%ld,%ld] ptr=%p\n",
           static_cast<int>(q.device().type()),
           static_cast<int>(q.scalar_type()),
           q.size(0), q.size(1), q.size(2), q.data_ptr());
    printf("[TQ-HOST] kv: dev=%d dtype=%d shape=[%ld,%ld,%ld,%ld] ptr=%p total=%ldB\n",
           static_cast<int>(kv.device().type()),
           static_cast<int>(kv.scalar_type()),
           kv.size(0), kv.size(1), kv.size(2), kv.size(3), kv.data_ptr(),
           kv.numel() * kv.element_size());
    printf("[TQ-HOST] ct: dev=%d dtype=%d numel=%ld ptr=%p\n",
           static_cast<int>(ct.device().type()),
           static_cast<int>(ct.scalar_type()),
           ct.numel(), ct.data_ptr());
    printf("[TQ-HOST] sl: dev=%d dtype=%d numel=%ld ptr=%p\n",
           static_cast<int>(sl.device().type()),
           static_cast<int>(sl.scalar_type()),
           sl.numel(), sl.data_ptr());
    printf("[TQ-HOST] bt: dev=%d dtype=%d shape=[%ld,%ld] ptr=%p\n",
           static_cast<int>(bt.device().type()),
           static_cast<int>(bt.scalar_type()),
           bt.size(0), bt.size(1), bt.data_ptr());
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

    // Copy tiling to device via aclrtMemcpy (synchronous H2D).
    // torch .to(device) may be async on torch_npu, causing the kernel
    // to read uninitialized NPU memory (poison pattern 0xD160D160).
    constexpr int64_t kTilingBufAligned =
        ((sizeof(TqFusedDecodeTilingData) + 31) / 32) * 32;  // = 96
    constexpr int64_t kTilingInt32Len = kTilingBufAligned / sizeof(int32_t); // = 24
    auto cpuTiling = torch::empty({kTilingInt32Len}, at::kInt);
    memcpy(cpuTiling.data_ptr<int32_t>(), &tiling, sizeof(tiling));

    // Allocate NPU tensor and copy synchronously
    auto tilingTensor = torch::empty({kTilingInt32Len}, q.options().dtype(at::kInt));
    aclError memcpyRet = aclrtMemcpy(
        tilingTensor.data_ptr(), kTilingBufAligned,
        cpuTiling.data_ptr(), kTilingBufAligned,
        ACL_MEMCPY_HOST_TO_DEVICE);
    printf("[TQ-HOST] tiling hex:");
    auto* tp = cpuTiling.data_ptr<int32_t>();
    for (int i = 0; i < 6; i++) printf(" %08x", static_cast<uint32_t>(tp[i]));
    printf(" ... npu_ptr=%p memcpy_ret=%d\n", tilingTensor.data_ptr(), (int)memcpyRet);
    fflush(stdout);

    // Readback verification: copy NPU→CPU and compare
    auto verifyTiling = torch::empty({kTilingInt32Len}, at::kInt);
    aclError rbRet = aclrtMemcpy(
        verifyTiling.data_ptr(), kTilingBufAligned,
        tilingTensor.data_ptr(), kTilingBufAligned,
        ACL_MEMCPY_DEVICE_TO_HOST);
    auto* vp = verifyTiling.data_ptr<int32_t>();
    bool match = true;
    for (int i = 0; i < 18; i++) {
        if (static_cast<uint32_t>(vp[i]) != static_cast<uint32_t>(tp[i])) {
            match = false;
            printf("[TQ-HOST] MISMATCH idx=%d host=%08x npu=%08x\n",
                   i, static_cast<uint32_t>(tp[i]), static_cast<uint32_t>(vp[i]));
        }
    }
    if (match) printf("[TQ-HOST] tiling readback OK (all 18 int32 match)\n");
    fflush(stdout);

    // Launch kernel via direct C function call (not <<<>>> syntax)
    printf("[TQ-HOST] Launching kernel blockDim=%d gridSize=%d B=%ld Hq=%ld "
           "Hk=%ld headDim=%d blockSize=%d slotSize=%d "
           "blockStride=%lu slotStride=%lu maxBlocks=%ld stream=%p\n",
           blockDim, tiling.gridSize, B, Hq, Hk,
           tiling.headDim, tiling.blockSize, tiling.slotSize,
           (unsigned long)tiling.blockStride, (unsigned long)tiling.slotStride,
           max_blocks, (void*)aclStream);
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
