#include "kernel_operator.h"
#include "tq_fused_decode_tiling.h"

using namespace AscendC;

// Set to 1 to enable diagnostic prints for head 19, 0 for production.
#define TQ_KERNEL_DEBUG 1

// Cache geometry constants
constexpr uint32_t SLOT_SIZE = 134;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t MSE_BYTES = 64;
constexpr uint32_t KEY_PACKED_SIZE = 66;
constexpr uint32_t VAL_DATA_BYTES = 64;
constexpr uint32_t CENTROID_TABLE_SIZE = 16;

// UB buffer sizes (all 32-byte aligned)
constexpr uint32_t QUERY_BUF_SIZE = HEAD_DIM * sizeof(float);           // 512
constexpr uint32_t CENTROID_BUF_SIZE = CENTROID_TABLE_SIZE * sizeof(float); // 64
constexpr uint32_t ACC_BUF_SIZE = HEAD_DIM * sizeof(float);             // 512
constexpr uint32_t VAL_BUF_SIZE = HEAD_DIM * sizeof(float);             // 512
constexpr uint32_t PRODUCT_BUF_SIZE = HEAD_DIM * sizeof(float);         // 512
// ReduceSum workLocal: min 8 floats for 128-element input; allocate 64 for margin
constexpr uint32_t WORK_BUF_SIZE = 64 * sizeof(float);                  // 256
constexpr uint32_t SLOT_BUF_ALIGNED = ((SLOT_SIZE + 31) / 32) * 32;    // 160
constexpr uint32_t TILING_BUF_ALIGNED = ((sizeof(TqFusedDecodeTilingData) + 31) / 32) * 32;
constexpr uint32_t SLOT_QUEUE_BUF = HEAD_DIM * sizeof(half);            // 256

class KernelTqFusedDecode {
public:
    __aicore__ inline void Init(GM_ADDR queryRot, GM_ADDR kvCache,
                                 GM_ADDR blockTable, GM_ADDR seqLens,
                                 GM_ADDR centroidsTiling, GM_ADDR output);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ReadFp16AsFp32(LocalTensor<uint8_t>& slotUb, uint32_t offset);
    __aicore__ inline void ComputeScoreAndAccumulate(LocalTensor<uint8_t>& slotUb);
    __aicore__ inline void DequantValueAndAccumulate(LocalTensor<uint8_t>& slotUb,
                                                      float weight);
    __aicore__ inline void ReadTiling(GM_ADDR centroidsTiling);

    TPipe pipe;

    // GM tensors
    GlobalTensor<half> queryRotGm;
    GlobalTensor<uint8_t> kvCacheGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> seqLensGm;
    GlobalTensor<float> centroidsGm;
    GlobalTensor<half> outputGm;

    GM_ADDR queryRotBase_;
    GM_ADDR kvCacheBase_;
    GM_ADDR outputBase_;

    // UB buffers — valBuf / productBuf are time-shared between score and
    // value-accumulate phases (no overlap in lifetime).
    TBuf<QuePosition::VECCALC> queryBuf;      // 512B — query fp32 [128]
    TBuf<QuePosition::VECCALC> centroidBuf;   //  64B — centroid table [16]
    TBuf<QuePosition::VECCALC> accBuf;        // 512B — output accumulator [128]
    TBuf<QuePosition::VECCALC> valBuf;        // 512B — gathered centroids (score) / dequant values (accum)
    TBuf<QuePosition::VECCALC> productBuf;    // 512B — query*centroid (score) / weight*val (accum)
    TBuf<QuePosition::VECCALC> workBuf;       // 256B — ReduceSum workspace + Exp/Sqrt temp
    TQue<QuePosition::VECIN, 1> slotQueue;    // 256B — slot data / fp16 staging

    // Tiling data
    TqFusedDecodeTilingData tiling;

    // Per-work-item state
    uint32_t batchIdx_;
    uint32_t qheadIdx_;
    uint32_t kvHead_;
    uint32_t seqLen_;
    uint32_t curTokenPos_;  // for diagnostics only

    // Online softmax state (scalar, kept in registers)
    float runningMax_;
    float runningSum_;
};

// ---- FP16 byte reinterpret ----

__aicore__ inline float KernelTqFusedDecode::ReadFp16AsFp32(
    LocalTensor<uint8_t>& slotUb, uint32_t offset) {
    uint16_t raw = (static_cast<uint16_t>(slotUb.GetValue(offset + 1)) << 8) |
                   static_cast<uint16_t>(slotUb.GetValue(offset));
    union { uint16_t u; half h; } cvt;
    cvt.u = raw;
    return static_cast<float>(cvt.h);
}

// ---- Tiling data read via DataCopy + GetValue ----

__aicore__ inline void KernelTqFusedDecode::ReadTiling(GM_ADDR centroidsTiling) {
    GlobalTensor<int32_t> tilingGm;
    tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(centroidsTiling));

    auto slotLocal = slotQueue.AllocTensor<int32_t>();
    constexpr uint32_t kTilingInt32Count = TILING_BUF_ALIGNED / sizeof(int32_t);  // 24
    DataCopy(slotLocal, tilingGm, kTilingInt32Count);
    PipeBarrier<PIPE_ALL>();

    tiling.batchSize       = slotLocal.GetValue(0);
    tiling.numQueryHeads   = slotLocal.GetValue(1);
    tiling.numKvHeads      = slotLocal.GetValue(2);
    tiling.gqaRatio        = slotLocal.GetValue(3);
    tiling.gridSize        = slotLocal.GetValue(4);
    tiling.blockSize       = slotLocal.GetValue(5);
    tiling.headDim         = slotLocal.GetValue(6);
    tiling.mseBytes        = slotLocal.GetValue(7);
    tiling.keyPackedSize   = slotLocal.GetValue(8);
    tiling.valDataBytes    = slotLocal.GetValue(9);
    tiling.slotSize        = slotLocal.GetValue(10);
    tiling.maxBlocksPerSeq = slotLocal.GetValue(11);
    uint32_t lo12 = slotLocal.GetValue(12);
    uint32_t hi13 = slotLocal.GetValue(13);
    tiling.blockStride = (static_cast<uint64_t>(hi13) << 32) | lo12;
    uint32_t lo14 = slotLocal.GetValue(14);
    uint32_t hi15 = slotLocal.GetValue(15);
    tiling.slotStride = (static_cast<uint64_t>(hi15) << 32) | lo14;
    tiling.normCorrection  = slotLocal.GetValue(16);
    uint32_t rawScale = slotLocal.GetValue(17);
    union { uint32_t u; float f; } cvtScale;
    cvtScale.u = rawScale;
    tiling.smScale = cvtScale.f;

    slotQueue.FreeTensor(slotLocal);
}

// ---- Init ----

__aicore__ inline void KernelTqFusedDecode::Init(
    GM_ADDR queryRot, GM_ADDR kvCache, GM_ADDR blockTable,
    GM_ADDR seqLens, GM_ADDR centroidsTiling, GM_ADDR output) {

    uint32_t blockIdx = GetBlockIdx();

    pipe.InitBuffer(slotQueue, 1, SLOT_QUEUE_BUF);
    pipe.InitBuffer(queryBuf, QUERY_BUF_SIZE);
    pipe.InitBuffer(centroidBuf, CENTROID_BUF_SIZE);
    pipe.InitBuffer(accBuf, ACC_BUF_SIZE);
    pipe.InitBuffer(valBuf, VAL_BUF_SIZE);
    pipe.InitBuffer(productBuf, PRODUCT_BUF_SIZE);
    pipe.InitBuffer(workBuf, WORK_BUF_SIZE);

    ReadTiling(centroidsTiling);

    if (blockIdx >= tiling.gridSize) return;

    batchIdx_ = blockIdx / tiling.numQueryHeads;
    qheadIdx_ = blockIdx % tiling.numQueryHeads;
    kvHead_ = qheadIdx_ / tiling.gqaRatio;

    // Bind GM tensors
    queryRotGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queryRot));
    kvCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache));
    blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    seqLensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    // Centroids start after tiling data (96 bytes = 24 float offsets)
    centroidsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(centroidsTiling) + 24);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));

    queryRotBase_ = queryRot;
    kvCacheBase_ = kvCache;
    outputBase_ = output;

    seqLen_ = static_cast<uint32_t>(seqLensGm.GetValue(batchIdx_));

    // Load query as fp16, then Cast to fp32 for computation
    uint64_t qOffset = static_cast<uint64_t>(batchIdx_) * tiling.numQueryHeads * tiling.headDim
                     + static_cast<uint64_t>(qheadIdx_) * tiling.headDim;
    GlobalTensor<half> queryFp16Gm;
    queryFp16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queryRotBase_) + qOffset);
    auto queryFp16 = slotQueue.AllocTensor<half>();
    DataCopy(queryFp16, queryFp16Gm, HEAD_DIM);
    PipeBarrier<PIPE_ALL>();

    auto queryFp32 = queryBuf.Get<float>();
    Cast(queryFp32, queryFp16, RoundMode::CAST_ROUND, HEAD_DIM);
    pipe_barrier(PIPE_V);
    slotQueue.FreeTensor(queryFp16);

    // Load centroid table (16 fp32)
    auto centroidLocal = centroidBuf.Get<float>();
    DataCopy(centroidLocal, centroidsGm, CENTROID_TABLE_SIZE);
    PipeBarrier<PIPE_ALL>();

    // Initialize accumulator to zero
    auto accLocal = accBuf.Get<float>();
    Duplicate(accLocal, 0.0f, HEAD_DIM);
    pipe_barrier(PIPE_V);

    // Initialize online softmax state
    runningMax_ = -3.4e38f;
    runningSum_ = 0.0f;
}

// ---- Vectorized score computation + online softmax + value accumulate ----
//
// Pipeline per token:
//   1. Scalar SetValue: gather 128 centroid values by 4-bit key indices → valBuf
//   2. Vector Mul + ReduceSum: normSq = sum(centroid^2)              → productBuf
//   3. Vector Mul + ReduceSum: rawScore = sum(query * centroid)      → productBuf
//   4. Scalar Sqrt via workBuf: normalize rawScore
//   5. Vector Exp(2): batched alpha + weight                         → workBuf
//   6. Vector Muls: acc *= alpha
//   7. DequantValueAndAccumulate: unpack + dequant + accumulate

__aicore__ inline void KernelTqFusedDecode::ComputeScoreAndAccumulate(
    LocalTensor<uint8_t>& slotUb) {
    auto queryLocal = queryBuf.Get<float>();
    auto centroidLocal = centroidBuf.Get<float>();
    auto gathered = valBuf.Get<float>();
    auto product = productBuf.Get<float>();
    auto work = workBuf.Get<float>();

    // Phase 1: Gather centroids by 4-bit key indices (scalar SetValue — pure stores,
    // no accumulation, compiler will not apply reduction auto-vectorization).
    for (uint32_t byteIdx = 0; byteIdx < MSE_BYTES; byteIdx++) {
        uint8_t packed = slotUb.GetValue(byteIdx);
        gathered.SetValue(byteIdx * 2 + 0, centroidLocal.GetValue(packed & 0xF));
        gathered.SetValue(byteIdx * 2 + 1, centroidLocal.GetValue((packed >> 4) & 0xF));
    }
    PipeBarrier<PIPE_ALL>();

#if TQ_KERNEL_DEBUG
    // DIAG-STEP1: verify gathered centroids after SetValue loop
    if (GetBlockIdx() == 19 && curTokenPos_ == 0) {
        AscendC::printf("TQ-S1 gathered[0:4]=%f %f %f %f q[0:4]=%f %f %f %f\n",
            gathered.GetValue(0), gathered.GetValue(1),
            gathered.GetValue(2), gathered.GetValue(3),
            queryLocal.GetValue(0), queryLocal.GetValue(1),
            queryLocal.GetValue(2), queryLocal.GetValue(3));
    }
#endif

    // Phase 2: normSq = sum(centroid^2) — computed before rawScore overwrites productBuf
    float normSq = 0.0f;
    if (tiling.normCorrection) {
        Mul(product, gathered, gathered, HEAD_DIM);
        pipe_barrier(PIPE_V);

#if TQ_KERNEL_DEBUG
        // DIAG-STEP2: verify centroid^2 products
        if (GetBlockIdx() == 19 && curTokenPos_ == 0) {
            AscendC::printf("TQ-S2 c^2[0:4]=%f %f %f %f\n",
                product.GetValue(0), product.GetValue(1),
                product.GetValue(2), product.GetValue(3));
        }
#endif

        ReduceSum(product, product, work, HEAD_DIM);
        PipeBarrier<PIPE_ALL>();
        normSq = product.GetValue(0);
    }

    // Phase 3: rawScore = sum(query * centroid)
    Mul(product, queryLocal, gathered, HEAD_DIM);
    pipe_barrier(PIPE_V);

#if TQ_KERNEL_DEBUG
    // DIAG-STEP3: verify query*centroid products BEFORE ReduceSum
    if (GetBlockIdx() == 19 && curTokenPos_ == 0) {
        AscendC::printf("TQ-S3 q*c[0:4]=%f %f %f %f\n",
            product.GetValue(0), product.GetValue(1),
            product.GetValue(2), product.GetValue(3));
    }
#endif

    ReduceSum(product, product, work, HEAD_DIM);
    PipeBarrier<PIPE_ALL>();
    float rawScore = product.GetValue(0);

    // Phase 4: Normalize rawScore /= sqrt(normSq + eps)
    if (tiling.normCorrection) {
        work.SetValue(0, normSq + 1e-16f);
        PipeBarrier<PIPE_ALL>();
        Sqrt(work, work, 1);
        PipeBarrier<PIPE_ALL>();
        rawScore /= work.GetValue(0);
    }

    // Phase 5: Final attention score = rawScore * vecNorm * smScale
    float vecNorm = ReadFp16AsFp32(slotUb, MSE_BYTES);
    float score = rawScore * vecNorm * tiling.smScale;

#if TQ_KERNEL_DEBUG
    if (GetBlockIdx() == 19 && (curTokenPos_ == 0 || curTokenPos_ == seqLen_ - 1)) {
        AscendC::printf("TQ-VEC SCORE tok=%d rawSc=%f nSq=%f vn=%f score=%f\n",
                         curTokenPos_, rawScore, normSq, vecNorm, score);
    }
#endif

    // Phase 6: Online softmax — batched vector Exp for both alpha and weight
    float newMax = (score > runningMax_) ? score : runningMax_;
    work.SetValue(0, runningMax_ - newMax);  // alpha input
    work.SetValue(1, score - newMax);        // weight input
    PipeBarrier<PIPE_ALL>();
    Exp(work, work, 2);
    PipeBarrier<PIPE_ALL>();
    float alpha = work.GetValue(0);
    float weight = work.GetValue(1);

    runningSum_ = runningSum_ * alpha + weight;

    // Phase 7: Scale accumulator: acc *= alpha
    auto accLocal = accBuf.Get<float>();
    Muls(accLocal, accLocal, alpha, HEAD_DIM);
    pipe_barrier(PIPE_V);

    // Phase 8: Dequantize value and accumulate
    DequantValueAndAccumulate(slotUb, weight);

    runningMax_ = newMax;
}

// ---- Value dequantization + accumulation ----
// Value uses uniform (linear) quantization: val = index * scale + zero
// The 4-bit indices are NOT centroid lookups, they are integer indices 0-15.

__aicore__ inline void KernelTqFusedDecode::DequantValueAndAccumulate(
    LocalTensor<uint8_t>& slotUb, float weight) {
    auto valLocal = valBuf.Get<float>();
    auto accLocal = accBuf.Get<float>();
    auto productLocal = productBuf.Get<float>();

    // Unpack 4-bit value indices into float (scalar SetValue — pure stores, safe)
    for (uint32_t byteIdx = 0; byteIdx < VAL_DATA_BYTES; byteIdx++) {
        uint8_t packed = slotUb.GetValue(KEY_PACKED_SIZE + byteIdx);
        valLocal.SetValue(byteIdx * 2 + 0, static_cast<float>(packed & 0xF));
        valLocal.SetValue(byteIdx * 2 + 1, static_cast<float>((packed >> 4) & 0xF));
    }
    PipeBarrier<PIPE_ALL>();

    float vScale = ReadFp16AsFp32(slotUb, KEY_PACKED_SIZE + VAL_DATA_BYTES);
    float vZero = ReadFp16AsFp32(slotUb, KEY_PACKED_SIZE + VAL_DATA_BYTES + 2);

    // Dequantize: val = index * scale + zero
    Muls(valLocal, valLocal, vScale, HEAD_DIM);
    pipe_barrier(PIPE_V);
    Adds(valLocal, valLocal, vZero, HEAD_DIM);
    pipe_barrier(PIPE_V);

    // acc += weight * val
    Muls(productLocal, valLocal, weight, HEAD_DIM);
    pipe_barrier(PIPE_V);
    Add(accLocal, accLocal, productLocal, HEAD_DIM);
    pipe_barrier(PIPE_V);
}

// ---- Process: main loop over all KV tokens ----

__aicore__ inline void KernelTqFusedDecode::Process() {
    if (GetBlockIdx() >= tiling.gridSize) return;

    // Handle empty sequence — output zeros
    if (seqLen_ == 0) {
        auto accLocal = accBuf.Get<float>();
        auto slotOut = slotQueue.AllocTensor<uint8_t>();
        auto outFp16 = slotOut.ReinterpretCast<half>();
        Cast(outFp16, accLocal, RoundMode::CAST_ROUND, tiling.headDim);
        pipe_barrier(PIPE_V);
        uint64_t outOff = static_cast<uint64_t>(batchIdx_) * tiling.numQueryHeads * tiling.headDim
                        + static_cast<uint64_t>(qheadIdx_) * tiling.headDim;
        GlobalTensor<half> outOffsetGm;
        outOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outputBase_) + outOff);
        DataCopy(outOffsetGm, outFp16, tiling.headDim);
        PipeBarrier<PIPE_ALL>();
        slotQueue.FreeTensor(slotOut);
        return;
    }

    // Allocate slot buffer once, reuse for all tokens
    auto slotLocal = slotQueue.AllocTensor<uint8_t>();

    for (uint32_t tokenPos = 0; tokenPos < seqLen_; tokenPos++) {
        // Paged address computation
        uint32_t virtualBlock = tokenPos / tiling.blockSize;
        uint32_t offsetInBlock = tokenPos % tiling.blockSize;
        uint32_t physicalBlock = static_cast<uint32_t>(
            blockTableGm.GetValue(
                static_cast<int64_t>(batchIdx_) * tiling.maxBlocksPerSeq + virtualBlock));

        uint64_t slotAddr = static_cast<uint64_t>(physicalBlock) * tiling.blockStride
                          + static_cast<uint64_t>(offsetInBlock) * tiling.slotStride
                          + static_cast<uint64_t>(kvHead_) * tiling.slotSize;

        // Read slot data (aligned size for DataCopy)
        GlobalTensor<uint8_t> kvSlotGm;
        kvSlotGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCacheBase_) + slotAddr);
        DataCopy(slotLocal, kvSlotGm, SLOT_BUF_ALIGNED);
        PipeBarrier<PIPE_ALL>();

        curTokenPos_ = tokenPos;
        ComputeScoreAndAccumulate(slotLocal);
    }

    // Final normalization: output = acc / runningSum
    auto accLocal = accBuf.Get<float>();

#if TQ_KERNEL_DEBUG
    if (GetBlockIdx() == 19) {
        AscendC::printf("TQ-VEC FINAL max=%f sum=%f acc0=%f acc1=%f\n",
                         runningMax_, runningSum_,
                         accLocal.GetValue(0), accLocal.GetValue(1));
    }
#endif

    float invSum = 1.0f / runningSum_;
    Muls(accLocal, accLocal, invSum, tiling.headDim);
    pipe_barrier(PIPE_V);

    // Cast fp32 -> fp16 and write output
    auto outFp16 = slotLocal.ReinterpretCast<half>();
    Cast(outFp16, accLocal, RoundMode::CAST_ROUND, tiling.headDim);
    pipe_barrier(PIPE_V);

    uint64_t outOff = static_cast<uint64_t>(batchIdx_) * tiling.numQueryHeads * tiling.headDim
                    + static_cast<uint64_t>(qheadIdx_) * tiling.headDim;
    GlobalTensor<half> outOffsetGm;
    outOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outputBase_) + outOff);
    DataCopy(outOffsetGm, outFp16, tiling.headDim);
    PipeBarrier<PIPE_ALL>();
    slotQueue.FreeTensor(slotLocal);
}

// ---- Kernel entry point ----

extern "C" __global__ __aicore__ void tq_fused_decode_kernel(
    GM_ADDR queryRot, GM_ADDR kvCache, GM_ADDR blockTable,
    GM_ADDR seqLens, GM_ADDR centroidsTiling, GM_ADDR output) {
    KernelTqFusedDecode op;
    op.Init(queryRot, kvCache, blockTable, seqLens, centroidsTiling, output);
    op.Process();
}
