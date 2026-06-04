#include "kernel_operator.h"
#include "tq_fused_decode_tiling.h"

using namespace AscendC;

// Cache geometry constants
constexpr uint32_t SLOT_SIZE = 134;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t MSE_BYTES = 64;
constexpr uint32_t KEY_PACKED_SIZE = 66;
constexpr uint32_t VAL_DATA_BYTES = 64;
constexpr uint32_t CENTROID_TABLE_SIZE = 16;

// UB buffer sizes (all 32-byte aligned)
constexpr uint32_t QUERY_BUF_SIZE = HEAD_DIM * sizeof(float);        // 512
constexpr uint32_t CENTROID_BUF_SIZE = CENTROID_TABLE_SIZE * sizeof(float); // 64
constexpr uint32_t ACC_BUF_SIZE = HEAD_DIM * sizeof(float);          // 512
constexpr uint32_t VAL_BUF_SIZE = HEAD_DIM * sizeof(float);          // 512
constexpr uint32_t WEIGHTED_BUF_SIZE = HEAD_DIM * sizeof(float);     // 512
// Slot data padded to 32-byte alignment for DataCopy
constexpr uint32_t SLOT_BUF_ALIGNED = ((SLOT_SIZE + 31) / 32) * 32; // 160
// Tiling struct padded to 32-byte alignment
constexpr uint32_t TILING_BUF_ALIGNED = ((sizeof(TqFusedDecodeTilingData) + 31) / 32) * 32;
// Slot queue buffer must hold the largest of: tiling (96B), slot data (160B),
// or query/output half values (HEAD_DIM * 2 = 256B).
constexpr uint32_t SLOT_QUEUE_BUF = HEAD_DIM * sizeof(half); // 256, already 32-byte aligned

class KernelTqFusedDecode {
public:
    __aicore__ inline void Init(GM_ADDR queryRot, GM_ADDR kvCache,
                                 GM_ADDR blockTable, GM_ADDR seqLens,
                                 GM_ADDR centroidsTiling, GM_ADDR output);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ScalarExp(float x);
    __aicore__ inline float ScalarSqrt(float x);
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

    // Raw GM base addresses for offset-based DataCopy
    GM_ADDR queryRotBase_;
    GM_ADDR kvCacheBase_;
    GM_ADDR outputBase_;

    // UB buffers
    TBuf<QuePosition::VECCALC> queryBuf;
    TBuf<QuePosition::VECCALC> centroidBuf;
    TBuf<QuePosition::VECCALC> accBuf;
    TBuf<QuePosition::VECCALC> valBuf;
    TBuf<QuePosition::VECCALC> weightedBuf;
    TQue<QuePosition::VECIN, 1> slotQueue;

    // Tiling data
    TqFusedDecodeTilingData tiling;

    // Per-work-item state
    uint32_t batchIdx_;
    uint32_t qheadIdx_;
    uint32_t kvHead_;
    uint32_t seqLen_;

    // Online softmax state (scalar, kept in registers)
    float runningMax_;
    float runningSum_;
};

// ---- Scalar math helpers using AscendC Vector API ----

__aicore__ inline float KernelTqFusedDecode::ScalarExp(float x) {
    LocalTensor<float> tmp;
    tmp = valBuf.Get<float>();
    tmp.SetValue(0, x);
    // SetValue is on the Scalar pipe; Exp is on Vector pipe.
    // pipe_barrier(PIPE_V) does NOT synchronize the Scalar pipe.
    // Without PIPE_ALL, the Vector unit may read stale UB.
    PipeBarrier<PIPE_ALL>();
    Exp(tmp, tmp, 1);
    // Same reasoning: GetValue on Scalar pipe must wait for Vector write.
    PipeBarrier<PIPE_ALL>();
    float result = tmp.GetValue(0);
    return result;
}

__aicore__ inline float KernelTqFusedDecode::ScalarSqrt(float x) {
    LocalTensor<float> tmp = valBuf.Get<float>();
    tmp.SetValue(0, x);
    PipeBarrier<PIPE_ALL>();
    Sqrt(tmp, tmp, 1);
    PipeBarrier<PIPE_ALL>();
    float result = tmp.GetValue(0);
    return result;
}

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
// Tiling is at offset 0 in the combined buffer. No pointer arithmetic needed.
// Centroids are at offset kTilingBufAligned (96 bytes = 24 floats) in the same buffer.

__aicore__ inline void KernelTqFusedDecode::ReadTiling(GM_ADDR centroidsTiling) {
    // Tiling is at the START of the combined buffer — no offset needed
    GlobalTensor<int32_t> tilingGm;
    tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(centroidsTiling));

    auto slotLocal = slotQueue.AllocTensor<int32_t>();
    constexpr uint32_t kTilingInt32Count = TILING_BUF_ALIGNED / sizeof(int32_t);  // 24
    DataCopy(slotLocal, tilingGm, kTilingInt32Count);
    PipeBarrier<PIPE_ALL>();

    // (removed per-core tiling dump — too verbose in production)
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

    // Init UB buffers
    pipe.InitBuffer(slotQueue, 1, SLOT_QUEUE_BUF);
    pipe.InitBuffer(queryBuf, QUERY_BUF_SIZE);
    pipe.InitBuffer(centroidBuf, CENTROID_BUF_SIZE);
    pipe.InitBuffer(accBuf, ACC_BUF_SIZE);
    pipe.InitBuffer(valBuf, VAL_BUF_SIZE);
    pipe.InitBuffer(weightedBuf, WEIGHTED_BUF_SIZE);

    ReadTiling(centroidsTiling);

    // Early exit if this core has no work
    if (blockIdx >= tiling.gridSize) return;

    batchIdx_ = blockIdx / tiling.numQueryHeads;
    qheadIdx_ = blockIdx % tiling.numQueryHeads;
    kvHead_ = qheadIdx_ / tiling.gqaRatio;

    // Bind GM tensors
    // Centroids are at float offset 24 (96 bytes) in the combined buffer,
    // after the tiling data (96 bytes = 24 int32).
    queryRotGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queryRot));
    kvCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache));
    blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    seqLensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    centroidsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(centroidsTiling) + 24);  // skip 96 bytes of tiling
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));

    // Store raw base addresses for offset-based DataCopy
    queryRotBase_ = queryRot;
    kvCacheBase_ = kvCache;
    outputBase_ = output;

    seqLen_ = static_cast<uint32_t>(seqLensGm.GetValue(batchIdx_));

    // Load query as fp16, then Cast to fp32 for computation.
    // The host always sends fp16 query (via .to(torch.float16)).
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

    // DIAG: verify query for head 19 (the one that keeps getting NaN)
    if (blockIdx == 19) {
        auto q = queryBuf.Get<float>();
        AscendC::printf("TQ-DIAG-H19 query[%d][%d] cast=%f %f %f %f seq=%d\n",
                         batchIdx_, qheadIdx_,
                         q.GetValue(0), q.GetValue(1), q.GetValue(2), q.GetValue(3),
                         seqLen_);
    }

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

// ---- Score + online softmax + value dequant + accumulate ----

__aicore__ inline void KernelTqFusedDecode::ComputeScoreAndAccumulate(
    LocalTensor<uint8_t>& slotUb) {
    auto queryLocal = queryBuf.Get<float>();
    auto centroidLocal = centroidBuf.Get<float>();

    float rawScore = 0.0f;
    float normSq = 0.0f;

    if (tiling.normCorrection) {
        for (uint32_t byteIdx = 0; byteIdx < MSE_BYTES; byteIdx++) {
            uint8_t packed = slotUb.GetValue(byteIdx);
            uint32_t loIdx = packed & 0xF;
            uint32_t hiIdx = (packed >> 4) & 0xF;
            float cLo = centroidLocal.GetValue(loIdx);
            float cHi = centroidLocal.GetValue(hiIdx);
            float qLo = queryLocal.GetValue(byteIdx * 2 + 0);
            float qHi = queryLocal.GetValue(byteIdx * 2 + 1);
            rawScore += qLo * cLo + qHi * cHi;
            normSq += cLo * cLo + cHi * cHi;
        }
        rawScore *= 1.0f / ScalarSqrt(normSq + 1e-16f);
    } else {
        for (uint32_t byteIdx = 0; byteIdx < MSE_BYTES; byteIdx++) {
            uint8_t packed = slotUb.GetValue(byteIdx);
            uint32_t loIdx = packed & 0xF;
            uint32_t hiIdx = (packed >> 4) & 0xF;
            float cLo = centroidLocal.GetValue(loIdx);
            float cHi = centroidLocal.GetValue(hiIdx);
            float qLo = queryLocal.GetValue(byteIdx * 2 + 0);
            float qHi = queryLocal.GetValue(byteIdx * 2 + 1);
            rawScore += qLo * cLo + qHi * cHi;
        }
    }

    float vecNorm = ReadFp16AsFp32(slotUb, MSE_BYTES);
    float score = rawScore * vecNorm * tiling.smScale;

    // Online softmax update
    float newMax = (score > runningMax_) ? score : runningMax_;
    float alpha = ScalarExp(runningMax_ - newMax);
    float weight = ScalarExp(score - newMax);

    runningSum_ = runningSum_ * alpha + weight;

    // Scale accumulator: acc *= alpha
    auto accLocal = accBuf.Get<float>();
    Muls(accLocal, accLocal, alpha, HEAD_DIM);
    pipe_barrier(PIPE_V);

    // Dequantize value and accumulate
    DequantValueAndAccumulate(slotUb, weight);

    runningMax_ = newMax;
}

__aicore__ inline void KernelTqFusedDecode::DequantValueAndAccumulate(
    LocalTensor<uint8_t>& slotUb, float weight) {
    auto valLocal = valBuf.Get<float>();
    auto accLocal = accBuf.Get<float>();
    auto weightedLocal = weightedBuf.Get<float>();

    // Unpack 4-bit value indices into float
    for (uint32_t byteIdx = 0; byteIdx < VAL_DATA_BYTES; byteIdx++) {
        uint8_t packed = slotUb.GetValue(KEY_PACKED_SIZE + byteIdx);
        valLocal.SetValue(byteIdx * 2 + 0, static_cast<float>(packed & 0xF));
        valLocal.SetValue(byteIdx * 2 + 1, static_cast<float>((packed >> 4) & 0xF));
    }
    // SetValue goes through the Scalar pipe; the subsequent Muls/Adds below
    // run on the Vector pipe and read valLocal from UB. Without PIPE_ALL,
    // the Vector unit can observe stale UB contents (the SetValue writes
    // are still in flight on the Scalar pipe). This was the most likely
    // root cause of the constant ±1.430 fused-output pattern.
    PipeBarrier<PIPE_ALL>();

    float vScale = ReadFp16AsFp32(slotUb, KEY_PACKED_SIZE + VAL_DATA_BYTES);
    float vZero = ReadFp16AsFp32(slotUb, KEY_PACKED_SIZE + VAL_DATA_BYTES + 2);

    // Dequantize: val = val * scale + zero
    Muls(valLocal, valLocal, vScale, HEAD_DIM);
    pipe_barrier(PIPE_V);
    Adds(valLocal, valLocal, vZero, HEAD_DIM);
    pipe_barrier(PIPE_V);

    // acc += weight * val
    Muls(weightedLocal, valLocal, weight, HEAD_DIM);
    pipe_barrier(PIPE_V);
    Add(accLocal, accLocal, weightedLocal, HEAD_DIM);
    pipe_barrier(PIPE_V);
}

// ---- Process: main loop over all KV tokens ----

__aicore__ inline void KernelTqFusedDecode::Process() {
    if (GetBlockIdx() >= tiling.gridSize) return;

    // Handle empty sequence
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

        // DIAG: print score for head 19 at tokens 0, 1, 62
        if (blockIdx == 19 && (tokenPos == 0 || tokenPos == 1 || tokenPos == seqLen_ - 1)) {
            float rawSc = 0.0f;
            auto cl = centroidBuf.Get<float>();
            auto ql = queryBuf.Get<float>();
            for (uint32_t bi = 0; bi < MSE_BYTES; bi++) {
                uint8_t pk = slotLocal.GetValue(bi);
                float cL = cl.GetValue(pk & 0xF);
                float cH = cl.GetValue((pk >> 4) & 0xF);
                rawSc += ql.GetValue(bi*2+0)*cL + ql.GetValue(bi*2+1)*cH;
            }
            float vn = ReadFp16AsFp32(slotLocal, MSE_BYTES);
            AscendC::printf("TQ-DIAG-H19 tok=%d pblk=%d addr=%lu rawSc=%f vn=%f\n",
                             tokenPos, physicalBlock, slotAddr, rawSc, vn);
        }

        ComputeScoreAndAccumulate(slotLocal);
    }

    // Final normalization: output = acc / runningSum
    auto accLocal = accBuf.Get<float>();

    // DIAG: head 19 final state
    if (blockIdx == 19) {
        AscendC::printf("TQ-DIAG-H19 FINAL max=%f sum=%f acc0=%f acc1=%f\n",
                         runningMax_, runningSum_,
                         accLocal.GetValue(0), accLocal.GetValue(1));
    }

    float invSum = 1.0f / runningSum_;
    Muls(accLocal, accLocal, invSum, tiling.headDim);
    pipe_barrier(PIPE_V);

    // Cast fp32 → fp16 and write output
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
