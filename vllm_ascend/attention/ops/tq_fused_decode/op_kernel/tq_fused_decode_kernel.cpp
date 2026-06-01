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
// Slot data padded to 32-byte alignment for DataCopyPad
constexpr uint32_t SLOT_BUF_ALIGNED = ((SLOT_SIZE + 31) / 32) * 32; // 160
// Tiling struct padded to 32-byte alignment
constexpr uint32_t TILING_BUF_ALIGNED = ((sizeof(TqFusedDecodeTilingData) + 31) / 32) * 32;

class KernelTqFusedDecode {
public:
    __aicore__ inline void Init(GM_ADDR queryRot, GM_ADDR kvCache,
                                 GM_ADDR blockTable, GM_ADDR seqLens,
                                 GM_ADDR centroids, GM_ADDR output,
                                 GM_ADDR tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float ScalarExp(float x);
    __aicore__ inline float ScalarSqrt(float x);
    __aicore__ inline float ReadFp16AsFp32(LocalTensor<uint8_t>& slotUb, uint32_t offset);
    __aicore__ inline void ComputeScoreAndAccumulate(LocalTensor<uint8_t>& slotUb);
    __aicore__ inline void DequantValueAndAccumulate(LocalTensor<uint8_t>& slotUb,
                                                      float weight);
    __aicore__ inline void ReadTiling(GM_ADDR tilingData);

    TPipe pipe;

    // GM tensors
    GlobalTensor<half> queryRotGm;
    GlobalTensor<uint8_t> kvCacheGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> seqLensGm;
    GlobalTensor<float> centroidsGm;
    GlobalTensor<half> outputGm;

    // Raw GM base addresses for offset-based DataCopyPad
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
    // Use a small LocalTensor to compute exp of a single scalar
    LocalTensor<float> tmp;
    // Allocate from valBuf temporarily (reuse valBuf since it's not
    // needed during score computation)
    tmp = valBuf.Get<float>();
    tmp.SetValue(0, x);
    pipe_barrier(PIPE_V);
    Exp(tmp, tmp, 1);
    pipe_barrier(PIPE_V);
    float result = tmp.GetValue(0);
    return result;
}

__aicore__ inline float KernelTqFusedDecode::ScalarSqrt(float x) {
    LocalTensor<float> tmp = valBuf.Get<float>();
    tmp.SetValue(0, x);
    pipe_barrier(PIPE_V);
    Sqrt(tmp, tmp, 1);
    pipe_barrier(PIPE_V);
    float result = tmp.GetValue(0);
    return result;
}

// ---- FP16 byte reinterpret ----

__aicore__ inline float KernelTqFusedDecode::ReadFp16AsFp32(
    LocalTensor<uint8_t>& slotUb, uint32_t offset) {
    // Read 2 bytes, reinterpret as fp16, cast to fp32
    uint16_t raw = (static_cast<uint16_t>(slotUb.GetValue(offset + 1)) << 8) |
                   static_cast<uint16_t>(slotUb.GetValue(offset));
    // Bit-cast uint16 → half → float using union (safe in Ascend C)
    union { uint16_t u; half h; } cvt;
    cvt.u = raw;
    return static_cast<float>(cvt.h);
}

// ---- Tiling data read via DataCopyPad + GetValue ----

__aicore__ inline void KernelTqFusedDecode::ReadTiling(GM_ADDR tilingData) {
    GlobalTensor<uint8_t> tilingGm;
    tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(tilingData));
    auto slotLocal = slotQueue.AllocTensor<uint8_t>();
    DataCopyPad(slotLocal, tilingGm,
                {1, static_cast<uint32_t>(sizeof(TqFusedDecodeTilingData)), 0, 0},
                {false, 0, 0, 0});
    pipe_barrier(PIPE_V);

    // Reinterpret slot data as int32 for structured reads
    auto s = slotLocal.ReinterpretCast<int32_t>();

    tiling.batchSize       = s.GetValue(0);
    tiling.numQueryHeads   = s.GetValue(1);
    tiling.numKvHeads      = s.GetValue(2);
    tiling.gqaRatio        = s.GetValue(3);
    tiling.gridSize        = s.GetValue(4);
    tiling.blockSize       = s.GetValue(5);
    tiling.headDim         = s.GetValue(6);
    tiling.mseBytes        = s.GetValue(7);
    tiling.keyPackedSize   = s.GetValue(8);
    tiling.valDataBytes    = s.GetValue(9);
    tiling.slotSize        = s.GetValue(10);
    tiling.maxBlocksPerSeq = s.GetValue(11);
    // uint64_t fields: blockStride at offset 12-13, slotStride at 14-15
    // (each uint64 occupies 2 × uint32 slots)
    uint32_t lo12 = s.GetValue(12);
    uint32_t hi13 = s.GetValue(13);
    tiling.blockStride = (static_cast<uint64_t>(hi13) << 32) | lo12;
    uint32_t lo14 = s.GetValue(14);
    uint32_t hi15 = s.GetValue(15);
    tiling.slotStride = (static_cast<uint64_t>(hi15) << 32) | lo14;
    // uint32_t normCorrection at offset 16
    tiling.normCorrection  = s.GetValue(16);
    // float smScale at offset 17 (reinterpret uint32 as float)
    uint32_t rawScale = s.GetValue(17);
    union { uint32_t u; float f; } cvtScale;
    cvtScale.u = rawScale;
    tiling.smScale = cvtScale.f;

    slotQueue.FreeTensor(slotLocal);
}

// ---- Init ----

__aicore__ inline void KernelTqFusedDecode::Init(
    GM_ADDR queryRot, GM_ADDR kvCache, GM_ADDR blockTable,
    GM_ADDR seqLens, GM_ADDR centroids, GM_ADDR output,
    GM_ADDR tilingData) {

    uint32_t blockIdx = GetBlockIdx();

    // Init UB buffers
    pipe.InitBuffer(slotQueue, 1, SLOT_BUF_ALIGNED);
    pipe.InitBuffer(queryBuf, QUERY_BUF_SIZE);
    pipe.InitBuffer(centroidBuf, CENTROID_BUF_SIZE);
    pipe.InitBuffer(accBuf, ACC_BUF_SIZE);
    pipe.InitBuffer(valBuf, VAL_BUF_SIZE);
    pipe.InitBuffer(weightedBuf, WEIGHTED_BUF_SIZE);

    // Read tiling data
    ReadTiling(tilingData);

    // Early exit if this core has no work
    if (blockIdx >= tiling.gridSize) return;

    batchIdx_ = blockIdx / tiling.numQueryHeads;
    qheadIdx_ = blockIdx % tiling.numQueryHeads;
    kvHead_ = qheadIdx_ / tiling.gqaRatio;

    // Bind GM tensors
    queryRotGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queryRot));
    kvCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache));
    blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    seqLensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    centroidsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));

    // Store raw base addresses for offset-based DataCopyPad
    queryRotBase_ = queryRot;
    kvCacheBase_ = kvCache;
    outputBase_ = output;

    seqLen_ = static_cast<uint32_t>(seqLensGm.GetValue(batchIdx_));

    // Load query vector (HEAD_DIM fp16 → fp32 in UB)
    auto queryFp32 = queryBuf.Get<float>();
    uint64_t qOffset = static_cast<uint64_t>(batchIdx_) * tiling.numQueryHeads * tiling.headDim
                     + static_cast<uint64_t>(qheadIdx_) * tiling.headDim;
    GlobalTensor<half> queryOffsetGm;
    queryOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queryRotBase_) + qOffset);
    auto slotForQ = slotQueue.AllocTensor<uint8_t>();
    auto queryHalfLocal = slotForQ.ReinterpretCast<half>();
    DataCopyPad(queryHalfLocal, queryOffsetGm,
                {1, static_cast<uint32_t>(HEAD_DIM * sizeof(half)), 0, 0},
                {false, 0, 0, 0});
    pipe_barrier(PIPE_V);
    Cast(queryFp32, queryHalfLocal, RoundMode::CAST_NONE, HEAD_DIM);
    pipe_barrier(PIPE_V);
    slotQueue.FreeTensor(slotForQ);

    // Load centroid table (16 fp32)
    auto centroidLocal = centroidBuf.Get<float>();
    DataCopyPad(centroidLocal, centroidsGm,
                {1, static_cast<uint32_t>(CENTROID_TABLE_SIZE * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    pipe_barrier(PIPE_V);

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
        // Compute raw_score and norm_sq in one pass
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
        // Compute raw_score only
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

    // Apply vec_norm and sm_scale
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

    // Read scale and zero (fp16 at fixed offsets)
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
        Cast(outFp16, accLocal, RoundMode::CAST_NONE, tiling.headDim);
        pipe_barrier(PIPE_V);
        uint64_t outOff = static_cast<uint64_t>(batchIdx_) * tiling.numQueryHeads * tiling.headDim
                        + static_cast<uint64_t>(qheadIdx_) * tiling.headDim;
        GlobalTensor<half> outOffsetGm;
        outOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outputBase_) + outOff);
        DataCopyPad(outOffsetGm, outFp16,
                    {1, static_cast<uint16_t>(tiling.headDim * sizeof(half)), 0, 0});
        pipe_barrier(PIPE_V);
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

        // Read slot data
        GlobalTensor<uint8_t> kvSlotGm;
        kvSlotGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCacheBase_) + slotAddr);
        DataCopyPad(slotLocal, kvSlotGm,
                    {1, SLOT_SIZE, 0, 0},
                    {false, 0, 0, 0});
        pipe_barrier(PIPE_V);

        ComputeScoreAndAccumulate(slotLocal);
        // No FreeTensor/AllocTensor — reuse same buffer
    }

    // Final normalization: output = acc / runningSum
    auto accLocal = accBuf.Get<float>();
    float invSum = 1.0f / runningSum_;
    Muls(accLocal, accLocal, invSum, tiling.headDim);
    pipe_barrier(PIPE_V);

    // Cast fp32 → fp16 and write output
    auto outFp16 = slotLocal.ReinterpretCast<half>();
    Cast(outFp16, accLocal, RoundMode::CAST_NONE, tiling.headDim);
    pipe_barrier(PIPE_V);

    uint64_t outOff = static_cast<uint64_t>(batchIdx_) * tiling.numQueryHeads * tiling.headDim
                    + static_cast<uint64_t>(qheadIdx_) * tiling.headDim;
    GlobalTensor<half> outOffsetGm;
    outOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outputBase_) + outOff);
    DataCopyPad(outOffsetGm, outFp16,
                {1, static_cast<uint16_t>(tiling.headDim * sizeof(half)), 0, 0});
    pipe_barrier(PIPE_V);
    slotQueue.FreeTensor(slotLocal);
}

// ---- Kernel entry point ----

extern "C" __global__ __aicore__ void tq_fused_decode_kernel(
    GM_ADDR queryRot, GM_ADDR kvCache, GM_ADDR blockTable,
    GM_ADDR seqLens, GM_ADDR centroids, GM_ADDR output,
    GM_ADDR tiling) {
    KernelTqFusedDecode op;
    op.Init(queryRot, kvCache, blockTable, seqLens, centroids, output, tiling);
    op.Process();
}
