#ifndef TQ_FUSED_DECODE_TILING_H
#define TQ_FUSED_DECODE_TILING_H

#include <cstdint>

struct TqFusedDecodeTilingData {
    // Work distribution
    uint32_t batchSize;
    uint32_t numQueryHeads;
    uint32_t numKvHeads;
    uint32_t gqaRatio;
    uint32_t gridSize;

    // Cache geometry
    uint32_t blockSize;
    uint32_t headDim;
    uint32_t mseBytes;
    uint32_t keyPackedSize;
    uint32_t valDataBytes;
    uint32_t slotSize;
    uint32_t maxBlocksPerSeq;

    // Derived strides (bytes)
    uint64_t blockStride;   // block_size * Hk * slot_size
    uint64_t slotStride;    // Hk * slot_size

    // Quantization params
    uint32_t normCorrection;
    float     smScale;
};

#endif
