#ifndef TQ_FUSED_DECODE_OPS_H
#define TQ_FUSED_DECODE_OPS_H

#include <torch/torch.h>

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
    bool norm_correction);

}

#endif
