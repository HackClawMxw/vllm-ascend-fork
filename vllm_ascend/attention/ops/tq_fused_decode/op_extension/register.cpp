#include <torch/torch.h>
#include "ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def(
        "tq_fused_decode(Tensor query_rot, "
        "Tensor kv_cache, "
        "Tensor block_table, "
        "Tensor seq_lens, "
        "Tensor centroids, "
        "float sm_scale, "
        "int mse_bytes, "
        "int key_packed_size, "
        "int val_data_bytes, "
        "int head_dim, "
        "int block_size, "
        "bool norm_correction) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("tq_fused_decode", TORCH_FN(ascend_kernel::tq_fused_decode_torch));
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("tq_fused_decode", [](const torch::Tensor& query_rot,
                                  const torch::Tensor& /*kv_cache*/,
                                  const torch::Tensor& /*block_table*/,
                                  const torch::Tensor& /*seq_lens*/,
                                  const torch::Tensor& /*centroids*/,
                                  double /*sm_scale*/,
                                  int64_t /*mse_bytes*/,
                                  int64_t /*key_packed_size*/,
                                  int64_t /*val_data_bytes*/,
                                  int64_t /*head_dim*/,
                                  int64_t /*block_size*/,
                                  bool /*norm_correction*/) {
        return at::empty(query_rot.sizes(), query_rot.options().dtype(at::kHalf));
    });
}
