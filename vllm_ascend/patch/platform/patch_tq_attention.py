# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch Attention.get_kv_cache_spec to return TQFullAttentionSpec for turboquant.

The base vllm's Attention layer has no turboquant branch in get_kv_cache_spec,
so without this patch turboquant layers would get a regular FullAttentionSpec
with wrong memory layout.
"""

import vllm.model_executor.layers.attention.attention as _attn_mod
from vllm.model_executor.layers.attention.attention import Attention

_original_get_kv_cache_spec = Attention.get_kv_cache_spec


def _patched_get_kv_cache_spec(self, vllm_config):
    if self.kv_cache_dtype.startswith("turboquant_"):
        from vllm_ascend.attention.tq_config import TurboQuantConfig
        from vllm_ascend.attention.tq_spec import TQFullAttentionSpec

        block_size = vllm_config.cache_config.block_size
        tq_config = TurboQuantConfig.from_cache_dtype(
            self.kv_cache_dtype, self.head_size
        )
        return TQFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            tq_slot_size=tq_config.slot_size_aligned,
        )
    return _original_get_kv_cache_spec(self, vllm_config)


Attention.get_kv_cache_spec = _patched_get_kv_cache_spec
