# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch CacheConfig and dtype mappings to accept TurboQuant cache dtypes.

Two changes:
1. Widen CacheConfig.cache_dtype annotation from CacheDType Literal to str
   so Pydantic validation accepts turboquant_* values.
2. Register turboquant dtype strings in vllm's STR_DTYPE_TO_TORCH_DTYPE
   so kv_cache_dtype_str_to_dtype() can resolve them to torch.uint8.
"""

import torch
import vllm.config.cache as _cache_mod
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

# --- 1. Widen CacheConfig.cache_dtype to accept turboquant strings ---
_original_field = CacheConfig.model_fields["cache_dtype"]
_original_field.annotation = str
CacheConfig.model_rebuild(force=True)

# --- 2. Register turboquant dtype → torch.uint8 mappings ---
_TQ_DTYPES = {
    "turboquant_k8v4": torch.uint8,
    "turboquant_4bit_nc": torch.uint8,
    "turboquant_k3v4_nc": torch.uint8,
    "turboquant_3bit_nc": torch.uint8,
}
for _name, _dtype in _TQ_DTYPES.items():
    STR_DTYPE_TO_TORCH_DTYPE.setdefault(_name, _dtype)
